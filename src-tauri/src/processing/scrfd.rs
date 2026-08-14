use crate::inference::types::{BoundingBox, FaceLandmarks, DetectedFace};

#[derive(Debug, Clone)]
pub struct ScrfdAnchor {
    pub x: f32,
    pub y: f32,
    pub stride: f32,    
}

#[derive(Debug, Clone)]
pub struct ScrfdConfig {
    pub strides: Vec<usize>,
    pub num_anchors: usize,
    pub score_threshold: f32,
}

impl Default for ScrfdConfig {
    fn default() -> Self {
        Self {
            strides: vec![8, 16, 32],
            num_anchors: 2,
            score_threshold: 0.5,
        }
    }
}

pub struct ScrfdDecoder {
    config: ScrfdConfig,
}

impl ScrfdDecoder {
    pub fn new(config: ScrfdConfig) -> Self {
        Self { config }
    }

    pub fn generate_anchors(&self, input_width: usize, input_height: usize) -> Vec<ScrfdAnchor> {
        let mut anchors = Vec::new();

        for &stride in &self.config.strides {
            let feat_w = (input_width + stride - 1) / stride;
            let feat_h = (input_height + stride - 1) / stride;

            for y in 0..feat_h {
                for x in 0..feat_w {
                    let anchor_x = (x * stride) as f32;
                    let anchor_y = (y * stride) as f32;

                    for _ in 0..self.config.num_anchors {
                        anchors.push(ScrfdAnchor {
                            x: anchor_x,
                            y: anchor_y,
                            stride: stride as f32,
                        });
                    }
                }
            }
        }

        anchors
    }

    pub fn decode(
        &self, 
        anchors: &[ScrfdAnchor],
        raw_scores: &[f32],
        raw_bboxes: &[f32],
        raw_kps: Option<&[f32]>,
    ) -> Vec<DetectedFace> {
        let mut detections = Vec::new();

        for (i, anchor) in anchors.iter().enumerate() {
            let score = raw_scores[i];
            if score < self.config.score_threshold {
                continue;
            }

            let bbox_offset = i * 4;
            let dx1 = raw_bboxes[bbox_offset] * anchor.stride;
            let dy1 = raw_bboxes[bbox_offset + 1] * anchor.stride;
            let dx2 = raw_bboxes[bbox_offset + 2] * anchor.stride;
            let dy2 = raw_bboxes[bbox_offset + 3] * anchor.stride;

            let x1 = anchor.x - dx1;
            let y1 = anchor.y - dy1;
            let x2 = anchor.x + dx2;
            let y2 = anchor.y + dy2;

            let bbox = BoundingBox { x1, y1, x2, y2, score };

            let landmarks = raw_kps.map(|kps_slice| {
                let kps_offset = i * 10;
                let mut points = [(0.0f32, 0.0f32); 5];

                for kp_idx in 0..5 {
                    let px = anchor.x + kps_slice[kps_offset + kp_idx * 2] * anchor.stride;
                    let py = anchor.y + kps_slice[kps_offset + kp_idx * 2 + 1] * anchor.stride;
                    points[kp_idx] = (px, py);
                }

                FaceLandmarks { points }
            });

            detections.push(DetectedFace { bbox, landmarks });
        }

        detections
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_anchor_generation_count() {
        let config = ScrfdConfig {
            strides: vec![8, 16, 32],
            num_anchors: 2,
            score_threshold: 0.5,
        };
        let decoder = ScrfdDecoder::new(config);

        // stride 8:  16 x 16 x 2 = 512
        // stride 16:  8 x 8 x 2  = 128
        // stride 32:  4 x 4 x 2  = 32
        // Total: 512 + 128 + 32 = 672
        let anchors = decoder.generate_anchors(128, 128);
        assert_eq!(anchors.len(), 672);
    }

    #[test]
    fn test_bbox_and_kps_decoding() {
        let config = ScrfdConfig {
            strides: vec![8],
            num_anchors: 1,
            score_threshold: 0.5,
        };
        let decoder = ScrfdDecoder::new(config);

        let anchors = vec![ScrfdAnchor { x: 40.0, y: 40.0, stride: 8.0 }];
        let scores = vec![0.95];
        
        let bboxes = vec![2.0, 2.0, 5.0, 5.0];

        // (1, 1), (3, 1), (2, 2), (1, 3), (3, 3)
        let kps = vec![
            1.0, 1.0,  // Eye L: (40+8, 40+8) = (48, 48)
            3.0, 1.0,  // Eye R: (40+24, 40+8) = (64, 48)
            2.0, 2.0,  // Nose:  (40+16, 40+16) = (56, 56)
            1.0, 3.0,  // Mouth L: (48, 64)
            3.0, 3.0,  // Mouth R: (64, 64)
        ];

        let faces = decoder.decode(&anchors, &scores, &bboxes, Some(&kps));

        assert_eq!(faces.len(), 1);
        let face = &faces[0];
        
        assert_eq!(face.bbox.x1, 24.0);
        assert_eq!(face.bbox.y1, 24.0);
        assert_eq!(face.bbox.x2, 80.0);
        assert_eq!(face.bbox.y2, 80.0);

        let landmarks = face.landmarks.as_ref().unwrap();
        assert_eq!(landmarks.points[0], (48.0, 48.0)); // Left eye
        assert_eq!(landmarks.points[2], (56.0, 56.0)); // Nose
    }
}