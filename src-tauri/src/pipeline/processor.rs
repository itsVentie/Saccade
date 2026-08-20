use super::config::PipelineConfig;
use crate::processing::{
    blending::{generate_feathered_mask, warp_affine_back, LandmarkSmoother},
    detector::ScrfdDetector,
    inswapper::InswapperPipeline,
    transform::{estimate_similarity_transform, warp_affine_128},
};
use ndarray::{Array2, Array4};

pub struct FrameProcessor {
    detector: ScrfdDetector,
    swapper: InswapperPipeline,
    smoother: LandmarkSmoother,
    mask_128: Array2<f32>,
}

impl FrameProcessor {
    pub fn new(config: &PipelineConfig) -> Result<Self, String> {
        let detector = ScrfdDetector::new(
            &config.scrfd_model_path,
            640,
            640,
            config.nms_threshold,
            config.det_threshold,
        )
        .map_err(|e| e.to_string())?;

        let swapper =
            InswapperPipeline::new(&config.inswapper_model_path).map_err(|e| e.to_string())?;

        let smoother = LandmarkSmoother::new(config.smoothing_alpha);
        let mask_128 = generate_feathered_mask(128, config.mask_margin_ratio);

        Ok(Self {
            detector,
            swapper,
            smoother,
            mask_128,
        })
    }

    pub fn process(
        &mut self,
        frame: &mut Array4<f32>,
        source_embedding: &[f32],
    ) -> Result<usize, String> {
        let detections = self
            .detector
            .detect(frame.clone())
            .map_err(|e| e.to_string())?;

        if detections.is_empty() {
            self.smoother.reset();
            return Ok(0);
        }

        let primary_face = &detections[0];

        let kps: [[f32; 2]; 5] = match &primary_face.landmarks {
            Some(lm) => [
                [lm.points[0].0, lm.points[0].1],
                [lm.points[1].0, lm.points[1].1],
                [lm.points[2].0, lm.points[2].1],
                [lm.points[3].0, lm.points[3].1],
                [lm.points[4].0, lm.points[4].1],
            ],
            None => {
                self.smoother.reset();
                return Ok(0);
            }
        };

        let smoothed_kps = self.smoother.update(&kps);

        let affine_mat = match estimate_similarity_transform(&smoothed_kps) {
            Some(mat) => mat,
            None => return Ok(0),
        };

        let aligned_face_128 = match warp_affine_128(frame, &affine_mat) {
            Some(face) => face,
            None => return Ok(0),
        };

        let swapped_face_128 = self
            .swapper
            .swap_face(aligned_face_128, source_embedding)
            .map_err(|e| e.to_string())?;

        warp_affine_back(frame, &swapped_face_128, &affine_mat, &self.mask_128);

        Ok(detections.len())
    }

    pub fn reset_smoother(&mut self) {
        self.smoother.reset();
    }
}
