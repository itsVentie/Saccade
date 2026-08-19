use ndarray::{Array2, Array4};
use super::config::PipelineConfig;
use crate::processing::{
    blending::{generate_feathered_mask, warp_affine_back, LandmarkSmoother},
    detector::FaceDetector,
    inswapper::InSwapper,
    transform::{estimate_similarity_transform, warp_affine_128},
};

pub struct FrameProcessor {
    detector: FaceDetector,
    swapper: InSwapper,
    smoother: LandmarkSmoother,
    mask_128: Array2<f32>,
}

impl FrameProcessor {
    pub fn new(config: &PipelineConfig) -> Result<Self, String> {
        let detector = FaceDetector::new(&config.scrfd_model_path)?;
        let swapper = InSwapper::new(&config.inswapper_model_path)?;
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
        let detections = self.detector.detect(frame)?;
        if detections.is_empty() {
            self.smoother.reset();
            return Ok(0);
        }

        let primary_face = &detections[0];

        let smoothed_kps = self.smoother.update(&primary_face.kps);

        let affine_mat = match estimate_similarity_transform(&smoothed_kps) {
            Some(mat) => mat,
            None => return Ok(0),
        };

        let aligned_face_128 = match warp_affine_128(frame, &affine_mat) {
            Some(face) => face,
            None => return Ok(0),
        };

        let swapped_face_128 = self.swapper.swap(&aligned_face_128, source_embedding)?;

        warp_affine_back(frame, &swapped_face_128, &affine_mat, &self.mask_128);

        Ok(detections.len())
    }

    pub fn reset_smoother(&mut self) {
        self.smoother.reset();
    }
}