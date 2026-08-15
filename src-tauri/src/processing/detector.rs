use anyhow::{anyhow, Result};
use ndarray::Array4;

use crate::inference::types::DetectedFace;
use crate::inference::InferenceEngine;
use crate::processing::nms::non_max_suppression;
use crate::processing::scrfd::{ScrfdAnchor, ScrfdConfig, ScrfdDecoder};

pub struct ScrfdDetector {
    engine: InferenceEngine,
    decoder: ScrfdDecoder,
    anchors: Vec<ScrfdAnchor>,
    iou_threshold: f32,
    input_size: (usize, usize),
}

impl ScrfdDetector {
    pub fn new(
        model_path: &str,
        input_width: usize,
        input_height: usize,
        iou_threshold: f32,
        score_threshold: f32,
    ) -> Result<Self> {
        let engine = InferenceEngine::new(model_path)?;

        let config = ScrfdConfig {
            score_threshold,
            ..Default::default()
        };

        let decoder = ScrfdDecoder::new(config);
        let anchors = decoder.generate_anchors(input_width, input_height);

        Ok(Self {
            engine,
            decoder,
            anchors,
            iou_threshold,
            input_size: (input_width, input_height),
        })
    }

    pub fn detect(&mut self, input_tensor: Array4<f32>) -> Result<Vec<DetectedFace>> {
    let raw_outputs = self.engine.run_inference(input_tensor)?;

        if raw_outputs.len() < 2 {
            return Err(anyhow!("SCRFD output count mismatch: expected at least 2 outputs (scores, bboxes)"));
        }

        let raw_scores = &raw_outputs[0];
        let raw_bboxes = &raw_outputs[1];
        let raw_kps = raw_outputs.get(2).map(|v| v.as_slice());

        let raw_detections = self.decoder.decode(
            &self.anchors,
            raw_scores,
            raw_bboxes,
            raw_kps,
        );

        let boxes_to_filter: Vec<_> = raw_detections.iter().map(|d| d.bbox.clone()).collect();
        let kept_boxes = non_max_suppression(boxes_to_filter, self.iou_threshold);

        let filtered_detections = raw_detections
            .into_iter()
            .filter(|d| kept_boxes.iter().any(|kb| (kb.score - d.bbox.score).abs() < f32::EPSILON))
            .collect();

        Ok(filtered_detections)
    }

    pub fn input_size(&self) -> (usize, usize) {
        self.input_size
    }
}