#[derive(Debug, Clone)]
pub struct PipelineConfig {
    pub scrfd_model_path: String,
    pub inswapper_model_path: String,
    pub smoothing_alpha: f32,
    pub mask_margin_ratio: f32,
    pub det_threshold: f32,
    pub nms_threshold: f32,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            scrfd_model_path: "models/scrfd_10g_bnkps.onnx".into(),
            inswapper_model_path: "models/inswapper_128.onnx".into(),
            smoothing_alpha: 0.4,
            mask_margin_ratio: 0.2,
            det_threshold: 0.5,
            nms_threshold: 0.4,
        }
    }
}