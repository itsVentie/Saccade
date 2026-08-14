use anyhow::{anyhow, Result};
use ndarray::Array4;
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::Tensor;

pub mod types;

pub struct InferenceEngine {
    session: Session,
}

impl InferenceEngine {
    pub fn new(model_path: &str) -> Result<Self> {
        let builder = Session::builder()
            .map_err(|e| anyhow!("Failed to initialize SessionBuilder: {e}"))?;

        let session = builder
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| anyhow!("Failed to set optimization level: {e}"))?
            .with_intra_threads(4)
            .map_err(|e| anyhow!("Failed to set intra threads: {e}"))?
            .commit_from_file(model_path)
            .map_err(|e| anyhow!("Failed to load model from file '{model_path}': {e}"))?;

        Ok(Self { session })
    }

    pub fn run_inference(&mut self, input_tensor: Array4<f32>) -> Result<Vec<f32>> {
    let shape = input_tensor.shape().to_vec();
    let data = input_tensor.into_raw_vec();

    let ort_input = Tensor::from_array((shape, data))
        .map_err(|e| anyhow!("Failed to create input tensor: {e}"))?;

    let outputs = self
        .session
        .run(ort::inputs!["input" => ort_input])
        .map_err(|e| anyhow!("Inference execution failed: {e}"))?;

    let (_shape, slice) = outputs["output"]
        .try_extract_tensor::<f32>()
        .map_err(|e| anyhow!("Failed to extract output tensor: {e}"))?;

    Ok(slice.to_vec())
}
}