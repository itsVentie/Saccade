use anyhow::{anyhow, Result};
use ndarray::{Array2, Array4};
use std::sync::Mutex;

use crate::inference::InferenceEngine;

pub struct InswapperPipeline {
    engine: Mutex<InferenceEngine>,
    input_shape: (usize, usize), // (128, 128)
}

impl InswapperPipeline {
    pub fn new(model_path: &str) -> Result<Self> {
        let engine = InferenceEngine::new(model_path)?;
        Ok(Self {
            engine: Mutex::new(engine),
            input_shape: (128, 128),
        })
    }

    pub fn validate_embedding(source_embedding: &[f32]) -> Result<()> {
        if source_embedding.len() != 512 {
            return Err(anyhow!(
                "Invalid source embedding length: expected 512, got {}",
                source_embedding.len()
            ));
        }
        Ok(())
    }

    pub fn swap_face(
        &self,
        target_crop: Array4<f32>,
        source_embedding: &[f32],
    ) -> Result<Array4<f32>> {
        Self::validate_embedding(source_embedding)?;

        let _embedding_tensor = Array2::from_shape_vec(
            (1, 512),
            source_embedding.to_vec(),
        ).map_err(|e| anyhow!("Failed to reshape source embedding tensor: {e}"))?;

        let mut engine = self
            .engine
            .lock()
            .map_err(|e| anyhow!("Failed to lock InferenceEngine Mutex: {e}"))?;

        let raw_outputs = engine.run_inference(target_crop)?;

        if raw_outputs.is_empty() {
            return Err(anyhow!("Inswapper model returned no output tensors"));
        }

        let output_data = &raw_outputs[0];
        
        let swapped_tensor = Array4::from_shape_vec(
            (1, 3, self.input_shape.0, self.input_shape.1),
            output_data.clone(),
        ).map_err(|e| anyhow!("Failed to reconstruct output swapped image tensor: {e}"))?;

        Ok(swapped_tensor)
    }

    pub fn input_shape(&self) -> (usize, usize) {
        self.input_shape
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_invalid_source_embedding_length() {
        let invalid_emb = vec![0.0f32; 256];
        let result = InswapperPipeline::validate_embedding(&invalid_emb);
        
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Invalid source embedding length"));
    }

    #[test]
    fn test_valid_source_embedding_length() {
        let valid_emb = vec![0.0f32; 512];
        let result = InswapperPipeline::validate_embedding(&valid_emb);
        
        assert!(result.is_ok());
    }
}