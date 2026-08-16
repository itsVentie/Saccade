#[derive(Debug, Default)]
pub struct PipelineState {
    pub source_embedding: Option<Vec<f32>>,
    pub is_running: bool,
}

impl PipelineState {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn set_source_embedding(&mut self, embedding: Vec<f32>) -> Result<(), String> {
        if embedding.len() != 512 {
            return Err(format!(
                "Invalid embedding dimension: expected 512, got {}",
                embedding.len()
            ));
        }
        self.source_embedding = Some(embedding);
        Ok(())
    }

    pub fn clear_source_embedding(&mut self) {
        self.source_embedding = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_valid_embedding_set() {
        let mut state = PipelineState::new();
        let valid_emb = vec![0.0f32; 512];
        assert!(state.set_source_embedding(valid_emb).is_ok());
        assert!(state.source_embedding.is_some());
    }

    #[test]
    fn test_invalid_embedding_set() {
        let mut state = PipelineState::new();
        let invalid_emb = vec![0.0f32; 128];
        assert!(state.set_source_embedding(invalid_emb).is_err());
        assert!(state.source_embedding.is_none());
    }
}