use std::{collections::HashMap, path::Path};

use anyhow::{Context, ensure};
use serde::Deserialize;

use crate::{Shape, Tensor};

/// Generic helper function to unfuse a tensor's data into multiple chunks.
/// Expects the input tensor `fused_tensor` (crate::Tensor<f32>) to contain flat data.
pub fn unfuse_crate_tensors(
    fused_tensor: Tensor<f32>,
    expected_chunk_len_elements: usize,
    num_chunks: usize,
) -> anyhow::Result<Vec<Vec<f32>>> {
    let data = fused_tensor.get_data();
    let total_elements = data.len();

    ensure!(
        expected_chunk_len_elements > 0,
        "expected_chunk_len_elements must be positive, got {expected_chunk_len_elements}"
    );
    ensure!(
        num_chunks > 0,
        "num_chunks must be positive, got {num_chunks}"
    );

    let expected_total_elements = expected_chunk_len_elements * num_chunks;
    ensure!(
        total_elements == expected_total_elements,
        "Tensor data size ({}) does not match expected total size ({} chunks * {} elements_per_chunk = {}). Original tensor shape: {:?}",
        total_elements,
        num_chunks,
        expected_chunk_len_elements,
        expected_total_elements,
        fused_tensor.shape()
    );

    let tensors_data: Vec<Vec<f32>> = data
        .chunks_exact(expected_chunk_len_elements)
        .map(|chunk| chunk.to_vec())
        .collect();

    ensure!(
        tensors_data.len() == num_chunks,
        "Unfused into {} tensors, expected {}",
        tensors_data.len(),
        num_chunks
    );

    Ok(tensors_data)
}

#[derive(Debug, Clone, Deserialize)]
pub struct JsonTensor {
    pub shape: Shape,
    pub data: Vec<f32>,
}

impl JsonTensor {
    pub fn as_tensor(&self) -> Tensor<f32> {
        Tensor::new(self.shape.clone(), self.data.clone())
    }
}

#[derive(Clone, Debug, Deserialize)]
pub struct JsonModel {
    pub metadata: HashMap<String, serde_json::Value>,
    pub tensors: HashMap<String, JsonTensor>,
}

#[derive(Clone, Debug)]
pub struct FileTensorLoader {
    pub content: JsonModel,
    pub prefix: String, // current path scope (e.g., "blk.00.")
}

impl FileTensorLoader {
    pub fn new_from_path<P: AsRef<Path>>(path: P) -> anyhow::Result<Self> {
        let file = std::fs::File::open(path.as_ref()).with_context(|| {
            format!("Failed to open JSON file at: {:?}", path.as_ref().display())
        })?;
        let content: JsonModel = serde_json::from_reader(file).with_context(|| {
            format!(
                "Failed to parse JSON from file at: {:?}",
                path.as_ref().display()
            )
        })?;
        Ok(Self {
            content,
            prefix: "".to_string(),
        })
    }

    pub fn pp(&self, sub: &str) -> Self {
        let mut new = self.clone();
        new.prefix = format!("{}{}", self.prefix, sub);
        new
    }

    fn resolve_key(&self, key: &str) -> Option<&JsonTensor> {
        let full_key = format!("{}{}", self.prefix, key);
        self.content.tensors.get(&full_key)
    }

    pub fn get_tensor(&self, key: &str) -> anyhow::Result<Tensor<f32>> {
        let tensor = self
            .resolve_key(key)
            .ok_or_else(|| anyhow::anyhow!("tensor not found: {key}"))?;
        Ok(Tensor::new(tensor.shape.clone(), tensor.data.clone()))
    }

    pub fn get_metadata(&self, key: &str) -> Option<&serde_json::Value> {
        self.content.metadata.get(key)
    }

    pub fn metadata_to_u32(&self, key: &str) -> anyhow::Result<u32> {
        Ok(self
            .get_metadata(key)
            .ok_or_else(|| anyhow::anyhow!("missing metadata {key}"))?
            .as_u64()
            .ok_or_else(|| anyhow::anyhow!("metadata {key} not a u32"))? as u32)
    }

    pub fn metadata_to_f32(&self, key: &str) -> anyhow::Result<f32> {
        Ok(self
            .get_metadata(key)
            .ok_or_else(|| anyhow::anyhow!("missing metadata {key}"))?
            .as_f64()
            .ok_or_else(|| anyhow::anyhow!("metadata {key} not a f32"))? as f32)
    }
}

#[cfg(test)]
pub mod test {
    use crate::parser::llm::LLMConfig;
    use std::path::PathBuf;

    use super::*;

    pub const TINY_GPT2_NAME: &str = "tiny_gpt2_weights.json";
    pub const TINY_GPT2_DEBUG_NAME: &str = "tiny_gpt2_debug_output.json";
    #[allow(dead_code)]
    pub const DISTIL_GPT2_NAME: &str = "distilgpt2_weights.json";
    #[allow(dead_code)]
    pub const DISTIL_GPT2_DEBUG_NAME: &str = "distilgpt2_debug_output.json";

    pub fn get_json_file(name: &str) -> anyhow::Result<String> {
        let path = PathBuf::from("assets/scripts/llms/").join(name);
        assert!(
            path.exists(),
            "Missing model `{}` run `python3 gpt2_internal.py --output-dir ./assets/scripts/llms/ --export-model`",
            path.display()
        );
        Ok(path.to_str().unwrap().to_string())
    }

    #[test]
    fn test_json_tensor_loader() -> anyhow::Result<()> {
        let path = get_json_file(TINY_GPT2_NAME)?;
        let loader = FileTensorLoader::new_from_path(path)?;
        println!("loader keys: {:?}", loader.content.metadata.keys());
        let config = LLMConfig::from_json(&loader)?;
        println!("tiny gpt2 config: {config:?}");
        config.model_json(&loader)?;
        Ok(())
    }
}
