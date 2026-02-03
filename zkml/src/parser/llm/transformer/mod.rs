use crate::{
    Number,
    layers::{
        Layer,
        transformer::{layernorm::LayerNorm, rmsnorm::RMSNorm},
    },
    parser::{gguf, llm::LLMConfig, safe},
    tensor::TensorTypeParam,
};

use serde::{Deserialize, Serialize};

pub mod attention_layer;
pub mod feed_forward;

#[derive(Copy, Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum NormType {
    LayerNorm,
    RMSNorm,
}
#[derive(Debug, Clone)]
pub enum Norm<N: Number> {
    LayerNorm(LayerNorm<N>),
    RMSNorm(RMSNorm<N>),
}

impl<N> Norm<N>
where
    N: TensorTypeParam,
{
    pub fn to_layer(self) -> Layer<N> {
        match self {
            Norm::LayerNorm(layer) => Layer::LayerNorm(layer),
            Norm::RMSNorm(layer) => Layer::RMSNorm(layer),
        }
    }
}

impl NormType {
    pub fn from_gguf(
        &self,
        loader: &gguf::FileTensorLoader,
        c: &LLMConfig,
    ) -> anyhow::Result<Norm<f32>> {
        Ok(match self {
            NormType::LayerNorm => Norm::LayerNorm(LayerNorm::from_gguf(loader, c)?),
            NormType::RMSNorm => Norm::RMSNorm(RMSNorm::from_gguf(loader, c)?),
        })
    }
    pub fn from_safetensors(
        &self,
        loader: &safe::FileTensorLoader,
        config: &safe::ConfigJSON,
        c: &LLMConfig,
    ) -> anyhow::Result<Norm<f32>> {
        Ok(match self {
            NormType::LayerNorm => Norm::LayerNorm(LayerNorm::from_safetensors(loader, config, c)?),
            NormType::RMSNorm => Norm::RMSNorm(RMSNorm::from_safetensors(loader, config)?),
        })
    }
}
