use crate::parser::{ModelLoader, llm::LLMConfig};

pub mod gemma3;
pub mod gpt2;

pub trait LLMModelLoader<DataFormat>: ModelLoader<DataFormat, ModelConfig = LLMConfig> {
    fn with_max_context_length(self, _max_ctx_length: usize) -> Self;
}

impl<DataFormat, T> LLMModelLoader<DataFormat> for T
where
    T: ModelLoader<DataFormat, ModelConfig = LLMConfig>,
{
    default fn with_max_context_length(self, _max_ctx_length: usize) -> Self {
        self
    }
}
