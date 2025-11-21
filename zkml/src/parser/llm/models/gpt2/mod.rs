use crate::parser::{Load, llm::models::gpt2::decoder::GPT2Decoder};

use crate::{
    Shape,
    layers::transformer::attention_mask::AttentionSpan,
    model::Model,
    parser::{
        ModelLoader,
        gguf::RawGGUF,
        json,
        json::RawJSON,
        llm::{
            HFTokenizer, LLMConfig, LLMModel,
            config::{AttentionConfig, AttentionHeadType, LLMStructure, PositionalConfig},
            tokenizer::TokenizerLoader,
            transformer::NormType,
        },
    },
};

pub mod decoder;

/// Loader for the GPT2 family of models.
/// For more information about GPT2, see
/// https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf
#[derive(Clone, Debug, Default)]
pub struct GPT2;

impl GPT2 {
    pub fn new() -> Self {
        GPT2
    }
}

pub const GPT2_VARIANTS: &[&str] = &[
    "gpt2",
    "Tmkrzx_X",
    "distilgpt2",
    "toy_gpt2",
    "sshleifer/tiny-gpt2",
];

pub const GPT2_NAME: &str = "gpt2";
pub const GPT2_GGUF_NAME: &str = GPT2_NAME;
pub const GPT2_Q8_0: &str = "gpt2.Q8_0.gguf";

pub fn is_gpt2_model(names: &[String]) -> bool {
    names
        .iter()
        .any(|name| GPT2_VARIANTS.contains(&name.as_str()))
}

impl TokenizerLoader<RawGGUF> for GPT2 {
    fn load_tokenizer(&self, raw: &RawGGUF) -> anyhow::Result<HFTokenizer> {
        let loader = raw.loader()?;
        let tokenizer = HFTokenizer::bpe_from_gguf(&loader)?;
        Ok(tokenizer)
    }
}

impl ModelLoader<RawGGUF> for GPT2 {
    type ModelConfig = LLMConfig;

    fn model_name(&self) -> String {
        GPT2_NAME.to_string()
    }

    fn parse(&self, raw: &RawGGUF) -> anyhow::Result<(Model<f32>, Self::ModelConfig)> {
        let loader = raw.loader()?;
        let config = LLMConfig::from_gguf(&loader, "gpt2")?;
        let structure = gpt2_structure(&config);
        let model = LLMModel::<GPT2Decoder>::from_loader(&loader, &structure)?;
        // even though the llm runtime doesn't care about the model input shape, which is designed for "static" input shapes, we still
        // need to provide one.
        let init_user_shape = Shape::from(vec![1]);
        let model = model.into_provable_model(init_user_shape)?;
        Ok((model, config))
    }
}

impl ModelLoader<RawJSON> for GPT2 {
    type ModelConfig = LLMConfig;

    fn model_name(&self) -> String {
        GPT2_NAME.to_string()
    }

    fn parse(&self, raw: &RawJSON) -> anyhow::Result<(Model<f32>, Self::ModelConfig)> {
        let loader = json::FileTensorLoader::new_from_path(&raw.0)?;
        let config = LLMConfig::from_json(&loader)?;
        let model = LLMModel::<GPT2Decoder>::from_loader(&loader, &config)?;
        let init_user_shape = Shape::from(vec![1]);
        let model = model.into_provable_model(init_user_shape)?;
        Ok((model, config))
    }
}

pub(crate) fn gpt2_structure(config: &LLMConfig) -> LLMStructure {
    LLMStructure {
        generic: config.clone(),
        norm_type: NormType::LayerNorm,
        global_positional: Some(PositionalConfig::FixedPositional),
        attention_config: AttentionConfig {
            span: (1..=config.num_block)
                .map(|_| AttentionSpan::Full)
                .collect(),
            head: AttentionHeadType::MHA,
        },
    }
}

#[cfg(test)]
pub mod tests {

    use tenstore::GenStore;

    use super::*;
    use crate::{
        Tensor,
        parser::{file_cache, llm::LLMTokenizer},
    };

    pub const GPT2_Q8_0: &str = "gpt2.Q8_0.gguf";

    #[test]
    fn test_gpt2_load_gguf_tokenizer() -> anyhow::Result<()> {
        let model_path = file_cache::from_cache(GPT2_Q8_0)?;
        let mygguf = RawGGUF::new(model_path);
        let tokenizer = GPT2.load_tokenizer(&mygguf)?;
        let s = "do or don't. there is no try.";
        let tokens = tokenizer.tokenize(s);
        let s2 = tokenizer.detokenize(&tokens);
        assert_eq!(s, s2);
        Ok(())
    }

    #[test]
    fn test_gpt2_load_gguf_model() -> anyhow::Result<()> {
        let model_path = file_cache::from_cache(GPT2_Q8_0)?;
        let mygguf = RawGGUF::new(model_path);
        let (model, config) = GPT2.parse(&mygguf)?;
        assert_eq!(config.num_heads, 12);
        assert_eq!(config.num_block, 12);
        assert_eq!(config.embedding_size, 768);
        assert_eq!(config.hidden_size, 768);
        assert_eq!(config.context_length, 1024);
        assert_eq!(config.norm_epsilon, 1e-5);
        assert_eq!(config.vocab_size, 50257);
        assert_eq!(config.eos_token, 50256usize.into());
        let input = Tensor::new(vec![1].into(), vec![546.0f32])?;
        model.run_float(vec![input], &mut GenStore::default())?;
        Ok(())
    }
}
