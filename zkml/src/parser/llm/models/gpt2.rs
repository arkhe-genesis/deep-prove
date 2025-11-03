use crate::parser::llm::Norm;
use anyhow::ensure;

use crate::{
    Shape,
    layers::transformer::{attention::attention_mask::AttentionSpan, layernorm::LayerNorm},
    model::Model,
    parser::{
        ModelLoader,
        gguf::{FileTensorLoader, RawGGUF},
        json,
        json::RawJSON,
        llm::{
            Attention, FeedForward, HFTokenizer, LLMConfig, LLMModel,
            config::{AttentionConfig, AttentionHeadType, LLMStructure, PositionalConfig},
            tokenizer::TokenizerLoader,
            transformer::NormType,
        },
    },
    tensor::KeyedTensor,
};

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
        let model = LLMModel::from_gguf(&loader, &structure, Attention::from_gguf_gpt2)?;
        // even though the llm runtime doesn't care about the model input shape, which is designed for "static" input shapes, we still
        // need to provide one.
        let init_user_shape = Shape::from(vec![1]);
        let model = model.into_provable_model(&structure, init_user_shape)?;
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
        let model = LLMModel::from_json(&loader, &config)?;
        let structure = gpt2_structure(&config);
        let init_user_shape = Shape::from(vec![1]);
        let model = model.into_provable_model(&structure, init_user_shape)?;
        Ok((model, config))
    }
}

pub(crate) fn gpt2_structure(config: &LLMConfig) -> LLMStructure {
    LLMStructure {
        generic: config.clone(),
        norm_type: NormType::LayerNorm,
        positional_config: PositionalConfig::FixedPositional,
        attention_config: AttentionConfig {
            span: (1..=config.num_block)
                .map(|_| AttentionSpan::Full)
                .collect(),
            head: AttentionHeadType::MHA,
        },
        final_proj: true,
    }
}

impl Attention<f32> {
    pub(crate) fn from_gguf_gpt2(
        loader: &FileTensorLoader,
        c: &LLMStructure,
    ) -> anyhow::Result<Self> {
        let embedding_size = c.generic.embedding_size;
        let hidden_size = c.generic.hidden_size;
        ensure!(
            embedding_size == hidden_size,
            "embedding_size must be equal to hidden_size"
        );
        let (qkv_key, mut unfused_weights) =
            loader.unfuse_tensors("attn_qkv.weight", embedding_size * embedding_size)?;
        ensure!(unfused_weights.len() == 3, "qkv_weight must have 3 chunks");
        let q = KeyedTensor::new(
            format!("{qkv_key}.q"),
            crate::Tensor::new(
                vec![embedding_size, hidden_size].into(),
                unfused_weights.remove(0),
            )?
            .transpose()?,
        );
        let k = KeyedTensor::new(
            format!("{qkv_key}.k"),
            crate::Tensor::new(
                vec![embedding_size, hidden_size].into(),
                unfused_weights.remove(0),
            )?
            .transpose()?,
        );
        let v = KeyedTensor::new(
            format!("{qkv_key}.v"),
            crate::Tensor::new(
                vec![embedding_size, hidden_size].into(),
                unfused_weights.remove(0),
            )?
            .transpose()?,
        );

        let (qkv_bias_key, mut unfused_biases) =
            loader.unfuse_tensors("attn_qkv.bias", embedding_size)?;
        ensure!(unfused_biases.len() == 3, "qkv_bias must have 3 chunks");
        let q_bias = KeyedTensor::new(
            format!("{qkv_bias_key}.q"),
            crate::Tensor::new(vec![hidden_size].into(), unfused_biases.remove(0))?,
        );
        let k_bias = KeyedTensor::new(
            format!("{qkv_bias_key}.k"),
            crate::Tensor::new(vec![hidden_size].into(), unfused_biases.remove(0))?,
        );
        let v_bias = KeyedTensor::new(
            format!("{qkv_bias_key}.v"),
            crate::Tensor::new(vec![hidden_size].into(), unfused_biases.remove(0))?,
        );

        let attn_norm_loader = loader.pp("attn_");
        // Use new LayerNorm::from_loader
        let pre_norm = LayerNorm::from_gguf(&attn_norm_loader, &c.generic)?;

        // attn_output.weight is stored as [out_features, in_features] in GGUF (same as PyTorch)
        // Our MatMul layer expects the right-hand constant to be in the orientation [in_features, out_features],
        // so we transpose it once here after loading.
        let out = loader
            .get_tensor("attn_output.weight")?
            .try_map_tensor(|t| t.transpose())?;
        let out_bias = loader.get_tensor("attn_output.bias")?;
        ensure!(
            out.shape().as_ref() == &[embedding_size, embedding_size],
            "out must have shape [hidden_size, hidden_size]"
        );
        ensure!(
            out_bias.shape().as_ref() == &[embedding_size],
            "out_bias must have shape [hidden_size]"
        );

        let ffn_norm_loader = loader.pp("ffn_");

        let pre_ffn_norm = LayerNorm::from_gguf(&ffn_norm_loader, &c.generic)?;

        // Use new FeedForward::from_loader
        let ff = FeedForward::from_gguf_gpt2(loader, c)?;
        Ok(Self {
            out,
            out_bias: Some(out_bias),
            pre_norm: Norm::LayerNorm(pre_norm),
            q,
            q_bias: Some(q_bias),
            q_norm: None,
            k,
            k_bias: Some(k_bias),
            k_norm: None,
            v,
            v_bias: Some(v_bias),
            pre_ffn_norm: Norm::LayerNorm(pre_ffn_norm),
            feedforward: ff,
            post_norm: None,
            post_ffn_norm: None,
            span: AttentionSpan::Full,
        })
    }
}

impl FeedForward<f32> {
    pub fn from_gguf_gpt2(loader: &FileTensorLoader, c: &LLMStructure) -> anyhow::Result<Self> {
        let up = loader
            .get_tensor("ffn_up.weight")?
            .try_map_tensor(|t| t.transpose())?;
        let up_bias = Some(loader.get_tensor("ffn_up.bias")?);
        let down = loader
            .get_tensor("ffn_down.weight")?
            .try_map_tensor(|t| t.transpose())?;
        let down_bias = Some(loader.get_tensor("ffn_down.bias")?);
        ensure!(
            up.shape()[0] == c.generic.hidden_size,
            "up have shape {:?} but in features should be equal to hidden_size: {}",
            up.shape(),
            c.generic.hidden_size
        );
        ensure!(
            down.shape()[1] == c.generic.embedding_size,
            "down have shape {:?} but out features should be equal to embedding_size: {}",
            down.shape(),
            c.generic.embedding_size
        );
        Ok(Self {
            gate: None,
            up,
            up_bias,
            down,
            down_bias,
        })
    }
}

#[cfg(test)]
pub mod tests {
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
        let input = Tensor::new(vec![1].into(), vec![546.0]).unwrap();
        model.run_float(&[input])?;
        Ok(())
    }
}
