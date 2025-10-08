pub mod config;
pub mod ffn;
pub mod tokenizer;
pub mod transformer;
pub use crate::parser::{
    gguf::{self},
    json,
    llm::{ffn::FeedForward, transformer::Attention},
};
use anyhow::bail;
pub use config::{LLMConfig, LLMVariant};
use serde::{Deserialize, Serialize};
pub use tokenizer::{HFTokenizer, LLMTokenizer};

use crate::{
    Shape,
    layers::{
        Layer,
        matrix_mul::MatMul,
        transformer::{
            embeddings::Embeddings, layernorm::LayerNorm, logits::Logits, positional::Positional,
        },
    },
    model::Model,
    number::Number,
    padding::PaddingMode,
    parser::llm::transformer::Norm,
};

#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    Hash,
    derive_more::From,
    derive_more::Into,
    Serialize,
    Deserialize,
)]
pub struct Token(pub(crate) usize);

// i64 is the type used by token_to_i
impl From<i64> for Token {
    fn from(t: i64) -> Self {
        Self(t as usize)
    }
}

impl From<Token> for i64 {
    fn from(t: Token) -> Self {
        t.0 as i64
    }
}

impl From<&Token> for i64 {
    fn from(t: &Token) -> Self {
        t.0 as i64
    }
}

impl From<u32> for Token {
    fn from(t: u32) -> Self {
        Self(t as usize)
    }
}

impl From<&Token> for u32 {
    fn from(t: &Token) -> Self {
        t.0 as u32
    }
}

impl Token {
    pub fn as_number<N: Number>(&self) -> N {
        N::from_usize(self.0)
    }
}

#[derive(Debug, Clone)]
pub struct LLMModel {
    pub embeddings: Embeddings<f32>,
    pub positional: Positional<f32>,
    pub blocks: Vec<Attention<f32>>,
    /// Final LayerNorm applied after all transformer blocks (ln_f in GPT-2)
    pub final_norm: Norm<f32>,
    /// final projection on token sizes to before selecting next token
    pub final_proj: Option<MatMul<f32>>,
}

impl LLMModel {
    pub fn new(
        embeddings: Embeddings<f32>,
        positional: Positional<f32>,
        blocks: Vec<Attention<f32>>,
        final_norm: Norm<f32>,
        final_proj: Option<MatMul<f32>>,
    ) -> Self {
        Self {
            embeddings,
            positional,
            blocks,
            final_norm,
            final_proj,
        }
    }

    pub fn from_loader(
        loader: &gguf::FileTensorLoader,
        config: &LLMConfig,
    ) -> anyhow::Result<Self> {
        let embeddings = Embeddings::from_loader(loader)?;
        let positional = Positional::from_loader(loader, config)?;

        let num_layers = config.num_block;
        let blocks = (0..num_layers)
            .map(|i| Attention::from_loader(&loader.pp(&format!("blk.{i}.")), config))
            .collect::<anyhow::Result<Vec<Attention<f32>>>>()?;
        let blocks = blocks
            .into_iter()
            .zip(config.attention_config.spans())
            .map(|(attention, span)| attention.with_span(span))
            .collect();
        let final_norm =
            config
                .variant
                .norm_type()
                .from_loader(&loader.pp("output_"), config, false)?;
        let final_proj = match config.variant {
            LLMVariant::GPT2 => {
                //  there might or not be a bias
                let proj_weights = loader
                    .get_tensor("output.weight")?
                    .map_tensor(|t| t.transpose());
                let proj_bias = loader.get_tensor("output.bias").ok();
                Some(MatMul::new_constant(proj_weights, proj_bias)?)
            }
            LLMVariant::Gemma3 => None,
        };
        Ok(Self::new(
            embeddings, positional, blocks, final_norm, final_proj,
        ))
    }

    pub fn from_json(l: &json::FileTensorLoader, config: &LLMConfig) -> anyhow::Result<Self> {
        if let LLMVariant::Gemma3 = config.variant {
            bail!("Gemma3 is not supported yet for custom JSON format");
        }
        let embeddings = Embeddings::from_json(l)?;
        let positional = Positional::from_json(l, config)?;
        let num_layers = config.num_block;
        let blocks = (0..num_layers)
            .map(|i| Attention::from_json(&l.pp(&format!("blk.{i}.")), config))
            .collect::<anyhow::Result<Vec<Attention<f32>>>>()?;
        let final_norm = Norm::LayerNorm(LayerNorm::from_json(&l.pp("output_"), config)?);
        let proj_weights = l.get_tensor("output.weight")?.map_tensor(|t| t.transpose());
        let proj_bias = l.get_tensor("output.bias").ok();
        let final_proj = MatMul::new_constant(proj_weights, proj_bias)?;
        Ok(Self::new(
            embeddings,
            positional,
            blocks,
            final_norm,
            Some(final_proj),
        ))
    }
    /// Creates a Model<f32> from the GPT2Model. Currently it does NOT support the embeddings and positional nor
    /// multiple passes.
    /// User input shape is the shape of the user input tensor.
    pub fn into_provable_model(
        self,
        c: &LLMConfig,
        user_input_shape: Shape,
    ) -> anyhow::Result<Model<f32>> {
        let mut model =
            Model::new_from_input_shapes(vec![user_input_shape], PaddingMode::NoPadding);

        let mut last_node_id =
            Some(model.add_consecutive_layer(Layer::Embeddings(self.embeddings), None)?);
        if let LLMVariant::GPT2 = c.variant {
            last_node_id =
                Some(model.add_consecutive_layer(
                    Layer::Positional(self.positional.clone()),
                    last_node_id,
                )?);
        }
        for block in self.blocks {
            let pos = if let LLMVariant::Gemma3 = c.variant {
                Some(self.positional.clone())
            } else {
                None
            };
            last_node_id = Some(block.write_to_model(&mut model, last_node_id, c, pos)?);
        }
        last_node_id = Some(model.add_consecutive_layer(self.final_norm.to_layer(), last_node_id)?);
        if let Some(final_proj) = self.final_proj {
            last_node_id =
                Some(model.add_consecutive_layer(Layer::MatMul(final_proj), last_node_id)?);
        }
        model.add_consecutive_layer(Layer::Logits(Logits::Argmax), last_node_id)?;
        model.automatic_output_labelling()?;
        Ok(model)
    }
}

#[cfg(test)]
mod tests {
    use crate::parser::{
        file_cache,
        gguf::tests::{GEMMA3_Q8, GPT2_Q8_0},
    };

    use super::*;

    fn test_load_model_from_gguf(path: &str) {
        let path = file_cache::from_cache(path).unwrap();
        let loader = gguf::FileTensorLoader::from_path(path).unwrap();
        let config = LLMConfig::from_content(&loader).unwrap();
        let intermediate = LLMModel::from_loader(&loader, &config).unwrap();
        intermediate
            .into_provable_model(&config, Shape::from(vec![1]))
            .unwrap();
    }

    #[test]
    fn test_load_model_from_gguf_gpt2() {
        test_load_model_from_gguf(GPT2_Q8_0);
    }

    #[test]
    fn test_load_model_from_gguf_gemma3() {
        test_load_model_from_gguf(GEMMA3_Q8);
    }
}
