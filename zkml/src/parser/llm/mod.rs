pub mod config;
pub mod ffn;
pub mod models;
pub mod tokenizer;
pub mod transformer;
pub use crate::parser::{
    gguf::{self},
    json,
    llm::{ffn::FeedForward, transformer::Attention},
};
pub use config::LLMConfig;
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
    parser::llm::{
        config::{LLMStructure, PositionalConfig},
        transformer::Norm,
    },
    tensor::TensorTypeParam,
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

impl From<u64> for Token {
    fn from(t: u64) -> Self {
        Self(t as usize)
    }
}

impl Token {
    pub fn as_number<N: Number>(&self) -> N {
        N::from_usize(self.0)
    }

    pub fn as_tensor_type_param<T: TensorTypeParam>(&self) -> T {
        T::from_usize(self.0)
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

    pub fn from_safetensors_loader(
        loader: &crate::parser::safe::FileTensorLoader,
        config: &crate::parser::safe::ConfigJSON,
        structure: &LLMStructure,
        attention_factory: fn(
            &crate::parser::safe::FileTensorLoader,
            &LLMStructure,
        ) -> anyhow::Result<Attention<f32>>,
    ) -> anyhow::Result<Self> {
        let embeddings = Embeddings::from_safetensors_loader(loader)?;
        let positional = Positional::from_safetensors_loader(loader, config, structure)?;

        let num_layers = structure.generic.num_block;
        let blocks = (0..num_layers)
            .map(|i| attention_factory(&loader.pp(&format!("model.layers.{i}.")), structure))
            .collect::<anyhow::Result<Vec<Attention<f32>>>>()?;
        let blocks = blocks
            .into_iter()
            .zip(structure.attention_config.spans())
            .map(|(attention, span)| attention.with_span(span))
            .collect();
        let final_norm = structure.norm_type.from_safetensors(
            &loader.pp("model."),
            config,
            &structure.generic,
            false,
        )?;
        let final_proj = if structure.final_proj {
            //  there might or not be a bias
            let proj_weights = loader.get_tensor("output.weight")?;
            let proj_bias = loader.get_tensor("output.bias").ok();
            Some(MatMul::new_constant(proj_weights, proj_bias)?)
        } else {
            None
        };
        Ok(Self::new(
            embeddings, positional, blocks, final_norm, final_proj,
        ))
    }

    pub fn from_gguf(
        loader: &gguf::FileTensorLoader,
        config: &LLMStructure,
        attention_factory: fn(
            &gguf::FileTensorLoader,
            &LLMStructure,
        ) -> anyhow::Result<Attention<f32>>,
    ) -> anyhow::Result<Self> {
        let embeddings = Embeddings::from_loader(loader)?;
        let positional = Positional::from_gguf(loader, config)?;

        let num_layers = config.generic.num_block;
        let blocks = (0..num_layers)
            .map(|i| attention_factory(&loader.pp(&format!("blk.{i}.")), config))
            .collect::<anyhow::Result<Vec<Attention<f32>>>>()?;
        let blocks = blocks
            .into_iter()
            .zip(config.attention_config.spans())
            .map(|(attention, span)| attention.with_span(span))
            .collect();
        let final_norm =
            config
                .norm_type
                .from_gguf(&loader.pp("output_"), &config.generic, false)?;
        let final_proj = if config.final_proj {
            //  there might or not be a bias
            let proj_weights = loader
                .get_tensor("output.weight")?
                .map_tensor(|t| t.transpose());
            let proj_bias = loader.get_tensor("output.bias").ok();
            Some(MatMul::new_constant(proj_weights, proj_bias)?)
        } else {
            None
        };
        Ok(Self::new(
            embeddings, positional, blocks, final_norm, final_proj,
        ))
    }

    pub fn from_json(l: &json::FileTensorLoader, config: &LLMConfig) -> anyhow::Result<Self> {
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
        c: &LLMStructure,
        user_input_shape: Shape,
    ) -> anyhow::Result<Model<f32>> {
        let mut model =
            Model::new_from_input_shapes(vec![user_input_shape], PaddingMode::NoPadding);

        let mut last_node_id =
            Some(model.add_consecutive_layer(Layer::Embeddings(self.embeddings), None)?);
        if let PositionalConfig::FixedPositional = &c.positional_config {
            last_node_id =
                Some(model.add_consecutive_layer(
                    Layer::Positional(self.positional.clone()),
                    last_node_id,
                )?);
        }
        for block in self.blocks {
            let pos = if let PositionalConfig::Rope(_) = &c.positional_config {
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
