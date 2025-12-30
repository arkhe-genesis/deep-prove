pub mod config;

pub mod models;
pub mod tokenizer;
pub mod transformer;
pub use crate::parser::{
    gguf::{self, FileTensorLoader as GGUFLoader},
    json::{self, FileTensorLoader as JSONLoader},
    safe::{ConfigJSON, FileTensorLoader as SafeLoader},
};
pub use config::LLMConfig;
use serde::{Deserialize, Serialize};
use tenstore::StorageKey;
pub use tokenizer::{HFTokenizer, LLMTokenizer};

use crate::{
    Shape,
    layers::{
        Layer,
        einsum::EinSum,
        transformer::{
            embeddings::Embeddings, layernorm::LayerNorm, logits::Logits, positional::Positional,
        },
    },
    model::{LayerInsertion, Model},
    number::Number,
    padding::PaddingMode,
    parser::{
        Load,
        llm::{config::LLMStructure, transformer::Norm},
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
pub struct LLMModel<T> {
    pub embeddings: Embeddings<f32>,
    pub global_positional: Option<Positional<f32>>,
    pub blocks: Vec<T>,
    /// Final LayerNorm applied after all transformer blocks (ln_f in GPT-2)
    pub final_norm: Norm<f32>,
    /// final projection on token sizes to before selecting next token
    pub final_proj: EinSum<f32>,
}

impl<T> LLMModel<T> {
    pub fn new(
        embeddings: Embeddings<f32>,
        global_positional: Option<Positional<f32>>,
        blocks: Vec<T>,
        final_norm: Norm<f32>,
        final_proj: EinSum<f32>,
    ) -> Self {
        Self {
            embeddings,
            global_positional,
            blocks,
            final_norm,
            final_proj,
        }
    }
}

impl<T: LayerInsertion> LLMModel<T> {
    /// Creates a Model<f32> from the [`LLMModel`].
    pub fn into_provable_model(self, user_input_shape: Shape) -> anyhow::Result<Model<f32>> {
        let mut model =
            Model::new_from_input_shapes(vec![user_input_shape], PaddingMode::NoPadding);

        let mut last_node_id =
            Some(model.add_consecutive_layer(Layer::Embeddings(self.embeddings), None)?);
        if let Some(positional) = self.global_positional {
            last_node_id =
                Some(model.add_consecutive_layer(Layer::Positional(positional), last_node_id)?);
        }
        for block in self.blocks {
            last_node_id = Some(block.add_to_model(&mut model, last_node_id)?);
        }
        last_node_id = Some(model.add_consecutive_layer(self.final_norm.to_layer(), last_node_id)?);

        last_node_id =
            Some(model.add_consecutive_layer(Layer::EinSum(self.final_proj), last_node_id)?);

        model.add_consecutive_layer(Layer::Logits(Logits::Argmax), last_node_id)?;
        model.automatic_output_labelling()?;
        Ok(model)
    }
}

impl<T> Load<GGUFLoader> for LLMModel<T>
where
    T: Load<GGUFLoader, Config = LLMStructure> + LayerInsertion,
{
    type Config = LLMStructure;

    fn from_loader(loader: &GGUFLoader, config: &Self::Config) -> anyhow::Result<Self>
    where
        Self: Sized,
    {
        let embeddings = Embeddings::from_loader(loader)?;
        let global_positional = config
            .global_positional
            .as_ref()
            .map(|p| Positional::from_gguf(loader, config, p))
            .transpose()?;

        let num_layers = config.generic.num_block;
        let blocks = (0..num_layers)
            .map(|i| T::from_loader(&loader.pp(&format!("blk.{i}.")), config))
            .collect::<anyhow::Result<Vec<T>>>()?;

        let final_norm = config
            .norm_type
            .from_gguf(&loader.pp("output_"), &config.generic)?;

        // Now we work out the final projection
        // there may or may not be a bias
        let input_terms = "X(se)@WE(ve)";
        let proj_bias = loader.get_tensor("output.bias").ok();
        let output_terms = if proj_bias.is_some() {
            "O(sv)+BIAS(v)"
        } else {
            "O(sv)"
        };
        let equation = format!("{input_terms}->{output_terms}");
        let mut proj_weights = embeddings.mat.clone();
        // We need to modify the key to avoid conflicts between the embeddings matrix and the final projection
        // the embeddings matrix is scaled _after_ parsing the model, but the final projection is *not* scaled
        // so in effect, these are two different tensors and thus need to be represented by two different keys
        proj_weights.key = StorageKey::from(format!("{}_final_proj", proj_weights.key));
        let final_proj = EinSum::<f32>::new(equation, vec![Some(proj_weights)], vec![proj_bias])?;

        Ok(Self::new(
            embeddings,
            global_positional,
            blocks,
            final_norm,
            final_proj,
        ))
    }
}

impl<T> Load<SafeLoader> for LLMModel<T>
where
    T: Load<SafeLoader, Config = (LLMStructure, ConfigJSON)> + LayerInsertion,
{
    type Config = (LLMStructure, ConfigJSON);
    fn from_loader(loader: &SafeLoader, loader_config: &Self::Config) -> anyhow::Result<Self>
    where
        Self: Sized,
    {
        let (structure, config) = loader_config;
        let embeddings = Embeddings::from_safetensors_loader(loader)?;
        let global_positional = structure
            .global_positional
            .as_ref()
            .map(|p| Positional::from_safetensors_loader(loader, structure, p))
            .transpose()?;

        let num_layers = structure.generic.num_block;
        let blocks = (0..num_layers)
            .map(|i| T::from_loader(&loader.pp(&format!("model.layers.{i}.")), loader_config))
            .collect::<anyhow::Result<Vec<T>>>()?;

        let final_norm = structure.norm_type.from_safetensors(
            &loader.pp("model."),
            config,
            &structure.generic,
        )?;
        // Now we work out the final projection
        // there may or may not be a bias
        let input_terms = "X(se)@WE(ve)";
        let proj_bias = loader.get_tensor("output.bias").ok();
        let output_terms = if proj_bias.is_some() {
            "O(sv)+BIAS(v)"
        } else {
            "O(sv)"
        };
        let equation = format!("{input_terms}->{output_terms}");
        let mut proj_weights = embeddings.mat.clone();
        // We need to modify the key to avoid conflicts between the embeddings matrix and the final projection
        // the embeddings matrix is scaled _after_ parsing the model, but the final projection is *not* scaled
        // so in effect, these are two different tensors and thus need to be represented by two different keys
        proj_weights.key = StorageKey::from(format!("{}_final_proj", proj_weights.key));
        let final_proj = EinSum::<f32>::new(equation, vec![Some(proj_weights)], vec![proj_bias])?;

        Ok(Self::new(
            embeddings,
            global_positional,
            blocks,
            final_norm,
            final_proj,
        ))
    }
}

impl<T> Load<JSONLoader> for LLMModel<T>
where
    T: Load<JSONLoader, Config = LLMConfig> + LayerInsertion,
{
    type Config = LLMConfig;

    fn from_loader(l: &JSONLoader, config: &Self::Config) -> anyhow::Result<Self>
    where
        Self: Sized,
    {
        let embeddings = Embeddings::from_json(l)?;
        let positional = Some(Positional::from_json(l, config)?);
        let num_layers = config.num_block;
        let blocks = (0..num_layers)
            .map(|i| T::from_loader(&l.pp(&format!("blk.{i}.")), config))
            .collect::<anyhow::Result<Vec<T>>>()?;
        let final_norm = Norm::LayerNorm(LayerNorm::from_json(&l.pp("output_"), config)?);

        // Now we work out the final projection
        // there may or may not be a bias
        let input_terms = "X(se)@WE(ve)";
        let proj_bias = l.get_tensor("output.bias").ok();
        let output_terms = if proj_bias.is_some() {
            "O(sv)+BIAS(v)"
        } else {
            "O(sv)"
        };
        let equation = format!("{input_terms}->{output_terms}");
        let proj_weights = embeddings.mat.clone();
        let final_proj = EinSum::<f32>::new(equation, vec![Some(proj_weights)], vec![proj_bias])?;
        Ok(Self::new(
            embeddings, positional, blocks, final_norm, final_proj,
        ))
    }
}
