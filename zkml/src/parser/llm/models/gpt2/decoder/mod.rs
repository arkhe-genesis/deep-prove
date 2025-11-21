//! Defines the decoder only transformer used in the GPT-2 model.

use crate::{
    graph::NodeId,
    layers::{
        Layer,
        activation::{ActivationLayer, GELU},
        add::Add,
        transformer::layernorm::LayerNorm,
    },
    model::{LayerInsertion, Model},
    parser::{
        Load,
        gguf::FileTensorLoader as GGUFLoader,
        json::FileTensorLoader as JSONLoader,
        llm::{
            LLMConfig, config::LLMStructure, models::gpt2::decoder::attention::GPT2Attention,
            transformer::feed_forward::FeedForwardNetwork,
        },
    },
};

use anyhow::{Context, Result, ensure};

pub mod attention;

pub struct GPT2Decoder {
    pre_attention_layernorm: LayerNorm<f32>,
    attention_mechanism: GPT2Attention,
    pre_ffn_layernorm: LayerNorm<f32>,
    feed_forward: FeedForwardNetwork,
}

impl LayerInsertion for GPT2Decoder {
    fn add_to_model(
        self,
        model: &mut Model<f32>,
        previous_node_id: Option<NodeId>,
    ) -> Result<NodeId> {
        // First we check that there is a previous node to connect to, a transformer block should never be the first node
        ensure!(
            previous_node_id.is_some(),
            "Transformer block cannot be the first node in the model"
        );

        let GPT2Decoder {
            pre_attention_layernorm,
            attention_mechanism,
            pre_ffn_layernorm,
            feed_forward,
        } = self;

        let initial_norm_id = model
            .add_consecutive_layer(Layer::LayerNorm(pre_attention_layernorm), previous_node_id)?;
        // Unwrap is safe because we have checked for previous_node_id above
        let residual_id = previous_node_id.unwrap();
        let post_attention_id = attention_mechanism.add_to_model(model, Some(initial_norm_id))?;

        let add_id = model.graph.add_inner(Layer::Add(Add::<f32>::new()))?;
        model.add_edge(residual_id, add_id, (0, 0))?;
        model.add_edge(post_attention_id, add_id, (0, 1))?;

        let pre_ffn_norm_id =
            model.add_consecutive_layer(Layer::LayerNorm(pre_ffn_layernorm), Some(add_id))?;

        let post_ffn_id = feed_forward.add_to_model(model, Some(pre_ffn_norm_id))?;
        let final_add_id = model.graph.add_inner(Layer::Add(Add::<f32>::new()))?;
        model.add_edge(add_id, final_add_id, (0, 0))?;
        model.add_edge(post_ffn_id, final_add_id, (0, 1))?;

        Ok(final_add_id)
    }
}

impl Load<GGUFLoader> for GPT2Decoder {
    type Config = LLMStructure;

    fn from_loader(loader: &GGUFLoader, structure: &Self::Config) -> anyhow::Result<Self>
    where
        Self: Sized,
    {
        let embedding_size = structure.generic.embedding_size;
        let hidden_size = structure.generic.hidden_size;
        ensure!(
            embedding_size == hidden_size,
            "embedding_size must be equal to hidden_size"
        );

        let attention_mechanism = GPT2Attention::from_loader(loader, structure)?;

        let attn_norm_loader = loader.pp("attn_");
        // Use new LayerNorm::from_loader
        let pre_norm = LayerNorm::from_gguf(&attn_norm_loader, &structure.generic)?;

        let ffn_norm_loader = loader.pp("ffn_");

        let pre_ffn_norm = LayerNorm::from_gguf(&ffn_norm_loader, &structure.generic)?;

        // Use new FeedForward::from_loader
        let feed_forward = {
            let up = loader
                .get_tensor("ffn_up.weight")?
                .try_map_tensor(|t| t.transpose())?;
            let up_bias = Some(loader.get_tensor("ffn_up.bias")?);
            let down = loader
                .get_tensor("ffn_down.weight")?
                .try_map_tensor(|t| t.transpose())?;
            let down_bias = Some(loader.get_tensor("ffn_down.bias")?);
            ensure!(
                up.shape()[0] == structure.generic.hidden_size,
                "up have shape {:?} but in features should be equal to hidden_size: {}",
                up.shape(),
                structure.generic.hidden_size
            );
            ensure!(
                down.shape()[1] == structure.generic.embedding_size,
                "down have shape {:?} but out features should be equal to embedding_size: {}",
                down.shape(),
                structure.generic.embedding_size
            );
            FeedForwardNetwork {
                gate: None,
                up,
                up_bias,
                down,
                down_bias,
                activation: ActivationLayer::Gelu(GELU::new()),
            }
        };

        Ok(Self {
            pre_attention_layernorm: pre_norm,
            attention_mechanism,
            pre_ffn_layernorm: pre_ffn_norm,
            feed_forward,
        })
    }
}

impl Load<JSONLoader> for GPT2Decoder {
    type Config = LLMConfig;

    fn from_loader(l: &JSONLoader, config: &Self::Config) -> anyhow::Result<Self> {
        let pre_attention_layernorm = LayerNorm::from_json(&l.pp("attn_"), config)
            .context("Failed to load LayerNorm for attention in from_json")?;

        let attention_mechanism = GPT2Attention::from_loader(l, config)
            .context("Failed to load GPT2Attention in from_json")?;

        let pre_ffn_norm = LayerNorm::from_json(&l.pp("ffn_"), config)?;
        let feed_forward = {
            let up = l.get_tensor("ffn_up.weight")?;
            let up_bias = l.get_tensor("ffn_up.bias")?;
            let down = l.get_tensor("ffn_down.weight")?;
            let down_bias = l.get_tensor("ffn_down.bias")?;
            ensure!(
                up.shape()[0] == config.hidden_size,
                "up have shape {:?} but in features should be equal to hidden_size: {}",
                up.shape(),
                config.hidden_size
            );
            ensure!(
                down.shape()[1] == config.embedding_size,
                "down have shape {:?} but out features should be equal to embedding_size: {}",
                down.shape(),
                config.embedding_size
            );
            FeedForwardNetwork {
                gate: None,
                up,
                up_bias: Some(up_bias),
                down,
                down_bias: Some(down_bias),
                activation: ActivationLayer::Gelu(GELU::new()),
            }
        };

        Ok(Self {
            pre_attention_layernorm,
            attention_mechanism,
            pre_ffn_layernorm: pre_ffn_norm,
            feed_forward,
        })
    }
}
