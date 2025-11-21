//! Defines the structure of the GPT-2 models attention Mechanism.

use crate::{
    Tensor,
    graph::NodeId,
    layers::{Layer, transformer::attention_mask::AttentionSpan},
    model::Model,
    parser::{
        Load,
        gguf::FileTensorLoader as GGUFLoader,
        json::{FileTensorLoader as JSONLoader, unfuse_crate_tensors},
        llm::{
            LLMConfig,
            config::LLMStructure,
            transformer::attention_layer::{AttentionCacheConfig, AttentionMechanism},
        },
    },
    tensor::KeyedTensor,
};

use anyhow::{Context, Result, ensure};
use tracing::trace;

#[derive(Clone, Debug)]
pub struct GPT2Attention {
    max_context_length: usize,
    num_heads: usize,
    head_dim: usize,
    wq: KeyedTensor<f32>,
    q_bias: KeyedTensor<f32>,
    wk: KeyedTensor<f32>,
    k_bias: KeyedTensor<f32>,
    wv: KeyedTensor<f32>,
    v_bias: KeyedTensor<f32>,
    wo: KeyedTensor<f32>,
    o_bias: KeyedTensor<f32>,
}

impl AttentionMechanism for GPT2Attention {
    fn total_heads(&self) -> usize {
        self.num_heads
    }

    fn kv_total_heads(&self) -> usize {
        self.num_heads
    }

    fn heads_per_kv_head(&self) -> usize {
        1
    }

    fn max_context_length(&self) -> usize {
        self.max_context_length
    }

    fn head_dim(&self) -> usize {
        self.head_dim
    }

    fn uses_qkv_bias(&self) -> bool {
        true
    }

    fn uses_out_bias(&self) -> bool {
        true
    }

    fn qkv_tensors(&self) -> (Vec<KeyedTensor<f32>>, Vec<Option<KeyedTensor<f32>>>) {
        let weights = vec![self.wq.clone(), self.wk.clone(), self.wv.clone()];
        let biases = vec![
            Some(self.q_bias.clone()),
            Some(self.k_bias.clone()),
            Some(self.v_bias.clone()),
        ];
        (weights, biases)
    }

    fn out_tensors(&self) -> (KeyedTensor<f32>, Option<KeyedTensor<f32>>) {
        (self.wo.clone(), Some(self.o_bias.clone()))
    }

    fn attention_span(&self) -> AttentionSpan {
        AttentionSpan::Full
    }

    fn insert_custom_logic(
        self,
        model: &mut Model<f32>,
        qkv_einsum_id: NodeId,
        query_key_id: NodeId,
    ) -> Result<()> {
        // First we add the caching to the QKV einsum
        let AttentionCacheConfig { caching_dim, .. } = self.caching_dim();

        let qkv_einsum = model.graph.node_mut(qkv_einsum_id).ok_or(anyhow::anyhow!("Could not insert GPT2Attention, QKV EinSum with ID({qkv_einsum_id}) has not been inserted"))?;
        if let Some(Layer::EinSum(einsum)) = qkv_einsum.as_inner_mut() {
            einsum.with_caches(vec![None, Some(caching_dim), Some(caching_dim)])?;
        } else {
            anyhow::bail!("QKV Einsum Node is not an EinSum layer");
        }

        model.add_edge(qkv_einsum_id, query_key_id, vec![(0, 0), (1, 1)])?;
        Ok(())
    }
}

impl Load<GGUFLoader> for GPT2Attention {
    type Config = LLMStructure;

    fn from_loader(loader: &GGUFLoader, c: &LLMStructure) -> anyhow::Result<Self>
    where
        Self: Sized,
    {
        let embedding_size = c.generic.embedding_size;
        let hidden_size = c.generic.hidden_size;
        ensure!(
            embedding_size == hidden_size,
            "embedding_size must be equal to hidden_size"
        );
        let (qkv_key, mut unfused_weights) =
            loader.unfuse_tensors("attn_qkv.weight", embedding_size * embedding_size)?;

        ensure!(unfused_weights.len() == 3, "qkv_weight must have 3 chunks");
        let wq = KeyedTensor::new(
            format!("{qkv_key}.q"),
            crate::Tensor::new(
                vec![embedding_size, hidden_size].into(),
                unfused_weights.remove(0),
            )?
            .transpose()?
            .reshaped(vec![embedding_size, c.generic.num_heads, c.generic.head_size].into())?,
        );
        let wk = KeyedTensor::new(
            format!("{qkv_key}.k"),
            crate::Tensor::new(
                vec![embedding_size, hidden_size].into(),
                unfused_weights.remove(0),
            )?
            .transpose()?
            .reshaped(vec![embedding_size, c.generic.num_heads, c.generic.head_size].into())?,
        );
        let wv = KeyedTensor::new(
            format!("{qkv_key}.v"),
            crate::Tensor::new(
                vec![embedding_size, hidden_size].into(),
                unfused_weights.remove(0),
            )?
            .transpose()?
            .reshaped(vec![embedding_size, c.generic.num_heads, c.generic.head_size].into())?,
        );

        let (qkv_bias_key, mut unfused_biases) =
            loader.unfuse_tensors("attn_qkv.bias", embedding_size)?;
        ensure!(unfused_biases.len() == 3, "qkv_bias must have 3 chunks");
        let q_bias = KeyedTensor::new(
            format!("{qkv_bias_key}.q"),
            crate::Tensor::new(
                vec![c.generic.num_heads, c.generic.head_size].into(),
                unfused_biases.remove(0),
            )?,
        );
        let k_bias = KeyedTensor::new(
            format!("{qkv_bias_key}.k"),
            crate::Tensor::new(
                vec![c.generic.num_heads, c.generic.head_size].into(),
                unfused_biases.remove(0),
            )?,
        );
        let v_bias = KeyedTensor::new(
            format!("{qkv_bias_key}.v"),
            crate::Tensor::new(
                vec![c.generic.num_heads, c.generic.head_size].into(),
                unfused_biases.remove(0),
            )?,
        );

        // attn_output.weight is stored as [out_features, in_features] in GGUF (same as PyTorch)
        // Our MatMul layer expects the right-hand constant to be in the orientation [in_features, out_features],
        // so we transpose it once here after loading.
        let wo = loader
            .get_tensor("attn_output.weight")?
            .try_map_tensor(|t| {
                t.transpose()?
                    .reshaped(vec![c.generic.num_heads, c.generic.head_size, embedding_size].into())
            })?;
        let o_bias = loader.get_tensor("attn_output.bias")?;
        ensure!(
            wo.shape().as_ref() == &[c.generic.num_heads, c.generic.head_size, embedding_size],
            "out must have shape [hidden_size, hidden_size]"
        );
        ensure!(
            o_bias.shape().as_ref() == &[embedding_size],
            "out_bias must have shape [hidden_size]"
        );

        Ok(Self {
            max_context_length: c.generic.context_length,
            num_heads: c.generic.num_heads,
            head_dim: c.generic.head_size,
            wq,
            q_bias,
            wk,
            k_bias,
            wv,
            v_bias,
            wo,
            o_bias,
        })
    }
}

impl Load<JSONLoader> for GPT2Attention {
    type Config = LLMConfig;

    fn from_loader(l: &JSONLoader, structure: &LLMConfig) -> anyhow::Result<Self>
    where
        Self: Sized,
    {
        let fused_qkv_weight = l
            .get_tensor("attn_qkv.weight")
            .context("Failed to load attn_qkv.weight in from_json")?;
        let fused_qkv_bias = l
            .get_tensor("attn_qkv.bias")
            .context("Failed to load attn_qkv.bias in from_json")?;

        let hidden_size = structure.hidden_size; // embedding_dim for GPT-2

        // Unfuse weights:
        // Expected shape of fused_qkv_weight is [3 * hidden_size, hidden_size] after python script transpose.
        // Each individual q, k, v weight matrix should be [hidden_size, hidden_size].
        // So, each chunk has hidden_size * hidden_size elements.
        let weight_chunk_elements = hidden_size * hidden_size;
        let mut unfused_weights_data =
            unfuse_crate_tensors(fused_qkv_weight.tensor(), weight_chunk_elements, 3)
                .context("Failed to unfuse QKV weights in from_json")?;

        let wq = KeyedTensor::new(
            format!("{}.q", fused_qkv_weight.key),
            Tensor::new(
                vec![
                    structure.embedding_size,
                    structure.num_heads,
                    structure.head_size,
                ]
                .into(),
                unfused_weights_data.remove(0),
            )?,
        );

        let wk = KeyedTensor::new(
            format!("{}.k", fused_qkv_weight.key),
            Tensor::new(
                vec![
                    structure.embedding_size,
                    structure.num_heads,
                    structure.head_size,
                ]
                .into(),
                unfused_weights_data.remove(0),
            )?,
        );
        let wv = KeyedTensor::new(
            format!("{}.v", fused_qkv_weight.key),
            Tensor::new(
                vec![
                    structure.embedding_size,
                    structure.num_heads,
                    structure.head_size,
                ]
                .into(),
                unfused_weights_data.remove(0),
            )?,
        );
        trace!("fused qkv: {fused_qkv_weight:?}");
        trace!("qkv full tensor {unfused_weights_data:?}");
        trace!("q_weight {:?}", wq.get_data());

        // Unfuse biases:
        // Expected shape of fused_qkv_bias is [3 * hidden_size].
        // Each individual q, k, v bias vector should be [hidden_size].
        // So, each chunk has hidden_size elements.
        let bias_chunk_elements = hidden_size;
        let fused_qvk_bias_key = fused_qkv_bias.key.clone();
        let mut unfused_biases_data =
            unfuse_crate_tensors(fused_qkv_bias.into_tensor(), bias_chunk_elements, 3)
                .context("Failed to unfuse QKV biases in from_json")?;

        let q_bias = KeyedTensor::new(
            format!("{fused_qvk_bias_key}.q"),
            Tensor::new(
                vec![structure.num_heads, structure.head_size].into(),
                unfused_biases_data.remove(0),
            )?,
        );
        let k_bias = KeyedTensor::new(
            format!("{fused_qvk_bias_key}.k"),
            Tensor::new(
                vec![structure.num_heads, structure.head_size].into(),
                unfused_biases_data.remove(0),
            )?,
        );
        let v_bias = KeyedTensor::new(
            format!("{fused_qvk_bias_key}.v"),
            Tensor::new(
                vec![structure.num_heads, structure.head_size].into(),
                unfused_biases_data.remove(0),
            )?,
        );

        // These are the individual Q, K, V matrices and biases now.
        // The QKV struct or logic that consumes these will handle them.
        // For now, let's assume Attention struct stores these directly if QKV is not used here.
        // Or, construct the QKV layer if that's the design.
        // The original struct for Attention<f32> directly stores q, q_bias, k, k_bias, v, v_bias.

        let wo = l
            .get_tensor("attn_output.weight")
            .context("Failed to load attn_output.weight in from_json")?
            .try_map_tensor(|t| {
                t.reshaped(
                    vec![
                        structure.num_heads,
                        structure.head_size,
                        structure.embedding_size,
                    ]
                    .into(),
                )
            })?;
        let o_bias = l
            .get_tensor("attn_output.bias")
            .context("Failed to load attn_output.bias in from_json")?;

        // Shape check for attn_output.weight: [hidden_size, hidden_size] for GPT-2
        // Python script exports it as [out_features, in_features]
        // For c_proj (attn_output), out_features = hidden_size, in_features = hidden_size
        ensure!(
            wo.shape().as_ref()
                == &[
                    structure.num_heads,
                    structure.head_size,
                    structure.embedding_size
                ],
            "Attention output weight tensor shape mismatch in from_json. Expected [{}, {}, {}], got {:?}",
            structure.num_heads,
            structure.head_size,
            structure.embedding_size,
            wo.shape()
        );
        ensure!(
            o_bias.shape().as_ref() == &[hidden_size],
            "Attention output bias tensor shape mismatch in from_json. Expected [{}], got {:?}",
            hidden_size,
            o_bias.shape()
        );

        Ok(Self {
            max_context_length: structure.context_length,
            num_heads: structure.num_heads,
            head_dim: structure.head_size,
            wq,
            q_bias,
            wk,
            k_bias,
            wv,
            v_bias,
            wo,
            o_bias,
        })
    }
}
