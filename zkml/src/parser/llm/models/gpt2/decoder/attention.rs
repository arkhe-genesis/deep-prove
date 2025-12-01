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
            ConfigJSON, LLMConfig, SafeLoader,
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

        let qkv_einsum = model.graph_mut().node_mut(qkv_einsum_id).ok_or(anyhow::anyhow!("Could not insert GPT2Attention, QKV EinSum with ID({qkv_einsum_id}) has not been inserted"))?;
        if let Some(Layer::EinSum(einsum)) = qkv_einsum.as_inner_mut() {
            einsum.with_caches(vec![None, Some(caching_dim), Some(caching_dim)])?;
        } else {
            anyhow::bail!("QKV Einsum Node is not an EinSum layer");
        }

        model.add_edge(qkv_einsum_id, query_key_id, vec![(0, 0), (1, 1)])?;
        Ok(())
    }
}

impl Load<SafeLoader> for GPT2Attention {
    type Config = (LLMStructure, ConfigJSON);

    /// Load GPT-2 attention weights from HuggingFace SafeTensors format.
    ///
    /// # Reference Implementation
    /// This loader matches the weight layout from HuggingFace Transformers:
    /// https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
    ///
    /// # Weight Format
    /// HuggingFace uses `Conv1D` layers which store weights as (in_features, out_features).
    /// Conv1D is essentially a Linear layer with transposed weight storage - it computes
    /// `output = input @ weight` directly (no transpose in forward pass).
    ///
    /// The `c_attn` layer fuses Q, K, V projections:
    /// ```python
    /// self.c_attn = Conv1D(3 * self.embed_dim, self.embed_dim)
    /// ```
    /// Weight shape: (embed_dim, 3*embed_dim) where columns are [Q | K | V]
    ///
    /// After forward pass, the output is split along the last dimension:
    /// ```python
    /// query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)
    /// ```
    fn from_loader(loader: &SafeLoader, (structure, _config): &Self::Config) -> anyhow::Result<Self>
    where
        Self: Sized,
    {
        let embedding_size = structure.generic.embedding_size;
        let hidden_size = structure.generic.hidden_size;
        ensure!(
            embedding_size == hidden_size,
            "embedding_size must be equal to hidden_size"
        );

        // Load fused QKV weight: shape [hidden_size, 3*hidden_size]
        // Matches HuggingFace Conv1D(3*embed_dim, embed_dim)
        let fused_qkv_weight = loader.get_tensor("attn.c_attn.weight")?;
        ensure!(
            fused_qkv_weight.shape().as_ref() == &[hidden_size, 3 * hidden_size],
            "attn.c_attn.weight must have shape [hidden_size, 3*hidden_size], got {:?}",
            fused_qkv_weight.shape()
        );

        // Split fused weight into Q, K, V matrices
        // The Conv1D output columns are organized as [Q_cols | K_cols | V_cols]
        // where each section has `hidden_size` columns.
        //
        // For row-major storage, we iterate through rows and extract the column ranges:
        // - Q columns: [0, hidden_size)
        // - K columns: [hidden_size, 2*hidden_size)
        // - V columns: [2*hidden_size, 3*hidden_size)
        let fused_tensor = fused_qkv_weight.tensor();
        let fused_data = fused_tensor.get_data();
        let mut q_data = Vec::with_capacity(hidden_size * hidden_size);
        let mut k_data = Vec::with_capacity(hidden_size * hidden_size);
        let mut v_data = Vec::with_capacity(hidden_size * hidden_size);

        for row in 0..hidden_size {
            let row_start = row * (3 * hidden_size);
            // Extract Q columns
            q_data.extend_from_slice(&fused_data[row_start..row_start + hidden_size]);
            // Extract K columns
            k_data.extend_from_slice(
                &fused_data[row_start + hidden_size..row_start + 2 * hidden_size],
            );
            // Extract V columns
            v_data.extend_from_slice(
                &fused_data[row_start + 2 * hidden_size..row_start + 3 * hidden_size],
            );
        }

        let wq = KeyedTensor::new(
            format!("{:?}.q", fused_qkv_weight.commitment_id()),
            Tensor::new(vec![embedding_size, hidden_size].into(), q_data)?.reshaped(
                vec![
                    embedding_size,
                    structure.generic.num_heads,
                    structure.generic.head_size,
                ]
                .into(),
            )?,
        );
        let wk = KeyedTensor::new(
            format!("{:?}.k", fused_qkv_weight.commitment_id()),
            Tensor::new(vec![embedding_size, hidden_size].into(), k_data)?.reshaped(
                vec![
                    embedding_size,
                    structure.generic.num_heads,
                    structure.generic.head_size,
                ]
                .into(),
            )?,
        );
        let wv = KeyedTensor::new(
            format!("{:?}.v", fused_qkv_weight.commitment_id()),
            Tensor::new(vec![embedding_size, hidden_size].into(), v_data)?.reshaped(
                vec![
                    embedding_size,
                    structure.generic.num_heads,
                    structure.generic.head_size,
                ]
                .into(),
            )?,
        );

        // Load fused QKV bias: shape [3*hidden_size]
        // Organized sequentially as [Q_bias | K_bias | V_bias]
        let fused_qkv_bias = loader.get_tensor("attn.c_attn.bias")?;
        ensure!(
            fused_qkv_bias.shape().as_ref() == &[3 * hidden_size],
            "attn.c_attn.bias must have shape [3*hidden_size], got {:?}",
            fused_qkv_bias.shape()
        );

        let bias_tensor = fused_qkv_bias.tensor();
        let bias_data = bias_tensor.get_data();
        // Split bias into Q, K, V components
        let q_bias = KeyedTensor::new(
            format!("{:?}.q", fused_qkv_bias.commitment_id()),
            Tensor::new(
                vec![structure.generic.num_heads, structure.generic.head_size].into(),
                bias_data[0..hidden_size].to_vec(),
            )?,
        );
        let k_bias = KeyedTensor::new(
            format!("{:?}.k", fused_qkv_bias.commitment_id()),
            Tensor::new(
                vec![structure.generic.num_heads, structure.generic.head_size].into(),
                bias_data[hidden_size..2 * hidden_size].to_vec(),
            )?,
        );
        let v_bias = KeyedTensor::new(
            format!("{:?}.v", fused_qkv_bias.commitment_id()),
            Tensor::new(
                vec![structure.generic.num_heads, structure.generic.head_size].into(),
                bias_data[2 * hidden_size..3 * hidden_size].to_vec(),
            )?,
        );

        // Load output projection: c_proj is Conv1D(embed_dim, embed_dim)
        // Weight shape: [hidden_size, hidden_size]
        // See: https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py#L279
        let wo = loader
            .get_tensor("attn.c_proj.weight")?
            .try_map_tensor(|t| {
                t.reshaped(
                    vec![
                        structure.generic.num_heads,
                        structure.generic.head_size,
                        embedding_size,
                    ]
                    .into(),
                )
            })?;
        let o_bias = loader.get_tensor("attn.c_proj.bias")?;

        ensure!(
            wo.shape().as_ref()
                == &[
                    structure.generic.num_heads,
                    structure.generic.head_size,
                    embedding_size
                ],
            "attn.c_proj.weight must have shape [num_heads, head_size, embedding_size], got {:?}",
            wo.shape()
        );
        ensure!(
            o_bias.shape().as_ref() == &[embedding_size],
            "attn.c_proj.bias must have shape [embedding_size], got {:?}",
            o_bias.shape()
        );

        Ok(Self {
            max_context_length: structure.generic.context_length,
            num_heads: structure.generic.num_heads,
            head_dim: structure.generic.head_size,
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
