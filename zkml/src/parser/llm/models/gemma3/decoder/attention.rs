//! Gemma3 attention mechanism implementation.

use crate::{
    graph::NodeId,
    layers::{
        Layer,
        transformer::{
            attention_mask::AttentionSpan,
            normalisation::rmsnorm::RMSNorm,
            positional::{Positional, RopeLayout},
        },
    },
    model::Model,
    parser::{
        Load,
        llm::{
            config::{AttentionHeadType, LLMStructure, PositionalConfig, RopeConfig},
            gguf::FileTensorLoader as GGUFLoader,
            transformer::attention_layer::{AttentionCacheConfig, AttentionMechanism},
        },
        safe::{ConfigJSON, FileTensorLoader as SafeLoader},
    },
    tensor::TensorHandle,
};

use super::scale_norm;

use anyhow::{Context, Result, anyhow, bail};

#[derive(Clone, Debug)]
pub struct Gemma3Attention {
    max_context_length: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    wq: TensorHandle<f32>,
    wk: TensorHandle<f32>,
    wv: TensorHandle<f32>,
    wo: TensorHandle<f32>,
    span: AttentionSpan,
    q_norm: Option<RMSNorm<f32>>,
    k_norm: Option<RMSNorm<f32>>,
    rope: Positional<f32>,
}

impl AttentionMechanism for Gemma3Attention {
    fn attention_span(&self) -> AttentionSpan {
        self.span
    }

    fn total_heads(&self) -> usize {
        self.num_heads
    }

    fn kv_total_heads(&self) -> usize {
        self.num_kv_heads
    }

    fn heads_per_kv_head(&self) -> usize {
        self.num_heads / self.num_kv_heads
    }

    fn max_context_length(&self) -> usize {
        self.max_context_length
    }

    fn head_dim(&self) -> usize {
        self.head_dim
    }

    fn uses_qkv_bias(&self) -> bool {
        false
    }

    fn uses_out_bias(&self) -> bool {
        false
    }

    fn qkv_tensors(&self) -> (Vec<TensorHandle<f32>>, Vec<Option<TensorHandle<f32>>>) {
        (
            vec![self.wq.clone(), self.wk.clone(), self.wv.clone()],
            vec![None, None, None],
        )
    }

    fn out_tensors(&self) -> (TensorHandle<f32>, Option<TensorHandle<f32>>) {
        (self.wo.clone(), None)
    }

    fn insert_custom_logic(
        self,
        model: &mut Model<f32>,
        qkv_einsum_id: NodeId,
        query_key_id: NodeId,
    ) -> Result<()> {
        // First we add the caching to the QKV einsum
        let AttentionCacheConfig {
            kv_rank,
            caching_dim,
        } = self.caching_dim();

        let qkv_einsum = model.graph_mut().node_mut(qkv_einsum_id).ok_or(anyhow!("Could not insert Gemma3Attention, QKV EinSum with ID({qkv_einsum_id}) has not been inserted"))?;

        if let Some(Layer::EinSum(einsum)) = qkv_einsum.as_inner_mut() {
            einsum.with_caches(vec![None, None, Some(caching_dim)])?;
        } else {
            bail!("QKV Einsum Node is not an EinSum layer");
        }

        // If we have a QK norm we add it here
        let (q_rope_id, k_rope_id) =
            if let (Some(q_norm), Some(k_norm)) = (self.q_norm, self.k_norm) {
                let q_norm_id = model.graph_mut().add_inner(Layer::RMSNorm(q_norm))?;
                let k_norm_id = model.graph_mut().add_inner(Layer::RMSNorm(k_norm))?;
                model.add_edge(qkv_einsum_id, q_norm_id, vec![(0, 0)])?;
                model.add_edge(qkv_einsum_id, k_norm_id, vec![(1, 0)])?;

                let q_rope_id = model
                    .add_consecutive_layer(Layer::Positional(self.rope.clone()), Some(q_norm_id))?;
                // We have to call `with_cache()` here to ensure that the positional cache for K is not shared with Q
                let k_rope_id = model.add_consecutive_layer(
                    Layer::Positional(
                        self.rope
                            .with_cache()
                            .with_rope_cache(kv_rank, caching_dim)?,
                    ),
                    Some(k_norm_id),
                )?;

                (q_rope_id, k_rope_id)
            } else {
                let q_rope_id = model
                    .graph_mut()
                    .add_inner(Layer::Positional(self.rope.clone()))?;
                // We have to call `with_cache()` here to ensure that the positional cache for K is not shared with Q
                let k_rope_id = model.graph_mut().add_inner(Layer::Positional(
                    self.rope
                        .with_cache()
                        .with_rope_cache(kv_rank, caching_dim)?,
                ))?;

                model.add_edge(qkv_einsum_id, q_rope_id, vec![(0, 0)])?;
                model.add_edge(qkv_einsum_id, k_rope_id, vec![(1, 0)])?;

                (q_rope_id, k_rope_id)
            };
        model.add_edge(q_rope_id, query_key_id, vec![(0, 0)])?;
        model.add_edge(k_rope_id, query_key_id, vec![(0, 1)])?;
        Ok(())
    }
}

impl Load<GGUFLoader> for Gemma3Attention {
    type Config = LLMStructure; // (structure, block_number)

    fn from_loader(loader: &GGUFLoader, c: &Self::Config) -> Result<Self>
    where
        Self: Sized,
    {
        let embedding_size = c.generic.embedding_size;
        let max_ctx_length = c.generic.context_length;
        let head_size = c.generic.head_size;
        let num_heads = c.generic.num_heads;
        let AttentionHeadType::GQA(num_groups) = c.attention_config.head else {
            bail!("GQA is expected for Gemma3 models");
        };

        let heads_per_group = num_heads / num_groups;

        let (wq_reshape, wk_wv_reshape, wo_reshape) = if num_groups == 1 {
            (
                vec![embedding_size, num_heads, head_size],
                vec![embedding_size, head_size],
                vec![num_heads, head_size, embedding_size],
            )
        } else {
            (
                vec![embedding_size, num_groups, heads_per_group, head_size],
                vec![embedding_size, num_groups, head_size],
                vec![heads_per_group, num_groups, head_size, embedding_size],
            )
        };
        let wq = loader
            .get_tensor("attn_q.weight")?
            .try_map_tensor(|t| t.transpose()?.reshaped(wq_reshape.clone().into()))?;
        let mut q_norm = RMSNorm::from_gguf(&loader.pp("attn_q_"), &c.generic)?;

        assert_eq!(
            wq.shape().as_ref(),
            wq_reshape.as_slice(),
            "embedding_size {} hidden_size {} num_heads {} heads per group {} head_size {}",
            c.generic.embedding_size,
            c.generic.hidden_size,
            num_heads,
            heads_per_group,
            head_size
        );

        let wk = loader
            .get_tensor("attn_k.weight")?
            .try_map_tensor(|t| t.transpose()?.reshaped(wk_wv_reshape.clone().into()))?;
        let mut k_norm = RMSNorm::from_gguf(&loader.pp("attn_k_"), &c.generic)?;
        assert_eq!(wk.shape().as_ref(), wk_wv_reshape.as_slice());

        let wv = loader
            .get_tensor("attn_v.weight")?
            .try_map_tensor(|t| t.transpose()?.reshaped(wk_wv_reshape.clone().into()))?;
        assert_eq!(wv.shape().as_ref(), wk_wv_reshape.as_slice());

        let wo = loader
            .get_tensor("attn_output.weight")?
            .try_map_tensor(|t| t.transpose()?.reshaped(wo_reshape.clone().into()))?;
        assert_eq!(wo.shape().as_ref(), wo_reshape.as_slice());

        let base_frequency = loader
            .metadata::<f32>("gemma3.rope.freq_base")
            .context("base_freq key not found")?;

        // Retrieve the block number from the loader prefix
        let block_number = loader
            .current_prefix()
            .trim_end_matches('.')
            .rsplit_once('.')
            .and_then(|(_, block_str)| block_str.parse::<usize>().ok())
            .ok_or(anyhow!(
                "Failed to determine block number from loader prefix"
            ))?;
        // Get the attention span type for this block
        let span = c.attention_config.span[block_number];
        let positional_config = match span {
            AttentionSpan::Full => PositionalConfig::Rope(RopeConfig {
                base_frequency,
                max_seq_length: max_ctx_length,
                layout: RopeLayout::RotateHalf,
            }),
            AttentionSpan::Local(_) => PositionalConfig::Rope(RopeConfig {
                // Here we divide by 10.0f32 as Gemma3 uses a different scaling for local attention
                // but it is not saved in the GGUF file.
                base_frequency: base_frequency / 10.0f32,
                max_seq_length: max_ctx_length,
                layout: RopeLayout::RotateHalf,
            }),
        };
        // Scale the RMSNorm alphas because Gemma3 expects that we add 1.0 to each element.
        scale_norm(&mut q_norm);
        scale_norm(&mut k_norm);

        Ok(Self {
            max_context_length: c.generic.context_length,
            num_heads,
            num_kv_heads: num_groups,
            head_dim: head_size,
            wq: wq.into(),
            wk: wk.into(),
            wv: wv.into(),
            wo: wo.into(),
            span,
            q_norm: Some(q_norm),
            k_norm: Some(k_norm),
            rope: Positional::from_gguf(loader, c, &positional_config)?,
        })
    }
}

impl Load<SafeLoader> for Gemma3Attention {
    type Config = (LLMStructure, ConfigJSON);

    fn from_loader(loader: &SafeLoader, (structure, config): &Self::Config) -> anyhow::Result<Self>
    where
        Self: Sized,
    {
        let embedding_size = structure.generic.embedding_size;
        let max_ctx_length = structure.generic.context_length;
        let head_size = structure.generic.head_size;
        let num_heads = structure.generic.num_heads;
        let AttentionHeadType::GQA(num_groups) = structure.attention_config.head else {
            bail!("GQA is expected for Gemma3 models");
        };

        let heads_per_group = num_heads / num_groups;

        let (wq_reshape, wk_wv_reshape, wo_reshape) = if num_groups == 1 {
            (
                vec![embedding_size, num_heads, head_size],
                vec![embedding_size, head_size],
                vec![num_heads, head_size, embedding_size],
            )
        } else {
            (
                vec![embedding_size, num_groups, heads_per_group, head_size],
                vec![embedding_size, num_groups, head_size],
                vec![heads_per_group, num_groups, head_size, embedding_size],
            )
        };
        let wq = loader
            .get_tensor("self_attn.q_proj.weight")?
            .try_map_tensor(|t| t.transpose()?.reshaped(wq_reshape.clone().into()))?;
        let mut q_norm = RMSNorm::from_safe(&loader.pp("self_attn.q_norm."), &structure.generic)?;

        assert_eq!(
            wq.shape().as_ref(),
            wq_reshape.as_slice(),
            "embedding_size {} hidden_size {} num_heads {} heads per group {} head_size {}",
            structure.generic.embedding_size,
            structure.generic.hidden_size,
            num_heads,
            heads_per_group,
            head_size
        );

        let wk = loader
            .get_tensor("self_attn.k_proj.weight")?
            .try_map_tensor(|t| t.transpose()?.reshaped(wk_wv_reshape.clone().into()))?;
        let mut k_norm = RMSNorm::from_safe(&loader.pp("self_attn.k_norm."), &structure.generic)?;
        assert_eq!(wk.shape().as_ref(), wk_wv_reshape.as_slice());

        assert_eq!(k_norm.normalisation_dim_size(), structure.generic.head_size);

        let wv = loader
            .get_tensor("self_attn.v_proj.weight")?
            .try_map_tensor(|t| t.transpose()?.reshaped(wk_wv_reshape.clone().into()))?;
        assert_eq!(wv.shape().as_ref(), wk_wv_reshape.as_slice());

        let wo = loader
            .get_tensor("self_attn.o_proj.weight")?
            .try_map_tensor(|t| t.transpose()?.reshaped(wo_reshape.clone().into()))?;
        assert_eq!(wo.shape().as_ref(), wo_reshape.as_slice());

        // We retrieve the block number from the loader prefix
        let block_number = loader
            .current_prefix()
            .trim_end_matches('.')
            .rsplit_once('.')
            .and_then(|(_, block_str)| block_str.parse::<usize>().ok())
            .ok_or(anyhow!(
                "Failed to determine block number from loader prefix"
            ))?;

        // Get the attention span type for this block
        let span = structure.attention_config.span[block_number];
        let positional_config = match span {
            AttentionSpan::Full => PositionalConfig::Rope(RopeConfig {
                base_frequency: config
                    .get::<f32, _>("rope_theta")
                    .context("full_freq not found")?,
                max_seq_length: max_ctx_length,
                layout: RopeLayout::RotateHalf,
            }),
            AttentionSpan::Local(_) => PositionalConfig::Rope(RopeConfig {
                base_frequency: config
                    .get::<f32, _>("rope_local_base_freq")
                    .context("local_local_freq not found")?,
                max_seq_length: max_ctx_length,
                layout: RopeLayout::RotateHalf,
            }),
        };

        // Scale the RMSNorm alphas because Gemma3 expects that we add 1.0 to each element.
        scale_norm(&mut q_norm);
        scale_norm(&mut k_norm);

        Ok(Self {
            max_context_length: structure.generic.context_length,
            num_heads,
            num_kv_heads: num_groups,
            head_dim: head_size,
            wq: wq.into(),
            wk: wk.into(),
            wv: wv.into(),
            wo: wo.into(),
            span,
            q_norm: Some(q_norm),
            k_norm: Some(k_norm),
            rope: Positional::from_safetensors_loader(loader, structure, &positional_config)?,
        })
    }
}
