use crate::{
    Number, Tensor,
    graph::NodeId,
    layers::{
        Layer, add,
        matrix_mul::MatMul,
        transformer::{
            attention::attention_mask::AttentionSpan, layernorm::LayerNorm, mha::Mha,
            positional::Positional, qkv::QKV, rmsnorm::RMSNorm,
        },
    },
    model::Model,
    parser::{
        gguf,
        json::{self, unfuse_crate_tensors},
        llm::{
            FeedForward, LLMConfig,
            config::{AttentionHeadType, LLMStructure, PositionalConfig},
        },
        safe,
    },
    tensor::KeyedTensor,
};
use anyhow::{Context, bail, ensure};
use tracing::trace;

use serde::{Deserialize, Serialize};

#[derive(Copy, Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum NormType {
    LayerNorm,
    RMSNorm,
}
#[derive(Debug, Clone)]
pub enum Norm<N: Number> {
    LayerNorm(LayerNorm<N>),
    RMSNorm(RMSNorm<N>),
}

#[derive(Debug, Clone)]
pub struct Attention<N: Number> {
    pub pre_norm: Norm<N>,
    pub q: KeyedTensor<N>,
    pub q_bias: Option<KeyedTensor<N>>,
    pub q_norm: Option<Norm<N>>,

    pub k: KeyedTensor<N>,
    pub k_bias: Option<KeyedTensor<N>>,
    pub k_norm: Option<Norm<N>>,

    pub v: KeyedTensor<N>,
    pub v_bias: Option<KeyedTensor<N>>,
    pub out: KeyedTensor<N>,
    pub out_bias: Option<KeyedTensor<N>>,
    pub post_norm: Option<Norm<N>>,
    pub pre_ffn_norm: Norm<N>,
    pub feedforward: FeedForward<N>,
    pub post_ffn_norm: Option<Norm<N>>,
    pub span: AttentionSpan,
}

impl Attention<f32> {
    pub fn write_to_model(
        self,
        model: &mut Model<f32>,
        input_node_id: Option<NodeId>,
        c: &LLMStructure,
        positional: Option<Positional<f32>>,
    ) -> anyhow::Result<NodeId> {
        let num_groups = match c.attention_config.head {
            AttentionHeadType::MHA => c.generic.num_heads,
            AttentionHeadType::GQA(num_groups) => num_groups,
        };
        let qkv = QKV::new(
            self.q,
            self.q_bias,
            self.k,
            self.k_bias,
            self.v,
            self.v_bias,
            c.generic.num_heads,
            num_groups,
        )?;
        // TODO : change for GQA if needed by also giving the Q and K norms and the ROPE table
        let mha = Mha::new(
            c.generic.context_length,
            c.generic.num_heads,
            c.generic.head_size,
        )?
        .with_attention_span(self.span)?;
        let out = MatMul::new_constant(self.out, self.out_bias)?;
        // input is [seq_len, emb_size]
        let mut last_node_id =
            model.add_consecutive_layer(self.pre_norm.to_layer(), input_node_id)?;
        // shape goes to [seq_len, hidden_size] for each, Q K and V
        last_node_id = model.add_consecutive_layer(Layer::QKV(qkv), Some(last_node_id))?;
        // QKV outputs three tensors, but we may need to apply a norm on the second and third one
        let (mut q_id, mut q_port): (NodeId, usize) = (last_node_id, 0);
        let (mut k_id, mut k_port): (NodeId, usize) = (last_node_id, 1);
        let (v_id, v_port): (NodeId, usize) = (last_node_id, 2);

        let mha_id = model.graph.add_inner(Layer::Mha(mha))?;
        if let Some(q_norm) = self.q_norm {
            (q_id, q_port) = {
                let q_norm_id = model.graph.add_inner(q_norm.to_layer())?;
                model.add_edge(q_id, q_norm_id, (q_port, 0))?;
                (q_norm_id, 0)
            };
        }
        if let Some(k_norm) = self.k_norm {
            (k_id, k_port) = {
                let k_norm_id = model.graph.add_inner(k_norm.to_layer())?;
                model.add_edge(k_id, k_norm_id, (k_port, 0))?;
                (k_norm_id, 0)
            };
        }
        if let PositionalConfig::Rope(_) = &c.positional_config {
            let Some(rope) = positional else {
                bail!("Positional encoding is expected after QK");
            };
            (q_id, q_port) = {
                // we need to build the cache for the Q tensor
                let rope_id =
                    model
                        .graph
                        .add_inner(Layer::Positional(Positional::new_from_variant(
                            rope.variant.clone(),
                        )))?;
                model.add_edge(q_id, rope_id, (q_port, 0))?;
                (rope_id, 0)
            };
            (k_id, k_port) = {
                // vector k doesn't need a cache since it's always of full sequence length
                let rope_id = model.graph.add_inner(Layer::Positional(
                    Positional::new_from_variant(rope.variant.clone()).with_no_cache(),
                ))?;
                model.add_edge(k_id, rope_id, (k_port, 0))?;
                (rope_id, 0)
            };
            // in Gemma3, there are distinct edges between QKV and MHA, because Q and K are suffixed with
            // norm and rope.
            // * first one is [num_heads, seq_len] (Q @ K^T - all heads concatenated)
            // * second one is [num_heads, seq_len, head_dim] (V)
            model.add_edge(q_id, mha_id, (q_port, 0))?;
            model.add_edge(k_id, mha_id, (k_port, 1))?;
            model.add_edge(v_id, mha_id, (v_port, 2))?;
        } else {
            // in GPT2, there is only one edge between QKV and MHA, and QKV outputs three tensors
            model.add_edge(q_id, mha_id, vec![(q_port, 0), (k_port, 1), (v_port, 2)])?;
        }

        last_node_id = model.add_consecutive_layer(Layer::MatMul(out), Some(mha_id))?;
        last_node_id = match self.post_norm {
            Some(norm) => model.add_consecutive_layer(norm.to_layer(), Some(last_node_id))?,
            None => last_node_id,
        };
        last_node_id = {
            let add_id = model.graph.add_inner(Layer::Add(add::Add::new()))?;
            match input_node_id {
                Some(id) => model.add_edge(id, add_id, (0, 0))?,
                // in this case, this is the input to the model
                None => unreachable!("never used"),
            };
            model.add_edge(last_node_id, add_id, (0, 1))?;
            add_id
        };

        let pre_ffn_residual_id = last_node_id;
        last_node_id =
            model.add_consecutive_layer(self.pre_ffn_norm.to_layer(), Some(last_node_id))?;
        last_node_id = self.feedforward.write_to_model(c, model, last_node_id)?;
        last_node_id = match self.post_ffn_norm {
            Some(norm) => model.add_consecutive_layer(norm.to_layer(), Some(last_node_id))?,
            None => last_node_id,
        };
        last_node_id = {
            let add_id = model.graph.add_inner(Layer::Add(add::Add::new()))?;
            model.add_edge(pre_ffn_residual_id, add_id, (0, 0))?;
            model.add_edge(last_node_id, add_id, (0, 1))?;
            add_id
        };
        Ok(last_node_id)
    }

    pub fn with_span(self, span: AttentionSpan) -> Self {
        Self { span, ..self }
    }

    pub fn from_json(l: &json::FileTensorLoader, c: &LLMConfig) -> anyhow::Result<Self> {
        let norm = LayerNorm::from_json(&l.pp("attn_"), c)
            .context("Failed to load LayerNorm for attention in from_json")?;

        let fused_qkv_weight = l
            .get_tensor("attn_qkv.weight")
            .context("Failed to load attn_qkv.weight in from_json")?;
        let fused_qkv_bias = l
            .get_tensor("attn_qkv.bias")
            .context("Failed to load attn_qkv.bias in from_json")?;

        let hidden_size = c.hidden_size; // embedding_dim for GPT-2

        // Unfuse weights:
        // Expected shape of fused_qkv_weight is [3 * hidden_size, hidden_size] after python script transpose.
        // Each individual q, k, v weight matrix should be [hidden_size, hidden_size].
        // So, each chunk has hidden_size * hidden_size elements.
        let weight_chunk_elements = hidden_size * hidden_size;
        let mut unfused_weights_data =
            unfuse_crate_tensors(fused_qkv_weight.tensor(), weight_chunk_elements, 3)
                .context("Failed to unfuse QKV weights in from_json")?;

        let q_weight = KeyedTensor::new(
            format!("{}.q", fused_qkv_weight.key),
            Tensor::new(
                vec![c.embedding_size, hidden_size].into(),
                unfused_weights_data.remove(0),
            ),
        );
        let k_weight = KeyedTensor::new(
            format!("{}.k", fused_qkv_weight.key),
            Tensor::new(
                vec![c.embedding_size, hidden_size].into(),
                unfused_weights_data.remove(0),
            ),
        );
        let v_weight = KeyedTensor::new(
            format!("{}.v", fused_qkv_weight.key),
            Tensor::new(
                vec![c.embedding_size, hidden_size].into(),
                unfused_weights_data.remove(0),
            ),
        );
        trace!("fused qkv: {fused_qkv_weight:?}");
        trace!("qkv full tensor {unfused_weights_data:?}");
        trace!("q_weight {:?}", q_weight.get_data());

        // Unfuse biases:
        // Expected shape of fused_qkv_bias is [3 * hidden_size].
        // Each individual q, k, v bias vector should be [hidden_size].
        // So, each chunk has hidden_size elements.
        let bias_chunk_elements = hidden_size;
        let fused_qvk_bias_key = fused_qkv_bias.key.clone();
        let mut unfused_biases_data =
            unfuse_crate_tensors(fused_qkv_bias.into_tensor(), bias_chunk_elements, 3)
                .context("Failed to unfuse QKV biases in from_json")?;

        let q_bias_vec = KeyedTensor::new(
            format!("{fused_qvk_bias_key}.q"),
            Tensor::new(vec![hidden_size].into(), unfused_biases_data.remove(0)),
        );
        let k_bias_vec = KeyedTensor::new(
            format!("{fused_qvk_bias_key}.k"),
            Tensor::new(vec![hidden_size].into(), unfused_biases_data.remove(0)),
        );
        let v_bias_vec = KeyedTensor::new(
            format!("{fused_qvk_bias_key}.v"),
            Tensor::new(vec![hidden_size].into(), unfused_biases_data.remove(0)),
        );

        // These are the individual Q, K, V matrices and biases now.
        // The QKV struct or logic that consumes these will handle them.
        // For now, let's assume Attention struct stores these directly if QKV is not used here.
        // Or, construct the QKV layer if that's the design.
        // The original struct for Attention<f32> directly stores q, q_bias, k, k_bias, v, v_bias.

        let out = l
            .get_tensor("attn_output.weight")
            .context("Failed to load attn_output.weight in from_json")?;
        let out_bias = l
            .get_tensor("attn_output.bias")
            .context("Failed to load attn_output.bias in from_json")?;

        // Shape check for attn_output.weight: [hidden_size, hidden_size] for GPT-2
        // Python script exports it as [out_features, in_features]
        // For c_proj (attn_output), out_features = hidden_size, in_features = hidden_size
        ensure!(
            out.shape().as_ref() == &[hidden_size, hidden_size],
            "Attention output weight tensor shape mismatch in from_json. Expected [{}, {}], got {:?}",
            hidden_size,
            hidden_size,
            out.shape()
        );
        ensure!(
            out_bias.shape().as_ref() == &[hidden_size],
            "Attention output bias tensor shape mismatch in from_json. Expected [{}], got {:?}",
            hidden_size,
            out_bias.shape()
        );

        let pre_ffn_norm = LayerNorm::from_json(&l.pp("ffn_"), c)?;
        let feedforward =
            FeedForward::from_json(l, c).context("Failed to load FeedForward in from_json")?;

        Ok(Self {
            pre_norm: Norm::LayerNorm(norm),
            q: q_weight,
            q_bias: Some(q_bias_vec),
            q_norm: None,
            k: k_weight,
            k_bias: Some(k_bias_vec),
            k_norm: None,
            v: v_weight,
            v_bias: Some(v_bias_vec),
            post_norm: None,
            out,
            out_bias: Some(out_bias),
            pre_ffn_norm: Norm::LayerNorm(pre_ffn_norm),
            feedforward,
            post_ffn_norm: None,
            span: AttentionSpan::Full,
        })
    }
}

impl<N: Number> Norm<N> {
    pub fn to_layer(self) -> Layer<N> {
        match self {
            Norm::LayerNorm(layer) => Layer::LayerNorm(layer),
            Norm::RMSNorm(layer) => Layer::RMSNorm(layer),
        }
    }
}

impl NormType {
    pub fn from_gguf(
        &self,
        loader: &gguf::FileTensorLoader,
        c: &LLMConfig,
        stack: bool,
    ) -> anyhow::Result<Norm<f32>> {
        Ok(match self {
            NormType::LayerNorm => Norm::LayerNorm(LayerNorm::from_gguf(loader, c)?),
            NormType::RMSNorm => Norm::RMSNorm(RMSNorm::from_gguf(loader, c, stack)?),
        })
    }
    pub fn from_safetensors(
        &self,
        loader: &safe::FileTensorLoader,
        config: &safe::ConfigJSON,
        c: &LLMConfig,
        stack: bool,
    ) -> anyhow::Result<Norm<f32>> {
        Ok(match self {
            NormType::LayerNorm => Norm::LayerNorm(LayerNorm::from_safetensors(loader, config, c)?),
            NormType::RMSNorm => {
                Norm::RMSNorm(RMSNorm::from_safetensors(loader, config, c, stack)?)
            }
        })
    }
}
pub(crate) fn expand<N: Number>(t: Tensor<N>, num_heads: usize) -> Tensor<N> {
    let (it, _) = t.slice_on_dim(0);
    let data = it
        .flat_map(|t| std::iter::repeat_n(t, num_heads).flatten())
        .cloned()
        .collect::<Vec<_>>();
    let mut shape = t.shape().clone();
    let new_dim = shape.dim(-1) * num_heads;
    shape.set_dim(-1, new_dim);
    Tensor::new(shape, data)
}
