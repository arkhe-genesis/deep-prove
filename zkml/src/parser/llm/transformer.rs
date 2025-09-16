use crate::{
    Number,
    layers::{
        activation::{Activation, GELU, Relu},
        add,
        matrix_mul::MatMul,
        provable::{Edge, Node},
        transformer::{layernorm::LayerNorm, mha::Mha, qkv::QKV, rmsnorm::RMSNorm},
    },
    parser::{
        gguf::FileTensorLoader,
        json::unfuse_crate_tensors,
        llm::{AttentionType, LLMVariant, NodeId},
    },
};
use anyhow::{Context, bail, ensure};
use candle_core::Device;
use tracing::trace;

use crate::{
    Tensor,
    layers::Layer,
    model::Model,
    parser::{gguf::unfuse_tensors, json, llm::LLMConfig},
};

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
    pub q: Tensor<N>,
    pub q_bias: Option<Tensor<N>>,
    pub q_norm: Option<Norm<N>>,

    pub k: Tensor<N>,
    pub k_bias: Option<Tensor<N>>,
    pub k_norm: Option<Norm<N>>,

    pub v: Tensor<N>,
    pub v_bias: Option<Tensor<N>>,
    pub out: Tensor<N>,
    pub out_bias: Option<Tensor<N>>,
    pub post_norm: Option<Norm<N>>,
    pub feedforward: FeedForward<N>,
    pub post_ffw_norm: Option<Norm<N>>,
}

#[derive(Debug, Clone)]
pub struct FeedForward<N: Number> {
    pub pre_norm: Norm<N>,
    pub up: Tensor<N>,
    pub up_bias: Option<Tensor<N>>,
    pub down: Tensor<N>,
    pub down_bias: Option<Tensor<N>>,
}

impl Attention<f32> {
    pub fn write_to_model(
        self,
        model: &mut Model<f32>,
        input_node_id: Option<NodeId>,
        c: &LLMConfig,
    ) -> anyhow::Result<NodeId> {
        let num_groups = match c.attention_type {
            AttentionType::MHA => c.num_heads,
            AttentionType::GQA(num_groups) => num_groups,
        };
        let qkv = QKV::new(
            self.q,
            self.q_bias,
            self.k,
            self.k_bias,
            self.v,
            self.v_bias,
            c.num_heads,
            num_groups,
        )?;
        // TODO : change for GQA if needed by also giving the Q and K norms and the ROPE table
        let mha = Mha::new(c.context_length, c.num_heads, c.head_size)?;
        let out = MatMul::new_constant(self.out, self.out_bias)?;
        // input is [seq_len, emb_size]
        let mut last_node_id =
            model.add_consecutive_layer(self.pre_norm.to_layer(), input_node_id)?;
        // shape goes to [seq_len, hidden_size] for each, Q K and V
        last_node_id = model.add_consecutive_layer(Layer::QKV(qkv), Some(last_node_id))?;
        // then this output two tensors:
        // * first one is [num_heads, seq_len] (Q @ K^T - all heads concatenated)
        // * second one is [num_heads, seq_len, head_dim] (V)
        // TODO : change for GQA
        let mha_id = model.add_consecutive_layer(Layer::Mha(mha), Some(last_node_id))?;
        last_node_id = model.add_consecutive_layer(Layer::MatMul(out), Some(mha_id))?;
        last_node_id = match self.post_norm {
            Some(norm) => model.add_consecutive_layer(norm.to_layer(), Some(last_node_id))?,
            None => last_node_id,
        };
        last_node_id = model.add_node(Node::new(
            vec![
                Edge {
                    // here we dont know if the input is the input to the model or an input coming from previous layers
                    // so if there is no layer before this attention, we take the input of the model
                    node: input_node_id,
                    index: 0,
                },
                Edge::new(last_node_id, 0),
            ],
            Layer::Add(add::Add::new()),
        ))?;
        last_node_id = self.feedforward.write_to_model(c, model, last_node_id)?;
        last_node_id = match self.post_ffw_norm {
            Some(norm) => model.add_consecutive_layer(norm.to_layer(), Some(last_node_id))?,
            None => last_node_id,
        };
        Ok(last_node_id)
    }
    // Replaces from_var_builder and from_tensor_loader
    // 'loader' is expected to be the block-level loader (e.g., scoped to "blk.N.")
    pub fn from_loader(loader: &FileTensorLoader, c: &LLMConfig) -> anyhow::Result<Self> {
        let embedding_size = c.embedding_size;
        let hidden_size = c.hidden_size;
        ensure!(
            embedding_size == hidden_size,
            "embedding_size must be equal to hidden_size"
        );
        match c.variant {
            LLMVariant::GPT2 => Self::from_loader_gpt2(loader, c),
            LLMVariant::Gemma3 => Self::from_loader_gemma3(loader, c),
        }
    }

    fn from_loader_gemma3(loader: &FileTensorLoader, c: &LLMConfig) -> anyhow::Result<Self> {
        let hidden_size = c.hidden_size;
        let head_size = c.head_size;
        let num_heads = c.num_heads;
        let num_groups = c.num_groups();

        let pre_norm = RMSNorm::from_loader(&loader.pp("attn_"), c)?;
        assert_eq!(
            pre_norm.alpha.as_ref().unwrap().shape().as_ref(),
            &[c.embedding_size]
        );

        let q_tensor = loader.get_tensor("attn_q.weight")?.transpose();
        let q_norm = RMSNorm::from_loader(&loader.pp("attn_q_"), c)?;
        assert_eq!(
            q_tensor.shape().as_ref(),
            &[c.hidden_size, num_heads * head_size],
            "embedding_size {} hidden_size {} num_heads {} head_size {}",
            c.embedding_size,
            c.hidden_size,
            num_heads,
            head_size
        );
        assert_eq!(
            q_norm.alpha.as_ref().unwrap().shape().as_ref(),
            &[c.head_size]
        );

        let k_tensor = loader.get_tensor("attn_k.weight")?.transpose();
        let k_norm = RMSNorm::from_loader(&loader.pp("attn_k_"), c)?;
        assert_eq!(
            k_tensor.shape().as_ref(),
            &[hidden_size, num_groups * head_size]
        );
        // head_dim = num_groups * head_size
        assert_eq!(
            k_norm.alpha.as_ref().unwrap().shape().as_ref(),
            &[c.head_size]
        );

        let v_tensor = loader.get_tensor("attn_v.weight")?.transpose();
        assert_eq!(
            v_tensor.shape().as_ref(),
            &[hidden_size, num_groups * head_size]
        );

        let out = loader.get_tensor("attn_output.weight")?.transpose();
        assert_eq!(out.shape().as_ref(), &[num_heads * head_size, hidden_size]);

        let post_attn_norm = RMSNorm::from_loader(&loader.pp("post_attention_"), c)?;
        assert_eq!(
            post_attn_norm.alpha.as_ref().unwrap().shape().as_ref(),
            &[c.hidden_size]
        );

        let ff = FeedForward::from_loader(loader, c)?;
        let scope_loader = loader.pp("post_ffw_");
        let post_ffw_norm = RMSNorm::from_loader(&scope_loader, c)?;
        assert_eq!(
            post_ffw_norm.alpha.as_ref().unwrap().shape().as_ref(),
            &[c.hidden_size]
        );

        Ok(Self {
            pre_norm: Norm::RMSNorm(pre_norm),
            q: q_tensor,
            q_bias: None,
            q_norm: Some(Norm::RMSNorm(q_norm)),
            k: k_tensor,
            k_bias: None,
            k_norm: Some(Norm::RMSNorm(k_norm)),
            v: v_tensor,
            v_bias: None,
            out,
            out_bias: None,
            post_norm: None,
            feedforward: ff,
            post_ffw_norm: Some(Norm::RMSNorm(post_ffw_norm)),
        })
    }

    fn from_loader_gpt2(loader: &FileTensorLoader, c: &LLMConfig) -> anyhow::Result<Self> {
        let embedding_size = c.embedding_size;
        let hidden_size = c.hidden_size;
        ensure!(
            embedding_size == hidden_size,
            "embedding_size must be equal to hidden_size"
        );
        let qkv_weight_qtensor = loader.get_qtensor("attn_qkv.weight")?;
        let qkv_weight_candle = qkv_weight_qtensor.dequantize(&Device::Cpu)?;
        let mut unfused_weights =
            unfuse_tensors(qkv_weight_candle, embedding_size * embedding_size)?;
        ensure!(unfused_weights.len() == 3, "qkv_weight must have 3 chunks");
        let q = crate::Tensor::new(
            vec![embedding_size, hidden_size].into(),
            unfused_weights.remove(0),
        )
        .transpose();
        let k = crate::Tensor::new(
            vec![embedding_size, hidden_size].into(),
            unfused_weights.remove(0),
        )
        .transpose();
        let v = crate::Tensor::new(
            vec![embedding_size, hidden_size].into(),
            unfused_weights.remove(0),
        )
        .transpose();

        let qkv_bias_qtensor = loader.get_qtensor("attn_qkv.bias")?;
        let qkv_bias_candle = qkv_bias_qtensor.dequantize(&Device::Cpu)?;
        let mut unfused_biases = unfuse_tensors(qkv_bias_candle, embedding_size)?;
        ensure!(unfused_biases.len() == 3, "qkv_bias must have 3 chunks");
        let q_bias = crate::Tensor::new(vec![hidden_size].into(), unfused_biases.remove(0));
        let k_bias = crate::Tensor::new(vec![hidden_size].into(), unfused_biases.remove(0));
        let v_bias = crate::Tensor::new(vec![hidden_size].into(), unfused_biases.remove(0));

        let attn_norm_loader = loader.pp("attn_");
        // Use new LayerNorm::from_loader
        let pre_norm = LayerNorm::from_loader(&attn_norm_loader, c)?;

        // attn_output.weight is stored as [out_features, in_features] in GGUF (same as PyTorch)
        // Our MatMul layer expects the right-hand constant to be in the orientation [in_features, out_features],
        // so we transpose it once here after loading.
        let out = loader.get_tensor("attn_output.weight")?.transpose();
        let out_bias = loader.get_tensor("attn_output.bias")?;
        ensure!(
            out.shape().as_ref() == &[embedding_size, embedding_size],
            "out must have shape [hidden_size, hidden_size]"
        );
        ensure!(
            out_bias.shape().as_ref() == &[embedding_size],
            "out_bias must have shape [hidden_size]"
        );

        // Use new FeedForward::from_loader
        let ff = FeedForward::from_loader(loader, c)?;

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
            feedforward: ff,
            post_norm: None,
            post_ffw_norm: None,
        })
    }
    pub fn from_json(l: &json::FileTensorLoader, c: &LLMConfig) -> anyhow::Result<Self> {
        if let LLMVariant::Gemma3 = c.variant {
            bail!("Gemma3 is not supported yet for custom JSON format");
        }
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
            unfuse_crate_tensors(fused_qkv_weight.clone(), weight_chunk_elements, 3)
                .context("Failed to unfuse QKV weights in from_json")?;

        let q_weight = Tensor::new(
            vec![c.embedding_size, hidden_size].into(),
            unfused_weights_data.remove(0),
        );
        let k_weight = Tensor::new(
            vec![c.embedding_size, hidden_size].into(),
            unfused_weights_data.remove(0),
        );
        let v_weight = Tensor::new(
            vec![c.embedding_size, hidden_size].into(),
            unfused_weights_data.remove(0),
        );
        trace!("fused qkv: {fused_qkv_weight:?}");
        trace!("qkv full tensor {unfused_weights_data:?}");
        trace!("q_weight {:?}", q_weight.get_data());

        // Unfuse biases:
        // Expected shape of fused_qkv_bias is [3 * hidden_size].
        // Each individual q, k, v bias vector should be [hidden_size].
        // So, each chunk has hidden_size elements.
        let bias_chunk_elements = hidden_size;
        let mut unfused_biases_data = unfuse_crate_tensors(fused_qkv_bias, bias_chunk_elements, 3)
            .context("Failed to unfuse QKV biases in from_json")?;

        let q_bias_vec = Tensor::new(vec![hidden_size].into(), unfused_biases_data.remove(0));
        let k_bias_vec = Tensor::new(vec![hidden_size].into(), unfused_biases_data.remove(0));
        let v_bias_vec = Tensor::new(vec![hidden_size].into(), unfused_biases_data.remove(0));

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
            feedforward,
            post_ffw_norm: None,
        })
    }
}

impl FeedForward<f32> {
    pub fn write_to_model(
        self,
        config: &LLMConfig,
        model: &mut Model<f32>,
        input_node_id: NodeId,
    ) -> anyhow::Result<NodeId> {
        let layernorm = self.pre_norm.to_layer();
        let up = MatMul::new_constant(self.up, self.up_bias)?;

        let activation = match config.variant {
            LLMVariant::GPT2 => Activation::Gelu(GELU::new()),
            // TODO: change
            LLMVariant::Gemma3 => Activation::Relu(Relu::new()),
        };
        // let down = MatMul::new_constant(self.down, self.down_bias);
        let down = MatMul::new_constant(self.down, self.down_bias)?;
        let add = add::Add::new();
        let last_node_id = model.add_consecutive_layer(layernorm, Some(input_node_id))?;
        let last_node_id = model.add_consecutive_layer(Layer::MatMul(up), Some(last_node_id))?;
        let last_node_id =
            model.add_consecutive_layer(Layer::Activation(activation), Some(last_node_id))?;
        let last_node_id = model.add_consecutive_layer(Layer::MatMul(down), Some(last_node_id))?;
        let last_node_id = model.add_node(Node::new(
            vec![Edge::new(input_node_id, 0), Edge::new(last_node_id, 0)],
            Layer::Add(add),
        ))?;
        Ok(last_node_id)
    }
    // Replaces from_var_builder and from_tensor_loader
    // 'loader' is expected to be the block-level loader (e.g., scoped to "blk.N.")
    pub fn from_loader(loader: &FileTensorLoader, c: &LLMConfig) -> anyhow::Result<Self> {
        // Create a sub-scope for the feed-forward network's LayerNorm
        let ffn_norm_loader = loader.pp("ffn_");

        let pre_norm = match c.variant.norm_type() {
            NormType::RMSNorm => Norm::RMSNorm(RMSNorm::from_loader(&ffn_norm_loader, c)?),
            NormType::LayerNorm => Norm::LayerNorm(LayerNorm::from_loader(&ffn_norm_loader, c)?),
        };

        let up = loader.get_tensor("ffn_up.weight")?.transpose();
        let up_bias = if !c.variant.has_biases() {
            None
        } else {
            Some(loader.get_tensor("ffn_up.bias")?)
        };
        let down = loader.get_tensor("ffn_down.weight")?.transpose();
        let down_bias = if !c.variant.has_biases() {
            None
        } else {
            Some(loader.get_tensor("ffn_down.bias")?)
        };
        ensure!(
            up.shape()[0] == c.hidden_size,
            "up have shape {:?} but in features should be equal to hidden_size: {}",
            up.shape(),
            c.hidden_size
        );
        ensure!(
            down.shape()[1] == c.embedding_size,
            "down have shape {:?} but out features should be equal to embedding_size: {}",
            down.shape(),
            c.embedding_size
        );
        Ok(Self {
            pre_norm,
            up,
            up_bias,
            down,
            down_bias,
        })
    }

    pub fn from_json(l: &json::FileTensorLoader, c: &LLMConfig) -> anyhow::Result<Self> {
        if let LLMVariant::Gemma3 = c.variant {
            bail!("Gemma3 is not supported yet for custom JSON format");
        }
        let pre_norm = LayerNorm::from_json(&l.pp("ffn_"), c)?;
        let up = l.get_tensor("ffn_up.weight")?;
        let up_bias = l.get_tensor("ffn_up.bias")?;
        let down = l.get_tensor("ffn_down.weight")?;
        let down_bias = l.get_tensor("ffn_down.bias")?;
        ensure!(
            up.shape()[0] == c.hidden_size,
            "up have shape {:?} but in features should be equal to hidden_size: {}",
            up.shape(),
            c.hidden_size
        );
        ensure!(
            down.shape()[1] == c.embedding_size,
            "down have shape {:?} but out features should be equal to embedding_size: {}",
            down.shape(),
            c.embedding_size
        );
        Ok(Self {
            pre_norm: Norm::LayerNorm(pre_norm),
            up,
            up_bias: Some(up_bias),
            down,
            down_bias: Some(down_bias),
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
    pub fn from_loader(
        &self,
        loader: &FileTensorLoader,
        c: &LLMConfig,
    ) -> anyhow::Result<Norm<f32>> {
        Ok(match self {
            NormType::LayerNorm => Norm::LayerNorm(LayerNorm::from_loader(loader, c)?),
            NormType::RMSNorm => Norm::RMSNorm(RMSNorm::from_loader(loader, c)?),
        })
    }
}
