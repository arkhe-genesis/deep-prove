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
        gguf::{FileTensorLoader, unfuse_tensors},
        json,
        json::unfuse_crate_tensors,
        llm::{FeedForward, LLMConfig, LLMVariant, config::AttentionHeadType},
    },
    tensor::KeyedTensor,
};
use anyhow::{Context, bail, ensure};
use candle_core::Device;
use tracing::trace;

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
        c: &LLMConfig,
        positional: Option<Positional<f32>>,
    ) -> anyhow::Result<NodeId> {
        let num_groups = match c.attention_config.head {
            AttentionHeadType::MHA => c.num_heads,
            AttentionHeadType::GQA(num_groups) => num_groups,
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
        let mha =
            Mha::new(c.context_length, c.num_heads, c.head_size)?.with_attention_span(self.span)?;
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
        if let LLMVariant::Gemma3 = c.variant {
            let q_norm = self.q_norm.context("in gemma3, q_norm is expected")?;
            (q_id, q_port) = {
                let q_norm_id = model.graph.add_inner(q_norm.to_layer())?;
                model.add_edge(q_id, q_norm_id, (q_port, 0))?;
                (q_norm_id, 0)
            };
            let k_norm = self.k_norm.context("in gemma3, k_norm is expected")?;
            (k_id, k_port) = {
                let k_norm_id = model.graph.add_inner(k_norm.to_layer())?;
                model.add_edge(k_id, k_norm_id, (k_port, 0))?;
                (k_norm_id, 0)
            };
            let rope = positional.context("in gemma3, rope is expected")?;
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
        } else if let LLMVariant::GPT2 = c.variant {
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

    pub fn with_span(self, span: AttentionSpan) -> Self {
        Self { span, ..self }
    }

    fn from_loader_gemma3(loader: &FileTensorLoader, c: &LLMConfig) -> anyhow::Result<Self> {
        let hidden_size = c.hidden_size;
        let head_size = c.head_size;
        let num_heads = c.num_heads;
        let num_groups = c.num_groups();

        let pre_norm = RMSNorm::from_loader(&loader.pp("attn_"), c, false)?;
        assert_eq!(
            pre_norm.alpha.as_ref().unwrap().shape().as_ref(),
            &[c.embedding_size]
        );

        let q_tensor = loader
            .get_tensor("attn_q.weight")?
            .map_tensor(|t| Tensor::transpose(&t));
        let q_norm = RMSNorm::from_loader(&loader.pp("attn_q_"), c, true)?;
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
            // HACK: stacking
            &[c.head_size * c.num_heads]
        );

        let k_tensor = loader
            .get_tensor("attn_k.weight")?
            .map_tensor(|t| Tensor::transpose(&t));
        let k_norm = RMSNorm::from_loader(&loader.pp("attn_k_"), c, true)?;
        assert_eq!(
            k_tensor.shape().as_ref(),
            &[hidden_size, num_groups * head_size]
        );
        // head_dim = num_groups * head_size
        assert_eq!(
            k_norm.alpha.as_ref().unwrap().shape().as_ref(),
            // HACK: stacking
            &[c.head_size * c.num_heads]
        );

        let v_tensor = loader
            .get_tensor("attn_v.weight")?
            .map_tensor(|t| Tensor::transpose(&t));
        assert_eq!(
            v_tensor.shape().as_ref(),
            &[hidden_size, num_groups * head_size]
        );

        // HACK: since we don't have proper GQA for now, we fake the "one" group by stacking multiple times
        // the K and V tensors on themselves, as many times as there are heads. In Gemma3 270M there are only
        // 4 heads so it's ok for now. This means when we split inside MHA per head, then each head will have
        // the same K and V tensors, effectively enforcing a single group.
        // TODO: remove this once we have proper GQA
        ensure!(num_groups == 1, "GQA is not supported yet");
        ensure!(
            num_heads == 4,
            "GQA is not supported yet so stacking is expensive"
        );

        // println!("LLM LOADER: k_tensor shape: {:?}", k_tensor.shape());
        // println!("LLM LOADER: v_tensor shape: {:?}", v_tensor.shape());
        let k_tensor = k_tensor.map_tensor(|t| expand(t, num_heads));
        let v_tensor = v_tensor.map_tensor(|t| expand(t, num_heads));

        let out = loader
            .get_tensor("attn_output.weight")?
            .map_tensor(|t| Tensor::transpose(&t));
        assert_eq!(out.shape().as_ref(), &[num_heads * head_size, hidden_size]);

        let post_attn_norm = RMSNorm::from_loader(&loader.pp("post_attention_"), c, false)?;
        assert_eq!(
            post_attn_norm.alpha.as_ref().unwrap().shape().as_ref(),
            &[c.hidden_size]
        );

        let ff = FeedForward::from_loader(loader, c)?;
        let scope_loader = loader.pp("post_ffw_");
        let post_ffn_norm = RMSNorm::from_loader(&scope_loader, c, false)?;
        assert_eq!(
            post_ffn_norm.alpha.as_ref().unwrap().shape().as_ref(),
            &[c.hidden_size]
        );
        let ffn_norm_loader = loader.pp("ffn_");

        let pre_ffn_norm = c
            .variant
            .norm_type()
            .from_loader(&ffn_norm_loader, c, false)?;
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
            pre_ffn_norm,
            feedforward: ff,
            post_ffn_norm: Some(Norm::RMSNorm(post_ffn_norm)),
            span: AttentionSpan::Full,
        })
    }

    fn from_loader_gpt2(loader: &FileTensorLoader, c: &LLMConfig) -> anyhow::Result<Self> {
        let embedding_size = c.embedding_size;
        let hidden_size = c.hidden_size;
        ensure!(
            embedding_size == hidden_size,
            "embedding_size must be equal to hidden_size"
        );
        let (qkv_key, qkv_weight_qtensor) = loader.get_qtensor("attn_qkv.weight")?;
        let qkv_weight_candle = qkv_weight_qtensor.dequantize(&Device::Cpu)?;
        let mut unfused_weights =
            unfuse_tensors(qkv_weight_candle, embedding_size * embedding_size)?;
        ensure!(unfused_weights.len() == 3, "qkv_weight must have 3 chunks");
        let q = KeyedTensor::new(
            format!("{qkv_key}.q"),
            crate::Tensor::new(
                vec![embedding_size, hidden_size].into(),
                unfused_weights.remove(0),
            )
            .transpose(),
        );
        let k = KeyedTensor::new(
            format!("{qkv_key}.k"),
            crate::Tensor::new(
                vec![embedding_size, hidden_size].into(),
                unfused_weights.remove(0),
            )
            .transpose(),
        );
        let v = KeyedTensor::new(
            format!("{qkv_key}.v"),
            crate::Tensor::new(
                vec![embedding_size, hidden_size].into(),
                unfused_weights.remove(0),
            )
            .transpose(),
        );

        let (qkv_bias_key, qkv_bias_qtensor) = loader.get_qtensor("attn_qkv.bias")?;
        let qkv_bias_candle = qkv_bias_qtensor.dequantize(&Device::Cpu)?;
        let mut unfused_biases = unfuse_tensors(qkv_bias_candle, embedding_size)?;
        ensure!(unfused_biases.len() == 3, "qkv_bias must have 3 chunks");
        let q_bias = KeyedTensor::new(
            format!("{qkv_bias_key}.q"),
            crate::Tensor::new(vec![hidden_size].into(), unfused_biases.remove(0)),
        );
        let k_bias = KeyedTensor::new(
            format!("{qkv_bias_key}.k"),
            crate::Tensor::new(vec![hidden_size].into(), unfused_biases.remove(0)),
        );
        let v_bias = KeyedTensor::new(
            format!("{qkv_bias_key}.v"),
            crate::Tensor::new(vec![hidden_size].into(), unfused_biases.remove(0)),
        );

        let attn_norm_loader = loader.pp("attn_");
        // Use new LayerNorm::from_loader
        let pre_norm = LayerNorm::from_loader(&attn_norm_loader, c)?;

        // attn_output.weight is stored as [out_features, in_features] in GGUF (same as PyTorch)
        // Our MatMul layer expects the right-hand constant to be in the orientation [in_features, out_features],
        // so we transpose it once here after loading.
        let out = loader
            .get_tensor("attn_output.weight")?
            .map_tensor(|t| t.transpose());
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

        let pre_ffn_norm = c
            .variant
            .norm_type()
            .from_loader(&ffn_norm_loader, c, false)?;

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
            pre_ffn_norm,
            feedforward: ff,
            post_norm: None,
            post_ffn_norm: None,
            span: AttentionSpan::Full,
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
    pub fn from_loader(
        &self,
        loader: &FileTensorLoader,
        c: &LLMConfig,
        stack: bool,
    ) -> anyhow::Result<Norm<f32>> {
        Ok(match self {
            NormType::LayerNorm => Norm::LayerNorm(LayerNorm::from_loader(loader, c)?),
            NormType::RMSNorm => Norm::RMSNorm(RMSNorm::from_loader(loader, c, stack)?),
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
