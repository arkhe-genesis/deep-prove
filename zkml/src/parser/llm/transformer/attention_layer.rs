//! Code that constructs attention modules for use in transformer layers.

use crate::{
    graph::NodeId,
    layers::{
        Layer,
        einsum::EinSum,
        transformer::{
            attention_mask::{AttentionMask, AttentionSpan},
            softmax::Softmax,
        },
    },
    model::{LayerInsertion, Model},
    tensor::TensorHandle,
};

use anyhow::{Result, ensure};

/// Struct holding the information needed to attach [ConcatenationCache][crate::layers::transformer::ConcatenationCache] to an attention mechanisms QKV projection layer.
#[derive(Debug, Clone)]
pub struct AttentionCacheConfig {
    /// Rank of the key/value tensor.
    pub kv_rank: usize,
    /// Dimension index along which to concatenate the cached key/values.
    pub caching_dim: usize,
}

/// Trait defining common functionality for attention mechanisms.
pub trait AttentionMechanism {
    /// Gets the total number of attention heads (query heads).
    fn total_heads(&self) -> usize;
    /// Gets the total number of key/value attention heads.
    fn kv_total_heads(&self) -> usize;
    /// Gets the number of query heads per key/value head.
    fn heads_per_kv_head(&self) -> usize;
    /// Getter for the maximum context length of the attention layer.
    fn max_context_length(&self) -> usize;
    /// Getter for the dimension of each attention head.
    fn head_dim(&self) -> usize;
    /// Getter that tells us whether the Q, K and V projections use biases.
    fn uses_qkv_bias(&self) -> bool;
    /// Getter that tells us if the out projection uses bias.
    fn uses_out_bias(&self) -> bool;
    /// Forms the einsum equations used in the attention mechanism.
    /// The returned array contains the equations in the following order:
    /// - QKV projection equation
    /// - Query-Key attention equation
    /// - Attention-Value equation
    /// - Output projection equation
    fn form_einsum_equations(&self) -> [String; 4] {
        match self.heads_per_kv_head() {
            // 1 query head per kv head -> Multi Headed Attention
            1 => form_mha_einsum_equations(self.uses_qkv_bias(), self.uses_out_bias()),
            // 1 kv head -> Multi Query Attention
            n if n == self.total_heads() => {
                form_mqa_einsum_equations(self.uses_qkv_bias(), self.uses_out_bias())
            }
            // Otherwise -> Group Query Attention
            _ => form_gqa_einsum_equations(self.uses_qkv_bias(), self.uses_out_bias()),
        }
    }
    /// Returns an [AttentionCacheConfig] indicating how to attach caching to this attention mechanism.
    fn caching_dim(&self) -> AttentionCacheConfig {
        // If its MHA or GQA the rank is 3 and caching dim is 1, if its MQA the rank is 2 caching dim is 0
        match self.heads_per_kv_head() {
            n if n == self.total_heads() => AttentionCacheConfig {
                kv_rank: 2,
                caching_dim: 0,
            },
            _ => AttentionCacheConfig {
                kv_rank: 3,
                caching_dim: 1,
            },
        }
    }

    /// Builds the QKV projection EinSum layer.
    fn build_qkv_einsum(
        equation: String,
        weights: Vec<TensorHandle<f32>>,
        biases: Vec<Option<TensorHandle<f32>>>,
    ) -> Result<EinSum<f32>> {
        ensure!(
            weights.len() == 3,
            "Expected 3 weight tensors for Q, K and V projections"
        );
        ensure!(
            biases.len() == 3,
            "Expected 3 bias tensors for Q, K and V projections"
        );
        EinSum::<f32>::new(equation, weights.into_iter().map(Some).collect(), biases)
    }

    /// Builds the query-key attention EinSum layer.
    fn build_query_key_attention_einsum(equation: String) -> Result<EinSum<f32>> {
        EinSum::<f32>::new(equation, vec![None], vec![None])
    }

    /// Builds the attention value EinSum layer.
    fn build_attention_value_einsum(equation: String) -> Result<EinSum<f32>> {
        EinSum::<f32>::new(equation, vec![None], vec![None])
    }

    /// Builds the output projection EinSum layer.
    fn build_output_einsum(
        equation: String,
        weight: Vec<TensorHandle<f32>>,
        bias: Vec<Option<TensorHandle<f32>>>,
    ) -> Result<EinSum<f32>> {
        ensure!(
            weight.len() == 1,
            "Expected 1 weight tensor for output projection"
        );
        ensure!(
            bias.len() == 1,
            "Expected 1 optional bias tensor for output projection"
        );
        EinSum::<f32>::new(equation, weight.into_iter().map(Some).collect(), bias)
    }

    /// Getter for the QKV weight and bias tensors.
    fn qkv_tensors(&self) -> (Vec<TensorHandle<f32>>, Vec<Option<TensorHandle<f32>>>);

    /// Getter for the output projection weight and bias tensors.
    fn out_tensors(&self) -> (TensorHandle<f32>, Option<TensorHandle<f32>>);

    /// Getter for the attention span of the attention mechanism.
    fn attention_span(&self) -> AttentionSpan;

    /// Method that defines custom logic to be inserted, and wiring for the nodes in the model between the QKV [`EinSum`] layer and the query-key attention [`EinSum`] layer.
    /// This could be a positional layer applied to queries and keys, normalisation or any other custom layer.
    fn insert_custom_logic(
        self,
        model: &mut Model<f32>,
        qkv_einsum_id: NodeId,
        query_key_id: NodeId,
    ) -> Result<()>;

    fn write_to_model(
        self,
        model: &mut Model<f32>,
        previous_node_id: Option<NodeId>,
    ) -> Result<NodeId>
    where
        Self: Sized,
    {
        // Retrieve the einsum equations
        let [
            qkv_equation,
            query_key_equation,
            attention_value_equation,
            output_equation,
        ] = self.form_einsum_equations();

        // Get the max context length and the head dimension
        let max_context_length = self.max_context_length();
        let head_dim = self.head_dim();

        // Extract the tensors from the attention mechanism
        let (weights, biases) = self.qkv_tensors();
        let qkv_einsum = Self::build_qkv_einsum(qkv_equation, weights, biases)?;
        let query_key_einsum =
            Self::build_query_key_attention_einsum(query_key_equation)?.disable_requantisation();
        let attention_value_einsum = Self::build_attention_value_einsum(attention_value_equation)?;
        let (out_weight, out_bias) = self.out_tensors();
        let output_einsum =
            Self::build_output_einsum(output_equation, vec![out_weight], vec![out_bias])?;
        // We also get the attention span
        let attention_span = self.attention_span();
        // We insert the query key EinSum into the model (but don't wire it up yet)
        let qkv_einsum_id =
            model.add_consecutive_layer(Layer::EinSum(qkv_einsum), previous_node_id)?;
        let query_key_id = model
            .graph_mut()
            .add_inner(Layer::EinSum(query_key_einsum))?;
        // We insert any custom logic specific to the attention mechanism
        self.insert_custom_logic(model, qkv_einsum_id, query_key_id)?;
        // We insert the attention mask layer and the softmax layer
        let softmax_id = match attention_span {
            AttentionSpan::Full => {
                let mask_id = model.add_consecutive_layer(
                    Layer::AttentionMask(
                        AttentionMask::<f32>::new(f32::NEG_INFINITY).with_span(attention_span)?,
                    ),
                    Some(query_key_id),
                )?;
                // We insert the Softmax layer

                model.add_consecutive_layer(
                    Layer::Softmax(
                        Softmax::<f32>::new(max_context_length.next_power_of_two())
                            .with_scale(1.0f32 / (head_dim as f32).sqrt()),
                    ),
                    Some(mask_id),
                )?
            }
            AttentionSpan::Local(effective_context_length) => {
                // Insert the attention mask layer with local span
                let mask_id = model.add_consecutive_layer(
                    Layer::AttentionMask(
                        AttentionMask::<f32>::new(f32::NEG_INFINITY).with_span(attention_span)?,
                    ),
                    Some(query_key_id),
                )?;
                let effective_context_length = effective_context_length
                    .min(max_context_length)
                    .next_power_of_two();
                // Now in the Softmax layer we know we only have at most effective_context_length non-zero entries (after applying exp)
                model.add_consecutive_layer(
                    Layer::Softmax(
                        Softmax::<f32>::new(effective_context_length)
                            .with_scale(1.0f32 / (head_dim as f32).sqrt()),
                    ),
                    Some(mask_id),
                )?
            }
        };

        // Finally we insert the attention value EinSum and output EinSum layers
        let attention_value_id = model
            .graph_mut()
            .add_inner(Layer::EinSum(attention_value_einsum))?;

        model.add_edge(softmax_id, attention_value_id, (0, 0))?;
        model.add_edge(qkv_einsum_id, attention_value_id, (2, 1))?;

        model.add_consecutive_layer(Layer::EinSum(output_einsum), Some(attention_value_id))
    }
}

/// Forms the einsum equations for Multi Headed Attention.
fn form_mha_einsum_equations(uses_qkv_bias: bool, uses_out_bias: bool) -> [String; 4] {
    // 'h' represent the number of heads, 'd' the head dimension
    // 's' is the sequence length and 'e' is the embedding dimension.
    let qkv_inputs = "X(se)@WQ(ehd):WK(ehd):WV(ehd)".to_string();
    let qkv_outputs = if uses_qkv_bias {
        "Q(hsd)+BIAS(hd):K(hsd)+BIAS(hd):V(hsd)+BIAS(hd)".to_string()
    } else {
        "Q(hsd):K(hsd):V(hsd)".to_string()
    };
    let qkv_equation = format!("{qkv_inputs}->{qkv_outputs}");
    // Here `q` is the query sequence length, `s` is the key/value sequence length, we use different values to allow for cached inference
    let qk_equation = "Q(hqd)@K(hsd)->A(hqs)".to_string();

    let attention_value_equation = "A(hqs)@V(hsd)->O(hqd)".to_string();

    let output_inputs = "O(hqd)@WO(hde)".to_string();
    let output_outputs = if uses_out_bias {
        "Y(qe)+BIAS(e)".to_string()
    } else {
        "Y(qe)".to_string()
    };
    let output_equation = format!("{output_inputs}->{output_outputs}");
    [
        qkv_equation,
        qk_equation,
        attention_value_equation,
        output_equation,
    ]
}

/// Forms the einsum equations for Multi Query Attention.
fn form_mqa_einsum_equations(uses_qkv_bias: bool, uses_out_bias: bool) -> [String; 4] {
    // 'h' represent the number of query heads, 'd' the head dimension
    // 's' is the sequence length and 'e' is the embedding dimension.
    let qkv_inputs = "X(se)@WQ(ehd):WK(ed):WV(ed)".to_string();
    let qkv_outputs = if uses_qkv_bias {
        "Q(hsd)+BIAS(hd):K(sd)+BIAS(d):V(sd)+BIAS(d)".to_string()
    } else {
        "Q(hsd):K(sd):V(sd)".to_string()
    };
    let qkv_equation = format!("{qkv_inputs}->{qkv_outputs}");
    // Here `q` is the query sequence length, `s` is the key/value sequence length, we use different values to allow for cached inference
    let qk_equation = "Q(hqd)@K(sd)->A(hqs)".to_string();

    let attention_value_equation = "A(hqs)@V(sd)->O(hqd)".to_string();

    let output_inputs = "O(hqd)@WO(hde)".to_string();
    let output_outputs = if uses_out_bias {
        "Y(qe)+BIAS(e)".to_string()
    } else {
        "Y(qe)".to_string()
    };
    let output_equation = format!("{output_inputs}->{output_outputs}");
    [
        qkv_equation,
        qk_equation,
        attention_value_equation,
        output_equation,
    ]
}

/// Forms the einsum equations for Group Query Attention.
fn form_gqa_einsum_equations(uses_qkv_bias: bool, uses_out_bias: bool) -> [String; 4] {
    // 'h' represent the number of key-value heads, 'g' the number of query heads per kv head, 'd' the head dimension
    // 's' is the sequence length and 'e' is the embedding dimension.
    let qkv_inputs = "X(se)@WQ(ehgd):WK(ehd):WV(ehd)".to_string();
    let qkv_outputs = if uses_qkv_bias {
        "Q(ghsd)+BIAS(ghd):K(hsd)+BIAS(hd):V(hsd)+BIAS(hd)".to_string()
    } else {
        "Q(ghsd):K(hsd):V(hsd)".to_string()
    };
    let qkv_equation = format!("{qkv_inputs}->{qkv_outputs}");
    // Here `q` is the query sequence length, `s` is the key/value sequence length, we use different values to allow for cached inference
    let qk_equation = "Q(ghqd)@K(hsd)->A(ghqs)".to_string();
    let attention_value_equation = "A(ghqs)@V(hsd)->O(ghqd)".to_string();
    let output_inputs = "O(ghqd)@WO(ghde)".to_string();
    let output_outputs = if uses_out_bias {
        "Y(qe)+BIAS(e)".to_string()
    } else {
        "Y(qe)".to_string()
    };
    let output_equation = format!("{output_inputs}->{output_outputs}");
    [
        qkv_equation,
        qk_equation,
        attention_value_equation,
        output_equation,
    ]
}

impl<A: AttentionMechanism> LayerInsertion for A {
    fn add_to_model(
        self,
        model: &mut Model<f32>,
        previous_node_id: Option<NodeId>,
    ) -> Result<NodeId> {
        self.write_to_model(model, previous_node_id)
    }
}
