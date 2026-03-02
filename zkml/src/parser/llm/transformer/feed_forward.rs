use crate::{
    graph::NodeId,
    layers::{
        Layer,
        activation::{Activation, ActivationLayer},
        einsum::EinSum,
    },
    model::Model,
    parser::{LayerInsertion, llm::metadata::FFNGraphID},
    tensor::KeyedTensor,
};

use anyhow::Result;

pub(crate) const FEED_FORWARD_UP_EINSUM_NAME: &str = "ffn_up_einsum";
pub(crate) const FEED_FORWARD_GATE_EINSUM_NAME: &str = "ffn_gate_einsum";
pub(crate) const FEED_FORWARD_DOWN_EINSUM_NAME: &str = "ffn_down_einsum";

#[derive(Debug, Clone)]
/// A general configuration for a feed-forward network (FFN) layer in a transformer model.
pub struct FeedForwardNetwork {
    /// Optional gate tensor for Gated Linear Unit (GLU) functionality.
    pub gate: Option<KeyedTensor<f32>>,
    /// Weight tensor for the "up" projection.
    pub up: KeyedTensor<f32>,
    /// Optional bias tensor for the "up" projection.
    pub up_bias: Option<KeyedTensor<f32>>,
    /// Weight tensor for the "down" projection.
    pub down: KeyedTensor<f32>,
    /// Optional bias tensor for the "down" projection.
    pub down_bias: Option<KeyedTensor<f32>>,
    /// The activation function to be used in the FFN layer.
    pub activation: ActivationLayer<f32>,
}

impl LayerInsertion for FeedForwardNetwork {
    type Metadata = FFNGraphID;
    fn add_to_model(
        self,
        model: &mut Model<f32>,
        previous_node_id: Option<NodeId>,
    ) -> Result<(NodeId, Self::Metadata)> {
        // First we create the initial EinSum node
        let FeedForwardNetwork {
            gate,
            up,
            up_bias,
            down,
            down_bias,
            activation,
        } = self;
        let use_gate = gate.is_some();
        let up_einsum_id = match gate {
            Some(gate_tensor) => {
                // If a gate tensor is provided we use a GLU EinSum
                // Here we use `s` to represent the sequence length dimension, `e` for embedding dimension and
                // `p` for the projection/intermediate dimension
                let input_terms = "X(se)@WG(ep):WU(ep)";
                let output_terms = if up_bias.is_some() {
                    "G(sp):U(sp)+BIAS(p)"
                } else {
                    "G(sp):U(sp)"
                };

                let glu_einsum = EinSum::<f32>::new(
                    format!("{}->{}", input_terms, output_terms),
                    vec![Some(gate_tensor.into()), Some(up.into())],
                    vec![None, up_bias.map(|tensor| tensor.into())],
                )?
                .no_requant()
                .with_name(FEED_FORWARD_GATE_EINSUM_NAME.to_owned());
                model.add_consecutive_layer(Layer::EinSum(glu_einsum), previous_node_id)?
            }
            None => {
                // Otherwise we use a standard EinSum
                // Here we use `s` to represent the sequence length dimension, `e` for embedding dimension and
                // `p` for the projection/intermediate dimension
                let input_terms = "X(se)@WU(ep)";
                let output_terms = if up_bias.is_some() {
                    "U(sp)+BIAS(p)"
                } else {
                    "U(sp)"
                };

                let einsum = EinSum::<f32>::new(
                    format!("{}->{}", input_terms, output_terms),
                    vec![Some(up.into())],
                    vec![up_bias.map(|tensor| tensor.into())],
                )?
                .no_requant()
                .with_name(FEED_FORWARD_UP_EINSUM_NAME.to_owned());
                model.add_consecutive_layer(Layer::EinSum(einsum), previous_node_id)?
            }
        };

        // Create the activation layer
        let activation_id = match use_gate {
            true => {
                let activation = Activation::new_for_glu(activation);
                let glu_id = model.graph_mut().add_inner(Layer::Activation(activation))?;
                // One edge connecting the output of the up projection to the GLU activation
                model.add_edge(
                    up_einsum_id,
                    glu_id,
                    vec![
                        (0, Activation::<f32>::GATE_INPUT_INDEX),
                        (1, Activation::<f32>::UP_INPUT_INDEX),
                    ],
                )?;
                glu_id
            }
            false => {
                let activation = Activation::new_plain(activation);
                model.add_consecutive_layer(Layer::Activation(activation), Some(up_einsum_id))?
            }
        };
        // Finally, create the down projection EinSum
        // Once again we use `s` for sequence length, `p` for projection/intermediate dimension and `e` for embedding dimension
        let down_input_terms = "X(sp)@WD(pe)";
        let down_output_terms = if down_bias.is_some() {
            "Y(se)+BIAS(e)"
        } else {
            "Y(se)"
        };
        let down_einsum = EinSum::<f32>::new(
            format!("{}->{}", down_input_terms, down_output_terms),
            vec![Some(down.into())],
            vec![down_bias.map(|tensor| tensor.into())],
        )?
        .no_requant()
        .with_name(FEED_FORWARD_DOWN_EINSUM_NAME.to_owned());
        let down_id =
            model.add_consecutive_layer(Layer::EinSum(down_einsum), Some(activation_id))?;
        let ffn_graph_id = FFNGraphID {
            up_id: up_einsum_id,
            down_id,
            activation_id,
            uses_glu: use_gate,
        };
        Ok((down_id, ffn_graph_id))
    }
}
