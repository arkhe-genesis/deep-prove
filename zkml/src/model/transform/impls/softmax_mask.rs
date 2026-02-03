//! Defines the [`ModelTransform`] that should be applied after quantising a [Softmax][crate::layers::transformer::softmax] layer.
use crate::{
    Element,
    graph::NodeId,
    layers::{
        Layer,
        transformer::{attention_mask::ATTENTION_MASK_LAYER, softmax::SOFTMAX_LAYER},
    },
    model::{Model, transform::ModelTransform},
};
use anyhow::{Context, Result, anyhow, ensure};

#[derive(Debug, Clone)]
/// This transform is used after quantising a [Softmax][crate::layers::transformer::softmax] layer to ensure that the
/// associated attention mask has the correct negative infinity value set. The [NodeId] provided should be the Softmax layer's NodeId.
/// The [Element] provided should be the negative infinity value used in the attention mask.
///
/// TODO: Add "Padding" variant to attention mask so that if the previous layer was not a mask we insert a new mask node.
pub struct SoftmaxMaskTransform(pub(crate) NodeId, pub(crate) Element);

impl SoftmaxMaskTransform {
    pub fn new(softmax_node: NodeId, neg_inf: Element) -> Self {
        Self(softmax_node, neg_inf)
    }
}

impl ModelTransform<Element> for SoftmaxMaskTransform {
    fn apply(&self, mut model: Model<Element>) -> Result<Model<Element>> {
        // First we get the input node, the provided NodeId should be the Softmax node
        let softmax_node = model
            .graph
            .node(self.0)
            .ok_or(anyhow!("Softmax node not found"))?
            .as_inner()
            .context("expected layer")?;

        ensure!(
            softmax_node.short_name() == SOFTMAX_LAYER,
            "Could not apply SoftmaxMaskTransform, provided NodeId was for layer: {}",
            softmax_node.short_name()
        );

        // Now we know that we have a Softmax Node check the input is an attention mask
        let input_node_id = {
            let input_nodes = model.graph.incoming_feeds(self.0);
            ensure!(input_nodes.len() == 1);
            input_nodes[0].source.node_id
        };

        let input_node = model
            .graph
            .node_mut(input_node_id)
            .expect("Softmax input node not found")
            .as_inner_mut()
            .context("expected attention mask layer before SoftmaxMaskTransform")?;

        // Make sure its an attention mask
        ensure!(
            input_node.short_name() == ATTENTION_MASK_LAYER,
            "Softmax input node is not an AttentionMask, found {}",
            input_node.short_name()
        );

        // Now we can set the mask negative infinity value to be correct
        let Layer::AttentionMask(mask) = input_node else {
            unreachable!()
        };

        mask.set_negative_infinity(self.1);
        Ok(model)
    }
}
