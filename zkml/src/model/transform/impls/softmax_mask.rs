//! Defines the [`ModelTransform`] that should be applied after quantising a [Softmax][crate::layers::transformer::softmax] layer.
use anyhow::Context;

use crate::{
    Element,
    graph::Direction,
    layers::{
        Layer,
        transformer::{attention::attention_mask::ATTENTION_MASK_LAYER, softmax::SOFTMAX_LAYER},
    },
    model::{Model, NodeID, transform::ModelTransform},
};

use anyhow::{Result, anyhow, ensure};

#[derive(Debug, Clone)]
/// This transform is used after quantising a [Softmax][crate::layers::transformer::softmax] layer to ensure that the
/// associated attention mask has the correct negative infinity value set. The [NodeID] provided should be the Softmax layer's NodeID.
/// The [Element] provided should be the negative infinity value used in the attention mask.
///
/// TODO: Add "Padding" variant to attention mask so that if the previous layer was not a mask we insert a new mask node.
pub struct SoftmaxMaskTransform(pub(crate) NodeID, pub(crate) Element);

impl SoftmaxMaskTransform {
    pub fn new(softmax_node: NodeID, neg_inf: Element) -> Self {
        Self(softmax_node, neg_inf)
    }
}

impl ModelTransform<Element> for SoftmaxMaskTransform {
    fn apply(&self, mut model: Model<Element>) -> Result<Model<Element>> {
        // First we get the input node, the provided NodeID should be the Softmax node
        let softmax_node = model
            .graph
            .node(&self.0)
            .ok_or(anyhow!("Softmax node not found"))?;

        ensure!(
            softmax_node.short_name() == SOFTMAX_LAYER,
            "Could not apply SoftmaxMaskTransform, provided NodeID was for layer: {}",
            softmax_node.short_name()
        );

        // Now we know that we have a Softmax Node check the input is an attention mask
        let input_node_id = {
            let mut input_nodes = model
                .graph
                .node_neighbors(&self.0, Direction::Incoming)
                .map(|(_, edge)| edge);
            // safe to unwrap since we called node_neighbors
            #[allow(clippy::clone_on_copy)]
            let input_node_id = input_nodes
                .next()
                .context("Expected 1 input node")?
                .source_id()
                .unwrap()
                .clone();
            ensure!(
                input_nodes.next().is_none(),
                "Softmax node should have 1 input, found more"
            );
            input_node_id
        };

        let mut input_node = model
            .graph
            .node_mut(&input_node_id)
            .expect("Softmax input node not found");
        // Make sure its an attention mask
        ensure!(
            input_node.short_name() == ATTENTION_MASK_LAYER,
            "Softmax input node is not an AttentionMask, found {}",
            input_node.short_name()
        );

        // Now we can set the mask negative infinity value to be correct
        let Layer::AttentionMask(mask) = &mut input_node else {
            unreachable!()
        };

        mask.set_negative_infinity(self.1);
        Ok(model)
    }
}
