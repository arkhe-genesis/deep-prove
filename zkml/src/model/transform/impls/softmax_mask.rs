//! Defines the [`ModelTransform`] that should be applied after quantising a [Softmax][crate::layers::transformer::softmax] layer.

use crate::{
    Element,
    layers::{
        Layer,
        provable::{NodeEdges, NodeId},
        transformer::{attention::attention_mask::ATTENTION_MASK_LAYER, softmax::SOFTMAX_LAYER},
    },
    model::{Model, transform::ModelTransform},
};

use anyhow::{Result, anyhow, ensure};

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
            .nodes
            .get(&self.0)
            .ok_or(anyhow!("Softmax node not found"))?;

        ensure!(
            softmax_node.operation.short_name() == SOFTMAX_LAYER,
            "Could not apply SoftmaxMaskTransform, provided NodeID was for layer: {}",
            softmax_node.operation.short_name()
        );

        // Now we know that we have a Softmax Node check the input is an attention mask
        let softmax_inputs = softmax_node.inputs();
        ensure!(
            softmax_inputs.len() == 1,
            "Softmax node should have 1 input, found {}",
            softmax_inputs.len()
        );

        let input_node_id = softmax_inputs[0]
            .node
            .ok_or(anyhow!("Softmax input node not found"))?;

        let input_node = model
            .nodes
            .get_mut(&input_node_id)
            .ok_or(anyhow!("Softmax input node not found"))?;
        // Make sure its an attention mask
        ensure!(
            input_node.operation.short_name() == ATTENTION_MASK_LAYER,
            "Softmax input node is not an AttentionMask, found {}",
            input_node.operation.short_name()
        );

        // Now we can set the mask negative infinity value to be correct
        let Layer::AttentionMask(mask) = &mut input_node.operation else {
            unreachable!()
        };

        mask.set_negative_infinity(self.1);
        Ok(model)
    }
}
