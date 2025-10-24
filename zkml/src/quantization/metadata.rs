//! Metadata related information for a model. These are the information derived from the
//! float based model weights and activations.
use super::ScalingFactor;
use crate::{
    graph::{NodeId, NodeOutput, PortId},
    model::Model,
};
use anyhow::{Result, anyhow};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};

/// Structure holding the scaling factors of the input and output of each layer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMetadata {
    /// Store the corresponding node ID for each input. This is used to retrieve
    /// input scaling factor in contexts where the whole model is not available,
    /// and thus input numbers can not be matched to node ID.
    input_nodes: Vec<NodeId>,
    /// Similar, but for the outputs.
    output_nodes: Vec<NodeId>,
    /// The [`ScalingFactor`] related to each [`NodeInput`] in the model.
    pub(crate) input_layers_scaling: HashMap<NodeId, BTreeMap<PortId, ScalingFactor>>,
    /// The [`ScalingFactor`] related to each [`NodeOutput`] in the model.
    pub(crate) output_layers_scaling: HashMap<NodeId, BTreeMap<PortId, ScalingFactor>>,
    pub float_model: Option<Model<f32>>,
}

impl ModelMetadata {
    /// Return the scaling factor for the `input_idx`'d global input of the model.
    pub fn input_scaling(&self, input_idx: usize) -> &ScalingFactor {
        &self.output_layers_scaling[&self.input_nodes[input_idx]][&(0.into())]
    }

    /// Return the scaling factor for the `output_idx`'d global output of the model.
    pub fn output_scaling(&self, output_idx: usize) -> &ScalingFactor {
        &self.input_layers_scaling[&self.output_nodes[output_idx]][&(0.into())]
    }

    /// Return a list of the scaling factors related to the outputs of the given
    /// node, ordered by port number.
    pub fn layer_input_scaling_factor(&self, node_id: NodeId) -> Vec<&ScalingFactor> {
        self.input_layers_scaling[&node_id].values().collect()
    }

    /// Return a list of the scaling factors related to the inputs of the given
    /// node, ordered by port number.
    pub fn layer_output_scaling_factor(&self, node_id: NodeId) -> Vec<&ScalingFactor> {
        self.output_layers_scaling[&node_id].values().collect()
    }
}

pub(crate) struct MetadataBuilder {
    input_layers_scaling: HashMap<NodeId, BTreeMap<PortId, ScalingFactor>>,
    output_layers_scaling: HashMap<NodeId, BTreeMap<PortId, ScalingFactor>>,
}

impl MetadataBuilder {
    pub fn new() -> Self {
        Self {
            output_layers_scaling: HashMap::new(),
            input_layers_scaling: HashMap::new(),
        }
    }

    pub fn insert_layer_scalings(
        &mut self,
        node_id: NodeId,
        output_scalings: Vec<ScalingFactor>,
        input_scalings: Vec<ScalingFactor>,
    ) {
        for (i, out_scaling) in output_scalings.into_iter().enumerate() {
            self.output_layers_scaling
                .entry(node_id)
                .or_default()
                .insert(i.into(), out_scaling);
        }

        for (i, in_scaling) in input_scalings.into_iter().enumerate() {
            self.input_layers_scaling
                .entry(node_id)
                .or_default()
                .insert(i.into(), in_scaling);
        }
    }

    pub(crate) fn get_output_layer_scaling(
        &self,
        node_out: NodeOutput,
    ) -> anyhow::Result<ScalingFactor> {
        self.output_layers_scaling
            .get(&node_out.node_id)
            .and_then(|sfs| sfs.get(&node_out.port))
            .map(|x| x.to_owned())
            .ok_or_else(|| anyhow!("fetching scaling for {node_out:?}"))
    }

    pub fn build(
        self,
        input_nodes: Vec<NodeId>,
        output_nodes: Vec<NodeId>,
    ) -> Result<ModelMetadata> {
        Ok(ModelMetadata {
            input_nodes,
            output_nodes,
            output_layers_scaling: self.output_layers_scaling,
            input_layers_scaling: self.input_layers_scaling,
            float_model: None,
        })
    }
}
