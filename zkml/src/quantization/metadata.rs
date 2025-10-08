//! Metadata related information for a model. These are the information derived from the
//! float based model weights and activations.
use std::collections::{BTreeMap, HashMap};

use anyhow::{Result, anyhow, ensure};
use serde::{Deserialize, Serialize};

use crate::{
    graph::{Edge, Ports, Source},
    model::{Model, NodeID},
};

use super::ScalingFactor;

/// Structure holding the scaling factors of the input and output of each layer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMetadata {
    pub input: Vec<ScalingFactor>,
    pub(crate) input_layers_scaling: HashMap<NodeID, Vec<ScalingFactor>>,
    pub(crate) output_layers_scaling: HashMap<NodeID, Vec<ScalingFactor>>,
    pub(crate) output: Vec<ScalingFactor>,
    pub float_model: Option<Model<f32>>,
}

impl ModelMetadata {
    pub fn output_scaling_factor(&self) -> Vec<ScalingFactor> {
        self.output.clone()
    }

    pub fn layer_output_scaling_factor(&self, node_id: NodeID) -> &[ScalingFactor] {
        self.output_layers_scaling
            .get(&node_id)
            .unwrap_or_else(|| panic!("Node {node_id} not found"))
    }

    pub fn layer_input_scaling_factor(&self, node_id: NodeID) -> &[ScalingFactor] {
        self.input_layers_scaling
            .get(&node_id)
            .unwrap_or_else(|| panic!("Node {node_id} not found"))
    }
}

pub(crate) struct MetadataBuilder {
    pub(crate) input_scaling: Vec<ScalingFactor>,
    output_layers_scaling: HashMap<NodeID, Vec<ScalingFactor>>,
    input_layers_scaling: HashMap<NodeID, Vec<ScalingFactor>>,
}

impl MetadataBuilder {
    pub fn new(input_scaling: Vec<ScalingFactor>) -> Self {
        Self {
            input_scaling,
            output_layers_scaling: HashMap::new(),
            input_layers_scaling: HashMap::new(),
        }
    }

    pub fn set_layers_scaling(
        &mut self,
        node_id: NodeID,
        output_scaling: Vec<ScalingFactor>,
        input_scaling: Vec<ScalingFactor>,
    ) {
        self.output_layers_scaling.insert(node_id, output_scaling);
        self.input_layers_scaling.insert(node_id, input_scaling);
    }

    /// Take all incoming edges and map the input scaling factors corresponding to each source port
    pub(crate) fn map_to_input_scaling<'a, I: Iterator<Item = &'a Edge<NodeID, ()>>>(
        &self,
        mut node_inputs: I,
    ) -> Result<Vec<ScalingFactor>> {
        Ok(node_inputs.try_fold(BTreeMap::new(), |mut acc, edge| {
            for port in edge.ports().iter() {
                match edge.source() {
                    Source::Node(n) => {
                        let scalings = self.get_output_layer_scaling(n).ok_or(
                            anyhow!("Scaling factors for node {n} not found")
                        )?;
                        ensure!(*port.source_port < scalings.len(),
                            "Getting scaling factor {} for node {n}, but there are only {} scaling factors",
                            *port.source_port,
                            scalings.len(),
                        );
                        acc.insert(*port.target_port, scalings[*port.source_port]);
                    }
                    Source::Input => {
                        ensure!(*port.source_port < self.input_scaling.len(),
                            "Getting scaling factor {} for model inputs, but there are only {} scaling factors",
                            *port.source_port,
                            self.input_scaling.len(),
                        );
                        acc.insert(*port.target_port, self.input_scaling[*port.source_port]);
                    }
                }
            }
            Ok(acc)
        })?
        .into_values().collect())
    }

    pub(crate) fn get_output_layer_scaling(&self, node_id: &NodeID) -> Option<&[ScalingFactor]> {
        self.output_layers_scaling
            .get(node_id)
            .map(|s| s.as_slice())
    }

    pub fn build<'a, I: Iterator<Item = (&'a NodeID, &'a Ports)>>(
        self,
        output_nodes: I,
    ) -> Result<ModelMetadata> {
        let mut output_scalings = BTreeMap::new();
        for (id, ports) in output_nodes.into_iter() {
            let scalings = self
                .get_output_layer_scaling(id)
                .ok_or(anyhow!("Scaling factors not found for node {id}"))?;
            ensure!(
                scalings.len() >= ports.iter().count(),
                "Number of scalings factors found for node {id} ({}) is different from
                the expected number of outputs of the node ({})",
                scalings.len(),
                ports.iter().count(),
            );
            for port in ports.iter() {
                ensure!(
                    output_scalings
                        // we register the model output at target_port to have the scaling
                        // from the output of the node located at the source_port
                        .insert(port.target_port, scalings[*port.source_port])
                        .is_none(),
                    "Scaling factor for output {} found twice",
                    port.target_port
                );
            }
        }
        // check that all scaling factors have been found
        ensure!(
            !output_scalings.is_empty(),
            "No output scaling factors found"
        );
        ensure!(
            **output_scalings.first_key_value().unwrap().0 == 0
                && **output_scalings.last_key_value().unwrap().0 == output_scalings.len() - 1,
            "Not all output scaling factors found"
        );

        Ok(ModelMetadata {
            input: self.input_scaling,
            output_layers_scaling: self.output_layers_scaling,
            input_layers_scaling: self.input_layers_scaling,
            output: output_scalings.into_values().collect(),
            float_model: None,
        })
    }
}
