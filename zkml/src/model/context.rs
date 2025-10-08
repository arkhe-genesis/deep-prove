use crate::{
    Claim, Shape,
    graph::{Direction, Graph, PortID, Source, Target},
    iop::context::ShapeStep,
    layers::LayerCtx,
};
use anyhow::ensure;
use ff_ext::ExtensionField;
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use std::collections::{BTreeMap, BTreeSet, HashMap};

use super::NodeID;

pub type ContextGraph<N> = Graph<LayerCtx<N>, (), NodeID>;

/// Collection of the proving contexts of all the nodes in the model
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound(serialize = "E: Serialize", deserialize = "E: DeserializeOwned"))]
pub struct ModelCtx<E: ExtensionField> {
    pub(crate) nodes: ContextGraph<E>,
}

impl<E: ExtensionField> ModelCtx<E> {
    pub fn new(nodes: ContextGraph<E>) -> Self {
        Self { nodes }
    }

    /// Computes the shape step for each node in the model, so each layer knows the expected input and output shape
    /// to correctly verify the proof.
    pub fn shape_steps(
        &self,
        unpadded_input_shapes: &[Shape],
        padded_input_shapes: &[Shape],
    ) -> anyhow::Result<HashMap<NodeID, ShapeStep>> {
        self.nodes.forward_iter().try_fold(
            HashMap::<NodeID, ShapeStep>::new(),
            |mut acc, (node_id, node_ctx)| {
                // binds unpadded and padded shape that are passed to each port of this node_id
                let mut shapes_per_port = BTreeMap::<PortID, (Shape, Shape)>::new();
                for (_, edge) in self.nodes.neighbors(&node_id, Direction::Incoming) {
                    // either take the info from the input shapes or from the output shapes
                    // of a node previously visited
                    let (unpadded, padded) = match edge.source() {
                        Source::Node(source_id) => {
                            let step = acc
                                .get(source_id)
                                .ok_or(anyhow::anyhow!("Shapes for node {source_id} not found"))?;
                            (
                                step.unpadded_output_shape.as_slice(),
                                step.padded_output_shape.as_slice(),
                            )
                        }
                        Source::Input => (unpadded_input_shapes, padded_input_shapes),
                    };
                    // for each target port of each incoming edge for this node, we need to fetch the shape
                    for port in edge.ports().iter() {
                        // we check if the port of the input is less than the given total number of inputs
                        ensure!(
                            *port.source_port < unpadded.len(),
                            "Required input {} for node {}, but there are only {} inputs shapes",
                            *port.source_port,
                            edge.source(),
                            unpadded.len(),
                        );
                        shapes_per_port.insert(
                            port.target_port,
                            (
                                // take the shape the predecessor outputted at his source port
                                unpadded[*port.source_port].clone(),
                                padded[*port.source_port].clone(),
                            ),
                        );
                    }
                }
                // since shapes_per_port is a BTreeMap, the shapes are already sorted by the target port in increasing order
                let (un, pad): (Vec<_>, Vec<_>) = shapes_per_port
                    .into_iter()
                    .map(|(_, (unpadded, padded))| (unpadded, padded))
                    .unzip();
                // now call the layer with these to get the final shape step
                let shape_step = node_ctx.shape_step(&un, &pad);
                acc.insert(node_id, shape_step);
                Ok(acc)
            },
        )
    }
    /// Get the claims corresponding to the output edges of a node.
    /// Requires the input claims for the nodes of the model using the
    /// outputs of the current node, and the claims of the output
    /// tensors of the model
    /// The result is a map mapping the input port of the node_id the the vector of claims
    /// it receives from the proving. Indeed, an input port can receive multiple claims, if
    /// the output of a node is used in different subsequent layers.
    /// These will be aggregated as one but here we need to return all of them.
    pub(crate) fn claims_for_node<'a, 'b>(
        &self,
        node_id: &NodeID,
        claims_produced_by_layers: &'a HashMap<NodeID, Vec<Claim<E>>>,
        output_claims: &'b [Claim<E>],
    ) -> anyhow::Result<BTreeMap<PortID, Vec<&'a Claim<E>>>>
    where
        'b: 'a,
    {
        self.nodes.neighbors(node_id, Direction::Outgoing).try_fold(
            BTreeMap::new(),
            |mut acc, (_, edge)| {
                for port in edge.ports().iter() {
                    let claims = match edge.target() {
                        Target::Node(n) => claims_produced_by_layers
                            .get(n)
                            .ok_or(anyhow::anyhow!("Claims not found for node {n}"))?,
                        Target::Output => output_claims,
                    };
                    ensure!(
                        claims.len() > port.target_port.into(),
                        "Claims not found for port {}",
                        port.target_port
                    );
                    acc.entry(port.source_port)
                        .or_insert(Vec::new())
                        .push(claims.get(*port.target_port).unwrap());
                }
                Ok(acc)
            },
        )
    }

    /// Requires as inputs the contexts for all the nodes in the model
    /// and the set of claims for the input tensors of all the nodes of
    /// the model
    /// The PortID is the source_port of the input, e.g. the position of the input that corresponds
    /// to the claim.
    #[allow(clippy::type_complexity)]
    pub(crate) fn input_claims<'a>(
        &'a self,
        claims_produced_by_nodes: &'a HashMap<NodeID, Vec<Claim<E>>>,
    ) -> anyhow::Result<BTreeMap<NodeID, Vec<(PortID, &'a Claim<E>)>>> {
        let mut input_edges = BTreeSet::new();
        let mut result = BTreeMap::new();
        for (node_id, ports) in self.nodes.input_nodes() {
            let claims_for_node = claims_produced_by_nodes
                .get(node_id)
                .ok_or(anyhow::anyhow!("Claim not found for node {node_id}"))?;

            input_edges.extend(ports.iter().map(|port| port.source_port));
            for port in ports.iter() {
                // each claim is found into the target port, since that is what the layer produces
                // during verification.
                let tuple = (
                    port.source_port,
                    claims_for_node.get(*port.target_port).unwrap(),
                );
                result.entry(*node_id).or_insert(Vec::new()).push(tuple);
            }
        }
        ensure!(
            !result.is_empty(),
            "No input claims found for the set of nodes provided"
        );
        ensure!(
            *input_edges.first().unwrap() == PortID::from(0)
                && *input_edges.last().unwrap() == PortID::from(input_edges.len() - 1),
            "Not all input claims were found"
        );
        Ok(result)
    }
}
