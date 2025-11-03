use crate::{
    Shape,
    graph::{Graph, Node, NodeId, order_by_in_port},
    iop::context::ShapeStep,
    layers::LayerCtx,
};
use anyhow::Context;
use ff_ext::ExtensionField;
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use std::collections::HashMap;

pub type ContextGraph<N> = Graph<LayerCtx<N>, usize, usize, ()>;

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

    /// Computes the shape step for each node in the model, so each layer knows
    /// the expected input and output shape to correctly verify the proof.
    pub fn shape_steps(
        &self,
        unpadded_input_shapes: &[Shape],
        padded_input_shapes: &[Shape],
    ) -> anyhow::Result<HashMap<NodeId, ShapeStep>> {
        self.nodes.forward_iter().try_fold(
            HashMap::<NodeId, ShapeStep>::new(),
            |mut shapes, (node_id, node)| {
                match node {
                    Node::Inner(layer) => {
                        let (un, pad): (Vec<Shape>, Vec<Shape>) = order_by_in_port(
                            self.nodes
                                .incomings(node_id)
                                .flat_map(|(_, e)| e.feeds())
                                .map(|feed| {
                                    // fetch the input shapes for this node in
                                    // the register, that will have been
                                    // recursively filled with all the preceding
                                    // nodes output shapes as the graph is
                                    // traversed.
                                    let ShapeStep {
                                        unpadded_output_shape,
                                        padded_output_shape,
                                        ..
                                    } = shapes
                                        .get(&feed.source.node_id)
                                        .with_context(|| {
                                            format!("fetching shape step for {:?}", feed.source)
                                        })
                                        .unwrap();
                                    (
                                        feed.target,
                                        (
                                            unpadded_output_shape[feed.source.port].clone(),
                                            padded_output_shape[feed.source.port].clone(),
                                        ),
                                    )
                                }),
                        )
                        .unzip();
                        shapes.insert(node_id, layer.shape_step(&un, &pad)?);
                    }
                    Node::Input(i) => {
                        shapes.insert(
                            node_id,
                            ShapeStep {
                                unpadded_input_shape: vec![],
                                unpadded_output_shape: vec![unpadded_input_shapes[*i].clone()],
                                padded_input_shape: vec![],
                                padded_output_shape: vec![padded_input_shapes[*i].clone()],
                            },
                        );
                    }
                    Node::Output(_) => {}
                }
                Ok(shapes)
            },
        )
    }
}
