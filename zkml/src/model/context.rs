use crate::{
    Shape,
    graph::{Direction, Graph, Node, NodeId, order_by_in_port},
    iop::{
        chunking::{ChunkingStrategy, ModelChunk},
        context::ShapeStep,
    },
    layers::LayerCtx,
};
use anyhow::{Context, ensure};
use ff_ext::ExtensionField;
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use std::collections::{BTreeMap, BTreeSet, HashMap};

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

    pub(crate) fn split_in_chunks<S: ChunkingStrategy>(
        &self,
        num_chunks: Option<usize>,
        strategy: &S,
    ) -> anyhow::Result<Vec<ModelChunk>> {
        ModelChunk::build_chunks(self, num_chunks, strategy)
    }

    /// This method allows to check that the given chunks are consistent with the model graph in `self`.
    /// In a nutshell, consistency means that:
    /// - the chunks form a partition of the model graph
    /// - the nodes in all the chunks have the same edges as in the original model graph
    /// - the incoming and outgoing edges of the chunk are coherent with the graph structure
    /// - the incoming and outgoing edges are correctly linked to the corresponding chunks
    ///
    /// This method is useful for the verifier to check that the prover is not providing chunks
    /// for a different model than `self`, and that chunks are built coherently with the model.
    /// Note that this method does not guarantee to the verifier that these were the actual chunks
    /// used by the prover; this will be prevented by forcing the prover to add some data about
    /// the chunks in the transcript.
    pub(crate) fn check_model_chunking<'a>(
        &self,
        chunks: impl Iterator<Item = &'a ModelChunk>,
    ) -> anyhow::Result<()> {
        // collect each chunk in a map, altogether with its id
        let chunks = chunks
            .map(|chunk| (chunk.chunk_id, chunk))
            .collect::<BTreeMap<_, _>>();
        // check that the chunks are a partition of the model graph
        // first, compute the set of nodes found in all the chunks
        let nodes_in_chunks =
            chunks
                .values()
                .try_fold(BTreeSet::new(), |mut node_ids, chunk| {
                    chunk.subgraph.nodes().try_for_each(|(node_id, node)| {
                        let not_yet_inserted = node_ids.insert(*node_id);
                        if node.is_inner() {
                            // we need to ensure that each inner node is found only once across all the chunks
                            ensure!(
                                not_yet_inserted,
                                "Node {node_id} found multiple times in model chunks"
                            );
                        }
                        Ok(())
                    })?;
                    anyhow::Ok(node_ids)
                })?;

        // check that the nodes in the model are the same as `node_in_chunks`
        let model_node_ids = self
            .nodes
            .nodes()
            .map(|(node_id, _)| *node_id)
            .collect::<BTreeSet<_>>();
        ensure!(
            nodes_in_chunks == model_node_ids,
            "The chunks do not form a partition of the model graph"
        );
        // check that all edges found in the chunks are the same as in the model graph
        let edges_in_chunks = chunks
            .values()
            .flat_map(|chunk| chunk.subgraph.edges())
            .collect::<BTreeMap<_, _>>();
        let edges_in_model = self.nodes.edges().collect::<BTreeMap<_, _>>();
        ensure!(
            edges_in_chunks == edges_in_model,
            "The edges in the chunks do not match the model graph"
        );
        // for each node in each chunk, check that all the edges are the same as in the model graph
        chunks.values().try_for_each(|chunk| {
            chunk.subgraph.nodes().try_for_each(|(node_id, _)| {
                let edges_in_chunk = chunk
                    .subgraph
                    .neighbors(*node_id, Direction::Any)
                    .collect::<BTreeMap<_, _>>();
                let edges_in_model = self
                    .nodes
                    .neighbors(*node_id, Direction::Any)
                    .collect::<BTreeMap<_, _>>();
                ensure!(
                    edges_in_chunk == edges_in_model,
                    "The edges for node {node_id} in the chunk do not match the model graph"
                );
                Ok(())
            })
        })?;

        // check that the incoming and outgoing edges of each chunk are correct and grouped accordingly
        let chunk_for_node = ModelChunk::node_to_chunk_map(chunks.values().copied());

        chunks.values().try_for_each(|chunk| {
            let built_incoming_edges = chunk.build_incoming_grouped_edges(&chunk_for_node)?;
            ensure!(
                built_incoming_edges == chunk.incoming_edges,
                "The incoming edges for chunk {} are not correctly grouped",
                chunk.chunk_id
            );
            let built_outgoing_edges = chunk.build_outgoing_grouped_edges(&chunk_for_node)?;
            ensure!(
                built_outgoing_edges == chunk.outgoing_edges,
                "The outgoing edges for chunk {} are not correctly grouped",
                chunk.chunk_id
            );
            Ok(())
        })?;

        // check that incoming and outgoing edges among linked chunks are consistent
        ModelChunk::check_edges_group_consistency(&chunks)
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
