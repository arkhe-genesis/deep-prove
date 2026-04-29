use std::{
    borrow::Borrow,
    collections::{BTreeMap, BTreeSet, HashMap, HashSet},
    iter::repeat,
};

use anyhow::{anyhow, ensure};
use ark_ff::PrimeField;
use dp_crypto::{
    arkyper::{
        CommitmentScheme,
        transcript::{AppendToTranscript, Transcript},
    },
    poly::dense::DensePolynomial,
};
use itertools::Itertools;
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use tracing::{debug, warn};

use crate::{
    Claim, Element, InitTranscript, Shape,
    graph::{
        Direction, Edge, EdgeId, Graph, Node, NodeId, NodeInput, NodeOutput, PortId, PortLink,
        PortType, order_by_in_port,
    },
    iop::{context::ShapeStep, prover::ModelLayersRef},
    layers::{
        LayerCtx,
        provable::{OpInfo, Splittable},
        recombination::RecombinationLayer,
        split::SplitLayer,
        transformer::positional::PositionalCtx,
    },
    lookup::context::LookupContext,
    model::{Model, ModelCtx, Trace, context::shape_steps_for_graph, trace::SplittedNodesInfo},
    padding::PaddingMode,
    poly_commit::verifier::VerifierCommitment,
    tensor::TensorTypeParam,
};

/// Default unique chunk identifier.
#[derive(
    Debug,
    Default,
    Copy,
    Clone,
    Hash,
    Serialize,
    Deserialize,
    PartialOrd,
    Ord,
    derive_more::From,
    derive_more::Into,
    derive_more::Display,
    derive_more::Deref,
    PartialEq,
    Eq,
)]
#[display("Chunk({_0})")]
pub struct ChunkID(pub usize);

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ChunkedLayer {
    pub(crate) original_node_id: NodeId,
    pub(crate) chunk_number: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum ChunkedNode {
    OriginalNode(()),
    ChunkedLayer(ChunkedLayer),
    SplitLayer(SplitLayer),
    RecombinationLayer(RecombinationLayer),
}

#[derive(Clone, Debug, Serialize, Deserialize, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub struct ChunkedIO<const IO_TYPE: u8> {
    pub(crate) io_id: usize,
    pub(crate) chunk_id: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize, Hash, PartialEq, Eq)]
pub enum ChunkedIONode<const IO_TYPE: u8> {
    OriginalNode(usize),
    Chunked(ChunkedIO<IO_TYPE>),
}

impl<const IO_TYPE: u8> From<&ChunkedIONode<IO_TYPE>> for ChunkedIO<IO_TYPE> {
    fn from(value: &ChunkedIONode<IO_TYPE>) -> Self {
        match value {
            ChunkedIONode::OriginalNode(io_id) => io_id.into(),
            ChunkedIONode::Chunked(io) => io.clone(),
        }
    }
}

impl<const IO_TYPE: u8> From<&usize> for ChunkedIO<IO_TYPE> {
    fn from(value: &usize) -> Self {
        ChunkedIO {
            io_id: *value,
            chunk_id: 0, // this input/output is not chunked, so it's like there is a single chunk
        }
    }
}

/// Map each input/output model being split in multiple chunks to the corresponding inputs/outputs nodes
/// employed to represent each chunk
pub(crate) type SplittedIOInfo = HashMap<usize, Vec<NodeId>>;

pub type ChunkedOutput = ChunkedIO<{ PortType::Output as u8 }>;
pub type ChunkedOutNode = ChunkedIONode<{ PortType::Output as u8 }>;
pub type ChunkedInput = ChunkedIO<{ PortType::Input as u8 }>;
pub type ChunkedInNode = ChunkedIONode<{ PortType::Input as u8 }>;

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct SplittedNode {
    // Map the id of each horizontal chunk for the splitted node to the corresponding node processing that chunk
    pub(crate) new_nodes: BTreeMap<usize, NodeId>,
    pub(crate) split_layer: Option<(NodeId, SplitLayer)>,
    pub(crate) recombination_layer: Option<(NodeId, RecombinationLayer)>,
}

pub(crate) type ChunkedGraph = Graph<ChunkedNode, ChunkedInNode, ChunkedOutNode, ()>;

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct SplittedNodes {
    pub(crate) inner_nodes: HashMap<NodeId, SplittedNode>,
    pub(crate) inputs: SplittedIOInfo,
    pub(crate) outputs: SplittedIOInfo,
}

/// Represents a chunk of the model context. Each chunk is a subgraph of the whole model.
/// The edges in this subgraph are either:
/// - `internal edges`: edges where both the source and target nodes belong to the chunk
/// - `incoming_edges`: edges where the target node belongs to the chunk but the source node
///   belongs to another chunk
/// - `outgoing_edges`: edges where the source node belongs to the chunk but the target node
///   belongs to another chunk
/// - `model_input_edges`: input edges of the whole model whose target node belongs to the chunk
/// - `model_output_edges`: output edges of the whole model whose source node belongs to the chunk
///
/// The `incoming_edges` and `outgoing_edges` are grouped according to the chunk where the source
/// or target node belongs to, respectively.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ModelChunk {
    // set of nodes of the model in this chunk
    pub(crate) subgraph: ChunkedGraph,
    pub(crate) chunk_id: ChunkID,
    // set of incoming edges for the chunk: these are the incoming
    // edges of nodes in the chunk whose source nodes belong to other chunks.
    // Each output edge is paired with the source chunk, identified by
    // its `ChunkID`
    pub(crate) incoming_edges: BoundaryEdges,
    // set of outgoing edges for the chunk: these are the outgoing
    // edges of nodes in the chunk whose target nodes belong to other chunks.
    // Each output edge is paired with the destination chunk, identified by
    // its `ChunkID`
    pub(crate) outgoing_edges: BoundaryEdges,
    // This map keeps track of the node of the original nodes being replaced by horizontal chunks.
    // It maps the id of the original node to the ids of the chunked nodes
    pub(crate) replaced_nodes: HashMap<NodeId, Vec<NodeId>>,
}

pub(crate) type BoundaryEdges = BTreeMap<EdgeId, ChunkID>;

// Specify whether a group of boundary edges of a chunk are incoming or outgoing edges
#[derive(Clone, Copy, Debug)]
pub(crate) enum BoundaryEdgeType {
    Incoming,
    Outgoing,
}

impl ModelChunk {
    pub(crate) fn from_subgraph(subgraph: ChunkedGraph, chunk_id: usize) -> Self {
        Self {
            subgraph,
            chunk_id: chunk_id.into(),
            ..Default::default()
        }
    }

    /// Utility method to compute a map mapping each node in the provided chunks to the chunk the node belongs to
    pub(crate) fn node_to_chunk_map<'a>(
        chunks: impl Iterator<Item = &'a ModelChunk>,
    ) -> HashMap<NodeId, ChunkID> {
        chunks
            .flat_map(|chunk| {
                chunk
                    .subgraph
                    .nodes()
                    .map(|(node_id, _)| (*node_id, chunk.chunk_id))
            })
            .collect()
    }

    /// Group the `incoming_edges` of the chunk according to the chunk where their source node belongs to
    pub(crate) fn build_incoming_boundary_edges(
        &self,
        chunk_for_node: &HashMap<NodeId, ChunkID>,
    ) -> anyhow::Result<BoundaryEdges> {
        self.subgraph
            .edges()
            .filter_map(|(edge_id, edge)| {
                let source_node_id = edge.source();
                self.subgraph.node(source_node_id).is_none().then(|| {
                    let source_chunk = chunk_for_node
                        .get(&source_node_id)
                        .ok_or(anyhow!("Source node {source_node_id} not found in chunks"))?;
                    Ok((*edge_id, *source_chunk))
                })
            })
            .collect()
    }

    /// Group the `outgoing_edges` of the chunk according to the chunk where their target node belongs to
    pub(crate) fn build_outgoing_boundary_edges(
        &self,
        chunk_for_node: &HashMap<NodeId, ChunkID>,
    ) -> anyhow::Result<BoundaryEdges> {
        self.subgraph
            .edges()
            .filter_map(|(edge_id, edge)| {
                let target_node_id = edge.target();
                self.subgraph.node(target_node_id).is_none().then(|| {
                    let target_chunk = chunk_for_node
                        .get(&target_node_id)
                        .ok_or(anyhow!("Target node {target_node_id} not found in chunks"))?;
                    Ok((*edge_id, *target_chunk))
                })
            })
            .collect()
    }

    /// Utility method to check that each input boundary edge in a chunk is paired with an output
    /// boundary edge in another chunk, and viceversa.
    pub(crate) fn check_boundary_edges_consistency<T: Borrow<ModelChunk>>(
        chunks: &BTreeMap<ChunkID, T>,
    ) -> anyhow::Result<()> {
        chunks.values().try_for_each(|chunk| {
            let chunk = chunk.borrow();
            let chunk_id = chunk.chunk_id;
            chunk.incoming_edges.iter().try_for_each(|(edge_id, source_chunk_id)| {
                let source_chunk = chunks.get(source_chunk_id)
                    .ok_or(anyhow!("Source chunk {source_chunk_id} not found"))?;
                let corresponding_chunk = source_chunk.borrow().outgoing_edges.get(edge_id)
                    .ok_or(anyhow!("Input boundary edge {edge_id} for chunk {chunk_id} not found among outgoing boundary edges of chunk {source_chunk_id}"))?;
                ensure!(
                    *corresponding_chunk == chunk_id,
                    "Input boundary edge {edge_id} for chunk {chunk_id} is linked to chunk {source_chunk_id}, but the corresponding outgoing boundary edge in chunk {source_chunk_id} is linked to chunk {corresponding_chunk}"
                );
                Ok(())
            })?;
            chunk.outgoing_edges.iter().try_for_each(|(edge_id, target_chunk_id)| {
                let target_chunk = chunks.get(target_chunk_id)
                    .ok_or(anyhow!("Target chunk {target_chunk_id} not found"))?;
                let corresponding_chunk = target_chunk.borrow().incoming_edges.get(edge_id)
                    .ok_or(anyhow!("Output boundary edge {edge_id} for chunk {chunk_id} not found among incoming boundary edges of chunk {target_chunk_id}"))?;
                ensure!(
                    *corresponding_chunk == chunk_id,
                    "Output boundary edge {edge_id} for chunk {chunk_id} is linked to chunk {target_chunk_id}, but the corresponding incoming boundary edge in chunk {target_chunk_id} is linked to chunk {corresponding_chunk}"
                );
                Ok(())
            })
        })
    }

    pub(crate) fn build_chunks<F: PrimeField, S: ChunkingStrategy>(
        model: &ModelCtx<F>,
        num_chunks: Option<usize>,
        strategy: &S,
        next_node_id: impl Iterator<Item = NodeId>,
    ) -> anyhow::Result<(Vec<Self>, SplittedNodesInfo)> {
        let num_chunks = num_chunks
            .map(anyhow::Ok)
            .unwrap_or_else(|| strategy.ideal_num_chunks(model))?;
        let (chunk_subgraphs, splitted_nodes) = strategy.split(model, num_chunks, next_node_id)?;
        let mut chunks: BTreeMap<ChunkID, _> = chunk_subgraphs
            .into_iter()
            .enumerate()
            .map(|(i, subgraph)| (i.into(), ModelChunk::from_subgraph(subgraph, i)))
            .collect();

        let chunk_for_node = Self::node_to_chunk_map(chunks.values());

        splitted_nodes
            .inner_nodes
            .iter()
            .try_for_each(|(&original_node_id, splitted_node)| {
                splitted_node
                    .new_nodes
                    .iter()
                    .try_for_each(|(_, &node_id)| {
                        let chunk = chunk_for_node
                            .get(&node_id)
                            .ok_or(anyhow!("Chunk for node {node_id} not found"))?;
                        chunks
                            .get_mut(chunk)
                            .ok_or(anyhow!("Chunk {chunk} not found"))?
                            .replaced_nodes
                            .entry(original_node_id)
                            .or_default()
                            .push(node_id);
                        anyhow::Ok(())
                    })
            })?;

        splitted_nodes
            .inputs
            .iter()
            .try_for_each(|(input_id, new_input_nodes)| {
                let original_node_id = model.nodes.input_node_id(*input_id)?;
                new_input_nodes.iter().try_for_each(|&node_id| {
                    let chunk = chunk_for_node
                        .get(&node_id)
                        .ok_or(anyhow!("Chunk for node {node_id} not found"))?;
                    chunks
                        .get_mut(chunk)
                        .ok_or(anyhow!("Chunk {chunk} not found"))?
                        .replaced_nodes
                        .entry(original_node_id)
                        .or_default()
                        .push(node_id);
                    anyhow::Ok(())
                })
            })?;

        splitted_nodes
            .outputs
            .iter()
            .try_for_each(|(output_id, new_output_nodes)| {
                let original_node_id = model.nodes.output_node_ids()[*output_id];
                new_output_nodes.iter().try_for_each(|&node_id| {
                    let chunk = chunk_for_node
                        .get(&node_id)
                        .ok_or(anyhow!("Chunk for node {node_id} not found"))?;
                    chunks
                        .get_mut(chunk)
                        .ok_or(anyhow!("Chunk {chunk} not found"))?
                        .replaced_nodes
                        .entry(original_node_id)
                        .or_default()
                        .push(node_id);
                    anyhow::Ok(())
                })
            })?;

        // group input and output wires of the chunk according to the chunks where these
        // nodes are employed (i.e., this is computing chunk.grouped_input_wires and
        // chunk.grouped_output_wires)
        chunks.values_mut().try_for_each(|chunk| {
            chunk.incoming_edges = chunk.build_incoming_boundary_edges(&chunk_for_node)?;
            chunk.outgoing_edges = chunk.build_outgoing_boundary_edges(&chunk_for_node)?;
            anyhow::Ok(())
        })?;

        Self::check_boundary_edges_consistency(&chunks)?;

        Ok((
            chunks.into_values().collect(),
            SplittedNodesInfo { splitted_nodes },
        ))
    }

    pub(crate) fn edge(&self, id: &EdgeId) -> anyhow::Result<&Edge<()>> {
        self.subgraph
            .edge(id)
            .ok_or(anyhow!("Edge {id} not found in chunk {}", self.chunk_id))
    }

    pub(crate) fn boundary_edges(&self, group_type: BoundaryEdgeType) -> &BoundaryEdges {
        match group_type {
            BoundaryEdgeType::Incoming => &self.incoming_edges,
            BoundaryEdgeType::Outgoing => &self.outgoing_edges,
        }
    }

    /// Get the claims corresponding to all output ports of the node `node_id` in the
    /// chunked graph. Each output port can be linked to:
    /// - An input port of another node in the chunk (including output nodes of the model);
    ///   in this case, the claim for the port is found in `claims_produced_by_layers`
    /// - An input port of a node in another chunk (i.e., the output port belongs to an outgoing edge of the chunk);
    ///   in this case, the claim for the port is found in `chunk_output_claims`
    ///
    /// The result is a map mapping each output port of the node to a vector of claims.
    /// Indeed, an output port can receive multiple claims, if the output port is linked to
    /// different targets.
    #[allow(clippy::type_complexity)]
    pub(crate) fn claims_for_node<'a, 'b, F: PrimeField>(
        &self,
        node_id: NodeId,
        claims_by_layers: &'a HashMap<NodeInput, Claim<F>>,
        chunk_output_claims: &'b HashMap<NodeOutput, Claim<F>>,
    ) -> anyhow::Result<BTreeMap<PortId, Vec<&'a Claim<F>>>>
    where
        'b: 'a,
    {
        // Save the set of node output ports claims already inserted from `chunk_output_claims`,
        // to avoid inserting these claims twice
        let mut inserted_out_ports = HashSet::new();
        self.subgraph.outgoing_feeds(node_id).into_iter().try_fold(
            BTreeMap::new(),
            |mut out_map, feed| {
                let claim = match claims_by_layers.get(&feed.target) {
                    Some(claim) => Some(claim),
                    None => {
                        // target node is not in this chunk, so we fetch the claim from `chunk_output_claims`
                        let output_port = NodeOutput::new(*node_id, feed.source.port);
                        let claim =
                            chunk_output_claims
                                .get(&output_port)
                                .ok_or(anyhow::anyhow!(
                                    "No claim found for source port {output_port:?}"
                                ))?;
                        if inserted_out_ports.insert(feed.source.port) {
                            // we never insert a claim for this source port, so we return it to insert it in `output_map`
                            Some(claim)
                        } else {
                            None
                        }
                    }
                };
                if let Some(claim) = claim {
                    out_map
                        .entry(feed.source.port)
                        .or_insert(Vec::new())
                        .push(claim);
                }
                Ok(out_map)
            },
        )
    }

    /// Compute the polynomials to be committed for the incoming or outgoing edges of the chunk;
    // the polynomials being committed are the MLEs of the tensors propagated through the edges.
    /// The `BoundaryEdgeType` parameter specifies whether the polynomials are computed for incoming
    /// or outgoing edges
    pub(crate) fn to_be_committed_polys<F: PrimeField>(
        &self,
        full_trace: &Trace<Element>,
        edge_type: BoundaryEdgeType,
    ) -> anyhow::Result<BTreeMap<NodeId, DensePolynomial<'static, F>>> {
        let chunk_id = self.chunk_id;
        self.boundary_edges(edge_type)
            .keys()
            .try_fold(BTreeMap::new(), |mut mles, edge_id| {
                let edge = self.edge(edge_id)?;
                let node_id = match edge_type {
                    BoundaryEdgeType::Incoming => edge.target(),
                    BoundaryEdgeType::Outgoing => edge.source(),
                };
                let step_data = &full_trace.get_step(&node_id).ok_or(anyhow!(
                    "Node {node_id} not found in trace for chunk {chunk_id}"
                ))?;
                edge.ports().iter().try_for_each(|port| {
                    let (tensor, port) = match edge_type {
                        BoundaryEdgeType::Incoming => (
                            step_data.input_tensor_at(port.target_port.into())?,
                            // if this is an incoming edge, we need to get the output port
                            // of the source of the edge (which is in the other chunk) in order
                            // to detect whether we have already added the MLE for this port
                            NodeOutput::new(edge.source(), port.source_port),
                        ),
                        BoundaryEdgeType::Outgoing => (
                            step_data.output_tensor_at(port.source_port.into())?,
                            NodeOutput::new(node_id, port.source_port),
                        ),
                    };
                    let poly_id = Self::compute_commitment_id(port);
                    mles.entry(poly_id.into()).or_insert_with(|| {
                        // we add the MLE to the set of MLEs to be committed only if we haven't already
                        // added an MLE for this `NodeOutput` port. Otherwise, we would be adding duplicate MLEs
                        tensor.pad_next_power_of_two().to_field_mle()
                    });
                    anyhow::Ok(())
                })?;
                anyhow::Ok(mles)
            })
    }

    // The output edges of the whole model whose source node is in the chunk
    pub(crate) fn model_outputs_in_chunk(&self) -> anyhow::Result<Vec<EdgeId>> {
        Ok(self
            .subgraph
            .edges()
            .filter_map(|(edge_id, edge)| {
                let target_node = self.subgraph.node(edge.target());
                target_node.and_then(|node| node.is_output().then_some(*edge_id))
            })
            .collect())
    }

    // The input edges of the whole model whose target node is in the chunk
    pub(crate) fn model_inputs_in_chunk(&self) -> anyhow::Result<Vec<EdgeId>> {
        Ok(self
            .subgraph
            .edges()
            .filter_map(|(edge_id, edge)| {
                let source_node = self.subgraph.node(edge.source());
                source_node.and_then(|node| node.is_input().then_some(*edge_id))
            })
            .collect())
    }

    /// Add to the transcript the chunk data that allows to identify the splitting points of the model
    /// to produce the current chunk; this chunk data corresponds to:
    /// - the chunk id
    /// - the incoming boundary edges
    /// - the outgoing boundary edges
    /// - the model input edges whose target node is in the chunk
    /// - the model output edges whose source node is in the chunk
    pub(crate) fn add_chunk_data_to_transcript<T: Transcript>(
        &self,
        transcript: &mut T,
    ) -> anyhow::Result<()> {
        // closure to append a set of grouped incoming/outgoing edges to the transcript
        let append_boundary_edges = |edges: &BoundaryEdges, t: &mut T| {
            // we append `edges.len()` and then the pair of `(edge_id, chunk_id)` for each edge to the transcript
            let append_payload = edges
                .len()
                .to_le_bytes()
                .into_iter()
                .chain(edges.iter().flat_map(|(edge_id, chunk_id)| {
                    edge_id
                        .to_le_bytes()
                        .into_iter()
                        .chain(chunk_id.to_le_bytes())
                }))
                .collect_vec();
            t.append_bytes(&append_payload);
        };
        // append chunk id
        transcript.append_bytes(&self.chunk_id.to_le_bytes());
        // append incoming edges, grouped by source chunk
        transcript.append_bytes("incoming".as_bytes());
        append_boundary_edges(&self.incoming_edges, transcript);
        // append model input edges of this chunk
        transcript.append_bytes("inputs".as_bytes());
        self.model_inputs_in_chunk()?
            .into_iter()
            .for_each(|edge_id| transcript.append_bytes(&edge_id.to_le_bytes()));
        // append outgoing edges, grouped by source chunk
        transcript.append_bytes("incoming".as_bytes());
        append_boundary_edges(&self.outgoing_edges, transcript);
        // append model output edges of this chunk
        transcript.append_bytes("outputs".as_bytes());
        self.model_outputs_in_chunk()?
            .into_iter()
            .for_each(|edge_id| transcript.append_bytes(&edge_id.to_le_bytes()));
        Ok(())
    }

    // derive the trace for chunk `self` from `full_trace`
    pub(crate) fn chunk_trace(
        &self,
        full_trace: &Trace<Element>,
    ) -> anyhow::Result<Trace<Element>> {
        let steps = self
            .subgraph
            .inner_nodes()
            .map(|(node_id, _)| {
                Ok((
                    node_id,
                    full_trace
                        .steps
                        .get(&node_id)
                        .ok_or(anyhow!("Node {node_id} not found in full trace"))?
                        .clone(),
                ))
            })
            .collect::<anyhow::Result<HashMap<_, _>>>()?;
        Ok(Trace {
            steps,
            input: vec![],        // they are unused in a chunk prover
            output: vec![],       // they are unused in a chunk prover
            ..Default::default()  // they are unused in a chunk prover
        })
    }

    pub(crate) fn chunk_layers<'a>(&self, model: &'a Model<Element>) -> ModelLayersRef<'a> {
        model
            .graph()
            .inner_nodes()
            .filter(|(node_id, _layer)| {
                // retain the node if it is in the current chunk
                self.subgraph.node(*node_id).is_some() || self.replaced_nodes.contains_key(node_id)
            })
            .collect()
    }

    // Returns the lookup context containing only the data relevant to the current subgraph / chunk.
    pub(crate) fn chunk_lookup_ctx(&self, lookup_ctx: &LookupContext) -> LookupContext {
        LookupContext {
            tables: lookup_ctx
                .tables
                .iter()
                .filter_map(|(table_type, node_ids)| {
                    let chunk_node_ids = node_ids
                        .iter()
                        .filter_map(|&node_id| {
                            if self.subgraph.node(node_id).is_some() {
                                Some(vec![node_id])
                            } else {
                                self.replaced_nodes
                                    .get(&node_id)
                                    .map(|node_ids| node_ids.to_vec())
                            }
                        })
                        .flatten()
                        .collect_vec();
                    if !chunk_node_ids.is_empty() {
                        Some((*table_type, chunk_node_ids))
                    } else {
                        None
                    }
                })
                .collect(),
        }
    }

    // check that the commitments for the input and output groups of `self` are equal
    // to the commitments of the corresponding input/output groups of other chunks
    pub(crate) fn check_chunk_commitment_consistency<PCS: CommitmentScheme>(
        &self,
        chunk_commitments_by_id: &HashMap<ChunkID, &ChunkIOCommitments<VerifierCommitment<PCS>>>,
    ) -> anyhow::Result<()> {
        let chunk_id = self.chunk_id;
        let input_commitments = &chunk_commitments_by_id
            .get(&self.chunk_id)
            .ok_or(anyhow!(
                "No chunk commitments found for chunk {}",
                self.chunk_id
            ))?
            .inputs;
        self.incoming_edges.iter().try_for_each(|(edge_id, source_chunk_id)| {
            let edge = self.edge(edge_id)?;
            // compare the commitments associated to this input edge with corresponding commitments
            // associated to the output edge in an another chunk
            // first, we fetch the commitments of output edges for the source chunk of the current edge
            let output_commitments = &chunk_commitments_by_id
                .get(source_chunk_id)
                .ok_or(anyhow!(
                    "No chunk commitments found for chunk {}",
                    source_chunk_id
                ))?
                .outputs;
            // then, for each port in the edge, we compare the commitments associated to that port in both 
            // inputs commitments and output commitments
            edge.ports().iter().try_for_each(|port| {
                let commitment_id = Self::compute_commitment_id(
                    NodeOutput::new(edge.source(), port.source_port)
                ).into();
                let input_commitment = input_commitments
                    .get(&commitment_id)
                    .ok_or(anyhow!(
                        "No input commitment found for polynomial id {commitment_id} related to edge {edge:?} of chunk {chunk_id}"
                    ))?;
                let output_commitment = output_commitments
                    .get(&commitment_id)
                    .ok_or(anyhow!(
                        "No output commitment found for polynomial id {commitment_id} related to edge {edge:?} of source chunk {source_chunk_id}"
                    ))?;
                ensure!(
                    input_commitment == output_commitment,
                    "Inconsistent commitment found for polynomial id {commitment_id} related to edge {edge:?} between source chunk {source_chunk_id} and chunk {chunk_id}",
                );
                Ok(())
            })
        })?;

        let output_commitments = &chunk_commitments_by_id
            .get(&self.chunk_id)
            .ok_or(anyhow!(
                "No chunk commitments found for chunk {}",
                self.chunk_id
            ))?
            .outputs;
        self.outgoing_edges.iter().try_for_each(|(edge_id, target_chunk_id)| {
            let edge = self.edge(edge_id)?;
            // compare the commitments associated to this output edge with corresponding commitments
            // associated to the input edge in an another chunk
            // first, we fetch the commitments of input edges for the chunk of the target node of the current edge
            let input_commitments = &chunk_commitments_by_id
                .get(target_chunk_id)
                .ok_or(anyhow!(
                    "No chunk commitments found for chunk {}",
                    target_chunk_id
                ))?
                .inputs;
            // then, for each port in the edge, we compare the commitments associated to that port in both 
            // inputs commitments and output commitments
            edge.ports().iter().try_for_each(|port| {
                let commitment_id = Self::compute_commitment_id(
                    NodeOutput::new(edge.source(), port.source_port)
                ).into();
                let input_commitment = input_commitments
                    .get(&commitment_id)
                    .ok_or(anyhow!(
                        "No input commitment found for polynomial id {commitment_id} related to edge {edge:?} of target chunk {target_chunk_id}"
                    ))?;
                let output_commitment = output_commitments
                    .get(&commitment_id)
                    .ok_or(anyhow!(
                        "No output commitment found for polynomial id {commitment_id} related to edge {edge:?} of chunk {chunk_id}"
                    ))?;
                ensure!(
                    input_commitment == output_commitment,
                    "Inconsistent commitment found for polynomial id {commitment_id} related to edge {edge:?} between chunk {chunk_id} and target chunk {target_chunk_id}",
                );
                Ok(())
            })
        })
    }

    // extract claims from `claims_by_layer` corresponding to the group of incoming edges
    // of the chunk `self`. It returns one or more claims for each MLE of the incoming boundary
    // edges
    pub(crate) fn compute_input_boundary_edges_claims<F: PrimeField>(
        &self,
        claims_by_layer: &HashMap<NodeInput, Claim<F>>,
    ) -> anyhow::Result<BTreeMap<NodeId, Vec<Claim<F>>>> {
        self.incoming_edges
            .keys()
            .try_fold(BTreeMap::new(), |mut claims, edge_id| {
                let edge = self.edge(edge_id)?;
                let target_node_id = edge.target();
                edge.ports().iter().try_for_each(|port| {
                    let target_port = NodeInput::new(*target_node_id, port.target_port);
                    let claim = claims_by_layer
                        .get(&target_port)
                        .ok_or(anyhow!("Claims for target port {target_port:?} not found",))?;
                    let source_port = NodeOutput::new(edge.source(), port.source_port);
                    let poly_id = Self::compute_commitment_id(source_port);
                    claims
                        .entry(poly_id.into())
                        .or_insert(vec![])
                        .push(claim.clone());
                    anyhow::Ok(())
                })?;
                anyhow::Ok(claims)
            })
    }

    // extract claims from `claims_by_port` corresponding to the group of outgoing edges
    // of the chunk `self`. It returns one or more claims for each MLE of the outgoing boundary
    // edges
    pub(crate) fn compute_output_boundary_edges_claims<F: PrimeField>(
        &self,
        claims_by_port: &HashMap<NodeOutput, Claim<F>>,
    ) -> anyhow::Result<BTreeMap<NodeId, Vec<Claim<F>>>> {
        self.outgoing_edges
            .keys()
            .try_fold(BTreeMap::new(), |mut claims, edge_id| {
                let edge = self.edge(edge_id)?;
                let source_node_id = edge.source();
                edge.ports().iter().try_for_each(|port| {
                    let source_port = NodeOutput::new(*source_node_id, port.source_port);
                    let claim = claims_by_port
                        .get(&source_port)
                        .ok_or(anyhow!("Claim not found for source port: {source_port}"))?;
                    let poly_id = Self::compute_commitment_id(source_port);
                    claims
                        .entry(poly_id.into())
                        .or_insert(vec![])
                        .push(claim.clone());
                    anyhow::Ok(())
                })?;
                anyhow::Ok(claims)
            })
    }

    pub(crate) fn compute_commitment_id(port: NodeOutput) -> usize {
        let bytes = port
            .node_id
            .to_le_bytes()
            .into_iter()
            .chain(port.port.to_le_bytes())
            .collect::<Vec<u8>>();

        // Should be 32 bytes take the first 8
        let byte_array: [u8; 8] = <sha2::Sha256 as sha2::Digest>::digest(&bytes)[0..8]
            .try_into()
            .expect("slice with incorrect length");
        usize::from_be_bytes(byte_array)
    }

    pub(crate) fn shape_steps_for_chunk<F: PrimeField>(
        &self,
        unpadded_input_shapes: &HashMap<ChunkedInput, Shape>,
        padded_input_shapes: &HashMap<ChunkedInput, Shape>,
        shapes: &mut HashMap<NodeId, ShapeStep>,
        model: &ModelCtx<F>,
    ) -> anyhow::Result<()> {
        shape_steps_for_graph(
            &self.subgraph,
            unpadded_input_shapes,
            padded_input_shapes,
            shapes,
            |node_id, node, unpad, pad| {
                let split_layer = if let ChunkedNode::SplitLayer(split_layer) = node {
                    let mut split_layer = split_layer.clone();
                    // modify the unpadded input shapes to properly compute the padded output shapes
                    split_layer.unpadded_input_shapes = unpad.to_vec();
                    Some(LayerCtx::Split(split_layer))
                } else {
                    None
                };
                let recombination_layer = if let ChunkedNode::RecombinationLayer(rec_layer) = node {
                    Some(LayerCtx::Recombination(rec_layer.clone()))
                } else {
                    None
                };
                let layer_ctx = match node {
                    ChunkedNode::OriginalNode(_) => model
                        .nodes
                        .node(node_id)
                        .ok_or(anyhow!("Node {node_id} not found verifier context"))?
                        .as_inner()
                        .expect("Node {node_id} must be an inner node"),
                    ChunkedNode::ChunkedLayer(chunked_layer) => model
                        .nodes
                        .node(chunked_layer.original_node_id)
                        .ok_or(anyhow!(
                            "Node {} not found verifier context",
                            chunked_layer.original_node_id
                        ))?
                        .as_inner()
                        .unwrap_or_else(|| {
                            panic!(
                                "Node {} must be an inner node",
                                chunked_layer.original_node_id
                            )
                        }),
                    ChunkedNode::SplitLayer(_) => split_layer.as_ref().unwrap(),
                    ChunkedNode::RecombinationLayer(_) => recombination_layer.as_ref().unwrap(),
                };
                layer_ctx.shape_step(unpad, pad)
            },
        )
    }

    fn visit_chunks<'a, const FORWARD_VISIT: bool>(
        chunks: impl Iterator<Item = &'a ModelChunk>,
    ) -> anyhow::Result<Vec<&'a ModelChunk>> {
        let mut visited_chunks = HashSet::new();
        let mut result = Vec::new();
        let mut chunks = chunks
            .map(|chunk| (chunk.chunk_id, chunk))
            .collect::<BTreeMap<_, _>>();
        while result.len() < chunks.len() {
            let new_visited_chunks = if FORWARD_VISIT {
                chunks
                    .iter()
                    .rev() // rev because chunks with higher chunks ids are more likely to be the next ones to be visited
                    .filter_map(|(chunk_id, chunk)| {
                        if chunk
                            .incoming_edges
                            .values()
                            .all(|source_chunk_id| visited_chunks.contains(source_chunk_id))
                        {
                            visited_chunks.insert(*chunk_id);
                            Some(*chunk_id)
                        } else {
                            None
                        }
                    })
                    .collect_vec()
            } else {
                chunks
                    .iter()
                    .filter_map(|(chunk_id, chunk)| {
                        if chunk
                            .outgoing_edges
                            .values()
                            .all(|source_chunk_id| visited_chunks.contains(source_chunk_id))
                        {
                            visited_chunks.insert(*chunk_id);
                            Some(*chunk_id)
                        } else {
                            None
                        }
                    })
                    .collect_vec()
            };
            for chunk_id in new_visited_chunks {
                let chunk = chunks
                    .remove(&chunk_id)
                    .ok_or(anyhow!("Chunk {chunk_id} visited twice"))?;
                result.push(chunk);
            }
        }
        Ok(result)
    }

    /// Return the chunks provided as input, sorted according to the dependencies of the inputs/outputs boundary
    /// edges of each chunk from other chunks, when traversing the graph model in forward direction, .i.e, from inputs to outputs
    pub(crate) fn visit_chunks_forward<'a>(
        chunks: impl Iterator<Item = &'a ModelChunk>,
    ) -> anyhow::Result<Vec<&'a ModelChunk>> {
        Self::visit_chunks::<true>(chunks)
    }

    /// Return the chunks provided as input, sorted according to the dependencies of the inputs/outputs boundary
    /// edges of each chunk from other chunks, when traversing the graph model in backward direction, .i.e, from outputs to inputs
    #[allow(unused)]
    pub(crate) fn visit_chunks_backward<'a>(
        chunks: impl Iterator<Item = &'a ModelChunk>,
    ) -> anyhow::Result<Vec<&'a ModelChunk>> {
        Self::visit_chunks::<false>(chunks)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(crate) struct ChunkIOCommitments<C> {
    pub(crate) inputs: BTreeMap<NodeId, C>,
    pub(crate) outputs: BTreeMap<NodeId, C>,
}

impl<C> Default for ChunkIOCommitments<C> {
    fn default() -> Self {
        Self {
            inputs: BTreeMap::new(),
            outputs: BTreeMap::new(),
        }
    }
}

impl<C> ChunkIOCommitments<C> {
    pub(crate) fn add_to_transcript<PCS: CommitmentScheme, T: Transcript>(
        &self,
        chunk_id: ChunkID,
        transcript: &mut T,
    ) where
        C: Borrow<VerifierCommitment<PCS>>,
    {
        self.inputs.iter().for_each(|(node_id, commitment)| {
            let comm = commitment.borrow();
            add_chunk_commitments_to_transcript::<T, _>(
                chunk_id,
                *node_id,
                comm,
                transcript,
                BoundaryEdgeType::Incoming,
            )
        });
        self.outputs.iter().for_each(|(node_id, commitment)| {
            let comm = commitment.borrow();
            add_chunk_commitments_to_transcript::<T, _>(
                chunk_id,
                *node_id,
                comm,
                transcript,
                BoundaryEdgeType::Outgoing,
            )
        });
    }
}

pub(crate) fn add_chunk_commitments_to_transcript<T: Transcript, A: AppendToTranscript>(
    chunk_id: ChunkID,
    node_id: NodeId,
    commitment: &A,
    transcript: &mut T,
    group_type: BoundaryEdgeType,
) {
    let commitment_descriptor = match group_type {
        BoundaryEdgeType::Incoming => format!("Input: {chunk_id}->{node_id}"),
        BoundaryEdgeType::Outgoing => format!("Output: {chunk_id}->{node_id}"),
    };
    transcript.append_bytes(commitment_descriptor.as_bytes());
    commitment.append_to_transcript(transcript);
}

pub trait ChunkingStrategy: Clone + Serialize + DeserializeOwned {
    /// Return the ideal number of chunks to split the model into;
    fn ideal_num_chunks<F: PrimeField>(&self, model: &ModelCtx<F>) -> anyhow::Result<usize>;

    /// Split the set of nodes in `num_chunks` chunks of consecutive nodes for proving.
    /// The `next_node_id` is an iterator to get node ids that can be employed by the implementor in
    /// case it needs to instantiate new nodes in the model graph for the chunking strategy
    fn split<F: PrimeField>(
        &self,
        model: &ModelCtx<F>,
        num_chunks: usize,
        next_node_id: impl Iterator<Item = NodeId>,
    ) -> anyhow::Result<(Vec<ChunkedGraph>, SplittedNodes)>;
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DefaultChunkingStrategy {
    unpadded_input_shapes: Vec<Shape>,
}

impl<D: TensorTypeParam> From<&Trace<D>> for DefaultChunkingStrategy {
    fn from(value: &Trace<D>) -> Self {
        Self {
            unpadded_input_shapes: value
                .inputs()
                .iter()
                .map(|input| input.unpadded_shape().clone())
                .collect(),
        }
    }
}

impl DefaultChunkingStrategy {
    pub fn new(unpadded_input_shapes: Vec<Shape>) -> Self {
        Self {
            unpadded_input_shapes,
        }
    }

    fn max_num_vertical_chunks<F: PrimeField>(model: &ModelCtx<F>) -> usize {
        // define a constant `NUM_NODES_PER_CHUNK` that specifies the ideal number of
        // nodes per chunk to be proven. The number of chunks is then computed by
        // ensuring that each chunk has at most `NUM_NODES_PER_CHUNK` nodes
        const NUM_NODES_PER_CHUNK: usize = 3;
        let num_nodes = model.nodes.inner_nodes_count();
        // return num chunks as `ceil(num_nodes / NUM_NODES_PER_CHUNK)`
        num_nodes.div_ceil(NUM_NODES_PER_CHUNK)
    }
}

fn get_input_shapes_for_node<F: PrimeField>(
    model: &ModelCtx<F>,
    node_id: NodeId,
    input_shapes: &HashMap<NodeOutput, Shape>,
) -> Vec<Shape> {
    order_by_in_port(model.nodes.incoming_feeds(node_id).into_iter().map(|feed| {
        (
            NodeInput::new(node_id, feed.target.port),
            input_shapes[&feed.source].clone(),
        )
    }))
    .collect_vec()
}

fn compute_num_chunks_for_group<'a, F: PrimeField>(
    group_nodes: impl Iterator<Item = &'a NodeId>,
    max_num_chunks_for_group: usize,
    model: &ModelCtx<F>,
    input_shapes: &HashMap<NodeOutput, Shape>,
) -> anyhow::Result<Option<usize>> {
    if max_num_chunks_for_group < 2 {
        return Ok(None);
    }
    group_nodes.map(|node_id| {
        let input_shapes = get_input_shapes_for_node(model, *node_id, input_shapes);
        let SplitLayer {
            num_chunks,
            ..
        } = SplitLayer::new_from_input_shapes(max_num_chunks_for_group, &input_shapes)?;
        num_chunks.into_iter().try_fold(None, |num_chunks, item| {
            if let Some(n) = num_chunks {
                ensure!(n == item, "Inconsistent number of chunks for different inputs of node {node_id}: {n} vs {item}");
            }
            Ok(Some(item))
        }).map(|x| (node_id, x))
    }).try_fold(None, |num_chunks_for_group, item| {
        let (node_id, num_chunks_for_node) = item?;
        let num_chunks_for_node = num_chunks_for_node.ok_or(
            anyhow!("No number of chunks foudn for node {node_id}")
        )?;
        if let Some(n) = num_chunks_for_group {
            ensure!(n == num_chunks_for_node, "Inconsistent number of chunks for different nodes in the same group: {n} vs {num_chunks_for_node}");
        }
        Ok(Some(num_chunks_for_node))
    }).map(|num_chunks_for_group|
        // ensure that number of chunks is at least 2, return None otherwise 
        num_chunks_for_group.and_then(|num_chunks|
            (num_chunks >= 2).then_some(num_chunks)
        )
    )
}

impl ChunkingStrategy for DefaultChunkingStrategy {
    fn ideal_num_chunks<F: PrimeField>(&self, model: &ModelCtx<F>) -> anyhow::Result<usize> {
        // define a constant `NUM_NODES_PER_CHUNK` that specifies the ideal number of
        // nodes per chunk to be proven. The number of chunks is then computed by
        // ensuring that each chunk has at most `NUM_NODES_PER_CHUNK` nodes
        const NUM_NODES_PER_CHUNK: usize = 3;
        let num_nodes = model.nodes.inner_nodes_count();
        // return num chunks as `ceil(num_nodes / NUM_NODES_PER_CHUNK)`
        Ok(num_nodes.div_ceil(NUM_NODES_PER_CHUNK))
    }

    fn split<F: PrimeField>(
        &self,
        model: &ModelCtx<F>,
        mut num_chunks: usize,
        mut next_node_id: impl Iterator<Item = NodeId>,
    ) -> anyhow::Result<(Vec<ChunkedGraph>, SplittedNodes)> {
        let max_num_vertical_chunks = Self::max_num_vertical_chunks(model);

        let mut shapes: HashMap<_, _> = model
            .nodes
            .input_node_ids()
            .into_iter()
            .zip(&self.unpadded_input_shapes)
            .map(|(input_node_id, input_shape)| {
                (NodeOutput::new(input_node_id, 0), input_shape.clone())
            })
            .collect();

        let horizontal_chunk_groups = if num_chunks > max_num_vertical_chunks {
            // we employ also horizontal chunking, if possible
            let splittable_layers = model
                .nodes
                .forward_inners()
                .filter_map(|(node_id, node)| {
                    let unpadded_input_shapes = get_input_shapes_for_node(model, node_id, &shapes);
                    let output_shapes = node.output_shapes(&unpadded_input_shapes, PaddingMode::NoPadding)
                        .expect("Shouldn't fail since we successfuly generated the context for this model");
                    shapes.extend(
                            output_shapes
                            .into_iter()
                            .enumerate()
                            .map(|(i, shape)| (NodeOutput::new(node_id, i), shape)),
                    );
                    node.ideal_num_chunks(&unpadded_input_shapes).and_then(|num_chunks| {
                        if num_chunks > 1 {
                            // A layer must be split in at least 2 chunks to be splittable
                            Some((node_id, num_chunks))
                        } else {
                            None
                        }
                    })
                })
                .collect::<HashMap<_,_>>();

            // now we group together consecutive splittable layers. More specifically, for the sake of simplicity,
            // for now we group together 2 splittable layers L1, L2 iff:
            // - All output wires of L1 are linked only to L2
            // - All input wires of L2 are linked only to L1
            let mut groups = vec![];
            let mut groups_by_layer = HashMap::new();
            for (node_id, _) in model.nodes.backward_iter().filter(|(node_id, node)| {
                node.is_inner() && splittable_layers.contains_key(node_id)
            }) {
                // check if this node is already in a group
                let group_id = if let Some(&group_id) = groups_by_layer.get(&node_id) {
                    group_id
                } else {
                    // instantiate a new group
                    let num_chunks = *splittable_layers
                        .get(&node_id)
                        .expect("Current node must be splittable");
                    groups.push((num_chunks, BTreeSet::from([node_id])));
                    groups_by_layer.insert(node_id, groups.len() - 1);
                    groups.len() - 1
                };
                // check if:
                // - the current node has only 1 preceding node
                // - the preceding node is splittable
                // - the current node is the only subsequent node of the preceding node
                // - the splitted input dimension of the preceding node is the same as the splitted input dimension of the current node
                // if these 4 conditions hold, then the preceding node can be placed in the same split group of current node
                let incoming_neighbors = model
                    .nodes
                    .neighbors(node_id, Direction::Incoming)
                    .map(|(_, edge)| edge.source())
                    .collect_vec();
                if incoming_neighbors.len() == 1 {
                    // we consider to merge splittable layers only if there is only one preceding layer for now
                    if let Some(&neighbor_num_chunks) =
                        splittable_layers.get(&incoming_neighbors[0])
                    {
                        let outgoing_neighbors = model
                            .nodes
                            .neighbors(incoming_neighbors[0], Direction::Outgoing)
                            .map(|(_, edge)| edge.target())
                            .collect_vec();
                        if outgoing_neighbors.len() == 1 {
                            ensure!(outgoing_neighbors[0] == node_id);
                            // check the splitted input dimension conditions
                            let neighbor_input_shapes =
                                get_input_shapes_for_node(model, incoming_neighbors[0], &shapes);
                            let node_input_shapes =
                                get_input_shapes_for_node(model, node_id, &shapes);
                            if
                                    neighbor_input_shapes.len() == node_input_shapes.len() &&
                                    neighbor_input_shapes.into_iter().zip(node_input_shapes.into_iter()).all(|(shape1, shape2)|
                                        shape1.dim(0) == shape2.dim(0) // for now, the splitted dimension is always the first one
                                    )
                                {
                                    // the 3 conditions are met, so the preceding node must be added to the current group
                                    let current_group = &mut groups[group_id];
                                    current_group.1.insert(incoming_neighbors[0]);
                                    current_group.0 = current_group.0.min(neighbor_num_chunks);
                                    groups_by_layer.insert(incoming_neighbors[0], group_id);
                                }
                        }
                    }
                }
            }

            // compute the number of available chunks for horizontal chunks
            let mut num_available_chunks = num_chunks - max_num_vertical_chunks;
            // now can determine how many horizontal chunks we use: we sort the groups by the biggest number
            // of chunks and by the number of nodes in the group, and we assign available chunks to all these
            // groups
            groups.sort_by(|a, b| {
                if a.0 == b.0 {
                    // compare number of nodes in the group
                    a.1.len().cmp(&b.1.len())
                } else {
                    a.0.cmp(&b.0)
                }
            });
            let mut num_split_groups = 0;

            let groups = groups
                .into_iter()
                .rev()
                .filter_map(|mut group| {
                    if num_available_chunks < 2 {
                        // doesn't make sense to chunk further if we have less than 2 chunks available
                        None
                    } else {
                        let max_num_chunks_for_group = num_available_chunks.min(group.0);
                        let maybe_num_chunks_for_group = compute_num_chunks_for_group(
                            group.1.iter(),
                            max_num_chunks_for_group,
                            model,
                            &shapes,
                        );
                        if let Err(err) = maybe_num_chunks_for_group {
                            return Some(Err(err));
                        }
                        let Some(num_chunks_for_group) = maybe_num_chunks_for_group.unwrap() else {
                            // no valid number of chunks possible for this group, so we skip it
                            return None;
                        };
                        group.0 = num_chunks_for_group;
                        num_available_chunks -= group.0;
                        num_split_groups += 1;
                        // check that we have enough chunks available for the rest of the layers:
                        // indeed, if we have `num_split_groups` groups, we will necessarily have `num_split_groups + 1`
                        // vertical chunks, so we need to check whether we have at least this number of chunks available
                        if num_split_groups + 1 > num_available_chunks + max_num_vertical_chunks {
                            // compute the number of extra chunks we need to remove from horizontal chunking
                            // in order to be able to have at least `num_split_groups + 1` vertical chunks available
                            // for the rest of the layers
                            let num_extra_chunks = num_split_groups + 1
                                - (num_available_chunks + max_num_vertical_chunks);
                            if num_extra_chunks > group.0 - 2 {
                                // we cannot reduce enough the number of chunks assigned to this group,
                                // so we don't split horizontally this group of layers, and we go on with the next group
                                num_available_chunks += group.0;
                                num_split_groups -= 1;
                                return None;
                            }
                            // otherwise, we reduce the number of chunks assigned to this group by `num_extra_chunks`
                            let new_max_num_chunks_for_group = group.0 - num_extra_chunks;
                            let maybe_num_chunks_for_group = compute_num_chunks_for_group(
                                group.1.iter(),
                                new_max_num_chunks_for_group,
                                model,
                                &shapes,
                            );
                            if let Err(err) = maybe_num_chunks_for_group {
                                return Some(Err(err));
                            }
                            let Some(num_chunks_for_group) = maybe_num_chunks_for_group.unwrap()
                            else {
                                // no valid number of chunks possible for this group, so we skip it
                                num_available_chunks += group.0;
                                num_split_groups -= 1;
                                return None;
                            };
                            num_available_chunks += group.0 - num_chunks_for_group;
                            group.0 = num_chunks_for_group;
                        }
                        Some(Ok(group))
                    }
                })
                .collect::<anyhow::Result<Vec<_>>>()?;

            Some(groups)
        } else {
            None
        };

        let (num_horizontal_chunks, num_splitted_nodes) = horizontal_chunk_groups
            .as_ref()
            .map(|groups| {
                groups
                    .iter()
                    .map(|(num_chunks, nodes)| (*num_chunks, nodes.len()))
                    .reduce(|a, b| (a.0 + b.0, a.1 + b.1))
                    .unwrap_or((0usize, 0usize))
            })
            .unwrap_or((0usize, 0usize));

        let num_nodes = model.nodes.inner_nodes_count();

        let mut num_vertical_chunks = num_chunks - num_horizontal_chunks;
        let mut num_unsplit_nodes = num_nodes - num_splitted_nodes;

        if num_vertical_chunks > num_unsplit_nodes {
            num_vertical_chunks = num_unsplit_nodes;
            warn!(
                "Using less chunks {} than specified as input ({num_chunks}) because there are not enough nodes in the model",
                num_vertical_chunks + num_horizontal_chunks
            );
            num_chunks = num_horizontal_chunks + num_vertical_chunks;
        }
        ensure!(
            num_vertical_chunks <= num_unsplit_nodes,
            "Number of vertical chunks ({num_vertical_chunks}) cannot be greater than number of unsplit nodes ({num_unsplit_nodes})"
        );

        let nodes_per_prover = |num_nodes, num_chunks| num_nodes / num_chunks;
        // determine the number of chunks that will have an extra node, in order to have exactly `num_vertical_chunks`
        // non-empty chunks
        let num_chunks_with_extra_node = |num_nodes, num_chunks| num_nodes % num_chunks;

        let mut subgraphs = vec![ChunkedGraph::new(); num_chunks];

        let to_be_splitted_nodes = horizontal_chunk_groups
            .as_ref()
            .map(|group| {
                group
                    .iter()
                    .flat_map(|(num_chunks, nodes)| nodes.iter().map(|node| (node, *num_chunks)))
                    .collect::<HashMap<_, _>>()
            })
            .unwrap_or_default();

        let mut current_chunk = 0;
        let mut nodes_in_current_chunk = 0;
        let mut next_node_id = || next_node_id.next().expect("Ran out of node ids");
        let mut max_edge_id: usize = model
            .nodes
            .edges()
            .map(|(&edge_id, _)| edge_id)
            .max()
            .unwrap()
            .into();
        let mut next_edge_id = || {
            max_edge_id += 1;
            max_edge_id.into()
        };
        let mut nodes_map = HashMap::new();
        let mut splitted_nodes = SplittedNodes::default();
        for (node_id, _) in model
            .nodes
            .backward_iter()
            .filter(|(_, node)| node.is_inner())
        {
            let outgoing_neighbors = model
                .nodes
                .neighbors(node_id, Direction::Outgoing)
                .collect_vec();
            if let Some(&num_chunks) = to_be_splitted_nodes.get(&node_id) {
                // we need to check whether we are already in a set of split nodes or not
                let new_set = if outgoing_neighbors.len() > 1 {
                    // safety check: all the outgoing neighbors are not splitted nodes
                    ensure!(
                        outgoing_neighbors
                            .iter()
                            .all(|(_, edge)| !to_be_splitted_nodes.contains_key(&edge.target()))
                    );
                    true
                } else if let Some(&neighbor_num_chunks) =
                    to_be_splitted_nodes.get(&outgoing_neighbors[0].1.target())
                {
                    ensure!(neighbor_num_chunks == num_chunks);
                    false
                } else {
                    true
                };
                // we need to horizontally split this node.
                let recombination_id = if new_set {
                    // we are starting a new set of split nodes, so we need to add a recombination layer to the current chunk
                    let num_outputs = model.nodes.outgoing_ports(node_id).len();
                    let recombination_layer =
                        RecombinationLayer::new(vec![num_chunks; num_outputs]);
                    let recombination_chunk = if nodes_in_current_chunk == 0 {
                        // it's an empty chunk, so we add the recombination layer to the previous chunk, if it exists
                        if current_chunk == 0 {
                            // we are in the first chunk, so we don't add a recombination layer only to handle the outputs
                            // of the model
                            current_chunk = num_chunks - 1; // set the current chunk to the latest horizontal chunk
                            None
                        } else {
                            Some(current_chunk - 1)
                        }
                    } else {
                        num_vertical_chunks -= 1;
                        num_unsplit_nodes -= nodes_in_current_chunk;
                        Some(current_chunk)
                    };
                    recombination_chunk
                        .map(|recombination_chunk| {
                            let recombination_id = next_node_id();
                            nodes_map.insert(recombination_id, recombination_chunk);
                            current_chunk = recombination_chunk + num_chunks;
                            subgraphs[recombination_chunk].add_node_with_id(
                                recombination_id,
                                Node::Inner(ChunkedNode::RecombinationLayer(
                                    recombination_layer.clone(),
                                )),
                            )?;
                            for (edge_id, edge) in
                                model.nodes.neighbors(node_id, Direction::Outgoing)
                            {
                                subgraphs[recombination_chunk].add_edge_raw_with_id(
                                    *edge_id,
                                    Edge::new(
                                        recombination_id,
                                        edge.target(),
                                        edge.ports().clone(),
                                        edge.weight,
                                    ),
                                )?;
                            }
                            splitted_nodes
                                .inner_nodes
                                .entry(node_id)
                                .or_default()
                                .recombination_layer =
                                Some((recombination_id, recombination_layer));
                            anyhow::Ok(recombination_id)
                        })
                        .transpose()?
                } else {
                    None
                };
                // We now need to add `num_chunks` different nodes, and assign each to a different chunk
                for i in 0..num_chunks {
                    let new_node_id = next_node_id();
                    nodes_map.insert(new_node_id, current_chunk - i);
                    let chunked_layer = ChunkedLayer {
                        original_node_id: node_id,
                        chunk_number: i,
                    };
                    subgraphs[current_chunk - i].add_node_with_id(
                        new_node_id,
                        Node::Inner(ChunkedNode::ChunkedLayer(chunked_layer)),
                    )?; // ToDo: use `ChunkedLayer` inside `Node::Inner`
                    // add edges to link outputs of new chunk node to the subsequent chunked node, if any
                    let original_edge = outgoing_neighbors[0].1;
                    if let Some(splitted_node) =
                        splitted_nodes.inner_nodes.get(&original_edge.target())
                    {
                        let neighbor_id = splitted_node.new_nodes[&i];
                        let new_edge_id = next_edge_id();
                        let edge = Edge::new(
                            new_node_id,
                            neighbor_id,
                            original_edge.ports().clone(),
                            None,
                        );
                        subgraphs[current_chunk - i]
                            .add_edge_raw_with_id(new_edge_id, edge.clone())?;
                        // add the same edge also to the chunk containing `neighbor_id`
                        let neighbor_chunk = nodes_map
                            .get(&neighbor_id)
                            .ok_or(anyhow!("Chunk not found for neighbor node {neighbor_id}"))?;
                        if *neighbor_chunk != current_chunk - i {
                            subgraphs[*neighbor_chunk].add_edge_raw_with_id(new_edge_id, edge)?
                        }
                    } else {
                        if let Some(recombination_id) = recombination_id {
                            // we need to link it to the recombination layer
                            // in this case we need to map output port j of the original node to input port j*num_chunks+i of recombination layer
                            let ports = model
                                .nodes
                                .outgoing_ports(node_id)
                                .into_iter()
                                .map(|out| {
                                    let out_port: usize = out.port.into();
                                    PortLink::new(out_port, out_port * num_chunks + i)
                                })
                                .collect_vec();
                            let new_edge_id = next_edge_id();
                            let edge = Edge::new(new_node_id, recombination_id, ports, None);
                            subgraphs[current_chunk - i]
                                .add_edge_raw_with_id(new_edge_id, edge.clone())?;
                            // add the same edge also to the chunk containing the recombination layer
                            subgraphs[current_chunk - num_chunks]
                                .add_edge_raw_with_id(new_edge_id, edge)?
                        } else {
                            // build output nodes for this chunk
                            for (edge_id, edge) in outgoing_neighbors.iter() {
                                let target = edge.target();
                                // check that target is an output node
                                let output_id = model.nodes.node(target).ok_or(
                                    anyhow!("Target {node_id} of edge {edge_id} not found in the model")
                                )?.as_output().ok_or(
                                    anyhow!("Target {node_id} of edge {edge_id} is not an output of the model")
                                )?;
                                let new_out_node = ChunkedOutNode::Chunked(ChunkedOutput {
                                    io_id: *output_id,
                                    chunk_id: i,
                                });
                                let new_out_node_id = next_node_id();
                                subgraphs[current_chunk - i].add_node_with_id(
                                    new_out_node_id,
                                    Node::Output(new_out_node),
                                )?;
                                let new_edge_id = next_edge_id();
                                let new_edge = Edge::new(
                                    new_node_id,
                                    new_out_node_id,
                                    edge.ports().clone(),
                                    edge.weight,
                                );
                                subgraphs[current_chunk - i]
                                    .add_edge_raw_with_id(new_edge_id, new_edge)?;
                                splitted_nodes
                                    .outputs
                                    .entry(*output_id)
                                    .or_default()
                                    .push(new_out_node_id);
                            }
                        }
                    }
                    splitted_nodes
                        .inner_nodes
                        .entry(node_id)
                        .or_default()
                        .new_nodes
                        .insert(i, new_node_id);
                }
            } else {
                // check whether we are ending a set of horizontally chunked nodes or not
                let mut splitted_neighbors = outgoing_neighbors
                    .into_iter()
                    .filter_map(|(_, edge)| {
                        to_be_splitted_nodes
                            .get(&edge.target())
                            .map(|&num_chunks| (edge, num_chunks))
                    })
                    .collect_vec();
                ensure!(
                    splitted_neighbors.len() <= 1,
                    "There should be at most one splitted neighbor, found {}",
                    splitted_neighbors.len()
                );
                if !splitted_neighbors.is_empty() {
                    // we are ending a set of horizontally chunked nodes, so we need to add a SplitLayer to the next chunk
                    current_chunk += 1;
                    nodes_in_current_chunk = 0;
                    let (split_edge, num_chunks) = splitted_neighbors.pop().unwrap();
                    let splitted_node_id = split_edge.target();
                    let num_inputs = model.nodes.incoming_feeds(splitted_node_id).len();
                    let split_layer = SplitLayer {
                        unpadded_input_shapes: get_input_shapes_for_node(
                            model,
                            splitted_node_id,
                            &shapes,
                        ),
                        num_chunks: vec![num_chunks; num_inputs],
                    };
                    let split_id = next_node_id();
                    nodes_map.insert(split_id, current_chunk);
                    // ToDo: Add SplitLayer to subgraph of recombination chunk
                    subgraphs[current_chunk].add_node_with_id(
                        split_id,
                        Node::Inner(ChunkedNode::SplitLayer(split_layer.clone())),
                    )?;
                    splitted_nodes
                        .inner_nodes
                        .entry(splitted_node_id)
                        .or_default()
                        .split_layer = Some((split_id, split_layer));
                    for (edge_id, edge) in
                        model.nodes.neighbors(splitted_node_id, Direction::Incoming)
                    {
                        subgraphs[current_chunk].add_edge_raw_with_id(
                            *edge_id,
                            Edge::new(edge.source(), split_id, edge.ports().clone(), edge.weight),
                        )?;
                    }
                    // add egdes to link SplitLayer to the chunked layers of the splitted node
                    let neighbor_node_ids = &splitted_nodes
                        .inner_nodes
                        .get(&splitted_node_id)
                        .expect("A splitted node must have chunked layers")
                        .new_nodes;
                    // for each input port i of the splitted node, we need to add `num_chunks` edge linking input port j of the i-th chunked node with
                    // the output port j*num_chunks + i of the split layer
                    for (&i, &neighbor_id) in neighbor_node_ids.iter() {
                        // build the port links between split layer and neighbor_id node, which corresponds to the i-th chunk of the
                        // splitted node inputs
                        let ports = (0..num_inputs)
                            .map(|j| PortLink::new(j * num_chunks + i, j))
                            .collect_vec();
                        let new_edge_id = next_edge_id();
                        let edge = Edge::new(split_id, neighbor_id, ports, None);
                        subgraphs[current_chunk].add_edge_raw_with_id(new_edge_id, edge.clone())?;
                        // add the same edge also to the chunk containing `neighbor_id`
                        let neighbor_chunk = nodes_map
                            .get(&neighbor_id)
                            .ok_or(anyhow!("Chunk not found for neighbor node {neighbor_id}"))?;
                        subgraphs[*neighbor_chunk].add_edge_raw_with_id(new_edge_id, edge)?
                    }
                }
                subgraphs[current_chunk]
                    .add_node_with_id(node_id, Node::Inner(ChunkedNode::OriginalNode(())))?;
                nodes_map.insert(node_id, current_chunk);
                nodes_in_current_chunk += 1;
                let max_nodes_in_current_chunk =
                    if num_chunks_with_extra_node(num_unsplit_nodes, num_vertical_chunks) > 0 {
                        nodes_per_prover(num_unsplit_nodes, num_vertical_chunks) + 1 // in the first `num_chunks_with_extra_node` chunks, there is an extra node
                    } else {
                        nodes_per_prover(num_unsplit_nodes, num_vertical_chunks)
                    };
                if nodes_in_current_chunk >= max_nodes_in_current_chunk {
                    num_vertical_chunks -= 1;
                    num_unsplit_nodes -= nodes_in_current_chunk;
                    current_chunk += 1;
                    nodes_in_current_chunk = 0;
                }
            }
        }

        // we need to add input nodes (and related edges) for the chunks containing splitted nodes, if any
        for (splitted_node, input_id, edge) in model.nodes.edges().filter_map(|(_, edge)| {
            splitted_nodes
                .inner_nodes
                .get(&edge.target())
                .and_then(|splitted_node| {
                    model
                        .nodes
                        .node(edge.source())
                        .expect("Source node must be in the graph")
                        .as_input()
                        .map(|input_id| (splitted_node, input_id, edge))
                })
        }) {
            // build num_chunks input nodes, and link each of them to the corresponding chunked layer
            for (chunk_id, node_id) in &splitted_node.new_nodes {
                let chunked_input = ChunkedInNode::Chunked(ChunkedInput {
                    io_id: *input_id,
                    chunk_id: *chunk_id,
                });
                let new_input_node_id = next_node_id();
                let node_chunk = nodes_map
                    .get(node_id)
                    .ok_or(anyhow!("No chunk found for splitted node {node_id}"))?;
                subgraphs[*node_chunk]
                    .add_node_with_id(new_input_node_id, Node::Input(chunked_input))?;
                // add edge between the new input node and the splitted node
                let new_edge_id = next_edge_id();
                let new_edge = Edge::new(
                    new_input_node_id,
                    *node_id,
                    edge.ports().clone(),
                    edge.weight,
                );
                subgraphs[*node_chunk].add_edge_raw_with_id(new_edge_id, new_edge)?;
                splitted_nodes
                    .inputs
                    .entry(*input_id)
                    .or_default()
                    .push(new_input_node_id);
            }
        }

        // check that there are no empty chunks
        ensure!(
            subgraphs.iter().all(|chunk| chunk.inner_nodes_count() > 0),
            "There should be no empty chunks"
        );

        // now we add relevant edges in each subgraph
        add_edges_to_chunk_subgraphs(model, &mut subgraphs, &nodes_map, &splitted_nodes)?;

        Ok((subgraphs, splitted_nodes))
    }
}

fn compute_unpadded_output_shapes<F: PrimeField>(
    model_unpadded_input_shapes: &[Shape],
    model: &ModelCtx<F>,
) -> HashMap<NodeOutput, Shape> {
    let mut shapes: HashMap<_, _> = model
        .nodes
        .input_node_ids()
        .into_iter()
        .zip(model_unpadded_input_shapes)
        .map(|(input_node_id, input_shape)| {
            (NodeOutput::new(input_node_id, 0), input_shape.clone())
        })
        .collect();
    for (node_id, node) in model.nodes.forward_inners() {
        let unpadded_input_shapes = get_input_shapes_for_node(model, node_id, &shapes);
        let output_shapes = node
            .output_shapes(&unpadded_input_shapes, PaddingMode::NoPadding)
            .expect("Shouldn't fail since we successfuly generated the context for this model");
        shapes.extend(
            output_shapes
                .into_iter()
                .enumerate()
                .map(|(i, shape)| (NodeOutput::new(node_id, i), shape)),
        );
    }
    shapes
}
/// A chunking strategy specifically devoted for LLMs.
/// It ensures that the `Add` layers don't cause multiple
/// commitments to be produced for each chunk, which however
/// prevent to split an attention layer across multiple chunks.
/// Thus, it makes sense to use this strategy only when there are
/// enough attention layers to have an high enough number of
/// chunks
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct LLMChunkingStrategy {
    // A cache of the unpadded shapes for the outputs of all nodes in the model
    unpadded_shapes: HashMap<NodeOutput, Shape>,
}

impl LLMChunkingStrategy {
    pub fn new<F: PrimeField>(input_length: usize, model: &ModelCtx<F>) -> Self {
        let model_unpadded_input_shapes = vec![Shape::new(vec![input_length])];
        Self::new_from_input_shape(model_unpadded_input_shapes, model)
    }

    fn new_from_input_shape<F: PrimeField>(
        model_unpadded_input_shapes: Vec<Shape>,
        model: &ModelCtx<F>,
    ) -> Self {
        let unpadded_shapes = compute_unpadded_output_shapes(&model_unpadded_input_shapes, model);
        Self { unpadded_shapes }
    }

    fn add_nodes<'a, F: PrimeField>(
        &self,
        model: &'a ModelCtx<F>,
    ) -> impl Iterator<Item = NodeId> + 'a {
        model
            .nodes
            .backward_iter()
            .filter_map(|(node_id, node)| node.as_inner().map(|n| (node_id, n)))
            .filter_map(|(node_id, ctx)| {
                if let &LayerCtx::Add(_) = ctx {
                    Some(node_id)
                } else {
                    None
                }
            })
    }

    fn get_last_logits_and_final_projection_nodes<
        'a,
        F: PrimeField,
        I: Iterator<Item = (NodeId, &'a Node<LayerCtx<F>>)>,
    >(
        nodes_backward_iter: &mut I,
    ) -> anyhow::Result<(NodeId, NodeId)> {
        let mut next_node = |node_type: String| {
            let (node_id, node) = nodes_backward_iter
                .next()
                .expect("No node found in LLM model?");
            ensure!(
                node.as_inner().expect("It's an inner node").variant_name() == node_type,
                "Last node of LLM is not {node_type}"
            );
            Ok(node_id)
        };
        let logits_node_id = next_node("Logits".to_string())?;
        let einsum_node_id = next_node("EinSum".to_string())?;
        Ok((logits_node_id, einsum_node_id))
    }

    fn num_ideal_chunks_for_logits_and_final_projection<F: PrimeField>(
        &self,
        logits_node_id: NodeId,
        einsum_node_id: NodeId,
        model: &ModelCtx<F>,
    ) -> Option<usize> {
        let einsum_node = model
            .nodes
            .node(einsum_node_id)
            .expect("Final projection node not found in LLM")
            .as_inner()
            .expect("Final projection must be an inner node");
        let einsum_num_chunks = einsum_node.ideal_num_chunks(&get_input_shapes_for_node(
            model,
            einsum_node_id,
            &self.unpadded_shapes,
        ));
        let logits_node = model
            .nodes
            .node(logits_node_id)
            .expect("Logits node not found in LLM")
            .as_inner()
            .expect("Logits must be an inner node");
        let logits_num_chunks = logits_node.ideal_num_chunks(&get_input_shapes_for_node(
            model,
            einsum_node_id,
            &self.unpadded_shapes,
        ));
        [einsum_num_chunks, logits_num_chunks]
            .into_iter()
            .flatten()
            .min()
    }
}

impl<D: TensorTypeParam, F: PrimeField> From<(&Trace<D>, &ModelCtx<F>)> for LLMChunkingStrategy {
    fn from(value: (&Trace<D>, &ModelCtx<F>)) -> Self {
        let unpadded_input_shapes = value
            .0
            .inputs()
            .iter()
            .map(|input| input.unpadded_shape().clone())
            .collect();
        Self::new_from_input_shape(unpadded_input_shapes, value.1)
    }
}

impl ChunkingStrategy for LLMChunkingStrategy {
    fn ideal_num_chunks<F: PrimeField>(&self, model: &ModelCtx<F>) -> anyhow::Result<usize> {
        // the ideal number of chunks is given by splitting the model at each `Add` node
        // hence, the ideal number of chunks is `num_add_nodes + 2`
        let num_add_nodes = self.add_nodes(model).count();
        // compute number of horizontal chunks for Logits and Einsum
        let (logits_node_id, einsum_node_id) = Self::get_last_logits_and_final_projection_nodes(
            &mut model
                .nodes
                .backward_iter()
                .filter(|(_, node)| node.is_inner()),
        )
        .expect("Last nodes in LLM are not Logits and Einsum");
        let num_extra_chunks = self
            .num_ideal_chunks_for_logits_and_final_projection(logits_node_id, einsum_node_id, model)
            .and_then(|max_num_chunks_for_group| {
                compute_num_chunks_for_group(
                    [einsum_node_id, logits_node_id].iter(),
                    max_num_chunks_for_group,
                    model,
                    &self.unpadded_shapes,
                )
                .transpose()
            })
            .unwrap_or(Ok(1))?; // if logits and einsum cannot be horizontally split, we consider only 1 extra vertical 
        // chunk for these 2 layers
        Ok(num_add_nodes + 1 + num_extra_chunks + 1) // for embedding chunk
    }

    fn split<F: PrimeField>(
        &self,
        model: &ModelCtx<F>,
        mut num_chunks: usize,
        mut next_node_id: impl Iterator<Item = NodeId>,
    ) -> anyhow::Result<(Vec<ChunkedGraph>, SplittedNodes)> {
        // this is the divisor employed to compute the minimun number of vertical chunks
        const FRACTION_FOR_MIN_VERTICAL_CHUNKS: usize = 4;
        const FRACTION_FOR_MIN_HORIZONTAL_CHUNKS: usize = 2;
        // first, find add layers there are in the model
        let add_nodes = self.add_nodes(model).collect::<HashSet<_>>();
        let num_add_nodes = add_nodes.len();
        let mut splitted_nodes = SplittedNodes::default();

        let mut subgraphs = vec![ChunkedGraph::new(); num_chunks];
        let mut nodes_map = HashMap::new();
        // check if there is an Absolute Positional layer at the beginning of the model
        let mut forward_iter = model.nodes.forward_inners();
        let (embedding_node_id, embedding_node) =
            forward_iter.next().expect("No iniital node found in LLM?");
        ensure!(
            embedding_node.variant_name() == "Embeddings",
            "First node of LLM is not Embeddings"
        );

        let extra_chunk_for_embedding = num_chunks >= 3;
        let embedding_chunk_layers = extra_chunk_for_embedding
            .then(|| {
                // check whether there is also a Positional layer to be added to this chunk
                let (next_node_id, next_node) = forward_iter
                    .next()
                    .expect("At least 2 nodes are expected in LLM");
                if matches!(next_node, LayerCtx::Positional(PositionalCtx::Absolute(_))) {
                    anyhow::Ok(vec![embedding_node_id, next_node_id])
                } else {
                    Ok(vec![embedding_node_id])
                }
            })
            .transpose()?;
        let mut num_extra_chunks = extra_chunk_for_embedding as usize;
        // whether we create an extra chunk at the end of the LLM for Argmax and final projection,
        // which are computationally intensive and so it is beneficial to have a standalone chunk for
        // them
        let add_extra_chunk = num_chunks >= 2;
        let mut nodes_iter = model
            .nodes
            .backward_iter()
            .filter(|(_, node)| node.is_inner())
            .take_while(|(node_id, _)| {
                if let Some(layers) = &embedding_chunk_layers {
                    node_id
                        != layers
                            .last()
                            .expect("At least 1 layer msut be found in embedding chunk")
                } else {
                    true
                }
            });
        let first_add_chunk = if add_extra_chunk {
            // the first chunk is given by Logits and previous EinSum node
            let (logits_node_id, einsum_node_id) =
                Self::get_last_logits_and_final_projection_nodes(&mut nodes_iter)?;
            let ideal_horizontal_chunks = self
                .num_ideal_chunks_for_logits_and_final_projection(
                    logits_node_id,
                    einsum_node_id,
                    model,
                )
                .unwrap_or(0);
            // compute the number of chunks we can reserve to horizontally split logits and einsum;
            // currently, we consider as potential horizontal chunks all the chunks left after the
            // model is split into the minimum number of vertical chunks
            // The minimum number of vertical chunks for now is computed as 1/FRACTION_FOR_MIN_VERTICAL_CHUNKS
            // of the number of `Add` nodes
            let min_vertical_chunks = (num_add_nodes + 1) / FRACTION_FOR_MIN_VERTICAL_CHUNKS;
            let num_available_chunks =
                num_chunks.saturating_sub(num_extra_chunks + min_vertical_chunks);
            // compute the number of minimum horizontal chunks we want to use;
            // we start by splitting logits and einsum into `min_horizontal_chunks` horizontal chunks rather than into
            // `ideal_horizontal_chunks`, because if `num_available_chunks > min_horizontal_chunks`, we prefer to employ
            // the additional available chunks to further split vertically the layers of the model into more chunks
            let min_horizontal_chunks =
                ideal_horizontal_chunks / FRACTION_FOR_MIN_HORIZONTAL_CHUNKS;
            // the number of horizontal chunks to be employed for logits and einsum is thus `min_horizontal_chunks`,
            // unless the number of available chunks is less than `min_horizontal chunks`
            let mut num_horizontal_chunks = min_horizontal_chunks.min(num_available_chunks);
            // now check whether we can fit all the remaining vertical chunks into the available chunks; indeed, if
            // that's the case, we want to use more horizontal chunks than `min_horizontal_chunks`, up to `ideal_horizontal_chunks`
            let remaining_vertical_chunks = num_add_nodes + 1 - min_vertical_chunks;
            if num_available_chunks - num_horizontal_chunks > remaining_vertical_chunks {
                // we can simply set `num_horizontal_chunks` to `num_available_chunks - remaining_vertical_chunks`,
                // up to `ideal_horizontal_chunks`
                num_horizontal_chunks =
                    ideal_horizontal_chunks.min(num_available_chunks - remaining_vertical_chunks);
            }
            // compute the actual num chunks for the 2 layers
            if let Some(num_chunks_for_group) = compute_num_chunks_for_group(
                [einsum_node_id, logits_node_id].iter(),
                num_horizontal_chunks,
                model,
                &self.unpadded_shapes,
            )? {
                // we need to insert the nodes for each chunk of both layers
                let mut next_node_id = || next_node_id.next().expect("Ran out of node ids");
                let mut max_edge_id: usize = model
                    .nodes
                    .edges()
                    .map(|(&edge_id, _)| edge_id)
                    .max()
                    .unwrap()
                    .into();
                let mut next_edge_id = || {
                    max_edge_id += 1;
                    max_edge_id.into()
                };
                for (chunk_number, chunk_subgraph) in
                    subgraphs.iter_mut().enumerate().take(num_chunks_for_group)
                {
                    // add chunked layer for logits
                    let new_logits_id = next_node_id();
                    chunk_subgraph.add_node_with_id(
                        new_logits_id,
                        Node::Inner(ChunkedNode::ChunkedLayer(ChunkedLayer {
                            original_node_id: logits_node_id,
                            chunk_number,
                        })),
                    )?;
                    nodes_map.insert(new_logits_id, chunk_number);
                    // add chunked layer for final projection
                    let new_einsum_id = next_node_id();
                    chunk_subgraph.add_node_with_id(
                        new_einsum_id,
                        Node::Inner(ChunkedNode::ChunkedLayer(ChunkedLayer {
                            original_node_id: einsum_node_id,
                            chunk_number,
                        })),
                    )?;
                    nodes_map.insert(new_einsum_id, chunk_number);
                    // add edge between the chunked nodes for this chunk
                    let new_edge_id = next_edge_id();
                    let original_edge = model.nodes.edges().find_map(|(_, edge)|
                        (edge.source() == einsum_node_id && edge.target() == logits_node_id)
                            .then_some(edge)
                    ).ok_or(
                        anyhow!("No edge found between final prjection node {einsum_node_id} and Logits node {logits_node_id}")
                    )?;
                    let new_edge = Edge::new(
                        new_einsum_id,
                        new_logits_id,
                        original_edge.ports().clone(),
                        original_edge.weight,
                    );
                    chunk_subgraph.add_edge_raw_with_id(new_edge_id, new_edge)?;
                    // add output edge for chunked logits
                    for (edge_id, edge) in
                        model.nodes.neighbors(logits_node_id, Direction::Outgoing)
                    {
                        let target = edge.target();
                        // check that target is an output node
                        let output_id = model
                            .nodes
                            .node(target)
                            .ok_or(anyhow!(
                                "Target {target} of edge {edge_id} not found in the model"
                            ))?
                            .as_output()
                            .ok_or(anyhow!(
                                "Target {target} of edge {edge_id} is not an output of the model"
                            ))?;
                        let new_out_node = ChunkedOutNode::Chunked(ChunkedOutput {
                            io_id: *output_id,
                            chunk_id: chunk_number,
                        });
                        let new_out_node_id = next_node_id();
                        chunk_subgraph
                            .add_node_with_id(new_out_node_id, Node::Output(new_out_node))?;
                        let new_edge_id = next_edge_id();
                        let new_edge = Edge::new(
                            new_logits_id,
                            new_out_node_id,
                            edge.ports().clone(),
                            edge.weight,
                        );
                        chunk_subgraph.add_edge_raw_with_id(new_edge_id, new_edge)?;
                        // add chunked outputs node to `splitted_nodes`
                        splitted_nodes
                            .outputs
                            .entry(*output_id)
                            .or_default()
                            .push(new_out_node_id);
                    }
                    // add nodes to `splitted_nodes` info
                    splitted_nodes
                        .inner_nodes
                        .entry(logits_node_id)
                        .or_default()
                        .new_nodes
                        .insert(chunk_number, new_logits_id);
                    splitted_nodes
                        .inner_nodes
                        .entry(einsum_node_id)
                        .or_default()
                        .new_nodes
                        .insert(chunk_number, new_einsum_id);
                }
                // Build the `SplitLayer` to be inserted before the final projection node.
                // It will end up in `num_chunks_for_group`-th chunk
                let num_inputs = model.nodes.incoming_feeds(einsum_node_id).len();
                let split_layer = SplitLayer {
                    unpadded_input_shapes: get_input_shapes_for_node(
                        model,
                        einsum_node_id,
                        &self.unpadded_shapes,
                    ),
                    num_chunks: vec![num_chunks_for_group; num_inputs],
                };
                let split_id = next_node_id();
                let split_node_chunk = num_chunks_for_group;
                nodes_map.insert(split_id, split_node_chunk);
                subgraphs[split_node_chunk].add_node_with_id(
                    split_id,
                    Node::Inner(ChunkedNode::SplitLayer(split_layer.clone())),
                )?;
                splitted_nodes
                    .inner_nodes
                    .entry(einsum_node_id)
                    .or_default()
                    .split_layer = Some((split_id, split_layer));
                // link the `SplitLayer` to all the sources of the Einsum node
                for (edge_id, edge) in model.nodes.neighbors(einsum_node_id, Direction::Incoming) {
                    subgraphs[split_node_chunk].add_edge_raw_with_id(
                        *edge_id,
                        Edge::new(edge.source(), split_id, edge.ports().clone(), edge.weight),
                    )?;
                }
                // add egdes to link SplitLayer to the chunked layers of the splitted node
                let new_einsum_node_ids = &splitted_nodes
                    .inner_nodes
                    .get(&einsum_node_id)
                    .expect("A splitted node must have chunked layers")
                    .new_nodes;
                // for each input port i of the splitted node, we need to add `num_chunks_for_group` edge linking input port j of the i-th chunked node with
                // the output port j*num_chunks + i of the split layer
                for (&i, &neighbor_id) in new_einsum_node_ids {
                    // build the port links between split layer and neighbor_id node, which corresponds to the i-th chunk of the
                    // splitted node inputs
                    let ports = (0..num_inputs)
                        .map(|j| PortLink::new(j * num_chunks + i, j))
                        .collect_vec();
                    let new_edge_id = next_edge_id();
                    let edge = Edge::new(split_id, neighbor_id, ports, None);
                    subgraphs[split_node_chunk].add_edge_raw_with_id(new_edge_id, edge.clone())?;
                    // add the same edge also to the chunk containing `neighbor_id`
                    let neighbor_chunk = nodes_map
                        .get(&neighbor_id)
                        .ok_or(anyhow!("Chunk not found for neighbor node {neighbor_id}"))?;
                    subgraphs[*neighbor_chunk].add_edge_raw_with_id(new_edge_id, edge)?
                }
                num_chunks_for_group
            } else {
                subgraphs[0]
                    .add_node_with_id(logits_node_id, Node::Inner(ChunkedNode::OriginalNode(())))?;
                nodes_map.insert(logits_node_id, 0);
                subgraphs[0]
                    .add_node_with_id(einsum_node_id, Node::Inner(ChunkedNode::OriginalNode(())))?;
                nodes_map.insert(einsum_node_id, 0);
                1
            }
        } else {
            0
        };

        num_extra_chunks += first_add_chunk;

        let mut num_remaining_chunks = num_chunks - num_extra_chunks;
        // we check if there are enough `Add` nodes to create `num_add_chunks` chunks by splitting
        // the model at `Add` nodes: if there are `num_add_nodes` split points, we get
        // `num_add_nodes + 1` chunks of the graph. If there aren't enough `Add` nodes for the given
        // number of chunks, we split the remaining nodes of the model in `num_remaining_chunks`,
        // without considering `Add` nodes, because it's better to have smaller chunks than saving
        // an extra commitment for each chunk
        let split_at_add_nodes = num_add_nodes + 1 >= num_remaining_chunks;
        let num_nodes_to_split = if split_at_add_nodes {
            num_add_nodes + 1
        } else {
            // compute the number of remaining nodes: these are the number of nodes in the model
            // except for the logits, final projection and nodes placed in the embedding chunk
            let nodes_in_other_chunks = embedding_chunk_layers
                .as_ref()
                .map(|layers| layers.len())
                .unwrap_or(0)
                + 2;
            let num_remaining_nodes = model.nodes.inner_nodes_count() - nodes_in_other_chunks;
            // we also need to check that there are enough nodes for the given number of chunks
            if num_remaining_nodes < num_remaining_chunks {
                // not enough nodes to create `num_remaining_chunks`, so we cap the number of remaining chunks
                // to `num_remaining_nodes`
                num_remaining_chunks = num_remaining_nodes;
                let new_num_chunks = num_remaining_chunks + num_extra_chunks;
                warn!(
                    "Using less chunks {new_num_chunks} than the number of chunks provided as inputs {num_chunks}"
                );
                num_chunks = new_num_chunks;
                subgraphs.truncate(num_chunks);
            }
            num_remaining_nodes
        };

        debug!("Chunking an LLM with {num_add_nodes} Add nodes in {num_chunks} chunks");
        // `num_remaining_chunks` might be smaller than `num_nodes_to_split`.
        // To get at most `num_remaining_chunks`, we split the graph after every `nodes_per_chunk`
        // nodes found, where `nodes_per_chunk = (num_nodes_to_split)/num_remaining_chunks`
        let nodes_per_chunk = num_nodes_to_split / num_remaining_chunks;

        let remaining_nodes = num_nodes_to_split - nodes_per_chunk * num_remaining_chunks;

        let nodes_per_chunk = (0..remaining_nodes)
            .map(|_| nodes_per_chunk + 1)
            .chain(repeat(nodes_per_chunk))
            .take(num_remaining_chunks)
            .collect::<Vec<usize>>();

        // we sweep over nodes to split the into subgraphs; the node where there is a partition between chunks
        // is the add node, because this should guarantee that the source
        // nodes of every add node ends up in the same chunk. Indeed, an `Add` node in the transformers, except
        // for the first one appearing in the model, takes its input from the previous `Add`` node and from the
        // previous node in the model. Therefore, if we split at `Add` nodes, then the `Add` node and the other
        // source node of the next `Add` node will alwayes end up in the same chunk.
        // While partitioning the nodes in chunks, we also compute a map that determines the chunk where a
        // given node is placed in
        nodes_iter.try_fold(
            (0, first_add_chunk),
            |(mut nodes_found, mut chunk), (node_id, _)| {
                subgraphs[chunk]
                    .add_node_with_id(node_id, Node::Inner(ChunkedNode::OriginalNode(())))?;
                nodes_map.insert(node_id, chunk);
                if split_at_add_nodes {
                    if add_nodes.contains(&node_id) {
                        nodes_found += 1;
                    }
                } else {
                    nodes_found += 1;
                }
                if nodes_found == nodes_per_chunk[chunk - first_add_chunk] {
                    // we move to the next chunk
                    chunk += 1;
                    nodes_found = 0;
                }
                anyhow::Ok((nodes_found, chunk))
            },
        )?;

        if let Some(layers) = embedding_chunk_layers {
            for layer_id in layers {
                subgraphs[num_chunks - 1]
                    .add_node_with_id(layer_id, Node::Inner(ChunkedNode::OriginalNode(())))?;
                nodes_map.insert(layer_id, num_chunks - 1);
            }
        }

        // now we add relevant edges in each subgraph
        add_edges_to_chunk_subgraphs(model, &mut subgraphs, &nodes_map, &splitted_nodes)?;

        Ok((subgraphs, splitted_nodes))
    }
}

/// Initialize the transcript to prove a given chunk, using the challenge squeezed from
/// the initial shared transcript
pub(crate) fn initialize_transcript_for_chunk<F: PrimeField, T: Transcript + InitTranscript>(
    challenge: F,
) -> T {
    let mut transcript = T::new(T::InitData::from(b"model_chunk_proving"));
    transcript.append_scalars(&[challenge]);
    transcript
}

fn add_edges_to_chunk_subgraphs<F: PrimeField>(
    model: &ModelCtx<F>,
    subgraphs: &mut [ChunkedGraph],
    nodes_map: &HashMap<NodeId, usize>,
    splitted_nodes: &SplittedNodes,
) -> anyhow::Result<()> {
    // we iterate over edges and we copy edges linked to a given node in the corresponding subgraph
    model.nodes.edges().try_for_each(|(edge_id, edge)| {
        let source_id = edge.source();
        let source_node = model.nodes.node(source_id).ok_or(anyhow!(
            "Source node {source_id} of edge {edge_id} not found in model"
        ))?;
        let target_id = edge.target();
        let target_node = model.nodes.node(target_id).ok_or(anyhow!(
            "Target node {target_id} of edge {edge_id} not found in model"
        ))?;
        if splitted_nodes.inner_nodes.contains_key(&source_id)
            || splitted_nodes.inner_nodes.contains_key(&target_id)
        {
            // this is an edge related a node replaced by horizontal chunks, so we can skip it
            return Ok(());
        }
        if source_node.as_input().is_none() {
            let chunk = nodes_map
                .get(&source_id)
                .ok_or(anyhow!("Node {source_id} not assigned to any chunk"))?;
            subgraphs[*chunk].add_edge_raw_with_id(*edge_id, edge.clone())?;
            if let Some(o) = target_node.as_output() {
                // we also add output node `o` to the chunk subgraph
                return subgraphs[*chunk]
                    .add_node_with_id(*target_id, Node::Output(ChunkedOutNode::OriginalNode(*o)));
            }
        }
        let chunk = nodes_map
            .get(&target_id)
            .ok_or(anyhow!("Node {target_id} not assigned to any chunk"))?;
        if subgraphs[*chunk].edge(edge_id).is_none() {
            // we add `edge` to `chunk` only if the same edge has not already been
            // added to the subgraph of `chunk` earlier
            subgraphs[*chunk].add_edge_raw_with_id(*edge_id, edge.clone())?;
        }
        if let Some(i) = source_node.as_input() {
            // we also add input node `i` to the chunk subgraph, only if the same
            // input node has not already been added to the subgraph of `chunk` earlier
            if subgraphs[*chunk].node(source_id).is_none() {
                subgraphs[*chunk]
                    .add_node_with_id(*source_id, Node::Input(ChunkedInNode::OriginalNode(*i)))?;
            }
        }
        anyhow::Ok(())
    })
}

#[cfg(test)]
mod test {
    use anyhow::Context;

    use crate::{
        init_test_logging,
        iop::chunking::{ChunkingStrategy, DefaultChunkingStrategy, LLMChunkingStrategy},
        model::{
            Model,
            llm::{Driver, WithMaxContext},
        },
        parser::{
            file_cache,
            gguf::RawGGUF,
            llm::models::gpt2::{GPT2, GPT2_Q8_0},
        },
        testing::Pcs,
    };

    type F = ark_bn254::Fr;

    #[test]
    fn test_default_chunking_strategy() {
        let num_dense_layers = 5;
        let (model, _input) = Model::random(num_dense_layers).unwrap();
        model.describe();
        let (prover_ctx, _verifier_ctx) = model
            .generate_contexts::<F, Pcs>()
            .expect("unable to generate contexts");
        let strategy = DefaultChunkingStrategy::new(model.input_shapes());
        let (chunks, _) = strategy
            .split(&prover_ctx.model_ctx, 3, prover_ctx.next_node_iter())
            .unwrap();

        assert_eq!(chunks.len(), 3);
    }

    #[test]
    fn test_llm_chunking_strategy() -> anyhow::Result<()> {
        init_test_logging("debug");
        const MAX_CONTEXT: usize = 1024;
        let model_path = file_cache::from_cache(GPT2_Q8_0)?;
        let cache_filename = {
            let mut hasher = blake3::Hasher::new();
            hasher
                .update_mmap(&model_path)
                .context("hashing model file")?;
            let hash = hasher.finalize();
            format!("cache-{GPT2_Q8_0}-{hash}.bin")
        };

        // Generate or load the prover & verifier contexts
        let (prover_ctx, _) = file_cache::deserialize_or_create_with(&cache_filename, || {
            let (driver, _metadata) = Driver::load_from_model(
                GPT2::new(),
                &RawGGUF::new(model_path.clone()),
                Some(MAX_CONTEXT),
            )?
            .into_provable_llm(None)?;

            let ctx = driver.context::<F, Pcs>()?.with_max_context(MAX_CONTEXT);

            Ok(ctx)
        })?;

        let strategy = LLMChunkingStrategy::new(MAX_CONTEXT, &prover_ctx.model_ctx);

        for num_chunks in 1..50 {
            let (chunks, _) = strategy
                .split(
                    &prover_ctx.model_ctx,
                    num_chunks,
                    prover_ctx.next_node_iter(),
                )
                .unwrap();

            assert!(chunks.len() <= num_chunks);
        }

        Ok(())
    }
}
