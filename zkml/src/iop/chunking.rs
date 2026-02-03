use std::{
    borrow::Borrow,
    collections::{BTreeMap, BTreeSet, HashMap, HashSet},
};

use anyhow::{anyhow, ensure};
use ff_ext::ExtensionField;
use itertools::Itertools;
use mpcs::PolynomialCommitmentScheme;
use multilinear_extensions::util::transpose;
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use tracing::{debug, warn};
use transcript::Transcript;
use witness::{InstancePaddingStrategy, RowMajorMatrix};

use crate::{
    Claim, Element, InitTranscript,
    commit::mmcs_context::CommitmentProverCtx,
    graph::{Direction, Edge, EdgeId, Graph, Node, NodeId, NodeInput, NodeOutput, PortId},
    iop::prover::ModelLayersRef,
    layers::LayerCtx,
    lookup::context::LookupContext,
    model::{Model, ModelCtx, Trace},
    to_base,
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

pub(crate) type ChunkedGraph = Graph<(), usize, usize, ()>;

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
    // The input edges are grouped according to the source chunk, identified by its
    // `ChunkID`
    pub(crate) incoming_edges: EdgesGroup,
    // set of outgoing edges for the chunk: these are the outgoing
    // edges of nodes in the chunk whose target nodes belong to other chunks.
    // The output edges are grouped according to the destination chunk, identified by
    // its `ChunkID`
    pub(crate) outgoing_edges: EdgesGroup,
}

/// Data type employed to represent a set of incoming or outgoing edges of a chunk;
/// The edges are grouped according to the chunk they are connected to
pub(crate) type EdgesGroup = HashMap<ChunkID, BTreeSet<EdgeId>>;

// Specify whether a group of boundary edges of a chunk are incoming or outgoing edges
#[derive(Clone, Copy, Debug)]
pub(crate) enum GroupType {
    Incoming,
    Outgoing,
}

impl ModelChunk {
    pub(crate) fn from_subgraph(subgraph: ChunkedGraph, chunk_id: usize) -> Self {
        Self {
            subgraph,
            chunk_id: chunk_id.into(),
            incoming_edges: HashMap::new(),
            outgoing_edges: HashMap::new(),
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
                    .inner_nodes()
                    .map(|(node_id, _)| (node_id, chunk.chunk_id))
            })
            .collect()
    }

    /// Group the `incoming_edges` of the chunk according to the chunk where their source node belongs to
    pub(crate) fn build_incoming_grouped_edges(
        &self,
        chunk_for_node: &HashMap<NodeId, ChunkID>,
    ) -> anyhow::Result<EdgesGroup> {
        let input_wires = self.incoming_edges()?;
        input_wires.into_iter().try_fold(
            HashMap::<ChunkID, BTreeSet<EdgeId>>::new(),
            |mut incoming_edges, edge_id| {
                let edge = self.subgraph.edge(&edge_id).ok_or(anyhow!(
                    "Edge {edge_id} not found in subgraph of chunk {}",
                    self.chunk_id
                ))?;
                let source_node_id = edge.source();
                let output_node_chunk = chunk_for_node
                    .get(&source_node_id)
                    .ok_or(anyhow!("Source node {source_node_id} not found in chunks"))?;
                incoming_edges
                    .entry(*output_node_chunk)
                    .or_default()
                    .insert(edge_id);
                anyhow::Ok(incoming_edges)
            },
        )
    }

    /// Group the `outgoing_edges` of the chunk according to the chunk where their target node belongs to
    pub(crate) fn build_outgoing_grouped_edges(
        &self,
        chunk_for_node: &HashMap<NodeId, ChunkID>,
    ) -> anyhow::Result<EdgesGroup> {
        let output_wires = self.outgoing_edges()?;
        output_wires.into_iter().try_fold(
            HashMap::<ChunkID, BTreeSet<EdgeId>>::new(),
            |mut outgoing_edges, edge_id| {
                let edge = self.subgraph.edge(&edge_id).ok_or(anyhow!(
                    "Edge {edge_id} not found in subgraph of chunk {}",
                    self.chunk_id
                ))?;
                let target_node_id = edge.target();
                let output_node_chunk = chunk_for_node
                    .get(&target_node_id)
                    .ok_or(anyhow!("Source node {target_node_id} not found in chunks"))?;

                outgoing_edges
                    .entry(*output_node_chunk)
                    .or_default()
                    .insert(edge_id);
                anyhow::Ok(outgoing_edges)
            },
        )
    }

    /// Utility method to check that each group of incoming edges in a chunk is paired with a corresponding
    /// group of outgoing edges in another chunk, and viceversa.
    pub(crate) fn check_edges_group_consistency<T: Borrow<ModelChunk>>(
        chunks: &BTreeMap<ChunkID, T>,
    ) -> anyhow::Result<()> {
        chunks.values().try_for_each(|chunk| {
            let chunk = chunk.borrow();
            let chunk_id = chunk.chunk_id;
            chunk.incoming_edges.iter().try_for_each(|(source_chunk_id, edges)| {
                let source_chunk = chunks.get(source_chunk_id)
                    .ok_or(anyhow!("Chunk {source_chunk_id} not found"))?;
                let source_group = source_chunk.borrow().outgoing_edges.get(&chunk_id)
                    .ok_or(anyhow!(
                        "Group of output wires routed to chunk {chunk_id} not found in chunk {source_chunk_id}"
                        )
                    )?;
                edges.iter().try_for_each(|edge_id| {
                    ensure!(
                        source_group.contains(edge_id),
                        "Edge {edge_id} not found in output group {chunk_id} of chunk {source_chunk_id}"
                    );
                    anyhow::Ok(())
                })
            })?;
            chunk.outgoing_edges.iter().try_for_each(|(target_chunk_id, edges)| {
                let target_chunk = chunks.get(target_chunk_id)
                    .ok_or(anyhow!("Chunk {target_chunk_id} not found"))?;
                let target_group = target_chunk.borrow().incoming_edges.get(&chunk_id).ok_or(
                    anyhow!("Group of input wires routed to chunk {chunk_id} not found in chunk {target_chunk_id}")
                    )?;
                edges.iter().try_fold(HashSet::new(), |mut output_ports, edge_id| {
                    ensure!(
                        target_group.contains(edge_id),
                        "Edge {edge_id} not found in input group {chunk_id} of chunk {target_chunk_id}"
                    );
                    // check if there is an output port that appears twice in this group
                    let edge = chunk.edge(edge_id)?;
                    edge.ports().iter().for_each(|port| {
                        let output_port = NodeOutput::new(
                        *edge.source(),
                            port.source_port,
                        );
                        if !output_ports.insert(output_port) {
                            warn!("Output port {output_port} appears twice in group {target_chunk_id} of chunk {chunk_id}")
                        }
                    });
                    anyhow::Ok(output_ports)
                })?;
                anyhow::Ok(())
            })
        })
    }

    pub(crate) fn build_chunks<E: ExtensionField, S: ChunkingStrategy>(
        model: &ModelCtx<E>,
        num_chunks: Option<usize>,
        strategy: &S,
    ) -> anyhow::Result<Vec<Self>> {
        let num_chunks = num_chunks.unwrap_or_else(|| strategy.ideal_num_chunks(model));
        let mut chunks: BTreeMap<ChunkID, _> = strategy
            .split(model, num_chunks)?
            .into_iter()
            .enumerate()
            .map(|(i, subgraph)| (i.into(), ModelChunk::from_subgraph(subgraph, i)))
            .collect();

        let chunk_for_node = Self::node_to_chunk_map(chunks.values());

        // group input and output wires of the chunk according to the chunks where these
        // nodes are employed (i.e., this is computing chunk.grouped_input_wires and
        // chunk.grouped_output_wires)
        chunks.values_mut().try_for_each(|chunk| {
            chunk.incoming_edges = chunk.build_incoming_grouped_edges(&chunk_for_node)?;
            chunk.outgoing_edges = chunk.build_outgoing_grouped_edges(&chunk_for_node)?;
            anyhow::Ok(())
        })?;

        Self::check_edges_group_consistency(&chunks)?;

        Ok(chunks.into_values().collect())
    }

    pub(crate) fn edge(&self, id: &EdgeId) -> anyhow::Result<&Edge<()>> {
        self.subgraph
            .edge(id)
            .ok_or(anyhow!("Edge {id} not found in chunk {}", self.chunk_id))
    }

    pub(crate) fn grouped_edges(&self, group_type: GroupType) -> &EdgesGroup {
        match group_type {
            GroupType::Incoming => &self.incoming_edges,
            GroupType::Outgoing => &self.outgoing_edges,
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
    pub(crate) fn claims_for_node<'a, 'b, E: ExtensionField>(
        &self,
        node_id: NodeId,
        claims_by_layers: &'a HashMap<NodeInput, Claim<E>>,
        chunk_output_claims: &'b HashMap<NodeOutput, Claim<E>>,
    ) -> anyhow::Result<BTreeMap<PortId, Vec<&'a Claim<E>>>>
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

    // utility method employed to compute a commitment for a group of edges, which could be
    // either incoming or outgoing edges for the chunk
    fn commitment_for_edge_group<'b, 'd, E: ExtensionField, PCS: PolynomialCommitmentScheme<E>>(
        &self,
        mut edges: impl Iterator<Item = &'b EdgeId>,
        commitment_ctx: &CommitmentProverCtx<E, PCS>,
        chunk_trace: &'d Trace<Element>,
        group_type: GroupType,
    ) -> anyhow::Result<PCS::CommitmentWithWitness>
    where
        PCS::CommitmentWithWitness: Serialize + DeserializeOwned,
    {
        let chunk_id = self.chunk_id;
        let rmms = edges.try_fold(vec![], |mut rmms, edge_id| {
            let edge = self.edge(edge_id)?;
            let node_id = match group_type {
                GroupType::Incoming => edge.target(),
                GroupType::Outgoing => edge.source(),
            };
            let step_data = &chunk_trace.get_step(&node_id).ok_or(anyhow!(
                "Node {node_id} not found in trace for chunk {chunk_id}"
            ))?;
            edge.ports().iter().try_for_each(|port| {
                let tensor = match group_type {
                    GroupType::Incoming => step_data.input_tensor_at(port.target_port.into())?,
                    GroupType::Outgoing => step_data.output_tensor_at(port.source_port.into())?,
                };
                let matrix_values = transpose(vec![to_base::<E, _>(tensor.data())]);
                let rmm = RowMajorMatrix::new_by_inner_matrix(
                    ceno_p3::matrix::dense::DenseMatrix::new(matrix_values.concat(), 1),
                    InstancePaddingStrategy::Default,
                );
                // ToDo: here we are potentially adding the same `rmm` twice, because there may be multiple targets for the
                // same output port of a node. To remove this duplication, we'd need to aggregate claims of input edges other
                // chunks, which is a bit more complex optimization. So, for now, we keep this duplication
                rmms.push(rmm);
                anyhow::Ok(())
            })?;
            anyhow::Ok(rmms)
        })?;
        commitment_ctx.batch_commit(rmms)
    }

    /// Compute the commitments to the groups of incoming or outgoing edges of the chunk;
    /// A batched commitment is computed for each group of edges; the polynomials being
    /// committed are the MLEs of the tensors propagated through the edges in the group.
    /// The `GroupType` parameter specifies whether the commitments are computed for incoming
    /// or outgoing edges
    pub(crate) fn commitments<E: ExtensionField, PCS: PolynomialCommitmentScheme<E>>(
        &self,
        commitment_ctx: &CommitmentProverCtx<E, PCS>,
        full_trace: &Trace<Element>,
        group_type: GroupType,
    ) -> anyhow::Result<BTreeMap<ChunkID, PCS::CommitmentWithWitness>>
    where
        PCS::CommitmentWithWitness: Serialize + DeserializeOwned,
    {
        self.grouped_edges(group_type)
            .iter()
            .map(|(group_id, edges)| {
                let commitment = self.commitment_for_edge_group(
                    edges.iter(),
                    commitment_ctx,
                    full_trace,
                    group_type,
                )?;
                Ok((*group_id, commitment))
            })
            .collect::<anyhow::Result<BTreeMap<_, _>>>()
    }

    /// Compute the incoming edges of a chunk: these are the edges where the target node is in the chunk and
    /// the source node is in an another chunk; note that these doesn't include
    /// input edges of the overall model
    pub(crate) fn incoming_edges(&self) -> anyhow::Result<Vec<EdgeId>> {
        Ok(self
            .subgraph
            .edges()
            .filter_map(|(edge_id, edge)| {
                self.subgraph
                    .node(edge.source())
                    .is_none()
                    .then_some(*edge_id)
            })
            .collect())
    }

    /// Compute the outgoing edges of a chunk: these are the edges where the source node is in the chunk
    /// and the target node is in an another chunk; note that these doesn't include
    /// output edges of the overall model
    pub(crate) fn outgoing_edges(&self) -> anyhow::Result<Vec<EdgeId>> {
        Ok(self
            .subgraph
            .edges()
            .filter_map(|(edge_id, edge)| {
                self.subgraph
                    .node(edge.target())
                    .is_none()
                    .then_some(*edge_id)
            })
            .collect())
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
    /// - the incoming edges, grouped by their source chunk, which is a neighbor chunk for the current chunk
    /// - the outgoing edges, grouped by their target chunk, which is a neighbor chunk for the current chunk
    /// - the model input edges whose target node is in the chunk
    /// - the model output edges whose source node is in the chunk
    pub(crate) fn add_chunk_data_to_transcript<E: ExtensionField, T: Transcript<E>>(
        &self,
        transcript: &mut T,
    ) -> anyhow::Result<()> {
        // closure to append a set of grouped incoming/outgoing edges to the transcript
        let append_group_edges = |group_edges: &EdgesGroup, t: &mut T| {
            for (chunk_id, edges) in group_edges {
                // we append `chunk_id`, `edges.len()` and `edges` to the transcript
                let append_payload = chunk_id
                    .to_le_bytes()
                    .into_iter()
                    .chain(edges.len().to_le_bytes())
                    .chain(edges.iter().flat_map(|edge_id| edge_id.to_le_bytes()))
                    .collect_vec();
                t.append_message(&append_payload);
            }
        };
        // append chunk id
        transcript.append_message(&self.chunk_id.to_le_bytes());
        // append incoming edges, grouped by source chunk
        transcript.append_message("incoming".as_bytes());
        append_group_edges(&self.incoming_edges, transcript);
        // append model input edges of this chunk
        transcript.append_message("inputs".as_bytes());
        self.model_inputs_in_chunk()?
            .into_iter()
            .for_each(|edge_id| transcript.append_message(&edge_id.to_le_bytes()));
        // append outgoing edges, grouped by source chunk
        transcript.append_message("incoming".as_bytes());
        append_group_edges(&self.outgoing_edges, transcript);
        // append model output edges of this chunk
        transcript.append_message("outputs".as_bytes());
        self.model_outputs_in_chunk()?
            .into_iter()
            .for_each(|edge_id| transcript.append_message(&edge_id.to_le_bytes()));
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
            input: vec![],  // they are unused in a chunk prover
            output: vec![], // they are unused in a chunk prover
        })
    }

    pub(crate) fn chunk_layers<'a>(&self, model: &'a Model<Element>) -> ModelLayersRef<'a> {
        model
            .graph()
            .inner_nodes()
            .filter_map(|(node_id, layer)| {
                // retain the node if it is in the current chunk
                self.subgraph.node(node_id).map(|_| (node_id, layer))
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
                        .filter(|node_id| self.subgraph.node(**node_id).is_some())
                        .cloned()
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
    pub(crate) fn check_chunk_commitment_consistency<
        E: ExtensionField,
        PCS: PolynomialCommitmentScheme<E>,
    >(
        &self,
        chunk_commitments_by_id: &HashMap<ChunkID, &ChunkIOCommitments<PCS::Commitment>>,
    ) -> anyhow::Result<()>
    where
        PCS::Commitment: PartialEq + Eq,
    {
        let chunk_id = self.chunk_id;
        self.incoming_edges.keys().try_for_each(|group_id| {
            // first, get the commitment corresponding to this input group
            let input_commitment = chunk_commitments_by_id
                .get(&self.chunk_id)
                .ok_or(anyhow!(
                    "No chunk commitments found for chunk {}",
                    self.chunk_id
                ))?
                .inputs
                .get(group_id)
                .ok_or(anyhow!(
                    "No input commitments found for group {group_id} of chunk {chunk_id}"
                ))?;
            // then, get the commitment of the corresponding output group in another chunk; the group id
            // is the id of the chunk which is expected to have a corresponding output group for this
            // input group
            let output_commitment = chunk_commitments_by_id
                .get(group_id)
                .ok_or(anyhow!("No chunk commitments found for chunk {group_id}"))?
                .outputs
                .get(&self.chunk_id)
                .ok_or(anyhow!(
                    "No output commitments found for group {chunk_id} of chunk {group_id}"
                ))?;
            ensure!(
                input_commitment == output_commitment,
                "Inconsistent commitment found for input group {group_id} of chunk {chunk_id}",
            );
            Ok(())
        })?;

        self.outgoing_edges.keys().try_for_each(|group_id| {
            // first, get the commitment corresponding to this output group
            let output_commitment = chunk_commitments_by_id
                .get(&self.chunk_id)
                .ok_or(anyhow!(
                    "No chunk commitments found for chunk {}",
                    self.chunk_id
                ))?
                .outputs
                .get(group_id)
                .ok_or(anyhow!(
                    "No output commitments found for group {group_id} of chunk {chunk_id}"
                ))?;
            // then, get the commitment of the corresponding input group in another chunk; the group id
            // is the id of the chunk which is expected to have a corresponding input group for this
            // output group
            let input_commitment = chunk_commitments_by_id
                .get(group_id)
                .ok_or(anyhow!("No chunk commitments found for chunk {group_id}"))?
                .inputs
                .get(&self.chunk_id)
                .ok_or(anyhow!(
                    "No input commitments found for group {chunk_id} of chunk {group_id}"
                ))?;
            ensure!(
                input_commitment == output_commitment,
                "Inconsistent commitment found for output group {group_id} of chunk {chunk_id}",
            );
            Ok(())
        })
    }

    // extract claims from `claims_by_layer` corresponding to the group of incoming edges with id
    // `group_id` in the chunk `self`
    pub(crate) fn compute_incoming_group_claims<E: ExtensionField>(
        &self,
        claims_by_layer: &HashMap<NodeInput, Claim<E>>,
        group_id: &ChunkID,
    ) -> anyhow::Result<GroupIOClaims<E>> {
        let chunk_id = self.chunk_id;
        let edges = self.incoming_edges.get(group_id).ok_or(anyhow!(
            "No incoming edges found for group {group_id} in chunk {chunk_id}"
        ))?;
        let group_claims = edges.iter().try_fold(vec![], |mut claims, edge_id| {
            let edge = self.edge(edge_id)?;
            let target_node_id = edge.target();
            edge.ports().iter().try_for_each(|port| {
                let target_port = NodeInput::new(*target_node_id, port.target_port);
                let claim = claims_by_layer
                    .get(&target_port)
                    .ok_or(anyhow!("Claims for target port {target_port:?} not found",))?;
                let Claim { point, eval } = claim;
                claims.push(Claim::new(point.clone(), *eval));
                anyhow::Ok(())
            })?;
            anyhow::Ok(claims)
        })?;
        let commitment_id = ModelChunk::compute_input_group_commitment_id(chunk_id, *group_id);
        Ok(GroupIOClaims {
            commitment_id: commitment_id.into(),
            claims: group_claims,
        })
    }

    // extract claims from `claims_by_port` corresponding to the group of outgoing edges with id
    // `group_id` in the chunk `self`
    pub(crate) fn compute_outgoing_group_claims<E: ExtensionField>(
        &self,
        group_id: &ChunkID,
        claims_by_port: &HashMap<NodeOutput, Claim<E>>,
    ) -> anyhow::Result<GroupIOClaims<E>> {
        let chunk_id = self.chunk_id;
        let edges = self.outgoing_edges.get(group_id).ok_or(anyhow!(
            "No output wires found for group id {group_id} in chunk {chunk_id}"
        ))?;
        let group_claims = edges.iter().try_fold(vec![], |mut claims, edge_id| {
            let edge = self.edge(edge_id)?;
            let source_node_id = edge.source();
            edge.ports().iter().try_for_each(|port| {
                let source_port = NodeOutput::new(source_node_id, port.source_port);
                let claim = claims_by_port
                    .get(&source_port)
                    .ok_or(anyhow!("Claim not found for source port: {source_port}"))?;
                let Claim { point, eval } = claim;
                claims.push(Claim::new(point.clone(), *eval));
                anyhow::Ok(())
            })?;
            anyhow::Ok(claims)
        })?;
        let commitment_id = ModelChunk::compute_output_group_commitment_id(chunk_id, *group_id);
        Ok(GroupIOClaims {
            commitment_id: commitment_id.into(),
            claims: group_claims,
        })
    }

    pub(crate) fn compute_group_commitment_id(
        chunk_id: ChunkID,
        group_id: ChunkID,
        group_type: GroupType,
    ) -> usize {
        let domain_separator = match group_type {
            GroupType::Incoming => "InputGroup",
            GroupType::Outgoing => "OutputGroup",
        };
        let bytes = chunk_id
            .to_le_bytes()
            .into_iter()
            .chain(group_id.to_le_bytes())
            .chain(domain_separator.as_bytes().iter().copied())
            .collect::<Vec<u8>>();

        // Should be 32 bytes take the first 8
        let output_bytes = <sha2::Sha256 as sha2::Digest>::digest(&bytes);
        let byte_array: [u8; 8] = output_bytes[0..8]
            .try_into()
            .expect("slice with incorrect length");
        usize::from_be_bytes(byte_array)
    }

    // compute the commitment id employed to uniquely identify the commitment for the output group `group_id`
    // of chunk with id `chunk_id`
    pub(crate) fn compute_output_group_commitment_id(
        chunk_id: ChunkID,
        group_id: ChunkID,
    ) -> usize {
        Self::compute_group_commitment_id(chunk_id, group_id, GroupType::Outgoing)
    }

    // compute the commitment id employed to uniquely identify the commitment for the input group `group_id`
    // of chunk with id `chunk_id`
    pub(crate) fn compute_input_group_commitment_id(chunk_id: ChunkID, group_id: ChunkID) -> usize {
        Self::compute_group_commitment_id(chunk_id, group_id, GroupType::Incoming)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(crate) struct ChunkIOCommitments<C> {
    pub(crate) inputs: BTreeMap<ChunkID, C>,
    pub(crate) outputs: BTreeMap<ChunkID, C>,
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
    pub(crate) fn add_to_transcript<
        E: ExtensionField,
        PCS: PolynomialCommitmentScheme<E>,
        T: Transcript<E>,
    >(
        &self,
        chunk_id: ChunkID,
        transcript: &mut T,
    ) -> anyhow::Result<()>
    where
        C: Borrow<PCS::Commitment>,
    {
        self.inputs.iter().try_for_each(|(group_id, commitment)| {
            add_group_commitment_to_transcript::<E, T, PCS>(
                chunk_id,
                *group_id,
                commitment.borrow(),
                transcript,
                GroupType::Incoming,
            )
        })?;
        self.outputs.iter().try_for_each(|(group_id, commitment)| {
            add_group_commitment_to_transcript::<E, T, PCS>(
                chunk_id,
                *group_id,
                commitment.borrow(),
                transcript,
                GroupType::Outgoing,
            )
        })?;
        Ok(())
    }
}

pub(crate) fn add_group_commitment_to_transcript<
    E: ExtensionField,
    T: Transcript<E>,
    PCS: PolynomialCommitmentScheme<E>,
>(
    chunk_id: ChunkID,
    group_id: ChunkID,
    commitment: &PCS::Commitment,
    transcript: &mut T,
    group_type: GroupType,
) -> anyhow::Result<()> {
    let commitment_descriptor = match group_type {
        GroupType::Incoming => format!("Input: {chunk_id}->{group_id}"),
        GroupType::Outgoing => format!("Output: {chunk_id}->{group_id}"),
    };
    transcript.append_message(commitment_descriptor.as_bytes());
    PCS::write_commitment(commitment, transcript)
        .map_err(|e| anyhow!("Error writing input commitment to transcript: {e:?}"))
}

pub type ChunkIOGroup = Vec<EdgeId>;

/// Set of claims for a group of incoming or outgoing edges of a chunk.
/// There is a batched commitment for each group of edges, identifier by `commitment_id`.
/// The claims contain one evaluation claim per committed polynomial in the group
pub(crate) struct GroupIOClaims<E> {
    pub(crate) commitment_id: NodeId,
    pub(crate) claims: Vec<Claim<E>>,
}

pub trait ChunkingStrategy: Clone + Serialize + DeserializeOwned {
    /// Return the ideal number of chunks to split the model into;
    fn ideal_num_chunks<E: ExtensionField>(&self, model: &ModelCtx<E>) -> usize;

    /// Split the set of nodes in `num_chunks` chunks of consecutive nodes for proving
    fn split<E: ExtensionField>(
        &self,
        model: &ModelCtx<E>,
        num_chunks: usize,
    ) -> anyhow::Result<Vec<ChunkedGraph>>;
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct DefaultChunkingStrategy(());

impl ChunkingStrategy for DefaultChunkingStrategy {
    fn ideal_num_chunks<E: ExtensionField>(&self, model: &ModelCtx<E>) -> usize {
        // define a constant `NUM_NODES_PER_CHUNK` that specifies the ideal number of
        // nodes per chunk to be proven. The number of chunks is then computed by
        // ensuring that each chunk has at most `NUM_NODES_PER_CHUNK` nodes
        const NUM_NODES_PER_CHUNK: usize = 3;
        let num_nodes = model.nodes.inner_nodes_count();
        // return num chunks as `ceil(num_nodes / NUM_NODES_PER_CHUNK)`
        num_nodes.div_ceil(NUM_NODES_PER_CHUNK)
    }

    fn split<E: ExtensionField>(
        &self,
        model: &ModelCtx<E>,
        num_chunks: usize,
    ) -> anyhow::Result<Vec<ChunkedGraph>> {
        let num_nodes = model.nodes.inner_nodes_count();

        ensure!(
            num_chunks <= num_nodes,
            "Number of chunks ({num_chunks}) cannot be greater than number of nodes ({num_nodes})"
        );

        let nodes_per_prover = num_nodes / num_chunks;
        // determine the number of chunks that will have an extra node, in order to have exactly `num_chunks`
        // non-empty chunks
        let num_chunks_with_extra_node = num_nodes % num_chunks;

        let mut subgraphs = vec![ChunkedGraph::new(); num_chunks];

        // we sweep over nodes to split the into subgraphs; while doing this, we also compute a map that
        // determines the chunk where a given node is placed in
        let nodes_map = model
            .nodes
            .backward_iter()
            .filter(|(_, node)| node.is_inner())
            .scan(
                (0, 0),
                |(current_chunk, nodes_in_current_chunk), (node_id, _)| {
                    let chunk = *current_chunk;
                    *nodes_in_current_chunk += 1;
                    let max_nodes_in_current_chunk = if chunk < num_chunks_with_extra_node {
                        nodes_per_prover + 1 // in the first `num_chunks_with_extra_node` chunks, there is an extra node
                    } else {
                        nodes_per_prover
                    };
                    if *nodes_in_current_chunk >= max_nodes_in_current_chunk {
                        *current_chunk += 1;
                        *nodes_in_current_chunk = 0;
                    }
                    Some(
                        subgraphs[chunk]
                            .add_node_with_id(node_id, Node::Inner(()))
                            .map(|_| (node_id, chunk)),
                    )
                },
            )
            .collect::<anyhow::Result<HashMap<_, _>>>()?;

        // now we add relevant edges in each subgraph
        add_edges_to_chunk_subgraphs(model, &mut subgraphs, &nodes_map)?;

        Ok(subgraphs)
    }
}

/// A chunking strategy specifically devoted for LLMs.
/// It ensures that the `Add` layers don't cause multiple
/// commitments to be produced for each chunk, which however
/// prevent to split an attention layer across multiple chunks.
/// Thus, it makes sense to use this strategy only when there are
/// enough attention layers to have an high enough number of
/// chunks
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct LLMChunkingStrategy;

impl LLMChunkingStrategy {
    fn add_nodes<'a, E: ExtensionField>(
        &self,
        model: &'a ModelCtx<E>,
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
}

impl ChunkingStrategy for LLMChunkingStrategy {
    fn ideal_num_chunks<E: ExtensionField>(&self, model: &ModelCtx<E>) -> usize {
        // the ideal number of chunks is given by splitting the model at each `Add` node
        // hence, the ideal number of chunks is `num_add_nodes + 2`
        let num_add_nodes = self.add_nodes(model).count();
        num_add_nodes + 2
    }

    fn split<E: ExtensionField>(
        &self,
        model: &ModelCtx<E>,
        num_chunks: usize,
    ) -> anyhow::Result<Vec<ChunkedGraph>> {
        let num_ideal_chunks = self.ideal_num_chunks(model);
        if num_ideal_chunks >= num_chunks {
            // whether we create an extra chunk at the end of the LLM for Argmax and final projection,
            // which are computationally intensive and so it is beneficial to have a standalone chunk for
            // them
            let num_extra_chunks = if num_chunks >= 2 { 1 } else { 0 };
            let mut subgraphs = vec![ChunkedGraph::new(); num_chunks];
            let mut nodes_map = HashMap::new();
            let mut nodes_iter = model
                .nodes
                .backward_iter()
                .filter(|(_, node)| node.is_inner());
            let first_add_chunk = if num_extra_chunks > 0 {
                // the first chunk is given by Logits and previous EinSum node
                let mut next_node = |node_type: String| {
                    let (node_id, node) = nodes_iter.next().expect("No node found in LLM model?");
                    ensure!(
                        node.as_inner().expect("It's an inner node").variant_name() == node_type,
                        "Last node of LLM is not {node_type}"
                    );
                    subgraphs[0].add_node_with_id(node_id, Node::Inner(()))?;
                    Ok(node_id)
                };
                nodes_map.insert(next_node("Logits".to_string())?, 0);
                nodes_map.insert(next_node("EinSum".to_string())?, 0);
                1
            } else {
                0
            };

            // first, find add layers there are in the model
            let add_nodes = self.add_nodes(model).collect::<HashSet<_>>();
            let num_add_nodes = add_nodes.len();
            let num_add_chunks = num_chunks - num_extra_chunks;
            // we check if there are enough `Add` nodes to create `num_add_chunks` chunks. Note that
            // the number of chunks we can create at most is `num_add_nodes + 1`, because the `Add`
            // nodes are the split points of the graph: thus, if there are `num_add_nodes` split
            // points, we get `num_add_nodes + 1` chunks of the graph
            ensure!(num_add_nodes + 1 >= num_add_chunks);
            // there are enough add nodes to create chunks, so we split the graph at each add node,
            // up until num_chunks are created
            debug!("Chunking an LLM with {num_add_nodes} Add nodes in {num_chunks} chunks");
            // `num_add_chunks` might be smaller than `num_add_nodes`. To get at most `num_add_chunks`, we
            // split the graph after every `add_nodes_per_chunk` `Add` nodes found, where
            // `add_nodes_per_chunk = (num_add_nodes +1)/num_add_chunks`
            let add_nodes_per_chunk = (num_add_nodes + 1) / num_add_chunks;

            // we sweep over nodes to split the into subgraphs; the node where there is a partition between chunks
            // is the add node, because this should guarantee that the source
            // nodes of every add node ends up in the same chunk. Indeed, an `Add` node in the transformers, except
            // for the first one appearing in the model, takes its input from the previous `Add`` node and from the
            // previous node in the model. Therefore, if we split at `Add` nodes, then the `Add` node and the other
            // source node of the next `Add` node will alwayes end up in the same chunk.
            // While partitioning the nodes in chunks, we also compute a map that determines the chunk where a
            // given node is placed in
            nodes_iter.try_fold(0, |mut add_nodes_found, (node_id, _)| {
                let chunk = add_nodes_found / add_nodes_per_chunk + first_add_chunk;
                // ensure that chunk is at most num_chunks - 1
                let chunk = chunk.min(num_chunks - 1);
                subgraphs[chunk].add_node_with_id(node_id, Node::Inner(()))?;
                nodes_map.insert(node_id, chunk);
                if add_nodes.contains(&node_id) {
                    add_nodes_found += 1;
                }
                anyhow::Ok(add_nodes_found)
            })?;

            // we check that the source nodes of each `Add` node end up in the same chunk
            add_nodes.iter().try_for_each(|add_node_id| {
                let chunks_for_inputs = model.nodes.neighbors(*add_node_id, Direction::Incoming)
                    .map(|(edge_id, edge)| {
                        let source_id = edge.source();
                        let source_node = model.nodes.node(source_id).ok_or(anyhow!(
                            "Source node {source_id} of edge {edge_id} not found in model"
                        ))?;
                        Ok(if source_node.is_input() {
                            num_chunks - 1 // if it's an input of the model, the other input must be in the last chunk
                        } else {
                            *nodes_map.get(&source_id).ok_or(
                                anyhow!("Node {source_id} not assigned to any chunk")
                            )?
                        })
                    })
                    .collect::<anyhow::Result<Vec<_>>>()?;
                ensure!(
                    chunks_for_inputs.iter().all_equal(),
                    "Found source nodes for add node {add_node_id} edges to be in different chunks: {:?}", chunks_for_inputs
                );
                Ok(())
            })?;

            // now we add relevant edges in each subgraph
            add_edges_to_chunk_subgraphs(model, &mut subgraphs, &nodes_map)?;

            Ok(subgraphs)
        } else {
            DefaultChunkingStrategy::default().split(model, num_chunks)
        }
    }
}

/// Initialize the transcript to prove a given chunk, using the challenge squeezed from
/// the initial shared transcript
pub(crate) fn initialize_transcript_for_chunk<
    E: ExtensionField,
    T: Transcript<E> + InitTranscript,
>(
    challenge: E,
) -> T {
    let mut transcript = T::new(T::InitData::from(b"model_chunk_proving"));
    transcript.append_field_element_ext(&challenge);
    transcript
}

fn add_edges_to_chunk_subgraphs<E: ExtensionField>(
    model: &ModelCtx<E>,
    subgraphs: &mut [ChunkedGraph],
    nodes_map: &HashMap<NodeId, usize>,
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
        if source_node.as_input().is_none() {
            let chunk = nodes_map
                .get(&source_id)
                .ok_or(anyhow!("Node {source_id} not assigned to any chunk"))?;
            subgraphs[*chunk].add_edge_raw_with_id(*edge_id, edge.clone())?;
            if let Some(o) = target_node.as_output() {
                // we also add output node `o` to the chunk subgraph
                return subgraphs[*chunk].add_node_with_id(target_id, Node::Output(*o));
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
                subgraphs[*chunk].add_node_with_id(source_id, Node::Input(*i))?;
            }
        }
        anyhow::Ok(())
    })
}

#[cfg(test)]
mod test {
    use anyhow::Context;
    use ff_ext::GoldilocksExt2;

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

    type F = GoldilocksExt2;

    #[test]
    fn test_default_chunking_strategy() {
        let num_dense_layers = 5;
        let (model, _input) = Model::random(num_dense_layers).unwrap();
        model.describe();
        let (prover_ctx, _verifier_ctx) = model
            .generate_contexts::<F, Pcs<F>>()
            .expect("unable to generate contexts");
        let strategy = DefaultChunkingStrategy(());
        let chunks = strategy.split(&prover_ctx.model_ctx, 3).unwrap();

        assert_eq!(chunks.len(), 3);
    }

    #[test]
    fn test_llm_chunking_strategy() -> anyhow::Result<()> {
        init_test_logging("debug");
        const MAX_CONTEXT: usize = 10;
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

            let ctx = driver
                .context::<GoldilocksExt2, Pcs<GoldilocksExt2>>()?
                .with_max_context(MAX_CONTEXT);

            Ok(ctx)
        })?;

        let strategy = LLMChunkingStrategy;

        for num_chunks in 1..50 {
            let chunks = strategy.split(&prover_ctx.model_ctx, num_chunks).unwrap();

            assert_eq!(chunks.len(), num_chunks);
        }

        Ok(())
    }
}
