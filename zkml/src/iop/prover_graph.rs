use std::marker::PhantomData;

use crate::{
    Proof, measure,
    model::{Model, Trace},
};
use anyhow::{bail, ensure};
use ff_ext::ExtensionField;
use mpcs::PolynomialCommitmentScheme;
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use tenstore::GenStore;
use transcript::Transcript;

use crate::{
    Element, InitTranscript, Prover, ProverContext,
    graph::scheduler::ExecNode,
    iop::{
        ChunkProof,
        chunking::{ModelChunk, initialize_transcript_for_chunk},
    },
};

// A node in the execution graph of chunks whose job is to prove a chunk.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SplitNode<'a, 'b, E, T, PCS> {
    chunks: Vec<ModelChunk>,
    _phantom: PhantomData<(E, T, PCS, &'a (), &'b ())>,
}

impl<'a, 'b, E, T, PCS> SplitNode<'a, 'b, E, T, PCS> {
    pub(crate) fn new(chunks: Vec<ModelChunk>) -> Self {
        Self {
            chunks,
            _phantom: Default::default(),
        }
    }
}

/// Nodes in the execution graph for the chunk prover
#[derive(Clone, Serialize, Deserialize)]
#[serde(bound(serialize = "E: Serialize", deserialize = "E: DeserializeOwned"))]
#[allow(clippy::large_enum_variant)]
pub enum ProverGraphNode<'a, 'b, E: ExtensionField, T, PCS> {
    // Executor for the task of splitting the trace in multiple chunks and setting up the distributed proving
    ProverSplit(SplitNode<'a, 'b, E, T, PCS>),
    // Executor for the task of proving each chunk
    ChunkProver(usize),
    // Final node to just collect the proofs and return a single coherent proof
    Final,
}

impl<'a, 'b, E: ExtensionField, T, PCS> std::fmt::Debug for ProverGraphNode<'a, 'b, E, T, PCS> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ProverGraphNode::ProverSplit(split_node) => {
                write!(f, "Node(Preprocess for {} chunks)", split_node.chunks.len())
            }
            ProverGraphNode::ChunkProver(node_id) => write!(f, "Node(Chunk({node_id}))"),
            ProverGraphNode::Final => write!(f, "Node(Final)"),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(bound(serialize = "E: Serialize", deserialize = "E: DeserializeOwned"))]
pub struct ChunkProverInput<E: ExtensionField> {
    chunk: ModelChunk,
    trace: Trace<Element>,
    challenge: E,
}

/// Input/Output data for the nodes of the chunk prover execution graph
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(bound(serialize = "E: Serialize", deserialize = "E: DeserializeOwned"))]
#[allow(clippy::large_enum_variant)]
pub enum ProverGraphIO<E: ExtensionField, PCS: PolynomialCommitmentScheme<E>> {
    // Input for the prover split task
    ProverSplitInput(Trace<Element>),
    // Input for the task of proving a single chunk
    ChunkProveInput(ChunkProverInput<E>),
    // Output data produced for a given chunk,
    ChunkProof(ChunkProof<E, PCS>),
    // Final proof to be outputted by the final node
    FinalProof(Proof<E, PCS>),
}

impl<E: ExtensionField, PCS: PolynomialCommitmentScheme<E>> ProverGraphIO<E, PCS> {
    /// Attach the store to the trace instances found in `ChunkProverII`.
    /// This method is needed when the IO needs to be serialized/deserialized (e.g.,
    /// in a distrbiuted setting), as the store cannot be serialized.
    /// Therefore, a referencet to a store needs to be attached to the trace after
    /// the inputs have been deserialized
    pub fn attach_store(&mut self, store: GenStore) {
        match self {
            ProverGraphIO::ProverSplitInput(trace) => trace.attach_store(store),
            ProverGraphIO::ChunkProveInput(chunk_prover_input) => {
                chunk_prover_input.trace.attach_store(store)
            }
            ProverGraphIO::ChunkProof(_) => (),
            ProverGraphIO::FinalProof(_) => (),
        }
    }
}

/// Global context for each chunk prover
pub struct LocalProverCtx<'a, 'b, E: ExtensionField, PCS: PolynomialCommitmentScheme<E>>
where
    PCS::CommitmentWithWitness: Serialize + DeserializeOwned,
{
    pub(crate) ctx: &'a ProverContext<E, PCS>,
    pub(crate) model: &'b Model<Element>,
}

impl<'a, 'b, E: ExtensionField, PCS: PolynomialCommitmentScheme<E>> LocalProverCtx<'a, 'b, E, PCS>
where
    PCS::CommitmentWithWitness: Serialize + DeserializeOwned,
{
    pub fn new(ctx: &'a ProverContext<E, PCS>, model: &'b Model<Element>) -> Self {
        Self { ctx, model }
    }
}

impl<'a, 'b, E, T, PCS> ExecNode for ProverGraphNode<'a, 'b, E, T, PCS>
where
    E: ExtensionField,
    T: Transcript<E> + InitTranscript,
    PCS: PolynomialCommitmentScheme<E> + Send + Sync + 'static,
    PCS::CommitmentWithWitness: Serialize + DeserializeOwned + Send + Sync,
    PCS::ProverParam: Send + Sync,
{
    type IO = ProverGraphIO<E, PCS>;

    type Context = LocalProverCtx<'a, 'b, E, PCS>;

    fn describe(&self) -> String {
        match self {
            ProverGraphNode::ProverSplit(split_node) => {
                format!("Split node for {} chunks", split_node.chunks.len())
            }
            ProverGraphNode::ChunkProver(node_id) => {
                format!("Chunk prover executor {node_id}")
            }
            ProverGraphNode::Final => "Final node".into(),
        }
    }

    fn run(&self, ctx: &Self::Context, mut inputs: Vec<Self::IO>) -> anyhow::Result<Vec<Self::IO>> {
        match self {
            ProverGraphNode::ProverSplit(split_node) => {
                ensure!(
                    inputs.len() == 1,
                    "Expected 1 input for preprocess proving step, found {} inputs",
                    inputs.len()
                );
                let ProverGraphIO::ProverSplitInput(full_trace) = inputs.pop().unwrap() else {
                    bail!("Expected input for chunk prover executor")
                };
                measure::r("chunk_split", || {
                    let mut transcript = Prover::<E, T, PCS>::initialise_transcript(ctx.ctx)?;
                    // squeeze the common challenge to initialize the transcript for each cbunk
                    let challenge = transcript.read_challenge();
                    split_node
                        .chunks
                        .iter()
                        .map(|chunk| {
                            let chunk_trace = chunk.chunk_trace(&full_trace)?;
                            Ok(ProverGraphIO::ChunkProveInput(ChunkProverInput {
                                chunk: chunk.clone(),
                                trace: chunk_trace,
                                challenge: challenge.elements,
                            }))
                        })
                        .collect()
                })
            }
            ProverGraphNode::ChunkProver(_) => {
                ensure!(
                    inputs.len() == 1,
                    "Expected 1 input for chunk executor, found {} inputs",
                    inputs.len()
                );
                let ProverGraphIO::ChunkProveInput(input) = inputs.pop().unwrap() else {
                    bail!("Expected input for chunk prover executor")
                };
                let chunk_proof = measure::r_if_bigger("chunk_prover", || {
                    let mut transcript: T = initialize_transcript_for_chunk(input.challenge);
                    let prover = Prover::new(ctx.ctx, &mut transcript);
                    let chunk_layers = input.chunk.chunk_layers(ctx.model);
                    prover.prove_chunk(input.chunk.clone(), &input.trace, &chunk_layers)
                })?;
                // initialise a prover for the given chunk

                Ok(vec![ProverGraphIO::ChunkProof(chunk_proof?)])
            }
            ProverGraphNode::Final => {
                let outputs = inputs
                    .into_iter()
                    .map(|output| {
                        if let ProverGraphIO::ChunkProof(chunk_proof) = output {
                            Ok(chunk_proof)
                        } else {
                            bail!("Invalid output found after running execution graph")
                        }
                    })
                    .collect::<anyhow::Result<Vec<_>>>()?;
                Ok(vec![ProverGraphIO::FinalProof(outputs.into())])
            }
        }
    }
}

pub(crate) type ProverGraph<'a, 'b, E, T, PCS> =
    crate::graph::scheduler::ExecGraph<ProverGraphNode<'a, 'b, E, T, PCS>, usize>;
