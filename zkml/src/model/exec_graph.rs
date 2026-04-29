use std::{collections::HashMap, fmt::Debug, ops::Deref};

use anyhow::{anyhow, bail, ensure};
use ark_ff::PrimeField;
use dp_crypto::arkyper::{CommitmentScheme, transcript::Transcript};
use itertools::Itertools;
use serde::{Deserialize, Serialize};
use tenstore::GenStore;

use crate::{
    Element, IO, InitTranscript, Proof, Prover, ProverContext, Tensor,
    graph::{
        Node, NodeInput,
        scheduler::{ExecNode, IntoColor},
    },
    iop::{
        chunking::ChunkingStrategy,
        prover_graph::{LocalProverCtx, ProverGraphIO, ProverGraphNode},
    },
    model::{Model, Trace, llm::Driver, trace::SplittedNodesInfo},
    quantization::ModelMetadata,
};

/// Context for the execution graph used for distributed proving
pub struct ExecGraphCtx<F: PrimeField, PCS: CommitmentScheme> {
    pub(crate) serializable_ctx: SerializableGraphCtx<F, PCS>,
    pub(crate) store: GenStore,
}

impl<F: PrimeField, PCS: CommitmentScheme> AsRef<SerializableGraphCtx<F, PCS>>
    for ExecGraphCtx<F, PCS>
{
    fn as_ref(&self) -> &SerializableGraphCtx<F, PCS> {
        &self.serializable_ctx
    }
}

/// This crate supports running generic models (CNNs, MLPs, etc) and auto regressive models (LLMs).
/// This enum reflects the different types of inference required to support these use cases.
#[derive(Debug, Serialize, Deserialize)]
pub enum InferenceEngine {
    Generic(Model<Element>),
    LLM(Driver<Element>),
}

impl InferenceEngine {
    pub fn run(
        &self,
        mut input: Vec<Tensor<Element>>,
        store: &mut GenStore,
        split_node_info: &SplittedNodesInfo,
    ) -> anyhow::Result<Trace<Element>> {
        match self {
            InferenceEngine::Generic(model) => {
                model.run_with_split_nodes_info(input, store, Some(split_node_info))
            }
            InferenceEngine::LLM(driver) => {
                ensure!(
                    input.len() == 1,
                    "LLM inference only supports one sequence of tokens - batch inference is not supported"
                );
                let input = input.pop().expect("size validated above");
                driver.run_elements_with_split_info(input, store, Some(split_node_info))
            }
        }
    }

    /// Returns the LLM Driver's max_context, if this is an LLM engine.
    pub fn llm_max_context(&self) -> Option<usize> {
        match self {
            InferenceEngine::LLM(driver) => driver.max_context(),
            InferenceEngine::Generic(_) => None,
        }
    }

    /// Update the max_context window for LLM inference.
    pub fn set_llm_max_context(&mut self, max_context: usize) {
        if let InferenceEngine::LLM(driver) = self {
            driver.with_max_context(max_context);
        }
    }

    /// Necessary to return the raw model when doing the proving - as proving is the same
    /// for both generic and LLM models
    pub fn model(&self) -> &Model<Element> {
        match self {
            InferenceEngine::Generic(model) => model,
            InferenceEngine::LLM(driver) => &driver.model,
        }
    }
}

/// Serializable version of the execution graph context.
/// Once deserialized, this structure can be converted to the full `ExecGraphCtx` to
/// be used to run the execution graph
#[derive(Debug, Serialize, Deserialize)]
#[serde(bound(
    serialize = "F: ark_serialize::CanonicalSerialize",
    deserialize = "F: ark_serialize::CanonicalDeserialize"
))]
pub struct SerializableGraphCtx<F: PrimeField, PCS: CommitmentScheme> {
    pub(crate) ctx: ProverContext<'static, F, PCS>,
    pub(crate) engine: InferenceEngine,
    /// Model metadata for input/output scaling. Required for input quantization.
    /// This field is optional for backward compatibility with existing serialized data.
    #[serde(default)]
    pub(crate) metadata: Option<ModelMetadata>,
}

impl<F: PrimeField, PCS: CommitmentScheme> SerializableGraphCtx<F, PCS> {
    pub fn new(ctx: ProverContext<'static, F, PCS>, engine: InferenceEngine) -> Self {
        Self {
            ctx,
            engine,
            metadata: None,
        }
    }

    /// Create a new SerializableGraphCtx with model metadata for input scaling.
    /// The metadata is required when the context will be cached and reused for
    /// different inputs.
    pub fn new_with_metadata(
        ctx: ProverContext<'static, F, PCS>,
        engine: InferenceEngine,
        metadata: ModelMetadata,
    ) -> Self {
        Self {
            ctx,
            engine,
            metadata: Some(metadata),
        }
    }

    /// Get the model metadata for input/output scaling.
    /// Returns an error if metadata is not available (older cached context).
    pub fn metadata(&self) -> anyhow::Result<&ModelMetadata> {
        self.metadata
            .as_ref()
            .ok_or_else(|| anyhow!("Model metadata not available in cached context"))
    }

    /// Get a reference to the inference engine.
    pub fn engine(&self) -> &InferenceEngine {
        &self.engine
    }

    /// Update the LLM max_context window on the underlying inference engine.
    pub fn set_llm_max_context(&mut self, max_context: usize) {
        self.engine.set_llm_max_context(max_context);
    }

    /// Build the full execution graph context from `SerializableGraphCtx`,
    /// attaching the given `GenStore`.
    pub fn to_full_ctx(self, store: GenStore) -> ExecGraphCtx<F, PCS> {
        ExecGraphCtx {
            serializable_ctx: self,
            store,
        }
    }
}

impl<F: PrimeField, PCS: CommitmentScheme> Deref for ExecGraphCtx<F, PCS> {
    type Target = SerializableGraphCtx<F, PCS>;

    fn deref(&self) -> &Self::Target {
        &self.serializable_ctx
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct InferenceIO {
    input_tensors: Vec<Tensor<Element>>,
    #[serde(skip)]
    store: GenStore,
}

/// Input/output data for nodes of the execution graph used for distributed proving
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(bound(
    serialize = "F: ark_serialize::CanonicalSerialize",
    deserialize = "F: ark_serialize::CanonicalDeserialize"
))]
#[allow(clippy::large_enum_variant)]
pub enum ExecGraphIO<F: PrimeField, PCS: CommitmentScheme> {
    // Input for inference task
    InferenceInput(InferenceIO),
    // Input for prover graph nodes
    Prover(ProverGraphIO<F, PCS>),
    // Model IO output to be provided to the verifier
    ModelIO(IO<F>),
}

/// A serializable intermediate output for GW mediated distributed execution.
///
/// This struct mirrors `PartitionOutput` but uses concrete types that can be
/// serialized/deserialized across network boundaries without lifetime parameters.
/// It is used for storing intermediate outputs in the GW DB and passing
/// them between workers.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(bound(
    serialize = "F: ark_serialize::CanonicalSerialize",
    deserialize = "F: ark_serialize::CanonicalDeserialize"
))]
pub struct SerializablePartitionOutput<F: PrimeField, PCS: CommitmentScheme> {
    /// The color/chunk_id of the partition that produced this output
    pub from: usize,
    /// The color/chunk_id of the partition that should receive this output.
    /// `None` indicates this is a final output of the entire computation.
    pub to: Option<usize>,
    /// The actual output data produced by the partition
    pub output: ExecGraphIO<F, PCS>,
}

impl<F: PrimeField, PCS: CommitmentScheme> SerializablePartitionOutput<F, PCS> {
    /// Create a new serializable partition output
    pub fn new(from: usize, to: Option<usize>, output: ExecGraphIO<F, PCS>) -> Self {
        Self { from, to, output }
    }

    /// Returns true if this output represents the final result of the computation.
    pub fn is_final_output(&self) -> bool {
        self.to.is_none()
    }
}

impl<F: PrimeField, PCS: CommitmentScheme> ExecGraphIO<F, PCS> {
    /// Attach the store to the trace instances found in `ChunkProverII`.
    /// This method is needed when the IO needs to be serialized/deserialized (e.g.,
    /// in a distrbiuted setting), as the store cannot be serialized.
    /// Therefore, a referencet to a store needs to be attached to the trace after
    /// the inputs have been deserialized
    pub fn attach_store(&mut self, store: GenStore) {
        match self {
            Self::InferenceInput(io) => io.store = store,
            Self::Prover(io) => io.attach_store(store),
            Self::ModelIO(_) => (),
        }
    }
}

/// Node of the execution graph used for distributed proving
#[derive(Clone, Serialize, Deserialize)]
#[serde(bound(
    serialize = "F: ark_serialize::CanonicalSerialize",
    deserialize = "F: ark_serialize::CanonicalDeserialize"
))]
pub enum ExecGraphNode<'a, 'b, F: PrimeField, T, PCS: CommitmentScheme> {
    /// Task for inference of the model
    Inference(SplittedNodesInfo),
    Prover(ProverGraphNode<'a, 'b, F, T, PCS>),
}

impl<'a, 'b, F: PrimeField, T, PCS: CommitmentScheme> Debug for ExecGraphNode<'a, 'b, F, T, PCS> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ExecGraphNode::Inference(split_info) => {
                write!(f, "ExecGraphNode::Inference({split_info:?})")
            }
            ExecGraphNode::Prover(node) => write!(f, "ExecGraphNode::Prover({:?})", node),
        }
    }
}

impl<'a, 'b, F, T, PCS> ExecNode for ExecGraphNode<'a, 'b, F, T, PCS>
where
    F: PrimeField,
    PCS: CommitmentScheme<Field = F> + 'static,
    T: Transcript + InitTranscript,
{
    type IO = ExecGraphIO<F, PCS>;

    type Context = ExecGraphCtx<F, PCS>;

    fn describe(&self) -> String {
        match self {
            ExecGraphNode::Inference(split_info) => format!(
                "Inference, number of splitted nodes: {}",
                split_info.splitted_nodes.inner_nodes.len()
            ),
            ExecGraphNode::Prover(generic_exec_graph_node) => {
                format!("Prover node({})", generic_exec_graph_node.describe())
            }
        }
    }

    fn run(&self, ctx: &Self::Context, mut inputs: Vec<Self::IO>) -> anyhow::Result<Vec<Self::IO>> {
        // first, attach the store to all inputs
        inputs
            .iter_mut()
            .for_each(|inp| inp.attach_store(ctx.store.clone()));
        match self {
            ExecGraphNode::Inference(split_info) => {
                ensure!(
                    inputs.len() == 1,
                    "Expected 1 input for inference node in distributed graph, found {}",
                    inputs.len()
                );
                let ExecGraphIO::InferenceInput(mut input) = inputs.pop().unwrap() else {
                    bail!("Expected tensors as input for inference task")
                };
                let trace = ctx
                    .engine
                    .run(input.input_tensors, &mut input.store, split_info)?;
                let io: IO<F> = trace.to_verifier_io()?;
                Ok(vec![
                    ExecGraphIO::Prover(ProverGraphIO::ProverSplitInput(trace)),
                    ExecGraphIO::ModelIO(io),
                ])
            }
            ExecGraphNode::Prover(generic_exec_graph_node) => {
                let local_ctx = LocalProverCtx::new(&ctx.ctx, ctx.engine.model());
                Ok(generic_exec_graph_node
                    .run(
                        &local_ctx,
                        inputs
                            .into_iter()
                            .map(|inp| {
                                let ExecGraphIO::Prover(prover_input) = inp else {
                                    bail!(
                                        "Expected prover input for prover node in execution graph"
                                    )
                                };
                                Ok(prover_input)
                            })
                            .collect::<anyhow::Result<Vec<_>>>()?,
                    )?
                    .into_iter()
                    .map(ExecGraphIO::Prover)
                    .collect())
            }
        }
    }
}

pub type ExecGraph<'a, 'b, F, T, PCS> =
    crate::graph::scheduler::ExecGraph<ExecGraphNode<'a, 'b, F, T, PCS>, usize>;

pub fn build_execution_graph<F, T, PCS, S>(
    ctx: &ExecGraphCtx<F, PCS>,
    num_chunks: Option<usize>,
    chunking_strategy: S,
) -> anyhow::Result<ExecGraph<'_, '_, F, T, PCS>>
where
    F: PrimeField,
    PCS: CommitmentScheme<Field = F> + 'static,
    S: ChunkingStrategy,
    T: Transcript + InitTranscript,
{
    let (chunks, split_info) = ctx.ctx.split_in_chunks(num_chunks, chunking_strategy)?;
    // The full execution graph is obtained by pre-pending the node related to inference task
    // to the prover execution graph. The inference node produces the full trace, which is the
    // input needed by the prover graph
    let mut prover_graph: ExecGraph<F, T, PCS> =
        Prover::<F, T, PCS>::build_execution_graph(chunks)?
            // we need to convert prover graph nodes to `ExecGraphNode`
            .try_into_map_forward(|_node_id, node, _feeds| {
                let inner_node = node
                    .into_inner()
                    .expect("all nodes in execution graph should be inner nodes");
                let color = *inner_node.color();
                Ok(Node::Inner(
                    ExecGraphNode::Prover(inner_node.node).colored(color),
                ))
            })?
            // we need to convert weights of edge of the prover graph, which are expected to carry
            // the inputs/outputs of the connected nodes,  to `ExecGraphIO`
            .try_map_weights(|w| Ok(ExecGraphIO::Prover(w)))?;
    let source_node = prover_graph.source_nodes().exactly_one().map_err(|e| {
        anyhow!(
            "Expected 1 source node for prover execution graph, found {}",
            e.count()
        )
    })?;
    let source_node_color = prover_graph
        .node(source_node)
        .ok_or(anyhow!(
            "Source node {source_node} not found in prover execution graph"
        ))?
        .as_inner()
        .expect("All nodes in execution graph must be Inner nodes")
        .color();
    let inference_node_id =
        prover_graph.add_inner(ExecGraphNode::Inference(split_info).colored(*source_node_color))?;
    // link inference node id with source node of prover graph
    prover_graph.add_consecutive_edge(inference_node_id, source_node, None)?;
    Ok(prover_graph)
}

pub fn graph_inputs<F, T, PCS>(
    input_tensors: Vec<Tensor<Element>>,
    store: GenStore,
    exec_graph: &ExecGraph<F, T, PCS>,
) -> anyhow::Result<HashMap<NodeInput, ExecGraphIO<F, PCS>>>
where
    F: PrimeField,
    T: Transcript + InitTranscript,
    PCS: CommitmentScheme<Field = F> + 'static,
{
    let source_node = exec_graph.source_nodes().exactly_one().map_err(|e| {
        anyhow!(
            "Expected 1 source node for execution graph, found {}",
            e.count()
        )
    })?;
    Ok([(
        NodeInput::new(source_node, 0),
        ExecGraphIO::InferenceInput(InferenceIO {
            input_tensors,
            store,
        }),
    )]
    .into())
}

pub type ProofWithIO<F, PCS> = (Proof<F, PCS>, IO<F>);

/// Utility method to extract the `ProofWithIO` from the outputs produced by the execution graph
pub fn extract_graph_outputs<F: PrimeField, PCS: CommitmentScheme>(
    outputs: impl IntoIterator<Item = ExecGraphIO<F, PCS>>,
) -> anyhow::Result<ProofWithIO<F, PCS>> {
    let (proof_opt, io_opt) =
        outputs
            .into_iter()
            .try_fold((None, None), |(proof, io), output| match output {
                ExecGraphIO::Prover(ProverGraphIO::FinalProof(proof)) => Ok((Some(proof), io)),
                ExecGraphIO::ModelIO(io) => Ok((proof, Some(io))),
                _ => bail!("Invalid output type received"),
            })?;
    ensure!(proof_opt.is_some(), "No final proof received");
    ensure!(io_opt.is_some(), "No model IO received");
    Ok((proof_opt.unwrap(), io_opt.unwrap()))
}

#[cfg(test)]
mod tests {
    use anyhow::Context;
    use crossbeam_channel::unbounded;
    use dp_crypto::arkyper::transcript::blake3::Blake3Transcript;
    use rayon::scope;
    use serde::{Serialize, de::DeserializeOwned};
    use std::collections::{BTreeMap, HashMap};
    use tenstore::GenStore;

    use crate::{
        graph::{
            NodeInput, NodeOutput,
            executor::Executor,
            scheduler::{ExecNode, GraphScheduler, ReadyNode, ReleasePolicy},
        },
        iop::chunking::DefaultChunkingStrategy,
        model::{
            Model,
            exec_graph::{
                ExecGraphCtx, InferenceEngine, SerializableGraphCtx, build_execution_graph,
                extract_graph_outputs, graph_inputs,
            },
        },
        testing::Pcs,
        verify,
    };

    type F = ark_bn254::Fr;
    type T = Blake3Transcript;

    /// Utility trait to convert a serializable type `T` to bytes
    /// independently from the given serializer being employed (e.g., bincode, serde_json)
    pub(crate) trait TryToBytes<T: Serialize> {
        fn try_to_bytes(&self, to_serialize: &T) -> anyhow::Result<Vec<u8>>;
    }

    /// Utility trait to convert from bytes a deserializable type `T`
    /// independently from the given deserializer being employed (e.g., bincode, serde_json)
    pub(crate) trait TryFromBytes<T: DeserializeOwned> {
        fn try_from_bytes(&self, bytes: &[u8]) -> anyhow::Result<T>;
    }

    struct BincodeSerializer;

    impl<T: Serialize> TryToBytes<T> for BincodeSerializer {
        fn try_to_bytes(&self, to_serialize: &T) -> anyhow::Result<Vec<u8>> {
            Ok(bincode::serde::encode_to_vec(
                to_serialize,
                bincode::config::standard(),
            )?)
        }
    }

    impl<T: DeserializeOwned> TryFromBytes<T> for BincodeSerializer {
        fn try_from_bytes(&self, bytes: &[u8]) -> anyhow::Result<T> {
            Ok(bincode::serde::decode_from_slice(bytes, bincode::config::standard())?.0)
        }
    }

    struct SerializedThreadPoolExecutor;

    impl<N, C> Executor<N, C> for SerializedThreadPoolExecutor
    where
        N::IO: Clone + Send + Sync + Serialize + DeserializeOwned,
        C: Clone + PartialEq + Send + Sync + Serialize + DeserializeOwned,
        N::Context: Sync, // Context is shared (not owned) across threads
        N: ExecNode + Clone + Send + Sync + Serialize + DeserializeOwned,
    {
        type Config = ();

        fn run(
            _config: &Self::Config,
            scheduler: GraphScheduler<N, C>,
            input_data: HashMap<NodeInput, N::IO>,
            context: &N::Context,
        ) -> anyhow::Result<BTreeMap<NodeOutput, N::IO>> {
            // Release all to keep the threadpool busy
            let mut scheduler = scheduler.with_release_policy(ReleasePolicy::All);
            let mut ready_nodes = scheduler.init_nodes(input_data)?;

            scope(move |s| -> anyhow::Result<BTreeMap<NodeOutput, N::IO>> {
                let mut outputs = BTreeMap::new();
                let (tx, rx) = unbounded();

                while !scheduler.is_done() {
                    for node in ready_nodes {
                        let tx = tx.clone();

                        // TEST: ensure the node can be encoded
                        let node_encoded = BincodeSerializer
                            .try_to_bytes(&node)
                            .context("Failed to serialize graph node")?;

                        s.spawn(move |_| {
                            // TEST: ensure the node can be decoded
                            let decoded: anyhow::Result<ReadyNode<N, C>> = BincodeSerializer
                                .try_from_bytes(&node_encoded)
                                .context("Failed to deserialize graph node");

                            let mut node = match decoded {
                                Ok(decoded) => decoded,
                                Err(err) => {
                                    tx.send((node.node_id, Err(err))).expect(
                                        "Sender channel closed unexpectedly, can not send error",
                                    );
                                    return;
                                }
                            };

                            match node.run(context) {
                                Ok(output) => tx.send((node.node_id, Ok(output))).expect(
                                    "Sender channel closed unexpectedly, can not send node result",
                                ),
                                err @ Err(_) => tx.send((node.node_id, err)).expect(
                                    "Sender channel closed unexpectedly, can not send error",
                                ),
                            };
                        });
                    }

                    let (node_id, output) = rx
                        .recv()
                        .expect("Receiver channel closed unexpectedly, can not read results");

                    match output {
                        Ok(output) => {
                            let graph_outputs = scheduler
                                .mark_done(node_id, &output)
                                .context("Failed to mark node with scheduler")?;
                            outputs.extend(graph_outputs);
                            ready_nodes = scheduler.next_ready_nodes()?;
                        }
                        Err(err) => {
                            return Err(err.context("Error running node {node_idx:?}"));
                        }
                    }
                }
                Ok(outputs)
            })
        }
    }

    /// Test the data can be serialized / deserialized as this is necessary for distributed proving.
    #[test]
    fn test_exec_graph_data_serialization() -> anyhow::Result<()> {
        let (model, input) = Model::random(6)?;
        let (prover_ctx, verifier_ctx) = model
            .generate_contexts::<F, Pcs>()
            .expect("unable to generate contexts");
        let store: GenStore = Default::default();
        let num_chunks = Some(3);
        let ctx = ExecGraphCtx {
            serializable_ctx: SerializableGraphCtx::new(
                prover_ctx,
                InferenceEngine::Generic(model),
            ),
            store: store.clone(),
        };
        let graph = build_execution_graph::<_, T, _, _>(
            &ctx,
            num_chunks,
            DefaultChunkingStrategy::new(
                input
                    .iter()
                    .map(|inp| inp.unpadded_shape().clone())
                    .collect(),
            ),
        )?;
        let inputs = graph_inputs(input, store.clone(), &graph)?;

        let serialized_ctx = BincodeSerializer.try_to_bytes(ctx.as_ref())?;
        let deserialized_ctx: SerializableGraphCtx<F, Pcs> =
            BincodeSerializer.try_from_bytes(&serialized_ctx)?;

        let ctx = deserialized_ctx.to_full_ctx(store.clone());

        let scheduler = GraphScheduler::new(graph);
        let outputs = SerializedThreadPoolExecutor::run(&(), scheduler, inputs, &ctx)?;
        let (proof, io) = extract_graph_outputs(outputs.into_values())?;

        verify::<_, T, _>(&verifier_ctx, proof, io)
    }
}
