use std::{collections::HashMap, fmt::Debug, ops::Deref};

use anyhow::{anyhow, bail, ensure};
use ff_ext::ExtensionField;
use itertools::Itertools;
use mpcs::PolynomialCommitmentScheme;
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use tenstore::GenStore;
use transcript::Transcript;

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
    model::{Model, Trace, llm::Driver},
};

/// Context for the execution graph used for distributed proving
pub struct ExecGraphCtx<E: ExtensionField, PCS: PolynomialCommitmentScheme<E>>
where
    PCS::CommitmentWithWitness: Serialize + DeserializeOwned,
{
    pub(crate) serializable_ctx: SerializableGraphCtx<E, PCS>,
    pub(crate) store: GenStore,
}

impl<E: ExtensionField, PCS: PolynomialCommitmentScheme<E>> AsRef<SerializableGraphCtx<E, PCS>>
    for ExecGraphCtx<E, PCS>
where
    PCS::CommitmentWithWitness: Serialize + DeserializeOwned,
{
    fn as_ref(&self) -> &SerializableGraphCtx<E, PCS> {
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
    pub fn run<E: ExtensionField>(
        &self,
        mut input: Vec<Tensor<Element>>,
        store: &mut GenStore,
    ) -> anyhow::Result<Trace<E, Element>> {
        match self {
            InferenceEngine::Generic(model) => model.run(input, store),
            InferenceEngine::LLM(driver) => {
                ensure!(
                    input.len() == 1,
                    "LLM inference only supports one sequnce of tokens - batch inference is not supported"
                );
                let input = input.pop().expect("size validated above");
                driver.run_elements(input, store)
            }
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
#[serde(bound(serialize = "E: Serialize", deserialize = "E: DeserializeOwned"))]
pub struct SerializableGraphCtx<E: ExtensionField, PCS: PolynomialCommitmentScheme<E>>
where
    PCS::CommitmentWithWitness: Serialize + DeserializeOwned,
{
    pub(crate) ctx: ProverContext<E, PCS>,
    pub(crate) engine: InferenceEngine,
}

impl<E: ExtensionField, PCS: PolynomialCommitmentScheme<E>> SerializableGraphCtx<E, PCS>
where
    PCS::CommitmentWithWitness: Serialize + DeserializeOwned,
{
    pub fn new(ctx: ProverContext<E, PCS>, engine: InferenceEngine) -> Self {
        Self { ctx, engine }
    }

    /// Build the full execution graph context from `SerializableGraphCtx`,
    /// attaching the given `GenStore`.
    pub fn to_full_ctx(self, store: GenStore) -> ExecGraphCtx<E, PCS> {
        ExecGraphCtx {
            serializable_ctx: self,
            store,
        }
    }
}

impl<E: ExtensionField, PCS: PolynomialCommitmentScheme<E>> Deref for ExecGraphCtx<E, PCS>
where
    PCS::CommitmentWithWitness: Serialize + DeserializeOwned,
{
    type Target = SerializableGraphCtx<E, PCS>;

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
#[serde(bound(serialize = "E: Serialize", deserialize = "E: DeserializeOwned"))]
#[allow(clippy::large_enum_variant)]
pub enum ExecGraphIO<E: ExtensionField, PCS: PolynomialCommitmentScheme<E>> {
    // Input for inference task
    InferenceInput(InferenceIO),
    // Input for prover graph nodes
    Prover(ProverGraphIO<E, PCS>),
    // Model IO output to be provided to the verifier
    ModelIO(IO<E>),
}

impl<E: ExtensionField, PCS: PolynomialCommitmentScheme<E>> ExecGraphIO<E, PCS> {
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
#[serde(bound(serialize = "E: Serialize", deserialize = "E: DeserializeOwned"))]
pub enum ExecGraphNode<'a, 'b, E: ExtensionField, T, PCS: PolynomialCommitmentScheme<E>>
where
    PCS::CommitmentWithWitness: Serialize + DeserializeOwned + Send + Sync,
    PCS::ProverParam: Send + Sync,
{
    /// Task for inference of the model
    Inference,
    Prover(ProverGraphNode<'a, 'b, E, T, PCS>),
}

impl<'a, 'b, E: ExtensionField, T, PCS: PolynomialCommitmentScheme<E>> Debug
    for ExecGraphNode<'a, 'b, E, T, PCS>
where
    PCS::CommitmentWithWitness: Serialize + DeserializeOwned + Send + Sync,
    PCS::ProverParam: Send + Sync,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ExecGraphNode::Inference => write!(f, "ExecGraphNode::Inference"),
            ExecGraphNode::Prover(node) => write!(f, "ExecGraphNode::Prover({:?})", node),
        }
    }
}

impl<'a, 'b, E, T, PCS> ExecNode for ExecGraphNode<'a, 'b, E, T, PCS>
where
    E: ExtensionField,
    PCS: PolynomialCommitmentScheme<E> + Send + Sync + 'static,
    T: Transcript<E> + InitTranscript,
    PCS::CommitmentWithWitness: Serialize + DeserializeOwned + Send + Sync,
    PCS::ProverParam: Send + Sync,
{
    type IO = ExecGraphIO<E, PCS>;

    type Context = ExecGraphCtx<E, PCS>;

    fn describe(&self) -> String {
        match self {
            ExecGraphNode::Inference => "Inference".into(),
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
            ExecGraphNode::Inference => {
                ensure!(
                    inputs.len() == 1,
                    "Expected 1 input for inference node in distributed graph, found {}",
                    inputs.len()
                );
                let ExecGraphIO::InferenceInput(mut input) = inputs.pop().unwrap() else {
                    bail!("Expected tensors as input for inference task")
                };
                let trace = ctx.engine.run(input.input_tensors, &mut input.store)?;
                let io = trace.to_verifier_io()?;
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

pub type ExecGraph<'a, 'b, E, T, PCS> =
    crate::graph::scheduler::ExecGraph<ExecGraphNode<'a, 'b, E, T, PCS>, usize>;

pub fn build_execution_graph<E, T, PCS, S>(
    ctx: &ExecGraphCtx<E, PCS>,
    num_chunks: Option<usize>,
    chunking_strategy: S,
) -> anyhow::Result<ExecGraph<'_, '_, E, T, PCS>>
where
    E: ExtensionField,
    PCS: PolynomialCommitmentScheme<E> + Send + Sync + 'static,
    PCS::CommitmentWithWitness: Serialize + DeserializeOwned + Send + Sync,
    PCS::ProverParam: Send + Sync,
    S: ChunkingStrategy,
    T: Transcript<E> + InitTranscript,
{
    let chunks = ctx.ctx.split_in_chunks(num_chunks, chunking_strategy)?;
    // The full execution graph is obtained by pre-pending the node related to inference task
    // to the prover execution graph. The inference node produces the full trace, which is the
    // input needed by the prover graph
    let mut prover_graph: ExecGraph<E, T, PCS> =
        Prover::<E, T, PCS>::build_execution_graph(chunks)?
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
        prover_graph.add_inner(ExecGraphNode::Inference.colored(*source_node_color))?;
    // link inference node id with source node of prover graph
    prover_graph.add_consecutive_edge(inference_node_id, source_node, None)?;
    Ok(prover_graph)
}

pub fn graph_inputs<E, T, PCS>(
    input_tensors: Vec<Tensor<Element>>,
    store: GenStore,
    exec_graph: &ExecGraph<E, T, PCS>,
) -> anyhow::Result<HashMap<NodeInput, ExecGraphIO<E, PCS>>>
where
    E: ExtensionField,
    T: Transcript<E> + InitTranscript,
    PCS: PolynomialCommitmentScheme<E> + Send + Sync + 'static,
    PCS::CommitmentWithWitness: Serialize + DeserializeOwned + Send + Sync,
    PCS::ProverParam: Send + Sync,
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

pub type ProofWithIO<E, PCS> = (Proof<E, PCS>, IO<E>);

/// Utility method to extract the `ProofWithIO` from the outputs produced by the execution graph
pub fn extract_graph_outputs<E: ExtensionField, PCS: PolynomialCommitmentScheme<E>>(
    outputs: impl IntoIterator<Item = ExecGraphIO<E, PCS>>,
) -> anyhow::Result<ProofWithIO<E, PCS>>
where
    PCS::CommitmentWithWitness: Serialize + DeserializeOwned + Send + Sync,
    PCS::ProverParam: Send + Sync,
{
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
    use crossbeam_channel::unbounded;
    use ff_ext::GoldilocksExt2;
    use rayon::scope;
    use serde::{Serialize, de::DeserializeOwned};
    use std::collections::HashMap;
    use tenstore::GenStore;
    use transcript::BasicTranscript;

    use crate::{
        graph::{
            NodeId, NodeInput, NodeOutput,
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

    type F = GoldilocksExt2;
    type T = BasicTranscript<F>;

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

    /// A ThreadPoolExecutor which serializes the nodes of the execution graph and the IO of each node
    pub(crate) struct SerializedThreadPoolExecutor<S>(S);

    impl<N, C, S> Executor<N, C> for SerializedThreadPoolExecutor<S>
    where
        N::IO: Clone + Send + Sync + Serialize + DeserializeOwned,
        C: Clone + PartialEq + Send + Sync + Serialize + DeserializeOwned,
        N::Context: Sync, // Context is shared (not owned) across threads
        N: ExecNode + Clone + Send + Sync + Serialize + DeserializeOwned,
        S: TryToBytes<ReadyNode<N, C>>
            + TryFromBytes<ReadyNode<N, C>>
            + TryToBytes<Vec<N::IO>>
            + TryFromBytes<Vec<N::IO>>
            + Send
            + Sync,
    {
        type Config = S;

        fn run(
            config: &Self::Config,
            scheduler: GraphScheduler<N, C>,
            input_data: HashMap<NodeInput, N::IO>,
            context: &N::Context,
        ) -> anyhow::Result<HashMap<NodeOutput, N::IO>> {
            // we want to release all ready nodes all the time so that the threadpool is always busy
            let mut scheduler = scheduler.with_release_policy(ReleasePolicy::All);
            // final vector to collect outputs on the main thread
            let mut outputs = HashMap::new();
            //  channel to send results from task thread to the scoped logic
            let (result_sender, result_receiver) = unbounded();
            // channel to send results from scoped logic to main thread
            // we need to indirections because we are spawning tasks dynamically
            // depending on the output of previous tasks so everything must happen in the scope
            // and we also need to collect the final outputs outside the scope to return them
            // NOTE: all the channels are used in only one direction and are sequential (a -> b -> c) so there is no risk of deadlock
            let (outputs_sender, outputs_receiver) = unbounded();
            let mut ready_nodes = scheduler.init_nodes(input_data)?;
            scope(move |s| -> anyhow::Result<()> {
                while !scheduler.is_done() {
                    // execute all ready tasks
                    for node in ready_nodes.drain(..) {
                        let result_sender_local = result_sender.clone();
                        let node_id = node.node_id;
                        let serialized_node = config.try_to_bytes(&node)?;
                        // we put the task in the rayon threadpool and it'll be executed as soon as possible
                        s.spawn(move |_| {
                            let mut node: ReadyNode<N, C> =
                                match config.try_from_bytes(&serialized_node) {
                                    Ok(node) => node,
                                    Err(e) => {
                                        result_sender_local.send((node_id, Err(e))).unwrap();
                                        return;
                                    }
                                };
                            match node.run(context) {
                                Ok(output) => {
                                    let serialized_output = config.try_to_bytes(&output);
                                    result_sender_local
                                        .send((node_id, serialized_output))
                                        .unwrap()
                                }
                                // transmit error back to the main thread
                                Err(e) => result_sender_local.send((node_id, Err(e))).unwrap(),
                            };
                        });
                    }
                    // wait for a result - there is always one result
                    // since we know the graph is not done yet and each time
                    // we have an output we check if the graph is done
                    let (node_idx, output): (NodeId, Result<Vec<u8>, anyhow::Error>) =
                        result_receiver.recv().unwrap();
                    match output {
                        Ok(serialized_output) => {
                            let output: Vec<N::IO> = config.try_from_bytes(&serialized_output)?;
                            let graph_outputs = scheduler.mark_done(node_idx, &output).unwrap();
                            if !graph_outputs.is_empty() {
                                outputs_sender.clone().send(Ok(graph_outputs)).unwrap()
                            }
                            ready_nodes = scheduler.next_ready_nodes()?;
                        }
                        Err(e) => {
                            // transmit the error back to the main thread
                            let err = anyhow::anyhow!("Error running node {node_idx:?}: {e}");
                            outputs_sender.send(Err(err)).unwrap();
                            return Ok(());
                        }
                    }
                }
                Ok(())
            })?;
            for output in outputs_receiver.iter() {
                outputs.extend(output?)
            }
            Ok(outputs)
        }
    }

    #[test]
    fn test_exec_graph_data_serialization() -> anyhow::Result<()> {
        let (model, input) = Model::random(6)?;
        let (prover_ctx, verifier_ctx) = model
            .generate_contexts::<F, Pcs<F>>()
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
            DefaultChunkingStrategy::default(),
        )?;
        let inputs = graph_inputs(input, store.clone(), &graph)?;

        // serialize and deserialize prover context
        let serialized_ctx = BincodeSerializer.try_to_bytes(ctx.as_ref())?;
        let deserialized_ctx: SerializableGraphCtx<F, Pcs<F>> =
            BincodeSerializer.try_from_bytes(&serialized_ctx)?;

        let ctx = deserialized_ctx.to_full_ctx(store.clone());

        // run the execution graph
        let scheduler = GraphScheduler::new(graph);
        let outputs =
            SerializedThreadPoolExecutor::run(&BincodeSerializer, scheduler, inputs, &ctx)?;
        let (proof, io) = extract_graph_outputs(outputs.into_values())?;

        verify::<_, T, _>(&verifier_ctx, proof, io)
    }
}
