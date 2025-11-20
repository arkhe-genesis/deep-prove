#![allow(clippy::print_stdout)]
use anyhow::{anyhow, ensure};
use crossbeam_channel::{Receiver, Sender, unbounded};
use ff_ext::GoldilocksExt2;
use mpcs::{Basefold, BasefoldRSParams};
use std::collections::HashMap;
use tenstore::GenStore;
use transcript::BasicTranscript;
use zkml::{
    Element,
    graph::{
        executor::ThreadPoolExecutor,
        partition::{Partition, PartitionScheduler},
        scheduler::ExecGraph,
    },
    inputs::Input,
    iop::{
        chunking::DefaultChunkingStrategy,
        distributed_graph::{
            ExecGraphNode, SerializableGraphCtx, build_execution_graph, extract_graph_outputs,
            graph_inputs,
        },
    },
    model::Model,
    parser::onnx::FloatOnnxLoader,
    quantization::InferenceObserver,
    verify,
};

type F = GoldilocksExt2;
// the hasher type is chosen depending on the feature flag inside the mpcs crate
type Pcs = Basefold<F, BasefoldRSParams>;

type T = BasicTranscript<F>;

// Type of nodes of the graph to execute
type Node<'a, 'b> = ExecGraphNode<'a, 'b, F, T, Pcs>;

// Type of execution graph to be partitioned and executed in the workers
type Graph<'a, 'b> = ExecGraph<Node<'a, 'b>, Color>;

// Color is used to create the partitions, assign different nodes to different workers.
// It can be usize or any other type such as IP address etc.
type Color = usize;

/// What a partition scheduler outputs
type PartitionOutput<'a, 'b> = zkml::graph::partition::PartitionOutput<Node<'a, 'b>, Color>;

type SerializableCtx = SerializableGraphCtx<F, Pcs>;

// Implement a serializer for the chunk data structures.
// We need to implement `ToBytes` and `FromBytes` traits
// for types that implement `Serialize/DeserializeOwned`,
// employing the desired serialization library.
// In this example, we use bincode.
// struct BincodeSerializer;

fn build_model<T: std::io::Read>(
    model_data: &[u8],
    inputs: T,
) -> anyhow::Result<(Model<Element>, Vec<Vec<Element>>)> {
    let run_inputs = Input::from_reader(inputs).expect("failed to load inputs");
    let (model, md) =
        FloatOnnxLoader::from_bytes_with_scaling_strategy(model_data, InferenceObserver::new())
            .with_keep_float(true)
            .build()?;
    Ok((model, run_inputs.to_elements(&md)))
}

#[allow(clippy::type_complexity)]
pub fn run_node(
    store: GenStore,
    serialized_ctx: Vec<u8>,
    serialized_partitions: Vec<u8>,
    channel_register: HashMap<usize, (Sender<Vec<u8>>, Receiver<Vec<u8>>)>,
    output_channel: Sender<(usize, Vec<u8>)>,
) -> anyhow::Result<()> {
    // THIS should be loaded directly by the worker from local disk or remote storage, it's fixed per model
    let deserialized_ctx: SerializableCtx = rmp_serde::from_slice(&serialized_ctx)?;
    // the full context is not deserializable since it contains the store, so we need to build it
    // from the deserialized context by attaching the store
    let ctx = deserialized_ctx.to_full_ctx(store);
    // This is part of the "task", it's one set per inference request
    let partitions: Vec<Partition<Node, usize>> = rmp_serde::from_slice(&serialized_partitions)?;
    let mut scheduler = PartitionScheduler::<_, _, ThreadPoolExecutor>::new(partitions, ctx, ())?;
    let my_incoming_channel = channel_register.get(&scheduler.color).unwrap().1.clone();
    rayon::spawn(move || {
        // create the first set of messages to send from the coordinator schedular
        let mut outputs = scheduler.try_run_partition().unwrap();

        loop {
            // dispatch each output to the right worker
            for output in outputs.into_iter() {
                // here we need to lookup the destination
                if let Some(to_node) = output.to {
                    let to_node_tx = channel_register.get(&to_node).unwrap().0.clone();
                    let serialized_output = rmp_serde::to_vec(&output).unwrap();
                    to_node_tx.send(serialized_output).unwrap();
                    println!(
                        "Node {} sending output to node {}",
                        scheduler.color, to_node
                    );
                } else {
                    println!("Node {} has final output", scheduler.color);
                    // final output of the execution graph - only path used by the coordinator
                    let serialized_output = rmp_serde::to_vec(&output.output).unwrap();
                    output_channel
                        .send((scheduler.color, serialized_output))
                        .unwrap();
                }
            }
            if scheduler.is_done() {
                break;
            }
            let new_output_from_other_node: PartitionOutput =
                rmp_serde::from_slice(&my_incoming_channel.recv().unwrap()).unwrap();
            println!(
                "Node {} received output from worker {}",
                scheduler.color, new_output_from_other_node.from
            );
            // tell that we've received the output
            scheduler
                .set_child_partition_output(new_output_from_other_node)
                .unwrap();
            // it's asking to see if there are some logic to run - if there are none, outputs will be empty and loop will try again
            // if there are some, then the loop will dispatch them to the right worker
            outputs = scheduler.try_run_partition().unwrap();
        }
    });
    Ok(())
}

fn main() -> anyhow::Result<()> {
    let num_workers = 6;
    use tracing_subscriber::EnvFilter;

    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("debug"));
    tracing_subscriber::fmt().with_env_filter(filter).init();
    // ------------------------------
    //          COORDINATOR
    // ------------------------------
    let (model, mut inputs) = build_model(
        include_bytes!("../assets/scripts/CNN/cnn-cifar-01.onnx"),
        zstd::Decoder::new(&include_bytes!("../assets/scripts/CNN/input.json.zst")[..])
            .expect("failed to parse zstd"),
    )?;
    println!("model: # nodes: {}", model.graph.inner_nodes_count());

    let input = inputs.pop().ok_or(anyhow!("Expected at least one input"))?;

    let (prover_ctx, verifier_ctx) = model
        .generate_contexts::<F, Pcs>()
        .expect("unable to generate context");

    let input_tensor = model.load_input_flat(vec![input])?;

    let store = GenStore::default();

    // build the context for the executor
    // THIS should be loaded from the disk / local file / with some custom creation logic
    //  e.g. to instantiate the trace over the network for example
    let deserialized_ctx = SerializableGraphCtx::new(prover_ctx, model);
    // the full context is not deserializable since it contains the store, so we need to build it
    // from the deserialized context by attaching the store
    let ctx = deserialized_ctx.to_full_ctx(store.clone());
    let graph: Graph =
        build_execution_graph(&ctx, Some(num_workers), DefaultChunkingStrategy::default())?;

    let inputs = graph_inputs(input_tensor, store.clone(), &graph)?;

    ensure!(
        inputs.len() == 1,
        "Expected exactly one input node (inference)"
    );

    let serialized_ctx = rmp_serde::to_vec(ctx.as_ref())?;

    let flat_inputs =
        inputs
            .into_iter()
            .fold(Vec::new(), |mut ios, (node_input, chunk_prover_io)| {
                ios.push((node_input.node_id, chunk_prover_io));
                ios
            });
    println!("graph: {:#?}", graph);
    let partitions = graph.partition_by_color(flat_inputs)?;

    // Creates channels pairs to communicate with all other nodes
    let channel_register = partitions
        .iter()
        .fold(HashMap::new(), |mut map, (color, _)| {
            let (send, recv) = unbounded();
            map.insert(*color, (send, recv));
            map
        });
    // jst a signal to say we're done
    let (done_send, done_recv) = unbounded();
    println!("# partitions: {}", partitions.len());
    let mut serialized_partitions = partitions
        .into_iter()
        .map(|(color, partitions)| {
            Ok((
                color,
                rmp_serde::to_vec(&partitions)?,
                // PartitionScheduler::<_, _, ThreadPoolExecutor>::new(partitions, new_ctx()?, ())?,
            ))
        })
        .collect::<anyhow::Result<HashMap<_, _>>>()?;

    // the coordinator is always the partition with color 0
    let coordinator_partitions = serialized_partitions.remove(&0).unwrap();
    run_node(
        store.clone(),
        serialized_ctx.clone(),
        coordinator_partitions,
        channel_register.clone(),
        done_send.clone(),
    )?;
    // launch the rest of the workers
    for (_, worker_partitions) in serialized_partitions.into_iter() {
        run_node(
            store.clone(),
            serialized_ctx.clone(),
            worker_partitions,
            channel_register.clone(),
            done_send.clone(),
        )?;
    }

    let outputs = (0..2)
        .map(|_| {
            let (color, serialized_output) = done_recv.recv()?;
            ensure!(
                color == 0,
                "Coordinator should be the only one to send the proof"
            );
            println!("Received output from node {}", color);
            let out = rmp_serde::from_slice(&serialized_output)?;
            Ok(out)
        })
        .collect::<anyhow::Result<Vec<_>>>()?;
    ensure!(
        outputs.len() == 2,
        "Expected 2 outputs for the graph, {} outputs received",
        outputs.len()
    );
    let (proof, io) = extract_graph_outputs(outputs)?;
    verify::<_, T, _>(&verifier_ctx, proof, io).unwrap();
    println!("Done");
    Ok(())
}
