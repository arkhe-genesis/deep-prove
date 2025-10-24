use super::{
    NodeId,
    scheduler::{ExecNode, GraphScheduler, ReleasePolicy},
};
use crate::graph::NodeInput;
use crossbeam_channel::unbounded;
use rayon::scope;
use std::collections::{HashMap, HashSet};

/// A trait defining execution strategies for computational graphs.
///
/// Executors are responsible for taking a scheduled graph and running it to completion,
/// managing the execution of individual nodes and collecting the final outputs.
/// Different executor implementations can provide various execution strategies:
///
/// - **Sequential execution**: Nodes run one at a time in dependency order
/// - **Parallel execution**: Multiple nodes run concurrently when possible
/// - **Distributed execution**: Nodes run across multiple machines
/// - **GPU execution**: Nodes run on specialized hardware
///
/// # Type Parameters
///
/// * `N` - The executable node type implementing [`ExecNode`]
/// * `C` - The color type used for scheduling and partitioning
/// * `NodeID` - The node identifier type (defaults to [`DefaultNodeID`])
pub trait Executor<N: ExecNode, C> {
    /// Configuration type for this executor.
    ///
    /// Different executors may require different configuration parameters
    /// (e.g., thread pool size, GPU device selection, network endpoints).
    type Config;

    /// Executes the given graph to completion and returns the final outputs.
    ///
    /// This method takes ownership of the scheduler and runs the graph until
    /// all nodes have been executed, collecting outputs from nodes that
    /// produce graph outputs.
    ///
    /// # Parameters
    ///
    /// * `config` - Executor-specific configuration
    /// * `scheduler` - The graph scheduler managing execution order
    /// * `input_data` - External input data for the graph
    /// * `context` - Execution context shared across all nodes
    ///
    /// # Returns
    ///
    /// A vector containing the outputs from all graph output nodes.
    ///
    /// # Errors
    ///
    /// Returns an error if any node execution fails or if there are
    /// scheduling/coordination issues.
    fn run(
        config: &Self::Config,
        scheduler: GraphScheduler<N, C>,
        input_data: HashMap<NodeInput, N::IO>,
        context: &N::Context,
    ) -> anyhow::Result<Vec<N::IO>>;
}

/// An executor that runs nodes sequentially in dependency order.
///
/// This executor provides the simplest execution strategy, running nodes one at a time
/// in the order determined by the scheduler. It offers:
///
/// - **Predictable behavior**: Deterministic execution order
/// - **Low resource usage**: No parallelism overhead
/// - **Easy debugging**: Simple execution flow
/// - **Compatibility**: Works with any node type and context
///
/// The sequential executor is ideal for:
/// - Development and testing
/// - Resource-constrained environments
/// - Scenarios where deterministic execution is required
/// - Debugging complex computational graphs
pub struct SequentialExecutor;

impl<N, C> Executor<N, C> for SequentialExecutor
where
    N::IO: Clone,
    C: Clone + PartialEq,
    N: ExecNode + Clone,
{
    /// No configuration needed for sequential execution.
    type Config = ();

    /// Executes the graph sequentially, running nodes in batches as they become ready.
    ///
    /// The execution proceeds in rounds:
    /// 1. Initialize the scheduler with input data
    /// 2. Execute all ready nodes in the current batch
    /// 3. Mark nodes as complete and get the next batch
    /// 4. Repeat until all nodes are executed
    ///
    /// Within each batch, nodes are executed sequentially even if they could
    /// run in parallel. This ensures predictable, deterministic execution.
    fn run(
        _config: &Self::Config,
        mut scheduler: GraphScheduler<N, C>,
        input_data: HashMap<NodeInput, N::IO>,
        context: &N::Context,
    ) -> anyhow::Result<Vec<N::IO>> {
        let mut ready_nodes = scheduler.init_nodes(input_data)?;
        let mut outputs = Vec::new();
        while !scheduler.is_done() {
            outputs = ready_nodes
                .iter_mut()
                .map(|node| node.run(context))
                .collect::<anyhow::Result<Vec<_>>>()?;
            ready_nodes
                .drain(..)
                .zip(outputs.clone())
                .for_each(|(node, output)| {
                    scheduler.mark_done(node.node_id, &output).unwrap();
                });
            ready_nodes = scheduler.next_ready_nodes();
        }
        Ok(outputs)
    }
}

/// An executor that runs nodes in parallel using a thread pool.
///
/// This executor maximizes CPU utilization by running ready nodes concurrently
/// in the Rayon thread pool. It provides:
///
/// - **High throughput**: Utilizes all available CPU cores
/// - **Dynamic scheduling**: Nodes execute as soon as their dependencies are satisfied
/// - **Load balancing**: Rayon automatically distributes work across threads
/// - **Scalability**: Performance scales with the number of available cores
///
/// The thread pool executor is ideal for:
/// - CPU-intensive computations
/// - Graphs with significant parallelism opportunities
/// - Production environments with multiple cores
/// - Scenarios where maximum performance is required
///
/// # Thread Safety Requirements
///
/// All types must be `Send + Sync` to enable safe parallel execution:
/// - Node data (`N::IO`) must be transferable between threads
/// - Execution context must be shareable across threads
/// - Node operations must be thread-safe
pub struct ThreadPoolExecutor;

impl<N, C> Executor<N, C> for ThreadPoolExecutor
where
    N::IO: Clone + Send + Sync,
    C: Clone + PartialEq + Send + Sync,
    N::Context: Sync, // Context is shared (not owned) across threads
    N: ExecNode + Clone + Send + Sync,
{
    /// No configuration needed - uses the global Rayon thread pool.
    ///
    /// Future versions might support custom thread pool configuration.
    type Config = ();

    /// Executes the graph in parallel using dynamic scheduling.
    ///
    /// This implementation uses the "All" release policy to maximize parallelism,
    /// allowing all ready nodes to execute concurrently. The execution flow:
    ///
    /// 1. Set release policy to allow maximum parallelism
    /// 2. Spawn ready nodes as Rayon tasks immediately
    /// 3. Use channels to coordinate completion and collect results
    /// 4. Continue until all nodes are executed
    ///
    /// The executor maintains two communication channels:
    /// - Task results: From worker threads back to the coordinator
    /// - Final outputs: From coordinator to the main thread
    fn run(
        _config: &Self::Config,
        scheduler: GraphScheduler<N, C>,
        input_data: HashMap<NodeInput, N::IO>,
        context: &N::Context,
    ) -> anyhow::Result<Vec<N::IO>> {
        // we want to release all ready nodes all the time so that the threadpool is always busy
        let mut scheduler = scheduler.with_release_policy(ReleasePolicy::All);
        let output_nodes: HashSet<_> = scheduler.output_nodes().into_iter().collect();
        // final vector to collect outputs on the main thread
        let mut outputs = Vec::with_capacity(output_nodes.len());
        //  channel to send results from task thread to the scoped logic
        let (result_sender, result_receiver) = unbounded();
        // channel to send results from scoped logic to main thread
        // we need to indirections because we are spawning tasks dynamically
        // depending on the output of previous tasks so everything must happen in the scope
        // and we also need to collect the final outputs outside the scope to return them
        // NOTE: all the channels are used in only one direction and are sequential (a -> b -> c) so there is no risk of deadlock
        let (outputs_sender, outputs_receiver) = unbounded();
        let mut ready_nodes = scheduler.init_nodes(input_data)?;
        scope(move |s| {
            while !scheduler.is_done() {
                // execute all ready tasks
                for mut node in ready_nodes.drain(..) {
                    let result_sender_local = result_sender.clone();
                    // we put the task in the rayon threadpool and it'll be executed as soon as possible
                    s.spawn(move |_| {
                        let node_id = node.node_id;
                        match node.run(context) {
                            Ok(output) => result_sender_local.send((node_id, Ok(output))).unwrap(),
                            // transmit error back to the main thread
                            Err(e) => result_sender_local.send((node_id, Err(e))).unwrap(),
                        };
                    });
                }
                // wait for a result - there is always one result
                // since we know the graph is not done yet and each time
                // we have an output we check if the graph is done
                let (node_idx, output): (NodeId, Result<N::IO, anyhow::Error>) =
                    result_receiver.recv().unwrap();
                match output {
                    Ok(output) => {
                        if output_nodes.contains(&node_idx) {
                            // signal the output to the main thread
                            outputs_sender.clone().send(Ok(output.clone())).unwrap();
                        }
                        scheduler.mark_done(node_idx, &output).unwrap();
                        ready_nodes = scheduler.next_ready_nodes();
                    }
                    Err(e) => {
                        // transmit the error back to the main thread
                        let err = anyhow::anyhow!("Error running node {node_idx:?}: {e}");
                        outputs_sender.send(Err(err)).unwrap();
                        return;
                    }
                }
            }
        });
        for output in outputs_receiver.iter() {
            outputs.push(output?);
        }
        Ok(outputs)
    }
}

#[cfg(test)]
pub mod tests {
    use crate::graph::{
        NodeInput, Ports,
        executor::{Executor, SequentialExecutor, ThreadPoolExecutor},
        scheduler::{ExecGraph, ExecNode, GraphScheduler, IntoColor},
    };
    use std::collections::HashMap;

    #[derive(Debug, Clone)]
    pub enum MathAST {
        Input(i32),
        Add,
        Mul,
        Div,
        Sub,
        Pow2,
    }

    impl ExecNode for MathAST {
        type IO = i32;
        type Context = ();
        fn describe(&self) -> String {
            match self {
                MathAST::Input(i) => format!("Input({i})"),
                MathAST::Add => "Add".to_string(),
                MathAST::Mul => "Mul".to_string(),
                MathAST::Div => "Div".to_string(),
                MathAST::Sub => "Sub".to_string(),
                MathAST::Pow2 => "Pow2".to_string(),
            }
        }
        fn run(&self, _ctx: &Self::Context, inputs: Vec<Self::IO>) -> anyhow::Result<Self::IO> {
            match self {
                MathAST::Input(_) => {
                    assert_eq!(inputs.len(), 1);
                    Ok(inputs[0])
                }
                MathAST::Add => Ok(inputs[0] + inputs[1]),
                MathAST::Mul => Ok(inputs[0] * inputs[1]),
                MathAST::Div => Ok(inputs[0] / inputs[1]),
                MathAST::Sub => Ok(inputs[0] - inputs[1]),
                MathAST::Pow2 => Ok(inputs[0] * inputs[0]),
            }
        }
    }

    #[test]
    fn test_graph_executor() {
        let mut graph = ExecGraph::default_exec_graph();
        let input_nodes = [
            graph.add_inner(MathAST::Input(1).colored(0)).unwrap(),
            graph.add_inner(MathAST::Input(2).colored(0)).unwrap(),
            graph.add_inner(MathAST::Input(3).colored(0)).unwrap(),
        ];

        // add1 = 1 + 2
        let add1 = graph.add_inner(MathAST::Add.colored(0)).unwrap();
        graph.add_edge(input_nodes[0], add1, (0, 0), None).unwrap();
        graph.add_edge(input_nodes[1], add1, (0, 1), None).unwrap();

        // mul = add1 * 3
        let mul = graph.add_inner(MathAST::Mul.colored(0)).unwrap();
        graph
            .add_edge(add1, mul, Ports::consecutive(), None)
            .unwrap();
        graph.add_edge(input_nodes[2], mul, (0, 1), None).unwrap();

        // add2 = add1 + mul = add1 + (add1 * 3) = (1 + 2) + ((1 + 2) * 3) = 12
        let add2 = graph.add_inner(MathAST::Add.colored(0)).unwrap();
        graph
            .add_edge(add1, add2, Ports::consecutive(), None)
            .unwrap();
        graph.add_edge(mul, add2, (0, 1), None).unwrap();

        let colored_graph = graph;
        let scheduler = GraphScheduler::new(colored_graph);
        let inputs: HashMap<NodeInput, i32> = [1, 2, 3]
            .into_iter()
            .enumerate()
            .map(|(i, x)| (NodeInput::new(input_nodes[i], 0), x))
            .collect();
        let output = SequentialExecutor::run(&(), scheduler.clone(), inputs.clone(), &()).unwrap();
        // (1+2) + ((1 + 2) * 3)  = 12
        let expected_output = vec![12];
        assert_eq!(output, expected_output);

        let thread_output = ThreadPoolExecutor::run(&(), scheduler.clone(), inputs, &()).unwrap();
        assert_eq!(thread_output, output);
    }
}
