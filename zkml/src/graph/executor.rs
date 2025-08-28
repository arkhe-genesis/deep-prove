use super::{
    GraphNode, NodeIdx,
    scheduler::{GraphScheduler, ReleasePolicy},
};
use crossbeam_channel::unbounded;
use rayon::scope;
use std::collections::HashSet;

/// A trait that defines the interface for an executor.
/// It is responsible for running the graph scheduler to completion and returning the outputs.
/// N corresponds to the generic node of the graph.
/// C corresponds to the generic coloring of the graph.
pub trait Executor<N: GraphNode, C> {
    type Config;
    fn run(
        config: &Self::Config,
        scheduler: GraphScheduler<N, C>,
        input_data: Vec<N::IO>,
        // The context is the local context for the executor of a (sub)graph.
        context: &N::Context,
    ) -> anyhow::Result<Vec<N::IO>>;
}

/// The sequential executor is just executing tasks sequentially, leading to the same CPU
/// usage as the "regular" proving logic.
pub struct SequentialExecutor;

impl<N, C> Executor<N, C> for SequentialExecutor
where
    N::IO: Clone,
    C: Clone + PartialEq,
    N: GraphNode + Clone,
{
    type Config = ();

    /// input_data is a vector of vectors of input data for each input node as described in the graph input nodes
    /// TODO: currently very simple runner - we can speed up by running them inside a threadpool so CPU usage is always at 100%.
    /// Currently it waits for all the tasks in the batch to finish before proceeding to the next batch.
    fn run(
        _config: &Self::Config,
        mut scheduler: GraphScheduler<N, C>,
        input_data: Vec<N::IO>,
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
                    scheduler.mark_done(node.node_idx, output).unwrap();
                });
            ready_nodes = scheduler.next_ready_nodes();
        }
        Ok(outputs)
    }
}

/// An executor that always put tasks ready to execute in the main rayon threadpool such
/// that at all times, all cores should always be busy as long as there are tasks available to execute.
pub struct ThreadPoolExecutor;

impl<N, C> Executor<N, C> for ThreadPoolExecutor
where
    N::IO: Clone + Send + Sync,
    C: Clone + PartialEq + Send + Sync,
    // we only need Sync for the context as we don't want to give ownership to any specific thread
    // we want to share it across all the tasks.
    N::Context: Sync,
    N: GraphNode + Clone + Send + Sync,
{
    // TODO: Maybe change it to designate a threadpool size or a specific threadpool...
    type Config = ();

    fn run(
        _config: &Self::Config,
        scheduler: GraphScheduler<N, C>,
        input_data: Vec<N::IO>,
        context: &N::Context,
    ) -> anyhow::Result<Vec<N::IO>> {
        // we want to release all nodes ready all the time so that the threadpool is always busy
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
        // NOTE: all the channels are used in only one direction so there is no risk of deadlock
        let (outputs_sender, outputs_receiver) = unbounded();
        let mut ready_nodes = scheduler.init_nodes(input_data)?;
        scope(move |s| {
            while !scheduler.is_done() {
                // execute all ready tasks
                for mut node in ready_nodes.drain(..) {
                    let node_idx = node.node_idx;
                    let result_sender_local = result_sender.clone();
                    // we put the task in the rayon threadpool and it'll be executed as soon as possible
                    s.spawn(move |_| {
                        match node.run(context) {
                            Ok(output) => result_sender_local.send((node_idx, Ok(output))).unwrap(),
                            // transmit error back to the main thread
                            Err(e) => result_sender_local.send((node_idx, Err(e))).unwrap(),
                        };
                    });
                }
                // wait for a result - there is always one result
                // since we know the graph is not done yet and each time
                // we have an output we check if the graph is done
                let (node_idx, output): (NodeIdx, Result<N::IO, anyhow::Error>) =
                    result_receiver.recv().unwrap();
                match output {
                    Ok(output) => {
                        if output_nodes.contains(&node_idx) {
                            // signal the output to the main thread
                            outputs_sender.clone().send(Ok(output.clone())).unwrap();
                        }
                        scheduler.mark_done(node_idx, output).unwrap();
                        ready_nodes = scheduler.next_ready_nodes();
                    }
                    Err(e) => {
                        // transmit the error back to the main thread
                        let err = anyhow::anyhow!("Error running node {:?}: {}", node_idx, e);
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

    use crate::graph::Edge;

    use super::super::{Colored, Graph};
    use crate::graph::{
        GraphNode,
        executor::{Executor, SequentialExecutor, ThreadPoolExecutor},
        scheduler::GraphScheduler,
    };

    #[derive(Debug, Clone)]
    pub enum MathAST {
        Add,
        Mul,
        Div,
        Sub,
        Pow2,
    }

    impl GraphNode for MathAST {
        type IO = i32;
        type Context = ();
        fn describe(&self) -> String {
            match self {
                MathAST::Add => "Add".to_string(),
                MathAST::Mul => "Mul".to_string(),
                MathAST::Div => "Div".to_string(),
                MathAST::Sub => "Sub".to_string(),
                MathAST::Pow2 => "Pow2".to_string(),
            }
        }
        fn run(&self, _ctx: &Self::Context, inputs: Vec<Self::IO>) -> anyhow::Result<Self::IO> {
            match self {
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
        let mut graph = Graph::new();
        let add_node = graph.add_node(
            Colored {
                node: MathAST::Add,
                color: 0,
            },
            vec![Edge::Input(0), Edge::Input(1)],
        );
        let mul_node = graph.add_node(
            Colored {
                node: MathAST::Mul,
                color: 0,
            },
            vec![Edge::Pred(add_node, None), Edge::Input(2)],
        );
        let _add_node_2 = graph.add_node(
            Colored {
                node: MathAST::Add,
                color: 0,
            },
            vec![Edge::Pred(add_node, None), Edge::Pred(mul_node, None)],
        );
        let colored_graph = graph;
        let scheduler = GraphScheduler::new(colored_graph);
        let output = SequentialExecutor::run(&(), scheduler.clone(), vec![1, 2, 3], &()).unwrap();
        // (1+2) + ((1 + 2) * 3)  = 12
        let expected_output = vec![12];
        assert_eq!(output, expected_output);

        let thread_output =
            ThreadPoolExecutor::run(&(), scheduler.clone(), vec![1, 2, 3], &()).unwrap();
        assert_eq!(thread_output, output);
    }
}
