use super::{ColoredGraph, GraphScheduler, ReleasePolicy, node::GraphNode};
use crate::graph::NodeIdx;
use crossbeam_channel::unbounded;
use rayon::scope;
use std::collections::HashSet;

/// A master executor that runs the graph on the local machine using rayon
/// It is responsible for creating tasks from operations and input data
/// and then distributing them to the threadpool. In this local implementation,
/// there is no need for a separate slave executor logic since the threadpool is local.
pub struct SequentialExecutor<N: GraphNode, C> {
    scheduler: GraphScheduler<N, C>,
    /// In this case, the local executor also holds the context. In distributed implementation,
    /// the workers would hold the context and receive the node and input from the network.
    ctx: N::Context,
}

impl<N, C> SequentialExecutor<N, C>
where
    N::IO: Clone,
    C: Clone + PartialEq,
    N: GraphNode,
{
    pub fn new(graph: ColoredGraph<N, C>, ctx: N::Context) -> Self {
        Self {
            scheduler: GraphScheduler::new(graph),
            ctx,
        }
    }

    /// input_data is a vector of vectors of input data for each input node as described in the graph input nodes
    /// TODO: currently very simple runner - we can speed up by running them inside a threadpool so CPU usage is always at 100%.
    /// Currently it waits for all the tasks in the batch to finish before proceeding to the next batch.
    pub fn run(&mut self, input_data: Vec<N::IO>) -> anyhow::Result<Vec<N::IO>> {
        let mut ready_nodes = self.scheduler.init_nodes(input_data)?;
        let mut outputs = Vec::new();
        while !self.scheduler.is_done() {
            outputs = ready_nodes
                .iter_mut()
                .map(|node| node.run(&self.ctx))
                .collect::<anyhow::Result<Vec<_>>>()?;
            ready_nodes
                .drain(..)
                .zip(outputs.clone())
                .for_each(|(node, output)| {
                    self.scheduler.mark_done(node.node_idx, output).unwrap();
                });
            ready_nodes = self.scheduler.next_ready_nodes();
        }
        Ok(outputs)
    }
}

/// An executor that always put tasks ready to execute in the main rayon threadpool such
/// that at all times, all cores should always be busy as long as there are tasks available to execute.
/// The sequential executor is just executing tasks sequentially, leading to the same CPU
/// usage as the "regular" proving logic.
pub struct ThreadPoolExecutor<N: GraphNode, C> {
    scheduler: GraphScheduler<N, C>,
    ctx: N::Context,
}

impl<N, C> ThreadPoolExecutor<N, C>
where
    N::IO: Clone + Send + Sync,
    C: Clone + PartialEq + Send + Sync,
    // we only need Sync for the context as we don't want to give ownership to any specific thread
    // we want to share it across all the tasks.
    N::Context: Sync,
    N: GraphNode + Send + Sync,
{
    pub fn new(graph: ColoredGraph<N, C>, ctx: N::Context) -> Self {
        Self {
            // we want to release all nodes all the time so that the threadpool is always busy
            scheduler: GraphScheduler::new(graph).with_release_policy(ReleasePolicy::All),
            ctx,
        }
    }

    pub fn run(mut self, input_data: Vec<N::IO>) -> anyhow::Result<Vec<N::IO>> {
        let output_nodes: HashSet<_> = self.scheduler.output_nodes().into_iter().collect();
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
        let mut ready_nodes = self.scheduler.init_nodes(input_data)?;
        let ctx = &self.ctx;
        scope(move |s| {
            while !self.scheduler.is_done() {
                // execute all ready tasks
                for mut node in ready_nodes.drain(..) {
                    let node_idx = node.node_idx;
                    let result_sender_local = result_sender.clone();
                    // we put the task in the rayon threadpool and it'll be executed as soon as possible
                    s.spawn(move |_| {
                        match node.run(ctx) {
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
                        self.scheduler.mark_done(node_idx, output).unwrap();
                        ready_nodes = self.scheduler.next_ready_nodes();
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
            match output {
                Ok(output) => outputs.push(output),
                Err(e) => return Err(e),
            }
        }
        Ok(outputs)
    }
}

#[cfg(test)]
mod tests {
    use anyhow::ensure;

    use crate::graph::Edge;

    use super::{super::ColoredNode, *};

    #[derive(Debug, Clone)]
    enum MathAST {
        Add,
        Mul,
    }

    impl GraphNode for MathAST {
        type IO = usize;
        type Context = ();
        fn describe(&self) -> String {
            match self {
                MathAST::Add => "Add".to_string(),
                MathAST::Mul => "Mul".to_string(),
            }
        }
        fn run(&self, _ctx: &Self::Context, inputs: Vec<Self::IO>) -> anyhow::Result<Self::IO> {
            ensure!(inputs.len() == 2, "Expected 2 inputs");
            match self {
                MathAST::Add => Ok(inputs[0] + inputs[1]),
                MathAST::Mul => Ok(inputs[0] * inputs[1]),
            }
        }
    }

    #[test]
    fn test_graph_executor() {
        let mut graph = ColoredGraph::new();
        let add_node = graph
            .add_node(
                ColoredNode {
                    proving_node: MathAST::Add,
                    color: 0,
                },
                vec![Edge::Input(0), Edge::Input(1)],
            )
            .unwrap();
        let mul_node = graph
            .add_node(
                ColoredNode {
                    proving_node: MathAST::Mul,
                    color: 0,
                },
                vec![Edge::Pred(add_node), Edge::Input(2)],
            )
            .unwrap();
        let _add_node_2 = graph.add_node(
            ColoredNode {
                proving_node: MathAST::Add,
                color: 0,
            },
            vec![Edge::Pred(add_node), Edge::Pred(mul_node)],
        );
        let colored_graph = graph;
        let mut executor = SequentialExecutor::new(colored_graph.clone(), ());
        let output = executor.run(vec![1, 2, 3]).unwrap();
        // (1+2) + ((1 + 2) * 3)  = 12
        let expected_output = vec![12];
        assert_eq!(output, expected_output);

        let executor = ThreadPoolExecutor::new(colored_graph, ());
        let thread_output = executor.run(vec![1, 2, 3]).unwrap();
        assert_eq!(thread_output, output);
    }
}
