use anyhow::ensure;

use super::{Colored, Direction, GraphNode, NodeIdx, RunnableGraph};
use petgraph::visit::EdgeRef;
use std::collections::{HashMap, HashSet};

/// Policy used to decide if a node should be released to run or not.
/// For example, we may want to release eagerly all nodes ready.
/// Or we may want to release only nodes of the same color.
/// Or maybe in between, only release up to certain number of nodes for each color.
/// This policy impacts how the executor will be able to perform.
#[derive(Debug, Clone, Default)]
pub enum ReleasePolicy {
    /// Release all nodes that are ready to run
    All,
    /// Release nodes that are ready to run and whose color is not already present in the pending nodes
    #[default]
    UniqueColoring,
}

impl ReleasePolicy {
    fn accept<N: GraphNode, C: PartialEq>(
        &self,
        node_index: NodeIdx,
        scheduler: &GraphScheduler<N, C>,
    ) -> bool {
        match self {
            ReleasePolicy::All => true,
            ReleasePolicy::UniqueColoring => {
                let node_color = &scheduler.graph.graph[node_index].color;
                scheduler
                    .running_nodes
                    .iter()
                    .all(|nidx| &scheduler.graph.graph[*nidx].color != node_color)
            }
        }
    }
}

/// A node that also contains the input to run.
/// It needs to be coupled with the context such that the node can finally be executed.
#[derive(Debug, Clone)]
pub struct ReadyNode<N: GraphNode, C> {
    /// Node that contains the operation and its color
    pub(crate) node: Colored<N, C>,
    /// Input data for the node's operation
    pub(crate) inputs: Vec<N::IO>,
    /// Index in the graph
    pub(crate) node_idx: NodeIdx,
}

impl<N: GraphNode, C> ReadyNode<N, C> {
    pub fn run(&mut self, ctx: &N::Context) -> anyhow::Result<N::IO> {
        self.node.run(ctx, self.inputs.drain(..).collect())
    }
}

/// A scheduler for a colored graph. It is responsible for scheduling the nodes to be executed,
/// marking the nodes as done and updating the ready nodes until all nodes have been executed.
#[derive(Debug, Clone)]
pub struct GraphScheduler<N: GraphNode, C> {
    /// The graph is composed of colored nodes and the edges are the IO of the execution
    graph: RunnableGraph<N, C>,
    /// For the nodes that have both edges that come from other nodes AND "input data" edges.
    waiting_input_data: HashMap<NodeIdx, Vec<N::IO>>,
    /// Nodes that are currently being executed
    running_nodes: HashSet<NodeIdx>,

    /// Nodes that are already executed
    done_nodes: HashSet<NodeIdx>,

    /// Policy to release nodes that are ready to run
    release_policy: ReleasePolicy,
}

impl<N, C> GraphScheduler<N, C>
where
    N: GraphNode + Clone,
    N::IO: Clone,
    C: PartialEq + Clone,
{
    pub fn new(graph: RunnableGraph<N, C>) -> Self {
        Self {
            graph,
            running_nodes: HashSet::new(),
            done_nodes: HashSet::new(),
            release_policy: ReleasePolicy::UniqueColoring,
            waiting_input_data: HashMap::new(),
        }
    }

    pub fn output_nodes(&self) -> Vec<NodeIdx> {
        self.graph.output_nodes()
    }

    pub fn with_release_policy(mut self, release_policy: ReleasePolicy) -> Self {
        self.release_policy = release_policy;
        self
    }

    /// spits out the list of input nodes along with the data required to run the node's operations
    /// input_data is a vector of vectors of input data for each input node as described in the graph input nodes
    pub fn init_nodes(&mut self, input_data: Vec<N::IO>) -> anyhow::Result<Vec<ReadyNode<N, C>>> {
        ensure!(self.running_nodes.is_empty(), "Running nodes must be empty");
        ensure!(
            self.graph
                .input_nodes
                .values()
                .flat_map(|indices| indices.iter())
                .collect::<HashSet<_>>()
                .len()
                == input_data.len(),
            "Number of pending nodes and input data must match"
        );
        // we filter out the input nodes that have incoming edges (they might have one input edge and another input edge coming from a node)
        // for the input nodes that also wait an incoming edge from other nodes, just register them in the waiting nodes under the constant input node index
        Ok(self
            .graph
            .input_nodes
            .iter()
            .filter_map(|(node_idx, input_indices)| {
                // for the initial nodes, we can only take the nodes that have no incoming edges
                if self
                    .graph
                    .graph
                    .neighbors_directed(*node_idx, Direction::Incoming)
                    .count()
                    == 0
                {
                    // if the input node has no incoming edges, it is an *full* input node
                    // in this case, we mark them directly as pending since we'll return them in this function
                    self.running_nodes.insert(*node_idx);
                    // and prepare the ready node
                    let node = self.graph.graph[*node_idx].clone();
                    Some(ReadyNode {
                        node,
                        // TODO: if we want to avoid cloning, then we should either have N::IO: Default and replace
                        // each entry in the vector by default, OR use an `enum InputData { Data(N::IO), Empty }` in the vector.
                        // for now, we keep it simple.
                        inputs: input_indices
                            .iter()
                            .map(|idx| input_data[*idx].clone())
                            .collect(),
                        node_idx: *node_idx,
                    })
                } else {
                    // in this case, we'll be waiting for the other nodes to be executed
                    // and then their outputs will be set on the corresponding edges such that this node
                    // will run after all its edges are filled with data.
                    // BUT we still need to save that input data somewhere so that when all dependencies are resolved,
                    // we can run this node
                    self.waiting_input_data.insert(
                        *node_idx,
                        input_indices
                            .iter()
                            .map(|idx| input_data[*idx].clone())
                            .collect(),
                    );
                    None
                }
            })
            .collect::<Vec<_>>())
    }

    /// Mark a node as done and give the output of its execution.
    pub fn mark_done(&mut self, node_idx: NodeIdx, output: N::IO) -> anyhow::Result<()> {
        ensure!(self.running_nodes.remove(&node_idx), "node was not running");
        ensure!(self.done_nodes.insert(node_idx), "node was already done");
        // now look at all the nodes that are ready to run and put them in pending
        let successors = self
            .graph
            .graph
            .neighbors_directed(node_idx, Direction::Outgoing)
            // do not take nodes that are already done or pending
            .filter(|idx| !self.done_nodes.contains(idx) && !self.running_nodes.contains(idx))
            .collect::<Vec<_>>();
        for idx in successors {
            // Set the data to the edge such that the successors may run afterwards
            // TODO: remove the clone by sharing by reference but it needs to be held somewhere
            self.graph
                .graph
                .update_edge(node_idx, idx, Some(output.clone()));
        }
        Ok(())
    }

    /// Returns a list of nodes which are ready to run, e.g. whose dependencies have all been resolved.
    pub fn next_ready_nodes(&mut self) -> Vec<ReadyNode<N, C>> {
        let mut ready = Vec::new();
        let ready_node_idx = self
            .graph
            .graph
            .node_indices()
            // only take nodes that have incoming edges - the ones that don't already have been delivered via init_nodes
            .filter(|nidx| {
                self.graph
                    .graph
                    .edges_directed(*nidx, Direction::Incoming)
                    .count()
                    > 0
            })
            // only take the nodes whose incoming edges are all filled with input data
            .filter(|nidx| {
                self.graph
                    .graph
                    .edges_directed(*nidx, Direction::Incoming)
                    .all(|e| e.weight().is_some())
            })
            // exclude nodes that are already pending - it shouldn't happen but just in case
            .filter(|node_idx| !self.running_nodes.contains(node_idx))
            .collect::<Vec<_>>();
        for node_idx in ready_node_idx {
            // need to check here if the node is ready to run - each time we add a new node to the pending nodes,
            // the policy might change decisions so we need to check again for future nodes
            if !self.release_policy.accept(node_idx, self) {
                continue;
            }
            self.running_nodes.insert(node_idx);
            // collect both the input data and the input from the edges
            let input_edges = self
                .graph
                .graph
                .edges_directed(node_idx, Direction::Incoming)
                .map(|e| e.id())
                .collect::<Vec<_>>();
            let input_data = input_edges
                .into_iter()
                .map(|edge_id| {
                    // This method edge_weight_mut() returns Option<&mut Option<N::IO>>:
                    // first unwrap because the graph API returns an option but in this case we are sure the edge exists
                    // second take + unwrap to take the input data and leave only none
                    self.graph
                        .graph
                        .edge_weight_mut(edge_id)
                        .unwrap()
                        .take()
                        .unwrap()
                    // also remove the potential input data for nodes that also expect inoput data + predecessors data
                })
                .chain(
                    self.waiting_input_data
                        .remove(&node_idx)
                        .unwrap_or_default()
                        .into_iter(),
                )
                .collect::<Vec<_>>();
            ready.push(ReadyNode {
                node: self.graph.graph[node_idx].clone(),
                inputs: input_data,
                node_idx,
            });
        }
        ready
    }

    /// Returns true when the graph is done, i.e. all nodes have been executed.
    pub fn is_done(&self) -> bool {
        self.running_nodes.is_empty() && self.done_nodes.len() == self.graph.graph.node_count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::tests::{TestOperation, instantiate};

    #[test]
    fn test_graph_scheduler() {
        let colored_graph = instantiate(2);
        println!("colored_graph: {colored_graph:?}");
        assert_eq!(colored_graph.graph.node_count(), 2);
        assert_eq!(colored_graph.input_nodes.len(), 1);
        for idx in colored_graph.graph.node_indices() {
            assert_eq!(
                colored_graph.graph[idx].color,
                idx.index(),
                "Node {idx:?} has color {:?} but should have color {idx:?}",
                colored_graph.graph[idx].color
            );
        }
        let mut scheduler = GraphScheduler::new(colored_graph);
        let mut ready_node = scheduler
            .init_nodes(vec!["CommitData".to_string()])
            .unwrap()
            .pop()
            .unwrap();
        assert!(matches!(ready_node.node.node, TestOperation::Test1));
        let output = ready_node.run(&()).unwrap();
        assert_eq!(
            output,
            format!("Test1: {:?}", vec!["CommitData".to_string()])
        );
        scheduler.mark_done(ready_node.node_idx, output).unwrap();
        assert_eq!(scheduler.done_nodes.len(), 1);
        assert_eq!(scheduler.running_nodes.len(), 0);

        let mut ready_node = scheduler.next_ready_nodes().pop().unwrap();
        assert!(
            matches!(ready_node.node.node, TestOperation::Test2),
            "Node {ready_node:?} has operation {:?}",
            ready_node.node.node
        );
        let output = ready_node.run(&()).unwrap();
        assert_eq!(
            output,
            format!(
                "Test2: {:?}",
                vec![format!("Test1: {:?}", vec!["CommitData".to_string()])]
            )
        );
        scheduler.mark_done(ready_node.node_idx, output).unwrap();
        println!("done_nodes: {:?}", scheduler.done_nodes);
        println!("running_nodes: {:?}", scheduler.running_nodes);
        assert!(scheduler.is_done());
    }
}
