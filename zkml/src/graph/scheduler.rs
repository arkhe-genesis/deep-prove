use crate::graph::{NodeId, NodeInput, NodeOutput, PortId, graph::Graph};
use anyhow::{Result, ensure};
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use std::{
    collections::{BTreeMap, HashMap, HashSet},
    fmt::Debug,
};

/// A trait defining executable operations that can be run as nodes in a computational graph.
///
/// This trait abstracts the execution logic for graph nodes, allowing different types
/// of operations to be represented and executed uniformly. Each node defines its
/// input/output data type and execution context requirements.
pub trait ExecNode {
    /// The data type for both inputs and outputs of this node.
    ///
    /// Must implement `Clone` to support data flow between nodes and potential
    /// caching/memoization strategies.
    type IO: Clone + Serialize + for<'a> Deserialize<'a>;

    /// The execution context type required by this node.
    ///
    /// The context contains setup parameters, configuration, or resources that
    /// nodes need for execution but shouldn't be serialized/sent over the network.
    /// Examples include database connections, cryptographic keys, or large lookup tables.
    type Context;

    /// Returns a human-readable description of this node's operation.
    ///
    /// Used for debugging, logging, and visualization of the computational graph.
    fn describe(&self) -> String;

    /// Executes the node's operation with the given context and input data.
    ///
    /// # Parameters
    ///
    /// * `ctx` - The execution context containing necessary resources
    /// * `inputs` - Vector of input data from predecessor nodes (ordered by port)
    ///
    /// # Returns
    ///
    /// The result of the computation, or an error if execution fails.
    fn run(&self, ctx: &Self::Context, inputs: Vec<Self::IO>) -> anyhow::Result<Vec<Self::IO>>;
}

/// A computational graph where nodes are colored for execution scheduling and partitioning.
///
/// This type alias represents a [`Graph`] specialized for executable nodes with colors.
/// The coloring enables:
/// - **Scheduling policies**: Determining which nodes can run concurrently
/// - **Partitioning**: Grouping nodes for distributed execution
/// - **Resource allocation**: Assigning nodes to specific machines or threads
///
/// # Type Parameters
///
/// * `N` - The executable node type implementing [`ExecNode`]
/// * `C` - The color type used for scheduling and partitioning
/// * `NodeID` - The node identifier type (defaults to [`DefaultNodeID`])
///
/// # Edge Weights
///
/// Edge weights store the output data (`N::IO`) from executed nodes, which flows
/// to successor nodes as input. The scheduler manages this data flow automatically.
pub type ExecGraph<N, C> = Graph<Colored<N, C>, (), (), <N as ExecNode>::IO>;

impl<N, C> ExecGraph<N, C>
where
    N: ExecNode,
{
    /// Creates a new empty executable graph.
    pub fn default_exec_graph() -> Self {
        Self::new()
    }
}

/// A wrapper that associates an executable node with a color for scheduling.
///
/// Colors are used to partition the computational graph for various execution strategies:
/// - **Machine assignment**: Nodes with the same color run on the same machine
/// - **Thread assignment**: Nodes with the same color run in the same thread pool
/// - **Scheduling policy**: Control which nodes can execute concurrently
/// - **Resource allocation**: Group nodes that share resources or constraints
///
/// # Type Parameters
///
/// * `N` - The executable node type implementing [`ExecNode`]
/// * `C` - The color type (often `usize`, `String`, or custom enum)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Colored<N, C> {
    /// The executable node
    pub node: N,
    /// The color/partition identifier
    pub color: C,
}

impl<N, C> Colored<N, C> {
    /// Creates a new colored node with the specified node and color.
    pub fn new(proving_node: N, color: C) -> Self {
        Self {
            node: proving_node,
            color,
        }
    }

    /// Returns a reference to the color of this node.
    pub fn color(&self) -> &C {
        &self.color
    }

    pub fn node_mut(&mut self) -> &mut N {
        &mut self.node
    }
}

/// A convenience trait for creating colored nodes.
///
/// This trait provides a fluent interface for associating colors with nodes,
/// making the code more readable when building colored graphs.
///
/// # Examples
///
/// ```rust
/// # use zkml::graph::scheduler::{IntoColor, ExecNode};
/// # #[derive(Clone, Debug)]
/// # enum MyOperation { Add, Multiply }
/// let colored_node = MyOperation::Add.colored("worker_1");
///
/// // Chainable for building graphs
pub trait IntoColor<C> {
    /// Associates this node with a color, creating a [`Colored`] wrapper.
    ///
    /// # Parameters
    ///
    /// * `color` - The color to associate with this node
    ///
    /// # Returns
    ///
    /// A [`Colored`] wrapper containing this node and the specified color.
    fn colored(self, color: C) -> Colored<Self, C>
    where
        Self: Sized;
}

impl<C, N> IntoColor<C> for N {
    /// Creates a colored node by wrapping this node with the given color.
    fn colored(self, color: C) -> Colored<Self, C> {
        Colored::new(self, color)
    }
}

/// Implementation that allows colored nodes to be executed directly.
///
/// This implementation delegates all [`ExecNode`] operations to the wrapped node,
/// making colored nodes transparent from an execution perspective while preserving
/// the color information for scheduling.
impl<N: ExecNode, C> ExecNode for Colored<N, C> {
    type IO = N::IO;
    type Context = N::Context;

    /// Executes the wrapped node's operation.
    fn run(&self, ctx: &Self::Context, input: Vec<Self::IO>) -> anyhow::Result<Vec<Self::IO>> {
        self.node.run(ctx, input)
    }

    /// Returns the wrapped node's description.
    fn describe(&self) -> String {
        self.node.describe()
    }
}

/// Policy for controlling which ready nodes are released for execution.
///
/// Release policies allow fine-grained control over parallelism and resource usage
/// by determining which nodes can run concurrently. Different policies optimize
/// for different scenarios:
///
/// - **Resource constraints**: Limit concurrent nodes to avoid resource exhaustion
/// - **Load balancing**: Distribute work evenly across workers/colors
/// - **Dependencies**: Ensure proper ordering when needed
#[derive(Debug, Clone, Default)]
pub enum ReleasePolicy {
    /// Release all ready nodes without restriction.
    ///
    /// This policy maximizes parallelism by allowing all ready nodes to execute
    /// concurrently, regardless of their colors. Best for scenarios with abundant
    /// resources and no coordination constraints.
    All,

    /// Release nodes only if no other node of the same color is currently running.
    ///
    /// This policy ensures that at most one node of each color runs at any time,
    /// which is useful for:
    /// - Resource partitioning (each color gets exclusive access to resources)
    /// - Avoiding conflicts between nodes of the same type
    /// - Simulating single-threaded execution per partition
    #[default]
    UniqueColoring,
}

impl ReleasePolicy {
    fn accept<N: ExecNode, C: PartialEq>(
        &self,
        node_index: NodeId,
        scheduler: &GraphScheduler<N, C>,
    ) -> bool {
        match self {
            ReleasePolicy::All => true,
            ReleasePolicy::UniqueColoring => {
                let node_color = &scheduler.graph[node_index].as_inner().unwrap().color;
                scheduler
                    .running_nodes
                    .iter()
                    .all(|nidx| &scheduler.graph[*nidx].as_inner().unwrap().color != node_color)
            }
        }
    }
}

/// A node that is ready for execution with all its input data available.
///
/// Ready nodes represent the executable units that the scheduler produces.
/// They contain everything needed for execution except the execution context,
/// which is provided separately to allow sharing contexts across multiple nodes.
///
/// # Type Parameters
///
/// * `N` - The executable node type implementing [`ExecNode`]
/// * `C` - The color type used for scheduling
/// * `NodeId` - The node identifier type (defaults to [`DefaultNodeId`])
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(bound(
    serialize = "N: Serialize, C: Serialize, N::IO: Serialize",
    deserialize = "N: DeserializeOwned, C: DeserializeOwned, N::IO: DeserializeOwned"
))]
pub struct ReadyNode<N: ExecNode, C> {
    /// The colored node containing the operation to execute
    pub(crate) node: Colored<N, C>,
    /// Input data collected from predecessor nodes and/or graph inputs
    pub(crate) inputs: Vec<N::IO>,
    /// The node's identifier in the original graph
    pub(crate) node_id: NodeId,
}

impl<N: ExecNode, C> ReadyNode<N, C> {
    /// Executes this ready node with the provided context.
    ///
    /// This method consumes the input data and runs the node's operation.
    /// After execution, the inputs are no longer available (they are drained).
    ///
    /// # Parameters
    ///
    /// * `ctx` - The execution context required by the node
    ///
    /// # Returns
    ///
    /// The result of executing the node's operation.
    pub fn run(&mut self, ctx: &N::Context) -> anyhow::Result<Vec<N::IO>> {
        self.node.run(ctx, self.inputs.drain(..).collect())
    }
}

/// A scheduler that manages the execution order and readiness of nodes in a computational graph.
///
/// The scheduler is responsible for:
/// - **Dependency tracking**: Ensuring nodes run only after their dependencies complete
/// - **Data flow management**: Routing output data from completed nodes to their successors
/// - **Readiness determination**: Identifying which nodes can execute based on available inputs
/// - **Release policy enforcement**: Controlling parallelism according to the configured policy
///
/// The scheduler operates in a push-pull model:
/// 1. Initialize with input data to get the first batch of ready nodes
/// 2. Execute ready nodes externally
/// 3. Mark nodes as done and provide their outputs
/// 4. Pull the next batch of ready nodes
/// 5. Repeat until all nodes are complete
///
/// # Type Parameters
///
/// * `N` - The executable node type implementing [`ExecNode`]
/// * `C` - The color type used for scheduling policies
/// * `NodeId` - The node identifier type (defaults to [`DefaultNodeId`])
#[derive(Debug, Clone)]
pub struct GraphScheduler<N: ExecNode, C> {
    /// The computational graph with colored nodes and data-carrying edges
    graph: ExecGraph<N, C>,
    /// Cached input data for nodes that receive both graph inputs and predecessor outputs
    ///
    /// Maps node IDs to their input data organized by port ID. This is used for nodes
    /// that have both external input edges and edges from other nodes.
    waiting_input_data: HashMap<NodeInput, N::IO>,
    /// Set of nodes currently being executed
    ///
    /// Used to track which nodes are in progress and enforce release policies.
    running_nodes: HashSet<NodeId>,
    /// Set of nodes that have completed execution
    ///
    /// Used to determine when the entire graph is complete and to avoid
    /// re-executing nodes.
    done_nodes: HashSet<NodeId>,
    /// Policy controlling which ready nodes are released for execution
    release_policy: ReleasePolicy,
}

impl<N, C> GraphScheduler<N, C>
where
    N: ExecNode + Clone,
    N::IO: Clone,
    C: PartialEq + Clone,
{
    /// Creates a new scheduler for the given executable graph.
    ///
    /// The scheduler is initialized with the default release policy
    /// ([`ReleasePolicy::UniqueColoring`]) and empty execution state.
    pub fn new(graph: ExecGraph<N, C>) -> Self {
        Self {
            graph,
            running_nodes: HashSet::new(),
            done_nodes: HashSet::new(),
            release_policy: ReleasePolicy::UniqueColoring,
            waiting_input_data: HashMap::new(),
        }
    }

    /// Returns the IDs of nodes that produce graph outputs.
    ///
    /// These are the nodes whose results should be collected as the final
    /// outputs of the computation.
    pub fn output_nodes(&self) -> Vec<NodeId> {
        self.graph.sink_nodes().collect()
    }

    /// Sets the release policy for this scheduler.
    ///
    /// The release policy controls which ready nodes are allowed to execute
    /// concurrently, enabling different parallelism and resource management strategies.
    pub fn with_release_policy(mut self, release_policy: ReleasePolicy) -> Self {
        self.release_policy = release_policy;
        self
    }

    /// Initializes the scheduler with input data and returns the first batch of ready nodes.
    ///
    /// This method must be called before any execution can begin. It processes the graph's
    /// input nodes and determines which ones can execute immediately (those with only
    /// external inputs) versus those that must wait for predecessor outputs.
    ///
    /// # Parameters
    ///
    /// * `input_data` - External input data for the graph, addressed by target
    ///   [`NodeInput`]. Except for exotic cases, the inputs should simply be
    ///   the port 0 of the target node.
    ///
    /// # Returns
    ///
    /// A vector of ready nodes that can be executed immediately.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The scheduler has already been initialized (has running nodes)
    /// - The input data length doesn't match the number of input ports
    ///
    /// NOTE: this function works under the current partitioning that all
    /// topological source nodes are pure input nodes.
    pub fn init_nodes(
        &mut self,
        input_data: HashMap<NodeInput, N::IO>,
    ) -> anyhow::Result<Vec<ReadyNode<N, C>>> {
        ensure!(self.running_nodes.is_empty(), "Running nodes must be empty");
        let input_data = input_data.into_iter().fold(
            HashMap::<NodeId, BTreeMap<PortId, N::IO>>::new(),
            |mut acc, (node_input, io)| {
                if self.graph.is_source(node_input.node_id) {
                    // If this is an input node, prepare its ordered inputs.
                    acc.entry(node_input.node_id)
                        .or_default()
                        .insert(node_input.port, io);
                } else {
                    // Otherwise, just store the IO data to be used later.
                    self.waiting_input_data.insert(node_input, io);
                }

                acc
            },
        );

        Ok(input_data
            .into_iter()
            .map(|(node_id, payload)| {
                let node = &self.graph[node_id];
                self.running_nodes.insert(node_id);
                ReadyNode {
                    node: node.as_inner().unwrap().clone(),
                    inputs: payload.into_values().collect(),
                    node_id,
                }
            })
            .collect::<Vec<_>>())
    }

    /// Marks a node as completed and propagates its output to successor nodes.
    ///
    /// This method updates the scheduler's internal state after a node has been executed.
    /// It removes the node from the running set, adds it to the completed set, and
    /// stores the output data on outgoing edges for successor nodes to consume.
    /// Furthermore, it returns the subset of outputs which corresponds to outputs of the graph,
    /// mapped with their output port in node `node_id`
    ///
    /// # Parameters
    ///
    /// * `node_id` - The ID of the completed node
    /// * `output` - The result produced by the node's execution
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The node was not in the running state
    /// - The node was already marked as done
    pub fn mark_done(
        &mut self,
        node_id: NodeId,
        outputs: &[N::IO],
    ) -> anyhow::Result<HashMap<NodeOutput, N::IO>> {
        ensure!(
            self.running_nodes.remove(&node_id),
            "{node_id} was not running"
        );
        ensure!(
            self.done_nodes.insert(node_id),
            "{node_id} was already done"
        );
        let mut graph_outputs: HashSet<_> = (0..outputs.len()).collect();
        // now look at all the nodes that are ready to run and put them in pending
        for (_, edge) in self
            .graph
            .outgoings_mut(node_id)
            // do not take nodes that are already done or pending
            .filter(|(_, edge)| {
                !self.done_nodes.contains(&edge.target())
                    && !self.running_nodes.contains(&edge.target())
            })
        {
            // Set the data to the edge such that the successors may run afterwards
            // we fetch the output corresponding to the edge source port
            ensure!(
                edge.ports().len() == 1,
                "Found an edge with more than one link, which is unsupported in execution graph"
            );
            let output_port = *edge.ports()[0].source_port;
            graph_outputs.remove(&output_port);
            edge.weight = Some(outputs[output_port].clone());
        }
        Ok(graph_outputs
            .into_iter()
            .map(|i| (NodeOutput::new(node_id, i), outputs[i].clone()))
            .collect())
    }

    /// Returns the next batch of nodes that are ready for execution.
    ///
    /// This method should be called after marking nodes as done to get the next
    /// set of executable nodes. It identifies nodes whose dependencies have been
    /// satisfied and applies the release policy to determine which ones can run.
    ///
    /// # Returns
    ///
    /// A vector of ready nodes that can be executed. The vector may be empty if:
    /// - No nodes are ready (waiting for more completions)
    /// - The release policy prevents nodes from running
    /// - All nodes have been completed
    pub fn next_ready_nodes(&mut self) -> Result<Vec<ReadyNode<N, C>>> {
        let mut ready = Vec::new();
        let ready_node_ids = self
            .graph
            .edges()
            .fold(HashMap::<NodeId, Vec<bool>>::new(), |mut acc, (_, edge)| {
                acc.entry(edge.target())
                    .or_default()
                    .push(edge.weight.is_some());
                acc
            })
            .into_iter()
            // only take the nodes whose ALL incoming edges are filled with input data
            .filter(|(_, edges)| edges.iter().all(|is_ready| *is_ready))
            // only take the nodes that are not already running
            .filter(|(node_id, _)| !self.running_nodes.contains(node_id))
            .map(|(node_id, _)| node_id)
            .collect::<Vec<_>>();

        for node_id in ready_node_ids {
            // need to check here if the node is ready to run - each time we add
            // a new node to the pending nodes, the policy might change
            // decisions so we need to check again for future nodes
            if !self.release_policy.accept(node_id, self) {
                continue;
            }
            self.running_nodes.insert(node_id);

            let mut input_data = BTreeMap::new();
            // collect both the input data and the input from the edges
            for (edge_id, edge) in self.graph.incomings_mut(node_id) {
                ensure!(
                    edge.weight.is_some(),
                    "Edge {edge_id:?} {} -> {} has no weight - invalid logic?",
                    edge.source(),
                    edge.target()
                );
                // take the data on this edge and set it to none - unwrap is
                // safe since index have been collected just before
                let data = edge.weight.take().unwrap();
                for port in edge.ports().iter() {
                    input_data.insert(port.target_port, data.clone());
                }
            }

            // also remove the potential input data for nodes that also
            // expect inoput data + predecessors data
            for (node_input, io) in self
                .waiting_input_data
                .extract_if(|node_input, _| node_input.node_id == node_id)
            {
                input_data.insert(node_input.port, io);
            }
            ready.push(ReadyNode {
                node: self.graph[node_id].as_inner().unwrap().clone(),
                inputs: input_data.into_values().collect(),
                node_id,
            });
        }
        Ok(ready)
    }

    /// Returns true if all nodes in the graph have been executed.
    ///
    /// The scheduler is considered done when there are no running nodes and
    /// the number of completed nodes equals the total number of nodes in the graph.
    pub fn is_done(&self) -> bool {
        self.running_nodes.is_empty() && self.done_nodes.len() == self.graph.node_count()
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
        assert_eq!(colored_graph.node_count(), 3);
        assert_eq!(colored_graph.source_nodes().count(), 1);
        let inputs = [(
            NodeInput::new(colored_graph.source_nodes().next().unwrap(), 0),
            "CommitData".to_string(),
        )];
        let mut scheduler = GraphScheduler::new(colored_graph);
        let mut ready_node = scheduler
            .init_nodes(inputs.into_iter().collect())
            .unwrap()
            .pop()
            .unwrap();
        let output = ready_node.run(&()).unwrap();
        assert_eq!(
            scheduler
                .mark_done(ready_node.node_id, &output)
                .unwrap()
                .len(),
            0
        );
        assert_eq!(scheduler.done_nodes.len(), 1);
        assert_eq!(scheduler.running_nodes.len(), 0);

        let mut ready_node = scheduler.next_ready_nodes().unwrap().pop().unwrap();
        assert!(matches!(ready_node.node.node, TestOperation::Test1));
        let output = ready_node.run(&()).unwrap();
        assert_eq!(
            output[0],
            format!("Test1: {:?}", vec!["CommitData".to_string()])
        );
        assert_eq!(
            scheduler
                .mark_done(ready_node.node_id, &output)
                .unwrap()
                .len(),
            0
        );

        let mut ready_node = scheduler.next_ready_nodes().unwrap().pop().unwrap();
        assert!(
            matches!(ready_node.node.node, TestOperation::Test2),
            "Node {ready_node:?} has operation {:?}",
            ready_node.node.node
        );
        let output = ready_node.run(&()).unwrap();
        assert_eq!(
            output[0],
            format!(
                "Test2: {:?}",
                vec![format!("Test1: {:?}", vec!["CommitData".to_string()])]
            )
        );
        assert_eq!(
            scheduler
                .mark_done(ready_node.node_id, &output)
                .unwrap()
                .len(),
            1
        );

        println!("done_nodes: {:?}", scheduler.done_nodes);
        println!("running_nodes: {:?}", scheduler.running_nodes);
        assert!(scheduler.is_done());
    }
}
