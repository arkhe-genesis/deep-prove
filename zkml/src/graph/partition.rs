use super::{
    executor::Executor,
    graph::Direction,
    scheduler::{Colored, ExecGraph, ExecNode, GraphScheduler},
};
use crate::{
    Deserialize, Serialize,
    graph::{NodeId, NodeInput, NodeOutput},
};
use anyhow::{anyhow, bail, ensure};
use std::{
    collections::{BTreeMap, HashMap, HashSet},
    fmt::Debug,
    hash::Hash,
};

/// A partition represents a subgraph of nodes that share the same color and can be executed together.
///
/// Partitions are the fundamental unit of distributed execution. Each partition contains:
/// - A subgraph of nodes with the same color
/// - Information about parent and child partitions for coordination
/// - Input data (for source partitions) or dependency information
///
/// The partitioning scheme enables distributed execution where different workers
/// can execute different partitions concurrently, with coordination happening
/// through partition outputs and inputs.
///
/// # Type Parameters
///
/// * `N` - The executable node type implementing [`ExecNode`]
/// * `C` - The color type used for partitioning
/// * `NodeID` - The node identifier type (defaults to [`DefaultNodeID`])
///
/// # Examples
///
/// ```rust
/// # use zkml::graph::partition::Partition;
/// # use zkml::graph::{Graph, scheduler::ExecGraph};
/// # use std::collections::BTreeSet;
/// // Partitions are typically created through graph partitioning
/// // See ExecGraph::partition_by_color for examples
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Partition<N: ExecNode, C: Eq + Hash> {
    /// The color identifier for this partition - all nodes in the partition share this color
    pub color: C,
    /// The executable subgraph for this partition
    ///
    /// Uses Option to allow consuming the graph during execution without cloning.
    /// The graph contains only nodes with the same color as this partition.
    pub graph: Option<ExecGraph<N, C>>,
    /// The colors of the parent partitions that should receive this partition's output
    ///
    /// `None` if this partition produces the final output of the entire computation.
    pub parent_partitions: HashMap<C, NodeOutput>,
    /// Set of child partition colors whose outputs this partition depends on
    ///
    /// When this partition executes, it must wait for outputs from all child partitions.
    /// The ordering is maintained through the BTreeSet to ensure deterministic execution.
    pub child_partition: HashMap<C, NodeInput>,
    /// Input data for source partitions
    ///
    /// Source partitions (those with no child partitions) receive external input data.
    /// Non-source partitions have empty inputs and wait for child partition outputs.
    pub inputs: HashMap<NodeId, N::IO>,
}

impl<N: ExecNode, C> Partition<N, C>
where
    C: Hash + Ord + Clone,
    <N as ExecNode>::IO: Clone,
{
    /// Creates a new partition with the specified parameters.
    ///
    /// # Parameters
    ///
    /// * `color` - The color identifier for this partition
    /// * `graph` - The executable subgraph containing nodes of this color
    /// * `child_partition` - Set of child partition colors this partition depends on
    /// * `inputs` - Input data (only for source partitions)
    /// * `parent_partition` - Color of parent partition (None for final output)
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The graph doesn't have exactly one output node
    /// - The parent partition color is the same as this partition's color
    /// - The child partitions contain this partition's color
    /// - Input/output constraints are violated (source partitions need inputs, others don't)
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use zkml::graph::partition::Partition;
    /// # use zkml::graph::{Graph, scheduler::ExecGraph};
    /// # use std::collections::BTreeSet;
    /// // Partitions are typically created through the partitioning process
    /// // rather than manually constructed
    /// ```
    pub fn new(
        color: C,
        graph: ExecGraph<N, C>,
        child_partition: HashMap<C, NodeInput>,
        inputs: HashMap<NodeId, N::IO>,
        parent_partitions: HashMap<C, NodeOutput>,
    ) -> anyhow::Result<Self> {
        ensure!(
            graph.sink_nodes().count() == 1,
            "graph should have exactly one output node"
        );
        ensure!(
            !parent_partitions.keys().any(|c| c == &color),
            "parent partition should not be the same as the current partition"
        );
        ensure!(
            !child_partition.contains_key(&color),
            "child partition should not contain the current partition"
        );
        let is_source = child_partition.is_empty();
        if is_source {
            ensure!(
                !inputs.is_empty(),
                "source partition should have exactly some inputs"
            );
        } else {
            ensure!(inputs.is_empty(), "sink partition should have no inputs");
        }
        Ok(Self {
            color,
            graph: Some(graph),
            child_partition,
            inputs,
            parent_partitions,
        })
    }
    /// Returns true if this is a source partition (has external inputs).
    ///
    /// Source partitions are those that receive external input data and have no
    /// child partitions to wait for. They can begin execution immediately.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use zkml::graph::partition::Partition;
    /// # use zkml::graph::{Graph, scheduler::ExecGraph};
    /// # use std::collections::BTreeSet;
    /// // For a partition with inputs
    /// // assert!(partition.is_source_partition());
    ///
    /// // For a partition without inputs (waits for child outputs)
    /// // assert!(!partition.is_source_partition());
    /// ```
    pub fn is_source_partition(&self) -> bool {
        !self.inputs.is_empty()
    }
}

/// Represents the output produced by a partition after execution.
///
/// Partition outputs are used to coordinate execution between partitions in a
/// distributed setting. Each output contains the result data and routing information
/// to determine where the output should be sent next.
///
/// # Type Parameters
///
/// * `N` - The executable node type implementing [`ExecNode`]
/// * `C` - The color type used for partition identification
///
/// # Examples
///
/// ```rust
/// # use zkml::graph::partition::PartitionOutput;
/// // Outputs are typically created by the partition scheduler
/// // during execution, not constructed manually
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PartitionOutput<N: ExecNode, C> {
    /// The color of the partition that produced this output
    pub from: C,
    /// The colors of the partitions that should receive this output.
    ///
    /// `None` indicates this is a final output of the entire computation
    pub to: Option<C>,
    /// The actual output data produced by the partition
    ///
    /// Due to the partitioning constraint of at most one edge between partition pairs,
    /// each partition produces exactly one output. However, partitions can have
    /// multiple inputs from different child partitions.
    pub output: N::IO,
}

impl<N: ExecNode, C> PartitionOutput<N, C> {
    /// Returns true if this output represents the final result of the computation.
    ///
    /// Final outputs have no destination partition (`to` is `None`) and represent
    /// the completed result that should be returned to the caller.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use zkml::graph::partition::PartitionOutput;
    /// // For a final output
    /// // assert!(output.is_final_output());
    ///
    /// // For an intermediate output
    /// // assert!(!output.is_final_output());
    /// ```
    pub fn is_final_output(&self) -> bool {
        self.to.is_none()
    }
}

/// A scheduler for executing multiple partitions of a graph in sequence.
///
/// The partition scheduler manages the execution of a series of partitions that belong
/// to the same color/worker. It handles:
/// - Sequential execution of partitions (only one active at a time)
/// - Coordination with child partitions through input/output passing
/// - State management for pending outputs from child partitions
///
/// This scheduler is designed for distributed execution scenarios where a single
/// worker may need to execute multiple disconnected partitions over time, with
/// coordination happening between different workers through partition outputs.
///
/// # Type Parameters
///
/// * `N` - The executable node type implementing [`ExecNode`]
/// * `C` - The color type used for partition identification
/// * `E` - The executor type for running individual partitions
///
/// # Examples
///
/// ```rust
/// # use zkml::graph::partition::PartitionScheduler;
/// # use zkml::graph::executor::SequentialExecutor;
/// // Schedulers are typically created with partitions from graph partitioning
/// // See the test cases for complete usage examples
/// ```
pub struct PartitionScheduler<N: ExecNode, C: Eq + Hash, E: Executor<N, C>> {
    /// Queue of partitions to be executed in order
    ///
    /// Only one partition is active at a time. If multiple partitions could run
    /// simultaneously, they would be merged into a single partition during the
    /// partitioning process.
    partitions: Vec<Partition<N, C>>,
    /// Configuration for the executor used to run each partition
    ///
    /// The executor is embedded in the scheduler because partitions are designed
    /// to be self-contained execution units. This encapsulation hides the internal
    /// node structure from external APIs.
    executor_config: E::Config,
    /// Buffer for outputs received from child partitions
    ///
    /// Maps child partition colors to their output data. The scheduler waits
    /// for all required child outputs before executing the next partition.
    pending_child_outputs: HashMap<C, N::IO>,
    /// Execution context shared across all partition executions
    context: N::Context,
    /// The color of all partitions for this scheduler
    pub color: C,
}

impl<N, C, E> PartitionScheduler<N, C, E>
where
    N: ExecNode + Clone,
    C: PartialEq + Eq + Clone + Hash + Ord + Debug,
    E: Executor<N, C>,
    <N as ExecNode>::IO: Clone + for<'a> Deserialize<'a> + Serialize + Debug,
{
    /// Creates a new partition scheduler with the given partitions and configuration.
    ///
    /// # Parameters
    ///
    /// * `partitions` - Vector of partitions to execute in order
    /// * `context` - Execution context for running partition nodes
    /// * `executor_config` - Configuration for the partition executor
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The partitions vector is empty
    /// - Any partition doesn't have a graph
    /// - Any partition doesn't have exactly one output node
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use zkml::graph::partition::PartitionScheduler;
    /// # use zkml::graph::executor::SequentialExecutor;
    /// // Typically created with partitions from graph.partition_by_color()
    /// // let scheduler = PartitionScheduler::<MyNode, usize, SequentialExecutor>::new(
    /// //     partitions, context, executor_config
    /// // ).unwrap();
    /// ```
    pub fn new(
        partitions: Vec<Partition<N, C>>,
        context: N::Context,
        executor_config: E::Config,
    ) -> anyhow::Result<Self> {
        ensure!(!partitions.is_empty(), "Partitions must be non-empty");
        ensure!(
            partitions.iter().all(|p| p.graph.is_some()),
            "All partitions must have a graph"
        );
        ensure!(
            partitions
                .iter()
                .all(|p| p.graph.as_ref().unwrap().sink_nodes().count() == 1),
            "All partitions must have exactly one output node"
        );
        let color = partitions.first().unwrap().color.clone();
        ensure!(
            partitions.iter().all(|p| p.color == color),
            "All partitions must have the same color"
        );
        Ok(Self {
            partitions,
            executor_config,
            pending_child_outputs: HashMap::new(),
            context,
            color,
        })
    }
    /// Attempts to execute the next partition in the queue.
    ///
    /// This method checks if the next partition is ready to run and executes it if possible.
    /// The readiness depends on the partition type:
    ///
    /// - **Source partitions**: Have input data and can run immediately
    /// - **Non-source partitions**: Must wait for all child partition outputs
    ///
    /// # Returns
    ///
    /// - `Ok(Some(output))` - Partition executed successfully, returns its output
    /// - `Ok(None)` - Partition not ready (waiting for child outputs) or no partitions left
    /// - `Err(_)` - Execution error
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use zkml::graph::partition::PartitionScheduler;
    /// # use zkml::graph::executor::SequentialExecutor;
    /// // let mut scheduler = PartitionScheduler::new(partitions, context, config).unwrap();
    /// //
    /// // while !scheduler.is_done() {
    /// //     if let Some(output) = scheduler.try_run_partition().unwrap() {
    /// //         // Handle partition output
    /// //         if output.is_final_output() {
    /// //             println!("Final result: {:?}", output.output);
    /// //         } else {
    /// //             // Send to next partition/worker
    /// //         }
    /// //     }
    /// // }
    /// ```
    pub fn try_run_partition(&mut self) -> anyhow::Result<Vec<PartitionOutput<N, C>>> {
        if self.partitions.is_empty() {
            return Ok(vec![]);
        }
        let next_partition = self.partitions.get_mut(0);
        let inputs: Option<HashMap<NodeInput, N::IO>> = match next_partition {
            Some(part) => {
                if !part.is_source_partition() {
                    // the next partition expects outputs from its child
                    // partitions (e.g. it has no graph data input). we need to
                    // check if all the child outputs have been received.
                    let all_present = part
                        .child_partition
                        .keys()
                        .all(|color| self.pending_child_outputs.contains_key(color));
                    if !all_present {
                        None
                    } else {
                        Some(
                            part.child_partition
                                .iter()
                                .map(|(color, node_input)| {
                                    (
                                        *node_input,
                                        self.pending_child_outputs.remove(color).unwrap(),
                                    )
                                })
                                .collect(),
                        )
                    }
                } else {
                    // otherwise, the partition is a source partition, i.e. a partition that doesn't have
                    // any child partitions so we just read their inputs.
                    Some(
                        part.inputs
                            .drain()
                            .map(|(node_id, io)| (NodeInput::new(node_id, 0), io))
                            .collect(),
                    )
                }
            }
            None => unreachable!("partition should not be empty - precheck passed"),
        };
        match inputs {
            // nothing to do for now
            None => Ok(vec![]),
            // we either are running the sink partition or any parent partition who has received all its inputs from other partitions.
            Some(inputs) => {
                let mut partition = self.partitions.remove(0);
                let scheduler = GraphScheduler::new(partition.graph.take().unwrap());
                let outputs = E::run(&self.executor_config, scheduler, inputs, &self.context)?;
                // this set is used to determine the subset of `outputs` which are not linked
                // to a node in another partition (i.e., they are output of the entire graph)
                let mut graph_outputs: HashSet<_> = outputs.keys().collect();
                let partition_outputs = partition
                    .parent_partitions
                    .into_iter()
                    .map(|(parent_color, output_port)| {
                        let output = outputs
                            .get(&output_port)
                            .ok_or(anyhow!(
                                "Output {output_port} not found in partition graph outputs"
                            ))?
                            .clone();
                        graph_outputs.remove(&output_port);
                        Ok(PartitionOutput {
                            output,
                            from: partition.color.clone(),
                            to: Some(parent_color),
                        })
                    })
                    .collect::<anyhow::Result<Vec<_>>>()?;
                // now, build the `PartitionOutput` for output ports which correspond to outputs of the
                // whole graph
                graph_outputs.into_iter().try_fold(
                    partition_outputs,
                    |mut partition_outputs, output_port| {
                        let output = outputs
                            .get(output_port)
                            .ok_or(anyhow!(
                                "Output {output_port} not found in partition graph outputs"
                            ))?
                            .clone();
                        partition_outputs.push(PartitionOutput {
                            output,
                            from: partition.color.clone(),
                            to: None,
                        });
                        Ok(partition_outputs)
                    },
                )
            }
        }
    }

    /// Provides output from a child partition to this scheduler.
    ///
    /// When a child partition completes execution, its output must be provided
    /// to parent partitions that depend on it. This method stores the output
    /// until all required child outputs are available.
    ///
    /// # Parameters
    ///
    /// * `output` - The output from a completed child partition
    ///
    /// # Errors
    ///
    /// Returns an error if the output is from an unexpected child partition
    /// (not in the current partition's child set).
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use zkml::graph::partition::{PartitionScheduler, PartitionOutput};
    /// # use zkml::graph::executor::SequentialExecutor;
    /// // let mut scheduler = PartitionScheduler::new(partitions, context, config).unwrap();
    /// // let child_output = PartitionOutput { /* ... */ };
    /// //
    /// // scheduler.set_child_partition_output(child_output).unwrap();
    /// //
    /// // // Now try to run the partition that was waiting for this output
    /// // if let Some(result) = scheduler.try_run_partition().unwrap() {
    /// //     // Partition executed with the child output
    /// // }
    /// ```
    pub fn set_child_partition_output(
        &mut self,
        output: PartitionOutput<N, C>,
    ) -> anyhow::Result<()> {
        self.set_child_output(output.from, output.output)
    }

    /// Injects an output from a child partition directly using the color and IO data.
    ///
    /// This method is useful when you have deserialized partition output data
    /// and need to inject it into the scheduler without constructing the full `PartitionOutput` type.
    ///
    /// # Arguments
    ///
    /// * `from` - The color of the child partition that produced this output
    /// * `output` - The actual IO data produced by the child partition
    ///
    /// # Errors
    ///
    /// Returns an error if the output is not expected by the current partition
    /// (i.e., if `from` is not in the current partition's `child_partition` map).
    pub fn set_child_output(&mut self, from: C, output: N::IO) -> anyhow::Result<()> {
        if self.partitions.is_empty() {
            return Ok(());
        }
        let next_partition = self.partitions.first().unwrap();
        if next_partition.child_partition.contains_key(&from) {
            // we know the output is expected so we save it internally, and
            // it'll be used at the next run if all outputs of all child
            // partitions have been received.
            self.pending_child_outputs.insert(from, output);
        } else {
            bail!(
                "output of child partition {:?} not expected for current partition {:?}",
                from,
                next_partition.color
            );
        }
        Ok(())
    }

    /// Returns true if all partitions have been executed.
    ///
    /// A scheduler is done when its partition queue is empty, meaning all
    /// partitions assigned to this scheduler have completed execution.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use zkml::graph::partition::PartitionScheduler;
    /// # use zkml::graph::executor::SequentialExecutor;
    /// // let mut scheduler = PartitionScheduler::new(partitions, context, config).unwrap();
    /// //
    /// // while !scheduler.is_done() {
    /// //     // Execute partitions...
    /// // }
    /// // println!("All partitions completed!");
    /// ```
    pub fn is_done(&self) -> bool {
        self.partitions.is_empty()
    }
}

impl<N, C> ExecGraph<N, C>
where
    C: PartialEq + Eq + Clone + Hash + Ord + Debug,
    N: ExecNode + Clone + Debug,
    N::IO: Clone,
    <N as ExecNode>::IO: Clone + Debug,
{
    /// Partitions the graph by node colors for distributed execution.
    ///
    /// This method splits the graph into independent partitions where each partition
    /// contains only nodes of the same color. The resulting partitions can be
    /// executed on different machines or workers.
    ///
    /// # Parameters
    ///
    /// * `inputs` - External input data for the graph
    ///
    /// # Returns
    ///
    /// A map from colors to vectors of partitions. Multiple partitions with the
    /// same color can exist if they are disconnected subgraphs.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use zkml::graph::{Graph, scheduler::ExecGraph};
    /// # use std::collections::HashMap;
    /// // let graph: ExecGraph<MyNode, usize> = create_colored_graph();
    /// // let inputs = vec![input_data1, input_data2];
    /// //
    /// // let partitions = graph.partition_by_color(inputs).unwrap();
    /// //
    /// // // Execute partitions on different workers
    /// // for (color, color_partitions) in partitions {
    /// //     // Send color_partitions to worker responsible for this color
    /// // }
    /// ```
    #[allow(clippy::type_complexity)]
    pub fn partition_by_color(
        &self,
        inputs: Vec<(NodeId, N::IO)>,
    ) -> anyhow::Result<HashMap<C, Vec<Partition<N, C>>>> {
        self.partition_by(|node| node.color(), inputs)
    }

    /// Partitions the graph using a custom color extraction function.
    ///
    /// This is the general partitioning method that allows custom logic for
    /// determining node colors. It performs a depth-first search to find
    /// connected components of the same color and creates partitions from them.
    ///
    /// # Parameters
    ///
    /// * `node_color` - Function to extract the color from a colored node
    /// * `inputs` - External input data for the graph
    ///
    /// # Returns
    ///
    /// A map from colors to vectors of partitions, where each partition is
    /// a connected subgraph of nodes with the same color.
    ///
    /// # Algorithm
    ///
    /// 1. Performs DFS from each unvisited node to find color-connected components
    /// 2. Creates subgraphs for each component
    /// 3. Establishes parent-child relationships between partitions
    /// 4. Sets up input/output edges for coordination
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use zkml::graph::{Graph, scheduler::ExecGraph};
    /// # use std::collections::HashMap;
    /// // Custom color extraction (e.g., based on node properties)
    /// // let partitions = graph.partition_by(
    /// //     |node| &node.custom_color_field(),
    /// //     inputs
    /// // ).unwrap();
    /// ```
    #[allow(clippy::type_complexity)]
    pub fn partition_by(
        &self,
        node_color: impl Fn(&Colored<N, C>) -> &C,
        inputs: Vec<(NodeId, N::IO)>,
    ) -> anyhow::Result<HashMap<C, Vec<Partition<N, C>>>> {
        let mut visited = HashSet::new();
        // for each color, we keep a list of its partitions:
        // first element is the vector graphs for all partitions
        // second element is the associated mapping original_graph_index => new_partition_index
        let mut map = BTreeMap::<C, Vec<ExecGraph<N, C>>>::new();

        // We start iterating from the input nodes of the original graph, so we
        // create the partitions "in order", starting from the lower partitions
        // to the higher ones as this is the order of the execution of the
        // graph.
        for (node, _) in self.forward_iter() {
            if visited.contains(&node) {
                continue;
            }
            let color = node_color(self[node].inner());

            // Try to reach all nodes from this given node using DFS and
            // only keep the ones having the same color
            let mut stack = vec![node];
            let mut partition = Vec::new();
            while let Some(n) = stack.pop() {
                // always make sure to visit nodes only once
                if !visited.insert(n) {
                    continue;
                }
                partition.push(n);
                for (_, edge) in self.neighbors(n, Direction::Any) {
                    // there is a directly connected node sharing the same
                    // color, so it's part of the same partition.
                    // unwrap is safe since we filtered out the edges that are
                    // not between nodes
                    let other_end = edge.other_end(n).unwrap();
                    if node_color(self[other_end].inner()) == color {
                        stack.push(other_end);
                    }
                }
            }

            // Build a new Graph from this partition
            let mut sub = ExecGraph::<N, C>::new();
            // add all nodes to the graph
            for &node_id in &partition {
                // we put empty edges for now since not all nodes in the
                // partition have been added to the graph, we don't know yet
                // their new index in the partition and therefore can't create
                // all edges yet.
                let node = self[node_id].clone();
                sub.add_node_with_id(node_id, node)?;
            }
            // add all edges inside the partition - excluding for now the input
            // and output edges
            for &node_id in &partition {
                // we only add incoming edges - since eventually we go over all
                // nodes of the graph, then we should have covered all the edges
                for (_, edge) in self.incomings(node_id) {
                    // if the source is in the same partition, then we add the
                    // edge
                    if let Some(_source_node) = sub.node(edge.source()) {
                        sub.add_edges_raw(vec![edge.clone()])?;
                    }
                }
            }

            map.entry(color.clone()).or_default().push(sub);
        }

        let mut graph_root = self.sink_nodes().collect::<Vec<_>>();
        ensure!(
            graph_root.len() == 1,
            "graph should have exactly one output node"
        );
        let graph_root = graph_root.remove(0);

        // At this point all the subgraphs have been built, but there some
        // information missing:
        // - the links between the subgraphs, we need to
        //   extract which color partition depends on which other color partition.
        // - and then from it create the input edge on sink partitions: one node
        //   in each "parent" partition must now become an input node in that
        //   partition to receive the output of the children partitions. The order
        //   is important here.
        // - Finding and setting the parent partition for
        //   each partition.
        map.into_iter()
            // all partitions sharing this color
            .map(|(color, partitions)| {
                // the partitions as graphs
                let mut final_partitions = Vec::with_capacity(partitions.len());
                for subgraph in partitions.into_iter() {
                    let mut child_partition_colors = HashMap::<C, NodeInput>::new();
                    let mut partition_inputs = HashMap::<NodeId, N::IO>::new();
                    // the input nodes in the new partition that should receive
                    // input data we need to take the _same_ order of the inputs
                    // - given graph has a HashMap we take the ordering of the
                    // graph
                    //
                    // TODO: maybe just turn the graph hashmap into a btreemap
                    // directly ?

                    // we are in a parent partition, so we need to manually add
                    // the input edges
                    if inputs.iter().all(|(node_id, _)| subgraph.node(*node_id).is_none()) {
                        let source_nodes: Vec<_> = subgraph.source_nodes().collect();
                        ensure!(
                            source_nodes.len() == 1,
                            "INVALID GRAPH: a parent partition should have exactly one source node"
                        );
                        let source_node = source_nodes[0];
                        // now search the incoming edges of the source node in
                        // the original graph. For each edge, we add that info
                        // to the new partition
                        let edges = self
                            .incomings(source_node)
                            .collect::<Vec<_>>();

                        for (_, edge) in edges.into_iter()
                        {
                            // and we keep track of the order of the colors
                            // unwrap is safe since we filtered out the edges
                            // that are not between nodes
                            let edge_color = self[edge.source()].inner().color().clone();
                            for link in edge.ports().iter() {
                                child_partition_colors.insert(
                                    edge_color.clone(),
                                    NodeInput::new(source_node, link.target_port)
                                );
                            }
                        }
                    } else {
                        // we are in a partition with raw inputs, select the
                        // subset of raw inputs that map into this partition
                        // (node IDs are stable between the original graph and
                        // the partitions).
                        partition_inputs.extend(
                            inputs
                                .iter()
                                .filter_map(|(node_id, payload)|
                                   if subgraph.node(*node_id).is_some() {
                                       Some((*node_id, payload.clone()))
                                   } else {
                                       None
                                   })
                        );
                    }

                    // now we want to find the parent partition for this
                    // partition such that its output can be sent to it. we have
                    // to first find the root of the partition, and then set it
                    // as an output
                    let mut partition_root = subgraph.nodes()
                        .filter_map(|(&node_id, _)| {
                            if subgraph.is_sink(node_id) {
                                Some(node_id)
                            } else {
                                None
                            }
                        }).collect::<Vec<_>>();
                    ensure!(
                        partition_root.len() == 1,
                        "graph should have exactly one output node: found {} roots on partition of color {color:?}: {:?} {:?}",
                        partition_root.len(),
                        subgraph.nodes().collect::<Vec<_>>(),
                        subgraph.edges,
                    );
                    let partition_root = partition_root.remove(0);
                    let parent_nodes = self
                        .outgoings(partition_root)
                        .map(|(_, e)| {
                            ensure!(
                                e.ports().len() == 1,
                                "Output edge of partition must have only one link, found {} links",
                                e.ports().len()
                            );
                            Ok((e.target(), e.ports()[0].source_port))
                        }).collect::<anyhow::Result<Vec<_>>>()?;

                    // check if the partition is the root partition
                    let parent_partitions = if graph_root != partition_root {
                        ensure!(!parent_nodes.is_empty(), "any non root partition should have at least one parent partition");
                        parent_nodes.into_iter().try_fold(
                            HashMap::new(),
                            |mut parent_partitions, (parent_node, out_port)| {
                            let parent_color = self[parent_node].inner().color().clone();
                            ensure!(
                                parent_partitions
                                    .insert(parent_color.clone(), NodeOutput::new(partition_root, out_port))
                                    .is_none(),
                                "Found multiple output edges from partition {color:?} to parent partition with color {parent_color:?}"
                            );
                            Ok(parent_partitions)
                        })?
                    } else {
                        HashMap::new()
                    };

                    final_partitions.push(Partition::<N, C>::new(
                        color.clone(),
                        subgraph,
                        child_partition_colors,
                        partition_inputs,
                        parent_partitions,
                    )?);
                }
                Ok(final_partitions)
            })
            .collect::<anyhow::Result<Vec<_>>>()
            .map(|m| {
                m.into_iter().flatten().fold(
                    HashMap::<C, Vec<Partition<N, C>>>::new(),
                    |mut acc, partition| {
                        acc.entry(partition.color.clone())
                            .or_default()
                            .push(partition);
                        acc
                    },
                )
            })
    }
}

#[cfg(test)]
mod tests {
    use itertools::Itertools;
    use rstest::rstest;

    use super::*;
    use crate::graph::{
        Graph, PortLink, Ports,
        executor::{SequentialExecutor, tests::MathAST},
        scheduler::IntoColor,
    };

    ///            Pow_1
    ///            Pow_3
    /// .          Sub_3
    /// .  Div_1            Div_2
    /// Add_1.  Sub_1.  Add_2 .  Sub_2
    /// The subscript indicates the color of the node.
    /// So there should be 3 partitions
    /// and the inputs indices should be [1,2,5,6,3,4,7,8]
    /// Reason to choose sub and div is to test the non commutativity nature of the tasks, so the partitioning
    /// should dispatch the inputs to the correct partition in the right order and place.
    /// if `additional_output` is true, then we add an `Sqrt` node after `Pow_3` to have a
    /// node with 2 outputs, with one of these outputs being an extra output for the graph
    fn create_graph(additional_output: bool) -> (ExecGraph<MathAST, usize>, NodeId) {
        let mut graph = Graph::new();
        let input_node_ids = [
            graph.add_inner(MathAST::Input(0).colored(0)).unwrap(),
            graph.add_inner(MathAST::Input(1).colored(0)).unwrap(),
            graph.add_inner(MathAST::Input(2).colored(1)).unwrap(),
            graph.add_inner(MathAST::Input(3).colored(1)).unwrap(),
            graph.add_inner(MathAST::Input(4).colored(0)).unwrap(),
            graph.add_inner(MathAST::Input(5).colored(0)).unwrap(),
            graph.add_inner(MathAST::Input(6).colored(1)).unwrap(),
            graph.add_inner(MathAST::Input(7).colored(1)).unwrap(),
        ];
        // first partition
        let add1 = graph.add_inner(MathAST::Add.colored(0)).unwrap();
        graph
            .add_edge(input_node_ids[0], add1, (0, 0), None)
            .unwrap();
        graph
            .add_edge(input_node_ids[1], add1, (0, 1), None)
            .unwrap();

        let sub1 = graph.add_inner(MathAST::Sub.colored(0)).unwrap();
        graph
            .add_edge(input_node_ids[4], sub1, (0, 0), None)
            .unwrap();
        graph
            .add_edge(input_node_ids[5], sub1, (0, 1), None)
            .unwrap();

        let agg1 = graph.add_inner(MathAST::Div.colored(0)).unwrap();
        graph
            .add_edge(add1, agg1, Ports::consecutive(), None)
            .unwrap();
        // (0,1) cause target slot on agg1 0 is already taken ^
        graph
            .add_edge(sub1, agg1, PortLink::new(0, 1), None)
            .unwrap();

        // second partition
        let add2 = graph.add_inner(MathAST::Add.colored(1)).unwrap();
        graph
            .add_edge(input_node_ids[2], add2, (0, 0), None)
            .unwrap();
        graph
            .add_edge(input_node_ids[3], add2, (0, 1), None)
            .unwrap();

        let sub2 = graph.add_inner(MathAST::Sub.colored(1)).unwrap();
        graph
            .add_edge(input_node_ids[6], sub2, (0, 0), None)
            .unwrap();
        graph
            .add_edge(input_node_ids[7], sub2, (0, 1), None)
            .unwrap();

        let agg2 = graph.add_inner(MathAST::Div.colored(1)).unwrap();
        graph
            .add_edge(add2, agg2, Ports::consecutive(), None)
            .unwrap();
        graph
            .add_edge(sub2, agg2, PortLink::new(0, 1), None)
            .unwrap();

        // third partition
        let agg3 = graph.add_inner(MathAST::Sub.colored(2)).unwrap();
        graph
            .add_edge(agg1, agg3, Ports::consecutive(), None)
            .unwrap();
        graph
            .add_edge(agg2, agg3, PortLink::new(0, 1), None)
            .unwrap();

        let mut agg33 = graph.add_inner(MathAST::Pow2.colored(2)).unwrap();
        graph
            .add_edge(agg3, agg33, Ports::consecutive(), None)
            .unwrap();

        if additional_output {
            // add an extra output to test multiple outputs for a graph
            // We add an Sqrt node to have a node with 2 outputs
            let sqrt = graph.add_inner(MathAST::Sqrt.colored(2)).unwrap();
            // Link the node to the previous Pow2 node
            graph
                .add_edge(agg33, sqrt, Ports::consecutive(), None)
                .unwrap();
            // add another Pow2 node to recompute the same output of previous `agg33` node
            agg33 = graph.add_inner(MathAST::Pow2.colored(2)).unwrap();
            graph
                .add_edge(sqrt, agg33, Ports::consecutive(), None)
                .unwrap();
        }

        let pow1 = graph.add_inner(MathAST::Pow2.colored(0)).unwrap();
        graph
            .add_edge(agg33, pow1, Ports::consecutive(), None)
            .unwrap();

        (graph, pow1)
    }

    #[test]
    fn test_partition_by_color() {
        let (graph, agg33) = create_graph(false);
        assert_eq!(graph.sink_nodes().collect::<Vec<_>>(), vec![agg33]);
        let partitions = graph
            .partition_by_color(
                [1, 2, 3, 4, 5, 6, 7, 8]
                    .into_iter()
                    .enumerate()
                    .map(|(i, io)| (i.into(), io))
                    .collect(),
            )
            .unwrap();
        assert_eq!(partitions.len(), 3);
        assert_eq!(partitions.get(&0).unwrap().len(), 2);
        assert_eq!(partitions.get(&1).unwrap().len(), 1);
        assert_eq!(partitions.get(&2).unwrap().len(), 1);
        assert_eq!(
            partitions.get(&0).unwrap()[0]
                .inputs
                .keys()
                .map(|x| **x)
                .collect::<HashSet<_>>(),
            [0, 1, 4, 5].into_iter().collect()
        );
        assert_eq!(
            partitions.get(&1).unwrap()[0]
                .inputs
                .keys()
                .map(|x| **x)
                .collect::<HashSet<_>>(),
            [2, 3, 6, 7].into_iter().collect()
        );
        assert_eq!(partitions.get(&2).unwrap()[0].inputs.len(), 0);
        assert_eq!(partitions.get(&2).unwrap()[0].child_partition.len(), 2);
        assert_eq!(
            partitions.get(&2).unwrap()[0]
                .child_partition
                .keys()
                .copied()
                .collect::<HashSet<_>>(),
            [0, 1].into_iter().collect()
        );
        assert!(
            partitions.get(&0).unwrap()[0]
                .parent_partitions
                .contains_key(&2)
        );
        assert!(
            partitions.get(&1).unwrap()[0]
                .parent_partitions
                .contains_key(&2)
        );
        assert!(
            partitions.get(&2).unwrap()[0]
                .parent_partitions
                .contains_key(&0)
        );
        assert_eq!(partitions.get(&0).unwrap()[1].parent_partitions.len(), 0);
    }

    /// A simple test to check the different partition schedulers can drive the graph to completion.
    /// There is no implementation of a local partition executor since that would be pointless, as the
    /// only reason to have a partition is to run it in different machines.
    #[rstest]
    #[case::base(false)]
    #[case::additional_output(true)]
    fn test_partition_scheduler(#[case] additional_output: bool) -> anyhow::Result<()> {
        let (graph, _agg33) = create_graph(additional_output);
        // add1[0,1] = 1+7
        // sub1[4,5] = 4-2
        // add2[2,3] = 3+4
        // sub2[6,7] = 6-3
        // agg1 = add1 / sub1 = 8 / 2 = 4
        // agg2 = add2 / sub2 = 7 / 3 = 2
        // agg3 = agg1 - agg2 = 4 - 2 = 2
        // agg33 = pow2(agg3) = 2^2 = 4
        // final output = pow1 = pow2(agg33) = 4^2 = 16
        // additional_output = -sqrt(agg33) = -2
        let partitions = graph.partition_by_color(
            [1, 7, 3, 4, 4, 2, 6, 3]
                .into_iter()
                .enumerate()
                .map(|(i, io)| (i.into(), io))
                .collect(),
        )?;
        let mut schedulers =
            partitions
                .into_iter()
                .fold(HashMap::new(), |mut map, (color, partitions)| {
                    map.insert(
                        color,
                        PartitionScheduler::<_, _, SequentialExecutor>::new(partitions, (), ())
                            .unwrap(),
                    );
                    map
                });
        let p1_outputs = schedulers.get_mut(&0).unwrap().try_run_partition()?;
        ensure!(!p1_outputs.is_empty() && p1_outputs.first().unwrap().to.unwrap() == 2);
        let p2_outputs = schedulers.get_mut(&1).unwrap().try_run_partition()?;
        ensure!(!p2_outputs.is_empty() && p2_outputs.first().unwrap().to.unwrap() == 2);
        schedulers
            .get_mut(&2)
            .unwrap()
            .set_child_partition_output(p1_outputs.first().unwrap().clone())?;
        // there should not be any computation possible on partition 2 since it doesn't have all its inputs
        ensure!(
            schedulers
                .get_mut(&2)
                .unwrap()
                .try_run_partition()?
                .is_empty()
        );
        schedulers
            .get_mut(&2)
            .unwrap()
            .set_child_partition_output(p2_outputs.first().unwrap().clone())?;
        let p3_outputs = schedulers.get_mut(&2).unwrap().try_run_partition()?;
        // goes back to partition 0
        let num_expected_outputs = 1 + additional_output as usize;
        ensure!(
            p3_outputs.len() == num_expected_outputs,
            "invalid number of outputs found from partition 2: expected {num_expected_outputs}, found {}",
            p3_outputs.len(),
        );
        let p3_output = if num_expected_outputs == 1 {
            p3_outputs.first().unwrap()
        } else {
            let (pos, p3_final_output) = p3_outputs.iter()
                .find_position(|out|
                    out.is_final_output()
                ).ok_or(
                    anyhow!("expected a final output from partition 2 in additional output test case, but none found")
                )?;
            ensure!(p3_final_output.output == -2);
            ensure!(p3_final_output.from == 2);
            // return the other output to be checked below
            &p3_outputs[1 - pos]
        };
        ensure!(!p3_outputs.first().unwrap().is_final_output());
        ensure!(p3_output.to.unwrap() == 0);
        ensure!(p3_output.output == 4);
        ensure!(p3_output.from == 2);
        schedulers
            .get_mut(&0)
            .unwrap()
            .set_child_partition_output(p3_outputs.first().unwrap().clone())?;
        let p0_final_outputs = schedulers.get_mut(&0).unwrap().try_run_partition()?;
        ensure!(
            !p0_final_outputs.is_empty() && p0_final_outputs.first().unwrap().is_final_output()
        );
        ensure!(p0_final_outputs.first().unwrap().output == 16);
        ensure!(p0_final_outputs.first().unwrap().from == 0);
        ensure!(
            schedulers
                .get_mut(&0)
                .unwrap()
                .try_run_partition()?
                .is_empty()
        );
        ensure!(
            schedulers
                .get_mut(&1)
                .unwrap()
                .try_run_partition()?
                .is_empty()
        );
        ensure!(
            schedulers
                .get_mut(&2)
                .unwrap()
                .try_run_partition()?
                .is_empty()
        );
        Ok(())
    }
}
