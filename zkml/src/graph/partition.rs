use std::{
    collections::{BTreeMap, BTreeSet, HashMap, HashSet},
    fmt::Debug,
    hash::Hash,
};

use anyhow::{Context, bail, ensure};
use petgraph::{Direction, visit::EdgeRef};

use super::{
    Colored, Edge, Graph, GraphNode, NodeIdx, RunnableGraph, executor::Executor,
    scheduler::GraphScheduler,
};

/// A partition of graph contains the subgraph whose nodes all share the same color.
/// The idea is that one worker on a network will receive multiple partitions and will execute them
/// one by one. Each time receiving inputs to drive a new partition to completion and each time outputting
/// outputs to send to other workers/partitions such that the whole graph can be executed to completion.
#[derive(Debug, Clone)]
pub struct Partition<N: GraphNode, C> {
    /// The color of the partition.
    color: C,
    /// A partition is still a colored graph. The option is only to allow consuming the graph
    /// inside the partition scheduler without cloning.
    graph: Option<RunnableGraph<N, C>>,
    /// When a partition is done, its output needs to be sent to a parent partition if any.
    parent_partition: Option<C>,
    /// The "parent partition" when it receives all the outputs of its children partitions, it must
    /// give them in the right order to the scheduler. Since child partitions outputs may come at any time,
    /// this field is used to determine the ordering. The C is the color of the child partition - since the
    /// partition is made in such a way that no partitions of the same color are connected to each other
    /// (that wouldn't be a partition anymore).
    child_partition: BTreeSet<C>,

    /// A partition have a list of inputs if they're a "source" partition, i.e. a partition which contains
    /// nodes that have no predecessors. If not, the vector is simply empty.
    inputs: Vec<N::IO>,
}

impl<N: GraphNode, C> Partition<N, C>
where
    C: Ord + Clone,
{
    pub fn new(
        color: C,
        graph: RunnableGraph<N, C>,
        child_partition: BTreeSet<C>,
        inputs: Vec<N::IO>,
        parent_partition: Option<C>,
    ) -> anyhow::Result<Self> {
        ensure!(
            graph.output_nodes().len() == 1,
            "graph should have exactly one output node"
        );
        if let Some(ref parent_color) = parent_partition {
            ensure!(
                parent_color != &color,
                "parent partition should not be the same as the current partition"
            );
        }
        ensure!(
            !child_partition.contains(&color),
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
            parent_partition,
        })
    }
    pub fn is_source_partition(&self) -> bool {
        !self.inputs.is_empty()
    }
}

/// The output of a partition that must be given to a parent partition.
#[derive(Debug, Clone)]
pub struct PartitionOutput<N: GraphNode, C> {
    /// The color of the partition that generated this output.
    from: C,
    /// The color of the partition that must receive this output.
    /// If None, it means the output is a final output of the graph.
    to: Option<C>,
    /// The output of the partition. Given we are forcing the graph to be partitioned such that there
    /// is only maximum one edge between each pair of partition, there is only one output possible for each partition.
    /// However, note that a partition can have multiple _inputs_ if connected to multiple children partitions.
    output: N::IO,
}

impl<N: GraphNode, C> PartitionOutput<N, C> {
    pub fn is_final_output(&self) -> bool {
        self.to.is_none()
    }
}

/// A scheduler that is able to run multiple disjoint partitions of the same graph.
/// This is useful for distributed execution where a node can execute some tasks at the beginning,
/// and then get some other tasks later from the graph that are disconnected from the first tasks.
/// For example, it allows multiple round trips during the distributed execution of the graph.
pub struct PartitionScheduler<N: GraphNode, C, E: Executor<N, C>> {
    /// The different partitions of the graph who share the same color
    /// There is always only one partition that is "active" at a time.
    /// Indeed, if we could run multiple partitions at the same time, then we wouldn't need
    /// to have two partitions in the first place, because they could be aggregated into one
    /// single graph with all nodes sharing the same color.
    partitions: Vec<Partition<N, C>>,
    /// The config of the executor to run each partition individually.
    /// NOTE: it's a design decision to have an executor inside the scheduler since a partition
    /// is meant to be fully self-contained and executable. There is no need to expose to the API
    /// the internal nodes of the partition, as the graph scheduler would do. An outside executor
    /// is still necessary to run all the partitions of the graph to completion.
    executor_config: E::Config,
    /// The child outputs that are pending to be received.
    pending_child_outputs: HashMap<C, N::IO>,
    /// The context of the executor to run each partition individually.
    context: N::Context,
}

impl<N, C, E> PartitionScheduler<N, C, E>
where
    N: GraphNode + Clone,
    C: PartialEq + Eq + Clone + Hash + Ord + Debug,
    E: Executor<N, C>,
{
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
                .all(|p| p.graph.as_ref().unwrap().output_nodes().len() == 1),
            "All partitions must have exactly one output node"
        );
        Ok(Self {
            partitions,
            executor_config,
            pending_child_outputs: HashMap::new(),
            context,
        })
    }
    /// if the current partition has no inputs, it means it is a "source" partition, i.e. a partition which contains
    /// only links to other child partitions. In this case, there is nothing to run until we fetch the outputs of the
    /// child partitions.
    /// if the current partition has inputs, it means it is a "sink" partition, i.e. a partition which contains
    /// actual graph input data. In this case, we need to run the partition and return the ready nodes.
    /// If the output is None, that means it needs to wait for other partitions to send their outputs.
    pub fn try_run_partition(&mut self) -> anyhow::Result<Option<PartitionOutput<N, C>>> {
        if self.partitions.is_empty() {
            return Ok(None);
        }
        let next_partition = self.partitions.get_mut(0);
        let inputs = match next_partition {
            Some(part) => {
                if !part.is_source_partition() {
                    // the next partition expects outputs from its child partitions (e.g. it has no graph data input).
                    // we need to check if all the child outputs have been received.
                    let all_present = part
                        .child_partition
                        .iter()
                        .all(|k| self.pending_child_outputs.contains_key(k));
                    if !all_present {
                        None
                    } else {
                        Some(
                            part.child_partition
                                .iter()
                                .map(|c| self.pending_child_outputs.remove(c).unwrap())
                                .collect(),
                        )
                    }
                } else {
                    // otherwise, the partition is a source partition, i.e. a partition that doesn't have
                    // any child partitions so we just read their inputs.
                    Some(part.inputs.drain(..).collect())
                }
            }
            None => unreachable!("partition should not be empty - precheck passed"),
        };
        match inputs {
            // nothing to do for now
            None => Ok(None),
            // we either are running the sink partition or any parent partition who has received all its inputs from other partitions.
            Some(inputs) => {
                let mut partition = self.partitions.remove(0);
                let scheduler = GraphScheduler::new(partition.graph.take().unwrap());
                let mut outputs = E::run(&self.executor_config, scheduler, inputs, &self.context)?;
                ensure!(
                    outputs.len() == 1,
                    "Expected exactly one output for each partition"
                );
                let partition_output = PartitionOutput {
                    output: outputs.remove(0),
                    from: partition.color,
                    to: partition.parent_partition,
                };
                Ok(Some(partition_output))
            }
        }
    }

    pub fn set_child_partition_output(
        &mut self,
        output: PartitionOutput<N, C>,
    ) -> anyhow::Result<()> {
        if self.partitions.is_empty() {
            return Ok(());
        }
        let next_partition = self.partitions.first().unwrap();
        if next_partition.child_partition.contains(&output.from) {
            // we know the output is expected so we save it internally, and it'll be used at the next run if
            // all outputs of all child partitions have been received.
            self.pending_child_outputs
                .insert(output.from, output.output);
        } else {
            bail!(
                "output of child partition {:?} not expected for current partition {:?}",
                output.from,
                next_partition.color
            );
        }
        Ok(())
    }

    pub fn is_done(&self) -> bool {
        self.partitions.is_empty()
    }
}

impl<N, C> RunnableGraph<N, C>
where
    C: PartialEq + Eq + Clone + Hash + Ord + Debug,
    N: GraphNode + Clone + Debug,
    N::IO: Clone,
{
    pub fn partition_by_color(
        &self,
        inputs: Vec<N::IO>,
    ) -> anyhow::Result<HashMap<C, Vec<Partition<N, C>>>> {
        self.partition_by(|node| node.color(), inputs)
    }
    /// Extract color-connected subgraphs.
    /// Returns, for each partition, a fresh Graph containing just that partition.
    pub fn partition_by(
        &self,
        node_color: impl Fn(&Colored<N, C>) -> &C,
        inputs: Vec<N::IO>,
    ) -> anyhow::Result<HashMap<C, Vec<Partition<N, C>>>> {
        let mut visited = HashSet::new();
        // for each color, we keep a list of its partitions:
        // first element is the vector graphs for all partitions
        // second element is the associated mapping original_graph_index => new_partition_index
        let mut map = BTreeMap::<C, Vec<(RunnableGraph<N, C>, HashMap<NodeIdx, NodeIdx>)>>::new();

        // We start itearting from the input nodes of the original graph, so we create the partitions "in order",
        // starting from the lower partitions to the higher ones as this is the order of the execution of the graph.
        let indices = self
            .input_nodes
            .keys()
            .cloned()
            .chain(self.graph.node_indices())
            .collect::<BTreeSet<_>>();
        for node in indices.into_iter() {
            if visited.contains(&node) {
                continue;
            }
            let color = node_color(&self.graph[node]);

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
                for (_, neighbor, _) in self.neighbors(n) {
                    // there is a directly connected node sharing the same color, so it's part
                    // of the same partition.
                    if node_color(&self.graph[neighbor]) == color {
                        stack.push(neighbor);
                    }
                }
            }

            // Build a new Graph from this partition
            let mut sub = Graph::new();
            let mut local_map = HashMap::new();
            // add all nodes to the graph
            for &n in &partition {
                // we put empty edges for now since not all nodes in the partition have been added to the graph,
                // we don't know yet their new index in the partition and therefore can't create all edges yet.
                let new_idx = sub.add_node(self.graph[n].clone(), vec![]);
                local_map.insert(n, new_idx);
            }
            // add all edges inside the partition
            for &n in &partition {
                // we only add incoming edges - since eventually we go over all nodes of the graph, then we should
                // have covered all the edges
                for edge in self.graph.edges_directed(n, Direction::Incoming) {
                    let source_idx = edge.source();
                    // if the source is in the same partition, then we add the edge
                    if local_map.contains_key(&source_idx) {
                        let new_source_idx = local_map[&source_idx];
                        let new_target_idx = local_map[&n];
                        sub.add_edge(new_target_idx, Edge::Pred(new_source_idx, None));
                    }
                }
            }

            map.entry(color.clone()).or_default().push((sub, local_map));
        }

        let graph_root = self.output_nodes();
        ensure!(
            graph_root.len() == 1,
            "graph should have exactly one output node"
        );
        let graph_root = graph_root.first().unwrap();
        // At this point all the subgraphs have been built, but there some information missing:
        //  - the links between the subgraphs, we need to extract which color partition depends on which other color partition.
        //  - and then from it create the input edge on sink partitions: one node in each "parent" partition must now become an
        //    input node in that partition to receive the output of the children partitions. The order is important here.
        //  - adapting the inputs of source partitions such that their indices match (now that elements in the input vector could be
        //    dispatched to different partitions))
        //  - Finding and setting the parent partition for each partition.
        map.into_iter()
            .map(|(color, partitions)| {
                let mut final_partitions = Vec::with_capacity(partitions.len());
                for (mut subgraph, map) in partitions.into_iter() {
                    let mut child_partition_colors = BTreeSet::new();
                    let mut flattened_partition_inputs = Vec::new();
                    // the input nodes in the new partition that should receive input data
                    // we need to take the _same_ order of the inputs - given graph has a HashMap we take the ordering of the graph
                    // TODO: maybe just turn the graph hashmap into a btreemap directly ?
                    let partition_input_nodes = self
                        .graph
                        .node_indices()
                        // check they're in the current partition
                        .filter(|idx| map.contains_key(idx))
                        // check they're registered as input nodes on the original graph
                        .filter_map(|idx| self.input_nodes.get(&idx).map(|indices| (idx, indices)))
                        .collect::<BTreeMap<_, _>>();
                    // 2 cases: either
                    // 1. we are in a source partition, and we need to adjust the graph input data edges (maybe this partition only has one input
                    // while the original graph had 3, the rest of the input nodes are in different partitions, so we need to adjust the indices of
                    // the input data edges)
                    // 2. we are in any parent partition, and we need to _create_ the graph input data edges
                    if partition_input_nodes.is_empty() {
                        // we are in a parent partition, so we need to manually add the input edges
                        let mut source_nodes: Vec<_> = subgraph
                            .graph
                            .node_indices()
                            .filter(|idx| {
                                subgraph
                                    .graph
                                    .edges_directed(*idx, Direction::Incoming)
                                    .count()
                                    == 0
                            })
                            .collect();
                        ensure!(
                            source_nodes.len() == 1,
                            "INVALID GRAPH: a parent partition should have exactly one source node"
                        );
                        let source_node = source_nodes.remove(0);
                        let og_source_node = map
                            .iter()
                            .find(|(_, v)| **v == source_node)
                            .map(|(k, _)| *k)
                            .context("graph_map should contain all nodes")?;
                        // now search the incoming edges of the source node in the original graph. For each edge, we add that info to the new partition
                        for edge in self
                            .graph
                            .edges_directed(og_source_node, Direction::Incoming)
                            .enumerate()
                        {
                            // we put the _position_ of the edge in the subgraph
                            subgraph.add_edge(source_node, Edge::Input(edge.0));
                            // and we keep track of the order of the colors
                            let og_edge_color = self.graph[edge.1.source()].color().clone();
                            child_partition_colors.insert(og_edge_color);
                        }
                    } else {
                        // we are in a source partition, so we need to adjust the input data indices
                        for (og_node_idx, input_indices) in partition_input_nodes.into_iter() {
                            let offset = flattened_partition_inputs.len();
                            let new_idx = map
                                .get(&og_node_idx)
                                .context("graph_map should contain all nodes")?;
                            // we also need to add the edges to the subgraph  - all inputs for this partition are concatenated together
                            // TODO: remove the clone and use default values instead?
                            flattened_partition_inputs
                                .extend(input_indices.iter().map(|idx| inputs[*idx].clone()));
                            for i in 0..input_indices.len() {
                                subgraph.add_edge(*new_idx, Edge::Input(offset + i));
                            }
                        }
                    }

                    // now we want to find the parent partition for this partition such that its output can be sent to it.
                    let og_partition_root = *subgraph
                        .output_nodes()
                        .first()
                        .context("graph should have at least one output node")?;
                    let partition_root = map
                        .iter()
                        .find(|(_, v)| **v == og_partition_root)
                        .map(|(k, _)| *k)
                        .context("graph_map should contain all nodes")?;
                    let parent_node = self
                        .graph
                        .edges_directed(partition_root, Direction::Outgoing)
                        .next()
                        .map(|e| e.target());

                    let parent_partition = if *graph_root != partition_root {
                        ensure!(
                            parent_node.is_some(),
                            "any non root partition should have one parent partition"
                        );
                        let parent_node = parent_node.as_ref().unwrap();
                        let parent_color = self.graph[*parent_node].color().clone();
                        Some(parent_color)
                    } else {
                        None
                    };

                    final_partitions.push(Partition::<N, C>::new(
                        color.clone(),
                        subgraph,
                        child_partition_colors,
                        flattened_partition_inputs,
                        parent_partition,
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
    use crate::graph::{
        NodeIdx,
        executor::{SequentialExecutor, tests::MathAST},
    };

    use super::*;

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
    fn create_graph() -> (RunnableGraph<MathAST, usize>, NodeIdx) {
        let mut graph = Graph::new();
        // first partition
        let add1 = graph.add_node(
            Colored::new(MathAST::Add, 0),
            vec![Edge::Input(0), Edge::Input(1)],
        );
        let mul1 = graph.add_node(
            Colored::new(MathAST::Sub, 0),
            vec![Edge::Input(4), Edge::Input(5)],
        );
        let agg1 = graph.add_node(
            Colored::new(MathAST::Div, 0),
            vec![Edge::Pred(add1, None), Edge::Pred(mul1, None)],
        );
        // second partition
        let add2 = graph.add_node(
            Colored::new(MathAST::Add, 1),
            vec![Edge::Input(2), Edge::Input(3)],
        );
        let mul2 = graph.add_node(
            Colored::new(MathAST::Sub, 1),
            vec![Edge::Input(6), Edge::Input(7)],
        );
        let agg2 = graph.add_node(
            Colored::new(MathAST::Div, 1),
            vec![Edge::Pred(add2, None), Edge::Pred(mul2, None)],
        );
        // third partition
        let agg3 = graph.add_node(
            Colored::new(MathAST::Sub, 2),
            vec![Edge::Pred(agg1, None), Edge::Pred(agg2, None)],
        );
        let agg33 = graph.add_node(Colored::new(MathAST::Pow2, 2), vec![Edge::Pred(agg3, None)]);
        let pow1 = graph.add_node(
            Colored::new(MathAST::Pow2, 0),
            vec![Edge::Pred(agg33, None)],
        );
        (graph, pow1)
    }

    #[test]
    fn test_partition_by_color() {
        let (graph, agg33) = create_graph();
        assert_eq!(graph.output_nodes(), vec![agg33]);
        let partitions = graph
            .partition_by_color(vec![1, 2, 3, 4, 5, 6, 7, 8])
            .unwrap();
        assert_eq!(partitions.len(), 3);
        assert_eq!(partitions.get(&0).unwrap().len(), 2);
        assert_eq!(partitions.get(&1).unwrap().len(), 1);
        assert_eq!(partitions.get(&2).unwrap().len(), 1);
        assert_eq!(partitions.get(&0).unwrap()[0].inputs, vec![1, 2, 5, 6]);
        assert_eq!(partitions.get(&1).unwrap()[0].inputs, vec![3, 4, 7, 8]);
        assert_eq!(partitions.get(&2).unwrap()[0].inputs.len(), 0);
        assert_eq!(partitions.get(&2).unwrap()[0].child_partition.len(), 2);
        assert_eq!(
            partitions.get(&2).unwrap()[0].child_partition,
            BTreeSet::from([0, 1])
        );
        assert_eq!(partitions.get(&0).unwrap()[0].parent_partition, Some(2));
        assert_eq!(partitions.get(&1).unwrap()[0].parent_partition, Some(2));
        assert_eq!(partitions.get(&2).unwrap()[0].parent_partition, Some(0));
        assert_eq!(partitions.get(&0).unwrap()[1].parent_partition, None);
    }

    /// A simple test to check the different partition schedulers can drive the graph to completion.
    /// There is no implementation of a local partition executor since that would be pointless, as the
    /// only reason to have a partition is to run it in different machines.
    #[test]
    fn test_partition_scheduler() -> anyhow::Result<()> {
        let (graph, _agg33) = create_graph();
        // add1[0,1] = 1+7
        // sub1[4,5] = 4-2
        // add2[2,3] = 3+4
        // sub2[6,7] = 6-3
        // agg1 = add1 / sub1 = 8 / 2 = 4
        // agg2 = add2 / sub2 = 7 / 3 = 2
        // agg3 = agg1 - agg2 = 4 - 2 = 2
        // agg33 = pow2(agg3) = 2^2 = 4
        // final output = pow1 = pow2(agg33) = 4^2 = 16
        let partitions = graph.partition_by_color(vec![1, 7, 3, 4, 4, 2, 6, 3])?;
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
        ensure!(p1_outputs.is_some() && p1_outputs.as_ref().unwrap().to.unwrap() == 2);
        let p2_outputs = schedulers.get_mut(&1).unwrap().try_run_partition()?;
        ensure!(p2_outputs.is_some() && p2_outputs.as_ref().unwrap().to.unwrap() == 2);
        schedulers
            .get_mut(&2)
            .unwrap()
            .set_child_partition_output(p1_outputs.unwrap())?;
        // there should not be any computation possible on partition 2 since it doesn't have all its inputs
        ensure!(
            schedulers
                .get_mut(&2)
                .unwrap()
                .try_run_partition()?
                .is_none()
        );
        schedulers
            .get_mut(&2)
            .unwrap()
            .set_child_partition_output(p2_outputs.unwrap())?;
        let p3_outputs = schedulers.get_mut(&2).unwrap().try_run_partition()?;
        // goes back to partition 0
        ensure!(
            p3_outputs.is_some()
                && p3_outputs.as_ref().unwrap().to.is_some()
                && p3_outputs.as_ref().unwrap().to.unwrap() == 0
        );
        ensure!(p3_outputs.as_ref().unwrap().output == 4);
        ensure!(p3_outputs.as_ref().unwrap().from == 2);
        ensure!(!p3_outputs.as_ref().unwrap().is_final_output());
        schedulers
            .get_mut(&0)
            .unwrap()
            .set_child_partition_output(p3_outputs.unwrap())?;
        let p0_final_outputs = schedulers.get_mut(&0).unwrap().try_run_partition()?;
        ensure!(p0_final_outputs.is_some() && p0_final_outputs.as_ref().unwrap().is_final_output());
        ensure!(p0_final_outputs.as_ref().unwrap().output == 16);
        ensure!(p0_final_outputs.as_ref().unwrap().from == 0);
        ensure!(
            schedulers
                .get_mut(&0)
                .unwrap()
                .try_run_partition()?
                .is_none()
        );
        ensure!(
            schedulers
                .get_mut(&1)
                .unwrap()
                .try_run_partition()?
                .is_none()
        );
        ensure!(
            schedulers
                .get_mut(&2)
                .unwrap()
                .try_run_partition()?
                .is_none()
        );
        Ok(())
    }
}
