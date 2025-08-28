//! This module contains the logic to represents any computation as a graph
//! as well as the scheduler and executor to get the output of the computation.
//! This graph module is used to represent the proving logic of the IOP such that nodes
//! in this graph can be executed in parallel and even over a network.
use petgraph::{Direction, visit::EdgeRef};
use std::collections::HashMap;

pub use petgraph::graph::{DiGraph, NodeIndex as NodeIdx};

pub mod executor;
pub mod partition;
pub mod scheduler;

/// A trait for operations that can be executed on a graph.
/// It is used to define the input and output types of the operation.
/// It is also used to define the run method that will be used to execute the operation.
pub trait GraphNode {
    /// The input and output type for the node
    type IO: Clone;
    /// The context necessary for the node to execute the operation. This method is meant to be
    /// called either locally or on remote worker. The context
    /// can hold references to the setup parameters that we don't want to send over the wire.
    type Context;
    /// A description of the node, helpful for debugging and logging purposes.
    fn describe(&self) -> String;
    /// Runs the operation with the given context and inputs.
    /// The inputs comes from the graph processing (output of predecessor nodes).
    fn run(&self, ctx: &Self::Context, inputs: Vec<Self::IO>) -> anyhow::Result<Self::IO>;
}

/// Basic structure that contains a graph and a list of input nodes.
/// The graph is colored, e.g. each node is associated with a color that corresponds to which machine or thread etc
/// it should be executed on.
/// NOTE: need to support the more general case where an output node can be used both as output of the graph
/// and as input to another node of the graph
/// NOTE: need to support the strict ordering of inputs when a node expects both inputs from its predecessors AND
/// input from the graph input data.
#[derive(Debug, Clone)]
pub struct Graph<N, E> {
    pub(crate) graph: DiGraph<N, E>,
    /// maps the indices of the input vector to which node they should be given to.
    pub(crate) input_nodes: HashMap<NodeIdx, Vec<usize>>,
}

/// Edge represents the type of connection a new node has to the graph.
/// E is the generic type of the edge, e.g. an edge can contain some data.
#[derive(Debug, Clone)]
pub enum Edge<E> {
    /// Signals the node designated by the index is a predecessor of the node we're adding.
    Pred(NodeIdx, E),
    /// Index in the input data vector given to the scheduler that
    /// should be fed to the node
    Input(usize),
}

impl<N, E> Default for Graph<N, E> {
    fn default() -> Self {
        Self {
            graph: DiGraph::new(),
            input_nodes: Default::default(),
        }
    }
}

impl<N, E> Graph<N, E> {
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a node to the graph.
    /// If the node has no predecessors, it is considered an input node.
    /// For every predecessors, it creates the edge with no data at the moment.
    pub fn add_node(&mut self, node: N, edges: Vec<Edge<E>>) -> NodeIdx {
        let nidx = self.graph.add_node(node);
        for edge in edges {
            self.add_edge(nidx, edge);
        }
        nidx
    }
    fn add_edge(&mut self, nidx: NodeIdx, edge: Edge<E>) {
        match edge {
            Edge::Pred(pred, data) => {
                // currently, there is no data so we just set None - when the predecessors have been executed,
                // the edge will be updated to contain the output
                self.graph.add_edge(pred, nidx, data);
            }
            Edge::Input(idx) => {
                self.input_nodes.entry(nidx).or_default().push(idx);
            }
        }
    }
    pub fn output_nodes(&self) -> Vec<NodeIdx> {
        self.graph
            .node_indices()
            .filter(|idx| {
                self.graph
                    .neighbors_directed(*idx, Direction::Outgoing)
                    .count()
                    == 0
            })
            .collect()
    }

    /// Returns the edges of a node that starts at `node` and goes in the direction `direction`.
    pub fn edges<'a>(
        &'a self,
        node: NodeIdx,
        direction: Direction,
    ) -> impl Iterator<Item = (NodeIdx, &'a E)> + use<'a, N, E> {
        self.graph
            .edges_directed(node, direction)
            .map(move |e| match direction {
                Direction::Outgoing => (e.target(), e.weight()),
                Direction::Incoming => (e.source(), e.weight()),
            })
    }

    /// Returns all the neighbors of a node with the direction starting from `node`.
    /// e.g. (10,Direction::Incoming) means that the node 10 is a predecessor of `node`, there is
    /// a link 10 -> node
    /// (reason for this method to exist is because petgraph only returns outgoing nodes by default)
    pub fn neighbors<'a>(
        &'a self,
        node: NodeIdx,
    ) -> impl Iterator<Item = (Direction, NodeIdx, &'a E)> + use<'a, N, E> {
        self.graph
            .edges_directed(node, Direction::Outgoing)
            .map(|e| (Direction::Outgoing, e.target(), e.weight()))
            .chain(
                self.graph
                    .edges_directed(node, Direction::Incoming)
                    .map(|e| (Direction::Incoming, e.source(), e.weight())),
            )
    }
}

/// A proving node that is also colored with a color.
/// The colors partition the graph such that each color
/// holds a set of tasks that can be executed on the same machine,
/// or same thread, etc.
#[derive(Debug, Clone)]
pub struct Colored<N, C> {
    node: N,
    color: C,
}

impl<N, C> Colored<N, C> {
    pub fn new(proving_node: N, color: C) -> Self {
        Self {
            node: proving_node,
            color,
        }
    }
    pub fn color(&self) -> &C {
        &self.color
    }
}

/// Wrapper implementation such that a colored node is also a graph node - it just delegates
/// all calls to the underlying type.
impl<N: GraphNode, C> GraphNode for Colored<N, C> {
    type IO = N::IO;
    type Context = N::Context;
    fn run(&self, ctx: &Self::Context, input: Vec<Self::IO>) -> anyhow::Result<Self::IO> {
        self.node.run(ctx, input)
    }

    fn describe(&self) -> String {
        self.node.describe()
    }
}

type RunnableGraph<N, C> = Graph<Colored<N, C>, Option<<N as GraphNode>::IO>>;

/// Helper macro to extract a variant from a vector of enums.
/// This is useful for nodes of the graph which are variants, so when one needs to extract
/// the inputs from a vector of enums, it can do so via this macro.
#[allow(unused_macros)]
macro_rules! try_extract_variant_vec {
    // case: variant with payload
    ($variant:ident :: $name:ident ( $inner:ident ), $vec:expr) => {{
        let mut out: Vec<$inner> = Vec::with_capacity($vec.len());
        let mut err_i: usize = usize::MAX;
        for (i, e) in $vec.into_iter().enumerate() {
            match e {
                $variant::$name(inner) => out.push(inner),
                _ => {
                    println!("Type mismatch {:?}", e);
                    err_i = i;
                    break;
                }
            }
        }
        if err_i != usize::MAX {
            Err(anyhow::anyhow!("Type mismatch at index {}", err_i))
        } else {
            Ok(out)
        }
    }};
    // case: variant without payload
    ($variant:ident :: $name:ident, $vec:expr) => {{
        let mut out: Vec<()> = Vec::with_capacity($vec.len());
        let mut err_i: usize = usize::MAX;
        for (i, e) in $vec.into_iter().enumerate() {
            match e {
                $variant::$name => out.push(()),
                _ => {
                    println!("Type mismatch {:?}", e);
                    err_i = i;
                    break;
                }
            }
        }
        if err_i != usize::MAX {
            Err(anyhow::anyhow!("Type mismatch at index {}", err_i))
        } else {
            Ok(out)
        }
    }};
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, Clone)]
    pub enum TestOperation {
        Test1,
        Test2,
    }

    impl GraphNode for TestOperation {
        type IO = String;
        type Context = ();
        fn describe(&self) -> String {
            match self {
                TestOperation::Test1 => "Test1".to_string(),
                TestOperation::Test2 => "Test2".to_string(),
            }
        }
        fn run(&self, _ctx: &Self::Context, input: Vec<Self::IO>) -> anyhow::Result<Self::IO> {
            match self {
                TestOperation::Test1 => {
                    println!("Test1: {input:?}");
                    Ok(format!("Test1: {input:?}"))
                }
                TestOperation::Test2 => {
                    println!("Test2: {input:?}");
                    Ok(format!("Test2: {input:?}"))
                }
            }
        }
    }

    /// Color each node with a color. It makes it so that each color
    /// is assigned related tasks that can be executed on the same machine easily.
    /// TODO: Dumb implementation - make it aware of each node dependencies
    pub fn instantiate(num_colors: usize) -> RunnableGraph<TestOperation, usize> {
        let mut graph = Graph::new();
        let mut color = 0;
        let test1_node = graph.add_node(
            Colored::new(TestOperation::Test1, color),
            vec![Edge::Input(0)],
        );
        color = (color + 1) % num_colors;
        let _test2_node = graph.add_node(
            Colored::new(TestOperation::Test2, color),
            vec![Edge::Pred(test1_node, None)],
        );
        graph
    }

    #[test]
    fn test_try_extract_variant_vec() {
        #[derive(Debug)]
        enum MyEnum {
            Variant1(i32),
            Variant2(f64),
        }

        let vec = vec![MyEnum::Variant1(1), MyEnum::Variant1(2)];
        let out = try_extract_variant_vec!(MyEnum::Variant1(i32), vec).unwrap();
        assert_eq!(out, vec![1, 2]);

        let vec = vec![MyEnum::Variant2(1.0), MyEnum::Variant2(2.0)];
        let out = try_extract_variant_vec!(MyEnum::Variant2(f64), vec).unwrap();
        assert_eq!(out, vec![1.0, 2.0]);

        let vec = vec![MyEnum::Variant1(1), MyEnum::Variant2(2.0)];
        assert!(try_extract_variant_vec!(MyEnum::Variant1(i32), vec).is_err());
    }
}
