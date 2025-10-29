/// ! The graph module implements a directed graph data structure with some additional features listed below.
///
/// - This graph is actually a port graph, meaning that each edge has a list of source and target ports that are
///   "connected" to each other. The invariant of a port graph is that all target ports on a given node are referenced
///   at most once by an input port on any other node.
/// - Edges can be weighted generically. For example, weight can store the output of a executable node
///   waiting to be picked up by its successor.
/// - It enforces that there is only one edge between two nodes.
/// - Nodes can be indexed by a custom type. This allows backwards compatibility with other graph implementations
///   like `petgraph` or `onnx`.
use anyhow::{Context, ensure};
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, BTreeSet, HashSet},
    fmt::Debug,
    hash::Hash,
    ops::Index,
    sync::atomic::{AtomicUsize, Ordering},
};

/// Counter to automatically generate indices for nodes.
static EDGE_INDEX_COUNTER: AtomicUsize = AtomicUsize::new(0);

/// The nodes the graph is built off.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum Node<L, I = usize, O = usize> {
    /// A node encoding the i-th input.
    Input(I),
    /// A internal node.
    Inner(L),
    /// A node encoding the i-th output.
    Output(O),
}
impl<L, I, O> Node<L, I, O> {
    pub fn is_input(&self) -> bool {
        matches!(self, Node::Input(_))
    }

    /// If this node is an input, returns its payload.
    pub fn as_input(&self) -> Option<&I> {
        match self {
            Node::Input(i) => Some(i),
            _ => None,
        }
    }

    pub fn is_inner(&self) -> bool {
        matches!(self, Node::Inner(_))
    }

    /// If this node carries an internal payload, returns a reference to it.
    pub fn as_inner(&self) -> Option<&L> {
        match self {
            Node::Inner(inner) => Some(inner),
            _ => None,
        }
    }

    /// If this node carries an internal payload, returns a mutable reference to it.
    pub fn as_inner_mut(&mut self) -> Option<&mut L> {
        match self {
            Node::Inner(inner) => Some(inner),
            _ => None,
        }
    }

    /// If this node carries an internal payload, consume it into its content.
    pub fn into_inner(self) -> Option<L> {
        match self {
            Node::Inner(inner) => Some(inner),
            _ => None,
        }
    }

    pub fn is_output(&self) -> bool {
        matches!(self, Node::Output(_))
    }

    /// If this node is an output, returns its position in the list of outputs as
    /// defined by the original model.
    pub fn as_output(&self) -> Option<&O> {
        match self {
            Node::Output(o) => Some(o),
            _ => None,
        }
    }
}

// Syntactic sugar for graphs using only inner nodes (e.g. partitions).
impl<L> Node<L, (), ()> {
    pub fn inner(&self) -> &L {
        self.as_inner().unwrap()
    }
}

#[derive(
    Copy,
    Clone,
    PartialEq,
    Eq,
    Hash,
    PartialOrd,
    Ord,
    derive_more::Debug,
    derive_more::Display,
    Serialize,
    Deserialize,
)]
#[debug("{node_id}{port}")]
#[display("{node_id}{port}")]
/// Uniquely identifies an input port of a given node.
pub struct NodeInput {
    /// The referenced node.
    pub node_id: NodeId,
    /// The concerned port of the referenced node.
    pub port: PortId,
}
impl NodeInput {
    pub fn new<N: Into<NodeId>, P: Into<PortId>>(node_id: N, port: P) -> Self {
        NodeInput {
            node_id: node_id.into(),
            port: port.into(),
        }
    }
}

#[derive(
    Copy,
    Clone,
    PartialEq,
    Eq,
    Hash,
    PartialOrd,
    Ord,
    derive_more::Display,
    derive_more::Debug,
    Serialize,
    Deserialize,
)]
#[debug("{node_id}{port}")]
#[display("{node_id}{port}")]
/// Uniquely identifies an output port of a given node.
pub struct NodeOutput {
    /// The referenced node.
    pub node_id: NodeId,
    /// The concerned port of the referenced node.
    pub port: PortId,
}
impl NodeOutput {
    pub fn new<N: Into<NodeId>, P: Into<PortId>>(node_id: N, port: P) -> Self {
        NodeOutput {
            node_id: node_id.into(),
            port: port.into(),
        }
    }
}

/// Uniquely identifies a link between an output port of a node and an input
/// port of another node.
///
/// This is used to uniquely identify a feed living on an edge between two
/// nodes.
#[derive(derive_more::Debug)]
#[debug("{source} -> {target}")]
pub struct Feed {
    /// The source port of the link between the two nodes.
    pub source: NodeOutput,
    /// The source port of the link between the two nodes.
    pub target: NodeInput,
}

/// Given an iterator over `T`s linked to a node output, order them by their
/// port number and strip the [`NodeOutput`] to obtain an iterator of `T`s
/// implicitly ordered by the port number their were attached to.
///
/// This is used to prepare data generated in non-specified order for use in
/// crypto code that expects vectors implicitly order by port number, typically
/// in backward graph traversal.
pub fn order_by_out_port<T, I: Iterator<Item = (NodeOutput, T)>>(i: I) -> impl Iterator<Item = T> {
    let mut outputs = i
        .map(|(node_out, x)| (node_out.port, x))
        .collect::<Vec<_>>();
    outputs.sort_by_key(|(port, _)| *port);
    outputs.into_iter().map(|(_, x)| x)
}

/// Given an iterator over `T`s linked to a node input, order them by their
/// port number and strip the [`NodeOutput`] to obtain an iterator of `T`s
/// implicitly ordered by the port number their were attached to.
///
/// This is used to prepare data generated in non-specified order for use in
/// crypto code that expects vectors implicitly order by port number, typically
/// in forward graph traversal.
pub fn order_by_in_port<T, I: Iterator<Item = (NodeInput, T)>>(i: I) -> impl Iterator<Item = T> {
    let mut outputs = i.map(|(node_in, x)| (node_in.port, x)).collect::<Vec<_>>();
    outputs.sort_by_key(|(port, _)| *port);
    outputs.into_iter().map(|(_, x)| x)
}

#[derive(Debug, Clone)]
pub enum Direction {
    Incoming,
    Outgoing,
    Any,
}

impl Direction {
    pub fn is_incoming(&self) -> bool {
        matches!(self, Direction::Incoming) || matches!(self, Direction::Any)
    }
    pub fn is_outgoing(&self) -> bool {
        matches!(self, Direction::Outgoing) || matches!(self, Direction::Any)
    }
    pub fn is_any(&self) -> bool {
        matches!(self, Direction::Any)
    }
}

/// Default unique node identifier.
#[derive(
    Copy,
    Clone,
    Hash,
    Serialize,
    Deserialize,
    PartialOrd,
    Ord,
    derive_more::From,
    derive_more::Into,
    derive_more::Display,
    derive_more::Debug,
    derive_more::Deref,
    PartialEq,
    Eq,
)]
#[display("N{_0}")]
#[debug("Node({_0})")]
pub struct NodeId(pub usize);
impl NodeId {
    /// Generate the [`NodeInput`] corresponding to the given port for this node.
    pub fn input_at<P: Into<PortId>>(&self, port: P) -> NodeInput {
        NodeInput::new(self.0, port)
    }

    /// Generate the [`NodeOutput`] corresponding to the given port for this node.
    pub fn output_at<P: Into<PortId>>(&self, port: P) -> NodeOutput {
        NodeOutput::new(self.0, port)
    }

    /// Consider this node as a model input and return its unique [`NodeOutput`].
    pub fn as_model_input(&self) -> NodeOutput {
        NodeOutput::new(self.0, 0)
    }

    /// Consider this node as a model output and return its unique [`NodeInput`].
    pub fn as_model_output(&self) -> NodeInput {
        NodeInput::new(self.0, 0)
    }
}

/// Unique identifier of an edge. The edge identifier is mostly useful to identify a particular edge and
/// remove it from the graph efficiently (e.g. without comparing the weights together since it can be expensive).
#[derive(
    Debug,
    Copy,
    Clone,
    Serialize,
    Deserialize,
    PartialEq,
    Eq,
    Ord,
    PartialOrd,
    Hash,
    derive_more::Display,
    derive_more::From,
    derive_more::Into,
)]
pub struct EdgeId(usize);

#[derive(
    Clone,
    Serialize,
    Deserialize,
    PartialEq,
    Eq,
    Copy,
    Hash,
    Ord,
    PartialOrd,
    derive_more::From,
    derive_more::Into,
    derive_more::Display,
    derive_more::Debug,
    derive_more::Deref,
)]
#[display("@{_0}")]
#[debug("Port({_0})")]
pub struct PortId(usize);

/// A port link is a link between an input port of a node and an output port of another node.
#[derive(
    Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash, Ord, PartialOrd, derive_more::Display,
)]
#[display("Link({source_port}->{target_port})")]
pub struct PortLink {
    pub source_port: PortId,
    pub target_port: PortId,
}
impl PortLink {
    pub fn new<I: Into<PortId>, I2: Into<PortId>>(source: I, target: I2) -> Self {
        Self {
            source_port: source.into(),
            target_port: target.into(),
        }
    }

    pub fn consecutive() -> Self {
        Self {
            source_port: PortId(0),
            target_port: PortId(0),
        }
    }
}

/// The weight inside the graph is a list of port links.
#[derive(
    Debug,
    Clone,
    Serialize,
    Deserialize,
    PartialEq,
    Eq,
    Default,
    Hash,
    derive_more::From,
    derive_more::Into,
)]
pub struct Ports(pub(crate) Vec<PortLink>);
impl Ports {
    pub fn new() -> Self {
        Self::default()
    }

    /// This returns a port link for two consecutive nodes assuming there is
    /// only a single output and input.
    pub fn consecutive() -> Self {
        Self(vec![PortLink {
            source_port: PortId(0),
            target_port: PortId(0),
        }])
    }

    pub fn sorted(self) -> Self {
        let mut ports = self.0;
        ports.sort_by_key(|p| p.source_port);
        Self(ports)
    }
}

/// An edge is a connection between a source and a target with a list of ports and a weight.
#[derive(derive_more::Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[debug("{source} -- [{ports:?}] --> {target}")]
pub struct Edge<W> {
    source: NodeId,
    target: NodeId,
    ports: Ports,
    /// An edge doesn't necessarily have a weight. Note only the weight is
    /// public since it's the only modifiable field from the user perspective.
    /// Ports shouldn't be allowed to be modified at will otherwise the
    /// invariant of the ports may be violated.
    pub weight: Option<W>,
}

impl<W> Edge<W> {
    pub fn new<P: Into<Ports>, S: Into<NodeId>, T: Into<NodeId>>(
        source: S,
        target: T,
        ports: P,
        weight: Option<W>,
    ) -> Self {
        Self {
            source: source.into(),
            target: target.into(),
            ports: ports.into().sorted(),
            weight,
        }
    }
    pub fn between_nodes<P: Into<Ports>>(
        source: NodeId,
        target: NodeId,
        ports: P,
        weight: Option<W>,
    ) -> Self {
        Self {
            source,
            target,
            ports: ports.into().sorted(),
            weight,
        }
    }
    pub fn is_incoming_to(&self, node_id: NodeId) -> bool {
        self.target == node_id
    }
    pub fn is_outgoing_from(&self, node_id: NodeId) -> bool {
        self.source == node_id
    }
    pub fn source(&self) -> NodeId {
        self.source
    }
    pub fn target(&self) -> NodeId {
        self.target
    }
    pub fn ports(&self) -> &Ports {
        &self.ports
    }
    pub fn feeds(&self) -> impl Iterator<Item = Feed> + use<'_, W> {
        self.ports.iter().map(|link| Feed {
            source: NodeOutput {
                node_id: self.source,
                port: link.source_port,
            },
            target: NodeInput {
                node_id: self.target,
                port: link.target_port,
            },
        })
    }

    /// Tries to find the other end of the edge given a node id.
    /// If the node id is the source, then the target is returned.
    /// If the node id is the target, then the source is returned.
    /// If the node id is not the source or the target, then None is returned.
    pub fn other_end(&self, node_id: NodeId) -> Option<NodeId> {
        if self.source == node_id {
            Some(self.target)
        } else if self.target == node_id {
            Some(self.source)
        } else {
            None
        }
    }
}

/// Basic structure that contains a graph and a list of input nodes.
/// The graph is colored, e.g. each node is associated with a color that
/// corresponds to which machine or thread etc it should be executed on.
/// NOTE: need to support the strict ordering of inputs when a node expects both
/// inputs from its predecessors AND input from the graph input data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Graph<N, I, O, W> {
    /// Nodes indexed by their index - we use a BTreeMap to make the graph
    /// iteration deterministic and sorted by increasing order of the node id,
    /// which is usually equivalent to the order of insertion.
    nodes: BTreeMap<NodeId, Node<N, I, O>>,
    /// Contains all the edges in the graph.
    //
    /// NOTE: currently O(n) to search but once API is stabilized, we can move
    /// to a multi key map indexed by both the source and the target node to
    /// search in O(1).
    pub(crate) edges: Vec<(EdgeId, Edge<W>)>,
}

impl<N, I, O, W> Graph<N, I, O, W>
where
    I: PartialEq + std::fmt::Debug,
    O: PartialEq + std::fmt::Debug,
    W: Clone,
{
    /// Create a new, empty graph.
    pub fn new() -> Self {
        Self::default()
    }

    /// Generate a [`NodeId`] that is not yet used in this graph. No
    /// sequentiality is guaranteed.
    ///
    /// NOTE: made `pub` so that it can be used by the
    /// [`GlobalCommitmentContext`] to generate a non-colliding table ID.
    pub fn next_node_id(&self) -> NodeId {
        self.nodes
            .keys()
            .map(|x| **x)
            .max()
            .map(|x| x + 1)
            .unwrap_or(0)
            .into()
    }

    /// Add a node to the graph. It will return the index of the added node.
    fn add_node(&mut self, node: Node<N, I, O>) -> anyhow::Result<NodeId> {
        let node_id = self.next_node_id();
        self.add_node_with_id(node_id, node).map(|_| node_id)
    }

    /// Create an inner node with the provided payload, and return its ID.
    pub fn add_inner(&mut self, x: N) -> anyhow::Result<NodeId> {
        self.add_node(Node::Inner(x))
    }

    /// Create an input node with the provided payload, and return its ID.
    pub fn add_input(&mut self, i: I) -> anyhow::Result<NodeId> {
        self.add_node(Node::Input(i))
    }

    /// Create an output node with the provided payload, and return its ID.
    pub fn add_output(&mut self, o: O) -> anyhow::Result<NodeId> {
        self.add_node(Node::Output(o))
    }

    /// Add a node to the graph with the given ID. Return an error if
    /// the ID is already assigned.
    pub fn add_node_with_id<II: Into<NodeId> + Clone>(
        &mut self,
        nidx: II,
        node: Node<N, I, O>,
    ) -> anyhow::Result<()> {
        let id = nidx.into();
        match self.nodes.insert(id, node) {
            None => Ok(()),
            Some(_) => anyhow::bail!("{id} already exists"),
        }
    }

    /// Wrapper method around adding an edge between two nodes.
    /// It returns the edge id of the added edge OR the modified edge if the weights
    /// have been accumulated.
    /// One can pass an individual edge or a vector of edges.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use zkml::graph::{Graph, Ports, PortLink};
    /// let mut graph: Graph<&str, (), (), ()> = Graph::new();
    /// let node1 = graph.add_inner("first").unwrap();
    /// let node2 = graph.add_inner("second").unwrap();
    ///
    /// // Simple consecutive connection
    /// let edge_id = graph.add_edge(node1, node2, Ports::consecutive(), Some(())).unwrap();
    ///
    /// // Custom port mapping using PortLink
    /// let node3 = graph.add_inner("third").unwrap();
    /// graph.add_edge(node2, node3, PortLink::new(0, 0), None).unwrap();
    ///
    /// // Or using a (usize, usize) tuple directly
    /// let node4 = graph.add_inner("fourth").unwrap();
    /// graph.add_edge(node3, node4, (0, 0), None).unwrap();
    /// ```
    pub fn add_edge<P: Into<Ports>, WO: Into<Option<W>>>(
        &mut self,
        source: NodeId,
        target: NodeId,
        ports: P,
        weight: WO,
    ) -> anyhow::Result<EdgeId> {
        let edge = Edge::new(source, target, ports, weight.into());
        Ok(self.add_edges_raw(vec![edge])?[0])
    }

    /// Wrapper method around adding a consecutive edge between two nodes.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use zkml::graph::Graph;
    /// let mut graph: Graph<&str, (), (), ()> = Graph::new();
    /// let node1 = graph.add_inner("first").unwrap();
    /// let node2 = graph.add_inner("second").unwrap();
    ///
    /// let edge_id = graph.add_consecutive_edge(node1, node2, Some(())).unwrap();
    /// assert_eq!(graph.neighbors(node1, zkml::graph::Direction::Outgoing).count(), 1);
    /// ```
    pub fn add_consecutive_edge<WO: Into<Option<W>>>(
        &mut self,
        source: NodeId,
        target: NodeId,
        weight: WO,
    ) -> anyhow::Result<EdgeId> {
        self.add_edge(source, target, Ports::consecutive(), weight)
    }

    /// Add a consecutive node to the graph.
    /// It adds a new node to the graph and connects it to the previous node with a consecutive edge.
    /// In this case, there is only one port link between the two nodes.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use zkml::graph::{Graph, Node};
    /// let mut graph: Graph<&str, (), (), ()> = Graph::new();
    /// let node1 = graph.add_inner("first").unwrap();
    ///
    /// // Add a node connected to an existing node
    /// let node2 = graph.add_consecutive_node(Node::Inner("second"), node1, Some(())).unwrap();
    /// assert_eq!(graph.neighbors(node1, zkml::graph::Direction::Outgoing).count(), 1);
    /// ```
    pub fn add_consecutive_node<WO: Into<Option<W>>>(
        &mut self,
        node: Node<N, I, O>,
        previous_node_id: NodeId,
        weight: WO,
    ) -> anyhow::Result<NodeId> {
        let new_node_id = self.add_node(node)?;
        self.add_edge(previous_node_id, new_node_id, Ports::consecutive(), weight)?;
        Ok(new_node_id)
    }

    pub fn add_edges_raw(&mut self, edges: Vec<Edge<W>>) -> anyhow::Result<Vec<EdgeId>> {
        let mut edge_ids = Vec::with_capacity(edges.len());
        for new_edge in edges {
            ensure!(new_edge.source != new_edge.target, "idempotent edge");
            // making sure the source and the targets exists
            ensure!(
                self.nodes.contains_key(&new_edge.source),
                "Source node {} not found",
                new_edge.source
            );
            ensure!(
                self.nodes.contains_key(&new_edge.target),
                "Target node {} not found",
                new_edge.target
            );

            // compare with all other edges to see if
            // 1. there are duplicates
            // 2. the ports are consistent, e.g. no target port is used twice on the same node
            ensure!(
                !self.edges.iter().any(|(_, current_edge)| {
                    current_edge.source == new_edge.source && current_edge.target == new_edge.target
                }),
                "Edge between {:?} and {:?} already exists",
                new_edge.source,
                new_edge.target
            );
            // no need to detect duplicates on already existing edges since it's already checked
            self.check_consistency(
                new_edge.target,
                new_edge.ports.iter().map(|port| port.target_port),
            )?;
            let edge_id = next_edge_id();
            edge_ids.push(edge_id);
            self.edges.push((edge_id, new_edge));
        }
        Ok(edge_ids)
    }

    /// Removes an edge from the graph by its ID.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use zkml::graph::Graph;
    /// # use zkml::graph::Ports;
    /// let mut graph: Graph<&str, (), (), ()> = Graph::new();
    /// let node1 = graph.add_inner("first").unwrap();
    /// let node2 = graph.add_inner("second").unwrap();
    /// let edge_id = graph.add_edge(node1, node2, Ports::consecutive(), Some(())).unwrap();
    ///
    /// assert_eq!(graph.edges().count(), 1);
    /// graph.remove_edge(edge_id).unwrap();
    /// assert_eq!(graph.edges().count(), 0);
    /// ```
    pub fn remove_edge(&mut self, edge_id: EdgeId) -> anyhow::Result<()> {
        let curr_len = self.edges.len();
        self.edges.retain(|(id, _)| *id != edge_id);
        if self.edges.len() == curr_len {
            anyhow::bail!("Edge with id {edge_id:?} not found");
        }
        Ok(())
    }

    /// Transmute an inner node in place, applying `f` to its payload and
    /// replacing the current one with the result.
    pub fn replace_inner<F>(&mut self, node_id: NodeId, f: F) -> anyhow::Result<()>
    where
        F: FnOnce(N) -> N,
    {
        let old_node = self
            .nodes
            .remove(&node_id)
            .context("Node not found")?
            .into_inner()
            .ok_or_else(|| anyhow::anyhow!("{node_id} is not an inner node"))?;
        self.nodes.insert(node_id, Node::Inner(f(old_node)));
        Ok(())
    }

    /// Returns a reference to the node with the given ID, if it exists.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use zkml::graph::{Graph, Node};
    /// let mut graph: Graph<&str, (), (), ()> = Graph::new();
    /// let node_id = graph.add_inner("test").unwrap();
    ///
    /// assert_eq!(graph.node(node_id).map(Node::inner), Some(&"test"));
    /// assert_eq!(graph.node(999.into()), None);
    /// ```
    pub fn node(&self, node_id: NodeId) -> Option<&Node<N, I, O>> {
        self.nodes.get(&node_id)
    }

    /// Returns a mutable reference to the node with the given ID, if it exists.
    pub fn node_mut(&mut self, node_id: NodeId) -> Option<&mut Node<N, I, O>> {
        self.nodes.get_mut(&node_id)
    }

    /// Returns the number of nodes in the graph.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use zkml::graph::Graph;
    /// let mut graph: Graph<&str, (), (), ()> = Graph::new();
    /// assert_eq!(graph.node_count(), 0);
    ///
    /// let node1 = graph.add_inner("first").unwrap();
    /// assert_eq!(graph.node_count(), 1);
    /// ```
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Returns an iterator over all nodes in the graph as (node_id, node_data) pairs.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use zkml::graph::Graph;
    /// let mut graph: Graph<&str, (), (), ()> = Graph::new();
    /// let node1 = graph.add_inner("first").unwrap();
    /// let node2 = graph.add_inner("second").unwrap();
    ///
    /// let nodes: Vec<_> = graph.nodes().collect();
    /// assert_eq!(nodes.len(), 2);
    /// ```
    pub fn nodes(&self) -> impl Iterator<Item = (&NodeId, &Node<N, I, O>)> + use<'_, N, I, O, W> {
        self.nodes.iter()
    }

    /// Return the topologigl source nodes of this graph, i.e. the nodes without
    /// any incoming edges.
    pub fn source_nodes(&self) -> impl Iterator<Item = NodeId> + use<'_, N, I, O, W> {
        self.nodes
            .keys()
            .copied()
            .filter(|node_id| self.incomings(*node_id).next().is_none())
    }

    /// Return the topologigl sink nodes of this graph, i.e. the nodes without
    /// any outgoing edges.
    pub fn sink_nodes(&self) -> impl Iterator<Item = NodeId> + use<'_, N, I, O, W> {
        self.nodes
            .keys()
            .copied()
            .filter(|node_id| self.outgoings(*node_id).next().is_none())
    }

    /// Return whether the provided node is a topological sink of the graph.
    pub fn is_sink(&self, node_id: NodeId) -> bool {
        self.outgoings(node_id).next().is_none()
    }

    /// Return whether the provided node is a topological source of the graph.
    pub fn is_source(&self, node_id: NodeId) -> bool {
        self.incomings(node_id).next().is_none()
    }

    /// Return the ID of and a reference to the first node satisfying the
    /// provided predicate.
    pub fn find_node(
        &self,
        p: impl Fn(NodeId, &Node<N, I, O>) -> bool,
    ) -> Option<(&NodeId, &Node<N, I, O>)> {
        self.nodes.iter().find(|(n_id, n)| p(**n_id, n))
    }

    /// Returns an iterator, in unspecified order, over this graph input nodes.
    pub fn input_nodes(&self) -> impl Iterator<Item = (NodeId, &I)> + use<'_, N, I, O, W> {
        self.nodes()
            .filter_map(|(n_id, n)| n.as_input().map(|i| (*n_id, i)))
    }

    /// Returns an iterator, in unspecified order, over this model output nodes.
    pub fn output_nodes(&self) -> impl Iterator<Item = (NodeId, &O)> + use<'_, N, I, O, W> {
        self.nodes()
            .filter_map(|(n_id, n)| n.as_output().map(|i| (*n_id, i)))
    }

    /// Returns an iterator, in unspecified order, over this graph inner nodes.
    pub fn inner_nodes(&self) -> impl Iterator<Item = (NodeId, &N)> {
        self.nodes()
            .filter_map(|(n_id, n)| n.as_inner().map(|n| (*n_id, n)))
    }

    /// Return, if it exists, the ID of the input node corresponding to the provided
    /// input ID.
    pub fn input_node_id(&self, input_id: I) -> anyhow::Result<NodeId> {
        self.find_node(|_, n| n.as_input().map(|i| *i == input_id).unwrap_or(false))
            .map(|(node_id, _)| *node_id)
            .ok_or_else(|| anyhow::anyhow!("fetching node ID for input {input_id:?}"))
    }

    /// Return, if it exists, the ID of the output node corresponding to the provided
    /// output ID.
    pub fn output_node_id(&self, output_id: O) -> anyhow::Result<NodeId> {
        self.find_node(|_, n| n.as_output().map(|o| *o == output_id).unwrap_or(false))
            .map(|(node_id, _)| *node_id)
            .ok_or_else(|| anyhow::anyhow!("fetching node ID for output {output_id:?}"))
    }

    /// Returns an iterator over all edges in the graph.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use zkml::graph::Graph;
    /// let mut graph: Graph<&str, (), (), ()> = Graph::new();
    /// let node1 = graph.add_inner("first").unwrap();
    /// let node2 = graph.add_inner("second").unwrap();
    /// let edge_id = graph.add_edge(node1, node2, zkml::graph::Ports::consecutive(), Some(())).unwrap();
    ///
    /// let edges: Vec<_> = graph.edges().collect();
    /// assert_eq!(edges.len(), 1);
    /// ```
    pub fn edges(&self) -> impl Iterator<Item = &Edge<W>> + use<'_, N, I, O, W> {
        self.edges.iter().map(|(_id, edge)| edge)
    }

    /// Checks that no two target_port is assigned twice amongst all edges that have the same target node.
    /// Checks that all ports are consecutive and fill the range (0..num_ports).
    fn check_consistency<II: IntoIterator<Item = PortId>>(
        &self,
        target_node: NodeId,
        new_ports: II,
    ) -> anyhow::Result<()> {
        let all_target_ports = self
            .edges
            .iter()
            .filter(move |(_, current_edge)| current_edge.target == target_node)
            .flat_map(|(_, current_edge)| current_edge.ports.0.iter())
            .map(|port| port.target_port)
            .collect::<Vec<_>>();
        let mut set = all_target_ports
            .into_iter()
            .try_fold(HashSet::new(), |mut acc, tport| match acc.insert(tport) {
                // detect if there was duplicate since now one node is modified
                true => Some(acc),
                false => None,
            })
            .ok_or_else(|| anyhow::anyhow!("Ports are not consistent"))?;
        ensure!(
            new_ports.into_iter().all(|port| set.insert(port)),
            "Ports already used"
        );
        let len = set.len();
        ensure!(
            (0..len).all(|i| set.contains(&PortId(i))),
            "Ports are not consecutive: {:?}",
            set
        );
        Ok(())
    }

    pub fn edge<'a>(&'a self, edge_id: &EdgeId) -> Option<&'a Edge<W>> {
        self.edges
            .iter()
            .find(|(id, _)| id == edge_id)
            .map(|(_, edge)| edge)
    }

    /// Returns the edges touching the provided node, in `direction`.
    pub fn neighbors<'a>(
        &'a self,
        node_id: NodeId,
        direction: Direction,
    ) -> impl Iterator<Item = (&'a EdgeId, &'a Edge<W>)> + use<'a, N, I, O, W> {
        self.edges
            .iter()
            .filter(move |(_, edge)| match direction {
                Direction::Outgoing => edge.source == node_id,
                Direction::Incoming => edge.target == node_id,
                Direction::Any => edge.source == node_id || edge.target == node_id,
            })
            .map(|(id, edge)| (id, edge))
    }

    /// Returns mutable references to the edges touching the provided node, in `direction`.
    pub fn neighbors_mut<'a>(
        &'a mut self,
        node_id: NodeId,
        direction: Direction,
    ) -> impl Iterator<Item = (EdgeId, &'a mut Edge<W>)> + use<'a, N, I, O, W> {
        self.edges
            .iter_mut()
            .filter(move |(_, edge)| match direction {
                Direction::Outgoing => edge.source == node_id,
                Direction::Incoming => edge.target == node_id,
                Direction::Any => edge.source == node_id || edge.target == node_id,
            })
            .map(|(id, edge)| (*id, edge))
    }

    /// Return an iterator of references to the edges incoming into the given node.
    pub fn incomings<'a>(
        &'a self,
        node_id: NodeId,
    ) -> impl Iterator<Item = (&'a EdgeId, &'a Edge<W>)> + use<'a, N, I, O, W> {
        self.neighbors(node_id, Direction::Incoming)
    }

    /// Return an iterator of mutable references to the edges incoming into the given node.
    pub fn incomings_mut<'a>(
        &'a mut self,
        node_id: NodeId,
    ) -> impl Iterator<Item = (EdgeId, &'a mut Edge<W>)> + use<'a, N, I, O, W> {
        self.neighbors_mut(node_id, Direction::Incoming)
    }

    /// Return an iterator of references to the edges outgoing from the given node.
    pub fn outgoings<'a>(
        &'a self,
        node_id: NodeId,
    ) -> impl Iterator<Item = (&'a EdgeId, &'a Edge<W>)> + use<'a, N, I, O, W> {
        self.neighbors(node_id, Direction::Outgoing)
    }

    /// Return an iterator of mutable references to the edges outgoing from the given node.
    pub fn outgoings_mut<'a>(
        &'a mut self,
        node_id: NodeId,
    ) -> impl Iterator<Item = (EdgeId, &'a mut Edge<W>)> + use<'a, N, I, O, W> {
        self.neighbors_mut(node_id, Direction::Outgoing)
    }

    /// Return a flattened, ordered list of all the incoming feeds into a given
    /// node, i.e. the list of port links carried by the incoming edges.
    ///
    /// These are guaranteed to be ordered by the port with which they are
    /// connected to the node.
    pub fn incoming_feeds(&self, n: NodeId) -> Vec<Feed> {
        let mut incomings = self
            .incomings(n)
            .flat_map(|(_, edge)| {
                edge.ports().iter().map(|link| Feed {
                    source: edge.source().output_at(link.source_port),
                    target: edge.target().input_at(link.target_port),
                })
            })
            .collect::<Vec<_>>();
        incomings.sort_by_key(|feed| feed.target.port);
        // TODO: check that all are consecutive
        incomings
    }

    /// Return a flattened, ordered list of all the outgoing feeds into a given
    /// node, i.e. the list of port links carried by the outgoing edges.
    ///
    /// These are guaranteed to be ordered by the port from which they are
    /// emerging from the node.
    pub fn outgoing_feeds(&self, n: NodeId) -> Vec<Feed> {
        let mut outgoings = self
            .outgoings(n)
            .flat_map(|(_, edge)| {
                edge.ports().iter().map(|link| Feed {
                    source: n.output_at(link.source_port),
                    target: edge.target().input_at(link.target_port),
                })
            })
            .collect::<Vec<_>>();
        outgoings.sort_by_key(|feed| feed.source.port);
        // TODO: check that all are consecutive
        outgoings
    }

    /// Returns the edges of a node that starts at `node` and goes in the direction `direction`.
    /// The edges are filtered to only include edges that are between inner nodes.
    pub fn node_neighbors<'a>(
        &'a self,
        node_id: NodeId,
        direction: Direction,
    ) -> impl Iterator<Item = (&'a EdgeId, &'a Edge<W>)> + use<'a, N, I, O, W> {
        self.neighbors(node_id, direction).filter(|(_, edge)| {
            self.nodes[&edge.source].is_inner() && self.nodes[&edge.target].is_inner()
        })
    }

    /// Returns an iterator that traverses the graph in topological order (forward direction).
    /// This assumes the graph is a DAG (directed acyclic graph).
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use zkml::graph::Graph;
    /// let mut graph: Graph<&str, (), (), ()> = Graph::new();
    /// let node1 = graph.add_inner("first").unwrap();
    /// let node2 = graph.add_inner("second").unwrap();
    /// let node3 = graph.add_inner("third").unwrap();
    ///
    /// graph.add_edge(node1, node2, zkml::graph::Ports::consecutive(), Some(())).unwrap();
    /// graph.add_edge(node2, node3, zkml::graph::Ports::consecutive(), Some(())).unwrap();
    ///
    /// let order: Vec<_> = graph.forward_iter().map(|(_, data)| *data.inner()).collect();
    /// assert_eq!(order, vec!["first", "second", "third"]);
    /// ```
    pub fn forward_iter(&self) -> impl Iterator<Item = (NodeId, &Node<N, I, O>)> {
        self.dag_order::<true>()
            .map(|node_id| (node_id, &self.nodes[&node_id]))
    }

    /// Return an iterator that traverses the graph in topological order
    /// (forward, inputs to outputs), but only yields inner nodes and ignores
    /// input and output nodes.
    pub fn forward_inners(&self) -> impl Iterator<Item = (NodeId, &N)> {
        self.dag_order::<true>()
            .map(|node_id| (node_id, &self.nodes[&node_id]))
            .filter_map(|(n_id, n)| n.as_inner().map(|l| (n_id, l)))
    }

    /// Returns an iterator that traverses the graph in reverse topological order (backward direction).
    /// This assumes the graph is a DAG (directed acyclic graph).
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use zkml::graph::Graph;
    /// let mut graph: Graph<&str, (), (), ()> = Graph::new();
    /// let node1 = graph.add_inner("first").unwrap();
    /// let node2 = graph.add_inner("second").unwrap();
    /// let node3 = graph.add_inner("third").unwrap();
    ///
    /// graph.add_edge(node1, node2, zkml::graph::Ports::consecutive(), Some(())).unwrap();
    /// graph.add_edge(node2, node3, zkml::graph::Ports::consecutive(), Some(())).unwrap();
    ///
    /// let order: Vec<_> = graph.backward_iter().map(|(_, data)| *data.inner()).collect();
    /// assert_eq!(order, vec!["third", "second", "first"]);
    /// ```
    pub fn backward_iter(&self) -> impl Iterator<Item = (NodeId, &Node<N, I, O>)> {
        self.dag_order::<false>()
            .map(|node_id| (node_id, &self.nodes[&node_id]))
    }

    pub fn try_map_forward<N2, I2, O2>(
        &self,
        mut f: impl FnMut(NodeId, &Node<N, I, O>) -> anyhow::Result<Node<N2, I2, O2>>,
    ) -> anyhow::Result<Graph<N2, I2, O2, W>> {
        let new_nodes = self
            .dag_order::<true>()
            .map(|node_id| f(node_id, &self.nodes[&node_id]).map(|new_node| (node_id, new_node)))
            .collect::<anyhow::Result<BTreeMap<_, _>>>()?;

        Ok(Graph {
            nodes: new_nodes,
            edges: self.edges.clone(),
        })
    }

    pub fn try_into_map_forward<N2, I2, O2>(
        mut self,
        mut f: impl FnMut(NodeId, Node<N, I, O>, Vec<Feed>) -> anyhow::Result<Node<N2, I2, O2>>,
    ) -> anyhow::Result<Graph<N2, I2, O2, W>> {
        let new_nodes = self
            .dag_order::<true>()
            .collect::<Vec<_>>()
            .into_iter()
            .map(|node_id| {
                let old_node = self.nodes.remove(&node_id).unwrap();
                let incoming_feeds = self.incoming_feeds(node_id);
                f(node_id, old_node, incoming_feeds).map(|new_node| (node_id, new_node))
            })
            .collect::<anyhow::Result<BTreeMap<_, _>>>()?;
        Ok(Graph {
            nodes: new_nodes,
            edges: self.edges,
        })
    }

    fn dag_order<const FORWARD: bool>(
        &self,
    ) -> impl Iterator<Item = NodeId> + use<'_, N, W, I, O, FORWARD> {
        let all_nodes = self
            .nodes()
            .map(|(node_id, _)| *node_id)
            .collect::<BTreeSet<_>>();
        (0..self.nodes.len()).scan(all_nodes, |unvisited_nodes, _| {
            let next_node = unvisited_nodes.iter().find_map(|&node_id| {
                let is_node_next = if FORWARD {
                    // if the node only has "input" edges, then this is true
                    // otherwise, we check that each predecessor has already been visited
                    self.neighbors(node_id, Direction::Incoming)
                        .all(|(_, edge)| !unvisited_nodes.contains(&edge.source()))
                } else {
                    self.neighbors(node_id, Direction::Outgoing)
                        .all(|(_, edge)| !unvisited_nodes.contains(&edge.target()))
                };
                if is_node_next { Some(node_id) } else { None }
            });
            if let Some(ref next_node) = next_node {
                unvisited_nodes.remove(next_node);
            }
            next_node
        })
    }
}
impl<N, O, W> Graph<N, usize, O, W>
where
    O: PartialEq + std::fmt::Debug,
    W: Clone,
{
    /// Return a vector matching the i-th input to the ID of the node
    /// representing it in the graph.
    pub fn input_node_ids(&self) -> Vec<NodeId> {
        let input_nodes = self.input_nodes().collect::<Vec<_>>();
        let mut r = vec![0.into(); input_nodes.len()];
        for (node_id, i) in input_nodes.into_iter() {
            r[*i] = node_id;
        }
        r
    }
}
impl<N, I, W> Graph<N, I, usize, W>
where
    I: PartialEq + std::fmt::Debug,
    W: Clone,
{
    /// Return a vector matching the i-th output to the ID of the node
    /// representing it in the graph.
    pub fn output_node_ids(&self) -> Vec<NodeId> {
        let output_nodes = self.output_nodes().collect::<Vec<_>>();
        let mut r = vec![0.into(); output_nodes.len()];
        for (node_id, i) in output_nodes.into_iter() {
            r[*i] = node_id;
        }
        r
    }
}

impl<N, I, O, E> Default for Graph<N, I, O, E> {
    fn default() -> Self {
        Self {
            nodes: Default::default(),
            edges: Default::default(),
        }
    }
}

impl<N, I, O, W> Index<NodeId> for Graph<N, I, O, W> {
    type Output = Node<N, I, O>;

    fn index(&self, idx: NodeId) -> &Self::Output {
        &self.nodes[&idx]
    }
}

impl<T> Index<PortId> for Vec<T> {
    type Output = T;

    fn index(&self, idx: PortId) -> &Self::Output {
        &self[idx.0]
    }
}

pub trait IntoVecUsize {
    fn into_vec(self) -> Vec<usize>;
}

impl IntoVecUsize for std::ops::Range<usize> {
    fn into_vec(self) -> Vec<usize> {
        self.collect()
    }
}

impl IntoVecUsize for std::ops::RangeInclusive<usize> {
    fn into_vec(self) -> Vec<usize> {
        self.collect()
    }
}

impl IntoVecUsize for Vec<usize> {
    fn into_vec(self) -> Vec<usize> {
        self
    }
}

impl IntoVecUsize for usize {
    fn into_vec(self) -> Vec<usize> {
        vec![self]
    }
}

fn next_edge_id() -> EdgeId {
    EDGE_INDEX_COUNTER.fetch_add(1, Ordering::Relaxed).into()
}

impl Ports {
    pub fn iter(&self) -> impl Iterator<Item = &PortLink> {
        self.0.iter()
    }
    pub fn len(&self) -> usize {
        self.0.len()
    }
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
    #[cfg(test)]
    pub fn insert<P: Into<Ports>>(&mut self, ports: P) -> anyhow::Result<()> {
        for port in ports.into().0.into_iter() {
            if self
                .0
                .iter()
                .any(|p| p.source_port == port.source_port && p.target_port == port.target_port)
            {
                anyhow::bail!("Port already exists")
            }
            if self.0.iter().any(|p| p.target_port == port.target_port) {
                anyhow::bail!("Target port already exists")
            }
            self.0.push(port);
        }
        self.0.sort_by_key(|p| p.source_port);
        Ok(())
    }
}

impl Index<usize> for Ports {
    type Output = PortLink;

    fn index(&self, idx: usize) -> &Self::Output {
        &self.0[idx]
    }
}

impl From<(usize, usize)> for Ports {
    fn from(value: (usize, usize)) -> Self {
        Ports(vec![PortLink::new(value.0, value.1)])
    }
}

impl From<Vec<(usize, usize)>> for Ports {
    fn from(value: Vec<(usize, usize)>) -> Self {
        Ports(
            value
                .into_iter()
                .map(|(a, b)| PortLink::new(a, b))
                .collect(),
        )
    }
}

impl From<Vec<(PortId, PortId)>> for Ports {
    fn from(value: Vec<(PortId, PortId)>) -> Self {
        Ports(
            value
                .into_iter()
                .map(|(a, b)| PortLink::new(a, b))
                .collect(),
        )
    }
}

impl From<Vec<(&PortId, &PortId)>> for Ports {
    fn from(value: Vec<(&PortId, &PortId)>) -> Self {
        Ports(
            value
                .into_iter()
                .map(|(a, b)| PortLink::new(*a, *b))
                .collect(),
        )
    }
}

impl From<(PortId, PortId)> for Ports {
    fn from(value: (PortId, PortId)) -> Self {
        Ports(vec![PortLink::new(value.0, value.1)])
    }
}

impl From<PortLink> for Ports {
    fn from(value: PortLink) -> Self {
        Ports(vec![value])
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    #[derive(Clone, Debug, PartialEq, Eq)]
    struct TestWeight(usize);

    #[test]
    fn test_graph() {
        let mut graph: Graph<usize, (), (), TestWeight> = Graph::new();
        let input0 = graph.add_node(Node::Input(())).unwrap();
        let node1 = graph.add_node(Node::Inner(1)).unwrap();
        let node2 = graph.add_node(Node::Inner(2)).unwrap();
        let output0 = graph.add_node(Node::Output(())).unwrap();
        // try inserting a normal edge
        let first_edge = graph
            .add_edge(node1, node2, Ports::consecutive(), TestWeight(1))
            .unwrap();
        assert_eq!(graph.neighbors(node1, Direction::Outgoing).count(), 1);
        assert_eq!(graph.neighbors(node2, Direction::Incoming).count(), 1);
        assert_eq!(
            graph
                .neighbors(node1, Direction::Outgoing)
                .next()
                .unwrap()
                .0,
            &first_edge
        );
        assert_eq!(
            graph
                .neighbors(node2, Direction::Incoming)
                .next()
                .unwrap()
                .0,
            &first_edge
        );
        assert_eq!(graph.neighbors(node1, Direction::Any).count(), 1);
        assert_eq!(graph.neighbors(node2, Direction::Any).count(), 1);
        assert_eq!(graph[node1], Node::Inner(1));
        assert_eq!(graph[node2], Node::Inner(2));
        assert_eq!(
            graph.edge(&first_edge),
            Some(&Edge {
                source: node1,
                target: node2,
                ports: Ports::consecutive(),
                weight: Some(TestWeight(1))
            })
        );

        // try adding a duplicate edge
        graph
            .add_edge(node1, node2, Ports::consecutive(), TestWeight(2))
            .unwrap_err();
        // 1 + 2 = 3
        assert_eq!(
            graph.edge(&first_edge),
            Some(&Edge {
                source: node1,
                target: node2,
                ports: Ports::consecutive(),
                weight: Some(TestWeight(1))
            })
        );

        // try inserting an input edge
        let input_edge = graph
            .add_edge(input0, node1, (0, 0), TestWeight(3))
            .unwrap();
        assert_eq!(graph.neighbors(node1, Direction::Incoming).count(), 1);
        assert_eq!(
            graph.neighbors(node1, Direction::Any).count(),
            2,
            "{:?}",
            graph.neighbors(node1, Direction::Any).collect::<Vec<_>>()
        );
        assert_eq!(
            graph
                .neighbors(node1, Direction::Incoming)
                .next()
                .unwrap()
                .1,
            &Edge {
                source: input0,
                target: node1,
                ports: Ports::consecutive(),
                weight: Some(TestWeight(3))
            },
        );

        // try inserting an output edge
        let output_edge = graph
            .add_edge(node2, output0, (0, 0), TestWeight(3))
            .unwrap();
        assert_eq!(graph.neighbors(node2, Direction::Outgoing).count(), 1);
        assert_eq!(
            graph.neighbors(node2, Direction::Any).count(),
            2,
            "{:?}",
            graph.neighbors(node2, Direction::Any).collect::<Vec<_>>()
        );
        assert_eq!(
            graph
                .neighbors(node2, Direction::Outgoing)
                .next()
                .unwrap()
                .1,
            &Edge {
                source: node2,
                target: output0,
                ports: Ports::consecutive(),
                weight: Some(TestWeight(3))
            },
        );

        assert_eq!(graph.source_nodes().count(), 1);
        assert_eq!(graph.sink_nodes().count(), 1);

        // try removing an edge
        graph.remove_edge(first_edge).unwrap();
        assert_eq!(graph.neighbors(node1, Direction::Outgoing).count(), 0);
        assert_eq!(graph.neighbors(node2, Direction::Incoming).count(), 0);
        assert_eq!(graph.neighbors(node1, Direction::Any).count(), 1);
        assert_eq!(graph.neighbors(node2, Direction::Any).count(), 1);

        // try removing an input edge
        graph.remove_edge(input_edge).unwrap();
        assert_eq!(graph.neighbors(node1, Direction::Incoming).count(), 0);

        // try removing an output edge
        graph.remove_edge(output_edge).unwrap();
        assert_eq!(graph.neighbors(node2, Direction::Outgoing).count(), 0);

        // try removing a non-existing edge
        assert!(graph.remove_edge(10.into()).is_err());
    }

    #[test]
    fn test_graph_ports_consecutive() {
        let ports = Ports::consecutive();
        assert_eq!(ports.len(), 1);
        assert_eq!(ports[0].source_port, PortId(0));
        assert_eq!(ports[0].target_port, PortId(0));
    }

    #[test]
    fn test_graph_ports() {
        // For some reason, Rust type inference doesn't work with the default type so we have
        // to explicitly specify the type at declaration time.
        let mut graph = Graph::<usize, (), (), TestWeight>::new();
        let node1 = graph.add_node(Node::Inner(1)).unwrap();
        let node2 = graph.add_node(Node::Inner(2)).unwrap();
        graph
            .add_edge(node1, node2, Ports::consecutive(), TestWeight(1))
            .unwrap();
        assert_eq!(graph.neighbors(node1, Direction::Outgoing).count(), 1);
        assert_eq!(graph.neighbors(node2, Direction::Incoming).count(), 1);
        let (_edge_id, edge) = graph.neighbors(node1, Direction::Outgoing).next().unwrap();
        assert_eq!(edge.ports.len(), 1);
        assert_eq!(edge.ports[0].source_port, 0.into());
        assert_eq!(edge.ports[0].target_port, 0.into());

        // try to add a conflicting port on destination node
        graph
            .add_edge(node1, node2, PortLink::new(1, 0), TestWeight(1))
            .unwrap_err();

        // try to add a consecutive node
        let node3 = graph
            .add_consecutive_node(Node::Inner(3), node2, None)
            .unwrap();
        let (_edge_id, edge) = graph.neighbors(node2, Direction::Outgoing).next().unwrap();
        assert_eq!(edge.ports.len(), 1);
        assert_eq!(edge.ports[0].source_port, PortId(0));
        assert_eq!(edge.ports[0].target_port, PortId(0));

        println!("node1: {:?}", node1);
        println!("node2: {:?}", node2);
        println!("node3: {:?}", node3);
        // try to add many ports at the same time with same API
        // this will fail because we can't add the same edge twice
        graph
            .add_edge(
                node2,
                node3,
                vec![PortLink::new(1, 1), PortLink::new(2, 2)],
                None,
            )
            .unwrap_err();
    }
}
