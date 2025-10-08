use anyhow::{Context, ensure};
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
use serde::Deserialize;
use serde::Serialize;
use std::{
    collections::{BTreeMap, BTreeSet, HashSet},
    fmt::Debug,
    hash::Hash,
    ops::Index,
    sync::atomic::{AtomicUsize, Ordering},
};

/// Counter to automatically generate indices for nodes.
static NODE_INDEX_COUNTER: AtomicUsize = AtomicUsize::new(0);
static EDGE_INDEX_COUNTER: AtomicUsize = AtomicUsize::new(0);

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

/// Generic identifier for a node.
pub trait GenericNodeID:
    Eq
    + Hash
    + Debug
    + Clone
    + std::fmt::Display
    + Ord
    + PartialOrd
    + Into<Source<Self>>
    + Into<Target<Self>>
    + From<usize>
    + Into<usize>
{
}

impl<T> GenericNodeID for T where
    T: Eq
        + Hash
        + Debug
        + std::fmt::Display
        + Clone
        + Ord
        + PartialOrd
        + From<usize>
        + Into<usize>
        + Into<Source<Self>>
        + Into<Target<Self>>
{
}

/// Default unique node identifier.
#[derive(
    Debug,
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
    derive_more::Deref,
    PartialEq,
    Eq,
)]
#[display("Node({_0})")]
pub struct DefaultNodeID(pub usize);

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
pub struct EdgeID(usize);

#[derive(
    Debug,
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
    derive_more::Deref,
)]
#[display("Port({_0})")]
pub struct PortID(usize);

/// A port link is a link between an input port of a node and an output port of another node.
#[derive(
    Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash, Ord, PartialOrd, derive_more::Display,
)]
#[display("Link({source_port}->{target_port})")]
pub struct PortLink {
    pub source_port: PortID,
    pub target_port: PortID,
}

impl PortLink {
    pub fn consecutive() -> Self {
        Self {
            source_port: PortID(0),
            target_port: PortID(0),
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
    /// This returns a port link for two consecutive nodes assuming there is only a single output and input.
    pub fn consecutive() -> Self {
        Self(vec![PortLink {
            source_port: PortID(0),
            target_port: PortID(0),
        }])
    }
    pub fn sorted(self) -> Self {
        let mut ports = self.0;
        ports.sort_by_key(|p| p.source_port);
        Self(ports)
    }
}

/// Source of an edge is either a node or it is an input offset
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash, derive_more::Display)]
pub enum Source<I> {
    #[display("Source({_0})")]
    Node(I),
    #[display("Input")]
    Input,
}

/// Target of an edge is either a node or it is an output offset
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash, derive_more::Display)]
pub enum Target<I> {
    #[display("Target({_0})")]
    Node(I),
    #[display("Output")]
    Output,
}

/// An edge is a connection between a source and a target with a list of ports and a weight.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct Edge<I, W> {
    source: Source<I>,
    target: Target<I>,
    ports: Ports,
    /// An edge doesn't necessarily have a weight.
    /// Note only the weight is public since it's the only modifiable field from the user perspective.
    /// Ports shouldn't be allowed to be modified at will otherwise the invariant of the ports may be violated.
    pub weight: Option<W>,
}

impl<I: GenericNodeID, W> Edge<I, W> {
    pub fn new<P: Into<Ports>, S: Into<Source<I>>, T: Into<Target<I>>>(
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
        source: I,
        target: I,
        ports: P,
        weight: Option<W>,
    ) -> Self {
        Self {
            source: Source::Node(source),
            target: Target::Node(target),
            ports: ports.into().sorted(),
            weight,
        }
    }
    pub fn input<P: Into<Ports>>(node: I, ports: P, weight: Option<W>) -> Self {
        Self {
            source: Source::Input,
            target: Target::Node(node),
            weight,
            ports: ports.into().sorted(),
        }
    }
    pub fn output<P: Into<Ports>>(node: I, ports: P, weight: Option<W>) -> Self {
        Self {
            source: Source::Node(node),
            target: Target::Output,
            ports: ports.into().sorted(),
            weight,
        }
    }
    pub fn is_between_nodes(&self) -> bool {
        matches!(self.source, Source::Node(_)) && matches!(self.target, Target::Node(_))
    }
    pub fn is_input(&self) -> bool {
        matches!(self.source, Source::Input)
    }
    pub fn is_output(&self) -> bool {
        matches!(self.target, Target::Output)
    }
    pub fn is_incoming_to(&self, node_id: &I) -> bool {
        match self.target {
            Target::Node(ref target) => target == node_id,
            Target::Output => false,
        }
    }
    pub fn is_outgoing_from(&self, node_id: &I) -> bool {
        match self.source {
            Source::Node(ref source) => source == node_id,
            Source::Input => false,
        }
    }
    pub fn source(&self) -> &Source<I> {
        &self.source
    }
    pub fn target(&self) -> &Target<I> {
        &self.target
    }
    pub fn source_id(&self) -> Option<&I> {
        match self.source {
            Source::Node(ref node) => Some(node),
            _ => None,
        }
    }
    pub fn target_id(&self) -> Option<&I> {
        match self.target {
            Target::Node(ref node) => Some(node),
            _ => None,
        }
    }
    pub fn ports(&self) -> &Ports {
        &self.ports
    }

    /// Tries to find the other end of the edge given a node id.
    /// If the node id is the source, then the target is returned.
    /// If the node id is the target, then the source is returned.
    /// If the node id is not the source or the target, then None is returned.
    /// If one of the end of the edge is not a node, then None is returned.
    pub fn other_end(&self, node_id: &I) -> Option<&I> {
        match (&self.source, &self.target) {
            (Source::Node(ref source), Target::Node(ref target)) => match node_id {
                _ if node_id == source => Some(target),
                _ if node_id == target => Some(source),
                _ => None,
            },
            _ => None,
        }
    }
}

impl PortLink {
    pub fn new<I: Into<PortID>, I2: Into<PortID>>(source: I, target: I2) -> Self {
        Self {
            source_port: source.into(),
            target_port: target.into(),
        }
    }
}

/// Basic structure that contains a graph and a list of input nodes.
/// The graph is colored, e.g. each node is associated with a color that corresponds to which machine or thread etc
/// it should be executed on.
/// NOTE: need to support the more general case where an output node can be used both as output of the graph
/// and as input to another node of the graph
/// NOTE: need to support the strict ordering of inputs when a node expects both inputs from its predecessors AND
/// input from the graph input data.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(bound(
    serialize = "NodeID: Serialize, N: Serialize, W: Serialize",
    deserialize = "NodeID: Eq + Hash + Ord + Deserialize<'de>, N: Deserialize<'de>, W: Deserialize<'de>"
))]
pub struct Graph<N, W, NodeID = DefaultNodeID> {
    /// Nodes indexed by their index - we use a BTreeMap to make the graph iteration deterministic
    /// and sorted by increasing order of the node id, which is usually equivalent to the order of insertion.
    nodes: BTreeMap<NodeID, N>,
    /// Contains all the edges in the graph.
    /// NOTE: currently O(n) to search but once API is stabilized, we can move to a multi key map
    /// indexed by both the source and the target node to search in O(1).
    pub(crate) edges: Vec<(EdgeID, Edge<NodeID, W>)>,
}

impl<NodeID, N, W> Graph<N, W, NodeID>
where
    NodeID: GenericNodeID,
    W: Clone,
{
    /// Add a node to the graph. It will return the index of the added node.
    /// It will return an error if the next node id to be picked up already exists.
    /// It can happen if one called `[add_node_with_id]` before with the next node id to be picked up.
    pub fn add_node(&mut self, node: N) -> anyhow::Result<NodeID> {
        let node_id = next_node_id();
        self.add_node_with_id(node_id, node).map(|_| node_id.into())
    }

    /// Add a node to the graph with the given index. It will return an error if the node id already exists.
    pub fn add_node_with_id<I: Into<NodeID> + Clone>(
        &mut self,
        nidx: I,
        node: N,
    ) -> anyhow::Result<()> {
        let id = nidx.into();
        match self.nodes.insert(id.clone(), node) {
            None => Ok(()),
            Some(_) => anyhow::bail!("Node with index {id:?} already exists"),
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
    /// let mut graph: Graph<&str, ()> = Graph::new();
    /// let node1 = graph.add_node("first").unwrap();
    /// let node2 = graph.add_node("second").unwrap();
    ///
    /// // Simple consecutive connection
    /// let edge_id = graph.add_edge(node1, node2, Ports::consecutive(), Some(())).unwrap();
    ///
    /// // Custom port mapping using PortLink
    /// let node3 = graph.add_node("third").unwrap();
    /// graph.add_edge(node2, node3, PortLink::new(0, 0), None).unwrap();
    ///
    /// // Or using a (usize, usize) tuple directly
    /// let node4 = graph.add_node("fourth").unwrap();
    /// graph.add_edge(node3, node4, (0, 0), None).unwrap();
    /// ```
    pub fn add_edge<P: Into<Ports>, WO: Into<Option<W>>>(
        &mut self,
        source: NodeID,
        target: NodeID,
        ports: P,
        weight: WO,
    ) -> anyhow::Result<EdgeID> {
        let edge = Edge::new(source, target, ports, weight.into());
        Ok(self.add_edges_raw(vec![edge])?[0])
    }

    /// Wrapper method around adding a consecutive edge between two nodes.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use zkml::graph::Graph;
    /// let mut graph: Graph<&str, ()> = Graph::new();
    /// let node1 = graph.add_node("first").unwrap();
    /// let node2 = graph.add_node("second").unwrap();
    ///
    /// let edge_id = graph.add_consecutive_edge(node1, node2, Some(())).unwrap();
    /// assert_eq!(graph.neighbors(&node1, zkml::graph::Direction::Outgoing).count(), 1);
    /// ```
    pub fn add_consecutive_edge<WO: Into<Option<W>>>(
        &mut self,
        source: NodeID,
        target: NodeID,
        weight: WO,
    ) -> anyhow::Result<EdgeID> {
        self.add_edge(source, target, Ports::consecutive(), weight)
    }

    /// Add a consecutive node to the graph.
    /// It adds a new node to the graph and connects it to the previous node with a consecutive edge.
    /// In this case, there is only one port link between the two nodes.
    /// If no previous node is provided, it adds an input edge to the new node.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use zkml::graph::Graph;
    /// let mut graph: Graph<&str, ()> = Graph::new();
    /// let node1 = graph.add_node("first").unwrap();
    ///
    /// // Add a node connected to an existing node
    /// let node2 = graph.add_consecutive_node("second", Some(node1), Some(())).unwrap();
    /// assert_eq!(graph.neighbors(&node1, zkml::graph::Direction::Outgoing).count(), 1);
    ///
    /// // Add an input node (no previous connection)
    /// let input_node = graph.add_consecutive_node("input", None, Some(())).unwrap();
    /// assert_eq!(graph.input_nodes().count(), 1);
    /// ```
    pub fn add_consecutive_node<WO: Into<Option<W>>>(
        &mut self,
        node: N,
        previous_node_id: Option<NodeID>,
        weight: WO,
    ) -> anyhow::Result<NodeID> {
        let new_node_id = self.add_node(node)?;
        match previous_node_id {
            Some(id) => self.add_edge(id, new_node_id.clone(), Ports::consecutive(), weight)?,
            None => {
                let next_input_offset: usize = self
                    .edges
                    .iter()
                    .filter(|(_, edge)| edge.source == Source::Input)
                    .flat_map(|(_, edge)| edge.ports.iter().map(|port| port.source_port.0 + 1))
                    .max()
                    .unwrap_or(0);
                self.add_edges_raw(vec![Edge::input(
                    new_node_id.clone(),
                    (next_input_offset, 0usize),
                    weight.into(),
                )])?[0]
            }
        };
        Ok(new_node_id)
    }
    /// Wrapper method around adding an input edge to a node.
    /// It associates every input offset to the next consecutive available target ports on the target id.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use zkml::graph::Graph;
    /// let mut graph: Graph<&str, ()> = Graph::new();
    /// let node = graph.add_node("node").unwrap();
    ///
    /// // Set input at index 0
    /// let input_id = graph.set_input(node, 0, Some(())).unwrap();
    /// assert_eq!(graph.input_nodes().count(), 1);
    /// ```
    pub fn set_input<I: IntoVecUsize, WO: Into<Option<W>>>(
        &mut self,
        node: NodeID,
        input_index: I,
        weight: WO,
    ) -> anyhow::Result<EdgeID> {
        let target_port_offset: usize = self
            .edges
            .iter()
            .filter(|(_, edge)| edge.target == Target::Node(node.clone()))
            .flat_map(|(_, edge)| edge.ports.iter().map(|port| port.target_port.0 + 1))
            .max()
            .unwrap_or(0);
        let ports = input_index
            .into_vec()
            .into_iter()
            .enumerate()
            .map(|(target_port, source_port)| (source_port, target_port + target_port_offset))
            .collect::<Vec<(usize, usize)>>();
        let edge = Edge::input(node, ports, weight.into());
        Ok(self.add_edges_raw(vec![edge])?[0])
    }
    /// Wrapper method around adding an output edge from a node.
    /// It associates every output offset to a consecutive source port on the source id.
    /// e.g. if target ports were 2,4 then source ports would be 0,1.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use zkml::graph::Graph;
    /// let mut graph: Graph<&str, ()> = Graph::new();
    /// let node = graph.add_node("node").unwrap();
    ///
    /// // Set output at index 0
    /// let output_id = graph.set_output(node, 0, Some(())).unwrap();
    /// assert_eq!(graph.output_nodes().count(), 1);
    /// ```
    pub fn set_output<I: IntoVecUsize, WO: Into<Option<W>>>(
        &mut self,
        node: NodeID,
        output_index: I,
        weight: WO,
    ) -> anyhow::Result<EdgeID> {
        let source_port_offset: usize = self
            .edges
            .iter()
            .filter(|(_, edge)| edge.source == Source::Node(node.clone()))
            .flat_map(|(_, edge)| edge.ports.iter().map(|port| port.source_port.0 + 1))
            .max()
            .unwrap_or(0);
        let ports = output_index
            .into_vec()
            .into_iter()
            .enumerate()
            .map(|(source, target)| (source + source_port_offset, target))
            .collect::<Vec<(usize, usize)>>();
        let edge = Edge::output(node, ports, weight.into());
        Ok(self.add_edges_raw(vec![edge])?[0])
    }

    pub fn add_edges_raw(&mut self, edges: Vec<Edge<NodeID, W>>) -> anyhow::Result<Vec<EdgeID>> {
        let mut edge_ids = Vec::with_capacity(edges.len());
        for new_edge in edges {
            // making sure the source and the targets exists
            if let Source::Node(ref source_id) = new_edge.source {
                ensure!(
                    self.nodes.contains_key(source_id),
                    "Source node {source_id} not found"
                );
            }
            if let Target::Node(ref target_id) = new_edge.target {
                ensure!(
                    self.nodes.contains_key(target_id),
                    "Target node {target_id} not found"
                );
            }
            // compare with all other edges to see if
            // 1. there are duplicates
            // 2. the ports are consistent, e.g. no target port is used twice on the same node
            let duplicate = self.edges.iter().any(|(_, current_edge)| {
                current_edge.source == new_edge.source && current_edge.target == new_edge.target
            });
            ensure!(
                !duplicate,
                "Edge between {:?} and {:?} already exists",
                new_edge.source,
                new_edge.target
            );
            // no need to detect duplicates on already existing edges since it's already checked
            self.check_consistency(
                &new_edge.target,
                new_edge.ports.iter().map(|port| &port.target_port),
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
    /// let mut graph: Graph<&str, ()> = Graph::new();
    /// let node1 = graph.add_node("first").unwrap();
    /// let node2 = graph.add_node("second").unwrap();
    /// let edge_id = graph.add_edge(node1, node2, zkml::graph::Ports::consecutive(), Some(())).unwrap();
    ///
    /// assert_eq!(graph.edges().count(), 1);
    /// graph.remove_edge(&edge_id).unwrap();
    /// assert_eq!(graph.edges().count(), 0);
    /// ```
    pub fn remove_edge(&mut self, edge_id: &EdgeID) -> anyhow::Result<()> {
        let curr_len = self.edges.len();
        self.edges.retain(|(id, _)| id != edge_id);
        if self.edges.len() == curr_len {
            anyhow::bail!("Edge with id {edge_id:?} not found");
        }
        Ok(())
    }

    /// Maps a node to a new node.
    pub fn replace_node<N2, F>(&mut self, node_id: &NodeID, f: F) -> anyhow::Result<()>
    where
        F: FnOnce(N) -> N,
    {
        let old_node = self.nodes.remove(node_id).context("Node not found")?;
        self.nodes.insert(node_id.clone(), f(old_node));
        Ok(())
    }

    /// Returns a reference to the node with the given ID, if it exists.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use zkml::graph::Graph;
    /// let mut graph: Graph<&str, ()> = Graph::new();
    /// let node_id = graph.add_node("test").unwrap();
    ///
    /// assert_eq!(graph.node(&node_id), Some(&"test"));
    /// assert_eq!(graph.node(&zkml::graph::DefaultNodeID(999)), None);
    /// ```
    pub fn node(&self, node_id: &NodeID) -> Option<&N> {
        self.nodes.get(node_id)
    }
    pub fn node_mut(&mut self, node_id: &NodeID) -> Option<&mut N> {
        self.nodes.get_mut(node_id)
    }

    /// Returns the number of nodes in the graph.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use zkml::graph::Graph;
    /// let mut graph: Graph<&str, ()> = Graph::new();
    /// assert_eq!(graph.node_count(), 0);
    ///
    /// let node1 = graph.add_node("first").unwrap();
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
    /// let mut graph: Graph<&str, ()> = Graph::new();
    /// let node1 = graph.add_node("first").unwrap();
    /// let node2 = graph.add_node("second").unwrap();
    ///
    /// let nodes: Vec<_> = graph.nodes().collect();
    /// assert_eq!(nodes.len(), 2);
    /// ```
    pub fn nodes(&self) -> impl Iterator<Item = (&NodeID, &N)> + use<'_, NodeID, N, W> {
        self.nodes.iter()
    }

    /// Returns the node ids of the output nodes alongside the associated ports that enforce
    /// which output is set at which index
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use zkml::graph::Graph;
    /// let mut graph: Graph<&str, ()> = Graph::new();
    /// let node = graph.add_node("node").unwrap();
    /// graph.set_output(node, 0, Some(())).unwrap();
    ///
    /// let outputs: Vec<_> = graph.output_nodes().collect();
    /// assert_eq!(outputs.len(), 1);
    /// ```
    pub fn output_nodes(&self) -> impl Iterator<Item = (&NodeID, &Ports)> + use<'_, NodeID, N, W> {
        self.edges.iter().filter_map(|(_, edge)| match edge.target {
            Target::Output => Some((edge.source_id().unwrap(), &edge.ports)),
            _ => None,
        })
    }

    /// Returns an iterator over input nodes and their associated ports.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use zkml::graph::Graph;
    /// let mut graph: Graph<&str, ()> = Graph::new();
    /// let node = graph.add_node("node").unwrap();
    /// graph.set_input(node, 0, Some(())).unwrap();
    ///
    /// let inputs: Vec<_> = graph.input_nodes().collect();
    /// assert_eq!(inputs.len(), 1);
    /// ```
    pub fn input_nodes(&self) -> impl Iterator<Item = (&NodeID, &Ports)> + use<'_, NodeID, N, W> {
        self.edges.iter().filter_map(|(_, edge)| match edge.source {
            Source::Input => Some((edge.target_id().unwrap(), &edge.ports)),
            _ => None,
        })
    }

    /// Returns an iterator over all edges in the graph as (edge_id, edge) pairs.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use zkml::graph::Graph;
    /// let mut graph: Graph<&str, ()> = Graph::new();
    /// let node1 = graph.add_node("first").unwrap();
    /// let node2 = graph.add_node("second").unwrap();
    /// let edge_id = graph.add_edge(node1, node2, zkml::graph::Ports::consecutive(), Some(())).unwrap();
    ///
    /// let edges: Vec<_> = graph.edges().collect();
    /// assert_eq!(edges.len(), 1);
    /// ```
    pub fn edges(
        &self,
    ) -> impl Iterator<Item = (&EdgeID, &Edge<NodeID, W>)> + use<'_, NodeID, N, W> {
        self.edges.iter().map(|(id, edge)| (id, edge))
    }

    pub fn edge_between<I: Into<Source<NodeID>>, I2: Into<Target<NodeID>>>(
        &self,
        source: I,
        target: I2,
    ) -> Option<&Edge<NodeID, W>> {
        let source = source.into();
        let target = target.into();
        self.edges
            .iter()
            .filter_map(|(_, edge)| {
                if edge.source == source && edge.target == target {
                    Some(edge)
                } else {
                    None
                }
            })
            .next()
    }

    /// mut_fn is a function that can be used to mutate the edge. The graph ensures
    /// that the edge is modified in a consistent way.
    /// If the modification does not respect the invariant of the ports, it will return an error.
    /// In such a case, the graph will be left in an inconsistent state and should not be used anymore.
    pub fn edge_between_mut<I: Into<Source<NodeID>>, I2: Into<Target<NodeID>>>(
        &mut self,
        source: I,
        target: I2,
        mut mut_fn: impl FnMut(&mut Edge<NodeID, W>) -> anyhow::Result<()>,
    ) -> anyhow::Result<()> {
        let source = source.into();
        let target = target.into();
        match self
            .edges
            .iter_mut()
            .filter_map(|(id, edge)| {
                if edge.source == source && edge.target == target {
                    Some((id, edge))
                } else {
                    None
                }
            })
            .next()
        {
            Some((id, edge)) => {
                mut_fn(edge).context(format!(
                    "Modification of edge between {:?} and {:?} failed: ",
                    source, target
                ))?;
                Some(id)
            }
            None => None,
        }
        .context(format!(
            "Edge between {:?} and {:?} not found",
            source, target
        ))?;
        self.check_consistency(&target, vec![])?;
        Ok(())
    }

    /// Checks that no two target_port is assigned twice amongst all edges that have the same target node.
    /// Checks that all ports are consecutive and fill the range (0..num_ports).
    fn check_consistency<'a, I: IntoIterator<Item = &'a PortID>>(
        &'a self,
        target_node: &'a Target<NodeID>,
        new_ports: I,
    ) -> anyhow::Result<()> {
        let all_target_ports = self
            .edges
            .iter()
            .filter(move |(_, current_edge)| &current_edge.target == target_node)
            .flat_map(|(_, current_edge)| current_edge.ports.0.iter())
            .map(|port| &port.target_port)
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
            (0..len).all(|i| set.contains(&PortID(i))),
            "Ports are not consecutive: {:?}",
            set
        );
        Ok(())
    }

    pub fn edge<'a>(&'a self, edge_id: &EdgeID) -> Option<&'a Edge<NodeID, W>> {
        self.edges
            .iter()
            .find(|(id, _)| id == edge_id)
            .map(|(_, edge)| edge)
    }

    pub fn edge_mut<'a>(&'a mut self, edge_id: &EdgeID) -> Option<&'a mut Edge<NodeID, W>> {
        self.edges
            .iter_mut()
            .find(|(id, _)| id == edge_id)
            .map(|(_, edge)| edge)
    }

    /// Returns the edges of a node that starts at `node` and goes in the direction `direction`.
    pub fn neighbors_mut<'a>(
        &'a mut self,
        // Note it's not a reference because otherwise it needs to be captured by the closure
        // and rust borrow struggles with that
        node: &'a NodeID,
        direction: Direction,
    ) -> impl Iterator<Item = (&'a mut EdgeID, &'a mut Edge<NodeID, W>)> + use<'a, NodeID, N, W>
    {
        self.edges
            .iter_mut()
            .filter(move |(_, edge)| match direction {
                Direction::Outgoing => edge.source == node.into(),
                Direction::Incoming => edge.target == node.into(),
                Direction::Any => true,
            })
            .map(|(id, edge)| (id, edge))
    }

    /// Returns the edges of a node that starts at `node` and goes in the direction `direction`.
    pub fn neighbors<'a>(
        &'a self,
        node: &'a NodeID,
        direction: Direction,
    ) -> impl Iterator<Item = (&'a EdgeID, &'a Edge<NodeID, W>)> + use<'a, N, NodeID, W> {
        self.edges
            .iter()
            .filter(move |(_, edge)| match direction {
                Direction::Outgoing => edge.source == node.into(),
                Direction::Incoming => edge.target == node.into(),
                Direction::Any => edge.source == node.into() || edge.target == node.into(),
            })
            .map(|(id, edge)| (id, edge))
    }

    /// Returns the edges of a node that starts at `node` and goes in the direction `direction`.
    /// The edges are filtered to only include edges that are between nodes.
    pub fn node_neighbors<'a>(
        &'a self,
        node: &'a NodeID,
        direction: Direction,
    ) -> impl Iterator<Item = (&'a EdgeID, &'a Edge<NodeID, W>)> + use<'a, N, NodeID, W> {
        self.neighbors(node, direction)
            .filter(|(_, edge)| edge.is_between_nodes())
    }

    /// Number of nodes in the graph
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use zkml::graph::Graph;
    /// let mut graph: Graph<&str, ()> = Graph::new();
    /// assert_eq!(graph.graph_order(), 0);
    ///
    /// graph.add_node("node").unwrap();
    /// assert_eq!(graph.graph_order(), 1);
    /// ```
    pub fn graph_order(&self) -> usize {
        self.nodes.len()
    }

    /// Returns an iterator that traverses the graph in topological order (forward direction).
    /// This assumes the graph is a DAG (directed acyclic graph).
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use zkml::graph::Graph;
    /// let mut graph: Graph<&str, ()> = Graph::new();
    /// let node1 = graph.add_node("first").unwrap();
    /// let node2 = graph.add_node("second").unwrap();
    /// let node3 = graph.add_node("third").unwrap();
    ///
    /// graph.add_edge(node1, node2, zkml::graph::Ports::consecutive(), Some(())).unwrap();
    /// graph.add_edge(node2, node3, zkml::graph::Ports::consecutive(), Some(())).unwrap();
    ///
    /// let order: Vec<_> = graph.forward_iter().map(|(_, data)| *data).collect();
    /// assert_eq!(order, vec!["first", "second", "third"]);
    /// ```
    pub fn forward_iter(&self) -> impl Iterator<Item = (NodeID, &N)> {
        self.dag_order::<true>()
            .map(|node_id| (node_id.clone(), &self.nodes[&node_id]))
    }

    /// Returns an iterator that traverses the graph in reverse topological order (backward direction).
    /// This assumes the graph is a DAG (directed acyclic graph).
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use zkml::graph::Graph;
    /// let mut graph: Graph<&str, ()> = Graph::new();
    /// let node1 = graph.add_node("first").unwrap();
    /// let node2 = graph.add_node("second").unwrap();
    /// let node3 = graph.add_node("third").unwrap();
    ///
    /// graph.add_edge(node1, node2, zkml::graph::Ports::consecutive(), Some(())).unwrap();
    /// graph.add_edge(node2, node3, zkml::graph::Ports::consecutive(), Some(())).unwrap();
    ///
    /// let order: Vec<_> = graph.backward_iter().map(|(_, data)| *data).collect();
    /// assert_eq!(order, vec!["third", "second", "first"]);
    /// ```
    pub fn backward_iter(&self) -> impl Iterator<Item = (NodeID, &N)> {
        self.dag_order::<false>()
            .map(|node_id| (node_id.clone(), &self.nodes[&node_id]))
    }

    pub fn try_map_forward<N2>(
        &self,
        mut f: impl FnMut(NodeID, &N) -> anyhow::Result<N2>,
    ) -> anyhow::Result<Graph<N2, W, NodeID>> {
        let new_nodes = self
            .dag_order::<true>()
            .map(|node_id| {
                f(node_id.clone(), &self.nodes[&node_id]).map(|new_node| (node_id, new_node))
            })
            .collect::<anyhow::Result<BTreeMap<_, _>>>()?;

        Ok(Graph {
            nodes: new_nodes,
            edges: self.edges.clone(),
        })
    }

    pub fn try_into_map_forward<N2>(
        mut self,
        mut f: impl FnMut(NodeID, N, &[&Edge<NodeID, W>]) -> anyhow::Result<N2>,
    ) -> anyhow::Result<Graph<N2, W, NodeID>> {
        let new_nodes = self
            .dag_order::<true>()
            .collect::<Vec<_>>()
            .into_iter()
            .map(|node_id| {
                let old_node = self.nodes.remove(&node_id).unwrap();
                let edges = self
                    .neighbors(&node_id, Direction::Any)
                    .map(|(_, edge)| edge)
                    .collect::<Vec<_>>();
                f(node_id.clone(), old_node, &edges).map(|new_node| (node_id, new_node))
            })
            .collect::<anyhow::Result<BTreeMap<_, _>>>()?;
        Ok(Graph {
            nodes: new_nodes,
            edges: self.edges.clone(),
        })
    }

    fn dag_order<const FORWARD: bool>(
        &self,
    ) -> impl Iterator<Item = NodeID> + use<'_, N, W, NodeID, FORWARD> {
        let all_nodes = self
            .nodes()
            .map(|(node_id, _)| node_id.clone())
            .collect::<BTreeSet<_>>();
        (0..self.nodes.len()).scan(all_nodes, |unvisited_nodes, _| {
            let next_node = unvisited_nodes.iter().find_map(|node_id| {
                let is_node_next = if FORWARD {
                    // if the node only has "input" edges, then this is true
                    // otherwise, we check that each predecessor has already been visited
                    self.neighbors(node_id, Direction::Incoming)
                        .filter(|(_, edge)| edge.is_between_nodes())
                        .all(|(_, edge)| {
                            edge.source_id()
                                .map(|id| !unvisited_nodes.contains(id))
                                .unwrap_or(false)
                        })
                } else {
                    // if the node only has "output" edges, then this is true
                    // otherwise, we check that each successor has already been visited
                    self.neighbors(node_id, Direction::Outgoing)
                        .filter(|(_, edge)| edge.is_between_nodes())
                        .all(|(_, edge)| {
                            edge.target_id()
                                .map(|id| !unvisited_nodes.contains(id))
                                .unwrap_or(false)
                        })
                };
                if is_node_next {
                    Some(node_id.clone())
                } else {
                    None
                }
            });
            if let Some(ref next_node) = next_node {
                unvisited_nodes.remove(next_node);
            }
            next_node
        })
    }
}

impl<NodeID, N, E> Default for Graph<N, E, NodeID> {
    fn default() -> Self {
        Self {
            nodes: Default::default(),
            edges: Default::default(),
        }
    }
}

impl<N, W, NodeID> Graph<N, W, NodeID> {
    pub fn new() -> Self {
        Self::default()
    }
}
impl<NodeID, N, W> Index<NodeID> for Graph<N, W, NodeID>
where
    NodeID: GenericNodeID,
{
    type Output = N;

    fn index(&self, idx: NodeID) -> &Self::Output {
        &self.nodes[&idx]
    }
}

impl<NodeID, N, W> Index<&NodeID> for Graph<N, W, NodeID>
where
    NodeID: GenericNodeID,
{
    type Output = N;

    fn index(&self, idx: &NodeID) -> &Self::Output {
        &self.nodes[idx]
    }
}

impl<T> Index<PortID> for Vec<T> {
    type Output = T;

    fn index(&self, idx: PortID) -> &Self::Output {
        &self[idx.0]
    }
}

impl<T> Index<&PortID> for Vec<T> {
    type Output = T;

    fn index(&self, idx: &PortID) -> &Self::Output {
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

impl From<DefaultNodeID> for Source<DefaultNodeID> {
    fn from(value: DefaultNodeID) -> Self {
        Source::Node(value)
    }
}

impl From<DefaultNodeID> for Target<DefaultNodeID> {
    fn from(value: DefaultNodeID) -> Self {
        Target::Node(value)
    }
}

impl<I: GenericNodeID> From<&I> for Source<I> {
    fn from(value: &I) -> Self {
        Source::Node(value.clone())
    }
}
impl<I: GenericNodeID> From<&I> for Target<I> {
    fn from(value: &I) -> Self {
        Target::Node(value.clone())
    }
}

fn next_node_id() -> usize {
    NODE_INDEX_COUNTER.fetch_add(1, Ordering::Relaxed)
}

fn next_edge_id() -> EdgeID {
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

impl<I: GenericNodeID> From<usize> for Source<I> {
    fn from(value: usize) -> Self {
        Source::Node(I::from(value))
    }
}
impl<I: GenericNodeID> From<usize> for Target<I> {
    fn from(value: usize) -> Self {
        Target::Node(I::from(value))
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

impl From<Vec<(PortID, PortID)>> for Ports {
    fn from(value: Vec<(PortID, PortID)>) -> Self {
        Ports(
            value
                .into_iter()
                .map(|(a, b)| PortLink::new(a, b))
                .collect(),
        )
    }
}

impl From<Vec<(&PortID, &PortID)>> for Ports {
    fn from(value: Vec<(&PortID, &PortID)>) -> Self {
        Ports(
            value
                .into_iter()
                .map(|(a, b)| PortLink::new(*a, *b))
                .collect(),
        )
    }
}

impl From<(PortID, PortID)> for Ports {
    fn from(value: (PortID, PortID)) -> Self {
        Ports(vec![PortLink::new(value.0, value.1)])
    }
}

impl From<PortLink> for Ports {
    fn from(value: PortLink) -> Self {
        Ports(vec![value])
    }
}

// impl<I: Into<PortLink>> From<I> for Vec<PortLink> {
//    fn from(value: I) -> Self {
//        vec![value.into()]
//    }
//}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    #[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
    pub struct TestNode(pub usize);
    #[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
    struct TestWeight(usize);

    #[test]
    fn test_graph() {
        let mut graph: Graph<TestNode, TestWeight> = Graph::new();
        let node1 = graph.add_node(TestNode(1)).unwrap();
        let node2 = graph.add_node(TestNode(2)).unwrap();
        // try inserting a normal edge
        let first_edge = graph
            .add_edge(node1, node2, Ports::consecutive(), TestWeight(1))
            .unwrap();
        assert_eq!(graph.neighbors(&node1, Direction::Outgoing).count(), 1);
        assert_eq!(graph.neighbors(&node2, Direction::Incoming).count(), 1);
        assert_eq!(
            graph
                .neighbors(&node1, Direction::Outgoing)
                .next()
                .unwrap()
                .0,
            &first_edge
        );
        assert_eq!(
            graph
                .neighbors(&node2, Direction::Incoming)
                .next()
                .unwrap()
                .0,
            &first_edge
        );
        assert_eq!(graph.neighbors(&node1, Direction::Any).count(), 1);
        assert_eq!(graph.neighbors(&node2, Direction::Any).count(), 1);
        assert_eq!(graph[node1], TestNode(1));
        assert_eq!(graph[node2], TestNode(2));
        assert_eq!(
            graph.edge(&first_edge),
            Some(&Edge {
                source: Source::Node(node1),
                target: Target::Node(node2),
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
                source: Source::Node(node1),
                target: Target::Node(node2),
                ports: Ports::consecutive(),
                weight: Some(TestWeight(1))
            })
        );

        // try inserting an input edge
        let input_id = graph.set_input(node1, 0, TestWeight(3)).unwrap();
        assert_eq!(graph.neighbors(&node1, Direction::Incoming).count(), 1);
        assert_eq!(
            graph.neighbors(&node1, Direction::Any).count(),
            2,
            "{:?}",
            graph.neighbors(&node1, Direction::Any).collect::<Vec<_>>()
        );
        assert_eq!(
            graph
                .neighbors(&node1, Direction::Incoming)
                .next()
                .unwrap()
                .1,
            &Edge {
                source: Source::Input,
                target: Target::Node(node1),
                ports: Ports::consecutive(),
                weight: Some(TestWeight(3))
            },
        );
        assert_eq!(graph.input_nodes().count(), 1);

        // try inserting an output edge
        let output_id = graph.set_output(node2, 0, TestWeight(3)).unwrap();
        assert_eq!(graph.neighbors(&node2, Direction::Outgoing).count(), 1);
        assert_eq!(
            graph.neighbors(&node2, Direction::Any).count(),
            2,
            "{:?}",
            graph.neighbors(&node2, Direction::Any).collect::<Vec<_>>()
        );
        assert_eq!(
            graph
                .neighbors(&node2, Direction::Outgoing)
                .next()
                .unwrap()
                .1,
            &Edge {
                source: Source::Node(node2),
                target: Target::Output,
                ports: Ports::consecutive(),
                weight: Some(TestWeight(3))
            },
        );
        assert_eq!(graph.output_nodes().count(), 1);

        // try removing an edge
        graph.remove_edge(&first_edge).unwrap();
        assert_eq!(graph.neighbors(&node1, Direction::Outgoing).count(), 0);
        assert_eq!(graph.neighbors(&node2, Direction::Incoming).count(), 0);
        assert_eq!(graph.neighbors(&node1, Direction::Any).count(), 1);
        assert_eq!(graph.neighbors(&node2, Direction::Any).count(), 1);

        // try removing an input edge
        graph.remove_edge(&input_id).unwrap();
        assert_eq!(graph.neighbors(&node1, Direction::Incoming).count(), 0);

        // try removing an output edge
        graph.remove_edge(&output_id).unwrap();
        assert_eq!(graph.neighbors(&node2, Direction::Outgoing).count(), 0);

        // try removing a non-existing edge
        assert!(graph.remove_edge(&EdgeID(10)).is_err());
    }

    #[test]
    fn test_graph_ports_consecutive() {
        let ports = Ports::consecutive();
        assert_eq!(ports.len(), 1);
        assert_eq!(ports[0].source_port, PortID(0));
        assert_eq!(ports[0].target_port, PortID(0));
    }

    #[test]
    fn test_graph_ports() {
        // For some reason, Rust type inference doesn't work with the default type so we have
        // to explicitly specify the type at declaration time.
        let mut graph = Graph::<TestNode, TestWeight, DefaultNodeID>::new();
        let node1 = graph.add_node(TestNode(1)).unwrap();
        let node2 = graph.add_node(TestNode(2)).unwrap();
        graph
            .add_edge(node1, node2, Ports::consecutive(), TestWeight(1))
            .unwrap();
        assert_eq!(graph.neighbors(&node1, Direction::Outgoing).count(), 1);
        assert_eq!(graph.neighbors(&node2, Direction::Incoming).count(), 1);
        let (_edge_id, edge) = graph
            .node_neighbors(&node1, Direction::Outgoing)
            .next()
            .unwrap();
        assert_eq!(edge.ports.len(), 1);
        assert_eq!(edge.ports[0].source_port, 0.into());
        assert_eq!(edge.ports[0].target_port, 0.into());

        // try to add a conflicting port on destination node
        graph
            .add_edge(node1, node2, PortLink::new(1, 0), TestWeight(1))
            .unwrap_err();

        // try to add a consecutive node
        let node3 = graph
            .add_consecutive_node(TestNode(3), Some(node2), None)
            .unwrap();
        let (_edge_id, edge) = graph
            .node_neighbors(&node2, Direction::Outgoing)
            .next()
            .unwrap();
        assert_eq!(edge.ports.len(), 1);
        assert_eq!(edge.ports[0].source_port, PortID(0));
        assert_eq!(edge.ports[0].target_port, PortID(0));

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
        // however, we can mut the edge - this will succeed as it has all the target ports used and consecutive
        graph
            .edge_between_mut(node2, node3, |edge| edge.ports.insert(PortLink::new(1, 1)))
            .unwrap();
        let (_edge_id, edge) = graph
            .node_neighbors(&node2, Direction::Outgoing)
            .next()
            .unwrap();
        assert_eq!(edge.ports.len(), 2, "{:?}", edge.ports);
        assert_eq!(edge.ports[1].source_port, PortID(1));
        assert_eq!(edge.ports[1].target_port, PortID(1));
        // this will fail because we can't add the same edge twice
        graph
            .edge_between_mut(node2, node3, |edge| edge.ports.insert(PortLink::new(1, 1)))
            .unwrap_err();
        // this wwill fail because we can't add a non consecutive target port
        graph
            .edge_between_mut(node2, node3, |edge| edge.ports.insert(PortLink::new(1, 8)))
            .unwrap_err();
    }
}
