use crate::{
    graph::{PortID, Ports, Source, Target},
    layers::{NodeOut, provable::ProvingData},
};
use std::collections::{BTreeMap, HashMap, HashSet};

use anyhow::{Context, Result, anyhow, ensure};
use ff_ext::{ExtensionField, GoldilocksExt2};
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use tenstore::{GenStore, GenericStore};
use trace::Trace;
use tracing::info;

use crate::{
    Shape, Tensor,
    graph::{Direction, Edge, GenericNodeID, Graph, IntoVecUsize, PortLink},
    layers::{
        Layer,
        provable::{Evaluate, OpInfo},
        requant::Requant,
    },
    number::Number,
    padding::PaddingMode,
    quantization::InferenceTracker,
    tensor::DryTensor,
};

mod context;
pub mod llm;
pub(crate) mod trace;
pub mod transform;
pub use context::ModelCtx;
pub use trace::{InferenceStep, InferenceTrace, StepData};

pub type NodeID = crate::graph::DefaultNodeID;
pub type ModelEdge = Edge<NodeID, ()>;

/// Graph of layers. We store no weights on the edges.
/// TODO?: maybe make a graph wrapper that deals with empty weights
pub type ModelGraph<N> = Graph<Layer<N>, (), NodeID>;

/// Represents a model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Model<N> {
    pub(crate) graph: ModelGraph<N>,
    // pub(crate) nodes: HashMap<NodeID, Node<N>>,
    pub(crate) input_shapes: Vec<Shape>,
    pub(crate) unpadded_input_shapes: Vec<Shape>,
}

impl<N> Model<N>
where
    N: Number,
{
    /// Returns an iterator over the nodes in the model, in arbitrary order.
    /// It is more efficient then `ForwardIterator` and `BackwardIterator`, so it
    /// can be used to iterate over the nodes when the order does not matter
    pub fn to_unstable_iterator(&self) -> impl Iterator<Item = (&NodeID, &Layer<N>)> {
        self.graph.nodes()
    }

    /// Utility method to pad the inputs shapes to the next power of two.
    fn compute_padded_input_shapes(unpadded_input_shapes: &[Shape]) -> Vec<Shape> {
        unpadded_input_shapes
            .iter()
            .map(|shape| shape.next_power_of_two())
            .collect()
    }

    /// Instantiate a model with the given input shape: the `padding` input specifies whether
    /// the provided inputs shapes should be padded or not.
    pub fn new_from_input_shapes(unpadded_input_shapes: Vec<Shape>, padding: PaddingMode) -> Self {
        let input_shapes = match padding {
            PaddingMode::NoPadding => unpadded_input_shapes.clone(),
            PaddingMode::Padding => Self::compute_padded_input_shapes(&unpadded_input_shapes),
        };
        Self {
            graph: ModelGraph::new(),
            input_shapes,
            unpadded_input_shapes,
        }
    }

    pub(crate) fn new(
        unpadded_input_shapes: Vec<Shape>,
        padding: PaddingMode,
        nodes: ModelGraph<N>,
    ) -> Self {
        let mut model = Self::new_from_input_shapes(unpadded_input_shapes, padding);
        model.graph = nodes;

        model
    }

    /// Instantiate a model from the set of nodes and the input shapes.
    /// `actual_input_shapes` correspond to the expected shape of the input
    /// tensors for the model; therefore, `actual_input_shapes` can be the same
    /// as `unpadded_input_shapes` if the input tensors of the model are
    /// not expected to be padded
    pub fn new_from_shapes(
        unpadded_input_shapes: Vec<Shape>,
        actual_input_shapes: Vec<Shape>,
        nodes: ModelGraph<N>,
    ) -> Self {
        Self {
            unpadded_input_shapes,
            input_shapes: actual_input_shapes,
            graph: nodes,
        }
    }

    /// Get the shapes of the input tensors, not padded
    pub(crate) fn unpadded_input_shapes(&self) -> Vec<Shape> {
        self.unpadded_input_shapes.clone()
    }

    /// Get the actual input shapes, which could be padded or unpadded
    /// depending on how the model was instantiated
    pub fn input_shapes(&self) -> Vec<Shape> {
        self.input_shapes.clone()
    }

    pub fn num_inputs(&self) -> usize {
        self.input_shapes.len()
    }

    /// Prepare the input tensors to be provided to the model according to the
    /// actual input shapes expected by the model
    pub fn prepare_inputs(&self, inputs: Vec<Tensor<N>>) -> Result<Vec<Tensor<N>>> {
        let input_shapes = self.input_shapes.clone();
        ensure!(
            input_shapes.len() == inputs.len(),
            "Unexpected number of inputs tensors: expected {}, found {}",
            input_shapes.len(),
            inputs.len()
        );
        Ok(inputs
            .into_iter()
            .zip(input_shapes)
            .map(|(mut input, shape)| {
                if input.shape().clone() == shape {
                    // no need to pad, simply return the input
                    input
                } else {
                    input.pad_to_shape(shape);
                    input
                }
            })
            .collect())
    }

    /// iterates over all layers and resets their internal state if any
    pub fn reset(&self) {
        for (_, node) in self.graph.nodes() {
            node.reset();
        }
    }

    /// Build the inputs tensors, according to the expected input shapes,
    /// from a set of flat data
    pub fn load_input_flat(&self, input: Vec<Vec<N>>) -> Result<Vec<Tensor<N>>> {
        let input_tensor = input
            .into_iter()
            .zip(self.unpadded_input_shapes())
            .map(|(inp, shape)| Tensor::new(shape, inp))
            .collect();
        self.prepare_inputs(input_tensor)
    }

    /// Compute the input shapes padded to the next power of two
    pub(crate) fn padded_input_shapes(&self) -> Vec<Shape> {
        Self::compute_padded_input_shapes(&self.unpadded_input_shapes)
    }

    /// Textual description of the model
    pub fn describe(&self) {
        info!("Model description:");
        info!("Unpadded input shapes: {:?}", self.unpadded_input_shapes);
        info!("Padded input shapes: {:?}", self.padded_input_shapes());
        for (id, layer) in self.graph.forward_iter() {
            let edges = self
                .graph
                .node_neighbors(&id, Direction::Any)
                .map(|(_, edge)| edge)
                .collect::<Vec<_>>();
            info!("\t- {}: {}", id, layer.describe());
            info!("\t\t- edges: {:?}", edges);
        }
        info!("Input nodes:");
        for (idx, offset) in self.graph.input_nodes() {
            info!("\t- {}:{:?}", idx, offset);
        }
        info!("Output nodes:");
        for (idx, offset) in self.graph.output_nodes() {
            info!("\t- {}:{:?}", idx, offset);
        }
    }

    /// Add re-quantization nodes to the model after the node with id `input_node_id`
    /// It creates as many requant layers as there are output wires of the input node
    pub(crate) fn add_requant_layer(
        &mut self,
        requants: Vec<Requant>,
        input_node_id: NodeID,
    ) -> anyhow::Result<Vec<NodeID>> {
        ensure!(
            self.graph.node(&input_node_id).is_some(),
            "Node {input_node_id} not found in the model"
        );
        // here we collect port links from the source port, since we add one requant _per source port_ only
        let source_edge_per_requant = self
            .graph
            .neighbors(&input_node_id, Direction::Outgoing)
            .fold(BTreeMap::new(), |mut acc, (_, edge)| {
                for port in edge.ports().iter() {
                    #[allow(clippy::clone_on_copy)]
                    acc.entry(port.source_port.clone())
                        .or_insert(Vec::new())
                        .push((edge.target().clone(), port.target_port.clone()));
                }
                acc
            });
        // enforce one requant per source port
        ensure!(
            source_edge_per_requant.len() == requants.len(),
            "Unexpected number of requants: expected {}, found {}",
            source_edge_per_requant.len(),
            requants.len()
        );
        // we can already delete the outgoing edges from the input node now that we have collected all info necessary
        // to do the link with requants layers
        let edges_to_remove = self
            .graph
            .neighbors(&input_node_id, Direction::Outgoing)
            .map(|(edge_id, _)| *edge_id)
            .collect::<Vec<_>>();
        for edge_id in edges_to_remove {
            self.graph.remove_edge(&edge_id)?;
        }

        let requant_nodes = source_edge_per_requant
            .into_iter()
            .zip(requants.into_iter())
            .map(|((source_port, targets), requant)| {
                // we take the next node id and add + 1. We can't let the graph generate the id since it might
                // generate an id that already exists in the graph since we can generate the graph
                // we specific node ids that might not be consecutive.
                // TODO: maybe make the graph choose it for us?
                let last_node_id = *self.graph.nodes().last().unwrap().0;
                let requant_node_id = NodeID::from(*last_node_id + 1);
                // first add the requant node to be able to reference it later when modifying the edges of the model
                self.graph
                    .add_node_with_id(requant_node_id, Layer::Requant(requant))?;
                // we create this new port link as the edge from input node -> requant
                // Given there is only **one** portlink on **one** edge between input_node_id and this requant, we always
                // set target_port to 0, e.g. first slot.
                self.graph
                    .add_edge(input_node_id, requant_node_id, (*source_port, 0), None)?;
                // we create this new port link as the edge from requant -> output
                // here we wanna take exactly the same as the currently existing ones, as if requant took the place
                // of the input node. Since source port can be connected to multiple target ports and we can only
                // insert a node _once_ then we index by edge_id first.
                let portlinks_by_edge_id =
                    targets
                        .into_iter()
                        .fold(HashMap::new(), |mut acc, (target, target_port)| {
                            acc.entry(target).or_insert(Vec::new()).push(*target_port);
                            acc
                        });
                for (target, target_ports) in portlinks_by_edge_id.into_iter() {
                    // add all the port links from requant -> successor
                    // NOTE: we enuemrate here since a source_port could be 1 but requant port starts at 0. In the end
                    // it doesn't matter which port we put on the requant, as long as the target port is the same as in the
                    // original graph and they're used in the same order.
                    let links = target_ports
                        .iter()
                        // Requant should always have one output port since it comes from a single source port on the node
                        .map(|target_port| (0, *target_port))
                        .collect::<Vec<_>>();
                    let edge =
                        Edge::new(Source::Node(requant_node_id), target.clone(), links, None);
                    self.graph.add_edges_raw(vec![edge])?;
                }
                Ok(requant_node_id)
            })
            .collect::<anyhow::Result<Vec<_>>>()?;
        Ok(requant_nodes)
    }

    pub fn num_outputs(&self, node_id: &NodeID) -> anyhow::Result<usize> {
        let Some(node) = self.graph.node(node_id) else {
            anyhow::bail!("Node {node_id} not found in model");
        };
        // how many targetports are attached to this node, e.g. how many inputs
        // does it receive
        let input_ports = self
            .graph
            .neighbors(node_id, Direction::Incoming)
            .flat_map(|(_, edge)| edge.ports().iter())
            .fold(HashSet::new(), |mut acc, port| {
                acc.insert(port.target_port);
                acc
            })
            .len();
        Ok(node.num_outputs(input_ports))
    }

    /// Corner-case method to add a node whose inputs correspond to the outputs of a node already inserted in the model
    /// The `NodeID` of the already inserted node is the `previous_node_id` input; if no id is provided, it is assumed
    /// that the inputs of the node correspond to the inputs of the model
    pub fn add_consecutive_layer(
        &mut self,
        layer: Layer<N>,
        previous_node_id: Option<NodeID>,
    ) -> anyhow::Result<NodeID> {
        // We need to correctly connect the outputs of the previous node to the inputs of the new node
        // For this we need to know how many outputs the previous node has
        // To know this, we need to count how many target ports are attached to the previous node, so the number
        // of inputs the previous node receives, and then call the `num_outputs` methods with that number.
        let num_outputs = if let Some(id) = &previous_node_id {
            self.num_outputs(id)?
        } else {
            // look at the number of port links from inputs
            self.input_shapes().len()
        };

        let new_node_id = self.graph.add_node(layer)?;
        match previous_node_id {
            Some(id) => {
                // map i-th port of previous node to i-th port of new node
                let links = (0..num_outputs)
                    .map(|i| PortLink::new(i, i))
                    .collect::<Vec<_>>();
                self.graph.add_edge(id, new_node_id, links, ())?;
            }
            None => {
                self.graph.set_input(new_node_id, 0..num_outputs, ())?;
            }
        };
        Ok(new_node_id)
    }
    pub fn add_layer(&mut self, layer: Layer<N>) -> anyhow::Result<NodeID> {
        self.graph.add_node(layer)
    }

    /// Add the node provided as input to the model, binding it to the `node id`
    /// provided as input
    pub fn add_layer_with_id(&mut self, node_id: NodeID, layer: Layer<N>) -> anyhow::Result<()> {
        self.graph.add_node_with_id(node_id, layer)
    }

    pub fn set_input<I: IntoVecUsize>(
        &mut self,
        node_id: NodeID,
        input_index: I,
    ) -> anyhow::Result<()> {
        self.graph.set_input(node_id, input_index, ()).map(|_| ())
    }

    pub fn set_output<I: IntoVecUsize>(
        &mut self,
        node_id: NodeID,
        output_index: I,
    ) -> anyhow::Result<()> {
        self.graph.set_output(node_id, output_index, ()).map(|_| ())
    }

    pub fn add_edge<P: Into<Ports>>(
        &mut self,
        source: NodeID,
        target: NodeID,
        ports: P,
    ) -> anyhow::Result<()> {
        self.graph.add_edge(source, target, ports, ()).map(|_| ())
    }

    pub fn add_raw_edge<S: Into<Source<NodeID>>, T: Into<Target<NodeID>>, P: Into<Ports>>(
        &mut self,
        source: S,
        target: T,
        ports: P,
    ) -> anyhow::Result<()> {
        let portlinks = ports.into();
        // a bit of weirdness when you don't have weights, you still need to specify the type of the weight
        let edge = Edge::new(source, target, portlinks, Option::<()>::None);
        self.graph.add_edges_raw(vec![edge]).map(|_| ())
    }

    /// Shortcut method for onnx to add an edge where the source can be either an input or another node.
    /// onnx doesn't differentiate between the two, e.g. an input also has an ID so we first check
    /// inside the model if the provided source is an input or a node.
    /// TODO/XXX: make that part of graph?
    pub fn add_maybe_input<S: Into<NodeID>, P: Into<Ports>>(
        &mut self,
        source: S,
        target: NodeID,
        ports: P,
    ) -> anyhow::Result<()> {
        let source_id = source.into();
        let input = match self.graph.node(&source_id).is_some() {
            false => Source::Input,
            true => source_id.into(),
        };
        self.add_raw_edge(input, target, ports)
    }

    // This method assumes there is a node without routed output edges, and the outputs of
    // this node will be labelled as the output edges of the model
    pub fn automatic_output_labelling(&mut self) -> Result<()> {
        // find the nodes with no output edges, which will be considered the output nodes
        let out_nodes = self
            .graph
            .nodes()
            .filter_map(|(id, _node)| {
                if self.graph.neighbors(id, Direction::Outgoing).count() == 0 {
                    Some(*id)
                } else {
                    None
                }
            })
            .collect::<Vec<_>>();
        ensure!(!out_nodes.is_empty(), "No output node found for model");
        // for each node, we collect how many outputs it will produce and set corresponding output edges
        let mut output_offset_index = 0;
        for node_id in out_nodes {
            let num_outputs = self.num_outputs(&node_id)?;
            self.graph.set_output(
                node_id,
                output_offset_index..(output_offset_index + num_outputs),
                (),
            )?;
            output_offset_index += num_outputs;
        }
        Ok(())
    }

    pub fn output_nodes(&self) -> impl Iterator<Item = (&NodeID, &Ports)> + use<'_, N> {
        self.graph.output_nodes()
    }

    /// Returns the order the [NodeIDs](NodeID) will be visited in a forward pass
    pub fn eval_order(&self) -> impl Iterator<Item = NodeID> + use<'_, N> {
        self.graph.forward_iter().map(|(id, _)| id)
    }
}

impl Model<f32> {
    pub fn run_float(&self, input: &[Tensor<f32>]) -> anyhow::Result<Vec<Tensor<f32>>> {
        self.run::<GoldilocksExt2>(input, None, &mut GenStore::default())?
            .outputs()
    }
}

impl<N: Number + Serialize + for<'a> Deserialize<'a>> Model<N> {
    pub(crate) fn run_with_tracker<E>(
        &self,
        inputs: &[Tensor<N>],
        unpadded_input_shapes: Option<Vec<Shape>>,
        mut tracker: Option<&mut InferenceTracker>,
        store: &mut GenStore,
    ) -> anyhow::Result<InferenceTrace<'_, E, N>>
    where
        E: ExtensionField + Serialize + DeserializeOwned,
        E::BaseField: Serialize + DeserializeOwned,
        Layer<N>: Evaluate<N>,
    {
        // Collect all the shapes indexed by the tensor key, across the whole inference
        let mut input_shape_register = HashMap::new();
        // collect all input dry tensors
        let mut input_dry_tensors = BTreeMap::new();
        // collect all (Source::Input, source_port) to generate the tensor keys and therefore the dry tensors
        // NOTE: This is currently sufficient because we create keys only from (source, source_port) since
        // that defines uniquely a tensor in the model.
        let source_ports =
            self.graph
                .input_nodes()
                .fold(HashSet::new(), |mut acc, (_id, ports)| {
                    // we want to create one key per input so we make sure we don't have duplicates using the hashmap.
                    // duplicates can occur if an input is used in multiple target ports/node.
                    for port in ports.iter() {
                        acc.insert(port.source_port);
                    }
                    acc
                });
        // now for each source port, we collect all the input tensors and store them in the store and the trace
        // TODO: iterator-ize this loop - borrow checker is unhappy currently
        for source_port in source_ports.into_iter() {
            let input_shape = inputs[*source_port].shape().clone();
            // re-create artificially the edge to derive its tensor keys, one key per source port
            // NOTE: see `to_tensor_keys` for more info but basically:
            // * we use DefaultNodeID(0) since that is the _target node id_ and it's not used to derive the tensor key
            // * 0 as target port is ok since it's not used to derive the tensor key
            let edge = Edge::<NodeID, ()>::input(NodeID::from(0), (*source_port, 0), None);
            // unwrap is safe since there is at least one key since we gave one port link
            let tensor_key = edge.to_tensor_keys().into_iter().next().unwrap().1;
            // save the key => shape relation
            input_shape_register.insert(tensor_key.clone(), input_shape.clone());
            // save the tensor to the store
            store.store(&tensor_key, inputs[*source_port].data_vec())?;
            // save the input dry tensors to store in the trace
            input_dry_tensors.insert(
                source_port,
                DryTensor::new(tensor_key.clone(), input_shape.clone()),
            );
        }
        let input_dry_tensors = input_dry_tensors.into_values().collect::<Vec<_>>();
        let mut trace = Trace {
            store: store.clone(),
            steps: HashMap::new(),
            input: input_dry_tensors,
            output: vec![],
        };
        let iter = self.graph.forward_iter();

        for (node_id, node) in iter {
            // need to collect all the inputs keys and shapes to run the next node
            let (unpadded_input_shapes, input_keys): (Vec<_>, Vec<_>) = self
                .graph
                .neighbors(&node_id, Direction::Incoming)
                .try_fold(BTreeMap::new(), |mut acc, (_id, edge)| {
                    let keys = edge.to_tensor_keys();
                    // if it is an edge between nodes,fetch the input shape from the trace
                    let shapes_array = if let Some(source_id) = edge.source_id() {
                        let Some(step) = trace.get_step(source_id) else {
                            anyhow::bail!("Node {source_id} not found in trace as predecessors of node {node_id}");
                        };
                        &step.step_data.unpadded_output_shapes
                    } else {
                        // otherwise fetch the input shape from the model or arg
                        &unpadded_input_shapes
                            .as_ref()
                            .unwrap_or(&self.unpadded_input_shapes())
                            .clone()
                    };
                    for link in edge.ports().iter() {
                        // get the associated key from this link
                        let key = keys.get(link).ok_or(anyhow!("Key for link {link:?} not found"))?.clone();
                        // then associate it with the shape (note we take the output port of the predecessor)
                        let shape = shapes_array[*link.source_port].clone();
                        // we collect all that and sort it by the target_port, since this is what defines the natural
                        // order of the inputs expected by the layer
                        acc.insert(link.target_port, (shape,key));
                    }
                    Ok(acc)
                })?.into_values().unzip();
            // now we can run the node
            let (mut output_per_port, proving_data) = self
                .run_layer(
                    node_id,
                    input_keys.as_slice(),
                    &unpadded_input_shapes,
                    &input_shape_register,
                    &mut tracker,
                    store,
                )
                .context(format!("Error occurred at node ID: {node_id}"))?;
            // even if the outputs are supposedly already sorted by the layers, better be safe.
            output_per_port.sort_by_key(|(source_port, _)| *source_port);
            // update the shape register with the shapes of the outputs of the node so next node can load its input from the input shape register
            input_shape_register.extend(output_per_port.iter().map(|(_, dry_tensor)| {
                (dry_tensor.key().to_owned(), dry_tensor.shape().to_owned())
            }));
            // Record the step into the trace
            let new_step = StepData {
                node_inputs: input_keys
                    .iter()
                    .map(|k| DryTensor::new(k.clone(), input_shape_register[k].clone()))
                    .collect(),
                node_outputs: NodeOut::new(
                    output_per_port
                        .into_iter()
                        .map(|(_, dry_tensor)| dry_tensor)
                        .collect(),
                    proving_data,
                ),
                unpadded_output_shapes: node
                    .output_shapes(&unpadded_input_shapes, PaddingMode::NoPadding),
                unpadded_input_shapes,
            };
            trace.new_step(
                node_id,
                InferenceStep {
                    op: node,
                    step_data: new_step,
                },
            );
        }

        // compute the output tensor from the outputs of the output nodes
        let output_nodes = self.graph.output_nodes();
        let mut outputs = BTreeMap::<usize, DryTensor<N>>::new();
        for (id, ports) in output_nodes {
            let node_outputs = trace
                .get_step(id)
                .ok_or(anyhow!("Output node {id} not found in trace"))?
                .outputs();
            ensure!(
                node_outputs.len() >= ports.len(),
                "Number of outputs found in trace ({}) for node {id} is smaller from number of expected outputs ({})",
                node_outputs.len(),
                ports.len()
            );
            for link in ports.iter() {
                // if this output wire is an output of the model, insert in the collection of the
                // model outputs, paired with the index among the outputs of the model
                ensure!(
                    outputs
                        .insert(
                            link.target_port.into(),
                            node_outputs[*link.source_port].clone()
                        )
                        .is_none(),
                    "Trying to insert twice an output value for the same index {}",
                    link.target_port
                );
            }
        }
        ensure!(
            *outputs.first_key_value().unwrap().0 == 0
                && *outputs.last_key_value().unwrap().0 == outputs.len() - 1
        );

        trace.output = outputs.into_values().collect();

        Ok(trace)
    }

    /// Run the inference of the model, producing the `InferenceTrace` necessary to
    /// later prove the model. The outputs of the model can be fetched from the returned
    /// trace
    pub fn run<E>(
        &self,
        input: &[Tensor<N>],
        unpadded_input_shapes: Option<Vec<Shape>>,
        store: &mut GenStore,
    ) -> anyhow::Result<InferenceTrace<'_, E, N>>
    where
        E::BaseField: Serialize + DeserializeOwned,
        E: ExtensionField + Serialize + DeserializeOwned,
        Layer<N>: Evaluate<N>,
    {
        self.run_with_tracker(input, unpadded_input_shapes, None, store)
    }

    #[allow(clippy::type_complexity)]
    pub(crate) fn run_layer<E: ExtensionField>(
        &self,
        node_id: NodeID,
        // inputs are assumed to be sorted by source port, e.g. in the order defined by the ports
        inputs: &[TensorKey<N>],
        unpadded_input_shapes: &[Shape],
        input_shape_register: &HashMap<TensorKey<N>, Shape>,
        tracker: &mut Option<&mut InferenceTracker>,
        store: &mut GenStore,
        // all outputs are associated with the corresponding source port of outgoing edges, e.g. the "output port"
    ) -> Result<(Vec<(PortID, DryTensor<N>)>, ProvingData<E>)>
    where
        N: Number,
        Layer<N>: Evaluate<N>,
    {
        let input_tensors = inputs
            .iter()
            .map(|key| {
                let data = store
                    .fetch(key)
                    .with_context(|| format!("fetching tensor data for tensor {key}"))
                    .unwrap();
                Ok(Tensor::new(input_shape_register[key].clone(), data))
            })
            .collect::<anyhow::Result<Vec<Tensor<N>>>>()?;
        let input_tensors_ref = input_tensors.iter().collect::<Vec<_>>();
        let layer = &self.graph[node_id];
        let expected_num_outputs = layer.num_outputs(input_tensors.len());

        // run the layer
        let layer_out = layer.evaluate(&input_tensors_ref, unpadded_input_shapes)?;
        assert!(expected_num_outputs == layer_out.outputs.len());

        // the keys under which to save the output tensor of this layer. We save one tensor
        // per source port so we index the output edges by the source port of the current node
        let output_keys = self.graph.neighbors(&node_id, Direction::Outgoing).fold(
            BTreeMap::new(),
            |mut acc, (_id, edge)| {
                // NOTE: it's ok to overwrite previous keys since they're from the same source port.
                // currently `to_tensor_keys` returns a unique key per source port, so it's safe to overwrite
                // in the case one source port is connected to multiple target port.
                for (link, key) in edge.to_tensor_keys() {
                    acc.insert(link.source_port, key);
                }
                acc
            },
        );
        ensure!(
            output_keys.len() >= expected_num_outputs,
            "Number of output keys ({}) does not match expected number of outputs ({}) for node {node_id}: {},\n: input shapes are {:?}.
            Check the output nodes are not set up correctly, one output port per output of the node.",
            output_keys.len(),
            self.graph.node(&node_id).unwrap().describe(),
            expected_num_outputs,
            unpadded_input_shapes
        );
        // store each output to the store and return the corresponding dry tensor
        let outputs = output_keys
            .iter()
            .map(|(source_port, key)| {
                let tensor = &layer_out.outputs[*source_port];
                store
                    .store(key, tensor.data_vec())
                    .with_context(|| format!("storing outputs for tensor {key}"))?;
                Ok((
                    *source_port,
                    DryTensor::new(key.clone(), tensor.shape().clone()),
                ))
            })
            .collect::<Result<Vec<_>>>()?;

        // add output tensors to tracker, if any
        if let Some(tracker) = tracker.as_mut() {
            for (source_port, _) in output_keys.iter() {
                tracker.track(
                    node_id,
                    **source_port,
                    layer_out.outputs[*source_port].to_f32()?,
                );
            }
            // track intermediate data, if any
            if let Some(tracked_data) = layer_out.tracked_layer_data {
                for (data_id, data) in tracked_data {
                    tracker.track_intermediate_data(node_id, data_id, data.to_f32()?);
                }
            }
        }

        Ok((outputs, layer_out.proving_data))
    }
}

use tenstore::TensorKey;
trait ToTensorKeys<N> {
    /// Returns a list of pairs containing the key and the corresponding source port.
    /// Since every tensor key is associated with a node id and a source port, this enables to
    /// precisely map which tensor key corresponds to which output port of a node.
    fn to_tensor_keys(&self) -> HashMap<PortLink, TensorKey<N>>;
}

impl<N, I: GenericNodeID, W> ToTensorKeys<N> for Edge<I, W> {
    /// Uniquely identifies any tensor by a key. Currently the key is set to be
    /// unique "per source" since we only want to save _one_ tensor per source port.
    /// So there is one key per source port, e.g. one key per output tensor of the source node of this edge.
    /// NOTE: ideally we would be able to save one tensor under aliases so we could uniquely identify
    /// any tensor by the tuple (source, source_port, target, target_port). Since a tensor
    /// can be used in multiple target_port we don't want to save it multiple times tho. But using
    /// this tuple we can quickly identify exactly where the tensor is in the graph.
    fn to_tensor_keys(&self) -> HashMap<PortLink, TensorKey<N>> {
        self.ports().iter().fold(HashMap::new(), |mut acc, link| {
            acc.insert(
                link.clone(),
                TensorKey::from_str(format!("Source[{}({})]", self.source(), link.source_port)),
            );
            acc
        })
    }
}

#[cfg(test)]
pub(crate) mod test {
    use crate::{
        Prover, ScalingFactor, ScalingStrategy, Shape, init_test_logging,
        init_test_logging_default,
        layers::{
            Layer,
            activation::Activation,
            convolution::{ConvCtx, Convolution},
            dense::Dense,
            matrix_mul::{Config, MatMul, OperandMatrix},
            pooling::{MAXPOOL2D_KERNEL_SIZE, Maxpool2D, Pooling},
            provable::{OpInfo, evaluate_layer},
            requant::Requant,
        },
        number::Number,
        padding::{PaddingMode, pad_model},
        quantization::{self, InferenceObserver},
        rng_from_env_or_random,
        tensor::KeyedTensor,
        testing::{Pcs, random_bool_vector, random_vector},
        util::from_mle_list_dimensions,
        verify,
    };
    use anyhow::{Ok, Result};
    use ark_std::rand::{Rng, RngCore};
    use either::Either;
    use ff_ext::GoldilocksExt2;
    use itertools::Itertools;
    use multilinear_extensions::{mle::IntoMLE, virtual_polys::VirtualPolynomialsBuilder};
    use sumcheck::{
        structs::{IOPProverState, IOPVerifierState},
        util::optimal_sumcheck_threads,
    };
    use tenstore::GenStore;
    use transcript::BasicTranscript;

    use super::Model;
    use crate::{Element, default_transcript, tensor::Tensor};

    pub type F = GoldilocksExt2;
    const SELECTOR_DENSE: usize = 0;
    const SELECTOR_RELU: usize = 1;
    const SELECTOR_POOLING: usize = 2;
    const MOD_SELECTOR: usize = 2;

    impl Model<Element> {
        pub fn random(num_dense_layers: usize) -> Result<(Self, Vec<Tensor<Element>>)> {
            let mut rng = rng_from_env_or_random();
            Self::random_with_rng(num_dense_layers, &mut rng)
        }
        /// Returns a random model with specified number of dense layers and a matching input.
        /// Note that currently everything is considered padded, e.g. unpadded_shape = padded_shape
        pub fn random_with_rng<R: RngCore>(
            num_dense_layers: usize,
            rng: &mut R,
        ) -> Result<(Self, Vec<Tensor<Element>>)> {
            let mut last_row: usize = rng.gen_range(3..15);
            let mut model = Self::new_from_input_shapes(
                vec![vec![last_row.next_power_of_two()].into()],
                PaddingMode::NoPadding,
            );

            let mut last_node_id = None;
            for selector in 0..num_dense_layers {
                if selector % MOD_SELECTOR == SELECTOR_DENSE {
                    // if true {
                    // last row becomes new column
                    let (nrows, ncols): (usize, usize) = (rng.gen_range(3..15), last_row);
                    last_row = nrows;
                    let dense = Dense::random(
                        vec![nrows.next_power_of_two(), ncols.next_power_of_two()].into(),
                        Some(format!("dense_{selector}").into()),
                    );
                    // Figure out the requant information such that output is still within range
                    let (min_output_range, max_output_range) =
                        dense.output_range(*quantization::MIN, *quantization::MAX);
                    let output_scaling_factor = ScalingFactor::from_scale(
                        ((max_output_range - min_output_range) as f64
                            / (*quantization::MAX - *quantization::MIN) as f64)
                            as f32,
                        None,
                    );
                    let input_scaling_factor = ScalingFactor::from_scale(1.0, None);
                    let max_model = dense.matrix.max_value().max(
                        dense
                            .bias
                            .as_ref()
                            .map(|b| b.max_value())
                            .unwrap_or(f32::MIN as i64),
                    ) as f32;
                    let model_scaling_factor = ScalingFactor::from_absolute_max(max_model, None);

                    let intermediate_bit_size = dense.output_bitsize();
                    let requant = Requant::from_scaling_factors(
                        input_scaling_factor,
                        model_scaling_factor,
                        output_scaling_factor,
                        intermediate_bit_size,
                    );

                    last_node_id =
                        Some(model.add_consecutive_layer(Layer::Dense(dense), last_node_id)?);
                    last_node_id =
                        Some(model.add_consecutive_layer(Layer::Requant(requant), last_node_id)?);
                } else if selector % MOD_SELECTOR == SELECTOR_RELU {
                    last_node_id = Some(model.add_consecutive_layer(
                        Layer::Activation(Activation::new_relu()),
                        last_node_id,
                    )?);
                    // no need to change the `last_row` since RELU layer keeps the same shape
                    // of outputs
                } else if selector % MOD_SELECTOR == SELECTOR_POOLING {
                    // Currently unreachable until Model is updated to work with higher dimensional tensors
                    // TODO: Implement higher dimensional tensor functionality.
                    last_node_id = Some(model.add_consecutive_layer(
                        Layer::Pooling(Pooling::Maxpool2D(Maxpool2D::default())),
                        last_node_id,
                    )?);
                    last_row -= MAXPOOL2D_KERNEL_SIZE - 1;
                } else {
                    panic!("random selection shouldn't be in that case");
                }
            }
            model.automatic_output_labelling().unwrap();
            let inputs = model.input_shapes().iter().map(Tensor::random).collect();
            Ok((model, inputs))
        }

        /// Returns a model that only contains pooling and relu layers.
        /// The output [`Model`] will contain `num_layers` [`Maxpool2D`] layers and a [`Dense`] layer as well.
        pub fn random_pooling(num_layers: usize) -> Result<(Self, Vec<Tensor<Element>>)> {
            let mut rng = rng_from_env_or_random();
            // Since Maxpool reduces the size of the output based on the kernel size and the stride we need to ensure that
            // Our starting input size is large enough for the number of layers.

            // If maxpool input matrix has dimensions w x h then output has width and height
            // out_w = (w - kernel_size) / stride + 1
            // out_h = (h - kernel_size) / stride + 1
            // Hence to make sure we have a large enough tensor for the last step
            // we need to have that w_first > 2^{num_layers + 1} + 2^{num_layers}
            // and likewise for h_first.

            let minimum_initial_size = (1 << num_layers) * (3usize);

            let mut input_shape = (0..3)
                .map(|i| {
                    if i < 1 {
                        rng.gen_range(1..5usize).next_power_of_two()
                    } else {
                        (minimum_initial_size + rng.gen_range(1..4usize)).next_power_of_two()
                    }
                })
                .collect::<Shape>();

            let mut model =
                Model::new_from_input_shapes(vec![input_shape.clone()], PaddingMode::NoPadding);

            let inputs = model.input_shapes().iter().map(Tensor::random).collect();

            let info = Maxpool2D::default();
            let mut last_node_id = None;
            for _ in 0..num_layers {
                input_shape
                    .iter_mut()
                    .skip(1)
                    .for_each(|dim| *dim = (*dim - info.kernel_size) / info.stride + 1);
                last_node_id = Some(model.add_consecutive_layer(
                    Layer::Pooling(Pooling::Maxpool2D(info)),
                    last_node_id,
                )?);
            }

            let (nrows, ncols): (usize, usize) =
                (rng.gen_range(3..15), input_shape.iter().product::<usize>());

            model.add_consecutive_layer(
                Layer::Dense(Dense::random(
                    vec![nrows.next_power_of_two(), ncols.next_power_of_two()].into(),
                    None,
                )),
                last_node_id,
            )?;

            model.automatic_output_labelling()?;

            Ok((model, inputs))
        }
    }

    #[test]
    fn test_model_long() {
        let (model, input) = Model::random(3).unwrap();
        model
            .run::<F>(&input, None, &mut Default::default())
            .unwrap();
    }

    fn random_vector_quant(n: usize) -> Vec<Element> {
        random_vector(n)
    }

    #[test]
    fn test_conv_maxpool() {
        let input_shape: Shape = vec![3usize, 32, 32].into();
        let shape1: Shape = vec![6, 3, 5, 5].into();
        let filter = KeyedTensor::new("conv_filter", Tensor::random(&shape1));
        let bias1 = KeyedTensor::new("conv_bias", Tensor::random(&vec![shape1[0]].into()));

        let mut model =
            Model::new_from_input_shapes(vec![input_shape.clone()], PaddingMode::Padding);
        let conv_layer = model
            .add_consecutive_layer(
                Layer::Convolution(
                    Convolution::new(filter.clone(), bias1.clone()).prepared_for_fft(&input_shape),
                ),
                None,
            )
            .unwrap();
        let _pool_layer = model
            .add_consecutive_layer(
                Layer::Pooling(Pooling::Maxpool2D(Maxpool2D::default())),
                Some(conv_layer),
            )
            .unwrap();
        model.automatic_output_labelling().unwrap();

        // TODO: have a "builder" for the model that automatically tracks the shape after each layer such that
        // we can just do model.prepare_input(&input).
        // Here is not possible since we didnt run through the onnx loader
        let input = Tensor::random(&input_shape);
        let input_padded = model.prepare_inputs(vec![input]).unwrap();
        let _ = model
            .run::<F>(&input_padded, None, &mut Default::default())
            .unwrap();
    }

    #[test]
    fn test_model_manual_run() {
        let dense1 = Dense::<Element>::random(
            vec![10usize.next_power_of_two(), 11usize.next_power_of_two()].into(),
            Some("dense_1".to_string().into()),
        );
        let dense2 = Dense::<Element>::random(
            vec![
                7usize.next_power_of_two(),
                dense1.ncols().next_power_of_two(),
            ]
            .into(),
            Some("dense_2".to_string().into()),
        );
        let input_shape = vec![dense1.ncols()].into();
        let input = Tensor::<Element>::random(&input_shape);
        let output1 = evaluate_layer::<GoldilocksExt2, _, _>(&dense1, &[&input], None)
            .unwrap()
            .outputs()[0]
            .clone();
        let final_output = evaluate_layer::<GoldilocksExt2, _, _>(&dense2, &[&output1], None)
            .unwrap()
            .outputs()[0]
            .clone();

        let mut model =
            Model::<Element>::new_from_input_shapes(vec![input_shape], PaddingMode::NoPadding);
        let first_id = model
            .add_consecutive_layer(Layer::Dense(dense1.clone()), None)
            .unwrap();
        let second_id = model
            .add_consecutive_layer(Layer::Dense(dense2.clone()), Some(first_id))
            .unwrap();
        model.automatic_output_labelling().unwrap();

        let mut store = GenStore::default();
        let trace = model.run::<F>(&[input], None, &mut store).unwrap();
        assert_eq!(trace.steps.len(), 2);
        // Verify first step

        assert_eq!(
            trace
                .get_step(&first_id)
                .unwrap()
                .step_data
                .output_tensor_at(0, &mut store)
                .unwrap(),
            output1
        );

        // Verify second step
        assert_eq!(
            trace
                .get_step(&second_id)
                .unwrap()
                .step_data
                .output_tensor_at(0, &mut store)
                .unwrap(),
            final_output.clone()
        );
        let (nrow, _) = (dense2.nrows(), dense2.ncols());
        assert_eq!(final_output.get_data().len(), nrow);
    }

    #[test]
    fn test_model_sequential() {
        let (model, input) = Model::random(1).unwrap();
        model.describe();
        let trace = model
            .run::<F>(&input, None, &mut Default::default())
            .unwrap()
            .into_fields()
            .unwrap();
        let mut store = trace.store.clone();
        let dense_layers = model
            .to_unstable_iterator()
            .flat_map(|(id, l)| match l {
                Layer::Dense(ref dense) => Some((*id, dense.clone())),
                _ => None,
            })
            .collect_vec();
        let matrices_mle = dense_layers
            .iter()
            .map(|(id, d)| (*id, d.matrix.to_2d_mle::<F>()))
            .collect_vec();
        assert_eq!(dense_layers.len(), 1);
        let point1 = random_bool_vector(dense_layers[0].1.nrows().ilog2() as usize);
        let computed_eval1 = trace
            .get_step(&dense_layers[0].0)
            .unwrap_or_else(|| panic!("Node with id {} not found", dense_layers[0].0))
            .step_data
            .output_tensor_at(0, &mut store)
            .unwrap()
            .get_data()
            .to_vec()
            .into_mle()
            .evaluate(&point1);
        let flatten_mat1 = matrices_mle[0].1.fix_high_variables(&point1);
        let bias_eval = dense_layers[0]
            .1
            .bias
            .as_ref()
            .unwrap() // safe because we know there is a bias
            .to_field::<F>()
            .into_mle()
            .evaluate(&point1);
        let computed_eval1_no_bias = computed_eval1 - bias_eval;
        let input_vector = trace.input_at(0).unwrap();
        // since y = SUM M(j,i) x(i) + B(j)
        // then
        // y(r) - B(r) = SUM_i m(r,i) x(i)
        let input_mle = input_vector.get_data().to_vec().into_mle();

        let num_vars = flatten_mat1.num_vars();
        let num_threads = optimal_sumcheck_threads(num_vars);
        let mut expr_builder = VirtualPolynomialsBuilder::<E>::new(num_threads, num_vars);
        let expr = expr_builder.lift(Either::Left(&flatten_mat1))
            * expr_builder.lift(Either::Left(&input_mle));
        let virtual_poly = expr_builder.to_virtual_polys(&[expr], &[]);
        let (proof, _state) = IOPProverState::prove(virtual_poly, &mut default_transcript());

        let given_eval1 = proof.extract_sum();

        assert_eq!(computed_eval1_no_bias, given_eval1);

        let aux_info = from_mle_list_dimensions(&[vec![num_vars, num_vars]]);
        let _subclaim = IOPVerifierState::<F>::verify(
            computed_eval1_no_bias,
            &proof,
            &aux_info,
            &mut default_transcript(),
        );
    }

    #[test]
    #[ignore = "This test should be deleted since there is no requant and it is not testing much"]
    fn test_single_matvec_prover() {
        let mut store = GenStore::default();
        let w1 = random_vector_quant(1024 * 1024);
        let conv1 = KeyedTensor::new(
            "matvec_weight",
            Tensor::new(vec![1024, 1024].into(), w1.clone()),
        );
        let w2 = random_vector_quant(1024);
        let conv2 = KeyedTensor::new("matvec_bias", Tensor::new(vec![1024].into(), w2.clone()));
        let input_shape = vec![1024].into();

        let mut model = Model::new_from_input_shapes(vec![input_shape], PaddingMode::Padding);
        let input = Tensor::random(&model.input_shapes()[0]);
        model
            .add_consecutive_layer(Layer::Dense(Dense::new(conv1, conv2)), None)
            .unwrap();
        model.automatic_output_labelling().unwrap();
        model.describe();
        let trace = model.run::<F>(&[input], None, &mut store).unwrap();
        let mut tr: BasicTranscript<F> = BasicTranscript::new(b"m2vec");
        let (prover_ctx, verifier_ctx) = model
            .generate_contexts::<F, Pcs<F>>()
            .expect("Unable to generate contexts");
        let io = trace.to_verifier_io().unwrap();
        let prover = Prover::new(&prover_ctx, &mut tr);
        let proof = prover.prove(&trace).expect("unable to generate proof");
        let mut verifier_transcript = BasicTranscript::new(b"m2vec");
        verify::<_, _, _>(&verifier_ctx, proof, io, &mut verifier_transcript).unwrap();
    }

    #[test]
    fn test_single_matmul_prover() {
        // layer matrix shape
        let m_shape: Shape = vec![100, 200].into();
        let m = random_vector_quant(m_shape[0] * m_shape[1]);
        let tensor_m = KeyedTensor::new("matmul_weight", Tensor::new(m_shape, m));
        let input_shape: Shape = vec![5, tensor_m.nrows_2d()].into();
        let mut model =
            Model::new_from_input_shapes(vec![input_shape.clone()], PaddingMode::Padding);
        let matmul_layer = MatMul::new(
            OperandMatrix::Input,
            OperandMatrix::new_weight_matrix(tensor_m),
        )
        .unwrap();
        let padded_layer = matmul_layer.pad_next_power_of_two().unwrap();
        model
            .add_consecutive_layer(Layer::MatMul(padded_layer), None)
            .unwrap();
        model.automatic_output_labelling().unwrap();
        model.describe();

        let input = random_vector_quant(input_shape[0] * input_shape[1]);
        let input_tensor = model
            .prepare_inputs(vec![Tensor::new(input_shape, input)])
            .unwrap();

        let mut store = GenStore::default();
        let trace = model.run::<F>(&input_tensor, None, &mut store).unwrap();
        let mut tr = BasicTranscript::<F>::new(b"matmul");
        let (prover_ctx, verifier_ctx) = model
            .generate_contexts::<F, Pcs<F>>()
            .expect("Unable to generate contexts");
        let io = trace.to_verifier_io().unwrap();
        let prover = Prover::new(&prover_ctx, &mut tr);
        let proof = prover.prove(&trace).expect("unable to generate proof");
        let mut verifier_transcript = BasicTranscript::<F>::new(b"matmul");
        verify::<_, _, _>(&verifier_ctx, proof, io, &mut verifier_transcript).unwrap();
    }

    #[test]
    fn test_single_cnn_prover() {
        let n_w = 1 << 2;
        let k_w = 1 << 4;
        let n_x = 1 << 5;
        let k_x = 1 << 1;

        let in_dimensions: Vec<Vec<usize>> =
            vec![vec![k_x, n_x, n_x], vec![16, 29, 29], vec![4, 26, 26]];

        let conv1 = KeyedTensor::new(
            "conv_filter",
            Tensor::random(&vec![k_w, k_x, n_w, n_w].into()),
        );
        let input_shape = vec![k_x, n_x, n_x].into();

        let mut model = Model::new_from_input_shapes(vec![input_shape], PaddingMode::Padding);
        let input = Tensor::random(&model.input_shapes()[0]);
        let bias = KeyedTensor::new("conv_bias", Tensor::random(&vec![conv1.dim(0)].into()));
        let conv_layer = Convolution::new(conv1.clone(), bias.clone())
            .prepared_for_fft(&in_dimensions[0].clone().into());
        let conv_layer_id = model
            .add_consecutive_layer(Layer::Convolution(conv_layer.clone()), None)
            .unwrap();

        assert_eq!(
            conv_layer.conv_context(conv_layer_id),
            ConvCtx {
                node_id: conv_layer_id,
                kw: 16,
                kx: 2,
                real_nw: 4,
                nw: 32,
                filter_size: 1024,
                unpadded_filter_shape: Shape::new(vec![16, 2, 4, 4]),
                padded_filter_shape: Shape::new(vec![16, 2, 4, 4]),
                filter_key: conv1.key(),
                bias_key: bias.key(),
            },
        );

        model.automatic_output_labelling().unwrap();
        model.describe();
        let mut store = GenStore::default();
        let trace = model.run::<F>(&[input], None, &mut store).unwrap();
        let mut tr: BasicTranscript<GoldilocksExt2> = BasicTranscript::new(b"m2vec");
        let (prover_ctx, verifier_ctx) = model
            .generate_contexts::<F, Pcs<F>>()
            .expect("Unable to generate contexts");

        let io = trace.to_verifier_io().unwrap();

        let prover: Prover<'_, '_, GoldilocksExt2, BasicTranscript<GoldilocksExt2>, _> =
            Prover::new(&prover_ctx, &mut tr);
        let proof = prover.prove(&trace).expect("unable to generate proof");

        let mut verifier_transcript: BasicTranscript<GoldilocksExt2> =
            BasicTranscript::new(b"m2vec");
        verify::<_, _, _>(&verifier_ctx, proof, io, &mut verifier_transcript).unwrap();
    }

    type E = GoldilocksExt2;
    type T = BasicTranscript<GoldilocksExt2>;
    type N = Element;

    fn build_test_model<N: Number, const INPUT_SIZE: usize>() -> Model<N> {
        let input_shape: Shape = vec![INPUT_SIZE].into();
        let mut model =
            Model::<N>::new_from_input_shapes(vec![input_shape.clone()], PaddingMode::NoPadding);
        // add input dense layer
        // generate random dense matrix
        let ncols = input_shape[0];
        let nrows = 42;
        let dense = Dense::random(
            vec![nrows, ncols].into(),
            Some("dense_1".to_string().into()),
        );
        let dense_out_shape =
            &dense.output_shapes(&model.unpadded_input_shapes(), PaddingMode::NoPadding)[0];
        let input_node = model
            .add_consecutive_layer(
                Layer::Dense(dense),
                None, // it's connected to the inputs of the model
            )
            .unwrap();
        // add activation layer
        let relu = Activation::new_relu();
        let relu_node = model
            .add_consecutive_layer(Layer::Activation(relu), Some(input_node))
            .unwrap();
        // add another dense layer as output
        let nrows = 37;
        let ncols = dense_out_shape[0]; // it's a vector, so it has only one dimension
        let dense = Dense::random(
            vec![nrows, ncols].into(),
            Some("dense_2".to_string().into()),
        );
        let output_node = model
            .add_consecutive_layer(Layer::Dense(dense), Some(relu_node))
            .unwrap();
        model.automatic_output_labelling().unwrap();

        assert_eq!(*model.output_nodes().next().unwrap().0, output_node);

        model
    }

    #[test]
    fn test_model_inference() {
        const INPUT_SIZE: usize = 45;
        let model = build_test_model::<N, INPUT_SIZE>();
        let input_shape = model.input_shapes()[0].clone();

        let input = random_vector(input_shape.iter().product());
        let input_tensor = Tensor::new(input_shape, input);
        let trace = model
            .run::<E>(&[input_tensor], None, &mut Default::default())
            .unwrap();
        assert_eq!(trace.steps.len(), 3);
    }

    #[test]
    fn test_model_float_inference() {
        const INPUT_SIZE: usize = 45;
        let model = build_test_model::<f32, INPUT_SIZE>();
        let input_shape = model.input_shapes()[0].clone();

        let input_tensor = Tensor::random(&input_shape);
        let trace = model
            .run::<E>(&[input_tensor], None, &mut Default::default())
            .unwrap();
        assert_eq!(trace.steps.len(), 3);
    }

    // Quantize and run a model over the given input, if any; returns the quantized model and the
    // quantized inputs; if `represantive_inputs` are provided, they are going to be employed to
    // compute scaling factors for quantization, otherwise, random data will be employed
    pub(crate) fn quantize_model(
        model: Model<f32>,
        float_inputs: Vec<Tensor<f32>>,
        representative_inputs: Option<Vec<Tensor<f32>>>,
        store: &mut GenStore,
    ) -> anyhow::Result<(Model<Element>, Vec<Tensor<Element>>)> {
        let (quantized_model, md) = if let Some(repr_inputs) = representative_inputs {
            InferenceObserver::new_with_representative_input(vec![
                repr_inputs
                    .iter()
                    .map(|input| input.get_data().to_vec())
                    .collect(),
            ])
        } else {
            InferenceObserver::new()
        }
        .quantize(model, store)?;

        // quantize input tensor
        let input_tensors = float_inputs
            .into_iter()
            .zip(&md.input)
            .map(|(tensor, s)| tensor.to_quantized(s))
            .collect_vec();

        Ok((quantized_model, input_tensors))
    }

    pub(crate) fn prove_quantized_model(
        model: Model<Element>,
        inputs: Vec<Tensor<Element>>,
        store: &mut GenStore,
    ) -> anyhow::Result<Vec<Tensor<Element>>> {
        let model = pad_model(model)?;

        model.describe();

        let input_tensors = model.prepare_inputs(inputs).unwrap();

        let trace = model.run(&input_tensors, None, store)?;
        let mut tr: BasicTranscript<GoldilocksExt2> = BasicTranscript::new(b"model");
        let (prover_ctx, verifier_ctx) = model
            .generate_contexts::<F, Pcs<F>>()
            .expect("Unable to generate contexts");
        let prover: Prover<'_, '_, E, T, _> = Prover::new(&prover_ctx, &mut tr);
        let io = trace.to_verifier_io().unwrap();
        let outputs = trace.outputs();
        let proof = prover.prove(&trace).expect("unable to generate proof");
        let mut verifier_transcript: BasicTranscript<GoldilocksExt2> =
            BasicTranscript::new(b"model");
        verify::<_, _, _>(&verifier_ctx, proof, io, &mut verifier_transcript)?;
        outputs
    }

    pub(crate) fn prove_model_with(
        model: Model<f32>,
        float_inputs: Vec<Tensor<f32>>,
        store: &mut GenStore,
    ) -> anyhow::Result<Vec<Tensor<Element>>> {
        let (quantized_model, quantized_inputs) = quantize_model(model, float_inputs, None, store)?;
        println!("QUANTIZED MODEL: {:?}", quantized_model.describe());
        prove_quantized_model(quantized_model, quantized_inputs, store)
    }

    pub(crate) fn prove_model(
        model: Model<f32>,
        store: &mut GenStore,
    ) -> anyhow::Result<Vec<Tensor<Element>>> {
        let float_inputs = model
            .input_shapes()
            .into_iter()
            .map(|shape| Tensor::random(&shape))
            .collect_vec();
        prove_model_with(model, float_inputs, store)
    }

    #[test]
    fn test_model_proving() {
        init_test_logging_default();
        const INPUT_SIZE: usize = 57;
        let model = build_test_model::<f32, INPUT_SIZE>();
        prove_model(model, &mut Default::default()).unwrap();
    }

    /// 2 relus connected. First relu receives two inputs and pass that to the second relu
    /// This test checks that when inserting a requant layer in between, the inputs and output edges
    /// are still correct.
    /// Relu is easy since in inference, it can support many inputs.
    /// Graph wise:
    ///      A
    ///     / \\  <-- double inputs for C
    ///    B   C
    /// should become:
    ///       A
    ///     /  \\
    ///    R1  R2  <-- distinct requant layers !
    ///    /    \\
    ///   B      C
    #[test]
    fn test_model_insert_requant() {
        init_test_logging_default();
        const FIRST_INPUT_SIZE: usize = 27;
        const SECOND_INPUT_SIZE: usize = 49;
        let input_shapes = vec![
            vec![FIRST_INPUT_SIZE].into(),
            vec![SECOND_INPUT_SIZE].into(),
        ];
        let mut model =
            Model::<Element>::new_from_input_shapes(input_shapes.clone(), PaddingMode::NoPadding);
        let relu1 = model
            .add_layer(Layer::Activation(Activation::new_relu()))
            .unwrap();
        model.set_input(relu1, vec![0, 1]).unwrap();
        // here we take the first two outputs of relu1
        let relu2 = model
            .add_layer(Layer::Activation(Activation::new_relu()))
            .unwrap();
        model.add_edge(relu1, relu2, vec![(0, 0), (1, 1)]).unwrap();
        // here we only want to take the first output of relu1
        let relu3 = model
            .add_layer(Layer::Activation(Activation::new_relu()))
            .unwrap();
        model.add_edge(relu1, relu3, vec![(0, 0)]).unwrap();

        let input_tensor = vec![
            Tensor::random(&input_shapes[0]),
            Tensor::random(&input_shapes[1]),
        ];
        let test_sf = ScalingFactor::from_scale(1.0, None);
        // 2 requants, one for each outgoing output wire (one for relu2 and one for relu3)
        let requants = vec![Requant::from_scaling_factors(test_sf, test_sf, test_sf, 10); 2];
        let requants_ids = model.add_requant_layer(requants, relu1).unwrap();
        assert_eq!(requants_ids.len(), 2);
        model.set_output(relu2, vec![0, 1]).unwrap();
        model.set_output(relu3, vec![2]).unwrap();
        model
            .run::<GoldilocksExt2>(&input_tensor, None, &mut Default::default())
            .unwrap();
    }

    #[test]
    fn test_model_multiple_outputs() {
        init_test_logging("debug");
        const FIRST_INPUT_SIZE: usize = 27;
        const SECOND_INPUT_SIZE: usize = 49;
        let input_shapes = vec![
            vec![FIRST_INPUT_SIZE].into(),
            vec![SECOND_INPUT_SIZE].into(),
        ];
        let mut model = Model::<f32>::new_from_input_shapes(input_shapes, PaddingMode::NoPadding);
        // add first dense layer
        // generate random dense matrix
        let ncols = FIRST_INPUT_SIZE;
        let nrows = 42;
        let dense = Dense::random(
            vec![nrows, ncols].into(),
            Some("dense_1".to_string().into()),
        );
        let first_dense_out_shape = &dense.output_shapes(
            &[model.unpadded_input_shapes()[0].clone()],
            PaddingMode::NoPadding,
        )[0];
        let first_input_dense = model.add_layer(Layer::Dense(dense)).unwrap();
        // set that it will consume the first input
        model.set_input(first_input_dense, 0).unwrap();
        // add second input dense layer
        let ncols = SECOND_INPUT_SIZE;
        let nrows = 47;
        let dense = Dense::random(
            vec![nrows, ncols].into(),
            Some("dense_2".to_string().into()),
        );
        let second_dense_out_shape = &dense.output_shapes(
            &[model.unpadded_input_shapes()[1].clone()],
            PaddingMode::NoPadding,
        )[0];
        let second_input_dense = model.add_layer(Layer::Dense(dense)).unwrap();
        model.set_input(second_input_dense, vec![1]).unwrap();

        // add Relu nodes
        let relu = Activation::new_relu();
        let first_relu_node = model
            .add_consecutive_layer(Layer::Activation(relu.clone()), Some(first_input_dense))
            .unwrap();
        let second_relu_node = model
            .add_consecutive_layer(Layer::Activation(relu), Some(second_input_dense))
            .unwrap();
        // add other dense nodes
        let nrows = 52;
        let ncols = second_dense_out_shape[0]; // it's a vector, so it has only one dimension
        let dense = Dense::random(
            vec![nrows, ncols].into(),
            Some("dense_out_1".to_string().into()),
        );
        let first_output_node = model
            .add_consecutive_layer(Layer::Dense(dense), Some(second_relu_node))
            .unwrap();
        let nrows = 17;
        let ncols = first_dense_out_shape[0];
        let dense = Dense::random(
            vec![nrows, ncols].into(),
            Some("dense_out_2".to_string().into()),
        );
        let second_output_node = model
            .add_consecutive_layer(Layer::Dense(dense), Some(first_relu_node))
            .unwrap();

        // this set automatically first_output(0) => output(0) and second_output(0) => output(1)
        model.automatic_output_labelling().unwrap();

        let out_node_ids = model.graph.output_nodes().map(|(id, _)| *id).collect_vec();

        assert_eq!(out_node_ids.len(), 2);
        assert!(out_node_ids.contains(&first_output_node));
        assert!(out_node_ids.contains(&second_output_node));

        model.describe();

        prove_model(model, &mut Default::default()).unwrap();
    }

    #[test]
    fn test_model_with_multiple_inputs() {
        let input_shapes = vec![vec![6, 9].into(), vec![9, 13].into(), vec![11, 9].into()];

        let mut model = Model::new_from_input_shapes(input_shapes, PaddingMode::NoPadding);

        // Add an input MatMul layer multiplying second with third input
        let first_input_node = model
            .add_layer(Layer::MatMul(
                MatMul::new(OperandMatrix::Input, OperandMatrix::Input).unwrap(),
            ))
            .unwrap();
        model.set_input(first_input_node, vec![2, 1]).unwrap();

        // Add another input MatMul layer multiplying second with first input
        let second_input_node = model
            .add_layer(Layer::MatMul(
                MatMul::new(OperandMatrix::Input, OperandMatrix::Input).unwrap(),
            ))
            .unwrap();
        model.set_input(second_input_node, vec![0, 1]).unwrap();

        // multiply the previous nodes
        let third = model
            .add_layer(Layer::MatMul(
                MatMul::new_with_config(
                    OperandMatrix::Input,
                    OperandMatrix::Input,
                    None,
                    crate::layers::matrix_mul::Config::TransposeB,
                )
                .unwrap(),
            ))
            .unwrap();
        model.add_edge(first_input_node, third, (0, 0)).unwrap();
        // same shorter notation
        model.add_edge(second_input_node, third, (0, 1)).unwrap();
        model.automatic_output_labelling().unwrap();

        prove_model(model, &mut Default::default()).unwrap();
    }

    #[test]
    fn test_model_with_multiple_output_edges() {
        let input_shapes = vec![vec![7, 11].into(), vec![11, 13].into()];

        let input_layer =
            Layer::MatMul(MatMul::new(OperandMatrix::Input, OperandMatrix::Input).unwrap());

        let mut model = Model::new_from_input_shapes(input_shapes, PaddingMode::NoPadding);

        let first_out_layer = Layer::MatMul(
            MatMul::new(
                OperandMatrix::Input,
                OperandMatrix::new_weight_matrix(KeyedTensor::new(
                    "first_out_weight",
                    Tensor::random(&vec![13, 9].into()),
                )),
            )
            .unwrap(),
        );

        let second_out_layer = Layer::MatMul(
            MatMul::new(
                OperandMatrix::Input,
                OperandMatrix::new_weight_matrix(KeyedTensor::new(
                    "second_out_weight",
                    Tensor::random(&vec![13, 13].into()),
                )),
            )
            .unwrap(),
        );

        let input_node_id = model.add_consecutive_layer(input_layer, None).unwrap();

        let _first_out_node_id = model
            .add_consecutive_layer(first_out_layer, Some(input_node_id))
            .unwrap();

        let _second_out_node_id = model
            .add_consecutive_layer(second_out_layer, Some(input_node_id))
            .unwrap();

        model.automatic_output_labelling().unwrap();

        prove_model(model, &mut Default::default()).unwrap();
    }

    #[test]
    fn test_model_with_duplicated_static_tensors() {
        // build a model with 2 MatMul layers sharing the same tensor
        let input_shape = vec![17, 14].into();
        let weight_shape = vec![14, 17].into();
        let bias_shape = vec![17].into();
        let matmul_weight = KeyedTensor::new("matmul_weight", Tensor::random(&weight_shape));
        let bias = KeyedTensor::new("matmul_bias", Tensor::random(&bias_shape));
        let mut model = Model::new_from_input_shapes(vec![input_shape], PaddingMode::NoPadding);
        let first_layer_id = model
            .add_consecutive_layer(
                Layer::MatMul(MatMul::new_constant(matmul_weight.clone(), Some(bias)).unwrap()),
                None,
            )
            .unwrap();
        let _ = model
            .add_consecutive_layer(
                Layer::MatMul(
                    MatMul::new_with_config(
                        OperandMatrix::Input,
                        OperandMatrix::new_weight_matrix(matmul_weight),
                        None,
                        Config::TransposeB,
                    )
                    .unwrap(),
                ),
                Some(first_layer_id),
            )
            .unwrap();
        model.automatic_output_labelling().unwrap();

        prove_model(model, &mut Default::default()).unwrap();
    }
}
