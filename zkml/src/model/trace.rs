use crate::{
    Element, IO, Shape, Tensor,
    graph::{Graph, NodeId, NodeOutput},
    iop::chunking::{SplittedIOInfo, SplittedNodes},
    layers::{
        NodeOut,
        provable::{Evaluate, LayerOut, OpInfo, ProvingHandle},
        split::SplitLayer,
    },
    model::{ModelCtx, ToStorageKey},
    padding::PaddingMode,
    quantization::{ModelMetadata, ToElement, ToField},
    tensor::{TensorHandle, TensorTypeParam},
    try_unzip,
};

use anyhow::{anyhow, bail, ensure};
use ark_ff::PrimeField;
use itertools::Itertools;
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use std::{
    borrow::Borrow, collections::HashMap, fmt::Debug, ops::Deref, sync::MappedRwLockReadGuard,
};
use tenstore::{GenStore, StorageKey, StoreError};

/// The trace produce by running the model during inference
#[derive(Default, Clone, Serialize, Deserialize)]
#[serde(bound(serialize = "D: Serialize", deserialize = "D: DeserializeOwned"))]
pub struct Trace<D>
where
    D: TensorTypeParam,
{
    pub(crate) steps: HashMap<NodeId, Step<D>>,
    pub(crate) input: Vec<TensorHandle<D>>,
    pub(crate) output: Vec<TensorHandle<D>>,
    // Information about how to split output tensors in multiple chunks, if applicable
    pub(crate) splitted_outputs: Option<SplittedIOInfo>,
    // Information about how to split input tensors in multiple chunks, if applicable
    pub(crate) splitted_inputs: Option<SplittedIOInfo>,
}

impl<D> Debug for Trace<D>
where
    D: TensorTypeParam,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Trace{{ # steps: {:?}, # input: {:?}, # output: {:?} }}",
            self.steps.len(),
            self.input.len(),
            self.output.len()
        )
    }
}

impl<D> Trace<D>
where
    D: TensorTypeParam,
{
    pub fn new(input: Vec<TensorHandle<D>>) -> Self {
        Self {
            input,
            ..Default::default()
        }
    }

    pub fn attach_split_info(&mut self, split_info: &SplittedNodesInfo) {
        self.splitted_outputs = Some(split_info.splitted_nodes.outputs.clone());
        self.splitted_inputs = Some(split_info.splitted_nodes.inputs.clone());
    }

    pub fn attach_store(&mut self, store: GenStore) {
        self.input
            .iter_mut()
            .for_each(|handle| handle.attach_store(store.clone()));
        self.output
            .iter_mut()
            .for_each(|handle| handle.attach_store(store.clone()));
        self.steps
            .values_mut()
            .for_each(|step| step.attach_store(store.clone()));
    }

    pub(crate) fn dry_handles(&self) {
        self.input.iter().for_each(TensorHandle::dry);
        self.output.iter().for_each(TensorHandle::dry);
        self.steps.values().for_each(|step| step.dry_handles());
    }

    /// Get the trace data for node `node_id`, if any
    pub fn get_step(&self, node_id: &NodeId) -> Option<&Step<D>> {
        self.steps.get(node_id)
    }

    /// Replace the trace step of each splitted node in the model with the trace steps of the chunked nodes replacing the
    /// splitted node, including any split and recombination layers added to the model in order to deal with the new chunked
    /// nodes
    pub fn replace_splitted_nodes<F: PrimeField>(
        &mut self,
        model_ctx: &ModelCtx<F>,
        split_nodes: &SplittedNodesInfo,
    ) -> anyhow::Result<()> {
        // iterate in forward order on the nodes of the original graph: indeed, if a splitted node `N1` gets its
        // inputs from another splitted node `N2`, we want the trace step for the source splitted node `N2` to be
        // replaced with chunked nodes before processing trace step for node `N1`, so that the handles for the outputs
        // of node `N2` can be already found in the trace

        for (node_id, _) in model_ctx
            .nodes
            .forward_inners()
            .filter(|(node_id, _)| split_nodes.splitted_nodes.inner_nodes.contains_key(node_id))
        {
            let mut step = self.steps.remove(&node_id).ok_or(anyhow!(
                "Node {node_id} (to be replaced with splitted nodes) not found in trace steps"
            ))?;
            let split_info = split_nodes.map_handles(
                node_id,
                &model_ctx.nodes,
                step.inputs(),
                step.outputs(),
                &step.node_outputs.proving_data,
            )?;
            let proving_data =
                std::mem::replace(&mut step.node_outputs.proving_data, ProvingHandle::None);
            let new_steps = self.new_steps_for_splitted_nodes(
                node_id,
                step.inputs(),
                step.outputs(),
                split_info,
                proving_data,
            )?;
            for (node_id, step) in new_steps {
                self.new_step(node_id, step);
            }
        }

        self.attach_split_info(split_nodes);

        Ok(())
    }

    /// Build the trace step for node `node_id`, using the input/output handles and the proving data provided as input.
    /// If the node is a splitted node, more than one trace step is built, the method builds a trace step for each of the
    /// chunked node replacing the splitted node, as well as for any split or recombination layer added to the model to deal
    /// with the splitted node. The `trace_split_info` input data structure is employed to determine how the input/output handles
    /// and the proving data should be split among the different chunked nodes and the split/recombination layer trace steps.
    /// `proving_handle_map` is a closure that specifies how to convert input and output handles to the handles actually employed
    /// to build the trace steps
    pub(crate) fn new_steps_for_splitted_nodes(
        &self,
        node_id: NodeId,
        input_handles: &[TensorHandle<D>],
        output_handles: &[TensorHandle<D>],
        trace_split_info: TraceSplitterInfo<D>,
        proving_data: ProvingHandle,
    ) -> anyhow::Result<HashMap<NodeId, Step<D>>> {
        let mut input_handles_by_node_id = HashMap::new();
        let mut output_handles_by_node_id = HashMap::new();
        for original_handle in output_handles.iter() {
            let storage_key = original_handle.storage_key();
            if let Some(output_handles) = trace_split_info.output_handles.get(storage_key) {
                for (handle, new_node_id) in output_handles {
                    let handle = handle.clone().into_dry_tensor()?;
                    // if there is a recombination layer associated to this node, we need to use this handle also as an input handle
                    // for such recombination layer
                    if let Some(recombination_id) = &trace_split_info.recombination_layer {
                        input_handles_by_node_id
                            .entry(*recombination_id)
                            .or_insert(vec![])
                            .push(handle.clone());
                    }
                    output_handles_by_node_id
                        .entry(*new_node_id)
                        .or_insert(vec![])
                        .push(handle);
                }
                // if there is a recombination layer associated to this node, we need to the `original_handle`
                // as an output handle for such recombination layer
                if let Some(recombination_id) = &trace_split_info.recombination_layer {
                    output_handles_by_node_id
                        .entry(*recombination_id)
                        .or_insert(vec![])
                        .push(original_handle.clone().into_dry_tensor()?);
                }
            } else {
                output_handles_by_node_id
                    .entry(node_id)
                    .or_insert(vec![])
                    .push(original_handle.clone().into_dry_tensor()?);
            }
        }

        if let Some((split_id, split_handles)) = &trace_split_info.split_layer {
            output_handles_by_node_id.insert(
                *split_id,
                split_handles
                    .iter()
                    .map(|handle| handle.clone().into_dry_tensor())
                    .collect::<anyhow::Result<Vec<_>>>()?,
            );
            // the input handles for the split layer are the same as the original node `node_id`
            input_handles_by_node_id.insert(
                *split_id,
                input_handles
                    .iter()
                    .map(|handle| handle.clone().into_dry_tensor())
                    .collect::<Result<_, _>>()?,
            );
        }

        // Add han
        for (new_input_node, handle) in &trace_split_info.model_input_handles {
            output_handles_by_node_id
                .entry(*new_input_node)
                .or_insert(vec![])
                .push(handle.clone().into_dry_tensor()?);
        }

        for handle in input_handles.iter() {
            let storage_key = handle.storage_key();
            if let Some(output_ports) = trace_split_info.input_handles.get(storage_key) {
                for (source, new_node_id) in output_ports {
                    let source_port: usize = source.port.into();
                    let handle = if let Some(step) = self.get_step(&source.node_id) {
                        step.outputs()[source_port].clone().into_dry_tensor()?
                    } else {
                        output_handles_by_node_id
                            .get(&source.node_id)
                            .ok_or(anyhow!(
                                "Output handles not found for node {}",
                                source.node_id
                            ))?[source_port]
                            .clone()
                            .into_dry_tensor()?
                    };
                    input_handles_by_node_id
                        .entry(*new_node_id)
                        .or_insert(vec![])
                        .push(handle)
                }
            } else {
                input_handles_by_node_id
                    .entry(node_id)
                    .or_insert(vec![])
                    .push(handle.clone().into_dry_tensor()?);
            }
        }

        let num_chunks = trace_split_info.new_proving_handles.len();
        let mut proving_data_by_node_id = if num_chunks > 0 {
            trace_split_info
                .new_proving_handles
                .into_iter()
                .map(|(new_node_id, proving_data)| {
                    proving_data
                        .try_into_map(|handle| handle.into_dry_tensor())
                        .map(|proving_handle| (new_node_id, proving_handle))
                })
                .collect::<anyhow::Result<_>>()?
        } else {
            let proving_data = proving_data.try_into_map(|handle| handle.into_dry_tensor())?;
            HashMap::from([(node_id, proving_data)])
        };

        // add proving data for split and recombination layer, if any
        if let Some((split_id, _)) = &trace_split_info.split_layer {
            proving_data_by_node_id.insert(*split_id, ProvingHandle::None);
        }

        if let Some(recombination_id) = &trace_split_info.recombination_layer {
            proving_data_by_node_id.insert(*recombination_id, ProvingHandle::None);
        }

        input_handles_by_node_id
            .into_iter()
            .map(|(node_id, input_handles)| {
                let output_handles = output_handles_by_node_id.remove(&node_id).ok_or(anyhow!(
                    "Output handles not found for node {node_id} in TraceRunner"
                ))?;
                let proving_data = proving_data_by_node_id.remove(&node_id).ok_or(anyhow!(
                    "Proving data not found for node {node_id} in TraceRunner"
                ))?;
                let new_step = Step {
                    node_inputs: input_handles,
                    node_outputs: NodeOut::new(output_handles, proving_data),
                };
                Ok((node_id, new_step))
            })
            .collect()
    }

    /// Insert the trace data `step` about node `node_id` in the trace
    pub(crate) fn new_step(&mut self, node_id: NodeId, step: Step<D>) {
        assert!(!self.steps.contains_key(&node_id));
        self.steps.insert(node_id, step);
    }

    /// Get the output tensors of the inference represented by this trace
    pub fn outputs(&self) -> &[TensorHandle<D>] {
        &self.output
    }

    /// Get the inputs tensors of the inference represented by this trace
    pub fn inputs(&self) -> &[TensorHandle<D>] {
        &self.input
    }

    /// Compute the inputs and outputs tensors from the trace, which are necessary
    /// for the verifier to verify the proof of the model inference
    pub fn to_verifier_io<F>(&self) -> anyhow::Result<IO<F>>
    where
        F: PrimeField + ToElement,
        D: ToField<F>,
    {
        let inputs = self
            .input
            .iter()
            .map(|handle| {
                let tensor_guard = handle.tensor()?;
                let padded = (*tensor_guard).pad_next_power_of_two();
                Ok(padded.to_field().into())
            })
            .collect::<anyhow::Result<Vec<_>>>()?;

        let outputs = self
            .output
            .iter()
            .map(|handle| {
                let tensor_guard = handle.tensor()?;
                let padded = (*tensor_guard).pad_next_power_of_two();
                Ok(padded.to_field().into())
            })
            .collect::<anyhow::Result<Vec<_>>>()?;

        IO::new(inputs, outputs)
            .with_splitted_inputs(self.splitted_inputs.as_ref())?
            .with_splitted_outputs(self.splitted_outputs.as_ref())
    }
}

impl Trace<Element> {
    /// Given as input a trace over quantized values, compute the equivalent
    /// trace with dequantized values
    pub fn dequantized(&self, model_metadata: &ModelMetadata) -> Result<Trace<f32>, StoreError> {
        let inputs = self
            .input
            .iter()
            .enumerate()
            .map(|(i, handle)| {
                let scaling_factor = model_metadata.input_scaling(i);
                handle.cast(|x| scaling_factor.dequantize(x))
            })
            .collect::<Result<Vec<TensorHandle<f32>>, StoreError>>()?;

        let outputs = self
            .output
            .iter()
            .enumerate()
            .map(|(i, handle)| {
                let scaling_factor = model_metadata.output_scaling(i);
                handle.cast(|x| scaling_factor.dequantize(x))
            })
            .collect::<Result<Vec<TensorHandle<f32>>, StoreError>>()?;

        let steps = self
            .steps
            .iter()
            .map(|(node_id, step)| {
                step.to_dequantize(model_metadata, *node_id)
                    .map(|step| (*node_id, step))
            })
            .collect::<Result<HashMap<_, _>, StoreError>>()?;

        Ok(Trace {
            steps,
            input: inputs,
            output: outputs,
            splitted_inputs: self.splitted_inputs.clone(),
            splitted_outputs: self.splitted_outputs.clone(),
        })
    }
}

/// Data found in the trace for each node of the model
#[derive(Clone, Serialize, Deserialize)]
#[serde(bound(serialize = "D: Serialize", deserialize = "D: DeserializeOwned"))]
pub struct Step<D>
where
    D: TensorTypeParam,
{
    /// Ordered by input port (e.g. target_port of the incoming edges)
    pub(crate) node_inputs: Vec<TensorHandle<D>>,
    /// Ordered by output port (e.g. source_port of the incoming edges)
    pub(crate) node_outputs: NodeOut<D>,
}

impl<D> Step<D>
where
    D: TensorTypeParam,
{
    /// Returns the output tensors of the node
    pub fn outputs(&self) -> &[TensorHandle<D>] {
        self.node_outputs.outputs.as_slice()
    }

    pub(crate) fn dry_handles(&self) {
        self.node_inputs.iter().for_each(TensorHandle::dry);
        self.node_outputs.outputs.iter().for_each(TensorHandle::dry);
        self.node_outputs
            .proving_data
            .handles()
            .for_each(TensorHandle::dry);
    }

    pub(crate) fn attach_store(&mut self, store: GenStore) {
        self.node_inputs
            .iter_mut()
            .for_each(|handle| handle.attach_store(store.clone()));
        self.node_outputs
            .outputs
            .iter_mut()
            .for_each(|handle| handle.attach_store(store.clone()));
        self.node_outputs.proving_data.attach_store(store);
    }

    /// Returns the input tensor handles of the node
    pub fn inputs(&self) -> &[TensorHandle<D>] {
        &self.node_inputs
    }

    /// Hydrate all the input tensors of the node corresponding to this step.
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn input_tensors(&self) -> anyhow::Result<Vec<Tensor<D>>> {
        self.node_inputs
            .iter()
            .map(|handle| handle.tensor().map(|read_guard| (*read_guard).clone()))
            .collect()
    }

    /// Hydrate one of the input tensors of the node corresponding to this step.
    pub(crate) fn input_tensor_at(
        &self,
        i: usize,
    ) -> anyhow::Result<MappedRwLockReadGuard<'_, Tensor<D>>> {
        self.node_inputs[i].tensor()
    }

    /// Hydrate all the output tensors of the node corresponding to this step.
    pub(crate) fn output_tensors(&self) -> anyhow::Result<Vec<Tensor<D>>> {
        self.node_outputs
            .outputs
            .iter()
            .map(|handle| handle.tensor().map(|read_guard| (*read_guard).clone()))
            .collect()
    }

    /// Hydrate one of the output tensors of the node corresponding to this step.
    pub(crate) fn output_tensor_at(
        &self,
        i: usize,
    ) -> anyhow::Result<MappedRwLockReadGuard<'_, Tensor<D>>> {
        self.node_outputs.outputs[i].tensor()
    }

    /// Hydrate all the input tensors of the node, padding each to the next power of two.
    pub(crate) fn padded_input_tensors(&self) -> anyhow::Result<Vec<Tensor<D>>> {
        self.node_inputs
            .iter()
            .map(|handle| {
                let tensor = handle.tensor()?;
                Ok(tensor.pad_next_power_of_two())
            })
            .collect()
    }

    /// Hydrate one of the input tensors of the node, padding it to the next power of two.
    pub(crate) fn padded_input_tensor_at(&self, i: usize) -> anyhow::Result<Tensor<D>> {
        let tensor = self.node_inputs[i].tensor()?;
        Ok(tensor.pad_next_power_of_two())
    }

    /// Hydrate all the output tensors of the node, padding each to the next power of two.
    pub(crate) fn padded_output_tensors(&self) -> anyhow::Result<Vec<Tensor<D>>> {
        self.node_outputs
            .outputs
            .iter()
            .map(|handle| {
                let tensor = handle.tensor()?;
                Ok(tensor.pad_next_power_of_two())
            })
            .collect()
    }
}

impl Step<Element> {
    pub fn to_dequantize(
        &self,
        model_metadata: &ModelMetadata,
        node_id: NodeId,
    ) -> Result<Step<f32>, StoreError> {
        let node_inputs = self
            .node_inputs
            .iter()
            .zip(model_metadata.layer_input_scaling_factor(node_id))
            .map(|(handle, scale_factor)| handle.cast(|x| scale_factor.dequantize(x)))
            .collect::<Result<Vec<_>, StoreError>>()?;

        Ok(Step {
            node_inputs,
            node_outputs: self.node_outputs.to_dequantize(model_metadata, node_id)?,
        })
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SplittedNodesInfo {
    pub(crate) splitted_nodes: SplittedNodes,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(bound(serialize = "N: Serialize", deserialize = "N: DeserializeOwned"))]
pub struct TraceSplitterInfo<N: TensorTypeParam> {
    pub(crate) output_handles: OutputHandleMap<N>,
    pub(crate) input_handles: InputHandleMap<N>,
    pub(crate) split_layer: Option<(NodeId, Vec<TensorHandle<N>>)>,
    pub(crate) recombination_layer: Option<NodeId>,
    pub(crate) new_proving_handles: HashMap<NodeId, ProvingHandle>,
    pub(crate) model_input_handles: HashMap<NodeId, TensorHandle<N>>,
}

pub(crate) type OutputHandleMap<N> = HashMap<StorageKey<Vec<N>>, Vec<(TensorHandle<N>, NodeId)>>;
pub(crate) type InputHandleMap<N> = HashMap<StorageKey<Vec<N>>, Vec<(NodeOutput, NodeId)>>;

pub(crate) fn split_tensors<N: TensorTypeParam, B: Borrow<TensorHandle<N>>>(
    handles: &[B],
    split_layer: &SplitLayer,
) -> anyhow::Result<(Vec<Shape>, LayerOut<N>)> {
    // ensure we can get `WrappedTensor` for all the input `handles`
    let handles = handles
        .iter()
        .map(|handle| handle.borrow().clone().wrapped_tensor_variant())
        .collect::<anyhow::Result<Vec<_>>>()?;
    let (unpadded_input_shapes, inputs): (Vec<_>, Vec<_>) =
        try_unzip(handles.iter().map(|handle| {
            anyhow::Ok((handle.unpadded_shape().clone(), handle.wrapped_tensor()?))
        }))?;
    let layer_out =
        split_layer.evaluate(&inputs.iter().map(|guard| guard.deref()).collect_vec())?;

    let unpadded_output_shapes =
        split_layer.output_shapes(&unpadded_input_shapes, PaddingMode::NoPadding)?;

    Ok((unpadded_output_shapes, layer_out))
}

impl SplittedNodesInfo {
    pub(crate) fn map_handles<N: TensorTypeParam, T>(
        &self,
        node_id: NodeId,
        model_graph: &Graph<T, usize, usize, ()>,
        input_handles: &[TensorHandle<N>],
        output_handles: &[TensorHandle<N>],
        proving_data: &ProvingHandle,
    ) -> anyhow::Result<TraceSplitterInfo<N>> {
        // check if the node is in the splitted nodes
        if let Some(splitted_node) = self.splitted_nodes.inner_nodes.get(&node_id) {
            let num_chunks = splitted_node.new_nodes.len();

            // build a split layer to split the output in `num_chunks` chunks
            let num_handles = output_handles.len();
            let split_layer = SplitLayer {
                unpadded_input_shapes: output_handles
                    .iter()
                    .map(|handle| handle.unpadded_shape().clone())
                    .collect_vec(),
                num_chunks: vec![num_chunks; num_handles],
            };

            let (unpadded_output_shapes, layer_out) = split_tensors(output_handles, &split_layer)?;
            let mut trace_split_info = TraceSplitterInfo::default();
            // add new node is to `trace_split_info`
            for (i, (out_tensor, out_shape)) in layer_out
                .outputs
                .into_iter()
                .zip(unpadded_output_shapes)
                .enumerate()
            {
                let chunk_number = i % num_chunks;
                let chunked_node_id = splitted_node.new_nodes.get(&chunk_number).ok_or(anyhow!(
                    "No node found for horizontal chunk {chunk_number} of node {node_id}"
                ))?;

                let output_port = i / num_chunks;
                let storage_key: StorageKey<Vec<N>> =
                    NodeOutput::new(*chunked_node_id, output_port).to_storage_key();
                let original_handle = &output_handles[output_port];
                let new_handle = TensorHandle::from_wrapped_tensor_with_unpadded_shape(
                    storage_key,
                    original_handle.store().clone(),
                    out_tensor,
                    out_shape.clone(),
                );
                trace_split_info
                    .output_handles
                    .entry(original_handle.storage_key().clone())
                    .or_insert(vec![])
                    .push((new_handle, *chunked_node_id));
            }

            // deal with recombination layer handles, if there is a recombination layer
            if let Some((recombination_id, _)) = &splitted_node.recombination_layer {
                // the input handles of the recombination layer are the same as chunked output handles for the current node
                trace_split_info.recombination_layer = Some(*recombination_id);
            }

            // map input handles
            let feeds = model_graph.incoming_feeds(node_id);
            let neighbor_splitted_node_id = feeds
                .iter()
                .find_map(|feed| self.splitted_nodes.inner_nodes.get(&feed.source.node_id));
            if let Some(neighbor_node) = neighbor_splitted_node_id {
                ensure!(
                    feeds.iter().map(|feed| feed.source.node_id).all_equal(),
                    "Expected a single source node for splitted node {node_id}"
                );
                let source_node_id = feeds[0].source.node_id;
                for feed in feeds {
                    let input_port: usize = feed.target.port.into();
                    let source_port = feed.source.port;
                    let output_ports = (0..num_chunks).map(|chunk_number| {
                        let new_node = neighbor_node.new_nodes.get(&chunk_number).ok_or(
                            anyhow!("No node found for horizontal chunk {chunk_number} of source node {source_node_id}")
                        )?;
                        let chunked_node_id = splitted_node.new_nodes.get(&chunk_number)
                            .ok_or(
                                anyhow!("No node found for horizontal chunk {chunk_number} of node {node_id}")
                            )?;
                        Ok((NodeOutput::new(*new_node, source_port), *chunked_node_id))
                    }).collect::<anyhow::Result<Vec<_>>>()?;
                    let original_storage_key = input_handles[input_port].storage_key().clone();
                    trace_split_info
                        .input_handles
                        .insert(original_storage_key, output_ports);
                }
            } else {
                if let Some((split_id, split_layer)) = splitted_node.split_layer.as_ref() {
                    ensure!(split_layer.num_chunks.len() == input_handles.len());

                    let store = input_handles
                        .first()
                        .expect("No inputs for layer {node_id}?")
                        .store();

                    let (unpadded_out_shapes, layer_out) =
                        split_tensors(input_handles, split_layer)?;

                    let split_output_handles = layer_out
                        .outputs
                        .into_iter()
                        .zip(unpadded_out_shapes)
                        .enumerate()
                        .map(|(i, (out_tensor, out_shape))| {
                            let storage_key = NodeOutput::new(*split_id, i).to_storage_key();
                            TensorHandle::from_wrapped_tensor_with_unpadded_shape(
                                storage_key,
                                store.clone(),
                                out_tensor,
                                out_shape,
                            )
                        })
                        .collect_vec();

                    for feed in feeds {
                        let input_port: usize = feed.target.port.into();
                        let source_port: usize = feed.source.port.into();
                        let output_ports = (0..num_chunks).map(|chunk_number| {
                            let split_layer_out_port = source_port*num_chunks + chunk_number;
                            let chunked_node_id = splitted_node.new_nodes.get(&chunk_number)
                                .ok_or(
                                    anyhow!("No node found for horizontal chunk {chunk_number} of node {node_id}")
                                )?;
                            Ok((NodeOutput::new(*split_id, split_layer_out_port), *chunked_node_id))
                        }).collect::<anyhow::Result<Vec<_>>>()?;
                        let original_storage_key = input_handles[input_port].storage_key().clone();
                        trace_split_info
                            .input_handles
                            .insert(original_storage_key, output_ports);
                    }

                    trace_split_info.split_layer = Some((*split_id, split_output_handles));
                } else {
                    // the splitted node is linked only to inputs of the model
                    for feed in feeds {
                        let source_node_id = feed.source.node_id;
                        let input_port: usize = feed.target.port.into();
                        let input_id = model_graph
                            .node(source_node_id)
                            .ok_or(anyhow!(
                                "Input node {source_node_id} not found in model graph"
                            ))?
                            .as_input()
                            .ok_or(anyhow!(
                                "Node {source_node_id} is not an input node of the graph"
                            ))?;
                        // check that this input is actually meant to be split in multiple chunks
                        let Some(new_input_nodes) = self.splitted_nodes.inputs.get(input_id) else {
                            bail!(
                                "Input node {source_node_id} of splitted node {node_id} is not chunked"
                            )
                        };

                        let num_chunks = new_input_nodes.len();
                        let original_handle = &input_handles[input_port];
                        let split_layer = SplitLayer {
                            unpadded_input_shapes: vec![
                                original_handle.unpadded_shape().clone();
                                1
                            ],
                            num_chunks: vec![num_chunks; 1],
                        };
                        let (unpadded_shapes, layer_out) =
                            split_tensors(&[original_handle], &split_layer)?;
                        let output_ports = new_input_nodes.iter()
                            .zip_eq(unpadded_shapes)
                            .zip_eq(layer_out.outputs)
                            .enumerate()
                            .map(|(chunk_number, ((new_node, out_shape), out_tensor))| {
                                let output_port = NodeOutput::new(*new_node, 0);
                                let chunked_node_id = splitted_node.new_nodes.get(&chunk_number)
                                    .ok_or(
                                        anyhow!("No node found for horizontal chunk {chunk_number} of node {node_id}")
                                    )?;
                                let storage_key = output_port.to_storage_key();
                                let new_handle = TensorHandle::from_wrapped_tensor_with_unpadded_shape(
                                    storage_key,
                                    original_handle.store().clone(),
                                    out_tensor,
                                    out_shape,
                                );
                                ensure!(
                                    trace_split_info.model_input_handles.insert(*new_node, new_handle).is_none(),
                                    "Trying to insert handle for the same chunked input node {new_node}"
                                );
                                anyhow::Ok(
                                    (output_port, *chunked_node_id)
                                )
                            }).collect::<anyhow::Result<Vec<_>>>()?;
                        let original_storage_key = original_handle.storage_key().clone();
                        trace_split_info
                            .input_handles
                            .insert(original_storage_key, output_ports);
                    }
                }
            }

            let chunked_proving_data = proving_data.split_proving_data(num_chunks)?;
            trace_split_info.new_proving_handles = chunked_proving_data
                .into_iter()
                .enumerate()
                .map(|(i, proving_data)| {
                    let new_node_id = splitted_node.new_nodes.get(&i).ok_or(anyhow!(
                        "No chunked node found for chunk number {i} of splitted node {node_id}"
                    ))?;
                    Ok((*new_node_id, proving_data))
                })
                .collect::<anyhow::Result<_>>()?;

            Ok(trace_split_info)
        } else {
            Ok(Default::default())
        }
    }
}
