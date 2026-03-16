use crate::{
    Element, IO, Tensor,
    graph::NodeId,
    layers::NodeOut,
    model::ModelGraph,
    quantization::{ModelMetadata, ToField},
    tensor::{TensorHandle, TensorTypeParam},
};

use anyhow::{anyhow, ensure};
use ff_ext::ExtensionField;
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use std::{
    collections::{BTreeMap, HashMap},
    fmt::Debug,
    sync::MappedRwLockReadGuard,
};
use tenstore::{GenStore, StoreError};

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
            steps: Default::default(),
            input,
            output: Default::default(),
        }
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

    /// Find the output tensor handlers.
    pub(crate) fn graph_outputs(
        &self,
        graph: &ModelGraph<D>,
    ) -> anyhow::Result<Vec<TensorHandle<D>>> {
        // compute the output tensor from the outputs of the output nodes
        let output_nodes = graph.output_nodes().map(|(node_id, _)| node_id);
        let mut outputs = BTreeMap::<usize, TensorHandle<D>>::new();
        for (output_node_id, in_feed) in output_nodes.into_iter().flat_map(|node_id| {
            graph
                .incoming_feeds(node_id)
                .into_iter()
                .map(move |in_feed| (node_id, in_feed))
        }) {
            let output_idx = graph[output_node_id].as_output().unwrap();
            let node_outputs = self
                .get_step(&in_feed.source.node_id)
                .ok_or(anyhow!("{in_feed:?} not found in trace"))?
                .outputs();
            ensure!(
                node_outputs.len() > *in_feed.source.port,
                "Number of outputs found in trace ({}) for node {} is smaller than expected number of outputs ({})",
                node_outputs.len(),
                in_feed.source.node_id,
                in_feed.source.port
            );
            // if this output wire is an output of the model, insert in the
            // collection of the model outputs, paired with the index among the
            // outputs of the model
            let old_output =
                outputs.insert(*output_idx, node_outputs[*in_feed.source.port].clone());
            ensure!(
                old_output.is_none(),
                "Trying to insert twice an output value for the same index {output_idx}",
            );
        }
        ensure!(*outputs.first_key_value().unwrap().0 == 0);
        ensure!(*outputs.last_key_value().unwrap().0 == outputs.len() - 1);

        Ok(outputs.into_values().collect())
    }

    /// Get the trace data for node `node_id`, if any
    pub fn get_step(&self, node_id: &NodeId) -> Option<&Step<D>> {
        self.steps.get(node_id)
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
    pub fn to_verifier_io<E>(&self) -> anyhow::Result<IO<E>>
    where
        E: ExtensionField,
        D: ToField<E>,
    {
        let inputs = self
            .input
            .iter()
            .map(|handle| {
                let tensor_guard = handle.tensor()?;
                let padded = (*tensor_guard).pad_next_power_of_two();
                Ok(padded.to_field())
            })
            .collect::<anyhow::Result<Vec<_>>>()?;

        let outputs = self
            .output
            .iter()
            .map(|handle| {
                let tensor_guard = handle.tensor()?;
                let padded = (*tensor_guard).pad_next_power_of_two();
                Ok(padded.to_field())
            })
            .collect::<anyhow::Result<Vec<_>>>()?;
        Ok(IO::new(inputs, outputs))
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
