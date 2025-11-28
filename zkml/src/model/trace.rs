use crate::{
    Element, IO, Shape, Tensor,
    graph::NodeId,
    layers::NodeOut,
    quantization::{Fieldizer, ModelMetadata},
    tensor::{TensorHandle, TensorTypeParam},
};
use ff_ext::ExtensionField;
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use std::{collections::HashMap, fmt::Debug, sync::MappedRwLockReadGuard};
use tenstore::{GenStore, StoreError};

/// The trace produce by running the model during inference
#[derive(Default, Clone, Serialize, Deserialize)]
#[serde(bound(serialize = "E: Serialize", deserialize = "E: DeserializeOwned"))]
pub struct Trace<E, D>
where
    E: ExtensionField,
    D: TensorTypeParam,
{
    pub(crate) steps: HashMap<NodeId, Step<E, D>>,
    pub(crate) input: Vec<TensorHandle<D>>,
    pub(crate) output: Vec<TensorHandle<D>>,
}

impl<E: ExtensionField, D: TensorTypeParam> Debug for Trace<E, D> {
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

impl<E: ExtensionField, D> Trace<E, D>
where
    E: ExtensionField,
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
}

impl<E: ExtensionField, D> Trace<E, D>
where
    E: ExtensionField,
    D: TensorTypeParam + Clone + Serialize + for<'b> Deserialize<'b>,
{
    /// Get the trace data for node `node_id`, if any
    pub fn get_step(&self, node_id: &NodeId) -> Option<&Step<E, D>> {
        self.steps.get(node_id)
    }

    /// Insert the trace data `step` about node `node_id` in the trace
    pub(crate) fn new_step(&mut self, node_id: NodeId, step: Step<E, D>) {
        assert!(!self.steps.contains_key(&node_id));
        self.steps.insert(node_id, step);
    }

    /// Compute the inputs and outputs tensors from the trace, which are necessary
    /// for the verifier to verify the proof of the model inference
    pub fn to_verifier_io(&self) -> anyhow::Result<IO<E>>
    where
        D: Fieldizer<E>,
    {
        let inputs = self
            .input
            .iter()
            .map(|handle| handle.hydrated_cast(|x| x.to_field()))
            .collect::<anyhow::Result<Vec<_>>>()?;

        let outputs = self
            .output
            .iter()
            .map(|handle| handle.hydrated_cast(|x| x.to_field()))
            .collect::<anyhow::Result<Vec<_>>>()?;
        Ok(IO::new(inputs, outputs))
    }

    /// Get the output tensors of the inference represented by this trace
    pub fn outputs(&self) -> &[TensorHandle<D>] {
        &self.output
    }

    /// Get the inputs tensors of the inference represented by this trace
    pub fn inputs(&self) -> &[TensorHandle<D>] {
        &self.input
    }
}

impl<E: ExtensionField> Trace<E, Element> {
    /// Given as input a trace over quantized values, compute the equivalent
    /// trace with dequantized values
    pub fn dequantized(&self, model_metadata: &ModelMetadata) -> Result<Trace<E, f32>, StoreError> {
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
#[serde(bound(
    serialize = "E: Serialize, D: Serialize",
    deserialize = "E: DeserializeOwned, D: DeserializeOwned"
))]
pub struct Step<E: ExtensionField, D: TensorTypeParam> {
    /// Ordered by input port (e.g. target_port of the incoming edges)
    pub(crate) node_inputs: Vec<TensorHandle<D>>,
    /// Ordered by output port (e.g. source_port of the incoming edges)
    pub(crate) node_outputs: NodeOut<D, E>,
    /// Ordered by output port (e.g. source_port of the outgoing edges)
    pub(crate) unpadded_output_shapes: Vec<Shape>,
    /// Ordered by input port (e.g. target_port of the incoming edges)
    pub(crate) unpadded_input_shapes: Vec<Shape>,
}

impl<E: ExtensionField, D> Step<E, D>
where
    E: ExtensionField,
    D: TensorTypeParam + Clone + Serialize + for<'b> Deserialize<'b>,
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
    }

    /// Returns the input tensor handles of the node
    pub fn inputs(&self) -> &[TensorHandle<D>] {
        &self.node_inputs
    }

    /// Hydrate all the input tensors of the node corresponding to this step.
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
}

impl<E: ExtensionField> Step<E, Element> {
    pub fn to_dequantize(
        &self,
        model_metadata: &ModelMetadata,
        node_id: NodeId,
    ) -> Result<Step<E, f32>, StoreError> {
        let node_inputs = self
            .node_inputs
            .iter()
            .zip(model_metadata.layer_input_scaling_factor(node_id))
            .map(|(handle, scale_factor)| handle.cast(|x| scale_factor.dequantize(x)))
            .collect::<Result<Vec<_>, StoreError>>()?;

        Ok(Step {
            node_inputs,
            node_outputs: self.node_outputs.to_dequantize(model_metadata, node_id)?,
            unpadded_output_shapes: self.unpadded_output_shapes.clone(),
            unpadded_input_shapes: self.unpadded_input_shapes.clone(),
        })
    }
}
