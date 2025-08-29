use std::{collections::HashMap, fmt::Debug};

use anyhow::Context;
use ff_ext::ExtensionField;
use serde::{Deserialize, Serialize};
use tenstore::{GenStore, StoreError};

use crate::{
    Element, IO, Shape, Tensor,
    layers::{Layer, NodeOut, provable::NodeId},
    quantization::{Fieldizer, ModelMetadata},
    tensor::DryTensor,
};

#[derive(Default, Clone)]
pub struct Trace<'a, E: ExtensionField, N, D> {
    pub(crate) store: GenStore,
    pub(crate) steps: HashMap<NodeId, InferenceStep<'a, E, N, D>>,
    // TODO: convert to TensorKey
    pub(crate) input: Vec<DryTensor<D>>,
    pub(crate) output: Vec<DryTensor<D>>,
}
// The trace produce by running the model during inference
pub type InferenceTrace<'a, E, N> = Trace<'a, E, N, N>;
// The trace used to prove the model
pub type ProvingTrace<'a, E, N> = Trace<'a, E, N, E>;

impl<'a, E: ExtensionField, N, D> Trace<'a, E, N, D>
where
    D: Serialize + for<'b> Deserialize<'b>,
{
    /// Get the trace data for node `node_id`, if any
    pub(crate) fn get_step(&self, node_id: &NodeId) -> Option<&InferenceStep<'a, E, N, D>> {
        self.steps.get(node_id)
    }

    /// Insert the trace data `step` about node `node_id` in the trace
    pub(crate) fn new_step(&mut self, node_id: NodeId, step: InferenceStep<'a, E, N, D>) {
        self.steps.insert(node_id, step);
    }

    /// Compute the inputs and outputs tensors from the trace, which are necessary
    /// for the verifier to verify the proof of the model inference
    pub fn to_verifier_io(&self) -> Result<IO<E>, StoreError>
    where
        D: Fieldizer<E>,
    {
        let inputs = self
            .input
            .iter()
            .map(|dry| dry.hydrated_cast(self.store.clone(), |x| x.to_field()))
            .collect::<Result<Vec<_>, StoreError>>()?;

        let outputs = self
            .output
            .iter()
            .map(|dry| dry.hydrated_cast(self.store.clone(), |x| x.to_field()))
            .collect::<Result<Vec<_>, StoreError>>()?;
        Ok(IO::new(inputs, outputs))
    }

    /// Convert an inference trace computed over integers to a trace over field elements, which is
    /// needed to prove the inference
    pub(crate) fn into_fields(self) -> anyhow::Result<ProvingTrace<'a, E, N>>
    where
        D: Fieldizer<E> + Serialize + for<'b> Deserialize<'b> + Debug,
    {
        let store = self.store.clone();
        let input = self
            .input
            .into_iter()
            .map(|dry| dry.dry_cast(self.store.clone(), |x: &D| x.to_field()))
            .collect::<Result<Vec<DryTensor<E>>, StoreError>>()
            .context("converting input")?;
        let output = self
            .output
            .into_iter()
            .map(|dry| dry.dry_cast(self.store.clone(), |x: &D| x.to_field()))
            .collect::<Result<Vec<DryTensor<E>>, StoreError>>()
            .context("converting output")?;
        let field_steps = self
            .steps
            .into_iter()
            .map(|(id, step)| {
                Ok((
                    id,
                    InferenceStep {
                        op: step.op,
                        step_data: StepData {
                            node_inputs: step
                                .step_data
                                .node_inputs
                                .into_iter()
                                .map(|dry| {
                                    dry.dry_cast(store.clone(), |x| x.to_field())
                                        .with_context(|| format!("converting `{:?}`", dry.key()))
                                })
                                .collect::<anyhow::Result<Vec<DryTensor<_>>>>()?,

                            node_outputs: step
                                .step_data
                                .node_outputs
                                .into_fields(store.clone())
                                .context("converting node_outputs")?,
                            unpadded_output_shapes: step.step_data.unpadded_output_shapes,
                            unpadded_input_shapes: step.step_data.unpadded_input_shapes,
                        },
                    },
                ))
            })
            .collect::<anyhow::Result<HashMap<_, _>>>()
            .context("converting steps")?;
        Ok(Trace {
            store: self.store,
            steps: field_steps,
            input,
            output,
        })
    }

    /// Get the output tensors of the inference represented by this trace
    pub fn outputs(&self) -> anyhow::Result<Vec<Tensor<D>>> {
        Ok(self
            .output
            .iter()
            .map(|dry| dry.hydrate(self.store.clone()))
            .collect::<Result<Vec<_>, StoreError>>()?)
    }

    /// Get the inputs tensors of the inference represented by this trace
    pub fn inputs(&self) -> anyhow::Result<Vec<Tensor<D>>> {
        Ok(self
            .input
            .iter()
            .map(|dry| dry.hydrate(self.store.clone()))
            .collect::<Result<Vec<_>, StoreError>>()?)
    }

    /// Get the (hydrated) ith input tensor of the inference represented by
    /// this trace
    pub fn input_at(&self, i: usize) -> anyhow::Result<Tensor<D>> {
        self.input.get(i).context("invalid index").and_then(|dry| {
            dry.hydrate(self.store.clone())
                .context("failed to fetch from store")
        })
    }

    /// Get the (hydrated) ith output tensor of the inference represented by
    /// this trace
    pub fn output_at(&self, i: usize) -> anyhow::Result<Tensor<D>> {
        self.output.get(i).context("invalid index").and_then(|dry| {
            dry.hydrate(self.store.clone())
                .context("failed to fetch from store")
        })
    }
}

impl<'a, E: ExtensionField> InferenceTrace<'a, E, Element> {
    /// Given as input a trace over quantized values, compute the equivalent
    /// trace with dequantized values
    pub fn dequantized(
        &self,
        md: &ModelMetadata,
    ) -> Result<Trace<'a, E, Element, f32>, StoreError> {
        let inputs = self
            .input
            .iter()
            .zip(&md.input)
            .map(|(dry, s)| dry.dry_cast(self.store.clone(), |x| s.dequantize(x)))
            .collect::<Result<Vec<DryTensor<f32>>, StoreError>>()?;

        let outputs = self
            .output
            .iter()
            .zip(&md.output)
            .map(|(dry, s)| dry.dry_cast(self.store.clone(), |x| s.dequantize(x)))
            .collect::<Result<Vec<DryTensor<f32>>, StoreError>>()?;
        let steps = self
            .steps
            .iter()
            .map(|(node_id, step)| {
                Ok((
                    *node_id,
                    InferenceStep {
                        op: step.op,
                        step_data: step.step_data.to_dequantize(
                            md,
                            self.store.clone(),
                            *node_id,
                        )?,
                    },
                ))
            })
            .collect::<Result<HashMap<_, _>, StoreError>>()?;
        Ok(Trace {
            store: self.store.clone(),
            steps,
            input: inputs,
            output: outputs,
        })
    }
}

// TODO: inline StepData
/// Data found in the trace for each node of the model
#[derive(Clone)]
pub struct InferenceStep<'a, E: ExtensionField, N, D> {
    pub(crate) op: &'a Layer<N>,
    pub(crate) step_data: StepData<D, E>,
}

impl<'a, E: ExtensionField, N, D> InferenceStep<'a, E, N, D>
where
    D: Serialize + for<'b> Deserialize<'b>,
{
    /// Returns the output tensors of the node
    pub fn outputs(&self) -> &[DryTensor<D>] {
        self.step_data.node_outputs.outputs.as_slice()
    }
}

impl<'a, E: ExtensionField, N> InferenceStep<'a, E, N, Element> {
    pub fn to_dequantize(
        &self,
        md: &ModelMetadata,
        store: GenStore,
        node_id: NodeId,
    ) -> Result<InferenceStep<'a, E, N, f32>, StoreError> {
        Ok(InferenceStep {
            op: self.op,
            step_data: self.step_data.to_dequantize(md, store, node_id)?,
        })
    }
}

/// Data about the input and output tensors in a trace
/// for each node in the model
#[derive(Clone)]
pub struct StepData<D, E: ExtensionField> {
    pub(crate) node_inputs: Vec<DryTensor<D>>,
    pub(crate) node_outputs: NodeOut<D, E>,
    pub(crate) unpadded_output_shapes: Vec<Shape>,
    pub(crate) unpadded_input_shapes: Vec<Shape>,
}
impl<D: Serialize + for<'a> Deserialize<'a>, E: ExtensionField> StepData<D, E> {
    /// Hydrate all the input tensors of the node corresponding to this step.
    pub(crate) fn input_tensors(&self, store: &mut GenStore) -> Result<Vec<Tensor<D>>, StoreError> {
        self.node_inputs
            .iter()
            .map(|dry| dry.hydrate(store.clone()))
            .collect::<Result<Vec<_>, StoreError>>()
    }

    /// Hydrate one of the input tensors of the node corresponding to this step.
    pub(crate) fn input_tensor_at(
        &self,
        i: usize,
        store: &mut GenStore,
    ) -> Result<Tensor<D>, StoreError> {
        self.node_inputs[i].hydrate(store.clone())
    }

    /// Hydrate all the output tensors of the node corresponding to this step.
    pub(crate) fn output_tensors(
        &self,
        store: &mut GenStore,
    ) -> Result<Vec<Tensor<D>>, StoreError> {
        self.node_outputs
            .outputs
            .iter()
            .map(|dry| dry.hydrate(store.clone()))
            .collect::<Result<Vec<_>, StoreError>>()
    }

    /// Hydrate one of the output tensors of the node corresponding to this step.
    pub(crate) fn output_tensor_at(
        &self,
        i: usize,
        store: &mut GenStore,
    ) -> Result<Tensor<D>, StoreError> {
        self.node_outputs.outputs[i].hydrate(store.clone())
    }
}

impl<E: ExtensionField> StepData<Element, E> {
    pub(crate) fn to_dequantize(
        &self,
        md: &ModelMetadata,
        store: GenStore,
        node_id: NodeId,
    ) -> Result<StepData<f32, E>, StoreError> {
        Ok(StepData {
            node_inputs: self
                .node_inputs
                .iter()
                .zip(md.layer_input_scaling_factor(node_id))
                .map(|(dry, scale_factor)| {
                    dry.dry_cast(store.clone(), |x| scale_factor.dequantize(x))
                })
                .collect::<Result<Vec<_>, StoreError>>()?,
            node_outputs: self.node_outputs.to_dequantize(md, store, node_id)?,
            unpadded_output_shapes: self.unpadded_output_shapes.clone(),
            unpadded_input_shapes: self.unpadded_input_shapes.clone(),
        })
    }
}
