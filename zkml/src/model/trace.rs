use crate::{
    Element, IO, Shape, Tensor,
    graph::NodeId,
    layers::{Layer, NodeOut},
    quantization::{Fieldizer, ModelMetadata},
    tensor::DryTensor,
};
use anyhow::Context;
use ff_ext::ExtensionField;
use serde::{Deserialize, Serialize};
use std::{collections::HashMap, fmt::Debug};
use tenstore::{GenStore, StoreError};

#[derive(Default, Clone)]
pub struct Trace<'a, E: ExtensionField, N, D> {
    pub(crate) store: GenStore,
    pub(crate) steps: HashMap<NodeId, Step<'a, E, N, D>>,
    // TODO: convert to TensorKey
    pub(crate) input: Vec<DryTensor<D>>,
    pub(crate) output: Vec<DryTensor<D>>,
}
impl<'a, E: ExtensionField, N, D> Trace<'a, E, N, D> {
    pub fn new(store: GenStore, input: Vec<DryTensor<D>>) -> Self {
        Self {
            store,
            steps: Default::default(),
            input,
            output: Default::default(),
        }
    }
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
    pub(crate) fn get_step(&self, node_id: NodeId) -> Option<&Step<'a, E, N, D>> {
        self.steps.get(&node_id)
    }

    /// Insert the trace data `step` about node `node_id` in the trace
    pub(crate) fn new_step(&mut self, node_id: NodeId, step: Step<'a, E, N, D>) {
        assert!(!self.steps.contains_key(&node_id));
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
        let field_steps =
            self.steps
                .into_iter()
                .map(|(id, step)| {
                    Ok((
                        id,
                        Step {
                            op: step.op,
                            node_inputs: step
                                .node_inputs
                                .into_iter()
                                .map(|dry| {
                                    dry.dry_cast(store.clone(), |x| x.to_field()).with_context(
                                        || format!("converting `{:?}`", dry.storage_key()),
                                    )
                                })
                                .collect::<anyhow::Result<Vec<DryTensor<_>>>>()?,

                            node_outputs: step
                                .node_outputs
                                .into_fields(store.clone())
                                .context("converting node_outputs")?,
                            unpadded_output_shapes: step.unpadded_output_shapes,
                            unpadded_input_shapes: step.unpadded_input_shapes,
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
            .enumerate()
            .map(|(i, dry)| {
                let sf = md.input_scaling(i);
                dry.dry_cast(self.store.clone(), |x| sf.dequantize(x))
            })
            .collect::<Result<Vec<DryTensor<f32>>, StoreError>>()?;

        let outputs = self
            .output
            .iter()
            .enumerate()
            .map(|(i, dry)| {
                let sf = md.output_scaling(i);
                dry.dry_cast(self.store.clone(), |x| sf.dequantize(x))
            })
            .collect::<Result<Vec<DryTensor<f32>>, StoreError>>()?;

        let steps = self
            .steps
            .iter()
            .map(|(node_id, step)| {
                step.to_dequantize(md, self.store.clone(), *node_id)
                    .map(|step| (*node_id, step))
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

/// Data found in the trace for each node of the model
#[derive(Clone)]
pub struct Step<'a, E: ExtensionField, N, D> {
    /// The operation that generated this trace element.
    pub(crate) op: &'a Layer<N>,
    /// Ordered by input port (e.g. target_port of the incoming edges)
    pub(crate) node_inputs: Vec<DryTensor<D>>,
    /// Ordered by output port (e.g. source_port of the incoming edges)
    pub(crate) node_outputs: NodeOut<D, E>,
    /// Ordered by output port (e.g. source_port of the outgoing edges)
    pub(crate) unpadded_output_shapes: Vec<Shape>,
    /// Ordered by input port (e.g. target_port of the incoming edges)
    pub(crate) unpadded_input_shapes: Vec<Shape>,
}

impl<'a, E: ExtensionField, N, D> Step<'a, E, N, D>
where
    D: Serialize + for<'b> Deserialize<'b>,
{
    /// Returns the output tensors of the node
    pub fn outputs(&self) -> &[DryTensor<D>] {
        self.node_outputs.outputs.as_slice()
    }

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

impl<'a, E: ExtensionField, N> Step<'a, E, N, Element> {
    pub fn to_dequantize(
        &self,
        md: &ModelMetadata,
        store: GenStore,
        node_id: NodeId,
    ) -> Result<Step<'a, E, N, f32>, StoreError> {
        Ok(Step {
            op: self.op,
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
