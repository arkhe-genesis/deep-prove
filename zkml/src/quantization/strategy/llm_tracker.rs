//! LLM Inference tracker for more advanced quantisation strategies.

use dp_crypto::util::ceil_log2;

use super::*;
use crate::{
    layers::{
        Layer, provable::Evaluate, requant::FIXED_POINT_SCALE,
        transformer::positional::PositionalVariant,
    },
    lookup::table::SHIFT_CHECK_TABLE_BIT_SIZE,
    model::{LayerRunner, ModelGraph, RunInput, RunResult},
    parser::llm::metadata::LLMMetadata,
};
use std::collections::HashSet;

pub struct LLMInferenceTracker {
    /// Streaming estimator of the selected statistics for each output of each
    /// node.
    accumulators: HashMap<NodeOutput, InferenceTrackingAccumulator>,
    add_layers: HashSet<NodeOutput>,
    /// Streaming estimator of the selected statistics for given intermediate data of
    /// each node, if any
    intermediate_data_trackers: HashMap<(NodeId, TrackedDataId), InferenceTrackingAccumulator>,
    /// Streaming estimator of selected statistics that are global across a model. For instance
    /// the inputs and outputs of activation functions for working out suitable tables.
    global_data_trackers: HashMap<TrackedDataId, InferenceTrackingAccumulator>,
    /// The LLM metadata
    llm_metadata: LLMMetadata,
    /// Embeddings matrix
    embeddings: Option<WrappedTensor<f32>>,
    /// Positional embeddings matrix
    positional_embeddings: Option<WrappedTensor<f32>>,
}

impl LLMInferenceTracker {
    pub fn new(llm_metadata: LLMMetadata) -> Self {
        Self {
            accumulators: HashMap::new(),
            add_layers: HashSet::new(),
            intermediate_data_trackers: HashMap::new(),
            global_data_trackers: HashMap::new(),
            llm_metadata,
            embeddings: None,
            positional_embeddings: None,
        }
    }

    pub(crate) fn track<N>(
        &mut self,
        output: NodeOutput,
        layer: &Layer<N>,
        handle: &TensorHandle<N>,
    ) -> anyhow::Result<()>
    where
        N: TensorTypeParam,
    {
        if self.embeddings.is_none() && matches!(layer, Layer::Embeddings(..)) {
            let Layer::Embeddings(em) = layer else {
                unreachable!();
            };
            let wrapped_embeddings = N::tensor_to_float(em.mat.wrapped_tensor()?.clone());
            self.embeddings = Some(wrapped_embeddings);
        } else if self.positional_embeddings.is_none() && matches!(layer, Layer::Positional(..)) {
            let Layer::Positional(pos_layer) = layer else {
                unreachable!();
            };
            if matches!(pos_layer.variant, PositionalVariant::Absolute(..)) {
                let abs_pos = match &pos_layer.variant {
                    PositionalVariant::Absolute(abs) => abs,
                    _ => unreachable!(),
                };
                let wrapped_tensor =
                    N::tensor_to_float(abs_pos.positional.wrapped_tensor()?.clone());
                self.positional_embeddings = Some(wrapped_tensor);
            }
        }

        if let Layer::Add(_) = layer {
            self.add_layers.insert(output);
        }

        let accumulator = self
            .accumulators
            .entry(output)
            .or_insert_with(|| InferenceTrackingMode::MinMax.new_accumulator());
        let guard = handle.wrapped_tensor().unwrap();
        let tensor = guard.deref();
        for el in N::tensor_to_float(tensor.clone()).get_data() {
            accumulator.add(el as f64);
        }
        Ok(())
    }

    pub(crate) fn calculate_embeddings_max(&self) -> Result<f32> {
        ensure!(self.embeddings.is_some(), "Embeddings not tracked");
        let embeddings = self.embeddings.as_ref().unwrap();
        match self.positional_embeddings {
            None => Ok(embeddings.clone().max_abs().get_data()[0]),
            Some(ref pos_embeddings) => {
                let WrappedTensor::Rank2 {
                    tensor: embeddings, ..
                } = embeddings.clone()
                else {
                    anyhow::bail!("Embeddings tensor is not rank 2");
                };
                let WrappedTensor::Rank2 {
                    tensor: pos_embeddings,
                    ..
                } = pos_embeddings.clone()
                else {
                    anyhow::bail!("Positional embeddings tensor is not rank 2");
                };

                let embeddings_maxes = embeddings.max_abs_dim(0);
                let positional_maxes = pos_embeddings.max_abs_dim(0);

                let full_maxes = embeddings_maxes.add(positional_maxes).max_abs();
                let shape = full_maxes.shape();
                Ok(WrappedTensor::Rank1 {
                    tensor: full_maxes,
                    unpadded_shape: shape,
                }
                .max_abs()
                .get_data()[0])
            }
        }
    }

    pub(crate) fn is_add_layer(&self, output: NodeOutput) -> bool {
        self.add_layers.contains(&output)
    }

    pub(crate) fn llm_metadata(&self) -> &LLMMetadata {
        &self.llm_metadata
    }

    pub(crate) fn track_inputs<N>(
        &mut self,
        output: NodeOutput,
        handle: &TensorHandle<N>,
    ) -> anyhow::Result<()>
    where
        N: TensorTypeParam,
    {
        let mode = InferenceTrackingMode::NSigmas(2);
        let accumulator = self
            .accumulators
            .entry(output)
            .or_insert_with(|| mode.new_accumulator());
        let guard = handle.wrapped_tensor().unwrap();
        let tensor = guard.deref();
        for el in N::tensor_to_float(tensor.clone()).get_data() {
            accumulator.add(el as f64);
        }
        Ok(())
    }

    pub(crate) fn track_intermediate_data<N>(
        &mut self,
        node_id: NodeId,
        data_id: TrackedDataId,
        layer: &Layer<N>,
        wrapped_tensor: WrappedTensor<N>,
    ) where
        N: TensorTypeParam,
    {
        let mode = InferenceTrackingMode::MinMax;
        let accumulator = if let Layer::Activation(..) = layer {
            self.global_data_trackers
                .entry(data_id)
                .or_insert_with(|| mode.new_accumulator())
        } else {
            self.intermediate_data_trackers
                .entry((node_id, data_id))
                .or_insert_with(|| mode.new_accumulator())
        };

        for el in N::tensor_to_float(wrapped_tensor.clone()).get_data() {
            accumulator.add(el as f64);
        }
    }

    pub fn scaling_range(&self, output: NodeOutput) -> Option<(f32, f32)> {
        self.accumulators.get(&output).map(|a| a.range())
    }

    pub(crate) fn scaling_factor_for_intermediate_data(
        &self,
        node_id: NodeId,
        data_id: TrackedDataId,
    ) -> ScalingFactor {
        let maybe_global = self.global_data_trackers.get(&data_id);
        let tracking_accumulator = if let Some(global_accumulator) = maybe_global {
            global_accumulator
        } else {
            self.intermediate_data_trackers
                .get(&(node_id, data_id))
                .unwrap()
        };

        let (min, max) = tracking_accumulator.range();
        // This determines how many bits we have to allow for for additions in the following linear layer
        if maybe_global.is_some() {
            let block = self
                .llm_metadata
                .transformers
                .iter()
                .find(|b| b.ffn.activation_id == node_id)
                .unwrap();
            let uses_glu = block.ffn.uses_glu;

            let contraction_size = ceil_log2(self.llm_metadata.config.intermediate_size);
            let max_bits = 63usize;
            let available_bits = if !uses_glu {
                max_bits - contraction_size - *quantization::BIT_LEN - 1 - FIXED_POINT_SCALE
            } else {
                max_bits - contraction_size - 2 * *quantization::BIT_LEN - FIXED_POINT_SCALE
            };
            let bit_size = (available_bits - 1).min(SHIFT_CHECK_TABLE_BIT_SIZE - 1);
            ScalingFactor::from_absolute_max(
                min.abs().max(max.abs()),
                Some((-1 << bit_size, (1 << bit_size) - 1)),
            )
        } else {
            ScalingFactor::from_absolute_max(min.abs().max(max.abs()), None)
        }
    }
}

/// A runner used apply an inference tracker.
pub struct LLMTrackerRunner<'a, I> {
    pub inner: I,
    pub tracker: &'a mut LLMInferenceTracker,
}

impl<'a, I, N> LayerRunner<N, RunInput<N>> for LLMTrackerRunner<'a, I>
where
    I: LayerRunner<N, RunInput<N>>,
    N: TensorTypeParam,
{
    fn model_inputs(
        &mut self,
        graph: &ModelGraph<N>,
        inputs: &[TensorHandle<N>],
    ) -> anyhow::Result<()> {
        for (i, handle) in inputs.iter().enumerate() {
            let input_node_id = graph.input_node_id(i)?;
            let port = 0;
            self.tracker
                .track_inputs(input_node_id.output_at(port), handle)?;
        }
        self.inner.model_inputs(graph, inputs)?;
        Ok(())
    }

    fn run_layer(
        &mut self,
        node_id: NodeId,
        graph: &ModelGraph<N>,
        layer: &Layer<N>,
        inputs: &RunInput<N>,
    ) -> RunResult<N>
    where
        Layer<N>: Evaluate<N>,
    {
        let mut layer_out = self.inner.run_layer(node_id, graph, layer, inputs)?;

        let out_ports = graph.outgoing_ports(node_id);
        for (port, handle) in out_ports.iter().zip(&layer_out.outputs) {
            self.tracker
                .track(node_id.output_at(port.port), layer, handle)?;
        }

        for (data_id, handle) in layer_out.tracked_data.drain() {
            self.tracker
                .track_intermediate_data(node_id, data_id, layer, handle);
        }

        Ok(layer_out)
    }
}
