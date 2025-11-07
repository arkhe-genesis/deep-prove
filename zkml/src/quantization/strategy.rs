use super::ScalingFactor;
use crate::{
    Element, Shape, Tensor,
    graph::{Node, NodeId, NodeOutput, PortId},
    layers::provable::{OpInfo, QuantizeOp, TrackedDataId},
    model::{Model, transform::apply_transformations},
    number::Number,
    padding::PaddingMode,
    quantization::{self, ModelMetadata, metadata::MetadataBuilder},
    rng_from_env_or_random,
};
use anyhow::{Result, anyhow, ensure};
use average::{Estimate, Max, Min, Quantile, Variance};
use ff_ext::GoldilocksExt2;
use itertools::Itertools;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};
use tenstore::GenStore;
use tracing::{debug, info, warn};

/// Trait for quantizing a float-based model into a quantized model. The current implementation
/// simply looks at the absolute maximum value of the model and uses that as the scaling factor
/// to quantize the model, one scaling factor per layer.
pub trait ScalingStrategy: std::fmt::Debug {
    type AuxData: Sized;

    fn quantize(
        &self,
        model: Model<f32>,
        store: &mut GenStore,
    ) -> Result<(Model<Element>, ModelMetadata)>;

    /// Returns the scaling factors for the outputs of the node with the given ID. The number of
    /// outputs is given by the `num_outputs` parameter. The scaling factors are computed based on
    /// the auxiliary data provided.
    fn scaling_factors_for_node(
        data: &Self::AuxData,
        node_id: NodeId,
        num_outputs: usize,
    ) -> Vec<ScalingFactor>;

    fn scaling_factor_for_intermediate_data(
        data: &Self::AuxData,
        node_id: NodeId,
        data_id: TrackedDataId,
    ) -> ScalingFactor;

    fn name(&self) -> String;
}

/// Implementors of [`ScalingStrategy`]
#[derive(Debug, Clone, Copy, derive_more::Display, Serialize, Deserialize)]
pub enum ScalingStrategyKind {
    InferenceObserver,
    AbsoluteMax,
}

/// Quantization strategy that observes the inference of the model with different inputs and uses the
/// min/max values of the output to determine the output scaling factor of each layer that needs
/// requantization afterwards.
#[derive(Debug)]
pub struct InferenceObserver {
    inputs: Vec<Vec<Vec<f32>>>,
}

impl Default for InferenceObserver {
    fn default() -> Self {
        Self::new()
    }
}

impl InferenceObserver {
    pub fn new_with_representative_input(inputs: Vec<Vec<Vec<f32>>>) -> Self {
        Self { inputs }
    }
    pub fn new() -> Self {
        Self { inputs: vec![] }
    }
}

// TODO: replace that with the actual input node ID
const INPUT_TRACKING_ID: usize = 10_000;

impl ScalingStrategy for InferenceObserver {
    type AuxData = InferenceTracker;

    fn name(&self) -> String {
        format!("inference [{},{}]", *quantization::MIN, *quantization::MAX)
    }

    fn quantize(
        &self,
        model: Model<f32>,
        store: &mut GenStore,
    ) -> Result<(Model<Element>, ModelMetadata)> {
        let tracking_mode = InferenceTrackingMode::MinMax;
        // Alternatively:
        // let tracking_mode = InferenceTrackingMode::Quantiles(0.05, 0.95);
        // let tracking_mode = InferenceTrackingMode::NSigmas(3);
        let mut tracker = InferenceTracker::new(tracking_mode);
        let input_shapes = model.input_shapes();
        let unpadded_input_shapes = model.unpadded_input_shapes();
        let inputs = if self.inputs.is_empty() {
            let mut rng = rng_from_env_or_random();
            warn!("No representative inputs provided, generating random ones");
            (0..1)
                .map(|_| {
                    input_shapes
                        .iter()
                        .map(|shape| {
                            let size = shape.product();
                            (0..size)
                                .map(|_| <f32 as Number>::random(&mut rng))
                                .collect_vec()
                        })
                        .collect_vec()
                })
                .collect()
        } else {
            info!(
                "Using the {} provided representative inputs to quantize model",
                self.inputs.len()
            );
            self.inputs.clone()
        };
        // 1. Run the inference multiple times with different inputs
        // TODO: integrate that within model.rs in a more elegant way with inference step - currently problematic
        // because of the generics and FFT requirement to take a field
        let mut nsamples = 0;
        for input in inputs.into_iter() {
            let input_tensors = input
                .into_iter()
                .zip(model.unpadded_input_shapes())
                .enumerate()
                .map(|(i, (inp, shape))| {
                    let input_tensor = Tensor::new(shape, inp)?;
                    tracker.track(INPUT_TRACKING_ID.into(), i, input_tensor.clone());
                    Ok(input_tensor)
                })
                .collect::<Result<Vec<_>>>()?;
            debug!("Running float inference with the {}-th input", nsamples + 1);
            model.run_with_tracker::<GoldilocksExt2>(input_tensors, Some(&mut tracker), store)?;
            nsamples += 1;
        }
        info!("InferenceObserver: {} total samples observed", nsamples);
        // 2. get the scaling factor of the input
        let num_model_inputs = unpadded_input_shapes.len();
        let input_scaling = (0..num_model_inputs)
            .map(|i| {
                let (input_min, input_max) = tracker.scaling_range(INPUT_TRACKING_ID.into(), i);
                ScalingFactor::from_absolute_max(input_min.abs().max(input_max.abs()), None)
            })
            .collect_vec();
        quantize_model::<InferenceObserver>(model, tracker, input_scaling)
    }

    fn scaling_factors_for_node(
        tracker: &InferenceTracker,
        node_id: NodeId,
        num_outputs: usize,
    ) -> Vec<ScalingFactor> {
        (0..num_outputs)
            .map(|i| {
                let (min, max) = tracker.scaling_range(node_id, i);
                ScalingFactor::from_absolute_max(min.abs().max(max.abs()), None)
            })
            .collect()
    }

    fn scaling_factor_for_intermediate_data(
        tracker: &InferenceTracker,
        node_id: NodeId,
        data_id: TrackedDataId,
    ) -> ScalingFactor {
        tracker.scaling_factor_for_intermediate_data(node_id, data_id)
    }
}

/// The inference tracker observes the execution of a model over a given set of
/// inputs to determine adequate scaling factors for each node.
pub struct InferenceTracker {
    /// What statistics to estimate.
    mode: InferenceTrackingMode,
    /// Streaming estimator of the selected statistics for each output of each
    /// node.
    accumulators: HashMap<(NodeId, usize), InferenceTrackingAccumulator>,
    /// Streaming estimator of the selected statistics for given intermediate data of
    /// each node, if any
    intermediate_data_trackers: HashMap<(NodeId, TrackedDataId), InferenceTrackingAccumulator>,
}
/// Selects the statistic to use to generate the scaling range.
enum InferenceTrackingMode {
    /// Register the min. and max. values encountered for each node.
    MinMax,
    /// Estimate the p- and q-quantiles from the encountered distribution of values.
    #[allow(dead_code)]
    Quantiles(f32, f32),
    /// Assume the distribution is gaussian and return the mean +/- n std. dev.
    #[allow(dead_code)]
    NSigmas(i32),
}
impl InferenceTrackingMode {
    /// Return a new accumulator adequate for this tracking mode.
    fn new_accumulator(&self) -> InferenceTrackingAccumulator {
        match self {
            InferenceTrackingMode::MinMax => {
                InferenceTrackingAccumulator::MinMax(Min::new(), Max::new())
            }
            InferenceTrackingMode::Quantiles(p, q) => InferenceTrackingAccumulator::Quantiles(
                Box::new(Quantile::new(*p as f64)),
                Box::new(Quantile::new(*q as f64)),
            ),
            InferenceTrackingMode::NSigmas(n) => {
                InferenceTrackingAccumulator::NSigmas(*n as f32, Variance::new())
            }
        }
    }
}
/// Aggregate, in a streaming fashion, statistics for the encountered values in
/// the given output of a node.
enum InferenceTrackingAccumulator {
    MinMax(Min, Max),
    Quantiles(Box<Quantile>, Box<Quantile>),
    NSigmas(f32, Variance),
}
impl InferenceTrackingAccumulator {
    fn scaling_range(&self) -> (f32, f32) {
        match self {
            InferenceTrackingAccumulator::MinMax(min, max) => (min.min() as f32, max.max() as f32),
            InferenceTrackingAccumulator::Quantiles(p, q) => {
                (p.quantile() as f32, q.quantile() as f32)
            }
            InferenceTrackingAccumulator::NSigmas(n, variance) => (
                variance.mean() as f32 - n * variance.sample_variance() as f32,
                variance.mean() as f32 + n * variance.sample_variance() as f32,
            ),
        }
    }

    fn add(&mut self, x: f64) {
        match self {
            InferenceTrackingAccumulator::MinMax(min, max) => {
                min.add(x);
                max.add(x);
            }
            InferenceTrackingAccumulator::Quantiles(p, q) => {
                p.add(x);
                q.add(x);
            }
            InferenceTrackingAccumulator::NSigmas(_, variance) => {
                variance.add(x);
            }
        }
    }
}
impl InferenceTracker {
    fn new(mode: InferenceTrackingMode) -> Self {
        Self {
            mode,
            accumulators: HashMap::new(),
            intermediate_data_trackers: HashMap::new(),
        }
    }
    pub(crate) fn track(&mut self, node_id: NodeId, output_index: usize, output: Tensor<f32>) {
        let accumulator = self
            .accumulators
            .entry((node_id, output_index))
            .or_insert_with(|| self.mode.new_accumulator());
        for x in output.get_data() {
            accumulator.add(*x as f64);
        }
    }

    pub(crate) fn track_intermediate_data(
        &mut self,
        node_id: NodeId,
        data_id: TrackedDataId,
        data: Tensor<f32>,
    ) {
        let accumulator = self
            .intermediate_data_trackers
            .entry((node_id, data_id))
            .or_insert_with(|| self.mode.new_accumulator());
        for x in data.get_data() {
            accumulator.add(*x as f64);
        }
    }

    pub(crate) fn scaling_range(&self, node_id: NodeId, output_index: usize) -> (f32, f32) {
        self.accumulators
            .get(&(node_id, output_index))
            .unwrap()
            .scaling_range()
    }

    pub(crate) fn scaling_factor_for_intermediate_data(
        &self,
        node_id: NodeId,
        data_id: TrackedDataId,
    ) -> ScalingFactor {
        let (min, max) = self
            .intermediate_data_trackers
            .get(&(node_id, data_id))
            .unwrap()
            .scaling_range();
        ScalingFactor::from_absolute_max(min.abs().max(max.abs()), None)
    }
}

#[derive(Debug)]
pub struct AbsoluteMax(Option<Vec<Vec<f32>>>);

impl Default for AbsoluteMax {
    fn default() -> Self {
        Self::new()
    }
}

impl AbsoluteMax {
    pub fn new_with_representative_input(input: Vec<Vec<f32>>) -> Self {
        Self(Some(input))
    }
    pub fn new() -> Self {
        Self(None)
    }
}

impl ScalingStrategy for AbsoluteMax {
    type AuxData = ();

    fn name(&self) -> String {
        "absolute_max".to_string()
    }

    fn quantize(
        &self,
        model: Model<f32>,
        _store: &mut GenStore,
    ) -> Result<(Model<Element>, ModelMetadata)> {
        let input_scaling_factor = if let Some(ref input) = self.0 {
            let input_tensor = model.load_input_flat(input.clone())?;
            model
                .input_shapes()
                .into_iter()
                .zip(&input_tensor)
                .try_for_each(|(shape, input)| {
                    ensure!(
                        shape == *input.shape(),
                        "input shape mismatch: expected {:?}, got {:?}",
                        shape,
                        input.shape()
                    );
                    Ok(())
                })?;
            input_tensor
                .into_iter()
                .map(|input| ScalingFactor::from_absolute_max(input.max_abs_output(), None))
                .collect_vec()
        } else {
            (0..model.num_inputs())
                .map(|_| ScalingFactor::default())
                .collect_vec()
        };
        quantize_model::<AbsoluteMax>(model, (), input_scaling_factor)
    }

    fn scaling_factors_for_node(
        _data: &Self::AuxData,
        _node_id: NodeId,
        num_outputs: usize,
    ) -> Vec<ScalingFactor> {
        vec![ScalingFactor::default(); num_outputs]
    }

    fn scaling_factor_for_intermediate_data(
        _data: &Self::AuxData,
        _node_id: NodeId,
        _data_id: TrackedDataId,
    ) -> ScalingFactor {
        ScalingFactor::default()
    }
}

fn quantize_model<S: ScalingStrategy>(
    model: Model<f32>,
    data: S::AuxData,
    input_scaling: Vec<ScalingFactor>,
) -> anyhow::Result<(Model<Element>, ModelMetadata)> {
    let input_shapes = model.input_shapes();
    let input_not_padded_shapes = model.unpadded_input_shapes();
    let mut md = MetadataBuilder::new();
    let mut requant_layers = vec![];
    let mut transforms = vec![];
    // Accumulate all the layer output shapes as they are being encountered,
    // required for `quantize_op`.
    let mut shape_map = HashMap::<NodeOutput, Shape>::new();
    let quantized_graph = model
        .graph
        // we create the quantized graph going in the inference order sometimes
        // some layer may need to know some parts of the previously visited
        // nodes
        //
        // XXX: is it true?
        .try_into_map_forward(|node_id, node, incoming_feeds| {
            Ok(match node {
                Node::Inner(layer) => {
                    tracing::debug!(
                        "Quantising node {}, with node ID {node_id}",
                        layer.short_name()
                    );
                    let shape_info = incoming_feeds
                        .iter()
                        .try_fold(BTreeMap::<PortId, Shape>::new(), |mut ax, in_feed| {
                            let shape = shape_map
                                .get(&in_feed.source)
                                .ok_or(anyhow!("fetching shape info for {:?}", in_feed.source))?;
                            ax.insert(in_feed.target.port, shape.clone());
                            anyhow::Result::<BTreeMap<PortId, Shape>>::Ok(ax)
                        })?
                        .into_values()
                        .collect::<Vec<Shape>>();

                    // Ordered list of the input scalings for each input port of
                    // this node. The ordering should be respected thanks to
                    // `.incomings`
                    let input_scalings = incoming_feeds
                        .iter()
                        .map(|feed| md.get_output_layer_scaling(feed.source))
                        .collect::<Result<Vec<_>>>()?;

                    let unpadded_output_shapes =
                        layer.output_shapes(&shape_info, PaddingMode::NoPadding)?;
                    let num_outputs = unpadded_output_shapes.len();
                    let output_scalings = S::scaling_factors_for_node(&data, node_id, num_outputs);

                    // Compute the quantization for this node
                    let quantized_out = layer.quantize_op::<S>(
                        &data,
                        node_id,
                        &input_scalings,
                        &shape_info,
                        &output_scalings,
                        &unpadded_output_shapes,
                    )?;
                    // Save this layer output scaling factors
                    md.insert_layer_scalings(
                        node_id,
                        quantized_out.output_scalings,
                        input_scalings,
                    );

                    // Extend the shape register with the output shapes for the
                    // current node.
                    shape_map.extend(
                        quantized_out
                            .quantized_op
                            .output_shapes(&shape_info, PaddingMode::NoPadding)?
                            .into_iter()
                            .enumerate()
                            .map(|(out_port, shape)| (NodeOutput::new(node_id, out_port), shape)),
                    );
                    if let Some(requant) = quantized_out.requant_layer {
                        requant_layers.push((node_id, requant));
                    }
                    if let Some(transform) = quantized_out.post_quant_rule {
                        transforms.push(transform);
                    }
                    Node::Inner(quantized_out.quantized_op)
                }
                // Looks silly, but the `Node` are not actually of the same
                // types left & right
                Node::Input(i) => {
                    md.insert_layer_scalings(node_id, vec![input_scaling[i]], vec![]);
                    shape_map.insert(
                        NodeOutput::new(node_id, 0),
                        input_not_padded_shapes[i].clone(),
                    );
                    Node::Input(i)
                }
                Node::Output(o) => Node::Output(o),
            })
        })?;
    let mut model = Model::new_from_shapes(input_not_padded_shapes, input_shapes, quantized_graph);

    // add scaling factor to `md` for requant layers: the scaling factors of
    // the inputs correspond to the scaling factors of the outputs of the
    // previous node
    for (input_node_id, requant) in requant_layers {
        let requant_ids = model.add_requant_layer(requant, input_node_id)?;
        for (i, requant_id) in requant_ids.into_iter().enumerate() {
            let node_out = NodeOutput::new(input_node_id, i);
            let scaling_factor = md.get_output_layer_scaling(node_out)?;

            md.insert_layer_scalings(requant_id, vec![scaling_factor], vec![scaling_factor]);
        }
    }

    // Apply any model transformations
    model = apply_transformations(model, transforms)?;
    let md = md.build(model.graph.input_node_ids(), model.graph.output_node_ids())?;
    info!("Quantized model with {} layers", model.graph.node_count());
    Ok((model, md))
}
