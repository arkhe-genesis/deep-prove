use crate::{
    Element, ScalingStrategy, Shape,
    graph::{NodeId, PortLink},
    layers::{
        Layer,
        activation::Activation,
        convolution::Convolution,
        pooling::{MAXPOOL2D_KERNEL_SIZE, Maxpool2D, Pooling},
    },
    model::Model,
    padding::pad_model,
    quantization::{AbsoluteMax, ModelMetadata},
    tensor::KeyedTensor,
};
use anyhow::{Context, Error, Result, bail, ensure};
use either::Either;
use std::{collections::HashMap, iter::Peekable};
use tenstore::GenStore;
use tracing::debug;
use tract_onnx::{
    pb::ModelProto,
    prelude::*,
    tract_core::{
        self,
        ops::{
            binary::TypedBinOp,
            cnn::{Conv, MaxPool},
            einsum::EinSum,
            source::TypedSource,
        },
    },
    tract_hir::{
        internal::AxisOp,
        ops::{cnn::PaddingSpec, konst::Const},
    },
};

/// Unified model loading function for onnx models
pub fn load_float_model(model: &ModelProto) -> Result<Model<f32>> {
    let model = from_proto(model)?;
    model.describe();
    Ok(model)
}

/// Utility struct for loading a onnx model with float weights and producing a quantized model
/// that can be used for inference and proving.
#[derive(Debug)]
pub struct FloatOnnxLoader<'a, S: ScalingStrategy> {
    /// Either a path to model file or memmap'd bytes
    model: Either<String, &'a [u8]>,
    scaling_strategy: S,
    keep_float: bool,
}

pub type DefaultFloatOnnxLoader<'a> = FloatOnnxLoader<'a, AbsoluteMax>;

impl DefaultFloatOnnxLoader<'_> {
    pub fn new(model_path: &str) -> Self {
        Self::new_with_scaling_strategy(model_path, AbsoluteMax::new())
    }
}

impl<'a, S: ScalingStrategy> FloatOnnxLoader<'a, S> {
    pub fn new_with_scaling_strategy(model_path: &str, scaling_strategy: S) -> Self {
        Self {
            model: Either::Left(model_path.to_string()),
            scaling_strategy,
            keep_float: false,
        }
    }
    pub fn from_bytes_with_scaling_strategy(model_bytes: &'a [u8], scaling_strategy: S) -> Self {
        Self {
            model: Either::Right(model_bytes),
            scaling_strategy,
            keep_float: false,
        }
    }

    pub fn with_scaling_strategy(mut self, scaling_strategy: S) -> Self {
        self.scaling_strategy = scaling_strategy;
        self
    }

    pub fn with_keep_float(mut self, keep_float: bool) -> Self {
        self.keep_float = keep_float;
        self
    }

    pub fn build(self) -> Result<(Model<Element>, ModelMetadata)> {
        let proto = match self.model {
            Either::Left(path) => load_proto_from_path(&path)?,
            Either::Right(bytes) => {
                use prost_tract_compat::Message;
                ModelProto::decode(bytes)
                    .map_err(|e| Error::msg(format!("Failed to load model: {e:?}")))?
            }
        };
        let float_model = load_float_model(&proto)?;
        debug!("Input shape: {:?}", float_model.input_shapes());
        let mut kept_float = None;
        if self.keep_float {
            kept_float = Some(float_model.clone());
        }

        // NOTE: this is running with the default store, which is reasonable for the current use.
        // We may wish to change the store type depending on the workload in the future.
        let (quantized_model, mut md) = self
            .scaling_strategy
            .quantize(float_model, &mut GenStore::default())?;
        let padded_model = pad_model(quantized_model)?;
        md.float_model = kept_float;
        Ok((padded_model, md))
    }
}

fn load_proto_from_path(path: &str) -> Result<ModelProto> {
    tract_onnx::onnx()
        .proto_model_for_path(path)
        .map_err(|e| Error::msg(format!("Failed to load model: {e:?}")))
}
type OnnxModel = Graph<TypedFact, Box<dyn TypedOp + 'static>>;
type OnnxNode = Node<TypedFact, Box<dyn TypedOp + 'static>>;

macro_rules! ensure_onnx {
        // Match with format args
        ($cond:expr, $err_fmt:literal, $($args:expr),+ $(,)?) => {
            ensure!($cond,
                "when parsing onnx model: {}",
                format!($err_fmt, $($args),+),
            );
        };
        // Match with plain string (no args)
        ($cond:expr, $err_msg:literal $(,)?) => {
            ensure!($cond,
                "when parsing onnx model: {}",
                $err_msg,
            );
        };
    }

pub fn from_proto(proto: &ModelProto) -> Result<Model<f32>> {
    let model = tract_onnx::onnx().model_for_proto_model(proto)?;
    from_inference_model(model)
}

pub fn from_path(path: &str) -> Result<Model<f32>> {
    let model = tract_onnx::onnx().model_for_path(path)?;
    from_inference_model(model)
}

fn from_inference_model(model: InferenceModel) -> Result<Model<f32>> {
    let model = {
        let pmodel = model.into_typed()?.into_decluttered()?;
        // so far we dont support batching
        let mut values = SymbolValues::default();
        let symbol = pmodel.sym("batch_size");
        values.set(&symbol, 1);
        pmodel.concretize_dims(&values)?
    };

    let plan = SimplePlan::new(model)?;
    let onnx_model = plan.model();
    let inference_order = plan.order_without_consts();
    let input_node = onnx_model.node(inference_order[0]);
    let input_source = downcast_to::<TypedSource>(input_node)?;
    debug!("onnx input_source: {:?}", input_source.fact.shape.to_tvec());
    let mut model_input_shape = input_source
        .fact
        .shape
        .to_tvec()
        .into_iter()
        .map(|x| tdim_to_usize(&x))
        .collect::<Result<Shape, _>>()?;
    // remove batch dimension if it's 1 as we dont support batching yet
    if model_input_shape[0] == 1 {
        model_input_shape.remove(0);
    }

    let mut pmodel = Model::new_from_input_shapes(vec![model_input_shape.clone()]);
    let mut it = inference_order[1..].iter().peekable();

    let mut input_mapping = HashMap::new();
    for input in onnx_model.input_outlets().unwrap().iter().enumerate() {
        input_mapping.insert(
            input.0,
            pmodel
                .graph()
                .input_nodes()
                .find(|(_, i)| **i == input.0)
                .unwrap_or_else(|| panic!("{:?} not found", input))
                .0,
        );
    }
    let mut parser = ParserFactory::init(input_mapping);

    while let Some(id) = parser
        .parse_node(onnx_model, &mut pmodel, &mut it)
        .transpose()?
    {
        debug!("parsed node id: {:?}", id);
    }

    for (i, outlet) in onnx_model.output_outlets()?.iter().enumerate() {
        let output_node_id = pmodel.graph_mut().add_output(i)?;
        pmodel.add_raw_edge(
            parser.node_mapping[&outlet.node],
            output_node_id,
            (outlet.slot, 0),
        )?;
    }
    Ok(pmodel)
}

type LoadFn<'a, I> = fn(
    node_mapping: &mut HashMap<usize, NodeId>,
    onnx: &OnnxModel,
    model: &mut Model<f32>,
    node_id: usize,
    node: &OnnxNode,
    iter: &mut Peekable<I>,
) -> Result<NodeId>;

struct ParserFactory<'a, I: Iterator<Item = &'a usize> + Sized> {
    parsers: HashMap<&'static str, LoadFn<'a, I>>,
    node_mapping: HashMap<usize, NodeId>,
}

impl<'a, I: Iterator<Item = &'a usize> + Sized> ParserFactory<'a, I> {
    fn init(node_mapping: HashMap<usize, NodeId>) -> Self {
        let mut m = HashMap::new();
        m.insert("Conv", load_conv as LoadFn<'a, I>);
        m.insert("Gemm.ab", load_gemm as LoadFn<'a, I>);
        m.insert("MatMul", load_gemm as LoadFn<'a, I>); //ToDo: currently MatMul is only used for dense layers without bias;
        // we would probably need an ad-hoc method when introducing general purpose matrix multiplication layer
        m.insert("Relu", load_relu as LoadFn<'a, I>);
        m.insert("Flatten", load_flatten as LoadFn<'a, I>);
        m.insert("Pool", load_maxpool as LoadFn<'a, I>);
        m.insert("Reshape", load_reshape as LoadFn<'a, I>);
        ParserFactory {
            parsers: m,
            node_mapping,
        }
    }

    fn parse_node(
        &mut self,
        onnx: &OnnxModel,
        model: &mut Model<f32>,
        iter: &mut Peekable<I>,
    ) -> Option<Result<NodeId>> {
        let curr_node_id = iter.next()?;
        let curr_node = onnx.node(*curr_node_id);
        debug!(
            "curr_node id {}: {:?} : {:?} <- inputs: {:?}",
            curr_node_id, curr_node.name, curr_node.name, curr_node.inputs
        );
        let op_name = &curr_node.name;
        let op_name_lower = op_name.to_lowercase();
        let Some(layer_name) = self
            .parsers
            .keys()
            .find(|&&layer_name| op_name_lower.contains(&layer_name.to_lowercase()))
        else {
            return Some(err(format!("Unknown node type: {op_name}: {curr_node:?}")));
        };
        debug!("current node {:?}", curr_node.op);
        let parser = self.parsers.get(layer_name).unwrap();

        match parser(
            &mut self.node_mapping,
            onnx,
            model,
            *curr_node_id,
            curr_node,
            iter,
        ) {
            Ok(zkml_node_id) => {
                debug!(
                    "parsed node id: {:?} -> {zkml_node_id} : {}",
                    curr_node_id,
                    model.graph().node(zkml_node_id).unwrap().describe(),
                );
                Some(Ok(zkml_node_id))
            }
            Err(e) => Some(err(format!(
                "Failed to parse node: {op_name}: {curr_node:?}: {:?}",
                e
            ))),
        }
    }
}

fn load_reshape<'a, I: Iterator<Item = &'a usize> + Sized>(
    node_mapping: &mut HashMap<usize, NodeId>,
    _onnx: &OnnxModel,
    model: &mut Model<f32>,
    node_id: usize,
    node: &OnnxNode,
    _iter: &mut Peekable<I>,
) -> Result<NodeId> {
    ensure_onnx!(
        node.inputs.len() == 1,
        "Reshape {} must have 1 input",
        node.name
    );
    let reshape_node = downcast_to::<AxisOp>(node)?;
    let AxisOp::Reshape(_, current_shape, new_shape) = reshape_node else {
        return err(format!("Reshape {} is not a Reshape node", node.name));
    };
    let current_shape: Shape = current_shape
        .iter()
        .map(tdim_to_usize)
        .collect::<Result<Vec<_>>>()?
        .into();
    let new_shape: Shape = new_shape
        .iter()
        .map(tdim_to_usize)
        .collect::<Result<Vec<_>>>()?
        .into();
    ensure_onnx!(
        current_shape.product() == new_shape.product(),
        "Reshape {} has incompatible shapes: {:?} -> {:?}",
        node.name,
        current_shape,
        new_shape
    );
    // Currently we only support reshape to flatten so we enforce that the reshape is a flattening operation
    ensure_onnx!(
        new_shape.rank() == 1,
        "Reshape {} is not a flattening operation: only supported operation is flattening WIP",
        node.name
    );
    let zkml_node_id = model
        .graph_mut()
        .add_inner(Layer::Flatten(crate::layers::flatten::Flatten(false)))?;
    model.add_edge(
        node_mapping[&node.inputs[0].node],
        zkml_node_id,
        (node.inputs[0].slot, 0),
    )?;
    node_mapping.insert(node_id, zkml_node_id);
    Ok(zkml_node_id)
}

fn load_flatten<'a, I: Iterator<Item = &'a usize> + Sized>(
    node_mapping: &mut HashMap<usize, NodeId>,
    _onnx: &OnnxModel,
    model: &mut Model<f32>,
    node_id: usize,
    node: &OnnxNode,
    _iter: &mut Peekable<I>,
) -> Result<NodeId> {
    ensure_onnx!(
        node.inputs.len() == 1,
        "Flatten {} must have 1 input",
        node.name
    );
    let zkml_node_id = model
        .graph_mut()
        .add_inner(Layer::Flatten(crate::layers::flatten::Flatten(false)))?;
    model.add_edge(
        node_mapping[&node.inputs[0].node],
        zkml_node_id,
        (node.inputs[0].slot, 0),
    )?;
    node_mapping.insert(node_id, zkml_node_id);
    Ok(zkml_node_id)
}

fn load_maxpool<'a, I: Iterator<Item = &'a usize> + Sized>(
    node_mapping: &mut HashMap<usize, NodeId>,
    _onnx: &OnnxModel,
    model: &mut Model<f32>,
    node_id: usize,
    node: &OnnxNode,
    _iter: &mut Peekable<I>,
) -> Result<NodeId> {
    ensure_onnx!(
        node.inputs.len() == 1,
        "MaxPool {} must have 1 input",
        node.name
    );
    let max_node = downcast_to::<MaxPool>(node)?;
    let expected_value: usize = MAXPOOL2D_KERNEL_SIZE;
    if let Some(ref strides) = max_node.pool_spec.strides {
        ensure_onnx!(
            strides.iter().all(|&x| x == expected_value),
            "Strides must be {}",
            expected_value
        );
    }
    match max_node.pool_spec.padding {
        PaddingSpec::Explicit(ref pad0, ref pad1) => {
            ensure_onnx!(
                pad0.iter().all(|&x| x == 0) && pad1.iter().all(|&x| x == 0),
                "Padding must be 0s"
            );
        }
        PaddingSpec::ExplicitOnnxPool(ref pad0, ref pad1, _) => {
            ensure_onnx!(
                pad0.iter().all(|&x| x == 0) && pad1.iter().all(|&x| x == 0),
                "Padding must be 0s"
            );
        }
        PaddingSpec::Valid => (),
        _ => {
            return err(format!(
                "Padding for {} must have valid padding {:?}",
                node.name, max_node.pool_spec.padding
            ));
        }
    }
    ensure_onnx!(
        max_node
            .pool_spec
            .kernel_shape
            .iter()
            .all(|&x| x == expected_value),
        "Kernel shape must be square with size {}",
        expected_value
    );
    if let Some(ref dil) = max_node.pool_spec.dilations {
        ensure_onnx!(dil.iter().all(|&x| x == 1), "Dilations must be 1");
    }
    let zkml_maxpool = Layer::Pooling(Pooling::Maxpool2D(Maxpool2D::default()));
    let zkml_node_id = model.graph_mut().add_inner(zkml_maxpool)?;
    model.add_edge(
        node_mapping[&node.inputs[0].node],
        zkml_node_id,
        (node.inputs[0].slot, 0),
    )?;
    node_mapping.insert(node_id, zkml_node_id);
    Ok(zkml_node_id)
}

fn load_relu<'a, I: Iterator<Item = &'a usize> + Sized>(
    node_mapping: &mut HashMap<usize, NodeId>,
    onnx: &OnnxModel,
    model: &mut Model<f32>,
    node_id: usize,
    node: &OnnxNode,
    _iter: &mut Peekable<I>,
) -> Result<NodeId> {
    // find the input node that corresponds to the const input of Relu - since
    // tract_onnx transforms a relu operation into Max(input, Const(0)) the
    // input node would be the other one.
    ensure_onnx!(
        node.inputs.len() == 2,
        "Relu {} must have 2 inputs",
        node.name
    );
    let real_input_id = match onnx.node(node.inputs[1].node).op_as::<Const>() {
        Some(_) => node.inputs[0],
        None => {
            ensure_onnx!(
                onnx.node(node.inputs[0].node).op_as::<Const>().is_some(),
                "Relu {} has no constant input",
                node.name
            );
            node.inputs[1]
        }
    };
    let zkml_node_id = model
        .graph_mut()
        .add_inner(Layer::Activation(Activation::new_relu()))?;
    model.add_edge(
        node_mapping[&real_input_id.node],
        zkml_node_id,
        PortLink::new(real_input_id.slot, 0),
    )?;
    node_mapping.insert(node_id, zkml_node_id);
    Ok(zkml_node_id)
}

fn load_gemm<'a, I: Iterator<Item = &'a usize> + Sized>(
    node_mapping: &mut HashMap<usize, NodeId>,
    onnx: &OnnxModel,
    model: &mut Model<f32>,
    node_id: usize,
    node: &OnnxNode,
    iter: &mut Peekable<I>,
) -> Result<NodeId> {
    let _matrix = downcast_to::<EinSum>(node)
        .with_context(|| format!("Gemm {} is not a EinSum node", node.name))?;
    // TODO: we only support matvec for now for onnx models
    // Fetch the input which is constant (e.g. the weights)
    ensure_onnx!(
        node.inputs.len() == 2,
        "Gemm {} must have 2 inputs",
        node.name
    );
    let Some(weight_link) = node
        .inputs
        .iter()
        .rev()
        .find(|&x| is_const(onnx.node(x.node)))
    else {
        return err(format!("Gemm {} has no constant input", node.name));
    };
    let mut weight = extract_const_tensor(onnx.node(weight_link.node))?;
    let weight_shape = weight.shape().clone();
    if weight_shape.len() > 2 {
        let input_flattened = weight_shape[1..].iter().product::<usize>();
        weight.reshape(Shape::new(vec![weight_shape[0], input_flattened]))?;
    } else if weight_shape.len() == 1 {
        // A Gemm is always a matrix - so if there's only one dimension, we need to add 1 to
        // to the output features
        weight.reshape(weight_shape.insert(0, 1))?;
    };
    ensure_onnx!(
        weight.shape().is_matrix(),
        "Weight for Gemm must be a matrix: {:?}",
        weight.shape()
    );
    // find the input node
    let Some(input_link) = node.inputs.iter().find(|&x| x.node != weight_link.node) else {
        return err(format!("Gemm {} has no input", node.name));
    };

    // check if the weight matrix needs to be transposed
    let input_node = onnx.node(input_link.node);
    let raw_input_shape = get_node_output_shape(input_node, input_link.slot)?;
    let input_size_flattened = raw_input_shape.iter().product::<usize>();
    let mut input_shape = vec![input_size_flattened];

    if weight_shape.len() > 2 {
        let weight_size_flattened = weight.data().len();
        ensure_onnx!(
            weight_size_flattened % input_size_flattened == 0,
            "Weight size {} is not divisible by input size {}",
            weight_size_flattened,
            input_size_flattened
        );
        let out_features = weight_size_flattened / input_size_flattened;

        if *weight_shape.last().unwrap() == out_features {
            // Layout is likely [...in_features, out_features].
            let in_features = weight_size_flattened / out_features;
            weight.reshape(Shape::new(vec![in_features, out_features]))?;
            // Transpose to get [out_features, in_features] for subsequent logic.
            weight = weight.try_map_tensor(|t| t.transpose())?;
        } else if weight_shape[0] == out_features {
            // Layout is likely [out_features, ...in_features].
            let in_features = weight_shape[1..].iter().product::<usize>();
            ensure_onnx!(
                in_features == input_size_flattened,
                "Incompatible shapes for Gemm: expected flattened input of size {}, got {}",
                in_features,
                input_size_flattened
            );
            weight.reshape(Shape::new(vec![out_features, in_features]))?;
        } else {
            return err(format!(
                "Could not determine layout of weights for Gemm. Shape: {weight_shape:?}, expecting output dim of size {out_features}"
            ));
        }
    }
    ensure_onnx!(
        weight.shape().is_matrix(),
        "Weight for Gemm must be a matrix 2: {:?}",
        weight.shape()
    );

    if input_shape.len() != 1 {
        ensure!(
            input_shape[0] == 1,
            "First dimension of Gemm layer input should be 1. Input shape was: {input_shape:?}"
        );
        input_shape.remove(0);
    }
    ensure_onnx!(
        input_shape.len() == 1,
        "Input shape for Gemm must be a vector, found {:?}",
        input_shape
    );

    let mut weight_shape = weight.shape();
    if weight_shape[1] != input_shape[0] {
        weight = weight.try_map_tensor(|t| t.transpose())?;
        weight_shape = weight.shape();
    }
    ensure_onnx!(
        weight_shape[1] == input_shape[0],
        "Incompatible shapes found for Gemm node: input shape is {:?}, weight shape is {:?}",
        input_shape,
        weight_shape,
    );
    let mut weight_shape = weight.shape();
    // If the weights are a 1D vector we insert a 1 in the shape after checking everything lines up
    if weight_shape.len() == 1 {
        ensure_onnx!(
            weight_shape[0] == input_shape[0],
            "Incompatible shapes found for Gemm node: input shape is {:?}, weight shape is {:?}",
            input_shape,
            weight_shape,
        );
        weight.shape_mut().insert(0, 1);
    } else {
        if weight_shape[1] != input_shape[0] {
            weight = weight.try_map_tensor(|t| t.transpose())?;
            weight_shape = weight.shape();
        }
        ensure_onnx!(
            *weight_shape.last().unwrap() == input_shape[0],
            "Incompatible shapes found for Gemm node: input shape is {:?}, weight shape is {:?}",
            input_shape,
            weight_shape,
        );
    }

    // also extract the bias if any. If there is one, that means the next node
    // is a Add node and we must make sure one of the inputs is the current
    // matrix node. Otherwise that's just a normal add that we don't support.
    // TODO: support general case with Add layer.
    let (edge_id, bias_node_id) = match iter.peek() {
        // no next node, no bias
        None => (node_id, None),
        Some(&&next_node_id) => {
            let next_node = onnx.node(next_node_id);
            // if there's a bias, the next op is a TypedBinOp( Add ) node
            match downcast_to::<TypedBinOp>(next_node) {
                // safety net
                _ if next_node.inputs.len() != 2 => {
                    // no bias, just return the matrix node
                    (node_id, None)
                }
                // the operation must be an Add
                Ok(binop) if binop.0.is::<tract_core::ops::math::Add>() => {
                    // now on this node, we need to ensure one of the inputs is the current matrix node
                    match next_node
                        .inputs
                        .iter()
                        .enumerate()
                        .find(|(_i, x)| x.node == node_id)
                    {
                        Some((idx, ..)) => {
                            // Now we need to find the bias node, which is the other input to the Add node
                            // and we can extract it as a constant tensor afterwards
                            // since only two elements, we can just do 1 - idx
                            let bias_input = next_node.inputs[1 - idx];
                            // let bias_node = model.node(bias_input.node);
                            // in that case, we move on the iterator, since we already saw the bias node and the Add is part of the dense layer
                            // unwrap is safe here since we peeked already
                            iter.next().unwrap();
                            (next_node_id, Some(bias_input.node))
                        }
                        None => {
                            // no bias, just return the matrix node
                            (node_id, None)
                        }
                    }
                }
                _ => {
                    // no bias, just return the matrix node
                    (node_id, None)
                }
            }
        }
    };
    let bias_tensor = bias_node_id
        .map(|bias| {
            let bias_node = onnx.node(bias);
            let mut bias_tensor = extract_const_tensor(bias_node)?;
            let bias_shape = bias_tensor.shape().clone();
            ensure_onnx!(
                bias_shape.rank() == 1 || bias_shape.rank() == 2,
                "Bias tensor must be 1D or 2D with batch: {:?}",
                bias_shape
            );
            if bias_shape.rank() == 2 {
                ensure_onnx!(
                    bias_shape[0] == 1,
                    "Bias tensor must be 1D with batch: {:?}",
                    bias_shape
                );
                bias_tensor.reshape(bias_shape.slice(1..))?;
            }
            ensure_onnx!(
                bias_tensor.shape()[0] == weight.shape()[0],
                "Bias tensor must have same size as filter's rows"
            );
            Ok(bias_tensor)
        })
        .transpose()?;

    let dense = crate::layers::einsum::EinSum::new_dense(
        weight.into(),
        bias_tensor.map(|tensor| tensor.into()),
    )?;

    // we put the bias id if present so next layers refer to it and not the gemm node
    let zkml_node_id = model.graph_mut().add_inner(Layer::EinSum(dense))?;
    model.add_edge(
        node_mapping[&input_link.node],
        zkml_node_id,
        (input_link.slot, 0),
    )?;

    // here since the bias addition is the _last_ operation, the next layers are
    // gonna refer to the id of the add node and not the gemm node.
    node_mapping.insert(edge_id, zkml_node_id);
    Ok(zkml_node_id)
}

fn load_conv<'a, I: Iterator<Item = &'a usize> + Sized>(
    node_mapping: &mut HashMap<usize, NodeId>,
    onnx: &OnnxModel,
    model: &mut Model<f32>,
    node_id: usize,
    node: &OnnxNode,
    _iter: &mut Peekable<I>,
) -> Result<NodeId> {
    let conv_node = downcast_to::<Conv>(node)?;
    // TODO: once we support different padding and strides, extract the data in this function
    check_conv2d_attributes(conv_node)?;
    // TODO: support for conv without bias
    ensure_onnx!(
        node.inputs.len() == 3,
        "ONNX Conv {} must have exactly 3 inputs: {}",
        node.name,
        node.inputs.len()
    );
    let input_link = node.inputs[0];
    let filter_link = node.inputs[1];
    let bias_link = node.inputs[2];
    let filter_node = onnx.node(filter_link.node);
    let bias_node = onnx.node(bias_link.node);
    let filter_const = extract_const_tensor(filter_node)?;
    let bias_const = extract_const_tensor(bias_node)?;
    let conv = if bias_const.shape().is_empty() {
        Convolution::new_without_bias(filter_const)?
    } else {
        Convolution::new(filter_const, bias_const)?
    };
    let zkml_node_id = model.graph_mut().add_inner(Layer::Convolution(conv))?;
    model.add_edge(
        node_mapping[&input_link.node],
        zkml_node_id,
        (input_link.slot, 0),
    )?;

    node_mapping.insert(node_id, zkml_node_id);
    Ok(zkml_node_id)
}

fn is_const(node: &OnnxNode) -> bool {
    downcast_to::<Const>(node).is_ok()
}

fn extract_const_tensor(node: &OnnxNode) -> Result<KeyedTensor<f32>> {
    let tensor = downcast_to::<Const>(node)?;
    let slice = tensor.0.as_slice::<f32>()?;
    ensure_onnx!(node.outputs.len() == 1, "constant output shape len == 1");
    let Some(shape) = node.outputs[0].fact.shape.as_concrete() else {
        return err(format!("Filter shape {} is not concrete", node.name));
    };
    Ok(KeyedTensor::new(
        format!("{}-{}", node.name, node.id),
        crate::Tensor::new(shape.to_vec().into(), slice.to_vec())?,
    ))
}

fn get_node_output_shape(node: &OnnxNode, output_idx: usize) -> Result<Shape> {
    ensure_onnx!(
        output_idx < node.outputs.len(),
        "Trying to get output {} of node {}, but there are only {} outputs",
        output_idx,
        node.name,
        node.outputs.len(),
    );
    let Some(shape) = node.outputs[output_idx].fact.shape.as_concrete() else {
        return err(format!("shape of node {} is not concrete", node.name));
    };
    Ok(shape.to_vec().into())
}

/// Get the conv2d attributes and assert if supported by DeepProve
fn check_conv2d_attributes(node: &Conv) -> Result<()> {
    let Some(ref strides) = node.pool_spec.strides else {
        return err(format!("Conv has no strides: {}", node.name()));
    };
    ensure_onnx!(strides.iter().all(|&x| x == 1), "Strides must be {}", 1);
    ensure_onnx!(strides.iter().all(|&x| x == 1), "Strides must be {}", 1);
    let PaddingSpec::Explicit(pad0, pad1) = &node.pool_spec.padding else {
        return err(format!("Conv has no pads: {}", node.name()));
    };
    ensure_onnx!(
        pad0.iter().all(|&x| x == 0),
        "Padding for {}must be 0s: {:?}",
        node.name(),
        pad0,
    );
    ensure_onnx!(
        pad1.iter().all(|&x| x == 0),
        "Padding for {}must be 0s: {:?}",
        node.name(),
        pad1,
    );
    let Some(ref dilations) = node.pool_spec.dilations else {
        return err(format!("Conv has no dilations: {}", node.name()));
    };
    ensure_onnx!(
        dilations.iter().all(|&x| x == 1),
        "Dilations for {} must be 1: {:?}",
        node.name(),
        dilations
    );
    let kernel_shape = &node.pool_spec.kernel_shape;
    ensure_onnx!(
        kernel_shape.iter().all(|&x| x > 1),
        "Kernel shape for {} must be > 1: {:?}",
        node.name(),
        kernel_shape
    );
    ensure_onnx!(
        kernel_shape.len() == 2,
        "Kernel shape for {} must be 2D: {:?}",
        node.name(),
        kernel_shape
    );
    ensure_onnx!(
        kernel_shape[0] == kernel_shape[1],
        "Kernel shape for {} must be square: {:?}",
        node.name(),
        kernel_shape
    );
    Ok(())
}

fn err<T>(msg: String) -> Result<T> {
    bail!("Onnx parsing: {msg}")
}

fn downcast_to<T: Op>(node: &OnnxNode) -> Result<&T> {
    match node.op_as::<T>() {
        Some(b) => Ok(b),
        None => err(format!(
            "Node {} is not a {}",
            node.name,
            std::any::type_name::<T>()
        )),
    }
}

fn tdim_to_usize(tdim: &TDim) -> anyhow::Result<usize> {
    match tdim {
        TDim::Val(v) => Ok(*v as usize),
        _ => bail!("Unsupported dimension: {tdim:?}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::Ok;
    use dp_crypto::arkyper::transcript::blake3::Blake3Transcript;
    use tenstore::GenStore;

    use crate::{
        Prover, init_test_logging_default,
        quantization::{InferenceObserver, Quantize},
        testing::Pcs,
        verify,
    };
    use tracing::info;

    type F = ark_bn254::Fr;
    type T = Blake3Transcript;

    type P<'a, 'b> = Prover<'a, 'b, F, T, Pcs>;

    #[test]
    fn test_load_mlp() {
        let filepath = "assets/scripts/MLP/mlp-iris-01.onnx";
        let result = FloatOnnxLoader::new(filepath).build();

        assert!(result.is_ok(), "Failed: {:?}", result.unwrap_err());
    }

    #[test]
    fn test_mlp_model_run() {
        init_test_logging_default();
        let filepath = "assets/scripts/MLP/mlp-iris-01.onnx";
        let (model, md) = FloatOnnxLoader::new(filepath).build().unwrap();
        let input = crate::tensor::Tensor::<f32>::random(&model.input_shapes()[0])
            .quantize(md.input_scaling(0));
        let inputs = model.prepare_inputs(vec![input]).unwrap();
        let trace = model.run(inputs, &mut GenStore::default()).unwrap();
        println!("Result: {:?}", trace.outputs());
    }

    #[test]
    #[ignore]
    fn test_covid_cnn() {
        let subscriber = tracing_subscriber::fmt::Subscriber::builder()
            .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
            .finish();
        tracing::subscriber::set_global_default(subscriber)
            .expect("Failed to set global subscriber");

        let filepath = "assets/scripts/covid/cnn-covid.onnx";
        let result = FloatOnnxLoader::new(filepath).build();

        assert!(result.is_ok(), "Failed: {:?}", result.unwrap_err());

        info!("CREAting random tensor input");
        let (model, md) = result.unwrap();
        let inputs = model
            .input_shapes()
            .into_iter()
            .enumerate()
            .map(|(i, shape)| {
                crate::tensor::Tensor::<f32>::random(&shape).quantize(md.input_scaling(i))
            })
            .collect();
        let inputs = model.prepare_inputs(inputs).unwrap();

        info!("RUNNING MODEL...");
        let trace = model.run(inputs, &mut GenStore::default()).unwrap();

        info!("RUNNING MODEL DONE...");
        println!("Result: {:?}", trace.outputs());

        info!("GENERATING CONTEXT...");
        let (prover_ctx, verifier_ctx) = model
            .generate_contexts::<F, Pcs>()
            .expect("Unable to generate contexts");

        info!("GENERATING CONTEXT DONE...");

        info!("GENERATING Proof...");
        let (proof, io) = P::prove(&prover_ctx, trace, &model).expect("unable to generate proof");
        info!("GENERATING Proof DONE...");
        verify::<_, T, _>(&verifier_ctx, proof, io).unwrap();
    }

    #[test]
    fn test_load_cnn() {
        init_test_logging_default();
        let filepath = "assets/scripts/CNN/cnn-cifar-01.onnx";
        let result =
            FloatOnnxLoader::new_with_scaling_strategy(filepath, InferenceObserver::new()).build();

        assert!(result.is_ok(), "Failed: {:?}", result.unwrap_err());

        let (model, md) = result.unwrap();
        // let model = pad_model(model).unwrap();
        model.describe();
        let native_input = model
            .input_shapes()
            .into_iter()
            .enumerate()
            .map(|(i, shape)| {
                crate::tensor::Tensor::<f32>::random(&shape).quantize(md.input_scaling(i))
            })
            .collect();
        let inputs = model.prepare_inputs(native_input).unwrap();
        let trace = model.run(inputs, &mut GenStore::default()).unwrap();

        let (prover_ctx, verifier_ctx) = model
            .generate_contexts::<F, Pcs>()
            .expect("Unable to generate contexts");

        let (proof, io) = P::prove(&prover_ctx, trace, &model).expect("unable to generate proof");
        verify::<_, T, _>(&verifier_ctx, proof, io).unwrap();
    }

    #[test]
    fn test_tract() {
        let filepath = "assets/scripts/CNN/cnn-cifar-01.onnx";
        let model = tract_onnx::onnx()
            .model_for_path(filepath)
            .map_err(|e| Error::msg(format!("Failed to load model: {e:?}")))
            .unwrap();
        for symbol in model.symbols.all_symbols().iter() {
            println!("symbol: {symbol:?}");
        }
        let opt = model.into_typed().unwrap();

        let eval_order = opt.eval_order().unwrap();
        eval_order.into_iter().for_each(|id| {
            let node = opt.node(id);
            let outputs = &node.outputs;
            for (i, output) in outputs.iter().enumerate() {
                println!(
                    "Cluttered Node: {},  Output {} shape: {:?}",
                    node,
                    i,
                    output.fact.shape.dims()
                );
            }
        });

        let opt = opt.into_decluttered().unwrap();

        let eval_order = opt.eval_order().unwrap();

        eval_order.into_iter().for_each(|id| {
            let node = opt.node(id);
            let outputs = &node.outputs;

            for (i, output) in outputs.iter().enumerate() {
                println!(
                    "Node {}: {},  Output {} shape: {:?}",
                    id,
                    node,
                    i,
                    output.fact.shape.dims()
                );
            }
        });

        let mut values = SymbolValues::default();
        let symbol = opt.sym("batch_size");
        values.set(&symbol, 1);

        let opt = opt.concretize_dims(&values).unwrap();
        let plan = SimplePlan::new(opt).unwrap();

        for node_id in plan.order_without_consts() {
            let node = plan.model().node(*node_id);
            println!(
                "planned node {}:{}: input {:?} -> op{:?}",
                node_id,
                node.name,
                node.inputs,
                node.op()
            );
        }
    }
    #[test]
    fn test_parser_load_conv() {
        let model = from_path("assets/scripts/CNN/cnn-cifar-01.onnx").unwrap();
        let input_shape = model.input_shapes()[0].clone();

        let input_tensor = crate::tensor::Tensor::random(&input_shape);
        let trace = model
            .run(vec![input_tensor], &mut GenStore::default())
            .unwrap();
        assert!(!trace.steps.is_empty());
    }

    #[test]
    #[ignore = "this test shows no gpt2 onnx out there are working with tract_onnx"]
    fn test_parser_onnx_gpt2() -> anyhow::Result<()> {
        // let path = "assets/scripts/llms/gpt2_simple.onnx";
        // let path = "gpt2_export/gpt2_simple.onnx";
        // let path = "assets/scripts/llms/gpt2_download1.onnx";
        // let path = "assets/scripts/llms/gpt2_onnxcommunity.onnx";
        let path = "assets/scripts/llms/gpt2_decoder.onnx";
        let model = {
            //.into_decluttered()?;
            tract_onnx::onnx().model_for_path(path)?.into_typed()?
            // so far we dont support batching
            // let mut values = SymbolValues::default();
            // let symbol = pmodel.sym("batch_size");
            // values.set(&symbol, 1);
            // pmodel.concretize_dims(&values)?
        };

        // let plan = SimplePlan::new(model)?;
        // let onnx_model = plan.model();
        // let inference_order = plan.order_without_consts();
        for node_id in model.eval_order()? {
            debug!("node {}: {:?}", node_id, model.node(node_id));
        }
        Ok(())
    }
}
