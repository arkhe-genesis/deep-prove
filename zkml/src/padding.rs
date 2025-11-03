use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
};

use anyhow::{Context, Result, bail, ensure};
use itertools::Itertools;
use serde::{Deserialize, Serialize};
use tracing::debug;

use crate::{
    Element, Shape, Tensor,
    graph::{Node, NodeInput, NodeOutput, order_by_in_port},
    layers::{
        concat_matmul::ConcatMatMul,
        dense::Dense,
        flatten::Flatten,
        matrix_mul::{MatMul, OperandMatrix},
        pooling::{Pooling, safe_maxpool2d_shape},
        provable::{OpInfo, PadOp},
        reshape::Reshape,
        transformer::{
            mha::pad_matrix_to_ignore_mha_garbage,
            qkv::{CacheQKV, QKV},
        },
    },
    model::Model,
};

#[derive(Clone, Debug)]
pub enum GarbagePad {
    Convolution((Shape, Shape)),
    MHA((Shape, Shape)),
}

impl GarbagePad {
    fn pad_matrix_to_ignore_garbage(
        &self,
        matrix: &mut Tensor<Element>,
        padded_matrix_shape: Shape,
    ) -> Result<()> {
        match self {
            GarbagePad::Convolution(previous_shape) => {
                let previous_input_shape_og = previous_shape.0.clone();
                let previous_input_shape_padded = previous_shape.1.clone();
                *matrix = matrix.pad_matrix_to_ignore_garbage(
                    previous_input_shape_og.as_ref(),
                    previous_input_shape_padded.as_ref(),
                    &padded_matrix_shape,
                )?;
            }
            GarbagePad::MHA(previous_shape) => {
                *matrix = pad_matrix_to_ignore_mha_garbage(
                    matrix,
                    &previous_shape.0,
                    &previous_shape.1,
                    padded_matrix_shape,
                )?;
            }
        }

        Ok(())
    }
}

#[derive(Clone, Debug, Copy, Serialize, Deserialize)]
pub enum PaddingMode {
    NoPadding,
    Padding,
}

#[derive(Clone, Debug)]
pub struct ShapeInfo {
    pub(crate) shapes: Vec<ShapeData>,
}

impl ShapeInfo {
    pub fn unpadded_input_shapes(&self) -> Vec<Shape> {
        self.shapes
            .iter()
            .map(|sd| sd.input_shape_og.clone())
            .collect()
    }

    pub fn padded_input_shapes(&self) -> Vec<Shape> {
        self.shapes
            .iter()
            .map(|sd| sd.input_shape_padded.clone())
            .collect()
    }
}

impl From<&[ShapeData]> for ShapeInfo {
    fn from(value: &[ShapeData]) -> Self {
        Self {
            shapes: value.to_vec(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct ShapeData {
    pub(crate) input_shape_padded: Shape,
    pub(crate) ignore_garbage_pad: Option<GarbagePad>,
    pub(crate) input_shape_og: Shape,
}

impl ShapeData {
    /// Build new shape data for an input tensor of a layer, given the unpadded input shape
    pub fn new(unpadded_input_shape: Shape) -> Self {
        Self {
            input_shape_padded: unpadded_input_shape.next_power_of_two(),
            ignore_garbage_pad: None,
            input_shape_og: unpadded_input_shape,
        }
    }

    pub fn new_with_garbage_pad(unpadded_input_shape: Shape, garbage_pad: GarbagePad) -> Self {
        Self {
            input_shape_padded: unpadded_input_shape.next_power_of_two(),
            ignore_garbage_pad: Some(garbage_pad),
            input_shape_og: unpadded_input_shape,
        }
    }

    pub(crate) fn with_garbage_pad(self, garbage_pad: GarbagePad) -> Self {
        Self {
            input_shape_padded: self.input_shape_padded,
            ignore_garbage_pad: Some(garbage_pad),
            input_shape_og: self.input_shape_og,
        }
    }
}

pub fn pad_model(mut model: Model<Element>) -> Result<Model<Element>> {
    let input_si = ShapeInfo {
        shapes: model
            .unpadded_input_shapes()
            .into_iter()
            .zip(model.padded_input_shapes())
            .map(|(unpadded_shape, padded_shape)| ShapeData {
                input_shape_padded: padded_shape,
                ignore_garbage_pad: None,
                input_shape_og: unpadded_shape,
            })
            .collect(),
    };
    let unpadded_input_shapes = model.unpadded_input_shapes();
    debug!(
        "Padding model with {} inputs: shapes {:?}",
        unpadded_input_shapes.len(),
        unpadded_input_shapes
    );
    // compute all shape infos to be able to pad the model afterwards.
    let mut shape_infos: HashMap<NodeOutput, ShapeData> = Default::default();

    let padded_graph = model
        .graph
        .try_into_map_forward(|node_id, node, incoming_feeds| {
            Ok(match node {
                Node::Inner(layer) => {
                    let mut si = ShapeInfo {
                        shapes: order_by_in_port(incoming_feeds.into_iter().map(|feed| {
                            let in_shape = shape_infos[&feed.source].clone();
                            (NodeInput::new(node_id, feed.target.port), in_shape)
                        }))
                        .collect(),
                    };

                    let desc = layer.describe();
                    let padded_layer = layer
                        .pad_node(&mut si)
                        .context(format!("padding layer {:?}: {}", node_id, desc))?;

                    shape_infos.extend(
                        si.shapes
                            .into_iter()
                            .enumerate()
                            .map(|(i, shape_data)| (NodeOutput::new(node_id, i), shape_data)),
                    );

                    Node::Inner(padded_layer)
                }
                Node::Input(i) => {
                    shape_infos.insert(node_id.as_model_input(), input_si.shapes[i].clone());
                    Node::Input(i)
                }
                Node::Output(o) => Node::Output(o),
            })
        })?;

    model = Model::<Element>::new(unpadded_input_shapes, PaddingMode::Padding, padded_graph);
    debug!("Padded model with {} layers", model.graph.node_count());
    Ok(model)
}

pub(crate) fn reshape(si: &mut ShapeInfo) -> Result<Flatten> {
    si.shapes.iter_mut().for_each(|sd| {
        sd.ignore_garbage_pad = Some(GarbagePad::Convolution((
            sd.input_shape_og.clone(),
            sd.input_shape_padded.clone(),
        )))
    });
    Ok(Flatten)
}

pub(crate) fn pooling(p: Pooling, si: &mut ShapeInfo) -> Result<Pooling> {
    for sd in si.shapes.iter_mut() {
        // Make sure that input shape is already padded and is well formed
        ensure!(
            sd.input_shape_padded.is_power_of_two(),
            "Input shape for max pool is not padded"
        );
        sd.input_shape_og = safe_maxpool2d_shape(&sd.input_shape_og)?;
        sd.input_shape_padded = safe_maxpool2d_shape(&sd.input_shape_padded)?;
    }
    Ok(p)
}

pub(crate) fn pad_dense(mut d: Dense<Element>, si: &mut ShapeInfo) -> Result<Dense<Element>> {
    // dense layer currently expects 1 input, so we check there is only 1 input shape
    ensure!(
        si.shapes.len() == 1,
        "More than 1 input shape found when padding dense layer"
    );
    let sd = si.shapes.first_mut().unwrap();
    let matrix_shape = d.matrix.shape().clone();
    let nrows = matrix_shape.nrows();
    sd.input_shape_og = vec![nrows].into();
    if let Some(ref bias) = d.bias {
        ensure!(
            bias.get_data().len() == nrows,
            "Bias length {} does not match matrix width {}",
            bias.get_data().len(),
            nrows,
        );
    }
    ensure!(
        sd.input_shape_padded.is_power_of_two(),
        "Input shape for dense is not padded"
    );
    if sd.input_shape_padded.rank() != 1 {
        sd.input_shape_padded = vec![sd.input_shape_padded.product()].into();
        sd.input_shape_og = vec![sd.input_shape_og.product()].into();
    }
    let mut new_cols = d.matrix.ncols_2d()?;
    if d.matrix.ncols_2d()? != sd.input_shape_padded.dim(0) {
        if d.matrix.ncols_2d()? < sd.input_shape_padded.dim(0) {
            new_cols = sd.input_shape_padded.dim(0);
        } else {
            // If we have too many columns, we can't shrink without losing information
            bail!(
                "Dense layer matrix has more columns ({}) than previous layer output size ({}).
                            Cannot shrink without losing information.",
                d.matrix.ncols_2d()?,
                sd.input_shape_padded.dim(0)
            );
        }
    }
    // The reason to pad to a minimum of 4 is that any subsequent activation function will
    // be needing at least input shape of total size 4 due to usage of lookups.
    // current logup gkr implementation requires at least 2 variables for poly.
    let ncols = pad_minimum(new_cols);
    let nrows = pad_minimum(d.matrix.nrows_2d()?);

    if let Some(garbage_pad) = sd.ignore_garbage_pad.as_ref() {
        garbage_pad.pad_matrix_to_ignore_garbage(&mut d.matrix, vec![nrows, ncols].into())?;
        sd.ignore_garbage_pad = None;
    } else {
        d.matrix
            .reshape_to_fit_inplace_2d(vec![nrows, ncols].into())?;
    }
    d.bias = d
        .bias
        .map(|b| b.try_map_tensor(|t| t.pad_1d(nrows)))
        .transpose()?;
    sd.input_shape_padded = vec![nrows].into();
    Ok(d)
}

pub(crate) fn pad_matmul(mut mat: MatMul<Element>, si: &mut ShapeInfo) -> Result<MatMul<Element>> {
    let expected_num_inputs = mat.num_inputs();
    ensure!(
        si.shapes.len() == expected_num_inputs,
        "Expected {expected_num_inputs} input shapes when padding MatMul, found {}",
        si.shapes.len(),
    );

    ensure!(
        si.shapes
            .iter()
            .all(|s| s.input_shape_og.rank() == 2 && s.input_shape_padded.rank() == 2),
        "Unpadded input shape for MatMul is not 2D"
    );
    let (unpadded_input_shapes, mut padded_input_shapes): (Vec<Shape>, Vec<Shape>) = si
        .shapes
        .iter()
        .map(|s| (s.input_shape_og.clone(), s.input_shape_padded.clone()))
        .collect();
    let mut unpadded_output_shapes =
        mat.output_shapes(&unpadded_input_shapes, PaddingMode::NoPadding)?;
    ensure!(
        unpadded_output_shapes.len() == 1,
        "Expected 1 unpadded output shape for MatMul, found {}",
        unpadded_output_shapes.len(),
    );
    let unpadded_output_shape = unpadded_output_shapes.pop().unwrap();
    let (left_shape, mut right_shape) = match (&mut mat.left_matrix, &mut mat.right_matrix) {
        (OperandMatrix::Weight(m), OperandMatrix::Input) => {
            let nrows = pad_minimum(m.tensor.nrows_2d()?);
            let ncols = pad_minimum(m.tensor.ncols_2d()?);
            m.tensor
                .reshape_to_fit_inplace_2d(vec![nrows, ncols].into())?;
            (
                m.tensor.shape().clone(),
                padded_input_shapes.pop().unwrap(), /* safe to unwrap since we checked the number of inputs at the beginning */
            )
        }
        (OperandMatrix::Input, OperandMatrix::Weight(m)) => {
            let nrows = pad_minimum(m.tensor.nrows_2d()?);
            let ncols = pad_minimum(m.tensor.ncols_2d()?);
            let padded_matrix_shape = vec![nrows, ncols].into();
            // check if there is garbage pad: this is the only case we support in matrix mul where there
            // could be garbage pad
            if let Some(garbage_pad) = &si.shapes[0].ignore_garbage_pad {
                garbage_pad.pad_matrix_to_ignore_garbage(&mut m.tensor, padded_matrix_shape)?;
                si.shapes[0].ignore_garbage_pad = None;
            } else {
                m.tensor.reshape_to_fit_inplace_2d(padded_matrix_shape)?
            };
            (padded_input_shapes.pop().unwrap(), m.tensor.shape().clone())
        }
        (OperandMatrix::Input, OperandMatrix::Input) => {
            let right_shape = padded_input_shapes.pop().unwrap();
            let left_shape = padded_input_shapes.pop().unwrap();
            (left_shape, right_shape)
        }
        (OperandMatrix::Weight(_), OperandMatrix::Weight(_)) => {
            unreachable!("Found MatMul layer with 2 weight matrices")
        }
    };
    if mat.is_right_transposed() {
        right_shape.reverse();
    }
    ensure!(
        left_shape[1] == right_shape[0],
        "While padding MatMul layer. number of columns in left matrix ({}) does not match with number of rows in right matrix ({})",
        left_shape[1],
        right_shape[0],
    );
    ensure!(
        si.shapes.iter().all(|sd| sd.ignore_garbage_pad.is_none()),
        "MatMul layer has garbage padding to be removed",
    );
    si.shapes = vec![ShapeData {
        input_shape_og: unpadded_output_shape,
        input_shape_padded: vec![left_shape[0], right_shape[1]].into(),
        ignore_garbage_pad: None,
    }];
    if let Some(bias) = &mut mat.bias {
        bias.pad_to_shape(right_shape.slice(1..))?;
    }
    Ok(mat)
}

pub(crate) fn pad_qkv(mut qkv: QKV<Element>, si: &mut ShapeInfo) -> Result<QKV<Element>> {
    // reset QKV cache, as it might contain data from a previous inference
    // NOTE: we don't really reset we create a new instance, otherwise the same instance would be shared
    // between the padded and non padded qkv layer
    qkv.cache = Arc::new(Mutex::new(CacheQKV::new()));
    // qkv layer currently expects 1 input, so we check there is only 1 input shape
    ensure!(
        si.shapes.len() == 1,
        "More than 1 input shape found when padding qkv layer"
    );
    let sd = si.shapes.first_mut().unwrap();

    ensure!(
        sd.input_shape_og.rank() == 2,
        "Unpadded input shape for QKV is not 2D"
    );
    ensure!(
        sd.input_shape_padded.rank() == 2,
        "Padded input shape for QKV is not 2D"
    );

    let unpadded_output_shapes = qkv.output_shapes(
        std::slice::from_ref(&sd.input_shape_og),
        PaddingMode::NoPadding,
    )?;
    let expected_num_outputs = qkv.num_outputs(1).unwrap();
    ensure!(
        unpadded_output_shapes.len() == expected_num_outputs,
        "Expected {expected_num_outputs} unpadded output shapes for QKV layer, found {}",
        unpadded_output_shapes.len(),
    );

    ensure!(
        sd.input_shape_padded
            .as_ref()
            .iter()
            .all(|d| d.is_power_of_two()),
        "Padded input shapes for QKV layer are not a power of 2"
    );

    // Pad weight matrices
    let head_dim = qkv.head_dim;
    let padded_head_dim = pad_minimum(head_dim);
    let padded_num_heads = pad_minimum(qkv.num_heads);
    [&mut qkv.q, &mut qkv.k, &mut qkv.v].into_iter().try_for_each(|weight_mat| {
        let weight_tensor = weight_mat;
        let nrows = weight_tensor.nrows_2d()?;
        ensure!(nrows <= sd.input_shape_padded.dim(1),
            "Weight matrices in QKV layer has more rows than the number of columns of padded input shapes: Expected at most {} rows, found {}",
            sd.input_shape_padded.dim(1), nrows,
        );

        weight_tensor.reshape(Shape::new(vec![
            nrows,
            qkv.num_heads,
            head_dim,
        ]))?;
        let nrows = pad_minimum(sd.input_shape_padded.dim(1));
        weight_tensor.pad_to_shape(
            vec![nrows, padded_num_heads, padded_head_dim].into()
        )?;
        weight_tensor.reshape(Shape::new(vec![
            nrows,
            padded_num_heads*padded_head_dim,
        ]))?;
        Ok(())
    })?;

    // Pad bias vectors
    [&mut qkv.q_bias, &mut qkv.k_bias, &mut qkv.v_bias]
        .into_iter()
        .try_for_each(|bias_vec| -> Result<()> {
            if let Some(bias) = bias_vec.as_mut() {
                bias.reshape(Shape::new(vec![qkv.num_heads, head_dim]))?;
                bias.pad_to_shape(vec![padded_num_heads, padded_head_dim].into())?;
                bias.reshape(Shape::new(vec![padded_num_heads * padded_head_dim]))?;
            }
            Ok(())
        })?;

    let padded_output_shapes = qkv.output_shapes(
        std::slice::from_ref(&sd.input_shape_padded),
        PaddingMode::Padding,
    )?;
    ensure!(
        unpadded_output_shapes.len() == padded_output_shapes.len(),
        "Number of unpadded output shapes different from number of padded output shapes for QKV layer"
    );

    ensure!(
        sd.ignore_garbage_pad.is_none(),
        "QKV layer has garbage padding to be removed",
    );

    si.shapes = unpadded_output_shapes
        .into_iter()
        .zip(padded_output_shapes)
        .map(|(unpadded_shape, padded_shape)| ShapeData {
            input_shape_padded: padded_shape,
            ignore_garbage_pad: None,
            input_shape_og: unpadded_shape,
        })
        .collect();

    qkv.cache.lock().unwrap().padding_mode = PaddingMode::Padding;
    ensure!(qkv.cache.lock().unwrap().full_seq_len() == 0);

    Ok(qkv)
}

pub(crate) fn pad_concat_mat_mul(mat: ConcatMatMul, si: &mut ShapeInfo) -> Result<ConcatMatMul> {
    // no padding is needed since we don't have constant matrices in this layer
    // So, we check input shapes are padded, and we update shape info
    ensure!(
        si.shapes.len() == 2,
        "Expected 2 input shapes when padding ConcatMatMul layer, found {}",
        si.shapes.len(),
    );
    let unpadded_input_shapes = si.unpadded_input_shapes();

    mat.ensure_shape_consistency(&unpadded_input_shapes)?;

    let unpadded_output_shapes =
        mat.output_shapes(&unpadded_input_shapes, PaddingMode::NoPadding)?;
    let expected_num_outputs = mat.num_outputs(2)?;
    ensure!(
        unpadded_output_shapes.len() == expected_num_outputs,
        "Expected {expected_num_outputs} unpadded output shapes when padding ConcatMatMul, found {}",
        unpadded_output_shapes.len(),
    );

    let padded_input_shapes = si.padded_input_shapes();

    mat.ensure_shape_consistency(&padded_input_shapes)?;

    padded_input_shapes.iter().try_for_each(|s| {
        ensure!(
            s.is_power_of_two(),
            "Padded input shape for ConcatMatMul is not properly padded"
        );
        Ok(())
    })?;

    let padded_output_shapes = mat.output_shapes(&padded_input_shapes, PaddingMode::Padding)?;

    ensure!(
        padded_output_shapes.len() == expected_num_outputs,
        "Expected {expected_num_outputs} padded output shapes when padding ConcatMatMul, found {}",
        unpadded_output_shapes.len(),
    );

    ensure!(
        si.shapes.iter().all(|sd| sd.ignore_garbage_pad.is_none()),
        "ConcatMatMul layer has garbage padding to be removed",
    );

    si.shapes = unpadded_output_shapes
        .into_iter()
        .zip(padded_output_shapes)
        .map(|(unpadded, padded)| ShapeData {
            input_shape_padded: padded,
            ignore_garbage_pad: None,
            input_shape_og: unpadded,
        })
        .collect_vec();

    Ok(mat)
}

pub(crate) fn pad_reshape_layer(reshape: Reshape, si: &mut ShapeInfo) -> Result<Reshape> {
    let unpadded_output_shapes =
        reshape.output_shapes(&si.unpadded_input_shapes(), PaddingMode::NoPadding)?;

    let padded_output_shapes =
        reshape.output_shapes(&si.padded_input_shapes(), PaddingMode::Padding)?;

    ensure!(
        unpadded_output_shapes.len() == padded_output_shapes.len(),
        "Different number of unpadded output shapes and padded output shapes: {} vs {}",
        unpadded_output_shapes.len(),
        padded_output_shapes.len(),
    );

    // pad reshape depending on the type of reshape operation
    let reshape = reshape.to_padded_reshape();

    si.shapes
        .iter_mut()
        .zip(unpadded_output_shapes)
        .zip(padded_output_shapes)
        .for_each(|((sd, unpadded_shape), padded_shape)| {
            sd.input_shape_og = unpadded_shape;
            sd.input_shape_padded = padded_shape;
        });

    Ok(reshape)
}

fn pad_minimum(dim: usize) -> usize {
    let r = dim.next_power_of_two();
    if r < 4 { 4 } else { r }
}
