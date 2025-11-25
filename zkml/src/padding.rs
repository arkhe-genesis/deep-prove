use std::collections::HashMap;

use anyhow::{Context, Result, ensure};

use serde::{Deserialize, Serialize};
use tracing::debug;

use crate::{
    Element, Shape, Tensor,
    graph::{Node, NodeInput, NodeOutput, order_by_in_port},
    layers::{
        einsum::EinSum,
        flatten::Flatten,
        pooling::{Pooling, safe_maxpool2d_shape},
        provable::{OpInfo, PadOp},
        reshape::Reshape,
    },
    model::Model,
};

#[derive(Clone, Debug)]
pub enum GarbagePad {
    Convolution((Shape, Shape)),
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
                        .with_context(|| format!("padding layer {:?}: {}", node_id, desc))?;

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
    Ok(Flatten(true))
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

pub(crate) fn pad_einsum(einsum: EinSum<Element>, si: &mut ShapeInfo) -> Result<EinSum<Element>> {
    let contains_garbage_pad = si.shapes.iter().any(|sd| sd.ignore_garbage_pad.is_some());

    let one_input = si.shapes.len() == 1;

    let garbage_pad_case = contains_garbage_pad && one_input;
    if !garbage_pad_case {
        // Update the shape data
        let unpadded_input_shapes = si.unpadded_input_shapes();
        let padded_input_shapes = si.padded_input_shapes();

        let unpadded_output_shapes =
            einsum.output_shapes(&unpadded_input_shapes, PaddingMode::NoPadding)?;
        let padded_output_shapes =
            einsum.output_shapes(&padded_input_shapes, PaddingMode::Padding)?;

        // We must pad any constant tensors and bias tensors to ensure they are compatible with the padded inputs.
        // However, we do not need to change the equation or mapping, as the padding is handled by the input shapes.
        let EinSum::<Element> {
            equation,
            mapping,
            evaluation_info,
            constant_tensors,
            constant_unpadded_shapes,
            biases,
            bias_unpadded_shapes,
            caches,
            ..
        } = einsum;

        let padded_constant_tensors = constant_tensors
            .into_iter()
            .map(|opt| opt.map(|tensor| tensor.map_tensor(|t| t.pad_next_power_of_two())))
            .collect::<Vec<_>>();

        let padded_biases = biases
            .into_iter()
            .map(|opt| opt.map(|tensor| tensor.map_tensor(|t| t.pad_next_power_of_two())))
            .collect::<Vec<_>>();

        // Currently we do not support garbage padding for einsum outputs, this is because we are in the process
        // of removing garbage padding from the library, so we do not want to add it here.
        si.shapes = unpadded_output_shapes
            .into_iter()
            .zip(padded_output_shapes)
            .map(|(input_shape_og, input_shape_padded)| ShapeData {
                input_shape_padded,
                ignore_garbage_pad: None,
                input_shape_og,
            })
            .collect();

        let padded_caches = caches
            .into_iter()
            .map(|cache| {
                cache.inspect(|c| {
                    let mut c_lock = c.lock().unwrap();
                    c_lock.set_padding_mode(PaddingMode::Padding);
                })
            })
            .collect();

        Ok(EinSum {
            equation,
            mapping,
            evaluation_info,
            constant_tensors: padded_constant_tensors,
            constant_unpadded_shapes,
            biases: padded_biases,
            bias_unpadded_shapes,
            padded: true,
            caches: padded_caches,
        })
    } else {
        // This is the case when the previous layer was a flatten
        let sd = si.shapes.first_mut().unwrap();
        ensure!(
            einsum.constant_tensors.len() == 1,
            "Expected exactly one constant tensor in einsum when padding with garbage pad, found {}",
            einsum.constant_tensors.len()
        );
        let mut matrix = einsum.constant_tensors[0].clone().unwrap();

        let matrix_shape = matrix.shape().clone();
        let nrows = matrix_shape.nrows();
        sd.input_shape_og = vec![nrows].into();
        if let Some(ref bias) = einsum.biases[0] {
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
        let mut new_cols = matrix.ncols_2d()?;
        if matrix.ncols_2d()? != sd.input_shape_padded.dim(0) {
            if matrix.ncols_2d()? < sd.input_shape_padded.dim(0) {
                new_cols = sd.input_shape_padded.dim(0);
            } else {
                // If we have too many columns, we can't shrink without losing information
                anyhow::bail!(
                    "EinSum layer matrix has more columns ({}) than previous layer output size ({}).
                            Cannot shrink without losing information.",
                    matrix.ncols_2d()?,
                    sd.input_shape_padded.dim(0)
                );
            }
        }
        // The reason to pad to a minimum of 4 is that any subsequent activation function will
        // be needing at least input shape of total size 4 due to usage of lookups.
        // current logup gkr implementation requires at least 2 variables for poly.
        let ncols = pad_minimum(new_cols);
        let nrows = pad_minimum(matrix.nrows_2d()?);

        if let Some(garbage_pad) = sd.ignore_garbage_pad.as_ref() {
            garbage_pad.pad_matrix_to_ignore_garbage(&mut matrix, vec![nrows, ncols].into())?;
            sd.ignore_garbage_pad = None;
        } else {
            matrix.reshape_to_fit_inplace_2d(vec![nrows, ncols].into())?;
        }

        let bias = if let Some(bias) = einsum.biases[0].clone() {
            Some(bias.try_map_tensor(|t| t.pad_1d(nrows))?)
        } else {
            None
        };

        let EinSum::<Element> {
            equation,
            mapping,
            evaluation_info,
            constant_tensors: _,
            constant_unpadded_shapes: _,
            biases: _,
            bias_unpadded_shapes,
            caches,
            ..
        } = einsum;

        let constant_unpadded_shapes = vec![Some(matrix.unpadded_shape().clone())];
        Ok(EinSum {
            equation,
            mapping,
            evaluation_info,
            constant_tensors: vec![Some(matrix)],
            constant_unpadded_shapes,
            biases: vec![bias],
            bias_unpadded_shapes,
            padded: true,
            caches,
        })
    }
}

fn pad_minimum(dim: usize) -> usize {
    let r = dim.next_power_of_two();
    if r < 4 { 4 } else { r }
}
