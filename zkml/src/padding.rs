use std::collections::HashMap;

use anyhow::{Context, Result, ensure};

use serde::{Deserialize, Serialize};
use tracing::debug;

use crate::{
    Element, NextPowerOfTwo, Shape,
    graph::{Node, NodeInput, NodeOutput, order_by_in_port},
    layers::{
        einsum::EinSum,
        flatten::Flatten,
        pooling::{Pooling, safe_maxpool2d_shape},
        provable::{OpInfo, PadOp},
        reshape::Reshape,
    },
    model::Model,
    tensor::TensorHandle,
};

#[derive(Clone, Debug)]
pub enum GarbagePad {
    Convolution((Shape, Shape)),
}

impl GarbagePad {
    fn pad_matrix_to_ignore_garbage(
        &self,
        matrix: &mut TensorHandle<Element>,
        padded_matrix_shape: Shape,
    ) -> Result<()> {
        match self {
            GarbagePad::Convolution(previous_shape) => {
                let wrapped = matrix.take_wrapped_tensor()?;
                let padded = wrapped.pad_matrix_to_ignore_garbage(
                    &previous_shape.0,
                    &previous_shape.1,
                    &padded_matrix_shape,
                )?;
                matrix.set_wrapped_tensor(padded)?;
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

    pub fn update_shapes<L: OpInfo>(&mut self, layer: &L) -> anyhow::Result<()> {
        let unpadded_output_shapes =
            layer.output_shapes(&self.unpadded_input_shapes(), PaddingMode::NoPadding)?;
        let padded_output_shapes =
            layer.output_shapes(&self.padded_input_shapes(), PaddingMode::Padding)?;
        self.shapes
            .resize(unpadded_output_shapes.len(), Default::default());
        for ((sd, unpadded_shape), shape) in self
            .shapes
            .iter_mut()
            .zip(unpadded_output_shapes)
            .zip(padded_output_shapes)
        {
            sd.input_shape_padded = shape;
            sd.input_shape_og = unpadded_shape;
        }
        Ok(())
    }
}

impl From<&[ShapeData]> for ShapeInfo {
    fn from(value: &[ShapeData]) -> Self {
        Self {
            shapes: value.to_vec(),
        }
    }
}

#[derive(Clone, Debug, Default)]
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
            .input_shapes()
            .into_iter()
            .map(|unpadded_shape| ShapeData {
                input_shape_padded: unpadded_shape.next_power_of_two(),
                ignore_garbage_pad: None,
                input_shape_og: unpadded_shape,
            })
            .collect(),
    };
    let unpadded_input_shapes = model.input_shapes();
    debug!(
        "Padding model with {} inputs: shapes {:?}",
        unpadded_input_shapes.len(),
        unpadded_input_shapes
    );
    // compute all shape infos to be able to pad the model afterwards.
    let mut shape_infos: HashMap<NodeOutput, ShapeData> = Default::default();

    let padded_graph =
        model
            .into_graph()
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

    model = Model::<Element>::new(unpadded_input_shapes, padded_graph);
    debug!("Padded model with {} layers", model.graph().node_count());
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
            requantise,
            name,
            ..
        } = einsum;

        let padded_constant_tensors = constant_tensors;

        let padded_biases = biases;

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

        Ok(EinSum {
            equation,
            name,
            mapping,
            evaluation_info,
            constant_tensors: padded_constant_tensors,
            constant_unpadded_shapes,
            biases: padded_biases,
            bias_unpadded_shapes,
            caches,
            requantise,
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

        let nrows = matrix.shape().nrows_2d();
        if let Some(ref bias) = einsum.biases[0] {
            let wrapped = bias.wrapped_tensor()?;
            ensure!(
                wrapped.shape().num_elements() == nrows,
                "Bias shape {} does not match matrix width {}",
                wrapped.shape(),
                nrows,
            );
        }
        // ncols must match the exact flatten output size (from shape tracking)
        let ncols = sd.input_shape_padded.product();
        ensure!(
            matrix.shape().ncols_2d() <= ncols,
            "EinSum layer matrix has more columns ({}) than previous layer output size ({}). \
             Cannot shrink without losing information.",
            matrix.shape().ncols_2d(),
            ncols,
        );
        let nrows = matrix.shape().nrows_2d();

        if let Some(garbage_pad) = sd.ignore_garbage_pad.as_ref() {
            garbage_pad
                .pad_matrix_to_ignore_garbage(&mut matrix, Shape::new(vec![nrows, ncols]))?;
            sd.ignore_garbage_pad = None;
        } else {
            let wrapped = matrix.take_wrapped_tensor()?;
            let reshaped = wrapped.reshape(Shape::new(vec![nrows, ncols]).into())?;
            matrix.set_wrapped_tensor(reshaped)?;
        }

        // Update shape tracking: output shape is [nrows]
        sd.input_shape_og = vec![nrows].into();
        sd.input_shape_padded = vec![nrows.next_power_of_two()].into();

        let bias = einsum.biases[0].clone().map(|mut handle| {
            let wrapped = handle.take_wrapped_tensor().unwrap();
            let shape = Shape::new(vec![nrows]);
            let reshaped = wrapped.pad(shape.into(), 0).unwrap();
            handle.set_wrapped_tensor(reshaped).unwrap();
            handle
        });

        let EinSum::<Element> {
            equation,
            name,
            mapping,
            evaluation_info,
            constant_tensors: _,
            constant_unpadded_shapes: _,
            biases: _,
            bias_unpadded_shapes,
            caches,
            requantise,
            ..
        } = einsum;

        let constant_unpadded_shapes = vec![Some(matrix.unpadded_shape().clone())];
        Ok(EinSum {
            equation,
            name,
            mapping,
            evaluation_info,
            constant_tensors: vec![Some(matrix)],
            constant_unpadded_shapes,
            biases: vec![bias],
            bias_unpadded_shapes,
            caches,
            requantise,
        })
    }
}
