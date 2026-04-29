use super::provable::{Evaluate, LayerOut, OpInfo, PadOp, ProveInfo};
use crate::{
    Element, NextPowerOfTwo, Shape,
    iop::context::ContextAux,
    layers::LayerCtx,
    padding::{PaddingMode, ShapeInfo, reshape},
    tensor::WrappedTensor,
};
use anyhow::{Result, ensure};
use ark_ff::PrimeField;
use serde::{Deserialize, Serialize};

/// Short name used to identify the flatten layer
pub const FLATTEN_LAYER: &str = "FLTT";

#[derive(Clone, Debug, Serialize, Deserialize, Default)]
/// The inner bool indicates whether the flattening is padded or not
pub struct Flatten(pub(crate) bool);

/// Even if empty, we need a context such that it implements the default
/// methods of `VerifiableCtx``
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FlattenCtx;

impl OpInfo for Flatten {
    fn output_shapes(
        &self,
        input_shapes: &[Shape],
        padding_mode: PaddingMode,
    ) -> Result<Vec<Shape>> {
        match (self, padding_mode) {
            (Flatten(true), PaddingMode::NoPadding) => {
                // In this case we cannot flattening the unpadded shape normally may lead to issues
                // during proving because shape.pad_next_power_of_two().flatten() != shape.flatten().pad_next_power_of_two()
                // So the unpadded shapes must be padded first and then flattened
                Ok(input_shapes
                    .iter()
                    .map(|s| Shape::new(vec![s.next_power_of_two().product()]))
                    .collect())
            }
            _ => {
                // Flatten(false) or PaddingMode::Padding
                Ok(input_shapes
                    .iter()
                    .map(|s| Shape::new(vec![s.product()]))
                    .collect())
            }
        }
    }

    fn num_outputs(&self, num_inputs: usize) -> Result<usize> {
        Ok(num_inputs)
    }

    fn describe(&self) -> String {
        "Reshape".to_string()
    }

    fn is_provable(&self) -> bool {
        false
    }
}

impl Evaluate<f32> for Flatten {
    fn evaluate(&self, inputs: &[&WrappedTensor<f32>]) -> Result<LayerOut<f32>> {
        ensure!(
            inputs.len() == 1,
            "Found more than 1 input when evaluating reshape layer"
        );
        let input = inputs[0];
        let out = input.clone().flatten_1d();
        Ok(LayerOut::from_vec(vec![out]))
    }
}

impl Evaluate<Element> for Flatten {
    /// EXCEPTION to the "run unpadded" rule: Flatten must pad its input before
    /// flattening because the 1D layout depends on padded (power-of-two) dimensions.
    /// Without padding first, the flat index mapping would be inconsistent between
    /// inference and proving.
    fn evaluate(&self, inputs: &[&WrappedTensor<Element>]) -> Result<LayerOut<Element>> {
        ensure!(
            inputs.len() == 1,
            "Found more than 1 input when evaluating reshape layer"
        );
        let input = inputs[0];
        let padded = input.clone().pad_next_power_of_two();
        let mut out = padded.flatten_1d();
        // Set unpadded_shape = shape so downstream layers treat this as "unpadded"
        let shape = out.shape().clone();
        out.set_unpadded_shape(shape);
        Ok(LayerOut::from_vec(vec![out]))
    }
}

impl ProveInfo for Flatten {
    fn step_info<F: PrimeField>(&self, mut aux: ContextAux) -> Result<(LayerCtx<F>, ContextAux)> {
        aux.last_output_shape
            .iter_mut()
            .for_each(|s| *s = s.next_power_of_two());
        Ok((LayerCtx::Flatten, aux))
    }
}

impl PadOp for Flatten {
    fn pad_node(self, si: &mut ShapeInfo) -> Result<Self>
    where
        Self: Sized,
    {
        reshape(si)
    }
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;
    use std::ops::Range;

    use crate::{Element, Shape, Tensor, tensor::TensorTypeParam};

    use super::*;

    proptest! {
        #[test]
        fn test_flatten_with_f32(input in any_input::<f32>(1..5, 1..8)) {
            let expected = input.to_flatten();

            let layer = Flatten(false);
                let computed = layer.evaluate(&[&input.as_wrapped()]).expect("flatten evaluation must be successful");

            prop_assert_eq!(&expected, &computed.outputs[0].to_native());
        }

        #[test]
        fn test_flatten_with_element(input in any_input::<Element>(1..5, 1..8)) {
            // Element evaluate pads-then-flattens
            let expected = input.pad_next_power_of_two().to_flatten();

            let layer = Flatten(false);
            let computed = layer.evaluate(&[&input.as_wrapped()]).expect("flatten evaluation must be successful");

            prop_assert_eq!(&expected, &computed.outputs[0].to_native());
        }
    }

    fn any_input<T: TensorTypeParam>(
        rank: Range<usize>,
        size: Range<usize>,
    ) -> impl Strategy<Value = Tensor<T>> {
        (rank, size).prop_flat_map(|(rank, size)| {
            let shape = Shape::new([size].repeat(rank));
            Tensor::<T>::any(shape)
        })
    }
}
