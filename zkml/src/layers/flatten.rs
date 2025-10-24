use super::provable::{Evaluate, LayerOut, OpInfo, PadOp, ProveInfo};
use crate::{
    Shape,
    graph::NodeId,
    iop::context::ContextAux,
    layers::LayerCtx,
    padding::{PaddingMode, ShapeInfo, reshape},
    tensor::{TensorTypeParam, WrappedTensor},
};
use anyhow::{Result, ensure};
use ff_ext::ExtensionField;
use serde::{Deserialize, Serialize};

/// Short name used to identify the flatten layer
pub const FLATTEN_LAYER: &str = "FLTT";

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Flatten;
/// Even if empty, we need a context such that it implements the default
/// methods of `VerifiableCtx``
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FlattenCtx;

impl OpInfo for Flatten {
    fn output_shapes(&self, input_shapes: &[Shape], _padding_mode: PaddingMode) -> Vec<Shape> {
        input_shapes
            .iter()
            .map(|s| Shape::new(vec![s.product()]))
            .collect()
    }

    fn num_outputs(&self, num_inputs: usize) -> usize {
        num_inputs
    }

    fn describe(&self) -> String {
        "Reshape".to_string()
    }

    fn is_provable(&self) -> bool {
        false
    }
}

impl<T> Evaluate<T> for Flatten
where
    T: TensorTypeParam,
{
    fn evaluate<E: ExtensionField>(
        &self,
        inputs: &[&WrappedTensor<T>],
        _unpadded_input_shapes: &[Shape],
    ) -> Result<LayerOut<T, E>> {
        ensure!(
            inputs.len() == 1,
            "Found more than 1 input when evaluating reshape layer"
        );
        let input = inputs[0];
        let out = input.clone().flatten_1d();
        Ok(LayerOut::from_vec(vec![out]))
    }
}

impl ProveInfo for Flatten {
    fn step_info<E: ExtensionField>(
        &self,
        _id: NodeId,
        mut aux: ContextAux,
    ) -> Result<(LayerCtx<E>, ContextAux)> {
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
    use ff_ext::GoldilocksExt2;
    use proptest::prelude::*;
    use std::ops::Range;

    use crate::{Element, Shape, Tensor};

    use super::*;

    proptest! {
        #[test]
        fn test_flatten_with_f32(input in any_input::<f32>(1..5, 1..8)) {
            let expected = input.to_flatten();

            let layer = Flatten;
            let computed = layer.evaluate::<GoldilocksExt2>(&[&input.as_wrapped()], &[]).expect("flatten evaluation must be successful");

            prop_assert_eq!(&expected, &computed.outputs[0].to_native());
        }

        #[test]
        fn test_flatten_with_element(input in any_input::<Element>(1..5, 1..8)) {
            let expected = input.to_flatten();

            let layer = Flatten;
            let computed = layer.evaluate::<GoldilocksExt2>(&[&input.as_wrapped()], &[]).expect("flatten evaluation must be successful");

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
