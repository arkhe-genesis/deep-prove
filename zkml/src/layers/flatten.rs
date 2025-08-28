use anyhow::{Result, ensure};
use ff_ext::ExtensionField;
use serde::{Deserialize, Serialize};

use crate::{
    Element, Tensor,
    iop::context::ContextAux,
    layers::LayerCtx,
    padding::{PaddingMode, ShapeInfo, reshape},
    tensor::{IntoBTensor, Number, Shape},
};

use super::provable::{Evaluate, LayerOut, NodeId, OpInfo, PadOp, ProveInfo};
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

impl Evaluate<f32> for Flatten {
    fn evaluate<E: ExtensionField>(
        &self,
        inputs: &[&Tensor<f32>],
        _unpadded_input_shapes: &[Shape],
    ) -> Result<LayerOut<f32, E>> {
        self.evaluate_internal(inputs, _unpadded_input_shapes)
    }
}

impl Evaluate<Element> for Flatten {
    fn evaluate<E: ExtensionField>(
        &self,
        inputs: &[&Tensor<Element>],
        _unpadded_input_shapes: &[Shape],
    ) -> Result<LayerOut<Element, E>> {
        self.evaluate_internal(inputs, _unpadded_input_shapes)
    }
}

impl Flatten {
    fn evaluate_internal<E, N>(
        &self,
        inputs: &[&Tensor<N>],
        _unpadded_input_shapes: &[Shape],
    ) -> Result<LayerOut<N, E>>
    where
        E: ExtensionField,
        N: Number + burn::tensor::Element,
        Tensor<N>: IntoBTensor,
    {
        ensure!(
            inputs.len() == 1,
            "Found more than 1 input when evaluating reshape layer"
        );
        let input = inputs[0];
        let rank = input.rank();
        let res = match rank {
            1 => {
                return Ok(LayerOut::from_vec(vec![input.clone()]));
            }
            2 => {
                let input = input.clone().into_btensor::<2>();
                input.flatten::<1>(0, rank - 1)
            }
            3 => {
                let input = input.clone().into_btensor::<3>();
                input.flatten::<1>(0, rank - 1)
            }
            4 => {
                let input = input.clone().into_btensor::<4>();
                input.flatten::<1>(0, rank - 1)
            }
            _ => {
                panic!("Unexpected rank {rank}")
            }
        };
        let data = res.to_data().into_vec().expect("Failed to compute Flatten");
        let shape = Shape::new(vec![data.len()]);
        let out = Tensor::<N>::new(shape, data);
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

    use crate::tensor::Shape;

    use super::*;

    proptest! {
        #[test]
        fn test_flatten_with_f32(input in any_input::<f32>(1..5, 1..8)) {
            let expected = input.flatten();

            let layer = Flatten;
            let computed = layer.evaluate::<GoldilocksExt2>(&[&input], &[]).expect("flatten evaluation must be successful");

            prop_assert_eq!(&expected, &computed.outputs[0]);
        }

        #[test]
        fn test_flatten_with_element(input in any_input::<Element>(1..5, 1..8)) {
            let expected = input.flatten();

            let layer = Flatten;
            let computed = layer.evaluate::<GoldilocksExt2>(&[&input], &[]).expect("flatten evaluation must be successful");

            prop_assert_eq!(&expected, &computed.outputs[0]);
        }
    }

    fn any_input<T: Number>(
        rank: Range<usize>,
        size: Range<usize>,
    ) -> impl Strategy<Value = Tensor<T>> {
        (rank, size).prop_flat_map(|(rank, size)| {
            let shape = Shape::new([size].repeat(rank));
            Tensor::<T>::any(shape)
        })
    }
}
