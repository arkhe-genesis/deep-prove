use anyhow::ensure;
use ff_ext::ExtensionField;

use crate::{
    Element, Shape,
    layers::provable::{Evaluate, LayerOut},
    tensor::WrappedTensor,
};

pub struct Permute {
    args: Vec<usize>,
}

impl Permute {
    pub fn new(args: Vec<usize>) -> Self {
        assert_eq!(
            args.len(),
            3,
            "Only 3D tensors currently supported by permute"
        );
        Self { args }
    }
}

impl Evaluate<Element> for Permute {
    fn evaluate<E: ExtensionField>(
        &self,
        inputs: &[&WrappedTensor<Element>],
        _unpadded_input_shapes: &[Shape],
    ) -> anyhow::Result<LayerOut<Element, E>> {
        ensure!(
            inputs.iter().all(|t| t.rank() == 3),
            "Permute expects 3D tensors"
        );

        let mut output = Vec::with_capacity(inputs.len());
        for input in inputs {
            ensure!(input.rank() == 3, "Permutation only supports 3D tensors");
            let axes: Vec<isize> = self.args.iter().map(|v| *v as isize).collect::<Vec<_>>();
            let result = (*input).clone().permute(&axes)?;
            output.push(result);
        }

        Ok(LayerOut::from_vec(output))
    }
}

impl Evaluate<f32> for Permute {
    fn evaluate<E: ExtensionField>(
        &self,
        inputs: &[&WrappedTensor<f32>],
        _unpadded_input_shapes: &[Shape],
    ) -> anyhow::Result<LayerOut<f32, E>> {
        ensure!(
            inputs.iter().all(|t| t.rank() == 3),
            "Permute expects 3D tensors"
        );

        let mut output = Vec::with_capacity(inputs.len());
        for input in inputs {
            ensure!(input.rank() == 3, "Permutation only supports 3D tensors");
            let axes: Vec<isize> = self.args.iter().map(|v| *v as isize).collect::<Vec<_>>();
            let result = (*input).clone().permute(&axes)?;
            output.push(result);
        }

        Ok(LayerOut::from_vec(output))
    }
}

#[cfg(test)]
mod test {
    use ff_ext::GoldilocksExt2;
    use proptest::prelude::*;

    use crate::{
        Element, Shape, Tensor,
        layers::{permute::Permute, provable::Evaluate},
    };

    #[test]
    fn test_permute() {
        let input = Tensor::<Element>::random(&vec![2, 3, 4].into());
        let permute = Permute::new(vec![1, 0, 2]);
        let output = permute
            .evaluate::<GoldilocksExt2>(&[&input.as_wrapped()], &[])
            .unwrap();
        assert_eq!(output.outputs()[0].shape(), vec![3_usize, 2, 4].into());
    }

    proptest! {
        #[test]
        fn proptest_permute_layer(a in 2usize..64, b in 2usize..64, c in 2usize..64) {
            let permutations = [
                [0, 1, 2],
                [1, 0, 2],
                [1, 2, 0],
                [0, 2, 1],
                [2, 0, 1],
                [2, 1, 0],
            ];

            let element_data = Tensor::<Element>::random(&Shape::new(vec![a, b, c]));
            for order in &permutations {
                let expected = element_data.permute3d(order);
                let layer = Permute::new(order.to_vec());
                let result = layer.evaluate::<GoldilocksExt2>(&[&element_data.as_wrapped()], &[]).unwrap();
                prop_assert_eq!(&expected, &result.outputs()[0].to_native());
            }

            let float_data = Tensor::<Element>::random(&Shape::new(vec![a, b, c]));
            for order in &permutations {
                let expected = float_data.permute3d(order);
                let layer = Permute::new(order.to_vec());
                let result = layer.evaluate::<GoldilocksExt2>(&[&float_data.as_wrapped()], &[]).unwrap();
                prop_assert_eq!(&expected, &result.outputs()[0].to_native());
            }
        }
    }
}
