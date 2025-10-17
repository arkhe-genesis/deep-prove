use ff_ext::ExtensionField;

use crate::tensor::{TensorTypeParam, WrappedTensor};

use super::provable::LayerOut;

pub struct ScalarMul<N: TensorTypeParam> {
    constant: N,
}

impl<N: TensorTypeParam> ScalarMul<N> {
    pub fn new(cst: N) -> Self {
        Self { constant: cst }
    }

    pub fn evaluate<E: ExtensionField>(
        &self,
        inputs: &[&WrappedTensor<N>],
    ) -> anyhow::Result<LayerOut<N, E>> {
        let result = inputs
            .iter()
            .map(|input| (*input).clone().mul_scalar(self.constant))
            .collect::<Vec<_>>();
        Ok(LayerOut::from_vec(result))
    }
}

#[cfg(test)]
mod test {
    use ff_ext::GoldilocksExt2;

    use crate::Tensor;

    use super::*;

    #[test]
    fn test_scalar_mul() {
        let scalar_mul = ScalarMul::new(2.0);
        let input = WrappedTensor::try_from(&Tensor::new(
            vec![1, 2, 3].into(),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        ))
        .unwrap();
        let result = scalar_mul.evaluate::<GoldilocksExt2>(&[&input]).unwrap();
        assert_eq!(
            result.outputs[0]
                .clone()
                .to_data()
                .as_slice::<f32>()
                .unwrap(),
            &[2.0, 4.0, 6.0, 8.0, 10.0, 12.0]
        );
    }
}
