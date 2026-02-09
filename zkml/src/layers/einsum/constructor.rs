//! Module containing methods that construct common types of [`EinSum`] layers.

use anyhow::bail;
#[cfg(test)]
use tenstore::StorageKey;

use super::*;

impl<T> EinSum<T>
where
    T: TensorTypeParam,
{
    /// Constructs a dense (fully-connected) layer with the given weight matrix and optional bias.
    pub fn new_dense(matrix: TensorHandle<T>, bias: Option<TensorHandle<T>>) -> Result<EinSum<T>> {
        let input_eq = "A(j)@W(ij)".to_string();
        let mut output_equation = "O(i)".to_string();
        if let Some(ref bbias) = bias {
            ensure!(matrix.shape().nrows_2d() == bbias.shape()[0]);
            output_equation = "O(i)+BIAS(i)".to_string();
        }

        EinSum::<T>::new(
            format!("{input_eq}->{output_equation}"),
            vec![Some(matrix)],
            vec![bias],
        )
    }

    /// Constructs a matrix multiplication layer with the given left and right matrices, optional bias,
    /// and an option to specify whether the right matrix would require transposition.
    pub fn new_matmul(
        left_matrix: Option<TensorHandle<T>>,
        right_matrix: Option<TensorHandle<T>>,
        transpose_right: bool,
        bias: Option<TensorHandle<T>>,
    ) -> Result<EinSum<T>> {
        let output_equation = if bias.is_some() {
            "O(ij)+BIAS(j)".to_string()
        } else {
            "O(ij)".to_string()
        };
        match (left_matrix, right_matrix) {
            (None, None) => {
                let input_equation = if transpose_right {
                    "A(ik)@B(jk)".to_string()
                } else {
                    "A(ik)@B(kj)".to_string()
                };
                EinSum::<T>::new(
                    format!("{input_equation}->{output_equation}"),
                    vec![None],
                    vec![bias],
                )
            }
            (None, Some(weight)) => {
                let input_equation = if transpose_right {
                    "A(ik)@W(jk)".to_string()
                } else {
                    "A(ik)@W(kj)".to_string()
                };
                EinSum::<T>::new(
                    format!("{input_equation}->{output_equation}"),
                    vec![Some(weight)],
                    vec![bias],
                )
            }
            (Some(weight), None) => {
                // In this case left matrix is the constant weight matrix
                // normally this would give L(ij)@R(jk)->O(ik) but because we need the constant on the RHS
                // we write it as R(jk)@L(ij)->O(ik)
                let input_equation = if transpose_right {
                    "B(jk)@W(ik)".to_string()
                } else {
                    "B(kj)@W(ik)".to_string()
                };
                EinSum::<T>::new(
                    format!("{input_equation}->{output_equation}"),
                    vec![Some(weight)],
                    vec![bias],
                )
            }
            (Some(_), Some(_)) => {
                bail!("At least one of the matrices in a matmul layer must be input-dependent")
            }
        }
    }

    #[cfg(test)]
    pub fn random_dense(shape: Shape, layer_name: Option<StorageKey<T>>) -> Self {
        use crate::Tensor;
        use tenstore::GenStore;

        assert_eq!(shape.len(), 2);
        let (nrows, ncols) = (shape[0], shape[1]);
        let layer_name = layer_name.unwrap_or("dense".to_string().into());
        let matrix = TensorHandle::from_tensor(
            StorageKey::from(format!("{layer_name}_weight")),
            GenStore::new_empty(),
            Tensor::<T>::random(&vec![nrows, ncols].into()),
        );

        let bias = TensorHandle::from_tensor(
            StorageKey::from(format!("{layer_name}_bias")),
            GenStore::new_empty(),
            Tensor::<T>::random(&vec![nrows].into()),
        );
        Self::new_dense(matrix, Some(bias)).unwrap()
    }
}
