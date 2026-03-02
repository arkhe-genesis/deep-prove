//! Module for calculating the rotation matrices used in DuQuant quantisation.

use crate::{Shape, tensor::WrappedTensor};

use super::*;

use anyhow::{Result, ensure};

/// Generates a Hadamard matrix of size 2^k that has not been normalised (so H * H^t will have diagonal elements equal to 2^k).
pub fn get_hadamard_matrix_no_normalisation(k: usize) -> Result<Tensor<f32>> {
    ensure!(k >= 1, "k must be at least 1, got {}", k);
    if k == 1 {
        let data = vec![1.0f32, 1.0f32, 1.0f32, -1.0f32];
        Tensor::<f32>::new(vec![2, 2].into(), data)
    } else {
        let prev_hadamard = get_hadamard_matrix_no_normalisation(k - 1)?.into_data();
        let size = 1usize << (k - 1); // 2^(k-1)
        let (top, bottom) =
            prev_hadamard
                .chunks(size)
                .fold((vec![], vec![]), |(mut top, mut bottom), row| {
                    top.extend_from_slice(row);
                    top.extend_from_slice(row);
                    bottom.extend_from_slice(row);
                    bottom.extend_from_slice(&row.iter().map(|x| -x).collect::<Vec<f32>>());

                    (top, bottom)
                });
        let data = [top, bottom].concat();
        Tensor::<f32>::new(vec![size * 2, size * 2].into(), data)
    }
}

pub fn apply_hadamard<const ON_RIGHT: bool, const INVERT: bool>(
    tensor: &Tensor<f32>,
    k: usize,
) -> Result<Tensor<f32>> {
    let mut hadamard_no_norm = get_hadamard_matrix_no_normalisation(k)?;
    if INVERT {
        hadamard_no_norm = hadamard_no_norm.transpose()?;
    }
    let hadamard_no_norm = WrappedTensor::try_from(hadamard_no_norm)?;
    let block_size = 1usize << k;
    let data = if ON_RIGHT {
        let full_size = tensor.shape().numel();
        let reshaped_tensor = WrappedTensor::try_from(tensor)?
            .reshape(Shape::new(vec![full_size / block_size, block_size]).into())?;
        reshaped_tensor.matmul(hadamard_no_norm)?.get_data()
    } else {
        let reshaped = WrappedTensor::try_from(reshape_for_block_diag_left(tensor, block_size)?)?;
        let transpose = hadamard_no_norm.transpose();
        let multiplied = transpose.matmul(reshaped)?;
        unshape_for_block_diagonal_left(&Tensor::try_from(multiplied)?, tensor.shape().dim(0))?
            .into_data()
    };

    let scalar = (block_size as f32).sqrt().recip();
    let scaled_data = data.iter().map(|v| v * scalar).collect::<Vec<f32>>();
    Tensor::<f32>::new(tensor.shape().clone(), scaled_data)
}

fn reshape_for_block_diag_left(tensor: &Tensor<f32>, block_size: usize) -> Result<Tensor<f32>> {
    let dim_size = tensor.shape().dim(0);
    let full_size = tensor.shape().numel();
    let row_size = full_size / dim_size;
    let new_row_size = full_size / block_size;

    let new_data = tensor
        .data()
        .chunks(row_size * block_size)
        .fold(vec![vec![]; block_size], |mut acc, chunk| {
            for (i, row) in chunk.chunks(row_size).enumerate() {
                acc[i].extend_from_slice(row);
            }
            acc
        })
        .concat();

    Tensor::<f32>::new(Shape::new(vec![block_size, new_row_size]), new_data)
}

fn unshape_for_block_diagonal_left(
    tensor: &Tensor<f32>,
    original_dim_size: usize,
) -> Result<Tensor<f32>> {
    let block_size = tensor.shape().dim(0);
    let full_size = tensor.shape().numel();
    let row_size = full_size / block_size;
    let new_row_size = full_size / original_dim_size;

    let new_data = tensor
        .data()
        .chunks(row_size)
        .fold(vec![vec![]; original_dim_size], |mut acc, chunk| {
            for (i, row) in chunk.chunks(new_row_size).enumerate() {
                acc[i].extend_from_slice(row);
            }
            acc
        })
        .concat();

    Tensor::<f32>::new(Shape::new(vec![original_dim_size, new_row_size]), new_data)
}

/// Builds a permutation matrix that moves the columns of the original tensor into blocks so that they are evenly distributed.
/// So for three blocks block 1 contains all the columns that are 0 mod 3, block 2 contains all the columns that are 1 mod 3, and block 3 contains all the columns that are 2 mod 3.
fn build_permutation_matrix(block_size: usize, dim_size: usize) -> Result<Tensor<f32>> {
    let number_of_blocks = dim_size / block_size;

    if number_of_blocks == 0 {
        return Ok(Tensor::<f32>::diagonal(dim_size, 1.0));
    }

    let mut data = vec![];
    for block_number in 0..number_of_blocks {
        for i in 0..block_size {
            let mut data_row = vec![0.0f32; dim_size];
            data_row[block_number + i * number_of_blocks] = 1.0;
            data.extend_from_slice(&data_row);
        }
    }

    let tmp = Tensor::<f32>::new(vec![dim_size, dim_size].into(), data)?;

    tmp.transpose()
}

pub fn apply_hadamard_with_perm<const ON_RIGHT: bool, const INVERT: bool>(
    tensor: &Tensor<f32>,
    k: usize,
) -> Result<Tensor<f32>> {
    let mut hadamard_no_norm = get_hadamard_matrix_no_normalisation(k)?;
    if INVERT {
        hadamard_no_norm = hadamard_no_norm.transpose()?;
    }
    let hadamard_no_norm = WrappedTensor::try_from(hadamard_no_norm)?;
    let block_size = 1usize << k;
    let data = if ON_RIGHT {
        let full_size = tensor.shape().numel();
        let dim_size = tensor.shape().dim(-1);
        let mut perm_matrix = build_permutation_matrix(block_size, dim_size)?;
        if INVERT {
            perm_matrix = perm_matrix.transpose()?;
        }
        let perm_matrix = WrappedTensor::try_from(perm_matrix)?;
        let reshaped_tensor = WrappedTensor::try_from(tensor)?
            .reshape(vec![full_size / block_size, block_size].into())?;

        let intermediate = reshaped_tensor
            .matmul(hadamard_no_norm.clone())?
            .reshape(Shape::new(vec![full_size / dim_size, dim_size]).into())?;
        let permuted = intermediate.matmul(perm_matrix)?;
        let second_reshaped = permuted.reshape(vec![full_size / block_size, block_size].into())?;
        second_reshaped.matmul(hadamard_no_norm.clone())?.get_data()
    } else {
        let full_size = tensor.shape().numel();
        let reshaped = WrappedTensor::try_from(reshape_for_block_diag_left(tensor, block_size)?)?;
        let transpose = hadamard_no_norm.transpose();
        let dim_size = tensor.shape().dim(0);
        let mut perm_matrix = build_permutation_matrix(block_size, dim_size)?;
        if INVERT {
            perm_matrix = perm_matrix.transpose()?;
        }
        let perm_matrix = WrappedTensor::try_from(perm_matrix)?;
        let multiplied = transpose.clone().matmul(reshaped)?;
        let intermediate =
            unshape_for_block_diagonal_left(&Tensor::try_from(multiplied)?, dim_size)?
                .reshaped(Shape::new(vec![dim_size, full_size / dim_size]))?;
        let intermediate = WrappedTensor::try_from(intermediate)?;
        let permuted = perm_matrix
            .matmul(intermediate)?
            .reshape(tensor.shape().into())?;
        let final_reshaped = reshape_for_block_diag_left(&Tensor::try_from(permuted)?, block_size)?;
        let final_mult = transpose.matmul(WrappedTensor::try_from(final_reshaped)?)?;
        unshape_for_block_diagonal_left(&Tensor::try_from(final_mult)?, dim_size)?.into_data()
    };

    let scalar = (block_size as f32).recip();
    let scaled_data = data.iter().map(|v| v * scalar).collect::<Vec<f32>>();
    Tensor::<f32>::new(tensor.shape().clone(), scaled_data)
}

#[cfg(test)]
mod tests {

    use itertools::Itertools;

    use super::*;

    #[test]
    fn test_orthoganol_constructors() -> Result<()> {
        orthogonality_tester::<true, _>(get_hadamard_matrix_no_normalisation)
    }

    fn orthogonality_tester<const USE_LOG_SIZE: bool, F: Fn(usize) -> Result<Tensor<f32>>>(
        function: F,
    ) -> Result<()> {
        for log_size in 2..12 {
            let size = 1 << log_size;
            let rotation_matrix = if USE_LOG_SIZE {
                function(log_size)?
            } else {
                function(size)?
            };
            assert_eq!(
                rotation_matrix.shape().as_slice(),
                vec![size, size].as_slice()
            );
            // Additional checks can be added here to verify the properties of the rotation matrix
            let inverse_matrix = rotation_matrix.transpose().unwrap();
            let identity_matrix = rotation_matrix.matmul(&inverse_matrix).unwrap();
            let mut diagonal_elements = vec![];
            for i in 0..size {
                for j in 0..size {
                    if i == j {
                        diagonal_elements.push(identity_matrix.data()[i * size + j]);
                    } else {
                        ensure!(
                            identity_matrix.data()[i * size + j].abs() < 1e-6,
                            "size: {size} i: {i}, j: {j}, value: {}",
                            identity_matrix.data()[i * size + j]
                        );
                    }
                }
            }
            ensure!(
                diagonal_elements.iter().all_equal(),
                "Diagonal elements are not all the same for size {size}: {diagonal_elements:?}"
            );
        }
        Ok(())
    }

    #[test]
    fn test_permutation_matrix() -> Result<()> {
        let block_size = 2;
        let dim_size = 6;
        let permutation_matrix = build_permutation_matrix(block_size, dim_size)?;

        let expected_data = vec![
            1.0, 0.0, 0.0, 0.0, 0.0, 0.0, //
            0.0, 0.0, 1.0, 0.0, 0.0, 0.0, //
            0.0, 0.0, 0.0, 0.0, 1.0, 0.0, //
            0.0, 1.0, 0.0, 0.0, 0.0, 0.0, //
            0.0, 0.0, 0.0, 1.0, 0.0, 0.0, //
            0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ];
        let expected_matrix = Tensor::<f32>::new(vec![dim_size, dim_size].into(), expected_data)?;
        let transpose = permutation_matrix.transpose()?;
        assert_eq!(permutation_matrix, expected_matrix);

        let identity = permutation_matrix.matmul(&transpose)?;
        println!("Identity matrix:\n{}", identity);
        Ok(())
    }

    #[test]
    fn test_apply_with_perm() -> Result<()> {
        let dim_size = 16usize;

        let diag = Tensor::<f32>::diagonal(dim_size, 1.0f32);

        let applied = apply_hadamard_with_perm::<true, false>(&diag, 3)?;

        let left_applied = apply_hadamard_with_perm::<false, true>(&diag, 3)?;

        let output = applied.matmul(&left_applied)?;

        ensure!(diag == output, "Output did not match identity");
        Ok(())
    }
}
