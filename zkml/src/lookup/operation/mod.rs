//! Defines the functionality for performing an operation that has to be verified via a lookup argument

use crate::{
    Claim, Element, Shape, Tensor, eval_zeroifier_mle,
    iop::ChallengeStorage,
    layers::requant::FIXED_POINT_SCALE,
    lookup::{
        context::{COLUMN_SEPARATOR, count_elements},
        logup_gkr::{
            prover::new_batch_multiple_sizes_prove,
            structs::{
                LogUpBatchProof, LogUpBatchVerifierClaim, LogUpInput, LogUpVerifierInstance,
                ProofType,
            },
            verifier::new_verify_logup_proof_multiple_sizes,
        },
        operation::{
            decomposer::{ChunkedInput, ChunkedOutput, ChunkingInfo},
            inputs::LookupInputConfig,
            variant::{LookupVariant, verifying::evaluate_dim_lt_poly},
        },
        table::Table,
    },
    quantization::ToField,
    tensor::WrappedTensor,
    to_bit_sequence_le,
};

use anyhow::{Result, anyhow, bail, ensure};

use either::Either;
use ff_ext::ExtensionField;

use multilinear_extensions::{
    Expression,
    mle::{IntoMLE, MultilinearExtension},
    util::ceil_log2,
};
use std::{collections::HashMap, sync::Arc};
use sumcheck::structs::IOPProof;
use transcript::Transcript;

pub mod decomposer;
pub mod generic_prove;
pub mod generic_verify;
pub mod inputs;
pub mod variant;

/// Trait defining the common functionality of an operation that requires a lookup argument to prove correct execution.
pub trait LookupOp {
    /// Applies the lookup operation to a [`WrappedTensor`], returning a new [`WrappedTensor`] containing the results.
    /// This method performs multiplication by the fixed point multiplier, right shift, and lookup table application.
    fn apply(
        &self,
        input: WrappedTensor<Element>,
        table: &Table,
    ) -> Result<WrappedTensor<Element>> {
        // Apply the fixed point multiplication, rounding and right shift
        let rescaled_input = input
            .mul_scalar(self.fixed_point_multiplier())
            .add_scalar(self.rounding_constant())
            .bitwise_right_shift_scalar(self.right_shift() as Element);
        // Perform the lookup, the table will handle clamping internally
        table.lookup_tensor(rescaled_input)
    }
    /// Getter for the intermediate bit size used during this lookup operation.
    /// The intermediate bit size is the maximum number of bits that can be represented in the lookup table inputs before the fixed point multiplier is applied.
    fn intermediate_bit_size(&self) -> usize;
    /// Getter for the size of the right shift performed during this lookup operation.
    fn right_shift(&self) -> usize;
    /// Getter for the fixed point multiplier used during this lookup operation.
    fn fixed_point_multiplier(&self) -> Element;
    /// Getter for the rounding value used during this lookup operation.
    fn rounding_constant(&self) -> Element {
        1 << (self.right_shift() - 1)
    }
    /// Getter that indicates whether this lookup operation is signed or unsigned. We classify
    /// signed operations as those that take inputs that can be positive or negative.
    /// Unsigned operations expect inputs that all have the same sign or are zero.
    fn is_signed(&self) -> bool;

    /// The value to use for padding to ensure the padded area of the output is zeroed out correctly.
    fn padding_value(&self) -> Element;

    /// The [`LookupVariant`] for this lookup operation. Defaults to [`LookupVariant::Standard`].
    fn variant(&self) -> LookupVariant {
        LookupVariant::Standard
    }

    /// Creates the [`ChunkingInfo`] for this lookup operation.
    fn chunking_info(&self, table: &Table) -> Result<ChunkingInfo> {
        ChunkingInfo::new(
            self.right_shift(),
            table,
            self.intermediate_bit_size() + FIXED_POINT_SCALE,
            1,
        )
    }

    /// Creates the [`LookupInputConfig`] for this lookup operation.
    fn input_config(
        &self,
        chunking_info: ChunkingInfo,
        unpadded_input_shape: &Shape,
    ) -> LookupInputConfig {
        LookupInputConfig::new(chunking_info, self.variant(), unpadded_input_shape.clone())
    }

    /// Generates the witness for the lookup operation. The input tensor is presumed to be in its unpadded form.
    fn generate_witness(
        &self,
        input: Tensor<Element>,
        value_table: &Table,
    ) -> Result<LookupOpWitness> {
        // Break the input tensor down into chunks over the final two dimensions
        let rank = input.shape().rank();
        let (second_last_dim, last_dim) = match rank {
            1 => (1, input.shape()[0]), // If rank 1, we treat as a single row
            r if r >= 2 => (input.shape()[rank - 2], input.shape()[rank - 1]),
            _ => bail!("Input tensor must have rank at least 1"),
        };

        let chunk_size = second_last_dim * last_dim;
        // We work out the size of the padded chunks so we can correctly pad the input before decomposition
        let padded_last_dim = last_dim.next_power_of_two();
        let last_dim_diff = padded_last_dim - last_dim;
        let padded_second_last_dim = second_last_dim.next_power_of_two();
        let chunk_diff = (padded_second_last_dim - second_last_dim) * padded_last_dim;
        // Create the ChunkingInfo to handle the decomposition of each chunk
        let chunking_info = self.chunking_info(value_table)?;
        // This takes each unpadded chunk, pads with the correct padding value, rescales and adds rounding constant, and then decomposes into chunks to be passed
        // to the various lookup tables.
        let chunked_inputs = input
            .into_data()
            .chunks(chunk_size)
            .map(|chunk| {
                // For each unpadded 2D chunk, pad with the correct padding value.
                let chunk_data = chunk
                    .chunks(last_dim)
                    .flat_map(|row| {
                        row.iter()
                            .map(|x| x * self.fixed_point_multiplier() + self.rounding_constant())
                            .chain(std::iter::repeat_n(
                                self.padding_value() * self.fixed_point_multiplier()
                                    + self.rounding_constant(),
                                last_dim_diff,
                            ))
                    })
                    .chain(std::iter::repeat_n(
                        self.padding_value() * self.fixed_point_multiplier()
                            + self.rounding_constant(),
                        chunk_diff,
                    ))
                    .collect::<Vec<Element>>();

                chunking_info.decompose_input(chunk_data)
            })
            .collect::<Vec<ChunkedInput>>();

        // Now we take the decomposed input chunks and perform lookups on each chunk to get the output chunks
        let chunked_outputs = chunked_inputs
            .iter()
            .map(|chunked_input| chunking_info.table_output(chunked_input, value_table))
            .collect::<Result<Vec<ChunkedOutput>>>()?;

        Ok(LookupOpWitness::new(chunked_inputs, chunked_outputs))
    }
}

#[derive(Clone, Debug)]
/// Struct containing the witness data for a lookup operation. This can then be used to create the necessary commitments.
pub struct LookupOpWitness {
    /// The evaluations of the chunked input polynomials used in the lookup operation.
    /// They are ordered as follows:
    /// - First come the shifted chunks, in order from least significant to most significant.
    /// - Next comes the value chunk.
    /// - Finally come the zeroing chunks, in order from least significant to most significant.
    /// - This pattern is repeated for each original 2D subtensor in the input tensor.
    pub chunked_inputs: Vec<ChunkedInput>,
    /// The chunked outputs of the lookup operation, corresponding to the output of the value chunk and the zeroing chunks.
    /// This pattern is repeated for each original 2D subtensor in the input tensor.
    pub chunked_outputs: Vec<ChunkedOutput>,
}

impl LookupOpWitness {
    /// Creates a new [`LookupOpWitness`] with the given [`ChunkedInput`], [`ChunkedOutput`] and extra data.
    pub fn new(chunked_inputs: Vec<ChunkedInput>, chunked_outputs: Vec<ChunkedOutput>) -> Self {
        Self {
            chunked_inputs,
            chunked_outputs,
        }
    }

    /// Creates an iterator over the [`ChunkedInput`] zipped with the [`ChunkedOutput`] for each chunk in the witness.
    pub fn input_output_iter(&self) -> impl Iterator<Item = (&ChunkedInput, &ChunkedOutput)> {
        self.chunked_inputs.iter().zip(self.chunked_outputs.iter())
    }

    /// Method that computes the number of times each lookup table entry is used in the witness.
    pub fn get_counts(&self, table: &Table) -> Vec<HashMap<Element, u64>> {
        let shift_check_iter = self
            .chunked_inputs
            .iter()
            .flat_map(|chunked_input| chunked_input.shifted_iter());

        let shift_check_counts = count_elements(shift_check_iter.copied());

        let value_counts = match table.num_columns() {
            1 => {
                let value_iter = self
                    .chunked_outputs
                    .iter()
                    .flat_map(|chunked_output| chunked_output.value_iter())
                    .copied();
                count_elements(value_iter)
            }
            2 => {
                let value_iter =
                    self.input_output_iter()
                        .flat_map(|(chunked_input, chunked_output)| {
                            chunked_input
                                .value_iter()
                                .zip(chunked_output.value_iter())
                                .map(|(input_val, output_val)| {
                                    input_val + COLUMN_SEPARATOR * output_val
                                })
                        });
                count_elements(value_iter)
            }
            _ => unreachable!("Tables only have 1 or 2 columns"),
        };

        let (zero_counts, signed_zero_counts) = match table.is_signed() {
            false => {
                let zero_table_counts =
                    self.input_output_iter()
                        .flat_map(|(chunked_input, chunked_output)| {
                            chunked_input
                                .zeroing_iter_no_sign()
                                .zip(chunked_output.zeroing_iter_no_sign())
                                .map(|(input_val, output_val)| {
                                    input_val + COLUMN_SEPARATOR * output_val
                                })
                        });
                (count_elements(zero_table_counts), HashMap::new())
            }
            true => {
                let zero_table_counts =
                    self.input_output_iter()
                        .flat_map(|(chunked_input, chunked_output)| {
                            chunked_input
                                .unsigned_chunk_iter()
                                .zip(chunked_output.unsigned_chunk_iter())
                                .map(|(input, output)| input + COLUMN_SEPARATOR * output)
                        });

                let signed_zero_table_counts =
                    self.input_output_iter()
                        .flat_map(|(chunked_input, chunked_output)| {
                            chunked_input
                                .signed_chunk_iter()
                                .zip(chunked_output.signed_chunk_iter())
                                .map(|(input, output)| input + COLUMN_SEPARATOR * output)
                        });
                (
                    count_elements(zero_table_counts),
                    count_elements(signed_zero_table_counts),
                )
            }
        };
        vec![
            shift_check_counts,
            value_counts,
            zero_counts,
            signed_zero_counts,
        ]
    }

    pub fn input_mle_evals<E: ExtensionField>(
        &self,
        num_value_columns: usize,
    ) -> Vec<Vec<E::BaseField>> {
        self.chunked_inputs
            .iter()
            .flat_map(|chunked_input| chunked_input.to_evals::<E>(num_value_columns))
            .collect()
    }

    pub fn output_mle_evals<E: ExtensionField>(&self) -> Vec<Vec<E::BaseField>> {
        self.chunked_outputs
            .iter()
            .flat_map(|chunked_output| chunked_output.to_evals::<E>())
            .collect()
    }
}

/// Function that given a [`LookupOp`], the unpadded input [`Shape`] and the evaluation point computes
/// the value that needs to be subtracted from the input [`Claim`] to remove the lookup padding value.
pub fn compute_lookup_padding_evaluation<E: ExtensionField>(
    padding_value: E,
    unpadded_input_shape: &Shape,
    dim_points: &[&[E]],
) -> Result<E> {
    let rank = unpadded_input_shape.rank();

    if unpadded_input_shape
        .iter()
        .skip(rank.saturating_sub(2))
        .all(|d| d.is_power_of_two())
    {
        // If all dimensions are powers of two then the padding evaluation is just the zero tensor evaluation which is zero
        return Ok(E::ZERO);
    }

    let unbroadcast_shape = unpadded_input_shape
        .iter()
        .enumerate()
        .map(|(i, &dim)| if i < rank.saturating_sub(2) { 1 } else { dim })
        .collect::<Vec<usize>>();

    let leading_lt_eval =
        unpadded_input_shape.broadcasting_evaluation(dim_points, &unbroadcast_shape)?;

    let (second_last_dim, last_dim) = if rank >= 2 {
        (
            unpadded_input_shape[rank - 2],
            unpadded_input_shape[rank - 1],
        )
    } else {
        (1, unpadded_input_shape[0])
    };

    let last_dim_lt_eval = if last_dim.is_power_of_two() {
        E::ZERO
    } else {
        let last_dim_lt_eval = evaluate_dim_lt_poly(dim_points[rank - 1], last_dim)?;
        E::ONE - last_dim_lt_eval
    };

    let final_eval = match second_last_dim {
        1 => leading_lt_eval * last_dim_lt_eval,
        d if d.is_power_of_two() => leading_lt_eval * last_dim_lt_eval,
        d => {
            let second_last_dim_lt_eval = evaluate_dim_lt_poly(dim_points[rank - 2], d)?;
            leading_lt_eval
                * (second_last_dim_lt_eval * last_dim_lt_eval + (E::ONE - second_last_dim_lt_eval))
        }
    } * padding_value;

    Ok(final_eval)
}

#[cfg(test)]
pub(crate) mod lookup_test_utils {
    use crate::{graph::NodeId, rng_from_env_or_random, to_base};
    use ark_std::{UniformRand, rand::Rng};
    use itertools::{Itertools, izip};

    use ff_ext::GoldilocksExt2 as F;

    use super::*;

    #[test]
    fn test_padding_eval_function() {
        let mut rng = rng_from_env_or_random();
        let padding_value: Element = 1;

        for _ in 0..10 {
            let rank = rng.gen_range(1usize..=4);
            let unpadded_shape = (0..rank)
                .map(|_| rng.gen_range(1usize..10))
                .collect::<Shape>();
            let num_sub_tensors = unpadded_shape
                .iter()
                .take(rank.saturating_sub(2))
                .product::<usize>();
            let sub_tensor_shape = unpadded_shape
                .iter()
                .skip(rank.saturating_sub(2))
                .copied()
                .collect::<Shape>();
            let intermediate_shape = unpadded_shape
                .iter()
                .enumerate()
                .map(|(i, &d)| {
                    if i < rank.saturating_sub(2) {
                        d
                    } else {
                        d.next_power_of_two()
                    }
                })
                .collect::<Shape>();
            let zero_tensor = Tensor::<Element>::zeros(sub_tensor_shape);
            let padded_sub_tensor = zero_tensor.pad_next_power_of_two_with_value(padding_value);
            let tensor_data = vec![padded_sub_tensor.into_data(); num_sub_tensors].concat();

            let input_tensor = Tensor::new(intermediate_shape, tensor_data)
                .unwrap()
                .pad_next_power_of_two();

            let tensor_variables = input_tensor.shape().num_vars().into_iter().sum::<usize>();

            let point = (0..tensor_variables)
                .map(|_| F::rand(&mut rng))
                .collect::<Vec<F>>();

            let input_mle = to_base::<F, _>(input_tensor.data()).into_mle();

            let mle_eval = input_mle.evaluate(&point);

            let dim_points = input_tensor.shape().split_point(&point).unwrap();
            let padding_eval = compute_lookup_padding_evaluation(
                padding_value.to_field(),
                &unpadded_shape,
                &dim_points,
            )
            .unwrap();

            assert_eq!(
                mle_eval, padding_eval,
                "Padding evaluation did not match MLE evaluation for rank {rank} and unpadded shape {unpadded_shape:?}, input tensor: {input_tensor}"
            );
        }
    }

    /// Utility function used in debugging to compare the lookup witness output with the layer evaluate output.
    #[allow(dead_code)]
    pub(crate) fn compare_output_and_witness<L>(
        lookup_op: &L,
        output: &Tensor<Element>,
        witness: &LookupOpWitness,
        table: &Table,
        node_id: NodeId,
    ) -> Result<()>
    where
        L: LookupOp,
    {
        let chunking_info = lookup_op.chunking_info(table)?;
        let mut full_output_chunks = Vec::<Element>::new();

        for output_chunk in witness.chunked_outputs.iter() {
            full_output_chunks.extend(chunking_info.combine_outputs(output_chunk));
        }

        let unpadded_shape = output.unpadded_shape();
        let rank = unpadded_shape.rank();
        let (non_padded, padded) = unpadded_shape.split_at(rank.saturating_sub(2));
        let tmp_shape = non_padded
            .iter()
            .copied()
            .chain(padded.iter().map(|d| d.next_power_of_two()))
            .collect::<Shape>();
        let witness_output_tensor =
            Tensor::new(tmp_shape, full_output_chunks)?.reduce_to_shape(unpadded_shape)?;

        let multi_cartesian = unpadded_shape
            .iter()
            .map(|d| 0..*d)
            .multi_cartesian_product();
        for (coord, &witness_out, &evaluate_out) in
            izip!(multi_cartesian, witness_output_tensor.data(), output.data())
        {
            if witness_out != evaluate_out {
                println!(
                    "Mismatch at coordinate {:?} for node {node_id} with table: {}: witness output was {witness_out} but evaluate output was {evaluate_out}",
                    coord,
                    table.name()
                );
            }
        }

        Ok(())
    }
}
