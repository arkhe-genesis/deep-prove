//! Module containign code for performing proving friendly requantisation. This is done via a [fixed point multiplication](https://en.wikipedia.org/wiki/Fixed-point_arithmetic#Binary_fixed-point_multiplication) and use of lookup arguments.

use crate::{
    Claim, Element, Prover, ProverContext, ScalingFactor, Tensor,
    commit::{compute_betas_eval, identity_eval},
    iop::{
        ChallengeStorage,
        context::{ContextAux, ShapeStep},
        verifier::Verifier,
    },
    layers::LayerProof,
    lookup::{
        context::{
            COLUMN_SEPARATOR, LayerLookupContext, LookupWitnessGen, TableType, count_elements,
        },
        logup_gkr::{
            prover::batch_multiple_sizes_prove,
            structs::{LogUpBatchProof, LogUpInput},
            verifier::verify_logup_proof_multiple_sizes,
        },
    },
    model::StepData,
    padding::PaddingMode,
    quantization::{self, Fieldizer},
    tensor::Shape,
    to_base,
};
use anyhow::{Context as CC, Result, anyhow, ensure};
use ceno_p3::field::FieldAlgebra;
use either::Either;
use ff_ext::ExtensionField;
use itertools::Itertools;
use multilinear_extensions::{
    Expression,
    mle::IntoMLE,
    util::{ceil_log2, transpose},
    utils::eval_by_expr_with_instance,
    virtual_poly::VPAuxInfo,
    virtual_polys::VirtualPolynomialsBuilder,
};
use witness::RowMajorMatrix;

use mpcs::PolynomialCommitmentScheme;

use rayon::prelude::*;
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use sumcheck::{
    structs::{IOPProof, IOPProverState, IOPVerifierState},
    util::optimal_sumcheck_threads,
};
use tenstore::TenStore;
use transcript::Transcript;

use super::{
    LayerCtx,
    provable::{Evaluate, LayerOut, NodeId, OpInfo, PadOp, ProvableOp, ProveInfo, VerifiableCtx},
};

/// Constant used in fixed point multiplication for normalised [`f32`] values
const FIXED_POINT_SCALE: usize = 25;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Copy, PartialOrd)]
/// This struct contains the information used in requantisation (i.e. rescaling and clamping)
/// The fields are:
/// - `multiplier`: This is the actual [`f32`] value calculated as `S1 * S2 / S3` and in traditional quantisation is what we would multiply by and then round to requantise
/// - `right_shift`: This is `multiplier.log2().trunc().abs()`
/// - `fixed_point_multiplier`: This is `2.0.powf(multiplier.log2().fract()) * (1 << `fp_scale`)`, `fp_scale` is chosen to be at least 25 bits as the [`f32`] mantissa is only 24 bits long so this should retain all bits.
/// - `fp_scale`: This is calculated so that `fp_scale + right_shift` is a multiple of [`quantization::BIT_LEN`], that way we only need one size of range table.
/// - `intermediate_bit_size`: This is the maximum number of bits a value can have before its requantised.
pub struct Requant {
    /// After multiplying by `self.fixed_point_multiplier` the value need to be shifted by this plus 25.
    pub right_shift: usize,
    /// The normalised scaling factor represented as a fixed point multiplier (it should have 24 fractional bits)
    pub fixed_point_multiplier: Element,
    /// The scale used for the fixed point multiplier, it is calculated to be the smallest value greater than or equal to [`FIXED_POINT_SCALE`] such that
    /// the right shift we perform is a multiple of [`quantization::BIT_LEN`]
    pub fp_scale: usize,
    /// THe actual multiplier, this is mainly used to compare accuracy, it has no purpose in actual proving
    pub multiplier: f32,
    /// This field represents how many bits the max absolute value can be
    pub(crate) intermediate_bit_size: usize,
}

/// Info related to the lookup protocol necessary to requantize
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound(serialize = "E: Serialize", deserialize = "E: DeserializeOwned"))]
pub struct RequantCtx<E: ExtensionField> {
    pub requant: Requant,
    pub node_id: NodeId,
    pub num_vars: usize,
    pub lookup_ctx: LayerLookupContext,
    pub sumcheck_expression: Expression<E>,
}

#[derive(Clone, Serialize, Deserialize)]
/// Struct holding all the information needed to verify requantisation was performed correctly.
/// This includes both lookup proofs and an additional sumcheck proof that we use so that all evaluations are at the same point.
#[serde(bound(serialize = "E: Serialize", deserialize = "E: DeserializeOwned"))]
pub struct RequantProof<E: ExtensionField, PCS: PolynomialCommitmentScheme<E>>
where
    E::BaseField: Serialize + DeserializeOwned,
{
    /// proof for the accumulation of the claim from activation + claim from lookup for the same poly
    /// e.g. the "link" between an activation and requant layer
    pub(crate) io_accumulation: IOPProof<E>,
    /// The evalaution claims about witness polynomials from the io_accumulation sumcheck
    pub(crate) io_eval: Vec<E>,
    /// The logup batch proof for all the lookups
    pub(crate) logup_proof: LogUpBatchProof<E>,
    /// COmmitments to lookup polynomials, they are in the order clamping commitments -> shifted commitments
    pub(crate) commitment: PCS::Commitment,
}

impl<E, PCS> RequantProof<E, PCS>
where
    E: ExtensionField,
    PCS: PolynomialCommitmentScheme<E>,
{
    pub(crate) fn write_commitment<T: Transcript<E>>(&self, transcript: &mut T) -> Result<()> {
        PCS::write_commitment(&self.commitment, transcript).map_err(|e| anyhow!("{e:?}"))
    }
}

const IS_PROVABLE: bool = true;

impl OpInfo for Requant {
    fn output_shapes(&self, input_shapes: &[Shape], _padding_mode: PaddingMode) -> Vec<Shape> {
        input_shapes.to_vec() // preserve the input shape
    }

    fn num_outputs(&self, num_inputs: usize) -> usize {
        num_inputs
    }

    fn describe(&self) -> String {
        format!(
            "Requant: right shift: {}, scale: {}",
            self.shift(),
            self.multiplier,
        )
    }

    fn is_provable(&self) -> bool {
        IS_PROVABLE
    }
}

impl Evaluate<Element> for Requant {
    fn evaluate<E: ExtensionField>(
        &self,
        inputs: &[&Tensor<Element>],
        _unpadded_input_shapes: &[Shape],
    ) -> Result<LayerOut<Element, E>> {
        let result = inputs
            .iter()
            .map(|input| {
                // We use this value to determine if any of the inputs are too large to be requantised (i.e. they fall outside the clamping table)
                let max_abs_val: Element = 1 << self.intermediate_bit_size;
                let res = input
                    .get_data()
                    .iter()
                    .enumerate()
                    .map(|(i, elem)| {
                        ensure!(
                            elem.abs() <= max_abs_val,
                            "Could not apply requantisation, tensor element {} had absolute value too large, given value: {}, max value: {}",
                            i, elem, max_abs_val
                        );

                        let rounding: Element = 1 << (self.shift() - 1);
                        let unclamped = (rounding + elem * self.fixed_point_multiplier) >> self.shift();

                        if unclamped >= *quantization::MAX {
                            Ok(*quantization::MAX)
                        } else if unclamped <= *quantization::MIN {
                            Ok(*quantization::MIN)
                        } else {
                            Ok(unclamped)
                        }
                    })
                    .collect::<Result<Vec<Element>, anyhow::Error>>()?;
                Ok(Tensor::<Element>::new(input.shape(), res))
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(LayerOut::from_vec(result))
    }
}

impl ProveInfo for Requant {
    fn step_info<E: ExtensionField>(
        &self,
        id: NodeId,
        mut aux: ContextAux,
    ) -> Result<(LayerCtx<E>, ContextAux)> {
        aux.tables.insert(TableType::Range);

        // Add ZeroTable to the aux if needed
        let lookup_ctx = if self.number_of_zero_chunks() != 0 {
            let (number_shifted_chunks, _, _) = self.shifted_chunks_data();
            aux.tables.insert(TableType::RequantZeroTable);
            let tables = vec![TableType::Range, TableType::RequantZeroTable];
            let instances_per_table = vec![number_shifted_chunks + 1, self.number_of_zero_chunks()];
            LayerLookupContext::new(tables, instances_per_table)
        } else {
            let (number_shifted_chunks, _, _) = self.shifted_chunks_data();

            let tables = vec![TableType::Range];
            let instances_per_table = vec![number_shifted_chunks + 1];
            LayerLookupContext::new(tables, instances_per_table)
        };

        let sumcheck_expression =
            build_requant_sumcheck_expression::<E>(self.number_of_zero_chunks());

        // `try_fold` would not allow returning of `Err` values
        // from here and would short-circuit
        // instead of looping over all values in the iterator
        #[allow(clippy::manual_try_fold)]
        let num_vars = aux
            .last_output_shape
            .iter_mut()
            .fold(Ok(None), |expected_num_vars, shape| {
                let num_vars = shape.iter().map(|dim| ceil_log2(*dim)).sum::<usize>();
                if let Some(vars) = expected_num_vars? {
                    ensure!(
                        vars == num_vars,
                        "All input shapes for requant layer \
                        must have the same number of variables"
                    );
                }
                Ok(Some(num_vars))
            })?
            .expect("No input shape found for requant layer?");
        // Set the model polys to be empty
        aux.model_polys = None;
        aux.max_poly_len = aux
            .last_output_shape
            .iter()
            .fold(aux.max_poly_len, |acc, shapes| {
                acc.max(shapes.next_power_of_two().product())
            });

        Ok((
            LayerCtx::Requant(RequantCtx {
                requant: *self,
                node_id: id,
                num_vars,
                lookup_ctx,
                sumcheck_expression,
            }),
            aux,
        ))
    }
}

/// Function used to construct the sumcheck to relate the outputs of the lookup to the input of the following layer.
/// The number of zero chunks dictates what this expression looks like. If `number_zero_chunks` is non-zero then we have
/// that the `last_claim.eval` should be the sum over the boolean hypercube of
///  `(PROD zero_chunk_i) * value + (1 - (PROD zero_chunk_i)) * (quant_max + sign * (quant_max - quant_min))`
/// in this case we also check that the `sign` polynomial is constructed correctly (it should only contain the values `0` and `-1`) using
///  `sign * (1 + sign)`
/// Finally in this case we also have to show that the `zero_chunk_i` and `value` polynomials are the same as the ones used in the lookup argument so we take a random
/// linear combination of `zero_chunk_i` and `value + quant_max` (we have to add `quant_max` here because the lookup on `value` is performed on it shifted so that all its values are positive).
///
/// When `number_zero_chunks` is zero, we have a simpler case where we just need to show that the `last_claim.eval` is equal to `value` and that `value + quant_max` is the polynomial used in the lookup.
fn build_requant_sumcheck_expression<E: ExtensionField>(
    number_zero_chunks: usize,
) -> Expression<E> {
    // The first polynomial fed to the `VirtualPolynomialsBuilder` will be the `value` poly.
    let value_expr = Expression::WitIn(0);
    if number_zero_chunks != 0 {
        // Here we construct the product of the zero chunks as well as their contribution to the random linear combination
        let (zero_out_prod_expr, zero_out_sum) = (1..=number_zero_chunks).fold(
            (
                Expression::Constant(Either::Right(E::ONE)),
                Expression::Constant(Either::Right(E::ZERO)),
            ),
            |(prod_acc, sum_acc), j| {
                (
                    prod_acc * Expression::WitIn(j as u16),
                    sum_acc
                        + Expression::WitIn(j as u16)
                            * Expression::Challenge(0, 2 + j, E::ONE, E::ZERO),
                )
            },
        );
        // This is the offset for the IDs of the remaining witness polynomials, they should be `sign` and then the three `eq_polys` needed.
        let id_offset = (1 + number_zero_chunks) as u16;
        let sign_expr = Expression::WitIn(id_offset);
        let last_claim_eq_expr = Expression::WitIn(id_offset + 1);
        let sign_eq_expr = Expression::WitIn(id_offset + 2);
        let logup_eq_expr = Expression::WitIn(id_offset + 3);
        // Constants that we need in the expression
        let quant_max_field: E = (*quantization::MAX).to_field();
        let value_const = E::from_canonical_u64(1 << (*quantization::BIT_LEN - 1));
        let sign_const = quant_max_field + quant_max_field + E::ONE;
        // This is the part of the expression which relates `last_claim.eval` to the lookup outputs.
        let first_part = last_claim_eq_expr
            * (zero_out_prod_expr.clone() * value_expr.clone()
                + (Expression::Constant(Either::Right(E::ONE)) - zero_out_prod_expr)
                    * (Expression::Constant(Either::Right(quant_max_field))
                        + sign_expr.clone() * Expression::Constant(Either::Right(sign_const))));
        // This part of the expression proves correct construction of `sign`
        let second_part = Expression::Challenge(0, 2, E::ONE, E::ZERO)
            * sign_eq_expr
            * (sign_expr.clone() * (Expression::Constant(Either::Right(E::ONE)) + sign_expr));
        // This part of the expression directly links the lookup output claims to this sumcheck.
        let third_part = logup_eq_expr
            * (Expression::Challenge(0, 1, E::ONE, E::ZERO)
                * (value_expr + Expression::Constant(Either::Right(value_const)))
                + zero_out_sum);
        first_part + second_part + third_part
    } else {
        // This is the case where there were no zero chunks so no clamping required, in which case the expression only deals with the `value` poly and
        // two `eq_polys`
        let value_const = E::from_canonical_u64(1 << (*quantization::BIT_LEN - 1));
        Expression::WitIn(1) * value_expr.clone()
            + Expression::Challenge(0, 1, E::ONE, E::ZERO)
                * Expression::WitIn(2)
                * (value_expr + Expression::Constant(Either::Right(value_const)))
    }
}

impl LayerLookupContext {
    /// Softmax behaves slightly differently to normal lookups so we have a custom method to generate the [`LogUpInput`].
    pub fn create_logup_inputs_requant<PCS, E>(
        &self,
        layer_commitment: &PCS::CommitmentWithWitness,
        challenge_storage: &ChallengeStorage<E>,
    ) -> anyhow::Result<Vec<LogUpInput<E>>>
    where
        E: ExtensionField,
        PCS: PolynomialCommitmentScheme<E>,
    {
        // First we extract the polynomials from the layer_commitment
        let polys = PCS::get_arc_mle_witness_from_commitment(layer_commitment);

        // There should be at least as many polynomials as there are lookup columns total
        let total_lookup_columns = self
            .tables
            .iter()
            .zip(self.instances_per_table.iter())
            .map(|(tt, &n)| tt.num_columns() * n)
            .sum::<usize>();

        ensure!(
            polys.len() >= total_lookup_columns,
            "Cannot create Softmax LogUp inputs because we were only provided with {} polynomials and expected {} lookup columns",
            polys.len(),
            total_lookup_columns
        );

        if self.tables.len() == 1 {
            let mut column_evals = polys
                .iter()
                .take(polys.len() - 1)
                .map(|p| p.get_base_field_vec().to_vec())
                .collect::<Vec<Vec<E::BaseField>>>();
            let value_evals = polys[polys.len() - 1]
                .get_base_field_vec()
                .iter()
                .map(|&eval| {
                    eval + E::BaseField::from_canonical_u64(1 << (*quantization::BIT_LEN - 1))
                })
                .collect::<Vec<E::BaseField>>();
            column_evals.push(value_evals);

            let (constant_challenge, column_separation_challenge) = challenge_storage
                .get_challenges_by_name(&self.tables[0].name())
                .ok_or(anyhow!(
                    "No challenges found for Table {}, cannot generate Softmax LogUp input",
                    self.tables[0].name()
                ))?;
            let logup_input = LogUpInput::<E>::new_lookup(
                column_evals,
                constant_challenge,
                column_separation_challenge,
                1,
            )?;
            Ok(vec![logup_input])
        } else {
            let number_zero_chunks = self.instances_per_table[1];
            // subtract one for the sign poly, one for value and then twice the number of shifted chunks
            let shifted_chunks = polys.len() - 2 - 2 * number_zero_chunks;

            let mut range_column_evals = polys[..shifted_chunks]
                .iter()
                .map(|p| p.get_base_field_vec().to_vec())
                .collect::<Vec<Vec<E::BaseField>>>();
            let value_evals = polys[shifted_chunks + number_zero_chunks]
                .get_base_field_vec()
                .iter()
                .map(|&eval| {
                    eval + E::BaseField::from_canonical_u64(1 << (*quantization::BIT_LEN - 1))
                })
                .collect::<Vec<E::BaseField>>();
            range_column_evals.push(value_evals);
            let (constant_challenge, column_separation_challenge) = challenge_storage
                .get_challenges_by_name(&self.tables[0].name())
                .ok_or(anyhow!(
                    "No challenges found for Table {}, cannot generate Softmax LogUp input",
                    self.tables[0].name()
                ))?;
            let range_input = LogUpInput::<E>::new_lookup(
                range_column_evals,
                constant_challenge,
                column_separation_challenge,
                1,
            )?;

            let zero_column_evals = polys
                .iter()
                .skip(shifted_chunks)
                .take(number_zero_chunks)
                .interleave(
                    polys
                        .iter()
                        .skip(shifted_chunks + number_zero_chunks + 1)
                        .take(number_zero_chunks),
                )
                .map(|p| p.get_base_field_vec().to_vec())
                .collect::<Vec<Vec<E::BaseField>>>();
            let (constant_challenge, column_separation_challenge) = challenge_storage
                .get_challenges_by_name(&self.tables[1].name())
                .ok_or(anyhow!(
                    "No challenges found for Table {}, cannot generate Softmax LogUp input",
                    self.tables[1].name()
                ))?;
            let zero_input = LogUpInput::<E>::new_lookup(
                zero_column_evals,
                constant_challenge,
                column_separation_challenge,
                2,
            )?;
            Ok(vec![range_input, zero_input])
        }
    }
}

impl PadOp for Requant {}

impl<E, PCS> ProvableOp<E, PCS> for Requant
where
    E: ExtensionField + Serialize + DeserializeOwned,
    E::BaseField: Serialize + DeserializeOwned,
    PCS: PolynomialCommitmentScheme<E> + Send + Sync,
    PCS::CommitmentWithWitness: Serialize + DeserializeOwned + Send + Sync,
    PCS::ProverParam: Send + Sync,
{
    type Ctx = RequantCtx<E>;

    fn prove<T: Transcript<E>>(
        &self,
        id: NodeId,
        ctx: &Self::Ctx,
        last_claims: Vec<&Claim<E>>,
        _step_data: &StepData<E, E>,
        prover: &mut Prover<E, T, PCS>,
        _store: &mut TenStore,
    ) -> Result<Vec<Claim<E>>> {
        let claim = self.prove_step(prover, last_claims[0], ctx, id)?;

        Ok(vec![claim])
    }

    fn gen_lookup_witness(
        &self,
        id: NodeId,
        ctx: &ProverContext<E, PCS>,
        step_data: &StepData<Element, E>,
        store: &mut TenStore,
    ) -> Result<LookupWitnessGen<E, PCS>> {
        let outputs = step_data.output_tensors(store)?;
        ensure!(
            step_data.node_inputs.len() == 1,
            "Found more than 1 input in inference step of requant layer"
        );
        ensure!(
            outputs.len() == 1,
            "Found more than 1 output in inference step of requant layer"
        );

        // We take the input, multiply by the fixed point multiplier and add the rounding constant. Then we split the resulting values into
        // parts that are either shifted away (these get range checked) or parts that contribute to the output.

        // Parts that contribute to the output are the bits remaining after applying the shift, we chunk this part into *quantization::BIT_LEN size chunks.
        // We work out how many bits will be left using the formula `remaining_bits = self.intermediate_bit_size + ceil_log2(self.fixed_point_multiplier as usize) - self.shift();`
        // Then we know that there are `most_sig_chunks = (remaining_bits - 1) / *quantization::BIT_LEN + 1` chunks that contribute to the value.
        let shift = self.shift();
        let rounding_constant: Element = 1 << (shift - 1);
        let mask: Element = (1 << shift) - 1;
        let bit_len_mask: Element = (1 << *quantization::BIT_LEN) - 1;
        // First let us multiply by the fixed point multiplier, add the rounding constant and then split into the part that is shifted away and the part
        // that contributes to the output.
        let ((output_part, sign), shifted_part): ((Vec<Element>, Vec<Element>), Vec<Element>) =
            step_data.node_inputs[0]
                .hydrate(store.clone())
                .context("hydrating tensor")?
                .get_data()
                .iter()
                .map(|&val| {
                    let tmp = val * self.fixed_point_multiplier + rounding_constant;
                    let sign = if tmp >= 0 { 0 } else { -1 };
                    let output = tmp >> shift;
                    let masked = tmp & mask;
                    ((output, sign), masked)
                })
                .unzip();
        // Now we chunk the shifted part into *quantization::BIT_LEN size chunks, for the final chunk we work out how many bits its actually using
        // and then multiply by an appropriate scalar to enforce that the chunk is in the correct smaller range. For example if `shift = 11` and `*qunatization::BIT_LEN = 8`
        // the final chunk should actually only be 3 bit numbers (so between 0 and 7). To make sure the prover doesn't cheat we multiply everything in this chunk by 2^5
        // because if it was in the correct range to begin with after multiplying by 2^5 it will be in the range 0..255, but if it was larger than 7 after multiplying by 2^5
        // it will no longer be in the range 0..255.
        let (number_shift_chunks, _, final_shift_chunk_multiplier) = self.shifted_chunks_data();
        let mut shifted_chunks = (0..number_shift_chunks)
            .map(|j| {
                if j != number_shift_chunks - 1 {
                    shifted_part
                        .par_iter()
                        .map(|v| (*v >> (j * *quantization::BIT_LEN)) & bit_len_mask)
                        .collect::<Vec<Element>>()
                } else {
                    shifted_part
                        .par_iter()
                        .map(|v| {
                            ((*v >> (j * *quantization::BIT_LEN)) & bit_len_mask)
                                * final_shift_chunk_multiplier
                        })
                        .collect::<Vec<Element>>()
                }
            })
            .collect::<Vec<Vec<Element>>>();
        let number_output_chunks = self.number_of_zero_chunks() + 1;
        let mut chunks: Vec<Vec<Element>> = vec![vec![]; number_output_chunks];

        output_part.into_iter().for_each(|mut value| {
            chunks.iter_mut().enumerate().for_each(|(i, c)| {
                let shift = i * *quantization::BIT_LEN;
                let byte = ((value >> shift) & bit_len_mask) as u8;
                let signed_byte = byte as i8;
                let limb = signed_byte as Element;

                value -= limb << shift;
                c.push(limb);
            })
        });

        let value_chunk = chunks.remove(0);
        let (zero_ins, zero_outs): (Vec<Vec<Element>>, Vec<Vec<Element>>) = chunks
            .into_iter()
            .map(|z_in| {
                let z_out = z_in
                    .iter()
                    .map(|&v| if v == 0 { 1 } else { 0 })
                    .collect::<Vec<Element>>();
                (z_in, z_out)
            })
            .unzip();
        // Make the multiplicity counts for each of the lookups
        let range_check_count = count_elements(
            shifted_chunks.iter().flatten().copied().chain(
                value_chunk
                    .iter()
                    .map(|val| val + (1 << (*quantization::BIT_LEN - 1))),
            ),
        );

        let zero_check_count = count_elements(zero_ins.iter().zip(zero_outs.iter()).flat_map(
            |(z_in, z_out)| {
                z_in.iter()
                    .zip(z_out.iter())
                    .map(|(&i, &o)| i + COLUMN_SEPARATOR * o)
                    .collect::<Vec<Element>>()
            },
        ));

        // Make the commitments
        // We group them by (shifted_parts, zero_ins) and (value, zero_outs, sign) because they will be evaluated at different points
        shifted_chunks.extend(zero_ins);
        let width1 = shifted_chunks.len();
        let values_1 = transpose(shifted_chunks);

        let (width2, values_2) = if self.number_of_zero_chunks() != 0 {
            let chained = [value_chunk]
                .into_iter()
                .chain(zero_outs)
                .chain([sign])
                .collect::<Vec<Vec<Element>>>();
            (chained.len(), transpose(chained))
        } else {
            (1, vec![value_chunk])
        };

        let rmm1 = RowMajorMatrix::<E::BaseField>::new_by_inner_matrix(
            ceno_p3::matrix::dense::DenseMatrix::new(to_base::<E, _>(values_1.concat()), width1),
            witness::InstancePaddingStrategy::Default,
        );
        let rmm2 = RowMajorMatrix::<E::BaseField>::new_by_inner_matrix(
            ceno_p3::matrix::dense::DenseMatrix::new(to_base::<E, _>(values_2.concat()), width2),
            witness::InstancePaddingStrategy::Default,
        );

        let layer_commitment = ctx.commitment_ctx.batch_commit(vec![rmm1, rmm2])?;

        let mut gen = LookupWitnessGen::<E, PCS>::default();

        if !zero_check_count.is_empty() {
            gen.insert_element_count(TableType::RequantZeroTable, zero_check_count);
        }

        gen.insert_element_count(TableType::Range, range_check_count);

        gen.insert_logup_witness(id, layer_commitment);
        Ok(gen)
    }
}

impl<E: ExtensionField> OpInfo for RequantCtx<E> {
    fn output_shapes(&self, input_shapes: &[Shape], _padding_mode: PaddingMode) -> Vec<Shape> {
        input_shapes.to_vec()
    }

    fn num_outputs(&self, num_inputs: usize) -> usize {
        Requant::num_outputs(&self.requant, num_inputs)
    }

    fn describe(&self) -> String {
        format!(
            "Requant ctx: fixed point multiplier: {}, right shift: {}",
            self.requant.fixed_point_multiplier,
            self.requant.shift(),
        )
    }

    fn is_provable(&self) -> bool {
        IS_PROVABLE
    }
}

impl<E, PCS> VerifiableCtx<E, PCS> for RequantCtx<E>
where
    E: ExtensionField + Serialize + DeserializeOwned,
    E::BaseField: Serialize + DeserializeOwned,
    PCS: PolynomialCommitmentScheme<E>,
{
    type Proof = RequantProof<E, PCS>;

    fn verify<T: Transcript<E>>(
        &self,
        proof: &Self::Proof,
        last_claims: &[&Claim<E>],
        verifier: &mut Verifier<E, T, PCS>,
        _shape_step: &ShapeStep,
    ) -> Result<Vec<Claim<E>>> {
        let claim = self.verify_requant(verifier, last_claims[0], proof)?;

        Ok(vec![claim])
    }

    fn write_proof_to_transcript<T: Transcript<E>>(
        &self,
        proof: &Self::Proof,
        transcript: &mut T,
    ) -> anyhow::Result<()> {
        proof.write_commitment(transcript)
    }
}

impl Requant {
    /// Method used to instantiate a new [`Requant`] from the multiplier employed to requantize the layer.
    /// The `intermediate_bit_size` is layer dependant and so should be passed as input. It can be calculated based on how many times you need to multiply and add
    /// to get each value in the output tensor.
    pub(crate) fn from_multiplier(multiplier: f32, intermediate_bit_size: usize) -> Requant {
        let log_m = multiplier.log2();
        // This is the right shift
        let int_part = log_m.trunc().abs() as usize;
        // This is used to calculate the fixed point multiplier
        let float_part = log_m.fract();

        let epsilon = 2.0f32.powf(float_part);

        let fp_scale = FIXED_POINT_SCALE;
        let fixed_point_multiplier =
            (epsilon * (1u64 << FIXED_POINT_SCALE) as f32).round() as Element;

        // Assertion to check that we can perform requantisation, we need intermediate_bit_size + fp_scale <= 63
        assert!(
            intermediate_bit_size + fp_scale <= 63,
            "intermediate bit size: {intermediate_bit_size}, fp scale: {fp_scale}, int part: {int_part}",
        );
        Requant {
            right_shift: int_part,
            fixed_point_multiplier,
            fp_scale,
            multiplier,
            intermediate_bit_size,
        }
    }
    /// Method used to instantiate a new [`Requant`] from the scaling factors of all tensors involved in a layer.
    /// The `intermediate_bit_size` is layer dependant and so should be passed as input. It can be calculated based on how many times you need to multiply and add
    /// to get each value in the output tensor.
    pub fn from_scaling_factors(
        input_scale: ScalingFactor,
        weights_scale: ScalingFactor,
        output_scale: ScalingFactor,
        intermediate_bit_size: usize,
    ) -> Requant {
        let m = input_scale.m(&weights_scale, &output_scale);
        Self::from_multiplier(m, intermediate_bit_size)
    }

    /// This returns the shift (including the part that depends on `S1 * S2/ S3`)
    pub(crate) fn shift(&self) -> usize {
        self.fp_scale + self.right_shift
    }

    /// This method calculates how many chunks to split the shifted away part into, the number of bits in the most
    /// significant of these chunks and what to multiply the most significant chunk by in order for the range check to
    /// enforce the correct range
    pub(crate) fn shifted_chunks_data(&self) -> (usize, usize, Element) {
        let shift = self.shift();
        let number_shift_chunks = (shift - 1) / *quantization::BIT_LEN + 1;
        let final_shift_chunk_bits = shift % *quantization::BIT_LEN;
        let final_chunk_multiplier: Element = if final_shift_chunk_bits == 0 {
            1
        } else {
            1 << (*quantization::BIT_LEN - final_shift_chunk_bits)
        };
        (
            number_shift_chunks,
            final_shift_chunk_bits,
            final_chunk_multiplier,
        )
    }

    /// Internal method that applies this op to an [`Element`]
    fn apply(&self, elem: &Element) -> Element {
        let rounding: Element = 1 << (self.shift() - 1);
        let unclamped = (rounding + elem * self.fixed_point_multiplier) >> self.shift();

        if unclamped >= *quantization::MAX {
            *quantization::MAX
        } else if unclamped <= *quantization::MIN {
            *quantization::MIN
        } else {
            unclamped
        }
    }

    /// API for performing this op on a quantised tensor.
    pub fn op(&self, input: &Tensor<Element>) -> Result<Tensor<Element>> {
        // We use this value to determine if any of the inputs are too large to be requantised (i.e. they fall outside the clamping table)
        let max_abs_val: Element = 1 << self.intermediate_bit_size;
        let res = input
            .get_data()
            .iter().enumerate()
            .map(|(i,e)| {if e.abs() <= max_abs_val {Ok(self.apply(e))} else {Err(anyhow!("Could not apply requantisation, tensor element {} had absolute value too large, given value: {}, max value: {}", i, e, max_abs_val))}})
            .collect::<Result<Vec<Element>, anyhow::Error>>()?;

        Ok(Tensor::<Element>::new(input.shape(), res))
    }

    /// Function that tells us how many bits are not shifted away
    pub(crate) fn output_bit_size(&self) -> usize {
        let fpm_bit_size = ceil_log2(self.fixed_point_multiplier as usize);
        self.intermediate_bit_size + fpm_bit_size - self.shift()
    }

    /// Function that returns how many zero-chunks the [`Requant`] contains
    pub(crate) fn number_of_zero_chunks(&self) -> usize {
        (self.output_bit_size() - 1) / *quantization::BIT_LEN
    }

    pub fn write_to_transcript<E: ExtensionField, T: Transcript<E>>(&self, t: &mut T) {
        t.append_field_element(&E::BaseField::from_canonical_u64(self.right_shift as u64));
        t.append_field_element(&E::BaseField::from_canonical_u64(
            self.fixed_point_multiplier as u64,
        ));
    }

    pub fn recombine_claims<E: ExtensionField>(
        &self,
        value_claim: E,
        shifted_claims: &[E],
        zero_in_claims: &[E],
    ) -> E {
        let (number_shift_eval, _, top_shift_mult) = self.shifted_chunks_data();
        let top_shift_mult_field: E = top_shift_mult.to_field();
        let top_shift_mult_inv = top_shift_mult_field.inverse();
        // First we recombine the shifted claims
        let shifted_part = shifted_claims
            .iter()
            .enumerate()
            .fold(E::ZERO, |acc, (j, &eval)| {
                if j != number_shift_eval - 1 {
                    acc + eval * E::from_canonical_u64(1 << (j * *quantization::BIT_LEN))
                } else {
                    acc + eval
                        * E::from_canonical_u64(1 << (j * *quantization::BIT_LEN))
                        * top_shift_mult_inv
                }
            });

        // Now we get the shift size
        let shift = self.shift();

        let full_eval = std::iter::once(&value_claim)
            .chain(zero_in_claims)
            .enumerate()
            .fold(shifted_part, |acc, (j, &eval)| {
                acc + eval * E::from_canonical_u64(1 << (shift + j * *quantization::BIT_LEN))
            });

        // Now subtract the rounding constant and multiply by the inverse of fixed point multiplier
        let rounding_constant: Element = 1 << (shift - 1);
        let rc_field: E = rounding_constant.to_field();
        let fpm_field: E = self.fixed_point_multiplier.to_field();
        let fpm_inv = fpm_field.inverse();
        (full_eval - rc_field) * fpm_inv
    }

    #[timed::timed_instrument(name = "Prover::prove_requant")]
    /// Method that proves requantisation was performed correctly. It does this by running any required lookups and then linking the `last_claim` to the
    /// `input` via a series of Sumchecks.
    pub(crate) fn prove_step<E, T: Transcript<E>, PCS>(
        &self,
        prover: &mut Prover<E, T, PCS>,
        last_claim: &Claim<E>,
        ctx: &RequantCtx<E>,
        id: NodeId,
    ) -> anyhow::Result<Claim<E>>
    where
        E: ExtensionField + Serialize + DeserializeOwned,
        E::BaseField: Serialize + DeserializeOwned,
        PCS: PolynomialCommitmentScheme<E> + Send + Sync,
        PCS::CommitmentWithWitness: Serialize + DeserializeOwned + Send + Sync,
        PCS::ProverParam: Send + Sync,
    {
        let layer_commitment = prover.lookup_witness(id)?;
        let logup_inputs = ctx
            .lookup_ctx
            .create_logup_inputs_requant::<PCS, E>(layer_commitment, &prover.challenge_storage)?;
        let layer_polys = PCS::get_arc_mle_witness_from_commitment(layer_commitment);
        let commitment = PCS::get_pure_commitment(layer_commitment);
        // Run the logup proving
        let logup_batch_proof = batch_multiple_sizes_prove(&logup_inputs, prover.transcript)?;
        let logup_point = logup_batch_proof.output_claims()[0].point.clone();

        let eq_mles = if self.number_of_zero_chunks() != 0 {
            let sign_challenges = (0..last_claim.point.len())
                .map(|_| {
                    prover
                        .transcript
                        .sample_and_append_challenge(b"sign")
                        .elements
                })
                .collect::<Vec<E>>();
            let last_claim_eq = compute_betas_eval(&last_claim.point).into_mle();
            let sign_eq = compute_betas_eval(&sign_challenges).into_mle();
            let logup_eq = compute_betas_eval(&logup_point).into_mle();

            vec![last_claim_eq, sign_eq, logup_eq]
        } else {
            let last_claim_eq = compute_betas_eval(&last_claim.point).into_mle();
            let logup_eq = compute_betas_eval(&logup_point).into_mle();
            vec![last_claim_eq, logup_eq]
        };

        let challenge = prover
            .transcript
            .sample_and_append_challenge(b"batching")
            .elements;
        let (number_shifted_chunks, _, _) = self.shifted_chunks_data();
        let either_mles = layer_polys
            .iter()
            .skip(number_shifted_chunks + self.number_of_zero_chunks())
            .map(|p| Either::Left(p.as_ref()))
            .chain(eq_mles.iter().map(Either::Left))
            .collect::<Vec<Either<_, _>>>();

        let num_vars = last_claim.point.len();
        let num_threads = optimal_sumcheck_threads(num_vars);
        let expr_builder =
            VirtualPolynomialsBuilder::<E>::new_with_mles(num_threads, num_vars, either_mles);
        let virtual_poly = expr_builder
            .to_virtual_polys(std::slice::from_ref(&ctx.sumcheck_expression), &[challenge]);

        let (claim_acc_proof, state) = IOPProverState::<E>::prove(virtual_poly, prover.transcript);
        let evals = state.get_mle_flatten_final_evaluations();
        let evaluations = if self.number_of_zero_chunks() != 0 {
            evals[..evals.len() - 3].to_vec()
        } else {
            evals[..evals.len() - 2].to_vec()
        };

        let point = state.collect_raw_challenges();

        // Now we calculate the claim about the input from the shifted chunks, zero_in evals, value eval and sign eval
        let logup_claims = logup_batch_proof.output_claims();

        let shifted_claims = logup_claims
            .iter()
            .take(number_shifted_chunks)
            .map(|c| c.eval)
            .collect::<Vec<E>>();
        let zero_in_evals = logup_claims
            .iter()
            .skip(number_shifted_chunks + 1)
            .step_by(2)
            .take(self.number_of_zero_chunks())
            .map(|c| c.eval)
            .collect::<Vec<E>>();
        let logup_value_eval = logup_claims[number_shifted_chunks].eval
            - E::from_canonical_u64(1 << (*quantization::BIT_LEN - 1));
        let input_eval = self.recombine_claims(logup_value_eval, &shifted_claims, &zero_in_evals);
        let input_claim = Claim::<E>::new(logup_point.clone(), input_eval);

        // Add all the commitments to the commitment prover
        let first_commit = (logup_point, [shifted_claims, zero_in_evals].concat());
        let second_commit = (point, evaluations.clone());

        prover.add_witness_claim(id, vec![first_commit, second_commit]);

        // Add the layer proof to the list
        prover.push_proof(
            id,
            LayerProof::Requant(RequantProof {
                io_accumulation: claim_acc_proof,
                io_eval: evaluations,
                logup_proof: logup_batch_proof,
                commitment,
            }),
        );

        Ok(input_claim)
    }
}

impl<E: ExtensionField> RequantCtx<E> {
    /// Method that verifies requantisation has been performed correctly when supplied with a [`RequantProof`].
    /// It verifies both lookup argument proofs, calculates the initial claim for the sumcheck proof using the lookup argument claims
    /// and then verifies the sumcheck using this initial claim. It then takes the output claims provided by the prover, checks they relate to the sumcheck
    /// subclaim, adds them to the list of claims of commitment openings and then calculates the next claim.
    pub(crate) fn verify_requant<T: Transcript<E>, PCS: PolynomialCommitmentScheme<E>>(
        &self,
        verifier: &mut Verifier<E, T, PCS>,
        last_claim: &Claim<E>,
        proof: &RequantProof<E, PCS>,
    ) -> anyhow::Result<Claim<E>>
    where
        E: ExtensionField + Serialize + DeserializeOwned,
        E::BaseField: Serialize + DeserializeOwned,
    {
        // 1. Verify the lookup proofs
        let RequantProof {
            io_accumulation,
            logup_proof,
            io_eval,
            commitment,
        } = proof;

        let batch_claim = verify_logup_proof_multiple_sizes(logup_proof, verifier.transcript)?;
        self.lookup_ctx
            .verify_logup_batch_claim(&batch_claim, &verifier.challenge_storage)?;

        let poly_evals = batch_claim.poly_evals();

        // Calculate the claimed sum for the sumcheck proof
        let (shifted_instances, _, _) = self.requant.shifted_chunks_data();

        let sign_challenges = if self.requant.number_of_zero_chunks() != 0 {
            (0..last_claim.point.len())
                .map(|_| {
                    verifier
                        .transcript
                        .sample_and_append_challenge(b"sign")
                        .elements
                })
                .collect::<Vec<E>>()
        } else {
            vec![]
        };

        let challenge = verifier
            .transcript
            .sample_and_append_challenge(b"batching")
            .elements;

        let challenge_cubed = challenge * challenge * challenge;
        let logup_value_eval = poly_evals[shifted_instances];
        let fold_initial = challenge * logup_value_eval;
        let logup_claim_part = poly_evals
            .iter()
            .skip(shifted_instances + 2)
            .step_by(2)
            .fold((fold_initial, challenge_cubed), |(acc, chal_acc), &eval| {
                (acc + eval * chal_acc, chal_acc * challenge)
            })
            .0;
        let claimed_sum = last_claim.eval + logup_claim_part;
        let aux_info = VPAuxInfo {
            max_num_variables: last_claim.point.len(),
            max_degree: 2 + self.requant.number_of_zero_chunks(),
            ..Default::default()
        };

        let subclaim = IOPVerifierState::<E>::verify(
            claimed_sum,
            io_accumulation,
            &aux_info,
            verifier.transcript,
        );
        let point = subclaim
            .point
            .iter()
            .map(|c| c.elements)
            .collect::<Vec<E>>();
        let last_claim_eq = identity_eval(&last_claim.point, &point);
        let logup_eq = identity_eval(batch_claim.point(), &point);

        let calc_claim = if self.requant.number_of_zero_chunks() != 0 {
            let sign_eq = identity_eval(&sign_challenges, &point);
            let evals = io_eval
                .iter()
                .copied()
                .chain([last_claim_eq, sign_eq, logup_eq])
                .collect::<Vec<E>>();
            eval_by_expr_with_instance(
                &[],
                &evals,
                &[],
                &[],
                &[challenge],
                &self.sumcheck_expression,
            )
            .right()
            .ok_or(anyhow!(
                "Couldn't verify Requant, calculated claim was not an ExtensionField element"
            ))?
        } else {
            let evals = io_eval
                .iter()
                .copied()
                .chain([last_claim_eq, logup_eq])
                .collect::<Vec<E>>();
            eval_by_expr_with_instance(
                &[],
                &evals,
                &[],
                &[],
                &[challenge],
                &self.sumcheck_expression,
            )
            .right()
            .ok_or(anyhow!(
                "Couldn't verify Requant, calculated claim was not an ExtensionField element"
            ))?
        };

        ensure!(
            calc_claim == subclaim.expected_evaluation,
            "Requant Verification failed, calculated claim {:?}, did not equal the expected evaluation {:?}",
            calc_claim,
            subclaim.expected_evaluation
        );

        // Make the input claim
        let shifted_claims = poly_evals
            .iter()
            .take(shifted_instances)
            .copied()
            .collect::<Vec<E>>();
        let zero_in_evals = poly_evals
            .iter()
            .skip(shifted_instances + 1)
            .step_by(2)
            .take(self.requant.number_of_zero_chunks())
            .copied()
            .collect::<Vec<E>>();
        let logup_value_eval = poly_evals[shifted_instances]
            - E::from_canonical_u64(1 << (*quantization::BIT_LEN - 1));
        let input_eval =
            self.requant
                .recombine_claims(logup_value_eval, &shifted_claims, &zero_in_evals);
        let input_claim = Claim::<E>::new(batch_claim.point().to_vec(), input_eval);

        let first_commit = (
            batch_claim.point().to_vec(),
            [shifted_claims, zero_in_evals].concat(),
        );
        let second_commit = (point, io_eval.clone());

        verifier.commit_verifier.add_witness_claim(
            self.node_id,
            commitment.clone(),
            vec![first_commit, second_commit],
        );

        // Pass the input claim
        Ok(input_claim)
    }
}

#[cfg(test)]
mod tests {

    use crate::{
        layers::{Layer, matrix_mul::MatMul},
        model::{Model, test::prove_model},
    };

    use super::*;

    #[test]
    fn test_requant_proving() {
        // To test requant proving we make a simple model with a matmul
        let [a, b, d] = [10, 20, 256];
        let first_input_shape = vec![a, b];
        let matrix_shape: Shape = vec![b, d].into();
        let mut model =
            Model::new_from_input_shapes(vec![first_input_shape.into()], PaddingMode::NoPadding);

        let mat = Tensor::<f32>::random(&matrix_shape);
        let bias = Tensor::<f32>::random(&vec![d].into());
        let matmul = MatMul::new_constant(mat, Some(bias)).unwrap();
        let _ = model
            .add_consecutive_layer(Layer::MatMul(matmul), None)
            .unwrap();
        model.route_output(None).unwrap();
        model.describe();
        prove_model(model, &mut TenStore::default()).unwrap();
    }
}
