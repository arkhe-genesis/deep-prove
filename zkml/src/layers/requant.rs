//! Module containign code for performing proving friendly requantisation. This is done via a [fixed point multiplication](https://en.wikipedia.org/wiki/Fixed-point_arithmetic#Binary_fixed-point_multiplication) and use of lookup arguments.

use crate::{
    Claim, Element, Prover, ProverContext, ScalingFactor, Tensor,
    commit::compute_betas_eval,
    iop::{
        context::{ContextAux, ShapeStep},
        verifier::Verifier,
    },
    layers::LayerProof,
    lookup::{
        context::{COLUMN_SEPARATOR, CommsAndEvals, LookupWitnessGen, TableType, count_elements},
        logup_gkr::{
            prover::batch_prove as logup_batch_prove, structs::LogUpProof,
            verifier::verify_logup_proof,
        },
        witness::LogUpWitness,
    },
    model::StepData,
    padding::PaddingMode,
    quantization::{self, Fieldizer},
    tensor::Shape,
    to_base,
};
use anyhow::{Context as CC, Result, anyhow, ensure};
use either::Either;
use ff_ext::ExtensionField;
use itertools::Itertools;
use multilinear_extensions::{
    Expression, util::ceil_log2, virtual_polys::VirtualPolynomialsBuilder,
};
use p3_field::FieldAlgebra;

use mpcs::{PolynomialCommitmentScheme, sum_check::eq_xy_eval};
use multilinear_extensions::mle::{IntoMLE, MultilinearExtension};

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
pub struct RequantCtx {
    pub requant: Requant,
    pub node_id: NodeId,
    pub num_vars: usize,
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
    pub(crate) accumulation_evals: Vec<E>,
    /// The lookup proofs, there are one or two depending on if zero checks are required
    pub(crate) lookup_proofs: Vec<LogUpProof<E>>,
    /// COmmitments to lookup polynomials, they are in the order clamping commitments -> shifted commitments
    pub(crate) commitments: Vec<PCS::Commitment>,
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
        ensure!(
            inputs.len() == 1,
            "Requant layer expects 1 input, got {}",
            inputs.len()
        );
        Ok(LayerOut::from_vec(
            inputs
                .iter()
                .map(|input| self.op(input))
                .collect::<Result<Vec<_>>>()?,
        ))
    }
}

impl<E> ProveInfo<E> for Requant
where
    E: ExtensionField + DeserializeOwned,
    E::BaseField: Serialize + DeserializeOwned,
{
    fn step_info(&self, id: NodeId, mut aux: ContextAux) -> Result<(LayerCtx<E>, ContextAux)> {
        aux.tables.insert(TableType::Range);

        // Add ZeroTable to the aux if needed
        if self.number_of_zero_chunks() != 0 {
            aux.tables.insert(TableType::RequantZeroTable);
        }

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
            }),
            aux,
        ))
    }
}

impl PadOp for Requant {}

impl<E, PCS> ProvableOp<E, PCS> for Requant
where
    E: ExtensionField + Serialize + DeserializeOwned,
    E::BaseField: Serialize + DeserializeOwned,
    PCS: PolynomialCommitmentScheme<E>,
{
    type Ctx = RequantCtx;

    fn prove<T: Transcript<E>>(
        &self,
        id: NodeId,
        ctx: &Self::Ctx,
        last_claims: Vec<&Claim<E>>,
        _step_data: &StepData<E, E>,
        prover: &mut Prover<E, T, PCS>,
        _store: &mut TenStore,
    ) -> Result<Vec<Claim<E>>> {
        let claim = match self.number_of_zero_chunks() {
            0 => self.prove_step_no_zero_chunks(prover, last_claims[0], ctx, id)?,
            _ => self.prove_step(prover, last_claims[0], ctx, id)?,
        };
        Ok(vec![claim])
    }

    fn gen_lookup_witness<'a>(
        &self,
        id: NodeId,
        ctx: &'a ProverContext<'a, E, PCS>,
        step_data: &StepData<Element, E>,
        store: &mut TenStore,
    ) -> Result<LookupWitnessGen<'a, E, PCS>> {
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
        let shifted_chunks = (0..number_shift_chunks)
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

        // Make the multiplicity counts for each of the lookups
        let range_check_count = count_elements(
            shifted_chunks.iter().flatten().copied().chain(
                value_chunk
                    .iter()
                    .map(|val| val + (1 << (*quantization::BIT_LEN - 1))),
            ),
        );

        // Make the commitments
        let num_vars = ceil_log2(value_chunk.len());

        let (mut range_commits, mut range_evals): CommsAndEvals<PCS, E> = shifted_chunks
            .into_par_iter()
            .map(|chunk| {
                let evaluations = to_base::<E, _>(chunk);
                let mle =
                    MultilinearExtension::<E>::from_evaluations_vec(num_vars, evaluations.clone());
                let commit = ctx.commitment_ctx.commit(&mle)?;
                Ok(((commit, mle), evaluations))
            })
            .collect::<Result<Vec<_>, anyhow::Error>>()?
            .into_iter()
            .unzip();

        let (value_poly_evals, value_evals): (Vec<E::BaseField>, Vec<E::BaseField>) = value_chunk
            .into_iter()
            .map(|v| {
                let poly_f: E = v.to_field();
                let f: E = (v + (1 << (*quantization::BIT_LEN - 1))).to_field();
                (poly_f.as_bases()[0], f.as_bases()[0])
            })
            .unzip();

        let value_mle = MultilinearExtension::from_evaluations_vec(num_vars, value_poly_evals);
        let value_commit = ctx.commitment_ctx.commit(&value_mle)?;

        range_commits.insert(0, (value_commit, value_mle));
        range_evals.insert(0, value_evals);

        let mut gen = LookupWitnessGen::<E, PCS>::default();

        if number_output_chunks > 1 {
            // This means there are chunks that need to be zero checked

            let zero_check_count = count_elements(chunks.iter().flat_map(|a_vec| {
                a_vec
                    .iter()
                    .map(|&a| if a != 0 { a } else { a + COLUMN_SEPARATOR })
            }));

            let (mut zero_commits, zero_evals): CommsAndEvals<PCS, E> = chunks
                .into_iter()
                .flat_map(|in_vec| {
                    let out_vec = in_vec
                        .iter()
                        .map(|&v| if v == 0 { 1 } else { 0 })
                        .collect::<Vec<Element>>();
                    [in_vec, out_vec]
                })
                .collect::<Vec<Vec<Element>>>()
                .into_par_iter()
                .map(|vals| {
                    let (commit_evals, evaluations): (Vec<E::BaseField>, Vec<E::BaseField>) = vals
                        .into_iter()
                        .map(|v| {
                            let f: E = v.to_field();
                            (f.as_bases()[0], f.as_bases()[0])
                        })
                        .unzip();
                    let mle =
                        MultilinearExtension::<E>::from_evaluations_vec(num_vars, commit_evals);
                    let commit = ctx.commitment_ctx.commit(&mle)?;
                    Ok(((commit, mle), evaluations))
                })
                .collect::<Result<Vec<_>, anyhow::Error>>()?
                .into_iter()
                .unzip();

            let sign_mle = MultilinearExtension::<E>::from_evaluations_vec(
                num_vars,
                sign.into_iter()
                    .map(|v| {
                        let f: E = v.to_field();
                        f.as_bases()[0]
                    })
                    .collect::<Vec<E::BaseField>>(),
            );
            let sign_commit = ctx.commitment_ctx.commit(&sign_mle)?;

            zero_commits.push((sign_commit, sign_mle));
            // Insert both logup witnesses
            gen.logup_witnesses.insert(
                id,
                vec![
                    LogUpWitness::<E, PCS>::new_lookup(
                        zero_commits,
                        zero_evals,
                        2,
                        TableType::RequantZeroTable,
                    ),
                    LogUpWitness::<E, PCS>::new_lookup(
                        range_commits,
                        range_evals,
                        1,
                        TableType::Range,
                    ),
                ],
            );

            gen.element_count
                .insert(TableType::RequantZeroTable, zero_check_count);
        } else {
            // No zero chunks
            // Only insert the range checks
            gen.logup_witnesses.insert(
                id,
                vec![LogUpWitness::<E, PCS>::new_lookup(
                    range_commits,
                    range_evals,
                    1,
                    TableType::Range,
                )],
            );
        }

        gen.element_count
            .insert(TableType::Range, range_check_count);

        Ok(gen)
    }
}

impl OpInfo for RequantCtx {
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

impl<E, PCS> VerifiableCtx<E, PCS> for RequantCtx
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
        let claim = match self.requant.number_of_zero_chunks() {
            0 => {
                let (constant_challenge, _) = verifier
                    .challenge_storage
                    .get_challenges_by_name(&TableType::Range.name())
                    .ok_or(anyhow!(
                        "Couldn't get challenges for LookupType: {}",
                        TableType::Range.name()
                    ))?;

                self.verify_requant_no_zero_chunks(
                    verifier,
                    last_claims[0],
                    proof,
                    constant_challenge,
                )
            }
            _ => {
                let (constant_challenge, column_separation_challenge) = verifier
                    .challenge_storage
                    .get_challenges_by_name(&TableType::RequantZeroTable.name())
                    .ok_or(anyhow!(
                        "Couldn't get challenges for LookupType: {}",
                        TableType::RequantZeroTable.name()
                    ))?;
                self.verify_requant(
                    verifier,
                    last_claims[0],
                    proof,
                    constant_challenge,
                    column_separation_challenge,
                )
            }
        };

        Ok(vec![claim?])
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
            .map(|(i,e)| {if e.abs() <= max_abs_val {Ok(self.apply(e))} else {Err(anyhow!("Could not apply requantisation, tensor element {} had absoloute value too large, given value: {}, max value: {}", i, e, max_abs_val))}})
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

    pub(crate) fn prove_step_no_zero_chunks<
        E,
        T: Transcript<E>,
        PCS: PolynomialCommitmentScheme<E>,
    >(
        &self,
        prover: &mut Prover<E, T, PCS>,
        last_claim: &Claim<E>,
        _requant_info: &RequantCtx,
        id: NodeId,
    ) -> anyhow::Result<Claim<E>>
    where
        E: ExtensionField + Serialize + DeserializeOwned,
        E::BaseField: Serialize + DeserializeOwned,
    {
        let mut logup_witnesses = prover.lookup_witness(id)?;
        // Check that we have one witness for requantisation
        if logup_witnesses.len() != 1 {
            return Err(anyhow!(
                "There should only be one lookup witness during requantisation, node: {}, number of witnesses: {}",
                id,
                logup_witnesses.len()
            ));
        }
        // Run the lookup protocol and return the lookup proof
        let range_logup_witness = logup_witnesses.remove(0);
        // Run the lookup protocol and return the lookup proofs
        let range_prover_info = range_logup_witness.get_logup_input(&prover.challenge_storage)?;
        let range_logup_proof = logup_batch_prove(&range_prover_info, prover.transcript)?;

        let mut range_commitments = range_logup_witness.into_commitments();
        let value_commitment = range_commitments.remove(0);

        let num_vars = ceil_log2(range_prover_info.column_evals()[0].len());

        let value_poly: MultilinearExtension<E> = value_commitment.1.clone();

        let shifted_chunks_polys = range_commitments
            .iter()
            .map(|(_, poly)| poly.clone())
            .collect::<Vec<MultilinearExtension<E>>>();

        let last_claim_eq: MultilinearExtension<E> =
            compute_betas_eval(&last_claim.point).into_mle();

        let range_eq: MultilinearExtension<E> =
            compute_betas_eval(&range_logup_proof.output_claims()[0].point).into_mle();

        // Now we have to perform a sumcheck that shows the following:
        //      1) last_claim_eq * (value_chunk)
        //      2) We need all the polys to be evaluated at the same point so we add all of them multiplied by range_eq.

        // Work out how many challenges we need for batching purposes
        let batch_challenge_count = ceil_log2(shifted_chunks_polys.len() + 4);
        let challenges = (0..batch_challenge_count)
            .map(|_| {
                prover
                    .transcript
                    .sample_and_append_challenge(b"batching")
                    .elements
            })
            .collect::<Vec<E>>();

        let batch_challenges = compute_betas_eval(&challenges);

        // let mut vp = VirtualPolynomial::<E>::new(num_vars);
        let value_challenge = batch_challenges[0];
        let value_const: E = (*quantization::MAX + 1).to_field();

        let num_threads = optimal_sumcheck_threads(num_vars);
        let mut expr_builder = VirtualPolynomialsBuilder::<E>::new(num_threads, num_vars);
        let mut all_mles = Vec::new();
        // We will add all the individual polynomials first
        let range_expr = expr_builder.lift(Either::Left(&range_eq));
        let value_expr = expr_builder.lift(Either::Left(&value_poly));
        let value_chal_expr = Expression::Constant(Either::Right(value_challenge));
        // vp.add_mle_list(vec![range_eq.clone().into(), value_poly.clone().into()], value_challenge);
        let expr1 = range_expr.clone() * value_expr.clone() * value_chal_expr.clone();
        all_mles.push(expr1);
        // vp.add_mle_list(vec![range_eq.clone().into()], value_const * value_challenge);
        let expr2 =
            range_expr.clone() * Expression::Constant(Either::Right(value_const * value_challenge));
        all_mles.push(expr2);

        // let mut vp = shifted_chunks_polys
        //    .iter()
        //    .zip(batch_challenges.iter().skip(1))
        //    .fold(vp, |mut vp_acc, (poly, &chal)| {
        //        vp_acc.add_mle_list(vec![range_eq.clone().into(), poly.clone().into()], chal);
        //        vp_acc
        //    });
        let shifted_exprs = shifted_chunks_polys
            .iter()
            .zip(batch_challenges.iter().skip(1))
            .map(|(poly, &chal)| {
                range_expr.clone()
                    * expr_builder.lift(Either::Left(poly))
                    * Expression::Constant(Either::Right(chal))
            })
            .collect::<Vec<Expression<E>>>();
        all_mles.extend(shifted_exprs);

        // And finally that the output corresponds to what we expect it to
        let current_chal_index = shifted_chunks_polys.len() + 2;
        let claim_challenge = batch_challenges[current_chal_index];
        let claim_chal_expr = Expression::Constant(Either::Right(claim_challenge));
        let last_claim_eq_expr = expr_builder.lift(Either::Left(&last_claim_eq));
        // vp.add_mle_list(vec![value_poly, last_claim_eq.clone()], claim_challenge);
        let expr3 = value_expr.clone() * last_claim_eq_expr * claim_chal_expr;
        all_mles.push(expr3);

        let virtual_poly = expr_builder.to_virtual_polys(&all_mles, &[]);
        // Run the sumcheck prover
        let (proof, state) = IOPProverState::<E>::prove(virtual_poly, prover.transcript);
        let evals = state.get_mle_flatten_final_evaluations();

        let range_evals = &evals[1..2 + shifted_chunks_polys.len()];
        let (value_eval, range_evals) = range_evals.split_at(1);

        let point = &state.collect_raw_challenges();

        // Now we calculate the claim about the input from the shifted chunks, zero_in evals, value eval and sign eval
        let input_eval = self.recombine_claims(value_eval[0], range_evals, &[]);
        let input_claim = Claim::<E>::new(point.clone(), input_eval);

        // Add all the commitments to the commitment prover
        let all_evals_iter = [&value_eval[0]].into_iter().chain(range_evals);
        let all_commits_iter = [value_commitment].into_iter().chain(range_commitments);

        let (commitments, evaluations): (Vec<PCS::Commitment>, Vec<E>) = all_commits_iter
            .zip(all_evals_iter)
            .map(|(comm_with_wit, &eval)| {
                let commitment = PCS::get_pure_commitment(&comm_with_wit.0);
                prover
                    .commit_prover
                    .add_witness_claim(comm_with_wit, Claim::<E>::new(point.clone(), eval))?;

                Result::<(PCS::Commitment, E), anyhow::Error>::Ok((commitment, eval))
            })
            .collect::<Result<Vec<(PCS::Commitment, E)>, anyhow::Error>>()?
            .into_iter()
            .unzip();

        // Add the layer proof to the list
        prover.push_proof(
            id,
            LayerProof::Requant(RequantProof {
                io_accumulation: proof,
                accumulation_evals: evaluations,
                lookup_proofs: vec![range_logup_proof],
                commitments,
            }),
        );

        Ok(input_claim)
    }

    #[timed::timed_instrument(name = "Prover::prove_requant")]
    /// Method that proves requantisation was performed correctly. It does this by running any required lookups and then linking the `last_claim` to the
    /// `input` via a series of Sumchecks.
    pub(crate) fn prove_step<E, T: Transcript<E>, PCS: PolynomialCommitmentScheme<E>>(
        &self,
        prover: &mut Prover<E, T, PCS>,
        last_claim: &Claim<E>,
        _requant_info: &RequantCtx,
        id: NodeId,
    ) -> anyhow::Result<Claim<E>>
    where
        E: ExtensionField + Serialize + DeserializeOwned,
        E::BaseField: Serialize + DeserializeOwned,
    {
        let mut logup_witnesses = prover.lookup_witness(id)?;
        // Check that we have two witnesses for requantisation
        if logup_witnesses.len() != 2 {
            return Err(anyhow!(
                "There should be two lookup witnesses during requantisation, node: {}, number of witnesses: {}",
                id,
                logup_witnesses.len()
            ));
        }
        // Run the lookup protocol and return the lookup proof
        let zero_check_logup_witness = logup_witnesses.remove(0);
        let zero_check_prover_info =
            zero_check_logup_witness.get_logup_input(&prover.challenge_storage)?;
        let range_logup_witness = logup_witnesses.remove(0);
        let range_prover_info = range_logup_witness.get_logup_input(&prover.challenge_storage)?;
        let mut range_commitments = range_logup_witness.into_commitments();
        let value_commitment = range_commitments.remove(0);
        let mut zero_commitments = zero_check_logup_witness.into_commitments();
        let sign_commitment = zero_commitments.pop().ok_or(anyhow!(
            "Had no zero check commitments during requant which should be impossible"
        ))?;

        // Run the lookup protocol and return the lookup proofs
        let zero_check_logup_proof = logup_batch_prove(&zero_check_prover_info, prover.transcript)?;
        let range_logup_proof = logup_batch_prove(&range_prover_info, prover.transcript)?;

        let zero_polys = zero_check_prover_info.column_evals();
        let num_vars = ceil_log2(zero_polys[0].len());

        // We need the input and output columns from the zero check lookup
        let (zero_in, zero_out): (Vec<MultilinearExtension<E>>, Vec<MultilinearExtension<E>>) =
            zero_commitments
                .chunks(2)
                .map(|chunk| (chunk[0].1.clone(), chunk[1].1.clone()))
                .unzip();

        let value_poly: MultilinearExtension<E> = value_commitment.1.clone();
        let sign_poly: MultilinearExtension<E> = sign_commitment.1.clone();

        let shifted_chunks_polys = range_commitments
            .iter()
            .map(|(_, poly)| poly.clone())
            .collect::<Vec<MultilinearExtension<E>>>();

        let last_claim_eq: MultilinearExtension<E> =
            compute_betas_eval(&last_claim.point).into_mle();
        let zero_eq: MultilinearExtension<E> =
            compute_betas_eval(&zero_check_logup_proof.output_claims()[0].point).into_mle();
        let range_eq: MultilinearExtension<E> =
            compute_betas_eval(&range_logup_proof.output_claims()[0].point).into_mle();

        // We squeeze a random challenge point from the transcript to enforce that sign_poly is either 0 or -1
        let challenge_point = (0..last_claim.point.len())
            .map(|_| {
                prover
                    .transcript
                    .sample_and_append_challenge(b"sign")
                    .elements
            })
            .collect::<Vec<E>>();
        let sign_eq: MultilinearExtension<E> = compute_betas_eval(&challenge_point).into_mle();

        // Now we have to perform a sumcheck that shows the following:
        //      1) last_claim_eq * (zero_out.product() * (value_chunk) + (1 - zero_out.product()) * (*quantization::MAX  + sign_poly * (*quantization::MAX + *quantization::MIN.abs()))) = last_claim.eval
        //      2) sign_eq * (sign_poly * (1 + sign_poly)) = 0
        //      3) We need all the polys to be evaluated at the same point so we add all of them multiplied by either zero_eq or range_eq depending on which lookup they came from.

        // Work out how many challenges we need for batching purposes
        let batch_challenge_count = ceil_log2(2 * zero_in.len() + shifted_chunks_polys.len() + 4);
        let challenges = (0..batch_challenge_count)
            .map(|_| {
                prover
                    .transcript
                    .sample_and_append_challenge(b"batching")
                    .elements
            })
            .collect::<Vec<E>>();

        let batch_challenges = compute_betas_eval(&challenges);

        // We will add all the individual polynomials first
        let num_threads = optimal_sumcheck_threads(num_vars);
        let mut expr_builder = VirtualPolynomialsBuilder::<E>::new(num_threads, num_vars);
        let mut all_exprs = Vec::new();
        // NOTE: careful about the order of the lift here, as lift(a) -> lift(b) is not the same as lift(b) -> lift(a)
        let zero_eq_expr = expr_builder.lift(Either::Left(&zero_eq));
        // let mut vp = zero_in
        //    .iter()
        //    .chain(zero_out.iter())
        //    .zip(batch_challenges.iter())
        //    .fold(
        //        VirtualPolynomial::<E>::new(num_vars),
        //        |mut vp_acc, (poly, &chal)| {
        //            vp_acc.add_mle_list(vec![zero_eq.clone(), poly.clone()], chal);
        //            vp_acc
        //        },
        //    );
        let zero_exprs = zero_in
            .iter()
            .chain(zero_out.iter())
            .zip(batch_challenges.iter())
            .map(|(poly, &chal)| {
                let chal_expr = Expression::Constant(Either::Right(chal));
                let poly_expr = expr_builder.lift(Either::Left(poly));
                zero_eq_expr.clone() * poly_expr.clone() * chal_expr.clone()
            })
            .collect::<Vec<_>>();
        all_exprs.extend(zero_exprs);

        // Add the value poly terms
        let value_const: E = (*quantization::MAX + 1).to_field();
        let value_const_expr = Expression::Constant(Either::Right(value_const));
        let value_challenge = batch_challenges[2 * zero_in.len()];
        let value_chal_expr = Expression::Constant(Either::Right(value_challenge));
        let range_eq_expr = expr_builder.lift(Either::Left(&range_eq));
        let value_expr = expr_builder.lift(Either::Left(&value_poly));
        // vp.add_mle_list(vec![range_eq.clone(), value_poly.clone()], value_challenge);
        let check_range_expr = range_eq_expr.clone() * value_expr * value_chal_expr.clone();
        all_exprs.push(check_range_expr);

        // vp.add_mle_list(vec![range_eq.clone()], value_const * value_challenge);
        let check_const_expr =
            range_eq_expr.clone() * value_const_expr.clone() * value_chal_expr.clone();
        all_exprs.push(check_const_expr);
        // let mut vp = shifted_chunks_polys
        //    .iter()
        //    .zip(batch_challenges.iter().skip(2 * zero_in.len() + 1))
        //    .fold(vp, |mut vp_acc, (poly, &chal)| {
        //        vp_acc.add_mle_list(vec![range_eq.clone(), poly.clone()], chal);
        //        vp_acc
        //    });
        let shifted_exprs = shifted_chunks_polys
            .iter()
            .zip(batch_challenges.iter().skip(2 * zero_in.len() + 1))
            .map(|(poly, &chal)| {
                let poly_expr = expr_builder.lift(Either::Left(poly));
                let chal_expr = Expression::Constant(Either::Right(chal));
                range_eq_expr.clone() * poly_expr.clone() * chal_expr.clone()
            })
            .collect::<Vec<_>>();
        all_exprs.extend(shifted_exprs);

        // Now the check that the sign poly is constructed correctly
        let current_chal_index = 2 * zero_in.len() + shifted_chunks_polys.len() + 1;
        let sign_challenge = batch_challenges[current_chal_index];
        let sign_poly_expr = expr_builder.lift(Either::Left(&sign_poly));
        let sign_eq_expr = expr_builder.lift(Either::Left(&sign_eq));
        let sign_chal_expr = Expression::Constant(Either::Right(sign_challenge));
        // vp.add_mle_list(vec![sign_poly.clone(), sign_eq.clone()], sign_challenge);
        let sign_expr = sign_poly_expr.clone() * sign_eq_expr.clone() * sign_chal_expr.clone();
        all_exprs.push(sign_expr);
        // vp.add_mle_list(
        //    vec![sign_poly.clone(), sign_poly.clone(), sign_eq.clone()],
        //    sign_challenge,
        //);
        let sign2_expr = sign_poly_expr.clone()
            * sign_poly_expr.clone()
            * sign_eq_expr.clone()
            * sign_chal_expr.clone();
        all_exprs.push(sign2_expr);

        // And finally that the output corresponds to what we expect it to
        let claim_challenge = batch_challenges[current_chal_index + 1];
        let quant_max_field: E = (*quantization::MAX).to_field();

        let prod_and_value = zero_out
            .iter()
            .chain([&value_poly, &last_claim_eq])
            .map(|poly| expr_builder.lift(Either::Left(poly)))
            .collect::<Vec<_>>();
        let prod_and_sign = zero_out
            .iter()
            .chain([&sign_poly, &last_claim_eq])
            .map(|poly| expr_builder.lift(Either::Left(poly)))
            .collect::<Vec<_>>();

        // vp.add_mle_list(prod_and_value, claim_challenge);
        let claim_chal_expr = Expression::Constant(Either::Right(claim_challenge));
        let prod_value_expr = prod_and_value
            .into_iter()
            .fold(Expression::Constant(Either::Right(E::ONE)), |acc, p| {
                acc * p
            });
        let prod_value_expr = prod_value_expr * claim_chal_expr;
        all_exprs.push(prod_value_expr);

        let prod_sign_expr = prod_and_sign
            .into_iter()
            .fold(Expression::Constant(Either::Right(E::ONE)), |acc, p| {
                acc * p
            });
        // vp.add_mle_list(
        //    prod_and_sign,
        //    -(quant_max_field + value_const) * claim_challenge,
        //);
        // Here we use `value_const` because it has the same value as `*quantization::MIN.abs()`
        let prod_sign_chal_expr = Expression::Constant(Either::Right(
            -(quant_max_field + value_const) * claim_challenge,
        ));
        let prod_sign_expr = prod_sign_expr * prod_sign_chal_expr;
        all_exprs.push(prod_sign_expr);

        // vp.add_mle_list(
        //    vec![sign_poly, last_claim_eq.clone()],
        //    (quant_max_field + value_const) * claim_challenge,
        //);
        let last_claim_eq_expr = expr_builder.lift(Either::Left(&last_claim_eq));
        let last_claim_chal_expr = Expression::Constant(Either::Right(
            (quant_max_field + value_const) * claim_challenge,
        ));
        let last_claim_sign_expr =
            sign_poly_expr * last_claim_eq_expr.clone() * last_claim_chal_expr;
        all_exprs.push(last_claim_sign_expr);

        // zero_out.push(last_claim_eq.clone());
        // vp.add_mle_list(zero_out, -quant_max_field * claim_challenge);
        let extended_zero_out = zero_out
            .iter()
            .chain(std::iter::once(&last_claim_eq))
            .collect::<Vec<_>>();
        let zero_out_new_expr = extended_zero_out
            .into_iter()
            .map(|poly| expr_builder.lift(Either::Left(poly)))
            .fold(Expression::Constant(Either::Right(E::ONE)), |acc, p| {
                acc * p
            });
        let zero_out_new_chal =
            Expression::Constant(Either::Right(-quant_max_field * claim_challenge));
        let zero_out_new_expr = zero_out_new_expr * zero_out_new_chal;
        all_exprs.push(zero_out_new_expr);

        // vp.add_mle_list(vec![last_claim_eq], quant_max_field * claim_challenge);
        let quant_max_field_expr =
            Expression::Constant(Either::Right(quant_max_field * claim_challenge));
        let last_claim_eq_expr = last_claim_eq_expr.clone() * quant_max_field_expr;
        all_exprs.push(last_claim_eq_expr);

        // Run the sumcheck prover
        let virtual_poly = expr_builder.to_virtual_polys(&all_exprs, &[]);
        let (claim_acc_proof, state) = IOPProverState::<E>::prove(virtual_poly, prover.transcript);
        let evals = state.get_mle_flatten_final_evaluations();
        let all_zero_evals = &evals[1..1 + 2 * zero_in.len()];
        let (zero_in_evals, zero_out_evals) = all_zero_evals.split_at(zero_in.len());
        let range_evals =
            &evals[2 + 2 * zero_in.len()..2 + 2 * zero_in.len() + shifted_chunks_polys.len() + 1];
        let (value_eval, range_evals) = range_evals.split_at(1);
        let sign_eval = evals[2 + 2 * zero_in.len() + shifted_chunks_polys.len() + 1];

        let point = state.collect_raw_challenges();
        // let point = &claim_acc_proof.point;

        // Now we calculate the claim about the input from the shifted chunks, zero_in evals, value eval and sign eval
        let input_eval = self.recombine_claims(value_eval[0], range_evals, zero_in_evals);
        let input_claim = Claim::<E>::new(point.clone(), input_eval);

        // Add all the commitments to the commitment prover
        let all_evals_iter = zero_in_evals
            .iter()
            .interleave(zero_out_evals)
            .chain([&sign_eval, &value_eval[0]])
            .chain(range_evals);
        let all_commits_iter = zero_commitments
            .into_iter()
            .chain([sign_commitment, value_commitment])
            .chain(range_commitments.into_iter());

        let (commitments, evaluations): (Vec<PCS::Commitment>, Vec<E>) = all_commits_iter
            .zip(all_evals_iter)
            .map(|(comm_with_wit, &eval)| {
                let commitment = PCS::get_pure_commitment(&comm_with_wit.0);
                prover
                    .commit_prover
                    .add_witness_claim(comm_with_wit, Claim::<E>::new(point.clone(), eval))?;

                Result::<(PCS::Commitment, E), anyhow::Error>::Ok((commitment, eval))
            })
            .collect::<Result<Vec<(PCS::Commitment, E)>, anyhow::Error>>()?
            .into_iter()
            .unzip();

        // Add the layer proof to the list
        prover.push_proof(
            id,
            LayerProof::Requant(RequantProof {
                io_accumulation: claim_acc_proof,
                accumulation_evals: evaluations,
                lookup_proofs: vec![zero_check_logup_proof, range_logup_proof],
                commitments,
            }),
        );

        Ok(input_claim)
    }
}

impl RequantCtx {
    /// Method that verifies requantisation has been performed correctly when supplied with a [`RequantProof`].
    /// It verifies both lookup argument proofs, calculates the initial claim for the sumcheck proof using the lookup argument claims
    /// and then verifies the sumcheck using this initial claim. It then takes the output claims provided by the prover, checks they relate to the sumcheck
    /// subclaim, adds them to the list of claims of commitment openings and then calculates the next claim.
    pub(crate) fn verify_requant<E, T: Transcript<E>, PCS: PolynomialCommitmentScheme<E>>(
        &self,
        verifier: &mut Verifier<E, T, PCS>,
        last_claim: &Claim<E>,
        proof: &RequantProof<E, PCS>,
        constant_challenge: E,
        column_separation_challenge: E,
    ) -> anyhow::Result<Claim<E>>
    where
        E: ExtensionField + Serialize + DeserializeOwned,
        E::BaseField: Serialize + DeserializeOwned,
    {
        // 1. Verify the lookup proofs
        let RequantProof {
            io_accumulation,
            accumulation_evals,
            lookup_proofs,
            commitments,
        } = proof;
        // Work out how many instances of range check and zero check there are
        let (shifted_instances, _, _) = self.requant.shifted_chunks_data();
        // Add one to shifted instances because the value_chunk is also range checked
        let range_instances = shifted_instances + 1;

        let zero_check_instances = self.requant.number_of_zero_chunks();

        // Verify both lookup arguments in the same order they are proved.
        let zero_check_claims = verify_logup_proof(
            &lookup_proofs[0],
            zero_check_instances,
            constant_challenge,
            column_separation_challenge,
            verifier.transcript,
        )?;
        let range_claims = verify_logup_proof(
            &lookup_proofs[1],
            range_instances,
            constant_challenge,
            E::ONE,
            verifier.transcript,
        )?;

        let (zero_in_evals, zero_out_evals): (Vec<E>, Vec<E>) = zero_check_claims
            .claims()
            .chunks(2)
            .map(|chunk| (chunk[0].eval, chunk[1].eval))
            .unzip();
        let all_range_claims = range_claims.claims();
        let value_eval = all_range_claims[0].eval;
        let shifted_evals = all_range_claims[1..]
            .iter()
            .map(|c| c.eval)
            .collect::<Vec<E>>();

        // We squeeze a random challenge point from the transcript for the check that sign_poly is either 0 or -1
        let challenge_point = (0..last_claim.point.len())
            .map(|_| {
                verifier
                    .transcript
                    .sample_and_append_challenge(b"sign")
                    .elements
            })
            .collect::<Vec<E>>();

        // Now we have to perform a sumcheck that shows the following:
        //      1) last_claim_eq * (zero_out.product() * (value_chunk) + (1 - zero_out.product()) * (*quantization::MAX  + sign_poly * (*quantization::MAX + *quantization::MIN.abs()))) = last_claim.eval
        //      2) sign_eq * (sign_poly * (1 + sign_poly)) = 0
        //      3) We need all the polys to be evaluated at the same point so we add all of them multiplied by either zero_eq or range_eq depending on which lookup they came from.

        // Work out how many challenges we need for batching purposes
        let batch_challenge_count = ceil_log2(2 * zero_in_evals.len() + shifted_evals.len() + 4);
        let challenges = (0..batch_challenge_count)
            .map(|_| {
                verifier
                    .transcript
                    .sample_and_append_challenge(b"batching")
                    .elements
            })
            .collect::<Vec<E>>();

        let batch_challenges = compute_betas_eval(&challenges);

        // Now we reconstruct the initial evaluation for the sumcheck proof
        let first_part = zero_in_evals
            .iter()
            .chain(zero_out_evals.iter())
            .chain(std::iter::once(&value_eval))
            .chain(shifted_evals.iter())
            .zip(batch_challenges.iter())
            .fold(E::ZERO, |acc, (&eval, &chal)| acc + eval * chal);

        // The sign poly check has inital evaluation 0 so now we just need to skip a challenge and then add last_claim.eval multiplied by the correct challenge
        let current_index = 2 * zero_in_evals.len() + 1 + shifted_evals.len();
        let claim_challenge = batch_challenges[current_index + 1];

        let last_claim_part = claim_challenge * last_claim.eval;

        let quant_max_field: E = (*quantization::MAX).to_field();
        let num_vars = last_claim.point.len();
        // The highest degree term is the product of all zero_out polys with an eq poly and the value poly.
        let aux_info =
            crate::util::from_mle_list_dimensions(&[vec![num_vars; zero_out_evals.len() + 2]]);
        // Run sumcheck verification
        let subclaim = IOPVerifierState::<E>::verify(
            first_part + last_claim_part,
            io_accumulation,
            &aux_info,
            verifier.transcript,
        );

        let point = subclaim
            .point
            .iter()
            .map(|p| p.elements)
            .collect::<Vec<_>>();

        // Now that we have the subclaim we must check it links to the evaluations provided by the prover
        let (zero_in_claims, zero_out_claims): (Vec<E>, Vec<E>) = accumulation_evals
            .chunks(2)
            .take(zero_in_evals.len())
            .map(|chunk| (chunk[0], chunk[1]))
            .unzip();
        let sign_claim = accumulation_evals[2 * zero_in_evals.len()];
        let value_claim = accumulation_evals[2 * zero_in_evals.len() + 1];
        let shifted_claims = &accumulation_evals[2 * zero_in_evals.len() + 2..];

        let last_claim_eq = eq_xy_eval(&last_claim.point, &point);
        let zero_eq = eq_xy_eval(&zero_check_claims.claims()[0].point, &point);
        let range_eq = eq_xy_eval(&range_claims.claims()[0].point, &point);
        let sign_eq = eq_xy_eval(&challenge_point, &point);

        // Reconstruct the subclaim eval
        let zero_part = zero_eq
            * zero_in_claims
                .iter()
                .chain(zero_out_claims.iter())
                .zip(batch_challenges.iter())
                .fold(E::ZERO, |acc, (&eval, &chal)| acc + eval * chal);

        let value_part = batch_challenges[2 * zero_in_claims.len()]
            * range_eq
            * (value_claim + quant_max_field + E::ONE);
        let with_range_part = shifted_claims
            .iter()
            .zip(batch_challenges.iter().skip(zero_in_claims.len() * 2 + 1))
            .fold(zero_part + value_part, |acc, (&eval, &chal)| {
                acc + range_eq * chal * eval
            });
        let with_sign_part = with_range_part
            + batch_challenges[current_index] * sign_eq * sign_claim * (E::ONE + sign_claim);

        // Finally the last claim part
        let zero_out_prod = zero_out_claims.iter().copied().product::<E>();

        let calc_subclaim = with_sign_part
            + claim_challenge
                * last_claim_eq
                * (zero_out_prod * value_claim
                    + (E::ONE - zero_out_prod)
                        * (quant_max_field
                            + sign_claim * (quant_max_field + quant_max_field + E::ONE)));

        // Check the subclaims line up
        ensure!(
            subclaim.expected_evaluation == calc_subclaim,
            "Requant verification failed because the calculated subclaim evaluation {:?} did not equal the expected subclaim evaluation {:?}",
            calc_subclaim,
            subclaim.expected_evaluation
        );

        // Recombine for the input claim
        let input_eval =
            self.requant
                .recombine_claims(value_claim, shifted_claims, &zero_in_claims);

        // Add the commitments to the commitment verifier
        commitments
            .iter()
            .zip(accumulation_evals.iter())
            .try_for_each(|(comm, &eval)| {
                verifier
                    .commit_verifier
                    .add_witness_claim(comm.clone(), Claim::<E>::new(point.clone(), eval))
            })?;

        // Pass the input claim
        Ok(Claim::<E>::new(point, input_eval))
    }

    pub(crate) fn verify_requant_no_zero_chunks<
        E,
        T: Transcript<E>,
        PCS: PolynomialCommitmentScheme<E>,
    >(
        &self,
        verifier: &mut Verifier<E, T, PCS>,
        last_claim: &Claim<E>,
        proof: &RequantProof<E, PCS>,
        constant_challenge: E,
    ) -> anyhow::Result<Claim<E>>
    where
        E: ExtensionField + Serialize + DeserializeOwned,
        E::BaseField: Serialize + DeserializeOwned,
    {
        // 1. Verify the lookup proofs
        let RequantProof {
            io_accumulation,
            accumulation_evals,
            lookup_proofs,
            commitments,
        } = proof;
        // Work out how many instances of range check and zero check there are
        let (shifted_instances, _, _) = self.requant.shifted_chunks_data();
        // Add one to shifted instances because the value_chunk is also range checked
        let range_instances = shifted_instances + 1;

        ensure!(
            lookup_proofs.len() == 1,
            "Found more than one lookup proof when verifying Requant with no zero chunks"
        );

        // Verify the lookup argument.

        let range_claims = verify_logup_proof(
            &lookup_proofs[0],
            range_instances,
            constant_challenge,
            E::ONE,
            verifier.transcript,
        )?;

        let all_range_claims = range_claims.claims();
        let value_eval = all_range_claims[0].eval;
        let shifted_evals = all_range_claims[1..]
            .iter()
            .map(|c| c.eval)
            .collect::<Vec<E>>();

        // Now we have to perform a sumcheck that shows the following:
        //      1) last_claim_eq * (value_chunk) = last_claim.eval
        //      2) We need all the polys to be evaluated at the same point so we add all of them multiplied by either zero_eq or range_eq depending on which lookup they came from.

        // Work out how many challenges we need for batching purposes
        let batch_challenge_count = ceil_log2(shifted_evals.len() + 4);
        let challenges = (0..batch_challenge_count)
            .map(|_| {
                verifier
                    .transcript
                    .sample_and_append_challenge(b"batching")
                    .elements
            })
            .collect::<Vec<E>>();

        let batch_challenges = compute_betas_eval(&challenges);

        // Now we reconstruct the initial evaluation for the sumcheck proof
        let first_part = std::iter::once(&value_eval)
            .chain(shifted_evals.iter())
            .zip(batch_challenges.iter())
            .fold(E::ZERO, |acc, (&eval, &chal)| acc + eval * chal);

        // The sign poly check has inital evaluation 0 so now we just need to skip a challenge and then add last_claim.eval multiplied by the correct challenge
        let current_index = 2 + shifted_evals.len();
        let claim_challenge = batch_challenges[current_index];

        let last_claim_part = claim_challenge * last_claim.eval;

        let num_vars = last_claim.point.len();
        // The highest degree term is the product of all zero_out polys with an eq poly and the value poly.
        let aux_info = crate::util::from_mle_list_dimensions(&[vec![num_vars; 2]]);

        // Run sumcheck verification
        let subclaim = IOPVerifierState::<E>::verify(
            first_part + last_claim_part,
            io_accumulation,
            &aux_info,
            verifier.transcript,
        );

        let point = subclaim
            .point
            .iter()
            .map(|p| p.elements)
            .collect::<Vec<_>>();

        // Now that we have the subclaim we must check it links to the evaluations provided by the prover
        let value_claim = accumulation_evals[0];
        let shifted_claims = &accumulation_evals[1..];

        let last_claim_eq = eq_xy_eval(&last_claim.point, &point);
        let range_eq = eq_xy_eval(&range_claims.claims()[0].point, &point);

        // Reconstruct the subclaim eval
        let value_const: E = (*quantization::MAX + 1).to_field();
        let value_part = batch_challenges[0] * range_eq * (value_claim + value_const);
        let range_part = shifted_claims
            .iter()
            .zip(batch_challenges.iter().skip(1))
            .fold(value_part, |acc, (&eval, &chal)| {
                acc + range_eq * chal * eval
            });

        // Finally the last claim part
        let calc_subclaim = range_part + claim_challenge * last_claim_eq * value_claim;

        // Check the subclaims line up
        ensure!(
            subclaim.expected_evaluation == calc_subclaim,
            "Requant verification failed because the calculated subclaim evaluation {:?} did not equal the expected subclaim evaluation {:?}",
            calc_subclaim,
            subclaim.expected_evaluation
        );

        // Recombine for the input claim
        let input_eval = self
            .requant
            .recombine_claims(value_claim, shifted_claims, &[]);

        // Add the commitments to the commitment verifier
        commitments
            .iter()
            .zip(accumulation_evals.iter())
            .try_for_each(|(comm, &eval)| {
                verifier
                    .commit_verifier
                    .add_witness_claim(comm.clone(), Claim::<E>::new(point.clone(), eval))
            })?;

        // Pass the input claim
        Ok(Claim::<E>::new(point, input_eval))
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
