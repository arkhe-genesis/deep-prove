//! Module containing code to verify an [`InSumProof`] against an [`InSumContext`].

use std::collections::HashMap;

use dp_crypto::{
    structs::IOPVerifierState, util::ceil_log2, utils::eval_by_expr_with_instance,
    virtual_poly::VPAuxInfo,
};
use either::Either;
use itertools::{Itertools, izip};

use crate::{
    Claim,
    layers::einsum::{
        axis::{FixedAxesMapping, FixedAxis},
        prove::reconstruct_full_point,
    },
    poly_commit::identity_eval,
    tensor::CommitmentId,
    try_unzip,
};

use super::*;

pub(crate) struct EinSumVerifierInfo<F: PrimeField> {
    pub(crate) claims: Vec<Claim<F>>,
    pub(crate) constants_map: HashMap<CommitmentId, Claim<F>>,
}

impl<F: PrimeField> EinSumContext<F> {
    pub(crate) fn verify_internal<T>(
        &self,
        proof: &EinSumProof<F>,
        last_claims: &[&Claim<F>],
        shape_step: &ShapeStep,
        transcript: &mut T,
    ) -> Result<EinSumVerifierInfo<F>>
    where
        T: Transcript,
    {
        // Check we have the correct number of claims
        ensure!(
            last_claims.len() == self.mapping.output_count(),
            "Expected {} last claims, got {}",
            self.mapping.output_count(),
            last_claims.len()
        );

        // Make the output shapes from the input shapes
        let mut unpadded_inputs_iter = shape_step.unpadded_input_shape.iter();
        let lhs_unpadded_shape = unpadded_inputs_iter
            .next()
            .ok_or(anyhow!("Missing LHS unpadded input shape"))?;

        let mut padded_inputs_iter = shape_step.padded_input_shape.iter();
        let lhs_padded_shape = padded_inputs_iter
            .next()
            .ok_or(anyhow!("Missing LHS padded input shape"))?;

        let (unpadded_input_shapes, padded_input_shapes): (Vec<_>, Vec<_>) = try_unzip(
            std::iter::once(anyhow::Ok((
                lhs_unpadded_shape.clone(),
                lhs_padded_shape.clone(),
            )))
            .chain(self.constant_unpadded_shapes.iter().map(|const_shape| {
                if let Some(s) = const_shape {
                    Ok((s.clone(), s.next_power_of_two()))
                } else {
                    Ok((
                        unpadded_inputs_iter
                            .next()
                            .ok_or(anyhow!("Missing unpadded input shape"))?
                            .clone(),
                        padded_inputs_iter
                            .next()
                            .ok_or(anyhow!("Missing padded input shape"))?
                            .clone(),
                    ))
                }
            })),
        )?;
        let output_shapes = self.mapping.output_shapes(&padded_input_shapes)?;
        let unpadded_output_shapes = self.mapping.output_shapes(&unpadded_input_shapes)?;

        let contraction_size =
            self.mapping.axes_sizes(&unpadded_input_shapes)?[AxisType::Contracted];
        // Unpack the proof
        let EinSumProof {
            bias_evals,
            einsum_sumcheck,
            einsum_evaluations,
            input_aggregation_sumcheck,
        } = proof;

        let mut bias_evals_iter = bias_evals.iter();
        // split the points according to the output shapes and subtract bias claims from them if needed
        let mut bias_usize_id = 0;
        let (split_points, claim_evaluations, bias_claims) = izip!(
            last_claims.iter(),
            output_shapes.iter(),
            unpadded_output_shapes.iter(),
            self.bias_unpadded_shapes.iter(),
            self.bias_keys.iter(),
        ).enumerate()
        .try_fold(
            (vec![], vec![], vec![]),
            |(mut split_points, mut claim_evaluations, mut bias_claims),
             (output_id, (claim, output_shape, unpadded_output_shape, bias_opt, bias_id))| {
                if bias_opt.is_some() {
                    let bias_eval_from_proof = bias_evals_iter
                        .next()
                        .ok_or(anyhow!("Not enough bias evaluations in proof"))?;
                    let split_point = output_shape.split_point(claim.point())?;
                    let (bias_eval, bias_claim) = self.mapping.compute_bias_evaluation::<F>(output_id, bias_usize_id, &split_point, *bias_eval_from_proof, unpadded_output_shape)?;
                    bias_usize_id += 1;
                    let claim_evaluation = claim.evaluation() - bias_eval;
                    claim_evaluations.push(claim_evaluation);
                    split_points.push(split_point);
                    bias_claims.push((bias_id.clone().unwrap(), bias_claim));
                    Result::<(Vec<_>, Vec<_>, Vec<_>)>::Ok((
                        split_points,
                        claim_evaluations,
                        bias_claims,
                    ))
                } else {
                    let split_point = output_shape.split_point(claim.point())?;
                    split_points.push(split_point);
                    claim_evaluations.push(claim.evaluation());
                    Result::<(Vec<_>, Vec<_>, Vec<_>)>::Ok((
                        split_points,
                        claim_evaluations,
                        bias_claims,
                    ))
                }
            },
        )?;

        // Squeeze the batching challenge for the einsum sumcheck
        let batching_challenge = transcript.append_and_sample(b"batching_challenge");

        // Calculate the initial sumcheck evaluation
        let claimed_sum = claim_evaluations
            .iter()
            .fold((F::ZERO, F::ONE), |(acc, coeff), &eval| {
                (acc + coeff * eval, coeff * batching_challenge)
            })
            .0;

        // Work out the number of variables in the sumcheck and the stacking coefficients
        let fixed_axes_mapping = self.mapping.sort_variables_to_axes(&split_points)?;
        let stacking_coeffs = fixed_axes_mapping.stacking_coefficients(&unpadded_input_shapes[0]);
        let FixedAxesMapping {
            lhs_fixes,
            rhs_fixes,
        } = fixed_axes_mapping;

        let num_vars = ceil_log2(contraction_size);

        let aux_info = VPAuxInfo {
            max_num_variables: num_vars,
            max_degree: 2,
            ..Default::default()
        };
        let einsum_subclaim =
            IOPVerifierState::<F>::verify(claimed_sum, einsum_sumcheck, &aux_info, transcript);

        let einsum_point = &einsum_subclaim.point;

        let stacking_coeffs_ref = stacking_coeffs
            .iter()
            .map(|s| s.as_ref())
            .collect::<Vec<&[F]>>();
        // Rebuild the einsum sumcheck expression to avoid storing it
        let einsum_sumcheck_expression = self.build_einsum_expression(&stacking_coeffs_ref);
        let calculated_evaluation = eval_by_expr_with_instance(
            &[],
            einsum_evaluations,
            &[],
            &[],
            &[batching_challenge],
            &einsum_sumcheck_expression,
        );

        ensure!(
            einsum_subclaim.expected_evaluation == calculated_evaluation,
            "Einsum sumcheck evaluation mismatch: expected {}, got {}",
            einsum_subclaim.expected_evaluation,
            calculated_evaluation,
        );

        // Now we split the RHS evaluations into input and constant claims
        let half = einsum_evaluations.len() / 2;
        let (mut input_claims, weight_claims): (Vec<_>, Vec<_>) = izip!(
            rhs_fixes,
            einsum_evaluations[half..].chunks(stacking_coeffs[0].len()),
            stacking_coeffs.iter(),
            self.constant_unpadded_shapes.iter(),
            self.constant_keys.iter(),
            padded_input_shapes.iter().skip(1)
        )
        .partition_map(
            |(rhs_point, eval_chunk, coeffs, weight_opt, poly_id, input_shape)| {
                let full_point =
                    reconstruct_full_point(&rhs_point, einsum_point, input_shape.as_slice());

                let eval = eval_chunk
                    .iter()
                    .zip(coeffs)
                    .fold(F::ZERO, |acc, (e, c)| acc + *e * *c);
                if weight_opt.is_some() {
                    Either::Right((poly_id.clone().unwrap(), Claim::<F>::new(full_point, eval)))
                } else {
                    Either::Left(Claim::<F>::new(full_point, eval))
                }
            },
        );

        let constant_polys_map = weight_claims
            .into_iter()
            .chain(bias_claims)
            .collect::<HashMap<CommitmentId, Claim<F>>>();

        if let Some(agg_proof) = input_aggregation_sumcheck {
            let eq_points = lhs_fixes
                .iter()
                .map(|split_point| {
                    reconstruct_full_point(
                        split_point,
                        einsum_point,
                        padded_input_shapes[0].as_slice(),
                    )
                })
                .collect::<Vec<Vec<F>>>();

            let num_vars = eq_points[0].len();
            let aux_info = VPAuxInfo {
                max_num_variables: num_vars,
                max_degree: 2,
                ..Default::default()
            };
            let input_batching_challenge: F =
                transcript.append_and_sample(b"input_batching_challenge");
            let claimed_sum = einsum_evaluations[..half]
                .chunks(stacking_coeffs[0].len())
                .zip(stacking_coeffs.iter())
                .fold((F::ZERO, F::ONE), |(acc, coeff), (eval_chunk, scs)| {
                    (
                        acc + eval_chunk
                            .iter()
                            .zip(scs)
                            .fold(F::ZERO, |acc, (e, c)| acc + *e * *c)
                            * coeff,
                        coeff * input_batching_challenge,
                    )
                })
                .0;
            let input_agg_subclaim =
                IOPVerifierState::<F>::verify(claimed_sum, agg_proof, &aux_info, transcript);
            let agg_point = &input_agg_subclaim.point;
            let eval_no_input = eq_points
                .iter()
                .map(|p| identity_eval(p, agg_point))
                .fold((F::ZERO, F::ONE), |(acc, coeff), eval| {
                    (acc + eval * coeff, coeff * input_batching_challenge)
                })
                .0;

            let lhs_input_eval = input_agg_subclaim.expected_evaluation
                * eval_no_input
                    .inverse()
                    .expect("Tried to invert zero in Einsum verifier");
            let lhs_claim = Claim::new(agg_point.clone(), lhs_input_eval);
            input_claims.insert(0, lhs_claim);

            Ok(EinSumVerifierInfo {
                claims: input_claims,
                constants_map: constant_polys_map,
            })
        } else {
            // Just need to work out the LHS claim
            let mut skip = 0usize;
            let lhs_point = lhs_fixes[0]
                .iter()
                .zip(padded_input_shapes[0].iter())
                .map(|(&p_opt, &input_dim)| {
                    // To make the point we have to take and insert the unfixed variables at the correct locations
                    match p_opt {
                        FixedAxis::Outer(point) | FixedAxis::Stacked(point) => point,
                        FixedAxis::Contracted => {
                            let dim_log = ceil_log2(input_dim);
                            let point = &einsum_point[skip..skip + dim_log];
                            skip += dim_log;
                            point
                        }
                    }
                })
                .rev()
                .flatten()
                .copied()
                .collect::<Vec<F>>();
            let lhs_input_eval = einsum_evaluations[..half]
                .chunks(stacking_coeffs[0].len())
                .zip(stacking_coeffs.iter())
                .fold(F::ZERO, |acc, (eval_chunk, scs)| {
                    acc + eval_chunk
                        .iter()
                        .zip(scs)
                        .fold(F::ZERO, |acc, (e, c)| acc + *e * *c)
                });
            let lhs_claim = Claim::new(lhs_point, lhs_input_eval);
            input_claims.insert(0, lhs_claim);
            Ok(EinSumVerifierInfo {
                claims: input_claims,
                constants_map: constant_polys_map,
            })
        }
    }
}
