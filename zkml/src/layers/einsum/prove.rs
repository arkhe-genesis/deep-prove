//! This module contains the code for the proving logic of an [`EinSum`] layer.

use std::collections::HashMap;

use super::*;

use crate::{
    Claim, Shape, Tensor,
    commit::compute_betas_eval,
    layers::einsum::axis::{FixedAxis, FixedPolys},
    quantization::TensorFielder,
    tensor::TensorKey,
};

use anyhow::{Result, anyhow, ensure};

use either::Either;
use ff_ext::ExtensionField;
use itertools::{Itertools, izip};

use multilinear_extensions::{
    mle::MultilinearExtension, util::ceil_log2, virtual_polys::VirtualPolynomialsBuilder,
};

use sumcheck::{structs::IOPProverState, util::optimal_sumcheck_threads};
use transcript::Transcript;

pub(crate) struct EinSumProofInfo<E: ExtensionField> {
    pub(crate) claims: Vec<Claim<E>>,
    pub(crate) proof: EinSumProof<E>,
    pub(crate) commitment_map: HashMap<TensorKey, Claim<E>>,
}

impl<E: ExtensionField> EinSumProofInfo<E> {
    fn new(
        claims: Vec<Claim<E>>,
        proof: EinSumProof<E>,
        commitment_map: HashMap<TensorKey, Claim<E>>,
    ) -> Self {
        Self {
            claims,
            proof,
            commitment_map,
        }
    }
}

impl EinSum<Element> {
    pub(crate) fn prove_internal<E, T>(
        &self,
        ctx: &EinSumContext<E>,
        last_claims: Vec<&Claim<E>>,
        inputs: &[Tensor<E>],
        unpadded_input_shapes: &[Shape],
        transcript: &mut T,
    ) -> Result<EinSumProofInfo<E>>
    where
        E: ExtensionField,
        T: Transcript<E>,
    {
        // Check that we have the correct number of `last_claims` and `inputs`
        let expected_num_inputs = self.mapping.input_count();
        let expected_num_outputs = self.mapping.output_count();
        ensure!(
            last_claims.len() == expected_num_outputs,
            "Expected {expected_num_outputs} last claims, got {}",
            last_claims.len()
        );
        let mut inputs_iter = inputs.iter();
        let first_input = inputs_iter
            .next()
            .ok_or_else(|| anyhow!("At least one input tensor must be provided"));

        let field_constants = self
            .constant_tensors
            .iter()
            .map(|t| t.as_ref().map(|tensor| tensor.tensor().to_fields()))
            .collect::<Vec<Option<Tensor<E>>>>();
        let full_inputs = std::iter::once(first_input)
            .chain(field_constants.iter().map(|t| {
                if let Some(tensor) = t.as_ref() {
                    Ok(tensor)
                } else {
                    inputs_iter
                        .next()
                        .ok_or_else(|| anyhow!("Not enough input tensors provided"))
                }
            }))
            .collect::<Result<Vec<&Tensor<E>>>>()?;

        ensure!(
            full_inputs.len() == expected_num_inputs,
            "Expected {expected_num_inputs} input tensors, got {}",
            full_inputs.len()
        );

        let mut unpadded_inputs_iter = unpadded_input_shapes.iter();
        let lhs_unpadded_shape = unpadded_inputs_iter
            .next()
            .ok_or(anyhow!("Missing LHS unpadded input shape"))?;

        let unpadded_input_shapes = std::iter::once(Ok(lhs_unpadded_shape.clone()))
            .chain(self.constant_unpadded_shapes.iter().map(|const_shape| {
                if let Some(s) = const_shape {
                    // This is a constant tensor, so we take the shape from the tensor itself
                    Ok(s.clone())
                } else {
                    unpadded_inputs_iter
                        .next()
                        .cloned()
                        .ok_or(anyhow!("Missing unpadded input shape"))
                }
            }))
            .collect::<Result<Vec<Shape>>>()?;
        ensure!(
            unpadded_input_shapes.len() == expected_num_inputs,
            "Expected {} unpadded input shapes, got {}",
            expected_num_inputs,
            unpadded_input_shapes.len()
        );

        let input_shapes = full_inputs
            .iter()
            .map(|t| t.shape().clone())
            .collect::<Vec<_>>();
        let output_shapes = self.mapping.output_shapes(&input_shapes)?;

        // Use the output shapes to split the last claim points into their respective dims.
        let split_points = output_shapes
            .iter()
            .zip(last_claims.iter())
            .map(|(shape, claim)| shape.split_point(claim.point()))
            .collect::<Result<Vec<Vec<&[E]>>>>()?;

        let FixedPolys {
            lhs: fixed_lhs_polys,
            rhs: fixed_rhs_polys,
            lhs_points,
            rhs_points,
            stacking_coeffs,
        } = self
            .mapping
            .fix_axes(&split_points, &full_inputs, &unpadded_input_shapes)?;
        let axes_sizes = self.mapping.axes_sizes(&unpadded_input_shapes)?;

        let contraction_size = axes_sizes[AxisType::Contracted];
        let stacking_size = axes_sizes[AxisType::Stacked];

        let batching_challenge = transcript
            .sample_and_append_challenge(b"batching_challenge")
            .elements;

        let num_vars = ceil_log2(contraction_size);

        let num_threads = optimal_sumcheck_threads(num_vars);

        let either_mles = fixed_lhs_polys
            .iter()
            .chain(fixed_rhs_polys.iter())
            .flatten()
            .map(Either::Left)
            .collect::<Vec<_>>();

        let expr_builder =
            VirtualPolynomialsBuilder::<E>::new_with_mles(num_threads, num_vars, either_mles);

        let stacking_coeffs_ref = stacking_coeffs
            .iter()
            .map(|s| s.as_ref())
            .collect::<Vec<_>>();
        let einsum_sumcheck_expression = ctx.build_einsum_expression(&stacking_coeffs_ref);
        let virtual_poly = expr_builder.to_virtual_polys(
            std::slice::from_ref(&einsum_sumcheck_expression),
            &[batching_challenge],
        );

        let (einsum_proof, state) = IOPProverState::<E>::prove(virtual_poly, transcript);

        // We will have twice as many evaluations as we have inputs, the first half relate only to the LHS poly, so we skip those and take the RHS claims when we make
        // any constant polynomial claims.
        let einsum_evaluations = state.get_mle_flatten_final_evaluations();
        let einsum_sumcheck_point = state.collect_raw_challenges();

        // Construct any bias claims
        let mut bias_id = 0;
        let bias_claims = izip!(self.biases.iter(), split_points.iter(),)
            .enumerate()
            .filter_map(|(output_id, (bias_opt, split_point))| {
                if let Some(bias) = bias_opt.as_ref() {
                    let key = bias.key();
                    let bias_field: Tensor<E> = bias.to_fields();
                    let bias_poly = bias_field.to_mle();
                    let bias_point = self
                        .mapping
                        .bias_evaluation_point(output_id, bias_id, split_point)
                        .expect("Point should be valid");
                    bias_id += 1;
                    let eval = bias_poly.evaluate(&bias_point);
                    Some((key, Claim::new(bias_point, eval)))
                } else {
                    None
                }
            })
            .collect::<Vec<_>>();

        // Separate out the bias evals
        let bias_evals = bias_claims
            .iter()
            .map(|(_, claim)| claim.evaluation())
            .collect::<Vec<E>>();

        let half = einsum_evaluations.len() / 2;
        // Separate out the input claims and the constant (weight) claims
        let (mut input_claims, rhs_constant_claims): (Vec<_>, Vec<_>) = izip!(
            rhs_points,
            einsum_evaluations[half..].chunks(stacking_size),
            stacking_coeffs.iter(),
            self.constant_tensors.iter(),
            input_shapes.iter().skip(1)
        )
        .partition_map(
            |(rhs_point, evals_chunk, stacking_coeffs, weight_opt, input_shape)| {
                let full_point = reconstruct_full_point(
                    &rhs_point,
                    &einsum_sumcheck_point,
                    input_shape.as_slice(),
                );

                let eval = stacking_coeffs
                    .iter()
                    .zip(evals_chunk.iter())
                    .fold(E::ZERO, |acc, (coeff, eval)| acc + *coeff * *eval);

                if let Some(weight) = weight_opt.as_ref() {
                    Either::Right((weight.key(), Claim::<E>::new(full_point, eval)))
                } else {
                    Either::Left(Claim::<E>::new(full_point, eval))
                }
            },
        );
        // Make the constant poly map
        let constant_poly_claims = rhs_constant_claims
            .into_iter()
            .chain(bias_claims)
            .collect::<HashMap<TensorKey, Claim<E>>>();

        if let Some(agg_expr) = ctx.input_aggregation_expression.as_ref() {
            let input_mle = inputs[0].to_mle();
            let num_vars = input_mle.num_vars();
            // In this case we have more than one claim about the LHS poly, so we need to build the eq_polys to aggregate the claims.
            let eq_polys = lhs_points
                .iter()
                .map(|split_point| {
                    let eq_point = reconstruct_full_point(
                        split_point,
                        &einsum_sumcheck_point,
                        input_shapes[0].as_slice(),
                    );
                    let evaluations = compute_betas_eval(&eq_point);
                    MultilinearExtension::from_evaluations_ext_vec(num_vars, evaluations)
                })
                .collect::<Vec<MultilinearExtension<E>>>();

            let input_batching_challenge = transcript
                .sample_and_append_challenge(b"input_batching_challenge")
                .elements;
            let num_threads = optimal_sumcheck_threads(num_vars);
            let either_mles = std::iter::once(&input_mle)
                .chain(eq_polys.iter())
                .map(Either::Left)
                .collect::<Vec<_>>();
            let expr_builder =
                VirtualPolynomialsBuilder::<E>::new_with_mles(num_threads, num_vars, either_mles);
            let virtual_poly = expr_builder
                .to_virtual_polys(std::slice::from_ref(agg_expr), &[input_batching_challenge]);
            let (agg_proof, agg_state) = IOPProverState::<E>::prove(virtual_poly, transcript);

            let lhs_input_point = agg_state.collect_raw_challenges();
            let lhs_input_eval = agg_state.get_mle_flatten_final_evaluations()[0];

            let einsum_proof = EinSumProof {
                bias_evals,
                einsum_sumcheck: einsum_proof,
                einsum_evaluations,
                input_aggregation_sumcheck: Some(agg_proof),
            };

            let lhs_input_claim = Claim::new(lhs_input_point, lhs_input_eval);
            input_claims.insert(0, lhs_input_claim);

            Ok(EinSumProofInfo::new(
                input_claims,
                einsum_proof,
                constant_poly_claims,
            ))
        } else {
            let lhs_input_point =
                reconstruct_full_point(&lhs_points[0], &einsum_sumcheck_point, &input_shapes[0]);
            let lhs_input_eval = einsum_evaluations[0];
            let lhs_input_claim = Claim::new(lhs_input_point, lhs_input_eval);

            let einsum_proof = EinSumProof {
                bias_evals,
                einsum_sumcheck: einsum_proof,
                einsum_evaluations,
                input_aggregation_sumcheck: None,
            };

            input_claims.insert(0, lhs_input_claim);

            Ok(EinSumProofInfo::new(
                input_claims,
                einsum_proof,
                constant_poly_claims,
            ))
        }
    }
}

/// Function used to reconstruct the full evaluation point of a tensor MLE from the fixed axes and the sumcheck point
pub(crate) fn reconstruct_full_point<E: ExtensionField>(
    fixed_axes: &[FixedAxis<&[E]>],
    sumcheck_point: &[E],
    input_shape: &[usize],
) -> Vec<E> {
    let mut skip = 0usize;
    fixed_axes
        .iter()
        .zip(input_shape.iter())
        .rev()
        .flat_map(|(&p_opt, &input_dim)| {
            // To make the point we have to take and insert the unfixed variables at the correct locations
            match p_opt {
                FixedAxis::Outer(point) | FixedAxis::Stacked(point) => point,
                FixedAxis::Contracted => {
                    let dim_log = ceil_log2(input_dim);
                    let point = &sumcheck_point[skip..skip + dim_log];
                    skip += dim_log;
                    point
                }
            }
        })
        .copied()
        .collect::<Vec<E>>()
}
