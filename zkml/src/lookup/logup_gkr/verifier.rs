//! Contains code for verifying a LogUpProof

use std::collections::BTreeSet;

use anyhow::{Result, anyhow};
use ark_ff::PrimeField;
use dp_crypto::{
    Expression, arkyper::transcript::Transcript, structs::IOPVerifierState,
    utils::eval_by_expr_with_instance,
};
use itertools::Itertools;

use crate::{
    Claim,
    lookup::logup_gkr::structs::{InstanceExpressions, LogUpVerifierInstance},
    poly_commit::identity_eval,
};

use super::structs::{LogUpBatchProof, LogUpBatchVerifierClaim};

struct PreppedLayerVerifyingInfo<F: PrimeField> {
    eq_points: Vec<Vec<F>>,
    sumcheck_expressions: Vec<Expression<F>>,
    claim_expression: Expression<F>,
}

impl<F: PrimeField> PreppedLayerVerifyingInfo<F> {
    pub fn new() -> Self {
        Self {
            eq_points: vec![],
            sumcheck_expressions: vec![],
            claim_expression: Expression::ZERO,
        }
    }

    pub fn compute_current_claim(
        &self,
        wit_evals: &[F],
        alpha: F,
        lambda: F,
        batching_challenge: F,
    ) -> F {
        eval_by_expr_with_instance(
            &[],
            wit_evals,
            &[],
            &[],
            &[alpha, lambda, batching_challenge],
            &self.claim_expression,
        )
    }

    pub fn full_sumcheck_expression(&self, witness_offset: u16) -> Expression<F> {
        self.sumcheck_expressions
            .iter()
            .enumerate()
            .fold(Expression::ZERO, |acc, (i, expr)| {
                acc + expr.clone() * Expression::WitIn(witness_offset + i as u16)
            })
    }

    pub fn eq_evaluations(&self, current_point: &[F]) -> Vec<F> {
        let current_point_len = current_point.len();
        self.eq_points
            .iter()
            .map(|eq_point| {
                identity_eval(
                    &current_point[current_point_len - eq_point.len()..],
                    eq_point,
                )
            })
            .collect()
    }
}

/// Function to verify a [`LogUpBatchProof`] that has been produced by [`super::prover::batch_multiple_sizes_prove`].
pub fn new_verify_logup_proof_multiple_sizes<F: PrimeField, T: Transcript>(
    proof: &LogUpBatchProof<F>,
    instances: &[LogUpVerifierInstance<F>],
    transcript: &mut T,
) -> Result<LogUpBatchVerifierClaim<F>> {
    // Append the number of instances along with their output evals to the transcript and then squeeze our first alpha and lambda
    let num_instances = instances.len();
    transcript.append_scalars(&[F::from(num_instances as u64)]);
    proof.append_to_transcript(transcript);
    // Extract the numerators and denominators
    let (numerators, denominators): (Vec<F>, Vec<F>) = proof.fractional_outputs();

    // Make the maps that store initial evaluations of each circuit by the size of the circuit
    let output_evals = instances
        .iter()
        .map(|instance| instance.num_vars())
        .zip(proof.circuit_outputs().iter().cloned())
        .into_group_map();
    // This stores how many different layer sizes there are
    let unique_layer_size: BTreeSet<usize> = instances
        .iter()
        .map(|instance| instance.num_vars())
        .collect();
    let total_layers = proof.sumcheck_proofs.len();

    // Squeeze the first challenges
    let mut batching_challenge = transcript.append_and_sample(b"initial_batching");
    let mut alpha = transcript.append_and_sample(b"initial_alpha");
    let mut lambda = transcript.append_and_sample(b"initial_lambda");

    let mut current_claim: F;

    // The initial sumcheck point is just the batching challenge
    let mut sumcheck_point: Vec<F> = vec![batching_challenge];

    for (i, (sumcheck_proof, round_evaluations)) in proof.proofs_and_evals().enumerate() {
        let remaining_layers = total_layers - i;
        // Work out if we have to include initial evaluations from one of the smaller circuits in this round.
        // If we do extend the prveious round_evaluations by these values.
        let wit_evals = if i != 0 {
            let new_evals = output_evals.get(&remaining_layers);
            if let Some(new_evals) = new_evals {
                proof.round_evaluations[i - 1]
                    .iter()
                    .chain(new_evals.iter().flatten())
                    .copied()
                    .collect::<Vec<F>>()
            } else {
                proof.round_evaluations[i - 1].to_vec()
            }
        } else {
            let new_evals = output_evals.get(&remaining_layers).ok_or(anyhow!(
                "No previous evals and no evals for this number of variables: {remaining_layers}"
            ))?;
            new_evals.iter().flatten().copied().collect::<Vec<F>>()
        };
        // The Expressions are grouped by the size of the circuit they refer to, so we need to work out how many of the
        // circuits are running in this round, this is just all the entries of `unique_layer_size` that are larger than `remaining_layers`.
        let mut layer_count = 0usize;
        let mut witness_offset = 0u16;
        let point_length = sumcheck_point.len();
        let verifying_info = unique_layer_size.iter().rev().fold(
            PreppedLayerVerifyingInfo::new(),
            |mut acc, &size| {
                if size >= remaining_layers {
                    // This multiplier is needed to scale the claim expressions from each layer size correctly.
                    // This is because smaller instances have fewer layers and so their claims need to be scaled
                    // by 1 << (total_layers - size) to line up with the largest instances.
                    let claim_expr_multiplier =
                        Expression::Constant(F::from(1u64 << (total_layers - size)));

                    let mut sumcheck_expr = Expression::ZERO;
                    let current_var_count: usize = size + 1 - remaining_layers;

                    instances.iter().for_each(|instance| {
                        if instance.num_vars() == size {
                            let InstanceExpressions {
                                sumcheck_expression,
                                claim_expression,
                            } = instance.instance_expressions(
                                &mut layer_count,
                                &mut witness_offset,
                                (i + 1) == total_layers,
                            );

                            // Add this layer's sumcheck expression and claim expression
                            sumcheck_expr = sumcheck_expr.clone() + sumcheck_expression;
                            acc.claim_expression = acc.claim_expression.clone()
                                + claim_expr_multiplier.clone() * claim_expression;
                        }
                    });

                    // Now compute the EQ polynomial for this layer size and add it to the proving info
                    let eq_point = sumcheck_point[point_length - current_var_count..].to_vec();

                    acc.eq_points.push(eq_point);
                    acc.sumcheck_expressions.push(sumcheck_expr);
                }
                acc
            },
        );
        // Compute the current_claim
        current_claim =
            verifying_info.compute_current_claim(&wit_evals, alpha, lambda, batching_challenge);

        // Append the current claim to the transcript
        transcript.append_scalars(&[current_claim]);

        // Run this rounds sumcheck verification
        let current_num_vars = i + 1;
        let aux_info = crate::util::from_mle_list_dimensions(&[vec![current_num_vars; 3]]);
        let sumcheck_subclaim =
            IOPVerifierState::<F>::verify(current_claim, sumcheck_proof, &aux_info, transcript);

        let current_point = &sumcheck_subclaim.point;

        batching_challenge = transcript.append_and_sample(b"logup_batching");

        let full_sumcheck_expression = verifying_info.full_sumcheck_expression(witness_offset);
        let eq_evaluations = verifying_info.eq_evaluations(current_point);

        let full_evaluations = round_evaluations
            .iter()
            .copied()
            .chain(eq_evaluations)
            .collect::<Vec<F>>();

        let calc_sumcheck_eval = eval_by_expr_with_instance(
            &[],
            &full_evaluations,
            &[],
            &[],
            &[alpha, lambda],
            &full_sumcheck_expression,
        );

        if (i + 1) != total_layers {
            // Squeeze the challenges to combine everything into a single sumcheck
            alpha = transcript.append_and_sample(b"logup_alpha");
            lambda = transcript.append_and_sample(b"logup_lambda");
        }

        if calc_sumcheck_eval != sumcheck_subclaim.expected_evaluation {
            return Err(anyhow!(
                "LogUp Verifier error: Calculated sumcheck claim: {:?} does not equal this rounds sumcheck output claim: {:?} at round: {}",
                calc_sumcheck_eval,
                sumcheck_subclaim.expected_evaluation,
                i
            ));
        }

        sumcheck_point = current_point.clone();
        sumcheck_point.push(batching_challenge);
    }

    let final_round_evaluations = proof
        .round_evaluations
        .last()
        .ok_or(anyhow!("LogUp verifier: No final round evaluations found"))?;
    let mut skip = 0usize;
    let output_claims = instances.iter().try_fold(Vec::new(), |mut acc, instance| {
        let evals_slice = &final_round_evaluations[skip..skip + instance.num_final_round_evals()];
        skip += instance.num_final_round_evals();
        let eval = instance.final_claim(evals_slice, batching_challenge, &sumcheck_point)?;
        acc.extend(eval);
        Result::<Vec<Claim<F>>>::Ok(acc)
    })?;
    Ok(LogUpBatchVerifierClaim::<F>::new(
        F::ZERO,
        output_claims,
        sumcheck_point,
        proof
            .output_claims()
            .iter()
            .map(|c| c.eval)
            .collect::<Vec<F>>(),
        numerators,
        denominators,
    ))
}
