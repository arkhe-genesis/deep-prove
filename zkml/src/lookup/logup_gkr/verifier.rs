//! Contains code for verifying a LogUpProof

use std::collections::{BTreeSet, HashMap};

use ceno_p3::field::FieldAlgebra;
use ff_ext::ExtensionField;
use itertools::Itertools;
use multilinear_extensions::{Expression, utils::eval_by_expr_with_instance};
use sumcheck::structs::IOPVerifierState;
use transcript::Transcript;

use crate::{
    commit::identity_eval, lookup::logup_gkr::circuit::construct_final_round_claim_expression,
};

use super::{
    circuit::construct_logup_expressions,
    error::LogUpError,
    structs::{LogUpBatchProof, LogUpBatchVerifierClaim, ProofType},
};

/// Function to verify a [`LogUpBatchProof`] that has been produced by [`super::prover::batch_multiple_sizes_prove`].
pub fn verify_logup_proof_multiple_sizes<E: ExtensionField, T: Transcript<E>>(
    proof: &LogUpBatchProof<E>,
    transcript: &mut T,
) -> Result<LogUpBatchVerifierClaim<E>, LogUpError> {
    // Append the number of instances along with their output evals to the transcript and then squeeze our first alpha and lambda
    transcript.append_field_element(&E::BaseField::from_canonical_u64(
        proof.num_instances() as u64
    ));
    proof.append_to_transcript(transcript);
    // Extract the numerators and denominators
    let (numerators, denominators): (Vec<E>, Vec<E>) = proof.fractional_outputs();

    // Make the maps that store inital evaluations of each circuit by the size of the circuit
    let output_evals = proof
        .num_vars_per_instance
        .iter()
        .copied()
        .zip(proof.circuit_outputs().iter().cloned())
        .into_group_map();
    // This stores how many different layer sizes there are
    let unique_layer_size: BTreeSet<usize> = proof.num_vars_per_instance.iter().copied().collect();
    let total_layers = proof.sumcheck_proofs.len();

    // Construct the exrpessions used, `sumcheck_layer_exprs` are used to verify the `subclaim`s output by the Sumcheck verification,
    // `claim_layer_exprs` are used to link those evaluations to the `claimed_sum` of the next sumcheck round.
    let (sumcheck_layer_exprs, claim_layer_exprs) =
        construct_logup_expressions::<E>(&unique_layer_size, &output_evals, total_layers);

    // Squeeze the first challenges
    let mut batching_challenge = transcript
        .sample_and_append_challenge(b"initial_batching")
        .elements;
    let mut alpha = transcript
        .sample_and_append_challenge(b"initial_alpha")
        .elements;
    let mut lambda = transcript
        .sample_and_append_challenge(b"initial_lambda")
        .elements;

    let mut current_claim: E;
    let mut final_claim_eval = E::ZERO;

    // The initial sumcheck point is just the batching challenge
    let mut sumcheck_point: Vec<E> = vec![batching_challenge];

    for (i, (sumcheck_proof, round_evaluations)) in proof.proofs_and_evals().enumerate() {
        let remaining_layers = total_layers - i;
        // Work out if we have to include intial evaluations from one of the smaller circuits in this round.
        // If we do extend the prveious round_evaluations by these values.
        let wit_evals = if i != 0 {
            let new_evals = output_evals.get(&remaining_layers);
            if let Some(new_evals) = new_evals {
                proof.round_evaluations[i - 1]
                    .iter()
                    .chain(new_evals.iter().flatten())
                    .copied()
                    .collect::<Vec<E>>()
            } else {
                proof.round_evaluations[i - 1].to_vec()
            }
        } else {
            let new_evals =
                output_evals
                    .get(&remaining_layers)
                    .ok_or(LogUpError::ParameterError(
                        format!("No previous evals and no evals for this number of variables: {remaining_layers}")
                    ))?;
            new_evals.iter().flatten().copied().collect::<Vec<E>>()
        };
        // The Expressions are grouped by the size of the circuit they refer to, so we need to work out how many of the
        // circuits are running in this round, this is just all the entries of `unique_layer_size` that are larger than `remaining_layers`.
        let num_expressions = unique_layer_size
            .iter()
            .filter_map(|&size| {
                if size >= remaining_layers {
                    Some(1)
                } else {
                    None
                }
            })
            .sum::<usize>();
        // Calculate the current claim
        current_claim = claim_layer_exprs
            .iter()
            .take(num_expressions)
            .map(|c_expr| {
                eval_by_expr_with_instance(
                    &[],
                    &wit_evals,
                    &[],
                    &[],
                    &[alpha, lambda, batching_challenge],
                    c_expr,
                )
                .right()
            })
            .try_fold(E::ZERO, |acc, opt_eval| opt_eval.map(|inner| acc + inner))
            .ok_or(LogUpError::ParameterError(
                "Couldn't sum claim expressions".to_string(),
            ))?;
        // Append the current claim to the transcript
        transcript.append_field_element_ext(&current_claim);

        // Run this rounds sumcheck verification
        let current_num_vars = i + 1;
        let aux_info = crate::util::from_mle_list_dimensions(&[vec![current_num_vars; 3]]);
        let sumcheck_subclaim =
            IOPVerifierState::<E>::verify(current_claim, sumcheck_proof, &aux_info, transcript);

        let current_point = sumcheck_subclaim
            .point
            .iter()
            .map(|c| c.elements)
            .collect::<Vec<E>>();
        // Calculate the eq_poly evaluations for this round
        let eq_evals = unique_layer_size
            .iter()
            .rev()
            .filter_map(|&size| {
                if size >= remaining_layers {
                    let diff = total_layers - size;
                    Some(identity_eval(
                        &sumcheck_point[diff..],
                        &current_point[diff..],
                    ))
                } else {
                    None
                }
            })
            .collect::<Vec<E>>();

        batching_challenge = transcript
            .sample_and_append_challenge(b"logup_batching")
            .elements;

        // Now we take the round evals and check their consistency with the sumcheck claim, provided that this isn't the final round
        // and the proof type isn't a lookup
        let calc_sumcheck_eval = if (i + 1) != total_layers {
            let sumcheck_exprs = sumcheck_layer_exprs
                .iter()
                .enumerate()
                .take(eq_evals.len())
                .map(|(i, expr)| {
                    expr.clone() * Expression::WitIn((round_evaluations.len() + i) as u16)
                })
                .collect::<Vec<Expression<E>>>();
            let all_evals = round_evaluations
                .iter()
                .copied()
                .chain(eq_evals)
                .collect::<Vec<E>>();
            let calc_sumcheck_claim = sumcheck_exprs
                .iter()
                .map(|s_expr| {
                    eval_by_expr_with_instance(&[], &all_evals, &[], &[], &[alpha, lambda], s_expr)
                        .right()
                })
                .try_fold(E::ZERO, |acc, opt_eval| opt_eval.map(|inner| acc + inner))
                .ok_or(LogUpError::ParameterError(
                    "Couldn't sum claim expressions".to_string(),
                ))?;
            // Squeeze the challenges to combine everything into a single sumcheck
            alpha = transcript
                .sample_and_append_challenge(b"logup_alpha")
                .elements;
            lambda = transcript
                .sample_and_append_challenge(b"logup_lambda")
                .elements;
            calc_sumcheck_claim
        } else {
            let eq_offset = match proof.proof_type {
                ProofType::Lookup => 2 * round_evaluations.len(),
                ProofType::Table => round_evaluations.len(),
            };
            let sumcheck_exprs = sumcheck_layer_exprs
                .iter()
                .enumerate()
                .take(eq_evals.len())
                .map(|(i, expr)| expr.clone() * Expression::WitIn((eq_offset + i) as u16))
                .collect::<Vec<Expression<E>>>();
            let (calc_sumcheck_eval, final_claim) = verify_final_eval_batch(
                proof,
                round_evaluations,
                &eq_evals,
                &unique_layer_size,
                &output_evals,
                alpha,
                lambda,
                batching_challenge,
                &sumcheck_exprs,
            )?;
            final_claim_eval = final_claim;
            calc_sumcheck_eval
        };

        if calc_sumcheck_eval != sumcheck_subclaim.expected_evaluation {
            return Err(LogUpError::VerifierError(format!(
                "Calculated sumcheck claim: {:?} does not equal this rounds sumcheck output claim: {:?} at round: {}",
                calc_sumcheck_eval, sumcheck_subclaim.expected_evaluation, i
            )));
        }

        sumcheck_point = current_point;
        sumcheck_point.push(batching_challenge);
    }

    Ok(LogUpBatchVerifierClaim::<E>::new(
        final_claim_eval,
        sumcheck_point,
        proof
            .output_claims()
            .iter()
            .map(|c| c.eval)
            .collect::<Vec<E>>(),
        alpha,
        lambda,
        numerators,
        denominators,
    ))
}

#[allow(clippy::too_many_arguments)]
/// Function used to perform the final verification for multiple logup instances of multiple sizes.
fn verify_final_eval_batch<E: ExtensionField>(
    proof: &LogUpBatchProof<E>,
    final_round_evaluations: &[E],
    eq_evals: &[E],
    variables_set: &BTreeSet<usize>,
    evals_set: &HashMap<usize, Vec<Vec<E>>>,
    alpha: E,
    lambda: E,
    batching_challenge: E,
    sumcheck_exprs: &[Expression<E>],
) -> Result<(E, E), LogUpError> {
    // If we a re verifying lookup instances we can just append `-1` to the evaluations rather than construct a whole new expression.
    let evals_for_sumcheck = match proof.proof_type {
        ProofType::Lookup => final_round_evaluations
            .chunks(2)
            .flat_map(|chunk| [-E::ONE, -E::ONE, chunk[0], chunk[1]])
            .chain(eq_evals.iter().copied())
            .collect::<Vec<E>>(),
        ProofType::Table => final_round_evaluations
            .iter()
            .chain(eq_evals.iter())
            .copied()
            .collect::<Vec<E>>(),
    };
    // Calculate the sumcheck evaluation so it can be checked against the subclaim.
    let sumcheck_eval = sumcheck_exprs
        .iter()
        .map(|s_expr| {
            eval_by_expr_with_instance(&[], &evals_for_sumcheck, &[], &[], &[alpha, lambda], s_expr)
                .right()
        })
        .try_fold(E::ZERO, |acc, opt_eval| opt_eval.map(|inner| acc + inner))
        .ok_or(LogUpError::ParameterError(
            "Couldn't sum claim expressions".to_string(),
        ))?;
    // Construct the final claim, this is a claim about the polynomials in the base layer of the GKR circuit (so relates to the merged columns)
    let claim_exprs =
        construct_final_round_claim_expression(variables_set, evals_set, proof.proof_type);
    let claim_eval = claim_exprs
        .iter()
        .map(|s_expr| {
            eval_by_expr_with_instance(
                &[],
                final_round_evaluations,
                &[],
                &[],
                &[alpha, lambda, batching_challenge],
                s_expr,
            )
            .right()
        })
        .try_fold(E::ZERO, |acc, opt_eval| opt_eval.map(|inner| acc + inner))
        .ok_or(LogUpError::ParameterError(
            "Couldn't sum claim expressions".to_string(),
        ))?;
    Ok((sumcheck_eval, claim_eval))
}
