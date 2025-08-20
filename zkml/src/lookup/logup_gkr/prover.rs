//! Contains the code for batch proving a number of LogUp GKR claims.

use std::collections::BTreeSet;

use ceno_p3::field::FieldAlgebra;
use either::Either;
use ff_ext::ExtensionField;
use itertools::Itertools;

use multilinear_extensions::{
    Expression,
    mle::{IntoMLE, MultilinearExtension},
    utils::eval_by_expr_with_instance,
    virtual_polys::VirtualPolynomialsBuilder,
};
use sumcheck::{structs::IOPProverState, util::optimal_sumcheck_threads};
use transcript::Transcript;

use crate::{Claim, commit::compute_betas_eval};

use super::{
    circuit::{LogUpCircuit, construct_final_round_logup_expressions, construct_logup_expressions},
    error::LogUpError,
    structs::{LogUpBatchProof, LogUpInput, ProofType},
};

/// Function that proves multiple LogUp instances of the same type, regardless of size and table type.
/// `input` - A list of [`LogUpInput`] that are all the same variant
/// `transcript` - an implementor of [`Transcript`]
///
/// The function works because after the different challenges have been applied to the columns all LogUp instances look the same.
/// Hence in each round of the GKR we can prove all the sumchecks together, reducing the proof size and the verificaiton cost.
///
/// To handle instances of differeing size we require that the values in `input` are ordered in a decreasing number of variables. Then
/// the largest instances begin proving in the first round with smaller instances being "rolled in" at the correct point so that every instance
/// finishes proving in the very last round.
pub fn batch_multiple_sizes_prove<E: ExtensionField, T: Transcript<E>>(
    input: &[LogUpInput<E>],
    transcript: &mut T,
) -> Result<LogUpBatchProof<E>, LogUpError> {
    let first_input = input.first().ok_or(LogUpError::ParameterError(
        "No inputs provided for LogUp".to_string(),
    ))?;
    let first_proof_type = match first_input {
        LogUpInput::Lookup { .. } => ProofType::Lookup,
        LogUpInput::Table { .. } => ProofType::Table,
    };
    // The proof type, we make sure every instance is the same variant
    let proof_type = input
        .iter()
        .skip(1)
        .map(|l_in| match l_in {
            LogUpInput::Lookup { .. } => ProofType::Lookup,
            LogUpInput::Table { .. } => ProofType::Table,
        })
        .try_fold(first_proof_type, |acc, f| {
            if acc == f {
                Ok(acc)
            } else {
                Err(LogUpError::ParameterError(
                    "Not all proof types matched".to_string(),
                ))
            }
        })?;

    // Work out how many instances we are dealing with and make the individual circuits
    let circuits = input
        .iter()
        .flat_map(|l| l.make_circuits())
        .collect::<Vec<LogUpCircuit<E>>>();
    let num_instances = circuits.len();

    // We make sure that the circuits decrease in size as the list goes on
    let mut total_layers = 0usize;
    if circuits.len() > 1 {
        circuits.windows(2).try_for_each(|window| {
            let first_vars = window[0].num_vars();
            let second_vars = window[1].num_vars();
            if first_vars < second_vars {
                Err(LogUpError::ParameterError(
                    "Circuits were not in decreasing size order".to_string(),
                ))
            } else {
                total_layers = std::cmp::max(total_layers, first_vars);
                Ok(())
            }
        })?;
    } else {
        total_layers = circuits[0].num_vars();
    }

    // Now for each distinct circuit size we build the expressions and layer iters
    let unique_layer_size: BTreeSet<usize> = circuits.iter().map(|c| c.num_vars()).collect();
    let mut iters_by_var_count = circuits
        .iter()
        .map(|c| (c.num_vars(), c.layers().iter().rev().skip(1)))
        .into_group_map();
    let output_evals = circuits
        .iter()
        .map(|c| (c.num_vars(), c.outputs()))
        .into_group_map();

    // Build the expressions that will be used throughout, `sumcheck_layers_expr` is the `Expression` used in the sumcheck,
    // `claim_layers_exprs` are the `Expression`s used to link the layers together.
    let (sumcheck_layer_exprs, claim_layer_exprs) =
        construct_logup_expressions(&unique_layer_size, &output_evals, total_layers);

    // Extract all the circuit outputs
    let circuit_outputs = circuits
        .iter()
        .map(|c| c.outputs())
        .collect::<Vec<Vec<E>>>();

    // When proving we want to work from the top down so we convert each of the circuits into an iterator over its layers in reverse order.
    // We also skip the first layer after reversing as this is just the output claims.

    // Append the number of instances along with their output evals to the transcript and then squeeze our first batching_challenge, alpha and lambda
    transcript.append_field_element(&E::BaseField::from_canonical_u64(num_instances as u64));
    circuit_outputs
        .iter()
        .for_each(|evals| transcript.append_field_element_exts(evals));

    // The batching_challenge is used because at each layer we have one claim about the "low" half of a polynomial and one claim about the "high" half
    // they can be combined together to get that `f(r1,...,rk-1,rk) = rk * f(r1,...,rk-1, 1) + (1 - rk) * f(r1,...,rk-1, 0)`. The Sumcheck gives us the challenges `r1,...,rk-1`
    // and then `rk` is the `batching_challenge`.
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
    // The initial sumcheck point is just the batching challenge as in the first round the polynomials are univariate
    let mut sumcheck_point: Vec<E> = vec![batching_challenge];

    let mut sumcheck_proofs = vec![];

    let mut round_evaluations: Vec<Vec<E>> = vec![];

    for current_layer_vars in 1..=total_layers {
        let remaining_layers = total_layers - (current_layer_vars - 1);
        // Here we check to see if any of the smaller instances need to be folded in this round.
        // If so we extend the last set of `round_evaluations` by the inital evaluations of the instances that are about to start proving.
        let wit_evals = if let Some(evals) = round_evaluations.last() {
            let new_evals = output_evals.get(&remaining_layers);
            if let Some(new_evals) = new_evals {
                evals
                    .iter()
                    .chain(new_evals.iter().flatten())
                    .copied()
                    .collect::<Vec<E>>()
            } else {
                evals.to_vec()
            }
        } else {
            let new_evals =
                output_evals
                    .get(&remaining_layers)
                    .ok_or(LogUpError::ParameterError(
                        "No previous evals and no evals for this number of variables".to_string(),
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

        let sumcheck_exprs = &sumcheck_layer_exprs[..num_expressions];

        // Then add all the terms to the sumcheck virtual polynomial
        let num_threads = optimal_sumcheck_threads(current_layer_vars);
        // `current_vars_count` is used so that we can construct all the different sized EQ polynomials as the sumcheck prover
        // does not allow products to include MLEs with differeing numbers of variables.
        let mut current_vars_counts = BTreeSet::<usize>::new();
        let mles = unique_layer_size
            .iter()
            .rev()
            .filter_map(|&size| {
                if size >= remaining_layers {
                    let layer_iter = iters_by_var_count.get_mut(&size).unwrap();
                    let polys_vec = layer_iter
                        .iter_mut()
                        .flat_map(|iter| {
                            let layer = iter.next().unwrap();
                            current_vars_counts.insert(layer.num_vars());
                            layer.new_get_mles()
                        })
                        .collect::<Vec<MultilinearExtension<E>>>();
                    Some(polys_vec)
                } else {
                    None
                }
            })
            .flatten()
            .collect::<Vec<MultilinearExtension<E>>>();

        let either_mles = mles.iter().map(Either::Left).collect::<Vec<Either<_, _>>>();
        let mut expr_builder = VirtualPolynomialsBuilder::<E>::new_with_mles(
            num_threads,
            current_layer_vars,
            either_mles,
        );
        let point_length = sumcheck_point.len();
        let layer_eq_polys = current_vars_counts
            .iter()
            .rev()
            .map(|&vars| compute_betas_eval(&sumcheck_point[point_length - vars..]).into_mle())
            .collect::<Vec<MultilinearExtension<E>>>();

        // If we are proving lookups, rather than tables, and its the final round then we have to use a different sumcheck expression
        let virtual_poly = if proof_type == ProofType::Lookup && current_layer_vars == total_layers
        {
            // If its the final layer and its a lookup proof then the expression changes
            let sumcheck_exprs =
                construct_final_round_logup_expressions(&unique_layer_size, &output_evals);
            let layer_exprs = sumcheck_exprs
                .iter()
                .zip(layer_eq_polys.iter())
                .map(|(sc, poly)| sc.clone() * expr_builder.lift(Either::Left(poly)))
                .collect::<Vec<Expression<E>>>();
            expr_builder.to_virtual_polys(&layer_exprs, &[alpha, lambda])
        } else {
            let layer_exprs = sumcheck_exprs
                .iter()
                .zip(layer_eq_polys.iter())
                .map(|(sc, poly)| sc.clone() * expr_builder.lift(Either::Left(poly)))
                .collect::<Vec<Expression<E>>>();
            expr_builder.to_virtual_polys(&layer_exprs, &[alpha, lambda])
        };

        let (proof, state) = IOPProverState::<E>::prove(virtual_poly, transcript);

        // Update the sumcheck point
        sumcheck_point = state.collect_raw_challenges();

        // Extract all the evaluations apart from the EQ polys
        let evals = state.get_mle_flatten_final_evaluations()[..mles.len()].to_vec();

        // Squeeze the challenges to combine everything into a single sumcheck
        batching_challenge = transcript
            .sample_and_append_challenge(b"logup_batching")
            .elements;
        // If its the final round we don't need to squeeze additional `alpha` and `lambda`
        if current_layer_vars != total_layers {
            alpha = transcript
                .sample_and_append_challenge(b"logup_alpha")
                .elements;
            lambda = transcript
                .sample_and_append_challenge(b"logup_lambda")
                .elements;
        }
        // Append the batching challenge to the proof point
        sumcheck_point.push(batching_challenge);
        // Append the sumcheck proof to the list of proofs
        sumcheck_proofs.push(proof);

        // Append the claimed evaluations from the end of this round to the proof.
        round_evaluations.push(evals);
    }

    // We take the final sumcheck point and produce a list of claims about all the columns looked up/ in the table and
    // also the multiplicity polynomial in the table case. These will be used by the verifier to check the final sumcheck proofs claim.
    // Then each of these claims should be verified either by another layer proof or via commitment opening proof.
    let point_length = sumcheck_point.len();
    let output_claims = input
        .iter()
        .flat_map(|li| {
            li.base_mles()
                .iter()
                .map(|mle| {
                    let num_vars = mle.num_vars();
                    let eval = mle.evaluate(&sumcheck_point[point_length - num_vars..]);
                    Claim::<E>::new(sumcheck_point[point_length - num_vars..].to_vec(), eval)
                })
                .collect::<Vec<Claim<E>>>()
        })
        .collect::<Vec<Claim<E>>>();

    Ok(LogUpBatchProof::<E> {
        sumcheck_proofs,
        round_evaluations,
        output_claims,
        circuit_outputs,
        proof_type,
        num_vars_per_instance: circuits
            .iter()
            .map(|c| c.num_vars())
            .collect::<Vec<usize>>(),
    })
}
