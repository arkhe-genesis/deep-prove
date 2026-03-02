//! Contains the code for batch proving a number of LogUp GKR claims.

use std::collections::BTreeSet;

use anyhow::{Result, anyhow};
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

use crate::{Claim, commit::compute_betas_eval, lookup::logup_gkr::circuit::MLEsAndExpressions};

use super::{
    circuit::LogUpCircuit,
    structs::{LogUpBatchProof, LogUpInput, ProofType},
};

struct PreppedLayerProvingInfo<'a, E: ExtensionField> {
    witness_mles_flat: Vec<MultilinearExtension<'a, E>>,
    eq_mles: Vec<MultilinearExtension<'a, E>>,
    sumcheck_expressions: Vec<Expression<E>>,
    claim_expression: Expression<E>,
}

impl<E: ExtensionField> PreppedLayerProvingInfo<'_, E> {
    pub fn new() -> Self {
        Self {
            witness_mles_flat: vec![],
            eq_mles: vec![],
            sumcheck_expressions: vec![],
            claim_expression: Expression::ZERO,
        }
    }

    pub fn compute_current_claim(
        &self,
        wit_evals: &[E],
        alpha: E,
        lambda: E,
        batching_challenge: E,
    ) -> Result<E> {
        let claim = eval_by_expr_with_instance(
            &[],
            wit_evals,
            &[],
            &[],
            &[alpha, lambda, batching_challenge],
            &self.claim_expression,
        )
        .right();
        claim.ok_or(anyhow!(
            "Couldn't compute current claim in LogUp proving: claim value was not in the extension field"
        ))
    }

    pub fn full_sumcheck_expression(&self) -> Expression<E> {
        let witness_offset = self.witness_mles_flat.len() as u16;
        self.sumcheck_expressions
            .iter()
            .enumerate()
            .fold(Expression::ZERO, |acc, (i, expr)| {
                acc + expr.clone() * Expression::WitIn(witness_offset + i as u16)
            })
    }

    pub fn either_mles(
        &self,
    ) -> Vec<Either<&'_ MultilinearExtension<'_, E>, &'_ mut MultilinearExtension<'_, E>>> {
        self.witness_mles_flat
            .iter()
            .chain(self.eq_mles.iter())
            .map(Either::Left)
            .collect()
    }
}

/// Function that proves multiple LogUp instances of the same type, regardless of size and table type.
/// `input` - A list of [`LogUpInput`] that are all the same variant
/// `transcript` - an implementor of [`Transcript`]
///
/// The function works because after the different challenges have been applied to the columns all LogUp instances look the same.
/// Hence in each round of the GKR we can prove all the sumchecks together, reducing the proof size and the verification cost.
///
/// To handle instances of differeing size we require that the values in `input` are ordered in a decreasing number of variables. Then
/// the largest instances begin proving in the first round with smaller instances being "rolled in" at the correct point so that every instance
/// finishes proving in the very last round.
pub fn new_batch_multiple_sizes_prove<E: ExtensionField, T: Transcript<E>>(
    input: &[LogUpInput<E>],
    transcript: &mut T,
) -> Result<LogUpBatchProof<E>> {
    let first_input = input
        .first()
        .ok_or(anyhow!("No inputs provided for LogUp"))?;
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
                Err(anyhow!("Not all proof types matched"))
            }
        })?;

    // Work out how many instances we are dealing with and make the individual circuits
    let circuits = input
        .iter()
        .flat_map(|l| l.make_new_circuits())
        .collect::<Vec<LogUpCircuit<E>>>();
    let num_instances = circuits.len();

    // We make sure that the circuits decrease in size as the list goes on
    let mut total_layers = 0usize;
    if circuits.len() > 1 {
        circuits.windows(2).try_for_each(|window| {
            let first_vars = window[0].num_vars();
            let second_vars = window[1].num_vars();
            if first_vars < second_vars {
                Err(anyhow!("Circuits were not in decreasing size order"))
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
        // If so we extend the last set of `round_evaluations` by the initial evaluations of the instances that are about to start proving.
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
            let new_evals = output_evals.get(&remaining_layers).ok_or(anyhow!(
                "Proving failed: No previous evals and no evals for this number of variables"
            ))?;
            new_evals.iter().flatten().copied().collect::<Vec<E>>()
        };
        // The Expressions are grouped by the size of the circuit they refer to, so we need to work out how many of the
        // circuits are running in this round, this is just all the entries of `unique_layer_size` that are larger than `remaining_layers`.

        // Then add all the terms to the sumcheck virtual polynomial
        let num_threads = optimal_sumcheck_threads(current_layer_vars);
        // `current_vars_count` is used so that we can construct all the different sized EQ polynomials as the sumcheck prover
        // does not allow products to include MLEs with differeing numbers of variables.
        let mut layer_count = 0usize;
        let mut witness_offset = 0u16;
        let point_length = sumcheck_point.len();
        let proving_info = unique_layer_size.iter().rev().fold(
            PreppedLayerProvingInfo::new(),
            |mut acc, &size| {
                if size >= remaining_layers {
                    // This multiplier is needed to scale the claim expressions from each layer size correctly.
                    // This is because smaller instances have fewer layers and so their claims need to be scaled
                    // by 1 << (total_layers - size) to line up with the largest instances.
                    let claim_expr_multiplier = Expression::Constant(Either::Right(
                        E::from_canonical_u64(1 << (total_layers - size)),
                    ));
                    let layer_iter = iters_by_var_count.get_mut(&size).unwrap();
                    let mut sumcheck_expr = Expression::ZERO;
                    let mut current_var_count: usize = 0;

                    layer_iter.iter_mut().for_each(|iter| {
                        let layer = iter.next().unwrap();
                        current_var_count = layer.num_vars();
                        let mles_and_expressions =
                            layer.layer_proving_info(layer_count, witness_offset);
                        layer_count += 1;
                        witness_offset += mles_and_expressions.num_witnesses();
                        let MLEsAndExpressions {
                            mles,
                            sumcheck_expression,
                            claim_expression,
                        } = mles_and_expressions;
                        // Append the MLEs to the proving info
                        acc.witness_mles_flat.extend(mles);
                        // Add this layer's sumcheck expression and claim expression
                        sumcheck_expr = sumcheck_expr.clone() + sumcheck_expression;
                        acc.claim_expression = acc.claim_expression.clone()
                            + claim_expr_multiplier.clone() * claim_expression;
                    });
                    // Now compute the EQ polynomial for this layer size and add it to the proving info
                    let eq_poly =
                        compute_betas_eval(&sumcheck_point[point_length - current_var_count..])
                            .into_mle();
                    acc.eq_mles.push(eq_poly);
                    acc.sumcheck_expressions.push(sumcheck_expr);
                }
                acc
            },
        );

        // Calculate the current claim
        current_claim =
            proving_info.compute_current_claim(&wit_evals, alpha, lambda, batching_challenge)?;

        // Append the current claim to the transcript
        transcript.append_field_element_ext(&current_claim);

        let either_mles = proving_info.either_mles();
        let full_sumcheck_expr = proving_info.full_sumcheck_expression();
        let expr_builder = VirtualPolynomialsBuilder::<E>::new_with_mles(
            num_threads,
            current_layer_vars,
            either_mles,
        );

        // If we are proving lookups, rather than tables, and its the final round then we have to use a different sumcheck expression
        let virtual_poly = expr_builder
            .to_virtual_polys(std::slice::from_ref(&full_sumcheck_expr), &[alpha, lambda]);

        let (proof, state) = IOPProverState::<E>::prove(virtual_poly, transcript);

        // Update the sumcheck point
        sumcheck_point = state.collect_raw_challenges();

        // Extract all the evaluations apart from the EQ polys
        let evals = state.get_mle_flatten_final_evaluations()[..witness_offset as usize].to_vec();

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
    let mut skip = 0usize;
    let final_round_eval_ref = round_evaluations.last().ok_or(anyhow!(
        "Proving failed: No round evaluations found".to_string(),
    ))?;
    let (output_claims, num_denominator_columns_per_instance) =
        circuits
            .iter()
            .try_fold((vec![], vec![]), |(mut acc, mut column_count), circuit| {
                let first_layer = circuit
                    .layers()
                    .first()
                    .ok_or(anyhow!("Proving failed: No layers in circuit"))?;
                // We add one here because the final layer includes the output layer which has one more variable than the number of layers
                let num_vars = circuit.num_vars() + 1;
                let final_round_evals = first_layer.final_eval_count();
                let circuit_final_evals = &final_round_eval_ref[skip..skip + final_round_evals];
                skip += final_round_evals;
                let evals =
                    first_layer.combine_final_evals(circuit_final_evals, batching_challenge)?;
                for eval in evals {
                    acc.push(Claim::<E>::new(
                        sumcheck_point[point_length - num_vars..].to_vec(),
                        eval,
                    ));
                }
                column_count.push(first_layer.denominator_column_count());
                Result::<(Vec<Claim<E>>, Vec<usize>)>::Ok((acc, column_count))
            })?;

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
        num_denominator_columns_per_instance,
    })
}
