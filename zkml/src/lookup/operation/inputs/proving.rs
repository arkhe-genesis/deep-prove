//! Lookup proving methods.

use dp_crypto::{
    arkyper::transcript::Transcript,
    structs::{IOPProof, IOPProverState},
    util::optimal_sumcheck_threads,
    virtual_polys::VirtualPolynomialsBuilder,
};

use super::*;

#[derive(Debug, Clone)]
pub struct LookupSumcheckProof<F: PrimeField> {
    /// The sumcheck proof linking the lookup and the main computation.
    pub sumcheck_proof: IOPProof<F>,
    /// The evaluations of the witness MLEs at the sumcheck point.
    pub evaluations: Vec<F>,
    /// The final point from the sumcheck protocol.
    pub sumcheck_point: Vec<F>,
    /// The evaluation of the weight MLE at the sumcheck point, if applicable.
    pub weight_eval: Option<F>,
}

impl LookupInputConfig {
    /// This method proves the Sumcheck that links the lookup argument to the inputs/outputs of the layer. It takes in the MLEs of the witnesses used in the Sumcheck expression, the current claim, the point used in the lookup proof and optionally the MLE of the weights (if this is a normalisation variant). It returns the sumcheck proof and the evaluations of the witness MLEs at the sumcheck point.
    pub fn prove_linking_sumcheck<F, T>(
        &self,
        mles: &[DensePolynomial<F>],
        transcript: &mut T,
        last_claim: &Claim<F>,
        logup_point: &[F],
        weight_mle: Option<DensePolynomial<F>>,
    ) -> Result<(LookupSumcheckProof<F>, Vec<F>)>
    where
        F: PrimeField,
        T: Transcript,
    {
        let variant = self.variant;
        let unpadded_input_shape = self.unpadded_input_shape();
        let (output_eq_point, mut batching_challenges) = unpadded_input_shape
            .compute_eq_point_and_batching_challenges::<F>(last_claim.point())?;

        // Now we squeeze the required challenges from the transcript, we always need the sum challenge so we squeeze that explicitly first.
        let sum_challenge = transcript.append_and_sample(b"sum");
        // We then get the other challenges required for batching from the variant, this is because different variants require different numbers of challenges depending on the number of checks they perform in the sumcheck and we want to avoid squeezing unnecessary challenges.
        let other_challenges: Vec<F> = variant.squeeze_sumcheck_challenges(transcript);
        // Arrange the challenges in the expected order
        batching_challenges.insert(0, sum_challenge);
        batching_challenges.extend(other_challenges);

        // The MLEs are presumed to be provided in the order of the Sumcheck expression so all we need to do is construct the eq polys required.
        let eq_polys = variant.build_eq_polys(
            &output_eq_point,
            logup_point,
            ceil_log2(unpadded_input_shape.dim(-1)),
        );

        let mut either_mles = mles
            .iter()
            .chain(eq_polys.iter())
            .map(Either::Left)
            .collect::<Vec<_>>();

        if let Some(weight) = weight_mle.as_ref() {
            either_mles.push(Either::Left(weight));
        }

        // Build the virtual polynomials
        let num_vars = output_eq_point.len();
        let num_threads = optimal_sumcheck_threads(num_vars);
        let expression = self.build_full_sumcheck_expression::<F>();

        #[cfg(test)]
        {
            self.debug_linking_sumcheck(
                mles,
                last_claim,
                &weight_mle,
                &batching_challenges,
                &eq_polys,
            )?;
        }

        let virtual_poly_builder =
            VirtualPolynomialsBuilder::<F>::new_with_mles(num_threads, num_vars, either_mles);
        let virtual_poly = virtual_poly_builder
            .to_virtual_polys(std::slice::from_ref(&expression), &batching_challenges);
        let (sumcheck_proof, state) = IOPProverState::<F>::prove(virtual_poly, transcript);
        let mut evaluations = state.get_mle_flatten_final_evaluations();
        let sumcheck_point = state.collect_raw_challenges();

        let weight_eval = if weight_mle.is_some() {
            // In this case the final evaluation is the evaluation for the weight tensor so we pop it off the end
            let final_eval = evaluations
                .pop()
                .ok_or(anyhow!("No evaluations present in lookup sumcheck proof"))?;
            Some(final_eval)
        } else {
            None
        };

        // Remove the eq poly evaluations from the list of evaluations
        evaluations.truncate(evaluations.len() - eq_polys.len());

        Ok((
            LookupSumcheckProof {
                sumcheck_proof,
                evaluations,
                sumcheck_point,
                weight_eval,
            },
            batching_challenges,
        ))
    }
}

#[cfg(test)]
mod test_utils {

    use dp_crypto::utils::eval_by_expr_with_instance;

    use super::*;
    use crate::lookup::operation::variant::LookupExpressions;

    impl LookupInputConfig {
        pub(crate) fn debug_linking_sumcheck<F>(
            &self,
            mles: &[DensePolynomial<F>],
            last_claim: &Claim<F>,
            weight_mle: &Option<DensePolynomial<F>>,
            batching_challenges: &[F],
            eq_polys: &[DensePolynomial<F>],
        ) -> Result<()>
        where
            F: PrimeField,
        {
            let unpadded_input_shape = self.unpadded_input_shape();
            let variant = self.variant;
            let chunking_info = self.chunking_info();
            let total_chunks = self.number_of_chunks();

            let output_witnesses_per_chunk = chunking_info.total_outputs_per_chunk();
            let total_output_witnesses = total_chunks * output_witnesses_per_chunk;

            let total_witnesses =
                total_chunks * variant.additional_witnesses_per_chunk() + total_output_witnesses;

            let mut total_norm = F::ZERO;
            let mut total_input = F::ZERO;
            let mut total_output = F::ZERO;
            let mut total_sum = F::ZERO;

            for current_chunk in 0..total_chunks {
                let LookupExpressions {
                    value,
                    prod_selector,
                    clamping_expression,
                    squared_clamping_expression,
                    sum,
                } = variant.build_lookup_output_expressions::<F>(current_chunk, chunking_info);

                let output_linking_eq = Expression::<F>::WitIn(total_witnesses as u16);
                let lookup_linking_eq = Expression::<F>::WitIn((total_witnesses + 1) as u16);

                let final_dim_size = unpadded_input_shape.dim(-1);
                let final_dim_log = ceil_log2(final_dim_size);
                let pow_two: F = (1i64 << final_dim_log).to_field();

                let prod_expression =
                    clamping_expression.clone() + prod_selector.clone() * value.clone();
                let mut output_part = output_linking_eq * prod_expression.clone();

                let mut all_mle_evals = mles
                    .iter()
                    .map(|either_mle| either_mle.evals())
                    .chain(eq_polys.iter().map(|eq_poly| eq_poly.evals()))
                    .collect::<Vec<Vec<F>>>();
                if let Some(weight) = weight_mle.as_ref() {
                    all_mle_evals.push(weight.evals());

                    output_part *= Expression::<F>::WitIn(all_mle_evals.len() as u16 - 1)
                }

                if matches!(variant, LookupVariant::GLU) {
                    output_part *= Expression::<F>::WitIn(
                        total_output_witnesses as u16 + current_chunk as u16,
                    );
                }
                let total_len = all_mle_evals[0].len();

                let sum_part = sum * lookup_linking_eq.clone();

                let mut chunk_magnitude = F::ZERO;
                let mut chunk_norm_sum = F::ZERO;
                let mut chunk_input = F::ZERO;
                let mut chunk_output = F::ZERO;
                let mut chunk_sum_part = F::ZERO;

                for eval_idx in 0..total_len {
                    let evals_at_idx = all_mle_evals
                        .iter()
                        .map(|mle_evals| mle_evals[eval_idx])
                        .collect::<Vec<F>>();

                    let output_eval_result = eval_by_expr_with_instance(
                        &[],
                        &evals_at_idx,
                        &[],
                        &[],
                        batching_challenges,
                        &output_part,
                    );

                    let sum_part_result = eval_by_expr_with_instance(
                        &[],
                        &evals_at_idx,
                        &[],
                        &[],
                        batching_challenges,
                        &sum_part,
                    );

                    if matches!(
                        variant,
                        LookupVariant::Normalisation { .. } | LookupVariant::Softmax { .. }
                    ) {
                        let normalisation_eq = Expression::<F>::WitIn((total_witnesses + 2) as u16);

                        let sum_part = normalisation_eq.clone() * prod_expression.clone();

                        let sum_eval_result = pow_two
                            * eval_by_expr_with_instance(
                                &[],
                                &evals_at_idx,
                                &[],
                                &[],
                                batching_challenges,
                                &sum_part,
                            );

                        chunk_norm_sum += sum_eval_result;

                        if matches!(variant, LookupVariant::Normalisation { .. }) {
                            // In order to keep te total degree of the sumcheck polynomial low we compute the squared product expression here
                            // as the prod_selector is boolean this is fine

                            let prod_squared_expression = squared_clamping_expression.clone()
                                + prod_selector.clone() * value.clone() * value.clone();

                            let magnitude_part =
                                normalisation_eq.clone() * prod_squared_expression.clone();

                            let magnitude_eval_result = pow_two
                                * eval_by_expr_with_instance(
                                    &[],
                                    &evals_at_idx,
                                    &[],
                                    &[],
                                    batching_challenges,
                                    &magnitude_part,
                                );

                            let input_chunk_expression = Expression::<F>::WitIn(
                                (total_output_witnesses + current_chunk) as u16,
                            );
                            let scaling_witness_expression = Expression::<F>::WitIn(
                                (total_output_witnesses + total_chunks + current_chunk) as u16,
                            );

                            let input_part = input_chunk_expression
                                * scaling_witness_expression
                                * lookup_linking_eq.clone();

                            let input_eval_result = eval_by_expr_with_instance(
                                &[],
                                &evals_at_idx,
                                &[],
                                &[],
                                batching_challenges,
                                &input_part,
                            );
                            chunk_magnitude += magnitude_eval_result;
                            chunk_input += input_eval_result;
                        }
                    }

                    chunk_output += output_eval_result;
                    chunk_sum_part += sum_part_result;
                }
                total_norm +=
                    (chunk_norm_sum + chunk_magnitude) * batching_challenges[1 + current_chunk];
                total_input += chunk_input * batching_challenges[1 + current_chunk];
                total_output += chunk_output * batching_challenges[1 + current_chunk];
                total_sum += chunk_sum_part * batching_challenges[0];
            }
            if total_output != last_claim.evaluation() {
                println!(
                    "Output evaluation does not match last claim evaluation!, table: {}",
                    chunking_info.table().name(),
                );
                println!(
                    "Unpadded input shape: {:?}, number of zero chunks: {}",
                    unpadded_input_shape,
                    chunking_info.number_of_zeroing_chunks()
                );
                println!("Total norm prover: {total_norm}");
                println!("Total input prover: {total_input}");
                println!("Total sum part prover: {total_sum}");
                println!("Total output prover: {total_output}");
                println!("last claim prover: {}", last_claim.evaluation());
            }
            Ok(())
        }
    }
}
