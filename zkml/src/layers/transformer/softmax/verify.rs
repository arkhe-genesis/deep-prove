//! Code for verifying a Softmax layer proof.
use super::*;

impl SoftmaxCtx {
    pub(crate) fn verify_internal<E, PCS, T>(
        &self,
        proof: &SoftmaxProof<E, PCS>,
        last_claims: &[&Claim<E>],
        verifier: &mut Verifier<E, T, PCS>,
        shape_step: &ShapeStep,
    ) -> Result<Vec<Claim<E>>>
    where
        E: ExtensionField,
        PCS: PolynomialCommitmentScheme<E>,
        T: transcript::Transcript<E>,
    {
        // First we check that we only have one claim in `last_claims`
        ensure!(
            last_claims.len() == 1,
            "Softmax only outputs 1 claim, received {} while verifying Softmax step",
            last_claims.len()
        );
        // First dim is the number of 2D sub-tensors we have (without padding)
        let unpadded_shape = &shape_step.unpadded_input_shape[0];
        let input_shape = &shape_step.padded_input_shape[0];
        let first_dim = unpadded_shape[..unpadded_shape.rank() - 2]
            .iter()
            .product::<usize>();
        let final_dim_size = *input_shape
            .last()
            .ok_or(anyhow!("Couldn't verify Softmax, had no input shape"))?;
        let last_claim = last_claims[0];
        let split_point = input_shape.split_point::<E>(last_claim.point())?;

        let dim_vars = ceil_log2(final_dim_size);
        let two = E::from_canonical_u64(2u64);
        let two_inv = two.inverse();

        let SoftmaxProof {
            logup_proof,
            commitment,
            sumcheck_proof,
            evaluations,
        } = proof;

        // Verify the lookup proof
        let batch_claim = verify_logup_proof_multiple_sizes(logup_proof, verifier.transcript)?;

        // Since the lookup ctx is built without knowing the unpadded first dim of the input shpe, here
        // we make a new one in order to verify the proof
        let LayerLookupContext {
            tables,
            instances_per_table,
        } = &self.lookup_ctx;
        let instances_per_table = instances_per_table
            .iter()
            .map(|&n| n * first_dim)
            .collect::<Vec<usize>>();
        let new_lookup_ctx = LayerLookupContext::new(tables.clone(), instances_per_table);
        new_lookup_ctx.verify_logup_batch_claim(&batch_claim, &verifier.challenge_storage)?;

        // Now we squeeze the batching challenge
        // Squeeze a batching challenge from the transcript
        let alphas = (0..first_dim)
            .map(|_| {
                verifier
                    .transcript
                    .sample_and_append_challenge(b"batching_challenge")
                    .elements
            })
            .collect::<Vec<E>>();

        // poly_evals will be in the order range_evals, exp_evals, zero_evals then error_evals
        let poly_evals = batch_claim.poly_evals();

        let number_of_range_checks = self.quant_info.number_of_range_checks();
        let number_of_zero_chunks = self.quant_info.number_of_zero_chunks();
        // Split the poly_evals into their respective sections
        let (range_evals, rest) = poly_evals.split_at(number_of_range_checks * first_dim);
        let (exp_evals, rest) = rest.split_at(2 * first_dim);
        let (zero_evals, error_evals) = rest.split_at(first_dim * 2 * number_of_zero_chunks);
        // We need to unzip the exp and zero evals into their input and output components
        let (exp_in_evals, exp_out_evals): (Vec<E>, Vec<E>) = exp_evals
            .chunks(2)
            .map(|chunk| (chunk[0], chunk[1]))
            .unzip();
        let (zero_in_evals, zero_out_evals): (Vec<E>, Vec<E>) = zero_evals
            .chunks(2)
            .map(|chunk| (chunk[0], chunk[1]))
            .unzip();

        let batch_chal_point = split_point[..input_shape.rank() - 2]
            .iter()
            .rev()
            .flat_map(|s| *s)
            .copied()
            .collect::<Vec<E>>();

        let batching_evals =
            calculate_batching_challenges(last_claim.point(), input_shape, unpadded_shape)?;
        // Now we can compute the initial claim for the sumcheck, this should be a random linear combination of
        // `last_claim.evaluation()`, the error lookup evaluation and the evaluations of the exp and zero lookups
        let initial_claim = izip!(
            alphas.iter(),
            error_evals.iter(),
            exp_out_evals.iter(),
            zero_out_evals.chunks(number_of_zero_chunks),
            batching_evals.iter()
        )
        .fold(
            last_claim.evaluation(),
            |acc, (&alpha, &error, &exp_out, zero_chunk, &batch)| {
                let sum_part = zero_chunk
                    .iter()
                    .fold(
                        (exp_out * alpha, alpha * alpha),
                        |(eval_acc, chal_acc), &e| (eval_acc + chal_acc * e, chal_acc * alpha),
                    )
                    .0;
                let error_part = batch * batch * error;
                acc + sum_part + error_part
            },
        );

        let last_claim_eq_point = split_point
            .iter()
            .skip(1)
            .rev()
            .flat_map(|s| *s)
            .copied()
            .collect::<Vec<E>>();

        // The error lookup is performed over the output summed on the final dimension so we need to extend the point used with correct number
        // of 2^-1 entries
        let full_error_point = std::iter::repeat_n(two_inv, dim_vars)
            .chain(batch_claim.point().iter().skip(dim_vars).copied())
            .collect::<Vec<E>>();

        let max_degree = 2 + number_of_zero_chunks;

        let aux_info = VPAuxInfo {
            max_num_variables: last_claim_eq_point.len(),
            max_degree,
            ..Default::default()
        };
        // Verify the Sumcheck proof
        let subclaim = IOPVerifierState::<E>::verify(
            initial_claim,
            sumcheck_proof,
            &aux_info,
            verifier.transcript,
        );
        let sumcheck_point = subclaim
            .point
            .iter()
            .map(|c| c.elements)
            .collect::<Vec<E>>();

        let last_claim_eq = identity_eval(&last_claim_eq_point, &sumcheck_point);
        let logup_eq = identity_eval(batch_claim.point(), &sumcheck_point);
        let error_eq = identity_eval(&full_error_point, &sumcheck_point);

        let all_sumcheck_evals = [last_claim_eq, error_eq, logup_eq]
            .into_iter()
            .chain(
                evaluations
                    .iter()
                    .take(first_dim * (1 + number_of_zero_chunks))
                    .copied(),
            )
            .collect::<Vec<E>>();

        let challenges = batching_evals
            .iter()
            .zip(alphas)
            .flat_map(|(&a, b)| [a, b])
            .collect::<Vec<E>>();
        // Check that the provided evaluation matches the expected evaluation from the sumcheck
        let sumcheck_expression = build_softmax_sumcheck_expression::<E>(
            number_of_zero_chunks,
            first_dim,
            final_dim_size,
        );
        let calc_subclaim = sumcheck_expression.iter().fold(E::ZERO, |acc, expr| {
            eval_by_expr_with_instance(&[], &all_sumcheck_evals, &[], &[], &challenges, expr)
                .right()
                .unwrap()
                + acc
        });

        ensure!(
            subclaim.expected_evaluation == calc_subclaim,
            "Softmax sumcheck subclaim evaluation did not match expected evaluation"
        );

        let shift_eval_point = batch_claim.point()[dim_vars..].to_vec();
        let shift_evals = evaluations
            .iter()
            .skip(first_dim * (1 + number_of_zero_chunks))
            .copied()
            .collect::<Vec<E>>();
        // Constants used to recombine the claims
        let base_multiplier = E::from_canonical_u64(1u64 << *quantization::BIT_LEN);
        let right_shift_field = E::from_canonical_u64(1u64 << self.quant_info.right_shift);
        let rounding = E::from_canonical_u64(1u64 << (self.quant_info.right_shift - 1));
        let fpm_field: E = self.quant_info.fixed_point_multiplier.to_field();
        let fpm_inv = fpm_field.inverse();
        let table_size_field: E = self.quant_info.lut.full_table_size().to_field();
        let zero_offset = table_size_field * right_shift_field;

        // Combine the range claims for each chunk
        let (low_parts, stacked_range_evals): (Vec<E>, Vec<Vec<E>>) = range_evals
            .chunks(number_of_range_checks)
            .map(|chunk| {
                let input_part = chunk
                    .iter()
                    .fold((E::ZERO, E::ONE), |(acc, pow_two), &b| {
                        (acc + pow_two * b, pow_two * base_multiplier)
                    })
                    .0;
                (input_part, chunk.to_vec())
            })
            .unzip();
        // Combine the exp claims
        let (exp_parts, stacked_exp_evals): (Vec<E>, Vec<E>) = exp_in_evals
            .iter()
            .map(|&e| (e * right_shift_field, e))
            .unzip();
        // Combine the zero claims for each chunk
        let (high_parts, stacked_high_evals): (Vec<E>, Vec<Vec<E>>) = zero_in_evals
            .chunks(number_of_zero_chunks)
            .map(|chunk| {
                let input_part = chunk
                    .iter()
                    .fold((E::ZERO, zero_offset), |(acc, pow_two), &b| {
                        (acc + pow_two * b, pow_two * base_multiplier)
                    })
                    .0;
                (input_part, chunk.to_vec())
            })
            .unzip();
        // Now we can recombine everything to get the input eval
        let batch_claim_point = batch_claim.point();
        let row_lt_eval =
            evaluate_row_lt_poly(&batch_claim_point[..dim_vars], unpadded_shape.dim(-1))?;
        let negative_infinity_field: E = self.quant_info.quantised_negative_infinity().to_field();
        let column_padding_eval = (E::ONE - row_lt_eval) * negative_infinity_field;
        let input_eval = izip!(
            low_parts,
            exp_parts,
            high_parts,
            shift_evals.iter(),
            batching_evals
        )
        .map(|(l, e, h, &shift, batch)| {
            ((l + e - h - rounding) * fpm_inv - row_lt_eval * shift - column_padding_eval) * batch
        })
        .sum::<E>();

        let first_commit_evals = izip!(stacked_range_evals, stacked_exp_evals, stacked_high_evals)
            .flat_map(|(mut rs, e, zs)| {
                rs.push(e);
                rs.extend(zs);
                rs
            })
            .collect::<Vec<E>>();

        let first_commit_point = batch_claim_point.to_vec();

        // The second commitment is the exp output and the zero outputs
        let second_commit_evals = evaluations
            .iter()
            .take(first_dim * (1 + number_of_zero_chunks))
            .copied()
            .collect::<Vec<E>>();
        let second_commit_point = sumcheck_point.clone();
        // Combine them all in the correct order and add them to the claim prover
        let layer_claims = vec![
            (first_commit_point, first_commit_evals),
            (second_commit_point, second_commit_evals),
            (shift_eval_point, shift_evals.clone()),
        ];

        verifier
            .commit_verifier
            .add_witness_claim(self.node_id, commitment.clone(), layer_claims);

        let input_claim = Claim::<E>::new(
            batch_claim
                .point()
                .iter()
                .copied()
                .chain(batch_chal_point)
                .collect::<Vec<E>>(),
            input_eval,
        );

        Ok(vec![input_claim])
    }
}
