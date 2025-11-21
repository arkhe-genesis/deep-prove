//! Code for proving a Softmax layer.

use super::*;

impl Softmax<Element> {
    #[allow(clippy::type_complexity)]
    pub(crate) fn prove_internal<
        E: ExtensionField,
        PCS: PolynomialCommitmentScheme<E> + Send + Sync,
        T: transcript::Transcript<E>,
    >(
        &self,
        node_id: NodeId,
        last_claims: Vec<&Claim<E>>,
        ctx: &SoftmaxCtx,
        unpadded_shape: &Shape,
        prover: &mut crate::Prover<E, T, PCS>,
    ) -> Result<(Vec<Claim<E>>, SoftmaxProof<E, PCS>)>
    where
        PCS::CommitmentWithWitness: Serialize + DeserializeOwned + Send + Sync,
        PCS::ProverParam: Send + Sync,
    {
        // Check number of claims
        ensure!(
            last_claims.len() == 1,
            "Softmax only produces one output claim but got: {}",
            last_claims.len()
        );
        let last_claim = last_claims[0];
        let input_shape = unpadded_shape.next_power_of_two();
        let final_dim_size = input_shape
            .last()
            .ok_or(anyhow!("Shifted input has no shape"))?
            .next_power_of_two();

        let first_dim = unpadded_shape[..unpadded_shape.rank() - 2]
            .iter()
            .product::<usize>();
        // Retrieve all the witness data
        let number_of_range_checks = ctx.quant_info.number_of_range_checks();
        let number_of_zero_chunks = ctx.quant_info.number_of_zero_chunks();
        let layer_commitment = prover.lookup_witness(node_id)?;
        // Prepare the lookup inputs from the layer commitment
        let logup_inputs = ctx.lookup_ctx.create_logup_inputs_softmax::<PCS, E>(
            layer_commitment,
            &prover.challenge_storage,
            final_dim_size,
            number_of_range_checks,
            number_of_zero_chunks,
            first_dim,
        )?;
        let layer_polys = PCS::get_arc_mle_witness_from_commitment(layer_commitment);
        let commitment = PCS::get_pure_commitment(layer_commitment);
        // Run the logup proving
        let logup_batch_proof = batch_multiple_sizes_prove(&logup_inputs, prover.transcript)?;

        let logup_point = &logup_batch_proof.output_claims()[0].point;
        // We need to know how many variables it takes to represent the normalisation dimension
        let dim_vars = ceil_log2(final_dim_size);
        let two = E::from_canonical_u64(2u64);
        let two_inv = two.inverse();

        // The error lookup is performed over the output summed on the final dimension so we need to extend the point used with correct number
        // of 2^-1 entries
        let full_error_point = std::iter::repeat_n(two_inv, dim_vars)
            .chain(logup_point.iter().skip(dim_vars).copied())
            .collect::<Vec<E>>();
        // Here we split the last claim point up according to input shape
        let split = input_shape.split_point(last_claim.point())?;
        // The batch challenge point is the first part of the split, the rest are the last claim points
        let batch_chal_point = split[0];
        let lc_eq_point = split
            .iter()
            .skip(1)
            .rev()
            .flat_map(|&v| v)
            .copied()
            .collect::<Vec<E>>();

        // Make all the eq polys
        let error_eq = compute_betas_eval(&full_error_point).into_mle();
        let logup_eq = compute_betas_eval(logup_point).into_mle();
        let last_claim_eq = compute_betas_eval(&lc_eq_point).into_mle();

        // We split the layer polys up here, all polys related to decomposition of the input come first
        // and there will be first_dim * (number_of_range_checks + 1 + number_of_zero_chunks) in total.
        // After we have split these of the next first_dim * (1 + number_of_zero_chunks) are used to calculate the output.
        let number_input_polys = first_dim * (number_of_range_checks + 1 + number_of_zero_chunks);
        let number_output_polys = first_dim * (number_of_zero_chunks + 1);
        let (_, rest) = layer_polys.split_at(number_input_polys);
        let (sumcheck_polys, shift_polys) = rest.split_at(number_output_polys);

        // Transform the polys into Either::Left so they can be passed to the VirtualPolynomialsBuilder
        let either_mles = [&last_claim_eq, &error_eq, &logup_eq]
            .into_iter()
            .map(Either::Left)
            .chain(sumcheck_polys.iter().map(|p| Either::Left(p.as_ref())))
            .collect::<Vec<Either<_, _>>>();

        // Squeeze a batching challenge from the transcript, powers of these challenges will be used to
        // link the MLEs used in the lookup to this sumcheck
        let alphas = (0..first_dim)
            .map(|_| {
                prover
                    .transcript
                    .sample_and_append_challenge(b"batching_challenge")
                    .elements
            })
            .collect::<Vec<E>>();

        let batching_evals =
            calculate_batching_challenges(last_claim.point(), &input_shape, unpadded_shape)?;
        let challenges = batching_evals
            .iter()
            .zip(alphas)
            .flat_map(|(&a, b)| [a, b])
            .collect::<Vec<E>>();
        // Make the VirtualPolynomials and run the sumcheck
        let num_vars = logup_point.len();
        let num_threads = optimal_sumcheck_threads(num_vars);
        let expr_builder =
            VirtualPolynomialsBuilder::<E>::new_with_mles(num_threads, num_vars, either_mles);

        let sumcheck_expression = build_softmax_sumcheck_expression::<E>(
            number_of_zero_chunks,
            first_dim,
            final_dim_size,
        );

        let virtual_poly = expr_builder.to_virtual_polys(&sumcheck_expression, &challenges);
        let (sumcheck_proof, state) = IOPProverState::<E>::prove(virtual_poly, prover.transcript);
        let sumcheck_point = state
            .challenges
            .iter()
            .map(|c| c.elements)
            .collect::<Vec<E>>();
        let all_evals = state.get_mle_flatten_final_evaluations();

        // We have all the range claims, then the exp claims, then zero claims, then error claims
        let logup_claims = logup_batch_proof.output_claims();
        let (range_claims, rest) = logup_claims.split_at(first_dim * number_of_range_checks);
        let (exp_claims, rest) = rest.split_at(2 * first_dim);
        let (zero_claims, _) = rest.split_at(first_dim * 2 * number_of_zero_chunks);

        // We evaluate the shift polys at the logup point (skipping the variables relating to the normalisation dimension entries)
        let shift_eval_point = logup_point[dim_vars..].to_vec();
        let shift_evals = shift_polys
            .iter()
            .map(|p| p.evaluate(&shift_eval_point))
            .collect::<Vec<E>>();

        // These constants are used to recombine the chunks from the lookups
        let base_multiplier = E::from_canonical_u64(1u64 << *quantization::BIT_LEN);
        let right_shift_field = E::from_canonical_u64(1u64 << ctx.quant_info.right_shift);
        let rounding = E::from_canonical_u64(1u64 << (ctx.quant_info.right_shift - 1));
        let fpm_field: E = ctx.quant_info.fixed_point_multiplier.to_field();
        let fpm_inv = fpm_field.inverse();
        let zero_offset = E::from_canonical_u64(
            1 << (ctx.quant_info.right_shift + ctx.quant_info.lut.table_bit_size()),
        );

        // Combine the range claims for each chunk
        let (low_parts, stacked_range_evals): (Vec<E>, Vec<Vec<E>>) = range_claims
            .chunks(number_of_range_checks)
            .map(|chunk| {
                let range_evals = chunk.iter().map(|c| c.evaluation()).collect::<Vec<E>>();
                let input_part = range_evals
                    .iter()
                    .fold((E::ZERO, E::ONE), |(acc, pow_two), &b| {
                        (acc + pow_two * b, pow_two * base_multiplier)
                    })
                    .0;
                (input_part, range_evals)
            })
            .unzip();
        // Combine the exp claims
        let (exp_parts, stacked_exp_evals): (Vec<E>, Vec<E>) = exp_claims
            .iter()
            .step_by(2)
            .map(|c| (c.evaluation() * right_shift_field, c.evaluation()))
            .unzip();
        // Combine the zero claims
        let (high_parts, stacked_high_evals): (Vec<E>, Vec<Vec<E>>) = zero_claims
            .chunks(2 * number_of_zero_chunks)
            .map(|chunk| {
                let high_evals = chunk
                    .iter()
                    .step_by(2)
                    .map(|c| c.evaluation())
                    .collect::<Vec<E>>();
                let input_part = high_evals
                    .iter()
                    .fold((E::ZERO, zero_offset), |(acc, pow_two), &b| {
                        (acc + pow_two * b, pow_two * base_multiplier)
                    })
                    .0;
                (input_part, high_evals)
            })
            .unzip();

        // Calculate the input evaluation
        let row_lt_eval = evaluate_row_lt_poly(&logup_point[..dim_vars], unpadded_shape.dim(-1))?;
        let negative_infinity_field: E = ctx.quant_info.quantised_negative_infinity().to_field();
        let column_padding_eval = negative_infinity_field * (E::ONE - row_lt_eval);
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
        // The first commitment is the range checks, then the exp inputs, then the zero inputs
        let first_commit_evals = izip!(stacked_range_evals, stacked_exp_evals, stacked_high_evals)
            .flat_map(|(mut rs, e, zs)| {
                rs.push(e);
                rs.extend(zs);
                rs
            })
            .collect::<Vec<E>>();

        let first_commit_point = logup_point.to_vec();

        // The second commitment is the exp output and the zero outputs
        let second_commit_evals = all_evals[3..].to_vec();
        let second_commit_point = sumcheck_point.clone();
        // Combine them all in the correct order and add them to the claim prover
        let layer_claims = vec![
            (first_commit_point, first_commit_evals),
            (second_commit_point, second_commit_evals),
            (shift_eval_point, shift_evals.clone()),
        ];
        prover.add_witness_claim(node_id, layer_claims);

        let input_claim = Claim::<E>::new(
            logup_point
                .iter()
                .chain(batch_chal_point.iter())
                .copied()
                .collect::<Vec<E>>(),
            input_eval,
        );

        let softmax_proof = SoftmaxProof {
            logup_proof: logup_batch_proof,
            commitment,
            sumcheck_proof,
            evaluations: [&all_evals[3..], shift_evals.as_slice()].concat(),
        };

        Ok((vec![input_claim], softmax_proof))
    }
}
