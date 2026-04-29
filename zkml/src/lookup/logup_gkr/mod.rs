pub mod circuit;
pub mod prover;
pub mod structs;
pub mod verifier;

#[cfg(test)]
mod tests {
    use ark_ff::UniformRand;
    use dp_crypto::IntoMLE;

    use crate::{
        default_transcript,
        lookup::logup_gkr::{
            prover::new_batch_multiple_sizes_prove,
            structs::{LogUpInput, LogUpVerifierInstance},
            verifier::new_verify_logup_proof_multiple_sizes,
        },
        rng_from_env_or_random,
        testing::random_field_vector,
    };

    type F = ark_bn254::Fr;

    #[test]
    fn test_logup_batch_prove() {
        let mut rng = rng_from_env_or_random();
        // First we make a few instances of different sizes
        let (inputs, instances): (Vec<LogUpInput<F>>, Vec<Vec<LogUpVerifierInstance<F>>>) = (7..12)
            .rev()
            .map(|n| {
                let column = random_field_vector::<F>(1 << n);
                let column_2 = random_field_vector::<F>(1 << n);

                let constant_challenge = F::rand(&mut rng);
                let column_separation_challenge = F::rand(&mut rng);

                let column_evals = vec![column.clone(), column_2.clone()];
                let input = LogUpInput::<F>::new_lookup(
                    column_evals.clone(),
                    constant_challenge,
                    column_separation_challenge,
                    1,
                )
                .unwrap();

                let instance = LogUpVerifierInstance::<F>::new(
                    constant_challenge,
                    column_separation_challenge,
                    1,
                    crate::lookup::logup_gkr::structs::ProofType::Lookup,
                    n - 1,
                );
                (input, vec![instance; 2])
            })
            .unzip();

        let mut prover_transcript = default_transcript();

        let proof = new_batch_multiple_sizes_prove(&inputs, &mut prover_transcript).unwrap();

        let mut verifier_transcript = default_transcript();
        let flat_instances = instances.iter().flatten().cloned().collect::<Vec<_>>();
        let logup_claim = new_verify_logup_proof_multiple_sizes(
            &proof,
            &flat_instances,
            &mut verifier_transcript,
        )
        .unwrap();

        let flat_columns = inputs
            .iter()
            .flat_map(|input| input.column_evals().to_vec())
            .collect::<Vec<_>>();
        for (verifier_claim, column) in logup_claim.output_claims().iter().zip(flat_columns) {
            let column_mle = column.into_mle();
            let expected_eval = column_mle.evaluate(verifier_claim.point()).unwrap();
            assert_eq!(verifier_claim.evaluation(), expected_eval);
        }
    }
}
