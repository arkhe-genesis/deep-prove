pub mod circuit;
pub mod prover;
pub mod structs;
pub mod verifier;

#[cfg(test)]
mod tests {
    use ceno_p3::goldilocks::Goldilocks;
    use ff_ext::{FromUniformBytes, GoldilocksExt2};
    use multilinear_extensions::mle::IntoMLE;

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

    #[test]
    fn test_logup_batch_prove() {
        let mut rng = rng_from_env_or_random();
        // First we make a few instances of different sizes
        let (inputs, instances): (
            Vec<LogUpInput<GoldilocksExt2>>,
            Vec<Vec<LogUpVerifierInstance<GoldilocksExt2>>>,
        ) = (7..12)
            .rev()
            .map(|n| {
                let column = random_field_vector::<Goldilocks>(1 << n);
                let column_2 = random_field_vector::<Goldilocks>(1 << n);

                let constant_challenge = GoldilocksExt2::random(&mut rng);
                let column_separation_challenge = GoldilocksExt2::random(&mut rng);

                let column_evals = vec![column.clone(), column_2.clone()];
                let input = LogUpInput::<GoldilocksExt2>::new_lookup(
                    column_evals.clone(),
                    constant_challenge,
                    column_separation_challenge,
                    1,
                )
                .unwrap();

                let instance = LogUpVerifierInstance::<GoldilocksExt2>::new(
                    constant_challenge,
                    column_separation_challenge,
                    1,
                    crate::lookup::logup_gkr::structs::ProofType::Lookup,
                    n - 1,
                );
                (input, vec![instance; 2])
            })
            .unzip();

        let mut prover_transcript = default_transcript::<GoldilocksExt2>();

        let proof = new_batch_multiple_sizes_prove(&inputs, &mut prover_transcript).unwrap();

        let mut verifier_transcript = default_transcript::<GoldilocksExt2>();
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
            let expected_eval = column_mle.evaluate(verifier_claim.point());
            assert_eq!(verifier_claim.evaluation(), expected_eval);
        }
    }
}
