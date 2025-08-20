pub mod circuit;
pub mod error;
pub mod prover;
pub mod structs;
pub mod verifier;

#[cfg(test)]
mod tests {
    use ceno_p3::{field::FieldAlgebra, goldilocks::Goldilocks};
    use ff_ext::{FromUniformBytes, GoldilocksExt2};

    use crate::{
        default_transcript,
        lookup::logup_gkr::{
            prover::batch_multiple_sizes_prove, structs::LogUpInput,
            verifier::verify_logup_proof_multiple_sizes,
        },
        rng_from_env_or_random,
        testing::random_field_vector,
    };

    #[test]
    fn test_logup_batch_prove() {
        let mut rng = rng_from_env_or_random();
        // First we make a few instances of different sizes
        let inputs = (7..12)
            .rev()
            .map(|n| {
                let column = random_field_vector::<Goldilocks>(1 << n);
                let column_2 = random_field_vector::<Goldilocks>(1 << n);

                let constant_challenge = GoldilocksExt2::random(&mut rng);
                let column_separation_challenge = GoldilocksExt2::random(&mut rng);

                let column_evals = vec![column.clone(), column_2.clone()];
                LogUpInput::<GoldilocksExt2>::new_lookup(
                    column_evals.clone(),
                    constant_challenge,
                    column_separation_challenge,
                    1,
                )
                .unwrap()
            })
            .collect::<Vec<LogUpInput<GoldilocksExt2>>>();

        let mut prover_transcript = default_transcript::<GoldilocksExt2>();
        let now = std::time::Instant::now();
        let proof = batch_multiple_sizes_prove(&inputs, &mut prover_transcript).unwrap();
        println!("Batch proof took: {:?}", now.elapsed());

        let mut verifier_transcript = default_transcript::<GoldilocksExt2>();
        let logup_claim =
            verify_logup_proof_multiple_sizes(&proof, &mut verifier_transcript).unwrap();

        let calc = inputs
            .iter()
            .fold(
                (GoldilocksExt2::ZERO, 0, GoldilocksExt2::ONE),
                |(acc, skip, batch), l_input| {
                    let LogUpInput::Lookup {
                        constant_challenge,
                        column_separation_challenge,
                        ..
                    } = l_input
                    else {
                        unreachable!()
                    };
                    let take = l_input.column_evals().len();
                    let evals = proof
                        .output_claims()
                        .iter()
                        .skip(skip)
                        .take(take)
                        .map(|c| c.eval)
                        .collect::<Vec<GoldilocksExt2>>();
                    let (value, new_batch) = evals.chunks(l_input.columns_per_instance()).fold(
                        (GoldilocksExt2::ZERO, batch),
                        |(acc, chal_acc), chunk| {
                            let (value, _) = chunk.iter().fold(
                                (*constant_challenge, GoldilocksExt2::ONE),
                                |(acc, chal_acc), &val| {
                                    (
                                        acc + chal_acc * val,
                                        chal_acc * *column_separation_challenge,
                                    )
                                },
                            );
                            (acc + value * chal_acc, chal_acc * logup_claim.alpha())
                        },
                    );
                    (acc + value, skip + take, new_batch)
                },
            )
            .0;

        assert_eq!(logup_claim.claim(), calc);
    }
}
