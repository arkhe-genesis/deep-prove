//! Code for proving an attention mask layer

use super::*;

impl AttentionMask<Element> {
    pub(crate) fn prove_internal<E, PCS, T>(
        &self,
        ctx: &AttentionMaskCtx<E>,
        mask_proving_data: MaskProvingData<E>,
        unpadded_seq_len: usize,
        prover: &mut Prover<E, T, PCS>,
    ) -> Result<(AttentionMaskProof<E>, Vec<Claim<E>>)>
    where
        E: ExtensionField,
        PCS: PolynomialCommitmentScheme<E> + Send + Sync,
        PCS::CommitmentWithWitness: Serialize + DeserializeOwned + Send + Sync,
        PCS::ProverParam: Send + Sync,
        T: Transcript<E>,
    {
        let MaskProvingData {
            batching_challenges,
            batching_point,
            eq_evals,
            input_polys,
        } = mask_proving_data;
        let num_vars = ceil_log2(eq_evals.len());
        let eq_poly = MultilinearExtension::from_evaluations_ext_vec(num_vars, eq_evals);
        // Since the mask is square the padded seq_len is just 1 << (num_vars >> 1)
        let mask_poly = self.make_mask_poly(1 << (num_vars >> 1));
        // These polys are used so that the sumcheck only takes into account the portions of the mask poly corresponding to non-padded areas
        let (row_lt_poly, column_lt_poly) = self.make_row_column_lt_polys::<E>(unpadded_seq_len);

        let input_polys = input_polys
            .into_iter()
            .map(|evals| MultilinearExtension::from_evaluations_ext_vec(num_vars, evals))
            .collect::<Vec<_>>();

        let either_mles = [&eq_poly, &mask_poly, &row_lt_poly, &column_lt_poly]
            .into_iter()
            .chain(input_polys.iter())
            .map(Either::Left)
            .collect::<Vec<_>>();
        let num_threads = optimal_sumcheck_threads(num_vars);
        let expr_builder =
            VirtualPolynomialsBuilder::<E>::new_with_mles(num_threads, num_vars, either_mles);

        let virtual_poly = expr_builder.to_virtual_polys(
            &ctx.sumcheck_expression[..input_polys.len()],
            &batching_challenges,
        );
        let (sumcheck_proof, state) = IOPProverState::<E>::prove(virtual_poly, prover.transcript);
        let evaluations = state.get_mle_flatten_final_evaluations()[4..].to_vec();

        let sumcheck_point = state.collect_raw_challenges();

        // We match the number of provided claims, so if we received one claim we output one claim, if we received n claims we output n claims
        let combined_eval = batching_challenges
            .iter()
            .zip(evaluations.iter())
            .fold(E::ZERO, |acc, (c, e)| acc + (*c) * (*e));
        let full_point = sumcheck_point
            .iter()
            .chain(batching_point.iter())
            .copied()
            .collect::<Vec<_>>();
        let input_claim = vec![Claim::<E>::new(full_point, combined_eval)];

        Ok((
            AttentionMaskProof {
                sumcheck_proof,
                evaluations,
            },
            input_claim,
        ))
    }

    /// NOTE: the function does NOT handle the single inference with caching case, and that's ok
    /// since we never want to prove a single token inference with caching enabled.
    /// We always prove the full sequence length, without caching.
    /// However, the evaluation needs to support both cases.
    fn make_mask_poly<E: ExtensionField>(&self, seq_len: usize) -> MultilinearExtension<'_, E> {
        let evals = (0..seq_len)
            .flat_map(|token| {
                (0..seq_len).map(move |other| {
                    let min = match self.span {
                        AttentionSpan::Full => 0,
                        // i - n
                        AttentionSpan::Local(n) => token.saturating_sub(n),
                    };
                    let max = token;
                    if (min..=max).contains(&other) {
                        E::BaseField::ONE
                    } else {
                        E::BaseField::ZERO
                    }
                })
            })
            .collect::<Vec<E::BaseField>>();

        let num_vars = 2 * ceil_log2(seq_len);
        MultilinearExtension::from_evaluations_vec(num_vars, evals)
    }

    /// Function to make the row and column less than polynomials for proving purposes
    fn make_row_column_lt_polys<E: ExtensionField>(
        &self,
        unpadded_seq_len: usize,
    ) -> (MultilinearExtension<'_, E>, MultilinearExtension<'_, E>) {
        let padded_seq_len = unpadded_seq_len.next_power_of_two();
        // First we make the row less than evaluations
        let row_evals = (0..padded_seq_len)
            .map(|row_index| {
                if row_index < unpadded_seq_len {
                    E::BaseField::ONE
                } else {
                    E::BaseField::ZERO
                }
            })
            .cycle()
            .take(padded_seq_len * padded_seq_len)
            .collect::<Vec<E::BaseField>>();

        // The column less than evaluations will be `unpadded_seq_len` rows of `padded_seq_len` ones followed by
        // `(padded_seq_len - unpadded_seq_len)` rows of `padded_seq_len` zeros
        let column_ones_count = unpadded_seq_len * padded_seq_len;
        let column_zeros_count = (padded_seq_len - unpadded_seq_len) * padded_seq_len;
        let column_evals = std::iter::repeat_n(E::BaseField::ONE, column_ones_count)
            .chain(std::iter::repeat_n(E::BaseField::ZERO, column_zeros_count))
            .collect::<Vec<E::BaseField>>();

        let num_vars = 2 * ceil_log2(unpadded_seq_len);
        (
            MultilinearExtension::from_evaluations_vec(num_vars, row_evals),
            MultilinearExtension::from_evaluations_vec(num_vars, column_evals),
        )
    }
}

#[derive(Debug, Clone)]
/// Struct storing all information to prove the application of an attention mask correctly without having to do proving work on padded parts.
pub(crate) struct MaskProvingData<E: ExtensionField> {
    /// These values are the evaluations of the eq-poly for the higher dims that aren't from padding
    batching_challenges: Vec<E>,
    /// This is the point used to make the batch challenges
    batching_point: Vec<E>,
    /// This is evaluations of the eq-poly for each of the rank-2 tensors that the mask is applied to
    eq_evals: Vec<E>,
    /// This list of evaluations are the rank-2 tensors forming the input that aren't padding parts
    input_polys: Vec<Vec<E>>,
}

impl<E: ExtensionField> MaskProvingData<E> {
    /// Create a new [`MaskProvingData`]
    pub fn new(
        batching_challenges: Vec<E>,
        batching_point: Vec<E>,
        eq_evals: Vec<E>,
        input_polys: Vec<Vec<E>>,
    ) -> Self {
        MaskProvingData {
            batching_challenges,
            batching_point,
            eq_evals,
            input_polys,
        }
    }

    pub fn from_claims_and_input(claim: &Claim<E>, input: &Tensor<E>) -> Result<Self> {
        let input_shape = input.shape().clone();
        let unpadded_shape = input.unpadded_shape();
        let rank = input_shape.rank();

        let unpadded_input = input.reduce_to_shape(unpadded_shape)?;
        let final_dim = unpadded_shape.dim(-1);
        let second_to_last_dim = unpadded_shape.dim(-2);
        let chunk_size = second_to_last_dim * final_dim;

        let input_polys = unpadded_input
            .get_data_into()
            .chunks(chunk_size)
            .map(|chunk| {
                Tensor::<E>::new(vec![second_to_last_dim, final_dim].into(), chunk.to_vec())
                    .map(|t| t.pad_next_power_of_two().get_data_into())
            })
            .collect::<Result<Vec<Vec<E>>>>()?;

        // Split the last claim point into the points corresponding to each dimension
        let dim_points = input_shape.split_point(claim.point())?;

        // Make the eq_poly evaluations from the points for the last two dimensions
        let eq_point = dim_points[rank - 2..]
            .iter()
            .rev()
            .flat_map(|p| *p)
            .copied()
            .collect::<Vec<E>>();

        let batching_challenges = dim_points[..rank - 2]
            .iter()
            .zip(unpadded_shape[..rank - 2].iter())
            .fold(vec![E::ONE], |mut acc, (point_slice, dim)| {
                let evals = compute_betas_eval(point_slice);
                acc = acc
                    .into_iter()
                    .flat_map(|c| evals.iter().take(*dim).map(|e| c * *e).collect::<Vec<E>>())
                    .collect::<Vec<E>>();
                acc
            });

        let batching_point = dim_points[..rank - 2]
            .iter()
            .rev()
            .flat_map(|p| *p)
            .copied()
            .collect::<Vec<E>>();

        Ok(MaskProvingData::new(
            batching_challenges,
            batching_point,
            compute_betas_eval(&eq_point),
            input_polys,
        ))
    }
}
