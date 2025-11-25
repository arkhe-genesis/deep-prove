//! Code for verifying an attention mask layer

use super::*;

impl AttentionMask<Element> {
    pub(crate) fn verify_internal<E, PCS, T>(
        &self,
        proof: &AttentionMaskProof<E>,
        last_claim: &Claim<E>,
        mask_verifying_data: MaskVerifyingData<E>,
        unpadded_seq_len: usize,
        verifier: &mut Verifier<E, T, PCS>,
    ) -> Result<Vec<Claim<E>>>
    where
        E: ExtensionField,
        PCS: PolynomialCommitmentScheme<E>,
        T: Transcript<E>,
    {
        let AttentionMaskProof {
            sumcheck_proof,
            evaluations,
        } = proof;

        let MaskVerifyingData {
            batching_challenges,
            batching_point,
            eq_point,
        } = mask_verifying_data;

        let num_vars = eq_point.len();
        let aux_info = VPAuxInfo {
            max_degree: 5,
            max_num_variables: num_vars,
            ..Default::default()
        };
        let subclaim = IOPVerifierState::<E>::verify(
            last_claim.evaluation(),
            sumcheck_proof,
            &aux_info,
            verifier.transcript,
        );
        let sumcheck_point = subclaim
            .point
            .iter()
            .map(|c| c.elements)
            .collect::<Vec<E>>();

        let dim_vars = num_vars >> 1;
        let eq_eval = eq_eval(&sumcheck_point, &eq_point);

        let (column_point, row_point) = sumcheck_point.split_at(dim_vars);

        let mask_eval = eval_zeroifier_mle(column_point, row_point);
        let (row_lt_eval, column_lt_eval) =
            self.evaluate_row_column_lt_polys::<E>(row_point, column_point, unpadded_seq_len)?;
        let lt_eval = row_lt_eval * column_lt_eval;
        let neg_inf_field: E = self.negative_infinity.to_field();
        let calc_eval = eq_eval
            * lt_eval
            * batching_challenges.iter().zip(evaluations.iter()).fold(
                E::ZERO,
                |acc, (chal, eval)| {
                    acc + *chal * (mask_eval * *eval + neg_inf_field * (E::ONE - mask_eval))
                },
            );

        ensure!(
            calc_eval == subclaim.expected_evaluation,
            "Casual Mask verification failed, expected evaluation {:?} got {:?}",
            subclaim.expected_evaluation,
            calc_eval
        );

        // Construct the input claim
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

        Ok(input_claim)
    }

    /// Given the row and column points, evaluates the row and column less than polynomials
    fn evaluate_row_column_lt_polys<E: ExtensionField>(
        &self,
        row_point: &[E],
        column_point: &[E],
        unpadded_seq_len: usize,
    ) -> Result<(E, E)> {
        let bit_len = ceil_log2(unpadded_seq_len);
        ensure!(
            row_point.len() == bit_len,
            "Row point length {} does not match unpadded seq len log2 {bit_len}",
            row_point.len(),
        );
        ensure!(
            column_point.len() == bit_len,
            "Column point length {} does not match unpadded seq len log2 {bit_len}",
            column_point.len(),
        );
        let seq_len_bits = to_bit_sequence_le(unpadded_seq_len - 1, bit_len)
            .map(E::from_canonical_usize)
            .collect::<Vec<E>>();
        let row_eval = eval_zeroifier_mle(row_point, &seq_len_bits);
        let column_eval = eval_zeroifier_mle(column_point, &seq_len_bits);
        Ok((row_eval, column_eval))
    }
}

#[derive(Debug, Clone)]
/// Struct storing all information to verify a [`AttentionMaskProof`]. We prove and verify the mask applied to each individual 2D sub tensor in a batched fashion.
/// This struct hold the information needed and is constructed from the shaps and the last claim.
pub(crate) struct MaskVerifyingData<E: ExtensionField> {
    /// These values are the evaluations of the eq-poly for the higher dims that aren't from padding
    batching_challenges: Vec<E>,
    /// This is the point used to make the batch challenges
    batching_point: Vec<E>,
    /// This is evaluations of the eq-poly for each of the rank-2 tensors that the mask is applied to
    eq_point: Vec<E>,
}

impl<E: ExtensionField> MaskVerifyingData<E> {
    /// Create a new [`MaskVerifyingData`]
    pub fn new(batching_challenges: Vec<E>, batching_point: Vec<E>, eq_point: Vec<E>) -> Self {
        MaskVerifyingData {
            batching_challenges,
            batching_point,
            eq_point,
        }
    }

    pub fn new_from_claim_and_shape_data(
        claim: &Claim<E>,
        input_shape: &Shape,
        unpadded_input_shape: &Shape,
    ) -> Result<Self> {
        let rank = input_shape.rank();

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
            .zip(unpadded_input_shape[..rank - 2].iter())
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

        Ok(MaskVerifyingData::new(
            batching_challenges,
            batching_point,
            eq_point,
        ))
    }
}
