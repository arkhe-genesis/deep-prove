//! Code for verifying an attention mask layer

use anyhow::anyhow;

use crate::poly_commit::identity_eval;

use dp_crypto::{
    poly::eq::evals,
    structs::IOPVerifierState,
    util::ceil_log2,
    virtual_poly::{VPAuxInfo, eq_eval},
};

use super::*;

impl AttentionMask<Element> {
    pub(crate) fn verify_internal<F, PCS, T>(
        &self,
        proof: &AttentionMaskProof<F>,
        last_claim: &Claim<F>,
        mask_verifying_data: MaskVerifyingData<F>,
        unpadded_seq_len: usize,
        verifier: &mut Verifier<F, T, PCS>,
    ) -> Result<Vec<Claim<F>>>
    where
        F: PrimeField,
        PCS: CommitmentScheme,
        T: Transcript,
    {
        let AttentionMaskProof {
            sumcheck_proof,
            evaluations,
            local_mask_proof,
        } = proof;

        let MaskVerifyingData {
            batching_challenges,
            batching_point,
            eq_point,
        } = mask_verifying_data;

        let num_vars = eq_point.len();
        let additional_mask_poly = local_mask_proof.is_some();
        let max_degree = if additional_mask_poly {
            ensure!(
                self.is_local_mask(),
                "Found sumcheck proof for local mask with a full attention mask"
            );
            6
        } else {
            5
        };
        let aux_info = VPAuxInfo {
            max_degree,
            max_num_variables: num_vars,
            ..Default::default()
        };
        let subclaim = IOPVerifierState::<F>::verify(
            last_claim.evaluation(),
            sumcheck_proof,
            &aux_info,
            verifier.transcript,
        );
        let sumcheck_point = &subclaim.point;

        let dim_vars = num_vars >> 1;
        let eq_eval = eq_eval(sumcheck_point, &eq_point);

        let (column_point, row_point) = sumcheck_point.split_at(dim_vars);

        let mut mask_eval = eval_zeroifier_mle(column_point, row_point);
        if additional_mask_poly {
            mask_eval *= evaluations[0]; // in this case, the first evaluation is for the additional mask polynomial
            self.verify_additional_mask_poly_eval(
                row_point,
                column_point,
                evaluations[0],
                local_mask_proof.as_ref().unwrap(),
                verifier,
            )?;
        }

        let input_evals = &evaluations[additional_mask_poly as usize..];

        let (row_lt_eval, column_lt_eval) =
            self.evaluate_row_column_lt_polys::<F>(row_point, column_point, unpadded_seq_len)?;
        let lt_eval = row_lt_eval * column_lt_eval;
        let neg_inf_field: F = self.negative_infinity.to_field();
        let calc_eval = eq_eval
            * lt_eval
            * batching_challenges.iter().zip(input_evals.iter()).fold(
                F::ZERO,
                |acc, (chal, eval)| {
                    acc + *chal * (mask_eval * *eval + neg_inf_field * (F::ONE - mask_eval))
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
            .zip(input_evals.iter())
            .fold(F::ZERO, |acc, (c, e)| acc + (*c) * (*e));
        let full_point = sumcheck_point
            .iter()
            .chain(batching_point.iter())
            .copied()
            .collect::<Vec<_>>();
        let input_claim = vec![Claim::<F>::new(full_point, combined_eval)];

        Ok(input_claim)
    }

    /// Given the row and column points, evaluates the row and column less than polynomials
    fn evaluate_row_column_lt_polys<F: PrimeField>(
        &self,
        row_point: &[F],
        column_point: &[F],
        unpadded_seq_len: usize,
    ) -> Result<(F, F)> {
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
            .map(|b| F::from(b as u64))
            .collect::<Vec<F>>();
        let row_eval = eval_zeroifier_mle(row_point, &seq_len_bits);
        let column_eval = eval_zeroifier_mle(column_point, &seq_len_bits);
        Ok((row_eval, column_eval))
    }

    fn verify_additional_mask_poly_eval<F, T, PCS>(
        &self,
        row_point: &[F],
        column_point: &[F],
        eval: F,
        local_mask_proof: &IOPProof<F>,
        verifier: &mut Verifier<F, T, PCS>,
    ) -> anyhow::Result<()>
    where
        F: PrimeField,
        PCS: CommitmentScheme,
        T: Transcript,
    {
        let AttentionSpan::Local(window) = self.span else {
            Err(anyhow!("Found sumcheck proof for full attention mask"))?
        };
        let num_row_vars = row_point.len();
        let seq_len = 1usize << num_row_vars;
        ensure!(
            window < seq_len,
            "Sliding window not smaller than sequence length"
        );

        let num_col_vars = column_point.len();
        // compute the evaluation for the padding polynomial, i.e., the polynomial P[i,j] = 1 iff j >= seq_len - window
        let pad_col_bits = to_bit_sequence_le(seq_len - window, num_col_vars)
            .map(|b| F::from(b as u64))
            .collect_vec();
        let padded_eval = F::ONE - eval_zeroifier_mle(column_point, &pad_col_bits);

        let aux_info = VPAuxInfo {
            max_degree: 2,
            max_num_variables: num_col_vars,
            ..Default::default()
        };
        let subclaim = IOPVerifierState::<F>::verify(
            eval - padded_eval,
            local_mask_proof,
            &aux_info,
            verifier.transcript,
        );

        let sumcheck_point = &subclaim.point;

        // evaluation for the upper diagonal MLE produced by the sumcheck. The evaluation is computed
        // using the formula for the lower trianfular MLE, but swapping the row point with the
        // column point (i.e., the ones given by sumcheck challenges)
        let upper_eval = eval_zeroifier_mle(row_point, sumcheck_point);

        // evaluation for the shift matrix MLE
        let shift_eval = eval_shift_matrix(sumcheck_point, column_point, window - 1);

        let calc_eval = upper_eval * shift_eval;
        ensure!(
            calc_eval == subclaim.expected_evaluation,
            "Mask sumcheck verification failed, expected evaluation {:?} got {:?}",
            subclaim.expected_evaluation,
            calc_eval
        );

        Ok(())
    }
}

/// Compute the evaluation for the MLE of a square shift matrix S for the given `shift` at the row and column points
/// provided as input. The shift matrix S is defined as S[i,j] = 1 iff i <= j + shift;
/// given a generic square matrix `A`, right-multiplying `S` to `A` shifts the columns of `A` `shift` times to the left,
/// leaving 0 columns in place of the shifted columns. The shift matrix is employed to construct the local attention mask  
fn eval_shift_matrix<F: PrimeField>(row_point: &[F], column_point: &[F], shift: usize) -> F {
    let b1 = evals(row_point);
    let num_row_vars = row_point.len();
    let num_col_vars = column_point.len();
    assert_eq!(b1.len(), 1 << num_row_vars);
    (0..(1 << num_row_vars))
        .zip(b1)
        .fold(F::ZERO, |sum, (row, beta)| {
            if row < shift {
                sum
            } else {
                let col = row - shift;
                let col_le_bits = to_bit_sequence_le(col, num_col_vars)
                    .map(|b| F::from(b as u64))
                    .collect_vec();
                let selector = beta * identity_eval(column_point, &col_le_bits);
                sum + selector
            }
        })
}
#[derive(Debug, Clone)]
/// Struct storing all information to verify a [`AttentionMaskProof`]. We prove and verify the mask applied to each individual 2D sub tensor in a batched fashion.
/// This struct hold the information needed and is constructed from the shaps and the last claim.
pub(crate) struct MaskVerifyingData<F: PrimeField> {
    /// These values are the evaluations of the eq-poly for the higher dims that aren't from padding
    batching_challenges: Vec<F>,
    /// This is the point used to make the batch challenges
    batching_point: Vec<F>,
    /// This is evaluations of the eq-poly for each of the rank-2 tensors that the mask is applied to
    eq_point: Vec<F>,
}

impl<F: PrimeField> MaskVerifyingData<F> {
    /// Create a new [`MaskVerifyingData`]
    pub fn new(batching_challenges: Vec<F>, batching_point: Vec<F>, eq_point: Vec<F>) -> Self {
        MaskVerifyingData {
            batching_challenges,
            batching_point,
            eq_point,
        }
    }

    pub fn new_from_claim_and_shape_data(
        claim: &Claim<F>,
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
            .collect::<Vec<F>>();

        let batching_challenges = dim_points[..rank - 2]
            .iter()
            .zip(unpadded_input_shape[..rank - 2].iter())
            .fold(vec![F::ONE], |mut acc, (point_slice, dim)| {
                let evals = evals(point_slice);
                acc = acc
                    .into_iter()
                    .flat_map(|c| evals.iter().take(*dim).map(|e| c * *e).collect::<Vec<F>>())
                    .collect::<Vec<F>>();
                acc
            });

        let batching_point = dim_points[..rank - 2]
            .iter()
            .rev()
            .flat_map(|p| *p)
            .copied()
            .collect::<Vec<F>>();

        Ok(MaskVerifyingData::new(
            batching_challenges,
            batching_point,
            eq_point,
        ))
    }
}
