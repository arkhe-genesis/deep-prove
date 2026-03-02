//! Module containing code for verifying a [`LayerNorm`] layer.

use crate::lookup::operation::{
    generic_prove::GenericLookupProof,
    generic_verify::{LookupVerifyResult, verify_lookup_op},
};

use super::*;

#[derive(Debug, Clone, Serialize, Deserialize, Copy)]
pub(crate) struct LayerNormLookupVerifier {
    right_shift: Element,
    normalisation_sumsq: Element,
    magnitude_error_bound: Element,
    normalisation_sum: Element,
    sum_error_bound: Element,
    intermediate_bit_size: usize,
}

impl LayerNormLookupVerifier {
    fn new<E, PCS>(ctx: &LayerNormCtx, proof: &LayerNormProof<E, PCS>) -> Self
    where
        E: ExtensionField,
        PCS: PolynomialCommitmentScheme<E>,
    {
        let intermediate_bit_size = ctx.mean_scaling.bit_size() + 1;
        let normalisation_sumsq = (ctx.normalisation_dim_size as f32
            / (ctx.normalisation_scaling.scale().powi(2)))
        .round_ties_even() as Element;
        let magnitude_error_bound = (ctx.normalisation_dim_size as f32
            * (2.0f32 * ctx.normalisation_scaling.scale().recip() + 1.0f32))
            .round_ties_even() as Element;

        let sum_error_bound = ctx.normalisation_dim_size as Element;

        LayerNormLookupVerifier {
            right_shift: proof.right_shift as Element,
            normalisation_sumsq,
            magnitude_error_bound,
            normalisation_sum: 0,
            sum_error_bound,
            intermediate_bit_size,
        }
    }

    pub(crate) fn new_from_parts(
        right_shift: Element,
        normalisation_sumsq: Element,
        magnitude_error_bound: Element,
        sum_error_bound: Element,
        intermediate_bit_size: usize,
    ) -> Self {
        LayerNormLookupVerifier {
            right_shift,
            normalisation_sumsq,
            magnitude_error_bound,
            normalisation_sum: 0,
            sum_error_bound,
            intermediate_bit_size,
        }
    }
}

impl LookupOp for LayerNormLookupVerifier {
    fn intermediate_bit_size(&self) -> usize {
        self.intermediate_bit_size
    }

    fn right_shift(&self) -> usize {
        self.right_shift.unsigned_abs() as usize
    }

    fn variant(&self) -> LookupVariant {
        LookupVariant::Normalisation {
            normalised_magnitude_value: self.normalisation_sumsq,
            magnitude_error_bound: self.magnitude_error_bound,
            normalised_sum_value: Some((self.normalisation_sum, self.sum_error_bound)),
            has_weight: true,
        }
    }

    fn chunking_info(&self, table: &Table) -> Result<ChunkingInfo> {
        let max_bit_size = self.intermediate_bit_size + FIXED_POINT_SCALE;
        ChunkingInfo::new(self.right_shift(), table, max_bit_size, 1)
    }

    fn fixed_point_multiplier(&self) -> Element {
        1
    }

    fn is_signed(&self) -> bool {
        true
    }

    fn padding_value(&self) -> Element {
        0
    }

    fn apply(
        &self,
        _input: WrappedTensor<Element>,
        _table: &Table,
    ) -> Result<WrappedTensor<Element>> {
        Err(anyhow!(
            "Cannot Apply the operation using LayerNormLookupVerifier"
        ))
    }

    fn generate_witness(
        &self,
        _input: Tensor<Element>,
        _value_table: &Table,
    ) -> Result<crate::lookup::operation::LookupOpWitness> {
        Err(anyhow!(
            "Cannot generate a witness for the operation using LayerNormLookupVerifier"
        ))
    }
}

impl LayerNormCtx {
    pub(crate) fn verify_internal<E, PCS, T>(
        &self,
        proof: &LayerNormProof<E, PCS>,
        id: NodeId,
        verifier: &mut Verifier<E, T, PCS>,
        last_claim: &Claim<E>,
        shape_step: &ShapeStep,
    ) -> Result<Vec<Claim<E>>>
    where
        E: ExtensionField,
        PCS: PolynomialCommitmentScheme<E>,
        T: Transcript<E>,
    {
        // First we make the lookup verifier context
        let lookup_op = LayerNormLookupVerifier::new(self, proof);
        let LayerNormProof {
            logup_proof,
            commitment,
            io_proof,
            io_evaluations,
            gamma_eval,
            beta_eval,
            mean_evals,
            ..
        } = proof;

        // Here we subtract the bias contribution from the last claim
        let dim_points = shape_step.padded_input_shape[0].split_point(last_claim.point())?;
        let unpadded_input_shape = &shape_step.unpadded_input_shape[0];
        let rank_diff = unpadded_input_shape.rank() - 1;
        let unbroadcast_shape = std::iter::repeat_n(1usize, rank_diff)
            .chain(std::iter::once(self.normalisation_dim_size))
            .collect::<Vec<usize>>();
        let bias_lt_eval =
            unpadded_input_shape.broadcasting_evaluation(&dim_points, &unbroadcast_shape)?;

        let beta_point = dim_points
            .last()
            .ok_or(anyhow!(
                "Could not get last dimension point for beta evaluation in LayerNorm proving"
            ))?
            .to_vec();

        let mut last_claim = last_claim.clone();
        last_claim.eval -= *beta_eval * bias_lt_eval;

        let generic_lookup_proof = GenericLookupProof::<E, PCS> {
            logup_proof: logup_proof.clone(),
            sumcheck_proof: io_proof.clone(),
            evaluations: io_evaluations.clone(),
            commitment: commitment.clone(),
            weight_evaluation: Some(*gamma_eval),
            shift_evaluations: Some(mean_evals.clone()),
        };

        let table = Table::new_normalisation(self.normalisation_scaling.bit_size() + 1);

        let LookupVerifyResult {
            input_claims,
            weight_evaluation_point,
        } = verify_lookup_op(
            &lookup_op,
            &last_claim,
            shape_step,
            &table,
            &generic_lookup_proof,
            verifier,
            id,
        )?;

        let gamma_point = weight_evaluation_point.ok_or(anyhow!(
            "Missing gamma weight evaluation point in LayerNorm verification"
        ))?;
        let claims_map = HashMap::from([
            (
                self.gamma_key.clone(),
                Claim::<E>::new(gamma_point, *gamma_eval),
            ),
            (
                self.beta_key.clone(),
                Claim::<E>::new(beta_point, *beta_eval),
            ),
        ]);

        verifier.add_common_claims(id, claims_map);

        Ok(input_claims)
    }
}
