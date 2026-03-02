//! Internal code for verifying RMSNorm layers.

use crate::lookup::operation::{
    generic_prove::GenericLookupProof,
    generic_verify::{LookupVerifyResult, verify_lookup_op},
};

use super::*;
#[derive(Debug, Clone, Serialize, Deserialize, Copy)]
pub(crate) struct RMSNormLookupVerifier {
    pub(crate) right_shift: Element,
    pub(crate) normalisation_sum_sq: Element,
    pub(crate) error_bound: Element,
    pub(crate) intermediate_bit_size: usize,
    pub(crate) has_weight: bool,
}

impl RMSNormLookupVerifier {
    fn new<E, PCS>(ctx: &RMSNormCtx, proof: &RMSNormProof<E, PCS>) -> Self
    where
        E: ExtensionField,
        PCS: PolynomialCommitmentScheme<E>,
    {
        let intermediate_bit_size = ctx.input_scaling.bit_size() + 1;
        let normalisation_sum_sq = (ctx.normalisation_dim_size as f32
            / (ctx.normalisation_scaling.scale().powi(2)))
        .round_ties_even() as Element;
        let error_bound = (ctx.normalisation_dim_size as f32
            * (2.0f32 * ctx.normalisation_scaling.scale().recip() + 1.0f32))
            .round_ties_even() as Element;
        RMSNormLookupVerifier {
            right_shift: proof.right_shift,
            normalisation_sum_sq,
            error_bound,
            intermediate_bit_size,
            has_weight: ctx.alpha_key.is_some(),
        }
    }

    pub(crate) fn new_from_parts(
        right_shift: Element,
        normalisation_sum_sq: Element,
        error_bound: Element,
        intermediate_bit_size: usize,
        has_weight: bool,
    ) -> Self {
        RMSNormLookupVerifier {
            right_shift,
            normalisation_sum_sq,
            error_bound,
            intermediate_bit_size,
            has_weight,
        }
    }
}

impl LookupOp for RMSNormLookupVerifier {
    fn intermediate_bit_size(&self) -> usize {
        self.intermediate_bit_size
    }

    fn right_shift(&self) -> usize {
        self.right_shift.unsigned_abs() as usize
    }

    fn variant(&self) -> LookupVariant {
        LookupVariant::Normalisation {
            normalised_magnitude_value: self.normalisation_sum_sq,
            magnitude_error_bound: self.error_bound,
            normalised_sum_value: None,
            has_weight: self.has_weight,
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
            "Cannot Apply the operation using RMSNormLookupVerifier"
        ))
    }

    fn generate_witness(
        &self,
        _input: Tensor<Element>,
        _value_table: &Table,
    ) -> Result<crate::lookup::operation::LookupOpWitness> {
        Err(anyhow!(
            "Cannot generate a witness for the operation using RMSNormLookupVerifier"
        ))
    }
}

impl RMSNormCtx {
    pub(crate) fn verify_internal<E, PCS, T>(
        &self,
        proof: &RMSNormProof<E, PCS>,
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
        let lookup_op = RMSNormLookupVerifier::new(self, proof);
        let RMSNormProof {
            logup_proof,
            commitment,
            io_proof,
            io_evaluations,
            alpha_evaluation,
            ..
        } = proof;

        let generic_lookup_proof = GenericLookupProof::<E, PCS> {
            logup_proof: logup_proof.clone(),
            sumcheck_proof: io_proof.clone(),
            evaluations: io_evaluations.clone(),
            commitment: commitment.clone(),
            weight_evaluation: *alpha_evaluation,
            shift_evaluations: None,
        };

        let normalisation_scaling_factor = self.normalisation_scaling();
        let table = Table::new_normalisation(normalisation_scaling_factor.bit_size() + 1);

        let LookupVerifyResult {
            input_claims,
            weight_evaluation_point,
        } = verify_lookup_op(
            &lookup_op,
            last_claim,
            shape_step,
            &table,
            &generic_lookup_proof,
            verifier,
            id,
        )?;

        match (*alpha_evaluation, weight_evaluation_point, self.alpha_key()) {
            (Some(alpha_eval), Some(weight_eval_point), Some(alpha_key)) => {
                // Add the weight commitment claim to the verifier
                let claims_map = HashMap::from([(
                    alpha_key.clone(),
                    Claim::<E>::new(weight_eval_point, alpha_eval),
                )]);
                verifier.add_common_claims(id, claims_map);
            }
            (None, None, None) => {
                // No alpha key, nothing to check
            }
            _ => {
                bail!("Inconsistent alpha key/evaluation state in RMSNorm verification");
            }
        }

        Ok(input_claims)
    }
}
