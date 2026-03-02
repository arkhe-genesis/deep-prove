//! Code for verifying a Softmax layer proof.
use crate::lookup::operation::{
    generic_prove::GenericLookupProof,
    generic_verify::{LookupVerifyResult, verify_lookup_op},
};

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
        let SoftmaxProof {
            logup_proof,
            commitment,
            sumcheck_proof,
            evaluations,
            shift_evaluations,
        } = proof;

        let generic_lookup_proof = GenericLookupProof::<E, PCS> {
            logup_proof: logup_proof.clone(),
            sumcheck_proof: sumcheck_proof.clone(),
            evaluations: evaluations.clone(),
            commitment: commitment.clone(),
            weight_evaluation: None,
            shift_evaluations: Some(shift_evaluations.clone()),
        };

        let table = self.quant_info.lut;

        let LookupVerifyResult { input_claims, .. } = verify_lookup_op(
            &self.quant_info,
            last_claims[0],
            shape_step,
            &table,
            &generic_lookup_proof,
            verifier,
            self.node_id,
        )?;

        Ok(input_claims)
    }
}
