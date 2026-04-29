//! Code for verifying a Softmax layer proof.
use crate::lookup::operation::{
    generic_prove::GenericLookupProof,
    generic_verify::{LookupVerifyResult, verify_lookup_op},
};

use super::*;

impl SoftmaxCtx {
    pub(crate) fn verify_internal<F, PCS, T>(
        &self,
        proof: &SoftmaxProof<F, PCS>,
        last_claims: &[&Claim<F>],
        verifier: &mut Verifier<F, T, PCS>,
        shape_step: &ShapeStep,
        node_id: NodeId,
    ) -> Result<Vec<Claim<F>>>
    where
        F: PrimeField,
        PCS: CommitmentScheme,
        T: Transcript,
    {
        let SoftmaxProof {
            logup_proof,
            commitments,
            sumcheck_proof,
            evaluations,
            shift_evaluations,
        } = proof;

        let generic_lookup_proof = GenericLookupProof::<F, PCS> {
            logup_proof: logup_proof.clone(),
            sumcheck_proof: sumcheck_proof.clone(),
            evaluations: evaluations.clone(),
            commitments: commitments.clone(),
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
            node_id,
        )?;

        Ok(input_claims)
    }
}
