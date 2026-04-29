//! Internal code for verifying the [`Requant`] layer.

use crate::lookup::operation::{
    generic_prove::GenericLookupProof,
    generic_verify::{LookupVerifyResult, verify_lookup_op},
};

use super::*;

impl RequantCtx {
    /// Method that verifies requantisation has been performed correctly when supplied with a [`RequantProof`].
    /// It verifies both lookup argument proofs, calculates the initial claim for the sumcheck proof using the lookup argument claims
    /// and then verifies the sumcheck using this initial claim. It then takes the output claims provided by the prover, checks they relate to the sumcheck
    /// subclaim, adds them to the list of claims of commitment openings and then calculates the next claim.
    pub(crate) fn verify_requant<F, T, PCS>(
        &self,
        verifier: &mut Verifier<F, T, PCS>,
        last_claim: &Claim<F>,
        proof: &RequantProof<F, PCS>,
        shape_step: &ShapeStep,
        node_id: NodeId,
    ) -> anyhow::Result<Vec<Claim<F>>>
    where
        F: PrimeField,
        T: Transcript,
        PCS: CommitmentScheme,
    {
        let RequantProof {
            io_accumulation,
            io_eval: evaluations,
            logup_proof: lookup,
            commitments,
        } = proof;

        let lookup_op = &self.requant.activation_lookup_data;

        let generic_lookup_proof = GenericLookupProof::<F, PCS> {
            logup_proof: lookup.clone(),
            sumcheck_proof: io_accumulation.clone(),
            evaluations: evaluations.clone(),
            commitments: commitments.clone(),
            weight_evaluation: None,
            shift_evaluations: None,
        };

        let LookupVerifyResult { input_claims, .. } = verify_lookup_op(
            lookup_op,
            last_claim,
            shape_step,
            &lookup_op.table,
            &generic_lookup_proof,
            verifier,
            node_id,
        )?;
        Ok(input_claims)
    }
}
