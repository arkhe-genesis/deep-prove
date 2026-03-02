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
    pub(crate) fn verify_requant<E, T, PCS>(
        &self,
        verifier: &mut Verifier<E, T, PCS>,
        last_claim: &Claim<E>,
        proof: &RequantProof<E, PCS>,
        shape_step: &ShapeStep,
    ) -> anyhow::Result<Vec<Claim<E>>>
    where
        E: ExtensionField,
        T: Transcript<E>,
        PCS: PolynomialCommitmentScheme<E>,
    {
        let RequantProof {
            io_accumulation,
            io_eval: evaluations,
            logup_proof: lookup,
            commitment: commit,
        } = proof;

        let lookup_op = &self.requant.activation_lookup_data;

        let generic_lookup_proof = GenericLookupProof::<E, PCS> {
            logup_proof: lookup.clone(),
            sumcheck_proof: io_accumulation.clone(),
            evaluations: evaluations.clone(),
            commitment: commit.clone(),
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
            self.node_id,
        )?;
        Ok(input_claims)
    }
}
