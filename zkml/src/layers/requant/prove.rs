//! Internal code for proving the [`Requant`] layer.

use crate::lookup::operation::generic_prove::{
    GenericLookupProof, LookupProverResult, prove_lookup_op,
};

use super::*;

impl Requant {
    #[timed::timed_instrument(name = "Prover::prove_requant")]
    /// Method that proves requantisation was performed correctly. It does this by running any required lookups and then linking the `last_claim` to the
    /// `input` via a series of Sumchecks.
    pub(crate) fn prove_step<E, T: Transcript<E>, PCS>(
        &self,
        prover: &mut Prover<E, T, PCS>,
        last_claim: &Claim<E>,
        step: &Step<Element>,
        id: NodeId,
    ) -> anyhow::Result<Vec<Claim<E>>>
    where
        E: ExtensionField,
        PCS: PolynomialCommitmentScheme<E> + Send + Sync,
        PCS::CommitmentWithWitness: Serialize + DeserializeOwned + Send + Sync,
        PCS::ProverParam: Send + Sync,
    {
        let lookup_op = &self.activation_lookup_data;

        let LookupProverResult {
            generic_proof,
            input_claims,
            ..
        } = prove_lookup_op(
            lookup_op,
            last_claim,
            step,
            &lookup_op.table,
            None,
            prover,
            id,
        )?;

        let GenericLookupProof {
            logup_proof,
            sumcheck_proof,
            evaluations,
            commitment,
            ..
        } = generic_proof;

        let proof = RequantProof {
            io_accumulation: sumcheck_proof,
            io_eval: evaluations,
            logup_proof,
            commitment,
        };

        // Add the proof to the prover
        prover.push_proof(id, LayerProof::Requant(proof));
        // Return the input claim for the next layer
        Ok(input_claims)
    }
}
