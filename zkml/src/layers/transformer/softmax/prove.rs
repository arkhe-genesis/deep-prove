//! Code for proving a Softmax layer.

use crate::lookup::operation::generic_prove::{
    GenericLookupProof, LookupProverResult, prove_lookup_op,
};

use super::*;

impl Softmax<Element> {
    #[allow(clippy::type_complexity)]
    pub(crate) fn prove_internal<
        E: ExtensionField,
        PCS: PolynomialCommitmentScheme<E> + Send + Sync,
        T: transcript::Transcript<E>,
    >(
        &self,
        node_id: NodeId,
        last_claims: Vec<&Claim<E>>,
        step: &Step<Element>,
        prover: &mut crate::Prover<E, T, PCS>,
    ) -> Result<Vec<Claim<E>>>
    where
        PCS::CommitmentWithWitness: Serialize + DeserializeOwned + Send + Sync,
        PCS::ProverParam: Send + Sync,
    {
        let lookup_op = self
            .quant_info()
            .ok_or(anyhow!("Could not get Softmax proving data for proving"))?;

        let table = &lookup_op.lut;

        let LookupProverResult {
            generic_proof,
            input_claims,
            ..
        } = prove_lookup_op(
            lookup_op,
            last_claims[0],
            step,
            table,
            None,
            prover,
            node_id,
        )?;

        let GenericLookupProof {
            logup_proof,
            sumcheck_proof,
            evaluations,
            commitment,
            shift_evaluations,
            ..
        } = generic_proof;

        let shift_evaluations = shift_evaluations.ok_or(anyhow!(
            "Could not get shift evaluations from lookup proof for Softmax"
        ))?;

        let softmax_proof = SoftmaxProof::<E, PCS> {
            logup_proof,
            commitment,
            sumcheck_proof,
            evaluations,
            shift_evaluations,
        };

        // Add the proof
        prover.push_proof(node_id, LayerProof::Softmax(softmax_proof));

        Ok(input_claims)
    }
}
