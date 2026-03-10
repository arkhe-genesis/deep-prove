//! Module containing code for proving [`LayerNorm`] layers.

use multilinear_extensions::mle::{IntoMLE, MultilinearExtension};

use crate::{
    layers::LayerProof,
    lookup::operation::generic_prove::{GenericLookupProof, LookupProverResult, prove_lookup_op},
    to_base,
};

use std::collections::HashMap;

use super::*;

impl LayerNorm<Element> {
    pub(crate) fn prove_internal<E, T, PCS>(
        &self,
        id: NodeId,
        last_claim: &Claim<E>,
        step_data: &Step<Element>,
        prover: &mut Prover<E, T, PCS>,
    ) -> Result<Vec<Claim<E>>>
    where
        E: ExtensionField,
        T: Transcript<E>,
        PCS: PolynomialCommitmentScheme<E> + Send + Sync,
        PCS::CommitmentWithWitness: Serialize + DeserializeOwned + Send + Sync,
        PCS::ProverParam: Send + Sync,
    {
        let lookup_op = step_data
            .node_outputs
            .try_layernorm_data()
            .ok_or(anyhow!(
                "Could not get LayerNorm proving data for lookup witness generation"
            ))?
            .to_proving_data()?;

        let (_, normalisation_scaling_factor) =
            self.get_quantisation_scaling_factors().ok_or(anyhow!(
                "Quantisation scaling factors not found for LayerNorm lookup witness generation"
            ))?;
        let table = Table::new_normalisation(normalisation_scaling_factor.bit_size() + 1);

        let wrapped_gamma = self.gamma.wrapped_tensor()?.clone();
        let gamma = Tensor::try_from(&wrapped_gamma)?;
        let weight_tensor = Some(gamma.as_ref());

        // Reconstruct the lookup output here
        let input = step_data.padded_input_tensor_at(0)?;
        let unpadded_input_shape = input.unpadded_shape();

        // Here we subtract the bias contribution from the last claim
        let unpadded_beta_shape = self.beta.unpadded_shape();
        let dim_points = input.shape().split_point(last_claim.point())?;
        let rank_diff = unpadded_input_shape.rank() - unpadded_beta_shape.rank();
        let unbroadcast_shape = std::iter::repeat_n(1usize, rank_diff)
            .chain(unpadded_beta_shape.iter().copied())
            .collect::<Vec<usize>>();
        let bias_lt_eval =
            unpadded_input_shape.broadcasting_evaluation(&dim_points, &unbroadcast_shape)?;
        let wrapped_beta = self.beta.wrapped_tensor()?.clone().pad_next_power_of_two();
        let beta_mle: MultilinearExtension<E> =
            to_base::<E, _>(wrapped_beta.get_data().iter()).into_mle();
        let beta_point = dim_points
            .last()
            .ok_or(anyhow!(
                "Could not get last dimension point for beta evaluation in LayerNorm proving"
            ))?
            .to_vec();
        let beta_eval = beta_mle.evaluate(&beta_point);

        let mut last_claim = last_claim.clone();
        last_claim.eval -= beta_eval * bias_lt_eval;

        let LookupProverResult {
            generic_proof,
            input_claims,
            weight_evaluation_point,
        } = prove_lookup_op(
            &lookup_op,
            &last_claim,
            step_data,
            &table,
            weight_tensor,
            prover,
            id,
        )?;

        let GenericLookupProof {
            logup_proof,
            sumcheck_proof,
            evaluations,
            commitment,
            weight_evaluation,
            shift_evaluations,
        } = generic_proof;

        let gamma_eval =
            weight_evaluation.ok_or(anyhow!("Missing gamma evaluation in LayerNorm proving"))?;

        let mean_evals =
            shift_evaluations.ok_or(anyhow!("Missing mean evaluations in LayerNorm proving"))?;

        let proof = LayerNormProof {
            logup_proof,
            right_shift: lookup_op.right_shift(),
            commitment,
            io_proof: sumcheck_proof,
            io_evaluations: evaluations,
            mean_evals,
            gamma_eval,
            beta_eval,
        };

        // Add the proof to the prover
        prover.push_proof(id, LayerProof::LayerNorm(proof));

        // Add the weight commitment claims to the prover
        let gamma_point = weight_evaluation_point.ok_or(anyhow!(
            "Missing gamma weight evaluation point in LayerNorm proving"
        ))?;

        let claims_map = HashMap::from([
            (
                CommitmentId::from(self.gamma.storage_key()),
                Claim::<E>::new(gamma_point, gamma_eval),
            ),
            (
                CommitmentId::from(self.beta.storage_key()),
                Claim::<E>::new(beta_point, beta_eval),
            ),
        ]);

        prover.add_common_claims(id, claims_map);

        // Return the input claim for the next layer
        Ok(input_claims)
    }
}
