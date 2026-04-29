//! Module contiaing code for generic verification of lookup operations.
use dp_crypto::{
    arkyper::{CommitmentScheme, transcript::Transcript},
    util::ceil_log2,
};
use itertools::Itertools;

use crate::{
    graph::NodeId,
    iop::{context::ShapeStep, verifier::Verifier},
    lookup::operation::{
        generic_prove::GenericLookupProof,
        inputs::{LookupEvaluations, proving::LookupSumcheckProof},
        variant::verifying::evaluate_dim_lt_poly,
    },
    poly_commit::verifier::VerifierClaim,
};

use super::*;

#[derive(Debug, Clone)]
pub struct LookupVerifyResult<F: PrimeField> {
    pub input_claims: Vec<Claim<F>>,
    pub weight_evaluation_point: Option<Vec<F>>,
}

impl<F: PrimeField> LookupVerifyResult<F> {
    pub fn new(input_claims: Vec<Claim<F>>, weight_evaluation_point: Option<Vec<F>>) -> Self {
        Self {
            input_claims,
            weight_evaluation_point,
        }
    }
}

/// Method to verify a lookup operation that has been proven by [`prove_lookup_op`](super::proving::prove_lookup_op). This method verifies the lookup proof, constructs the claims for the input witnesses and adds the commitment claims to the verifier for later use in the generic batch verification of the polynomial commitment scheme.
pub fn verify_lookup_op<L, F, T, PCS>(
    lookup_op: &L,
    last_claim: &Claim<F>,
    shape_step: &ShapeStep,
    table: &Table,
    proof: &GenericLookupProof<F, PCS>,
    verifier: &mut Verifier<F, T, PCS>,
    node_id: NodeId,
) -> Result<LookupVerifyResult<F>>
where
    L: LookupOp,
    F: PrimeField,
    T: Transcript,
    PCS: CommitmentScheme,
{
    let unpadded_input_shape = &shape_step.unpadded_input_shape[0];
    let GenericLookupProof {
        logup_proof,
        sumcheck_proof,
        evaluations,
        weight_evaluation,
        shift_evaluations,
        commitments,
        ..
    } = proof;
    // 1. Verify the lookup proof
    let chunking_info = lookup_op.chunking_info(table)?;
    let input_config = lookup_op.input_config(chunking_info, unpadded_input_shape);
    // Add the fractional outputs to the verifier's numerator and denominator storage
    input_config.sort_fractional_outputs::<F, T, PCS>(verifier, logup_proof)?;

    // Verify the actual logup proof.
    // Build the verifier instances for the LogUp proof
    let logup_instances =
        input_config.create_logup_verifier_instances::<F>(&verifier.challenge_storage)?;

    // Now Verify the LogUp proof
    let logup_claim =
        new_verify_logup_proof_multiple_sizes(logup_proof, &logup_instances, verifier.transcript)?;

    // 2. Construct the lookup sumcheck proof to be passed to the verification method.
    let lookup_sumcheck_proof = LookupSumcheckProof {
        sumcheck_proof: sumcheck_proof.clone(),
        evaluations: evaluations.clone(),
        sumcheck_point: vec![],
        weight_eval: *weight_evaluation,
    };

    let variant = lookup_op.variant();
    let chunking_info = lookup_op.chunking_info(table)?;
    let input_config = lookup_op.input_config(chunking_info, unpadded_input_shape);

    // We need to build the lt poly eval and multiply the shift evaluations if they exist
    let shift_evals = if let Some(evals) = shift_evaluations {
        let final_dim = unpadded_input_shape.dim(-1);
        let final_dim_vars = ceil_log2(final_dim);
        let lt_eval = evaluate_dim_lt_poly(&logup_claim.point()[..final_dim_vars], final_dim)?;
        let rescaled = evals.iter().map(|e| *e * lt_eval).collect::<Vec<F>>();
        Some(rescaled)
    } else {
        None
    };

    // Verify the linking sumcheck proof and get the challenges and point to be used in the next steps
    let (batching_challenges, sumcheck_point) = input_config.verify_linking_sumcheck(
        &lookup_sumcheck_proof,
        verifier.transcript,
        last_claim,
        &logup_claim,
        &shift_evals,
    )?;

    let logup_evals = logup_claim
        .output_claims()
        .iter()
        .map(|c| c.evaluation())
        .collect::<Vec<F>>();

    // 3. Construct the claims for the input witnesses and the evaluations for the commitment claims and add them to the verifier.
    let LookupEvaluations {
        input_commitment_evals,
        output_commitment_evals,
        normalisation_commitment_evals,
        input_claim_evals,
    } = input_config.construct_lookup_evaluations(
        &logup_evals,
        evaluations,
        &batching_challenges[1..],
        &shift_evals,
    )?;

    let mut commitment_claims = input_commitment_evals
        .into_iter()
        .map(|input_eval| Claim::new(logup_proof.point().to_vec(), input_eval))
        .chain(
            output_commitment_evals
                .into_iter()
                .map(|out_eval| Claim::new(sumcheck_point.clone(), out_eval)),
        )
        .collect_vec();

    let final_dim_vars = ceil_log2(shape_step.unpadded_input_shape[0].dim(-1));

    // Construct the input claims to be passed to the next layer.
    let input_claims = variant.produce_input_claims(
        unpadded_input_shape,
        lookup_op,
        last_claim.point(),
        logup_proof.point(),
        &sumcheck_point,
        input_claim_evals,
    )?;

    if matches!(variant, LookupVariant::Normalisation { .. }) {
        // In this case we don't need to deal with rounding constants and the fixed point multiplier
        // because the input evaluation comes directly from the Sumcheck (since we had to prove that the values in the lookup
        // are the elementwise product of the input tensor and a witness specific tensor of normalising values).
        for eval in normalisation_commitment_evals {
            commitment_claims.push(Claim::new(sumcheck_point[final_dim_vars..].to_vec(), eval))
        }
    }

    if let Some(shift_evals) = shift_evaluations {
        let shift_point = &logup_claim.point()[final_dim_vars..];
        for &eval in shift_evals {
            commitment_claims.push(Claim::new(shift_point.to_vec(), eval))
        }
    }

    verifier.commit_verifier.add_witness_claim(
        node_id,
        commitment_claims
            .into_iter()
            .zip_eq(commitments)
            .map(|(claim, comm)| VerifierClaim::from((comm.clone(), claim)))
            .collect(),
    );

    let weight_point = weight_evaluation.map(|_| sumcheck_point[..final_dim_vars].to_vec());

    Ok(LookupVerifyResult::new(input_claims, weight_point))
}
