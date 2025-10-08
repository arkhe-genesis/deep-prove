use crate::iop::model_output_claims;
use std::collections::HashMap;

use crate::{
    Claim, Element,
    commit::mmcs_context::CommitmentVerifier,
    graph::PortID,
    iop::{
        ChallengeStorage,
        context::VerifierContext,
        prover::{MergeClaimNodeProof, MergeClaimsProof},
    },
    layers::{
        LayerCtx, LayerProof,
        provable::{OpInfo, VerifiableCtx},
    },
    lookup::{context::LookupContext, logup_gkr::verifier::verify_logup_proof_multiple_sizes},
    model::NodeID,
    tensor::{Tensor, TensorKey},
    try_unzip,
};
use anyhow::{Context as _, anyhow, ensure};
use ff_ext::ExtensionField;
use itertools::Itertools;
use mpcs::{Point, PolynomialCommitmentScheme};
use std::collections::BTreeMap;
use tracing::trace;

use serde::{Deserialize, Serialize, de::DeserializeOwned};
use transcript::Transcript;

use super::{Proof, TableProof};

/// What the verifier must have besides the proof
#[derive(Clone, Serialize, Deserialize)]
pub struct IO<E> {
    /// Input of the inference given to the model
    pub input: Vec<Tensor<E>>,
    /// Output of the inference
    pub output: Vec<Tensor<E>>,
}

impl<E> IO<E> {
    pub fn new(input: Vec<Tensor<E>>, output: Vec<Tensor<E>>) -> Self {
        Self { input, output }
    }
    pub fn inputs(&self) -> &[Tensor<E>] {
        &self.input
    }
}

impl<E: ExtensionField> IO<E> {
    pub fn to_element(self) -> IO<Element> {
        IO {
            input: self
                .input
                .into_iter()
                .map(|t| t.map_data(|e| e.to_canonical_u64_vec()[0] as Element))
                .collect(),
            output: self
                .output
                .into_iter()
                .map(|t| t.map_data(|e| e.to_canonical_u64_vec()[0] as Element))
                .collect(),
        }
    }
}

pub struct Verifier<'a, E: ExtensionField, T: Transcript<E>, PCS>
where
    E::BaseField: Serialize + DeserializeOwned,
    E: Serialize + DeserializeOwned,
    PCS: PolynomialCommitmentScheme<E>,
{
    pub(crate) io: IO<E>,
    pub(crate) commit_verifier: CommitmentVerifier<E, PCS>,
    pub(crate) transcript: &'a mut T,
    pub(crate) challenge_storage: ChallengeStorage<E>,
}

impl<'a, E: ExtensionField, T: Transcript<E>, PCS: PolynomialCommitmentScheme<E>>
    Verifier<'a, E, T, PCS>
where
    E::BaseField: Serialize + DeserializeOwned,
    E: Serialize + DeserializeOwned,
{
    pub(crate) fn new(transcript: &'a mut T, io: IO<E>) -> Self {
        let commit_verifier = CommitmentVerifier::<E, PCS>::new();
        Self {
            io,
            commit_verifier,
            transcript,
            challenge_storage: ChallengeStorage::<E>::default(),
        }
    }

    pub(crate) fn verify(
        mut self,
        ctx: &VerifierContext<E, PCS>,
        proof: Proof<E, PCS>,
    ) -> anyhow::Result<()> {
        // 1. Instantiate everything and append relevant info to the transcript
        let mut numerators = Vec::<E>::new();
        let mut denominators = Vec::<E>::new();

        ctx.write_to_transcript(self.transcript)?;

        // iterate over the step proofs in inference order
        for (node_id, layer_ctx) in ctx.model.nodes.forward_iter() {
            if !layer_ctx.has_proof() {
                // if the current node is not provable, there is no proof, so we can skip it
                continue;
            }
            let node_proof = proof
                .steps
                .get(&node_id)
                .ok_or(anyhow!("Proof for node {node_id} not found"))?;
            if let Some((num, denom)) = node_proof.get_lookup_data() {
                numerators.extend(num.into_iter());
                denominators.extend(denom.into_iter());
            }
            layer_ctx.write_proof_to_transcript(node_proof, self.transcript)?;
        }

        proof.table_proofs.iter().try_for_each(|proof| {
            let (nums, denoms) = proof.lookup.fractional_outputs();
            numerators.extend(nums);
            denominators.extend(denoms);
            proof.write_commitment(self.transcript)
        })?;

        // Here we generate and store all lookup related challenges
        // TODO: make this part of verifier struct
        self.challenge_storage = if ctx.lookup.is_empty() {
            ChallengeStorage::<E>::default()
        } else {
            ChallengeStorage::<E>::initialise(ctx, self.transcript)
        };

        // 2. Derive output claims
        // first, we bind each output to the node that computes it, so that we know whether we
        // need to compute the output claim or not
        let out_claims = model_output_claims(self.transcript, &self.io.output);
        let shape_steps = ctx.model.shape_steps(
            &ctx.unpadded_input_shapes,
            &self
                .io
                .input
                .iter()
                .map(|t| t.shape().clone())
                .collect_vec(),
        )?;

        // 4. Verify each proof sequentially, Always make sure the proof corresponds to the expected type of proof in the context.
        // claims_by_layer is a map from node_id to the claims this layer generated, e.g. the claims that corresponds
        // to the _inputs_ of that layer.
        let mut claims_produced_by_layers: HashMap<NodeID, Vec<Claim<E>>> = HashMap::new();
        for (node_id, layer) in ctx.model.nodes.backward_iter() {
            let node_proof = if layer.has_proof() {
                proof
                    .steps
                    .get(&node_id)
                    .ok_or(anyhow!("Proof for node {node_id} not found"))?
            } else {
                &LayerProof::Dummy
            };
            let shape_step = shape_steps
                .get(&node_id)
                .ok_or(anyhow!("Shape for node {node_id} not found"))?;
            trace!(
                "Verifying proof {} for node {node_id}",
                node_proof.variant_name(),
            );
            // all the claims that are incoming to this node
            let claims_for_node =
                ctx.model
                    .claims_for_node(&node_id, &claims_produced_by_layers, &out_claims)?;
            let claims_for_verify = self.verify_merge_claims_proof(
                claims_for_node,
                proof.merge_claim_proofs.get(&node_id),
            )?;

            let claims = {
                if layer.is_provable() {
                    // we verify the proof
                    layer
                        .verify(
                            node_proof,
                            &claims_for_verify.iter().collect_vec(),
                            &mut self,
                            shape_step,
                        )
                        .context(format!("Verification failed for node with ID {node_id}"))?
                } else {
                    // we only propagate the claims, without changing them, as a non-provable layer
                    // shouldn't change the input values
                    claims_for_verify
                }
            };
            claims_produced_by_layers.insert(node_id, claims);
        }

        // 5. Verify the lookup table proofs
        if !proof.table_proofs.is_empty() {
            verify_table::<_, _, _>(
                &proof.table_proofs[0],
                &ctx.lookup,
                ctx.commitment_ctx.table_node_id(),
                &mut self.commit_verifier,
                self.transcript,
                &self.challenge_storage,
            )?;
        }

        // get each claim associated with the corresponding input node and the index in the input vector
        let input_claims = ctx.model.input_claims(&claims_produced_by_layers)?;

        // 6. input verification: evaluating the input at the random evaluation point from the sumcheck
        let num_inputs = self.io.input.len();
        for (node_id, claims) in input_claims.into_iter() {
            let (inputs, claims): (Vec<_>, Vec<_>) = try_unzip(claims.into_iter()
                // each claim is positioned at the 
                .map(|(index, claim)| {
                    ensure!(*index < num_inputs,
                        "Processing claim associated to input {index}, but there are only {num_inputs} inputs",
                    );
                    Ok((
                        &self.io.input[*index],
                        claim,
                    ))
                }))?;
            let layer_ctx = ctx
                .model
                .nodes
                .node(&node_id)
                .ok_or(anyhow!("Node {node_id} not found"))?;
            <LayerCtx<E> as VerifiableCtx<E, PCS>>::verify_input_claim(
                layer_ctx,
                inputs.as_slice(),
                &claims,
            )?;
        }

        // 7. verify the opening of the accumulation of claims
        self.commit_verifier
            .verify(&ctx.commitment_ctx, proof.commit, self.transcript)?;

        let num_len = numerators.len();
        // 8. verify that the accumulated numerator is zero and accumulated denominator is non-zero
        let (final_num, final_denom) = numerators.into_iter().zip(denominators).fold(
            (E::ZERO, E::ONE),
            |(acc_num, acc_denom), (num, denom)| {
                (acc_num * denom + num * acc_denom, acc_denom * denom)
            },
        );

        ensure!(
            final_num == E::ZERO,
            "Final numerator was non-zero, got: {final_num:?} - numerator.len(): {num_len}"
        );
        ensure!(
            final_denom != E::ZERO,
            "Final denominator was zero, lookup arguments are invalid"
        );

        Ok(())
    }

    fn verify_merge_claims_proof(
        &mut self,
        claims: BTreeMap<PortID, Vec<&Claim<E>>>,
        proof: Option<&MergeClaimsProof<E>>,
    ) -> anyhow::Result<Vec<Claim<E>>> {
        if proof.is_none() {
            ensure!(claims.iter().all(|(_, claims)| claims.len() == 1));
            return Ok(claims
                .into_values()
                .map(|claims| claims[0].clone())
                .collect());
        }
        let proof = proof.unwrap();
        claims
            .into_iter()
            .map(|(port, claims)| {
                if claims.len() == 1 {
                    // there is only one claim, no need to merge anything
                    Ok(claims[0].clone())
                } else {
                    let merge_claim_proof = proof.get_proof(*port).ok_or(anyhow!(
                        "Merge claim proof for output index {} not found",
                        port
                    ))?;
                    self.verify_merge_claim_proof(&claims, merge_claim_proof)
                }
            })
            .collect()
    }

    fn verify_merge_claim_proof(
        &mut self,
        claims: &[&Claim<E>],
        proof: &MergeClaimNodeProof<E>,
    ) -> anyhow::Result<Claim<E>> {
        proof.verify_proof(self.transcript, claims)
    }

    pub(crate) fn add_common_claims(
        &mut self,
        node_id: NodeID,
        claims: HashMap<TensorKey, Claim<E>>,
    ) {
        self.commit_verifier.add_common_claims(
            claims
                .into_iter()
                .map(|(poly_id, claim)| (poly_id, HashMap::from([(node_id, claim)])))
                .collect(),
        )
    }
}

/// Verifies an inference proof given a context, a proof and the input / output of the model.
pub fn verify<E, T: Transcript<E>, PCS: PolynomialCommitmentScheme<E>>(
    ctx: &VerifierContext<E, PCS>,
    proof: Proof<E, PCS>,
    io: IO<E>,
    transcript: &mut T,
) -> anyhow::Result<()>
where
    E::BaseField: Serialize + DeserializeOwned,
    E: ExtensionField + Serialize + DeserializeOwned,
{
    let verifier = Verifier::new(transcript, io);
    verifier.verify(ctx, proof)
}

fn verify_table<E, T: Transcript<E>, PCS: PolynomialCommitmentScheme<E>>(
    proof: &TableProof<E, PCS>,
    lookup_ctx: &LookupContext,
    table_node_id: NodeID,
    witness_verifier: &mut CommitmentVerifier<E, PCS>,
    t: &mut T,
    challenge_storage: &ChallengeStorage<E>,
) -> anyhow::Result<()>
where
    E::BaseField: Serialize + DeserializeOwned,
    E: ExtensionField + Serialize + DeserializeOwned,
{
    // 1. Verify the lookup proof
    let TableProof {
        multiplicity_commit,
        lookup,
    } = proof;
    let batch_claim = verify_logup_proof_multiple_sizes(lookup, t)?;

    let poly_evals = batch_claim.poly_evals();
    let point = batch_claim.point();
    let point_len = point.len();
    let alpha = batch_claim.alpha();
    let lambda = batch_claim.lambda();

    let (mult_claims, _, calc_claim, _) = lookup_ctx.iter().try_fold(
        (vec![], 0, E::ZERO, E::ONE),
        |(mut acc, skip, eval_acc, chal_acc), tt| {
            let take = tt.num_columns() + 1;
            let nv = tt.multiplicity_poly_vars();
            let evals = poly_evals
                .iter()
                .skip(skip)
                .take(take)
                .copied()
                .collect::<Vec<E>>();
            let mult_eval = evals[0];
            let current_point = point[point_len - nv..].to_vec();
            let mut column_evals = tt.evaluate_table_columns(&current_point)?;

            acc.push((point[point_len - nv..].to_vec(), mult_eval));
            if tt.has_committed_claims() {
                column_evals.push(evals[take - 1]);
                witness_verifier.add_table_claim(
                    table_node_id,
                    tt,
                    Claim::<E>::new(point[point_len - nv..].to_vec(), evals[take - 1]),
                );
            }

            let (constant_challenge, csc) = challenge_storage
                .get_challenges_by_name(&tt.name())
                .ok_or(anyhow!("No challenges for table type {}", tt.name()))?;
            let column_eval = column_evals
                .into_iter()
                .fold((constant_challenge, E::ONE), |(acc, csc_acc), e| {
                    (acc + csc_acc * e, csc_acc * csc)
                })
                .0;
            Result::<(_, _, _, _), anyhow::Error>::Ok((
                acc,
                skip + take,
                eval_acc + chal_acc * (mult_eval + lambda * column_eval),
                chal_acc * alpha,
            ))
        },
    )?;

    let grouped = mult_claims
        .into_iter()
        .into_group_map()
        .into_iter()
        .sorted_by(|a, b| Ord::cmp(&b.0.len(), &a.0.len()))
        .collect::<Vec<(Point<E>, Vec<E>)>>();

    witness_verifier.add_witness_claim(table_node_id, multiplicity_commit.clone(), grouped);

    ensure!(
        calc_claim == batch_claim.claim(),
        "Table Proof was incorrect, calculated claim: {:?} was not equal to claim from LogUp proof {:?}",
        calc_claim,
        batch_claim.claim()
    );

    Ok(())
}
