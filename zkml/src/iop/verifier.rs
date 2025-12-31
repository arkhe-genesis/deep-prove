use crate::{
    InitTranscript, Proof,
    iop::{
        ChunkProof, TableProof,
        chunking::{ChunkID, GroupIOClaims, initialize_transcript_for_chunk},
        compute_claim,
        context::ShapeStep,
    },
};

use crate::{
    Claim, Element, VectorTranscript,
    commit::mmcs_context::CommitmentVerifier,
    graph::{Node, NodeId, NodeInput, PortId},
    iop::{ChallengeStorage, context::VerifierContext, prover::MergeClaimsProof},
    layers::{
        LayerCtx, LayerProof,
        provable::{OpInfo, VerifiableCtx},
    },
    lookup::{context::LookupContext, logup_gkr::verifier::verify_logup_proof_multiple_sizes},
    tensor::{CommitmentId, Tensor},
};
use anyhow::{Context as _, anyhow, ensure};
use ff_ext::ExtensionField;
use itertools::Itertools;
use mpcs::{Point, PolynomialCommitmentScheme};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};
use tracing::{info, info_span, trace};
use transcript::Transcript;

/// What the verifier must have besides the proof
#[derive(Clone, Debug, Serialize, Deserialize)]
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
    PCS: PolynomialCommitmentScheme<E>,
{
    pub(crate) io: &'a IO<E>,
    pub(crate) commit_verifier: CommitmentVerifier<E, PCS>,
    pub(crate) transcript: &'a mut T,
    pub(crate) challenge_storage: ChallengeStorage<E>,
}

impl<'a, E: ExtensionField, T: Transcript<E>, PCS: PolynomialCommitmentScheme<E>>
    Verifier<'a, E, T, PCS>
{
    pub(crate) fn new(transcript: &'a mut T, io: &'a IO<E>) -> Self {
        let commit_verifier = CommitmentVerifier::<E, PCS>::new();
        Self {
            io,
            commit_verifier,
            transcript,
            challenge_storage: ChallengeStorage::<E>::default(),
        }
    }

    fn initialise_transcript(ctx: &VerifierContext<E, PCS>) -> anyhow::Result<T>
    where
        T: InitTranscript,
    {
        let mut transcript = T::new(T::InitData::from(b"model_proving"));
        ctx.write_to_transcript(&mut transcript)?;
        Ok(transcript)
    }

    pub(crate) fn verify_chunk(
        mut self,
        ctx: &VerifierContext<E, PCS>,
        proof: ChunkProof<E, PCS>,
        shape_steps: &HashMap<NodeId, ShapeStep>,
    ) -> anyhow::Result<()>
    where
        PCS::Commitment: PartialEq + Eq,
    {
        let chunk = &proof.chunk_data.model_chunk;
        let chunk_id = chunk.chunk_id;
        // Add chunk splitting info to the transcript
        chunk.add_chunk_data_to_transcript(self.transcript)?;

        let lookup_ctx = chunk.chunk_lookup_ctx(&ctx.lookup);

        // Instantiate everything and append relevant info to the transcript
        let mut numerators = Vec::<E>::new();
        let mut denominators = Vec::<E>::new();

        // iterate over the step proofs in inference order
        for (node_id, _) in chunk.subgraph.forward_inners() {
            let layer_ctx = ctx
                .model
                .nodes
                .node(node_id)
                .ok_or(anyhow!(
                    "Node {node_id} of chunk {chunk_id} not found in model context"
                ))?
                .as_inner()
                .unwrap_or_else(|| panic!("Node {node_id} must be an inner node in model context"));
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

        if let Some(table_proof) = &proof.table_proof {
            let (nums, denoms) = table_proof.lookup.fractional_outputs();
            numerators.extend(nums);
            denominators.extend(denoms);
            table_proof.write_commitment(self.transcript)?;
        }

        // Add chunk commitments to the transcript
        let chunk_commitments = &proof.chunk_data.commitments;
        chunk_commitments.add_to_transcript::<E, PCS, T>(chunk_id, self.transcript)?;

        // Here we generate and store all lookup related challenges
        // TODO: make this part of verifier struct
        self.challenge_storage = if lookup_ctx.is_empty() {
            ChallengeStorage::<E>::default()
        } else {
            ChallengeStorage::<E>::initialise(&lookup_ctx, self.transcript)
        };

        // compute the claims for the model outputs produced in this chunk, each identified by the
        // model output port ID
        let output_claims_by_port = chunk.model_outputs_in_chunk()?.into_iter()
            .try_fold(
                BTreeMap::new(), // we first collect all the output tensors, sorted by the output port ID
                |mut outputs, edge_id| {
                let output_edge = chunk.edge(&edge_id)?;
                 let target_node = chunk.subgraph.target_node(&edge_id)?;
                let output_id = target_node.as_output().ok_or(
                    anyhow!("Edge {edge_id} is not an output edge of the model")
                )?;
                ensure!(
                    output_edge.ports().len() == 1,
                    "Expected 1 port link for model output edge {edge_id} in chunk {chunk_id}, found {}",
                    output_edge.ports().len()
                );
                ensure!(
                    *output_id < self.io.output.len(),
                    "No model output found for {output_id}, there are only {} outputs",
                    outputs.len(),
                );
                let output_tensor = self.io.output[*output_id].clone();
                ensure!(
                    outputs.insert(output_id, output_tensor).is_none(),
                    "Found output tensor twice for output port {output_id} in chunk {chunk_id}"
                );
                Ok(outputs)
            })? // then, we compute the claims for each output
            .into_iter()
            .map(|(port_id, tensor)| {
                // For the output, we manually evaluate the MLE and check if it's the same as what prover
                // gave. Note prover could ellude that but it's simpler to avoid that special check right
                // now.
                (port_id, compute_claim(self.transcript, tensor))
            }).collect::<HashMap<_,_>>();

        let chunk_output_claims = proof
            .chunk_data
            .output_evals
            .iter()
            .map(|(port, poly_eval)| {
                let point = self.transcript.read_challenges(poly_eval.num_vars);
                (*port, Claim::new(point, poly_eval.eval))
            })
            .collect();

        // ===== Verify each proof sequentially =====
        //
        // always make sure the proof corresponds to the expected type of proof
        // in the context.
        let claims = chunk.subgraph.backward_iter().try_fold(
            HashMap::<NodeInput, Claim<E>>::new(),
            |mut claims, (node_id, _)| -> anyhow::Result<HashMap<NodeInput, Claim<E>>> {
                let node = ctx.model.nodes.node(node_id).
                    ok_or(anyhow!("Node {node_id} not found verifier context"))?;
                match node {
                    Node::Inner(layer) => {
                        let node_proof = if layer.has_proof() {
                            proof
                                .steps
                                .get(&node_id)
                                .ok_or(anyhow!("Proof for node {node_id} not found"))?
                        } else {
                            &LayerProof::Dummy
                        };

                        // In a bug-less situation, there is no reason for that
                        // to ever happen.
                        let shape_step = shape_steps
                            .get(&node_id)
                            .ok_or(anyhow!("Shape for node {node_id} not found"))?;
                        trace!(
                            "Verifying proof {} for node {node_id}",
                            node_proof.variant_name(),
                        );

                        // ===== Compute the claims generated by this node =====
                        //
                        // There is one claim per input port for this node. This
                        // claim is computed from the claims fed into this node
                        // output ports. These are the claims associated to the
                        // input port by which this node successors are
                        // connected to this node.
                        //
                        // --
                        //  |
                        //  -> A -> B
                        //  ->
                        //  |
                        // --
                        //
                        // A will generate two claims, for its ports 1 and 2,
                        // computed from its incoming claim on output port 0,
                        // which is the claim generated by B on its input port 0.
                        let claims_for_node = chunk.claims_for_node(
                            node_id,
                            &claims,
                            &chunk_output_claims,
                        )?;

                        // Merge claims that are redundant. If a node out port
                        // is fed to two distinct nodes, then this will result
                        // in this port receiving two claims. These must be
                        // aggregated into a single one.
                        let claims_for_verify = self.verify_merge_claims_proof(
                            claims_for_node,
                            proof.merge_claim_proofs.get(&node_id),
                        )?;

                        let my_claims = {
                            if layer.is_provable() {
                                // we verify the proof
                                layer
                                    .verify(
                                        node_proof,
                                        &claims_for_verify.iter().collect_vec(),
                                        &mut self,
                                        shape_step,
                                    )
                                    .with_context(|| format!(
                                        "Verification failed for node with ID {node_id}"
                                    ))?
                            } else {
                                // we only propagate the claims, without
                                // changing them, as a non-provable layer
                                // shouldn't change the input values
                                claims_for_verify
                            }
                        };

                        // Insert the claims generated by this node in the
                        // global claim register.
                        claims.extend(
                            my_claims
                                .into_iter()
                                .enumerate()
                                .map(|(i, claim)| (NodeInput::new(node_id, i), claim)),
                        );

                        // ===== Input verification =====
                        //
                        // evaluating the input at the random evaluation point
                        // from the sumcheck.
                        let (inputs, input_claims) = chunk.model_inputs_in_chunk()?
                            .into_iter()
                            .map(|edge_id| {
                                let edge = chunk.edge(&edge_id)?;
                                Ok((edge_id, edge))
                            })
                            .filter_map_ok(|(edge_id, edge)| {
                                (edge.target() == node_id).then_some((edge_id, edge))
                            })
                            .try_fold(
                                (Vec::new(), Vec::new()),
                                |(mut inputs, mut input_claims), res: anyhow::Result<_>| {
                                    let (edge_id, edge) = res?;
                                    let input_id = chunk
                                        .subgraph
                                        .source_node(&edge_id)?
                                        .as_input()
                                        .ok_or(anyhow!(
                                            "Edge {edge_id} is not a model input edge in chunk {chunk_id}"
                                        ))?;
                                    edge.ports().iter().for_each(|port| {
                                        inputs.push(
                                            &self.io.input[*input_id],
                                        );
                                        input_claims
                                            .push(&claims[&NodeInput::new(node_id, port.target_port)])
                                    });
                                    anyhow::Ok((inputs, input_claims))
                                }
                            )?;
                        if !inputs.is_empty() {
                            <LayerCtx<E> as VerifiableCtx<E, PCS>>::verify_input_claim(
                                layer,
                                &inputs,
                                &input_claims,
                            )?;
                        }
                    }
                    Node::Input(_) => {}
                    Node::Output(o) => {
                        claims.insert(NodeInput::new(node_id, 0), output_claims_by_port[o].clone());
                    }
                };
                Ok(claims)
            },
        )?;

        // Now we need add the claims about the input and output of the chunk
        chunk.outgoing_edges.keys().try_for_each(|group_id| {
            let GroupIOClaims {
                commitment_id,
                claims: group_claims,
            } = chunk.compute_outgoing_group_claims(group_id, &chunk_output_claims)?;
            let commitment = chunk_commitments.outputs.get(group_id).ok_or(anyhow!(
                "No commitment found for output group {group_id} in chunk {chunk_id}"
            ))?;
            self.commit_verifier.add_witness_claim(
                commitment_id,
                commitment.clone(),
                group_claims
                    .into_iter()
                    .map(|c| (c.point, vec![c.eval]))
                    .collect(),
            );
            anyhow::Ok(())
        })?;

        chunk.incoming_edges.keys().try_for_each(|group_id| {
            let GroupIOClaims {
                commitment_id,
                claims: group_claims,
            } = chunk.compute_incoming_group_claims(&claims, group_id)?;
            let commitment = chunk_commitments.inputs.get(group_id).ok_or(anyhow!(
                "No commitment found for input group {group_id} in chunk {chunk_id}"
            ))?;
            self.commit_verifier.add_witness_claim(
                commitment_id,
                commitment.clone(),
                group_claims
                    .into_iter()
                    .map(|c| (c.point, vec![c.eval]))
                    .collect(),
            );
            anyhow::Ok(())
        })?;

        // ===== Verify the lookup table proofs =====
        if let Some(proof) = &proof.table_proof {
            verify_table::<_, _, _>(
                proof,
                &lookup_ctx,
                chunk_id,
                ctx.commitment_ctx.table_node_id(),
                &mut self.commit_verifier,
                self.transcript,
                &self.challenge_storage,
            )?;
        }

        // ===== Verify the opening of the accumulation of claims =====
        self.commit_verifier
            .verify(&ctx.commitment_ctx, proof.commit, self.transcript)?;

        // ===== Verify that the accumulated numerator is zero and accumulated denominator is non-zero =====
        let num_len = numerators.len();
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

        info!("Proof verified successfully");
        Ok(())
    }

    pub(crate) fn verify(
        ctx: &VerifierContext<E, PCS>,
        io: &IO<E>,
        proof: Proof<E, PCS>,
    ) -> anyhow::Result<()>
    where
        PCS::Commitment: PartialEq + Eq,
        T: InitTranscript,
    {
        let mut transcript = Self::initialise_transcript(ctx)?;
        let verifier = Verifier::<'_, E, T, PCS>::new(&mut transcript, io);

        let shape_steps = ctx.model.shape_steps(
            &ctx.unpadded_input_shapes,
            &verifier
                .io
                .input
                .iter()
                .map(|t| t.shape().clone())
                .collect_vec(),
        )?;

        // verify chunks are well defined
        ctx.model.check_model_chunking(
            proof
                .chunk_proofs
                .iter()
                .map(|chunk_proof| &chunk_proof.chunk_data.model_chunk),
        )?;

        // verify consistency of commitments among chunks
        let chunk_commitments_by_id = proof
            .chunk_proofs
            .iter()
            .map(|chunk_proof| {
                let chunk_data = &chunk_proof.chunk_data;
                (chunk_data.model_chunk.chunk_id, &chunk_data.commitments)
            })
            .collect();
        proof.chunk_proofs.iter().try_for_each(|chunk_proof| {
            chunk_proof
                .chunk_data
                .model_chunk
                .check_chunk_commitment_consistency::<E, PCS>(&chunk_commitments_by_id)
        })?;

        // verify chunks
        // there is a distinct proof for model claims, so we need to verify each chunk
        // and then verify the model opening proof
        // first, squeeze the common challenge to initialize the transcript for each cbunk
        let challenge = verifier.transcript.read_challenge();
        proof.chunk_proofs.into_iter().try_for_each(|proof| {
            // initialise a verifier for the given chunk
            let mut transcript: T = initialize_transcript_for_chunk(challenge.elements);
            let verifier = Verifier::new(&mut transcript, verifier.io);
            verifier.verify_chunk(ctx, proof, &shape_steps)
        })?;

        Ok(())
    }

    fn verify_merge_claims_proof(
        &mut self,
        claims: BTreeMap<PortId, Vec<&Claim<E>>>,
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
                    merge_claim_proof.verify_proof(self.transcript, &claims)
                }
            })
            .collect()
    }

    pub(crate) fn add_common_claims(
        &mut self,
        node_id: NodeId,
        claims: HashMap<CommitmentId, Claim<E>>,
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
pub fn verify<E, T: Transcript<E> + InitTranscript, PCS: PolynomialCommitmentScheme<E>>(
    ctx: &VerifierContext<E, PCS>,
    proof: Proof<E, PCS>,
    io: IO<E>,
) -> anyhow::Result<()>
where
    E: ExtensionField,
    PCS::Commitment: PartialEq + Eq,
{
    let span = info_span!(
        "zkml_verify_proof",
        inputs = io.input.len(),
        outputs = io.output.len()
    );
    let _guard = span.enter();
    Verifier::<E, T, PCS>::verify(ctx, &io, proof)
}

fn verify_table<E, T: Transcript<E>, PCS: PolynomialCommitmentScheme<E>>(
    proof: &TableProof<E, PCS>,
    lookup_ctx: &LookupContext,
    chunk_id: ChunkID,
    table_node_id: NodeId,
    witness_verifier: &mut CommitmentVerifier<E, PCS>,
    t: &mut T,
    challenge_storage: &ChallengeStorage<E>,
) -> anyhow::Result<()>
where
    E: ExtensionField,
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
                    chunk_id.0.into(),
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
