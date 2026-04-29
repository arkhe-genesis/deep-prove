use std::collections::{BTreeMap, HashMap};

use anyhow::ensure;
use ark_ff::PrimeField;
use dp_crypto::{
    arkyper::{
        CommitmentScheme,
        transcript::{AppendToTranscript, Transcript},
    },
    poly::dense::Point,
    structs::{IOPProof, IOPVerifierState},
    utils::eval_by_expr_with_instance,
    virtual_poly::VPAuxInfo,
};
use itertools::Itertools;
use serde::{Deserialize, Serialize};

use crate::{
    Claim,
    graph::NodeId,
    lookup::table::Table,
    poly_commit::{
        ChunkedCommitment, OpeningProof, build_sumcheck_expression,
        context::{CommitmentVerifierCtx, CommittedPolynomial},
        identity_eval,
        prover::ModelClaims,
        table_poly_id,
    },
    tensor::CommitmentId,
};

/// Data structure employed to represent a commitment for the verifier
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VerifierCommitment<PCS: CommitmentScheme> {
    chunk_commitments: Vec<PCS::Commitment>,
    num_vars: usize,
}

impl<PCS: CommitmentScheme> ChunkedCommitment for VerifierCommitment<PCS> {
    fn num_chunks(&self) -> usize {
        self.chunk_commitments.len()
    }

    fn num_vars(&self) -> usize {
        self.num_vars
    }
}

impl<PCS: CommitmentScheme> PartialEq for VerifierCommitment<PCS> {
    fn eq(&self, other: &Self) -> bool {
        self.chunk_commitments == other.chunk_commitments && self.num_vars == other.num_vars
    }
}

impl<'a, F: PrimeField, PCS: CommitmentScheme> From<&CommittedPolynomial<'a, F, PCS>>
    for VerifierCommitment<PCS>
{
    fn from(value: &CommittedPolynomial<'a, F, PCS>) -> Self {
        Self {
            chunk_commitments: value.chunk_commitments.clone(),
            num_vars: value.polynomial.num_vars(),
        }
    }
}

impl<'a, F: PrimeField, PCS: CommitmentScheme> From<CommittedPolynomial<'a, F, PCS>>
    for VerifierCommitment<PCS>
{
    fn from(value: CommittedPolynomial<'a, F, PCS>) -> Self {
        Self {
            chunk_commitments: value.chunk_commitments,
            num_vars: value.polynomial.num_vars(),
        }
    }
}

impl<PCS: CommitmentScheme> AppendToTranscript for VerifierCommitment<PCS> {
    fn append_to_transcript<ProofTranscript: Transcript>(&self, transcript: &mut ProofTranscript) {
        self.chunk_commitments
            .iter()
            .for_each(|comm| comm.append_to_transcript(transcript))
    }
}

#[derive(Debug)]
pub struct VerifierClaim<F: PrimeField, PCS: CommitmentScheme> {
    pub(crate) commitment: VerifierCommitment<PCS>,
    /// There could be one or more claims for each committed polynomial
    pub(crate) claims: Vec<Claim<F>>,
}

impl<F: PrimeField, PCS: CommitmentScheme> From<(VerifierCommitment<PCS>, Claim<F>)>
    for VerifierClaim<F, PCS>
{
    fn from(value: (VerifierCommitment<PCS>, Claim<F>)) -> Self {
        Self {
            commitment: value.0,
            claims: vec![value.1],
        }
    }
}

/// Data structure employed for the output of the [`CommitmentVerifier::claims_batching_sumcheck`] method
struct ClaimsBatchingVerifier<F, PCS: CommitmentScheme> {
    /// Opening point for all the claims obtained from the sumcheck
    point: Point<F>,
    /// Number of variables of each polynomial being involved in the sumcheck
    commitments: Vec<PCS::Commitment>,
}

pub struct CommitmentVerifier<F: PrimeField, PCS: CommitmentScheme> {
    /// A map storing all the claims for tensors fixed by the model.
    /// The `NodeId` is only employed to sort the claims related to the same
    /// static tensor, assuming that only one claim for a static tensor is
    /// produced in each node
    pub(crate) model_claims: ModelClaims<F>,
    /// The list of claims about the witness
    pub(crate) witness_claims: BTreeMap<NodeId, Vec<VerifierClaim<F, PCS>>>,
}

impl<F: PrimeField, PCS: CommitmentScheme> Default for CommitmentVerifier<F, PCS> {
    fn default() -> Self {
        Self {
            model_claims: Default::default(),
            witness_claims: Default::default(),
        }
    }
}

impl<F: PrimeField, PCS: CommitmentScheme> CommitmentVerifier<F, PCS> {
    pub fn add_witness_claim(&mut self, node_id: NodeId, claims: Vec<VerifierClaim<F, PCS>>) {
        self.witness_claims.insert(node_id, claims);
    }

    pub fn add_common_claims(&mut self, claims: HashMap<CommitmentId, HashMap<NodeId, Claim<F>>>) {
        claims.into_iter().for_each(|(poly_id, claims)| {
            let poly_claims = self.model_claims.entry(poly_id).or_default();
            claims
                .into_iter()
                .for_each(|(node_id, claim)| assert!(poly_claims.insert(node_id, claim).is_none()))
        });
    }

    pub fn add_table_claim(&mut self, table_node_id: NodeId, table: &Table, claim: Claim<F>) {
        assert!(
            self.model_claims
                .entry(table_poly_id(table.name()))
                .or_default()
                .insert(table_node_id, claim)
                .is_none()
        );
    }

    pub fn verify<T: Transcript>(
        &mut self,
        commit_ctx: &CommitmentVerifierCtx<PCS>,
        proof: OpeningProof<F, PCS>,
        transcript: &mut T,
    ) -> anyhow::Result<()>
    where
        PCS: CommitmentScheme<Field = F>,
    {
        let OpeningProof {
            sumcheck_proof,
            sumcheck_evals,
            pcs_proof,
        } = proof;

        let ClaimsBatchingVerifier { point, commitments } =
            self.claims_batching_sumcheck(sumcheck_proof, &sumcheck_evals, commit_ctx, transcript)?;

        // Sample the challenges to combine the commitments of all the polynomials being opened;
        // Before sampling the challenges, we add all claims to the transcript
        transcript.append_scalars::<F>(&point);
        transcript.append_scalars::<F>(&sumcheck_evals);
        let challenges = (0..commitments.len())
            .map(|_| transcript.challenge_scalar())
            .collect_vec();

        // combine the commitments of individual polynomials
        let commitment = PCS::combine_commitments(&commitments, &challenges)?;

        // combine evaluations of each polynomial over the opening point `point`, using the
        // same `challenges` employed to combine the commitments.
        let combined_eval = sumcheck_evals
            .into_iter()
            .zip(challenges)
            .fold(F::ZERO, |combined_eval, (eval, challenge)| {
                combined_eval + eval * challenge
            });

        // Run the PCS batch_verify protocol
        PCS::verify(
            &commit_ctx.verifier_params,
            &pcs_proof,
            transcript,
            &point,
            &combined_eval,
            &commitment,
        )
    }

    fn claims_batching_sumcheck<T: Transcript>(
        &mut self,
        sumcheck_proof: IOPProof<F>,
        sumcheck_evals: &[F],
        commit_ctx: &CommitmentVerifierCtx<PCS>,
        transcript: &mut T,
    ) -> anyhow::Result<ClaimsBatchingVerifier<F, PCS>> {
        let model_claims = &mut self.model_claims;
        let witness_claims = std::mem::take(&mut self.witness_claims);

        // We compute `eq_points` and `evals` vectors, which are the points and the evaluations, respectively,
        // that the original claims relate to, ordered in the same way as the corresponding polys are.
        // Finally we compute also the `num_claims_per_poly` vector, whose i-th entry specifies how many claims
        // are found for each polynomial employed in the sumcheck.
        let mut eq_points = vec![];
        let mut evals = vec![];
        let mut num_claims_per_poly = vec![];
        let mut commitments = vec![];

        // compute the maximum number of variables across all the polys involved in the batching sumcheck
        let mut max_num_vars = 0;

        let mut batching_coefficients_per_poly = commit_ctx
            .model_commitments
            .iter()
            .filter_map(|(poly_id, comm)| {
                if let Some(claims) = model_claims.remove(poly_id) {
                    num_claims_per_poly.push(claims.len());
                    let batching_coeffs = claims
                        .into_values()
                        .map(|claim| {
                            // Append all of the evaluations in the correct order to the transcript
                            transcript.append_scalars(&[claim.eval]);
                            eq_points.push(comm.eq_point_per_chunk(&claim.point).to_vec());
                            evals.push(claim.eval);
                            comm.batching_coefficients_for_chunks(&claim.point)
                        })
                        .collect_vec();
                    max_num_vars = max_num_vars.max(comm.num_vars_for_chunk());
                    commitments.extend_from_slice(&comm.chunk_commitments);
                    Some(comm.to_chunked_info(batching_coeffs))
                } else {
                    None
                }
            })
            .collect_vec();

        for (_, claims) in witness_claims {
            for verifier_claim in claims {
                num_claims_per_poly.push(verifier_claim.claims.len());
                let batching_coeffs = verifier_claim
                    .claims
                    .into_iter()
                    .map(|claim| {
                        // Append all of the evaluations in the correct order to the transcript
                        transcript.append_scalars(&[claim.eval]);
                        eq_points.push(
                            verifier_claim
                                .commitment
                                .eq_point_per_chunk(&claim.point)
                                .to_vec(),
                        );
                        evals.push(claim.eval);
                        verifier_claim
                            .commitment
                            .batching_coefficients_for_chunks(&claim.point)
                    })
                    .collect_vec();
                max_num_vars = max_num_vars.max(verifier_claim.commitment.num_vars_for_chunk());
                batching_coefficients_per_poly
                    .push(verifier_claim.commitment.to_chunked_info(batching_coeffs));
                commitments.extend(verifier_claim.commitment.chunk_commitments);
            }
        }

        let sumcheck_expression = build_sumcheck_expression(batching_coefficients_per_poly)?;

        let challenge = transcript.append_and_sample(b"model_batching");
        // Calculate the initial sum
        let (claimed_sum, _) = evals
            .into_iter()
            .fold((F::ZERO, F::ONE), |(sum_acc, chal_acc), e| {
                (sum_acc + chal_acc * e, chal_acc * challenge)
            });

        let aux_info = VPAuxInfo {
            max_degree: 2,
            max_num_variables: max_num_vars,
            ..Default::default()
        };
        // Run Sumcheck verification
        let subclaim =
            IOPVerifierState::<F>::verify(claimed_sum, &sumcheck_proof, &aux_info, transcript);
        let sumcheck_point = &subclaim.point;
        let witness_evals = sumcheck_evals
            .iter()
            .copied()
            .chain(eq_points.iter().map(|eq_point| {
                // in the sumcheck we pad each `eq_poly` with 0s up to `max_num_vars`.
                // Therefore, to recompute the evaluation for the padded `eq_poly`,
                // we need to multiply the evaluation of the unpadded `eq_poly` by
                // the product of `(1-r_i)`, for all coordinates `r_i` of the `sumcheck_point`
                // which refers to the extra variables added because of the padding
                let (identity_point, padding_point) = sumcheck_point.split_at(eq_point.len());
                padding_point
                    .iter()
                    .fold(identity_eval(identity_point, eq_point), |acc, p| {
                        acc * (F::ONE - p)
                    })
            }))
            .collect::<Vec<_>>();
        // Check that the supplied evaluations correspond to the subclaim we get from sumcheck verification
        let calc_eval = eval_by_expr_with_instance(
            &[],
            &witness_evals,
            &[],
            &[],
            &[challenge],
            &sumcheck_expression,
        );

        ensure!(
            calc_eval == subclaim.expected_evaluation,
            "PCS verification failed, model sumcheck calculated evaluation {:?} did not equal subclaim expected evaluation: {:?}",
            calc_eval,
            subclaim.expected_evaluation
        );

        Ok(ClaimsBatchingVerifier {
            point: subclaim.point,
            commitments,
        })
    }
}
