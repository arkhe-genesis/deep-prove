//! Module containing code to verify a [`ModelOpeningProof`].
use super::{CommitmentVerifierCtx, OpeningProof};
use crate::{
    Claim,
    commit::{
        identity_eval,
        mmcs_context::{ModelClaims, build_sumcheck_expression, table_poly_id},
    },
    graph::NodeId,
    lookup::context::TableType,
    tensor::CommitmentId,
};
use anyhow::{Result, anyhow, ensure};
use ff_ext::ExtensionField;
use mpcs::PolynomialCommitmentScheme;
use multilinear_extensions::{
    mle::Point, utils::eval_by_expr_with_instance, virtual_poly::VPAuxInfo,
};
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use std::collections::{BTreeMap, HashMap};
use sumcheck::structs::{IOPProof, IOPVerifierState};
use transcript::Transcript;

#[derive(Clone, Serialize, Deserialize)]
#[serde(bound(serialize = "E: Serialize", deserialize = "E: DeserializeOwned"))]
/// A claim about batch committed witness polynomials to be verified.
pub struct VerifierClaim<E, PCS>
where
    E: ExtensionField,
    PCS: PolynomialCommitmentScheme<E>,
{
    /// The commitment to the witness polynomials
    pub commitment: PCS::Commitment,
    /// The claims about the witness polynomials
    pub witness_claims: Vec<(Point<E>, Vec<E>)>,
}

impl<E, PCS> VerifierClaim<E, PCS>
where
    E: ExtensionField,
    PCS: PolynomialCommitmentScheme<E>,
{
    /// Create a new [`VerifierClaim`] from its constituent parts.
    pub fn new(commitment: PCS::Commitment, witness_claims: Vec<(Point<E>, Vec<E>)>) -> Self {
        VerifierClaim {
            commitment,
            witness_claims,
        }
    }
}

impl<E, PCS> From<VerifierClaim<E, PCS>> for (PCS::Commitment, Vec<(usize, (Point<E>, Vec<E>))>)
where
    E: ExtensionField,
    PCS: PolynomialCommitmentScheme<E>,
{
    fn from(claim: VerifierClaim<E, PCS>) -> Self {
        let VerifierClaim {
            commitment,
            witness_claims,
        } = claim;
        let rounds = witness_claims
            .into_iter()
            .map(|(point, evals)| (point.len(), (point, evals)))
            .collect();
        (commitment, rounds)
    }
}

pub struct CommitmentVerifier<E: ExtensionField, PCS: PolynomialCommitmentScheme<E>> {
    /// A map storing all the claims for tensors fixed by the model.
    /// The `NodeId` is only employed to sort the claims related to the same
    /// static tensor, assuming that only one claim for a static tensor is
    /// produced in each node
    pub(crate) model_claims: ModelClaims<E>,
    /// The list of claims about the witness
    witness_claims: BTreeMap<NodeId, VerifierClaim<E, PCS>>,
}

impl<E: ExtensionField, PCS: PolynomialCommitmentScheme<E>> CommitmentVerifier<E, PCS> {
    pub fn new() -> CommitmentVerifier<E, PCS> {
        CommitmentVerifier {
            model_claims: ModelClaims::new(),
            witness_claims: BTreeMap::new(),
        }
    }

    pub fn add_witness_claim(
        &mut self,
        node_id: NodeId,
        commitment: PCS::Commitment,
        claim: Vec<(Point<E>, Vec<E>)>,
    ) {
        self.witness_claims
            .insert(node_id, VerifierClaim::new(commitment, claim));
    }

    pub fn add_common_claims(&mut self, claims: HashMap<CommitmentId, HashMap<NodeId, Claim<E>>>) {
        claims.into_iter().for_each(|(poly_id, claims)| {
            let poly_claims = self.model_claims.entry(poly_id).or_default();
            claims
                .into_iter()
                .for_each(|(node_id, claim)| assert!(poly_claims.insert(node_id, claim).is_none()))
        });
    }

    pub fn add_table_claim(
        &mut self,
        table_node_id: NodeId,
        table_type: &TableType,
        claim: Claim<E>,
    ) {
        assert!(
            self.model_claims
                .entry(table_poly_id(table_type.name()))
                .or_default()
                .insert(table_node_id, claim)
                .is_none()
        );
    }

    pub fn verify<T: Transcript<E>>(
        &mut self,
        ctx: &CommitmentVerifierCtx<E, PCS>,
        proof: OpeningProof<E, PCS>,
        transcript: &mut T,
    ) -> Result<()> {
        let OpeningProof {
            sumcheck_proof,
            sumcheck_evals,
            pcs_proof,
        } = proof;
        // Order the witness claims
        let mut rounds = Vec::new();
        let witness_claims = std::mem::take(&mut self.witness_claims);
        witness_claims
            .into_iter()
            .for_each(|(_, claim)| rounds.push(claim.into()));

        // First we verify the Sumcheck proof that allows us to open all of the model polynomials at the same point
        let mut model_claims = std::mem::take(&mut self.model_claims);
        if let Some(model_commitment) = &ctx.model_commitment && !model_claims.is_empty() {
            let model_claim = Self::verify_model_sumcheck(
                &mut model_claims,
                sumcheck_proof,
                sumcheck_evals,
                ctx,
                transcript,
            )?;

            let verifier_claim =
                VerifierClaim::<E, PCS>::new(model_commitment.clone(), model_claim);
            rounds.push(verifier_claim.into());
        } else {
            ensure!(
                sumcheck_proof.proofs.is_empty() && sumcheck_evals.is_empty(),
                "There was no Model Commitment but the Model Sumcheck proof was not trivial"
            );
        }
        // Run the PCS batch_verify protocol
        PCS::batch_verify(&ctx.verifier_params, rounds, &pcs_proof, transcript)
            .map_err(|e| anyhow!("{e:?}"))
    }

    fn verify_model_sumcheck<T: Transcript<E>>(
        model_claims: &mut HashMap<CommitmentId, BTreeMap<NodeId, Claim<E>>>,
        sumcheck_proof: IOPProof<E>,
        sumcheck_evals: Vec<E>,
        commit_ctx: &CommitmentVerifierCtx<E, PCS>,
        transcript: &mut T,
    ) -> Result<Vec<(Point<E>, Vec<E>)>> {
        // First we order our model claims, splitting into `polys_per_var` which is how many polys there are for each number of variables.
        // Then, we compute `eq_points_vec` and `evals_vec`, which are the points and the evaluations, respectively, that the original claims
        // relate to, ordered in the same way that the model polys are.
        // Finally we compute also the `num_claims_per_poly` vector, whose i-th entry specifies how many claims
        // are found in `model_claims` for the i-th model polynomial employed in the sumcheck.
        let (eq_points_vec, evals_vec, num_claims_per_poly) =
            commit_ctx.model_comms_map.iter().rev().try_fold(
                (vec![], vec![], vec![]),
                |(mut eq_points_acc, mut evals, mut num_claims_per_poly), (&nv, claim_keys)| {
                    // If the claim is about a polynomial with fewer variables than the maximum number of variables then
                    // we have to multiply the evalaution by 2^variables_diff.
                    let mult = E::from_canonical_u64(1 << (commit_ctx.max_model_num_vars - nv));
                    let (eq_points, inner_evals, num_claims) = claim_keys.iter().try_fold(
                        (vec![], vec![], vec![]),
                        |(mut point_acc, mut inner_evals_acc, mut num_claims_per_poly), key| {
                            let claims = if let Some(claims) = model_claims
                                .remove(key) {
                                    claims
                                } else {
                                    [(
                                        0.into(), // we use a dummy node id since it will not be used anyhow 
                                        commit_ctx.precomputed_model_claims.get(key)
                                            .ok_or(anyhow!("No precomputed claim found for model poly {key}"))?
                                            .clone()
                                    )].into_iter().collect()
                                };
                            num_claims_per_poly.push(claims.len());
                            claims.into_iter().for_each(|(_, claim)| {
                                let Claim { point, eval } = claim;
                                // Append all of the evaluations in the correct order to the transcript
                                transcript.append_field_element_ext(&eval);
                                point_acc.push(point);
                                inner_evals_acc.push(eval * mult);
                            });

                            Result::<(Vec<Point<E>>, Vec<E>, Vec<usize>), anyhow::Error>::Ok((
                                point_acc,
                                inner_evals_acc,
                                num_claims_per_poly,
                            ))
                        },
                    )?;
                    eq_points_acc.extend(eq_points);
                    evals.extend(inner_evals);
                    num_claims_per_poly.extend(num_claims);
                    Result::<(Vec<Point<E>>, Vec<E>, Vec<usize>), anyhow::Error>::Ok((
                        eq_points_acc,
                        evals,
                        num_claims_per_poly,
                    ))
                },
            )?;

        let sumcheck_expression = build_sumcheck_expression(num_claims_per_poly);

        let challenge = transcript
            .sample_and_append_challenge(b"model_batching")
            .elements;
        // Calculate the initial sum
        let (claimed_sum, _) = evals_vec
            .into_iter()
            .fold((E::ZERO, E::ONE), |(sum_acc, chal_acc), e| {
                (sum_acc + chal_acc * e, chal_acc * challenge)
            });

        let aux_info = VPAuxInfo {
            max_degree: 2,
            max_num_variables: commit_ctx.max_model_num_vars,
            ..Default::default()
        };
        // Run Sumcheck verification
        let subclaim =
            IOPVerifierState::<E>::verify(claimed_sum, &sumcheck_proof, &aux_info, transcript);
        let sumcheck_point = subclaim
            .point
            .iter()
            .map(|c| c.elements)
            .collect::<Vec<E>>();
        let point_len = sumcheck_point.len();
        let witness_evals = sumcheck_evals
            .iter()
            .copied()
            .chain(eq_points_vec.iter().map(|eq_point| {
                identity_eval(&sumcheck_point[point_len - eq_point.len()..], eq_point)
            }))
            .collect::<Vec<E>>();
        // Check that the supplied evaluations correspond to the subclaim we get from sumcheck verification
        let calc_eval = eval_by_expr_with_instance(
            &[],
            &witness_evals,
            &[],
            &[],
            &[challenge],
            &sumcheck_expression,
        )
        .right()
        .ok_or(anyhow!(
            "PCS verify sumcheck calculated eval was not an extension field element"
        ))?;

        ensure!(
            calc_eval == subclaim.expected_evaluation,
            "PCS verification failed, model sumcheck calculated evaluation {:?} did not equal subclaim expected evaluation: {:?}",
            calc_eval,
            subclaim.expected_evaluation
        );

        // Now we build the VerifierClaim for the model
        let (model_claim, _) = commit_ctx.model_comms_map.iter().rev().fold(
            (vec![], 0),
            |(mut claim_acc, skip), (&nv, claim_keys)| {
                let count = claim_keys.len();
                let evals = sumcheck_evals[skip..skip + count].to_vec();
                let eval_point = sumcheck_point[point_len - nv..].to_vec();
                claim_acc.push((eval_point, evals));
                (claim_acc, skip + count)
            },
        );

        Ok(model_claim)
    }
}
