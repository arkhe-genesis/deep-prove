//! Module containing code for the model commitment prover
use super::{CommitmentProverCtx, OpeningProof};
use crate::{
    Claim,
    commit::{
        compute_betas_eval,
        mmcs_context::{ModelClaims, build_sumcheck_expression, table_poly_id},
    },
    graph::NodeId,
    lookup::context::TableType,
    tensor::CommitmentId,
};
use anyhow::{Result, anyhow};
use either::Either;
use ff_ext::ExtensionField;
use itertools::Itertools;
use mpcs::PolynomialCommitmentScheme;
use multilinear_extensions::{
    mle::{IntoMLE, MultilinearExtension, Point},
    virtual_polys::VirtualPolynomialsBuilder,
};
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use std::{
    collections::{BTreeMap, HashMap},
    marker::PhantomData,
    slice,
};
use sumcheck::{
    structs::{IOPProof, IOPProverState},
    util::optimal_sumcheck_threads,
};
use transcript::Transcript;

#[derive(Debug, Clone, Serialize, Deserialize)]
/// A list of points and evaluations for polynomials stored in a batch commitment, together with the [`NodeId`] the evaluations were generated in. We have one [`Point`]
/// per [`ceno_witness::RowMajorMatrix`] in the commitment, and the length of each [`Vec<E>`] should be the same as the number of columns in said [`ceno_witness::RowMajorMatrix`].
pub struct BatchCommitmentClaim<E> {
    /// The claim on the commitment for that layer, a list of [`Point`] together with a [`Vec<E>`] that stores the evaluations
    /// of the polynomials found in one [`ceno_witness::RowMajorMatrix`] at [`Point`].
    claims: Vec<(Point<E>, Vec<E>)>,
}

struct ProverClaim<'a, E, PCS>
where
    E: ExtensionField,
    PCS: PolynomialCommitmentScheme<E>,
{
    commitment: &'a PCS::CommitmentWithWitness,
    claims: Vec<(Point<E>, Vec<E>)>,
}

impl<'a, E, PCS> ProverClaim<'a, E, PCS>
where
    E: ExtensionField,
    PCS: PolynomialCommitmentScheme<E>,
{
    /// Create a new [`ProverClaim`] from its constituent parts.
    pub fn new(
        commitment: &'a PCS::CommitmentWithWitness,
        claims: Vec<(Point<E>, Vec<E>)>,
    ) -> Self {
        ProverClaim { commitment, claims }
    }
}

impl<E> BatchCommitmentClaim<E> {
    /// Create a new [`BatchCommitmentClaim`] from its constituent parts.
    pub fn new(claims: Vec<(Point<E>, Vec<E>)>) -> BatchCommitmentClaim<E> {
        BatchCommitmentClaim { claims }
    }
}

/// Struct used in this file for the return type of the [`CommitmentProver::model_polys_sumcheck`] method.
struct ModelSumcheckProof<E: ExtensionField> {
    model_claim: Vec<(Point<E>, Vec<E>)>,
    sumcheck_proof: IOPProof<E>,
    sumcheck_evals: Vec<E>,
}

pub struct CommitmentProver<E: ExtensionField, PCS: PolynomialCommitmentScheme<E>>
where
    PCS::CommitmentWithWitness: Serialize + DeserializeOwned,
{
    /// A map storing all the claims for tensors fixed by the model.
    /// The `NodeId` is only employed to sort the claims related to the same
    /// static tensor, assuming that only one claim for a static tensor is
    /// produced in each node
    pub(crate) model_claims: ModelClaims<E>,
    /// The list of claims about the witness
    witness_claims: BTreeMap<NodeId, BatchCommitmentClaim<E>>,
    _phantom: PhantomData<PCS>,
}

impl<E: ExtensionField, PCS: PolynomialCommitmentScheme<E>> Default for CommitmentProver<E, PCS>
where
    PCS::CommitmentWithWitness: Serialize + DeserializeOwned,
{
    fn default() -> Self {
        CommitmentProver::new()
    }
}

impl<E: ExtensionField, PCS: PolynomialCommitmentScheme<E>> CommitmentProver<E, PCS>
where
    PCS::CommitmentWithWitness: Serialize + DeserializeOwned,
{
    pub fn new() -> CommitmentProver<E, PCS> {
        CommitmentProver {
            model_claims: ModelClaims::new(),
            witness_claims: BTreeMap::new(),
            _phantom: PhantomData,
        }
    }

    pub fn add_witness_claim(&mut self, node_id: NodeId, claim: Vec<(Point<E>, Vec<E>)>) {
        self.witness_claims
            .insert(node_id, BatchCommitmentClaim::<E>::new(claim));
    }

    pub fn add_common_claims(&mut self, claims: HashMap<CommitmentId, Vec<(NodeId, Claim<E>)>>) {
        claims.into_iter().for_each(|(poly_id, claims)| {
            let poly_claims = self.model_claims.entry(poly_id.clone()).or_default();
            claims.into_iter().for_each(|(node_id, claim)| {
                assert!(
                    poly_claims.insert(node_id, claim).is_none(),
                    "Failed for poly {poly_id} in node {node_id}"
                )
            });
        });
    }

    pub fn add_table_claim(&mut self, table_id: NodeId, table_type: &TableType, claim: Claim<E>) {
        assert!(
            self.model_claims
                .entry(table_poly_id(table_type.name()))
                .or_default()
                .insert(table_id, claim)
                .is_none()
        );
    }

    /// Using a provided mapping from [`NodeId`] to [`PolynomialCommitmentScheme::CommitmentWithWitness`] construct
    /// the data to be fed to the [`PolynomialCommitmentScheme::batch_open`] method.
    fn prep_for_open<'a>(
        &mut self,
        node_id_mapping: &'a HashMap<NodeId, PCS::CommitmentWithWitness>,
    ) -> Result<Vec<ProverClaim<'a, E, PCS>>> {
        let witness_map = std::mem::take(&mut self.witness_claims);
        witness_map
            .into_iter()
            .map(|(node_id, batch_claim)| {
                let commitment = node_id_mapping.get(&node_id).ok_or(anyhow!(
                    "Proving failed, No commitment found for NodeId {node_id}"
                ))?;
                let BatchCommitmentClaim { claims } = batch_claim;
                Ok(ProverClaim::new(commitment, claims))
            })
            .collect()
    }

    pub fn prove<T: Transcript<E>>(
        mut self,
        ctx: &CommitmentProverCtx<E, PCS>,
        witness_commitments: &HashMap<NodeId, PCS::CommitmentWithWitness>,
        transcript: &mut T,
    ) -> Result<OpeningProof<E, PCS>> {
        // First we replace the `NodeId`s in the witness claims with the actual PCS::CommitmentWithWitness
        let mut rounds = self.prep_for_open(witness_commitments)?;

        // Now we arrange the model claims in the correct order, we iterate over the model_comms_map in reverse so the largest number of variables is
        // the first key value pair we visit.
        let mut model_claims = std::mem::take(&mut self.model_claims);
        let (sumcheck_proof, sumcheck_evals) = if let Some(ref model_commitment) =
            ctx.model_commitment
            && !model_claims.is_empty()
        {
            let ModelSumcheckProof {
                model_claim,
                sumcheck_proof,
                sumcheck_evals,
            } = Self::model_polys_sumcheck::<T>(&mut model_claims, ctx, transcript)?;
            rounds.push(ProverClaim::new(model_commitment, model_claim));
            (sumcheck_proof, sumcheck_evals)
        } else {
            (IOPProof::<E>::default(), vec![])
        };

        // Make the PCS batch proof
        let pcs_proof = PCS::batch_open(
            &ctx.prover_params,
            rounds
                .into_iter()
                .map(|claim| {
                    let ProverClaim { commitment, claims } = claim;
                    (commitment, claims)
                })
                .collect(),
            transcript,
        )
        .map_err(|e| anyhow!("{e:?}"))?;

        Ok(OpeningProof {
            sumcheck_proof,
            sumcheck_evals,
            pcs_proof,
        })
    }

    fn model_polys_sumcheck<T: Transcript<E>>(
        model_claims: &mut HashMap<CommitmentId, BTreeMap<NodeId, Claim<E>>>,
        commit_ctx: &CommitmentProverCtx<E, PCS>,
        transcript: &mut T,
    ) -> Result<ModelSumcheckProof<E>> {
        // Here we iterate over the model_comms_map so that we can construct the EQ polys and claimed evalautions for each of the committed model polys.
        // In addition we also compute the `num_claims_per_poly` vector, whose i-th entry specifies how many claims
        // are found in `model_claims` for the i-th model polynomial employed in the sumcheck.
        let (eq_polys_vec, num_claims_per_poly) =
            commit_ctx.model_comms_map.values().rev().try_fold(
                (Vec::new(), Vec::new()),
                |(mut eq_polys, mut num_claims_per_poly), claim_keys| {
                    let (eqs, num_claims) = claim_keys.iter().try_fold(
                        (vec![], vec![]),
                        |(mut eq_polys, mut num_claims_per_poly), key| {
                            let eqs = if let Some(claims) = model_claims.remove(key) {
                                claims
                                    .into_values()
                                    .map(|claim| {
                                        let Claim { point, eval } = claim;
                                        // Append the evaluations to the transcript
                                        transcript.append_field_element_ext(&eval);
                                        compute_betas_eval(&point).into_mle()
                                    })
                                    .collect_vec()
                            } else {
                                let precomputed_claim =
                                    commit_ctx.precomputed_model_claims.get(key).ok_or(anyhow!(
                                        "No precomputed claim found for model poly {key}"
                                    ))?;
                                // append the evaluations to the transcript
                                transcript.append_field_element_ext(&precomputed_claim.claim.eval);
                                vec![precomputed_claim.beta_evals.clone().into_mle()]
                            };
                            let num_claims = eqs.len();
                            eq_polys.extend(eqs);
                            num_claims_per_poly.push(num_claims);
                            anyhow::Result::<(Vec<MultilinearExtension<E>>, Vec<usize>)>::Ok((
                                eq_polys,
                                num_claims_per_poly,
                            ))
                        },
                    )?;
                    eq_polys.extend(eqs);
                    num_claims_per_poly.extend(num_claims);
                    Result::<(Vec<MultilinearExtension<E>>, Vec<usize>), anyhow::Error>::Ok((
                        eq_polys,
                        num_claims_per_poly,
                    ))
                },
            )?;

        let total_polys = num_claims_per_poly.len();
        let sumcheck_expression = build_sumcheck_expression(num_claims_per_poly);

        // The unwrap here is safe as this function should only be called after checking that `model_commitment` is Some.
        let model_polys =
            PCS::get_arc_mle_witness_from_commitment(commit_ctx.model_commitment.as_ref().unwrap());
        let either_polys = model_polys
            .iter()
            .map(|p| Either::Left(p.as_ref()))
            .chain(eq_polys_vec.iter().map(Either::Left))
            .collect::<Vec<Either<_, _>>>();

        let challenge = transcript
            .sample_and_append_challenge(b"model_batching")
            .elements;
        // Make the VirtualPolynomials and run Sumcheck proving
        let num_threads = optimal_sumcheck_threads(commit_ctx.max_model_num_vars);
        let expr_builder = VirtualPolynomialsBuilder::<E>::new_with_mles(
            num_threads,
            commit_ctx.max_model_num_vars,
            either_polys,
        );
        let virtual_poly =
            expr_builder.to_virtual_polys(slice::from_ref(&sumcheck_expression), &[challenge]);
        let (sumcheck_proof, state) = IOPProverState::<E>::prove(virtual_poly, transcript);
        let all_evals = state.get_mle_flatten_final_evaluations();
        let point = state.collect_raw_challenges();
        let point_len = point.len();

        // Now we construct the point eval pairs for the model batch commitment
        let (model_claim, _) = commit_ctx.model_comms_map.iter().rev().fold(
            (vec![], 0),
            |(mut claims_acc, skip), (nv, claim_keys)| {
                let poly_count = claim_keys.len();
                let evals = all_evals[skip..skip + poly_count].to_vec();
                let eval_point = point[point_len - nv..].to_vec();
                claims_acc.push((eval_point, evals));
                (claims_acc, skip + poly_count)
            },
        );
        // all_evals contains the model poly evals and the eq poly evals; we want to include in the proof
        // only the model poly evals, which are the first `total_polys` evaluations, according to how the
        // sumcheck expression is built
        let sumcheck_evals = all_evals[..total_polys].to_vec();

        Ok(ModelSumcheckProof {
            model_claim,
            sumcheck_proof,
            sumcheck_evals,
        })
    }
}
