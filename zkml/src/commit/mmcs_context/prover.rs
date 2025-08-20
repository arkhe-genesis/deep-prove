//! Module containing code for the model commitment prover

use super::{CommitmentProverCtx, ModelOpeningProof, PolyId};
use crate::{
    Claim, commit::compute_betas_eval, layers::provable::NodeId, lookup::context::TableType,
};

use std::{
    collections::{BTreeMap, HashMap},
    marker::PhantomData,
    slice,
};

use anyhow::{Result, anyhow};

use either::Either;
use ff_ext::ExtensionField;
use mpcs::PolynomialCommitmentScheme;
use multilinear_extensions::{
    mle::{IntoMLE, MultilinearExtension, Point},
    virtual_polys::VirtualPolynomialsBuilder,
};

use serde::{Deserialize, Serialize, de::DeserializeOwned};
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
    model_claims: HashMap<(NodeId, String), Claim<E>>,
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
            model_claims: HashMap::new(),
            witness_claims: BTreeMap::new(),
            _phantom: PhantomData,
        }
    }
    pub fn add_witness_claim(&mut self, node_id: NodeId, claim: Vec<(Point<E>, Vec<E>)>) {
        self.witness_claims
            .insert(node_id, BatchCommitmentClaim::<E>::new(claim));
    }

    pub fn add_common_claims(&mut self, node_id: NodeId, claims: HashMap<PolyId, Claim<E>>) {
        claims.into_iter().for_each(|(poly_id, claim)| {
            self.model_claims.insert((node_id, poly_id), claim);
        });
    }

    pub fn add_table_claim(
        &mut self,
        table_node_id: NodeId,
        table_type: &TableType,
        claim: Claim<E>,
    ) {
        self.model_claims
            .insert((table_node_id, table_type.name()), claim);
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
    ) -> Result<ModelOpeningProof<E, PCS>> {
        // First we replace the `NodeId`s in the witness claims with the actual PCS::CommitmentWithWitness
        let mut rounds = self.prep_for_open(witness_commitments)?;

        // Now we arrange the model claims in the correct order, we iterate over the model_comms_map in reverse so the largest number of variables is
        // the first key value pair we visit.
        let mut model_claims = std::mem::take(&mut self.model_claims);
        let (sumcheck_proof, sumcheck_evals) =
            if let Some(model_commitment) = ctx.model_commitment.as_ref() {
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
        .map_err(|e| anyhow!("{:?}", e))?;

        Ok(ModelOpeningProof {
            sumcheck_proof,
            sumcheck_evals,
            pcs_proof,
        })
    }

    fn model_polys_sumcheck<T: Transcript<E>>(
        model_claims: &mut HashMap<(NodeId, PolyId), Claim<E>>,
        commit_ctx: &CommitmentProverCtx<E, PCS>,
        transcript: &mut T,
    ) -> Result<ModelSumcheckProof<E>> {
        // Here we iterate over the model_comms_map so that we can construct the EQ polys and claimed evalautions for each of the committed model polys.
        // In addition we also return `polys_per_var` which is a vector which stores how many polys there are for each number of variables.
        let eq_polys_vec = commit_ctx.model_comms_map.values().rev().try_fold(
            Vec::new(),
            |mut acc, claim_keys| {
                let eqs = claim_keys
                    .iter()
                    .map(|key| {
                        let Claim { point, eval } = model_claims.remove(key).ok_or(anyhow!(
                            "No Claim found for mode poly NodeId {}, PolyId {}",
                            key.0,
                            key.1
                        ))?;
                        let eq_poly = compute_betas_eval(&point).into_mle();
                        // Append the evaluations to the transcript
                        transcript.append_field_element_ext(&eval);
                        Ok(eq_poly)
                    })
                    .collect::<Result<Vec<MultilinearExtension<E>>, anyhow::Error>>()?;
                acc.extend(eqs);
                Result::<Vec<MultilinearExtension<E>>, anyhow::Error>::Ok(acc)
            },
        )?;

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
        let virtual_poly = expr_builder.to_virtual_polys(
            slice::from_ref(&commit_ctx.sumcheck_expression),
            &[challenge],
        );
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
        // all_evals is exactly twice as long as just the model poly evals because we have one eq poly eval for each model poly
        let sumcheck_evals = all_evals[..all_evals.len() / 2].to_vec();

        Ok(ModelSumcheckProof {
            model_claim,
            sumcheck_proof,
            sumcheck_evals,
        })
    }
}
