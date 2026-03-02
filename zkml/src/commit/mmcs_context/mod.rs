//! Module containing the logic to commit to instance and witness polynomials for a model using a MMCS
use crate::{
    Claim, VectorTranscript, commit::compute_betas_eval, graph::NodeId, lookup::table::Table,
    tensor::CommitmentId,
};
use anyhow::{Context, Result, anyhow};
use either::Either;
use ff_ext::ExtensionField;
use mpcs::PolynomialCommitmentScheme;
use multilinear_extensions::{
    Expression,
    mle::{IntoMLE, MultilinearExtension},
    util::transpose,
};
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use std::{
    collections::{BTreeMap, HashMap},
    fmt::Debug,
};
use sumcheck::structs::IOPProof;
use tracing::debug;
use transcript::{BasicTranscript, Transcript};
use utils::Metrics;
use witness::{InstancePaddingStrategy, RowMajorMatrix};

mod prover;
pub use prover::CommitmentProver;

mod verifier;
pub use verifier::CommitmentVerifier;

#[derive(Serialize, Deserialize)]
#[serde(bound(serialize = "E: Serialize", deserialize = "E: DeserializeOwned"))]
/// Struct that contains all the data needed for proving/verifying commitments relating to a model.
pub struct GlobalCommitmentContext<E, PCS>
where
    E: ExtensionField,
    PCS: PolynomialCommitmentScheme<E>,
    PCS::CommitmentWithWitness: Serialize + DeserializeOwned,
{
    /// Prover parameters for the [`PolynomialCommitmentScheme`]
    prover_params: PCS::ProverParam,
    /// Verifier parameters for the [`PolynomialCommitmentScheme`]
    verifier_params: PCS::VerifierParam,
    /// The batch commitment for the model, currently this is an [`Option`] because for some small tests
    /// the model has no weights/table commitments.
    model_commitment: Option<PCS::CommitmentWithWitness>,
    /// Map that stores the position of each individual polynomial in the batch commitment
    model_comms_map: BTreeMap<usize, Vec<CommitmentId>>,
    /// Set of precomputed claims for each model polynomial being committed.
    /// These claims are employed when there is no claim to be opened for the corresponding model polynomial
    precomputed_model_claims: HashMap<CommitmentId, Claim<E>>,
    /// This is the [`NodeId`] used for tables in this model
    table_node_id: NodeId,
    /// This is the largest number of variables of any of the polynomials in `model_commitment`
    max_model_num_vars: usize,
}

impl<E, PCS> Debug for GlobalCommitmentContext<E, PCS>
where
    E: ExtensionField,
    PCS: PolynomialCommitmentScheme<E>,
    PCS::CommitmentWithWitness: Serialize + DeserializeOwned,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GlobalCommitmentContext")
            .field("prover_params", &self.prover_params)
            .field("verifier_params", &self.verifier_params)
            .field("model_comms_map", &self.model_comms_map)
            .field("precomputed_model_claims", &self.precomputed_model_claims)
            .field("table_node_id", &self.table_node_id)
            .field("max_model_num_vars", &self.max_model_num_vars)
            .finish()
    }
}

/// Compute the `CommitmentId` employed to identify the constant polynomials
/// associated to the lookup tables
fn table_poly_id(table_name: String) -> CommitmentId {
    format!("table_{table_name}").into()
}

/// Build the sumcheck expression for model static polynomails. It requires as input a vector where
/// the i-th entry specifies the number of claims for the i-th model polynomial employed in the
/// sumcheck
fn build_sumcheck_expression<E: ExtensionField>(num_claims_per_poly: Vec<usize>) -> Expression<E> {
    let total_polys = num_claims_per_poly.len();
    // basically, for each pair of (poly, claim), we need to add a term to the sumcheck of the
    // type `challenge*poly*eq_poly(claim.point)`. Given that there might be more than one
    // input claim for each model polynomial, we need to make sure to use the same `poly`
    // for all the terms referring to the same polynomial; instead, the `eq_poly` will be different
    // in each term (even for terms related to the same model polynomial), since it depends on
    // `claim.point`
    num_claims_per_poly
        .into_iter()
        .enumerate()
        .fold(
            (Expression::Constant(Either::Right(E::ZERO)), 0),
            |(expr, total_num_terms), (i, num_claims)| {
                // total_num_terms keeps track of how many terms we added so far
                (
                    (0..num_claims).fold(expr, |inner_expr, j| {
                        inner_expr
                            + Expression::Challenge(0, total_num_terms + j, E::ONE, E::ZERO)
                                * Expression::WitIn(i as u16)
                                * Expression::WitIn((total_num_terms + j + total_polys) as u16)
                    }),
                    total_num_terms + num_claims,
                )
            },
        )
        .0
}

impl<E, PCS> GlobalCommitmentContext<E, PCS>
where
    PCS: PolynomialCommitmentScheme<E>,
    E: ExtensionField,
    PCS::CommitmentWithWitness: Serialize + DeserializeOwned,
{
    /// Make a new [`GlobalCommitmentContext`]
    pub fn new(
        witness_poly_size: usize,
        polys: HashMap<CommitmentId, MultilinearExtension<E>>,
        lookup_ctx: &[&Table],
        max_node_id: NodeId,
    ) -> Result<GlobalCommitmentContext<E, PCS>> {
        // Find the maximum size so we can generate params
        let max_poly_size = polys
            .iter()
            .fold(witness_poly_size, |acc, (_, poly)| {
                acc.max(1 << poly.num_vars())
            })
            .next_power_of_two();

        let m = Metrics::new();
        let (prover_params, verifier_params) = {
            let param = PCS::setup(max_poly_size, mpcs::SecurityLevel::Conjecture100bits)
                .map_err(|e| anyhow!("{e:?}"))
                .context("setting up params")?;

            PCS::trim(param, max_poly_size)
                .map_err(|e| anyhow!("{e:?}"))
                .context("trimming params")?
        };
        debug!("{} PPs & VPs built", m.to_span());

        // Find the maximum node id used in this model so we can pick a unique node id for table related commitments.
        let table_node_id = NodeId::from(max_node_id.0 + 1);

        // First we take all the model polys and sort them by the number of variables they have.
        // Then we do the same for any table commitments but here we set all of them to have `table_node_id`.
        let table_commitments_check = lookup_ctx.iter().any(|table| table.commit_output_column());
        let (model_commitment, model_comms_map, dummy_model_claims) = if !polys.is_empty()
            || table_commitments_check
        {
            let m = Metrics::new();
            let (map, dummy_model_claims) = polys
                .into_iter()
                .map(|(poly_id, poly)| (poly.num_vars(), (poly_id, poly)))
                .chain(lookup_ctx.iter().filter_map(|table| {
                    if table.commit_output_column() {
                        let mle = table.committed_columns::<E>().into_mle();
                        Some((mle.num_vars(), (table_poly_id(table.name()), mle)))
                    } else {
                        None
                    }
                }))
                .fold(
                    (BTreeMap::new(), HashMap::new()),
                    |(mut map_acc, mut dummy_claims), (num_vars, (poly_id, mle))| {
                        let (ids, polys): &mut (Vec<CommitmentId>, Vec<MultilinearExtension<E>>) =
                            map_acc
                                .entry(num_vars)
                                .or_insert_with(|| (Vec::new(), Vec::new()));
                        dummy_claims.insert(
                            poly_id.clone(),
                            Self::precomputed_claim_for_poly(poly_id.clone(), &mle),
                        );
                        ids.push(poly_id);
                        polys.push(mle);
                        (map_acc, dummy_claims)
                    },
                );
            debug!("{} map created", m.to_span());

            // Here we build the RowMajorMatrices and `model_comms_map`.
            // The `model_comms_map` stores the order of the polynomials in each RowMajorMatrix
            let m = Metrics::new();
            let (model_comms_map, rmms) = map.into_iter().rev().fold(
                (BTreeMap::new(), Vec::new()),
                |(mut map_acc, mut rmm_acc), (nv, (values, polys))| {
                    let im = Metrics::new();
                    let matrix_values = transpose(
                        polys
                            .into_iter()
                            .map(|p| p.get_base_field_vec().to_vec())
                            .collect::<Vec<Vec<E::BaseField>>>(),
                    );
                    let rmm = RowMajorMatrix::new_by_inner_matrix(
                        ceno_p3::matrix::dense::DenseMatrix::new(
                            matrix_values.concat(),
                            values.len(),
                        ),
                        InstancePaddingStrategy::Default,
                    );
                    rmm_acc.push(rmm);
                    map_acc.insert(nv, values);
                    debug!("{} {nv} processed.", im.to_span());

                    (map_acc, rmm_acc)
                },
            );
            debug!("{} model_comms_map built", m.to_span());

            // Build the batch commitment
            let m = Metrics::new();
            let model_commitment = PCS::batch_commit(&prover_params, rmms)
                .map_err(|e| anyhow!("{e:?}"))
                .context("Batch Commitment")?;
            debug!("{} model commitment built", m.to_span());
            (Some(model_commitment), model_comms_map, dummy_model_claims)
        } else {
            (None, BTreeMap::new(), HashMap::new())
        };

        let max_model_num_vars = model_comms_map.keys().max().copied().unwrap_or(0usize);
        Ok(GlobalCommitmentContext {
            verifier_params,
            prover_params,
            model_commitment,
            model_comms_map,
            precomputed_model_claims: dummy_model_claims,
            table_node_id,
            max_model_num_vars,
        })
    }

    fn precomputed_claim_for_poly(
        poly_id: CommitmentId,
        mle: &MultilinearExtension<E>,
    ) -> Claim<E> {
        let mut transcript = BasicTranscript::new(b"dummy_model_claim");
        transcript.append_message(String::from(poly_id).as_bytes());
        // squeeze a random point to evaluate the `mle`
        let point = transcript.read_challenges(mle.num_vars());
        let eval = mle.evaluate(&point);
        Claim::new(point, eval)
    }

    /// Generate a prover/verifier context for the `witness_poly_size` specified as input;
    /// `witness_poly_size` represents the size of the biggest witness polynomial to be
    /// committed to. If no `witness_poly_size` is provided as input, this method generates
    /// the prover/verifier context for `self.max_poly_size()`
    pub(crate) fn generate_contexts(
        self,
    ) -> Result<(CommitmentProverCtx<E, PCS>, CommitmentVerifierCtx<E, PCS>)> {
        let GlobalCommitmentContext {
            prover_params,
            verifier_params,
            model_commitment,
            model_comms_map,
            precomputed_model_claims,
            table_node_id,
            max_model_num_vars,
            ..
        } = self;

        let verifier_ctx = CommitmentVerifierCtx {
            verifier_params,
            model_comms_map: model_comms_map.clone(),
            model_commitment: model_commitment
                .as_ref()
                .map(|commit_with_wit| PCS::get_pure_commitment(commit_with_wit)),
            precomputed_model_claims: precomputed_model_claims.clone(),
            table_node_id,
            max_model_num_vars,
        };

        let prover_ctx = CommitmentProverCtx {
            prover_params,
            model_comms_map,
            precomputed_model_claims: precomputed_model_claims
                .into_par_iter()
                .map(|(poly_id, claim)| {
                    let beta_evals = compute_betas_eval(&claim.point);
                    (poly_id, PrecomputedModelClaim { claim, beta_evals })
                })
                .collect(),
            model_commitment,
            table_node_id,
            max_model_num_vars,
        };

        Ok((prover_ctx, verifier_ctx))
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct PrecomputedModelClaim<E> {
    /// The precomputed model claim
    claim: Claim<E>,
    /// The evaluation of the beta polynomials over claim.point, which is needed
    /// in the opening proof
    beta_evals: Vec<E>,
}

#[derive(Serialize, Deserialize)]
#[serde(bound(serialize = "E: Serialize", deserialize = "E: DeserializeOwned"))]
/// Context data for the commitment prover
pub struct CommitmentProverCtx<E, PCS>
where
    PCS: PolynomialCommitmentScheme<E>,
    PCS::CommitmentWithWitness: Serialize + DeserializeOwned,
    E: ExtensionField,
{
    /// Prover parameters for the [`PolynomialCommitmentScheme`]
    prover_params: PCS::ProverParam,
    /// The batch commitment for the model
    model_commitment: Option<PCS::CommitmentWithWitness>,
    /// Map that stores the position of each individual polynomial in the batch commitment
    model_comms_map: BTreeMap<usize, Vec<CommitmentId>>,
    /// Set of precomputed claims for each model polynomial being committed.
    /// These claims are employed when there is no claim to be opened for the corresponding model polynomial
    precomputed_model_claims: HashMap<CommitmentId, PrecomputedModelClaim<E>>,
    /// This is the [`NodeId`] used for tables in this model
    table_node_id: NodeId,
    /// This is the largest number of variables of any of the polynomials in `model_commitment`
    max_model_num_vars: usize,
}

impl<E, PCS> Debug for CommitmentProverCtx<E, PCS>
where
    E: ExtensionField,
    PCS: PolynomialCommitmentScheme<E>,
    PCS::CommitmentWithWitness: Serialize + DeserializeOwned,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CommitmentProverCtx")
            .field("prover_params", &self.prover_params)
            .field("model_comms_map", &self.model_comms_map)
            .field("precomputed_model_claims", &self.precomputed_model_claims)
            .field("table_node_id", &self.table_node_id)
            .field("max_model_num_vars", &self.max_model_num_vars)
            .finish()
    }
}

impl<E, PCS> CommitmentProverCtx<E, PCS>
where
    PCS: PolynomialCommitmentScheme<E>,
    E: ExtensionField,
    PCS::CommitmentWithWitness: Serialize + DeserializeOwned,
{
    /// Helper method to commit to polynomial.
    pub fn batch_commit(
        &self,
        rmms: Vec<RowMajorMatrix<E::BaseField>>,
    ) -> Result<PCS::CommitmentWithWitness> {
        PCS::batch_commit(&self.prover_params, rmms).map_err(|e| anyhow!("{e:?}"))
    }

    /// Write the commitment context to the transcript
    pub fn write_to_transcript<T: Transcript<E>>(&self, transcript: &mut T) -> Result<()> {
        if let Some(model_comm) = self.model_commitment.as_ref() {
            let comm = PCS::get_pure_commitment(model_comm);
            PCS::write_commitment(&comm, transcript)
                .map_err(|e| anyhow!("{e:?}"))
                .context("Could not write model commitment".to_string())
        } else {
            Ok(())
        }
    }

    pub fn table_node_id(&self) -> NodeId {
        self.table_node_id
    }
}

/// Context data for the commitment verifier
#[derive(Clone, Serialize, Deserialize)]
#[serde(bound(serialize = "E: Serialize", deserialize = "E: DeserializeOwned"))]
pub struct CommitmentVerifierCtx<E, PCS>
where
    PCS: PolynomialCommitmentScheme<E>,
    E: ExtensionField,
{
    /// Verifier parameters for the [`PolynomialCommitmentScheme`]
    verifier_params: PCS::VerifierParam,
    /// The batch commitment for the model
    model_commitment: Option<PCS::Commitment>,
    /// Map that stores the position of each individual polynomial in the batch commitment
    model_comms_map: BTreeMap<usize, Vec<CommitmentId>>,
    /// Set of precomputed dummy claims for each model polynomial being committed.
    /// These claims are employed when there is no claim to be opened for the corresponding model polynomial
    precomputed_model_claims: HashMap<CommitmentId, Claim<E>>,
    /// This is the [`NodeId`] used for tables in this model
    table_node_id: NodeId,
    /// This is the largest number of variables of any of the polynomials in `model_commitment`
    max_model_num_vars: usize,
}

impl<E, PCS> Debug for CommitmentVerifierCtx<E, PCS>
where
    E: ExtensionField,
    PCS: PolynomialCommitmentScheme<E>,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CommitmentVerifierCtx")
            .field("verifier_params", &self.verifier_params)
            .field("model_comms_map", &self.model_comms_map)
            .field("precomputed_model_claims", &self.precomputed_model_claims)
            .field("table_node_id", &self.table_node_id)
            .field("max_model_num_vars", &self.max_model_num_vars)
            .finish()
    }
}

impl<E, PCS> CommitmentVerifierCtx<E, PCS>
where
    PCS: PolynomialCommitmentScheme<E>,
    E: ExtensionField,
{
    pub fn write_to_transcript<T: Transcript<E>>(&self, transcript: &mut T) -> Result<()> {
        if let Some(model_comm) = self.model_commitment.as_ref() {
            PCS::write_commitment(model_comm, transcript)
                .map_err(|e| anyhow!("{e:?}"))
                .context("Could not write model commitment".to_string())
        } else {
            Ok(())
        }
    }

    pub fn table_node_id(&self) -> NodeId {
        self.table_node_id
    }
}

// Type alias used to represent the set of claims of the model polys
pub(crate) type ModelClaims<E> = HashMap<CommitmentId, BTreeMap<NodeId, Claim<E>>>;

#[derive(Clone, Serialize, Deserialize)]
#[serde(bound(serialize = "E: Serialize", deserialize = "E: DeserializeOwned"))]
pub struct OpeningProof<E, PCS: PolynomialCommitmentScheme<E>>
where
    E: ExtensionField,
{
    /// This is the sumcheck proof that is used so that all model polynomials are evaluated at the same point.
    sumcheck_proof: IOPProof<E>,
    /// This is the list of evals for all the model commitments after the sumcheck.
    sumcheck_evals: Vec<E>,
    /// The opening proof for the commitments FOR the witness polynomials
    pcs_proof: PCS::Proof,
}
