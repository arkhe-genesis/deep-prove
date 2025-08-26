//! Module containing the logic to commit to instance and witness polynomials for a model using a MMCS
use std::{
    collections::{BTreeMap, HashMap},
    fmt::Debug,
};

use crate::{layers::provable::NodeId, lookup::context::TableType};
use anyhow::{Context, Result, anyhow};
use either::Either;
use ff_ext::ExtensionField;

use mpcs::PolynomialCommitmentScheme;
use multilinear_extensions::{Expression, mle::MultilinearExtension, util::transpose};
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use tracing::debug;
use utils::Metrics;
use witness::{InstancePaddingStrategy, RowMajorMatrix};

use sumcheck::structs::IOPProof;
use transcript::Transcript;

mod prover;
pub use prover::CommitmentProver;

mod verifier;
pub use verifier::CommitmentVerifier;

pub type PolyId = String;

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
    model_comms_map: BTreeMap<usize, Vec<(NodeId, PolyId)>>,
    /// This is the [`NodeId`] used for tables in this model
    table_node_id: NodeId,
    /// This is the [`Expression`] used in the sumcheck so that everything is evaluated at the same point
    sumcheck_expression: Expression<E>,
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
            .field("table_node_id", &self.table_node_id)
            .field("sumcheck_expression", &self.sumcheck_expression)
            .field("max_model_num_vars", &self.max_model_num_vars)
            .finish()
    }
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
        polys: Vec<(NodeId, HashMap<PolyId, MultilinearExtension<E>>)>,
        lookup_ctx: &[TableType],
        max_node_id: NodeId,
    ) -> Result<GlobalCommitmentContext<E, PCS>> {
        // Find the maximum size so we can generate params
        let max_poly_size = polys
            .iter()
            .fold(witness_poly_size, |mut acc, (_, poly_vec)| {
                poly_vec
                    .iter()
                    .for_each(|(_, poly)| acc = acc.max(1 << poly.num_vars()));
                acc
            })
            .next_power_of_two();

        let m = Metrics::new();
        let (prover_params, verifier_params) = {
            let param = PCS::setup(max_poly_size, mpcs::SecurityLevel::Conjecture100bits)
                .map_err(|e| anyhow!("{:?}", e))
                .context("setting up params")?;

            PCS::trim(param, max_poly_size)
                .map_err(|e| anyhow!("{:?}", e))
                .context("trimming params")?
        };
        debug!("{} PPs & VPs built", m.to_span());

        // Find the maximum node id used in this model so we can pick a unique node id for table related commitments.
        let table_node_id = NodeId(max_node_id.0 + 1);

        // First we take all the model polys and sort them by the number of variables they have.
        // Then we do the same for any table commitments but here we set all of them to have `table_node_id`.
        let table_commitments_check = lookup_ctx.iter().any(|tt| tt.has_committed_claims());
        let (model_commitment, model_comms_map) = if !polys.is_empty() || table_commitments_check {
            let m = Metrics::new();
            let map = polys
                .into_iter()
                .flat_map(|(node_id, hash_map)| {
                    hash_map
                        .into_iter()
                        .map(|(k, v)| (v.num_vars(), (node_id, k, v)))
                        .collect::<Vec<(usize, (NodeId, String, MultilinearExtension<E>))>>()
                })
                .chain(lookup_ctx.iter().filter_map(|table_type| {
                    table_type
                        .committed_columns()
                        .map(|mle| (mle.num_vars(), (table_node_id, table_type.name(), mle)))
                }))
                .fold(
                    BTreeMap::new(),
                    |mut map_acc, (num_vars, (node_id, name, mle))| {
                        let (ids, polys): &mut (
                            Vec<(NodeId, PolyId)>,
                            Vec<MultilinearExtension<E>>,
                        ) = map_acc
                            .entry(num_vars)
                            .or_insert_with(|| (Vec::new(), Vec::new()));
                        ids.push((node_id, name));
                        polys.push(mle);
                        map_acc
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
                .map_err(|e| anyhow!("{:?}", e))
                .context("Batch Commitment")?;
            debug!("{} model commitment built", m.to_span());
            (Some(model_commitment), model_comms_map)
        } else {
            (None, BTreeMap::new())
        };
        // Work out how many polynomials we have in total so that we can pre-make the sumcheck expression
        let total_polys = model_comms_map
            .values()
            .map(|polys| polys.len())
            .sum::<usize>();
        let sumcheck_expression =
            (0..total_polys).fold(Expression::Constant(Either::Right(E::ZERO)), |acc, j| {
                acc + Expression::Challenge(0, j, E::ONE, E::ZERO)
                    * Expression::WitIn(j as u16)
                    * Expression::WitIn((j + total_polys) as u16)
            });

        let max_model_num_vars = model_comms_map.keys().max().copied().unwrap_or(0usize);
        Ok(GlobalCommitmentContext {
            verifier_params,
            prover_params,
            model_commitment,
            model_comms_map,
            table_node_id,
            sumcheck_expression,
            max_model_num_vars,
        })
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
            table_node_id,
            sumcheck_expression,
            max_model_num_vars,
            ..
        } = self;

        let verifier_ctx = CommitmentVerifierCtx {
            verifier_params,
            model_comms_map: model_comms_map.clone(),
            model_commitment: model_commitment
                .as_ref()
                .map(|commit_with_wit| PCS::get_pure_commitment(commit_with_wit)),
            table_node_id,
            sumcheck_expression: sumcheck_expression.clone(),
            max_model_num_vars,
        };

        let prover_ctx = CommitmentProverCtx {
            prover_params,
            model_comms_map,
            model_commitment,
            table_node_id,
            sumcheck_expression,
            max_model_num_vars,
        };

        Ok((prover_ctx, verifier_ctx))
    }
}

#[derive(Serialize, Deserialize)]
#[serde(bound(serialize = "E: Serialize", deserialize = "E: DeserializeOwned"))]
/// Context data for the commitment prover
pub struct CommitmentProverCtx<E, PCS>
where
    PCS: PolynomialCommitmentScheme<E>,
    PCS::CommitmentWithWitness: Serialize + DeserializeOwned,
    E::BaseField: Serialize + DeserializeOwned,
    E: ExtensionField + Serialize + DeserializeOwned,
{
    /// Prover parameters for the [`PolynomialCommitmentScheme`]
    prover_params: PCS::ProverParam,
    /// The batch commitment for the model
    model_commitment: Option<PCS::CommitmentWithWitness>,
    /// Map that stores the position of each individual polynomial in the batch commitment
    model_comms_map: BTreeMap<usize, Vec<(NodeId, PolyId)>>,
    /// This is the [`NodeId`] used for tables in this model
    table_node_id: NodeId,
    /// This is the [`Expression`] used in the sumcheck so that everything is evaluated at the same point
    sumcheck_expression: Expression<E>,
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
            .field("table_node_id", &self.table_node_id)
            .field("sumcheck_expression", &self.sumcheck_expression)
            .field("max_model_num_vars", &self.max_model_num_vars)
            .finish()
    }
}

impl<E, PCS> CommitmentProverCtx<E, PCS>
where
    PCS: PolynomialCommitmentScheme<E>,
    E::BaseField: Serialize + DeserializeOwned,
    E: ExtensionField + Serialize + DeserializeOwned,
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
    E::BaseField: Serialize + DeserializeOwned,
    E: ExtensionField + Serialize + DeserializeOwned,
{
    /// Verifier parameters for the [`PolynomialCommitmentScheme`]
    verifier_params: PCS::VerifierParam,
    /// The batch commitment for the model
    model_commitment: Option<PCS::Commitment>,
    /// Map that stores the position of each individual polynomial in the batch commitment
    model_comms_map: BTreeMap<usize, Vec<(NodeId, PolyId)>>,
    /// This is the [`NodeId`] used for tables in this model
    table_node_id: NodeId,
    /// This is the [`Expression`] used in the sumcheck so that everything is evaluated at the same point
    sumcheck_expression: Expression<E>,
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
            .field("table_node_id", &self.table_node_id)
            .field("sumcheck_expression", &self.sumcheck_expression)
            .field("max_model_num_vars", &self.max_model_num_vars)
            .finish()
    }
}

impl<E, PCS> CommitmentVerifierCtx<E, PCS>
where
    PCS: PolynomialCommitmentScheme<E>,
    E::BaseField: Serialize + DeserializeOwned,
    E: ExtensionField + Serialize + DeserializeOwned,
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

#[derive(Clone, Serialize, Deserialize)]
#[serde(bound(serialize = "E: Serialize", deserialize = "E: DeserializeOwned"))]
pub struct ModelOpeningProof<E, PCS: PolynomialCommitmentScheme<E>>
where
    E: ExtensionField + Serialize + DeserializeOwned,
{
    /// This is the sumcheck proof that is used so that all model polynomials are evaluated at the same point.
    sumcheck_proof: IOPProof<E>,
    /// This is the list of evals for all the model commitments after the sumcheck.
    sumcheck_evals: Vec<E>,
    /// The opening proof for the commitments
    pcs_proof: PCS::Proof,
}
