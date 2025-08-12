//! This module contains logic to prove the correct opening of several claims from several independent
//! polynomials.

use std::collections::{BTreeMap, HashMap, HashSet};

use crate::{
    Claim, default_transcript,
    layers::provable::NodeId,
    lookup::context::{LookupContext, TableType},
};
use ff_ext::ExtensionField;

use anyhow::{Context, Result, anyhow, ensure};
use itertools::Itertools;
use mpcs::{Evaluation, PolynomialCommitmentScheme};
use multilinear_extensions::{
    mle::{FieldType, MultilinearExtension},
    smart_slice::SmartSlice,
};
use rayon::prelude::*;
use serde::{Deserialize, Serialize, de::DeserializeOwned};

use tracing::debug;
use transcript::Transcript;

pub type PolyId = String;

type ModelCommitmentsMap<'a, PCS, E> = BTreeMap<
    NodeId,
    BTreeMap<
        PolyId,
        (
            <PCS as PolynomialCommitmentScheme<E>>::CommitmentWithWitness,
            MultilinearExtension<'a, E>,
        ),
    >,
>;

/// Data structure representing the context data that is necessary to properly derive
/// a pair of `CommitmentProverCtx`, `CommitmentVerifierCtx` for a given size of the
/// biggest polynomial to be committed to. This structure allows to derive a pair
/// of prover/verifier contexts as long as the size of the biggest polynomial to be
/// committed is at most `self.max_poly_size`
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound(serialize = "E: Serialize", deserialize = "E: DeserializeOwned"))]
pub(crate) struct GlobalCommitmentCtx<'a, E, PCS>
where
    PCS: PolynomialCommitmentScheme<E>,
    E::BaseField: Serialize + DeserializeOwned,
    E: ExtensionField + Serialize + DeserializeOwned,
{
    /// Parameters for the [`PolynomialCommitmentScheme`]
    params: PCS::Param,
    /// Size of the maximum polynomial suppoted by `params`
    max_poly_size: usize,
    /// Size of the maximum constant polynomial found in the model
    constant_polys_max_size: usize,
    /// This field contains the constant polynomials associated to each node in the model, identifier by their [`PolyId`].
    model_polys: Vec<(NodeId, HashMap<PolyId, MultilinearExtension<'a, E>>)>,
    /// This field contains the polynomials associated to each lookup table employed in the model
    table_polys: HashMap<TableType, MultilinearExtension<'a, E>>,
}

pub(crate) struct ContextGenerator<'a, E: ExtensionField, PCS: PolynomialCommitmentScheme<E>> {
    poly_sizes: Box<dyn Iterator<Item = usize>>,
    ctx: GlobalCommitmentCtx<'a, E, PCS>,
}

impl<'a, E, PCS> Iterator for ContextGenerator<'a, E, PCS>
where
    E: ExtensionField,
    PCS: PolynomialCommitmentScheme<E>,
{
    type Item = Result<(
        usize,
        (
            CommitmentProverCtx<'a, E, PCS>,
            CommitmentVerifierCtx<E, PCS>,
        ),
    )>;

    fn next(&mut self) -> Option<Self::Item> {
        self.poly_sizes.as_mut().next().map(|poly_size| {
            self.ctx
                .clone()
                .generate_contexts(Some(poly_size))
                .map(|(prover_ctx, verifier_ctx)| (poly_size, (prover_ctx, verifier_ctx)))
        })
    }
}

impl<'a, E, PCS> GlobalCommitmentCtx<'a, E, PCS>
where
    PCS: PolynomialCommitmentScheme<E>,
    E::BaseField: Serialize + DeserializeOwned,
    E: ExtensionField + Serialize + DeserializeOwned,
{
    /// Instantiate a new instance of `Self`. `witness_poly_size` must be the size of the
    /// biggest witness polynomial to be committed that is found in the model,
    /// while `polys` is the set of constant polynomials employed across all the layers
    /// of the model that needs to be committed
    pub(crate) fn new(
        witness_poly_size: usize,
        polys: Vec<(NodeId, HashMap<PolyId, MultilinearExtension<'a, E>>)>,
        lookup_ctx: &LookupContext,
    ) -> Result<GlobalCommitmentCtx<'a, E, PCS>> {
        let table_polys: HashMap<_, _> = lookup_ctx
            .iter()
            .filter_map(|table_type| {
                table_type
                    .committed_columns()
                    .map(|poly| (table_type.clone(), mle_to_owned(poly)))
            })
            .collect();
        // Find the maximum size so we can generate params
        let constant_polys_max_size = polys
            .iter()
            .flat_map(|(node_id, poly_vec)| {
                debug!(
                    "Context Commitment: node {node_id} has {} polynomials of sizes {:?}",
                    poly_vec.len(),
                    poly_vec
                        .values()
                        .map(|poly| poly.num_vars())
                        .collect::<Vec<_>>()
                );
                poly_vec.values().collect_vec()
            })
            .chain(table_polys.values())
            .fold(0usize, |acc, poly| acc.max(1 << poly.num_vars()))
            .next_power_of_two();

        let max_poly_size = constant_polys_max_size.max(witness_poly_size.next_power_of_two());
        debug!("Setting up PCS params for max size {} poly", max_poly_size);
        let params = PCS::setup(max_poly_size).context("setting up params")?;

        Ok(Self {
            params,
            max_poly_size,
            constant_polys_max_size,
            model_polys: polys,
            table_polys,
        })
    }

    /// Generate a prover/verifier context for the `witness_poly_size` specified as input;
    /// `witness_poly_size` represents the size of the biggest witness polynomial to be
    /// committed to. If no `witness_poly_size` is provided as input, this method generates
    /// the prover/verifier context for `self.max_poly_size()`
    pub(crate) fn generate_contexts(
        self,
        witness_poly_size: Option<usize>,
    ) -> Result<(
        CommitmentProverCtx<'a, E, PCS>,
        CommitmentVerifierCtx<E, PCS>,
    )> {
        let trimmed_poly_size = if let Some(poly_size) = witness_poly_size {
            ensure!(
                self.max_poly_size >= poly_size,
                "Witness polynomial size {poly_size} is larger than the maximum polynomial size supported by Global Context {}",
                self.max_poly_size
            );
            self.constant_polys_max_size
                .max(poly_size.next_power_of_two())
        } else {
            self.max_poly_size
        };

        let (prover_params, verifier_params) = PCS::trim(self.params, trimmed_poly_size)?;

        let model_comms_map = self
            .model_polys
            .into_par_iter()
            .map(|(node_id, polys_vec)| {
                let model_comms = polys_vec
                    .into_iter()
                    .map(|(id, poly)| {
                        let commit = PCS::commit(&prover_params, poly.clone())
                            .with_context(|| format!("committing to polynomial {id}"))?;
                        Result::<_, anyhow::Error>::Ok((id, (commit, poly)))
                    })
                    .collect::<Result<BTreeMap<_, _>, _>>()
                    .with_context(|| format!("collecting node {node_id} commitments"))?;
                Result::<_, anyhow::Error>::Ok((node_id, model_comms))
            })
            .collect::<Result<BTreeMap<NodeId, _>, _>>()
            .context(format!(
                "collecting model commitments for size {trimmed_poly_size}"
            ))?;

        let table_comms_map =
            self.table_polys
                .into_par_iter()
                .map(|(table_type, poly)| {
                    let commit = PCS::commit(&prover_params, poly.clone())?;
                    Ok((table_type, (commit, poly)))
                })
                .collect::<Result<
                    BTreeMap<TableType, (PCS::CommitmentWithWitness, MultilinearExtension<E>)>,
                >>()?;

        let verifier_ctx = CommitmentVerifierCtx {
            verifier_params,
            model_comms_map: model_comms_map
                .iter()
                .map(|(&node_id, comms_vec)| {
                    (
                        node_id,
                        comms_vec
                            .iter()
                            .map(|(id, (comm, _))| (id.clone(), PCS::get_pure_commitment(comm)))
                            .collect::<BTreeMap<PolyId, PCS::Commitment>>(),
                    )
                })
                .collect(),
            table_comms_map: table_comms_map
                .iter()
                .map(|(table_type, (comm, _))| (table_type.clone(), PCS::get_pure_commitment(comm)))
                .collect(),
        };

        let prover_ctx = CommitmentProverCtx {
            prover_params,
            model_comms_map,
            table_comms_map,
        };

        Ok((prover_ctx, verifier_ctx))
    }

    /// Generate a set of prover/verifier contexts for the `witness_poly_sizes` specified as input;
    /// Each `witness_poly_size` represents the size of the biggest witness polynomial to be
    /// committed to
    pub(crate) fn generate_all_contexts(
        self,
        witness_poly_sizes: Vec<usize>,
    ) -> Result<ContextGenerator<'a, E, PCS>> {
        // first, build the set of different poly size for which we need to build a context. Note that multiple
        // `witness_poly_sizes` might be mapped to the same poly size for the context, so the returned set might
        // have less entries than `witness_poly_sizes`
        let actual_poly_sizes: HashSet<usize> = witness_poly_sizes
            .into_iter()
            .map(|witness_poly_size| {
                self.constant_polys_max_size
                    .max(witness_poly_size.next_power_of_two())
            })
            .collect();

        let iterator = ContextGenerator {
            poly_sizes: Box::new(actual_poly_sizes.into_iter()),
            ctx: self,
        };

        Ok(iterator)
    }
}

/// Context data for the commitment prover
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound(serialize = "E: Serialize", deserialize = "E: DeserializeOwned"))]
pub struct CommitmentProverCtx<'a, E, PCS>
where
    PCS: PolynomialCommitmentScheme<E>,
    E::BaseField: Serialize + DeserializeOwned,
    E: ExtensionField + Serialize + DeserializeOwned,
{
    /// Prover parameters for the [`PolynomialCommitmentScheme`]
    prover_params: PCS::ProverParam,
    /// This field contains a [`BTreeMap`] where the key is a [`NodeId`] and the value is a vector of tuples of [`PolynomialCommitmentScheme::CommitmentWithWitness`]  and [`DenseMultilinearExtension<E>`] corresponding to that ID.
    /// A [`BTreeMap`] is used to ensure that the commitments are written in a deterministic order
    /// to the transcript
    model_comms_map: ModelCommitmentsMap<'a, PCS, E>,
    /// This field contains a [`BTreeMap`] relating to lookup tables used by the model.
    /// A [`BTreeMap`] is used to ensure that the commitments are written in a deterministic order
    /// to the transcript
    table_comms_map: BTreeMap<TableType, (PCS::CommitmentWithWitness, MultilinearExtension<'a, E>)>,
}

impl<'a, E, PCS> CommitmentProverCtx<'a, E, PCS>
where
    PCS: PolynomialCommitmentScheme<E>,
    E::BaseField: Serialize + DeserializeOwned,
    E: ExtensionField + Serialize + DeserializeOwned,
{
    /// Helper method to commit to polynomial.
    pub fn commit(&self, mle: &MultilinearExtension<'a, E>) -> Result<PCS::CommitmentWithWitness> {
        PCS::commit(&self.prover_params, mle.clone()).map_err(|e| e.into())
    }

    /// Write the commitment context to the transcript
    pub fn write_to_transcript<T: Transcript<E>>(&self, transcript: &mut T) -> Result<()> {
        self.model_comms_map
            .iter()
            .try_for_each(|(node_id, comms_vec)| {
                comms_vec.iter().try_for_each(|(id, (comm, _))| {
                    let v_comm = PCS::get_pure_commitment(comm);
                    PCS::write_commitment(&v_comm, transcript).context(format!(
                        "Could not write commitment for polynomial {id} of node {node_id}"
                    ))
                })
            })?;
        self.table_comms_map
            .iter()
            .try_for_each(|(table_type, (comm, _))| {
                let v_comm = PCS::get_pure_commitment(comm);
                PCS::write_commitment(&v_comm, transcript).context(format!(
                    "Could not write commitment for polynomial of table {}",
                    table_type.name()
                ))
            })
    }
}

/// Context data for the commitment verifier
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound(serialize = "E: Serialize", deserialize = "E: DeserializeOwned"))]
pub struct CommitmentVerifierCtx<E, PCS>
where
    PCS: PolynomialCommitmentScheme<E>,
    E::BaseField: Serialize + DeserializeOwned,
    E: ExtensionField + Serialize + DeserializeOwned,
{
    /// Verifier parameters for the [`PolynomialCommitmentScheme`]
    verifier_params: PCS::VerifierParam,
    /// This field contains a [`BTreeMap`] where the key is a [`NodeId`] and the value is a vector of tuples of [`PolynomialCommitmentScheme::Commitment`] corresponding to that ID.
    /// A [`BTreeMap`] is used to ensure that the commitments are written in a deterministic order
    /// to the transcript
    model_comms_map: BTreeMap<NodeId, BTreeMap<PolyId, PCS::Commitment>>,
    /// This field contains a [`BTreeMap`] relating to lookup tables used by the model
    /// A [`BTreeMap`] is used to ensure that the commitments are written in a deterministic order
    /// to the transcript
    table_comms_map: BTreeMap<TableType, PCS::Commitment>,
}

impl<E, PCS> CommitmentVerifierCtx<E, PCS>
where
    PCS: PolynomialCommitmentScheme<E>,
    E::BaseField: Serialize + DeserializeOwned,
    E: ExtensionField + Serialize + DeserializeOwned,
{
    pub fn write_to_transcript<T: Transcript<E>>(&self, transcript: &mut T) -> Result<()> {
        self.model_comms_map
            .iter()
            .try_for_each(|(node_id, comms_vec)| {
                comms_vec.iter().try_for_each(|(id, comm)| {
                    PCS::write_commitment(comm, transcript).context(format!(
                        "Could not write commitment for polynomial {id} of node {node_id}"
                    ))
                })
            })?;
        self.table_comms_map
            .iter()
            .try_for_each(|(table_type, comm)| {
                PCS::write_commitment(comm, transcript).context(format!(
                    "Could not write commitment for polynomial of table {}",
                    table_type.name()
                ))
            })
    }
}

#[derive(Clone, Debug)]
/// Claim about a polynomial used by the prover (so contain witness as well)
pub struct CommitmentClaim<'a, E, PCS>
where
    PCS: PolynomialCommitmentScheme<E>,
    E::BaseField: Serialize + DeserializeOwned,
    E: ExtensionField + Serialize + DeserializeOwned,
{
    commitment: PCS::CommitmentWithWitness,
    poly: MultilinearExtension<'a, E>,
    claim: Claim<E>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound(serialize = "E: Serialize", deserialize = "E: DeserializeOwned"))]
/// Claim about a commitment used by the verifier (so no witness is included).
pub struct VerifierClaim<E, PCS>
where
    PCS: PolynomialCommitmentScheme<E>,
    E::BaseField: Serialize + DeserializeOwned,
    E: ExtensionField,
{
    commitment: PCS::Commitment,
    claim: Claim<E>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
/// The opening proof for a model inference. We may have trivial proofs that occur when the prover has to commit
/// to small witness polynomials.
pub struct ModelOpeningProof<E, PCS>
where
    PCS: PolynomialCommitmentScheme<E>,
    E::BaseField: Serialize + DeserializeOwned,
    E: ExtensionField,
{
    batch_proof: PCS::Proof,
    trivial_proofs: Vec<PCS::Proof>,
}

impl<E, PCS> ModelOpeningProof<E, PCS>
where
    PCS: PolynomialCommitmentScheme<E>,
    E::BaseField: Serialize + DeserializeOwned,
    E: ExtensionField,
{
    /// Creates a new [`ModelOpeningProof`] from constituent parts.
    pub fn new(
        batch_proof: PCS::Proof,
        trivial_proofs: Vec<PCS::Proof>,
    ) -> ModelOpeningProof<E, PCS> {
        ModelOpeningProof {
            batch_proof,
            trivial_proofs,
        }
    }

    /// Getter for the batch proof
    pub fn batch_proof(&self) -> &PCS::Proof {
        &self.batch_proof
    }

    /// Getter for the trivial proofs
    pub fn trivial_proofs(&self) -> &[PCS::Proof] {
        &self.trivial_proofs
    }
}

fn mle_to_owned<'a, E: ExtensionField>(
    mle: MultilinearExtension<'a, E>,
) -> MultilinearExtension<'static, E> {
    let evaluations = match &mle.evaluations {
        FieldType::Base(smart_slice) => FieldType::Base(SmartSlice::Owned(smart_slice.to_vec())),
        FieldType::Ext(smart_slice) => FieldType::Ext(SmartSlice::Owned(smart_slice.to_vec())),
        FieldType::Unreachable => unreachable!(),
    };

    MultilinearExtension {
        evaluations,
        num_vars: mle.num_vars,
    }
}

#[derive(Debug, Clone)]
/// Struct used to batch prove all commitment openings in a model proof.
pub struct CommitmentProver<'a, E, PCS>
where
    PCS: PolynomialCommitmentScheme<E>,
    E::BaseField: Serialize + DeserializeOwned,
    E: ExtensionField + Serialize + DeserializeOwned,
{
    /// Claims that are made about non-trivial commitments
    claims: Vec<CommitmentClaim<'a, E, PCS>>,
    /// Claims about trivial commitments (fewer than 8 variables, in this case its more efficient just to evaluate the polynomial)
    trivial_claims: Vec<CommitmentClaim<'a, E, PCS>>,
}

impl<'a, E, PCS> CommitmentProver<'a, E, PCS>
where
    PCS: PolynomialCommitmentScheme<E>,
    E::BaseField: Serialize + DeserializeOwned,
    E: ExtensionField + Serialize + DeserializeOwned,
{
    /// Create a new [`CommitmentProver`] from the [`CommitmentContext`] for the model.
    pub fn new() -> CommitmentProver<'a, E, PCS> {
        CommitmentProver {
            claims: vec![],
            trivial_claims: vec![],
        }
    }
    /// Add a claim about a witness polynomial.
    pub fn add_witness_claim(
        &mut self,
        (commitment, mle): (PCS::CommitmentWithWitness, MultilinearExtension<'a, E>),
        claim: Claim<E>,
    ) -> Result<()> {
        if mle.num_vars() <= PCS::trivial_num_vars() {
            self.trivial_claims.push(CommitmentClaim {
                commitment,
                poly: mle,
                claim,
            });
        } else {
            self.claims.push(CommitmentClaim {
                commitment,
                poly: mle,
                claim,
            });
        }
        Ok(())
    }
    /// Add claims about model weights and biases for a certain node
    pub fn add_common_claims(
        &mut self,
        ctx: &CommitmentProverCtx<'a, E, PCS>,
        node_id: NodeId,
        mut claims: HashMap<PolyId, Claim<E>>,
    ) -> Result<()> {
        if claims.is_empty() {
            // No claims to be added
            return Ok(());
        }
        let node_commitments = ctx.model_comms_map.get(&node_id).cloned().ok_or(anyhow!(
            "No commitments stored for node with id: {}",
            node_id
        ))?;
        node_commitments
            .into_iter()
            .try_for_each(|(id, comm_with_wit)| {
                let claim = claims.remove(&id).ok_or_else(|| {
                    anyhow!("No claim found for poly id {} in node {}", id, node_id)
                })?;
                self.add_witness_claim(comm_with_wit, claim)
            })
    }

    /// Adds a claim about a table polynomial
    pub fn add_table_claim(
        &mut self,
        ctx: &CommitmentProverCtx<'a, E, PCS>,
        table_type: TableType,
        claim: Claim<E>,
    ) -> Result<()> {
        let table_commitment = ctx
            .table_comms_map
            .get(&table_type)
            .cloned()
            .ok_or(anyhow!(
                "No table commitments stored for table of type: {}",
                table_type.name()
            ))?;

        self.add_witness_claim(table_commitment, claim)
    }

    /// Produce the [`ModelOpeningProof`] for this inference trace.
    pub fn prove<T: Transcript<E>>(
        &mut self,
        commitment_context: &CommitmentProverCtx<E, PCS>,
        transcript: &mut T,
    ) -> Result<ModelOpeningProof<E, PCS>> {
        // Prepare the parts that go into the batch proof
        #[allow(clippy::type_complexity)]
        let (comms, (polys, (points, evaluations))): (
            Vec<PCS::CommitmentWithWitness>,
            (
                Vec<MultilinearExtension<'_, E>>,
                (Vec<Vec<E>>, Vec<Evaluation<E>>),
            ),
        ) = self
            .claims
            .par_drain(..)
            .enumerate()
            .map(|(i, claim)| {
                let CommitmentClaim {
                    commitment,
                    poly,
                    claim,
                } = claim;
                let Claim { point, eval } = claim;

                let evaluation = Evaluation::<E>::new(i, i, eval);
                (commitment, (poly, (point, evaluation)))
            })
            .unzip();
        // Make the trivial proofs.
        let trivial_proofs = self
            .trivial_claims
            .iter()
            .map(|claim| {
                let CommitmentClaim {
                    commitment,
                    poly,
                    claim: inner_claim,
                } = claim;
                let Claim { point, eval } = inner_claim;
                PCS::open(
                    &commitment_context.prover_params,
                    poly,
                    commitment,
                    point,
                    eval,
                    transcript,
                )
                .map_err(|e| anyhow!("Could not open trivial commitment: {:?}", e))
            })
            .collect::<Result<Vec<PCS::Proof>, anyhow::Error>>()?;

        // Make the batch proof
        let batch_proof = PCS::batch_open(
            &commitment_context.prover_params,
            &polys,
            &comms,
            &points,
            &evaluations,
            transcript,
        )?;

        Ok(ModelOpeningProof::new(batch_proof, trivial_proofs))
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(bound(serialize = "E: Serialize", deserialize = "E: DeserializeOwned"))]
/// The struct used to verify all of the commitment openings in a model proof.
pub struct CommitmentVerifier<E, PCS>
where
    PCS: PolynomialCommitmentScheme<E>,
    E::BaseField: Serialize + DeserializeOwned,
    E: ExtensionField,
{
    model_comms_map: BTreeMap<NodeId, BTreeMap<PolyId, PCS::Commitment>>,
    table_comms_map: BTreeMap<TableType, PCS::Commitment>,
    claims: Vec<VerifierClaim<E, PCS>>,
    trivial_claims: Vec<VerifierClaim<E, PCS>>,
}

impl<E, PCS> CommitmentVerifier<E, PCS>
where
    PCS: PolynomialCommitmentScheme<E>,
    E::BaseField: Serialize + DeserializeOwned,
    E: ExtensionField + Serialize + DeserializeOwned,
{
    /// Create a new [`CommitmentVerifier`] from the models [`CommitmentContext`].
    pub fn new(ctx: &CommitmentVerifierCtx<E, PCS>) -> CommitmentVerifier<E, PCS> {
        CommitmentVerifier {
            model_comms_map: ctx.model_comms_map.clone(),
            table_comms_map: ctx.table_comms_map.clone(),
            claims: vec![],
            trivial_claims: vec![],
        }
    }
    /// Add a claim about a witness poly to be verified.
    pub fn add_witness_claim(
        &mut self,
        commitment: PCS::Commitment,
        claim: Claim<E>,
    ) -> Result<()> {
        if claim.point.len() <= PCS::trivial_num_vars() {
            self.trivial_claims
                .push(VerifierClaim { commitment, claim });
        } else {
            self.claims.push(VerifierClaim { commitment, claim });
        }
        Ok(())
    }

    /// Add claims about model weights and biases for a certain node
    pub fn add_common_claims(
        &mut self,
        node_id: NodeId,
        mut claims: HashMap<PolyId, Claim<E>>,
    ) -> Result<()> {
        if claims.is_empty() {
            // No claims to be added
            return Ok(());
        }
        let node_commitments = self.model_comms_map.remove(&node_id).ok_or(anyhow!(
            "No commitments stored for node with id: {}",
            node_id
        ))?;

        node_commitments
            .into_iter()
            .try_for_each(|(id, comm_with_wit)| {
                let claim = claims.remove(&id).ok_or_else(|| {
                    anyhow!("No claim found for poly id {} in node {}", id, node_id)
                })?;
                self.add_witness_claim(comm_with_wit, claim)
            })
    }

    /// Adds a claim about a table polynomial
    pub fn add_table_claim(&mut self, table_type: TableType, claim: Claim<E>) -> Result<()> {
        let table_commitment = self.table_comms_map.remove(&table_type).ok_or(anyhow!(
            "No table commitments stored for table of type: {}",
            table_type.name()
        ))?;

        self.add_witness_claim(table_commitment, claim)
    }

    /// Verify the [`ModelOpeningProof`] for this inference trace.
    pub fn verify<T: Transcript<E>>(
        &mut self,
        commitment_context: &CommitmentVerifierCtx<E, PCS>,
        proof: &ModelOpeningProof<E, PCS>,
        transcript: &mut T,
    ) -> Result<()> {
        // Check that all the model commitments have been used
        ensure!(
            self.model_comms_map.is_empty(),
            "Not all model commits have been used, had {} remaining",
            self.model_comms_map.len()
        );
        // Check all the table commitments have been used
        ensure!(
            self.table_comms_map.is_empty(),
            "Not all table commits have been used, had {} remaining",
            self.table_comms_map.len()
        );

        // Prepare the parts that go into the batch proof
        let (comms, points, evaluations) = self.claims.drain(..).enumerate().fold(
            (vec![], vec![], vec![]),
            |(mut comms_acc, mut points_acc, mut evals_acc), (i, claim)| {
                let VerifierClaim { commitment, claim } = claim;
                let Claim { point, eval } = claim;

                let evaluation = Evaluation::<E>::new(i, i, eval);
                comms_acc.push(commitment);

                points_acc.push(point);
                evals_acc.push(evaluation);
                (comms_acc, points_acc, evals_acc)
            },
        );

        // Ensure that if we have trivial claims then we also have the same number of trivial proofs
        let trivial_proofs = proof.trivial_proofs();

        ensure!(
            self.trivial_claims.len() == trivial_proofs.len(),
            "Openign proof had {} trivial proofs, but the verifier has {} trivial claims",
            trivial_proofs.len(),
            self.trivial_claims.len()
        );

        // Check all trivial commitments are correct
        self.trivial_claims
            .par_iter()
            .zip(trivial_proofs.par_iter())
            .try_for_each(|(claim, proof)| {
                let VerifierClaim {
                    commitment,
                    claim: inner_claim,
                } = claim;
                let Claim { point, eval } = inner_claim;
                // Check that the commitments align, we can use a default transcript because trivial openings don't require a transcript
                let mut t = default_transcript::<E>();
                PCS::verify(
                    &commitment_context.verifier_params,
                    commitment,
                    point,
                    eval,
                    proof,
                    &mut t,
                )?;
                Result::<(), anyhow::Error>::Ok(())
            })?;
        // Verify the batch opening
        PCS::batch_verify(
            &commitment_context.verifier_params,
            &comms,
            &points,
            &evaluations,
            proof.batch_proof(),
            transcript,
        )
        .map_err(|e| anyhow!("Error in PCS batch verification: {:?}", e))
    }
}
