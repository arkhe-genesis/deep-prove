use anyhow::{Context, Result};
use itertools::Itertools;
use lazy_static::lazy_static;
use parking_lot::RwLock;
use std::collections::{BTreeMap, HashMap};

use ark_ff::PrimeField;
use ark_std::rand::Rng;
use dp_crypto::{
    ArcMultilinearExtension,
    arkyper::{
        CommitmentScheme,
        transcript::{AppendToTranscript, Transcript},
    },
    poly::dense::DensePolynomial,
};
use serde::{Deserialize, Serialize};
use tracing::debug;
use utils::Metrics;

use crate::{
    graph::NodeId,
    lookup::table::Table,
    poly_commit::{
        ChunkedCommitment, num_vars_for_chunk, table_poly_id, verifier::VerifierCommitment,
    },
    tensor::CommitmentId,
};

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct CommittedPolynomial<'a, F: PrimeField, PCS: CommitmentScheme> {
    pub(super) chunk_commitments: Vec<PCS::Commitment>,
    pub(crate) polynomial: ArcMultilinearExtension<'a, F>,
}

impl<'a, F: PrimeField, PCS: CommitmentScheme> ChunkedCommitment
    for CommittedPolynomial<'a, F, PCS>
{
    fn num_chunks(&self) -> usize {
        self.chunk_commitments.len()
    }

    fn num_vars(&self) -> usize {
        self.polynomial.num_vars()
    }
}

impl<'a, F: PrimeField, PCS: CommitmentScheme> CommittedPolynomial<'a, F, PCS> {
    pub fn polynomial(&self) -> &DensePolynomial<'a, F> {
        self.polynomial.as_ref()
    }

    pub(crate) fn chunked_polys(&self) -> Vec<DensePolynomial<'_, F>> {
        self.polynomial.as_view_chunks(self.num_chunks())
    }

    pub(crate) fn batch_commit(
        prover_params: &PCS::ProverSetup,
        polys: Vec<DensePolynomial<'a, F>>,
    ) -> anyhow::Result<Vec<Self>>
    where
        PCS: CommitmentScheme<Field = F>,
    {
        let num_chunks_per_poly = polys
            .iter()
            .map(|poly| num_chunks_for_polynomial(poly.num_vars()))
            .collect_vec();
        let chunked_mles = polys
            .iter()
            .zip(&num_chunks_per_poly)
            .flat_map(|(poly, &num_chunks)| poly.as_view_chunks(num_chunks))
            .collect_vec();
        let mut commitments_iter = PCS::batch_commit(prover_params, &chunked_mles)?.into_iter();
        Ok(polys
            .into_iter()
            .zip(num_chunks_per_poly)
            .map(|(poly, num_chunks)| {
                let comms = (0..num_chunks)
                    .map_while(|_| commitments_iter.next().map(|(comm, _)| comm))
                    .collect();
                CommittedPolynomial {
                    chunk_commitments: comms,
                    polynomial: poly.into(),
                }
            })
            .collect())
    }
}

impl<'a, F: PrimeField, PCS: CommitmentScheme> AppendToTranscript
    for CommittedPolynomial<'a, F, PCS>
{
    fn append_to_transcript<ProofTranscript: Transcript>(&self, transcript: &mut ProofTranscript) {
        self.chunk_commitments
            .iter()
            .for_each(|comm| comm.append_to_transcript(transcript))
    }
}

// This is the threshold we employ to split committed polynomials into multiple chunks;
// any polynomial with more variables than this is split into chunks
const DEFAULT_CHUNK_SIZE_THRESHOLD: usize = 20;

lazy_static! {
    pub(crate) static ref CHUNK_SIZE_THRESHOLD: RwLock<Option<usize>> = RwLock::new(None);
}
// This is the maximum number of chunks we allow for each polynomial, to ensure that
// the verifier work doesn't get too intensive. Indeed, the verifier work for each committed
// polynomial is proportional to the number of chunks. Must be a power of 2
const MAX_CHUNKS: usize = 128;

/// Compute the number of chunks a polynomial with `poly_num_vars` variables is split into
/// in order to be committed
fn num_chunks_for_polynomial(poly_num_vars: usize) -> usize {
    let chunk_size_threshold = CHUNK_SIZE_THRESHOLD
        .read_recursive()
        .unwrap_or(DEFAULT_CHUNK_SIZE_THRESHOLD);
    let num_chunks = (1usize << poly_num_vars).div_ceil(1usize << chunk_size_threshold);
    // we need to cap the number of chunks to `MAX_CHUNKS`
    num_chunks.min(MAX_CHUNKS)
}

#[derive(Debug, Serialize, Deserialize)]
/// Struct that contains all the data needed for proving/verifying commitments relating to a model.
pub struct GlobalCommitmentContext<'a, F, PCS>
where
    F: PrimeField,
    PCS: CommitmentScheme,
{
    /// Prover parameters for the [`PolynomialCommitmentScheme`]
    prover_params: PCS::ProverSetup,
    /// Verifier parameters for the [`PolynomialCommitmentScheme`]
    verifier_params: PCS::VerifierSetup,
    /// Commitment to the model static polynomials.
    model_commitments: BTreeMap<CommitmentId, CommittedPolynomial<'a, F, PCS>>,
    /// This is the [`NodeId`] used for tables in this model
    table_node_id: NodeId,
    /// This is the largest number of variables of any of the polynomials in `model_commitment`
    max_model_num_vars: usize,
}

impl<'a, F, PCS> GlobalCommitmentContext<'a, F, PCS>
where
    PCS: CommitmentScheme<Field = F>,
    F: PrimeField,
{
    /// Make a new [`GlobalCommitmentContext`]
    pub fn new<R: Rng>(
        witness_poly_num_vars: usize,
        polys: HashMap<CommitmentId, DensePolynomial<'a, F>>,
        lookup_ctx: &[&Table],
        max_node_id: NodeId,
        rng: &mut R,
    ) -> Result<GlobalCommitmentContext<'a, F, PCS>> {
        // Find the maximum size so we can generate params
        let max_num_vars = polys.iter().fold(witness_poly_num_vars, |acc, (_, poly)| {
            acc.max(poly.num_vars())
        });

        let pcs_max_num_vars =
            num_vars_for_chunk(max_num_vars, num_chunks_for_polynomial(max_num_vars));

        debug!("Building PCS params for {pcs_max_num_vars} max variables...");
        let m = Metrics::new();
        let (prover_params, verifier_params) = PCS::test_setup(rng, pcs_max_num_vars);
        debug!("{} PPs & VPs built", m.to_span());

        // Find the maximum node id used in this model so we can pick a unique node id for table related commitments.
        let table_node_id = NodeId::from(max_node_id.0 + 1);

        // First we take all the model polys and sort them by the number of variables they have.
        // Then we do the same for any table commitments but here we set all of them to have `table_node_id`.
        let table_commitments_check = lookup_ctx.iter().any(|table| table.commit_output_column());
        let model_commitments = if !polys.is_empty() || table_commitments_check {
            let (model_poly_ids, model_polys): (Vec<_>, Vec<_>) = polys
                .into_iter()
                .chain(lookup_ctx.iter().filter_map(|table| {
                    table
                        .committed_columns::<F>()
                        .map(|mle| (table_poly_id(table.name()), mle))
                }))
                .unzip();
            // let (model_comms_map, model_polys) = polys
            // .into_iter()
            // .map(|(poly_id, poly)| (poly.num_vars(), (poly_id, poly)))
            // .chain(lookup_ctx.iter().filter_map(|table_type| {
            // table_type
            // .committed_columns()
            // .map(|mle| (mle.num_vars(), (table_poly_id(table_type.name()), mle)))
            // }))
            // .fold(
            // (BTreeMap::new(), Vec::new()),
            // |(mut map_acc, mut model_polys), (num_vars, (poly_id, mle))| {
            // map_acc
            // .entry(num_vars)
            // .or_insert(Vec::new())
            // .push(poly_id.clone());
            // model_polys.push(PolynomialWithId {
            // id: poly_id,
            // polynomial: mle,
            // });
            // (map_acc, model_polys)
            // },
            // );
            // debug!("{} model_comms_map built", m.to_span());

            // Commit to model_polys
            let m = Metrics::new();
            let model_commitments = CommittedPolynomial::batch_commit(&prover_params, model_polys)
                .context("Model commitment for poly {poly_id}")?
                .into_iter()
                .zip(model_poly_ids)
                .map(|(commitment, poly_id)| (poly_id, commitment))
                .collect();
            debug!("{} model commitments built", m.to_span());
            model_commitments
        } else {
            BTreeMap::new()
        };

        let max_model_num_vars = model_commitments
            .values()
            .map(|poly| poly.polynomial.num_vars())
            .max()
            .unwrap_or(0usize);
        Ok(GlobalCommitmentContext {
            verifier_params,
            prover_params,
            model_commitments,
            table_node_id,
            max_model_num_vars,
        })
    }

    /// Generate a prover/verifier context for the `witness_poly_size` specified as input;
    /// `witness_poly_size` represents the size of the biggest witness polynomial to be
    /// committed to. If no `witness_poly_size` is provided as input, this method generates
    /// the prover/verifier context for `self.max_poly_size()`
    #[allow(dead_code)]
    pub(crate) fn generate_contexts(
        self,
    ) -> Result<(CommitmentProverCtx<'a, F, PCS>, CommitmentVerifierCtx<PCS>)> {
        let GlobalCommitmentContext {
            prover_params,
            verifier_params,
            model_commitments,
            table_node_id,
            max_model_num_vars,
            ..
        } = self;

        let verifier_ctx = CommitmentVerifierCtx {
            verifier_params,
            model_commitments: model_commitments
                .iter()
                .map(|(poly_id, committed_poly)| {
                    (poly_id.clone(), VerifierCommitment::from(committed_poly))
                })
                .collect(),
            table_node_id,
            max_model_num_vars,
        };

        let prover_ctx = CommitmentProverCtx {
            prover_params,
            model_commitments,
            table_node_id,
            max_model_num_vars,
        };

        Ok((prover_ctx, verifier_ctx))
    }
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(bound(
    serialize = "F: ark_serialize::CanonicalSerialize",
    deserialize = "F: ark_serialize::CanonicalDeserialize"
))]
/// Context data for the commitment prover
pub struct CommitmentProverCtx<'a, F, PCS>
where
    F: PrimeField,
    PCS: CommitmentScheme,
{
    /// Prover parameters for the [`PolynomialCommitmentScheme`]
    pub(super) prover_params: PCS::ProverSetup,
    /// Commitment to the model static polynomials.
    pub(super) model_commitments: BTreeMap<CommitmentId, CommittedPolynomial<'a, F, PCS>>,
    /// This is the [`NodeId`] used for tables in this model
    table_node_id: NodeId,
    /// This is the largest number of variables of any of the polynomials in `model_commitment`
    pub(super) max_model_num_vars: usize,
}

impl<'a, F: PrimeField, PCS: CommitmentScheme<Field = F>> CommitmentProverCtx<'a, F, PCS> {
    /// Helper method to commit to polynomial.
    pub fn commit(
        &self,
        poly: DensePolynomial<'a, F>,
    ) -> anyhow::Result<CommittedPolynomial<'a, F, PCS>> {
        let num_chunks = num_chunks_for_polynomial(poly.num_vars());
        if num_chunks == 1 {
            PCS::commit(&self.prover_params, &poly).map(|(commitment, _)| CommittedPolynomial {
                chunk_commitments: vec![commitment],
                polynomial: poly.into(),
            })
        } else {
            let chunked_poly = poly.as_view_chunks(num_chunks);
            PCS::batch_commit(&self.prover_params, &chunked_poly).map(|chunk_commitments| {
                CommittedPolynomial {
                    chunk_commitments: chunk_commitments
                        .into_iter()
                        .map(|(comm, _)| comm)
                        .collect(),
                    polynomial: poly.into(),
                }
            })
        }
    }

    /// Helper method to commit to a set of polynomials.
    /// It returns one commitment for each polynomial provided as input
    pub fn batch_commit<'b>(
        &self,
        polys: Vec<DensePolynomial<'b, F>>,
    ) -> anyhow::Result<Vec<CommittedPolynomial<'b, F, PCS>>> {
        CommittedPolynomial::batch_commit(&self.prover_params, polys)
    }

    /// Write the commitment context to the transcript
    pub fn write_to_transcript<T: Transcript>(&self, transcript: &mut T) {
        self.model_commitments
            .iter()
            .for_each(|(_, committed_poly)| {
                committed_poly
                    .chunk_commitments
                    .iter()
                    .for_each(|comm| comm.append_to_transcript(transcript))
            })
    }

    pub fn table_node_id(&self) -> NodeId {
        self.table_node_id
    }
}

/// Context data for the commitment verifier
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct CommitmentVerifierCtx<PCS>
where
    PCS: CommitmentScheme,
{
    /// Verifier parameters for the [`PolynomialCommitmentScheme`]
    pub(super) verifier_params: PCS::VerifierSetup,
    /// Commitment to the model static polynomials.
    pub(super) model_commitments: BTreeMap<CommitmentId, VerifierCommitment<PCS>>,
    /// This is the [`NodeId`] used for tables in this model
    table_node_id: NodeId,
    /// This is the largest number of variables of any of the polynomials in `model_commitment`
    max_model_num_vars: usize,
}

impl<PCS> CommitmentVerifierCtx<PCS>
where
    PCS: CommitmentScheme,
{
    pub fn write_to_transcript<T: Transcript>(&self, transcript: &mut T) {
        self.model_commitments
            .iter()
            .for_each(|(_, commitment)| commitment.append_to_transcript(transcript))
    }

    pub fn table_node_id(&self) -> NodeId {
        self.table_node_id
    }
}
