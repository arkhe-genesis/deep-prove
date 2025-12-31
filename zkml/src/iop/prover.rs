use super::{ChallengeStorage, Proof, TableProof};
use crate::{
    Claim, Element, InitTranscript, Tensor,
    commit::{compute_betas_eval, mmcs_context, same_poly},
    graph::{
        Node, NodeId, NodeInput, NodeOutput, PortId,
        executor::{Executor, SequentialExecutor},
        scheduler::{GraphScheduler, IntoColor},
    },
    iop::{
        ChunkProof, ChunkProofData,
        chunking::{
            ChunkID, ChunkIOCommitments, ChunkingStrategy, DefaultChunkingStrategy, GroupIOClaims,
            GroupType, ModelChunk,
        },
        claim::PolynomialEvaluation,
        compute_claim,
        context::ProverContext,
        prover_graph::{LocalProverCtx, ProverGraph, ProverGraphIO, ProverGraphNode, SplitNode},
    },
    layers::{
        Layer, LayerProof,
        provable::{OpInfo, ProvableOp},
    },
    lookup::{
        context::{
            GenerateWitness, LookupContext, LookupWitness, TableType,
            generate_lookup_witness_for_chunk,
        },
        logup_gkr::prover::batch_multiple_sizes_prove,
    },
    model::{Model, Trace},
    quantization::ToField,
    tensor::{CommitmentId, get_root_of_unity},
};
use anyhow::{Context as _, Result, anyhow, bail, ensure};
use either::Either;
use ff_ext::ExtensionField;
use itertools::Itertools;
use mpcs::{Point, PolynomialCommitmentScheme};
use multilinear_extensions::{
    Expression,
    mle::{IntoMLE, MultilinearExtension},
    virtual_polys::VirtualPolynomialsBuilder,
};
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use std::collections::{BTreeMap, HashMap};
use sumcheck::{structs::IOPProverState, util::optimal_sumcheck_threads};
use timed::timed_instrument;
use tracing::{debug, info_span, trace};
use transcript::Transcript;
use utils::{Metrics, stream_metrics};

/// Prover generates a series of sumcheck proofs to prove the inference of a model
pub struct Prover<'a, 'b, E: ExtensionField, T: Transcript<E>, PCS: PolynomialCommitmentScheme<E>>
where
    PCS::CommitmentWithWitness: Serialize + DeserializeOwned,
    PCS::ProverParam: Send + Sync,
{
    ctx: &'a ProverContext<E, PCS>,
    // proofs for each layer being filled
    proofs: HashMap<NodeId, LayerProof<E, PCS>>,
    merge_claim_proofs: HashMap<NodeId, MergeClaimsProof<E>>,
    pub(crate) transcript: &'b mut T,
    /// Proves commitment openings
    pub(crate) commit_prover: mmcs_context::CommitmentProver<E, PCS>,
    /// The lookup witnesses
    pub(crate) lookup_witness: HashMap<NodeId, PCS::CommitmentWithWitness>,
    /// Stores all the challenges for the different lookup/table types
    pub(crate) challenge_storage: ChallengeStorage<E>,
}

pub struct BatchFFTProof<E: ExtensionField> {
    pub proof: sumcheck::structs::IOPProof<E>,
    pub claims: Vec<E>,
    pub point: Vec<E>,
    pub matrix_eval: (Vec<sumcheck::structs::IOPProof<E>>, Vec<Vec<E>>),
    pub delegation_points: Vec<Vec<E>>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound(serialize = "E: Serialize", deserialize = "E: DeserializeOwned"))]
pub(crate) struct MergeClaimsProof<E: ExtensionField> {
    // Map an output index for a given to a node to the proof for merging the claims
    // related to this output
    proofs: HashMap<usize, MergeClaimNodeProof<E>>,
}

impl<E: ExtensionField> MergeClaimsProof<E> {
    pub(crate) fn get_proof(&self, index: usize) -> Option<&MergeClaimNodeProof<E>> {
        self.proofs.get(&index)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound(serialize = "E: Serialize", deserialize = "E: DeserializeOwned"))]
pub(crate) struct MergeClaimNodeProof<E: ExtensionField> {
    proof: same_poly::Proof<E>,
    agg_claim: Claim<E>,
    num_vars: usize,
}

impl<E: ExtensionField> MergeClaimNodeProof<E> {
    pub(crate) fn generate_proof<T: Transcript<E>>(
        t: &mut T,
        claims: &[&Claim<E>],
        output: &Tensor<E>,
    ) -> anyhow::Result<MergeClaimNodeProof<E>> {
        let output_mle = output.clone().into_mle();
        let num_vars = output_mle.num_vars();
        let mut same_poly_prover = same_poly::Prover::new(output_mle);

        claims
            .iter()
            .try_for_each(|&claim| same_poly_prover.add_claim(claim.clone()))?;

        let (proof, claim) = same_poly_prover.prove(t)?;

        Ok(Self {
            proof,
            num_vars,
            agg_claim: claim,
        })
    }

    pub(crate) fn verify_proof<T: Transcript<E>>(
        &self,
        t: &mut T,
        claims: &[&Claim<E>],
    ) -> anyhow::Result<Claim<E>> {
        let ctx = same_poly::Context::new(self.num_vars);

        let mut verifier = same_poly::Verifier::new(&ctx);

        claims
            .iter()
            .try_for_each(|&claim| verifier.add_claim(claim.clone()))?;

        verifier.verify(&self.proof, t)
    }
}

pub(crate) type GenericModelLayers<M> = HashMap<NodeId, M>;
/// Type alias for the set of layers in a model, indexed by its `NodeId`
pub type ModelLayers = GenericModelLayers<Layer<Element>>;
/// Type alias for the set of references to layers of a model, indexed by its `NodeId`
pub type ModelLayersRef<'a> = GenericModelLayers<&'a Layer<Element>>;

impl<'a, 'b, E, T, PCS> Prover<'a, 'b, E, T, PCS>
where
    T: Transcript<E>,
    E: ExtensionField,
    PCS: PolynomialCommitmentScheme<E> + Send + Sync,
    PCS::CommitmentWithWitness: Serialize + DeserializeOwned + Send + Sync,
    PCS::ProverParam: Send + Sync,
{
    pub fn new(ctx: &'a ProverContext<E, PCS>, transcript: &'b mut T) -> Self {
        Self {
            ctx,
            transcript,
            proofs: Default::default(),
            merge_claim_proofs: Default::default(),
            commit_prover: mmcs_context::CommitmentProver::<E, PCS>::new(),
            lookup_witness: HashMap::default(),
            challenge_storage: ChallengeStorage::default(),
        }
    }

    pub(crate) fn add_common_claims(
        &mut self,
        node_id: NodeId,
        claims: HashMap<CommitmentId, Claim<E>>,
    ) {
        self.commit_prover.add_common_claims(
            claims
                .into_iter()
                .map(|(poly_id, claim)| (poly_id, vec![(node_id, claim)]))
                .collect(),
        )
    }

    pub(crate) fn add_table_claim(
        &mut self,
        table_type: &TableType,
        chunk_id: ChunkID,
        claim: Claim<E>,
    ) {
        let table_node_id = chunk_id.0.into();
        self.commit_prover
            .add_table_claim(table_node_id, table_type, claim);
    }

    pub(crate) fn add_witness_claim(&mut self, node_id: NodeId, claims: Vec<(Point<E>, Vec<E>)>) {
        self.commit_prover.add_witness_claim(node_id, claims);
    }

    pub(crate) fn lookup_witness(&self, id: NodeId) -> anyhow::Result<&PCS::CommitmentWithWitness> {
        self.lookup_witness
            .get(&id)
            .ok_or(anyhow!("No lookup witness found for node {id}!"))
    }

    pub(crate) fn push_proof(&mut self, node_id: NodeId, proof: LayerProof<E, PCS>) {
        self.proofs.insert(node_id, proof);
    }

    #[timed::timed_instrument(level = "debug")]
    fn prove_tables(
        &mut self,
        chunk_id: ChunkID,
        lookup_ctx: &LookupContext,
    ) -> anyhow::Result<Option<TableProof<E, PCS>>> {
        if lookup_ctx.is_empty() {
            Ok(None)
        } else {
            let table_node_id = self.ctx.commitment_ctx.table_node_id();
            let multiplicity_witness = self
                .lookup_witness(table_node_id)
                .context("No mutliplicity commitment found during table proving")?;
            let logup_inputs = lookup_ctx
                .create_logup_inputs::<PCS, E>(multiplicity_witness, &self.challenge_storage)?;
            let multiplicity_commit = PCS::get_pure_commitment(multiplicity_witness);
            // Run LogUp batch proving for all the tables at once
            let logup_batch_proof = batch_multiple_sizes_prove(&logup_inputs, self.transcript)?;

            // Now we takes the evals and append the correct values for commitment opening
            let all_claims = logup_batch_proof.output_claims();
            let (mul_claims, commit_claims) = lookup_ctx
                .iter()
                .scan(0, |skip, tt| {
                    let take = 1 + tt.num_columns();
                    let table_claims = &all_claims[*skip..*skip + take];
                    *skip += take;
                    Some((tt, table_claims))
                })
                .fold(
                    (vec![], vec![]),
                    |(mut acc, mut claims_acc), (tt, table_claims)| {
                        let mul_point = table_claims[0].point.clone();
                        let mul_eval = table_claims[0].eval;
                        if tt.has_committed_claims() {
                            claims_acc.push((tt, table_claims.last().unwrap().clone()));
                        }
                        acc.push((mul_point, mul_eval));
                        (acc, claims_acc)
                    },
                );

            commit_claims
                .into_iter()
                .for_each(|(tt, claim)| self.add_table_claim(tt, chunk_id, claim));
            let grouped = mul_claims
                .into_iter()
                .into_group_map()
                .into_iter()
                .sorted_by(|a, b| Ord::cmp(&b.0.len(), &a.0.len()))
                .collect::<Vec<(Point<E>, Vec<E>)>>();
            self.add_witness_claim(table_node_id, grouped);

            Ok(Some(TableProof {
                multiplicity_commit,
                lookup: logup_batch_proof,
            }))
        }
    }

    // Protocol for proving the correct computation of the FFT/iFFT matrix.
    // For more details look at the zkCNN paper.
    // F_middle : all intermediate evaluations retrieved by the phiGinit algorithm
    // r1: the initial random point used to reduce the matrix into vector
    // r2: the random point produced by the sumcheck
    #[allow(clippy::type_complexity)]
    pub fn delegate_matrix_evaluation(
        &mut self,
        f_middle: &mut [Vec<E>],
        r1: &[E],
        mut r2: Vec<E>,
        is_fft: bool,
    ) -> (
        Vec<sumcheck::structs::IOPProof<E>>,
        Vec<Vec<E>>,
        Vec<Vec<E>>,
    ) {
        let mut omegas = vec![E::ZERO; 1 << r1.len()];
        Self::phi_pow_init(&mut omegas, r1.len(), is_fft);

        let mut proofs: Vec<sumcheck::structs::IOPProof<E>> = Vec::new();
        let mut claims: Vec<Vec<E>> = Vec::new();
        let mut points: Vec<Vec<E>> = Vec::new();

        for l in (0..(r1.len() - 1)).rev() {
            let mut phi = vec![E::ZERO; f_middle[l].len()];
            let beta = compute_betas_eval(&r2[0..(r2.len() - 1)]);

            for i in 0..(phi.len()) {
                if !is_fft && l == f_middle.len() - 1 {
                    phi[i] = (E::ONE - r2[r2.len() - 1])
                        * (E::ONE - r1[(f_middle.len() - 1) - l]
                            + r1[(f_middle.len() - 1) - l]
                                * omegas[i << ((f_middle.len() - 1) - l)]);
                } else {
                    phi[i] = E::ONE - r1[(f_middle.len() - 1) - l]
                        + (E::ONE - E::from_canonical_u64(2) * r2[r2.len() - 1])
                            * r1[(f_middle.len() - 1) - l]
                            * omegas[i << ((f_middle.len() - 1) - l)];
                }
            }

            let f1 = beta.into_mle();
            let f2 = phi.into_mle();
            let num_vars = f1.num_vars();
            let num_threads = optimal_sumcheck_threads(num_vars);
            let f3 = MultilinearExtension::<E>::from_evaluations_ext_slice(num_vars, &f_middle[l]);
            let mut expr_builder = VirtualPolynomialsBuilder::<E>::new(num_threads, num_vars);
            let expr = [&f1, &f2, &f3]
                .into_iter()
                .fold(Expression::Constant(Either::Right(E::ONE)), |acc, p| {
                    acc * expr_builder.lift(Either::Left(p))
                });
            let virtual_poly = expr_builder.to_virtual_polys(&[expr], &[]);
            let (proof, state) = IOPProverState::<E>::prove(virtual_poly, self.transcript);

            let claim: Vec<E> = state.get_mle_flatten_final_evaluations();
            let point = state.collect_raw_challenges();
            r2 = point.clone();
            proofs.push(proof);
            claims.push(claim);
            points.push(point);
        }
        (proofs, claims, points)
    }

    // Compute powers of roots of unity
    pub fn phi_pow_init(phi_mul: &mut [E], n: usize, is_fft: bool) {
        let length = 1 << n;
        let rou: E = get_root_of_unity(n);

        let mut phi = rou;
        if is_fft {
            phi = phi.inverse();
        }
        phi_mul[0] = E::ONE;
        for i in 1..length {
            phi_mul[i] = phi_mul[i - 1] * phi;
        }
    }

    // Efficiently compute the omegas of FFT/iFFT matrix reduced at rx
    // This is a copy-paste implementation from zkCNN paper
    pub fn phi_g_init(
        phi_g: &mut [E],
        mid_phi_g: &mut [Vec<E>],
        rx: Vec<E>,
        scale: E,
        n: usize,
        is_fft: bool,
    ) {
        let mut phi_mul = vec![E::ZERO; 1 << n];
        Self::phi_pow_init(&mut phi_mul, n, is_fft);
        if is_fft {
            phi_g[0] = scale;
            phi_g[1] = scale;
            for i in 1..(n + 1) {
                for b in 0..(1 << (i - 1)) {
                    let l = b;
                    let r = b ^ (1 << (i - 1));
                    let m = n - i;
                    let tmp1 = E::ONE - rx[m];
                    let tmp2 = rx[m] * phi_mul[b << m];
                    phi_g[r] = phi_g[l] * (tmp1 - tmp2);
                    phi_g[l] *= tmp1 + tmp2;
                }
                if i < n {
                    mid_phi_g[i - 1] = vec![E::ZERO; 1 << (i)];
                    mid_phi_g[i - 1][..(1 << (i))].copy_from_slice(&phi_g[..(1 << (i))]);
                }
            }
        } else {
            phi_g[0] = scale;
            for i in 1..n {
                for b in 0..(1 << (i - 1)) {
                    let l = b;
                    let r = b ^ (1 << (i - 1));
                    let m = n - i;

                    let tmp1 = E::ONE - rx[m];
                    let tmp2 = rx[m] * phi_mul[b << m];

                    phi_g[r] = phi_g[l] * (tmp1 - tmp2);
                    phi_g[l] *= tmp1 + tmp2;
                }
                mid_phi_g[i - 1] = vec![E::ZERO; 1 << i];
                mid_phi_g[i - 1][..(1 << (i))].copy_from_slice(&phi_g[..(1 << (i))]);
            }
            for (b, item) in phi_mul.iter().enumerate().take(1 << (n - 1)) {
                let l = b;
                let tmp1 = E::ONE - rx[0];
                let tmp2 = rx[0] * *item;
                phi_g[l] *= tmp1 + tmp2;
            }
        }
    }
    // The prove_batch_fft and prove_batch_ifft are extensions of prove_fft and prove_ifft but in the batch setting.
    // Namely when we want to proof fft or ifft for MORE THAN ONE INSTANCES.
    // In particular, instead of proving y = Wx we want to prove Y = WX where Y,X are matrixes.
    // Following the matrix to matrix multiplication protocol, let y_eval = Y(r1,r2).
    // Then we want to prove a sumcheck instance of the form y_eval = sum_{i \in [n]}W(r1,i)X(i,r2).
    pub fn prove_batch_fft(&mut self, r: Vec<E>, x: &mut [Vec<E>]) -> BatchFFTProof<E> {
        let padded_rows = 2 * x[0].len();
        for item in x.iter_mut() {
            item.resize(padded_rows, E::ZERO);
        }
        // Partition r in (r1,r2)
        let mut r1 = vec![E::ZERO; x[0].len().ilog2() as usize];
        let mut r2 = vec![E::ZERO; x.len().ilog2() as usize];
        let r1_len = r1.len();
        r1.copy_from_slice(&r[..r1_len]);

        for i in 0..r2.len() {
            r2[i] = r[i + r1.len()];
        }
        // compute W(r1,i)
        let mut w_red: Vec<E> = vec![E::ZERO; x[0].len()];
        let mut f_middle: Vec<Vec<E>> = vec![Vec::new(); r1.len() - 1];
        Self::phi_g_init(
            &mut w_red,
            &mut f_middle,
            r1.clone(),
            E::ONE,
            x[0].len().ilog2() as usize,
            false,
        );
        // compute X(i,r2)

        let mut f_m = x.iter().flatten().cloned().collect::<Vec<_>>().into_mle();

        f_m.fix_high_variables_in_place(&r2);

        // Construct the virtual polynomial and run the sumcheck prover
        let f_red = w_red.into_mle();
        let num_vars = f_m.num_vars();
        let num_threads = optimal_sumcheck_threads(num_vars);
        let mut expr_builder = VirtualPolynomialsBuilder::<E>::new(num_threads, num_vars);
        let expr = expr_builder.lift(Either::Left(&f_m)) * expr_builder.lift(Either::Left(&f_red));
        let virtual_poly = expr_builder.to_virtual_polys(&[expr], &[]);
        let (proof, state) = IOPProverState::<E>::prove(virtual_poly, self.transcript);

        let claims = state.get_mle_flatten_final_evaluations();
        let out_point = state.collect_raw_challenges();
        let (matrix_proofs, matrix_claims, delegation_points) =
            self.delegate_matrix_evaluation(&mut f_middle, &r1, out_point.clone(), false);
        BatchFFTProof {
            proof,
            claims,
            point: out_point,
            matrix_eval: (matrix_proofs, matrix_claims),
            delegation_points,
        }
    }

    pub fn prove_batch_ifft(&mut self, r: Vec<E>, prod: &[Vec<E>]) -> Result<BatchFFTProof<E>> {
        let scale: E = E::from_canonical_u64(prod[0].len() as u64).inverse();

        // Partition r in (r1,r2)
        let mut r1 = vec![E::ZERO; prod[0].len().ilog2() as usize];
        let mut r2 = vec![E::ZERO; prod.len().ilog2() as usize];
        let r1_len = r1.len();
        r1.copy_from_slice(&r[..r1_len]);
        ensure!(
            r1[r1.len() - 1] == E::ZERO,
            "Error in randomness init batch ifft {:?}",
            r1[r1.len() - 1]
        );
        for i in 0..r2.len() {
            r2[i] = r[i + r1.len()];
        }
        // compute W(r1,i)
        let mut w_red: Vec<E> = vec![E::ZERO; prod[0].len()];
        let mut f_middle: Vec<Vec<E>> = vec![Vec::new(); r1.len() - 1];
        Self::phi_g_init(
            &mut w_red,
            &mut f_middle,
            r1.clone(),
            scale,
            prod[0].len().ilog2() as usize,
            true,
        );
        let f_red = w_red.into_mle();
        // compute X(i,r2)
        let mut f_m = prod
            .iter()
            .flatten()
            .cloned()
            .collect::<Vec<_>>()
            .into_mle();
        f_m.fix_high_variables_in_place(&r2);

        let num_vars = f_m.num_vars();
        let num_threads = optimal_sumcheck_threads(num_vars);
        let mut expr_builder = VirtualPolynomialsBuilder::<E>::new(num_threads, num_vars);
        let expr = expr_builder.lift(Either::Left(&f_m)) * expr_builder.lift(Either::Left(&f_red));
        let virtual_poly = expr_builder.to_virtual_polys(&[expr], &[]);
        let (proof, state) = IOPProverState::<E>::prove(virtual_poly, self.transcript);

        let claims = state.get_mle_flatten_final_evaluations();

        let out_point = state.collect_raw_challenges();
        let (proofs, matrix_claims, points) =
            self.delegate_matrix_evaluation(&mut f_middle, &r1, out_point.clone(), true);

        Ok(BatchFFTProof {
            proof,
            claims,
            point: out_point,
            matrix_eval: (proofs, matrix_claims),
            delegation_points: points,
        })
    }

    fn generate_chunk_commitments(
        &mut self,
        chunk: &ModelChunk,
        chunk_trace: &Trace<Element>,
        group_type: GroupType,
    ) -> anyhow::Result<BTreeMap<ChunkID, PCS::Commitment>> {
        let commitments = chunk.commitments(&self.ctx.commitment_ctx, chunk_trace, group_type)?;
        // add commitment witness to lookup_witness and convert them to pure commitments,
        // which are provided to the verifier
        let chunk_id = chunk.chunk_id;
        commitments
            .into_iter()
            .map(|(group_id, commitment)| {
                let comm = PCS::get_pure_commitment(&commitment);
                let commitment_id =
                    ModelChunk::compute_group_commitment_id(chunk_id, group_id, group_type);
                ensure!(
                    self.lookup_witness
                        .insert(commitment_id.into(), commitment)
                        .is_none(),
                    "Commitment already exists in lookup witness for id {commitment_id}"
                );
                Ok((group_id, comm))
            })
            .collect()
    }

    fn generate_chunk_io_commitments(
        &mut self,
        chunk: &ModelChunk,
        chunk_trace: &Trace<Element>,
    ) -> anyhow::Result<ChunkIOCommitments<PCS::Commitment>> {
        let input_commitments =
            self.generate_chunk_commitments(chunk, chunk_trace, GroupType::Incoming)?;
        let output_commitments =
            self.generate_chunk_commitments(chunk, chunk_trace, GroupType::Outgoing)?;
        let chunk_commitments = ChunkIOCommitments {
            inputs: input_commitments,
            outputs: output_commitments,
        };
        chunk_commitments.add_to_transcript::<E, PCS, T>(chunk.chunk_id, self.transcript)?;
        Ok(chunk_commitments)
    }

    pub(crate) fn initialise_transcript(ctx: &ProverContext<E, PCS>) -> anyhow::Result<T>
    where
        T: InitTranscript,
    {
        let mut transcript = T::new(T::InitData::from(b"model_proving"));
        ctx.write_to_transcript(&mut transcript)?;
        Ok(transcript)
    }

    pub(crate) fn prove_chunk<'d>(
        mut self,
        chunk: ModelChunk,
        chunk_trace: &Trace<Element>,
        chunk_layers: &ModelLayersRef<'d>,
    ) -> anyhow::Result<ChunkProof<E, PCS>> {
        let chunk_id = chunk.chunk_id;
        // add chunk splitting info to the transcript
        chunk.add_chunk_data_to_transcript(self.transcript)?;

        let lookup_ctx = chunk.chunk_lookup_ctx(&self.ctx.lookup);

        debug!("== Instantiate witness context ==");
        let metrics = Metrics::new();

        // NOTE: until https://github.com/Plonky3/Plonky3/pull/999 is fixed, we have
        // to use the sequential executor and not the threadpool executor.
        self.instantiate_witness_ctx_for_chunk::<SequentialExecutor>(
            &chunk,
            chunk_trace,
            &lookup_ctx,
            chunk_layers,
            &(),
        )?;

        // we need to compute commitments of the input/output edges of each chunk and add them
        // to the transcript.
        let chunk_commitments = self.generate_chunk_io_commitments(&chunk, chunk_trace)?;

        let span = metrics.to_span();
        stream_metrics("Witness context", &span);
        debug!("== Witness context metrics {} ==", span);

        debug!("== Challenge storage ==");
        let metrics = Metrics::new();
        // initialize challenge storgae for this chunk
        self.challenge_storage = if lookup_ctx.is_empty() {
            ChallengeStorage::<E>::default()
        } else {
            ChallengeStorage::<E>::initialise(&lookup_ctx, self.transcript)
        };
        debug!("== Challenge storage metrics {} ==", metrics.to_span());

        debug!("== Generating claims ==");
        let metrics = Metrics::new();

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
                let source_id = output_edge.source();
                let trace_step = chunk_trace.get_step(&source_id)
                    .ok_or(
                        anyhow!("Node {source_id} not found in trace for chunk {chunk_id}")
                    )?;
                ensure!(
                    output_edge.ports().len() == 1,
                    "Expected 1 port link for model output edge {edge_id} in chunk {chunk_id}, found {}",
                    output_edge.ports().len()
                );
                let source_port = &output_edge.ports()[0].source_port;
                let output_tensor ={
                    let output_tensor_guard = trace_step.output_tensor_at(
                        **source_port,
                    )?;
                    output_tensor_guard.to_field()
                };
                ensure!(
                    outputs.insert(output_id, output_tensor).is_none(),
                    "Found output tensor twice for output id {} in chunk {chunk_id}",
                    output_id,
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

        // `chunk_output_claims` is a map storing claims related to the subset of input ports of layers in the model
        // which are connected to an output port of a node found in the current chunk. Here, we initialize
        // this map by claims about the input ports of layers that don't belong to this chunk, i.e., layers
        // of other chunks that use the outputs produced by layers in the current chunk
        let chunk_output_claims = chunk.outgoing_edges()?
            .into_iter()
            .try_fold(BTreeMap::new(), // we first collect all the tensors, sorted by the corresponding output port
            |mut claims_map, edge_id| {
                let edge = chunk.edge(&edge_id)?;
                let source_node_id = edge.source();
                let trace_step = chunk_trace.get_step(&source_node_id)
                    .ok_or(
                        anyhow!("Trace step not found for node {source_node_id} in chunk {}", chunk.chunk_id)
                    )?;
                edge.ports().iter().try_for_each(|port| {
                    let source_port = NodeOutput::new(*source_node_id, port.source_port);
                    if let std::collections::btree_map::Entry::Vacant(e) = claims_map.entry(source_port) {
                         // compute new claim and insert in cache
                         let output_tensor = trace_step.output_tensor_at(
                             port.source_port.into(),
                         )?;
                         e.insert(output_tensor);
                    }
                    anyhow::Ok(())
                })?;
                anyhow::Ok(claims_map)
            })?
            .into_iter() // then, we compute the claims for each output
            .map(|(port, tensor)| {
                let claim = compute_claim(self.transcript, tensor.to_field());
                (port, claim)
        }).collect();
        // each layer generates claims about its inputs. Each claim is indexed by
        // the id of the corresponding "input port" of the node, e.g. target_port when
        // considering incoming edges to this node.
        let mut claims: HashMap<NodeInput, Claim<E>> = HashMap::new();
        for (node_id, node) in chunk.subgraph.backward_iter() {
            match node {
                Node::Inner(_) => {
                    let section = chunk_trace
                        .get_step(&node_id)
                        .ok_or(anyhow!("Step in trace not found for node {node_id}"))?;
                    let op = chunk_layers
                        .get(&node_id)
                        .ok_or(anyhow!("Node {node_id} not found in model"))?;
                    trace!("Proving node with id {node_id}: {:?}", op.describe());

                    // Load all output tensors and convert to target field
                    let handles = section
                        .node_outputs
                        .outputs
                        .iter()
                        .map(|handle| {
                            handle
                                .tensor()
                                .with_context(|| {
                                    format!("hydrating tensor {}", handle.storage_key())
                                })
                                .map(|tensor_guard| tensor_guard.to_field())
                        })
                        .collect::<anyhow::Result<Vec<_>>>()?;

                    // The claims for this node, i.e. the claims stemming from
                    // the input nodes of the successor nodes connected to this
                    // nodes output nodes, are collected and ordered by output
                    // port (on this node) number. Remember that the graph is
                    // traversed backwards, so output nodes are conceptually
                    // inputs, and vice-versa.
                    let claims_for_node =
                        chunk.claims_for_node(node_id, &claims, &chunk_output_claims)?;

                    // Just like for verification, there might be claims to be
                    // merged if they are connected to the same output port for
                    // this node.
                    let claims_for_prove = self.flatten_and_merge_claims(
                        claims_for_node,
                        &handles.iter().collect::<Vec<_>>(),
                        node_id,
                    )?;

                    // prove or propagate the claims
                    let ctx = self
                        .ctx
                        .model_ctx
                        .nodes
                        .node(node_id)
                        .ok_or(anyhow!("Node {node_id} not found in proving context"))?
                        .as_inner()
                        .ok_or(anyhow!(
                            "Node {node_id} is not an inner node in proving context"
                        ))?;
                    let my_claims = if op.is_provable() {
                        op.prove(
                            node_id,
                            ctx,
                            claims_for_prove.iter().collect::<Vec<_>>(),
                            section,
                            &mut self,
                        )
                        .with_context(|| format!("proving {}: {}", node_id, op.describe()))?
                    } else {
                        // we only propagate the claims, without changing them, as a non-provable layer
                        // shouldn't change the input values
                        claims_for_prove
                    };

                    // Update the claim register with the input claims for this
                    // node, that will become the data from which its
                    // topological predecessors (but traversal successor,
                    // remember the backward traversal) input claims will in
                    // turn be derived.
                    claims.extend(
                        my_claims
                            .into_iter()
                            .enumerate()
                            .map(|(i, claim)| (NodeInput::new(node_id, i), claim)),
                    );
                }
                Node::Input(_) => {}
                Node::Output(o) => {
                    // Seed the claim register.
                    claims.insert(NodeInput::new(node_id, 0), output_claims_by_port[o].clone());
                }
            }
        }

        let span = metrics.to_span();
        stream_metrics("Claims", &span);
        debug!("== Claims generation metrics {} ==", span);

        // Now we need add the claims about the input and output of the chunk
        chunk.outgoing_edges.keys().try_for_each(|group_id| {
            let GroupIOClaims {
                commitment_id,
                claims: group_claims,
            } = chunk.compute_outgoing_group_claims(group_id, &chunk_output_claims)?;
            self.add_witness_claim(
                commitment_id,
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
            self.add_witness_claim(
                commitment_id,
                group_claims
                    .into_iter()
                    .map(|c| (c.point, vec![c.eval]))
                    .collect(),
            );
            anyhow::Ok(())
        })?;

        // Now we have to make the table proofs
        debug!("== Generating Lookup Table claims ==");
        let metrics = Metrics::new();
        let table_proof = self.prove_tables(chunk_id, &lookup_ctx)?;
        let span = metrics.to_span();
        stream_metrics("Tables", &span);
        debug!("== Lookup Table claims generation metrics {} ==", span);

        debug!("== Generate proof ==");
        let metrics = Metrics::new();

        let commit_proof = self.commit_prover.prove(
            &self.ctx.commitment_ctx,
            &self.lookup_witness,
            self.transcript,
        )?;
        let chunk_data = ChunkProofData {
            output_evals: chunk_output_claims
                .into_iter()
                .map(|(port, claim)| {
                    (
                        port,
                        PolynomialEvaluation {
                            num_vars: claim.point.len(),
                            eval: claim.eval,
                        },
                    )
                })
                .collect(),
            commitments: chunk_commitments,
            model_chunk: chunk,
        };

        let chunk_proof = ChunkProof {
            steps: self.proofs,
            merge_claim_proofs: self.merge_claim_proofs,
            table_proof,
            commit: commit_proof,
            chunk_data,
        };

        let span = metrics.to_span();
        stream_metrics("Proof", &span);
        debug!("== Generate proof metrics {} ==", span);

        Ok(chunk_proof)
    }

    /// Build the execution graph to run the proving of chunks `chunks`.
    /// It currently assigns one node per chunk, and the first node, with id 0,
    /// is assigned as a coordinator, that starts the process (e.g. executes the first task)
    /// and finishes it (outputs the final proof).
    #[allow(clippy::type_complexity)]
    pub(crate) fn build_execution_graph<'c>(
        chunks: Vec<ModelChunk>,
    ) -> anyhow::Result<ProverGraph<'a, 'c, E, T, PCS>>
    where
        T: InitTranscript,
        PCS: 'static,
    {
        // add the input node of the graph, which is doing the preprocessing task, and the output node
        // of the graph, which is doing the opening of the model claims
        let mut exec_graph = ProverGraph::new();
        let num_chunks = chunks.len();
        let init_node_id = exec_graph
            .add_inner(ProverGraphNode::ProverSplit(SplitNode::new(chunks)).colored(0))?;
        let final_node_id = exec_graph.add_inner(ProverGraphNode::Final.colored(0))?;
        // add one node in the graph for each chunk to be proven
        for i in 0..num_chunks {
            let color = i + 1;
            let node_id = exec_graph.add_inner(ProverGraphNode::ChunkProver(i).colored(color))?;
            // link initial node and final node to the current chunk node
            exec_graph.add_edge(init_node_id, node_id, (i, 0), None)?;
            exec_graph.add_edge(node_id, final_node_id, (0, i), None)?;
        }
        Ok(exec_graph)
    }

    /// Return the inputs to be provided to the execution graph `graph`
    pub(crate) fn graph_inputs(
        full_trace: Trace<Element>,
        graph: &ProverGraph<E, T, PCS>,
    ) -> anyhow::Result<HashMap<NodeInput, ProverGraphIO<E, PCS>>>
    where
        T: InitTranscript,
        PCS: 'static,
    {
        let source_node = graph.source_nodes().exactly_one().map_err(|e| {
            anyhow!(
                "Expected 1 source node for execution graph, found {}",
                e.count()
            )
        })?;
        Ok([(
            NodeInput::new(source_node, 0),
            ProverGraphIO::ProverSplitInput(full_trace),
        )]
        .into())
    }

    fn run_execution_graph<'c, Ex>(
        exec_graph: ProverGraph<'a, 'c, E, T, PCS>,
        inputs: HashMap<NodeInput, ProverGraphIO<E, PCS>>,
        context: &LocalProverCtx<'a, 'c, E, PCS>,
        config: Ex::Config,
    ) -> anyhow::Result<Proof<E, PCS>>
    where
        T: InitTranscript,
        Ex: Executor<ProverGraphNode<'a, 'c, E, T, PCS>, usize>,
        PCS: 'static,
    {
        let scheduler = GraphScheduler::new(exec_graph);
        Ex::run(&config, scheduler, inputs, context)?
            .into_values()
            .exactly_one()
            .map_err(|e| {
                anyhow!(
                    "Expected one output after running the graph, found {}",
                    e.count()
                )
            })
            .and_then(|output| {
                if let ProverGraphIO::FinalProof(proof) = output {
                    Ok(proof)
                } else {
                    bail!("Expected final proof as output of execution graph")
                }
            })
    }

    /// Prove by splitting the proving computation into multiple chunks, employing the `ChunkingStrategy`
    /// provided as input to build the chunks. The number of chunks can be specified as input, otherwise
    /// the chunking strategy will decide how many chunks to build.
    pub fn chunked_prove_local<
        'd,
        S: ChunkingStrategy,
        Ex: Executor<ProverGraphNode<'a, 'd, E, T, PCS>, usize>,
    >(
        ctx: &'a ProverContext<E, PCS>,
        full_trace: Trace<Element>,
        num_chunks: Option<usize>,
        chunking_strategy: S,
        model: &'d Model<Element>,
        executor_conf: Ex::Config,
    ) -> anyhow::Result<Proof<E, PCS>>
    where
        T: InitTranscript,
        PCS: 'static,
    {
        // we can go deeper in the span tree to trace each step of chunked proving
        let span = info_span!(
            "zkml_prove_chunked",
            chunks = num_chunks.unwrap_or_default()
        );
        let _guard = span.enter();
        // split in chunks
        let chunks = ctx
            .model_ctx
            .split_in_chunks(num_chunks, &chunking_strategy)?;

        let global_metrics = Metrics::new();

        let output_proof = {
            // build the computational graph to prove chunks
            let graph = Self::build_execution_graph(chunks)?;
            let inputs = Self::graph_inputs(full_trace, &graph)?;
            let context = LocalProverCtx::new(ctx, model);
            Self::run_execution_graph::<Ex>(graph, inputs, &context, executor_conf)?
        };

        let global_metrics_span = global_metrics.to_span();
        stream_metrics("Global", &global_metrics_span);
        debug!("== Global metrics {} ==", global_metrics_span);
        Ok(output_proof)
    }

    pub fn prove<'d>(
        ctx: &'a ProverContext<E, PCS>,
        full_trace: Trace<Element>,
        model: &'d Model<Element>,
    ) -> anyhow::Result<Proof<E, PCS>>
    where
        T: InitTranscript,
        PCS: 'static,
    {
        let span = info_span!("zkml_prove");
        let _guard = span.enter();
        Self::chunked_prove_local::<_, SequentialExecutor>(
            ctx,
            full_trace,
            Some(1),
            DefaultChunkingStrategy::default(),
            model,
            (),
        )
    }

    /// Flattens all the claims to give to the proving logic of the node. If
    /// there are claims linked to the same port, the claims will be merged.
    fn flatten_and_merge_claims(
        &mut self,
        claims: BTreeMap<PortId, Vec<&Claim<E>>>,
        outputs: &[&Tensor<E>],
        node_id: NodeId,
    ) -> anyhow::Result<Vec<Claim<E>>> {
        let mut merge_claim_proofs = HashMap::new();
        ensure!(
            claims.len() == outputs.len(),
            "Number of claims and outputs is not the same for node {node_id}: {} vs {}",
            claims.len(),
            outputs.len(),
        );
        let claims = claims
            .into_iter()
            .map(|(port, mut claims)| {
                let output = outputs[*port];
                if claims.len() == 1 {
                    // there is already only one claim, so we return it
                    Ok(claims.remove(0).clone())
                } else {
                    // we have to merge the claims
                    let (merged_claim, proof) = self.merge_claims(&claims, output)?;
                    merge_claim_proofs.insert(*port, proof);
                    Ok(merged_claim)
                }
            })
            .collect::<anyhow::Result<Vec<_>>>()?;

        self.merge_claim_proofs.insert(
            node_id,
            MergeClaimsProof {
                proofs: merge_claim_proofs,
            },
        );

        Ok(claims)
    }

    fn merge_claims(
        &mut self,
        claims: &[&Claim<E>],
        output: &Tensor<E>,
    ) -> anyhow::Result<(Claim<E>, MergeClaimNodeProof<E>)> {
        let proof = MergeClaimNodeProof::generate_proof(self.transcript, claims, output)?;
        Ok((proof.agg_claim.clone(), proof))
    }

    /// Looks at all the individual polys to accumulate from the witnesses and create the context from that.
    #[timed_instrument]
    fn instantiate_witness_ctx_for_chunk<
        'c,
        'd,
        Ex: Executor<GenerateWitness<'c, 'a, 'd, E, PCS>, usize>,
    >(
        &mut self,
        chunk: &ModelChunk,
        chunk_trace: &'c Trace<Element>,
        lookup_ctx: &LookupContext,
        chunk_layers: &'d ModelLayersRef<'d>,
        executor_config: &Ex::Config,
    ) -> anyhow::Result<()> {
        let LookupWitness {
            logup_witnesses,
            table_witnesses,
        } = generate_lookup_witness_for_chunk::<E, T, PCS, _, Ex>(
            &chunk.subgraph,
            lookup_ctx,
            chunk_trace,
            self.ctx,
            self.transcript,
            chunk_layers,
            executor_config,
        )?;
        self.lookup_witness = logup_witnesses;

        if let Some(commit) = table_witnesses {
            let table_node_id = self.ctx.commitment_ctx.table_node_id();
            self.lookup_witness.insert(table_node_id, commit);
        }
        Ok(())
    }
}
