use super::{ChallengeStorage, Proof, TableProof};
use crate::{
    Claim, Element, IO, InitTranscript, Tensor, VectorTranscript, get_root_of_unity,
    graph::{
        Node, NodeId, NodeInput, NodeOutput, PortId,
        executor::{Executor, SequentialExecutor},
        scheduler::{GraphScheduler, IntoColor},
    },
    iop::{
        ChunkProof, ChunkProofData,
        chunking::{
            ChunkID, ChunkIOCommitments, ChunkedNode, ChunkedOutput, ChunkingStrategy,
            DefaultChunkingStrategy, ModelChunk,
        },
        claim::PolynomialEvaluation,
        compute_claim,
        context::ProverContext,
        prover_graph::{LocalProverCtx, ProverGraph, ProverGraphIO, ProverGraphNode, SplitNode},
        same_poly,
    },
    layers::{
        Layer, LayerCtx, LayerProof,
        provable::{OpInfo, ProvableOp},
    },
    lookup::{
        context::{LookupContext, LookupWitness, generate_lookup_witness_for_chunk},
        logup_gkr::prover::new_batch_multiple_sizes_prove,
        table::Table,
    },
    measure::{self, LAYER_WISE_MEASURE_PREFIX},
    model::{Model, Trace},
    poly_commit::{
        context::CommittedPolynomial, prover::CommitmentProver, verifier::VerifierCommitment,
    },
    quantization::ToField,
    tensor::CommitmentId,
};
use anyhow::{Context as _, Result, anyhow, bail, ensure};
use ark_ff::PrimeField;
use dp_crypto::{
    Expression, IntoMLE,
    arkyper::{CommitmentScheme, transcript::Transcript},
    poly::{dense::DensePolynomial, eq::evals, slice::SmartSlice},
    structs::{IOPProof, IOPProverState},
    util::optimal_sumcheck_threads,
    virtual_polys::VirtualPolynomialsBuilder,
};
use either::Either;
use itertools::Itertools;
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, HashMap},
    time::Instant,
};
use timed::timed_instrument;
use tracing::{debug, info_span, trace};
use utils::Metrics;

/// Prover generates a series of sumcheck proofs to prove the inference of a model
pub struct Prover<'a, 'b, F: PrimeField, T: Transcript, PCS: CommitmentScheme> {
    ctx: &'a ProverContext<'a, F, PCS>,
    // proofs for each layer being filled
    proofs: HashMap<NodeId, LayerProof<F, PCS>>,
    merge_claim_proofs: HashMap<NodeId, MergeClaimsProof<F>>,
    pub(crate) transcript: &'b mut T,
    /// Proves commitment openings
    pub(crate) commit_prover: CommitmentProver<F, PCS>,
    /// The lookup witnesses
    pub(crate) lookup_witness: HashMap<NodeId, Vec<CommittedPolynomial<'a, F, PCS>>>,
    /// Stores all the challenges for the different lookup/table types
    pub(crate) challenge_storage: ChallengeStorage<F>,
}

pub struct BatchFFTProof<F: PrimeField> {
    pub proof: IOPProof<F>,
    pub claims: Vec<F>,
    pub point: Vec<F>,
    pub matrix_eval: (Vec<IOPProof<F>>, Vec<Vec<F>>),
    pub delegation_points: Vec<Vec<F>>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound(
    serialize = "F: ark_serialize::CanonicalSerialize",
    deserialize = "F: ark_serialize::CanonicalDeserialize"
))]
pub(crate) struct MergeClaimsProof<F: PrimeField> {
    // Map an output index for a given to a node to the proof for merging the claims
    // related to this output
    proofs: HashMap<usize, MergeClaimNodeProof<F>>,
}

impl<F: PrimeField> MergeClaimsProof<F> {
    pub(crate) fn get_proof(&self, index: usize) -> Option<&MergeClaimNodeProof<F>> {
        self.proofs.get(&index)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound(
    serialize = "F: ark_serialize::CanonicalSerialize",
    deserialize = "F: ark_serialize::CanonicalDeserialize"
))]
pub(crate) struct MergeClaimNodeProof<F: PrimeField> {
    proof: same_poly::Proof<F>,
    agg_claim: Claim<F>,
    num_vars: usize,
}

impl<F: PrimeField> MergeClaimNodeProof<F> {
    pub(crate) fn generate_proof<T: Transcript>(
        t: &mut T,
        claims: &[&Claim<F>],
        output: &Tensor<F>,
    ) -> anyhow::Result<MergeClaimNodeProof<F>> {
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

    pub(crate) fn verify_proof<T: Transcript>(
        &self,
        t: &mut T,
        claims: &[&Claim<F>],
    ) -> anyhow::Result<Claim<F>> {
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

impl<'a, 'b, F, T, PCS> Prover<'a, 'b, F, T, PCS>
where
    T: Transcript,
    F: PrimeField,
    PCS: CommitmentScheme<Field = F>,
{
    pub fn new(ctx: &'a ProverContext<F, PCS>, transcript: &'b mut T) -> Self {
        Self {
            ctx,
            transcript,
            proofs: Default::default(),
            merge_claim_proofs: Default::default(),
            commit_prover: CommitmentProver::<F, PCS>::default(),
            lookup_witness: HashMap::default(),
            challenge_storage: ChallengeStorage::default(),
        }
    }

    pub(crate) fn add_common_claims(
        &mut self,
        node_id: NodeId,
        claims: HashMap<CommitmentId, Claim<F>>,
    ) {
        self.commit_prover.add_common_claims(
            claims
                .into_iter()
                .map(|(poly_id, claim)| (poly_id, vec![(node_id, claim)]))
                .collect(),
        )
    }

    pub(crate) fn add_table_claim(&mut self, table: &Table, chunk_id: ChunkID, claim: Claim<F>) {
        let table_node_id = chunk_id.0.into();
        self.commit_prover
            .add_table_claim(table_node_id, table, claim);
    }

    pub(crate) fn add_witness_claim(&mut self, node_id: NodeId, claims: Vec<Vec<Claim<F>>>) {
        self.commit_prover.add_witness_claim(node_id, claims);
    }

    /// Variant of `add_witness_claim` that can be employed for simplicity when there is one
    /// claim per witness polynomial
    pub(crate) fn add_witness_claim_per_poly(&mut self, node_id: NodeId, claims: Vec<Claim<F>>) {
        self.commit_prover.add_witness_claim(
            node_id,
            claims.into_iter().map(|claim| vec![claim]).collect(),
        );
    }

    pub(crate) fn lookup_witness(
        &self,
        id: NodeId,
    ) -> anyhow::Result<&Vec<CommittedPolynomial<'a, F, PCS>>> {
        self.lookup_witness
            .get(&id)
            .ok_or(anyhow!("No lookup witness found for node {id}!"))
    }

    pub(crate) fn push_proof(&mut self, node_id: NodeId, proof: LayerProof<F, PCS>) {
        self.proofs.insert(node_id, proof);
    }

    #[timed::timed_instrument(level = "debug")]
    fn prove_tables(
        &mut self,
        chunk_id: ChunkID,
        lookup_ctx: &LookupContext,
    ) -> anyhow::Result<Option<TableProof<F, PCS>>> {
        if lookup_ctx.is_empty() {
            Ok(None)
        } else {
            let table_node_id = self.ctx.commitment_ctx.table_node_id();
            let multiplicity_witness = self
                .lookup_witness(table_node_id)
                .context("No multiplicity commitment found during table proving")?;
            let logup_inputs = lookup_ctx
                .create_logup_inputs::<PCS, F>(multiplicity_witness, &self.challenge_storage)?;
            let multiplicity_commit = multiplicity_witness
                .iter()
                .map(VerifierCommitment::from)
                .collect_vec();
            // Run LogUp batch proving for all the tables at once
            let logup_batch_proof = new_batch_multiple_sizes_prove(&logup_inputs, self.transcript)?;

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
                        let mul_eval = table_claims[0].evaluation();
                        if tt.commit_output_column() {
                            claims_acc.push((tt, table_claims.last().unwrap().clone()));
                        }
                        acc.push(Claim::new(mul_point, mul_eval));
                        (acc, claims_acc)
                    },
                );

            commit_claims
                .into_iter()
                .for_each(|(tt, claim)| self.add_table_claim(tt, chunk_id, claim));
            self.add_witness_claim_per_poly(table_node_id, mul_claims);

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
        f_middle: &mut [Vec<F>],
        r1: &[F],
        mut r2: Vec<F>,
        is_fft: bool,
    ) -> anyhow::Result<(Vec<IOPProof<F>>, Vec<Vec<F>>, Vec<Vec<F>>)> {
        let mut omegas = vec![F::ZERO; 1 << r1.len()];
        Self::phi_pow_init(&mut omegas, r1.len(), is_fft)?;

        let mut proofs: Vec<IOPProof<F>> = Vec::new();
        let mut claims: Vec<Vec<F>> = Vec::new();
        let mut points: Vec<Vec<F>> = Vec::new();

        for l in (0..(r1.len() - 1)).rev() {
            let mut phi = vec![F::ZERO; f_middle[l].len()];
            let beta = evals(&r2[0..(r2.len() - 1)]);

            for i in 0..(phi.len()) {
                if !is_fft && l == f_middle.len() - 1 {
                    phi[i] = (F::ONE - r2[r2.len() - 1])
                        * (F::ONE - r1[(f_middle.len() - 1) - l]
                            + r1[(f_middle.len() - 1) - l]
                                * omegas[i << ((f_middle.len() - 1) - l)]);
                } else {
                    phi[i] = F::ONE - r1[(f_middle.len() - 1) - l]
                        + (F::ONE - F::from(2) * r2[r2.len() - 1])
                            * r1[(f_middle.len() - 1) - l]
                            * omegas[i << ((f_middle.len() - 1) - l)];
                }
            }

            let f1 = beta.into_mle();
            let f2 = phi.into_mle();
            let num_vars = f1.num_vars();
            let num_threads = optimal_sumcheck_threads(num_vars);
            let f3 = DensePolynomial::new_from_smart_slice(SmartSlice::Borrowed(&f_middle[l]));
            let mut expr_builder = VirtualPolynomialsBuilder::<F>::new(num_threads, num_vars);
            let expr = [&f1, &f2, &f3]
                .into_iter()
                .fold(Expression::Constant(F::ONE), |acc, p| {
                    acc * expr_builder.lift(Either::Left(p))
                });
            let virtual_poly = expr_builder.to_virtual_polys(&[expr], &[]);
            let (proof, state) = IOPProverState::<F>::prove(virtual_poly, self.transcript);

            let claim: Vec<F> = state.get_mle_flatten_final_evaluations();
            let point = state.collect_raw_challenges();
            r2 = point.clone();
            proofs.push(proof);
            claims.push(claim);
            points.push(point);
        }
        Ok((proofs, claims, points))
    }

    // Compute powers of roots of unity
    pub fn phi_pow_init(phi_mul: &mut [F], n: usize, is_fft: bool) -> anyhow::Result<()> {
        let length = 1 << n;
        let rou: F = get_root_of_unity(n)?;

        let mut phi = rou;
        if is_fft {
            phi = phi.inverse().expect("Tried to invert 0 in FFT prover");
        }
        phi_mul[0] = F::ONE;
        for i in 1..length {
            phi_mul[i] = phi_mul[i - 1] * phi;
        }
        Ok(())
    }

    // Efficiently compute the omegas of FFT/iFFT matrix reduced at rx
    // This is a copy-paste implementation from zkCNN paper
    pub fn phi_g_init(
        phi_g: &mut [F],
        mid_phi_g: &mut [Vec<F>],
        rx: Vec<F>,
        scale: F,
        n: usize,
        is_fft: bool,
    ) -> anyhow::Result<()> {
        let mut phi_mul = vec![F::ZERO; 1 << n];
        Self::phi_pow_init(&mut phi_mul, n, is_fft)?;
        if is_fft {
            phi_g[0] = scale;
            phi_g[1] = scale;
            for i in 1..(n + 1) {
                for b in 0..(1 << (i - 1)) {
                    let l = b;
                    let r = b ^ (1 << (i - 1));
                    let m = n - i;
                    let tmp1 = F::ONE - rx[m];
                    let tmp2 = rx[m] * phi_mul[b << m];
                    phi_g[r] = phi_g[l] * (tmp1 - tmp2);
                    phi_g[l] *= tmp1 + tmp2;
                }
                if i < n {
                    mid_phi_g[i - 1] = vec![F::ZERO; 1 << (i)];
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

                    let tmp1 = F::ONE - rx[m];
                    let tmp2 = rx[m] * phi_mul[b << m];

                    phi_g[r] = phi_g[l] * (tmp1 - tmp2);
                    phi_g[l] *= tmp1 + tmp2;
                }
                mid_phi_g[i - 1] = vec![F::ZERO; 1 << i];
                mid_phi_g[i - 1][..(1 << (i))].copy_from_slice(&phi_g[..(1 << (i))]);
            }
            for (b, item) in phi_mul.iter().enumerate().take(1 << (n - 1)) {
                let l = b;
                let tmp1 = F::ONE - rx[0];
                let tmp2 = rx[0] * *item;
                phi_g[l] *= tmp1 + tmp2;
            }
        }
        Ok(())
    }
    // The prove_batch_fft and prove_batch_ifft are extensions of prove_fft and prove_ifft but in the batch setting.
    // Namely when we want to proof fft or ifft for MORE THAN ONE INSTANCES.
    // In particular, instead of proving y = Wx we want to prove Y = WX where Y,X are matrixes.
    // Following the matrix to matrix multiplication protocol, let y_eval = Y(r1,r2).
    // Then we want to prove a sumcheck instance of the form y_eval = sum_{i \in [n]}W(r1,i)X(i,r2).
    pub fn prove_batch_fft(
        &mut self,
        r: Vec<F>,
        x: &mut [Vec<F>],
    ) -> anyhow::Result<BatchFFTProof<F>> {
        let padded_rows = 2 * x[0].len();
        for item in x.iter_mut() {
            item.resize(padded_rows, F::ZERO);
        }
        // Partition r in (r1,r2)
        let mut r1 = vec![F::ZERO; x[0].len().ilog2() as usize];
        let mut r2 = vec![F::ZERO; x.len().ilog2() as usize];
        let r1_len = r1.len();
        r1.copy_from_slice(&r[..r1_len]);

        for i in 0..r2.len() {
            r2[i] = r[i + r1.len()];
        }
        // compute W(r1,i)
        let mut w_red: Vec<F> = vec![F::ZERO; x[0].len()];
        let mut f_middle: Vec<Vec<F>> = vec![Vec::new(); r1.len() - 1];
        Self::phi_g_init(
            &mut w_red,
            &mut f_middle,
            r1.clone(),
            F::ONE,
            x[0].len().ilog2() as usize,
            false,
        )?;
        // compute X(i,r2)

        let mut f_m = x.iter().flatten().cloned().collect::<Vec<_>>().into_mle();

        f_m.fix_high_variables_in_place_parallel(&r2);

        // Construct the virtual polynomial and run the sumcheck prover
        let f_red = w_red.into_mle();
        let num_vars = f_m.num_vars();
        let num_threads = optimal_sumcheck_threads(num_vars);
        let mut expr_builder = VirtualPolynomialsBuilder::<F>::new(num_threads, num_vars);
        let expr = expr_builder.lift(Either::Left(&f_m)) * expr_builder.lift(Either::Left(&f_red));
        let virtual_poly = expr_builder.to_virtual_polys(&[expr], &[]);
        let (proof, state) = IOPProverState::<F>::prove(virtual_poly, self.transcript);

        let claims = state.get_mle_flatten_final_evaluations();
        let out_point = state.collect_raw_challenges();
        let (matrix_proofs, matrix_claims, delegation_points) =
            self.delegate_matrix_evaluation(&mut f_middle, &r1, out_point.clone(), false)?;
        Ok(BatchFFTProof {
            proof,
            claims,
            point: out_point,
            matrix_eval: (matrix_proofs, matrix_claims),
            delegation_points,
        })
    }

    pub fn prove_batch_ifft(&mut self, r: Vec<F>, prod: &[Vec<F>]) -> Result<BatchFFTProof<F>> {
        let scale = F::from(prod[0].len() as u64)
            .inverse()
            .expect("Tried to invert 0 in iFFT prover");

        // Partition r in (r1,r2)
        let mut r1 = vec![F::ZERO; prod[0].len().ilog2() as usize];
        let mut r2 = vec![F::ZERO; prod.len().ilog2() as usize];
        let r1_len = r1.len();
        r1.copy_from_slice(&r[..r1_len]);
        ensure!(
            r1[r1.len() - 1] == F::ZERO,
            "Error in randomness init batch ifft {:?}",
            r1[r1.len() - 1]
        );
        for i in 0..r2.len() {
            r2[i] = r[i + r1.len()];
        }
        // compute W(r1,i)
        let mut w_red: Vec<F> = vec![F::ZERO; prod[0].len()];
        let mut f_middle: Vec<Vec<F>> = vec![Vec::new(); r1.len() - 1];
        Self::phi_g_init(
            &mut w_red,
            &mut f_middle,
            r1.clone(),
            scale,
            prod[0].len().ilog2() as usize,
            true,
        )?;
        let f_red = w_red.into_mle();
        // compute X(i,r2)
        let mut f_m = prod
            .iter()
            .flatten()
            .cloned()
            .collect::<Vec<_>>()
            .into_mle();
        f_m.fix_high_variables_in_place_parallel(&r2);

        let num_vars = f_m.num_vars();
        let num_threads = optimal_sumcheck_threads(num_vars);
        let mut expr_builder = VirtualPolynomialsBuilder::<F>::new(num_threads, num_vars);
        let expr = expr_builder.lift(Either::Left(&f_m)) * expr_builder.lift(Either::Left(&f_red));
        let virtual_poly = expr_builder.to_virtual_polys(&[expr], &[]);
        let (proof, state) = IOPProverState::<F>::prove(virtual_poly, self.transcript);

        let claims = state.get_mle_flatten_final_evaluations();

        let out_point = state.collect_raw_challenges();
        let (proofs, matrix_claims, points) =
            self.delegate_matrix_evaluation(&mut f_middle, &r1, out_point.clone(), true)?;

        Ok(BatchFFTProof {
            proof,
            claims,
            point: out_point,
            matrix_eval: (proofs, matrix_claims),
            delegation_points: points,
        })
    }

    pub(crate) fn initialise_transcript(ctx: &ProverContext<F, PCS>) -> anyhow::Result<T>
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
        chunk_layers: &'d ModelLayersRef<'d>,
    ) -> anyhow::Result<ChunkProof<F, PCS>>
    where
        T: InitTranscript,
        'd: 'a,
    {
        let chunk_id = chunk.chunk_id;
        // add chunk splitting info to the transcript
        chunk.add_chunk_data_to_transcript(self.transcript)?;

        let lookup_ctx = chunk.chunk_lookup_ctx(&self.ctx.lookup);

        debug!("== Instantiate witness context ==");
        let metrics = Metrics::new();

        let chunk_io_commitments =
            self.instantiate_witness_ctx_for_chunk(&chunk, chunk_trace, &lookup_ctx, chunk_layers)?;

        let span = metrics.to_span();
        debug!("== Witness context metrics {} ==", span);

        debug!("== Challenge storage ==");
        let metrics = Metrics::new();
        // initialize challenge storgae for this chunk
        self.challenge_storage = if lookup_ctx.is_empty() {
            ChallengeStorage::<F>::default()
        } else {
            ChallengeStorage::<F>::initialise(&lookup_ctx, self.transcript)
        };
        debug!("== Challenge storage metrics {} ==", metrics.to_span());

        debug!("== Generating claims ==");
        let metrics = Metrics::new();

        let now = Instant::now();
        // compute the claims for the model outputs produced in this chunk, each identified by the
        // model output port ID
        let output_claims_by_port = chunk.model_outputs_in_chunk()?.into_iter()
            .try_fold(
                BTreeMap::new(), // we first collect all the output tensors, sorted by the output port ID
                |mut outputs, edge_id| {
                let output_edge = chunk.edge(&edge_id)?;
                let target_node = chunk.subgraph.target_node(&edge_id)?;
                let output_id: ChunkedOutput = target_node.as_output().ok_or(
                    anyhow!("Edge {edge_id} is not an output edge of the model")
                )?.into();
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
                let output_tensor: Tensor<F> = {
                    let tensor = trace_step.output_tensor_at(**source_port)?;
                    tensor.to_field().pad_next_power_of_two()
                };
                ensure!(
                    outputs.insert(output_id.clone(), output_tensor).is_none(),
                    "Found output tensor twice for chunk {} of output id {} in chunk {chunk_id}",
                    output_id.chunk_id,
                    output_id.io_id,
                );
                Ok(outputs)
            })? // then, we compute the claims for each output
            .into_iter()
            .map(|(port_id, tensor): (_, Tensor<F>)| {
                // For the output, we manually evaluate the MLE and check if it's the same as what prover
                // gave. Note prover could ellude that but it's simpler to avoid that special check right
                // now.
                Ok((port_id, compute_claim(self.transcript, tensor)?))
            }).collect::<anyhow::Result<HashMap<_,_>>>()?;

        // `chunk_output_claims` is a map storing claims related to the subset of input ports of layers in the model
        // which are connected to an output port of a node found in the current chunk. Here, we initialize
        // this map by claims about the input ports of layers that don't belong to this chunk, i.e., layers
        // of other chunks that use the outputs produced by layers in the current chunk
        let chunk_output_claims = chunk.outgoing_edges.keys()
            .try_fold(BTreeMap::new(), // we first collect all the tensors, sorted by the corresponding output port
            |mut claims_map, edge_id| {
                let edge = chunk.edge(edge_id)?;
                let source_node_id = edge.source();
                let trace_step = chunk_trace.get_step(&source_node_id)
                    .ok_or(
                        anyhow!("Trace step not found for node {source_node_id} in chunk {}", chunk.chunk_id)
                    )?;
                edge.ports().iter().try_for_each(|port| {
                    let source_port = NodeOutput::new(*source_node_id, port.source_port);
                    if let std::collections::btree_map::Entry::Vacant(e) = claims_map.entry(source_port) {
                         // Convert to field first, then pad
                         let tensor = trace_step.output_tensor_at(
                             port.source_port.into(),
                         )?;
                         let output_field: Tensor<F> = tensor.to_field().pad_next_power_of_two();
                         e.insert(output_field);
                    }
                    anyhow::Ok(())
                })?;
                anyhow::Ok(claims_map)
            })?
            .into_iter() // then, we compute the claims for each output
            .try_fold((HashMap::new(), vec![]), |(mut output_claims, mut common_point), (port, tensor)| {
                let mle = tensor.into_mle();
                if mle.num_vars() > common_point.len() {
                    // we need to add `mle.num_vars() - common_point.len()` coordinates to `common_point`
                    let mut new_coordinates = self.transcript.read_challenges(mle.num_vars() - common_point.len());
                    common_point.append(&mut new_coordinates);
                }
                let eval_point =  common_point[..mle.num_vars()].to_vec();
                let eval =  mle.evaluate(&eval_point)?;
                let claim = Claim::new(
                    eval_point,
                    eval
                );
                output_claims.insert(port, claim);
                anyhow::Ok((output_claims, common_point))
        })?.0;
        // each layer generates claims about its inputs. Each claim is indexed by
        // the id of the corresponding "input port" of the node, e.g. target_port when
        // considering incoming edges to this node.
        let mut claims: HashMap<NodeInput, Claim<F>> = HashMap::new();
        for (node_id, node) in chunk.subgraph.backward_iter() {
            match node {
                Node::Inner(node) => {
                    let section = chunk_trace
                        .get_step(&node_id)
                        .ok_or(anyhow!("Step in trace not found for node {node_id}"))?;
                    let split_layer = if let ChunkedNode::SplitLayer(split_layer) = node {
                        Some(Layer::Split(split_layer.clone()))
                    } else {
                        None
                    };
                    let recombination_layer =
                        if let ChunkedNode::RecombinationLayer(rec_layer) = node {
                            Some(Layer::Recombination(rec_layer.clone()))
                        } else {
                            None
                        };
                    let op = match node {
                        ChunkedNode::OriginalNode(_) => chunk_layers
                            .get(&node_id)
                            .ok_or(anyhow!("Node {node_id} not found in model"))?,
                        ChunkedNode::ChunkedLayer(chunked_layer) => chunk_layers
                            .get(&chunked_layer.original_node_id)
                            .ok_or(anyhow!(
                                "Node {} not found in model",
                                chunked_layer.original_node_id
                            ))?,
                        ChunkedNode::SplitLayer(_) => split_layer.as_ref().unwrap(),
                        ChunkedNode::RecombinationLayer(_) => recombination_layer.as_ref().unwrap(),
                    };
                    trace!("Proving node with id {node_id}: {:?}", op.describe());

                    // Load all output tensors, convert to field, then pad
                    let handles = section
                        .output_tensors()?
                        .into_iter()
                        .map(|tensor| tensor.to_field().pad_next_power_of_two())
                        .collect::<Vec<_>>();

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
                    let split_layer = if let ChunkedNode::SplitLayer(split_layer) = node {
                        Some(LayerCtx::Split(split_layer.clone()))
                    } else {
                        None
                    };
                    let recombination_layer =
                        if let ChunkedNode::RecombinationLayer(rec_layer) = node {
                            Some(LayerCtx::Recombination(rec_layer.clone()))
                        } else {
                            None
                        };
                    let ctx = match node {
                        ChunkedNode::OriginalNode(_) => self
                            .ctx
                            .model_ctx
                            .nodes
                            .node(node_id)
                            .ok_or(anyhow!("Node {node_id} not found in proving context"))?
                            .as_inner()
                            .ok_or(anyhow!(
                                "Node {node_id} is not an inner node in proving context"
                            ))?,
                        ChunkedNode::ChunkedLayer(chunked_layer) => self
                            .ctx
                            .model_ctx
                            .nodes
                            .node(chunked_layer.original_node_id)
                            .ok_or(anyhow!("Node {node_id} not found in proving context"))?
                            .as_inner()
                            .ok_or(anyhow!(
                                "Node {node_id} is not an inner node in proving context"
                            ))?,
                        ChunkedNode::SplitLayer(_) => split_layer.as_ref().unwrap(),
                        ChunkedNode::RecombinationLayer(_) => recombination_layer.as_ref().unwrap(),
                    };
                    let my_claims = if op.is_provable() {
                        debug!("proving layer {:?}", op.describe());
                        measure::r_and_accumulate(
                            format!("{LAYER_WISE_MEASURE_PREFIX}layer_{}", op.as_kind_str())
                                .as_str(),
                            || {
                                op.prove(
                                    node_id,
                                    ctx,
                                    claims_for_prove.iter().collect::<Vec<_>>(),
                                    section,
                                    &mut self,
                                )
                                .with_context(|| format!("proving {}: {}", node_id, op.describe()))
                            },
                            Some(|a, b| a + b),
                        )??
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
                    claims.insert(
                        NodeInput::new(node_id, 0),
                        output_claims_by_port[&o.into()].clone(),
                    );
                }
            }
        }

        let span = metrics.to_span();
        debug!("== Claims generation metrics {} ==", span);

        // Now we need add the claims about the input and output of the chunk
        chunk
            .compute_output_boundary_edges_claims(&chunk_output_claims)?
            .into_iter()
            .for_each(|(poly_id, claims)| self.add_witness_claim(poly_id, vec![claims]));

        chunk
            .compute_input_boundary_edges_claims(&claims)?
            .into_iter()
            .for_each(|(poly_id, claims)| self.add_witness_claim(poly_id, vec![claims]));

        // Now we have to make the table proofs
        debug!("== Generating Lookup Table claims ==");
        let metrics = Metrics::new();
        let table_proof = self.prove_tables(chunk_id, &lookup_ctx)?;
        let span = metrics.to_span();
        debug!("== Lookup Table claims generation metrics {} ==", span);

        measure::record_timing("prove_claims", now.elapsed());

        debug!("== Generate proof ==");
        let metrics = Metrics::new();

        let commit_proof = measure::r("prove_commitment_opening", || {
            self.commit_prover.prove(
                &self.ctx.commitment_ctx,
                &self.lookup_witness,
                self.transcript,
            )
        })?;
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
            commitments: chunk_io_commitments,
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
        debug!("== Generate proof metrics {} ==", span);

        Ok(chunk_proof)
    }

    /// Build the execution graph to run the proving of chunks `chunks`.
    /// It currently assigns one node per chunk, and the first node, with id 0,
    /// is assigned as a coordinator, that starts the process (e.g. executes the first task)
    /// and finishes it (outputs the final proof).
    #[allow(clippy::type_complexity)]
    pub fn build_execution_graph<'c>(
        chunks: Vec<ModelChunk>,
    ) -> anyhow::Result<ProverGraph<'a, 'c, F, T, PCS>>
    where
        T: InitTranscript,
        PCS: 'static,
    {
        Self::build_execution_graph_internal(chunks, false)
    }

    /// Build the execution graph to run the proving of chunks `chunks` for local proving.
    #[allow(clippy::type_complexity)]
    fn build_local_execution_graph<'c>(
        chunks: Vec<ModelChunk>,
    ) -> anyhow::Result<ProverGraph<'a, 'c, F, T, PCS>>
    where
        T: InitTranscript,
        PCS: 'static,
    {
        Self::build_execution_graph_internal(chunks, true)
    }

    /// Build the execution graph to run the proving of chunks `chunks`.
    /// It currently assigns one node per chunk, and the first node, with id 0,
    /// is assigned as a coordinator, that starts the process (e.g. executes the first task)
    /// and finishes it (outputs the final proof). The `local` flag is used to specify whether
    /// the execution graph is built for local proving or distributed proving.
    fn build_execution_graph_internal<'c>(
        chunks: Vec<ModelChunk>,
        local: bool,
    ) -> anyhow::Result<ProverGraph<'a, 'c, F, T, PCS>>
    where
        T: InitTranscript,
        PCS: 'static,
    {
        // add the input node of the graph, which is doing the preprocessing task, and the output node
        // of the graph, which is doing the opening of the model claims
        let mut exec_graph = ProverGraph::new();
        let num_chunks = chunks.len();
        let init_node_id = exec_graph.add_inner(
            ProverGraphNode::ProverSplit(if local {
                // when building the execution graph for local proving, we avoid drying the trace,
                // as we don't want to have to re-load the tensors in the trace
                SplitNode::new(chunks).disable_dry_trace()
            } else {
                SplitNode::new(chunks)
            })
            .colored(0),
        )?;
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
    pub fn graph_inputs(
        full_trace: Trace<Element>,
        graph: &ProverGraph<F, T, PCS>,
    ) -> anyhow::Result<HashMap<NodeInput, ProverGraphIO<F, PCS>>>
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
        exec_graph: ProverGraph<'a, 'c, F, T, PCS>,
        inputs: HashMap<NodeInput, ProverGraphIO<F, PCS>>,
        context: &LocalProverCtx<'a, 'c, F, PCS>,
        config: Ex::Config,
    ) -> anyhow::Result<Proof<F, PCS>>
    where
        T: InitTranscript,
        Ex: Executor<ProverGraphNode<'a, 'c, F, T, PCS>, usize>,
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
        Ex: Executor<ProverGraphNode<'a, 'd, F, T, PCS>, usize>,
    >(
        ctx: &'a ProverContext<F, PCS>,
        mut full_trace: Trace<Element>,
        num_chunks: Option<usize>,
        chunking_strategy: S,
        model: &'d Model<Element>,
        executor_conf: Ex::Config,
    ) -> anyhow::Result<(Proof<F, PCS>, IO<F>)>
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
        let (chunks, split_info) = ctx.split_in_chunks(num_chunks, chunking_strategy)?;

        full_trace.replace_splitted_nodes(&ctx.model_ctx, &split_info)?;

        let io = full_trace.to_verifier_io()?;

        let global_metrics = Metrics::new();

        let output_proof = measure::r("prove_full", || {
            // build the computational graph to prove chunks
            let graph = Self::build_local_execution_graph(chunks)?;
            let inputs = Self::graph_inputs(full_trace, &graph)?;
            let context = LocalProverCtx::new(ctx, model);
            Self::run_execution_graph::<Ex>(graph, inputs, &context, executor_conf)
        })?;

        let global_metrics_span = global_metrics.to_span();
        debug!("== Global metrics {} ==", global_metrics_span);
        Ok((output_proof, io))
    }

    pub fn prove<'d>(
        ctx: &'a ProverContext<F, PCS>,
        full_trace: Trace<Element>,
        model: &'d Model<Element>,
    ) -> anyhow::Result<(Proof<F, PCS>, IO<F>)>
    where
        T: InitTranscript,
        PCS: 'static,
    {
        let span = info_span!("zkml_prove");
        let _guard = span.enter();
        let chunking_strategy = DefaultChunkingStrategy::from(&full_trace);
        Self::chunked_prove_local::<_, SequentialExecutor>(
            ctx,
            full_trace,
            Some(1),
            chunking_strategy,
            model,
            (),
        )
    }

    /// Flattens all the claims to give to the proving logic of the node. If
    /// there are claims linked to the same port, the claims will be merged.
    fn flatten_and_merge_claims(
        &mut self,
        claims: BTreeMap<PortId, Vec<&Claim<F>>>,
        outputs: &[&Tensor<F>],
        node_id: NodeId,
    ) -> anyhow::Result<Vec<Claim<F>>> {
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
        claims: &[&Claim<F>],
        output: &Tensor<F>,
    ) -> anyhow::Result<(Claim<F>, MergeClaimNodeProof<F>)> {
        let proof = MergeClaimNodeProof::generate_proof(self.transcript, claims, output)?;
        Ok((proof.agg_claim.clone(), proof))
    }

    /// Looks at all the individual polys to accumulate from the witnesses and create the context from that.
    #[timed_instrument]
    fn instantiate_witness_ctx_for_chunk<'c, 'd>(
        &mut self,
        chunk: &ModelChunk,
        chunk_trace: &'c Trace<Element>,
        lookup_ctx: &LookupContext,
        chunk_layers: &'d ModelLayersRef<'d>,
    ) -> anyhow::Result<ChunkIOCommitments<VerifierCommitment<PCS>>>
    where
        'd: 'a,
    {
        let LookupWitness {
            logup_witnesses,
            table_witnesses,
            chunk_commitments,
        } = generate_lookup_witness_for_chunk::<F, T, PCS>(
            chunk,
            lookup_ctx,
            chunk_trace,
            self.ctx,
            self.transcript,
            chunk_layers,
        )?;
        self.lookup_witness = logup_witnesses;

        if let Some(commit) = table_witnesses {
            let table_node_id = self.ctx.commitment_ctx.table_node_id();
            self.lookup_witness.insert(table_node_id, commit);
        }
        Ok(chunk_commitments)
    }
}
