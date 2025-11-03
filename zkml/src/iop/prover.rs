use super::{ChallengeStorage, Proof, TableProof};
use crate::{
    Claim, Element, Tensor,
    commit::{compute_betas_eval, mmcs_context, same_poly},
    graph::{Node, NodeId, NodeInput, PortId},
    iop::{context::ProverContext, model_output_claims},
    layers::{
        LayerProof,
        provable::{OpInfo, ProvableOp},
    },
    lookup::{
        context::{LookupWitness, TableType, generate_lookup_witnesses},
        logup_gkr::prover::batch_multiple_sizes_prove,
    },
    model::InferenceTrace,
    tensor::{CommitmentId, get_root_of_unity},
};
use anyhow::{Context as _, Result, anyhow, ensure};
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
use tracing::{debug, trace};
use transcript::Transcript;
use utils::{Metrics, stream_metrics};

/// Prover generates a series of sumcheck proofs to prove the inference of a model
pub struct Prover<'a, 'b, E: ExtensionField, T: Transcript<E>, PCS: PolynomialCommitmentScheme<E>>
where
    E::BaseField: Serialize + DeserializeOwned,
    E: Serialize + DeserializeOwned,
    PCS::CommitmentWithWitness: Serialize + DeserializeOwned,
    PCS::ProverParam: Send + Sync,
{
    ctx: &'a ProverContext<E, PCS>,
    // proofs for each layer being filled
    proofs: HashMap<NodeId, LayerProof<E, PCS>>,
    table_proofs: Vec<TableProof<E, PCS>>,
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

impl<'a, 'b, E, T, PCS> Prover<'a, 'b, E, T, PCS>
where
    T: Transcript<E>,
    E: ExtensionField,
    E::BaseField: Serialize + DeserializeOwned,
    E: Serialize + DeserializeOwned,
    PCS: PolynomialCommitmentScheme<E> + Send + Sync,
    PCS::CommitmentWithWitness: Serialize + DeserializeOwned + Send + Sync,
    PCS::ProverParam: Send + Sync,
{
    pub fn new(ctx: &'a ProverContext<E, PCS>, transcript: &'b mut T) -> Self {
        Self {
            ctx,
            transcript,
            proofs: Default::default(),
            table_proofs: Vec::default(),
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

    pub(crate) fn add_table_claim(&mut self, table_type: &TableType, claim: Claim<E>) {
        let table_node_id = self.ctx.commitment_ctx.table_node_id();
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
    fn prove_tables(&mut self) -> anyhow::Result<()> {
        if self.ctx.lookup.is_empty() {
            Ok(())
        } else {
            let table_node_id = self.ctx.commitment_ctx.table_node_id();
            let multiplicity_witness = self
                .lookup_witness(table_node_id)
                .context("No mutliplicity commitment found during table proving")?;
            let logup_inputs = self
                .ctx
                .lookup
                .create_logup_inputs::<PCS, E>(multiplicity_witness, &self.challenge_storage)?;
            let multiplicity_commit = PCS::get_pure_commitment(multiplicity_witness);
            // Run LogUp batch proving for all the tables at once
            let logup_batch_proof = batch_multiple_sizes_prove(&logup_inputs, self.transcript)?;

            // Now we takes the evals and append the correct values for commitment opening
            let all_claims = logup_batch_proof.output_claims();
            let (mul_claims, commit_claims, _) = self.ctx.lookup.iter().fold(
                (vec![], vec![], 0),
                |(mut acc, mut claims_acc, skip), tt| {
                    let take = 1 + tt.num_columns();
                    let table_claims = all_claims
                        .iter()
                        .skip(skip)
                        .take(take)
                        .collect::<Vec<&Claim<E>>>();
                    let mul_point = table_claims[0].point.clone();
                    let mul_eval = table_claims[0].eval;
                    if tt.has_committed_claims() {
                        claims_acc.push((tt, table_claims[take - 1].clone()));
                    }
                    acc.push((mul_point, mul_eval));
                    (acc, claims_acc, skip + take)
                },
            );

            commit_claims
                .into_iter()
                .for_each(|(tt, claim)| self.add_table_claim(tt, claim));
            let grouped = mul_claims
                .into_iter()
                .into_group_map()
                .into_iter()
                .sorted_by(|a, b| Ord::cmp(&b.0.len(), &a.0.len()))
                .collect::<Vec<(Point<E>, Vec<E>)>>();
            self.add_witness_claim(table_node_id, grouped);

            self.table_proofs.push(TableProof {
                multiplicity_commit,
                lookup: logup_batch_proof,
            });

            Ok(())
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

    pub fn prove<'d: 'a>(
        mut self,
        full_trace: &'d InferenceTrace<'d, E, Element>,
    ) -> anyhow::Result<Proof<E, PCS>> {
        debug!("== Instantiate witness context ==");

        let global_metrics = Metrics::new();
        let metrics = Metrics::new();
        self.ctx.write_to_transcript(self.transcript)?;

        self.instantiate_witness_ctx(full_trace)?;

        let span = metrics.to_span();
        stream_metrics("Witness context", &span);
        debug!("== Witness context metrics {} ==", span);

        debug!("== Generating claims ==");
        let metrics = Metrics::new();
        let trace = full_trace
            .clone()
            .into_fields()
            .context("converting trace to fields")?;
        // this is the random set of variables to fix at each step derived as the output of
        // sumcheck.
        // For the first step, so before the first sumcheck, we generate it from FS.
        // The dimension is simply the number of variables needed to address all the space of the
        // input vector.
        let out_claims = model_output_claims(self.transcript, &trace.outputs()?);
        let mut store = trace.store.clone();

        // each layer generates claims about its inputs. Each claim is stored at
        // the right position amongst all the "input ports" of the node, e.g. target_port when
        // considering incoming edges to this node.
        let mut claims: HashMap<NodeInput, Claim<E>> = HashMap::new();
        for (node_id, node) in self.ctx.model_ctx.nodes.backward_iter() {
            match node {
                Node::Inner(ctx) => {
                    let section = trace
                        .get_step(node_id)
                        .ok_or(anyhow!("Step in trace not found for node {node_id}"))?;
                    trace!(
                        "Proving node with id {node_id}: {:?}",
                        section.op.describe()
                    );

                    // Hydrate all the output tensors of this node
                    let tensors = section
                        .node_outputs
                        .outputs
                        .iter()
                        .map(|t| {
                            t.hydrate(store.clone())
                                .with_context(|| format!("hydrating tensor {}", t.storage_key()))
                        })
                        .collect::<anyhow::Result<Vec<_>>>()?;

                    // The claims for this node, i.e. the claims stemming from
                    // the input nodes of the successor nodes connected to this
                    // nodes output nodes, are collected and ordered by output
                    // port (on this node) number. Remember that the graph is
                    // traversed backwards, so output nodes are conceptually
                    // inputs, and vice-versa.
                    let claims_for_node: BTreeMap<PortId, Vec<&Claim<E>>> = self
                        .ctx
                        .model_ctx
                        .nodes
                        .outgoing_feeds(node_id)
                        .into_iter()
                        .fold(BTreeMap::new(), |mut ax, feed| {
                            ax.entry(feed.source.port)
                                .or_default()
                                .push(&claims[&feed.target]);
                            ax
                        });

                    // Just like for verification, there might be claims to be
                    // merged if they are connected to the same output port for
                    // this node.
                    let claims_for_prove = self.flatten_and_merge_claims(
                        claims_for_node,
                        &tensors.iter().collect::<Vec<_>>(),
                        node_id,
                    )?;

                    // prove or propagate the claims
                    let my_claims = if section.op.is_provable() {
                        section
                            .op
                            .prove(
                                node_id,
                                ctx,
                                claims_for_prove.iter().collect::<Vec<_>>(),
                                section,
                                &mut self,
                                &mut store,
                            )
                            .with_context(|| {
                                format!("proving {}: {}", node_id, section.op.describe())
                            })?
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
                    claims.insert(NodeInput::new(node_id, 0), out_claims[*o].clone());
                }
            }
        }

        let span = metrics.to_span();
        stream_metrics("Claims", &span);
        debug!("== Claims generation metrics {} ==", span);

        // Now we have to make the table proofs
        debug!("== Generating Lookup Table claims ==");
        let metrics = Metrics::new();
        self.prove_tables()?;
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
        let output_proof = Proof {
            steps: self.proofs,
            merge_claim_proofs: self.merge_claim_proofs,
            table_proofs: self.table_proofs,
            commit: commit_proof,
        };

        let span = metrics.to_span();
        stream_metrics("Proof", &span);
        debug!("== Generate proof metrics {} ==", span);

        let global_metrics_span = global_metrics.to_span();
        stream_metrics("Global", &global_metrics_span);
        debug!("== Global metrics {} ==", global_metrics_span);
        Ok(output_proof)
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
            "Number of claims and outputs must be the same"
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
    fn instantiate_witness_ctx<'d: 'a>(
        &mut self,
        trace: &'d InferenceTrace<'d, E, Element>,
    ) -> anyhow::Result<()> {
        let LookupWitness {
            challenge_storage,
            logup_witnesses,
            table_witnesses,
        } = generate_lookup_witnesses::<E, T, PCS>(trace, self.ctx, self.transcript)?;
        self.challenge_storage = challenge_storage;
        self.lookup_witness = logup_witnesses;

        if let Some(commit) = table_witnesses {
            let table_node_id = self.ctx.commitment_ctx.table_node_id();
            self.lookup_witness.insert(table_node_id, commit);
        }
        Ok(())
    }
}
