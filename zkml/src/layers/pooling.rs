use super::{
    LayerCtx,
    provable::{Evaluate, LayerOut, OpInfo, PadOp, ProvableOp, ProveInfo, VerifiableCtx},
};
use crate::{
    Claim, Element, Prover, ProverContext, Shape, Tensor,
    backend::Maxpool2dConfig,
    commit::{compute_betas_eval, identity_eval},
    graph::NodeId,
    iop::{context::ShapeStep, verifier::Verifier},
    layers::{ContextAux, LayerProof, convolution::check_cnn_input},
    lookup::{
        context::{LayerLookupContext, LookupWitnessGen, TableType},
        logup_gkr::{
            prover::batch_multiple_sizes_prove as logup_batch_prove, structs::LogUpBatchProof,
            verifier::verify_logup_proof_multiple_sizes,
        },
    },
    model::Step,
    number::Number,
    padding::{PaddingMode, ShapeInfo, pooling},
    quantization::{Fieldizer, IntoElement},
    tensor::WrappedTensor,
    to_base,
};
use anyhow::{Context, Result, ensure};

use either::Either;
use ff_ext::ExtensionField;
use itertools::{Itertools, izip};
use mpcs::PolynomialCommitmentScheme;
use multilinear_extensions::{
    Expression,
    mle::MultilinearExtension,
    util::{ceil_log2, transpose},
    virtual_polys::VirtualPolynomialsBuilder,
};
use rayon::prelude::*;
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use std::collections::HashMap;
use sumcheck::{
    structs::{IOPProof, IOPProverState, IOPVerifierState},
    util::optimal_sumcheck_threads,
};
use tenstore::GenStore;
use transcript::Transcript;
use witness::RowMajorMatrix;

/// Short name used to identify the pooling layer.
pub const POOLING_LAYER: &str = "POOL";

pub const MAXPOOL2D_KERNEL_SIZE: usize = 2;

#[derive(Clone, Debug, Serialize, Deserialize, Copy)]
pub enum Pooling {
    Maxpool2D(Maxpool2D),
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PoolingCtx {
    pub poolinfo: Maxpool2D,
    pub node_id: NodeId,
    pub lookup_ctx: LayerLookupContext,
}

/// Contains proof material related to one step of the inference
#[derive(Clone, Serialize, Deserialize)]
#[serde(bound(serialize = "E: Serialize", deserialize = "E: DeserializeOwned"))]
pub struct PoolingProof<E: ExtensionField, PCS: PolynomialCommitmentScheme<E>>
where
    E::BaseField: Serialize + DeserializeOwned,
{
    /// the actual sumcheck proof proving that the product of correct terms is always zero
    pub(crate) sumcheck: IOPProof<E>,
    /// The lookup proof showing that the diff is always in the correct range
    pub(crate) lookup: LogUpBatchProof<E>,
    /// The output evaluations of the diff polys produced by the zerocheck
    pub(crate) zerocheck_evals: Vec<E>,
    /// This tells the verifier how far apart the variables get fixed on the input MLE
    pub(crate) variable_gap: usize,
    /// Commitments that are part of the commitment opening proof for this layer
    pub(crate) commitment: PCS::Commitment,
}

impl<E, PCS> PoolingProof<E, PCS>
where
    E: ExtensionField,
    PCS: PolynomialCommitmentScheme<E>,
{
    pub(crate) fn write_commitment<T: Transcript<E>>(&self, transcript: &mut T) -> Result<()> {
        PCS::write_commitment(&self.commitment, transcript).map_err(|e| anyhow::anyhow!("{e:?}"))
    }
}

const IS_PROVABLE: bool = true;

impl OpInfo for Pooling {
    fn output_shapes(
        &self,
        input_shapes: &[Shape],
        _padding_mode: PaddingMode,
    ) -> Result<Vec<Shape>> {
        match self {
            Pooling::Maxpool2D(maxpool2_d) => Ok(input_shapes
                .iter()
                .map(|shape| maxpool2_d.output_shape(shape))
                .collect()),
        }
    }

    fn num_outputs(&self, num_inputs: usize) -> Result<usize> {
        Ok(num_inputs)
    }

    fn describe(&self) -> String {
        match self {
            Pooling::Maxpool2D(maxpool2d) => format!(
                "MaxPool2D{{ kernel size: {}, stride: {} }}",
                maxpool2d.kernel_size, maxpool2d.stride
            ),
        }
    }

    fn is_provable(&self) -> bool {
        IS_PROVABLE
    }
}

impl Evaluate<Element> for Pooling {
    fn evaluate<E: ExtensionField>(
        &self,
        inputs: &[&WrappedTensor<Element>],
    ) -> Result<LayerOut<Element, E>> {
        ensure!(
            inputs.len() == 1,
            "Found more than 1 input when evaluating pooling layer"
        );
        let input = inputs[0];

        match self {
            Pooling::Maxpool2D(maxpool2d) => {
                ensure!(input.rank() >= 2, "The rank must be at least 2D");

                // normalize the shape to 4D by unqueezing
                let rank_difference = 4 - input.rank();
                let binput = input.clone().unsqueeze_dim_4();

                let config = Maxpool2dConfig {
                    kernel_size: maxpool2d.kernel_size,
                    stride: maxpool2d.stride,
                };
                let mut result = WrappedTensor::<Element>::max_pool2d(binput, config)?;

                for _ in 0..rank_difference {
                    result = result.squeeze(0)?;
                }

                Ok(LayerOut::from_vec(vec![result]))
            }
        }
    }
}

impl Evaluate<f32> for Pooling {
    fn evaluate<E: ExtensionField>(
        &self,
        inputs: &[&WrappedTensor<f32>],
    ) -> Result<LayerOut<f32, E>> {
        ensure!(
            inputs.len() == 1,
            "Found more than 1 input when evaluating pooling layer"
        );
        let input = inputs[0];

        match self {
            Pooling::Maxpool2D(maxpool2d) => {
                ensure!(input.rank() >= 2, "The rank must be at least 2D");

                // normalize the shape to 4D by unqueezing
                let rank_difference = 4 - input.rank();
                let binput = input.clone().unsqueeze_dim_4();

                let kernel_size = [maxpool2d.kernel_size, maxpool2d.kernel_size];
                let stride = [maxpool2d.stride, maxpool2d.stride];
                let padding = [0, 0];
                let dilation = [1, 1];

                let mut result = WrappedTensor::<f32>::max_pool2d(
                    binput,
                    kernel_size,
                    stride,
                    padding,
                    dilation,
                )?;

                for _ in 0..rank_difference {
                    result = result.squeeze(0)?;
                }

                Ok(LayerOut::from_vec(vec![result]))
            }
        }
    }
}

impl ProveInfo for Pooling {
    fn step_info<E: ExtensionField>(
        &self,
        id: NodeId,
        mut aux: ContextAux,
    ) -> Result<(LayerCtx<E>, ContextAux)> {
        let info = match self {
            Pooling::Maxpool2D(info) => {
                aux.tables.insert(TableType::Range);

                aux.last_output_shape =
                    self.output_shapes(&aux.last_output_shape, PaddingMode::Padding)?;

                // Set the model polys to be empty
                aux.model_polys = None;
                aux.max_poly_len = aux
                    .last_output_shape
                    .iter()
                    .fold(aux.max_poly_len, |acc, shapes| {
                        acc.max(shapes.next_power_of_two().product())
                    });

                let lookup_ctx = LayerLookupContext::new(vec![TableType::Range], vec![4]);
                LayerCtx::Pooling(PoolingCtx {
                    poolinfo: *info,
                    node_id: id,
                    lookup_ctx,
                })
            }
        };
        Ok((info, aux))
    }
}

impl PadOp for Pooling {
    fn pad_node(self, si: &mut ShapeInfo) -> Result<Self>
    where
        Self: Sized,
    {
        pooling(self, si)
    }
}

impl<E, PCS> ProvableOp<E, PCS> for Pooling
where
    E: ExtensionField + Serialize + DeserializeOwned,
    E::BaseField: Serialize + DeserializeOwned,
    PCS: PolynomialCommitmentScheme<E> + Send + Sync,
    PCS::CommitmentWithWitness: Serialize + DeserializeOwned + Send + Sync,
    PCS::ProverParam: Send + Sync,
{
    type Ctx = PoolingCtx;

    fn prove<T: Transcript<E>>(
        &self,
        id: NodeId,
        ctx: &Self::Ctx,
        last_claims: Vec<&Claim<E>>,
        step_data: &Step<E, Element, E>,
        prover: &mut Prover<E, T, PCS>,
        store: &mut GenStore,
    ) -> Result<Vec<Claim<E>>> {
        let input_tensors = step_data.input_tensors(store)?;

        Ok(vec![self.prove_pooling(
            prover,
            last_claims[0],
            &input_tensors[0],
            ctx,
            id,
        )?])
    }

    fn gen_lookup_witness(
        &self,
        id: NodeId,
        ctx: &ProverContext<E, PCS>,
        step_data: &Step<E, Element, Element>,
        store: &mut GenStore,
    ) -> Result<LookupWitnessGen<E, PCS>> {
        let input_tensors = step_data.input_tensors(store)?;
        let output_tensors = step_data.output_tensors(store)?;

        ensure!(
            input_tensors.len() == 1,
            "Input for pooling layer with invalid length. expected: 1 got: {}",
            input_tensors.len(),
        );
        ensure!(
            output_tensors.len() == 1,
            "Output for pooling layer with invalid length. expected: 1 got: {}",
            output_tensors.len(),
        );

        let mut element_count = HashMap::<Element, u64>::new();

        let mut column_evals = match self {
            Pooling::Maxpool2D(maxpool2d) => {
                let field_vecs = maxpool2d.compute_polys::<E>(&input_tensors[0]).unwrap();

                for value in field_vecs.iter().flat_map(|v| v.iter()) {
                    let el = E::from(*value).to_element();
                    *element_count.entry(el).or_default() += 1;
                }

                field_vecs
            }
        };
        // Commit to the witnes polys
        let output_poly = to_base::<E, _>(output_tensors[0].get_data());
        column_evals.push(output_poly);
        let width = column_evals.len();
        let values = transpose(column_evals);
        let rmm = RowMajorMatrix::<E::BaseField>::new_by_inner_matrix(
            ceno_p3::matrix::dense::DenseMatrix::new(values.concat(), width),
            witness::InstancePaddingStrategy::Default,
        );

        let layer_commitment = ctx.commitment_ctx.batch_commit(vec![rmm])?;

        let mut gen_w = LookupWitnessGen::<E, PCS>::default();
        gen_w.insert_logup_witness(id, layer_commitment);
        gen_w.insert_element_count(TableType::Range, element_count);

        Ok(gen_w)
    }
}

impl OpInfo for PoolingCtx {
    fn output_shapes(
        &self,
        input_shapes: &[Shape],
        _padding_mode: PaddingMode,
    ) -> Result<Vec<Shape>> {
        Ok(input_shapes
            .iter()
            .map(|shape| self.poolinfo.output_shape(shape))
            .collect())
    }

    fn num_outputs(&self, num_inputs: usize) -> Result<usize> {
        Ok(Pooling::num_outputs(num_inputs))
    }

    fn describe(&self) -> String {
        format!(
            "MaxPool2D ctx{{ kernel size: {}, stride: {} }}",
            self.poolinfo.kernel_size, self.poolinfo.stride
        )
    }

    fn is_provable(&self) -> bool {
        IS_PROVABLE
    }
}

impl<E, PCS> VerifiableCtx<E, PCS> for PoolingCtx
where
    E: ExtensionField + Serialize + DeserializeOwned,
    E::BaseField: Serialize + DeserializeOwned,
    PCS: PolynomialCommitmentScheme<E>,
{
    type Proof = PoolingProof<E, PCS>;

    fn verify<T: Transcript<E>>(
        &self,
        proof: &Self::Proof,
        last_claims: &[&Claim<E>],
        verifier: &mut Verifier<E, T, PCS>,
        shape_step: &ShapeStep,
    ) -> Result<Vec<Claim<E>>> {
        Ok(vec![self.verify_pooling(
            verifier,
            last_claims[0],
            proof,
            shape_step,
        )?])
    }

    fn write_proof_to_transcript<T: Transcript<E>>(
        &self,
        proof: &Self::Proof,
        transcript: &mut T,
    ) -> anyhow::Result<()> {
        proof.write_commitment(transcript)
    }
}

impl Pooling {
    fn num_outputs(num_inputs: usize) -> usize {
        num_inputs
    }

    pub fn op<T: Number>(&self, input: &Tensor<T>) -> Result<Tensor<T>> {
        match self {
            Pooling::Maxpool2D(maxpool2d) => {
                input.maxpool2d(maxpool2d.kernel_size, maxpool2d.stride)
            }
        }
    }

    fn num_vars_for_outputs(output_shapes: &[Shape]) -> Result<usize> {
        Ok(output_shapes
            .iter()
            .try_fold(None, |expected_num_vars, shape| {
                let num_vars = shape.iter().map(|dim| ceil_log2(*dim)).sum::<usize>();
                if let Some(vars) = expected_num_vars {
                    ensure!(
                        vars == num_vars,
                        "All output shapes for pooling must have the same number of variables"
                    );
                }
                Ok(Some(num_vars))
            })?
            .expect("No output shape found for pooling layer?"))
    }

    #[timed::timed_instrument(name = "Prover::prove_pooling_step")]
    pub fn prove_pooling<E, T: Transcript<E>, PCS>(
        &self,
        prover: &mut Prover<E, T, PCS>,
        // last random claim made
        last_claim: &Claim<E>,
        // input to the dense layer
        input: &Tensor<E>,
        info: &PoolingCtx,
        id: NodeId,
    ) -> anyhow::Result<Claim<E>>
    where
        E::BaseField: Serialize + DeserializeOwned,
        E: ExtensionField + Serialize + DeserializeOwned,
        PCS: PolynomialCommitmentScheme<E> + Send + Sync,
        PCS::CommitmentWithWitness: Serialize + DeserializeOwned + Send + Sync,
        PCS::ProverParam: Send + Sync,
    {
        ensure!(
            input.rank() == 3,
            "Maxpool needs 3D inputs, got {}",
            input.rank()
        );
        let output_shapes = self.output_shapes(&[input.shape().clone()], PaddingMode::Padding)?;
        let num_vars = Self::num_vars_for_outputs(output_shapes.as_slice())?;
        // Should only be one prover_info for this step
        let layer_commitment = prover.lookup_witness(id)?;
        let logup_inputs = info
            .lookup_ctx
            .create_logup_inputs::<PCS, E>(layer_commitment, &prover.challenge_storage)?;
        let layer_polys = PCS::get_arc_mle_witness_from_commitment(layer_commitment);
        let commitment = PCS::get_pure_commitment(layer_commitment);

        let logup_proof = logup_batch_prove(&logup_inputs, prover.transcript)?;

        // Run the Zerocheck that checks enforces that output does contain the maximum value for the kernel
        let num_threads = optimal_sumcheck_threads(num_vars);
        let either_mles = layer_polys
            .iter()
            .map(|p| Either::Left(p.as_ref()))
            .collect::<Vec<Either<_, _>>>();
        let mut expr_builder =
            VirtualPolynomialsBuilder::<E>::new_with_mles(num_threads, num_vars, either_mles);

        // We reuse the logup point here for the zerocheck challenge
        let lookup_point = &logup_proof.output_claims()[0].point;

        // Compute the identity poly
        let batch_challenge = prover
            .transcript
            .sample_and_append_challenge(b"batch_pooling")
            .elements;

        let beta_eval = compute_betas_eval(lookup_point);
        let beta_poly = MultilinearExtension::from_evaluations_ext_vec(num_vars, beta_eval);

        let last_claim_beta_eval = compute_betas_eval(&last_claim.point);
        let last_claim_beta =
            MultilinearExtension::from_evaluations_ext_vec(num_vars, last_claim_beta_eval.clone());

        let beta_expr = expr_builder.lift(Either::Left(&beta_poly));
        let last_claim_expr = expr_builder.lift(Either::Left(&last_claim_beta));
        let prod_expr = Expression::WitIn(0)
            * Expression::WitIn(1)
            * Expression::WitIn(2)
            * Expression::WitIn(3)
            * beta_expr.clone();
        let sum_expr = (0u16..4).fold(Expression::Constant(Either::Right(E::ZERO)), |acc, j| {
            acc + Expression::Challenge(0, (j + 1) as usize, E::ONE, E::ZERO) * Expression::WitIn(j)
        });
        let diffs_expr = prod_expr + (beta_expr * sum_expr);
        let out_expr =
            Expression::Challenge(0, 5, E::ONE, E::ZERO) * last_claim_expr * Expression::WitIn(4);
        let virtual_poly =
            expr_builder.to_virtual_polys(&[diffs_expr, out_expr], &[batch_challenge]);
        let (proof, sumcheck_state) = IOPProverState::<E>::prove(virtual_poly, prover.transcript);

        // Extract all claims about committed witness polys
        //
        // `collect_raw_challenges` instead of the `.point` field
        // since https://github.com/scroll-tech/ceno/pull/959
        let zerocheck_point = &sumcheck_state.collect_raw_challenges();
        let sumcheck_evals = sumcheck_state.get_mle_flatten_final_evaluations();
        let kernel_size = info.poolinfo.kernel_size * info.poolinfo.kernel_size;

        let output_eval = sumcheck_evals[kernel_size];
        let commit_evals = (
            zerocheck_point.to_vec(),
            sumcheck_evals[..kernel_size + 1].to_vec(),
        );
        prover
            .commit_prover
            .add_witness_claim(id, vec![commit_evals]);

        // Now we must do the samething accumulating evals for the input poly as we fix variables on the input poly.
        // The point length is 2 longer because for now we only support MaxPool2D.

        let padded_input_shape = input.shape();
        let padded_input_row_length_log = ceil_log2(padded_input_shape[2]);
        // We can batch all of the claims for the input poly with 00, 10, 01, 11 fixed into one with random challenges
        let [r1, r2] = [prover
            .transcript
            .sample_and_append_challenge(b"input_batching")
            .elements; 2];

        let one_minus_r1 = E::ONE - r1;
        let one_minus_r2 = E::ONE - r2;
        // To the input claims we add evaluations at both the zerocheck point and lookup point
        // in the order 00, 01, 10, 11. These will be used in conjunction with r1 and r2 by the verifier to link the claims output by the sumcheck and lookup GKR
        // proofs with the claims fed to the same poly verifier.

        let multiplicands = [
            one_minus_r1 * one_minus_r2,
            one_minus_r1 * r2,
            r1 * one_minus_r2,
            r1 * r2,
        ];

        let zc_in_claim = izip!(
            multiplicands.iter(),
            sumcheck_state
                .get_mle_final_evaluations()
                .iter()
                .flatten()
                .take(kernel_size),
        )
        .fold(E::ZERO, |zc_acc, (m, zc)| zc_acc + *m * (output_eval - *zc));

        let point_1 = [
            &[r1],
            &zerocheck_point[..padded_input_row_length_log - 1],
            &[r2],
            &zerocheck_point[padded_input_row_length_log - 1..],
        ]
        .concat();

        let next_claim = Claim {
            point: point_1,
            eval: zc_in_claim,
        };

        // We don't need the last eval of the the sumcheck state as it is the beta poly

        let zerocheck_evals = [
            &sumcheck_state
                .get_mle_flatten_final_evaluations()
                .into_iter()
                .collect::<Vec<_>>()[..kernel_size],
            &[output_eval],
        ]
        .concat();
        // Push the step proof to the list
        prover.push_proof(
            id,
            LayerProof::Pooling(PoolingProof {
                sumcheck: proof,
                lookup: logup_proof,
                zerocheck_evals,
                variable_gap: padded_input_row_length_log - 1,
                commitment,
            }),
        );
        Ok(next_claim)
    }
}

impl PoolingCtx {
    pub fn output_shape(&self, input_shape: &Shape) -> Shape {
        maxpool2d_shape(input_shape)
    }
    pub(crate) fn verify_pooling<E, T: Transcript<E>, PCS: PolynomialCommitmentScheme<E>>(
        &self,
        verifier: &mut Verifier<E, T, PCS>,
        last_claim: &Claim<E>,
        proof: &PoolingProof<E, PCS>,
        shape_step: &ShapeStep,
    ) -> anyhow::Result<Claim<E>>
    where
        E::BaseField: Serialize + DeserializeOwned,
        E: ExtensionField + Serialize + DeserializeOwned,
    {
        // 1. Verify the lookup proof
        let batch_claim = verify_logup_proof_multiple_sizes(&proof.lookup, verifier.transcript)?;

        self.lookup_ctx
            .verify_logup_batch_claim(&batch_claim, &verifier.challenge_storage)?;

        let poly_evals = batch_claim.poly_evals();
        // 2. Verify the sumcheck proof
        let output_shapes =
            self.output_shapes(&shape_step.padded_input_shape, PaddingMode::Padding)?;
        let num_vars = Pooling::num_vars_for_outputs(&output_shapes)?;
        let poly_aux = crate::util::from_mle_list_dimensions(&[vec![num_vars; 5]]);
        let batching_challenge = verifier
            .transcript
            .sample_and_append_challenge(b"batch_pooling")
            .elements;
        let (initial_value_no_last_claim, final_batching_challenge) = poly_evals
            .iter()
            .fold((E::ZERO, batching_challenge), |(acc, comb), &e| {
                (acc + e * comb, comb * batching_challenge)
            });
        let initial_value =
            initial_value_no_last_claim + final_batching_challenge * last_claim.eval;
        let subclaim = IOPVerifierState::<E>::verify(
            initial_value,
            &proof.sumcheck,
            &poly_aux,
            verifier.transcript,
        );

        // Verify the sumcheck output claim and add commitment claims to the commit verifier
        let zerocheck_point = subclaim.point.iter().map(|p| p.elements).collect_vec();
        let beta_eval = identity_eval(batch_claim.point(), &zerocheck_point);

        let last_claim_beta_eval = identity_eval(&last_claim.point, &zerocheck_point);

        let kernel_size = proof.zerocheck_evals.len() - 1;

        let (prod_claim, sum_claim, batch_chal) =
            proof.zerocheck_evals.iter().take(kernel_size).fold(
                (beta_eval, E::ZERO, batching_challenge),
                |(prod_acc, sum_acc, challenge_comb), &eval| {
                    (
                        prod_acc * eval,
                        sum_acc + challenge_comb * eval,
                        challenge_comb * batching_challenge,
                    )
                },
            );

        let output_eval = proof.zerocheck_evals[kernel_size];

        let expected_eval =
            prod_claim + sum_claim * beta_eval + output_eval * last_claim_beta_eval * batch_chal;

        ensure!(
            expected_eval == subclaim.expected_evaluation,
            "Expected pooling zerocheck claim did not equal the verifier claim, expected: {:?}, got: {:?}",
            expected_eval,
            subclaim.expected_evaluation
        );

        let commit_claim = (zerocheck_point.clone(), proof.zerocheck_evals.clone());
        verifier.commit_verifier.add_witness_claim(
            self.node_id,
            proof.commitment.clone(),
            vec![commit_claim],
        );

        // Challenegs used to batch input poly claims together and link them with zerocheck and lookup verification output
        let [r1, r2] = [verifier
            .transcript
            .sample_and_append_challenge(b"input_batching")
            .elements; 2];
        let one_minus_r1 = E::ONE - r1;
        let one_minus_r2 = E::ONE - r2;

        let eval_multiplicands = [
            one_minus_r1 * one_minus_r2,
            one_minus_r1 * r2,
            r1 * one_minus_r2,
            r1 * r2,
        ];

        let zerocheck_point = [
            &[r1],
            &zerocheck_point[..proof.variable_gap],
            &[r2],
            &zerocheck_point[proof.variable_gap..],
        ]
        .concat();

        let zerocheck_input_eval = izip!(
            proof.zerocheck_evals.iter().take(kernel_size),
            eval_multiplicands.iter()
        )
        .fold(E::ZERO, |zerocheck_acc, (&ze, &me)| {
            zerocheck_acc + (output_eval - ze) * me
        });

        let out_claim = Claim {
            point: zerocheck_point,
            eval: zerocheck_input_eval,
        };

        Ok(out_claim)
    }
}

/// Information about a maxpool2d step
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, Copy, PartialOrd, Ord, Hash)]
pub struct Maxpool2D {
    pub kernel_size: usize,
    pub stride: usize,
}

impl Default for Maxpool2D {
    fn default() -> Self {
        Maxpool2D {
            kernel_size: MAXPOOL2D_KERNEL_SIZE,
            stride: MAXPOOL2D_KERNEL_SIZE,
        }
    }
}

impl Maxpool2D {
    pub fn output_shape(&self, input_shape: &Shape) -> Shape {
        maxpool2d_shape(input_shape)
    }

    /// Computes MLE evaluations related to proving Maxpool function.
    /// The outputs of this function are the four polynomials corresponding to the input to the Maxpool, each with two variables fixed
    /// so that PROD (Output - poly_i) == 0 at every evaluation point.
    pub fn compute_polys<E: ExtensionField>(
        &self,
        input: &Tensor<Element>,
    ) -> Result<Vec<Vec<E::BaseField>>> {
        let padded_input = input.pad_next_power_of_two();

        let padded_output = input
            .maxpool2d(self.kernel_size, self.stride)?
            .pad_next_power_of_two();
        let padded_input_shape = padded_input.shape();

        let new_fixed = (0..padded_input_shape[2] << 1)
            .into_par_iter()
            .map(|i| {
                padded_input
                    .get_data()
                    .iter()
                    .skip(i)
                    .step_by(padded_input_shape[2] << 1)
                    .copied()
                    .collect::<Vec<Element>>()
            })
            .collect::<Vec<Vec<Element>>>();

        let new_even = new_fixed
            .iter()
            .step_by(2)
            .cloned()
            .collect::<Vec<Vec<Element>>>();

        let new_odd = new_fixed
            .iter()
            .skip(1)
            .step_by(2)
            .cloned()
            .collect::<Vec<Vec<Element>>>();

        #[allow(clippy::type_complexity)]
        let (even_diff, odd_diff): (Vec<Vec<E::BaseField>>, Vec<Vec<E::BaseField>>) = new_even
            .par_chunks(padded_input_shape[2] >> 1)
            .zip(new_odd.par_chunks(padded_input_shape[2] >> 1))
            .map(|(even_chunk, odd_chunk)| {
                let mut even_merged = even_chunk.to_vec();
                let mut odd_merged = odd_chunk.to_vec();
                for i in (0..ceil_log2(padded_input_shape[2]) - 1).rev() {
                    let mid_point = 1 << i;
                    let (even_low, even_high) = even_merged.split_at(mid_point);
                    let (odd_low, odd_high) = odd_merged.split_at(mid_point);
                    even_merged = even_low
                        .iter()
                        .zip(even_high.iter())
                        .map(|(l, h)| {
                            l.iter()
                                .interleave(h.iter())
                                .copied()
                                .collect::<Vec<Element>>()
                        })
                        .collect::<Vec<Vec<Element>>>();
                    odd_merged = odd_low
                        .iter()
                        .zip(odd_high.iter())
                        .map(|(l, h)| {
                            l.iter()
                                .interleave(h.iter())
                                .copied()
                                .collect::<Vec<Element>>()
                        })
                        .collect::<Vec<Vec<Element>>>();
                }

                izip!(
                    even_merged[0].iter(),
                    odd_merged[0].iter(),
                    padded_output.get_data()
                )
                .map(|(e, o, data)| {
                    let e_field: E = (data - e).to_field();
                    let o_field: E = (data - o).to_field();
                    (e_field.as_bases()[0], o_field.as_bases()[0])
                })
                .unzip::<_, _, Vec<E::BaseField>, Vec<E::BaseField>>()
            })
            .unzip();

        Ok([even_diff, odd_diff].concat())
    }
}

/// Assumes kernel=2, stride=2, padding=0, and dilation=1
/// https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html
pub fn maxpool2d_shape(input_shape: &Shape) -> Shape {
    let stride = 2usize;
    let padding = 0usize;
    let kernel = 2usize;
    let dilation = 1usize;

    let d1 = input_shape[0];
    let d2 = (input_shape[1] + 2 * padding - dilation * (kernel - 1) - 1) / stride + 1;

    Shape::new(vec![d1, d2, d2])
}

pub fn safe_maxpool2d_shape(input_shape: &Shape) -> anyhow::Result<Shape> {
    check_cnn_input(input_shape).context("maxpool2d: invalid input shape")?;
    Ok(maxpool2d_shape(input_shape))
}

#[cfg(test)]
mod tests {
    use crate::{commit::compute_betas_eval, default_transcript, rng_from_env_or_random, to_base};

    use super::*;
    use crate::util::from_mle_list_dimensions;

    use crate::tensor::TensorTypeParam;
    use ark_std::rand::Rng;
    use ceno_p3::field::FieldAlgebra;
    use ff_ext::{FromUniformBytes, GoldilocksExt2};
    use itertools::Itertools;
    use multilinear_extensions::{mle::MultilinearExtension, util::ceil_log2};
    use proptest::prelude::*;
    use sumcheck::structs::{IOPProverState, IOPVerifierState};

    type F = GoldilocksExt2;

    #[test]
    fn test_max_pool_zerocheck() {
        let mut rng = rng_from_env_or_random();
        for _ in 0..50 {
            let random_shape = (0..4)
                .map(|i| {
                    if i < 2 {
                        rng.gen_range(2usize..6)
                    } else {
                        2 * rng.gen_range(2usize..5)
                    }
                })
                .collect::<Shape>();
            let input_data_size = random_shape.product();
            let data = (0..input_data_size)
                .map(|_| rng.gen_range(-128..128))
                .collect::<Vec<Element>>();
            let input = Tensor::<Element>::new(random_shape, data).unwrap();

            let info = Maxpool2D {
                kernel_size: MAXPOOL2D_KERNEL_SIZE,
                stride: MAXPOOL2D_KERNEL_SIZE,
            };

            let output = input.maxpool2d(info.kernel_size, info.stride).unwrap();

            let padded_input = input.pad_next_power_of_two();

            let padded_output = output.pad_next_power_of_two();

            let padded_input_shape = padded_input.shape();

            let num_vars = padded_input.get_data().len().ilog2() as usize;
            let output_num_vars = padded_output.get_data().len().ilog2() as usize;

            let mle = MultilinearExtension::<'_, F>::from_evaluations_vec(
                num_vars,
                to_base::<F, _>(padded_input.get_data()),
            );

            // This should give all possible combinations of fixing the lowest three bits in ascending order

            let fixed_mles = (0..padded_input_shape[3] << 1)
                .map(|i| {
                    let point = (0..ceil_log2(padded_input_shape[3]) + 1)
                        .map(|n| F::from_canonical_u64((i as u64 >> n) & 1))
                        .collect::<Vec<F>>();

                    mle.fix_variables(&point)
                })
                .collect::<Vec<MultilinearExtension<'_, F>>>();
            // f(0,x,0,..) = x * f(0,1,0,...) + (1 - x) * f(0,0,0,...)
            let even_mles = fixed_mles
                .iter()
                .cloned()
                .step_by(2)
                .collect::<Vec<MultilinearExtension<'_, F>>>();
            let odd_mles = fixed_mles
                .iter()
                .skip(1)
                .cloned()
                .step_by(2)
                .collect::<Vec<MultilinearExtension<'_, F>>>();

            let even_merged = even_mles
                .chunks(padded_input_shape[3] >> 1)
                .map(|mle_chunk| {
                    let mut mles_vec = mle_chunk
                        .iter()
                        .map(|m| m.get_ext_field_vec().to_vec())
                        .collect::<Vec<Vec<F>>>();
                    while mles_vec.len() > 1 {
                        let half = mles_vec.len() / 2;

                        mles_vec = mles_vec[..half]
                            .iter()
                            .zip(mles_vec[half..].iter())
                            .map(|(l, h)| {
                                l.iter().interleave(h.iter()).copied().collect::<Vec<F>>()
                            })
                            .collect::<Vec<Vec<F>>>();
                    }

                    MultilinearExtension::<'_, F>::from_evaluations_ext_vec(
                        output_num_vars,
                        mles_vec[0].clone(),
                    )
                })
                .collect::<Vec<MultilinearExtension<'_, F>>>();

            let odd_merged = odd_mles
                .chunks(padded_input_shape[3] >> 1)
                .map(|mle_chunk| {
                    let mut mles_vec = mle_chunk
                        .iter()
                        .map(|m| m.get_ext_field_vec().to_vec())
                        .collect::<Vec<Vec<F>>>();
                    while mles_vec.len() > 1 {
                        let half = mles_vec.len() / 2;

                        mles_vec = mles_vec[..half]
                            .iter()
                            .zip(mles_vec[half..].iter())
                            .map(|(l, h)| {
                                l.iter().interleave(h.iter()).copied().collect::<Vec<F>>()
                            })
                            .collect::<Vec<Vec<F>>>();
                    }

                    MultilinearExtension::<'_, F>::from_evaluations_ext_vec(
                        output_num_vars,
                        mles_vec[0].clone(),
                    )
                })
                .collect::<Vec<MultilinearExtension<'_, F>>>();

            let merged_input_mles = [even_merged, odd_merged].concat();

            let output_mle = MultilinearExtension::<'_, F>::from_evaluations_ext_vec(
                output_num_vars,
                padded_output.to_field::<F>(),
            );

            let num_threads = optimal_sumcheck_threads(output_num_vars);
            let mut expr_builder =
                VirtualPolynomialsBuilder::<F>::new(num_threads, output_num_vars);
            let diff_mles = merged_input_mles
                .iter()
                .map(|in_mle| {
                    MultilinearExtension::<'_, F>::from_evaluations_ext_vec(
                        output_num_vars,
                        in_mle
                            .get_ext_field_vec()
                            .iter()
                            .zip(output_mle.get_ext_field_vec().iter())
                            .map(|(input, output)| *output - *input)
                            .collect::<Vec<F>>(),
                    )
                })
                .collect::<Vec<MultilinearExtension<F>>>();

            let diff_exprs = diff_mles
                .iter()
                .map(|diff_mle| expr_builder.lift(Either::Left(diff_mle)))
                .collect::<Vec<Expression<F>>>();

            (0..1 << output_num_vars).for_each(|j| {
                let values = diff_mles
                    .iter()
                    .map(|mle| mle.get_ext_field_vec()[j])
                    .collect::<Vec<F>>();
                assert_eq!(values.into_iter().product::<F>(), F::ZERO)
            });

            let random_point = (0..output_num_vars)
                .map(|_| <F as FromUniformBytes>::random(&mut rng))
                .collect::<Vec<F>>();

            let beta_evals = compute_betas_eval(&random_point);

            let beta_mle: MultilinearExtension<F> =
                MultilinearExtension::<'_, F>::from_evaluations_ext_vec(
                    output_num_vars,
                    beta_evals,
                );

            let beta_expr = expr_builder.lift(Either::Left(&beta_mle));
            let expr = diff_exprs.into_iter().fold(beta_expr, |acc, d| acc * d);
            let virtual_poly = expr_builder.to_virtual_polys(&[expr], &[]);

            let aux_info = from_mle_list_dimensions(&[vec![output_num_vars; 5]]);

            let mut prover_transcript = default_transcript::<F>();

            #[allow(deprecated)]
            let (proof, state) = IOPProverState::<F>::prove(virtual_poly, &mut prover_transcript);

            let mut verifier_transcript = default_transcript::<F>();

            let subclaim =
                IOPVerifierState::<F>::verify(F::ZERO, &proof, &aux_info, &mut verifier_transcript);

            let point = subclaim
                .point
                .iter()
                .map(|chal| chal.elements)
                .collect::<Vec<F>>();

            let fixed_points = [
                [F::ZERO, F::ZERO],
                [F::ZERO, F::ONE],
                [F::ONE, F::ZERO],
                [F::ONE, F::ONE],
            ]
            .map(|pair| {
                [
                    &[pair[0]],
                    &point[..ceil_log2(padded_input_shape[3]) - 1],
                    &[pair[1]],
                    &point[ceil_log2(padded_input_shape[3]) - 1..],
                ]
                .concat()
            });

            let output_eval = output_mle.evaluate(&point);
            let input_evals = fixed_points
                .iter()
                .map(|p| mle.evaluate(p))
                .collect::<Vec<F>>();

            let eq_eval = beta_mle.evaluate(&point);

            let calc_eval = input_evals
                .iter()
                .map(|ie| output_eval - *ie)
                .product::<F>()
                * eq_eval;

            assert_eq!(calc_eval, subclaim.expected_evaluation);

            // in order output - 00, output - 10, output - 01, output - 11, eq I believe
            let final_mle_evals = state.get_mle_flatten_final_evaluations();

            // let (r1, r2) = (<F as Field>::random(&mut rng), <F as Field>::random(&mut rng));
            let [r1, r2] = [<F as FromUniformBytes>::random(&mut rng); 2];
            let one_minus_r1 = F::ONE - r1;
            let one_minus_r2 = F::ONE - r2;

            let maybe_eval = (output_eval - final_mle_evals[0]) * one_minus_r1 * one_minus_r2
                + (output_eval - final_mle_evals[2]) * one_minus_r1 * r2
                + (output_eval - final_mle_evals[1]) * r1 * one_minus_r2
                + (output_eval - final_mle_evals[3]) * r1 * r2;

            let mle_eval = mle.evaluate(
                &[
                    &[r1],
                    &point[..ceil_log2(padded_input_shape[3]) - 1],
                    &[r2],
                    &point[ceil_log2(padded_input_shape[3]) - 1..],
                ]
                .concat(),
            );

            assert_eq!(mle_eval, maybe_eval);
        }
    }

    #[test]
    fn test_maxpool_f32_simple() {
        let input = Tensor::new(Shape::new(vec![2, 2]), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let expected = input
            .clone()
            .maxpool2d(MAXPOOL2D_KERNEL_SIZE, MAXPOOL2D_KERNEL_SIZE)
            .unwrap();
        let winput = input.as_wrapped();
        let result = Pooling::Maxpool2D(Maxpool2D::default())
            .evaluate::<GoldilocksExt2>(&[&winput])
            .unwrap();
        assert_eq!(result.outputs.len(), 1);
        assert_eq!(&expected, &result.outputs[0].to_native());
    }

    fn proptest_input<T: TensorTypeParam>() -> impl Strategy<Value = Tensor<T>> {
        (2usize..10).prop_flat_map(|size| Tensor::<T>::any(Shape::new(vec![size, size])))
    }

    proptest! {
        #[test]
        fn proptest_maxpool_f32(input in proptest_input::<f32>()) {
            let expected = input.maxpool2d(MAXPOOL2D_KERNEL_SIZE, MAXPOOL2D_KERNEL_SIZE).unwrap();

            let winput = input.as_wrapped();
            let result = Pooling::Maxpool2D(Maxpool2D::default()).evaluate::<GoldilocksExt2>(&[&winput]).unwrap();
            prop_assert_eq!(result.outputs.len(), 1);
            prop_assert_eq!(&expected, &result.outputs[0].to_native());
        }

        #[test]
        fn proptest_maxpool_element(input in proptest_input::<Element>()) {
            let expected = input.maxpool2d(MAXPOOL2D_KERNEL_SIZE, MAXPOOL2D_KERNEL_SIZE).unwrap();

            let winput = input.as_wrapped();
            let result = Pooling::Maxpool2D(Maxpool2D::default()).evaluate::<GoldilocksExt2>(&[&winput]).unwrap();
            prop_assert_eq!(result.outputs.len(), 1);
            prop_assert_eq!(&expected, &result.outputs[0].to_native());
        }
    }
}
