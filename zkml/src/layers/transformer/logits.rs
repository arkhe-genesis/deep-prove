use std::collections::HashMap;

use crate::{
    Claim, Element, Prover, ProverContext, ScalingFactor, ScalingStrategy, Shape,
    backend::Backend,
    commit::{compute_betas_eval, identity_eval},
    iop::{
        ChallengeStorage,
        context::{ContextAux, ShapeStep},
        verifier::Verifier,
    },
    layers::{
        LayerCtx, LayerProof,
        provable::{
            NodeId, PadOp, ProvableOp, ProveInfo, ProvingData, QuantizeOp, QuantizeOutput,
            VerifiableCtx,
        },
    },
    lookup::{
        context::{LayerLookupContext, LookupWitnessGen, TableType},
        logup_gkr::{
            prover::batch_multiple_sizes_prove,
            structs::{LogUpBatchProof, LogUpInput},
            verifier::verify_logup_proof_multiple_sizes,
        },
    },
    model::StepData,
    padding::{PaddingMode, ShapeData, ShapeInfo},
    quantization::{IntoElement, TensorFielder},
    to_bit_sequence_le,
    util::from_mle_list_dimensions,
};
use anyhow::{anyhow, ensure};
use ceno_p3::field::FieldAlgebra;
use either::Either;
use ff_ext::ExtensionField;
use itertools::Itertools;
use mpcs::PolynomialCommitmentScheme;
use multilinear_extensions::{Expression, mle::IntoMLE, virtual_polys::VirtualPolynomialsBuilder};
use rayon::prelude::*;
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use sumcheck::{
    structs::{IOPProof, IOPProverState, IOPVerifierState},
    util::optimal_sumcheck_threads,
};
use tenstore::GenStore;
use transcript::Transcript;
use witness::{InstancePaddingStrategy, RowMajorMatrix};

use crate::{
    Tensor,
    layers::provable::{Evaluate, LayerOut, OpInfo},
    tensor::Number,
};

#[derive(Clone, Debug)]
pub struct ArgmaxData<E> {
    max_values: Vec<Tensor<E>>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LogitsCtx {
    lookup_ctx: LayerLookupContext,
    node_id: NodeId,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound(serialize = "E: Serialize", deserialize = "E: DeserializeOwned"))]
pub struct LogitsProof<E: ExtensionField, PCS: PolynomialCommitmentScheme<E>> {
    logup_proof: LogUpBatchProof<E>,
    /// Evaluation of the vector of maximum values
    max_eval: E,
    /// Commitment to the MLE of the vector of maximum values
    max_commitment: PCS::Commitment,
    /// Proof of hadamard product sum-check
    hadamard_proof: IOPProof<E>,
    /// Evaluation of the input tensor MLE got from the hadamard product sum-check
    input_eval: E,
}

impl<E: ExtensionField, PCS: PolynomialCommitmentScheme<E>> LogitsProof<E, PCS> {
    pub(crate) fn get_lookup_data(&self) -> (Vec<E>, Vec<E>) {
        self.logup_proof.fractional_outputs()
    }
    pub(crate) fn write_commitment<T: Transcript<E>>(
        &self,
        transcript: &mut T,
    ) -> anyhow::Result<()> {
        PCS::write_commitment(&self.max_commitment, transcript).map_err(|e| anyhow!("{e:?}"))
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Logits {
    Argmax,
}

impl Logits {
    fn evaluate_with_argmax_data_f32<E: ff_ext::ExtensionField>(
        &self,
        inputs: &[&Tensor<f32>],
        _unpadded_input_shapes: &[Shape],
    ) -> anyhow::Result<(LayerOut<f32, E>, ArgmaxData<f32>)> {
        ensure!(
            inputs.iter().all(|i| i.rank() >= 2),
            "Argmax is for tensors of rank >= 2",
        );
        match self {
            Logits::Argmax => {
                let (indices, maximums): (Vec<_>, Vec<_>) = inputs
                    .iter()
                    .map(|input| {
                        let (flat_data, rows, last_dim) = input.flatten_leading_dims_view();
                        let binput: burn::tensor::Tensor<Backend, 2> =
                            burn::tensor::Tensor::from_data(
                                burn::tensor::TensorData::new(flat_data.to_vec(), [rows, last_dim]),
                                &Default::default(),
                            );
                        let (max_bt, indices_bt) = binput.max_dim_with_indices(1);
                        let indices_vec: Vec<f32> = indices_bt
                            .to_data()
                            .into_vec()
                            .expect("convert indices btensor to vec")
                            .into_iter()
                            .map(|i: i64| i as f32)
                            .collect();
                        let max_vals: Vec<f32> = max_bt
                            .to_data()
                            .into_vec()
                            .expect("convert max btensor to vec");
                        (
                            Tensor::new(Shape::new(vec![rows, 1]), indices_vec),
                            Tensor::new(Shape::new(vec![rows, 1]), max_vals),
                        )
                    })
                    .unzip();
                Ok((
                    LayerOut::from_vec(indices),
                    ArgmaxData {
                        max_values: maximums,
                    },
                ))
            }
        }
    }

    fn evaluate_with_argmax_data_element<E: ff_ext::ExtensionField>(
        &self,
        inputs: &[&Tensor<Element>],
        _unpadded_input_shapes: &[Shape],
    ) -> anyhow::Result<(LayerOut<Element, E>, ArgmaxData<Element>)> {
        ensure!(
            inputs.iter().all(|i| i.rank() >= 2),
            "Argmax is for tensors of rank >= 2",
        );
        match self {
            Logits::Argmax => {
                let (indices, maximums): (Vec<_>, Vec<_>) = inputs
                    .iter()
                    .map(|input| {
                        let (flat_data, rows, last_dim) = input.flatten_leading_dims_view();
                        let binput: burn::tensor::Tensor<Backend, 2, burn::tensor::Int> =
                            burn::tensor::Tensor::from_data(
                                burn::tensor::TensorData::new(flat_data.to_vec(), [rows, last_dim]),
                                &Default::default(),
                            );
                        let (max_bt, indices_bt) = binput.max_dim_with_indices(1);
                        let indices_vec: Vec<Element> = indices_bt
                            .to_data()
                            .into_vec()
                            .expect("convert indices btensor to vec")
                            .into_iter()
                            .map(|i: i64| Element::from_usize(i as usize))
                            .collect();
                        let max_vals: Vec<Element> = max_bt
                            .to_data()
                            .into_vec()
                            .expect("convert max btensor to vec")
                            .into_iter()
                            .map(|v: i64| Element::from_usize(v as usize))
                            .collect();
                        (
                            Tensor::new(Shape::new(vec![rows, 1]), indices_vec),
                            Tensor::new(Shape::new(vec![rows, 1]), max_vals),
                        )
                    })
                    .unzip();
                Ok((
                    LayerOut::from_vec(indices),
                    ArgmaxData {
                        max_values: maximums,
                    },
                ))
            }
        }
    }

    fn output_shapes(input_shapes: &[Shape], _padding_mode: PaddingMode) -> Vec<Shape> {
        input_shapes
            .iter()
            .map(|s| {
                let rows: usize = (0..s.rank() - 1).map(|d| s.dim(d)).product();
                vec![rows, 1].into()
            })
            .collect()
    }

    fn split_claim_point<E: ExtensionField>(
        point: &[E],
        num_row_vars: usize,
    ) -> anyhow::Result<(&[E], &[E])> {
        // row variables are the most significant ones, so we splice between the last `num_row_vars` coordinates
        // and the other ones
        let split_item = point.len() - num_row_vars;
        let row_point = &point[split_item..];
        Ok((&point[..split_item], row_point))
    }

    /// Squeeze from the transcript `t` a challenge necessary to batch the claim about the input tensor
    /// `input_claim` with another claim about the input
    fn squeeze_challenge<E: ExtensionField, T: Transcript<E>>(
        t: &mut T,
        input_claim: &Claim<E>,
    ) -> E {
        // first, we add `input_claim` and `sub_pos_claim` to the transcript
        t.append_field_element_exts(&input_claim.point);
        t.append_field_element_ext(&input_claim.eval);

        t.read_challenge().elements
    }
}

impl Evaluate<f32> for Logits {
    fn evaluate<E: ff_ext::ExtensionField>(
        &self,
        inputs: &[&Tensor<f32>],
        unpadded_input_shapes: &[Shape],
    ) -> anyhow::Result<LayerOut<f32, E>> {
        let (output, _) = self.evaluate_with_argmax_data_f32(inputs, unpadded_input_shapes)?;

        Ok(output)
    }
}

impl Evaluate<Element> for Logits {
    fn evaluate<E: ff_ext::ExtensionField>(
        &self,
        inputs: &[&Tensor<Element>],
        unpadded_input_shapes: &[Shape],
    ) -> anyhow::Result<LayerOut<Element, E>> {
        let (output, argmax_data) =
            self.evaluate_with_argmax_data_element(inputs, unpadded_input_shapes)?;

        // convert argmax_data to field elements
        let argmax_data = ArgmaxData {
            max_values: argmax_data
                .max_values
                .into_iter()
                .map(|m| m.to_fields())
                .collect(),
        };

        Ok(output.with_proving_data(ProvingData::ArgMax(argmax_data)))
    }
}

impl OpInfo for Logits {
    fn output_shapes(
        &self,
        input_shapes: &[Shape],
        padding_mode: crate::padding::PaddingMode,
    ) -> Vec<Shape> {
        Self::output_shapes(input_shapes, padding_mode)
    }

    fn num_outputs(&self, num_inputs: usize) -> usize {
        num_inputs
    }

    fn describe(&self) -> String {
        "Logits::Argmax".to_string()
    }

    fn is_provable(&self) -> bool {
        true
    }
}

impl ProveInfo for Logits {
    fn step_info<E: ExtensionField>(
        &self,
        id: NodeId,
        mut aux: ContextAux,
    ) -> anyhow::Result<(LayerCtx<E>, ContextAux)> {
        ensure!(
            aux.last_output_shape.len() == 1,
            "Expected 1 input shape in ContextAux for Logits layer, found {}",
            aux.last_output_shape.len(),
        );

        aux.last_output_shape = self.output_shapes(&aux.last_output_shape, PaddingMode::Padding);
        aux.tables.insert(TableType::Range);

        let lookup_ctx = LayerLookupContext::new(vec![TableType::Range], vec![1]);
        Ok((
            LayerCtx::Logits(LogitsCtx {
                lookup_ctx,
                node_id: id,
            }),
            aux,
        ))
    }
}

impl QuantizeOp for Logits {
    type QuantizedOp = Logits;

    fn quantize_op<S: ScalingStrategy>(
        self,
        data: &S::AuxData,
        node_id: NodeId,
        input_scaling: &[ScalingFactor],
    ) -> anyhow::Result<QuantizeOutput<Self::QuantizedOp>> {
        // no need to quantize, we just propagate the scaling factors
        let num_inputs = input_scaling.len();
        let num_outputs = self.num_outputs(num_inputs);
        let output_scalings = S::scaling_factors_for_node(data, node_id, num_outputs);

        Ok(QuantizeOutput::new(self, output_scalings))
    }
}

impl PadOp for Logits {
    fn pad_node(self, si: &mut ShapeInfo) -> anyhow::Result<Self>
    where
        Self: Sized,
    {
        let unpadded_input_shapes = si.unpadded_input_shapes();
        let unpadded_output_shapes =
            self.output_shapes(&unpadded_input_shapes, PaddingMode::NoPadding);

        let padded_input_shapes = si.padded_input_shapes();
        let padded_output_shapes = self.output_shapes(&padded_input_shapes, PaddingMode::Padding);

        ensure!(
            si.shapes.iter().all(|s| s.ignore_garbage_pad.is_none()),
            "Unexpected garbage padding to be removed in Logits layer"
        );

        si.shapes = unpadded_output_shapes
            .into_iter()
            .zip(padded_output_shapes)
            .map(|(unpadded_s, padded_s)| ShapeData {
                input_shape_padded: padded_s,
                ignore_garbage_pad: None,
                input_shape_og: unpadded_s,
            })
            .collect();

        Ok(self)
    }
}

impl<E: ExtensionField, PCS> ProvableOp<E, PCS> for Logits
where
    PCS: PolynomialCommitmentScheme<E> + Send + Sync,
    PCS::CommitmentWithWitness: Serialize + DeserializeOwned + Send + Sync,
    PCS::ProverParam: Send + Sync,
{
    type Ctx = LogitsCtx;

    fn prove<T: transcript::Transcript<E>>(
        &self,
        node_id: NodeId,
        ctx: &Self::Ctx,
        _last_claims: Vec<&Claim<E>>,
        step_data: &StepData<E, E>,
        prover: &mut Prover<E, T, PCS>,
        store: &mut GenStore,
    ) -> anyhow::Result<Vec<Claim<E>>> {
        ensure!(
            step_data.node_inputs.len() == 1,
            "Expected 1 input tensor for Logits layer, found {}",
            step_data.node_inputs.len()
        );
        let input = step_data.input_tensor_at(0, store)?;
        let outputs = step_data.output_tensors(store)?;

        ensure!(
            outputs.len() == 1,
            "Expected 1 output tensor for Logits layer, found {}",
            outputs.len()
        );

        let output = outputs[0]
            .get_data()
            .iter()
            .map(|out| out.to_element() as usize)
            .collect_vec();

        let layer_commitment = prover.lookup_witness(node_id)?;

        let layer_polys = PCS::get_arc_mle_witness_from_commitment(layer_commitment);
        let commitment = PCS::get_pure_commitment(layer_commitment);

        let max_mle = layer_polys[0].as_ref();
        let logup_input = ctx.build_lookup_input(
            max_mle.get_base_field_vec(),
            &input,
            &prover.challenge_storage,
        )?;
        let logup_batch_proof = batch_multiple_sizes_prove(&[logup_input], prover.transcript)?;

        // get the claim about the difference between max_values and input data
        let output_claims = logup_batch_proof.output_claims();
        ensure!(
            output_claims.len() == 1,
            "Expected 1 claim from logup proof in Logits layer, found {}",
            output_claims.len(),
        );
        let diff_claim = &output_claims[0];

        // evaluate max_values MLE on the same point of `diff_claim`
        // we need to extract the row-related coordinates from `diff_claim.point`
        let num_row_vars = max_mle.num_vars();
        let num_col_vars = diff_claim.point.len() - num_row_vars;
        let (_, row_point) = Self::split_claim_point(&diff_claim.point, num_row_vars)?;

        let max_eval = max_mle.evaluate(row_point);

        prover
            .commit_prover
            .add_witness_claim(node_id, vec![(row_point.to_vec(), vec![max_eval])]);

        let input_claim = Claim::new(diff_claim.point.clone(), max_eval - diff_claim.eval);

        let input_shape = input.shape();

        let two_inv = E::TWO.inverse();
        let num_cols = E::from_canonical_usize(1 << num_col_vars);
        let sum_eq_point = vec![&two_inv; num_col_vars]
            .into_iter()
            .chain(row_point)
            .cloned()
            .collect_vec();
        let sum_eq_mle = compute_betas_eval(&sum_eq_point).into_mle();

        let input_mle = input.to_mle_2d();

        // build one-hot encoded output matrix
        let mut one_hot_output = vec![E::BaseField::ZERO; input_shape.product()];
        output.iter().enumerate().for_each(|(i, out)| {
            let index = i * input_shape.dim(input_shape.rank() - 1) + out;
            one_hot_output[index] = E::BaseField::ONE;
        });
        let one_hot_mle = one_hot_output.into_mle();
        // compute the beta evaluations over `input_claim.point`
        let input_eq_mle = compute_betas_eval(&input_claim.point).into_mle();

        let num_vars = input_mle.num_vars();
        let num_threads = optimal_sumcheck_threads(num_vars);
        let mut expr_builder = VirtualPolynomialsBuilder::<E>::new(num_threads, num_vars);
        let input_expr = expr_builder.lift(Either::Left(&input_mle));
        let one_hot_expr = expr_builder.lift(Either::Left(&one_hot_mle));
        let sum_eq_expr = expr_builder.lift(Either::Left(&sum_eq_mle));
        let input_eq_expr = expr_builder.lift(Either::Left(&input_eq_mle));

        let expr = input_expr
            * (sum_eq_expr * one_hot_expr * Expression::Constant(Either::Right(num_cols))
                + Expression::Challenge(0, 1, E::ONE, E::ZERO) * input_eq_expr);
        // squeeze the challenge to include `input_claim` produced by the lookup in the hadamard product sum-check
        let challenge = Self::squeeze_challenge(prover.transcript, &input_claim);
        let virtual_poly = expr_builder.to_virtual_polys(&[expr], &[challenge]);
        let (hadamard_proof, state) = IOPProverState::<E>::prove(virtual_poly, prover.transcript);

        let input_eval = state.get_mle_flatten_final_evaluations()[0];

        let final_input_claim = Claim::new(state.collect_raw_challenges(), input_eval);

        let proof = LogitsProof {
            logup_proof: logup_batch_proof,
            max_eval,
            max_commitment: commitment,
            hadamard_proof,
            input_eval,
        };

        prover.push_proof(node_id, LayerProof::Logits(proof));

        Ok(vec![final_input_claim])
    }

    fn gen_lookup_witness(
        &self,
        id: NodeId,
        ctx: &ProverContext<E, PCS>,
        step_data: &StepData<Element, E>,
        store: &mut GenStore,
    ) -> anyhow::Result<LookupWitnessGen<E, PCS>> {
        ensure!(
            step_data.node_inputs.len() == 1,
            "Expected 1 input tensor for Logits witness generation, found {}",
            step_data.node_inputs.len()
        );

        let inputs = step_data.input_tensors(store)?;
        let input = &inputs[0];

        ensure!(
            matches!(self, Logits::Argmax),
            "Only Argmax is currently supported in Logits layer"
        );

        let argmax_data = step_data.node_outputs.try_argmax_data().ok_or(anyhow!(
            "Argmax data not found when generating witness for Logits layer"
        ))?;

        ensure!(
            argmax_data.max_values.len() == 1,
            "Expected 1 tensor of max values for Logits argmax, found {}",
            argmax_data.max_values.len(),
        );

        let max_values = &argmax_data.max_values[0];

        let input_shape = input.shape();
        // For rank>2 inputs we flattened all leading dimensions when computing argmax which is enforced here
        let expected_rows: usize = (0..input_shape.rank() - 1)
            .map(|d| input_shape.dim(d))
            .product();
        ensure!(
            max_values.shape().dim(0) == expected_rows,
            "Incompatible shapes between max values tensor (rows={}) and flattened input rows (expected_rows={}) for input shape {:?}",
            max_values.shape().dim(0),
            expected_rows,
            input.shape(),
        );

        let merged_diff = input
            .get_data()
            .into_par_iter()
            .enumerate()
            .map(|(i, input)| {
                let row_index = i / input_shape.dim(input_shape.len() - 1);
                let current_max = max_values.get_data()[row_index];
                let max_element = current_max.to_element();
                max_element - input
            })
            .collect::<Vec<Element>>();
        let element_count = merged_diff.iter().fold(HashMap::new(), |mut acc, diff| {
            *acc.entry(*diff).or_default() += 1;
            acc
        });

        // commit to max values
        let commit_data = max_values
            .get_data()
            .iter()
            .map(|v| v.as_bases()[0])
            .collect::<Vec<E::BaseField>>();

        let rmm = RowMajorMatrix::<E::BaseField>::new_by_inner_matrix(
            ceno_p3::matrix::dense::DenseMatrix::new(commit_data, 1),
            InstancePaddingStrategy::Default,
        );
        let layer_commitment = ctx.commitment_ctx.batch_commit(vec![rmm])?;

        let mut gen = LookupWitnessGen::<E, PCS>::default();
        gen.insert_logup_witness(id, layer_commitment);
        gen.insert_element_count(TableType::Range, element_count);

        Ok(gen)
    }
}

impl OpInfo for LogitsCtx {
    fn output_shapes(
        &self,
        input_shapes: &[Shape],
        padding_mode: crate::padding::PaddingMode,
    ) -> Vec<Shape> {
        Logits::output_shapes(input_shapes, padding_mode)
    }

    fn num_outputs(&self, num_inputs: usize) -> usize {
        num_inputs
    }

    fn describe(&self) -> String {
        "Logit::Argmax".to_string()
    }

    fn is_provable(&self) -> bool {
        true
    }
}

impl<E: ExtensionField, PCS: PolynomialCommitmentScheme<E>> VerifiableCtx<E, PCS> for LogitsCtx {
    type Proof = LogitsProof<E, PCS>;

    fn verify<T: Transcript<E>>(
        &self,
        proof: &Self::Proof,
        _last_claims: &[&Claim<E>],
        verifier: &mut Verifier<E, T, PCS>,
        shape_step: &ShapeStep,
    ) -> anyhow::Result<Vec<Claim<E>>> {
        let batch_claim =
            verify_logup_proof_multiple_sizes(&proof.logup_proof, verifier.transcript)?;

        self.lookup_ctx
            .verify_logup_batch_claim(&batch_claim, &verifier.challenge_storage)?;

        let poly_evals = batch_claim.poly_evals();

        ensure!(
            poly_evals.len() == 1,
            "Expected 1 claim for logup when verifying Logis layer, found {}",
            poly_evals.len(),
        );

        ensure!(
            shape_step.padded_input_shape.len() == 1,
            "Expected 1 padded input shape when verifying Logits layer, found {}",
            shape_step.padded_input_shape.len(),
        );

        let input_shape = &shape_step.padded_input_shape[0];
        let num_row_vars = (0..input_shape.rank() - 1)
            .map(|d| input_shape.dim(d))
            .product::<usize>()
            .ilog2() as usize;
        let num_col_vars = input_shape.dim(input_shape.rank() - 1).ilog2() as usize;
        let (_, row_point) = Logits::split_claim_point(batch_claim.point(), num_row_vars)?;

        let input_claim = Claim::new(batch_claim.point().to_vec(), proof.max_eval - poly_evals[0]);

        verifier.commit_verifier.add_witness_claim(
            self.node_id,
            proof.max_commitment.clone(),
            vec![(row_point.to_vec(), vec![proof.max_eval])],
        );

        let two_inv = E::TWO.inverse();
        let num_cols = E::from_canonical_usize(1 << num_col_vars);
        let sum_eq_point = vec![two_inv; num_col_vars]
            .iter()
            .chain(row_point)
            .cloned()
            .collect_vec();

        let challenge = Logits::squeeze_challenge(verifier.transcript, &input_claim);

        // verify hadamard product sum-check
        let input_num_vars = shape_step.padded_input_shape[0].num_vars().iter().sum();
        let hadamard_poly_aux =
            from_mle_list_dimensions(&[vec![input_num_vars, input_num_vars, input_num_vars]]);
        let subclaim = IOPVerifierState::verify(
            proof.max_eval + challenge * input_claim.eval,
            &proof.hadamard_proof,
            &hadamard_poly_aux,
            verifier.transcript,
        );

        let sumcheck_point = subclaim
            .point
            .iter()
            .map(|p| p.elements)
            .collect::<Vec<_>>();
        let beta_eval = identity_eval(&sum_eq_point, &sumcheck_point);
        let input_eq_eval = identity_eval(&input_claim.point, &sumcheck_point);

        // get expected evaluation of the claim for the output tensor MLE computed by the sum-check; we have that
        // `subclaim.expected_evaluation = num_cols*beta_eval*proof.input_eval*expected_output_eval + challenge*proof.input_eval*input_eq_eval`,
        // so we compute `expected_output_eval` as `(subclaim.expected_evaluation - challenge*proof.input_eval*input_eq_eval)/(num_cols*beta_eval*proof.input_eval)`

        let expected_output_eval = (subclaim.expected_evaluation
            - challenge * proof.input_eval * input_eq_eval)
            * (num_cols * beta_eval * proof.input_eval).inverse();

        Self::verify_output_evaluation(
            verifier,
            Claim::new(sumcheck_point.clone(), expected_output_eval),
        )?;
        let final_input_claim = Claim::new(sumcheck_point, proof.input_eval);

        Ok(vec![final_input_claim])
    }

    fn compute_model_output_claims<T: Transcript<E>>(
        &self,
        _transcript: &mut T,
        outputs: &[&Tensor<E>],
    ) -> Vec<Claim<E>> {
        // simply return default claims, as the verifier of this layer doesn't need to employ any
        // claim about the output tensors. Indeed, the claims about the output tensors are computed
        // by the prover, and are verified directly in `LogitsCtx::verify_output_evaluation` method
        vec![Claim::default(); outputs.len()]
    }

    fn write_proof_to_transcript<T: Transcript<E>>(
        &self,
        proof: &Self::Proof,
        transcript: &mut T,
    ) -> anyhow::Result<()> {
        proof.write_commitment(transcript)
    }
}

impl LogitsCtx {
    fn verify_output_evaluation<
        E: ExtensionField,
        T: Transcript<E>,
        PCS: PolynomialCommitmentScheme<E>,
    >(
        verifier: &mut Verifier<E, T, PCS>,
        output_claim: Claim<E>,
    ) -> anyhow::Result<()> {
        ensure!(
            verifier.io.output.len() == 1,
            "Expected 1 output tensor when verifying logits layer output claim, found {}",
            verifier.io.output.len(),
        );
        let output = &verifier.io.output[0];
        ensure!(
            output.shape().is_power_of_two(),
            "Output shape in Logits layer is not a power of 2"
        );
        let num_row_vars = output.shape().dim(0).ilog2() as usize;
        let (column_point, row_point) =
            Logits::split_claim_point(&output_claim.point, num_row_vars)?;
        let beta = compute_betas_eval(row_point);
        let computed_eval = output
            .get_data()
            .iter()
            .zip(beta)
            .fold(E::ZERO, |sum, (token, b1)| {
                let token_value = token.to_canonical_u64_vec()[0] as usize;
                let le_bits = to_bit_sequence_le(token_value, column_point.len())
                    .map(|b| E::from_canonical_usize(b))
                    .collect_vec();
                let selector = b1 * identity_eval(column_point, &le_bits);
                sum + selector
            });
        ensure!(
            computed_eval == output_claim.eval,
            "Output claim evaluation check failed for Logits layer: Expected {}, found {}",
            computed_eval,
            output_claim.eval,
        );
        Ok(())
    }

    fn build_lookup_input<E: ExtensionField>(
        &self,
        max_evals: &[E::BaseField],
        input: &Tensor<E>,
        challenge_storage: &ChallengeStorage<E>,
    ) -> anyhow::Result<LogUpInput<E>> {
        let input_shape = input.shape();
        let last_dim = input_shape.dim(input_shape.len() - 1);

        let column_evals = input
            .get_data()
            .par_chunks(last_dim)
            .zip(max_evals.par_iter())
            .map(|(chunk, &max)| {
                chunk
                    .iter()
                    .map(|val| max - val.as_bases()[0])
                    .collect::<Vec<E::BaseField>>()
            })
            .collect::<Vec<Vec<E::BaseField>>>();

        let (constant_challenge, column_separation_challenge) = challenge_storage
            .get_challenges_by_name(&self.lookup_ctx.tables[0].name())
            .ok_or(anyhow!(
                "No challenges found for Table Type: {}, cannot prove Logits ArgMax",
                self.lookup_ctx.tables[0].name()
            ))?;
        LogUpInput::<E>::new_lookup(
            vec![column_evals.concat()],
            constant_challenge,
            column_separation_challenge,
            1,
        )
        .map_err(|e| anyhow!("{:?}", e))
    }
}

#[cfg(test)]
mod test {
    use ff_ext::GoldilocksExt2;

    use super::*;
    use crate::{
        layers::{Layer, provable::Evaluate},
        model::{Model, test::prove_model},
        tensor::Tensor,
    };
    use proptest::prelude::*;
    use rand::{Rng, SeedableRng, rngs::StdRng};

    // Strategy to generate random shapes (rank 2 or 3) and data with unique maxima per row
    fn logits_input_strategy() -> impl Strategy<Value = (Vec<usize>, Vec<f32>, Vec<f32>)> {
        (1usize..4, 1usize..4, 2usize..9, any::<u64>(), any::<bool>()).prop_flat_map(
            |(d1, d2, last_dim, seed, rank3)| {
                let mut rng = StdRng::seed_from_u64(seed);
                let shape: Vec<usize> = if rank3 {
                    vec![d1, d2, last_dim]
                } else {
                    vec![d1, last_dim]
                };
                let rows: usize = shape[..shape.len() - 1].iter().product();
                let mut data = vec![0f32; rows * last_dim];
                let mut expected = Vec::with_capacity(rows);
                for r in 0..rows {
                    for c in 0..last_dim {
                        data[r * last_dim + c] = (rng.random::<f32>() * 10.0).floor();
                    }
                    let arg_idx = rng.random_range(0..last_dim);
                    data[r * last_dim + arg_idx] += 1000.0;
                    expected.push(arg_idx as f32);
                }
                Just((shape, data, expected))
            },
        )
    }

    #[test]
    fn test_logits_argmax() -> anyhow::Result<()> {
        let input = Tensor::new(vec![3, 2].into(), vec![0.0, 1.0, 3.0, 2.0, 4.0, 5.0]);
        let logits = Logits::Argmax;
        let out = logits.evaluate::<GoldilocksExt2>(&[&input], &[])?;
        // first slice is [0,1] so argmax here is 1
        // second slice is [3,2] so argmax here is 0
        // the last dimension is [4,5] so argmax here is 1
        assert_eq!(out.outputs()[0].get_data(), vec![1.0, 0.0, 1.0]);
        Ok(())
    }

    #[test]
    fn test_proven_logits_argmax() {
        let seq_len = 13;
        let vocab_size = 17;
        let input_shape = Shape::new(vec![seq_len, vocab_size]);
        let mut model = Model::new_from_input_shapes(vec![input_shape], PaddingMode::NoPadding);

        let _ = model
            .add_consecutive_layer(Layer::Logits(Logits::Argmax), None)
            .unwrap();

        model.route_output(None).unwrap();

        prove_model(model, &mut GenStore::default()).unwrap();
    }

    #[test]
    fn test_logits_argmax_higher_rank() -> anyhow::Result<()> {
        // Original shape: [2, 3, 4]. All leading dims (2 * 3) are flattened -> rows = 6; last_dim = 4.
        // Expected flattened output indices tensor (shape [6,1])
        let mut data = Vec::new();
        for r in 0..6 {
            for c in 0..4 {
                let val = if c == r % 4 { 10.0 } else { c as f32 }; // ensure unique max
                data.push(val);
            }
        }
        let input = Tensor::new(vec![2, 3, 4].into(), data);
        let logits = Logits::Argmax;
        let out = logits.evaluate::<GoldilocksExt2>(&[&input], &[])?;
        let indices = out.outputs()[0].get_data();
        assert_eq!(indices.len(), 6);
        for (r, &idx) in indices.iter().enumerate() {
            assert_eq!(idx as usize, r % 4);
        }
        Ok(())
    }

    proptest! {
        #[test]
        fn prop_logits_argmax_f32((shape, data, expected) in logits_input_strategy()) {
            //  rows = product of all leading dims (shape[..rank-1]).
            //  Each row has a unique argmax and the output tensor shape is [rows,1].
            let input = Tensor::new(shape.clone().into(), data);
            let logits = Logits::Argmax;
            let out = logits.evaluate::<GoldilocksExt2>(&[&input], &[]).unwrap();
            let indices = out.outputs()[0].get_data();
            prop_assert_eq!(indices.len(), expected.len());
            for (i, idx) in indices.iter().enumerate() { prop_assert!((*idx - expected[i]).abs() < 1e-6); }
        }
    }

    proptest! {
        #[test]
        fn prop_logits_argmax_element((shape, data, expected) in logits_input_strategy()) {
            let data_elem: Vec<Element> = data.into_iter().map(|v| v.round() as Element).collect();
            let input = Tensor::new(shape.clone().into(), data_elem);
            let logits = Logits::Argmax;
            let out = logits.evaluate::<GoldilocksExt2>(&[&input], &[]).unwrap();
            let indices = out.outputs()[0].get_data();
            prop_assert_eq!(indices.len(), expected.len());
            for (i, idx) in indices.iter().enumerate() { prop_assert_eq!(*idx as usize, expected[i] as usize); }
        }
    }
}
