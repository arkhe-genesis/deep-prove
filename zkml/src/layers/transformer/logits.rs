use crate::{
    Claim, Element, Prover, ProverContext, ScalingFactor, ScalingStrategy, Shape, Tensor,
    commit::{compute_betas_eval, identity_eval},
    graph::NodeId,
    iop::{
        ChallengeStorage,
        context::{ContextAux, ShapeStep},
        verifier::Verifier,
    },
    layers::{
        LayerCtx, LayerProof,
        provable::{
            Evaluate, LayerOut, OpInfo, PadOp, ProvableOp, ProveInfo, ProvingData, QuantizeOp,
            QuantizeOutput, VerifiableCtx,
        },
        transformer::mha::eval_zeroifier_mle,
    },
    lookup::{
        context::{LayerLookupContext, LookupWitnessGen, TableType},
        logup_gkr::{
            prover::batch_multiple_sizes_prove,
            structs::{LogUpBatchProof, LogUpInput},
            verifier::verify_logup_proof_multiple_sizes,
        },
    },
    model::Step,
    padding::{PaddingMode, ShapeData, ShapeInfo},
    quantization::{IntoElement, TensorFielder},
    tensor::{TensorTypeParam, WrappedTensor},
    to_bit_sequence_le,
    util::from_mle_list_dimensions,
};
use anyhow::{Result, anyhow, bail, ensure};
use burn::tensor::Shape as BShape;
use ceno_p3::field::FieldAlgebra;
use either::Either;
use ff_ext::ExtensionField;
use itertools::Itertools;
use mpcs::PolynomialCommitmentScheme;
use multilinear_extensions::{Expression, mle::IntoMLE, virtual_polys::VirtualPolynomialsBuilder};
use rayon::prelude::*;
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use std::collections::HashMap;
use sumcheck::{
    structs::{IOPProof, IOPProverState, IOPVerifierState},
    util::optimal_sumcheck_threads,
};
use transcript::Transcript;
use witness::{InstancePaddingStrategy, RowMajorMatrix};

/// The short name used to identify the logits layer.
pub const LOGITS_LAYER: &str = "LGIT";

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ArgmaxData<E> {
    max_values: Vec<Tensor<E>>,
}

#[derive(Clone, Debug)]
pub struct ArgmaxDataNew<E: TensorTypeParam> {
    max_values: Vec<WrappedTensor<E>>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LogitsCtx {
    pub(crate) lookup_ctx: LayerLookupContext,
    node_id: NodeId,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound(serialize = "E: Serialize", deserialize = "E: DeserializeOwned"))]
pub struct LogitsProof<E: ExtensionField, PCS: PolynomialCommitmentScheme<E>> {
    logup_proof: LogUpBatchProof<E>,
    /// Commitment to the MLE of the vector of maximum values
    max_commitment: PCS::Commitment,
    /// Proof of hadamard product sum-check
    hadamard_proof: IOPProof<E>,
    /// Evaluation of the input tensor MLE and max values MLE in this order, got from the hadamard product sum-check
    sumcheck_evals: Vec<E>,
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
        inputs: &[&WrappedTensor<f32>],
    ) -> anyhow::Result<(LayerOut<f32, E>, ArgmaxDataNew<f32>)> {
        ensure!(
            inputs.iter().all(|i| i.rank() >= 2),
            "Argmax is for tensors of rank >= 2",
        );
        match self {
            Logits::Argmax => {
                let (indices, maximums): (Vec<_>, Vec<_>) = inputs
                    .iter()
                    .map(|input| {
                        let binput = (**input).clone().flatten_to_dim_2(0, input.rank() - 2);
                        let (max_bt, indices_bt) = binput.max_dim_with_indices(1);
                        (indices_bt.float(), max_bt)
                    })
                    .unzip();
                Ok((
                    LayerOut::from_vec(indices),
                    ArgmaxDataNew {
                        max_values: maximums,
                    },
                ))
            }
        }
    }

    fn evaluate_with_argmax_data_element<E: ff_ext::ExtensionField>(
        &self,
        inputs: &[&WrappedTensor<Element>],
    ) -> anyhow::Result<(LayerOut<Element, E>, ArgmaxDataNew<Element>)> {
        ensure!(
            inputs.iter().all(|i| i.rank() >= 2),
            "Argmax is for tensors of rank >= 2",
        );
        match self {
            Logits::Argmax => {
                let (indices, maximums): (Vec<_>, Vec<_>) = itertools::process_results(
                    inputs.iter().map(|input| {
                        let shape: Shape = input.unpadded_shape().into();
                        let unpadded_dim_size = shape[input.rank() - 1];
                        let input_shape = input.shape();
                        let input_rank = input.rank();
                        let rows = if input_rank == 2 {
                            input_shape.dims[0]
                        } else {
                            (0..input_rank - 1).map(|d| input_shape.dims[d]).product()
                        };
                        let binput = (*input)
                            .clone()
                            .flatten_to_dim_2(0, input.rank() - 2)
                            .reduce_to_shape(&BShape::from(vec![rows, unpadded_dim_size]))?;
                        let (max_bt, indices_bt) = binput.max_dim_with_indices(1);
                        anyhow::Ok((indices_bt, max_bt))
                    }),
                    |iter| iter.unzip(),
                )?;
                Ok((
                    LayerOut::from_vec(indices),
                    ArgmaxDataNew {
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
        inputs: &[&WrappedTensor<f32>],
    ) -> anyhow::Result<LayerOut<f32, E>> {
        let (output, _) = self.evaluate_with_argmax_data_f32(inputs)?;

        Ok(output)
    }
}

impl Evaluate<Element> for Logits {
    fn evaluate<E: ff_ext::ExtensionField>(
        &self,
        inputs: &[&WrappedTensor<Element>],
    ) -> anyhow::Result<LayerOut<Element, E>> {
        let (output, argmax_data) = self.evaluate_with_argmax_data_element(inputs)?;

        // convert argmax_data to field elements
        let argmax_data = ArgmaxData {
            max_values: argmax_data
                .max_values
                .into_iter()
                .map(|m| {
                    let t: Tensor<E> = Tensor::try_from(m)?.to_fields();
                    anyhow::Ok(t)
                })
                .collect::<anyhow::Result<_>>()?,
        };

        Ok(output.with_proving_data(ProvingData::ArgMax(argmax_data)))
    }
}

impl OpInfo for Logits {
    fn output_shapes(
        &self,
        input_shapes: &[Shape],
        padding_mode: crate::padding::PaddingMode,
    ) -> Result<Vec<Shape>> {
        Ok(Self::output_shapes(input_shapes, padding_mode))
    }

    fn num_outputs(&self, num_inputs: usize) -> Result<usize> {
        Ok(num_inputs)
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

        aux.last_output_shape = self.output_shapes(&aux.last_output_shape, PaddingMode::Padding)?;

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
        _data: &S::AuxData,
        _node_id: NodeId,
        _input_scaling: &[ScalingFactor],
        _unpadded_input_shapes: &[Shape],
        output_scalings: &[ScalingFactor],
        _unpadded_output_shapes: &[Shape],
    ) -> anyhow::Result<QuantizeOutput<Self::QuantizedOp>> {
        Ok(QuantizeOutput::new(self, output_scalings.to_vec()))
    }
}

impl PadOp for Logits {
    fn pad_node(self, si: &mut ShapeInfo) -> anyhow::Result<Self>
    where
        Self: Sized,
    {
        let unpadded_input_shapes = si.unpadded_input_shapes();
        let unpadded_output_shapes =
            self.output_shapes(&unpadded_input_shapes, PaddingMode::NoPadding)?;

        let padded_input_shapes = si.padded_input_shapes();
        let padded_output_shapes =
            self.output_shapes(&padded_input_shapes, PaddingMode::Padding)?;

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

impl<E, PCS> ProvableOp<E, PCS> for Logits
where
    E: ExtensionField,
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
        step_data: &Step<E, Element>,
        prover: &mut Prover<E, T, PCS>,
    ) -> anyhow::Result<Vec<Claim<E>>> {
        ensure!(
            step_data.node_inputs.len() == 1,
            "Expected 1 input tensor for Logits layer, found {}",
            step_data.node_inputs.len()
        );
        let input = step_data.input_tensor_at(0)?;
        let outputs = step_data.output_tensors()?;
        // We need the final dim size to build the less than polynomial
        let unpadded_dim_size =
            step_data.unpadded_input_shapes[0].dim(step_data.unpadded_input_shapes[0].rank() - 1);
        ensure!(
            outputs.len() == 1,
            "Expected 1 output tensor for Logits layer, found {}",
            outputs.len()
        );

        let layer_commitment = prover.lookup_witness(node_id)?;

        let layer_polys = PCS::get_arc_mle_witness_from_commitment(layer_commitment);
        let commitment = PCS::get_pure_commitment(layer_commitment);

        let max_mle = layer_polys[0].as_ref();
        let logup_input = ctx.build_lookup_input(
            max_mle.get_base_field_vec(),
            input.to_fields(),
            &prover.challenge_storage,
            unpadded_dim_size,
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

        let input_claim = Claim::new(diff_claim.point.clone(), diff_claim.eval);

        let input_shape = input.shape();

        let two_inv = E::TWO.inverse();
        let num_cols = E::from_canonical_usize(1 << num_col_vars);
        let sum_eq_point = vec![&two_inv; num_col_vars]
            .into_iter()
            .chain(row_point)
            .cloned()
            .collect_vec();
        let sum_eq_mle = compute_betas_eval(&sum_eq_point).into_mle();

        // Here we build the less than polynomial, for each row it should have 1s at every evaluation less than `unpadded_dim_size` and 0s otherwise
        let base_lt_evals = (0usize..1 << num_col_vars)
            .map(|i| {
                if i < unpadded_dim_size {
                    E::BaseField::ONE
                } else {
                    E::BaseField::ZERO
                }
            })
            .collect::<Vec<E::BaseField>>();
        // `base_lt_evals` is the MLE evals on the boolean hypercube for one row, now we need to repeat this number of rows times
        let lt_mle = vec![base_lt_evals; 1 << num_row_vars].concat().into_mle();
        let input_mle = input.to_fields().into_mle_2d()?;

        let padded_max_evals = max_mle
            .get_base_field_vec()
            .iter()
            .flat_map(|v| vec![*v; 1 << num_col_vars])
            .collect::<Vec<E::BaseField>>()
            .into_mle();
        // build one-hot encoded output matrix
        let mut one_hot_output = vec![E::BaseField::ZERO; input_shape.product()];

        for (i, out) in outputs[0].iter().cloned().enumerate() {
            let out = out as usize;
            let index = i * input_shape.dim(input_shape.rank() - 1) + out;
            one_hot_output[index] = E::BaseField::ONE;
        }

        let one_hot_mle = one_hot_output.into_mle();
        // compute the beta evaluations over `input_claim.point`
        let input_eq_mle = compute_betas_eval(&input_claim.point).into_mle();

        let num_vars = input_mle.num_vars();
        let num_threads = optimal_sumcheck_threads(num_vars);
        let mut expr_builder = VirtualPolynomialsBuilder::<E>::new(num_threads, num_vars);
        let input_expr = expr_builder.lift(Either::Left(&input_mle));
        let padded_max_expr = expr_builder.lift(Either::Left(&padded_max_evals));
        let one_hot_expr = expr_builder.lift(Either::Left(&one_hot_mle));
        let sum_eq_expr = expr_builder.lift(Either::Left(&sum_eq_mle));
        let input_eq_expr = expr_builder.lift(Either::Left(&input_eq_mle));
        let lt_expr = expr_builder.lift(Either::Left(&lt_mle));

        // Now we have to perform a sumcheck linking the range check to the inputs/outputs
        // We show that the selected maximum values did come from the input via the check
        // num_cols.inverse() * padded_max_expr - num_cols * sum_eq_expr * one_hot_expr * input_expr == 0
        // and we show that max_values and input were used to construct the lookup input via
        // input_eq_expr * lt_expr * (padded_max_expr - input_expr) == diff_claim
        let expr_first_part = sum_eq_expr
            * (padded_max_expr.clone()
                - Expression::Constant(Either::Right(num_cols))
                    * one_hot_expr.clone()
                    * input_expr.clone());
        let expr_second_part = input_eq_expr.clone()
            * lt_expr.clone()
            * (padded_max_expr.clone() - input_expr.clone());
        let expr =
            expr_first_part + Expression::Challenge(0, 1, E::ONE, E::ZERO) * expr_second_part;

        // squeeze the challenge to include `input_claim` produced by the lookup in the hadamard product sum-check
        let challenge = Self::squeeze_challenge(prover.transcript, &input_claim);
        let virtual_poly = expr_builder.to_virtual_polys(&[expr], &[challenge]);
        let (hadamard_proof, state) = IOPProverState::<E>::prove(virtual_poly, prover.transcript);

        let point = state.collect_raw_challenges();
        let sumcheck_evals = state.get_mle_flatten_final_evaluations()[..2].to_vec();
        let input_eval = sumcheck_evals[0];
        let max_eval = sumcheck_evals[1];

        prover.commit_prover.add_witness_claim(
            node_id,
            vec![(point[num_col_vars..].to_vec(), vec![max_eval])],
        );

        let final_input_claim = Claim::new(state.collect_raw_challenges(), input_eval);

        let proof = LogitsProof {
            logup_proof: logup_batch_proof,
            max_commitment: commitment,
            hadamard_proof,
            sumcheck_evals,
        };

        prover.push_proof(node_id, LayerProof::Logits(proof));

        Ok(vec![final_input_claim])
    }

    fn gen_lookup_witness(
        &self,
        id: NodeId,
        ctx: &ProverContext<E, PCS>,
        step_data: &Step<E, Element>,
    ) -> anyhow::Result<LookupWitnessGen<E, PCS>> {
        ensure!(
            step_data.node_inputs.len() == 1,
            "Expected 1 input tensor for Logits witness generation, found {}",
            step_data.node_inputs.len()
        );

        let inputs = step_data.input_tensors()?;
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

        let unpadded_dim_size =
            step_data.unpadded_input_shapes[0].dim(step_data.unpadded_input_shapes[0].rank() - 1);

        let merged_diff = input
            .slice_last_dim()
            .zip(max_values.get_data().iter())
            .flat_map(|(row, row_max)| {
                let current_max = row_max.to_element();
                row.iter()
                    .enumerate()
                    .map(|(j, r)| {
                        if j < unpadded_dim_size {
                            current_max - *r
                        } else {
                            0
                        }
                    })
                    .collect::<Vec<Element>>()
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
    ) -> Result<Vec<Shape>> {
        Ok(Logits::output_shapes(input_shapes, padding_mode))
    }

    fn num_outputs(&self, num_inputs: usize) -> Result<usize> {
        Ok(num_inputs)
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

        let input_claim = Claim::new(batch_claim.point().to_vec(), poly_evals[0]);

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
            challenge * input_claim.eval,
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

        // Here we make the less than poly eval
        let unpadded_dim_size =
            shape_step.unpadded_input_shape[0].dim(shape_step.unpadded_input_shape[0].rank() - 1);
        // We subtract 1 from the unpadded dimension size because this function calculates the evaluation of the mle that checks x <= y.
        let dim_size_bits = to_bit_sequence_le(unpadded_dim_size - 1, num_col_vars)
            .map(E::from_canonical_usize)
            .collect::<Vec<E>>();
        let lt_eval = eval_zeroifier_mle(&sumcheck_point[..num_col_vars], &dim_size_bits);

        // get expected evaluation of the claim for the output tensor MLE computed by the sum-check; we have that
        // `subclaim.expected_evaluation = beta_eval * (max_eval - num_cols * proof.input_eval * expected_output_eval) + challenge * input_eq_eval * lt_eval * (max_eval - proof.input_eval)`,
        // so we compute `expected_output_eval` as `(subclaim.expected_evaluation - challenge * input_eq_eval * lt_eval *(max_eval - proof.input_eval) - beta_eval * max_eval)/(-num_cols * beta_eval * proof.input_eval)`
        let sumcheck_evals = &proof.sumcheck_evals;
        let input_eval = sumcheck_evals[0];
        let max_eval = sumcheck_evals[1];
        let expected_output_eval = match (-num_cols * beta_eval * input_eval).try_inverse() {
            Some(inv) => {
                (subclaim.expected_evaluation
                    - challenge * lt_eval * input_eq_eval * (max_eval - input_eval)
                    - beta_eval * max_eval)
                    * inv
            }
            None => {
                let output = &verifier.io.output[0];
                ensure!(
                    output.shape().is_power_of_two(),
                    "Output shape in Logits layer is not a power of 2"
                );
                let num_row_vars = output.shape().dim(0).ilog2() as usize;
                let (column_point, row_point) =
                    Logits::split_claim_point(&sumcheck_point, num_row_vars)?;
                let row_part = compute_betas_eval(row_point).into_iter().sum::<E>();
                let column_part = column_point.iter().map(|p| E::ONE - *p).product::<E>();
                assert!(
                    (subclaim.expected_evaluation
                        - challenge * lt_eval * input_eq_eval * (max_eval - input_eval)
                        - beta_eval * max_eval)
                        == E::ZERO
                );
                row_part * column_part
            }
        };

        Self::verify_output_evaluation(
            verifier,
            Claim::new(sumcheck_point.clone(), expected_output_eval),
            unpadded_dim_size,
        )?;

        // Add the max_values claim to the commitment verifier
        verifier.commit_verifier.add_witness_claim(
            self.node_id,
            proof.max_commitment.clone(),
            vec![(sumcheck_point[num_col_vars..].to_vec(), vec![max_eval])],
        );
        let final_input_claim = Claim::new(sumcheck_point, input_eval);

        Ok(vec![final_input_claim])
    }

    // The verifier of this layer doesn't need to employ any
    // claim about the output tensors. Indeed, the claims about the output tensors are computed
    // by the prover, and are verified directly in `LogitsCtx::verify_output_evaluation` method
    // HOWEVER, we still need to make the transcript advance since for the prover it always
    // tries to derive claims from the transcript, for the output tensor if nothing else.
    // fn compute_model_output_claims<T: Transcript<E>>(

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
        unpadded_dim_size: usize,
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
        let column_beta = compute_betas_eval(column_point);
        let computed_eval =
            output
                .get_data()
                .iter()
                .zip(beta)
                .try_fold(E::ZERO, |sum, (token, b1)| {
                    let token_value = token.to_canonical_u64_vec()[0] as usize;
                    if token_value >= unpadded_dim_size {
                        bail!(
                            "Token value {token_value} exceeds unpadded dimension size {unpadded_dim_size}"
                        );
                    }
                    let selector = b1 * column_beta[token_value];
                    Ok(sum + selector)
                })?;
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
        input: Tensor<E>,
        challenge_storage: &ChallengeStorage<E>,
        unpadded_dim_size: usize,
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
                    .enumerate()
                    .map(|(i, val)| {
                        if i < unpadded_dim_size {
                            max - val.as_bases()[0]
                        } else {
                            E::BaseField::ZERO
                        }
                    })
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
        .map_err(|e| anyhow!("{e:?}"))
    }
}

#[cfg(test)]
mod test {
    use ff_ext::GoldilocksExt2;
    use tenstore::GenStore;

    use super::*;
    use crate::{
        layers::{Layer, provable::Evaluate},
        model::{
            Model,
            test::{prove_model, prove_model_with},
        },
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
        let input = Tensor::new(vec![3, 2].into(), vec![0.0, 1.0, 3.0, 2.0, 4.0, 5.0]).unwrap();
        let logits = Logits::Argmax;

        let out = logits.evaluate::<GoldilocksExt2>(&[&input.as_wrapped()])?;
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

        model.automatic_output_labelling().unwrap();

        prove_model(model, &mut GenStore::default()).unwrap();
    }

    #[test]
    fn test_proven_null_logits_argmax() {
        let seq_len = 13;
        let vocab_size = 17;
        let input_shape = Shape::new(vec![seq_len, vocab_size]);
        let mut model =
            Model::new_from_input_shapes(vec![input_shape.clone()], PaddingMode::NoPadding);

        let _ = model
            .add_consecutive_layer(Layer::Logits(Logits::Argmax), None)
            .unwrap();

        model.automatic_output_labelling().unwrap();
        let inputs = Tensor::zeros(input_shape);

        prove_model_with(model, vec![inputs], &mut GenStore::default()).unwrap();
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
        let input = Tensor::new(vec![2, 3, 4].into(), data).unwrap();
        let logits = Logits::Argmax;
        let out = logits.evaluate::<GoldilocksExt2>(&[&input.as_wrapped()])?;
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
            let input = Tensor::new(shape.clone().into(), data).unwrap();
            let logits = Logits::Argmax;
            let out = logits.evaluate::<GoldilocksExt2>(&[&input.as_wrapped()]).unwrap();
            let indices = out.outputs()[0].get_data();
            prop_assert_eq!(indices.len(), expected.len());
            for (i, idx) in indices.iter().enumerate() { prop_assert!((*idx - expected[i]).abs() < 1e-6); }
        }

        #[test]
        fn prop_logits_argmax_element((shape, data, expected) in logits_input_strategy()) {
            let data_elem: Vec<Element> = data.into_iter().map(|v| v.round_ties_even() as Element).collect();
            let input = Tensor::new(shape.clone().into(), data_elem).unwrap();
            let logits = Logits::Argmax;

            let out = logits.evaluate::<GoldilocksExt2>(&[&input.as_wrapped()]).unwrap();
            let indices = out.outputs()[0].get_data();
            prop_assert_eq!(indices.len(), expected.len());
            for (i, idx) in indices.iter().enumerate() { prop_assert_eq!(*idx as usize, expected[i] as usize); }
        }
    }
}
