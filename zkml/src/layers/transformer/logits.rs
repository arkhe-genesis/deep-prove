use crate::{
    Claim, Element, Number, Prover, ProverContext, ScalingFactor, ScalingStrategy, Shape, Tensor,
    eval_zeroifier_mle,
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
            QuantizeOutput, Splittable, VerifiableCtx,
        },
        split::SplitLayer,
    },
    lookup::{
        context::LookupWitnessGen,
        logup_gkr::{
            prover::new_batch_multiple_sizes_prove,
            structs::{LogUpBatchProof, LogUpInput, LogUpVerifierInstance},
            verifier::new_verify_logup_proof_multiple_sizes,
        },
        table::{SHIFT_CHECK_TABLE_BIT_SIZE, Table},
    },
    model::{Step, trace::split_tensors},
    padding::{PaddingMode, ShapeData, ShapeInfo},
    poly_commit::{
        identity_eval,
        verifier::{VerifierClaim, VerifierCommitment},
    },
    quantization::{ToElement, ToField},
    tensor::{TensorHandle, WrappedTensor},
    to_bit_sequence_le,
    util::from_mle_list_dimensions,
};
use anyhow::{Result, anyhow, bail, ensure};
use ark_ff::PrimeField;
use burn::tensor::Shape as BShape;
use dp_crypto::{
    ArcMultilinearExtension, Expression, IntoMLE,
    arkyper::{
        CommitmentScheme,
        transcript::{AppendToTranscript, Transcript},
    },
    poly::eq::evals,
    structs::{IOPProof, IOPProverState, IOPVerifierState},
    util::{ceil_log2, optimal_sumcheck_threads},
    virtual_polys::VirtualPolynomialsBuilder,
};
use either::Either;
use itertools::Itertools;
use lazy_static::lazy_static;
use parking_lot::RwLock;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::{collections::HashMap, iter::once};
use tenstore::StorageKey;
use tract_onnx::tract_hir::internal::num_integer::div_ceil;

/// The short name used to identify the logits layer.
pub const LOGITS_LAYER: &str = "LGIT";

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ArgmaxData {
    max_values: Vec<WrappedTensor<Element>>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ArgmaxHandle {
    pub(crate) max_values: Vec<TensorHandle<Element>>,
}
impl ArgmaxHandle {
    pub(crate) fn new(
        storage_key: StorageKey<Vec<Element>>,
        argmax_data: ArgmaxData,
        store: tenstore::GenStore,
    ) -> Self {
        let max_values = argmax_data
            .max_values
            .into_iter()
            .enumerate()
            .map(|(i, tensor)| {
                let key = StorageKey::new(format!("{}-argmax-{}", storage_key.id(), i));
                TensorHandle::from_wrapped_tensor(key, store.clone(), tensor)
            })
            .collect();

        Self { max_values }
    }

    pub(crate) fn attach_store(&mut self, store: tenstore::GenStore) {
        self.max_values
            .iter_mut()
            .for_each(|handle| handle.attach_store(store.clone()));
    }

    pub(crate) fn isolate(&self) -> ArgmaxHandle {
        Self {
            max_values: self
                .max_values
                .iter()
                .map(|handle| handle.isolate())
                .collect(),
        }
    }

    pub(crate) fn split_tensors(&self, num_chunks: usize) -> anyhow::Result<Vec<Self>> {
        let num_handles = self.max_values.len();
        let split_layer = SplitLayer {
            unpadded_input_shapes: self
                .max_values
                .iter()
                .map(|handle| handle.unpadded_shape().clone())
                .collect_vec(),
            num_chunks: vec![num_chunks; num_handles],
        };
        let (unpadded_output_shapes, layer_out) = split_tensors(&self.max_values, &split_layer)?;
        let mut new_handles = vec![Self { max_values: vec![] }; num_chunks];
        for (i, (out_tensor, out_shape)) in layer_out
            .outputs
            .into_iter()
            .zip(unpadded_output_shapes)
            .enumerate()
        {
            let chunk_number = i % num_chunks;
            let original_handle = &self.max_values[i / num_chunks];
            let storage_key = StorageKey::new(format!(
                "{}-chunk-{}",
                original_handle.storage_key().id(),
                chunk_number
            ));
            let new_handle = TensorHandle::from_wrapped_tensor_with_unpadded_shape(
                storage_key,
                original_handle.store().clone(),
                out_tensor,
                out_shape,
            );
            new_handles[chunk_number].max_values.push(new_handle);
        }
        Ok(new_handles)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LogitsCtx {
    range_check_chunks: usize,
}

impl LogitsCtx {
    /// Getter for the lookup tables used in this layer.
    pub fn lookup_tables(&self) -> Vec<Table> {
        vec![Table::new_shift_check()]
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound(
    serialize = "F: ark_serialize::CanonicalSerialize",
    deserialize = "F: ark_serialize::CanonicalDeserialize"
))]
pub struct LogitsProof<F: PrimeField, PCS: CommitmentScheme> {
    logup_proof: LogUpBatchProof<F>,
    /// Commitment to the MLE of the vector of maximum values
    commitments: Vec<VerifierCommitment<PCS>>,
    /// Proof of hadamard product sum-check
    hadamard_proof: IOPProof<F>,
    /// Evaluation of the input tensor MLE and max values MLE in this order, got from the hadamard product sum-check
    #[serde(with = "dp_crypto::serialization")]
    sumcheck_evals: Vec<F>,
}

impl<F: PrimeField, PCS: CommitmentScheme> LogitsProof<F, PCS> {
    pub(crate) fn write_commitment<T: Transcript>(&self, transcript: &mut T) -> anyhow::Result<()> {
        self.commitments
            .iter()
            .for_each(|comm| comm.append_to_transcript(transcript));
        Ok(())
    }
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizedArgMaxData {
    input_bit_size: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Logits {
    Argmax(Option<QuantizedArgMaxData>),
}

impl Logits {
    pub fn new_argmax() -> Self {
        Logits::Argmax(None)
    }

    fn evaluate_with_argmax_data_f32(
        &self,
        inputs: &[&WrappedTensor<f32>],
    ) -> anyhow::Result<(LayerOut<f32>, Vec<WrappedTensor<f32>>)> {
        ensure!(
            inputs.iter().all(|i| i.rank() >= 2),
            "Argmax is for tensors of rank >= 2",
        );
        match self {
            Logits::Argmax(_) => {
                let (indices, maximums): (Vec<_>, Vec<_>) = inputs
                    .iter()
                    .map(|input| {
                        let binput = (**input).clone().flatten_to_dim_2(0, input.rank() - 2);
                        let (max_bt, indices_bt) = binput.max_dim_with_indices(1);
                        (indices_bt.float(), max_bt)
                    })
                    .unzip();
                Ok((LayerOut::from_vec(indices), maximums))
            }
        }
    }

    fn evaluate_with_argmax_data_element(
        &self,
        inputs: &[&WrappedTensor<Element>],
    ) -> anyhow::Result<(LayerOut<Element>, Vec<WrappedTensor<Element>>)> {
        ensure!(
            inputs.iter().all(|i| i.rank() >= 2),
            "Argmax is for tensors of rank >= 2",
        );
        match self {
            Logits::Argmax(_) => {
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
                Ok((LayerOut::from_vec(indices), maximums))
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

    fn split_claim_point<F: PrimeField>(
        point: &[F],
        num_row_vars: usize,
    ) -> anyhow::Result<(&[F], &[F])> {
        // row variables are the most significant ones, so we splice between the last `num_row_vars` coordinates
        // and the other ones
        let split_item = point.len() - num_row_vars;
        let row_point = &point[split_item..];
        Ok((&point[..split_item], row_point))
    }

    /// Squeeze from the transcript `t` a challenge necessary to batch the claim about the input tensor
    /// `input_claim` with another claim about the input
    fn squeeze_challenge<F: PrimeField, T: Transcript>(t: &mut T, input_claim: &Claim<F>) -> F {
        // first, we add `input_claim` and `sub_pos_claim` to the transcript
        t.append_scalars::<F>(&input_claim.point);
        t.append_scalars::<F>(&[input_claim.eval]);

        t.challenge_scalar()
    }

    fn range_check_chunks(&self) -> usize {
        match self {
            Logits::Argmax(Some(data)) => {
                let bit_size = data.input_bit_size + 1; // +1 since this is the upper bound for max - input values
                // ensure that `bit_size` is at least `quantization::BIT_LEN`
                let bit_size = bit_size.max(SHIFT_CHECK_TABLE_BIT_SIZE);
                bit_size.div_ceil(SHIFT_CHECK_TABLE_BIT_SIZE)
            }
            Logits::Argmax(None) => {
                unimplemented!("Range check chunks not implemented for unquantized logits")
            }
        }
    }

    fn build_lookup_input<F: PrimeField>(
        &self,
        committed_polys: &[ArcMultilinearExtension<F>],
        challenge_storage: &ChallengeStorage<F>,
        input: Tensor<F>,
        unpadded_dim_size: usize,
    ) -> anyhow::Result<LogUpInput<F>> {
        let table = Table::new_shift_check();
        let column_evals = if committed_polys.len() == 1 {
            let input_shape = input.shape();
            let last_dim = input_shape.dim(input_shape.len() - 1);
            vec![
                input
                    .data()
                    .par_chunks(last_dim)
                    .zip(committed_polys[0].evals_ref().par_iter())
                    .map(|(chunk, &max)| {
                        chunk
                            .iter()
                            .enumerate()
                            .map(|(i, val)| {
                                if i < unpadded_dim_size {
                                    max - val
                                } else {
                                    F::ZERO
                                }
                            })
                            .collect::<Vec<F>>()
                    })
                    .collect::<Vec<Vec<F>>>()
                    .concat(),
            ]
        } else {
            committed_polys[1..]
                .iter()
                .map(|mle| mle.evals())
                .collect::<Vec<Vec<F>>>()
        };

        let (constant_challenge, column_separation_challenge) = challenge_storage
            .get_challenges_by_name(&table.name())
            .ok_or(anyhow!(
                "No challenges found for Table Type: {}, cannot prove Logits ArgMax",
                table.name()
            ))?;
        LogUpInput::<F>::new_lookup(
            column_evals,
            constant_challenge,
            column_separation_challenge,
            table.num_columns(),
        )
        .map_err(|e| anyhow!("{e:?}"))
    }

    fn recombine_range_check_claims<F: PrimeField>(
        logup_claims: &[Claim<F>],
    ) -> anyhow::Result<Claim<F>> {
        ensure!(!logup_claims.is_empty());
        let point = logup_claims[0].point.clone();
        let multiplier = F::from(1u64 << SHIFT_CHECK_TABLE_BIT_SIZE);
        let eval = logup_claims
            .iter()
            .try_fold((F::ZERO, F::ONE), |(eval, factor), claim| {
                ensure!(
                    claim.point == point,
                    "All logup claims must be on the same point to recombine"
                );
                Ok((eval + factor * claim.eval, factor * multiplier))
            })?
            .0;
        Ok(Claim { point, eval })
    }
}

impl Evaluate<f32> for Logits {
    fn evaluate(&self, inputs: &[&WrappedTensor<f32>]) -> anyhow::Result<LayerOut<f32>> {
        let (output, _) = self.evaluate_with_argmax_data_f32(inputs)?;

        Ok(output)
    }
}

impl Evaluate<Element> for Logits {
    fn evaluate(&self, inputs: &[&WrappedTensor<Element>]) -> anyhow::Result<LayerOut<Element>> {
        let (output, argmax_data) = self.evaluate_with_argmax_data_element(inputs)?;

        // convert argmax_data to field elements
        let argmax_data = ArgmaxData {
            max_values: argmax_data,
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
    fn step_info<F: PrimeField>(
        &self,
        mut aux: ContextAux,
    ) -> anyhow::Result<(LayerCtx<F>, ContextAux)> {
        ensure!(
            aux.last_output_shape.len() == 1,
            "Expected 1 input shape in ContextAux for Logits layer, found {}",
            aux.last_output_shape.len(),
        );

        if self.range_check_chunks() == 1 {
            // the MLE to commit is the tensor of max values, so we set the size the size of this tensor as
            // the biggest MLE being committed
            aux.max_poly_len = aux.last_output_shape[0]
                .iter()
                .take(aux.last_output_shape[0].rank() - 1)
                .product();
        } else {
            // otherwise, we commit to the range-checked inputs, which are the biggest MLEs committed in this layer
            aux.max_poly_len = aux.last_output_shape[0].numel();
        }

        aux.last_output_shape = self.output_shapes(&aux.last_output_shape, PaddingMode::Padding)?;

        Ok((
            LayerCtx::Logits(LogitsCtx {
                range_check_chunks: self.range_check_chunks(),
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
        input_scaling: &[ScalingFactor],
        _unpadded_input_shapes: &[Shape],
        output_scalings: &[ScalingFactor],
        _unpadded_output_shapes: &[Shape],
    ) -> anyhow::Result<QuantizeOutput<Self::QuantizedOp>> {
        ensure!(
            input_scaling.len() == 1,
            "Logits layer expects a single input scaling factor, found {}",
            input_scaling.len()
        );
        // compute bit size of the input
        let input_bit_size = {
            let (min, max) = input_scaling[0].domain();
            ceil_log2(min.abs().max(max.abs()) as usize)
        };
        Ok(QuantizeOutput::new(
            Logits::Argmax(Some(QuantizedArgMaxData { input_bit_size })),
            output_scalings.to_vec(),
        ))
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

impl<F, PCS> ProvableOp<F, PCS> for Logits
where
    F: PrimeField,
    PCS: CommitmentScheme<Field = F>,
{
    type Ctx = LogitsCtx;

    fn prove<T: Transcript>(
        &self,
        node_id: NodeId,
        _ctx: &Self::Ctx,
        _last_claims: Vec<&Claim<F>>,
        step_data: &Step<Element>,
        prover: &mut Prover<F, T, PCS>,
    ) -> anyhow::Result<Vec<Claim<F>>> {
        ensure!(
            step_data.node_inputs.len() == 1,
            "Expected 1 input tensor for Logits layer, found {}",
            step_data.node_inputs.len()
        );
        let input = step_data.padded_input_tensor_at(0)?;
        let outputs = step_data.padded_output_tensors()?;
        // We need the final dim size to build the less than polynomial
        let unpadded_dim_size = *input.unpadded_shape().last().ok_or(anyhow::anyhow!(
            "Input shape must have at least one dimension"
        ))?;
        ensure!(
            outputs.len() == 1,
            "Expected 1 output tensor for Logits layer, found {}",
            outputs.len()
        );

        let layer_commitment = prover.lookup_witness(node_id)?;

        let (layer_polys, commitments): (Vec<_>, Vec<_>) = layer_commitment
            .iter()
            .map(|committed_poly| {
                (
                    committed_poly.polynomial.clone(),
                    VerifierCommitment::from(committed_poly),
                )
            })
            .unzip();

        let max_mle = &layer_polys[0].as_ref();
        let logup_input = self.build_lookup_input(
            &layer_polys,
            &prover.challenge_storage,
            input.to_field(),
            unpadded_dim_size,
        )?;
        let logup_batch_proof = new_batch_multiple_sizes_prove(&[logup_input], prover.transcript)?;

        // get the claim about the difference between max_values and input data
        let output_claims = logup_batch_proof.output_claims();
        ensure!(
            output_claims.len() == self.range_check_chunks(),
            "Expected {} claims from logup proof in Logits layer, found {}",
            self.range_check_chunks(),
            output_claims.len(),
        );

        // we need to compute a claim about the polynomial given the difference between max_mle and input values:
        // the `output_claims` are about the range-checked chunks of such a difference, so we just need to re-combine them
        let diff_claim = Self::recombine_range_check_claims(output_claims)?;

        // evaluate max_values MLE on the same point of `diff_claim`
        // we need to extract the row-related coordinates from `diff_claim.point`
        let num_row_vars = max_mle.num_vars();
        let num_col_vars = diff_claim.point.len() - num_row_vars;
        let (_, row_point) = Self::split_claim_point(&diff_claim.point, num_row_vars)?;

        let input_claim = Claim::new(diff_claim.point.clone(), diff_claim.eval);

        let input_shape = input.shape();

        let two_inv = F::from(2).inverse().expect("Cannot fail when inverting 2");
        let num_cols = F::from(Element::unit() << num_col_vars);
        let sum_eq_point = vec![&two_inv; num_col_vars]
            .into_iter()
            .chain(row_point)
            .cloned()
            .collect_vec();
        let sum_eq_mle = evals(&sum_eq_point).into_mle();

        // Here we build the less than polynomial, for each row it should have 1s at every evaluation less than `unpadded_dim_size` and 0s otherwise
        let base_lt_evals = (0usize..1 << num_col_vars)
            .map(|i| {
                if i < unpadded_dim_size {
                    F::ONE
                } else {
                    F::ZERO
                }
            })
            .collect::<Vec<F>>();
        // `base_lt_evals` is the MLE evals on the boolean hypercube for one row, now we need to repeat this number of rows times
        let lt_mle = vec![base_lt_evals; 1 << num_row_vars].concat().into_mle();
        let input_field: Tensor<F> = input.to_field();
        let input_mle = input_field.into_mle_2d()?;

        let padded_max_evals = max_mle
            .evals_ref()
            .iter()
            .flat_map(|v| vec![*v; 1 << num_col_vars])
            .collect::<Vec<F>>()
            .into_mle();
        // build one-hot encoded output matrix
        let mut one_hot_output = vec![F::ZERO; input_shape.product()];

        for (i, out) in outputs[0].data().iter().cloned().enumerate() {
            let out = out as usize;
            let index = i * input_shape.dim(input_shape.rank() - 1) + out;
            one_hot_output[index] = F::ONE;
        }

        let one_hot_mle = one_hot_output.into_mle();
        // compute the beta evaluations over `input_claim.point`
        let input_eq_mle = evals(&input_claim.point).into_mle();

        let num_vars = input_mle.num_vars();
        let num_threads = optimal_sumcheck_threads(num_vars);
        let mut expr_builder = VirtualPolynomialsBuilder::<F>::new(num_threads, num_vars);
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
                - Expression::Constant(num_cols) * one_hot_expr.clone() * input_expr.clone());
        let expr_second_part = input_eq_expr.clone()
            * lt_expr.clone()
            * (padded_max_expr.clone() - input_expr.clone());
        let expr =
            expr_first_part + Expression::Challenge(0, 1, F::ONE, F::ZERO) * expr_second_part;

        // squeeze the challenge to include `input_claim` produced by the lookup in the hadamard product sum-check
        let challenge = Self::squeeze_challenge(prover.transcript, &input_claim);
        let virtual_poly = expr_builder.to_virtual_polys(&[expr], &[challenge]);
        let (hadamard_proof, state) = IOPProverState::<F>::prove(virtual_poly, prover.transcript);

        let point = state.collect_raw_challenges();
        let sumcheck_evals = state.get_mle_flatten_final_evaluations()[..2].to_vec();
        let input_eval = sumcheck_evals[0];
        let max_eval = sumcheck_evals[1];

        if self.range_check_chunks() == 1 {
            prover.add_witness_claim_per_poly(
                node_id,
                vec![Claim::new(point[num_col_vars..].to_vec(), max_eval)],
            );
        } else {
            prover.add_witness_claim_per_poly(
                node_id,
                once(Claim::new(point[num_col_vars..].to_vec(), max_eval))
                    .chain(output_claims.to_vec())
                    .collect(),
            );
        }

        let final_input_claim = Claim::new(state.collect_raw_challenges(), input_eval);

        let proof = LogitsProof {
            logup_proof: logup_batch_proof,
            commitments,
            hadamard_proof,
            sumcheck_evals,
        };

        prover.push_proof(node_id, LayerProof::Logits(proof));

        Ok(vec![final_input_claim])
    }

    fn gen_lookup_witness(
        &self,
        id: NodeId,
        _ctx: &ProverContext<F, PCS>,
        step_data: &Step<Element>,
    ) -> anyhow::Result<LookupWitnessGen<'static, F>> {
        ensure!(
            step_data.node_inputs.len() == 1,
            "Expected 1 input tensor for Logits witness generation, found {}",
            step_data.node_inputs.len()
        );

        let input = step_data.padded_input_tensor_at(0)?;

        ensure!(
            matches!(self, Logits::Argmax(Some(_))),
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

        // max_values was computed during unpadded inference, so use unpadded shape for the row count check
        let unpadded_input_shape = input.unpadded_shape();
        let expected_rows: usize = unpadded_input_shape
            .iter()
            .take(unpadded_input_shape.rank() - 1)
            .product();
        ensure!(
            max_values.shape()[0] == expected_rows,
            "Incompatible shapes between max values tensor (rows={}) and flattened input rows (expected_rows={}) for input shape {:?}",
            max_values.shape()[0],
            expected_rows,
            input.unpadded_shape(),
        );

        let unpadded_dim_size = *input.unpadded_shape().last().ok_or(anyhow::anyhow!(
            "Input shape must have at least one dimension"
        ))?;

        let max_values_guard = max_values.tensor()?;
        let padded_input_shape = input.shape();
        let padded_rows: usize = padded_input_shape
            .iter()
            .take(padded_input_shape.rank() - 1)
            .product();
        let padded_cols = padded_input_shape.dim(padded_input_shape.rank() - 1);
        let mut merged_diff = input
            .slice_last_dim()
            .zip(max_values_guard.data())
            .flat_map(|(row, row_max)| {
                let current_max = row_max;
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
        // Extend with zeros for padded rows (padded input is zero, max is zero, diff is zero)
        merged_diff.resize(padded_rows * padded_cols, 0);
        // split merged diff in chunks
        let mut element_count = HashMap::new();
        let chunks = (0..self.range_check_chunks())
            .map(|i| {
                merged_diff
                    .iter()
                    .map(|v| {
                        let shifted = v >> (i * SHIFT_CHECK_TABLE_BIT_SIZE);
                        let mask = (1i64 << SHIFT_CHECK_TABLE_BIT_SIZE) - 1;
                        let value = shifted & mask;
                        *element_count.entry(value).or_default() += 1;
                        F::from(value)
                    })
                    .collect::<Vec<F>>()
            })
            .collect::<Vec<Vec<F>>>();

        // commit to max values
        let max_mle = max_values_guard.pad_next_power_of_two().to_field_mle();

        let mles = if self.range_check_chunks() == 1 {
            // no need to commit to a single chunk
            vec![max_mle]
        } else {
            once(max_mle)
                .chain(chunks.into_iter().map(|chunk| chunk.into_mle()))
                .collect_vec()
        };

        let mut witness_gen = LookupWitnessGen::<F>::default();
        witness_gen.insert_layer_witness_data(
            id,
            mles,
            vec![Table::new_shift_check()],
            vec![element_count],
        );

        Ok(witness_gen)
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

const DEFAULT_MIN_CHUNK_SIZE: usize = 128;

lazy_static! {
    pub(crate) static ref MIN_CHUNK_SIZE: RwLock<Option<usize>> = RwLock::new(None);
}

impl Splittable for LogitsCtx {
    fn ideal_num_chunks(&self, input_shapes: &[Shape]) -> Option<usize> {
        let min_chunk_size = MIN_CHUNK_SIZE
            .read_recursive()
            .unwrap_or(DEFAULT_MIN_CHUNK_SIZE);
        (input_shapes.len() == 1).then(|| {
            assert_eq!(input_shapes[0].rank(), 2); // for now we support splitting only for rank 2 input tensors in this layer
            let seq_len = input_shapes[0].dim(0);
            div_ceil(seq_len, min_chunk_size)
        })
    }

    fn is_splittable(&self) -> bool {
        true
    }
}

impl<F: PrimeField, PCS: CommitmentScheme> VerifiableCtx<F, PCS> for LogitsCtx {
    type Proof = LogitsProof<F, PCS>;

    fn verify<T: Transcript>(
        &self,
        proof: &Self::Proof,
        last_claims: &[&Claim<F>],
        verifier: &mut Verifier<F, T, PCS>,
        shape_step: &ShapeStep,
        node_id: NodeId,
    ) -> anyhow::Result<Vec<Claim<F>>> {
        self.verify_chunk(proof, last_claims, verifier, shape_step, node_id, 0)
    }

    fn verify_chunk<T: Transcript>(
        &self,
        proof: &Self::Proof,
        _last_claims: &[&Claim<F>],
        verifier: &mut Verifier<F, T, PCS>,
        shape_step: &ShapeStep,
        node_id: NodeId,
        chunk_number: usize,
    ) -> anyhow::Result<Vec<Claim<F>>> {
        ensure!(
            shape_step.padded_input_shape.len() == 1,
            "Expected 1 padded input shape when verifying Logits layer, found {}",
            shape_step.padded_input_shape.len(),
        );

        let input_shape = &shape_step.padded_input_shape[0];

        let full_vars = ceil_log2(input_shape.numel());
        let table = Table::new_shift_check();
        let (constant_challenge, column_separation_challenge) = verifier
            .challenge_storage
            .get_challenges_by_name(&table.name())
            .ok_or(anyhow!(
                "Cannot verify Argmax, cannot get {} table challenges",
                table.name()
            ))?;
        let num_range_checks = self.range_check_chunks;
        let instances = vec![
            LogUpVerifierInstance::new(
                constant_challenge,
                column_separation_challenge,
                table.num_columns(),
                crate::lookup::logup_gkr::structs::ProofType::Lookup,
                full_vars - 1,
            );
            num_range_checks
        ];

        // Append the fractional outputs to the verifier
        let (numerators, denominators) = proof.logup_proof.fractional_outputs();
        let (shift_num, shift_denom) = verifier
            .numerators_and_denominators
            .entry(table.name())
            .or_insert((F::ZERO, F::ONE));
        for (num, denom) in numerators.iter().zip(denominators.iter()) {
            ensure!(
                *denom != F::ZERO,
                "Denominator in fractional output for Logits layer lookup cannot be zero"
            );
            *shift_num = *num * *shift_denom + *shift_num * *denom;
            *shift_denom *= *denom;
        }
        let batch_claim = new_verify_logup_proof_multiple_sizes(
            &proof.logup_proof,
            &instances,
            verifier.transcript,
        )?;

        let poly_claims = batch_claim
            .poly_evals()
            .iter()
            .map(|eval| Claim::new(batch_claim.point().to_vec(), *eval))
            .collect::<Vec<Claim<F>>>();

        ensure!(
            poly_claims.len() == num_range_checks,
            "Expected {} claims for logup when verifying Logis layer, found {}",
            num_range_checks,
            poly_claims.len(),
        );

        let num_row_vars = (0..input_shape.rank() - 1)
            .map(|d| input_shape.dim(d))
            .product::<usize>()
            .ilog2() as usize;
        let num_col_vars = input_shape.dim(input_shape.rank() - 1).ilog2() as usize;
        let (_, row_point) = Logits::split_claim_point(batch_claim.point(), num_row_vars)?;

        let input_claim = Logits::recombine_range_check_claims(&poly_claims)?;

        let two_inv = F::from(2).inverse().expect("Cannot fail when inverting 2");
        let num_cols = F::from(Element::unit() << num_col_vars);
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

        let sumcheck_point = &subclaim.point;
        let beta_eval = identity_eval(&sum_eq_point, sumcheck_point);
        let input_eq_eval = identity_eval(&input_claim.point, sumcheck_point);

        // Here we make the less than poly eval
        let unpadded_dim_size =
            shape_step.unpadded_input_shape[0].dim(shape_step.unpadded_input_shape[0].rank() - 1);
        // We subtract 1 from the unpadded dimension size because this function calculates the evaluation of the mle that checks x <= y.
        let dim_size_bits = to_bit_sequence_le(unpadded_dim_size - 1, num_col_vars)
            .map(|b| F::from(b as u64))
            .collect::<Vec<F>>();
        let lt_eval = eval_zeroifier_mle(&sumcheck_point[..num_col_vars], &dim_size_bits);

        // get expected evaluation of the claim for the output tensor MLE computed by the sum-check; we have that
        // `subclaim.expected_evaluation = beta_eval * (max_eval - num_cols * proof.input_eval * expected_output_eval) + challenge * input_eq_eval * lt_eval * (max_eval - proof.input_eval)`,
        // so we compute `expected_output_eval` as `(subclaim.expected_evaluation - challenge * input_eq_eval * lt_eval *(max_eval - proof.input_eval) - beta_eval * max_eval)/(-num_cols * beta_eval * proof.input_eval)`
        let sumcheck_evals = &proof.sumcheck_evals;
        let input_eval = sumcheck_evals[0];
        let max_eval = sumcheck_evals[1];
        let expected_output_eval = match (-num_cols * beta_eval * input_eval).inverse() {
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
                    Logits::split_claim_point(sumcheck_point, num_row_vars)?;
                let row_part = evals(row_point).into_iter().sum::<F>();
                let column_part = column_point.iter().map(|p| F::ONE - *p).product::<F>();
                assert!(
                    (subclaim.expected_evaluation
                        - challenge * lt_eval * input_eq_eval * (max_eval - input_eval)
                        - beta_eval * max_eval)
                        == F::ZERO
                );
                row_part * column_part
            }
        };

        Self::verify_output_evaluation(
            verifier,
            Claim::new(sumcheck_point.clone(), expected_output_eval),
            unpadded_dim_size,
            chunk_number,
        )?;

        // Add the max_values claim to the commitment verifier
        if num_range_checks == 1 {
            ensure!(
                proof.commitments.len() == 1,
                "Expected 1 commitment in Logits proof, found {}",
                proof.commitments.len(),
            );
            verifier.commit_verifier.add_witness_claim(
                node_id,
                vec![VerifierClaim::from((
                    proof.commitments[0].clone(),
                    Claim::new(sumcheck_point[num_col_vars..].to_vec(), max_eval),
                ))],
            );
        } else {
            ensure!(
                proof.commitments.len() == num_range_checks + 1,
                "Expected {} commitments in Logits proof, found {}",
                num_range_checks + 1,
                proof.commitments.len(),
            );
            verifier.commit_verifier.add_witness_claim(
                node_id,
                once(VerifierClaim::from((
                    proof.commitments[0].clone(),
                    Claim::new(sumcheck_point[num_col_vars..].to_vec(), max_eval),
                )))
                .chain(
                    proof.commitments[1..]
                        .iter()
                        .zip(poly_claims)
                        .map(|(commitment, claim)| {
                            VerifierClaim::from((commitment.clone(), claim))
                        }),
                )
                .collect(),
            );
        }

        let final_input_claim = Claim::new(sumcheck_point.clone(), input_eval);

        Ok(vec![final_input_claim])
    }

    // The verifier of this layer doesn't need to employ any
    // claim about the output tensors. Indeed, the claims about the output tensors are computed
    // by the prover, and are verified directly in `LogitsCtx::verify_output_evaluation` method
    // HOWEVER, we still need to make the transcript advance since for the prover it always
    // tries to derive claims from the transcript, for the output tensor if nothing else.
    // fn compute_model_output_claims<T: Transcript<E>>(

    fn write_proof_to_transcript<T: Transcript>(
        &self,
        proof: &Self::Proof,
        transcript: &mut T,
    ) -> anyhow::Result<()> {
        proof.write_commitment(transcript)
    }
}

impl LogitsCtx {
    fn verify_output_evaluation<F: PrimeField, T: Transcript, PCS: CommitmentScheme>(
        verifier: &mut Verifier<F, T, PCS>,
        output_claim: Claim<F>,
        unpadded_dim_size: usize,
        chunk_number: usize,
    ) -> anyhow::Result<()> {
        ensure!(
            verifier.io.output.len() == 1,
            "Expected 1 output tensor when verifying logits layer output claim, found {}",
            verifier.io.output.len(),
        );
        let output = verifier
            .io
            .splitted_outputs
            .as_ref()
            .and_then(|splitted_outputs| {
                splitted_outputs
                    .get(&0)
                    .map(|outputs| &outputs[chunk_number])
            })
            .unwrap_or(&verifier.io.output[0]);
        ensure!(
            output.shape().is_power_of_two(),
            "Output shape in Logits layer is not a power of 2"
        );
        let num_row_vars = output.shape().dim(0).ilog2() as usize;
        let (column_point, row_point) =
            Logits::split_claim_point(&output_claim.point, num_row_vars)?;
        let beta = evals(row_point);
        let column_beta = evals(column_point);
        let computed_eval =
            output
                .data()
                .iter()
                .zip(beta)
                .try_fold(F::ZERO, |sum, (token, b1)| {
                    let token_value = token.to_element() as usize;
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
}

#[cfg(test)]
mod test {

    use ark_std::rand::Rng as ArkRng;
    use dp_crypto::arkyper::transcript::blake3::Blake3Transcript;
    use parking_lot::{RwLockUpgradableReadGuard, RwLockWriteGuard};
    use tenstore::GenStore;

    use super::*;
    use crate::{
        graph::executor::ThreadPoolExecutor,
        init_test_logging,
        iop::chunking::DefaultChunkingStrategy,
        layers::{Layer, einsum::EinSum, provable::Evaluate},
        model::{
            Model,
            test::{prove_model, prove_model_with, quantize_model},
        },
        padding::pad_model,
        rng_from_env_or_random,
        tensor::Tensor,
        testing::Pcs,
        verify,
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
        let logits = Logits::Argmax(None);

        let out = logits.evaluate(&[&input.as_wrapped()])?;
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
        let mut model = Model::new_from_input_shapes(vec![input_shape]);

        let _ = model
            .add_consecutive_layer(Layer::Logits(Logits::Argmax(None)), None)
            .unwrap();

        model.automatic_output_labelling().unwrap();

        prove_model(model, &mut GenStore::default()).unwrap();
    }

    #[test]
    fn test_proven_null_logits_argmax() {
        let seq_len = 13;
        let vocab_size = 17;
        let input_shape = Shape::new(vec![seq_len, vocab_size]);
        let mut model = Model::new_from_input_shapes(vec![input_shape.clone()]);

        let _ = model
            .add_consecutive_layer(Layer::Logits(Logits::Argmax(None)), None)
            .unwrap();

        model.automatic_output_labelling().unwrap();
        let inputs = Tensor::zeros(input_shape);

        prove_model_with(model, vec![inputs], &mut GenStore::default()).unwrap();
    }

    fn build_model_with_argmax_and_einsum(
        seq_len: usize,
        vocab_size: usize,
        hidden_size: usize,
    ) -> anyhow::Result<Model<f32>> {
        let input_shape = Shape::new(vec![seq_len, hidden_size]);

        let einsum = Layer::EinSum(
            EinSum::new_matmul(
                None,
                Some(TensorHandle::from_tensor(
                    StorageKey::from("embedding_matrix"),
                    GenStore::new_empty(),
                    Tensor::random(&Shape::new(vec![hidden_size, vocab_size])),
                )),
                false,
                None,
            )?
            .disable_requantisation(),
        );
        let mut model = Model::new_from_input_shapes(vec![input_shape.clone()]);
        let einsum_id = model.add_consecutive_layer(einsum, None)?;
        let _ =
            model.add_consecutive_layer(Layer::Logits(Logits::Argmax(None)), Some(einsum_id))?;

        model.automatic_output_labelling()?;

        Ok(model)
    }

    #[test]
    fn test_proven_argmax_after_einsum() {
        let seq_len = 13;
        let vocab_size = 17;
        let hidden_size = 23;

        let model = build_model_with_argmax_and_einsum(seq_len, vocab_size, hidden_size).unwrap();
        prove_model(model, &mut GenStore::default()).unwrap();
    }

    type F = ark_bn254::Fr;

    type T = Blake3Transcript;
    type P<'a, 'b> = Prover<'a, 'b, F, T, Pcs>;

    #[test]
    fn test_chunked_argmax() -> anyhow::Result<()> {
        init_test_logging("debug");
        let mut rng = rng_from_env_or_random();
        let seq_len = rng.gen_range(17..95);
        let vocab_size = 17;
        let hidden_size = 23;
        let model = build_model_with_argmax_and_einsum(seq_len, vocab_size, hidden_size)?;
        let inputs = model.input_shapes().iter().map(Tensor::random).collect();

        let (quantized_model, quantized_inputs) =
            quantize_model(model, inputs, None, &mut Default::default())?;

        let model = pad_model(quantized_model)?;
        let inputs = model.prepare_inputs(quantized_inputs)?;

        model.describe();

        let trace = model.run(inputs, &mut Default::default()).unwrap();
        let (prover_ctx, verifier_ctx) = model
            .generate_contexts::<F, Pcs>()
            .expect("unable to generate contexts");

        let mut lock = MIN_CHUNK_SIZE.write();
        lock.replace(4); // set the minimum chunk size to 4 to force horizontal chunking on small inputs

        // dowgrade to a read lock to prevent any other test thread to change `MIN_CHUNK_SIZE_TEST` while this test is running
        let read_lock = RwLockWriteGuard::downgrade_to_upgradable(lock);
        let chunking_strategy = DefaultChunkingStrategy::from(&trace);
        let (proof, io) = P::chunked_prove_local::<_, ThreadPoolExecutor>(
            &prover_ctx,
            trace,
            Some(6),
            chunking_strategy,
            &model,
            (),
        )
        .expect("unable to generate proof");
        verify::<_, T, _>(&verifier_ctx, proof, io).expect("invalid proof");

        // reset to None
        RwLockUpgradableReadGuard::upgrade(read_lock).take();

        Ok(())
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
        let logits = Logits::Argmax(None);
        let out = logits.evaluate(&[&input.as_wrapped()])?;
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
            let logits = Logits::Argmax(None);
            let out = logits.evaluate(&[&input.as_wrapped()]).unwrap();
            let indices = out.outputs()[0].get_data();
            prop_assert_eq!(indices.len(), expected.len());
            for (i, idx) in indices.iter().enumerate() { prop_assert!((*idx - expected[i]).abs() < 1e-6); }
        }

        #[test]
        fn prop_logits_argmax_element((shape, data, expected) in logits_input_strategy()) {
            let data_elem: Vec<Element> = data.into_iter().map(|v| v.round_ties_even() as Element).collect();
            let input = Tensor::new(shape.clone().into(), data_elem).unwrap();
            let logits = Logits::Argmax(None);

            let out = logits.evaluate(&[&input.as_wrapped()]).unwrap();
            let indices = out.outputs()[0].get_data();
            prop_assert_eq!(indices.len(), expected.len());
            for (i, idx) in indices.iter().enumerate() { prop_assert_eq!(*idx as usize, expected[i] as usize); }
        }
    }
}
