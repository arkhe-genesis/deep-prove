use crate::{
    Claim, Element, Prover, ScalingFactor, ScalingStrategy, Shape, Tensor,
    commit::{compute_betas_eval, identity_eval},
    eval_zeroifier_mle,
    graph::NodeId,
    iop::{
        context::{ContextAux, ShapeStep},
        verifier::Verifier,
    },
    layers::{
        LayerCtx, LayerProof,
        add::{Add, AddCtx, AddProof},
        provable::{
            Evaluate, LayerOut, PadOp, ProveInfo, QuantizeOp, QuantizeOutput, VerifiableCtx,
        },
        transformer::positional::{Positional, PositionalCache, PositionalCtx, PositionalProof},
    },
    model::Step,
    quantization::TensorFielder,
    tensor::{CommitmentId, KeyedTensor, TensorSlice, TensorTypeParam, WrappedTensor},
    to_bit_sequence_le,
    util::from_mle_list_dimensions,
};
use anyhow::ensure;
use either::Either;
use ff_ext::ExtensionField;
use mpcs::PolynomialCommitmentScheme;
use multilinear_extensions::{
    mle::IntoMLE, util::ceil_log2, virtual_polys::VirtualPolynomialsBuilder,
};
use p3_field::FieldAlgebra;
use rayon::prelude::*;
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use std::{
    iter::once,
    ops::Deref,
    sync::{Arc, Mutex},
};
use sumcheck::{
    structs::{IOPProof, IOPProverState, IOPVerifierState},
    util::optimal_sumcheck_threads,
};

use transcript::Transcript;

/// Data structure containing the proof data for the absolute variant of positional encoding layer
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound(serialize = "E: Serialize", deserialize = "E: DeserializeOwned"))]
pub struct AbsoluteProof<E: ExtensionField> {
    /// The sumcheck proof used to check we have used the correct subslice of the positional matrix
    sumcheck_proof: IOPProof<E>,
    // Evaluations of the sub-matrices required to compute the claim
    // about the positional matrix. Each sub-matrix is identified by
    // an incremental integer that corresponds to an extra variable to be processed
    // to get to the number of variables of the positional matrix.
    sub_matrix_evals: Vec<E>,
    // Proofs for addition of the slice of the positional matrix with an
    // input tensor
    add_proof: AddProof<E>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AbsoluteCtx {
    add_ctx: AddCtx,
    pub(super) unpadded_shape: Shape,
    num_vars_positional_matrix: usize,
    node_id: NodeId,
    positional_key: CommitmentId,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Absolute<N> {
    pub(crate) positional: KeyedTensor<N>,
    pub(super) unpadded_shape: Shape,
    add_layer: Add<N>,
}

impl<N: TensorTypeParam> Absolute<N> {
    fn num_vars(&self) -> usize {
        let num_vars = self.positional.shape().num_vars_2d();
        num_vars.0 + num_vars.1
    }

    pub(super) fn new(matrix: KeyedTensor<N>) -> Self {
        let unpadded_shape = matrix.shape().clone();
        Self {
            positional: matrix,
            unpadded_shape,
            add_layer: Add::new(),
        }
    }
}

impl<N> Absolute<N> {
    pub(super) fn evaluate<E: ExtensionField>(
        &self,
        input: &WrappedTensor<N>,
        positional_cache: &Arc<Mutex<PositionalCache>>,
    ) -> anyhow::Result<LayerOut<N, E>>
    where
        N: TensorTypeParam,
        Add<N>: Evaluate<N>,
    {
        let past_length = positional_cache.lock().unwrap().seq_len;

        let is_padded = input.is_padded();

        let sub_bt = WrappedTensor::try_from(
            &self
                .positional
                .slice_2d(past_length, past_length + input.unpadded_shape().dims[0])?,
        )?;

        let sub_bt = if is_padded {
            sub_bt.pad_next_power_of_two()
        } else {
            sub_bt
        };

        positional_cache
            .lock()
            .unwrap()
            .set_seq_len(past_length + input.unpadded_shape().dims[0])?;
        let mut outputs = self.add_layer.evaluate::<E>(&[input, &sub_bt])?.outputs;
        ensure!(
            outputs.len() == 1,
            "Expected 1 output from add in positional encoding layer, got {}",
            outputs.len()
        );
        let output = outputs.pop().unwrap();
        Ok(LayerOut::from_vec(vec![output]))
    }
}

impl Absolute<f32> {
    pub(super) fn quantize<S: ScalingStrategy>(
        self,
        data: &S::AuxData,
        node_id: NodeId,
        input_scaling: ScalingFactor,
        unpadded_input_shapes: &[Shape],
        output_scalings: &[ScalingFactor],
        unpadded_output_shapes: &[Shape],
    ) -> anyhow::Result<QuantizeOutput<Absolute<Element>>> {
        // quantize positional matrix
        let max = self.positional.max_abs_output();
        let pos_scaling = ScalingFactor::from_absolute_max(max, None);

        ensure!(
            output_scalings.len() == 1,
            "Expected 1 output scaling factor for absolute positional layer, found {}",
            output_scalings.len()
        );
        ensure!(
            unpadded_output_shapes.len() == 1,
            "Expected 1 output shape for absolute positional layer, found {}",
            unpadded_output_shapes.len()
        );

        let quantized_add = self.add_layer.quantize_op::<S>(
            data,
            node_id,
            &[input_scaling, pos_scaling],
            unpadded_input_shapes,
            output_scalings,
            unpadded_output_shapes,
        )?;

        let quantized_pos = Absolute {
            positional: self.positional.quantize(&pos_scaling),
            unpadded_shape: self.unpadded_shape,
            add_layer: quantized_add.quantized_op,
        };

        Ok(QuantizeOutput {
            quantized_op: quantized_pos,
            output_scalings: quantized_add.output_scalings,
            requant_layer: quantized_add.requant_layer,
            post_quant_rule: None,
        })
    }
}

impl PadOp for Absolute<Element> {
    fn pad_node(mut self, _si: &mut crate::padding::ShapeInfo) -> anyhow::Result<Self>
    where
        Self: Sized,
    {
        self.positional = self.positional.map_tensor(|t| t.pad_next_power_of_two());
        Ok(self)
    }
}

impl Absolute<Element> {
    pub(super) fn step_info<E: ExtensionField>(
        &self,
        id: NodeId,
        aux: ContextAux,
    ) -> anyhow::Result<(AbsoluteCtx, ContextAux)> {
        let (ctx, mut aux) = self.add_layer.step_info(id, aux)?;

        let LayerCtx::<E>::Add(add_ctx) = ctx else {
            unreachable!()
        };

        aux.model_polys = Some(
            aux.model_polys
                .unwrap_or_default()
                .into_iter()
                .chain(once((
                    self.positional.commitment_id(),
                    self.positional.pad_next_power_of_two().into_data(),
                )))
                .collect(),
        );

        let ctx = AbsoluteCtx {
            add_ctx,
            unpadded_shape: self.unpadded_shape.clone(),
            num_vars_positional_matrix: self.num_vars(),
            node_id: id,
            positional_key: self.positional.commitment_id(),
        };

        Ok((ctx, aux))
    }

    pub(super) fn prove_step<
        E: ExtensionField,
        T: Transcript<E>,
        PCS: PolynomialCommitmentScheme<E> + Send + Sync,
    >(
        &self,
        node_id: NodeId,
        output_claim: &Claim<E>,
        step_data: &Step<E, Element>,
        prover: &mut Prover<E, T, PCS>,
    ) -> anyhow::Result<Vec<Claim<E>>>
    where
        PCS::CommitmentWithWitness: Serialize + DeserializeOwned + Send + Sync,
        PCS::ProverParam: Send + Sync,
    {
        let input = &step_data.input_tensors()?[0];

        // derive sub-matrix to be added to input. ToDo: place it in proving data
        let matrix_slice = TensorSlice::from(self.positional.deref());

        let masked_sub_pos = matrix_slice
            .slice_over_first_dim(0, input.unpadded_shape()[0])
            .to_tensor()?
            .pad_next_power_of_two();
        let sub_pos: Tensor<E> = matrix_slice
            .slice_over_first_dim(0, input.shape()[0])
            .to_tensor()?
            .to_fields();

        let (mut claims, add_proof) = self.add_layer.prove_step(
            node_id,
            vec![output_claim],
            &[input, &masked_sub_pos],
            prover,
        )?;

        ensure!(
            claims.len() == 2,
            "Expected 2 claims from Add proving in position layer, found {} claims",
            claims.len(),
        );

        let sub_pos_claim = claims.pop().unwrap();
        let input_claim = claims.pop().unwrap();

        // Now we prove that the sub_pos_claim is correctly formed from the positional matrix.
        // Here the `row_num_vars` corresponds to the number of variables needed to index a row of the matrix.
        // For example if `row_num_vars = 3` then we have 8 columns in the positional matrix and the point `[0, 1, 0]` would correspond
        // to the the element in the row with index `2` (assuming zero indexing).
        let row_num_vars = ceil_log2(input.shape().dim(-1));
        let row_point = &sub_pos_claim.point()[..row_num_vars];
        let row_evals = compute_betas_eval(row_point);
        let column_eq = compute_betas_eval(&sub_pos_claim.point()[row_num_vars..]).into_mle();
        // All of the rows in the positional matrix are fixed for the Sumcheck
        let fixed_positional = sub_pos
            .get_data()
            .par_chunks(input.shape().dim(-1))
            .with_min_len(64)
            .map(|row| {
                row.iter()
                    .take(input.unpadded_shape().dim(-1))
                    .zip(row_evals.iter())
                    .map(|(v, r)| *v * *r)
                    .sum::<E>()
            })
            .collect::<Vec<E>>()
            .into_mle();
        // This polynomial is 1 for rows less than the input length and 0 otherwise
        let lt_poly = (0..input.shape().dim(0))
            .map(|i| {
                if i < input.unpadded_shape().dim(0) {
                    E::BaseField::ONE
                } else {
                    E::BaseField::ZERO
                }
            })
            .collect::<Vec<E::BaseField>>()
            .into_mle();

        let num_vars = column_eq.num_vars();
        let num_threads = optimal_sumcheck_threads(num_vars);
        let mut expr_builder = VirtualPolynomialsBuilder::<E>::new(num_threads, num_vars);
        let fixed_positional_expr = expr_builder.lift(Either::Left(&fixed_positional));
        let lt_expr = expr_builder.lift(Either::Left(&lt_poly));
        let column_eq_expr = expr_builder.lift(Either::Left(&column_eq));
        // The expression checks that the sub-positional matrix is just the rows of the positional matrix up to the input length
        let virtual_poly =
            expr_builder.to_virtual_polys(&[fixed_positional_expr * lt_expr * column_eq_expr], &[]);
        // This sumcheck checks that the positional matrix subslice has been correctly formed
        let (proof, state) = IOPProverState::<E>::prove(virtual_poly, prover.transcript);

        let sumcheck_point = state.collect_raw_challenges();
        let all_evals = state.get_mle_flatten_final_evaluations();
        let sub_matrix_eval = all_evals[0];
        let sub_pos_claim = Claim::<E>::new(
            row_point
                .iter()
                .chain(sumcheck_point.iter())
                .copied()
                .collect(),
            sub_matrix_eval,
        );

        // we now need to bind the claim about the `sub_pos` tensor with a claim about `positional_matrix`
        let (sub_matrix_evals, positional_matrix_claim) =
            Positional::<Element>::bind_sub_claim_to_positional_matrix(
                sub_pos_claim,
                output_claim,
                &matrix_slice,
                &self.positional,
                input.shape()[0],
                prover.transcript,
            )?;

        prover.add_common_claims(
            node_id,
            [(self.positional.commitment_id(), positional_matrix_claim)]
                .into_iter()
                .collect(),
        );

        prover.push_proof(
            node_id,
            LayerProof::Positional(PositionalProof::Absolute(AbsoluteProof {
                sumcheck_proof: proof,
                sub_matrix_evals,
                add_proof,
            })),
        );

        Ok(vec![input_claim])
    }
}

impl AbsoluteCtx {
    pub(super) fn verify<
        E: ExtensionField,
        T: Transcript<E>,
        PCS: PolynomialCommitmentScheme<E>,
    >(
        &self,
        proof: &AbsoluteProof<E>,
        verifier: &mut Verifier<E, T, PCS>,
        output_claim: &Claim<E>,
        shape_step: &ShapeStep,
    ) -> anyhow::Result<Vec<Claim<E>>> {
        // compute shape step for add sub-layer
        let unpadded_input_shapes = vec![shape_step.unpadded_input_shape[0].clone(); 2];
        let padded_input_shapes = vec![shape_step.padded_input_shape[0].clone(); 2];
        let shape_step = LayerCtx::<E>::Add(self.add_ctx.clone())
            .shape_step(&unpadded_input_shapes, &padded_input_shapes)?;

        let AbsoluteProof {
            sumcheck_proof,
            sub_matrix_evals,
            add_proof,
        } = proof;
        let mut claims = self
            .add_ctx
            .verify(add_proof, &[output_claim], verifier, &shape_step)?;

        ensure!(
            claims.len() == 2,
            "Expected 2 claims from Add verifier in position layer, found {} claims",
            claims.len(),
        );

        let sub_pos_claim = claims.pop().unwrap();

        let input_claim = claims.pop().unwrap();

        // Now we verify the sumcheck proof for the positional matrix subslice
        let num_row_vars = ceil_log2(padded_input_shapes[0].dim(-1));
        let num_col_vars = ceil_log2(padded_input_shapes[0].dim(0));

        let subclaim = IOPVerifierState::<E>::verify(
            sub_pos_claim.evaluation(),
            sumcheck_proof,
            &from_mle_list_dimensions(&[vec![num_col_vars, num_col_vars, num_col_vars]]),
            verifier.transcript,
        );

        // Calculate the sub matrix claim from the subclaim and lt and eq evals
        let sumcheck_point = subclaim
            .point
            .iter()
            .map(|c| c.elements)
            .collect::<Vec<E>>();
        let column_eq_eval = identity_eval(&sub_pos_claim.point()[num_row_vars..], &sumcheck_point);

        let unpadded_seq_len = unpadded_input_shapes[0].dim(0);
        let bits = to_bit_sequence_le(unpadded_seq_len - 1, num_col_vars)
            .map(E::from_canonical_usize)
            .collect::<Vec<E>>();
        let lt_eval = eval_zeroifier_mle(&sumcheck_point, &bits);

        let multiplier = (lt_eval * column_eq_eval).inverse();
        let sub_pos_eval = subclaim.expected_evaluation * multiplier;

        let sub_pos_claim = Claim::<E>::new(
            sub_pos_claim
                .point()
                .iter()
                .take(num_row_vars)
                .cloned()
                .chain(sumcheck_point)
                .collect(),
            sub_pos_eval,
        );

        let positional_matrix_claim = PositionalCtx::build_positional_matrix_claim(
            sub_pos_claim,
            output_claim,
            self.num_vars_positional_matrix,
            verifier.transcript,
            sub_matrix_evals,
        )?;

        verifier.add_common_claims(
            self.node_id,
            [(self.positional_key.clone(), positional_matrix_claim)]
                .into_iter()
                .collect(),
        );

        Ok(vec![input_claim])
    }
}

#[cfg(test)]
mod tests {
    use std::{
        fmt::Debug,
        ops::Deref,
        slice::from_ref,
        sync::{Arc, Mutex},
    };

    use rstest::rstest;

    use tenstore::{GenStore, StorageKey};

    use crate::{
        Element, NextPowerOfTwo, Tensor,
        layers::{
            Layer,
            provable::{Evaluate, PadOp},
            transformer::positional::{Positional, PositionalCache, absolute::Absolute},
        },
        model::{Model, test::prove_model},
        padding::{PaddingMode, ShapeData, ShapeInfo},
        quantization::{AbsoluteMax, ScalingFactor},
        tensor::{KeyedTensor, TensorSlice, TensorTypeParam, is_close_with_tolerance},
    };
    use ff_ext::GoldilocksExt2;
    use proptest::prelude::*;

    #[rstest]
    #[case::less_input_than_context_length(14, 17, 31)]
    #[case::same_input_as_context_length(31, 17, 31)]
    fn test_proven_absolute_positional_layer(
        #[case] seq_len: usize,
        #[case] embedding_size: usize,
        #[case] context_length: usize,
    ) {
        let input_shape = vec![seq_len, embedding_size];

        let mut model =
            Model::new_from_input_shapes(vec![input_shape.into()], PaddingMode::NoPadding);

        // build positional matrix
        let matrix_shape = vec![context_length, embedding_size];
        let positional_matrix = KeyedTensor::new(
            StorageKey::new("absolute_positional_mat"),
            Tensor::random(&matrix_shape.into()),
        );

        let _ = model
            .add_consecutive_layer(
                Layer::Positional(Positional::new_absolute(positional_matrix)),
                None,
            )
            .unwrap();

        model.automatic_output_labelling().unwrap();

        let _ = prove_model(model, &mut GenStore::default()).unwrap();
    }

    #[derive(Clone)]
    struct Input<T> {
        seq_len: usize,
        embedding_size: usize,
        context_length: usize,
        input: Tensor<T>,
        pos: KeyedTensor<T>,
    }

    impl<T: Debug> Debug for Input<T> {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.debug_struct("AbsoluteInput")
                .field("seq_len", &self.seq_len)
                .field("embedding_size", &self.embedding_size)
                .field("context_length", &self.context_length)
                .finish_non_exhaustive()
        }
    }

    fn input<T: TensorTypeParam>() -> impl Strategy<Value = Input<T>> {
        (1..32usize, 1..64usize).prop_flat_map(|(seq_len, embedding_size)| {
            (seq_len..=64usize).prop_map(move |context_length| {
                let input = Tensor::<T>::random(&vec![seq_len, embedding_size].into());
                let pos = KeyedTensor::new(
                    StorageKey::new("absolute_positional_mat"),
                    Tensor::<T>::random(&vec![context_length, embedding_size].into()),
                );
                Input {
                    seq_len,
                    embedding_size,
                    context_length,
                    input,
                    pos,
                }
            })
        })
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(3))]

        #[test]
        fn test_absolute_f32(input in input::<f32>()) {
            let Input { seq_len, embedding_size, input, pos, .. } = input.clone();
            let layer = Absolute::<f32>::new(pos.clone());

            let cache = Arc::new(Mutex::new(PositionalCache::new()));
            let out = layer
                .evaluate::<GoldilocksExt2>(&input.as_wrapped(), &cache)
                .expect("absolute evaluate should succeed")
                .outputs
                .pop()
                .unwrap();

            let in_data = input.data();
            let pos_data = pos.data();
            let mut expected_data = Vec::with_capacity(seq_len * embedding_size);
            for i in 0..seq_len { for j in 0..embedding_size {
                expected_data.push(in_data[i * embedding_size + j] + pos_data[i * embedding_size + j]);
            }}
            let expected = Tensor::new(vec![seq_len, embedding_size].into(), expected_data).unwrap();
            let close = is_close_with_tolerance(&out.get_data(), expected.data(), 1e-6, 1e-5);
            prop_assert!(close);
        }

        #[test]
        fn test_absolute_element(input in input::<f32>()) {
            let Input { seq_len, input, pos, embedding_size, .. } = input.clone();
            let layer = Absolute::<f32>::new(pos.clone());
            let input_sf = ScalingFactor::from_tensor(&input, None);
            let shape = crate::Shape::new(vec![seq_len, embedding_size]);
            let output_sf = ScalingFactor::from_tensor(&input, None);
            let q = layer
                .quantize::<AbsoluteMax>(
                    &(),
                    0.into(),
                    input_sf,
                    from_ref(&shape),
                    from_ref(&output_sf),
                    from_ref(&shape),
                )
                .expect("quantize absolute should succeed");
            let layer_q = q.quantized_op;
            let input_q = input.to_quantized(&input_sf);

            let (pos_q, add_q) = (layer_q.positional.clone(), &layer_q.add_layer);
            let sub_slice = TensorSlice::from(pos_q.deref()).slice_over_first_dim(0, seq_len);

            let sub_pos_q = Tensor::new(sub_slice.get_shape(), sub_slice.get_data().to_vec()).unwrap();

            let cache = Arc::new(Mutex::new(PositionalCache::new()));
            let out = layer_q

                .evaluate::<GoldilocksExt2>(&input_q.as_wrapped(),&cache)
                .expect("quantized absolute evaluate should succeed")
                .outputs
                .pop()
                .unwrap();

            let expected = add_q

                .evaluate::<GoldilocksExt2>(&[&input_q.as_wrapped(), &sub_pos_q.as_wrapped()], )
                .expect("quantized add evaluate should succeed")
                .outputs
                .pop()
                .unwrap();
            prop_assert_eq!(out.to_native(), expected.to_native());
        }

        #[test]
        fn test_absolute_padding_prop(input in input::<Element>()) {
            let Input { seq_len, embedding_size, pos: positional_matrix, .. } = input.clone();

            let layer = Absolute::<Element>::new(positional_matrix.clone());

            let mut si = ShapeInfo::from(vec![ShapeData::new(vec![seq_len, embedding_size].into())].as_slice());
            let padded_layer = PadOp::pad_node(layer, &mut si).expect("pad_node should succeed");

            let padded_shape = padded_layer.positional.shape();
            prop_assert_eq!(&padded_layer.unpadded_shape, positional_matrix.shape());
            prop_assert_eq!(padded_shape, &positional_matrix.shape().next_power_of_two());

            for i in 0..padded_shape[0] {
                for j in 0..padded_shape[1] {
                    if i < padded_layer.unpadded_shape[0] && j < padded_layer.unpadded_shape[1] {
                        prop_assert_eq!(padded_layer.positional.get_2d(i, j), positional_matrix.get_2d(i, j));
                    } else {
                        prop_assert_eq!(padded_layer.positional.get_2d(i, j), 0);
                    }
                }
            }
        }

        #[test]
        fn test_absolute_proving_prop(input in input::<f32>()) {
            let Input { seq_len, embedding_size, pos: positional_matrix, .. } = input.clone();
            prop_assume!(seq_len >= 2 && embedding_size >= 2);

            let input_shape = vec![seq_len, embedding_size];
            let mut model = Model::new_from_input_shapes(vec![input_shape.into()], PaddingMode::NoPadding);

            model
                .add_consecutive_layer(Layer::Positional(Positional::new_absolute(positional_matrix)), None)
                .expect("add layer");
            model.automatic_output_labelling().expect("route output");

            let _ = prove_model(model, &mut GenStore::default()).expect("prove model");
        }
    }
}
