use std::{
    iter::once,
    sync::{Arc, Mutex},
};

use anyhow::{Context, ensure};
use ff_ext::ExtensionField;
use mpcs::PolynomialCommitmentScheme;
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use tenstore::GenStore;
use transcript::Transcript;

use crate::{
    Claim, Element, Prover, ScalingFactor, ScalingStrategy, Shape, Tensor,
    iop::{
        context::{ContextAux, ShapeStep},
        verifier::Verifier,
    },
    layers::{
        LayerCtx, LayerProof,
        add::{Add, AddCtx, AddProof},
        provable::{
            Evaluate, LayerOut, NodeId, PadOp, ProveInfo, QuantizeOp, QuantizeOutput, VerifiableCtx,
        },
        transformer::positional::{Positional, PositionalCache, PositionalCtx, PositionalProof},
    },
    model::StepData,
    quantization::TensorFielder,
    tensor::{Number, TensorSlice},
};

/// Data structure containing the proof data for the absolute variant of positional encoding layer
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AbsoluteProof<E> {
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
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Absolute<N> {
    pub(crate) positional: Tensor<N>,
    pub(super) unpadded_shape: Shape,
    add_layer: Add<N>,
}

impl<N: Number> Absolute<N> {
    fn num_vars(&self) -> usize {
        let num_vars = self.positional.shape().num_vars_2d();
        num_vars.0 + num_vars.1
    }

    pub(super) fn new(matrix: Tensor<N>) -> Self {
        let unpadded_shape = matrix.shape().clone();
        Self {
            positional: matrix,
            unpadded_shape,
            add_layer: Add::new(),
        }
    }

    pub(super) fn evaluate<E: ExtensionField>(
        &self,
        input: &Tensor<N>,
        unpadded_input_shape: &Shape,
        positional_cache: &Arc<Mutex<PositionalCache>>,
    ) -> anyhow::Result<LayerOut<N, E>>
    where
        Add<N>: Evaluate<N>,
    {
        let past_length = positional_cache.lock().unwrap().seq_len;
        let sub_pos = self
            .positional
            .slice_2d(past_length, past_length + input.shape()[0]);
        positional_cache
            .lock()
            .unwrap()
            .set_seq_len(past_length + unpadded_input_shape[0])?;
        let output = self
            .add_layer
            .evaluate::<E>(&[input, &sub_pos], &vec![self.unpadded_shape.clone(); 2])?
            .outputs
            .pop()
            .context("Expected at least 1 output from add in positional encoding layer")?;
        Ok(LayerOut::from_vec(vec![output]))
    }
}

impl Absolute<f32> {
    pub(super) fn quantize<S: ScalingStrategy>(
        self,
        data: &S::AuxData,
        node_id: NodeId,
        input_scaling: ScalingFactor,
    ) -> anyhow::Result<QuantizeOutput<Absolute<Element>>> {
        // quantize positional matrix
        let max = self.positional.max_abs_output();
        let pos_scaling = ScalingFactor::from_absolute_max(max, None);

        let quantized_add =
            self.add_layer
                .quantize_op::<S>(data, node_id, &[input_scaling, pos_scaling])?;

        let quantized_pos = Absolute {
            positional: self.positional.to_quantized(&pos_scaling),
            unpadded_shape: self.unpadded_shape,
            add_layer: quantized_add.quantized_op,
        };

        Ok(QuantizeOutput {
            quantized_op: quantized_pos,
            output_scalings: quantized_add.output_scalings,
            requant_layer: quantized_add.requant_layer,
        })
    }
}

impl PadOp for Absolute<Element> {
    fn pad_node(mut self, _si: &mut crate::padding::ShapeInfo) -> anyhow::Result<Self>
    where
        Self: Sized,
    {
        self.positional = self.positional.pad_next_power_of_two();
        Ok(self)
    }
}

const POSITIONAL_POLY_ID: &str = "PositionalMatrix";

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
                    POSITIONAL_POLY_ID.to_string(),
                    self.positional.pad_next_power_of_two().into_data(),
                )))
                .collect(),
        );

        let ctx = AbsoluteCtx {
            add_ctx,
            unpadded_shape: self.unpadded_shape.clone(),
            num_vars_positional_matrix: self.num_vars(),
            node_id: id,
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
        step_data: &StepData<E, E>,
        prover: &mut Prover<E, T, PCS>,
        store: &mut GenStore,
    ) -> anyhow::Result<Vec<Claim<E>>>
    where
        PCS::CommitmentWithWitness: Serialize + DeserializeOwned + Send + Sync,
        PCS::ProverParam: Send + Sync,
    {
        let input = &step_data.node_inputs[0];

        // derive sub-matrix to be added to input. ToDo: place it in proving data
        let matrix_slice = TensorSlice::from(&self.positional);
        let input = input.hydrate(store.clone())?;
        let sub_pos = matrix_slice
            .slice_over_first_dim(0, input.shape()[0])
            .to_fields();

        let (mut claims, add_proof) =
            self.add_layer
                .prove_step(node_id, vec![output_claim], &[&input, &sub_pos], prover)?;

        ensure!(
            claims.len() == 2,
            "Expected 2 claims from Add proving in position layer, found {} claims",
            claims.len(),
        );

        let sub_pos_claim = claims.pop().unwrap();
        let input_claim = claims.pop().unwrap();

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
            [(POSITIONAL_POLY_ID.to_string(), positional_matrix_claim)]
                .into_iter()
                .collect(),
        );

        prover.push_proof(
            node_id,
            LayerProof::Positional(PositionalProof::Absolute(AbsoluteProof {
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
            .shape_step(&unpadded_input_shapes, &padded_input_shapes);

        let mut claims =
            self.add_ctx
                .verify(&proof.add_proof, &[output_claim], verifier, &shape_step)?;

        ensure!(
            claims.len() == 2,
            "Expected 2 claims from Add verifier in position layer, found {} claims",
            claims.len(),
        );

        let sub_pos_claim = claims.pop().unwrap();

        let input_claim = claims.pop().unwrap();

        let positional_matrix_claim = PositionalCtx::build_positional_matrix_claim(
            sub_pos_claim,
            output_claim,
            self.num_vars_positional_matrix,
            verifier.transcript,
            &proof.sub_matrix_evals,
        )?;

        verifier.add_common_claims(
            self.node_id,
            [(POSITIONAL_POLY_ID.to_string(), positional_matrix_claim)]
                .into_iter()
                .collect(),
        );

        Ok(vec![input_claim])
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use tenstore::GenStore;

    use crate::{
        Tensor,
        layers::{Layer, transformer::positional::Positional},
        model::{Model, test::prove_model},
        padding::PaddingMode,
    };

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
        let positional_matrix = Tensor::random(&matrix_shape.into());

        let _ = model
            .add_consecutive_layer(
                Layer::Positional(Positional::new_absolute(positional_matrix)),
                None,
            )
            .unwrap();

        model.route_output(None).unwrap();

        let _ = prove_model(model, &mut GenStore::default()).unwrap();
    }
}
