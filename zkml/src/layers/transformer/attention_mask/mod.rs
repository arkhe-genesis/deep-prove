//! Implementation of various types of attention masks for transformer models.
use crate::{
    Claim, Element, Number, ScalingFactor, Shape, Tensor,
    commit::compute_betas_eval,
    eval_zeroifier_mle,
    graph::NodeId,
    iop::{
        context::{ContextAux, ShapeStep},
        prover::Prover,
        verifier::Verifier,
    },
    layers::{
        LayerCtx, LayerProof,
        provable::{
            Evaluate, LayerOut, OpInfo, PadOp, ProvableOp, ProveInfo, QuantizeOp, QuantizeOutput,
            VerifiableCtx,
        },
    },
    model::Step,
    padding::PaddingMode,
    quantization::{Fieldizer, TensorFielder},
    tensor::{TensorTypeParam, WrappedTensor},
    to_bit_sequence_le,
};
use anyhow::{Result, ensure};
use burn::tensor::Tensor as BTensor;
use either::Either;
use ff_ext::ExtensionField;
use itertools::Itertools;
use mpcs::PolynomialCommitmentScheme;
use multilinear_extensions::{
    Expression,
    mle::MultilinearExtension,
    util::ceil_log2,
    virtual_poly::{VPAuxInfo, eq_eval},
    virtual_polys::VirtualPolynomialsBuilder,
};
use p3_field::FieldAlgebra;
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use sumcheck::{
    structs::{IOPProof, IOPProverState, IOPVerifierState},
    util::optimal_sumcheck_threads,
};

use transcript::Transcript;

pub mod evaluate;
pub mod prove;
pub(crate) use prove::MaskProvingData;
pub mod verify;
pub(crate) use verify::MaskVerifyingData;

/// The short name used to identify the attention mask layer
pub const ATTENTION_MASK_LAYER: &str = "MASK";

/// Mask used in attention so that tokens can only see certain values.
/// Masks are assumed to always operate on matrices, if a tensor has higher rank than 2
/// the mask will be applied to the last two dimensions repeatedly.
/// NOTE: the attention mask supports the single token inference with caching AND
/// the regular inference with square matrices for EVALUATION only.
/// Attention mask proving logic only works for the regular inference with square matrices,
/// and that's ok since we prove "full sequence length", we dont prove intermediate
/// cached inferences for now.
#[derive(Clone, Debug, Serialize, Deserialize, Copy)]
pub struct AttentionMask<N> {
    /// Since a casual mask is always square this is the dim size for both the rows and the columns
    pub dim_size: usize,
    span: AttentionSpan,
    /// The value for negative infinity
    negative_infinity: N,
}

/// AttentionSpan determines which entries do the mask keep.
#[derive(Clone, Debug, Serialize, Deserialize, Copy, Default, PartialEq, Eq)]
pub enum AttentionSpan {
    /// The mask is applied to each previous token of the one of interest
    #[default]
    Full,
    /// The mask is applied only to the last previous `n` tokens of the one of interest
    Local(usize),
}

impl<N: TensorTypeParam> Default for AttentionMask<N> {
    fn default() -> Self {
        AttentionMask {
            negative_infinity: <N as Number>::MIN,
            dim_size: Default::default(),
            span: Default::default(),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound(serialize = "E: Serialize", deserialize = "E: DeserializeOwned"))]
/// Mask used in attention so that tokens can only see "previous" values.
pub struct AttentionMaskProof<E: ExtensionField> {
    /// The sumcheck proof for correct application of the attention mask
    sumcheck_proof: IOPProof<E>,
    /// The input evaluations
    evaluations: Vec<E>,
}

impl<N: TensorTypeParam> AttentionMask<N> {
    /// Creates a new mask given the unpadded input shape and the value to use for `-inf`
    pub fn new(dim_size: usize, negative_inf: N) -> AttentionMask<N> {
        AttentionMask {
            dim_size,
            negative_infinity: negative_inf,
            span: Default::default(),
        }
    }
    pub fn with_span(self, span: AttentionSpan) -> anyhow::Result<AttentionMask<N>> {
        if let AttentionSpan::Local(n) = span {
            ensure!(
                n < self.dim_size,
                "Span cannot be greater than the dimension size"
            );
        }
        Ok(AttentionMask { span, ..self })
    }

    /// Sets the negative infinity value
    pub fn set_negative_infinity(&mut self, negative_infinity: N) {
        self.negative_infinity = negative_infinity;
    }
}

impl<T: TensorTypeParam> Evaluate<T> for AttentionMask<T> {
    fn evaluate<E: ExtensionField>(&self, inputs: &[&WrappedTensor<T>]) -> Result<LayerOut<T, E>> {
        inputs
            .iter()
            .map(|input| self.evaluate_internal(input))
            .collect::<Result<Vec<WrappedTensor<T>>>>()
            .map(LayerOut::from_vec)
    }
}

impl PadOp for AttentionMask<Element> {}

impl<N: TensorTypeParam> OpInfo for AttentionMask<N> {
    fn output_shapes(
        &self,
        input_shapes: &[Shape],
        _padding_mode: PaddingMode,
    ) -> Result<Vec<Shape>> {
        Ok(input_shapes.to_vec())
    }

    fn num_outputs(&self, num_inputs: usize) -> Result<usize> {
        Ok(num_inputs)
    }

    fn describe(&self) -> String {
        format!("CasualMask(neg_inf: {:?})", self.negative_infinity)
    }

    fn is_provable(&self) -> bool {
        true
    }
}

impl QuantizeOp for AttentionMask<f32> {
    type QuantizedOp = AttentionMask<Element>;

    fn quantize_op<S: crate::ScalingStrategy>(
        self,
        _data: &S::AuxData,
        _node_id: NodeId,
        input_scaling: &[ScalingFactor],
        _unpadded_input_shapes: &[Shape],
        _output_scaling: &[ScalingFactor],
        _unpadded_output_shapes: &[Shape],
    ) -> Result<QuantizeOutput<Self::QuantizedOp>> {
        // Ensure we have some scaling factors
        ensure!(
            !input_scaling.is_empty(),
            "Cannot quantize Casual Mask as no input scaling factors are provided"
        );

        // If we have multiple input scaling factors we need to make sure all of their quantised domains have the same minimum
        ensure!(
            input_scaling.iter().map(|sf| sf.domain().0).all_equal(),
            "Cannot quantize Casual Mask as not all inputs have the same minimum quantised domain"
        );

        // We just have to replace negative_infinity with its quantized version
        let quantized_negative_infinity = input_scaling
            .first()
            .map(|sf| sf.domain().0)
            .expect("We have ensured there is at least one scaling factor");
        let AttentionMask { dim_size, span, .. } = self;

        let quantised_mask =
            AttentionMask::<Element>::new(dim_size, quantized_negative_infinity).with_span(span)?;

        Ok(QuantizeOutput::new(quantised_mask, input_scaling.to_vec()))
    }
}

impl ProveInfo for AttentionMask<Element> {
    fn step_info<E: ExtensionField>(
        &self,
        id: NodeId,
        mut aux: ContextAux,
    ) -> Result<(LayerCtx<E>, ContextAux)> {
        // We need the previous output shapes to compute the sumcheck expressions
        let shapes = &aux.last_output_shape;

        ensure!(
            shapes.len() == 1,
            "Casual Mask can only be applied to a single input"
        );
        let input_shape = &shapes[0];
        let negative_inf = self.negative_infinity;
        let sumcheck_expression = build_sumcheck_expression::<E>(input_shape, negative_inf);
        let layer_ctx =
            LayerCtx::AttentionMask(AttentionMaskCtx::<E>::new(*self, sumcheck_expression, id));

        // We make sure to update the ContextAux so that it knows there are no model commitments for this layer
        aux.model_polys = None;

        Ok((layer_ctx, aux))
    }
}

fn build_sumcheck_expression<E: ExtensionField>(
    input_shape: &Shape,
    negative_inf: Element,
) -> Vec<Expression<E>> {
    // If the input shape has rank greater than two we treat it as a batch of 2D images.
    // Since we handle it in this way we make the eq_poly and less than poly the first two witness indices.
    let eq_expr = Expression::<E>::WitIn(0);
    let mask_expr = Expression::<E>::WitIn(1);
    // This expr ensures we only apply the mask to the unpadded portion of each row of the input
    let row_lt_expr = Expression::<E>::WitIn(2);
    // This expr ensures we only apply the mask to the unpadded portion of each column of the input
    let column_lt_expr = Expression::<E>::WitIn(3);
    let lt_expr = row_lt_expr * column_lt_expr;
    let one_minus_mask_expr = Expression::Constant(Either::Right(E::ONE)) - mask_expr.clone();
    // The input offset is 4 since we have eq_poly, mask_poly and two less than polys before the input polys start
    const INPUT_OFFSET: u16 = 4;
    if input_shape.rank() > 2 {
        let batch_size = input_shape.slice(..input_shape.rank() - 2).numel();
        (0..batch_size as u16)
            .map(|j| {
                let wit_poly_id = j + INPUT_OFFSET;
                // Each term is the same it is just the challenge that changes
                eq_expr.clone()
                    * lt_expr.clone()
                    * Expression::Challenge(j, 1, E::ONE, E::ZERO)
                    * (mask_expr.clone() * Expression::WitIn(wit_poly_id)
                        + Expression::Constant(Either::Right(negative_inf.to_field()))
                            * one_minus_mask_expr.clone())
            })
            .collect()
    } else {
        vec![
            eq_expr
                * lt_expr
                * (mask_expr * Expression::WitIn(INPUT_OFFSET)
                    + Expression::Constant(Either::Right(negative_inf.to_field()))
                        * one_minus_mask_expr),
        ]
    }
}

/// Context for the attention mask operation, needed by the verifier to check a mask was applied correctly.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound(serialize = "E: Serialize", deserialize = "E: DeserializeOwned"))]
pub struct AttentionMaskCtx<E: ExtensionField> {
    pub op: AttentionMask<Element>,
    pub sumcheck_expression: Vec<Expression<E>>,
    pub node_id: NodeId,
}

impl<E: ExtensionField> AttentionMaskCtx<E> {
    /// Create a new [`AttentionMaskCtx`]
    pub fn new(
        op: AttentionMask<Element>,
        sumcheck_expression: Vec<Expression<E>>,
        node_id: NodeId,
    ) -> Self {
        AttentionMaskCtx {
            op,
            sumcheck_expression,
            node_id,
        }
    }

    /// Getter for the dim size for the mask
    pub fn dim_size(&self) -> usize {
        self.op.dim_size
    }
}

impl<E: ExtensionField> OpInfo for AttentionMaskCtx<E> {
    fn output_shapes(
        &self,
        input_shapes: &[Shape],
        _padding_mode: PaddingMode,
    ) -> Result<Vec<Shape>> {
        Ok(input_shapes.to_vec())
    }

    fn num_outputs(&self, num_inputs: usize) -> Result<usize> {
        Ok(num_inputs)
    }

    fn describe(&self) -> String {
        format!(
            "CasualMask {{ neg_inf: {:?}, span: {:?} }}",
            self.op.negative_infinity, self.op.span
        )
    }

    fn is_provable(&self) -> bool {
        true
    }
}

impl<E, PCS> ProvableOp<E, PCS> for AttentionMask<Element>
where
    E: ExtensionField,
    PCS: PolynomialCommitmentScheme<E> + Send + Sync,
    PCS::CommitmentWithWitness: Serialize + DeserializeOwned + Send + Sync,
    PCS::ProverParam: Send + Sync,
{
    type Ctx = AttentionMaskCtx<E>;

    fn prove<'a, 'b, 'c, 'd, T: transcript::Transcript<E>>(
        &'a self,
        node_id: NodeId,
        ctx: &'b Self::Ctx,
        last_claims: Vec<&Claim<E>>,
        step_data: &Step<E, Element>,
        prover: &mut Prover<'c, 'd, E, T, PCS>,
    ) -> Result<Vec<Claim<E>>> {
        let inputs = step_data.input_tensors()?;
        let unpadded_seq_len =
            *step_data.unpadded_input_shapes[0]
                .last()
                .ok_or(anyhow::anyhow!(
                    "Input shape must have at least one dimension"
                ))?;

        ensure!(
            inputs.len() == 1,
            "CasualMask can only be applied to a single input"
        );

        ensure!(
            last_claims.len() == 1,
            "Expected a single claim for AttentionMask proving, got {}",
            last_claims.len()
        );

        let mask_proving_data =
            MaskProvingData::from_claims_and_input(last_claims[0], &inputs[0].to_fields())?;

        let (proof, claims) =
            self.prove_internal(ctx, mask_proving_data, unpadded_seq_len, prover)?;
        prover.push_proof(node_id, LayerProof::AttentionMask(proof));
        Ok(claims)
    }
}

impl<E, PCS> VerifiableCtx<E, PCS> for AttentionMaskCtx<E>
where
    E: ExtensionField,
    PCS: PolynomialCommitmentScheme<E>,
{
    type Proof = AttentionMaskProof<E>;

    fn verify<T: Transcript<E>>(
        &self,
        proof: &Self::Proof,
        last_claims: &[&Claim<E>],
        verifier: &mut Verifier<E, T, PCS>,
        shape_step: &ShapeStep,
    ) -> Result<Vec<Claim<E>>> {
        let input_shapes = &shape_step.padded_input_shape;
        ensure!(
            input_shapes.len() == 1,
            "CasualMask can only be applied to a single input"
        );
        ensure!(
            input_shapes[0].rank() >= 2,
            "CasualMask can only be applied to rank 2 or higher tensors"
        );
        ensure!(
            last_claims.len() == 1,
            "Expected one claim for AttentionMask verifying, got {}",
            last_claims.len()
        );
        let input_shape = &input_shapes[0];
        let unpadded_input_shape = &shape_step.unpadded_input_shape[0];

        let mask_verifying_data = MaskVerifyingData::new_from_claim_and_shape_data(
            last_claims[0],
            input_shape,
            unpadded_input_shape,
        )?;
        let unpadded_seq_len = unpadded_input_shape.dim(-1);
        self.op.verify_internal(
            proof,
            last_claims[0],
            mask_verifying_data,
            unpadded_seq_len,
            verifier,
        )
    }

    fn write_proof_to_transcript<T: Transcript<E>>(
        &self,
        _proof: &Self::Proof,
        _transcript: &mut T,
    ) -> anyhow::Result<()> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use ark_std::rand::Rng;
    use tenstore::GenStore;

    use crate::{
        layers::{Layer, einsum::EinSum},
        model::{Model, test::prove_model},
        rng_from_env_or_random,
    };

    use super::*;

    #[test]
    fn test_attention_mask_proving() -> Result<()> {
        let mut rng = rng_from_env_or_random();

        let dim_size: usize = rng.gen_range(3..10);

        for rank in 3..4 {
            println!("Testing rank: {rank}: DIM SIZE: {dim_size}");
            let mask = AttentionMask::new(dim_size, f32::NEG_INFINITY);
            test_causal_mask_proving_helper(rank, mask)?;
        }
        Ok(())
    }

    fn test_causal_mask_proving_helper(rank: usize, mask: AttentionMask<f32>) -> Result<()> {
        let mut rng = rng_from_env_or_random();

        let dim_size = mask.dim_size;

        let shape: Shape = (0..rank - 2)
            .map(|_| rng.gen_range(3..8))
            .chain([dim_size, dim_size])
            .collect::<Vec<usize>>()
            .into();
        println!("Testing mask on shape: {shape:?}");

        let common_chars = ('a'..).take(rank - 2).collect::<Vec<char>>();
        let mut left_chars = common_chars.clone();
        let mut right_chars = common_chars.clone();
        let mut output_chars = common_chars;
        left_chars.push('x');
        left_chars.push('z');
        right_chars.push('z');
        right_chars.push('y');
        output_chars.push('x');
        output_chars.push('y');

        let left_dims = left_chars.into_iter().collect::<String>();
        let right_dims = right_chars.into_iter().collect::<String>();
        let output_dims = output_chars.into_iter().collect::<String>();

        let equation = format!("L({left_dims})@R({right_dims})->O({output_dims})");

        let einsum = EinSum::new(equation, vec![None], vec![None])?;

        // we test over a model where EinSum is the first layer, so we need 2 input shapes
        let input_shape_left = shape
            .slice(..rank - 2)
            .extend(&Shape::new(vec![dim_size, 10]));
        let input_shape_right = shape
            .slice(..rank - 2)
            .extend(&Shape::new(vec![10, dim_size]));

        let mut model = Model::new_from_input_shapes(
            vec![input_shape_left, input_shape_right],
            PaddingMode::NoPadding,
        );

        let id = model
            .add_consecutive_layer(Layer::EinSum(einsum), None)
            .unwrap();

        let _ = model
            .add_consecutive_layer(Layer::AttentionMask(mask), Some(id))
            .unwrap();

        model.automatic_output_labelling().unwrap();
        model.describe();
        prove_model(model, &mut GenStore::default()).unwrap();
        Ok(())
    }
}
