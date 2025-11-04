use super::{
    LayerCtx, LayerProof,
    flatten::Flatten,
    requant::Requant,
    transformer::{layernorm::LayerNormData, softmax::SoftmaxData},
};
use crate::{
    Claim, Element, Prover, ProverContext, ScalingFactor, ScalingStrategy, Shape, Tensor,
    graph::NodeId,
    iop::{
        context::{ContextAux, ShapeStep},
        verifier::Verifier,
    },
    layers::{
        activation::ActivationData,
        convolution::ConvFFTData,
        transformer::{logits::ArgmaxData, mha::MhaData},
    },
    lookup::context::LookupWitnessGen,
    model::{trace::Step, transform::ModelTransform},
    padding::{PaddingMode, ShapeInfo},
    tensor::{TensorTypeParam, WrappedTensor},
};
use anyhow::{Result, bail, ensure};
use derive_more::{From, Into};
use ff_ext::ExtensionField;
use mpcs::PolynomialCommitmentScheme;
use multilinear_extensions::mle::IntoMLE;
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use std::{collections::HashMap, fmt::Debug};
use tenstore::GenStore;
use transcript::Transcript;

/// Enum if the output of evaluating a layer returns extra data needed during proving.
/// This should only be implemented for quantised layers.
#[derive(Clone, Debug)]
#[allow(clippy::large_enum_variant)]
pub enum ProvingData<E: ExtensionField> {
    /// Variant for extra data used in proving that we compute during evalaution of quantised convolution.
    Convolution(ConvFFTData),
    /// Variant for extra data used to prove [Softmax][`crate::layers::transformer::softmax::Softmax`] that we compute anyway during quantised evaluation.
    Softmax(SoftmaxData),
    /// Variant for extra data used to prove Mha layer, computed during quantised evaluation
    Mha(MhaData<E>),
    /// Variant used for extra data used to prove [LayerNorm][`crate::layers::transformer::layernorm::LayerNorm`]
    LayerNorm(LayerNormData),
    /// Variant used for extra data used to prove [ArgMax][`crate::layers::transformer::logits::Logits`]
    ArgMax(ArgmaxData<E>),
    /// Variant used for extra data used to prove activation layer
    Activation(ActivationData),
    /// Variant used when no extra data is returned.
    None,
}

/// Identifier for an intermediate tensor of a layer, i.e., a tensor which is neither an
/// input nor an output tensor of the layer. The ID is employed to keep track of the tensor
/// for quantization purposes
#[derive(
    Clone,
    From,
    Into,
    Hash,
    PartialEq,
    Eq,
    Serialize,
    Deserialize,
    PartialOrd,
    Ord,
    derive_more::Debug,
)]
pub struct TrackedDataId(String);

#[derive(Clone, Debug)]
pub struct LayerOut<T: TensorTypeParam, E: ExtensionField> {
    pub(crate) outputs: Vec<WrappedTensor<T>>,
    pub(crate) proving_data: ProvingData<E>,
    pub(crate) tracked_layer_data: Option<HashMap<TrackedDataId, WrappedTensor<T>>>,
}

impl<T: TensorTypeParam, E: ExtensionField> LayerOut<T, E> {
    pub(crate) fn from_vec(out: Vec<WrappedTensor<T>>) -> Self {
        Self {
            outputs: out,
            proving_data: ProvingData::None,
            tracked_layer_data: None,
        }
    }

    pub(crate) fn with_proving_data(self, data: ProvingData<E>) -> Self {
        Self {
            outputs: self.outputs,
            proving_data: data,
            tracked_layer_data: self.tracked_layer_data,
        }
    }

    /// Add a set of intermediate data tensors to be tracked for quantization purposes;
    /// Each intermediate tensor is identified by a corresponding `TrackedDataId`
    pub(crate) fn with_data_to_be_tracked(
        self,
        data: HashMap<TrackedDataId, WrappedTensor<T>>,
    ) -> Self {
        Self {
            outputs: self.outputs,
            proving_data: self.proving_data,
            tracked_layer_data: Some(data),
        }
    }

    pub fn outputs(&self) -> Vec<&WrappedTensor<T>> {
        self.outputs.iter().collect()
    }

    pub fn from_tensor(out: WrappedTensor<T>) -> Self {
        Self::from_vec(vec![out])
    }

    pub fn try_convdata(&self) -> Option<&ConvFFTData> {
        match self.proving_data {
            ProvingData::Convolution(ref conv_data) => Some(conv_data),
            _ => None,
        }
    }

    pub fn try_softmax_data(&self) -> Option<&SoftmaxData> {
        match self.proving_data {
            ProvingData::Softmax(ref softmax_data) => Some(softmax_data),
            _ => None,
        }
    }

    pub fn try_mha_data(&self) -> Option<&MhaData<E>> {
        match self.proving_data {
            ProvingData::Mha(ref mha_data) => Some(mha_data),
            _ => None,
        }
    }

    pub fn try_argmax_data(&self) -> Option<&ArgmaxData<E>> {
        match self.proving_data {
            ProvingData::ArgMax(ref argmax_data) => Some(argmax_data),
            _ => None,
        }
    }

    pub fn try_layernorm_data(&self) -> Option<&LayerNormData> {
        match self.proving_data {
            ProvingData::LayerNorm(ref layernorm_data) => Some(layernorm_data),
            _ => None,
        }
    }
}

pub trait OpInfo {
    /// Returns the shapes of the outputs (in the same order)
    fn output_shapes(
        &self,
        input_shapes: &[Shape],
        padding_mode: PaddingMode,
    ) -> Result<Vec<Shape>>;

    /// Compute the number of output tensors, given the number of input tensors
    /// `num_inputs`
    fn num_outputs(&self, num_inputs: usize) -> Result<usize>;

    /// Textual description of the operation
    fn describe(&self) -> String;

    /// Specify whether the operation needs to be proven or not
    fn is_provable(&self) -> bool;
}

pub trait Evaluate<T: TensorTypeParam> {
    /// Evaluates the operation given any inputs tensors and constant inputs.
    fn evaluate<E: ExtensionField>(&self, inputs: &[&WrappedTensor<T>]) -> Result<LayerOut<T, E>>;
}

/// Helper method employed to call `Evaluate::evaluate` when there are no `unpadded_input_shapes`
/// or when the `E` type cannot be inferred automatically by the compiler
pub fn evaluate_layer<E: ExtensionField, T: TensorTypeParam, O: Evaluate<T>>(
    layer: &O,
    inputs: &[&WrappedTensor<T>],
) -> Result<LayerOut<T, E>> {
    layer.evaluate(inputs)
}

pub trait ProveInfo {
    /// Compute the proving context for the operation
    fn step_info<E: ExtensionField>(
        &self,
        id: NodeId,
        aux: ContextAux,
    ) -> Result<(LayerCtx<E>, ContextAux)>;
}

/// Output of `QuantizeOp` method over a layer
pub struct QuantizeOutput<Op> {
    /// The actual layer after quantization
    pub quantized_op: Op,
    /// The scaling factor of the output wires of the operation
    pub(crate) output_scalings: Vec<ScalingFactor>,
    /// The requant layer to be added to the model, if any
    pub(crate) requant_layer: Option<Vec<Requant>>,
    /// Optional rule to apply after quantisation
    pub(crate) post_quant_rule: Option<Box<dyn ModelTransform<Element>>>,
}

impl<Op> QuantizeOutput<Op> {
    pub fn new(quantized_op: Op, output_scalings: Vec<ScalingFactor>) -> Self {
        Self {
            quantized_op,
            output_scalings,
            requant_layer: None,
            post_quant_rule: None,
        }
    }
    pub fn with_requant(self, requant: Requant) -> Result<Self> {
        ensure!(
            self.output_scalings.len() == 1,
            "Number of output scalings must be 1"
        );
        Self::with_requants(self, vec![requant])
    }
    pub fn with_requants(self, requants: Vec<Requant>) -> Result<Self> {
        ensure!(self.requant_layer.is_none(), "Requant layer already exists");
        ensure!(
            self.output_scalings.len() == requants.len(),
            "Number of output scalings and requants must be the same"
        );
        Ok(Self {
            quantized_op: self.quantized_op,
            output_scalings: self.output_scalings,
            requant_layer: Some(requants),
            post_quant_rule: self.post_quant_rule,
        })
    }
    pub fn with_transform(self, transform: Box<dyn ModelTransform<Element>>) -> Result<Self> {
        ensure!(
            self.post_quant_rule.is_none(),
            "Post quantization rule already exists"
        );
        Ok(Self {
            quantized_op: self.quantized_op,
            output_scalings: self.output_scalings,
            requant_layer: self.requant_layer,
            post_quant_rule: Some(transform),
        })
    }
    pub fn maybe_requants(self, requant: Option<Vec<Requant>>) -> Result<Self> {
        match requant {
            Some(requant) => self.with_requants(requant),
            None => Ok(self),
        }
    }
    pub fn maybe_transform(
        self,
        transform: Option<Box<dyn ModelTransform<Element>>>,
    ) -> Result<Self> {
        match transform {
            Some(transform) => self.with_transform(transform),
            None => Ok(self),
        }
    }
}

pub trait QuantizeOp {
    type QuantizedOp: Sized;

    /// Convert an operation into its quantized version
    fn quantize_op<S: ScalingStrategy>(
        self,
        data: &S::AuxData,
        node_id: NodeId,
        input_scaling: &[ScalingFactor],
        unpadded_input_shapes: &[Shape],
        output_scaling: &[ScalingFactor],
        unpadded_output_shapes: &[Shape],
    ) -> anyhow::Result<QuantizeOutput<Self::QuantizedOp>>;
}

pub trait PadOp {
    // Pad the dimensions of the tensors in node `self`, updating the `ShapeInfo` with the output shapes
    // of the node
    fn pad_node(self, _si: &mut ShapeInfo) -> Result<Self>
    where
        Self: Sized,
    {
        Ok(self)
    }
}

pub trait ProvableOp<E, PCS>: OpInfo + PadOp + ProveInfo
where
    E: ExtensionField,
    E::BaseField: Serialize + DeserializeOwned,
    E: Serialize + DeserializeOwned,
    PCS: PolynomialCommitmentScheme<E>,
    PCS::CommitmentWithWitness: Serialize + DeserializeOwned + Send + Sync,
    PCS::ProverParam: Send + Sync,
{
    type Ctx: VerifiableCtx<E, PCS>;

    /// Produces a proof of correct execution for this operation.
    fn prove<'a, 'b, 'c, 'd, T: Transcript<E>>(
        &'a self,
        _node_id: NodeId,
        _ctx: &'b Self::Ctx,
        _last_claims: Vec<&Claim<E>>,
        _step_data: &Step<E, Element, E>,
        _prover: &mut Prover<'c, 'd, E, T, PCS>,
        _store: &mut GenStore,
    ) -> Result<Vec<Claim<E>>> {
        // Default implementation, to avoid having to implement this method in case `is_provable` is false
        ensure!(
            !self.is_provable(),
            "Running default prove implementation for a provable operation! Implement prove method"
        );
        Ok(vec![Claim::default()])
    }

    /// Generate witness for a node where a lookup table is employed in proving
    fn gen_lookup_witness(
        &self,
        _id: NodeId,
        _ctx: &ProverContext<E, PCS>,
        _step_data: &Step<E, Element, Element>,
        _store: &mut GenStore,
    ) -> Result<LookupWitnessGen<E, PCS>> {
        Ok(Default::default())
    }
}

pub trait VerifiableCtx<E, PCS>: Debug + OpInfo
where
    E: ExtensionField,
    E::BaseField: Serialize + DeserializeOwned,
    E: Serialize + DeserializeOwned,
    PCS: PolynomialCommitmentScheme<E>,
{
    type Proof: Sized;

    /// Verify proof for the given operation
    fn verify<T: Transcript<E>>(
        &self,
        proof: &Self::Proof,
        last_claims: &[&Claim<E>],
        verifier: &mut Verifier<E, T, PCS>,
        shape_step: &ShapeStep,
    ) -> Result<Vec<Claim<E>>>;

    /// Verify the claim about the input of the model. Sometimes
    /// the input needs to be processed in a certain way before being evaluated.
    /// For example, Embeddings use one hot encoding of the input before
    /// running the matmul protocol.
    /// By default, it simply evaluates the input against the input claim.
    fn verify_input_claim<A: AsRef<Tensor<E>>>(
        &self,
        inputs: &[A],
        claims: &[&Claim<E>],
    ) -> anyhow::Result<()> {
        ensure!(
            inputs.len() == claims.len(),
            "number of input tensors and claims must be the same"
        );
        for (i, (input, claim)) in inputs.iter().zip(claims).enumerate() {
            let computed = input
                .as_ref()
                .get_data()
                .to_vec()
                .into_mle()
                .evaluate(&claim.point);
            ensure!(
                computed == claim.eval,
                "input claim {:?} is incorrect, computed: {:?}, given: {:?}",
                i,
                computed,
                claim.eval,
            );
        }
        Ok(())
    }

    /// Writes the associated type [`Self::Proof`] to the transcript if it contains any [`PCS::Commitment`].
    fn write_proof_to_transcript<T: Transcript<E>>(
        &self,
        proof: &Self::Proof,
        transcript: &mut T,
    ) -> anyhow::Result<()>;
}

pub(crate) fn verify_input_claim<E, PCS, V, A>(
    ctx: &V,
    inputs: &[A],
    claims: &[&Claim<E>],
) -> anyhow::Result<()>
where
    V: VerifiableCtx<E, PCS>,
    E: ExtensionField,
    PCS: PolynomialCommitmentScheme<E>,
    A: AsRef<Tensor<E>>,
{
    <V as VerifiableCtx<E, PCS>>::verify_input_claim(ctx, inputs, claims)
}

pub(crate) fn write_proof_to_transcript<
    E: ExtensionField,
    PCS: PolynomialCommitmentScheme<E>,
    T: Transcript<E>,
    V: VerifiableCtx<E, PCS>,
>(
    ctx: &V,
    proof: &<V as VerifiableCtx<E, PCS>>::Proof,
    transcript: &mut T,
) -> anyhow::Result<()> {
    <V as VerifiableCtx<E, PCS>>::write_proof_to_transcript(ctx, proof, transcript)
}

#[derive(Clone, Debug)]
pub(crate) struct NonProvableVerifierCtx<'a, O>(&'a O);

impl<'a, O: OpInfo> OpInfo for NonProvableVerifierCtx<'a, O> {
    fn output_shapes(
        &self,
        input_shapes: &[Shape],
        padding_mode: PaddingMode,
    ) -> Result<Vec<Shape>> {
        self.0.output_shapes(input_shapes, padding_mode)
    }

    fn num_outputs(&self, num_inputs: usize) -> Result<usize> {
        self.0.num_outputs(num_inputs)
    }

    fn describe(&self) -> String {
        self.0.describe()
    }

    fn is_provable(&self) -> bool {
        false
    }
}

impl<'a, O: OpInfo + Debug, E: ExtensionField, PCS: PolynomialCommitmentScheme<E>>
    VerifiableCtx<E, PCS> for NonProvableVerifierCtx<'a, O>
{
    type Proof = ();

    fn verify<T: Transcript<E>>(
        &self,
        _proof: &Self::Proof,
        _last_claims: &[&Claim<E>],
        _verifier: &mut Verifier<E, T, PCS>,
        _shape_step: &ShapeStep,
    ) -> Result<Vec<Claim<E>>> {
        // Default implementation, to avoid having to implement this method in case `is_provable` is false
        ensure!(
            !self.is_provable(),
            "Running default prove implementation for a provable operation! Implement prove method"
        );
        Ok(vec![Claim::default()])
    }

    fn write_proof_to_transcript<T: Transcript<E>>(
        &self,
        _proof: &Self::Proof,
        _transcript: &mut T,
    ) -> anyhow::Result<()> {
        // No commitment so just return Ok(())
        Ok(())
    }
}

impl<E: ExtensionField> OpInfo for LayerCtx<E> {
    fn output_shapes(
        &self,
        input_shapes: &[Shape],
        padding_mode: PaddingMode,
    ) -> Result<Vec<Shape>> {
        match self {
            LayerCtx::Dense(dense_ctx) => dense_ctx.output_shapes(input_shapes, padding_mode),
            LayerCtx::Convolution(conv_ctx) => conv_ctx.output_shapes(input_shapes, padding_mode),
            LayerCtx::MatMul(mat_ctx) => mat_ctx.output_shapes(input_shapes, padding_mode),
            LayerCtx::QKV(qkv_ctx) => qkv_ctx.output_shapes(input_shapes, padding_mode),
            LayerCtx::Mha(mha_ctx) => mha_ctx.output_shapes(input_shapes, padding_mode),
            LayerCtx::ConcatMatMul(ctx) => ctx.output_shapes(input_shapes, padding_mode),
            LayerCtx::Positional(ctx) => ctx.output_shapes(input_shapes, padding_mode),
            LayerCtx::Add(ctx) => ctx.output_shapes(input_shapes, padding_mode),
            LayerCtx::LayerNorm(ctx) => ctx.output_shapes(input_shapes, padding_mode),
            LayerCtx::RMSNorm(ctx) => ctx.output_shapes(input_shapes, padding_mode),
            LayerCtx::Softmax(softmax_ctx) => softmax_ctx.output_shapes(input_shapes, padding_mode),
            LayerCtx::Logits(ctx) => ctx.output_shapes(input_shapes, padding_mode),
            LayerCtx::Embeddings(ctx) => ctx.output_shapes(input_shapes, padding_mode),
            LayerCtx::Reshape(ctx) => ctx.output_shapes(input_shapes, padding_mode),
            LayerCtx::Activation(activation_ctx) => {
                activation_ctx.output_shapes(input_shapes, padding_mode)
            }
            LayerCtx::Requant(requant_ctx) => requant_ctx.output_shapes(input_shapes, padding_mode),
            LayerCtx::Pooling(pooling_ctx) => pooling_ctx.output_shapes(input_shapes, padding_mode),
            LayerCtx::Flatten => {
                <Flatten as OpInfo>::output_shapes(&Flatten, input_shapes, padding_mode)
            }
            LayerCtx::AttentionMask(attention_mask_ctx) => {
                attention_mask_ctx.output_shapes(input_shapes, padding_mode)
            }
            LayerCtx::EinSum(einsum_ctx) => einsum_ctx.output_shapes(input_shapes, padding_mode),
        }
    }

    fn num_outputs(&self, num_inputs: usize) -> Result<usize> {
        match self {
            LayerCtx::Dense(dense_ctx) => dense_ctx.num_outputs(num_inputs),
            LayerCtx::Convolution(conv_ctx) => conv_ctx.num_outputs(num_inputs),
            LayerCtx::MatMul(mat_ctx) => mat_ctx.num_outputs(num_inputs),
            LayerCtx::QKV(qkv_ctx) => qkv_ctx.num_outputs(num_inputs),
            LayerCtx::Mha(mha_ctx) => mha_ctx.num_outputs(num_inputs),
            LayerCtx::ConcatMatMul(ctx) => ctx.num_outputs(num_inputs),
            LayerCtx::Positional(ctx) => ctx.num_outputs(num_inputs),
            LayerCtx::Add(ctx) => ctx.num_outputs(num_inputs),
            LayerCtx::LayerNorm(ctx) => ctx.num_outputs(num_inputs),
            LayerCtx::RMSNorm(ctx) => ctx.num_outputs(num_inputs),
            LayerCtx::Softmax(softmax_ctx) => softmax_ctx.num_outputs(num_inputs),
            LayerCtx::Logits(ctx) => ctx.num_outputs(num_inputs),
            LayerCtx::Embeddings(ctx) => ctx.num_outputs(num_inputs),
            LayerCtx::Reshape(ctx) => ctx.num_outputs(num_inputs),
            LayerCtx::Activation(activation_ctx) => activation_ctx.num_outputs(num_inputs),
            LayerCtx::Requant(requant_ctx) => requant_ctx.num_outputs(num_inputs),
            LayerCtx::Pooling(pooling_ctx) => pooling_ctx.num_outputs(num_inputs),
            LayerCtx::Flatten => <Flatten as OpInfo>::num_outputs(&Flatten, num_inputs),
            LayerCtx::AttentionMask(attention_mask_ctx) => {
                attention_mask_ctx.num_outputs(num_inputs)
            }
            LayerCtx::EinSum(einsum_ctx) => einsum_ctx.num_outputs(num_inputs),
        }
    }

    fn describe(&self) -> String {
        match self {
            LayerCtx::Dense(dense_ctx) => dense_ctx.describe(),
            LayerCtx::Convolution(conv_ctx) => conv_ctx.describe(),
            LayerCtx::MatMul(mat_ctx) => mat_ctx.describe(),
            LayerCtx::QKV(qkv_ctx) => qkv_ctx.describe(),
            LayerCtx::Mha(mha_ctx) => mha_ctx.describe(),
            LayerCtx::ConcatMatMul(ctx) => ctx.describe(),
            LayerCtx::Add(ctx) => ctx.describe(),
            LayerCtx::Positional(ctx) => ctx.describe(),
            LayerCtx::LayerNorm(ctx) => ctx.describe(),
            LayerCtx::RMSNorm(ctx) => ctx.describe(),
            LayerCtx::Softmax(softmax_ctx) => softmax_ctx.describe(),
            LayerCtx::Logits(ctx) => ctx.describe(),
            LayerCtx::Embeddings(ctx) => ctx.describe(),
            LayerCtx::Reshape(ctx) => ctx.describe(),
            LayerCtx::Activation(activation_ctx) => activation_ctx.describe(),
            LayerCtx::Requant(requant_ctx) => requant_ctx.describe(),
            LayerCtx::Pooling(pooling_ctx) => pooling_ctx.describe(),
            LayerCtx::Flatten => Flatten.describe(),
            LayerCtx::AttentionMask(attention_mask_ctx) => attention_mask_ctx.describe(),
            LayerCtx::EinSum(einsum_ctx) => einsum_ctx.describe(),
        }
    }

    fn is_provable(&self) -> bool {
        match self {
            LayerCtx::Dense(dense_ctx) => dense_ctx.is_provable(),
            LayerCtx::Convolution(conv_ctx) => conv_ctx.is_provable(),
            LayerCtx::MatMul(mat_ctx) => mat_ctx.is_provable(),
            LayerCtx::QKV(qkv_ctx) => qkv_ctx.is_provable(),
            LayerCtx::Mha(mha_ctx) => mha_ctx.is_provable(),
            LayerCtx::ConcatMatMul(ctx) => ctx.is_provable(),
            LayerCtx::Activation(activation_ctx) => activation_ctx.is_provable(),
            LayerCtx::Positional(ctx) => ctx.is_provable(),
            LayerCtx::Add(ctx) => ctx.is_provable(),
            LayerCtx::LayerNorm(ctx) => ctx.is_provable(),
            LayerCtx::RMSNorm(ctx) => ctx.is_provable(),
            LayerCtx::Softmax(softmax_ctx) => softmax_ctx.is_provable(),
            LayerCtx::Logits(ctx) => ctx.is_provable(),
            LayerCtx::Embeddings(ctx) => ctx.is_provable(),
            LayerCtx::Reshape(ctx) => ctx.is_provable(),
            LayerCtx::Requant(requant_ctx) => requant_ctx.is_provable(),
            LayerCtx::Pooling(pooling_ctx) => pooling_ctx.is_provable(),
            LayerCtx::Flatten => Flatten.is_provable(),
            LayerCtx::AttentionMask(attention_mask_ctx) => attention_mask_ctx.is_provable(),
            LayerCtx::EinSum(einsum_ctx) => einsum_ctx.is_provable(),
        }
    }
}

impl<E: ExtensionField, PCS: PolynomialCommitmentScheme<E>> VerifiableCtx<E, PCS> for LayerCtx<E>
where
    E::BaseField: Serialize + DeserializeOwned,
    E: Serialize + DeserializeOwned,
{
    type Proof = LayerProof<E, PCS>;

    fn verify<T: Transcript<E>>(
        &self,
        proof: &LayerProof<E, PCS>,
        last_claims: &[&Claim<E>],
        verifier: &mut Verifier<E, T, PCS>,
        shape_step: &ShapeStep,
    ) -> Result<Vec<Claim<E>>> {
        match (self, proof) {
            (LayerCtx::Dense(dense_ctx), LayerProof::Dense(proof)) => {
                dense_ctx.verify(proof, last_claims, verifier, shape_step)
            }
            (LayerCtx::Convolution(conv_ctx), LayerProof::Convolution(proof)) => {
                conv_ctx.verify(proof, last_claims, verifier, shape_step)
            }
            (LayerCtx::MatMul(matmul_ctx), LayerProof::MatMul(proof)) => {
                matmul_ctx.verify(proof, last_claims, verifier, shape_step)
            }
            (LayerCtx::QKV(qkv_ctx), LayerProof::QKV(proof)) => {
                qkv_ctx.verify(proof, last_claims, verifier, shape_step)
            }
            (LayerCtx::ConcatMatMul(matmul_ctx), LayerProof::ConcatMatMul(proof)) => {
                matmul_ctx.verify(proof, last_claims, verifier, shape_step)
            }
            (LayerCtx::Mha(mha_ctx), LayerProof::Mha(proof)) => {
                mha_ctx.verify(proof, last_claims, verifier, shape_step)
            }
            (LayerCtx::Embeddings(ctx), LayerProof::Embeddings(proof)) => {
                ctx.verify(proof, last_claims, verifier, shape_step)
            }
            (LayerCtx::Positional(pos_ctx), LayerProof::Positional(proof)) => {
                pos_ctx.verify(proof, last_claims, verifier, shape_step)
            }
            (LayerCtx::Add(ctx), LayerProof::Add(proof)) => {
                ctx.verify(proof, last_claims, verifier, shape_step)
            }
            (LayerCtx::Logits(ctx), LayerProof::Logits(proof)) => {
                ctx.verify(proof, last_claims, verifier, shape_step)
            }
            (LayerCtx::Activation(activation_ctx), LayerProof::Activation(proof)) => {
                activation_ctx.verify(proof, last_claims, verifier, shape_step)
            }
            (LayerCtx::LayerNorm(layernorm_ctx), LayerProof::LayerNorm(proof)) => {
                layernorm_ctx.verify(proof, last_claims, verifier, shape_step)
            }
            (LayerCtx::RMSNorm(rmsnorm_ctx), LayerProof::RMSNorm(proof)) => {
                rmsnorm_ctx.verify(proof, last_claims, verifier, shape_step)
            }
            (LayerCtx::Requant(requant_ctx), LayerProof::Requant(proof)) => {
                requant_ctx.verify(proof, last_claims, verifier, shape_step)
            }
            (LayerCtx::Pooling(pooling_ctx), LayerProof::Pooling(proof)) => {
                pooling_ctx.verify(proof, last_claims, verifier, shape_step)
            }
            (LayerCtx::Softmax(softmax_ctx), LayerProof::Softmax(proof)) => {
                softmax_ctx.verify(proof, last_claims, verifier, shape_step)
            }
            (LayerCtx::Flatten, _) | (LayerCtx::Reshape(_), _) => {
                unreachable!("Trying to verify a non-provable layer")
            }
            (LayerCtx::AttentionMask(attention_mask_ctx), LayerProof::AttentionMask(proof)) => {
                attention_mask_ctx.verify(proof, last_claims, verifier, shape_step)
            }
            (LayerCtx::EinSum(einsum_ctx), LayerProof::EinSum(proof)) => {
                einsum_ctx.verify(proof, last_claims, verifier, shape_step)
            }
            _ => bail!(
                "Incompatible layer {} and proof {} found",
                self.describe(),
                proof.variant_name()
            ),
        }
    }

    fn verify_input_claim<A: AsRef<Tensor<E>>>(
        &self,
        inputs: &[A],
        claims: &[&Claim<E>],
    ) -> anyhow::Result<()> {
        match self {
            LayerCtx::Dense(dense_ctx) => {
                verify_input_claim::<E, PCS, _, _>(dense_ctx, inputs, claims)
            }
            LayerCtx::Convolution(conv_ctx) => {
                verify_input_claim::<E, PCS, _, _>(conv_ctx, inputs, claims)
            }
            LayerCtx::MatMul(mat_ctx) => {
                verify_input_claim::<E, PCS, _, A>(mat_ctx, inputs, claims)
            }
            LayerCtx::QKV(qkv_ctx) => verify_input_claim::<E, PCS, _, A>(qkv_ctx, inputs, claims),
            LayerCtx::Mha(ctx) => verify_input_claim::<E, PCS, _, A>(ctx, inputs, claims),
            LayerCtx::ConcatMatMul(ctx) => verify_input_claim::<E, PCS, _, A>(ctx, inputs, claims),
            LayerCtx::Activation(activation_ctx) => {
                verify_input_claim::<E, PCS, _, A>(activation_ctx, inputs, claims)
            }
            LayerCtx::LayerNorm(ctx) => verify_input_claim::<E, PCS, _, _>(ctx, inputs, claims),
            LayerCtx::RMSNorm(ctx) => verify_input_claim::<E, PCS, _, _>(ctx, inputs, claims),
            LayerCtx::Softmax(ctx) => verify_input_claim::<E, PCS, _, _>(ctx, inputs, claims),
            LayerCtx::Logits(ctx) => verify_input_claim::<E, PCS, _, _>(ctx, inputs, claims),
            LayerCtx::Embeddings(ctx) => verify_input_claim::<E, PCS, _, _>(ctx, inputs, claims),
            LayerCtx::Add(ctx) => verify_input_claim::<E, PCS, _, _>(ctx, inputs, claims),
            LayerCtx::Positional(ctx) => verify_input_claim::<E, PCS, _, _>(ctx, inputs, claims),
            LayerCtx::Reshape(ctx) => {
                verify_input_claim::<E, PCS, _, _>(&NonProvableVerifierCtx(ctx), inputs, claims)
            }
            LayerCtx::Requant(requant_ctx) => {
                verify_input_claim::<E, PCS, _, _>(requant_ctx, inputs, claims)
            }
            LayerCtx::Pooling(pooling_ctx) => {
                verify_input_claim::<E, PCS, _, _>(pooling_ctx, inputs, claims)
            }
            LayerCtx::Flatten => verify_input_claim::<E, PCS, _, _>(
                &NonProvableVerifierCtx(&Flatten),
                inputs,
                claims,
            ),
            LayerCtx::AttentionMask(ctx) => verify_input_claim::<E, PCS, _, _>(ctx, inputs, claims),
            LayerCtx::EinSum(ctx) => verify_input_claim::<E, PCS, _, _>(ctx, inputs, claims),
        }
    }

    fn write_proof_to_transcript<T: Transcript<E>>(
        &self,
        proof: &Self::Proof,
        transcript: &mut T,
    ) -> anyhow::Result<()> {
        match (self, proof) {
            (LayerCtx::Dense(ctx), LayerProof::Dense(p)) => {
                write_proof_to_transcript::<E, PCS, _, _>(ctx, p, transcript)
            }
            (LayerCtx::Convolution(ctx), LayerProof::Convolution(p)) => {
                write_proof_to_transcript::<E, PCS, _, _>(ctx, p, transcript)
            }
            (LayerCtx::MatMul(ctx), LayerProof::MatMul(p)) => {
                write_proof_to_transcript::<E, PCS, _, _>(ctx, p, transcript)
            }
            (LayerCtx::QKV(ctx), LayerProof::QKV(p)) => {
                write_proof_to_transcript::<E, PCS, _, _>(ctx, p, transcript)
            }
            (LayerCtx::Mha(ctx), LayerProof::Mha(p)) => {
                write_proof_to_transcript::<E, PCS, _, _>(ctx, p, transcript)
            }
            (LayerCtx::ConcatMatMul(ctx), LayerProof::ConcatMatMul(p)) => {
                write_proof_to_transcript::<E, PCS, _, _>(ctx, p, transcript)
            }
            (LayerCtx::Activation(ctx), LayerProof::Activation(p)) => {
                write_proof_to_transcript::<E, PCS, _, _>(ctx, p, transcript)
            }
            (LayerCtx::LayerNorm(ctx), LayerProof::LayerNorm(p)) => {
                write_proof_to_transcript::<E, PCS, _, _>(ctx, p, transcript)
            }
            (LayerCtx::RMSNorm(ctx), LayerProof::RMSNorm(p)) => {
                write_proof_to_transcript::<E, PCS, _, _>(ctx, p, transcript)
            }
            (LayerCtx::Softmax(ctx), LayerProof::Softmax(p)) => {
                write_proof_to_transcript::<E, PCS, _, _>(ctx, p, transcript)
            }
            (LayerCtx::Logits(ctx), LayerProof::Logits(p)) => {
                write_proof_to_transcript::<E, PCS, _, _>(ctx, p, transcript)
            }
            (LayerCtx::Embeddings(ctx), LayerProof::Embeddings(p)) => {
                write_proof_to_transcript::<E, PCS, _, _>(ctx, p, transcript)
            }
            (LayerCtx::Add(ctx), LayerProof::Add(p)) => {
                write_proof_to_transcript::<E, PCS, _, _>(ctx, p, transcript)
            }
            (LayerCtx::Positional(ctx), LayerProof::Positional(p)) => {
                write_proof_to_transcript::<E, PCS, _, _>(ctx, p, transcript)
            }
            (LayerCtx::Requant(ctx), LayerProof::Requant(p)) => {
                write_proof_to_transcript::<E, PCS, _, _>(ctx, p, transcript)
            }
            (LayerCtx::Pooling(ctx), LayerProof::Pooling(p)) => {
                write_proof_to_transcript::<E, PCS, _, _>(ctx, p, transcript)
            }
            (LayerCtx::AttentionMask(ctx), LayerProof::AttentionMask(p)) => {
                write_proof_to_transcript::<E, PCS, _, _>(ctx, p, transcript)
            }
            (LayerCtx::EinSum(ctx), LayerProof::EinSum(p)) => {
                write_proof_to_transcript::<E, PCS, _, _>(ctx, p, transcript)
            }
            (LayerCtx::Flatten, _) => Ok(()),
            (LayerCtx::Reshape(_), _) => Ok(()),
            _ => bail!(
                "Could not append LayerProof to transcript, Incompatible layer {} and proof {} found",
                self.describe(),
                proof.variant_name()
            ),
        }
    }
}
