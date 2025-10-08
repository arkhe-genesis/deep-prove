pub mod activation;
pub mod add;
pub mod concat_matmul;
pub mod convolution;
pub mod dense;
pub mod flatten;
pub mod hadamard;
pub mod matrix_mul;
pub mod matvec;
pub mod mul;
pub mod permute;
pub mod pooling;
pub mod provable;
pub mod requant;
pub mod reshape;
pub mod transformer;

use std::{fmt::Debug, marker::PhantomData};

use anyhow::{Context as _, Result, bail};
use ff_ext::ExtensionField;
use flatten::Flatten;
use mpcs::PolynomialCommitmentScheme;
use pooling::{PoolingCtx, PoolingProof};
use provable::{
    Evaluate, LayerOut, OpInfo, PadOp, ProvableOp, ProveInfo, ProvingData, QuantizeOp,
    QuantizeOutput,
};
use requant::RequantCtx;
use tenstore::{GenStore, StoreError};
use transcript::Transcript;
use transformer::{
    layernorm::LayerNormData, logits::ArgmaxData, mha::MhaData, softmax::SoftmaxData,
};

use crate::{
    Element, ProverContext, ScalingStrategy, Shape, Tensor,
    iop::context::{ContextAux, ShapeStep},
    layers::{
        activation::{ACTIVATION_LAYER, Activation, ActivationData, ActivationProof},
        add::{ADD_LAYER, Add, AddCtx, AddProof},
        concat_matmul::{CONCAT_MATMUL_LAYER, ConcatMatMul, ConcatMatMulCtx, ConcatMatMulProof},
        convolution::{CONVOLUTION_LAYER, Convolution},
        dense::{DENSE_LAYER, Dense},
        flatten::FLATTEN_LAYER,
        matrix_mul::MATMUL_LAYER,
        pooling::{POOLING_LAYER, Pooling},
        requant::{REQUANT_LAYER, Requant, RequantProof},
        reshape::{RESHAPE_LAYER, Reshape, ReshapeCtx},
        transformer::{
            attention::attention_mask::{
                ATTENTION_MASK_LAYER, AttentionMask, AttentionMaskCtx, AttentionMaskProof,
            },
            embeddings::{EMBEDDINGS_LAYER, Embeddings, EmbeddingsCtx, EmbeddingsProof},
            layernorm::{LAYERNORM_LAYER, LayerNorm, LayerNormCtx, LayerNormProof},
            logits::{LOGITS_LAYER, Logits, LogitsCtx, LogitsProof},
            mha::{MHA_LAYER, Mha, MhaCtx, MhaProof},
            positional::{POSITIONAL_LAYER, Positional, PositionalCtx, PositionalProof},
            qkv::{QKV, QKV_LAYER, QKVCtx, QKVProof},
            rmsnorm::{RMSNORM_LAYER, RMSNorm, RMSNormCtx, RMSNormProof},
            softmax::{SOFTMAX_LAYER, Softmax, SoftmaxCtx, SoftmaxProof},
        },
    },
    lookup::context::LookupWitnessGen,
    model::{NodeID, StepData},
    number::Number,
    padding::{PaddingMode, ShapeInfo},
    quantization::{Fieldizer, ModelMetadata, ScalingFactor},
    tensor::{ConvFFTData, DryTensor},
};
use activation::ActivationCtx;
use convolution::{ConvCtx, ConvProof};
use dense::{DenseCtx, DenseProof};
use matrix_mul::{MatMul, MatMulCtx, MatMulProof};
use serde::{Deserialize, Serialize, de::DeserializeOwned};

#[derive(Clone, Debug, Serialize, Deserialize)]
#[allow(clippy::large_enum_variant)]
pub enum Layer<T> {
    Dense(Dense<T>),
    MatMul(MatMul<T>),
    Convolution(Convolution<T>),
    Activation(Activation<T>),
    // this is the output quant info. Since we always do a requant layer after each dense,
    // then we assume the inputs requant info are default()
    Requant(Requant),
    Pooling(Pooling),
    // TODO: so far it's only flattening the input tensor, e.g. new_shape = vec![shape.iter().product()]
    Flatten(Flatten),
    QKV(QKV<T>),
    Mha(Mha<T>),
    ConcatMatMul(ConcatMatMul),
    LayerNorm(LayerNorm<T>),
    RMSNorm(RMSNorm<T>),
    Softmax(Softmax<T>),
    Add(Add<T>),
    Reshape(Reshape),
    Embeddings(Embeddings<T>),
    Positional(Positional<T>),
    AttentionMask(AttentionMask<T>),
    Logits(Logits),
}
impl<T> Layer<T> {
    pub fn short_name(&self) -> &str {
        let r = match self {
            Layer::Dense(_) => DENSE_LAYER,
            Layer::MatMul(_) => MATMUL_LAYER,
            Layer::Convolution(_) => CONVOLUTION_LAYER,
            Layer::Activation(_) => ACTIVATION_LAYER,
            Layer::Requant(_) => REQUANT_LAYER,
            Layer::Pooling(_) => POOLING_LAYER,
            Layer::Flatten(_) => FLATTEN_LAYER,
            Layer::QKV(_) => QKV_LAYER,
            Layer::Mha(_) => MHA_LAYER,
            Layer::ConcatMatMul(_) => CONCAT_MATMUL_LAYER,
            Layer::LayerNorm(_) => LAYERNORM_LAYER,
            Layer::RMSNorm(_) => RMSNORM_LAYER,
            Layer::Softmax(_) => SOFTMAX_LAYER,
            Layer::Add(_) => ADD_LAYER,
            Layer::Reshape(_) => RESHAPE_LAYER,
            Layer::Embeddings(_) => EMBEDDINGS_LAYER,
            Layer::Positional(_) => POSITIONAL_LAYER,
            Layer::Logits(_) => LOGITS_LAYER,
            Layer::AttentionMask(_) => ATTENTION_MASK_LAYER,
        };
        assert_eq!(r.len(), 4, "layer short name must be 4 chars long: {r}");
        r
    }
}

impl<T: Number> Layer<T> {
    /// Resets the internal state of the layer if any
    pub fn reset(&self) {
        if let Layer::QKV(qkv) = self {
            qkv.reset_cache();
        } else if let Layer::Positional(pos) = self {
            pos.reset_cache();
        }
    }
}

/// Describes a steps wrt the polynomial to be proven/looked at. Verifier needs to know
/// the sequence of steps and the type of each step from the setup phase so it can make sure the prover is not
/// cheating on this.
/// NOTE: The context automatically appends a requant step after each dense layer.
#[allow(clippy::large_enum_variant)]
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound(serialize = "E: Serialize", deserialize = "E: DeserializeOwned"))]
pub enum LayerCtx<E: ExtensionField> {
    Dense(DenseCtx),
    MatMul(MatMulCtx),
    Convolution(ConvCtx),
    Activation(ActivationCtx<E>),
    Requant(RequantCtx<E>),
    Pooling(PoolingCtx),
    QKV(QKVCtx),
    Mha(MhaCtx<E>),
    ConcatMatMul(ConcatMatMulCtx),
    LayerNorm(LayerNormCtx<E>),
    RMSNorm(RMSNormCtx<E>),
    Flatten,
    Add(AddCtx),
    Softmax(SoftmaxCtx<E>),
    Reshape(ReshapeCtx),
    Embeddings(EmbeddingsCtx),
    Positional(PositionalCtx),
    AttentionMask(AttentionMaskCtx<E>),
    Logits(LogitsCtx),
}

#[derive(Clone, Serialize, Deserialize)]
#[serde(bound(serialize = "E: Serialize", deserialize = "E: DeserializeOwned"))]
pub enum LayerProof<E, PCS>
where
    E: ExtensionField,
    E::BaseField: Serialize + DeserializeOwned,
    PCS: PolynomialCommitmentScheme<E>,
{
    Dense(DenseProof<E>),
    MatMul(MatMulProof<E>),
    Convolution(Box<ConvProof<E>>),
    Activation(ActivationProof<E, PCS>),
    Requant(RequantProof<E, PCS>),
    Pooling(PoolingProof<E, PCS>),
    QKV(QKVProof<E>),
    Mha(MhaProof<E, PCS>),
    ConcatMatMul(ConcatMatMulProof<E>),
    Add(AddProof<E>),
    LayerNorm(LayerNormProof<E, PCS>),
    RMSNorm(RMSNormProof<E, PCS>),
    Softmax(SoftmaxProof<E, PCS>),
    Embeddings(EmbeddingsProof<E>),
    Logits(LogitsProof<E, PCS>),
    Positional(PositionalProof<E>),
    AttentionMask(AttentionMaskProof<E>),
    Dummy, // To be used for non-provable layers
}

impl<T> Layer<T> {
    /// Convert a layer to a string only containing its kind
    pub fn as_kind_str(&self) -> &'static str {
        match self {
            Layer::Dense(_) => "dense",
            Layer::Convolution(_) => "convolution",
            Layer::Activation(_) => "activation",
            Layer::Requant(_) => "requant",
            Layer::Pooling(_) => "pooling",
            Layer::Flatten(_) => "flatten",
            Layer::QKV(_) => "qkv",
            Layer::Mha(_) => "mha-qk",
            Layer::MatMul(_) => "mat-mul",
            Layer::ConcatMatMul(_) => "concat-mat-mul",
            Layer::LayerNorm(_) => "layer-norm",
            Layer::RMSNorm(_) => "rms-norm",
            Layer::Softmax(_) => "softmax",
            Layer::Add(_) => "add",
            Layer::Reshape(_) => "reshape",
            Layer::Embeddings(_) => "embeddings",
            Layer::Positional(_) => "positional",
            Layer::Logits(_) => "logits",
            Layer::AttentionMask(_) => "attention-mask",
        }
    }
}

impl<E: ExtensionField> LayerCtx<E> {
    pub fn variant_name(&self) -> String {
        match self {
            Self::Dense(_) => "Dense".to_string(),
            Self::MatMul(_) => "Matrix Multiplication".to_string(),
            Self::QKV(_) => "QKV".to_string(),
            Self::Mha(_) => "MHA".to_string(),
            Self::ConcatMatMul(_) => "ConcatMatMul".to_string(),
            Self::LayerNorm(_) => "LayerNorm".to_string(),
            Self::RMSNorm(_) => "RMSNorm".to_string(),
            Self::Softmax(_) => "Softmax".to_string(),
            Self::Add(_) => "Add".to_string(),
            Self::Logits(_) => "Logits".to_string(),
            Self::Reshape(_) => "Reshape".to_string(),
            Self::Positional(_) => "Positional".to_string(),
            Self::Embeddings(_) => "Embeddings".to_string(),
            Self::Convolution(_) => "Convolution".to_string(),
            Self::Activation(_) => "Activation".to_string(),
            Self::Requant(_) => "Requant".to_string(),
            Self::Pooling(_) => "Pooling".to_string(),
            Self::Flatten => "Reshape".to_string(),
            Self::AttentionMask(_) => "AttentionMask".to_string(),
        }
    }

    pub fn has_proof(&self) -> bool {
        !matches!(self, Self::Flatten)
    }

    pub fn next_shape_step(&self, last_step: &ShapeStep) -> ShapeStep {
        let unpadded_output =
            self.output_shapes(&last_step.unpadded_output_shape, PaddingMode::NoPadding);
        let padded_output =
            self.output_shapes(&last_step.padded_output_shape, PaddingMode::Padding);
        ShapeStep::next_step(last_step, unpadded_output, padded_output)
    }

    pub fn shape_step(&self, unpadded_input: &[Shape], padded_input: &[Shape]) -> ShapeStep {
        let unpadded_output = self.output_shapes(unpadded_input, PaddingMode::NoPadding);
        let padded_output = self.output_shapes(padded_input, PaddingMode::Padding);
        ShapeStep::new(
            unpadded_input.to_vec(),
            padded_input.to_vec(),
            unpadded_output,
            padded_output,
        )
    }
}

#[derive(Clone)]
pub(crate) struct NodeOut<T, E: ExtensionField> {
    _t: PhantomData<T>,
    pub(crate) outputs: Vec<DryTensor<T>>,
    pub(crate) proving_data: ProvingData<E>,
}
impl<T, E: ExtensionField> NodeOut<T, E> {
    pub(crate) fn new(outputs: Vec<DryTensor<T>>, proving_data: ProvingData<E>) -> Self {
        Self {
            _t: PhantomData,
            outputs,
            proving_data,
        }
    }
    pub(crate) fn into_fields<U>(self, store: GenStore) -> anyhow::Result<NodeOut<U, E>>
    where
        T: Serialize + for<'a> Deserialize<'a>,
        U: Serialize + for<'a> Deserialize<'a>,
        T: Fieldizer<U> + Debug,
    {
        Ok(NodeOut::<U, E> {
            _t: PhantomData::<U>,
            outputs: self
                .outputs
                .into_iter()
                .map(|dry| {
                    dry.dry_cast(store.clone(), |x| x.to_field())
                        .with_context(|| format!("converting {:?}", dry.key()))
                })
                .collect::<anyhow::Result<Vec<DryTensor<U>>>>()?,
            proving_data: self.proving_data,
        })
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

    pub fn try_activation_data(&self) -> Option<&ActivationData> {
        match self.proving_data {
            ProvingData::Activation(ref data) => Some(data),
            _ => None,
        }
    }
}

impl<E: ExtensionField> NodeOut<Element, E> {
    pub(crate) fn to_dequantize(
        &self,
        md: &ModelMetadata,
        store: GenStore,
        node_id: NodeID,
    ) -> Result<NodeOut<f32, E>, StoreError> {
        Ok(NodeOut {
            _t: PhantomData,
            outputs: self
                .outputs
                .iter()
                .zip(md.layer_output_scaling_factor(node_id))
                .map(|(dry, scale_factor)| {
                    dry.dry_cast(store.clone(), |x| scale_factor.dequantize(x))
                })
                .collect::<Result<Vec<_>, StoreError>>()?,
            proving_data: self.proving_data.clone(),
        })
    }
}

impl<N: Number> OpInfo for Layer<N> {
    fn output_shapes(&self, input_shapes: &[Shape], padding_mode: PaddingMode) -> Vec<Shape> {
        match self {
            Layer::Dense(dense) => dense.output_shapes(input_shapes, padding_mode),
            Layer::Convolution(convolution) => {
                convolution.output_shapes(input_shapes, padding_mode)
            }
            Layer::MatMul(mat) => mat.output_shapes(input_shapes, padding_mode),
            Layer::Mha(mha) => mha.output_shapes(input_shapes, padding_mode),
            Layer::ConcatMatMul(concat_matmul) => {
                concat_matmul.output_shapes(input_shapes, padding_mode)
            }
            Layer::QKV(qkv) => qkv.output_shapes(input_shapes, padding_mode),
            Layer::Add(add) => add.output_shapes(input_shapes, padding_mode),
            Layer::Logits(logits) => logits.output_shapes(input_shapes, padding_mode),
            Layer::Positional(positional) => positional.output_shapes(input_shapes, padding_mode),
            Layer::LayerNorm(layernorm) => layernorm.output_shapes(input_shapes, padding_mode),
            Layer::RMSNorm(rmsnorm) => rmsnorm.output_shapes(input_shapes, padding_mode),
            Layer::Softmax(softmax) => softmax.output_shapes(input_shapes, padding_mode),
            Layer::Embeddings(embeddings) => embeddings.output_shapes(input_shapes, padding_mode),
            Layer::Reshape(reshape) => reshape.output_shapes(input_shapes, padding_mode),
            Layer::Activation(activation) => activation.output_shapes(input_shapes, padding_mode),
            Layer::Requant(requant) => requant.output_shapes(input_shapes, padding_mode),
            Layer::Pooling(pooling) => pooling.output_shapes(input_shapes, padding_mode),
            Layer::Flatten(reshape) => reshape.output_shapes(input_shapes, padding_mode),
            Layer::AttentionMask(attention_mask) => {
                attention_mask.output_shapes(input_shapes, padding_mode)
            }
        }
    }

    fn num_outputs(&self, num_inputs: usize) -> usize {
        match self {
            Layer::Dense(dense) => dense.num_outputs(num_inputs),
            Layer::Convolution(convolution) => convolution.num_outputs(num_inputs),
            Layer::MatMul(mat) => mat.num_outputs(num_inputs),
            Layer::QKV(qkv) => qkv.num_outputs(num_inputs),
            Layer::Mha(mha) => mha.num_outputs(num_inputs),
            Layer::ConcatMatMul(concat_matmul) => concat_matmul.num_outputs(num_inputs),
            Layer::LayerNorm(layernorm) => layernorm.num_outputs(num_inputs),
            Layer::RMSNorm(rmsnorm) => rmsnorm.num_outputs(num_inputs),
            Layer::Softmax(softmax) => softmax.num_outputs(num_inputs),
            Layer::Add(add) => add.num_outputs(num_inputs),
            Layer::Logits(logits) => logits.num_outputs(num_inputs),
            Layer::Reshape(reshape) => reshape.num_outputs(num_inputs),
            Layer::Positional(positional) => positional.num_outputs(num_inputs),
            Layer::Embeddings(embeddings) => embeddings.num_outputs(num_inputs),
            Layer::Activation(activation) => activation.num_outputs(num_inputs),
            Layer::Requant(requant) => requant.num_outputs(num_inputs),
            Layer::Pooling(pooling) => pooling.num_outputs(num_inputs),
            Layer::Flatten(reshape) => reshape.num_outputs(num_inputs),
            Layer::AttentionMask(attention_mask) => attention_mask.num_outputs(num_inputs),
        }
    }

    fn describe(&self) -> String {
        match self {
            Layer::Dense(dense) => dense.describe(),
            Layer::Convolution(convolution) => convolution.describe(),
            Layer::MatMul(mat) => mat.describe(),
            Layer::QKV(qkv) => qkv.describe(),
            Layer::Mha(mha) => mha.describe(),
            Layer::ConcatMatMul(concat_matmul) => concat_matmul.describe(),
            Layer::LayerNorm(layernorm) => layernorm.describe(),
            Layer::RMSNorm(rmsnorm) => rmsnorm.describe(),
            Layer::Softmax(softmax) => softmax.describe(),
            Layer::Add(add) => add.describe(),
            Layer::Logits(logits) => logits.describe(),
            Layer::Positional(positional) => positional.describe(),
            Layer::Reshape(reshape) => reshape.describe(),
            Layer::Embeddings(embeddings) => embeddings.describe(),
            Layer::Activation(activation) => activation.describe(),
            Layer::Requant(requant) => requant.describe(),
            Layer::Pooling(pooling) => pooling.describe(),
            Layer::Flatten(reshape) => reshape.describe(),
            Layer::AttentionMask(attention_mask) => attention_mask.describe(),
        }
    }

    fn is_provable(&self) -> bool {
        match self {
            Layer::Dense(dense) => dense.is_provable(),
            Layer::Convolution(convolution) => convolution.is_provable(),
            Layer::MatMul(mat) => mat.is_provable(),
            Layer::QKV(qkv) => qkv.is_provable(),
            Layer::Mha(mha) => mha.is_provable(),
            Layer::ConcatMatMul(concat_matmul) => concat_matmul.is_provable(),
            Layer::LayerNorm(layernorm) => layernorm.is_provable(),
            Layer::RMSNorm(rmsnorm) => rmsnorm.is_provable(),
            Layer::Softmax(softmax) => softmax.is_provable(),
            Layer::Positional(positional) => positional.is_provable(),
            Layer::Add(add) => add.is_provable(),
            Layer::Logits(logits) => logits.is_provable(),
            Layer::Reshape(reshape) => reshape.is_provable(),
            Layer::Embeddings(embeddings) => embeddings.is_provable(),
            Layer::Activation(activation) => activation.is_provable(),
            Layer::Requant(requant) => requant.is_provable(),
            Layer::Pooling(pooling) => pooling.is_provable(),
            Layer::Flatten(reshape) => reshape.is_provable(),
            Layer::AttentionMask(attention_mask) => attention_mask.is_provable(),
        }
    }
}

impl Evaluate<f32> for Layer<f32> {
    fn evaluate<E: ExtensionField>(
        &self,
        inputs: &[&Tensor<f32>],
        unpadded_input_shapes: &[Shape],
    ) -> Result<LayerOut<f32, E>> {
        match self {
            Layer::Dense(dense) => dense.evaluate(inputs, unpadded_input_shapes),
            Layer::Convolution(convolution) => convolution.evaluate(inputs, unpadded_input_shapes),
            Layer::MatMul(mat) => mat.evaluate(inputs, unpadded_input_shapes),
            Layer::QKV(qkv) => qkv.evaluate(inputs, unpadded_input_shapes),
            Layer::Mha(mha) => mha.evaluate(inputs, unpadded_input_shapes),
            Layer::ConcatMatMul(concat_matmul) => {
                concat_matmul.evaluate(inputs, unpadded_input_shapes)
            }
            Layer::LayerNorm(layernorm) => layernorm.evaluate(inputs, unpadded_input_shapes),
            Layer::RMSNorm(rmsnorm) => rmsnorm.evaluate(inputs, unpadded_input_shapes),
            Layer::Softmax(softmax) => softmax.evaluate(inputs, unpadded_input_shapes),
            Layer::Add(add) => add.evaluate(inputs, unpadded_input_shapes),
            Layer::Logits(logits) => logits.evaluate(inputs, unpadded_input_shapes),
            Layer::Positional(positional) => positional.evaluate(inputs, unpadded_input_shapes),
            Layer::Reshape(reshape) => reshape.evaluate(inputs, unpadded_input_shapes),
            Layer::Embeddings(embeddings) => embeddings.evaluate(inputs, unpadded_input_shapes),
            Layer::Activation(activation) => activation.evaluate(inputs, unpadded_input_shapes),
            Layer::Requant(_) => unreachable!("Requant layer found when evaluating over float"),
            Layer::Pooling(pooling) => pooling.evaluate(inputs, unpadded_input_shapes),
            Layer::Flatten(reshape) => reshape.evaluate(inputs, unpadded_input_shapes),
            Layer::AttentionMask(attention_mask) => {
                attention_mask.evaluate(inputs, unpadded_input_shapes)
            }
        }
    }
}

impl Evaluate<Element> for Layer<Element> {
    fn evaluate<E: ExtensionField>(
        &self,
        inputs: &[&Tensor<Element>],
        unpadded_input_shapes: &[Shape],
    ) -> Result<LayerOut<Element, E>> {
        let output = match self {
            Layer::Dense(dense) => dense.evaluate(inputs, unpadded_input_shapes),
            Layer::Convolution(convolution) => convolution.evaluate(inputs, unpadded_input_shapes),
            Layer::MatMul(mat) => mat.evaluate(inputs, unpadded_input_shapes),
            Layer::QKV(qkv) => qkv.evaluate(inputs, unpadded_input_shapes),
            Layer::Mha(mha) => mha.evaluate(inputs, unpadded_input_shapes),
            Layer::ConcatMatMul(concat_matmul) => {
                concat_matmul.evaluate(inputs, unpadded_input_shapes)
            }
            Layer::LayerNorm(layernorm) => layernorm.evaluate(inputs, unpadded_input_shapes),
            Layer::RMSNorm(rmsnorm) => rmsnorm.evaluate(inputs, unpadded_input_shapes),
            Layer::Softmax(softmax) => softmax.evaluate(inputs, unpadded_input_shapes),
            Layer::Add(add) => add.evaluate(inputs, unpadded_input_shapes),
            Layer::Logits(logits) => logits.evaluate(inputs, unpadded_input_shapes),
            Layer::Positional(positional) => positional.evaluate(inputs, unpadded_input_shapes),
            Layer::Embeddings(embeddings) => embeddings.evaluate(inputs, unpadded_input_shapes),
            Layer::Reshape(reshape) => reshape.evaluate(inputs, unpadded_input_shapes),
            Layer::Activation(activation) => activation.evaluate(inputs, unpadded_input_shapes),
            Layer::Requant(requant) => requant.evaluate(inputs, unpadded_input_shapes),
            Layer::Pooling(pooling) => pooling.evaluate(inputs, unpadded_input_shapes),
            Layer::Flatten(reshape) => reshape.evaluate(inputs, unpadded_input_shapes),
            Layer::AttentionMask(attention_mask) => {
                attention_mask.evaluate(inputs, unpadded_input_shapes)
            }
        };

        #[cfg(feature = "capture-layers-quant")]
        {
            if let Ok(output) = output.as_ref() {
                let layer_kind = self.as_kind_str();
                let out_dir = std::path::PathBuf::from("layers-quant").join(layer_kind);
                crate::capture::store(&out_dir, &(self, inputs), &output.outputs);
            }
        }

        output
    }
}

impl ProveInfo for Layer<Element> {
    fn step_info<E: ExtensionField>(
        &self,
        id: NodeID,
        aux: ContextAux,
    ) -> Result<(LayerCtx<E>, ContextAux)> {
        match self {
            Layer::Dense(dense) => dense.step_info(id, aux),
            Layer::QKV(qkv) => qkv.step_info(id, aux),
            Layer::Mha(mha) => mha.step_info(id, aux),
            Layer::ConcatMatMul(concat_matmul) => concat_matmul.step_info(id, aux),
            Layer::Add(add) => add.step_info(id, aux),
            Layer::LayerNorm(layernorm) => layernorm.step_info(id, aux),
            Layer::RMSNorm(rmsnorm) => rmsnorm.step_info(id, aux),
            Layer::Softmax(softmax) => softmax.step_info(id, aux),
            Layer::Logits(logits) => logits.step_info(id, aux),
            Layer::Positional(positional) => positional.step_info(id, aux),
            Layer::Embeddings(embeddings) => embeddings.step_info(id, aux),
            Layer::Reshape(reshape) => reshape.step_info(id, aux),
            Layer::MatMul(mat) => mat.step_info(id, aux),
            Layer::Convolution(conv) => conv.step_info(id, aux),
            Layer::Activation(activation) => activation.step_info(id, aux),
            Layer::Requant(requant) => requant.step_info(id, aux),
            Layer::Pooling(pooling) => pooling.step_info(id, aux),
            Layer::Flatten(reshape) => reshape.step_info(id, aux),
            Layer::AttentionMask(attention_mask) => attention_mask.step_info(id, aux),
        }
    }
}

impl PadOp for Layer<Element> {
    fn pad_node(self, si: &mut ShapeInfo) -> Result<Self>
    where
        Self: Sized,
    {
        Ok(match self {
            Layer::Dense(dense) => Layer::Dense(dense.pad_node(si)?),
            Layer::Convolution(convolution) => Layer::Convolution(convolution.pad_node(si)?),
            Layer::QKV(qkv) => Layer::QKV(qkv.pad_node(si)?),
            Layer::Mha(mha) => Layer::Mha(mha.pad_node(si)?),
            Layer::ConcatMatMul(concat_matmul) => Layer::ConcatMatMul(concat_matmul.pad_node(si)?),
            Layer::Add(add) => Layer::Add(add.pad_node(si)?),
            Layer::LayerNorm(layernorm) => Layer::LayerNorm(layernorm.pad_node(si)?),
            Layer::RMSNorm(rmsnorm) => Layer::RMSNorm(rmsnorm.pad_node(si)?),
            Layer::Softmax(softmax) => Layer::Softmax(softmax.pad_node(si)?),
            Layer::Logits(logits) => Layer::Logits(logits.pad_node(si)?),
            Layer::Positional(positional) => Layer::Positional(positional.pad_node(si)?),
            Layer::Embeddings(embeddings) => Layer::Embeddings(embeddings.pad_node(si)?),
            Layer::MatMul(mat) => Layer::MatMul(mat.pad_node(si)?),
            Layer::Activation(activation) => Layer::Activation(activation.pad_node(si)?),
            Layer::Requant(requant) => Layer::Requant(requant.pad_node(si)?),
            Layer::Pooling(pooling) => Layer::Pooling(pooling.pad_node(si)?),
            Layer::Flatten(flatten) => Layer::Flatten(flatten.pad_node(si)?),
            Layer::Reshape(reshape) => Layer::Reshape(reshape.pad_node(si)?),
            Layer::AttentionMask(attention_mask) => {
                Layer::AttentionMask(attention_mask.pad_node(si)?)
            }
        })
    }
}

impl<E, PCS> ProvableOp<E, PCS> for Layer<Element>
where
    E::BaseField: Serialize + DeserializeOwned,
    E: ExtensionField + Serialize + DeserializeOwned,
    PCS: PolynomialCommitmentScheme<E> + Send + Sync,
    PCS::CommitmentWithWitness: Serialize + DeserializeOwned + Send + Sync,
    PCS::ProverParam: Send + Sync,
{
    type Ctx = LayerCtx<E>;

    fn prove<'a, 'b, 'c, 'd, T: Transcript<E>>(
        &'a self,
        node_id: NodeID,
        ctx: &'b Self::Ctx,
        last_claims: Vec<&crate::Claim<E>>,
        step_data: &StepData<E, E>,
        prover: &mut crate::Prover<'c, 'd, E, T, PCS>,
        store: &mut GenStore,
    ) -> Result<Vec<crate::Claim<E>>> {
        match (self, ctx) {
            (Layer::Dense(dense), LayerCtx::Dense(info)) => {
                dense.prove(node_id, info, last_claims, step_data, prover, store)
            }
            (Layer::Convolution(convolution), LayerCtx::Convolution(info)) => {
                convolution.prove(node_id, info, last_claims, step_data, prover, store)
            }
            (Layer::MatMul(m), LayerCtx::MatMul(info)) => {
                m.prove(node_id, info, last_claims, step_data, prover, store)
            }
            (Layer::QKV(qkv), LayerCtx::QKV(info)) => {
                qkv.prove(node_id, info, last_claims, step_data, prover, store)
            }
            (Layer::Mha(mha), LayerCtx::Mha(info)) => {
                mha.prove(node_id, info, last_claims, step_data, prover, store)
            }
            (Layer::ConcatMatMul(concat_matmul), LayerCtx::ConcatMatMul(info)) => {
                concat_matmul.prove(node_id, info, last_claims, step_data, prover, store)
            }
            (Layer::Embeddings(embeddings), LayerCtx::Embeddings(ctx)) => {
                embeddings.prove(node_id, ctx, last_claims, step_data, prover, store)
            }
            (Layer::Positional(positional), LayerCtx::Positional(info)) => {
                positional.prove(node_id, info, last_claims, step_data, prover, store)
            }
            (Layer::Add(add), LayerCtx::Add(info)) => {
                add.prove(node_id, info, last_claims, step_data, prover, store)
            }
            (Layer::Logits(logits), LayerCtx::Logits(info)) => {
                logits.prove(node_id, info, last_claims, step_data, prover, store)
            }
            (Layer::Activation(activation), LayerCtx::Activation(info)) => {
                activation.prove(node_id, info, last_claims, step_data, prover, store)
            }
            (Layer::Requant(requant), LayerCtx::Requant(info)) => {
                requant.prove(node_id, info, last_claims, step_data, prover, store)
            }
            (Layer::Pooling(pooling), LayerCtx::Pooling(info)) => {
                pooling.prove(node_id, info, last_claims, step_data, prover, store)
            }
            (Layer::Flatten(_), LayerCtx::Flatten) => {
                unreachable!("prove cannot be called for reshape")
            }
            (Layer::Softmax(softmax), LayerCtx::Softmax(info)) => {
                softmax.prove(node_id, info, last_claims, step_data, prover, store)
            }
            (Layer::LayerNorm(layernorm), LayerCtx::LayerNorm(info)) => {
                layernorm.prove(node_id, info, last_claims, step_data, prover, store)
            }
            (Layer::RMSNorm(rmsnorm), LayerCtx::RMSNorm(info)) => {
                rmsnorm.prove(node_id, info, last_claims, step_data, prover, store)
            }
            (Layer::AttentionMask(attention_mask), LayerCtx::AttentionMask(info)) => {
                attention_mask.prove(node_id, info, last_claims, step_data, prover, store)
            }

            _ => bail!(
                "Incompatible layer {} and ctx {} found for node id {}",
                self.describe(),
                ctx.variant_name(),
                node_id
            ),
        }
    }

    fn gen_lookup_witness(
        &self,
        id: NodeID,
        ctx: &ProverContext<E, PCS>,
        step_data: &StepData<Element, E>,
        store: &mut GenStore,
    ) -> Result<LookupWitnessGen<E, PCS>> {
        match self {
            Layer::Dense(dense) => dense.gen_lookup_witness(id, ctx, step_data, store),
            Layer::Convolution(convolution) => {
                convolution.gen_lookup_witness(id, ctx, step_data, store)
            }
            Layer::MatMul(m) => m.gen_lookup_witness(id, ctx, step_data, store),
            Layer::QKV(qkv) => qkv.gen_lookup_witness(id, ctx, step_data, store),
            Layer::Mha(mha) => mha.gen_lookup_witness(id, ctx, step_data, store),
            Layer::ConcatMatMul(concat_matmul) => {
                concat_matmul.gen_lookup_witness(id, ctx, step_data, store)
            }
            Layer::Add(add) => add.gen_lookup_witness(id, ctx, step_data, store),
            Layer::Softmax(softmax) => softmax.gen_lookup_witness(id, ctx, step_data, store),
            Layer::Logits(logits) => logits.gen_lookup_witness(id, ctx, step_data, store),
            Layer::LayerNorm(layernorm) => layernorm.gen_lookup_witness(id, ctx, step_data, store),
            Layer::RMSNorm(rmsnorm) => rmsnorm.gen_lookup_witness(id, ctx, step_data, store),
            Layer::Positional(positional) => {
                positional.gen_lookup_witness(id, ctx, step_data, store)
            }
            Layer::Embeddings(embeddings) => {
                embeddings.gen_lookup_witness(id, ctx, step_data, store)
            }
            Layer::Activation(activation) => {
                activation.gen_lookup_witness(id, ctx, step_data, store)
            }
            Layer::Requant(requant) => requant.gen_lookup_witness(id, ctx, step_data, store),
            Layer::Pooling(pooling) => pooling.gen_lookup_witness(id, ctx, step_data, store),
            Layer::Reshape(r) => {
                assert!(!r.is_provable());
                Ok(Default::default())
            }
            Layer::Flatten(r) => {
                assert!(!r.is_provable());
                Ok(Default::default())
            }
            Layer::AttentionMask(attention_mask) => {
                attention_mask.gen_lookup_witness(id, ctx, step_data, store)
            }
        }
    }
}

impl QuantizeOp for Layer<f32> {
    type QuantizedOp = Layer<Element>;

    fn quantize_op<S: ScalingStrategy>(
        self,
        data: &S::AuxData,
        node_id: NodeID,
        input_scaling: &[ScalingFactor],
    ) -> anyhow::Result<QuantizeOutput<Self::QuantizedOp>> {
        Ok(match self {
            Layer::Dense(dense) => {
                let output = dense.quantize_op::<S>(data, node_id, input_scaling)?;
                QuantizeOutput::new(Layer::Dense(output.quantized_op), output.output_scalings)
                    .maybe_requants(output.requant_layer)
                    .maybe_transform(output.post_quant_rule)
            }
            Layer::Convolution(convolution) => {
                let output = convolution.quantize_op::<S>(data, node_id, input_scaling)?;
                QuantizeOutput::new(
                    Layer::Convolution(output.quantized_op),
                    output.output_scalings,
                )
                .maybe_requants(output.requant_layer)
                .maybe_transform(output.post_quant_rule)
            }
            Layer::MatMul(mat) => {
                let output = mat.quantize_op::<S>(data, node_id, input_scaling)?;
                QuantizeOutput::new(Layer::MatMul(output.quantized_op), output.output_scalings)
                    .maybe_requants(output.requant_layer)
                    .maybe_transform(output.post_quant_rule)
            }
            Layer::QKV(qkv) => {
                let output = qkv.quantize_op::<S>(data, node_id, input_scaling)?;
                QuantizeOutput::new(Layer::QKV(output.quantized_op), output.output_scalings)
                    .maybe_requants(output.requant_layer)
                    .maybe_transform(output.post_quant_rule)
            }
            Layer::Mha(mha) => {
                let output = mha.quantize_op::<S>(data, node_id, input_scaling)?;
                QuantizeOutput::new(Layer::Mha(output.quantized_op), output.output_scalings)
                    .maybe_requants(output.requant_layer)
                    .maybe_transform(output.post_quant_rule)
            }
            Layer::ConcatMatMul(concat_matmul) => {
                let output = concat_matmul.quantize_op::<S>(data, node_id, input_scaling)?;
                QuantizeOutput::new(
                    Layer::ConcatMatMul(output.quantized_op),
                    output.output_scalings,
                )
                .maybe_requants(output.requant_layer)
                .maybe_transform(output.post_quant_rule)
            }
            Layer::LayerNorm(layernorm) => {
                let output = layernorm.quantize_op::<S>(data, node_id, input_scaling)?;
                QuantizeOutput::new(
                    Layer::LayerNorm(output.quantized_op),
                    output.output_scalings,
                )
                .maybe_requants(output.requant_layer)
                .maybe_transform(output.post_quant_rule)
            }
            Layer::RMSNorm(rmsnorm) => {
                let output = rmsnorm.quantize_op::<S>(data, node_id, input_scaling)?;
                QuantizeOutput::new(Layer::RMSNorm(output.quantized_op), output.output_scalings)
                    .maybe_requants(output.requant_layer)
                    .maybe_transform(output.post_quant_rule)
            }
            Layer::Softmax(softmax) => {
                let output = softmax.quantize_op::<S>(data, node_id, input_scaling)?;
                QuantizeOutput::new(Layer::Softmax(output.quantized_op), output.output_scalings)
                    .maybe_requants(output.requant_layer)
                    .maybe_transform(output.post_quant_rule)
            }
            Layer::Add(add) => {
                let output = add.quantize_op::<S>(data, node_id, input_scaling)?;
                QuantizeOutput::new(Layer::Add(output.quantized_op), output.output_scalings)
                    .maybe_requants(output.requant_layer)
                    .maybe_transform(output.post_quant_rule)
            }
            Layer::Logits(logits) => {
                let output = logits.quantize_op::<S>(data, node_id, input_scaling)?;
                QuantizeOutput::new(Layer::Logits(output.quantized_op), output.output_scalings)
                    .maybe_requants(output.requant_layer)
                    .maybe_transform(output.post_quant_rule)
            }
            Layer::Positional(positional) => {
                let output = positional.quantize_op::<S>(data, node_id, input_scaling)?;
                QuantizeOutput::new(
                    Layer::Positional(output.quantized_op),
                    output.output_scalings,
                )
                .maybe_requants(output.requant_layer)
                .maybe_transform(output.post_quant_rule)
            }
            Layer::Embeddings(embeddings) => {
                let output = embeddings.quantize_op::<S>(data, node_id, input_scaling)?;
                QuantizeOutput::new(
                    Layer::Embeddings(output.quantized_op),
                    output.output_scalings,
                )
                .maybe_requants(output.requant_layer)
                .maybe_transform(output.post_quant_rule)
            }
            Layer::Activation(activation) => {
                let output = activation.quantize_op::<S>(data, node_id, input_scaling)?;
                QuantizeOutput::new(
                    Layer::Activation(output.quantized_op),
                    output.output_scalings,
                )
                .maybe_requants(output.requant_layer)
            }
            Layer::Requant(requant) => {
                QuantizeOutput::new(Layer::Requant(requant), input_scaling.to_vec())
            }
            Layer::Pooling(pooling) => {
                QuantizeOutput::new(Layer::Pooling(pooling), input_scaling.to_vec())
            }
            Layer::Flatten(flatten) => {
                QuantizeOutput::new(Layer::Flatten(flatten), input_scaling.to_vec())
            }
            Layer::Reshape(reshape) => {
                QuantizeOutput::new(Layer::Reshape(reshape), input_scaling.to_vec())
            }
            Layer::AttentionMask(attention_mask) => {
                let output = attention_mask.quantize_op::<S>(data, node_id, input_scaling)?;
                QuantizeOutput::new(
                    Layer::AttentionMask(output.quantized_op),
                    output.output_scalings,
                )
                .maybe_requants(output.requant_layer)
                .maybe_transform(output.post_quant_rule)
            }
        })
    }
}

impl<E, PCS> LayerProof<E, PCS>
where
    E: ExtensionField,
    E::BaseField: Serialize + DeserializeOwned,
    PCS: PolynomialCommitmentScheme<E>,
{
    pub fn variant_name(&self) -> String {
        match self {
            Self::Dense(_) => "Dense".to_string(),
            Self::MatMul(_) => "Matmul".to_string(),
            Self::QKV(_) => "QKV".to_string(),
            Self::Mha(_) => "MHA".to_string(),
            Self::ConcatMatMul(..) => "ConcatMatMul".to_string(),
            Self::LayerNorm(_) => "LayerNorm".to_string(),
            Self::RMSNorm(_) => "RMSNorm".to_string(),
            Self::Softmax(_) => "Softmax".to_string(),
            Self::Logits(_) => "Logits".to_string(),
            Self::Positional(_) => "Positional".to_string(),
            Self::Add(_) => "Add".to_string(),
            Self::Embeddings(_) => "Embeddings".to_string(),
            Self::Convolution(_) => "Convolution".to_string(),
            Self::Activation(_) => "Activation".to_string(),
            Self::Requant(_) => "Requant".to_string(),
            Self::Pooling(_) => "Pooling".to_string(),
            Self::Dummy => "Dummy".to_string(),
            Self::AttentionMask(_) => "AttentionMask".to_string(),
        }
    }

    pub fn get_lookup_data(&self) -> Option<(Vec<E>, Vec<E>)> {
        match self {
            LayerProof::Dense(..) => None,
            LayerProof::MatMul(..) => None,
            LayerProof::QKV(..) => None,
            LayerProof::Mha(proof) => Some(proof.get_lookup_data()),
            LayerProof::ConcatMatMul(..) => None,
            LayerProof::Add(_) => None,
            LayerProof::LayerNorm(proof) => Some(proof.get_lookup_data()),
            LayerProof::RMSNorm(proof) => Some(proof.get_lookup_data()),
            LayerProof::Softmax(proof) => Some(proof.get_lookup_data()),
            LayerProof::Logits(proof) => Some(proof.get_lookup_data()),
            LayerProof::Positional(_) => None,
            LayerProof::Embeddings(..) => None,
            LayerProof::Convolution(..) => None,
            LayerProof::Dummy => None,
            LayerProof::Activation(ActivationProof { lookup, .. }) => {
                Some(lookup.fractional_outputs())
            }
            LayerProof::Pooling(PoolingProof { lookup, .. }) => Some(lookup.fractional_outputs()),
            LayerProof::Requant(RequantProof { logup_proof, .. }) => {
                Some(logup_proof.fractional_outputs())
            }
            LayerProof::AttentionMask(_) => None,
        }
    }
}
impl<T: Number> std::fmt::Display for Layer<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.describe())
    }
}
