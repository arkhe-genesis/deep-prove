use crate::{
    Claim, Element, ProverContext, ScalingFactor, ScalingStrategy, Shape, Tensor,
    commit::{compute_betas_eval, identity_eval},
    graph::NodeId,
    iop::{
        context::{ContextAux, ShapeStep},
        prover::Prover,
        verifier::Verifier,
    },
    layers::{
        LayerCtx, LayerProof, Requant,
        provable::{
            Evaluate, LayerOut, OpInfo, PadOp, ProvableOp, ProveInfo, ProvingData, QuantizeOp,
            QuantizeOutput, VerifiableCtx,
        },
    },
    lookup::{
        context::{
            COLUMN_SEPARATOR, InverseSQRTTableData, LayerLookupContext, LookupWitnessGen, TableType,
        },
        logup_gkr::{
            prover::batch_multiple_sizes_prove, structs::LogUpBatchProof,
            verifier::verify_logup_proof_multiple_sizes,
        },
    },
    model::Step,
    number::Number,
    padding::PaddingMode,
    parser::{
        gguf, json,
        llm::{LLMConfig, transformer::NormType},
        safe,
    },
    quantization::{self, Fieldizer},
    tensor::{CommitmentId, KeyedTensor, TensorTypeParam, WrappedTensor},
    to_base,
};
use anyhow::{Context, Result, anyhow, ensure};
use ark_std::Zero;
use either::Either;
use ff_ext::ExtensionField;
use mpcs::PolynomialCommitmentScheme;
use multilinear_extensions::{
    Expression,
    mle::{IntoMLE, MultilinearExtension},
    util::{ceil_log2, transpose},
    utils::eval_by_expr_with_instance,
    virtual_poly::VPAuxInfo,
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
use tracing::trace;
use transcript::Transcript;
use witness::{InstancePaddingStrategy, RowMajorMatrix};

/// The short name used to identify the LayerNorm layer.
pub(crate) const LAYERNORM_LAYER: &str = "LNRM";

/// The base 2 logarithm of the scale factor used in the inverse square root lookup tables
pub(crate) const LOG_LAYERNORM_SCALE_FACTOR: usize = 20;
/// The scale factor for our fixed point arithmetic
pub(crate) const LAYERNORM_SCALE_FACTOR: usize = 1 << LOG_LAYERNORM_SCALE_FACTOR;
/// The scale factor of the outputs of the inverse square root lookup tables lookup
pub(crate) const LAYERNORM_OUTPUT_SCALE_FACTOR: usize = 1 << 20;

/// Struct storing all information needed to perform LayerNorm.
///
/// The `gamma` and `beta` fields are normally learned parameters that are
/// applied elementwise. The `eps` field is used for normalisation when
/// calculating the inverse square root.
///
/// # References
///
/// - PyTorch's [LayerNorm](https://docs.pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html)
/// - [Layer Normalization](https://arxiv.org/abs/1607.06450) paper
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerNorm<N> {
    /// Each element of the normalisation dimension is multiplied elementwise by this
    pub gamma: KeyedTensor<N>,
    /// Added elementwise to each element in the normalisation dimension
    pub beta: KeyedTensor<N>,
    /// Normalisation factor
    pub eps: f32,
    /// Contains information needed to perform quantised evaluation
    pub quant_info: Option<QuantisedLayerNormData>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Copy)]
/// This struct is used to store information used when evaluating the quantised version of [`LayerNorm`] on
/// [`Element`]s.
pub struct QuantisedLayerNormData {
    /// The [`ScalingFactor`] of the inputs
    input_scale_factor: ScalingFactor,
    /// This is the multiplier we have to rescale the inputs with
    multiplier: Element,
    /// This stores the [`InverseSQRTTableData`]
    lut: InverseSQRTTableData,
    /// The size of the dimension we average over
    dim_size: usize,
    /// The base 2 log of the value we have to multiply the most significant range check chunk by
    top_chunk_scalar_log: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
/// Data obtained during quantised evaluation of [`LayerNorm`] that is used during proving
pub struct LayerNormData {
    /// The output of the inverse square root lookup
    lookup_output: Vec<Element>,

    /// The full value of the input.
    ///
    /// Both the part of the input that need to be range checked and the input
    /// of the inverse square root lookup can be derived from this value.
    full_value: Vec<Element>,
}

impl<N: Number> LayerNorm<N> {
    pub fn new(gamma: KeyedTensor<N>, beta: KeyedTensor<N>, eps: f32) -> Self {
        assert_eq!(
            gamma.shape(),
            beta.shape(),
            "Gamma and beta shape must match. gamma {:?} beta {:?}",
            gamma.shape(),
            beta.shape(),
        );
        assert_eq!(
            gamma.rank(),
            1,
            "Gamma and beta must be 1D. gamma {:?} beta {:?}",
            gamma.shape(),
            beta.shape(),
        );

        Self {
            gamma,
            beta,
            eps,
            quant_info: None,
        }
    }

    /// Returns the size of the dimension normalisation occurs over.
    pub fn normalisation_dim_size(&self) -> usize {
        self.gamma.shape()[0]
    }

    /// Returns the [`QuantisedLayerNormData`] if there is any.
    pub fn quant_info(&self) -> Option<&QuantisedLayerNormData> {
        self.quant_info.as_ref()
    }

    /// Quantise the layer. To do this we want to have a common scale factor so that lookup tables can be reused, so we use the
    /// constant [`LAYERNORM_SCALE_FACTOR`] as the input column scale factor. We need to work out how big the table needs to be to cover
    /// all of our possible inputs.
    ///
    /// This method returns the quantised [`LayerNorm`] as well as the `intermediate_bit_size` for the following requant layer.
    pub fn quantise(
        self,
        input_scaling: ScalingFactor,
        model_scaling: ScalingFactor,
    ) -> Result<(LayerNorm<Element>, usize, ScalingFactor)> {
        // The input to the lookup table is `N*sum2 - sum1^{2}` where `sum2 = \sum xi^{2}` and `sum1 = \sum xi`.
        // We use this value because the standard deviation can be calculated by `(N*sum2 - sum1^{2}).sqrt() / N`
        // Since each `xi` is a value between `*quantisation::MIN` and `*quantisation::MAX` it has bit-size `*quantization::BIT_LEN - 1`.
        // This means `sum1` has bit-size `ceil_log2(N) + *quantization::BIT_LEN - 1` and `sum2` has bit-size `2(*quantization::BIT_LEN - 1)`
        // Then `sum1^{2}` has bit-size `2(ceil_log2(N) + *quantization::BIT_LEN - 1)` and `Nsum2` has bit_size `ceil_log2(N) + 2(*quantization::BIT_LEN - 1)`.
        // Finally we have to multiply all of this by `multiplier = LAYERNORM_SCALE_FACTOR * input_scaling.scale() * input_scaling.scale()` so we have `ceil_log2(multiplier)`
        // additional bits on top of this.

        // Get the input scale
        let input_scale = input_scaling.scale();
        // Get the dim size (N)
        let dim_size = self.normalisation_dim_size();
        // We work out what we have to multiply by so that everything is scaled to `LAYERNORM_SCALE_FACTOR` in quantised world
        let multiplier = (LAYERNORM_SCALE_FACTOR as f32 * input_scale * input_scale)
            .round_ties_even() as Element;
        // Work out the number of variables the table requires, this is likely to be far too large to actually materialise as a table
        let full_table_bit_size = 2 * (ceil_log2(dim_size) + *quantization::BIT_LEN - 1)
            + ceil_log2(multiplier as usize)
            + 1;
        // To get around this we use the fact that we should only have roughly `2*(*quantization::BIT_LEN -1)` bits of precision i.e. only the most significant `2*(*quantization::BIT_LEN -1)`
        // can actually be "trusted" the rest are essentially junk because they don't come from the actual inputs and are just guesses at the part that we have already "rounded away" in quantisation.
        // So the actual part we perform inverse square root on is size `2*(*quantization::BIT_LEN -1)` and then we just need the discarded part to be range checked (which we do via a separate lookup).
        let range_checked_bits = full_table_bit_size - 2 * (*quantization::BIT_LEN - 1);

        // The final chunk might be values with fewer than *quantization::BIT_LEN bits so we work out what we need to scale the value up by in order to use our standard range check table.
        let remainder_bits = range_checked_bits % *quantization::BIT_LEN;
        let top_chunk_scalar_log = if !remainder_bits.is_zero() {
            *quantization::BIT_LEN - remainder_bits
        } else {
            0
        };
        // Calculate the lookup table
        let table_max: Element = 1 << (2 * (*quantization::BIT_LEN - 1));
        let table_min = -table_max;
        // Because we don't use the same formula for the standard deviation as LayerNorm does in float we have to rescale `self.eps` in this case to be `N^2 * self.eps`
        let rescaled_eps = (dim_size * dim_size) as f32 * self.eps;
        let table_data = InverseSQRTTableData::new(rescaled_eps.to_bits(), range_checked_bits);

        let max_lut_value = (table_min..table_max)
            .map(|v| table_data.table_output(v).abs())
            .max()
            .unwrap();
        // The value is positive so we just convert to usize
        let max_lut_value_bits = ceil_log2(max_lut_value as usize);

        // Make the QuantisedLayerNormData
        let quant_info = QuantisedLayerNormData {
            input_scale_factor: input_scaling,
            multiplier,
            lut: table_data,
            dim_size,
            top_chunk_scalar_log,
        };

        let quant_gamma = self.gamma.try_map_tensor(|gamma| {
            let quant_gamma_data = gamma
                .get_data()
                .iter()
                .map(|v| {
                    let vf32 = v.to_f32()?;
                    Ok(model_scaling.quantize(&vf32))
                })
                .collect::<Result<Vec<Element>, anyhow::Error>>()?;

            Ok(Tensor::<Element>::new(
                gamma.shape().clone(),
                quant_gamma_data,
            ))
        })?;
        // Work out how to quantise the bias, it needs to have the same scale factor as the end product.
        // This will be `input_scaling.scale() * model_scaling.scale() * 1.0f32 / LAYERNORM_OUTPUT_SCALE_FACTOR as f32`
        let bias_scale = input_scale * model_scaling.scale() / LAYERNORM_OUTPUT_SCALE_FACTOR as f32;

        let bias_max = self.beta.max_abs_output().to_f32()?;

        let quant_bias_min = (-bias_max / bias_scale).round_ties_even() as Element;
        let quant_bias_max = (bias_max / bias_scale).round_ties_even() as Element;

        let bias_scaling = ScalingFactor::from_parts(
            bias_max,
            -bias_max,
            bias_scale,
            (quant_bias_min, quant_bias_max),
        );

        let quant_beta = self.beta.try_map_tensor(|beta| {
            let quant_bias_data = beta
                .get_data()
                .iter()
                .map(|v| {
                    let vf32 = v.to_f32()?;
                    Ok(bias_scaling.quantize(&vf32))
                })
                .collect::<Result<Vec<Element>, anyhow::Error>>()?;

            Ok(Tensor::<Element>::new(
                beta.shape().clone(),
                quant_bias_data,
            ))
        })?;

        ensure!(
            quant_gamma.shape() == quant_beta.shape(),
            "Quantised gamma and beta must have the same shape. gamma {:?} beta {:?}",
            quant_gamma.shape(),
            quant_beta.shape(),
        );
        ensure!(
            quant_gamma.rank() == 1,
            "Quantised gamma and beta must be 1D. gamma {:?} beta {:?}",
            quant_gamma.shape(),
            quant_beta.shape(),
        );

        // To calculate the intermediate bit size we have that the output is `self.gamma * (N * input - SUM input) * lookup_output + self.beta`
        // So lets work out the left hand bit size
        let lhs_bit_size =
            2 * (*quantization::BIT_LEN - 1) + ceil_log2(dim_size) + 1 + max_lut_value_bits;

        let intermediate_bit_size = if quant_bias_max > 0 {
            lhs_bit_size.max(ceil_log2(quant_bias_max as usize)) + 1
        } else {
            lhs_bit_size + 1
        };

        Ok((
            LayerNorm::<Element> {
                gamma: quant_gamma,
                beta: quant_beta,
                eps: rescaled_eps,
                quant_info: Some(quant_info),
            },
            intermediate_bit_size,
            bias_scaling,
        ))
    }
}

impl LayerNorm<f32> {
    pub fn from_json(l: &json::FileTensorLoader, _c: &LLMConfig) -> anyhow::Result<Self> {
        trace!("from_json: current path: {:?}", l.prefix);
        let gamma = l.get_tensor("norm.weight")?;
        let beta = l.get_tensor("norm.bias")?;
        let eps = l.metadata_to_f32("norm_epsilon")?;
        Ok(Self::new(gamma, beta, eps))
    }
    // Replaces from_var_builder and from_tensor_loader
    // The 'loader' passed here is expected to be pre-scoped by the caller
    // (e.g., loader.pp("attn_") or loader.pp("ffn_"))
    pub fn from_gguf(loader: &gguf::FileTensorLoader, c: &LLMConfig) -> anyhow::Result<Self> {
        let gamma = loader.get_tensor("norm.weight")?;
        let beta = loader.get_tensor("norm.bias")?;
        ensure!(
            gamma.shape().as_ref() == &[c.embedding_size],
            "norm_gamma must have shape [{}] vs given {:?}",
            c.embedding_size,
            gamma.shape()
        );
        ensure!(
            beta.shape().as_ref() == &[c.embedding_size],
            "norm_beta must have shape [{}] vs given {:?}",
            c.embedding_size,
            beta.shape()
        );
        let eps = loader
            .metadata::<f32>(&loader.norm_epsilon_key(&c.model_name, NormType::LayerNorm))
            .context("norm_epsilon not found")?;
        Ok(Self::new(gamma, beta, eps))
    }

    pub fn from_safetensors(
        loader: &safe::FileTensorLoader,
        config: &safe::ConfigJSON,
        c: &LLMConfig,
    ) -> anyhow::Result<Self> {
        let gamma = loader.get_tensor("norm.weight")?;
        let beta = loader.get_tensor("norm.bias")?;
        let eps = config
            .get::<f32, _>("norm_epsilon")
            .context("norm_epsilon not found")?;
        ensure!(
            gamma.shape().as_ref() == &[c.embedding_size],
            "norm_gamma must have shape [{}] vs given {:?}",
            c.embedding_size,
            gamma.shape()
        );
        ensure!(
            beta.shape().as_ref() == &[c.embedding_size],
            "norm_beta must have shape [{}] vs given {:?}",
            c.embedding_size,
            beta.shape()
        );
        Ok(Self::new(gamma, beta, eps))
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "E: ExtensionField + DeserializeOwned")]
pub struct LayerNormCtx<E: ExtensionField> {
    node_id: NodeId,
    /// The result of calling [`f32::to_bits`] on the epsilon used for normalisation purposes
    eps: u32,
    /// The number of bits that get range checked (so we can know how many instances there are in the range lookup)
    range_check_bits: usize,
    /// The size of the dimension we normalise over (unpadded)
    dim_size: usize,
    /// The multiplier used to scale up inputs to the lookup table.
    multiplier: Element,
    /// The base 2 logarithm of the multiplier for the most significant chunk we range check
    top_chunk_scalar_log: usize,
    /// The lookup info for the layer
    lookup_ctx: LayerLookupContext,
    /// The sumcheck expression for verifying the lookup input and layer output are correctly calculated
    first_sumcheck_expression: Vec<Expression<E>>,
    /// The sumcheck expression that verifies the mean has been calculated correctly
    mean_sumcheck_expression: Vec<Expression<E>>,
    gamma_key: CommitmentId,
    beta_key: CommitmentId,
}

impl<E: ExtensionField> OpInfo for LayerNormCtx<E> {
    // https://docs.rs/burn/0.17.0/burn/nn/struct.LayerNorm.html#method.forward
    fn output_shapes(&self, input_shapes: &[Shape], _padding_mode: PaddingMode) -> Vec<Shape> {
        input_shapes.to_vec()
    }

    fn num_outputs(&self, num_inputs: usize) -> usize {
        num_inputs
    }

    fn describe(&self) -> String {
        format!(
            "LayerNormCtx(dimension size: {}, epsilon: {})",
            self.dim_size, self.eps
        )
    }

    fn is_provable(&self) -> bool {
        true
    }
}

impl<N: TensorTypeParam> OpInfo for LayerNorm<N> {
    // https://docs.rs/burn/0.17.0/burn/nn/struct.LayerNorm.html#method.forward
    fn output_shapes(&self, input_shapes: &[Shape], _padding_mode: PaddingMode) -> Vec<Shape> {
        input_shapes.to_vec()
    }

    fn num_outputs(&self, num_inputs: usize) -> usize {
        num_inputs
    }

    fn describe(&self) -> String {
        format!("LayerNorm(dimension size: {:?})", self.gamma.shape(),)
    }

    fn is_provable(&self) -> bool {
        true
    }
}

impl Evaluate<f32> for LayerNorm<f32> {
    fn evaluate<E: ExtensionField>(
        &self,
        inputs: &[&WrappedTensor<f32>],
    ) -> anyhow::Result<LayerOut<f32, E>> {
        assert_eq!(
            inputs.len(),
            1,
            "Exactly one input must be provided to layer norm. got {}",
            inputs.len(),
        );
        let input = inputs[0].clone();

        ensure!(
            input.rank() == 2,
            "layernorm input must have shape [seq_len, embedding_size]: found {:?}",
            input.shape(),
        );
        let embedding_size = input.shape().dims[1];
        // NOTE: simply use the burn tensor API for now as we want to move towards using more burn features
        // instead of re-implementing everything ourselves.
        // copy implementation https://docs.rs/burn-core/0.17.0/src/burn_core/nn/norm/layer.rs.html#67
        let gamma = WrappedTensor::try_from(&self.gamma)?;
        let beta = WrappedTensor::try_from(&self.beta)?;
        let output =
            WrappedTensor::layer_norm(input, embedding_size, self.eps as f64, gamma, beta)?;
        Ok(LayerOut::from_tensor(output))
    }
}

impl Evaluate<Element> for LayerNorm<Element> {
    fn evaluate<E: ExtensionField>(
        &self,
        inputs: &[&WrappedTensor<Element>],
    ) -> Result<LayerOut<Element, E>> {
        // First we check to see if there is any quant_info, if not error
        ensure!(
            self.quant_info.is_some(),
            "Cannot perform quantised LayerNorm evaluation if self.quant_info is None",
        );
        // Ensure we have a single input
        ensure!(
            inputs.len() == 1,
            "LayerNorm should have a single input, had: {}",
            inputs.len(),
        );
        let input = inputs[0].clone();

        assert_eq!(self.gamma.rank(), 1, "Gamma must be 1D");
        assert_eq!(self.beta.rank(), 1, "Beta must be 1D");

        let QuantisedLayerNormData {
            multiplier,
            lut,
            dim_size,
            ..
        } = self
            .quant_info
            .as_ref()
            .ok_or(anyhow!("Missing QuantisedLayerNormData"))?;

        // So we need to take the input data and calculate `N * multiplier * SUM (xi * xi) - multiplier * (SUM xi) * (SUM xi)`
        let final_dim = *input.shape().dims.last().ok_or(anyhow!(
            "Cannot evaluate LayerNorm, input didn't have a shape",
        ))?;

        assert_eq!(
            final_dim,
            self.gamma.dim(0),
            "Input's final dimension must be equal to gamma's size. input: {:?} gamma: {:?}",
            input.shape(),
            self.gamma.shape(),
        );
        assert_eq!(
            final_dim,
            self.beta.dim(0),
            "Input's final dimension must be equal to beta's size. input: {:?} beta: {:?}",
            input.shape(),
            self.beta.shape(),
        );

        let shape = input.shape();

        let sum = input.clone().sum_dim(1);
        let square_sum = sum.clone().mul(sum.clone())?;
        let sum_square = (input.clone().mul(input.clone())?).sum_dim(1);

        let full_value = sum_square
            .mul_scalar(*dim_size as Element * multiplier)
            .sub(square_sum.mul_scalar(*multiplier))?;

        let value = full_value
            .clone()
            // clear low bits
            .bitwise_right_shift_scalar(lut.range_check_bits() as Element)
            .bitwise_left_shift_scalar(lut.range_check_bits() as Element)
            .float();

        let inv_sqrt = value
            // compute `v/LAYERNORM_SCALE_FACTOR + eps`
            .div_scalar(LAYERNORM_SCALE_FACTOR as f32)
            .add_scalar(lut.float_epsilon())
            // compute `1/sqrt(v)`
            .sqrt()
            .recip()
            .mul_scalar(LAYERNORM_OUTPUT_SCALE_FACTOR as f32)
            .round()
            .int();

        let gamma = WrappedTensor::try_from(&self.gamma)?
            .unsqueeze_dim_2()
            .expand([shape.dims[0] as i32, -1])?;

        let beta = WrappedTensor::try_from(&self.beta)?
            .unsqueeze_dim_2()
            .expand([shape.dims[0] as i32, -1])?;

        let denominator = inv_sqrt
            .clone()
            .unsqueeze_dim_2()
            .expand([-1, final_dim as i32])?;

        let output = input
            .mul_scalar(*dim_size as Element)
            .sub(sum)?
            .mul(gamma)?
            .mul(denominator)?
            .add(beta)?;

        let lookup_output = inv_sqrt
            .to_data()
            .into_vec()
            .expect("Failed to compute LayerNorm");
        let full_value = full_value
            .to_data()
            .into_vec()
            .expect("Failed to compute LayerNorm");

        let layernorm_data = LayerNormData {
            lookup_output,
            full_value,
        };

        Ok(LayerOut::from_tensor(output).with_proving_data(ProvingData::LayerNorm(layernorm_data)))
    }
}

fn is_close_to_integer(x: f32, tol: f32) -> bool {
    (x - x.round_ties_even()).abs() < tol
}

/// Given a `bit` position, return the bitmask to all bits equal to or lower than it.
fn bit_to_mask(bit: usize) -> Element {
    (1 << bit) - 1
}

impl Requant {
    /// We implement a special way of formulating a [`Requant`] layer here where `s1*s2/s3 = 2^-s` where
    /// s is a positive integer (so the requant layer only needs to perform a shift rather than a shift and a rescaling)
    pub(crate) fn new_shift(
        input_scale: f32,
        output_scale: f32,
        intermediate_bit_size: usize,
    ) -> Result<Requant> {
        // First we check that we can actually use this method
        let input_log = input_scale.log2();
        let output_log = output_scale.log2();
        let m = input_scale / output_scale;
        let m_log = m.log2();
        let int_part = m_log.trunc().abs();
        // We allow for a possible floating point error that results in an imperfect division
        ensure!(
            is_close_to_integer((input_log - output_log).abs(), 1e-5),
            "Cannot perform shift only Requant as the fractional part of the exponent was too large, input {},output {} -> diff {}",
            input_log,
            output_log,
            is_close_to_integer((input_log - output_log).abs(), 1e-5),
        );

        // We want the part that gets shifted away to be a multiple of the quantisation bit length (that way we can use the same range table for each chunk)
        let next_multiple = (int_part as usize).next_multiple_of(*quantization::BIT_LEN);
        let fp_scale = next_multiple - int_part as usize;
        let fixed_point_multiplier: Element = 1 << fp_scale;

        // Assertion to check that we can perform requantisation, we need intermediate_bit_size + fp_scale <= 63
        ensure!(
            intermediate_bit_size + fp_scale <= 63,
            "Cannot construct shift only Requant, intermediate bit size: {intermediate_bit_size}, fp scale: {fp_scale}, int part: {int_part}",
        );
        Ok(Requant {
            right_shift: int_part as usize,
            fixed_point_multiplier,
            fp_scale,
            multiplier: m,
            intermediate_bit_size,
        })
    }
}

impl QuantizeOp for LayerNorm<f32> {
    type QuantizedOp = LayerNorm<Element>;

    fn quantize_op<S: ScalingStrategy>(
        self,
        data: &S::AuxData,
        node_id: NodeId,
        input_scaling: &[ScalingFactor],
        _unpadded_input_shapes: &[Shape],
    ) -> Result<QuantizeOutput<Self::QuantizedOp>> {
        // First check we have one input_scaling
        ensure!(
            input_scaling.len() == 1,
            "Could not quantise LayerNorm, too many input scaling factors {}, expected 1",
            input_scaling.len()
        );
        let input_scaling_factor = input_scaling[0];
        // Now we construct the `model_scaling` from `self.gamma`
        let model_scaling = ScalingFactor::from_tensor(&self.gamma, None);

        let (quantised_layernorm, intermediate_bit_size, intermediate_scaling) =
            self.quantise(input_scaling_factor, model_scaling)?;
        // We will use the `intermediate_scaling` to work out a suitable `output_scaling`. Ideally `output_scaling` is 2^-s where the fractional part of `s` is the same as the fractional part of `intermediate_scaling`
        // and the integer part is such that 2^-s is as close as possible to the observed scaling factor.
        let observed_scalings = S::scaling_factors_for_node(data, node_id, 1);
        ensure!(
            observed_scalings.len() == 1,
            "Observed scaling factors for LayerNorm layer different from 1, observed {}",
            observed_scalings.len()
        );
        let observed_scaling = observed_scalings[0];
        let observed_scale = observed_scaling.scale();
        let obs_log = observed_scale.log2();
        let obs_fract = obs_log.fract().abs();
        let obs_int = obs_log.trunc().abs();
        let intermediate_scale = intermediate_scaling.scale();
        let inter_log = intermediate_scale.log2();
        let inter_fract = inter_log.fract().abs();
        // The value diff = (obs_fract - inter_fract) is between -1 and 1 and where it falls in this range defines what we should set `output_scaling` to be
        // if its positive that means `obs_fract` was larger and then we have two cases:
        //  1) diff < 0.5 => output_scaling should have scale 2^-{obs_int + inter_fract}
        //  2) diff >= 0.5 => output_scaling should have scale 2^-{obs_int + 1 + inter_fract}
        // if it is negative that means `inter_fract was larger` and then we have two cases:
        //  1) 0>= diff > -0.5 => output_scaling should have scale 2^-{obs_int + inter_fract}
        //  2) -0.5 >= diff > -1 => output_scaling should have scale 2^-{obs_int - 1 + inter_fract}
        let output_scale = match obs_fract - inter_fract {
            0.5f32..1.0f32 => 2.0f32.powf(-(obs_int + 1.0f32 + inter_fract)),
            -0.5f32..0.5f32 => 2.0f32.powf(-(obs_int + inter_fract)),
            -1.0f32..0.5f32 => 2.0f32.powf(-(obs_int - 1.0f32 + inter_fract)),
            _ => unreachable!(),
        };

        let output_scaling = ScalingFactor::from_parts(
            observed_scaling.max(),
            observed_scaling.min(),
            output_scale,
            observed_scaling.domain(),
        );
        // Make the requant layer
        let requant = Requant::new_shift(
            intermediate_scaling.scale(),
            output_scaling.scale(),
            intermediate_bit_size,
        )?;

        Ok(QuantizeOutput::new(quantised_layernorm, vec![output_scaling]).with_requant(requant))
    }
}

impl ProveInfo for LayerNorm<Element> {
    fn step_info<E: ExtensionField>(
        &self,
        id: NodeId,
        mut aux: ContextAux,
    ) -> Result<(LayerCtx<E>, ContextAux)> {
        if let Some(quant_info) = self.quant_info() {
            let QuantisedLayerNormData {
                multiplier,
                dim_size,
                top_chunk_scalar_log,
                lut,
                ..
            } = quant_info;

            // Add the tables that LayerNorm requires
            aux.tables.insert(TableType::Range);
            aux.tables.insert(TableType::InverseSQRT(*lut));

            let num_range_checks = (lut.range_check_bits() - 1) / *quantization::BIT_LEN + 1;
            let tables = vec![TableType::Range, TableType::InverseSQRT(*lut)];
            let instances_per_table = vec![num_range_checks, 1];

            let lookup_ctx = LayerLookupContext::new(tables, instances_per_table);

            // Add the Gamma and Beta commitments
            let gamma_evals = self.gamma.pad_next_power_of_two().into_data();
            let beta_evals = self.beta.pad_next_power_of_two().into_data();

            aux.model_polys = {
                let mut model_polys = HashMap::new();
                model_polys.insert(self.gamma.commitment_id(), gamma_evals);
                model_polys.insert(self.beta.commitment_id(), beta_evals);
                Some(model_polys)
            };

            aux.max_poly_len = aux
                .last_output_shape
                .iter()
                .fold(aux.max_poly_len, |acc, shapes| {
                    acc.max(shapes.next_power_of_two().product())
                });

            let (first_expr, second_expr) = build_sumcheck_expressions::<E>(*multiplier, *dim_size);
            // The output shape is the same as the input shape so we don't need to update it
            // return the LayerCtx and the updated ContextAux
            Ok((
                LayerCtx::LayerNorm(LayerNormCtx {
                    node_id: id,
                    eps: self.eps.to_bits(),
                    range_check_bits: lut.range_check_bits(),
                    dim_size: *dim_size,
                    multiplier: *multiplier,
                    top_chunk_scalar_log: *top_chunk_scalar_log,
                    lookup_ctx,
                    first_sumcheck_expression: vec![first_expr],
                    mean_sumcheck_expression: vec![second_expr],
                    gamma_key: self.gamma.commitment_id(),
                    beta_key: self.beta.commitment_id(),
                }),
                aux,
            ))
        } else {
            Err(anyhow!(
                "LayerNorm operation has not been quantised so no proving info available"
            ))
        }
    }
}

/// Builds the sumcheck expressions used in `LayerNorm` proving/verifying.
/// The first [`Expression`] returned links the inverse square root lookup input to the layer input and the `last_claim.eval` to the inverse square root output.
/// The second [`Expression`] returned shows that the `mean_poly` is equal to the row wise sum of the `input_poly`. That is if the input poly had one row that was
/// `[0, 1, 2, 3]` then the second [`Expression`] shows that `mean_poly` is equal to `[6, 6, 6, 6]`.
fn build_sumcheck_expressions<E: ExtensionField>(
    multiplier: Element,
    dim_size: usize,
) -> (Expression<E>, Expression<E>) {
    // The first sumcheck expression
    // Define constant values needed
    let multiplier_field: E = multiplier.to_field();
    let dim_size_field: E = (dim_size as Element).to_field();
    let dim_vars = ceil_log2(dim_size);
    let two_mul = E::from_canonical_u64(1 << dim_vars);

    // This expression is `N * 2^k * (input_poly * input_poly - mean_poly * mean_poly)`, `input_poly` has WitnessId 0 and `mean_poly` has WitnessId 1
    let variance_expr = Expression::Constant(Either::Right(dim_size_field * two_mul))
        * Expression::WitIn(0)
        * Expression::WitIn(0)
        - Expression::WitIn(1) * Expression::WitIn(1);
    // `inv_sqrt_out` (so the output of the inverse square root lookup) has WitnessId 2
    // so we skip this Id and use 3 and 4 for the gamma and beta polys that are applied row-wise.
    let gamma_expression = Expression::WitIn(3);
    let beta_expression = Expression::WitIn(4);

    // Multiply the variance expression by `multiplier_field` so that it has the correct scaling factor to be used in the lookup.
    let first_part = Expression::Constant(Either::Right(multiplier_field)) * variance_expr;
    // This expression links `last_claim.eval` to the output of the inverse square root lookup
    let second_part = gamma_expression
        * Expression::WitIn(2)
        * (Expression::Constant(Either::Right(dim_size_field)) * Expression::WitIn(0)
            - Expression::WitIn(1))
        + beta_expression;
    // The `eq_polys` will always be the last polynomials registered.
    let input_eq = Expression::WitIn(5);
    let last_claim_eq = Expression::WitIn(6);
    // This is the expression linking inputs and outputs to the next layer/last_claim
    let first_expr = input_eq
        * (first_part + Expression::Challenge(0, 1, E::ONE, E::ZERO) * Expression::WitIn(2))
        + Expression::Challenge(0, 2, E::ONE, E::ZERO) * last_claim_eq * second_part;
    // This is the expression that shows `mean_poly` was correctly constructed.
    let second_expr = Expression::WitIn(0)
        * (Expression::WitIn(1)
            + Expression::Challenge(0, 1, two_mul, E::ZERO) * Expression::WitIn(2));

    (first_expr, second_expr)
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound(serialize = "E: Serialize", deserialize = "E: DeserializeOwned"))]
/// Proof for correct execution of a quantised [`LayerNorm`] operation.
pub struct LayerNormProof<E, PCS>
where
    E: ExtensionField,
    PCS: PolynomialCommitmentScheme<E>,
{
    /// The LogUp proofs for LayerNorm, they are ordered `inv_sqrt_lookup`, `range_lookup`.
    pub(crate) logup_proof: LogUpBatchProof<E>,
    /// Witness commitments for this layer
    pub(crate) commitment: PCS::Commitment,
    /// The IO proof that links all claims to `last_claim` and the input
    pub(crate) io_proof: IOPProof<E>,
    /// The final sumcheck proof used to prove that `mean_poly` is the sum along the correct dim of `input_poly`
    pub(crate) mean_proof: IOPProof<E>,
    /// The claimed evaluations of the commitments
    pub(crate) io_evaluations: Vec<E>,
}

impl<E: ExtensionField, PCS: PolynomialCommitmentScheme<E>> LayerNormProof<E, PCS> {
    pub(crate) fn get_lookup_data(&self) -> (Vec<E>, Vec<E>) {
        self.logup_proof.fractional_outputs()
    }
    pub(crate) fn write_commitment<T: Transcript<E>>(&self, transcript: &mut T) -> Result<()> {
        PCS::write_commitment(&self.commitment, transcript).map_err(|e| anyhow!("{e:?}"))
    }
}

impl PadOp for LayerNorm<Element> {
    fn pad_node(self, _si: &mut crate::padding::ShapeInfo) -> Result<Self>
    where
        Self: Sized,
    {
        let LayerNorm {
            gamma,
            beta,
            eps,
            quant_info,
        } = self;
        let padded_gamma = gamma.map_tensor(|t| t.pad_next_power_of_two());
        let padded_beta = beta.map_tensor(|t| t.pad_next_power_of_two());

        Ok(LayerNorm::<Element> {
            gamma: padded_gamma,
            beta: padded_beta,
            eps,
            quant_info,
        })
    }
}

impl<E, PCS> ProvableOp<E, PCS> for LayerNorm<Element>
where
    E: ExtensionField + Serialize + DeserializeOwned,
    E::BaseField: Serialize + DeserializeOwned,
    PCS: PolynomialCommitmentScheme<E> + Send + Sync,
    PCS::CommitmentWithWitness: Serialize + DeserializeOwned + Send + Sync,
    PCS::ProverParam: Send + Sync,
{
    type Ctx = LayerNormCtx<E>;

    fn prove<T: transcript::Transcript<E>>(
        &self,
        node_id: NodeId,
        ctx: &Self::Ctx,
        last_claims: Vec<&Claim<E>>,
        step_data: &Step<E, Element, E>,
        prover: &mut Prover<E, T, PCS>,
        store: &mut GenStore,
    ) -> Result<Vec<Claim<E>>> {
        let input_tensors = step_data.input_tensors(store)?;
        // Check there is a single input
        ensure!(
            input_tensors.len() == 1,
            "LayerNorm step should only have one input, received {}",
            input_tensors.len()
        );
        let input_mle: MultilinearExtension<E> = input_tensors[0].get_data().to_vec().into_mle();
        // We also make the MLE for the sum of each dim we perform layernorm on
        let last_dim = *input_tensors[0]
            .shape()
            .last()
            .ok_or(anyhow!("Step data input tensor had no shape in LayerNorm"))?;
        let mean_mle = input_tensors[0]
            .get_data()
            .chunks(last_dim)
            .flat_map(|chunk| {
                let sum = chunk.iter().copied().sum::<E>();
                vec![sum; last_dim]
            })
            .collect::<Vec<E>>()
            .into_mle();
        let (claims, proof) =
            self.prove_step(node_id, last_claims, ctx, input_mle, mean_mle, prover)?;
        // Add the proof to the proof list
        prover.push_proof(node_id, LayerProof::<E, PCS>::LayerNorm(proof));

        Ok(claims)
    }

    fn gen_lookup_witness(
        &self,
        id: NodeId,
        ctx: &ProverContext<E, PCS>,
        step_data: &Step<E, Element, Element>,
        store: &mut GenStore,
    ) -> Result<LookupWitnessGen<E, PCS>> {
        let output_tensors = step_data.output_tensors(store)?;
        ensure!(
            step_data.node_inputs.len() == 1,
            "Found more than 1 input in inference step of LayerNorm layer"
        );
        ensure!(
            output_tensors.len() == 1,
            "Found more than 1 output in inference step of LayerNorm layer"
        );
        let layernorm_data = step_data.node_outputs.try_layernorm_data().ok_or(anyhow!(
            "LayerNorm data not found in inference step for LayerNorm layer"
        ))?;
        self.lookup_witness(id, ctx, layernorm_data)
    }
}

type ProveOut<E, PCS> = (Vec<Claim<E>>, LayerNormProof<E, PCS>);
impl LayerNorm<Element> {
    pub(crate) fn prove_step<E, T, PCS>(
        &self,
        node_id: NodeId,
        last_claims: Vec<&Claim<E>>,
        ctx: &LayerNormCtx<E>,
        input_poly: MultilinearExtension<E>,
        mean_poly: MultilinearExtension<E>,
        prover: &mut Prover<E, T, PCS>,
    ) -> Result<ProveOut<E, PCS>>
    where
        E: ExtensionField,
        T: transcript::Transcript<E>,
        PCS: PolynomialCommitmentScheme<E> + Send + Sync,
        PCS::CommitmentWithWitness: Serialize + DeserializeOwned + Send + Sync,
        PCS::ProverParam: Send + Sync,
    {
        // Check we have the correct number of claims
        ensure!(
            last_claims.len() == 1,
            "LayerNorm only produces one output claim but got: {}",
            last_claims.len()
        );
        let last_claim = last_claims[0];

        let layer_commitment = prover.lookup_witness(node_id)?;
        let logup_inputs = ctx
            .lookup_ctx
            .create_logup_inputs::<PCS, E>(layer_commitment, &prover.challenge_storage)?;
        let layer_polys = PCS::get_arc_mle_witness_from_commitment(layer_commitment);
        let commitment = PCS::get_pure_commitment(layer_commitment);

        // Run the lookup proof
        let logup_batch_proof = batch_multiple_sizes_prove(&logup_inputs, prover.transcript)?;

        let batching_challenge = prover
            .transcript
            .sample_and_append_challenge(b"batching")
            .elements;

        let num_vars = input_poly.num_vars();
        let diff = num_vars - logup_batch_proof.output_claims()[0].point.len();
        let logup_vars = logup_batch_proof.output_claims()[0].point.len();
        let num_threads = optimal_sumcheck_threads(num_vars);

        let full_point = std::iter::repeat_n(E::TWO.inverse(), diff)
            .chain(logup_batch_proof.output_claims()[0].point.iter().copied())
            .collect::<Vec<E>>();

        let input_eq = compute_betas_eval(&full_point).into_mle();
        let last_claim_eq = compute_betas_eval(&last_claim.point).into_mle();

        let inv_sqrt_poly_evals = layer_polys
            .last()
            .ok_or(anyhow!("No Layer Polys for Layer Norm so cannot prove"))?
            .get_base_field_vec();
        let inv_sqrt_poly = MultilinearExtension::<E>::from_evaluations_vec(
            num_vars,
            inv_sqrt_poly_evals
                .iter()
                .flat_map(|&v| vec![v; 1 << diff])
                .collect::<Vec<E::BaseField>>(),
        );

        let gamma_poly: MultilinearExtension<E> =
            std::iter::repeat_n(self.gamma.to_field::<E>(), 1 << logup_vars)
                .flatten()
                .collect::<Vec<E>>()
                .into_mle();
        let beta_poly: MultilinearExtension<E> =
            std::iter::repeat_n(self.beta.to_field::<E>(), 1 << logup_vars)
                .flatten()
                .collect::<Vec<E>>()
                .into_mle();
        let either_mles = [
            &input_poly,
            &mean_poly,
            &inv_sqrt_poly,
            &gamma_poly,
            &beta_poly,
            &input_eq,
            &last_claim_eq,
        ]
        .into_iter()
        .map(Either::Left)
        .collect::<Vec<Either<_, _>>>();

        let expr_builder =
            VirtualPolynomialsBuilder::<E>::new_with_mles(num_threads, num_vars, either_mles);
        let virtual_poly =
            expr_builder.to_virtual_polys(&ctx.first_sumcheck_expression, &[batching_challenge]);
        let (io_proof, state) = IOPProverState::<E>::prove(virtual_poly, prover.transcript);
        let io_point = state
            .challenges
            .iter()
            .map(|c| c.elements)
            .collect::<Vec<E>>();
        let io_evaluations = state.get_mle_flatten_final_evaluations()[..5].to_vec();

        // Now we perform the sumcheck for the mean
        let input_io_eq_poly = compute_betas_eval(&io_point).into_mle();
        let mean_point = std::iter::repeat_n(E::TWO.inverse(), diff)
            .chain(io_point.iter().skip(diff).copied())
            .collect::<Vec<E>>();
        let mean_io_eq_poly = compute_betas_eval(&mean_point).into_mle();

        let batching_challenge = prover
            .transcript
            .sample_and_append_challenge(b"batching")
            .elements;

        let either_mles = [&input_poly, &input_io_eq_poly, &mean_io_eq_poly]
            .into_iter()
            .map(Either::Left)
            .collect::<Vec<Either<_, _>>>();
        let expr_builder =
            VirtualPolynomialsBuilder::<E>::new_with_mles(num_threads, num_vars, either_mles);
        let virtual_poly =
            expr_builder.to_virtual_polys(&ctx.mean_sumcheck_expression, &[batching_challenge]);
        let (mean_proof, state) = IOPProverState::<E>::prove(virtual_poly, prover.transcript);
        let input_eval = state.get_mle_flatten_final_evaluations()[0];
        let input_point = state
            .challenges
            .iter()
            .map(|c| c.elements)
            .collect::<Vec<E>>();
        let input_claim = Claim::<E>::new(input_point, input_eval);

        // Add the commitment claims to the commitment prover
        let first_commit_claims = (
            logup_batch_proof.output_claims()[0].point.to_vec(),
            logup_batch_proof
                .output_claims()
                .iter()
                .take(logup_batch_proof.output_claims().len() - 1)
                .map(|claim| claim.eval)
                .collect::<Vec<E>>(),
        );
        let second_commit_claim = (io_point[diff..].to_vec(), vec![io_evaluations[2]]);

        prover.add_witness_claim(node_id, vec![first_commit_claims, second_commit_claim]);

        let common_claims = {
            let point = io_point.iter().take(diff).copied().collect::<Vec<E>>();
            let mut claims = HashMap::new();
            claims.insert(
                self.gamma.commitment_id(),
                Claim::<E>::new(point.clone(), io_evaluations[3]),
            );
            claims.insert(
                self.beta.commitment_id(),
                Claim::<E>::new(point, io_evaluations[4]),
            );
            claims
        };
        prover.add_common_claims(node_id, common_claims);

        let proof = LayerNormProof::<E, PCS> {
            logup_proof: logup_batch_proof,
            commitment,
            io_proof,
            mean_proof,
            io_evaluations,
        };

        Ok((vec![input_claim], proof))
    }

    /// Internal method for generating the [`LogUpWitness`] for a [`LayerNorm`] step.
    fn lookup_witness<E, PCS>(
        &self,
        id: NodeId,
        ctx: &ProverContext<E, PCS>,
        layernorm_data: &LayerNormData,
    ) -> Result<LookupWitnessGen<E, PCS>>
    where
        E: ExtensionField,
        PCS: PolynomialCommitmentScheme<E>,
        PCS::CommitmentWithWitness: Serialize + DeserializeOwned + Send + Sync,
        PCS::ProverParam: Send + Sync,
    {
        let mut wit_gen = LookupWitnessGen::<E, PCS>::default();
        // Get the data generated during quantised evaluation
        let LayerNormData {
            full_value,
            lookup_output,
        } = layernorm_data;

        // We need to work out how many chunks to split the shifted away part into to be range checked
        let QuantisedLayerNormData {
            top_chunk_scalar_log,
            lut,
            ..
        } = self.quant_info().ok_or(anyhow!(
            "Could not prove LayerNorm because it had no quantisation data"
        ))?;
        let number_range_checks = (lut.range_check_bits() - 1) / *quantization::BIT_LEN + 1;

        let range_check_mask: Element = bit_to_mask(lut.range_check_bits());
        let lookup_input: Vec<Element> = full_value
            .iter()
            .map(|v| v >> lut.range_check_bits())
            .collect();
        let range_check: Vec<Element> = full_value.iter().map(|v| v & range_check_mask).collect();

        // Split `range_check` into its constituent parts
        let range_mask: Element = (1 << *quantization::BIT_LEN) - 1;
        let top_chunk_scalar: Element = 1 << top_chunk_scalar_log;
        let mut range_checks = (0..number_range_checks)
            .into_par_iter()
            .map(|j| {
                if j != number_range_checks - 1 {
                    range_check
                        .iter()
                        .map(|&elem| {
                            let tmp = elem >> (j * *quantization::BIT_LEN);
                            tmp & range_mask
                        })
                        .collect::<Vec<Element>>()
                } else {
                    // In the final chunk after being shifted everything has to get multiplied by 1 << top_chunk_scalar_log
                    range_check
                        .iter()
                        .map(|&elem| {
                            let tmp = elem >> (j * *quantization::BIT_LEN);
                            (tmp & range_mask) * top_chunk_scalar
                        })
                        .collect::<Vec<Element>>()
                }
            })
            .collect::<Vec<Vec<Element>>>();
        let range_elements_count =
            range_checks
                .iter()
                .fold(HashMap::<Element, u64>::new(), |mut acc, range_check| {
                    range_check
                        .iter()
                        .for_each(|v| *acc.entry(*v).or_default() += 1);
                    acc
                });

        let inv_sqrt_element_count = lookup_input.iter().zip(lookup_output.iter()).fold(
            HashMap::<Element, u64>::new(),
            |mut acc, (&input, &output)| {
                *acc.entry(input + output * COLUMN_SEPARATOR).or_default() += 1;
                acc
            },
        );

        // Make the commitments to the lookups
        let width = 1 + number_range_checks;
        range_checks.push(lookup_input.clone());
        let transposed = transpose(range_checks);
        let first_values = to_base::<E, _>(transposed.concat());
        let rmm1 = RowMajorMatrix::<E::BaseField>::new_by_inner_matrix(
            ceno_p3::matrix::dense::DenseMatrix::new(first_values, width),
            InstancePaddingStrategy::Default,
        );
        let rmm2 = RowMajorMatrix::<E::BaseField>::new_by_inner_matrix(
            ceno_p3::matrix::dense::DenseMatrix::new(to_base::<E, _>(lookup_output), 1),
            InstancePaddingStrategy::Default,
        );
        let layer_commitment = ctx.commitment_ctx.batch_commit(vec![rmm1, rmm2])?;

        // Add the merged columns to the lookups lists
        wit_gen.insert_element_count(TableType::Range, range_elements_count);

        wit_gen.insert_element_count(TableType::InverseSQRT(*lut), inv_sqrt_element_count);

        // Insert the LogUpWitnesses
        wit_gen.insert_logup_witness(id, layer_commitment);
        Ok(wit_gen)
    }
}

impl<E, PCS> VerifiableCtx<E, PCS> for LayerNormCtx<E>
where
    E: ExtensionField + Serialize + DeserializeOwned,
    E::BaseField: Serialize + DeserializeOwned,
    PCS: PolynomialCommitmentScheme<E>,
{
    type Proof = LayerNormProof<E, PCS>;

    fn verify<T: transcript::Transcript<E>>(
        &self,
        proof: &Self::Proof,
        last_claims: &[&Claim<E>],
        verifier: &mut Verifier<E, T, PCS>,
        _shape_step: &ShapeStep,
    ) -> Result<Vec<Claim<E>>> {
        // First we check that we only have one claim in `last_claims`
        ensure!(
            last_claims.len() == 1,
            "LayerNorm only outputs 1 claim, received {} while verifying LayerNorm step",
            last_claims.len()
        );

        let last_claim = last_claims[0];

        let LayerNormProof {
            logup_proof,
            commitment,
            io_proof,
            mean_proof,
            io_evaluations,
        } = proof;

        // Verify the lookup proof
        let batch_claim = verify_logup_proof_multiple_sizes(logup_proof, verifier.transcript)?;
        self.lookup_ctx
            .verify_logup_batch_claim(&batch_claim, &verifier.challenge_storage)?;

        // Now we squeeze the batching challenge
        let alpha = verifier
            .transcript
            .sample_and_append_challenge(b"batching")
            .elements;

        let poly_evals = batch_claim.poly_evals();
        let range_evals = &poly_evals[..poly_evals.len() - 2];
        let inv_sqrt_input_eval = poly_evals[poly_evals.len() - 2];
        let inv_sqrt_output_eval = poly_evals[poly_evals.len() - 1];

        let pow_two_multiplier = E::from_canonical_u64(1 << *quantization::BIT_LEN);
        let (partial_eval, power_two) = range_evals.iter().take(range_evals.len() - 1).fold(
            (
                inv_sqrt_input_eval * E::from_canonical_u64(1 << self.range_check_bits),
                E::ONE,
            ),
            |(acc, pow), &eval| (acc + eval * pow, pow * pow_two_multiplier),
        );
        // The last range evaluation has to be rescaled
        let top_chunk_scalar_inv = E::from_canonical_u64(1 << self.top_chunk_scalar_log).inverse();

        let claimed_sum = partial_eval
            + *range_evals.last().unwrap() * top_chunk_scalar_inv * power_two
            + alpha * (inv_sqrt_output_eval + alpha * last_claim.eval);
        let aux_info = VPAuxInfo {
            max_num_variables: last_claim.point.len(),
            max_degree: 4,
            ..Default::default()
        };
        let io_subclaim =
            IOPVerifierState::<E>::verify(claimed_sum, io_proof, &aux_info, verifier.transcript);
        let io_point = io_subclaim
            .point
            .iter()
            .map(|c| c.elements)
            .collect::<Vec<E>>();

        let diff = io_point.len() - batch_claim.point().len();
        let full_point = std::iter::repeat_n(E::TWO.inverse(), diff)
            .chain(batch_claim.point().iter().copied())
            .collect::<Vec<E>>();

        let input_eq = identity_eval(&full_point, &io_point);
        let last_claim_eq = identity_eval(&last_claim.point, &io_point);
        let witnesses = io_evaluations
            .iter()
            .copied()
            .chain([input_eq, last_claim_eq])
            .collect::<Vec<E>>();

        let calc_claim = eval_by_expr_with_instance(&[], &witnesses, &[], &[], &[alpha], &self.first_sumcheck_expression[0]).right().ok_or(anyhow!("LayerNorm verification failed, first sumcheck expression did not evaluate to an extension field element"))?;

        ensure!(
            calc_claim == io_subclaim.expected_evaluation,
            "LayerNorm verification failed, calculated claim {:?} did not equal the expected IO evaluation {:?}",
            calc_claim,
            io_subclaim.expected_evaluation
        );

        // Now we verify the mean sumcheck
        let alpha = verifier
            .transcript
            .sample_and_append_challenge(b"batching")
            .elements;

        let claimed_sum = io_evaluations[0] + alpha * io_evaluations[1];
        let aux_info = VPAuxInfo {
            max_num_variables: io_point.len(),
            max_degree: 2,
            ..Default::default()
        };
        let mean_subclaim =
            IOPVerifierState::<E>::verify(claimed_sum, mean_proof, &aux_info, verifier.transcript);
        let mean_point = mean_subclaim
            .point
            .iter()
            .map(|c| c.elements)
            .collect::<Vec<E>>();
        let input_eq = identity_eval(&io_point, &mean_point);

        let sum_point = std::iter::repeat_n(E::TWO.inverse(), diff)
            .chain(io_point.iter().skip(diff).copied())
            .collect::<Vec<E>>();
        let sum_eq = identity_eval(&sum_point, &mean_point);

        let mult = input_eq + E::from_canonical_u64(1 << diff) * alpha * sum_eq;

        let input_eval = mean_subclaim.expected_evaluation * mult.inverse();
        let input_claim = Claim::<E>::new(mean_point, input_eval);

        // Now we add the commitments to the verifier
        let first_commit = (
            batch_claim.point().to_vec(),
            poly_evals[..poly_evals.len() - 1].to_vec(),
        );
        let second_commit = (io_point[diff..].to_vec(), vec![io_evaluations[2]]);

        verifier.commit_verifier.add_witness_claim(
            self.node_id,
            commitment.clone(),
            vec![first_commit, second_commit],
        );

        let common_claims = {
            let point = io_point.iter().take(diff).copied().collect::<Vec<E>>();
            let mut claims = HashMap::new();
            claims.insert(
                self.gamma_key.clone(),
                Claim::<E>::new(point.clone(), io_evaluations[3]),
            );
            claims.insert(
                self.beta_key.clone(),
                Claim::<E>::new(point, io_evaluations[4]),
            );
            claims
        };
        verifier.add_common_claims(self.node_id, common_claims);

        Ok(vec![input_claim])
    }

    fn write_proof_to_transcript<T: Transcript<E>>(
        &self,
        proof: &Self::Proof,
        transcript: &mut T,
    ) -> anyhow::Result<()> {
        proof.write_commitment(transcript)
    }
}

#[cfg(test)]
mod tests {
    use ff_ext::GoldilocksExt2;
    use itertools::izip;
    use proptest::prelude::*;
    use std::{
        fmt::{Debug, Display},
        ops::Range,
    };

    use crate::{
        init_test_logging_default,
        layers::{Evaluate, Layer},
        model::{Model, test::prove_model},
        tensor::is_close_with_tolerance,
    };

    use super::*;

    impl<N: Number + TensorTypeParam> LayerNorm<N> {
        pub fn random(size: usize, layer_name: Option<CommitmentId>) -> Self {
            let layer_name = layer_name.unwrap_or("layernorm".to_string().into());
            let gamma = KeyedTensor::new(
                format!("{layer_name}_gamma"),
                Tensor::<N>::random(&vec![size].into()),
            );
            let beta = KeyedTensor::new(
                format!("{layer_name}_beta"),
                Tensor::<N>::random(&vec![size].into()),
            );
            let eps = 1e-5;
            Self::new(gamma, beta, eps)
        }
    }

    type E = GoldilocksExt2;

    #[test]
    fn test_layernorm() {
        let gamma = KeyedTensor::new(
            "layernorm_gamma",
            Tensor::<f32>::new(vec![1024].into(), vec![1.0; 1024]),
        );
        let beta = KeyedTensor::new(
            "layernorm_beta",
            Tensor::<f32>::new(vec![1024].into(), vec![0.0; 1024]),
        );
        let eps = 1e-5;
        let layernorm = LayerNorm {
            gamma,
            beta,
            eps,
            quant_info: None,
        };
        let input = Tensor::<f32>::new(vec![1, 1024].into(), vec![0.0; 1024]).into_wrapped();
        let output = layernorm.evaluate::<E>(&[&input]).unwrap();
        assert_eq!(output.outputs[0].shape(), vec![1_usize, 1024].into());
        assert_eq!(output.outputs[0].get_data(), vec![0.0; 1024]);
    }

    #[test]
    fn test_quantise_layernorm() {
        let layernorm = LayerNorm::random(100, None);
        // Make a random float input tensor and derive the input ScalingFactor
        let input_tensor = Tensor::<f32>::random(&vec![2, 100].into());
        let input_scaling = ScalingFactor::from_tensor(&input_tensor, None);
        // We quantise the float input to obtain `quant_tensor` and then we dequantise to obtain `dequant_input`
        // this lets us run quantised evaluation and floating point evaluation and compare the outputs.
        let quant_tensor = input_tensor.to_quantized(&input_scaling);
        let dequant_input = quant_tensor.dequantize(&input_scaling);

        let dequant_output = layernorm
            .evaluate::<E>(&[&dequant_input.as_wrapped()])
            .unwrap()
            .outputs[0]
            .clone();
        // Construct the quantised LayerNorm
        let (quant_layernorm, _, output_scaling) =
            layernorm.quantise(input_scaling, input_scaling).unwrap();

        let quant_output = quant_layernorm
            .evaluate::<E>(&[&quant_tensor.as_wrapped()])
            .unwrap()
            .outputs[0]
            .clone();

        let quant_output_dequant = quant_output.to_native().dequantize(&output_scaling);
        let a = quant_output_dequant.get_data();
        let b = dequant_output.get_data();
        assert!(
            is_close_with_tolerance(a, &b, 5e-2_f32, 1e-1_f32),
            "Wasn't close enough to floating point version"
        );
    }

    #[test]
    fn test_layernorm_proving() {
        init_test_logging_default();

        let layernorm = LayerNorm::random(100, None);
        let mut model =
            Model::new_from_input_shapes(vec![vec![15, 100].into()], PaddingMode::NoPadding);

        let _ = model
            .add_consecutive_layer(Layer::LayerNorm(layernorm), None)
            .unwrap();

        model.automatic_output_labelling().unwrap();
        model.describe();
        prove_model(model, &mut GenStore::default()).unwrap();
    }

    #[derive(Clone)]
    struct Input<T> {
        input: Tensor<T>,
        beta: KeyedTensor<T>,
        gamma: KeyedTensor<T>,
    }

    impl<T: Debug> Debug for Input<T> {
        fn fmt(
            &self,
            fmt: &mut std::fmt::Formatter<'_>,
        ) -> std::result::Result<(), std::fmt::Error> {
            fmt.debug_struct("Input")
                .field("input", &format_args!("{:?}", self.input))
                .field("beta", &format_args!("{:?}", self.beta))
                .field("gamma", &format_args!("{:?}", self.gamma))
                .finish()
        }
    }

    impl<T> Display for Input<T> {
        fn fmt(&self, fmt: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
            write!(
                fmt,
                "Input{{input: {:?}, beta: {:?}, gamma: {:?}}}",
                self.input.shape(),
                self.beta.shape(),
                self.gamma.shape(),
            )
        }
    }

    fn input<T: TensorTypeParam>(
        dim0: Range<usize>,
        dim1: Range<usize>,
    ) -> impl Strategy<Value = Input<T>> {
        (dim0, dim1).prop_flat_map(|(dim0, dim1)| {
            let input = Tensor::any(Shape::new(vec![dim0, dim1]));
            let beta = Tensor::any(Shape::new(vec![dim1]));
            let gamma = Tensor::any(Shape::new(vec![dim1]));
            (input, beta, gamma).prop_map(|(input, beta, gamma)| Input {
                input,
                beta: KeyedTensor::new("layernorm_beta", beta),
                gamma: KeyedTensor::new("layernorm_gamma", gamma),
            })
        })
    }

    fn evaluate(
        input: &Tensor<Element>,
        beta: &Tensor<Element>,
        gamma: &Tensor<Element>,
        quant_info: &QuantisedLayerNormData,
    ) -> (Vec<Element>, Vec<Element>, Vec<Element>) {
        let QuantisedLayerNormData {
            multiplier,
            lut,
            dim_size,
            ..
        } = quant_info;
        let final_dim = *input.shape().last().unwrap();

        let (inv_sqrt_output, full_value): (Vec<Element>, Vec<Element>) = input
            .get_data()
            .chunks(final_dim)
            .map(|chunk| {
                let sum_squares = chunk.iter().map(|x| *x * *x).sum::<Element>();
                let sum = chunk.iter().sum::<Element>();
                let full_value =
                    *dim_size as Element * multiplier * sum_squares - multiplier * sum * sum;
                let inv_sqrt = full_value >> lut.range_check_bits();
                let inv_sqrt_output = lut.table_output(inv_sqrt);

                (inv_sqrt_output, full_value)
            })
            .unzip();

        let output_data = input
            .get_data()
            .chunks(final_dim)
            .zip(inv_sqrt_output.iter())
            .flat_map(|(input_chunk, denominator)| {
                let sum = input_chunk.iter().sum::<Element>();
                izip!(input_chunk, gamma.get_data(), beta.get_data())
                    .map(|(&v, &gamma, &beta)| {
                        gamma * (*dim_size as Element * v - sum) * *denominator + beta
                    })
                    .collect::<Vec<Element>>()
            })
            .collect::<Vec<Element>>();

        (inv_sqrt_output, full_value, output_data)
    }

    #[test]
    fn test_layernorm_simple() {
        let dim0 = 2;
        let dim1 = 5;

        let input = Tensor::<Element>::random(&Shape::new(vec![dim0, dim1]));

        // NOTE:
        // Layer quantisation changes the values of beta and gamma, use the
        // values stored in the layer for the comparison.
        let layer = LayerNorm::<f32>::random(dim1, None);
        let input_scaling = ScalingFactor::from_tensor(&input, None);
        let (layer, _, _) = layer.quantise(input_scaling, input_scaling).unwrap();

        let expected = evaluate(
            &input,
            &layer.beta,
            &layer.gamma,
            layer.quant_info.as_ref().unwrap(),
        );

        let result = layer
            .evaluate::<GoldilocksExt2>(&[&input.as_wrapped()])
            .unwrap();
        assert_eq!(
            &result.outputs()[0].get_data(),
            &expected.2,
            "Output mismatch"
        );

        let expected_proof_data = result.try_layernorm_data().unwrap();
        assert_eq!(
            &expected_proof_data.full_value, &expected.1,
            "Full value mismatch"
        );
        assert_eq!(
            &expected_proof_data.lookup_output, &expected.0,
            "Lookup output mismatch"
        );
    }

    /// Ensures the CPU and GPU implementation agrees on the rounding of 0.5 values.
    ///
    /// The values of the test below have been found via property testing.
    #[test]
    fn test_regression_layernorm_rounding() {
        let dim0 = 2;
        let dim1 = 25;

        let input = Tensor::<Element>::new(
            Shape::new(vec![dim0, dim1]),
            vec![
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10, 69,
                -126, -128, 105, -111, -47, 89, -113, 9, -64, 42, -111, 54, 104, 62, 127, 12, 84,
                7, -54, -80, 0, -122, -41,
            ],
        );
        let gamma = KeyedTensor::new(
            "layernorm_gamma",
            Tensor::<Element>::new(
                Shape::new(vec![dim1]),
                vec![
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                ],
            ),
        );
        let beta = KeyedTensor::new(
            "layernorm_beta",
            Tensor::new(
                Shape::new(vec![dim1]),
                vec![
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                ],
            ),
        );

        // NOTE:
        // Layer quantisation changes the values of beta and gamma, use the
        // values stored in the layer for the comparison.
        let layer = LayerNorm::new(gamma, beta, 1e-5);
        let input_scaling = ScalingFactor::from_tensor(&input, None);
        let (layer, _, _) = layer.quantise(input_scaling, input_scaling).unwrap();

        let expected = evaluate(
            &input,
            &layer.beta,
            &layer.gamma,
            layer.quant_info.as_ref().unwrap(),
        );

        let result = layer
            .evaluate::<GoldilocksExt2>(&[&input.as_wrapped()])
            .unwrap();
        assert_eq!(
            &result.outputs()[0].get_data(),
            &expected.2,
            "Output mismatch"
        );

        let expected_proof_data = result.try_layernorm_data().unwrap();
        assert_eq!(
            &expected_proof_data.full_value, &expected.1,
            "Full value mismatch"
        );
        assert_eq!(
            &expected_proof_data.lookup_output, &expected.0,
            "Lookup output mismatch"
        );
    }

    proptest! {
        #[test]
        fn proptest_layer_norm_evaluate_element(input in input(1usize..64, 1usize..64)) {
            // NOTE:
            // Layer quantisation changes the values of beta and gamma, use the
            // values stored in the layer for the comparison.
            let data = input.clone();
            let layer = LayerNorm::new(input.gamma, input.beta, 1e-5);
            let input_scaling = ScalingFactor::from_tensor(&input.input, None);
            let (layer, _, _) = layer.quantise(input_scaling, input_scaling).unwrap();

            let expected = evaluate(&input.input, &layer.beta, &layer.gamma, layer.quant_info.as_ref().unwrap());

            let result = layer.evaluate::<GoldilocksExt2>(&[&input.input.as_wrapped()]).unwrap();
            prop_assert_eq!(&result.outputs()[0].get_data(), &expected.2, "Output mismatch. input {:?}", data);

            let expected_proof_data = result.try_layernorm_data().unwrap();
            prop_assert_eq!(&expected_proof_data.full_value, &expected.1, "Full value mismatch. input {:?}", data);
            prop_assert_eq!(&expected_proof_data.lookup_output, &expected.0, "Lookup output mismatch. input {:?}", data);
        }

    }
}
