//! This layer applies the softmax function to the last dimension of the input tensor
use core::f32;
use std::fmt::Debug;

use crate::{
    Claim, Element, ScalingStrategy, Shape, Tensor,
    backend::Backend,
    commit::{compute_betas_eval, identity_eval},
    iop::{
        ChallengeStorage,
        context::{ContextAux, ProverContext, ShapeStep},
        verifier::Verifier,
    },
    layers::{
        LayerCtx, LayerProof,
        provable::{
            Evaluate, LayerOut, NodeId, OpInfo, PadOp, ProvableOp, ProveInfo, ProvingData,
            QuantizeOp, QuantizeOutput, VerifiableCtx,
        },
        requant::FIXED_POINT_SCALE,
    },
    lookup::{
        context::{
            COLUMN_SEPARATOR, LayerLookupContext, LookupWitnessGen, TableType, count_elements,
        },
        logup_gkr::{
            prover::batch_multiple_sizes_prove,
            structs::{LogUpBatchProof, LogUpInput},
            verifier::verify_logup_proof_multiple_sizes,
        },
    },
    model::{StepData, transform::impls::softmax_mask::SoftmaxMaskTransform},
    number::Number,
    padding::PaddingMode,
    quantization::{self, Fieldizer, ScalingFactor},
    to_base,
};

use anyhow::{Result, anyhow, bail, ensure};
use burn::tensor::{Int as BInt, Tensor as BTensor, TensorData, activation::softmax};
use either::Either;
use ff_ext::ExtensionField;
use itertools::{Itertools, izip};
use mpcs::PolynomialCommitmentScheme;
use multilinear_extensions::{
    Expression,
    mle::IntoMLE,
    util::{ceil_log2, transpose},
    utils::eval_by_expr_with_instance,
    virtual_poly::VPAuxInfo,
    virtual_polys::VirtualPolynomialsBuilder,
};
use rayon::prelude::*;
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use sumcheck::{
    structs::{IOPProof, IOPProverState, IOPVerifierState},
    util::optimal_sumcheck_threads,
};
use tenstore::GenStore;
use transcript::Transcript;
use witness::RowMajorMatrix;

/// The short name used to identify the Softmax layer
pub const SOFTMAX_LAYER: &str = "SFTM";

#[derive(Debug, Clone, Serialize, Deserialize)]
/// Stores data about the Softmax operation, which is used to map a tensor of values to a tensor of probability distributions.
/// This is done by picking a dimension to normalise over and calculating
///             `x -> exp(scale * x) / (\sum_{i \in dim} exp(scale * x_{i}))`.
pub struct Softmax<N> {
    // By default, it's equal to 1
    /// In the floating point case this is the factor we multiply by before exponentiating, when thought of as a Boltzmann distribution this is
    /// often referred to as the "Temperature".
    ///
    /// For the quantised version this should be 1 as the temperature will be absorbed into the rescaling factor.
    pub scalar: N,
    /// This is the maximum size of dimension that we will normalise over. For example in an Attention layer this would be the maximum context size.
    max_size: usize,
    /// This is the extra information required to compute the quantised version, it defaults to [`None`].
    quant_info: Option<QuantisedSoftmaxData>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Copy)]
/// This struct is used to store information used when evaluating the quantised version of [`Softmax`] on
/// [`Element`]s.
pub(crate) struct QuantisedSoftmaxData {
    /// After multiplying by `self.fixed_point_multiplier` the value need to be shifted by this plus 25.
    pub right_shift: usize,
    /// The normalised scaling factor including temperature rescaling represented as a fixed point multiplier (it should have 24 fractional bits)
    pub fixed_point_multiplier: Element,
    /// The scale used for the fixed point multiplier
    pub fp_scale: usize,
    /// The actual multiplier, this is mainly used to compare accuracy, it has no purpose in actual proving
    pub multiplier: f32,
    /// The intermediate bit size, allowing us to work out how many zero tables we need
    pub intermediate_bit_size: usize,
    /// This stores the [`ExpTable`]
    pub(crate) lut: ExpTable,
    /// The error bound as calculated by the formulae given in the zkLLM paper, this is the relative error bound on the normalisation sum.
    error_bound: f32,
    /// The original [`ScalingFactor`] of the input
    input_scaling_factor: ScalingFactor,
    /// The temperature
    temperature: f32,
}

impl QuantisedSoftmaxData {
    /// Function that tells us how many bits are not shifted away
    pub(crate) fn output_bit_size(&self) -> usize {
        let fpm_bit_size = ceil_log2(self.fixed_point_multiplier as usize);
        self.intermediate_bit_size + 1 + fpm_bit_size - self.right_shift
    }

    /// Function that returns how many zero-chunks the [`Softmax`] contains
    pub(crate) fn number_of_zero_chunks(&self) -> usize {
        // We take the output bit size and subtract the ExpTable bit size (as this many bits are passed to the ExpTable)
        // and then divide by the quantization::BIT_LEN
        let out_bit_size = self.output_bit_size();
        if out_bit_size <= self.lut.table_bit_size() {
            // We always have at least one zero chunk because of having to mask out padding
            1
        } else {
            1 + (out_bit_size - self.lut.table_bit_size() - 1) / *quantization::BIT_LEN
        }
    }

    /// Calculates how many range checks are needed for the Softmax operation
    pub(crate) fn number_of_range_checks(&self) -> usize {
        // This is just the right shift ceiling divided by the quantization::BIT_LEN
        1 + (self.right_shift - 1) / *quantization::BIT_LEN
    }

    /// Calculates the largest value that will be mapped to zero in quantised evaluation of Softmax.
    pub(crate) fn quantised_negative_infinity(&self) -> Element {
        // This is the largest possible value that can appear after fixed point multiplication and before right shift
        let max_poss_bits_after_mult = (*quantization::BIT_LEN * self.number_of_zero_chunks())
            + self.lut.table_bit_size()
            + self.right_shift;
        // We subtract FIXED_POINT_SCALE + 1 because this will give us the largest value pre fixed point multiplication and shift addition and the addition of the row max
        let max_poss_bits_pre_mult = max_poss_bits_after_mult - FIXED_POINT_SCALE - 2;
        -1 << max_poss_bits_pre_mult
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
/// Struct used to store Softmax table data
pub struct ExpTable {
    /// This is the input scale factor stored as the bit representation of a f32
    input_sf: u32,
    /// This is the output scale factor stored as the bit representation of a f32
    output_sf: u32,
    /// The bit size of the exp table
    bit_size: usize,
}

impl ExpTable {
    /// Creates a new [`ExpTable`] with the given input and output scale factors and bit size.
    pub fn new(input_sf: f32, output_sf: f32, bit_size: usize) -> Self {
        ExpTable {
            input_sf: input_sf.to_bits(),
            output_sf: output_sf.to_bits(),
            bit_size,
        }
    }
    /// Returns the input scale factor as a [`f32`]
    pub(crate) fn input_sf(&self) -> f32 {
        f32::from_bits(self.input_sf)
    }
    /// Returns the output scale factor as a [`f32`]
    pub(crate) fn output_sf(&self) -> f32 {
        f32::from_bits(self.output_sf)
    }
    /// Returns the bit size of the exp table
    pub(crate) fn table_bit_size(&self) -> usize {
        self.bit_size
    }
    /// Returns the full size of the exp table as an [`Element`]
    pub(crate) fn full_table_size(&self) -> Element {
        1 << self.bit_size
    }
    /// Given an [`Element`] as input, calculates the output of the exp table as an [`Element`]. It is important to note that
    /// this method does not check that the input is within the bounds of the table, it is the caller's responsibility to ensure this.
    pub(crate) fn table_output(&self, j: Element) -> Element {
        let input_sf = self.input_sf();
        let output_sf = self.output_sf();

        let float_exp = (j as f32 / input_sf).exp();
        (float_exp * output_sf).round_ties_even() as Element
    }
}

#[derive(Clone, Serialize, Deserialize)]
#[serde(bound(serialize = "E: Serialize", deserialize = "E: DeserializeOwned"))]
/// Proof for correct execution of a quantised [`Softmax`] operation.
pub struct SoftmaxProof<E, PCS>
where
    E: ExtensionField,
    PCS: PolynomialCommitmentScheme<E>,
{
    /// The LogUp proofs for Softmax, they are ordered `exp_lookup`, `range_lookup`, `error_lookup` and then `zero_table_lookup` if it exists
    pub(crate) logup_proof: LogUpBatchProof<E>,
    /// Witness commitments for this layer
    pub(crate) commitment: PCS::Commitment,
    /// The sumcheck proof we use to make sure everything is evaluated at the same point.
    pub(crate) sumcheck_proof: IOPProof<E>,
    /// The claimed evaluations of the polynomials used in the sumcheck proof.
    pub(crate) evaluations: Vec<E>,
}

impl<E: ExtensionField, PCS: PolynomialCommitmentScheme<E>> Debug for SoftmaxProof<E, PCS> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "SoftmaxProof {{logup_proofs: {:?}, sumcheck_proof: {:?}, evaluations: {:?} }}",
            self.logup_proof, self.sumcheck_proof, self.evaluations
        )
    }
}

impl<E: ExtensionField, PCS: PolynomialCommitmentScheme<E>> SoftmaxProof<E, PCS> {
    pub(crate) fn get_lookup_data(&self) -> (Vec<E>, Vec<E>) {
        self.logup_proof.fractional_outputs()
    }
    pub(crate) fn write_commitment<T: Transcript<E>>(&self, transcript: &mut T) -> Result<()> {
        PCS::write_commitment(&self.commitment, transcript).map_err(|e| anyhow!("{e:?}"))
    }
}

/// This the data needed to perform quantised [`Softmax`] evalaution and show that the result is within an acceptable error bound.
struct SoftmaxErrorData {
    /// This is the input scale factor for the exp table, if `r` is the floating point value then the quantised version will use `r * input_sf` as the input to the exp table.
    input_sf: f32,
    /// This is the output scale factor for the exp table, if `e` is the output of the exp table then the floating point version will use `e / output_sf` as the dequantised value.
    output_sf: f32,
    /// The relative error bound on the normalisation sum, the result of summing along the normalisation dimension should be within `(output_sf * relative_error).abs()` of `output_sf`.
    relative_error: f32,
    /// The bit size of the exp table
    table_bit_size: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
/// Stores the shift tensor computed during inference.
pub struct SoftmaxData {
    /// This is the tensor of normalisation shifts to apply in quantised evaluation.
    shift_tensor: Tensor<Element>,
}

impl<N: Number> Softmax<N> {
    pub fn new(context_length: usize) -> Self {
        Softmax {
            scalar: N::unit(),
            max_size: context_length,
            quant_info: None,
        }
    }

    pub fn new_with_scale(scale: N, max_context_size: usize) -> Softmax<N> {
        Softmax {
            scalar: scale,
            max_size: max_context_size,
            quant_info: None,
        }
    }
    /// Method to quantise the [`Softmax`] operation, this takes in the input scaling factor and the intermediate bit size.
    /// The returned [`Softmax`] will have the [`QuantisedSoftmaxData`] set.
    pub fn quantise(
        &self,
        input_scaling: ScalingFactor,
        intermediate_bit_size: usize,
    ) -> Result<Softmax<Element>> {
        // We work out the input scale factor required for the table
        // The error in normalisation arising from the input scale factor is given by (1.0 / (2.0 * input_scale_factor * input_scale_factor * temp)).exp() - 1.0
        // Hence if we wish to have this error contribution be less than `epsilon` we calculate the required input scale factor as
        // `(1.0 /(2.0 * (epsilon + 1.0).ln() * temp)).sqrt() = input_scale_factor`

        // For now we fix epsilon as 0,005f32.
        let SoftmaxErrorData {
            input_sf,
            output_sf,
            relative_error,
            table_bit_size,
        } = self.calc_scale_factors_and_error_based_on_context_size(input_scaling);

        let input_scale_factor = input_scaling.scale();

        let temperature = self.scalar.to_f32()?;

        // This is the multiplier we will use to rescale the input before it is passed to the exp table.
        // It is given by input_scale_factor * temperature * input_sf, where input_sf is the scale factor we calculated above.
        // If we let `r` denote the real value used in the Softmax, then we have `r = input_scale_factor * q1` where `q1` is the quantised input value.
        // Since during Softmax we calculate `(r * temperature).exp()` we have that the quantised exp input is given by `input_sf * temperature * q1`.
        // However we may have multiple Softmax steps within a Model, and we don't want a different lookup table for each of them so we decide on a common scaling factor for the input
        // to exp tables. So we have `q2 / input_sf = r * temperature` where `q2` is the value passed to the exp table. Hence `q2 = input_sf * temperature * input_scale_factor * q1`.
        let rescaling_mult = input_scale_factor * temperature * input_sf;

        // Now we need to convert this rescaling multiplier into a fixed point multiplier and a right shift.
        let log_m = rescaling_mult.log2();
        // This is the right shift
        let int_part = log_m.trunc().abs() as usize;
        // This is used to calculate the fixed point multiplier
        let float_part = log_m.fract();

        let epsilon = 2.0f32.powf(float_part);

        let fp_scale = FIXED_POINT_SCALE;
        let fixed_point_multiplier =
            (epsilon * (1u64 << FIXED_POINT_SCALE) as f32).round_ties_even() as Element;

        // Assertion to check that we can perform requantisation, we need intermediate_bit_size + fp_scale <= 63
        assert!(
            intermediate_bit_size + fp_scale <= 63,
            "intermediate bit size: {intermediate_bit_size}, fp scale: {fp_scale}, int part: {int_part}",
        );
        // Now we can create the ExpTable
        let lut = ExpTable::new(input_sf, output_sf, table_bit_size);

        let quant_info = QuantisedSoftmaxData {
            right_shift: int_part + FIXED_POINT_SCALE,
            fixed_point_multiplier,
            fp_scale,
            multiplier: rescaling_mult,
            intermediate_bit_size,
            lut,
            error_bound: relative_error,
            input_scaling_factor: input_scaling,
            temperature: 1.0 / temperature,
        };

        // Return the quantised `Softmax` operator
        Ok(Softmax::<Element> {
            scalar: 1,
            max_size: self.max_size,
            quant_info: Some(quant_info),
        })
    }
    /// Getter for the [`QuantisedSoftmaxData`] if it exists
    pub(crate) fn quant_info(&self) -> Option<&QuantisedSoftmaxData> {
        self.quant_info.as_ref()
    }
    /// Method to set the temperature for the Softmax operation
    pub fn with_scale(self, scale: N) -> Self {
        Self {
            scalar: scale,
            ..self
        }
    }

    /// Method to calculate the scale factors, error and required size for the [`ExpTable`] in order to prform quantised [`Softmax`].
    /// We use the fact that we wish to achieve a small L1 error (< 0.01) on the normalised sum. Each individual value looked up will
    /// have relative error (1.0 / (2.0 * input_sf)).exp() - 1.0, and absolute error (1.0 / (2.0 * output_sf)). Then when we sum along the normalised row
    /// this will give the relative error of the sum as:
    ///
    ///  `rel_error_sum = (1.0 / (2.0 * input_sf)).exp() - 1.0 + n * (1.0 / (2.0 * output_sf))`
    ///
    /// Here `n` is the maximum context size.
    fn calc_scale_factors_and_error_based_on_context_size(
        &self,
        input_scaling: ScalingFactor,
    ) -> SoftmaxErrorData {
        let max_context_size = self.max_size as f32;

        // This works out the maximum possible output bitsize based on the fact that the following layer will have
        // to do a matrix multiplication of size `max_context_size` and that the Primefield we are using allows for 63 bits.
        let max_poss_out_sf_log =
            63 - ceil_log2(self.max_size) - FIXED_POINT_SCALE - *quantization::BIT_LEN;
        // Then ideally we would like to strike a balance between the error table size and the exp table size. To achieve this
        // we would like max_context_size / (2.0 * output_sf) < 0.005 (0.005 is picked so that the exp table doesn't become too large)

        let mut log_output_sf = ceil_log2(self.max_size);
        let limit = max_context_size / 0.01;
        while log_output_sf < max_poss_out_sf_log && limit > (1 << log_output_sf) as f32 {
            log_output_sf += 1;
        }

        let output_sf = 1 << log_output_sf;
        // Then from this we can work out the rounding error incurred on each output of the exp table
        let table_rounding = 1.0 / (2.0 * output_sf as f32);

        // Now we need the lower bound on the exp table to be such that (-table_lower_bound).exp() <= table_rounding => -table_lower_bound <= table_rounding.ln()
        let table_rounding_ln = table_rounding.ln();
        let table_lower_bound = -table_rounding_ln;

        // Now we work out the pair (last_table_value, input_sf) such that the relative error given by (1.0 / (2.0 * input_sf)).exp() - 1.0 <= 0.01 - max_context_size * table_rounding
        // So we iterate through powers of two calculating temp_sf = table_lower_bound / 2^i until 1.0 / (2.0 * (1.0 + 0.01 - max_context_size * table_rounding).ln()) <= temp_sf
        let mut initial_power = *quantization::BIT_LEN;
        let mut input_sf = (1 << initial_power) as f32 / table_lower_bound;

        let limit = 1.0f32 / (2.0 * (1.01 - max_context_size * table_rounding).ln());

        // Loops through and gives us the largest input_sf we can have while keeping the error bound
        // reasonable
        loop {
            let tmp_power = initial_power + 1;
            let tmp_input_sf = (1 << tmp_power) as f32 / table_lower_bound;
            if limit > tmp_input_sf {
                initial_power += 1;
                input_sf = tmp_input_sf;
            } else {
                break;
            }
        }
        // The case that may cause issues seems to always be when all the values on a row are the same, so we quickly check here what the error for that case would be
        let temperature = self.scalar.to_f32().unwrap_or(1.0);
        let all_same_shift = (-(max_context_size.ln()) / (input_scaling.scale() * temperature))
            .round_ties_even() as Element;
        let rescaling_mult = input_scaling.scale() * temperature * input_sf;

        // Now we need to convert this rescaling multiplier into a fixed point multiplier and a right shift.
        let log_m = rescaling_mult.log2();
        // This is the right shift
        let int_part = log_m.trunc().abs() as usize;
        // This is used to calculate the fixed point multiplier
        let float_part = log_m.fract();

        let epsilon = 2.0f32.powf(float_part);

        let fixed_point_multiplier =
            (epsilon * (1u64 << FIXED_POINT_SCALE) as f32).round_ties_even() as Element;

        let rescaling_error = ((input_scaling.scale() / 2.0f32) * fixed_point_multiplier as f32
            + 2.0f32.powf((int_part + FIXED_POINT_SCALE - 1) as f32))
            / (2.0f32.powf((int_part + FIXED_POINT_SCALE) as f32));

        let rescaled_shift =
            ((all_same_shift as f32) * rescaling_mult).round_ties_even() as Element;
        let exp_out = ((rescaled_shift as f32 / input_sf).exp() * output_sf as f32)
            .round_ties_even() as Element;
        let row_sum = (exp_out as f32) * max_context_size;
        let expected_sum = output_sf as f32;
        let diff = (row_sum - expected_sum).abs();
        let relative_sum_error = diff / expected_sum;

        // Now we can calculate the relative error
        let input_error_factor = input_sf.min(1.0 / input_scaling.scale());
        let input_error_factor = input_error_factor.min(1.0 / rescaling_error);
        let first_part = (1.0 / (2.0 * input_error_factor)).exp() - 1.0;
        let table_max_value: Element = 1 + (-1 << initial_power);
        let val_too_large_error = (table_max_value as f32 / input_sf).exp();
        let other_error_part = table_rounding.max(val_too_large_error);

        let other_relative_error = first_part + max_context_size * other_error_part;

        let relative_error = relative_sum_error.max(other_relative_error);
        SoftmaxErrorData {
            input_sf,
            output_sf: output_sf as f32,
            relative_error,
            table_bit_size: initial_power,
        }
    }
}

impl Softmax<Element> {
    /// Method that given a quantised input [`Tensor`] calculates the `shift` we apply along each dim and returns the result as the `bias` field of
    /// as [`AttentionMask`].
    pub(crate) fn calculate_shift_data(
        &self,
        binput: &BTensor<Backend, 2, BInt>,
        shift_shape: Shape,
    ) -> Result<(Tensor<Element>, BTensor<Backend, 2, BInt>)> {
        let QuantisedSoftmaxData {
            input_scaling_factor,
            temperature,
            ..
        } = self.quant_info().ok_or(anyhow!("Attempted to calculate shift data for quantised Softmax with no QuantisedSoftmaxData present"))?;
        // Unwrap is safe here because previous line would have errored if quant_info was None
        let negative_infinity = self.quant_info().unwrap().quantised_negative_infinity();

        let scalar = input_scaling_factor.scale() / temperature;
        let binput_mask = binput.clone().equal_elem(negative_infinity);

        let dim_maxes = binput.clone().max_dim(1);
        let log_sum_exp = (binput.clone() - dim_maxes.clone())
            .float()
            .mul_scalar(scalar)
            .mask_fill(binput_mask.clone(), f32::NEG_INFINITY)
            .exp()
            .sum_dim(1)
            .log();

        let quantising_scalar = temperature / input_scaling_factor.scale();
        let shift_btensor = log_sum_exp.mul_scalar(-quantising_scalar).round().int() - dim_maxes;
        let shift_data: Vec<Element> = shift_btensor
            .to_data()
            .into_vec()
            .map_err(|e| anyhow!("Could not convert burn Softmax shift data to Element: {e:?}"))?;

        let shift_tensor = Tensor::<Element>::new(shift_shape, shift_data);

        Ok((shift_tensor, shift_btensor))
    }
}

impl Evaluate<f32> for Softmax<f32> {
    fn evaluate<E: ExtensionField>(
        &self,
        inputs: &[&Tensor<f32>],
        _unpadded_input_shapes: &[Shape],
    ) -> anyhow::Result<LayerOut<f32, E>> {
        ensure!(
            inputs.len() == 1,
            "softmax expects exactly one input tensor currently"
        );
        let input = inputs[0];

        // Convert to a 2D burn tensor, rescale and apply softmax.
        let b_input = input.clone().flatten(0..input.rank() - 1).to_btensor::<2>() * self.scalar;
        let probabilities = softmax(b_input, 1);

        // Extract the output data
        let output_data: Vec<f32> = probabilities
            .to_data()
            .into_vec()
            .map_err(|e| anyhow!("Could not convert burn Softmax output to f32: {e:?}"))?;

        let output_tensor = Tensor::new(input.shape().clone(), output_data);
        Ok(LayerOut::from_vec(vec![output_tensor]))
    }
}

impl<N: Number> OpInfo for Softmax<N> {
    fn output_shapes(
        &self,
        input_shapes: &[Shape],
        _padding_mode: crate::padding::PaddingMode,
    ) -> Vec<Shape> {
        input_shapes.to_vec()
    }

    fn num_outputs(&self, num_inputs: usize) -> usize {
        num_inputs
    }

    fn describe(&self) -> String {
        "Softmax".to_string()
    }

    fn is_provable(&self) -> bool {
        true
    }
}

impl Evaluate<Element> for Softmax<Element> {
    fn evaluate<E: ExtensionField>(
        &self,
        inputs: &[&Tensor<Element>],
        unpadded_input_shapes: &[Shape],
    ) -> Result<LayerOut<Element, E>> {
        // First we check that we have some quantisation info.
        ensure!(
            self.quant_info.is_some(),
            "Could not evaluate quantised softmax because the operation has not been quantised"
        );
        // Check that we only have one input
        ensure!(
            inputs.len() == 1,
            "Expected a single input to quantised softmax, got: {}",
            inputs.len()
        );

        // Since we have checked that quant info exists this unwrap is safe
        let QuantisedSoftmaxData {
            lut,
            right_shift,
            fixed_point_multiplier,
            ..
        } = self.quant_info().unwrap();

        // We expect the input tensor to have rank 2 or 3, if it has rank 2 we will treat it as having shape [1, shape[0], shape[1]]
        let input_rank = inputs[0].shape().rank();
        ensure!(
            input_rank == 2 || input_rank == 3,
            "Expected input to quantised softmax to have rank 2 or 3, got: {input_rank}",
        );

        let (input, unpadded_input_shape) = if input_rank == 2 {
            (
                inputs[0].clone().unsqueeze(0),
                unpadded_input_shapes[0].insert(0, 1),
            )
        } else {
            (inputs[0].clone(), unpadded_input_shapes[0].clone())
        };

        let shift_shape = Shape::new(vec![unpadded_input_shape[0], input.shape().dim(1), 1]);

        // We work over 2D chunks (skipping any padding chunks)
        let full_input_size = input.shape().numel();
        let strides = input.shape().strides();
        let rounding: Element = 1 << (*right_shift - 1);

        let data_to_take = strides[0] * unpadded_input_shape[0];
        // Now we flatten the input to 2D and take only the data that corresponds to 2D sub-tensors that don't arise from padding.
        let flat_data = input
            .iter()
            .take(data_to_take)
            .cloned()
            .collect::<Vec<Element>>();
        let b_input = BTensor::<Backend, 2, BInt>::from_data(
            TensorData::new(
                flat_data,
                vec![
                    unpadded_input_shape[0] * input.shape().dim(1),
                    input.shape().dim(2),
                ],
            ),
            &Default::default(),
        );
        let (shift_tensor, shift_btensor) = self.calculate_shift_data(&b_input, shift_shape)?;

        let multiplied_b_input = (b_input + shift_btensor)
            .mul_scalar(*fixed_point_multiplier)
            .add_scalar(rounding)
            .bitwise_right_shift_scalar(*right_shift as Element);

        let multiplied_b_input_data: Vec<Element> =
            multiplied_b_input.into_data().to_vec().map_err(|e| {
                anyhow!("Failed to convert multiplied_b_input to Vec<Element> in Softmax: {e:?}")
            })?;
        let output_data = multiplied_b_input_data
            .into_iter()
            .map(|intermediate| {
                if intermediate <= -(1 << lut.table_bit_size()) {
                    0
                } else {
                    lut.table_output(intermediate)
                }
            })
            .chain(std::iter::repeat(0))
            .take(full_input_size)
            .collect::<Vec<Element>>();

        // Make the output tensor
        let output = if input_rank == 2 {
            Tensor::<Element>::new(inputs[0].shape().clone(), output_data)
        } else {
            Tensor::<Element>::new(input.shape().clone(), output_data)
        };

        Ok(LayerOut {
            outputs: vec![output],
            proving_data: ProvingData::Softmax(SoftmaxData { shift_tensor }),
            tracked_layer_data: None,
        })
    }
}

impl PadOp for Softmax<Element> {}

struct RangeChecks {
    number_of_chunks: usize,
    chunks: Vec<Vec<Element>>,
}

impl RangeChecks {
    fn new(number_of_chunks: usize) -> Self {
        Self {
            number_of_chunks,
            chunks: vec![vec![]; number_of_chunks],
        }
    }

    fn push(&mut self, value: Element) {
        let bit_len_mask: Element = (1 << *quantization::BIT_LEN) - 1;
        (0..self.number_of_chunks).for_each(|j| {
            let shift = j * *quantization::BIT_LEN;
            let chunk_val = (value >> shift) & bit_len_mask;
            self.chunks[j].push(chunk_val);
        });
    }

    fn merge(&mut self, other: RangeChecks) {
        assert_eq!(self.number_of_chunks, other.number_of_chunks);
        let RangeChecks { chunks, .. } = other;
        self.chunks
            .iter_mut()
            .zip(chunks)
            .for_each(|(a, b)| a.extend(b));
    }

    fn count_iterator(&self) -> Vec<Element> {
        self.chunks.concat()
    }
}

struct ExpLookup {
    input: Vec<Element>,
    output: Vec<Element>,
}

impl ExpLookup {
    fn new() -> Self {
        Self {
            input: Vec::<Element>::new(),
            output: Vec::<Element>::new(),
        }
    }

    fn push(&mut self, input: Element, output: Element) {
        self.input.push(input);
        self.output.push(output);
    }

    fn merge(&mut self, other: ExpLookup) {
        let ExpLookup { input, output } = other;
        self.input.extend(input);
        self.output.extend(output);
    }

    fn count_iterator(&self) -> Vec<Element> {
        self.input
            .iter()
            .zip(self.output.iter())
            .map(|(a, b)| a + COLUMN_SEPARATOR * b)
            .collect::<Vec<Element>>()
    }
}

struct ZeroChecks {
    number_of_chunks: usize,
    input_chunks: Vec<Vec<Element>>,
    output_chunks: Vec<Vec<Element>>,
}

impl ZeroChecks {
    fn new(number_of_chunks: usize) -> Self {
        Self {
            number_of_chunks,
            input_chunks: vec![vec![]; number_of_chunks],
            output_chunks: vec![vec![]; number_of_chunks],
        }
    }

    fn push(&mut self, input: Element) {
        let bit_len_mask: Element = (1 << *quantization::BIT_LEN) - 1;
        (0..self.number_of_chunks).for_each(|j| {
            let shift = j * *quantization::BIT_LEN;
            let in_val = (input >> shift) & bit_len_mask;

            self.input_chunks[j].push(in_val);

            if in_val != 0 {
                self.output_chunks[j].push(0);
            } else {
                self.output_chunks[j].push(1);
            }
        });
    }

    fn merge(&mut self, other: ZeroChecks) {
        assert_eq!(self.number_of_chunks, other.number_of_chunks);
        let ZeroChecks {
            input_chunks,
            output_chunks,
            ..
        } = other;
        self.input_chunks
            .iter_mut()
            .zip(input_chunks)
            .for_each(|(a, b)| a.extend(b));
        self.output_chunks
            .iter_mut()
            .zip(output_chunks)
            .for_each(|(a, b)| a.extend(b));
    }

    fn count_iterator(&self) -> Vec<Element> {
        self.input_chunks
            .iter()
            .zip(self.output_chunks.iter())
            .flat_map(|(input_chunk, output_chunk)| {
                input_chunk
                    .iter()
                    .zip(output_chunk.iter())
                    .map(|(a, b)| a + COLUMN_SEPARATOR * b)
                    .collect::<Vec<Element>>()
            })
            .collect::<Vec<Element>>()
    }
}

impl Softmax<Element> {
    #[allow(clippy::type_complexity)]
    pub(crate) fn prove_step<
        E: ExtensionField,
        PCS: PolynomialCommitmentScheme<E> + Send + Sync,
        T: transcript::Transcript<E>,
    >(
        &self,
        node_id: NodeId,
        last_claims: Vec<&Claim<E>>,
        ctx: &SoftmaxCtx<E>,
        softmax_data: &SoftmaxData,
        input_shape: &Shape,
        prover: &mut crate::Prover<E, T, PCS>,
    ) -> Result<(Vec<Claim<E>>, SoftmaxProof<E, PCS>)>
    where
        PCS::CommitmentWithWitness: Serialize + DeserializeOwned + Send + Sync,
        PCS::ProverParam: Send + Sync,
    {
        // Check number of claims
        ensure!(
            last_claims.len() == 1,
            "Softmax only produces one output claim but got: {}",
            last_claims.len()
        );
        let last_claim = last_claims[0];

        let shift_shape = softmax_data.shift_tensor.shape();
        let final_dim_size = input_shape
            .last()
            .ok_or(anyhow!("Shifted input has no shape"))?
            .next_power_of_two();
        let first_dim = shift_shape[0];
        // Retrieve all the witness data
        let number_of_range_checks = ctx.quant_info.number_of_range_checks();
        let number_of_zero_chunks = ctx.quant_info.number_of_zero_chunks();
        let layer_commitment = prover.lookup_witness(node_id)?;
        // Prepare the lookup inputs from the layer commitment
        let logup_inputs = ctx.lookup_ctx.create_logup_inputs_softmax::<PCS, E>(
            layer_commitment,
            &prover.challenge_storage,
            final_dim_size,
            number_of_range_checks,
            number_of_zero_chunks,
            first_dim,
        )?;
        let layer_polys = PCS::get_arc_mle_witness_from_commitment(layer_commitment);
        let commitment = PCS::get_pure_commitment(layer_commitment);
        // Run the logup proving
        let logup_batch_proof = batch_multiple_sizes_prove(&logup_inputs, prover.transcript)?;

        let logup_point = &logup_batch_proof.output_claims()[0].point;
        // We need to know how many variables it takes to represent the normalisation dimension
        let dim_vars = ceil_log2(final_dim_size);
        let two = E::from_canonical_u64(2u64);
        let two_inv = two.inverse();

        // The error lookup is performed over the output summed on the final dimension so we need to extend the point used with correct number
        // of 2^-1 entries
        let full_error_point = std::iter::repeat_n(two_inv, dim_vars)
            .chain(logup_point.iter().skip(dim_vars).copied())
            .collect::<Vec<E>>();
        // Here we split the last claim point up according to input shape
        let split = input_shape.split_point(last_claim.point())?;
        // The batch challenge point is the first part of the split, the rest are the last claim points
        let batch_chal_point = split[0];
        let lc_eq_point = split
            .iter()
            .skip(1)
            .rev()
            .flat_map(|&v| v)
            .copied()
            .collect::<Vec<E>>();

        // Make all the eq polys
        let error_eq = compute_betas_eval(&full_error_point).into_mle();
        let logup_eq = compute_betas_eval(logup_point).into_mle();
        let last_claim_eq = compute_betas_eval(&lc_eq_point).into_mle();

        // We split the layer polys up here, all polys related to decomposition of the input come first
        // and there will be first_dim * (number_of_range_checks + 1 + number_of_zero_chunks) in total.
        // After we have split these of the next first_dim * (1 + number_of_zero_chunks) are used to calculate the output.
        let number_input_polys = first_dim * (number_of_range_checks + 1 + number_of_zero_chunks);
        let number_output_polys = first_dim * (number_of_zero_chunks + 1);
        let (_, rest) = layer_polys.split_at(number_input_polys);
        let (sumcheck_polys, shift_polys) = rest.split_at(number_output_polys);

        // Transform the polys into Either::Left so they can be passed to the VirtualPolynomialsBuilder
        let either_mles = [&last_claim_eq, &error_eq, &logup_eq]
            .into_iter()
            .map(Either::Left)
            .chain(sumcheck_polys.iter().map(|p| Either::Left(p.as_ref())))
            .collect::<Vec<Either<_, _>>>();

        // Squeeze a batching challenge from the transcript, powers of these challenges will be used to
        // link the MLEs used in the lookup to this sumcheck
        let alphas = (0..first_dim)
            .map(|_| {
                prover
                    .transcript
                    .sample_and_append_challenge(b"batching_challenge")
                    .elements
            })
            .collect::<Vec<E>>();

        let batching_evals = compute_betas_eval(batch_chal_point);
        let challenges = batching_evals
            .iter()
            .zip(alphas)
            .flat_map(|(&a, b)| [a, b])
            .collect::<Vec<E>>();
        // Make the VirtualPolynomials and run the sumcheck
        let num_vars = logup_point.len();
        let num_threads = optimal_sumcheck_threads(num_vars);
        let expr_builder =
            VirtualPolynomialsBuilder::<E>::new_with_mles(num_threads, num_vars, either_mles);
        let virtual_poly =
            expr_builder.to_virtual_polys(&ctx.sumcheck_expression[..first_dim], &challenges);
        let (sumcheck_proof, state) = IOPProverState::<E>::prove(virtual_poly, prover.transcript);
        let sumcheck_point = state
            .challenges
            .iter()
            .map(|c| c.elements)
            .collect::<Vec<E>>();
        let all_evals = state.get_mle_flatten_final_evaluations();

        // We have all the range claims, then the exp claims, then zero claims, then error claims
        let logup_claims = logup_batch_proof.output_claims();
        let (range_claims, rest) = logup_claims.split_at(first_dim * number_of_range_checks);
        let (exp_claims, rest) = rest.split_at(2 * first_dim);
        let (zero_claims, _) = rest.split_at(first_dim * 2 * number_of_zero_chunks);

        // We evaluate the shift polys at the logup point (skipping the variables relating to the normalisation dimension entries)
        let shift_eval_point = logup_point[dim_vars..].to_vec();
        let shift_evals = shift_polys
            .iter()
            .map(|p| p.evaluate(&shift_eval_point))
            .collect::<Vec<E>>();

        // These constants are used to recombine the chunks from the lookups
        let base_multiplier = E::from_canonical_u64(1u64 << *quantization::BIT_LEN);
        let right_shift_field = E::from_canonical_u64(1u64 << ctx.quant_info.right_shift);
        let rounding = E::from_canonical_u64(1u64 << (ctx.quant_info.right_shift - 1));
        let fpm_field: E = ctx.quant_info.fixed_point_multiplier.to_field();
        let fpm_inv = fpm_field.inverse();
        let zero_offset = E::from_canonical_u64(
            1 << (ctx.quant_info.right_shift + ctx.quant_info.lut.table_bit_size()),
        );

        // Combine the range claims for each chunk
        let (low_parts, stacked_range_evals): (Vec<E>, Vec<Vec<E>>) = range_claims
            .chunks(number_of_range_checks)
            .map(|chunk| {
                let range_evals = chunk.iter().map(|c| c.evaluation()).collect::<Vec<E>>();
                let input_part = range_evals
                    .iter()
                    .fold((E::ZERO, E::ONE), |(acc, pow_two), &b| {
                        (acc + pow_two * b, pow_two * base_multiplier)
                    })
                    .0;
                (input_part, range_evals)
            })
            .unzip();
        // Combine the exp claims
        let (exp_parts, stacked_exp_evals): (Vec<E>, Vec<E>) = exp_claims
            .iter()
            .step_by(2)
            .map(|c| (c.evaluation() * right_shift_field, c.evaluation()))
            .unzip();
        // Combine the zero claims
        let (high_parts, stacked_high_evals): (Vec<E>, Vec<Vec<E>>) = zero_claims
            .chunks(2 * number_of_zero_chunks)
            .map(|chunk| {
                let high_evals = chunk
                    .iter()
                    .step_by(2)
                    .map(|c| c.evaluation())
                    .collect::<Vec<E>>();
                let input_part = high_evals
                    .iter()
                    .fold((E::ZERO, zero_offset), |(acc, pow_two), &b| {
                        (acc + pow_two * b, pow_two * base_multiplier)
                    })
                    .0;
                (input_part, high_evals)
            })
            .unzip();

        // Calculate the input evaluation
        let input_eval = izip!(
            low_parts,
            exp_parts,
            high_parts,
            shift_evals.iter(),
            batching_evals
        )
        .map(|(l, e, h, &shift, batch)| ((l + e - h - rounding) * fpm_inv - shift) * batch)
        .sum::<E>();
        // The first commitment is the range checks, then the exp inputs, then the zero inputs
        let first_commit_evals = izip!(stacked_range_evals, stacked_exp_evals, stacked_high_evals)
            .flat_map(|(mut rs, e, zs)| {
                rs.push(e);
                rs.extend(zs);
                rs
            })
            .collect::<Vec<E>>();

        let first_commit_point = logup_point.to_vec();

        // The second commitment is the exp output and the zero outputs
        let second_commit_evals = all_evals[3..].to_vec();
        let second_commit_point = sumcheck_point.clone();
        // Combine them all in the correct order and add them to the claim prover
        let layer_claims = vec![
            (first_commit_point, first_commit_evals),
            (second_commit_point, second_commit_evals),
            (shift_eval_point, shift_evals.clone()),
        ];
        prover.add_witness_claim(node_id, layer_claims);

        let input_claim = Claim::<E>::new(
            logup_point
                .iter()
                .chain(batch_chal_point.iter())
                .copied()
                .collect::<Vec<E>>(),
            input_eval,
        );

        let softmax_proof = SoftmaxProof {
            logup_proof: logup_batch_proof,
            commitment,
            sumcheck_proof,
            evaluations: [&all_evals[3..], shift_evals.as_slice()].concat(),
        };

        Ok((vec![input_claim], softmax_proof))
    }

    pub(crate) fn lookup_witness<E: ExtensionField, PCS: PolynomialCommitmentScheme<E>>(
        &self,
        id: NodeId,
        ctx: &ProverContext<E, PCS>,
        input: &Tensor<Element>,
        output: &Tensor<Element>,
        softmax_data: &SoftmaxData,
    ) -> Result<LookupWitnessGen<E, PCS>>
    where
        PCS::CommitmentWithWitness: Serialize + DeserializeOwned + Send + Sync,
        PCS::ProverParam: Send + Sync,
    {
        // Get the data generated during quantised evaluation
        let SoftmaxData { shift_tensor } = softmax_data;

        // We need to work out how many chunks to split the normalisation into to be range checked.
        let quant_info = self.quant_info().ok_or(anyhow!(
            "Could not prove Softmax because it had no quantisation data"
        ))?;
        let QuantisedSoftmaxData {
            right_shift,
            fixed_point_multiplier,
            error_bound,
            lut,
            ..
        } = quant_info;
        let allowable_error = (*error_bound * lut.output_sf()).round() as Element;

        // Now we construct the polynomials used in the lookups
        // To do this we need the size of the last dimension
        let final_dim_size = *output
            .shape()
            .last()
            .ok_or(anyhow!("Softmax output tensor did not have a shape"))?;

        let input_shape = input.shape();
        let input_rank = input_shape.rank();
        // We need to chunk the input into its 2D sub-tensors, we will ignore any 2D sub-tensors that arise from padding
        let chunk_size = if input_rank == 2 {
            input_shape.numel()
        } else {
            let strides = input_shape.strides();
            strides[0]
        };

        let shift_shape = shift_tensor.shape();
        let shift_chunk_size = shift_shape.strides()[0];
        // These are the sums of the rows after Softmax, we check that these are all within the allowable error of quantised 1.0.
        let normalisation_lookups = output
            .get_data()
            .chunks(chunk_size)
            .take(shift_shape[0])
            .map(|outer_chunk| {
                outer_chunk
                    .chunks(final_dim_size)
                    .map(|chunk| chunk.iter().sum::<Element>())
                    .collect::<Vec<Element>>()
            })
            .collect::<Vec<Vec<Element>>>();

        // This is the rounding constant used during the fixed point multiplication and right shift
        let rounding: Element = 1 << (*right_shift - 1);
        // This is the bit mask used to extract the bits used in the lookup table for exp after performing the
        // fixed point multiplication and right shift
        let exp_bit_mask = lut.full_table_size() - 1;

        let (chunked_range_checks, chunked_exp_lookup, chunked_zero_checks) = input
            .get_data()
            .par_chunks(chunk_size)
            .zip(shift_tensor.get_data().par_chunks(shift_chunk_size))
            .fold(
                || (vec![], vec![], vec![]),
                |(mut range, mut exp, mut zero), (outer_input_chunk, outer_shift_chunk)| {
                    // For each outer chunk we have to decompose it as we did during inference
                    let (chunk_range_checks, chunk_exp_lookup, chunk_zero_checks) =
                        outer_input_chunk
                            .chunks(final_dim_size)
                            .zip(outer_shift_chunk)
                            .fold(
                                (
                                    RangeChecks::new(quant_info.number_of_range_checks()),
                                    ExpLookup::new(),
                                    ZeroChecks::new(quant_info.number_of_zero_chunks()),
                                ),
                                |(mut outer_range, mut outer_exp, mut outer_zero),
                                 (input_chunk, shift)| {
                                    let (inner_range, inner_exp, inner_zero) =
                                        input_chunk.iter().fold(
                                            (
                                                RangeChecks::new(
                                                    quant_info.number_of_range_checks(),
                                                ),
                                                ExpLookup::new(),
                                                ZeroChecks::new(quant_info.number_of_zero_chunks()),
                                            ),
                                            |(
                                                mut range_checks,
                                                mut exp_lookups,
                                                mut zero_checks,
                                            ),
                                             elem| {
                                                // Add the normalisation shift
                                                let shifted = elem + shift;
                                                // Perform fixed point multiplication and add the rounding constant
                                                let scaled =
                                                    shifted * fixed_point_multiplier + rounding;

                                                let intermediate = scaled >> *right_shift;
                                                // Extract the low bits to be range checked
                                                let low = scaled - (intermediate << *right_shift);
                                                // Extract the bits to be used in the exp lookup
                                                let exp_in = intermediate.abs() & exp_bit_mask;
                                                // If any of the remaining high bits are non-zero we will be out of range for the exp table
                                                // so we check these are zero
                                                let high =
                                                    intermediate.abs() >> lut.table_bit_size();

                                                range_checks.push(low);
                                                exp_lookups
                                                    .push(-exp_in, lut.table_output(-exp_in));
                                                zero_checks.push(high);
                                                (range_checks, exp_lookups, zero_checks)
                                            },
                                        );
                                    outer_range.merge(inner_range);
                                    outer_exp.merge(inner_exp);
                                    outer_zero.merge(inner_zero);
                                    (outer_range, outer_exp, outer_zero)
                                },
                            );
                    range.push(chunk_range_checks);
                    exp.push(chunk_exp_lookup);
                    zero.push(chunk_zero_checks);
                    (range, exp, zero)
                },
            )
            .reduce(
                || (vec![], vec![], vec![]),
                |(mut range_acc, mut exp_acc, mut zero_acc), (range, exp, zero)| {
                    range_acc.extend(range);
                    exp_acc.extend(exp);
                    zero_acc.extend(zero);
                    (range_acc, exp_acc, zero_acc)
                },
            );

        let range_elements_count =
            count_elements(chunked_range_checks.iter().flat_map(|c| c.count_iterator()));
        let exp_elements_count =
            count_elements(chunked_exp_lookup.iter().flat_map(|c| c.count_iterator()));
        let zero_table_elements_count =
            count_elements(chunked_zero_checks.iter().flat_map(|c| c.count_iterator()));

        // We create 3 separate RMMs here, the first corresponds to the lookup inputs, for each chunk the polys are in order
        // range_checks, exp_in, zero_checks_in
        // The second RMM is to do with lookup outputs and the ordering is
        // exp_out, zero_chunks_out
        // The third and final RMM is the shift for each chunk

        let (rmm1_polys, rmm2_polys) = izip!(
            chunked_range_checks,
            chunked_exp_lookup,
            chunked_zero_checks
        )
        .fold(
            (vec![], vec![]),
            |(mut rmm1_acc, mut rmm2_acc), (range_checks, exp_lookup, zero_checks)| {
                let RangeChecks { chunks, .. } = range_checks;
                let ExpLookup { input, output } = exp_lookup;
                let ZeroChecks {
                    input_chunks,
                    output_chunks,
                    ..
                } = zero_checks;
                rmm1_acc.extend(
                    chunks
                        .into_iter()
                        .chain(std::iter::once(input))
                        .chain(input_chunks),
                );
                rmm2_acc.extend(std::iter::once(output).chain(output_chunks));
                (rmm1_acc, rmm2_acc)
            },
        );

        // The width of the first rmm is the number of chunks we decomopose into (given by `quant_info.number_of_range_checks() + 1 + quant_info.number_of_zero_chunks()`)
        // multiplied by the number of 2D tensors we have (given by shift_shape[0])
        let width_one = shift_shape[0]
            * (quant_info.number_of_range_checks() + 1 + quant_info.number_of_zero_chunks());
        let transposed_one = transpose(rmm1_polys);
        let rmm1 = RowMajorMatrix::<E::BaseField>::new_by_inner_matrix(
            ceno_p3::matrix::dense::DenseMatrix::new(
                to_base::<E, _>(transposed_one.into_iter().flatten()),
                width_one,
            ),
            witness::InstancePaddingStrategy::Default,
        );
        // The width of the second rmm is the number of output polys we have (given by 1 + quant_info.number_of_zero_chunks())
        // multiplied by the number of 2D tensors we have (given by shift_shape[0])
        let width_two = shift_shape[0] * (1 + quant_info.number_of_zero_chunks());
        let transposed_two = transpose(rmm2_polys);
        let rmm2 = RowMajorMatrix::<E::BaseField>::new_by_inner_matrix(
            ceno_p3::matrix::dense::DenseMatrix::new(
                to_base::<E, _>(transposed_two.into_iter().flatten()),
                width_two,
            ),
            witness::InstancePaddingStrategy::Default,
        );
        // The final rmm is the shift values, its width is just shift_shape[0]
        let shift_evals = shift_tensor
            .get_data()
            .chunks(shift_chunk_size)
            .map(|chunk| chunk.to_vec())
            .collect::<Vec<_>>();
        let shift_transposed = transpose(shift_evals);
        let rmm3 = RowMajorMatrix::<E::BaseField>::new_by_inner_matrix(
            ceno_p3::matrix::dense::DenseMatrix::new(
                to_base::<E, _>(shift_transposed.into_iter().flatten()),
                shift_shape[0],
            ),
            witness::InstancePaddingStrategy::Default,
        );

        let layer_commit = ctx.commitment_ctx.batch_commit(vec![rmm1, rmm2, rmm3])?;

        let mut gen_w = LookupWitnessGen::<E, PCS>::default();

        // Add the looked up values to the generator so we can make multiplicity polys later
        gen_w.insert_element_count(TableType::Range, range_elements_count);

        // Need to recreate the parameters for the Softmax table
        gen_w.insert_element_count(TableType::ExpTable(*lut), exp_elements_count);

        let quant_one = lut.output_sf() as Element;
        gen_w.insert_element_count(
            TableType::ErrorTable(quant_one, allowable_error),
            count_elements(normalisation_lookups.into_iter().flatten()),
        );

        gen_w.insert_element_count(TableType::ZeroTable, zero_table_elements_count);

        gen_w.insert_logup_witness(id, layer_commit);
        Ok(gen_w)
    }
}

impl<E, PCS> ProvableOp<E, PCS> for Softmax<Element>
where
    E: ExtensionField + Serialize + DeserializeOwned,
    E::BaseField: Serialize + DeserializeOwned,
    PCS: PolynomialCommitmentScheme<E> + Send + Sync,
    PCS::CommitmentWithWitness: Serialize + DeserializeOwned + Send + Sync,
    PCS::ProverParam: Send + Sync,
{
    type Ctx = SoftmaxCtx<E>;

    fn prove<T: transcript::Transcript<E>>(
        &self,
        node_id: NodeId,
        ctx: &Self::Ctx,
        last_claims: Vec<&Claim<E>>,
        step_data: &StepData<E, E>,
        prover: &mut crate::Prover<E, T, PCS>,
        _store: &mut GenStore,
    ) -> Result<Vec<Claim<E>>> {
        let softmax_data = step_data.node_outputs.try_softmax_data().ok_or(anyhow!(
            "Softmax LayerOut didn't have any ProvingData::Softmax"
        ))?;

        let input_shape = step_data.node_inputs[0].shape();
        let (claims, proof) =
            self.prove_step(node_id, last_claims, ctx, softmax_data, input_shape, prover)?;
        // Add the proof to the proof list
        prover.push_proof(node_id, LayerProof::<E, PCS>::Softmax(proof));

        Ok(claims)
    }

    fn gen_lookup_witness(
        &self,
        id: NodeId,
        ctx: &ProverContext<E, PCS>,
        step_data: &StepData<Element, E>,
        store: &mut GenStore,
    ) -> Result<LookupWitnessGen<E, PCS>> {
        let input_tensors = step_data.input_tensors(store)?;
        let output_tensors = step_data.output_tensors(store)?;

        ensure!(
            step_data.node_inputs.len() == 1,
            "Found more than 1 input in inference step of Softmax layer"
        );
        ensure!(
            output_tensors.len() == 1,
            "Found more than 1 output in inference step of Softmax layer"
        );
        let softmax_data = step_data.node_outputs.try_softmax_data().ok_or(anyhow!(
            "Softmax data not found in inference step for Softmax layer"
        ))?;

        self.lookup_witness(id, ctx, &input_tensors[0], &output_tensors[0], softmax_data)
    }
}

impl QuantizeOp for Softmax<f32> {
    type QuantizedOp = Softmax<Element>;

    fn quantize_op<S: ScalingStrategy>(
        self,
        _data: &S::AuxData,
        node_id: NodeId,
        input_scaling: &[ScalingFactor],
    ) -> anyhow::Result<QuantizeOutput<Self::QuantizedOp>> {
        ensure!(
            input_scaling.len() == 1,
            "More than one input scaling factor provided for Softmax. Received {} input scaling factor",
            input_scaling.len()
        );
        // We can work out the intermediate bit size (for now we assume we are using Softmax in an Attention layer)
        let intermediate_bit_size = 2 * (*quantization::BIT_LEN - 1) + ceil_log2(self.max_size);

        let quantised_op = self.quantise(input_scaling[0], intermediate_bit_size)?;
        let output_sf = quantised_op
            .quant_info
            .map(|info| info.lut.output_sf())
            .unwrap();

        let output_scaling = ScalingFactor::from_parts(
            1.0f32,
            0.0f32,
            1.0f32 / output_sf,
            (0, output_sf as Element),
        );
        // To be able to run the quantised Softmax with padding we need an AttentionMask as the previous Layer.
        // For this we need to know what the quantised negative infinity value is and after the rest of the quantisation procedure
        // is finished we need to go back and update the model with this value.
        let negative_infinity = quantised_op
            .quant_info
            .map(|info| info.quantised_negative_infinity())
            .unwrap();

        let mask_transform = SoftmaxMaskTransform::new(node_id, negative_infinity);

        Ok(QuantizeOutput::<Softmax<Element>> {
            quantized_op: quantised_op,
            output_scalings: vec![output_scaling],
            requant_layer: None,
            post_quant_rule: Some(Box::new(mask_transform)),
        })
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "E: ExtensionField + DeserializeOwned")]
pub struct SoftmaxCtx<E: ExtensionField> {
    node_id: NodeId,
    /// This is the quantisation data for the [`Softmax`] op
    quant_info: QuantisedSoftmaxData,
    /// The data about the lookups that are performed in this layer
    lookup_ctx: LayerLookupContext,
    /// The expression used in the sumcheck for the layer
    sumcheck_expression: Vec<Expression<E>>,
}

impl LayerLookupContext {
    /// Softmax behaves slightly differently to normal lookups so we have a custom method to generate the [`LogUpInput`].
    pub fn create_logup_inputs_softmax<PCS, E>(
        &self,
        layer_commitment: &PCS::CommitmentWithWitness,
        challenge_storage: &ChallengeStorage<E>,
        dim_size: usize,
        number_of_range_checks: usize,
        number_of_zero_chunks: usize,
        first_dim: usize,
    ) -> anyhow::Result<Vec<LogUpInput<E>>>
    where
        E: ExtensionField,
        PCS: PolynomialCommitmentScheme<E>,
    {
        // First we extract the polynomials from the layer_commitment
        let polys = PCS::get_arc_mle_witness_from_commitment(layer_commitment);
        // In total we should have first_dim * (number_of_range_checks + number_of_zero_chunks + 1) + first_dim * (1 + number_of_zero_chunks) + first_dim polynomials
        let input_poly_chunk_size = number_of_range_checks + 1 + number_of_zero_chunks;
        let num_input_polys = first_dim * input_poly_chunk_size;
        let output_poly_chunk_size = 1 + number_of_zero_chunks;
        let num_output_polys = first_dim * output_poly_chunk_size;

        // Split the polys up accordingly
        let (input_polys, rest) = polys.split_at(num_input_polys);
        let (output_polys, _) = rest.split_at(num_output_polys);
        // We group all of the columns looked up, for tables that have two columns (e.g. ExpTable) the columns are grouped by instance
        // (so `exp_columns` is ordered as exp_in_1, exp_out_1, exp_in_2, exp_out_2, ..., exp_in_n, exp_out_n for example).
        let (range_columns, exp_columns, zero_columns, error_columns) = input_polys
            .chunks(input_poly_chunk_size)
            .zip(output_polys.chunks(output_poly_chunk_size))
            .fold(
                (vec![], vec![], vec![], vec![]),
                |(mut range_vec, mut exp_vec, mut zero_vec, mut error_vec),
                 (input_chunk, output_chunk)| {
                    let (range_polys, rest) = input_chunk.split_at(number_of_range_checks);
                    let (exp_in_poly, zero_in_polys) = rest.split_at(1);
                    let (exp_out_poly, zero_out_polys) = output_chunk.split_at(1);

                    let range_evals = range_polys
                        .iter()
                        .map(|p| p.get_base_field_vec().to_vec())
                        .collect::<Vec<Vec<E::BaseField>>>();
                    let exp_in_evals = exp_in_poly[0].get_base_field_vec().to_vec();
                    let zero_in_evals = zero_in_polys
                        .iter()
                        .map(|p| p.get_base_field_vec().to_vec())
                        .collect::<Vec<Vec<E::BaseField>>>();
                    let exp_out_evals = exp_out_poly[0].get_base_field_vec().to_vec();
                    let zero_out_evals = zero_out_polys
                        .iter()
                        .map(|p| p.get_base_field_vec().to_vec())
                        .collect::<Vec<Vec<E::BaseField>>>();

                    // We have to reconstruct the error lookup
                    let output = exp_out_evals
                        .iter()
                        .enumerate()
                        .map(|(i, &e)| {
                            e * zero_out_evals
                                .iter()
                                .map(|n| n[i])
                                .product::<E::BaseField>()
                        })
                        .collect::<Vec<E::BaseField>>();

                    let error_evals = output
                        .chunks(dim_size)
                        .map(|chunk| chunk.iter().copied().sum::<E::BaseField>())
                        .collect::<Vec<E::BaseField>>();

                    range_vec.extend(range_evals);
                    exp_vec.push(exp_in_evals);
                    exp_vec.push(exp_out_evals);
                    zero_vec.extend(zero_in_evals.into_iter().interleave(zero_out_evals));
                    error_vec.push(error_evals);
                    (range_vec, exp_vec, zero_vec, error_vec)
                },
            );

        // Here we convert the columns into the correct format for LogUpInput
        self.tables
            .iter()
            .zip([range_columns, exp_columns, zero_columns, error_columns])
            .try_fold(
                Vec::<LogUpInput<E>>::new(),
                |mut inputs_acc, (tt, column_evals)| {
                    let (constant_challenge, column_separation_challenge) = challenge_storage
                        .get_challenges_by_name(&tt.name())
                        .ok_or(anyhow!(
                            "No challenges found for Table {}, cannot generate LogUp input",
                            tt.name()
                        ))?;

                    let logup_input = LogUpInput::<E>::new_lookup(
                        column_evals,
                        constant_challenge,
                        column_separation_challenge,
                        tt.num_columns(),
                    )?;
                    inputs_acc.push(logup_input);
                    Result::<Vec<LogUpInput<E>>, anyhow::Error>::Ok(inputs_acc)
                },
            )
    }
}

impl<E: ExtensionField> OpInfo for SoftmaxCtx<E> {
    fn output_shapes(&self, input_shapes: &[Shape], _padding_mode: PaddingMode) -> Vec<Shape> {
        input_shapes.to_vec()
    }

    fn num_outputs(&self, num_inputs: usize) -> usize {
        num_inputs
    }

    fn describe(&self) -> String {
        "Softmax".to_string()
    }

    fn is_provable(&self) -> bool {
        true
    }
}

impl ProveInfo for Softmax<Element> {
    fn step_info<E: ExtensionField>(
        &self,
        id: NodeId,
        mut aux: ContextAux,
    ) -> Result<(LayerCtx<E>, ContextAux)> {
        if let Some(&quant_info) = self.quant_info() {
            let QuantisedSoftmaxData {
                lut, error_bound, ..
            } = quant_info;

            // Calculate the allowable error in normalisation as an Element
            let allowable_error = (error_bound * lut.output_sf()).round() as Element;

            // Add the tables that Softmax requires
            aux.tables.insert(TableType::Range);
            aux.tables.insert(TableType::ExpTable(lut));
            aux.tables.insert(TableType::ErrorTable(
                lut.output_sf() as Element,
                allowable_error,
            ));

            // If there is one add the ZeroTable
            let number_zero_chunks = quant_info.number_of_zero_chunks();
            let number_of_range_checks = quant_info.number_of_range_checks();
            aux.tables.insert(TableType::ZeroTable);
            let tables = vec![
                TableType::Range,
                TableType::ExpTable(lut),
                TableType::ZeroTable,
                TableType::ErrorTable(lut.output_sf() as Element, allowable_error),
            ];
            let instances_per_table = vec![number_of_range_checks, 1, number_zero_chunks, 1];
            let lookup_ctx = LayerLookupContext::new(tables, instances_per_table);

            // There are no common commitments for this layer
            aux.model_polys = None;
            aux.max_poly_len = aux
                .last_output_shape
                .iter()
                .fold(aux.max_poly_len, |acc, shapes| {
                    acc.max(shapes.next_power_of_two().product())
                });
            let shape = &aux.last_output_shape[0];
            ensure!(
                shape.rank() == 2 || shape.rank() == 3,
                "Softmax only supports 2D or 3D tensors"
            );
            let first_dim = if shape.rank() == 3 { shape[0] } else { 1 };
            let last_dim = if shape.rank() == 3 {
                shape[2]
            } else {
                shape[1]
            };
            let sumcheck_expression =
                build_softmax_sumcheck_expression::<E>(number_zero_chunks, first_dim, last_dim);

            // The output shape is the same as the input shape so we don't need to update it
            // return the LayerCtx and the updated ContextAux
            Ok((
                LayerCtx::Softmax(SoftmaxCtx {
                    node_id: id,
                    quant_info,
                    lookup_ctx,
                    sumcheck_expression,
                }),
                aux,
            ))
        } else {
            bail!("Softmax operation has not been quantised so no proving info available");
        }
    }
}

/// Builds the [`Expression`] used in [`Softmax`] proving to link lookup inputs and outputs to Layer inputs and outputs.
/// We have to show that the normalisation error is within the acceptable range, that `last_claim.eval` relates to the correct combination of the outputs
/// of the `exp` lookup and the `zero` lookups and also that the inputs to the lookups came from masking the shifted layer input.
///
/// We have to check the lookup outputs product is last claim for each 2D tensor and that the error lookup is performed on the output row wise
fn build_softmax_sumcheck_expression<E: ExtensionField>(
    number_zero_chunks: usize,
    first_dim: usize,
    last_dim: usize,
) -> Vec<Expression<E>> {
    let last_claim_eq = Expression::WitIn(0);
    let error_eq = Expression::WitIn(1);
    let logup_eq = Expression::WitIn(2);

    let last_dim_vars = ceil_log2(last_dim);
    let two_mult = E::from_canonical_u64(1 << last_dim_vars);
    (0..first_dim)
        .map(|i| {
            let offset = 3 + i * (1 + number_zero_chunks);
            let challenge_id = (2 * i) as u16;
            // The output expression is the product of the exp output with the zero check outputs.
            // The linking expression is a random linear combination of the exp output and zero check outputs so that
            // we can prove they are the same MLEs used in the lookup
            let (output_expr, linking_expr) = (0..number_zero_chunks).fold(
                (
                    Expression::WitIn(offset as u16)
                        * Expression::Challenge(challenge_id, 1, E::ONE, E::ZERO),
                    Expression::WitIn(offset as u16)
                        * Expression::Challenge(challenge_id + 1, 1, E::ONE, E::ZERO),
                ),
                |(prod_acc, sum_acc), j| {
                    let current_id = (offset + j + 1) as u16;
                    (
                        prod_acc * Expression::WitIn(current_id),
                        sum_acc
                            + Expression::Challenge(challenge_id + 1, j + 2, E::ONE, E::ZERO)
                                * Expression::WitIn(current_id),
                    )
                },
            );
            // We use the output expression twice, once to link to the last_claim and once to show that the row wise sum of the output was used
            // in the error check.
            output_expr
                * (Expression::Challenge(challenge_id, 1, two_mult, E::ZERO) * error_eq.clone()
                    + last_claim_eq.clone())
                + linking_expr * logup_eq.clone()
        })
        .collect::<Vec<Expression<E>>>()
}

impl<E, PCS> VerifiableCtx<E, PCS> for SoftmaxCtx<E>
where
    E: ExtensionField + Serialize + DeserializeOwned,
    E::BaseField: Serialize + DeserializeOwned,
    PCS: PolynomialCommitmentScheme<E>,
{
    type Proof = SoftmaxProof<E, PCS>;
    fn verify<T: transcript::Transcript<E>>(
        &self,
        proof: &Self::Proof,
        last_claims: &[&Claim<E>],
        verifier: &mut Verifier<E, T, PCS>,
        shape_step: &ShapeStep,
    ) -> Result<Vec<Claim<E>>> {
        // First we check that we only have one claim in `last_claims`
        ensure!(
            last_claims.len() == 1,
            "Softmax only outputs 1 claim, received {} while verifying Softmax step",
            last_claims.len()
        );
        // First dim is the number of 2D sub-tensors we have (without padding)
        let first_dim = shape_step.unpadded_input_shape[0][0];
        let input_shape = &shape_step.padded_input_shape[0];
        let final_dim_size = *input_shape
            .last()
            .ok_or(anyhow!("Couldn't verify Softmax, had no input shape"))?;
        let last_claim = last_claims[0];
        let split_point = input_shape.split_point::<E>(last_claim.point())?;

        let dim_vars = ceil_log2(final_dim_size);
        let two = E::from_canonical_u64(2u64);
        let two_inv = two.inverse();

        let SoftmaxProof {
            logup_proof,
            commitment,
            sumcheck_proof,
            evaluations,
        } = proof;

        // Verify the lookup proof
        let batch_claim = verify_logup_proof_multiple_sizes(logup_proof, verifier.transcript)?;

        // Since the lookup ctx is built without knowing the unpadded first dim of the input shpe, here
        // we make a new one in order to verify the proof
        let LayerLookupContext {
            tables,
            instances_per_table,
        } = &self.lookup_ctx;
        let instances_per_table = instances_per_table
            .iter()
            .map(|&n| n * first_dim)
            .collect::<Vec<usize>>();
        let new_lookup_ctx = LayerLookupContext::new(tables.clone(), instances_per_table);
        new_lookup_ctx.verify_logup_batch_claim(&batch_claim, &verifier.challenge_storage)?;

        // Now we squeeze the batching challenge
        // Squeeze a batching challenge from the transcript
        let alphas = (0..first_dim)
            .map(|_| {
                verifier
                    .transcript
                    .sample_and_append_challenge(b"batching_challenge")
                    .elements
            })
            .collect::<Vec<E>>();

        // poly_evals will be in the order range_evals, exp_evals, zero_evals then error_evals
        let poly_evals = batch_claim.poly_evals();

        let number_of_range_checks = self.quant_info.number_of_range_checks();
        let number_of_zero_chunks = self.quant_info.number_of_zero_chunks();
        // Split the poly_evals into their respective sections
        let (range_evals, rest) = poly_evals.split_at(number_of_range_checks * first_dim);
        let (exp_evals, rest) = rest.split_at(2 * first_dim);
        let (zero_evals, error_evals) = rest.split_at(first_dim * 2 * number_of_zero_chunks);
        // We need to unzip the exp and zero evals into their input and output components
        let (exp_in_evals, exp_out_evals): (Vec<E>, Vec<E>) = exp_evals
            .chunks(2)
            .map(|chunk| (chunk[0], chunk[1]))
            .unzip();
        let (zero_in_evals, zero_out_evals): (Vec<E>, Vec<E>) = zero_evals
            .chunks(2)
            .map(|chunk| (chunk[0], chunk[1]))
            .unzip();

        let batch_chal_point = split_point[0];
        let batching_evals = compute_betas_eval(batch_chal_point);
        // Now we can compute the initial claim for the sumcheck, this should be a random linear combination of
        // `last_claim.evaluation()`, the error lookup evaluation and the evaluations of the exp and zero lookups
        let initial_claim = izip!(
            alphas.iter(),
            error_evals.iter(),
            exp_out_evals.iter(),
            zero_out_evals.chunks(number_of_zero_chunks),
            batching_evals.iter()
        )
        .fold(
            last_claim.evaluation(),
            |acc, (&alpha, &error, &exp_out, zero_chunk, &batch)| {
                let sum_part = zero_chunk
                    .iter()
                    .fold(
                        (exp_out * alpha, alpha * alpha),
                        |(eval_acc, chal_acc), &e| (eval_acc + chal_acc * e, chal_acc * alpha),
                    )
                    .0;
                let error_part = batch * batch * error;
                acc + sum_part + error_part
            },
        );

        let last_claim_eq_point = split_point
            .iter()
            .skip(1)
            .rev()
            .flat_map(|s| *s)
            .copied()
            .collect::<Vec<E>>();

        // The error lookup is performed over the output summed on the final dimension so we need to extend the point used with correct number
        // of 2^-1 entries
        let full_error_point = std::iter::repeat_n(two_inv, dim_vars)
            .chain(batch_claim.point().iter().skip(dim_vars).copied())
            .collect::<Vec<E>>();

        let max_degree = 2 + number_of_zero_chunks;

        let aux_info = VPAuxInfo {
            max_num_variables: last_claim_eq_point.len(),
            max_degree,
            ..Default::default()
        };
        // Verify the Sumcheck proof
        let subclaim = IOPVerifierState::<E>::verify(
            initial_claim,
            sumcheck_proof,
            &aux_info,
            verifier.transcript,
        );
        let sumcheck_point = subclaim
            .point
            .iter()
            .map(|c| c.elements)
            .collect::<Vec<E>>();

        let last_claim_eq = identity_eval(&last_claim_eq_point, &sumcheck_point);
        let logup_eq = identity_eval(batch_claim.point(), &sumcheck_point);
        let error_eq = identity_eval(&full_error_point, &sumcheck_point);

        let all_sumcheck_evals = [last_claim_eq, error_eq, logup_eq]
            .into_iter()
            .chain(
                evaluations
                    .iter()
                    .take(first_dim * (1 + number_of_zero_chunks))
                    .copied(),
            )
            .collect::<Vec<E>>();

        let challenges = batching_evals
            .iter()
            .zip(alphas)
            .flat_map(|(&a, b)| [a, b])
            .collect::<Vec<E>>();
        // Check that the provided evaluation matches the expected evaluation from the sumcheck
        let calc_subclaim =
            self.sumcheck_expression
                .iter()
                .take(first_dim)
                .fold(E::ZERO, |acc, expr| {
                    eval_by_expr_with_instance(
                        &[],
                        &all_sumcheck_evals,
                        &[],
                        &[],
                        &challenges,
                        expr,
                    )
                    .right()
                    .unwrap()
                        + acc
                });

        ensure!(
            subclaim.expected_evaluation == calc_subclaim,
            "Softmax sumcheck subclaim evaluation did not match expected evaluation"
        );

        let shift_eval_point = batch_claim.point()[dim_vars..].to_vec();
        let shift_evals = evaluations
            .iter()
            .skip(first_dim * (1 + number_of_zero_chunks))
            .copied()
            .collect::<Vec<E>>();
        // Constants used to recombine the claims
        let base_multiplier = E::from_canonical_u64(1u64 << *quantization::BIT_LEN);
        let right_shift_field = E::from_canonical_u64(1u64 << self.quant_info.right_shift);
        let rounding = E::from_canonical_u64(1u64 << (self.quant_info.right_shift - 1));
        let fpm_field: E = self.quant_info.fixed_point_multiplier.to_field();
        let fpm_inv = fpm_field.inverse();
        let table_size_field: E = self.quant_info.lut.full_table_size().to_field();
        let zero_offset = table_size_field * right_shift_field;

        // Combine the range claims for each chunk
        let (low_parts, stacked_range_evals): (Vec<E>, Vec<Vec<E>>) = range_evals
            .chunks(number_of_range_checks)
            .map(|chunk| {
                let input_part = chunk
                    .iter()
                    .fold((E::ZERO, E::ONE), |(acc, pow_two), &b| {
                        (acc + pow_two * b, pow_two * base_multiplier)
                    })
                    .0;
                (input_part, chunk.to_vec())
            })
            .unzip();
        // Combine the exp claims
        let (exp_parts, stacked_exp_evals): (Vec<E>, Vec<E>) = exp_in_evals
            .iter()
            .map(|&e| (e * right_shift_field, e))
            .unzip();
        // Combine the zero claims for each chunk
        let (high_parts, stacked_high_evals): (Vec<E>, Vec<Vec<E>>) = zero_in_evals
            .chunks(number_of_zero_chunks)
            .map(|chunk| {
                let input_part = chunk
                    .iter()
                    .fold((E::ZERO, zero_offset), |(acc, pow_two), &b| {
                        (acc + pow_two * b, pow_two * base_multiplier)
                    })
                    .0;
                (input_part, chunk.to_vec())
            })
            .unzip();
        // Now we can recombine everything to get the input eval
        let input_eval = izip!(
            low_parts,
            exp_parts,
            high_parts,
            shift_evals.iter(),
            batching_evals
        )
        .map(|(l, e, h, &shift, batch)| ((l + e - h - rounding) * fpm_inv - shift) * batch)
        .sum::<E>();

        let first_commit_evals = izip!(stacked_range_evals, stacked_exp_evals, stacked_high_evals)
            .flat_map(|(mut rs, e, zs)| {
                rs.push(e);
                rs.extend(zs);
                rs
            })
            .collect::<Vec<E>>();

        let first_commit_point = batch_claim.point().to_vec();

        // The second commitment is the exp output and the zero outputs
        let second_commit_evals = evaluations
            .iter()
            .take(first_dim * (1 + number_of_zero_chunks))
            .copied()
            .collect::<Vec<E>>();
        let second_commit_point = sumcheck_point.clone();
        // Combine them all in the correct order and add them to the claim prover
        let layer_claims = vec![
            (first_commit_point, first_commit_evals),
            (second_commit_point, second_commit_evals),
            (shift_eval_point, shift_evals.clone()),
        ];

        verifier
            .commit_verifier
            .add_witness_claim(self.node_id, commitment.clone(), layer_claims);

        let input_claim = Claim::<E>::new(
            batch_claim
                .point()
                .iter()
                .chain(batch_chal_point.iter())
                .copied()
                .collect::<Vec<E>>(),
            input_eval,
        );

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

    use core::f32;
    use ff_ext::GoldilocksExt2;

    use crate::{
        Tensor, init_test_logging,
        layers::{Layer, transformer::attention::attention_mask::AttentionMask},
        model::{Model, test::prove_model},
        padding::PaddingMode,
        tensor::is_close_with_tolerance,
    };
    // use burn::tensor::{Int as BInt, Tensor as BTensor, TensorData as BTensorData};
    use proptest::prelude::*;

    use super::*;

    #[test]
    fn test_quantise() {
        // For now we test with GPT2 like parameters
        let scale = 1.0f32 / 64.0f32.sqrt();
        let softmax = Softmax::<f32>::new_with_scale(scale, 1024);

        for num_tokens in 1024..=1024 {
            // Make random q and k vectors
            let test_q = Tensor::<f32>::random(&vec![num_tokens, 768].into());
            let test_k = Tensor::<f32>::random(&vec![768, num_tokens].into());

            let q_scaling = ScalingFactor::from_tensor(&test_q, None);
            let k_scaling = ScalingFactor::from_tensor(&test_k, None);

            // Pick the quantised domain to be Some((-1i128 << 24, 1i128 << 24)) since matrix multiplication on 768 columns adds at most 10 to the bit size
            // (already at bit size 14 before this due to multiplication of two 8 bit quant integers)
            let qk_scaling = ScalingFactor::from_scale(
                q_scaling.scale() * k_scaling.scale(),
                Some((-1 << 24, 1 << 24)),
            );

            let test_q_quant = test_q.to_quantized(&q_scaling);
            let test_k_quant = test_k.to_quantized(&k_scaling);

            let test_qk_quant = test_q_quant.matmul(&test_k_quant);

            let test_qk_dequant = test_qk_quant.dequantize(&qk_scaling);

            let intermerdiate_bit_size = 2 * (*quantization::BIT_LEN - 1) + ceil_log2(768);

            // Now to test the quantised softmax we quantise `float_input` and run the quantised evaluation.
            // We also quantise and dequantise `float_input` and run this data through the float evaluation and then compare the two results.

            let quant_softmax = softmax
                .quantise(qk_scaling, intermerdiate_bit_size)
                .unwrap();

            // Obtain the quantised output
            let quant_output = quant_softmax
                .evaluate::<GoldilocksExt2>(
                    &[&test_qk_quant],
                    &[vec![num_tokens, num_tokens].into()],
                )
                .unwrap();
            // The result of running the quantised input as floats
            let dequant_output = softmax
                .evaluate::<GoldilocksExt2>(
                    &[&test_qk_dequant],
                    &[vec![num_tokens, num_tokens].into()],
                )
                .unwrap();

            let QuantisedSoftmaxData {
                lut, error_bound, ..
            } = quant_softmax.quant_info.as_ref().unwrap();

            // The relative error comes from quantising the shift factor
            // The absolute error comes from the tables output scale factor
            let rel_error = (1.0 / (2.0f32 * lut.input_sf())).exp() - 1.0;
            let out_error = 1.0 / (2.0f32 * lut.output_sf());

            for (q_chunk, f_chunk) in quant_output.outputs[0]
                .get_data()
                .chunks(num_tokens)
                .zip(dequant_output.outputs[0].get_data().chunks(num_tokens))
            {
                for (&q, f) in q_chunk.iter().zip(f_chunk.iter()) {
                    let float_q = q as f32 / lut.output_sf();
                    // println!("quant, {q}, float_q {float_q}, dequant {f}");
                    let quant_dequant_diff = (float_q - f).abs();
                    assert!(
                        is_close_with_tolerance(&[float_q], &[*f], out_error, rel_error),
                        "Quant dequant diff was larger than expected got: {quant_dequant_diff}, expected less than {}",
                        *f * rel_error + out_error
                    );
                }
            }

            let max_error = error_bound * lut.output_sf();

            quant_output.outputs[0]
                .get_data()
                .chunks(num_tokens)
                .for_each(|chunk| {
                    let row_sum = chunk.iter().sum::<Element>();

                    let diff_from_one = (row_sum - lut.output_sf() as Element).abs();
                    // println!("diff: {diff_from_one}, lut output sf {}", lut.output_sf());
                    assert!(diff_from_one <= max_error.round_ties_even() as Element, "Row sum diff was larger than expected got: {diff_from_one}, expected less than {max_error}, error_bound {error_bound}");
                });
        }
    }

    #[test]
    fn test_softmax() {
        let softmax = Softmax::<f32>::new(3);
        let input = Tensor::new(
            vec![1, 3, 3].into(),
            vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        );
        let output = softmax
            .evaluate::<GoldilocksExt2>(&[&input], &[vec![1, 3, 3].into()])
            .unwrap();
        assert_eq!(*output.outputs[0].shape(), vec![1, 3, 3].into());

        output.outputs[0].get_data().chunks(3).for_each(|chunk| {
            assert!((chunk.iter().sum::<f32>() - 1.0) < f32::EPSILON);
        });
    }

    #[test]
    fn test_softmax_proving() {
        init_test_logging("debug");
        let dim_size = 1000;
        let input_shape = vec![12, dim_size, dim_size];

        let mut model =
            Model::new_from_input_shapes(vec![input_shape.into()], PaddingMode::NoPadding);

        let mask = AttentionMask::<f32>::new(dim_size, f32::NEG_INFINITY);
        let softmax = Softmax::<f32>::new_with_scale(1.0f32 / 64.0f32.sqrt(), 1024);

        let mask_id = model
            .add_consecutive_layer(Layer::AttentionMask(mask), None)
            .unwrap();
        let _ = model
            .add_consecutive_layer(Layer::Softmax(softmax), Some(mask_id))
            .unwrap();

        model.route_output(None).unwrap();
        model.describe();
        prove_model(model, &mut GenStore::default()).unwrap();
    }

    #[derive(Clone)]
    struct SoftmaxInput {
        n: usize,
        data: Vec<f32>,
    }
    impl core::fmt::Debug for SoftmaxInput {
        fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
            f.debug_struct("SoftmaxInput")
                .field("n", &self.n)
                .field("data", &self.data)
                .finish()
        }
    }

    fn any_softmax_input(range: core::ops::Range<f32>) -> impl Strategy<Value = SoftmaxInput> {
        // We start from n = 3 because the n = 2 case would use the sigmoid function instead.
        (3usize..16).prop_flat_map(move |n| {
            let len = n * n;
            prop::collection::vec(range.clone(), len).prop_map(move |v| SoftmaxInput { n, data: v })
        })
    }

    proptest! {
        #[test]
        fn prop_softmax_f32(input in any_softmax_input(-4.0..4.0)) {
            let SoftmaxInput { n, data } = input;
            let tensor = Tensor::new(vec![1, n, n].into(), data.clone());
            let layer = Softmax::<f32>::new(n);
            let eval = layer.evaluate::<GoldilocksExt2>(&[&tensor], &[vec![1,n,n].into()]).unwrap();
            let got = eval.outputs[0].get_data();

            for row in got.chunks(n) { prop_assert!(((row.iter().sum::<f32>() - 1.0).abs()) < 1e-4 * n as f32 + 1e-6); }
        }

        #[test]
        fn prop_softmax_quantized(input in any_softmax_input(-2.0..2.0)) {
            let SoftmaxInput { n, data } = input;
            let float_tensor = Tensor::<f32>::new(vec![1, n, n].into(), data.clone());
            let scaling = ScalingFactor::from_tensor(&float_tensor, None);
            let quant_input = float_tensor.to_quantized(&scaling);

            let layer_f = Softmax::<f32>::new(n);
            let layer_q = layer_f.quantise(scaling, *quantization::BIT_LEN).unwrap();

            let out_q = layer_q.evaluate::<GoldilocksExt2>(&[&quant_input], &[vec![1,n,n].into()]).unwrap();

            let quant_rows = out_q.outputs[0].get_data();
            let qi = layer_q.quant_info().unwrap();
            let row_err_bound_scaled = (qi.error_bound * qi.lut.output_sf()).round() as Element;


            for (j ,row_q) in quant_rows.chunks(n).enumerate() {
                // Row sum closeness (integer domain)
                let row_sum: Element = row_q.iter().copied().sum();
                let diff = (row_sum - qi.lut.output_sf() as Element).abs();
                prop_assert!(diff <= row_err_bound_scaled, "row {j} sum {row_sum}, row {row_q:?}, expected {}, diff {diff}, allowed {}, float allowed {}",  qi.lut.output_sf(), row_err_bound_scaled, qi.error_bound * qi.lut.output_sf());
            }
        }
    }
}
