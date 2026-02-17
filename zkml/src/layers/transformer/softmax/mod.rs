//! This layer applies the softmax function to the last dimension of the input tensor
use crate::{
    Claim, Element, NextPowerOfTwo, Number, ScalingStrategy, Shape, Tensor,
    commit::{compute_betas_eval, identity_eval},
    eval_zeroifier_mle,
    graph::NodeId,
    iop::{
        ChallengeStorage,
        context::{ContextAux, ProverContext, ShapeStep},
        verifier::Verifier,
    },
    layers::{
        LayerCtx, LayerProof,
        provable::{
            Evaluate, LayerOut, OpInfo, PadOp, ProvableOp, ProveInfo, ProvingData, QuantizeOp,
            QuantizeOutput, VerifiableCtx,
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
    model::{Step, transform::impls::softmax_mask::SoftmaxMaskTransform},
    padding::PaddingMode,
    quantization::{self, ScalingFactor, ToField},
    tensor::{TensorHandle, TensorTypeParam, WrappedTensor},
    to_base, to_bit_sequence_le,
};
use anyhow::{Result, anyhow, bail, ensure};
use burn::tensor::TensorData;
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
use std::fmt::Debug;
use sumcheck::{
    structs::{IOPProof, IOPProverState, IOPVerifierState},
    util::optimal_sumcheck_threads,
};
use tenstore::StorageKey;
use transcript::Transcript;
use witness::RowMajorMatrix;

pub mod evaluate;
pub mod lookup;
pub mod prove;
pub mod verify;

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
    pub(crate) max_size: usize,
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

/// Stores the shift tensor computed during inference.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SoftmaxData {
    /// This is the tensor of normalisation shifts to apply in quantised evaluation.
    shift_tensor: WrappedTensor<Element>,
}

/// Stores the shift tensor computed during inference.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SoftmaxHandle {
    /// This is the tensor of normalisation shifts to apply in quantised evaluation.
    pub(crate) shift_handle: TensorHandle<Element>,
}

impl SoftmaxHandle {
    pub(crate) fn new(
        storage_key: StorageKey<Vec<Element>>,
        softmax_data: SoftmaxData,
        store: tenstore::GenStore,
    ) -> Self {
        let handle =
            TensorHandle::from_wrapped_tensor(storage_key, store, softmax_data.shift_tensor);
        Self {
            shift_handle: handle,
        }
    }
}

impl<N: TensorTypeParam> Softmax<N> {
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

        let temperature = Number::to_f32(&self.scalar)?;

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
        let int_part = log_m.trunc() as isize;
        // This is used to calculate the fixed point multiplier
        let float_part = log_m.fract();

        let epsilon = 2.0f32.powf(float_part);

        let fp_scale = FIXED_POINT_SCALE;
        let fixed_point_multiplier =
            (epsilon * (1u64 << FIXED_POINT_SCALE) as f32).round_ties_even() as Element;

        // Assertion to check that we can perform requantisation, we need intermediate_bit_size + fp_scale <= 63
        ensure!(
            intermediate_bit_size + fp_scale <= 63,
            "intermediate bit size: {intermediate_bit_size}, fp scale: {fp_scale}, int part: {int_part}",
        );
        // Now we can create the ExpTable
        let lut = ExpTable::new(input_sf, output_sf, table_bit_size);

        let quant_info = QuantisedSoftmaxData {
            right_shift: (FIXED_POINT_SCALE as isize - int_part) as usize,
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
        tracing::info!(
            "Calculating scale factors for Softmax with max context size {}",
            self.max_size
        );
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
        // We set the starting power to 16 as this should be large enough for most cases while still giving a very manageable table size.
        let mut initial_power = 16;
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
        let temperature = Number::to_f32(&self.scalar).unwrap_or(1.0);
        let all_same_shift = (-(max_context_size.ln()) / (input_scaling.scale() * temperature))
            .round_ties_even() as Element;
        let rescaling_mult = input_scaling.scale() * temperature * input_sf;

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

impl Evaluate<f32> for Softmax<f32> {
    fn evaluate(&self, inputs: &[&WrappedTensor<f32>]) -> anyhow::Result<LayerOut<f32>> {
        ensure!(
            inputs.len() == 1,
            "softmax expects exactly one input tensor currently"
        );
        let input = inputs[0];

        // Convert to a 2D tensor, rescale and apply softmax.
        let b_input = input
            .clone()
            .flatten_to_dim_2(0, input.rank() - 2)
            .mul_scalar(self.scalar);
        let out = b_input.softmax(1)?;
        let out = out.reshape(input.shape())?;

        Ok(LayerOut::from_tensor(out))
    }
}

impl<N> OpInfo for Softmax<N> {
    fn output_shapes(
        &self,
        input_shapes: &[Shape],
        _padding_mode: crate::padding::PaddingMode,
    ) -> Result<Vec<Shape>> {
        Ok(input_shapes.to_vec())
    }

    fn num_outputs(&self, num_inputs: usize) -> Result<usize> {
        Ok(num_inputs)
    }

    fn describe(&self) -> String {
        "Softmax".to_string()
    }

    fn is_provable(&self) -> bool {
        true
    }
}

impl Evaluate<Element> for Softmax<Element> {
    fn evaluate(&self, inputs: &[&WrappedTensor<Element>]) -> Result<LayerOut<Element>> {
        self.evaluate_internal(inputs)
    }
}

impl PadOp for Softmax<Element> {}

impl<E, PCS> ProvableOp<E, PCS> for Softmax<Element>
where
    E: ExtensionField,
    PCS: PolynomialCommitmentScheme<E> + Send + Sync,
    PCS::CommitmentWithWitness: Serialize + DeserializeOwned + Send + Sync,
    PCS::ProverParam: Send + Sync,
{
    type Ctx = SoftmaxCtx;

    fn prove<T: transcript::Transcript<E>>(
        &self,
        node_id: NodeId,
        ctx: &Self::Ctx,
        last_claims: Vec<&Claim<E>>,
        step_data: &Step<Element>,
        prover: &mut crate::Prover<E, T, PCS>,
    ) -> Result<Vec<Claim<E>>> {
        let input = step_data.input_tensor_at(0)?;
        let unpadded_shape = input.unpadded_shape();
        let (claims, proof) =
            self.prove_internal(node_id, last_claims, ctx, unpadded_shape, prover)?;
        prover.push_proof(node_id, LayerProof::<E, PCS>::Softmax(proof));

        Ok(claims)
    }

    fn gen_lookup_witness(
        &self,
        id: NodeId,
        ctx: &ProverContext<E, PCS>,
        step_data: &Step<Element>,
    ) -> Result<LookupWitnessGen<E, PCS>> {
        let input_tensors = step_data.input_tensors()?;
        let output_tensors = step_data.output_tensors()?;

        ensure!(
            step_data.node_inputs.len() == 1,
            "Found more than 1 input in inference step of Softmax layer"
        );
        ensure!(
            output_tensors.len() == 1,
            "Found more than 1 output in inference step of Softmax layer"
        );
        let softmax_handle = step_data.node_outputs.try_softmax_data().ok_or(anyhow!(
            "Softmax data not found in inference step for Softmax layer"
        ))?;

        self.lookup_witness(
            id,
            ctx,
            &input_tensors[0],
            &output_tensors[0],
            softmax_handle,
        )
    }
}

impl QuantizeOp for Softmax<f32> {
    type QuantizedOp = Softmax<Element>;

    fn quantize_op<S: ScalingStrategy>(
        self,
        _data: &S::AuxData,
        node_id: NodeId,
        input_scaling: &[ScalingFactor],
        _unpadded_input_shapes: &[Shape],
        _output_scalings: &[ScalingFactor],
        _unpadded_output_shapes: &[Shape],
    ) -> anyhow::Result<QuantizeOutput<Self::QuantizedOp>> {
        ensure!(
            input_scaling.len() == 1,
            "More than one input scaling factor provided for Softmax. Received {} input scaling factor",
            input_scaling.len()
        );
        // We can work out the intermediate bit size (for now we assume we are using Softmax in an Attention layer)
        // let intermediate_bit_size = 2 * (*quantization::BIT_LEN - 1) + ceil_log2(self.max_size);
        let (input_min, input_max) = input_scaling[0].domain();
        let intermediate_bit_size = ceil_log2(input_max.abs().max(input_min.abs()) as usize);

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
pub struct SoftmaxCtx {
    node_id: NodeId,
    /// This is the quantisation data for the [`Softmax`] op
    quant_info: QuantisedSoftmaxData,
    /// The data about the lookups that are performed in this layer
    pub(crate) lookup_ctx: LayerLookupContext,
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

impl OpInfo for SoftmaxCtx {
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

            // If there is one add the ZeroTable
            let number_zero_chunks = quant_info.number_of_zero_chunks();
            let number_of_range_checks = quant_info.number_of_range_checks();
            let tables = vec![
                TableType::Range,
                TableType::ExpTable(lut),
                TableType::ZeroTable,
                TableType::ErrorTable(lut.output_sf() as Element, allowable_error),
            ];
            let instances_per_table = vec![number_of_range_checks, 1, number_zero_chunks, 1];
            let lookup_ctx = LayerLookupContext::new(tables, instances_per_table);

            let shape = &aux.last_output_shape[0];
            ensure!(
                shape.rank() == 2 || shape.rank() == 3,
                "Softmax only supports 2D or 3D tensors"
            );

            // There are no common commitments for this layer
            aux.model_polys = None;
            aux.max_poly_len = {
                let shape_2d: Shape = shape[shape.len() - 2..].to_vec().into();
                shape_2d.numel().next_power_of_two()
            };

            // The output shape is the same as the input shape so we don't need to update it
            // return the LayerCtx and the updated ContextAux
            Ok((
                LayerCtx::Softmax(SoftmaxCtx {
                    node_id: id,
                    quant_info,
                    lookup_ctx,
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

impl<E, PCS> VerifiableCtx<E, PCS> for SoftmaxCtx
where
    E: ExtensionField,
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
        self.verify_internal(proof, last_claims, verifier, shape_step)
    }

    fn write_proof_to_transcript<T: Transcript<E>>(
        &self,
        proof: &Self::Proof,
        transcript: &mut T,
    ) -> anyhow::Result<()> {
        proof.write_commitment(transcript)
    }
}

/// Calculates the batching challenges for the given point and shape.
pub(crate) fn calculate_batching_challenges<E: ExtensionField>(
    point: &[E],
    shape: &Shape,
    unpadded_shape: &Shape,
) -> Result<Vec<E>> {
    // For now we only support up to rank 4 tensors
    ensure!(
        shape.rank() <= 4,
        "Unsupported tensor rank {}",
        shape.rank()
    );
    let split_point = shape.split_point(point)?;
    // The batching challenges are the eq_poly evals that correspond to the non-padded dimensions, up to the last two dims
    let batching_points = &split_point[..shape.rank() - 2];
    let unpadded_dims_slice = &unpadded_shape[..shape.rank() - 2];
    let batching_challenges = batching_points.iter().zip(unpadded_dims_slice.iter()).fold(
        vec![E::ONE],
        |mut chal_acc, (dim_point, unpadded_dim)| {
            let dim_evals = compute_betas_eval(dim_point);
            chal_acc = chal_acc
                .into_iter()
                .flat_map(|c| {
                    dim_evals[..*unpadded_dim]
                        .iter()
                        .map(|&d| c * d)
                        .collect::<Vec<E>>()
                })
                .collect();
            chal_acc
        },
    );

    Ok(batching_challenges)
}

/// Evaluates the row less than polynomial at the given row point for the given unpadded sequence length.
pub(crate) fn evaluate_row_lt_poly<E: ExtensionField>(
    row_point: &[E],
    unpadded_seq_len: usize,
) -> Result<E> {
    let bit_len = ceil_log2(unpadded_seq_len);
    ensure!(
        row_point.len() == bit_len,
        "Row point length {} does not match unpadded seq len log2 {bit_len}",
        row_point.len(),
    );

    let seq_len_bits = to_bit_sequence_le(unpadded_seq_len - 1, bit_len)
        .map(E::from_canonical_usize)
        .collect::<Vec<E>>();
    let row_eval = eval_zeroifier_mle(row_point, &seq_len_bits);
    Ok(row_eval)
}

#[cfg(test)]
mod tests {
    use core::f32;
    use tenstore::GenStore;

    use crate::{
        Tensor, init_test_logging,
        layers::{Layer, transformer::attention_mask::AttentionMask},
        model::{Model, test::prove_model},
        padding::PaddingMode,
        quantization::{Dequantize, Quantize},
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

        for num_tokens in 1015..=1024 {
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

            let test_q_quant = test_q.quantize(&q_scaling);
            let test_k_quant = test_k.quantize(&k_scaling);

            let test_qk_quant = test_q_quant.matmul(&test_k_quant).unwrap();

            let test_qk_dequant = test_qk_quant.dequantize(&qk_scaling);

            let intermerdiate_bit_size = 2 * (*quantization::BIT_LEN - 1) + ceil_log2(768);

            // Now to test the quantised softmax we quantise `float_input` and run the quantised evaluation.
            // We also quantise and dequantise `float_input` and run this data through the float evaluation and then compare the two results.

            let quant_softmax = softmax
                .quantise(qk_scaling, intermerdiate_bit_size)
                .unwrap();

            // Obtain the quantised output
            let quant_output = quant_softmax
                .evaluate(&[&test_qk_quant.as_wrapped()])
                .unwrap();
            // The result of running the quantised input as floats
            let dequant_output = softmax.evaluate(&[&test_qk_dequant.as_wrapped()]).unwrap();

            let QuantisedSoftmaxData {
                lut, error_bound, ..
            } = quant_softmax.quant_info.as_ref().unwrap();

            // The relative error comes from quantising the shift factor
            // The absolute error comes from the tables output scale factor
            let input_error_factor = (1.0 / (2.0f32 * lut.input_sf())).exp() - 1.0;
            let table_max_value: Element = 1 + (-1 << lut.table_bit_size());
            let val_too_large_error = (table_max_value as f32 / lut.input_sf()).exp();
            let table_rounding = 1.0 / (2.0 * lut.output_sf());
            let other_error_part = table_rounding.max(val_too_large_error);

            let rel_error = input_error_factor + other_error_part + f32::EPSILON;
            let out_error = 1.0 / (2.0f32 * lut.output_sf()) + f32::EPSILON;

            for (q_chunk, f_chunk) in quant_output.outputs[0]
                .get_data()
                .chunks(num_tokens)
                .zip(dequant_output.outputs[0].get_data().chunks(num_tokens))
            {
                for (&q, f) in q_chunk.iter().zip(f_chunk.iter()) {
                    let float_q = q as f32 / lut.output_sf();

                    let quant_dequant_diff = (float_q - f).abs();
                    assert!(
                        is_close_with_tolerance(&[float_q], &[*f], out_error, rel_error),
                        "Quant dequant diff was larger than expected got: {quant_dequant_diff}, expected less than {}",
                        *f * rel_error + out_error,
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
        )
        .unwrap();
        let output = softmax.evaluate(&[&input.as_wrapped()]).unwrap();
        assert_eq!(output.outputs[0].shape(), vec![1_usize, 3, 3].into());

        output.outputs[0].get_data().chunks(3).for_each(|chunk| {
            assert!((chunk.iter().sum::<f32>() - 1.0) < f32::EPSILON);
        });
    }

    #[test]
    fn test_softmax_proving() {
        init_test_logging("debug");
        let dim_size = 200;
        let input_shape = vec![12, dim_size, dim_size];

        let mut model =
            Model::new_from_input_shapes(vec![input_shape.into()], PaddingMode::NoPadding);

        let mask = AttentionMask::<f32>::new(f32::NEG_INFINITY);
        let softmax = Softmax::<f32>::new_with_scale(1.0f32 / 64.0f32.sqrt(), 1024);

        let mask_id = model
            .add_consecutive_layer(Layer::AttentionMask(mask), None)
            .unwrap();
        let _ = model
            .add_consecutive_layer(Layer::Softmax(softmax), Some(mask_id))
            .unwrap();

        model.automatic_output_labelling().unwrap();
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
            let tensor = Tensor::new(vec![1, n, n].into(), data.clone()).unwrap();

            let layer = Softmax::<f32>::new(n);
                let eval = layer.evaluate(&[&tensor.as_wrapped()]).unwrap();
            let got = eval.outputs[0].get_data();

            for row in got.chunks(n) { prop_assert!(((row.iter().sum::<f32>() - 1.0).abs()) < 1e-4 * n as f32 + 1e-6); }
        }

        #[test]
        fn prop_softmax_quantized(input in any_softmax_input(-2.0..2.0)) {
            let SoftmaxInput { n, data } = input;
            let float_tensor = Tensor::<f32>::new(vec![1, n, n].into(), data.clone()).unwrap();
            let scaling = ScalingFactor::from_tensor(&float_tensor, None);
            let quant_input = float_tensor.quantize(&scaling);

            let layer_f = Softmax::<f32>::new(n);
            let layer_q = layer_f.quantise(scaling, *quantization::BIT_LEN).unwrap();

            let out_q = layer_q.evaluate(&[&quant_input.as_wrapped()]).unwrap();

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
