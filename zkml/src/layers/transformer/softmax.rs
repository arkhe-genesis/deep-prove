//! This layer applies the softmax function to the last dimension of the input tensor
use core::f32;
use std::{fmt::Debug, marker::PhantomData};

use crate::{
    Claim, Element, ScalingStrategy, Tensor,
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
        transformer::mha::eval_zeroifier_mle,
    },
    lookup::{
        context::{
            COLUMN_SEPARATOR, LayerLookupContext, LookupWitnessGen, SoftmaxTableData, TableType,
            count_elements,
        },
        logup_gkr::{
            prover::batch_multiple_sizes_prove,
            structs::{LogUpBatchProof, LogUpInput},
            verifier::verify_logup_proof_multiple_sizes,
        },
    },
    model::StepData,
    padding::PaddingMode,
    quantization::{self, Fieldizer, ScalingFactor},
    tensor::{Number, Shape},
    to_base,
};

use anyhow::{Result, anyhow, ensure};

use ark_std::Zero;
use either::Either;
use ff_ext::ExtensionField;
use itertools::Itertools;
use mpcs::PolynomialCommitmentScheme;
use multilinear_extensions::{
    Expression,
    mle::{IntoMLE, MultilinearExtension},
    util::{ceil_log2, transpose},
    virtual_poly::VPAuxInfo,
    virtual_polys::VirtualPolynomialsBuilder,
};
use witness::RowMajorMatrix;

use serde::{Deserialize, Serialize, de::DeserializeOwned};
use sumcheck::{
    structs::{IOPProof, IOPProverState, IOPVerifierState},
    util::optimal_sumcheck_threads,
};
use tenstore::TenStore;
use transcript::Transcript;

/// The base 2 logarithm of the scale factor used in exponential lookup tables
pub(crate) const LOG_SCALE_FACTOR: usize = 24;
/// The scale factor for our fixed point arithmetic
pub(crate) const SCALE_FACTOR: usize = 1 << LOG_SCALE_FACTOR;
/// The scale factor of the outputs of the `exp` lookup
pub(crate) const OUTPUT_SCALE_FACTOR: usize = 1 << 12;

#[derive(Debug, Clone, Serialize, Deserialize)]
/// Stores data about the Softmax operation, which is used to map a tensor of values to a tensor of probability distributions.
/// This is done by picking a dimension to normalise over and calculating
///             `x -> exp(scale * x) / (\sum_{i \in dim} exp(scale * x_{i}))`.
pub struct Softmax<N> {
    // By default, it's equal to 1
    /// In the floating point case this is the factor we multiply by before exponentiating, when thought of as a Boltzmann distribution this is
    /// often referred to as the "Temperature".
    ///
    /// For the quantised version this is the factor we must rescale by in order to make use of the lookup table.
    pub scalar: N,
    /// This is the maximum size of dimension that we will normalise over. For example in an Attention layer this would be the maximum context size.
    max_size: usize,
    /// This is the extra information required to compute the quantised version, it defaults to [`None`].
    quant_info: Option<QuantisedSoftmaxData>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Copy)]
/// This struct is used to store information used when evaluating the quantised version of [`Softmax`] on
/// [`Element`]s.
struct QuantisedSoftmaxData {
    /// The [`ScalingFactor`] of the inputs
    input_scale_factor: ScalingFactor,
    /// This stores the [`SoftmaxTableData`]
    lut: SoftmaxTableData,
    /// The error bound as calculated by the formulae given in the zkLLM paper
    error_bound: f32,
    /// This is the inverse of the float temperature for calculating row normalisation
    inv_float_temperature: f32,
    /// This value indicates the point that we map everything greater than this to zero
    bkm: Element,
    /// This value tells use how many chunks we need to make after the exp lookup chunk
    number_zero_chunks: usize,
    /// This value tells us how many variables the zeroing table has
    zero_table_vars: usize,
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

impl<N: Number> Default for Softmax<N> {
    fn default() -> Self {
        Softmax {
            scalar: N::unit(),
            max_size: 1024usize,
            quant_info: None,
        }
    }
}

impl<N: Number> Softmax<N> {
    pub fn new() -> Self {
        Softmax::<N>::default()
    }

    pub fn new_with_scale(scale: N, max_context_size: usize) -> Softmax<N> {
        Softmax {
            scalar: scale,
            max_size: max_context_size,
            quant_info: None,
        }
    }
    pub fn quantise(&self, input_scaling: ScalingFactor) -> Result<Softmax<Element>> {
        // First we work out what we need to multiply by to get the input scale factor to be `SCALE_FACTOR`
        let input_scale_factor = input_scaling.scale();
        let temperature = self.scalar.to_f32()?;
        let inv_float_temperature = 1.0f32 / temperature;
        let multiplier = (SCALE_FACTOR as f32 * input_scale_factor).round() as Element;

        // We want to be able to cover all possible inputs, to do this we need to work out what the minimum quantised input is.
        // this can be calculated by taking `input_scaling.domain().0` and then subtracting the maximum possible shift for normalisation.
        let (quantised_min, _) = input_scaling.domain();

        // The maximum shift would be if every element in the row is `quantised_max`, in this case it can be calculated as
        // (-SCALE_FACTOR as f32) * (inv_float_temperature * (self.max_size as f32).ln() + input_scaling.max())
        let max_shift = (-(SCALE_FACTOR as f32)
            * (inv_float_temperature * (self.max_size as f32).ln() + input_scaling.max()))
        .round() as Element;

        // So the minimum possible input is `quantised_min * multiplier + max_shift`, we multiply by `multiplier` so everything has scaling factor `SCALE_FACTOR`.
        let min_softmax_input = quantised_min * multiplier + max_shift;

        // The smallest 16 bits of `min_softmax_input` relate to values that are so small that after exponentiating they are so close to 1 that we just map them all to 1.
        // Due to this the bottom 16 bits gets sliced off and are just range checked, so for the actual softmax input we only need `min_softmax_input >> 16`.
        let significant_min_input = min_softmax_input >> 16;

        // Now we work out how many bits it takes to represent this number (it will always be less than zero so we take an abs() first)
        let min_input_bits = ceil_log2(significant_min_input.unsigned_abs() as usize);

        // Now we want to work out the value "bkm" such that anything with absolute value greater than bkm should just be mapped to zero
        // by the exponential. We will have K total tables, L of which are used for values that are so insignificant they get mapped to 1 and M of which
        // contain values that are all greater than bkm. We aim to make K - M - L = 1 because results from testing tell us that this allows
        // us to make an exp table with 17 variables which isn't too large (as it gets reused across every softmax in something like Multiheaded attention).
        let base: Element = 1 << 16;
        let (float_error, bkm_float) = calc_softmax_error(
            base,
            self.max_size as f32,
            OUTPUT_SCALE_FACTOR as f32,
            SCALE_FACTOR as f32,
            inv_float_temperature,
        );

        let float_error = float_error.abs();
        let bkm = bkm_float.round() as Element;
        // Now that we have bkm we set the Softmax table size as `ceil_log2(bkm as usize >> 16)` (which is 17 in practice)
        let softmax_table_size = ceil_log2(bkm as usize >> 16);
        // We also work out how many additional chunks we need to cover anything between bkm >> 16 and significant_min_input
        let (number_zero_chunks, zero_table_vars) = if min_input_bits > softmax_table_size {
            let remaining_bits = min_input_bits - softmax_table_size;
            // Here we ceiling divide
            let number_chunks = (remaining_bits - 1) / *quantization::BIT_LEN + 1;
            // If number of tables is 1 we check to see if we can use < softmax_table_size bits
            let zeroing_table_bit_size = remaining_bits % *quantization::BIT_LEN;
            (number_chunks, zeroing_table_bit_size)
        } else {
            (0usize, 0usize)
        };

        // Make the exp lookup table
        let table_data =
            SoftmaxTableData::new(inv_float_temperature.to_bits(), softmax_table_size, bkm);

        // Store all the quantised info for quantised evaluation
        let quant_info = QuantisedSoftmaxData {
            input_scale_factor: input_scaling,
            lut: table_data,
            error_bound: float_error,
            inv_float_temperature,
            bkm,
            number_zero_chunks,
            zero_table_vars,
        };

        // Return the quantised `Softmax` operator
        Ok(Softmax::<Element> {
            scalar: multiplier,
            max_size: self.max_size,
            quant_info: Some(quant_info),
        })
    }

    fn quant_info(&self) -> Option<&QuantisedSoftmaxData> {
        self.quant_info.as_ref()
    }
    pub fn with_scale(self, scale: N) -> Self {
        Self {
            scalar: scale,
            ..self
        }
    }
}

impl Softmax<Element> {
    /// Method that given a quantised input [`Tensor`] calculates the `shift` we apply along each dim and returns the result as the `bias` field of
    /// as [`AttentionMask`].
    pub(crate) fn calculate_shift_data(
        &self,
        input: &Tensor<Element>,
        unpadded_input_shape: &[usize],
    ) -> Result<(Tensor<Element>, AttentionMask<Element>)> {
        let QuantisedSoftmaxData {
            input_scale_factor,
            inv_float_temperature,
            bkm,
            ..
        } = self.quant_info().ok_or(anyhow!("Attempted to calculate shift data for quantised Softmax with no QuantisedSoftmaxData present"))?;

        // We need to calculate the shift we should apply together with the mask
        // To do this we:
        // 1. dequantise the input
        // 2. apply a float mask
        // 3. sum along the desired dim
        let negative_infinity = -((bkm >> 16) + 1) << 16;

        // New way is calculate shift row by row (as if a mask is being used)
        // apply shift
        // apply mask

        // We need a mask
        let final_dim = *input
            .shape()
            .last()
            .ok_or(anyhow!("Input tensor had no shape in quantised Softmax"))?;
        // We also need the second to last dim
        let second_dim = input.shape()[input.shape().len() - 2];
        let shift_data = if second_dim == 1 && second_dim != final_dim {
            input
                .get_data()
                .chunks(final_dim)
                .map(|chunk| {
                    let max = *chunk
                        .iter()
                        .take(*unpadded_input_shape.last().unwrap())
                        .max()
                        .unwrap();
                    let sum = chunk
                        .iter()
                        .take(*unpadded_input_shape.last().unwrap())
                        .map(|x| {
                            (input_scale_factor.dequantize(&(x - max)) / inv_float_temperature)
                                .exp()
                        })
                        .sum::<f32>();
                    let log_sum = sum.ln();
                    -(SCALE_FACTOR as f32 * inv_float_temperature * log_sum).round() as Element
                        - max * self.scalar
                })
                .collect::<Vec<Element>>()
        } else {
            input
                .get_data()
                .chunks(final_dim)
                .enumerate()
                .map(|(i, chunk)| {
                    // We add the check here to see if we are in the first row of a new channel, the first row has to be calculated
                    // differently so as to avoid getting rounding errors that lead to values we can't lookup.
                    if i % second_dim == 0 {
                        -chunk[0] * self.scalar
                    } else {
                        let max = *chunk.iter().take(i % second_dim + 1).max().unwrap();
                        let sum = chunk
                            .iter()
                            .take(i % second_dim + 1)
                            .map(|x| {
                                (input_scale_factor.dequantize(&(x - max)) / inv_float_temperature)
                                    .exp()
                            })
                            .sum::<f32>();
                        let log_sum = sum.ln();
                        -(SCALE_FACTOR as f32 * inv_float_temperature * log_sum).round() as Element
                            - max * self.scalar
                    }
                })
                .collect::<Vec<Element>>()
        };
        // Make a tensor for the shift data
        let shift_shape = input
            .shape()
            .iter()
            .take(unpadded_input_shape.len() - 1)
            .copied()
            .chain(std::iter::once(1usize))
            .collect::<Vec<usize>>();
        let shift_tensor = Tensor::<Element>::new(shift_shape.into(), shift_data);
        let mask = AttentionMask::<Element>::new(
            input.shape().as_slice(),
            unpadded_input_shape,
            negative_infinity,
        )?;

        Ok((shift_tensor, mask))
    }
}

/// Calculates the error as an [`f32`] when applying softmax as described in zkLLM.
/// This functions returns the error together with the value `bkm` such that anything smaller
/// than `bkm` should be mapped to zero.
pub(crate) fn calc_softmax_error(
    bl: Element,
    max_context_size: f32,
    output_sf: f32,
    input_sf: f32,
    temp: f32,
) -> (f32, f32) {
    // First we calculate the optimal point to map everything to zero (to minimise the L1 error)
    // we assume the total number of tables that don't map everything to 1 or 0 is exactly 1.
    let kml = 1.0f32;
    let bkm_multiplier = kml * (2.0f32 * max_context_size).ln() + output_sf.ln();
    let bkm = input_sf * temp * bkm_multiplier / (kml + 1.0f32);
    // Now that we have bkm we calculate the allowable float error
    let common_denom = kml * input_sf * temp;
    let first_term = (bl as f32 / common_denom).exp();
    let second_term = (bkm / common_denom).exp() / (2.0f32 * output_sf.powf(1.0 / kml));
    // This is the C constant referenced in the appendix of zkLLM
    let c = (first_term + second_term).powf(kml) - 1.0f32;
    // These terms are used to give the L1 error bound
    let term_one = c * (1.0f32 / (2.0f32 * input_sf * temp)).exp();
    let term_two = (max_context_size - 1.0f32) * (-bkm / input_sf * temp).exp();
    (term_one + term_two, bkm)
}

impl Evaluate<f32> for Softmax<f32> {
    fn evaluate<E: ExtensionField>(
        &self,
        inputs: &[&Tensor<f32>],
        unpadded_input_shapes: &[Shape],
    ) -> anyhow::Result<LayerOut<f32, E>> {
        ensure!(
            inputs.len() == 1,
            "softmax expects exactly one input tensor currently"
        );
        let input = inputs[0];
        // Make the attention mask
        let mask = AttentionMask::<f32>::new(
            &input.shape(),
            &unpadded_input_shapes[0],
            f32::NEG_INFINITY,
        )?;
        let masked_input = mask.apply(input)?;

        let chunk_size = *input
            .shape()
            .last()
            .ok_or(anyhow!("Input shape was empty for float Softmax"))?;
        let output = masked_input
            .get_data()
            .chunks(chunk_size)
            .flat_map(|vec| {
                let max: f32 = *vec
                    .iter()
                    .max_by(|i, j| i.partial_cmp(j).unwrap_or(std::cmp::Ordering::Less))
                    .unwrap();
                let scaled = vec
                    .iter()
                    .map(|x| {
                        if *x != f32::NEG_INFINITY {
                            self.scalar * (x - max)
                        } else {
                            *x
                        }
                    })
                    .map(|x| x.exp())
                    .collect::<Vec<_>>();
                let sum = scaled.iter().sum::<f32>();
                scaled.iter().map(|x| x / sum).collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let output_tensor = Tensor::new(input.shape(), output);
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

#[derive(Debug, Clone)]
/// Struct containing data useful for proving correctness of [`Softmax`]. This is data that we compute anyway
/// during quantised evaluation.
pub struct SoftmaxData<E>
where
    E: Clone + ExtensionField,
{
    /// This is the natural logarithm of the sum of the exponentiated input along the given dimension
    shift_tensor: Tensor<Element>,
    /// This is the input tensor after applying the shift
    shifted_input: Tensor<Element>,
    /// This is the mask used during the attention process
    mask: AttentionMask<Element>,
    /// The lowest 8-bits of the input (after rescaling)
    low_range_check: Vec<Element>,
    /// The second lowest 8 bits of the input (after rescaling)
    high_range_check: Vec<Element>,
    /// The inputs and outputs of the exponential lookup table
    exp_lookup: (Vec<Element>, Vec<Element>),
    /// The inputs and outputs of the most significant chunks lookups
    zero_table_lookups: (Vec<Vec<Element>>, Vec<Vec<Element>>),
    _phantom: PhantomData<E>,
}

impl<E: Clone + ExtensionField> Default for SoftmaxData<E> {
    fn default() -> Self {
        Self {
            shift_tensor: Tensor::<Element>::new(vec![].into(), vec![]),
            shifted_input: Tensor::<Element>::new(vec![].into(), vec![]),
            mask: AttentionMask::<Element>::default(),
            low_range_check: Vec::default(),
            high_range_check: Vec::default(),
            exp_lookup: (Vec::default(), Vec::default()),
            zero_table_lookups: (Vec::default(), Vec::default()),
            _phantom: PhantomData::<E>,
        }
    }
}

impl Evaluate<Element> for Softmax<Element> {
    fn evaluate<E: ExtensionField>(
        &self,
        inputs: &[&Tensor<Element>],
        unpadded_input_shapes: &[Shape],
    ) -> Result<LayerOut<Element, E>> {
        // First we heck that we have some quantisation info.
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
            number_zero_chunks,
            bkm,
            ..
        } = self.quant_info().unwrap();

        let input = inputs[0];
        let (shift_tensor, mask) = self.calculate_shift_data(input, &unpadded_input_shapes[0])?;

        let dim = *input.shape().last().ok_or(anyhow!(
            "Softmax input had no shape in quantised evaluation"
        ))?;
        let shifted_input_data = input
            .get_data()
            .chunks(dim)
            .zip(shift_tensor.get_data().iter())
            .flat_map(|(row, shift)| {
                // For each row we rescale the input to the correct scale factor and add the shift (its already been negated)
                row.iter()
                    .map(|elem| elem * self.scalar + shift)
                    .collect::<Vec<Element>>()
            })
            .collect::<Vec<Element>>();

        let shifted_input = Tensor::<Element>::new(input.shape(), shifted_input_data);
        // Apply the mask to the shifted input
        let masked_input = mask.apply(&shifted_input)?;

        // We use the mask to extract 8-bit chunks of the input, these are the smallest fractional bits
        // and so we can assume that they get mapped to 1 under `exp`
        let bit_mask: Element = 255;
        let softmax_table_vars = ceil_log2(*bkm as usize >> 16);
        let softmax_table_mask: Element = (1 << softmax_table_vars) - 1;
        let zero_table_mask: Element = (1 << *quantization::BIT_LEN) - 1;
        // Now we chunk the rescaled, masked input
        let mut low_range_check = Vec::<Element>::new();
        let mut high_range_check = Vec::<Element>::new();
        let mut lookups = Vec::<Element>::new();
        let mut outputs = Vec::<Element>::new();
        let mut zero_chunks_in: Vec<Vec<Element>> = vec![vec![]; *number_zero_chunks];
        let mut zero_chunks_out: Vec<Vec<Element>> = vec![vec![]; *number_zero_chunks];
        let mut softmax_outputs: Vec<Element> = Vec::<Element>::new();

        for input_elem in masked_input.get_data().iter() {
            // We take the absolute value as this is guaranteed to be negative or zero
            let mut rescaled = input_elem.abs();
            low_range_check.push(rescaled & bit_mask);
            rescaled >>= 8;
            high_range_check.push(rescaled & bit_mask);
            rescaled >>= 8;
            let lookup = rescaled & softmax_table_mask;
            let exp_output = lut.table_output(lookup);
            outputs.push(exp_output);
            lookups.push(lookup);
            rescaled >>= softmax_table_vars;
            // Now we iterate over the number of zero chunks, if any of these are non-zero the output of softmax should be 0 for this element.
            // We fold with initial input exp_output, at each step we append the zero chunk lookup values to their respective lists.
            let softmax_output = zero_chunks_in
                .iter_mut()
                .zip(zero_chunks_out.iter_mut())
                .fold(exp_output, |acc, (in_vec, out_vec)| {
                    let in_lookup = rescaled & zero_table_mask;
                    let out_lookup: Element = if in_lookup != 0 { 0 } else { 1 };
                    in_vec.push(in_lookup);
                    out_vec.push(out_lookup);
                    rescaled >>= *quantization::BIT_LEN;
                    acc * out_lookup
                });
            softmax_outputs.push(softmax_output);
        }

        // We store all the information that has been computed in this step that will be useful later for proving.
        let proving_data = ProvingData::Softmax(SoftmaxData {
            shift_tensor,
            shifted_input,
            mask,
            low_range_check,
            high_range_check,
            exp_lookup: (lookups, outputs),
            zero_table_lookups: (zero_chunks_in, zero_chunks_out),
            _phantom: PhantomData::<E>,
        });

        // Make the output tensor
        let output = Tensor::<Element>::new(input.shape(), softmax_outputs);

        Ok(LayerOut {
            outputs: vec![output],
            proving_data,
        })
    }
}

impl PadOp for Softmax<Element> {}

impl Softmax<Element> {
    #[allow(clippy::type_complexity)]
    pub(crate) fn prove_step<
        E: ExtensionField,
        PCS: PolynomialCommitmentScheme<E>,
        T: transcript::Transcript<E>,
    >(
        &self,
        node_id: NodeId,
        last_claims: Vec<&Claim<E>>,
        ctx: &SoftmaxCtx<E>,
        softmax_data: &SoftmaxData<E>,
        prover: &mut crate::Prover<E, T, PCS>,
    ) -> Result<(Vec<Claim<E>>, SoftmaxProof<E, PCS>)>
    where
        PCS::CommitmentWithWitness: Serialize + DeserializeOwned,
    {
        // Check number of claims
        ensure!(
            last_claims.len() == 1,
            "Softmax only produces one output claim but got: {}",
            last_claims.len()
        );
        let last_claim = last_claims[0];
        let final_dim_size = softmax_data
            .shifted_input
            .shape()
            .last()
            .ok_or(anyhow!("Shifted input has no shape"))?
            .next_power_of_two();
        // Retrieve all the witness data
        let layer_commitment = prover.lookup_witness(node_id)?;
        let logup_inputs = ctx.lookup_ctx.create_logup_inputs_softmax::<PCS, E>(
            layer_commitment,
            &prover.challenge_storage,
            final_dim_size,
        )?;
        let layer_polys = PCS::get_arc_mle_witness_from_commitment(layer_commitment);
        let commitment = PCS::get_pure_commitment(layer_commitment);
        // Run the logup proving
        let logup_batch_proof = batch_multiple_sizes_prove(&logup_inputs, prover.transcript)?;

        // Make the polynomials that aren't involved in the lookup but are involved in the sumcheck
        let mut mask = softmax_data.mask.clone();
        mask.pad()?;
        let shifted_input = softmax_data.shifted_input.pad_next_power_of_two();

        let tril_mle: MultilinearExtension<E> = to_base::<E, _>(mask.tril.get_data()).into_mle();
        let bias_mle: MultilinearExtension<E> = to_base::<E, _>(mask.bias.get_data()).into_mle();
        let shifted_input_mle: MultilinearExtension<E> =
            to_base::<E, _>(shifted_input.get_data()).into_mle();

        // The layer_polys will always be odd in length and we only need the ones after the zero input columns for the sumcheck to verify the output claim
        // the numbers 5,3 and 2 are here because the number of zero chunks can be variable but we always commit to 2 range checks, the input and output of exp and the normalisation shift.
        // So to work out the number of zero table related polys we do layer_polys.len() - 5.
        // Then when we commit we do it in the order
        // low_range_check, high_range_check, exp_input, zero_inputs, exp_output, zero_outputs, normalisation_shift,
        // so we need to skip always the first 3 polys and then number_zero_polys / 2 because thats how many zero_inputs there are.
        let number_zero_polys = layer_polys.len() - 5;
        let polys_to_skip = 3 + (number_zero_polys / 2);

        let logup_point = &logup_batch_proof.output_claims()[0].point;

        let dim_vars = ceil_log2(final_dim_size);
        let two = E::from_canonical_u64(2u64);
        let two_inv = two.inverse();
        let two_mul = E::from_canonical_u64(1u64 << dim_vars);

        // The error lookup is performed over the output summed on the final dimension so we need to extend the point used with correct number
        // of 2^-1 entries
        let full_error_point = std::iter::repeat_n(two_inv, dim_vars)
            .chain(logup_point.iter().skip(dim_vars).copied())
            .collect::<Vec<E>>();
        // Make all the eq polys
        let error_eq = compute_betas_eval(&full_error_point).into_mle();
        let logup_eq = compute_betas_eval(logup_point).into_mle();
        let last_claim_eq = compute_betas_eval(&last_claim.point).into_mle();

        // Transform the polys into Either::Left so they cna be passed to the VirtualPolynomialsBuilder
        let either_mles = layer_polys
            .iter()
            .skip(polys_to_skip)
            .take(polys_to_skip - 2)
            .map(|p| Either::Left(p.as_ref()))
            .chain(
                [
                    &shifted_input_mle,
                    &tril_mle,
                    &bias_mle,
                    &error_eq,
                    &last_claim_eq,
                    &logup_eq,
                ]
                .into_iter()
                .map(Either::Left),
            )
            .collect::<Vec<Either<_, _>>>();

        // Squeeze a batching challenge from the transcript
        let alpha = prover
            .transcript
            .sample_and_append_challenge(b"batching_challenge")
            .elements;
        // Make the VirtualPolynomials and run the sumcheck
        let num_vars = logup_point.len();
        let num_threads = optimal_sumcheck_threads(num_vars);
        let expr_builder =
            VirtualPolynomialsBuilder::<E>::new_with_mles(num_threads, num_vars, either_mles);
        let virtual_poly =
            expr_builder.to_virtual_polys(&ctx.sumcheck_expression, &[alpha, two_mul]);
        let (sumcheck_proof, state) = IOPProverState::<E>::prove(virtual_poly, prover.transcript);
        let sumcheck_point = state
            .challenges
            .iter()
            .map(|c| c.elements)
            .collect::<Vec<E>>();
        let all_evals = state.get_mle_flatten_final_evaluations();
        // Now we add the commitment claims to the commitment prover
        // the first commitment is the range evals, the exp input and the zero inputs (if there are any)
        let logup_claims = logup_batch_proof.output_claims();
        let first_commit_evals = logup_claims
            .iter()
            .take(polys_to_skip)
            .map(|claim| claim.eval)
            .collect::<Vec<E>>();
        let first_commit_point = logup_point.to_vec();
        // Get the evaluation for the shift
        let shift_point = sumcheck_point[dim_vars..].to_vec();
        let shift_eval = layer_polys
            .last()
            .map(|p| p.evaluate(&shift_point))
            .ok_or(anyhow!("Got no layer polys for Softmax proving"))?;
        // The second commitment is the exp output and the zero outputs
        let second_commit_evals = all_evals[..1 + number_zero_polys / 2].to_vec();
        let second_commit_point = sumcheck_point.clone();
        // COmbine them all in the correct order and add them to the claim prover
        let layer_claims = vec![
            (first_commit_point, first_commit_evals),
            (second_commit_point, second_commit_evals),
            (shift_point, vec![shift_eval]),
        ];
        prover.add_witness_claim(node_id, layer_claims);

        let field_scalar: E = self.scalar.to_field();
        let field_scalar_inverse = field_scalar.inverse();
        let input_claim = Claim::<E>::new(
            sumcheck_point,
            (all_evals[1 + number_zero_polys / 2] - shift_eval) * field_scalar_inverse,
        );
        let softmax_proof = SoftmaxProof {
            logup_proof: logup_batch_proof,
            commitment,
            sumcheck_proof,
            evaluations: [&all_evals[..1 + number_zero_polys / 2], &[shift_eval]].concat(),
        };

        Ok((vec![input_claim], softmax_proof))
    }

    pub(crate) fn lookup_witness<E: ExtensionField, PCS: PolynomialCommitmentScheme<E>>(
        &self,
        id: NodeId,
        ctx: &ProverContext<E, PCS>,
        output: &Tensor<Element>,
        softmax_data: &SoftmaxData<E>,
    ) -> Result<LookupWitnessGen<E, PCS>>
    where
        PCS::CommitmentWithWitness: Serialize + DeserializeOwned,
    {
        // Get the data generated during quantised evaluation
        let SoftmaxData {
            shift_tensor,
            low_range_check,
            high_range_check,
            exp_lookup: (exp_input, exp_output),
            zero_table_lookups: (zero_in, zero_out),
            ..
        } = softmax_data;

        // We need to work out how many chunks to split the normalisation into to be range checked.
        let QuantisedSoftmaxData {
            error_bound, lut, ..
        } = self.quant_info().ok_or(anyhow!(
            "Could not prove Softmax because it had no quantisation data"
        ))?;
        let allowable_error = (*error_bound * OUTPUT_SCALE_FACTOR as f32).round() as Element;

        // Now we construct the polynomials used in the lookups
        // To do this we need the size of the last dimension
        let final_dim_size = *output
            .shape()
            .last()
            .ok_or(anyhow!("Softmax output tensor did not have a shape"))?;
        let mut error_chunks = vec![];
        let normalisation_lookup = output
            .get_data()
            .chunks(final_dim_size)
            .enumerate()
            .map(|(i, chunk)| {
                let sum = chunk.iter().sum::<Element>();
                let quant_one = OUTPUT_SCALE_FACTOR as Element;
                if (sum < quant_one - allowable_error || sum > quant_one + allowable_error)
                    && sum != 0
                {
                    error_chunks.push(i);
                    // println!("Sum was {sum} on chunk {i}");
                    // chunk.iter().for_each(|v| println!("chunk value {v}"));
                }
                sum
            })
            .collect::<Vec<Element>>();

        let range_elements_count = count_elements(
            low_range_check
                .iter()
                .chain(high_range_check.iter())
                .cloned(),
        );
        let softman_elements_count = count_elements(
            exp_input
                .iter()
                .zip(exp_output.iter())
                .map(|(input, output)| input + output * COLUMN_SEPARATOR),
        );

        let zero_table_elements_count = count_elements(
            zero_in
                .iter()
                .zip(zero_out.iter())
                .flat_map(|(input, output)| input.iter().zip(output.iter()))
                .map(|(input, output)| input + output * COLUMN_SEPARATOR),
        );

        // We add zero table lookups if there are any
        // We make two rmms here even though all of these polys have the same size, this is because `exp_output` and all the `zero_out`
        // have to be used in an additional sumcheck and so will be evaluated at different points
        let width_1 = 3 + zero_in.len();
        let width_2 = 1 + zero_out.len();
        let poly_evals_one = transpose(
            [low_range_check, high_range_check, exp_input]
                .into_iter()
                .chain(zero_in)
                .cloned()
                .collect::<Vec<_>>(),
        );
        let poly_evals_two = transpose(
            [exp_output]
                .into_iter()
                .chain(zero_out)
                .cloned()
                .collect::<Vec<_>>(),
        );

        let rmm1 = RowMajorMatrix::<E::BaseField>::new_by_inner_matrix(
            ceno_p3::matrix::dense::DenseMatrix::new(
                to_base::<E, _>(poly_evals_one.into_iter().flatten()),
                width_1,
            ),
            witness::InstancePaddingStrategy::Default,
        );
        let rmm2 = RowMajorMatrix::<E::BaseField>::new_by_inner_matrix(
            ceno_p3::matrix::dense::DenseMatrix::new(
                to_base::<E, _>(poly_evals_two.into_iter().flatten()),
                width_2,
            ),
            witness::InstancePaddingStrategy::Default,
        );

        // Now we make the rmm for the error lookup and the shift data
        let shift_tensor = shift_tensor.pad_next_power_of_two();
        let small_evals_field = to_base::<E, _>(shift_tensor.get_data().iter());
        let small_rmm = RowMajorMatrix::<E::BaseField>::new_by_inner_matrix(
            ceno_p3::matrix::dense::DenseMatrix::new(small_evals_field, 1),
            witness::InstancePaddingStrategy::Default,
        );

        let layer_commit = ctx
            .commitment_ctx
            .batch_commit(vec![rmm1, rmm2, small_rmm])?;

        let mut gen_w = LookupWitnessGen::<E, PCS>::default();

        // Add the looked up values to the generator so we can make multiplicity polys later
        gen_w
            .element_count
            .insert(TableType::Range, range_elements_count);

        // Need to recreate the parameters for the Softmax table
        gen_w
            .element_count
            .insert(TableType::Softmax(*lut), softman_elements_count);

        let quant_one = OUTPUT_SCALE_FACTOR as Element;
        gen_w.element_count.insert(
            TableType::ErrorTable(quant_one, allowable_error),
            count_elements(normalisation_lookup),
        );

        if !zero_table_elements_count.is_empty() {
            gen_w
                .element_count
                .insert(TableType::ZeroTable, zero_table_elements_count);
        }

        gen_w.logup_witnesses.insert(id, layer_commit);
        Ok(gen_w)
    }
}

impl<E, PCS> ProvableOp<E, PCS> for Softmax<Element>
where
    E: ExtensionField + Serialize + DeserializeOwned,
    E::BaseField: Serialize + DeserializeOwned,
    PCS: PolynomialCommitmentScheme<E>,
    PCS::CommitmentWithWitness: Serialize + DeserializeOwned,
{
    type Ctx = SoftmaxCtx<E>;

    fn prove<T: transcript::Transcript<E>>(
        &self,
        node_id: NodeId,
        ctx: &Self::Ctx,
        last_claims: Vec<&Claim<E>>,
        step_data: &StepData<E, E>,
        prover: &mut crate::Prover<E, T, PCS>,
        _store: &mut TenStore,
    ) -> Result<Vec<Claim<E>>> {
        let softmax_data = step_data.node_outputs.try_softmax_data().ok_or(anyhow!(
            "Softmax LayerOut didn't have any ProvingData::Softmax"
        ))?;

        let (claims, proof) = self.prove_step(node_id, last_claims, ctx, softmax_data, prover)?;

        // Add the proof to the proof list
        prover.push_proof(node_id, LayerProof::<E, PCS>::Softmax(proof));

        Ok(claims)
    }

    fn gen_lookup_witness(
        &self,
        id: NodeId,
        ctx: &ProverContext<E, PCS>,
        step_data: &StepData<Element, E>,
        store: &mut TenStore,
    ) -> Result<LookupWitnessGen<E, PCS>> {
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
            "Softmax data not found in inference step for Sopftmax layer"
        ))?;
        self.lookup_witness(id, ctx, &output_tensors[0], softmax_data)
    }
}

impl QuantizeOp for Softmax<f32> {
    type QuantizedOp = Softmax<Element>;

    fn quantize_op<S: ScalingStrategy>(
        self,
        _data: &S::AuxData,
        _node_id: NodeId,
        input_scaling: &[ScalingFactor],
    ) -> anyhow::Result<QuantizeOutput<Self::QuantizedOp>> {
        ensure!(
            input_scaling.len() == 1,
            "More than one input scaling factor provided for Softmax. Received {} input scaling factor",
            input_scaling.len()
        );

        let quantised_op = self.quantise(input_scaling[0])?;

        // We want to keep track of the min and max output from this layer in floats. Softmax has to output values between 0.0 and 1.0
        // so we set max and min to these values. The scale is `1 / OUTPUT_SCALE_FACTOR` as this is what we multiply by to dequantise the quantised
        // outputs and the quantised domain is `(0.0 / scale, 1.0/ scale)`.
        let output_scaling = ScalingFactor::from_parts(
            1.0f32,
            0.0f32,
            1.0f32 / OUTPUT_SCALE_FACTOR as f32,
            (0, OUTPUT_SCALE_FACTOR as Element),
        );
        Ok(QuantizeOutput::<Softmax<Element>> {
            quantized_op: quantised_op,
            output_scalings: vec![output_scaling],
            requant_layer: None,
        })
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "E: ExtensionField + DeserializeOwned")]
pub struct SoftmaxCtx<E: ExtensionField> {
    node_id: NodeId,
    /// The absolute value of the allowable error
    allowable_error: Element,
    /// The value that determines when we map to zero in the exp lookup
    bkm: Element,
    /// The result of calling [`f32::to_bits`] on the temperature
    temperature_bits: u32,
    /// The number of variables used for the lookup table
    size: usize,
    /// The scalar multiplier used to ensure that the inputs have the correct scale factor
    scalar: Element,
    /// The number of lookups into the zero table
    number_zero_chunks: usize,
    /// The number of bits the zero table size is
    zero_table_vars: usize,
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
    ) -> anyhow::Result<Vec<LogUpInput<E>>>
    where
        E: ExtensionField,
        PCS: PolynomialCommitmentScheme<E>,
    {
        // First we extract the polynomials from the layer_commitment
        let polys = PCS::get_arc_mle_witness_from_commitment(layer_commitment);

        // There should be at least as many polynomials as there are lookup columns total
        let total_lookup_columns = self
            .tables
            .iter()
            .zip(self.instances_per_table.iter())
            .map(|(tt, &n)| tt.num_columns() * n)
            .sum::<usize>();

        ensure!(
            polys.len() >= total_lookup_columns,
            "Cannot create Softmax LogUp inputs because we were only provided with {} polynomials and expected {} lookup columns",
            polys.len(),
            total_lookup_columns
        );

        // We know the first 2 polys will always be the range checks and the third is always the exp input
        let (constant_challenge, column_separation_challenge) = challenge_storage
            .get_challenges_by_name(&self.tables[0].name())
            .ok_or(anyhow!(
                "No challenges found for Table {}, cannot generate Softmax LogUp input",
                self.tables[0].name()
            ))?;
        let column_evals = polys
            .iter()
            .take(2)
            .map(|p| p.get_base_field_vec().to_vec())
            .collect::<Vec<Vec<E::BaseField>>>();
        let range_input = LogUpInput::<E>::new_lookup(
            column_evals,
            constant_challenge,
            column_separation_challenge,
            self.tables[0].num_columns(),
        )?;

        let exp_column_evals = if self.tables.len() == 3 {
            polys
                .iter()
                .skip(2)
                .take(2)
                .map(|p| p.get_base_field_vec().to_vec())
                .collect::<Vec<Vec<E::BaseField>>>()
        } else {
            let number_zero_columns = self.instances_per_table[2];
            polys
                .iter()
                .skip(2)
                .step_by(1 + number_zero_columns)
                .take(2)
                .map(|p| p.get_base_field_vec().to_vec())
                .collect::<Vec<Vec<E::BaseField>>>()
        };
        let (constant_challenge, column_separation_challenge) = challenge_storage
            .get_challenges_by_name(&self.tables[1].name())
            .ok_or(anyhow!(
                "No challenges found for Table {}, cannot generate Softmax LogUp input",
                self.tables[1].name()
            ))?;
        let exp_input = LogUpInput::<E>::new_lookup(
            exp_column_evals,
            constant_challenge,
            column_separation_challenge,
            self.tables[1].num_columns(),
        )?;

        // Now we do the zero part and the error part
        let mut logup_inputs = vec![range_input, exp_input];

        if self.tables.len() == 4 {
            let number_zero_columns = self.instances_per_table[2];
            let zero_column_evals = polys
                .iter()
                .skip(3)
                .take(number_zero_columns)
                .interleave(
                    polys
                        .iter()
                        .skip(4 + number_zero_columns)
                        .take(number_zero_columns),
                )
                .map(|p| p.get_base_field_vec().to_vec())
                .collect::<Vec<Vec<E::BaseField>>>();
            let (zero_const_chal, zero_csc) = challenge_storage
                .get_challenges_by_name(&self.tables[2].name())
                .ok_or(anyhow!(
                    "No challenges found for Table {}, cannot generate Softmax LogUp input",
                    self.tables[2].name()
                ))?;

            let zero_logup_input = LogUpInput::<E>::new_lookup(
                zero_column_evals,
                zero_const_chal,
                zero_csc,
                self.tables[2].num_columns(),
            )?;
            let transposed = transpose(
                polys
                    .iter()
                    .skip(3 + number_zero_columns)
                    .take(1 + number_zero_columns)
                    .map(|p| p.get_base_field_vec().to_vec())
                    .collect::<Vec<Vec<E::BaseField>>>(),
            );
            let error_column_eval = transposed
                .into_iter()
                .map(|prod| prod.into_iter().product::<E::BaseField>())
                .chunks(dim_size)
                .into_iter()
                .map(|chunk| chunk.into_iter().sum::<E::BaseField>())
                .collect::<Vec<E::BaseField>>();
            let (error_const_chal, error_csc) = challenge_storage
                .get_challenges_by_name(&self.tables[3].name())
                .ok_or(anyhow!(
                    "No challenges found for Table {}, cannot generate Softmax LogUp input",
                    self.tables[3].name()
                ))?;
            let error_input = LogUpInput::<E>::new_lookup(
                vec![error_column_eval],
                error_const_chal,
                error_csc,
                self.tables[3].num_columns(),
            )?;
            logup_inputs.push(zero_logup_input);
            logup_inputs.push(error_input);
        } else {
            let error_column_eval = polys[3]
                .get_base_field_vec()
                .chunks(dim_size)
                .map(|chunk| chunk.iter().copied().sum::<E::BaseField>())
                .collect::<Vec<E::BaseField>>();
            let (error_const_chal, error_csc) = challenge_storage
                .get_challenges_by_name(&self.tables[2].name())
                .ok_or(anyhow!(
                    "No challenges found for Table {}, cannot generate Softmax LogUp input",
                    self.tables[2].name()
                ))?;
            let error_input = LogUpInput::<E>::new_lookup(
                vec![error_column_eval],
                error_const_chal,
                error_csc,
                self.tables[2].num_columns(),
            )?;
            logup_inputs.push(error_input);
        }

        Ok(logup_inputs)
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
        if let Some(quant_info) = self.quant_info() {
            let QuantisedSoftmaxData {
                lut,
                error_bound,
                inv_float_temperature,
                bkm,
                number_zero_chunks,
                zero_table_vars,
                ..
            } = quant_info;

            // We convert the `f32` to bits so that the compiler doesn't complain about trait implementations
            let float_temp_bits = inv_float_temperature.to_bits();
            // Calculate the allowable error in normalisation as an Element
            let allowable_error = (*error_bound * OUTPUT_SCALE_FACTOR as f32).round() as Element;

            // Add the tables that Softmax requires
            aux.tables.insert(TableType::Range);
            aux.tables.insert(TableType::Softmax(*lut));
            aux.tables.insert(TableType::ErrorTable(
                OUTPUT_SCALE_FACTOR as Element,
                allowable_error,
            ));

            // If there is one add the ZeroTable
            let lookup_ctx = if !number_zero_chunks.is_zero() {
                aux.tables.insert(TableType::ZeroTable);
                let tables = vec![
                    TableType::Range,
                    TableType::Softmax(*lut),
                    TableType::ZeroTable,
                    TableType::ErrorTable(OUTPUT_SCALE_FACTOR as Element, allowable_error),
                ];
                let instances_per_table = vec![2, 1, *number_zero_chunks, 1];
                LayerLookupContext::new(tables, instances_per_table)
            } else {
                let tables = vec![
                    TableType::Range,
                    TableType::Softmax(*lut),
                    TableType::ErrorTable(OUTPUT_SCALE_FACTOR as Element, allowable_error),
                ];
                let instances_per_table = vec![2, 1, 1];
                LayerLookupContext::new(tables, instances_per_table)
            };

            // There are no common commitments for this layer
            aux.model_polys = None;
            aux.max_poly_len = aux
                .last_output_shape
                .iter()
                .fold(aux.max_poly_len, |acc, shapes| {
                    acc.max(shapes.next_power_of_two().product())
                });

            let expr = build_softmax_sumcheck_expression::<E>(*number_zero_chunks);

            // The output shape is the same as the input shape so we don't need to update it
            // return the LayerCtx and the updated ContextAux
            Ok((
                LayerCtx::Softmax(SoftmaxCtx {
                    node_id: id,
                    allowable_error,
                    bkm: *bkm,
                    temperature_bits: float_temp_bits,
                    size: lut.full_table_size() as usize,
                    scalar: self.scalar,
                    number_zero_chunks: *number_zero_chunks,
                    zero_table_vars: *zero_table_vars,
                    lookup_ctx,
                    sumcheck_expression: vec![expr],
                }),
                aux,
            ))
        } else {
            Err(anyhow!(
                "Softmax operation has not been quantised so no proving info available"
            ))
        }
    }
}

/// Builds the [`Expression`] used in [`Softmax`] proving to link lookup inputs and outputs to Layer inputs and outputs.
/// We have to show that the normalisation error is within the acceptable range, that `last_claim.eval` relates to the correct combination of the outputs
/// of the `exp` lookup and the `zero` lookups and also that the inputs to the lookups came from masking the shifted layer input.
fn build_softmax_sumcheck_expression<E: ExtensionField>(
    number_zero_chunks: usize,
) -> Expression<E> {
    // The first polynomial is the exp_output, followed by the zero outputs if there are any, then shifted input, then tril, then bias, then eq_polys
    let (output_expr, lookup_linking_expr) = if !number_zero_chunks.is_zero() {
        (0..number_zero_chunks).fold(
            (
                Expression::WitIn(0),
                Expression::WitIn(0) * Expression::Challenge(0, 3, E::ONE, E::ZERO),
            ),
            |(prod_acc, sum_acc), j| {
                (
                    prod_acc * Expression::WitIn(j as u16 + 1),
                    sum_acc
                        + Expression::WitIn(j as u16 + 1)
                            * Expression::Challenge(0, 4 + j, E::ONE, E::ZERO),
                )
            },
        )
    } else {
        (
            Expression::WitIn(0),
            Expression::WitIn(0) * Expression::Challenge(0, 3, E::ONE, E::ZERO),
        )
    };

    let start_id = (number_zero_chunks + 1) as u16;
    let mask_expr = Expression::WitIn(start_id) * Expression::WitIn(start_id + 1)
        + Expression::WitIn(start_id + 2);

    let error_eq = Expression::WitIn(start_id + 3);
    let last_claim_eq = Expression::WitIn(start_id + 4);
    let logup_eq = Expression::WitIn(start_id + 5);

    output_expr
        * (Expression::Challenge(1, 1, E::ONE, E::ZERO) * error_eq
            + Expression::Challenge(0, 1, E::ONE, E::ZERO) * last_claim_eq)
        + logup_eq
            * (Expression::Challenge(0, 2, E::ONE, E::ZERO) * mask_expr + lookup_linking_expr)
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

        let last_claim = last_claims[0];
        let SoftmaxProof {
            logup_proof,
            commitment,
            sumcheck_proof,
            evaluations,
        } = proof;

        // Verify the lookup proof
        let batch_claim = verify_logup_proof_multiple_sizes(logup_proof, verifier.transcript)?;
        self.lookup_ctx
            .verify_logup_batch_claim(&batch_claim, &verifier.challenge_storage)?;

        // Now we squeeze the batching challenge
        let alpha = verifier
            .transcript
            .sample_and_append_challenge(b"batching_challenge")
            .elements;

        // poly_evals will be in the order low_range_check, high_range_check, exp_in, exp_out, (zero_in, zero_out)_i, error
        let poly_evals = batch_claim.poly_evals();

        let low_range = poly_evals[0];
        let high_range = poly_evals[1];
        let exp_in = poly_evals[2];
        let exp_out = poly_evals[3];

        let (zero_in_evals, zero_out_evals): (Vec<E>, Vec<E>) = poly_evals[4..poly_evals.len() - 1]
            .chunks(2)
            .map(|chunk| (chunk[0], chunk[1]))
            .unzip();
        let error_eval = poly_evals[poly_evals.len() - 1];

        // Now we work out the claimed input for the sumcheck
        let two_to_the_16 = E::from_canonical_u64(1u64 << 16);
        let two_to_the_8 = E::from_canonical_u64(1u64 << 8);

        let initial_for_fold = low_range + high_range * two_to_the_8 + exp_in * two_to_the_16;
        let softmax_table_vars = ceil_log2(self.bkm as usize >> 16);
        let zero_table_init_multiplier = E::from_canonical_u64(1u64 << (16 + softmax_table_vars));
        let zero_table_size = E::from_canonical_u64(1u64 << *quantization::BIT_LEN);
        let shifted_input_claim = zero_in_evals
            .iter()
            .fold(
                (initial_for_fold, zero_table_init_multiplier),
                |(acc, mult_acc), &e| (acc + mult_acc * e, mult_acc * zero_table_size),
            )
            .0;

        let linking_challenge = alpha * alpha * alpha;
        let (lookup_linking, _) = zero_out_evals.iter().fold(
            (linking_challenge * exp_out, linking_challenge * alpha),
            |(eval_acc, chal_acc), &e| (eval_acc + chal_acc * e, chal_acc * alpha),
        );
        let claimed_sum =
            error_eval + alpha * (last_claim.eval - alpha * shifted_input_claim) + lookup_linking;
        let aux_info = VPAuxInfo {
            max_num_variables: batch_claim.point().len(),
            max_degree: (2 + zero_out_evals.len()).max(3),
            ..Default::default()
        };

        let subclaim = IOPVerifierState::<E>::verify(
            claimed_sum,
            sumcheck_proof,
            &aux_info,
            verifier.transcript,
        );
        let sumcheck_point = subclaim
            .point
            .iter()
            .map(|c| c.elements)
            .collect::<Vec<E>>();

        let padded_shape = &shape_step.padded_output_shape[0];
        let num_dims = padded_shape.len();
        let dim_vars = ceil_log2(padded_shape[num_dims - 1]);
        let last_claim_eq = identity_eval(&last_claim.point, &sumcheck_point);
        let logup_eq = identity_eval(batch_claim.point(), &sumcheck_point);

        let two_inv = E::TWO.inverse();
        let two_mul = E::from_canonical_u64(1u64 << dim_vars);

        let full_error_point = std::iter::repeat_n(two_inv, dim_vars)
            .chain(batch_claim.point().iter().skip(dim_vars).copied())
            .collect::<Vec<E>>();
        let error_eq = identity_eval(&full_error_point, &sumcheck_point);

        let output_part = evaluations
            .iter()
            .take(evaluations.len() - 1)
            .copied()
            .product::<E>()
            * (error_eq * two_mul + alpha * last_claim_eq);

        let linking_challenge = alpha * alpha * alpha;
        let linking_part = logup_eq
            * evaluations
                .iter()
                .take(evaluations.len() - 1)
                .fold((E::ZERO, linking_challenge), |(acc, chal_acc), &e| {
                    (acc + chal_acc * e, chal_acc * alpha)
                })
                .0;
        // Calculate the tril and bias evaluations

        let rows = ceil_log2(padded_shape[num_dims - 2]);
        let columns = dim_vars;
        let column_point = sumcheck_point
            .iter()
            .take(columns)
            .copied()
            .collect::<Vec<E>>();
        let row_point = sumcheck_point
            .iter()
            .skip(columns)
            .take(rows)
            .copied()
            .collect::<Vec<E>>();
        let tril_eval = eval_zeroifier_mle(&column_point, &row_point);
        let negative_infinity: E = (-((self.bkm >> 16) + 1) << 16).to_field();
        let bias_eval = negative_infinity * (E::ONE - tril_eval);
        let mult_tril = alpha * alpha * logup_eq * tril_eval;
        let mult_bias = alpha * alpha * logup_eq * bias_eval;
        let mult_inv = mult_tril.inverse();

        // Now the shifted input eval is `(sumcheck_subclaim - mult_bias) * mult_inv`
        let shifted_input_eval =
            (subclaim.expected_evaluation - output_part - linking_part - mult_bias) * mult_inv;
        // To get the output claim eval we subtract the shift eval and multiply by the inverse of `self.scalar`
        let shift_eval = evaluations[evaluations.len() - 1];
        let field_scalar: E = self.scalar.to_field();
        let input_eval = (shifted_input_eval - shift_eval) * field_scalar.inverse();

        let first_comm_claim = (
            batch_claim.point().to_vec(),
            [&[low_range, high_range, exp_in], zero_in_evals.as_slice()].concat(),
        );
        let second_comm_claim = (
            sumcheck_point.clone(),
            evaluations[..evaluations.len() - 1].to_vec(),
        );
        let shift_claim = (sumcheck_point[dim_vars..].to_vec(), vec![shift_eval]);

        verifier.commit_verifier.add_witness_claim(
            self.node_id,
            commitment.clone(),
            vec![first_comm_claim, second_comm_claim, shift_claim],
        );

        Ok(vec![Claim::<E>::new(sumcheck_point.clone(), input_eval)])
    }

    fn write_proof_to_transcript<T: Transcript<E>>(
        &self,
        proof: &Self::Proof,
        transcript: &mut T,
    ) -> anyhow::Result<()> {
        proof.write_commitment(transcript)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
/// Mask used in attention so that tokens can only see "previous" values.
pub struct AttentionMask<N> {
    /// This is the tensor we multiply elementwise to zero out the correct locations
    pub tril: Tensor<N>,
    /// This is the bias we add elementwise to ensure all zeroes are replaced with `-inf`
    pub bias: Tensor<N>,
    /// The value for negative infinity
    negative_infinity: N,
}

impl<N: Number> Default for AttentionMask<N> {
    fn default() -> Self {
        AttentionMask {
            tril: Tensor::<N>::new(vec![].into(), vec![]),
            bias: Tensor::<N>::new(vec![].into(), vec![]),
            negative_infinity: N::MIN,
        }
    }
}

impl<N: Number> AttentionMask<N> {
    /// Creates a new mask given the unpadded input shape and the value to use for `-inf`
    pub fn new(
        shape: &[usize],
        unpadded_shape: &[usize],
        negative_inf: N,
    ) -> Result<AttentionMask<N>> {
        // The input shape should have length either 2 or 3 and the final 2 dimensions should be equal
        let num_dims = unpadded_shape.len();

        let correct_num_dims = num_dims == 2 || num_dims == 3;
        ensure!(
            correct_num_dims,
            "In order to create an Attention Mask the input should have either 2 or 3 dimensions, got: {}",
            num_dims
        );

        // Now check that either the final two dimensions are the same or the second to last dimension is 1
        let dims_equal = unpadded_shape[num_dims - 2] == unpadded_shape[num_dims - 1];
        let single_token = unpadded_shape[num_dims - 2] == 1;

        ensure!(
            dims_equal || single_token,
            "Final two dimensions should be equal, got: second to last: {}, last: {}",
            unpadded_shape[num_dims - 2],
            unpadded_shape[num_dims - 1]
        );

        // Now that we know all the dimensions line up make the lower triangular tensor
        let shape = if num_dims == 2 {
            let mut shape = shape.to_vec();
            shape.insert(0, 1);
            shape
        } else {
            shape.to_vec()
        };

        // If we only have a single token we only need to mask the padding (if there is any)
        if single_token {
            let tril_single_row = std::iter::repeat_n(N::unit(), *unpadded_shape.last().unwrap())
                .chain(std::iter::repeat(N::default()))
                .take(*shape.last().unwrap())
                .collect::<Vec<N>>();
            let tril_data = vec![tril_single_row.clone(); shape[0]].concat();
            let bias_single_row =
                std::iter::repeat_n(N::default(), *unpadded_shape.last().unwrap())
                    .chain(std::iter::repeat(negative_inf))
                    .take(*shape.last().unwrap())
                    .collect::<Vec<N>>();
            let bias_data = vec![bias_single_row; shape[0]].concat();
            let tril = Tensor::<N>::new(shape.clone().into(), tril_data);

            let bias = Tensor::<N>::new(shape.into(), bias_data);

            Ok(AttentionMask {
                tril,
                bias,
                negative_infinity: negative_inf,
            })
        } else {
            // Make the tril and bias tensor
            let tril = Tensor::<N>::tril(shape[2], shape[0], 0);

            let bias = Tensor::<N>::tri(shape[2], shape[0], 0, N::default(), negative_inf);

            Ok(AttentionMask {
                tril,
                bias,
                negative_infinity: negative_inf,
            })
        }
    }

    /// Pads the [`AttentionMask`] for proving purposes
    fn pad(&mut self) -> Result<()> {
        // First check that the bias and tril shapes agree
        let shapes_equal = self
            .tril
            .shape()
            .iter()
            .zip(self.bias.shape().iter())
            .all(|(t, b)| *t == *b);
        ensure!(
            shapes_equal,
            "Can't pad Attention Mask as tril and bias had different shapes"
        );

        // Now we check to see if everything is already a power of two
        if self.tril.shape().iter().all(|s| s.is_power_of_two()) {
            return Ok(());
        }

        // Calculate padded tensors
        // For tril and bias we just expand to a larger lower/upper triangular matrix
        let padded_shape = self
            .bias
            .shape()
            .iter()
            .map(|dim| dim.next_power_of_two())
            .collect::<Vec<usize>>();
        self.tril = Tensor::<N>::tril(padded_shape[2], padded_shape[0], 0);
        self.bias = Tensor::<N>::tri(
            padded_shape[2],
            padded_shape[0],
            0,
            N::default(),
            self.negative_infinity,
        );

        Ok(())
    }

    /// Apply the mask to an input, this method allows the input to have two or three dims and adjusts accordingly.
    /// It elementwise multiplies by `self.tril` and then adds `self.bias`.
    fn apply(&self, input: &Tensor<N>) -> Result<Tensor<N>> {
        // Check the the input has 2 or 3 dims
        let num_input_dims = input.rank();
        ensure!(
            num_input_dims == 2 || num_input_dims == 3,
            "To apply Attention Mask input need to have 2 or 3 dims, got: {}",
            num_input_dims
        );
        // If the input only has 2 dims reshape to have 3
        if num_input_dims == 3 {
            if !input
                .shape()
                .iter()
                .zip(self.tril.shape().iter())
                .all(|(a, b)| *a == *b)
            {
                return Err(anyhow!(
                    "Cannot apply attention mask, input did not have the same shape as mask"
                ));
            }

            Ok(input.mul(&self.tril).add(&self.bias))
        } else {
            let new_shape = input.shape().insert(0, 1);
            let new_input = input.clone().reshape(new_shape);

            if !new_input
                .shape()
                .iter()
                .zip(self.tril.shape().iter())
                .all(|(a, b)| *a == *b)
            {
                return Err(anyhow!(
                    "Cannot apply attention mask, input did not have the same shape as mask"
                ));
            }

            let output = new_input.mul(&self.tril).add(&self.bias);
            Ok(output.reshape(input.shape()))
        }
    }
}

#[cfg(test)]
mod tests {

    use ff_ext::GoldilocksExt2;

    use crate::{
        Tensor,
        layers::Layer,
        model::{Model, test::prove_model},
        padding::PaddingMode,
    };

    use super::*;

    #[test]
    fn test_softmax() {
        let softmax = Softmax::default();
        let input = Tensor::new(
            vec![1, 3, 3].into(),
            vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        );
        let output = softmax
            .evaluate::<GoldilocksExt2>(&[&input], &[vec![1, 3, 3].into()])
            .unwrap();
        assert_eq!(output.outputs[0].shape(), vec![1, 3, 3].into());

        output.outputs[0].get_data().chunks(3).for_each(|chunk| {
            assert_eq!(chunk.iter().sum::<f32>(), 1.0);
        });
    }

    #[test]
    fn test_quantise() {
        // For now we test with GPT2 like parameters
        let scale = 1.0f32 / 768.0f32.sqrt();
        let softmax = Softmax::<f32>::new_with_scale(scale, 1024);

        for num_tokens in 1015..1016 {
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

            let test_q_quant = test_q.clone().quantize(&q_scaling);
            let test_k_quant = test_k.clone().quantize(&k_scaling);

            let test_qk_quant = test_q_quant.matmul(&test_k_quant);

            let test_qk_dequant = test_qk_quant.dequantize(&qk_scaling);

            // Now to test the quantised softmax we quantise `float_input` and run the quantised evaluation.
            // We also quantise and dequantise `float_input` and run this data through the float evaluation and then compare the two results.

            let quant_softmax = softmax.quantise(qk_scaling).unwrap();

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

            for (q_chunk, f_chunk) in quant_output.outputs[0]
                .get_data()
                .chunks(num_tokens)
                .zip(dequant_output.outputs[0].get_data().chunks(num_tokens))
            {
                for (&q, f) in q_chunk.iter().zip(f_chunk.iter()) {
                    let float_q = q as f32 / OUTPUT_SCALE_FACTOR as f32;

                    let quant_dequant_diff = (float_q - f).abs();

                    // Make sure we are always within 1/100 th of the actual value
                    assert!(
                        quant_dequant_diff < 0.01,
                        "quant dequant diff was too large got: {quant_dequant_diff}"
                    );
                }
            }

            let max_error =
                quant_softmax.quant_info.as_ref().unwrap().error_bound * OUTPUT_SCALE_FACTOR as f32;

            quant_output.outputs[0]
                .get_data()
                .chunks(num_tokens)
                .for_each(|chunk| {
                    let row_sum = chunk.iter().sum::<Element>();

                    let diff_from_one = (row_sum - OUTPUT_SCALE_FACTOR as Element).abs();

                    assert!(diff_from_one < max_error.round() as Element);
                });
        }
    }

    #[test]
    fn test_softmax_with_scale() {
        let softmax = Softmax::new_with_scale(1.0 / 2.0, 1024);
        let input = Tensor::new(
            vec![3, 3].into(),
            vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        );
        let output = softmax
            .evaluate::<GoldilocksExt2>(&[&input], &[vec![3, 3].into()])
            .unwrap();
        // Since this is a masked evaluation, each row should sum to 1 and the first row should have 1 non-zero value, the second two non-zero
        // and so on.
        assert_eq!(
            output.outputs[0].get_data(),
            vec![
                1.0,
                0.0,
                0.0,
                0.5,
                0.5,
                0.0,
                1.0 / 3.0,
                1.0 / 3.0,
                1.0 / 3.0,
            ]
        );
    }

    #[test]
    fn test_softmax_proving() {
        let input_shape = vec![12, 200, 200];

        let mut model =
            Model::new_from_input_shapes(vec![input_shape.into()], PaddingMode::NoPadding);

        let softmax = Softmax::<f32>::new_with_scale(1.0f32 / 768.0f32.sqrt(), 1024);

        let _ = model
            .add_consecutive_layer(Layer::Softmax(softmax), None)
            .unwrap();

        model.route_output(None).unwrap();
        model.describe();
        prove_model(model, &mut TenStore::default()).unwrap();
    }
}
