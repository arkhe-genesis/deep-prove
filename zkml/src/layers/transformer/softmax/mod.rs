//! This layer applies the softmax function to the last dimension of the input tensor
use crate::{
    Claim, Element, NextPowerOfTwo, Number, ScalingStrategy, Shape, Tensor,
    graph::NodeId,
    iop::{
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
        transformer::ConcatenationCache,
    },
    lookup::{
        context::LookupWitnessGen,
        logup_gkr::structs::LogUpBatchProof,
        table::{Table, ZERO_CHECK_TABLE_BIT_SIZE},
    },
    model::{Step, transform::impls::softmax_mask::SoftmaxMaskTransform},
    padding::PaddingMode,
    quantization::{self, ScalingFactor},
    tensor::{TensorHandle, TensorTypeParam, WrappedTensor},
    to_base,
};
use anyhow::{Result, anyhow, bail, ensure};

use ff_ext::ExtensionField;

use mpcs::PolynomialCommitmentScheme;
use multilinear_extensions::util::{ceil_log2, transpose};

use serde::{Deserialize, Serialize, de::DeserializeOwned};
use std::{
    fmt::Debug,
    sync::{Arc, Mutex},
};
use sumcheck::structs::IOPProof;
use tenstore::StorageKey;
use transcript::Transcript;
use witness::RowMajorMatrix;

pub mod evaluate;
pub mod lookup;
pub mod prove;
pub mod quantise;
pub mod verify;

/// The short name used to identify the Softmax layer
pub const SOFTMAX_LAYER: &str = "SFTM";

fn default_shift_cache<N: TensorTypeParam>() -> Arc<Mutex<ConcatenationCache<N>>> {
    Arc::new(Mutex::new(ConcatenationCache::new_dynamic(
        -2,
        PaddingMode::NoPadding,
    )))
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(bound(serialize = "N: Serialize", deserialize = "N: DeserializeOwned"))]
/// Stores data about the Softmax operation, which is used to map a tensor of values to a tensor of probability distributions.
/// This is done by picking a dimension to normalise over and calculating
///             `x -> exp(scale * x) / (\sum_{i \in dim} exp(scale * x_{i}))`.
pub struct Softmax<N>
where
    N: TensorTypeParam,
{
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
    /// Transient shift cache populated by a full-trace call (`dim[-2] > 1`) and consumed by
    /// subsequent cached-trace calls (`dim[-2] == 1`). Skipped during (de)serialisation so
    /// that it is always reset to `None` on load; the next full-trace call repopulates it.
    #[serde(skip, default = "default_shift_cache")]
    shift_cache: Arc<Mutex<ConcatenationCache<N>>>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Copy)]
/// This struct is used to store information used when evaluating the quantised version of [`Softmax`] on
/// [`Element`]s.
pub(crate) struct QuantisedSoftmaxData {
    /// After multiplying by `self.fixed_point_multiplier` the value need to be shifted by this plus 25.
    pub right_shift: usize,
    /// The normalised scaling factor including temperature rescaling represented as a fixed point multiplier (it should have 24 fractional bits)
    pub fixed_point_multiplier: Element,
    /// The intermediate bit size, allowing us to work out how many zero tables we need
    pub intermediate_bit_size: usize,
    /// This stores the [`ExpTable`]
    pub(crate) lut: Table,
    /// The value of quantised negative infinity
    pub negative_infinity: Element,
    /// The error bound as calculated by the formulae given in the zkLLM paper, this is the relative error bound on the normalisation sum.
    error_bound: f32,
    /// The original [`ScalingFactor`] of the input
    input_scaling_factor: ScalingFactor,
    /// The temperature
    temperature: f32,
}

impl QuantisedSoftmaxData {
    /// Calculates the largest value that will be mapped to zero in quantised evaluation of Softmax.
    pub(crate) fn quantised_negative_infinity(&self) -> Element {
        self.negative_infinity
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
    /// The shift tensor evaluations
    pub(crate) shift_evaluations: Vec<E>,
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

    pub(crate) fn attach_store(&mut self, store: tenstore::GenStore) {
        self.shift_handle.attach_store(store);
    }

    pub(crate) fn isolate(&self) -> SoftmaxHandle {
        Self {
            shift_handle: self.shift_handle.isolate(),
        }
    }

    pub(crate) fn to_proving_data(&self) -> Result<SoftmaxData> {
        let shift_native = self.shift_handle.tensor()?;
        let shift_tensor = WrappedTensor::try_from(shift_native.as_ref())?;
        Ok(SoftmaxData { shift_tensor })
    }
}

impl<N: TensorTypeParam> Softmax<N> {
    pub fn new(context_length: usize) -> Self {
        Softmax {
            scalar: N::unit(),
            max_size: context_length,
            quant_info: None,
            shift_cache: Arc::new(Mutex::new(ConcatenationCache::<N>::new_dynamic(
                -2,
                PaddingMode::NoPadding,
            ))),
        }
    }

    pub fn new_with_scale(scale: N, max_context_size: usize) -> Softmax<N> {
        Softmax {
            scalar: scale,
            max_size: max_context_size,
            quant_info: None,
            shift_cache: Arc::new(Mutex::new(ConcatenationCache::<N>::new_dynamic(
                -2,
                PaddingMode::NoPadding,
            ))),
        }
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

    /// Returns whether the shift cache has been initialised
    pub fn shift_cache_initialised(&self) -> bool {
        self.shift_cache.lock().unwrap().is_initialized()
    }

    /// Appends data to the [shift_cache][Self::shift_cache]
    pub fn update_shift_cache(&self, shift_tensor: WrappedTensor<N>) -> Result<()> {
        let mut cache = self.shift_cache.lock().unwrap();
        let _ = cache.concatenate(shift_tensor)?;
        Ok(())
    }

    /// Resets the [shift_cache][Self::shift_cache]
    pub fn reset_cache(&self) {
        self.shift_cache.lock().unwrap().reset();
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

impl<N: TensorTypeParam> OpInfo for Softmax<N> {
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
        _ctx: &Self::Ctx,
        last_claims: Vec<&Claim<E>>,
        step_data: &Step<Element>,
        prover: &mut crate::Prover<E, T, PCS>,
    ) -> Result<Vec<Claim<E>>> {
        self.prove_internal(node_id, last_claims, step_data, prover)
    }

    fn gen_lookup_witness(
        &self,
        id: NodeId,
        ctx: &ProverContext<E, PCS>,
        step_data: &Step<Element>,
    ) -> Result<LookupWitnessGen<E, PCS>> {
        self.lookup_witness(id, ctx, step_data)
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

        let quantised_op = self.quantise(input_scaling[0])?;
        let output_sf = quantised_op
            .quant_info
            .map(|info| info.lut.output_scale_factor())
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
}

impl SoftmaxCtx {
    pub fn lookup_tables(&self) -> Vec<Table> {
        vec![
            Table::new_shift_check(),
            self.quant_info.lut,
            Table::new_zero_check(),
        ]
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
        if let Some(quant_info) = self.quant_info() {
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
                    quant_info: *quant_info,
                }),
                aux,
            ))
        } else {
            bail!("Softmax operation has not been quantised so no proving info available");
        }
    }
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

#[cfg(test)]
mod tests {
    use core::f32;
    use tenstore::GenStore;

    use crate::{
        Tensor,
        layers::{Layer, transformer::attention_mask::AttentionMask},
        model::{Model, test::prove_model},
        padding::PaddingMode,
        quantization::Quantize,
    };

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
            let intermediate_bit_size = 2 * (*quantization::BIT_LEN - 1) + ceil_log2(768);
            let qk_scaling = ScalingFactor::from_scale(
                q_scaling.scale() * k_scaling.scale(),
                Some((-1 << intermediate_bit_size, 1 << intermediate_bit_size)),
            );

            let test_q_quant = test_q.quantize(&q_scaling);
            let test_k_quant = test_k.quantize(&k_scaling);

            let test_qk_quant = test_q_quant.matmul(&test_k_quant).unwrap();

            // let test_qk_dequant = test_qk_quant.dequantize(&qk_scaling);

            // Now to test the quantised softmax we quantise `float_input` and run the quantised evaluation.
            // We also quantise and dequantise `float_input` and run this data through the float evaluation and then compare the two results.

            let quant_softmax = softmax.quantise(qk_scaling).unwrap();

            // We have to quatize and dequantize the the float input with respect to the table scaling factor
            let QuantisedSoftmaxData {
                lut, error_bound, ..
            } = quant_softmax.quant_info.as_ref().unwrap();
            // let test_qk_dequant = test_qk_dequant
            //     .as_wrapped()
            //     .mul_scalar(lut.input_scale_factor())
            //     .round()
            //     .div_scalar(lut.input_scale_factor())
            //     .to_native();

            // Obtain the quantised output
            let quant_output = quant_softmax
                .evaluate(&[&test_qk_quant.as_wrapped()])
                .unwrap();
            // // The result of running the quantised input as floats
            // let dequant_output = softmax.evaluate(&[&test_qk_dequant.as_wrapped()]).unwrap();

            // // The relative error comes from quantising the shift factor
            // // The absolute error comes from the tables output scale factor
            // let input_error_factor = (1.0 / (2.0f32 * lut.input_scale_factor())).exp() - 1.0;
            // let table_max_value: Element = 1 + (-1 << lut.table_bit_size());

            // let val_too_large_error = (table_max_value as f32 / lut.input_scale_factor()).exp();
            // let table_rounding = 1.0 / (2.0 * lut.output_scale_factor());
            // let other_error_part = table_rounding.max(val_too_large_error);

            // let rel_error = input_error_factor + other_error_part + f32::EPSILON;
            // let out_error = 1.0 / (2.0f32 * lut.output_scale_factor()) + f32::EPSILON;

            // let shape = quant_output.outputs[0].shape();
            // let multicartesian_product = shape.iter().map(|&d| 0..d).multi_cartesian_product();
            // for (&q, f, coord) in izip!(
            //     quant_output.outputs[0].get_data().iter(),
            //     dequant_output.outputs[0].get_data().iter(),
            //     multicartesian_product
            // ) {
            //     let float_q = q as f32 / lut.output_scale_factor();

            //     let quant_dequant_diff = (float_q - f).abs();
            //     assert!(
            //         is_close_with_tolerance(&[float_q], &[*f], out_error, rel_error),
            //         "Quant dequant diff was larger than expected at coord {coord:?} got: {quant_dequant_diff}, quant {float_q}, orig {f}, expected less than {}, table step size: {}",
            //         *f * rel_error + out_error,
            //         lut.output_scale_factor().recip()
            //     );
            // }

            let max_error = error_bound * lut.output_scale_factor();

            quant_output.outputs[0]
                .get_data()
                .chunks(num_tokens)
                .for_each(|chunk| {
                    let row_sum = chunk.iter().sum::<Element>();

                    let diff_from_one = (row_sum - lut.output_scale_factor() as Element).abs();
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
        let dim_size = 5;
        let input_shape = vec![3, dim_size, dim_size];

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
            let layer_q = layer_f.quantise(scaling).unwrap();

            let out_q = layer_q.evaluate(&[&quant_input.as_wrapped()]).unwrap();

            let quant_rows = out_q.outputs[0].get_data();
            let qi = layer_q.quant_info().unwrap();
            let row_err_bound_scaled = (qi.error_bound * qi.lut.output_scale_factor()).round() as Element;


            for (j ,row_q) in quant_rows.chunks(n).enumerate() {
                // Row sum closeness (integer domain)
                let row_sum: Element = row_q.iter().copied().sum();
                let diff = (row_sum - qi.lut.output_scale_factor() as Element).abs();
                prop_assert!(diff <= row_err_bound_scaled, "row {j} sum {row_sum}, row {row_q:?}, expected {}, diff {diff}, allowed {}, float allowed {}",  qi.lut.output_scale_factor(), row_err_bound_scaled, qi.error_bound * qi.lut.output_scale_factor());
            }
        }
    }
}
