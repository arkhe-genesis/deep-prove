//! Module containing code for performing proving friendly requantisation. This is done via a [fixed point multiplication](https://en.wikipedia.org/wiki/Fixed-point_arithmetic#Binary_fixed-point_multiplication) and use of lookup arguments.
use super::{
    LayerCtx,
    provable::{Evaluate, LayerOut, OpInfo, PadOp, ProvableOp, ProveInfo, VerifiableCtx},
};
use crate::{
    Claim, Element, NextPowerOfTwo, Prover, ProverContext, ScalingFactor, Shape, Tensor,
    graph::NodeId,
    iop::{
        context::{ContextAux, ShapeStep},
        verifier::Verifier,
    },
    layers::{LayerProof, activation::lookup_data::ActivationLookupData},
    lookup::{
        context::LookupWitnessGen, logup_gkr::structs::LogUpBatchProof, operation::LookupOp,
        table::Table,
    },
    model::Step,
    padding::PaddingMode,
    quantization,
    tensor::WrappedTensor,
};
use anyhow::{Result, anyhow, ensure};
use ceno_p3::field::FieldAlgebra;
use ff_ext::ExtensionField;

use mpcs::PolynomialCommitmentScheme;
use multilinear_extensions::util::{ceil_log2, transpose};

use serde::{Deserialize, Serialize, de::DeserializeOwned};
use sumcheck::structs::IOPProof;
use transcript::Transcript;
use witness::RowMajorMatrix;

mod evaluate;
mod lookup;
mod prove;
mod verify;

/// Constant used to identify the requantisation layer
pub const REQUANT_LAYER: &str = "REQU";

/// Constant used in fixed point multiplication for normalised [`f32`] values
pub(crate) const FIXED_POINT_SCALE: usize = 15;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Copy, PartialOrd)]
/// This struct contains the information used in requantisation (i.e. rescaling and clamping)
/// The fields are:
/// - `multiplier`: This is the actual [`f32`] value calculated as `S1 * S2 / S3` and in traditional quantisation is what we would multiply by and then round to requantise
/// - `right_shift`: This is `multiplier.log2().trunc().abs()`
/// - `fixed_point_multiplier`: This is `2.0.powf(multiplier.log2().fract()) * (1 << `fp_scale`)`, `fp_scale` is chosen to be at least 25 bits as the [`f32`] mantissa is only 24 bits long so this should retain all bits.
/// - `fp_scale`: This is calculated so that `fp_scale + right_shift` is a multiple of [`quantization::BIT_LEN`], that way we only need one size of range table.
/// - `intermediate_bit_size`: This is the maximum number of bits a value can have before its requantised.
pub struct Requant {
    /// The output scaling factor after requantisation
    pub output_scaling: ScalingFactor,
    /// The requantisation table
    pub table: Table,
    pub activation_lookup_data: ActivationLookupData,
}

impl LookupOp for Requant {
    fn intermediate_bit_size(&self) -> usize {
        self.activation_lookup_data.intermediate_bit_size()
    }

    fn right_shift(&self) -> usize {
        self.activation_lookup_data.right_shift()
    }

    fn fixed_point_multiplier(&self) -> Element {
        self.activation_lookup_data.fixed_point_multiplier()
    }

    fn is_signed(&self) -> bool {
        self.activation_lookup_data.is_signed()
    }

    fn padding_value(&self) -> Element {
        self.activation_lookup_data.padding_value()
    }
}

/// Info related to the lookup protocol necessary to requantize
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RequantCtx {
    pub requant: Requant,
    pub node_id: NodeId,
    pub num_vars: usize,
}

#[derive(Clone, Serialize, Deserialize)]
/// Struct holding all the information needed to verify requantisation was performed correctly.
/// This includes both lookup proofs and an additional sumcheck proof that we use so that all evaluations are at the same point.
#[serde(bound(serialize = "E: Serialize", deserialize = "E: DeserializeOwned"))]
pub struct RequantProof<E: ExtensionField, PCS: PolynomialCommitmentScheme<E>> {
    /// proof for the accumulation of the claim from activation + claim from lookup for the same poly
    /// e.g. the "link" between an activation and requant layer
    pub(crate) io_accumulation: IOPProof<E>,
    /// The evalaution claims about witness polynomials from the io_accumulation sumcheck
    pub(crate) io_eval: Vec<E>,
    /// The logup batch proof for all the lookups
    pub(crate) logup_proof: LogUpBatchProof<E>,
    /// COmmitments to lookup polynomials, they are in the order clamping commitments -> shifted commitments
    pub(crate) commitment: PCS::Commitment,
}

impl<E, PCS> RequantProof<E, PCS>
where
    E: ExtensionField,
    PCS: PolynomialCommitmentScheme<E>,
{
    pub(crate) fn write_commitment<T: Transcript<E>>(&self, transcript: &mut T) -> Result<()> {
        PCS::write_commitment(&self.commitment, transcript).map_err(|e| anyhow!("{e:?}"))
    }
}

const IS_PROVABLE: bool = true;

impl OpInfo for Requant {
    fn output_shapes(
        &self,
        input_shapes: &[Shape],
        _padding_mode: PaddingMode,
    ) -> Result<Vec<Shape>> {
        Ok(input_shapes.to_vec()) // preserve the input shape
    }

    fn num_outputs(&self, num_inputs: usize) -> Result<usize> {
        Ok(num_inputs)
    }

    fn describe(&self) -> String {
        format!(
            "Requant: right shift: {}, scale: {}",
            self.shift(),
            self.activation_lookup_data.fixed_point_multiplier(),
        )
    }

    fn is_provable(&self) -> bool {
        IS_PROVABLE
    }
}

impl Evaluate<Element> for Requant {
    fn evaluate(&self, inputs: &[&WrappedTensor<Element>]) -> Result<LayerOut<Element>> {
        self.evaluate_internal(inputs)
    }
}

impl ProveInfo for Requant {
    fn step_info<E: ExtensionField>(
        &self,
        id: NodeId,
        mut aux: ContextAux,
    ) -> Result<(LayerCtx<E>, ContextAux)> {
        // `try_fold` would not allow returning of `Err` values
        // from here and would short-circuit
        // instead of looping over all values in the iterator
        #[allow(clippy::manual_try_fold)]
        let num_vars = aux
            .last_output_shape
            .iter_mut()
            .fold(Ok(None), |expected_num_vars, shape| {
                let num_vars = shape.iter().map(|dim| ceil_log2(*dim)).sum::<usize>();
                if let Some(vars) = expected_num_vars? {
                    ensure!(
                        vars == num_vars,
                        "All input shapes for requant layer \
                        must have the same number of variables"
                    );
                }
                Ok(Some(num_vars))
            })?
            .expect("No input shape found for requant layer?");
        // Set the model polys to be empty
        aux.model_polys = None;
        aux.max_poly_len = aux
            .last_output_shape
            .iter()
            .fold(aux.max_poly_len, |acc, shapes| {
                acc.max(shapes.next_power_of_two().product())
            });

        Ok((
            LayerCtx::Requant(RequantCtx {
                requant: *self,
                node_id: id,
                num_vars,
            }),
            aux,
        ))
    }
}

impl PadOp for Requant {}

impl<E, PCS> ProvableOp<E, PCS> for Requant
where
    E: ExtensionField,
    PCS: PolynomialCommitmentScheme<E> + Send + Sync,
    PCS::CommitmentWithWitness: Serialize + DeserializeOwned + Send + Sync,
    PCS::ProverParam: Send + Sync,
{
    type Ctx = RequantCtx;

    fn prove<T: Transcript<E>>(
        &self,
        id: NodeId,
        _ctx: &Self::Ctx,
        last_claims: Vec<&Claim<E>>,
        step_data: &Step<Element>,
        prover: &mut Prover<E, T, PCS>,
    ) -> Result<Vec<Claim<E>>> {
        self.prove_step(prover, last_claims[0], step_data, id)
    }

    fn gen_lookup_witness(
        &self,
        id: NodeId,
        ctx: &ProverContext<E, PCS>,
        step_data: &Step<Element>,
    ) -> Result<LookupWitnessGen<E, PCS>> {
        ensure!(
            step_data.node_inputs.len() == 1,
            "Found more than 1 input in inference step of requant layer"
        );
        ensure!(
            step_data.outputs().len() == 1,
            "Found more than 1 output in inference step of requant layer"
        );

        let input = step_data.input_tensor_at(0)?;

        self.lookup_witness(id, ctx, &input)
    }
}

impl OpInfo for RequantCtx {
    fn output_shapes(
        &self,
        input_shapes: &[Shape],
        _padding_mode: PaddingMode,
    ) -> Result<Vec<Shape>> {
        Ok(input_shapes.to_vec())
    }

    fn num_outputs(&self, num_inputs: usize) -> Result<usize> {
        Requant::num_outputs(&self.requant, num_inputs)
    }

    fn describe(&self) -> String {
        format!(
            "Requant ctx: fixed point multiplier: {}, right shift: {}",
            self.requant.activation_lookup_data.fixed_point_multiplier(),
            self.requant.activation_lookup_data.right_shift(),
        )
    }

    fn is_provable(&self) -> bool {
        IS_PROVABLE
    }
}

impl<E, PCS> VerifiableCtx<E, PCS> for RequantCtx
where
    E: ExtensionField,
    PCS: PolynomialCommitmentScheme<E>,
{
    type Proof = RequantProof<E, PCS>;

    fn verify<T: Transcript<E>>(
        &self,
        proof: &Self::Proof,
        last_claims: &[&Claim<E>],
        verifier: &mut Verifier<E, T, PCS>,
        shape_step: &ShapeStep,
    ) -> Result<Vec<Claim<E>>> {
        self.verify_requant(verifier, last_claims[0], proof, shape_step)
    }

    fn write_proof_to_transcript<T: Transcript<E>>(
        &self,
        proof: &Self::Proof,
        transcript: &mut T,
    ) -> anyhow::Result<()> {
        proof.write_commitment(transcript)
    }
}

impl Requant {
    /// Method used to instantiate a new [`Requant`] from the multiplier employed to requantize the layer.
    /// The `intermediate_bit_size` is layer dependant and so should be passed as input. It can be calculated based on how many times you need to multiply and add
    /// to get each value in the output tensor.
    pub fn from_multiplier(
        multiplier: f32,
        intermediate_bit_size: usize,
        output_scaling: ScalingFactor,
    ) -> Requant {
        let log_m = multiplier.log2();
        // This is the right shift
        let int_part = log_m.trunc() as Element;
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

        let right_shift = (int_part - fp_scale as Element).unsigned_abs() as usize;

        let table = Table::new_requantise();
        let output_bit_size = output_scaling.bit_size() + 1;
        // We work out how many value chunks for this requantisation operation here, it must be a multiple of the requantisation BIT_LEN
        let value_chunks = output_bit_size / *quantization::BIT_LEN;

        assert!(
            output_bit_size.is_multiple_of(*quantization::BIT_LEN),
            "Output bit size after requantisation must be a multiple of {}, got {}",
            *quantization::BIT_LEN,
            output_bit_size
        );

        let activation_lookup_data = ActivationLookupData::new(
            right_shift,
            fixed_point_multiplier,
            intermediate_bit_size,
            0,
            table,
            false,
            value_chunks,
        );
        Requant {
            output_scaling,
            table: Table::new_requantise(),
            activation_lookup_data,
        }
    }
    /// Method used to instantiate a new [`Requant`] from the scaling factors of all tensors involved in a layer.
    /// The `intermediate_bit_size` is layer dependant and so should be passed as input. It can be calculated based on how many times you need to multiply and add
    /// to get each value in the output tensor.
    pub fn from_scaling_factors(
        input_scale: ScalingFactor,
        weights_scale: ScalingFactor,
        output_scale: ScalingFactor,
        intermediate_bit_size: usize,
    ) -> Requant {
        let m = input_scale.m(&weights_scale, &output_scale);
        Self::from_multiplier(m, intermediate_bit_size, output_scale)
    }

    /// This returns the shift (including the part that depends on `S1 * S2/ S3`)
    pub(crate) fn shift(&self) -> usize {
        self.activation_lookup_data.right_shift()
    }

    pub fn write_to_transcript<E: ExtensionField, T: Transcript<E>>(&self, t: &mut T) {
        t.append_field_element(&E::BaseField::from_canonical_u64(
            self.activation_lookup_data.right_shift() as u64,
        ));
        t.append_field_element(&E::BaseField::from_canonical_u64(
            self.activation_lookup_data.fixed_point_multiplier() as u64,
        ));
    }
}

impl RequantCtx {
    pub fn lookup_tables(&self) -> Vec<Table> {
        let lookup_data = self.requant.activation_lookup_data;
        let chunking_info = lookup_data.chunking_info(&lookup_data.table).unwrap();

        match chunking_info.number_of_zeroing_chunks() {
            0 => vec![Table::new_shift_check(), Table::new_requantise()],
            1 => vec![
                Table::new_shift_check(),
                Table::new_requantise(),
                Table::new_signed_zero_check(),
            ],
            _ => vec![
                Table::new_shift_check(),
                Table::new_requantise(),
                Table::new_zero_check(),
                Table::new_signed_zero_check(),
            ],
        }
    }
}

#[cfg(test)]
mod tests {
    use ark_std::rand::Rng;
    use tenstore::GenStore;

    use crate::{
        layers::{Layer, einsum::EinSum, provable::QuantizeOutput},
        model::{Model, test::prove_model},
        quantization::{Dequantize, Quantize, model_scaling_factor_from_tensor_and_bias},
        rng_from_env_or_random,
        tensor::{KeyedTensor, TensorHandle},
    };

    use proptest::prelude::*;

    use super::*;

    #[test]
    fn test_proving() {
        for _ in 0..25 {
            let Input {
                rows: _,
                columns: _,
                weight,
                bias,
                input: random_input,
            } = Input::random(25, 25);

            let input_rank = random_input.shape().rank();
            let equation = match input_rank {
                1 => "I(j)@W(ij)->O(i)+BIAS(i)",
                2 => "I(aj)@W(ij)->O(ai)+BIAS(i)",
                3 => "I(abj)@W(ij)->O(abi)+BIAS(i)",
                4 => "I(abcj)@W(ij)->O(abci)+BIAS(i)",
                _ => panic!("Input rank too high for test"),
            }
            .to_string();
            let dense = EinSum::<f32>::new(
                equation.to_owned(),
                vec![Some(weight.into())],
                vec![Some(bias.into())],
            )
            .unwrap();

            let mut model = Model::new_from_input_shapes(vec![random_input.shape().clone()]);

            let _ = model
                .add_consecutive_layer(Layer::EinSum(dense), None)
                .unwrap();
            model.automatic_output_labelling().unwrap();
            model.describe();
            prove_model(model, &mut GenStore::default()).unwrap();
        }
    }

    #[derive(Clone)]
    struct Input {
        rows: usize,
        columns: usize,
        weight: KeyedTensor<f32>,
        bias: KeyedTensor<f32>,
        input: Tensor<f32>,
    }

    impl std::fmt::Debug for Input {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.debug_struct("Input")
                .field("rows", &self.rows)
                .field("columns", &self.columns)
                .field("input", self.input.shape())
                .finish()
        }
    }

    impl Input {
        fn random(rows_max: usize, columns_max: usize) -> Input {
            let mut rng = rng_from_env_or_random();
            let rows = rng.gen_range(4..rows_max);
            let columns = rng.gen_range(4..columns_max);
            let matrix_size = rows * columns;
            let weight_data: Vec<f32> = (0..matrix_size)
                .map(|_| rng.gen_range(-10.0..10.0))
                .collect();
            let bias_data: Vec<f32> = (0..rows).map(|_| rng.gen_range(-10.0..10.0)).collect();

            let input_rank = rng.gen_range(1usize..=4);
            let mut all_dims: Vec<usize> =
                (0..(input_rank - 1)).map(|_| rng.gen_range(3..8)).collect();
            all_dims.push(columns);
            let total_data_size = all_dims.iter().product::<usize>();
            let input_shape = Shape::from(all_dims);
            let input_data: Vec<f32> = (0..total_data_size)
                .map(|_| rng.gen_range(-10.0..10.0))
                .collect();

            Input {
                rows,
                columns,
                weight: KeyedTensor::new(
                    "W".to_string(),
                    Tensor::new(vec![rows, columns].into(), weight_data).unwrap(),
                ),
                bias: KeyedTensor::new(
                    "BIAS".to_string(),
                    Tensor::new(vec![rows].into(), bias_data).unwrap(),
                ),
                input: Tensor::new(input_shape, input_data).unwrap(),
            }
        }
    }

    fn requantise_input(
        rows: std::ops::Range<usize>,
        columns: std::ops::Range<usize>,
        value_range: std::ops::Range<f32>,
    ) -> impl Strategy<Value = Input> {
        (rows, columns).prop_flat_map(move |(row_count, column_count)| {
            let matrix_size = row_count * column_count;
            let weight_data_strategy = prop::collection::vec(value_range.clone(), matrix_size);
            let bias_data_strategy = prop::collection::vec(value_range.clone(), row_count);
            let input_data_strategy = prop::collection::vec(value_range.clone(), column_count);
            (
                weight_data_strategy,
                bias_data_strategy,
                input_data_strategy,
            )
                .prop_map(move |(weight_data, bias_data, inputs)| Input {
                    rows: row_count,
                    columns: column_count,
                    weight: KeyedTensor::new(
                        "W".to_string(),
                        Tensor::new(vec![row_count, column_count].into(), weight_data).unwrap(),
                    ),
                    bias: KeyedTensor::new(
                        "BIAS".to_string(),
                        Tensor::new(vec![row_count].into(), bias_data).unwrap(),
                    ),
                    input: Tensor::new(vec![column_count].into(), inputs).unwrap(),
                })
        })
    }

    proptest! {
        #[test]
        fn proptest_requantise(inp in requantise_input(4..25, 4..25, -10.0f32..10.0f32)) {
            let Input { rows, columns, weight, bias, input: random_input } = inp.clone();

            let dense = EinSum::<f32>::new("I(j)@W(ij)->O(i)+BIAS(i)".to_string(), vec![Some(weight.clone().into())], vec![Some(bias.clone().into())]).unwrap();

            let wrapped_input = WrappedTensor::try_from(&random_input).unwrap();
            let mut outputs = dense.evaluate_internal(std::slice::from_ref(&&wrapped_input)).unwrap();
            let output = outputs.remove(0).to_native();

            let input_scale_factor = ScalingFactor::from_tensor(&random_input, None);
            let output_scale_factor = ScalingFactor::from_tensor(&output, None);

            let intermediate_bit_size = input_scale_factor.bit_size() + *crate::quantization::BIT_LEN + ceil_log2(columns);
            let (weight_scale_factor, bias_scale_factor) = model_scaling_factor_from_tensor_and_bias(&input_scale_factor, weight.max_abs(), intermediate_bit_size);

            let quantised_dense = dense.clone().quantise(
                std::slice::from_ref(&input_scale_factor),
                std::slice::from_ref(&output_scale_factor),
                std::slice::from_ref(random_input.shape()),
            ).unwrap();
            let weight_handle: TensorHandle<f32> = weight.into();
            let bias_handle: TensorHandle<f32> = bias.into();
            let comparison_weight = weight_handle.quantize(&weight_scale_factor).dequantize(&weight_scale_factor);
            let comparison_bias = bias_handle.quantize(&bias_scale_factor).dequantize(&bias_scale_factor);
            let comparison_dense = EinSum::<f32>::new("I(j)@W(ij)->O(i)+BIAS(i)".to_string(), vec![Some(comparison_weight)], vec![Some(comparison_bias)]).unwrap();

            let QuantizeOutput {
                quantized_op: element_dense,
                output_scalings: _,
                requant_layer,
                ..
            } = quantised_dense;

            let requantise = requant_layer.unwrap()[0];

            let quantised_input = random_input.quantize(&input_scale_factor);
            let dequantised_input = quantised_input.dequantize(&input_scale_factor);

            let wrapped_quantised_input = WrappedTensor::try_from(&quantised_input).unwrap();
            let mut quantised_outputs =
                element_dense.evaluate_internal(std::slice::from_ref(&&wrapped_quantised_input)).unwrap();
            let quantised_output = quantised_outputs.remove(0);

            let native_quantised_dense_out = quantised_output.to_native();
            let dequant_quantised_dense_out = native_quantised_dense_out.dequantize(&bias_scale_factor).into_data();


            let wrapped_comparison_input = WrappedTensor::try_from(&dequantised_input).unwrap();
            let mut comparison_outputs =
                comparison_dense.evaluate_internal(std::slice::from_ref(&&wrapped_comparison_input)).unwrap();

            let comparison_output_tmp = comparison_outputs.remove(0).to_native();

            let comparison_output = comparison_output_tmp
                .quantize(&output_scale_factor)
                .dequantize(&output_scale_factor);

            let requantise_table = Table::new_requantise();
            let final_quantised_output = requantise
                .apply(quantised_output, &requantise_table).unwrap()
                .to_native();
            let calculated_output = final_quantised_output.dequantize(&output_scale_factor);
            for (i, (calc, comp)) in calculated_output.data().iter().zip(comparison_output.data().iter()).enumerate() {
                // They should either be equal or the difference is due to rounding and so should be almost exactly the output scaling factor.
                let diff = (calc - comp).abs();
                let diff_with_output_scale = (diff - output_scale_factor.scale()).abs();
                let diff_zero = diff == 0.0f32;
                let diff_due_to_rounding = diff_with_output_scale < 1e-3;
                prop_assert!(diff_zero || diff_due_to_rounding, "Requantisation lookup did not produce correct result for shape {rows}x{columns} at index {i}, calculated {calc}, expected {comp}, dequantised input {input} diff {diff} \n input scale: {in_scale}, weight scale: {weight_scale}, bias scale: {bias_scale}, output scale: {out_scale}",
                rows=rows, columns=columns, i=i, calc=calc, comp=comp, input=dequant_quantised_dense_out[i], diff=(calc-comp).abs(), in_scale=input_scale_factor.scale(), weight_scale=weight_scale_factor.scale(), bias_scale=bias_scale_factor.scale(), out_scale=output_scale_factor.scale());
            }
        }
    }
}
