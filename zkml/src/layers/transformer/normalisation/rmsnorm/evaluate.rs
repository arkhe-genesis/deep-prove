//! Module containing code for evaluating [`RMSNorm`] layers using both floating point and quantised inference.
use super::*;

use crate::layers::{provable::ProvingData, requant::FIXED_POINT_SCALE};
use burn::tensor::TensorData;
use tenstore::GenStore;

impl RMSNorm<f32> {
    /// Evaluates the [`RMSNorm`] layer over [`f32`] tensors.
    pub fn evaluate_float_internal(&self, inputs: &[&WrappedTensor<f32>]) -> Result<LayerOut<f32>> {
        let input = inputs[0];
        // Check that the final dimension of the input matches the expected dim_size
        let final_dim_size = input.dim(-1)?;
        ensure!(
            final_dim_size == self.normalisation_dim_size(),
            "RMSNorm input must have final dimension of {}: found {:?}",
            self.normalisation_dim_size(),
            input.shape(),
        );

        let alpha_opt = self
            .alpha
            .as_ref()
            .map(|keyed_alpha| keyed_alpha.wrapped_tensor())
            .transpose()?
            .map(|t| t.clone());

        let output = WrappedTensor::rms_norm_forward(
            input.clone(),
            self.normalisation_dim_size(),
            self.eps as f64,
            alpha_opt,
        )?;

        Ok(LayerOut::from_tensor(output))
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RMSNormHandle {
    pub(crate) normalisation: TensorHandle<Element>,
    pub(crate) lookup_verifier: RMSNormLookupVerifier,
}

impl RMSNormHandle {
    pub(crate) fn new(
        storage_key: StorageKey<Vec<Element>>,
        rmsnorm_data: RMSNormProvingData,
        store: tenstore::GenStore,
    ) -> Self {
        let key = StorageKey::new(format!("{}-normalisation", storage_key.id()));
        let normalisation =
            TensorHandle::from_wrapped_tensor(key, store, rmsnorm_data.normalisation);
        Self {
            normalisation,
            lookup_verifier: rmsnorm_data.lookup_verifier,
        }
    }

    pub(crate) fn to_proving_data(&self) -> Result<RMSNormProvingData> {
        let normalisation_native = self.normalisation.tensor()?;
        let normalisation = WrappedTensor::try_from(normalisation_native.as_ref())?;
        Ok(RMSNormProvingData {
            normalisation,
            lookup_verifier: self.lookup_verifier,
        })
    }

    pub(crate) fn isolate(&self) -> RMSNormHandle {
        Self {
            normalisation: self.normalisation.isolate(),
            lookup_verifier: self.lookup_verifier,
        }
    }

    pub(crate) fn attach_store(&mut self, store: GenStore) {
        self.normalisation.attach_store(store);
    }
}

impl RMSNorm<Element> {
    /// Evaluates the [`RMSNorm`] layer over quantized tensors.
    pub fn evaluate_quantised_internal(
        &self,
        inputs: &[&WrappedTensor<Element>],
    ) -> Result<LayerOut<Element>> {
        ensure!(
            inputs.len() == 1,
            "Exactly one input must be provided to RMSnorm. got {}",
            inputs.len(),
        );
        let input = inputs[0].clone().reduce_to_unpadded_shape()?;
        let shape = input.shape();
        // Ensure we have the quantisation scaling factors
        let (input_scaling_factor, normalisation_scaling_factor) = self
            .get_quantisation_scaling_factors()
            .context("Quantisation scaling factors not found for RMSNorm evaluation")?;

        let dim_size = self.normalisation_dim_size() as f32;

        let normalising_factor = input
            .clone()
            .mul(input.clone())?
            .sum_dim(-1)
            .float()
            .mul_scalar(input_scaling_factor.scale() * input_scaling_factor.scale() / dim_size)
            .add_scalar(self.eps)
            .sqrt()
            .recip();

        let rescaling_factor =
            input_scaling_factor.scale() * normalisation_scaling_factor.scale().recip();

        let norm_with_scale = normalising_factor.mul_scalar(rescaling_factor);
        let norm_with_scale_shape = norm_with_scale.shape();
        let mut cache = self.cache.lock().unwrap();
        let (mut fract_data, shift_data) = norm_with_scale.get_data().iter().map(|x| {
            let log_x = x.log2();
            let fract_mul = (2.0f32.powf(log_x.fract()) * (1u64 << FIXED_POINT_SCALE) as f32).round_ties_even() as Element;
            let shift_amount = log_x.trunc() as Element - FIXED_POINT_SCALE as Element;
            if shift_amount < 0 {
                cache.update(shift_amount.abs());
                (fract_mul, shift_amount.abs())
            } else {
                panic!("RMSNorm normalising factor fract part calculation produced invalid shift amount {}", shift_amount);
            }
        }).unzip::<Element, Element, Vec<Element>, Vec<Element>>();

        let max_shift = cache.get_shift();

        for (mul_val, shift_val) in fract_data.iter_mut().zip(shift_data.iter()) {
            let shift_diff = max_shift - *shift_val;
            *mul_val <<= shift_diff;
        }

        let mult_tensor_data = TensorData::new(fract_data, norm_with_scale_shape);
        let mult = WrappedTensor::try_from(mult_tensor_data)?;

        let normalised_sum_sq = (self.normalisation_dim_size() as f32
            / (normalisation_scaling_factor.scale() * normalisation_scaling_factor.scale()))
        .round_ties_even() as Element;
        let error_bound = (self.normalisation_dim_size() as f32
            * (2.0f32 * normalisation_scaling_factor.scale().recip() + 1.0f32))
            .round_ties_even() as Element;

        let proving_data = RMSNormProvingData::new(
            mult,
            max_shift.unsigned_abs() as usize,
            normalised_sum_sq,
            error_bound,
            input_scaling_factor.bit_size() + 1,
            self.alpha.is_some(),
        );

        let table = Table::new_normalisation(normalisation_scaling_factor.bit_size() + 1);
        let tmp_output = proving_data.apply(input, &table)?;

        // Get the alpha
        let output = match self.alpha.as_ref() {
            Some(keyed_alpha) => {
                let alpha_tensor = keyed_alpha
                    .wrapped_tensor()?
                    .clone()
                    .reduce_to_unpadded_shape()?;
                let expanded_alpha = match shape.rank() {
                    1 => alpha_tensor,
                    2 => alpha_tensor.unsqueeze_dim_2(),
                    3 => alpha_tensor.unsqueeze_dim_3(),
                    4 => alpha_tensor.unsqueeze_dim_4(),
                    _ => {
                        anyhow::bail!(
                            "Unsupported input rank for RMSNorm quantised evaluation: {}",
                            shape.rank()
                        );
                    }
                };
                tmp_output.mul(expanded_alpha)?
            }
            None => tmp_output,
        };

        if inputs[0].is_padded() {
            let output = output.pad_next_power_of_two();
            Ok(LayerOut::from_tensor(output).with_proving_data(ProvingData::RMSNorm(proving_data)))
        } else {
            Ok(LayerOut::from_tensor(output).with_proving_data(ProvingData::RMSNorm(proving_data)))
        }
    }
}

#[cfg(test)]
mod tests {

    use super::*;

    use proptest::prelude::*;
    use std::{
        fmt::{Debug, Display},
        ops::Range,
    };

    #[test]
    fn test_quantised() {
        let input = Tensor::<f32>::random(&Shape::new(vec![60, 640]));
        let input_shape = input.shape();
        let layer = RMSNorm::new(None, 1e-5, Some(input_shape.dim(-1))).unwrap();

        let wrapped_input = WrappedTensor::try_from(&input).unwrap();
        let LayerOut { outputs, .. } = layer.evaluate_float_internal(&[&wrapped_input]).unwrap();

        let input_scaling = ScalingFactor::from_tensor(&input, None);
        let output_native = outputs[0].to_native();
        let output_scaling = ScalingFactor::from_tensor(&output_native, None);

        // We want to calculate the largest quantisation domain that will allow us to fit the magnitude normalisation check into
        // a single shift check table. This means we need twice the error bound to have a bit length <= SHIFT_CHECK_TABLE_BIT_SIZE.
        // The error bound is normalisation_dim_size * (1.0 / normalisation_scaling_factor + 0.25) so we can rearrange to find the normalisation_scaling_factor that will allow us to fit into the table.
        // The rearrangement gives us (2^SHIFT_CHECK_TABLE_BIT_SIZE / normalisation_dim_size) - 0.25 >= 1.0 / normalisation_scaling_factor, so normalisation_scaling_factor >= 1.0 / ((2^SHIFT_CHECK_TABLE_BIT_SIZE / normalisation_dim_size) - 0.25).
        let minimum_normalising_scale = 2.0
            / ((1u64 << (SHIFT_CHECK_TABLE_BIT_SIZE - 1)) as f32
                / layer.normalisation_dim_size as f32
                - 1.0);

        // The scale is calculated as (float_max - float_min) / (quant_max - quant_min)
        let mut normalisation_bits = *quantization::BIT_LEN;
        let norm_max = (layer.normalisation_dim_size() as f32).sqrt();
        let norm_min = -norm_max;

        let mut test_scale = (norm_max - norm_min) / ((1 << normalisation_bits) as f32);
        while test_scale < minimum_normalising_scale && normalisation_bits > 1 {
            normalisation_bits -= 1;
            test_scale = (norm_max - norm_min) / ((1 << normalisation_bits) as f32);
        }

        let norm_quant_min: Element = -1 << (normalisation_bits - 1);
        let norm_quant_max: Element = (1 << (normalisation_bits - 1)) - 1;
        let normalisation_scaling_factor = ScalingFactor::from_parts(
            norm_max,
            norm_min,
            test_scale,
            (norm_quant_min, norm_quant_max),
        );

        let QuantizeOutput { quantized_op, .. } = layer
            .quantise(input_scaling, normalisation_scaling_factor, output_scaling)
            .unwrap();

        let quantised_input = input.quantize(&input_scaling);
        let wrapped_quantised_input = WrappedTensor::try_from(&quantised_input).unwrap();
        let quantised_layer_out = quantized_op
            .evaluate_quantised_internal(&[&wrapped_quantised_input])
            .unwrap();

        let quantised_output = quantised_layer_out.outputs();
        let RMSNormProvingData {
            normalisation: _,
            lookup_verifier,
        } = quantised_layer_out.try_rmsnorm_data().unwrap();

        let dim_size = quantized_op.normalisation_dim_size();
        let quantized_dim_size = lookup_verifier.normalisation_sum_sq;
        let bound_dim_size = lookup_verifier.error_bound;

        for output_chunk in quantised_output[0].get_data().chunks(dim_size) {
            let square_sum = output_chunk.iter().map(|x| x * x).sum::<Element>();
            let diff = (quantized_dim_size - square_sum).abs();
            assert!(
                diff <= bound_dim_size || square_sum == 0,
                "Norm not within range bound, got square sum {square_sum} expected to be within {bound_dim_size} of {quantized_dim_size} or equal 0, was actually {diff}, out by {}, chunk: {:?}",
                diff - bound_dim_size,
                output_chunk
            );
        }
    }

    #[derive(Clone)]
    struct Input {
        input: Tensor<f32>,
    }

    impl Debug for Input {
        fn fmt(
            &self,
            fmt: &mut std::fmt::Formatter<'_>,
        ) -> std::result::Result<(), std::fmt::Error> {
            let non_zero = self.input.data().iter().filter(|&&x| x != 0.0).count();
            fmt.debug_struct("Input")
                .field(
                    "input",
                    &format_args!(
                        "Input{{shape: {:?}, non_zero: {}}}",
                        self.input.shape(),
                        non_zero
                    ),
                )
                .finish()
        }
    }

    impl Display for Input {
        fn fmt(&self, fmt: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
            write!(fmt, "Input{{input: {:?}}}", self.input.shape(),)
        }
    }

    fn rmsnorm_input(
        normalisation_dim: Range<usize>,
        other_dims: Range<usize>,
    ) -> impl Strategy<Value = Input> {
        (
            normalisation_dim,
            prop::collection::vec(other_dims, 1usize..=3),
        )
            .prop_flat_map(move |(normalisation_dim, mut shape)| {
                shape.push(normalisation_dim);
                let input = Tensor::any(Shape::new(shape));
                input.prop_map(|input| Input { input })
            })
    }

    proptest! {
        #[test]
        fn proptest_rmsnorm_evaluate_quantised(input in rmsnorm_input(10usize..768, 2usize..64)) {
            let Input {
                input,
            } = input;

            let input_shape = input.shape();
            let layer = RMSNorm::new(None, 1e-5, Some(input_shape.dim(-1))).unwrap();

            let wrapped_input = WrappedTensor::try_from(&input).unwrap();
            let LayerOut { outputs, .. } = layer.evaluate_float_internal(&[&wrapped_input]).unwrap();


            let input_scaling = ScalingFactor::from_tensor(&input, None);
            let output_native = outputs[0].to_native();
            let output_scaling = ScalingFactor::from_tensor(&output_native, None);

            // We want to calculate the largest quantisation domain that will allow us to fit the magnitude normalisation check into
            // a single shift check table. This means we need twice the error bound to have a bit length <= SHIFT_CHECK_TABLE_BIT_SIZE.
            // The error bound is normalisation_dim_size * (1.0 / normalisation_scaling_factor + 0.25) so we can rearrange to find the normalisation_scaling_factor that will allow us to fit into the table.
            // The rearrangement gives us (2^SHIFT_CHECK_TABLE_BIT_SIZE / normalisation_dim_size) - 0.25 >= 1.0 / normalisation_scaling_factor, so normalisation_scaling_factor >= 1.0 / ((2^SHIFT_CHECK_TABLE_BIT_SIZE / normalisation_dim_size) - 0.25).
            let minimum_normalising_scale = 2.0
            / ((1u64 << (SHIFT_CHECK_TABLE_BIT_SIZE - 1)) as f32
                / input_shape.dim(-1) as f32
                - 1.0);

            // The scale is calculated as (float_max - float_min) / (quant_max - quant_min)
            let mut normalisation_bits = *quantization::BIT_LEN;
            let norm_max = (layer.normalisation_dim_size() as f32).sqrt();
            let norm_min = -norm_max;

            let mut test_scale = (norm_max - norm_min) / ((1 << normalisation_bits) as f32);
            while test_scale < minimum_normalising_scale && normalisation_bits > 1 {
                normalisation_bits -= 1;
                test_scale = (norm_max - norm_min) / ((1 << normalisation_bits) as f32);
            }

            let norm_quant_min: Element = -1 << (normalisation_bits - 1);
            let norm_quant_max: Element = (1 << (normalisation_bits - 1)) - 1;
            let normalisation_scaling_factor = ScalingFactor::from_parts(
                norm_max,
                norm_min,
                test_scale,
                (norm_quant_min, norm_quant_max),
            );

            let QuantizeOutput { quantized_op, .. } = layer.quantise(input_scaling, normalisation_scaling_factor, output_scaling).unwrap();

            let quantised_input = input.quantize(&input_scaling);
            let wrapped_quantised_input = WrappedTensor::try_from(&quantised_input).unwrap();

            let quantised_layer_out = quantized_op.evaluate_quantised_internal(&[&wrapped_quantised_input]).unwrap();


            let quantised_output = quantised_layer_out.outputs();
            let RMSNormProvingData {
                normalisation: _,
                lookup_verifier,
            } = quantised_layer_out.try_rmsnorm_data().unwrap();

            let normalised_sumsq = lookup_verifier.normalisation_sum_sq;
            let error_bound = lookup_verifier.error_bound;

            let dim_size = quantized_op.normalisation_dim_size();
            let scale_squared = normalisation_scaling_factor.scale().powi(2);
            for output_chunk in quantised_output[0].get_data().chunks(dim_size) {
                let square_sum = output_chunk.iter().map(|x| x * x).sum::<Element>();
                let diff = (normalised_sumsq - square_sum).abs();
                prop_assert!(diff <= error_bound || square_sum == 0,
                    "Norm not within range bound, got square sum {square_sum} expected to be within {error_bound} (float: {}) of {normalised_sumsq} or equal 0, was actually {diff}, out by {}, (float: {}, scale factor: {}), chunk: {:?} (float: {:?}, norm max: {}, norm min: {})",
                 error_bound as f32 * scale_squared, diff - error_bound, (diff - error_bound) as f32 * scale_squared, scale_squared, output_chunk, output_chunk.iter().map(|x| *x as f32 * normalisation_scaling_factor.scale()).collect::<Vec<f32>>(), normalisation_scaling_factor.max(), normalisation_scaling_factor.min());
            }
        }
    }
}
