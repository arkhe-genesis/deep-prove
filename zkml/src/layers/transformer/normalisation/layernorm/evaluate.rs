//! Module containing code for evaluating [`LayerNorm`] layers.

use tenstore::GenStore;

use crate::layers::{provable::ProvingData, requant::FIXED_POINT_SCALE};

use super::*;

impl LayerNorm<f32> {
    /// Evaluates the [`LayerNorm`] layer over [`f32`] tensors.
    pub fn evaluate_float_internal(&self, inputs: &[&WrappedTensor<f32>]) -> Result<LayerOut<f32>> {
        ensure!(
            inputs.len() == 1,
            "Exactly one input must be provided to layer norm. got {}",
            inputs.len(),
        );
        let input = inputs[0].clone();

        let embedding_size = input.dim(-1)?;
        // NOTE: simply use the burn tensor API for now as we want to move towards using more burn features
        // instead of re-implementing everything ourselves.
        // copy implementation https://docs.rs/burn-core/0.17.0/src/burn_core/nn/norm/layer.rs.html#67
        let gamma = self.gamma.wrapped_tensor()?;
        let beta = self.beta.wrapped_tensor()?;
        let output = WrappedTensor::layer_norm(
            input.clone(),
            embedding_size,
            self.eps as f64,
            gamma.clone(),
            beta.clone(),
        )?;

        Ok(LayerOut::from_tensor(output))
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerNormHandle {
    pub(crate) mean: TensorHandle<Element>,
    pub(crate) std_dev: TensorHandle<Element>,
    pub(crate) lookup_verifier: LayerNormLookupVerifier,
}

impl LayerNormHandle {
    pub(crate) fn new(
        storage_key: StorageKey<Vec<Element>>,
        layer_norm_data: LayerNormProvingData,
        store: tenstore::GenStore,
    ) -> Self {
        let key = StorageKey::new(format!("{}-mean", storage_key.id()));
        let mean = TensorHandle::from_wrapped_tensor(key, store.clone(), layer_norm_data.mean);

        let key = StorageKey::new(format!("{}-std-dev", storage_key.id()));
        let std_dev =
            TensorHandle::from_wrapped_tensor(key, store.clone(), layer_norm_data.std_dev);

        Self {
            mean,
            std_dev,
            lookup_verifier: layer_norm_data.lookup_verifier,
        }
    }
    pub(crate) fn to_proving_data(&self) -> Result<LayerNormProvingData> {
        let mean_native = self.mean.tensor()?;
        let mean = WrappedTensor::try_from(mean_native.as_ref())?;
        let std_dev_native = self.std_dev.tensor()?;
        let std_dev = WrappedTensor::try_from(std_dev_native.as_ref())?;
        Ok(LayerNormProvingData {
            mean,
            std_dev,
            lookup_verifier: self.lookup_verifier,
        })
    }

    pub(crate) fn isolate(&self) -> LayerNormHandle {
        Self {
            mean: self.mean.isolate(),
            std_dev: self.std_dev.isolate(),
            lookup_verifier: self.lookup_verifier,
        }
    }

    pub(crate) fn attach_store(&mut self, store: GenStore) {
        self.mean.attach_store(store.clone());
        self.std_dev.attach_store(store);
    }
}

impl LayerNorm<Element> {
    /// Evaluates the [`LayerNorm`] layer over quantized tensors.
    pub fn evaluate_quantised_internal(
        &self,
        inputs: &[&WrappedTensor<Element>],
    ) -> Result<LayerOut<Element>> {
        ensure!(
            inputs.len() == 1,
            "Exactly one input must be provided to layer norm. got {}",
            inputs.len(),
        );

        let input = inputs[0].clone().reduce_to_unpadded_shape()?;
        let shape = input.shape();
        // Ensure we have the quantisation scaling factors
        let (mean_scaling_factor, intermediate_scaling_factor) = self
            .get_quantisation_scaling_factors()
            .context("Quantisation scaling factors not found for LayerNorm evaluation")?;

        let dim_size = self.normalisation_dim_size() as f32;

        let n_mean = input.clone().sum_dim(-1);

        let zero_mean_input = input
            .clone()
            .mul_scalar(self.normalisation_dim_size() as Element)
            .sub(n_mean.clone())?
            .float()
            .mul_scalar(mean_scaling_factor.scale() / dim_size);
        let normalising_factor = zero_mean_input
            .clone()
            .mul(zero_mean_input.clone())?
            .sum_dim(-1)
            .div_scalar(dim_size)
            .add_scalar(self.eps)
            .sqrt()
            .recip();

        let rescaling_factor = mean_scaling_factor.scale() / intermediate_scaling_factor.scale();

        // In order to retain as much precision as possible we want to subtract the mean after applying the mult tensor
        // So we compute the rescaled mean as follows:
        // rescaled_mean = mean * (1 / (std_dev * intermediate_scaling))
        let rescaled_mean = n_mean
            .float()
            .mul_scalar(rescaling_factor / dim_size)
            .mul(normalising_factor.clone())?;

        let norm_with_scale = normalising_factor.mul_scalar(rescaling_factor);
        let norm_with_scale_shape: Shape = norm_with_scale.shape().into();
        let mut cache = self.cache.lock().unwrap();
        let (mut fract_data, shift_data) = norm_with_scale.get_data().iter().map(|&x| {
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

        let mult = WrappedTensor::try_from(&Tensor::<Element>::new(
            norm_with_scale_shape.clone(),
            fract_data,
        )?)?;

        let mean_to_sub = rescaled_mean
            .mul_scalar((1u64 << max_shift) as f32)
            .round()
            .int();

        let proving_data = LayerNormProvingData::new(
            mean_to_sub,
            mult,
            max_shift.unsigned_abs() as usize,
            self.normalisation_dim_size(),
            intermediate_scaling_factor,
            mean_scaling_factor.bit_size() + 1,
        );

        // Get the gamma and beta tensors
        let gamma = self
            .gamma
            .wrapped_tensor()?
            .clone()
            .reduce_to_unpadded_shape()?;
        let beta = self
            .beta
            .wrapped_tensor()?
            .clone()
            .reduce_to_unpadded_shape()?;
        let (gamma, beta) = match shape.rank() {
            1 => (gamma, beta),
            2 => (gamma.unsqueeze_dim_2(), beta.unsqueeze_dim_2()),
            3 => (gamma.unsqueeze_dim_3(), beta.unsqueeze_dim_3()),
            4 => (gamma.unsqueeze_dim_4(), beta.unsqueeze_dim_4()),
            _ => {
                anyhow::bail!(
                    "Unsupported input rank for LayerNorm quantised evaluation: {}",
                    shape.rank()
                );
            }
        };

        let table = Table::new_normalisation(intermediate_scaling_factor.bit_size() + 1);

        let output = proving_data.apply(input, &table)?.mul(gamma)?.add(beta)?;

        if inputs[0].is_padded() {
            let output = output.pad_next_power_of_two();
            Ok(LayerOut::from_tensor(output)
                .with_proving_data(ProvingData::LayerNorm(proving_data)))
        } else {
            Ok(LayerOut::from_tensor(output)
                .with_proving_data(ProvingData::LayerNorm(proving_data)))
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{rng_from_env_or_random, tensor::KeyedTensor};

    use super::*;
    use ark_std::rand::Rng;
    use multilinear_extensions::util::ceil_log2;
    use proptest::prelude::*;
    use std::{
        fmt::{Debug, Display},
        ops::Range,
    };

    use itertools::izip;

    #[derive(Clone)]
    struct Input {
        input: Tensor<f32>,
        beta: KeyedTensor<f32>,
        gamma: KeyedTensor<f32>,
    }

    impl Debug for Input {
        fn fmt(
            &self,
            fmt: &mut std::fmt::Formatter<'_>,
        ) -> std::result::Result<(), std::fmt::Error> {
            fmt.debug_struct("Input")
                .field("input", &format_args!("{:?}", self.input.shape()))
                .field("beta", &format_args!("{:?}", self.beta.shape()))
                .field("gamma", &format_args!("{:?}", self.gamma.shape()))
                .finish()
        }
    }

    impl Display for Input {
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

    fn layernorm_input(
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
                let beta = Tensor::any(Shape::new(vec![normalisation_dim]));
                let gamma = Tensor::any(Shape::new(vec![normalisation_dim]));
                (input, beta, gamma).prop_map(|(input, beta, gamma)| Input {
                    input,
                    beta: KeyedTensor::new("layernorm_beta", beta),
                    gamma: KeyedTensor::new("layernorm_gamma", gamma),
                })
            })
    }

    #[test]
    fn test_layernorm_evaluate_quant() {
        let mut rng = rng_from_env_or_random();
        let dim0 = 5;

        for _ in 0..10 {
            let dim1 = rng.gen_range(400..1000);
            let shape = Shape::new(vec![dim0, dim1]);
            let data1 = (0..shape.numel())
                .map(|_| rng.gen_range(-10.0f32..10.0f32))
                .collect::<Vec<f32>>();
            let data2 = (0..shape.numel())
                .map(|_| rng.gen_range(-10.0f32..10.0f32))
                .collect::<Vec<f32>>();
            let input = Tensor::<f32>::new(shape.clone(), data1).unwrap();
            let input2 = Tensor::<f32>::new(shape.clone(), data2).unwrap();
            let gamma = KeyedTensor::new(
                "layernorm_gamma",
                Tensor::<f32>::new(Shape::new(vec![dim1]), vec![1.0f32; dim1]).unwrap(),
            );
            let beta = KeyedTensor::new(
                "layernorm_beta",
                Tensor::<f32>::new(Shape::new(vec![dim1]), vec![0.0f32; dim1]).unwrap(),
            );

            let layer = LayerNorm::new(gamma.into(), beta.into(), 1e-5).unwrap();

            let wrapped_input = WrappedTensor::try_from(&input).unwrap();
            let LayerOut { outputs, .. } =
                layer.evaluate_float_internal(&[&wrapped_input]).unwrap();

            let input_scaling = ScalingFactor::from_tensor(&input, None);
            let output_native = outputs[0].to_native();
            let output_scaling = ScalingFactor::from_tensor(&output_native, None);

            let norm_max = (layer.normalisation_dim_size() as f32).sqrt();
            let norm_min = -norm_max;
            let std_dev_scaling = ScalingFactor::from_span(norm_min, norm_max, None);

            let QuantizeOutput { quantized_op, .. } = layer
                .quantise(input_scaling, std_dev_scaling, output_scaling)
                .unwrap();

            let quantised_input = input2.quantize(&input_scaling);
            let wrapped_quantised_input = WrappedTensor::try_from(&quantised_input).unwrap();
            let quantised_layer_out = quantized_op
                .evaluate_quantised_internal(&[&wrapped_quantised_input])
                .unwrap();

            let _quantised_output = quantised_layer_out.outputs();
            let LayerNormProvingData {
                mean,
                std_dev,
                lookup_verifier,
            } = quantised_layer_out.try_layernorm_data().unwrap();

            let right_shift = lookup_verifier.right_shift();
            let rounding_constant: Element = 1 << (right_shift - 1);
            let dim_size = quantized_op.normalisation_dim_size();

            let affine_scaling = std_dev_scaling.scale();

            let error_bound = affine_scaling / 2.0f32;

            let quant_error_bound = (error_bound
                / (input_scaling.scale()
                    * input_scaling.scale()
                    * std_dev_scaling.scale()
                    * std_dev_scaling.scale()))
            .round_ties_even() as Element;

            let quant_dim_size = (dim_size as f32
                / (std_dev_scaling.scale() * std_dev_scaling.scale()))
            .round_ties_even() as Element;
            println!("===== Testing LayerNorm dim size {dim_size} =====");

            for (input_chunk, row_mean, std_dev_val) in izip!(
                quantised_input.data().chunks(dim_size),
                mean.get_data().iter(),
                std_dev.get_data().iter()
            ) {
                let sum = input_chunk
                    .iter()
                    .map(|x| {
                        let scaled = x * std_dev_val;
                        let subbed = scaled - row_mean + rounding_constant;
                        let shifted = subbed >> right_shift;
                        shifted * shifted
                    })
                    .sum::<Element>();
                let quant_diff = (quant_dim_size - sum).abs();
                println!(
                    "Row sum of squares: {sum}, quant error bound: {quant_error_bound}, quant error bound log2 {} quant diff: {quant_diff}, diff within bound: {}",
                    ceil_log2(quant_error_bound as usize),
                    quant_diff <= quant_error_bound
                );
                let float_value = (sum as f32) * std_dev_scaling.scale();
                let diff = (dim_size as f32 - float_value).abs();
                println!(
                    "Row size: {dim_size}, float value: {float_value}, error bound: {error_bound}, diff: {diff}, diff within bound: {}",
                    diff <= error_bound
                );
                println!("---------next row---------");
            }
            println!("=============================");
        }
    }

    proptest! {
        #[test]
        fn proptest_layernorm_evaluate_quantised(input in layernorm_input(2usize..768, 1usize..64)) {
            let Input {
                input,
                beta,
                gamma,
            } = input;

            let gamma = gamma.try_map_tensor(|t| {
                let ones = vec![1.0f32; t.shape().numel()];
                Tensor::<f32>::new(t.shape().clone(), ones)
            }).unwrap();

            let beta = beta.try_map_tensor(|t| {
                let zeros = vec![0.0f32; t.shape().numel()];
                Tensor::<f32>::new(t.shape().clone(), zeros)
            }).unwrap();
            let layer = LayerNorm::new(gamma.into(), beta.into(), 1e-5).unwrap();

            let wrapped_input = WrappedTensor::try_from(&input).unwrap();
            let LayerOut { outputs, .. } = layer.evaluate_float_internal(&[&wrapped_input]).unwrap();

            let input_scaling = ScalingFactor::from_tensor(&input, None);
            let output_native = outputs[0].to_native();
            let output_scaling = ScalingFactor::from_tensor(&output_native, None);

            let norm_max = (layer.normalisation_dim_size() as f32).sqrt();
            let norm_min = -norm_max;

            let intermediate_scaling_factor = ScalingFactor::from_span(norm_min, norm_max, None);

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
            let norm_min = intermediate_scaling_factor.min() - 1.0f32;
            let norm_max = intermediate_scaling_factor.max() + 1.0f32;

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

            let _quantised_output = quantised_layer_out.outputs();
            let LayerNormProvingData { mean, std_dev, lookup_verifier } = quantised_layer_out.try_layernorm_data().unwrap();

            let right_shift = lookup_verifier.right_shift();
            let rounding_constant: Element = 1 << (right_shift -1);
            let dim_size = quantized_op.normalisation_dim_size();

            for (input_chunk, row_mean, row_std_dev) in itertools::izip!(quantised_input.data().chunks(dim_size), mean.get_data().iter(), std_dev.get_data().iter()) {
                let rescaled_row = input_chunk.iter().map(|x| {
                    let scaled = x * row_std_dev;
                    let subbed = scaled - row_mean + rounding_constant;
                    subbed >> right_shift
                }).collect::<Vec<Element>>();
                let sum = rescaled_row.iter().sum::<Element>();
                let abs_sum = sum.unsigned_abs() as usize;
                prop_assert!(abs_sum <= dim_size, "Mean not correctly normalised, got sum {sum} expected less than {}, rescaled row: {:?}", dim_size / 2, rescaled_row);
            }
        }

    }
}
