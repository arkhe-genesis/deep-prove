//! Internal methods for quantising an [`EinSum`] layer's weights and biases.

use crate::{
    ScalingFactor,
    layers::{provable::QuantizeOutput, requant::Requant},
    quantization::{self, Quantize, bias_scaling_matmul},
};

use super::*;

use itertools::izip;
use multilinear_extensions::util::ceil_log2;

/// Returns the scaling factors for the main tensor and for the bias tensor. These are the "model" scaling factors, or
/// S2 in the formula S1 * S2 / S3.
pub fn model_scaling_factor_from_tensor_and_bias(
    input: &ScalingFactor,
    main: &TensorHandle<f32>,
    bias: Option<&TensorHandle<f32>>,
) -> anyhow::Result<(ScalingFactor, ScalingFactor)> {
    let max_weight = main.max_abs()?;
    let max_value = if let Some(bias) = bias {
        let max_bias = bias.max_abs()?;
        max_weight.max(max_bias)
    } else {
        max_weight
    };
    let main_sf = ScalingFactor::from_absolute_max(max_value, None);
    let bias_sf = bias_scaling_matmul(input, &main_sf);
    Ok((main_sf, bias_sf))
}

impl EinSum<f32> {
    pub(crate) fn quantise(
        self,
        input_scaling_factors: &[ScalingFactor],
        output_scaling_factors: &[ScalingFactor],
        unpadded_input_shapes: &[Shape],
    ) -> Result<QuantizeOutput<EinSum<Element>>> {
        // Check that the number of output scaling factors agrees with the number of outputs
        let output_count = self.mapping.output_count();
        ensure!(
            output_scaling_factors.len() == output_count,
            "Number of output scaling factors ({}) does not match number of outputs in equation {} (expected: {} outputs)",
            output_scaling_factors.len(),
            self.equation,
            output_count
        );

        // The first input scaling factor and unpadded_input_shape are for the LHS input
        let mut input_scalings_iter = input_scaling_factors.iter();
        let mut unpadded_input_shapes_iter = unpadded_input_shapes.iter();
        let lhs_input_scaling = *input_scalings_iter
            .next()
            .ok_or(anyhow!("Missing LHS input scaling factor"))?;
        let lhs_unpadded_shape = unpadded_input_shapes_iter
            .next()
            .ok_or(anyhow!("Missing LHS unpadded input shape"))?;

        let mut full_shapes = self
            .constant_tensors
            .iter()
            .map(|t| {
                if let Some(tensor) = t {
                    Ok(tensor.shape().clone())
                } else {
                    // This is an input tensor, so we take the shape from the unpadded input shapes
                    unpadded_input_shapes_iter.next().cloned().ok_or(anyhow!(
                        "Missing unpadded input shape for einsum input tensor"
                    ))
                }
            })
            .collect::<Result<Vec<Shape>>>()?;
        full_shapes.insert(0, lhs_unpadded_shape.clone());

        // Calculates the amount of bits the contraction will add to the output values
        let contracted_size = self.mapping.axes_sizes(&full_shapes)?[AxisType::Contracted];
        let contraction_bits = ceil_log2(contracted_size);
        let (lhs_min, lhs_max) = lhs_input_scaling.domain();
        let lhs_bit_size = ceil_log2(lhs_max.abs().max(lhs_min.abs()) as usize);

        // Now we can iterate through the rhs (that are either constant tensors or inputs) and calculate the requant info for each
        let (requants, quant_weights, quant_biases, output_scalings) = izip!(
            self.constant_tensors.iter(),
            self.biases.iter(),
            output_scaling_factors.iter()
        )
        .fold(
            (vec![], vec![], vec![], vec![]),
            |(mut requants, mut weight, mut bias, mut output_scalings),
             (weight_opt, bias_opt, output_scaling)| {
                let intermediate_bit_size = if bias_opt.is_some() {
                    lhs_bit_size + *quantization::BIT_LEN + contraction_bits + 1 // If we have a bias we need an extra bit for the addition
                } else {
                    lhs_bit_size + *quantization::BIT_LEN + contraction_bits
                };
                if let Some(weight_tensor) = weight_opt {
                    // In this case we have to calculate the scaling factor for the RHS
                    let (weight_scaling, bias_scaling) = model_scaling_factor_from_tensor_and_bias(
                        &lhs_input_scaling,
                        weight_tensor,
                        bias_opt.as_ref(),
                    )
                    .unwrap();
                    let quantized_weight = weight_tensor.quantize(&weight_scaling);
                    let quantized_bias = bias_opt.as_ref().map(|bias| bias.quantize(&bias_scaling));
                    // If `self.requantise` is set to `true` we include a requantisation step after evaluation
                    if self.requantise() {
                        let requant = Requant::from_scaling_factors(
                            lhs_input_scaling,
                            weight_scaling,
                            *output_scaling,
                            intermediate_bit_size,
                        );

                        requants.push(requant);
                        output_scalings.push(*output_scaling);
                    } else {
                        // Otherwise we return no requant step and calculate the output scaling factor accordingly
                        let output_max = output_scaling.max();
                        let output_min = output_scaling.min();
                        let quantised_min: Element = -1 << (intermediate_bit_size - 1);
                        let quantised_max: Element = (1 << (intermediate_bit_size - 1)) - 1;
                        let updated_scaling = ScalingFactor::from_parts(
                            output_max,
                            output_min,
                            bias_scaling.scale(),
                            (quantised_min, quantised_max),
                        );
                        output_scalings.push(updated_scaling);
                    }
                    weight.push(Some(quantized_weight));
                    bias.push(quantized_bias);
                } else {
                    // In this case we need the next input_scaling
                    let rhs_input_scaling = *input_scalings_iter
                        .next()
                        .expect("Missing input scaling factor for einsum input tensor");
                    let (rhs_min, rhs_max) = rhs_input_scaling.domain();
                    let rhs_bit_size = ceil_log2(rhs_max.abs().max(rhs_min.abs()) as usize);
                    let intermediate_bit_size = if bias_opt.is_some() {
                        lhs_bit_size + rhs_bit_size + contraction_bits + 1 // If we have a bias we need an extra bit for the addition
                    } else {
                        lhs_bit_size + rhs_bit_size + contraction_bits
                    };
                    let bias_scaling = bias_scaling_matmul(&lhs_input_scaling, &rhs_input_scaling);
                    let quantized_bias = bias_opt.as_ref().map(|bias| bias.quantize(&bias_scaling));
                    // If `self.requantise` is set to `true` we include a requantisation step after evaluation
                    if self.requantise() {
                        let requant = Requant::from_scaling_factors(
                            lhs_input_scaling,
                            rhs_input_scaling,
                            *output_scaling,
                            intermediate_bit_size,
                        );

                        requants.push(requant);
                        output_scalings.push(*output_scaling);
                    } else {
                        // Otherwise we return no requant step and calculate the output scaling factor accordingly
                        let output_max = output_scaling.max();
                        let output_min = output_scaling.min();
                        let quantised_min: Element = -1 << (intermediate_bit_size - 1);
                        let quantised_max: Element = (1 << (intermediate_bit_size - 1)) - 1;
                        let updated_scaling = ScalingFactor::from_parts(
                            output_max,
                            output_min,
                            bias_scaling.scale(),
                            (quantised_min, quantised_max),
                        );
                        output_scalings.push(updated_scaling);
                    }

                    weight.push(None);
                    bias.push(quantized_bias);
                }
                (requants, weight, bias, output_scalings)
            },
        );

        let quantized_op = EinSum {
            equation: self.equation,
            mapping: self.mapping,
            evaluation_info: self.evaluation_info,
            constant_tensors: quant_weights,
            constant_unpadded_shapes: self.constant_unpadded_shapes,
            biases: quant_biases,
            bias_unpadded_shapes: self.bias_unpadded_shapes,
            padded: self.padded,
            caches: self.caches,
            requantise: self.requantise,
        };

        if !requants.is_empty() {
            QuantizeOutput::new(quantized_op, output_scalings).with_requants(requants)
        } else {
            Ok(QuantizeOutput::new(quantized_op, output_scalings))
        }
    }
}
