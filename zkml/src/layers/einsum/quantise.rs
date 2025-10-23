//! Internal methods for quantising an [`EinSum`] layer's weights and biases.

use crate::{
    ScalingFactor,
    layers::{provable::QuantizeOutput, requant::Requant},
    quantization::{self, bias_scaling_matmul, model_scaling_factor_from_tensor_and_bias},
};

use super::*;

use itertools::izip;
use multilinear_extensions::util::ceil_log2;

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
        let intermediate_bit_size = 2 * (*quantization::BIT_LEN - 1) + contraction_bits + 1;
        // Now we can iterate through the rhs (that are either constant tensors or inputs) and calculate the requant info for each
        let (requants, quant_weights, quant_biases) = izip!(
            self.constant_tensors.iter(),
            self.biases.iter(),
            output_scaling_factors.iter()
        )
        .fold(
            (vec![], vec![], vec![]),
            |(mut requants, mut weight, mut bias), (weight_opt, bias_opt, output_scaling)| {
                let intermediate_bit_size = if bias_opt.is_some() {
                    intermediate_bit_size + 1 // If we have a bias we need an extra bit for the addition
                } else {
                    intermediate_bit_size
                };
                if let Some(weight_tensor) = weight_opt {
                    // In this case we have to calculate the scaling factor for the RHS
                    let (weight_scaling, bias_scaling) = model_scaling_factor_from_tensor_and_bias(
                        &lhs_input_scaling,
                        weight_tensor,
                        &bias_opt.clone().map(|t| t.tensor()),
                    );
                    let quantized_weight = weight_tensor
                        .clone()
                        .map_tensor(|t| t.to_quantized(&weight_scaling));
                    let quantized_bias = bias_opt
                        .clone()
                        .map(|bias| bias.map_tensor(|t| t.to_quantized(&bias_scaling)));
                    let requant = Requant::from_scaling_factors(
                        lhs_input_scaling,
                        weight_scaling,
                        *output_scaling,
                        intermediate_bit_size,
                    );
                    requants.push(requant);
                    weight.push(Some(quantized_weight));
                    bias.push(quantized_bias);
                } else {
                    // In this case we need the next input_scaling
                    let rhs_input_scaling = *input_scalings_iter
                        .next()
                        .expect("Missing input scaling factor for einsum input tensor");
                    let bias_scaling = bias_scaling_matmul(&lhs_input_scaling, &rhs_input_scaling);
                    let quantized_bias = bias_opt
                        .clone()
                        .map(|bias| bias.map_tensor(|t| t.to_quantized(&bias_scaling)));
                    let requant = Requant::from_scaling_factors(
                        lhs_input_scaling,
                        rhs_input_scaling,
                        *output_scaling,
                        intermediate_bit_size,
                    );
                    requants.push(requant);
                    weight.push(None);
                    bias.push(quantized_bias);
                }
                (requants, weight, bias)
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
        };
        Ok(
            QuantizeOutput::new(quantized_op, output_scaling_factors.to_vec())
                .with_requants(requants),
        )
    }
}
