//! Code for evaluating a Softmax layer.

use super::*;

impl Softmax<Element> {
    /// Method that given a quantised input [`Tensor`] calculates the `shift` we apply along each dim and returns the unpadded tensor as the result.
    pub(crate) fn calculate_shift_data_new(
        &self,
        input: &WrappedTensor<Element>,
    ) -> Result<WrappedTensor<Element>> {
        let QuantisedSoftmaxData {
            input_scaling_factor,
            temperature,
            ..
        } = self.quant_info().ok_or(anyhow!("Attempted to calculate shift data for quantised Softmax with no QuantisedSoftmaxData present"))?;
        // Unwrap is safe here because previous line would have errored if quant_info was None
        let negative_infinity = self.quant_info().unwrap().quantised_negative_infinity();

        let scalar = input_scaling_factor.scale() / temperature;
        let input_mask = match input.rank() {
            2 => input.clone().equal_elem::<2>(negative_infinity)?,
            3 => {
                let mask_3d = input.clone().equal_elem::<3>(negative_infinity)?;
                // Reduce to 2D by taking just the first "head"
                mask_3d.slice_dim(0, 0..1).squeeze_dim::<2>(0)
            }
            4 => {
                let mask_4d = input.clone().equal_elem::<4>(negative_infinity)?;
                // Reduce to 2D by taking just the first "batch" and "head"
                mask_4d
                    .slice_dim(0, 0..1)
                    .slice_dim(1, 0..1)
                    .squeeze_dims::<2>(&[0, 1])
            }
            unsupported_rank => bail!("Unsupported input rank: {unsupported_rank}"),
        };

        let dim_maxes = input.clone().max_dim(-1);
        let log_sum_exp = (input.clone().sub(dim_maxes.clone())?)
            .float()
            .mul_scalar(scalar)
            .mask_fill(input_mask, f32::NEG_INFINITY)?
            .exp()
            .sum_dim(-1)
            .log();

        let quantising_scalar = temperature / input_scaling_factor.scale();
        log_sum_exp
            .mul_scalar(-quantising_scalar)
            .round()
            .int()
            .sub(dim_maxes)
    }

    pub(crate) fn evaluate_internal<E: ExtensionField>(
        &self,
        inputs: &[&WrappedTensor<Element>],
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
        let rounding: Element = 1 << (*right_shift - 1);

        // Reduce the input to its unpadded shape if necessary
        let maybe_padded_input = inputs[0].clone();
        let shape = maybe_padded_input.shape();
        let unpadded_shape = maybe_padded_input.unpadded_shape();
        let padded = shape.as_slice() != unpadded_shape.as_slice();
        let input = if padded {
            maybe_padded_input.reduce_to_unpadded_shape()?
        } else {
            maybe_padded_input
        };
        let shape = input.shape();
        let shift_tensor = self.calculate_shift_data_new(&input)?;

        let rescaled_input = input
            .add(shift_tensor.clone())?
            .mul_scalar(*fixed_point_multiplier)
            .add_scalar(rounding)
            .bitwise_right_shift_scalar(*right_shift as Element);

        let rescaled_input_data: Vec<Element> = rescaled_input.to_data().to_vec().map_err(|e| {
            anyhow!("Failed to convert rescaled_input to Vec<Element> in Softmax: {e:?}")
        })?;
        let output_data = rescaled_input_data
            .into_iter()
            .map(|intermediate| {
                if intermediate <= -(1 << lut.table_bit_size()) {
                    0
                } else {
                    lut.table_output(intermediate)
                }
            })
            .collect::<Vec<Element>>();

        let output_tensor =
            WrappedTensor::<Element>::from_data(TensorData::new(output_data, shape))?;

        let shift_shape = shift_tensor.shape();
        let shift_data = shift_tensor.to_data().to_vec().map_err(|e| {
            anyhow!("Failed to convert shift_tensor to Vec<Element> in Softmax: {e:?}")
        })?;

        let shift_tensor = Tensor::<Element>::new(shift_shape.into(), shift_data)?;

        if padded {
            Ok(LayerOut {
                outputs: vec![output_tensor.pad_next_power_of_two()],
                proving_data: ProvingData::Softmax(SoftmaxData { shift_tensor }),
                tracked_layer_data: None,
            })
        } else {
            Ok(LayerOut {
                outputs: vec![output_tensor],
                proving_data: ProvingData::Softmax(SoftmaxData { shift_tensor }),
                tracked_layer_data: None,
            })
        }
    }
}
