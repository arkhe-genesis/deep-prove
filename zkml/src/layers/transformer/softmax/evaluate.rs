//! Code for evaluating a Softmax layer.

use crate::lookup::operation::LookupOp;

use super::*;

impl Softmax<Element> {
    pub(crate) fn evaluate_internal(
        &self,
        inputs: &[&WrappedTensor<Element>],
    ) -> Result<LayerOut<Element>> {
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
        let quant_info = self.quant_info().ok_or(anyhow!(
            "Attempted to evaluate quantised Softmax with no QuantisedSoftmaxData present"
        ))?;

        // Reduce the input to its unpadded shape if necessary
        let input = if inputs[0].is_padded() {
            inputs[0].clone().reduce_to_unpadded_shape()?
        } else {
            inputs[0].clone()
        };

        let shift_tensor = self.calculate_shift_data(&input)?;
        let input = input
            .mul_scalar(quant_info.fixed_point_multiplier())
            .sub(shift_tensor.clone())?;
        let output_tensor = quant_info.apply(input, &quant_info.lut)?;

        if inputs[0].is_padded() {
            Ok(LayerOut {
                outputs: vec![output_tensor.pad_next_power_of_two()],
                proving_data: ProvingData::Softmax(SoftmaxData { shift_tensor }),
                tracked_layer_data: Default::default(),
            })
        } else {
            Ok(LayerOut {
                outputs: vec![output_tensor],
                proving_data: ProvingData::Softmax(SoftmaxData { shift_tensor }),
                tracked_layer_data: Default::default(),
            })
        }
    }

    /// Method that given a quantised input [`Tensor`] calculates the `shift` we apply along each dim and returns the unpadded tensor as the result.
    pub(crate) fn calculate_shift_data(
        &self,
        input: &WrappedTensor<Element>,
    ) -> Result<WrappedTensor<Element>> {
        if self.shift_cache_initialised() {
            // The cache is initialised, if the current cache sequence length matches
            // the last two dimensions of the shape then just use the cached tensor. Otherwise
            // calculate the new row shift and add it to the cache.
            let shift_cache = self.shift_cache.lock().unwrap();
            let current_sequence_length = shift_cache.current_sequence_length();
            let second_last_dim = input.dim(-2)?;
            let last_dim = input.dim(-1)?;
            let dims_equal = second_last_dim == last_dim;
            if dims_equal && current_sequence_length == last_dim {
                return shift_cache.get_cached();
            }
        }

        // Get the quant info
        let quant_info = self.quant_info().ok_or(anyhow!(
            "Could not calculate quantised softmax shift data as there was no quantisation data"
        ))?;
        let input_scaling_factor = quant_info.input_scaling_factor;
        let temperature = quant_info.temperature;
        let lut = quant_info.lut;

        let negative_infinity = quant_info.quantised_negative_infinity();

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

        let min_table_float = 1.0f32 / (lut.input_scale_factor() * 2.0f32);
        let rescaled_dim_maxes = match dim_maxes.rank() {
            2 => {
                let tensor_zeroes = Tensor::<Element>::zeros(dim_maxes.shape().into());
                let zeroes = WrappedTensor::try_from(tensor_zeroes)?;
                let mask_2d = log_sum_exp
                    .clone()
                    .abs()
                    .lower_equal_elem::<2>(min_table_float)?;

                let rounding_sub = zeroes.mask_fill(mask_2d, quant_info.rounding_constant())?;
                dim_maxes
                    .mul_scalar(quant_info.fixed_point_multiplier())
                    .add(rounding_sub)?
            }
            3 => {
                let tensor_zeroes = Tensor::<Element>::zeros(dim_maxes.shape().into());
                let zeroes = WrappedTensor::try_from(tensor_zeroes)?;
                let mask_3d = log_sum_exp
                    .clone()
                    .abs()
                    .lower_equal_elem::<3>(min_table_float)?;

                let rounding_sub = zeroes.mask_fill_3d(mask_3d, quant_info.rounding_constant())?;
                dim_maxes
                    .mul_scalar(quant_info.fixed_point_multiplier())
                    .add(rounding_sub)?
            }
            4 => {
                let tensor_zeroes = Tensor::<Element>::zeros(dim_maxes.shape().into());
                let zeroes = WrappedTensor::try_from(tensor_zeroes)?;
                let mask_4d = log_sum_exp
                    .clone()
                    .abs()
                    .lower_equal_elem::<4>(min_table_float)?;

                let rounding_sub = zeroes.mask_fill_4d(mask_4d, quant_info.rounding_constant())?;
                dim_maxes
                    .mul_scalar(quant_info.fixed_point_multiplier())
                    .add(rounding_sub)?
            }
            unsupported_rank => bail!("Unsupported input rank: {unsupported_rank}"),
        };

        let rescaled_log_sum_exp = log_sum_exp.mul_scalar(lut.input_scale_factor());

        let shift_tensor = rescaled_log_sum_exp
            .mul_scalar((1u64 << quant_info.right_shift()) as f32)
            .round()
            .int()
            .add(rescaled_dim_maxes)?;

        let mut cache = self.shift_cache.lock().unwrap();
        let _ = cache.concatenate(shift_tensor.clone());
        Ok(shift_tensor)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{backend::Backend, quantization::Quantize};
    use burn::tensor::{Bool, Tensor as BTensor};
    use proptest::prelude::*;

    #[derive(Clone)]
    struct Input {
        pub heads: usize,
        pub n: usize,
        pub flat_floats: Vec<f32>,
    }

    impl std::fmt::Debug for Input {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.debug_struct("Input")
                .field("heads", &self.heads)
                .field("n", &self.n)
                .finish()
        }
    }

    fn input_strategy() -> impl Strategy<Value = Input> {
        (1usize..=4_usize, 128usize..=256_usize).prop_flat_map(|(heads, n)| {
            prop::collection::vec(-4.0f32..4.0f32, heads * n * n).prop_map(move |v| Input {
                heads,
                n,
                flat_floats: v,
            })
        })
    }

    proptest! {
        /// Checks that [`Softmax::calculate_shift_data_new`] produces the same integer
        /// shift when processing a full `[heads, n, n]` causally-masked attention matrix all
        /// at once as when processing each query position individually with only its valid
        /// prefix (`[heads, 1, row+1]`).
        ///
        /// This exactly mirrors the full-trace vs cached-trace scenario: the full trace calls
        /// `calculate_shift_data_new` once on the entire `[heads, n, n]` matrix (with the
        /// upper triangle set to `neg_inf`), whereas the cached trace calls it n times, each
        /// time on `[heads, 1, row+1]` containing only the valid context logits for that
        /// query position across all heads. A counterexample proves the discrepancy is real;
        /// passing builds confidence it is not.
        #[test]
        fn prop_shift_matches_row_by_row(
            input in input_strategy()
        ) {
            let Input { heads, n, flat_floats} = input;
            let max_context = n.next_power_of_two().max(4);

            // Derive the scaling factor from all heads*n*n float values, as a real Q@K^T
            // quantisation step would (before masking).
            let float_matrix = Tensor::new(vec![heads, n, n].into(), flat_floats.clone()).unwrap();
            let scaling = ScalingFactor::from_tensor(&float_matrix, None);
            // Skip degenerate matrices where all values are identical (scale ≈ 0).
            prop_assume!(scaling.scale() > 0.0);

            let quant_matrix = float_matrix.quantize(&scaling);

            // Build quantised Softmax using the same input scaling.
            let softmax = Softmax::<f32>::new(max_context)
                .quantise(scaling)
                .unwrap();
            let neg_inf = softmax.quant_info().unwrap().quantised_negative_infinity();
            // Skip if any valid logit coincides with the sentinel.
            prop_assume!(quant_matrix.data().iter().all(|&v| v != neg_inf));

            // Build the full [heads, n, n] causal matrix: upper triangle of each head's
            // [n, n] attention matrix set to neg_inf.  All heads share the same causal
            // pattern, which is what the 3-D mask branch in calculate_shift_data
            // assumes (it reads the mask from the first head only).
            let mask = BTensor::<Backend, 2, Bool>::tril_mask([n, n], 0, &Default::default());
            let wrapped_quant = WrappedTensor::try_from(quant_matrix).unwrap();
            let casual_tensor = wrapped_quant.mask_fill(mask, neg_inf).unwrap();

            // Cached-trace: for each query position, process [heads, 1, row+1] — all heads
            // together but only the valid key prefix for that position.
            let mut all_row_shifts = Vec::<Vec<Element>>::with_capacity(n);
            for (row, full_row_tensor) in casual_tensor.clone().iter_dim(1).enumerate() {
                let valid_len = row + 1;
                let row_tensor = full_row_tensor.slice_dim(2, 0..valid_len);

                // Output has shape [heads, 1, 1] — one shift value per head.
                let row_shift = softmax.calculate_shift_data(&row_tensor).unwrap();
                let row_shift_val: Vec<Element> = row_shift.to_data().to_vec().unwrap();

                all_row_shifts.push(row_shift_val);

            }

            // Full-trace: process all heads and rows at once.
            // Output shift tensor has shape [heads, n, 1], laid out as heads*n elements.
            let full_shifts = softmax.calculate_shift_data(&casual_tensor).unwrap();
            let full_shift_data: Vec<Element> = full_shifts.to_data().to_vec().unwrap();

            for (row, row_shift_val) in all_row_shifts.into_iter().enumerate() {
                for head in 0..heads {
                    let full_val = full_shift_data[head * n + row];
                    let cached_val = row_shift_val[head];
                    prop_assert_eq!(
                        full_val, cached_val,
                        "head={} row={} shift mismatch: full={}, cached={}, n={}, heads={}",
                        head, row, full_val, cached_val, n, heads,
                    );
                }
            }
        }
    }
}
