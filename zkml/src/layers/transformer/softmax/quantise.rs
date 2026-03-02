//! Module containing code for quantising [`Softmax`] layers.
use crate::lookup::table::SHIFT_CHECK_TABLE_BIT_SIZE;

use super::*;

impl Softmax<f32> {
    /// Method to quantise the [`Softmax`] operation, this takes in the input scaling factor and the intermediate bit size.
    /// The returned [`Softmax`] will have the [`QuantisedSoftmaxData`] set.
    pub fn quantise(&self, input_scaling: ScalingFactor) -> Result<Softmax<Element>> {
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

        let temperature = self.scalar;

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

        let intermediate_bit_size = input_scaling.bit_size() + 1;
        // Assertion to check that we can perform requantisation, we need intermediate_bit_size + fp_scale <= 63
        ensure!(
            intermediate_bit_size + fp_scale <= 63,
            "intermediate bit size: {intermediate_bit_size}, fp scale: {fp_scale}, int part: {int_part}",
        );
        // Now we can create the ExpTable
        let lut = Table::new_exp(input_sf, output_sf, table_bit_size);

        // Since we always want to have atleast one zero chunk (for padding/masking purposes) we set the intermediate bit size to be at least the table bit size + zero check bit size
        let intermediate_bit_size =
            intermediate_bit_size.max(table_bit_size + ZERO_CHECK_TABLE_BIT_SIZE);

        let tmp_right_shift = int_part - FIXED_POINT_SCALE as isize;
        ensure!(
            tmp_right_shift < 0,
            "Right shift must be negative as an isize, got {tmp_right_shift}"
        );

        // The padding value to use should be the most negative value that fits into the intermediate bit size
        let negative_infinity: Element = -1 << (intermediate_bit_size - 1);

        let quant_info = QuantisedSoftmaxData {
            right_shift: tmp_right_shift.unsigned_abs(),
            fixed_point_multiplier,
            negative_infinity,
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
            shift_cache: Arc::new(Mutex::new(ConcatenationCache::<Element>::new_dynamic(
                -2,
                PaddingMode::NoPadding,
            ))),
        })
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
        let max_context_size = self.max_size as f32;

        // This works out the maximum possible output bitsize based on the fact that the following layer will have
        // to do a matrix multiplication of size `max_context_size` and that the Primefield we are using allows for 63 bits.
        let max_poss_out_sf_log =
            (63 - ceil_log2(self.max_size) - FIXED_POINT_SCALE - *quantization::BIT_LEN)
                .min(SHIFT_CHECK_TABLE_BIT_SIZE);

        let output_sf = 1 << max_poss_out_sf_log;
        // Then from this we can work out the rounding error incurred on each output of the exp table
        let table_rounding = 1.0 / (2.0 * output_sf as f32);

        // Now we need the lower bound on the exp table to be such that (-table_lower_bound).exp() <= table_rounding => -table_lower_bound <= table_rounding.ln()
        let table_rounding_ln = table_rounding.ln();
        let table_lower_bound = -table_rounding_ln;

        // Now we work out the pair (last_table_value, input_sf) such that the relative error given by (1.0 / (2.0 * input_sf)).exp() - 1.0 <= 0.01 - max_context_size * table_rounding
        // So we iterate through powers of two calculating temp_sf = table_lower_bound / 2^i until 1.0 / (2.0 * (1.0 + 0.01 - max_context_size * table_rounding).ln()) <= temp_sf
        let mut initial_power = *quantization::BIT_LEN;
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

        if initial_power < SHIFT_CHECK_TABLE_BIT_SIZE {
            initial_power = SHIFT_CHECK_TABLE_BIT_SIZE;
            input_sf = (1 << initial_power) as f32 / table_lower_bound;
        }

        // The case that may cause issues seems to always be when all the values on a row are the same, so we quickly check here what the error for that case would be
        let temperature = Number::to_f32(&self.scalar).unwrap_or(1.0);
        let all_same_shift = (-(max_context_size.ln()) / (input_scaling.scale() * temperature))
            .round_ties_even() as Element;
        let rescaling_mult = input_scaling.scale() * temperature * input_sf;

        // Now we need to convert this rescaling multiplier into a fixed point multiplier and a right shift.
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
