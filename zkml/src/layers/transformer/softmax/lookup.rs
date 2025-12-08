//! Code related to the lookup part of Softmax layer.

use super::*;

struct RangeChecks {
    number_of_chunks: usize,
    chunks: Vec<Vec<Element>>,
}

impl RangeChecks {
    fn new(number_of_chunks: usize) -> Self {
        Self {
            number_of_chunks,
            chunks: vec![vec![]; number_of_chunks],
        }
    }

    fn push(&mut self, value: Element) {
        let bit_len_mask: Element = (1 << *quantization::BIT_LEN) - 1;
        (0..self.number_of_chunks).for_each(|j| {
            let shift = j * *quantization::BIT_LEN;
            let chunk_val = (value >> shift) & bit_len_mask;
            self.chunks[j].push(chunk_val);
        });
    }

    fn merge(&mut self, other: RangeChecks) {
        assert_eq!(self.number_of_chunks, other.number_of_chunks);
        let RangeChecks { chunks, .. } = other;
        self.chunks
            .iter_mut()
            .zip(chunks)
            .for_each(|(a, b)| a.extend(b));
    }

    fn count_iterator(&self) -> Vec<Element> {
        self.chunks.concat()
    }
}

struct ExpLookup {
    input: Vec<Element>,
    output: Vec<Element>,
}

impl ExpLookup {
    fn new() -> Self {
        Self {
            input: Vec::<Element>::new(),
            output: Vec::<Element>::new(),
        }
    }

    fn push(&mut self, input: Element, output: Element) {
        self.input.push(input);
        self.output.push(output);
    }

    fn merge(&mut self, other: ExpLookup) {
        let ExpLookup { input, output } = other;
        self.input.extend(input);
        self.output.extend(output);
    }

    fn count_iterator(&self) -> Vec<Element> {
        self.input
            .iter()
            .zip(self.output.iter())
            .map(|(a, b)| a + COLUMN_SEPARATOR * b)
            .collect::<Vec<Element>>()
    }
}

struct ZeroChecks {
    number_of_chunks: usize,
    input_chunks: Vec<Vec<Element>>,
    output_chunks: Vec<Vec<Element>>,
}

impl ZeroChecks {
    fn new(number_of_chunks: usize) -> Self {
        Self {
            number_of_chunks,
            input_chunks: vec![vec![]; number_of_chunks],
            output_chunks: vec![vec![]; number_of_chunks],
        }
    }

    fn push(&mut self, input: Element) {
        let bit_len_mask: Element = (1 << *quantization::BIT_LEN) - 1;
        (0..self.number_of_chunks).for_each(|j| {
            let shift = j * *quantization::BIT_LEN;
            let in_val = (input >> shift) & bit_len_mask;

            self.input_chunks[j].push(in_val);

            if in_val != 0 {
                self.output_chunks[j].push(0);
            } else {
                self.output_chunks[j].push(1);
            }
        });
    }

    fn merge(&mut self, other: ZeroChecks) {
        assert_eq!(self.number_of_chunks, other.number_of_chunks);
        let ZeroChecks {
            input_chunks,
            output_chunks,
            ..
        } = other;
        self.input_chunks
            .iter_mut()
            .zip(input_chunks)
            .for_each(|(a, b)| a.extend(b));
        self.output_chunks
            .iter_mut()
            .zip(output_chunks)
            .for_each(|(a, b)| a.extend(b));
    }

    fn count_iterator(&self) -> Vec<Element> {
        self.input_chunks
            .iter()
            .zip(self.output_chunks.iter())
            .flat_map(|(input_chunk, output_chunk)| {
                input_chunk
                    .iter()
                    .zip(output_chunk.iter())
                    .map(|(a, b)| a + COLUMN_SEPARATOR * b)
                    .collect::<Vec<Element>>()
            })
            .collect::<Vec<Element>>()
    }
}

impl Softmax<Element> {
    pub(crate) fn lookup_witness<E: ExtensionField, PCS: PolynomialCommitmentScheme<E>>(
        &self,
        id: NodeId,
        ctx: &ProverContext<E, PCS>,
        input: &Tensor<Element>,
        output: &Tensor<Element>,
        softmax_handle: &SoftmaxHandle,
    ) -> Result<LookupWitnessGen<E, PCS>>
    where
        PCS::CommitmentWithWitness: Serialize + DeserializeOwned + Send + Sync,
        PCS::ProverParam: Send + Sync,
    {
        // Get the data generated during quantised evaluation
        let SoftmaxHandle { shift_handle } = softmax_handle;

        // We need to work out how many chunks to split the normalisation into to be range checked.
        let quant_info = self.quant_info().ok_or(anyhow!(
            "Could not prove Softmax because it had no quantisation data"
        ))?;
        let QuantisedSoftmaxData {
            right_shift,
            fixed_point_multiplier,
            error_bound,
            lut,
            ..
        } = quant_info;
        let allowable_error = (*error_bound * lut.output_sf()).round() as Element;
        let negative_infinity = quant_info.quantised_negative_infinity();

        let unpadded_input_shape = input.unpadded_shape();

        let unpadded_input = input.reduce_to_shape(unpadded_input_shape)?;

        let shape_2d: Shape = unpadded_input_shape[input.rank() - 2..].to_vec().into();
        let chunk_size = shape_2d.numel();

        let total_2d_chunks = unpadded_input_shape[..input.rank() - 2]
            .iter()
            .product::<usize>();

        let final_dim_size = unpadded_input_shape.dim(-1);

        let shift_data_guard = shift_handle.tensor()?;
        let shifted_data = unpadded_input
            .get_data()
            .chunks(final_dim_size)
            .zip(shift_data_guard.get_data())
            .flat_map(|(input_chunk, shift_elem)| {
                input_chunk
                    .iter()
                    .map(|input_elem| input_elem + shift_elem)
                    .collect::<Vec<Element>>()
            })
            .collect::<Vec<Element>>();

        let padded_chunks = shifted_data
            .chunks(chunk_size)
            .map(|chunk| {
                Tensor::<Element>::new(shape_2d.clone(), chunk.to_vec())
                    .expect("Failed to create chunk tensor in SOftmax witness gen")
                    .pad_next_power_of_two_with_value(negative_infinity)
                    .get_data_into()
            })
            .collect::<Vec<Vec<Element>>>();

        // Now we construct the polynomials used in the lookups.
        // These are the sums of the rows after Softmax, we check that these are all within the allowable error of quantised 1.0.
        let unpadded_output = output.reduce_to_shape(unpadded_input_shape)?;
        let normalisation_lookups = unpadded_output
            .get_data_into()
            .chunks(chunk_size)
            .map(|outer_chunk| {
                let num_repeats = final_dim_size.next_power_of_two() - final_dim_size;
                outer_chunk
                    .chunks(final_dim_size)
                    .map(|chunk| chunk.iter().sum::<Element>())
                    .chain(std::iter::repeat_n(0, num_repeats))
                    .collect::<Vec<Element>>()
            })
            .collect::<Vec<Vec<Element>>>();

        // This is the rounding constant used during the fixed point multiplication and right shift
        let rounding: Element = 1 << (*right_shift - 1);
        // This is the bit mask used to extract the bits used in the lookup table for exp after performing the
        // fixed point multiplication and right shift
        let exp_bit_mask = lut.full_table_size() - 1;

        let (chunked_range_checks, chunked_exp_lookup, chunked_zero_checks) = padded_chunks
            .into_par_iter()
            .fold(
                || (vec![], vec![], vec![]),
                |(mut range, mut exp, mut zero), outer_input_chunk| {
                    // For each outer chunk we have to decompose it as we did during inference
                    let (chunk_range_checks, chunk_exp_lookup, chunk_zero_checks) =
                        outer_input_chunk.chunks(final_dim_size).fold(
                            (
                                RangeChecks::new(quant_info.number_of_range_checks()),
                                ExpLookup::new(),
                                ZeroChecks::new(quant_info.number_of_zero_chunks()),
                            ),
                            |(mut outer_range, mut outer_exp, mut outer_zero), input_chunk| {
                                let (inner_range, inner_exp, inner_zero) = input_chunk.iter().fold(
                                    (
                                        RangeChecks::new(quant_info.number_of_range_checks()),
                                        ExpLookup::new(),
                                        ZeroChecks::new(quant_info.number_of_zero_chunks()),
                                    ),
                                    |(mut range_checks, mut exp_lookups, mut zero_checks),
                                     &shifted| {
                                        // Perform fixed point multiplication and add the rounding constant
                                        let scaled = shifted * fixed_point_multiplier + rounding;

                                        let intermediate = scaled >> *right_shift;
                                        // Extract the low bits to be range checked
                                        let low = scaled - (intermediate << *right_shift);
                                        // Extract the bits to be used in the exp lookup
                                        let exp_in = intermediate.abs() & exp_bit_mask;
                                        // If any of the remaining high bits are non-zero we will be out of range for the exp table
                                        // so we check these are zero
                                        let high = intermediate.abs() >> lut.table_bit_size();

                                        range_checks.push(low);
                                        exp_lookups.push(-exp_in, lut.table_output(-exp_in));
                                        zero_checks.push(high);
                                        (range_checks, exp_lookups, zero_checks)
                                    },
                                );
                                outer_range.merge(inner_range);
                                outer_exp.merge(inner_exp);
                                outer_zero.merge(inner_zero);
                                (outer_range, outer_exp, outer_zero)
                            },
                        );
                    range.push(chunk_range_checks);
                    exp.push(chunk_exp_lookup);
                    zero.push(chunk_zero_checks);
                    (range, exp, zero)
                },
            )
            .reduce(
                || (vec![], vec![], vec![]),
                |(mut range_acc, mut exp_acc, mut zero_acc), (range, exp, zero)| {
                    range_acc.extend(range);
                    exp_acc.extend(exp);
                    zero_acc.extend(zero);
                    (range_acc, exp_acc, zero_acc)
                },
            );

        let range_elements_count =
            count_elements(chunked_range_checks.iter().flat_map(|c| c.count_iterator()));
        let exp_elements_count =
            count_elements(chunked_exp_lookup.iter().flat_map(|c| c.count_iterator()));
        let zero_table_elements_count =
            count_elements(chunked_zero_checks.iter().flat_map(|c| c.count_iterator()));

        // We create 3 separate RMMs here, the first corresponds to the lookup inputs, for each chunk the polys are in order
        // range_checks, exp_in, zero_checks_in
        // The second RMM is to do with lookup outputs and the ordering is
        // exp_out, zero_chunks_out
        // The third and final RMM is the shift for each chunk

        let (rmm1_polys, rmm2_polys) = izip!(
            chunked_range_checks,
            chunked_exp_lookup,
            chunked_zero_checks
        )
        .fold(
            (vec![], vec![]),
            |(mut rmm1_acc, mut rmm2_acc), (range_checks, exp_lookup, zero_checks)| {
                let RangeChecks { chunks, .. } = range_checks;
                let ExpLookup { input, output } = exp_lookup;
                let ZeroChecks {
                    input_chunks,
                    output_chunks,
                    ..
                } = zero_checks;
                rmm1_acc.extend(
                    chunks
                        .into_iter()
                        .chain(std::iter::once(input))
                        .chain(input_chunks),
                );
                rmm2_acc.extend(std::iter::once(output).chain(output_chunks));
                (rmm1_acc, rmm2_acc)
            },
        );

        // The width of the first rmm is the number of chunks we decomopose into (given by `quant_info.number_of_range_checks() + 1 + quant_info.number_of_zero_chunks()`)
        // multiplied by the number of 2D tensors we have (given by shift_shape[0])
        let width_one = total_2d_chunks
            * (quant_info.number_of_range_checks() + 1 + quant_info.number_of_zero_chunks());
        let transposed_one = transpose(rmm1_polys);
        let rmm1 = RowMajorMatrix::<E::BaseField>::new_by_inner_matrix(
            ceno_p3::matrix::dense::DenseMatrix::new(
                to_base::<E, _>(transposed_one.into_iter().flatten()),
                width_one,
            ),
            witness::InstancePaddingStrategy::Default,
        );
        // The width of the second rmm is the number of output polys we have (given by 1 + quant_info.number_of_zero_chunks())
        // multiplied by the number of 2D tensors we have (given by shift_shape[0])
        let width_two = total_2d_chunks * (1 + quant_info.number_of_zero_chunks());
        let transposed_two = transpose(rmm2_polys);
        let rmm2 = RowMajorMatrix::<E::BaseField>::new_by_inner_matrix(
            ceno_p3::matrix::dense::DenseMatrix::new(
                to_base::<E, _>(transposed_two.into_iter().flatten()),
                width_two,
            ),
            witness::InstancePaddingStrategy::Default,
        );
        // The final rmm is the shift values, its width is just shift_shape[0]

        let shift_chunk_size = shift_handle.shape()[input.rank() - 2..]
            .iter()
            .product::<usize>();
        let shift_chunk_diff = shift_chunk_size.next_power_of_two() - shift_chunk_size;
        let shift_evals = shift_data_guard
            .get_data()
            .chunks(shift_chunk_size)
            .map(|chunk| {
                chunk
                    .iter()
                    .copied()
                    .chain(std::iter::repeat_n(negative_infinity, shift_chunk_diff))
                    .collect::<Vec<Element>>()
            })
            .collect::<Vec<_>>();
        let shift_transposed = transpose(shift_evals);
        let rmm3 = RowMajorMatrix::<E::BaseField>::new_by_inner_matrix(
            ceno_p3::matrix::dense::DenseMatrix::new(
                to_base::<E, _>(shift_transposed.into_iter().flatten()),
                total_2d_chunks,
            ),
            witness::InstancePaddingStrategy::Default,
        );

        let layer_commit = ctx.commitment_ctx.batch_commit(vec![rmm1, rmm2, rmm3])?;

        let mut gen_w = LookupWitnessGen::<E, PCS>::default();

        // Add the looked up values to the generator so we can make multiplicity polys later
        gen_w.insert_element_count(TableType::Range, range_elements_count);

        // Need to recreate the parameters for the Softmax table
        gen_w.insert_element_count(TableType::ExpTable(*lut), exp_elements_count);

        let quant_one = lut.output_sf() as Element;
        gen_w.insert_element_count(
            TableType::ErrorTable(quant_one, allowable_error),
            count_elements(normalisation_lookups.into_iter().flatten()),
        );

        gen_w.insert_element_count(TableType::ZeroTable, zero_table_elements_count);

        gen_w.insert_logup_witness(id, layer_commit);
        Ok(gen_w)
    }
}
