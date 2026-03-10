//! Code related to the lookup part of Softmax layer.

use super::*;

use crate::lookup::{
    operation::{
        LookupOp, LookupOpWitness,
        decomposer::{ChunkedInput, ChunkedOutput},
        variant::LookupVariant,
    },
    table::Table,
};

impl LookupOp for QuantisedSoftmaxData {
    fn intermediate_bit_size(&self) -> usize {
        self.intermediate_bit_size
    }

    fn right_shift(&self) -> usize {
        self.right_shift
    }

    fn variant(&self) -> LookupVariant {
        let normalised_sum_value = self.lut.output_scale_factor() as Element;
        let error_bound = (self.error_bound * self.lut.output_scale_factor()).round() as Element;
        LookupVariant::Softmax {
            normalised_sum_value,
            error_bound,
        }
    }

    /// Softmax uses a custom witness generation method, which takes the input tensor after rescaling and shifting have been applied, pads and rescales as necessary, and then decomposes into chunks to be passed to the lookup tables. This method returns that witness.
    fn generate_witness(
        &self,
        input: Tensor<Element>,
        value_table: &Table,
    ) -> Result<LookupOpWitness> {
        // Break the input tensor down into chunks over the final two dimensions
        let rank = input.shape().rank();
        let (second_last_dim, last_dim) = match rank {
            1 => (1, input.shape()[0]), // If rank 1, we treat as a single row
            r if r >= 2 => (input.shape()[rank - 2], input.shape()[rank - 1]),
            _ => bail!("Input tensor must have rank at least 1"),
        };

        let chunk_size = second_last_dim * last_dim;
        // We work out the size of the padded chunks so we can correctly pad the input before decomposition
        let padded_last_dim = last_dim.next_power_of_two();
        let last_dim_diff = padded_last_dim - last_dim;
        let padded_second_last_dim = second_last_dim.next_power_of_two();
        let chunk_diff = (padded_second_last_dim - second_last_dim) * padded_last_dim;
        // Create the ChunkingInfo to handle the decomposition of each chunk
        let chunking_info = self.chunking_info(value_table)?;
        // This takes each unpadded chunk, pads with the correct padding value, rescales and adds rounding constant, and then decomposes into chunks to be passed
        // to the various lookup tables.
        let chunked_inputs = input
            .into_data()
            .chunks(chunk_size)
            .map(|chunk| {
                // For each unpadded 2D chunk, pad with the correct padding value.
                let chunk_data = chunk
                    .chunks(last_dim)
                    .flat_map(|row| {
                        row.iter()
                            .map(|x| x + self.rounding_constant())
                            .chain(std::iter::repeat_n(
                                self.padding_value() * self.fixed_point_multiplier()
                                    + self.rounding_constant(),
                                last_dim_diff,
                            ))
                    })
                    .chain(std::iter::repeat_n(
                        self.padding_value() * self.fixed_point_multiplier()
                            + self.rounding_constant(),
                        chunk_diff,
                    ))
                    .collect::<Vec<Element>>();

                chunking_info.decompose_input(chunk_data)
            })
            .collect::<Vec<ChunkedInput>>();

        // Now we take the decomposed input chunks and perform lookups on each chunk to get the output chunks
        let chunked_outputs = chunked_inputs
            .iter()
            .map(|chunked_input| chunking_info.table_output(chunked_input, value_table))
            .collect::<Result<Vec<ChunkedOutput>>>()?;

        Ok(LookupOpWitness::new(chunked_inputs, chunked_outputs))
    }

    /// In [`Softmax`] we assume that the `input` has already had the `shift_data` added to it.
    fn apply(
        &self,
        input: WrappedTensor<Element>,
        table: &Table,
    ) -> Result<WrappedTensor<Element>> {
        let shifted_input = input
            .add_scalar(self.rounding_constant())
            .neg() // We have to negate, shift and then negate again to provide consistency with the proving side.
            .bitwise_right_shift_scalar(self.right_shift() as Element)
            .neg();
        table.lookup_tensor(shifted_input)
    }

    fn fixed_point_multiplier(&self) -> Element {
        self.fixed_point_multiplier
    }

    fn is_signed(&self) -> bool {
        self.lut.is_signed()
    }

    fn padding_value(&self) -> Element {
        self.quantised_negative_infinity()
    }
}

impl Softmax<Element> {
    pub(crate) fn lookup_witness<E: ExtensionField, PCS: PolynomialCommitmentScheme<E>>(
        &self,
        id: NodeId,
        ctx: &ProverContext<E, PCS>,
        step: &Step<Element>,
    ) -> Result<LookupWitnessGen<E, PCS>>
    where
        PCS::CommitmentWithWitness: Serialize + DeserializeOwned + Send + Sync,
        PCS::ProverParam: Send + Sync,
    {
        // Get the data generated during quantised evaluation
        let SoftmaxData { shift_tensor } = step
            .node_outputs
            .try_softmax_data()
            .ok_or(anyhow!(
                "No Softmax Proving Data in Step, cannot generate lookup witness"
            ))?
            .to_proving_data()?;

        let quant_info = self.quant_info().ok_or(anyhow!(
            "Could not generate lookup witness for Softmax because it had no quantisation data"
        ))?;
        let table = &quant_info.lut;
        let input = step.input_tensor_at(0)?;
        let input_shape = input.shape().clone();

        let prepped = WrappedTensor::try_from(input.as_ref())?
            .mul_scalar(quant_info.fixed_point_multiplier())
            .sub(shift_tensor.clone())?;
        let witness_gen_input = Tensor::try_from(&prepped)?;
        let rank = input_shape.rank();
        let number_of_chunks = input_shape[..rank.saturating_sub(2)]
            .iter()
            .product::<usize>();

        let lookup_witness = quant_info.generate_witness(witness_gen_input, table)?;

        let mut element_counts = lookup_witness.get_counts(table);
        let variant = quant_info.variant();

        let output = step.output_tensor_at(0)?;
        let mut norm_counts =
            variant.compute_normalisation_witness_counts(number_of_chunks, output.as_ref())?;

        // Merge in the normalisation counts
        let norm_map = norm_counts.remove(0);
        for (key, val) in norm_map {
            let entry = element_counts[0].entry(key).or_insert(0);
            *entry += val;
        }

        let input_evals = lookup_witness.input_mle_evals::<E>(table.num_columns());
        let input_width = input_evals.len();
        let output_evals = lookup_witness.output_mle_evals::<E>();
        let output_width = output_evals.len();

        // Add the witness polynomials that we need to commit to
        let transposed_input = transpose(input_evals);
        let input_rmm = RowMajorMatrix::new_by_inner_matrix(
            ceno_p3::matrix::dense::DenseMatrix::new(transposed_input.concat(), input_width),
            witness::InstancePaddingStrategy::Default,
        );

        let transposed_output = transpose(output_evals);
        let output_rmm = RowMajorMatrix::new_by_inner_matrix(
            ceno_p3::matrix::dense::DenseMatrix::new(transposed_output.concat(), output_width),
            witness::InstancePaddingStrategy::Default,
        );

        // Make the commitments to the shift tensor
        let unpadded_dim_size = if rank >= 2 { input_shape[rank - 2] } else { 1 };

        let shift_evals = shift_tensor
            .get_data()
            .chunks(unpadded_dim_size)
            .map(|chunk| {
                let evals = chunk.iter().copied().chain(std::iter::repeat_n(
                    0,
                    unpadded_dim_size.next_power_of_two() - chunk.len(),
                ));
                to_base::<E, _>(evals)
            })
            .collect::<Vec<Vec<E::BaseField>>>();

        let shift_width = shift_evals.len();
        let transposed_shift_evals = transpose(shift_evals);

        let shift_rmm = RowMajorMatrix::new_by_inner_matrix(
            ceno_p3::matrix::dense::DenseMatrix::new(transposed_shift_evals.concat(), shift_width),
            witness::InstancePaddingStrategy::Default,
        );

        let commit = ctx
            .commitment_ctx
            .batch_commit(vec![input_rmm, output_rmm, shift_rmm])?;

        let mut gen_w = LookupWitnessGen::<E, PCS>::default();
        let tables = vec![
            Table::new_shift_check(),
            *table,
            Table::new_zero_check(),
            Table::new_signed_zero_check(),
        ];
        gen_w.insert_layer_witness_data(id, commit, tables, element_counts);

        Ok(gen_w)
    }
}
