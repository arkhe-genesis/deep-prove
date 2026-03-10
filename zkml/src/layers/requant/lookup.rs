//! Code for generating lookup witnesses for the Requant layer.

use super::*;

impl Requant {
    pub(crate) fn lookup_witness<E: ExtensionField, PCS: PolynomialCommitmentScheme<E>>(
        &self,
        id: NodeId,
        ctx: &ProverContext<E, PCS>,
        input: &Tensor<Element>,
    ) -> Result<LookupWitnessGen<E, PCS>>
    where
        PCS::CommitmentWithWitness: Serialize + DeserializeOwned + Send + Sync,
    {
        let lookup_data = self.activation_lookup_data;
        let lookup_witness = lookup_data.get_lookup_witness(input.clone())?;
        let element_counts = lookup_witness.get_counts(&lookup_data.table);

        let input_evals = lookup_witness.input_mle_evals::<E>(lookup_data.table.num_columns());
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

        let commit = ctx
            .commitment_ctx
            .batch_commit(vec![input_rmm, output_rmm])?;

        let mut gen_w = LookupWitnessGen::<E, PCS>::default();
        let tables = vec![
            Table::new_shift_check(),
            lookup_data.table,
            Table::new_zero_check(),
            Table::new_signed_zero_check(),
        ];
        gen_w.insert_layer_witness_data(id, commit, tables, element_counts);

        Ok(gen_w)
    }
}
