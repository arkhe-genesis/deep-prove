//! Code for generating lookup witnesses for the Requant layer.

use dp_crypto::IntoMLE;

use super::*;

impl Requant {
    pub(crate) fn lookup_witness<F: PrimeField>(
        &self,
        id: NodeId,
        input: &Tensor<Element>,
    ) -> Result<LookupWitnessGen<'_, F>> {
        let lookup_data = self.activation_lookup_data;
        let lookup_witness = lookup_data.get_lookup_witness(input.clone())?;
        let element_counts = lookup_witness.get_counts(&lookup_data.table);

        let input_evals = lookup_witness.input_mle_evals::<F>(lookup_data.table.num_columns());
        let output_evals = lookup_witness.output_mle_evals::<F>();

        let mles = input_evals
            .into_iter()
            .chain(output_evals)
            .map(|evals| evals.into_mle())
            .collect();

        let mut gen_w = LookupWitnessGen::<F>::default();
        let tables = vec![
            Table::new_shift_check(),
            lookup_data.table,
            Table::new_zero_check(),
            Table::new_signed_zero_check(),
        ];
        gen_w.insert_layer_witness_data(id, mles, tables, element_counts);

        Ok(gen_w)
    }
}
