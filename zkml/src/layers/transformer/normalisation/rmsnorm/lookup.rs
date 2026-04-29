//! Internal code for generating the [`RMSNorm`] looup witness.

use crate::{ProverContext, lookup::context::LookupWitnessGen, model::Step, to_base};

use super::*;

use multilinear_extensions::util::transpose;
use witness::RowMajorMatrix;

impl RMSNorm<Element> {
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
        let rmsnorm_data = step
            .node_outputs
            .try_rmsnorm_data()
            .ok_or(anyhow!(
                "Could not get RMSNorm proving data for lookup witness generation"
            ))?
            .to_proving_data()?;

        let input = step.input_tensor_at(0)?;
        let input_shape = input.shape().clone();
        let (_, normalisation_scaling_factor) = self.get_quantisation_scaling_factors().ok_or(
            anyhow!("Quantisation scaling factors not found for RMSNorm lookup witness generation"),
        )?;
        let table = Table::new_normalisation(normalisation_scaling_factor.bit_size() + 1);

        let wrapped_input = WrappedTensor::try_from(input.as_ref())?;
        let lookup_output = rmsnorm_data.apply(wrapped_input.clone(), &table)?;
        let scaled = wrapped_input.mul(rmsnorm_data.normalisation.clone())?;
        let scaled_input = Tensor::try_from(&scaled)?;
        let output = Tensor::try_from(&lookup_output)?;

        let rank = input_shape.rank();
        let number_of_chunks = input_shape[..rank.saturating_sub(2)]
            .iter()
            .product::<usize>();

        let lookup_witness = rmsnorm_data.generate_witness(scaled_input, &table)?;

        let mut element_counts = lookup_witness.get_counts(&table);
        let variant = rmsnorm_data.variant();

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

        // Make the commitments to the multipliers
        let dim_size = if rank >= 2 { input_shape[rank - 2] } else { 1 };

        let normalisation_evals = rmsnorm_data
            .normalisation
            .get_data()
            .chunks(dim_size)
            .map(|chunk| {
                let evals = chunk.iter().copied().chain(std::iter::repeat_n(
                    0,
                    dim_size.next_power_of_two() - chunk.len(),
                ));
                to_base::<E, _>(evals)
            })
            .collect::<Vec<Vec<E::BaseField>>>();

        let norm_width = normalisation_evals.len();
        let transposed_norm_evals = transpose(normalisation_evals);

        let normalisation_rmm = RowMajorMatrix::new_by_inner_matrix(
            ceno_p3::matrix::dense::DenseMatrix::new(transposed_norm_evals.concat(), norm_width),
            witness::InstancePaddingStrategy::Default,
        );

        let commit =
            ctx.commitment_ctx
                .batch_commit(vec![input_rmm, output_rmm, normalisation_rmm])?;

        let mut gen_w = LookupWitnessGen::<E, PCS>::default();
        let tables = vec![
            Table::new_shift_check(),
            table,
            Table::new_zero_check(),
            Table::new_signed_zero_check(),
        ];
        gen_w.insert_layer_witness_data(id, commit, tables, element_counts);

        Ok(gen_w)
    }
}
