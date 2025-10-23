//! Module containing specific evaluation related code for the [`EinSum`] layer.

use super::*;

use crate::{Shape, Tensor, layers::concat_matmul::Permutation};

use anyhow::{Result, anyhow};
use burn::prelude::Shape as BShape;
use itertools::{Itertools, izip};

impl<N> EinSum<N>
where
    N: TensorTypeParam,
{
    /// Convert the given input [Tensors][Tensor] to 3D [BurnTensors][BurnTensor], applying an optional [Permutation] and reshaping.
    /// Then performs the [`EinSum`] operation using batched matrix multiplications and performs any bias additions.
    ///
    /// # Arguments
    ///
    /// * `inputs` - The input tensors to be converted.
    ///
    /// # Returns
    ///
    /// * `Vec<Tensor<N>>` - A vector of output tensors resulting from the einsum operation.
    ///
    /// This method will error if the provided input tensors do not match the expected shapes
    /// as defined by the einsum equation or if any tensor operations fail.
    pub(crate) fn evaluate_internal(
        &self,
        inputs: &[&WrappedTensor<N>],
        unpadded_input_shapes: &[Shape],
    ) -> Result<Vec<WrappedTensor<N>>> {
        // Prepare the input tensors, applying permutations and reshaping as needed
        let mut unpadded_inputs_iter = inputs
            .iter()
            .zip(unpadded_input_shapes.iter())
            .map(|(&input_tens, unpadded_shape)| {
                input_tens
                    .clone()
                    .reduce_to_shape(&BShape::from(unpadded_shape.as_slice()))
            })
            .collect::<Result<Vec<WrappedTensor<N>>>>()?
            .into_iter();
        // The LHS is never a constant tensor, so we take it from the inputs
        let lhs_input = unpadded_inputs_iter
            .next()
            .ok_or(anyhow!("No input tensors provided"));

        let unpadded_inputs = std::iter::once(lhs_input)
            .chain(
                self.constant_tensors
                    .iter()
                    .zip(self.constant_unpadded_shapes.iter())
                    .map(|(opt_const, unpadded_shape)| {
                        if let (Some(const_tensor), Some(unpadded_shape)) =
                            (opt_const, unpadded_shape)
                        {
                            WrappedTensor::try_from(&const_tensor.reduce_to_shape(unpadded_shape))
                        } else {
                            unpadded_inputs_iter
                                .next()
                                .ok_or_else(|| anyhow!("Not enough input tensors provided"))
                        }
                    }),
            )
            .collect::<Result<Vec<WrappedTensor<N>>>>()?;

        // If the stack_axes_size is 1 and the lhs has rank 2 then we can make 2D matmuls instead of 3D batched matmuls
        let unpadded_shapes = unpadded_inputs
            .iter()
            .map(|t| {
                let BShape { dims } = t.shape();
                Shape::new(dims)
            })
            .collect::<Vec<_>>();
        let stack_axes_size = self.mapping.axes_sizes(&unpadded_shapes)?[AxisType::Stacked];
        let lhs_rank = unpadded_shapes[0].rank();
        if lhs_rank <= 2 && stack_axes_size == 1 {
            self.burn_evaluation::<2>(unpadded_inputs, &unpadded_shapes)
        } else {
            self.burn_evaluation::<3>(unpadded_inputs, &unpadded_shapes)
        }
    }

    /// Internal method that performs the [`EinSum`] operation using the Burn library.
    /// The const generic parameter `D` represents the rank of the tensors being processed (either 2 or 3).
    fn burn_evaluation<const D: usize>(
        &self,
        inputs: Vec<WrappedTensor<N>>,
        shapes: &[Shape],
    ) -> Result<Vec<WrappedTensor<N>>> {
        // Ensure that D is either 2 or 3
        ensure!(
            D == 2 || D == 3,
            "EinSum Burn evaluation only supports rank 2 or 3 tensors"
        );
        // Check that the input shapes are compatible with the einsum equation
        self.mapping.check_shapes(shapes)?;

        let mut prepped_inputs = izip!(
            inputs,
            self.evaluation_info.input_permutations(),
            self.evaluation_info.input_reshapes()
        )
        .map(|(input, permutation, reshape)| {
            let permuted = if let Some(perm) = permutation {
                input.permute(&perm.0.iter().map(|&d| d as isize).collect::<Vec<_>>())?
            } else {
                input
            };

            let permuted_shape = permuted.shape();

            let mut skip = 0;
            let mut reshape_array = [0usize; D];

            reshape_array
                .iter_mut()
                .zip(reshape[3 - D..].iter())
                .for_each(|(new_dim, to_take)| {
                    *new_dim = permuted_shape
                        .dims
                        .iter()
                        .skip(skip)
                        .take(*to_take)
                        .product();
                    skip += *to_take;
                });
            permuted.reshape(BShape::from(reshape_array.as_slice()))
        })
        .collect::<Result<Vec<WrappedTensor<N>>>>()?;

        // Remove the LHS input from the prepared inputs
        let lhs_burn = prepped_inputs.remove(0);

        // Iterate through the RHS inputs and perform batched matmuls
        let intermediate_results = prepped_inputs
            .into_iter()
            .map(|rhs| lhs_burn.clone().matmul(rhs))
            .collect::<Result<Vec<WrappedTensor<N>>>>()?;

        // Now that we have the intermediate outputs as tensors from batched matmuls, we need to reshape them to their intermediate forms
        // and then permute if required
        let intermediate_shapes = self.mapping.intermediate_shapes(shapes)?;

        izip!(
            intermediate_results,
            intermediate_shapes,
            self.evaluation_info.output_permutations(),
            self.biases
                .iter()
                .zip(self.bias_unpadded_shapes.iter())
                .map(|(b, shape)| if let (Some(bias), Some(s)) = (b, shape) {
                    Some(bias.reduce_to_shape(s))
                } else {
                    None
                })
        )
        .map(
            |(intermediate, intermediate_shape, output_permutation, bias)| {
                // Reshape the burn tensor to the target rank
                let reshaped = intermediate.reshape(BShape::from(intermediate_shape.as_slice()))?;

                // Apply the output permutation if provided
                let permuted = if let Some(perm) = output_permutation {
                    reshaped.permute(&perm.0.iter().map(|d| *d as isize).collect::<Vec<_>>())?
                } else {
                    reshaped
                };
                // Add the bias if provided
                let with_bias = if let Some(bias_tensor) = bias {
                    let wrapped_bias = WrappedTensor::try_from(&bias_tensor)?;
                    permuted.add(wrapped_bias)?
                } else {
                    permuted
                };

                if self.padded {
                    let BShape { dims } = with_bias.shape();
                    let data: Vec<N> = with_bias.to_data().into_vec().map_err(|e| {
                        anyhow!(
                            "Could not compute EinSum, failure converting output data to vec: {e:?}"
                        )
                    })?;
                    let tmp_shape = Shape::new(dims);
                    let with_bias = Tensor::new(tmp_shape, data);
                    WrappedTensor::try_from(&with_bias.pad_next_power_of_two())
                } else {
                    Ok(with_bias)
                }
            },
        )
        .collect()
    }
}

/// Function that looks at the output axes, and finds their positions in the input axes if it exists.
fn find_output_dims_in_inputs_option(
    output_dims: &[char],
    input_dims: &[char],
) -> Vec<Option<usize>> {
    output_dims
        .iter()
        .map(|dim| input_dims.iter().position(|&d| d == *dim))
        .collect()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
/// Struct containing the [Permutations][crate::layers::concat_matmul::Permutation] to be applied to each input and output tensor to get them into the right order for the operation,
/// as well as the reshapes we need to apply to view the operation as one on 3D tensors, all einsums can be reduced to one 3D operation.
pub struct EvaluationInformation3D {
    /// The permutation to apply to the LHS of the input equation (if any).
    lhs_permutation: Option<Permutation>,
    /// The Reshape to apply to the LHS of the input equation after permuting.
    lhs_reshape: [usize; 3],
    /// The permutations to apply to the RHS of the input equation (if any).
    rhs_permutation: Vec<Option<Permutation>>,
    /// The Reshape to apply to the RHS of the input equation after permuting.
    rhs_reshape: Vec<[usize; 3]>,
    /// The permutations to apply to the outputs of the equation (if any).
    /// These are applied after the matmul and reshape back to the original output shape.
    /// There is one for each output tensor.
    output_permutation: Vec<Option<Permutation>>,
}

/// Temporary struct used in creating [`EvaluationInformation3D`] from an [`AxesMapping`].
/// Used to sort the dimensions of an input tensor into Stacked, Outer and Contracted axes.
/// The `usize` values represent the indices of these dimensions in the original tensor shape.
struct DimensionSorter {
    stacked: Vec<usize>,
    contracted: Vec<usize>,
    outer: Vec<usize>,
}

impl DimensionSorter {
    fn new() -> Self {
        Self {
            stacked: Vec::new(),
            contracted: Vec::new(),
            outer: Vec::new(),
        }
    }

    fn push_stacked(&mut self, dim: usize) {
        self.stacked.push(dim);
    }
    fn push_contracted(&mut self, dim: usize) {
        self.contracted.push(dim);
    }
    fn push_outer(&mut self, dim: usize) {
        self.outer.push(dim);
    }

    fn to_perm_and_reshape<const LHS: bool>(&self) -> Result<(Option<Permutation>, [usize; 3])> {
        if LHS {
            let reshape = [self.stacked.len(), self.outer.len(), self.contracted.len()];
            let mut new_order = vec![];
            new_order.extend(self.stacked.iter());
            new_order.extend(self.outer.iter());
            new_order.extend(self.contracted.iter());
            if new_order.iter().enumerate().all(|(i, &&p)| i == p) {
                Ok((None, reshape))
            } else {
                Ok((
                    Some(Permutation::new(new_order.into_iter().copied().collect())),
                    reshape,
                ))
            }
        } else {
            // We need to make sure the contraction dims are in the same order as the LHS contraction dims,
            // since self.contracted stores the indices of the contracted axes in the tensor shape we check that
            // i.e. self.contracted[i - 1] < self.contracted[i]
            let sorted_dims = self
                .contracted
                .iter()
                .copied()
                .sorted()
                .collect::<Vec<usize>>();
            ensure!(
                sorted_dims == self.contracted,
                "Contraction axes in RHS tensor are not in the same order as the LHS tensor"
            );

            let reshape = [self.stacked.len(), self.contracted.len(), self.outer.len()];

            let mut new_order = vec![];
            new_order.extend(self.stacked.iter());
            new_order.extend(self.contracted.iter());
            new_order.extend(self.outer.iter());
            if new_order.iter().enumerate().all(|(i, &&p)| i == p) {
                Ok((None, reshape))
            } else {
                Ok((
                    Some(Permutation::new(new_order.into_iter().copied().collect())),
                    reshape,
                ))
            }
        }
    }
}

/// Temporary struct used in creating [`EvaluationInformation3D`] from an [`AxesMapping`].
/// Used to sort the dimensions of an output tensor into the order they appear in the inputs.
struct OutputDimensionSorter {
    order: Vec<(usize, char)>,
}

impl OutputDimensionSorter {
    fn new() -> Self {
        Self { order: Vec::new() }
    }

    fn push(&mut self, dim: usize, repr: char) {
        self.order.push((dim, repr));
    }

    fn finalise(&mut self) {
        // We want to zip all the dims together with the chars, then sort by ascending order of dims
        self.order = self
            .order
            .iter()
            .sorted_by(|a, b| a.0.cmp(&b.0))
            .copied()
            .collect();
    }

    fn calculate_permutation(&mut self, intermediate_order: &[char]) -> Option<Permutation> {
        self.finalise();

        let actual_order = self.order.iter().map(|&(_, c)| c).collect::<Vec<char>>();
        if intermediate_order
            .iter()
            .zip(actual_order.iter())
            .all(|(&i, &p)| i == p)
        {
            None
        } else {
            let output_order = find_output_dims_in_inputs_option(&actual_order, intermediate_order)
                .into_iter()
                .map(|o| o.expect("Output dimension not found, should be impossible"))
                .collect::<Vec<usize>>();
            Some(Permutation::new(output_order))
        }
    }
}

impl EvaluationInformation3D {
    /// Construct the [`EvaluationInformation3D`] from the given [`AxesMapping`].
    ///
    /// Given an example equation of "A(isjk)@B(ilk)->C(isjl)", where A is the LHS, B is the RHS and C is the output, this method will return:
    /// - LHS permutation: None (no permutation needed)
    /// - RHS permutation: Some(Permutation([0, 2, 1])) that moves 'i' to the front and 'l' to the end, resulting in (ilk) -> (ikl)
    /// - Output permutation: None (no permutation needed)
    ///
    /// In general the "Stacked axes" are the axes that appear in both inputs and the output, the "Outer axes" are the axes that appear in only one input and the output,
    /// and the "Contraction axes" are the axes that appear in both inputs but not the output. The LHS permutation will arrange the axes in the order "Stacked axes, Outer axes, Contraction Axes",
    /// the RHS permutation will arrange the axes in the order "Stacked axes, Contraction Axes, Outer axes". This gives an intermediate result in the order "Stacked axes", "LHS outer axes", "RHS outer axes" and the output permutation will take this intermediate output and rearrange it to match the desired output.
    pub fn new(axes_mapping: &AxesMapping) -> Result<EvaluationInformation3D> {
        let mut input_sorters = (0..axes_mapping.input_count())
            .map(|_| DimensionSorter::new())
            .collect::<Vec<_>>();
        let mut output_sorters = (0..axes_mapping.output_count())
            .map(|_| OutputDimensionSorter::new())
            .collect::<Vec<_>>();
        let mut stack_axes = Vec::<char>::new();
        let mut outer_axes = vec![vec![]; axes_mapping.input_count()];

        axes_mapping.axes().for_each(|axis| {
            axis.inputs().enumerate().for_each(|(input_id, dimension)| {
                if let Dimension::Present(dim) = dimension {
                    match axis.axis_type {
                        AxisType::Stacked => {
                            input_sorters[input_id].push_stacked(*dim);
                            stack_axes.push(axis.repr)
                        }
                        AxisType::Outer => {
                            input_sorters[input_id].push_outer(*dim);
                            outer_axes[input_id].push(axis.repr);
                        }
                        AxisType::Contracted => {
                            input_sorters[input_id].push_contracted(*dim);
                        }
                    }
                }
            });

            axis.outputs
                .iter()
                .enumerate()
                .for_each(|(output_id, dimension)| {
                    if let Dimension::Present(dim) = dimension {
                        output_sorters[output_id].push(*dim, axis.repr);
                    }
                });
        });
        // Remove duplicates from stack_axes
        stack_axes.dedup();
        let mut perms_and_reshapes = input_sorters
            .into_iter()
            .enumerate()
            .map(|(i, sorter)| {
                if i == 0 {
                    sorter.to_perm_and_reshape::<true>()
                } else {
                    sorter.to_perm_and_reshape::<false>()
                }
            })
            .collect::<Result<Vec<_>>>()?;
        let (lhs_permutation, lhs_reshape) = perms_and_reshapes.remove(0);
        let (rhs_permutation, rhs_reshape): (Vec<_>, Vec<_>) =
            perms_and_reshapes.into_iter().unzip();

        let lhs_outer_axes = outer_axes.remove(0);
        let intermediate_orders = outer_axes
            .iter()
            .map(|o| {
                let mut order = vec![];
                order.extend(stack_axes.iter());
                order.extend(lhs_outer_axes.iter());
                order.extend(o.iter());
                order
            })
            .collect::<Vec<Vec<char>>>();
        let output_permutation = output_sorters
            .iter_mut()
            .zip(intermediate_orders.iter())
            .map(|(sorter, order)| sorter.calculate_permutation(order))
            .collect::<Vec<Option<Permutation>>>();

        Ok(EvaluationInformation3D {
            lhs_permutation,
            lhs_reshape,
            rhs_permutation,
            rhs_reshape,
            output_permutation,
        })
    }

    /// Get the permutation to apply to the LHS of the input equation (if any).
    pub fn lhs_permutation(&self) -> Option<&Permutation> {
        self.lhs_permutation.as_ref()
    }

    /// Get the Reshape to apply to the LHS of the input equation after permuting.
    pub fn lhs_reshape(&self) -> [usize; 3] {
        self.lhs_reshape
    }

    /// Get the permutations to apply to the RHS of the input equation (if any).
    pub fn rhs_permutation(&self) -> &[Option<Permutation>] {
        &self.rhs_permutation
    }

    /// Get the Reshape to apply to the RHS of the input equation after permuting.
    pub fn rhs_reshape(&self) -> &[[usize; 3]] {
        &self.rhs_reshape
    }

    /// Get the permutations to apply to the outputs of the equation (if any).
    pub fn output_permutations(&self) -> impl Iterator<Item = Option<&Permutation>> {
        self.output_permutation.iter().map(|p| p.as_ref())
    }

    /// Get the input permutations as an iterator
    pub fn input_permutations(&self) -> impl Iterator<Item = Option<&Permutation>> {
        std::iter::once(self.lhs_permutation.as_ref())
            .chain(self.rhs_permutation.iter().map(|p| p.as_ref()))
    }

    /// Get the input reshapes as an iterator
    pub fn input_reshapes(&self) -> impl Iterator<Item = [usize; 3]> + '_ {
        std::iter::once(self.lhs_reshape).chain(self.rhs_reshape.iter().copied())
    }
}

#[cfg(test)]
mod tests {
    use ark_std::rand::Rng;

    use super::*;
    use crate::{Element, rng_from_env_or_random, tensor::IntoBTensor};
    use burn::tensor::Tensor as BurnTensor;

    #[test]
    fn test_simple_matmul() {
        test_simple_matmul_helper::<f32>();
        test_simple_matmul_helper::<Element>();
    }

    fn test_simple_matmul_helper<N>()
    where
        N: TensorTypeParam,
    {
        let einsum: EinSum<N> =
            EinSum::new("A(ab)@B(bc)->C(ac)".to_string(), vec![None], vec![None])
                .expect("Failed to create EinSum layer");

        let mut rng = rng_from_env_or_random();

        for _ in 0..10 {
            let dim1: usize = rng.gen_range(1..15);
            let dim2: usize = rng.gen_range(1..15);
            let dim3: usize = rng.gen_range(1..15);
            let a_shape = Shape::new(vec![dim1, dim2]);
            let b_shape = Shape::new(vec![dim2, dim3]);
            let c_shape = Shape::new(vec![dim1, dim3]);

            let a = Tensor::<N>::random(&a_shape);
            let b = Tensor::<N>::random(&b_shape);
            let output = einsum
                .evaluate_internal(
                    &[&a.as_wrapped(), &b.as_wrapped()],
                    &[a_shape.clone(), b_shape.clone()],
                )
                .expect("Failed to evaluate EinSum layer");

            let a_burn = a.to_btensor::<2>();
            let b_burn = b.to_btensor::<2>();
            let expected_burn = a_burn.matmul(b_burn);

            let burn_data: Vec<N> = expected_burn
                .into_data()
                .into_vec()
                .expect("Failed to convert expected output to vec");
            let expected = Tensor::new(c_shape.clone(), burn_data);
            let output = Tensor::new(c_shape, output[0].clone().get_data());
            assert_eq!(
                output.clone().get_data_into(),
                expected.clone().get_data_into(),
                "Failed for shapes A: {a_shape:?}, B: {b_shape:?}, Calculated: {output}, Expected: {expected}",
            );
        }
    }

    #[test]
    fn test_simple_matmul_with_bias() {
        test_simple_matmul_with_bias_helper::<f32>();
        test_simple_matmul_with_bias_helper::<Element>();
    }

    fn test_simple_matmul_with_bias_helper<N>()
    where
        N: TensorTypeParam,
    {
        let mut rng = rng_from_env_or_random();

        for _ in 0..10 {
            let dim1: usize = rng.gen_range(1..15);
            let dim2: usize = rng.gen_range(1..15);
            let dim3: usize = rng.gen_range(1..15);
            let a_shape = Shape::new(vec![dim1, dim2]);
            let b_shape = Shape::new(vec![dim2, dim3]);
            let c_shape = Shape::new(vec![dim1, dim3]);
            let bias_shape = Shape::new(vec![dim3]);
            let bias = Tensor::<N>::random(&bias_shape);
            let keyed_bias = KeyedTensor::new("BIAS".to_string(), bias.clone());
            let einsum: EinSum<N> = EinSum::new(
                "A(ab)@B(bc)->C(ac)+BIAS(c)".to_string(),
                vec![None],
                vec![Some(keyed_bias)],
            )
            .expect("Failed to create EinSum layer");

            let a = Tensor::<N>::random(&a_shape);
            let b = Tensor::<N>::random(&b_shape);
            let output = einsum
                .evaluate_internal(
                    &[&a.as_wrapped(), &b.as_wrapped()],
                    &[a_shape.clone(), b_shape.clone()],
                )
                .expect("Failed to evaluate EinSum layer");

            let a_burn = a.to_btensor::<2>();
            let b_burn = b.to_btensor::<2>();
            let expected_burn_matmul = a_burn.matmul(b_burn);

            let burn_bias = bias.to_btensor::<1>().unsqueeze::<2>();
            let expected_burn = expected_burn_matmul.add(burn_bias);

            let burn_data: Vec<N> = expected_burn
                .into_data()
                .into_vec()
                .expect("Failed to convert expected output to vec");
            let expected = Tensor::new(c_shape.clone(), burn_data);
            let output = Tensor::new(c_shape, output[0].clone().get_data());
            assert_eq!(
                output.clone().get_data_into(),
                expected.clone().get_data_into(),
                "Failed for shapes A: {a_shape:?}, B: {b_shape:?}, Calculated: {output}, Expected: {expected}",
            );
        }
    }

    #[test]
    fn test_simple_batched_matmul() {
        test_simple_batched_matmul_helper::<f32>();
        test_simple_batched_matmul_helper::<Element>();
    }

    fn test_simple_batched_matmul_helper<N>()
    where
        N: TensorTypeParam,
    {
        let einsum: EinSum<N> =
            EinSum::new("A(xab)@B(xbc)->C(xac)".to_string(), vec![None], vec![None])
                .expect("Failed to create EinSum layer");

        let mut rng = rng_from_env_or_random();

        for _ in 0..10 {
            let batch: usize = rng.gen_range(1..5);
            let dim1: usize = rng.gen_range(1..15);
            let dim2: usize = rng.gen_range(1..15);
            let dim3: usize = rng.gen_range(1..15);
            let a_shape = Shape::new(vec![batch, dim1, dim2]);
            let b_shape = Shape::new(vec![batch, dim2, dim3]);
            let c_shape = Shape::new(vec![batch, dim1, dim3]);

            let a = Tensor::<N>::random(&a_shape);
            let b = Tensor::<N>::random(&b_shape);
            let output = einsum
                .evaluate_internal(
                    &[&a.as_wrapped(), &b.as_wrapped()],
                    &[a_shape.clone(), b_shape.clone()],
                )
                .expect("Failed to evaluate EinSum layer");

            let a_burn = a.to_btensor::<3>();
            let b_burn = b.to_btensor::<3>();
            let expected_burn_vec = a_burn
                .iter_dim(0)
                .zip(b_burn.iter_dim(0))
                .map(|(a_batch, b_batch)| a_batch.matmul(b_batch))
                .collect::<Vec<_>>();
            let expected_burn = BurnTensor::cat(expected_burn_vec, 0);

            let burn_data: Vec<N> = expected_burn
                .into_data()
                .into_vec()
                .expect("Failed to convert expected output to vec");
            let expected = Tensor::new(c_shape.clone(), burn_data);
            let output = Tensor::new(c_shape, output[0].clone().get_data());
            assert_eq!(
                output.clone().get_data_into(),
                expected.clone().get_data_into(),
                "Failed for shapes A: {a_shape:?}, B: {b_shape:?}, Calculated: {output}, Expected: {expected}",
            );
        }
    }

    #[test]
    fn test_multi_output_batched_matmul() {
        test_multi_output_batched_matmul_helper::<f32>();
        test_multi_output_batched_matmul_helper::<Element>();
    }

    fn test_multi_output_batched_matmul_helper<N>()
    where
        N: TensorTypeParam,
    {
        let einsum: EinSum<N> = EinSum::new(
            "A(xab)@B(xbc):C(xbe)->D(xac):E(xae)".to_string(),
            vec![None, None],
            vec![None, None],
        )
        .expect("Failed to create EinSum layer");
        let mut rng = rng_from_env_or_random();

        for _ in 0..10 {
            let batch: usize = rng.gen_range(1..5);
            let dim1: usize = rng.gen_range(1..15);
            let dim2: usize = rng.gen_range(1..15);
            let dim3: usize = rng.gen_range(1..15);
            let dim4: usize = rng.gen_range(1..15);
            let a_shape = Shape::new(vec![batch, dim1, dim2]);
            let b_shape = Shape::new(vec![batch, dim2, dim3]);
            let c_shape = Shape::new(vec![batch, dim2, dim4]);
            let d_shape = Shape::new(vec![batch, dim1, dim3]);
            let e_shape = Shape::new(vec![batch, dim1, dim4]);

            let a = Tensor::<N>::random(&a_shape);
            let b = Tensor::<N>::random(&b_shape);
            let c = Tensor::<N>::random(&c_shape);
            let outputs = einsum
                .evaluate_internal(
                    &[&a.as_wrapped(), &b.as_wrapped(), &c.as_wrapped()],
                    &[a_shape.clone(), b_shape.clone(), c_shape.clone()],
                )
                .expect("Failed to evaluate EinSum layer");

            let a_burn = a.to_btensor::<3>();
            let b_burn = b.to_btensor::<3>();
            let c_burn = c.to_btensor::<3>();

            // The first output is D(xac) which is a standard batched matmul of A and B
            let expected_d_burn_vec = a_burn
                .clone()
                .iter_dim(0)
                .zip(b_burn.iter_dim(0))
                .map(|(a_batch, b_batch)| a_batch.matmul(b_batch))
                .collect::<Vec<_>>();
            let expected_d_burn = BurnTensor::cat(expected_d_burn_vec, 0);

            // The second output is E(xae) which is a batched matmul of A and C
            let expected_e_burn_vec = a_burn
                .iter_dim(0)
                .zip(c_burn.iter_dim(0))
                .map(|(a_batch, c_batch)| a_batch.matmul(c_batch))
                .collect::<Vec<_>>();
            let expected_e_burn = BurnTensor::cat(expected_e_burn_vec, 0);

            for (i, (output, burn_output, shape)) in izip!(
                outputs,
                [expected_d_burn, expected_e_burn],
                [d_shape, e_shape]
            )
            .enumerate()
            {
                let burn_data: Vec<N> = burn_output
                    .into_data()
                    .into_vec()
                    .expect("Failed to convert expected output to vec");

                let expected = Tensor::new(shape.clone(), burn_data);
                let output = Tensor::new(shape, output.clone().get_data());
                assert_eq!(
                    output.clone().get_data_into(),
                    expected.clone().get_data_into(),
                    "Failed for output {i} shapes A: {a_shape:?}, B: {b_shape:?}, C: {c_shape:?}, Calculated: {output}, Expected: {expected}"
                );
            }
        }
    }

    #[test]
    fn test_grouped_qkv() {
        test_grouped_qkv_helper::<f32>();
        test_grouped_qkv_helper::<Element>();
    }

    fn test_grouped_qkv_helper<N>()
    where
        N: TensorTypeParam,
    {
        let mut rng = rng_from_env_or_random();

        for _ in 0..10 {
            let heads: usize = rng.gen_range(1..5);
            let seq_len: usize = rng.gen_range(1..15);
            let embedding_dim = rng.gen_range(1..15);
            let head_dim: usize = rng.gen_range(1..15);

            let x_shape = Shape::new(vec![seq_len, embedding_dim]);
            let wq_shape = Shape::new(vec![embedding_dim, heads, head_dim]);
            let wk_shape = Shape::new(vec![embedding_dim, head_dim]);
            let wv_shape = Shape::new(vec![embedding_dim, head_dim]);
            let q_shape = Shape::new(vec![heads, seq_len, head_dim]);

            let k_shape = Shape::new(vec![seq_len, head_dim]);
            let v_shape = Shape::new(vec![seq_len, head_dim]);

            let x = Tensor::<N>::random(&x_shape);
            let wq = Tensor::<N>::random(&wq_shape);
            let wk = Tensor::<N>::random(&wk_shape);
            let wv = Tensor::<N>::random(&wv_shape);

            let keyed_wq = KeyedTensor::new("WQ".to_string(), wq.clone());
            let keyed_wk = KeyedTensor::new("WK".to_string(), wk.clone());
            let keyed_wv = KeyedTensor::new("WV".to_string(), wv.clone());

            let einsum: EinSum<N> = EinSum::new(
                "X(se)@WQ(ehd):WK(ed):WV(ed)->Q(hsd):K(sd):V(sd)".to_string(),
                vec![Some(keyed_wq), Some(keyed_wk), Some(keyed_wv)],
                vec![None, None, None],
            )
            .expect("Failed to create EinSum layer");

            let output = einsum
                .evaluate_internal(&[&x.as_wrapped()], std::slice::from_ref(&x_shape))
                .expect("Failed to evaluate EinSum layer");

            // Manually compute the expected output
            let x_burn = x.to_btensor::<2>();
            let wq_burn = wq.to_btensor::<3>();
            let wk_burn = wk.to_btensor::<2>();
            let wv_burn = wv.to_btensor::<2>();

            // First compute X @ WQ to get Q
            let q_burn_vec = wq_burn
                .iter_dim(1)
                .map(|wq_head| {
                    x_burn
                        .clone()
                        .matmul(wq_head.reshape([embedding_dim, head_dim]))
                })
                .collect::<Vec<_>>();
            let q_burn = BurnTensor::cat(q_burn_vec, 0);

            // Then compute X @ WK to get K
            let k_burn = x_burn.clone().matmul(wk_burn);

            // Then compute X @ WV to get V
            let v_burn = x_burn.matmul(wv_burn);

            for (output, burn_output, shape, name) in izip!(
                output,
                [q_burn, k_burn, v_burn],
                [q_shape, k_shape, v_shape],
                ["Q", "K", "V"]
            ) {
                let burn_data: Vec<N> = burn_output
                    .into_data()
                    .into_vec()
                    .expect("Failed to convert expected output to vec");

                let expected = Tensor::new(shape.clone(), burn_data);
                let output = Tensor::new(shape, output.clone().get_data());
                assert_eq!(
                    output.clone().get_data_into(),
                    expected.clone().get_data_into(),
                    "Failed for output {name} shapes X: {x_shape:?}, WQ: {wq_shape:?}, WK: {wk_shape:?}, WV: {wv_shape:?}, Calculated: {output}, Expected: {expected}"
                );
            }
        }
    }

    #[test]
    fn test_grouped_qk_transpose() {
        test_grouped_qk_transpose_helper::<f32>();
        test_grouped_qk_transpose_helper::<Element>();
    }

    fn test_grouped_qk_transpose_helper<N>()
    where
        N: TensorTypeParam,
    {
        let einsum: EinSum<N> = EinSum::new(
            // Here the Q uses "q" for seq_len while K uses "s", this is so that both single token inference and full sequence can be handled
            // in a single einsum layer. The "g" dimension is the grouping of the query heads.
            "Q(ghqd)@K(hsd)->QKT(ghqs)".to_string(),
            vec![None],
            vec![None],
        )
        .expect("Failed to create EinSum layer");
        let mut rng = rng_from_env_or_random();

        for _ in 0..10 {
            let heads: usize = rng.gen_range(1..5);
            let group_size: usize = rng.gen_range(1..4);
            let q_len = 1usize;
            let seq_len: usize = rng.gen_range(1..15);
            let head_dim: usize = rng.gen_range(1..15);

            // First we test the full sequence length case where q_len == seq_len
            let q_full_shape = Shape::new(vec![group_size, heads, seq_len, head_dim]);
            let k_shape = Shape::new(vec![heads, seq_len, head_dim]);
            let qkt_full_shape = Shape::new(vec![group_size, heads, seq_len, seq_len]);

            let q_full = Tensor::<N>::random(&q_full_shape);
            let k = Tensor::<N>::random(&k_shape);
            let output_full = einsum
                .evaluate_internal(
                    &[&q_full.as_wrapped(), &k.as_wrapped()],
                    &[q_full_shape.clone(), k_shape.clone()],
                )
                .expect("Failed to evaluate EinSum layer");

            let k_burn = k.to_btensor::<3>();

            let calc_expected_output = |q: Tensor<N>| -> Vec<N> {
                let q_burn = q.to_btensor::<4>();
                let expected_burn_vec = q_burn
                    .iter_dim(0)
                    .map(|q_head| {
                        let q_shaped = q_head.squeeze::<3>(0);
                        let intermediate_vec = q_shaped
                            .iter_dim(0)
                            .zip(k_burn.clone().iter_dim(0))
                            .map(|(q, k)| q.matmul(k.transpose()))
                            .collect::<Vec<_>>();
                        BurnTensor::cat(intermediate_vec, 0).unsqueeze::<4>()
                    })
                    .collect::<Vec<_>>();
                let expected_burn = BurnTensor::cat(expected_burn_vec, 0);

                expected_burn
                    .into_data()
                    .into_vec()
                    .expect("Failed to convert expected output to vec")
            };

            let expected_full_data = calc_expected_output(q_full);
            let expected_full = Tensor::new(qkt_full_shape.clone(), expected_full_data);
            let output_full = Tensor::new(qkt_full_shape, output_full[0].clone().get_data());
            assert_eq!(
                output_full.clone().get_data_into(),
                expected_full.clone().get_data_into(),
                "Failed for full sequence shapes Q: {q_full_shape:?}, K: {k_shape:?}, Calculated: {output_full}, Expected: {expected_full}",
            );
            // Now we test the single token case where q_len == 1
            let q_single_shape = Shape::new(vec![group_size, heads, q_len, head_dim]);
            let qkt_single_shape = Shape::new(vec![group_size, heads, q_len, seq_len]);
            let q_single = Tensor::<N>::random(&q_single_shape);
            let output_single = einsum
                .evaluate_internal(
                    &[&q_single.as_wrapped(), &k.as_wrapped()],
                    &[q_single_shape.clone(), k_shape.clone()],
                )
                .expect("Failed to evaluate EinSum layer");

            let expected_single_data = calc_expected_output(q_single);
            let expected_single = Tensor::new(qkt_single_shape.clone(), expected_single_data);
            let output_single = Tensor::new(qkt_single_shape, output_single[0].clone().get_data());
            assert_eq!(
                output_single.clone().get_data_into(),
                expected_single.clone().get_data_into(),
                "Failed for single token shapes Q: {q_single_shape:?}, K: {k_shape:?}, Calculated: {output_single}, Expected: {expected_single}",
            );
        }
    }
}
