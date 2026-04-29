//! Module that provides methods for generating [`EinsSumContext`] from an [`EinSum`] layer.
use super::*;
use crate::{Element, Tensor, iop::context::ContextAux};
use anyhow::Result;

impl EinSum<Element> {
    /// Create an [`EinSumContext`] from the current [`EinSum`] layer.
    pub fn to_context<F: PrimeField>(
        &self,
        mut aux: ContextAux,
    ) -> Result<(EinSumContext<F>, ContextAux)> {
        // Update the output shapes
        let mut inputs_shapes_iter = aux.last_output_shape.iter();
        let lhs_shape = inputs_shapes_iter
            .next()
            .ok_or(anyhow!("Missing LHS input shape"))?
            .clone();

        let shapes = std::iter::once(Ok(lhs_shape))
            .chain(self.constant_unpadded_shapes.iter().map(|const_shape| {
                if let Some(s) = const_shape {
                    Ok(s.next_power_of_two())
                } else {
                    inputs_shapes_iter
                        .next()
                        .ok_or(anyhow!("Missing unpadded input shape"))
                        .cloned()
                }
            }))
            .collect::<Result<Vec<Shape>>>()?;
        aux.last_output_shape = self.mapping.output_shapes(&shapes)?;

        // Return the constant and bias tensors as padded polynomials
        // We only include them if at least one is present
        let constant_poly_check = self
            .constant_tensors
            .iter()
            .chain(self.biases.iter())
            .any(|x| x.is_some());
        if constant_poly_check {
            aux.model_polys = Some(
                self.constant_tensors
                    .iter()
                    .chain(self.biases.iter())
                    .filter_map(|tensor_opt| {
                        tensor_opt.as_ref().map(|tensor| {
                            (
                                CommitmentId::from(tensor.storage_key()),
                                Tensor::try_from(tensor)
                                    .unwrap()
                                    .pad_next_power_of_two()
                                    .into_data(),
                            )
                        })
                    })
                    .collect(),
            );
        } else {
            aux.model_polys = None;
        }
        // Create the sumcheck expression
        let input_aggregation_expression = self.build_aggregation_expression::<F>();

        let constant_keys = self
            .constant_tensors
            .iter()
            .map(|tensor_opt| {
                tensor_opt
                    .as_ref()
                    .map(|tensor| CommitmentId::from(tensor.storage_key()))
            })
            .collect();
        let bias_keys = self
            .biases
            .iter()
            .map(|tensor_opt| {
                tensor_opt
                    .as_ref()
                    .map(|tensor| CommitmentId::from(tensor.storage_key()))
            })
            .collect();

        // determine if this EinSum layer is splittable. For now, we only support splitting iff:
        // - The first dimension of all left inputs is an outer axis
        // - All the RHS are constant tensors
        // - The bias, if any, does not involve the outer axis being splitted

        // first, find the axis corresponding to the first dimension of the LHS
        let first_dim_axis = self.mapping.axes().find(|axis| {
            if let Dimension::Present(pos) = axis.lhs_input
                && pos == 0
            {
                true
            } else {
                false
            }
        });

        ensure!(
            first_dim_axis.is_some(),
            "First dimension of LHS input is not present in the axis mapping?"
        );

        let first_dim_axis = first_dim_axis.unwrap();
        let is_splittable = {
            // check that this axis is an outer axis
            if let AxisType::Outer = first_dim_axis.axis_type {
                // check that this axis does not appear in the bias unpadded shapes
                let bias_check = first_dim_axis
                    .biases
                    .iter()
                    .all(|bias_axis| matches!(bias_axis, Dimension::Absent));
                // check that all RHS inputs are constant tensors
                let rhs_constant_check = self.constant_tensors.iter().all(|t| t.is_some());
                bias_check && rhs_constant_check
            } else {
                false
            }
        };

        Ok((
            EinSumContext {
                equation: self.equation.clone(),
                mapping: self.mapping.clone(),
                constant_keys,
                constant_unpadded_shapes: self.constant_unpadded_shapes.clone(),
                bias_keys,
                bias_unpadded_shapes: self.bias_unpadded_shapes.clone(),

                input_aggregation_expression,
                is_splittable,
            },
            aux,
        ))
    }

    fn build_aggregation_expression<F: PrimeField>(&self) -> Option<Expression<F>> {
        let total_inputs = self.mapping.input_count();

        if total_inputs > 2 {
            // The input poly will be the the first expression, then eq_polys for the rest of the inputs
            let input_expr = Expression::WitIn(0);
            let expr = (0..total_inputs - 1).fold(Expression::ZERO, |acc, i| {
                acc + input_expr.clone()
                    * Expression::WitIn((i + 1) as u16)
                    * Expression::Challenge(0, i, F::ONE, F::ZERO)
            });
            Some(expr)
        } else {
            None
        }
    }
}

impl<F: PrimeField> EinSumContext<F> {
    /// Build the einsum expression for the sumcheck from the stacking coefficients
    /// We do this on the fly because the stacking coefficients depend on the input shapes
    /// which are only known at proving/verification time.
    pub(crate) fn build_einsum_expression(&self, stacking_coeffs: &[&[F]]) -> Expression<F> {
        let total_inputs = self.mapping.input_count();
        let rhs_inputs = total_inputs - 1;
        // the outer length of `stacking_coeffs` should be the number of operations that are being batched in this einsum
        // the inner length of `stacking_coeffs` should be the size of the stacking axis
        assert_eq!(
            stacking_coeffs.len(),
            total_inputs - 1,
            "Number of stacking coefficients ({}) does not match number of RHS inputs ({rhs_inputs})",
            stacking_coeffs.len(),
        );
        stacking_coeffs
            .iter()
            .enumerate()
            .fold(Expression::ZERO, |acc, (i, coeffs)| {
                // coeffs.len() will be the same each time because they all correspond to the same stacking axis
                // and so have the same size
                let stack_dim_size = coeffs.len();
                let offset = i * stack_dim_size;
                let rhs_offset = rhs_inputs * stack_dim_size + offset;
                // We need to make the initial term otherwise it complains about a zero product
                let initial_expr = Expression::Constant(coeffs[0])
                    * Expression::WitIn(offset as u16)
                    * Expression::WitIn(rhs_offset as u16);
                acc + Expression::Challenge(0, i, F::ONE, F::ZERO)
                    * coeffs
                        .iter()
                        .enumerate()
                        .skip(1)
                        .fold(initial_expr, |inner_acc, (j, &c)| {
                            Expression::Constant(c)
                                * Expression::WitIn((j + offset) as u16)
                                * Expression::WitIn((j + rhs_offset) as u16)
                                + inner_acc
                        })
            })
    }
}
