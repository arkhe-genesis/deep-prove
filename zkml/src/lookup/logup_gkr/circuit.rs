//! File containing code for generating the LogUp GKR circuit

use ark_ff::PrimeField;
use ark_std::Zero;
use dp_crypto::{
    Expression,
    poly::{dense::DensePolynomial, slice::SmartSlice},
    util::ceil_log2,
};

use itertools::izip;

use super::structs::Fraction;

#[derive(Debug, Clone)]
/// The different variants of layer that can be found in the LogUp GKR protocol.
pub enum LogUpLayer<F: PrimeField> {
    /// This variant is used for every layer apart from the first in the LogUp protocol.
    Generic {
        numerator: Vec<F>,
        denominator: Vec<F>,
    },
    /// This is the first layer of the GKR protocol when proving a fractional sumcheck for a table.
    /// The numerator is the multiplicity polynomial and the denominator is the merged table polynomial.
    InitialTable {
        numerator: Vec<F>,
        constant_challenge: F,
        column_separation_challenge: F,
        denominator_columns: Vec<Vec<F>>,
    },
    /// This is the first layer of the GKR protocol when proving a fractional sumcheck for a set of lookups.
    /// The numerators will all be `-1` on this layer so we only store the denominator which is the merged lookups.
    InitialLookup {
        constant_challenge: F,
        column_separation_challenge: F,
        denominator_columns: Vec<Vec<F>>,
    },
}

#[derive(Debug, Clone)]
pub struct MLEsAndExpressions<'a, F: PrimeField> {
    pub(crate) mles: Vec<DensePolynomial<'a, F>>,
    pub(crate) sumcheck_expression: Expression<F>,
    pub(crate) claim_expression: Expression<F>,
}

impl<F: PrimeField> MLEsAndExpressions<'_, F> {
    pub fn mles(&self) -> &Vec<DensePolynomial<'_, F>> {
        &self.mles
    }

    pub fn sumcheck_expression(&self) -> &Expression<F> {
        &self.sumcheck_expression
    }

    pub fn claim_expression(&self) -> &Expression<F> {
        &self.claim_expression
    }

    pub fn num_witnesses(&self) -> u16 {
        self.mles.len() as u16
    }
}

impl<F: PrimeField> LogUpLayer<F> {
    /// Returns the number of variables that the [`MultilinearExtension`]s at this layer have
    pub fn num_vars(&self) -> usize {
        // We right shift the denominator length by 1 because at each level of the GKR circuit the polynomials we run the sumcheck over have half the length
        match self {
            LogUpLayer::Generic { denominator, .. } => ceil_log2(denominator.len() >> 1),
            LogUpLayer::InitialTable {
                denominator_columns,
                ..
            }
            | LogUpLayer::InitialLookup {
                denominator_columns,
                ..
            } => ceil_log2(denominator_columns[0].len() >> 1),
        }
    }

    /// Function used to check if this is the final layer
    pub fn is_final_layer(&self) -> bool {
        self.num_vars().is_zero()
    }

    /// Produces the next layer of the GKR circuit (if there is meant to be one)
    pub fn next_layer(&self) -> Option<LogUpLayer<F>> {
        // First we check to see if we are at the final layer, if so return None
        if self.is_final_layer() {
            return None;
        }

        let half_layer_size = 1 << self.num_vars();

        match self {
            LogUpLayer::Generic {
                numerator,
                denominator,
            } => {
                // Split the numerator and denominator at the halfway point and sum the fractions
                let (num1, num2) = numerator.split_at(half_layer_size);
                let (denom1, denom2) = denominator.split_at(half_layer_size);

                let (next_numerator, next_denominator): (Vec<F>, Vec<F>) =
                    izip!(num1, denom1, num2, denom2)
                        .map(|(n1, d1, n2, d2)| {
                            (Fraction::<F>::new(*n1, *d1) + Fraction::<F>::new(*n2, *d2)).as_tuple()
                        })
                        .unzip();

                Some(LogUpLayer::Generic {
                    numerator: next_numerator,
                    denominator: next_denominator,
                })
            }
            LogUpLayer::InitialTable {
                numerator,
                constant_challenge,
                column_separation_challenge,
                denominator_columns,
            } => {
                let (num1, num2) = numerator.split_at(half_layer_size);
                let (next_numerator, next_denominator): (Vec<F>, Vec<F>) = num1
                    .iter()
                    .zip(num2.iter())
                    .enumerate()
                    .map(|(i, (&n1, &n2))| {
                        let (denom1, denom2, _) = denominator_columns
                            .iter()
                            .map(|col| (col[i], col[i + half_layer_size]))
                            .fold(
                                (*constant_challenge, *constant_challenge, F::ONE),
                                |(acc1, acc2, chal_acc), (val1, val2)| {
                                    (
                                        acc1 + chal_acc * val1,
                                        acc2 + chal_acc * val2,
                                        chal_acc * *column_separation_challenge,
                                    )
                                },
                            );

                        (Fraction::<F>::new(n1, denom1) + Fraction::<F>::new(n2, denom2)).as_tuple()
                    })
                    .unzip();
                Some(LogUpLayer::Generic {
                    numerator: next_numerator,
                    denominator: next_denominator,
                })
            }
            LogUpLayer::InitialLookup {
                constant_challenge,
                column_separation_challenge,
                denominator_columns,
            } => {
                // In this case we only need to split the denominator polynomial in half as the numerators are all -1
                let (next_numerator, next_denominator): (Vec<F>, Vec<F>) = (0..half_layer_size)
                    .map(|i| {
                        let (d1, d2, _) = denominator_columns
                            .iter()
                            .map(|col| (col[i], col[i + half_layer_size]))
                            .fold(
                                (*constant_challenge, *constant_challenge, F::ONE),
                                |(acc1, acc2, chal_acc), (val1, val2)| {
                                    (
                                        acc1 + chal_acc * val1,
                                        acc2 + chal_acc * val2,
                                        chal_acc * *column_separation_challenge,
                                    )
                                },
                            );

                        (Fraction::<F>::new(-F::ONE, d1) + Fraction::<F>::new(-F::ONE, d2))
                            .as_tuple()
                    })
                    .unzip();

                Some(LogUpLayer::Generic {
                    numerator: next_numerator,
                    denominator: next_denominator,
                })
            }
        }
    }

    /// Gets the output values in order (`numerator`, `denominator`) if its the output layer, returns `None` otherwise.
    pub fn output_values(&self) -> Option<(F, F)> {
        match (self, self.is_final_layer()) {
            (
                LogUpLayer::Generic {
                    numerator,
                    denominator,
                },
                true,
            ) => Some((
                numerator[0] * denominator[1] + numerator[1] * denominator[0],
                denominator[0] * denominator[1],
            )),
            _ => None,
        }
    }

    /// Returns all evals at this layer in order numerators , denominators
    pub fn flat_evals(&self) -> Vec<F> {
        match self {
            LogUpLayer::Generic {
                numerator,
                denominator,
            } => [numerator.as_slice(), denominator.as_slice()].concat(),
            LogUpLayer::InitialLookup {
                denominator_columns,
                ..
            } => denominator_columns
                .iter()
                .flat_map(|col| col.iter().copied())
                .collect(),
            LogUpLayer::InitialTable {
                numerator,
                denominator_columns,
                ..
            } => [
                numerator.as_slice(),
                denominator_columns.concat().as_slice(),
            ]
            .concat()
            .into_iter()
            .collect(),
        }
    }

    pub fn layer_proving_info(
        &self,
        layer_count: usize,
        initial_witness_id: u16,
    ) -> MLEsAndExpressions<'_, F> {
        // Get the number of variables for this layer
        let num_vars = self.num_vars();
        let half_layer_size = 1 << num_vars;

        // build the claim expression which is used to link to the previous layer to this one, common to all layer types
        let claim_num_low_id = layer_count as u16 * 4;
        let claim_num_high_id = claim_num_low_id + 1;
        let claim_denom_low_id = claim_num_high_id + 1;
        let claim_denom_high_id = claim_denom_low_id + 1;
        let claim_expression = Expression::Challenge(0u16, layer_count, F::ONE, F::ZERO)
            * (Expression::Challenge(2, 1, F::ONE, F::ZERO)
                * (Expression::WitIn(claim_num_high_id) - Expression::WitIn(claim_num_low_id))
                + Expression::WitIn(claim_num_low_id)
                + Expression::Challenge(1u16, 1, F::ONE, F::ZERO)
                    * (Expression::Challenge(2, 1, F::ONE, F::ZERO)
                        * (Expression::WitIn(claim_denom_high_id)
                            - Expression::WitIn(claim_denom_low_id))
                        + Expression::WitIn(claim_denom_low_id)));

        match self {
            LogUpLayer::Generic {
                numerator,
                denominator,
            } => {
                let (num_low, num_high) = numerator.split_at(half_layer_size);
                let (denom_low, denom_high) = denominator.split_at(half_layer_size);
                let num_low_mle =
                    DensePolynomial::new_from_smart_slice(SmartSlice::Borrowed(num_low));
                let num_high_mle =
                    DensePolynomial::new_from_smart_slice(SmartSlice::Borrowed(num_high));
                let denom_low_mle =
                    DensePolynomial::new_from_smart_slice(SmartSlice::Borrowed(denom_low));
                let denom_high_mle =
                    DensePolynomial::new_from_smart_slice(SmartSlice::Borrowed(denom_high));

                let mles = vec![num_low_mle, num_high_mle, denom_low_mle, denom_high_mle];

                let num_low_id = initial_witness_id;
                let num_high_id = num_low_id + 1;
                let denom_low_id = num_high_id + 1;
                let denom_high_id = denom_low_id + 1;

                // build the sumcheck expression which is used to prove this layer links to the next one
                let sumcheck_expr = Expression::Challenge(0u16, layer_count, F::ONE, F::ZERO)
                    * (Expression::WitIn(num_low_id) * Expression::WitIn(denom_high_id)
                        + Expression::WitIn(num_high_id) * Expression::WitIn(denom_low_id)
                        + Expression::Challenge(1u16, 1, F::ONE, F::ZERO)
                            * Expression::WitIn(denom_low_id)
                            * Expression::WitIn(denom_high_id));

                MLEsAndExpressions {
                    mles,
                    sumcheck_expression: sumcheck_expr,
                    claim_expression,
                }
            }
            LogUpLayer::InitialTable {
                numerator,
                constant_challenge,
                column_separation_challenge,
                denominator_columns,
            } => {
                let (num_low, num_high) = numerator.split_at(half_layer_size);
                let num_low_mle =
                    DensePolynomial::new_from_smart_slice(SmartSlice::Borrowed(num_low));
                let num_high_mle =
                    DensePolynomial::new_from_smart_slice(SmartSlice::Borrowed(num_high));

                let (denom_low_mles, denom_high_mles) = denominator_columns.iter().fold(
                    (vec![], vec![]),
                    |(mut low, mut high), column| {
                        let (denom_low, denom_high) = column.split_at(half_layer_size);
                        low.push(DensePolynomial::new_from_smart_slice(SmartSlice::Borrowed(
                            denom_low,
                        )));
                        high.push(DensePolynomial::new_from_smart_slice(SmartSlice::Borrowed(
                            denom_high,
                        )));
                        (low, high)
                    },
                );

                let num_low_id = initial_witness_id;
                let num_high_id = num_low_id + 1;
                // Now we make the denominator expressions
                let column_count = denominator_columns.len() as u16;
                let constant_expr = Expression::Constant(*constant_challenge);
                let (denom_low_expr, denom_high_expr, _) = (0..denominator_columns.len()).fold(
                    (constant_expr.clone(), constant_expr.clone(), F::ONE),
                    |(acc_low, acc_high, chal_acc), i| {
                        let chal_expr = Expression::Constant(chal_acc);
                        (
                            acc_low
                                + chal_expr.clone() * Expression::WitIn(num_high_id + 1 + i as u16),
                            acc_high
                                + chal_expr
                                    * Expression::WitIn(num_high_id + 1 + column_count + i as u16),
                            chal_acc * *column_separation_challenge,
                        )
                    },
                );

                // build the sumcheck expression which is used to prove this layer links to the next one
                let sumcheck_expr = Expression::Challenge(0u16, layer_count, F::ONE, F::ZERO)
                    * (Expression::WitIn(num_low_id) * denom_high_expr.clone()
                        + Expression::WitIn(num_high_id) * denom_low_expr.clone()
                        + Expression::Challenge(1u16, 1, F::ONE, F::ZERO)
                            * denom_low_expr.clone()
                            * denom_high_expr.clone());

                let mut mles = vec![num_low_mle, num_high_mle];
                mles.extend(denom_low_mles);
                mles.extend(denom_high_mles);
                MLEsAndExpressions {
                    mles,
                    sumcheck_expression: sumcheck_expr,
                    claim_expression,
                }
            }
            LogUpLayer::InitialLookup {
                constant_challenge,
                column_separation_challenge,
                denominator_columns,
            } => {
                let (denom_low_mles, denom_high_mles) = denominator_columns.iter().fold(
                    (vec![], vec![]),
                    |(mut low, mut high), column| {
                        let (denom_low, denom_high) = column.split_at(half_layer_size);
                        low.push(DensePolynomial::new_from_smart_slice(SmartSlice::Borrowed(
                            denom_low,
                        )));
                        high.push(DensePolynomial::new_from_smart_slice(SmartSlice::Borrowed(
                            denom_high,
                        )));
                        (low, high)
                    },
                );

                // Now we make the denominator expressions
                let column_count = denominator_columns.len() as u16;
                let constant_expr = Expression::Constant(*constant_challenge);
                let (denom_low_expr, denom_high_expr, _) = (0..denominator_columns.len()).fold(
                    (constant_expr.clone(), constant_expr.clone(), F::ONE),
                    |(acc_low, acc_high, chal_acc), i| {
                        let chal_expr = Expression::Constant(chal_acc);
                        (
                            acc_low
                                + chal_expr.clone()
                                    * Expression::WitIn(initial_witness_id + i as u16),
                            acc_high
                                + chal_expr
                                    * Expression::WitIn(
                                        initial_witness_id + column_count + i as u16,
                                    ),
                            chal_acc * *column_separation_challenge,
                        )
                    },
                );
                // The numerator is all -1 so we can use a constant expression

                // build the sumcheck expression which is used to prove this layer links to the next one
                let sumcheck_expr = Expression::Challenge(0u16, layer_count, F::ONE, F::ZERO)
                    * (-denom_high_expr.clone() - denom_low_expr.clone()
                        + Expression::Challenge(1u16, 1, F::ONE, F::ZERO)
                            * denom_low_expr.clone()
                            * denom_high_expr.clone());

                let mut mles = denom_low_mles;
                mles.extend(denom_high_mles);
                MLEsAndExpressions {
                    mles,
                    sumcheck_expression: sumcheck_expr,
                    claim_expression,
                }
            }
        }
    }

    pub fn final_eval_count(&self) -> usize {
        match self {
            LogUpLayer::InitialLookup {
                denominator_columns,
                ..
            } => denominator_columns.len() * 2,
            LogUpLayer::InitialTable {
                denominator_columns,
                ..
            } => denominator_columns.len() * 2 + 2,
            LogUpLayer::Generic { .. } => 0,
        }
    }

    pub fn denominator_column_count(&self) -> usize {
        match self {
            LogUpLayer::InitialLookup {
                denominator_columns,
                ..
            }
            | LogUpLayer::InitialTable {
                denominator_columns,
                ..
            } => denominator_columns.len(),
            LogUpLayer::Generic { .. } => 0,
        }
    }

    pub fn combine_final_evals(
        &self,
        evals: &[F],
        batching_challenge: F,
    ) -> anyhow::Result<Vec<F>> {
        match self {
            LogUpLayer::InitialLookup {
                denominator_columns,
                ..
            } => {
                // In this case the evals are expected to be all of the low parts of the denominator followed by all of the high parts
                let total_columns = denominator_columns.len();
                anyhow::ensure!(
                    evals.len() == total_columns * 2,
                    "Expected {} final evaluations but got {}",
                    total_columns * 2,
                    evals.len()
                );
                let (low_evals, high_evals) = evals.split_at(total_columns);
                low_evals
                    .iter()
                    .zip(high_evals.iter())
                    .map(|(&low, &high)| {
                        let combined_eval = batching_challenge * (high - low) + low;
                        Ok(combined_eval)
                    })
                    .collect()
            }
            LogUpLayer::InitialTable {
                denominator_columns,
                ..
            } => {
                let total_columns = denominator_columns.len();
                anyhow::ensure!(
                    evals.len() == total_columns * 2 + 2,
                    "Expected {} final evaluations but got {}",
                    total_columns * 2 + 2,
                    evals.len()
                );
                let (numerator_evals, rest) = evals.split_at(2);
                let (low_evals, high_evals) = rest.split_at(total_columns);
                let numerator_eval = batching_challenge * (numerator_evals[1] - numerator_evals[0])
                    + numerator_evals[0];
                let combined_evals: Vec<F> = std::iter::once(numerator_eval)
                    .chain(
                        low_evals
                            .iter()
                            .zip(high_evals.iter())
                            .map(|(&low, &high)| batching_challenge * (high - low) + low),
                    )
                    .collect();

                Ok(combined_evals)
            }
            LogUpLayer::Generic { .. } => {
                anyhow::bail!("combine_final_evals should only be called on initial layers")
            }
        }
    }
}

#[derive(Clone, Debug)]
/// A LogUp GKR circuit stored as its evaluations on each layer.
pub struct LogUpCircuit<F: PrimeField> {
    /// The [`LogUpLayer`]s forming this circuit, from bottom to top
    layers: Vec<LogUpLayer<F>>,
}

impl<F: PrimeField> LogUpCircuit<F> {
    /// Creates a new circuit from a [`LogUpLayer`]
    pub fn new(initial_layer: LogUpLayer<F>) -> LogUpCircuit<F> {
        let layers = std::iter::successors(Some(initial_layer), |layer| layer.next_layer())
            .collect::<Vec<LogUpLayer<F>>>();
        LogUpCircuit { layers }
    }

    /// Crates a new [`LogUpCircuit`] for a lookup variant, the inputs to this function are:
    /// - `lookup_evals`, a series of slices of equal length that represent the columns we are looking up
    /// - `constant_challenge`, the first term of the LogUp denominator
    /// - `column_separation_challenge`, the challenge used as domain separation for each of the columns
    pub fn new_lookup_circuit(
        lookup_columns: &[Vec<F>],
        constant_challenge: F,
        column_separation_challenge: F,
    ) -> LogUpCircuit<F> {
        let initial_layer = LogUpLayer::InitialLookup {
            constant_challenge,
            column_separation_challenge,
            denominator_columns: lookup_columns.to_vec(),
        };

        LogUpCircuit::<F>::new(initial_layer)
    }

    /// Creates a new [`LogUpCircuit`] for a table variant, the inputs to this function are:
    /// - `table_evals`, a series of slices of equal length that represent the columns of the table,
    /// - `multiplicities`, a slice of [`F`] elements that are the evaluations of the multiplicity poly,
    /// - `constant_challenge`, the first term of the LogUp denominator,
    /// - `column_separation_challenge`, the challenge used as domain separation for each of the columns,
    pub fn new_table_circuit(
        table_columns: &[Vec<F>],
        multiplicities: &[F],
        constant_challenge: F,
        column_separation_challenge: F,
    ) -> LogUpCircuit<F> {
        let initial_layer = LogUpLayer::InitialTable {
            numerator: multiplicities.to_vec(),
            constant_challenge,
            column_separation_challenge,
            denominator_columns: table_columns.to_vec(),
        };

        LogUpCircuit::<F>::new(initial_layer)
    }

    /// Retrieves the output values of the circuit in the order `numerator`, `denominator`
    pub fn outputs(&self) -> Vec<F> {
        self.layers.last().map(|layer| layer.flat_evals()).unwrap()
    }

    /// Works out the total number of variables the [`LogUpCircuit`] has in its
    /// largest (aka input) layer.
    pub fn num_vars(&self) -> usize {
        self.layers[0].num_vars()
    }

    /// Getter for the layers.
    pub fn layers(&self) -> &[LogUpLayer<F>] {
        &self.layers
    }

    pub fn final_eval_count(&self) -> anyhow::Result<usize> {
        self.layers
            .first()
            .ok_or(anyhow::anyhow!("No Layers in LogUpCircuit"))
            .map(|layer| layer.final_eval_count())
    }

    pub fn combine_final_evals(
        &self,
        evals: &[F],
        batching_challenge: F,
    ) -> anyhow::Result<Vec<F>> {
        self.layers
            .first()
            .ok_or(anyhow::anyhow!("No Layers in LogUpCircuit"))
            .and_then(|layer| layer.combine_final_evals(evals, batching_challenge))
    }
}

#[cfg(test)]
mod tests {
    use std::slice;

    use crate::testing::random_field_vector;
    use ark_ff::{AdditiveGroup, Field};

    type F = ark_bn254::Fr;

    use super::*;

    #[test]
    fn test_circuit_construction() {
        let column = random_field_vector::<F>(1 << 10);

        let circuit =
            LogUpCircuit::<F>::new_lookup_circuit(slice::from_ref(&column), F::ZERO, F::ONE);

        let out = circuit.outputs();

        assert_eq!(out.len(), 4);

        let value = column.iter().fold(F::ZERO, |acc, val| {
            let inv = val.inverse().unwrap();
            acc - F::from(inv)
        });

        let out_num = out[0] * out[3] + out[1] * out[2];
        let out_denom = out[2] * out[3];
        let calculated = out_num * out_denom.inverse().unwrap();

        assert_eq!(value, calculated);
    }
}
