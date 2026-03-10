//! Module containing utility structs for working with LogUp GKR circuits.

use std::{
    borrow::Borrow,
    iter::{Product, Sum},
    ops::{Add, AddAssign, Mul, MulAssign},
};

use anyhow::{Result, anyhow, ensure};
use either::Either;
use ff_ext::ExtensionField;
use multilinear_extensions::{Expression, mle::MultilinearExtension};
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use sumcheck::structs::IOPProof;
use transcript::Transcript;

use super::circuit::LogUpCircuit;
use crate::Claim;
use rayon::prelude::*;

#[derive(Clone, Debug, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
/// Struct used to perform arithmetic on fractions
pub struct Fraction<F> {
    pub numerator: F,
    pub denominator: F,
}

impl<F> Fraction<F> {
    /// Create a new instance of a [`Fraction`].
    pub fn new(numerator: F, denominator: F) -> Fraction<F> {
        Fraction::<F> {
            numerator,
            denominator,
        }
    }

    /// Turns this fraction into a tuple, the first element is the numerator, the second is the denominator
    pub fn as_tuple(&self) -> (F, F)
    where
        F: Clone,
    {
        (self.numerator.clone(), self.denominator.clone())
    }
}

impl<F: ExtensionField, T: Borrow<Fraction<F>>> AddAssign<T> for Fraction<F> {
    fn add_assign(&mut self, rhs: T) {
        let rhs: &Fraction<F> = rhs.borrow();
        let numerator = (self.numerator * rhs.denominator) + (self.denominator * rhs.numerator);
        let denominator = self.denominator * rhs.denominator;
        *self = Fraction {
            numerator,
            denominator,
        };
    }
}

impl<F: ExtensionField, T: Borrow<Fraction<F>>> Add<T> for &Fraction<F> {
    type Output = Fraction<F>;

    fn add(self, rhs: T) -> Self::Output {
        let mut output = *self;
        output += rhs;
        output
    }
}

impl<F: ExtensionField, T: Borrow<Fraction<F>>> Add<T> for Fraction<F> {
    type Output = Fraction<F>;

    fn add(self, rhs: T) -> Self::Output {
        let mut output = self;
        output += rhs;
        output
    }
}

impl<F: ExtensionField, T: Borrow<Fraction<F>>> MulAssign<T> for Fraction<F> {
    fn mul_assign(&mut self, rhs: T) {
        let rhs: &Fraction<F> = rhs.borrow();
        self.numerator *= rhs.numerator;
        self.denominator *= rhs.denominator;
    }
}

impl<F: ExtensionField, T: Borrow<Fraction<F>>> Mul<T> for &Fraction<F> {
    type Output = Fraction<F>;

    fn mul(self, rhs: T) -> Self::Output {
        let mut output = *self;
        output *= rhs;
        output
    }
}

impl<F: ExtensionField, T: Borrow<Fraction<F>>> Mul<T> for Fraction<F> {
    type Output = Fraction<F>;

    fn mul(self, rhs: T) -> Self::Output {
        let mut output = self;
        output *= rhs;
        output
    }
}

impl<F: ExtensionField> Sum for Fraction<F> {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Fraction<F> {
        iter.fold(Fraction::<F>::ZERO, |acc, term| acc + term)
    }
}

impl<F: ExtensionField> Product for Fraction<F> {
    fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Fraction::<F>::ONE, |acc, term| acc * term)
    }
}

impl<F: ExtensionField> Fraction<F> {
    const ZERO: Fraction<F> = Fraction {
        numerator: F::ZERO,
        denominator: F::ONE,
    };

    const ONE: Fraction<F> = Fraction {
        numerator: F::ONE,
        denominator: F::ONE,
    };
    /// Checks whether this is the zero element.
    pub fn is_zero(&self) -> bool {
        (self.numerator == F::ZERO) && (self.denominator != F::ZERO)
    }
}

#[derive(Clone, Debug)]
/// Enum defining inputs to LogUp proofs.
/// We split lookup inputs and table inputs as different optimisations can be made in each case. Additionally it allows us to only do work proportional to the table size in the table proving case
/// which is useful when multiple different model layers use the same table.
pub enum LogUpInput<E: ExtensionField> {
    /// Lookup variant can have multiple instances in one [`LogUpInput::Lookup`], `columns_per_instance` is used to work out how many batches we need to prove.
    Lookup {
        column_evals: Vec<Vec<E::BaseField>>,
        constant_challenge: E,
        column_separation_challenge: E,
        columns_per_instance: usize,
    },
    /// Input for a Table proof.
    Table {
        column_evals: Vec<Vec<E::BaseField>>,
        multiplicities: Vec<E::BaseField>,
        constant_challenge: E,
        column_separation_challenge: E,
    },
}

impl<E: ExtensionField> LogUpInput<E> {
    pub fn new_lookup(
        column_evals: Vec<Vec<E::BaseField>>,
        constant_challenge: E,
        column_separation_challenge: E,
        columns_per_instance: usize,
    ) -> Result<LogUpInput<E>> {
        ensure!(
            !column_evals.is_empty(),
            "No column evals were provided for Lookup input"
        );

        // Unwrap is safe
        let first_evals_len = column_evals.first().unwrap().len();
        ensure!(
            first_evals_len.is_power_of_two(),
            "Need a power of two number of evaluations got: {first_evals_len}"
        );

        column_evals.iter().skip(1).try_for_each(|evals| {
            if evals.len() != first_evals_len {
                Err(anyhow!("All sets of evaluations should be the same length"))
            } else {
                Ok(())
            }
        })?;

        Ok(LogUpInput::Lookup {
            column_evals,
            constant_challenge,
            column_separation_challenge,
            columns_per_instance,
        })
    }

    pub fn new_table(
        column_evals: Vec<Vec<E::BaseField>>,
        multiplicities: Vec<E::BaseField>,
        constant_challenge: E,
        column_separation_challenge: E,
    ) -> Result<LogUpInput<E>> {
        ensure!(
            !column_evals.is_empty(),
            "No column evals were provided for Lookup input"
        );

        // Unwrap is safe
        let first_evals_len = column_evals.first().unwrap().len();

        ensure!(
            first_evals_len.is_power_of_two(),
            "Need a power of two number of evaluations got: {first_evals_len}"
        );

        column_evals.iter().skip(1).try_for_each(|evals| {
            if evals.len() != first_evals_len {
                Err(anyhow!("All sets of evaluations should be the same length"))
            } else {
                Ok(())
            }
        })?;
        ensure!(
            multiplicities.len() == first_evals_len,
            "Multiplicities length was not equal to column evaluations length, multiplicities: {}, columns: {}",
            multiplicities.len(),
            first_evals_len
        );

        Ok(LogUpInput::Table {
            column_evals,
            multiplicities,
            constant_challenge,
            column_separation_challenge,
        })
    }

    pub fn column_evals(&self) -> &[Vec<E::BaseField>] {
        match self {
            LogUpInput::Lookup { column_evals, .. } | LogUpInput::Table { column_evals, .. } => {
                column_evals
            }
        }
    }

    pub fn make_new_circuits(&self) -> Vec<LogUpCircuit<E>> {
        match self {
            LogUpInput::Lookup {
                column_evals,
                constant_challenge,
                column_separation_challenge,
                columns_per_instance,
            } => column_evals
                .par_chunks(*columns_per_instance)
                .map(|column_evals| {
                    LogUpCircuit::<E>::new_lookup_circuit(
                        column_evals,
                        *constant_challenge,
                        *column_separation_challenge,
                    )
                })
                .collect(),
            LogUpInput::Table {
                column_evals,
                multiplicities,
                constant_challenge,
                column_separation_challenge,
            } => {
                vec![LogUpCircuit::<E>::new_table_circuit(
                    column_evals,
                    multiplicities,
                    *constant_challenge,
                    *column_separation_challenge,
                )]
            }
        }
    }

    pub fn base_mles(&self) -> Vec<MultilinearExtension<'_, E>> {
        match self {
            LogUpInput::Lookup { column_evals, .. } => column_evals
                .iter()
                .map(|evaluations| {
                    let num_vars = evaluations.len().ilog2() as usize;
                    MultilinearExtension::<E>::from_evaluations_slice(num_vars, evaluations)
                })
                .collect(),
            LogUpInput::Table {
                column_evals,
                multiplicities,
                ..
            } => std::iter::once(multiplicities)
                .chain(column_evals.iter())
                .map(|evaluations| {
                    let num_vars = evaluations.len().ilog2() as usize;
                    MultilinearExtension::<E>::from_evaluations_slice(num_vars, evaluations)
                })
                .collect(),
        }
    }

    pub fn columns_per_instance(&self) -> usize {
        match self {
            LogUpInput::Lookup {
                columns_per_instance,
                ..
            } => *columns_per_instance,
            LogUpInput::Table { column_evals, .. } => column_evals.len(),
        }
    }
}

#[derive(Clone, Debug, Copy, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
/// Information that is supplied by the verifier based on what they expect in order to verify a [`LogUpBatchProof`].
pub struct LogUpVerifierInstance<E> {
    /// The constant challenge used to combine columns.
    constant_challenge: E,
    /// The column separation challenge used to combine columns.
    column_separation_challenge: E,
    /// The number of columns in this instance.
    num_columns: usize,
    /// Whether this instance is a lookup or table instance.
    instance_type: ProofType,
    /// The number of variables in this instance.
    num_vars: usize,
}

pub struct InstanceExpressions<E: ExtensionField> {
    pub(crate) sumcheck_expression: Expression<E>,
    pub(crate) claim_expression: Expression<E>,
}

impl<E: ExtensionField> LogUpVerifierInstance<E> {
    /// Create a new instance of a [`LogUpVerifierInstance`].
    pub fn new(
        constant_challenge: E,
        column_separation_challenge: E,
        num_columns: usize,
        instance_type: ProofType,
        num_vars: usize,
    ) -> LogUpVerifierInstance<E> {
        LogUpVerifierInstance {
            constant_challenge,
            column_separation_challenge,
            num_columns,
            instance_type,
            num_vars,
        }
    }

    /// Getter for the constant challenge.
    pub fn constant_challenge(&self) -> E {
        self.constant_challenge
    }
    /// Getter for the column separation challenge.
    pub fn column_separation_challenge(&self) -> E {
        self.column_separation_challenge
    }
    /// Getter for the number of columns.
    pub fn num_columns(&self) -> usize {
        self.num_columns
    }
    /// Getter for the instance type.
    pub fn instance_type(&self) -> ProofType {
        self.instance_type
    }
    /// Getter for the number of variables.
    pub fn num_vars(&self) -> usize {
        self.num_vars
    }

    /// Get the number of evaluations needed in the final round for this instance.
    pub fn num_final_round_evals(&self) -> usize {
        match self.instance_type() {
            ProofType::Lookup => 2 * self.num_columns(),
            ProofType::Table => 2 * (1 + self.num_columns()),
        }
    }

    /// Method that calculates the contribution of this instance to the final claim in the GKR circuit.
    /// It returns the contribution to the claim along with the full evaluations of the polynomials at the final round.
    pub fn final_claim(
        &self,
        final_round_evals: &[E],
        batching_challenge: E,
        sumcheck_point: &[E],
    ) -> anyhow::Result<Vec<Claim<E>>> {
        let point_length = sumcheck_point.len();
        match self.instance_type() {
            ProofType::Lookup => {
                let (low_evals, high_evals) = final_round_evals.split_at(self.num_columns());
                let output_claims = low_evals
                    .iter()
                    .zip(high_evals.iter())
                    .map(|(&low, &high)| {
                        let eval = batching_challenge * (high - low) + low;
                        Claim::<E>::new(
                            // self.num_vars() + 1 is subtracted because the final layer of the GKR circuit is over 1 fewer variable than the full column MLEs
                            sumcheck_point[point_length - 1 - self.num_vars()..].to_vec(),
                            eval,
                        )
                    })
                    .collect::<Vec<Claim<E>>>();

                Ok(output_claims)
            }
            ProofType::Table => {
                let (multiplicity_evals, table_column_evals) = final_round_evals.split_at(2);

                let (low_evals, high_evals) = table_column_evals.split_at(self.num_columns());
                let multiplicity_claim = batching_challenge
                    * (multiplicity_evals[1] - multiplicity_evals[0])
                    + multiplicity_evals[0];
                let output_claims = std::iter::once(multiplicity_claim)
                    .chain(
                        low_evals
                            .iter()
                            .zip(high_evals.iter())
                            .map(|(&low, &high)| batching_challenge * (high - low) + low),
                    )
                    .map(|eval| {
                        Claim::<E>::new(
                            // self.num_vars() + 1 is subtracted because the final layer of the GKR circuit is over 1 fewer variable than the full column MLEs
                            sumcheck_point[point_length - 1 - self.num_vars()..].to_vec(),
                            eval,
                        )
                    })
                    .collect::<Vec<Claim<E>>>();

                Ok(output_claims)
            }
        }
    }

    pub fn instance_expressions(
        &self,
        layer_count: &mut usize,
        initial_witness_id: &mut u16,
        is_final_round: bool,
    ) -> InstanceExpressions<E> {
        // build the claim expression which is used to link to the previous layer to this one, common to all layer types
        let claim_num_low_id = *layer_count as u16 * 4;
        let claim_num_high_id = claim_num_low_id + 1;
        let claim_denom_low_id = claim_num_high_id + 1;
        let claim_denom_high_id = claim_denom_low_id + 1;
        let claim_expression = Expression::Challenge(0u16, *layer_count, E::ONE, E::ZERO)
            * (Expression::Challenge(2, 1, E::ONE, E::ZERO)
                * (Expression::WitIn(claim_num_high_id) - Expression::WitIn(claim_num_low_id))
                + Expression::WitIn(claim_num_low_id)
                + Expression::Challenge(1u16, 1, E::ONE, E::ZERO)
                    * (Expression::Challenge(2, 1, E::ONE, E::ZERO)
                        * (Expression::WitIn(claim_denom_high_id)
                            - Expression::WitIn(claim_denom_low_id))
                        + Expression::WitIn(claim_denom_low_id)));

        if !is_final_round {
            let num_low_id = *initial_witness_id;
            let num_high_id = num_low_id + 1;
            let denom_low_id = num_high_id + 1;
            let denom_high_id = denom_low_id + 1;

            // build the sumcheck expression which is used to prove this layer links to the next one
            let sumcheck_expr = Expression::Challenge(0u16, *layer_count, E::ONE, E::ZERO)
                * (Expression::WitIn(num_low_id) * Expression::WitIn(denom_high_id)
                    + Expression::WitIn(num_high_id) * Expression::WitIn(denom_low_id)
                    + Expression::Challenge(1u16, 1, E::ONE, E::ZERO)
                        * Expression::WitIn(denom_low_id)
                        * Expression::WitIn(denom_high_id));
            // Increment the layer count and witness id for the next call
            *layer_count += 1;
            *initial_witness_id += 4;
            InstanceExpressions {
                sumcheck_expression: sumcheck_expr,
                claim_expression,
            }
        } else {
            match self.instance_type() {
                ProofType::Table => {
                    let num_low_id = *initial_witness_id;
                    let num_high_id = num_low_id + 1;
                    // Now we make the denominator expressions
                    let constant_expr =
                        Expression::Constant(Either::Right(self.constant_challenge()));
                    let (denom_low_expr, denom_high_expr, _) = (0..self.num_columns() as u16).fold(
                        (constant_expr.clone(), constant_expr.clone(), E::ONE),
                        |(acc_low, acc_high, chal_acc), i| {
                            let chal_expr = Expression::Constant(Either::Right(chal_acc));
                            (
                                acc_low
                                    + chal_expr.clone() * Expression::WitIn(num_high_id + 1 + i),
                                acc_high
                                    + chal_expr
                                        * Expression::WitIn(
                                            num_high_id + 1 + self.num_columns() as u16 + i,
                                        ),
                                chal_acc * self.column_separation_challenge(),
                            )
                        },
                    );

                    // build the sumcheck expression which is used to prove this layer links to the next one
                    let sumcheck_expr = Expression::Challenge(0u16, *layer_count, E::ONE, E::ZERO)
                        * (Expression::WitIn(num_low_id) * denom_high_expr.clone()
                            + Expression::WitIn(num_high_id) * denom_low_expr.clone()
                            + Expression::Challenge(1u16, 1, E::ONE, E::ZERO)
                                * denom_low_expr.clone()
                                * denom_high_expr.clone());
                    // Increment the layer count and witness id for the next call
                    *layer_count += 1;
                    *initial_witness_id += (2 * (1 + self.num_columns())) as u16;
                    InstanceExpressions {
                        sumcheck_expression: sumcheck_expr,
                        claim_expression,
                    }
                }
                ProofType::Lookup => {
                    // Now we make the denominator expressions
                    let constant_expr =
                        Expression::Constant(Either::Right(self.constant_challenge()));
                    let (denom_low_expr, denom_high_expr, _) = (0..self.num_columns() as u16).fold(
                        (constant_expr.clone(), constant_expr.clone(), E::ONE),
                        |(acc_low, acc_high, chal_acc), i| {
                            let chal_expr = Expression::Constant(Either::Right(chal_acc));
                            (
                                acc_low
                                    + chal_expr.clone()
                                        * Expression::WitIn(*initial_witness_id + i),
                                acc_high
                                    + chal_expr
                                        * Expression::WitIn(
                                            *initial_witness_id + self.num_columns() as u16 + i,
                                        ),
                                chal_acc * self.column_separation_challenge(),
                            )
                        },
                    );
                    // The numerator is all -1 so we can use a constant expression

                    // build the sumcheck expression which is used to prove this layer links to the next one
                    let sumcheck_expr = Expression::Challenge(0u16, *layer_count, E::ONE, E::ZERO)
                        * (-denom_high_expr.clone() - denom_low_expr.clone()
                            + Expression::Challenge(1u16, 1, E::ONE, E::ZERO)
                                * denom_low_expr.clone()
                                * denom_high_expr.clone());
                    // Increment the layer count and witness id for the next call
                    *layer_count += 1;
                    *initial_witness_id += (2 * self.num_columns()) as u16;
                    InstanceExpressions {
                        sumcheck_expression: sumcheck_expr,
                        claim_expression,
                    }
                }
            }
        }
    }
}

#[derive(Clone, Debug, Copy, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub enum ProofType {
    Lookup,
    Table,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound(serialize = "E: Serialize", deserialize = "E: DeserializeOwned"))]
/// Struct used to store all information needed to verify a LogUp GKR argument.
pub struct LogUpBatchProof<E: ExtensionField> {
    /// Sumcheck proofs for each round
    pub sumcheck_proofs: Vec<IOPProof<E>>,
    /// The evaluations of the polynomials at each layer
    pub round_evaluations: Vec<Vec<E>>,
    /// Claims about the individual column evals from the last round (so before they are merged using the column challenges)
    pub output_claims: Vec<Claim<E>>,
    /// The outputs of the circuit
    pub circuit_outputs: Vec<Vec<E>>,
    /// Whether this proof wa for lookups or tables
    pub proof_type: ProofType,
    /// How many variables each instance is.
    pub num_vars_per_instance: Vec<usize>,
    /// The number of denominator columns for each instance.
    pub num_denominator_columns_per_instance: Vec<usize>,
}

impl<E: ExtensionField> LogUpBatchProof<E> {
    pub fn append_to_transcript<T: Transcript<E>>(&self, transcript: &mut T) {
        self.circuit_outputs
            .iter()
            .for_each(|evals| transcript.append_field_element_exts(evals));
    }

    pub fn fractional_outputs(&self) -> (Vec<E>, Vec<E>) {
        self.circuit_outputs
            .iter()
            .map(|evals| {
                (
                    evals[0] * evals[3] + evals[1] * evals[2],
                    evals[2] * evals[3],
                )
            })
            .unzip()
    }

    pub fn proofs_and_evals(&self) -> impl Iterator<Item = (&IOPProof<E>, &Vec<E>)> {
        self.sumcheck_proofs
            .iter()
            .zip(self.round_evaluations.iter())
    }

    pub fn circuit_outputs(&self) -> &[Vec<E>] {
        &self.circuit_outputs
    }

    pub fn output_claims(&self) -> &[Claim<E>] {
        &self.output_claims
    }

    pub fn proof_type(&self) -> ProofType {
        self.proof_type
    }

    pub fn num_instances(&self) -> usize {
        self.num_vars_per_instance.len()
    }

    pub fn final_round_evals(&self) -> Vec<E> {
        self.round_evaluations.last().unwrap().clone()
    }

    pub fn point(&self) -> &[E] {
        self.output_claims[0].point()
    }
}

#[derive(Debug, Clone)]
pub struct LogUpBatchVerifierClaim<E: ExtensionField> {
    /// This is the final claim returned by the GKR circuit
    claim: E,
    /// The output claims about the individual polynomials
    output_claims: Vec<Claim<E>>,
    /// The full point for the final claims
    point: Vec<E>,
    /// All poly evaluations in order
    poly_evals: Vec<E>,
    numerators: Vec<E>,
    denominators: Vec<E>,
}

impl<E: ExtensionField> LogUpBatchVerifierClaim<E> {
    pub fn new(
        claim: E,
        output_claims: Vec<Claim<E>>,
        point: Vec<E>,
        poly_evals: Vec<E>,
        numerators: Vec<E>,
        denominators: Vec<E>,
    ) -> LogUpBatchVerifierClaim<E> {
        LogUpBatchVerifierClaim {
            claim,
            output_claims,
            point,
            poly_evals,
            numerators,
            denominators,
        }
    }

    pub fn numerators(&self) -> &[E] {
        &self.numerators
    }

    pub fn denominators(&self) -> &[E] {
        &self.denominators
    }

    pub fn point(&self) -> &[E] {
        &self.point
    }

    pub fn claim(&self) -> E {
        self.claim
    }

    pub fn output_claims(&self) -> &[Claim<E>] {
        &self.output_claims
    }

    pub fn poly_evals(&self) -> &[E] {
        &self.poly_evals
    }
}
