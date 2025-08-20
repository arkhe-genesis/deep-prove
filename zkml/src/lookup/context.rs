//! File containing code for lookup witness generation.

use std::{
    cmp::Ordering,
    collections::{BTreeMap, BTreeSet, HashMap, btree_map},
};

use anyhow::{anyhow, ensure};
use ceno_p3::field::{Field, FieldAlgebra};
use ff_ext::ExtensionField;
use itertools::Itertools;
use mpcs::PolynomialCommitmentScheme;
use multilinear_extensions::{
    mle::MultilinearExtension,
    util::{ceil_log2, transpose},
};
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use tracing::{debug, warn};
use transcript::Transcript;
use utils::Metrics;
use witness::{InstancePaddingStrategy, RowMajorMatrix};

use super::logup_gkr::error::LogUpError;
use crate::{
    Claim, Element,
    iop::{ChallengeStorage, context::ProverContext},
    layers::{
        activation::{GeluTableData, Relu},
        provable::{NodeId, ProvableOp},
        transformer::{
            layernorm::{LAYERNORM_OUTPUT_SCALE_FACTOR, LAYERNORM_SCALE_FACTOR},
            softmax::{OUTPUT_SCALE_FACTOR, SCALE_FACTOR},
        },
    },
    lookup::logup_gkr::structs::{LogUpBatchVerifierClaim, LogUpInput},
    model::{InferenceTrace, ToIterator},
    quantization::{self, Fieldizer},
    to_base,
};
use rayon::prelude::*;

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize, Copy)]
/// Enum used for establishing the different table types needed to prove non-linear functions in a model.
pub enum TableType {
    /// Table used for the Relu activation function
    Relu,
    /// Table used for the GELU activation function
    GELU(GeluTableData),
    /// Table used for range checking (its size is determined by the quantisation bit size)
    Range,
    /// Table type used for computing Softmax, see the [`SoftmaxTableData`] struct for more info.
    Softmax(SoftmaxTableData),
    /// Table used for checking the normalisation error in Softmax operations, the first inner [`Element`] is `1` quantised by the scale factor, the second inner [`Element`] is the absolute value of the allowable error
    ErrorTable(Element, Element),
    /// Table used in requantisation
    RequantZeroTable,
    /// Table use to check if a value is zero or not, returns 1 if the value is zero and zero otherwise.
    ZeroTable,
    /// Table used to calculate inverse square root, see the [`InverseSQRTTableData`] struct for more info.
    InverseSQRT(InverseSQRTTableData),
}

// We impl PartialOrd and Ord ourselves on TableType, that way in a BTreeMap or BtreeSet they will always be ordered by the table with the most variables first.
impl PartialOrd for TableType {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for TableType {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        match Ord::cmp(
            &other.multiplicity_poly_vars(),
            &self.multiplicity_poly_vars(),
        ) {
            Ordering::Equal => Ord::cmp(&self.name(), &other.name()),
            order => order,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
/// Struct used to store Softmax table data
pub struct SoftmaxTableData {
    /// This is the result of calling [`f32::to_bits`] on the temperature value.
    float_bits: u32,
    /// The bit length of table size.
    table_size: usize,
    /// Any value larger than this gets mapped to zero.
    bkm: Element,
}

impl SoftmaxTableData {
    pub(crate) fn new(float_bits: u32, table_size: usize, bkm: Element) -> SoftmaxTableData {
        SoftmaxTableData {
            float_bits,
            table_size,
            bkm,
        }
    }

    pub(crate) fn float_temperature(&self) -> f32 {
        f32::from_bits(self.float_bits)
    }

    pub(crate) fn full_table_size(&self) -> Element {
        1 << self.table_size
    }

    pub(crate) fn size(&self) -> usize {
        self.table_size
    }

    pub(crate) fn bkm(&self) -> Element {
        self.bkm
    }

    pub(crate) fn table_output(&self, j: Element) -> Element {
        let float_temperature = self.float_temperature();
        let base: Element = 1 << 16;
        let bkm = self.bkm();
        let prod = base * j;
        if prod >= bkm {
            0
        } else {
            let float_exp = (-prod as f32 / (SCALE_FACTOR as f32 * float_temperature)).exp();
            (float_exp * OUTPUT_SCALE_FACTOR as f32).round() as Element
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
/// Struct used to store Softmax table data
pub struct InverseSQRTTableData {
    /// This is the result of calling [`f32::to_bits`] on the epsilon value.
    eps_bits: u32,
    /// The the number of bits to shift left by.
    range_check_bits: usize,
}

impl InverseSQRTTableData {
    pub(crate) fn new(eps_bits: u32, range_check_bits: usize) -> InverseSQRTTableData {
        InverseSQRTTableData {
            eps_bits,
            range_check_bits,
        }
    }

    pub(crate) fn float_epsilon(&self) -> f32 {
        f32::from_bits(self.eps_bits)
    }

    pub(crate) fn table_output(&self, j: Element) -> Element {
        let epsilon = self.float_epsilon();
        // First we have to shift by `range_checked_bits`
        let shifted_val = j << self.range_check_bits;
        // Now we convert back to float and perform the operation
        let float_output =
            1.0f32 / ((shifted_val as f32 / LAYERNORM_SCALE_FACTOR as f32) + epsilon).sqrt();
        // Now we use the output scale factor to recover the element value
        (float_output * LAYERNORM_OUTPUT_SCALE_FACTOR as f32).round() as Element
    }
}

impl TableType {
    pub fn get_table_columns<E: ExtensionField>(&self) -> Vec<Vec<E::BaseField>> {
        match self {
            TableType::GELU(qd) => {
                let (col_one, col_two): (Vec<E::BaseField>, Vec<E::BaseField>) = qd
                    .table()
                    .map(|(i, v)| {
                        let i_field: E = (i as Element).to_field();
                        let out_field: E = v.to_field();

                        (i_field.as_bases()[0], out_field.as_bases()[0])
                    })
                    .unzip();
                vec![col_one, col_two]
            }
            TableType::Relu => {
                #[allow(clippy::type_complexity)]
                let (col_one, col_two): (Vec<E::BaseField>, Vec<E::BaseField>) =
                    (*quantization::MIN..=*quantization::MAX)
                        .map(|i| {
                            let out = Relu::apply(i);
                            let i_field: E = i.to_field();
                            let out_field: E = out.to_field();

                            (i_field.as_bases()[0], out_field.as_bases()[0])
                        })
                        .unzip();
                vec![col_one, col_two]
            }
            TableType::Range => {
                let field = (0..1 << *quantization::BIT_LEN)
                    .map(|i| {
                        let i_field: E = i.to_field();
                        i_field.as_bases()[0]
                    })
                    .collect::<Vec<E::BaseField>>();
                vec![field]
            }

            TableType::Softmax(table_data) => {
                let table_size = table_data.full_table_size();

                let (in_column, out_column): (Vec<E::BaseField>, Vec<E::BaseField>) = (0
                    ..table_size)
                    .map(|j| {
                        let out_elem = table_data.table_output(j);
                        let in_field: E = j.to_field();
                        let out_field: E = out_elem.to_field();

                        (in_field.as_bases()[0], out_field.as_bases()[0])
                    })
                    .unzip();
                vec![in_column, out_column]
            }
            TableType::ErrorTable(quant_one, allowable_error) => {
                // Work out the minimum and maximum elements of the table
                let table_min = *quant_one - *allowable_error;
                let table_max = *quant_one + *allowable_error;
                // Work out the full table size
                let table_size = 1usize << ceil_log2(2 * *allowable_error as usize);
                let field = (table_min..=table_max)
                    .map(|elem| {
                        let f: E = elem.to_field();
                        f.as_bases()[0]
                    })
                    .chain(std::iter::repeat(E::BaseField::ZERO))
                    .take(table_size)
                    .collect::<Vec<E::BaseField>>();
                vec![field]
            }
            TableType::RequantZeroTable => {
                let (col_one, col_two): (Vec<E::BaseField>, Vec<E::BaseField>) =
                    (*quantization::MIN..=*quantization::MAX)
                        .map(|i| {
                            let out = if i != 0 { 0 } else { 1 };
                            let i_field: E = i.to_field();
                            let out_field: E = out.to_field();

                            (i_field.as_bases()[0], out_field.as_bases()[0])
                        })
                        .unzip();
                vec![col_one, col_two]
            }
            TableType::ZeroTable => {
                let table_size: Element = 1 << *quantization::BIT_LEN;
                let (in_column, out_column): (Vec<E::BaseField>, Vec<E::BaseField>) = (0
                    ..table_size)
                    .map(|i| {
                        let out: Element = if i != 0 { 0 } else { 1 };
                        let i_field: E = i.to_field();
                        let out_field: E = out.to_field();
                        (i_field.as_bases()[0], out_field.as_bases()[0])
                    })
                    .unzip();
                vec![in_column, out_column]
            }
            TableType::InverseSQRT(table_data) => {
                let table_max: Element = 1 << (2 * (*quantization::BIT_LEN - 1));
                let table_min = -table_max;
                let (in_column, out_column): (Vec<E::BaseField>, Vec<E::BaseField>) = (table_min
                    ..table_max)
                    .map(|i| {
                        let out = table_data.table_output(i);
                        let i_field: E = i.to_field();
                        let out_field: E = out.to_field();
                        (i_field.as_bases()[0], out_field.as_bases()[0])
                    })
                    .unzip();
                vec![in_column, out_column]
            }
        }
    }
    pub fn get_merged_table_column(&self, column_separator: Element) -> Vec<Element> {
        match self {
            TableType::GELU(qd) => qd
                .table()
                .map(|(i, v)| i as Element + v * column_separator)
                .collect(),
            TableType::Relu => (*quantization::MIN..=*quantization::MAX)
                .map(|i| {
                    let out = Relu::apply(i);

                    i + out * column_separator
                })
                .collect(),
            TableType::Range => (0..1 << *quantization::BIT_LEN).collect(),
            TableType::Softmax(table_data) => {
                let table_size = table_data.full_table_size();

                (0..table_size)
                    .map(|j| {
                        let out_elem = table_data.table_output(j);

                        j + COLUMN_SEPARATOR * out_elem
                    })
                    .collect()
            }
            TableType::ErrorTable(quant_one, allowable_error) => {
                // Work out the minimum and maximum elements of the table
                let table_min = *quant_one - *allowable_error;
                let table_max = *quant_one + *allowable_error;
                // Work out the full table size
                let table_size = 1usize << ceil_log2(2 * *allowable_error as usize);
                (table_min..=table_max)
                    .chain(std::iter::repeat(0))
                    .take(table_size)
                    .collect()
            }
            TableType::ZeroTable => {
                let table_size: Element = 1 << *quantization::BIT_LEN;
                (0..table_size)
                    .map(|i| {
                        let out: Element = if i != 0 { 0 } else { 1 };
                        i + COLUMN_SEPARATOR * out
                    })
                    .collect()
            }
            TableType::InverseSQRT(table_data) => {
                let table_max: Element = 1 << (2 * (*quantization::BIT_LEN - 1));
                let table_min = -table_max;

                (table_min..table_max)
                    .map(|i| {
                        let out = table_data.table_output(i);
                        i + COLUMN_SEPARATOR * out
                    })
                    .collect()
            }
            TableType::RequantZeroTable => (*quantization::MIN..=*quantization::MAX)
                .map(|i| {
                    let out = if i != 0 { 0 } else { 1 };

                    i + out * column_separator
                })
                .collect(),
        }
    }

    pub fn name(&self) -> String {
        match self {
            TableType::Relu => "Relu".to_string(),
            TableType::GELU(qd) => format!("GELU: {qd:?}"),
            TableType::Range => "Range".to_string(),
            TableType::Softmax(table_data) => {
                format!("Softmax - temperature: {}", table_data.float_temperature())
            }
            TableType::ErrorTable(quant_one, allowable_error) => {
                format!(
                    "Error Table - quantised one: {quant_one}, allowable error: {allowable_error}",
                )
            }
            TableType::ZeroTable => "Zero".to_string(),
            TableType::InverseSQRT(table_data) => format!(
                "InverseSQRT - normalisation: {}, shift: {}",
                table_data.float_epsilon(),
                table_data.range_check_bits
            ),
            TableType::RequantZeroTable => "Requant Zero Table".to_string(),
        }
    }

    /// Called by the verifier to evaluate _some_ columns itself. If the verifier can't verify the table
    /// efficiently, then it is done by regular PCS.
    pub fn evaluate_table_columns<E: ExtensionField>(
        &self,
        point: &[E],
    ) -> Result<Vec<E>, LogUpError> {
        match self {
            TableType::Range => {
                if point.len() != *quantization::BIT_LEN {
                    return Err(LogUpError::VerifierError(format!(
                        "Point was not the correct size to produce a range table evaluation, point size: {}, expected: {}",
                        point.len(),
                        *quantization::BIT_LEN
                    )));
                }

                Ok(vec![
                    point.iter().enumerate().fold(E::ZERO, |acc, (index, p)| {
                        acc + *p * E::from_canonical_u64(1u64 << index)
                    }),
                ])
            }
            TableType::Relu => {
                if point.len() != *quantization::BIT_LEN {
                    return Err(LogUpError::VerifierError(format!(
                        "Point was not the correct size to produce a relu table evaluation, point size: {}, expected: {}",
                        point.len(),
                        *quantization::BIT_LEN
                    )));
                }

                let first_column = point.iter().enumerate().fold(E::ZERO, |acc, (index, p)| {
                    acc + *p * E::from_canonical_u64(1u64 << index)
                }) - E::from_canonical_u64(1u64 << (*quantization::BIT_LEN - 1));

                let second_column = point.iter().enumerate().take(point.len() - 1).fold(
                    E::ZERO,
                    |acc, (index, p)| {
                        acc + *p * E::from_canonical_u64(1u64 << index) * point[point.len() - 1]
                    },
                );
                Ok(vec![first_column, second_column])
            }
            TableType::GELU(qd) => {
                let size = qd.table_size();
                if point.len() != size {
                    return Err(LogUpError::VerifierError(format!(
                        "Point was not the correct size to produce a Gelu table evaluation, point size: {}, expected: {}",
                        point.len(),
                        size
                    )));
                }
                let first_column = point.iter().enumerate().fold(E::ZERO, |acc, (index, p)| {
                    acc + *p * E::from_canonical_u64(1u64 << index)
                }) - E::from_canonical_u64(1u64 << (size - 1));
                Ok(vec![first_column])
            }
            TableType::Softmax(table_data) => {
                let size = table_data.size();
                if point.len() != size {
                    return Err(LogUpError::VerifierError(format!(
                        "Point was not the correct size to produce a softmax table evaluation, point size: {}, expected: {}",
                        point.len(),
                        size
                    )));
                }

                Ok(vec![
                    point.iter().enumerate().fold(E::ZERO, |acc, (index, p)| {
                        acc + *p * E::from_canonical_u64(1u64 << index)
                    }),
                ])
            }
            TableType::ErrorTable(..) => Ok(vec![]),
            TableType::ZeroTable => {
                if point.len() != *quantization::BIT_LEN {
                    return Err(LogUpError::VerifierError(format!(
                        "Point was not the correct size to produce a softmax table evaluation, point size: {}, expected: {}",
                        point.len(),
                        *quantization::BIT_LEN
                    )));
                }

                let (in_column_eval, out_column_eval) = point.iter().enumerate().fold(
                    (E::ZERO, E::ONE),
                    |(in_acc, out_acc), (index, p)| {
                        (
                            in_acc + *p * E::from_canonical_u64(1u64 << index),
                            out_acc * (E::ONE - *p),
                        )
                    },
                );
                Ok(vec![in_column_eval, out_column_eval])
            }
            TableType::InverseSQRT(..) => {
                if point.len() != 2 * (*quantization::BIT_LEN - 1) + 1 {
                    return Err(LogUpError::VerifierError(format!(
                        "Point was not the correct size to produce an InverseSQRT table evaluation, point size: {}, expected: {}",
                        point.len(),
                        2 * (*quantization::BIT_LEN - 1) + 1
                    )));
                }

                let first_column =
                    point.iter().enumerate().fold(E::ZERO, |acc, (index, p)| {
                        acc + *p * E::from_canonical_u64(1u64 << index)
                    }) - E::from_canonical_u64(1u64 << (2 * (*quantization::BIT_LEN - 1)));

                Ok(vec![first_column])
            }
            TableType::RequantZeroTable => {
                if point.len() != *quantization::BIT_LEN {
                    return Err(LogUpError::VerifierError(format!(
                        "Point was not the correct size to produce a requant zero check table evaluation, point size: {}, expected: {}",
                        point.len(),
                        *quantization::BIT_LEN
                    )));
                }

                let (in_column_eval, out_column_eval) = point.iter().enumerate().fold(
                    (
                        -E::from_canonical_u64(1u64 << (*quantization::BIT_LEN - 1)),
                        E::ONE,
                    ),
                    |(in_acc, out_acc), (index, p)| {
                        if index != *quantization::BIT_LEN - 1 {
                            (
                                in_acc + *p * E::from_canonical_u64(1u64 << index),
                                out_acc * (E::ONE - *p),
                            )
                        } else {
                            (
                                in_acc + *p * E::from_canonical_u64(1u64 << index),
                                out_acc * *p,
                            )
                        }
                    },
                );

                Ok(vec![in_column_eval, out_column_eval])
            }
        }
    }

    pub fn generate_challenge<E: ExtensionField, T: Transcript<E>>(&self, transcript: &mut T) -> E {
        match self {
            TableType::GELU(_) => transcript.sample_and_append_challenge(b"GELU").elements,
            TableType::Relu => transcript.sample_and_append_challenge(b"Relu").elements,
            TableType::Range | TableType::ErrorTable(..) => {
                // Theres only one column for a range check so we don't need to generate a challenge
                E::ONE
            }
            TableType::Softmax(..) => transcript.sample_and_append_challenge(b"Softmax").elements,
            TableType::ZeroTable => transcript.sample_and_append_challenge(b"Zero").elements,
            TableType::InverseSQRT(..) => {
                transcript
                    .sample_and_append_challenge(b"InverseSQRT")
                    .elements
            }
            TableType::RequantZeroTable => {
                transcript
                    .sample_and_append_challenge(b"RequantZero")
                    .elements
            }
        }
    }

    /// Gets the number of variables that the multiplicity polynomial will have for this table
    pub fn multiplicity_poly_vars(&self) -> usize {
        match self {
            TableType::GELU(qd) => qd.table_size(),
            TableType::Range
            | TableType::Relu
            | TableType::RequantZeroTable
            | TableType::ZeroTable => *quantization::BIT_LEN,
            TableType::Softmax(table_data) => table_data.size(),
            TableType::ErrorTable(_, allowable_error) => ceil_log2(2 * *allowable_error as usize),
            TableType::InverseSQRT(..) => 2 * (*quantization::BIT_LEN - 1) + 1,
        }
    }

    /// Gets the number of columns this able has
    pub fn num_columns(&self) -> usize {
        match self {
            TableType::GELU(..)
            | TableType::InverseSQRT(..)
            | TableType::Softmax(..)
            | TableType::ZeroTable
            | TableType::Relu
            | TableType::RequantZeroTable => 2,
            TableType::Range | TableType::ErrorTable(..) => 1,
        }
    }

    /// Function that returns any MLEs that have to be committed for this [`TableType`]
    pub fn committed_columns<'a, E: ExtensionField>(
        &'a self,
    ) -> Option<MultilinearExtension<'a, E>> {
        match self {
            TableType::GELU(qd) => {
                let out_column = to_base::<E, _>(qd.table().map(|(_, elem)| elem));
                Some(MultilinearExtension::<E>::from_evaluations_vec(
                    qd.table_size(),
                    out_column,
                ))
            }
            TableType::Softmax(table_data) => {
                let table_size = table_data.full_table_size();

                let out_column =
                    to_base::<E, _>((0..table_size).map(|j| table_data.table_output(j)));

                Some(MultilinearExtension::<E>::from_evaluations_vec(
                    table_data.size(),
                    out_column,
                ))
            }
            TableType::ErrorTable(quant_one, allowable_error) => {
                // Work out the minimum and maximum elements of the table
                let table_min = quant_one - allowable_error;
                let table_max = quant_one + allowable_error;
                // Work out the full table size
                let num_vars = ceil_log2(2 * *allowable_error as usize);
                let table_size = 1usize << num_vars;
                let column = (table_min..=table_max)
                    .map(|elem| {
                        let f: E = elem.to_field();
                        f.as_bases()[0]
                    })
                    .chain(std::iter::repeat(E::BaseField::ZERO))
                    .take(table_size)
                    .collect::<Vec<E::BaseField>>();
                Some(MultilinearExtension::<E>::from_evaluations_vec(
                    num_vars, column,
                ))
            }
            TableType::InverseSQRT(table_data) => {
                let table_max: Element = 1 << (2 * (*quantization::BIT_LEN - 1));
                let table_min = -table_max;
                let column =
                    to_base::<E, _>((table_min..table_max).map(|i| table_data.table_output(i)));
                let num_vars = 2 * (*quantization::BIT_LEN - 1) + 1;
                Some(MultilinearExtension::<E>::from_evaluations_vec(
                    num_vars, column,
                ))
            }

            _ => None,
        }
    }

    /// Method that takes all of the claims output by a logup table proof and outputs only those that need to be checked via commitment opening (excluding the multiplicity poly claim)
    pub fn table_claims<E: ExtensionField>(&self, claims: &[Claim<E>]) -> Vec<Claim<E>> {
        match self {
            TableType::Softmax(..)
            | TableType::ErrorTable(..)
            | TableType::InverseSQRT(..)
            | TableType::GELU(..) => {
                // For Softmax, InverSQRT and Error Table we just need the output column claim so the last of the slice
                vec![claims.last().cloned().unwrap()]
            }

            _ => vec![],
        }
    }

    pub fn has_committed_claims(&self) -> bool {
        matches!(
            self,
            TableType::Softmax(..)
                | TableType::ErrorTable(..)
                | TableType::InverseSQRT(..)
                | TableType::GELU(..)
        )
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
/// Struct stored in layers that use lookups that allows easy conversion of committed witness data to lookup data.
pub struct LayerLookupContext {
    /// These are the table types used, they are arranged in order so that the lookups with
    /// the most variables correspond to the first [`TableType`] and so on.
    pub(crate) tables: Vec<TableType>,
    /// The number of instances of each lookup
    pub(crate) instances_per_table: Vec<usize>,
}

impl LayerLookupContext {
    pub fn new(tables: Vec<TableType>, instances_per_table: Vec<usize>) -> LayerLookupContext {
        LayerLookupContext {
            tables,
            instances_per_table,
        }
    }

    pub fn create_logup_inputs<PCS, E>(
        &self,
        layer_commitment: &PCS::CommitmentWithWitness,
        challenge_storage: &ChallengeStorage<E>,
    ) -> anyhow::Result<Vec<LogUpInput<E>>>
    where
        E: ExtensionField,
        PCS: PolynomialCommitmentScheme<E>,
    {
        // First we extract the polynomials from the layer_commitment
        let polys = PCS::get_arc_mle_witness_from_commitment(layer_commitment);

        // There should be at least as many polynomials as there are lookup columns total
        let total_lookup_columns = self
            .tables
            .iter()
            .zip(self.instances_per_table.iter())
            .map(|(tt, &n)| tt.num_columns() * n)
            .sum::<usize>();

        ensure!(
            polys.len() >= total_lookup_columns,
            "Cannot create LogUp inputs because we were only provided with {} polynomials and expected {} lookup columns",
            polys.len(),
            total_lookup_columns
        );

        // Now we try_fold to make our output
        let (logup_inputs, _) = self
            .tables
            .iter()
            .zip(self.instances_per_table.iter())
            .try_fold(
                (Vec::<LogUpInput<E>>::new(), 0),
                |(mut inputs_acc, skip), (tt, &n)| {
                    let (constant_challenge, column_separation_challenge) = challenge_storage
                        .get_challenges_by_name(&tt.name())
                        .ok_or(anyhow!(
                            "No challenges found for Table {}, cannot generate LogUp input",
                            tt.name()
                        ))?;
                    let take = tt.num_columns() * n;
                    let column_evals = polys
                        .iter()
                        .skip(skip)
                        .take(take)
                        .map(|p| p.get_base_field_vec().to_vec())
                        .collect::<Vec<Vec<E::BaseField>>>();
                    let logup_input = LogUpInput::<E>::new_lookup(
                        column_evals,
                        constant_challenge,
                        column_separation_challenge,
                        tt.num_columns(),
                    )?;
                    inputs_acc.push(logup_input);
                    Result::<(Vec<LogUpInput<E>>, usize), anyhow::Error>::Ok((
                        inputs_acc,
                        skip + take,
                    ))
                },
            )?;

        Ok(logup_inputs)
    }

    pub fn verify_logup_batch_claim<E: ExtensionField>(
        &self,
        batch_claim: &LogUpBatchVerifierClaim<E>,
        challenge_storage: &ChallengeStorage<E>,
    ) -> anyhow::Result<()> {
        let poly_evals = batch_claim.poly_evals();
        let alpha = batch_claim.alpha();

        let (calc_claim, _, _) = self
            .tables
            .iter()
            .zip(self.instances_per_table.iter())
            .try_fold((E::ZERO, E::ONE, 0), |(acc, chal_acc, skip), (tt, &n)| {
                let take = tt.num_columns() * n;
                let (constant_challenge, csc) = challenge_storage
                    .get_challenges_by_name(&tt.name())
                    .ok_or(anyhow!(
                        "No challenges found for Table {}, cannot verify LogUp batch claim",
                        tt.name()
                    ))?;
                let (combined_eval, chal_update) = poly_evals
                    .iter()
                    .skip(skip)
                    .take(take)
                    .chunks(tt.num_columns())
                    .into_iter()
                    .map(|chunk| {
                        chunk
                            .into_iter()
                            .fold((constant_challenge, E::ONE), |(a, column_chal_acc), &e| {
                                (a + e * column_chal_acc, column_chal_acc * csc)
                            })
                            .0
                    })
                    .fold((acc, chal_acc), |(inner_acc, inner_chal_acc), e| {
                        (inner_acc + e * inner_chal_acc, inner_chal_acc * alpha)
                    });

                Result::<(E, E, usize), anyhow::Error>::Ok((
                    combined_eval,
                    chal_update,
                    skip + take,
                ))
            })?;

        ensure!(
            calc_claim == batch_claim.claim(),
            "Lookup verification failed, calculated claim {:?} did not equal LogUp claim {:?}",
            calc_claim,
            batch_claim.claim()
        );
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LookupContext {
    pub(crate) tables: Vec<TableType>,
}

impl LookupContext {
    pub fn new(set: &BTreeSet<TableType>) -> LookupContext {
        LookupContext {
            tables: set.iter().cloned().collect(),
        }
    }

    pub fn iter(&self) -> impl Iterator<Item = &TableType> {
        self.tables.iter()
    }

    pub fn is_empty(&self) -> bool {
        self.tables.is_empty()
    }

    pub fn create_logup_inputs<PCS, E>(
        &self,
        multiplicities_commitment: &PCS::CommitmentWithWitness,
        challenge_storage: &ChallengeStorage<E>,
    ) -> anyhow::Result<Vec<LogUpInput<E>>>
    where
        E: ExtensionField,
        PCS: PolynomialCommitmentScheme<E>,
    {
        // First we retrieve the multiplicity polynomials
        let mulitplicity_polys =
            PCS::get_arc_mle_witness_from_commitment(multiplicities_commitment);
        self.iter()
            .zip(mulitplicity_polys.iter())
            .map(|(tt, m_poly)| {
                let multiplicities = m_poly.get_base_field_vec().to_vec();
                let column_evals = tt.get_table_columns::<E>();
                let (constant_challenge, column_separation_challenge) = challenge_storage
                    .get_challenges_by_name(&tt.name())
                    .ok_or(anyhow!(
                        "No challenges found for Table {}, cannot generate LogUp input",
                        tt.name()
                    ))?;
                LogUpInput::<E>::new_table(
                    column_evals,
                    multiplicities,
                    constant_challenge,
                    column_separation_challenge,
                )
                .map_err(|e| anyhow!("{:?}", e))
            })
            .collect::<Result<Vec<LogUpInput<E>>, anyhow::Error>>()
    }
}

pub(crate) fn count_elements<I: IntoIterator<Item = Element>>(i: I) -> HashMap<Element, u64> {
    let mut count = HashMap::<Element, u64>::new();
    for v in i.into_iter() {
        *count.entry(v).or_default() += 1;
    }
    count
}

pub struct LookupWitnessGen<E: ExtensionField, PCS: PolynomialCommitmentScheme<E>> {
    /// Contains the count of elements per table type.
    ///
    /// These values are later used to compute the GKR's multiplicities.
    pub(crate) element_count: BTreeMap<TableType, HashMap<Element, u64>>,
    pub(crate) logup_witnesses: HashMap<NodeId, PCS::CommitmentWithWitness>,
}

impl<E: ExtensionField, PCS: PolynomialCommitmentScheme<E>> Default for LookupWitnessGen<E, PCS> {
    fn default() -> Self {
        LookupWitnessGen {
            element_count: BTreeMap::default(),
            logup_witnesses: HashMap::default(),
        }
    }
}

impl<E: ExtensionField, PCS: PolynomialCommitmentScheme<E>> LookupWitnessGen<E, PCS> {
    /// Consume the lookups and witness of `other` into this instance.
    fn consume(&mut self, other: Self) {
        for (table_type, elements) in other.element_count.into_iter() {
            match self.element_count.entry(table_type) {
                btree_map::Entry::Vacant(vacant_entry) => {
                    vacant_entry.insert(elements);
                }
                btree_map::Entry::Occupied(mut occupied_entry) => {
                    let agg_count = occupied_entry.get_mut();
                    for (element, count) in elements.into_iter() {
                        *agg_count.entry(element).or_default() += count;
                    }
                }
            }
        }
        self.logup_witnesses.extend(other.logup_witnesses);
    }
}

pub(crate) const COLUMN_SEPARATOR: Element = 1 << 32;

#[derive(Debug)]
pub struct LookupWitness<E: ExtensionField, PCS: PolynomialCommitmentScheme<E>> {
    pub challenge_storage: ChallengeStorage<E>,
    pub logup_witnesses: HashMap<NodeId, PCS::CommitmentWithWitness>,
    pub table_witnesses: Option<PCS::CommitmentWithWitness>,
}

impl<E: ExtensionField, PCS: PolynomialCommitmentScheme<E>> Default for LookupWitness<E, PCS> {
    fn default() -> Self {
        LookupWitness {
            challenge_storage: ChallengeStorage::default(),
            logup_witnesses: HashMap::default(),
            table_witnesses: None,
        }
    }
}

pub fn generate_lookup_witnesses<'a, 'b, E, T: Transcript<E>, PCS: PolynomialCommitmentScheme<E>>(
    trace: &'b InferenceTrace<'a, E, Element>,
    ctx: &ProverContext<E, PCS>,
    transcript: &mut T,
) -> Result<LookupWitness<E, PCS>, LogUpError>
where
    E: ExtensionField + Serialize + DeserializeOwned,
    E::BaseField: Serialize + DeserializeOwned,
    PCS::CommitmentWithWitness: Serialize + DeserializeOwned,
{
    // If the lookup context is empty then there are no lookup witnesses to generate so we return default values
    if ctx.lookup.is_empty() {
        warn!("Lookup witness generation: no tables found, returning empty context TEST?");
        return Ok(LookupWitness::default());
    }

    // Make the witness gen struct that stores relevant table lookup data
    debug!("== Witness poly fields generation ==");
    let metrics = Metrics::new();
    let mut witness_gen = LookupWitnessGen::<E, PCS>::default();
    let mut store = trace.store.clone();

    for (node_id, _) in ctx.steps_info.to_forward_iterator() {
        let step = trace
            .get_step(&node_id)
            .ok_or(LogUpError::ProvingError(format!(
                "Node {node_id} not found in trace"
            )))?;
        let gen_w = step
            .op
            .gen_lookup_witness(node_id, ctx, &step.step_data, &mut store)
            .map_err(|e| {
                LogUpError::ParameterError(format!(
                    "Error generating lookup witness for node {node_id} with error: {e}"
                ))
            })?;
        witness_gen.consume(gen_w);
    }
    debug!(
        "== Witness poly fields generation metrics {} ==",
        metrics.to_span()
    );

    debug!("== Witness table multiplicities generation ==");
    let metrics = Metrics::new();
    // calculate the table multiplicities

    let multiplicities = witness_gen
        .element_count
        .par_iter()
        .map(|(table_type, table_lookup_data)| {
            let table_column = table_type.get_merged_table_column(COLUMN_SEPARATOR);

            // Check to see that all the lookup values are present in the table
            #[cfg(test)]
            {
                for key in table_lookup_data.keys() {
                    let check = table_column.contains(key);
                    if !check {
                        println!(
                            "Tried to lookup key: {}, for table: {}",
                            key,
                            table_type.name()
                        );
                    }
                }
            }
            // We have to account for repeated entries in the lookup table. This is usually the case if the table we want to lookup from is not a power of two size, in that case we pick a row from the table
            // and repeat it until the table has the desired size.
            let table_column_map =
                table_column
                    .iter()
                    .fold(BTreeMap::<Element, u64>::new(), |mut map, elem| {
                        *map.entry(*elem).or_insert(0) += 1;
                        map
                    });
            let multiplicities = table_column
                .iter()
                .map(|table_val| {
                    if let Some(lookup_count) = table_lookup_data.get(table_val) {
                        let table_count = *table_column_map.get(table_val).unwrap();
                        let inv = if table_count != 1 {
                            E::BaseField::from_canonical_u64(table_count).inverse()
                        } else {
                            E::BaseField::ONE
                        };
                        E::BaseField::from_canonical_u64(*lookup_count) * inv
                    } else {
                        E::BaseField::ZERO
                    }
                })
                .collect::<Vec<E::BaseField>>();

            Ok(multiplicities)
        })
        .collect::<Result<Vec<Vec<E::BaseField>>, LogUpError>>()?;

    let grouped_by_vars = witness_gen
        .element_count
        .keys()
        .map(|table_type| (table_type.multiplicity_poly_vars(), *table_type))
        .into_group_map();
    let rmms = grouped_by_vars
        .into_iter()
        .map(|(nv, list)| (nv, list.len()))
        .sorted_by(|a, b| Ord::cmp(&b.0, &a.0))
        .fold((vec![], 0), |(mut acc, skip), (_, take)| {
            let transposed = transpose(
                multiplicities
                    .iter()
                    .skip(skip)
                    .take(take)
                    .cloned()
                    .collect::<Vec<Vec<E::BaseField>>>(),
            );

            let rmm = RowMajorMatrix::new_by_inner_matrix(
                ceno_p3::matrix::dense::DenseMatrix::new(transposed.concat(), take),
                InstancePaddingStrategy::Default,
            );
            acc.push(rmm);
            (acc, skip + take)
        })
        .0;

    let table_witness = ctx
        .commitment_ctx
        .batch_commit(rmms)
        .map_err(|e| LogUpError::ParameterError(format!("{e:?}")))?;
    debug!(
        "== Witness table multiplicities metrics {} ==",
        metrics.to_span()
    );

    // Write the witness commitments to the transcript
    for (node_id, _) in ctx.steps_info.to_forward_iterator() {
        if let Some(prover_commit) = witness_gen.logup_witnesses.get(&node_id) {
            let comm = PCS::get_pure_commitment(prover_commit);
            PCS::write_commitment(&comm, transcript)
                .map_err(|e| LogUpError::ParameterError(format!("{e:?}")))?;
        }
    }

    let table_comm = PCS::get_pure_commitment(&table_witness);
    PCS::write_commitment(&table_comm, transcript)
        .map_err(|e| LogUpError::ParameterError(format!("{e:?}")))?;

    debug!("== Challenge storage ==");
    let metrics = Metrics::new();
    let challenge_storage = initialise_from_table_set::<E, T, _>(ctx.lookup.iter(), transcript);
    debug!("== Challenge storage metrics {} ==", metrics.to_span());

    Ok(LookupWitness {
        challenge_storage,
        logup_witnesses: witness_gen.logup_witnesses,
        table_witnesses: Some(table_witness),
    })
}

fn initialise_from_table_set<
    'a,
    E: ExtensionField,
    T: Transcript<E>,
    I: Iterator<Item = &'a TableType>,
>(
    set: I,
    transcript: &mut T,
) -> ChallengeStorage<E> {
    let constant_challenge = transcript
        .sample_and_append_challenge(b"table_constant")
        .elements;
    let challenge_map = set
        .map(|table_type| {
            let challenge = table_type.generate_challenge(transcript);

            (table_type.name(), challenge)
        })
        .collect::<HashMap<String, E>>();
    ChallengeStorage::<E> {
        constant_challenge,
        challenge_map,
    }
}
