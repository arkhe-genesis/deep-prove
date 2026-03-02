//! Module containing definitions of the supported table types for lookup arguments.

use anyhow::ensure;

use super::*;
use transcript::Transcript;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord, Hash, Copy)]
/// Enum representing the different types of lookup tables available.
pub enum TableType {
    Exp,
    GeLU,
    ReLU,
    Requantise,
    ShiftCheck,
    SignedZeroCheck,
    ZeroCheck,
}

impl Display for TableType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let name = match self {
            TableType::Exp => "ExpTable",
            TableType::GeLU => "GeLUTable",
            TableType::ReLU => "ReLUTable",
            TableType::Requantise => "RequantiseTable",
            TableType::ShiftCheck => "ShiftCheckTable",
            TableType::SignedZeroCheck => "SignedZeroCheckTable",
            TableType::ZeroCheck => "ZeroCheckTable",
        };
        write!(f, "{}", name)
    }
}

#[derive(Clone, Debug, Copy)]
/// Enum used to indicate whether a table expects all positive, all negative or mixed sign inputs.
pub enum TableSign {
    /// Table expects all positive (or zero) inputs.
    Positive,
    /// Table expects all negative (or zero) inputs.
    Negative,
    /// Table can handle mixed sign inputs.
    Mixed,
}

impl TableType {
    /// Returns the number of columns for the table type.
    pub fn num_columns(&self) -> usize {
        match self {
            TableType::Requantise | TableType::ShiftCheck => 1,
            _ => 2,
        }
    }

    /// Flag indicating whether the table has committed columns.
    pub fn commit_output_column(&self) -> bool {
        // Everything except ReLU, Requantise and ShiftCheck has committed output columns
        !matches!(
            self,
            TableType::ReLU | TableType::Requantise | TableType::ShiftCheck
        )
    }

    /// Method returning the sign expectation of the table input.
    pub fn input_sign(&self) -> TableSign {
        match self {
            TableType::Exp => TableSign::Negative,
            TableType::ShiftCheck | TableType::ZeroCheck => TableSign::Positive,
            _ => TableSign::Mixed,
        }
    }

    /// Indicates whether the table input always has the same sign (or zero).
    pub fn is_signed(&self) -> bool {
        matches!(self.input_sign(), TableSign::Mixed)
    }

    /// Applies the table operation to the given input value, considering the input and output scale factors,
    /// as well as the minimum and maximum input values.
    pub(crate) fn apply_op(
        &self,
        input: Element,
        input_sf: f32,
        output_sf: f32,
        input_min: Element,
        input_max: Element,
    ) -> Result<Element> {
        let output = match self {
            TableType::Exp => {
                if input <= input_min {
                    0
                } else if input >= input_max {
                    output_sf.round_ties_even() as Element
                } else {
                    let float_exp = (input as f32 / input_sf).exp();
                    (float_exp * output_sf).round_ties_even() as Element
                }
            }
            TableType::GeLU => {
                if input < input_min {
                    (gelu_float(&(input_min as f32 / input_sf)) * output_sf).round_ties_even()
                        as Element
                } else if input > input_max {
                    (gelu_float(&(input_max as f32 / input_sf)) * output_sf).round_ties_even()
                        as Element
                } else {
                    let input_float = (input as f32) / input_sf;
                    let gelu_result = gelu_float(&input_float);

                    (gelu_result * output_sf).round_ties_even() as Element
                }
            }
            TableType::ReLU => input.clamp(0, input_max),
            TableType::Requantise => input.clamp(input_min, input_max),
            TableType::ShiftCheck => {
                ensure!(
                    (input_min..=input_max).contains(&input),
                    "Input value {input} out of range for ShiftCheckTable, min: {input_min}, max: {input_max}"
                );
                input
            }
            TableType::SignedZeroCheck => {
                ensure!(
                    (input_min..=input_max).contains(&input),
                    "Input value {input} out of range for SignedZeroCheckTable, min: {input_min}, max: {input_max}"
                );
                input.signum()
            }
            TableType::ZeroCheck => {
                ensure!(
                    (input_min..=input_max).contains(&input),
                    "Input value {input} out of range for ZeroCheckTable, min: {input_min}, max: {input_max}"
                );
                if input == 0 { 1 } else { 0 }
            }
        };

        Ok(output)
    }

    pub fn generate_challenge<E: ExtensionField, T: Transcript<E>>(&self, transcript: &mut T) -> E {
        match self {
            TableType::GeLU => transcript.sample_and_append_challenge(b"GeLU").elements,
            TableType::ReLU => transcript.sample_and_append_challenge(b"ReLU").elements,
            TableType::Requantise => {
                transcript
                    .sample_and_append_challenge(b"Requantise")
                    .elements
            }
            TableType::Exp => transcript.sample_and_append_challenge(b"Exp").elements,
            TableType::ZeroCheck => transcript.sample_and_append_challenge(b"Zero").elements,
            TableType::SignedZeroCheck => {
                transcript
                    .sample_and_append_challenge(b"SignedZero")
                    .elements
            }
            TableType::ShiftCheck => {
                transcript
                    .sample_and_append_challenge(b"ShiftCheck")
                    .elements
            }
        }
    }
}

/// Compute the GeLU
pub(crate) fn gelu_float(x: &f32) -> f32 {
    // NOTE: The commented-out formula below is based on [1]
    //
    // [1]: https://docs.pytorch.org/docs/stable/generated/torch.nn.GELU.html
    //
    // let c = (2.0f32 / std::f32::consts::PI).sqrt();

    // let x_cubed = x * x * x;
    // let inner_term = c * (x + 0.044715 * x_cubed);
    // 0.5 * x * (1.0 + inner_term.tanh())

    // NOTE This formula matches burn's GELU implementation.
    // The difference from PyTorch is that instead of tanh approx. it uses
    // error function from `libm::erf`
    let inner_term = libm::erf((x / 2.0_f32.sqrt()) as f64) as f32;
    0.5 * x * (1.0 + inner_term)
}
