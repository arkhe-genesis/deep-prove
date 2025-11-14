//! Axis definition for use in EinSum layers.

use std::{collections::HashSet, ops::Index, sync::OnceLock};

use crate::{
    Claim, Shape, Tensor, commit::compute_betas_eval, layers::transformer::mha::eval_zeroifier_mle,
    to_bit_sequence_le,
};

use anyhow::{Result, anyhow, bail, ensure};
use ff_ext::ExtensionField;
use itertools::{Itertools, izip};
use multilinear_extensions::{mle::MultilinearExtension, util::ceil_log2};
use serde::{Deserialize, Serialize};

use rayon::prelude::*;

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
/// Used in [`Axis`] to indicate whether an input/output tensor has this axis or not, and if it
/// does what dimension it corresponds to in that tensors [`Shape`].
pub enum Dimension {
    /// The tensor has this axis, and it corresponds to the given dimension in its shape.
    Present(usize),
    /// The tensor does not have this axis.
    Absent,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
/// Used in [`Axis`] to indicate whether this will be a "Stacked", "Contracted" or "Outer" axis in the
/// evaluation of the EinSum operation.
pub enum AxisType {
    /// This axis appears in both input and output tensors, and is not summed over.
    Stacked,
    /// This axis appears in both input tensors but not the output, and is summed over.
    Contracted,
    /// This axis appears in only one input and the output, and is not summed over.
    Outer,
}

impl Index<AxisType> for [usize; 3] {
    type Output = usize;
    fn index(&self, index: AxisType) -> &usize {
        match index {
            AxisType::Stacked => &self[0],
            AxisType::Outer => &self[1],
            AxisType::Contracted => &self[2],
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
/// An axis that can be common to multiple tensors in an operation.
/// This is used to define how to fix the various axes of the inputs in the EinSum operation so that it can be proven via Sumcheck.
/// It also allows us to calculate the output shape given the input shapes.
pub struct Axis {
    /// This indicates whether the LHS of the operation has this axs present
    pub(crate) lhs_input: Dimension,
    /// This vector indicates the presence or absence of the axis in each input tensor on the RHS of the operation.
    pub(crate) rhs_inputs: Vec<Dimension>,
    /// This vector indicates the presence or absence of the axis in each output tensor of the operation.
    pub(crate) outputs: Vec<Dimension>,
    /// This vector indicates the presence or absence of the axis in each bias tensor of the operation.
    pub(crate) biases: Vec<Dimension>,
    /// A character representation of the axis, used in Einstein summation notation.
    pub repr: char,
    /// The type of the axis in the EinSum operation.
    pub axis_type: AxisType,
}

impl Axis {
    /// Create a new "empty" [`Axis`].
    pub fn new(
        input_count: usize,
        output_count: usize,
        bias_count: usize,
        repr: char,
        axis_type: AxisType,
    ) -> Self {
        Self {
            lhs_input: Dimension::Absent,
            rhs_inputs: vec![Dimension::Absent; input_count],
            outputs: vec![Dimension::Absent; output_count],
            biases: vec![Dimension::Absent; bias_count],
            repr,
            axis_type,
        }
    }
    /// Returns an iterator of all input dimensions (LHS and RHS) for this axis.
    pub fn inputs(&self) -> impl Iterator<Item = &Dimension> {
        std::iter::once(&self.lhs_input).chain(self.rhs_inputs.iter())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
/// A mapping of axes for an Einstein summation operation, defining how axes are shared among input and output tensors.
pub struct AxesMapping {
    /// The number of input tensors in the operation.
    input_count: usize,
    /// The number of output tensors in the operation.
    output_count: usize,
    /// The number of bias tensors in the operation.
    bias_count: usize,
    /// A list of all axes involved in the operation.
    axes: Vec<Axis>,
}

#[derive(Debug, Clone)]
/// Intermediate struct used during parsing of an equation to hold the dimension information for a tensor.
struct TensorDimInfo {
    input_id: usize,
    dims: Vec<char>,
}

impl TensorDimInfo {
    fn new(input_id: usize, dims: Vec<char>) -> Self {
        Self { input_id, dims }
    }
}

impl AxesMapping {
    /// Parses an [`AxesMapping`] from an equation of the form "A(ij)@B(jk):C(jl)->F(ik)+BIAS(k):G(il)".
    /// Capital letters are used to identify tensors, lower-case letters identify axes.
    /// The '@' symbol indicates tensors on the left are acting on tensors on the right, ':' separates multiple tensors being acted on by the same tensor
    /// and the '->' indicates the start of the output tensors. Additionally after the '+' symbol we can have bias tensors that are added to the outputs.
    pub fn from_string(mut equation: String) -> Result<AxesMapping> {
        // Strip whitespace from the equation
        equation.retain(|c| !c.is_whitespace());
        let mut equation = equation.split("->");
        let inputs_side = equation
            .next()
            .ok_or(anyhow!("Invalid equation, no inputs"))?;
        let outputs_side = equation
            .next()
            .ok_or(anyhow!("Invalid equation, no outputs"))?;

        // Split the inputs part into its LHS and RHS components
        let input_parts = inputs_side
            .split(&['@', ':'])
            .enumerate()
            .map(parse_input_term)
            .collect::<Result<Vec<TensorDimInfo>>>()?;
        ensure!(
            input_parts.len() >= 2,
            "Invalid inputs, need at least one LHS and one RHS tensor"
        );

        // Collect the output identifiers and their dimensions, along with any biases
        let (output_dims, bias_dims) = parse_output_terms(outputs_side)?;
        // We need the unique characters to build the AxesMapping
        let unique_chars = input_parts
            .iter()
            .chain(output_dims.iter())
            .flat_map(|info| &info.dims)
            .copied()
            .collect::<HashSet<char>>();

        // Now we can build the AxesMapping
        let input_count = input_parts.len();
        let output_count = output_dims.len();
        let bias_count = bias_dims.len();

        // Iterate over the unique chars and build the axes
        let axes = unique_chars
            .into_iter()
            .map(|repr| {
                let mut lhs_input = Dimension::Absent;
                let mut rhs_inputs = vec![Dimension::Absent; output_count];
                let mut outputs = vec![Dimension::Absent; output_count];
                let mut biases = vec![Dimension::Absent; bias_count];

                let mut present_lhs = false;
                let mut present_rhs = false;
                let mut present_output = false;
                input_parts.iter().for_each(|info| {
                    if let Some(pos) = info.dims.iter().position(|&d| d == repr) {
                        if info.input_id == 0 {
                            lhs_input = Dimension::Present(pos);
                            present_lhs = true;
                        } else {
                            rhs_inputs[info.input_id - 1] = Dimension::Present(pos);
                            present_rhs = true;
                        }
                    }
                });

                output_dims.iter().for_each(|info| {
                    if let Some(pos) = info.dims.iter().position(|&d| d == repr) {
                        outputs[info.input_id] = Dimension::Present(pos);
                        present_output = true;
                    }
                });
                bias_dims.iter().for_each(|info| {
                    if let Some(pos) = info.dims.iter().position(|&d| d == repr) {
                        biases[info.input_id] = Dimension::Present(pos);
                    }
                });
                let axis_type = match (present_lhs, present_rhs, present_output) {
                    (true, true, true) => AxisType::Stacked,
                    (true, true, false) => AxisType::Contracted,
                    (true, false, true) | (false, true, true) => AxisType::Outer,
                    x => bail!(
                        "Axis must be present in at least one input and the output, got {x:?}"
                    ),
                };
                Ok(Axis {
                    lhs_input,
                    rhs_inputs,
                    outputs,
                    biases,
                    repr,
                    axis_type,
                })
            })
            .collect::<Result<Vec<Axis>>>()?;

        let mut mapping = AxesMapping {
            input_count,
            output_count,
            bias_count,
            axes,
        };
        // Sort the axes in the mapping based on their first occurrence in the input tensors
        mapping.sort();
        Ok(mapping)
    }

    /// Returns an iterator over the axes in the mapping.
    pub fn axes(&self) -> impl Iterator<Item = &Axis> {
        self.axes.iter()
    }

    /// Returns the number of input tensors in the operation.
    pub fn input_count(&self) -> usize {
        self.input_count
    }

    /// Returns the number of output tensors in the operation.
    pub fn output_count(&self) -> usize {
        self.output_count
    }

    /// Returns the number of bias tensors in the operation.
    pub fn bias_count(&self) -> usize {
        self.bias_count
    }

    /// Returns the output [Shapes](crate::Shape) of the operation given the input shapes.
    pub fn output_shapes(&self, input_shapes: &[Shape]) -> Result<Vec<Shape>> {
        ensure!(
            input_shapes.len() == self.input_count,
            "Mismatched number of input shapes, expected {}, got {}",
            self.input_count,
            input_shapes.len()
        );
        let mut output_shapes = vec![vec![]; self.output_count];

        for axis in &self.axes {
            for (input_id, dim) in axis.inputs().enumerate() {
                if let Dimension::Present(pos) = dim {
                    let dim_size = input_shapes[input_id].get(*pos).ok_or(anyhow!(
                        "Input tensor {} does not have dimension {} for axis {}",
                        input_id,
                        pos,
                        axis.repr
                    ))?;

                    for (output_id, out_dim) in axis.outputs.iter().enumerate() {
                        if let Dimension::Present(out_pos) = out_dim {
                            // Ensure the output shape is large enough
                            if output_shapes[output_id].len() <= *out_pos {
                                output_shapes[output_id].resize(*out_pos + 1, 0);
                            }
                            let out_dim_size = &mut output_shapes[output_id][*out_pos];
                            if *out_dim_size == 0 {
                                *out_dim_size = *dim_size;
                            } else {
                                ensure!(
                                    *out_dim_size == *dim_size,
                                    "Mismatched dimension sizes for axis {}: input tensor {} has size {}, output tensor {} has size {}",
                                    axis.repr,
                                    input_id,
                                    dim_size,
                                    output_id,
                                    out_dim_size
                                );
                            }
                        }
                    }
                }
            }
        }
        Ok(output_shapes.into_iter().map(Shape::new).collect())
    }

    /// Given the simplified [`Shape`] of a bias tensor (i.e. the non-broadcasted shape), this method computes the new [`Shape`] needed to add it to its corresponding output tensor via broadcasting.
    pub(crate) fn compute_new_bias_shape(
        &self,
        output_id: usize,
        bias_id: usize,
        bias_shape: &Shape,
    ) -> Result<Shape> {
        ensure!(
            bias_id < self.bias_count,
            "Invalid bias id {}, only {} biases in mapping",
            bias_id,
            self.bias_count
        );
        let mut new_shape = vec![];
        for axis in self.axes() {
            match (axis.outputs[output_id], axis.biases[bias_id]) {
                (Dimension::Present(output_pos), Dimension::Present(bias_pos)) => {
                    // This axis is present in both the output and the bias, so we take the size from the bias shape
                    let dim_size = bias_shape.get(bias_pos).ok_or(anyhow!(
                        "Bias tensor {} does not have dimension {} for axis {}",
                        bias_id,
                        bias_pos,
                        axis.repr
                    ))?;
                    let current_len = new_shape.len();
                    if current_len <= output_pos {
                        new_shape.resize(output_pos + 1, 1);
                    }
                    new_shape[output_pos] = *dim_size;
                }
                (Dimension::Present(output_pos), Dimension::Absent) => {
                    // This axis is present in the output but not the bias, so we set it to 1 for broadcasting
                    let current_len = new_shape.len();
                    if current_len <= output_pos {
                        new_shape.resize(output_pos + 1, 1);
                    }
                    new_shape[output_pos] = 1;
                }
                (Dimension::Absent, Dimension::Present(bias_pos)) => {
                    bail!(
                        "Bias tensor {} has dimension {} for axis {} that is not present in output tensor {}",
                        bias_id,
                        bias_pos,
                        axis.repr,
                        output_id
                    );
                }
                (Dimension::Absent, Dimension::Absent) => {
                    // This axis is not present in either the output or the bias, so we do nothing
                }
            }
        }

        Ok(Shape::new(new_shape))
    }

    /// Returns the sizes of the stacked, outer and contracted axes given the unpadded input shapes.
    /// The axes sizes can be recovered from the result by indexing using [`AxisType`].
    pub(crate) fn axes_sizes(&self, unpadded_input_shapes: &[Shape]) -> Result<[usize; 3]> {
        ensure!(
            unpadded_input_shapes.len() == self.input_count,
            "Mismatched number of input shapes, expected {}, got {}",
            self.input_count,
            unpadded_input_shapes.len()
        );
        let mut axes_size = [1usize; 3];

        for axis in &self.axes {
            let (input_id, pos) = axis
                .inputs()
                .enumerate()
                .find_map(|(input_id, dim)| {
                    if let Dimension::Present(pos) = dim {
                        Some((input_id, *pos))
                    } else {
                        None
                    }
                })
                .ok_or(anyhow!("No present dimension found for axis {}", axis.repr))?;
            let dim_size = unpadded_input_shapes[input_id].get(pos).ok_or(anyhow!(
                "Input tensor {} does not have dimension {} for axis {}",
                input_id,
                pos,
                axis.repr
            ))?;

            match axis.axis_type {
                AxisType::Stacked => axes_size[0] *= dim_size,
                AxisType::Outer => axes_size[1] *= dim_size,
                AxisType::Contracted => axes_size[2] *= dim_size,
            }
        }
        Ok(axes_size)
    }

    /// Returns the intermediate shapes used during evaluation given the input shapes.
    /// When evaluating an EinSum operation we always view the product as an action of one 3D tensor on another.
    /// This leads to an "intermediate shape" for each output tensor that is 3D and of the form [stacked_dims..., lhs_outer_dims..., rhs_outer_dims...].
    /// This method returns these intermediate shapes from the input shapes.
    pub fn intermediate_shapes(&self, input_shapes: &[Shape]) -> Result<Vec<Shape>> {
        let mut stack_dims = vec![];
        let mut lhs_outer_dims = vec![];
        let mut rhs_outer_dims = vec![vec![]; self.input_count - 1];

        ensure!(
            input_shapes.len() == self.input_count,
            "Mismatched number of input shapes, expected {}, got {}",
            self.input_count,
            input_shapes.len()
        );

        for axis in self.axes() {
            if let Dimension::Present(pos) = axis.lhs_input {
                let dim_size = input_shapes[0].get(pos).ok_or(anyhow!(
                    "Input tensor 0 does not have dimension {} for axis {}",
                    pos,
                    axis.repr
                ))?;
                match axis.axis_type {
                    AxisType::Stacked => stack_dims.push(*dim_size),
                    AxisType::Outer => {
                        lhs_outer_dims.push(*dim_size);
                    }
                    AxisType::Contracted => {}
                }
            }

            axis.rhs_inputs
                .iter()
                .enumerate()
                .try_for_each(|(input_id, dim)| match axis.axis_type {
                    AxisType::Stacked | AxisType::Contracted => Result::<()>::Ok(()),
                    AxisType::Outer => {
                        if let Dimension::Present(pos) = dim {
                            let dim_size = input_shapes[input_id + 1].get(*pos).ok_or(anyhow!(
                                "Input tensor {} does not have dimension {} for axis {}",
                                input_id + 1,
                                pos,
                                axis.repr
                            ))?;
                            rhs_outer_dims[input_id].push(*dim_size);
                            Ok(())
                        } else {
                            Ok(())
                        }
                    }
                })?;
        }

        let intermediate_shapes: Vec<Shape> = rhs_outer_dims
            .iter()
            .map(|rhs_outer_dims| {
                let mut intermediate_order = vec![];
                intermediate_order.extend(stack_dims.iter());
                intermediate_order.extend(lhs_outer_dims.iter());
                intermediate_order.extend(rhs_outer_dims.iter());
                Shape::new(intermediate_order.into_iter().copied().collect())
            })
            .collect();

        Ok(intermediate_shapes)
    }

    /// Given the LHS [`Tensor`], the RHS [Tensors][`Tensor`] and the claim points for each output tensor, this method returns the MLES
    /// with all variables fixed except those corresponding to the contraction axes.
    pub fn fix_axes<'a, E: ExtensionField>(
        &self,
        claim_points: &[Vec<&'a [E]>],
        full_inputs: &[Tensor<E>],
        unpadded_shapes: &[Shape],
    ) -> Result<FixedPolys<'a, E>> {
        ensure!(
            full_inputs.len() == self.input_count,
            "Mismatched number of input tensors, expected {}, got {}",
            self.input_count,
            full_inputs.len()
        );
        ensure!(
            unpadded_shapes.len() == self.input_count,
            "Mismatched number of unpadded shapes, expected {}, got {}",
            self.input_count,
            unpadded_shapes.len()
        );
        let fixed_axes = self.sort_variables_to_axes::<E>(claim_points)?;

        // We now fix the variables in the input tensors according to the fixed axes
        fixed_axes.into_fixed_polys(full_inputs, unpadded_shapes)
    }

    /// After being provided with the output points split into their respective axes, this method returns all the variables to fix in the input polynomials
    pub(crate) fn sort_variables_to_axes<'a, E: Clone>(
        &self,
        claim_points: &[Vec<&'a [E]>],
    ) -> Result<FixedAxesMapping<'a, E>> {
        ensure!(
            claim_points.len() == self.output_count,
            "Mismatched number of output points, expected {}, got {}",
            self.output_count,
            claim_points.len()
        );

        let mut lhs_fixes = vec![vec![]; self.output_count];
        let mut rhs_fixes = vec![vec![]; self.output_count];

        for axes in self.axes.iter() {
            for (output_id, out_dim) in axes.outputs.iter().enumerate() {
                if let Dimension::Present(out_pos) = out_dim {
                    let claim_point = claim_points[output_id][*out_pos];
                    let mut found_in_input = false;
                    // Find the position of this axis in the LHS or RHS input tensors
                    if let Dimension::Present(lhs_pos) = axes.lhs_input {
                        // This axis is present in the LHS tensor
                        let current_len = lhs_fixes[output_id].len();
                        if current_len <= lhs_pos {
                            lhs_fixes[output_id].resize(lhs_pos + 1, FixedAxis::Contracted);
                        }
                        lhs_fixes[output_id][lhs_pos] = FixedAxis::new(claim_point, axes.axis_type);
                        found_in_input = true;
                    }

                    if let Some(Dimension::Present(rhs_pos)) = axes.rhs_inputs.get(output_id) {
                        // This axis is present in its corresponding RHS tensor
                        let current_len = rhs_fixes[output_id].len();

                        if current_len <= *rhs_pos {
                            rhs_fixes[output_id].resize(*rhs_pos + 1, FixedAxis::Contracted);
                        }
                        rhs_fixes[output_id][*rhs_pos] =
                            FixedAxis::new(claim_point, axes.axis_type);
                        found_in_input = true;
                    }
                    // We have to find the axes in one of the inputs otherwise we can't fix it
                    ensure!(
                        found_in_input,
                        "Axis {} in output tensor {output_id} not found in any input tensor",
                        axes.repr,
                    );
                } else if let AxisType::Contracted = axes.axis_type {
                    // This axis is contracted and not present in the output, so we add a Contracted fix
                    match (axes.lhs_input, axes.rhs_inputs.get(output_id)) {
                        (Dimension::Present(lhs_pos), Some(Dimension::Present(rhs_pos))) => {
                            // This axis is present in both the LHS and its corresponding RHS tensor
                            let current_len = lhs_fixes[output_id].len();
                            if current_len <= lhs_pos {
                                lhs_fixes[output_id].resize(lhs_pos + 1, FixedAxis::Contracted);
                            }
                            lhs_fixes[output_id][lhs_pos] = FixedAxis::Contracted;
                            let current_len = rhs_fixes[output_id].len();
                            if current_len <= *rhs_pos {
                                rhs_fixes[output_id].resize(*rhs_pos + 1, FixedAxis::Contracted);
                            }
                            rhs_fixes[output_id][*rhs_pos] = FixedAxis::Contracted;
                        }
                        _ => bail!(
                            "Contracted axis {} not present in both LHS and RHS tensors for output tensor {}",
                            axes.repr,
                            output_id
                        ),
                    }
                }
            }
        }

        Ok(FixedAxesMapping {
            lhs_fixes,
            rhs_fixes,
        })
    }

    /// Method returns the point to evaluate the bias at given the output claim point.
    pub(crate) fn bias_evaluation_point<E: Clone>(
        &self,
        output_id: usize,
        bias_id: usize,
        claim_point: &[&[E]],
    ) -> Result<Vec<E>> {
        let mut point_slices = vec![];
        ensure!(
            output_id < self.output_count,
            "Invalid output id {}, only {} outputs in mapping",
            output_id,
            self.output_count
        );
        ensure!(
            bias_id < self.bias_count,
            "Invalid bias id {}, only {} biases in mapping",
            bias_id,
            self.bias_count
        );
        self.axes().try_for_each(|axis| {
            match (axis.outputs[output_id], axis.biases[bias_id]) {
                (Dimension::Present(output_pos), Dimension::Present(bias_pos)) => {
                    // This axis is present in both the output and the bias, so we take the value from the claim point
                    let claim_slice = claim_point[output_pos];
                    let current_len = point_slices.len();
                    if current_len <= bias_pos {
                        point_slices.resize(bias_pos + 1, None);
                    }
                    point_slices[bias_pos] = Some(claim_slice);
                }
                (Dimension::Absent, Dimension::Present(_)) => bail!(
                    "Bias tensor {} has an axis {} that is not present in output tensor {}",
                    bias_id,
                    axis.repr,
                    output_id
                ),
                (Dimension::Absent, Dimension::Absent)
                | (Dimension::Present(_), Dimension::Absent) => {}
            };
            Ok(())
        })?;

        point_slices
            .into_iter()
            .rev()
            .try_fold(Vec::new(), |mut acc, opt| {
                let unwrapped = opt.ok_or(anyhow!(
                    "Bias tensor {} does not have all its axes present in output tensor {}",
                    bias_id,
                    output_id
                ))?;
                acc.extend_from_slice(unwrapped);
                Ok(acc)
            })
    }

    /// Method used by the verifier to compute the correct bias broadcasted evaluation.
    /// By this we mean if the output is `O(ijkl)` then the bias tensor can have axes that are any subset of `(i,j,k,l)`.
    /// The correct evaluation is computed using a combination of the prover provided bias evaluation and less than checks for the broadcasted axes.
    ///
    /// If the output was again `O(ijkl)` and the bias was `BIAS(jl)` and the unpadded output shape was `(2, 3, 4, 5)` then to compute the correct evaluation
    /// from a point `r = (r_i, r_j, r_k, r_l)` and a prover provided bias evaluation `b = BIAS(r_j, r_l)` we would compute:
    /// `lt_poly(r_i, 2) * lt_poly(r_k, 4) * b` where `lt_poly(x, n)` is the less than polynomial that evaluates to 1 if `x < n` and 0 otherwise.
    ///
    /// # Arguments
    /// - `output_id`: The index of the output tensor the bias is being added to.
    /// - `bias_id`: The index of the bias tensor being evaluated.
    /// - `claim_point`: The point at which we have an evaluation claim for the output tensor.
    /// - `bias_eval`: The evaluation of the bias tensor at the point computed by the prover, if the output is `O(ijkl) + BIAS(kl)` and the claim point is `(i,j,k,l)` then this is `BIAS(k,l)`.
    /// - `output_shape`: The shape of the output tensor, used to compute less than checks for broadcasted axes.
    ///
    /// Returns a tuple of the broadcasted evaluation for the bias tensor and the corresponding claim on the unbroadcasted tensor.
    pub(crate) fn compute_bias_evaluation<E: ExtensionField>(
        &self,
        output_id: usize,
        bias_id: usize,
        claim_point: &[&[E]],
        bias_eval: E,
        output_shape: &Shape,
    ) -> Result<(E, Claim<E>)> {
        let (eval, claim_slices) =
            self.axes()
                .fold((bias_eval, vec![]), |(eval, mut acc), axis| {
                    match (axis.outputs[output_id], axis.biases[bias_id]) {
                        (Dimension::Present(output_pos), Dimension::Absent) => {
                            // This axis is present in the output but not the bias so we need to compute a less than check
                            let claim_slice = claim_point[output_pos];
                            let dim_size = output_shape[output_pos];
                            let dim_size_bits = to_bit_sequence_le(dim_size - 1, claim_slice.len())
                                .map(E::from_canonical_usize)
                                .collect::<Vec<E>>();
                            (eval * eval_zeroifier_mle(claim_slice, &dim_size_bits), acc)
                        }
                        (Dimension::Present(output_pos), Dimension::Present(bias_pos)) => {
                            // This axis is present in both the output and the bias, so we take the value from the claim point
                            let claim_slice = claim_point[output_pos];
                            let current_len = acc.len();
                            if current_len <= bias_pos {
                                acc.resize(bias_pos + 1, None);
                            }
                            acc[bias_pos] = Some(claim_slice);
                            (eval, acc)
                        }
                        _ => {
                            // This axis is either not present in both the output and the bias, or present in both so we do nothing
                            (eval, acc)
                        }
                    }
                });

        let point = claim_slices
            .into_iter()
            .rev()
            .try_fold(Vec::<E>::new(), |mut acc, opt| {
                let unwrapped = opt.ok_or(anyhow!(
                    "Bias tensor {} does not have all its axes present in output tensor {}",
                    bias_id,
                    output_id
                ))?;
                acc.extend_from_slice(unwrapped);
                Result::<Vec<E>>::Ok(acc)
            })?;
        Ok((eval, Claim::<E>::new(point, bias_eval)))
    }

    /// Sorts the axes in the mapping based on their first occurrence in the input tensors.
    pub fn sort(&mut self) {
        let order: Vec<(usize, usize, char)> = self
            .axes
            .iter()
            .flat_map(|axis| {
                // We iterate over the inputs only here, as during construction it was checked that the output dimensions
                // are a subset of the input dimensions.
                axis.inputs()
                    .enumerate()
                    .filter_map(|(i, dim)| {
                        if let Dimension::Present(pos) = dim {
                            Some((i, *pos, axis.repr))
                        } else {
                            None
                        }
                    })
                    .collect::<Vec<_>>()
            })
            .sorted()
            .dedup()
            .collect();

        self.axes
            .sort_by_key(|axis| order.iter().position(|tuple| tuple.2 == axis.repr).unwrap());
    }

    pub(crate) fn check_shapes(&self, input_shapes: &[Shape]) -> Result<()> {
        ensure!(
            input_shapes.len() == self.input_count,
            "Mismatched number of input shapes, expected {}, got {}",
            self.input_count,
            input_shapes.len()
        );
        self.axes.iter().try_for_each(|axis| {
            let dim_size: OnceLock<usize> = OnceLock::new();
            axis.inputs().enumerate().try_for_each(|(i, dim)| {
                if let Dimension::Present(pos) = dim {
                    let size = input_shapes[i].get(*pos).ok_or(anyhow!(
                        "Input tensor {} does not have dimension {} for axis {}",
                        i,
                        pos,
                        axis.repr
                    ))?;
                    let current_dim = dim_size.get_or_init(|| *size);
                    ensure!(
                        *current_dim == *size,
                        "Mismatched dimension sizes for axis {}: input tensor {} has size {}, previous size was {}",
                        axis.repr,
                        i,
                        size,
                        current_dim
                    );
                }
                Ok(())
            })
        })
    }
}

/// Parses an input term from the equation, returning the dimensions as a string slice.
fn parse_input_term((index, term): (usize, &str)) -> Result<TensorDimInfo> {
    let (term_identifier, term_dims) = term
        .find('(')
        .map(|pos| term.split_at(pos))
        .ok_or(anyhow!("Invalid tensor spec in LHS, no '('"))?;
    ensure!(
        term_identifier.chars().all(|c| c.is_ascii_uppercase()),
        "Invalid tensor identifier in LHS: {term_identifier}",
    );

    // Trim the parentheses from the dimensions
    let term_dims = term_dims.trim_matches(&['(', ')'][..]);
    // Check there are no repeated dimensions
    ensure!(
        term_dims.chars().all(|c| term_dims.matches(c).count() == 1),
        "Repeated dimensions in input tensor: {term_identifier}({term_dims})",
    );
    Ok(TensorDimInfo::new(index, term_dims.chars().collect()))
}

fn parse_output_terms(output_side: &str) -> Result<(Vec<TensorDimInfo>, Vec<TensorDimInfo>)> {
    let mut bias_count = 0usize;
    output_side.split(':').filter(|s| !s.is_empty()).enumerate().try_fold((vec![], vec![]), |(mut outputs, mut biases), (term_index, term)| {
        let mut output_chars = HashSet::<char>::new();
        term.split('+').enumerate().try_for_each(|(i, part)| {
            let (identifier, dims) = part
                .find('(')
                .map(|pos| part.split_at(pos))
                .ok_or(anyhow!("Invalid tensor spec in output, no '('"))?;
            // Trim the parentheses from the dimensions
                let dims = dims.trim_matches(&['(', ')'][..]);
                // Check there are no repeated dimensions
                ensure!(
                    dims.chars().all(|c| dims.matches(c).count() == 1),
                    "Repeated dimensions in tensor: {identifier}({dims})"
                );
            if i == 0 {
                ensure!(
            identifier.chars().all(|c| c.is_ascii_uppercase()),
            "Invalid tensor identifier in output: {identifier}",
        );
                output_chars = dims.chars().collect();
                outputs.push(TensorDimInfo::new(term_index, dims.chars().collect()));
                Ok(())
            } else {
                ensure!(
                identifier == "BIAS",
                "Invalid bias tensor identifier in output: {identifier}",
            );
            let bias_chars = dims.chars().collect::<HashSet<char>>();
            ensure!(
                bias_chars.is_subset(&output_chars),
                "Bias tensor dimensions must be a subset of the output tensor dimensions: {identifier}({dims})"
            );
                biases.push(TensorDimInfo::new(bias_count, dims.chars().collect()));
                bias_count += 1;
                Ok(())
            }
        })?;
        Ok((outputs, biases))
    })
}

#[derive(Debug, Clone, Copy)]
/// Enum used to represent different fixed axes types, allowing us to distinguish between axes that are fixed and those that are not.
pub enum FixedAxis<T> {
    /// Stacked axes are those that appear in both the LHS and RHS tensors, but the corresponding variables in their MLEs are not fixed.
    Stacked(T),
    /// Outer axes are those that appear in the LHS tensor or the RHS tensor, but not both. These variables are fixed for the Sumcheck.
    Outer(T),
    /// Contracted axes are those that have been contracted out during the operation, these are the axes that are summed over in the Sumcheck.
    Contracted,
}

impl<T> FixedAxis<T> {
    /// Creates a new [`FixedAxis`] from an optional slice of points and the [`AxisType`].
    pub fn new(point: T, axis_type: AxisType) -> Self {
        match axis_type {
            AxisType::Stacked => FixedAxis::Stacked(point),
            AxisType::Outer => FixedAxis::Outer(point),
            AxisType::Contracted => FixedAxis::Contracted,
        }
    }

    /// Maps the [`FixedAxis`] to an optional slice of points to fix the axis at.
    /// Returns `None` if the axis is contracted or unused.
    pub fn map<F, N>(&self, f: F) -> FixedAxis<N>
    where
        F: Fn(&T) -> N,
    {
        match self {
            FixedAxis::Stacked(point) => FixedAxis::Stacked(f(point)),
            FixedAxis::Outer(point) => FixedAxis::Outer(f(point)),
            FixedAxis::Contracted => FixedAxis::Contracted,
        }
    }
}

#[derive(Debug, Clone)]
/// Struct that is used to store the points to fix each axis at for each input tensor during proving.
/// The fixes that are performed using this struct are used to create the multilinear extensions that are passed to the Sumcheck protocol.
/// If variant is [`FixedAxis::Contracted`], then the axis is not fixed and is summed over in the Sumcheck, so we do nothing to it here.
/// If the variant is [`FixedAxis::Outer`], then the axis is fixed to the provided point.
/// If the variant is [`FixedAxis::Stacked`], then the axis is not fixed and instead the total number of MLEs produced is equal to the unpadded stacking axes size.
///
/// The fixes are given in the order of the axes in the inputs corresponding [`Shape`] without any permutation applied.
pub struct FixedAxesMapping<'a, E> {
    /// This is the points to fix each of the corresponding LHS axes at for each output tensor.
    /// The outer [`Vec`] corresponds to which output tensor, the inner [`Vec`] corresponds to which axis in the LHS tensor.
    pub(crate) lhs_fixes: Vec<Vec<FixedAxis<&'a [E]>>>,
    /// This is the points to fix each of the corresponding RHS axes at for each output tensor.
    /// The outer [`Vec`] corresponds to which RHS tensor, the inner [`Vec`] corresponds to which axis in that tensor.
    pub(crate) rhs_fixes: Vec<Vec<FixedAxis<&'a [E]>>>,
}

type FixedPolysResult<E> = (Vec<Vec<MultilinearExtension<'static, E>>>, Vec<Vec<E>>);

impl<'a, E: ExtensionField> FixedAxesMapping<'a, E> {
    /// Returns the LHS tensors after being fixed for each fix point and split along the stacking axes.
    /// That is if there are `s` stacking axes, the returned [`Vec`] will have length `s`.
    pub fn lhs_fixes(
        &self,
        lhs: &Tensor<E>,
        unpadded_shape: &Shape,
    ) -> Result<FixedPolysResult<E>> {
        // Transform the stacking/fixing points into eq_poly evals.
        let lhs_evals = self
            .lhs_fixes
            .iter()
            .map(|fixes| {
                fixes
                    .iter()
                    .zip(unpadded_shape.iter())
                    .map(|(dim_point, size)| {
                        dim_point.map(|point| compute_betas_eval(point)[..*size].to_vec())
                    })
                    .collect::<Vec<FixedAxis<Vec<E>>>>()
            })
            .collect::<Vec<Vec<FixedAxis<Vec<E>>>>>();

        // Work out the stacking coefficients for the LHS
        // these can be thought of as the coefficients of the unpadded heads of the LHS tensor.
        // Say we had stacking axes `i` and `j` of sizes 2 and 3 respectively, then the stacking coefficients would be
        // eq_i(0, ri) * eq_j(0, rj), eq_i(0, ri) * eq_j(1, rj), eq_i(0, ri) * eq_j(2, rj),
        // eq_i(1, ri) * eq_j(0, rj), eq_i(1, ri) * eq_j(1, rj), eq_i(1, ri) * eq_j(2, rj)
        // where ri and rj are the evaluation points for the axes i and j respectively.
        let stacking_coeffs = lhs_evals
            .iter()
            .map(|fixed_axes| {
                fixed_axes.iter().rev().fold(vec![E::ONE], |acc, axis| {
                    if let FixedAxis::Stacked(beta_evals) = axis {
                        beta_evals
                            .iter()
                            .flat_map(|&b| {
                                acc.par_iter()
                                    .with_min_len(64)
                                    .map(|&a| a * b)
                                    .collect::<Vec<E>>()
                            })
                            .collect()
                    } else {
                        acc
                    }
                })
            })
            .collect::<Vec<Vec<E>>>();
        // Now we can construct the multilinear extensions that have been fixed along the correct axes.
        let lhs_mles = lhs_evals
            .into_iter()
            .map(|evals| Self::mles_from_tensor(evals, lhs, unpadded_shape))
            .collect::<Result<Vec<_>>>()?;

        Ok((lhs_mles, stacking_coeffs))
    }

    /// Returns only the stacking coefficients, used by the verifier.
    pub fn stacking_coefficients(&self, unpadded_shape: &Shape) -> Vec<Vec<E>> {
        self.lhs_fixes
            .iter()
            .map(|fixed_axes| {
                fixed_axes.iter().zip(unpadded_shape.iter()).rev().fold(
                    vec![E::ONE],
                    |acc, (axis, size)| {
                        if let FixedAxis::Stacked(point) = axis {
                            compute_betas_eval(point)[..*size]
                                .to_vec()
                                .iter()
                                .flat_map(|&b| {
                                    acc.par_iter()
                                        .with_min_len(64)
                                        .map(|&a| a * b)
                                        .collect::<Vec<E>>()
                                })
                                .collect()
                        } else {
                            acc
                        }
                    },
                )
            })
            .collect::<Vec<Vec<E>>>()
    }

    /// Returns the RHS tensors after being fixed for each fix point and split along the stacking axes.
    /// That is if there are `r` RHS tensors and the stacking axis are of size `s`, the returned [`Vec`] will have length `r` and each inner [`Vec`] will have length `s`.
    pub fn rhs_fixes(
        &self,
        rhs: &[Tensor<E>],
        unpadded_shapes: &[Shape],
    ) -> Result<Vec<Vec<MultilinearExtension<'static, E>>>> {
        ensure!(
            rhs.len() == self.rhs_fixes.len(),
            "Mismatched number of RHS tensors, expected {}, got {}",
            self.rhs_fixes.len(),
            rhs.len()
        );
        ensure!(
            rhs.len() == unpadded_shapes.len(),
            "Mismatched number of RHS shapes, expected {}, got {}",
            self.rhs_fixes.len(),
            unpadded_shapes.len()
        );
        // Transform the stacking/fixing points into eq_poly evals.
        let rhs_evals = self
            .rhs_fixes
            .iter()
            .zip(unpadded_shapes.iter())
            .map(|(fixes, unpadded_shape)| {
                fixes
                    .iter()
                    .zip(unpadded_shape.iter())
                    .map(|(dim_point, size)| {
                        dim_point.map(|point| compute_betas_eval(point)[..*size].to_vec())
                    })
                    .collect::<Vec<FixedAxis<Vec<E>>>>()
            })
            .collect::<Vec<Vec<FixedAxis<Vec<E>>>>>();

        // Now we can construct the multilinear extensions that have been fixed along the correct axes.
        izip!(rhs_evals, rhs, unpadded_shapes)
            .map(|(rhs_evals, tensor, unpadded_shape)| {
                Self::mles_from_tensor(rhs_evals, tensor, unpadded_shape)
            })
            .collect::<Result<Vec<_>>>()
    }

    /// Method that constructs the multilinear extensions from a tensor after fixing the correct axes.
    fn mles_from_tensor(
        fixed_axes: Vec<FixedAxis<Vec<E>>>,
        tensor: &Tensor<E>,
        unpadded_shape: &Shape,
    ) -> Result<Vec<MultilinearExtension<'static, E>>> {
        // First we reduce the tensor to the unpadded shape
        let unpadded_data = tensor.reduce_to_shape(unpadded_shape)?.into_data();

        // Now we can construct the multilinear extensions that has been fixed at the correct locations
        let (evaluations, _, contraction_size) =
            fixed_axes.iter().zip(unpadded_shape.iter()).rev().fold(
                (unpadded_data, 1usize, 0usize),
                |(mut current_evals, chunk_size, contraction_acc), (fixed_axis, dim_size)| {
                    match fixed_axis {
                        FixedAxis::Outer(beta_evals) => {
                            // If this is an outer axis we need to fix these variables
                            if chunk_size == 1 {
                                // If the chunk size is 1 we can just do a direct dot product,
                                // this is because if chunk size is 1 then either every axis before this is fixed
                                // or this is the first axis we are looking at and so we can just do a direct dot product
                                // with the beta evaluations. SO we take chunks of size dim_size and do a dot product with the beta evaluations
                                // to reduce the evaluations down.
                                current_evals = current_evals
                                    .par_chunks(*dim_size)
                                    .with_min_len(64)
                                    .map(|chunk| {
                                        chunk.iter().zip(beta_evals).map(|(v, b)| *v * *b).sum()
                                    })
                                    .collect();
                            } else {
                                // In this case we have a chunk size greater than 1 so we need to be a bit more careful.
                                // The chunk_size is the product of all the AxisType::Contracted and AxisType::Stacked axes we have seen so far.
                                // So we can take chunks of size chunk_size * dim_size.
                                // For example if chunk_size = 12 and dim_size = 23 then we take chunks of size 276, then for each of the 12 sub-chunks
                                // of size 23 we take the first element of each sub-chunk and do a dot product with the beta evaluations, then the second element of each sub-chunk and so on.
                                // This gives us a new chunk of size chunk_size.
                                current_evals = current_evals
                                    .chunks(chunk_size * dim_size)
                                    .flat_map(|full_chunk| {
                                        (0..chunk_size)
                                            .into_par_iter()
                                            .with_min_len(64)
                                            .map(|i| {
                                                full_chunk
                                                    .iter()
                                                    .skip(i)
                                                    .step_by(chunk_size)
                                                    .zip(beta_evals)
                                                    .map(|(v, b)| *v * *b)
                                                    .sum()
                                            })
                                            .collect::<Vec<E>>()
                                    })
                                    .collect();
                            }
                            (current_evals, chunk_size, contraction_acc)
                        }
                        FixedAxis::Stacked(_) => {
                            // If this is a stacking axis we don't fix these variables
                            // so we update the chunk size but not the contraction size
                            let new_chunk_size = chunk_size * dim_size;
                            (current_evals, new_chunk_size, contraction_acc)
                        }
                        FixedAxis::Contracted => {
                            // If this is a contracted axis we don't fix these variables
                            // so we update the chunk size and the contraction size
                            let new_chunk_size = chunk_size * dim_size;
                            let new_contraction_acc = contraction_acc + dim_size;
                            (current_evals, new_chunk_size, new_contraction_acc)
                        }
                    }
                },
            );
        // The sumcheck is performed over the contraction axes and we have been working with unpadded data
        // so we need to pad the evaluations to the next power of two of the contraction axes size.
        let num_vars = ceil_log2(contraction_size);
        let diff = (1 << num_vars) - contraction_size;
        // `evaluations` represents the MLE of the full tensor (without padding) with all the outer axes fixed.
        // We now need to split this into multiple MLEs, with the total number of MLEs equal to the size of the combined stacking axes
        // and each individual MLE being over the contraction axes only.
        Ok(evaluations
            .chunks(contraction_size)
            .map(|eval_chunk| {
                MultilinearExtension::<E>::from_evaluations_ext_vec(
                    num_vars,
                    eval_chunk
                        .iter()
                        .copied()
                        .chain(std::iter::repeat_n(E::ZERO, diff))
                        .collect(),
                )
            })
            .collect::<Vec<_>>())
    }

    fn into_fixed_polys<'b>(
        self,
        tensors: &[Tensor<E>],
        unpadded_shapes: &[Shape],
    ) -> Result<FixedPolys<'b, E>>
    where
        E: ExtensionField,
        'a: 'b,
    {
        let (lhs, stacking_coeffs) = self.lhs_fixes(&tensors[0], &unpadded_shapes[0])?;
        let rhs = self.rhs_fixes(&tensors[1..], &unpadded_shapes[1..])?;

        let FixedAxesMapping {
            lhs_fixes,
            rhs_fixes,
        } = self;
        Ok(FixedPolys {
            lhs,
            rhs,
            lhs_points: lhs_fixes,
            rhs_points: rhs_fixes,
            stacking_coeffs,
        })
    }
}

/// Struct used to reduce type complexity when returning fixed polynomials from [`AxesMapping::fix_axes`]
pub struct FixedPolys<'a, E: ExtensionField> {
    /// The length of the outer [`Vec`] is equal to the total number of batched operations being performed.
    /// The the length of the inner [`Vec`] is equal to the total stacking dimension size of the LHS tensor.
    pub(crate) lhs: Vec<Vec<MultilinearExtension<'static, E>>>,
    /// The length of the outer [`Vec`] is equal to the number of RHS tensors.
    /// The length of the inner [`Vec`] is equal to the total stacking dimension size of mapping.
    pub(crate) rhs: Vec<Vec<MultilinearExtension<'static, E>>>,
    /// The points used to fix the axes in the above lhs tensors. The outer vec should have the same length as the outer vec in `lhs`.
    /// The inner vec should have the same length as the number of axes in the LHS tensor.
    pub(crate) lhs_points: Vec<Vec<FixedAxis<&'a [E]>>>,
    /// The points used to fix the axes in the above rhs tensors. The outer vec should have the same length as the outer vec in `rhs`.
    /// The inner vec should have the same length as the number of axes in the corresponding RHS tensor.
    pub(crate) rhs_points: Vec<Vec<FixedAxis<&'a [E]>>>,
    /// These are the stacking coefficients for the mapping, they correspond to eq poly evals for the stacking axes.
    pub(crate) stacking_coeffs: Vec<Vec<E>>,
}

#[cfg(test)]
mod tests {
    use anyhow::Context;
    use ark_std::rand::Rng;
    use ff_ext::{FromUniformBytes, GoldilocksExt2 as F};

    use crate::{Element, quantization::TensorFielder, rng_from_env_or_random};

    use super::*;

    #[test]
    fn test_axes_mapping_parsing() -> Result<()> {
        let mat_mul_test_case = AxesMappingParsingTestCase {
            equation: "A(ij)@B(jk)->C(ik)".to_string(),
            expected_input_count: 2,
            expected_output_count: 1,
            expected_axis_count: 3,
            expected_dim_order: vec!['i', 'j', 'k'],
        };

        let qkv_test_case = AxesMappingParsingTestCase {
            // Here "WQ", "WK", "WV" are the weights matrices for the queries, keys and values respectively
            // "X" is the input tensor, "Q", "K", "V" are the output tensors
            // "s" is the sequence length, "e" is the embedding dimension,
            // we the query weights has the extra dim "h" to represent the grouped query attention case where the query dimension
            // can be a multiple of the key/value dimension.
            equation: "X(se)@WQ(eha):WK(ea):WV(ea)->Q(has):K(sa):V(sa)".to_string(),
            expected_input_count: 4,
            expected_output_count: 3,
            expected_axis_count: 4,
            expected_dim_order: vec!['s', 'e', 'h', 'a'],
        };

        let transpose_test_case = AxesMappingParsingTestCase {
            equation: "A(ij)@B(kj)->C(ik)".to_string(),
            expected_input_count: 2,
            expected_output_count: 1,
            expected_axis_count: 3,
            expected_dim_order: vec!['i', 'j', 'k'],
        };

        let stacked_test_case = AxesMappingParsingTestCase {
            equation: "A(sijk)@B(sjl)->C(silk)".to_string(),
            expected_input_count: 2,
            expected_output_count: 1,
            expected_axis_count: 5,
            expected_dim_order: vec!['s', 'i', 'j', 'k', 'l'],
        };

        for test_case in [
            mat_mul_test_case,
            qkv_test_case,
            transpose_test_case,
            stacked_test_case,
        ] {
            test_axes_mapping_parsing_helper(test_case)?;
        }

        // We also test a few invalid cases to ensure they error as expected
        let invalid_equation = "A(iij)@B(jk)->C(iik)".to_string();
        let result = AxesMapping::from_string(invalid_equation);
        assert!(
            result.is_err(),
            "Expected error for invalid equation with repeated dimensions"
        );
        let invalid_equation = "A(ij)@B(jk)->C(ikr)".to_string();
        let result = AxesMapping::from_string(invalid_equation);
        assert!(
            result.is_err(),
            "Expected error for invalid equation with extra dimensions in output"
        );
        Ok(())
    }

    struct AxesMappingParsingTestCase {
        equation: String,
        expected_input_count: usize,
        expected_output_count: usize,
        expected_axis_count: usize,
        expected_dim_order: Vec<char>,
    }

    fn test_axes_mapping_parsing_helper(test_case: AxesMappingParsingTestCase) -> Result<()> {
        let AxesMappingParsingTestCase {
            equation,
            expected_input_count,
            expected_output_count,
            expected_axis_count,
            expected_dim_order,
        } = test_case;
        let mut axes_mapping = AxesMapping::from_string(equation.clone()).context(format!(
            "Failed to parse axes mapping from equation {equation}"
        ))?;
        assert_eq!(
            axes_mapping.input_count, expected_input_count,
            "Input count mismatch for equation: {equation}, expected {expected_input_count}, got {}",
            axes_mapping.input_count
        );
        assert_eq!(
            axes_mapping.output_count, expected_output_count,
            "Output count mismatch for equation: {equation}, expected {expected_output_count}, got {}",
            axes_mapping.output_count
        );
        assert_eq!(
            axes_mapping.axes.len(),
            expected_axis_count,
            "Axis count mismatch for equation: {equation}, expected {expected_axis_count}, got {}",
            axes_mapping.axes.len()
        );
        // Now we sort the mapping and check it returns the correct order for the axes
        axes_mapping.sort();
        let actual_dim_order: Vec<char> = axes_mapping.axes.iter().map(|axis| axis.repr).collect();
        for (i, (expected, actual)) in expected_dim_order
            .iter()
            .zip(actual_dim_order.iter())
            .enumerate()
        {
            assert_eq!(
                expected, actual,
                "Dimension order mismatch at position {i} for equation: {equation}, expected {expected}, got {actual}"
            );
        }
        Ok(())
    }

    #[test]
    fn test_output_shapes() -> Result<()> {
        let equation = "A(ij)@B(jk)->C(ik)".to_string();
        let axes_mapping = AxesMapping::from_string(equation.clone())?;
        let mut rng = rng_from_env_or_random();
        for _ in 0..10 {
            let i = rng.gen_range(1..10);
            let j = rng.gen_range(1..10);
            let k = rng.gen_range(1..10);
            let inputs = vec![Shape::new(vec![i, j]), Shape::new(vec![j, k])];
            let outputs = axes_mapping.output_shapes(&inputs)?;
            assert_eq!(
                outputs.len(),
                1,
                "Output count mismatch for equation: {equation}, expected 1, got {}",
                outputs.len()
            );
            assert_eq!(
                outputs[0],
                Shape::new(vec![i, k]),
                "Output shape mismatch for equation: {equation}, expected [{i}, {k}], got {:?}",
                outputs[0]
            );
        }
        Ok(())
    }

    #[test]
    fn test_broadcasted_bias_eval() {
        let equation = "A(ijk)@B(ikl)->C(ijl)+BIAS(il)".to_string();
        let axes_mapping = AxesMapping::from_string(equation).unwrap();
        let mut rng = rng_from_env_or_random();
        for _ in 0..10 {
            let i = rng.gen_range(1..10);
            let j = rng.gen_range(2..10);
            let k = rng.gen_range(1..10);
            let l = rng.gen_range(1..10);
            let inputs = vec![Shape::new(vec![i, j, k]), Shape::new(vec![i, k, l])];
            let outputs = axes_mapping.output_shapes(&inputs).unwrap();
            assert_eq!(outputs.len(), 1);
            let expected_output_shape = Shape::new(vec![i, j, l]);
            assert_eq!(outputs[0], expected_output_shape);
            let output_id = 0;
            let bias_id = 0;
            let padded_output_shape = expected_output_shape.next_power_of_two();
            let total_variables = ceil_log2(padded_output_shape.numel());

            // Make a random evaluation point for the bias tensor
            let point = (0..total_variables)
                .map(|_| F::random(&mut rng))
                .collect::<Vec<F>>();

            let bias_tensor = Tensor::<Element>::random(&Shape::new(vec![i, l]));
            // Make the unpadded broadcasted bias tensor
            let broadcasted_data = bias_tensor
                .get_data()
                .chunks(l)
                .flat_map(|row| row.iter().copied().cycle().take(j * l))
                .collect::<Vec<Element>>();
            let broadcasted_bias =
                Tensor::<Element>::new(expected_output_shape.clone(), broadcasted_data).unwrap();

            // Pad both tensors and convert to fields
            let bias_field: Tensor<F> = bias_tensor.pad_next_power_of_two().to_fields();
            let broadcasted_bias_field: Tensor<F> =
                broadcasted_bias.pad_next_power_of_two().to_fields();

            let broadcasted_mle = broadcasted_bias_field.to_mle();
            let broadcasted_eval = broadcasted_mle.evaluate(&point);

            let split_point = padded_output_shape.split_point(&point).unwrap();

            // Get the bias evaluation point from the axes mapping
            let bias_point = axes_mapping
                .bias_evaluation_point(output_id, bias_id, &split_point)
                .unwrap();
            let bias_mle = bias_field.to_mle();
            let bias_eval = bias_mle.evaluate(&bias_point);

            let computed_broadcasted_eval = axes_mapping
                .compute_bias_evaluation(
                    output_id,
                    bias_id,
                    &split_point,
                    bias_eval,
                    &expected_output_shape,
                )
                .unwrap()
                .0;

            // The computed bias eval should be equal to the original bias eval as there are no broadcasted axes
            assert_eq!(computed_broadcasted_eval, broadcasted_eval);
        }
    }
}
