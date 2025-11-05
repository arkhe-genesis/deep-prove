//! Wrapper type for burn's tensor.

pub use burn::tensor::{Shape as BShape, TensorKind as BTensorKind};
use serde::{Deserialize, Serialize};

use crate::{
    NextPowerOfTwo, Number,
    backend::{Backend, Conv2dConfig, Maxpool2dConfig, zkml_conv2d_i, zkml_max_pool2d_i},
};
use anyhow::{Context, Result, bail};
use burn::{
    module::Param,
    nn::{LayerNormConfig, RmsNormConfig},
    tensor::{
        BasicOps, BroadcastArgs, DimIter as BDimIter, Numeric, SliceArg, Tensor as BTensor,
        TensorData, activation,
        ops::{ConvOptions, IntTensorOps},
    },
};

use super::{Element, KeyedTensor, Shape, Tensor};

/// Burn tensor wrapper type with a dynamic rank
#[derive(Debug, Clone)]
pub enum WrappedTensor<T>
where
    T: TensorTypeParam,
{
    Rank1(BTensor<Backend, 1, T::Kind>, BShape),
    Rank2(BTensor<Backend, 2, T::Kind>, BShape),
    Rank3(BTensor<Backend, 3, T::Kind>, BShape),
    Rank4(BTensor<Backend, 4, T::Kind>, BShape),
}

pub enum DimIter<T>
where
    T: TensorTypeParam,
{
    Rank1(BDimIter<Backend, 1, T::Kind>, BShape),
    Rank2(BDimIter<Backend, 2, T::Kind>, BShape),
    Rank3(BDimIter<Backend, 3, T::Kind>, BShape),
    Rank4(BDimIter<Backend, 4, T::Kind>, BShape),
}

/// Delegate a `WrappedTensor` method to burn tensor method
macro_rules! delegate_plain {
    // Method with generic type param(s) given in parentheses before any fn args
    ($tensor: expr, $method: ident, ( $($type_arg: tt),* ), $($arg: expr),*) => {
        match $tensor {
            WrappedTensor::Rank1(tensor, ..) => tensor.$method::<$($type_arg),*>($($arg),*),
            WrappedTensor::Rank2(tensor, ..) => tensor.$method::<$($type_arg),*>($($arg),*),
            WrappedTensor::Rank3(tensor, ..) => tensor.$method::<$($type_arg),*>($($arg),*),
            WrappedTensor::Rank4(tensor, ..) => tensor.$method::<$($type_arg),*>($($arg),*),
        }
    };

    ($tensor: expr, $method: ident $(, $($arg: expr),* )?) => {
        match $tensor {
            WrappedTensor::Rank1(tensor, ..) => tensor.$method($($($arg),*)?),
            WrappedTensor::Rank2(tensor, ..) => tensor.$method($($($arg),*)?),
            WrappedTensor::Rank3(tensor, ..) => tensor.$method($($($arg),*)?),
            WrappedTensor::Rank4(tensor, ..) => tensor.$method($($($arg),*)?),
        }
    };
}

/// Delegate a `WrappedTensor` method to burn tensor method and re-wrap the
/// result.
///
/// Note that the rank output of the delegated method must match the rank of the
/// input.
macro_rules! delegate {
    ($tensor: expr, $method: ident $(, $($arg: expr),* )?) => {
        match $tensor {
            WrappedTensor::Rank1(tensor, bshape) => WrappedTensor::Rank1(tensor.$method($($($arg),*)?), bshape),
            WrappedTensor::Rank2(tensor, bshape) => WrappedTensor::Rank2(tensor.$method($($($arg),*)?), bshape),
            WrappedTensor::Rank3(tensor, bshape) => WrappedTensor::Rank3(tensor.$method($($($arg),*)?), bshape),
            WrappedTensor::Rank4(tensor, bshape) => WrappedTensor::Rank4(tensor.$method($($($arg),*)?), bshape),
        }
    };
}

/// Delegate a `WrappedTensor` method that also takes a wrapped tensor as an
/// arg(s) to burn tensor method and re-wrap the result.
///
/// Note that the rank of the arg(s) and output of the delegated method must
/// match the rank of the input.
macro_rules! delegate_with_arg {
    ($tensor: expr, $method: ident, $arg: expr) => {{
        let left_rank = $tensor.rank();
        let right_rank = $arg.rank();
        let out = match ($tensor, $arg) {
            (WrappedTensor::Rank1(tensor, bshape), WrappedTensor::Rank1(arg0, ..)) => {
                WrappedTensor::Rank1(tensor.$method(arg0), bshape)
            }
            (WrappedTensor::Rank2(tensor, bshape), WrappedTensor::Rank2(arg0, ..)) => {
                WrappedTensor::Rank2(tensor.$method(arg0), bshape)
            }
            (WrappedTensor::Rank3(tensor, bshape), WrappedTensor::Rank3(arg0, ..)) => {
                WrappedTensor::Rank3(tensor.$method(arg0), bshape)
            }
            (WrappedTensor::Rank4(tensor, bshape), WrappedTensor::Rank4(arg0, ..)) => {
                WrappedTensor::Rank4(tensor.$method(arg0), bshape)
            }
            _ => bail!("Unmatched input ranks. Left: {left_rank}, right: {right_rank}."),
        };
        Ok(out)
    }};
}

impl<T> WrappedTensor<T>
where
    T: TensorTypeParam,
{
    pub const fn rank(&self) -> usize {
        match self {
            Self::Rank1(..) => 1,
            Self::Rank2(..) => 2,
            Self::Rank3(..) => 3,
            Self::Rank4(..) => 4,
        }
    }

    pub fn unpadded_shape(&self) -> &BShape {
        match self {
            WrappedTensor::Rank1(_, shape) => shape,
            WrappedTensor::Rank2(_, shape) => shape,
            WrappedTensor::Rank3(_, shape) => shape,
            WrappedTensor::Rank4(_, shape) => shape,
        }
    }

    pub fn set_unpadded_shape(&mut self, new_shape: BShape) {
        match self {
            WrappedTensor::Rank1(_, shape)
            | WrappedTensor::Rank2(_, shape)
            | WrappedTensor::Rank3(_, shape)
            | WrappedTensor::Rank4(_, shape) => *shape = new_shape,
        }
    }

    /// Reshape the tensor to have the given shape.
    pub fn reshape(self, shape: burn::tensor::Shape) -> Result<WrappedTensor<T>> {
        let rank = shape.num_dims();
        let unpadded_shape = self.unpadded_shape().clone();

        let out = match rank {
            1 => WrappedTensor::Rank1(
                delegate_plain!(self, reshape, (1, _), shape),
                unpadded_shape,
            ),
            2 => WrappedTensor::Rank2(
                delegate_plain!(self, reshape, (2, _), shape),
                unpadded_shape,
            ),
            3 => WrappedTensor::Rank3(
                delegate_plain!(self, reshape, (3, _), shape),
                unpadded_shape,
            ),
            4 => WrappedTensor::Rank4(
                delegate_plain!(self, reshape, (4, _), shape),
                unpadded_shape,
            ),
            _ => bail!("Unexpected tensor rank: {rank}."),
        };
        Ok(out)
    }

    /// Converts the data of the current tensor.
    pub fn to_data(self) -> TensorData {
        delegate_plain!(self, to_data)
    }

    /// Returns a copy of the tensor data.
    #[cfg(test)]
    pub fn get_data(&self) -> Vec<T> {
        self.clone().to_data().into_vec().unwrap()
    }

    /// Returns the shape of the current tensor.
    pub fn shape(&self) -> BShape {
        delegate_plain!(self, shape)
    }

    pub fn cat(tensors: Vec<Self>, dim: usize) -> Result<Self> {
        let mut to_concat_r1 = Vec::with_capacity(tensors.len());
        let mut to_concat_r2 = Vec::with_capacity(tensors.len());
        let mut to_concat_r3 = Vec::with_capacity(tensors.len());
        let mut to_concat_r4 = Vec::with_capacity(tensors.len());
        let mut rank = None;
        // TODO: which unpadded shape to take here?
        let unpadded_shape = tensors[0].unpadded_shape().clone();

        for tensor in tensors.into_iter() {
            if let Some(rank) = rank {
                if tensor.rank() != rank {
                    bail!("Unmatched tensor ranks");
                }
            } else {
                rank = Some(tensor.rank());
            }
            match tensor {
                WrappedTensor::Rank1(tensor, ..) => to_concat_r1.push(tensor),
                WrappedTensor::Rank2(tensor, ..) => to_concat_r2.push(tensor),
                WrappedTensor::Rank3(tensor, ..) => to_concat_r3.push(tensor),
                WrappedTensor::Rank4(tensor, ..) => to_concat_r4.push(tensor),
            };
        }
        if let Some(rank) = rank {
            match rank {
                1 => {
                    let output = BTensor::<Backend, _, T::Kind>::cat(to_concat_r1, dim);
                    Ok(WrappedTensor::Rank1(output, unpadded_shape))
                }
                2 => {
                    let output = BTensor::<Backend, _, T::Kind>::cat(to_concat_r2, dim);
                    Ok(WrappedTensor::Rank2(output, unpadded_shape))
                }
                3 => {
                    let output = BTensor::<Backend, _, T::Kind>::cat(to_concat_r3, dim);
                    Ok(WrappedTensor::Rank3(output, unpadded_shape))
                }
                4 => {
                    let output = BTensor::<Backend, _, T::Kind>::cat(to_concat_r4, dim);
                    Ok(WrappedTensor::Rank4(output, unpadded_shape))
                }
                _ => unreachable!(),
            }
        } else {
            bail!("Cannot concat empty vec of tensors")
        }
    }

    /// Find the maximum absolute value.
    pub fn max_abs(self) -> Result<T> {
        let tensor = delegate_plain!(self, max_abs);
        Ok(tensor
            .into_data()
            .as_slice::<T>()
            .map_err(|e| anyhow::format_err!("{e:?}"))?[0])
    }

    /// Transpose the tensor.
    ///
    /// For a 2D tensor, this is the standard matrix transpose. For `D > 2`, the transpose is
    /// applied on the last two dimensions. For example, the transpose of a tensor with shape
    pub fn transpose(self) -> Self {
        delegate!(self, transpose)
    }

    /// Squeeze the tensor along the given dimension, removing the specified dimension
    /// of size one, and effectively reducing the rank of the tensor by one.
    pub fn squeeze(self, dim: usize) -> Result<Self> {
        let unpadded_shape = self.unpadded_shape().clone();
        let out = match self {
            WrappedTensor::Rank1(..) => bail!("Cannot squeeze 1D tensor"),
            WrappedTensor::Rank2(tensor, ..) => {
                WrappedTensor::Rank1(tensor.squeeze_dim(dim), unpadded_shape)
            }
            WrappedTensor::Rank3(tensor, ..) => {
                WrappedTensor::Rank2(tensor.squeeze_dim(dim), unpadded_shape)
            }
            WrappedTensor::Rank4(tensor, ..) => {
                WrappedTensor::Rank3(tensor.squeeze_dim(dim), unpadded_shape)
            }
        };
        Ok(out)
    }

    /// Applies element wise multiplication operation with a scalar.
    pub fn mul_scalar(self, other: T) -> Self {
        delegate!(self, mul_scalar, other)
    }

    /// Applies element wise addition operation with a scalar.
    pub fn add_scalar(self, other: T) -> Self {
        delegate!(self, add_scalar, other)
    }

    /// Applies element wise division operation with a scalar.
    pub fn div_scalar(self, other: T) -> Self {
        delegate!(self, div_scalar, other)
    }

    /// Clamp element wise over a minimum value.
    pub fn clamp_min(self, min: T) -> Self {
        delegate!(self, clamp_min, min)
    }

    /// Clamp element wise over a maximum value.
    pub fn clamp_max(self, max: T) -> Self {
        delegate!(self, clamp_max, max)
    }

    /// Aggregate all elements along the given dimension or axis in the tensor with the sum operation.
    pub fn sum_dim(self, dim: usize) -> Self {
        delegate!(self, sum_dim, dim)
    }

    /// Returns a new tensor with the same shape and device as the current tensor filled with the provided value.
    pub fn full_like(self, fill_value: T) -> Self {
        delegate!(self, full_like, fill_value)
    }

    /// Flatten the tensor along a given range of dimensions into 2 dimensions.
    pub fn flatten_to_dim_2(self, start_dim: usize, end_dim: usize) -> Self {
        let unpadded_shape = self.unpadded_shape().clone();

        let out = delegate_plain!(self, flatten, start_dim, end_dim);
        Self::Rank2(out, unpadded_shape)
    }

    ///  Find the maximum value along the given dimension.
    pub fn max_dim_with_indices(self, dim: usize) -> (Self, WrappedTensor<Element>) {
        match self {
            Self::Rank1(tensor, shape) => {
                let (maxes, indices) = tensor.max_dim_with_indices(dim);
                (
                    Self::Rank1(maxes, shape.clone()),
                    WrappedTensor::Rank1(indices, shape),
                )
            }
            Self::Rank2(tensor, shape) => {
                let (maxes, indices) = tensor.max_dim_with_indices(dim);
                (
                    Self::Rank2(maxes, shape.clone()),
                    WrappedTensor::Rank2(indices, shape.clone()),
                )
            }
            Self::Rank3(tensor, shape) => {
                let (maxes, indices) = tensor.max_dim_with_indices(dim);
                (
                    Self::Rank3(maxes, shape.clone()),
                    WrappedTensor::Rank3(indices, shape),
                )
            }
            Self::Rank4(tensor, shape) => {
                let (maxes, indices) = tensor.max_dim_with_indices(dim);
                (
                    Self::Rank4(maxes, shape.clone()),
                    WrappedTensor::Rank4(indices, shape),
                )
            }
        }
    }

    /// Unsqueeze the current tensor. Create new leading dimensions to fit 2 dimensions.
    pub fn unsqueeze_dim_2(self) -> Self {
        let unpadded_shape = self.unpadded_shape().clone();
        let result = delegate_plain!(self, unsqueeze, (2),);
        WrappedTensor::Rank2(result, unpadded_shape)
    }

    /// Unsqueeze the current tensor. Create new leading dimensions to fit 3 dimensions.
    pub fn unsqueeze_dim_3(self) -> Self {
        let unpadded_shape = self.unpadded_shape().clone();
        let result = delegate_plain!(self, unsqueeze, (3),);
        WrappedTensor::Rank3(result, unpadded_shape)
    }

    /// Unsqueeze the current tensor. Create new leading dimensions to fit 4 dimensions.
    pub fn unsqueeze_dim_4(self) -> Self {
        let unpadded_shape = self.unpadded_shape().clone();
        let result = delegate_plain!(self, unsqueeze, (4),);
        WrappedTensor::Rank4(result, unpadded_shape)
    }

    /// Creates a new tensor with a dimension of size one inserted at the specified position.
    pub fn unsqueeze_dim(self, dim: usize) -> Result<Self> {
        let out = match self {
            WrappedTensor::Rank1(tensor, shape) => {
                WrappedTensor::Rank2(tensor.unsqueeze_dim(dim), shape)
            }
            WrappedTensor::Rank2(tensor, shape) => {
                WrappedTensor::Rank3(tensor.unsqueeze_dim(dim), shape)
            }
            WrappedTensor::Rank3(tensor, shape) => {
                WrappedTensor::Rank4(tensor.unsqueeze_dim(dim), shape)
            }
            WrappedTensor::Rank4(..) => bail!("Cannot unsqueeze 4D tensor"),
        };
        Ok(out)
    }

    /// Iterate over slices of tensors alongside a given dimension.
    pub fn iter_dim(self, dim: usize) -> DimIter<T> {
        match self {
            Self::Rank1(tensor, unpadded_shape) => {
                DimIter::Rank1(tensor.iter_dim(dim), unpadded_shape)
            }
            Self::Rank2(tensor, unpadded_shape) => {
                DimIter::Rank2(tensor.iter_dim(dim), unpadded_shape)
            }
            Self::Rank3(tensor, unpadded_shape) => {
                DimIter::Rank3(tensor.iter_dim(dim), unpadded_shape)
            }
            Self::Rank4(tensor, unpadded_shape) => {
                DimIter::Rank4(tensor.iter_dim(dim), unpadded_shape)
            }
        }
    }

    /// Permute the dimensions of the tensor.
    pub fn permute(self, axes: &[isize]) -> Result<Self> {
        let out = match self {
            Self::Rank1(tensor, unpadded_shape) => {
                let axes: [isize; 1] = TryFrom::try_from(axes).context(format!(
                    "Unexpected permutation axes length. Expected 1, got {}",
                    axes.len(),
                ))?;
                Self::Rank1(tensor.permute(axes), unpadded_shape)
            }
            Self::Rank2(tensor, unpadded_shape) => {
                let axes: [isize; 2] = TryFrom::try_from(axes).context(format!(
                    "Unexpected permutation axes length. Expected 2, got {}",
                    axes.len(),
                ))?;
                Self::Rank2(tensor.permute(axes), unpadded_shape)
            }
            Self::Rank3(tensor, unpadded_shape) => {
                let axes: [isize; 3] = TryFrom::try_from(axes).context(format!(
                    "Unexpected permutation axes length. Expected 3, got {}",
                    axes.len(),
                ))?;
                Self::Rank3(tensor.permute(axes), unpadded_shape)
            }
            Self::Rank4(tensor, unpadded_shape) => {
                let axes: [isize; 4] = TryFrom::try_from(axes).context(format!(
                    "Unexpected permutation axes length. Expected 4, got {}",
                    axes.len(),
                ))?;
                Self::Rank4(tensor.permute(axes), unpadded_shape)
            }
        };
        Ok(out)
    }

    /// Returns a tensor containing the elements selected from the given ranges.
    pub fn slice<const R2: usize, R: SliceArg<R2>>(self, ranges: R) -> Self {
        match self {
            Self::Rank1(tensor, unpadded_shape) => {
                Self::Rank1(tensor.slice(ranges), unpadded_shape)
            }
            Self::Rank2(tensor, unpadded_shape) => {
                Self::Rank2(tensor.slice(ranges), unpadded_shape)
            }
            Self::Rank3(tensor, unpadded_shape) => {
                Self::Rank3(tensor.slice(ranges), unpadded_shape)
            }
            Self::Rank4(tensor, unpadded_shape) => {
                Self::Rank4(tensor.slice(ranges), unpadded_shape)
            }
        }
    }

    /// Returns the size of the given dimension.
    ///
    /// When given a negative `dim` indexes from the back.
    pub fn dim(&self, dim: isize) -> Result<usize> {
        match self {
            WrappedTensor::Rank1(tensor, ..) => {
                let dim = if dim < 0 {
                    (tensor.dims().len() as isize + dim) as usize
                } else {
                    dim as usize
                };
                tensor
                    .dims()
                    .get(dim)
                    .context("Dimension out of bounds")
                    .copied()
            }
            WrappedTensor::Rank2(tensor, ..) => {
                let dim = if dim < 0 {
                    (tensor.dims().len() as isize + dim) as usize
                } else {
                    dim as usize
                };
                tensor
                    .dims()
                    .get(dim)
                    .context("Dimension out of bounds")
                    .copied()
            }
            WrappedTensor::Rank3(tensor, ..) => {
                let dim = if dim < 0 {
                    (tensor.dims().len() as isize + dim) as usize
                } else {
                    dim as usize
                };
                tensor
                    .dims()
                    .get(dim)
                    .context("Dimension out of bounds")
                    .copied()
            }
            WrappedTensor::Rank4(tensor, ..) => {
                let dim = if dim < 0 {
                    (tensor.dims().len() as isize + dim) as usize
                } else {
                    dim as usize
                };
                tensor
                    .dims()
                    .get(dim)
                    .context("Dimension out of bounds")
                    .copied()
            }
        }
    }

    /// Flatten the tensor into 1D shape.
    pub fn flatten_1d(self) -> Self {
        let end_dim = self.rank() - 1;
        let unpadded_shape = self.unpadded_shape().clone();
        WrappedTensor::Rank1(delegate_plain!(self, flatten, 0, end_dim), unpadded_shape)
    }

    /// Converts the tensor into a primitive tensor.
    pub fn into_primitive(self) -> <T::Kind as BTensorKind<Backend>>::Primitive {
        delegate_plain!(self, into_primitive)
    }

    /// Broadcast the tensor to the given shape.
    pub fn expand<const D: usize, S: BroadcastArgs<D, D>>(self, shape: S) -> Result<Self> {
        let shape = shape.into_shape(&self.shape());
        let unpadded_shape = self.unpadded_shape().clone();

        let rank = shape.num_dims();
        let out = match rank {
            1 => WrappedTensor::Rank1(delegate_plain!(self, expand, (1, _), shape), unpadded_shape),
            2 => WrappedTensor::Rank2(delegate_plain!(self, expand, (2, _), shape), unpadded_shape),
            3 => WrappedTensor::Rank3(delegate_plain!(self, expand, (3, _), shape), unpadded_shape),
            4 => WrappedTensor::Rank4(delegate_plain!(self, expand, (4, _), shape), unpadded_shape),
            _ => bail!("Unexpected tensor rank: {rank}."),
        };
        Ok(out)
    }

    /// Copies the sub-slice `shape` from this tensor.
    ///
    /// Returns a new [Tensor] with shape `shape` initialised from `self`.
    pub fn reduce_to_shape(self, shape: &BShape) -> Result<Self> {
        let out = match shape.num_dims() {
            1 => self.slice(shape.dims::<1>().map(|i| 0..i)),
            2 => self.slice(shape.dims::<2>().map(|i| 0..i)),
            3 => self.slice(shape.dims::<3>().map(|i| 0..i)),
            4 => self.slice(shape.dims::<4>().map(|i| 0..i)),
            rank => bail!("Unexpected shape rank: {rank}."),
        };
        Ok(out)
    }

    /// Update the given tensor with the value where the mask is true.
    pub fn mask_fill_4d(
        self,
        mask: BTensor<Backend, 4, burn::tensor::Bool>,
        value: T,
    ) -> Result<Self> {
        let input_rank = self.rank();
        let Self::Rank4(input, unpadded_shape) = self else {
            bail!("Unexpected input rank: {input_rank}, expected 4.")
        };
        Ok(WrappedTensor::Rank4(
            input.mask_fill(mask, value),
            unpadded_shape,
        ))
    }

    /// Flatten the tensor into 1D.
    pub fn to_flatten(self) -> Self {
        match self {
            Self::Rank1(tensor, unpadded_shape) => {
                Self::Rank1(tensor.flatten(0, 0), unpadded_shape)
            }
            Self::Rank2(tensor, unpadded_shape) => {
                Self::Rank1(tensor.flatten(0, 1), unpadded_shape)
            }
            Self::Rank3(tensor, unpadded_shape) => {
                Self::Rank1(tensor.flatten(0, 2), unpadded_shape)
            }
            Self::Rank4(tensor, unpadded_shape) => {
                Self::Rank1(tensor.flatten(0, 3), unpadded_shape)
            }
        }
    }

    /// Pads the tensor to the next power-of-two.
    pub fn pad_next_power_of_two(self) -> Self {
        let BShape { dims } = self.shape();
        let shape = BShape {
            dims: dims.next_power_of_two(),
        };
        match self {
            WrappedTensor::Rank1(tensor, unpadded_shape) => {
                #[allow(clippy::single_range_in_vec_init)]
                let ranges = [0..dims[0]];
                let out = BTensor::full(shape, T::zero(), &Default::default());
                let out = out.slice_assign(ranges, tensor);
                WrappedTensor::Rank1(out, unpadded_shape)
            }
            WrappedTensor::Rank2(tensor, unpadded_shape) => {
                let ranges = [0..dims[0], 0..dims[1]];
                let out = BTensor::full(shape, T::zero(), &Default::default());
                let out = out.slice_assign(ranges, tensor);
                WrappedTensor::Rank2(out, unpadded_shape)
            }
            WrappedTensor::Rank3(tensor, unpadded_shape) => {
                let ranges = [0..dims[0], 0..dims[1], 0..dims[2]];
                let out = BTensor::full(shape, T::zero(), &Default::default());
                let out = out.slice_assign(ranges, tensor);
                WrappedTensor::Rank3(out, unpadded_shape)
            }
            WrappedTensor::Rank4(tensor, unpadded_shape) => {
                let ranges = [0..dims[0], 0..dims[1], 0..dims[2], 0..dims[3]];
                let out = BTensor::full(shape, T::zero(), &Default::default());
                let out = out.slice_assign(ranges, tensor);
                WrappedTensor::Rank4(out, unpadded_shape)
            }
        }
    }

    pub fn random(shape: &Shape) -> Self {
        Self::try_from(&Tensor::random(shape)).unwrap()
    }

    #[cfg(test)]
    pub fn pad_1d(self, new_len: usize) -> Self {
        let input_len = self.shape().num_elements();
        let Self::Rank1(input, unpadded_shape) = self else {
            panic!("pad_1d only works for 1d tensors, e.g. vectors")
        };
        let out = BTensor::full(BShape::from(vec![new_len]), T::zero(), &Default::default());
        #[allow(clippy::single_range_in_vec_init)]
        let out = out.slice_assign([0..input_len], input);
        WrappedTensor::Rank1(out, unpadded_shape)
    }

    #[cfg(test)]
    pub fn into_native(self) -> Tensor<T> {
        Tensor::try_from(self).unwrap()
    }

    #[cfg(test)]
    pub fn to_native(&self) -> Tensor<T> {
        Tensor::try_from(self.clone()).unwrap()
    }
}

impl<T> WrappedTensor<T>
where
    T: TensorTypeParam,
    <T as TensorTypeParam>::Kind: Numeric<Backend>,
{
    /// Performs the + operation
    #[allow(clippy::should_implement_trait)]
    pub fn add(self, other: Self) -> Result<Self> {
        delegate_with_arg!(self, add, other)
    }

    /// Performs the - operation
    #[allow(clippy::should_implement_trait)]
    pub fn sub(self, other: Self) -> Result<Self> {
        delegate_with_arg!(self, sub, other)
    }

    /// Applies the matrix multiplication operation.
    pub fn matmul(self, other: Self) -> Result<Self> {
        delegate_with_arg!(self, matmul, other)
    }

    /// Applies element wise multiplication operation.
    #[allow(clippy::should_implement_trait)]
    pub fn mul(self, other: Self) -> Result<Self> {
        delegate_with_arg!(self, mul, other)
    }

    /// Switch sign of each element in the tensor.
    #[allow(clippy::should_implement_trait)]
    pub fn neg(self) -> Self {
        delegate!(self, neg)
    }
}

impl WrappedTensor<f32> {
    /// Applies element wise root square operation.
    pub fn sqrt(self) -> Self {
        delegate!(self, sqrt)
    }

    /// Applies reciprocal operation (or multiplicative inverse) element wise.
    pub fn recip(self) -> Self {
        delegate!(self, recip)
    }

    /// Applies element wise round operation.
    pub fn round(self) -> Self {
        delegate!(self, round)
    }

    /// Returns a new tensor with the same shape and device as the current tensor and the data cast to Integer.
    pub fn int(self) -> WrappedTensor<Element> {
        delegate!(self, int)
    }

    /// Applies the Gaussian Error Linear Units function as described in the paper
    /// [Gaussian Error Linear Units (GELUs)](https://arxiv.org/pdf/1606.08415v3.pdf).
    pub fn gelu(input: Self) -> Self {
        match input {
            WrappedTensor::Rank1(input, unpadded_shape) => {
                WrappedTensor::Rank1(burn::tensor::activation::gelu(input), unpadded_shape)
            }
            WrappedTensor::Rank2(input, unpadded_shape) => {
                WrappedTensor::Rank2(burn::tensor::activation::gelu(input), unpadded_shape)
            }
            WrappedTensor::Rank3(input, unpadded_shape) => {
                WrappedTensor::Rank3(burn::tensor::activation::gelu(input), unpadded_shape)
            }
            WrappedTensor::Rank4(input, unpadded_shape) => {
                WrappedTensor::Rank4(burn::tensor::activation::gelu(input), unpadded_shape)
            }
        }
    }

    pub fn conv2d(
        input: Self,
        weight: Self,
        bias: Option<Self>,
        options: ConvOptions<2>,
    ) -> Result<Self> {
        let input_rank = input.rank();
        let Self::Rank4(input, unpadded_shape_input) = input else {
            bail!("Unexpected input rank: {input_rank}, expected 4.")
        };
        let weight_rank = weight.rank();
        let Self::Rank4(weight, ..) = weight else {
            bail!("Unexpected weight rank: {weight_rank}, expected 4.")
        };
        let bias = match bias {
            Some(bias) => {
                let bias_rank = bias.rank();
                let Self::Rank1(bias, ..) = bias else {
                    bail!("Unexpected bias rank: {bias_rank}, expected 1.")
                };
                Some(bias)
            }
            None => None,
        };
        let out = burn::tensor::module::conv2d(input, weight, bias, options);
        Ok(WrappedTensor::Rank4(out, unpadded_shape_input))
    }

    pub fn max_pool2d(
        input: Self,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        dilation: [usize; 2],
    ) -> Result<Self> {
        let input_rank = input.rank();
        let Self::Rank4(input, unpadded_shape) = input else {
            bail!("Unexpected input rank: {input_rank}, expected 4.")
        };
        let result =
            burn::tensor::module::max_pool2d(input, kernel_size, stride, padding, dilation);
        Ok(WrappedTensor::Rank4(result, unpadded_shape))
    }

    pub fn layer_norm(
        input: Self,
        embedding_size: usize,
        epsilon: f64,
        gamma: Self,
        beta: Self,
    ) -> Result<Self> {
        // NOTE: simply use the burn tensor API for now as we want to move towards using more burn features
        // instead of re-implementing everything ourselves.
        // copy implementation https://docs.rs/burn-core/0.17.0/src/burn_core/nn/norm/layer.rs.html#67
        let input_rank = input.rank();
        let Self::Rank2(input, unpadded_shape) = input else {
            bail!("Unexpected input rank: {input_rank}, expected 2.")
        };
        let gamma_rank = gamma.rank();
        let Self::Rank1(gamma, ..) = gamma else {
            bail!("Unexpected gamma rank: {gamma_rank}, expected 1.")
        };
        let beta_rank = beta.rank();
        let Self::Rank1(beta, ..) = beta else {
            bail!("Unexpected beta rank: {beta_rank}, expected 1.")
        };
        let config = LayerNormConfig::new(embedding_size).with_epsilon(epsilon);
        let device = Default::default();
        let mut norm = config.init(&device);
        norm.gamma = Param::from_tensor(gamma);
        norm.beta = Param::from_tensor(beta);
        let output = norm.forward(input);
        Ok(Self::Rank2(output, unpadded_shape))
    }

    pub fn softmax(tensor: Self, dim: usize) -> Result<Self> {
        let tensor_rank = tensor.rank();
        let Self::Rank2(tensor, unpadded_shape) = tensor else {
            bail!("Unexpected tensor rank: {tensor_rank}, expected 2.")
        };
        Ok(Self::Rank2(
            activation::softmax(tensor, dim),
            unpadded_shape,
        ))
    }

    pub fn rms_norm_forward(
        input: Self,
        embedding_size: usize,
        epsilon: f64,
        gamma: Self,
    ) -> Result<Self> {
        // NOTE: simply use the burn tensor API for now as we want to move towards using more burn features
        // instead of re-implementing everything ourselves.
        // copy implementation https://docs.rs/burn-core/0.17.0/src/burn_core/nn/norm/rms.rs.html#71
        let input_rank = input.rank();
        let Self::Rank2(input, unpadded_shape) = input else {
            bail!("Unexpected input rank: {input_rank}, expected 2.")
        };
        let gamma_rank = gamma.rank();
        let Self::Rank1(gamma, ..) = gamma else {
            bail!("Unexpected gamma rank: {gamma_rank}, expected 1.")
        };
        let config = RmsNormConfig::new(embedding_size).with_epsilon(epsilon);
        let device = Default::default();
        let mut norm = config.init(&device);
        norm.gamma = Param::from_tensor(gamma);
        let output = norm.forward(input);
        Ok(Self::Rank2(output, unpadded_shape))
    }
}

impl WrappedTensor<Element> {
    /// Applies the bitwise right shift operation with the scalar.
    pub fn bitwise_right_shift_scalar(self, other: Element) -> Self {
        delegate!(self, bitwise_right_shift_scalar, other)
    }

    /// Applies the bitwise left shift operation with the scalar.
    pub fn bitwise_left_shift_scalar(self, other: Element) -> Self {
        delegate!(self, bitwise_left_shift_scalar, other)
    }

    /// Convert the element tensor into a float tensor.
    pub fn float(self) -> WrappedTensor<f32> {
        delegate!(self, float)
    }

    pub fn conv2d(x: Self, weight: Self, bias: Self, options: Conv2dConfig) -> Result<Self> {
        let x_rank = x.rank();
        let Self::Rank4(x, unpadded_shape) = x else {
            bail!("Unexpected x rank: {x_rank}, expected 4.")
        };
        let weight_rank = weight.rank();
        let Self::Rank4(weight, ..) = weight else {
            bail!("Unexpected weight rank: {weight_rank}, expected 4.")
        };
        let bias_rank = bias.rank();
        let Self::Rank1(bias, ..) = bias else {
            bail!("Unexpected bias rank: {bias_rank}, expected 1.")
        };
        let out = zkml_conv2d_i(x, weight, bias, options)?;
        Ok(WrappedTensor::Rank4(out, unpadded_shape))
    }

    pub fn max_pool2d(input: Self, config: Maxpool2dConfig) -> Result<Self> {
        let input_rank = input.rank();
        let Self::Rank4(input, unpadded_shape) = input else {
            bail!("Unexpected input rank: {input_rank}, expected 4.")
        };
        Ok(WrappedTensor::Rank4(
            zkml_max_pool2d_i(input, config)?,
            unpadded_shape,
        ))
    }
}

pub trait WrappedModuleFn {
    fn linear(input: Self, weight: Self, bias: Option<Self>) -> Result<Self>
    where
        Self: Sized;

    /// Applies the rectified linear unit function element-wise
    /// as described in the paper [Deep Learning using Rectified Linear Units (ReLU)](https://arxiv.org/pdf/1803.08375).
    fn relu(input: Self) -> Self;
}

impl WrappedModuleFn for WrappedTensor<f32> {
    fn linear(input: Self, weight: Self, bias: Option<Self>) -> Result<Self> {
        let weight_rank = weight.rank();
        let Self::Rank2(weight, ..) = weight else {
            bail!("Unexpected weight rank: {weight_rank}, expected 2.")
        };
        let bias = match bias {
            Some(bias) => {
                let bias_rank = bias.rank();
                let Self::Rank1(bias, ..) = bias else {
                    bail!("Unexpected bias rank: {bias_rank}, expected 1.")
                };
                Some(bias)
            }
            None => None,
        };
        let out = match input {
            WrappedTensor::Rank1(input, unpadded_shape) => WrappedTensor::Rank1(
                burn::tensor::module::linear(input, weight, bias),
                unpadded_shape,
            ),
            WrappedTensor::Rank2(input, unpadded_shape) => WrappedTensor::Rank2(
                burn::tensor::module::linear(input, weight, bias),
                unpadded_shape,
            ),
            WrappedTensor::Rank3(input, unpadded_shape) => WrappedTensor::Rank3(
                burn::tensor::module::linear(input, weight, bias),
                unpadded_shape,
            ),
            WrappedTensor::Rank4(input, unpadded_shape) => WrappedTensor::Rank4(
                burn::tensor::module::linear(input, weight, bias),
                unpadded_shape,
            ),
        };
        Ok(out)
    }

    /// Applies the rectified linear unit function element-wise
    /// as described in the paper [Deep Learning using Rectified Linear Units (ReLU)](https://arxiv.org/pdf/1803.08375).
    fn relu(input: Self) -> Self {
        match input {
            WrappedTensor::Rank1(input, unpadded_shape) => {
                WrappedTensor::Rank1(burn::tensor::activation::relu(input), unpadded_shape)
            }
            WrappedTensor::Rank2(input, unpadded_shape) => {
                WrappedTensor::Rank2(burn::tensor::activation::relu(input), unpadded_shape)
            }
            WrappedTensor::Rank3(input, unpadded_shape) => {
                WrappedTensor::Rank3(burn::tensor::activation::relu(input), unpadded_shape)
            }
            WrappedTensor::Rank4(input, unpadded_shape) => {
                WrappedTensor::Rank4(burn::tensor::activation::relu(input), unpadded_shape)
            }
        }
    }
}

impl WrappedModuleFn for WrappedTensor<Element> {
    fn linear(input: Self, weight: Self, bias: Option<Self>) -> Result<Self> {
        let input = input.unsqueeze_dim(1)?;
        let matmul = weight.transpose().matmul(input)?;
        let matmul = matmul.squeeze(1)?;
        let out = if let Some(bias) = bias {
            matmul.add(bias)?
        } else {
            matmul
        };
        Ok(out)
    }

    fn relu(input: Self) -> Self {
        match input {
            WrappedTensor::Rank1(input, unpadded_shape) => {
                let input = input.into_primitive();
                let mask = Backend::int_lower_equal_elem(input.clone(), 0);
                let out = Backend::int_mask_fill(input, mask, 0);
                let out = BTensor::from_primitive(out);
                WrappedTensor::Rank1(out, unpadded_shape)
            }
            WrappedTensor::Rank2(input, unpadded_shape) => {
                let input = input.into_primitive();
                let mask = Backend::int_lower_equal_elem(input.clone(), 0);
                let out = Backend::int_mask_fill(input, mask, 0);
                let out = BTensor::from_primitive(out);
                WrappedTensor::Rank2(out, unpadded_shape)
            }
            WrappedTensor::Rank3(input, unpadded_shape) => {
                let input = input.into_primitive();
                let mask = Backend::int_lower_equal_elem(input.clone(), 0);
                let out = Backend::int_mask_fill(input, mask, 0);
                let out = BTensor::from_primitive(out);
                WrappedTensor::Rank3(out, unpadded_shape)
            }
            WrappedTensor::Rank4(input, unpadded_shape) => {
                let input = input.into_primitive();
                let mask = Backend::int_lower_equal_elem(input.clone(), 0);
                let out = Backend::int_mask_fill(input, mask, 0);
                let out = BTensor::from_primitive(out);
                WrappedTensor::Rank4(out, unpadded_shape)
            }
        }
    }
}

pub trait Conversion {
    fn float(self) -> WrappedTensor<f32>;
}

impl<T: TensorTypeParam> Conversion for WrappedTensor<T> {
    fn float(self) -> WrappedTensor<f32> {
        T::tensor_to_float(self)
    }
}

pub trait IntoBTensor {
    type Kind: BTensorKind<Backend> + BasicOps<Backend> + Numeric<Backend>;

    fn to_btensor<const D: usize>(&self) -> BTensor<Backend, D, Self::Kind>;
}

impl<T> IntoBTensor for Tensor<T>
where
    T: TensorTypeParam,
{
    type Kind = <T as TensorTypeParam>::Kind;

    fn to_btensor<const D: usize>(&self) -> BTensor<Backend, D, Self::Kind> {
        let shape = self.shape().clone();
        BTensor::from_data(
            TensorData::new(self.data.clone(), shape),
            &Default::default(),
        )
    }
}

/// Tensor parameter type ([`f32`] or [`Element`]/[`i64`])
pub trait TensorTypeParam:
    burn::tensor::Element + Number + PartialEq + Serialize + for<'de> Deserialize<'de>
{
    /// Burn TensorKind
    type Kind: BTensorKind<Backend> + BasicOps<Backend> + Numeric<Backend>;

    fn tensor_to_float(tensor: WrappedTensor<Self>) -> WrappedTensor<f32>;

    /// Convert tensor into a wrapper burn tensor
    fn wrap(tensor: &Tensor<Self>) -> Result<WrappedTensor<Self>>
    where
        Self: Sized,
    {
        let rank = tensor.rank();
        let unpadded_shape = tensor.unpadded_shape().clone().into();

        let out = match rank {
            1 => {
                let input = tensor.clone().to_btensor::<1>();
                WrappedTensor::Rank1(input, unpadded_shape)
            }
            2 => {
                let input = tensor.clone().to_btensor::<2>();
                WrappedTensor::Rank2(input, unpadded_shape)
            }
            3 => {
                let input = tensor.clone().to_btensor::<3>();
                WrappedTensor::Rank3(input, unpadded_shape)
            }
            4 => {
                let input = tensor.clone().to_btensor::<4>();
                WrappedTensor::Rank4(input, unpadded_shape)
            }
            _ => {
                bail!("Unexpected tensor rank: {rank}")
            }
        };
        Ok(out)
    }
}

impl TensorTypeParam for f32 {
    type Kind = burn::tensor::Float;

    fn tensor_to_float(tensor: WrappedTensor<Self>) -> WrappedTensor<f32> {
        tensor
    }
}

impl TensorTypeParam for Element {
    type Kind = burn::tensor::Int;

    fn tensor_to_float(tensor: WrappedTensor<Self>) -> WrappedTensor<f32> {
        tensor.float()
    }
}

impl<T> TryFrom<&Tensor<T>> for WrappedTensor<T>
where
    T: TensorTypeParam,
{
    type Error = anyhow::Error;

    fn try_from(value: &Tensor<T>) -> Result<Self> {
        <T as TensorTypeParam>::wrap(value)
    }
}

impl<T> TryFrom<WrappedTensor<T>> for Tensor<T>
where
    T: TensorTypeParam,
{
    type Error = anyhow::Error;

    fn try_from(tensor: WrappedTensor<T>) -> Result<Self, Self::Error> {
        let shape = tensor.shape().into();
        let data = tensor
            .to_data()
            .into_vec()
            .map_err(|e| anyhow::format_err!("{e:?}"))?;
        Tensor::<T>::new(shape, data)
    }
}

impl<T> TryFrom<&KeyedTensor<T>> for WrappedTensor<T>
where
    T: TensorTypeParam,
{
    type Error = anyhow::Error;

    fn try_from(value: &KeyedTensor<T>) -> Result<Self> {
        <T as TensorTypeParam>::wrap(&value.tensor)
    }
}

impl<T> Iterator for DimIter<T>
where
    T: TensorTypeParam,
{
    type Item = WrappedTensor<T>;

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            DimIter::Rank1(iter, unpadded_shape) => iter
                .next()
                .map(|i| WrappedTensor::Rank1(i, unpadded_shape.clone())),
            DimIter::Rank2(iter, unpadded_shape) => iter
                .next()
                .map(|i| WrappedTensor::Rank2(i, unpadded_shape.clone())),
            DimIter::Rank3(iter, unpadded_shape) => iter
                .next()
                .map(|i| WrappedTensor::Rank3(i, unpadded_shape.clone())),
            DimIter::Rank4(iter, unpadded_shape) => iter
                .next()
                .map(|i| WrappedTensor::Rank4(i, unpadded_shape.clone())),
        }
    }
}

impl<T> Serialize for WrappedTensor<T>
where
    T: TensorTypeParam,
{
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        Tensor::try_from(self.clone())
            .map_err(serde::ser::Error::custom)?
            .serialize(serializer)
    }
}

impl<'de, T> Deserialize<'de> for WrappedTensor<T>
where
    T: TensorTypeParam,
{
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let tensor: Tensor<T> = Tensor::deserialize(deserializer)?;
        WrappedTensor::try_from(&tensor).map_err(serde::de::Error::custom)
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_tensor_next_pow_of_two() {
        let shape = Shape::new(vec![1, 1, 1, 1]);
        let tensor = Tensor::new(shape.clone(), vec![1]).unwrap().into_wrapped();
        assert_eq!(
            tensor.clone().pad_next_power_of_two().to_native(),
            tensor.to_native(),
            "Tensor should not change if the shape is already power of two",
        );

        let shape = Shape::new(vec![2, 2]);
        let tensor = Tensor::new(shape.clone(), vec![1, 2, 1, 2])
            .unwrap()
            .into_wrapped();
        assert_eq!(
            tensor.clone().pad_next_power_of_two().to_native(),
            tensor.to_native(),
            "Tensor should not change if the shape is already power of two",
        );

        let shape = Shape::new(vec![4, 4]);
        let tensor = WrappedTensor::<Element>::random(&shape.clone());
        assert_eq!(
            tensor.clone().pad_next_power_of_two().to_native(),
            tensor.to_native(),
            "Tensor should not change if the shape is already power of two",
        );

        let shape = Shape::new(vec![3, 3]);
        let tensor = Tensor::new(shape.clone(), vec![1, 2, 3, 1, 2, 3, 1, 2, 3])
            .unwrap()
            .into_wrapped();
        let new_tensor = tensor.pad_next_power_of_two();
        assert_eq!(
            Shape::from(new_tensor.shape()),
            Shape::new(vec![4, 4]),
            "Tensor padding to next power of two failed."
        );
        assert_eq!(
            &new_tensor.get_data(),
            &[1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 0, 0, 0, 0],
            "Tensor padding to next power of two failed."
        );

        let shape = Shape::new(vec![3, 2]);
        let tensor = Tensor::new(shape.clone(), vec![1, 2, 1, 2, 1, 2])
            .unwrap()
            .into_wrapped();
        let new_tensor = tensor.pad_next_power_of_two();
        assert_eq!(
            Shape::from(new_tensor.shape()),
            Shape::new(vec![4, 2]),
            "Tensor padding to next power of two failed."
        );
        assert_eq!(
            &new_tensor.get_data(),
            &[1, 2, 1, 2, 1, 2, 0, 0],
            "Tensor padding to next power of two failed."
        );

        let shape = Shape::new(vec![2, 3, 3]);
        let tensor = Tensor::new(
            shape.clone(),
            vec![
                1, 1, 1, 2, 2, 2, 3, 3, 3, 11, 11, 11, 12, 12, 12, 13, 13, 13,
            ],
        )
        .unwrap()
        .into_wrapped();
        let new_tensor = tensor.pad_next_power_of_two();
        assert_eq!(
            Shape::from(new_tensor.shape()),
            Shape::new(vec![2, 4, 4]),
            "Tensor padding to next power of two failed."
        );
        assert_eq!(
            &new_tensor.get_data(),
            &[
                1, 1, 1, 0, 2, 2, 2, 0, 3, 3, 3, 0, 0, 0, 0, 0, 11, 11, 11, 0, 12, 12, 12, 0, 13,
                13, 13, 0, 0, 0, 0, 0,
            ],
            "Tensor padding to next power of two failed."
        );
    }
}
