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
    Rank1(BTensor<Backend, 1, T::Kind>),
    Rank2(BTensor<Backend, 2, T::Kind>),
    Rank3(BTensor<Backend, 3, T::Kind>),
    Rank4(BTensor<Backend, 4, T::Kind>),
}

pub enum DimIter<T>
where
    T: TensorTypeParam,
{
    Rank1(BDimIter<Backend, 1, T::Kind>),
    Rank2(BDimIter<Backend, 2, T::Kind>),
    Rank3(BDimIter<Backend, 3, T::Kind>),
    Rank4(BDimIter<Backend, 4, T::Kind>),
}

/// Delegate a `WrappedTensor` method to burn tensor method
macro_rules! delegate_plain {
    // Method with generic type param(s) given in parentheses before any fn args
    ($tensor: expr, $method: ident, ( $($type_arg: tt),* ), $($arg: expr),*) => {
        match $tensor {
            WrappedTensor::Rank1(tensor) => tensor.$method::<$($type_arg),*>($($arg),*),
            WrappedTensor::Rank2(tensor) => tensor.$method::<$($type_arg),*>($($arg),*),
            WrappedTensor::Rank3(tensor) => tensor.$method::<$($type_arg),*>($($arg),*),
            WrappedTensor::Rank4(tensor) => tensor.$method::<$($type_arg),*>($($arg),*),
        }
    };

    ($tensor: expr, $method: ident $(, $($arg: expr),* )?) => {
        match $tensor {
            WrappedTensor::Rank1(tensor) => tensor.$method($($($arg),*)?),
            WrappedTensor::Rank2(tensor) => tensor.$method($($($arg),*)?),
            WrappedTensor::Rank3(tensor) => tensor.$method($($($arg),*)?),
            WrappedTensor::Rank4(tensor) => tensor.$method($($($arg),*)?),
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
            WrappedTensor::Rank1(tensor) => WrappedTensor::Rank1(tensor.$method($($($arg),*)?)),
            WrappedTensor::Rank2(tensor) => WrappedTensor::Rank2(tensor.$method($($($arg),*)?)),
            WrappedTensor::Rank3(tensor) => WrappedTensor::Rank3(tensor.$method($($($arg),*)?)),
            WrappedTensor::Rank4(tensor) => WrappedTensor::Rank4(tensor.$method($($($arg),*)?)),
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
            (WrappedTensor::Rank1(tensor), WrappedTensor::Rank1(arg0)) => {
                WrappedTensor::Rank1(tensor.$method(arg0))
            }
            (WrappedTensor::Rank2(tensor), WrappedTensor::Rank2(arg0)) => {
                WrappedTensor::Rank2(tensor.$method(arg0))
            }
            (WrappedTensor::Rank3(tensor), WrappedTensor::Rank3(arg0)) => {
                WrappedTensor::Rank3(tensor.$method(arg0))
            }
            (WrappedTensor::Rank4(tensor), WrappedTensor::Rank4(arg0)) => {
                WrappedTensor::Rank4(tensor.$method(arg0))
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
            Self::Rank1(_) => 1,
            Self::Rank2(_) => 2,
            Self::Rank3(_) => 3,
            Self::Rank4(_) => 4,
        }
    }

    /// Reshape the tensor to have the given shape.
    pub fn reshape(self, shape: burn::tensor::Shape) -> Result<WrappedTensor<T>> {
        let rank = shape.num_dims();
        let out = match rank {
            1 => WrappedTensor::Rank1(delegate_plain!(self, reshape, (1, _), shape)),
            2 => WrappedTensor::Rank2(delegate_plain!(self, reshape, (2, _), shape)),
            3 => WrappedTensor::Rank3(delegate_plain!(self, reshape, (3, _), shape)),
            4 => WrappedTensor::Rank4(delegate_plain!(self, reshape, (4, _), shape)),
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
        for tensor in tensors.into_iter() {
            if let Some(rank) = rank {
                if tensor.rank() != rank {
                    bail!("Unmatched tensor ranks");
                }
            } else {
                rank = Some(tensor.rank());
            }
            match tensor {
                WrappedTensor::Rank1(tensor) => to_concat_r1.push(tensor),
                WrappedTensor::Rank2(tensor) => to_concat_r2.push(tensor),
                WrappedTensor::Rank3(tensor) => to_concat_r3.push(tensor),
                WrappedTensor::Rank4(tensor) => to_concat_r4.push(tensor),
            };
        }
        if let Some(rank) = rank {
            match rank {
                1 => {
                    let output = BTensor::<Backend, _, T::Kind>::cat(to_concat_r1, dim);
                    Ok(WrappedTensor::Rank1(output))
                }
                2 => {
                    let output = BTensor::<Backend, _, T::Kind>::cat(to_concat_r2, dim);
                    Ok(WrappedTensor::Rank2(output))
                }
                3 => {
                    let output = BTensor::<Backend, _, T::Kind>::cat(to_concat_r3, dim);
                    Ok(WrappedTensor::Rank3(output))
                }
                4 => {
                    let output = BTensor::<Backend, _, T::Kind>::cat(to_concat_r4, dim);
                    Ok(WrappedTensor::Rank4(output))
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
        let out = match self {
            WrappedTensor::Rank1(_) => bail!("Cannot squeeze 1D tensor"),
            WrappedTensor::Rank2(tensor) => WrappedTensor::Rank1(tensor.squeeze_dim(dim)),
            WrappedTensor::Rank3(tensor) => WrappedTensor::Rank2(tensor.squeeze_dim(dim)),
            WrappedTensor::Rank4(tensor) => WrappedTensor::Rank3(tensor.squeeze_dim(dim)),
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

    /// Clamp element wise over a mimimum value.
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
        let out = delegate_plain!(self, flatten, start_dim, end_dim);
        Self::Rank2(out)
    }

    ///  Find the maximum value along the given dimension.
    pub fn max_dim_with_indices(self, dim: usize) -> (Self, WrappedTensor<Element>) {
        match self {
            Self::Rank1(tensor) => {
                let (maxes, indices) = tensor.max_dim_with_indices(dim);
                (Self::Rank1(maxes), WrappedTensor::Rank1(indices))
            }
            Self::Rank2(tensor) => {
                let (maxes, indices) = tensor.max_dim_with_indices(dim);
                (Self::Rank2(maxes), WrappedTensor::Rank2(indices))
            }
            Self::Rank3(tensor) => {
                let (maxes, indices) = tensor.max_dim_with_indices(dim);
                (Self::Rank3(maxes), WrappedTensor::Rank3(indices))
            }
            Self::Rank4(tensor) => {
                let (maxes, indices) = tensor.max_dim_with_indices(dim);
                (Self::Rank4(maxes), WrappedTensor::Rank4(indices))
            }
        }
    }

    /// Unsqueeze the current tensor. Create new leading dimensions to fit 2 dimensions.
    pub fn unsqueeze_dim_2(self) -> Self {
        let result = delegate_plain!(self, unsqueeze, (2),);
        WrappedTensor::Rank2(result)
    }

    /// Unsqueeze the current tensor. Create new leading dimensions to fit 3 dimensions.
    pub fn unsqueeze_dim_3(self) -> Self {
        let result = delegate_plain!(self, unsqueeze, (3),);
        WrappedTensor::Rank3(result)
    }

    /// Unsqueeze the current tensor. Create new leading dimensions to fit 4 dimensions.
    pub fn unsqueeze_dim_4(self) -> Self {
        let result = delegate_plain!(self, unsqueeze, (4),);
        WrappedTensor::Rank4(result)
    }

    /// Creates a new tensor with a dimension of size one inserted at the specified position.
    pub fn unsqueeze_dim(self, dim: usize) -> Result<Self> {
        let out = match self {
            WrappedTensor::Rank1(tensor) => WrappedTensor::Rank2(tensor.unsqueeze_dim(dim)),
            WrappedTensor::Rank2(tensor) => WrappedTensor::Rank3(tensor.unsqueeze_dim(dim)),
            WrappedTensor::Rank3(tensor) => WrappedTensor::Rank4(tensor.unsqueeze_dim(dim)),
            WrappedTensor::Rank4(_) => bail!("Cannot unsqueeze 4D tensor"),
        };
        Ok(out)
    }

    /// Iterate over slices of tensors alongside a given dimension.
    pub fn iter_dim(self, dim: usize) -> DimIter<T> {
        match self {
            Self::Rank1(tensor) => DimIter::Rank1(tensor.iter_dim(dim)),
            Self::Rank2(tensor) => DimIter::Rank2(tensor.iter_dim(dim)),
            Self::Rank3(tensor) => DimIter::Rank3(tensor.iter_dim(dim)),
            Self::Rank4(tensor) => DimIter::Rank4(tensor.iter_dim(dim)),
        }
    }

    /// Permute the dimensions of the tensor.
    pub fn permute(self, axes: &[isize]) -> Result<Self> {
        let out = match self {
            Self::Rank1(tensor) => {
                let axes: [isize; 1] = TryFrom::try_from(axes).context(format!(
                    "Unexpected permutation axes length. Expected 1, got {}",
                    axes.len(),
                ))?;
                Self::Rank1(tensor.permute(axes))
            }
            Self::Rank2(tensor) => {
                let axes: [isize; 2] = TryFrom::try_from(axes).context(format!(
                    "Unexpected permutation axes length. Expected 2, got {}",
                    axes.len(),
                ))?;
                Self::Rank2(tensor.permute(axes))
            }
            Self::Rank3(tensor) => {
                let axes: [isize; 3] = TryFrom::try_from(axes).context(format!(
                    "Unexpected permutation axes length. Expected 3, got {}",
                    axes.len(),
                ))?;
                Self::Rank3(tensor.permute(axes))
            }
            Self::Rank4(tensor) => {
                let axes: [isize; 4] = TryFrom::try_from(axes).context(format!(
                    "Unexpected permutation axes length. Expected 4, got {}",
                    axes.len(),
                ))?;
                Self::Rank4(tensor.permute(axes))
            }
        };
        Ok(out)
    }

    /// Returns a tensor containing the elements selected from the given ranges.
    pub fn slice<const R2: usize, R: SliceArg<R2>>(self, ranges: R) -> Self {
        match self {
            Self::Rank1(tensor) => Self::Rank1(tensor.slice(ranges)),
            Self::Rank2(tensor) => Self::Rank2(tensor.slice(ranges)),
            Self::Rank3(tensor) => Self::Rank3(tensor.slice(ranges)),
            Self::Rank4(tensor) => Self::Rank4(tensor.slice(ranges)),
        }
    }

    /// Returns the size of the given dimension.
    ///
    /// When given a negative `dim` indexes from the back.
    pub fn dim(&self, dim: isize) -> Result<usize> {
        match self {
            WrappedTensor::Rank1(tensor) => {
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
            WrappedTensor::Rank2(tensor) => {
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
            WrappedTensor::Rank3(tensor) => {
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
            WrappedTensor::Rank4(tensor) => {
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
        WrappedTensor::Rank1(delegate_plain!(self, flatten, 0, end_dim))
    }

    /// Converts the tensor into a primitive tensor.
    pub fn into_primitive(self) -> <T::Kind as BTensorKind<Backend>>::Primitive {
        delegate_plain!(self, into_primitive)
    }

    /// Broadcast the tensor to the given shape.
    pub fn expand<const D: usize, S: BroadcastArgs<D, D>>(self, shape: S) -> Result<Self> {
        let shape = shape.into_shape(&self.shape());
        let rank = shape.num_dims();
        let out = match rank {
            1 => WrappedTensor::Rank1(delegate_plain!(self, expand, (1, _), shape)),
            2 => WrappedTensor::Rank2(delegate_plain!(self, expand, (2, _), shape)),
            3 => WrappedTensor::Rank3(delegate_plain!(self, expand, (3, _), shape)),
            4 => WrappedTensor::Rank4(delegate_plain!(self, expand, (4, _), shape)),
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
        let Self::Rank4(input) = self else {
            bail!("Unexpected input rank: {input_rank}, expected 4.")
        };
        Ok(WrappedTensor::Rank4(input.mask_fill(mask, value)))
    }

    /// Flatten the tensor into 1D.
    pub fn to_flatten(self) -> Self {
        match self {
            Self::Rank1(tensor) => Self::Rank1(tensor.flatten(0, 0)),
            Self::Rank2(tensor) => Self::Rank1(tensor.flatten(0, 1)),
            Self::Rank3(tensor) => Self::Rank1(tensor.flatten(0, 2)),
            Self::Rank4(tensor) => Self::Rank1(tensor.flatten(0, 3)),
        }
    }

    /// Pads the tensor to the next power-of-two.
    pub fn pad_next_power_of_two(self) -> Self {
        let BShape { dims } = self.shape();
        let shape = BShape {
            dims: dims.next_power_of_two(),
        };
        match self {
            WrappedTensor::Rank1(tensor) => {
                #[allow(clippy::single_range_in_vec_init)]
                let ranges = [0..dims[0]];
                let out = BTensor::full(shape, T::zero(), &Default::default());
                let out = out.slice_assign(ranges, tensor);
                WrappedTensor::Rank1(out)
            }
            WrappedTensor::Rank2(tensor) => {
                let ranges = [0..dims[0], 0..dims[1]];
                let out = BTensor::full(shape, T::zero(), &Default::default());
                let out = out.slice_assign(ranges, tensor);
                WrappedTensor::Rank2(out)
            }
            WrappedTensor::Rank3(tensor) => {
                let ranges = [0..dims[0], 0..dims[1], 0..dims[2]];
                let out = BTensor::full(shape, T::zero(), &Default::default());
                let out = out.slice_assign(ranges, tensor);
                WrappedTensor::Rank3(out)
            }
            WrappedTensor::Rank4(tensor) => {
                let ranges = [0..dims[0], 0..dims[1], 0..dims[2], 0..dims[3]];
                let out = BTensor::full(shape, T::zero(), &Default::default());
                let out = out.slice_assign(ranges, tensor);
                WrappedTensor::Rank4(out)
            }
        }
    }

    pub fn random(shape: &Shape) -> Self {
        Self::try_from(&Tensor::random(shape)).unwrap()
    }

    #[cfg(test)]
    pub fn pad_1d(self, new_len: usize) -> Self {
        let input_len = self.shape().num_elements();
        let Self::Rank1(input) = self else {
            panic!("pad_1d only works for 1d tensors, e.g. vectors")
        };
        let out = BTensor::full(BShape::from(vec![new_len]), T::zero(), &Default::default());
        #[allow(clippy::single_range_in_vec_init)]
        let out = out.slice_assign([0..input_len], input);
        WrappedTensor::Rank1(out)
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
            WrappedTensor::Rank1(input) => {
                WrappedTensor::Rank1(burn::tensor::activation::gelu(input))
            }
            WrappedTensor::Rank2(input) => {
                WrappedTensor::Rank2(burn::tensor::activation::gelu(input))
            }
            WrappedTensor::Rank3(input) => {
                WrappedTensor::Rank3(burn::tensor::activation::gelu(input))
            }
            WrappedTensor::Rank4(input) => {
                WrappedTensor::Rank4(burn::tensor::activation::gelu(input))
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
        let Self::Rank4(input) = input else {
            bail!("Unexpected input rank: {input_rank}, expected 4.")
        };
        let weight_rank = weight.rank();
        let Self::Rank4(weight) = weight else {
            bail!("Unexpected weight rank: {weight_rank}, expected 4.")
        };
        let bias = match bias {
            Some(bias) => {
                let bias_rank = bias.rank();
                let Self::Rank1(bias) = bias else {
                    bail!("Unexpected bias rank: {bias_rank}, expected 1.")
                };
                Some(bias)
            }
            None => None,
        };
        let out = burn::tensor::module::conv2d(input, weight, bias, options);
        Ok(WrappedTensor::Rank4(out))
    }

    pub fn max_pool2d(
        input: Self,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        dilation: [usize; 2],
    ) -> Result<Self> {
        let input_rank = input.rank();
        let Self::Rank4(input) = input else {
            bail!("Unexpected input rank: {input_rank}, expected 4.")
        };
        let result =
            burn::tensor::module::max_pool2d(input, kernel_size, stride, padding, dilation);
        Ok(WrappedTensor::Rank4(result))
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
        let Self::Rank2(input) = input else {
            bail!("Unexpected input rank: {input_rank}, expected 2.")
        };
        let gamma_rank = gamma.rank();
        let Self::Rank1(gamma) = gamma else {
            bail!("Unexpected gamma rank: {gamma_rank}, expected 1.")
        };
        let beta_rank = beta.rank();
        let Self::Rank1(beta) = beta else {
            bail!("Unexpected beta rank: {beta_rank}, expected 1.")
        };
        let config = LayerNormConfig::new(embedding_size).with_epsilon(epsilon);
        let device = Default::default();
        let mut norm = config.init(&device);
        norm.gamma = Param::from_tensor(gamma);
        norm.beta = Param::from_tensor(beta);
        let output = norm.forward(input);
        Ok(Self::Rank2(output))
    }

    pub fn softmax(tensor: Self, dim: usize) -> Result<Self> {
        let tensor_rank = tensor.rank();
        let Self::Rank2(tensor) = tensor else {
            bail!("Unexpected tensor rank: {tensor_rank}, expected 2.")
        };
        Ok(Self::Rank2(activation::softmax(tensor, dim)))
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
        let Self::Rank2(input) = input else {
            bail!("Unexpected input rank: {input_rank}, expected 2.")
        };
        let gamma_rank = gamma.rank();
        let Self::Rank1(gamma) = gamma else {
            bail!("Unexpected gamma rank: {gamma_rank}, expected 1.")
        };
        let config = RmsNormConfig::new(embedding_size).with_epsilon(epsilon);
        let device = Default::default();
        let mut norm = config.init(&device);
        norm.gamma = Param::from_tensor(gamma);
        let output = norm.forward(input);
        Ok(Self::Rank2(output))
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
        let Self::Rank4(x) = x else {
            bail!("Unexpected x rank: {x_rank}, expected 4.")
        };
        let weight_rank = weight.rank();
        let Self::Rank4(weight) = weight else {
            bail!("Unexpected weight rank: {weight_rank}, expected 4.")
        };
        let bias_rank = bias.rank();
        let Self::Rank1(bias) = bias else {
            bail!("Unexpected bias rank: {bias_rank}, expected 1.")
        };
        let out = zkml_conv2d_i(x, weight, bias, options);
        Ok(WrappedTensor::Rank4(out))
    }

    pub fn max_pool2d(input: Self, config: Maxpool2dConfig) -> Result<Self> {
        let input_rank = input.rank();
        let Self::Rank4(input) = input else {
            bail!("Unexpected input rank: {input_rank}, expected 4.")
        };
        Ok(WrappedTensor::Rank4(zkml_max_pool2d_i(input, config)))
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
        let Self::Rank2(weight) = weight else {
            bail!("Unexpected weight rank: {weight_rank}, expected 2.")
        };
        let bias = match bias {
            Some(bias) => {
                let bias_rank = bias.rank();
                let Self::Rank1(bias) = bias else {
                    bail!("Unexpected bias rank: {bias_rank}, expected 1.")
                };
                Some(bias)
            }
            None => None,
        };
        let out = match input {
            WrappedTensor::Rank1(input) => {
                WrappedTensor::Rank1(burn::tensor::module::linear(input, weight, bias))
            }
            WrappedTensor::Rank2(input) => {
                WrappedTensor::Rank2(burn::tensor::module::linear(input, weight, bias))
            }
            WrappedTensor::Rank3(input) => {
                WrappedTensor::Rank3(burn::tensor::module::linear(input, weight, bias))
            }
            WrappedTensor::Rank4(input) => {
                WrappedTensor::Rank4(burn::tensor::module::linear(input, weight, bias))
            }
        };
        Ok(out)
    }

    /// Applies the rectified linear unit function element-wise
    /// as described in the paper [Deep Learning using Rectified Linear Units (ReLU)](https://arxiv.org/pdf/1803.08375).
    fn relu(input: Self) -> Self {
        match input {
            WrappedTensor::Rank1(input) => {
                WrappedTensor::Rank1(burn::tensor::activation::relu(input))
            }
            WrappedTensor::Rank2(input) => {
                WrappedTensor::Rank2(burn::tensor::activation::relu(input))
            }
            WrappedTensor::Rank3(input) => {
                WrappedTensor::Rank3(burn::tensor::activation::relu(input))
            }
            WrappedTensor::Rank4(input) => {
                WrappedTensor::Rank4(burn::tensor::activation::relu(input))
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
            WrappedTensor::Rank1(input) => {
                let input = input.into_primitive();
                let mask = Backend::int_lower_equal_elem(input.clone(), 0);
                let out = Backend::int_mask_fill(input, mask, 0);
                let out = BTensor::from_primitive(out);
                WrappedTensor::Rank1(out)
            }
            WrappedTensor::Rank2(input) => {
                let input = input.into_primitive();
                let mask = Backend::int_lower_equal_elem(input.clone(), 0);
                let out = Backend::int_mask_fill(input, mask, 0);
                let out = BTensor::from_primitive(out);
                WrappedTensor::Rank2(out)
            }
            WrappedTensor::Rank3(input) => {
                let input = input.into_primitive();
                let mask = Backend::int_lower_equal_elem(input.clone(), 0);
                let out = Backend::int_mask_fill(input, mask, 0);
                let out = BTensor::from_primitive(out);
                WrappedTensor::Rank3(out)
            }
            WrappedTensor::Rank4(input) => {
                let input = input.into_primitive();
                let mask = Backend::int_lower_equal_elem(input.clone(), 0);
                let out = Backend::int_mask_fill(input, mask, 0);
                let out = BTensor::from_primitive(out);
                WrappedTensor::Rank4(out)
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
        let out = match rank {
            1 => {
                let input = tensor.clone().to_btensor::<1>();
                WrappedTensor::Rank1(input)
            }
            2 => {
                let input = tensor.clone().to_btensor::<2>();
                WrappedTensor::Rank2(input)
            }
            3 => {
                let input = tensor.clone().to_btensor::<3>();
                WrappedTensor::Rank3(input)
            }
            4 => {
                let input = tensor.clone().to_btensor::<4>();
                WrappedTensor::Rank4(input)
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
        Ok(Tensor::<T>::new(shape, data))
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
            DimIter::Rank1(iter) => iter.next().map(WrappedTensor::Rank1),
            DimIter::Rank2(iter) => iter.next().map(WrappedTensor::Rank2),
            DimIter::Rank3(iter) => iter.next().map(WrappedTensor::Rank3),
            DimIter::Rank4(iter) => iter.next().map(WrappedTensor::Rank4),
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
        let tensor = Tensor::new(shape.clone(), vec![1]).into_wrapped();
        assert_eq!(
            tensor.clone().pad_next_power_of_two().to_native(),
            tensor.to_native(),
            "Tensor should not change if the shape is already power of two",
        );

        let shape = Shape::new(vec![2, 2]);
        let tensor = Tensor::new(shape.clone(), vec![1, 2, 1, 2]).into_wrapped();
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
        let tensor = Tensor::new(shape.clone(), vec![1, 2, 3, 1, 2, 3, 1, 2, 3]).into_wrapped();
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
        let tensor = Tensor::new(shape.clone(), vec![1, 2, 1, 2, 1, 2]).into_wrapped();
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
