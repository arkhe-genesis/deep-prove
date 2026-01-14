//! Wrapper type for burn's tensor.

pub use burn::tensor::{Shape as BShape, TensorKind as BTensorKind};
use serde::{Deserialize, Serialize};

use crate::{
    NextPowerOfTwo, Number,
    backend::{Backend, Conv2dConfig, Maxpool2dConfig, zkml_conv2d_i, zkml_max_pool2d_i},
};
use anyhow::{Context, Result, bail, ensure};
use burn::{
    module::Param,
    nn::{LayerNormConfig, RmsNormConfig},
    tensor::{
        AsIndex, BasicOps, BroadcastArgs, DimIter as BDimIter, Numeric, SliceArg,
        Tensor as BTensor, TensorData, activation,
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
    Rank1 {
        tensor: BTensor<Backend, 1, T::Kind>,
        unpadded_shape: BShape,
    },
    Rank2 {
        tensor: BTensor<Backend, 2, T::Kind>,
        unpadded_shape: BShape,
    },
    Rank3 {
        tensor: BTensor<Backend, 3, T::Kind>,
        unpadded_shape: BShape,
    },
    Rank4 {
        tensor: BTensor<Backend, 4, T::Kind>,
        unpadded_shape: BShape,
    },
}

pub enum DimIter<T>
where
    T: TensorTypeParam,
{
    Rank1 {
        tensor: BDimIter<Backend, 1, T::Kind>,
        unpadded_shape: BShape,
    },
    Rank2 {
        tensor: BDimIter<Backend, 2, T::Kind>,
        unpadded_shape: BShape,
    },
    Rank3 {
        tensor: BDimIter<Backend, 3, T::Kind>,
        unpadded_shape: BShape,
    },
    Rank4 {
        tensor: BDimIter<Backend, 4, T::Kind>,
        unpadded_shape: BShape,
    },
}

/// Delegate a `WrappedTensor` method to burn tensor method
macro_rules! delegate_plain {
    // Method with generic type param(s) given in parentheses before any fn args
    ($tensor: expr, $method: ident, ( $($type_arg: tt),* ), $($arg: expr),*) => {
        match $tensor {
            WrappedTensor::Rank1{tensor, ..} => tensor.$method::<$($type_arg),*>($($arg),*),
            WrappedTensor::Rank2{tensor, ..} => tensor.$method::<$($type_arg),*>($($arg),*),
            WrappedTensor::Rank3{tensor, ..} => tensor.$method::<$($type_arg),*>($($arg),*),
            WrappedTensor::Rank4{tensor, ..} => tensor.$method::<$($type_arg),*>($($arg),*),
        }
    };

    ($tensor: expr, $method: ident $(, $($arg: expr),* )?) => {
        match $tensor {
            WrappedTensor::Rank1{tensor, ..} => tensor.$method($($($arg),*)?),
            WrappedTensor::Rank2{tensor, ..} => tensor.$method($($($arg),*)?),
            WrappedTensor::Rank3{tensor, ..} => tensor.$method($($($arg),*)?),
            WrappedTensor::Rank4{tensor, ..} => tensor.$method($($($arg),*)?),
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
            WrappedTensor::Rank1{tensor, unpadded_shape} => {
                let tensor = tensor.$method($($($arg),*)?);
                WrappedTensor::Rank1{
                    tensor,
                    unpadded_shape,
                }
            },
            WrappedTensor::Rank2{tensor, unpadded_shape} => {
                let tensor = tensor.$method($($($arg),*)?);
                WrappedTensor::Rank2{
                    tensor,
                    unpadded_shape,
                }
            },
            WrappedTensor::Rank3{tensor, unpadded_shape} => {
                let tensor = tensor.$method($($($arg),*)?);
                WrappedTensor::Rank3{
                    tensor,
                    unpadded_shape,
                }
            },
            WrappedTensor::Rank4{tensor, unpadded_shape} => {
                let tensor = tensor.$method($($($arg),*)?);
                WrappedTensor::Rank4{
                    tensor,
                    unpadded_shape,
                }
            },
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
            (
                WrappedTensor::Rank1 {
                    tensor,
                    unpadded_shape,
                },
                WrappedTensor::Rank1 { tensor: arg0, .. },
            ) => WrappedTensor::Rank1 {
                tensor: tensor.$method(arg0),
                unpadded_shape,
            },
            (
                WrappedTensor::Rank2 {
                    tensor,
                    unpadded_shape,
                },
                WrappedTensor::Rank2 { tensor: arg0, .. },
            ) => WrappedTensor::Rank2 {
                tensor: tensor.$method(arg0),
                unpadded_shape,
            },
            (
                WrappedTensor::Rank3 {
                    tensor,
                    unpadded_shape,
                },
                WrappedTensor::Rank3 { tensor: arg0, .. },
            ) => WrappedTensor::Rank3 {
                tensor: tensor.$method(arg0),
                unpadded_shape,
            },
            (
                WrappedTensor::Rank4 {
                    tensor,
                    unpadded_shape,
                },
                WrappedTensor::Rank4 { tensor: arg0, .. },
            ) => WrappedTensor::Rank4 {
                tensor: tensor.$method(arg0),
                unpadded_shape,
            },
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
            Self::Rank1 { .. } => 1,
            Self::Rank2 { .. } => 2,
            Self::Rank3 { .. } => 3,
            Self::Rank4 { .. } => 4,
        }
    }

    pub fn unpadded_shape(&self) -> &BShape {
        match self {
            WrappedTensor::Rank1 { unpadded_shape, .. } => unpadded_shape,
            WrappedTensor::Rank2 { unpadded_shape, .. } => unpadded_shape,
            WrappedTensor::Rank3 { unpadded_shape, .. } => unpadded_shape,
            WrappedTensor::Rank4 { unpadded_shape, .. } => unpadded_shape,
        }
    }

    pub fn set_unpadded_shape(&mut self, new_shape: BShape) {
        match self {
            WrappedTensor::Rank1 { unpadded_shape, .. }
            | WrappedTensor::Rank2 { unpadded_shape, .. }
            | WrappedTensor::Rank3 { unpadded_shape, .. }
            | WrappedTensor::Rank4 { unpadded_shape, .. } => *unpadded_shape = new_shape,
        }
    }

    pub fn is_padded(&self) -> bool {
        self.unpadded_shape() != &self.shape()
    }

    /// Reshape the tensor to have the given shape.
    pub fn reshape(self, new_shape: BShape) -> Result<WrappedTensor<T>> {
        let new_unpadded = if !self.is_padded() {
            new_shape.clone()
        } else {
            // XXX: Figure out how to split unpadded shape
            ensure!(
                new_shape.len() <= self.unpadded_shape().len(),
                "Increasing the number of dimensions is not supported. shape: {:?} unppaded_shape: {:?} new_shape: {new_shape:?}",
                self.shape(),
                self.unpadded_shape(),
            );

            let curr_shape = self.shape();
            let mut curr_dims = curr_shape.iter();
            let mut curr_unpadded = self.unpadded_shape().iter();

            let mut new_unpadded = Vec::with_capacity(new_shape.len());
            for new in new_shape.iter() {
                let mut curr = *curr_dims
                    .next()
                    .expect("Current shape rank is at least equal to new");
                let mut unpadded = *curr_unpadded
                    .next()
                    .expect("Current unpadded shape rank is at least equal to new");

                while *new != curr {
                    curr *= curr_dims
                        .next()
                        .expect("Current shape rank is at least equal to new");
                    unpadded *= curr_unpadded
                        .next()
                        .expect("Current unpadded shape rank is at least equal to new");
                }

                new_unpadded.push(unpadded);
            }
            BShape::from(new_unpadded)
        };

        let out = match new_shape.num_dims() {
            1 => WrappedTensor::Rank1 {
                tensor: delegate_plain!(self, reshape, (1, _), new_shape),
                unpadded_shape: new_unpadded,
            },
            2 => WrappedTensor::Rank2 {
                tensor: delegate_plain!(self, reshape, (2, _), new_shape),
                unpadded_shape: new_unpadded,
            },
            3 => WrappedTensor::Rank3 {
                tensor: delegate_plain!(self, reshape, (3, _), new_shape),
                unpadded_shape: new_unpadded,
            },
            4 => WrappedTensor::Rank4 {
                tensor: delegate_plain!(self, reshape, (4, _), new_shape),
                unpadded_shape: new_unpadded,
            },
            _ => bail!("Unexpected tensor rank: {}.", new_shape.num_dims()),
        };

        Ok(out)
    }

    /// Converts the data of the current tensor.
    pub fn to_data(&self) -> TensorData {
        delegate_plain!(self, to_data)
    }

    /// Creates a tensor from [`TensorData`].
    pub fn from_data(data: TensorData) -> Result<Self> {
        let rank = data.shape.len();
        let unpadded_shape = BShape::from(&data.shape);
        let out = match rank {
            1 => WrappedTensor::Rank1 {
                tensor: BTensor::<Backend, 1, T::Kind>::from_data(data, &Default::default()),
                unpadded_shape,
            },
            2 => WrappedTensor::Rank2 {
                tensor: BTensor::<Backend, 2, T::Kind>::from_data(data, &Default::default()),
                unpadded_shape,
            },
            3 => WrappedTensor::Rank3 {
                tensor: BTensor::<Backend, 3, T::Kind>::from_data(data, &Default::default()),
                unpadded_shape,
            },
            4 => WrappedTensor::Rank4 {
                tensor: BTensor::<Backend, 4, T::Kind>::from_data(data, &Default::default()),
                unpadded_shape,
            },
            _ => bail!("Unexpected tensor rank: {rank}."),
        };
        Ok(out)
    }

    /// Returns a copy of the tensor data.
    pub fn get_data(&self) -> Vec<T> {
        self.clone().to_data().into_vec().unwrap()
    }

    /// Returns the shape of the current tensor.
    pub fn shape(&self) -> BShape {
        delegate_plain!(self, shape)
    }

    /// Concatenates all tensors into a new one along the given dimension.
    pub fn cat(tensors: Vec<Self>, dim: usize) -> Result<Self> {
        ensure!(tensors.len() > 1, "There must be at least one tensor");

        let mut unpadded_shape = tensors[0].unpadded_shape().clone();
        let summed_concat_dim = tensors
            .iter()
            .map(|tensor| tensor.shape().dims[dim])
            .sum::<usize>();
        unpadded_shape.dims[dim] = summed_concat_dim;

        match tensors[0].rank() {
            1 => {
                let to_concat = tensors
                    .into_iter()
                    .map(|tensor| match tensor {
                        WrappedTensor::Rank1 { tensor, .. } => Ok(tensor),
                        _ => bail!("Cat does not support mixed ranks"),
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(WrappedTensor::Rank1 {
                    tensor: BTensor::cat(to_concat, dim),
                    unpadded_shape,
                })
            }
            2 => {
                let to_concat = tensors
                    .into_iter()
                    .map(|tensor| match tensor {
                        WrappedTensor::Rank2 { tensor, .. } => Ok(tensor),
                        _ => bail!("Cat does not support mixed ranks"),
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(WrappedTensor::Rank2 {
                    tensor: BTensor::cat(to_concat, dim),
                    unpadded_shape,
                })
            }
            3 => {
                let to_concat = tensors
                    .into_iter()
                    .map(|tensor| match tensor {
                        WrappedTensor::Rank3 { tensor, .. } => Ok(tensor),
                        _ => bail!("Cat does not support mixed ranks"),
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(WrappedTensor::Rank3 {
                    tensor: BTensor::cat(to_concat, dim),
                    unpadded_shape,
                })
            }
            4 => {
                let to_concat = tensors
                    .into_iter()
                    .map(|tensor| match tensor {
                        WrappedTensor::Rank4 { tensor, .. } => Ok(tensor),
                        _ => bail!("Cat does not support mixed ranks"),
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(WrappedTensor::Rank4 {
                    tensor: BTensor::cat(to_concat, dim),
                    unpadded_shape,
                })
            }
            _ => unreachable!(),
        }
    }

    /// Attempts to split the tensor into a specified number of chunks along a
    /// given dimension.
    ///
    /// May return less chunks than requested if the tensor size is not
    /// divisible by the number of chunks.
    pub fn chunk(self, chunks: usize, dim: usize) -> Result<Vec<Self>> {
        ensure!(
            dim < self.rank(),
            "Chunk dimension {dim} out of bounds for tensor of rank {}",
            self.rank(),
        );

        let original_shape = self.shape();
        let original_unpadded = self.unpadded_shape().clone();

        // Every dimension should have at least one non padding element.
        // XXX: Unclear what exactly should be the rule here, this is a relaxed approach.
        let padding = original_shape.dims[dim] - original_unpadded.dims[dim];
        ensure!(
            padding <= chunks,
            "Chunk must not be larger than the padding to include at least one non-padding element",
        );

        let mut out: Vec<Self> = match self {
            WrappedTensor::Rank1 { tensor, .. } => tensor
                .chunk(chunks, dim)
                .into_iter()
                .map(WrappedTensor::from)
                .collect(),
            WrappedTensor::Rank2 { tensor, .. } => tensor
                .chunk(chunks, dim)
                .into_iter()
                .map(WrappedTensor::from)
                .collect(),
            WrappedTensor::Rank3 { tensor, .. } => tensor
                .chunk(chunks, dim)
                .into_iter()
                .map(WrappedTensor::from)
                .collect(),
            WrappedTensor::Rank4 { tensor, .. } => tensor
                .chunk(chunks, dim)
                .into_iter()
                .map(WrappedTensor::from)
                .collect(),
        };

        // Fix the last tensor unpadded shape
        if original_shape != original_unpadded {
            let last = out.last_mut().expect("At least one tensor");

            let new_size = original_unpadded.dims[dim] % chunks;
            let mut new_unpadded_shape = last.unpadded_shape().clone();
            new_unpadded_shape.dims[dim] = new_size;

            last.set_unpadded_shape(new_unpadded_shape);
        }

        Ok(out)
    }

    /// Find the maximum absolute value.
    pub fn max_abs(self) -> Self {
        let tensor = delegate_plain!(self, max_abs);
        let unpadded_shape = tensor.shape();
        WrappedTensor::Rank1 {
            tensor,
            unpadded_shape,
        }
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
        let mut unpadded_shape = self.unpadded_shape().clone();
        ensure!(
            unpadded_shape.dims.remove(dim) == 1,
            "Dimensions must be equalt to 1 to be squeezed"
        );
        let out = match self {
            WrappedTensor::Rank1 { .. } => bail!("Cannot squeeze 1D tensor"),
            WrappedTensor::Rank2 { tensor, .. } => WrappedTensor::Rank1 {
                tensor: tensor.squeeze_dim(dim),
                unpadded_shape,
            },
            WrappedTensor::Rank3 { tensor, .. } => WrappedTensor::Rank2 {
                tensor: tensor.squeeze_dim(dim),
                unpadded_shape,
            },
            WrappedTensor::Rank4 { tensor, .. } => WrappedTensor::Rank3 {
                tensor: tensor.squeeze_dim(dim),
                unpadded_shape,
            },
        };
        Ok(out)
    }

    /// Select tensor elements along the given dimension corresponding to the given indices.
    pub fn select(
        self,
        dim: impl AsIndex,
        indices: WrappedTensor<Element>,
    ) -> anyhow::Result<Self> {
        let mut unpadded_shape = self.unpadded_shape().clone();

        let WrappedTensor::Rank1 {
            tensor: indices, ..
        } = indices
        else {
            bail!("Only 1D indices is supported");
        };
        // XXX: how should indices outside of the unpadded shape be handled?
        unpadded_shape.dims[dim.index() as usize] = indices.shape().num_elements();

        let res = match self {
            WrappedTensor::Rank1 { tensor, .. } => WrappedTensor::Rank1 {
                tensor: tensor.select(dim, indices),
                unpadded_shape,
            },
            WrappedTensor::Rank2 { tensor, .. } => WrappedTensor::Rank2 {
                tensor: tensor.select(dim, indices),
                unpadded_shape,
            },
            WrappedTensor::Rank3 { tensor, .. } => WrappedTensor::Rank3 {
                tensor: tensor.select(dim, indices),
                unpadded_shape,
            },
            WrappedTensor::Rank4 { tensor, .. } => WrappedTensor::Rank4 {
                tensor: tensor.select(dim, indices),
                unpadded_shape,
            },
        };

        Ok(res)
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
    pub fn sum_dim(self, dim: isize) -> Self {
        delegate!(self, sum_dim, dim)
    }

    /// Returns a new tensor with the same shape and device as the current tensor filled with the provided value.
    pub fn full_like(self, fill_value: T) -> Self {
        delegate!(self, full_like, fill_value)
    }

    /// Flatten the tensor into 1D shape.
    pub fn flatten_1d(self) -> Self {
        let end_dim = self.rank() - 1;
        let unpadded_shape = self.unpadded_shape().clone().flatten();
        WrappedTensor::Rank1 {
            tensor: delegate_plain!(self, flatten, 0, end_dim),
            unpadded_shape,
        }
    }

    /// Flatten the tensor along a given range of dimensions into 2 dimensions.
    pub fn flatten_to_dim_2(self, start_dim: usize, end_dim: usize) -> Self {
        let mut unpadded_shape = self.unpadded_shape().clone();
        let new_size = unpadded_shape.dims.drain(start_dim..=end_dim).product();
        unpadded_shape.dims.insert(start_dim, new_size);
        let tensor = delegate_plain!(self, flatten, start_dim, end_dim);
        Self::Rank2 {
            tensor,
            unpadded_shape,
        }
    }

    /// Find the maximum value along the given dimension.
    pub fn max_dim(self, dim: isize) -> Self {
        delegate!(self, max_dim, dim)
    }

    ///  Find the maximum value along the given dimension.
    pub fn max_dim_with_indices(self, dim: usize) -> (Self, WrappedTensor<Element>) {
        // XXX: fix unpadded shapes
        match self {
            WrappedTensor::Rank1 {
                tensor,
                unpadded_shape,
            } => {
                let (maxes, indices) = tensor.max_dim_with_indices(dim);
                (
                    WrappedTensor::Rank1 {
                        tensor: maxes,
                        unpadded_shape: unpadded_shape.clone(),
                    },
                    WrappedTensor::Rank1 {
                        tensor: indices,
                        unpadded_shape,
                    },
                )
            }
            WrappedTensor::Rank2 {
                tensor,
                unpadded_shape,
            } => {
                let (maxes, indices) = tensor.max_dim_with_indices(dim);
                (
                    WrappedTensor::Rank2 {
                        tensor: maxes,
                        unpadded_shape: unpadded_shape.clone(),
                    },
                    WrappedTensor::Rank2 {
                        tensor: indices,
                        unpadded_shape,
                    },
                )
            }
            WrappedTensor::Rank3 {
                tensor,
                unpadded_shape,
            } => {
                let (maxes, indices) = tensor.max_dim_with_indices(dim);
                (
                    WrappedTensor::Rank3 {
                        tensor: maxes,
                        unpadded_shape: unpadded_shape.clone(),
                    },
                    WrappedTensor::Rank3 {
                        tensor: indices,
                        unpadded_shape,
                    },
                )
            }
            WrappedTensor::Rank4 {
                tensor,
                unpadded_shape,
            } => {
                let (maxes, indices) = tensor.max_dim_with_indices(dim);
                (
                    WrappedTensor::Rank4 {
                        tensor: maxes,
                        unpadded_shape: unpadded_shape.clone(),
                    },
                    WrappedTensor::Rank4 {
                        tensor: indices,
                        unpadded_shape,
                    },
                )
            }
        }
    }

    /// Unsqueeze the current tensor. Create new leading dimensions to fit 2 dimensions.
    pub fn unsqueeze_dim_2(self) -> Self {
        let mut unpadded_shape = self.unpadded_shape().clone();
        while unpadded_shape.len() != 2 {
            unpadded_shape.dims.insert(0, 1);
        }

        let tensor = delegate_plain!(self, unsqueeze, (2),);
        WrappedTensor::Rank2 {
            tensor,
            unpadded_shape,
        }
    }

    /// Unsqueeze the current tensor. Create new leading dimensions to fit 3 dimensions.
    pub fn unsqueeze_dim_3(self) -> Self {
        let mut unpadded_shape = self.unpadded_shape().clone();
        while unpadded_shape.len() != 3 {
            unpadded_shape.dims.insert(0, 1);
        }

        let tensor = delegate_plain!(self, unsqueeze, (3),);
        WrappedTensor::Rank3 {
            tensor,
            unpadded_shape,
        }
    }

    /// Unsqueeze the current tensor. Create new leading dimensions to fit 4 dimensions.
    pub fn unsqueeze_dim_4(self) -> Self {
        let mut unpadded_shape = self.unpadded_shape().clone();
        while unpadded_shape.len() != 4 {
            unpadded_shape.dims.insert(0, 1);
        }

        let tensor = delegate_plain!(self, unsqueeze, (4),);
        WrappedTensor::Rank4 {
            tensor,
            unpadded_shape,
        }
    }

    /// Creates a new tensor with a dimension of size one inserted at the specified position.
    pub fn unsqueeze_dim(self, dim: usize) -> Result<Self> {
        let mut unpadded_shape = self.unpadded_shape().clone();
        unpadded_shape.dims.insert(dim, 1);

        let out = match self {
            WrappedTensor::Rank1 { tensor, .. } => WrappedTensor::Rank2 {
                tensor: tensor.unsqueeze_dim(dim),
                unpadded_shape,
            },
            WrappedTensor::Rank2 { tensor, .. } => WrappedTensor::Rank3 {
                tensor: tensor.unsqueeze_dim(dim),
                unpadded_shape,
            },
            WrappedTensor::Rank3 { tensor, .. } => WrappedTensor::Rank4 {
                tensor: tensor.unsqueeze_dim(dim),
                unpadded_shape,
            },
            WrappedTensor::Rank4 { .. } => bail!("Cannot unsqueeze 4D tensor"),
        };
        Ok(out)
    }

    /// Iterate over slices of tensors alongside a given dimension.
    pub fn iter_dim(self, dim: usize) -> DimIter<T> {
        match self {
            WrappedTensor::Rank1 {
                tensor,
                unpadded_shape,
            } => DimIter::Rank1 {
                tensor: tensor.iter_dim(dim),
                unpadded_shape,
            },
            WrappedTensor::Rank2 {
                tensor,
                unpadded_shape,
            } => DimIter::Rank2 {
                tensor: tensor.iter_dim(dim),
                unpadded_shape,
            },
            WrappedTensor::Rank3 {
                tensor,
                unpadded_shape,
            } => DimIter::Rank3 {
                tensor: tensor.iter_dim(dim),
                unpadded_shape,
            },
            WrappedTensor::Rank4 {
                tensor,
                unpadded_shape,
            } => DimIter::Rank4 {
                tensor: tensor.iter_dim(dim),
                unpadded_shape,
            },
        }
    }

    /// Permute the dimensions of the tensor.
    pub fn permute(self, axes: &[isize]) -> Result<Self> {
        let out = match self {
            WrappedTensor::Rank1 {
                tensor,
                unpadded_shape,
            } => {
                let axes: [isize; 1] = TryFrom::try_from(axes).with_context(|| {
                    format!(
                        "Unexpected permutation axes length. Expected 1, got {}",
                        axes.len(),
                    )
                })?;
                let shape_axes = axes.map(|d| d as usize);
                let unpadded_shape = unpadded_shape
                    .permute(shape_axes.as_slice())
                    .map_err(|e| anyhow::anyhow!("Could not permute unpadded shape: permutation: {shape_axes:?}, inner error:{e:?}"))?;
                WrappedTensor::Rank1 {
                    tensor: tensor.permute(axes),
                    unpadded_shape,
                }
            }
            WrappedTensor::Rank2 {
                tensor,
                unpadded_shape,
            } => {
                let axes: [isize; 2] = TryFrom::try_from(axes).with_context(|| {
                    format!(
                        "Unexpected permutation axes length. Expected 2, got {}",
                        axes.len(),
                    )
                })?;
                let shape_axes = axes.map(|d| d as usize);
                let unpadded_shape = unpadded_shape
                    .permute(shape_axes.as_slice())
                    .map_err(|e| anyhow::anyhow!("Could not permute unpadded shape: permutation: {shape_axes:?}, inner error:{e:?}"))?;
                WrappedTensor::Rank2 {
                    tensor: tensor.permute(axes),
                    unpadded_shape,
                }
            }
            WrappedTensor::Rank3 {
                tensor,
                unpadded_shape,
            } => {
                let axes: [isize; 3] = TryFrom::try_from(axes).with_context(|| {
                    format!(
                        "Unexpected permutation axes length. Expected 3, got {}",
                        axes.len(),
                    )
                })?;
                let shape_axes = axes.map(|d| d as usize);
                let unpadded_shape = unpadded_shape
                    .permute(shape_axes.as_slice())
                    .map_err(|e| anyhow::anyhow!("Could not permute unpadded shape: permutation: {shape_axes:?}, inner error:{e:?}"))?;
                WrappedTensor::Rank3 {
                    tensor: tensor.permute(axes),
                    unpadded_shape,
                }
            }
            WrappedTensor::Rank4 {
                tensor,
                unpadded_shape,
            } => {
                let axes: [isize; 4] = TryFrom::try_from(axes).with_context(|| {
                    format!(
                        "Unexpected permutation axes length. Expected 4, got {}",
                        axes.len(),
                    )
                })?;
                let shape_axes = axes.map(|d| d as usize);
                let unpadded_shape = unpadded_shape
                    .permute(shape_axes.as_slice())
                    .map_err(|e| anyhow::anyhow!("Could not permute unpadded shape: permutation: {shape_axes:?}, inner error:{e:?}"))?;
                WrappedTensor::Rank4 {
                    tensor: tensor.permute(axes),
                    unpadded_shape,
                }
            }
        };
        Ok(out)
    }

    /// Returns a tensor containing the elements selected from the given ranges.
    pub fn slice<R: Clone + SliceArg>(self, ranges: R) -> Self {
        fn to_shape<R: SliceArg>(shape: &BShape, ranges: R) -> BShape {
            let slices = ranges.into_slices(shape);
            let dims: Vec<usize> = slices
                .iter()
                .enumerate()
                .map(|(dim, slice)| slice.output_size(shape.dims[dim]))
                .collect();
            BShape::from(dims)
        }

        match self {
            WrappedTensor::Rank1 {
                tensor,
                unpadded_shape,
            } => WrappedTensor::Rank1 {
                tensor: tensor.slice(ranges.clone()),
                unpadded_shape: to_shape(&unpadded_shape, ranges),
            },
            WrappedTensor::Rank2 {
                tensor,
                unpadded_shape,
            } => WrappedTensor::Rank2 {
                tensor: tensor.slice(ranges.clone()),
                unpadded_shape: to_shape(&unpadded_shape, ranges),
            },
            WrappedTensor::Rank3 {
                tensor,
                unpadded_shape,
            } => WrappedTensor::Rank3 {
                tensor: tensor.slice(ranges.clone()),
                unpadded_shape: to_shape(&unpadded_shape, ranges),
            },
            WrappedTensor::Rank4 {
                tensor,
                unpadded_shape,
            } => WrappedTensor::Rank4 {
                tensor: tensor.slice(ranges.clone()),
                unpadded_shape: to_shape(&unpadded_shape, ranges),
            },
        }
    }

    /// Returns the size of the given dimension.
    ///
    /// When given a negative `dim` indexes from the back.
    pub fn dim(&self, dim: isize) -> Result<usize> {
        match self {
            WrappedTensor::Rank1 { tensor, .. } => {
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
            WrappedTensor::Rank2 { tensor, .. } => {
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
            WrappedTensor::Rank3 { tensor, .. } => {
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
            WrappedTensor::Rank4 { tensor, .. } => {
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

    /// Converts the tensor into a primitive tensor.
    pub fn into_primitive(self) -> <T::Kind as BTensorKind<Backend>>::Primitive {
        delegate_plain!(self, into_primitive)
    }

    /// Broadcast the tensor to the given shape.
    pub fn expand<const D: usize, S: Clone + BroadcastArgs<D, D>>(self, shape: S) -> Result<Self> {
        let unpadded_shape = shape.clone().into_shape(self.unpadded_shape());
        let shape = shape.into_shape(&self.shape());

        let out = match shape.num_dims() {
            1 => WrappedTensor::Rank1 {
                tensor: delegate_plain!(self, expand, (1, _), shape),
                unpadded_shape,
            },
            2 => WrappedTensor::Rank2 {
                tensor: delegate_plain!(self, expand, (2, _), shape),
                unpadded_shape,
            },
            3 => WrappedTensor::Rank3 {
                tensor: delegate_plain!(self, expand, (3, _), shape),
                unpadded_shape,
            },
            4 => WrappedTensor::Rank4 {
                tensor: delegate_plain!(self, expand, (4, _), shape),
                unpadded_shape,
            },
            _ => bail!("Unexpected tensor rank: {}.", shape.num_dims()),
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

    pub fn reduce_to_unpadded_shape(self) -> Result<Self> {
        let unpadded_shape = self.unpadded_shape().clone();
        self.reduce_to_shape(&unpadded_shape)
    }

    /// Update the given tensor with the value where the mask is true.
    pub fn mask_fill_4d(
        self,
        mask: BTensor<Backend, 4, burn::tensor::Bool>,
        value: T,
    ) -> Result<Self> {
        let input_rank = self.rank();
        let Self::Rank4 {
            tensor,
            unpadded_shape,
        } = self
        else {
            bail!("Unexpected input rank: {input_rank}, expected 4.")
        };
        Ok(WrappedTensor::Rank4 {
            tensor: tensor.mask_fill(mask, value),
            unpadded_shape,
        })
    }

    pub fn mask_fill(
        self,
        mask: BTensor<Backend, 2, burn::tensor::Bool>,
        value: T,
    ) -> Result<Self> {
        match self {
            WrappedTensor::Rank1 { .. } => {
                bail!("mask_fill only works for rank 2 or higher tensors")
            }
            WrappedTensor::Rank2 {
                tensor,
                unpadded_shape,
            } => Ok(WrappedTensor::Rank2 {
                tensor: tensor.mask_fill(mask, value),
                unpadded_shape,
            }),
            WrappedTensor::Rank3 {
                tensor,
                unpadded_shape,
            } => {
                // We expand the mask to rank 3 by unsqueezing a new leading dimension.
                let mask = mask.unsqueeze::<3>().expand(tensor.shape());
                Ok(WrappedTensor::Rank3 {
                    tensor: tensor.mask_fill(mask, value),
                    unpadded_shape,
                })
            }
            WrappedTensor::Rank4 {
                tensor,
                unpadded_shape,
            } => {
                // We expand the mask to rank 4 by unsqueezing a new leading dimension.
                let mask = mask.unsqueeze::<4>().expand(tensor.shape());
                Ok(WrappedTensor::Rank4 {
                    tensor: tensor.mask_fill(mask, value),
                    unpadded_shape,
                })
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
            WrappedTensor::Rank1 {
                tensor,
                unpadded_shape,
            } => {
                #[allow(clippy::single_range_in_vec_init)]
                let ranges = [0..dims[0]];
                let out = BTensor::full(shape, T::zero(), &Default::default());
                let out = out.slice_assign(ranges, tensor);
                WrappedTensor::Rank1 {
                    tensor: out,
                    unpadded_shape,
                }
            }
            WrappedTensor::Rank2 {
                tensor,
                unpadded_shape,
            } => {
                let ranges = [0..dims[0], 0..dims[1]];
                let out = BTensor::full(shape, T::zero(), &Default::default());
                let out = out.slice_assign(ranges, tensor);
                WrappedTensor::Rank2 {
                    tensor: out,
                    unpadded_shape,
                }
            }
            WrappedTensor::Rank3 {
                tensor,
                unpadded_shape,
            } => {
                let ranges = [0..dims[0], 0..dims[1], 0..dims[2]];
                let out = BTensor::full(shape, T::zero(), &Default::default());
                let out = out.slice_assign(ranges, tensor);
                WrappedTensor::Rank3 {
                    tensor: out,
                    unpadded_shape,
                }
            }
            WrappedTensor::Rank4 {
                tensor,
                unpadded_shape,
            } => {
                let ranges = [0..dims[0], 0..dims[1], 0..dims[2], 0..dims[3]];
                let out = BTensor::full(shape, T::zero(), &Default::default());
                let out = out.slice_assign(ranges, tensor);
                WrappedTensor::Rank4 {
                    tensor: out,
                    unpadded_shape,
                }
            }
        }
    }

    pub fn random(shape: &Shape) -> Self {
        Self::try_from(&Tensor::random(shape)).unwrap()
    }

    pub fn equal_elem<const D: usize>(
        self,
        elem: T,
    ) -> Result<BTensor<Backend, D, burn::tensor::Bool>> {
        anyhow::ensure!(
            self.rank() == D,
            "Unexpected tensor rank: {}, expected {}",
            self.rank(),
            D
        );
        let shape = self.shape();
        match D {
            1 => {
                let Self::Rank1 { tensor, .. } = self else {
                    unreachable!()
                };
                let result = tensor.equal_elem(elem);
                Ok(result.reshape::<D, _>(shape))
            }
            2 => {
                let Self::Rank2 { tensor, .. } = self else {
                    unreachable!()
                };
                let result = tensor.equal_elem(elem);
                Ok(result.reshape::<D, _>(shape))
            }
            3 => {
                let Self::Rank3 { tensor, .. } = self else {
                    unreachable!()
                };
                let result = tensor.equal_elem(elem);
                Ok(result.reshape::<D, _>(shape))
            }
            4 => {
                let Self::Rank4 { tensor, .. } = self else {
                    unreachable!()
                };
                let result = tensor.equal_elem(elem);
                Ok(result.reshape::<D, _>(shape))
            }
            _ => unreachable!(),
        }
    }

    /// Utility to make tests more readable
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

    /// Performs exp element-wise on the tensor.
    pub fn exp(self) -> Self {
        delegate!(self, exp)
    }

    /// Performs ln (natural logarithm) element-wise on the tensor.
    pub fn log(self) -> Self {
        delegate!(self, log)
    }

    /// Applies the Gaussian Error Linear Units function as described in the paper
    /// [Gaussian Error Linear Units (GELUs)](https://arxiv.org/pdf/1606.08415v3.pdf).
    pub fn gelu(input: Self) -> Self {
        match input {
            WrappedTensor::Rank1 {
                tensor,
                unpadded_shape,
            } => WrappedTensor::Rank1 {
                tensor: burn::tensor::activation::gelu(tensor),
                unpadded_shape,
            },
            WrappedTensor::Rank2 {
                tensor,
                unpadded_shape,
            } => WrappedTensor::Rank2 {
                tensor: burn::tensor::activation::gelu(tensor),
                unpadded_shape,
            },
            WrappedTensor::Rank3 {
                tensor,
                unpadded_shape,
            } => WrappedTensor::Rank3 {
                tensor: burn::tensor::activation::gelu(tensor),
                unpadded_shape,
            },
            WrappedTensor::Rank4 {
                tensor,
                unpadded_shape,
            } => WrappedTensor::Rank4 {
                tensor: burn::tensor::activation::gelu(tensor),
                unpadded_shape,
            },
        }
    }

    pub fn conv2d(
        input: Self,
        weight: Self,
        bias: Option<Self>,
        options: ConvOptions<2>,
    ) -> Result<Self> {
        let input_rank = input.rank();
        let Self::Rank4 {
            tensor: input,
            unpadded_shape,
        } = input
        else {
            bail!("Unexpected input rank: {input_rank}, expected 4.")
        };
        let weight_rank = weight.rank();
        let Self::Rank4 { tensor: weight, .. } = weight else {
            bail!("Unexpected weight rank: {weight_rank}, expected 4.")
        };
        let bias = match bias {
            Some(bias) => {
                let bias_rank = bias.rank();
                let Self::Rank1 { tensor: bias, .. } = bias else {
                    bail!("Unexpected bias rank: {bias_rank}, expected 1.")
                };
                Some(bias)
            }
            None => None,
        };
        let out = burn::tensor::module::conv2d(input, weight, bias, options);
        Ok(WrappedTensor::Rank4 {
            tensor: out,
            unpadded_shape,
        })
    }

    pub fn max_pool2d(
        input: Self,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        dilation: [usize; 2],
    ) -> Result<Self> {
        let input_rank = input.rank();
        let Self::Rank4 {
            tensor,
            unpadded_shape,
        } = input
        else {
            bail!("Unexpected input rank: {input_rank}, expected 4.")
        };
        let result =
            burn::tensor::module::max_pool2d(tensor, kernel_size, stride, padding, dilation, false);
        Ok(WrappedTensor::Rank4 {
            tensor: result,
            unpadded_shape,
        })
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
        let Self::Rank2 {
            tensor: input,
            unpadded_shape,
        } = input
        else {
            bail!("Unexpected input rank: {input_rank}, expected 2.")
        };
        let gamma_rank = gamma.rank();
        let Self::Rank1 { tensor: gamma, .. } = gamma else {
            bail!("Unexpected gamma rank: {gamma_rank}, expected 1.")
        };
        let beta_rank = beta.rank();
        let Self::Rank1 { tensor: beta, .. } = beta else {
            bail!("Unexpected beta rank: {beta_rank}, expected 1.")
        };
        let config = LayerNormConfig::new(embedding_size).with_epsilon(epsilon);
        let device = Default::default();
        let mut norm = config.init(&device);
        norm.gamma = Param::from_tensor(gamma);
        norm.beta = Some(Param::from_tensor(beta));
        let output = norm.forward(input);
        Ok(Self::Rank2 {
            tensor: output,
            unpadded_shape,
        })
    }

    pub fn softmax(tensor: Self, dim: usize) -> Result<Self> {
        anyhow::ensure!(
            dim < tensor.rank(),
            "Softmax dimension {dim} out of bounds, (tensor rank: {}).",
            tensor.rank()
        );
        match tensor {
            WrappedTensor::Rank1 {
                tensor,
                unpadded_shape,
            } => Ok(WrappedTensor::Rank1 {
                tensor: activation::softmax(tensor, dim),
                unpadded_shape,
            }),
            WrappedTensor::Rank2 {
                tensor,
                unpadded_shape,
            } => Ok(WrappedTensor::Rank2 {
                tensor: activation::softmax(tensor, dim),
                unpadded_shape,
            }),
            WrappedTensor::Rank3 {
                tensor,
                unpadded_shape,
            } => Ok(WrappedTensor::Rank3 {
                tensor: activation::softmax(tensor, dim),
                unpadded_shape,
            }),
            WrappedTensor::Rank4 {
                tensor,
                unpadded_shape,
            } => Ok(WrappedTensor::Rank4 {
                tensor: activation::softmax(tensor, dim),
                unpadded_shape,
            }),
        }
    }

    pub fn rms_norm_forward(
        input: Self,
        embedding_size: usize,
        epsilon: f64,
        gamma: Option<Self>,
    ) -> Result<Self> {
        // NOTE: simply use the burn tensor API for now as we want to move towards using more burn features
        // instead of re-implementing everything ourselves.
        // copy implementation https://docs.rs/burn-core/0.17.0/src/burn_core/nn/norm/rms.rs.html#71
        let config = RmsNormConfig::new(embedding_size).with_epsilon(epsilon);
        let device = Default::default();
        let norm = if let Some(gamma) = gamma {
            let mut norm = config.init(&device);
            let gamma_rank = gamma.rank();
            let Self::Rank1 { tensor: gamma, .. } = gamma else {
                bail!("Unexpected gamma rank: {gamma_rank}, expected 1.")
            };
            norm.gamma = Param::from_tensor(gamma);
            norm
        } else {
            config.init(&device)
        };

        match input {
            WrappedTensor::Rank1 {
                tensor: input,
                unpadded_shape,
            } => {
                let output = norm.forward(input);
                Ok(WrappedTensor::Rank1 {
                    tensor: output,
                    unpadded_shape,
                })
            }
            WrappedTensor::Rank2 {
                tensor: input,
                unpadded_shape,
            } => {
                let output = norm.forward(input);
                Ok(WrappedTensor::Rank2 {
                    tensor: output,
                    unpadded_shape,
                })
            }
            WrappedTensor::Rank3 {
                tensor: input,
                unpadded_shape,
            } => {
                let output = norm.forward(input);
                Ok(WrappedTensor::Rank3 {
                    tensor: output,
                    unpadded_shape,
                })
            }
            WrappedTensor::Rank4 {
                tensor: input,
                unpadded_shape,
            } => {
                let output = norm.forward(input);
                Ok(WrappedTensor::Rank4 {
                    tensor: output,
                    unpadded_shape,
                })
            }
        }
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
        let Self::Rank4 {
            tensor: input,
            unpadded_shape,
        } = x
        else {
            bail!("Unexpected x rank: {x_rank}, expected 4.")
        };
        let weight_rank = weight.rank();
        let Self::Rank4 { tensor: weight, .. } = weight else {
            bail!("Unexpected weight rank: {weight_rank}, expected 4.")
        };
        let bias_rank = bias.rank();
        let Self::Rank1 { tensor: bias, .. } = bias else {
            bail!("Unexpected bias rank: {bias_rank}, expected 1.")
        };
        let out = zkml_conv2d_i(input, weight, bias, options)?;
        Ok(WrappedTensor::Rank4 {
            tensor: out,
            unpadded_shape,
        })
    }

    pub fn max_pool2d(input: Self, config: Maxpool2dConfig) -> Result<Self> {
        let input_rank = input.rank();
        let Self::Rank4 {
            tensor: input,
            unpadded_shape,
        } = input
        else {
            bail!("Unexpected input rank: {input_rank}, expected 4.")
        };
        Ok(WrappedTensor::Rank4 {
            tensor: zkml_max_pool2d_i(input, config)?,
            unpadded_shape,
        })
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
        let Self::Rank2 { tensor: weight, .. } = weight else {
            bail!("Unexpected weight rank: {weight_rank}, expected 2.")
        };
        let bias = match bias {
            Some(bias) => {
                let bias_rank = bias.rank();
                let Self::Rank1 { tensor: bias, .. } = bias else {
                    bail!("Unexpected bias rank: {bias_rank}, expected 1.")
                };
                Some(bias)
            }
            None => None,
        };
        let out = match input {
            WrappedTensor::Rank1 {
                tensor: input,
                unpadded_shape,
            } => WrappedTensor::Rank1 {
                tensor: burn::tensor::module::linear(input, weight, bias),
                unpadded_shape,
            },
            WrappedTensor::Rank2 {
                tensor: input,
                unpadded_shape,
            } => WrappedTensor::Rank2 {
                tensor: burn::tensor::module::linear(input, weight, bias),
                unpadded_shape,
            },
            WrappedTensor::Rank3 {
                tensor: input,
                unpadded_shape,
            } => WrappedTensor::Rank3 {
                tensor: burn::tensor::module::linear(input, weight, bias),
                unpadded_shape,
            },
            WrappedTensor::Rank4 {
                tensor: input,
                unpadded_shape,
            } => WrappedTensor::Rank4 {
                tensor: burn::tensor::module::linear(input, weight, bias),
                unpadded_shape,
            },
        };
        Ok(out)
    }

    /// Applies the rectified linear unit function element-wise
    /// as described in the paper [Deep Learning using Rectified Linear Units (ReLU)](https://arxiv.org/pdf/1803.08375).
    fn relu(input: Self) -> Self {
        match input {
            WrappedTensor::Rank1 {
                tensor: input,
                unpadded_shape,
            } => WrappedTensor::Rank1 {
                tensor: burn::tensor::activation::relu(input),
                unpadded_shape,
            },
            WrappedTensor::Rank2 {
                tensor: input,
                unpadded_shape,
            } => WrappedTensor::Rank2 {
                tensor: burn::tensor::activation::relu(input),
                unpadded_shape,
            },
            WrappedTensor::Rank3 {
                tensor: input,
                unpadded_shape,
            } => WrappedTensor::Rank3 {
                tensor: burn::tensor::activation::relu(input),
                unpadded_shape,
            },
            WrappedTensor::Rank4 {
                tensor: input,
                unpadded_shape,
            } => WrappedTensor::Rank4 {
                tensor: burn::tensor::activation::relu(input),
                unpadded_shape,
            },
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
            WrappedTensor::Rank1 {
                tensor: input,
                unpadded_shape,
            } => {
                let input = input.into_primitive();
                let mask = Backend::int_lower_equal_elem(input.clone(), 0);
                let out = Backend::int_mask_fill(input, mask, 0);
                let out = BTensor::from_primitive(out);
                WrappedTensor::Rank1 {
                    tensor: out,
                    unpadded_shape,
                }
            }
            WrappedTensor::Rank2 {
                tensor: input,
                unpadded_shape,
            } => {
                let input = input.into_primitive();
                let mask = Backend::int_lower_equal_elem(input.clone(), 0);
                let out = Backend::int_mask_fill(input, mask, 0);
                let out = BTensor::from_primitive(out);
                WrappedTensor::Rank2 {
                    tensor: out,
                    unpadded_shape,
                }
            }
            WrappedTensor::Rank3 {
                tensor: input,
                unpadded_shape,
            } => {
                let input = input.into_primitive();
                let mask = Backend::int_lower_equal_elem(input.clone(), 0);
                let out = Backend::int_mask_fill(input, mask, 0);
                let out = BTensor::from_primitive(out);
                WrappedTensor::Rank3 {
                    tensor: out,
                    unpadded_shape,
                }
            }
            WrappedTensor::Rank4 {
                tensor: input,
                unpadded_shape,
            } => {
                let input = input.into_primitive();
                let mask = Backend::int_lower_equal_elem(input.clone(), 0);
                let out = Backend::int_mask_fill(input, mask, 0);
                let out = BTensor::from_primitive(out);
                WrappedTensor::Rank4 {
                    tensor: out,
                    unpadded_shape,
                }
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
                let input = tensor.to_btensor::<1>();
                WrappedTensor::Rank1 {
                    tensor: input,
                    unpadded_shape,
                }
            }
            2 => {
                let input = tensor.to_btensor::<2>();
                WrappedTensor::Rank2 {
                    tensor: input,
                    unpadded_shape,
                }
            }
            3 => {
                let input = tensor.to_btensor::<3>();
                WrappedTensor::Rank3 {
                    tensor: input,
                    unpadded_shape,
                }
            }
            4 => {
                let input = tensor.to_btensor::<4>();
                WrappedTensor::Rank4 {
                    tensor: input,
                    unpadded_shape,
                }
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

impl<T> TryFrom<&WrappedTensor<T>> for Tensor<T>
where
    T: TensorTypeParam,
{
    type Error = anyhow::Error;

    fn try_from(tensor: &WrappedTensor<T>) -> Result<Self, Self::Error> {
        let shape = tensor.shape().into();
        let data = tensor.get_data();
        Tensor::<T>::new(shape, data)
    }
}

impl<T> TryFrom<&[T]> for WrappedTensor<T>
where
    T: TensorTypeParam,
{
    type Error = anyhow::Error;

    fn try_from(data: &[T]) -> Result<Self, Self::Error> {
        let data = TensorData::new(data.to_vec(), [data.len()]);
        WrappedTensor::from_data(data)
    }
}

impl<T> TryFrom<Vec<T>> for WrappedTensor<T>
where
    T: TensorTypeParam,
{
    type Error = anyhow::Error;

    fn try_from(data: Vec<T>) -> Result<Self, Self::Error> {
        let shape = [data.len()];
        let data = TensorData::new(data, shape);
        WrappedTensor::from_data(data)
    }
}

impl<T> TryFrom<WrappedTensor<T>> for Tensor<T>
where
    T: TensorTypeParam,
{
    type Error = anyhow::Error;

    fn try_from(tensor: WrappedTensor<T>) -> Result<Self, Self::Error> {
        let shape = tensor.shape().into();
        let data = tensor.get_data();
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

impl<T> From<BTensor<Backend, 1, T::Kind>> for WrappedTensor<T>
where
    T: TensorTypeParam,
{
    fn from(tensor: BTensor<Backend, 1, T::Kind>) -> Self {
        let unpadded_shape = tensor.shape().clone();
        WrappedTensor::Rank1 {
            tensor,
            unpadded_shape,
        }
    }
}

impl<T> From<BTensor<Backend, 2, T::Kind>> for WrappedTensor<T>
where
    T: TensorTypeParam,
{
    fn from(tensor: BTensor<Backend, 2, T::Kind>) -> Self {
        let unpadded_shape = tensor.shape().clone();
        WrappedTensor::Rank2 {
            tensor,
            unpadded_shape,
        }
    }
}

impl<T> From<BTensor<Backend, 3, T::Kind>> for WrappedTensor<T>
where
    T: TensorTypeParam,
{
    fn from(tensor: BTensor<Backend, 3, T::Kind>) -> Self {
        let unpadded_shape = tensor.shape().clone();
        WrappedTensor::Rank3 {
            tensor,
            unpadded_shape,
        }
    }
}

impl<T> From<BTensor<Backend, 4, T::Kind>> for WrappedTensor<T>
where
    T: TensorTypeParam,
{
    fn from(tensor: BTensor<Backend, 4, T::Kind>) -> Self {
        let unpadded_shape = tensor.shape().clone();
        WrappedTensor::Rank4 {
            tensor,
            unpadded_shape,
        }
    }
}

impl<T> Iterator for DimIter<T>
where
    T: TensorTypeParam,
{
    type Item = WrappedTensor<T>;

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            DimIter::Rank1 {
                tensor,
                unpadded_shape,
            } => tensor.next().map(|tensor| WrappedTensor::Rank1 {
                tensor,
                unpadded_shape: unpadded_shape.clone(),
            }),
            DimIter::Rank2 {
                tensor,
                unpadded_shape,
            } => tensor.next().map(|tensor| WrappedTensor::Rank2 {
                tensor,
                unpadded_shape: unpadded_shape.clone(),
            }),
            DimIter::Rank3 {
                tensor,
                unpadded_shape,
            } => tensor.next().map(|tensor| WrappedTensor::Rank3 {
                tensor,
                unpadded_shape: unpadded_shape.clone(),
            }),
            DimIter::Rank4 {
                tensor,
                unpadded_shape,
            } => tensor.next().map(|tensor| WrappedTensor::Rank4 {
                tensor,
                unpadded_shape: unpadded_shape.clone(),
            }),
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
            tensor.clone().pad_next_power_of_two().get_data(),
            tensor.get_data(),
            "Tensor should not change if the shape is already power of two",
        );

        let shape = Shape::new(vec![2, 2]);
        let tensor = Tensor::new(shape.clone(), vec![1, 2, 1, 2])
            .unwrap()
            .into_wrapped();
        assert_eq!(
            tensor.clone().pad_next_power_of_two().get_data(),
            tensor.get_data(),
            "Tensor should not change if the shape is already power of two",
        );

        let shape = Shape::new(vec![4, 4]);
        let tensor = WrappedTensor::<Element>::random(&shape.clone());
        assert_eq!(
            tensor.clone().pad_next_power_of_two().get_data(),
            tensor.get_data(),
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
