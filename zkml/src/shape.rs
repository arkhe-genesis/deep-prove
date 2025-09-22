use crate::{NextPowerOfTwo, quantization};
use multilinear_extensions::util::ceil_log2;
use serde::{Deserialize, Serialize};
use std::{
    cmp::PartialEq,
    ops::{Bound, Range, RangeBounds},
};

/// Return's the filter size.
///
/// This work for convolutions filters/inputs/outputs.
///
/// NOTE: This only works if the filter is square (height == width).
pub(crate) fn filter_size(shape: &Shape) -> usize {
    // NOTE:
    // - This assumes the filter is square, meaning width and height are the same
    // - Given the above, this works for both 3D and 4D tensors.
    debug_assert!(
        shape.rank() != 4 || shape.dim(2) == shape.dim(3),
        "Width and height must match. shape {shape:?}",
    );
    debug_assert!(
        shape.rank() != 3 || shape.dim(1) == shape.dim(2),
        "Width and height must match. shape {shape:?}",
    );

    shape.dim(2) * shape.dim(2)
}

/// Structure that holds a shape of a tensor.
/// NOTE: it is currently being phased in incrementally the codebase currently. There will be places where we still use Vec<usize>
#[derive(
    Debug,
    Clone,
    derive_more::From,
    derive_more::Into,
    derive_more::AsRef,
    derive_more::Index,
    derive_more::IndexMut,
    derive_more::Deref,
    derive_more::DerefMut,
    Serialize,
    Deserialize,
    PartialEq,
    Eq,
)]
pub struct Shape(Vec<usize>);

impl<const T: usize> From<[usize; T]> for Shape {
    fn from(value: [usize; T]) -> Self {
        Self(value.into_iter().collect())
    }
}

impl Shape {
    /// Creates a new shape from the iterator.
    ///
    /// ```
    /// # use zkml::Shape;
    /// let first = Shape::from_it([1, 2, 3]);
    /// Shape::from_it(first.iter());
    /// ```
    pub fn from_it<V: std::borrow::Borrow<usize>, I: IntoIterator<Item = V>>(iter: I) -> Self {
        Self(iter.into_iter().map(|v| *v.borrow()).collect())
    }

    /// Creates a new [Shape].
    ///
    /// # Panics
    ///
    /// If `shape` is an empty vector.
    pub fn new(shape: Vec<usize>) -> Self {
        assert!(!shape.is_empty(), "Shape can not be empty");
        Self(shape)
    }

    /// Returns a new [Shape] with the `dimensions` flattened.
    ///
    /// ```
    /// # use zkml::Shape;
    /// let shape = Shape::from_it([2, 3, 5, 7]);
    /// assert_eq!(shape.flatten(1..3), Shape::from_it([2, 15, 7]))
    /// ```
    ///
    /// # Panics
    ///
    /// If the given dimensions are out-of-bounds.
    pub fn flatten(&self, dims: Range<usize>) -> Shape {
        let mut newdims = Vec::with_capacity(self.len() - (dims.end - dims.start));
        newdims.extend(&self.0[0..dims.start]);
        newdims.push(self.0[dims.clone()].iter().product());
        newdims.extend(&self.0[dims.end..]);
        Shape(newdims)
    }

    pub fn slice<R: RangeBounds<usize>>(&self, range: R) -> Shape {
        let len = self.0.len();
        let start = match range.start_bound() {
            Bound::Included(&s) => s,
            Bound::Excluded(&s) => s + 1,
            Bound::Unbounded => 0,
        };
        let end = match range.end_bound() {
            Bound::Included(&e) => e + 1,
            Bound::Excluded(&e) => e,
            Bound::Unbounded => len,
        };
        Shape(self.0[start..end].to_vec())
    }

    /// Returns the size of a given dimension.
    ///
    /// ```
    /// # use zkml::Shape;
    /// let shape = Shape::new(vec![3, 5, 7]);
    /// assert_eq!(shape.dim(0), 3);
    /// assert_eq!(shape.dim(1), 5);
    /// assert_eq!(shape.dim(2), 7);
    /// ```
    ///
    /// # Panics
    ///
    /// If the given dimensions are out-of-bounds.
    pub fn dim(&self, index: usize) -> usize {
        self.0[index]
    }

    /// Sets the value of a given dimension.
    ///
    /// ```
    /// # use zkml::Shape;
    /// let mut shape = Shape::new(vec![3, 5]);
    /// shape.set_dim(1, 10);
    /// assert_eq!(shape.dim(1), 10);
    /// ```
    ///
    /// # Panics
    ///
    /// If the given dimensions are out-of-bounds.
    pub fn set_dim(&mut self, index: usize, value: usize) {
        assert!(index < self.0.len(), "Index out of bounds");
        self.0[index] = value;
    }

    /// Removes `dims` dimensions from the front of [Shape].
    ///
    /// # Panics
    ///
    /// If any of the dimensions value is not `1` or if `dims` is bigger than `Shape`.
    ///
    /// ```
    /// # use zkml::Shape;
    /// let shape = Shape::new(vec![1, 1, 3, 5]);
    /// let new_shape = shape.squeeze_front(2);
    /// assert_eq!(new_shape.dim(0), 3);
    /// assert_eq!(new_shape.dim(1), 5);
    /// ```
    pub fn squeeze_front(&self, dims: usize) -> Self {
        assert!(
            self.0.iter().take(dims).all(|v| *v == 1),
            "Squeezing only allowed for dimensions of size 1. shape {self:?}",
        );

        let new_size = self.0.len() - dims;
        let mut new_shape = Vec::with_capacity(new_size);
        new_shape.extend(&self.0[dims..]);

        debug_assert_eq!(new_shape.iter().product::<usize>(), self.product());
        Self(new_shape)
    }

    /// Removes a dimension with size `1` from [Shape].
    ///
    /// # Panics
    ///
    /// Panics if `index` is larger than this shape size or if the dimension size
    /// is not `1`.
    ///
    /// ```
    /// # use zkml::Shape;
    /// let shape = Shape::new(vec![3, 1, 5]);
    /// let new_shape = shape.squeeze(1);
    /// assert_eq!(new_shape.dim(0), 3);
    /// assert_eq!(new_shape.dim(1), 5);
    /// ```
    pub fn squeeze(&self, index: usize) -> Self {
        let mut new_shape = self.0.clone();
        assert_eq!(
            new_shape.remove(index),
            1,
            "Squeezed dimension must be 1. shape {self:?}",
        );
        Self(new_shape)
    }

    /// Adds `dims` extra dimensions with size `1` to the front of [Shape].
    ///
    /// ```
    /// # use zkml::Shape;
    /// let shape = Shape::new(vec![3, 5]);
    /// let new_shape = shape.unsqueeze_front(2);
    /// assert_eq!(new_shape.dim(0), 1);
    /// assert_eq!(new_shape.dim(1), 1);
    /// assert_eq!(new_shape.dim(2), 3);
    /// assert_eq!(new_shape.dim(3), 5);
    /// ```
    pub fn unsqueeze_front(&self, dims: usize) -> Self {
        let new_size = self.0.len() + dims;
        let mut new_shape = Vec::with_capacity(new_size);
        new_shape.resize(new_size, 1);
        new_shape[dims..].copy_from_slice(self.0.as_slice());

        debug_assert_eq!(new_shape.iter().product::<usize>(), self.product());
        Self(new_shape)
    }

    /// Adds an extra dimension with size `1` to [Shape].
    ///
    /// # Panics
    ///
    /// Panics if `index` is larger than this shape size.
    ///
    /// ```
    /// # use zkml::Shape;
    /// let shape = Shape::new(vec![3, 5]);
    /// let new_shape = shape.unsqueeze(1);
    /// assert_eq!(new_shape.dim(0), 3);
    /// assert_eq!(new_shape.dim(1), 1);
    /// assert_eq!(new_shape.dim(2), 5);
    /// ```
    pub fn unsqueeze(&self, index: usize) -> Self {
        let mut new_shape = self.0.clone();
        new_shape.insert(index, 1);
        Self(new_shape)
    }

    /// Returns the strides for this [Shape] in row major order.
    ///
    /// The values in the stride vector determine the offset
    /// needed to go to the next element of a given dimension.
    ///
    /// ```
    /// # use zkml::Shape;
    /// let shape = Shape::new(vec![3, 5, 7]);
    /// let strides = shape.strides();
    /// // row major order, inner most dimension changes the quickest
    /// assert_eq!(strides[0], 35);
    /// assert_eq!(strides[1], 7);
    /// assert_eq!(strides[2], 1);
    /// ```
    pub fn strides(&self) -> Vec<usize> {
        let mut strides = self
            .0
            .iter()
            .rev()
            .scan(1usize, |state, item| {
                let el = Some(*state);
                *state *= item;
                el
            })
            .collect::<Vec<_>>();

        strides.reverse();
        strides
    }

    /// Inserts a new dimension at the given index.
    ///
    /// # Panics
    ///
    /// Panics if `index` is larger than this shape size.
    ///
    /// ```
    /// # use zkml::Shape;
    /// let shape = Shape::new(vec![3, 5]);
    /// let new_shape = shape.insert(1, 1);
    /// assert_eq!(new_shape.dim(0), 3);
    /// assert_eq!(new_shape.dim(1), 1);
    /// assert_eq!(new_shape.dim(2), 5);
    /// ```
    pub fn insert(&self, index: usize, value: usize) -> Self {
        let mut new_shape = self.0.clone();
        new_shape.insert(index, value);
        Self(new_shape)
    }

    pub fn permute(&self, permutation: &[usize]) -> Self {
        Self(permutation.iter().map(|i| self.0[*i]).collect())
    }
    pub fn next_power_of_two(&self) -> Self {
        Self(self.0.next_power_of_two())
    }
    pub fn extend(&self, other: &Self) -> Self {
        let mut new_shape = self.0.clone();
        new_shape.extend(other.0.clone());
        Self(new_shape)
    }
    pub fn concat(&self, other: &Self) -> Self {
        assert!(
            self.rank() == other.rank(),
            "Shapes must have the same rank"
        );
        assert!(
            self.0
                .iter()
                .zip(other.0.iter())
                .skip(1)
                .all(|(a, b)| a == b)
        );
        let mut new_shape = self.0.clone();
        new_shape[0] += other.0[0];
        Self(new_shape)
    }
    pub fn into_vec(self) -> Vec<usize> {
        self.0
    }
    pub fn rank(&self) -> usize {
        self.0.len()
    }

    /// True if the tensor is 0D, i.e. has no dimensions.
    pub fn is_empty(&self) -> bool {
        self.rank() == 0
    }

    /// True if the tensor is 1D, or if it is 2D but the first dimensions is 1.
    pub fn is_vector(&self) -> bool {
        self.rank() == 1 || (self.rank() == 2 && self.dim(0) == 1)
    }

    /// True if the tensor is 2D.
    pub fn is_matrix(&self) -> bool {
        self.rank() == 2
    }

    /// True if the tensor is 4D.
    pub fn is_convolution(&self) -> bool {
        self.rank() == 4
    }

    pub fn ncols(&self) -> usize {
        assert!(self.is_matrix(), "Tensor is not a matrix");
        self.0[1]
    }
    pub fn nrows(&self) -> usize {
        assert!(self.is_matrix(), "Tensor is not a matrix");
        self.0[0]
    }
    // Compute the bitsize of the output of the matrix multiplication of a tensor with shape `self`
    // with another matrix with a compatible shape. It requires the optional inputs to specify the range
    // of the quantized values in `self` and in the other matrix being multiplied with `self`
    pub fn matmul_output_bitsize(
        &self,
        quantized_self_input_range: Option<usize>,
        quantized_other_input_range: Option<usize>,
    ) -> usize {
        assert!(self.is_matrix(), "Tensor is not a matrix");
        // formula is 2^{2 * BIT_LEN + log(c) + 1} where c is the number of columns and +1 because of the bias
        let ncols = self.ncols();
        // - 1 because numbers are signed so only half of the range is used when doing multiplication
        quantized_self_input_range
            .map(ceil_log2)
            .unwrap_or(*quantization::BIT_LEN - 1)
            + quantized_other_input_range
                .map(ceil_log2)
                .unwrap_or(*quantization::BIT_LEN - 1)
            + ceil_log2(ncols)
            + 1
    }
    pub fn is_power_of_two(&self) -> bool {
        self.0.iter().all(|x| x.is_power_of_two())
    }
    pub fn product(&self) -> usize {
        self.0.iter().product()
    }
    pub fn numel(&self) -> usize {
        self.product()
    }

    /// Returns the number of variables in each dimension of the shape.
    /// Assumes that the shape is already a padded shape
    pub fn num_vars(&self) -> Vec<usize> {
        assert!(self.is_power_of_two());
        self.0.iter().map(|s| s.ilog2() as usize).collect()
    }

    /// Get the number of rows from the matrix
    pub fn nrows_2d(&self) -> usize {
        let mut cols = 0;
        let dims = &self.0;
        if self.is_matrix() {
            cols = dims[0];
        } else if self.is_convolution() {
            cols = dims[0] * dims[2] * dims[2];
        }
        assert!(cols != 0, "Shape is not a matrix or convolution");
        cols
    }

    /// Get the number of cols from the matrix
    pub fn ncols_2d(&self) -> usize {
        let mut cols = 0;
        let dims = &self.0;
        if self.is_matrix() {
            cols = dims[1];
        } else if self.is_convolution() {
            cols = dims[1] * dims[2] * dims[2];
        }
        assert!(cols != 0, "Shape is not a matrix or convolution");

        cols
    }

    pub fn num_vars_2d(&self) -> (usize, usize) {
        assert!(self.is_matrix(), "Shape is not a matrix");
        (
            self.nrows_2d().ilog2() as usize,
            self.ncols_2d().ilog2() as usize,
        )
    }
}

impl FromIterator<usize> for Shape {
    fn from_iter<T: IntoIterator<Item = usize>>(iter: T) -> Self {
        Self::new(iter.into_iter().collect::<Vec<usize>>())
    }
}

impl From<burn::prelude::Shape> for Shape {
    fn from(value: burn::prelude::Shape) -> Self {
        Self(value.dims)
    }
}

impl From<&burn::prelude::Shape> for Shape {
    fn from(value: &burn::prelude::Shape) -> Self {
        Self(value.dims.to_vec())
    }
}

#[cfg(test)]
mod test {
    use crate::Shape;
    use std::panic::catch_unwind;

    #[test]
    fn test_shape() {
        let shape = Shape::new(vec![2, 3, 4]);
        let permuted = shape.permute(&[1, 0, 2]);
        assert_eq!(permuted.as_ref(), &[3, 2, 4]);
    }

    #[test]
    fn test_shape_concat() {
        let shape1 = Shape::new(vec![2, 3, 4]);
        let shape2 = Shape::new(vec![3, 4, 5]);
        assert!(catch_unwind(|| { shape1.concat(&shape2) }).is_err());
        let shape3 = Shape::new(vec![3, 3, 4]);
        let new = shape1.concat(&shape3);
        assert_eq!(new, vec![5, 3, 4].into());
    }

    #[test]
    fn test_squeeze() {
        let shape = Shape::new(vec![7]);
        for i in [1, 2, 3, 4] {
            let roundtrip = shape.unsqueeze_front(i).squeeze_front(i);
            assert_eq!(roundtrip, shape);
        }

        let shape = Shape::new(vec![5, 7]);
        for i in [1, 2, 3, 4] {
            let roundtrip = shape.unsqueeze_front(i).squeeze_front(i);
            assert_eq!(roundtrip, shape);
        }

        let shape = Shape::new(vec![3, 5, 7]);
        for i in [1, 2, 3, 4] {
            let roundtrip = shape.unsqueeze_front(i).squeeze_front(i);
            assert_eq!(roundtrip, shape);
        }

        let shape = Shape::new(vec![2, 3, 5, 7]);
        for i in [1, 2, 3, 4] {
            let roundtrip = shape.unsqueeze_front(i).squeeze_front(i);
            assert_eq!(roundtrip, shape);
        }
    }
}
