use crate::{Shape, Tensor};

pub struct TensorSlice<'a, T> {
    data: &'a [T],
    shape: Shape,
}

impl<'a, T> From<&'a Tensor<T>> for TensorSlice<'a, T> {
    fn from(value: &'a Tensor<T>) -> Self {
        Self {
            data: &value.data,
            shape: value.shape.clone(),
        }
    }
}

impl<'a, T> TensorSlice<'a, T> {
    pub(crate) fn get_shape(&self) -> Shape {
        self.shape.clone()
    }

    pub(crate) fn get_data(&self) -> &[T] {
        self.data
    }

    pub(crate) fn slice_over_first_dim(&self, dim2_start: usize, dim2_end: usize) -> Self {
        let range = dim2_start * self.shape[1]..dim2_end * self.shape[1];
        let data = &self.data[range];
        let mut new_shape = self.shape.clone();
        new_shape[0] = dim2_end - dim2_start;
        Self {
            data,
            shape: new_shape,
        }
    }
}

impl<'a, T: Clone> TensorSlice<'a, T> {
    pub(crate) fn to_tensor(&self) -> anyhow::Result<Tensor<T>> {
        Tensor::new(self.shape.clone(), self.data.to_vec())
    }
}
