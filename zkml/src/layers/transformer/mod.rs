use crate::{
    Shape,
    padding::PaddingMode,
    tensor::{TensorTypeParam, WrappedTensor},
};

use burn::tensor::TensorData;
use serde::{Deserialize, Serialize};

pub mod attention_mask;
pub mod embeddings;
pub mod layernorm;
pub mod logits;
pub mod positional;
pub mod rmsnorm;
pub mod softmax;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConcatenationCache {
    cached_tensor: Option<TensorData>,
    rank: usize,
    concatenation_dim: usize,
    padding_mode: PaddingMode,
}

impl ConcatenationCache {
    pub fn new(rank: usize, concatenation_dim: usize, padding_mode: PaddingMode) -> Self {
        Self {
            cached_tensor: None,
            rank,
            concatenation_dim,
            padding_mode,
        }
    }
    pub fn reset(&mut self) {
        self.cached_tensor = None;
    }
    pub fn is_initialized(&self) -> bool {
        self.cached_tensor.is_some()
    }

    pub fn concatenate<N: TensorTypeParam>(
        &mut self,
        new_tensor: WrappedTensor<N>,
    ) -> anyhow::Result<WrappedTensor<N>> {
        let reduced = if let PaddingMode::NoPadding = self.padding_mode {
            new_tensor
        } else {
            let unpadded_shape = new_tensor.unpadded_shape().clone();
            new_tensor.reduce_to_shape(&unpadded_shape)?
        };

        let output = if self.is_initialized() {
            let mut placeholder = None;
            std::mem::swap(&mut placeholder, &mut self.cached_tensor);

            // Unwrap is safe because we are initialised
            let cached_tensor = WrappedTensor::from_data(placeholder.unwrap())?;
            let catted = WrappedTensor::cat(vec![cached_tensor, reduced], self.concatenation_dim)?;
            self.cached_tensor = Some(catted.clone().to_data());
            catted
        } else {
            self.cached_tensor = Some(reduced.clone().to_data());
            reduced
        };

        match self.padding_mode {
            PaddingMode::NoPadding => Ok(output),
            PaddingMode::Padding => Ok(output.pad_next_power_of_two()),
        }
    }

    /// Given a [`Shape`], returns the next shape after concatenation.
    /// Here `padding_mode` determines whether to pad the new shape to the next power of two.
    pub fn next_shape(&self, shape: Shape, padding_mode: PaddingMode) -> Shape {
        let mut new_shape = shape;
        new_shape[self.concatenation_dim] += self.current_sequence_length();
        if let PaddingMode::Padding = padding_mode {
            new_shape = new_shape.next_power_of_two();
        }
        new_shape
    }

    pub fn get_cached<N: TensorTypeParam>(&self) -> anyhow::Result<WrappedTensor<N>> {
        if let PaddingMode::NoPadding = self.padding_mode {
            let inner_data = self
                .cached_tensor
                .clone()
                .ok_or(anyhow::anyhow!("ConcatenationCache is not initialized"))?;
            WrappedTensor::from_data(inner_data)
        } else {
            let inner_data = self
                .cached_tensor
                .clone()
                .ok_or(anyhow::anyhow!("ConcatenationCache is not initialized"))?;
            WrappedTensor::from_data(inner_data).map(|t| t.pad_next_power_of_two())
        }
    }

    pub fn set_padding_mode(&mut self, padding_mode: PaddingMode) {
        // We reset the cache when we change the padding mode
        self.reset();
        self.padding_mode = padding_mode;
    }

    pub fn current_sequence_length(&self) -> usize {
        if let Some(tensor) = &self.cached_tensor {
            tensor.shape[self.concatenation_dim]
        } else {
            0
        }
    }

    pub fn cache_info(&self) -> (usize, usize) {
        (self.rank, self.concatenation_dim)
    }
}
