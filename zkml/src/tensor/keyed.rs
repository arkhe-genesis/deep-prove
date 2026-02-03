use std::ops::{Deref, DerefMut};

use serde::{Deserialize, Serialize};
use tenstore::StorageKey;

use crate::{
    ScalingFactor, Tensor,
    quantization::Quantize,
    tensor::{CommitmentId, TensorTypeParam, WrappedTensor},
};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct KeyedTensor<T> {
    pub(crate) key: StorageKey<T>,
    pub(crate) tensor: Tensor<T>,
}

impl<T> KeyedTensor<T> {
    pub fn new<S>(key: S, tensor: Tensor<T>) -> Self
    where
        S: Into<StorageKey<T>>,
    {
        Self {
            key: key.into(),
            tensor,
        }
    }

    /// Returns a reference to this tensor's [StorageKey].
    pub fn storage_key(&self) -> &StorageKey<T> {
        &self.key
    }

    pub fn commitment_id(&self) -> CommitmentId {
        (&self.key).into()
    }

    /// Consumes the [KeyedTensor] and returns the internal [Tensor].
    pub fn into_tensor(self) -> Tensor<T> {
        self.tensor
    }

    /// Returns a reference to the internal [Tensor].
    pub fn tensor(&self) -> &Tensor<T> {
        &self.tensor
    }

    /// Sets the value of `tensor`.
    pub(crate) fn set_tensor(&mut self, tensor: Tensor<T>) {
        self.tensor = tensor;
    }

    /// Returns a mutable reference to the internal [Tensor].
    pub fn tensor_mut(&mut self) -> &mut Tensor<T> {
        &mut self.tensor
    }

    pub fn try_map_tensor<U>(
        self,
        f: impl FnOnce(Tensor<T>) -> anyhow::Result<Tensor<U>>,
    ) -> anyhow::Result<KeyedTensor<U>> {
        Ok(KeyedTensor {
            key: self.key.cast::<U>(),
            tensor: f(self.tensor)?,
        })
    }

    pub fn try_new_map_tensor<U>(
        &self,
        f: impl FnOnce(&Tensor<T>) -> anyhow::Result<Tensor<U>>,
    ) -> anyhow::Result<KeyedTensor<U>> {
        Ok(KeyedTensor {
            key: self.key.cast::<U>(),
            tensor: f(&self.tensor)?,
        })
    }

    /// Consumes the [KeyedTensor] and returns its parts.
    pub(crate) fn into_parts(self) -> (Tensor<T>, StorageKey<T>) {
        (self.tensor, self.key)
    }
}

impl<T: Copy + Default> KeyedTensor<T> {
    /// Pads the tensor to the next power-of-two.
    pub fn pad_next_power_of_two(&self) -> Self {
        let tensor = self.tensor.pad_next_power_of_two();
        Self {
            key: self.key.clone(),
            tensor,
        }
    }
}

impl<T> KeyedTensor<T>
where
    T: TensorTypeParam,
{
    pub(crate) fn from_wrapped_tensor(
        tensor: WrappedTensor<T>,
        key: StorageKey<T>,
    ) -> anyhow::Result<Self> {
        let tensor = Tensor::try_from(tensor)?;
        Ok(Self { tensor, key })
    }
}

impl<T> Deref for KeyedTensor<T> {
    type Target = Tensor<T>;

    fn deref(&self) -> &Self::Target {
        &self.tensor
    }
}

impl<T> DerefMut for KeyedTensor<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.tensor
    }
}

impl<T> Quantize for KeyedTensor<T>
where
    T: Quantize,
{
    type Output = KeyedTensor<<T as Quantize>::Output>;

    fn quantize(&self, scaling: &ScalingFactor) -> Self::Output {
        KeyedTensor {
            key: self.key.cast::<<T as Quantize>::Output>(),
            tensor: self.tensor.quantize(scaling),
        }
    }
}
