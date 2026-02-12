use std::{
    ops::{Deref, DerefMut},
    sync::{Arc, MappedRwLockReadGuard, RwLock, RwLockReadGuard},
};

use anyhow::{Context, bail};
use burn::tensor::{Shape as BShape, TensorData};
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use tenstore::{GenStore, GenericStore, StorageKey, StoreError};

use crate::{
    Element, NextPowerOfTwo, ScalingFactor, Shape, Tensor,
    quantization::Quantize,
    tensor::{KeyedTensor, TensorTypeParam, WrappedTensor},
};

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound(serialize = "T: Serialize", deserialize = "T: DeserializeOwned"))]
pub struct InnerWrappedTensor<T>
where
    T: TensorTypeParam,
{
    /// A unique key for this tensor.
    storage_key: StorageKey<Vec<T>>,

    /// Storage used to save or load the underlying data.
    #[serde(skip)]
    store: GenStore,

    /// Accelerated tensor data, if available.
    ///
    /// If the tensor data is not available, the handler will try to hydrate it
    /// by reading from the corresponding tensor, which might need to be read
    /// from the store.
    wrapped_tensor: Arc<RwLock<Option<WrappedTensor<T>>>>,

    /// The shape of the tensor.
    shape: Shape,
    unpadded_shape: Shape,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound(serialize = "T: Serialize", deserialize = "T: DeserializeOwned"))]
pub struct InnerTensor<T>
where
    T: TensorTypeParam,
{
    /// A unique key for this tensor.
    storage_key: StorageKey<Vec<T>>,

    /// Storage used to save or load the underlying data.
    #[serde(skip)]
    store: GenStore,

    /// Tensor data, if available.
    ///
    /// If the tensor data is not available, the handler will try to hydrate it
    /// by reading from the corresponding store.
    tensor: Arc<RwLock<Option<Tensor<T>>>>,

    /// The shape of the tensor.
    shape: Shape,
    unpadded_shape: Shape,
}

/// A handle to manage different representations of the same data.
///
/// Tensors are used in different contexts, each requiring a different
/// representation / implementation. This struct wraps the necessary metadata to
/// load, store, and transform the tensors to different representation.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound(serialize = "T: Serialize", deserialize = "T: DeserializeOwned"))]
pub enum TensorHandle<T>
where
    T: TensorTypeParam,
{
    WrappedTensor(InnerWrappedTensor<T>),
    Tensor(InnerTensor<T>),
}

impl<T> TensorHandle<T>
where
    T: TensorTypeParam + Serialize + for<'a> Deserialize<'a>,
{
    /// Returns a reference to the cached [`Tensor`].
    ///
    /// NOTE: If the [`Tensor`] is not cached, this will load the data from
    /// the store.
    pub fn tensor(&self) -> anyhow::Result<MappedRwLockReadGuard<'_, Tensor<T>>> {
        match self {
            TensorHandle::WrappedTensor(..) => {
                bail!("Tensor is unavailable for a wrapped tensor handler")
            }
            TensorHandle::Tensor(InnerTensor { tensor, .. }) => loop {
                {
                    // scope for the read guard, it must be dropped before load can run
                    let guard = tensor.read().expect("Lock should not be poisoned");
                    if guard.is_some() {
                        let res = RwLockReadGuard::map(guard, |v| match v {
                            Some(v) => v,
                            None => {
                                unreachable!(
                                    "The option was checked above, this is in a read only region"
                                )
                            }
                        });
                        return Ok(res);
                    }
                }
                self.load()?;
            },
        }
    }
}

impl<T> TensorHandle<T>
where
    T: TensorTypeParam + Serialize + for<'a> Deserialize<'a>,
{
    /// Creates a [TensorHandle] from a [Tensor].
    pub fn from_tensor(
        storage_key: StorageKey<Vec<T>>,
        store: GenStore,
        tensor: Tensor<T>,
    ) -> Self {
        let shape = tensor.shape().clone();
        let unpadded_shape = tensor.unpadded_shape().clone();
        Self::Tensor(InnerTensor {
            storage_key,
            store,
            tensor: Arc::new(RwLock::new(Some(tensor))),
            shape,
            unpadded_shape,
        })
    }
}

impl<T> TensorHandle<T>
where
    T: TensorTypeParam + Serialize + for<'a> Deserialize<'a>,
{
    /// Takes ownership of the inner [WrappedTensor].
    pub(crate) fn take_wrapped_tensor(&self) -> anyhow::Result<WrappedTensor<T>> {
        match self {
            TensorHandle::WrappedTensor(inner_wrapped_tensor) => inner_wrapped_tensor
                .wrapped_tensor
                .write()
                .expect("Lock should not be poisoned")
                .take()
                .context("wrapped tensor is dry"),
            TensorHandle::Tensor(..) => bail!("Not a wrapped tensor variant"),
        }
    }

    /// Sets the wrapped tensor.
    ///
    /// # Errors
    ///
    /// - If this is not a wrapped tensor variant.
    pub(crate) fn set_wrapped_tensor(&mut self, tensor: WrappedTensor<T>) -> anyhow::Result<()> {
        match self {
            TensorHandle::WrappedTensor(inner_wrapped_tensor) => {
                inner_wrapped_tensor.shape = tensor.shape().into();
                inner_wrapped_tensor.unpadded_shape = tensor.unpadded_shape().into();
                *inner_wrapped_tensor
                    .wrapped_tensor
                    .write()
                    .expect("Lock should not be poisoned")
                    .deref_mut() = Some(tensor);
                Ok(())
            }
            TensorHandle::Tensor(..) => bail!("Not a wrapped tensor variant"),
        }
    }

    /// Ensures this is a [WrappedTensor] variant.
    ///
    /// NOTE: For external GPU this will perform an upload of the data.
    pub(crate) fn wrapped_tensor_variant(self) -> anyhow::Result<Self> {
        match self {
            result @ TensorHandle::WrappedTensor { .. } => Ok(result),
            TensorHandle::Tensor(InnerTensor {
                storage_key,
                store,
                tensor,
                shape,
                unpadded_shape,
            }) => {
                let guard = tensor.read().expect("Lock should not be poisioned");
                let wrapped_tensor = match guard.deref() {
                    Some(tensor) => Some(tensor.try_into()?),
                    None => None,
                };
                Ok(TensorHandle::WrappedTensor(InnerWrappedTensor {
                    storage_key,
                    store,
                    wrapped_tensor: Arc::new(RwLock::new(wrapped_tensor)),
                    shape,
                    unpadded_shape,
                }))
            }
        }
    }

    /// Ensures this is a [Tensor] variant dropping the accelerated data if needed.
    pub(crate) fn into_dry_tensor(self) -> anyhow::Result<Self> {
        match self {
            TensorHandle::WrappedTensor(InnerWrappedTensor {
                storage_key,
                store,
                shape,
                unpadded_shape,
                ..
            }) => {
                Ok(TensorHandle::Tensor(InnerTensor {
                    storage_key,
                    store,
                    // XXX: downloading the wrapped tensor data here causes tests to
                    // fail, this needs to be investigated.
                    tensor: Arc::new(RwLock::new(None)),
                    shape,
                    unpadded_shape,
                }))
            }
            result @ TensorHandle::Tensor { .. } => Ok(result),
        }
    }

    /// Returns the [StorageKey] used to identify the data in the store.
    pub(crate) fn storage_key(&self) -> &StorageKey<Vec<T>> {
        match self {
            TensorHandle::WrappedTensor(inner) => &inner.storage_key,
            TensorHandle::Tensor(inner) => &inner.storage_key,
        }
    }

    /// Returns the [StorageKey] used to identify the data in the store.
    pub(crate) fn set_storage_key(&mut self, storage_key: StorageKey<Vec<T>>) {
        match self {
            TensorHandle::WrappedTensor(inner) => inner.storage_key = storage_key,
            TensorHandle::Tensor(inner) => inner.storage_key = storage_key,
        }
    }

    /// Returns a reference to the shape of this tensor.
    pub(crate) fn shape(&self) -> &Shape {
        match self {
            TensorHandle::WrappedTensor(inner) => &inner.shape,
            TensorHandle::Tensor(inner) => &inner.shape,
        }
    }

    /// Returns a reference to the unpadded shape of this tensor.
    pub(crate) fn unpadded_shape(&self) -> &Shape {
        match self {
            TensorHandle::WrappedTensor(inner) => &inner.unpadded_shape,
            TensorHandle::Tensor(inner) => &inner.unpadded_shape,
        }
    }

    /// Returns a reference to the unpadded shape of this tensor.
    pub(crate) fn store(&self) -> &GenStore {
        match self {
            TensorHandle::WrappedTensor(inner) => &inner.store,
            TensorHandle::Tensor(inner) => &inner.store,
        }
    }

    /// Sets the `store` for this handle.
    pub(crate) fn attach_store(&mut self, value: GenStore) {
        match self {
            TensorHandle::WrappedTensor(inner) => inner.store = value,
            TensorHandle::Tensor(inner) => inner.store = value,
        }
    }

    /// Utility to load data from the store.
    fn load(&self) -> anyhow::Result<()> {
        match self {
            TensorHandle::WrappedTensor(InnerWrappedTensor {
                storage_key,
                store,
                wrapped_tensor,
                shape,
                ..
            }) => {
                let mut guard = wrapped_tensor.write().expect("Lock should not be poisoned");

                if guard.is_none() {
                    let tensor = store.fetch(storage_key).map(|data| {
                        let data = TensorData::new(data, BShape::from(shape.clone()));
                        WrappedTensor::from_data(data)
                    })??;
                    *guard = Some(tensor);
                }
            }
            TensorHandle::Tensor(InnerTensor {
                storage_key,
                store,
                tensor,
                shape,
                unpadded_shape,
            }) => {
                let mut guard = tensor.write().expect("Lock should not be poisoned");

                if guard.is_none() {
                    let tensor = store.fetch(storage_key).map(|data| {
                        Tensor::new_with_unpadded_shape(shape.clone(), unpadded_shape.clone(), data)
                    })??;
                    *guard = Some(tensor);
                }
            }
        }

        Ok(())
    }

    /// Returns a reference to the cached [`WrappedTensor`].
    ///
    /// NOTE: If the [`WrappedTensor`] is not cached, it will be created, to create
    /// the tensor the corresponding [`Tensor`] must be available, if it is not, the
    /// data will be loaded from the store.
    pub(crate) fn wrapped_tensor(
        &self,
    ) -> anyhow::Result<MappedRwLockReadGuard<'_, WrappedTensor<T>>> {
        match self {
            TensorHandle::WrappedTensor(InnerWrappedTensor { wrapped_tensor, .. }) => loop {
                {
                    // scope for the read guard, it must be dropped before load can run
                    let guard = wrapped_tensor.read().expect("Lock should not be poisoned");
                    if guard.is_some() {
                        let res = RwLockReadGuard::map(guard, |v| match v {
                            Some(v) => v,
                            None => {
                                unreachable!(
                                    "The option was checked above, this is in a read only region"
                                )
                            }
                        });
                        return Ok(res);
                    }
                }
                self.load()?;
            },
            TensorHandle::Tensor(..) => {
                bail!("Wrapped tensor is unavailable for a tensor handler")
            }
        }
    }

    /// Dries the current handle.
    ///
    /// Drying a handle frees the cached values to free memory.
    pub(crate) fn dry(&self) {
        match self {
            TensorHandle::WrappedTensor(InnerWrappedTensor { wrapped_tensor, .. }) => {
                let mut guard = wrapped_tensor
                    .write()
                    .expect("Lock should not be poisioned");
                *guard = None;
            }
            TensorHandle::Tensor(InnerTensor { tensor, .. }) => {
                let mut guard = tensor.write().expect("Lock should not be poisioned");
                *guard = None;
            }
        }
    }

    /// Ensure the transformed version of this tensor with type [S] exists in the store.
    ///
    /// If the transformed version does not exist yet, apply `f` over `self` and
    /// save it to the store.
    pub(crate) fn cast<S, F>(&self, f: F) -> Result<TensorHandle<S>, StoreError>
    where
        S: TensorTypeParam + Serialize + for<'a> Deserialize<'a>,
        F: Fn(&T) -> S,
    {
        match self {
            TensorHandle::WrappedTensor(InnerWrappedTensor {
                storage_key,
                store,
                shape,
                unpadded_shape,
                ..
            }) => {
                let storage_key =
                    store.cast(storage_key, |xs| xs.iter().map(&f).collect::<Vec<S>>())?;
                Ok(TensorHandle::<S>::WrappedTensor(InnerWrappedTensor {
                    storage_key,
                    store: store.clone(),
                    wrapped_tensor: Default::default(),
                    shape: shape.clone(),
                    unpadded_shape: unpadded_shape.clone(),
                }))
            }
            TensorHandle::Tensor(InnerTensor {
                storage_key,
                store,
                shape,
                unpadded_shape,
                ..
            }) => {
                let storage_key =
                    store.cast(storage_key, |xs| xs.iter().map(&f).collect::<Vec<S>>())?;
                Ok(TensorHandle::<S>::Tensor(InnerTensor {
                    storage_key,
                    store: store.clone(),
                    tensor: Default::default(),
                    shape: shape.clone(),
                    unpadded_shape: unpadded_shape.clone(),
                }))
            }
        }
    }

    /// Loads the transformed version of this tensor with type [S].
    ///
    /// If the transformed version does not yet exist, apply `f` over `self`,
    /// save it to store and returns a copy.
    pub(crate) fn hydrated_cast<S, F>(&self, f: F) -> anyhow::Result<Tensor<S>>
    where
        S: Serialize + for<'a> Deserialize<'a>,
        F: Fn(&T) -> S,
    {
        self.store()
            .cast_and_fetch(self.storage_key(), |xs| {
                xs.iter().map(&f).collect::<Vec<S>>()
            })
            .map(|bytes| {
                Tensor::new_with_unpadded_shape(
                    self.shape().clone(),
                    self.unpadded_shape().clone(),
                    bytes.1,
                )
            })?
    }

    /// Pads the tensor to the next power-of-two.
    pub(crate) fn pad_next_power_of_two(&self) -> Self {
        match self {
            TensorHandle::WrappedTensor(InnerWrappedTensor {
                storage_key,
                store,
                wrapped_tensor,
                shape,
                unpadded_shape,
            }) => {
                let guard = wrapped_tensor.read().expect("Lock should not be poisioned");
                let wrapped_tensor = guard
                    .as_ref()
                    .map(|wrapped_tensor| wrapped_tensor.clone().pad_next_power_of_two());

                TensorHandle::WrappedTensor(InnerWrappedTensor {
                    storage_key: storage_key.clone(),
                    store: store.clone(),
                    wrapped_tensor: Arc::new(RwLock::new(wrapped_tensor)),
                    shape: shape.next_power_of_two(),
                    unpadded_shape: unpadded_shape.next_power_of_two(),
                })
            }
            TensorHandle::Tensor(InnerTensor {
                storage_key,
                store,
                tensor,
                shape,
                unpadded_shape,
            }) => {
                let guard = tensor.read().expect("Lock should not be poisioned");
                let tensor = guard
                    .as_ref()
                    .map(|tensor| tensor.clone().pad_next_power_of_two());

                TensorHandle::Tensor(InnerTensor {
                    storage_key: storage_key.clone(),
                    store: store.clone(),
                    tensor: Arc::new(RwLock::new(tensor)),
                    shape: shape.next_power_of_two(),
                    unpadded_shape: unpadded_shape.next_power_of_two(),
                })
            }
        }
    }

    pub(crate) fn max_abs(&self) -> anyhow::Result<T> {
        match self {
            TensorHandle::WrappedTensor(inner_wrapped_tensor) => Ok(inner_wrapped_tensor
                .wrapped_tensor
                .read()
                .expect("Lock should not be posioned")
                .clone()
                .context("Wrapped tensor is dried")?
                .max_abs()
                .get_data()[0]),
            TensorHandle::Tensor(inner_tensor) => Ok(inner_tensor
                .tensor
                .read()
                .expect("Lock should not be poisioned")
                .as_ref()
                .context("Tensor is dried")?
                .max_abs()),
        }
    }
}

impl TensorHandle<f32> {
    pub(crate) fn mean_center_rows(&self) -> Self {
        match self {
            TensorHandle::WrappedTensor(InnerWrappedTensor {
                storage_key,
                store,
                wrapped_tensor,
                shape,
                unpadded_shape,
            }) => {
                let guard = wrapped_tensor.read().expect("Lock should not be poisioned");
                let wrapped_tensor = guard
                    .as_ref()
                    .map(|wrapped_tensor| wrapped_tensor.clone().mean_center_rows());

                TensorHandle::WrappedTensor(InnerWrappedTensor {
                    storage_key: storage_key.clone(),
                    store: store.clone(),
                    wrapped_tensor: Arc::new(RwLock::new(wrapped_tensor)),
                    shape: shape.clone(),
                    unpadded_shape: unpadded_shape.clone(),
                })
            }
            TensorHandle::Tensor(InnerTensor {
                storage_key,
                store,
                tensor,
                shape,
                unpadded_shape,
            }) => {
                let guard = tensor.read().expect("Lock should not be poisioned");
                let tensor = guard
                    .as_ref()
                    .map(|tensor| tensor.clone().mean_center_rows());

                TensorHandle::Tensor(InnerTensor {
                    storage_key: storage_key.clone(),
                    store: store.clone(),
                    tensor: Arc::new(RwLock::new(tensor)),
                    shape: shape.clone(),
                    unpadded_shape: unpadded_shape.clone(),
                })
            }
        }
    }
}

impl<T> TensorHandle<T>
where
    T: Serialize + for<'a> Deserialize<'a> + TensorTypeParam,
{
    pub(crate) fn from_wrapped_tensor(
        storage_key: StorageKey<Vec<T>>,
        store: GenStore,
        wrapped_tensor: WrappedTensor<T>,
    ) -> Self {
        let shape = Shape::from(wrapped_tensor.shape());
        let unpadded_shape = Shape::from(wrapped_tensor.unpadded_shape());
        Self::WrappedTensor(InnerWrappedTensor {
            storage_key,
            store,
            wrapped_tensor: Arc::new(RwLock::new(Some(wrapped_tensor))),
            shape,
            unpadded_shape,
        })
    }

    pub(crate) fn from_wrapped_tensor_with_unpadded_shape(
        storage_key: StorageKey<Vec<T>>,
        store: GenStore,
        mut wrapped_tensor: WrappedTensor<T>,
        unpadded_shape: Shape,
    ) -> Self {
        let shape = Shape::from(wrapped_tensor.shape());
        wrapped_tensor.set_unpadded_shape(unpadded_shape.clone().into());
        Self::WrappedTensor(InnerWrappedTensor {
            storage_key,
            store,
            wrapped_tensor: Arc::new(RwLock::new(Some(wrapped_tensor))),
            shape,
            unpadded_shape,
        })
    }
}

impl Quantize for TensorHandle<f32> {
    type Output = TensorHandle<Element>;

    fn quantize(&self, scaling: &ScalingFactor) -> Self::Output {
        match self {
            TensorHandle::WrappedTensor(InnerWrappedTensor {
                storage_key,
                store,
                wrapped_tensor,
                shape,
                unpadded_shape,
            }) => {
                let guard = wrapped_tensor.read().expect("Lock should not be posioned");
                let wrapped_tensor = guard
                    .as_ref()
                    .map(|wrapped_tensor| wrapped_tensor.quantize(scaling));
                TensorHandle::WrappedTensor(InnerWrappedTensor {
                    storage_key: storage_key.cast(),
                    store: store.clone(),
                    wrapped_tensor: Arc::new(RwLock::new(wrapped_tensor)),
                    shape: shape.clone(),
                    unpadded_shape: unpadded_shape.clone(),
                })
            }
            TensorHandle::Tensor(InnerTensor {
                storage_key,
                store,
                tensor,
                shape,
                unpadded_shape,
            }) => {
                let guard = tensor.read().expect("Lock should not be posioned");
                let tensor = guard.as_ref().map(|tensor| tensor.quantize(scaling));
                TensorHandle::Tensor(InnerTensor {
                    storage_key: storage_key.cast(),
                    store: store.clone(),
                    tensor: Arc::new(RwLock::new(tensor)),
                    shape: shape.clone(),
                    unpadded_shape: unpadded_shape.clone(),
                })
            }
        }
    }
}

impl Quantize for TensorHandle<Element> {
    type Output = TensorHandle<Element>;

    fn quantize(&self, scaling: &ScalingFactor) -> Self::Output {
        match self {
            TensorHandle::WrappedTensor(InnerWrappedTensor {
                storage_key,
                store,
                wrapped_tensor,
                shape,
                unpadded_shape,
            }) => {
                let guard = wrapped_tensor.read().expect("Lock should not be posioned");
                let wrapped_tensor = guard
                    .as_ref()
                    .map(|wrapped_tensor| wrapped_tensor.quantize(scaling));
                TensorHandle::WrappedTensor(InnerWrappedTensor {
                    storage_key: storage_key.cast(),
                    store: store.clone(),
                    wrapped_tensor: Arc::new(RwLock::new(wrapped_tensor)),
                    shape: shape.clone(),
                    unpadded_shape: unpadded_shape.clone(),
                })
            }
            TensorHandle::Tensor(InnerTensor {
                storage_key,
                store,
                tensor,
                shape,
                unpadded_shape,
            }) => {
                let guard = tensor.read().expect("Lock should not be posioned");
                let tensor = guard.as_ref().map(|tensor| tensor.quantize(scaling));
                TensorHandle::Tensor(InnerTensor {
                    storage_key: storage_key.cast(),
                    store: store.clone(),
                    tensor: Arc::new(RwLock::new(tensor)),
                    shape: shape.clone(),
                    unpadded_shape: unpadded_shape.clone(),
                })
            }
        }
    }
}

impl<T> From<KeyedTensor<T>> for TensorHandle<T>
where
    T: TensorTypeParam,
{
    fn from(value: KeyedTensor<T>) -> Self {
        let (tensor, storage_key) = value.into_parts();
        TensorHandle::from_tensor(storage_key.cast(), GenStore::new_empty(), tensor)
    }
}

impl<T> TryFrom<TensorHandle<T>> for Tensor<T>
where
    T: TensorTypeParam,
{
    type Error = anyhow::Error;

    fn try_from(value: TensorHandle<T>) -> anyhow::Result<Self> {
        match value {
            TensorHandle::WrappedTensor(InnerWrappedTensor { wrapped_tensor, .. }) => {
                let guard = wrapped_tensor.read().expect("Lock should not be poisioned");
                guard
                    .as_ref()
                    .map(Tensor::try_from)
                    .context("TensorHandle::WrappedTensor is dry")
                    .flatten()
            }
            TensorHandle::Tensor(InnerTensor { tensor, .. }) => tensor
                .write()
                .expect("Lock should not be poisioned")
                .take()
                .context("TensorHandle::Tensor is dry"),
        }
    }
}

impl<T> TryFrom<&TensorHandle<T>> for Tensor<T>
where
    T: TensorTypeParam,
{
    type Error = anyhow::Error;

    fn try_from(value: &TensorHandle<T>) -> anyhow::Result<Self> {
        match value {
            TensorHandle::WrappedTensor(InnerWrappedTensor { wrapped_tensor, .. }) => {
                wrapped_tensor
                    .read()
                    .expect("Lock should not be poisioned")
                    .as_ref()
                    .map(Tensor::try_from)
                    .context("TensorHandle::WrappedTensor is dry")
                    .flatten()
            }
            TensorHandle::Tensor(InnerTensor { tensor, .. }) => tensor
                .read()
                .expect("Lock should not be poisioned")
                .clone()
                .context("TensorHandle::Tensor is dry"),
        }
    }
}

impl<T> TryFrom<&TensorHandle<T>> for Vec<T>
where
    T: TensorTypeParam,
{
    type Error = anyhow::Error;

    fn try_from(value: &TensorHandle<T>) -> anyhow::Result<Self> {
        match value {
            TensorHandle::WrappedTensor(InnerWrappedTensor { wrapped_tensor, .. }) => {
                wrapped_tensor
                    .read()
                    .expect("Lock should not be poisioned")
                    .as_ref()
                    .map(|wrapped_tensor| wrapped_tensor.get_data())
                    .context("TensorHandle::WrappedTensor is dry")
            }
            TensorHandle::Tensor(InnerTensor { tensor, .. }) => tensor
                .read()
                .expect("Lock should not be poisioned")
                .as_ref()
                .map(|tensor| tensor.data.clone())
                .context("TensorHandle::Tensor is dry"),
        }
    }
}

impl<T> TryFrom<&TensorHandle<T>> for KeyedTensor<T>
where
    T: TensorTypeParam,
{
    type Error = anyhow::Error;

    fn try_from(value: &TensorHandle<T>) -> anyhow::Result<Self> {
        match value {
            TensorHandle::WrappedTensor(InnerWrappedTensor {
                wrapped_tensor,
                storage_key,
                ..
            }) => wrapped_tensor
                .read()
                .expect("Lock should not be poisioned")
                .as_ref()
                .map(|wrapped_tensor| {
                    let tensor = Tensor::try_from(wrapped_tensor)?;
                    Ok(KeyedTensor::new(storage_key.cast(), tensor))
                })
                .context("TensorHandle::WrappedTensor is dry")
                .flatten(),
            TensorHandle::Tensor(InnerTensor {
                tensor,
                storage_key,
                ..
            }) => tensor
                .read()
                .expect("Lock should not be poisioned")
                .as_ref()
                .map(|tensor| Ok(KeyedTensor::new(storage_key.cast(), tensor.clone())))
                .context("TensorHandle::Tensor is dry")
                .flatten(),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::{
        ops::Deref,
        sync::{Arc, RwLock},
    };

    use crate::tensor::{
        TensorHandle, TensorTypeParam,
        handle::{InnerTensor, InnerWrappedTensor},
    };

    impl<T> TensorHandle<T>
    where
        T: TensorTypeParam,
    {
        /// Ensures this is a [Tensor] variant, copying data if available.
        pub(crate) fn tensor_variant(self) -> anyhow::Result<Self> {
            match self {
                result @ TensorHandle::Tensor { .. } => Ok(result),
                TensorHandle::WrappedTensor(InnerWrappedTensor {
                    storage_key,
                    store,
                    wrapped_tensor,
                    shape,
                    unpadded_shape,
                }) => {
                    let guard = wrapped_tensor.read().expect("Lock should not be poisioned");
                    let tensor = match guard.deref() {
                        Some(wrapped_tensor) => Some(wrapped_tensor.try_into()?),
                        None => None,
                    };
                    Ok(TensorHandle::Tensor(InnerTensor {
                        storage_key,
                        store,
                        tensor: Arc::new(RwLock::new(tensor)),
                        shape,
                        unpadded_shape,
                    }))
                }
            }
        }
    }
}
