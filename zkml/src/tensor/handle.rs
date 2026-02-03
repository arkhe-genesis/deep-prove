use std::{
    ops::Deref,
    sync::{Arc, MappedRwLockReadGuard, RwLock, RwLockReadGuard},
};

use anyhow::bail;
use burn::tensor::{Shape as BShape, TensorData};
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use tenstore::{GenStore, GenericStore, StorageKey, StoreError};

use crate::{
    Shape, Tensor,
    tensor::{TensorTypeParam, WrappedTensor},
};

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
    WrappedTensor {
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
        #[serde(skip)]
        wrapped_tensor: Arc<RwLock<Option<WrappedTensor<T>>>>,

        /// The shape of the tensor.
        shape: Shape,
        unpadded_shape: Shape,
    },
    Tensor {
        /// A unique key for this tensor.
        storage_key: StorageKey<Vec<T>>,

        /// Storage used to save or load the underlying data.
        #[serde(skip)]
        store: GenStore,

        /// Tensor data, if available.
        ///
        /// If the tensor data is not available, the handler will try to hydrate it
        /// by reading from the corresponding store.
        #[serde(skip)]
        tensor: Arc<RwLock<Option<Tensor<T>>>>,

        /// The shape of the tensor.
        shape: Shape,
        unpadded_shape: Shape,
    },
}

impl<T> TensorHandle<T>
where
    T: TensorTypeParam + Serialize + for<'a> Deserialize<'a>,
{
    /// Creates a [TensorHandle] from a [Tensor].
    pub(crate) fn from_tensor(
        storage_key: StorageKey<Vec<T>>,
        store: GenStore,
        tensor: Tensor<T>,
    ) -> Self {
        let shape = tensor.shape().clone();
        let unpadded_shape = tensor.unpadded_shape().clone();
        Self::Tensor {
            storage_key,
            store,
            tensor: Arc::new(RwLock::new(Some(tensor))),
            shape,
            unpadded_shape,
        }
    }

    /// Ensures this is a [WrappedTensor] variant.
    pub(crate) fn into_wrapped_tensor(self) -> anyhow::Result<Self> {
        match self {
            result @ TensorHandle::WrappedTensor { .. } => Ok(result),
            TensorHandle::Tensor {
                storage_key,
                store,
                tensor,
                shape,
                unpadded_shape,
            } => {
                let guard = tensor.read().expect("Lock should not be poisioned");
                let wrapped_tensor = match guard.deref() {
                    Some(tensor) => Some(tensor.try_into()?),
                    None => None,
                };
                Ok(TensorHandle::WrappedTensor {
                    storage_key,
                    store,
                    wrapped_tensor: Arc::new(RwLock::new(wrapped_tensor)),
                    shape,
                    unpadded_shape,
                })
            }
        }
    }

    /// Ensures this is a [Tensor] variant.
    pub(crate) fn into_dry_tensor(self) -> anyhow::Result<Self> {
        match self {
            TensorHandle::WrappedTensor {
                storage_key,
                store,
                shape,
                unpadded_shape,
                ..
            } => Ok(TensorHandle::Tensor {
                storage_key,
                store,
                tensor: Arc::new(RwLock::new(None)),
                shape,
                unpadded_shape,
            }),
            result @ TensorHandle::Tensor { .. } => Ok(result),
        }
    }

    /// Returns the [StorageKey] used to identify the data in the store.
    pub(crate) fn storage_key(&self) -> &StorageKey<Vec<T>> {
        match self {
            TensorHandle::WrappedTensor { storage_key, .. } => storage_key,
            TensorHandle::Tensor { storage_key, .. } => storage_key,
        }
    }

    /// Returns a reference to the shape of this tensor.
    pub(crate) fn shape(&self) -> &Shape {
        match self {
            TensorHandle::WrappedTensor { shape, .. } => shape,
            TensorHandle::Tensor { shape, .. } => shape,
        }
    }

    /// Returns a reference to the unpadded shape of this tensor.
    pub(crate) fn unpadded_shape(&self) -> &Shape {
        match self {
            TensorHandle::WrappedTensor { unpadded_shape, .. } => unpadded_shape,
            TensorHandle::Tensor { unpadded_shape, .. } => unpadded_shape,
        }
    }

    /// Returns a reference to the unpadded shape of this tensor.
    pub(crate) fn store(&self) -> &GenStore {
        match self {
            TensorHandle::WrappedTensor { store, .. } => store,
            TensorHandle::Tensor { store, .. } => store,
        }
    }

    /// Sets the `store` for this handle.
    pub(crate) fn attach_store(&mut self, value: GenStore) {
        match self {
            TensorHandle::WrappedTensor { store, .. } => *store = value,
            TensorHandle::Tensor { store, .. } => *store = value,
        }
    }

    /// Utility to load data from the store.
    fn load(&self) -> anyhow::Result<()> {
        match self {
            TensorHandle::WrappedTensor {
                storage_key,
                store,
                wrapped_tensor,
                shape,
                ..
            } => {
                let mut guard = wrapped_tensor.write().expect("Lock should not be poisoned");

                if guard.is_none() {
                    let tensor = store.fetch(storage_key).map(|data| {
                        let data = TensorData::new(data, BShape::from(shape.clone()));
                        WrappedTensor::from_data(data)
                    })??;
                    *guard = Some(tensor);
                }
            }
            TensorHandle::Tensor {
                storage_key,
                store,
                tensor,
                shape,
                unpadded_shape,
            } => {
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
    /// Returns a reference to the cached [`Tensor`].
    ///
    /// NOTE: If the [`Tensor`] is not cached, this will load the data from
    /// the store.
    pub fn tensor(&self) -> anyhow::Result<MappedRwLockReadGuard<'_, Tensor<T>>> {
        match self {
            TensorHandle::WrappedTensor { .. } => {
                bail!("Tensor is unavailable for a wrapped tensor handler")
            }
            TensorHandle::Tensor { tensor, .. } => loop {
                {
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

    /// Returns a reference to the cached [`WrappedTensor`].
    ///
    /// NOTE: If the [`WrappedTensor`] is not cached, it will be created, to create
    /// the tensor the corresponding [`Tensor`] must be available, if it is not, the
    /// data will be loaded from the store.
    pub fn wrapped_tensor(&self) -> anyhow::Result<MappedRwLockReadGuard<'_, WrappedTensor<T>>> {
        match self {
            TensorHandle::WrappedTensor { wrapped_tensor, .. } => loop {
                {
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
                    self.load()?;
                }
            },
            TensorHandle::Tensor { .. } => {
                bail!("Wrapped tensor is unavailable for a tensor handler")
            }
        }
    }

    /// Dries the current handle.
    ///
    /// Drying a handle frees the cached values to free memory.
    pub(crate) fn dry(&self) {
        match self {
            TensorHandle::WrappedTensor { wrapped_tensor, .. } => {
                let mut guard = wrapped_tensor
                    .write()
                    .expect("Lock should not be poisioned");
                *guard = None;
            }
            TensorHandle::Tensor { tensor, .. } => {
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
            TensorHandle::WrappedTensor {
                storage_key,
                store,
                shape,
                unpadded_shape,
                ..
            } => {
                let storage_key =
                    store.cast(storage_key, |xs| xs.iter().map(&f).collect::<Vec<S>>())?;
                Ok(TensorHandle::<S>::WrappedTensor {
                    storage_key,
                    store: store.clone(),
                    wrapped_tensor: Default::default(),
                    shape: shape.clone(),
                    unpadded_shape: unpadded_shape.clone(),
                })
            }
            TensorHandle::Tensor {
                storage_key,
                store,
                shape,
                unpadded_shape,
                ..
            } => {
                let storage_key =
                    store.cast(storage_key, |xs| xs.iter().map(&f).collect::<Vec<S>>())?;
                Ok(TensorHandle::<S>::Tensor {
                    storage_key,
                    store: store.clone(),
                    tensor: Default::default(),
                    shape: shape.clone(),
                    unpadded_shape: unpadded_shape.clone(),
                })
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
        Self::WrappedTensor {
            storage_key,
            store,
            wrapped_tensor: Arc::new(RwLock::new(Some(wrapped_tensor))),
            shape,
            unpadded_shape,
        }
    }

    pub(crate) fn from_wrapped_tensor_with_unpadded_shape(
        storage_key: StorageKey<Vec<T>>,
        store: GenStore,
        mut wrapped_tensor: WrappedTensor<T>,
        unpadded_shape: Shape,
    ) -> Self {
        let shape = Shape::from(wrapped_tensor.shape());
        wrapped_tensor.set_unpadded_shape(unpadded_shape.clone().into());
        Self::WrappedTensor {
            storage_key,
            store,
            wrapped_tensor: Arc::new(RwLock::new(Some(wrapped_tensor))),
            shape,
            unpadded_shape,
        }
    }
}
