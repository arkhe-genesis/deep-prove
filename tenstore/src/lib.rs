mod error;
use std::{
    hash::Hash,
    marker::PhantomData,
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
};

pub use error::TenstoreError;
pub mod local;

use compact_str::CompactString;
use local::LocalStore;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug)]
/// Used to unequivocally address a tensor backing in a [`TensorStore`].
pub struct TensorKey<T> {
    /// An ID for this tensor, unique among tensors of a given type.
    id: CompactString,
    /// A marker of this tensor underlying data.
    t: PhantomData<T>,
}
impl<T> TensorKey<T> {
    /// Convert this key into one for the same ID, but over another data type.
    pub fn cast<U>(&self) -> TensorKey<U> {
        TensorKey {
            id: self.id.clone(),
            t: PhantomData,
        }
    }

    /// Create a new key for this tensor, ensuring its uniquenes across tensor
    /// types.
    #[allow(clippy::should_implement_trait)]
    pub fn from_str<S: AsRef<str>>(id: S) -> Self {
        TensorKey {
            id: id.as_ref().into(),
            t: PhantomData,
        }
    }

    pub(crate) fn to_key(&self) -> CompactString {
        CompactString::new(format!("{}-{}", self.id, std::any::type_name::<T>()))
    }
}
// PartialEq/Eq and Hash have to be written manually, because the derive-based
// versions are not smart enough to recognize that T does not have to be
// PartialEq/Eq & Hash either.
impl<T> PartialEq for TensorKey<T> {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}
impl<T> Eq for TensorKey<T> {}
impl<T> Hash for TensorKey<T> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        state.write(self.to_key().as_bytes());
    }
}

impl<T> std::fmt::Display for TensorKey<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}@{}", self.id, std::any::type_name::<T>())
    }
}

/// A [`TensorStore`] provides an interface to reserve, set, and fetch tensor
/// backings. It ensures that data sizes remain coherent all along their lifetime.
pub trait TensorStore {
    /// Completely drain this store.
    fn clear(&mut self);

    /// Return, if possible, the tensor backing data (`Vec<T>`) stored under `k`.
    fn fetch<T: Serialize + for<'a> Deserialize<'a>>(
        &mut self,
        k: &TensorKey<T>,
    ) -> Result<Vec<T>, TenstoreError>;

    /// Store, if possible, a tensor backing data (`Vec<T>`) under `k`.
    fn store<T: Serialize + for<'a> Deserialize<'a>>(
        &mut self,
        k: &TensorKey<T>,
        data: impl AsRef<[T]>,
    ) -> Result<(), TenstoreError>;

    /// Simultaneously reserve and fill the storage space for a new tensor, with
    /// the data provided, for multiple tensor backings.
    fn store_many<T: Serialize + for<'a> Deserialize<'a>>(
        &mut self,
        data: &[(TensorKey<T>, impl AsRef<[T]>)],
    ) -> Result<(), TenstoreError> {
        for (k, d) in data.iter() {
            self.store(k, d)?;
        }
        Ok(())
    }
}

#[derive(Clone)]
pub enum TenStore {
    /// A local-only store, rooted in some directory on disk.
    LocalStore(Arc<Mutex<local::LocalStore<PathBuf>>>),
    /// The temporary folder-based store is a dedicated enum, for it must keep
    /// ownership of the `TempDir`, otherwise it would get dropped and erased
    /// from disk.
    TmpStore(Arc<Mutex<local::LocalStore<tempfile::TempDir>>>),
}
impl TenStore {
    pub fn new_local<S: AsRef<Path>>(root: S, cache_size: usize) -> Result<Self, TenstoreError> {
        Ok(TenStore::LocalStore(Arc::new(Mutex::new(LocalStore::new(
            root.as_ref().to_path_buf(),
            cache_size,
        )?))))
    }

    pub fn new_temporary(cache_size: usize) -> Result<Self, TenstoreError> {
        Ok(TenStore::TmpStore(Arc::new(Mutex::new(LocalStore::new(
            tempfile::tempdir().unwrap(),
            cache_size,
        )?))))
    }

    /// Ensure that the tensor indexed by `new_k` exists. If it does not, create
    /// it by mapping `f` over the tensor `old_k`.
    pub fn cast<
        T: Serialize + for<'a> Deserialize<'a>,
        U: Serialize + for<'a> Deserialize<'a>,
        F: Fn(&T) -> U,
    >(
        &mut self,
        old_k: &TensorKey<T>,
        f: F,
    ) -> Result<TensorKey<U>, TenstoreError> {
        self.cast_and_fetch(old_k, f).map(|x| x.0)
    }

    /// Attempt to return the tensor indexed by `new_k`. If it does not exist,
    /// allocate a new tensor with the same size as `old_k`, and generate its
    /// data by mapping `f` unto it.
    pub fn cast_and_fetch<
        T: Serialize + for<'a> Deserialize<'a>,
        U: Serialize + for<'a> Deserialize<'a>,
        F: Fn(&T) -> U,
    >(
        &mut self,
        old_k: &TensorKey<T>,
        f: F,
    ) -> Result<(TensorKey<U>, Vec<U>), TenstoreError> {
        let new_k = old_k.cast::<U>();
        match self.fetch::<U>(&new_k) {
            Ok(data) => Ok((new_k, data)),
            Err(TenstoreError::KeyUnknown) => {
                let old_tensor = self.fetch::<T>(old_k)?;
                let new_tensor = old_tensor.iter().map(f).collect::<Vec<_>>();
                self.store(&new_k, &new_tensor)?;
                Ok((new_k, new_tensor))
            }
            Err(err) => Err(err),
        }
    }
}
impl Default for TenStore {
    fn default() -> Self {
        const TWO_HUNDRED_MB: usize = 200 * 1024 * 1024;
        Self::new_temporary(TWO_HUNDRED_MB).unwrap()
    }
}
impl TensorStore for TenStore {
    fn clear(&mut self) {
        match self {
            TenStore::LocalStore(store) => store.lock().unwrap().clear(),
            TenStore::TmpStore(store) => store.lock().unwrap().clear(),
        }
    }

    fn fetch<T: Serialize + for<'a> Deserialize<'a>>(
        &mut self,
        k: &TensorKey<T>,
    ) -> Result<Vec<T>, TenstoreError> {
        match self {
            TenStore::LocalStore(store) => store.lock().unwrap().fetch(k),
            TenStore::TmpStore(store) => store.lock().unwrap().fetch(k),
        }
    }

    fn store<T: Serialize + for<'a> Deserialize<'a>>(
        &mut self,
        k: &TensorKey<T>,
        data: impl AsRef<[T]>,
    ) -> Result<(), TenstoreError> {
        match self {
            TenStore::LocalStore(store) => store.lock().unwrap().store(k, data),
            TenStore::TmpStore(store) => store.lock().unwrap().store(k, data),
        }
    }
}
