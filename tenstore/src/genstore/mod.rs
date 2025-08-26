use std::{
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
};

use serde::{Deserialize, Serialize};
pub mod local;
use crate::{StoreError, StoreKey};

pub trait GenericStore {
    /// Completely drain this store.
    fn clear(&mut self);

    fn fetch<T: Serialize + for<'a> Deserialize<'a>>(
        &mut self,
        k: &StoreKey<T>,
    ) -> Result<T, StoreError>;

    fn store<T: Serialize + for<'a> Deserialize<'a>>(
        &mut self,
        k: &StoreKey<T>,
        data: &T,
    ) -> Result<(), StoreError>;

    fn store_many<T: Serialize + for<'a> Deserialize<'a>>(
        &mut self,
        data: &[(StoreKey<T>, impl AsRef<T>)],
    ) -> Result<(), StoreError> {
        for (k, d) in data.iter() {
            self.store(k, d.as_ref())?;
        }
        Ok(())
    }
}

#[derive(Clone)]
pub enum GenStore {
    /// A local-only store, rooted in some directory on disk.
    LocalStore(Arc<Mutex<local::LocalStore<PathBuf>>>),
    /// The temporary folder-based store is a dedicated enum, for it must keep
    /// ownership of the `TempDir`, otherwise it would get dropped and erased
    /// from disk.
    TmpStore(Arc<Mutex<local::LocalStore<tempfile::TempDir>>>),
}
impl std::fmt::Debug for GenStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GenStore::LocalStore(mutex) => write!(f, "{:?}", mutex.lock().unwrap()),
            GenStore::TmpStore(mutex) => write!(f, "{:?}", mutex.lock().unwrap()),
        }
    }
}
impl GenStore {
    pub fn new_local<S: AsRef<Path>>(root: S, cache_size: usize) -> Result<Self, StoreError> {
        Ok(GenStore::LocalStore(Arc::new(Mutex::new(
            local::LocalStore::new(root.as_ref().to_path_buf(), cache_size)?,
        ))))
    }

    pub fn new_temporary(cache_size: usize) -> Result<Self, StoreError> {
        Ok(GenStore::TmpStore(Arc::new(Mutex::new(
            local::LocalStore::new(tempfile::tempdir().unwrap(), cache_size)?,
        ))))
    }

    /// Create a new page of type `U` derived from `old_k` by mapping `f` on it.
    /// If it already exists, do nothing.
    pub fn cast<
        T: Serialize + for<'a> Deserialize<'a>,
        U: Serialize + for<'a> Deserialize<'a>,
        F: Fn(&T) -> U,
    >(
        &mut self,
        old_k: &StoreKey<T>,
        f: F,
    ) -> Result<StoreKey<U>, StoreError> {
        self.cast_and_fetch(old_k, f).map(|x| x.0)
    }

    /// Create and return a new page of type `U` derived from `old_k` by mapping
    /// `f` on it. If it already exists, do nothing.
    pub fn cast_and_fetch<
        T: Serialize + for<'a> Deserialize<'a>,
        U: Serialize + for<'a> Deserialize<'a>,
        F: Fn(&T) -> U,
    >(
        &mut self,
        old_k: &StoreKey<T>,
        f: F,
    ) -> Result<(StoreKey<U>, U), StoreError> {
        let new_k = old_k.cast::<U>();
        match self.fetch::<U>(&new_k) {
            Ok(data) => Ok((new_k, data)),
            Err(StoreError::KeyUnknown) => {
                let old_page = self.fetch::<T>(old_k)?;
                let new_page = f(&old_page);
                self.store(&new_k, &new_page)?;
                Ok((new_k, new_page))
            }
            Err(err) => Err(err),
        }
    }
}
impl Default for GenStore {
    fn default() -> Self {
        const TWO_HUNDRED_MB: usize = 200 * 1024 * 1024;
        Self::new_temporary(TWO_HUNDRED_MB).unwrap()
    }
}
impl GenericStore for GenStore {
    fn clear(&mut self) {
        match self {
            GenStore::LocalStore(store) => store.lock().unwrap().clear(),
            GenStore::TmpStore(store) => store.lock().unwrap().clear(),
        }
    }

    fn fetch<T: Serialize + for<'a> Deserialize<'a>>(
        &mut self,
        k: &StoreKey<T>,
    ) -> Result<T, StoreError> {
        match self {
            GenStore::LocalStore(store) => store.lock().unwrap().fetch(k),
            GenStore::TmpStore(store) => store.lock().unwrap().fetch(k),
        }
    }

    fn store<T: Serialize + for<'a> Deserialize<'a>>(
        &mut self,
        k: &StoreKey<T>,
        data: &T,
    ) -> Result<(), StoreError> {
        match self {
            GenStore::LocalStore(store) => store.lock().unwrap().store(k, data),
            GenStore::TmpStore(store) => store.lock().unwrap().store(k, data),
        }
    }
}
