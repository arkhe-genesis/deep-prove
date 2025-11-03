use std::{
    path::PathBuf,
    sync::{Arc, Mutex},
};

use serde::{Deserialize, Serialize};
pub mod local;
use crate::{StorageKey, StoreError};

pub trait GenericStore {
    /// Fetches the data specified by `storage_key`.
    fn fetch<T>(&self, storage_key: &StorageKey<T>) -> Result<T, StoreError>
    where
        T: for<'a> Deserialize<'a>;

    /// Saves the `data` under `storage_key`.
    fn store<T>(&self, storage_key: &StorageKey<T>, data: &T) -> Result<(), StoreError>
    where
        T: Serialize;
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
    pub fn new_temporary(cache_size: usize) -> Result<Self, StoreError> {
        Ok(GenStore::TmpStore(Arc::new(Mutex::new(
            local::LocalStore::new(tempfile::tempdir().unwrap(), cache_size)?,
        ))))
    }

    /// Create a new page of type `U` derived from `old_k` by mapping `f` on it.
    /// If it already exists, do nothing.
    pub fn cast<T, U, F>(
        &mut self,
        storage_key: &StorageKey<T>,
        op: F,
    ) -> Result<StorageKey<U>, StoreError>
    where
        T: for<'a> Deserialize<'a>,
        U: Serialize + for<'a> Deserialize<'a>,
        F: Fn(&T) -> U,
    {
        self.cast_and_fetch(storage_key, op).map(|x| x.0)
    }

    /// Create and return a new page of type `U` derived from `old_k` by mapping
    /// `f` on it. If it already exists, do nothing.
    pub fn cast_and_fetch<T, U, F>(
        &mut self,
        storage_key: &StorageKey<T>,
        op: F,
    ) -> Result<(StorageKey<U>, U), StoreError>
    where
        T: for<'a> Deserialize<'a>,
        U: Serialize + for<'a> Deserialize<'a>,
        F: Fn(&T) -> U,
    {
        let new_k = storage_key.cast::<U>();
        match self.fetch::<U>(&new_k) {
            Ok(data) => Ok((new_k, data)),
            Err(StoreError::KeyUnknown) => {
                let old_page = self.fetch::<T>(storage_key)?;
                let new_page = op(&old_page);
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
    fn fetch<T>(&self, storage_key: &StorageKey<T>) -> Result<T, StoreError>
    where
        T: for<'a> Deserialize<'a>,
    {
        match self {
            GenStore::LocalStore(store) => store.lock().unwrap().fetch(storage_key),
            GenStore::TmpStore(store) => store.lock().unwrap().fetch(storage_key),
        }
    }

    fn store<T>(&self, storage_key: &StorageKey<T>, data: &T) -> Result<(), StoreError>
    where
        T: Serialize,
    {
        match self {
            GenStore::LocalStore(store) => store.lock().unwrap().store(storage_key, data),
            GenStore::TmpStore(store) => store.lock().unwrap().store(storage_key, data),
        }
    }
}
