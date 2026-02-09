use crate::{StorageKey, StoreError, local::DiskStore, remote::RemoteStore};
use serde::{Deserialize, Serialize};
use std::{
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
};
use uuid::Uuid;

pub trait GenericStore {
    /// Prefetches the data specified by `storage_key`.
    fn prefetch<T>(&self, storage_key: &StorageKey<T>) -> Result<(), StoreError>
    where
        T: for<'a> Deserialize<'a>;

    /// Fetches the data specified by `storage_key`.
    fn fetch<T>(&self, storage_key: &StorageKey<T>) -> Result<T, StoreError>
    where
        T: for<'a> Deserialize<'a>;

    /// Saves the `data` under `storage_key`.
    fn store<T>(&self, storage_key: &StorageKey<T>, data: &T) -> Result<(), StoreError>
    where
        T: Serialize;
}

#[derive(Clone, Debug)]
pub struct GenStore {
    kind: GenStoreKind,
    /// Unique run ID is used to differentiate keys from distinct runs
    run_id: Uuid,
}

#[derive(Clone)]
pub enum GenStoreKind {
    /// A local-only store, rooted in some directory on disk.
    Local(Arc<Mutex<DiskStore<PathBuf>>>),

    /// The temporary folder-based store is a dedicated enum, for it must keep
    /// ownership of the `TempDir`, otherwise it would get dropped and erased
    /// from disk.
    Tmp(Arc<Mutex<DiskStore<tempfile::TempDir>>>),

    /// A remote store combined with a local-store for caching.
    Remote(Arc<Mutex<RemoteStore>>),

    /// An empty store
    Empty,
}

impl std::fmt::Debug for GenStoreKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GenStoreKind::Local(mutex) => write!(f, "{:?}", mutex.lock().unwrap()),
            GenStoreKind::Tmp(mutex) => write!(f, "{:?}", mutex.lock().unwrap()),
            GenStoreKind::Remote(mutex) => write!(f, "{:?}", mutex.lock().unwrap()),
            GenStoreKind::Empty => write!(f, "GenStore::Empty"),
        }
    }
}

impl GenStore {
    pub fn new_kind(kind: GenStoreKind) -> Self {
        Self {
            kind,
            run_id: Uuid::new_v4(),
        }
    }

    pub fn new_empty() -> Self {
        Self {
            kind: GenStoreKind::Empty,
            run_id: Uuid::new_v4(),
        }
    }

    pub fn new_temporary(cache_size: usize) -> Result<Self, StoreError> {
        let kind = GenStoreKind::Tmp(Arc::new(Mutex::new(DiskStore::new(
            tempfile::tempdir().unwrap(),
            cache_size,
        )?)));
        Ok(Self::new_kind(kind))
    }

    pub fn new_local(root: PathBuf, cache_size: usize) -> Result<Self, StoreError> {
        let kind = GenStoreKind::Local(Arc::new(Mutex::new(DiskStore::new(root, cache_size)?)));
        Ok(Self::new_kind(kind))
    }

    pub fn new_remote<P>(
        root: P,
        cache_size: usize,
        server_addr: url::Url,
    ) -> Result<Self, StoreError>
    where
        P: 'static + AsRef<Path> + Send,
    {
        let kind = GenStoreKind::Remote(Arc::new(Mutex::new(
            RemoteStore::new(root, cache_size, server_addr)
                .map_err(StoreError::RemoteStoreError)?,
        )));
        Ok(Self::new_kind(kind))
    }

    /// Create a new page of type `U` derived from `old_k` by mapping `f` on it.
    /// If it already exists, do nothing.
    pub fn cast<T, U, F>(
        &self,
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
        &self,
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

    /// Clean-up data from the current run.
    pub fn clean_up(self) -> Result<(), StoreError> {
        match &self.kind {
            GenStoreKind::Remote(store) => {
                store.lock().unwrap().clean_up(self.run_id)?;
            }
            GenStoreKind::Local(store) => {
                store.lock().unwrap().clean_up(self.run_id)?;
            }
            GenStoreKind::Tmp(store) => {
                store.lock().unwrap().clean_up(self.run_id)?;
            }
            GenStoreKind::Empty => panic!("Cant call cleanup on empty store"),
        }
        Ok(())
    }

    /// Returns a copy of `self` referring to a new run ID.
    pub fn start_new_run(&self) -> Self {
        let mut new_store = self.clone();
        new_store.run_id = Uuid::new_v4();
        new_store
    }
}

impl Default for GenStore {
    fn default() -> Self {
        const TWO_HUNDRED_MB: usize = 200 * 1024 * 1024;
        Self::new_temporary(TWO_HUNDRED_MB).unwrap()
    }
}

impl GenericStore for GenStore {
    fn prefetch<T>(&self, storage_key: &StorageKey<T>) -> Result<(), StoreError>
    where
        T: for<'a> Deserialize<'a>,
    {
        match &self.kind {
            GenStoreKind::Local(_) | GenStoreKind::Tmp(_) | GenStoreKind::Empty => {
                // No-op
                Ok(())
            }
            GenStoreKind::Remote(store) => store.lock().unwrap().prefetch(self.run_id, storage_key),
        }
    }

    fn fetch<T>(&self, storage_key: &StorageKey<T>) -> Result<T, StoreError>
    where
        T: for<'a> Deserialize<'a>,
    {
        match &self.kind {
            GenStoreKind::Local(store) => store.lock().unwrap().fetch(self.run_id, storage_key),
            GenStoreKind::Tmp(store) => store.lock().unwrap().fetch(self.run_id, storage_key),
            GenStoreKind::Remote(store) => store.lock().unwrap().fetch(self.run_id, storage_key),
            GenStoreKind::Empty => panic!("Cant call fetch on empty store"),
        }
    }

    fn store<T>(&self, storage_key: &StorageKey<T>, data: &T) -> Result<(), StoreError>
    where
        T: Serialize,
    {
        match &self.kind {
            GenStoreKind::Local(store) => {
                store.lock().unwrap().store(self.run_id, storage_key, data)
            }
            GenStoreKind::Tmp(store) => store.lock().unwrap().store(self.run_id, storage_key, data),
            GenStoreKind::Remote(store) => {
                store.lock().unwrap().store(self.run_id, storage_key, data)
            }
            GenStoreKind::Empty => panic!("Cant call store on empty store"),
        }
    }
}
