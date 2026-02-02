use crate::{LocalStore, StorageKey, StoreError};
use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap,
    fmt::Display,
    fs::remove_file,
    hash::{DefaultHasher, Hasher},
    io::{BufReader, BufWriter, Read, Write},
    num::NonZero,
    path::{Path, PathBuf},
};
use uuid::Uuid;
use weight_lru::LruCache;

#[derive(Clone, Hash, PartialEq, Eq)]
struct Storage {
    file: PathBuf,
}

impl Storage {
    fn file_size(&self) -> u64 {
        std::fs::metadata(&self.file).unwrap().len()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct InternalKey {
    // Note that this is not included in `to_string`, because in remote store server
    // we're using it separately as a dir to store key-vals that can be cleaned
    // up easily when the run completes
    run_id: Uuid,
    id: String,
    kind: &'static str,
}

impl InternalKey {
    fn rooted_at(&self, path: &Path) -> PathBuf {
        // The data is assumed to be trusted, and a CHF is not needed
        let mut hasher = DefaultHasher::new();
        hasher.write_usize(self.kind.len());
        hasher.write(self.kind.as_bytes());
        hasher.write_usize(self.id.len());
        hasher.write(self.id.as_bytes());
        let hash = hasher.finish();

        path.to_path_buf().join(hash.to_string())
    }

    pub fn from_storage_key_with_run_id<T>(run_id: Uuid, storage_key: &StorageKey<T>) -> Self {
        Self {
            run_id,
            id: storage_key.id().to_string(),
            kind: std::any::type_name::<T>(),
        }
    }
}

impl Display for InternalKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}:{}", self.id, self.kind)
    }
}

/// A disk-backed page store featuring a bounded memory cache of the most
/// accessed pages.
pub struct DiskStore<P> {
    /// Keep track of the storage details associated to a stored page.
    ///
    /// This is string-indexed instead of [`StoreKey`]-indexed because data
    /// of multiple type can be stored in the same place.
    storage: HashMap<Uuid, HashMap<InternalKey, Storage>>,

    /// A LRU cache of the serialized value of the data.
    cache: LruCache<InternalKey, Vec<u8>>,

    /// The root folder of where to store the file-backing of the data.
    root: P,
}

impl<P: AsRef<Path>> DiskStore<P> {
    pub fn new(root: P, max_cache_size: usize) -> Result<Self, StoreError> {
        const DEFAULT_CACHE_SIZE: NonZero<usize> = NonZero::new(1024 * 1024).expect("1MiB > 0");

        if root.as_ref().is_file() {
            return Err(StoreError::NotADir(root.as_ref().to_owned()));
        }
        std::fs::create_dir_all(root.as_ref()).map_err(StoreError::from)?;

        Ok(Self {
            storage: Default::default(),
            cache: LruCache::new(NonZero::new(max_cache_size).unwrap_or(DEFAULT_CACHE_SIZE)),
            root,
        })
    }
}

impl<P> std::fmt::Debug for DiskStore<P> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "L {:50} {:12} Filename", "ID", "Size")?;
        for (_run, ks) in self.storage.iter() {
            for (k, s) in ks.iter() {
                writeln!(
                    f,
                    "{} {k:50?} {:12} {}",
                    if self.cache.contains(k) { "*" } else { " " },
                    s.file_size(),
                    s.file.display()
                )?;
            }
        }
        Ok(())
    }
}

impl<P: AsRef<Path>> DiskStore<P> {
    /// Fetch and decode data associated with the given run ID and key
    pub(crate) fn fetch<T>(
        &mut self,
        run_id: Uuid,
        storage_key: &StorageKey<T>,
    ) -> Result<T, StoreError>
    where
        T: for<'a> Deserialize<'a>,
    {
        let bytes = self.fetch_bytes(run_id, storage_key)?;
        Ok(rmp_serde::from_slice(bytes)?)
    }

    /// Encode and store data under the given run ID and key
    pub(crate) fn store<T>(
        &mut self,
        run_id: Uuid,
        storage_key: &StorageKey<T>,
        data: &T,
    ) -> Result<(), StoreError>
    where
        T: Serialize,
    {
        let serialized = rmp_serde::to_vec(&data).map_err(StoreError::from)?;
        self.store_bytes(run_id, storage_key, serialized)
    }

    /// Clean-up all files stored for the given run ID
    pub(crate) fn clean_up(&mut self, run_id: Uuid) -> Result<(), StoreError> {
        if let Some(storage) = self.storage.remove(&run_id) {
            for storage in storage.values() {
                remove_file(&storage.file)?;
            }
        }
        Ok(())
    }

    /// Fetch data associated with the given run ID and key
    fn fetch_bytes<T>(
        &mut self,
        run_id: Uuid,
        storage_key: &StorageKey<T>,
    ) -> Result<&Vec<u8>, StoreError> {
        let key = InternalKey::from_storage_key_with_run_id(run_id, storage_key);
        self.fetch_bytes_internal(key)
    }

    /// Store data under the given run ID and key
    fn store_bytes<T>(
        &mut self,
        run_id: Uuid,
        storage_key: &StorageKey<T>,
        data: Vec<u8>,
    ) -> Result<(), StoreError> {
        let key = InternalKey::from_storage_key_with_run_id(run_id, storage_key);
        self.store_bytes_internal(key, data)
    }

    /// Check if the storage contains the given internal key
    fn contains_internal(&mut self, key: &InternalKey) -> bool {
        let backing = self
            .storage
            .get(&key.run_id)
            .and_then(|ks| ks.get(key))
            .cloned();
        if let Some(storage) = backing {
            self.cache.contains(key) || storage.file.exists()
        } else {
            false
        }
    }

    /// Fetch data associated with the given  internal key
    fn fetch_bytes_internal(&mut self, key: InternalKey) -> Result<&Vec<u8>, StoreError> {
        let backing = self
            .storage
            .get(&key.run_id)
            .and_then(|ks| ks.get(&key))
            .cloned();
        if let Some(storage) = backing {
            let data = self.cache.try_get_or_insert(key, || {
                // This is an over-allocation, as serialization will
                // typically compress, even if slightly, the content.
                let mut buffer = Vec::with_capacity(storage.file_size() as usize);
                let mut reader =
                    BufReader::new(std::fs::File::open(storage.file).map_err(StoreError::from)?);
                reader.read_to_end(&mut buffer).map_err(StoreError::from)?;

                let buffer_len = NonZero::new(buffer.len()).ok_or(StoreError::EmptyStore)?;

                Ok::<_, StoreError>((buffer, buffer_len))
            })?;

            Ok(data)
        } else {
            Err(StoreError::KeyUnknown)
        }
    }

    /// Store data under the given internal key
    fn store_bytes_internal(&mut self, key: InternalKey, data: Vec<u8>) -> Result<(), StoreError> {
        let storage = self
            .storage
            .entry(key.run_id)
            .or_default()
            .entry(key.clone())
            .or_insert_with(|| {
                let file = key.rooted_at(self.root.as_ref());
                Storage { file }
            });

        let weight = NonZero::new(data.len()).ok_or(StoreError::EmptyStore)?;
        BufWriter::new(std::fs::File::create(&storage.file).map_err(StoreError::from)?)
            .write_all(&data)?;
        self.cache.put(key, data, weight);

        Ok(())
    }
}

impl<P: AsRef<Path>> LocalStore for DiskStore<P> {
    type Error = StoreError;
    type Key = InternalKey;

    fn contains(&mut self, storage_key: &Self::Key) -> bool {
        self.contains_internal(storage_key)
    }

    fn fetch(&mut self, storage_key: Self::Key) -> anyhow::Result<&Vec<u8>, Self::Error> {
        self.fetch_bytes_internal(storage_key)
    }

    fn store(&mut self, storage_key: Self::Key, data: Vec<u8>) -> anyhow::Result<(), Self::Error> {
        self.store_bytes_internal(storage_key, data)
    }

    fn clean_up(&mut self, run_id: Uuid) -> anyhow::Result<(), Self::Error> {
        self.clean_up(run_id)
    }
}
