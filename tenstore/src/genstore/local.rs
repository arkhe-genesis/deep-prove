use crate::{StorageKey, StoreError};
use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap,
    hash::{DefaultHasher, Hasher},
    io::{BufReader, BufWriter, Read, Write},
    num::NonZero,
    path::{Path, PathBuf},
};
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
struct InternalKey {
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
}

impl<T> From<&StorageKey<T>> for InternalKey {
    fn from(value: &StorageKey<T>) -> Self {
        Self {
            id: value.id().to_string(),
            kind: std::any::type_name::<T>(),
        }
    }
}

/// A disk-backed page store featuring a bounded memory cache of the most
/// accessed pages.
pub struct LocalStore<P: AsRef<Path>> {
    /// Keep track of the storage details associated to a stored page.
    ///
    /// This is string-indexed instead of [`StoreKey`]-indexed because data
    /// of multiple type can be stored in the same place.
    storage: HashMap<InternalKey, Storage>,

    /// A LRU cache of the serialized value of the data.
    cache: LruCache<InternalKey, Vec<u8>>,

    /// The root folder of where to store the file-backing of the data.
    root: P,
}

impl<P: AsRef<Path>> LocalStore<P> {
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

impl<P: AsRef<Path>> std::fmt::Debug for LocalStore<P> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for (k, s) in self.storage.iter() {
            writeln!(
                f,
                "{}{:?} {:12} {}",
                if self.cache.contains(k) { "*" } else { " " },
                k,
                s.file_size(),
                s.file.display()
            )?;
        }
        Ok(())
    }
}

impl<P: AsRef<Path>> LocalStore<P> {
    pub(crate) fn fetch<T>(&mut self, storage_key: &StorageKey<T>) -> Result<T, StoreError>
    where
        T: for<'a> Deserialize<'a>,
    {
        let key = InternalKey::from(storage_key);
        let backing = self.storage.get(&key).cloned();
        if let Some(storage) = backing {
            let data: T = rmp_serde::from_slice(self.cache.try_get_or_insert(key, || {
                // This is an over-allocation, as serialization will
                // typically compress, even if slightly, the content.
                let mut buffer = Vec::with_capacity(storage.file_size() as usize);
                let mut reader =
                    BufReader::new(std::fs::File::open(storage.file).map_err(StoreError::from)?);
                reader.read_to_end(&mut buffer).map_err(StoreError::from)?;

                let buffer_len = NonZero::new(buffer.len()).ok_or(StoreError::EmptyStore)?;

                Ok::<_, StoreError>((buffer, buffer_len))
            })?)?;

            Ok(data)
        } else {
            Err(StoreError::KeyUnknown)
        }
    }

    pub(crate) fn store<T>(
        &mut self,
        storage_key: &StorageKey<T>,
        data: &T,
    ) -> Result<(), StoreError>
    where
        T: Serialize,
    {
        let key = InternalKey::from(storage_key);

        let storage = self.storage.entry(key.clone()).or_insert_with(|| {
            let file = key.rooted_at(self.root.as_ref());
            Storage { file }
        });

        let serialized = rmp_serde::to_vec(&data).map_err(StoreError::from)?;
        let weight = NonZero::new(serialized.len()).ok_or(StoreError::EmptyStore)?;
        BufWriter::new(std::fs::File::create(&storage.file).map_err(StoreError::from)?)
            .write_all(&serialized)?;
        self.cache.put(key, serialized, weight);

        Ok(())
    }
}
