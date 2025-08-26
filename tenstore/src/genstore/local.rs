use crate::{StoreError, StoreKey};
use compact_str::CompactString;
use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap,
    io::{BufReader, BufWriter, Read, Write},
    num::NonZero,
    path::{Path, PathBuf},
};
use weight_lru::LruCache;

use super::GenericStore;

#[derive(Clone, Hash, PartialEq, Eq)]
struct Storage {
    file: PathBuf,
}
impl Storage {
    fn file_size(&self) -> u64 {
        std::fs::metadata(&self.file).unwrap().len()
    }
}

/// A disk-backed tensor store featuring a bounded memory cache of the most
/// accessed tensors.
pub struct LocalStore<P: AsRef<Path>> {
    /// Keep track of the storage details associated to a stored tensor.
    ///
    /// This is string-indexed instead of [`TensorKey`]-indexed because data
    /// of multiple type can be stored in the same place.
    storage: HashMap<CompactString, Storage>,
    /// A LRU cache of the serialized value of the data.
    cache: LruCache<CompactString, Vec<u8>>,
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
                "{}{:10} {:12} {}",
                if self.cache.contains(k) { "*" } else { " " },
                k,
                s.file_size(),
                s.file.display()
            )?;
        }
        Ok(())
    }
}
impl<P: AsRef<Path>> GenericStore for LocalStore<P> {
    fn clear(&mut self) {
        self.storage.clear();
        self.cache.clear();
    }

    fn fetch<T: Serialize + for<'a> Deserialize<'a>>(
        &mut self,
        k: &StoreKey<T>,
    ) -> Result<T, StoreError> {
        let k_str = k.to_key();
        let backing = self.storage.get(&k_str).cloned();
        if let Some(s) = backing {
            let data: T = rmp_serde::from_slice(self.cache.try_get_or_insert(k_str, || {
                // This is an over-allocation, as serialization will
                // typically compress, even if slightly, the content.
                let mut buffer = Vec::with_capacity(s.file_size() as usize);
                let mut reader =
                    BufReader::new(std::fs::File::open(s.file).map_err(StoreError::from)?);
                reader.read_to_end(&mut buffer).map_err(StoreError::from)?;

                let buffer_len = NonZero::new(buffer.len()).ok_or(StoreError::EmptyPage)?;

                Ok::<_, StoreError>((buffer, buffer_len))
            })?)?;

            Ok(data)
        } else {
            Err(StoreError::KeyUnknown)
        }
    }

    fn store<T: Serialize + for<'a> Deserialize<'a>>(
        &mut self,
        k: &StoreKey<T>,
        data: &T,
    ) -> Result<(), StoreError> {
        let k_str = k.to_key();

        let storage = self.storage.entry(k_str.clone()).or_insert_with(|| {
            let mut file = self.root.as_ref().to_path_buf();
            file.push(k.to_key());
            Storage { file }
        });

        let serialized = rmp_serde::to_vec(&data).map_err(StoreError::from)?;
        let weight = NonZero::new(serialized.len()).ok_or(StoreError::EmptyPage)?;
        BufWriter::new(std::fs::File::create(&storage.file).map_err(StoreError::from)?)
            .write_all(&serialized)?;
        self.cache.put(k_str, serialized, weight);

        Ok(())
    }
}
