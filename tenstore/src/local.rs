use std::{
    collections::HashMap,
    io::{BufReader, BufWriter, Read, Write},
    num::NonZero,
    path::{Path, PathBuf},
};

use compact_str::CompactString;
use serde::{Deserialize, Serialize};
use weight_lru::LruCache;

use crate::{TensorKey, TensorStore, TenstoreError};

#[derive(Clone, Hash, PartialEq, Eq)]
struct Storage {
    allocated: usize,
    file: PathBuf,
}

/// A disk-backed tensor store featuring a bounded memory cache of the most
/// accessed tensors.
pub struct LocalStore<P: AsRef<Path>> {
    /// Keep track of the storage details associated to a stored tensor.
    ///
    /// This is string-indexed instead of [`TensorKey`]-indexed because tensors
    /// of multiple type can be stored in the same place.
    storage: HashMap<CompactString, Storage>,
    /// A LRU cache of the serialized value of the tensors.
    cache: LruCache<CompactString, Vec<u8>>,
    /// The root folder of where to store the file-backing of the tensors.
    root: P,
}
impl<P: AsRef<Path>> LocalStore<P> {
    pub fn new(root: P, max_cache_size: usize) -> Result<Self, TenstoreError> {
        const DEFAULT_CACHE_SIZE: NonZero<usize> = NonZero::new(1024 * 1024).expect("1MiB > 0");

        if root.as_ref().is_file() {
            return Err(TenstoreError::NotADir(root.as_ref().to_owned()));
        }
        std::fs::create_dir_all(root.as_ref()).map_err(TenstoreError::from)?;

        Ok(Self {
            storage: Default::default(),
            cache: LruCache::new(NonZero::new(max_cache_size).unwrap_or(DEFAULT_CACHE_SIZE)),
            root,
        })
    }
}
impl<P: AsRef<Path>> std::fmt::Debug for LocalStore<P> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for (k, Storage { allocated, file }) in self.storage.iter() {
            writeln!(
                f,
                "{}{:10} {:12} {}",
                if self.cache.contains(k) { "*" } else { " " },
                k,
                allocated,
                file.display()
            )?;
        }
        Ok(())
    }
}
impl<P: AsRef<Path>> TensorStore for LocalStore<P> {
    fn clear(&mut self) {
        self.storage.clear();
        self.cache.clear();
    }

    fn fetch<T: Serialize + for<'a> Deserialize<'a>>(
        &mut self,
        k: &TensorKey<T>,
    ) -> Result<Vec<T>, TenstoreError> {
        let k_str = k.to_key();
        let backing = self.storage.get(&k_str).cloned();
        if let Some(Storage { allocated, file }) = backing {
            let data: Vec<T> =
                rmp_serde::from_slice(self.cache.try_get_or_insert(k_str, || {
                    // This is an over-allocation, as serialization will
                    // typically compress, even if slightly, the content.
                    let mut buffer = Vec::with_capacity(allocated * std::mem::size_of::<T>());
                    let mut reader =
                        BufReader::new(std::fs::File::open(file).map_err(TenstoreError::from)?);
                    reader
                        .read_to_end(&mut buffer)
                        .map_err(TenstoreError::from)?;

                    let buffer_len =
                        NonZero::new(buffer.len()).ok_or(TenstoreError::EmptyTensor)?;

                    Ok::<_, TenstoreError>((buffer, buffer_len))
                })?)?;

            if data.len() != allocated {
                return Err(TenstoreError::InvalidSize {
                    allocated,
                    provided: data.len(),
                });
            }
            Ok(data)
        } else {
            Err(TenstoreError::KeyUnknown)
        }
    }

    fn store<T: Serialize + for<'a> Deserialize<'a>>(
        &mut self,
        k: &TensorKey<T>,
        data: impl AsRef<[T]>,
    ) -> Result<(), TenstoreError> {
        let k_str = k.to_key();
        let data = data.as_ref();
        let provided = data.len();

        let Storage { allocated, file } = self.storage.entry(k_str.clone()).or_insert_with(|| {
            let mut file = self.root.as_ref().to_path_buf();
            file.push(k.to_key());
            Storage {
                allocated: data.len(),
                file,
            }
        });

        if *allocated != provided {
            return Err(TenstoreError::InvalidSize {
                allocated: *allocated,
                provided,
            });
        }

        let serialized = rmp_serde::to_vec(&data).map_err(TenstoreError::from)?;
        let weight = NonZero::new(serialized.len()).ok_or(TenstoreError::EmptyTensor)?;
        BufWriter::new(std::fs::File::create(file).map_err(TenstoreError::from)?)
            .write_all(&serialized)
            .map_err(TenstoreError::from)?;

        self.cache.put(k_str, serialized, weight);

        Ok(())
    }
}
