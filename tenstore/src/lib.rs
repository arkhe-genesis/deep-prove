use std::{
    fmt::{Debug, Display},
    hash::Hash,
    marker::PhantomData,
};

mod error;
mod genstore;

pub use error::StoreError;
pub use genstore::{GenStore, GenericStore};

/// Identifier for storage data.
#[derive(Clone)]
pub struct StorageKey<T> {
    /// User defined `ID`.
    id: String,

    /// The type of the data function as a namespace.
    ///
    /// This allows the same `id` to be reused for multiple types without
    /// conflicts.
    kind: PhantomData<T>,
}

impl<T> StorageKey<T> {
    /// Creates a new [StorageKey<T>].
    ///
    /// NOTE: The key itself does not guarantee the store is populated with
    /// data, since that may be its first use.
    pub fn new(id: String) -> Self {
        StorageKey {
            id,
            kind: PhantomData,
        }
    }

    /// Convert this key into one for the same ID, but over another data type.
    pub fn cast<U>(&self) -> StorageKey<U> {
        StorageKey {
            id: self.id.clone(),
            kind: PhantomData,
        }
    }

    /// Returns a reference to this key's `id`.
    pub fn id(&self) -> &str {
        &self.id
    }
}

impl<T> Debug for StorageKey<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StorageKey")
            .field("id", &self.id)
            .finish_non_exhaustive()
    }
}

impl<T> PartialEq for StorageKey<T> {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl<T> Eq for StorageKey<T> {}

impl<T> Hash for StorageKey<T> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        state.write(self.id.as_bytes());
        state.write(std::any::type_name::<T>().as_bytes());
    }
}

impl<T> Display for StorageKey<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}@{}", self.id, std::any::type_name::<T>())
    }
}

impl<T> From<StorageKey<T>> for String {
    fn from(value: StorageKey<T>) -> Self {
        value.id
    }
}
