use serde::{Deserialize, Serialize};
use tenstore::StorageKey;

#[derive(
    Clone,
    Debug,
    PartialEq,
    Eq,
    Hash,
    Serialize,
    Deserialize,
    derive_more::From,
    derive_more::Into,
    derive_more::Display,
)]
#[display("{_0}")]
pub struct CommitmentId(String);

impl From<&str> for CommitmentId {
    fn from(value: &str) -> Self {
        value.to_string().into()
    }
}

impl<T> From<&StorageKey<T>> for CommitmentId {
    fn from(value: &StorageKey<T>) -> Self {
        Self(value.id().to_string())
    }
}

impl<T> From<StorageKey<T>> for CommitmentId {
    fn from(value: StorageKey<T>) -> Self {
        Self(value.id().to_string())
    }
}

impl<T> From<CommitmentId> for StorageKey<T> {
    fn from(value: CommitmentId) -> Self {
        StorageKey::<T>::new(value.0)
    }
}

impl<T> From<&CommitmentId> for StorageKey<T> {
    fn from(value: &CommitmentId) -> Self {
        StorageKey::<T>::new(value.0.clone())
    }
}
