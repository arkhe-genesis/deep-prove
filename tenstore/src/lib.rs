use compact_str::CompactString;
use std::{hash::Hash, marker::PhantomData};

mod error;
mod genstore;

pub use error::StoreError;
pub use genstore::{GenStore, GenericStore};

/// A `TensorKey` wraps a `StoreKey`, ensuring is addresses a vector of values.
pub type TensorKey<T> = StoreKey<Vec<T>>;

#[derive(Clone)]
/// Used to unequivocally address a page backing in a [`GenericStore`].
pub struct StoreKey<T> {
    /// An ID for this page, unique among pages of a given type.
    id: CompactString,
    /// A marker of this page underlying data.
    t: PhantomData<T>,
}
impl<T> StoreKey<T> {
    /// Convert this key into one for the same ID, but over another data type.
    pub fn cast<U>(&self) -> StoreKey<U> {
        StoreKey {
            id: self.id.clone(),
            t: PhantomData,
        }
    }

    /// Create a new key for this page, ensuring its uniquenes across data
    /// types.
    #[allow(clippy::should_implement_trait)]
    pub fn from_str<S: AsRef<str>>(id: S) -> Self {
        StoreKey {
            id: id.as_ref().into(),
            t: PhantomData,
        }
    }

    pub(crate) fn to_key(&self) -> CompactString {
        CompactString::new(format!("{}-{}", self.id, std::any::type_name::<T>()))
    }
}
impl<T> std::fmt::Debug for StoreKey<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.id)
    }
}
// PartialEq/Eq and Hash have to be written manually, because the derive-based
// versions are not smart enough to recognize that T does not have to be
// PartialEq/Eq & Hash either.
impl<T> PartialEq for StoreKey<T> {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}
impl<T> Eq for StoreKey<T> {}
impl<T> Hash for StoreKey<T> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        state.write(self.to_key().as_bytes());
    }
}

impl<T> std::fmt::Display for StoreKey<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}@{}", self.id, std::any::type_name::<T>())
    }
}
