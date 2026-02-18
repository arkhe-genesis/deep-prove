use std::path::PathBuf;

use thiserror::Error;

#[derive(Error, Debug)]
pub enum StoreError {
    #[error("`{0}` is not a directory")]
    NotADir(PathBuf),

    #[error("page is unknown")]
    KeyUnknown,

    #[error("remote key not found")]
    RemoteKeyNotFound,

    #[error("I/O error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("serialization failed: {0}")]
    SerializationError(#[from] rmp_serde::encode::Error),

    #[error("deserialization failed: {0}")]
    DeserializationError(#[from] rmp_serde::decode::Error),

    #[error("empty page")]
    EmptyStore,

    #[error("remote store failed: {0}")]
    RemoteStoreError(anyhow::Error),
}
