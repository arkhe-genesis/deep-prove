use std::path::PathBuf;

use thiserror::Error;

#[derive(Error, Debug)]
pub enum TenstoreError {
    #[error("`{0}` is not a directory")]
    NotADir(PathBuf),
    #[error("tensor is unknown")]
    KeyUnknown,
    #[error("key too long: {0} characters > 24")]
    KeyTooLong(usize),
    #[error("incompatible sizes: allocated {allocated}, but requested {provided}")]
    InvalidSize { allocated: usize, provided: usize },
    #[error("I/O error: {0}")]
    IoError(#[from] std::io::Error),
    #[error("serialization failed: {0}")]
    SerializationError(#[from] rmp_serde::encode::Error),
    #[error("deserialization failed: {0}")]
    DeserializationError(#[from] rmp_serde::decode::Error),
    #[error("empty tensor")]
    EmptyTensor,
}
