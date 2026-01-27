use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Length of [`ResponseGet`]
pub const RESPONSE_GET_BYTES: usize = 8;
/// Length of [`ResponsePut`]
pub const RESPONSE_PUT_BYTES: usize = 1;

/// [`ResponsePut`] contains this single bit
const RESPONSE_PUT_BIT: u8 = 1;

/// Server request
#[derive(Debug, Serialize, Deserialize)]
pub enum Request {
    /// Get the data associated with the given run ID and storage key
    Get { run_id: Uuid, storage_key: String },
    /// Store the data under the given run ID and storage key
    Put {
        run_id: Uuid,
        storage_key: String,
        data_len: u64,
    },
    /// Clean-up the data stored under the given run ID
    CleanUp { run_id: Uuid },
}

#[derive(Debug)]
/// Response for [`Request::Get`]
pub struct ResponseGet {
    pub data_len: u64,
}

impl ResponseGet {
    pub fn to_bytes(&self) -> [u8; RESPONSE_GET_BYTES] {
        self.data_len.to_be_bytes()
    }

    pub fn from_bytes(bytes: [u8; RESPONSE_GET_BYTES]) -> Self {
        ResponseGet {
            data_len: u64::from_be_bytes(bytes),
        }
    }
}

#[derive(Debug)]
/// Response for [`Request::Put`]
pub struct ResponsePut;

impl ResponsePut {
    pub fn to_bytes(&self) -> [u8; RESPONSE_PUT_BYTES] {
        [RESPONSE_PUT_BIT]
    }

    pub fn try_from_bytes(bytes: &[u8; RESPONSE_PUT_BYTES]) -> Option<Self> {
        if let [RESPONSE_PUT_BIT] = bytes {
            Some(ResponsePut)
        } else {
            None
        }
    }
}
