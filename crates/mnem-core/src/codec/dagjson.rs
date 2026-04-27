//! DAG-JSON - a human-readable debug / export codec.
//!
//! **Never hashed, never canonical.** DAG-JSON is useful for
//! `mnem cat-file --json`, test-vector inspection, and error messages.
//! It must never appear on the hash input path - SPEC §3 says canonical
//! encoding is DAG-CBOR and nothing else.

use bytes::Bytes;
use serde::{Serialize, de::DeserializeOwned};

use crate::error::CodecError;

/// Encode a value as DAG-JSON for debug / inspection.
///
/// # Errors
///
/// Returns [`CodecError::Encode`] if the value cannot be serialized.
pub fn to_json_bytes<T: Serialize>(value: &T) -> Result<Bytes, CodecError> {
    serde_ipld_dagjson::to_vec(value)
        .map(Bytes::from)
        .map_err(|e| CodecError::Encode(e.to_string()))
}

/// Decode a value from DAG-JSON bytes.
///
/// # Errors
///
/// Returns [`CodecError::Decode`] if the bytes are malformed or do not
/// match the target type `T`.
pub fn from_json_bytes<T: DeserializeOwned>(bytes: &[u8]) -> Result<T, CodecError> {
    serde_ipld_dagjson::from_slice(bytes).map_err(|e| CodecError::Decode(e.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::id::NodeId;
    use serde::{Deserialize, Serialize};

    #[derive(Serialize, Deserialize, PartialEq, Debug)]
    struct Fixture {
        id: NodeId,
        label: String,
    }

    #[test]
    fn json_round_trip_restores_value() {
        let original = Fixture {
            id: NodeId::from_bytes_raw([3u8; 16]),
            label: "debug".into(),
        };
        let bytes = to_json_bytes(&original).expect("encode");
        let decoded: Fixture = from_json_bytes(&bytes).expect("decode");
        assert_eq!(original, decoded);
    }
}
