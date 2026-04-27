//! Frozen constants + newtype key for the Prolly tree.
//!
//! The values here are part of the mnem format and MUST NOT drift across
//! implementations. A change to any of them is a wire-format-breaking
//! change that requires a `mnem/N+1` version bump and a migration story.

use core::fmt;

use serde::de::{self, Visitor};
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::id::{ChangeId, EdgeId, NodeId};

/// Width in bytes of a Prolly tree key.
///
/// 16 bytes = 128 bits, matching the stable-ID width used throughout mnem
/// (`NodeId`, `EdgeId`, `ChangeId`; see SPEC §2.3 and ).
pub const PROLLY_KEY_BYTES: usize = 16;

/// A 16-byte Prolly tree key - the unit the chunker and tree operate on.
///
/// This is a newtype over `[u8; 16]` with custom Serde impls that emit the
/// value as a CBOR byte string (major type 2) per SPEC §4.3. The default
/// serde derive would emit a CBOR array of sixteen `u8` integers, which
/// is incorrect for the mnem canonical form.
///
/// All four stable IDs (`NodeId`, `EdgeId`, `ChangeId`, `OperationId`)
/// convert into `ProllyKey` via `From`. Construct raw keys with
/// [`ProllyKey::new`] or the tuple literal `ProllyKey([u8; 16])`.
#[derive(Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ProllyKey(pub [u8; PROLLY_KEY_BYTES]);

impl ProllyKey {
    /// Wrap raw bytes.
    #[must_use]
    pub const fn new(bytes: [u8; PROLLY_KEY_BYTES]) -> Self {
        Self(bytes)
    }

    /// Borrow the underlying bytes.
    #[must_use]
    pub const fn as_bytes(&self) -> &[u8; PROLLY_KEY_BYTES] {
        &self.0
    }

    /// Extract the underlying bytes.
    #[must_use]
    pub const fn into_bytes(self) -> [u8; PROLLY_KEY_BYTES] {
        self.0
    }
}

impl fmt::Debug for ProllyKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ProllyKey(")?;
        for b in &self.0 {
            write!(f, "{b:02x}")?;
        }
        f.write_str(")")
    }
}

// ---------- Stable-ID conversions ----------

impl From<NodeId> for ProllyKey {
    fn from(v: NodeId) -> Self {
        Self(v.into_bytes())
    }
}

impl From<EdgeId> for ProllyKey {
    fn from(v: EdgeId) -> Self {
        Self(v.into_bytes())
    }
}

impl From<ChangeId> for ProllyKey {
    fn from(v: ChangeId) -> Self {
        Self(v.into_bytes())
    }
}

// ---------- Serde (byte-string wire form) ----------

impl Serialize for ProllyKey {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_bytes(&self.0)
    }
}

struct ProllyKeyVisitor;

impl<'de> Visitor<'de> for ProllyKeyVisitor {
    type Value = ProllyKey;

    fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("a 16-byte prolly key byte string")
    }

    fn visit_bytes<E: de::Error>(self, v: &[u8]) -> Result<Self::Value, E> {
        if v.len() != PROLLY_KEY_BYTES {
            return Err(E::invalid_length(v.len(), &"16"));
        }
        let mut arr = [0u8; PROLLY_KEY_BYTES];
        arr.copy_from_slice(v);
        Ok(ProllyKey(arr))
    }

    fn visit_borrowed_bytes<E: de::Error>(self, v: &'de [u8]) -> Result<Self::Value, E> {
        self.visit_bytes(v)
    }

    fn visit_byte_buf<E: de::Error>(self, v: Vec<u8>) -> Result<Self::Value, E> {
        self.visit_bytes(&v)
    }
}

impl<'de> Deserialize<'de> for ProllyKey {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        deserializer.deserialize_bytes(ProllyKeyVisitor)
    }
}

// ---------- Rolling-hash / chunk-size constants ----------

/// Size of the rolling-hash window. SPEC §5.2: 64 bytes = four keys.
pub const ROLLING_WINDOW_BYTES: usize = 64;

/// The 32-byte BLAKE3 keyed-hash key used by the rolling hash.
///
/// Fixed for `mnem/0.1`. MUST NOT be changed within this format version.
/// Bytes are the ASCII literal `"mnem-prolly-rh-1"` (16 bytes) followed
/// by 16 zero bytes (padding to BLAKE3's 32-byte key size).
///
/// See SPEC §5.2.
pub const ROLLING_KEY: [u8; 32] = [
    0x6d, 0x6e, 0x65, 0x6d, // 'm', 'n', 'e', 'm'
    0x2d, 0x70, 0x72, 0x6f, // '-', 'p', 'r', 'o'
    0x6c, 0x6c, 0x79, 0x2d, // 'l', 'l', 'y', '-'
    0x72, 0x68, 0x2d, 0x31, // 'r', 'h', '-', '1'
    0x00, 0x00, 0x00, 0x00, //  zero padding
    0x00, 0x00, 0x00, 0x00, //
    0x00, 0x00, 0x00, 0x00, //
    0x00, 0x00, 0x00, 0x00, //
];

/// Hard minimum entries per chunk. Below this, a boundary MUST NOT fire.
pub const MIN_ENTRIES_PER_CHUNK: usize = 16;

/// Target average entries per chunk. ~4 KiB on typical mnem payloads
/// (16-byte key + ~40-byte link + ~8-byte CBOR framing per entry).
pub const TARGET_AVG_ENTRIES_PER_CHUNK: usize = 64;

/// Hard maximum entries per chunk. At this count the chunker MUST
/// emit a boundary regardless of hash.
pub const MAX_ENTRIES_PER_CHUNK: usize = 512;

/// Boundary threshold for the rolling hash - integer form.
///
/// When `entries_in_chunk` is strictly between [`MIN_ENTRIES_PER_CHUNK`]
/// and [`MAX_ENTRIES_PER_CHUNK`], the chunker emits a boundary whenever
/// the 64-bit rolling hash value is `< THRESHOLD`.
///
/// Chosen so the per-entry boundary probability is `~1/48`, yielding an
/// expected chunk size of `MIN + 48 = 64` entries which matches
/// [`TARGET_AVG_ENTRIES_PER_CHUNK`].
pub const THRESHOLD: u64 = u64::MAX / 48;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codec::{from_canonical_bytes, to_canonical_bytes};

    #[test]
    fn rolling_key_is_ascii_prefix_plus_zeros() {
        assert_eq!(&ROLLING_KEY[..16], b"mnem-prolly-rh-1");
        assert!(ROLLING_KEY[16..].iter().all(|&b| b == 0));
    }

    #[test]
    fn threshold_probability_is_about_one_in_48() {
        let ratio = THRESHOLD as f64 / u64::MAX as f64;
        assert!((ratio - 1.0 / 48.0).abs() < 1e-9, "ratio was {ratio}");
    }

    #[test]
    fn bounds_are_ordered() {
        assert!(MIN_ENTRIES_PER_CHUNK < TARGET_AVG_ENTRIES_PER_CHUNK);
        assert!(TARGET_AVG_ENTRIES_PER_CHUNK < MAX_ENTRIES_PER_CHUNK);
    }

    #[test]
    fn prolly_key_cbor_round_trip_as_byte_string() {
        let original = ProllyKey([0xAB; PROLLY_KEY_BYTES]);
        let bytes = to_canonical_bytes(&original).expect("encode");
        // Header byte 0x50 = major-type-2 (byte string), length 16.
        assert_eq!(bytes[0], 0x50, "expected CBOR byte-string-16 prefix");
        assert_eq!(bytes.len(), 17);
        let decoded: ProllyKey = from_canonical_bytes(&bytes).expect("decode");
        assert_eq!(original, decoded);
    }

    #[test]
    fn node_id_converts_to_prolly_key() {
        let n = NodeId::from_bytes_raw([7u8; 16]);
        let k: ProllyKey = n.into();
        assert_eq!(k.as_bytes(), &[7u8; 16]);
    }
}
