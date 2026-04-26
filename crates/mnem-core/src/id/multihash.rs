//! Self-describing hashes per the [multihash] spec.
//!
//! Every content hash in mnem is a `Multihash` - the algorithm code is part
//! of the hash value, so the default hash algorithm can change in a future
//! mnem version without invalidating existing content. //!
//! [multihash]: https://github.com/multiformats/multihash

use core::fmt;

use serde::{Deserialize, Serialize};

use crate::error::IdError;

// Re-export the underlying crate's error so callers don't need a direct
// `multihash` dependency.
pub use multihash::Error as MultihashError;

/// Multihash algorithm code for SHA-256 (32-byte digest). Default for mnem/0.1.
pub const HASH_SHA2_256: u64 = 0x12;

/// Multihash algorithm code for BLAKE3-256 (32-byte digest).
///
/// Available when `multihash-codetable/blake3` is enabled in the dep graph,
/// which mnem-core does by default. BLAKE3 is a candidate default for a
/// future mnem format version
pub const HASH_BLAKE3_256: u64 = 0x1e;

/// A content hash tagged with its algorithm code.
///
/// Internally wraps `multihash::Multihash<64>` - 64 bytes of buffer, enough
/// for any algorithm on mnem's allow-list (SHA-256, BLAKE3-256 today;
/// SHA-512, BLAKE3-512, etc. if later expanded).
///
/// # Equality
///
/// Two `Multihash` values are equal only if they have the same algorithm
/// code AND the same digest bytes. A SHA-256 and a BLAKE3-256 of the same
/// input never compare equal even if their digest bytes collide.
#[derive(Clone, Eq, PartialEq, Hash, Ord, PartialOrd, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Multihash(multihash::Multihash<64>);

impl Multihash {
    /// Compute a SHA-256 multihash of `bytes`.
    ///
    /// This is the default content-hash algorithm for mnem/0.1.
    #[must_use]
    pub fn sha2_256(bytes: &[u8]) -> Self {
        use multihash_codetable::{Code, MultihashDigest};
        Self(Code::Sha2_256.digest(bytes))
    }

    /// Compute a BLAKE3-256 multihash of `bytes`.
    ///
    /// Produces a 32-byte digest with algorithm code `0x1e`. Faster than
    /// SHA-256 in most conditions; accepted by mnem but not the default
    /// for mnem/0.1 .
    #[must_use]
    pub fn blake3_256(bytes: &[u8]) -> Self {
        use multihash_codetable::{Code, MultihashDigest};
        Self(Code::Blake3_256.digest(bytes))
    }

    /// Wrap a raw algorithm code + digest bytes into a `Multihash`.
    ///
    /// # Errors
    ///
    /// Returns [`IdError::Multihash`] if `digest.len() > 64` or if the
    /// digest length is otherwise invalid for the declared code.
    pub fn wrap(code: u64, digest: &[u8]) -> Result<Self, IdError> {
        multihash::Multihash::<64>::wrap(code, digest)
            .map(Self)
            .map_err(|source| IdError::Multihash { source })
    }

    /// The algorithm code byte. See [`HASH_SHA2_256`] and friends.
    #[must_use]
    pub const fn code(&self) -> u64 {
        self.0.code()
    }

    /// Digest length in bytes.
    #[must_use]
    pub const fn size(&self) -> u8 {
        self.0.size()
    }

    /// Borrow the digest bytes.
    #[must_use]
    pub fn digest(&self) -> &[u8] {
        self.0.digest()
    }

    /// Serialize to the multihash wire format: `<code: varint> <size: varint> <digest>`.
    #[must_use]
    pub fn to_bytes(&self) -> Vec<u8> {
        self.0.to_bytes()
    }

    /// Parse a multihash from its wire format.
    ///
    /// # Errors
    ///
    /// Returns [`IdError::Multihash`] if the bytes are malformed.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, IdError> {
        multihash::Multihash::<64>::from_bytes(bytes)
            .map(Self)
            .map_err(|source| IdError::Multihash { source })
    }

    /// Consume and return the underlying `multihash::Multihash<64>`.
    /// Crate-internal interop with [`crate::id::cid`].
    #[must_use]
    pub(crate) const fn into_inner(self) -> multihash::Multihash<64> {
        self.0
    }
}

impl fmt::Debug for Multihash {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "multihash(code=0x{:x}, size={}, digest=",
            self.code(),
            self.size()
        )?;
        for b in self.digest().iter().take(4) {
            write!(f, "{b:02x}")?;
        }
        f.write_str("…)")
    }
}

impl From<multihash::Multihash<64>> for Multihash {
    fn from(m: multihash::Multihash<64>) -> Self {
        Self(m)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sha2_256_deterministic() {
        let a = Multihash::sha2_256(b"hello");
        let b = Multihash::sha2_256(b"hello");
        assert_eq!(a, b);
        assert_eq!(a.code(), HASH_SHA2_256);
        assert_eq!(a.size(), 32);
        assert_eq!(a.digest().len(), 32);
    }

    #[test]
    fn different_inputs_different_hashes() {
        let a = Multihash::sha2_256(b"hello");
        let b = Multihash::sha2_256(b"world");
        assert_ne!(a, b);
    }

    #[test]
    fn different_algos_distinct_even_on_empty() {
        let sha = Multihash::sha2_256(&[]);
        let blake = Multihash::blake3_256(&[]);
        assert_ne!(
            sha, blake,
            "sha2-256 and blake3 of empty must not compare equal"
        );
        assert_ne!(sha.code(), blake.code());
    }

    #[test]
    fn wire_round_trip() {
        let original = Multihash::sha2_256(b"round-trip me");
        let bytes = original.to_bytes();
        let decoded = Multihash::from_bytes(&bytes).expect("decode");
        assert_eq!(original, decoded);
        // Header: 0x12 (sha2-256) then 0x20 (32) then 32 bytes digest
        assert_eq!(bytes[0], 0x12);
        assert_eq!(bytes[1], 0x20);
        assert_eq!(bytes.len(), 34);
    }

    #[test]
    fn wrap_roundtrip() {
        let digest = [0xabu8; 32];
        let m = Multihash::wrap(HASH_SHA2_256, &digest).expect("wrap");
        assert_eq!(m.code(), HASH_SHA2_256);
        assert_eq!(m.digest(), &digest);
    }
}
