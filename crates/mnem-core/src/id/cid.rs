//! Content identifiers (`CIDv1`) per the [CID] spec.
//!
//! A [`Cid`] is the canonical address of a content-hashed object in mnem.
//! It bundles three self-describing fields: version (always v1), codec
//! (DAG-CBOR or raw), and multihash. Two CIDs are equal only when all
//! three match.
//!
//! and SPEC §2.2.
//!
//! [CID]: https://github.com/multiformats/cid

use core::fmt;
use core::str::FromStr;

use serde::{Deserialize, Serialize};

use crate::error::IdError;
use crate::id::multihash::Multihash;

/// Re-export the underlying error type so callers don't need a direct
/// `cid` crate dependency.
pub use cid::Error as CidError;

/// Multicodec code for DAG-CBOR (0x71). The default codec for mnem structured
/// objects (Nodes, Edges, Trees, Commits, Operations, Views).
pub const CODEC_DAG_CBOR: u64 = 0x71;

/// Multicodec code for Raw (0x55). Used for opaque blob content.
pub const CODEC_RAW: u64 = 0x55;

/// A content identifier - `CIDv1` wrapping a codec + multihash.
///
/// Internally a `cid::CidGeneric<64>` with buffer sized for any multihash in
/// mnem's allow-list.
///
/// # String form
///
/// The canonical text form is base32 lowercase with a `b` multibase prefix,
/// e.g. `bafkreicxxxxxxx…`. Display / `FromStr` use this encoding.
#[derive(Clone, Eq, PartialEq, Hash, Ord, PartialOrd, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Cid(cid::CidGeneric<64>);

impl Cid {
    /// Construct a `CIDv1` from a codec code and a multihash.
    #[must_use]
    pub const fn new(codec: u64, hash: Multihash) -> Self {
        Self(cid::Cid::new_v1(codec, hash.into_inner()))
    }

    /// The multicodec code of this CID (e.g. [`CODEC_DAG_CBOR`]).
    #[must_use]
    pub const fn codec(&self) -> u64 {
        self.0.codec()
    }

    /// The multihash addressing the content.
    ///
    /// Named `multihash` rather than `hash` so as not to shadow
    /// `std::hash::Hash::hash`, which the derived `Hash` impl uses internally.
    #[must_use]
    pub fn multihash(&self) -> Multihash {
        Multihash::from(*self.0.hash())
    }

    /// Serialize to the binary wire form: `<version-varint=1> <codec-varint> <multihash>`.
    #[must_use]
    pub fn to_bytes(&self) -> Vec<u8> {
        self.0.to_bytes()
    }

    /// Parse from the binary wire form.
    ///
    /// # Errors
    ///
    /// Returns [`IdError::Cid`] if the bytes are not a valid `CIDv1`.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, IdError> {
        cid::CidGeneric::<64>::try_from(bytes)
            .map(Self)
            .map_err(|source| IdError::Cid { source })
    }

    /// Parse from the multibase-encoded string form (`bafy…`, `z…`, etc.).
    ///
    /// # Errors
    ///
    /// Returns [`IdError::Cid`] if the string is not a valid CID.
    pub fn parse_str(s: &str) -> Result<Self, IdError> {
        cid::CidGeneric::<64>::from_str(s)
            .map(Self)
            .map_err(|source| IdError::Cid { source })
    }

    // `inner` / `into_inner` accessors are deliberately absent. If downstream
    // modules need to reach the underlying `cid::CidGeneric<64>`, add a
    // narrowly-scoped accessor at that time; pre-adding them tempts leaking
    // the underlying crate's types into public APIs .
}

impl fmt::Display for Cid {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // cid::Cid's Display uses base32 lowercase with `b` prefix by default,
        // which matches SPEC §2.2.
        fmt::Display::fmt(&self.0, f)
    }
}

impl fmt::Debug for Cid {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Cid({self})")
    }
}

impl From<cid::CidGeneric<64>> for Cid {
    fn from(c: cid::CidGeneric<64>) -> Self {
        Self(c)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dag_cbor_cid_round_trips() {
        let hash = Multihash::sha2_256(b"content");
        let original = Cid::new(CODEC_DAG_CBOR, hash);
        assert_eq!(original.codec(), CODEC_DAG_CBOR);

        let bytes = original.to_bytes();
        let decoded = Cid::from_bytes(&bytes).expect("decode");
        assert_eq!(original, decoded);

        let s = original.to_string();
        assert!(s.starts_with('b'), "expected multibase b prefix, got {s}");
        let decoded_str = Cid::parse_str(&s).expect("parse string");
        assert_eq!(original, decoded_str);
    }

    #[test]
    fn different_codecs_produce_distinct_cids() {
        let hash = Multihash::sha2_256(b"content");
        let dag = Cid::new(CODEC_DAG_CBOR, hash.clone());
        let raw = Cid::new(CODEC_RAW, hash);
        assert_ne!(dag, raw);
        assert_eq!(dag.multihash(), raw.multihash());
        assert_ne!(dag.codec(), raw.codec());
    }

    #[test]
    fn different_content_distinct_cids() {
        let a = Cid::new(CODEC_DAG_CBOR, Multihash::sha2_256(b"a"));
        let b = Cid::new(CODEC_DAG_CBOR, Multihash::sha2_256(b"b"));
        assert_ne!(a, b);
    }
}
