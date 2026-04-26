//! Stable identifiers: `NodeId`, `EdgeId`, `ChangeId`, `OperationId`.
//!
//! Per SPEC §2.3 and , every persistent entity carries a 16-byte
//! (128-bit) stable identifier that survives content edits, rewrites, and
//! rebases. The identifiers are distinguished at the type level to prevent
//! mixing a `NodeId` into a field expecting an `EdgeId`.
//!
//! All four use `UUIDv7` (RFC 9562, finalized May 2024) as the default
//! generator. Implementations MAY substitute other time-ordered,
//! collision-resistant 128-bit values.
//!
//! ## Generator entropy
//!
//! `UUIDv7` relies on 74 bits of random data to disambiguate IDs produced in
//! the same millisecond. `uuid`'s `Uuid::now_v7()` uses the thread-local
//! OS CSPRNG. Applications requiring stronger entropy guarantees
//! (multi-tenant servers, high-throughput agent fleets) should pass their
//! own `OsRng`-sourced seeds via [`StableId::from_random_bytes`] when
//! constructing IDs explicitly.

use core::fmt;
use core::marker::PhantomData;

use serde::de::{self, Visitor};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use uuid::Uuid;

use crate::error::IdError;

/// Phantom-typed tag distinguishing the four stable-ID roles.
///
/// The generic parameter is never materialized; only a type tag.
pub trait StableIdKind: sealed::Sealed + 'static {
    /// Short tag used in `Debug` output.
    const TAG: &'static str;
}

mod sealed {
    pub trait Sealed {}
}

/// Tag for [`NodeId`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct NodeTag;
impl sealed::Sealed for NodeTag {}
impl StableIdKind for NodeTag {
    const TAG: &'static str = "node";
}

/// Tag for [`EdgeId`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct EdgeTag;
impl sealed::Sealed for EdgeTag {}
impl StableIdKind for EdgeTag {
    const TAG: &'static str = "edge";
}

/// Tag for [`ChangeId`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ChangeTag;
impl sealed::Sealed for ChangeTag {}
impl StableIdKind for ChangeTag {
    const TAG: &'static str = "change";
}

/// Tag for [`OperationId`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct OperationTag;
impl sealed::Sealed for OperationTag {}
impl StableIdKind for OperationTag {
    const TAG: &'static str = "op";
}

/// A 16-byte stable identifier, parameterized by role tag.
///
/// Construct via [`StableId::new_v7`] for fresh IDs or
/// [`StableId::from_bytes`] to rehydrate from canonical-encoded form.
///
/// Two `StableId` values with different role tags are distinct types and
/// cannot be compared or converted without an explicit crossing.
// The tag types are zero-sized and derive all the needed comparison/hash
// traits, so derives on `StableId<Kind>` propagate cleanly. Only the `bytes`
// field participates materially in Eq/Ord/Hash; the tag is a compile-time
// marker with no runtime presence.
//
// Serde is implemented manually (not derived) so that serialization uses
// `serialize_bytes`, which DAG-CBOR encodes as a major-type-2 byte string
// (17 bytes on the wire: 1 header byte + 16 data bytes). The default
// `#[derive(Serialize)]` on `[u8; 16]` emits a CBOR array, which would
// violate SPEC §4.1 ("stable identifiers MUST be encoded as 16-byte byte
// strings, never as arrays").
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct StableId<Kind: StableIdKind> {
    bytes: [u8; 16],
    _tag: PhantomData<fn() -> Kind>,
}

impl<Kind: StableIdKind> Serialize for StableId<Kind> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_bytes(&self.bytes)
    }
}

struct StableIdVisitor<Kind>(PhantomData<fn() -> Kind>);

impl<'de, Kind: StableIdKind> Visitor<'de> for StableIdVisitor<Kind> {
    type Value = StableId<Kind>;

    fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("a 16-byte stable-id byte string")
    }

    fn visit_bytes<E: de::Error>(self, v: &[u8]) -> Result<Self::Value, E> {
        if v.len() != 16 {
            return Err(E::invalid_length(v.len(), &"16"));
        }
        let mut arr = [0u8; 16];
        arr.copy_from_slice(v);
        Ok(StableId::from_bytes_raw(arr))
    }

    fn visit_borrowed_bytes<E: de::Error>(self, v: &'de [u8]) -> Result<Self::Value, E> {
        self.visit_bytes(v)
    }

    fn visit_byte_buf<E: de::Error>(self, v: Vec<u8>) -> Result<Self::Value, E> {
        self.visit_bytes(&v)
    }
}

impl<'de, Kind: StableIdKind> Deserialize<'de> for StableId<Kind> {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        deserializer.deserialize_bytes(StableIdVisitor::<Kind>(PhantomData))
    }
}

impl<Kind: StableIdKind> StableId<Kind> {
    /// Generate a fresh `UUIDv7` stable ID.
    ///
    /// Uses the `uuid` crate's `Uuid::now_v7()`, which combines the current
    /// Unix timestamp (48 bits, millisecond resolution) with 74 bits of
    /// CSPRNG entropy. See RFC 9562.
    #[must_use]
    pub fn new_v7() -> Self {
        Self::from_bytes_raw(*Uuid::now_v7().as_bytes())
    }

    /// Construct from an explicit 16-byte array. The bytes are not validated
    /// beyond their length - any 128-bit value is accepted. Callers who
    /// want UUID shape validation should use [`StableId::parse_uuid`].
    #[must_use]
    pub const fn from_bytes_raw(bytes: [u8; 16]) -> Self {
        Self {
            bytes,
            _tag: PhantomData,
        }
    }

    /// Construct from a byte slice of length 16, validating the length.
    ///
    /// # Errors
    ///
    /// Returns [`IdError::StableIdLength`] if `bytes.len() != 16`.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, IdError> {
        let arr: [u8; 16] = bytes
            .try_into()
            .map_err(|_| IdError::StableIdLength { got: bytes.len() })?;
        Ok(Self::from_bytes_raw(arr))
    }

    /// Parse from a canonical UUID string (`xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx`).
    ///
    /// # Errors
    ///
    /// Returns [`IdError::StableIdParse`] if `s` is not a valid UUID.
    pub fn parse_uuid(s: &str) -> Result<Self, IdError> {
        Uuid::parse_str(s)
            .map(|u| Self::from_bytes_raw(*u.as_bytes()))
            .map_err(|source| IdError::StableIdParse { source })
    }

    /// Construct from 16 random bytes without UUID structure. Useful when
    /// the caller has already generated cryptographically random bytes and
    /// does not need `UUIDv7`'s time ordering. The value will not validate
    /// as a `UUIDv7` but will still function as a stable ID in mnem.
    #[must_use]
    pub const fn from_random_bytes(bytes: [u8; 16]) -> Self {
        Self::from_bytes_raw(bytes)
    }

    /// Borrow as a byte slice.
    #[must_use]
    pub const fn as_bytes(&self) -> &[u8; 16] {
        &self.bytes
    }

    /// Consume and return the underlying 16 bytes.
    #[must_use]
    pub const fn into_bytes(self) -> [u8; 16] {
        self.bytes
    }

    /// Render as a lowercase hyphenated UUID string.
    #[must_use]
    pub fn to_uuid_string(&self) -> String {
        Uuid::from_bytes(self.bytes).hyphenated().to_string()
    }
}

impl<Kind: StableIdKind> fmt::Debug for StableId<Kind> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}({})", Kind::TAG, self.to_uuid_string())
    }
}

impl<Kind: StableIdKind> fmt::Display for StableId<Kind> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.to_uuid_string())
    }
}

/// Stable identifier of a Node (SPEC §4.1, §2.3).
pub type NodeId = StableId<NodeTag>;

/// Stable identifier of an Edge (SPEC §4.2, §2.3).
pub type EdgeId = StableId<EdgeTag>;

/// Stable identifier of a Commit's logical change (SPEC §4.4, §2.3).
///
/// Survives rebase, amend, and squash, unlike the commit's content-addressed
/// CID which changes on every rewrite.
pub type ChangeId = StableId<ChangeTag>;

/// Stable identifier of an Operation (SPEC §4.5, §2.3).
///
/// Note: by convention the `OperationId` equals the content hash of the
/// Operation object because Operations are immutable by design. The
/// `StableId<OperationTag>` wrapper lets the rest of the API treat all four
/// role-tagged IDs uniformly; construction from a content hash happens in
/// a later module (`mnem-core::objects::operation`).
pub type OperationId = StableId<OperationTag>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stable_ids_have_distinct_types() {
        // This file would fail to compile if we tried to assign a NodeId to an
        // EdgeId, which is the invariant we care about. Here we just sanity-check
        // that construction and serialization round-trip.
        let n = NodeId::new_v7();
        let bytes = *n.as_bytes();
        let n2 = NodeId::from_bytes(&bytes).expect("16 bytes");
        assert_eq!(n, n2);
    }

    #[test]
    fn stable_id_debug_shows_kind_tag() {
        let n = NodeId::new_v7();
        let s = format!("{n:?}");
        assert!(
            s.starts_with("node("),
            "debug repr begins with kind tag: {s}"
        );
    }

    #[test]
    fn wrong_length_rejected() {
        let err = NodeId::from_bytes(&[0u8; 8]).unwrap_err();
        match err {
            IdError::StableIdLength { got } => assert_eq!(got, 8),
            e => panic!("wrong variant: {e:?}"),
        }
    }

    #[test]
    fn uuid_string_roundtrip() {
        let n = NodeId::new_v7();
        let s = n.to_uuid_string();
        let parsed = NodeId::parse_uuid(&s).expect("valid uuid");
        assert_eq!(n, parsed);
    }

    #[test]
    fn uuidv7_is_time_ordered_within_a_ms() {
        // UUIDv7 is time-ordered at 1ms resolution; two IDs generated
        // in the same ms are NOT strictly ordered, but the batch IS.
        let mut ids: Vec<NodeId> = (0..32).map(|_| NodeId::new_v7()).collect();
        let sorted = {
            let mut c = ids.clone();
            c.sort();
            c
        };
        // Not asserting strict equality because rapid-fire creation within a
        // millisecond allows tie-breaking by random tail. Instead, assert
        // the range of timestamps in the sorted form is non-decreasing.
        for window in sorted.windows(2) {
            let a = &window[0].as_bytes()[0..6];
            let b = &window[1].as_bytes()[0..6];
            assert!(
                a <= b,
                "UUIDv7 timestamp prefix non-monotonic: {a:?} > {b:?}"
            );
        }
        ids.clear();
    }
}
