//! [`Tombstone`] - logical "forget this node" marker (SPEC §4, mnem/0.2+).
//!
//! Agents periodically need to revoke a memory ("User said forget X")
//! without mutating the append-only, content-addressed node record. A
//! Tombstone is a small side-record stored on the [`View`] that records
//! the intent-to-forget: a [`NodeId`][crate::id::NodeId], the reason, and
//! the microsecond timestamp the tombstone was written.
//!
//! Semantics:
//!
//! - The underlying [`crate::objects::Node`] remains in the node Prolly
//!   tree. Its CID is unchanged. Prior commits that referenced it still
//!   resolve.
//! - Retrieval paths filter tombstoned nodes by default
//!   ([`crate::retrieve::Retriever::include_tombstoned`] opts out).
//! - Re-tombstoning the same `NodeId` is a no-op at the semantic level:
//!   the second call overwrites the first's reason and timestamp, but
//!   no additional state change is observable to a retrieve or to a
//!   subsequent `is_tombstoned` query.
//!
//! Documented in SPEC §4.10.
//!
//! [`View`]: crate::objects::View

use std::collections::BTreeMap;

use ipld_core::ipld::Ipld;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

/// A logical forget-marker attached to a [`NodeId`].
///
/// [`NodeId`]: crate::id::NodeId
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Tombstone {
    /// Free-form UTF-8 reason string. MAY be empty.
    pub reason: String,
    /// Microseconds since Unix epoch when the tombstone was recorded.
    pub tombstoned_at: u64,
    /// Forward-compat extension map (SPEC §3.2).
    pub extra: BTreeMap<String, Ipld>,
}

impl Tombstone {
    /// The `_kind` discriminator on the wire.
    pub const KIND: &'static str = "tombstone";

    /// Construct a tombstone with the given reason + timestamp.
    #[must_use]
    pub fn new(reason: impl Into<String>, tombstoned_at: u64) -> Self {
        Self {
            reason: reason.into(),
            tombstoned_at,
            extra: BTreeMap::new(),
        }
    }
}

// ---------------- Serde ----------------

#[derive(Serialize, Deserialize)]
struct TombstoneWire {
    #[serde(rename = "_kind")]
    kind: String,
    reason: String,
    tombstoned_at: u64,
    #[serde(flatten, default, skip_serializing_if = "BTreeMap::is_empty")]
    extra: BTreeMap<String, Ipld>,
}

impl Serialize for Tombstone {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        TombstoneWire {
            kind: Self::KIND.into(),
            reason: self.reason.clone(),
            tombstoned_at: self.tombstoned_at,
            extra: self.extra.clone(),
        }
        .serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for Tombstone {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let w = TombstoneWire::deserialize(deserializer)?;
        if w.kind != Self::KIND {
            return Err(serde::de::Error::custom(format!(
                "expected _kind='{}', got '{}'",
                Self::KIND,
                w.kind
            )));
        }
        Ok(Self {
            reason: w.reason,
            tombstoned_at: w.tombstoned_at,
            extra: w.extra,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codec::{from_canonical_bytes, to_canonical_bytes};

    #[test]
    fn tombstone_round_trip_byte_identity() {
        let t = Tombstone::new("user asked to forget", 1_700_000_000_000_000);
        let bytes = to_canonical_bytes(&t).unwrap();
        let decoded: Tombstone = from_canonical_bytes(&bytes).unwrap();
        assert_eq!(t, decoded);
        let bytes2 = to_canonical_bytes(&decoded).unwrap();
        assert_eq!(bytes, bytes2);
    }

    #[test]
    fn tombstone_kind_rejection() {
        let w = TombstoneWire {
            kind: "node".into(),
            reason: "x".into(),
            tombstoned_at: 0,
            extra: BTreeMap::new(),
        };
        let bytes = serde_ipld_dagcbor::to_vec(&w).unwrap();
        let err = serde_ipld_dagcbor::from_slice::<Tombstone>(&bytes).unwrap_err();
        assert!(err.to_string().contains("_kind"));
    }

    #[test]
    fn tombstone_empty_reason_round_trips() {
        let t = Tombstone::new("", 42);
        let bytes = to_canonical_bytes(&t).unwrap();
        let decoded: Tombstone = from_canonical_bytes(&bytes).unwrap();
        assert_eq!(t, decoded);
        assert!(decoded.reason.is_empty());
    }
}
