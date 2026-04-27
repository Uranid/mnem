//! View object (SPEC §4.6) - a snapshot of the mutable state of a repo.
//!
//! Carries:
//!
//! - `heads` - current head commits (≥1, or 0 for the root View per §7.5)
//! - `refs` - named references (branches, tags) as a map of name → [`RefTarget`]
//! - `remote_refs` - optional per-remote named references
//! - `wc_commit` - optional working-copy pointer
//!
//! `RefTargets` are either `Normal(Cid)` or `Conflicted { adds, removes }`;
//! see SPEC §4.6 and amendments.

use std::collections::BTreeMap;

use ipld_core::ipld::Ipld;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::id::{Cid, NodeId};
use crate::objects::tombstone::Tombstone;

/// A named reference in a [`View`].
///
/// Per SPEC §4.6 the on-wire form has a `kind` discriminator:
///
/// ```text
/// { "kind": "normal",     "target": Link<Commit> }
/// { "kind": "conflicted", "adds": [Link], "removes": [Link] }
/// ```
///
/// Canonical form for `Conflicted`: `adds` and `removes` MUST each be
/// strictly ascending by CID byte representation, with no duplicates
/// and not both empty. See SPEC §4.6 amendments.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "lowercase")]
pub enum RefTarget {
    /// A single Commit target.
    Normal {
        /// The commit this ref points at.
        target: Cid,
    },
    /// Unresolved concurrent-update state. Canonical form sorts
    /// `adds` and `removes` ascending.
    Conflicted {
        /// Candidate new targets.
        adds: Vec<Cid>,
        /// Previously-observed targets removed on one side of the merge.
        removes: Vec<Cid>,
    },
}

impl RefTarget {
    /// Construct a normal ref pointing at `target`.
    #[must_use]
    pub const fn normal(target: Cid) -> Self {
        Self::Normal { target }
    }

    /// Construct a conflicted ref, sorting `adds` and `removes`
    /// canonically.
    #[must_use]
    pub fn conflicted(mut adds: Vec<Cid>, mut removes: Vec<Cid>) -> Self {
        adds.sort();
        adds.dedup();
        removes.sort();
        removes.dedup();
        Self::Conflicted { adds, removes }
    }
}

/// A snapshot of the repository's mutable state at a single instant.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct View {
    /// Current head commits.
    pub heads: Vec<Cid>,
    /// Named references.
    pub refs: BTreeMap<String, RefTarget>,
    /// Per-remote named references. Outer key is the remote name
    /// (e.g. `"origin"` matching a `[remote.origin]` section in
    /// `.mnem/config.toml`); inner map is that remote's server-side
    /// refs (e.g. `"refs/heads/main"` → `RefTarget`) as observed on
    /// the last `mnem fetch`. PR 2 on the remote-transport track
    /// adds the [`View::with_tracking_ref`] / [`View::tracking_ref`]
    /// helpers; PR 3 wires up the actual network fetch. Absent /
    /// empty maps are omitted from the wire encoding so pre-0.3
    /// Views round-trip byte-identically. See .
    ///
    ///
    pub remote_refs: Option<BTreeMap<String, BTreeMap<String, RefTarget>>>,
    /// Working-copy commit pointer.
    pub wc_commit: Option<Cid>,
    /// Logical "forget this node" markers (SPEC §4.10, mnem/0.2+).
    ///
    /// Maps a [`NodeId`] to the [`Tombstone`] record that revoked it.
    /// The underlying Node block stays in the node Prolly tree; its CID
    /// is unchanged. Retrieval paths filter out tombstoned nodes by
    /// default (see
    /// [`crate::retrieve::Retriever::include_tombstoned`]).
    ///
    /// Re-tombstoning the same `NodeId` overwrites the previous entry.
    /// Store shape mirrors `remote_refs`: inline `BTreeMap`, encoded as
    /// an optional list that is skipped on the wire when empty. That
    /// keeps pre-0.2 Views byte-identical after a round-trip through a
    /// newer decoder.
    pub tombstones: BTreeMap<NodeId, Tombstone>,
    /// Forward-compat extension map (SPEC §3.2).
    pub extra: BTreeMap<String, Ipld>,
}

impl Default for View {
    fn default() -> Self {
        Self::new()
    }
}

impl View {
    /// The `_kind` discriminator on the wire.
    pub const KIND: &'static str = "view";

    /// An empty View (no heads, no refs). The root View of a freshly-
    /// initialized repository (SPEC §7.5).
    #[must_use]
    pub const fn new() -> Self {
        Self {
            heads: Vec::new(),
            refs: BTreeMap::new(),
            remote_refs: None,
            wc_commit: None,
            tombstones: BTreeMap::new(),
            extra: BTreeMap::new(),
        }
    }

    /// Add a head commit. Returns `self` for chaining.
    #[must_use]
    pub fn with_head(mut self, head: Cid) -> Self {
        self.heads.push(head);
        self
    }

    /// Add a named ref. Returns `self` for chaining.
    #[must_use]
    pub fn with_ref(mut self, name: impl Into<String>, target: RefTarget) -> Self {
        self.refs.insert(name.into(), target);
        self
    }

    /// Record a tracking ref for a named remote, e.g. after a `mnem
    /// fetch origin` converges the server's `refs/heads/main` to a
    /// local `origin/main` pointer. `remote` is the short name
    /// registered in `.mnem/config.toml` (`[remote.origin]`),
    /// `ref_name` is the server-side refname (`refs/heads/main`),
    /// and `target` is the Commit CID the remote had for it at fetch
    /// time. Subsequent fetches overwrite.
    ///
    /// Returns `self` for chaining. Lazily allocates the
    /// `remote_refs` map; empty Views still encode to the pre-0.3
    /// byte sequence (the map is omitted when empty).
    #[must_use]
    pub fn with_tracking_ref(
        mut self,
        remote: impl Into<String>,
        ref_name: impl Into<String>,
        target: Cid,
    ) -> Self {
        let remote = remote.into();
        let ref_name = ref_name.into();
        let rt = RefTarget::normal(target);
        let map = self.remote_refs.get_or_insert_with(BTreeMap::new);
        map.entry(remote).or_default().insert(ref_name, rt);
        self
    }

    /// Convenience accessor: the tracking ref the last `mnem fetch`
    /// recorded for a `{remote, ref_name}` pair, or `None` if the
    /// remote is unknown or does not carry that ref. Mirrors the
    /// Git behaviour of `git rev-parse origin/main`.
    #[must_use]
    pub fn tracking_ref(&self, remote: &str, ref_name: &str) -> Option<&RefTarget> {
        self.remote_refs.as_ref()?.get(remote)?.get(ref_name)
    }
}

// ---------------- Serde for View ----------------

/// On-wire shape for a single tombstone entry on a View. A list of
/// these (sorted ascending by `node_id`) is the canonical encoding of
/// [`View::tombstones`]. We don't key the CBOR map by `NodeId` directly
/// because DAG-CBOR requires string keys for maps; a list-of-records is
/// the idiomatic shape for `Map<bytes, struct>` in this codec.
#[derive(Serialize, Deserialize)]
struct TombstoneEntry {
    node_id: NodeId,
    #[serde(flatten)]
    tombstone: Tombstone,
}

#[derive(Serialize, Deserialize)]
struct ViewWire {
    #[serde(rename = "_kind")]
    kind: String,
    heads: Vec<Cid>,
    refs: BTreeMap<String, RefTarget>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    remote_refs: Option<BTreeMap<String, BTreeMap<String, RefTarget>>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    wc_commit: Option<Cid>,
    /// Sorted-ascending list of tombstone entries. Sorted on `NodeId`
    /// bytes so two Views with the same logical tombstone set encode
    /// byte-identically (determinism contract, same rule as `refs`).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    tombstones: Vec<TombstoneEntry>,
    #[serde(flatten, default, skip_serializing_if = "BTreeMap::is_empty")]
    extra: BTreeMap<String, Ipld>,
}

impl Serialize for View {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        // BTreeMap iteration is already sorted, so the emitted list is
        // sorted-ascending by NodeId bytes.
        let tombstones: Vec<TombstoneEntry> = self
            .tombstones
            .iter()
            .map(|(id, ts)| TombstoneEntry {
                node_id: *id,
                tombstone: ts.clone(),
            })
            .collect();
        ViewWire {
            kind: Self::KIND.into(),
            heads: self.heads.clone(),
            refs: self.refs.clone(),
            remote_refs: self.remote_refs.clone(),
            wc_commit: self.wc_commit.clone(),
            tombstones,
            extra: self.extra.clone(),
        }
        .serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for View {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let w = ViewWire::deserialize(deserializer)?;
        if w.kind != Self::KIND {
            return Err(serde::de::Error::custom(format!(
                "expected _kind='{}', got '{}'",
                Self::KIND,
                w.kind
            )));
        }
        let mut tombstones = BTreeMap::new();
        for entry in w.tombstones {
            tombstones.insert(entry.node_id, entry.tombstone);
        }
        Ok(Self {
            heads: w.heads,
            refs: w.refs,
            remote_refs: w.remote_refs,
            wc_commit: w.wc_commit,
            tombstones,
            extra: w.extra,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codec::{from_canonical_bytes, to_canonical_bytes};
    use crate::id::{CODEC_RAW, Multihash};

    fn raw(n: u32) -> Cid {
        Cid::new(CODEC_RAW, Multihash::sha2_256(&n.to_be_bytes()))
    }

    #[test]
    fn empty_view_round_trip() {
        let original = View::new();
        let bytes = to_canonical_bytes(&original).unwrap();
        let decoded: View = from_canonical_bytes(&bytes).unwrap();
        assert_eq!(original, decoded);
    }

    #[test]
    fn view_with_heads_and_refs_round_trip() {
        let v = View::new()
            .with_head(raw(1))
            .with_ref("refs/heads/main", RefTarget::normal(raw(1)))
            .with_ref(
                "refs/heads/feature",
                RefTarget::conflicted(vec![raw(2), raw(3)], vec![raw(1)]),
            );
        let bytes = to_canonical_bytes(&v).unwrap();
        let decoded: View = from_canonical_bytes(&bytes).unwrap();
        assert_eq!(v, decoded);
    }

    #[test]
    fn conflicted_ref_sorts_adds_and_removes() {
        let r = RefTarget::conflicted(vec![raw(3), raw(1), raw(2)], vec![raw(5), raw(4)]);
        match r {
            RefTarget::Conflicted { adds, removes } => {
                assert!(adds.windows(2).all(|w| w[0] < w[1]));
                assert!(removes.windows(2).all(|w| w[0] < w[1]));
            }
            _ => panic!(),
        }
    }

    #[test]
    fn ref_target_normal_round_trip() {
        let r = RefTarget::normal(raw(42));
        let bytes = to_canonical_bytes(&r).unwrap();
        let decoded: RefTarget = from_canonical_bytes(&bytes).unwrap();
        assert_eq!(r, decoded);
    }

    #[test]
    fn view_with_tracking_refs_round_trip() {
        // Tracking-refs field on View survives a DAG-CBOR round trip
        // unchanged; exercises `remote_refs` on the wire shape and
        // the `with_tracking_ref` / `tracking_ref` helpers that PR 2
        // added on the remote-transport track.
        let v = View::new()
            .with_head(raw(1))
            .with_ref("refs/heads/main", RefTarget::normal(raw(1)))
            .with_tracking_ref("origin", "refs/heads/main", raw(10))
            .with_tracking_ref("origin", "refs/heads/feature", raw(11))
            .with_tracking_ref("backup", "refs/heads/main", raw(20));

        let bytes = to_canonical_bytes(&v).unwrap();
        let decoded: View = from_canonical_bytes(&bytes).unwrap();
        assert_eq!(v, decoded);

        // Helper accessors survive the round-trip identically.
        assert_eq!(
            decoded.tracking_ref("origin", "refs/heads/main"),
            Some(&RefTarget::normal(raw(10))),
        );
        assert_eq!(
            decoded.tracking_ref("backup", "refs/heads/main"),
            Some(&RefTarget::normal(raw(20))),
        );
        assert!(decoded.tracking_ref("unknown", "refs/heads/main").is_none());
        assert!(
            decoded
                .tracking_ref("origin", "refs/heads/missing")
                .is_none()
        );
    }

    #[test]
    fn view_without_tracking_refs_stays_backward_compatible() {
        // An empty / unused `remote_refs` field MUST be omitted from
        // the wire encoding so pre-0.3 Views round-trip byte-identically.
        let v_without = View::new()
            .with_head(raw(1))
            .with_ref("refs/heads/main", RefTarget::normal(raw(1)));
        let v_with_empty = View::new()
            .with_head(raw(1))
            .with_ref("refs/heads/main", RefTarget::normal(raw(1)));

        let a = to_canonical_bytes(&v_without).unwrap();
        let b = to_canonical_bytes(&v_with_empty).unwrap();
        assert_eq!(a, b, "empty remote_refs must not change bytes");
    }

    #[test]
    fn view_kind_rejection() {
        let w = ViewWire {
            kind: "commit".into(),
            heads: Vec::new(),
            refs: BTreeMap::new(),
            remote_refs: None,
            wc_commit: None,
            tombstones: Vec::new(),
            extra: BTreeMap::new(),
        };
        let bytes = serde_ipld_dagcbor::to_vec(&w).unwrap();
        let err = serde_ipld_dagcbor::from_slice::<View>(&bytes).unwrap_err();
        assert!(err.to_string().contains("_kind"));
    }
}
