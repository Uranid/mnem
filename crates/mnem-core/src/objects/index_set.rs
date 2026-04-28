//! Index objects (Phase 2 secondary indexes).
//!
//! An [`IndexSet`] lives alongside a [`Commit`](super::Commit) and
//! carries pointers to helper Prolly trees that make common agent
//! queries fast:
//!
//! - **`nodes_by_label`** - for each node label, a Prolly tree keyed by
//!   `NodeId` (16 bytes) whose values are node CIDs. Turns
//!   "all Person nodes" from O(n) into a label-scoped cursor iteration.
//! - **`nodes_by_prop`** - for each (label, `prop_name`), a Prolly tree
//!   keyed by `blake3(canonical_ipld(value))[..16]` whose values are
//!   node CIDs. Turns "node where name='Alice'" into O(log n) point
//!   lookup. Single-valued: duplicate (label, prop, value) tuples
//!   collide at the key level; last-write-wins (use the
//!   `resolve_or_create_node` helper on Transaction to avoid
//!   creating duplicates).
//! - **`outgoing`** - Prolly tree keyed by **source** `NodeId` whose
//!   values are CIDs of [`AdjacencyBucket`] objects - a small sorted
//!   list of `(edge_label, edge_cid)` pairs for that node. Turns
//!   "outgoing edges of X" into O(log n) + one bucket read. Previously
//!   named `adjacency`; the on-wire field alias `adjacency` is still
//!   accepted on decode for repos written before the incoming-index
//!   addition.
//! - **`incoming`** - Prolly tree keyed by **destination** `NodeId`
//!   whose values are CIDs of [`IncomingAdjacencyBucket`] objects - a
//!   small sorted list of `(etype, src, edge_cid)` triples for that
//!   node. Turns "who points at X" into O(log n) + one bucket read.
//!   Added alongside the incoming-index feature; mandates `IndexSet`
//!   rebuild on format-bump commits.
//!
//! All indexes are derived from the node + edge Prolly trees owned by
//! the Commit. They are entirely optional; a commit with
//! `indexes = None` still functions (queries fall back to full scan).

use std::collections::BTreeMap;

use ipld_core::ipld::Ipld;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::id::{Cid, NodeId};

/// Top-level secondary-index aggregator (SPEC §4.8, added in mnem/0.2,
/// extended with `incoming` in mnem/0.3).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct IndexSet {
    /// Map from node label (`ntype`) to the root CID of a Prolly tree
    /// keyed by `NodeId` (16 bytes) with values = node CIDs.
    pub nodes_by_label: BTreeMap<String, Cid>,

    /// Two-level map `label -> prop_name -> Cid`. Each leaf CID is the
    /// root of a Prolly tree keyed by
    /// `blake3(canonical_ipld(value))[..16]` with values = node CIDs.
    pub nodes_by_prop: BTreeMap<String, BTreeMap<String, Cid>>,

    /// Outgoing adjacency index. Prolly tree keyed by **source**
    /// `NodeId` whose values are CIDs of [`AdjacencyBucket`] objects.
    /// `None` on a repo without edges.
    ///
    /// Historically serialised under the field name `adjacency`; on
    /// decode both `outgoing` and the legacy `adjacency` are accepted.
    /// New writes always emit `outgoing`.
    pub outgoing: Option<Cid>,

    /// Incoming adjacency index. Prolly tree keyed by **destination**
    /// `NodeId` whose values are CIDs of [`IncomingAdjacencyBucket`]
    /// objects. `None` on a repo without edges OR on a repo whose
    /// `IndexSet` was built by a pre-0.3 implementation (in which case
    /// callers gracefully degrade to "no back-edges known"; see
    /// SPEC §4.8).
    pub incoming: Option<Cid>,

    /// Forward-compat extension map (SPEC §3.2).
    pub extra: BTreeMap<String, Ipld>,
}

impl IndexSet {
    /// The `_kind` discriminator on the wire.
    pub const KIND: &'static str = "index_set";
}

/// Per-source-node bucket of outgoing edges. Stored as a standalone
/// object so the adjacency Prolly tree can cheaply reference a list
/// without inlining it into every leaf.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct AdjacencyBucket {
    /// Outgoing edges, sorted lexicographically by `(label, edge_cid)`
    /// for byte-stable canonical form.
    pub edges: Vec<AdjacencyEntry>,
    /// Forward-compat extension map.
    pub extra: BTreeMap<String, Ipld>,
}

impl AdjacencyBucket {
    /// The `_kind` discriminator on the wire.
    pub const KIND: &'static str = "adjacency_bucket";
}

/// One outgoing-edge record inside an [`AdjacencyBucket`].
///
/// The bucket is keyed by `src` `NodeId` in the outer Prolly tree, so
/// the source is known from context; only the edge label and edge CID
/// live in each entry. The CID resolves to an `Edge` object, from
/// which `dst` and `edge_id` are recovered.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct AdjacencyEntry {
    /// Edge label (`etype`).
    pub label: String,
    /// Content-addressed edge CID.
    pub edge: Cid,
}

/// Per-destination-node bucket of incoming edges. Stored as a
/// standalone object so the incoming-adjacency Prolly tree can cheaply
/// reference a list without inlining it into every leaf.
///
/// Structurally distinct from [`AdjacencyBucket`] because each entry
/// must carry the **source** `NodeId` (the bucket is keyed by `dst`,
/// so the outer key tells you nothing about `src`). Callers answering
/// "who points at me?" read the bucket and walk `entries` without any
/// `Edge` decode work (the `src` is already here).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct IncomingAdjacencyBucket {
    /// Incoming edges, sorted lexicographically by
    /// `(label, src, edge_cid)` for byte-stable canonical form.
    pub edges: Vec<IncomingAdjacencyEntry>,
    /// Forward-compat extension map.
    pub extra: BTreeMap<String, Ipld>,
}

impl IncomingAdjacencyBucket {
    /// The `_kind` discriminator on the wire.
    pub const KIND: &'static str = "incoming_adjacency_bucket";
}

/// One incoming-edge record inside an [`IncomingAdjacencyBucket`].
///
/// Unlike [`AdjacencyEntry`] (which can omit `src` because it lives in
/// the outer Prolly key), this entry carries the source `NodeId`
/// explicitly: the outer key is `dst`, so without `src` here callers
/// would have to decode the Edge object just to answer "who pointed
/// at me?".
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct IncomingAdjacencyEntry {
    /// Edge label (`etype`).
    pub label: String,
    /// Source `NodeId` - the node that owns the out-edge pointing at
    /// this bucket's dst.
    pub src: NodeId,
    /// Content-addressed edge CID. Resolving it gives the full `Edge`
    /// object (with `edge_id`, props, etc.).
    pub edge: Cid,
}

// ---------------- Serde: IndexSet ----------------

/// On-wire shape for `IndexSet`.
///
/// Both the new `outgoing` field and the legacy `adjacency` alias are
/// accepted on decode. `adjacency` is retained so older repos (written
/// before the incoming-adjacency feature) still round-trip through
/// `Deserialize`; new writes always emit `outgoing`.
#[derive(Serialize, Deserialize)]
struct IndexSetWire {
    #[serde(rename = "_kind")]
    kind: String,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    nodes_by_label: BTreeMap<String, Cid>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    nodes_by_prop: BTreeMap<String, BTreeMap<String, Cid>>,
    // Accept the legacy field name on decode; new writes emit `outgoing`.
    #[serde(default, skip_serializing_if = "Option::is_none", alias = "adjacency")]
    outgoing: Option<Cid>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    incoming: Option<Cid>,
    #[serde(flatten, default, skip_serializing_if = "BTreeMap::is_empty")]
    extra: BTreeMap<String, Ipld>,
}

impl Serialize for IndexSet {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        IndexSetWire {
            kind: Self::KIND.into(),
            nodes_by_label: self.nodes_by_label.clone(),
            nodes_by_prop: self.nodes_by_prop.clone(),
            outgoing: self.outgoing.clone(),
            incoming: self.incoming.clone(),
            extra: self.extra.clone(),
        }
        .serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for IndexSet {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let w = IndexSetWire::deserialize(deserializer)?;
        if w.kind != Self::KIND {
            return Err(serde::de::Error::custom(format!(
                "expected _kind='{}', got '{}'",
                Self::KIND,
                w.kind
            )));
        }
        Ok(Self {
            nodes_by_label: w.nodes_by_label,
            nodes_by_prop: w.nodes_by_prop,
            outgoing: w.outgoing,
            incoming: w.incoming,
            extra: w.extra,
        })
    }
}

// ---------------- Serde: AdjacencyBucket ----------------

#[derive(Serialize, Deserialize)]
struct AdjacencyBucketWire {
    #[serde(rename = "_kind")]
    kind: String,
    edges: Vec<AdjacencyEntry>,
    #[serde(flatten, default, skip_serializing_if = "BTreeMap::is_empty")]
    extra: BTreeMap<String, Ipld>,
}

impl Serialize for AdjacencyBucket {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        AdjacencyBucketWire {
            kind: Self::KIND.into(),
            edges: self.edges.clone(),
            extra: self.extra.clone(),
        }
        .serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for AdjacencyBucket {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let w = AdjacencyBucketWire::deserialize(deserializer)?;
        if w.kind != Self::KIND {
            return Err(serde::de::Error::custom(format!(
                "expected _kind='{}', got '{}'",
                Self::KIND,
                w.kind
            )));
        }
        Ok(Self {
            edges: w.edges,
            extra: w.extra,
        })
    }
}

// ---------------- Serde: IncomingAdjacencyBucket ----------------

#[derive(Serialize, Deserialize)]
struct IncomingAdjacencyBucketWire {
    #[serde(rename = "_kind")]
    kind: String,
    edges: Vec<IncomingAdjacencyEntry>,
    #[serde(flatten, default, skip_serializing_if = "BTreeMap::is_empty")]
    extra: BTreeMap<String, Ipld>,
}

impl Serialize for IncomingAdjacencyBucket {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        IncomingAdjacencyBucketWire {
            kind: Self::KIND.into(),
            edges: self.edges.clone(),
            extra: self.extra.clone(),
        }
        .serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for IncomingAdjacencyBucket {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let w = IncomingAdjacencyBucketWire::deserialize(deserializer)?;
        if w.kind != Self::KIND {
            return Err(serde::de::Error::custom(format!(
                "expected _kind='{}', got '{}'",
                Self::KIND,
                w.kind
            )));
        }
        Ok(Self {
            edges: w.edges,
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
    fn index_set_round_trip() {
        let mut set = IndexSet::default();
        set.nodes_by_label.insert("Person".into(), raw(1));
        set.nodes_by_label.insert("Document".into(), raw(2));
        let mut person_props = BTreeMap::new();
        person_props.insert("name".into(), raw(3));
        set.nodes_by_prop.insert("Person".into(), person_props);
        set.outgoing = Some(raw(4));
        set.incoming = Some(raw(5));

        let bytes = to_canonical_bytes(&set).unwrap();
        let decoded: IndexSet = from_canonical_bytes(&bytes).unwrap();
        assert_eq!(set, decoded);
    }

    #[test]
    fn empty_index_set_round_trips() {
        let set = IndexSet::default();
        let bytes = to_canonical_bytes(&set).unwrap();
        let decoded: IndexSet = from_canonical_bytes(&bytes).unwrap();
        assert_eq!(set, decoded);
    }

    #[test]
    fn index_set_decodes_legacy_adjacency_alias() {
        // Older repos (pre-incoming-adjacency) wrote `adjacency: Cid`.
        // New code reads that into `outgoing`, and `incoming` stays
        // `None` (older repos had no back-index).
        #[derive(Serialize)]
        struct LegacyWire {
            #[serde(rename = "_kind")]
            kind: String,
            adjacency: Cid,
        }
        let legacy = LegacyWire {
            kind: "index_set".into(),
            adjacency: raw(42),
        };
        let bytes = to_canonical_bytes(&legacy).unwrap();
        let decoded: IndexSet = from_canonical_bytes(&bytes).unwrap();
        assert_eq!(decoded.outgoing, Some(raw(42)));
        assert!(decoded.incoming.is_none());
    }

    #[test]
    fn adjacency_bucket_round_trip() {
        let b = AdjacencyBucket {
            edges: vec![
                AdjacencyEntry {
                    label: "knows".into(),
                    edge: raw(10),
                },
                AdjacencyEntry {
                    label: "works_at".into(),
                    edge: raw(11),
                },
            ],
            extra: BTreeMap::new(),
        };
        let bytes = to_canonical_bytes(&b).unwrap();
        let decoded: AdjacencyBucket = from_canonical_bytes(&bytes).unwrap();
        assert_eq!(b, decoded);
    }

    #[test]
    fn incoming_adjacency_bucket_round_trip() {
        let b = IncomingAdjacencyBucket {
            edges: vec![
                IncomingAdjacencyEntry {
                    label: "knows".into(),
                    src: NodeId::from_bytes_raw([1u8; 16]),
                    edge: raw(10),
                },
                IncomingAdjacencyEntry {
                    label: "works_at".into(),
                    src: NodeId::from_bytes_raw([2u8; 16]),
                    edge: raw(11),
                },
            ],
            extra: BTreeMap::new(),
        };
        let bytes = to_canonical_bytes(&b).unwrap();
        let decoded: IncomingAdjacencyBucket = from_canonical_bytes(&bytes).unwrap();
        assert_eq!(b, decoded);
    }

    #[test]
    fn wrong_kind_rejected() {
        let wire = AdjacencyBucketWire {
            kind: "not_adjacency".into(),
            edges: vec![],
            extra: BTreeMap::new(),
        };
        let bytes = serde_ipld_dagcbor::to_vec(&wire).unwrap();
        let err = serde_ipld_dagcbor::from_slice::<AdjacencyBucket>(&bytes).unwrap_err();
        assert!(err.to_string().contains("_kind"));
    }

    #[test]
    fn incoming_bucket_wrong_kind_rejected() {
        let wire = IncomingAdjacencyBucketWire {
            kind: "not_incoming".into(),
            edges: vec![],
            extra: BTreeMap::new(),
        };
        let bytes = serde_ipld_dagcbor::to_vec(&wire).unwrap();
        let err = serde_ipld_dagcbor::from_slice::<IncomingAdjacencyBucket>(&bytes).unwrap_err();
        assert!(err.to_string().contains("_kind"));
    }
}
