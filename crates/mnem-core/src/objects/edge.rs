//! The [`Edge`] object.
//!
//! Per SPEC §4.2:
//!
//! ```text
//! Edge: {
//!   _kind: "edge",
//!   id:    EdgeId (16 bytes),
//!   etype: string,
//!   src:   NodeId (16 bytes),
//!   dst:   NodeId (16 bytes),
//!   props: map<string, Ipld>,
//! }
//! ```
//!
//! Edges reference their endpoints by **stable `NodeId`**, never by content
//! hash (SPEC §4.2, ): a node property edit does not invalidate
//! edges referencing it.

use std::collections::BTreeMap;

use ipld_core::ipld::Ipld;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::id::{EdgeId, NodeId};

/// A typed, directed link between two Nodes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Edge {
    /// Stable edge identity. Survives property edits.
    pub id: EdgeId,
    /// Free-form edge-type label (`"knows"`, `"cites"`, …).
    pub etype: String,
    /// Source `NodeId`.
    pub src: NodeId,
    /// Destination `NodeId`.
    pub dst: NodeId,
    /// Edge properties. Values are any DAG-CBOR value.
    pub props: BTreeMap<String, Ipld>,
    /// Forward-compat extension map per SPEC §3.2.
    pub extra: BTreeMap<String, Ipld>,
}

impl Edge {
    /// The `_kind` discriminator for edges. `"edge"` on the wire.
    pub const KIND: &'static str = "edge";

    /// Construct an edge with no properties.
    #[must_use]
    pub fn new(id: EdgeId, etype: impl Into<String>, src: NodeId, dst: NodeId) -> Self {
        Self {
            id,
            etype: etype.into(),
            src,
            dst,
            props: BTreeMap::new(),
            extra: BTreeMap::new(),
        }
    }

    /// Attach a property. Returns `self` for chaining.
    #[must_use]
    pub fn with_prop(mut self, key: impl Into<String>, value: impl Into<Ipld>) -> Self {
        self.props.insert(key.into(), value.into());
        self
    }
}

// ---------------- Edge serde ----------------

#[derive(Serialize, Deserialize)]
struct EdgeWire {
    #[serde(rename = "_kind")]
    kind: String,
    id: EdgeId,
    etype: String,
    src: NodeId,
    dst: NodeId,
    props: BTreeMap<String, Ipld>,
    #[serde(flatten, default, skip_serializing_if = "BTreeMap::is_empty")]
    extra: BTreeMap<String, Ipld>,
}

impl Serialize for Edge {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        EdgeWire {
            kind: Self::KIND.into(),
            id: self.id,
            etype: self.etype.clone(),
            src: self.src,
            dst: self.dst,
            props: self.props.clone(),
            extra: self.extra.clone(),
        }
        .serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for Edge {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let wire = EdgeWire::deserialize(deserializer)?;
        if wire.kind != Self::KIND {
            return Err(serde::de::Error::custom(format!(
                "expected _kind='{}', got '{}'",
                Self::KIND,
                wire.kind
            )));
        }
        Ok(Self {
            id: wire.id,
            etype: wire.etype,
            src: wire.src,
            dst: wire.dst,
            props: wire.props,
            extra: wire.extra,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codec::{from_canonical_bytes, to_canonical_bytes};

    fn sample() -> Edge {
        Edge::new(
            EdgeId::from_bytes_raw([3u8; 16]),
            "knows",
            NodeId::from_bytes_raw([1u8; 16]),
            NodeId::from_bytes_raw([2u8; 16]),
        )
        .with_prop("since", Ipld::Integer(2020))
    }

    #[test]
    fn edge_round_trip_byte_identity() {
        let original = sample();
        let bytes = to_canonical_bytes(&original).expect("encode");
        let decoded: Edge = from_canonical_bytes(&bytes).expect("decode");
        assert_eq!(original, decoded);
        let bytes2 = to_canonical_bytes(&decoded).expect("re-encode");
        assert_eq!(bytes, bytes2);
    }

    #[test]
    fn edge_kind_rejection() {
        let wire = EdgeWire {
            kind: "node".into(),
            id: EdgeId::from_bytes_raw([4u8; 16]),
            etype: "x".into(),
            src: NodeId::from_bytes_raw([1u8; 16]),
            dst: NodeId::from_bytes_raw([2u8; 16]),
            props: BTreeMap::new(),
            extra: BTreeMap::new(),
        };
        let bytes = serde_ipld_dagcbor::to_vec(&wire).expect("encode");
        let err = serde_ipld_dagcbor::from_slice::<Edge>(&bytes).unwrap_err();
        assert!(err.to_string().contains("_kind"));
    }
}
