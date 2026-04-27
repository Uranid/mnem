//! Operation object (SPEC §4.5) - the unit of the op-log.
//!
//! Every repository-mutating command writes exactly one Operation whose
//! `parents` point at the op-heads observed at command start and whose
//! `view` is the CID of a [`View`] snapshotting heads / refs / working-
//! copy after the mutation.
//!
//! [`View`]: crate::objects::View

use std::collections::BTreeMap;

use ipld_core::ipld::Ipld;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::id::{ChangeId, Cid};
use crate::objects::Signature;

/// A single mutation of repository state.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Operation {
    /// Parent operations (the op-heads observed when this op started).
    /// 0 for the root op; ≥2 after a concurrent-write merge.
    pub parents: Vec<Cid>,
    /// Snapshot of the post-mutation state. A `Link<View>`.
    pub view: Cid,
    /// For each rewritten commit, the `ChangeIds` of its predecessors.
    /// Keys are UUID-string representations of `ChangeIds` (DAG-CBOR map
    /// keys MUST be strings; SPEC §4.5).
    pub predecessors: Option<BTreeMap<String, Vec<ChangeId>>>,
    /// Free-form author identifier.
    pub author: String,
    /// AI-agent identifier (when machine-generated).
    pub agent_id: Option<String>,
    /// Task / tool-call identifier for provenance.
    pub task_id: Option<String>,
    /// Host identifier.
    pub host: Option<String>,
    /// Microseconds since Unix epoch.
    pub time: u64,
    /// Short human description (e.g. `"commit: feat(auth): add OAuth"`).
    pub description: String,
    /// Optional cryptographic signature (SPEC §9.1). Attached via
    /// [`crate::sign::Signer::sign_operation`]; verified via
    /// [`crate::sign::Verifier::verify_operation`].
    pub signature: Option<Signature>,
    /// Forward-compat extension map (SPEC §3.2).
    pub extra: BTreeMap<String, Ipld>,
}

impl Operation {
    /// The `_kind` discriminator on the wire.
    pub const KIND: &'static str = "operation";

    /// Construct an operation with required fields and no parents / optionals.
    #[must_use]
    pub fn new(
        view: Cid,
        author: impl Into<String>,
        time: u64,
        description: impl Into<String>,
    ) -> Self {
        Self {
            parents: Vec::new(),
            view,
            predecessors: None,
            author: author.into(),
            agent_id: None,
            task_id: None,
            host: None,
            time,
            description: description.into(),
            signature: None,
            extra: BTreeMap::new(),
        }
    }

    /// Append a parent operation. Returns `self` for chaining.
    #[must_use]
    pub fn with_parent(mut self, parent: Cid) -> Self {
        self.parents.push(parent);
        self
    }

    /// Attach an agent identifier.
    #[must_use]
    pub fn with_agent(mut self, agent_id: impl Into<String>) -> Self {
        self.agent_id = Some(agent_id.into());
        self
    }

    /// Attach a task identifier.
    #[must_use]
    pub fn with_task(mut self, task_id: impl Into<String>) -> Self {
        self.task_id = Some(task_id.into());
        self
    }

    /// Attach a host identifier.
    #[must_use]
    pub fn with_host(mut self, host: impl Into<String>) -> Self {
        self.host = Some(host.into());
        self
    }
}

// ---------------- Serde ----------------

#[derive(Serialize, Deserialize)]
struct OperationWire {
    #[serde(rename = "_kind")]
    kind: String,
    parents: Vec<Cid>,
    view: Cid,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    predecessors: Option<BTreeMap<String, Vec<ChangeId>>>,
    author: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    agent_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    task_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    host: Option<String>,
    time: u64,
    description: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    signature: Option<Signature>,
    #[serde(flatten, default, skip_serializing_if = "BTreeMap::is_empty")]
    extra: BTreeMap<String, Ipld>,
}

impl Serialize for Operation {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        OperationWire {
            kind: Self::KIND.into(),
            parents: self.parents.clone(),
            view: self.view.clone(),
            predecessors: self.predecessors.clone(),
            author: self.author.clone(),
            agent_id: self.agent_id.clone(),
            task_id: self.task_id.clone(),
            host: self.host.clone(),
            time: self.time,
            description: self.description.clone(),
            signature: self.signature.clone(),
            extra: self.extra.clone(),
        }
        .serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for Operation {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let w = OperationWire::deserialize(deserializer)?;
        if w.kind != Self::KIND {
            return Err(serde::de::Error::custom(format!(
                "expected _kind='{}', got '{}'",
                Self::KIND,
                w.kind
            )));
        }
        Ok(Self {
            parents: w.parents,
            view: w.view,
            predecessors: w.predecessors,
            author: w.author,
            agent_id: w.agent_id,
            task_id: w.task_id,
            host: w.host,
            time: w.time,
            description: w.description,
            signature: w.signature,
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

    fn sample() -> Operation {
        Operation::new(
            raw(1),
            "alice@example.org",
            1_700_000_000_000_000,
            "commit: init",
        )
        .with_agent("agent:claude")
        .with_task("task:001")
        .with_host("workstation-1")
    }

    #[test]
    fn operation_round_trip_byte_identity() {
        let original = sample();
        let bytes = to_canonical_bytes(&original).unwrap();
        let decoded: Operation = from_canonical_bytes(&bytes).unwrap();
        assert_eq!(original, decoded);
        let bytes2 = to_canonical_bytes(&decoded).unwrap();
        assert_eq!(bytes, bytes2);
    }

    #[test]
    fn operation_with_predecessors_round_trip() {
        let mut op = sample();
        let mut preds = BTreeMap::new();
        let key = ChangeId::from_bytes_raw([1u8; 16]).to_uuid_string();
        preds.insert(key, vec![ChangeId::from_bytes_raw([2u8; 16])]);
        op.predecessors = Some(preds);
        let bytes = to_canonical_bytes(&op).unwrap();
        let decoded: Operation = from_canonical_bytes(&bytes).unwrap();
        assert_eq!(op, decoded);
    }

    #[test]
    fn operation_kind_rejection() {
        let w = OperationWire {
            kind: "commit".into(),
            parents: vec![],
            view: raw(1),
            predecessors: None,
            author: "x".into(),
            agent_id: None,
            task_id: None,
            host: None,
            time: 0,
            description: String::new(),
            signature: None,
            extra: BTreeMap::new(),
        };
        let bytes = serde_ipld_dagcbor::to_vec(&w).unwrap();
        let err = serde_ipld_dagcbor::from_slice::<Operation>(&bytes).unwrap_err();
        assert!(err.to_string().contains("_kind"));
    }

    #[test]
    fn operation_with_multiple_parents_round_trip() {
        let op = sample().with_parent(raw(10)).with_parent(raw(11));
        let bytes = to_canonical_bytes(&op).unwrap();
        let decoded: Operation = from_canonical_bytes(&bytes).unwrap();
        assert_eq!(decoded.parents.len(), 2);
        assert_eq!(op, decoded);
    }
}
