//! Commit object (SPEC §4.4).
//!
//! A commit is a versioned snapshot of the graph. It points at three
//! Prolly-tree roots (nodes, edges, schema) and carries provenance
//! metadata - author, agent, task, timestamp - plus an optional
//! Ed25519 signature.

use std::collections::BTreeMap;

use bytes::Bytes;
use ipld_core::ipld::Ipld;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::id::{ChangeId, Cid};

/// Cryptographic signature on a [`Commit`] or [`crate::objects::Operation`].
///
/// Per SPEC §9.1, the signature is computed over the canonical DAG-CBOR
/// encoding of the containing object with the `signature` field absent.
/// M12 will add verification helpers; for now we model the shape only.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Signature {
    /// Algorithm identifier. MUST be `"ed25519"` for mnem/0.1.
    pub algo: String,
    /// Signer's public key. 32 bytes for Ed25519.
    pub public_key: Bytes,
    /// Signature bytes. 64 bytes for Ed25519.
    pub sig: Bytes,
}

/// A versioned snapshot of the graph.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Commit {
    /// Stable change identity (survives rewrite / rebase / amend).
    pub change_id: ChangeId,
    /// Parent commits (empty = root, ≥2 = merge).
    pub parents: Vec<Cid>,
    /// Root of the node Prolly tree.
    pub nodes: Cid,
    /// Root of the edge Prolly tree.
    pub edges: Cid,
    /// Root of the schema Prolly tree.
    pub schema: Cid,
    /// Optional `DeltaSet` link (reserved; not emitted in mnem/0.1).
    pub delta: Option<Cid>,
    /// Optional secondary-index root ([`crate::objects::IndexSet`],
    /// SPEC §4.8). Agents that only need Prolly-lookup by stable id
    /// can ignore this; query paths (label / property / adjacency)
    /// use it when present.
    pub indexes: Option<Cid>,
    /// Optional embedding-sidecar Prolly root. Tree keyed by 32-byte
    /// `NodeCid` digest, value = [`crate::objects::EmbeddingBucket`].
    /// Lifts dense embedding vectors out of `Node` canonical bytes so
    /// the Node CID stays byte-stable across ORT thread counts (f32
    /// reduction ordering is non-deterministic; vectors drift by the
    /// LSB across thread counts). `None` on commits that carry no
    /// embed-bearing nodes.
    ///
    /// **Intentionally excluded from `content_cid`.** Content CID is
    /// the deterministic "what graph is this" digest; including the
    /// embeddings root would re-couple it to ORT thread count and
    /// undo the determinism guarantee. Two machines re-deriving the
    /// same source text on different cores produce the same
    /// `content_cid`, just with per-machine drift in
    /// `commit.embeddings`.
    pub embeddings: Option<Cid>,
    /// Free-form author identifier.
    pub author: String,
    /// AI agent identifier (when the commit was machine-generated).
    pub agent_id: Option<String>,
    /// Task / tool-call identifier for provenance.
    pub task_id: Option<String>,
    /// Microseconds since Unix epoch.
    pub time: u64,
    /// UTF-8 commit message. May be empty.
    pub message: String,
    /// Optional cryptographic signature.
    pub signature: Option<Signature>,
    /// Forward-compat extension map (SPEC §3.2).
    pub extra: BTreeMap<String, Ipld>,
}

impl Commit {
    /// The `_kind` discriminator on the wire.
    pub const KIND: &'static str = "commit";

    /// Build a commit with the required fields, empty optionals / parents / extras.
    #[must_use]
    pub fn new(
        change_id: ChangeId,
        nodes: Cid,
        edges: Cid,
        schema: Cid,
        author: impl Into<String>,
        time: u64,
        message: impl Into<String>,
    ) -> Self {
        Self {
            change_id,
            parents: Vec::new(),
            nodes,
            edges,
            schema,
            delta: None,
            indexes: None,
            embeddings: None,
            author: author.into(),
            agent_id: None,
            task_id: None,
            time,
            message: message.into(),
            signature: None,
            extra: BTreeMap::new(),
        }
    }

    /// Append a parent commit. Returns `self` for chaining.
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

    /// (partial): a deterministic CID over only
    /// the data-DAG portion of the commit -- the three Prolly tree
    /// roots (nodes, edges, schema), the optional indexes root, and
    /// the parents list. Excludes time, change_id, author, message,
    /// agent_id, task_id, signature, and extra.
    ///
    /// Two ingest runs against byte-identical input on different
    /// machines (or at different times) MUST produce the same
    /// `content_cid`. The standard `commit_cid` continues to embed
    /// wall-clock + UUIDv7 metadata for audit-trail purposes; that
    /// CID is intentionally time-varying.
    ///
    /// # Errors
    /// Propagates encoding failures from
    /// [`crate::codec::dagcbor::hash_to_cid`].
    ///
    /// # Migration note
    /// Wire format is unchanged: `content_cid` is computed from
    /// existing fields, so older blockstores stay readable. A
    /// follow-up may persist `content_cid` alongside `commit_cid` in
    /// the operation log for cheap lookup.
    pub fn content_cid(&self) -> Result<Cid, crate::error::CodecError> {
        // Sort parents to make the hash insensitive to merge order
        // (a future merge that swaps the parent list order would
        // otherwise produce a different content_cid for an identical
        // resulting graph). Parent order is not semantically meaningful
        // for content-addressing.
        let mut parents = self.parents.clone();
        parents.sort_by(|a, b| a.to_string().cmp(&b.to_string()));

        let payload = ContentCidPayload {
            schema_version: 1,
            nodes: self.nodes.clone(),
            edges: self.edges.clone(),
            schema: self.schema.clone(),
            indexes: self.indexes.clone(),
            parents,
        };
        let (_bytes, cid) = crate::codec::dagcbor::hash_to_cid(&payload)?;
        Ok(cid)
    }
}

/// Stable wire shape for `Commit::content_cid()`. The struct is
/// intentionally NOT exposed publicly: `content_cid` is purely a
/// derived value, and the on-disk Commit format does not change.
/// Schema version 1 is the post-audit baseline; any future
/// content_cid layout change MUST bump `schema_version` so that two
/// versions of the codebase agree on whether they would compare equal.
#[derive(Serialize)]
struct ContentCidPayload {
    schema_version: u8,
    nodes: Cid,
    edges: Cid,
    schema: Cid,
    #[serde(skip_serializing_if = "Option::is_none")]
    indexes: Option<Cid>,
    parents: Vec<Cid>,
}

// ---------------- Serde ----------------

#[derive(Serialize, Deserialize)]
struct CommitWire {
    #[serde(rename = "_kind")]
    kind: String,
    change_id: ChangeId,
    parents: Vec<Cid>,
    nodes: Cid,
    edges: Cid,
    schema: Cid,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    delta: Option<Cid>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    indexes: Option<Cid>,
    /// `skip_serializing_if` keeps absence-on-encode so commits without
    /// an embedding sidecar round-trip byte-identically.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    embeddings: Option<Cid>,
    author: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    agent_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    task_id: Option<String>,
    time: u64,
    message: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    signature: Option<Signature>,
    #[serde(flatten, default, skip_serializing_if = "BTreeMap::is_empty")]
    extra: BTreeMap<String, Ipld>,
}

impl Serialize for Commit {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        CommitWire {
            kind: Self::KIND.into(),
            change_id: self.change_id,
            parents: self.parents.clone(),
            nodes: self.nodes.clone(),
            edges: self.edges.clone(),
            schema: self.schema.clone(),
            delta: self.delta.clone(),
            indexes: self.indexes.clone(),
            embeddings: self.embeddings.clone(),
            author: self.author.clone(),
            agent_id: self.agent_id.clone(),
            task_id: self.task_id.clone(),
            time: self.time,
            message: self.message.clone(),
            signature: self.signature.clone(),
            extra: self.extra.clone(),
        }
        .serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for Commit {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let w = CommitWire::deserialize(deserializer)?;
        if w.kind != Self::KIND {
            return Err(serde::de::Error::custom(format!(
                "expected _kind='{}', got '{}'",
                Self::KIND,
                w.kind
            )));
        }
        Ok(Self {
            change_id: w.change_id,
            parents: w.parents,
            nodes: w.nodes,
            edges: w.edges,
            schema: w.schema,
            delta: w.delta,
            indexes: w.indexes,
            embeddings: w.embeddings,
            author: w.author,
            agent_id: w.agent_id,
            task_id: w.task_id,
            time: w.time,
            message: w.message,
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

    fn sample() -> Commit {
        Commit::new(
            ChangeId::from_bytes_raw([1u8; 16]),
            raw(1),
            raw(2),
            raw(3),
            "alice@example.org",
            1_700_000_000_000_000,
            "init",
        )
        .with_agent("agent:claude")
        .with_task("task:001")
    }

    #[test]
    fn commit_round_trip_byte_identity() {
        let original = sample();
        let bytes = to_canonical_bytes(&original).unwrap();
        let decoded: Commit = from_canonical_bytes(&bytes).unwrap();
        assert_eq!(original, decoded);
        let bytes2 = to_canonical_bytes(&decoded).unwrap();
        assert_eq!(bytes, bytes2);
    }

    /// two commits with byte-identical data
    /// roots but different timestamps, change_ids, authors, and
    /// messages MUST share `content_cid` while their `commit_cid`
    /// differs.
    #[test]
    fn content_cid_is_stable_across_metadata() {
        let mut a = Commit::new(
            ChangeId::from_bytes_raw([1u8; 16]),
            raw(10),
            raw(20),
            raw(30),
            "alice@example.org",
            1_700_000_000_000_000,
            "init",
        );
        a.indexes = Some(raw(40));

        let mut b = Commit::new(
            // Different change_id (UUIDv7 typically embeds a timestamp).
            ChangeId::from_bytes_raw([2u8; 16]),
            // SAME data roots:
            raw(10),
            raw(20),
            raw(30),
            // Different author + time + message:
            "bob@example.org",
            1_777_000_000_000_000,
            "different message entirely",
        );
        b.indexes = Some(raw(40));

        assert_eq!(
            a.content_cid().unwrap(),
            b.content_cid().unwrap(),
            "content_cid must ignore metadata (time, change_id, author, message)"
        );

        let (a_bytes, a_commit_cid) = crate::codec::dagcbor::hash_to_cid(&a).unwrap();
        let (b_bytes, b_commit_cid) = crate::codec::dagcbor::hash_to_cid(&b).unwrap();
        let _ = (a_bytes, b_bytes);
        assert_ne!(
            a_commit_cid, b_commit_cid,
            "commit_cid SHOULD differ when metadata differs (audit-trail invariant)"
        );
    }

    /// content_cid MUST change when any data root changes.
    #[test]
    fn content_cid_distinguishes_data_roots() {
        let a = Commit::new(
            ChangeId::from_bytes_raw([1u8; 16]),
            raw(10),
            raw(20),
            raw(30),
            "alice",
            1,
            "msg",
        );
        let b = Commit::new(
            ChangeId::from_bytes_raw([1u8; 16]),
            raw(11), // different nodes root
            raw(20),
            raw(30),
            "alice",
            1,
            "msg",
        );
        assert_ne!(a.content_cid().unwrap(), b.content_cid().unwrap());
    }

    /// Load-bearing invariant: two commits with byte-identical data
    /// roots but DIFFERENT `embeddings` sidecar Cids MUST share
    /// `content_cid`. If this fails, a future change re-coupled
    /// `ContentCidPayload` to the embedding sidecar - exactly the
    /// architectural error this design exists to prevent. Federated
    /// dedup (two machines indexing the same source produce the same
    /// content_cid) would silently break.
    #[test]
    fn content_cid_ignores_embeddings_field() {
        let mut a = sample();
        a.embeddings = Some(raw(100));
        let mut b = sample();
        b.embeddings = Some(raw(200)); // different embedding sidecar
        assert_eq!(
            a.content_cid().unwrap(),
            b.content_cid().unwrap(),
            "content_cid MUST ignore the embeddings sidecar - that is the G16 contract"
        );

        // Also: a commit with `embeddings = None` and a commit with
        // `embeddings = Some(_)` must share the same content_cid when
        // every other data root matches.
        let mut c = sample();
        c.embeddings = None;
        let mut d = sample();
        d.embeddings = Some(raw(300));
        assert_eq!(
            c.content_cid().unwrap(),
            d.content_cid().unwrap(),
            "absence of embeddings must not change content_cid either"
        );
    }

    /// `Commit.embeddings: Some(cid)` survives encode → decode →
    /// re-encode byte-identically. Pins the wire-form contract for
    /// the new G16 field.
    #[test]
    fn commit_with_embeddings_some_round_trips() {
        let mut original = sample();
        original.embeddings = Some(raw(42));
        let bytes = to_canonical_bytes(&original).unwrap();
        let decoded: Commit = from_canonical_bytes(&bytes).unwrap();
        assert_eq!(original, decoded);
        assert_eq!(decoded.embeddings, Some(raw(42)));
        let bytes2 = to_canonical_bytes(&decoded).unwrap();
        assert_eq!(
            bytes, bytes2,
            "round-trip must be byte-identical - wire form is contract-bound"
        );
    }

    /// Backwards-compat: a CBOR commit written without the
    /// `embeddings` key must decode cleanly with `embeddings = None`
    /// and re-encode byte-identically. The wire emitter omits the
    /// key when `None`, so legacy bytes round-trip.
    #[test]
    fn commit_legacy_no_embeddings_key_round_trips() {
        // Construct a commit with `embeddings = None` (wire form
        // omits the key entirely under `skip_serializing_if`).
        let original = sample();
        assert_eq!(original.embeddings, None);
        let bytes = to_canonical_bytes(&original).unwrap();

        // Verify the wire form does NOT contain the `embeddings` key.
        // The literal byte string "embeddings" cannot appear by chance
        // in a Cid digest, so this is a robust negative probe.
        assert!(
            !bytes
                .windows(b"embeddings".len())
                .any(|w| w == b"embeddings"),
            "wire form must omit the `embeddings` key when None"
        );

        // Decode back; field defaults to `None`.
        let decoded: Commit = from_canonical_bytes(&bytes).unwrap();
        assert_eq!(decoded.embeddings, None);
        assert_eq!(decoded, original);

        // Re-encode; bytes must match exactly.
        let bytes2 = to_canonical_bytes(&decoded).unwrap();
        assert_eq!(bytes, bytes2, "legacy CBOR must re-encode byte-identically");
    }

    #[test]
    fn commit_kind_rejection() {
        let wire = CommitWire {
            kind: "node".into(),
            change_id: ChangeId::from_bytes_raw([1u8; 16]),
            parents: vec![],
            nodes: raw(1),
            edges: raw(2),
            schema: raw(3),
            delta: None,
            indexes: None,
            embeddings: None,
            author: "x".into(),
            agent_id: None,
            task_id: None,
            time: 0,
            message: String::new(),
            signature: None,
            extra: BTreeMap::new(),
        };
        let bytes = serde_ipld_dagcbor::to_vec(&wire).unwrap();
        let err = serde_ipld_dagcbor::from_slice::<Commit>(&bytes).unwrap_err();
        assert!(err.to_string().contains("_kind"));
    }

    #[test]
    fn commit_with_parents_round_trip() {
        let c = sample().with_parent(raw(100)).with_parent(raw(101));
        let bytes = to_canonical_bytes(&c).unwrap();
        let decoded: Commit = from_canonical_bytes(&bytes).unwrap();
        assert_eq!(c, decoded);
        assert_eq!(decoded.parents.len(), 2);
    }

    #[test]
    fn commit_with_signature_round_trip() {
        let mut c = sample();
        c.signature = Some(Signature {
            algo: "ed25519".into(),
            public_key: Bytes::from(vec![0xAAu8; 32]),
            sig: Bytes::from(vec![0xBBu8; 64]),
        });
        let bytes = to_canonical_bytes(&c).unwrap();
        let decoded: Commit = from_canonical_bytes(&bytes).unwrap();
        assert_eq!(c, decoded);
        assert_eq!(decoded.signature.as_ref().unwrap().algo, "ed25519");
    }

    #[test]
    fn commit_extra_fields_preserved() {
        let mut c = sample();
        c.extra
            .insert("x-future-field".into(), Ipld::String("v9".into()));
        let bytes = to_canonical_bytes(&c).unwrap();
        let decoded: Commit = from_canonical_bytes(&bytes).unwrap();
        assert_eq!(c, decoded);
        let bytes2 = to_canonical_bytes(&decoded).unwrap();
        assert_eq!(bytes, bytes2);
    }
}
