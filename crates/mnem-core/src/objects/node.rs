//! The [`Node`] object and its embedding substructure.
//!
//! Per SPEC §4.1:
//!
//! ```text
//! Node: {
//!   _kind:   "node",
//!   id:      NodeId (16 bytes),
//!   ntype:   string,
//!   summary: string (optional),
//!   props:   map<string, Ipld>,
//!   content: bytes (optional),
//! }
//! ```
//!
//! Dense vector embeddings live in the per-commit sidecar
//! (`Commit.embeddings` Prolly tree, keyed by NodeCid). Keeping them
//! out of the canonical Node bytes prevents nondeterministic dense
//! producers (e.g. ORT thread-count drift) from leaking into
//! `NodeCid` and breaking federated dedup.
//!
//! Legacy DAG-CBOR carrying an explicit `embed` map round-trips
//! losslessly: the field-less wire decode plus the `extra` flatten
//! sink absorbs and re-emits the bytes byte-identically, so existing
//! NodeCids stay stable for repos written before this change.

use std::collections::BTreeMap;

use bytes::Bytes;
use ipld_core::ipld::Ipld;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::error::ObjectError;
use crate::id::NodeId;
use crate::sparse::SparseEmbed;

// ---------------- Dtype + Embedding ----------------

/// Numeric element type for an [`Embedding`] vector (SPEC §4.1).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
#[derive(Default)]
pub enum Dtype {
    /// IEEE 754 half precision - 2 bytes per element.
    F16,
    /// IEEE 754 single precision - 4 bytes per element. Default.
    #[default]
    F32,
    /// IEEE 754 double precision - 8 bytes per element.
    F64,
    /// Signed 8-bit integer (quantized embeddings) - 1 byte per element.
    I8,
}

impl Dtype {
    /// Bytes per vector element.
    #[must_use]
    pub const fn byte_width(self) -> usize {
        match self {
            Self::F16 => 2,
            Self::F32 => 4,
            Self::F64 => 8,
            Self::I8 => 1,
        }
    }
}

/// A dense vector embedding produced by a named model.
///
/// Embeddings live in the per-commit sidecar (`Commit.embeddings`
/// Prolly tree, keyed by NodeCid) rather than inline on `Node`. This
/// keeps dense bytes out of the canonical Node hash so nondeterministic
/// producers (e.g. ORT thread-count drift) cannot perturb `NodeCid`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Embedding {
    /// Free-form model identifier (`"text-embedding-3-small"`,
    /// `"nomic-embed-text-v1.5"`, etc.).
    pub model: String,
    /// Element type of the vector. Defaults to `f32` on encode when absent.
    #[serde(default)]
    pub dtype: Dtype,
    /// Vector dimension.
    pub dim: u32,
    /// Vector bytes. Length MUST equal `dim * dtype.byte_width()` per
    /// SPEC §4.1; validate with [`Embedding::validate`] after decoding.
    pub vector: Bytes,
}

impl Embedding {
    /// Validate the `vector.len() == dim × byte_width` invariant.
    ///
    /// # Errors
    ///
    /// Returns [`ObjectError::EmbeddingSizeMismatch`] if the invariant is
    /// violated.
    pub const fn validate(&self) -> Result<(), ObjectError> {
        let expected = (self.dim as usize) * self.dtype.byte_width();
        if self.vector.len() == expected {
            Ok(())
        } else {
            Err(ObjectError::EmbeddingSizeMismatch {
                expected,
                got: self.vector.len(),
            })
        }
    }
}

// ---------------- Node ----------------

/// A graph vertex.
///
/// See SPEC §4.1 and [the module docs](super). Construct via [`Node::new`]
/// and add properties with the `with_*` fluent helpers.
///
/// Node is `PartialEq` but not `Eq`: `sparse_embed` carries `Vec<f32>`
/// whose values cannot be `Eq` (NaN). Use CID equality via
/// `hash_to_cid` when you need a canonical identity check; field-wise
/// `==` comparison still works for non-NaN data.
#[derive(Clone, Debug, PartialEq)]
pub struct Node {
    /// Stable node identity. Survives content edits; edges reference this.
    pub id: NodeId,
    /// Free-form node-type label (`"Person"`, `"mnem:Class"`, …).
    pub ntype: String,
    /// Optional short natural-language summary. Intended as the
    /// token-cheap representation of this node for LLM-facing retrieval:
    /// the field agents read when assembling context under a token
    /// budget. Distinct from `props` (structured) and `content`
    /// (opaque payload).
    pub summary: Option<String>,
    /// Property map. Values are any DAG-CBOR value, including `Link`s.
    pub props: BTreeMap<String, Ipld>,
    /// Optional opaque payload (a document body, a file, …).
    pub content: Option<Bytes>,
    /// Optional learned-sparse embedding . Produced
    /// by a `SparseEncoder` adapter (`OpenSearch` neural-sparse-doc-v3-
    /// distill, BGE-M3-sparse, etc.) and indexed by
    /// `crate::index::sparse::SparseInvertedIndex::build_from_repo`.
    ///
    /// Additive: existing nodes with `sparse_embed = None` keep
    /// byte-identical CIDs because the wire serializer omits the field
    /// via `skip_serializing_if = "Option::is_none"`.
    pub sparse_embed: Option<SparseEmbed>,
    /// Optional contextualized-chunk prefix . An
    /// LLM-generated one-sentence placement cue ("This paragraph is
    /// from Section 3 of a legal contract between Alice and Bob's
    /// employer...") stored alongside the node. The ingest pipeline
    /// prepends it to `summary` before embedding so the dense + sparse
    /// lanes capture positional and relational context the chunk
    /// alone would lose.
    ///
    /// Anthropic's 2024 Contextual Retrieval paper reports -49% to
    /// -67% retrieval-failure reduction when this prefix is present;
    /// mnem stores it on the node so the render path can surface it
    /// back to the agent for faithful source attribution.
    ///
    /// Additive: existing nodes with `context_sentence = None` keep
    /// byte-identical CIDs (same `skip_serializing_if = "Option::is_none"`
    /// pattern as `sparse_embed`).
    pub context_sentence: Option<String>,
    /// Forward-compat extension map per SPEC §3.2 - holds fields this
    /// version doesn't recognize and preserves them on re-encode so signed
    /// Nodes remain verifiable across version upgrades.
    pub extra: BTreeMap<String, Ipld>,
}

impl Node {
    /// The `_kind` discriminator for nodes. `"node"` on the wire.
    pub const KIND: &'static str = "node";

    /// Default `ntype` value used when a caller wants to ingest a node
    /// without choosing a category. Applied by the HTTP bulk/single
    /// handlers when the caller omits `label` or sends an empty string.
    /// Direct Rust callers of [`Node::new`] still pass ntype explicitly;
    /// [`Node::new_default`] is the zero-arg convenience.
    pub const DEFAULT_NTYPE: &'static str = "Node";

    /// Construct a Node with no summary, no props, no content.
    #[must_use]
    pub fn new(id: NodeId, ntype: impl Into<String>) -> Self {
        Self {
            id,
            ntype: ntype.into(),
            summary: None,
            props: BTreeMap::new(),
            content: None,
            sparse_embed: None,
            context_sentence: None,
            extra: BTreeMap::new(),
        }
    }

    /// Construct a Node with the project default `ntype = "Node"`.
    /// Convenience for callers that don't want to categorise on write;
    /// equivalent to `Node::new(id, Node::DEFAULT_NTYPE)`.
    #[must_use]
    pub fn new_default(id: NodeId) -> Self {
        Self::new(id, Self::DEFAULT_NTYPE)
    }

    /// Attach a short summary. Returns `self` for chaining.
    #[must_use]
    pub fn with_summary(mut self, summary: impl Into<String>) -> Self {
        self.summary = Some(summary.into());
        self
    }

    /// Attach a property. Returns `self` for chaining.
    #[must_use]
    pub fn with_prop(mut self, key: impl Into<String>, value: impl Into<Ipld>) -> Self {
        self.props.insert(key.into(), value.into());
        self
    }

    /// Attach opaque content.
    #[must_use]
    pub fn with_content(mut self, content: Bytes) -> Self {
        self.content = Some(content);
        self
    }

    /// Attach a learned-sparse embedding. Consumed by the sparse lane in
    /// `Retriever` via `crate::index::sparse::SparseInvertedIndex`.
    #[must_use]
    pub fn with_sparse_embed(mut self, sparse_embed: SparseEmbed) -> Self {
        self.sparse_embed = Some(sparse_embed);
        self
    }

    /// Attach an LLM-generated contextualized-chunk prefix .
    /// The render path prepends this to the summary so the agent sees
    /// where this chunk sits in its source document.
    ///
    /// Typical callers run this at ingest time via a `TextGenerator`
    /// from `mnem-llm-providers` with a prompt like:
    ///
    /// > "Give a single sentence that situates the following chunk
    /// > within its source so a retrieval model can understand where
    /// > it came from. Chunk: `{summary}` Document context: `{doc_title}`"
    #[must_use]
    pub fn with_context_sentence(mut self, context: impl Into<String>) -> Self {
        self.context_sentence = Some(context.into());
        self
    }

    // ---------------- Typed-property accessors ----------------
    //
    // Agent code usually wants `"name" -> "Alice"` as `&str`, not the
    // raw `Option<&Ipld>` every call-site has to pattern-match. The
    // helpers below provide the common scalar extractions without
    // adding a new dependency or a bespoke value type.

    /// Get a property as `&str`. Returns `None` if absent or not a string.
    #[must_use]
    pub fn get_str(&self, key: &str) -> Option<&str> {
        match self.props.get(key)? {
            Ipld::String(s) => Some(s.as_str()),
            _ => None,
        }
    }

    /// Get a property as `i128`. Returns `None` if absent or not an integer.
    #[must_use]
    pub fn get_int(&self, key: &str) -> Option<i128> {
        match self.props.get(key)? {
            Ipld::Integer(n) => Some(*n),
            _ => None,
        }
    }

    /// Get a property as `bool`. Returns `None` if absent or not a bool.
    #[must_use]
    pub fn get_bool(&self, key: &str) -> Option<bool> {
        match self.props.get(key)? {
            Ipld::Bool(b) => Some(*b),
            _ => None,
        }
    }

    /// Get a property as `f64`. Returns `None` if absent or not a float.
    #[must_use]
    pub fn get_float(&self, key: &str) -> Option<f64> {
        match self.props.get(key)? {
            Ipld::Float(f) => Some(*f),
            _ => None,
        }
    }

    /// Get a property as a byte slice. Returns `None` if absent or not bytes.
    #[must_use]
    pub fn get_bytes(&self, key: &str) -> Option<&[u8]> {
        match self.props.get(key)? {
            Ipld::Bytes(b) => Some(b.as_slice()),
            _ => None,
        }
    }
}

// ---------------- Node serde (hand-rolled to enforce _kind) ----------------

// The on-wire shape is a DAG-CBOR map with a `_kind` string field. We
// serialize via `NodeWire` (an internal helper that is structurally
// identical to Node plus the `_kind` field) and validate `_kind` on
// deserialize. This keeps the public `Node` struct ergonomic while
// enforcing the discriminator at the codec boundary.

#[derive(Serialize, Deserialize)]
struct NodeWire {
    #[serde(rename = "_kind")]
    kind: String,
    id: NodeId,
    ntype: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    summary: Option<String>,
    props: BTreeMap<String, Ipld>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    content: Option<Bytes>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    sparse_embed: Option<SparseEmbed>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    context_sentence: Option<String>,
    // Forward-compat sink. Absorbs unknown keys including legacy `embed`
    // maps written before dense vectors moved to the sidecar; the
    // flatten round-trip keeps NodeCid byte-stable for those repos.
    #[serde(flatten, default, skip_serializing_if = "BTreeMap::is_empty")]
    extra: BTreeMap<String, Ipld>,
}

impl Serialize for Node {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        NodeWire {
            kind: Self::KIND.into(),
            id: self.id,
            ntype: self.ntype.clone(),
            summary: self.summary.clone(),
            props: self.props.clone(),
            content: self.content.clone(),
            sparse_embed: self.sparse_embed.clone(),
            context_sentence: self.context_sentence.clone(),
            extra: self.extra.clone(),
        }
        .serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for Node {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let wire = NodeWire::deserialize(deserializer)?;
        if wire.kind != Self::KIND {
            return Err(serde::de::Error::custom(format!(
                "expected _kind='{}', got '{}'",
                Self::KIND,
                wire.kind
            )));
        }
        Ok(Self {
            id: wire.id,
            ntype: wire.ntype,
            summary: wire.summary,
            props: wire.props,
            content: wire.content,
            sparse_embed: wire.sparse_embed,
            context_sentence: wire.context_sentence,
            extra: wire.extra,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codec::{from_canonical_bytes, hash_to_cid, to_canonical_bytes};

    fn alice() -> Node {
        Node::new(NodeId::from_bytes_raw([1u8; 16]), "Person")
            .with_prop("name", Ipld::String("Alice".into()))
            .with_prop("age", Ipld::Integer(30))
    }

    #[test]
    fn node_round_trip_byte_identity() {
        let original = alice();
        let bytes = to_canonical_bytes(&original).expect("encode");
        let decoded: Node = from_canonical_bytes(&bytes).expect("decode");
        assert_eq!(original, decoded);
        let bytes2 = to_canonical_bytes(&decoded).expect("re-encode");
        assert_eq!(bytes, bytes2);
    }

    #[test]
    fn node_cid_is_deterministic() {
        let a1 = alice();
        let a2 = alice();
        let (_, c1) = hash_to_cid(&a1).expect("hash");
        let (_, c2) = hash_to_cid(&a2).expect("hash");
        assert_eq!(c1, c2);
    }

    #[test]
    fn new_default_uses_default_ntype() {
        let n = Node::new_default(NodeId::from_bytes_raw([7u8; 16]));
        assert_eq!(n.ntype, Node::DEFAULT_NTYPE);
        assert_eq!(n.ntype, "Node");
    }

    #[test]
    fn new_default_and_explicit_new_match_when_ntype_equal() {
        // `Node::new_default(id)` must be byte-identical to
        // `Node::new(id, Node::DEFAULT_NTYPE)`. CID stability test.
        let id = NodeId::from_bytes_raw([9u8; 16]);
        let default_node = Node::new_default(id);
        let explicit_node = Node::new(id, Node::DEFAULT_NTYPE);
        let (_, c_default) = hash_to_cid(&default_node).expect("hash default");
        let (_, c_explicit) = hash_to_cid(&explicit_node).expect("hash explicit");
        assert_eq!(c_default, c_explicit);
    }

    #[test]
    fn node_kind_rejection() {
        // Encode something whose _kind = "edge"; decoding as Node must fail.
        let wire = NodeWire {
            kind: "edge".into(),
            id: NodeId::from_bytes_raw([1u8; 16]),
            ntype: "x".into(),
            summary: None,
            props: BTreeMap::new(),
            content: None,
            sparse_embed: None,
            context_sentence: None,
            extra: BTreeMap::new(),
        };
        let bytes = serde_ipld_dagcbor::to_vec(&wire).expect("encode wire");
        let err = serde_ipld_dagcbor::from_slice::<Node>(&bytes).unwrap_err();
        assert!(
            err.to_string().contains("_kind"),
            "expected _kind rejection, got: {err}"
        );
    }

    #[test]
    fn node_extra_fields_round_trip() {
        // Start with a NodeWire that includes an unknown field.
        let mut wire = NodeWire {
            kind: "node".into(),
            id: NodeId::from_bytes_raw([2u8; 16]),
            ntype: "Future".into(),
            summary: None,
            props: BTreeMap::new(),
            content: None,
            sparse_embed: None,
            context_sentence: None,
            extra: BTreeMap::new(),
        };
        wire.extra.insert(
            "x-future-field".into(),
            Ipld::String("value-from-v99".into()),
        );
        let bytes_in = serde_ipld_dagcbor::to_vec(&wire).expect("encode");

        // Decode as Node - the unknown field lands in `extra`.
        let decoded: Node = serde_ipld_dagcbor::from_slice(&bytes_in).expect("decode");
        assert_eq!(
            decoded.extra.get("x-future-field"),
            Some(&Ipld::String("value-from-v99".into())),
        );

        // Re-encode as Node - bytes must match the input.
        let bytes_out = to_canonical_bytes(&decoded).expect("re-encode");
        assert_eq!(bytes_in, bytes_out);
    }

    #[test]
    fn legacy_embed_field_round_trips_through_extra() {
        // Legacy DAG-CBOR encoded under the prior schema where the Node
        // map carried an explicit `embed` sub-map. After the field
        // removal the wire decoder no longer recognises `embed`, so the
        // serde(flatten) `extra` sink absorbs the key. Re-encoding emits
        // it unchanged - bytes are byte-identical and the NodeCid stays
        // stable across the reader transition.
        //
        // We synthesise the legacy bytes by encoding a separate wire
        // struct that still has an `embed` field so this test does not
        // depend on a baked binary fixture.
        #[derive(Serialize)]
        struct LegacyNodeWire {
            #[serde(rename = "_kind")]
            kind: String,
            id: NodeId,
            ntype: String,
            #[serde(skip_serializing_if = "Option::is_none")]
            summary: Option<String>,
            props: BTreeMap<String, Ipld>,
            #[serde(skip_serializing_if = "Option::is_none")]
            content: Option<Bytes>,
            embed: Embedding,
        }

        let legacy = LegacyNodeWire {
            kind: "node".into(),
            id: NodeId::from_bytes_raw([42u8; 16]),
            ntype: "Doc".into(),
            summary: None,
            props: BTreeMap::new(),
            content: None,
            embed: Embedding {
                model: "openai:text-embedding-3-small".into(),
                dtype: Dtype::F32,
                dim: 2,
                vector: Bytes::from(vec![
                    0x00, 0x00, 0x80, 0x3f, // 1.0_f32 LE
                    0x00, 0x00, 0x00, 0x00, // 0.0_f32 LE
                ]),
            },
        };
        let bytes_in = serde_ipld_dagcbor::to_vec(&legacy).expect("encode legacy");

        // Decode as Node: `embed` is unknown to the new wire, so the
        // flatten sink absorbs it.
        let decoded: Node = serde_ipld_dagcbor::from_slice(&bytes_in).expect("decode legacy");
        assert!(
            decoded.extra.contains_key("embed"),
            "legacy embed must land in extra"
        );

        // Re-encoding produces the same byte sequence.
        let bytes_out = to_canonical_bytes(&decoded).expect("re-encode");
        assert_eq!(bytes_in, bytes_out, "legacy bytes must round-trip exactly");

        // NodeCid is stable across the reader transition: hashing the
        // re-encoded bytes via the a future version reader path must produce the
        // same Cid as hashing the legacy v0.1.0 bytes directly. Equality
        // here is the load-bearing federated-dedup invariant.
        let (bytes_from_node, cid_from_node) = hash_to_cid(&decoded).expect("hash node");
        assert_eq!(
            bytes_in.as_slice(),
            bytes_from_node.as_ref(),
            "a future version re-encode must match legacy bytes byte-for-byte"
        );
        let cid_via_legacy_bytes = {
            let mh = crate::id::Multihash::sha2_256(&bytes_in);
            crate::id::Cid::new(crate::id::CODEC_DAG_CBOR, mh)
        };
        assert_eq!(
            cid_from_node, cid_via_legacy_bytes,
            "NodeCid via a future version reader must equal NodeCid via legacy bytes"
        );
    }

    #[test]
    fn node_round_trip_with_summary() {
        let n = Node::new(NodeId::from_bytes_raw([3u8; 16]), "Person")
            .with_summary("Alice, 30, based in Berlin.")
            .with_prop("name", Ipld::String("Alice".into()));
        let bytes = to_canonical_bytes(&n).expect("encode");
        let decoded: Node = from_canonical_bytes(&bytes).expect("decode");
        assert_eq!(
            decoded.summary.as_deref(),
            Some("Alice, 30, based in Berlin.")
        );
        assert_eq!(n, decoded);

        // Summary participates in the content hash: same node without
        // the summary must hash to a different CID.
        let bare = Node::new(NodeId::from_bytes_raw([3u8; 16]), "Person")
            .with_prop("name", Ipld::String("Alice".into()));
        let (_, c_with) = hash_to_cid(&n).expect("hash");
        let (_, c_without) = hash_to_cid(&bare).expect("hash");
        assert_ne!(c_with, c_without);
    }

    #[test]
    fn node_sparse_embed_round_trips() {
        let s = crate::sparse::SparseEmbed::new(vec![1, 5, 9], vec![0.5, 0.2, 0.1], "test-vocab")
            .unwrap();
        let n = Node::new(NodeId::from_bytes_raw([6u8; 16]), "Doc").with_sparse_embed(s.clone());
        let bytes = to_canonical_bytes(&n).expect("encode");
        let decoded: Node = from_canonical_bytes(&bytes).expect("decode");
        assert_eq!(decoded.sparse_embed.as_ref(), Some(&s));
        // Re-encode determinism: byte-identical.
        let bytes2 = to_canonical_bytes(&decoded).expect("re-encode");
        assert_eq!(bytes, bytes2);
    }

    #[test]
    fn node_context_sentence_round_trips() {
        let ctx = "This paragraph is from Section 3 of the 2024 lease.";
        let n = Node::new(NodeId::from_bytes_raw([9u8; 16]), "Paragraph")
            .with_summary("The tenant shall maintain the premises...")
            .with_context_sentence(ctx);
        let bytes = to_canonical_bytes(&n).expect("encode");
        let decoded: Node = from_canonical_bytes(&bytes).expect("decode");
        assert_eq!(decoded.context_sentence.as_deref(), Some(ctx));
        let bytes2 = to_canonical_bytes(&decoded).expect("re-encode");
        assert_eq!(bytes, bytes2);
    }

    #[test]
    fn node_context_sentence_absent_not_emitted() {
        // Same CID-stability property as sparse_embed: a node without
        // context_sentence must not emit the field on the wire.
        let n = Node::new(NodeId::from_bytes_raw([10u8; 16]), "Plain");
        let bytes = to_canonical_bytes(&n).expect("encode");
        assert!(
            !bytes.windows(16).any(|w| w == b"context_sentence"),
            "absent context_sentence should not appear on the wire"
        );
    }

    #[test]
    fn node_context_sentence_participates_in_cid() {
        let base = Node::new(NodeId::from_bytes_raw([11u8; 16]), "P").with_summary("x");
        let with_ctx = base.clone().with_context_sentence("cue");
        let (_, c1) = hash_to_cid(&base).unwrap();
        let (_, c2) = hash_to_cid(&with_ctx).unwrap();
        assert_ne!(c1, c2, "context_sentence must participate in the CID");
    }

    #[test]
    fn node_sparse_embed_absent_not_emitted() {
        // A node without sparse_embed must not emit "sparse_embed" on
        // the wire. This is the property that keeps pre-schema-change
        // CIDs stable when the field is not populated.
        let n = Node::new(NodeId::from_bytes_raw([7u8; 16]), "Thing");
        let bytes = to_canonical_bytes(&n).expect("encode");
        assert!(
            !bytes.windows(12).any(|w| w == b"sparse_embed"),
            "absent sparse_embed should not appear on the wire"
        );
    }

    #[test]
    fn node_sparse_embed_participates_in_cid() {
        // Two nodes identical except for sparse_embed must produce
        // different CIDs - sparse_embed is content-hash-bearing.
        let s = crate::sparse::SparseEmbed::new(vec![1], vec![1.0], "v").unwrap();
        let n_with = Node::new(NodeId::from_bytes_raw([8u8; 16]), "Doc").with_sparse_embed(s);
        let n_without = Node::new(NodeId::from_bytes_raw([8u8; 16]), "Doc");
        let (_, c_with) = hash_to_cid(&n_with).unwrap();
        let (_, c_without) = hash_to_cid(&n_without).unwrap();
        assert_ne!(c_with, c_without);
    }

    #[test]
    fn node_summary_absent_not_emitted() {
        // An unset summary must not emit a `summary: null` field on the
        // wire; skip_serializing_if keeps the CID of pre-summary nodes
        // stable.
        let n = Node::new(NodeId::from_bytes_raw([4u8; 16]), "Thing");
        let bytes = to_canonical_bytes(&n).expect("encode");
        assert!(
            !bytes.windows(7).any(|w| w == b"summary"),
            "absent summary should not appear on the wire"
        );
    }

    #[test]
    fn embedding_validate_ok_and_err() {
        let ok = Embedding {
            model: "m".into(),
            dtype: Dtype::F32,
            dim: 4,
            vector: Bytes::from(vec![0u8; 16]),
        };
        ok.validate().unwrap();

        let bad = Embedding {
            model: "m".into(),
            dtype: Dtype::F32,
            dim: 4,
            vector: Bytes::from(vec![0u8; 10]),
        };
        let err = bad.validate().unwrap_err();
        match err {
            ObjectError::EmbeddingSizeMismatch { expected, got } => {
                assert_eq!(expected, 16);
                assert_eq!(got, 10);
            }
            e => panic!("wrong variant: {e:?}"),
        }
    }
}
