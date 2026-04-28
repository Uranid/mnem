//! `EmbeddingBucket` - per-node leaf object inside the Prolly sidecar
//! that lifts the embedding vector out of the
//! [`Node`](super::Node) canonical bytes.
//!
//! # Why this exists
//!
//! When the embedding vector lives inline on `Node`:
//!
//! ```text
//! NodeCid = blake3(canonical_bytes(Node)) // includes embed.vector
//! ```
//!
//! ORT reorders f32 sums across thread counts (TBB-style work-stealing
//! reductions are not associative on `f32`), so two machines re-deriving
//! the same source text on different core counts produce vectors that
//! differ in the last bit. Different vector → different Node bytes →
//! different `NodeCid` for embed-bearing chunks. That breaks mnem's
//! "two machines indexing the same logical event produce identical
//! Node CIDs" federated-dedup promise as soon as the runtime uses
//! `available_parallelism()` instead of a single thread.
//!
//! Fix: vectors live in a separate Prolly tree referenced by
//! `Commit.embeddings: Option<Cid>` (the sibling slot to
//! `Commit.indexes`). The tree is keyed by 32-byte `NodeCid` digest;
//! values are `EmbeddingBucket`s carrying one `(model, Embedding)`
//! pair per simultaneously-indexed embedder. Identity bytes (Node)
//! and derived bytes (Embedding) are content-addressed independently.
//! Multi-thread ORT no longer leaks into Node CIDs.
//!
//! # Pattern source
//!
//! Mirrors the [`AdjacencyBucket`](super::AdjacencyBucket) shape from
//! the existing [`IndexSet`](super::IndexSet) sidecar: sorted entry
//! list inside each leaf, hand-rolled `Serialize`/`Deserialize`
//! carrying a `_kind` discriminator and a `#[serde(flatten)] extra`
//! forward-compat carrier so unrelated schema bumps stay
//! round-trippable.

use std::collections::BTreeMap;

use ipld_core::ipld::Ipld;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use super::node::Embedding;

/// Per-node bucket of embeddings inside the
/// [`Commit.embeddings`](super::Commit::embeddings) Prolly tree.
///
/// One bucket per node. Each bucket holds a sorted
/// `(model, Embedding)` list so a node may carry multiple
/// embeddings simultaneously - e.g. one local MiniLM vector plus
/// one OpenAI vector for the same chunk text. Lookups index into
/// the bucket by `model` string after the outer Prolly walk has
/// returned the bucket itself.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct EmbeddingBucket {
    /// Entries sorted lexicographically by `model` for byte-stable
    /// canonical form. The sort is enforced on every serialize, so
    /// callers may push entries in any order without breaking CID
    /// determinism on the bucket itself.
    pub entries: Vec<EmbeddingEntry>,
    /// Forward-compat extension carrier. Unknown CBOR fields land
    /// here on decode and are emitted verbatim on re-encode, so a
    /// future schema bump that adds a per-bucket field stays
    /// round-trippable on today's reader.
    pub extra: BTreeMap<String, Ipld>,
}

impl EmbeddingBucket {
    /// On-wire `_kind` discriminator. Every content-addressed object
    /// in mnem/0.x carries a `_kind` field as the first canonical key
    /// so a corrupt bucket / wrong-type decode fails fast with an
    /// actionable error instead of silently mis-decoding.
    pub const KIND: &'static str = "embedding_bucket";

    /// Look up an embedding by model string. Returns `None` when this
    /// bucket has no entry for the requested embedder; the caller
    /// typically falls back to lazy compute via the configured
    /// embed provider.
    #[must_use]
    pub fn get(&self, model: &str) -> Option<&Embedding> {
        self.entries
            .iter()
            .find(|e| e.model == model)
            .map(|e| &e.embedding)
    }

    /// Insert or replace an entry by `model`. Returns the previous
    /// embedding for that model when one existed (so callers can
    /// detect a refresh vs first write).
    pub fn upsert(&mut self, model: String, embedding: Embedding) -> Option<Embedding> {
        if let Some(slot) = self.entries.iter_mut().find(|e| e.model == model) {
            return Some(std::mem::replace(&mut slot.embedding, embedding));
        }
        self.entries.push(EmbeddingEntry { model, embedding });
        None
    }

    /// Remove an entry by `model`. Returns the removed embedding when
    /// one existed.
    pub fn remove(&mut self, model: &str) -> Option<Embedding> {
        let i = self.entries.iter().position(|e| e.model == model)?;
        Some(self.entries.remove(i).embedding)
    }
}

/// One `(model, Embedding)` pair inside an [`EmbeddingBucket`].
///
/// Kept as a separate type rather than a tuple so future schema bumps
/// can add per-entry fields (provenance, deprecation, signature) under
/// the same canonical-form contract every other mnem object uses.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct EmbeddingEntry {
    /// Embedder identifier. Conventionally `"<provider>:<model>"`
    /// (matches the `model` string inside `Embedding`); the bucket
    /// indexes on this exact string for `get` / `upsert` / `remove`.
    pub model: String,
    /// The embedding vector and dim/dtype metadata. Its own
    /// `validate()` invariant (`vector.len() == dim * dtype.byte_width()`)
    /// is enforced where embeddings are constructed; the bucket does
    /// not re-validate on decode (cheap-decode contract). Untrusted
    /// callers (HTTP / MCP / replication) are expected to call
    /// `Embedding::validate()` themselves before storing.
    pub embedding: Embedding,
}

// ---------------- Serde wire shape ----------------
//
// Same hand-rolled pattern as `Node`/`Commit`/`AdjacencyBucket`:
// internal `*Wire` mirror with explicit field defaults +
// `_kind` discriminator + `extra` flatten. Encode sorts entries by
// `model` so bucket bytes (and therefore the bucket CID) are
// independent of insertion order. Decode rejects wrong `_kind`
// values up front.

#[derive(Serialize, Deserialize)]
struct EmbeddingBucketWire {
    #[serde(rename = "_kind")]
    kind: String,
    #[serde(default)]
    entries: Vec<EmbeddingEntry>,
    #[serde(flatten, default, skip_serializing_if = "BTreeMap::is_empty")]
    extra: BTreeMap<String, Ipld>,
}

impl Serialize for EmbeddingBucket {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        // Canonical order: sorted by `model`. We clone-then-sort
        // rather than mutate the borrowed field - the public API
        // does not promise any particular insertion order, but the
        // wire form is contract-bound to be deterministic.
        let mut sorted = self.entries.clone();
        sorted.sort_by(|a, b| a.model.cmp(&b.model));
        EmbeddingBucketWire {
            kind: Self::KIND.into(),
            entries: sorted,
            extra: self.extra.clone(),
        }
        .serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for EmbeddingBucket {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let w = EmbeddingBucketWire::deserialize(deserializer)?;
        if w.kind != Self::KIND {
            return Err(serde::de::Error::custom(format!(
                "expected _kind='{}', got '{}'",
                Self::KIND,
                w.kind
            )));
        }
        Ok(Self {
            entries: w.entries,
            extra: w.extra,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codec::{from_canonical_bytes, to_canonical_bytes};
    use crate::objects::node::Dtype;

    fn sample_embedding(model: &str, dim: u32) -> Embedding {
        // Cheap deterministic dummy vector: one f32 per dim, all
        // zeroes. Exercises the validate invariant while staying
        // independent of any embedder.
        let bytes_len = (dim as usize) * Dtype::F32.byte_width();
        Embedding {
            model: model.into(),
            dtype: Dtype::F32,
            dim,
            vector: bytes::Bytes::from(vec![0u8; bytes_len]),
        }
    }

    #[test]
    fn empty_bucket_round_trips() {
        let original = EmbeddingBucket::default();
        let bytes = to_canonical_bytes(&original).unwrap();
        let decoded: EmbeddingBucket = from_canonical_bytes(&bytes).unwrap();
        assert_eq!(original, decoded);
        let bytes2 = to_canonical_bytes(&decoded).unwrap();
        assert_eq!(bytes, bytes2, "round-trip must be byte-identical");
    }

    #[test]
    fn populated_bucket_round_trips() {
        let mut bucket = EmbeddingBucket::default();
        bucket.upsert(
            "openai:text-embedding-3-small".into(),
            sample_embedding("openai:text-embedding-3-small", 1536),
        );
        bucket.upsert(
            "onnx:all-MiniLM-L6-v2".into(),
            sample_embedding("onnx:all-MiniLM-L6-v2", 384),
        );
        let bytes = to_canonical_bytes(&bucket).unwrap();
        let decoded: EmbeddingBucket = from_canonical_bytes(&bytes).unwrap();
        assert_eq!(bucket.entries.len(), decoded.entries.len());
        // Decoded copy is canonical (sorted by model). Sort the
        // original before equating because the public API does not
        // promise input order is preserved.
        let mut sorted_orig = bucket.entries.clone();
        sorted_orig.sort_by(|a, b| a.model.cmp(&b.model));
        assert_eq!(sorted_orig, decoded.entries);
    }

    #[test]
    fn wire_form_sorts_by_model_regardless_of_insert_order() {
        // Insert in z-then-a order; canonical bytes must equal the
        // alphabetical order's canonical bytes.
        let mut a = EmbeddingBucket::default();
        a.upsert("zzz".into(), sample_embedding("zzz", 4));
        a.upsert("aaa".into(), sample_embedding("aaa", 4));
        let mut b = EmbeddingBucket::default();
        b.upsert("aaa".into(), sample_embedding("aaa", 4));
        b.upsert("zzz".into(), sample_embedding("zzz", 4));
        assert_eq!(
            to_canonical_bytes(&a).unwrap(),
            to_canonical_bytes(&b).unwrap(),
            "encode must sort entries by model so bucket CIDs are insertion-order-invariant"
        );
    }

    #[test]
    fn wrong_kind_fails_decode() {
        // Manually craft a CBOR map with `_kind = "node"` and verify
        // the EmbeddingBucket decoder rejects it. Uses
        // serde_ipld_dagcbor::to_vec on a small inline struct rather
        // than hand-encoding bytes.
        #[derive(Serialize)]
        struct Wrong {
            #[serde(rename = "_kind")]
            kind: String,
            entries: Vec<EmbeddingEntry>,
        }
        let bytes = serde_ipld_dagcbor::to_vec(&Wrong {
            kind: "node".into(),
            entries: vec![],
        })
        .unwrap();
        let res: Result<EmbeddingBucket, _> = from_canonical_bytes(&bytes);
        assert!(res.is_err(), "decode must reject wrong _kind discriminator");
        let msg = format!("{}", res.unwrap_err());
        assert!(
            msg.contains("embedding_bucket"),
            "error must reference the expected kind; got: {msg}"
        );
    }

    #[test]
    fn upsert_returns_previous_value_on_replace() {
        let mut bucket = EmbeddingBucket::default();
        let first = sample_embedding("m", 4);
        let second = sample_embedding("m", 4);
        assert_eq!(bucket.upsert("m".into(), first.clone()), None);
        assert_eq!(bucket.upsert("m".into(), second), Some(first));
    }

    #[test]
    fn get_finds_inserted_entry() {
        let mut bucket = EmbeddingBucket::default();
        let emb = sample_embedding("m", 4);
        bucket.upsert("m".into(), emb.clone());
        assert_eq!(bucket.get("m"), Some(&emb));
        assert_eq!(bucket.get("missing"), None);
    }

    #[test]
    fn remove_removes_existing_entry() {
        let mut bucket = EmbeddingBucket::default();
        let emb = sample_embedding("m", 4);
        bucket.upsert("m".into(), emb.clone());
        assert_eq!(bucket.remove("m"), Some(emb));
        assert_eq!(bucket.get("m"), None);
        assert_eq!(bucket.remove("m"), None);
    }

    #[test]
    fn extra_fields_round_trip() {
        // Forward-compat: a future schema bump adding a sidecar field
        // (e.g. `provenance`) on a bucket should round-trip through
        // today's reader. Simulate by manually injecting an `extra`
        // entry, encoding, and asserting the decoded bucket carries it.
        let mut bucket = EmbeddingBucket::default();
        bucket
            .extra
            .insert("future_field".into(), Ipld::String("forward-compat".into()));
        let bytes = to_canonical_bytes(&bucket).unwrap();
        let decoded: EmbeddingBucket = from_canonical_bytes(&bytes).unwrap();
        assert_eq!(bucket, decoded, "extra fields must survive round-trip");
        assert!(decoded.extra.contains_key("future_field"));
    }
}
