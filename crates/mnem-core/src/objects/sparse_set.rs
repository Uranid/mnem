//! `SparseBucket` - per-node leaf object inside the Prolly sidecar
//! that lifts the sparse embedding out of the
//! [`Node`](super::Node) canonical bytes.
//!
//! # Why this exists
//!
//! When the sparse embedding lives inline on `Node`:
//!
//! ```text
//! NodeCid = blake3(canonical_bytes(Node)) // includes sparse_embed
//! ```
//!
//! Different sparse encoders and vocabulary differences produce
//! different byte representations, so two machines indexing the same
//! logical source text with different encoder versions produce different
//! `NodeCid` values. That breaks mnem's federated-dedup promise.
//!
//! Fix: sparse embeddings live in a separate Prolly tree referenced by
//! `Commit.sparse: Option<Cid>` (the sibling slot to
//! `Commit.embeddings`). The tree is keyed by 16-byte truncated blake3
//! of the `NodeCid` wire form; values are `SparseBucket`s carrying one
//! `(vocab_id, SparseEmbed)` pair per indexed vocabulary. Identity bytes
//! (Node) and derived bytes (SparseEmbed) are content-addressed
//! independently. Vocab differences no longer leak into Node CIDs.
//!
//! # Pattern source
//!
//! Mirrors the [`EmbeddingBucket`](super::EmbeddingBucket) shape from
//! G16 and the [`AdjacencyBucket`](super::AdjacencyBucket) shape from
//! the existing [`IndexSet`](super::IndexSet) sidecar: sorted entry
//! list inside each leaf, hand-rolled `Serialize`/`Deserialize`
//! carrying a `_kind` discriminator and a `#[serde(flatten)] extra`
//! forward-compat carrier so unrelated schema bumps stay
//! round-trippable.

use std::collections::BTreeMap;

use ipld_core::ipld::Ipld;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::sparse::SparseEmbed;

/// Per-node bucket of sparse embeddings inside the
/// [`Commit.sparse`](super::Commit::sparse) Prolly tree.
///
/// One bucket per node. Each bucket holds a sorted
/// `(vocab_id, SparseEmbed)` list so a node may carry multiple
/// sparse embeddings simultaneously - e.g. one BGE-M3 vector plus
/// one OpenSearch-distill vector for the same chunk text. Lookups
/// index into the bucket by `vocab_id` string after the outer Prolly
/// walk has returned the bucket itself.
#[derive(Clone, Debug, Default)]
pub struct SparseBucket {
    /// Entries sorted lexicographically by `vocab_id` for byte-stable
    /// canonical form. The sort is enforced on every serialize, so
    /// callers may push entries in any order without breaking CID
    /// determinism on the bucket itself.
    pub entries: Vec<SparseEntry>,
    /// Forward-compat extension carrier. Unknown CBOR fields land
    /// here on decode and are emitted verbatim on re-encode, so a
    /// future schema bump that adds a per-bucket field stays
    /// round-trippable on today's reader.
    pub extra: BTreeMap<String, Ipld>,
}

impl SparseBucket {
    /// On-wire `_kind` discriminator. Every content-addressed object
    /// in mnem/0.x carries a `_kind` field as the first canonical key
    /// so a corrupt bucket / wrong-type decode fails fast with an
    /// actionable error instead of silently mis-decoding.
    pub const KIND: &'static str = "sparse_bucket";

    /// Look up a sparse embedding by vocab_id string. Returns `None` when
    /// this bucket has no entry for the requested vocabulary; the caller
    /// typically falls back to lazy compute via the configured sparse
    /// encoder adapter.
    #[must_use]
    pub fn get(&self, vocab_id: &str) -> Option<&SparseEmbed> {
        self.entries
            .iter()
            .find(|e| e.vocab_id == vocab_id)
            .map(|e| &e.sparse)
    }

    /// Insert or replace an entry by `vocab_id`. The bucket does not
    /// return the previous value (unlike `EmbeddingBucket::upsert`)
    /// because `SparseEmbed` contains `Vec<f32>` which is not `PartialEq`
    /// in a meaningful sense for callers here.
    pub fn upsert(&mut self, vocab_id: String, sparse: SparseEmbed) {
        if let Some(slot) = self.entries.iter_mut().find(|e| e.vocab_id == vocab_id) {
            slot.sparse = sparse;
            return;
        }
        self.entries.push(SparseEntry { vocab_id, sparse });
    }

    /// Remove an entry by `vocab_id`.
    pub fn remove(&mut self, vocab_id: &str) {
        if let Some(i) = self.entries.iter().position(|e| e.vocab_id == vocab_id) {
            self.entries.remove(i);
        }
    }
}

/// One `(vocab_id, SparseEmbed)` pair inside a [`SparseBucket`].
///
/// Kept as a separate type rather than a tuple so future schema bumps
/// can add per-entry fields (provenance, deprecation, signature) under
/// the same canonical-form contract every other mnem object uses.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SparseEntry {
    /// Vocabulary identifier. Conventionally a short string identifying
    /// the encoder and vocab (e.g. `"bge-m3"`, `"opensearch-distill-v3"`).
    /// The bucket indexes on this exact string for `get` / `upsert` / `remove`.
    pub vocab_id: String,
    /// The sparse embedding produced by the encoder. Contains `indices`
    /// (token ids) and `values` (non-zero weights) alongside `vocab_id`.
    pub sparse: SparseEmbed,
}

// ---------------- Serde wire shape ----------------
//
// Same hand-rolled pattern as `Node`/`Commit`/`EmbeddingBucket`:
// internal `*Wire` mirror with explicit field defaults +
// `_kind` discriminator + `extra` flatten. Encode sorts entries by
// `vocab_id` so bucket bytes (and therefore the bucket CID) are
// independent of insertion order. Decode rejects wrong `_kind`
// values up front.

#[derive(Serialize, Deserialize)]
struct SparseBucketWire {
    #[serde(rename = "_kind")]
    kind: String,
    #[serde(default)]
    entries: Vec<SparseEntry>,
    #[serde(flatten, default, skip_serializing_if = "BTreeMap::is_empty")]
    extra: BTreeMap<String, Ipld>,
}

impl Serialize for SparseBucket {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        // Canonical order: sorted by `vocab_id`. We clone-then-sort
        // rather than mutate the borrowed field - the public API
        // does not promise any particular insertion order, but the
        // wire form is contract-bound to be deterministic.
        let mut sorted = self.entries.clone();
        sorted.sort_by(|a, b| a.vocab_id.cmp(&b.vocab_id));
        SparseBucketWire {
            kind: Self::KIND.into(),
            entries: sorted,
            extra: self.extra.clone(),
        }
        .serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for SparseBucket {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let w = SparseBucketWire::deserialize(deserializer)?;
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
    use crate::sparse::SparseEmbed;

    fn sample_sparse(vocab_id: &str) -> SparseEmbed {
        SparseEmbed::new(vec![1, 5, 9], vec![0.5, 0.2, 0.1], vocab_id).unwrap()
    }

    #[test]
    fn empty_bucket_round_trips() {
        let original = SparseBucket::default();
        let bytes = to_canonical_bytes(&original).unwrap();
        let decoded: SparseBucket = from_canonical_bytes(&bytes).unwrap();
        assert_eq!(original.entries.len(), decoded.entries.len());
        let bytes2 = to_canonical_bytes(&decoded).unwrap();
        assert_eq!(bytes, bytes2, "round-trip must be byte-identical");
    }

    #[test]
    fn populated_bucket_round_trips() {
        let mut bucket = SparseBucket::default();
        bucket.upsert("bge-m3".into(), sample_sparse("bge-m3"));
        bucket.upsert(
            "opensearch-distill".into(),
            sample_sparse("opensearch-distill"),
        );
        let bytes = to_canonical_bytes(&bucket).unwrap();
        let decoded: SparseBucket = from_canonical_bytes(&bytes).unwrap();
        assert_eq!(bucket.entries.len(), decoded.entries.len());
    }

    #[test]
    fn wire_form_sorts_by_vocab_id_regardless_of_insert_order() {
        // Insert in z-then-a order; canonical bytes must equal the
        // alphabetical order's canonical bytes.
        let mut a = SparseBucket::default();
        a.upsert("zzz".into(), sample_sparse("zzz"));
        a.upsert("aaa".into(), sample_sparse("aaa"));
        let mut b = SparseBucket::default();
        b.upsert("aaa".into(), sample_sparse("aaa"));
        b.upsert("zzz".into(), sample_sparse("zzz"));
        assert_eq!(
            to_canonical_bytes(&a).unwrap(),
            to_canonical_bytes(&b).unwrap(),
            "encode must sort entries by vocab_id so bucket CIDs are insertion-order-invariant"
        );
    }

    #[test]
    fn wrong_kind_fails_decode() {
        #[derive(Serialize)]
        struct Wrong {
            #[serde(rename = "_kind")]
            kind: String,
            entries: Vec<SparseEntry>,
        }
        let bytes = serde_ipld_dagcbor::to_vec(&Wrong {
            kind: "node".into(),
            entries: vec![],
        })
        .unwrap();
        let res: Result<SparseBucket, _> = from_canonical_bytes(&bytes);
        assert!(res.is_err(), "decode must reject wrong _kind discriminator");
        let msg = format!("{}", res.unwrap_err());
        assert!(
            msg.contains("sparse_bucket"),
            "error must reference the expected kind; got: {msg}"
        );
    }

    #[test]
    fn upsert_overwrites_existing_entry() {
        let mut bucket = SparseBucket::default();
        bucket.upsert("v0".into(), sample_sparse("v0"));
        // Upsert again with different data.
        let new_sparse = SparseEmbed::new(vec![100], vec![0.9], "v0").unwrap();
        bucket.upsert("v0".into(), new_sparse);
        // Only one entry should exist.
        assert_eq!(bucket.entries.len(), 1);
        assert_eq!(bucket.get("v0").unwrap().indices, vec![100]);
    }

    #[test]
    fn get_finds_inserted_entry() {
        let mut bucket = SparseBucket::default();
        let sp = sample_sparse("v0");
        bucket.upsert("v0".into(), sp.clone());
        assert_eq!(bucket.get("v0").unwrap().vocab_id, "v0");
        assert!(bucket.get("missing").is_none());
    }

    #[test]
    fn extra_fields_round_trip() {
        let mut bucket = SparseBucket::default();
        bucket
            .extra
            .insert("future_field".into(), Ipld::String("forward-compat".into()));
        let bytes = to_canonical_bytes(&bucket).unwrap();
        let decoded: SparseBucket = from_canonical_bytes(&bytes).unwrap();
        assert_eq!(bucket.extra.len(), decoded.extra.len());
        assert!(decoded.extra.contains_key("future_field"));
    }
}
