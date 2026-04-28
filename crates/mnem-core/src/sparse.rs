// SPLADE, BGE-M3, BEIR, WordPiece, OpenSearch are well-known external
// identifiers; backticking every mention in the module doc degrades
// rendered rustdoc readability.
#![allow(clippy::doc_markdown)]

//! Sparse (learned) embedding primitives for SPLADE / BGE-M3-sparse
//! integration .
//!
//! # Why
//!
//! Learned-sparse retrievers (SPLADE v3, opensearch-doc-v3-distill,
//! BGE-M3-sparse, granite-embedding-30m-sparse) produce a sparse
//! vector over a WordPiece vocabulary that can be scored via an
//! inverted index with semantic term weights learned end-to-end.
//! BEIR nDCG@10 on sparse neural retrievers lands around +3-5 points
//! over classical lexical keyword scoring on zero-shot domains; this
//! lane replaces that legacy lexical lane entirely .
//!
//! # What this module provides
//!
//! - [`SparseEmbed`] - canonical sparse-vector shape (ascending
//!   `indices` + aligned `values`) with a `vocab_id` tag so two
//!   models with different vocabularies never get mixed in one
//!   posting list.
//! - [`SparseEncoder`] trait - adapter-side hook for ONNX / candle
//!   backends to implement. Mirrors the [`crate::rerank::Reranker`]
//!   trait shape.
//! - `MockSparseEncoder` - deterministic test-only encoder.
//!
//! The actual inverted-index over `SparseEmbed` values lives in
//! [`crate::index::sparse`] so the index stays next to its sibling
//! (brute-force vector index).
//!
//! Storage in [`crate::objects::Node`]: a future `Node.sparse_embed:
//! Option<SparseEmbed>` field. Additive, so existing CIDs stay
//! byte-identical because the serializer omits `None` via
//! `skip_serializing_if`. CBOR canonicality is preserved because
//! `indices` is sorted ascending at construction (checked by
//! [`SparseEmbed::new`]).

use std::fmt::Debug;

use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Error surface for sparse-encoder adapters. Same shape as
/// [`crate::llm::LlmError`] and [`crate::rerank::RerankError`].
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum SparseError {
    /// Network / transport failure when the adapter runs remotely
    /// (sidecar) or fetches weights.
    #[error("network error: {0}")]
    Network(String),
    /// Adapter config invalid (missing weights file, bad URL, etc.).
    #[error("config error: {0}")]
    Config(String),
    /// Model / tokenizer returned an error.
    #[error("inference error: {0}")]
    Inference(String),
    /// Caller attempted to encode empty text.
    #[error("empty input")]
    EmptyInput,
}

/// A sparse embedding over a fixed vocabulary.
///
/// `indices` MUST be strictly ascending; `values` MUST have the same
/// length as `indices`. Both invariants are checked by [`Self::new`]
/// and enforced on deserialise in a future CBOR round-trip test.
/// `vocab_id` pins the model family so two adapters with different
/// vocabs never fuse posting lists; compare as a string (e.g.
/// `"bert-base-uncased@30522"`).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SparseEmbed {
    /// Token IDs in the vocabulary, strictly ascending.
    pub indices: Vec<u32>,
    /// Non-zero weights, aligned with `indices`.
    pub values: Vec<f32>,
    /// Vocabulary identifier.
    pub vocab_id: String,
}

impl SparseEmbed {
    /// Construct a [`SparseEmbed`]. Panics (debug) / errors (release)
    /// if the invariants are violated. `indices` is taken as-is; if
    /// the caller is unsure whether it is sorted, use
    /// [`Self::from_unsorted`] instead.
    ///
    /// # Errors
    ///
    /// - [`SparseError::Config`] if `indices.len() != values.len()`
    ///   or `indices` contains duplicates / non-ascending entries.
    pub fn new(
        indices: Vec<u32>,
        values: Vec<f32>,
        vocab_id: impl Into<String>,
    ) -> Result<Self, SparseError> {
        if indices.len() != values.len() {
            return Err(SparseError::Config(format!(
                "indices.len() {} != values.len() {}",
                indices.len(),
                values.len()
            )));
        }
        for w in indices.windows(2) {
            if w[0] >= w[1] {
                return Err(SparseError::Config(format!(
                    "indices must be strictly ascending; saw {} then {}",
                    w[0], w[1]
                )));
            }
        }
        Ok(Self {
            indices,
            values,
            vocab_id: vocab_id.into(),
        })
    }

    /// Construct from an unsorted `(index, value)` pair list.
    /// Duplicate indices are kept as the maximum value (SPLADE's
    /// own pooling rule). Useful from ONNX-side decoders that
    /// produce vectors in token-emission order.
    pub fn from_unsorted(
        pairs: impl IntoIterator<Item = (u32, f32)>,
        vocab_id: impl Into<String>,
    ) -> Self {
        use std::collections::BTreeMap;
        let mut bucket: BTreeMap<u32, f32> = BTreeMap::new();
        for (i, v) in pairs {
            let e = bucket.entry(i).or_insert(f32::NEG_INFINITY);
            if v > *e {
                *e = v;
            }
        }
        let (indices, values): (Vec<_>, Vec<_>) =
            bucket.into_iter().filter(|(_, v)| *v > 0.0).unzip();
        Self {
            indices,
            values,
            vocab_id: vocab_id.into(),
        }
    }

    /// Number of non-zero entries.
    #[must_use]
    pub const fn nnz(&self) -> usize {
        self.indices.len()
    }

    /// Dot product with another sparse embedding (must share vocab_id).
    /// Returns `None` if the vocab_ids differ.
    #[must_use]
    pub fn dot(&self, other: &Self) -> Option<f32> {
        if self.vocab_id != other.vocab_id {
            return None;
        }
        let mut i = 0;
        let mut j = 0;
        let mut sum = 0.0f32;
        while i < self.indices.len() && j < other.indices.len() {
            use std::cmp::Ordering;
            match self.indices[i].cmp(&other.indices[j]) {
                Ordering::Less => i += 1,
                Ordering::Greater => j += 1,
                Ordering::Equal => {
                    sum += self.values[i] * other.values[j];
                    i += 1;
                    j += 1;
                }
            }
        }
        Some(sum)
    }
}

/// Learned-sparse encoder: given text, produce a [`SparseEmbed`]
/// over a fixed vocabulary. Adapter crates implement this over
/// SPLADE-ONNX, BGE-M3-sparse-ONNX, or a remote sidecar.
pub trait SparseEncoder: Send + Sync + Debug {
    /// Provider + model identifier. Lowercase, colon-separated by
    /// convention (e.g. `"splade:opensearch-doc-v3-distill"`,
    /// `"bgem3:sparse"`, `"mock:len-inverse"`).
    fn model(&self) -> &str;

    /// Vocabulary identifier. Passed through to
    /// [`SparseEmbed::vocab_id`] on every emitted embedding.
    fn vocab_id(&self) -> &str;

    /// Encode a document-side text string into a sparse vector.
    /// This is the path run at ingest time.
    ///
    /// # Errors
    ///
    /// Any [`SparseError`] the adapter surfaces. The caller fallback
    /// policy matches the rerank / LLM pattern: on error, the sparse
    /// lane is simply dropped from fusion and the hybrid still runs.
    fn encode(&self, text: &str) -> Result<SparseEmbed, SparseError>;

    /// Encode a query-side text string into a sparse vector.
    ///
    /// Default implementation delegates to [`Self::encode`]. Adapters
    /// with asymmetric inference (OpenSearch
    /// `neural-sparse-encoding-doc-v3-distill` ships a distilled
    /// `idf.json` table so the query side is tokenise + IDF-lookup
    /// with zero neural compute) override this to skip the forward
    /// pass. The overridden path keeps retrieval latency microsecond-
    /// level even when documents use a 67M-parameter encoder.
    fn encode_query(&self, text: &str) -> Result<SparseEmbed, SparseError> {
        self.encode(text)
    }
}

/// FNV-1a 32-bit offset basis. Standard value from the FNV
/// specification (Fowler-Noll-Vo hash, Landon Curt Noll, 1991);
/// see <http://www.isthe.com/chongo/tech/comp/fnv/>. Used as the
/// seed state for the `MockSparseEncoder` token hash.
const FNV_OFFSET_BASIS_32: u32 = 2_166_136_261;

/// FNV-1a 32-bit prime. Standard value from the FNV specification
/// (<http://www.isthe.com/chongo/tech/comp/fnv/>). Each byte of the
/// input is XOR-then-multiplied by this prime to diffuse bits
/// across the output word.
const FNV_PRIME_32: u32 = 16_777_619;

/// Mock vocabulary width. The token hash is reduced modulo this
/// number of slots so the encoder emits indices in `0..1024`,
/// matching the `"mock:1024"` `vocab_id` tag. Kept tiny so tests
/// exercise collision handling cheaply.
const MOCK_VOCAB_SIZE: u32 = 1024;

/// Deterministic test-only encoder. Produces a `SparseEmbed` by
/// hashing each whitespace-separated token into the first 1024
/// vocabulary slots with a length-inverse weight
/// (1.0 / (1.0 + token_len)).
///
/// Not a real SPLADE; do not use in benchmarks. Its purpose is to
/// let `Retriever::with_sparse_ranker(...)` unit-test the fusion
/// lane without pulling ONNX Runtime into `mnem-core`'s test deps.
#[derive(Debug, Clone)]
pub struct MockSparseEncoder {
    vocab_id: String,
}

impl Default for MockSparseEncoder {
    fn default() -> Self {
        Self {
            vocab_id: "mock:1024".into(),
        }
    }
}

impl SparseEncoder for MockSparseEncoder {
    fn model(&self) -> &str {
        "mock:len-inverse"
    }

    fn vocab_id(&self) -> &str {
        &self.vocab_id
    }

    fn encode(&self, text: &str) -> Result<SparseEmbed, SparseError> {
        if text.trim().is_empty() {
            return Err(SparseError::EmptyInput);
        }
        let pairs = text.split_whitespace().map(|tok| {
            let h: u32 = tok.bytes().fold(FNV_OFFSET_BASIS_32, |acc, b| {
                acc.wrapping_mul(FNV_PRIME_32).wrapping_add(u32::from(b))
            });
            let idx = h % MOCK_VOCAB_SIZE;
            let weight = 1.0f32 / (1.0 + tok.len() as f32);
            (idx, weight)
        });
        Ok(SparseEmbed::from_unsorted(pairs, &self.vocab_id))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sparse_embed_rejects_unsorted_indices() {
        let e = SparseEmbed::new(vec![5, 3], vec![0.5, 0.5], "v0").unwrap_err();
        assert!(matches!(e, SparseError::Config(_)));
    }

    #[test]
    fn sparse_embed_rejects_length_mismatch() {
        let e = SparseEmbed::new(vec![1, 2], vec![0.5], "v0").unwrap_err();
        assert!(matches!(e, SparseError::Config(_)));
    }

    #[test]
    fn from_unsorted_sorts_and_max_pools() {
        let s = SparseEmbed::from_unsorted([(5, 0.1), (3, 0.9), (5, 0.3), (1, 0.2)], "v0");
        assert_eq!(s.indices, vec![1, 3, 5]);
        assert!(
            (s.values[2] - 0.3).abs() < 1e-6,
            "max-pool should keep 0.3 for index 5"
        );
    }

    #[test]
    fn from_unsorted_drops_zero_weights() {
        let s = SparseEmbed::from_unsorted([(1, 0.0), (2, 0.5), (3, -0.1)], "v0");
        assert_eq!(s.indices, vec![2]);
    }

    #[test]
    fn dot_product_on_disjoint_is_zero() {
        let a = SparseEmbed::new(vec![1, 2], vec![1.0, 1.0], "v").unwrap();
        let b = SparseEmbed::new(vec![3, 4], vec![1.0, 1.0], "v").unwrap();
        assert_eq!(a.dot(&b), Some(0.0));
    }

    #[test]
    fn dot_product_on_overlap() {
        let a = SparseEmbed::new(vec![1, 2, 5], vec![0.5, 0.5, 0.2], "v").unwrap();
        let b = SparseEmbed::new(vec![2, 5, 9], vec![0.4, 0.3, 0.1], "v").unwrap();
        // Overlap at 2 (0.5*0.4=0.2) and 5 (0.2*0.3=0.06) -> 0.26.
        let d = a.dot(&b).unwrap();
        assert!((d - 0.26).abs() < 1e-6, "got {d}");
    }

    #[test]
    fn dot_product_different_vocabs_is_none() {
        let a = SparseEmbed::new(vec![1], vec![1.0], "v0").unwrap();
        let b = SparseEmbed::new(vec![1], vec![1.0], "v1").unwrap();
        assert_eq!(a.dot(&b), None);
    }

    #[test]
    fn mock_encoder_is_deterministic() {
        let e = MockSparseEncoder::default();
        let a = e.encode("hello world").unwrap();
        let b = e.encode("hello world").unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn mock_encoder_empty_input_errors() {
        let e = MockSparseEncoder::default();
        assert!(matches!(
            e.encode("   ").unwrap_err(),
            SparseError::EmptyInput
        ));
    }

    #[test]
    fn mock_encoder_vocab_id_carries_through() {
        let e = MockSparseEncoder::default();
        let emb = e.encode("hello").unwrap();
        assert_eq!(emb.vocab_id, e.vocab_id());
    }
}
