//! Reranker trait: post-fusion rescoring by a model that jointly
//! encodes `(query, candidate)` pairs.
//!
//! # Why
//!
//! The retrieve pipeline's base rankers do not read `(query, candidate)`
//! jointly:
//!
//! - The dense vector ranker is a bi-encoder. It embeds the whole query
//!   into one vector and each doc summary into one vector, then compares
//!   cosine similarity. It sees phrases but produces only one score per
//!   doc; the embeddings are encoded independently and never read
//!   together.
//! - The learned-sparse ranker scores via a sparse dot product over a
//!   shared vocabulary. It too is a bi-encoder.
//!
//! For compositional paraphrase like "father's sister == aunt", you
//! need a model that reads the query and a candidate side-by-side and
//! scores their relevance as a pair. That is a cross-encoder.
//!
//! # What this module provides
//!
//! A [`Reranker`] trait that adapter crates implement. Industry
//! cross-encoder providers (Cohere rerank, Voyage rerank, Jina rerank,
//! local BGE-reranker ONNX) all fit this shape.
//!
//! Mnem-core stays tokio-free; adapter crates live next to
//! [`mnem-embed-providers`](https://github.com/Uranid/mnem) and
//! do the HTTP work. This file contains only the trait, the error
//! type, and a deterministic mock for tests.
//!
//! # How it plugs in
//!
//! [`crate::retrieve::Retriever::with_reranker`] takes a
//! `Arc<dyn Reranker>`. If set, the retriever re-scores the top-K of
//! the fused list before budget packing. Failures fall back to the
//! original fused order (same graceful-degrade policy as the embedder
//! auto-fuse in the CLI).
//!
//!
use std::fmt::Debug;

use thiserror::Error;

/// Error surface for cross-encoder reranker adapters.
///
/// Marked `#[non_exhaustive]` so provider crates can grow their own
/// failure modes without a breaking change here.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum RerankError {
    /// TLS / TCP / DNS / timeout failure reaching the provider.
    #[error("network error: {0}")]
    Network(String),
    /// Provider rejected credentials.
    #[error("authentication failed: {0}")]
    Auth(String),
    /// Provider rate-limited the request.
    #[error("rate limited: {0}")]
    RateLimited(String),
    /// 4xx from the provider.
    #[error("bad request ({status}): {body}")]
    BadRequest {
        /// HTTP status code.
        status: u16,
        /// Response body or best-effort error string.
        body: String,
    },
    /// 5xx from the provider.
    #[error("server error ({status}): {body}")]
    Server {
        /// HTTP status code.
        status: u16,
        /// Response body or best-effort error string.
        body: String,
    },
    /// Response decoder failed (malformed JSON, missing score field, ...).
    #[error("decode error: {0}")]
    Decode(String),
    /// Adapter config invalid (bad URL, missing env var, etc.).
    #[error("config error: {0}")]
    Config(String),
    /// Model / tokenizer / ONNX session runtime failure, distinct from
    /// config-time validation. Mirrors [`crate::sparse::SparseError::Inference`]
    /// so sibling provider traits surface runtime failures with a
    /// consistent shape.
    #[error("inference error: {0}")]
    Inference(String),
    /// Provider returned a different number of scores than candidates.
    /// Implementations MUST reject this up front; the retriever would
    /// otherwise zip mismatched pairs.
    #[error("score count mismatch: expected {expected}, got {got}")]
    ScoreCountMismatch {
        /// Number of candidates sent.
        expected: usize,
        /// Number of scores returned.
        got: usize,
    },
}

/// Cross-encoder-style reranker: given a query and a list of
/// candidate texts, return one relevance score per candidate
/// (higher is better).
///
/// The returned `Vec<f32>` MUST be in the SAME order and same length
/// as the input `candidates`; the retriever sorts by score and zips
/// back to node ids. Score range is implementation-defined (Cohere
/// returns logits; Voyage returns [0, 1]; local ONNX depends on the
/// head). Callers who need to mix scores from different rerankers
/// should normalise.
///
/// Implementations handle internal batching if the provider has a
/// per-request cap; the caller passes the full candidate slice.
pub trait Reranker: Send + Sync + Debug {
    /// Provider + model identifier. Lowercase, colon-separated by
    /// convention (e.g. `"cohere:rerank-v3.5"`,
    /// `"local:bge-reranker-v2-m3"`). Used for logging and cache keys.
    fn model(&self) -> &str;

    /// Re-score `candidates` against `query`.
    ///
    /// # Errors
    ///
    /// Any [`RerankError`] the adapter surfaces. The retriever
    /// gracefully falls back to the fused order on error; it does
    /// not propagate the failure to the user.
    fn rerank(&self, query: &str, candidates: &[&str]) -> Result<Vec<f32>, RerankError>;
}

/// Deterministic test-only reranker that scores candidates by their
/// token-overlap Jaccard similarity to the query.
///
/// Useful for Retriever-integration tests where a real cross-encoder
/// is not available; the score is meaningful (shared-tokens ratio)
/// so rerank-changes-top-1 tests can rely on predictable behaviour.
/// It is **not** a substitute for a real cross-encoder on
/// compositional-paraphrase queries - Jaccard is a keyword metric,
/// so "father's sister" will still not match "aunt."
#[derive(Debug, Clone, Default)]
pub struct MockJaccardReranker;

impl Reranker for MockJaccardReranker {
    fn model(&self) -> &str {
        "mock:jaccard"
    }

    fn rerank(&self, query: &str, candidates: &[&str]) -> Result<Vec<f32>, RerankError> {
        let q = token_set(query);
        Ok(candidates
            .iter()
            .map(|c| {
                let c_tokens = token_set(c);
                let inter = q.intersection(&c_tokens).count() as f32;
                let union_ = q.union(&c_tokens).count() as f32;
                if union_ == 0.0 { 0.0 } else { inter / union_ }
            })
            .collect())
    }
}

/// Test-only reranker that always errors. Proves the graceful
/// fallback path in [`crate::retrieve::Retriever::execute`].
#[derive(Debug, Clone, Default)]
pub struct AlwaysFailReranker;

impl Reranker for AlwaysFailReranker {
    fn model(&self) -> &str {
        "mock:always-fail"
    }

    fn rerank(&self, _query: &str, _candidates: &[&str]) -> Result<Vec<f32>, RerankError> {
        Err(RerankError::Network(
            "intentional failure for test".to_string(),
        ))
    }
}

/// Simple ASCII/unicode alphanumeric tokenizer used only by the
/// deterministic mock reranker. Lowercases, splits on non-alphanumeric
/// runs, drops empty tokens. Not exposed publicly; the mock is the
/// only caller.
fn token_set(text: &str) -> std::collections::HashSet<String> {
    let mut out: std::collections::HashSet<String> = std::collections::HashSet::new();
    let mut buf = String::new();
    for ch in text.chars() {
        if ch.is_alphanumeric() {
            for lc in ch.to_lowercase() {
                buf.push(lc);
            }
        } else if !buf.is_empty() {
            out.insert(std::mem::take(&mut buf));
        }
    }
    if !buf.is_empty() {
        out.insert(buf);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mock_jaccard_identical_query_and_candidate_scores_one() {
        let r = MockJaccardReranker;
        let s = r.rerank("alice bob", &["alice bob"]).unwrap();
        assert!((s[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn mock_jaccard_disjoint_scores_zero() {
        let r = MockJaccardReranker;
        let s = r.rerank("alice", &["zed"]).unwrap();
        assert_eq!(s[0], 0.0);
    }

    #[test]
    fn mock_jaccard_partial_overlap() {
        let r = MockJaccardReranker;
        // Query tokens: {alice, bob}. Candidate: {alice, carol}.
        // Inter = 1, Union = 3, score = 1/3.
        let s = r.rerank("alice bob", &["alice carol"]).unwrap();
        assert!((s[0] - (1.0 / 3.0)).abs() < 1e-6);
    }

    #[test]
    fn mock_jaccard_length_matches_candidates_len() {
        let r = MockJaccardReranker;
        let s = r
            .rerank("alpha", &["alpha beta", "alpha gamma", "delta"])
            .unwrap();
        assert_eq!(s.len(), 3);
    }

    #[test]
    fn mock_jaccard_empty_candidates_empty_output() {
        let r = MockJaccardReranker;
        let s = r.rerank("alpha", &[]).unwrap();
        assert!(s.is_empty());
    }

    #[test]
    fn always_fail_reranker_returns_err() {
        let r = AlwaysFailReranker;
        assert!(r.rerank("q", &["c"]).is_err());
    }

    #[test]
    fn model_id_contains_provider_prefix() {
        assert_eq!(MockJaccardReranker.model(), "mock:jaccard");
        assert_eq!(AlwaysFailReranker.model(), "mock:always-fail");
    }
}
