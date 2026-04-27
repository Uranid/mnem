//! # mnem-extract
//!
//! Statistical, embedding-based entity + relation extraction for mnem.
//!
//! This crate is the default path for experiment E3 of the GraphRAG
//! research track: it replaces LLM-driven NER with a KeyBERT-style
//! candidate-scoring pass over the chunk embedding that mnem-ingest
//! already computes. That keeps extraction deterministic, fully
//! offline, and cost-free at ingest time.
//!
//! ## Scope
//!
//! - [`traits::Extractor`] - pluggable extractor surface. One default
//!   implementation ([`keybert::KeyBertExtractor`]) ships with the
//!   crate; callers can swap in authored or LLM-backed extractors by
//!   implementing the trait themselves.
//! - [`keybert::KeyBertExtractor`] - KeyBERT-style n-gram ranking
//!   against a supplied chunk embedding, with MMR (Maximal Marginal
//!   Relevance) diversification and deterministic tiebreaks.
//! - [`cooccurrence::mine_relations`] - PMI-weighted co-occurrence
//!   relation miner that emits one [`traits::Relation`] per sentence-
//!   local entity pair whose pointwise mutual information exceeds a
//!   configurable threshold.
//!
//! ## Determinism
//!
//! Every public extractor in this crate is deterministic: same input
//! text + same embedder → byte-identical [`traits::Entity`] and
//! [`traits::Relation`] streams across runs. The proptest suite under
//! `tests/proptest_determinism.rs` enforces this as a first-class
//! property.
//!
//! ## Non-goals
//!
//! - No LLM calls. No network. No tokio.
//! - No training, no fine-tuning: the extractor consumes whatever
//!   [`mnem_embed_providers::Embedder`] the caller already configured.
//! - No HTTP / MCP / CLI wiring lives in this crate; `mnem-ingest`
//!   exposes the integration and `mnem-cli` surfaces the flag.

#![deny(missing_docs)]
#![forbid(unsafe_code)]

pub mod cooccurrence;
pub mod keybert;
pub mod traits;

/// Optional typed-relation inference (gap 03). Gated behind the
/// `typed-relations` Cargo feature. Default OFF per solution.md R3.
#[cfg(feature = "typed-relations")]
pub mod inference;

/// Adversarial trust-boundary gate for opt-in typed-relation
/// inference (gap 03). Gated behind the `typed-relations` Cargo
/// feature. Default OFF.
#[cfg(feature = "typed-relations")]
pub mod trust;

pub use cooccurrence::{CoOccurrenceMiner, mine_relations};
pub use keybert::KeyBertExtractor;
pub use traits::{Entity, ExtractionSource, Extractor, Relation};

#[cfg(feature = "typed-relations")]
pub use inference::{InferenceBudget, InferenceMethod, TypedRelation};
#[cfg(feature = "typed-relations")]
pub use trust::{
    AuthorFingerprint, AuthorRateLimiter, Candidate, PPR_AMPLIFICATION_FLOOR, TrustBoundary,
};
