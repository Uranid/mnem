//! Public traits and value types for mnem-extract.
//!
//! The [`Extractor`] trait is the single integration point between an
//! ingest pipeline and any statistical / LLM-backed extractor. The
//! default implementation is [`crate::keybert::KeyBertExtractor`].

use serde::{Deserialize, Serialize};

/// An entity mention located in a chunk of source text.
///
/// Fields are deliberately flat and serde-round-trippable so that an
/// ingest pipeline can attach the list verbatim to a Node's `props`
/// bag or persist it as an audit artefact without an intermediate DTO.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Entity {
    /// Surface form of the mention, exactly as it appears in the
    /// source text (whitespace-normalised but otherwise untouched).
    pub mention: String,
    /// Extractor-assigned score in `[0.0, 1.0]`. For the KeyBERT
    /// extractor this is the cosine similarity between the candidate
    /// embedding and the chunk embedding after MMR diversification.
    pub score: f32,
    /// Byte span `(start, end)` of the mention in the original chunk
    /// text. `end` is exclusive, matching `str::get(start..end)`.
    pub span: (usize, usize),
}

/// A candidate relation between two previously-extracted entities.
///
/// The payload stays flat: a statistical miner emits raw
/// `(subject_mention, object_mention, weight)` triples without
/// predicate names. Callers that want to name the edge type map
/// [`ExtractionSource`] → edge label at ingest time.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Relation {
    /// Subject entity mention.
    pub src: String,
    /// Object entity mention.
    pub dst: String,
    /// Extractor-assigned weight. For PMI-based mining this is the
    /// pointwise mutual information in natural-log units.
    pub weight: f32,
    /// Provenance of the triple.
    pub source: ExtractionSource,
}

/// How an [`Entity`] or [`Relation`] was produced.
///
/// The enum is `#[non_exhaustive]` so downstream crates can add new
/// variants (e.g. a gazetteer source) without a semver break in the
/// consumers that only match on the KeyBERT + authored cases.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExtractionSource {
    /// Hand-authored by the caller (e.g. entities from a front-matter
    /// YAML block or an explicit CLI flag). Always trusted.
    Authored,
    /// Produced by a statistical extractor in this crate (KeyBERT,
    /// co-occurrence PMI).
    Statistical,
    /// Produced by an LLM-backed extractor; the inner string is the
    /// fully-qualified model identifier so provenance survives
    /// round-tripping.
    LlmModel(String),
}

/// Pluggable statistical entity + relation extractor.
///
/// Implementations MUST be `Send + Sync` so `mnem-ingest` can hand them
/// across thread boundaries when a future batch driver parallelises
/// ingest. They SHOULD be deterministic: the proptest harness under
/// `tests/proptest_determinism.rs` enforces byte-identical output for
/// the in-crate default.
pub trait Extractor: Send + Sync {
    /// Extract entity mentions from `text`.
    ///
    /// `chunk_embed` is the embedding of the enclosing chunk, produced
    /// by the same [`mnem_embed_providers::Embedder`] the extractor
    /// will use for candidates. Its length MUST match the embedder's
    /// `dim()`; mismatches are a programming error and extractors may
    /// return an empty vec in that case rather than panic.
    fn extract_entities(&self, text: &str, chunk_embed: &[f32]) -> Vec<Entity>;

    /// Mine candidate relations over an already-extracted entity set.
    ///
    /// The default implementation returns an empty vec, letting
    /// callers opt into relation mining explicitly via the
    /// [`crate::cooccurrence`] module.
    fn extract_relations(&self, _text: &str, _entities: &[Entity]) -> Vec<Relation> {
        Vec::new()
    }

    /// Optionally infer *typed* relations between already-extracted
    /// entities, subject to the supplied [`InferenceBudget`].
    ///
    /// Gated behind the `typed-relations` Cargo feature. Default OFF
    /// per solution.md R3: no extractor emits typed relations unless
    /// the caller explicitly opts in at ingest time.
    ///
    /// The default implementation returns an empty vec - safe for
    /// every existing extractor, no behaviour change on the default
    /// build. Implementors must enforce:
    ///
    /// 1. Every emitted [`TypedRelation`] carries
    ///    `source_label = "inferred:<method>"` (auto-derived by
    ///    [`TypedRelation::new`]).
    /// 2. Wall-clock work does not exceed `budget.effective_ms()`.
    /// 3. Emitted edge count does not exceed `budget.max_types`.
    ///
    /// Downstream consumers (PPR, multihop) MUST gate admission with
    /// [`crate::trust::TrustBoundary::admit`] before using any edge.
    #[cfg(feature = "typed-relations")]
    fn infer_typed_relations(
        &self,
        _text: &str,
        _entities: &[Entity],
        _budget: &crate::inference::InferenceBudget,
    ) -> Vec<crate::inference::TypedRelation> {
        Vec::new()
    }
}
