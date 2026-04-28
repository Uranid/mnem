//! Optional typed-relation inference for mnem-extract.
//!
//! Gated behind the `typed-relations` Cargo feature. Default OFF.
//!
//! # What this module defines
//!
//! - [`InferenceBudget`] - shared primitive, wall-clock + volume caps
//!   for the inference stage of one commit.
//! - [`TypedRelation`] - edge payload emitted by
//!   [`crate::traits::Extractor::infer_typed_relations`].
//! - [`InferenceMethod`] - provenance tag attached to every inferred
//!   edge (solution.md R3 §2).
//!
//! # What this module does NOT define
//!
//! - Clustering (Leiden). Lives in a downstream crate.
//! - Canary suite. See `gap-catalog/shared/canary-suite.md`.
//! - CLI wiring. That's `mnem-ingest` + `mnem-cli`.
//!
//! # Floor-c tunable
//!
//! [`InferenceBudget::MAX_INFERENCE_MS_PER_COMMIT`] = 250ms. Half of
//! `max_cooccurrence_ms = 500` (commit-envelope reserve, gap 10).
//! Exposed as gauge `mnem_inference_budget_effective_ms` via
//! [`InferenceBudget::effective_ms_gauge`] and enforced by proptest
//! [`proptests::budget_respected`].

use serde::{Deserialize, Serialize};

use crate::traits::ExtractionSource;

/// Gauge name for the runtime-effective inference budget.
///
/// Emitted by [`InferenceBudget::effective_ms_gauge`]. The three-
/// condition floor-c apparatus (named constant + gauge + proptest)
/// lives in this module's tests.
pub const BUDGET_GAUGE_NAME: &str = "mnem_inference_budget_effective_ms";

/// Shared primitive: wall-clock and volume caps for the inference
/// stage of a single commit.
///
/// All fields are floor-c tunables (solution.md R6 §Constant
/// classification table) or corpus-derived. No magic numbers.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct InferenceBudget {
    /// Rolling wall-clock budget for the extract-and-embed phase of
    /// inference, derived from `3 * rolling.p50_ingest_phrase_embed_ms`
    /// (fallback 500ms). Class `a` (rolling telemetry).
    pub extract_latency_budget_ms: u32,

    /// Hard wall-clock ceiling per commit. Class `c` (reference
    /// standard): 250ms = 50% of `max_cooccurrence_ms = 500`.
    pub max_inference_ms_per_commit: u32,

    /// Hard ceiling on bridging-phrase embeddings per commit. Class
    /// `a` (corpus-derived): `min(50_000, sqrt(N_phrases) * 100)`.
    pub max_phrases_embedded: u32,

    /// Max inferred relation types emitted per commit. Class `a`
    /// (corpus-derived): `ceil(log2(corpus_size))`.
    pub max_types: u32,

    /// Per-author bridging-phrase cap per commit. Class `a`
    /// (corpus-derived): `max(200, 0.01 * mean_phrases_per_author)`.
    pub author_rate_limit_per_commit: u32,
}

impl InferenceBudget {
    /// Floor-c reference standard for the per-commit hard wall.
    ///
    /// 250ms. See `solution.md` R6 §Floor-c apparatus.
    pub const MAX_INFERENCE_MS_PER_COMMIT: u32 = 250;

    /// Fallback for `extract_latency_budget_ms` when rolling p50
    /// telemetry is unavailable. See
    /// `shared/inference-budget.md` §API sketch.
    pub const FALLBACK_EXTRACT_LATENCY_MS: u32 = 500;

    /// Conservative defaults for CI / proptest / initial runs.
    ///
    /// Real deployments derive via a stats-aware constructor in the
    /// ingest crate; this keeps `mnem-extract` free of clock or
    /// telemetry dependencies.
    #[must_use]
    pub const fn conservative() -> Self {
        Self {
            extract_latency_budget_ms: Self::MAX_INFERENCE_MS_PER_COMMIT,
            max_inference_ms_per_commit: Self::MAX_INFERENCE_MS_PER_COMMIT,
            max_phrases_embedded: 10_000,
            max_types: 8,
            author_rate_limit_per_commit: 200,
        }
    }

    /// Effective runtime budget = `min(extract_latency_budget_ms,
    /// max_inference_ms_per_commit)`.
    ///
    /// The hard wall is always the ceiling, even when rolling
    /// telemetry computes a higher extract budget.
    #[must_use]
    pub const fn effective_ms(&self) -> u32 {
        let extract = self.extract_latency_budget_ms;
        let hard = self.max_inference_ms_per_commit;
        if extract < hard { extract } else { hard }
    }

    /// Sample for gauge `mnem_inference_budget_effective_ms`.
    ///
    /// Caller is responsible for emission; this module does not link
    /// a metrics backend. Returning `(name, value)` lets the caller
    /// use either `metrics::gauge!` or a custom registry.
    #[must_use]
    pub fn effective_ms_gauge(&self) -> (&'static str, f64) {
        (BUDGET_GAUGE_NAME, f64::from(self.effective_ms()))
    }

    /// Validate that the budget is internally consistent.
    ///
    /// Returns `Err` with a static reason when:
    ///
    /// 1. `max_inference_ms_per_commit` is zero.
    /// 2. `extract_latency_budget_ms` is zero.
    /// 3. `max_phrases_embedded` is zero.
    ///
    /// Zero-valued caps are always a programming error: they would
    /// make the entire inference pass a no-op and silently hide bugs.
    pub fn validate(&self) -> Result<(), &'static str> {
        if self.max_inference_ms_per_commit == 0 {
            return Err("max_inference_ms_per_commit must be > 0");
        }
        if self.extract_latency_budget_ms == 0 {
            return Err("extract_latency_budget_ms must be > 0");
        }
        if self.max_phrases_embedded == 0 {
            return Err("max_phrases_embedded must be > 0");
        }
        Ok(())
    }
}

impl Default for InferenceBudget {
    fn default() -> Self {
        Self::conservative()
    }
}

/// Provenance tag describing *how* a typed relation was inferred.
///
/// Serialised into `TypedRelation::source_label` as
/// `"inferred:<method>"` per solution.md R3: every edge carries its
/// inference method so rollback and audit can filter by origin.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InferenceMethod {
    /// KeyBERT-style pattern-embedding clustering (Leiden).
    PatternEmbedding,
    /// Co-occurrence PMI promoted to a named type via clustering.
    CooccurrencePmi,
    /// Caller-supplied custom method. String is forensics-tagged in
    /// the provenance label verbatim; keep it short and snake_case.
    Custom(String),
}

impl InferenceMethod {
    /// Render as the `inferred:<method>` tag used in provenance
    /// labels and the `mnem commit` audit stream.
    #[must_use]
    pub fn provenance_label(&self) -> String {
        match self {
            Self::PatternEmbedding => "inferred:pattern_embedding".to_string(),
            Self::CooccurrencePmi => "inferred:cooccurrence_pmi".to_string(),
            Self::Custom(s) => format!("inferred:{s}"),
        }
    }
}

/// An inferred typed edge between two previously-extracted entities.
///
/// Distinct from [`crate::traits::Relation`]:
///
/// - [`Relation`](crate::traits::Relation) is a raw `(src, dst,
///   weight)` triple with no named predicate.
/// - `TypedRelation` has a clustering-assigned predicate and a
///   confidence in `[0.0, 1.0]` so the trust gate can filter it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TypedRelation {
    /// Subject entity mention.
    pub src: String,
    /// Object entity mention.
    pub dst: String,
    /// Clustering-assigned predicate label (e.g. `"causes"`).
    pub predicate: String,
    /// Confidence in `[0.0, 1.0]`. Fed to
    /// [`crate::trust::TrustBoundary::admit`] downstream.
    pub confidence: f32,
    /// Provenance in the full taxonomy ([`ExtractionSource`]).
    pub source: ExtractionSource,
    /// Short human-readable provenance label, always of the shape
    /// `"inferred:<method>"`. Derived from [`InferenceMethod`] at
    /// emission time to keep the struct `serde`-flat.
    pub source_label: String,
}

impl TypedRelation {
    /// Build a new typed relation with the provenance label
    /// auto-derived from `method`.
    #[must_use]
    pub fn new(
        src: impl Into<String>,
        dst: impl Into<String>,
        predicate: impl Into<String>,
        confidence: f32,
        method: &InferenceMethod,
    ) -> Self {
        Self {
            src: src.into(),
            dst: dst.into(),
            predicate: predicate.into(),
            confidence,
            source: ExtractionSource::Statistical,
            source_label: method.provenance_label(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn conservative_budget_passes_validation() {
        let b = InferenceBudget::conservative();
        assert!(b.validate().is_ok());
    }

    #[test]
    fn effective_ms_is_minimum_of_extract_and_hard_wall() {
        let mut b = InferenceBudget::conservative();
        b.extract_latency_budget_ms = 100;
        assert_eq!(b.effective_ms(), 100);
        b.extract_latency_budget_ms = 1_000;
        assert_eq!(
            b.effective_ms(),
            InferenceBudget::MAX_INFERENCE_MS_PER_COMMIT
        );
    }

    #[test]
    fn hard_wall_matches_spec_pinned_value() {
        // solution.md R6 §Floor-c apparatus: 250ms.
        assert_eq!(InferenceBudget::MAX_INFERENCE_MS_PER_COMMIT, 250);
    }

    #[test]
    fn gauge_emits_stable_name_and_effective_value() {
        let b = InferenceBudget::conservative();
        let (name, val) = b.effective_ms_gauge();
        assert_eq!(name, "mnem_inference_budget_effective_ms");
        assert!((val - f64::from(b.effective_ms())).abs() < f64::EPSILON);
    }

    #[test]
    fn validate_rejects_zero_caps() {
        let mut b = InferenceBudget::conservative();
        b.max_inference_ms_per_commit = 0;
        assert!(b.validate().is_err());
        let mut b = InferenceBudget::conservative();
        b.extract_latency_budget_ms = 0;
        assert!(b.validate().is_err());
        let mut b = InferenceBudget::conservative();
        b.max_phrases_embedded = 0;
        assert!(b.validate().is_err());
    }

    #[test]
    fn inference_method_renders_provenance_label() {
        assert_eq!(
            InferenceMethod::PatternEmbedding.provenance_label(),
            "inferred:pattern_embedding",
        );
        assert_eq!(
            InferenceMethod::CooccurrencePmi.provenance_label(),
            "inferred:cooccurrence_pmi",
        );
        assert_eq!(
            InferenceMethod::Custom("my_method".into()).provenance_label(),
            "inferred:my_method",
        );
    }

    #[test]
    fn typed_relation_auto_tags_provenance_label() {
        let r = TypedRelation::new(
            "alice",
            "bob",
            "knows",
            0.9,
            &InferenceMethod::PatternEmbedding,
        );
        assert_eq!(r.source_label, "inferred:pattern_embedding");
        assert_eq!(r.source, ExtractionSource::Statistical);
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        /// Floor-c proptest for `mnem_inference_budget_effective_ms`.
        ///
        /// Over arbitrary `(extract_ms, hard_ms)`, the effective
        /// budget is always `min(extract_ms, hard_ms)` and never
        /// exceeds the hard wall. The gauge reports exactly this.
        #[test]
        fn budget_respected(
            extract_ms in 1u32..10_000,
            hard_ms in 1u32..10_000,
            max_phrases in 1u32..100_000,
            max_types in 1u32..64,
            author_cap in 1u32..10_000,
        ) {
            let b = InferenceBudget {
                extract_latency_budget_ms: extract_ms,
                max_inference_ms_per_commit: hard_ms,
                max_phrases_embedded: max_phrases,
                max_types,
                author_rate_limit_per_commit: author_cap,
            };
            prop_assert!(b.validate().is_ok());
            let eff = b.effective_ms();
            prop_assert!(eff <= extract_ms);
            prop_assert!(eff <= hard_ms);
            prop_assert!(eff == extract_ms.min(hard_ms));
            let (_, val) = b.effective_ms_gauge();
            prop_assert!((val - f64::from(eff)).abs() < f64::EPSILON);
        }

        /// Under the conservative default, the effective budget
        /// equals the spec-pinned 250ms floor.
        #[test]
        fn conservative_default_matches_hard_wall(_n in 0u32..8) {
            let b = InferenceBudget::conservative();
            prop_assert_eq!(
                b.effective_ms(),
                InferenceBudget::MAX_INFERENCE_MS_PER_COMMIT,
            );
        }
    }
}
