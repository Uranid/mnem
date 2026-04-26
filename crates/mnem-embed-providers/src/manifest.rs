//! Embedder manifest: self-describing metadata each provider publishes
//! so downstream code never has to guess a semantic-similarity floor.
//!
//! # Motivation (Gap 15: ingest-no-edges)
//!
//! Before this module, `mnem-ingest` and `mnem-graphrag` hard-coded a
//! single magic similarity floor (`0.25`) for every embedder. The
//! value was empirically calibrated against one model (MiniLM-L6-v2);
//! swapping in a different embedder (BGE-M3, OpenAI, an ONNX variant)
//! silently kept using the wrong floor. The symptom was silent
//! ingest: every co-occurrence score sat below the floor, every
//! candidate edge was discarded, and the node graph landed on disk
//! with no edges at all.
//!
//! The fix is to let each embedder **declare** its measured noise
//! floor. `EmbedderManifest::noise_floor` is the similarity below
//! which two embeddings of unrelated texts typically land; callers
//! use it as a gate on "is this pair co-occurring meaningfully?"
//! without any global constant.
//!
//! # Values
//!
//! Per-provider values come from empirical measurement (see
//! `research/gap-catalog/15-ingest-no-edges/solution.md`). They are
//! stable per `(provider, model)` tuple and therefore live next to the
//! provider's dim + model id on the same manifest struct.

/// Self-describing metadata for an [`crate::Embedder`].
///
/// Every provider publishes one of these via
/// [`crate::Embedder::manifest`]. The three fields together are
/// everything downstream ingest / retrieve code needs to reason about
/// an embedder without knowing which concrete provider produced it.
///
/// # Invariants
///
/// - `model_id` is the same string returned by [`crate::Embedder::model`].
/// - `dim` is the same value returned by [`crate::Embedder::dim`].
/// - `noise_floor` is a finite, non-negative `f32` in `[0.0, 1.0]`.
///   `0.0` means "this embedder has no measurable noise floor"
///   (used by [`crate::MockEmbedder`] in tests).
#[derive(Debug, Clone, PartialEq)]
pub struct EmbedderManifest {
    /// Fully-qualified model id, e.g. `"openai:text-embedding-3-small"`,
    /// `"ollama:bge-m3"`, `"onnx:all-MiniLM-L6-v2"`.
    pub model_id: String,
    /// Vector dimension of every embedding this provider produces.
    pub dim: u32,
    /// Empirically-measured cosine-similarity floor: the similarity
    /// between embeddings of unrelated texts under this model.
    /// Callers use this as the gate for co-occurrence edge creation
    /// during ingest.
    pub noise_floor: f32,
}

impl EmbedderManifest {
    /// Construct a manifest.
    ///
    /// # Panics
    ///
    /// Panics if `noise_floor` is NaN, infinite, negative, or > 1.0.
    /// The panic is intentional: a provider that reports a malformed
    /// noise floor is a programming error, not a runtime condition.
    #[must_use]
    pub fn new(model_id: impl Into<String>, dim: u32, noise_floor: f32) -> Self {
        assert!(
            noise_floor.is_finite() && (0.0..=1.0).contains(&noise_floor),
            "noise_floor must be a finite f32 in [0.0, 1.0]; got {noise_floor}"
        );
        Self {
            model_id: model_id.into(),
            dim,
            noise_floor,
        }
    }
}

// ---------------------------------------------------------------------------
// Runtime-derivation helpers
// ---------------------------------------------------------------------------
//
// These helpers replace a second class of magic numbers: the
// per-node time budgets (`max_cooccurrence_ms`, `max_knn_ingest_ms`)
// that Gap 15's ingest loop used to hard-code. The derivation rules
// are lifted from `solution.md` verbatim so any change to the budget
// is one edit in one file.

/// Default global latency budget per node, in milliseconds.
///
/// Derivation: 4% of a 5-second human interaction budget, rounded to
/// the nearest 10 ms. Exposed as a constant so callers that want to
/// fall back to the default can do so by passing `None` to
/// [`derive_max_cooccurrence_ms`].
pub const DEFAULT_LATENCY_BUDGET_MS: u32 = 200;

/// Derive the per-node co-occurrence compute budget from a global
/// latency budget.
///
/// Rule: co-occurrence gets 50% of the node budget, floored at 20 ms
/// so the ingest loop never starves on a misconfigured budget.
///
/// Passing `None` uses [`DEFAULT_LATENCY_BUDGET_MS`].
#[must_use]
pub fn derive_max_cooccurrence_ms(latency_budget: Option<u32>) -> u32 {
    let budget = latency_budget.unwrap_or(DEFAULT_LATENCY_BUDGET_MS);
    (budget / 2).max(20)
}

/// Derive the per-node kNN ingest compute budget given the
/// co-occurrence budget and the ingest batch size.
///
/// Rule: kNN per-node budget is the co-occurrence budget divided by
/// the batch size, floored at 5 ms. Wider batches amortise the kNN
/// index walk across more nodes, so the per-node slice shrinks.
///
/// A `batch` of 0 is coerced to 1 so the helper never divides by zero.
#[must_use]
pub fn derive_max_knn_ingest_per_node_ms(coocc_ms: u32, batch: u32) -> u32 {
    let divisor = batch.max(1);
    (coocc_ms / divisor).max(5)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn manifest_round_trip() {
        let m = EmbedderManifest::new("mock:test", 16, 0.0);
        assert_eq!(m.model_id, "mock:test");
        assert_eq!(m.dim, 16);
        assert!((m.noise_floor - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    #[should_panic(expected = "noise_floor")]
    fn manifest_rejects_nan() {
        let _ = EmbedderManifest::new("bad:model", 16, f32::NAN);
    }

    #[test]
    #[should_panic(expected = "noise_floor")]
    fn manifest_rejects_negative() {
        let _ = EmbedderManifest::new("bad:model", 16, -0.01);
    }

    #[test]
    #[should_panic(expected = "noise_floor")]
    fn manifest_rejects_above_one() {
        let _ = EmbedderManifest::new("bad:model", 16, 1.01);
    }

    #[test]
    fn derive_coocc_uses_default_when_none() {
        assert_eq!(
            derive_max_cooccurrence_ms(None),
            DEFAULT_LATENCY_BUDGET_MS / 2
        );
    }

    #[test]
    fn derive_coocc_is_half_of_budget() {
        assert_eq!(derive_max_cooccurrence_ms(Some(400)), 200);
    }

    #[test]
    fn derive_coocc_has_floor() {
        // A 10 ms budget would otherwise give 5 ms; the 20 ms floor
        // keeps the loop from starving.
        assert_eq!(derive_max_cooccurrence_ms(Some(10)), 20);
    }

    #[test]
    fn derive_knn_divides_by_batch() {
        assert_eq!(derive_max_knn_ingest_per_node_ms(100, 10), 10);
    }

    #[test]
    fn derive_knn_has_floor() {
        assert_eq!(derive_max_knn_ingest_per_node_ms(10, 1000), 5);
    }

    #[test]
    fn derive_knn_handles_zero_batch() {
        // batch=0 must not panic; it is coerced to 1.
        assert_eq!(derive_max_knn_ingest_per_node_ms(100, 0), 100);
    }
}
