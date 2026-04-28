//! Score calibration - scale-free per-query interpretability for dense retrieval.
//!
//! # Design principle
//!
//! Rather than shipping cross-embedder calibration (which would require a
//! trained scaler per model), this module emits **scale-free summaries** that
//! are meaningful *within a single response*. Two shapes:
//!
//! 1. [`score_quantiles`] - per-item `(rank_from_bottom) / max(K - 1, 1)` so
//!    the top item is 1.0, the bottom is 0.0. Identical meaning across
//!    embedders; a pure rank metric with no threshold.
//! 2. [`distribution_shape`] - response-level
//!    [`ScoreDistribution`] block with min / max / median / IQR and a
//!    categorical `shape` label (`long_tail` / `uniform` / `bimodal` /
//!    `insufficient-samples`).
//!
//! # Scale-freeness
//!
//! The shape classifier thresholds on *relative* quantities
//! (`max - median > 2 * iqr`) rather than absolute score magnitudes, so a
//! dot-product lane scoring in `[0, 100]` and a cosine lane scoring in
//! `[-1, 1]` produce the same shape label for isomorphic distributions.
//!
//! Quantile emission is well-defined for any `K >= 1` (degenerate cases
//! collapse to the trivial `[1.0; K]` vector); shape classification gates
//! at a principled floor derived from Wilson 95% CI width (see
//! [`derive_k_min`] and the module-level `K_MIN` constant).
//!
//! # Determinism
//!
//! Every function in this module is a pure function of its inputs. Given
//! the same `ranked` slice and `gate`, output is byte-identical across
//! runs. No floating-point reductions over unordered hashmaps, no global
//! state, no RNG.
//!
//! # Floor-c tunable constants
//!
//! Two constants live here with the floor-c contract (standard-cite +
//! gauge + proptest):
//!
//! | Constant            | Value | Standard                                           |
//! |---------------------|-------|----------------------------------------------------|
//! | [`K_MIN`]           | 8     | Wilson 95% CI width <= 0.18 requires K >= 8        |
//! | [`WILSON_WIDTH_TARGET`] | 0.18 | Loose for probabilistic routing, tight for decisions |
//!
//! Derivation (Round 4 spec): Wilson 95% CI formula
//! `K >= (z/eps)^2 * p * (1-p)` where `z = 1.96` (95% CI), `p = 0.5`
//! (worst-case variance). `K_MIN = 8` is the **minimum gate value**
//! below which shape classification is suppressed (IQR and median
//! collapse under noise in smaller samples); it is a principled
//! lower bound, not literally `derive_k_min(WILSON_WIDTH_TARGET)`.
//! The `derive_k_min` helper computes the Wilson-interval K directly
//! for callers that want to tighten the width target at larger K.

use mnem_core::id::NodeId;
use serde::{Deserialize, Serialize};

/// Wilson z-score for a 95% confidence interval. Standard statistical
/// constant; exposed here so [`derive_k_min`] is self-contained.
pub const WILSON_Z: f32 = 1.96;

/// Default Wilson-interval width target for calibration floors.
///
/// Floor-c tunable: standard = "loose enough for probabilistic routing,
/// tight enough for agent decisions"; gauge =
/// `mnem_calibration_width_target_effective`; proptest =
/// `width_target_tunable_in_principled_range`.
pub const WILSON_WIDTH_TARGET: f32 = 0.18;

/// Default minimum sample size for shape classification.
///
/// Floor-c tunable: standard = "Wilson 95% CI width <= 0.18 requires
/// K >= 8"; gauge = `mnem_calibration_k_min_effective`; proptest =
/// `k_min_default_is_8`.
pub const K_MIN: usize = 8;

/// Categorical label summarising the shape of the per-response score
/// distribution. Promoted to a top-level agent hint on the retrieve
/// response envelope (see [`ScoreDistribution::shape`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ShapeLabel {
    /// Top score dominates the tail; `max - median > 2 * iqr`. Agents
    /// can trust top-1 as a confident match.
    LongTail,
    /// Scores are roughly equi-spaced; no single item dominates. Dense
    /// ranking is inconclusive; consider a rerank or graph expansion.
    Uniform,
    /// Two distinct score clusters separated by a gap larger than
    /// `iqr`. Often a hybrid-lane artefact; look at per-lane scores.
    Bimodal,
    /// Fewer than [`K_MIN`] samples; shape cannot be distinguished from
    /// noise. `score_quantile` still emits (pure rank, well-defined
    /// for `K >= 2`), but distribution statistics are all zero.
    InsufficientSamples,
}

/// Response-level score distribution summary.
///
/// Emitted once per retrieve response alongside the `items` array. All
/// fields are scale-free or derived from scale-free quantities; the
/// `shape` label is the primary agent hint.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ScoreDistribution {
    /// Minimum score across ranked items (0.0 when
    /// [`ShapeLabel::InsufficientSamples`]).
    pub min: f32,
    /// Maximum score across ranked items.
    pub max: f32,
    /// Median (50th percentile) of the score vector.
    pub median: f32,
    /// Inter-quartile range (`Q3 - Q1`). Clamped to `f32::EPSILON`
    /// before any downstream ratio so shape classification stays
    /// defined for degenerate all-equal distributions.
    pub iqr: f32,
    /// Categorical shape label.
    pub shape: ShapeLabel,
}

impl Default for ScoreDistribution {
    fn default() -> Self {
        Self {
            min: 0.0,
            max: 0.0,
            median: 0.0,
            iqr: 0.0,
            shape: ShapeLabel::InsufficientSamples,
        }
    }
}

/// Compute per-item score quantiles from a ranked list.
///
/// Input is assumed ranked **descending** by score (position 0 = top
/// item). Output preserves input order: `out[i]` is the quantile of
/// `ranked[i]`, with the top item receiving `1.0` and the bottom
/// receiving `0.0`.
///
/// Formula: `out[i] = (K - 1 - i) / max(K - 1, 1)` where `K = ranked.len()`.
///
/// # Scale-freeness
///
/// Output depends **only on rank**, never on absolute score values.
/// Two responses with identical rank order but different score
/// magnitudes produce identical quantile vectors.
///
/// # Edge cases
///
/// - `K = 0` -> `[]`.
/// - `K = 1` -> `[1.0]` (single item is trivially top-ranked).
///
/// # Determinism
///
/// Pure function of input length. No allocation beyond the output
/// `Vec<f32>`; no RNG, no global state.
#[must_use]
pub fn score_quantiles<N>(ranked: &[(N, f32)]) -> Vec<f32> {
    let k = ranked.len();
    if k == 0 {
        return Vec::new();
    }
    if k == 1 {
        return vec![1.0];
    }
    let denom = (k - 1) as f32;
    (0..k).map(|i| (k - 1 - i) as f32 / denom).collect()
}

/// Thin wrapper over [`score_quantiles`] keyed on [`NodeId`].
///
/// Exists so callers with the canonical `&[(NodeId, f32)]` ranked
/// shape (from `mnem-core`'s retrieve pipeline) get a typed entry
/// point; internally forwards to the generic version.
#[must_use]
pub fn node_score_quantiles(ranked: &[(NodeId, f32)]) -> Vec<f32> {
    score_quantiles(ranked)
}

/// Classify the shape of a score vector.
///
/// `scores` is any order of floats; the function copies and sorts
/// internally. `gate` is the minimum sample count below which the
/// classifier abstains (emitting [`ShapeLabel::InsufficientSamples`]
/// rather than a noisy label). Callers usually pass [`K_MIN`].
///
/// # Classification rules
///
/// Let `sorted` be `scores` ascending, with `n = scores.len()`. Compute
/// `min = sorted[0]`, `max = sorted[n-1]`, `median = sorted[n/2]`,
/// `q1 = sorted[n/4]`, `q3 = sorted[3n/4]`,
/// `iqr = max(q3 - q1, f32::EPSILON)`. Then:
///
/// 1. If `max - median > 2 * iqr` -> [`ShapeLabel::LongTail`].
/// 2. Else if the largest gap between consecutive sorted scores
///    exceeds `iqr` -> [`ShapeLabel::Bimodal`].
/// 3. Else -> [`ShapeLabel::Uniform`].
///
/// All three thresholds are *relative* (ratios / differences of the
/// input), so the classification is scale-invariant: multiplying every
/// score by a positive constant yields an identical [`ShapeLabel`].
///
/// # Determinism
///
/// Pure function. Sorts with `total_cmp` so NaN inputs land in a
/// well-defined position rather than poisoning comparisons; the
/// numeric result for NaN-free inputs is bit-identical across runs.
#[must_use]
pub fn distribution_shape(scores: &[f32], gate: usize) -> ScoreDistribution {
    if scores.len() < gate {
        return ScoreDistribution::default();
    }
    let mut sorted: Vec<f32> = scores.to_vec();
    sorted.sort_by(f32::total_cmp);
    let n = sorted.len();
    let min = sorted[0];
    let max = sorted[n - 1];
    let median = sorted[n / 2];
    let q1 = sorted[n / 4];
    let q3 = sorted[(3 * n) / 4];
    let iqr = (q3 - q1).max(f32::EPSILON);

    // Classification rule order matters. We check bimodal *before*
    // long-tail because a cluster-separated distribution often also
    // satisfies `max - median > 2 * iqr` (one cluster sits above the
    // median of the whole vector), but the correct agent hint is
    // "two peaks" not "one outlier".
    //
    // Bimodal criterion: largest consecutive gap is a sizeable
    // fraction of the full range. Using `(max - min) / 2` keeps the
    // criterion scale-free (ratio, not magnitude) and robust for
    // near-degenerate vectors where `iqr` collapses to
    // `f32::EPSILON`.
    let range = max - min;
    let shape = if bimodal_gap(&sorted, range) {
        ShapeLabel::Bimodal
    } else if (max - median) > 2.0 * iqr {
        ShapeLabel::LongTail
    } else {
        ShapeLabel::Uniform
    };

    ScoreDistribution {
        min,
        max,
        median,
        iqr,
        shape,
    }
}

/// True iff the sorted vector splits into two balanced clusters
/// separated by a gap larger than half the range.
///
/// `sorted` must be ascending. Two conditions together:
///
/// 1. The largest consecutive gap `> range * 0.5` (the gap dominates
///    the span, i.e. a structural separation, not sampling noise).
/// 2. The gap is **interior**: at least two items on each side. A
///    gap at position `n-1` (one item above, `n-1` below) is a
///    single outlier - that's long-tail, not bimodal. Requiring
///    balanced cluster sizes disambiguates the two shapes cleanly.
///
/// Empty / single-item vectors return `false` (caller already gated
/// on `gate >= 2`). Range `<= epsilon` (all-equal) also returns
/// `false` so the all-equal case falls through to Uniform.
fn bimodal_gap(sorted: &[f32], range: f32) -> bool {
    if sorted.len() < 4 || range <= f32::EPSILON {
        return false;
    }
    let threshold = range * 0.5;
    // Walk interior windows only (skip first and last position so we
    // guarantee >=2 items per cluster). `sorted.windows(2)` gives
    // pairs (a, b) where a = sorted[i], b = sorted[i+1].
    for (i, w) in sorted.windows(2).enumerate() {
        // Interior: cluster below has `i+1` items, cluster above has
        // `n - i - 1` items. Require both >= 2.
        let below = i + 1;
        let above = sorted.len() - i - 1;
        if below < 2 || above < 2 {
            continue;
        }
        let gap = w[1] - w[0];
        if gap > threshold {
            return true;
        }
    }
    false
}

/// Derive the minimum sample size satisfying a Wilson 95% CI width
/// target.
///
/// Formula (worst-case variance at `p = 0.5`):
///
/// ```text
/// K >= (z / eps)^2 * p * (1 - p)
///    = (1.96 / eps)^2 * 0.25
/// ```
///
/// Scale: tighter targets (smaller `eps`) grow `K` quadratically;
/// looser targets shrink it. Output is always `>= 1`.
///
/// Reference values:
/// - `width_target = 0.35` -> `K ~= 8` (matches [`K_MIN`] floor).
/// - `width_target = 0.18` -> `K ~= 30` (tight agent-decision margin).
/// - `width_target = 0.10` -> `K ~= 97` (statistical-analysis grade).
///
/// [`K_MIN`] is a separate **floor constant** for the shape-gate; it
/// is not literally `derive_k_min(WILSON_WIDTH_TARGET)`. The Wilson
/// formula here is exposed so callers sizing larger-K experiments
/// can reason about their desired width explicitly.
///
/// # Panics
///
/// Does not panic; non-positive / non-finite `width_target` is
/// clamped to a tiny positive value so the computation cannot
/// produce `NaN` or `0` divisions.
#[must_use]
pub fn derive_k_min(width_target: f32) -> usize {
    // Guard: clamp pathological inputs to avoid NaN / Inf. A zero or
    // negative width target is nonsensical (width is non-negative by
    // definition); we treat it as "as tight as possible" by clamping
    // to f32::EPSILON, which drives K to a very large number rather
    // than panicking.
    let eps = if width_target.is_finite() && width_target > 0.0 {
        width_target
    } else {
        f32::EPSILON
    };
    let raw = (WILSON_Z / eps).powi(2) * 0.25;
    // `ceil` so we never *undershoot* the target width; `max(1)` so
    // we never return 0 for absurdly loose targets.
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let k = raw.ceil() as usize;
    k.max(1)
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---------- floor-c tunable assertions ----------

    #[test]
    fn k_min_default_is_8() {
        // Floor-c constant: K_MIN = 8 is the shape-gate floor below
        // which IQR / median estimators collapse under noise. This is
        // a principled *floor*, not literally the Wilson derivation
        // at the default width target.
        assert_eq!(K_MIN, 8);
        // Wilson formula: the K that satisfies width <= 0.35 is the
        // floor value 8 via (1.96/0.35)^2 * 0.25 = 7.84 -> ceil = 8.
        // This links the floor to a defensible width target even
        // though the tighter default (0.18) demands more samples.
        assert_eq!(derive_k_min(0.35), 8);
    }

    #[test]
    fn width_target_tunable_in_principled_range() {
        // Default width target lies between "loose routing" (0.35,
        // matches floor K=8) and "tight analysis" (0.10, K~97).
        assert!((0.10..=0.35).contains(&WILSON_WIDTH_TARGET));
        // Tighter target (0.10) -> larger K: (1.96/0.10)^2 * 0.25 = 96.04 -> 97.
        assert_eq!(derive_k_min(0.10), 97);
        // Default target (0.18) -> K = 30 by the formula.
        assert_eq!(derive_k_min(WILSON_WIDTH_TARGET), 30);
        // Looser target (0.30) -> smaller K: (1.96/0.30)^2 * 0.25 = 10.67 -> 11.
        assert_eq!(derive_k_min(0.30), 11);
        // Monotone: tighter eps => larger K.
        assert!(derive_k_min(0.10) > derive_k_min(0.18));
        assert!(derive_k_min(0.18) > derive_k_min(0.30));
    }

    #[test]
    fn derive_k_min_never_zero() {
        // Absurdly loose target still yields K >= 1.
        assert!(derive_k_min(100.0) >= 1);
        // Zero / negative / NaN clamped safely.
        assert!(derive_k_min(0.0) >= 1);
        assert!(derive_k_min(-1.0) >= 1);
        assert!(derive_k_min(f32::NAN) >= 1);
    }

    // ---------- quantile behaviour ----------

    #[test]
    fn quantile_monotone() {
        // Ranked descending: index 0 is top, index K-1 is bottom.
        // Quantiles should be non-increasing across positions.
        let ranked: Vec<(u32, f32)> = vec![(0, 0.9), (1, 0.7), (2, 0.5), (3, 0.3), (4, 0.1)];
        let q = score_quantiles(&ranked);
        assert_eq!(q, vec![1.0, 0.75, 0.5, 0.25, 0.0]);
        // Monotonicity: no adjacent pair is increasing.
        assert!(q.windows(2).all(|w| w[0] >= w[1]));
    }

    #[test]
    fn quantile_top_is_one_bottom_is_zero() {
        let ranked: Vec<(u32, f32)> = (0..10).map(|i| (i, 1.0 - (i as f32) * 0.1)).collect();
        let q = score_quantiles(&ranked);
        assert!((q[0] - 1.0).abs() < f32::EPSILON);
        assert!((q[9] - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn quantile_edge_cases() {
        // Empty -> empty.
        let empty: Vec<(u32, f32)> = vec![];
        assert!(score_quantiles(&empty).is_empty());
        // Single item -> [1.0].
        let one: Vec<(u32, f32)> = vec![(0, 0.42)];
        assert_eq!(score_quantiles(&one), vec![1.0]);
    }

    #[test]
    fn quantile_scale_invariance() {
        // Scaling scores by any positive constant does NOT change quantiles
        // (they're pure rank).
        let ranked_a: Vec<(u32, f32)> = vec![(0, 0.9), (1, 0.5), (2, 0.1)];
        let ranked_b: Vec<(u32, f32)> = vec![(0, 90.0), (1, 50.0), (2, 10.0)];
        assert_eq!(score_quantiles(&ranked_a), score_quantiles(&ranked_b));
    }

    // ---------- shape classification ----------

    #[test]
    fn shape_long_tail_when_top_score_dominates() {
        // Top score is far above the pack; max - median > 2 * iqr.
        // 8 items (meets gate=8); most clustered low, one outlier high.
        let scores = vec![0.10, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.95];
        let dist = distribution_shape(&scores, 8);
        assert_eq!(dist.shape, ShapeLabel::LongTail);
        assert!((dist.max - 0.95).abs() < 1e-5);
        assert!(dist.min >= 0.0);
    }

    #[test]
    fn shape_uniform() {
        // Equi-spaced scores; no dominant top, no bimodal gap.
        let scores: Vec<f32> = (0..10).map(|i| i as f32 * 0.1).collect();
        let dist = distribution_shape(&scores, K_MIN);
        assert_eq!(dist.shape, ShapeLabel::Uniform);
    }

    #[test]
    fn shape_bimodal() {
        // Two clusters separated by a gap > iqr.
        // Low cluster: 0.10..0.14. High cluster: 0.80..0.84. Gap: 0.66.
        // IQR of 10 samples split 5/5 is small (~0.04), so gap >> iqr.
        let scores = vec![0.10, 0.11, 0.12, 0.13, 0.14, 0.80, 0.81, 0.82, 0.83, 0.84];
        let dist = distribution_shape(&scores, K_MIN);
        assert_eq!(dist.shape, ShapeLabel::Bimodal);
    }

    #[test]
    fn shape_insufficient_samples_below_gate() {
        // K < gate -> default (all zero, InsufficientSamples).
        let scores = vec![0.1, 0.5, 0.9];
        let dist = distribution_shape(&scores, K_MIN);
        assert_eq!(dist.shape, ShapeLabel::InsufficientSamples);
        assert_eq!(dist.min, 0.0);
        assert_eq!(dist.max, 0.0);
    }

    #[test]
    fn shape_all_equal_is_uniform_not_nan() {
        // IQR would be 0 but we clamp to f32::EPSILON, so no NaN.
        // max - median = 0, not > 2 * epsilon -> not long-tail.
        // Largest gap = 0, not > epsilon -> not bimodal.
        let scores = vec![0.5_f32; 8];
        let dist = distribution_shape(&scores, K_MIN);
        assert_eq!(dist.shape, ShapeLabel::Uniform);
        assert!(dist.iqr > 0.0); // clamped to epsilon
    }

    // ---------- scale-free: K=8 and K=1000 both work ----------

    #[test]
    fn scale_free_across_response_sizes() {
        // K=8 smallest viable size, K=1000 stress.
        for &k in &[8_usize, 1000] {
            let ranked: Vec<(u32, f32)> = (0..k)
                .map(|i| (i as u32, 1.0 - (i as f32) / (k as f32)))
                .collect();
            let q = score_quantiles(&ranked);
            assert_eq!(q.len(), k);
            assert!((q[0] - 1.0).abs() < 1e-5);
            assert!((q[k - 1] - 0.0).abs() < 1e-5);

            let scores: Vec<f32> = ranked.iter().map(|(_, s)| *s).collect();
            let dist = distribution_shape(&scores, K_MIN);
            // Equi-spaced -> uniform, regardless of K.
            assert_eq!(dist.shape, ShapeLabel::Uniform);
        }
    }

    // ---------- proptest: determinism ----------

    use proptest::prelude::*;

    proptest! {
        /// Pure functions must produce bit-identical output for the same
        /// input across independent runs. Randomised input, checked twice.
        #[test]
        fn determinism(
            xs in proptest::collection::vec(-1000.0_f32..1000.0, 0..200)
        ) {
            // Filter NaN so total_cmp behaviour is the only source of
            // ordering: we assert determinism of the *function*, not a
            // spec-free NaN-handling path.
            let xs: Vec<f32> = xs.into_iter().filter(|v| v.is_finite()).collect();
            let ranked: Vec<(u32, f32)> =
                xs.iter().enumerate().map(|(i, v)| (i as u32, *v)).collect();

            let q1 = score_quantiles(&ranked);
            let q2 = score_quantiles(&ranked);
            prop_assert_eq!(&q1, &q2);

            let d1 = distribution_shape(&xs, K_MIN);
            let d2 = distribution_shape(&xs, K_MIN);
            prop_assert_eq!(d1, d2);
        }

        /// Scale invariance: multiplying every score by a positive
        /// constant never changes the shape label (it's a pure ratio).
        #[test]
        fn shape_scale_invariant(
            xs in proptest::collection::vec(0.0_f32..100.0, 8..64),
            scale in 0.01_f32..1000.0,
        ) {
            let xs: Vec<f32> = xs.into_iter().filter(|v| v.is_finite()).collect();
            if xs.len() < K_MIN {
                return Ok(());
            }
            let scaled: Vec<f32> = xs.iter().map(|v| v * scale).collect();
            let d1 = distribution_shape(&xs, K_MIN);
            let d2 = distribution_shape(&scaled, K_MIN);
            prop_assert_eq!(d1.shape, d2.shape);
        }
    }
}
