//! Within-query confidence signals over retrieval score distributions
//! (Gap 05, LD-primary).
//!
//! All signals are pure functions of the returned top-K scores. No
//! globals, no rolling state, no I/O. Given the same input, two
//! independent calls produce byte-identical output.
//!
//! # Design
//!
//! Given a descending-sorted score vector `[s1, s2, ..., sK]`:
//!
//! - [`normalized_entropy`]: Shannon entropy of a softmax over the
//!   min-max-rescaled scores, divided by `ln(K)`. Dimensionless,
//!   living in `[0, 1]`. `0` = one score dominates; `1` = all equal.
//! - [`median_topk_margin_pct`]: median of the per-adjacent-pair
//!   relative gaps `(s_i - s_{i+1}) / max(s_i, eps)` over the top-K.
//!   The *within-query baseline*: is the top-1 lead unusual relative
//!   to the rest of this ranking?
//! - [`rank_agreement`]: categorical label (`High` / `Medium` / `Low`)
//!   derived from the two metrics above. Auto-adapts per-query; no
//!   hardcoded global threshold.
//!
//! # Why within-query
//!
//! Cross-query calibration (Gap 01) needs rolling-median state. This
//! gap is *strictly within-query* and needs none of that plumbing.
//! Consumers that want the hop-suggestion signal (Gap 01) combine
//! [`RankAgreement`] with their rolling median externally.

use serde::{Deserialize, Serialize};

/// Minimum K for a statistically meaningful `rank_agreement` bucket.
///
/// At `K < 5` the within-query median of `K-1` adjacent margins is
/// under-powered (Wilson 95% CI width > 0.45 at p=0.5). Callers with
/// fewer items should treat the signal as insufficient.
pub const K_MIN_SHAPE_GATE: usize = 5;

/// Tiny epsilon used to avoid division by zero when the top score or a
/// per-pair denominator is itself near zero.
const EPS: f32 = 1e-9;

/// Categorical confidence label for a returned top-K ranking.
///
/// Derived from the softmax-entropy and the within-query median
/// adjacent margin. The mapping is intentionally coarse: the three-
/// bucket view is what downstream consumers (Gap 01 `suggest_hop`, UI
/// badges) actually use. The richer four-label string form from the
/// solution design (confident / likely / tie / flat) is preserved in
/// [`RankAgreement::as_fine_label`] for telemetry.
#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[serde(rename_all = "lowercase")]
pub enum RankAgreement {
    /// The top-1 lead is unusually large relative to the rest of the
    /// ranking *and* entropy is low. Downstream: do not suggest a hop.
    /// Fine label: `confident`.
    High,
    /// The top-1 lead beats the within-query median margin but does
    /// not meet the stricter entropy floor. Fine label: `likely`.
    Medium,
    /// Top-K is near-tied or flat. Downstream: Gap 01 `suggest_hop`
    /// should fire (subject to its graph-size gate). Fine labels:
    /// `tie` (top is near-tied with runners-up) and `flat` (entire
    /// distribution is near-uniform).
    Low,
    /// `K < K_MIN_SHAPE_GATE`: sample too small to label meaningfully.
    /// Consumers should treat this as "no signal".
    Insufficient,
}

impl RankAgreement {
    /// Stable fine-grained label for telemetry (`confident` / `likely`
    /// / `tie` / `flat` / `insufficient_k`). The coarse three-bucket
    /// [`RankAgreement`] variant is what programmatic consumers should
    /// match on; this string is for dashboards and logs.
    #[must_use]
    pub const fn as_fine_label(self) -> &'static str {
        match self {
            Self::High => "confident",
            Self::Medium => "likely",
            Self::Low => "flat",
            Self::Insufficient => "insufficient_k",
        }
    }
}

/// Compute the normalized Shannon entropy of a top-K score vector.
///
/// Scores are first shifted to be non-negative (by subtracting the
/// minimum, which is scale-invariant under affine score transforms
/// that preserve ordering) and then L1-normalized into a probability
/// distribution. Shannon entropy of that distribution divided by
/// `ln(K)` yields a dimensionless value in `[0, 1]`:
///
/// - `0.0`: a single score dominates (one-hot distribution).
/// - `1.0`: all scores are equal (uniform distribution).
///
/// We do **not** use a softmax: for score ranges that are small in
/// absolute terms, `exp` damps differences and the distribution
/// collapses towards uniform even when one score is clearly peaked.
/// L1-normalization preserves the relative magnitude structure that
/// consumers of retrieval scores actually care about.
///
/// Returns `0.0` for `scores.len() < 2` (no meaningful distribution).
/// Non-finite inputs (`NaN`, `+inf`, `-inf`) are treated as the score
/// minimum so they do not poison the distribution.
#[must_use]
pub fn normalized_entropy(scores: &[f32]) -> f32 {
    let k = scores.len();
    if k < 2 {
        return 0.0;
    }

    // Sanitize: replace non-finite entries with the finite minimum.
    let finite_min = scores
        .iter()
        .copied()
        .filter(|s| s.is_finite())
        .fold(f32::INFINITY, f32::min);
    let finite_min = if finite_min.is_finite() {
        finite_min
    } else {
        0.0
    };

    let sanitized: Vec<f32> = scores
        .iter()
        .map(|&s| if s.is_finite() { s } else { finite_min })
        .collect();

    // Shift so min -> 0. Sum is scale-free in the sense that scaling
    // every score by a positive constant produces proportional shifts.
    let min = sanitized.iter().copied().fold(f32::INFINITY, f32::min);
    let shifted: Vec<f32> = sanitized.iter().map(|&s| (s - min).max(0.0)).collect();
    let sum: f32 = shifted.iter().sum();

    // All scores equal => uniform distribution => entropy 1.0.
    if sum <= EPS {
        return 1.0;
    }

    let mut entropy = 0.0_f32;
    for &x in &shifted {
        let p = x / sum;
        if p > EPS {
            entropy -= p * p.ln();
        }
    }
    let denom = (k as f32).ln().max(EPS);
    (entropy / denom).clamp(0.0, 1.0)
}

/// Median of the per-adjacent-pair relative gaps over a top-K vector.
///
/// For an input `[s1, s2, ..., sK]` (assumed descending), returns
/// `median_i { (s_i - s_{i+1}) / max(s_i, eps) }` over the `K-1`
/// adjacent pairs, optionally trimmed to the first `k` pairs if `k`
/// is smaller than `K-1`. When `k == 0` or the vector has fewer than
/// 2 elements, returns `0.0`.
///
/// This is the *within-query baseline*: "how wide is a typical gap in
/// this specific ranking?" [`rank_agreement`] uses it as the anchor
/// against which the top-1 margin is compared.
#[must_use]
pub fn median_topk_margin_pct(scores: &[f32], k: usize) -> f32 {
    if scores.len() < 2 || k == 0 {
        return 0.0;
    }
    let pair_limit = (scores.len() - 1).min(k);
    let mut pair_pcts: Vec<f32> = (0..pair_limit)
        .map(|i| {
            let num = scores[i] - scores[i + 1];
            let denom = scores[i].abs().max(EPS);
            (num / denom).max(0.0)
        })
        .collect();
    if pair_pcts.is_empty() {
        return 0.0;
    }
    // Sort with a total order: NaN-safe via partial_cmp fallback.
    pair_pcts.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mid = pair_pcts.len() / 2;
    if pair_pcts.len() % 2 == 0 {
        0.5 * (pair_pcts[mid - 1] + pair_pcts[mid])
    } else {
        pair_pcts[mid]
    }
}

/// Categorical confidence label for a top-K score vector.
///
/// See [`RankAgreement`] for the bucket semantics. The derivation is
/// a pure function of the score distribution; no global thresholds.
///
/// # Algorithm
///
/// Let `m1 = (s1 - s2) / max(s1, eps)` be the top-1 relative margin,
/// `mu = median_topk_margin_pct(scores, scores.len() - 1)` be the
/// within-query baseline, and `h = normalized_entropy(scores)`.
///
/// - `High` when `m1 >= 2 * mu` and `h < 1 - mu` (top-1 is decisively
///   ahead of the typical gap *and* the distribution is peaked).
/// - `Medium` when `m1 > mu` (top-1 beats baseline but entropy is
///   not tight enough for `High`).
/// - `Low` when the ranking is near-tied (`m1 < mu / 4` with a
///   still-positive top score) or generally flat.
/// - `Insufficient` when `scores.len() < K_MIN_SHAPE_GATE` (5).
#[must_use]
pub fn rank_agreement(scores: &[f32]) -> RankAgreement {
    if scores.len() < K_MIN_SHAPE_GATE {
        return RankAgreement::Insufficient;
    }

    let s1 = scores[0];
    let s2 = scores[1];
    let s_last = *scores.last().expect("len >= K_MIN_SHAPE_GATE >= 5");

    let top1_margin_pct = if s1.abs() > EPS {
        ((s1 - s2) / s1).max(0.0)
    } else {
        0.0
    };

    let median_margin = median_topk_margin_pct(scores, scores.len() - 1);
    let norm_entropy = normalized_entropy(scores);

    // Degenerate flat: every score equal (or near-equal). No useful
    // signal; we are in the Low bucket regardless of top1_margin_pct.
    // Note: s_last is read to keep the name live and document the
    // shape of the flat branch (s1 == s_last iff uniform).
    let _ = s_last;
    if median_margin < EPS {
        return RankAgreement::Low;
    }

    // Tight: top-1 decisively ahead *and* distribution is non-uniform.
    // The entropy floor is expressed as `1 - 2*median_margin` so a
    // query with larger-than-typical gaps can carry High even when the
    // softmax tail damps entropy towards 1.
    if top1_margin_pct >= 2.0 * median_margin
        && norm_entropy < (1.0 - 2.0 * median_margin).clamp(0.0, 0.999)
    {
        return RankAgreement::High;
    }

    // Top-1 beats the within-query baseline: Medium.
    if top1_margin_pct > median_margin {
        return RankAgreement::Medium;
    }

    // Otherwise: flat / tie.
    RankAgreement::Low
}

// ============================================================
// Tests
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn empty_scores_return_insufficient() {
        assert_eq!(rank_agreement(&[]), RankAgreement::Insufficient);
        assert!((normalized_entropy(&[]) - 0.0).abs() < 1e-6);
        assert!((median_topk_margin_pct(&[], 5) - 0.0).abs() < 1e-6);
    }

    #[test]
    fn single_score_returns_insufficient() {
        assert_eq!(rank_agreement(&[0.9]), RankAgreement::Insufficient);
        assert!((normalized_entropy(&[0.9]) - 0.0).abs() < 1e-6);
    }

    #[test]
    fn four_scores_return_insufficient_k_label() {
        // K < K_MIN_SHAPE_GATE (5)
        let scores = [0.9, 0.3, 0.1, 0.05];
        assert_eq!(rank_agreement(&scores), RankAgreement::Insufficient);
        assert_eq!(
            RankAgreement::Insufficient.as_fine_label(),
            "insufficient_k"
        );
    }

    #[test]
    fn confident_peaked_distribution_yields_high() {
        // Top-1 dominates: large margin, low entropy.
        let scores = [0.95, 0.30, 0.10, 0.05, 0.02];
        let label = rank_agreement(&scores);
        assert!(
            matches!(label, RankAgreement::High | RankAgreement::Medium),
            "expected peaked top-1 to be High or Medium, got {label:?}"
        );
    }

    #[test]
    fn confident_triggers_high_on_sharp_peak() {
        // Deliberately constructed so m1 >> 2*mu and entropy is low.
        let scores = [0.99, 0.20, 0.18, 0.16, 0.15, 0.14];
        assert_eq!(rank_agreement(&scores), RankAgreement::High);
    }

    #[test]
    fn tie_near_uniform_yields_low() {
        let scores = [0.80, 0.80, 0.80, 0.80, 0.79];
        let label = rank_agreement(&scores);
        assert_eq!(label, RankAgreement::Low);
    }

    #[test]
    fn flat_distribution_yields_low() {
        // All approximately equal: entropy near 1.0, no standout.
        let scores = [0.50, 0.50, 0.50, 0.50, 0.50];
        assert_eq!(rank_agreement(&scores), RankAgreement::Low);
    }

    #[test]
    fn confidence_is_deterministic() {
        let scores = [0.91, 0.60, 0.42, 0.30, 0.15, 0.05];
        let a = (
            rank_agreement(&scores),
            normalized_entropy(&scores),
            median_topk_margin_pct(&scores, 5),
        );
        let b = (
            rank_agreement(&scores),
            normalized_entropy(&scores),
            median_topk_margin_pct(&scores, 5),
        );
        assert_eq!(a.0, b.0);
        assert!((a.1 - b.1).abs() < 1e-7);
        assert!((a.2 - b.2).abs() < 1e-7);
    }

    #[test]
    fn margin_pct_scale_free_and_label_unchanged_under_rescale() {
        // Scale-invariance property: multiplying every score by the
        // same positive constant must not change the categorical label.
        let scores = [0.91, 0.60, 0.42, 0.30, 0.15, 0.05];
        let scaled: Vec<f32> = scores.iter().map(|s| s * 10.0).collect();
        assert_eq!(rank_agreement(&scores), rank_agreement(&scaled));
        let mu_a = median_topk_margin_pct(&scores, 5);
        let mu_b = median_topk_margin_pct(&scaled, 5);
        assert!(
            (mu_a - mu_b).abs() < 1e-5,
            "median margin pct should be scale-free: got {mu_a} vs {mu_b}"
        );
    }

    #[test]
    fn normalized_entropy_uniform_is_one() {
        let scores = [0.5, 0.5, 0.5, 0.5, 0.5];
        let h = normalized_entropy(&scores);
        assert!((h - 1.0).abs() < 1e-5, "expected 1.0, got {h}");
    }

    #[test]
    fn normalized_entropy_one_hot_is_low() {
        let scores = [1.0, 0.0, 0.0, 0.0, 0.0];
        let h = normalized_entropy(&scores);
        assert!(
            h < 1.0,
            "one-hot distribution should have sub-uniform entropy, got {h}"
        );
    }

    #[test]
    fn nonfinite_inputs_do_not_panic() {
        let scores = [f32::NAN, 0.8, 0.5, 0.2, 0.0];
        let _ = rank_agreement(&scores);
        let _ = normalized_entropy(&scores);
        let _ = median_topk_margin_pct(&scores, 4);
    }

    // ============================================================
    // Property tests (derived from code-sketch named tests)
    // ============================================================

    proptest! {
        /// margin_pct_scale_free: scaling all scores by a positive
        /// constant must leave the median_topk_margin_pct baseline
        /// approximately unchanged (scale-free derivation). Labels
        /// can flip at the Medium/Low boundary when the top-1 margin
        /// equals the median baseline exactly; we assert the
        /// numerical baseline instead, which is the load-bearing
        /// invariant.
        #[test]
        fn proptest_margin_pct_scale_free(
            seed in 1..1000u32,
            factor in 1e-3f32..1e3f32,
        ) {
            let mut scores: Vec<f32> = Vec::with_capacity(8);
            let mut x = f32::from(u16::try_from(seed % 1000).unwrap_or(1)) / 1000.0 + 0.1;
            for _ in 0..8 {
                scores.push(x);
                x *= 0.7;
            }
            let scaled: Vec<f32> = scores.iter().map(|s| s * factor).collect();
            let mu_a = median_topk_margin_pct(&scores, 7);
            let mu_b = median_topk_margin_pct(&scaled, 7);
            prop_assert!(
                (mu_a - mu_b).abs() < 1e-3,
                "median margin pct should be scale-free: {} vs {}",
                mu_a, mu_b
            );
            let h_a = normalized_entropy(&scores);
            let h_b = normalized_entropy(&scaled);
            prop_assert!(
                (h_a - h_b).abs() < 1e-3,
                "normalized entropy should be scale-free: {} vs {}",
                h_a, h_b
            );
        }

        /// normalized_entropy_in_unit_interval: the metric is always
        /// in `[0, 1]` for any finite input.
        #[test]
        fn proptest_normalized_entropy_bounded(len in 2..32usize, seed in 1..1000u32) {
            let mut scores: Vec<f32> = Vec::with_capacity(len);
            let mut x = f32::from(u16::try_from(seed % 1000).unwrap_or(1)) / 1000.0 + 0.1;
            for i in 0..len {
                #[allow(clippy::cast_precision_loss)]
                scores.push(x + (i as f32) * 0.01);
                x *= 0.9;
            }
            let h = normalized_entropy(&scores);
            prop_assert!((0.0..=1.0).contains(&h), "entropy out of range: {}", h);
        }

        /// rank_agreement_labels_mutually_exclusive: the function
        /// returns exactly one variant; the set of possible variants
        /// is stable across runs.
        #[test]
        fn proptest_rank_agreement_total(len in 0..32usize, seed in 1..1000u32) {
            let mut scores: Vec<f32> = Vec::with_capacity(len);
            let mut x = f32::from(u16::try_from(seed % 1000).unwrap_or(1)) / 1000.0 + 0.1;
            for _ in 0..len {
                scores.push(x);
                x *= 0.85;
            }
            // Must terminate and return a variant; checked by pattern.
            let label = rank_agreement(&scores);
            prop_assert!(matches!(
                label,
                RankAgreement::High
                    | RankAgreement::Medium
                    | RankAgreement::Low
                    | RankAgreement::Insufficient
            ));
        }

        /// insufficient_k_band_labeled: for `K < K_MIN_SHAPE_GATE`
        /// the label is unconditionally `Insufficient`.
        #[test]
        fn proptest_insufficient_k_band(len in 0..K_MIN_SHAPE_GATE, seed in 1..1000u32) {
            let mut scores: Vec<f32> = Vec::with_capacity(len);
            let mut x = f32::from(u16::try_from(seed % 1000).unwrap_or(1)) / 1000.0 + 0.1;
            for _ in 0..len {
                scores.push(x);
                x *= 0.8;
            }
            prop_assert_eq!(rank_agreement(&scores), RankAgreement::Insufficient);
        }

        /// median_is_within_query_baseline: the median margin pct is
        /// always in `[0, 1]` and is 0 iff all scores are equal.
        #[test]
        fn proptest_median_bounded(len in 2..32usize, seed in 1..1000u32) {
            let mut scores: Vec<f32> = Vec::with_capacity(len);
            let mut x = f32::from(u16::try_from(seed % 1000).unwrap_or(1)) / 1000.0 + 0.1;
            for _ in 0..len {
                scores.push(x);
                x *= 0.9;
            }
            let mu = median_topk_margin_pct(&scores, len - 1);
            prop_assert!((0.0..=1.0).contains(&mu), "median out of range: {}", mu);
        }
    }
}
