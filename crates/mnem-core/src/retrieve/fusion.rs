//! Rank-list fusion functions + the `prefetch_and_filter`
//! helper that threads a candidate set through label /
//! property / temporal gates before ranker scoring.
//!
//! Extracted from `retrieve.rs` in R3; bodies unchanged.

use std::collections::{HashMap, HashSet};

use crate::error::Error;
use crate::id::NodeId;
use crate::index::PropPredicate;
use crate::objects::Node;
use crate::repo::readonly::ReadonlyRepo;

/// Blend two `(NodeId, score)` lists by min-max-normalising each
/// list's scores to `[0, 1]` and returning the weighted sum. An
/// alternative to RRF for callers who prefer the fused score to carry
/// recognizable magnitude information (a 0.92 means "close to both
/// ranker's best hit"; a 0.05 means "barely on the map").
///
/// RRF is deliberately *rank-based*; its score is a function of where
/// a document landed, not how strong the match actually was. That is a
/// feature when combining heterogeneous rankers (cosine is in
/// `[-1, 1]`; sparse dot products are unbounded positive; fusing
/// their native numbers by naive addition would be dominated by the
/// larger range). Min-max normalisation inside each list first brings
/// them to a common scale, then the blend is apples-to-apples.
///
/// Documents appearing in only one list get the normalised score from
/// that list; the other contribution is zero. Ties break on `NodeId`.
///
/// Caveat: this blend is more sensitive to the score distribution than
/// RRF. One outlier at the top of a list will compress every other hit
/// in that list toward zero. Prefer RRF when the score distributions
/// are likely to be skewed.
pub fn score_normalized_fusion(
    a_hits: &[(NodeId, f32)],
    a_weight: f32,
    b_hits: &[(NodeId, f32)],
    b_weight: f32,
) -> Vec<(NodeId, f32)> {
    fn normalize(hits: &[(NodeId, f32)]) -> HashMap<NodeId, f32> {
        if hits.is_empty() {
            return HashMap::new();
        }
        let scores: Vec<f32> = hits.iter().map(|(_, s)| *s).collect();
        let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let min = scores.iter().copied().fold(f32::INFINITY, f32::min);
        let range = max - min;
        let mut out = HashMap::with_capacity(hits.len());
        if range.abs() < 1e-12 {
            // Degenerate: every score identical. Give every hit the
            // same mid-value so the blend sees "uniformly present" not
            // "uniformly at max".
            for (id, _) in hits {
                out.insert(*id, 0.5);
            }
        } else {
            for (id, s) in hits {
                out.insert(*id, (s - min) / range);
            }
        }
        out
    }

    let a_norm = normalize(a_hits);
    let b_norm = normalize(b_hits);
    let mut all: HashMap<NodeId, f32> = HashMap::with_capacity(a_norm.len() + b_norm.len());
    for (id, s) in &a_norm {
        *all.entry(*id).or_insert(0.0) += a_weight * s;
    }
    for (id, s) in &b_norm {
        *all.entry(*id).or_insert(0.0) += b_weight * s;
    }
    let mut fused: Vec<(NodeId, f32)> = all.into_iter().collect();
    fused.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.0.cmp(&b.0))
    });
    fused
}

/// N-lane min-max normalised convex-combination fusion.
///
/// Bruch et al. 2023 (arXiv:2210.11934) shows this form beats RRF for
/// hybrid retrieval both in-domain and out-of-domain whenever any
/// tuning data exists. Each lane's scores are rescaled to `[0, 1]`
/// (degenerate lanes where every score is identical map to a uniform
/// 0.5 contribution, same rule as [`score_normalized_fusion`]). The
/// fused score is the per-node weighted sum across lanes.
///
/// `lanes` is `&[(hits, weight)]` where `hits` is `(NodeId, score)`
/// pairs in score-descending order (whatever the caller produced).
/// Ties break on NodeId ascending, matching every other ranker in the
/// crate.
pub fn convex_min_max_fusion(lanes: &[(Vec<(NodeId, f32)>, f32)]) -> Vec<(NodeId, f32)> {
    let mut totals: HashMap<NodeId, f32> = HashMap::new();
    for (hits, weight) in lanes {
        if hits.is_empty() || *weight == 0.0 {
            continue;
        }
        // Degenerate-score guard: if max == min, every hit contributes
        // 0.5 * weight. Lets a tied-scores lane still pull in its IDs
        // without inflating them to 1.0.
        let (min, max) = hits
            .iter()
            .map(|(_, s)| *s)
            .fold((f32::INFINITY, f32::NEG_INFINITY), |(lo, hi), s| {
                (lo.min(s), hi.max(s))
            });
        let range = max - min;
        for (id, s) in hits {
            let norm = if range.abs() < 1e-12 {
                0.5
            } else {
                (s - min) / range
            };
            *totals.entry(*id).or_insert(0.0) += weight * norm;
        }
    }
    let mut fused: Vec<(NodeId, f32)> = totals.into_iter().collect();
    fused.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.0.cmp(&b.0))
    });
    fused
}

/// Reciprocal Rank Fusion over a set of ranked `NodeId` lists.
///
/// Convenience wrapper around [`weighted_reciprocal_rank_fusion`] with
/// all lists weighted 1.0 - the canonical RRF behaviour.
pub fn reciprocal_rank_fusion(lists: &[Vec<NodeId>], k: f32) -> Vec<(NodeId, f32)> {
    let weighted: Vec<(Vec<NodeId>, f32)> = lists.iter().map(|l| (l.clone(), 1.0)).collect();
    weighted_reciprocal_rank_fusion(&weighted, k)
}

/// Weighted Reciprocal Rank Fusion over a set of `(ranked list, weight)`
/// pairs.
///
/// For each list, a node at zero-based rank `i` contributes
/// `weight / (k + i + 1)`. A node's final score is the sum of
/// contributions across the lists that surfaced it. The returned vector
/// is sorted by score DESC, ties broken by `NodeId` ASC.
///
/// Weights let callers bias retrieval toward a particular ranker -
/// e.g. `sparse_weight=0.3, vector_weight=1.0` when the dense
/// embedding space is considered more reliable than the sparse lane.
pub fn weighted_reciprocal_rank_fusion(lists: &[(Vec<NodeId>, f32)], k: f32) -> Vec<(NodeId, f32)> {
    let mut scores: HashMap<NodeId, f32> = HashMap::new();
    for (list, weight) in lists {
        if *weight == 0.0 {
            continue;
        }
        for (rank, id) in list.iter().enumerate() {
            let contrib = weight / (k + (rank as f32) + 1.0);
            *scores.entry(*id).or_insert(0.0) += contrib;
        }
    }
    let mut fused: Vec<(NodeId, f32)> = scores.into_iter().collect();
    fused.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.0.cmp(&b.0))
    });
    fused
}

/// Prefetch the `Node` for every fused candidate, optionally gating on
/// label / prop filters. Returning `(id, score, node)` tuples lets the
/// pack loop skip a second `lookup_node` round-trip per surviving
/// candidate. Duplicates in the fused list (a node surfaced by more
/// than one ranker) collapse to first occurrence.
pub(super) fn prefetch_and_filter(
    repo: &ReadonlyRepo,
    ranked: Vec<(NodeId, f32)>,
    label: Option<&str>,
    prop: Option<&(String, PropPredicate)>,
) -> Result<Vec<(NodeId, f32, Node)>, Error> {
    let mut out = Vec::with_capacity(ranked.len());
    let mut seen: HashSet<NodeId> = HashSet::with_capacity(ranked.len());
    for (id, score) in ranked {
        if !seen.insert(id) {
            continue;
        }
        let Some(node) = repo.lookup_node(&id)? else {
            continue;
        };
        if let Some(lbl) = label
            && node.ntype != lbl
        {
            continue;
        }
        if let Some((name, PropPredicate::Eq(value))) = prop
            && node.props.get(name) != Some(value)
        {
            continue;
        }
        out.push((id, score, node));
    }
    Ok(out)
}
