//! Community **expander** stage for the retrieval pipeline (experiment E1).
//!
//! Historical note: v0.1.0 shipped a `CommunityFilter` that grouped the
//! fused top-K by `community_of(node_id)` and **dropped** candidates
//! from low-weight communities. Matrix v4 (LoCoMo) showed that drop
//! path regresses R@10 by -29pp: multi-hop answers live in the
//! minority communities the filter was pruning. C3 FIX-1 replaces the
//! drop semantic with an **additive** expander: we never drop a
//! candidate, we only *add* community-cohesive neighbours of the top
//! seeds. Worst case is neutral (no new members found); best case
//! lifts recall by surfacing community-siblings the base rankers
//! missed.
//!
//! # Algorithm
//!
//! Input: the fused top-K candidate list (K typically 50+) and a
//! [`CommunityLookup`] that resolves both `community_of(node)` and
//! `members_of(community)`.
//!
//! 1. Take the first `expand_seeds` candidates as seeds (default 3).
//! 2. For each seed, resolve its community via `community_of`.
//! 3. For each distinct seed community, fetch up to `max_per_community`
//!    additional members via `members_of`, skipping ids already in
//!    the candidate list.
//! 4. Score each new member as `seed.score * decay` (default decay
//!    0.85). The seed used is the *highest-scoring* seed that mapped
//!    to that community; ties broken by first-occurrence in the input
//!    list.
//! 5. Append the new scored members to the original candidate list,
//!    preserving the original order at the front.
//!
//! # Additive contract
//!
//! The expander MUST be additive: every element of the input list
//! appears in the output in the same relative order, with the same
//! score. New elements are appended at the end. The
//! `expander_superset` property test pins this invariant.
//!
//! # Flag-off contract
//!
//! When [`CommunityFilterCfg::enabled`] is `false`, or the lookup has
//! no `members_of` data, the stage is a byte-exact identity on the
//! candidate list. The `community_expander_zero_impact` test pins
//! this invariant.

use crate::id::NodeId;

/// Community identifier (opaque integer). Kept `u32` to match
/// `mnem_graphrag::CommunityId` without forcing a cross-crate
/// dependency.
pub type CommunityId = u32;

/// Configuration for the community-expander retrieval stage.
///
/// Default: disabled (expander is staged OFF until benchmarked).
/// See module docs for the expansion rule.
///
/// The `min_coverage` field is retained for DTO-level backward
/// compatibility with v0.1.0 clients but is **ignored** at runtime:
/// the expander has no coverage threshold (it never drops anything).
#[derive(Clone, Copy, Debug)]
pub struct CommunityFilterCfg {
    /// When `false`, the stage is a pass-through.
    pub enabled: bool,
    /// Number of top candidates treated as seeds for community
    /// expansion. Default 3.
    pub expand_seeds: usize,
    /// Per seed community, max number of additional members pulled
    /// into the candidate list. Default 10.
    pub max_per_community: usize,
    /// Score decay applied to expanded members relative to the seed
    /// score: `member_score = seed_score * decay`. Default 0.85.
    pub decay: f32,
    /// Legacy field retained for DTO compatibility with v0.1.0 clients.
    /// Ignored by the expander.
    pub min_coverage: f32,
}

impl Default for CommunityFilterCfg {
    fn default() -> Self {
        Self {
            enabled: false,
            expand_seeds: 3,
            max_per_community: 10,
            decay: 0.85,
            min_coverage: 0.5,
        }
    }
}

/// Opaque lookup from a node to the community it belongs to, plus
/// the inverse membership list.
///
/// Kept as a pair of boxed closures so `mnem-core` does not need to
/// depend on `mnem-graphrag`. The retriever builder wraps whatever
/// type the caller holds (typically `mnem_graphrag::CommunityAssignment`).
pub struct CommunityLookup {
    community_of_fn: Box<dyn Fn(&NodeId) -> Option<CommunityId> + Send + Sync + 'static>,
    members_of_fn: Box<dyn Fn(CommunityId) -> Vec<NodeId> + Send + Sync + 'static>,
}

impl CommunityLookup {
    /// Wrap a `community_of` closure as a [`CommunityLookup`] with
    /// an empty `members_of` (returns `vec![]` for all communities).
    /// The expander degenerates to the identity in this mode, which
    /// is the safe default for callers that only have the forward
    /// mapping.
    pub fn new<F>(f: F) -> Self
    where
        F: Fn(&NodeId) -> Option<CommunityId> + Send + Sync + 'static,
    {
        Self {
            community_of_fn: Box::new(f),
            members_of_fn: Box::new(|_| Vec::new()),
        }
    }

    /// Wrap both forward (`community_of`) and inverse (`members_of`)
    /// closures as a [`CommunityLookup`]. Required for the expander
    /// to actually add candidates.
    pub fn new_with_members<F, G>(community_of_fn: F, members_of_fn: G) -> Self
    where
        F: Fn(&NodeId) -> Option<CommunityId> + Send + Sync + 'static,
        G: Fn(CommunityId) -> Vec<NodeId> + Send + Sync + 'static,
    {
        Self {
            community_of_fn: Box::new(community_of_fn),
            members_of_fn: Box::new(members_of_fn),
        }
    }

    /// Look up the community of `node`.
    #[must_use]
    pub fn community_of(&self, node: &NodeId) -> Option<CommunityId> {
        (self.community_of_fn)(node)
    }

    /// List the members of `community`. Returns an empty vec when
    /// the caller did not supply an inverse map.
    #[must_use]
    pub fn members_of(&self, community: CommunityId) -> Vec<NodeId> {
        (self.members_of_fn)(community)
    }
}

impl std::fmt::Debug for CommunityLookup {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CommunityLookup").finish_non_exhaustive()
    }
}

/// Apply the community-expander stage to a fused candidate list.
///
/// Returns a new `Vec<(NodeId, f32)>` that is a **superset** of
/// `candidates` (input candidates preserved in original order, new
/// community-sibling candidates appended at the end). When
/// `cfg.enabled` is `false`, returns `candidates` unchanged.
///
/// # Additive guarantee
///
/// For every `(nid, score)` in the input, the output contains the
/// same `(nid, score)` at the same position as the input's N-th
/// occurrence. Expanded members are strictly appended; they cannot
/// displace or reorder the input.
#[must_use]
pub fn apply_community_filter(
    candidates: Vec<(NodeId, f32)>,
    lookup: &CommunityLookup,
    cfg: CommunityFilterCfg,
) -> Vec<(NodeId, f32)> {
    if !cfg.enabled || candidates.is_empty() || cfg.expand_seeds == 0 || cfg.max_per_community == 0
    {
        return candidates;
    }

    // Collect ids already present so we can skip them when expanding.
    let existing: std::collections::HashSet<NodeId> =
        candidates.iter().map(|(nid, _)| *nid).collect();

    // For each seed, find its community; remember the *best* (highest
    // score, earliest in input on tie) seed score per community so
    // expanded members get a principled score.
    //
    // Iteration uses a BTreeMap keyed by `CommunityId` so the
    // resulting per-community expansion order is deterministic.
    let seed_count = cfg.expand_seeds.min(candidates.len());
    let mut best_seed_score: std::collections::BTreeMap<CommunityId, f32> =
        std::collections::BTreeMap::new();
    let mut seed_order: Vec<CommunityId> = Vec::with_capacity(seed_count);
    for (nid, score) in candidates.iter().take(seed_count) {
        if let Some(cid) = lookup.community_of(nid) {
            if !best_seed_score.contains_key(&cid) {
                seed_order.push(cid);
            }
            let entry = best_seed_score.entry(cid).or_insert(*score);
            if *score > *entry {
                *entry = *score;
            }
        }
    }

    if seed_order.is_empty() {
        return candidates;
    }

    let decay = cfg.decay.clamp(0.0, 1.0);
    let mut out = candidates;
    let mut appended: std::collections::HashSet<NodeId> = std::collections::HashSet::new();
    for cid in seed_order {
        let seed_score = best_seed_score.get(&cid).copied().unwrap_or(0.0);
        let members = lookup.members_of(cid);
        let mut added = 0usize;
        for member in members {
            if added >= cfg.max_per_community {
                break;
            }
            if existing.contains(&member) || appended.contains(&member) {
                continue;
            }
            out.push((member, seed_score * decay));
            appended.insert(member);
            added += 1;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::id::{NodeId, StableId};
    use std::collections::BTreeMap;

    fn nid(i: u8) -> NodeId {
        let mut b = [0_u8; 16];
        b[15] = i;
        StableId::from_bytes(&b).unwrap()
    }

    #[test]
    fn disabled_is_passthrough() {
        let cands = vec![(nid(0), 1.0_f32), (nid(1), 0.5)];
        let lookup = CommunityLookup::new(|_| Some(42));
        let out = apply_community_filter(
            cands.clone(),
            &lookup,
            CommunityFilterCfg {
                enabled: false,
                ..Default::default()
            },
        );
        assert_eq!(out, cands);
    }

    #[test]
    fn empty_members_is_passthrough() {
        // `CommunityLookup::new` supplies an empty members_of. The
        // expander cannot add anything, so output == input even when
        // enabled=true.
        let cands = vec![(nid(0), 1.0_f32), (nid(1), 0.5)];
        let lookup = CommunityLookup::new(|_| Some(0));
        let out = apply_community_filter(
            cands.clone(),
            &lookup,
            CommunityFilterCfg {
                enabled: true,
                ..Default::default()
            },
        );
        assert_eq!(out, cands);
    }

    #[test]
    fn expander_appends_community_members() {
        // Seed nid(0) is in community 0 whose members are [nid(0),
        // nid(10), nid(11)]. Expander should append nid(10) and
        // nid(11) after the input list.
        let seed = nid(0);
        let m1 = nid(10);
        let m2 = nid(11);
        let cands = vec![(seed, 1.0_f32)];
        let members: BTreeMap<CommunityId, Vec<NodeId>> =
            [(0, vec![seed, m1, m2])].into_iter().collect();
        let lookup = CommunityLookup::new_with_members(
            move |n| if *n == seed { Some(0) } else { None },
            move |cid| members.get(&cid).cloned().unwrap_or_default(),
        );
        let out = apply_community_filter(
            cands.clone(),
            &lookup,
            CommunityFilterCfg {
                enabled: true,
                expand_seeds: 3,
                max_per_community: 10,
                decay: 0.85,
                min_coverage: 0.5,
            },
        );
        assert_eq!(out.len(), 3);
        // Input preserved in order, at the front.
        assert_eq!(out[0], (seed, 1.0));
        // Appended with decayed score.
        assert!(
            out.iter()
                .any(|(n, s)| *n == m1 && (*s - 0.85).abs() < 1e-6)
        );
        assert!(
            out.iter()
                .any(|(n, s)| *n == m2 && (*s - 0.85).abs() < 1e-6)
        );
    }

    #[test]
    fn additive_superset_property() {
        // For an arbitrary (small) candidate list and lookup, every
        // input candidate must appear in the output in its original
        // position.
        let a = nid(0);
        let b = nid(1);
        let c = nid(2);
        let extra = nid(42);
        let cands = vec![(a, 0.9_f32), (b, 0.5), (c, 0.1)];
        let members: BTreeMap<CommunityId, Vec<NodeId>> =
            [(0, vec![a, extra])].into_iter().collect();
        let lookup = CommunityLookup::new_with_members(
            move |n| if *n == a { Some(0) } else { None },
            move |cid| members.get(&cid).cloned().unwrap_or_default(),
        );
        let out = apply_community_filter(
            cands.clone(),
            &lookup,
            CommunityFilterCfg {
                enabled: true,
                expand_seeds: 3,
                max_per_community: 10,
                decay: 0.85,
                min_coverage: 0.5,
            },
        );
        // All input candidates in the same order at the front.
        assert!(out.len() >= cands.len());
        for (i, c) in cands.iter().enumerate() {
            assert_eq!(out[i], *c);
        }
    }

    #[test]
    fn zero_expand_seeds_is_passthrough() {
        let cands = vec![(nid(0), 1.0_f32)];
        let members: BTreeMap<CommunityId, Vec<NodeId>> =
            [(0, vec![nid(0), nid(1)])].into_iter().collect();
        let lookup = CommunityLookup::new_with_members(
            move |_| Some(0),
            move |cid| members.get(&cid).cloned().unwrap_or_default(),
        );
        let out = apply_community_filter(
            cands.clone(),
            &lookup,
            CommunityFilterCfg {
                enabled: true,
                expand_seeds: 0,
                max_per_community: 10,
                decay: 0.85,
                min_coverage: 0.5,
            },
        );
        assert_eq!(out, cands);
    }

    #[test]
    fn unknown_community_is_passthrough() {
        // Seed has no community assignment; expander degenerates to
        // identity.
        let cands = vec![(nid(0), 1.0_f32), (nid(1), 1.0)];
        let lookup = CommunityLookup::new_with_members(|_| None, |_| Vec::new());
        let out = apply_community_filter(
            cands.clone(),
            &lookup,
            CommunityFilterCfg {
                enabled: true,
                ..Default::default()
            },
        );
        assert_eq!(out, cands);
    }

    #[test]
    fn deduplicates_existing_candidates() {
        // Community member already in input list should not be
        // re-added.
        let seed = nid(0);
        let dup = nid(1);
        let cands = vec![(seed, 1.0_f32), (dup, 0.5)];
        let members: BTreeMap<CommunityId, Vec<NodeId>> =
            [(0, vec![seed, dup, nid(7)])].into_iter().collect();
        let lookup = CommunityLookup::new_with_members(
            move |_| Some(0),
            move |cid| members.get(&cid).cloned().unwrap_or_default(),
        );
        let out = apply_community_filter(
            cands,
            &lookup,
            CommunityFilterCfg {
                enabled: true,
                expand_seeds: 3,
                max_per_community: 10,
                decay: 0.85,
                min_coverage: 0.5,
            },
        );
        // dup must appear exactly once.
        let dup_count = out.iter().filter(|(n, _)| *n == dup).count();
        assert_eq!(dup_count, 1);
        // nid(7) was appended.
        assert!(out.iter().any(|(n, _)| *n == nid(7)));
    }
}
