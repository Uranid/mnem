//! Property tests for the C3 FIX-1 CommunityExpander.
//!
//! These assert the two contracts the expander guarantees:
//!
//! 1. **Additive superset**: every candidate in the input list
//!    appears in the output at the same position, with the same
//!    score. New members are strictly appended.
//! 2. **Byte-identity when lookup is empty**: a `CommunityLookup`
//!    built without an inverse map (equivalent to the pre-C3 API)
//!    makes the expander a no-op even when `enabled = true`.
//! 3. **Determinism**: same inputs -> byte-identical output across
//!    repeated runs.
//!
//! The recall-safe property (expanded_then_rerank recall >= baseline
//! recall) is a strict consequence of (1): the reranker sees a
//! superset of the input, so any ground-truth candidate present in
//! the baseline is also present in the expanded list, so R@K cannot
//! decrease except for tie-break reorder at the K boundary. That
//! tie-break risk is bounded by the decay factor (new members get
//! `seed_score * 0.85`, strictly less than any top-3 seed score),
//! so the top-K prefix is preserved. We pin the superset property
//! directly and rely on that derivation for the recall guarantee.

use std::collections::BTreeMap;

use mnem_core::id::{NodeId, StableId};
use mnem_core::retrieve::{
    CommunityFilterCfg, CommunityId, CommunityLookup, apply_community_filter,
};
use proptest::prelude::*;

fn nid(i: u32) -> NodeId {
    let mut b = [0_u8; 16];
    let be = i.to_be_bytes();
    b[12..16].copy_from_slice(&be);
    StableId::from_bytes(&b).unwrap()
}

/// Build a small deterministic fixture: `N` candidates with scores
/// descending from 1.0. Community assignment: node `i` in community
/// `i % num_communities`. Members per community: every node index
/// that maps to it, plus `extra_members_per_community` synthetic
/// members at high indices (so expansion is non-trivial).
fn fixture(
    n: usize,
    num_communities: u32,
    extras_per_com: usize,
) -> (Vec<(NodeId, f32)>, CommunityLookup) {
    let candidates: Vec<(NodeId, f32)> = (0..n)
        .map(|i| {
            let score = 1.0 - (i as f32) / (n as f32 + 1.0);
            (nid(i as u32), score)
        })
        .collect();

    let mut members: BTreeMap<CommunityId, Vec<NodeId>> = BTreeMap::new();
    for i in 0..n {
        let c = (i as u32) % num_communities;
        members.entry(c).or_default().push(nid(i as u32));
    }
    // Add synthetic extras per community at index space [1000, ...].
    for c in 0..num_communities {
        for k in 0..extras_per_com {
            let extra_idx = 1000 + (c as usize) * 100 + k;
            members.entry(c).or_default().push(nid(extra_idx as u32));
        }
    }
    let members_forward = members.clone();
    let members_inverse = members;
    let lookup = CommunityLookup::new_with_members(
        move |node| {
            for (cid, vs) in &members_forward {
                if vs.contains(node) {
                    return Some(*cid);
                }
            }
            None
        },
        move |cid| members_inverse.get(&cid).cloned().unwrap_or_default(),
    );
    (candidates, lookup)
}

#[test]
fn byte_identity_when_members_empty() {
    let cands: Vec<(NodeId, f32)> = (0..20).map(|i| (nid(i), 1.0 - (i as f32) / 21.0)).collect();
    // `CommunityLookup::new` supplies an empty members_of - the
    // expander cannot add anything.
    let lookup = CommunityLookup::new(|_| Some(0));
    let out = apply_community_filter(
        cands.clone(),
        &lookup,
        CommunityFilterCfg {
            enabled: true,
            expand_seeds: 3,
            max_per_community: 10,
            decay: 0.85,
            min_coverage: 0.1,
        },
    );
    assert_eq!(out, cands, "empty members_of must produce byte-identity");
}

#[test]
fn superset_preserves_input_prefix() {
    let (cands, lookup) = fixture(20, 3, 5);
    let out = apply_community_filter(
        cands.clone(),
        &lookup,
        CommunityFilterCfg {
            enabled: true,
            expand_seeds: 3,
            max_per_community: 10,
            decay: 0.85,
            min_coverage: 0.1,
        },
    );
    assert!(out.len() >= cands.len());
    for (i, c) in cands.iter().enumerate() {
        assert_eq!(
            out[i], *c,
            "input candidate {i} displaced: expander is not additive"
        );
    }
}

#[test]
fn expander_is_deterministic() {
    let (cands, lookup) = fixture(15, 4, 6);
    let cfg = CommunityFilterCfg {
        enabled: true,
        expand_seeds: 3,
        max_per_community: 10,
        decay: 0.85,
        min_coverage: 0.1,
    };
    let a = apply_community_filter(cands.clone(), &lookup, cfg);
    let b = apply_community_filter(cands.clone(), &lookup, cfg);
    let c = apply_community_filter(cands, &lookup, cfg);
    assert_eq!(a, b);
    assert_eq!(b, c);
}

proptest! {
    /// Property: expander output is a superset of the input
    /// candidate list (every input `(nid, score)` appears at the
    /// same position in the output). Scanned over arbitrary
    /// candidate-list shapes + community assignments.
    #[test]
    fn prop_expander_superset(
        n in 1usize..30,
        num_coms in 1u32..6,
        extras in 0usize..8,
        enabled in any::<bool>(),
        seeds in 0usize..5,
        max_per in 0usize..12,
    ) {
        let (cands, lookup) = fixture(n, num_coms, extras);
        let cfg = CommunityFilterCfg {
            enabled,
            expand_seeds: seeds,
            max_per_community: max_per,
            decay: 0.85,
            min_coverage: 0.1,
        };
        let out = apply_community_filter(cands.clone(), &lookup, cfg);
        prop_assert!(out.len() >= cands.len());
        for (i, c) in cands.iter().enumerate() {
            prop_assert_eq!(out[i], *c);
        }
    }

    /// Property: when disabled, output is byte-identical to input
    /// regardless of the lookup or any other knob.
    #[test]
    fn prop_disabled_is_passthrough(
        n in 1usize..30,
        num_coms in 1u32..6,
        extras in 0usize..8,
        seeds in 0usize..5,
        max_per in 0usize..12,
    ) {
        let (cands, lookup) = fixture(n, num_coms, extras);
        let cfg = CommunityFilterCfg {
            enabled: false,
            expand_seeds: seeds,
            max_per_community: max_per,
            decay: 0.85,
            min_coverage: 0.1,
        };
        let out = apply_community_filter(cands.clone(), &lookup, cfg);
        prop_assert_eq!(out, cands);
    }

    /// Property: "recall-safe" derived. If we define the
    /// ground-truth set as any subset of the candidate list, then
    /// recall@K of the expanded list is >= recall@K of the
    /// baseline list for every K >= input_len (since the input
    /// prefix is preserved). For K < input_len, the top-K of the
    /// expanded list is identical to top-K of the baseline
    /// (additive: new elements only appear at positions >=
    /// input_len), so R@K is exactly equal.
    #[test]
    fn prop_topk_prefix_identical(
        n in 1usize..20,
        num_coms in 1u32..6,
        extras in 0usize..8,
        k in 1usize..25,
    ) {
        let (cands, lookup) = fixture(n, num_coms, extras);
        let cfg = CommunityFilterCfg {
            enabled: true,
            expand_seeds: 3,
            max_per_community: 10,
            decay: 0.85,
            min_coverage: 0.1,
        };
        let out = apply_community_filter(cands.clone(), &lookup, cfg);
        let k_eff = k.min(cands.len());
        prop_assert_eq!(&out[..k_eff], &cands[..k_eff]);
    }
}
