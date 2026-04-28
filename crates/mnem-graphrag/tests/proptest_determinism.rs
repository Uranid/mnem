//! Property: the content CID of a Leiden assignment is invariant
//! under permutations of the input edge list (for any fixed edge
//! set + seed). This is the determinism contract stated in
//! `compute_communities` docs.

use mnem_core::id::{NodeId, StableId};
use mnem_core::index::AuthoredSliceAdjacency;
use mnem_graphrag::compute_communities;
use proptest::prelude::*;

fn nid(i: u8) -> NodeId {
    let mut b = [0_u8; 16];
    b[15] = i;
    StableId::from_bytes(&b).unwrap()
}

proptest! {
    // Keep the case count modest: each case builds a BTreeMap and
    // runs Leiden twice. 48 is the per-batch budget used in E0.
    #![proptest_config(ProptestConfig { cases: 48, ..ProptestConfig::default() })]

    #[test]
    fn content_cid_invariant_under_edge_permutation(
        edges in prop::collection::vec((0_u8..15, 0_u8..15), 0..30),
        seed in any::<u64>(),
    ) {
        // Dedupe + strip self-loops so the two shuffled lists really
        // represent the same edge set (compute_communities does this
        // internally too, but we also want both orderings to feed
        // identical multisets into iter_edges).
        let mut canon: std::collections::BTreeSet<(u8, u8)> =
            std::collections::BTreeSet::new();
        for (a, b) in &edges {
            if a == b { continue; }
            let key = if a < b { (*a, *b) } else { (*b, *a) };
            canon.insert(key);
        }
        let ordered: Vec<(NodeId, NodeId)> =
            canon.iter().map(|&(a, b)| (nid(a), nid(b))).collect();
        let mut shuffled = ordered.clone();
        shuffled.reverse(); // simple non-trivial permutation

        let adj_a = AuthoredSliceAdjacency::new(&ordered);
        let adj_b = AuthoredSliceAdjacency::new(&shuffled);

        let ra = compute_communities(&adj_a, seed);
        let rb = compute_communities(&adj_b, seed);

        prop_assert_eq!(ra.content_cid(), rb.content_cid());
        prop_assert_eq!(ra.seed, rb.seed);
    }

    #[test]
    fn rerun_produces_identical_cid(
        edges in prop::collection::vec((0_u8..10, 0_u8..10), 0..20),
        seed in any::<u64>(),
    ) {
        let ids: Vec<(NodeId, NodeId)> = edges
            .iter()
            .filter(|(a, b)| a != b)
            .map(|(a, b)| (nid(*a), nid(*b)))
            .collect();
        let adj = AuthoredSliceAdjacency::new(&ids);
        let r1 = compute_communities(&adj, seed);
        let r2 = compute_communities(&adj, seed);
        prop_assert_eq!(r1.content_cid(), r2.content_cid());
    }
}
