//! PPR determinism proptest.
//!
//! E2 turn T2. For any small directed graph + any small personalization
//! vector, two runs of `ppr` produce byte-identical `f32` outputs. This
//! is the load-bearing property for agent-memory replay.

use std::collections::BTreeMap;

use mnem_core::id::NodeId;
use mnem_core::index::hybrid::AuthoredSliceAdjacency;
use mnem_core::ppr::{PprConfig, ppr};
use proptest::prelude::*;

fn nid(i: u8) -> NodeId {
    let mut bytes = [0u8; 16];
    bytes[15] = i;
    NodeId::from_bytes_raw(bytes)
}

/// Strategy for a small directed graph: up to 12 nodes, up to 40 edges,
/// no self-loops forced (PPR handles them either way but the test is
/// stricter if they show up).
fn arb_graph() -> impl Strategy<Value = Vec<(u8, u8)>> {
    // Node ids live in 0..12; edges in 0..40 count; (src, dst) each in
    // 0..12. `prop::collection::vec` over the tuple strategy gives us
    // a Vec<(u8, u8)>.
    prop::collection::vec((0u8..12, 0u8..12), 1..40)
}

/// Personalization values are small positive floats; keys are node ids
/// in 0..12 so they overlap the graph's id space.
fn arb_pers() -> impl Strategy<Value = Vec<(u8, f32)>> {
    prop::collection::vec((0u8..12, 0.1f32..10.0), 0..6)
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(48))]

    /// Two `ppr` runs over the same inputs must produce byte-identical
    /// f32 outputs. Compared via `f32::to_bits`.
    #[test]
    fn ppr_byte_identical_across_runs(raw_edges in arb_graph(), pers_raw in arb_pers()) {
        let edges: Vec<(NodeId, NodeId)> = raw_edges
            .into_iter()
            .map(|(s, d)| (nid(s), nid(d)))
            .collect();
        let adj = AuthoredSliceAdjacency::new(&edges);
        let mut pers: BTreeMap<NodeId, f32> = BTreeMap::new();
        for (k, v) in pers_raw {
            pers.insert(nid(k), v);
        }
        let cfg = PprConfig::default();
        let a = ppr(&adj, &pers, cfg);
        let b = ppr(&adj, &pers, cfg);
        prop_assert_eq!(a.len(), b.len());
        for (id, va) in &a {
            let vb = b.get(id).copied().unwrap_or(f32::NAN);
            prop_assert_eq!(
                va.to_bits(),
                vb.to_bits(),
                "byte-identity violated at node {:?}",
                id
            );
        }
    }

    /// PPR output is invariant under personalization L1-scaling. The
    /// function L1-normalises internally, so `{n0: 1.0, n1: 1.0}` and
    /// `{n0: 5.0, n1: 5.0}` must yield identical scores.
    #[test]
    fn ppr_invariant_under_pers_scaling(raw_edges in arb_graph(), pers_raw in arb_pers(), scale in 1.0f32..100.0) {
        let edges: Vec<(NodeId, NodeId)> = raw_edges
            .into_iter()
            .map(|(s, d)| (nid(s), nid(d)))
            .collect();
        let adj = AuthoredSliceAdjacency::new(&edges);
        let mut p1: BTreeMap<NodeId, f32> = BTreeMap::new();
        let mut p2: BTreeMap<NodeId, f32> = BTreeMap::new();
        for (k, v) in &pers_raw {
            p1.insert(nid(*k), *v);
            p2.insert(nid(*k), *v * scale);
        }
        let cfg = PprConfig::default();
        let a = ppr(&adj, &p1, cfg);
        let b = ppr(&adj, &p2, cfg);
        for (id, va) in &a {
            let vb = b.get(id).copied().unwrap_or(f32::NAN);
            prop_assert!(
                (va - vb).abs() < 1e-3,
                "scaling invariance violated: {} vs {} at {:?}",
                va, vb, id
            );
        }
    }

    /// L1 mass conservation: the output distribution sums to ~1.0 (or
    /// 0 for an empty graph).
    #[test]
    fn ppr_mass_conservation(raw_edges in arb_graph(), pers_raw in arb_pers()) {
        let edges: Vec<(NodeId, NodeId)> = raw_edges
            .into_iter()
            .map(|(s, d)| (nid(s), nid(d)))
            .collect();
        let adj = AuthoredSliceAdjacency::new(&edges);
        let mut pers: BTreeMap<NodeId, f32> = BTreeMap::new();
        for (k, v) in pers_raw {
            pers.insert(nid(k), v);
        }
        let scores = ppr(&adj, &pers, PprConfig::default());
        if scores.is_empty() {
            return Ok(());
        }
        let total: f32 = scores.values().sum();
        prop_assert!(
            (total - 1.0).abs() < 1e-2,
            "L1 mass drifted outside tolerance: {}",
            total
        );
    }
}
