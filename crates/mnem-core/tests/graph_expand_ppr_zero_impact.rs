//! Zero-impact test: HybridAdjacency with `EmptyKnnSource` + authored
//! edges must produce byte-identical PPR output to the authored edges
//! alone.
//!
//! E2 turn T2. This is the load-bearing guarantee for the staged-
//! rollout feature flag: turning the KNN-derived edge source "off" by
//! wiring `EmptyKnnSource` into the retriever's adjacency index MUST
//! leave retrieval output exactly where it was pre-E2, modulo the f32
//! power-iteration arithmetic (which the proptest already pins byte-
//! identically).
//!
//! Scope: We prove the claim at the PPR module boundary (the code
//! consumers will actually hit). Retriever-level integration proof
//! lives in the E2 T3 consumer tests where a repo + HybridAdjacency
//! are wired end-to-end.

use std::collections::BTreeMap;

use mnem_core::id::NodeId;
use mnem_core::index::hybrid::{
    AdjacencyIndex, AuthoredSliceAdjacency, EmptyKnnSource, HybridAdjacency,
};
use mnem_core::ppr::{PprConfig, ppr};

fn nid(i: u8) -> NodeId {
    let mut bytes = [0u8; 16];
    bytes[15] = i;
    NodeId::from_bytes_raw(bytes)
}

#[test]
fn ppr_over_hybrid_with_empty_knn_matches_authored_alone() {
    // 3 authored edges on a 4-node graph.
    let n0 = nid(0);
    let n1 = nid(1);
    let n2 = nid(2);
    let n3 = nid(3);
    let edges = [(n0, n1), (n1, n2), (n2, n3)];
    let authored = AuthoredSliceAdjacency::new(&edges);

    // Snapshot AdjEdge stream from authored-only and from the hybrid-
    // with-empty-knn for a direct structural equality check first. This
    // mirrors the hybrid_adjacency_union test at E0, but scoped to the
    // PPR consumer.
    let authored_clone = AuthoredSliceAdjacency::new(&edges);
    let hybrid = HybridAdjacency::new(authored_clone, EmptyKnnSource);
    let via_authored: Vec<_> = authored.iter_edges().collect();
    let via_hybrid: Vec<_> = hybrid.iter_edges().collect();
    assert_eq!(via_authored, via_hybrid);

    // Run PPR on both sources with the same personalization + config
    // and assert byte-identical f32 outputs. This is the end-to-end
    // contract: feature-flag off (EmptyKnnSource) produces
    // cryptographic-grade identity with the pre-E2 path.
    let mut pers: BTreeMap<NodeId, f32> = BTreeMap::new();
    pers.insert(n0, 1.0);
    let cfg = PprConfig::default();

    let via_authored_adj = AuthoredSliceAdjacency::new(&edges);
    let via_hybrid_adj = HybridAdjacency::new(AuthoredSliceAdjacency::new(&edges), EmptyKnnSource);

    let scores_a = ppr(&via_authored_adj, &pers, cfg);
    let scores_b = ppr(&via_hybrid_adj, &pers, cfg);

    assert_eq!(scores_a.len(), scores_b.len());
    for (id, va) in &scores_a {
        let vb = scores_b[id];
        assert_eq!(
            va.to_bits(),
            vb.to_bits(),
            "byte-identity violated at {id:?}: authored={va} hybrid-empty={vb}"
        );
    }
}

#[test]
fn default_graph_expand_mode_is_decay_preserving() {
    // A retriever-level Smoke check: GraphExpand::default() must have
    // mode == Decay so the zero-impact contract holds at the CLI / HTTP
    // entry points without any caller-side opt-in.
    use mnem_core::retrieve::{GraphExpand, GraphExpandMode};
    let ge = GraphExpand::default();
    assert_eq!(ge.mode, GraphExpandMode::Decay);
}

#[test]
fn with_ppr_builder_sets_mode_and_params() {
    use mnem_core::retrieve::{GraphExpand, GraphExpandMode};
    let ge = GraphExpand::default().with_ppr(0.9, 20, 1e-5);
    match ge.mode {
        GraphExpandMode::Ppr {
            damping,
            max_iter,
            eps,
        } => {
            assert!((damping - 0.9).abs() < 1e-6);
            assert_eq!(max_iter, 20);
            assert!((eps - 1e-5).abs() < 1e-7);
        }
        GraphExpandMode::Decay => panic!("with_ppr failed to set Ppr mode"),
    }
}
