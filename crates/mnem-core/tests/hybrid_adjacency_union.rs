//! Integration tests for [`HybridAdjacency`] (experiment E0).
//!
//! Demonstrates the three contracts the E1/E2/E4 consumers rely on:
//! 1. Union semantics: authored + KNN yield the sum (minus overlaps).
//! 2. Provenance tags: every edge is tagged Authored or Knn.
//! 3. Flag-off byte-identity: with an empty KNN source the hybrid view
//!    is identical to the authored source alone.

use mnem_core::id::NodeId;
use mnem_core::index::hybrid::{
    AdjEdge, AdjacencyIndex, AuthoredSliceAdjacency, EdgeProvenance, EmptyKnnSource,
    HybridAdjacency, KnnEdgeSource,
};

/// Minimal in-memory KNN source: a `Vec<(NodeId, NodeId, f32)>`.
/// Defined here so the test does not depend on `mnem-ann` (which would
/// introduce a dev-time cycle through `mnem-backend-redb` downstream).
struct VecKnn(Vec<(NodeId, NodeId, f32)>);

impl KnnEdgeSource for VecKnn {
    fn iter_knn(&self) -> Box<dyn Iterator<Item = (NodeId, NodeId, f32)> + '_> {
        Box::new(self.0.iter().copied())
    }
    fn knn_len(&self) -> usize {
        self.0.len()
    }
}

fn mk_ids(n: usize) -> Vec<NodeId> {
    (0..n as u8)
        .map(|i| {
            let mut b = [0u8; 16];
            b[15] = i + 1;
            NodeId::from_bytes(&b).unwrap()
        })
        .collect()
}

#[test]
fn authored_plus_knn_yields_eight_distinct_edges() {
    let ids = mk_ids(6);
    let authored_pairs = vec![(ids[0], ids[1]), (ids[1], ids[2]), (ids[2], ids[3])];
    let knn_triples = vec![
        (ids[0], ids[4], 0.9_f32),
        (ids[1], ids[4], 0.8_f32),
        (ids[2], ids[5], 0.7_f32),
        (ids[3], ids[5], 0.6_f32),
        (ids[4], ids[5], 0.5_f32),
    ];

    let authored = AuthoredSliceAdjacency::new(&authored_pairs);
    let knn = VecKnn(knn_triples);
    let hybrid = HybridAdjacency::new(authored, knn);

    let edges: Vec<AdjEdge> = hybrid.iter_edges().collect();

    assert_eq!(edges.len(), 3 + 5, "expected 8 distinct edges");

    let authored_count = edges
        .iter()
        .filter(|e| e.provenance == EdgeProvenance::Authored)
        .count();
    let knn_count = edges
        .iter()
        .filter(|e| e.provenance == EdgeProvenance::Knn)
        .count();
    assert_eq!(authored_count, 3);
    assert_eq!(knn_count, 5);
}

#[test]
fn authored_wins_on_overlap() {
    let ids = mk_ids(4);
    let shared = (ids[0], ids[1]);
    let authored_pairs = vec![shared, (ids[1], ids[2])];
    let knn_triples = vec![
        // Duplicate of authored[0]; must be dropped.
        (shared.0, shared.1, 0.99_f32),
        (ids[2], ids[3], 0.5_f32),
    ];

    let hybrid = HybridAdjacency::new(
        AuthoredSliceAdjacency::new(&authored_pairs),
        VecKnn(knn_triples),
    );
    let edges: Vec<AdjEdge> = hybrid.iter_edges().collect();

    assert_eq!(edges.len(), 3, "overlap must be de-duped");
    // The shared edge is tagged Authored, never Knn.
    let shared_edge = edges
        .iter()
        .find(|e| e.src == shared.0 && e.dst == shared.1)
        .unwrap();
    assert_eq!(shared_edge.provenance, EdgeProvenance::Authored);
    assert!(
        (shared_edge.weight - 1.0).abs() < 1e-6,
        "authored default weight 1.0 survives, not the KNN 0.99"
    );
}

/// Flag-off byte-identity: with an empty KNN source, the hybrid view
/// emits exactly the authored edges in authored order. Using
/// `Vec<AdjEdge>::PartialEq` the comparison is field-wise strict, which
/// in the authored-only slice implementation is effectively bytewise.
#[test]
fn empty_knn_is_byte_identical_to_authored_alone() {
    let ids = mk_ids(5);
    let authored_pairs = vec![(ids[0], ids[1]), (ids[1], ids[2]), (ids[3], ids[4])];

    let authored = AuthoredSliceAdjacency::new(&authored_pairs);
    let direct: Vec<AdjEdge> = authored.iter_edges().collect();

    let hybrid = HybridAdjacency::new(authored.clone(), EmptyKnnSource);
    let via_hybrid: Vec<AdjEdge> = hybrid.iter_edges().collect();

    assert_eq!(direct, via_hybrid);
    assert_eq!(direct.len(), 3);
    assert!(
        direct
            .iter()
            .all(|e| e.provenance == EdgeProvenance::Authored)
    );
}
