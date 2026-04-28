//! Modularity must improve over a trivial all-singletons baseline.

use mnem_core::id::{NodeId, StableId};
use mnem_core::index::AuthoredSliceAdjacency;
use mnem_graphrag::compute_communities;

fn nid(i: u8) -> NodeId {
    let mut b = [0_u8; 16];
    b[15] = i;
    StableId::from_bytes(&b).unwrap()
}

#[test]
fn two_cliques_beats_singletons() {
    // Two K4 cliques joined by a single bridge edge - classic
    // two-community fixture. Leiden should find modularity well
    // above the singleton-partition score (which is negative for any
    // graph with at least one edge).
    let n: Vec<NodeId> = (0_u8..8).map(nid).collect();
    let edges = vec![
        // K4 on {0,1,2,3}
        (n[0], n[1]),
        (n[0], n[2]),
        (n[0], n[3]),
        (n[1], n[2]),
        (n[1], n[3]),
        (n[2], n[3]),
        // K4 on {4,5,6,7}
        (n[4], n[5]),
        (n[4], n[6]),
        (n[4], n[7]),
        (n[5], n[6]),
        (n[5], n[7]),
        (n[6], n[7]),
        // bridge
        (n[3], n[4]),
    ];
    let adj = AuthoredSliceAdjacency::new(&edges);
    let a = compute_communities(&adj, 0);

    assert!(
        a.community_count() >= 2,
        "Expected two cliques to split into >= 2 communities"
    );
    // Singleton modularity on a connected graph with m edges and all
    // degree > 0 is negative; any non-trivial partition must beat 0.
    assert!(
        a.modularity > 0.0,
        "Leiden partition modularity {} should exceed singleton baseline (0)",
        a.modularity
    );
    // For two K4 + 1 bridge, optimal Q = (6/13 + 6/13) - 2*(7/26)^2
    //   = 12/13 - 2*(49/676) = 12/13 - 98/676 ~= 0.923 - 0.145 = 0.778
    // Practical threshold: > 0.35 is more than comfortable.
    assert!(
        a.modularity > 0.35,
        "Two-cliques modularity {} unexpectedly low",
        a.modularity
    );
}

#[test]
fn empty_graph_is_empty_assignment() {
    let edges: Vec<(NodeId, NodeId)> = Vec::new();
    let adj = AuthoredSliceAdjacency::new(&edges);
    let a = compute_communities(&adj, 7);
    assert_eq!(a.map.len(), 0);
    assert!(a.modularity.abs() < f32::EPSILON);
    assert_eq!(a.seed, 7);
}
