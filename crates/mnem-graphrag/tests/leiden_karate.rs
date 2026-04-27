//! Zachary's Karate Club ground-truth test.
//!
//! 34-node, 78-edge benchmark graph. Classic modularity ground
//! truth: at least 2 communities, node 0 ("Mr Hi") and node 33
//! ("John A") separated. Modularity > 0.35.

use mnem_core::id::{NodeId, StableId};
use mnem_core::index::AuthoredSliceAdjacency;
use mnem_graphrag::compute_communities;

/// The 78 undirected edges of Zachary's Karate Club (1977). Standard
/// edge list from the Pajek/NetworkX distribution, 0-indexed.
const KARATE_EDGES: &[(usize, usize)] = &[
    (0, 1),
    (0, 2),
    (0, 3),
    (0, 4),
    (0, 5),
    (0, 6),
    (0, 7),
    (0, 8),
    (0, 10),
    (0, 11),
    (0, 12),
    (0, 13),
    (0, 17),
    (0, 19),
    (0, 21),
    (0, 31),
    (1, 2),
    (1, 3),
    (1, 7),
    (1, 13),
    (1, 17),
    (1, 19),
    (1, 21),
    (1, 30),
    (2, 3),
    (2, 7),
    (2, 8),
    (2, 9),
    (2, 13),
    (2, 27),
    (2, 28),
    (2, 32),
    (3, 7),
    (3, 12),
    (3, 13),
    (4, 6),
    (4, 10),
    (5, 6),
    (5, 10),
    (5, 16),
    (6, 16),
    (8, 30),
    (8, 32),
    (8, 33),
    (9, 33),
    (13, 33),
    (14, 32),
    (14, 33),
    (15, 32),
    (15, 33),
    (18, 32),
    (18, 33),
    (19, 33),
    (20, 32),
    (20, 33),
    (22, 32),
    (22, 33),
    (23, 25),
    (23, 27),
    (23, 29),
    (23, 32),
    (23, 33),
    (24, 25),
    (24, 27),
    (24, 31),
    (25, 31),
    (26, 29),
    (26, 33),
    (27, 33),
    (28, 31),
    (28, 33),
    (29, 32),
    (29, 33),
    (30, 32),
    (30, 33),
    (31, 32),
    (31, 33),
    (32, 33),
];

fn make_karate_ids() -> [NodeId; 34] {
    let mut ids = [StableId::from_bytes(&[0_u8; 16]).unwrap(); 34];
    for (i, id) in ids.iter_mut().enumerate() {
        // Embed i in byte 15 so ids are ordered and stable.
        let mut b = [0_u8; 16];
        b[15] = i as u8;
        *id = StableId::from_bytes(&b).unwrap();
    }
    ids
}

#[test]
fn karate_has_multiple_communities_and_good_modularity() {
    let ids = make_karate_ids();
    let edges: Vec<(NodeId, NodeId)> = KARATE_EDGES
        .iter()
        .map(|&(a, b)| (ids[a], ids[b]))
        .collect();
    let adj = AuthoredSliceAdjacency::new(&edges);

    let assignment = compute_communities(&adj, 42);

    // Ground-truth separation: Mr Hi (0) vs John A (33).
    let c0 = assignment
        .community_of(ids[0])
        .expect("node 0 in some community");
    let c33 = assignment
        .community_of(ids[33])
        .expect("node 33 in some community");
    assert_ne!(
        c0, c33,
        "Karate club: node 0 (Mr Hi) and node 33 (John A) must land in different communities"
    );

    assert!(
        assignment.community_count() >= 2,
        "Expected >= 2 communities, got {}",
        assignment.community_count()
    );

    assert!(
        assignment.modularity > 0.35,
        "Modularity {} < 0.35 threshold (Newman 2006 ground truth)",
        assignment.modularity
    );
}

#[test]
fn karate_determinism_same_seed_same_partition() {
    let ids = make_karate_ids();
    let edges: Vec<(NodeId, NodeId)> = KARATE_EDGES
        .iter()
        .map(|&(a, b)| (ids[a], ids[b]))
        .collect();
    let adj = AuthoredSliceAdjacency::new(&edges);

    let a = compute_communities(&adj, 42);
    let b = compute_communities(&adj, 42);
    assert_eq!(a.content_cid(), b.content_cid());
    assert!((a.modularity - b.modularity).abs() < 1e-6);
}
