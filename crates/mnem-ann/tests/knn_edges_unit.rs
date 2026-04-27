//! Unit tests for the KNN-edge substrate (experiment E0).
//!
//! Builds a tiny deterministic 10-vector fixture, derives the KNN
//! edge set with k=3, and asserts edge count, per-node neighbour
//! correctness under L2, and `(src, dst)`-ASC canonical ordering.

#![cfg(feature = "hnsw")]

use mnem_ann::{
    DistanceMetric, HnswConfig, HnswVectorIndex, KnnEdgeIndex, derive_knn_edges,
    derive_knn_edges_from_vectors,
};
use mnem_core::id::{CODEC_DAG_CBOR, Cid, Multihash, NodeId};

/// Produce a fixed 10-point 3D fixture. Each point sits on one of the
/// three principal axes at a fixed distance from the origin so the
/// nearest-3 neighbour set per point is unambiguous.
fn fixture_10() -> (Vec<NodeId>, Vec<Vec<f32>>) {
    // Use fixed UUID bytes so NodeId ordering is reproducible across runs.
    let ids: Vec<NodeId> = (0..10_u8)
        .map(|i| {
            let mut bytes = [0u8; 16];
            bytes[15] = i + 1;
            NodeId::from_bytes(&bytes).unwrap()
        })
        .collect();
    // 10 points: 4 close to +x axis, 3 close to +y, 3 close to +z.
    let vecs: Vec<Vec<f32>> = vec![
        normalise(vec![1.0, 0.00, 0.00]),
        normalise(vec![0.9, 0.10, 0.00]),
        normalise(vec![0.8, 0.20, 0.00]),
        normalise(vec![0.7, 0.30, 0.00]),
        normalise(vec![0.00, 1.0, 0.00]),
        normalise(vec![0.10, 0.9, 0.00]),
        normalise(vec![0.20, 0.8, 0.00]),
        normalise(vec![0.00, 0.00, 1.0]),
        normalise(vec![0.00, 0.10, 0.9]),
        normalise(vec![0.00, 0.20, 0.8]),
    ];
    (ids, vecs)
}

fn normalise(mut v: Vec<f32>) -> Vec<f32> {
    let mut s = 0.0_f32;
    for x in &v {
        s += x * x;
    }
    if s > 0.0 {
        let inv = s.sqrt().recip();
        for x in &mut v {
            *x *= inv;
        }
    }
    v
}

fn demo_root_cid() -> Cid {
    Cid::new(CODEC_DAG_CBOR, Multihash::sha2_256(b"fixture-root"))
}

#[test]
fn edge_count_is_n_times_k() {
    let (ids, vecs) = fixture_10();
    let edges = derive_knn_edges_from_vectors(&ids, &vecs, 3, DistanceMetric::Cosine);
    assert_eq!(edges.len(), 10 * 3, "expected 30 edges");
}

#[test]
fn no_self_loops() {
    let (ids, vecs) = fixture_10();
    let edges = derive_knn_edges_from_vectors(&ids, &vecs, 3, DistanceMetric::Cosine);
    for e in &edges {
        assert_ne!(e.src, e.dst, "self-loop present: {e:?}");
    }
}

#[test]
fn edges_sorted_by_src_then_dst() {
    let (ids, vecs) = fixture_10();
    let edges = derive_knn_edges_from_vectors(&ids, &vecs, 3, DistanceMetric::Cosine);
    for pair in edges.windows(2) {
        let a = &pair[0];
        let b = &pair[1];
        let ord = a.src.cmp(&b.src).then_with(|| a.dst.cmp(&b.dst));
        assert!(
            ord != std::cmp::Ordering::Greater,
            "edges out of order: {a:?} then {b:?}"
        );
    }
}

#[test]
fn per_node_neighbours_are_the_three_nearest() {
    // For the x-cluster at ids[0..4], the three nearest neighbours of
    // each are the OTHER three x-cluster members (all three x-cluster
    // points share the x-axis and outrank the y/z clusters).
    let (ids, vecs) = fixture_10();
    let edges = derive_knn_edges_from_vectors(&ids, &vecs, 3, DistanceMetric::Cosine);

    let x_cluster: std::collections::HashSet<NodeId> = ids[0..4].iter().copied().collect();
    for src_id in &ids[0..4] {
        let out: Vec<NodeId> = edges
            .iter()
            .filter(|e| e.src == *src_id)
            .map(|e| e.dst)
            .collect();
        assert_eq!(out.len(), 3);
        for dst in &out {
            assert!(
                x_cluster.contains(dst),
                "x-cluster source {src_id:?} picked non-x-cluster neighbour {dst:?}"
            );
        }
    }
}

#[test]
fn cid_stable_across_two_derivations() {
    let (ids, vecs) = fixture_10();
    let e1 = derive_knn_edges_from_vectors(&ids, &vecs, 3, DistanceMetric::Cosine);
    let e2 = derive_knn_edges_from_vectors(&ids, &vecs, 3, DistanceMetric::Cosine);
    let idx1 = KnnEdgeIndex {
        root_cid: demo_root_cid(),
        k: 3,
        metric: DistanceMetric::Cosine,
        edges: e1,
    };
    let idx2 = KnnEdgeIndex {
        root_cid: demo_root_cid(),
        k: 3,
        metric: DistanceMetric::Cosine,
        edges: e2,
    };
    assert_eq!(idx1.compute_cid().unwrap(), idx2.compute_cid().unwrap());
}

#[test]
fn hnsw_backed_derivation_matches_pure_kernel() {
    let (ids, vecs) = fixture_10();
    let cfg = HnswConfig::default();
    let hnsw = HnswVectorIndex::from_parts_for_test("m", 3, ids.clone(), vecs.clone(), &cfg);
    let derived = derive_knn_edges(&hnsw, 3, demo_root_cid());
    let expected_edges = derive_knn_edges_from_vectors(&ids, &vecs, 3, DistanceMetric::Cosine);
    assert_eq!(derived.edges, expected_edges);
    assert_eq!(derived.k, 3);
    assert_eq!(derived.metric, DistanceMetric::Cosine);
}
