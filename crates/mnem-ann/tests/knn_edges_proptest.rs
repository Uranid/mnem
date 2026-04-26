//! Property-based tests for the KNN-edge substrate (experiment E0).
//!
//! The central property: given an arbitrary vector set of ≤50 dim-8
//! vectors and an arbitrary insertion order, two separate derivations
//! produce byte-identical edge CBOR. This protects the content-address
//! stability contract E1/E2/E4 rely on.

#![cfg(feature = "hnsw")]

use mnem_ann::{DistanceMetric, KnnEdgeIndex, derive_knn_edges_from_vectors};
use mnem_core::codec::to_canonical_bytes;
use mnem_core::id::{CODEC_DAG_CBOR, Cid, Multihash, NodeId};
use proptest::prelude::*;

fn demo_root_cid() -> Cid {
    Cid::new(CODEC_DAG_CBOR, Multihash::sha2_256(b"proptest-root"))
}

fn arb_vec_dim8() -> impl Strategy<Value = Vec<f32>> {
    // Bound each coord to avoid NaN/Inf paths and keep the norm finite.
    proptest::collection::vec(-10.0_f32..10.0_f32, 8..=8)
}

fn arb_nodeid() -> impl Strategy<Value = NodeId> {
    any::<[u8; 16]>().prop_map(|b| NodeId::from_bytes(&b).unwrap())
}

fn arb_corpus() -> impl Strategy<Value = Vec<(NodeId, Vec<f32>)>> {
    // 2..=50 vectors; every NodeId generated independently (prop will
    // retry the rare duplicate and the impl handles duplicates anyway).
    proptest::collection::vec((arb_nodeid(), arb_vec_dim8()), 2..=50)
}

fn l2_normalise(mut v: Vec<f32>) -> Vec<f32> {
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

fn build_idx(corpus: &[(NodeId, Vec<f32>)], k: u32) -> KnnEdgeIndex {
    let ids: Vec<NodeId> = corpus.iter().map(|(i, _)| *i).collect();
    let vecs: Vec<Vec<f32>> = corpus
        .iter()
        .map(|(_, v)| l2_normalise(v.clone()))
        .collect();
    let edges = derive_knn_edges_from_vectors(&ids, &vecs, k, DistanceMetric::Cosine);
    KnnEdgeIndex {
        root_cid: demo_root_cid(),
        k,
        metric: DistanceMetric::Cosine,
        edges,
    }
}

proptest! {
    // Keep case count modest so the Windows CI runtime stays predictable.
    #![proptest_config(ProptestConfig { cases: 48, .. ProptestConfig::default() })]

    #[test]
    fn edges_bytes_are_byte_identical_across_redrive(
        corpus in arb_corpus(),
        k in 1_u32..=8,
    ) {
        let idx1 = build_idx(&corpus, k);
        let idx2 = build_idx(&corpus, k);
        let b1 = to_canonical_bytes(&idx1).unwrap();
        let b2 = to_canonical_bytes(&idx2).unwrap();
        prop_assert_eq!(b1, b2);
    }

    #[test]
    fn compute_cid_stable_across_redrive(
        corpus in arb_corpus(),
        k in 1_u32..=8,
    ) {
        let idx1 = build_idx(&corpus, k);
        let idx2 = build_idx(&corpus, k);
        prop_assert_eq!(idx1.compute_cid().unwrap(), idx2.compute_cid().unwrap());
    }

    #[test]
    fn permuted_insertion_yields_same_cid(
        corpus in arb_corpus(),
        k in 1_u32..=8,
        rot in 0_usize..50,
    ) {
        // Rotating the corpus preserves the multi-set of (id, vec) but
        // changes insertion order. Because the derivation sorts by
        // (src, dst) before encoding, the CID must be invariant.
        let n = corpus.len();
        let r = rot % n.max(1);
        let mut permuted = corpus.clone();
        permuted.rotate_left(r);

        let a = build_idx(&corpus, k);
        let b = build_idx(&permuted, k);
        prop_assert_eq!(a.compute_cid().unwrap(), b.compute_cid().unwrap());
    }

    #[test]
    fn edge_count_bounded_by_n_times_k(
        corpus in arb_corpus(),
        k in 1_u32..=8,
    ) {
        let idx = build_idx(&corpus, k);
        let n = corpus.len();
        // With a de-duped corpus each source has exactly min(k, n-1)
        // outgoing edges. With possible duplicate NodeIds (rare in
        // proptest) the upper bound n*k still holds.
        prop_assert!(idx.edges.len() <= n * (k as usize));
    }
}
