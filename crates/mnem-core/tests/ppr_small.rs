//! PPR unit test on a tiny known-result graph.
//!
//! E2 turn T2. Validates that the power-iteration PPR output matches a
//! closed-form reference on a 5-node directed graph.
//!
//! ## Reference
//!
//! Graph (directed, strongly connected hub-and-spoke on 5 nodes):
//!
//! ```text
//!   0 -> 1, 0 -> 2
//!   1 -> 0
//!   2 -> 0, 2 -> 3
//!   3 -> 0, 3 -> 4
//!   4 -> 0
//! ```
//!
//! Personalization = `{0: 1.0}`, damping = `0.85`, max_iter = `200`,
//! eps = `1e-9`. A closed-form reference was computed by running the
//! exact same power-iteration recurrence in f64 Python (same teleport-
//! on-dangling semantics, same L1 renormalisation per step, same
//! damping). The reference converged to:
//!
//! ```text
//!   n0 ≈ 0.4745
//!   n1 ≈ 0.2017
//!   n2 ≈ 0.2017
//!   n3 ≈ 0.0857
//!   n4 ≈ 0.0364
//! ```
//!
//! f32 vs f64 drift after 200 iterations is <= 5e-3 absolute; the test
//! checks a 1e-2 tolerance to be safe. The qualitative invariants
//! (seed wins, symmetric siblings n1/n2 tie, downstream chain loses
//! mass monotonically) also hold exactly.

use std::collections::BTreeMap;

use mnem_core::id::NodeId;
use mnem_core::index::hybrid::AuthoredSliceAdjacency;
use mnem_core::ppr::{PprConfig, ppr};

/// Build a deterministic fixed-bytes NodeId for test stability.
fn nid(i: u8) -> NodeId {
    let mut bytes = [0u8; 16];
    bytes[15] = i;
    NodeId::from_bytes_raw(bytes)
}

#[test]
fn ppr_matches_reference_distribution() {
    let n0 = nid(0);
    let n1 = nid(1);
    let n2 = nid(2);
    let n3 = nid(3);
    let n4 = nid(4);
    let edges = [
        (n0, n1),
        (n0, n2),
        (n1, n0),
        (n2, n0),
        (n2, n3),
        (n3, n0),
        (n3, n4),
        (n4, n0),
    ];
    let adj = AuthoredSliceAdjacency::new(&edges);
    let mut pers: BTreeMap<NodeId, f32> = BTreeMap::new();
    pers.insert(n0, 1.0);
    let cfg = PprConfig {
        damping: 0.85,
        max_iter: 200,
        eps: 1e-9,
    };
    let scores = ppr(&adj, &pers, cfg);

    // 1. Every graph node appears.
    assert_eq!(scores.len(), 5);

    // 2. Mass conservation: L1 sum ≈ 1.0.
    let total: f32 = scores.values().sum();
    assert!((total - 1.0).abs() < 1e-4, "L1 mass not conserved: {total}");

    // 3. Seed dominates every other node.
    let s0 = scores[&n0];
    let s1 = scores[&n1];
    let s2 = scores[&n2];
    let s3 = scores[&n3];
    let s4 = scores[&n4];
    assert!(s0 > s1, "seed n0 ({s0}) should outrank n1 ({s1})");
    assert!(s0 > s2, "seed n0 ({s0}) should outrank n2 ({s2})");
    assert!(s0 > s3);
    assert!(s0 > s4);

    // 4. Symmetric siblings n1 and n2 (both direct out-neighbors of n0
    //    with a single back-edge) would tie exactly except that n2
    //    also leaks mass forward into n3/n4 and so MUST rank below n1
    //    - or tie within f32 rounding, which the reference says they
    //    do to 4 decimals.
    assert!(
        s1 >= s2 - 1e-3,
        "n1 ({s1}) should not rank meaningfully below n2 ({s2})"
    );

    // 5. Downstream chain loses mass monotonically.
    assert!(
        s3 > s4,
        "chain head n3 ({s3}) should outrank tail n4 ({s4})"
    );

    // 6. Closed-form match within 1e-2 absolute tolerance.
    let tol = 1e-2f32;
    let expected = [
        (n0, 0.4745f32),
        (n1, 0.2017),
        (n2, 0.2017),
        (n3, 0.0857),
        (n4, 0.0364),
    ];
    for (id, exp) in expected {
        let got = scores[&id];
        assert!(
            (got - exp).abs() < tol,
            "node {id:?}: PPR {got} deviates from reference {exp} by > {tol}"
        );
    }
}

#[test]
fn empty_graph_returns_empty() {
    let edges: [(NodeId, NodeId); 0] = [];
    let adj = AuthoredSliceAdjacency::new(&edges);
    let pers: BTreeMap<NodeId, f32> = BTreeMap::new();
    let scores = ppr(&adj, &pers, PprConfig::default());
    assert!(scores.is_empty());
}

#[test]
fn zero_personalization_falls_back_to_uniform() {
    let n0 = nid(0);
    let n1 = nid(1);
    let edges = [(n0, n1), (n1, n0)];
    let adj = AuthoredSliceAdjacency::new(&edges);
    let pers: BTreeMap<NodeId, f32> = BTreeMap::new();
    let scores = ppr(&adj, &pers, PprConfig::default());
    assert_eq!(scores.len(), 2);
    // Symmetric graph with uniform teleport -> ~0.5 each.
    for v in scores.values() {
        assert!((v - 0.5).abs() < 1e-3, "uniform fallback broken, got {v}");
    }
}

#[test]
fn deterministic_repeated_runs_byte_identical() {
    let n0 = nid(0);
    let n1 = nid(1);
    let n2 = nid(2);
    let edges = [(n0, n1), (n1, n2), (n2, n0)];
    let adj = AuthoredSliceAdjacency::new(&edges);
    let mut pers: BTreeMap<NodeId, f32> = BTreeMap::new();
    pers.insert(n0, 1.0);
    let cfg = PprConfig::default();
    let a = ppr(&adj, &pers, cfg);
    let b = ppr(&adj, &pers, cfg);
    // Compare bit-exact via f32::to_bits so +0/-0 and NaN discussions
    // don't muddy the determinism claim (there should be no NaN here
    // because the graph is dense enough to keep the mat-vec defined).
    for (id, va) in &a {
        let vb = b[id];
        assert_eq!(
            va.to_bits(),
            vb.to_bits(),
            "non-deterministic PPR output at {id:?}"
        );
    }
}
