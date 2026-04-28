//! Byte-identity contract: [`ppr_with_matrix`] must produce the same
//! output as [`ppr`] when given a matrix freshly built from the same
//! adjacency.
//!
//! C3 FIX-1: the HTTP layer caches the [`SparseTransition`] per op-id
//! and calls `ppr_with_matrix` across requests. The cached-vs-rebuilt
//! paths MUST produce byte-identical outputs for the cache to be safe
//! to turn on by default (no recall drift, no bench-gaming).

use std::collections::BTreeMap;

use mnem_core::id::NodeId;
use mnem_core::index::hybrid::AuthoredSliceAdjacency;
use mnem_core::ppr::{PprConfig, ppr, ppr_with_matrix, sparse_transition_matrix};

fn nid(i: u8) -> NodeId {
    let mut bytes = [0u8; 16];
    bytes[15] = i;
    NodeId::from_bytes_raw(bytes)
}

/// Minimum contract: byte-identical score vectors for the two entry
/// points on a tiny graph.
#[test]
fn ppr_with_matrix_matches_ppr_on_small_graph() {
    let n0 = nid(0);
    let n1 = nid(1);
    let n2 = nid(2);
    let n3 = nid(3);
    let edges = [(n0, n1), (n0, n2), (n1, n0), (n2, n3), (n3, n0)];
    let adj = AuthoredSliceAdjacency::new(&edges);

    let mut pers: BTreeMap<NodeId, f32> = BTreeMap::new();
    pers.insert(n0, 1.0);
    let cfg = PprConfig::default();

    let direct = ppr(&adj, &pers, cfg);
    let m = sparse_transition_matrix(&adj);
    let via_matrix = ppr_with_matrix(&m, &pers, cfg);

    assert_eq!(direct.len(), via_matrix.len());
    for (id, v_direct) in &direct {
        let v_matrix = via_matrix[id];
        // Byte-identity: both paths run the same power iteration on
        // the same CSR layout, so the outputs must be exactly equal.
        assert_eq!(
            v_direct.to_bits(),
            v_matrix.to_bits(),
            "byte-drift at {id:?}: direct={v_direct} via_matrix={v_matrix}"
        );
    }
}

/// Cache-reuse scenario: build the matrix ONCE, run PPR multiple times
/// with different personalization vectors. Every run must match the
/// equivalent direct `ppr()` call byte-for-byte.
#[test]
fn ppr_with_matrix_reuses_matrix_across_personalizations() {
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
    let m = sparse_transition_matrix(&adj);
    let cfg = PprConfig::default();

    let personalizations = [
        vec![(n0, 1.0_f32)],
        vec![(n2, 1.0_f32)],
        vec![(n1, 0.3_f32), (n4, 0.7_f32)],
    ];
    for pers_pairs in &personalizations {
        let pers: BTreeMap<NodeId, f32> = pers_pairs.iter().copied().collect();
        let direct = ppr(&adj, &pers, cfg);
        let via_matrix = ppr_with_matrix(&m, &pers, cfg);
        for (id, v_direct) in &direct {
            assert_eq!(
                v_direct.to_bits(),
                via_matrix[id].to_bits(),
                "drift for pers={pers_pairs:?} at {id:?}"
            );
        }
    }
}

/// Tight-convergence vs default-eps: with `eps=1e-10` and many iters
/// vs the workload's actual `eps=1e-6` default, the two runs must
/// agree within a bound proportional to `eps / (1 - damping)`. This
/// pins the L1 early-stop as a principled (Page/Brin 1998) termination
/// rather than an arbitrary short run.
#[test]
fn l1_early_stop_agrees_with_long_run() {
    let edges: Vec<(NodeId, NodeId)> = (0..10u8)
        .flat_map(|i| [(nid(i), nid((i + 1) % 10)), (nid(i), nid((i + 3) % 10))])
        .collect();
    let adj = AuthoredSliceAdjacency::new(&edges);
    let m = sparse_transition_matrix(&adj);

    let mut pers: BTreeMap<NodeId, f32> = BTreeMap::new();
    pers.insert(nid(0), 1.0);

    let damping = 0.85_f32;
    let default_eps = 1e-6_f32;
    let tight = ppr_with_matrix(
        &m,
        &pers,
        PprConfig {
            damping,
            max_iter: 500,
            eps: 1e-10,
        },
    );
    let early = ppr_with_matrix(
        &m,
        &pers,
        PprConfig {
            damping,
            max_iter: 500,
            eps: default_eps,
        },
    );

    // L1 bound: PPR is a contraction with rate `damping`, so the
    // distance from the iterate that triggered the eps-stop to the
    // fixed point is at most eps / (1 - damping). Apply the same
    // bound to the gap between the two early-stops; the 10x slack
    // absorbs the defensive L1 renorm and f32 rounding.
    let bound = 10.0 * default_eps / (1.0 - damping);
    let mut l1 = 0.0_f32;
    for (id, v_tight) in &tight {
        let v_early = early.get(id).copied().unwrap_or(0.0);
        l1 += (v_tight - v_early).abs();
    }
    assert!(
        l1 < bound,
        "early-stop vs tight L1 distance {l1} exceeds bound {bound}"
    );
}
