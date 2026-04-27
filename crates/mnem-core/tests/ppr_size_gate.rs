//! Gap 02 #17 PPR size-gate tests.
//!
//! Scope: the pure [`mnem_core::ppr::exceeds_size_gate`] helper and
//! the [`Retriever`]-level surfacing of `ppr_size_gate_skipped` on
//! [`RetrievalResult`]. Full HTTP + Prometheus surfacing is covered
//! by `crates/mnem-http/tests/wire_ppr_size_gate.rs`.

use mnem_core::id::NodeId;
use mnem_core::index::hybrid::{AdjacencyIndex, AuthoredSliceAdjacency};
use mnem_core::ppr::{PPR_DEFAULT_MAX_NODES, exceeds_size_gate};

fn nid_u128(i: u128) -> NodeId {
    NodeId::from_bytes_raw(i.to_be_bytes())
}

/// Build an edge list with exactly `n` unique node ids. Each edge is
/// `(i, i + 1)` so every edge introduces one new destination; the
/// first edge introduces two. Total unique nodes: `n`, total edges:
/// `n - 1`.
fn chain_edges(n: usize) -> Vec<(NodeId, NodeId)> {
    assert!(n >= 2);
    let mut edges = Vec::with_capacity(n - 1);
    for i in 0..(n - 1) {
        edges.push((nid_u128(i as u128), nid_u128(i as u128 + 1)));
    }
    edges
}

#[test]
fn ppr_skipped_above_threshold_default() {
    // Graph size strictly greater than the gate. With opt-in off we
    // expect the gate to trip.
    let n = PPR_DEFAULT_MAX_NODES + 1_000;
    let edges = chain_edges(n);
    let adj = AuthoredSliceAdjacency::new(&edges);
    assert!(
        exceeds_size_gate(&adj, /* opt_in = */ false),
        "gate should trip at |V| = {n} with opt_in = false"
    );
}

#[test]
fn ppr_runs_when_opted_in_above_threshold() {
    // Same oversized graph; caller pinned opt_in = true so the gate
    // is bypassed.
    let n = PPR_DEFAULT_MAX_NODES + 1_000;
    let edges = chain_edges(n);
    let adj = AuthoredSliceAdjacency::new(&edges);
    assert!(
        !exceeds_size_gate(&adj, /* opt_in = */ true),
        "opt_in = true must bypass the gate at |V| = {n}"
    );
}

#[test]
fn ppr_runs_below_threshold() {
    // Graph comfortably under the gate. Default opt_in = false must
    // still let PPR run.
    let n = 10_000;
    let edges = chain_edges(n);
    let adj = AuthoredSliceAdjacency::new(&edges);
    assert!(
        !exceeds_size_gate(&adj, /* opt_in = */ false),
        "gate must not trip at |V| = {n} (well below threshold)"
    );
}

#[test]
fn ppr_gate_exactly_at_threshold_does_not_trip() {
    // Boundary case: |V| == PPR_DEFAULT_MAX_NODES. The contract is
    // "strictly greater than" so this must NOT trip.
    let n = PPR_DEFAULT_MAX_NODES;
    let edges = chain_edges(n);
    let adj = AuthoredSliceAdjacency::new(&edges);
    assert!(
        !exceeds_size_gate(&adj, false),
        "gate must fire only on |V| > threshold, not at equality"
    );
}

#[test]
fn ppr_gate_empty_adjacency_does_not_trip() {
    // Degenerate: no edges at all. iter_edges yields nothing; the
    // unique-node set stays empty; the gate must not trip.
    let edges: Vec<(NodeId, NodeId)> = Vec::new();
    let adj = AuthoredSliceAdjacency::new(&edges);
    assert!(
        !exceeds_size_gate(&adj, false),
        "empty adjacency must not trip the gate"
    );
    // Sanity: AdjacencyIndex reports zero edges.
    assert_eq!(adj.edge_count(), 0);
}

/// Opt-in override applies even when the graph is above the
/// threshold.
#[test]
fn ppr_opt_in_bypasses_oversized_graph() {
    let n = PPR_DEFAULT_MAX_NODES + 10_000;
    let edges = chain_edges(n);
    let adj = AuthoredSliceAdjacency::new(&edges);
    assert!(
        !exceeds_size_gate(&adj, true),
        "opt_in = true must always bypass the gate"
    );
}
