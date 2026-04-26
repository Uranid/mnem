//! Q0 `put_trusted` migration safety net.
//!
//! A2 (security-hardening) landed unconditional CID recomputation on
//! every `Blockstore::put`. Q0 introduces `Blockstore::put_trusted` as
//! an audited fast-path skipping the recompute for in-tree callers
//! that just produced `(bytes, cid)` from `hash_to_cid`. This file is
//! the integration-level safety net: if the `put_trusted` migration
//! silently corrupts any block along the commit path (commit, view,
//! op, node tree, edge tree, index set, adjacency buckets), the CIDs
//! below diverge from the pre-Q0 values and the assertions fail.
//!
//! Complements the lib-level
//! `deterministic_commit_opts_yield_identical_commit_cid` test in
//! `repo::transaction::tests`, which covers the same property for
//! the commit head only. This file additionally pins the node root,
//! edge root, and indexes CID so a regression isolated to one
//! sub-tree (Prolly leaf/internal put_trusted, adjacency-bucket
//! put_trusted, index-set put_trusted) still trips a loud failure,
//! even if the commit header itself is coincidentally the same.
//!
//! Note on scope: we do NOT pin the View CID or Operation CID here
//! because both depend on `ReadonlyRepo::init`'s wall-clock time
//! (the root operation carries `now_micros()`, which varies per
//! run). Pinning those would require also pinning init-time, which
//! is a broader determinism story than Q0 needs.

use std::sync::Arc;

use ipld_core::ipld::Ipld;
use mnem_core::id::{ChangeId, Cid, EdgeId, NodeId};
use mnem_core::objects::{Edge, Node};
use mnem_core::repo::{CommitOptions, ReadonlyRepo};
use mnem_core::store::{Blockstore, MemoryBlockstore, MemoryOpHeadsStore, OpHeadsStore};

/// Fixed input set driving byte-stable CIDs:
/// - two nodes with pinned `NodeId`s + deterministic props,
/// - one edge with a pinned `EdgeId`,
/// - pinned `ChangeId` + `time_micros` on commit_opts.
fn build_commit() -> (Cid, Cid, Cid, Cid) {
    let bs: Arc<dyn Blockstore> = Arc::new(MemoryBlockstore::new());
    let ohs: Arc<dyn OpHeadsStore> = Arc::new(MemoryOpHeadsStore::new());
    let repo = ReadonlyRepo::init(bs, ohs).unwrap();

    let alice_id = NodeId::from_bytes_raw([0xaa; 16]);
    let bob_id = NodeId::from_bytes_raw([0xbb; 16]);
    let edge_id = EdgeId::from_bytes_raw([0xcc; 16]);

    let alice = Node::new(alice_id, "Person")
        .with_prop("name", Ipld::String("alice".into()))
        .with_prop("age", Ipld::Integer(30));
    let bob = Node::new(bob_id, "Person")
        .with_prop("name", Ipld::String("bob".into()))
        .with_prop("age", Ipld::Integer(31));
    let knows = Edge::new(edge_id, "knows", alice_id, bob_id);

    let mut tx = repo.start_transaction();
    tx.add_node(&alice).unwrap();
    tx.add_node(&bob).unwrap();
    tx.add_edge(&knows).unwrap();

    let fixed_time: u64 = 1_700_000_000_000_000;
    let fixed_change = ChangeId::from_bytes_raw([0x11; 16]);
    let new_repo = tx
        .commit_opts(
            CommitOptions::new("alice", "q0 safety net")
                .with_time_micros(fixed_time)
                .with_change_id(fixed_change),
        )
        .unwrap();

    let head = new_repo
        .view()
        .heads
        .first()
        .expect("one head after commit")
        .clone();
    let commit = new_repo.head_commit().unwrap();
    let nodes_root = commit.nodes.clone();
    let edges_root = commit.edges.clone();
    let indexes = commit.indexes.clone().expect("indexes present");
    (head, nodes_root, edges_root, indexes)
}

#[test]
fn q0_put_trusted_preserves_commit_side_dag_cids() {
    // Build the same commit twice on disjoint repos. Every CID on the
    // commit-side DAG must be byte-identical across runs - this is
    // the cross-process-determinism invariant, plus our migration
    // safety net: if `put_trusted` ever silently corrupts any of
    // these sub-trees, the second run produces different CIDs and
    // this test fails loudly.
    let a = build_commit();
    let b = build_commit();
    assert_eq!(a.0, b.0, "commit head CID diverged across fresh repos");
    assert_eq!(a.1, b.1, "nodes-root CID diverged across fresh repos");
    assert_eq!(a.2, b.2, "edges-root CID diverged across fresh repos");
    assert_eq!(a.3, b.3, "indexes-root CID diverged across fresh repos");
}
