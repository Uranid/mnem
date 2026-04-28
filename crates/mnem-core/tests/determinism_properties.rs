//! Proptest-based determinism properties for mnem-core.
//!
//! # Scope
//!
//! Complements the three fixed-case golden-vector tests
//! (`dual_adjacency_commit_cid_is_order_independent`,
//! `incremental_and_full_index_build_produce_identical_index_set`,
//! `incremental_and_full_preserve_both_direction_adjacency_cids`) by
//! exercising the same invariants across *arbitrary* graph shapes. The
//! golden tests pin hard-coded CIDs (regression guard for the
//! serde/codec boundary itself); the properties here show the
//! invariant holds over the whole input space.
//!
//! # Properties
//!
//! - **P1** add-order permutation: a shuffled op sequence produces the
//!   same commit CID.
//! - **P2** incremental vs full: splitting N ops into K batches
//!   (K in `1..=8`) yields the same final `IndexSet` CID.
//! - **P3** retrieve determinism: the same repo + the same query yields
//!   byte-identical serialized results.
//! - **P5** tombstone-idempotent: `[Tombstone(n)]` and
//!   `[Tombstone(n), Tombstone(n)]` produce the same commit CID.
//!
//! P4 (CAR round-trip) lives in `mnem-transport`.
//!
//! # Runtime budget
//!
//! `ProptestConfig { cases: 64, max_shrink_iters: 1000, .. }` per
//! property, targeting <30s each on a dev laptop. The `ci-slow` feature
//! gates 256-case variants that run in nightly CI.

#![allow(clippy::needless_pass_by_value)]

use std::sync::Arc;

use ipld_core::ipld::Ipld;
use proptest::prelude::*;

use mnem_core::id::{ChangeId, Cid, EdgeId, NodeId};
use mnem_core::objects::{Edge, Node};
use mnem_core::repo::{CommitOptions, ReadonlyRepo};
use mnem_core::store::{Blockstore, MemoryBlockstore, MemoryOpHeadsStore, OpHeadsStore};

// ============================================================
// Strategies
// ============================================================

/// Generate a 16-byte [`NodeId`]. 16 bytes is the stable raw form used
/// by `NodeId::from_bytes_raw`; proptest's rng keeps the shrink tree
/// deterministic.
fn arb_node_id() -> impl Strategy<Value = NodeId> {
    any::<[u8; 16]>().prop_map(NodeId::from_bytes_raw)
}

/// Generate a 16-byte [`EdgeId`]. Same rationale as `arb_node_id`.
fn arb_edge_id() -> impl Strategy<Value = EdgeId> {
    any::<[u8; 16]>().prop_map(EdgeId::from_bytes_raw)
}

/// Generate one of four stable node-type labels. Small cardinality
/// keeps shrinking fast; the set is large enough to exercise the
/// type-index bucket.
fn arb_ntype() -> impl Strategy<Value = String> {
    prop_oneof![
        Just("Person".to_string()),
        Just("Place".to_string()),
        Just("Event".to_string()),
        Just("Concept".to_string()),
    ]
}

/// Generate one of four edge-label strings.
fn arb_rel() -> impl Strategy<Value = String> {
    prop_oneof![
        Just("knows".to_string()),
        Just("located_in".to_string()),
        Just("part_of".to_string()),
        Just("caused_by".to_string()),
    ]
}

/// Generate a [`Node`] with the given id, a random type, an optional
/// short summary, and 0..4 simple string props.
fn arb_node(id: NodeId) -> impl Strategy<Value = Node> {
    (
        arb_ntype(),
        "[a-z ]{0,24}",
        prop::collection::btree_map("[a-z]{1,6}", "[a-z0-9]{0,12}", 0..4),
    )
        .prop_map(move |(ntype, summary, props)| {
            let mut n = Node::new(id, ntype).with_summary(summary);
            for (k, v) in props {
                n = n.with_prop(k, Ipld::String(v));
            }
            n
        })
}

/// A transaction operation. Flat enum so permutation and batching are
/// trivial list operations.
#[derive(Clone, Debug)]
enum TxOp {
    AddNode(Node),
    AddEdge(Edge),
    Tombstone(NodeId),
}

/// Generate a sequence of `TxOp` values: up to `n_nodes` `AddNode`,
/// up to `n_edges` `AddEdge` (with endpoints drawn from the node set),
/// and up to `n_nodes/4` tombstones. Add-nodes are emitted first in
/// the base ordering so every edge references an extant node in the
/// unshuffled sequence; the shuffled test (P1) must still succeed
/// because the Transaction API is order-independent within a single
/// commit.
fn arb_commit_sequence(n_nodes: usize, n_edges: usize) -> impl Strategy<Value = Vec<TxOp>> {
    prop::collection::vec(arb_node_id(), 1..=n_nodes)
        // Dedupe NodeIds - proptest may draw the same raw bytes twice
        // and `add_node` with a duplicate id inside one tx would be a
        // semantic error, not a determinism bug.
        .prop_map(|ids| {
            let mut seen = std::collections::HashSet::new();
            ids.into_iter()
                .filter(|id| seen.insert(*id))
                .collect::<Vec<_>>()
        })
        .prop_flat_map(move |ids: Vec<NodeId>| {
            let node_count = ids.len();
            let ids_for_nodes = ids.clone();
            let ids_for_edges = ids.clone();
            let ids_for_tombs = ids.clone();

            let nodes =
                prop::collection::vec(0u64..u64::MAX, node_count).prop_flat_map(move |_seeds| {
                    let per_node: Vec<_> = ids_for_nodes
                        .iter()
                        .copied()
                        .map(|id| arb_node(id).prop_map(TxOp::AddNode).boxed())
                        .collect();
                    per_node
                });

            let edges = prop::collection::vec(
                (0..node_count, 0..node_count, arb_rel(), arb_edge_id()),
                0..=n_edges,
            )
            .prop_map(move |es| {
                // Dedupe EdgeIds the same way we deduped NodeIds.
                let mut seen = std::collections::HashSet::new();
                es.into_iter()
                    .filter(|(_, _, _, eid)| seen.insert(*eid))
                    .map(|(s, d, rel, eid)| {
                        TxOp::AddEdge(Edge::new(eid, rel, ids_for_edges[s], ids_for_edges[d]))
                    })
                    .collect::<Vec<_>>()
            });

            let tombstone_count = node_count / 4;
            let tombs =
                prop::collection::vec(0..node_count, 0..=tombstone_count).prop_map(move |idxs| {
                    let mut seen = std::collections::HashSet::new();
                    idxs.into_iter()
                        .filter(|i| seen.insert(*i))
                        .map(|i| TxOp::Tombstone(ids_for_tombs[i]))
                        .collect::<Vec<_>>()
                });

            (nodes, edges, tombs).prop_map(|(n, e, t)| {
                let mut out = Vec::with_capacity(n.len() + e.len() + t.len());
                out.extend(n);
                out.extend(e);
                out.extend(t);
                out
            })
        })
}

// ============================================================
// Helpers
// ============================================================

fn fresh_stores() -> (Arc<dyn Blockstore>, Arc<dyn OpHeadsStore>) {
    (
        Arc::new(MemoryBlockstore::new()),
        Arc::new(MemoryOpHeadsStore::new()),
    )
}

fn pinned_opts<'a>(author: &'a str, message: &'a str) -> CommitOptions<'a> {
    // Pinning time + change_id is load-bearing for every property here:
    // without it, CID equality across two runs would fail on wall-clock
    // drift alone and tell us nothing about add-order / batching.
    CommitOptions::new(author, message)
        .with_time_micros(1_700_000_000_000_000)
        .with_change_id(ChangeId::from_bytes_raw([0x42; 16]))
}

/// Apply `ops` in a single commit. Returns the head commit CID on
/// success, or `None` if the op-sequence is invalid in isolation
/// (empty / only-tombstones-of-unknown-nodes / duplicate edge id /
/// etc). Properties treat `None` as a skip via `prop_assume!`.
fn apply_single_commit(ops: &[TxOp]) -> Option<Cid> {
    let (bs, ohs) = fresh_stores();
    let repo = ReadonlyRepo::init(bs, ohs).ok()?;
    let mut tx = repo.start_transaction();
    for op in ops {
        match op {
            TxOp::AddNode(n) => {
                tx.add_node(n).ok()?;
            }
            TxOp::AddEdge(e) => {
                tx.add_edge(e).ok()?;
            }
            TxOp::Tombstone(id) => {
                tx.tombstone_node(*id, "proptest").ok()?;
            }
        }
    }
    let repo = tx.commit_opts(pinned_opts("proptest", "single")).ok()?;
    repo.view().heads.first().cloned()
}

/// Apply `ops` split into `k` batches, one commit per batch. Returns
/// the final head commit CID + IndexSet CID.
fn apply_batched(ops: &[TxOp], k: usize) -> Option<(Cid, Cid)> {
    let k = k.max(1).min(ops.len().max(1));
    let (bs, ohs) = fresh_stores();
    let mut repo = ReadonlyRepo::init(bs, ohs).ok()?;
    let chunk_size = ops.len().div_ceil(k);
    for (batch_idx, chunk) in ops.chunks(chunk_size).enumerate() {
        let mut tx = repo.start_transaction();
        for op in chunk {
            match op {
                TxOp::AddNode(n) => {
                    tx.add_node(n).ok()?;
                }
                TxOp::AddEdge(e) => {
                    tx.add_edge(e).ok()?;
                }
                TxOp::Tombstone(id) => {
                    tx.tombstone_node(*id, "proptest").ok()?;
                }
            }
        }
        // Pin the per-batch change_id by the batch index so two
        // different `k` values walk distinct change_ids yet remain
        // deterministic within one run.
        let mut cid_bytes = [0u8; 16];
        cid_bytes[0] = batch_idx as u8;
        let opts = CommitOptions::new("proptest", "batch")
            .with_time_micros(1_700_000_000_000_000 + batch_idx as u64)
            .with_change_id(ChangeId::from_bytes_raw(cid_bytes));
        repo = tx.commit_opts(opts).ok()?;
    }
    let head = repo.view().heads.first().cloned()?;
    let commit = repo.head_commit()?;
    let index_set = commit.indexes.clone()?;
    Some((head, index_set))
}

// ============================================================
// Properties
// ============================================================

proptest! {
    #![proptest_config(ProptestConfig {
        cases: 64,
        max_shrink_iters: 1000,
        .. ProptestConfig::default()
    })]

    /// P1: commit CID is invariant under permutation of the op sequence
    /// inside a single commit. Generalises
    /// `dual_adjacency_commit_cid_is_order_independent` to arbitrary
    /// graphs with tombstones.
    #[test]
    fn p1_add_order_permutation(
        ops in arb_commit_sequence(16, 24),
        shuffle_seed in any::<u64>(),
    ) {
        let cid_a = apply_single_commit(&ops);
        prop_assume!(cid_a.is_some());

        // Fisher-Yates shuffle seeded from `shuffle_seed` so the
        // shrink tree stays deterministic. We deliberately don't use
        // `SliceRandom::shuffle` since that depends on the global rng.
        let mut shuffled = ops.clone();
        let mut state = shuffle_seed | 1; // non-zero
        for i in (1..shuffled.len()).rev() {
            // xorshift64 - fast, deterministic, adequate for shuffling.
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let j = (state as usize) % (i + 1);
            shuffled.swap(i, j);
        }
        let cid_b = apply_single_commit(&shuffled);

        prop_assert_eq!(
            cid_a, cid_b,
            "commit CID must be invariant under op-order permutation"
        );
    }

    /// P2: splitting the same op sequence into `k` append-only batches
    /// yields the same final IndexSet CID for every `k in 1..=8`.
    /// Generalises `incremental_and_full_index_build_produce_identical_index_set`.
    ///
    /// We compare `k=1` (single commit, full-rebuild path) against `k>1`
    /// (multi-commit, incremental path triggered from commit 2 onward).
    #[test]
    fn p2_incremental_vs_full(
        ops in arb_commit_sequence(16, 16),
        k in 2usize..=8,
    ) {
        // Only append-only ops in this property: the incremental
        // fast path gates on "no removals, no tombstones". Filter
        // `TxOp::Tombstone` out up-front so we exercise the invariant
        // the implementation actually claims.
        let append_only: Vec<TxOp> = ops
            .into_iter()
            .filter(|op| !matches!(op, TxOp::Tombstone(_)))
            .collect();
        prop_assume!(!append_only.is_empty());

        let full = apply_batched(&append_only, 1);
        let incr = apply_batched(&append_only, k);
        prop_assume!(full.is_some() && incr.is_some());

        let (_full_head, full_idx) = full.unwrap();
        let (_incr_head, incr_idx) = incr.unwrap();

        prop_assert_eq!(
            full_idx, incr_idx,
            "IndexSet CID must match between full rebuild and k={} incremental batches",
            k
        );
    }

    /// P3: given the same committed graph and the same query, the
    /// returned hits are byte-identical. We do not encode the hits
    /// via CBOR (QueryHit isn't Serialize); we instead compare the
    /// ordered `Vec<NodeId>` projection, which is the observable
    /// surface of a query from an agent's perspective.
    #[test]
    fn p3_retrieve_determinism(
        ops in arb_commit_sequence(12, 8),
    ) {
        // Filter out tombstones + edges to keep the test focused on
        // the retrieval path (not the tombstone-mask or graph-expand
        // paths, which have their own targeted tests).
        let nodes_only: Vec<TxOp> = ops
            .into_iter()
            .filter(|op| matches!(op, TxOp::AddNode(_)))
            .collect();
        prop_assume!(!nodes_only.is_empty());

        // Build the same repo twice from the same ops in the same
        // order. The property is "same input -> same output"; the
        // add-order permutation invariant is tested by P1.
        let (bs1, ohs1) = fresh_stores();
        let repo1 = ReadonlyRepo::init(bs1, ohs1).unwrap();
        let mut tx = repo1.start_transaction();
        for op in &nodes_only {
            if let TxOp::AddNode(n) = op {
                tx.add_node(n).unwrap();
            }
        }
        let repo1 = tx.commit_opts(pinned_opts("proptest", "p3")).unwrap();

        let (bs2, ohs2) = fresh_stores();
        let repo2 = ReadonlyRepo::init(bs2, ohs2).unwrap();
        let mut tx = repo2.start_transaction();
        for op in &nodes_only {
            if let TxOp::AddNode(n) = op {
                tx.add_node(n).unwrap();
            }
        }
        let repo2 = tx.commit_opts(pinned_opts("proptest", "p3")).unwrap();

        // Query the same label on both repos. Project each hit to
        // its NodeId so the comparison is stable across runs.
        for label in ["Person", "Place", "Event", "Concept"] {
            let hits1 = repo1
                .query()
                .label(label)
                .execute()
                .unwrap()
                .into_iter()
                .map(|h| h.node.id)
                .collect::<Vec<_>>();
            let hits2 = repo2
                .query()
                .label(label)
                .execute()
                .unwrap()
                .into_iter()
                .map(|h| h.node.id)
                .collect::<Vec<_>>();
            prop_assert_eq!(
                hits1, hits2,
                "query(label={}) must return byte-identical hit sequence",
                label
            );
        }
    }

    /// P5: tombstoning the same node twice inside one commit is
    /// idempotent - the commit CID equals the commit CID for the
    /// same tombstone applied once.
    #[test]
    fn p5_tombstone_idempotent(
        id in arb_node_id(),
        extra in arb_node_id(),
    ) {
        prop_assume!(id != extra);

        // Seed: one committed node so the tombstone targets something
        // real. (A tombstone against an absent node is a separate
        // code path not covered by this property.)
        let seed_ops = vec![TxOp::AddNode(Node::new(id, "Person"))];
        let cid_seed = apply_single_commit(&seed_ops);
        prop_assume!(cid_seed.is_some());

        // Commit a tombstone once vs twice, starting from a
        // pre-seeded repo each time. We reconstruct the repo from
        // scratch so the test doesn't depend on mutable state.
        let once_ops = vec![
            TxOp::AddNode(Node::new(id, "Person")),
            TxOp::Tombstone(id),
        ];
        let twice_ops = vec![
            TxOp::AddNode(Node::new(id, "Person")),
            TxOp::Tombstone(id),
            TxOp::Tombstone(id),
        ];

        let cid_once = apply_single_commit(&once_ops);
        let cid_twice = apply_single_commit(&twice_ops);

        prop_assert_eq!(
            cid_once, cid_twice,
            "tombstone_node must be idempotent within one commit"
        );
    }
}

// ============================================================
// ci-slow 256-case variants
// ============================================================

#[cfg(feature = "ci-slow")]
proptest! {
    #![proptest_config(ProptestConfig {
        cases: 256,
        max_shrink_iters: 2000,
        .. ProptestConfig::default()
    })]

    #[test]
    fn p1_add_order_permutation_slow(
        ops in arb_commit_sequence(64, 128),
        shuffle_seed in any::<u64>(),
    ) {
        let cid_a = apply_single_commit(&ops);
        prop_assume!(cid_a.is_some());

        let mut shuffled = ops.clone();
        let mut state = shuffle_seed | 1;
        for i in (1..shuffled.len()).rev() {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let j = (state as usize) % (i + 1);
            shuffled.swap(i, j);
        }
        let cid_b = apply_single_commit(&shuffled);
        prop_assert_eq!(cid_a, cid_b);
    }
}
