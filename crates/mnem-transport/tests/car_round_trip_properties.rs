//! P4: CAR round-trip preserves every reachable CID byte-for-byte.
//!
//! For any graph committed to a source blockstore, exporting the head
//! commit to a CAR v1 archive and importing into a fresh blockstore
//! must result in:
//!
//! 1. Every CID reachable from the old head resolving in the new repo,
//!    AND
//! 2. The bytes stored under each CID being byte-identical on both
//!    sides.
//!
//! This generalises the fixed-case round-trip in
//! `examples/export_then_import.rs` over arbitrary graph shapes.

#![allow(clippy::needless_pass_by_value)]

use std::sync::Arc;

use ipld_core::ipld::Ipld;
use proptest::prelude::*;

use mnem_core::id::{ChangeId, EdgeId, NodeId};
use mnem_core::objects::{Edge, Node};
use mnem_core::repo::{CommitOptions, ReadonlyRepo};
use mnem_core::store::{Blockstore, MemoryBlockstore, MemoryOpHeadsStore, OpHeadsStore};

// ============================================================
// Strategies (kept local to the crate - no shared test crate)
// ============================================================

fn arb_node_id() -> impl Strategy<Value = NodeId> {
    any::<[u8; 16]>().prop_map(NodeId::from_bytes_raw)
}

fn arb_edge_id() -> impl Strategy<Value = EdgeId> {
    any::<[u8; 16]>().prop_map(EdgeId::from_bytes_raw)
}

fn arb_ntype() -> impl Strategy<Value = String> {
    prop_oneof![
        Just("Person".to_string()),
        Just("Place".to_string()),
        Just("Event".to_string()),
        Just("Concept".to_string()),
    ]
}

fn arb_rel() -> impl Strategy<Value = String> {
    prop_oneof![
        Just("knows".to_string()),
        Just("located_in".to_string()),
        Just("part_of".to_string()),
        Just("caused_by".to_string()),
    ]
}

#[derive(Clone, Debug)]
struct GraphShape {
    nodes: Vec<Node>,
    edges: Vec<Edge>,
}

fn arb_graph(n_nodes: usize, n_edges: usize) -> impl Strategy<Value = GraphShape> {
    prop::collection::vec(arb_node_id(), 1..=n_nodes)
        .prop_map(|ids| {
            let mut seen = std::collections::HashSet::new();
            ids.into_iter()
                .filter(|id| seen.insert(*id))
                .collect::<Vec<_>>()
        })
        .prop_flat_map(move |ids: Vec<NodeId>| {
            let count = ids.len();
            let ids_for_nodes = ids.clone();
            let ids_for_edges = ids.clone();
            let nodes = prop::collection::vec((arb_ntype(), "[a-z ]{0,16}"), count..=count)
                .prop_map(move |specs| {
                    specs
                        .into_iter()
                        .zip(ids_for_nodes.iter().copied())
                        .map(|((ntype, summary), id)| Node::new(id, ntype).with_summary(summary))
                        .collect::<Vec<_>>()
                });
            let edges =
                prop::collection::vec((0..count, 0..count, arb_rel(), arb_edge_id()), 0..=n_edges)
                    .prop_map(move |es| {
                        let mut seen = std::collections::HashSet::new();
                        es.into_iter()
                            .filter(|(_, _, _, eid)| seen.insert(*eid))
                            .map(|(s, d, rel, eid)| {
                                Edge::new(eid, rel, ids_for_edges[s], ids_for_edges[d])
                            })
                            .collect::<Vec<_>>()
                    });
            (nodes, edges).prop_map(|(nodes, edges)| GraphShape { nodes, edges })
        })
}

fn build_source_repo(graph: &GraphShape) -> Option<(Arc<dyn Blockstore>, mnem_core::id::Cid)> {
    let bs: Arc<dyn Blockstore> = Arc::new(MemoryBlockstore::new());
    let ohs: Arc<dyn OpHeadsStore> = Arc::new(MemoryOpHeadsStore::new());
    let repo = ReadonlyRepo::init(bs.clone(), ohs).ok()?;
    let mut tx = repo.start_transaction();
    for n in &graph.nodes {
        tx.add_node(n).ok()?;
    }
    for e in &graph.edges {
        // Silently drop edges that reference nodes filtered by
        // dedupe above - those are the only failure mode here.
        let _ = tx.add_edge(e);
    }
    let opts = CommitOptions::new("proptest", "p4-seed")
        .with_time_micros(1_700_000_000_000_000)
        .with_change_id(ChangeId::from_bytes_raw([0x55; 16]));
    let repo = tx.commit_opts(opts).ok()?;
    let head = repo.view().heads.first().cloned()?;
    // Just ignore the Ipld import so rustc doesn't warn on test code
    // that could later reach for `Ipld::*` without an explicit use.
    let _ = Ipld::Null;
    Some((bs, head))
}

// ============================================================
// Property P4
// ============================================================

proptest! {
    #![proptest_config(ProptestConfig {
        cases: 64,
        max_shrink_iters: 1000,
        .. ProptestConfig::default()
    })]

    /// Export the head commit's reachable subtree to a CAR buffer,
    /// import into a FRESH `MemoryBlockstore`, assert every CID
    /// reachable from the source resolves on the destination AND
    /// that the block bytes are byte-identical.
    #[test]
    fn p4_car_round_trip_preserves_reachable_blocks(
        graph in arb_graph(16, 24),
    ) {
        let seeded = build_source_repo(&graph);
        prop_assume!(seeded.is_some());
        let (src_bs, head_cid) = seeded.unwrap();

        // Export.
        let mut buf: Vec<u8> = Vec::new();
        let export_stats = mnem_transport::export(&*src_bs, &head_cid, &mut buf)
            .expect("export must succeed on an in-memory repo");

        // Import into a fresh blockstore.
        let dst_bs = MemoryBlockstore::new();
        let import_stats =
            mnem_transport::import(&mut buf.as_slice(), &dst_bs)
                .expect("import must succeed on a well-formed CAR");

        prop_assert_eq!(
            import_stats.blocks, export_stats.blocks,
            "imported block count must equal exported block count"
        );
        prop_assert_eq!(
            import_stats.roots.len(),
            1,
            "single-root export"
        );
        prop_assert_eq!(
            &import_stats.roots[0], &head_cid,
            "imported root must equal exported root"
        );

        // Byte-identity: for every block reachable from the head on
        // the source, the destination must carry the same bytes. We
        // walk via `iter_from_root` which is the same iterator
        // `export` used, so if any CID leaked (no-op verified below)
        // or a block was re-encoded on import the check fires.
        for entry in src_bs.iter_from_root(&head_cid) {
            let (cid, src_bytes) = entry.expect("source iter must not fail");
            let dst_bytes = dst_bs
                .get(&cid)
                .expect("dst get must not fail")
                .expect("every reachable CID must be present on destination");
            prop_assert_eq!(
                src_bytes.as_ref(),
                dst_bytes.as_ref(),
                "block bytes must be byte-identical across CAR round-trip for cid={}",
                cid
            );
        }
    }
}

#[cfg(feature = "ci-slow")]
proptest! {
    #![proptest_config(ProptestConfig {
        cases: 256,
        max_shrink_iters: 2000,
        .. ProptestConfig::default()
    })]

    #[test]
    fn p4_car_round_trip_preserves_reachable_blocks_slow(
        graph in arb_graph(64, 128),
    ) {
        let seeded = build_source_repo(&graph);
        prop_assume!(seeded.is_some());
        let (src_bs, head_cid) = seeded.unwrap();

        let mut buf: Vec<u8> = Vec::new();
        mnem_transport::export(&*src_bs, &head_cid, &mut buf).unwrap();
        let dst_bs = MemoryBlockstore::new();
        mnem_transport::import(&mut buf.as_slice(), &dst_bs).unwrap();

        for entry in src_bs.iter_from_root(&head_cid) {
            let (cid, src_bytes) = entry.unwrap();
            let dst_bytes = dst_bs.get(&cid).unwrap().unwrap();
            prop_assert_eq!(src_bytes.as_ref(), dst_bytes.as_ref());
        }
    }
}
