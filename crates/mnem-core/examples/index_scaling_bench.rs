//! Scaling benchmark for the secondary-index retrieval path.
//!
//! Seeds graphs of 1k, 10k, 50k Person nodes (with 3 props each) and
//! measures:
//!   - commit time (index rebuild is O(n); naive build path)
//!   - point lookup by property (must stay sub-millisecond)
//!   - label-scoped iteration top-5 (must stay under ~50 ms at 10k)
//!   - 1-hop traversal via adjacency (must stay under ~20 ms at 10k)
//!
//! `MemoryBlockstore` (the default) measures the index-layer ceiling -
//! no I/O, no fsync. Future work will add a `RedbBlockstore` lane that
//! exercises the real persistent path; the placeholder is noted in
//! the banner.
//!
//! Run:
//! ```console
//! cargo run --release -p mnem-core --example index_scaling_bench
//! ```

use std::sync::Arc;
use std::time::Instant;

use ipld_core::ipld::Ipld;
use mnem_core::id::{EdgeId, NodeId};
use mnem_core::index::{PropPredicate, Query};
use mnem_core::objects::{Edge, Node};
use mnem_core::repo::ReadonlyRepo;
use mnem_core::store::{Blockstore, MemoryBlockstore, MemoryOpHeadsStore, OpHeadsStore};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("# mnem index scaling benchmark");
    println!("# mnem-core {}", mnem_core::VERSION);
    println!("# backend:   MemoryBlockstore (in-process, no fsync, no I/O)");
    println!("#            This is the index-layer ceiling, NOT persistent-");
    println!("#            store performance. Redb numbers will be slower");
    println!("#            (multiple read-tx opens per Prolly hop, fsync");
    println!("#            per put) review triggers.");
    println!();
    println!(
        "{:>8}  {:>10}  {:>10}  {:>10}  {:>10}",
        "n", "commit", "point_lookup", "label_top5", "traverse"
    );

    for &n in &[1_000usize, 10_000, 50_000] {
        run_size(n)?;
    }

    Ok(())
}

fn run_size(n: usize) -> Result<(), Box<dyn std::error::Error>> {
    let bs: Arc<dyn Blockstore> = Arc::new(MemoryBlockstore::new());
    let ohs: Arc<dyn OpHeadsStore> = Arc::new(MemoryOpHeadsStore::new());
    let repo = ReadonlyRepo::init(bs, ohs)?;

    // Seed.
    let t = Instant::now();
    let mut tx = repo.start_transaction();
    let mut ids = Vec::with_capacity(n);
    for i in 0..n {
        let node = Node::new(NodeId::new_v7(), "Person")
            .with_prop("name", Ipld::String(format!("P_{i:07}")))
            .with_prop(
                "role",
                Ipld::String(if i % 3 == 0 { "Eng" } else { "Mgr" }.into()),
            );
        tx.add_node(&node)?;
        ids.push(node.id);
    }
    // Add edges (5x nodes).
    for i in 0..(n * 5) {
        let from = ids[i % n];
        let to = ids[(i * 31 + 7) % n];
        if from == to {
            continue;
        }
        let e = Edge::new(EdgeId::new_v7(), "knows", from, to);
        tx.add_edge(&e)?;
    }
    let repo = tx.commit("bench", "seed")?;
    let commit_ms = t.elapsed().as_millis();

    // Point lookup.
    let query_target = format!("P_{:07}", n / 2);
    let t = Instant::now();
    let hits = Query::new(&repo)
        .label("Person")
        .where_prop(
            "name",
            PropPredicate::Eq(Ipld::String(query_target.clone())),
        )
        .execute()?;
    let point_us = t.elapsed().as_micros();
    assert_eq!(hits.len(), 1);

    // Label-scoped top-5.
    let t = Instant::now();
    let hits = Query::new(&repo).label("Person").limit(5).execute()?;
    let label_us = t.elapsed().as_micros();
    assert_eq!(hits.len(), 5);

    // 1-hop traversal with outgoing edges.
    let t = Instant::now();
    let hits = Query::new(&repo)
        .label("Person")
        .where_prop("name", PropPredicate::Eq(Ipld::String(query_target)))
        .with_outgoing("knows")
        .execute()?;
    let traverse_us = t.elapsed().as_micros();
    let _edges = hits.first().map_or(0, |h| h.edges.len());

    println!("{n:>8}  {commit_ms:>9} ms  {point_us:>9} µs  {label_us:>9} µs  {traverse_us:>9} µs");
    Ok(())
}
