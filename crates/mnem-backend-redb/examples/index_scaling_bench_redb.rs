//! Disk-backed companion to `mnem-core/examples/index_scaling_bench`.
//!
//! Same workload, same measurements, but against the real redb
//! blockstore (every `get`/`put` opens a read/write transaction and
//! fsyncs). This is what an agent-memory server actually pays at
//! runtime. Numbers will be several multiples slower than the
//! `MemoryBlockstore` ceiling - that is the point.
//!
//! Run:
//! ```console
//! cargo run --release -p mnem-backend-redb --example index_scaling_bench_redb
//! ```

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use ipld_core::ipld::Ipld;
use mnem_backend_redb::open_or_init;
use mnem_core::ReadonlyRepo;
use mnem_core::id::{EdgeId, NodeId};
use mnem_core::objects::{Edge, Node};
use mnem_core::store::{Blockstore, OpHeadsStore};

fn db_path(n: usize) -> PathBuf {
    let p = std::env::temp_dir().join(format!(
        "mnem-index-bench-redb-{n}-{}-{}.redb",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64
    ));
    let _ = std::fs::remove_file(&p);
    p
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("# mnem index scaling benchmark - REDB (disk-backed)");
    println!("# mnem-core {}", mnem_core::VERSION);
    println!("# backend:   RedbBlockstore (per-op read/write transaction, fsync on every put)");
    println!("#            These numbers reflect real persistent-store cost, not the");
    println!("#            MemoryBlockstore ceiling shown in the mnem-core bench.");
    println!();
    println!(
        "{:>8}  {:>10}  {:>12}  {:>10}  {:>10}",
        "n", "commit", "point_lookup", "label_top5", "traverse"
    );

    // Redb commits are slow (~100x MemoryBlockstore due to per-put fsync
    // before batched-write-tx lands); cap the scaling sweep at 1k for
    // this run to stay inside CI budgets. 10k+ rows should be re-run
    // once the batched-commit refactor exists.
    for &n in &[1_000usize] {
        run_size(n)?;
    }
    println!();
    println!("# 10k/50k omitted here: naive per-put fsync makes the commit path");
    println!("# impractical without batched redb write-tx .");

    Ok(())
}

fn run_size(n: usize) -> Result<(), Box<dyn std::error::Error>> {
    let path = db_path(n);
    let (bs, ohs, _file): (Arc<dyn Blockstore>, Arc<dyn OpHeadsStore>, _) = open_or_init(&path)?;
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
    let hits = repo
        .query()
        .label("Person")
        .where_eq("name", query_target.clone())
        .execute()?;
    let point_us = t.elapsed().as_micros();
    assert_eq!(hits.len(), 1);

    // Label-scoped top-5.
    let t = Instant::now();
    let hits = repo.query().label("Person").limit(5).execute()?;
    let label_us = t.elapsed().as_micros();
    assert_eq!(hits.len(), 5);

    // 1-hop traversal.
    let t = Instant::now();
    let _hits = repo
        .query()
        .label("Person")
        .where_eq("name", query_target)
        .with_outgoing("knows")
        .execute()?;
    let traverse_us = t.elapsed().as_micros();

    println!("{n:>8}  {commit_ms:>9} ms  {point_us:>11} µs  {label_us:>9} µs  {traverse_us:>9} µs");

    // Clean up the temp file so the bench doesn't leak gigabytes across runs.
    drop(repo);
    let _ = std::fs::remove_file(&path);
    Ok(())
}
