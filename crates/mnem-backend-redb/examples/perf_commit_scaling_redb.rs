//! Fix X1 verification on the redb backend.
//!
//! Mirror of `perf_commit_scaling` (mnem-core/examples) but runs
//! the same measurement against redb instead of `MemoryBlockstore`.
//! The question this answers: does Fix X1's incremental
//! IndexSet-build path actually speed up the commit wall on
//! persistent storage, or is the redb fsync / page-cache cost the
//! real bottleneck?
//!
//! Pattern matches the `LongMemEval` adapter: each commit uses a
//! UNIQUE label so the fast path hits its happy case (fresh
//! label-tree, no cursor-walk of pre-existing entries in that
//! label group). Secondary indexes get the fast path; the main
//! node tree still takes the existing `rebuild_tree` cursor-walk
//! (Fix X2 territory).
//!
//! Run:
//!
//! ```console
//! cargo run --release -p mnem-backend-redb --example perf_commit_scaling_redb
//! ```

use std::sync::Arc;
use std::time::Instant;

use ipld_core::ipld::Ipld;
use mnem_core::id::NodeId;
use mnem_core::objects::Node;
use mnem_core::repo::ReadonlyRepo;
use mnem_core::store::{Blockstore, MemoryBlockstore, MemoryOpHeadsStore, OpHeadsStore};

const NS: &[usize] = &[0, 1_000, 5_000, 10_000, 25_000];
const BULK: usize = 50;

fn seed_repo(repo: ReadonlyRepo, n: usize) -> Result<ReadonlyRepo, Box<dyn std::error::Error>> {
    let mut tx = repo.start_transaction();
    for i in 0..n {
        let node = Node::new(NodeId::new_v7(), "Doc")
            .with_summary(format!("seed-{i}"))
            .with_prop("name", Ipld::String(format!("p{i}")));
        tx.add_node(&node)?;
    }
    Ok(tx.commit("bench", "seed")?)
}

fn append_commit(
    repo: ReadonlyRepo,
    label: &str,
) -> Result<std::time::Duration, Box<dyn std::error::Error>> {
    let mut tx = repo.start_transaction();
    for i in 0..BULK {
        let node = Node::new(NodeId::new_v7(), label)
            .with_summary(format!("new-{i}"))
            .with_prop("id_in_batch", Ipld::Integer(i as i128));
        tx.add_node(&node)?;
    }
    let t = Instant::now();
    let _ = tx.commit("bench", "append-50")?;
    Ok(t.elapsed())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("# Fix X1 verification: 50-node append on memory vs redb backends");
    println!();
    println!(
        "{:>7} | {:>14} | {:>14} | {:>6}",
        "N_base", "memory_wall", "redb_wall", "ratio"
    );

    for &n in NS {
        // ---- memory backend ----
        let mem_bs: Arc<dyn Blockstore> = Arc::new(MemoryBlockstore::new());
        let mem_ohs: Arc<dyn OpHeadsStore> = Arc::new(MemoryOpHeadsStore::new());
        let mem_repo = if n == 0 {
            ReadonlyRepo::init(mem_bs, mem_ohs)?
        } else {
            seed_repo(ReadonlyRepo::init(mem_bs, mem_ohs)?, n)?
        };
        let mem_wall = append_commit(mem_repo, &format!("NewMem_N{n}"))?;

        // ---- redb backend (fresh tmp file per N) ----
        let tmp = std::env::temp_dir().join(format!(
            "mnem-bench-commit-scaling-{n}-{}.redb",
            std::process::id()
        ));
        let _ = std::fs::remove_file(&tmp);
        let (redb_bs, redb_ohs, _file) = mnem_backend_redb::open_or_init(&tmp)?;
        let redb_repo = if n == 0 {
            ReadonlyRepo::init(redb_bs, redb_ohs)?
        } else {
            seed_repo(ReadonlyRepo::init(redb_bs, redb_ohs)?, n)?
        };
        let redb_wall = append_commit(redb_repo, &format!("NewRedb_N{n}"))?;
        let _ = std::fs::remove_file(&tmp);

        let ratio = redb_wall.as_secs_f64() / mem_wall.as_secs_f64().max(1e-9);
        println!("{n:>7} | {mem_wall:>14.2?} | {redb_wall:>14.2?} | {ratio:>5.1}x");
    }

    println!();
    println!("# Interpretation:");
    println!("# - memory_wall: pure-algorithm cost. Post-Fix-X1, should");
    println!("#   stay sublinear in N for unique-label appends.");
    println!("# - redb_wall: real-storage cost. Includes fsync per");
    println!("#   commit + page-cache effects.");
    println!("# - ratio: redb overhead above memory. If ratio climbs");
    println!("#   sharply with N, Fix X2 (incremental main-node-tree)");
    println!("#   is the next lever. If ratio stays ~2-5x across N,");
    println!("#   fsync-per-commit is the residual and Fix D (already");
    println!("#   applied in adapter) is doing what it can.");

    Ok(())
}
