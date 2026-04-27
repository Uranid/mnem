//! Compare the in-memory blockstore against the redb persistent
//! backend on the same ingest + retrieve workload. Answers "what do
//! I lose by going persistent?" in measured numbers.
//!
//! Run:
//!
//! ```console
//! cargo run --release -p mnem-backend-redb --example redb_vs_memory
//! ```

use std::sync::Arc;
use std::time::Instant;

use ipld_core::ipld::Ipld;
use mnem_core::id::NodeId;
use mnem_core::objects::Node;
use mnem_core::repo::ReadonlyRepo;
use mnem_core::store::{Blockstore, MemoryBlockstore, MemoryOpHeadsStore, OpHeadsStore};

const NS: &[usize] = &[1_000, 10_000];

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("# redb vs memory - same workload, different backend");
    println!("# mnem-backend-redb 0.1.0\n");
    println!(
        "{:>7} | {:>18} | {:>18} | {:>14} | {:>14}",
        "n", "memory ingest", "redb ingest", "memory lookup", "redb lookup"
    );

    for &n in NS {
        bench_one(n)?;
    }
    println!();
    println!("# notes");
    println!("# - 'ingest' is a single transaction with N add_node calls + commit.");
    println!("# - 'lookup' is a point-query by property (name=X), not vector search.");
    println!("# - redb opens one read-tx per Prolly hop; expect multiple orders");
    println!("#   slower on hot path than MemoryBlockstore.");
    println!("# - benchmarks are warm-cache; first redb run is cold-fsync'd.");
    Ok(())
}

fn bench_one(n: usize) -> Result<(), Box<dyn std::error::Error>> {
    // ---- memory baseline ----
    let (mem_ingest, mem_lookup) = bench_memory(n)?;

    // ---- redb persistent ----
    let tmp = std::env::temp_dir().join(format!("mnem-bench-redb-{n}-{}.redb", std::process::id()));
    let _ = std::fs::remove_file(&tmp);
    let (redb_ingest, redb_lookup) = bench_redb(n, &tmp)?;
    let _ = std::fs::remove_file(&tmp);

    println!(
        "{n:>7} | {mem_ingest:>18.2?} | {redb_ingest:>18.2?} | {mem_lookup:>14.2?} | \
         {redb_lookup:>14.2?}"
    );
    Ok(())
}

fn bench_memory(
    n: usize,
) -> Result<(std::time::Duration, std::time::Duration), Box<dyn std::error::Error>> {
    let bs: Arc<dyn Blockstore> = Arc::new(MemoryBlockstore::new());
    let ohs: Arc<dyn OpHeadsStore> = Arc::new(MemoryOpHeadsStore::new());
    let r = ReadonlyRepo::init(bs.clone(), ohs.clone())?;
    bench_ingest_and_lookup(&r, n)
}

fn bench_redb(
    n: usize,
    path: &std::path::Path,
) -> Result<(std::time::Duration, std::time::Duration), Box<dyn std::error::Error>> {
    let (bs, ohs, _) = mnem_backend_redb::open_or_init(path)?;
    let r = ReadonlyRepo::init(bs.clone(), ohs.clone())?;
    bench_ingest_and_lookup(&r, n)
}

fn bench_ingest_and_lookup(
    r: &ReadonlyRepo,
    n: usize,
) -> Result<(std::time::Duration, std::time::Duration), Box<dyn std::error::Error>> {
    // --- ingest timing ---
    let t = Instant::now();
    let mut tx = r.start_transaction();
    for i in 0..n {
        let node = Node::new(NodeId::new_v7(), "Person")
            .with_prop("name", Ipld::String(format!("Person_{i:06}")))
            .with_prop("age", Ipld::Integer(((i % 80) + 18) as i128));
        tx.add_node(&node)?;
    }
    let r = tx.commit("bench", "seed")?;
    let ingest = t.elapsed();

    // --- point-lookup timing ---
    // Pick a property value that exists in the last third of the
    // cursor so the Prolly walk is non-trivial.
    let target = format!("Person_{:06}", (n as f32 * 0.7) as usize);
    let t = Instant::now();
    const ROUNDS: u32 = 100;
    for _ in 0..ROUNDS {
        use mnem_core::index::PropPredicate;
        let _ = r
            .query()
            .label("Person")
            .where_prop("name", PropPredicate::Eq(Ipld::String(target.clone())))
            .execute()?;
    }
    let lookup = t.elapsed() / ROUNDS;

    Ok((ingest, lookup))
}
