//! Q0 sanity benchmark: ingest 10k small nodes via
//! `Transaction::commit` batches and report wall time.
//!
//! Compared branch-to-branch (pre-Q0 `bobby-cat-integration-r3` vs
//! post-Q0 `q0-put-trusted`) to verify the ~0.1-0.3% expected
//! speed-up from skipping the blockstore's CID recompute on the
//! hot path. Big swings either way flag a bug.
//!
//! Usage:
//!
//! ```console
//! cargo run --release -p mnem-core --example q0_put_trusted_ingest_bench
//! ```

use std::sync::Arc;
use std::time::Instant;

use ipld_core::ipld::Ipld;
use mnem_core::id::NodeId;
use mnem_core::objects::Node;
use mnem_core::repo::ReadonlyRepo;
use mnem_core::store::{Blockstore, MemoryBlockstore, MemoryOpHeadsStore, OpHeadsStore};

const TOTAL: usize = 10_000;
const PER_BATCH: usize = 50;

fn run_once() -> Result<std::time::Duration, Box<dyn std::error::Error>> {
    let bs: Arc<dyn Blockstore> = Arc::new(MemoryBlockstore::new());
    let ohs: Arc<dyn OpHeadsStore> = Arc::new(MemoryOpHeadsStore::new());
    let mut repo = ReadonlyRepo::init(bs, ohs)?;

    let start = Instant::now();
    let batches = TOTAL / PER_BATCH;
    for b in 0..batches {
        let mut tx = repo.start_transaction();
        for i in 0..PER_BATCH {
            let idx = b * PER_BATCH + i;
            let node = Node::new(NodeId::new_v7(), "Doc")
                .with_summary(format!("q0-bench-{idx}"))
                .with_prop("batch", Ipld::Integer(b as i128))
                .with_prop("name", Ipld::String(format!("p{i}")));
            tx.add_node(&node)?;
        }
        repo = tx.commit("bench", "q0 ingest")?;
    }
    Ok(start.elapsed())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Warm-up run (JIT-y path caches, first-time allocations).
    let _ = run_once()?;

    // Three timed runs; report min, median, max.
    let mut samples: Vec<std::time::Duration> = Vec::new();
    for _ in 0..3 {
        samples.push(run_once()?);
    }
    samples.sort();
    let min = samples[0];
    let med = samples[1];
    let max = samples[2];

    println!("# Q0 put_trusted ingest bench: {TOTAL} nodes, {PER_BATCH}/batch");
    println!("# samples (sorted): {min:?}, {med:?}, {max:?}");
    println!("min_ms,{}", min.as_secs_f64() * 1_000.0);
    println!("med_ms,{}", med.as_secs_f64() * 1_000.0);
    println!("max_ms,{}", max.as_secs_f64() * 1_000.0);
    Ok(())
}
