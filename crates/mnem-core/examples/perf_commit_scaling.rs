//! Fix X1 verification: measure commit cost of adding 50 nodes to a
//! repo pre-seeded to N in {0, 1k, 5k, 10k, 25k}.
//!
//! Before Fix X1: `Transaction::commit()` rebuilds every secondary
//! index from scratch, so the cost grows linearly with the existing
//! repo size. A 50-node commit at N=25k takes ~30 s in production
//! (measured directly on a live mnem-http container).
//!
//! After Fix X1: the fast path in `index::incremental_append_indexes`
//! only touches sub-trees that the new nodes actually modify. Cost
//! should be roughly flat in N for a pure-append workload.
//!
//! This bench uses `MemoryBlockstore` so the absolute numbers are
//! tiny, but the SHAPE of the curve (flat vs linear growth) is what
//! tells us the fix works.
//!
//! Run:
//!
//! ```console
//! cargo run --release -p mnem-core --example perf_commit_scaling
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

fn make_repo() -> Result<ReadonlyRepo, Box<dyn std::error::Error>> {
    let bs: Arc<dyn Blockstore> = Arc::new(MemoryBlockstore::new());
    let ohs: Arc<dyn OpHeadsStore> = Arc::new(MemoryOpHeadsStore::new());
    Ok(ReadonlyRepo::init(bs, ohs)?)
}

fn seed(
    repo: ReadonlyRepo,
    n: usize,
    seed_offset: u64,
) -> Result<ReadonlyRepo, Box<dyn std::error::Error>> {
    let mut tx = repo.start_transaction();
    for i in 0..n {
        let node = Node::new(NodeId::new_v7(), "Doc")
            .with_summary(format!("seed-{i}"))
            .with_prop("batch", Ipld::Integer((seed_offset + i as u64) as i128))
            .with_prop("name", Ipld::String(format!("p{i}")));
        tx.add_node(&node)?;
    }
    Ok(tx.commit("bench", "seed")?)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("# Fix X1 verification: commit cost of a 50-node append");
    println!("# at varying pre-seeded repo sizes.");
    println!();
    println!("# Pre-fix expectation: commit time grows linearly with N");
    println!("# (full IndexSet rebuild walks every node in the repo).");
    println!();
    println!("# Post-fix expectation: commit time roughly flat in N");
    println!("# because the fast path only touches sub-trees the new");
    println!("# nodes modify (same label + same props).");
    println!();
    println!(
        "{:>7} | {:>14} | {:>15}",
        "N_base", "commit_50_wall", "per_node_us"
    );

    for &n in NS {
        // Seed.
        let repo = if n == 0 {
            make_repo()?
        } else {
            seed(make_repo()?, n, 0)?
        };

        // Matches the LongMemEval adapter's pattern: every commit uses
        // a UNIQUE label (e.g. "LmeQs:{qid}") so the new nodes land in
        // a fresh sub-tree with no existing entries. The Fix X1 fast
        // path then does zero cursor work on the pre-existing 25k
        // label trees; it only builds the new label's tree from the
        // 50 additions.
        let unique_label = format!("NewBatchAtN{n}");
        let mut tx = repo.start_transaction();
        for i in 0..BULK {
            let node = Node::new(NodeId::new_v7(), &unique_label)
                .with_summary(format!("newbatch-{i}"))
                .with_prop("id_in_batch", Ipld::Integer(i as i128))
                .with_prop("newflag", Ipld::Bool(true));
            tx.add_node(&node)?;
        }
        let t = Instant::now();
        let _ = tx.commit("bench", "append-50")?;
        let elapsed = t.elapsed();

        let per_node_us = elapsed.as_micros() as f64 / BULK as f64;
        println!("{n:>7} | {elapsed:>14.2?} | {per_node_us:>13.1}");
    }

    println!();
    println!("# If `commit_50_wall` grows roughly linearly with N_base,");
    println!("# Fix X1 is NOT effective. If the curve is flat, Fix X1");
    println!("# works as intended.");

    Ok(())
}
