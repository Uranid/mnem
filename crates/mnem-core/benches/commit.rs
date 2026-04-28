//! Criterion bench: `Repo::commit` throughput across pre-seeded repo
//! sizes. Ports `examples/perf_commit_scaling.rs` onto the statistical
//! harness so CI can track wall-time drift commit-over-commit.
//!
//! Phase-B1b (PHASE-A-2 bench plan §2). Run on demand:
//!
//! ```console
//! cargo bench -p mnem-core --bench commit
//! ```

use std::sync::Arc;

use criterion::{BatchSize, BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use ipld_core::ipld::Ipld;
use mnem_core::id::NodeId;
use mnem_core::objects::Node;
use mnem_core::repo::ReadonlyRepo;
use mnem_core::store::{Blockstore, MemoryBlockstore, MemoryOpHeadsStore, OpHeadsStore};

const BULK: u64 = 50;

fn make_repo() -> ReadonlyRepo {
    let bs: Arc<dyn Blockstore> = Arc::new(MemoryBlockstore::new());
    let ohs: Arc<dyn OpHeadsStore> = Arc::new(MemoryOpHeadsStore::new());
    ReadonlyRepo::init(bs, ohs).expect("init in-memory repo")
}

fn seed(mut repo: ReadonlyRepo, n: u64) -> ReadonlyRepo {
    if n == 0 {
        return repo;
    }
    let mut tx = repo.start_transaction();
    for i in 0..n {
        let node = Node::new(NodeId::new_v7(), "Doc")
            .with_summary(format!("seed-{i}"))
            .with_prop("name", Ipld::String(format!("p{i}")));
        tx.add_node(&node).expect("add_node seed");
    }
    repo = tx.commit("bench", "seed").expect("seed commit");
    repo
}

fn bench_commit(c: &mut Criterion) {
    let mut group = c.benchmark_group("commit_append_50");
    group.throughput(Throughput::Elements(BULK));

    for &n in &[0u64, 1_000, 5_000, 10_000] {
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, &n| {
            b.iter_batched(
                || seed(make_repo(), n),
                |repo| {
                    let mut tx = repo.start_transaction();
                    for i in 0..BULK {
                        let node = Node::new(NodeId::new_v7(), "DocAppend")
                            .with_summary(format!("new-{i}"))
                            .with_prop("id_in_batch", Ipld::Integer(i as i128));
                        tx.add_node(&node).expect("add_node append");
                    }
                    tx.commit("bench", "append-50").expect("append commit")
                },
                BatchSize::LargeInput,
            );
        });
    }

    group.finish();
}

criterion_group!(benches, bench_commit);
criterion_main!(benches);
