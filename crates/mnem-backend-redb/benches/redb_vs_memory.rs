//! Criterion bench: backend comparison (memory vs redb) on the
//! 50-node append commit workload. Ports
//! `examples/perf_commit_scaling_redb.rs` onto the statistical
//! harness so persistent-backend drift shows up in CI.
//!
//! Phase-B1b (PHASE-A-2 bench plan §2). Run on demand:
//!
//! ```console
//! cargo bench -p mnem-backend-redb --bench redb_vs_memory
//! ```

use std::sync::Arc;

use criterion::{BatchSize, BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use ipld_core::ipld::Ipld;
use mnem_core::id::NodeId;
use mnem_core::objects::Node;
use mnem_core::repo::ReadonlyRepo;
use mnem_core::store::{Blockstore, MemoryBlockstore, MemoryOpHeadsStore, OpHeadsStore};

const BULK: u64 = 50;

fn seed(mut repo: ReadonlyRepo, n: u64) -> ReadonlyRepo {
    if n == 0 {
        return repo;
    }
    let mut tx = repo.start_transaction();
    for i in 0..n {
        let node = Node::new(NodeId::new_v7(), "Doc")
            .with_summary(format!("seed-{i}"))
            .with_prop("name", Ipld::String(format!("p{i}")));
        tx.add_node(&node).expect("add_node");
    }
    repo = tx.commit("bench", "seed").expect("seed commit");
    repo
}

fn append_commit(repo: ReadonlyRepo) -> ReadonlyRepo {
    let mut tx = repo.start_transaction();
    for i in 0..BULK {
        let node = Node::new(NodeId::new_v7(), "DocAppend")
            .with_summary(format!("new-{i}"))
            .with_prop("id_in_batch", Ipld::Integer(i as i128));
        tx.add_node(&node).expect("add_node");
    }
    tx.commit("bench", "append-50").expect("append commit")
}

fn mem_repo() -> ReadonlyRepo {
    let bs: Arc<dyn Blockstore> = Arc::new(MemoryBlockstore::new());
    let ohs: Arc<dyn OpHeadsStore> = Arc::new(MemoryOpHeadsStore::new());
    ReadonlyRepo::init(bs, ohs).expect("init mem")
}

fn redb_repo(tag: &str) -> (ReadonlyRepo, std::path::PathBuf) {
    let tmp =
        std::env::temp_dir().join(format!("mnem-bench-redb-{tag}-{}.redb", std::process::id()));
    let _ = std::fs::remove_file(&tmp);
    let (bs, ohs, _file) = mnem_backend_redb::open_or_init(&tmp).expect("open_or_init");
    let repo = ReadonlyRepo::init(bs, ohs).expect("init redb");
    (repo, tmp)
}

fn bench_backend_append(c: &mut Criterion) {
    let mut group = c.benchmark_group("backend_append_50");
    group.throughput(Throughput::Elements(BULK));

    for &n in &[0u64, 1_000, 5_000] {
        // Memory backend.
        group.bench_with_input(BenchmarkId::new("memory", n), &n, |b, &n| {
            b.iter_batched(|| seed(mem_repo(), n), append_commit, BatchSize::LargeInput);
        });

        // redb backend. Fresh tmp file per iteration; cleaned up post-run.
        group.bench_with_input(BenchmarkId::new("redb", n), &n, |b, &n| {
            b.iter_batched(
                || {
                    let (repo, path) = redb_repo(&format!("append-{n}"));
                    (seed(repo, n), path)
                },
                |(repo, path)| {
                    let out = append_commit(repo);
                    let _ = std::fs::remove_file(&path);
                    out
                },
                BatchSize::PerIteration,
            );
        });
    }

    group.finish();
}

criterion_group!(benches, bench_backend_append);
criterion_main!(benches);
