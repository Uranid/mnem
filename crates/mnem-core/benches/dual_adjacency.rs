//! Criterion bench: dual-adjacency (forward + reverse) insert cost
//! across batch sizes. Exercises the `add_edge` path that indexes
//! both directions of the edge relation.
//!
//! Phase-B1b (PHASE-A-2 bench plan §2). Run on demand:
//!
//! ```console
//! cargo bench -p mnem-core --bench dual_adjacency
//! ```

use std::sync::Arc;

use criterion::{BatchSize, BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use mnem_core::id::{EdgeId, NodeId};
use mnem_core::objects::{Edge, Node};
use mnem_core::repo::ReadonlyRepo;
use mnem_core::store::{Blockstore, MemoryBlockstore, MemoryOpHeadsStore, OpHeadsStore};

fn make_repo_with_nodes(n: u64) -> (ReadonlyRepo, Vec<NodeId>) {
    let bs: Arc<dyn Blockstore> = Arc::new(MemoryBlockstore::new());
    let ohs: Arc<dyn OpHeadsStore> = Arc::new(MemoryOpHeadsStore::new());
    let repo = ReadonlyRepo::init(bs, ohs).expect("init");
    let mut tx = repo.start_transaction();
    let mut ids = Vec::with_capacity(n as usize);
    for i in 0..n {
        let id = NodeId::new_v7();
        ids.push(id);
        let node = Node::new(id, "Doc").with_summary(format!("n-{i}"));
        tx.add_node(&node).expect("add_node");
    }
    (tx.commit("bench", "seed-nodes").expect("seed commit"), ids)
}

fn bench_dual_adjacency(c: &mut Criterion) {
    let mut group = c.benchmark_group("dual_adjacency_insert");

    for &m in &[1_000u64, 10_000] {
        // We need at least `m+1` endpoints so the "ring" edge pattern
        // has distinct src/dst pairs. 2*m is a safe upper bound.
        let n_nodes = m * 2;
        let (base_repo, ids) = make_repo_with_nodes(n_nodes);

        group.throughput(Throughput::Elements(m));
        group.bench_with_input(BenchmarkId::from_parameter(m), &m, |b, &m| {
            b.iter_batched(
                || base_repo.clone(),
                |repo| {
                    let mut tx = repo.start_transaction();
                    for i in 0..m {
                        let src = ids[i as usize];
                        let dst = ids[(i as usize + 1) % ids.len()];
                        let edge = Edge::new(EdgeId::new_v7(), "knows", src, dst);
                        tx.add_edge(&edge).expect("add_edge");
                    }
                    tx.commit("bench", "edges").expect("commit")
                },
                BatchSize::LargeInput,
            );
        });
    }

    group.finish();
}

criterion_group!(benches, bench_dual_adjacency);
criterion_main!(benches);
