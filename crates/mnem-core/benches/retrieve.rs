//! Criterion bench: hot-path `Retriever::execute()` with a pre-built
//! vector index (warm). Ports `examples/perf_layer_attribution.rs`
//! onto the statistical harness for per-query wall time.
//!
//! Phase-B1b (PHASE-A-2 bench plan §2). Run on demand:
//!
//! ```console
//! cargo bench -p mnem-core --bench retrieve
//! ```

use std::sync::Arc;

use bytes::Bytes;
use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use mnem_core::id::NodeId;
use mnem_core::objects::{Dtype, Embedding, Node};
use mnem_core::repo::ReadonlyRepo;
use mnem_core::retrieve::Retriever;
use mnem_core::store::{Blockstore, MemoryBlockstore, MemoryOpHeadsStore, OpHeadsStore};

const EMBED_MODEL: &str = "bench:fake-384";
const DIM: usize = 384;

/// Tiny deterministic LCG. Same generator as
/// `examples/perf_layer_attribution.rs`. Avoids pulling `rand` into
/// the bench dep graph.
struct Lcg(u64);
impl Lcg {
    const fn new(seed: u64) -> Self {
        Self(seed.wrapping_add(0x9e37_79b9_7f4a_7c15))
    }
    fn next_f32(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let bits = (self.0 >> 33) as u32;
        f32::from_bits(0x3f80_0000 | (bits & 0x007f_ffff)) - 1.5
    }
}

fn unit_vec(seed: u64) -> Embedding {
    let mut lcg = Lcg::new(seed);
    let mut raw = [0f32; DIM];
    let mut norm_sq = 0f32;
    for v in &mut raw {
        *v = lcg.next_f32();
        norm_sq += *v * *v;
    }
    let norm = norm_sq.sqrt().max(1e-12);
    let mut bytes = Vec::with_capacity(DIM * 4);
    for v in raw {
        bytes.extend_from_slice(&(v / norm).to_le_bytes());
    }
    Embedding {
        model: EMBED_MODEL.into(),
        dtype: Dtype::F32,
        dim: DIM as u32,
        vector: Bytes::from(bytes),
    }
}

fn seed_repo(n: u64) -> ReadonlyRepo {
    let bs: Arc<dyn Blockstore> = Arc::new(MemoryBlockstore::new());
    let ohs: Arc<dyn OpHeadsStore> = Arc::new(MemoryOpHeadsStore::new());
    let repo = ReadonlyRepo::init(bs, ohs).expect("init");
    let mut tx = repo.start_transaction();
    for i in 0..n {
        let node =
            Node::new(NodeId::new_v7(), "Doc").with_summary(format!("seed-doc-{i}"));
        let cid = tx.add_node(&node).expect("add_node");
        let embed = unit_vec(i);
        tx.set_embedding(cid, embed.model.clone(), embed)
            .expect("set_embedding");
    }
    tx.commit("bench", "retrieve-seed").expect("seed commit")
}

fn bench_retrieve(c: &mut Criterion) {
    let mut group = c.benchmark_group("retrieve_vector_warm");

    for &n in &[1_000u64, 5_000, 10_000] {
        group.throughput(Throughput::Elements(1));
        let repo = seed_repo(n);
        let idx = Arc::new(repo.build_vector_index(EMBED_MODEL).expect("idx build"));
        let probe = unit_vec(9_999_999);
        let probe_f: Vec<f32> = probe
            .vector
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();

        group.bench_with_input(BenchmarkId::new("k10", n), &n, |b, _| {
            b.iter(|| {
                Retriever::new(&repo)
                    .vector(EMBED_MODEL, probe_f.clone())
                    .with_vector_index(idx.clone())
                    .limit(10)
                    .execute()
                    .expect("retrieve")
            });
        });

        group.bench_with_input(BenchmarkId::new("k50", n), &n, |b, _| {
            b.iter(|| {
                Retriever::new(&repo)
                    .vector(EMBED_MODEL, probe_f.clone())
                    .with_vector_index(idx.clone())
                    .limit(50)
                    .execute()
                    .expect("retrieve")
            });
        });
    }

    group.finish();
}

criterion_group!(benches, bench_retrieve);
criterion_main!(benches);
