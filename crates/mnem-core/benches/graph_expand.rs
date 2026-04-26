//! Criterion bench: depth-k BFS graph expansion during retrieve.
//! Seeds a ring-of-hubs graph and measures `Retriever::execute()`
//! with `with_graph_expand(GraphExpand::new().with_depth(k))` for
//! k in {1, 2, 3}.
//!
//! Phase-B1b (PHASE-A-2 bench plan §2). Run on demand:
//!
//! ```console
//! cargo bench -p mnem-core --bench graph_expand
//! ```

use std::sync::Arc;

use bytes::Bytes;
use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use mnem_core::id::{EdgeId, NodeId};
use mnem_core::objects::{Dtype, Edge, Embedding, Node};
use mnem_core::repo::ReadonlyRepo;
use mnem_core::retrieve::{GraphExpand, Retriever};
use mnem_core::store::{Blockstore, MemoryBlockstore, MemoryOpHeadsStore, OpHeadsStore};

const EMBED_MODEL: &str = "bench:fake-384";
const DIM: usize = 384;
const SEED_NODES: u64 = 2_000;
const FANOUT: u64 = 8;

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

fn build_fixture() -> ReadonlyRepo {
    let bs: Arc<dyn Blockstore> = Arc::new(MemoryBlockstore::new());
    let ohs: Arc<dyn OpHeadsStore> = Arc::new(MemoryOpHeadsStore::new());
    let repo = ReadonlyRepo::init(bs, ohs).expect("init");
    let mut tx = repo.start_transaction();
    let mut ids = Vec::with_capacity(SEED_NODES as usize);
    for i in 0..SEED_NODES {
        let id = NodeId::new_v7();
        ids.push(id);
        let node = Node::new(id, "Doc").with_summary(format!("n-{i}"));
        let cid = tx.add_node(&node).expect("add_node");
        let embed = unit_vec(i);
        tx.set_embedding(cid, embed.model.clone(), embed)
            .expect("set_embedding");
    }
    // Each node links to the next FANOUT nodes in a ring.
    for (i, src) in ids.iter().enumerate() {
        for k in 1..=FANOUT {
            let dst = ids[(i + k as usize) % ids.len()];
            let edge = Edge::new(EdgeId::new_v7(), "links", *src, dst);
            tx.add_edge(&edge).expect("add_edge");
        }
    }
    tx.commit("bench", "graph-seed").expect("commit")
}

fn bench_graph_expand(c: &mut Criterion) {
    let repo = build_fixture();
    let idx = Arc::new(repo.build_vector_index(EMBED_MODEL).expect("idx build"));
    let probe = unit_vec(9_999_999);
    let probe_f: Vec<f32> = probe
        .vector
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

    let mut group = c.benchmark_group("graph_expand_k");
    for &depth in &[1usize, 2, 3] {
        group.bench_with_input(BenchmarkId::from_parameter(depth), &depth, |b, &depth| {
            b.iter(|| {
                Retriever::new(&repo)
                    .vector(EMBED_MODEL, probe_f.clone())
                    .with_vector_index(idx.clone())
                    .with_graph_expand(GraphExpand::new().with_depth(depth))
                    .limit(10)
                    .execute()
                    .expect("retrieve")
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_graph_expand);
criterion_main!(benches);
