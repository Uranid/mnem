//! Side-by-side bench of [`HnswVectorIndex`] vs
//! [`BruteForceVectorIndex`] on the same synthetic corpus.
//!
//! Prints build + search latency at N = 1k, 10k, 50k, plus a recall
//! check against the brute-force ground-truth top-10.
//!
//! Run:
//!
//! ```console
//! cargo run --release -p mnem-ann --example hnsw_vs_brute
//! ```

use std::sync::Arc;
use std::time::Instant;

use bytes::Bytes;
use mnem_ann::HnswVectorIndex;
use mnem_core::id::NodeId;
use mnem_core::index::vector::{BruteForceVectorIndex, VectorIndex};
use mnem_core::objects::{Dtype, Embedding, Node};
use mnem_core::repo::ReadonlyRepo;
use mnem_core::store::{Blockstore, MemoryBlockstore, MemoryOpHeadsStore, OpHeadsStore};

const MODEL: &str = "bench:synthetic-1024";
const DIM: usize = 1024;
const NS: &[usize] = &[1_000, 10_000, 50_000];

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("# HNSW vs BruteForce - same corpus, same queries");
    println!("# mnem-ann 0.1.0, dim={DIM}, model={MODEL}\n");
    println!(
        "{:>7} | {:>12} | {:>12} | {:>10} | {:>10} | {:>10}",
        "n", "brute build", "hnsw build", "brute q", "hnsw q", "recall@10"
    );

    for &n in NS {
        bench_one(n)?;
    }
    Ok(())
}

fn bench_one(n: usize) -> Result<(), Box<dyn std::error::Error>> {
    // ---- build a repo with N nodes carrying synthetic embeddings ----
    let bs: Arc<dyn Blockstore> = Arc::new(MemoryBlockstore::new());
    let ohs: Arc<dyn OpHeadsStore> = Arc::new(MemoryOpHeadsStore::new());
    let r = ReadonlyRepo::init(bs.clone(), ohs.clone())?;

    let mut tx = r.start_transaction();
    for i in 0..n {
        let v = synth_vec(i);
        let mut bytes = Vec::with_capacity(DIM * 4);
        for x in &v {
            bytes.extend_from_slice(&x.to_le_bytes());
        }
        let embed = Embedding {
            model: MODEL.into(),
            dtype: Dtype::F32,
            dim: DIM as u32,
            vector: Bytes::from(bytes),
        };
        let node = Node::new(NodeId::new_v7(), "Vec");
        let cid = tx.add_node(&node)?;
        tx.set_embedding(cid, embed.model.clone(), embed)?;
    }
    let r = tx.commit("bench", "seed")?;

    // ---- build both indexes ----
    let t = Instant::now();
    let brute = BruteForceVectorIndex::build_from_repo(&r, MODEL)?;
    let brute_build = t.elapsed();

    let t = Instant::now();
    let hnsw = HnswVectorIndex::build_from_repo(&r, MODEL)?;
    let hnsw_build = t.elapsed();

    // ---- 50 random queries; measure mean latency + recall@10 ----
    const Q: usize = 50;
    const K: usize = 10;
    let queries: Vec<Vec<f32>> = (0..Q).map(|i| synth_vec(n + i)).collect();

    let t = Instant::now();
    let mut brute_top: Vec<Vec<NodeId>> = Vec::with_capacity(Q);
    for q in &queries {
        let hits = brute.search(q, K)?;
        brute_top.push(hits.into_iter().map(|h| h.node_id).collect());
    }
    let brute_q = t.elapsed() / Q as u32;

    let t = Instant::now();
    let mut hnsw_top: Vec<Vec<NodeId>> = Vec::with_capacity(Q);
    for q in &queries {
        let hits = hnsw.search(q, K)?;
        hnsw_top.push(hits.into_iter().map(|h| h.node_id).collect());
    }
    let hnsw_q = t.elapsed() / Q as u32;

    // Recall@10: fraction of brute's top-10 also present in hnsw's.
    let mut hit_sum = 0usize;
    for (bt, ht) in brute_top.iter().zip(hnsw_top.iter()) {
        let hset: std::collections::HashSet<_> = ht.iter().collect();
        hit_sum += bt.iter().filter(|id| hset.contains(id)).count();
    }
    let recall = hit_sum as f32 / (Q * K) as f32;

    println!(
        "{n:>7} | {brute_build:>12.2?} | {hnsw_build:>12.2?} | {brute_q:>10.2?} | \
         {hnsw_q:>10.2?} | {recall:>9.3}"
    );
    Ok(())
}

/// Synthesise a `DIM`-dim vector deterministically keyed by `i`. We
/// seed it with a small periodic pattern so close `i` values cluster
/// (mimics the real-world "similar docs have similar embeddings"
/// distribution) without needing a real embedder.
fn synth_vec(i: usize) -> Vec<f32> {
    let mut v = vec![0.0_f32; DIM];
    // Four cluster-driving coords keyed by i, plus light noise on
    // the rest. Gives the HNSW graph real structure to exploit.
    v[i % DIM] = 1.0;
    v[(i / 7) % DIM] += 0.5;
    v[(i / 53) % DIM] += 0.25;
    v[(i / 389) % DIM] += 0.125;
    for k in 0..8 {
        let idx = ((i.wrapping_mul(2_654_435_761).wrapping_add(k)) % DIM as usize) as usize;
        v[idx] += 0.01 * ((i + k) as f32).sin();
    }
    v
}
