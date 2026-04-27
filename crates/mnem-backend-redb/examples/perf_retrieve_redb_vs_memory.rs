//! I0d for the 2026-04-21 perf sweep plan: redb vs memory on the
//! full `Retriever::execute()` path.
//!
//! The sibling `perf_layer_attribution` example in mnem-core measures
//! fresh vs warm retrieve on a memory blockstore. This one runs the
//! same measurement on *both* memory and redb backends so the ratio
//! quantifies how much of a per-query cost is fsync + read-tx
//! overhead, which is the ceiling Fix C in the perf sweep plan is
//! trying to lift.
//!
//! Run:
//!
//! ```console
//! cargo run --release -p mnem-backend-redb --example perf_retrieve_redb_vs_memory
//! ```

use std::sync::Arc;
use std::time::{Duration, Instant};

use bytes::Bytes;
use mnem_core::id::NodeId;
use mnem_core::objects::{Dtype, Embedding, Node};
use mnem_core::repo::ReadonlyRepo;
use mnem_core::retrieve::Retriever;
use mnem_core::store::{Blockstore, MemoryBlockstore, MemoryOpHeadsStore, OpHeadsStore};

const EMBED_MODEL: &str = "perf:fake-384";
const DIM: usize = 384;
const NS: &[usize] = &[1_000, 5_000, 10_000];
const RETRIEVE_ROUNDS: usize = 10;

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
        let x = (self.0 >> 33) as u32;
        (x as f32 / (u32::MAX as f32 / 2.0)) - 1.0
    }
}

fn unit_vec(seed: u64) -> Embedding {
    let mut rng = Lcg::new(seed);
    let mut raw = Vec::with_capacity(DIM);
    for _ in 0..DIM {
        raw.push(rng.next_f32());
    }
    let norm: f32 = raw.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-9);
    for x in raw.iter_mut() {
        *x /= norm;
    }
    let mut bytes = Vec::with_capacity(DIM * 4);
    for v in &raw {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    Embedding {
        model: EMBED_MODEL.to_string(),
        dtype: Dtype::F32,
        dim: DIM as u32,
        vector: Bytes::from(bytes),
    }
}

fn seed_into(
    repo: &ReadonlyRepo,
    n: usize,
    from_seed: u64,
) -> Result<ReadonlyRepo, Box<dyn std::error::Error>> {
    let mut tx = repo.start_transaction();
    for i in 0..n {
        let node = Node::new(NodeId::new_v7(), "Doc").with_summary(format!("seed-doc-{i}"));
        let cid = tx.add_node(&node)?;
        let embed = unit_vec(from_seed + i as u64);
        tx.set_embedding(cid, embed.model.clone(), embed)?;
    }
    Ok(tx.commit("bench", "seed")?)
}

struct Measurements {
    idx_build: Duration,
    fresh_avg: Duration,
    warm_avg: Duration,
}

fn measure_one(repo: &ReadonlyRepo) -> Result<Measurements, Box<dyn std::error::Error>> {
    let probe = unit_vec(9_999_999);
    let probe_floats: Vec<f32> = probe
        .vector
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

    let t = Instant::now();
    let idx = Arc::new(repo.build_vector_index(EMBED_MODEL)?);
    let idx_build = t.elapsed();

    let mut warm_total = Duration::ZERO;
    for _ in 0..RETRIEVE_ROUNDS {
        let t = Instant::now();
        let _ = Retriever::new(repo)
            .vector(EMBED_MODEL, probe_floats.clone())
            .with_vector_index(idx.clone())
            .limit(10)
            .execute()?;
        warm_total += t.elapsed();
    }
    let warm_avg = warm_total / RETRIEVE_ROUNDS as u32;

    let mut fresh_total = Duration::ZERO;
    for _ in 0..RETRIEVE_ROUNDS {
        let t = Instant::now();
        let _ = Retriever::new(repo)
            .vector(EMBED_MODEL, probe_floats.clone())
            .limit(10)
            .execute()?;
        fresh_total += t.elapsed();
    }
    let fresh_avg = fresh_total / RETRIEVE_ROUNDS as u32;

    Ok(Measurements {
        idx_build,
        fresh_avg,
        warm_avg,
    })
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("# I0d: redb vs memory on the Retriever::execute() path");
    println!("# mnem-backend-redb perf-retrieve bench");
    println!();
    println!("# Each N is measured twice: once on MemoryBlockstore (in-process,");
    println!("# no fsync), once on redb (persistent, fsync per commit).");
    println!("# ratio_* = redb / memory. Values >1 are fsync + read-tx overhead.");
    println!();
    println!(
        "{:>6} | {:>11} {:>11} {:>6} | {:>11} {:>11} {:>6} | {:>11} {:>11} {:>6}",
        "n",
        "mem_build",
        "redb_build",
        "ratio",
        "mem_fresh",
        "redb_fresh",
        "ratio",
        "mem_warm",
        "redb_warm",
        "ratio",
    );

    for &n in NS {
        // --- memory ---
        let bs: Arc<dyn Blockstore> = Arc::new(MemoryBlockstore::new());
        let ohs: Arc<dyn OpHeadsStore> = Arc::new(MemoryOpHeadsStore::new());
        let mem_repo = seed_into(&ReadonlyRepo::init(bs, ohs)?, n, 1)?;
        let mem = measure_one(&mem_repo)?;
        drop(mem_repo);

        // --- redb (fresh tmp file per N) ---
        let tmp = std::env::temp_dir().join(format!(
            "mnem-bench-retr-redb-{n}-{}.redb",
            std::process::id()
        ));
        let _ = std::fs::remove_file(&tmp);
        let (bs, ohs, _file) = mnem_backend_redb::open_or_init(&tmp)?;
        let redb_repo = seed_into(&ReadonlyRepo::init(bs, ohs)?, n, 1)?;
        let redb = measure_one(&redb_repo)?;
        drop(redb_repo);
        let _ = std::fs::remove_file(&tmp);

        let ratio_build = redb.idx_build.as_secs_f64() / mem.idx_build.as_secs_f64();
        let ratio_fresh = redb.fresh_avg.as_secs_f64() / mem.fresh_avg.as_secs_f64();
        let ratio_warm = redb.warm_avg.as_secs_f64() / mem.warm_avg.as_secs_f64();

        println!(
            "{:>6} | {:>11.2?} {:>11.2?} {:>5.1}x | {:>11.2?} {:>11.2?} {:>5.1}x | {:>11.2?} {:>11.2?} {:>5.1}x",
            n,
            mem.idx_build,
            redb.idx_build,
            ratio_build,
            mem.fresh_avg,
            redb.fresh_avg,
            ratio_fresh,
            mem.warm_avg,
            redb.warm_avg,
            ratio_warm,
        );
    }

    println!();
    println!("# Conclusions:");
    println!("# - ratio_build above 3-5x means the redb read-tx cost per node");
    println!("#   during index scan is significant; Fix C (in-memory backend");
    println!("#   for bench) is justified.");
    println!("# - ratio_warm near 1.0 means the retrieve path itself is CPU-");
    println!("#   bound and backend-independent once the index is cached.");
    println!("# - ratio_fresh = the combined build+search overhead of going");
    println!("#   persistent. This is what a bench question sees.");

    Ok(())
}
