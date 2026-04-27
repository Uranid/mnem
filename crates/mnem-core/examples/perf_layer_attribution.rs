//!  investigation for the 2026-04-21 perf sweep plan.
//!
//! Answers the three questions that gate every subsequent fix:
//!
//! 1. **Fresh vs warm.** At scale, how much of a single retrieve's
//!    wall time goes to rebuilding the vector index versus the
//!    actual search? Measures `repo.build_vector_index` in isolation,
//!    then `Retriever::execute()` with and without
//!    `with_vector_index(Arc<...>)` pre-built override.
//! 2. **Growth shape.** Does index-rebuild cost grow linearly in the
//!    corpus size (as expected for a brute-force index), and is that
//!    the dominant term in `execute()` at N=25k?
//! 3. **Interspersed ingest-retrieve.** Repeats the `LongMemEval`
//!    access pattern (50 writes, then 1 retrieve, 10 rounds) to
//!    measure the O(Q²) cost per the plan's Fix A hypothesis.
//!
//! No ONNX, no HTTP, no cache layer. Just the mnem-core retrieval
//! primitives timed with `Instant::now()`. Fake 384-dim unit vectors
//! from a deterministic LCG so results are reproducible across runs.
//!
//! Run:
//!
//! ```console
//! cargo run --release -p mnem-core --example perf_layer_attribution
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
const NS: &[usize] = &[1_000, 5_000, 10_000, 25_000];
const RETRIEVE_ROUNDS: usize = 10;

/// Tiny LCG so we don't pull a rand dep into mnem-core. Deterministic,
/// fast, and fine for generating fake vectors whose only requirement
/// is "different per node so the index can't trivially collapse."
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
        // [-1, 1] centred
        (x as f32 / (u32::MAX as f32 / 2.0)) - 1.0
    }
}

fn unit_vec(seed: u64) -> Embedding {
    let mut rng = Lcg::new(seed);
    let mut raw = Vec::with_capacity(DIM);
    for _ in 0..DIM {
        raw.push(rng.next_f32());
    }
    // L2 normalise so cosine is well-defined
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

fn make_repo() -> Result<ReadonlyRepo, Box<dyn std::error::Error>> {
    let bs: Arc<dyn Blockstore> = Arc::new(MemoryBlockstore::new());
    let ohs: Arc<dyn OpHeadsStore> = Arc::new(MemoryOpHeadsStore::new());
    Ok(ReadonlyRepo::init(bs, ohs)?)
}

fn seed(
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

fn measure_fresh_vs_warm(n: usize) -> Result<(), Box<dyn std::error::Error>> {
    let repo = seed(&make_repo()?, n, 1)?;

    // Probe vector (another deterministic unit vec, not seeded into the repo).
    let probe = unit_vec(9_999_999);
    let probe_floats: Vec<f32> = probe
        .vector
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

    // --- Isolated: index build cost.
    let t = Instant::now();
    let idx = Arc::new(repo.build_vector_index(EMBED_MODEL)?);
    let build_ms = t.elapsed();

    // --- Warm retrieves (override path).
    let mut warm_total = Duration::ZERO;
    for _ in 0..RETRIEVE_ROUNDS {
        let t = Instant::now();
        let _ = Retriever::new(&repo)
            .vector(EMBED_MODEL, probe_floats.clone())
            .with_vector_index(idx.clone())
            .limit(10)
            .execute()?;
        warm_total += t.elapsed();
    }
    let warm_avg = warm_total / RETRIEVE_ROUNDS as u32;

    // --- Fresh retrieves (no override, full rebuild each call).
    let mut fresh_total = Duration::ZERO;
    for _ in 0..RETRIEVE_ROUNDS {
        let t = Instant::now();
        let _ = Retriever::new(&repo)
            .vector(EMBED_MODEL, probe_floats.clone())
            .limit(10)
            .execute()?;
        fresh_total += t.elapsed();
    }
    let fresh_avg = fresh_total / RETRIEVE_ROUNDS as u32;

    let build_overhead_pct = 100.0 * build_ms.as_secs_f64() / fresh_avg.as_secs_f64();

    println!(
        "{n:>8} | {build_ms:>12.2?} | {fresh_avg:>12.2?} | {warm_avg:>12.2?} | \
         {build_overhead_pct:>6.1}%"
    );
    Ok(())
}

fn measure_interspersed() -> Result<(), Box<dyn std::error::Error>> {
    println!();
    println!("# I0c: interspersed ingest-retrieve pattern");
    println!("# 50 writes, then 1 retrieve, 10 rounds. Mirrors");
    println!("# LongMemEval per-question session ingest + question query.");
    println!();
    println!(
        "{:>6} | {:>12} | {:>12} | {:>12}",
        "round", "n_total", "ingest50", "retrieve1"
    );

    let mut repo = make_repo()?;
    let probe = unit_vec(9_999_999);
    let probe_floats: Vec<f32> = probe
        .vector
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

    for round in 1..=10 {
        // --- ingest 50 ---
        let t = Instant::now();
        let mut tx = repo.start_transaction();
        for i in 0..50 {
            let node =
                Node::new(NodeId::new_v7(), "Doc").with_summary(format!("round{round}-doc{i}"));
            let cid = tx.add_node(&node)?;
            let embed = unit_vec(round as u64 * 10_000 + i as u64);
            tx.set_embedding(cid, embed.model.clone(), embed)?;
        }
        repo = tx.commit("bench", "batch")?;
        let ingest_t = t.elapsed();

        // --- retrieve (fresh, no override) ---
        let t = Instant::now();
        let _ = Retriever::new(&repo)
            .vector(EMBED_MODEL, probe_floats.clone())
            .limit(10)
            .execute()?;
        let retr_t = t.elapsed();

        println!(
            "{:>6} | {:>12} | {:>12.2?} | {:>12.2?}",
            round,
            round * 50,
            ingest_t,
            retr_t
        );
    }
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("# mnem perf layer-attribution bench ");
    println!("# Fake 384-dim vectors, MemoryBlockstore, no ONNX, no HTTP.");
    println!("# Numbers are for relative comparison only; absolute times");
    println!("# will change with real ONNX embed bytes and redb backend.");
    println!();
    println!("# I0a + I0b: fresh vs warm retrieve at varying N");
    println!("# fresh = Retriever::execute() with no override (mnem-core");
    println!("#   naive path, forces a fresh index build each call).");
    println!("# warm  = Retriever::execute() with with_vector_index(Arc<..>)");
    println!("#   override (mimics mnem-http warm IndexCache).");
    println!("# build%% = build / fresh ratio (how much of a fresh retrieve");
    println!("#   is pure index-build overhead).");
    println!();
    println!(
        "{:>8} | {:>12} | {:>12} | {:>12} | {:>7}",
        "n", "idx_build", "fresh_avg", "warm_avg", "build%"
    );
    for &n in NS {
        measure_fresh_vs_warm(n)?;
    }

    measure_interspersed()?;

    println!();
    println!("# Conclusions (to be filled in by docs/perf-sweep-plan after run):");
    println!("#  - If build% approaches 100% at N=25k, Fix A is mandatory.");
    println!("#  - If retrieve1 grows linearly in n_total during the");
    println!("#    interspersed pattern, that confirms the O(Q^2) hypothesis.");
    println!("#  - If warm_avg is flat in N, the search path itself is");
    println!("#    already acceptable; the O(N^2) behaviour lives in the");
    println!("#    mnem-http cache-invalidation policy, not the algorithm.");

    Ok(())
}
