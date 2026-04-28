//! ConvoMem smoke gate (10-item slice by default).
//!
//! Defaults to the bundled MiniLM-L6-v2 ONNX embedder so this
//! mirrors the headline measurement path. Pass `--limit N` to run
//! more items; pass `--bag-of-tokens` to confirm a regression is in
//! the embedder path vs the retriever path. Walks
//! `mnem_bench::datasets::convomem::fetch_into` to materialise the
//! merged blob if the cache is empty.

use std::path::PathBuf;

use anyhow::{Result, anyhow, bail};
use mnem_bench::Bench;
use mnem_bench::adapters::MnemAdapter;
use mnem_bench::datasets::convomem;
use mnem_bench::embed::{BenchEmbedder, DEFAULT_DIM};
use mnem_bench::score::convomem as scorer;

fn main() -> Result<()> {
    let opts = parse_args()?;

    let path = resolve_dataset()?;
    eprintln!("[smoke-convomem] using dataset at {}", path.display());
    let mut items = convomem::load(&path)?;
    if items.len() > opts.limit {
        items.truncate(opts.limit);
    }
    if items.is_empty() {
        bail!("no convomem items to score");
    }

    let (embedder, used_real_minilm) = build_embedder(opts.bag_of_tokens)?;
    eprintln!(
        "[smoke-convomem] embedder = {} (dim {})",
        embedder.model(),
        embedder.dim()
    );
    let mut adapter = MnemAdapter::with_embedder(embedder)
        .map_err(|e| anyhow!("constructing mnem adapter: {e}"))?;
    let (report, rows) = scorer::run(&mut adapter, &items, 10, &path)?;
    let avg_recall = report.overall.get("avg_recall").copied().unwrap_or(0.0);

    eprintln!();
    eprintln!(
        "=== mnem-bench smoke (ConvoMem, {} items) ===",
        report.n_questions
    );
    eprintln!("avg_recall         : {avg_recall:.4}");
    eprintln!("runtime_seconds    : {:.2}", report.runtime_seconds);
    eprintln!("ingest_s           : {:.2}", report.timing.ingest_s);
    eprintln!("retrieve_s         : {:.2}", report.timing.retrieve_s);
    eprintln!("score_s            : {:.2}", report.timing.score_s);
    eprintln!("rows               : {}", rows.len());
    println!("{}", serde_json::to_string_pretty(&report)?);

    if used_real_minilm {
        let floor = 0.50;
        if avg_recall < floor {
            bail!(
                "smoke FAILED: avg_recall = {avg_recall:.4} (must be >= {floor:.2} \
                 with real MiniLM on ConvoMem)."
            );
        }
        eprintln!(
            "\n[smoke-convomem] PASS - avg_recall = {avg_recall:.4} >= {floor:.2} (real MiniLM gate)."
        );
    } else if avg_recall <= 0.0 {
        bail!("smoke FAILED: avg_recall = {avg_recall:.4} (must be > 0).");
    } else {
        eprintln!("\n[smoke-convomem] PASS - avg_recall = {avg_recall:.4} > 0 (fallback gate).");
    }
    Ok(())
}

struct Opts {
    limit: usize,
    bag_of_tokens: bool,
}

fn parse_args() -> Result<Opts> {
    let mut limit: usize = 10;
    let mut bag_of_tokens = false;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "-h" | "--help" => {
                eprintln!("Usage: smoke_convomem [--limit N] [--bag-of-tokens]");
                std::process::exit(0);
            }
            "--bag-of-tokens" => bag_of_tokens = true,
            "--limit" => {
                let v = args
                    .next()
                    .ok_or_else(|| anyhow!("--limit requires a value"))?;
                limit = v.parse().map_err(|e| anyhow!("--limit {v}: {e}"))?;
            }
            s if s.starts_with("--limit=") => {
                let v = &s["--limit=".len()..];
                limit = v.parse().map_err(|e| anyhow!("--limit {v}: {e}"))?;
            }
            other => bail!("unknown arg: {other}"),
        }
    }
    if limit == 0 {
        bail!("--limit must be > 0");
    }
    Ok(Opts {
        limit,
        bag_of_tokens,
    })
}

fn build_embedder(force_bag: bool) -> Result<(BenchEmbedder, bool)> {
    if force_bag {
        return Ok((BenchEmbedder::bag_of_tokens(DEFAULT_DIM), false));
    }
    #[cfg(feature = "onnx-minilm")]
    {
        match BenchEmbedder::onnx_minilm() {
            Ok(e) => Ok((e, true)),
            Err(e) => {
                eprintln!(
                    "[smoke-convomem] WARNING: onnx-minilm init failed ({e}); \
                     falling back to bag-of-tokens."
                );
                Ok((BenchEmbedder::bag_of_tokens(DEFAULT_DIM), false))
            }
        }
    }
    #[cfg(not(feature = "onnx-minilm"))]
    {
        Ok((BenchEmbedder::bag_of_tokens(DEFAULT_DIM), false))
    }
}

fn resolve_dataset() -> Result<PathBuf> {
    if let Ok(p) = std::env::var("MNEM_BENCH_DATA") {
        let cand = PathBuf::from(p)
            .join("convomem")
            .join("convomem_evidence.json");
        if cand.is_file() {
            return Ok(cand);
        }
    }
    if let Ok(p) = mnem_bench::datasets::cached_path(Bench::Convomem) {
        if p.is_file() {
            return Ok(p);
        }
    }
    eprintln!("[smoke-convomem] cache miss; fetching shards from HuggingFace...");
    mnem_bench::datasets::fetch(Bench::Convomem, true, |_d, _t| {})
}
