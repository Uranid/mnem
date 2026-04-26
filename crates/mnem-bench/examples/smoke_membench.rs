//! MemBench smoke gate (10 items per slice by default).
//!
//! Runs the simple/roles slice by default; pass
//! `--slice highlevel-movie` to exercise the high-level/movie cell.
//! Both slices share the same scoring code (`score::membench::run_*`).

use std::path::PathBuf;

use anyhow::{Result, anyhow, bail};
use mnem_bench::Bench;
use mnem_bench::adapters::MnemAdapter;
use mnem_bench::datasets::membench;
use mnem_bench::embed::{BenchEmbedder, DEFAULT_DIM};
use mnem_bench::score::membench as scorer;

fn main() -> Result<()> {
    let opts = parse_args()?;

    let (bench, category, topic, label) = match opts.slice.as_str() {
        "simple-roles" => (
            Bench::MembenchSimpleRoles,
            "simple",
            "roles",
            "MemBench simple-roles",
        ),
        "highlevel-movie" => (
            Bench::MembenchHighlevelMovie,
            "highlevel",
            "movie",
            "MemBench highlevel-movie",
        ),
        other => bail!("unknown --slice: {other} (valid: simple-roles, highlevel-movie)"),
    };

    let path = resolve_dataset(bench)?;
    eprintln!("[smoke-membench] using dataset at {}", path.display());
    let mut items = membench::load_filtered(&path, category, Some(topic))?;
    if items.len() > opts.limit {
        items.truncate(opts.limit);
    }
    if items.is_empty() {
        bail!("no membench items to score");
    }

    let (embedder, used_real_minilm) = build_embedder(opts.bag_of_tokens)?;
    eprintln!(
        "[smoke-membench] embedder = {} (dim {})",
        embedder.model(),
        embedder.dim()
    );
    let mut adapter = MnemAdapter::with_embedder(embedder)
        .map_err(|e| anyhow!("constructing mnem adapter: {e}"))?;
    let (report, rows) = match opts.slice.as_str() {
        "simple-roles" => scorer::run_simple_roles(&mut adapter, &items, 5, &path)?,
        _ => scorer::run_highlevel_movie(&mut adapter, &items, 5, &path)?,
    };
    let r5 = report.overall.get("recall@5").copied().unwrap_or(0.0);

    eprintln!();
    eprintln!("=== mnem-bench smoke ({label}, {} items) ===", report.n_questions);
    eprintln!("recall@5           : {r5:.4}");
    eprintln!("runtime_seconds    : {:.2}", report.runtime_seconds);
    eprintln!("ingest_s           : {:.2}", report.timing.ingest_s);
    eprintln!("retrieve_s         : {:.2}", report.timing.retrieve_s);
    eprintln!("score_s            : {:.2}", report.timing.score_s);
    eprintln!("rows               : {}", rows.len());
    println!("{}", serde_json::to_string_pretty(&report)?);

    if used_real_minilm {
        let floor = 0.30;
        if r5 < floor {
            bail!("smoke FAILED: recall@5 = {r5:.4} (must be >= {floor:.2}).");
        }
        eprintln!(
            "\n[smoke-membench] PASS - recall@5 = {r5:.4} >= {floor:.2} (real MiniLM gate)."
        );
    } else if r5 <= 0.0 {
        bail!("smoke FAILED: recall@5 = {r5:.4} (must be > 0).");
    } else {
        eprintln!("\n[smoke-membench] PASS - recall@5 = {r5:.4} > 0 (fallback gate).");
    }
    Ok(())
}

struct Opts {
    limit: usize,
    bag_of_tokens: bool,
    slice: String,
}

fn parse_args() -> Result<Opts> {
    let mut limit: usize = 10;
    let mut bag_of_tokens = false;
    let mut slice = "simple-roles".to_string();
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "-h" | "--help" => {
                eprintln!(
                    "Usage: smoke_membench [--limit N] [--slice simple-roles|highlevel-movie] [--bag-of-tokens]"
                );
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
            "--slice" => {
                slice = args
                    .next()
                    .ok_or_else(|| anyhow!("--slice requires a value"))?;
            }
            s if s.starts_with("--slice=") => {
                slice = s["--slice=".len()..].to_string();
            }
            other => bail!("unknown arg: {other}"),
        }
    }
    if limit == 0 {
        bail!("--limit must be > 0");
    }
    Ok(Opts { limit, bag_of_tokens, slice })
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
                    "[smoke-membench] WARNING: onnx-minilm init failed ({e}); falling back."
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

fn resolve_dataset(bench: Bench) -> Result<PathBuf> {
    if let Ok(p) = std::env::var("MNEM_BENCH_DATA") {
        let suffix = match bench {
            Bench::MembenchSimpleRoles => "simple.json",
            Bench::MembenchHighlevelMovie => "highlevel.json",
            _ => "simple.json",
        };
        let cand = PathBuf::from(&p).join(bench.metadata().id).join(suffix);
        if cand.is_file() {
            return Ok(cand);
        }
        let lab_root = PathBuf::from(&p)
            .join("../membench/FirstAgent")
            .join(suffix);
        if lab_root.is_file() {
            return Ok(lab_root);
        }
    }
    let lab_local = PathBuf::from("./datasets/membench/FirstAgent").join(
        match bench {
            Bench::MembenchSimpleRoles => "simple.json",
            Bench::MembenchHighlevelMovie => "highlevel.json",
            _ => "simple.json",
        },
    );
    if lab_local.is_file() {
        return Ok(lab_local);
    }
    if let Ok(p) = mnem_bench::datasets::cached_path(bench) {
        if p.is_file() {
            return Ok(p);
        }
    }
    eprintln!("[smoke-membench] cache miss; fetching from HuggingFace...");
    mnem_bench::datasets::fetch(bench, true, |_d, _t| {})
}
