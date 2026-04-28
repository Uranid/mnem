//! `mnem bench` subcommand. Wraps the `mnem-bench` crate.
//!
//! Surface:
//!
//! - `mnem bench`                interactive TUI (default).
//! - `mnem bench list`           JSON of available benches.
//! - `mnem bench fetch [BENCH]`  download dataset(s) into the cache.
//! - `mnem bench run [...]`      explicit args, CI-friendly.
//! - `mnem bench results [DIR]`  re-render RESULTS.md from prior run.

use std::path::PathBuf;

use anyhow::{Context, Result, anyhow, bail};
use clap::{Args, Subcommand};

use mnem_bench::bench::{AdapterKind, Bench, EmbedderChoice, RunMode};
use mnem_bench::output;
use mnem_bench::runner::{self, RunPlan};

/// Top-level `mnem bench` clap surface.
#[derive(Args, Debug)]
#[command(after_long_help = "\
Examples:

  # Interactive setup wizard.
  mnem bench

  # Print available benches as JSON (CI-friendly).
  mnem bench list

  # Download a dataset to ~/.mnem/bench-data/.
  mnem bench fetch longmemeval

  # Non-interactive run, single bench, single adapter.
  mnem bench run --benches longmemeval --with mnem --mode cpu-local \\
                 --out ./bench-out --top-k 10 --limit 5 --non-interactive

  # Re-render RESULTS.md from a prior run directory.
  mnem bench results ./bench-out
")]
pub(crate) struct BenchArgs {
    /// Subcommand. Omit to enter the interactive TUI.
    #[command(subcommand)]
    pub sub: Option<BenchSub>,
}

/// Bench subcommands.
#[derive(Subcommand, Debug)]
pub(crate) enum BenchSub {
    /// Print every available bench (JSON).
    List(ListArgs),
    /// Download a dataset into the local cache.
    Fetch(FetchArgs),
    /// Run one or more benches non-interactively.
    Run(RunArgs),
    /// Re-render RESULTS.md from `<dir>/<bench>.json` files.
    Results(ResultsArgs),
}

/// `mnem bench list` flags.
#[derive(Args, Debug)]
pub(crate) struct ListArgs {
    /// Pretty-print the JSON. Default: minified.
    #[arg(long)]
    pub pretty: bool,
}

/// `mnem bench fetch` flags.
#[derive(Args, Debug)]
pub(crate) struct FetchArgs {
    /// Bench id (e.g. `longmemeval`). Omit to fetch every shipped
    /// bench.
    pub bench: Option<String>,
    /// Force re-download even when a verified cached copy exists.
    #[arg(long)]
    pub no_cache: bool,
}

/// `mnem bench run` flags.
#[derive(Args, Debug)]
pub(crate) struct RunArgs {
    /// Comma-separated bench ids (e.g. `longmemeval,locomo`).
    /// Defaults to every shipped bench.
    #[arg(long, value_delimiter = ',')]
    pub benches: Vec<String>,

    /// Comma-separated adapter ids (e.g. `mnem`).
    #[arg(long = "with", value_delimiter = ',', default_values_t = vec!["mnem".to_string()])]
    pub with: Vec<String>,

    /// Run mode.
    #[arg(long, default_value = "cpu-local")]
    pub mode: String,

    /// Output directory.
    #[arg(long, default_value = "./bench-out")]
    pub out: PathBuf,

    /// Top-K depth for retrieve.
    #[arg(long, default_value_t = 10)]
    pub top_k: usize,

    /// Embedder selector. Default `onnx-minilm` runs the bundled
    /// MiniLM-L6-v2 in-process (matches headline numbers). Pass
    /// `bag-of-tokens` for offline / CI runs (deterministic toy,
    /// network-free).
    #[arg(long, default_value = "onnx-minilm")]
    pub embedder: String,

    /// Skip the interactive wizard. Honoured even when the args
    /// would have triggered it (defensive default for CI).
    #[arg(long)]
    pub non_interactive: bool,

    /// Force re-download even when a verified cached copy exists.
    #[arg(long)]
    pub no_cache: bool,

    /// Cap on per-bench question / conversation count.
    #[arg(long)]
    pub limit: Option<usize>,
}

/// `mnem bench results` flags.
#[derive(Args, Debug)]
pub(crate) struct ResultsArgs {
    /// Run directory to re-render. Defaults to `./bench-out`.
    #[arg(default_value = "./bench-out")]
    pub dir: PathBuf,
}

/// Entry point invoked from `main.rs`.
pub(crate) fn run(args: BenchArgs) -> Result<()> {
    match args.sub {
        None => run_interactive(),
        Some(BenchSub::List(a)) => run_list(a),
        Some(BenchSub::Fetch(a)) => run_fetch(a),
        Some(BenchSub::Run(a)) => run_run(a),
        Some(BenchSub::Results(a)) => run_results(a),
    }
}

fn run_interactive() -> Result<()> {
    let plan = mnem_bench::tui::run_tui("./bench-out")?;
    if let Some(plan) = plan {
        let outcomes = runner::run(&plan)?;
        print_outcome_summary(&outcomes);
    }
    Ok(())
}

fn run_list(a: ListArgs) -> Result<()> {
    let entries = output::list_benches();
    let s = if a.pretty {
        serde_json::to_string_pretty(&entries)?
    } else {
        serde_json::to_string(&entries)?
    };
    println!("{s}");
    Ok(())
}

fn run_fetch(a: FetchArgs) -> Result<()> {
    let explicit = a.bench.is_some();
    let benches: Vec<Bench> = match a.bench.as_deref() {
        Some(id) => vec![Bench::from_id(id).ok_or_else(|| anyhow!("unknown bench: {id}"))?],
        None => Bench::all().to_vec(),
    };
    for b in benches {
        // hybrid-v4 reuses the LongMemEval cache. Skip the duplicate
        // download when the operator did not name it explicitly.
        if matches!(b, Bench::LongMemEvalHybridV4) && !explicit {
            eprintln!(
                "[mnem bench] {} reuses the longmemeval cache; skipping duplicate fetch.",
                b.metadata().id
            );
            continue;
        }
        eprintln!("[mnem bench] fetching {}...", b.metadata().id);
        let path = mnem_bench::datasets::fetch(b, !a.no_cache, |d, t| {
            if t > 0 {
                eprint!("\r  {d}/{t} bytes");
            }
        })
        .with_context(|| format!("fetching {}", b.metadata().id))?;
        eprintln!("\n  cached at {}", path.display());
    }
    Ok(())
}

fn run_run(a: RunArgs) -> Result<()> {
    let benches: Vec<Bench> = if a.benches.is_empty() {
        Bench::all().to_vec()
    } else {
        let mut out = Vec::with_capacity(a.benches.len());
        for id in &a.benches {
            out.push(Bench::from_id(id).ok_or_else(|| anyhow!("unknown bench: {id}"))?);
        }
        out
    };
    let mut adapters = Vec::with_capacity(a.with.len());
    for id in &a.with {
        adapters.push(AdapterKind::from_id(id).ok_or_else(|| anyhow!("unknown adapter: {id}"))?);
    }
    let mode = RunMode::from_id(&a.mode).ok_or_else(|| anyhow!("unknown mode: {}", a.mode))?;
    let embedder = EmbedderChoice::from_id(&a.embedder)
        .ok_or_else(|| anyhow!("unknown embedder: {}", a.embedder))?;

    if benches.is_empty() {
        bail!("no benches selected");
    }
    if adapters.is_empty() {
        bail!("no adapters selected (pass --with mnem)");
    }

    let plan = RunPlan {
        benches,
        adapters,
        mode,
        embedder,
        out: a.out,
        top_k: a.top_k,
        limit: a.limit,
        no_cache: a.no_cache,
        quiet: a.non_interactive,
    };
    let outcomes = runner::run(&plan)?;
    print_outcome_summary(&outcomes);
    Ok(())
}

fn run_results(a: ResultsArgs) -> Result<()> {
    output::rerender_from_dir(&a.dir)
        .with_context(|| format!("rerendering RESULTS.md from {}", a.dir.display()))?;
    println!("rendered {}/RESULTS.md", a.dir.display());
    Ok(())
}

fn print_outcome_summary(outcomes: &[mnem_bench::BenchOutcome]) {
    println!();
    println!("=== mnem-bench summary ===");
    for o in outcomes {
        match &o.report {
            Some(r) => {
                // Headline metric pick: ConvoMem reports
                // `avg_recall`; every other shipped bench reports
                // `recall@5` + `recall@10`.
                let metric_line = if let Some(v) = r.overall.get("recall@5") {
                    let r10 = r.overall.get("recall@10").copied().unwrap_or(0.0);
                    format!("R@5={v:.4} R@10={r10:.4}")
                } else if let Some(v) = r.overall.get("avg_recall") {
                    format!("avg_recall={v:.4}")
                } else {
                    "(no headline metric)".to_string()
                };
                println!(
                    "  {} ({}): n={} {metric_line} runtime={:.1}s",
                    o.bench.metadata().id,
                    o.adapter.id(),
                    r.n_questions,
                    r.runtime_seconds,
                );
            }
            None => {
                println!(
                    "  {} ({}): SKIPPED ({})",
                    o.bench.metadata().id,
                    o.adapter.id(),
                    o.skipped_reason,
                );
            }
        }
    }
}
