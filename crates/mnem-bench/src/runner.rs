//! Top-level dispatch: take a [`RunPlan`], run each shipped bench,
//! emit RESULTS.md + per-bench JSON / JSONL into the output dir.

use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result, anyhow, bail};

use crate::adapters::MnemAdapter;
use crate::bench::{AdapterKind, Bench, EmbedderChoice, RunMode};
use crate::datasets;
use crate::embed::{BenchEmbedder, DEFAULT_DIM};
use crate::output;
use crate::score::{PerQuestionRow, ScoreReport};

/// Plan for one `mnem bench run` invocation.
#[derive(Clone, Debug)]
pub struct RunPlan {
    /// Benches to attempt.
    pub benches: Vec<Bench>,
    /// Adapters (systems-under-test) to run.
    pub adapters: Vec<AdapterKind>,
    /// Run mode.
    pub mode: RunMode,
    /// Embedder choice.
    pub embedder: EmbedderChoice,
    /// Output directory. Created if missing.
    pub out: PathBuf,
    /// Top-K depth for per-bench retrieves.
    pub top_k: usize,
    /// Per-bench question / conversation cap. `None` = no cap.
    pub limit: Option<usize>,
    /// Skip the cached download check; force re-download.
    pub no_cache: bool,
    /// Suppress all stderr progress output.
    pub quiet: bool,
}

/// Per-bench outcome.
#[derive(Clone, Debug)]
pub struct BenchOutcome {
    /// Bench identity.
    pub bench: Bench,
    /// Adapter run for this outcome.
    pub adapter: AdapterKind,
    /// Final score report (None when the bench was skipped).
    pub report: Option<ScoreReport>,
    /// Free-form skip reason. Empty when the bench succeeded.
    pub skipped_reason: String,
}

/// Dispatch the plan. Returns one [`BenchOutcome`] per
/// `(bench, adapter)` pair.
pub fn run(plan: &RunPlan) -> Result<Vec<BenchOutcome>> {
    fs::create_dir_all(&plan.out).with_context(|| format!("creating {}", plan.out.display()))?;
    let logs_dir = plan.out.join("logs");
    fs::create_dir_all(&logs_dir).context("creating logs/ subdir")?;

    // OnnxMiniLm needs the (default-on) `onnx-minilm` feature; if the
    // crate was built without it, [`build_embedder`] silently falls back
    // to bag-of-tokens. Surface the notice up-front.
    if matches!(plan.embedder, EmbedderChoice::OnnxMiniLm) {
        #[cfg(not(feature = "onnx-minilm"))]
        eprintln!(
            "[mnem bench] embedder 'onnx-minilm' was selected but mnem-bench was \
             built without the `onnx-minilm` feature; falling back to bag-of-tokens. \
             Rebuild with `cargo build -p mnem-bench --features onnx-minilm`."
        );
    }

    let timing_log_path = plan.out.join("timing.log");
    let mut timing_log = fs::File::create(&timing_log_path)
        .with_context(|| format!("creating {}", timing_log_path.display()))?;
    let t_total = Instant::now();

    let mut outcomes = Vec::new();

    for adapter_kind in &plan.adapters {
        for bench in &plan.benches {
            let meta = bench.metadata();
            let embedder =
                build_embedder(plan.embedder).map_err(|e| anyhow!("constructing embedder: {e}"))?;
            let mut adapter = MnemAdapter::with_embedder(embedder)
                .map_err(|e| anyhow!("constructing mnem adapter: {e}"))?;

            let outcome = match *bench {
                Bench::LongMemEval => run_longmemeval(&mut adapter, plan, &logs_dir),
                Bench::Locomo => run_locomo(&mut adapter, plan, &logs_dir),
                Bench::Convomem => run_convomem(&mut adapter, plan, &logs_dir),
                Bench::MembenchSimpleRoles => {
                    run_membench(&mut adapter, plan, &logs_dir, MembenchSlice::SimpleRoles)
                }
                Bench::MembenchHighlevelMovie => {
                    run_membench(&mut adapter, plan, &logs_dir, MembenchSlice::HighlevelMovie)
                }
                Bench::LongMemEvalHybridV4 => {
                    run_longmemeval_hybrid_v4(&mut adapter, plan, &logs_dir)
                }
            };
            match outcome {
                Ok((report, rows)) => {
                    write_outputs(plan, *bench, &report, &rows)?;
                    writeln!(
                        timing_log,
                        "[{}] {} runtime_s={:.2} ingest_s={:.2} retrieve_s={:.2} score_s={:.2}",
                        adapter_kind.id(),
                        meta.id,
                        report.runtime_seconds,
                        report.timing.ingest_s,
                        report.timing.retrieve_s,
                        report.timing.score_s,
                    )
                    .ok();
                    outcomes.push(BenchOutcome {
                        bench: *bench,
                        adapter: *adapter_kind,
                        report: Some(report),
                        skipped_reason: String::new(),
                    });
                }
                Err(e) => {
                    let msg = format!("bench {} failed: {e:#}", meta.id);
                    eprintln!("[mnem bench] {msg}");
                    let log_path = logs_dir.join(format!("{}.log", meta.id));
                    let _ = fs::write(&log_path, msg.as_bytes());
                    outcomes.push(BenchOutcome {
                        bench: *bench,
                        adapter: *adapter_kind,
                        report: None,
                        skipped_reason: format!("error: {e}"),
                    });
                }
            }
        }
    }

    writeln!(
        timing_log,
        "[total] elapsed_s={:.2}",
        t_total.elapsed().as_secs_f64()
    )
    .ok();
    output::write_results_md(&plan.out, &outcomes)?;
    Ok(outcomes)
}

fn run_longmemeval(
    adapter: &mut MnemAdapter,
    plan: &RunPlan,
    logs_dir: &Path,
) -> Result<(ScoreReport, Vec<PerQuestionRow>)> {
    let path = resolve_dataset(Bench::LongMemEval, plan)?;
    let mut all = crate::datasets::longmemeval::load(&path)?;
    if let Some(n) = plan.limit
        && all.len() > n
    {
        all.truncate(n);
    }
    if all.is_empty() {
        bail!(
            "longmemeval dataset at {} contained no questions",
            path.display()
        );
    }
    if !plan.quiet {
        eprintln!("[mnem bench] longmemeval: {} questions", all.len());
    }
    let log_path = logs_dir.join("longmemeval.log");
    let _ = fs::write(
        &log_path,
        format!(
            "longmemeval dataset={} n_questions={} top_k={}\n",
            path.display(),
            all.len(),
            plan.top_k
        ),
    );
    crate::score::longmemeval::run(adapter, &all, plan.top_k, &path)
}

fn run_locomo(
    adapter: &mut MnemAdapter,
    plan: &RunPlan,
    logs_dir: &Path,
) -> Result<(ScoreReport, Vec<PerQuestionRow>)> {
    let path = resolve_dataset(Bench::Locomo, plan)?;
    let mut all = crate::datasets::locomo::load(&path)?;
    if let Some(n) = plan.limit
        && all.len() > n
    {
        all.truncate(n);
    }
    if all.is_empty() {
        bail!(
            "locomo dataset at {} contained no conversations",
            path.display()
        );
    }
    if !plan.quiet {
        eprintln!("[mnem bench] locomo: {} conversations", all.len());
    }
    let log_path = logs_dir.join("locomo.log");
    let _ = fs::write(
        &log_path,
        format!(
            "locomo dataset={} n_conversations={} top_k={}\n",
            path.display(),
            all.len(),
            plan.top_k
        ),
    );
    crate::score::locomo::run(adapter, &all, plan.top_k, &path)
}

fn run_convomem(
    adapter: &mut MnemAdapter,
    plan: &RunPlan,
    logs_dir: &Path,
) -> Result<(ScoreReport, Vec<PerQuestionRow>)> {
    let path = resolve_dataset(Bench::Convomem, plan)?;
    let mut items = crate::datasets::convomem::load(&path)?;
    if let Some(n) = plan.limit
        && items.len() > n
    {
        items.truncate(n);
    }
    if items.is_empty() {
        bail!("convomem dataset at {} contained no items", path.display());
    }
    if !plan.quiet {
        eprintln!("[mnem bench] convomem: {} items", items.len());
    }
    let log_path = logs_dir.join("convomem.log");
    let _ = fs::write(
        &log_path,
        format!(
            "convomem dataset={} n_items={} top_k={}\n",
            path.display(),
            items.len(),
            plan.top_k
        ),
    );
    crate::score::convomem::run(adapter, &items, plan.top_k, &path)
}

/// Which MemBench slice the runner should execute.
#[derive(Clone, Copy, Debug)]
enum MembenchSlice {
    /// `simple.json` filtered by topic=`roles`.
    SimpleRoles,
    /// `highlevel.json` filtered by topic=`movie`.
    HighlevelMovie,
}

/// Default headline-slice cap for MemBench (matches the n=100 used by
/// MemPalace's published numbers, so the comparison is apples-to-
/// apples). User-supplied `--limit` overrides.
const MEMBENCH_HEADLINE_CAP: usize = 100;

fn run_membench(
    adapter: &mut MnemAdapter,
    plan: &RunPlan,
    logs_dir: &Path,
    slice: MembenchSlice,
) -> Result<(ScoreReport, Vec<PerQuestionRow>)> {
    let (bench, category, topic) = match slice {
        MembenchSlice::SimpleRoles => (Bench::MembenchSimpleRoles, "simple", "roles"),
        MembenchSlice::HighlevelMovie => (Bench::MembenchHighlevelMovie, "highlevel", "movie"),
    };
    let path = resolve_dataset(bench, plan)?;
    let mut items = crate::datasets::membench::load_filtered(&path, category, Some(topic))?;
    let effective_limit = plan.limit.unwrap_or(MEMBENCH_HEADLINE_CAP);
    if items.len() > effective_limit {
        items.truncate(effective_limit);
    }
    if items.is_empty() {
        bail!(
            "membench dataset at {} contained no items for topic={}",
            path.display(),
            topic
        );
    }
    if !plan.quiet {
        eprintln!(
            "[mnem bench] {}: {} items (topic={})",
            bench.metadata().id,
            items.len(),
            topic
        );
    }
    let log_path = logs_dir.join(format!("{}.log", bench.metadata().id));
    let _ = fs::write(
        &log_path,
        format!(
            "{} dataset={} n_items={} top_k={} topic={}\n",
            bench.metadata().id,
            path.display(),
            items.len(),
            plan.top_k,
            topic,
        ),
    );
    match slice {
        MembenchSlice::SimpleRoles => {
            crate::score::membench::run_simple_roles(adapter, &items, plan.top_k, &path)
        }
        MembenchSlice::HighlevelMovie => {
            crate::score::membench::run_highlevel_movie(adapter, &items, plan.top_k, &path)
        }
    }
}

fn run_longmemeval_hybrid_v4(
    adapter: &mut MnemAdapter,
    plan: &RunPlan,
    logs_dir: &Path,
) -> Result<(ScoreReport, Vec<PerQuestionRow>)> {
    // Reuses the LongMemEval cache. No separate dataset blob exists.
    let path = resolve_dataset(Bench::LongMemEval, plan)?;
    let mut all = crate::datasets::longmemeval::load(&path)?;
    if let Some(n) = plan.limit
        && all.len() > n
    {
        all.truncate(n);
    }
    if all.is_empty() {
        bail!(
            "longmemeval dataset at {} contained no questions (hybrid-v4)",
            path.display()
        );
    }
    if !plan.quiet {
        eprintln!(
            "[mnem bench] longmemeval-hybrid-v4: {} questions (boost_weight={})",
            all.len(),
            crate::score::hybrid_v4::DEFAULT_BOOST_WEIGHT
        );
    }
    let log_path = logs_dir.join("longmemeval-hybrid-v4.log");
    let _ = fs::write(
        &log_path,
        format!(
            "longmemeval-hybrid-v4 dataset={} n_questions={} top_k={} boost_weight={}\n",
            path.display(),
            all.len(),
            plan.top_k,
            crate::score::hybrid_v4::DEFAULT_BOOST_WEIGHT,
        ),
    );
    crate::score::hybrid_v4::run(
        adapter,
        &all,
        plan.top_k,
        &path,
        crate::score::hybrid_v4::DEFAULT_BOOST_WEIGHT,
    )
}

/// Locate the dataset for `bench`. If a copy exists in the cache
/// dir we use it; otherwise call into [`datasets::fetch`] to
/// download. Mirrors the upstream Python "fail-fast if file
/// missing" UX when network is unavailable.
fn resolve_dataset(bench: Bench, plan: &RunPlan) -> Result<PathBuf> {
    let cached = datasets::cached_path(bench)?;
    if cached.is_file() && !plan.no_cache {
        return Ok(cached);
    }
    if !plan.quiet {
        eprintln!("[mnem bench] fetching {} dataset...", bench.metadata().id);
    }
    datasets::fetch(bench, !plan.no_cache, |_d, _t| {})
}

/// Resolve the runtime embedder for a given [`EmbedderChoice`].
/// `OnnxMiniLm` needs the `onnx-minilm` feature; absent the feature
/// we silently fall back to bag-of-tokens (the runner already printed
/// a notice).
fn build_embedder(choice: EmbedderChoice) -> Result<BenchEmbedder> {
    match choice {
        EmbedderChoice::BagOfTokens => Ok(BenchEmbedder::bag_of_tokens(DEFAULT_DIM)),
        EmbedderChoice::OnnxMiniLm => {
            #[cfg(feature = "onnx-minilm")]
            {
                BenchEmbedder::onnx_minilm().map_err(|e| anyhow!("onnx-minilm init: {e}"))
            }
            #[cfg(not(feature = "onnx-minilm"))]
            {
                Ok(BenchEmbedder::bag_of_tokens(DEFAULT_DIM))
            }
        }
    }
}

fn write_outputs(
    plan: &RunPlan,
    bench: Bench,
    report: &ScoreReport,
    rows: &[PerQuestionRow],
) -> Result<()> {
    let id = bench.metadata().id;
    let json_path = plan.out.join(format!("{id}.json"));
    fs::write(&json_path, serde_json::to_vec_pretty(report)?)
        .with_context(|| format!("writing {}", json_path.display()))?;
    let jsonl_path = plan.out.join(format!("{id}.jsonl"));
    let mut jsonl = fs::File::create(&jsonl_path)
        .with_context(|| format!("creating {}", jsonl_path.display()))?;
    for row in rows {
        serde_json::to_writer(&mut jsonl, row)?;
        jsonl.write_all(b"\n")?;
    }
    Ok(())
}
