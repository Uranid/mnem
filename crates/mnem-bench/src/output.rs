//! Markdown + JSON output writers.
//!
//! `RESULTS.md` matches the format used by `benchmarks/README.md`
//! so operators see one continuous shape across the legacy Bash
//! harness and `mnem bench`.

use std::fs;
use std::path::Path;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use crate::bench::Bench;
use crate::runner::BenchOutcome;
use crate::score::ScoreReport;

/// Write `RESULTS.md` summarising every outcome under `out`.
pub fn write_results_md(out: &Path, outcomes: &[BenchOutcome]) -> Result<()> {
    let mut s = String::new();
    s.push_str("# mnem-bench results\n\n");
    s.push_str("| Bench | Adapter | n | metric | value | runtime (s) |\n");
    s.push_str("|-------|---------|---|--------|------:|-----------:|\n");
    for o in outcomes {
        let bench_id = o.bench.metadata().id;
        let adapter_id = o.adapter.id();
        match &o.report {
            Some(r) => {
                // Headline picker: ConvoMem reports `avg_recall`,
                // every other shipped bench reports `recall@5`.
                let (metric, value) = if let Some(v) = r.overall.get("recall@5") {
                    ("recall@5", *v)
                } else if let Some(v) = r.overall.get("avg_recall") {
                    ("avg_recall", *v)
                } else {
                    ("--", 0.0)
                };
                s.push_str(&format!(
                    "| {bench_id} | {adapter_id} | {} | {metric} | {value:.4} | {:.1} |\n",
                    r.n_questions, r.runtime_seconds,
                ));
            }
            None => {
                s.push_str(&format!(
                    "| {bench_id} | {adapter_id} | -- | -- | -- | skipped: {} |\n",
                    o.skipped_reason,
                ));
            }
        }
    }
    s.push_str("\n");
    s.push_str("## Notes\n\n");
    s.push_str("- Benches: LongMemEval, LoCoMo, ConvoMem, MemBench (simple-roles + ");
    s.push_str("highlevel-movie), LongMemEval-hybrid-v4. All run against the in-process mnem adapter.\n");
    s.push_str("- Default embedder: ONNX MiniLM-L6-v2 (bundled, in-process).\n");
    s.push_str("  Pass `--embedder bag-of-tokens` for offline / CI runs that\n");
    s.push_str("  skip the ONNX model load (toy embedder; recall is not\n");
    s.push_str("  comparable to headline ONNX figures).\n");
    let p = out.join("RESULTS.md");
    fs::write(&p, s).with_context(|| format!("writing {}", p.display()))?;
    Ok(())
}

/// Re-render `RESULTS.md` from previously-written `<bench>.json`
/// files in `dir`. Used by `mnem bench results <dir>`.
pub fn rerender_from_dir(dir: &Path) -> Result<()> {
    let mut outcomes = Vec::new();
    for bench in Bench::all() {
        let id = bench.metadata().id;
        let p = dir.join(format!("{id}.json"));
        if !p.is_file() {
            continue;
        }
        let bytes = fs::read(&p).with_context(|| format!("reading {}", p.display()))?;
        let report: ScoreReport = serde_json::from_slice(&bytes)
            .with_context(|| format!("parsing {}", p.display()))?;
        // Best-effort adapter recovery: parse the harness id back.
        let adapter = crate::bench::AdapterKind::from_id(&report.adapter)
            .unwrap_or(crate::bench::AdapterKind::Mnem);
        outcomes.push(BenchOutcome {
            bench: *bench,
            adapter,
            report: Some(report),
            skipped_reason: String::new(),
        });
    }
    write_results_md(dir, &outcomes)
}

/// Sidecar shape used by `mnem bench list` so the JSON the CLI
/// prints is self-describing.
#[derive(Serialize, Deserialize)]
pub struct BenchListEntry {
    /// Stable identifier.
    pub id: &'static str,
    /// Display name.
    pub display: &'static str,
    /// Approximate ETA (seconds) for the full run.
    pub eta_seconds: u64,
    /// Approximate dataset size in bytes.
    pub dataset_bytes: u64,
    /// Description.
    pub description: &'static str,
}

/// Enumerate every bench as a serialisable struct (for
/// `mnem bench list --json`).
#[must_use]
pub fn list_benches() -> Vec<BenchListEntry> {
    Bench::all()
        .iter()
        .map(|b| {
            let m = b.metadata();
            BenchListEntry {
                id: m.id,
                display: m.display,
                eta_seconds: m.eta_seconds,
                dataset_bytes: m.dataset_bytes,
                description: m.description,
            }
        })
        .collect()
}
