//! Interactive TUI driven by `dialoguer`. Surfaces every option the
//! 0.1.0 harness supports and routes to the runner.

use std::path::PathBuf;

use anyhow::{Result, anyhow};
use dialoguer::{Confirm, Input, MultiSelect, Select};

use crate::bench::{AdapterKind, Bench, EmbedderChoice, RunMode};
use crate::runner::RunPlan;

/// Drive the TUI and return a [`RunPlan`] the runner can execute.
/// Returns `Ok(None)` if the user backed out of the wizard.
pub fn run_tui(default_out: &str) -> Result<Option<RunPlan>> {
    eprintln!("mnem bench - 0.1.0 interactive setup");
    eprintln!();

    // 1. Multi-select benches. All checked by default.
    let bench_items: Vec<String> = Bench::all().iter().map(|b| b.metadata().display.to_string()).collect();
    let defaults: Vec<bool> = Bench::all().iter().map(|_| true).collect();
    let bench_choices = MultiSelect::new()
        .with_prompt("Benchmarks (space to toggle, enter to confirm)")
        .items(&bench_items)
        .defaults(&defaults)
        .interact()
        .map_err(|e| anyhow!("dialoguer: {e}"))?;
    let benches: Vec<Bench> = bench_choices.into_iter().map(|i| Bench::all()[i]).collect();
    if benches.is_empty() {
        eprintln!("[mnem bench] no benches selected; exiting.");
        return Ok(None);
    }

    // 2. Multi-select systems-under-test. All checked by default.
    let adapter_items: Vec<String> = AdapterKind::all().iter().map(|a| a.display().to_string()).collect();
    let adapter_defaults: Vec<bool> = AdapterKind::all().iter().map(|_| true).collect();
    let adapter_choices = MultiSelect::new()
        .with_prompt("Systems-under-test")
        .items(&adapter_items)
        .defaults(&adapter_defaults)
        .interact()
        .map_err(|e| anyhow!("dialoguer: {e}"))?;
    let adapters: Vec<AdapterKind> = adapter_choices
        .into_iter()
        .map(|i| AdapterKind::all()[i])
        .collect();
    if adapters.is_empty() {
        eprintln!("[mnem bench] no adapters selected; exiting.");
        return Ok(None);
    }

    // 3. Run mode (single-select).
    let mode_items: Vec<String> = RunMode::all().iter().map(|m| m.display().to_string()).collect();
    let mode_idx = Select::new()
        .with_prompt("Run mode")
        .items(&mode_items)
        .default(0)
        .interact()
        .map_err(|e| anyhow!("dialoguer: {e}"))?;
    let mode = RunMode::all()[mode_idx];

    // 4. Embedder (single-select). Default = first listed (ONNX MiniLM).
    let emb_items: Vec<String> = EmbedderChoice::all().iter().map(|e| e.display().to_string()).collect();
    let emb_idx = Select::new()
        .with_prompt("Embedder")
        .items(&emb_items)
        .default(0)
        .interact()
        .map_err(|e| anyhow!("dialoguer: {e}"))?;
    let embedder = EmbedderChoice::all()[emb_idx];

    // 5. Top-K.
    let top_k: usize = Input::new()
        .with_prompt("Top-K")
        .default(10usize)
        .interact_text()
        .map_err(|e| anyhow!("dialoguer: {e}"))?;

    // 6. Output dir.
    let out_str: String = Input::new()
        .with_prompt("Output directory")
        .default(default_out.to_string())
        .interact_text()
        .map_err(|e| anyhow!("dialoguer: {e}"))?;
    let out = PathBuf::from(out_str);

    // 7. Skip cached download?
    let skip_cached = Confirm::new()
        .with_prompt("Use cached datasets when available?")
        .default(true)
        .interact()
        .map_err(|e| anyhow!("dialoguer: {e}"))?;

    Ok(Some(RunPlan {
        benches,
        adapters,
        mode,
        embedder,
        out,
        top_k,
        limit: None,
        no_cache: !skip_cached,
        quiet: false,
    }))
}
