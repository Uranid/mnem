//! MemBench scorer.
//!
//! One scorer module, two bench-id entry points:
//!
//! - [`run_simple_roles`]    -> `simple.json` filtered by topic=`roles`
//! - [`run_highlevel_movie`] -> `highlevel.json` filtered by topic=`movie`
//!
//! Per item:
//!   1. `reset` the adapter.
//!   2. Flatten `message_list` into per-turn docs under label
//!      `MemBenchI:<cat>:<topic>:<idx>:<tid>`. `external_id` = the
//!      turn's `sid` (or `mid`/global-index fallback).
//!   3. Retrieve top-K (oversampled) against `QA.question`.
//!   4. Hit if any retrieved turn's `sid` appears in the gold
//!      `target_step_id[*][0]` set.
//!
//! Final report: `overall.recall@K` plus per-tid breakdown.

use std::collections::BTreeMap;
use std::path::Path;
use std::time::Instant;

use anyhow::{Result, anyhow};
use indicatif::{ProgressBar, ProgressStyle};
use serde_json::Value;

use crate::adapter::{BenchAdapter, IngestDoc};
use crate::datasets::membench::{Item, flatten_turns, render_turn, turn_sid};
use crate::score::{PerQuestionRow, ScoreReport, TimingBreakdown};

/// Default per-turn character cap for the embedded summary text.
const TURN_CHAR_CAP: usize = 1024;

/// Default oversample (matches the Python adapter).
const RETRIEVE_LIMIT_DEFAULT: usize = 500;

/// Run the simple/roles slice. Thin wrapper around [`run`] with a
/// fixed harness id.
pub fn run_simple_roles<A: BenchAdapter>(
    adapter: &mut A,
    items: &[Item],
    top_k: usize,
    dataset_path: &Path,
) -> Result<(ScoreReport, Vec<PerQuestionRow>)> {
    run(
        adapter,
        items,
        top_k,
        dataset_path,
        "mnem-membench-simple-roles",
    )
}

/// Run the highlevel/movie slice.
pub fn run_highlevel_movie<A: BenchAdapter>(
    adapter: &mut A,
    items: &[Item],
    top_k: usize,
    dataset_path: &Path,
) -> Result<(ScoreReport, Vec<PerQuestionRow>)> {
    run(
        adapter,
        items,
        top_k,
        dataset_path,
        "mnem-membench-highlevel-movie",
    )
}

/// Shared scoring core. `harness` is the free-form id stamped on
/// the [`ScoreReport`].
fn run<A: BenchAdapter>(
    adapter: &mut A,
    items: &[Item],
    top_k: usize,
    dataset_path: &Path,
    harness: &str,
) -> Result<(ScoreReport, Vec<PerQuestionRow>)> {
    let mut rows = Vec::with_capacity(items.len());
    let mut totals_by_topic: BTreeMap<String, u64> = BTreeMap::new();
    let mut hits_by_topic: BTreeMap<String, u64> = BTreeMap::new();
    let mut t_ingest = 0f64;
    let mut t_retrieve = 0f64;
    let mut t_score = 0f64;
    let t0 = Instant::now();
    let mut total: u64 = 0;
    let mut hits_total: u64 = 0;

    // Use a short bench id (`membench-simple-roles`) for the progress
    // tag so log lines stay scannable.
    let progress_tag = harness.strip_prefix("mnem-").unwrap_or(harness);
    let n_total = items.len();
    let pb = ProgressBar::new(n_total as u64);
    pb.set_style(
        ProgressStyle::with_template(
            " [{elapsed_precise}] {bar:32.cyan/blue} {pos}/{len} ({percent}%) ETA {eta} {msg}",
        )
        .unwrap()
        .progress_chars("=>-"),
    );
    pb.set_message(format!("[{progress_tag}]"));

    for (idx, it) in items.iter().enumerate() {
        pb.inc(1);
        let cat = if it.category.is_empty() {
            "unknown".to_string()
        } else {
            it.category.clone()
        };
        let topic = if it.topic.is_empty() {
            "unknown".to_string()
        } else {
            it.topic.clone()
        };
        let label = format!("MemBenchI:{cat}:{topic}:{idx}:{}", it.tid);

        // Build target sid set from QA.target_step_id.
        let target_sids: std::collections::HashSet<i64> = it
            .qa
            .target_step_id
            .iter()
            .filter_map(|pair| match pair {
                Value::Array(arr) if !arr.is_empty() => arr[0].as_i64(),
                _ => None,
            })
            .collect();

        // Flatten turns.
        let flat = flatten_turns(&it.message_list);
        if flat.is_empty() || it.qa.question.is_empty() {
            continue;
        }

        let mut docs: Vec<IngestDoc> = Vec::with_capacity(flat.len());
        for (g, s_idx, t_idx, turn) in &flat {
            let sid = turn_sid(*g, turn);
            let mut summary = render_turn(turn);
            if summary.len() > TURN_CHAR_CAP {
                let mut end = TURN_CHAR_CAP;
                while end > 0 && !summary.is_char_boundary(end) {
                    end -= 1;
                }
                summary.truncate(end);
            }
            let mut props = serde_json::Map::new();
            props.insert(
                "sid".to_string(),
                serde_json::Value::Number(serde_json::Number::from(sid)),
            );
            props.insert(
                "s_idx".to_string(),
                serde_json::Value::Number(serde_json::Number::from(*s_idx)),
            );
            props.insert(
                "t_idx".to_string(),
                serde_json::Value::Number(serde_json::Number::from(*t_idx)),
            );
            docs.push(IngestDoc {
                // Encode the sid as the external id; the scorer
                // recovers it directly from the hit.
                external_id: sid.to_string(),
                label: label.clone(),
                text: summary,
                props,
            });
        }
        if docs.is_empty() {
            continue;
        }

        adapter.reset().map_err(|e| anyhow!("adapter reset: {e}"))?;
        let _t = Instant::now();
        adapter
            .ingest(&docs)
            .map_err(|e| anyhow!("adapter ingest: {e}"))?;
        t_ingest += _t.elapsed().as_secs_f64();

        let limit = top_k
            .max(10)
            .max(RETRIEVE_LIMIT_DEFAULT.min(docs.len() * 5));
        let _t = Instant::now();
        let hits = adapter
            .retrieve(&label, &it.qa.question, limit)
            .map_err(|e| anyhow!("adapter retrieve: {e}"))?;
        t_retrieve += _t.elapsed().as_secs_f64();

        let _t = Instant::now();
        // Recover top-K unique sids in rank order.
        let mut top_sids: Vec<i64> = Vec::with_capacity(top_k);
        let mut seen: std::collections::HashSet<i64> = std::collections::HashSet::new();
        for h in &hits {
            let Ok(s) = h.external_id.parse::<i64>() else {
                continue;
            };
            if !seen.insert(s) {
                continue;
            }
            top_sids.push(s);
            if top_sids.len() >= top_k {
                break;
            }
        }
        let hit_at_k = !target_sids.is_empty() && top_sids.iter().any(|s| target_sids.contains(s));
        t_score += _t.elapsed().as_secs_f64();

        total += 1;
        if hit_at_k {
            hits_total += 1;
        }
        *totals_by_topic.entry(topic.clone()).or_default() += 1;
        if hit_at_k {
            *hits_by_topic.entry(topic.clone()).or_default() += 1;
        }

        rows.push(PerQuestionRow {
            qid: format!("{topic}#{}", it.tid),
            qtype: Some(topic.clone()),
            hit_at_5: u8::from(hit_at_k),
            hit_at_10: u8::from(hit_at_k),
            top5: top_sids.iter().take(5).map(i64::to_string).collect(),
            gold: target_sids.iter().map(i64::to_string).collect(),
        });
    }

    let recall_at_k = if total > 0 {
        hits_total as f64 / total as f64
    } else {
        0.0
    };
    let mut overall = BTreeMap::new();
    let key = format!("recall@{top_k}");
    overall.insert(key.clone(), recall_at_k);
    // Mirror under the standard recall@5 / recall@10 keys so the
    // RESULTS.md table renders without bespoke handling.
    if top_k != 5 {
        overall.insert("recall@5".to_string(), recall_at_k);
    }
    overall.insert("recall@10".to_string(), recall_at_k);

    let mut by_category = BTreeMap::new();
    for (topic, n) in &totals_by_topic {
        if *n == 0 {
            continue;
        }
        let h = hits_by_topic.get(topic).copied().unwrap_or(0);
        let mut entry = BTreeMap::new();
        entry.insert("n".to_string(), *n as f64);
        entry.insert(key.clone(), h as f64 / *n as f64);
        by_category.insert(topic.clone(), entry);
    }

    pb.finish_and_clear();
    let elapsed = t0.elapsed().as_secs_f64();
    eprintln!("[{progress_tag}] done in {elapsed:.1}s ({n_total} items)");

    let report = ScoreReport {
        harness: harness.to_string(),
        adapter: adapter.name().to_string(),
        dataset: dataset_path.display().to_string(),
        n_questions: total as usize,
        runtime_seconds: elapsed,
        timing: TimingBreakdown {
            ingest_s: t_ingest,
            retrieve_s: t_retrieve,
            score_s: t_score,
        },
        overall,
        by_category,
    };
    Ok((report, rows))
}
