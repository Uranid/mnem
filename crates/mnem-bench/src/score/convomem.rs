//! ConvoMem scorer.
//!
//! For each evidence item:
//!   1. `reset` the adapter (fresh corpus per item).
//!   2. Ingest every message in `conversations[*].messages` as a
//!      doc under label `ConvoMemI:<cat>:<idx>`. `external_id` is
//!      the message's flat index in the corpus.
//!   3. Retrieve top-K (oversampled to `max(500, top_k * 50)`)
//!      against `question`.
//!   4. recall_i = fraction of `message_evidences[*].text` that
//!      substring-match (either direction) any retrieved candidate.
//!
//! Final report: `overall.avg_recall` plus `by_category[<cat>].avg_recall`.

use std::collections::BTreeMap;
use std::path::Path;
use std::time::Instant;

use anyhow::{Result, anyhow};
use indicatif::{ProgressBar, ProgressStyle};

use crate::adapter::{BenchAdapter, IngestDoc};
use crate::datasets::convomem::EvidenceItem;
use crate::score::{PerQuestionRow, ScoreReport, TimingBreakdown};

/// Default oversampling factor (matches the Python adapter).
const RETRIEVE_LIMIT_DEFAULT: usize = 500;

/// Run ConvoMem against `adapter` over `items`. Returns the summary
/// + per-question rows.
pub fn run<A: BenchAdapter>(
    adapter: &mut A,
    items: &[EvidenceItem],
    top_k: usize,
    dataset_path: &Path,
) -> Result<(ScoreReport, Vec<PerQuestionRow>)> {
    let mut rows = Vec::with_capacity(items.len());
    let mut totals_by_cat: BTreeMap<String, u64> = BTreeMap::new();
    let mut recall_sum_by_cat: BTreeMap<String, f64> = BTreeMap::new();

    let mut t_ingest = 0f64;
    let mut t_retrieve = 0f64;
    let mut t_score = 0f64;
    let t0 = Instant::now();

    let mut all_recall_sum = 0f64;
    let mut all_recall_n = 0u64;

    let n_total = items.len();
    let pb = ProgressBar::new(n_total as u64);
    pb.set_style(
        ProgressStyle::with_template(
            " [{elapsed_precise}] {bar:32.cyan/blue} {pos}/{len} ({percent}%) ETA {eta} {msg}",
        )
        .unwrap()
        .progress_chars("=>-"),
    );
    pb.set_message("[convomem]");

    for (idx, item) in items.iter().enumerate() {
        pb.inc(1);
        let cat = if item.category_key.is_empty() {
            "unknown".to_string()
        } else {
            item.category_key.clone()
        };
        let label = format!("ConvoMemI:{cat}:{idx}");

        // Build the per-item evidence set up front; an empty
        // evidence list is recall = 1.0 (mirrors the Python adapter).
        let evidence: Vec<String> = item
            .message_evidences
            .iter()
            .map(|e| e.text.trim().to_lowercase())
            .filter(|s| !s.is_empty())
            .collect();
        if evidence.is_empty() {
            *totals_by_cat.entry(cat.clone()).or_default() += 1;
            *recall_sum_by_cat.entry(cat.clone()).or_default() += 1.0;
            all_recall_sum += 1.0;
            all_recall_n += 1;
            rows.push(PerQuestionRow {
                qid: format!("convomem#{idx}"),
                qtype: Some(cat.clone()),
                hit_at_5: 1,
                hit_at_10: 1,
                top5: Vec::new(),
                gold: Vec::new(),
            });
            continue;
        }

        // Build ingest docs.
        let mut docs: Vec<IngestDoc> = Vec::new();
        let mut corpus: Vec<String> = Vec::new();
        for conv in &item.conversations {
            for msg in &conv.messages {
                let text = msg.text.trim();
                if text.is_empty() {
                    continue;
                }
                let ext = format!("{idx}:{}", corpus.len());
                let mut props = serde_json::Map::new();
                props.insert(
                    "speaker".to_string(),
                    serde_json::Value::String(msg.speaker.clone()),
                );
                props.insert(
                    "idx".to_string(),
                    serde_json::Value::Number(serde_json::Number::from(corpus.len())),
                );
                docs.push(IngestDoc {
                    external_id: ext,
                    label: label.clone(),
                    text: text.to_string(),
                    props,
                });
                corpus.push(text.to_lowercase());
            }
        }
        if docs.is_empty() {
            *totals_by_cat.entry(cat.clone()).or_default() += 1;
            *recall_sum_by_cat.entry(cat.clone()).or_default() += 0.0;
            all_recall_n += 1;
            rows.push(PerQuestionRow {
                qid: format!("convomem#{idx}"),
                qtype: Some(cat.clone()),
                hit_at_5: 0,
                hit_at_10: 0,
                top5: Vec::new(),
                gold: evidence.clone(),
            });
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
            .retrieve(&label, &item.question, limit)
            .map_err(|e| anyhow!("adapter retrieve: {e}"))?;
        t_retrieve += _t.elapsed().as_secs_f64();

        let _t = Instant::now();
        // Recover the original text per hit via the external_id ->
        // corpus-index map.
        let mut retrieved_texts: Vec<&str> = Vec::with_capacity(hits.len());
        for h in &hits {
            // external_id = "<idx>:<corpus_index>"
            if let Some((_, n)) = h.external_id.split_once(':')
                && let Ok(n) = n.parse::<usize>()
                && n < corpus.len()
            {
                retrieved_texts.push(corpus[n].as_str());
            }
        }

        // Substring-match either direction.
        let mut found = 0u64;
        for ev in &evidence {
            for ret in &retrieved_texts {
                if ret.contains(ev) || ev.contains(ret) {
                    found += 1;
                    break;
                }
            }
        }
        let recall = found as f64 / evidence.len() as f64;
        t_score += _t.elapsed().as_secs_f64();

        *totals_by_cat.entry(cat.clone()).or_default() += 1;
        *recall_sum_by_cat.entry(cat.clone()).or_default() += recall;
        all_recall_sum += recall;
        all_recall_n += 1;

        // Repurpose hit_at_5 / hit_at_10 as boolean signals
        // (full-recall / any-recall). Keeps the JSONL row schema
        // homogeneous across benches.
        rows.push(PerQuestionRow {
            qid: format!("convomem#{idx}"),
            qtype: Some(cat.clone()),
            hit_at_5: u8::from(recall >= 1.0),
            hit_at_10: u8::from(recall > 0.0),
            top5: hits.iter().take(5).map(|h| h.external_id.clone()).collect(),
            gold: evidence,
        });
    }

    let mut overall = BTreeMap::new();
    let avg_recall = if all_recall_n > 0 {
        all_recall_sum / all_recall_n as f64
    } else {
        0.0
    };
    overall.insert("avg_recall".to_string(), avg_recall);
    overall.insert("n".to_string(), all_recall_n as f64);

    let mut by_category = BTreeMap::new();
    for (cat, n) in &totals_by_cat {
        if *n == 0 {
            continue;
        }
        let sum = recall_sum_by_cat.get(cat).copied().unwrap_or(0.0);
        let mut entry = BTreeMap::new();
        entry.insert("n".to_string(), *n as f64);
        entry.insert("avg_recall".to_string(), sum / *n as f64);
        by_category.insert(cat.clone(), entry);
    }

    pb.finish_and_clear();
    let elapsed = t0.elapsed().as_secs_f64();
    eprintln!("[convomem] done in {elapsed:.1}s ({n_total} items)");

    let report = ScoreReport {
        harness: "mnem-convomem".to_string(),
        adapter: adapter.name().to_string(),
        dataset: dataset_path.display().to_string(),
        n_questions: all_recall_n as usize,
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
