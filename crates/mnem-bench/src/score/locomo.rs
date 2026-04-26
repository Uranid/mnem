//! LoCoMo scorer (session-granularity).
//!
//! For each conversation:
//!   1. `reset` the adapter.
//!   2. Ingest one doc per dialog turn under label
//!      `LoCoMoC:<sample_id>`. `external_id` = `dia_id`. We also
//!      stash `session_<idx>` in props so the session-granularity
//!      MAX-aggregate has a key to roll up to.
//!   3. For each QA, retrieve top-K. Aggregate scores MAX per
//!      session, then take top-5 / top-10 sessions.
//!   4. Hit if any of the gold dialog ids in `evidence` belongs to
//!      a retrieved session id.

use std::collections::{BTreeMap, HashMap};
use std::path::Path;
use std::time::Instant;

use anyhow::{Result, anyhow};
use indicatif::{ProgressBar, ProgressStyle};

use crate::adapter::{BenchAdapter, IngestDoc};
use crate::datasets::locomo::{Conversation, category_name, iter_sessions};
use crate::score::{PerQuestionRow, ScoreReport, TimingBreakdown};

/// Hard cap on per-turn payload (matches the upstream Python
/// `--summary-char-cap 2000`).
pub const TURN_CHAR_CAP: usize = 2000;

/// Run LoCoMo against `adapter` over `conversations`.
pub fn run<A: BenchAdapter>(
    adapter: &mut A,
    conversations: &[Conversation],
    top_k: usize,
    dataset_path: &Path,
) -> Result<(ScoreReport, Vec<PerQuestionRow>)> {
    let mut rows = Vec::new();
    let mut totals_by_cat: BTreeMap<String, u64> = BTreeMap::new();
    let mut hits5_by_cat: BTreeMap<String, u64> = BTreeMap::new();
    let mut hits10_by_cat: BTreeMap<String, u64> = BTreeMap::new();

    let mut t_ingest = 0f64;
    let mut t_retrieve = 0f64;
    let mut t_score = 0f64;

    let t0 = Instant::now();
    let n_total = conversations.len();
    let pb = ProgressBar::new(n_total as u64);
    pb.set_style(
        ProgressStyle::with_template(
            " [{elapsed_precise}] {bar:32.cyan/blue} {pos}/{len} ({percent}%) ETA {eta} {msg}",
        )
        .unwrap()
        .progress_chars("=>-"),
    );
    pb.set_message("[locomo]");

    for (ci, conv) in conversations.iter().enumerate() {
        pb.inc(1);
        let sample_id = conv.sample_id.clone().unwrap_or_else(|| format!("conv_{ci}"));
        let label = format!("LoCoMoC:{sample_id}");

        // Build dialog->session lookup AND ingest docs.
        let mut docs: Vec<IngestDoc> = Vec::new();
        let mut dia_to_session: HashMap<String, String> = HashMap::new();
        for (sidx, _date, dialogs) in iter_sessions(&conv.conversation) {
            let skey = format!("session_{sidx}");
            for d in dialogs {
                if d.dia_id.is_empty() {
                    continue;
                }
                let speaker = if d.speaker.is_empty() { "speaker".to_string() } else { d.speaker };
                let text = d.text.trim();
                if text.is_empty() {
                    continue;
                }
                let truncated = if text.len() > TURN_CHAR_CAP {
                    let mut end = TURN_CHAR_CAP;
                    while end > 0 && !text.is_char_boundary(end) {
                        end -= 1;
                    }
                    &text[..end]
                } else {
                    text
                };
                let summary = format!("{speaker}: {truncated}");
                let mut props = serde_json::Map::new();
                props.insert("dia_id".to_string(), serde_json::Value::String(d.dia_id.clone()));
                props.insert("session".to_string(), serde_json::Value::String(skey.clone()));
                docs.push(IngestDoc {
                    external_id: d.dia_id.clone(),
                    label: label.clone(),
                    text: summary,
                    props,
                });
                dia_to_session.insert(d.dia_id.clone(), skey.clone());
            }
        }
        if docs.is_empty() {
            continue;
        }

        adapter
            .reset()
            .map_err(|e| anyhow!("adapter reset: {e}"))?;
        let _t = Instant::now();
        adapter
            .ingest(&docs)
            .map_err(|e| anyhow!("adapter ingest: {e}"))?;
        t_ingest += _t.elapsed().as_secs_f64();

        // Per-QA retrieve + score.
        for q in &conv.qa {
            let cat = category_name(q.category).to_string();
            let ev_sessions: std::collections::HashSet<String> = q
                .evidence
                .iter()
                .filter_map(|d| dia_to_session.get(d).cloned())
                .collect();

            // Skip questions whose evidence cannot be located in
            // this conv (the upstream Python silently skipped these
            // too).
            if ev_sessions.is_empty() {
                continue;
            }

            let limit = top_k.max(10).max(50);
            let _t = Instant::now();
            let hits = adapter
                .retrieve(&label, &q.question, limit)
                .map_err(|e| anyhow!("adapter retrieve: {e}"))?;
            t_retrieve += _t.elapsed().as_secs_f64();

            let _t = Instant::now();
            // MAX-aggregate dialog scores up to session keys.
            let mut session_scores: HashMap<String, f32> = HashMap::new();
            for h in &hits {
                let Some(sk) = dia_to_session.get(&h.external_id) else {
                    continue;
                };
                let cur = session_scores.entry(sk.clone()).or_insert(f32::MIN);
                if h.score > *cur {
                    *cur = h.score;
                }
            }
            let mut ranked: Vec<(String, f32)> = session_scores.into_iter().collect();
            ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            let ranked_keys: Vec<String> = ranked.into_iter().map(|(k, _)| k).collect();

            let hit5 = ranked_keys.iter().take(5).any(|s| ev_sessions.contains(s));
            let hit10 = ranked_keys.iter().take(10).any(|s| ev_sessions.contains(s));

            *totals_by_cat.entry(cat.clone()).or_default() += 1;
            if hit5 {
                *hits5_by_cat.entry(cat.clone()).or_default() += 1;
            }
            if hit10 {
                *hits10_by_cat.entry(cat.clone()).or_default() += 1;
            }

            let qid = format!("{sample_id}#{}", rows.len());
            rows.push(PerQuestionRow {
                qid,
                qtype: Some(cat),
                hit_at_5: u8::from(hit5),
                hit_at_10: u8::from(hit10),
                top5: ranked_keys.iter().take(5).cloned().collect(),
                gold: ev_sessions.into_iter().collect(),
            });
            t_score += _t.elapsed().as_secs_f64();
        }
    }

    let total: u64 = totals_by_cat.values().sum();
    let hits5_total: u64 = hits5_by_cat.values().sum();
    let hits10_total: u64 = hits10_by_cat.values().sum();
    let r5 = if total > 0 { hits5_total as f64 / total as f64 } else { 0.0 };
    let r10 = if total > 0 { hits10_total as f64 / total as f64 } else { 0.0 };

    let mut overall = BTreeMap::new();
    overall.insert("recall@5".to_string(), r5);
    overall.insert("recall@10".to_string(), r10);

    let mut by_category = BTreeMap::new();
    for (cat, n) in &totals_by_cat {
        if *n == 0 {
            continue;
        }
        let h5 = hits5_by_cat.get(cat).copied().unwrap_or(0);
        let h10 = hits10_by_cat.get(cat).copied().unwrap_or(0);
        let mut entry = BTreeMap::new();
        entry.insert("n".to_string(), *n as f64);
        entry.insert("recall@5".to_string(), h5 as f64 / *n as f64);
        entry.insert("recall@10".to_string(), h10 as f64 / *n as f64);
        by_category.insert(cat.clone(), entry);
    }

    pb.finish_and_clear();
    let elapsed = t0.elapsed().as_secs_f64();
    eprintln!("[locomo] done in {elapsed:.1}s ({n_total} conversations)");

    let report = ScoreReport {
        harness: "mnem-locomo".to_string(),
        adapter: adapter.name().to_string(),
        dataset: dataset_path.display().to_string(),
        n_questions: total as usize,
        runtime_seconds: elapsed,
        timing: TimingBreakdown { ingest_s: t_ingest, retrieve_s: t_retrieve, score_s: t_score },
        overall,
        by_category,
    };
    Ok((report, rows))
}
