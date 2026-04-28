//! LongMemEval scorer (per-session chunking variant).
//!
//! For each question:
//!   1. `reset` the adapter (fresh haystack per question).
//!   2. Ingest one doc per session under label `LmeQs:<qid>`,
//!      with `external_id = session_id`.
//!   3. Retrieve top-K (default 50) for `question` under that label.
//!   4. Project hits back to session ids.
//!   5. R@5 / R@10 = any retrieved session id in `answer_session_ids`.
//!
//! Per-question rows + an aggregate JSON report are written to disk.

use std::collections::BTreeMap;
use std::path::Path;
use std::time::Instant;

use anyhow::{Result, anyhow};
use indicatif::{ProgressBar, ProgressStyle};

use crate::adapter::{BenchAdapter, IngestDoc};
use crate::datasets::longmemeval::{Question, render_session};
use crate::score::{PerQuestionRow, ScoreReport, TimingBreakdown};

/// Hard cap on per-session character payload before embedding. The
/// upstream Python used 8192; same number ports cleanly. The
/// renderer truncates to fit.
pub const SESSION_CHAR_CAP: usize = 8192;

/// Default oversampling factor for the retrieve top-K; mirrors the
/// upstream adapter's `max(500, top_k * 50)` rule. Capped at 500
/// because session counts in LongMemEval-S sit around 50-150 so
/// 500 is effectively "all of them".
pub const RETRIEVE_LIMIT_DEFAULT: usize = 500;

/// Run LongMemEval against `adapter` over `questions`. Returns the
/// summary + per-question rows; the runner writes them to disk.
pub fn run<A: BenchAdapter>(
    adapter: &mut A,
    questions: &[Question],
    top_k: usize,
    dataset_path: &Path,
) -> Result<(ScoreReport, Vec<PerQuestionRow>)> {
    let mut rows = Vec::with_capacity(questions.len());
    let mut totals_by_type: BTreeMap<String, u64> = BTreeMap::new();
    let mut hits5_by_type: BTreeMap<String, u64> = BTreeMap::new();
    let mut hits10_by_type: BTreeMap<String, u64> = BTreeMap::new();

    let mut t_ingest = 0f64;
    let mut t_retrieve = 0f64;
    let mut t_score = 0f64;

    let t0 = Instant::now();
    let n_total = questions.len();
    let pb = ProgressBar::new(n_total as u64);
    pb.set_style(
        ProgressStyle::with_template(
            " [{elapsed_precise}] {bar:32.cyan/blue} {pos}/{len} ({percent}%) ETA {eta} {msg}",
        )
        .unwrap()
        .progress_chars("=>-"),
    );
    pb.set_message("[longmemeval]");

    for q in questions.iter() {
        pb.inc(1);
        // Skip records that have neither sessions nor gold ids.
        if q.haystack_session_ids.is_empty()
            || q.haystack_sessions.is_empty()
            || q.haystack_session_ids.len() != q.haystack_sessions.len()
        {
            continue;
        }

        let label = format!("LmeQs:{}", q.question_id);

        // Build ingest docs.
        let mut docs = Vec::with_capacity(q.haystack_sessions.len());
        for (sid, turns) in q
            .haystack_session_ids
            .iter()
            .zip(q.haystack_sessions.iter())
        {
            let summary = render_session(turns, SESSION_CHAR_CAP);
            if summary.is_empty() {
                continue;
            }
            let mut props = serde_json::Map::new();
            props.insert(
                "session_id".to_string(),
                serde_json::Value::String(sid.clone()),
            );
            docs.push(IngestDoc {
                external_id: sid.clone(),
                label: label.clone(),
                text: summary,
                props,
            });
        }
        if docs.is_empty() {
            continue;
        }

        // Per-question fresh repo.
        let _t = Instant::now();
        adapter.reset().map_err(|e| anyhow!("adapter reset: {e}"))?;
        let _t2 = Instant::now();
        adapter
            .ingest(&docs)
            .map_err(|e| anyhow!("adapter ingest: {e}"))?;
        t_ingest += _t2.elapsed().as_secs_f64();

        let _t = Instant::now();
        let limit = top_k
            .max(10)
            .max(RETRIEVE_LIMIT_DEFAULT.min(docs.len() * 5));
        let hits = adapter
            .retrieve(&label, &q.question, limit)
            .map_err(|e| anyhow!("adapter retrieve: {e}"))?;
        t_retrieve += _t.elapsed().as_secs_f64();

        let _t = Instant::now();
        let answer_set: std::collections::HashSet<&str> =
            q.answer_session_ids.iter().map(String::as_str).collect();

        // Each hit IS a session id (session-chunking; no
        // turn->session collapse required).
        let ranked: Vec<String> = hits.into_iter().map(|h| h.external_id).collect();

        let hit5 = ranked
            .iter()
            .take(5)
            .any(|s| answer_set.contains(s.as_str()));
        let hit10 = ranked
            .iter()
            .take(10)
            .any(|s| answer_set.contains(s.as_str()));

        let qtype_owned = q
            .question_type
            .clone()
            .unwrap_or_else(|| "unknown".to_string());
        *totals_by_type.entry(qtype_owned.clone()).or_default() += 1;
        if hit5 {
            *hits5_by_type.entry(qtype_owned.clone()).or_default() += 1;
        }
        if hit10 {
            *hits10_by_type.entry(qtype_owned.clone()).or_default() += 1;
        }

        rows.push(PerQuestionRow {
            qid: q.question_id.clone(),
            qtype: q.question_type.clone(),
            hit_at_5: u8::from(hit5),
            hit_at_10: u8::from(hit10),
            top5: ranked.iter().take(5).cloned().collect(),
            gold: q.answer_session_ids.clone(),
        });
        t_score += _t.elapsed().as_secs_f64();
    }

    let total: u64 = totals_by_type.values().sum();
    let hits5_total: u64 = hits5_by_type.values().sum();
    let hits10_total: u64 = hits10_by_type.values().sum();
    let r5 = if total > 0 {
        hits5_total as f64 / total as f64
    } else {
        0.0
    };
    let r10 = if total > 0 {
        hits10_total as f64 / total as f64
    } else {
        0.0
    };

    let mut overall = BTreeMap::new();
    overall.insert("recall@5".to_string(), r5);
    overall.insert("recall@10".to_string(), r10);

    let mut by_category = BTreeMap::new();
    for (qt, n) in &totals_by_type {
        if *n == 0 {
            continue;
        }
        let h5 = hits5_by_type.get(qt).copied().unwrap_or(0);
        let h10 = hits10_by_type.get(qt).copied().unwrap_or(0);
        let mut entry = BTreeMap::new();
        entry.insert("n".to_string(), *n as f64);
        entry.insert("recall@5".to_string(), h5 as f64 / *n as f64);
        entry.insert("recall@10".to_string(), h10 as f64 / *n as f64);
        by_category.insert(qt.clone(), entry);
    }

    pb.finish_and_clear();
    let elapsed = t0.elapsed().as_secs_f64();
    eprintln!("[longmemeval] done in {elapsed:.1}s ({n_total} items)");

    let report = ScoreReport {
        harness: "mnem-lme-session".to_string(),
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
