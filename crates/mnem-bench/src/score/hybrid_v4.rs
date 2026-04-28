//! LongMemEval hybrid-v4 scorer.
//!
//! Reuses the LongMemEval per-session pipeline but applies a
//! deterministic BM25-derived score boost over the retrieved
//! candidates BEFORE final top-K selection.
//!
//! Boost formula (mirrors `hybrid_v4_boost` in
//! `./benchmarks/adapters/mnem/longmemeval_session.py`):
//!
//!   score' = dense_score + boost_weight * overlap + bonus
//!
//! where
//!
//!   overlap = |q_tokens ∩ d_tokens| / max(|q_tokens|, 1)
//!   bonus   = 0.10 if "when/what year/..." in question and a date
//!             literal appears in the doc;
//!             0.05 if "how many/..." and any digit appears in the
//!             doc.
//!
//! `boost_weight` defaults to `0.3`.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::Path;
use std::time::Instant;

use anyhow::{Result, anyhow};
use indicatif::{ProgressBar, ProgressStyle};

use crate::adapter::{BenchAdapter, IngestDoc};
use crate::datasets::longmemeval::{Question, render_session};
use crate::score::longmemeval::SESSION_CHAR_CAP;
use crate::score::{PerQuestionRow, ScoreReport, TimingBreakdown};

/// Default hybrid-v4 boost weight (matches the upstream Python).
pub const DEFAULT_BOOST_WEIGHT: f32 = 0.3;

/// Run LongMemEval-hybrid-v4 against `adapter`. Same dataset, same
/// labelling, same retrieve as the plain LongMemEval scorer; only
/// the post-fusion ranking differs.
pub fn run<A: BenchAdapter>(
    adapter: &mut A,
    questions: &[Question],
    top_k: usize,
    dataset_path: &Path,
    boost_weight: f32,
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
    pb.set_message("[hybrid-v4]");

    for q in questions.iter() {
        pb.inc(1);
        if q.haystack_session_ids.is_empty()
            || q.haystack_sessions.is_empty()
            || q.haystack_session_ids.len() != q.haystack_sessions.len()
        {
            continue;
        }
        let label = format!("LmeQs:{}", q.question_id);

        // Build ingest docs AND retain a `external_id -> text` map
        // so the boost can score on the doc text without an
        // adapter-side text fetch.
        let mut docs: Vec<IngestDoc> = Vec::new();
        let mut text_by_ext: HashMap<String, String> = HashMap::new();
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
            text_by_ext.insert(sid.clone(), summary.clone());
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

        adapter.reset().map_err(|e| anyhow!("adapter reset: {e}"))?;
        let _t = Instant::now();
        adapter
            .ingest(&docs)
            .map_err(|e| anyhow!("adapter ingest: {e}"))?;
        t_ingest += _t.elapsed().as_secs_f64();

        let limit = top_k.max(10).max(500usize.min(docs.len() * 5));
        let _t = Instant::now();
        let mut hits = adapter
            .retrieve(&label, &q.question, limit)
            .map_err(|e| anyhow!("adapter retrieve: {e}"))?;
        t_retrieve += _t.elapsed().as_secs_f64();

        let _t = Instant::now();

        // ---- Hybrid-v4 boost ----
        let q_lower = q.question.to_lowercase();
        let q_tokens: HashSet<String> = tokenize(&q_lower);
        let want_date = looks_like_date_question(&q_lower);
        let want_num = looks_like_number_question(&q_lower);
        let q_token_n = q_tokens.len().max(1) as f32;
        for h in &mut hits {
            let Some(text) = text_by_ext.get(&h.external_id) else {
                continue;
            };
            let lower = text.to_lowercase();
            let d_tokens = tokenize(&lower);
            let overlap = q_tokens.intersection(&d_tokens).count() as f32 / q_token_n;
            let mut bonus = 0.0f32;
            if want_date && contains_date_literal(&lower) {
                bonus += 0.10;
            }
            if want_num && contains_digit(&lower) {
                bonus += 0.05;
            }
            h.score = h.score + boost_weight * overlap + bonus;
        }
        // Re-sort by boosted score, descending.
        hits.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let answer_set: HashSet<&str> = q.answer_session_ids.iter().map(String::as_str).collect();
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
    eprintln!("[hybrid-v4] done in {elapsed:.1}s ({n_total} items)");

    let report = ScoreReport {
        harness: "mnem-lme-hybrid-v4".to_string(),
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

/// Tokenize on `\w+`-equivalent runs of alphanumerics (ASCII-only,
/// like the Python regex `\w+` which we match in spirit). Lower-case
/// is applied by the caller.
fn tokenize(s: &str) -> HashSet<String> {
    let mut out = HashSet::new();
    let mut cur = String::new();
    for ch in s.chars() {
        if ch.is_alphanumeric() || ch == '_' {
            cur.push(ch);
        } else if !cur.is_empty() {
            out.insert(std::mem::take(&mut cur));
        }
    }
    if !cur.is_empty() {
        out.insert(cur);
    }
    out
}

/// Cheap "is the question asking about a date?" check.
fn looks_like_date_question(q_lower: &str) -> bool {
    q_lower.contains("when ")
        || q_lower.contains("what year")
        || q_lower.contains("what month")
        || q_lower.contains("date")
}

/// Cheap "is the question asking about a count / quantity?" check.
fn looks_like_number_question(q_lower: &str) -> bool {
    q_lower.contains("how many")
        || q_lower.contains("how much")
        || q_lower.contains("number of")
        || q_lower.contains("count")
}

/// Detect a coarse date literal (`20XX`, `M/D`, `D <month>`).
/// Conservative: we only need a strict-enough match to deliver the
/// 0.10 bonus when the document shape clearly looks date-like.
fn contains_date_literal(text: &str) -> bool {
    // 20XX year.
    if let Some(idx) = text.find("20") {
        let bytes = text.as_bytes();
        if idx + 4 <= bytes.len() {
            let yr = &bytes[idx + 2..idx + 4];
            if yr[0].is_ascii_digit() && yr[1].is_ascii_digit() {
                return true;
            }
        }
    }
    // M/D form: any digit followed by `/` followed by digit.
    let mut prev_is_digit = false;
    let mut saw_slash = false;
    for ch in text.chars() {
        if ch.is_ascii_digit() {
            if saw_slash {
                return true;
            }
            prev_is_digit = true;
        } else if ch == '/' && prev_is_digit {
            saw_slash = true;
        } else {
            prev_is_digit = false;
            saw_slash = false;
        }
    }
    // Month abbreviations.
    for m in [
        "jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec",
    ] {
        if text.contains(m) {
            return true;
        }
    }
    false
}

/// True iff `text` contains any ASCII digit. Cheap stand-in for the
/// Python `\b\d+\b` check.
fn contains_digit(text: &str) -> bool {
    text.chars().any(|c| c.is_ascii_digit())
}
