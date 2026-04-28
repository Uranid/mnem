//! LongMemEval-S (per-session-cleaned) dataset spec + loader.
//!
//! Source: HuggingFace `xiaowu0162/longmemeval-cleaned` repo, file
//! `longmemeval_s_cleaned.json` (single JSON, not JSONL).
//! ~264 MB. The cached copy lives at
//! `~/.mnem/bench-data/longmemeval/longmemeval_s_cleaned.json`.

use std::fs;
use std::path::Path;

use anyhow::{Context, Result};
use serde::Deserialize;

use super::DatasetSpec;
use crate::bench::Bench;

/// Static spec. `sha256` left empty: the 264MB upstream blob is not
/// digest-pinned. Anyone bypassing via `MNEM_BENCH_DATA` hits the
/// same accept path.
pub const SPEC: DatasetSpec = DatasetSpec {
    bench: Bench::LongMemEval,
    filename: "longmemeval_s_cleaned.json",
    url: "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json",
    sha256: "",
    bytes: 264 * 1024 * 1024,
};

/// One session of one question. Mirrors the LongMemEval-S record
/// shape closely enough that the same JSON loads against either
/// the per-turn or per-session adapter.
#[derive(Clone, Debug, Deserialize)]
pub struct Question {
    /// Stable question id.
    pub question_id: String,
    /// Question category. Optional in some splits.
    #[serde(default)]
    pub question_type: Option<String>,
    /// The question itself.
    pub question: String,
    /// Set of session ids that the gold answer lives in.
    #[serde(default)]
    pub answer_session_ids: Vec<String>,
    /// Haystack session ids, parallel to `haystack_sessions`.
    #[serde(default)]
    pub haystack_session_ids: Vec<String>,
    /// Per-session conversation turns.
    /// Each session is a list of `{role, content}` turns. We only
    /// concatenate user-role content (matches the upstream
    /// per-session adapter).
    #[serde(default)]
    pub haystack_sessions: Vec<Vec<Turn>>,
}

/// One conversational turn inside a session.
#[derive(Clone, Debug, Deserialize)]
pub struct Turn {
    /// `"user"` / `"assistant"` etc. The session-rendering helper
    /// keeps only `"user"` turns.
    #[serde(default)]
    pub role: String,
    /// Free-form turn content.
    #[serde(default)]
    pub content: String,
}

/// Load + parse the LongMemEval JSON at `path`. Accepts either a
/// JSON array of questions or a JSON-Lines file (one question per
/// line); both are observed in the wild.
pub fn load(path: &Path) -> Result<Vec<Question>> {
    let bytes = fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    // Try array first.
    if let Ok(v) = serde_json::from_slice::<Vec<Question>>(&bytes) {
        return Ok(v);
    }
    // Fallback to JSON-Lines.
    let text = std::str::from_utf8(&bytes).context("longmemeval file is not utf-8")?;
    let mut out = Vec::new();
    for (lineno, line) in text.lines().enumerate() {
        let trim = line.trim();
        if trim.is_empty() {
            continue;
        }
        let q: Question = serde_json::from_str(trim)
            .with_context(|| format!("parsing line {} of {}", lineno + 1, path.display()))?;
        out.push(q);
    }
    Ok(out)
}

/// Concatenate user-role turns into the per-session string the
/// embedder consumes. Mirrors the upstream Python adapter:
///
/// > only `role == "user"` turns; non-empty strips; joined on `\n`;
/// > truncated to `cap` characters if `cap > 0`.
#[must_use]
pub fn render_session(turns: &[Turn], cap: usize) -> String {
    let mut lines: Vec<&str> = Vec::with_capacity(turns.len());
    for t in turns {
        if t.role != "user" {
            continue;
        }
        let s = t.content.trim();
        if !s.is_empty() {
            lines.push(s);
        }
    }
    let s = lines.join("\n");
    if cap > 0 && s.len() > cap {
        let mut end = cap;
        while end > 0 && !s.is_char_boundary(end) {
            end -= 1;
        }
        s[..end].to_string()
    } else {
        s
    }
}
