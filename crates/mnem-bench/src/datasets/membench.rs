//! MemBench (`import-myself/Membench`) dataset spec + loader.
//!
//! Two scorers consume one dataset:
//!
//! - `membench-simple-roles`    -> `simple.json`   filtered by topic=`roles`
//! - `membench-highlevel-movie` -> `highlevel.json` filtered by topic=`movie`
//!
//! Source: HuggingFace `import-myself/Membench`,
//! `FirstAgent/<file>.json`. Each file is keyed by topic; values
//! are arrays of items with `tid`, `message_list`, `QA`.
//!
//! Cache layout (single shared dir for both scorers):
//!
//! ```text
//! ~/.mnem/bench-data/membench-simple-roles/simple.json
//! ~/.mnem/bench-data/membench-highlevel-movie/highlevel.json
//! ```
//!
//! We keep them under separate per-bench dirs so the
//! [`super::cached_path`] / `is_cached` machinery stays uniform with
//! LongMemEval / LoCoMo.

use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::DatasetSpec;
use crate::bench::Bench;

/// Spec for the `simple-roles` slice.
pub const SIMPLE_ROLES_SPEC: DatasetSpec = DatasetSpec {
    bench: Bench::MembenchSimpleRoles,
    filename: "simple.json",
    url: "https://huggingface.co/datasets/import-myself/Membench/resolve/main/FirstAgent/simple.json",
    sha256: "",
    bytes: 4 * 1024 * 1024,
};

/// Spec for the `highlevel-movie` slice.
pub const HIGHLEVEL_MOVIE_SPEC: DatasetSpec = DatasetSpec {
    bench: Bench::MembenchHighlevelMovie,
    filename: "highlevel.json",
    url: "https://huggingface.co/datasets/import-myself/Membench/resolve/main/FirstAgent/highlevel.json",
    sha256: "",
    bytes: 6 * 1024 * 1024,
};

/// One conversation turn from `message_list`.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Turn {
    /// User-facing message body. Other adapters fall back to `user`
    /// when this key is absent; we accept both via `serde(alias)`.
    #[serde(default, alias = "user")]
    pub user_message: String,
    /// Assistant reply (unused by the scorer, kept for completeness).
    #[serde(default, alias = "assistant")]
    pub assistant_message: String,
    /// Global step id within the item. The scorer uses this to
    /// match `target_step_id`.
    #[serde(default)]
    pub sid: Option<i64>,
    /// Fallback id field some shards use.
    #[serde(default)]
    pub mid: Option<i64>,
    /// Optional timestamp string ("2024-03-12 14:00").
    #[serde(default)]
    pub time: String,
    /// Optional location string.
    #[serde(default)]
    pub place: String,
}

/// Question-answer record. We only use `question` and
/// `target_step_id`.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Qa {
    /// Question text.
    #[serde(default)]
    pub question: String,
    /// Target step ids: list of `[sid, turn_idx]` pairs. The scorer
    /// counts a hit if any retrieved turn's `sid` matches one of
    /// these `sid`s.
    #[serde(default)]
    pub target_step_id: Vec<Value>,
}

/// One MemBench item (a `tid` plus its conversation + QA).
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Item {
    /// Item id (used for logging only).
    #[serde(default)]
    pub tid: i64,
    /// `message_list`. Either a flat list of turns or a list of
    /// session lists. [`flatten_turns`] handles both.
    #[serde(default, rename = "message_list")]
    pub message_list: Value,
    /// Question + ground truth.
    #[serde(rename = "QA")]
    pub qa: Qa,
    /// Filled in by the loader, never present upstream.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub category: String,
    /// Filled in by the loader.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub topic: String,
}

/// Flatten a `message_list` value into `[(global_idx, s_idx, t_idx, turn), ...]`.
/// Accepts either a flat list of turn dicts or a list of session
/// lists (both shapes appear in the wild).
#[must_use]
pub fn flatten_turns(message_list: &Value) -> Vec<(usize, usize, usize, Turn)> {
    let arr = match message_list.as_array() {
        Some(a) => a,
        None => return Vec::new(),
    };
    if arr.is_empty() {
        return Vec::new();
    }
    let sessions: Vec<&[Value]> = if arr.first().map(|v| v.is_object()).unwrap_or(false) {
        // Flat list of turns.
        vec![arr.as_slice()]
    } else {
        arr.iter()
            .filter_map(|v| v.as_array().map(|a| a.as_slice()))
            .collect()
    };
    let mut flat = Vec::new();
    let mut g = 0usize;
    for (s_idx, sess) in sessions.iter().enumerate() {
        for (t_idx, raw) in sess.iter().enumerate() {
            if let Ok(turn) = serde_json::from_value::<Turn>(raw.clone()) {
                flat.push((g, s_idx, t_idx, turn));
                g += 1;
            }
        }
    }
    flat
}

/// Render a turn into the natural-language string the embedder
/// consumes. Mirrors `render_turn` in the Python adapter:
/// `[<time>] <user_message> (@<place>)` with both prefix/suffix
/// dropped when empty.
#[must_use]
pub fn render_turn(turn: &Turn) -> String {
    let user = turn.user_message.trim();
    let prefix = if turn.time.is_empty() {
        String::new()
    } else {
        format!("[{}] ", turn.time)
    };
    let suffix = if turn.place.is_empty() {
        String::new()
    } else {
        format!(" (@{})", turn.place)
    };
    format!("{prefix}{user}{suffix}")
}

/// Resolved sid for a turn: prefers `sid`, then `mid`, then the
/// global index `g` as a fallback. Mirrors the Python adapter.
#[must_use]
pub fn turn_sid(g: usize, turn: &Turn) -> i64 {
    turn.sid
        .or(turn.mid)
        .unwrap_or_else(|| i64::try_from(g).unwrap_or(i64::MAX))
}

/// Load a MemBench file, filtered to the requested topic. The
/// upstream JSON is `{ "<topic>": [item, ...], ... }`; we walk every
/// topic key and keep items whose key matches `topic` (or all if
/// `topic` is `None`). The optional category tag is propagated onto
/// every item so the scorer can label its rows.
pub fn load_filtered(
    path: &Path,
    category: &str,
    topic: Option<&str>,
) -> Result<Vec<Item>> {
    let bytes = fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    let by_topic: BTreeMap<String, Vec<Item>> = serde_json::from_slice(&bytes)
        .with_context(|| format!("parsing {}", path.display()))?;
    let mut out = Vec::new();
    for (k, items) in by_topic {
        if let Some(want) = topic
            && k != want
        {
            continue;
        }
        for mut it in items {
            it.category = category.to_string();
            it.topic = k.clone();
            out.push(it);
        }
    }
    Ok(out)
}
