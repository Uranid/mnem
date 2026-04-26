//! LoCoMo (Snap 2024) dataset spec + loader.
//!
//! Source: snap-research/LoCoMo `locomo10.json`. ~3 MB.
//! Cached at `~/.mnem/bench-data/locomo/locomo10.json`.

use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

use anyhow::{Context, Result};
use serde::Deserialize;
use serde_json::Value;

use super::DatasetSpec;
use crate::bench::Bench;

/// Static spec. sha256 left empty (upstream raw blob is not pinned).
pub const SPEC: DatasetSpec = DatasetSpec {
    bench: Bench::Locomo,
    filename: "locomo10.json",
    url: "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json",
    sha256: "",
    bytes: 3 * 1024 * 1024,
};

/// One LoCoMo conversation record.
#[derive(Clone, Debug, Deserialize)]
pub struct Conversation {
    /// Stable id used to scope ingest. Falls back to "conv_<idx>"
    /// in the loader if absent.
    #[serde(default)]
    pub sample_id: Option<String>,
    /// QA pairs evaluated against this conversation.
    #[serde(default)]
    pub qa: Vec<Qa>,
    /// Conversation body, as a free-form JSON object whose keys
    /// look like `session_1`, `session_1_date_time`, `session_2`,
    /// ... See [`iter_sessions`] for the helper that walks them.
    #[serde(default)]
    pub conversation: BTreeMap<String, Value>,
}

/// One QA pair for one conversation.
#[derive(Clone, Debug, Deserialize)]
pub struct Qa {
    /// Question text.
    #[serde(default)]
    pub question: String,
    /// Reference answer (unused by retrieval scoring; upstream JSON
    /// mixes strings and integers, so we keep it as a free-form Value).
    #[serde(default)]
    pub answer: Value,
    /// List of dialog ids (e.g. `"D1:3"`) that contain the
    /// evidence.
    #[serde(default)]
    pub evidence: Vec<String>,
    /// Numeric category id; mapped to a label by the scorer.
    #[serde(default)]
    pub category: u32,
}

/// One dialog turn: `(speaker, text, dia_id)`.
#[derive(Clone, Debug, Deserialize)]
pub struct Dialog {
    /// Free-form speaker id.
    #[serde(default)]
    pub speaker: String,
    /// Dialog text.
    #[serde(default)]
    pub text: String,
    /// Dialog id, e.g. `"D2:5"`. Used for evidence matching.
    #[serde(default)]
    pub dia_id: String,
}

/// Category id -> label mapping. Mirrors the upstream Python.
#[must_use]
pub fn category_name(c: u32) -> &'static str {
    match c {
        1 => "single-hop",
        2 => "multi-hop",
        3 => "open-domain",
        4 => "temporal",
        5 => "common-sense",
        6 => "adversarial",
        _ => "unknown",
    }
}

/// Load the entire LoCoMo file.
pub fn load(path: &Path) -> Result<Vec<Conversation>> {
    let bytes = fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    serde_json::from_slice(&bytes).with_context(|| format!("parsing {}", path.display()))
}

/// Walk session keys (`session_1`, `session_2`, ...) in order.
/// Yields `(session_index, date_string, dialogs)`.
pub fn iter_sessions(
    conv: &BTreeMap<String, Value>,
) -> impl Iterator<Item = (usize, String, Vec<Dialog>)> + '_ {
    SessionIter { conv, idx: 1 }
}

struct SessionIter<'a> {
    conv: &'a BTreeMap<String, Value>,
    idx: usize,
}

impl Iterator for SessionIter<'_> {
    type Item = (usize, String, Vec<Dialog>);

    fn next(&mut self) -> Option<Self::Item> {
        let key = format!("session_{}", self.idx);
        let raw = self.conv.get(&key)?;
        let date = self
            .conv
            .get(&format!("{key}_date_time"))
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        let dialogs: Vec<Dialog> = match raw {
            Value::Array(arr) => arr
                .iter()
                .filter_map(|v| serde_json::from_value(v.clone()).ok())
                .collect(),
            _ => Vec::new(),
        };
        let i = self.idx;
        self.idx += 1;
        Some((i, date, dialogs))
    }
}
