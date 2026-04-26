//! Scoring engines per benchmark.
//!
//! Ships [`longmemeval`], [`locomo`], [`convomem`], [`membench`], and
//! [`hybrid_v4`].

pub mod convomem;
pub mod hybrid_v4;
pub mod locomo;
pub mod longmemeval;
pub mod membench;

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Common shape every scorer writes to disk as `<bench>.json`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ScoreReport {
    /// Free-form harness id ("mnem-lme-session", "mnem-locomo", ...).
    pub harness: String,
    /// Adapter that ran (e.g. "mnem").
    pub adapter: String,
    /// Path to the dataset file consumed.
    pub dataset: String,
    /// Total questions scored.
    pub n_questions: usize,
    /// Wall-time seconds for the run.
    pub runtime_seconds: f64,
    /// Per-phase wall-time split.
    pub timing: TimingBreakdown,
    /// Headline metrics (`recall@5`, `recall@10`, ...).
    pub overall: BTreeMap<String, f64>,
    /// Optional per-category breakdown. Empty when the bench has
    /// no category split.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub by_category: BTreeMap<String, BTreeMap<String, f64>>,
}

/// Per-phase wall-time split.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct TimingBreakdown {
    /// Seconds spent in adapter `ingest` calls.
    pub ingest_s: f64,
    /// Seconds spent in adapter `retrieve` calls.
    pub retrieve_s: f64,
    /// Seconds spent computing recall + writing rows.
    pub score_s: f64,
}

/// One per-question row written to `<bench>.jsonl`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PerQuestionRow {
    /// Question id (or category-specific synthetic id).
    pub qid: String,
    /// Optional question type / category.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub qtype: Option<String>,
    /// Hit at top-5 (boolean as 0/1).
    pub hit_at_5: u8,
    /// Hit at top-10.
    pub hit_at_10: u8,
    /// Top-5 retrieved external ids, in rank order.
    pub top5: Vec<String>,
    /// Gold external ids the bench expected to see.
    pub gold: Vec<String>,
}
