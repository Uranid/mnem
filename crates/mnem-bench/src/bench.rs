//! Benchmark catalog + dataset metadata.
//!
//! Adding a new benchmark = adding a `Bench` variant + filling in
//! [`Bench::metadata`].

use serde::{Deserialize, Serialize};
use std::fmt;

/// Benchmarks the 0.1.0 harness ships. Order matches the TUI display
/// order.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Bench {
    /// LongMemEval-S, per-session chunking variant.
    LongMemEval,
    /// LoCoMo, session-granularity.
    Locomo,
    /// ConvoMem (Snap 2024).
    Convomem,
    /// MemBench simple/roles slice.
    MembenchSimpleRoles,
    /// MemBench high-level/movie slice.
    MembenchHighlevelMovie,
    /// LongMemEval with the v4 hybrid post-filter.
    LongMemEvalHybridV4,
}

/// Static metadata for one benchmark.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BenchMeta {
    /// Stable identifier used on the CLI (`--benches longmemeval`).
    pub id: &'static str,
    /// Human-readable display name (TUI + RESULTS.md headers).
    pub display: &'static str,
    /// Approximate wall time in seconds for the full run on a
    /// typical laptop (cpu-local mode, ONNX MiniLM). Surfaced in the
    /// TUI so users know what they signed up for.
    pub eta_seconds: u64,
    /// Approximate dataset size on disk (bytes).
    pub dataset_bytes: u64,
    /// One-line description shown in `mnem bench list`.
    pub description: &'static str,
}

impl Bench {
    /// Static catalog. Single source of truth.
    #[must_use]
    pub const fn all() -> &'static [Bench] {
        &[
            Bench::LongMemEval,
            Bench::Locomo,
            Bench::Convomem,
            Bench::MembenchSimpleRoles,
            Bench::MembenchHighlevelMovie,
            Bench::LongMemEvalHybridV4,
        ]
    }

    /// Look up by stable id (case-insensitive). Returns `None` for
    /// an unknown id.
    #[must_use]
    pub fn from_id(s: &str) -> Option<Self> {
        let lower = s.to_ascii_lowercase();
        for b in Self::all() {
            if b.metadata().id.eq_ignore_ascii_case(&lower) {
                return Some(*b);
            }
        }
        None
    }

    /// Static metadata for this benchmark.
    #[must_use]
    pub const fn metadata(self) -> BenchMeta {
        match self {
            Self::LongMemEval => BenchMeta {
                id: "longmemeval",
                display: "LongMemEval (per-session, 500q)",
                eta_seconds: 600,
                dataset_bytes: 264 * 1024 * 1024,
                description: "500 questions, MAX-aggregate turn->session, R@5 / R@10.",
            },
            Self::Locomo => BenchMeta {
                id: "locomo",
                display: "LoCoMo (session granularity)",
                eta_seconds: 300,
                dataset_bytes: 3 * 1024 * 1024,
                description: "10 conversations x ~200 QA, per-conv label, session R@5 / R@10.",
            },
            Self::Convomem => BenchMeta {
                id: "convomem",
                display: "ConvoMem (5 categories, avg recall)",
                eta_seconds: 240,
                dataset_bytes: 5 * 1024 * 1024,
                description: "5 headline evidence categories, substring-match avg_recall.",
            },
            Self::MembenchSimpleRoles => BenchMeta {
                id: "membench-simple-roles",
                display: "MemBench simple-roles (R@5)",
                eta_seconds: 180,
                dataset_bytes: 4 * 1024 * 1024,
                description: "MemBench simple/roles slice, target_step_id R@5 over 100 items.",
            },
            Self::MembenchHighlevelMovie => BenchMeta {
                id: "membench-highlevel-movie",
                display: "MemBench high-level/movie (R@5)",
                eta_seconds: 180,
                dataset_bytes: 6 * 1024 * 1024,
                description: "MemBench highlevel/movie slice, target_step_id R@5 over 100 items.",
            },
            Self::LongMemEvalHybridV4 => BenchMeta {
                id: "longmemeval-hybrid-v4",
                display: "LongMemEval hybrid-v4 (BM25 boost, R@5)",
                eta_seconds: 600,
                dataset_bytes: 264 * 1024 * 1024,
                description: "LongMemEval with BM25-derived post-fusion boost; reuses LME cache.",
            },
        }
    }
}

impl fmt::Display for Bench {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.metadata().display)
    }
}

/// Adapter (system-under-test) catalog.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum AdapterKind {
    /// In-process mnem via `mnem-core`.
    Mnem,
}

impl AdapterKind {
    /// Stable id used on the CLI.
    #[must_use]
    pub const fn id(self) -> &'static str {
        match self {
            Self::Mnem => "mnem",
        }
    }

    /// Display name for the TUI.
    #[must_use]
    pub const fn display(self) -> &'static str {
        match self {
            Self::Mnem => "mnem",
        }
    }

    /// Catalog order (display order in the TUI).
    #[must_use]
    pub const fn all() -> &'static [Self] {
        &[Self::Mnem]
    }

    /// Look up by id, case-insensitive.
    #[must_use]
    pub fn from_id(s: &str) -> Option<Self> {
        for a in Self::all() {
            if a.id().eq_ignore_ascii_case(s) {
                return Some(*a);
            }
        }
        None
    }
}

/// Run mode for the harness.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum RunMode {
    /// In-process, single-threaded.
    CpuLocal,
}

impl RunMode {
    /// Stable id used on the CLI.
    #[must_use]
    pub const fn id(self) -> &'static str {
        match self {
            Self::CpuLocal => "cpu-local",
        }
    }

    /// Display name for the TUI.
    #[must_use]
    pub const fn display(self) -> &'static str {
        match self {
            Self::CpuLocal => "CPU local (in-process)",
        }
    }

    /// Catalog order.
    #[must_use]
    pub const fn all() -> &'static [Self] {
        &[Self::CpuLocal]
    }

    /// Look up by id, case-insensitive.
    #[must_use]
    pub fn from_id(s: &str) -> Option<Self> {
        for m in Self::all() {
            if m.id().eq_ignore_ascii_case(s) {
                return Some(*m);
            }
        }
        None
    }
}

/// Embedder selector. Both variants ship in 0.1.0.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum EmbedderChoice {
    /// ONNX MiniLM via the bundled embedder. Production-grade; matches
    /// headline benchmark numbers.
    OnnxMiniLm,
    /// Deterministic hashed bag-of-tokens embedder built into mnem-bench.
    /// Network-free, offline. Toy embedder; recall is not comparable to
    /// ONNX figures. Useful for CI smoke tests that can't load the model.
    BagOfTokens,
}

impl EmbedderChoice {
    /// Stable id used on the CLI.
    #[must_use]
    pub const fn id(self) -> &'static str {
        match self {
            Self::BagOfTokens => "bag-of-tokens",
            Self::OnnxMiniLm => "onnx-minilm",
        }
    }

    /// Display name for the TUI.
    #[must_use]
    pub const fn display(self) -> &'static str {
        match self {
            Self::BagOfTokens => "bag-of-tokens (built-in, deterministic)",
            Self::OnnxMiniLm => "ONNX MiniLM (default, bundled)",
        }
    }

    /// Catalog order. ONNX MiniLM is listed first so the TUI's default
    /// picks the production-grade embedder instead of the offline toy.
    #[must_use]
    pub const fn all() -> &'static [Self] {
        &[Self::OnnxMiniLm, Self::BagOfTokens]
    }

    /// Look up by id, case-insensitive.
    #[must_use]
    pub fn from_id(s: &str) -> Option<Self> {
        for e in Self::all() {
            if e.id().eq_ignore_ascii_case(s) {
                return Some(*e);
            }
        }
        None
    }
}
