//! # mnem-bench
//!
//! 0.1.0 benchmark harness for [mnem](https://github.com/Uranid/mnem).
//!
//! ## Scope
//!
//! Benches:
//!
//! - [`Bench::LongMemEval`] (per-session chunking, R@5 / R@10).
//! - [`Bench::Locomo`] (session granularity, R@5 / R@10).
//! - [`Bench::Convomem`] (5 evidence categories, avg_recall).
//! - [`Bench::MembenchSimpleRoles`] (R@5 over the simple/roles slice).
//! - [`Bench::MembenchHighlevelMovie`] (R@5 over highlevel/movie).
//! - [`Bench::LongMemEvalHybridV4`] (BM25-boost post-fusion variant).
//!
//! Adapter: [`adapters::MnemAdapter`] (in-process via `mnem-core`).
//! Run mode: [`RunMode::CpuLocal`] (single-threaded, in-process).
//! Cache: SHA-256 verified dataset cache at `~/.mnem/bench-data/`.
//! Output: `RESULTS.md` + per-bench `.json` + `.jsonl`.
//! TUI: [`tui::run_tui`] interactive setup wizard.
//!
//! ## Embedders
//!
//! - [`EmbedderChoice::OnnxMiniLm`] - real
//!   `sentence-transformers/all-MiniLM-L6-v2` via
//!   `mnem-embed-providers` (`onnx-bundled`). 384-dim. Default; matches
//!   headline benchmark numbers. Gated on the default-on `onnx-minilm`
//!   Cargo feature.
//! - [`EmbedderChoice::BagOfTokens`] - hashed bag-of-tokens, always
//!   compiled. Network-free, toy. Useful as the
//!   `--no-default-features` fallback for CI smoke tests.
//!
//! See ``
//! for the design rationale.
//!
//! ## Quick start
//!
//! ```no_run
//! use mnem_bench::{
//!     bench::{AdapterKind, Bench, EmbedderChoice, RunMode},
//!     runner::{self, RunPlan},
//! };
//! use std::path::PathBuf;
//!
//! let plan = RunPlan {
//!     benches: vec![Bench::LongMemEval],
//!     adapters: vec![AdapterKind::Mnem],
//!     mode: RunMode::CpuLocal,
//!     embedder: EmbedderChoice::BagOfTokens,
//!     out: PathBuf::from("./out"),
//!     top_k: 10,
//!     limit: Some(5),
//!     no_cache: false,
//!     quiet: true,
//! };
//! let _outcomes = runner::run(&plan).unwrap();
//! ```

#![forbid(unsafe_code)]
#![deny(missing_docs)]

pub mod adapter;
pub mod adapters;
pub mod bench;
pub mod datasets;
pub mod embed;
pub mod output;
pub mod runner;
pub mod score;
pub mod tui;

pub use adapter::{BenchAdapter, Hit, IngestDoc};
pub use bench::{AdapterKind, Bench, EmbedderChoice, RunMode};
pub use runner::{BenchOutcome, RunPlan};

/// Library version (tracks the workspace package version).
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
