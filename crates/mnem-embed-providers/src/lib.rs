// OpenAI, Ollama proper-noun identifiers appear throughout the
// provider docs; backticking them adds no signal.
#![allow(clippy::doc_markdown)]

//! # mnem-embed-providers
//!
//! Embedding-provider adapters for mnem. Ships OpenAI and Ollama out
//! of the box; both behind opt-in (on-by-default) cargo features.
//!
//! ## Scope
//!
//! This crate turns a user-configured provider into a concrete
//! [`Embedder`] that the mnem CLI, MCP server, and Python bindings use
//! to (a) auto-embed node summaries on write and (b) auto-embed query
//! strings on retrieve. That is the piece that makes `mnem retrieve
//! --text ...` semantic-hybrid by default once a provider is
//! configured.
//!
//! ## Invariants
//!
//! - **No tokio / no async.** All adapters are sync, built on top of
//!   [`ureq`] (rustls-backed). Mnem cannot afford to drag an async
//!   runtime into the CLI or the MCP server.
//! - **No API keys in config / on disk.** The config stores the *name*
//!   of the env var holding the key (`api_key_env`). The key itself is
//!   read from the environment at adapter-construction time and is
//!   never persisted by this crate.
//! - **Deterministic outputs.** Adapters only wrap providers whose
//!   `embed(text)` is a pure function of `(provider, model, text)`.
//!   Randomised projections would break mnem's agent-replay guarantee.
//! - **`mnem-core` is not a dependency of the HTTP layer.** `mnem-core`
//!   still has zero network / HTTP / tokio in its dep tree, preserving
//!   the WASM-embeddability promises.
//!
//! ## Usage
//!
//! ```no_run
//! # use mnem_embed_providers::{open, Embedder, ProviderConfig, OpenAiConfig};
//! # fn demo() -> Result<(), Box<dyn std::error::Error>> {
//! let cfg = ProviderConfig::Openai(OpenAiConfig {
//!     model: "text-embedding-3-small".into(),
//!     ..Default::default()
//! });
//! let embedder = open(&cfg)?;
//! let v = embedder.embed("Alice lives in Berlin")?;
//! assert_eq!(v.len(), embedder.dim() as usize);
//! # Ok(()) }
//! ```

#![forbid(unsafe_code)]
#![deny(missing_docs)]

pub mod config;
pub mod embedder;
pub mod error;
pub(crate) mod http;
pub mod manifest;

#[cfg(any(test, feature = "mock"))]
pub mod mock;
#[cfg(feature = "ollama")]
pub mod ollama;
// `onnx` and `onnx-bundled` differ only in the ort runtime source
// (load-dynamic vs download-binaries). The Rust-level adapter is
// identical, so the module compiles on either feature. The
// `compile_error!` at the top of `onnx.rs` rejects the
// both-enabled combination at compile time.
#[cfg(any(feature = "onnx", feature = "onnx-bundled"))]
pub mod onnx;
#[cfg(feature = "openai")]
pub mod openai;

pub use config::{OllamaConfig, OnnxConfig, OpenAiConfig, ProviderConfig, open};
pub use embedder::{Embedder, to_embedding};
pub use error::EmbedError;
pub use manifest::{
    DEFAULT_LATENCY_BUDGET_MS, EmbedderManifest, derive_max_cooccurrence_ms,
    derive_max_knn_ingest_per_node_ms,
};

#[cfg(any(test, feature = "mock"))]
pub use mock::MockEmbedder;

#[cfg(any(feature = "onnx", feature = "onnx-bundled"))]
pub use onnx::{ModelKind as OnnxModelKind, OnnxEmbedder};
