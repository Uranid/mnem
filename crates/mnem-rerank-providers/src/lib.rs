//! # mnem-rerank-providers
//!
//! Cross-encoder reranker adapters for mnem. Ships Cohere, Voyage, and
//! Jina out of the box; all three behind opt-in (on-by-default) cargo
//! features.
//!
//! ## Scope
//!
//! Per , `mnem-core` defines a
//! [`Reranker`][mnem_core::rerank::Reranker] trait
//! and wires it into the retrieve
//! pipeline as an optional post-fusion pass over the top-K. This crate
//! provides the production adapters. The adapters all read `(query,
//! candidate)` pairs jointly, which is what makes them useful for
//! compositional paraphrase that defeats dense + sparse bi-encoder
//! fusion ("father's sister" == "aunt").
//!
//! ## Invariants
//!
//! - **No tokio / no async.** All adapters are sync, built on top of
//!   [`ureq`] (rustls-backed). Matches `mnem-embed-providers`.
//! - **No API keys in config / on disk.** The config stores the *name*
//!   of the env var holding the key (`api_key_env`). The key itself is
//!   read from the environment at adapter-construction time and is
//!   never persisted by this crate.
//! - **`mnem-core` is not pulled onto an HTTP client.** `mnem-core`
//!   still has zero network / HTTP / tokio in its dep tree, preserving
//!   the WASM-embeddability promises.
//!
//! ## Usage
//!
//! ```no_run
//! # use std::sync::Arc;
//! # use mnem_rerank_providers::{open, ProviderConfig, CohereConfig};
//! # use mnem_core::rerank::Reranker;
//! # fn demo() -> Result<(), Box<dyn std::error::Error>> {
//! let cfg = ProviderConfig::Cohere(CohereConfig {
//!     model: "rerank-v3.5".into(),
//!     ..Default::default()
//! });
//! let rr: Arc<dyn Reranker> = open(&cfg)?;
//! let scores = rr.rerank("who is my father's sister", &["Eve is my aunt", "Bob is my cousin"])?;
//! assert_eq!(scores.len(), 2);
//! # Ok(()) }
//! ```
//!
//!

#![forbid(unsafe_code)]
#![deny(missing_docs)]

pub mod config;
pub(crate) mod http;

#[cfg(feature = "cohere")]
pub mod cohere;
#[cfg(feature = "jina")]
pub mod jina;
#[cfg(feature = "onnx")]
pub mod onnx;
#[cfg(feature = "voyage")]
pub mod voyage;

pub use config::{CohereConfig, JinaConfig, ProviderConfig, VoyageConfig, open};
#[cfg(feature = "onnx")]
pub use onnx::{OnnxReranker, RerankerModel};
