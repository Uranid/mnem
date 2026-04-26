// OpenAI, Ollama, Anthropic proper-noun identifiers appear throughout
// the provider docs; backticking them adds no signal.
#![allow(clippy::doc_markdown)]

//! # mnem-llm-providers
//!
//! Text-generation adapters for mnem. Ships OpenAI chat completions
//! and Ollama chat out of the box; both behind opt-in (on-by-default)
//! cargo features.
//!
//! ## Scope
//!
//! Per [`mnem_core::llm`], `mnem-core` defines a
//! [`TextGenerator`][mnem_core::llm::TextGenerator]
//! trait. This crate provides the production adapters. Used today by
//! `mnem retrieve --hyde`. The multi-query / RAG-Fusion variant is
//! planned and will share the same trait. Future LLM-in-the-loop
//! features (query rewriting, answer synthesis, retrieval grading)
//! will build on this surface too.
//!
//! ## Invariants
//!
//! - **No tokio / no async.** All adapters are sync, built on top of
//!   [`ureq`] (rustls-backed). Matches `mnem-embed-providers` and
//!   `mnem-rerank-providers`.
//! - **No API keys in config / on disk.** The config stores the *name*
//!   of the env var holding the key (`api_key_env`). The key itself is
//!   read from the environment at adapter-construction time.
//! - **`mnem-core` stays on no HTTP client.** `mnem-core` still has
//!   zero network / HTTP / tokio in its dep tree, preserving the
//!   WASM-embeddability promises.

#![forbid(unsafe_code)]
#![deny(missing_docs)]

pub mod config;
pub(crate) mod http;

#[cfg(feature = "ollama")]
pub mod ollama;
#[cfg(feature = "openai")]
pub mod openai;

pub use config::{OllamaLlmConfig, OpenAiLlmConfig, ProviderConfig, open};
