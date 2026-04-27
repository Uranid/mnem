//! # mnem-ingest
//!
//! Ingest pipeline for [mnem].
//!
//! Converts external source artifacts (Markdown, plain text, PDFs, and
//! chat-conversation exports) into the chunk-and-section intermediate
//! representation that downstream stages (extraction, embedding, graph
//! commit) consume.
//!
//! ## Scope (through Phase-B5c)
//!
//! - [`md::parse_markdown`] - `CommonMark` + GFM tables/code fences with
//!   heading hierarchy preserved.
//! - [`text::parse_text`] - single-section pass-through for plain text.
//! - [`pdf::parse_pdf`] - pure-Rust text-layer extraction via
//!   `pdf-extract`, page-boundary detection on form-feed.
//! - [`conversation::parse_conversation`] - `ChatGPT` / Claude / generic
//!   JSON exports flattened into one [`Section`] per turn.
//! - [`chunk::chunk`] - three chunker strategies:
//!     - [`ChunkerKind::Paragraph`] - double-newline split.
//!     - [`ChunkerKind::Recursive`] - token-budgeted sliding window.
//!     - [`ChunkerKind::Session`] - contiguous conversation messages
//!       grouped until role returns to `user` or a cap is hit.
//! - [`chunk::auto_chunker`] - picks a sensible [`ChunkerKind`] per
//!   [`SourceKind`].
//! - [`extract::RuleExtractor`] - deterministic rule-based NER over
//!   URLs, emails, dates, keywords, and capitalized phrases.
//! - [`pipeline::Ingester`] - end-to-end driver that writes Doc +
//!   Chunk + Entity nodes and the relation edges between them into a
//!   borrowed [`mnem_core::repo::Transaction`].
//!
//! ## Optional extensions (Phase-B5e)
//!
//! - [`extract_llm::OllamaExtractor`] - schema-constrained NER via a
//!   local Ollama server. Gated behind the `ollama` Cargo feature.
//!   Hallucinated spans are re-verified against section text and
//!   rejected; failures (timeout, schema-invalid) degrade to empty
//!   `Vec` rather than an error, so the rule-based baseline remains
//!   the load-bearing path.
//! - [`sidecar::Sidecar`] - escalation hook to an external
//!   `docling` / `unstructured-ingest` CLI for PDFs whose text-layer
//!   extraction is too thin. Gated behind `sidecar-docling` /
//!   `sidecar-unstructured`.
//!
//! ## Non-goals still outstanding
//!
//! - No CLI / MCP / HTTP wiring (Phase-B5d).
//!
//! ## Example
//!
//! ```
//! use mnem_ingest::{md::parse_markdown, chunk::{chunk, ChunkerKind}};
//!
//! let sections = parse_markdown("# Title\n\nFirst para.\n\nSecond para.").unwrap();
//! let chunks = chunk(&sections, &ChunkerKind::Paragraph);
//! assert!(!chunks.is_empty());
//! ```
//!
//! [mnem]: https://github.com/Uranid/mnem

#![deny(missing_docs)]
#![forbid(unsafe_code)]

pub mod chunk;
pub mod conversation;
pub mod error;
pub mod extract;
#[cfg(feature = "keybert")]
pub mod extract_keybert;
#[cfg(feature = "ollama")]
pub mod extract_llm;
pub mod md;
pub mod pdf;
pub mod pipeline;
#[cfg(any(feature = "sidecar-docling", feature = "sidecar-unstructured"))]
pub mod sidecar;
pub mod text;
pub mod types;

pub use chunk::{ChunkerKind, auto_chunker, chunk};
pub use error::Error;
pub use extract::{EntityKind, EntitySpan, Extractor, RelationSpan, RuleExtractor};
#[cfg(feature = "keybert")]
pub use extract_keybert::{KEYBERT_RELATION_LABEL, KeyBertAdapter};
#[cfg(feature = "ollama")]
pub use extract_llm::{
    DEFAULT_OLLAMA_MODEL, DEFAULT_OLLAMA_URL, LLM_ENTITY_CONFIDENCE, LLM_RELATION_CONFIDENCE,
    OllamaExtractor,
};
pub use pipeline::{EmbedText, EmbedderArc, Ingester};
pub use types::{
    Chunk, ChunkerAuto, ConversationFormat, ExtractorConfig, IngestConfig, IngestResult, Message,
    Section, SourceKind,
};

// Re-export Cid so downstream crates can refer to `mnem_ingest::Cid`
// without having to pull mnem-core directly.
pub use mnem_core::id::Cid as IngestCid;
