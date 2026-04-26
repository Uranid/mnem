//! Shared data types used throughout the ingest pipeline.
//!
//! Kept in a single file for B5a; if the surface grows past ~150 lines in
//! later sub-waves we will split (`section.rs`, `chunk.rs`, `config.rs`).

use std::ops::Range;

use mnem_core::id::Cid;
use serde::{Deserialize, Serialize};

/// A hierarchical text region extracted from a source.
///
/// Produced by parsers in [`crate::md`] / [`crate::text`] and consumed by
/// chunkers in [`mod@crate::chunk`]. The `byte_range` always refers to offsets
/// in the *original* source input (not the post-parse normalized text), so
/// downstream stages can slice back into the raw document for diffing or
/// provenance tracking.
///
/// Heading depth uses `CommonMark`'s 1-indexed convention (`# H1 → 1`). A
/// depth of `0` indicates "no heading" (e.g. top-of-file prose before any
/// heading, or the synthetic root produced by [`crate::text::parse_text`]).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Section {
    /// Heading text, without the leading `#` markers and trimmed.
    pub heading: Option<String>,
    /// Heading depth (1–6 for actual headings, 0 for headless prose).
    pub depth: u8,
    /// Body text contained under this heading (code blocks are kept intact).
    pub text: String,
    /// Byte range in the original source input.
    pub byte_range: Range<usize>,
}

/// A single chunk emitted by a [`crate::chunk::ChunkerKind`].
///
/// `section_path` records the hierarchy of headings that enclose this
/// chunk, from the root of the document down. It is used by downstream
/// stages for breadcrumb display and for attaching graph edges back to
/// the enclosing `Doc` node.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Chunk {
    /// Heading hierarchy from outermost to innermost.
    pub section_path: Vec<String>,
    /// Chunk body text.
    pub text: String,
    /// Whitespace-split token count (deterministic estimate).
    pub tokens_estimate: u32,
}

/// The kind of source being ingested.
///
/// Only `Markdown` and `Text` are handled in Phase-B5a; the other variants
/// are declared here so public signatures remain stable across sub-waves.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SourceKind {
    /// `CommonMark` + GFM (tables, fenced code).
    Markdown,
    /// UTF-8 plain text, no structure inferred.
    Text,
    /// PDF (text-layer extraction). Handled in Phase-B5b.
    Pdf,
    /// Chat transcript (JSON/JSONL). Handled in Phase-B5b.
    Conversation,
}

/// Which chunker strategy to use, and its parameters.
///
/// Re-exported from [`mod@crate::chunk`] for convenience.
pub type ChunkerKind = crate::chunk::ChunkerKind;

/// Configuration for an ingest run.
///
/// `ntype` is the `Node::ntype` string applied to the root document node
/// once Phase-B5c wires commit. Typical values: `"Doc"`, `"Note"`,
/// `"Transcript"`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct IngestConfig {
    /// Which chunker to use.
    pub chunker: ChunkerKind,
    /// `Node::ntype` of the root Doc node.
    pub ntype: String,
    /// Target maximum tokens per chunk (advisory; used by recursive chunker).
    pub max_tokens: u32,
    /// Overlap tokens between adjacent chunks (recursive chunker only).
    pub overlap: u32,
}

impl Default for IngestConfig {
    fn default() -> Self {
        Self {
            chunker: ChunkerKind::Paragraph,
            ntype: "Doc".into(),
            max_tokens: 512,
            overlap: 32,
        }
    }
}

/// Outcome of a completed ingest run.
///
/// Phase-B5c wires the real pipeline: `commit_cid` is `Some(_)` whenever
/// the caller committed the transaction after [`crate::Ingester::ingest`]
/// returned; `None` when they ran a dry-run (ingest without commit) or
/// when the underlying backend reports no change. `node_count` counts
/// every `Node` added (the Doc root, one per chunk, one per unique
/// entity). `entity_count` and `relation_count` report extraction
/// output before dedup. `chunk_count` reports the number of chunks
/// produced by the chunker stage.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct IngestResult {
    /// Commit produced by the run, if any.
    pub commit_cid: Option<Cid>,
    /// Number of graph nodes created.
    pub node_count: u64,
    /// Number of chunks produced.
    pub chunk_count: u64,
    /// Number of entity nodes created (deduplicated across the run).
    pub entity_count: u64,
    /// Number of relation edges created.
    pub relation_count: u64,
    /// Wall-clock elapsed time in milliseconds.
    pub elapsed_ms: u64,
}

/// Recognised conversation-export formats.
///
/// Used by [`crate::conversation::parse_conversation`] to route JSON into
/// the right schema decoder. [`Self::Generic`] is the fallback for
/// `[{"role", "content", "timestamp"?}]` shaped payloads.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ConversationFormat {
    /// `ChatGPT` export (`conversations.json`) with a `mapping` tree of
    /// message nodes keyed by UUID.
    ChatGpt,
    /// Claude export with a flat `{"conversation": [{role, content}]}`
    /// top-level object.
    Claude,
    /// Generic `[{role, content, timestamp?}]` array.
    Generic,
}

/// A single turn in a conversation.
///
/// `timestamp` is an optional Unix epoch in seconds - some exports
/// (Claude, generic) omit it and we preserve that absence rather than
/// fabricating zeroes.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Message {
    /// Speaker role, e.g. `"user"`, `"assistant"`, `"system"`, `"tool"`.
    pub role: String,
    /// Turn text content. Multi-part `ChatGPT` messages are concatenated
    /// with `"\n\n"` separators by the parser.
    pub content: String,
    /// Unix epoch seconds, if the source provided one.
    pub timestamp: Option<u64>,
}

/// Configuration for the rule-based entity + relation extractor.
///
/// Wired in Phase-B5c. Defaults emit every [`crate::extract::EntityKind`]
/// variant, carry an empty keyword list, and use a 6-token relation
/// window. Callers tighten the set when they only want, say, emails
/// and URLs out of a source.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExtractorConfig {
    /// Which entity kinds the extractor should emit. An empty
    /// [`HashSet`](std::collections::HashSet) would suppress every
    /// layer, so we store an ordered [`Vec`] of kinds and look up via
    /// `contains` - correct even when the caller passes duplicates.
    pub emit_kinds: Vec<crate::extract::EntityKind>,
    /// Caller-supplied keyword list fed into the Aho-Corasick layer.
    /// Case-insensitive, left-most-longest match semantics.
    pub keywords: Vec<String>,
    /// Maximum number of whitespace-separated tokens between two entity
    /// spans that may still be linked by a proximity relation.
    pub relation_window_tokens: usize,
}

impl Default for ExtractorConfig {
    fn default() -> Self {
        use crate::extract::EntityKind;
        Self {
            emit_kinds: vec![
                EntityKind::Person,
                EntityKind::Organization,
                EntityKind::Location,
                EntityKind::Date,
                EntityKind::Url,
                EntityKind::Email,
                EntityKind::Keyword,
            ],
            keywords: Vec::new(),
            relation_window_tokens: 6,
        }
    }
}

/// Advisory inputs for [`crate::chunk::auto_chunker`].
///
/// Defaults match the production heuristics documented on each
/// [`crate::SourceKind`] → [`ChunkerKind`] mapping. Callers only need to
/// override when they want tighter or looser chunking than the out-of-
/// the-box behaviour.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChunkerAuto {
    /// Override `max_tokens` for recursive chunking. `None` picks the
    /// per-source-kind default.
    pub max_tokens: Option<u32>,
    /// Override `overlap` for recursive chunking. `None` picks the
    /// per-source-kind default.
    pub overlap: Option<u32>,
    /// Override the session-chunker boundary for conversations. `None`
    /// picks the default of 10 messages per chunk.
    pub max_messages: Option<usize>,
}
