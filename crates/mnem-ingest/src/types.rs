//! Shared data types used throughout the ingest pipeline.
//!
//! Kept in a single file for B5a; if the surface grows past ~150 lines in
//! later sub-waves we will split (`section.rs`, `chunk.rs`, `config.rs`).

use std::ops::Range;

use mnem_core::id::Cid;
use mnem_ner_providers::NerConfig;
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

/// Programming language for [`SourceKind::Code`].
///
/// New languages can be added without breaking the public API because
/// `source_kind_for_path` falls back to [`SourceKind::Text`] for any
/// extension not listed here.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum CodeLanguage {
    /// Rust source (`.rs`).
    Rust,
    /// Python source (`.py`, `.pyi`).
    Python,
    /// JavaScript source (`.js`, `.mjs`, `.cjs`).
    JavaScript,
    /// TypeScript source (`.ts`, `.tsx`, `.mts`, `.cts`).
    TypeScript,
    /// Go source (`.go`).
    Go,
    /// Java source (`.java`).
    Java,
    /// C source (`.c`, `.h`).
    C,
    /// C++ source (`.cpp`, `.cc`, `.cxx`, `.hpp`, `.hxx`).
    Cpp,
    /// Ruby source (`.rb`, `.gemspec`, `.rake`, `.erb`).
    Ruby,
    /// C# source (`.cs`, `.csx`).
    CSharp,
}

impl CodeLanguage {
    /// Map a lowercase file extension to a language variant, or `None` if
    /// the extension is not a recognised code file.
    #[must_use]
    pub fn from_extension(ext: &str) -> Option<Self> {
        match ext {
            "rs" => Some(Self::Rust),
            "py" | "pyi" => Some(Self::Python),
            "js" | "mjs" | "cjs" => Some(Self::JavaScript),
            "ts" | "tsx" | "mts" | "cts" => Some(Self::TypeScript),
            "go" => Some(Self::Go),
            "java" => Some(Self::Java),
            "c" | "h" => Some(Self::C),
            "cpp" | "cc" | "cxx" | "c++" | "hpp" | "hxx" => Some(Self::Cpp),
            "rb" | "gemspec" | "rake" | "erb" => Some(Self::Ruby),
            "cs" | "csx" => Some(Self::CSharp),
            _ => None,
        }
    }

    /// Short lowercase name used in `mnem:source_kind` props and diagnostics.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Rust => "rust",
            Self::Python => "python",
            Self::JavaScript => "javascript",
            Self::TypeScript => "typescript",
            Self::Go => "go",
            Self::Java => "java",
            Self::C => "c",
            Self::Cpp => "cpp",
            Self::Ruby => "ruby",
            Self::CSharp => "csharp",
        }
    }
}

/// The kind of source being ingested.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SourceKind {
    /// `CommonMark` + GFM (tables, fenced code).
    Markdown,
    /// UTF-8 plain text, no structure inferred.
    Text,
    /// PDF (text-layer extraction).
    Pdf,
    /// Chat transcript (JSON/JSONL).
    Conversation,
    /// Source code file parsed with tree-sitter.
    Code(CodeLanguage),
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
    /// NER provider selection. Defaults to [`NerConfig::Rule`] (the
    /// capitalized-phrase heuristic). Set to [`NerConfig::None`] to
    /// suppress all entity extraction.
    #[serde(default)]
    pub ner: NerConfig,
}

impl Default for IngestConfig {
    fn default() -> Self {
        Self {
            chunker: ChunkerKind::Paragraph,
            ntype: "Doc".into(),
            max_tokens: 512,
            overlap: 32,
            ner: NerConfig::default(),
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

/// Configuration for the entity + relation extractor.
///
/// Entity extraction is handled entirely by the NER provider wired via
/// [`IngestConfig::ner`]. The provider may return any label strings it
/// chooses; there is no fixed vocabulary.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExtractorConfig {
    /// Call the NER provider for named-entity extraction. All labels
    /// returned by the provider pass through unconditionally.
    #[serde(default = "default_true")]
    pub extract_ner: bool,
    /// Maximum number of whitespace-separated tokens between two entity
    /// spans that may still be linked by a proximity relation.
    pub relation_window_tokens: usize,
}

fn default_true() -> bool {
    true
}

impl Default for ExtractorConfig {
    fn default() -> Self {
        Self {
            extract_ner: true,
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
