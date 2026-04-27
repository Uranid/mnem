//! Agent-facing retrieval: compose structured filters, dense vector
//! similarity, and learned-sparse retrieval under a token budget.
//!
//! The indexes in [`crate::index`] each answer one question in
//! isolation: "which nodes carry label X", "which are semantically
//! close to this embedding", "which fire on this sparse query". Real
//! agents need all three at once and cannot afford to overflow their
//! LLM context window.
//!
//! [`Retriever`] is the composition layer. It:
//!
//! 1. Collects candidate node IDs from each ranker (vector, sparse).
//! 2. Fuses ranked lists with Reciprocal Rank Fusion (RRF, k=60).
//! 3. Gates fused candidates through label / property filters.
//! 4. Renders each surviving node to a compact text form.
//! 5. Greedily packs results in RRF-rank order until the caller's
//!    token budget is exhausted (rank-order skip: if a node does not
//!    fit, move on; never reorder to exploit slack).
//!
//! The return value ([`RetrievalResult`]) carries both the packed
//! items and cost metadata (`tokens_used`, `dropped`, `candidates_seen`)
//! so callers can detect "the budget was tight and we left good stuff
//! out" without a second round-trip.
//!
//! # Determinism
//!
//! All upstream rankers return hits in `(score desc, node_id asc)`
//! order, RRF is a pure function of ranks, and rendering is a pure
//! function of the node. Two independent processes with the same repo
//! head and the same [`Retriever`] configuration produce byte-
//! identical [`RetrievalResult`] instances. This is the property that
//! lets agent replay and regression tests work.
//!
//! # Example
//!
//! ```no_run
//! # use mnem_core::repo::ReadonlyRepo;
//! # fn demo(repo: &ReadonlyRepo, embedding: Vec<f32>) -> Result<(), Box<dyn std::error::Error>> {
//! let result = repo
//!     .retrieve()
//!     .label("Document")
//!     .vector("openai:text-embedding-3-small", embedding)
//!     .token_budget(2000)
//!     .execute()?;
//!
//! println!(
//!     "packed {} nodes in {}/{} tokens, {} dropped",
//!     result.items.len(),
//!     result.tokens_used,
//!     result.tokens_budget,
//!     result.dropped,
//! );
//! for item in &result.items {
//!     println!("{}", item.rendered);
//! }
//! # Ok(()) }
//! ```

pub mod community_filter;
pub mod fusion;
pub mod retriever;
pub mod session_reservoir;
pub mod types;
pub mod warnings;

pub use community_filter::{
    CommunityFilterCfg, CommunityId, CommunityLookup, apply_community_filter,
};
pub use fusion::{
    convex_min_max_fusion, reciprocal_rank_fusion, score_normalized_fusion,
    weighted_reciprocal_rank_fusion,
};
pub use retriever::Retriever;
pub use types::{
    FusionStrategy, GraphExpand, GraphExpandDirection, GraphExpandMode, Lane, RetrievalResult,
    RetrievedItem, TemporalFilter,
};
pub use warnings::{WARNINGS_CAP, Warning, WarningCode, cap_warnings};

use std::fmt::Write as _;

use ipld_core::ipld::Ipld;

use crate::objects::Node;

// ============================================================
// Token estimation
// ============================================================

/// A byte/char counter that approximates an LLM tokenizer.
///
/// Implementations must be deterministic pure functions of their input.
/// The default [`HeuristicEstimator`] covers the common case without
/// pulling in tokenizer dependencies; agents that need exact OpenAI /
/// Anthropic / Llama counts plug in their own impl.
pub trait TokenEstimator: Send + Sync + std::fmt::Debug {
    /// Estimate the number of tokens `text` consumes under the target
    /// tokenizer. Returning zero for the empty string is required;
    /// otherwise the return value MAY be conservative (overestimate)
    /// but MUST NOT vary between calls with the same input.
    fn estimate(&self, text: &str) -> u32;
}

/// Byte / character heuristic tuned for modern LLM tokenizers.
///
/// - ASCII bytes are counted as `ceil(bytes / 4)` - the OpenAI rule of
///   thumb, accurate within ~20% for English prose under `cl100k_base`
///   and `o200k_base` tokenizers.
/// - Non-ASCII characters (Unicode scalars outside `[0x00, 0x7F]`) are
///   counted as `ceil(chars / 1.5)` - roughly one token per CJK glyph
///   and two per emoji or Arabic/Cyrillic run, again within ~25% of
///   actual tokenizer output.
///
/// The two contributions are summed. Good enough for budget packing;
/// swap in a real tokenizer for exact accounting.
#[derive(Debug, Default, Clone, Copy)]
pub struct HeuristicEstimator;

impl TokenEstimator for HeuristicEstimator {
    fn estimate(&self, text: &str) -> u32 {
        if text.is_empty() {
            return 0;
        }
        let mut ascii_bytes: u32 = 0;
        let mut non_ascii_chars: u32 = 0;
        for ch in text.chars() {
            if ch.is_ascii() {
                ascii_bytes += 1;
            } else {
                non_ascii_chars += 1;
            }
        }
        // `f32` precision is plenty for < 2^24 bytes; we ceil at the end
        // so under-counts are impossible.
        let ascii_tokens = (ascii_bytes as f32 / 4.0).ceil() as u32;
        let non_ascii_tokens = (non_ascii_chars as f32 / 1.5).ceil() as u32;
        ascii_tokens + non_ascii_tokens
    }
}

// ============================================================
// Node rendering
// ============================================================

/// Character cap applied to `summary` + `context_sentence` in
/// `render_node`. An unbounded summary silently consumed the entire
/// token budget on a single oversized node, producing zero-recall
/// retrieves; capping the render-time string protects the budget
/// packer without losing the underlying data on the node itself.
///
/// Override with `MNEM_RENDER_SUMMARY_CAP_CHARS`. Measured in
/// `char` count (Unicode scalars) so multi-byte scripts don't hit
/// byte-boundary panics.
pub const DEFAULT_RENDER_SUMMARY_CAP_CHARS: usize = 8192;

fn render_summary_cap() -> usize {
    static CAP: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *CAP.get_or_init(|| {
        std::env::var("MNEM_RENDER_SUMMARY_CAP_CHARS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(DEFAULT_RENDER_SUMMARY_CAP_CHARS)
    })
}

/// Truncate to at most `cap` chars. Does NOT split a Unicode scalar
/// mid-sequence because `.chars()` iterates by scalar. Appends a
/// trailing `" <...+N chars>"` marker when truncation happens so a
/// downstream LLM can see the chunk was clipped.
fn clip_for_render(s: &str, cap: usize) -> String {
    let total = s.chars().count();
    if total <= cap {
        return s.to_string();
    }
    let kept: String = s.chars().take(cap).collect();
    let dropped = total - cap;
    format!("{kept} <...+{dropped} chars>")
}

/// Render a [`Node`] to a compact, deterministic text representation
/// suitable for LLM consumption.
///
/// The format is YAML-like and stable across versions:
///
/// ```text
/// ntype: <ntype>
/// id: <uuid>
/// context: <context_sentence>
/// summary: <summary>
/// <prop_key>: <prop_value>
/// ...
/// ```
///
/// - `ntype` and `id` are always present.
/// - `context` is emitted iff `node.context_sentence` is `Some`. Sits
///   BEFORE `summary` so an LLM reading the rendered node sees the
///   chunk's positional cue first (, Anthropic 2024
///   contextual-retrieval recipe).
/// - `summary` is emitted iff `node.summary` is `Some`. Clipped at
///   [`DEFAULT_RENDER_SUMMARY_CAP_CHARS`] (8192) chars by default,
///   overridable via `MNEM_RENDER_SUMMARY_CAP_CHARS`. A 1 MiB
///   summary on a single node would otherwise consume the entire
///   token budget and starve every other item out of the result.
/// - Scalar props (`String`, `Integer`, `Float`, `Bool`) are emitted in
///   BTreeMap order (alphabetical). Non-scalar props (`Link`, `Map`,
///   `List`, `Bytes`, `Null`) are skipped - an agent chasing a link
///   should follow it with a separate `mnem_get_node` call.
/// - Opaque `content` bytes are never rendered.
///
/// Determinism: since `Node.props` is a `BTreeMap`, iteration order is
/// byte-stable and the rendered string is therefore also byte-stable.
#[must_use]
pub fn render_node(node: &Node) -> String {
    let cap = render_summary_cap();
    let mut s = String::new();
    let _ = writeln!(s, "ntype: {}", node.ntype);
    let _ = writeln!(s, "id: {}", node.id);
    if let Some(context) = &node.context_sentence {
        let _ = writeln!(s, "context: {}", clip_for_render(context, cap));
    }
    if let Some(summary) = &node.summary {
        let _ = writeln!(s, "summary: {}", clip_for_render(summary, cap));
    }
    for (key, value) in &node.props {
        if let Some(rendered) = render_scalar(value) {
            let _ = writeln!(s, "{key}: {rendered}");
        }
    }
    s
}

/// Like [`render_node`] but augments the output with two graph
/// adjacency blocks derived from `repo`'s current commit:
///
/// ```text
/// ntype: Doc
/// id: ...
/// summary: ...
/// Outgoing:
///   tagged -> <topic_id>
/// Incoming:
///   authored <- <alice_id>
/// ```
///
/// Each block is capped at `per_direction_cap` entries; if the
/// bucket had more, a trailing `... (+N more)` line is emitted so
/// the reader knows the display was clipped. Blocks with no edges
/// are omitted entirely. Entry order is the adjacency bucket's
/// stored order (SPEC §4.9 `(label, src|dst, edge_cid)` sort),
/// so the rendered string is byte-stable.
///
/// Use this from CLI / MCP / agent-facing paths that benefit from
/// showing surrounding graph context. The hot `Retriever::execute`
/// render path keeps calling the cheaper [`render_node`] because
/// adjacency-aware rendering there would add a per-item O(log n)
/// bucket fetch for every ranked candidate, and callers that do
/// not need the blocks should not pay the cost.
#[must_use]
pub fn render_node_with_adjacency(
    node: &Node,
    repo: &crate::repo::ReadonlyRepo,
    per_direction_cap: usize,
) -> String {
    let mut s = render_node(node);
    if let Ok(edges) = repo.outgoing_edges(&node.id, None)
        && !edges.is_empty()
    {
        let total = edges.len();
        let shown = total.min(per_direction_cap);
        let _ = writeln!(s, "Outgoing:");
        for edge in edges.iter().take(shown) {
            let _ = writeln!(s, "  {} -> {}", edge.etype, edge.dst);
        }
        if total > shown {
            let _ = writeln!(s, "  ... (+{} more)", total - shown);
        }
    }
    if let Ok(edges) = repo.incoming_edges(&node.id, None)
        && !edges.is_empty()
    {
        let total = edges.len();
        let shown = total.min(per_direction_cap);
        let _ = writeln!(s, "Incoming:");
        for edge in edges.iter().take(shown) {
            let _ = writeln!(s, "  {} <- {}", edge.etype, edge.src);
        }
        if total > shown {
            let _ = writeln!(s, "  ... (+{} more)", total - shown);
        }
    }
    s
}

fn render_scalar(v: &Ipld) -> Option<String> {
    match v {
        Ipld::String(s) => Some(s.clone()),
        Ipld::Integer(n) => Some(n.to_string()),
        Ipld::Float(f) => Some(f.to_string()),
        Ipld::Bool(b) => Some(b.to_string()),
        _ => None,
    }
}

#[cfg(test)]
mod tests;
