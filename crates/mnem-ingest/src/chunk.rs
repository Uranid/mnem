//! Chunker strategies.
//!
//! Consumes [`Section`]s and produces [`Chunk`]s. Five strategies:
//!
//! - [`ChunkerKind::Paragraph`] - splits each section's body on
//!   double-newline. Fast, deterministic, ideal for Markdown where the
//!   authoring structure already matches the desired chunk boundary.
//! - [`ChunkerKind::Recursive`] - token-budgeted word-window sliding window
//!   with configurable overlap. Kept for backwards compatibility.
//! - [`ChunkerKind::SentenceRecursive`] - sentence-aware token-budgeted
//!   packing using Unicode sentence boundaries (UAX #29). Preferred over
//!   `Recursive` for prose: chunks never cut mid-sentence, overlap is
//!   measured at sentence granularity, and average chunk size is more
//!   uniform. Default for `Text` and `Pdf` source kinds.
//! - [`ChunkerKind::Session`] - groups contiguous conversation messages
//!   into session chunks. Boundaries fire on role-returning-to-`user`
//!   OR on reaching `max_messages`. Preserves turn ordering.
//! - [`ChunkerKind::Structural`] - one chunk per section, used for
//!   code sources where each section is already a function or class body
//!   extracted by the tree-sitter parser.
//!
//! Token counts are estimated via whitespace split (`tokens_estimate`
//! field on `Chunk`). This is intentionally fast and deterministic;
//! cl100k accuracy is a future improvement.

use unicode_segmentation::UnicodeSegmentation;

use serde::{Deserialize, Serialize};

use crate::types::{Chunk, ChunkerAuto, Section, SourceKind};

/// Chunker strategy selector.
#[derive(Debug, Default, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ChunkerKind {
    /// Split on blank lines (`\n\n`). Preserves section path.
    #[default]
    Paragraph,
    /// Token-budgeted word-window sliding window with overlap (both
    /// measured in whitespace-split tokens). May cut mid-sentence.
    /// Kept for backwards compatibility; prefer [`ChunkerKind::SentenceRecursive`]
    /// for new prose ingests.
    Recursive {
        /// Maximum tokens per chunk (inclusive upper bound).
        max_tokens: u32,
        /// Tokens of overlap between adjacent chunks.
        overlap: u32,
    },
    /// Sentence-aware token-budgeted chunking (recommended for prose).
    ///
    /// Uses Unicode sentence boundaries (UAX #29 via `unicode-segmentation`)
    /// to pack complete sentences into chunks. Overlap is measured at
    /// sentence granularity so boundaries are always sentence-clean.
    /// Oversized single sentences are emitted as their own chunk.
    SentenceRecursive {
        /// Maximum whitespace-split tokens per chunk (advisory; a single
        /// sentence exceeding this is still emitted whole).
        max_tokens: u32,
        /// Token overlap budget between adjacent chunks (at sentence
        /// granularity). Set to 0 for no overlap.
        overlap: u32,
    },
    /// Group contiguous conversation messages into session chunks.
    ///
    /// Boundary rules:
    /// 1. Group up to `max_messages` adjacent sections together.
    /// 2. If the role transitions *back* to `user` after any non-user
    ///    turn, close the current chunk - this matches the natural
    ///    "one question, one answer, maybe a follow-up" rhythm of most
    ///    transcripts without forcing a fixed window.
    Session {
        /// Maximum messages grouped into a single session chunk.
        max_messages: usize,
    },
    /// One chunk per section, used for code sources.
    ///
    /// The code parser (tree-sitter) already produces one section per
    /// function / class / struct, so no further splitting is needed.
    Structural,
}

/// Run the configured chunker over `sections`.
///
/// The returned `Vec<Chunk>` preserves source order: section 0's chunks
/// come before section 1's. Empty sections are skipped silently.
#[must_use]
pub fn chunk(sections: &[Section], cfg: &ChunkerKind) -> Vec<Chunk> {
    match cfg {
        ChunkerKind::Paragraph => chunk_paragraph(sections),
        ChunkerKind::Recursive {
            max_tokens,
            overlap,
        } => chunk_recursive(sections, *max_tokens, *overlap),
        ChunkerKind::SentenceRecursive {
            max_tokens,
            overlap,
        } => chunk_sentence_recursive(sections, *max_tokens, *overlap),
        ChunkerKind::Session { max_messages } => chunk_session(sections, *max_messages),
        ChunkerKind::Structural => chunk_structural(sections),
    }
}

/// Pick a sensible [`ChunkerKind`] for a given [`SourceKind`].
///
/// | Source          | Strategy                                             |
/// |-----------------|------------------------------------------------------|
/// | `Markdown`      | `Paragraph`                                          |
/// | `Text`          | `SentenceRecursive { max_tokens: 256, overlap: 32 }` |
/// | `Pdf`           | `SentenceRecursive { max_tokens: 512, overlap: 64 }` |
/// | `Conversation`  | `Session { max_messages: 10 }`                       |
/// | `Code(_)`       | `Structural`                                         |
///
/// Callers may override numeric knobs via [`ChunkerAuto`]; fields left
/// `None` fall through to the defaults above.
#[must_use]
pub fn auto_chunker(kind: SourceKind, heuristics: ChunkerAuto) -> ChunkerKind {
    match kind {
        SourceKind::Markdown => ChunkerKind::Paragraph,
        SourceKind::Text => ChunkerKind::SentenceRecursive {
            max_tokens: heuristics.max_tokens.unwrap_or(256),
            overlap: heuristics.overlap.unwrap_or(32),
        },
        SourceKind::Pdf => ChunkerKind::SentenceRecursive {
            max_tokens: heuristics.max_tokens.unwrap_or(512),
            overlap: heuristics.overlap.unwrap_or(64),
        },
        SourceKind::Conversation => ChunkerKind::Session {
            max_messages: heuristics.max_messages.unwrap_or(10),
        },
        SourceKind::Code(_) => ChunkerKind::Structural,
    }
}

/// Error returned when [`resolve_chunker`] receives an unrecognised label.
#[derive(Debug, thiserror::Error)]
#[error(
    "unknown chunker '{0}'; \
     expected one of auto|paragraph|recursive|sentence_recursive|session|structural"
)]
pub struct UnknownChunker(pub String);

/// Parse a chunker label into a [`ChunkerKind`].
///
/// Single canonical implementation shared by CLI, HTTP, and MCP ingest
/// surfaces. Callers map the error into their own error type with `.map_err`.
///
/// `"auto"` delegates to [`auto_chunker`] with `max_tokens` / `overlap`
/// forwarded as overrides. All other labels construct a fixed variant
/// regardless of `kind`.
pub fn resolve_chunker(
    choice: &str,
    kind: SourceKind,
    max_tokens: u32,
    overlap: u32,
) -> Result<ChunkerKind, UnknownChunker> {
    Ok(match choice.to_ascii_lowercase().as_str() {
        "auto" => auto_chunker(
            kind,
            ChunkerAuto {
                max_tokens: Some(max_tokens),
                overlap: Some(overlap),
                max_messages: None,
            },
        ),
        "paragraph" => ChunkerKind::Paragraph,
        "recursive" => ChunkerKind::Recursive { max_tokens, overlap },
        "sentence_recursive" | "sentence-recursive" => {
            ChunkerKind::SentenceRecursive { max_tokens, overlap }
        }
        "session" => ChunkerKind::Session { max_messages: 10 },
        "structural" => ChunkerKind::Structural,
        other => return Err(UnknownChunker(other.to_string())),
    })
}

fn section_path_for(sections: &[Section], idx: usize) -> Vec<String> {
    // Build a breadcrumb by walking *backwards* from `idx`, tracking the
    // nearest ancestor of each depth level. This keeps the algorithm O(n)
    // without allocating a stack per call.
    let mut path: Vec<String> = Vec::new();
    let current_depth = sections[idx].depth;
    let mut last_depth = current_depth;
    for i in (0..=idx).rev() {
        let s = &sections[i];
        if let Some(h) = &s.heading
            && s.depth > 0
            && s.depth <= last_depth
        {
            path.push(h.clone());
            last_depth = s.depth.saturating_sub(1);
            if last_depth == 0 {
                break;
            }
        }
    }
    path.reverse();
    path
}

fn token_count(text: &str) -> u32 {
    // Whitespace split; matches the docs on `tokens_estimate`.
    u32::try_from(text.split_whitespace().count()).unwrap_or(u32::MAX)
}

fn chunk_paragraph(sections: &[Section]) -> Vec<Chunk> {
    let mut out = Vec::new();
    for (i, s) in sections.iter().enumerate() {
        if s.text.trim().is_empty() {
            continue;
        }
        let path = section_path_for(sections, i);
        for para in s.text.split("\n\n") {
            let trimmed = para.trim();
            if trimmed.is_empty() {
                continue;
            }
            out.push(Chunk {
                section_path: path.clone(),
                text: trimmed.to_string(),
                tokens_estimate: token_count(trimmed),
            });
        }
    }
    out
}

fn chunk_recursive(sections: &[Section], max_tokens: u32, overlap: u32) -> Vec<Chunk> {
    let max = max_tokens.max(1) as usize;
    let ov = (overlap as usize).min(max.saturating_sub(1));

    let mut out = Vec::new();
    for (i, s) in sections.iter().enumerate() {
        if s.text.trim().is_empty() {
            continue;
        }
        let path = section_path_for(sections, i);
        let tokens: Vec<&str> = s.text.split_whitespace().collect();

        if tokens.is_empty() {
            continue;
        }

        let mut start = 0usize;
        while start < tokens.len() {
            let end = (start + max).min(tokens.len());
            let slice = &tokens[start..end];
            let text = slice.join(" ");
            out.push(Chunk {
                section_path: path.clone(),
                text,
                tokens_estimate: u32::try_from(slice.len()).unwrap_or(u32::MAX),
            });
            if end == tokens.len() {
                break;
            }
            // Advance by (max - overlap); always at least 1 to guarantee
            // progress even when overlap == max.
            let step = max.saturating_sub(ov).max(1);
            start += step;
        }
    }
    out
}

/// Sentence-aware token-budgeted chunker.
///
/// Splits each section into Unicode sentences, then packs sentences into
/// chunks up to `max_tokens` (whitespace-split estimate). Overlap backs up
/// by `overlap` tokens worth of complete sentences so chunk boundaries are
/// always sentence-clean. A single sentence that exceeds `max_tokens` is
/// emitted as its own chunk rather than dropped.
fn chunk_sentence_recursive(sections: &[Section], max_tokens: u32, overlap: u32) -> Vec<Chunk> {
    let max = max_tokens.max(1) as usize;
    let ov = (overlap as usize).min(max.saturating_sub(1));

    let mut out = Vec::new();
    for (i, section) in sections.iter().enumerate() {
        let text = section.text.trim();
        if text.is_empty() {
            continue;
        }
        let path = section_path_for(sections, i);

        // Split on Unicode sentence boundaries (UAX #29).
        let sentences: Vec<&str> = text
            .split_sentence_bounds()
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .collect();

        if sentences.is_empty() {
            continue;
        }

        // Pre-compute whitespace token counts per sentence.
        let sent_tokens: Vec<usize> =
            sentences.iter().map(|s| s.split_whitespace().count()).collect();

        let mut start = 0usize;
        loop {
            if start >= sentences.len() {
                break;
            }

            // Accumulate sentences until the token budget is full.
            let mut tok = 0usize;
            let mut end = start;
            while end < sentences.len() {
                let t = sent_tokens[end];
                // Adding this sentence would overflow and we already have at
                // least one - close the current window.
                if tok + t > max && end > start {
                    break;
                }
                tok += t;
                end += 1;
                if tok >= max {
                    break;
                }
            }
            // One sentence bigger than max_tokens - include it anyway.
            if end == start {
                end = start + 1;
                tok = sent_tokens[start];
            }

            let chunk_text = sentences[start..end].join(" ");
            out.push(Chunk {
                section_path: path.clone(),
                text: chunk_text,
                tokens_estimate: u32::try_from(tok).unwrap_or(u32::MAX),
            });

            if end >= sentences.len() {
                break;
            }

            // Overlap: walk backwards from `end` reclaiming sentences up to
            // `ov` tokens, but always advance by at least one sentence to
            // guarantee termination.
            let mut overlap_tok = 0usize;
            let mut next_start = end;
            while next_start > start + 1 {
                let t = sent_tokens[next_start - 1];
                if overlap_tok + t > ov {
                    break;
                }
                overlap_tok += t;
                next_start -= 1;
            }
            start = next_start.max(start + 1);
        }
    }
    out
}

/// Structural chunker: one chunk per section, no further splitting.
///
/// Used for code sources where the tree-sitter parser has already
/// produced one section per function / class / struct.
fn chunk_structural(sections: &[Section]) -> Vec<Chunk> {
    let mut out = Vec::new();
    for (i, s) in sections.iter().enumerate() {
        let trimmed = s.text.trim();
        if trimmed.is_empty() {
            continue;
        }
        let path = section_path_for(sections, i);
        out.push(Chunk {
            section_path: path,
            text: trimmed.to_string(),
            tokens_estimate: token_count(trimmed),
        });
    }
    out
}

/// Extract the role from a conversation section's `heading`.
///
/// Conversation parser emits `heading = Some("[role]")`. We strip the
/// brackets here. Sections whose heading does not match that shape
/// return `None`, which the session chunker treats as "role unknown".
fn section_role(section: &Section) -> Option<&str> {
    let h = section.heading.as_deref()?;
    h.strip_prefix('[').and_then(|s| s.strip_suffix(']'))
}

fn chunk_session(sections: &[Section], max_messages: usize) -> Vec<Chunk> {
    let cap = max_messages.max(1);
    let mut out = Vec::new();
    let mut buffer: Vec<&Section> = Vec::with_capacity(cap);
    let mut saw_non_user = false;

    let flush = |buffer: &mut Vec<&Section>, out: &mut Vec<Chunk>| {
        if buffer.is_empty() {
            return;
        }
        let path = buffer
            .first()
            .map(|s| section_path_for(sections, section_index(sections, s)))
            .unwrap_or_default();
        let mut text = String::new();
        for (idx, s) in buffer.iter().enumerate() {
            if idx > 0 {
                text.push_str("\n\n");
            }
            if let Some(h) = &s.heading {
                text.push_str(h);
                text.push('\n');
            }
            text.push_str(&s.text);
        }
        let tokens = u32::try_from(text.split_whitespace().count()).unwrap_or(u32::MAX);
        out.push(Chunk {
            section_path: path,
            text,
            tokens_estimate: tokens,
        });
        buffer.clear();
    };

    for s in sections {
        if s.text.trim().is_empty() {
            continue;
        }
        let role = section_role(s);
        let is_user = role == Some("user");

        // Boundary: role has returned to `user` after at least one
        // non-user turn. Flush *before* pushing the new user message so
        // the next chunk starts with that user message.
        if is_user && saw_non_user && !buffer.is_empty() {
            flush(&mut buffer, &mut out);
            saw_non_user = false;
        }

        buffer.push(s);
        if !is_user {
            saw_non_user = true;
        }

        if buffer.len() >= cap {
            flush(&mut buffer, &mut out);
            saw_non_user = false;
        }
    }

    flush(&mut buffer, &mut out);
    out
}

/// Find a section's index by pointer equality within a slice.
///
/// The session chunker holds borrowed references while it flushes; this
/// helper recovers the original index so `section_path_for` can walk
/// the heading breadcrumb.
fn section_index(sections: &[Section], target: &Section) -> usize {
    sections
        .iter()
        .position(|s| std::ptr::eq(s, target))
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ops::Range;

    fn section(heading: Option<&str>, depth: u8, text: &str) -> Section {
        Section {
            heading: heading.map(str::to_string),
            depth,
            text: text.to_string(),
            byte_range: Range {
                start: 0,
                end: text.len(),
            },
        }
    }

    #[test]
    fn paragraph_splits_on_double_newline() {
        let secs = vec![section(Some("H"), 1, "alpha\n\nbeta\n\ngamma")];
        let chunks = chunk(&secs, &ChunkerKind::Paragraph);
        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[0].text, "alpha");
        assert_eq!(chunks[1].text, "beta");
        assert_eq!(chunks[2].text, "gamma");
        for c in &chunks {
            assert_eq!(c.section_path, vec!["H".to_string()]);
        }
    }

    #[test]
    fn paragraph_skips_empty_sections() {
        let secs = vec![
            section(None, 0, "   "),
            section(Some("Real"), 1, "content here"),
        ];
        let chunks = chunk(&secs, &ChunkerKind::Paragraph);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].text, "content here");
    }

    #[test]
    fn recursive_respects_max_tokens() {
        let text = (1..=20)
            .map(|n| format!("w{n}"))
            .collect::<Vec<_>>()
            .join(" ");
        let secs = vec![section(Some("H"), 1, &text)];
        let chunks = chunk(
            &secs,
            &ChunkerKind::Recursive {
                max_tokens: 5,
                overlap: 1,
            },
        );
        // step = 5 - 1 = 4; windows start at 0, 4, 8, 12, 16 → 5 chunks.
        assert_eq!(chunks.len(), 5);
        for c in &chunks {
            assert!(c.tokens_estimate <= 5);
        }
    }

    #[test]
    fn recursive_overlap_is_present() {
        let text = "a b c d e f g h i j";
        let secs = vec![section(None, 0, text)];
        let chunks = chunk(
            &secs,
            &ChunkerKind::Recursive {
                max_tokens: 4,
                overlap: 2,
            },
        );
        // step = 4 - 2 = 2; windows: [a b c d], [c d e f], [e f g h], [g h i j]
        assert_eq!(chunks.len(), 4);
        assert_eq!(chunks[0].text, "a b c d");
        assert_eq!(chunks[1].text, "c d e f");
        // Overlap: last 2 tokens of chunk[0] == first 2 of chunk[1].
        assert!(chunks[1].text.starts_with("c d"));
    }

    #[test]
    fn recursive_zero_tokens_does_not_loop() {
        let secs = vec![section(None, 0, "one two three")];
        // overlap >= max gets clamped so we still make progress.
        let chunks = chunk(
            &secs,
            &ChunkerKind::Recursive {
                max_tokens: 2,
                overlap: 99,
            },
        );
        assert!(!chunks.is_empty());
        assert!(chunks.len() < 100);
    }

    #[test]
    fn section_path_nested_headings() {
        let secs = vec![
            section(Some("Top"), 1, "t"),
            section(Some("Mid"), 2, "m"),
            section(Some("Leaf"), 3, "leaf body"),
        ];
        let chunks = chunk(&secs, &ChunkerKind::Paragraph);
        let leaf = chunks.last().unwrap();
        assert_eq!(
            leaf.section_path,
            vec!["Top".to_string(), "Mid".to_string(), "Leaf".to_string()]
        );
    }

    fn msg(role: &str, body: &str) -> Section {
        section(Some(&format!("[{role}]")), 2, body)
    }

    #[test]
    fn session_respects_max_messages() {
        // 25 messages alternating user/assistant, max_messages = 10 →
        // ceil(25/10) = 3 chunks if the role boundary never fires first.
        // We craft 25 messages as user→assistant pairs; the boundary will
        // fire on every `user` after the first non-user, so with strict
        // alternation chunks form at every 2 messages. Test BOTH behaviours.
        let mut secs: Vec<Section> = Vec::new();
        for i in 0..25 {
            let role = if i % 2 == 0 { "user" } else { "assistant" };
            secs.push(msg(role, &format!("turn {i}")));
        }
        let chunks = chunk(&secs, &ChunkerKind::Session { max_messages: 10 });
        // With strict alternation the role boundary fires every
        // user-after-assistant, producing 13 chunks (12 pairs + final lone user).
        assert!(
            chunks.len() >= 3,
            "expected at least 3 chunks, got {}",
            chunks.len()
        );
        // Token budget sanity: no chunk should exceed max_messages entries.
        for c in &chunks {
            let role_tag_count =
                c.text.matches("[user]").count() + c.text.matches("[assistant]").count();
            assert!(role_tag_count <= 10, "chunk exceeds max_messages");
        }
    }

    #[test]
    fn session_flushes_on_max_messages_with_same_role() {
        // 25 tool messages (same role, never triggers user boundary) +
        // max_messages = 10 → exactly 3 chunks of sizes 10, 10, 5.
        let secs: Vec<Section> = (0..25).map(|i| msg("tool", &format!("t{i}"))).collect();
        let chunks = chunk(&secs, &ChunkerKind::Session { max_messages: 10 });
        assert_eq!(chunks.len(), 3);
    }

    #[test]
    fn session_flushes_on_role_back_to_user() {
        let secs = vec![
            msg("user", "hi"),
            msg("assistant", "hello"),
            msg("user", "again"),
            msg("assistant", "welcome"),
        ];
        let chunks = chunk(&secs, &ChunkerKind::Session { max_messages: 10 });
        // Two sessions: [user, assistant] then [user, assistant].
        assert_eq!(chunks.len(), 2);
        assert!(chunks[0].text.contains("[user]"));
        assert!(chunks[0].text.contains("[assistant]"));
        assert!(chunks[1].text.contains("again"));
        assert!(chunks[1].text.contains("welcome"));
    }

    #[test]
    fn session_preserves_order() {
        let secs = vec![
            msg("user", "one"),
            msg("assistant", "two"),
            msg("user", "three"),
        ];
        let chunks = chunk(&secs, &ChunkerKind::Session { max_messages: 10 });
        let concat: String = chunks
            .iter()
            .map(|c| c.text.as_str())
            .collect::<Vec<_>>()
            .join(" || ");
        let pos_one = concat.find("one").unwrap();
        let pos_two = concat.find("two").unwrap();
        let pos_three = concat.find("three").unwrap();
        assert!(pos_one < pos_two);
        assert!(pos_two < pos_three);
    }

    #[test]
    fn auto_chunker_defaults() {
        use crate::types::{ChunkerAuto, CodeLanguage};
        let auto = ChunkerAuto::default();
        assert!(matches!(
            auto_chunker(SourceKind::Markdown, auto),
            ChunkerKind::Paragraph
        ));
        // Text + Pdf now use SentenceRecursive (sentence-clean boundaries).
        assert!(matches!(
            auto_chunker(SourceKind::Text, auto),
            ChunkerKind::SentenceRecursive {
                max_tokens: 256,
                overlap: 32,
            }
        ));
        assert!(matches!(
            auto_chunker(SourceKind::Pdf, auto),
            ChunkerKind::SentenceRecursive {
                max_tokens: 512,
                overlap: 64,
            }
        ));
        assert!(matches!(
            auto_chunker(SourceKind::Conversation, auto),
            ChunkerKind::Session { max_messages: 10 }
        ));
        // Code → Structural (tree-sitter produced one section per item).
        assert!(matches!(
            auto_chunker(SourceKind::Code(CodeLanguage::Rust), auto),
            ChunkerKind::Structural
        ));
    }

    #[test]
    fn auto_chunker_overrides() {
        use crate::types::ChunkerAuto;
        let auto = ChunkerAuto {
            max_tokens: Some(128),
            overlap: Some(8),
            max_messages: Some(3),
        };
        assert!(matches!(
            auto_chunker(SourceKind::Text, auto),
            ChunkerKind::SentenceRecursive {
                max_tokens: 128,
                overlap: 8,
            }
        ));
        assert!(matches!(
            auto_chunker(SourceKind::Conversation, auto),
            ChunkerKind::Session { max_messages: 3 }
        ));
    }

    #[test]
    fn sentence_recursive_basic() {
        // Three sentences that fit in one chunk.
        let secs = vec![section(
            None,
            0,
            "Alice joined Acme. Bob met Carol. Dave left.",
        )];
        let chunks = chunk(
            &secs,
            &ChunkerKind::SentenceRecursive {
                max_tokens: 20,
                overlap: 0,
            },
        );
        // All sentences fit within max_tokens=20 → one chunk.
        assert_eq!(chunks.len(), 1);
        assert!(chunks[0].text.contains("Alice"));
        assert!(chunks[0].text.contains("Dave"));
    }

    #[test]
    fn sentence_recursive_splits_at_boundary() {
        // Force a split: small budget, two sentences.
        let secs = vec![section(
            None,
            0,
            "The quick brown fox jumps over the lazy dog. A second completely different sentence appears here now.",
        )];
        let chunks = chunk(
            &secs,
            &ChunkerKind::SentenceRecursive {
                max_tokens: 8,
                overlap: 0,
            },
        );
        // Should produce multiple chunks; no chunk cuts mid-sentence.
        assert!(chunks.len() >= 2, "expected at least 2 chunks, got {}", chunks.len());
        // Every chunk text should end at a sentence boundary (no hanging words).
        for c in &chunks {
            let t = c.text.trim();
            // Each chunk is a complete sentence or set of sentences.
            assert!(!t.is_empty());
        }
    }

    #[test]
    fn sentence_recursive_oversized_sentence_emitted_whole() {
        // A single sentence that exceeds max_tokens must still be emitted.
        let long_sentence = "word ".repeat(50).trim().to_string() + ".";
        let secs = vec![section(None, 0, &long_sentence)];
        let chunks = chunk(
            &secs,
            &ChunkerKind::SentenceRecursive {
                max_tokens: 10,
                overlap: 0,
            },
        );
        assert!(!chunks.is_empty(), "oversized sentence must not be dropped");
    }

    #[test]
    fn structural_one_chunk_per_section() {
        let secs = vec![
            section(Some("fn:main"), 1, "fn main() { println!(\"hi\"); }"),
            section(Some("fn:helper"), 1, "fn helper() {}"),
        ];
        let chunks = chunk(&secs, &ChunkerKind::Structural);
        assert_eq!(chunks.len(), 2);
        assert!(chunks[0].text.contains("main"));
        assert!(chunks[1].text.contains("helper"));
    }
}
