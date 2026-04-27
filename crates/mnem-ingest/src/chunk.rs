//! Chunker strategies.
//!
//! Consumes [`Section`]s and produces [`Chunk`]s. Three strategies ship
//! through Phase-B5b:
//!
//! - [`ChunkerKind::Paragraph`] - splits each section's body on
//!   double-newline. Fast, deterministic, ideal for Markdown where the
//!   authoring structure already matches the desired chunk boundary.
//! - [`ChunkerKind::Recursive`] - token-budgeted sliding window with
//!   configurable overlap. Used when sections are long-form prose
//!   (transcripts, scraped HTML) and paragraph boundaries are either
//!   missing or too fine-grained.
//! - [`ChunkerKind::Session`] - groups contiguous conversation messages
//!   into session chunks. Boundaries fire on role-returning-to-`user`
//!   OR on reaching `max_messages`. Preserves turn ordering.
//!
//! Token counts in B5b are still estimated via whitespace split; a
//! cl100k-accurate tokenizer (`tiktoken-rs`) is deferred to a later
//! sub-wave once the surface-area tests stabilize.

use serde::{Deserialize, Serialize};

use crate::types::{Chunk, ChunkerAuto, Section, SourceKind};

/// Chunker strategy selector.
#[derive(Debug, Default, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ChunkerKind {
    /// Split on blank lines (`\n\n`). Preserves section path.
    #[default]
    Paragraph,
    /// Token-budgeted sliding window with overlap (both measured in
    /// whitespace-split tokens).
    Recursive {
        /// Maximum tokens per chunk (inclusive upper bound).
        max_tokens: u32,
        /// Tokens of overlap between adjacent chunks.
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
        ChunkerKind::Session { max_messages } => chunk_session(sections, *max_messages),
    }
}

/// Pick a sensible [`ChunkerKind`] for a given [`SourceKind`].
///
/// Defaults:
///
/// | Source          | Strategy                                 |
/// |-----------------|------------------------------------------|
/// | `Markdown`      | `Paragraph`                              |
/// | `Text`          | `Recursive { max_tokens: 256, overlap: 32 }` |
/// | `Pdf`           | `Recursive { max_tokens: 512, overlap: 64 }` |
/// | `Conversation`  | `Session { max_messages: 10 }`           |
///
/// Callers may override any numeric knob via [`ChunkerAuto`]; fields
/// left `None` fall through to these defaults.
#[must_use]
pub fn auto_chunker(kind: SourceKind, heuristics: ChunkerAuto) -> ChunkerKind {
    match kind {
        SourceKind::Markdown => ChunkerKind::Paragraph,
        SourceKind::Text => ChunkerKind::Recursive {
            max_tokens: heuristics.max_tokens.unwrap_or(256),
            overlap: heuristics.overlap.unwrap_or(32),
        },
        SourceKind::Pdf => ChunkerKind::Recursive {
            max_tokens: heuristics.max_tokens.unwrap_or(512),
            overlap: heuristics.overlap.unwrap_or(64),
        },
        SourceKind::Conversation => ChunkerKind::Session {
            max_messages: heuristics.max_messages.unwrap_or(10),
        },
    }
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
        use crate::types::ChunkerAuto;
        let auto = ChunkerAuto::default();
        assert!(matches!(
            auto_chunker(SourceKind::Markdown, auto),
            ChunkerKind::Paragraph
        ));
        assert!(matches!(
            auto_chunker(SourceKind::Text, auto),
            ChunkerKind::Recursive {
                max_tokens: 256,
                overlap: 32,
            }
        ));
        assert!(matches!(
            auto_chunker(SourceKind::Pdf, auto),
            ChunkerKind::Recursive {
                max_tokens: 512,
                overlap: 64,
            }
        ));
        assert!(matches!(
            auto_chunker(SourceKind::Conversation, auto),
            ChunkerKind::Session { max_messages: 10 }
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
            ChunkerKind::Recursive {
                max_tokens: 128,
                overlap: 8,
            }
        ));
        assert!(matches!(
            auto_chunker(SourceKind::Conversation, auto),
            ChunkerKind::Session { max_messages: 3 }
        ));
    }
}
