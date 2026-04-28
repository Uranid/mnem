//! `CommonMark` / GFM parser that emits [`Section`]s.
//!
//! Uses [`pulldown_cmark`] to walk the event stream. Each `Heading` event
//! opens a new section; subsequent text/paragraph/code events append to
//! that section's body until the next heading (of any depth) is
//! encountered. Code blocks are preserved verbatim inside their enclosing
//! section - they are never split by the downstream chunker (the chunker
//! treats any section emitted here as an atomic unit when chunking by
//! paragraph; the recursive chunker respects code-fence boundaries via
//! the `tokens_estimate` gate).

use pulldown_cmark::{CodeBlockKind, Event, HeadingLevel, Options, Parser, Tag, TagEnd};

use crate::{error::Error, types::Section};

/// Parse a Markdown input into a flat `Vec<Section>`.
///
/// Heading hierarchy is preserved via [`Section::depth`] (1–6). Text that
/// appears before any heading is emitted as a single depth-0 section with
/// `heading == None`. The `byte_range` of each section spans from the
/// start of the heading (or 0 for the pre-heading prose) to the byte
/// before the next heading.
///
/// # Errors
///
/// Currently infallible (pulldown-cmark is tolerant), but the signature
/// returns `Result` so later sub-waves can add validation (e.g. max depth,
/// duplicate heading slugs) without a breaking change.
#[allow(clippy::too_many_lines)] // one linear pass; splitting hurts readability
pub fn parse_markdown(input: &str) -> Result<Vec<Section>, Error> {
    let mut opts = Options::empty();
    opts.insert(Options::ENABLE_TABLES);
    opts.insert(Options::ENABLE_STRIKETHROUGH);
    opts.insert(Options::ENABLE_TASKLISTS);

    let parser = Parser::new_ext(input, opts).into_offset_iter();

    let mut sections: Vec<Section> = Vec::new();
    let mut current: Option<Section> = Some(Section {
        heading: None,
        depth: 0,
        text: String::new(),
        byte_range: 0..0,
    });

    // Heading-capture state: when inside a heading tag, appended text
    // populates the heading string rather than the body.
    let mut in_heading: Option<u8> = None;
    let mut heading_start: usize = 0;
    let mut heading_buf = String::new();

    // Code-block capture state: we keep fences attached to the body so
    // the original formatting survives the round-trip.
    let mut in_code = false;

    for (event, range) in parser {
        match event {
            Event::Start(Tag::Heading { level, .. }) => {
                // Close out the previous section. Drop the synthetic
                // depth-0 root if it captured nothing of substance.
                if let Some(mut s) = current.take() {
                    s.byte_range.end = range.start;
                    let keep = s.heading.is_some() || !s.text.trim().is_empty();
                    if keep {
                        sections.push(s);
                    }
                }
                in_heading = Some(heading_level_to_depth(level));
                heading_start = range.start;
                heading_buf.clear();
            }
            Event::End(TagEnd::Heading(_)) => {
                let depth = in_heading.take().unwrap_or(1);
                current = Some(Section {
                    heading: Some(heading_buf.trim().to_string()),
                    depth,
                    text: String::new(),
                    byte_range: heading_start..range.end,
                });
                heading_buf.clear();
            }
            Event::Start(Tag::CodeBlock(kind)) => {
                in_code = true;
                if let Some(s) = current.as_mut() {
                    match kind {
                        CodeBlockKind::Fenced(lang) => {
                            s.text.push_str("```");
                            s.text.push_str(&lang);
                            s.text.push('\n');
                        }
                        CodeBlockKind::Indented => { /* indented form has no fence */ }
                    }
                }
            }
            Event::End(TagEnd::CodeBlock) => {
                in_code = false;
                if let Some(s) = current.as_mut() {
                    // Ensure closing fence + trailing newline.
                    if !s.text.ends_with('\n') {
                        s.text.push('\n');
                    }
                    s.text.push_str("```\n");
                }
            }
            Event::Text(t) => {
                if in_heading.is_some() {
                    heading_buf.push_str(&t);
                } else if let Some(s) = current.as_mut() {
                    s.text.push_str(&t);
                    if in_code && !t.ends_with('\n') {
                        // preserve internal newlines in code blocks
                    }
                }
            }
            Event::Code(c) => {
                if in_heading.is_some() {
                    heading_buf.push_str(&c);
                } else if let Some(s) = current.as_mut() {
                    s.text.push('`');
                    s.text.push_str(&c);
                    s.text.push('`');
                }
            }
            Event::SoftBreak | Event::HardBreak => {
                if let Some(s) = current.as_mut()
                    && in_heading.is_none()
                {
                    s.text.push('\n');
                }
            }
            Event::End(TagEnd::Paragraph) => {
                if let Some(s) = current.as_mut()
                    && !s.text.ends_with("\n\n")
                {
                    s.text.push_str("\n\n");
                }
            }
            _ => {}
        }
    }

    // Flush final section.
    if let Some(mut s) = current.take() {
        s.byte_range.end = input.len();
        // Skip the leading synthetic section if empty.
        if !(s.depth == 0 && s.heading.is_none() && s.text.trim().is_empty()) {
            sections.push(s);
        }
    }

    // Trim trailing whitespace on each section body for determinism.
    for s in &mut sections {
        while s.text.ends_with('\n') || s.text.ends_with(' ') {
            s.text.pop();
        }
    }

    Ok(sections)
}

const fn heading_level_to_depth(level: HeadingLevel) -> u8 {
    match level {
        HeadingLevel::H1 => 1,
        HeadingLevel::H2 => 2,
        HeadingLevel::H3 => 3,
        HeadingLevel::H4 => 4,
        HeadingLevel::H5 => 5,
        HeadingLevel::H6 => 6,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn headings_parsed_with_correct_depth() {
        let md = "# A\n\npara a\n\n## B\n\npara b\n\n### C\n\npara c\n";
        let sections = parse_markdown(md).unwrap();
        assert_eq!(sections.len(), 3);
        assert_eq!(sections[0].heading.as_deref(), Some("A"));
        assert_eq!(sections[0].depth, 1);
        assert_eq!(sections[1].heading.as_deref(), Some("B"));
        assert_eq!(sections[1].depth, 2);
        assert_eq!(sections[2].heading.as_deref(), Some("C"));
        assert_eq!(sections[2].depth, 3);
    }

    #[test]
    fn prose_before_heading_is_depth_zero() {
        let md = "intro line\n\n# First\n\nbody\n";
        let sections = parse_markdown(md).unwrap();
        assert_eq!(sections.len(), 2);
        assert_eq!(sections[0].depth, 0);
        assert!(sections[0].heading.is_none());
        assert!(sections[0].text.contains("intro line"));
    }

    #[test]
    fn code_block_is_atomic() {
        let md = "# Code\n\n```rust\nfn f() {}\n```\n\ntrailing\n";
        let sections = parse_markdown(md).unwrap();
        assert_eq!(sections.len(), 1);
        let body = &sections[0].text;
        // The code fence must appear contiguously with its body (not broken
        // by intervening section boundaries).
        assert!(body.contains("```rust"));
        assert!(body.contains("fn f() {}"));
        assert!(body.contains("```"));
        assert!(body.contains("trailing"));
    }

    #[test]
    fn empty_input_yields_no_sections() {
        let sections = parse_markdown("").unwrap();
        assert!(sections.is_empty());
    }

    #[test]
    fn snapshot_simple_doc() {
        let md = "# Intro\n\nHello world.\n\n## Details\n\nSome body text here.\n";
        let sections = parse_markdown(md).unwrap();
        insta::assert_yaml_snapshot!("simple_doc", sections);
    }

    #[test]
    fn snapshot_code_heavy_doc() {
        let md = "# Example\n\nPrelude paragraph.\n\n```rust\nfn main() {\n    println!(\"hi\");\n}\n```\n\nAnd a follow-up.\n";
        let sections = parse_markdown(md).unwrap();
        insta::assert_yaml_snapshot!("code_heavy_doc", sections);
    }
}
