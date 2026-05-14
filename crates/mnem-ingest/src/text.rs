//! Plain-text parser with optional structural section detection.
//!
//! For most inputs a single headless [`Section`] is returned (the whole
//! text as one block for the downstream chunker). When the input contains
//! recognisable heading patterns (legal / academic / report documents), the
//! parser splits on those headings and returns one section per heading.
//!
//! Recognised heading shapes (case-sensitive where noted):
//! - `ARTICLE I`, `ARTICLE 1`, `CHAPTER 2`, `PART III` (all-caps + roman/arabic)
//! - `Section 1`, `Section 1.2`, `Section 1.2.3`
//! - `I. Title`, `IV. Title` (roman numeral + period + word)
//! - `ALL CAPS LINE` - a line consisting entirely of uppercase letters,
//!   spaces, hyphens, or ampersands, at least 4 chars long (e.g. `BACKGROUND`)
//!
//! Structural detection only fires when at least **two** headings are found,
//! so ordinary prose never regresses to multi-section output.

use std::sync::OnceLock;

use regex::Regex;

use crate::{error::Error, types::Section};

fn heading_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(
            // Each alternation is a recognised heading shape.  The outer group
            // is non-capturing; `(?m)` makes `^` match at line starts.
            r"(?m)^[ \t]*(?:(?:ARTICLE|CHAPTER|PART)\s+(?:[IVXLCDM]{1,8}|\d+)|(?:SECTION|Section)\s+\d+(?:\.\d+)*|[IVXLCDM]{1,5}\.\s+[A-Z]|[A-Z][A-Z &\-/]{2,}[A-Z])[ \t]*[:\.]?[ \t]*$",
        )
        .expect("heading regex is valid")
    })
}

/// Parse a plain-text input into `Vec<Section>`.
///
/// When the input contains at least two recognised structural headings the
/// text is split into headed sections. Otherwise a single headless section
/// covering the full input is returned.
///
/// # Errors
///
/// Currently infallible; returns `Result` for signature parity with
/// [`crate::md::parse_markdown`].
pub fn parse_text(input: &str) -> Result<Vec<Section>, Error> {
    if input.is_empty() {
        return Ok(Vec::new());
    }

    // Collect all heading matches.
    let re = heading_re();
    let headings: Vec<_> = re.find_iter(input).collect();

    if headings.len() < 2 {
        // No structural headings: single-section pass-through.
        return Ok(vec![Section {
            heading: None,
            depth: 0,
            text: input.to_string(),
            byte_range: 0..input.len(),
        }]);
    }

    let mut sections = Vec::new();

    // Pre-preamble: text before the first heading.
    let first_start = headings[0].start();
    if first_start > 0 {
        let preamble = input[..first_start].trim();
        if !preamble.is_empty() {
            sections.push(Section {
                heading: None,
                depth: 0,
                text: preamble.to_string(),
                byte_range: 0..first_start,
            });
        }
    }

    // Each heading spans from the end of the heading line to the start of
    // the next heading (or EOF).
    for (idx, m) in headings.iter().enumerate() {
        let heading_text = m.as_str().trim().to_string();
        let body_start = m.end();
        let body_end = headings
            .get(idx + 1)
            .map(|next| next.start())
            .unwrap_or(input.len());

        let body = input[body_start..body_end].trim();
        sections.push(Section {
            heading: Some(heading_text),
            depth: 1,
            text: body.to_string(),
            byte_range: m.start()..body_end,
        });
    }

    Ok(sections)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_section_pass_through() {
        let input = "hello\nworld\n";
        let sections = parse_text(input).unwrap();
        assert_eq!(sections.len(), 1);
        assert!(sections[0].heading.is_none());
        assert_eq!(sections[0].depth, 0);
        assert_eq!(sections[0].text, input);
    }

    #[test]
    fn byte_range_covers_full_input() {
        let input = "some plain text";
        let sections = parse_text(input).unwrap();
        assert_eq!(sections[0].byte_range, 0..input.len());
    }

    #[test]
    fn empty_input_yields_no_sections() {
        let sections = parse_text("").unwrap();
        assert!(sections.is_empty());
    }

    #[test]
    fn structural_article_splits() {
        let input = "Preamble here.\n\nARTICLE I\nContent of article one.\n\nARTICLE II\nContent of article two.\n";
        let sections = parse_text(input).unwrap();
        // Should have: preamble + 2 article sections = 3
        assert!(
            sections.len() >= 2,
            "expected at least 2 sections, got {}",
            sections.len()
        );
        let headings: Vec<_> = sections.iter().filter_map(|s| s.heading.as_deref()).collect();
        assert!(
            headings.iter().any(|h| h.contains("ARTICLE I")),
            "ARTICLE I not found in {headings:?}"
        );
        assert!(
            headings.iter().any(|h| h.contains("ARTICLE II")),
            "ARTICLE II not found in {headings:?}"
        );
    }

    #[test]
    fn structural_section_numbering() {
        let input = "Section 1\nIntroduction text.\n\nSection 2\nBody text here.\n";
        let sections = parse_text(input).unwrap();
        assert!(sections.len() >= 2, "got {}", sections.len());
        assert!(sections.iter().any(|s| s.heading.as_deref() == Some("Section 1")));
        assert!(sections.iter().any(|s| s.heading.as_deref() == Some("Section 2")));
    }

    #[test]
    fn no_structural_headings_single_section() {
        // Only one heading-like line: should NOT trigger structural detection.
        let input = "ARTICLE I\nOnly one article here.\n";
        let sections = parse_text(input).unwrap();
        // Less than 2 headings: single-section pass-through.
        assert_eq!(sections.len(), 1);
        assert!(sections[0].heading.is_none());
    }
}
