//! Plain-text parser.
//!
//! No structure is inferred; the entire input becomes a single headless
//! [`Section`] whose `byte_range` spans the whole input. Downstream
//! chunkers will still split it into paragraph / token-budgeted chunks.

use crate::{error::Error, types::Section};

/// Parse a plain-text input into a single-element `Vec<Section>`.
///
/// The resulting section has `heading == None`, `depth == 0`, and a
/// `byte_range` covering the entire input. The `text` field is a copy of
/// the input with no normalization applied (callers that need trimming or
/// BOM stripping should do it upstream).
///
/// # Errors
///
/// Currently infallible. Returns `Result` for signature parity with
/// [`crate::md::parse_markdown`] so callers can dispatch on
/// [`crate::SourceKind`] uniformly.
pub fn parse_text(input: &str) -> Result<Vec<Section>, Error> {
    if input.is_empty() {
        return Ok(Vec::new());
    }
    Ok(vec![Section {
        heading: None,
        depth: 0,
        text: input.to_string(),
        byte_range: 0..input.len(),
    }])
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
}
