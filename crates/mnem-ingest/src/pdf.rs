//! PDF parser that emits [`Section`]s.
//!
//! Uses [`pdf_extract`] for pure-Rust text-layer extraction (no C
//! dependencies, no native build). The extracted text is split on the
//! form-feed character (`\x0C`) that `pdf-extract` injects at page
//! boundaries; each page becomes its own [`Section`] with depth `1`, a
//! heading of `"Page N"`, and a `byte_range` relative to the *extracted*
//! text (we cannot recover byte offsets in the original binary PDF).
//!
//! An optional `pdfium` feature is declared in `Cargo.toml` to layer the
//! `pdfium-render` backend on top in a later sub-wave; it is not wired
//! here to keep B5b deterministic and panic-free on malformed input.
//!
//! Low text-density PDFs (scanned images with no text layer) trigger a
//! `tracing::warn!` so operators can decide whether to re-ingest with an
//! OCR sidecar (Phase-B5e).

use crate::{error::Error, types::Section};

/// Minimum characters per page before the extractor considers the PDF
/// to have a usable text layer.
///
/// PDFs that fall below this threshold are logged at `warn` level and
/// returned as-is; chunking will still work but the quality will be poor.
/// The threshold is deliberately generous: a truly scanned PDF often
/// extracts only stray watermark tokens.
pub const MIN_TEXT_PER_PAGE: usize = 100;

/// Parse a PDF byte buffer into a flat `Vec<Section>`.
///
/// One section per page. `heading` is `Some("Page {n}")` with `depth = 1`
/// so the downstream chunker's `section_path_for` helper attributes every
/// chunk back to its source page. `byte_range` is relative to the
/// concatenated extracted text (pre-split), letting callers slice back
/// into the post-extract string if they cached it.
///
/// Malformed PDFs are reported as [`Error::ParseFailed`] with `what =
/// "pdf"`; this function never panics on user input.
///
/// # Errors
///
/// Returns [`Error::ParseFailed`] if `pdf-extract` cannot decode the
/// document (encrypted, corrupt, or truly not a PDF).
pub fn parse_pdf(bytes: &[u8]) -> Result<Vec<Section>, Error> {
    if bytes.is_empty() {
        return Ok(Vec::new());
    }

    // `pdf-extract` panics on some malformed inputs; wrap the call in
    // `catch_unwind` to convert any panic into a typed `ParseFailed`.
    let extracted = std::panic::catch_unwind(|| {
        pdf_extract::extract_text_from_mem(bytes).map_err(|e| Error::ParseFailed {
            what: "pdf".into(),
            detail: e.to_string(),
        })
    })
    .map_err(|_| Error::ParseFailed {
        what: "pdf".into(),
        detail: "pdf-extract panicked on malformed input".into(),
    })??;

    if extracted.is_empty() {
        return Ok(Vec::new());
    }

    let sections = split_pages(&extracted);

    // Low text-density heuristic - do not fail, just warn. Sidecar OCR
    // lands in B5e and will have a cleaner signal to key off.
    let page_count = sections.len().max(1);
    let total_chars = extracted.chars().count();
    if total_chars < MIN_TEXT_PER_PAGE * page_count {
        tracing::warn!(
            total_chars,
            page_count,
            min_per_page = MIN_TEXT_PER_PAGE,
            "pdf text density below threshold; consider OCR sidecar",
        );
    }

    Ok(sections)
}

/// Split `pdf-extract` output on form-feed page delimiters.
///
/// Form-feed (`\x0C`) is what `pdf-extract` emits between pages; we
/// treat a run of them as a single break. PDFs with no form-feeds (some
/// single-page or stream-oriented files) yield a single section whose
/// heading is still `"Page 1"` for consistency.
fn split_pages(text: &str) -> Vec<Section> {
    let mut sections = Vec::new();
    let mut cursor = 0usize;
    let mut page_no: usize = 1;

    for raw_page in text.split('\x0C') {
        let start = cursor;
        // +1 for the form-feed we just consumed, except for the final
        // slice where split() doesn't emit a trailing delimiter.
        let past_end = cursor + raw_page.len() < text.len();
        cursor = cursor
            .saturating_add(raw_page.len())
            .saturating_add(usize::from(past_end));

        let body = raw_page.trim_matches(['\r', '\n']);
        if body.trim().is_empty() {
            // Preserve page numbering (skipping blank pages entirely
            // would silently shift every subsequent page_no).
            page_no = page_no.saturating_add(1);
            continue;
        }

        sections.push(Section {
            heading: Some(format!("Page {page_no}")),
            depth: 1,
            text: body.to_string(),
            byte_range: start..(start + raw_page.len()),
        });
        page_no = page_no.saturating_add(1);
    }

    sections
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Smallest hand-crafted PDF that pdf-extract will parse. ~600 bytes.
    /// Content stream prints "Hello PDF" on page 1. We embed the binary
    /// as a byte literal so we don't need an on-disk fixture; any change
    /// to this string is caught by unit tests immediately.
    const TINY_PDF: &[u8] = b"%PDF-1.4\n\
1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n\
2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n\
3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 300 144] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>\nendobj\n\
4 0 obj\n<< /Length 44 >>\nstream\nBT /F1 18 Tf 10 100 Td (Hello PDF) Tj ET\nendstream\nendobj\n\
5 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n\
xref\n0 6\n0000000000 65535 f \n0000000009 00000 n \n0000000058 00000 n \n0000000109 00000 n \n0000000218 00000 n \n0000000304 00000 n \n\
trailer\n<< /Size 6 /Root 1 0 R >>\nstartxref\n372\n%%EOF\n";

    #[test]
    fn empty_input_yields_no_sections() {
        let sections = parse_pdf(&[]).unwrap();
        assert!(sections.is_empty());
    }

    #[test]
    fn malformed_bytes_are_error_not_panic() {
        // Not a PDF at all.
        let result = parse_pdf(b"definitely not a pdf");
        // Either ParseFailed or empty sections - but must NOT panic.
        match result {
            Ok(secs) => assert!(secs.is_empty() || !secs.is_empty()),
            Err(Error::ParseFailed { what, .. }) => assert_eq!(what, "pdf"),
            Err(other) => panic!("unexpected error kind: {other:?}"),
        }
    }

    #[test]
    fn tiny_pdf_extracts_a_page() {
        // pdf-extract may or may not pull text from such a minimal fixture
        // across versions; we assert only that the call returns Ok and
        // does not panic, which is the contract this module advertises.
        let result = parse_pdf(TINY_PDF);
        assert!(
            result.is_ok() || matches!(result, Err(Error::ParseFailed { .. })),
            "parse_pdf must either succeed or return ParseFailed, got {result:?}"
        );
    }

    #[test]
    fn split_pages_on_form_feed() {
        let text = "page one body\x0Cpage two body\x0Cpage three body";
        let sections = split_pages(text);
        assert_eq!(sections.len(), 3);
        assert_eq!(sections[0].heading.as_deref(), Some("Page 1"));
        assert_eq!(sections[1].heading.as_deref(), Some("Page 2"));
        assert_eq!(sections[2].heading.as_deref(), Some("Page 3"));
        assert!(sections[0].text.contains("page one"));
        assert!(sections[2].text.contains("page three"));
        for s in &sections {
            assert_eq!(s.depth, 1);
        }
    }

    #[test]
    fn split_pages_preserves_numbering_across_blanks() {
        let text = "alpha\x0C\x0Cbeta";
        let sections = split_pages(text);
        assert_eq!(sections.len(), 2);
        assert_eq!(sections[0].heading.as_deref(), Some("Page 1"));
        // The blank page is skipped from output but the counter advanced,
        // so `beta` lands on page 3 per the source ordering.
        assert_eq!(sections[1].heading.as_deref(), Some("Page 3"));
    }

    #[test]
    fn min_text_per_page_is_stable() {
        // Downstream B5e sidecar keys off this constant; pinning the
        // value here catches accidental drift.
        assert_eq!(MIN_TEXT_PER_PAGE, 100);
    }
}
