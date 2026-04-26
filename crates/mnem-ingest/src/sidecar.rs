//! Optional sidecar escalation for scanned / text-layer-thin PDFs.
//!
//! Phase-B5e ships this module behind two independent Cargo features:
//!
//! - `sidecar-docling` - shells out to the
//!   [`docling`](https://github.com/docling-project/docling) CLI.
//! - `sidecar-unstructured` - shells out to the
//!   [`unstructured-ingest`](https://github.com/Unstructured-IO/unstructured)
//!   CLI.
//!
//! Both backends are optional because they are heavy: each pulls a
//! Python runtime and multi-GB model weights. Operators who already
//! deploy one of them get native mnem integration; everyone else keeps
//! the default pure-Rust path via [`crate::pdf::parse_pdf`].
//!
//! # Escalation contract
//!
//! [`crate::pdf::parse_pdf`] succeeds on any PDF with a readable text
//! layer. Scanned PDFs that fall below
//! [`crate::pdf::MIN_TEXT_PER_PAGE`] characters per page return a
//! usable but thin `Vec<Section>`. Callers who want higher fidelity
//! can route the same bytes through a [`Sidecar`] - typically by
//! constructing a [`DoclingSidecar`] or an [`UnstructuredSidecar`],
//! calling [`Sidecar::extract_pdf`], and comparing the total character
//! count against the baseline.
//!
//! The sidecar path is **not** triggered automatically by the pipeline
//! today: the dispatch decision belongs to the operator (via CLI / HTTP
//! config), which keeps the core ingest deterministic. An auto-
//! escalation hook is documented here so future waves can wire it in.
//!
//! # Failure handling
//!
//! Every failure mode resolves to [`crate::Error::Sidecar`]:
//! - binary not on `PATH`
//! - non-zero exit status (stderr captured)
//! - malformed stdout JSON
//!
//! The extractor never panics on user input and never leaks a temp
//! file beyond the lifetime of the call.

use std::io::Write as _;
use std::process::{Command, Stdio};

use serde::Deserialize;

use crate::error::Error;
use crate::types::Section;

/// Pluggable sidecar extractor for heavy PDF / document backends.
pub trait Sidecar: Send + Sync {
    /// Parse the PDF `bytes` through the sidecar backend and return the
    /// flattened section list. Implementations should mirror
    /// [`crate::pdf::parse_pdf`] semantics: one [`Section`] per page,
    /// `heading = Some("Page N")`, `depth = 1`, `byte_range` relative
    /// to the extracted text.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Sidecar`] when the binary is missing, exits
    /// non-zero, or emits malformed output.
    fn extract_pdf(&self, bytes: &[u8]) -> Result<Vec<Section>, Error>;
}

/// [`Sidecar`] backed by the `docling` CLI.
#[cfg(feature = "sidecar-docling")]
#[derive(Debug, Default)]
pub struct DoclingSidecar;

#[cfg(feature = "sidecar-docling")]
impl Sidecar for DoclingSidecar {
    fn extract_pdf(&self, bytes: &[u8]) -> Result<Vec<Section>, Error> {
        let bin = which::which("docling").map_err(|e| Error::Sidecar {
            tool: "docling".into(),
            detail: format!("binary not found on PATH: {e}"),
        })?;
        // docling expects a file path; write bytes to a tempfile first.
        let mut tmp = tempfile_new("docling-input.pdf")?;
        tmp.write_all(bytes).map_err(Error::IoError)?;
        let path = tmp.path().to_path_buf();
        // Release the writer handle so docling can reopen on Windows.
        drop(tmp);
        let out = Command::new(bin)
            .arg(&path)
            .arg("--output-format")
            .arg("json")
            .arg("--to")
            .arg("-")
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output()
            .map_err(|e| Error::Sidecar {
                tool: "docling".into(),
                detail: format!("spawn: {e}"),
            })?;
        if !out.status.success() {
            return Err(Error::Sidecar {
                tool: "docling".into(),
                detail: String::from_utf8_lossy(&out.stderr).into_owned(),
            });
        }
        parse_docling_json(&out.stdout)
    }
}

/// [`Sidecar`] backed by the `unstructured-ingest` CLI.
#[cfg(feature = "sidecar-unstructured")]
#[derive(Debug, Default)]
pub struct UnstructuredSidecar;

#[cfg(feature = "sidecar-unstructured")]
impl Sidecar for UnstructuredSidecar {
    fn extract_pdf(&self, bytes: &[u8]) -> Result<Vec<Section>, Error> {
        let bin = which::which("unstructured-ingest").map_err(|e| Error::Sidecar {
            tool: "unstructured-ingest".into(),
            detail: format!("binary not found on PATH: {e}"),
        })?;
        let mut tmp = tempfile_new("unstructured-input.pdf")?;
        tmp.write_all(bytes).map_err(Error::IoError)?;
        let path = tmp.path().to_path_buf();
        drop(tmp);
        let out = Command::new(bin)
            .arg("local")
            .arg("--input-path")
            .arg(&path)
            .arg("--output-format")
            .arg("json")
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output()
            .map_err(|e| Error::Sidecar {
                tool: "unstructured-ingest".into(),
                detail: format!("spawn: {e}"),
            })?;
        if !out.status.success() {
            return Err(Error::Sidecar {
                tool: "unstructured-ingest".into(),
                detail: String::from_utf8_lossy(&out.stderr).into_owned(),
            });
        }
        parse_unstructured_json(&out.stdout)
    }
}

// ---------------- Wire schemas ----------------

#[derive(Debug, Deserialize)]
struct DoclingDoc {
    #[serde(default)]
    pages: Vec<DoclingPage>,
}

#[derive(Debug, Deserialize)]
struct DoclingPage {
    #[serde(default)]
    text: String,
}

fn parse_docling_json(stdout: &[u8]) -> Result<Vec<Section>, Error> {
    let doc: DoclingDoc = serde_json::from_slice(stdout).map_err(|e| Error::Sidecar {
        tool: "docling".into(),
        detail: format!("malformed JSON: {e}"),
    })?;
    Ok(pages_to_sections(doc.pages.into_iter().map(|p| p.text)))
}

#[derive(Debug, Deserialize)]
struct UnstructuredElement {
    #[serde(default)]
    text: String,
    #[serde(default)]
    metadata: UnstructuredMeta,
}

#[derive(Debug, Default, Deserialize)]
struct UnstructuredMeta {
    #[serde(default)]
    page_number: Option<u32>,
}

fn parse_unstructured_json(stdout: &[u8]) -> Result<Vec<Section>, Error> {
    let elements: Vec<UnstructuredElement> =
        serde_json::from_slice(stdout).map_err(|e| Error::Sidecar {
            tool: "unstructured-ingest".into(),
            detail: format!("malformed JSON: {e}"),
        })?;

    // Group elements by page_number (fallback: one page for the whole doc).
    let mut pages: std::collections::BTreeMap<u32, String> = std::collections::BTreeMap::new();
    for el in elements {
        let pn = el.metadata.page_number.unwrap_or(1);
        let entry = pages.entry(pn).or_default();
        if !entry.is_empty() {
            entry.push('\n');
        }
        entry.push_str(&el.text);
    }
    Ok(pages_to_sections(pages.into_values()))
}

fn pages_to_sections(pages: impl IntoIterator<Item = String>) -> Vec<Section> {
    let mut out = Vec::new();
    let mut cursor = 0usize;
    for (idx, text) in pages.into_iter().enumerate() {
        let body = text.trim();
        if body.is_empty() {
            continue;
        }
        let len = body.len();
        out.push(Section {
            heading: Some(format!("Page {}", idx + 1)),
            depth: 1,
            text: body.to_string(),
            byte_range: cursor..(cursor + len),
        });
        cursor = cursor.saturating_add(len).saturating_add(1);
    }
    out
}

/// Cross-platform tempfile helper. Returns the writable file; the
/// caller is responsible for flushing and dropping before the sidecar
/// reopens the path (Windows file-locking).
fn tempfile_new(_hint: &'static str) -> Result<tempfile::NamedTempFile, Error> {
    tempfile::Builder::new()
        .prefix("mnem-ingest-sidecar-")
        .suffix(".pdf")
        .tempfile()
        .map_err(Error::IoError)
}

// ---------------- Auto-escalation hook ----------------

/// Decide whether a baseline [`crate::pdf::parse_pdf`] result is thin
/// enough to warrant sidecar escalation.
///
/// Returns `true` when the total character count across all sections
/// falls below [`crate::pdf::MIN_TEXT_PER_PAGE`] × `page_count`. The
/// caller's escalation logic typically looks like:
///
/// ```ignore
/// let sections = mnem_ingest::pdf::parse_pdf(&bytes)?;
/// if mnem_ingest::sidecar::should_escalate(&sections) {
///     let sidecar = DoclingSidecar::default();
///     if let Ok(better) = sidecar.extract_pdf(&bytes) {
///         return Ok(better);
///     }
/// }
/// ```
///
/// The pipeline does not wire this automatically today - the decision
/// belongs to the operator via CLI / HTTP config so that ingest stays
/// deterministic without a sidecar in `PATH`.
#[must_use]
pub fn should_escalate(sections: &[Section]) -> bool {
    let page_count = sections.len().max(1);
    let total_chars: usize = sections.iter().map(|s| s.text.chars().count()).sum();
    total_chars < crate::pdf::MIN_TEXT_PER_PAGE * page_count
}

// ---------------- Tests ----------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn docling_parses_fixture_json() {
        let fixture = br#"{"pages":[{"text":"Page one body."},{"text":"Page two body."}]}"#;
        let sections = parse_docling_json(fixture).unwrap();
        assert_eq!(sections.len(), 2);
        assert_eq!(sections[0].heading.as_deref(), Some("Page 1"));
        assert_eq!(sections[0].text, "Page one body.");
        assert_eq!(sections[1].heading.as_deref(), Some("Page 2"));
    }

    #[test]
    fn unstructured_parses_fixture_json() {
        let fixture = br#"[
          {"text":"Para A","metadata":{"page_number":1}},
          {"text":"Para B","metadata":{"page_number":1}},
          {"text":"Para C","metadata":{"page_number":2}}
        ]"#;
        let sections = parse_unstructured_json(fixture).unwrap();
        assert_eq!(sections.len(), 2);
        assert!(sections[0].text.contains("Para A"));
        assert!(sections[0].text.contains("Para B"));
        assert!(sections[1].text.contains("Para C"));
    }

    #[test]
    fn malformed_json_surfaces_sidecar_error() {
        let fixture = br"not-json-at-all";
        let err = parse_docling_json(fixture).unwrap_err();
        match err {
            Error::Sidecar { tool, .. } => assert_eq!(tool, "docling"),
            other => panic!("expected Sidecar error, got {other:?}"),
        }
    }

    #[test]
    fn should_escalate_fires_on_thin_text() {
        let thin = vec![Section {
            heading: Some("Page 1".into()),
            depth: 1,
            text: "short".into(),
            byte_range: 0..5,
        }];
        assert!(should_escalate(&thin));
    }

    #[test]
    fn should_escalate_quiet_on_dense_text() {
        // One page with plenty of characters - must NOT escalate.
        let body: String = "lorem ipsum dolor sit amet ".repeat(20);
        let dense = vec![Section {
            heading: Some("Page 1".into()),
            depth: 1,
            text: body.clone(),
            byte_range: 0..body.len(),
        }];
        assert!(!should_escalate(&dense));
    }

    #[cfg(feature = "sidecar-docling")]
    #[test]
    fn docling_binary_absence_is_typed_error() {
        // Skip gracefully when docling IS installed (developer laptop).
        if which::which("docling").is_ok() {
            return;
        }
        let err = DoclingSidecar.extract_pdf(b"fake-pdf-bytes").unwrap_err();
        match err {
            Error::Sidecar { tool, detail } => {
                assert_eq!(tool, "docling");
                assert!(detail.contains("binary not found"));
            }
            other => panic!("expected Sidecar error, got {other:?}"),
        }
    }

    #[cfg(feature = "sidecar-unstructured")]
    #[test]
    fn unstructured_binary_absence_is_typed_error() {
        if which::which("unstructured-ingest").is_ok() {
            return;
        }
        let err = UnstructuredSidecar.extract_pdf(b"fake").unwrap_err();
        match err {
            Error::Sidecar { tool, .. } => assert_eq!(tool, "unstructured-ingest"),
            other => panic!("expected Sidecar error, got {other:?}"),
        }
    }
}
