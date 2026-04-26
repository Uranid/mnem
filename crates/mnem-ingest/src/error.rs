//! Error type for the ingest pipeline.
//!
//! Kept intentionally small in Phase-B5a; additional variants (LLM failure,
//! sidecar unavailable, embedder error) land in later sub-waves alongside
//! the modules that raise them.

use thiserror::Error;

/// Errors produced by the ingest pipeline.
#[derive(Debug, Error)]
pub enum Error {
    /// Parsing the source bytes failed.
    ///
    /// `what` names the parser (`"markdown"`, `"text"`) and `detail`
    /// carries an upstream message suitable for logging - not for
    /// end-user display.
    #[error("ingest parse failed ({what}): {detail}")]
    ParseFailed {
        /// Short identifier of the parser that failed.
        what: String,
        /// Human-readable detail (may include upstream error text).
        detail: String,
    },

    /// The supplied [`crate::SourceKind`] is not yet supported in this
    /// sub-wave (e.g. PDF or Conversation before Phase-B5b/B5e).
    #[error("ingest source unsupported: {what}")]
    UnsupportedSource {
        /// Which source / feature was requested.
        what: String,
    },

    /// I/O failure reading a source artifact.
    #[error("ingest I/O error: {0}")]
    IoError(#[from] std::io::Error),

    /// A downstream `Transaction::add_node` / `add_edge` returned an
    /// error during pipeline execution. The embedded string carries the
    /// upstream `mnem_core::Error` display text so we avoid a
    /// cross-crate re-export while still preserving detail for logs.
    #[error("ingest commit error: {0}")]
    Commit(String),

    /// An [`crate::extract::Extractor`] or embedder raised an error the
    /// pipeline could not recover from.
    #[error("ingest extractor error: {0}")]
    Extractor(String),

    /// An optional sidecar binary (docling / unstructured, feature-gated
    /// in Phase-B5e) was requested but is not available on `PATH`, or
    /// returned a non-zero exit status / malformed output.
    ///
    /// The `tool` field names the sidecar (`"docling"`,
    /// `"unstructured-ingest"`); `detail` carries the failure mode
    /// (`"binary not found"`, upstream stderr, or parse error).
    #[error("ingest sidecar failure ({tool}): {detail}")]
    Sidecar {
        /// Short identifier of the sidecar that failed.
        tool: String,
        /// Human-readable detail (may include upstream stderr or parse
        /// error text).
        detail: String,
    },

    /// Catch-all for internal invariants. Prefer a typed variant for
    /// anything a caller might branch on.
    #[error("ingest error: {0}")]
    Other(String),
}

impl Error {
    /// Convert a `mnem_core::Error` into the wrapped [`Self::Commit`]
    /// variant. Kept private to the module's consumers via this helper
    /// so we do not have to publish a `From<mnem_core::Error>` impl
    /// (which would force every downstream crate to depend on the same
    /// mnem-core major version).
    pub(crate) fn commit(err: impl std::fmt::Display) -> Self {
        Self::Commit(err.to_string())
    }
}
