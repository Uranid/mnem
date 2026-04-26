//! Error type for embedding-provider adapters.
//!
//! Deliberately coarse categories that a CLI can surface with concrete
//! remediation (missing env var, network down, model not found, dim
//! mismatch, ...). Avoids leaking internal HTTP library types.

use thiserror::Error;

/// Every fallible operation on an [`crate::Embedder`] returns this.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum EmbedError {
    /// Network / transport failure: TCP refused, DNS, TLS handshake,
    /// read timeout.
    #[error("network error: {0}")]
    Network(String),

    /// Provider rejected our credentials (HTTP 401 / 403).
    #[error("provider authentication failed: {0}")]
    Auth(String),

    /// Provider rate-limited (HTTP 429). Fail-fast; a retry-with-backoff
    /// layer belongs in the caller, not in the adapter.
    #[error("provider rate-limited the request (HTTP 429): {0}")]
    RateLimited(String),

    /// Client-side 4xx other than 401/403/429 (bad model name, invalid
    /// JSON, input too long, ...).
    #[error("provider rejected the request (HTTP {status}): {body}")]
    BadRequest {
        /// HTTP status code.
        status: u16,
        /// Response body (usually a JSON-encoded error message).
        body: String,
    },

    /// Provider-side 5xx. Treated as transient; caller decides retry.
    #[error("provider returned 5xx (HTTP {status}): {body}")]
    Server {
        /// HTTP status code.
        status: u16,
        /// Response body.
        body: String,
    },

    /// Response parsed as JSON but did not match the expected shape.
    #[error("failed to decode provider response: {0}")]
    Decode(String),

    /// Provider returned a vector whose length disagrees with the
    /// configured dimension.
    #[error("dim mismatch: expected {expected}, got {got}")]
    DimMismatch {
        /// Dimension the adapter expected (from config or prior calls).
        expected: u32,
        /// Dimension the provider actually returned.
        got: u32,
    },

    /// Config is malformed (e.g. unknown model string, invalid URL).
    #[error("config error: {0}")]
    Config(String),

    /// The env var named by `api_key_env` is not set in the process
    /// environment. Surfaced verbatim so the CLI can suggest `export`.
    #[error("environment variable {var} is not set")]
    MissingApiKey {
        /// Name of the missing env var, exactly as configured.
        var: String,
    },
}
