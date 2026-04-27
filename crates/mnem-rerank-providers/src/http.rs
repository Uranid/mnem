//! Thin shared ureq helpers mirroring mnem-embed-providers/http.rs:
//! classify ureq errors into [`RerankError`] and bound the echoed body
//! on 4xx/5xx responses.

use mnem_core::rerank::RerankError;

/// Maximum bytes of response body retained on a 4xx/5xx error path.
/// Keeps the error message readable even if the provider returns a
/// multi-megabyte HTML trace.
const ERR_BODY_CAP: usize = 4096;

/// Convert a [`ureq::Error`] into the semantically-right
/// [`RerankError`]. Called from every adapter's HTTP call site.
pub(crate) fn classify_ureq_error(e: ureq::Error) -> RerankError {
    match e {
        ureq::Error::Status(status, resp) => {
            let body = resp
                .into_string()
                .unwrap_or_else(|ioe| format!("<response body read error: {ioe}>"));
            let body = if body.len() > ERR_BODY_CAP {
                body.chars().take(ERR_BODY_CAP).collect()
            } else {
                body
            };
            match status {
                401 | 403 => RerankError::Auth(body),
                429 => RerankError::RateLimited(body),
                400..=499 => RerankError::BadRequest { status, body },
                500..=599 => RerankError::Server { status, body },
                _ => RerankError::Server { status, body },
            }
        }
        ureq::Error::Transport(t) => RerankError::Network(t.to_string()),
    }
}

/// Parse an `ok` response as JSON into `T`, mapping I/O / decode
/// failures to [`RerankError::Decode`].
pub(crate) fn decode_json<T: serde::de::DeserializeOwned>(
    resp: ureq::Response,
) -> Result<T, RerankError> {
    resp.into_json::<T>()
        .map_err(|e| RerankError::Decode(e.to_string()))
}
