//! Thin shared ureq helpers: error classification and a bounded
//! `into_string()` that ~caps the body size on 4xx/5xx responses to
//! avoid pathological echo back of server errors.

use crate::error::EmbedError;

/// Maximum bytes of response body retained on a 4xx/5xx error path.
/// Keeps the error message readable even if the provider returns a
/// multi-megabyte HTML trace.
const ERR_BODY_CAP: usize = 4096;

/// Convert a `ureq::Error` into the semantically-right [`EmbedError`].
///
/// Called from every adapter's HTTP call site.
pub(crate) fn classify_ureq_error(e: ureq::Error) -> EmbedError {
    match e {
        ureq::Error::Status(status, resp) => {
            let body = resp
                .into_string()
                .unwrap_or_else(|ioe| format!("<response body read error: {ioe}>"));
            let body = if body.len() > ERR_BODY_CAP {
                // Truncate by chars to avoid splitting a multibyte scalar.
                body.chars().take(ERR_BODY_CAP).collect()
            } else {
                body
            };
            match status {
                401 | 403 => EmbedError::Auth(body),
                429 => EmbedError::RateLimited(body),
                400..=499 => EmbedError::BadRequest { status, body },
                500..=599 => EmbedError::Server { status, body },
                _ => EmbedError::Server { status, body },
            }
        }
        ureq::Error::Transport(t) => EmbedError::Network(t.to_string()),
    }
}

/// Parse an `ok` response as JSON into `T`, mapping I/O / decode
/// failures to [`EmbedError::Decode`].
pub(crate) fn decode_json<T: serde::de::DeserializeOwned>(
    resp: ureq::Response,
) -> Result<T, EmbedError> {
    resp.into_json::<T>()
        .map_err(|e| EmbedError::Decode(e.to_string()))
}
