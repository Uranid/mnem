//! Thin shared ureq helpers: classify ureq errors into [`LlmError`]
//! and bound the echoed body on 4xx/5xx. Mirrors the embed-providers
//! and rerank-providers http modules.

use mnem_core::llm::LlmError;

const ERR_BODY_CAP: usize = 4096;

pub(crate) fn classify_ureq_error(e: ureq::Error) -> LlmError {
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
                401 | 403 => LlmError::Auth(body),
                429 => LlmError::RateLimited(body),
                400..=499 => LlmError::BadRequest { status, body },
                500..=599 => LlmError::Server { status, body },
                _ => LlmError::Server { status, body },
            }
        }
        ureq::Error::Transport(t) => LlmError::Network(t.to_string()),
    }
}

pub(crate) fn decode_json<T: serde::de::DeserializeOwned>(
    resp: ureq::Response,
) -> Result<T, LlmError> {
    resp.into_json::<T>()
        .map_err(|e| LlmError::Decode(e.to_string()))
}
