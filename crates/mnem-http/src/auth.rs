//! Bearer-token authentication extractor for `/remote/v1/*` write
//! routes.
//!
//! Two write verbs (`push-blocks`, `advance-head`) require a bearer
//! token on every request; the other two (`refs`, `fetch-blocks`) are
//! read-open. The expected token lives in the `MNEM_HTTP_PUSH_TOKEN`
//! environment variable and is read at server startup, cached in
//! [`AppState`], and never written to disk or to tracing spans.
//!
//! # Wire
//!
//! Clients attach `Authorization: Bearer <token>` on every
//! authenticated request. Missing header, malformed header, or
//! mismatched token all produce HTTP 401 with
//! `WWW-Authenticate: Bearer realm="mnem"` per RFC 6750 §3. The
//! response body is the crate-standard RFC 7807
//! `application/problem+json` envelope (see [`crate::error`]).
//!
//! # Why an axum extractor (not a tower layer)
//!
//! Layers run before route dispatch and would need to re-parse the
//! path to decide whether the current request hits an authenticated
//! route. Mounting the extractor on the two write handlers directly
//! is zero-cost for the read-open verbs and makes the authorization
//! surface visible at every callsite.
//!
//! # Token handling
//!
//! - Source of truth: `MNEM_HTTP_PUSH_TOKEN` env var at server boot.
//! - In-memory lifetime: stored as `Option<String>` inside
//!   [`AppState::push_token`]. Constant-time compare on every request.
//! - NEVER logged. The [`tracing`] spans emitted from the handler
//!   never include the token value (only `authenticated=true|false`
//!   when debugging).
//! - Absent token (env var unset) means `push-blocks` and
//!   `advance-head` return 503-Service-Unavailable via
//!   [`crate::error::RemoteError::AuthUnconfigured`]. Fail-closed:
//!   better to refuse writes than accept anonymous ones.

use axum::extract::FromRequestParts;
use axum::http::StatusCode;
use axum::http::header::{AUTHORIZATION, HeaderMap, WWW_AUTHENTICATE};
use axum::http::request::Parts;
use axum::response::{IntoResponse, Response};

use crate::state::AppState;

/// Axum extractor that enforces `Authorization: Bearer <token>`
/// against the server-configured push token.
///
/// Success yields a zero-size marker; on failure the extractor
/// short-circuits to a 401 or 503 response without the handler
/// running. Place on the first handler parameter of every
/// authenticated route.
///
/// The type intentionally carries no data; the token never escapes
/// the extractor. Constructing a `RequireBearer` value means the
/// request is authenticated; that's all the handler needs to know.
#[derive(Debug)]
pub(crate) struct RequireBearer;

impl FromRequestParts<AppState> for RequireBearer {
    type Rejection = AuthRejection;

    async fn from_request_parts(
        parts: &mut Parts,
        state: &AppState,
    ) -> Result<Self, Self::Rejection> {
        let Some(expected) = state.push_token.as_deref() else {
            // Env var unset -> writes administratively disabled.
            // Fail-closed; an operator can enable writes by setting
            // MNEM_HTTP_PUSH_TOKEN at restart.
            return Err(AuthRejection::Unconfigured);
        };
        let presented = extract_bearer(&parts.headers).ok_or(AuthRejection::Missing)?;
        if constant_time_eq(presented.as_bytes(), expected.as_bytes()) {
            Ok(Self)
        } else {
            Err(AuthRejection::Mismatch)
        }
    }
}

/// Parse `Authorization: Bearer <token>`. Returns the token slice
/// without the scheme prefix, or `None` if the header is missing or
/// not a Bearer scheme.
fn extract_bearer(headers: &HeaderMap) -> Option<String> {
    let raw = headers.get(AUTHORIZATION)?.to_str().ok()?;
    // Scheme match is case-insensitive per RFC 7235 §2.1. Trim a
    // single space between scheme and token; reject anything else
    // so a mis-spaced header is treated as missing rather than
    // silently accepted.
    let (scheme, token) = raw.split_once(' ')?;
    if !scheme.eq_ignore_ascii_case("Bearer") {
        return None;
    }
    let token = token.trim();
    if token.is_empty() {
        return None;
    }
    Some(token.to_string())
}

/// Reject a request with 401 + `WWW-Authenticate` (missing/mismatch)
/// or 503 (server has no token configured).
#[derive(Debug, Clone, Copy)]
pub(crate) enum AuthRejection {
    /// No `Authorization` header or the header was not a Bearer
    /// scheme.
    Missing,
    /// Presented token did not match `MNEM_HTTP_PUSH_TOKEN`.
    Mismatch,
    /// Server has no token configured; writes are disabled.
    Unconfigured,
}

impl IntoResponse for AuthRejection {
    fn into_response(self) -> Response {
        match self {
            Self::Missing | Self::Mismatch => {
                // 401 with RFC 6750 challenge header + RFC 7807 body.
                let body = serde_json::json!({
                    "type": "https://mnem.dev/errors/auth",
                    "title": "Unauthorized",
                    "status": 401,
                    "detail": match self {
                        Self::Missing => "missing or malformed Authorization: Bearer header",
                        Self::Mismatch => "bearer token did not match",
                        Self::Unconfigured => unreachable!(),
                    },
                });
                (
                    StatusCode::UNAUTHORIZED,
                    [
                        (WWW_AUTHENTICATE, "Bearer realm=\"mnem\""),
                        (axum::http::header::CONTENT_TYPE, "application/problem+json"),
                    ],
                    body.to_string(),
                )
                    .into_response()
            }
            Self::Unconfigured => {
                let body = serde_json::json!({
                    "type": "https://mnem.dev/errors/auth-unconfigured",
                    "title": "Service Unavailable",
                    "status": 503,
                    "detail": "push authentication not configured on this server",
                });
                (
                    StatusCode::SERVICE_UNAVAILABLE,
                    [(axum::http::header::CONTENT_TYPE, "application/problem+json")],
                    body.to_string(),
                )
                    .into_response()
            }
        }
    }
}

/// Byte-wise constant-time equality. Length is compared first (which
/// already leaks the length of the expected token - acceptable because
/// the token is a fixed operator-chosen secret, not a per-user value).
///
/// Kept inline instead of pulling `subtle` as a dep: the token check
/// runs once per write and never on a hot path, and the whole
/// surface is 12 lines.
#[inline]
fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut diff: u8 = 0;
    for (x, y) in a.iter().zip(b.iter()) {
        diff |= x ^ y;
    }
    diff == 0
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::http::HeaderValue;

    fn state_with(token: Option<&str>) -> AppState {
        crate::state::test_support::state_with_token(token.map(str::to_string))
    }

    #[test]
    fn constant_time_eq_matches_only_equal_bytes() {
        assert!(constant_time_eq(b"abc", b"abc"));
        assert!(!constant_time_eq(b"abc", b"abd"));
        assert!(!constant_time_eq(b"abc", b"abcd"));
        assert!(constant_time_eq(b"", b""));
    }

    #[test]
    fn extract_bearer_happy_path() {
        let mut h = HeaderMap::new();
        h.insert(AUTHORIZATION, HeaderValue::from_static("Bearer tok123"));
        assert_eq!(extract_bearer(&h).as_deref(), Some("tok123"));
    }

    #[test]
    fn extract_bearer_case_insensitive_scheme() {
        let mut h = HeaderMap::new();
        h.insert(AUTHORIZATION, HeaderValue::from_static("bearer tok123"));
        assert_eq!(extract_bearer(&h).as_deref(), Some("tok123"));
    }

    #[test]
    fn extract_bearer_rejects_wrong_scheme() {
        let mut h = HeaderMap::new();
        h.insert(
            AUTHORIZATION,
            HeaderValue::from_static("Basic dXNlcjpwYXNz"),
        );
        assert!(extract_bearer(&h).is_none());
    }

    #[test]
    fn extract_bearer_rejects_empty_token() {
        let mut h = HeaderMap::new();
        h.insert(AUTHORIZATION, HeaderValue::from_static("Bearer "));
        assert!(extract_bearer(&h).is_none());
    }

    #[test]
    fn extract_bearer_missing_header() {
        let h = HeaderMap::new();
        assert!(extract_bearer(&h).is_none());
    }

    #[tokio::test]
    async fn extractor_accepts_matching_token() {
        let state = state_with(Some("secret"));
        let req = axum::http::Request::builder()
            .uri("/remote/v1/push-blocks")
            .header(AUTHORIZATION, "Bearer secret")
            .body(())
            .unwrap();
        let (mut parts, _) = req.into_parts();
        let r = RequireBearer::from_request_parts(&mut parts, &state).await;
        assert!(r.is_ok(), "expected ok, got {r:?}");
    }

    #[tokio::test]
    async fn extractor_rejects_missing_header() {
        let state = state_with(Some("secret"));
        let req = axum::http::Request::builder()
            .uri("/remote/v1/push-blocks")
            .body(())
            .unwrap();
        let (mut parts, _) = req.into_parts();
        let r = RequireBearer::from_request_parts(&mut parts, &state).await;
        assert!(matches!(r, Err(AuthRejection::Missing)));
    }

    #[tokio::test]
    async fn extractor_rejects_mismatched_token() {
        let state = state_with(Some("secret"));
        let req = axum::http::Request::builder()
            .uri("/remote/v1/push-blocks")
            .header(AUTHORIZATION, "Bearer wrong")
            .body(())
            .unwrap();
        let (mut parts, _) = req.into_parts();
        let r = RequireBearer::from_request_parts(&mut parts, &state).await;
        assert!(matches!(r, Err(AuthRejection::Mismatch)));
    }

    #[tokio::test]
    async fn extractor_returns_unconfigured_when_token_missing() {
        let state = state_with(None);
        let req = axum::http::Request::builder()
            .uri("/remote/v1/push-blocks")
            .header(AUTHORIZATION, "Bearer anything")
            .body(())
            .unwrap();
        let (mut parts, _) = req.into_parts();
        let r = RequireBearer::from_request_parts(&mut parts, &state).await;
        assert!(matches!(r, Err(AuthRejection::Unconfigured)));
    }

    /// RFC 6750 §3 mandates a `WWW-Authenticate: Bearer` challenge on
    /// every 401 from a bearer-protected resource. A silent drop of
    /// this header would break clients that key their retry logic off
    /// the challenge realm; the regression test binds the header shape
    /// directly against the `IntoResponse` impl so a future refactor
    /// that reorders / drops the tuple entries fails loudly here.
    #[test]
    fn rejection_401_carries_www_authenticate_bearer_realm() {
        for rej in [AuthRejection::Missing, AuthRejection::Mismatch] {
            let resp = rej.into_response();
            assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
            let hdr = resp
                .headers()
                .get(WWW_AUTHENTICATE)
                .unwrap_or_else(|| panic!("WWW-Authenticate missing for {rej:?}"));
            assert_eq!(
                hdr.to_str().unwrap(),
                "Bearer realm=\"mnem\"",
                "challenge shape drifted for {rej:?}"
            );
            // Body is `application/problem+json` per RFC 7807.
            let ct = resp
                .headers()
                .get(axum::http::header::CONTENT_TYPE)
                .expect("content-type present");
            assert_eq!(ct.to_str().unwrap(), "application/problem+json");
        }
    }

    /// The 503 (Unconfigured) branch deliberately does NOT emit a
    /// `WWW-Authenticate` challenge: the server is not asking the
    /// client to retry with better credentials, it is saying writes
    /// are administratively disabled. Pin that asymmetry.
    #[test]
    fn rejection_503_omits_www_authenticate() {
        let resp = AuthRejection::Unconfigured.into_response();
        assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
        assert!(
            resp.headers().get(WWW_AUTHENTICATE).is_none(),
            "503 must not emit a bearer challenge"
        );
    }
}
