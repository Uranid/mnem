//! Per-request correlation id (a.k.a. `X-Request-Id`) middleware.
//!
//! Contract:
//!
//! - If the incoming request carries an `X-Request-Id` header whose
//!   value is a printable ASCII string of length 8..=128, we REUSE it.
//!   Callers (gateways, edge proxies, other services) get to thread
//!   one id end-to-end.
//! - Otherwise we MINT a fresh `UUIDv7` (time-sortable, 128-bit).
//! - The id is attached to a per-request tracing span named
//!   `http_request` so every log line emitted under that request
//!   carries it automatically. Structured-log consumers filter by
//!   `correlation_id=` to follow a request across the codebase.
//! - The id is ECHOED back in the `X-Request-Id` response header so
//!   the caller can correlate their own logs with ours.
//!
//! Cardinality discipline : correlation ids live in tracing
//! fields + response headers ONLY. They are NEVER used as a Prometheus
//! label. A 128-bit id would blow out every histogram bucket family
//! in the registry; see `docs/LOGGING.md` for the operator guidance.

use axum::extract::Request;
use axum::http::HeaderName;
use axum::http::header::{HeaderValue, InvalidHeaderValue};
use axum::middleware::Next;
use axum::response::Response;
use tracing::Instrument;
use uuid::Uuid;

/// Canonical correlation-id header. Lower-case because `hyper` stores
/// header names case-insensitively but we construct one static
/// `HeaderName` to avoid per-request allocation.
pub(crate) const CORRELATION_HEADER: &str = "x-request-id";

/// Minimum accepted length for a caller-supplied correlation id. Eight
/// chars rejects empty / single-char values that are almost certainly
/// misconfigurations. Shorter than the 16 chars the typical
/// shortened-UUID carries, so trace-id echoes from opentelemetry
/// clients still round-trip.
const MIN_CALLER_ID_LEN: usize = 8;

/// Maximum accepted length. 128 is 2x a 36-char hyphenated UUID and
/// still well under the 8 KiB header-line limit most proxies enforce;
/// rejects pathological inputs that would bloat log lines.
const MAX_CALLER_ID_LEN: usize = 128;

/// Axum middleware that extracts-or-mints a correlation id, attaches
/// it to a per-request tracing span, and echoes it in the response.
///
/// Installed ONCE at router construction. Fires BEFORE the
/// `track_metrics` middleware so metric recording happens inside the
/// span (and so the response header propagates through the metrics
/// layer untouched).
pub(crate) async fn correlation_id(req: Request, next: Next) -> Response {
    let id = extract_or_mint(&req);

    // Instrument the downstream response future with a span carrying
    // `correlation_id` as a string field. `tracing_subscriber`'s
    // JSON formatter renders this as one field per log line; the
    // `pretty` formatter renders it at the end of every event's
    // header. Span name `http_request` matches the `tower_http::trace`
    // convention so existing grep patterns keep working.
    let span = tracing::info_span!(
        "http_request",
        correlation_id = %id,
        method = %req.method(),
        uri = %req.uri(),
    );

    let mut response = next.run(req).instrument(span).await;

    // Echo the id back. Unwrap is safe: `id` is either a
    // caller-supplied value already validated as printable ASCII or a
    // UUIDv7 hex that is ASCII by construction. If the caller sent a
    // header `hyper` rejected upstream, we wouldn't be here at all.
    if let Ok(val) = HeaderValue::from_str(&id) {
        response
            .headers_mut()
            .insert(HeaderName::from_static(CORRELATION_HEADER), val);
    }
    response
}

/// Resolve the correlation id for this request. Pulled out as a pure
/// function so unit tests can cover reuse-vs-mint cases without
/// spinning up axum.
pub(crate) fn extract_or_mint(req: &Request) -> String {
    if let Some(raw) = req.headers().get(CORRELATION_HEADER)
        && let Ok(s) = raw.to_str()
    {
        let trimmed = s.trim();
        if is_acceptable_caller_id(trimmed) {
            return trimmed.to_string();
        }
    }
    mint_uuid_v7()
}

/// `UUIDv7`: 48-bit Unix-ms timestamp + 80 random bits. Time-sortable,
/// collision-resistant (2^80 within the same ms), and safe to emit in
/// logs.
fn mint_uuid_v7() -> String {
    Uuid::now_v7().to_string()
}

/// Reject caller-supplied ids that are empty, too short, too long, or
/// contain control characters. Everything else round-trips verbatim so
/// upstream tracing systems (`traceparent`, `x-amzn-trace-id`, ...)
/// keep working unchanged.
fn is_acceptable_caller_id(s: &str) -> bool {
    let len = s.len();
    if !(MIN_CALLER_ID_LEN..=MAX_CALLER_ID_LEN).contains(&len) {
        return false;
    }
    // Printable ASCII only. Rules out control chars, UTF-8 multibyte,
    // and header-injection attempts (CR/LF are already caught by
    // `HeaderValue::from_str` downstream, but we belt-and-braces here
    // because an accepted id flows into log lines too).
    s.chars().all(|c| c.is_ascii_graphic() || c == ' ')
}

/// Intentionally unused: re-exported for tests that want to assert
/// header-value parsing parity with axum's own rules.
#[allow(dead_code)]
pub(crate) fn _header_value(id: &str) -> Result<HeaderValue, InvalidHeaderValue> {
    HeaderValue::from_str(id)
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use axum::http::Request;

    #[test]
    fn reuses_valid_caller_id() {
        let req = Request::builder()
            .uri("/v1/healthz")
            .header(CORRELATION_HEADER, "req-abc12345")
            .body(Body::empty())
            .unwrap();
        assert_eq!(extract_or_mint(&req), "req-abc12345");
    }

    #[test]
    fn mints_when_header_missing() {
        let req = Request::builder()
            .uri("/v1/healthz")
            .body(Body::empty())
            .unwrap();
        let id = extract_or_mint(&req);
        // UUIDv7 hex string: 36 chars, 4 hyphens.
        assert_eq!(id.len(), 36);
        assert_eq!(id.matches('-').count(), 4);
    }

    #[test]
    fn rejects_too_short_caller_id() {
        // Length 7, just under the floor. We mint instead of reusing.
        let req = Request::builder()
            .uri("/v1/healthz")
            .header(CORRELATION_HEADER, "short-1")
            .body(Body::empty())
            .unwrap();
        let id = extract_or_mint(&req);
        assert_ne!(id, "short-1");
        assert_eq!(id.len(), 36, "expected UUIDv7, got {id}");
    }

    #[test]
    fn rejects_too_long_caller_id() {
        let too_long = "x".repeat(MAX_CALLER_ID_LEN + 1);
        let req = Request::builder()
            .uri("/v1/healthz")
            .header(CORRELATION_HEADER, &too_long)
            .body(Body::empty())
            .unwrap();
        let id = extract_or_mint(&req);
        assert_ne!(id, too_long);
    }

    #[test]
    fn two_minted_ids_differ() {
        // Time-sortable but still collision-resistant: two adjacent
        // mints must not collide.
        let a = mint_uuid_v7();
        let b = mint_uuid_v7();
        assert_ne!(a, b);
    }

    #[test]
    fn accepts_opentelemetry_style_id() {
        // `traceparent`-shaped ids: hex + hyphens, well within the
        // length window. Must round-trip unchanged.
        let caller = "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01";
        let req = Request::builder()
            .uri("/v1/healthz")
            .header(CORRELATION_HEADER, caller)
            .body(Body::empty())
            .unwrap();
        assert_eq!(extract_or_mint(&req), caller);
    }

    #[test]
    fn rejects_control_chars() {
        // `HeaderValue` would reject CR/LF anyway; we defend on our
        // own layer because an accepted id flows into log fields.
        // Axum's `HeaderValue::from_str` rejects control chars so this
        // header never reaches our middleware in practice, but the
        // pure-function contract still must refuse them.
        assert!(!is_acceptable_caller_id("abcdef\x01gh"));
        assert!(!is_acceptable_caller_id("abc\tdefg"));
    }
}
