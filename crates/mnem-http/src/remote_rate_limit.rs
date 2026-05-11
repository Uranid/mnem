//! Token-bucket rate limiter for the `/remote/v1/*` transport surface.
//!
//! Protects the four remote endpoints (`refs`, `fetch-blocks`,
//! `push-blocks`, `advance-head`) from request floods that would exhaust
//! CPU, blockstore I/O, or CAR-serialisation memory. The limiter is
//! **process-global** (shared across all remote routes) and enforces a
//! combined ceiling of 100 requests/second with a burst allowance of 50.
//!
//! # Algorithm — token bucket
//!
//! A token bucket is maintained with:
//! - `capacity` = 50 tokens (burst ceiling)
//! - `refill rate` = 100 tokens/second (steady-state throughput)
//!
//! On each incoming request the middleware tries to consume one token.
//! If a token is available the request proceeds; if the bucket is empty
//! it returns **HTTP 429 Too Many Requests** with a
//! `application/problem+json` RFC 7807 body and a `Retry-After: 1`
//! header (the bucket refills within one second).
//!
//! The bucket is represented as the number of available tokens in 1/100
//! fractional token units stored in an `AtomicU64`. Refill is lazy: the
//! middleware computes how many tokens have accrued since the last refill
//! by comparing the current timestamp against a second `AtomicU64` that
//! holds the last-refill instant as `tokio::time::Instant::elapsed`
//! nanoseconds. No background task or timer is required.
//!
//! # Scope
//!
//! Rate-limiting is applied only to the `/remote/v1/*` group (via
//! `axum::middleware::from_fn`). The standard `/v1/*` surface is not
//! affected. The limiter is keyed on the server as a whole, not per
//! IP, because remote sync clients are trusted peers whose IP space
//! is not predictable.
//!
//! # Tunability
//!
//! Two environment variables override the defaults at server startup:
//! - `MNEM_REMOTE_RATE_PER_SEC` — token refill rate (requests/second).
//!   Default: 100.
//! - `MNEM_REMOTE_RATE_BURST` — bucket capacity (burst ceiling).
//!   Default: 50.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};

/// Shared state for the token-bucket limiter.
///
/// Clone-cheap: all fields are wrapped in `Arc`; cloning just bumps
/// refcounts so axum's middleware infrastructure can `Clone` the value
/// freely without copying the atomic state.
#[derive(Clone, Debug)]
pub(crate) struct RemoteRateLimiter {
    inner: Arc<RateLimiterInner>,
}

#[derive(Debug)]
struct RateLimiterInner {
    /// Available tokens × 1000 (fixed-point). Using integer math avoids
    /// floating-point atomics. capacity × 1000 is the ceiling.
    tokens_milli: AtomicU64,
    /// Nanoseconds since `epoch` of the last refill. `epoch` is
    /// `Instant::now()` at construction and is subtracted from all
    /// subsequent `Instant::now()` calls to produce a monotonic u64.
    last_refill_ns: AtomicU64,
    /// The `Instant` captured at construction, used as the epoch for
    /// converting `Instant::now()` to a monotonic `u64`. `Instant`
    /// arithmetic is platform-monotonic; wrapping through `AtomicU64`
    /// keeps the state lock-free.
    epoch: Instant,
    /// Tokens added per nanosecond × 1_000_000 (fixed-point u64).
    /// Precomputed at construction: `per_sec * 1_000_000 / 1_000_000_000`
    /// = `per_sec / 1_000` (but we keep the numerator larger to avoid
    /// integer truncation on low rates). Stored as
    /// `per_sec * 1_000_000` so the refill formula stays in u64.
    refill_per_ns_milli_mega: u64,
    /// Maximum tokens × 1000 (fixed-point ceiling).
    capacity_milli: u64,
}

impl RemoteRateLimiter {
    /// Construct a new limiter. `per_second` is the steady-state
    /// throughput cap; `burst` is the bucket capacity.
    ///
    /// The bucket starts full so a freshly-started server can handle an
    /// immediate burst without penalising the first client.
    pub(crate) fn new(per_second: u64, burst: u64) -> Self {
        let capacity_milli = burst.saturating_mul(1_000);
        // refill_per_ns: we want to add `per_second` tokens per 1e9 ns.
        // In milli-tokens: add `per_second * 1_000` milli-tokens per 1e9 ns.
        // Stored as the numerator of (x / 1_000_000_000) to keep u64 resolution.
        // On consume we multiply elapsed_ns * refill_per_ns_milli_mega / 1_000_000.
        let refill_per_ns_milli_mega = per_second.saturating_mul(1_000_000);
        Self {
            inner: Arc::new(RateLimiterInner {
                tokens_milli: AtomicU64::new(capacity_milli),
                last_refill_ns: AtomicU64::new(0),
                epoch: Instant::now(),
                refill_per_ns_milli_mega,
                capacity_milli,
            }),
        }
    }

    /// Attempt to consume one token. Returns `true` if the request is
    /// allowed, `false` if the bucket is exhausted.
    pub(crate) fn try_consume(&self) -> bool {
        let inner = &*self.inner;

        // --- refill ---
        let now_ns = inner.epoch.elapsed().as_nanos().min(u64::MAX as u128) as u64;

        // Load last refill timestamp. We try to CAS it to `now_ns` so
        // only one concurrent caller actually performs the refill; the
        // others see a stale but still-correct token count (they may
        // get a false deny on a pathological race, acceptable).
        let last_ns = inner.last_refill_ns.load(Ordering::Relaxed);
        let elapsed_ns = now_ns.saturating_sub(last_ns);

        if elapsed_ns > 0 {
            // Compute tokens to add in milli-token units.
            // accrued = elapsed_ns * per_sec * 1_000_000 / 1_000_000_000
            //         = elapsed_ns * refill_per_ns_milli_mega / 1_000_000_000
            let accrued_milli =
                elapsed_ns.saturating_mul(inner.refill_per_ns_milli_mega) / 1_000_000_000;

            if accrued_milli > 0 {
                // CAS the refill timestamp; if another thread beat us,
                // skip (their refill already ran or will run).
                if inner
                    .last_refill_ns
                    .compare_exchange(last_ns, now_ns, Ordering::Relaxed, Ordering::Relaxed)
                    .is_ok()
                {
                    // Saturating-add tokens up to capacity.
                    let prev = inner
                        .tokens_milli
                        .fetch_add(accrued_milli, Ordering::Relaxed);
                    let total = prev.saturating_add(accrued_milli);
                    if total > inner.capacity_milli {
                        inner
                            .tokens_milli
                            .store(inner.capacity_milli, Ordering::Relaxed);
                    }
                }
            }
        }

        // --- consume ---
        // 1 request = 1_000 milli-tokens.
        const ONE_TOKEN_MILLI: u64 = 1_000;
        let prev = inner.tokens_milli.load(Ordering::Relaxed);
        if prev < ONE_TOKEN_MILLI {
            return false;
        }
        // Best-effort CAS; on failure another concurrent request already
        // consumed. Retry once; if that also fails, deny conservatively.
        // This keeps the limiter lock-free without spinning.
        inner
            .tokens_milli
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |t| {
                t.checked_sub(ONE_TOKEN_MILLI)
            })
            .is_ok()
    }

    /// Load the defaults from environment variables, falling back to
    /// the standard parameters (100 req/s, burst 50).
    pub(crate) fn from_env() -> Self {
        let per_sec: u64 = std::env::var("MNEM_REMOTE_RATE_PER_SEC")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(100);
        let burst: u64 = std::env::var("MNEM_REMOTE_RATE_BURST")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(50);
        Self::new(per_sec, burst)
    }
}

/// Axum middleware function. Injects the [`RemoteRateLimiter`] as a
/// closure-captured `Arc`; axum clones the closure on each request so
/// the `Arc` refcount is the only allocation overhead.
///
/// On exhaustion: returns **429** with a minimal RFC 7807 problem
/// document and `Retry-After: 1` (bucket refills within one second).
pub(crate) async fn remote_rate_limit_middleware(
    limiter: Arc<RemoteRateLimiter>,
    req: Request<Body>,
    next: Next,
) -> Response {
    if limiter.try_consume() {
        next.run(req).await
    } else {
        rate_limit_response()
    }
}

/// Build the 429 response emitted when the bucket is empty.
fn rate_limit_response() -> Response {
    let body = serde_json::json!({
        "type": "https://mnem.dev/errors/remote/rate-limited",
        "title": "Too Many Requests",
        "status": 429,
        "detail": "remote protocol rate limit exceeded; retry after 1 second",
    });
    (
        StatusCode::TOO_MANY_REQUESTS,
        [
            (axum::http::header::CONTENT_TYPE, "application/problem+json"),
            // RFC 7231 §7.1.3: Retry-After in seconds.
            (axum::http::header::RETRY_AFTER, "1"),
        ],
        body.to_string(),
    )
        .into_response()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn full_bucket_allows_burst_requests() {
        // burst = 5, per_sec = 10
        let limiter = RemoteRateLimiter::new(10, 5);
        // A full bucket should allow exactly 5 requests before exhausting.
        for i in 0..5 {
            assert!(
                limiter.try_consume(),
                "request {i} should be allowed from a full bucket"
            );
        }
        // 6th request should be denied.
        assert!(
            !limiter.try_consume(),
            "6th request should be denied (bucket exhausted)"
        );
    }

    #[test]
    fn empty_bucket_refills_after_elapsed_time() {
        // Drain a small bucket.
        let limiter = RemoteRateLimiter::new(1_000_000, 3);
        assert!(limiter.try_consume());
        assert!(limiter.try_consume());
        assert!(limiter.try_consume());
        assert!(!limiter.try_consume()); // empty

        // Simulate elapsed time by backdating last_refill_ns to 2
        // seconds ago so the next try_consume sees 2_000_000_000 ns of
        // elapsed time and accrues 2_000_000 * 1 milli-tokens >> 3_000
        // needed for one token.
        let two_seconds_ago_ns = limiter
            .inner
            .epoch
            .elapsed()
            .as_nanos()
            .min(u64::MAX as u128) as u64;
        let backdated = two_seconds_ago_ns.saturating_sub(2_000_000_000);
        limiter
            .inner
            .last_refill_ns
            .store(backdated, Ordering::Relaxed);

        // Now a token should be available.
        assert!(
            limiter.try_consume(),
            "bucket should have refilled after simulated 2 s elapsed"
        );
    }

    #[test]
    fn from_env_uses_defaults_when_vars_absent() {
        // Ensure env vars are not set (they won't be in CI).
        std::env::remove_var("MNEM_REMOTE_RATE_PER_SEC");
        std::env::remove_var("MNEM_REMOTE_RATE_BURST");
        let limiter = RemoteRateLimiter::from_env();
        // Capacity should be 50 * 1000 = 50_000 milli-tokens.
        assert_eq!(
            limiter.inner.capacity_milli, 50_000,
            "default burst should be 50"
        );
    }
}
