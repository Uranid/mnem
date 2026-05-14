//! Item 11: Prometheus counter increment tests.
//!
//! Verifies that counters and histograms registered in [`mnem_http::Metrics`]
//! actually change value when the corresponding HTTP operations happen.
//!
//! Strategy: build a metrics-enabled app, perform one or more operations via
//! `tower::ServiceExt::oneshot`, then scrape `/metrics` and assert that the
//! relevant counter/histogram text output changed.
//!
//! ## Naming note
//!
//! `prometheus-client` 0.23 follows the OpenMetrics convention: it appends
//! `_total` to every counter metric name in the text-exposition output.  When
//! a metric is *registered* as `mnem_http_requests_total` the emitted line is:
//!
//! ```text
//! mnem_http_requests_total_total{method="GET",...} 1
//! ```
//!
//! Histograms use `_sum`, `_count`, and `_bucket{le="..."}` suffixes.  The
//! tests below use these exact rendered names so the assertions pin the
//! actual wire format.
//!
//! ## Why parse text instead of reading `Counter::get()`?
//!
//! `Histogram::get()` is `pub(crate)` in prometheus-client 0.23 and cannot
//! be called from here.  Rather than reach into the internal state for
//! simple counters (which would require exposing `AppState` constructors),
//! all assertions go through the `/metrics` HTTP endpoint -- the same path a
//! real Prometheus scraper would use.
//!
//! Tests are grouped by the operation that drives the increment:
//!   - `http_requests_total_total` via the `track_metrics` middleware
//!   - `commit_duration_seconds` histogram via POST /v1/nodes
//!   - `retrieve_latency_seconds` histogram via GET /v1/retrieve
//!   - `ingest_chunks_total_total` counter via POST /v1/ingest
//!   - `remote_advance_head_total_total` family via POST /remote/v1/advance-head

use axum::body::Body;
use axum::http::{Request, StatusCode};
use http_body_util::BodyExt as _;
use tempfile::TempDir;
use tower::ServiceExt;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Build a metrics-enabled app backed by a fresh temp-dir repo.
/// Returns the router (cloneable) and a `TempDir` guard (must be kept alive
/// for the router's lifetime).
fn make_app() -> (axum::Router, TempDir) {
    let td = TempDir::new().expect("tmp dir");
    let opts = mnem_http::AppOptions {
        allow_labels: Some(true),
        in_memory: false,
        metrics_enabled: true,
        push_token: Some("test-token".into()),
    };
    let app = mnem_http::app_with_options(td.path(), opts).expect("build app");
    (app, td)
}

/// Scrape the `/metrics` endpoint and return its body as a `String`.
async fn scrape(app: axum::Router) -> String {
    let resp = app
        .oneshot(
            Request::builder()
                .uri("/metrics")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(
        resp.status(),
        StatusCode::OK,
        "GET /metrics must return 200"
    );
    let bytes = resp.into_body().collect().await.unwrap().to_bytes();
    String::from_utf8(bytes.to_vec()).expect("metrics body must be UTF-8")
}

/// Parse the value from the first non-comment line that starts with `name`
/// and is followed immediately by `{` or a space.  Returns `None` when no
/// matching line exists.
///
/// prometheus-client 0.23 emits lines in one of these shapes:
///   `<name>{<labels>} <value>`
///   `<name> <value>`
///
/// For counters registered as e.g. `mnem_ingest_chunks_total` the rendered
/// line is `mnem_ingest_chunks_total_total <value>` (extra `_total` suffix).
fn extract_metric_value(text: &str, name: &str) -> Option<f64> {
    for line in text.lines() {
        if line.starts_with('#') {
            continue;
        }
        // strip_prefix returns None for lines that don't match; use
        // `match` + `continue` (not `?`) to avoid returning `None`
        // from the whole function on the first non-matching line.
        let rest = match line.strip_prefix(name) {
            Some(r) => r,
            None => continue,
        };
        // Name boundary: must be followed by `{` (labelled) or space (bare).
        if !rest.starts_with('{') && !rest.starts_with(' ') {
            continue;
        }
        let value_str = line.split_whitespace().last()?;
        return value_str.parse::<f64>().ok();
    }
    None
}

/// Sum all values from lines whose name starts with `prefix` and is bounded
/// by `{` or space.  Used to aggregate across label-sets (e.g. all methods
/// and routes for `mnem_http_requests_total_total`).
fn sum_metric_values_with_prefix(text: &str, prefix: &str) -> f64 {
    let mut total = 0.0_f64;
    for line in text.lines() {
        if line.starts_with('#') {
            continue;
        }
        let rest = match line.strip_prefix(prefix) {
            Some(r) => r,
            None => continue,
        };
        if !rest.starts_with('{') && !rest.starts_with(' ') {
            continue;
        }
        if let Some(val_str) = line.split_whitespace().last() {
            if let Ok(v) = val_str.parse::<f64>() {
                total += v;
            }
        }
    }
    total
}

// ---------------------------------------------------------------------------
// Tests: `mnem_http_requests_total` (track_metrics middleware)
//
// prometheus-client appends `_total`; the registered name is already
// `mnem_http_requests_total`, so the rendered line is
// `mnem_http_requests_total_total{...} N`.
// ---------------------------------------------------------------------------

/// Sending any non-`/metrics` request must bump the request counter.
/// We perform a GET /v1/healthz (the cheapest route) and verify that:
///   (a) the rendered `_total_total` sum increases, AND
///   (b) a counter line with `method="GET"` appears, confirming the method
///       label is correctly wired for non-POST requests.
#[tokio::test]
async fn http_requests_total_increments_on_healthz() {
    let (app, _td) = make_app();

    // Baseline: fresh app, no requests yet.
    let before_text = scrape(app.clone()).await;
    let before =
        sum_metric_values_with_prefix(&before_text, "mnem_http_requests_total_total");

    // Drive a GET /v1/healthz.
    let _resp = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/v1/healthz")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    let after_text = scrape(app.clone()).await;
    let after =
        sum_metric_values_with_prefix(&after_text, "mnem_http_requests_total_total");

    assert!(
        after > before,
        "mnem_http_requests_total_total should increase after GET /v1/healthz \
         (before={before}, after={after})"
    );

    // Verify the method="GET" label is emitted -- a bug that hardcodes method="POST"
    // on all requests would pass the sum check above but fail here.
    let has_get_label = after_text.lines().any(|l| {
        l.starts_with("mnem_http_requests_total_total") && l.contains("method=\"GET\"")
    });
    assert!(
        has_get_label,
        "expected a mnem_http_requests_total_total line with method=\"GET\" after \
         GET /v1/healthz; got:\n{after_text}"
    );
}

/// POST /v1/nodes must produce a `status=\"200\"` series in the counter family.
#[tokio::test]
async fn http_requests_total_records_post_node_200() {
    let (app, _td) = make_app();

    let body = serde_json::json!({
        "label": "Fact",
        "summary": "counter test node",
        "author": "test-suite"
    });
    let resp = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/nodes")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_vec(&body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(
        resp.status(),
        StatusCode::OK,
        "POST /v1/nodes must return 200 for the counter label assertion to be valid"
    );

    let text = scrape(app.clone()).await;

    // Look for a rendered counter line with the POST + 200 labels.
    let has_200 = text.lines().any(|l| {
        l.starts_with("mnem_http_requests_total_total")
            && l.contains("status=\"200\"")
            && l.contains("method=\"POST\"")
    });
    assert!(
        has_200,
        "expected mnem_http_requests_total_total{{method=\"POST\",status=\"200\"}} \
         after POST /v1/nodes; got:\n{text}"
    );
}

/// Even a non-2xx response bumps the counter -- `track_metrics` fires on
/// every response regardless of status code.  `?text=hello` without an
/// embedder returns 503 (no embedder configured), which exercises this path.
#[tokio::test]
async fn http_requests_total_increments_on_4xx_retrieve() {
    let (app, _td) = make_app();

    let before_text = scrape(app.clone()).await;
    let before =
        sum_metric_values_with_prefix(&before_text, "mnem_http_requests_total_total");

    // GET /v1/retrieve with `text=` and no embedder configured returns 503.
    let resp = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/v1/retrieve?text=hello")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    // Confirm the request actually failed (4xx or 5xx) -- without this the sum
    // check below would pass even if the server returned 200 (vacuous increment).
    // The server returns 503 when no embedder is configured, which is non-success.
    assert!(
        !resp.status().is_success(),
        "GET /v1/retrieve?text=hello must return a non-2xx without an embedder (got {})",
        resp.status()
    );

    let after_text = scrape(app.clone()).await;
    let after =
        sum_metric_values_with_prefix(&after_text, "mnem_http_requests_total_total");

    assert!(
        after > before,
        "mnem_http_requests_total_total should increase even on a 4xx response \
         (before={before}, after={after})"
    );

    // Verify the emitted label carries the actual non-200 status code.
    // A bug that hardcodes status="200" on all requests would pass the sum
    // check above but fail here.
    let has_non_200_label = after_text.lines().any(|l| {
        l.starts_with("mnem_http_requests_total_total")
            && l.contains("method=\"GET\"")
            && !l.contains("status=\"200\"")
    });
    assert!(
        has_non_200_label,
        "expected a mnem_http_requests_total_total line with method=\"GET\" and a \
         non-200 status label after the 4xx retrieve; got:\n{after_text}"
    );
}

// ---------------------------------------------------------------------------
// Tests: `mnem_http_request_duration_seconds` histogram (middleware)
// ---------------------------------------------------------------------------

/// The `http_duration` histogram receives a sample for every non-`/metrics`
/// request.  After one GET /v1/healthz the `_count` line must increase.
#[tokio::test]
async fn http_duration_histogram_gets_sample_on_healthz() {
    let (app, _td) = make_app();

    let before_text = scrape(app.clone()).await;
    let before_count = extract_metric_value(
        &before_text,
        "mnem_http_request_duration_seconds_count",
    )
    .unwrap_or(0.0);

    let _resp = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/v1/healthz")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    let after_text = scrape(app.clone()).await;
    let after_count = extract_metric_value(
        &after_text,
        "mnem_http_request_duration_seconds_count",
    )
    .expect("mnem_http_request_duration_seconds_count must be present in /metrics after a request");

    assert!(
        after_count > before_count,
        "mnem_http_request_duration_seconds_count must increase after GET /v1/healthz \
         (before={before_count}, after={after_count})"
    );
}

// ---------------------------------------------------------------------------
// Tests: `mnem_commit_duration_seconds` histogram
// ---------------------------------------------------------------------------

/// POST /v1/nodes commits a node -- the commit-duration histogram must record
/// a sample (count increases by 1, sum increases by a positive amount).
#[tokio::test]
async fn commit_duration_histogram_gets_sample_after_post_node() {
    let (app, _td) = make_app();

    let before_text = scrape(app.clone()).await;
    let before_count =
        extract_metric_value(&before_text, "mnem_commit_duration_seconds_count").unwrap_or(0.0);

    let body = serde_json::json!({
        "summary": "histogram test node",
        "author": "test-suite"
    });
    let resp = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/nodes")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_vec(&body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(
        resp.status(),
        StatusCode::OK,
        "POST /v1/nodes must succeed for this test to be valid"
    );

    let after_text = scrape(app.clone()).await;
    let after_count =
        extract_metric_value(&after_text, "mnem_commit_duration_seconds_count")
            .expect("mnem_commit_duration_seconds_count must be present after a successful POST /v1/nodes");

    assert!(
        after_count > before_count,
        "mnem_commit_duration_seconds_count must increase after POST /v1/nodes \
         (before={before_count}, after={after_count})"
    );
}

/// After a successful POST /v1/nodes the `_sum` line must be present and
/// non-negative, proving the observed latency was recorded (not clamped to zero).
#[tokio::test]
async fn commit_duration_histogram_sum_is_non_negative_after_commit() {
    let (app, _td) = make_app();

    let body = serde_json::json!({
        "summary": "commit sum test",
        "author": "test-suite"
    });
    let resp = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/nodes")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_vec(&body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK, "POST /v1/nodes must succeed");

    let text = scrape(app.clone()).await;
    let sum = extract_metric_value(&text, "mnem_commit_duration_seconds_sum")
        .expect("mnem_commit_duration_seconds_sum must be present in /metrics after a successful commit");

    assert!(
        sum > 0.0,
        "mnem_commit_duration_seconds_sum must be > 0 after a successful commit \
         (a zero sum means no duration was recorded; got {sum})"
    );
}

// ---------------------------------------------------------------------------
// Tests: `mnem_retrieve_latency_seconds` histogram
// ---------------------------------------------------------------------------

/// GET /v1/retrieve with a `?label=Fact` filter exercises the text-free path
/// (no embedder needed) and calls `ret.execute()` successfully on a fresh repo.
/// The retrieve-latency histogram must record a sample after the call.
///
/// A plain `GET /v1/retrieve` with no parameters is rejected by mnem-core
/// ("no filters or rankers configured") before `execute()` is even called, so
/// the histogram would not be observed.  Using `?label=L` supplies the filter
/// the retriever requires without needing an embedder.
#[tokio::test]
async fn retrieve_latency_histogram_gets_sample_on_label_filter_retrieve() {
    let (app, _td) = make_app();

    let before_text = scrape(app.clone()).await;
    let before_count =
        extract_metric_value(&before_text, "mnem_retrieve_latency_seconds_count").unwrap_or(0.0);

    // First commit a node so the retriever has something to scan.
    let commit_body = serde_json::json!({
        "label": "Fact",
        "summary": "retrieve latency test node",
        "author": "test-suite"
    });
    let commit_resp = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/nodes")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_vec(&commit_body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(
        commit_resp.status(),
        StatusCode::OK,
        "must commit a node before retrieving"
    );

    // GET /v1/retrieve?label=Fact: label filter requires allow_labels=true
    // (set in make_app via allow_labels: Some(true)).  execute() runs
    // synchronously and the latency is observed regardless of result count.
    let resp = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/v1/retrieve?label=Fact")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(
        resp.status(),
        StatusCode::OK,
        "GET /v1/retrieve?label=Fact must return 200"
    );

    let after_text = scrape(app.clone()).await;
    let after_count =
        extract_metric_value(&after_text, "mnem_retrieve_latency_seconds_count")
            .expect("mnem_retrieve_latency_seconds_count must be present after a successful retrieve");

    assert!(
        after_count > before_count,
        "mnem_retrieve_latency_seconds_count must increase after GET /v1/retrieve?label=Fact \
         (before={before_count}, after={after_count})"
    );

    // The histogram _sum must be strictly positive -- any measurable latency is > 0.
    let sum = extract_metric_value(&after_text, "mnem_retrieve_latency_seconds_sum")
        .expect("mnem_retrieve_latency_seconds_sum must be present after a successful retrieve");
    assert!(
        sum > 0.0,
        "mnem_retrieve_latency_seconds_sum must be > 0 after a successful retrieve (got {sum})"
    );
}

/// GET /v1/retrieve with `?text=hello` and no embedder configured fails before
/// `ret.execute()` is reached (returns 4xx).  The retrieve-latency histogram
/// must NOT be incremented -- the latency is only recorded after a successful
/// `execute()` call.
///
/// This is the mirror of the advance-head 401 no-increment test: the handler
/// early-exits before the metric code, so the counter stays flat.
#[tokio::test]
async fn retrieve_latency_not_incremented_on_text_query_without_embedder() {
    let (app, _td) = make_app();

    let before_text = scrape(app.clone()).await;
    let before_count =
        extract_metric_value(&before_text, "mnem_retrieve_latency_seconds_count").unwrap_or(0.0);

    // GET /v1/retrieve?text=hello with no embedder configured returns 4xx
    // before execute() is called, so no latency sample should be recorded.
    let resp = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/v1/retrieve?text=hello")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert!(
        !resp.status().is_success(),
        "GET /v1/retrieve?text=hello must fail without an embedder (got {})",
        resp.status()
    );

    let after_text = scrape(app.clone()).await;
    let after_count =
        extract_metric_value(&after_text, "mnem_retrieve_latency_seconds_count").unwrap_or(0.0);

    assert_eq!(
        after_count, before_count,
        "mnem_retrieve_latency_seconds_count must NOT increment when retrieve \
         fails before execute() (before={before_count}, after={after_count})"
    );
}

// ---------------------------------------------------------------------------
// Tests: `mnem_ingest_chunks_total` counter
// ---------------------------------------------------------------------------

/// POST /v1/ingest with a Markdown payload must bump `mnem_ingest_chunks_total`.
/// The rendered line is `mnem_ingest_chunks_total_total N` (double `_total`).
///
/// The JSON body field for inline text is `text`, NOT `content` (see
/// `IngestJsonBody` in handlers_ingest.rs). Sending the wrong key causes a
/// 400 before the ingest path is reached and no counter is recorded.
#[tokio::test]
async fn ingest_chunks_total_increments_after_ingest() {
    let (app, _td) = make_app();

    let before_text = scrape(app.clone()).await;
    // prometheus-client appends _total to the registered name
    // "mnem_ingest_chunks_total", so the wire line is
    // "mnem_ingest_chunks_total_total N".
    let before =
        extract_metric_value(&before_text, "mnem_ingest_chunks_total_total").unwrap_or(0.0);

    let body = serde_json::json!({
        "text": "# Test\nThis is a test document for counter increment verification.",
        "kind": "markdown",
        "author": "test-suite"
    });
    let resp = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/ingest")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_vec(&body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(
        resp.status(),
        StatusCode::OK,
        "POST /v1/ingest must return 200 for the counter assertion to be valid"
    );

    let after_text = scrape(app.clone()).await;
    let after =
        extract_metric_value(&after_text, "mnem_ingest_chunks_total_total")
            .expect("mnem_ingest_chunks_total_total must be present after a successful ingest");

    assert!(
        after > before,
        "mnem_ingest_chunks_total_total must increase after a successful ingest \
         (before={before}, after={after})"
    );
}

/// POST /v1/ingest that omits the required `text` field (sending `content`
/// instead) is rejected before the ingest handler body executes.
/// `mnem_ingest_chunks_total_total` must NOT increment -- the counter is only
/// updated after chunks are actually produced.
///
/// Strategy: first perform a successful ingest so the counter line is
/// guaranteed to exist in the scrape output (non-zero baseline), then
/// verify the bad-field request leaves it unchanged.  This avoids the
/// vacuous 0==0 pass that would occur if the counter line is absent before
/// any ingest has run.
///
/// This is the ingest mirror of the advance-head 401 no-increment test.
#[tokio::test]
async fn ingest_chunks_not_incremented_on_bad_field() {
    let (app, _td) = make_app();

    // Seed: one successful ingest to ensure the counter is initialized
    // and present in the scrape output.
    let seed_body = serde_json::json!({
        "text": "seed document to initialize the chunk counter",
        "kind": "markdown",
        "author": "test-suite"
    });
    let seed_resp = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/ingest")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_vec(&seed_body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(
        seed_resp.status(),
        StatusCode::OK,
        "seed ingest must succeed so the counter is initialized"
    );

    // Baseline: counter is guaranteed non-zero and present in the scrape.
    let before_text = scrape(app.clone()).await;
    let before =
        extract_metric_value(&before_text, "mnem_ingest_chunks_total_total")
            .expect("mnem_ingest_chunks_total_total must be present after the seed ingest");

    // `IngestJsonBody.text` is required.  Sending `content` (the wrong key)
    // causes serde deserialization to fail, returning 4xx before any chunks
    // are produced or the counter is touched.
    let bad_body = serde_json::json!({
        "content": "this key is wrong and will cause a 4xx",
        "author": "test-suite"
    });
    let resp = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/ingest")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_vec(&bad_body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert!(
        !resp.status().is_success(),
        "POST /v1/ingest with wrong field must fail (got {})",
        resp.status()
    );

    let after_text = scrape(app.clone()).await;
    let after =
        extract_metric_value(&after_text, "mnem_ingest_chunks_total_total")
            .expect("mnem_ingest_chunks_total_total must still be present after the failed request");

    assert_eq!(
        before, after,
        "mnem_ingest_chunks_total_total must NOT increment when ingest fails before \
         chunks are produced (before={before}, after={after})"
    );
}

/// `mnem_ingest_duration_seconds` must accumulate at least one sample
/// (_count > 0) after a successful ingest run.
#[tokio::test]
async fn ingest_duration_histogram_gets_sample_on_success() {
    let (app, _td) = make_app();

    let before_text = scrape(app.clone()).await;
    let before_count =
        extract_metric_value(&before_text, "mnem_ingest_duration_seconds_count").unwrap_or(0.0);

    let body = serde_json::json!({
        "text": "Ingest duration histogram test document.",
        "kind": "markdown",
        "author": "test-suite"
    });
    let resp = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/ingest")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_vec(&body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(
        resp.status(),
        StatusCode::OK,
        "POST /v1/ingest must return 200 for the histogram assertion to be valid"
    );

    let after_text = scrape(app.clone()).await;
    let after_count =
        extract_metric_value(&after_text, "mnem_ingest_duration_seconds_count")
            .expect("mnem_ingest_duration_seconds_count must be present after a successful ingest");

    assert!(
        after_count > before_count,
        "mnem_ingest_duration_seconds_count must increase after a successful ingest \
         (before={before_count}, after={after_count})"
    );

    // The histogram _sum must also be strictly positive.
    let sum = extract_metric_value(&after_text, "mnem_ingest_duration_seconds_sum")
        .expect("mnem_ingest_duration_seconds_sum must be present after a successful ingest");
    assert!(
        sum > 0.0,
        "mnem_ingest_duration_seconds_sum must be > 0 after a successful ingest (got {sum})"
    );
}

// ---------------------------------------------------------------------------
// Tests: `mnem_remote_advance_head_total` Family counter
// ---------------------------------------------------------------------------

/// POST /remote/v1/advance-head with an invalid bearer token must NOT bump any
/// `advance-head` counter -- the `RequireBearer` extractor short-circuits
/// before the metric code is reached.
#[tokio::test]
async fn remote_advance_head_not_incremented_on_auth_failure() {
    let (app, _td) = make_app();

    let before_text = scrape(app.clone()).await;
    // Family counter; sum over all label-sets (rendered as _total_total).
    let before = sum_metric_values_with_prefix(
        &before_text,
        "mnem_remote_advance_head_total_total",
    );

    let mh = mnem_core::id::Multihash::sha2_256(b"x");
    let cid = mnem_core::id::Cid::new(mnem_core::id::CODEC_RAW, mh);
    let body_json = serde_json::json!({ "old": cid.to_string(), "new": cid.to_string() });

    let resp = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/remote/v1/advance-head")
                .header("authorization", "Bearer wrong-token")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_vec(&body_json).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);

    let after_text = scrape(app.clone()).await;
    let after = sum_metric_values_with_prefix(
        &after_text,
        "mnem_remote_advance_head_total_total",
    );

    assert_eq!(
        before, after,
        "mnem_remote_advance_head_total must NOT increment on a 401 auth failure \
         (before={before}, after={after})"
    );
}

/// POST /remote/v1/advance-head with a correct token but stale CAS must
/// return 409 Conflict AND bump the `result=\"cas_mismatch\"` counter series.
#[tokio::test]
async fn remote_advance_head_cas_mismatch_counter_increments() {
    let (app, _td) = make_app();

    let before_text = scrape(app.clone()).await;
    let before = sum_metric_values_with_prefix(
        &before_text,
        "mnem_remote_advance_head_total_total",
    );

    // A CID that is not the current head -> CAS mismatch.
    let mh = mnem_core::id::Multihash::sha2_256(b"nonexistent");
    let cid = mnem_core::id::Cid::new(mnem_core::id::CODEC_RAW, mh);
    let body_json = serde_json::json!({ "old": cid.to_string(), "new": cid.to_string() });

    let resp = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/remote/v1/advance-head")
                .header("authorization", "Bearer test-token")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_vec(&body_json).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(
        resp.status(),
        StatusCode::CONFLICT,
        "advance-head with wrong CAS must return 409"
    );

    let after_text = scrape(app.clone()).await;
    let after = sum_metric_values_with_prefix(
        &after_text,
        "mnem_remote_advance_head_total_total",
    );

    assert!(
        after > before,
        "mnem_remote_advance_head_total_total must increase on a CAS mismatch \
         (before={before}, after={after})"
    );

    // Verify the closed-vocabulary label `cas_mismatch` appears.
    let has_cas_label = after_text.lines().any(|l| {
        l.starts_with("mnem_remote_advance_head_total_total")
            && l.contains("result=\"cas_mismatch\"")
    });
    assert!(
        has_cas_label,
        "expected mnem_remote_advance_head_total_total{{result=\"cas_mismatch\"}} \
         series after a 409 CAS mismatch; got:\n{after_text}"
    );
}

/// POST /remote/v1/advance-head with the correct token AND the matching
/// current head CID must return 200 OK and bump the `result=\"success\"`
/// counter series.
///
/// Strategy: discover the real current head via `GET /remote/v1/refs` (which
/// returns the root-op CID written by `ReadonlyRepo::init`), then POST
/// advance-head with `old = that CID`.  The ancestry check is skipped because
/// the root-op block decodes as an Operation, not a Commit, so `old_is_commit`
/// is false and the walk is bypassed, allowing any valid `new` CID.
#[tokio::test]
async fn remote_advance_head_success_counter_increments() {
    let (app, _td) = make_app();

    // Step 1: discover the current head via GET /remote/v1/refs.
    let refs_resp = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/remote/v1/refs")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(refs_resp.status(), StatusCode::OK, "GET /remote/v1/refs must return 200");
    let refs_bytes = refs_resp.into_body().collect().await.unwrap().to_bytes();
    let refs_json: serde_json::Value =
        serde_json::from_slice(&refs_bytes).expect("refs response must be valid JSON");

    // Step 2: baseline counter before advance-head.
    let before_text = scrape(app.clone()).await;
    let before = sum_metric_values_with_prefix(
        &before_text,
        "mnem_remote_advance_head_total_total",
    );

    // Step 3: build the advance-head request body.
    // `ReadonlyRepo::init` always writes a root-op so `head` is non-null.
    // If for some reason it is null, omit `old` to exercise the first-push path.
    let mh = mnem_core::id::Multihash::sha2_256(b"advance-head-success-new-target");
    let new_cid = mnem_core::id::Cid::new(mnem_core::id::CODEC_RAW, mh);
    // `ReadonlyRepo::init` always writes a root-op, so `head` must be present.
    // Panic loudly if absent -- that would indicate a bug in the server, not a
    // valid first-push scenario.
    let current_head = refs_json["head"]
        .as_str()
        .expect("GET /remote/v1/refs must return a non-null `head` field (ReadonlyRepo::init always writes a root-op)");
    let body_json = serde_json::json!({ "old": current_head, "new": new_cid.to_string() });

    // Step 4: POST advance-head with the correct token and matching `old`.
    let resp = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/remote/v1/advance-head")
                .header("authorization", "Bearer test-token")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_vec(&body_json).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(
        resp.status(),
        StatusCode::OK,
        "advance-head with correct old CID and valid token must return 200"
    );

    // Step 5: counter must have increased with label result="success".
    let after_text = scrape(app.clone()).await;
    let after = sum_metric_values_with_prefix(
        &after_text,
        "mnem_remote_advance_head_total_total",
    );
    assert!(
        after > before,
        "mnem_remote_advance_head_total_total must increase on a successful advance-head \
         (before={before}, after={after})"
    );
    let has_success_label = after_text.lines().any(|l| {
        l.starts_with("mnem_remote_advance_head_total_total")
            && l.contains("result=\"success\"")
    });
    assert!(
        has_success_label,
        "expected mnem_remote_advance_head_total_total{{result=\"success\"}} \
         series after a 200 advance-head; got:\n{after_text}"
    );
}

// ---------------------------------------------------------------------------
// Tests: monotonic accumulation across multiple operations
// ---------------------------------------------------------------------------

/// Three sequential POST /v1/nodes requests must push both the request counter
/// and the commit-duration histogram count up by at least three.
#[tokio::test]
async fn counters_accumulate_across_multiple_commits() {
    let (app, _td) = make_app();

    let before_text = scrape(app.clone()).await;
    let before_req =
        sum_metric_values_with_prefix(&before_text, "mnem_http_requests_total_total");
    let before_commit =
        extract_metric_value(&before_text, "mnem_commit_duration_seconds_count").unwrap_or(0.0);

    for i in 0_u32..3 {
        let body = serde_json::json!({
            "summary": format!("accumulation test node {i}"),
            "author": "test-suite"
        });
        let resp = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/nodes")
                    .header("content-type", "application/json")
                    .body(Body::from(serde_json::to_vec(&body).unwrap()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK, "commit {i} must succeed");
    }

    let after_text = scrape(app.clone()).await;
    let after_req =
        sum_metric_values_with_prefix(&after_text, "mnem_http_requests_total_total");
    let after_commit =
        extract_metric_value(&after_text, "mnem_commit_duration_seconds_count").unwrap_or(0.0);

    // Each test uses make_app() which creates a fresh registry with no shared
    // state, so the delta must be exactly 3, not just >= 3.
    assert_eq!(
        after_req,
        before_req + 3.0,
        "mnem_http_requests_total_total should increase by exactly 3 after 3 POSTs \
         (before={before_req}, after={after_req})"
    );
    assert_eq!(
        after_commit,
        before_commit + 3.0,
        "mnem_commit_duration_seconds_count should increase by exactly 3 after 3 commits \
         (before={before_commit}, after={after_commit})"
    );
}

// ---------------------------------------------------------------------------
// Tests: scrape exemption -- GET /metrics must NOT increment the counter
// ---------------------------------------------------------------------------

/// The `track_metrics` middleware explicitly skips the `/metrics` route to
/// avoid scrape-induced counter inflation.  Two sequential scrapes must not
/// change either `mnem_http_requests_total_total` (the counter family) or
/// `mnem_http_request_duration_seconds_count` (the duration histogram).
#[tokio::test]
async fn scrape_does_not_increment_http_requests_total() {
    let (app, _td) = make_app();

    let first_text = scrape(app.clone()).await;
    let req_count_after_first =
        sum_metric_values_with_prefix(&first_text, "mnem_http_requests_total_total");
    let dur_count_after_first =
        extract_metric_value(&first_text, "mnem_http_request_duration_seconds_count")
            .unwrap_or(0.0);

    let second_text = scrape(app.clone()).await;
    let req_count_after_second =
        sum_metric_values_with_prefix(&second_text, "mnem_http_requests_total_total");
    let dur_count_after_second =
        extract_metric_value(&second_text, "mnem_http_request_duration_seconds_count")
            .unwrap_or(0.0);

    assert_eq!(
        req_count_after_first, req_count_after_second,
        "GET /metrics scrapes must not bump mnem_http_requests_total_total \
         (first={req_count_after_first}, second={req_count_after_second})"
    );
    assert_eq!(
        dur_count_after_first, dur_count_after_second,
        "GET /metrics scrapes must not bump mnem_http_request_duration_seconds_count \
         (first={dur_count_after_first}, second={dur_count_after_second})"
    );
}

// ---------------------------------------------------------------------------
// Tests: /metrics endpoint presence when metrics_enabled: true / false
// ---------------------------------------------------------------------------

/// When `AppOptions::metrics_enabled` is `true`, GET /metrics must return 200
/// with a non-empty OpenMetrics body.  This is the happy-path contract of the
/// metrics endpoint itself (the `scrape()` helper also asserts 200, but this
/// test makes the contract explicit and visible).
#[tokio::test]
async fn metrics_endpoint_returns_200_when_enabled() {
    let (app, _td) = make_app();
    let resp = app
        .oneshot(
            Request::builder()
                .uri("/metrics")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(
        resp.status(),
        StatusCode::OK,
        "GET /metrics must return 200 when metrics_enabled is true"
    );
    let bytes = resp.into_body().collect().await.unwrap().to_bytes();
    assert!(
        !bytes.is_empty(),
        "GET /metrics body must be non-empty when metrics_enabled is true"
    );
}

/// When `AppOptions::metrics_enabled` is `false`, the `/metrics` route is not
/// mounted.  GET /metrics must return 404 Not Found.
///
/// The tracking middleware still runs (counters are still updated in RAM),
/// but the text-exposition endpoint is simply absent so a scraper can't reach
/// it.  This tests the "opt-in exposure" contract of `metrics_enabled`.
#[tokio::test]
async fn metrics_endpoint_absent_when_metrics_disabled() {
    let td = TempDir::new().expect("tmp dir");
    let opts = mnem_http::AppOptions {
        allow_labels: Some(false),
        in_memory: false,
        metrics_enabled: false,
        push_token: None,
    };
    let app = mnem_http::app_with_options(td.path(), opts).expect("build disabled-metrics app");

    let resp = app
        .oneshot(
            Request::builder()
                .uri("/metrics")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(
        resp.status(),
        StatusCode::NOT_FOUND,
        "GET /metrics must return 404 when metrics_enabled is false"
    );
}
