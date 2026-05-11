//! Verify the `/metrics` Prometheus endpoint is mounted (or not)
//! according to `AppOptions.metrics_enabled`.
//!
//! audit-2026-04-25 R1 (Stage E re-fix): regression test added after a
//! P2-7 banner refactor accidentally gated /metrics off on loopback
//! binds, breaking pre-P2-7 always-on contract. The mnem http binary
//! now defaults `metrics_enabled = true`; this suite locks that in.

use axum::body::Body;
use axum::http::{Request, StatusCode};
use http_body_util::BodyExt as _;
use tempfile::TempDir;
use tower::ServiceExt;

fn make_app(metrics_enabled: bool) -> (axum::Router, TempDir) {
    let td = TempDir::new().expect("tmp dir");
    let opts = mnem_http::AppOptions {
        allow_labels: Some(true),
        in_memory: false,
        metrics_enabled,
        push_token: None,
    };
    let app = mnem_http::app_with_options(td.path(), opts).expect("build app");
    (app, td)
}

#[tokio::test]
async fn metrics_endpoint_mounted_when_enabled() {
    let (app, _td) = make_app(true);
    let resp = app
        .oneshot(
            Request::builder()
                .uri("/metrics")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    // Prometheus text-exposition content type. Be lenient on suffix
    // (charset / version) -- match by prefix only.
    let ct = resp
        .headers()
        .get(axum::http::header::CONTENT_TYPE)
        .map(|v| v.to_str().unwrap_or("").to_string())
        .unwrap_or_default();
    assert!(
        ct.starts_with("text/plain"),
        "unexpected content-type: {ct:?}"
    );
    let bytes = resp.into_body().collect().await.unwrap().to_bytes();
    let body = String::from_utf8_lossy(&bytes);
    assert!(
        body.contains("mnem_http_requests_total"),
        "/metrics body missing the canonical counter: {body}"
    );
}

#[tokio::test]
async fn metrics_endpoint_omitted_when_disabled() {
    let (app, _td) = make_app(false);
    let resp = app
        .oneshot(
            Request::builder()
                .uri("/metrics")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn route_table_advertises_metrics_only_when_enabled() {
    // Banner / route_table parity: the /metrics row appears iff the
    // route is mounted. Without this, the docs and the router drift.
    let with = mnem_http::route_table(true);
    assert!(with.iter().any(|(_, p, _)| *p == "/metrics"));
    let without = mnem_http::route_table(false);
    assert!(!without.iter().any(|(_, p, _)| *p == "/metrics"));
}
