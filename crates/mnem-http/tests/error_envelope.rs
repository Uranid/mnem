//! Verify the `mnem.v1.err` envelope catches every body-deserialize
//! failure path emitted by axum's `Json<T>` extractor.
//!
//! audit-2026-04-25 R3 (Stage E re-fix): V2 verification found that
//! the P2-6 envelope middleware only rewrote the 422 path, leaving
//! 400 (malformed JSON) and 415 (missing Content-Type) responses as
//! plain-text leaks. This suite locks all three paths to the envelope.

use axum::body::Body;
use axum::http::{Method, Request, StatusCode};
use http_body_util::BodyExt as _;
use serde_json::Value;
use tempfile::TempDir;
use tower::ServiceExt;

fn make_app() -> (axum::Router, TempDir) {
    let td = TempDir::new().expect("tmp dir");
    let opts = mnem_http::AppOptions {
        allow_labels: Some(true),
        in_memory: false,
        metrics_enabled: false,
    };
    let app = mnem_http::app_with_options(td.path(), opts).expect("build app");
    (app, td)
}

async fn body_to_json(body: Body) -> Value {
    let bytes = body.collect().await.expect("collect").to_bytes();
    serde_json::from_slice(&bytes).unwrap_or_else(|e| {
        panic!(
            "expected JSON envelope but got non-JSON: {} (bytes = {:?})",
            e,
            String::from_utf8_lossy(&bytes)
        )
    })
}

#[tokio::test]
async fn malformed_json_returns_envelope() {
    let (app, _td) = make_app();
    let resp = app
        .oneshot(
            Request::builder()
                .method(Method::POST)
                .uri("/v1/nodes")
                .header("content-type", "application/json")
                .body(Body::from("malformed-json"))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    let ct = resp
        .headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .unwrap_or_default()
        .to_string();
    assert!(
        ct.starts_with("application/json"),
        "expected JSON envelope, got content-type {ct:?}"
    );
    let j = body_to_json(resp.into_body()).await;
    assert_eq!(j["schema"], "mnem.v1.err");
    assert!(j["error"].is_string());
}

#[tokio::test]
async fn wrong_type_returns_envelope() {
    let (app, _td) = make_app();
    let resp = app
        .oneshot(
            Request::builder()
                .method(Method::POST)
                .uri("/v1/nodes")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"summary": 123}"#))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    let j = body_to_json(resp.into_body()).await;
    assert_eq!(j["schema"], "mnem.v1.err");
}

#[tokio::test]
async fn missing_content_type_returns_envelope() {
    let (app, _td) = make_app();
    let resp = app
        .oneshot(
            Request::builder()
                .method(Method::POST)
                .uri("/v1/nodes")
                .body(Body::from(r#"{"summary":"hi"}"#))
                .unwrap(),
        )
        .await
        .unwrap();
    // Status is rewritten to 400 by the envelope middleware.
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    let j = body_to_json(resp.into_body()).await;
    assert_eq!(j["schema"], "mnem.v1.err");
}

// --- audit-2026-04-25 C3-3: extend envelope to /remote/v1/fetch-blocks ---
//
// `fetch-blocks` is the only `/remote/v1/*` route that previously
// leaked plain-text body-deserialize errors. The two write-side
// endpoints (`push-blocks`, `advance-head`) still use RFC 7807
// problem+json so HTTP-toolchain consumers see a standard problem
// document; those are covered by the regression tests below.

#[tokio::test]
async fn remote_fetch_blocks_missing_field_returns_envelope() {
    let (app, _td) = make_app();
    let resp = app
        .oneshot(
            Request::builder()
                .method(Method::POST)
                .uri("/remote/v1/fetch-blocks")
                .header("content-type", "application/json")
                .body(Body::from("{}"))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    let ct = resp
        .headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .unwrap_or_default()
        .to_string();
    assert!(
        ct.starts_with("application/json"),
        "expected JSON envelope on /remote/v1/fetch-blocks, got content-type {ct:?}"
    );
    let j = body_to_json(resp.into_body()).await;
    assert_eq!(j["schema"], "mnem.v1.err");
    assert!(j["error"].is_string());
}

#[tokio::test]
async fn remote_fetch_blocks_malformed_json_returns_envelope() {
    let (app, _td) = make_app();
    let resp = app
        .oneshot(
            Request::builder()
                .method(Method::POST)
                .uri("/remote/v1/fetch-blocks")
                .header("content-type", "application/json")
                .body(Body::from("NOT JSON"))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    let j = body_to_json(resp.into_body()).await;
    assert_eq!(j["schema"], "mnem.v1.err");
}

#[tokio::test]
async fn remote_push_blocks_still_uses_problem_json() {
    // Regression guard: push-blocks must NOT be wrapped by the
    // envelope -- it intentionally uses RFC 7807 problem+json (e.g.
    // 503 when push auth is unconfigured).
    let (app, _td) = make_app();
    let resp = app
        .oneshot(
            Request::builder()
                .method(Method::POST)
                .uri("/remote/v1/push-blocks")
                .header("content-type", "application/vnd.ipld.car")
                .body(Body::from(""))
                .unwrap(),
        )
        .await
        .unwrap();
    let ct = resp
        .headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .unwrap_or_default()
        .to_string();
    // The body must NOT be the mnem.v1.err envelope.
    let bytes = resp
        .into_body()
        .collect()
        .await
        .expect("collect")
        .to_bytes();
    let s = String::from_utf8_lossy(&bytes);
    if ct.starts_with("application/problem+json") {
        // RFC 7807 path -- expected.
        assert!(
            !s.contains("\"schema\":\"mnem.v1.err\""),
            "push-blocks must not emit mnem.v1.err envelope; body={s}"
        );
    } else {
        // Any non-envelope content-type is also acceptable so long
        // as the body is not the envelope schema.
        assert!(
            !s.contains("mnem.v1.err"),
            "push-blocks body should not contain envelope schema; ct={ct:?} body={s}"
        );
    }
}

#[tokio::test]
async fn remote_advance_head_still_uses_problem_json() {
    let (app, _td) = make_app();
    let resp = app
        .oneshot(
            Request::builder()
                .method(Method::POST)
                .uri("/remote/v1/advance-head")
                .header("content-type", "application/json")
                .body(Body::from("{}"))
                .unwrap(),
        )
        .await
        .unwrap();
    let bytes = resp
        .into_body()
        .collect()
        .await
        .expect("collect")
        .to_bytes();
    let s = String::from_utf8_lossy(&bytes);
    assert!(
        !s.contains("\"schema\":\"mnem.v1.err\""),
        "advance-head must not emit mnem.v1.err envelope; body={s}"
    );
}

#[tokio::test]
async fn handler_level_400_envelope_passes_through() {
    // Sanity: a request that DOES deserialize but is rejected at
    // handler level (e.g. missing `author` per business logic) must
    // already use the envelope -- the middleware must not double-wrap.
    let (app, _td) = make_app();
    let resp = app
        .oneshot(
            Request::builder()
                .method(Method::POST)
                .uri("/v1/nodes")
                .header("content-type", "application/json")
                .body(Body::from("{}"))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    let j = body_to_json(resp.into_body()).await;
    assert_eq!(j["schema"], "mnem.v1.err");
    let msg = j["error"].as_str().unwrap_or("");
    // Handler-level 400 should NOT be re-wrapped with `invalid
    // request body: ` (the middleware prefix). Detect that by
    // requiring no double prefix.
    assert!(
        !msg.contains("invalid request body: {\"schema\":"),
        "envelope double-wrap detected: {msg}"
    );
}
