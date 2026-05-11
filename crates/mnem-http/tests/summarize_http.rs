//! Integration tests for the E4 T2 optional summarize hook on
//! `POST /v1/retrieve`.
//!
//! Asserts the flag-off default contract: omitting `summarize` from
//! the request body produces a response with no `summary` field at
//! all. Proves the pathway is zero-impact when not requested.
//!
//! Also asserts that `summarize: true` + `summarize_k: 3` yields a
//! `summary` field in the response (possibly empty when no embedder
//! is configured on the test server, with a `summarize_skipped`
//! reason string). This preserves the "never 500, always explain"
//! posture of mnem http.

use axum::body::Body;
use axum::http::{Request, StatusCode};
use http_body_util::BodyExt as _;
use serde_json::{Value, json};
use tempfile::TempDir;
use tower::ServiceExt;

fn make_app() -> (axum::Router, TempDir) {
    let td = TempDir::new().expect("tmp dir");
    let opts = mnem_http::AppOptions {
        allow_labels: Some(true),
        in_memory: false,
        metrics_enabled: false,
        push_token: None,
    };
    let app = mnem_http::app_with_options(td.path(), opts).expect("build app");
    (app, td)
}

async fn to_json(body: Body) -> Value {
    let bytes = body.collect().await.expect("collect body").to_bytes();
    serde_json::from_slice(&bytes).expect("valid JSON")
}

async fn post_node(app: &axum::Router, summary: &str) {
    let body = json!({
        "label": "Memory",
        "summary": summary,
        "author": "tests",
    });
    let resp = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/nodes")
                .header("content-type", "application/json")
                .body(Body::from(body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
}

#[tokio::test]
async fn retrieve_without_summarize_has_no_summary_field() {
    let (app, _td) = make_app();
    post_node(&app, "Alice lives in Berlin").await;

    // POST /v1/retrieve with NO `summarize` key in the body.
    let body = json!({ "label": "Memory", "limit": 5 });
    let resp = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/retrieve")
                .header("content-type", "application/json")
                .body(Body::from(body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK, "retrieve without summarize");
    let j = to_json(resp.into_body()).await;
    assert_eq!(j["schema"], "mnem.v1.retrieve");
    // Core invariant: `summary` MUST be absent when summarize is off.
    assert!(
        j.get("summary").is_none(),
        "summary field leaked when summarize was not requested: {j}"
    );
    assert!(
        j.get("summarize_skipped").is_none(),
        "summarize_skipped field leaked when summarize was off: {j}"
    );
}

#[tokio::test]
async fn retrieve_with_summarize_includes_summary_field() {
    let (app, _td) = make_app();
    for s in [
        "Alice lives in Berlin and climbs on weekends.",
        "Bob runs a coffee shop in Lisbon.",
        "Photosynthesis converts sunlight into energy.",
    ] {
        post_node(&app, s).await;
    }

    // POST /v1/retrieve with summarize=true, summarize_k=3.
    let body = json!({
        "label": "Memory",
        "limit": 5,
        "summarize": true,
        "summarize_k": 3,
    });
    let resp = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/retrieve")
                .header("content-type", "application/json")
                .body(Body::from(body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK, "retrieve with summarize");
    let j = to_json(resp.into_body()).await;
    assert_eq!(j["schema"], "mnem.v1.retrieve");
    // `summary` is present. When the test server has no embedder
    // configured, it is an empty array + `summarize_skipped` reason.
    // When an embedder is configured, it is a non-empty array. Either
    // way, the KEY must exist.
    assert!(
        j.get("summary").is_some(),
        "summary field missing when summarize was requested: {j}"
    );
    let summary = &j["summary"];
    assert!(summary.is_array(), "summary is not a JSON array: {summary}");
}
