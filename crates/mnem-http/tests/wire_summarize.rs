//! C3 FIX-4 wire proof: `summarize: true` yields a `summary` key;
//! `summarize: false` (or absent) omits it. The existing
//! `summarize_http.rs` test covers the OFF path for the pre-wire
//! contract; this test pins the ON-path behaviour post-C3.

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
    let body = json!({ "label": "Memory", "summary": summary, "author": "tests" });
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
async fn summarize_true_emits_summary_field_summarize_false_omits_it() {
    let (app, _td) = make_app();
    post_node(&app, "one two three").await;

    // summarize=true: summary key MUST be present.
    let body = json!({ "label": "Memory", "limit": 5, "summarize": true, "summarize_k": 1 });
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
    assert_eq!(resp.status(), StatusCode::OK);
    let j = to_json(resp.into_body()).await;
    assert!(
        j.as_object().unwrap().contains_key("summary"),
        "summarize=true must emit a summary field"
    );

    // summarize=false: summary key MUST NOT be present.
    let body = json!({ "label": "Memory", "limit": 5, "summarize": false });
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
    assert_eq!(resp.status(), StatusCode::OK);
    let j = to_json(resp.into_body()).await;
    assert!(
        !j.as_object().unwrap().contains_key("summary"),
        "summarize=false must omit the summary field"
    );
}
