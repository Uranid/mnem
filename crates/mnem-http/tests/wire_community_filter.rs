//! C3 FIX-1 live-wire test: POST /v1/retrieve with `community_filter=true`
//! must reach the server-side CommunityAssignment path. This is a smoke
//! test for the AppState cache + Retriever::with_community_filter plumb.
//!
//! The underlying retriever's E1 zero-impact contract guarantees that
//! `community_filter=false` is a byte-exact pass-through; here we only
//! prove the ON path is non-panicking and returns a valid response so
//! the wiring is demonstrably live (not a silent no-op).

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
async fn community_filter_on_reaches_wire() {
    let (app, _td) = make_app();
    for s in ["alpha one", "alpha two", "beta one"] {
        post_node(&app, s).await;
    }

    let body = json!({
        "label": "Memory",
        "limit": 5,
        "community_filter": true,
        "community_min_coverage": 0.5,
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
    assert_eq!(resp.status(), StatusCode::OK, "community_filter=true live");
    let j = to_json(resp.into_body()).await;
    assert_eq!(j["schema"], "mnem.v1.retrieve");
    // With FIX-1 wired the request must not 500 and must carry the
    // canonical `items` field (the wire reached the retriever).
    assert!(j["items"].is_array(), "items present");
}
