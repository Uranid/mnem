//! Regression test for audit-2026-04-25 P1-1:
//! `POST /v1/traverse_answer` was defined as a handler but never
//! registered in the router, so real traffic hit 404 instead of the
//! 410-Gone opt-in response. After the fix, a fresh default build must
//! return NOT 404 (exact status is 410 until the experimental flag
//! flips, per architect Decision 4).

use axum::body::Body;
use axum::http::{Request, StatusCode};
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

#[tokio::test]
async fn traverse_answer_route_is_registered_default_gated() {
    let (app, _td) = make_app();
    let body = serde_json::json!({ "text": "x" });
    let resp = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/traverse_answer")
                .header("content-type", "application/json")
                .body(Body::from(body.to_string()))
                .unwrap(),
        )
        .await
        .expect("send request");
    // The route MUST be registered: no 404.
    assert_ne!(
        resp.status(),
        StatusCode::NOT_FOUND,
        "traverse_answer should be wired into the router"
    );
    // Default config: experimental.single_call_multihop = false, so the
    // handler returns 410 Gone + opt-in pointer. That proves the route
    // resolved to the handler rather than 404'ing at the router.
    assert_eq!(resp.status(), StatusCode::GONE);
}
