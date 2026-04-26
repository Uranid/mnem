//! C3 FIX-2 live-wire test: POST /v1/retrieve with `graph_mode="ppr"`
//! must reach the server-side HybridAdjacency path. The underlying
//! ppr_over_hybrid_with_empty_knn test in mnem-core proves byte-
//! identical output vs decay when no KNN is wired, so here we only
//! smoke the wire itself.

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
async fn graph_mode_ppr_reaches_wire() {
    let (app, _td) = make_app();
    for s in ["x", "y", "z"] {
        post_node(&app, s).await;
    }

    let body = json!({
        "label": "Memory",
        "limit": 5,
        "graph_expand": 2,
        "graph_mode": "ppr",
        "ppr_damping": 0.85,
        "ppr_iter": 5,
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
    assert_eq!(resp.status(), StatusCode::OK, "graph_mode=ppr live");
    let j = to_json(resp.into_body()).await;
    assert_eq!(j["schema"], "mnem.v1.retrieve");
    assert!(j["items"].is_array());
}
