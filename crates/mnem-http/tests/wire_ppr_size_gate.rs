//! Gap 02 #17 live-wire test: POST /v1/retrieve with `graph_mode=ppr`
//! and the `ppr_opt_in` knob must be accepted by the handler, and
//! the `/metrics` endpoint must expose the new families registered
//! by `Metrics::new`.
//!
//! Building a >250000-node repo inside a unit test is too heavy for
//! CI (tens of GB of memory + minutes of commit time). The numeric
//! threshold and the gate decision are instead exhaustively covered
//! by the pure-function tests in
//! `crates/mnem-core/tests/ppr_size_gate.rs`. The wire test here
//! smokes the HTTP surface: request deserialisation of `ppr_opt_in`
//! and the presence of the two new Prometheus families.

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
        metrics_enabled: true,
        push_token: None,
    };
    let app = mnem_http::app_with_options(td.path(), opts).expect("build app");
    (app, td)
}

async fn to_json(body: Body) -> Value {
    let bytes = body.collect().await.expect("collect body").to_bytes();
    serde_json::from_slice(&bytes).expect("valid JSON")
}

async fn to_text(body: Body) -> String {
    let bytes = body.collect().await.expect("collect body").to_bytes();
    String::from_utf8(bytes.to_vec()).expect("utf-8")
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
async fn ppr_opt_in_field_accepted_by_handler() {
    // Small graph => gate must not trip regardless of opt_in. We are
    // checking that the handler deserialises the knob and runs to
    // completion; the byte-level semantics come from the mnem-core
    // unit tests.
    let (app, _td) = make_app();
    for s in ["a", "b", "c"] {
        post_node(&app, s).await;
    }

    let body = json!({
        "label": "Memory",
        "limit": 5,
        "graph_expand": 2,
        "graph_mode": "ppr",
        "ppr_opt_in": true,
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
    assert_eq!(resp.status(), StatusCode::OK, "ppr_opt_in=true accepted");
    let j = to_json(resp.into_body()).await;
    assert_eq!(j["schema"], "mnem.v1.retrieve");
    // Small graph => gate does NOT trip => no PprSizeGateSkipped
    // warning in the response regardless of opt_in.
    if let Some(ws) = j.get("warnings").and_then(Value::as_array) {
        for w in ws {
            let code = w.get("code").and_then(Value::as_str).unwrap_or("");
            assert_ne!(
                code, "ppr_size_gate_skipped",
                "small graph must not emit PprSizeGateSkipped"
            );
        }
    }
}

#[tokio::test]
async fn ppr_size_gate_metrics_families_registered() {
    // Scrape /metrics and verify both new Gap 02 #17 metric families
    // are present and the threshold gauge reports PPR_DEFAULT_MAX_NODES.
    let (app, _td) = make_app();
    let resp = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/metrics")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let text = to_text(resp.into_body()).await;
    assert!(
        text.contains("mnem_ppr_size_gate_skipped_total"),
        "counter family missing from /metrics:\n{text}"
    );
    assert!(
        text.contains("mnem_ppr_size_gate_threshold"),
        "threshold gauge missing from /metrics:\n{text}"
    );
    // The gauge should report the compile-time constant.
    assert!(
        text.contains("mnem_ppr_size_gate_threshold 250000"),
        "threshold gauge must mirror PPR_DEFAULT_MAX_NODES=250000:\n{text}"
    );
}
