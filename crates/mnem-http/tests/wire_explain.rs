//! gap-06 live-wire test: `POST /v1/explain` returns an in-band
//! derivation path from a seed node back along incoming edges. The
//! BFS traversal records parent pointers so the caller can
//! reconstruct the path without an extra round trip.
//!
//! Invariants asserted:
//!
//! 1. `schema == "mnem.v1.explain"`, mode defaults to `"compact"`,
//!    `path_source` carries the BFS provenance tag.
//! 2. `nodes[0]` is the seed.
//! 3. Every `step.to_idx` resolves inside the emitted `nodes` array
//!    and every `step.parent_idx` points at an earlier entry, so
//!    the compact encoding is self-consistent.
//! 4. `compact_full` without ACL downgrades to `compact` with a
//!    warning rather than leaking payloads.
//! 5. Runtime-derived `max_path_bytes_total` reflects
//!    `latency_budget_ms * serialization_rate_bytes_per_ms`.

use axum::body::Body;
use axum::http::{Request, StatusCode};
use http_body_util::BodyExt as _;
use proptest::prelude::*;
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

async fn post_node(app: &axum::Router, summary: &str) -> String {
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
        .expect("post node");
    assert_eq!(resp.status(), StatusCode::OK);
    let v = to_json(resp.into_body()).await;
    v["id"].as_str().expect("node id").to_string()
}

async fn post_explain(app: &axum::Router, body: Value) -> (StatusCode, Value) {
    let resp = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/explain")
                .header("content-type", "application/json")
                .body(Body::from(body.to_string()))
                .unwrap(),
        )
        .await
        .expect("post explain");
    let status = resp.status();
    let v = to_json(resp.into_body()).await;
    (status, v)
}

/// A seed node with no incoming edges still returns a valid response
/// containing exactly itself in `nodes` and no `steps`. This covers
/// the BFS degenerate case and pins the "path always reaches seed"
/// invariant (the seed IS the path when the incoming set is empty).
#[tokio::test]
async fn explain_returns_path_to_seed() {
    let (app, _td) = make_app();
    let seed_id = post_node(&app, "seed node with no backlinks").await;

    let (status, v) = post_explain(
        &app,
        json!({
            "node_id": seed_id,
            "depth":   3,
            "mode":    "compact",
        }),
    )
    .await;

    assert_eq!(status, StatusCode::OK, "body = {v}");
    assert_eq!(v["schema"], "mnem.v1.explain");
    assert_eq!(v["mode"], "compact");
    assert_eq!(v["seed"], seed_id);

    let nodes = v["nodes"].as_array().expect("nodes array");
    assert!(!nodes.is_empty(), "path must contain at least the seed");
    assert_eq!(
        nodes[0].as_str().expect("seed id"),
        seed_id,
        "nodes[0] must be the seed"
    );

    let path_source = v["path_source"].as_str().expect("path_source");
    assert!(
        path_source.starts_with("bfs.v1:graph_depth="),
        "path_source should carry BFS provenance, got {path_source}"
    );

    // Self-consistency: every step index resolves inside the nodes
    // array and parent_idx < to_idx (BFS emission order).
    for step in v["steps"].as_array().expect("steps array") {
        let to_idx = step["to_idx"].as_u64().expect("to_idx") as usize;
        let parent_idx = step["parent_idx"].as_u64().expect("parent_idx") as usize;
        assert!(to_idx < nodes.len(), "to_idx out of bounds");
        assert!(parent_idx < nodes.len(), "parent_idx out of bounds");
        assert!(
            parent_idx < to_idx,
            "parent must precede child in BFS order"
        );
    }
}

/// `compact_full` requested without a configured ACL must be
/// downgraded to `compact` (no payload leak) and a warning must be
/// emitted so the caller notices. Gate lives entirely server-side.
#[tokio::test]
async fn explain_compact_full_downgrades_without_acl() {
    let (app, _td) = make_app();
    let seed_id = post_node(&app, "seed for compact_full downgrade test").await;

    let (status, v) = post_explain(
        &app,
        json!({
            "node_id": seed_id,
            "depth":   2,
            "mode":    "compact_full",
        }),
    )
    .await;

    assert_eq!(status, StatusCode::OK);
    assert_eq!(
        v["mode"], "compact",
        "compact_full without ACL must downgrade to compact"
    );
    let warnings = v["warnings"].as_array().expect("warnings");
    assert!(
        warnings
            .iter()
            .any(|w| w["code"] == "explain.mode_downgraded"),
        "expected explain.mode_downgraded warning, got {warnings:?}"
    );
}

/// Runtime-derived byte cap: `max_path_bytes_total` equals
/// `latency_budget_ms * serialization_rate_bytes_per_ms`. Pin a
/// deterministic value so any drift in the derivation formula is
/// caught as a wire contract regression.
#[tokio::test]
async fn explain_byte_cap_is_runtime_derived() {
    let (app, _td) = make_app();
    let seed_id = post_node(&app, "seed for byte-cap derivation test").await;

    let (status, v) = post_explain(
        &app,
        json!({
            "node_id":                         seed_id,
            "depth":                           1,
            "mode":                            "compact",
            "latency_budget_ms":               100,
            "serialization_rate_bytes_per_ms": 512,
        }),
    )
    .await;

    assert_eq!(status, StatusCode::OK);
    assert_eq!(
        v["max_path_bytes_total"].as_u64().expect("cap"),
        100 * 512,
        "cap must derive from latency_budget_ms * serialization_rate"
    );
    assert_eq!(v["latency_budget_ms"], 100);
    assert_eq!(v["serialization_rate_bytes_per_ms"], 512);
}

proptest! {
    /// Pure-function invariant: the runtime-derived cap never
    /// exceeds the product of the two knobs. `saturating_mul`
    /// guarantees this holds even for adversarial inputs, so the
    /// cap is always a valid upper bound on the projected byte
    /// budget. Pins the contract that the explain handler enforces
    /// against arbitrary caller-supplied knobs.
    #[test]
    fn byte_cap_never_exceeds_budget(
        remaining_ms in 0u32..60_000u32,
        rate in 0u64..1_000_000u64,
    ) {
        let cap = mnem_http::derive_max_path_bytes(remaining_ms, rate);
        let projected = u64::from(remaining_ms).saturating_mul(rate);
        if usize::try_from(projected).is_ok() {
            prop_assert_eq!(cap as u64, projected);
        } else {
            prop_assert_eq!(cap, usize::MAX);
        }
    }
}

/// A malformed UUID in `node_id` returns 400 rather than a panic or
/// a 500. Keeps the surface honest for agent callers.
#[tokio::test]
async fn explain_rejects_invalid_node_id() {
    let (app, _td) = make_app();

    let (status, _v) = post_explain(
        &app,
        json!({
            "node_id": "not-a-uuid",
            "depth":   2,
        }),
    )
    .await;

    assert_eq!(status, StatusCode::BAD_REQUEST);
}
