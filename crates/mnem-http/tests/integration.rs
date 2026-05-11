//! Integration tests for `mnem http`.
//!
//! Uses `tower::ServiceExt::oneshot` to drive the router without
//! binding a TCP port. Every test runs against a fresh temp-dir repo.

use axum::body::Body;
use axum::http::{Request, StatusCode};
use http_body_util::BodyExt as _;
use serde_json::Value;
use tempfile::TempDir;
use tower::ServiceExt;

fn make_app() -> (axum::Router, TempDir) {
    // These tests assert on the caller-supplied `label` being
    // round-tripped (`"Memory"` rather than `Node::DEFAULT_NTYPE`). The
    // production default is to hide labels entirely, gated by
    // `MNEM_BENCH=1`. Tests opt in via the programmatic override so
    // they exercise the benchmark-path schema without touching the
    // process environment (unsafe under Rust 2024).
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

#[tokio::test]
async fn healthz_returns_ok() {
    let (app, _td) = make_app();
    let resp = app
        .oneshot(
            Request::builder()
                .uri("/v1/healthz")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let j = to_json(resp.into_body()).await;
    assert_eq!(j["ok"], true);
    assert_eq!(j["schema"], "mnem.v1.healthz");
}

#[tokio::test]
async fn stats_returns_op_id_and_schema() {
    let (app, _td) = make_app();
    let resp = app
        .oneshot(
            Request::builder()
                .uri("/v1/stats")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let j = to_json(resp.into_body()).await;
    assert_eq!(j["schema"], "mnem.v1.stats");
    assert!(j["op_id"].as_str().unwrap().starts_with("bafyrei"));
}

#[tokio::test]
async fn post_node_then_get_then_retrieve() {
    let (app, _td) = make_app();

    // POST /v1/nodes
    let body = serde_json::json!({
        "label": "Memory",
        "summary": "Alice lives in Berlin",
        "author": "tests"
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
    assert_eq!(resp.status(), StatusCode::OK, "post_node");
    let j = to_json(resp.into_body()).await;
    let id = j["id"].as_str().unwrap().to_string();
    assert_eq!(j["label"], "Memory");

    // GET /v1/nodes/{id}
    let resp = app
        .clone()
        .oneshot(
            Request::builder()
                .uri(format!("/v1/nodes/{id}"))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK, "get_node");
    let j = to_json(resp.into_body()).await;
    assert_eq!(j["summary"], "Alice lives in Berlin");
    assert_eq!(j["label"], "Memory");

    // GET /v1/retrieve. No embedder is configured in this test repo,
    // so the only usable filter is label-based (the legacy BM25 text
    // lane was removed in ; text queries without an embedder
    // + sparse provider now return empty, not a stopword-matched
    // fallback). Label-filter retrieval still exercises the full
    // render + token-budget path.
    let resp = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/v1/retrieve?label=Memory&budget=200")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK, "retrieve");
    let j = to_json(resp.into_body()).await;
    assert_eq!(j["schema"], "mnem.v1.retrieve");
    assert!(j["items"].as_array().unwrap().len() >= 1);
    assert_eq!(j["items"][0]["summary"], "Alice lives in Berlin");
    assert_eq!(j["tokens_budget"], 200);
}

#[tokio::test]
async fn delete_node_round_trip() {
    // DELETE /v1/nodes/{id} requires a bearer token (BUG-16 fix).
    // Build a dedicated app instance with push_token configured so the
    // RequireBearer extractor has a token to validate against.
    let td = TempDir::new().expect("tmp dir");
    let opts = mnem_http::AppOptions {
        allow_labels: Some(true),
        in_memory: false,
        metrics_enabled: false,
        push_token: Some("test-token".into()),
    };
    let app = mnem_http::app_with_options(td.path(), opts).expect("build app");

    // Create
    let body = serde_json::json!({
        "label": "Memory",
        "summary": "scratch",
        "author": "tests"
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
    let id = to_json(resp.into_body()).await["id"]
        .as_str()
        .unwrap()
        .to_string();

    // Delete -- must supply bearer token; missing/wrong token yields 401.
    let resp = app
        .clone()
        .oneshot(
            Request::builder()
                .method("DELETE")
                .uri(format!("/v1/nodes/{id}?author=tests"))
                .header("authorization", "Bearer test-token")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let j = to_json(resp.into_body()).await;
    assert_eq!(j["existed"], true);

    // Get after delete is 404
    let resp = app
        .oneshot(
            Request::builder()
                .uri(format!("/v1/nodes/{id}"))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn bad_uuid_is_400_not_500() {
    let (app, _td) = make_app();
    let resp = app
        .oneshot(
            Request::builder()
                .uri("/v1/nodes/not-a-uuid")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    let j = to_json(resp.into_body()).await;
    assert_eq!(j["schema"], "mnem.v1.err");
    assert!(j["error"].as_str().unwrap().contains("invalid UUID"));
}

#[tokio::test]
async fn empty_label_falls_back_to_default_ntype() {
    // Previously this asserted a 400 on empty label, but the
    // `MNEM_BENCH`-gated rework silently substitutes
    // `Node::DEFAULT_NTYPE` whenever the caller-supplied label is empty
    // (same as the unset case). Post still succeeds; the stored label
    // is "Node". This test pins that behaviour so a future change to
    // either the gate or the fallback is a deliberate decision.
    let (app, _td) = make_app();
    let body = serde_json::json!({
        "label": "",
        "summary": "x",
        "author": "tests"
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
    assert_eq!(resp.status(), StatusCode::OK);
    let j = to_json(resp.into_body()).await;
    // Fetch the stored node and confirm the server substituted the
    // default ntype. The POST response echoes the caller's empty input
    // rather than the server-side label, so we have to round-trip.
    let id = j["id"].as_str().unwrap().to_string();
    let resp = app
        .oneshot(
            Request::builder()
                .uri(format!("/v1/nodes/{id}"))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let j = to_json(resp.into_body()).await;
    assert_eq!(j["label"], "Node");
}

#[tokio::test]
async fn tombstone_node_round_trip_returns_schema_and_op_id() {
    // Happy path: POST /v1/nodes creates a node, POST
    // /v1/nodes/{id}/tombstone returns 200 with the expected schema
    // envelope. A subsequent GET still resolves the node (logical
    // tombstone, not physical delete); the node just drops out of
    // retrieve (covered in mnem-core integration tests).
    let (app, _td) = make_app();
    let body = serde_json::json!({
        "label": "Memory",
        "summary": "Alice likes jazz",
        "author": "tests"
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
    assert_eq!(resp.status(), StatusCode::OK);
    let id = to_json(resp.into_body()).await["id"]
        .as_str()
        .unwrap()
        .to_string();

    let body = serde_json::json!({
        "reason": "user asked to forget",
        "author": "tests"
    });
    let resp = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri(format!("/v1/nodes/{id}/tombstone"))
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_vec(&body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK, "tombstone call ok");
    let j = to_json(resp.into_body()).await;
    assert_eq!(j["schema"], "mnem.v1.tombstone");
    assert_eq!(j["node_id"], id);
    assert!(j["op_id"].as_str().is_some());

    // Node itself still resolves: tombstone is logical.
    let resp = app
        .oneshot(
            Request::builder()
                .uri(format!("/v1/nodes/{id}"))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
}

#[tokio::test]
async fn tombstone_returns_404_for_missing_and_409_for_already_tombstoned() {
    // Negative paths for POST /v1/nodes/{id}/tombstone: 404 if the
    // node never existed, 409 if it is already tombstoned.
    let (app, _td) = make_app();

    // 404: random well-formed UUID that never existed.
    let fake_id = "00000000-0000-0000-0000-000000000001";
    let resp = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri(format!("/v1/nodes/{fake_id}/tombstone"))
                .header("content-type", "application/json")
                .body(Body::from(
                    serde_json::to_vec(&serde_json::json!({
                        "reason": "r",
                        "author": "tests"
                    }))
                    .unwrap(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(
        resp.status(),
        StatusCode::NOT_FOUND,
        "missing node must be 404"
    );

    // Create, tombstone, then re-tombstone -> 409.
    let body = serde_json::json!({
        "label": "Memory",
        "summary": "ephemeral",
        "author": "tests"
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
    let id = to_json(resp.into_body()).await["id"]
        .as_str()
        .unwrap()
        .to_string();
    let ts_body = serde_json::json!({ "reason": "first", "author": "tests" });
    let resp = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri(format!("/v1/nodes/{id}/tombstone"))
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_vec(&ts_body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK, "first tombstone ok");

    // Second call: 409.
    let resp = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri(format!("/v1/nodes/{id}/tombstone"))
                .header("content-type", "application/json")
                .body(Body::from(
                    serde_json::to_vec(&serde_json::json!({
                        "reason": "second",
                        "author": "tests"
                    }))
                    .unwrap(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(
        resp.status(),
        StatusCode::CONFLICT,
        "double tombstone must be 409"
    );
}

#[tokio::test]
async fn retrieve_with_no_filters_errors() {
    let (app, _td) = make_app();
    let resp = app
        .oneshot(
            Request::builder()
                .uri("/v1/retrieve")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    // No filters + no rankers -> core returns an error; http surface
    // should bubble as 500 (or a 400 once we classify more tightly).
    assert!(
        resp.status() == StatusCode::BAD_REQUEST
            || resp.status() == StatusCode::INTERNAL_SERVER_ERROR,
        "unexpected status: {}",
        resp.status()
    );
}

// ---------- input clamps (R2-A security hardening) ----------
//
// Every numeric knob the retrieve path exposes has a boundary cap.
// The goal is not to impose product shape, just to prevent an
// accidental or adversarial `u64::MAX` from triggering a downstream
// allocator blow-up. The failure mode is a 400 with a specific
// message that names the offending knob.

#[tokio::test]
async fn retrieve_get_rejects_oversized_limit() {
    let (app, _td) = make_app();
    let resp = app
        .oneshot(
            Request::builder()
                .uri("/v1/retrieve?limit=99999999")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    let j = to_json(resp.into_body()).await;
    let err = j["error"].as_str().unwrap();
    assert!(
        err.contains("limit=99999999"),
        "error must name the rejected knob + value: {err}"
    );
    assert!(
        err.contains("max of"),
        "error must state the ceiling: {err}"
    );
}

#[tokio::test]
async fn retrieve_post_rejects_oversized_vector_cap() {
    let (app, _td) = make_app();
    let body = serde_json::json!({
        "vector_cap": 9_999_999_u64
    });
    let resp = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/retrieve")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_vec(&body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    let j = to_json(resp.into_body()).await;
    let err = j["error"].as_str().unwrap();
    assert!(
        err.contains("vector_cap"),
        "error must name the knob: {err}"
    );
}

#[tokio::test]
async fn retrieve_post_rejects_oversized_rerank_top_k() {
    let (app, _td) = make_app();
    let body = serde_json::json!({
        "rerank_top_k": 10_000_u64
    });
    let resp = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/retrieve")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_vec(&body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    let j = to_json(resp.into_body()).await;
    let err = j["error"].as_str().unwrap();
    assert!(
        err.contains("rerank_top_k"),
        "error must name the knob: {err}"
    );
}

#[tokio::test]
async fn retrieve_post_accepts_at_limit() {
    // Exactly at the cap is allowed; only strictly-greater is rejected.
    // Body sends no retrieval signal so the core rejects with 400 or
    // 500, NOT because of our clamp - that's the property this test
    // defends.
    let (app, _td) = make_app();
    let body = serde_json::json!({
        "limit": 1000,
        "vector_cap": 100_000,
        "rerank_top_k": 500
    });
    let resp = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/retrieve")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_vec(&body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();
    let status = resp.status();
    if status == StatusCode::BAD_REQUEST {
        // Must NOT be the clamp error - check the message doesn't
        // name any of our three knobs.
        let j = to_json(resp.into_body()).await;
        let err = j["error"].as_str().unwrap_or("");
        assert!(
            !(err.contains("limit=1000 exceeds")
                || err.contains("vector_cap=100000 exceeds")
                || err.contains("rerank_top_k=500 exceeds")),
            "at-cap values must not trip the clamp: {err}"
        );
    }
    // Any non-clamp outcome is fine; we only care about the clamp
    // not firing at the exact ceiling.
}

// ---------- correlation-id (R3-B) ----------

#[tokio::test]
async fn response_carries_minted_correlation_id() {
    // No header on the request -> middleware mints a UUIDv7, echoes
    // it in `X-Request-Id`. Asserts the canonical hex-with-hyphens
    // shape so downstream log-parsing tools can rely on it.
    let (app, _td) = make_app();
    let resp = app
        .oneshot(
            Request::builder()
                .uri("/v1/healthz")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let id = resp
        .headers()
        .get("x-request-id")
        .expect("x-request-id echoed on every response")
        .to_str()
        .expect("ascii header")
        .to_string();
    assert_eq!(
        id.len(),
        36,
        "minted UUIDv7 has 36 chars (32 hex + 4 hyphens), got {id}"
    );
    assert_eq!(id.matches('-').count(), 4, "UUIDv7 has 4 hyphens, got {id}");
}

#[tokio::test]
async fn response_reuses_caller_supplied_correlation_id() {
    // Caller-supplied id in the acceptable window -> echoed back
    // verbatim. Lets upstream gateways thread one id end-to-end.
    let (app, _td) = make_app();
    let caller = "req-test-correlation-0001";
    let resp = app
        .oneshot(
            Request::builder()
                .uri("/v1/healthz")
                .header("x-request-id", caller)
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    assert_eq!(
        resp.headers()
            .get("x-request-id")
            .and_then(|v| v.to_str().ok()),
        Some(caller),
        "caller-supplied correlation id must round-trip"
    );
}

// ============================================================
// POST /v1/ingest (Phase-B5d)
// ============================================================

/// Build a minimal multipart body by hand. Avoids pulling in a
/// multipart-writer crate just for two tests; the bytes below are the
/// literal RFC 7578 shape axum's `Multipart` extractor accepts.
fn multipart_body(
    boundary: &str,
    file_name: &str,
    file_bytes: &[u8],
    fields: &[(&str, &str)],
) -> Vec<u8> {
    let mut out = Vec::new();
    for (k, v) in fields {
        out.extend_from_slice(format!("--{boundary}\r\n").as_bytes());
        out.extend_from_slice(
            format!("Content-Disposition: form-data; name=\"{k}\"\r\n\r\n{v}\r\n").as_bytes(),
        );
    }
    out.extend_from_slice(format!("--{boundary}\r\n").as_bytes());
    out.extend_from_slice(
        format!(
            "Content-Disposition: form-data; name=\"file\"; filename=\"{file_name}\"\r\n\
             Content-Type: application/octet-stream\r\n\r\n"
        )
        .as_bytes(),
    );
    out.extend_from_slice(file_bytes);
    out.extend_from_slice(b"\r\n");
    out.extend_from_slice(format!("--{boundary}--\r\n").as_bytes());
    out
}

#[tokio::test]
async fn ingest_multipart_markdown_commits_subgraph() {
    let (app, _td) = make_app();
    let boundary = "----mnemTestBoundary";
    let body = multipart_body(
        boundary,
        "hello.md",
        b"# Title\n\nAlice Johnson met Bob Lee at Acme Corp on 2026-04-24.\n",
        &[
            ("author", "http-ingest-test"),
            ("message", "ingest roundtrip"),
        ],
    );
    let resp = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/ingest")
                .header(
                    "content-type",
                    format!("multipart/form-data; boundary={boundary}"),
                )
                .body(Body::from(body))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let j = to_json(resp.into_body()).await;
    assert_eq!(j["schema"], "mnem.v1.ingest");
    assert!(j["chunk_count"].as_u64().unwrap_or(0) >= 1);
    assert!(j["node_count"].as_u64().unwrap_or(0) >= 2);
    assert!(j["commit_cid"].is_string());
}

#[tokio::test]
async fn ingest_json_body_text_kind_commits_subgraph() {
    let (app, _td) = make_app();
    let body = serde_json::json!({
        "text": "Alice Johnson joined Acme Corp on 2026-04-24.",
        "kind": "text",
        "author": "json-ingest-test"
    });
    let resp = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/ingest")
                .header("content-type", "application/json")
                .body(Body::from(body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let j = to_json(resp.into_body()).await;
    assert_eq!(j["schema"], "mnem.v1.ingest");
    assert!(j["chunk_count"].as_u64().unwrap_or(0) >= 1);
}

#[tokio::test]
async fn ingest_json_body_missing_author_is_bad_request() {
    let (app, _td) = make_app();
    let body = serde_json::json!({
        "text": "no author on this one",
        "kind": "text"
    });
    let resp = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/ingest")
                .header("content-type", "application/json")
                .body(Body::from(body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn ingest_json_body_max_tokens_clamp_is_bad_request() {
    // `max_tokens` over 8192 must 400 with a clear message; this
    // mirrors the CLI + MCP guardrails documented in B5d.
    let (app, _td) = make_app();
    let body = serde_json::json!({
        "text": "irrelevant",
        "kind": "text",
        "author": "clamp-test",
        "max_tokens": 999_999
    });
    let resp = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/ingest")
                .header("content-type", "application/json")
                .body(Body::from(body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    let j = to_json(resp.into_body()).await;
    let err = j["error"].as_str().unwrap_or_default();
    assert!(err.contains("8192"), "expected clamp message, got {err:?}");
}

// ---------- POST /v1/edges ----------

/// Helper: create a node and return its UUID string.
async fn create_node(app: &axum::Router, summary: &str) -> String {
    let body = serde_json::json!({
        "label": "Entity",
        "summary": summary,
        "author": "tests"
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
    assert_eq!(resp.status(), StatusCode::OK, "create_node should succeed");
    to_json(resp.into_body()).await["id"]
        .as_str()
        .unwrap()
        .to_string()
}

#[tokio::test]
async fn post_edge_happy_path() {
    // Build app with push_token so RequireBearer passes.
    let td = tempfile::TempDir::new().unwrap();
    let opts = mnem_http::AppOptions {
        allow_labels: Some(true),
        in_memory: false,
        metrics_enabled: false,
        push_token: Some("tok".into()),
    };
    let app = mnem_http::app_with_options(td.path(), opts).unwrap();

    let src_id = create_node(&app, "Alice").await;
    let dst_id = create_node(&app, "Berlin").await;

    let edge_body = serde_json::json!({
        "src": src_id,
        "dst": dst_id,
        "etype": "lives_in",
        "author": "tests"
    });
    let resp = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/edges")
                .header("content-type", "application/json")
                .header("authorization", "Bearer tok")
                .body(Body::from(serde_json::to_vec(&edge_body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK, "post_edge happy path");
    let j = to_json(resp.into_body()).await;
    assert_eq!(j["schema"], "mnem.v1.post-edge");
    assert!(
        j["edge_id"].as_str().unwrap().len() > 10,
        "edge_id should be a UUID string"
    );
    assert!(
        j["op_id"].as_str().unwrap().starts_with("bafyrei"),
        "op_id should be a CID"
    );
}

#[tokio::test]
async fn post_edge_missing_src_is_404() {
    let td = tempfile::TempDir::new().unwrap();
    let opts = mnem_http::AppOptions {
        allow_labels: Some(true),
        in_memory: false,
        metrics_enabled: false,
        push_token: Some("tok".into()),
    };
    let app = mnem_http::app_with_options(td.path(), opts).unwrap();

    let dst_id = create_node(&app, "Berlin").await;
    let ghost_src = "00000000-0000-7000-8000-000000000001";

    let edge_body = serde_json::json!({
        "src": ghost_src,
        "dst": dst_id,
        "etype": "lives_in",
        "author": "tests"
    });
    let resp = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/edges")
                .header("content-type", "application/json")
                .header("authorization", "Bearer tok")
                .body(Body::from(serde_json::to_vec(&edge_body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::NOT_FOUND, "missing src => 404");
}

#[tokio::test]
async fn post_edge_missing_dst_is_404() {
    let td = tempfile::TempDir::new().unwrap();
    let opts = mnem_http::AppOptions {
        allow_labels: Some(true),
        in_memory: false,
        metrics_enabled: false,
        push_token: Some("tok".into()),
    };
    let app = mnem_http::app_with_options(td.path(), opts).unwrap();

    let src_id = create_node(&app, "Alice").await;
    let ghost_dst = "00000000-0000-7000-8000-000000000002";

    let edge_body = serde_json::json!({
        "src": src_id,
        "dst": ghost_dst,
        "etype": "lives_in",
        "author": "tests"
    });
    let resp = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/edges")
                .header("content-type", "application/json")
                .header("authorization", "Bearer tok")
                .body(Body::from(serde_json::to_vec(&edge_body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::NOT_FOUND, "missing dst => 404");
}

#[tokio::test]
async fn post_edge_bad_uuid_is_400() {
    let td = tempfile::TempDir::new().unwrap();
    let opts = mnem_http::AppOptions {
        allow_labels: Some(true),
        in_memory: false,
        metrics_enabled: false,
        push_token: Some("tok".into()),
    };
    let app = mnem_http::app_with_options(td.path(), opts).unwrap();

    let edge_body = serde_json::json!({
        "src": "not-a-uuid",
        "dst": "also-not-a-uuid",
        "etype": "relates_to",
        "author": "tests"
    });
    let resp = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/edges")
                .header("content-type", "application/json")
                .header("authorization", "Bearer tok")
                .body(Body::from(serde_json::to_vec(&edge_body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST, "bad UUID => 400");
    let j = to_json(resp.into_body()).await;
    assert_eq!(j["schema"], "mnem.v1.err");
}

#[tokio::test]
async fn post_edge_missing_author_is_400() {
    let td = tempfile::TempDir::new().unwrap();
    let opts = mnem_http::AppOptions {
        allow_labels: Some(true),
        in_memory: false,
        metrics_enabled: false,
        push_token: Some("tok".into()),
    };
    let app = mnem_http::app_with_options(td.path(), opts).unwrap();

    let src_id = create_node(&app, "Alice").await;
    let dst_id = create_node(&app, "Berlin").await;

    let edge_body = serde_json::json!({
        "src": src_id,
        "dst": dst_id,
        "etype": "lives_in"
        // author intentionally omitted
    });
    let resp = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/edges")
                .header("content-type", "application/json")
                .header("authorization", "Bearer tok")
                .body(Body::from(serde_json::to_vec(&edge_body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(
        resp.status(),
        StatusCode::BAD_REQUEST,
        "missing author => 400"
    );
}

#[tokio::test]
async fn post_edge_without_auth_is_401() {
    let td = tempfile::TempDir::new().unwrap();
    let opts = mnem_http::AppOptions {
        allow_labels: Some(true),
        in_memory: false,
        metrics_enabled: false,
        push_token: Some("tok".into()),
    };
    let app = mnem_http::app_with_options(td.path(), opts).unwrap();

    let edge_body = serde_json::json!({
        "src": "00000000-0000-7000-8000-000000000001",
        "dst": "00000000-0000-7000-8000-000000000002",
        "etype": "relates_to",
        "author": "tests"
    });
    let resp = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/edges")
                .header("content-type", "application/json")
                // No Authorization header
                .body(Body::from(serde_json::to_vec(&edge_body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(
        resp.status(),
        StatusCode::UNAUTHORIZED,
        "no bearer token => 401"
    );
}

// ---------- GET /v1/nodes/{id}/embedding ----------

/// Build an app and pre-seed it with a node that has an embedding attached
/// to its sidecar. Returns the router, the node UUID string, and the model
/// string so the tests can re-use the live embedding.
///
/// Seeds data first (before building the app) so the redb exclusive lock
/// is not held by two openers simultaneously. After seeding, the seeder
/// drops its file handle and `app_with_options` opens cleanly.
fn make_app_with_embedding() -> (axum::Router, TempDir, String, String) {
    use bytes::Bytes;
    use mnem_core::id::NodeId;
    use mnem_core::objects::{Dtype, Embedding, Node};

    let td = TempDir::new().expect("tmp dir");

    // --- Seed phase: open the redb, write a node + embedding, then close. ---
    let data_dir = td.path().join(".mnem");
    std::fs::create_dir_all(&data_dir).expect("create .mnem");
    let db_path = data_dir.join("repo.redb");

    let model = "onnx:all-MiniLM-L6-v2".to_string();
    let node_id_str;
    {
        let (bs, ohs, _db) = mnem_backend_redb::open_or_init(&db_path).expect("open redb");
        let bs_arc: std::sync::Arc<dyn mnem_core::store::Blockstore> = bs;
        let ohs_arc: std::sync::Arc<dyn mnem_core::store::OpHeadsStore> = ohs;

        let repo = mnem_core::repo::ReadonlyRepo::init(bs_arc.clone(), ohs_arc.clone())
            .expect("init repo");

        let dim: usize = 384;
        let vector: Vec<f32> = (0..dim).map(|i| i as f32 / dim as f32).collect();
        let mut vector_bytes = Vec::with_capacity(dim * 4);
        for v in &vector {
            vector_bytes.extend_from_slice(&v.to_le_bytes());
        }

        let node = Node::new(NodeId::new_v7(), "Memory").with_summary("test embedding node");
        node_id_str = node.id.to_uuid_string();

        let emb = Embedding {
            model: model.clone(),
            dtype: Dtype::F32,
            dim: dim as u32,
            vector: Bytes::from(vector_bytes),
        };

        let mut tx = repo.start_transaction();
        let node_cid = tx.add_node(&node).expect("add node");
        tx.set_embedding(node_cid, model.clone(), emb)
            .expect("set embedding");
        tx.commit("tests", "seed embedding test").expect("commit");
        // `_db` (the redb Database handle) is dropped here, releasing the lock.
    }

    // --- App phase: open the already-seeded redb. ---
    let opts = mnem_http::AppOptions {
        allow_labels: Some(true),
        in_memory: false,
        metrics_enabled: false,
        push_token: None,
    };
    let app = mnem_http::app_with_options(td.path(), opts).expect("build app");

    (app, td, node_id_str, model)
}

#[tokio::test]
async fn get_node_embedding_returns_vector() {
    let (app, _td, node_id, model) = make_app_with_embedding();

    let resp = app
        .oneshot(
            Request::builder()
                .uri(format!("/v1/nodes/{node_id}/embedding?model={model}"))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK, "embedding found");
    let j = to_json(resp.into_body()).await;
    assert_eq!(j["schema"], "mnem.v1.node_embedding");
    assert_eq!(j["node_id"], node_id.as_str());
    assert_eq!(j["model"], model.as_str());
    assert_eq!(j["dim"], 384);
    assert_eq!(j["dtype"], "f32");
    let vec_arr = j["vector"].as_array().expect("vector is array");
    assert!(!vec_arr.is_empty(), "vector must be non-empty");
    assert_eq!(vec_arr.len(), 384);
}

#[tokio::test]
async fn get_node_embedding_missing_model_is_404() {
    let (app, _td, node_id, _model) = make_app_with_embedding();

    let resp = app
        .oneshot(
            Request::builder()
                .uri(format!(
                    "/v1/nodes/{node_id}/embedding?model=nonexistent-model"
                ))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::NOT_FOUND, "missing model is 404");
    let j = to_json(resp.into_body()).await;
    assert_eq!(j["schema"], "mnem.v1.err");
    assert!(
        j["error"]
            .as_str()
            .unwrap()
            .contains("no embedding for model=nonexistent-model"),
        "error message mentions model"
    );
}

#[tokio::test]
async fn get_node_embedding_missing_node_is_404() {
    let (app, _td) = make_app();
    let fake_id = "00000000-0000-7000-8000-000000000001";
    let resp = app
        .oneshot(
            Request::builder()
                .uri(format!("/v1/nodes/{fake_id}/embedding?model=some-model"))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::NOT_FOUND, "missing node is 404");
    let j = to_json(resp.into_body()).await;
    assert_eq!(j["schema"], "mnem.v1.err");
}

/// Regression test for "mnem-http silent-drop fix (long-summary nodes get
/// dropped on ingest)". Investigation of `handlers.rs` and
/// `handlers_ingest.rs` shows no truncation or drop path in the
/// `POST /v1/nodes` handler for large `summary` fields: the summary is
/// written verbatim via `node.with_summary(sum)` with no size check.
///
/// This test pins the correct behaviour: a node with a 10,000-character
/// summary must be stored and retrieved with the full, untruncated text.
/// If a future change introduces a size limit or truncation, this test
/// will break with a clear `summary` mismatch rather than a silent data
/// loss.
#[tokio::test]
async fn long_summary_node_round_trips_without_truncation() {
    let (app, _td) = make_app();

    // Build a 10,000-character summary: repeating "x" is cheap and
    // deterministic; any change in the returned byte count is detectable.
    let long_summary = "x".repeat(10_000);

    let body = serde_json::json!({
        "label": "Memory",
        "summary": long_summary,
        "author": "tests"
    });

    // POST /v1/nodes - must succeed and return an id.
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
        "POST /v1/nodes with 10k-char summary must succeed"
    );
    let j = to_json(resp.into_body()).await;
    let id = j["id"].as_str().unwrap().to_string();

    // GET /v1/nodes/{id} - must return the full, untruncated summary.
    let resp = app
        .oneshot(
            Request::builder()
                .uri(format!("/v1/nodes/{id}"))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(
        resp.status(),
        StatusCode::OK,
        "GET /v1/nodes/{id} must succeed for long-summary node"
    );
    let j = to_json(resp.into_body()).await;
    let returned_summary = j["summary"].as_str().unwrap_or("");
    assert_eq!(
        returned_summary.len(),
        10_000,
        "retrieved summary must be exactly 10,000 chars (no truncation), got {}",
        returned_summary.len()
    );
    assert_eq!(
        returned_summary,
        "x".repeat(10_000),
        "retrieved summary bytes must be byte-identical to the original"
    );
}
