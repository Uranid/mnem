//! Integration tests for the JSON-RPC dispatch path.
//!
//! Covers:
//!
//! - `initialize` + `tools/list` smoke tests (unchanged from A3).
//! - Per-tool round-trip: dispatch each registered tool through
//!   `handle_line` and assert the response shape is valid JSON-RPC 2.0
//!   with a `result.content` array carrying the `_meta` telemetry
//!   contract.
//! - Malformed-args failure tests: a handful of tools that require
//!   specific argument fields should return tool-level errors (NOT
//!   JSON-RPC protocol errors) when those fields are missing or
//!   ill-typed.
//! - Permission-boundary tests: `allow_labels = true` (the default
//!   as of the 2026-04-25 G3 audit fix) vs `allow_labels = false`
//!   (explicit operator opt-out via `MNEM_LABELS=0`). Schemas are
//!   stable across the boundary; dispatch behaviour differs in the
//!   documented ways.

use mnem_mcp::{Server, tool_names};
use serde_json::{Value, json};
use tempfile::TempDir;

/// Wrap a JSON-RPC 2.0 request line for a method + params with an
/// auto-incrementing id. Keeps the tests readable.
fn rpc(method: &str, params: Value, id: u64) -> String {
    serde_json::to_string(&serde_json::json!({
        "jsonrpc": "2.0",
        "id": id,
        "method": method,
        "params": params,
    }))
    .expect("serialise rpc")
}

/// Build a fresh `Server` bound to a throwaway tempdir. Returns the
/// `TempDir` so the caller can keep it alive for the duration of the
/// test (dropping it removes the underlying redb file).
fn fresh_server(allow_labels: bool) -> (Server, TempDir) {
    let tmp = TempDir::new().expect("mktemp");
    let mut server = Server::new(tmp.path().to_path_buf());
    server.allow_labels = allow_labels;
    (server, tmp)
}

/// Invoke `tools/call` for `name` with `args` and parse the JSON-RPC
/// response. Panics on a malformed response (which should never happen
/// in a well-behaved handler).
fn tools_call(server: &mut Server, name: &str, args: Value, id: u64) -> Value {
    let req = rpc(
        "tools/call",
        json!({
            "name": name,
            "arguments": args,
        }),
        id,
    );
    let line = server
        .handle_line(&req)
        .expect("tools/call must produce a response");
    serde_json::from_str(&line).expect("response must be JSON")
}

// ============================================================
// Smoke tests (pre-R2 baseline).
// ============================================================

#[test]
fn tools_list_advertises_every_registered_tool() {
    let tmp = TempDir::new().expect("mktemp");
    let mut server = Server::new(tmp.path().to_path_buf());

    // Expected names come from the library's own registry so the test
    // stays in sync without hard-coding the list here.
    let expected: Vec<&'static str> = tool_names(server.allow_labels);
    assert!(
        !expected.is_empty(),
        "tool_names() returned an empty list; registry regression"
    );

    let req = rpc("tools/list", serde_json::json!({}), 1);
    let line = server
        .handle_line(&req)
        .expect("tools/list should produce a response");

    let resp: Value = serde_json::from_str(&line).expect("parse response as JSON");
    assert_eq!(resp["jsonrpc"], "2.0");
    assert_eq!(resp["id"], 1);
    let tools = resp["result"]["tools"]
        .as_array()
        .expect("result.tools must be an array");

    let got: Vec<String> = tools
        .iter()
        .map(|t| {
            t["name"]
                .as_str()
                .expect("each tool must have a string name")
                .to_string()
        })
        .collect();

    for name in &expected {
        assert!(
            got.iter().any(|g| g == name),
            "tools/list response is missing `{name}`; got {got:?}"
        );
    }
    assert_eq!(
        got.len(),
        expected.len(),
        "tool count drift: registry reports {}, handler returned {}",
        expected.len(),
        got.len()
    );
}

#[test]
fn initialize_reports_protocol_version() {
    let tmp = TempDir::new().expect("mktemp");
    let mut server = Server::new(tmp.path().to_path_buf());

    let req = rpc("initialize", serde_json::json!({}), 42);
    let line = server
        .handle_line(&req)
        .expect("initialize should produce a response");

    let resp: Value = serde_json::from_str(&line).expect("parse response as JSON");
    assert_eq!(resp["id"], 42);
    assert_eq!(
        resp["result"]["protocolVersion"],
        mnem_mcp::MCP_PROTOCOL_VERSION,
        "handshake must expose the crate-level protocol version constant"
    );
    assert_eq!(resp["result"]["serverInfo"]["name"], "mnem-mcp");
}

#[test]
fn unknown_method_returns_method_not_found() {
    let tmp = TempDir::new().expect("mktemp");
    let mut server = Server::new(tmp.path().to_path_buf());

    let req = rpc("nope/not-real", serde_json::json!({}), 7);
    let line = server.handle_line(&req).expect("response expected");

    let resp: Value = serde_json::from_str(&line).expect("parse response as JSON");
    assert_eq!(resp["id"], 7);
    assert_eq!(
        resp["error"]["code"], -32601,
        "unknown method must map to JSON-RPC METHOD_NOT_FOUND"
    );
}

// ============================================================
// Per-tool round-trip tests (R2-C addition).
//
// For every tool registered in the gate-off registry, execute one
// representative `tools/call` and assert:
//   1. The response is a valid JSON-RPC 2.0 success (no `error`).
//   2. `result.content[0].type == "text"`.
//   3. `_meta.bytes`, `_meta.latency_micros`, `_meta.tokens_estimate`
//      all present (the agent-observability telemetry contract).
//
// We DO NOT assert on the textual content because some tools legitimately
// return messages like "no nodes" on a fresh repo. The point of this
// layer is the dispatch seam, not the rendering logic.
// ============================================================

/// Validate that `resp` is a well-formed success from `tools/call`.
fn assert_success_response(resp: &Value, tool: &str) {
    assert_eq!(resp["jsonrpc"], "2.0", "tool {tool}: jsonrpc must be 2.0");
    assert!(
        resp.get("error").is_none(),
        "tool {tool}: unexpected error field: {resp:?}"
    );
    let content = &resp["result"]["content"];
    let arr = content
        .as_array()
        .unwrap_or_else(|| panic!("tool {tool}: result.content must be an array, got {content:?}"));
    assert!(
        !arr.is_empty(),
        "tool {tool}: result.content must not be empty"
    );
    assert_eq!(
        arr[0]["type"], "text",
        "tool {tool}: result.content[0].type must be `text`"
    );
    let meta = &resp["result"]["_meta"];
    for key in ["bytes", "latency_micros", "tokens_estimate"] {
        assert!(
            meta.get(key).is_some(),
            "tool {tool}: _meta.{key} missing (telemetry contract broken): {meta:?}"
        );
    }
}

#[test]
fn roundtrip_mnem_stats_returns_success() {
    let (mut s, _td) = fresh_server(false);
    let resp = tools_call(&mut s, "mnem_stats", json!({}), 1);
    assert_success_response(&resp, "mnem_stats");
}

#[test]
fn roundtrip_mnem_schema_returns_success() {
    let (mut s, _td) = fresh_server(false);
    let resp = tools_call(&mut s, "mnem_schema", json!({}), 1);
    assert_success_response(&resp, "mnem_schema");
}

#[test]
fn roundtrip_mnem_search_empty_repo_returns_success() {
    let (mut s, _td) = fresh_server(false);
    let resp = tools_call(&mut s, "mnem_search", json!({}), 1);
    assert_success_response(&resp, "mnem_search");
}

#[test]
fn roundtrip_mnem_list_nodes_returns_success() {
    let (mut s, _td) = fresh_server(false);
    let resp = tools_call(&mut s, "mnem_list_nodes", json!({}), 1);
    assert_success_response(&resp, "mnem_list_nodes");
}

#[test]
fn roundtrip_mnem_recent_returns_success() {
    let (mut s, _td) = fresh_server(false);
    let resp = tools_call(&mut s, "mnem_recent", json!({ "limit": 5 }), 1);
    assert_success_response(&resp, "mnem_recent");
}

#[test]
fn roundtrip_mnem_commit_creates_node_and_returns_success() {
    let (mut s, _td) = fresh_server(false);
    let resp = tools_call(
        &mut s,
        "mnem_commit",
        json!({
            "agent_id": "round-trip-test",
            "nodes": [
                { "summary": "hello" }
            ]
        }),
        1,
    );
    assert_success_response(&resp, "mnem_commit");
    // Text content must mention the default ntype ("Node") since gate
    // is off; this is a belt-and-suspenders cross-check on gate
    // enforcement separate from the permission-boundary tests below.
    let text = resp["result"]["content"][0]["text"].as_str().unwrap();
    assert!(
        text.contains("Node ") || text.contains("- Node"),
        "mnem_commit round-trip text should show default ntype when gate off: {text}"
    );
}

#[test]
fn roundtrip_mnem_resolve_or_create_returns_success() {
    let (mut s, _td) = fresh_server(false);
    // `mnem_resolve_or_create` requires `prop_name` and `value`. Omitting
    // `label` is fine when the gate is off.
    let resp = tools_call(
        &mut s,
        "mnem_resolve_or_create",
        json!({
            "agent_id": "roc-test",
            "prop_name": "name",
            "value": "alice"
        }),
        1,
    );
    assert_success_response(&resp, "mnem_resolve_or_create");
}

#[test]
fn roundtrip_mnem_get_node_missing_id_returns_tool_error() {
    // `mnem_get_node` with a well-formed but absent UUID should return
    // a success response whose text says "no node found" -- it's a
    // semantic hit/miss, not a protocol error.
    let (mut s, _td) = fresh_server(false);
    let resp = tools_call(
        &mut s,
        "mnem_get_node",
        json!({ "id": "00000000-0000-0000-0000-000000000000" }),
        1,
    );
    assert_success_response(&resp, "mnem_get_node");
    let text = resp["result"]["content"][0]["text"].as_str().unwrap();
    assert!(
        text.contains("no node") || text.contains("not found"),
        "mnem_get_node on an absent id should say so in text: {text}"
    );
}

#[test]
fn roundtrip_mnem_vector_search_without_embed_reports_error() {
    // No embedder configured in a fresh tempdir, so the tool must
    // surface a tool-level error (isError=true) rather than panic.
    let (mut s, _td) = fresh_server(false);
    let resp = tools_call(
        &mut s,
        "mnem_vector_search",
        json!({ "query": "anything", "k": 3 }),
        1,
    );
    // Either a graceful tool-error (isError=true) or a graceful
    // text response indicating no model is configured; both are
    // acceptable. The protocol-level response MUST be jsonrpc=2.0
    // and MUST NOT be a JSON-RPC `error` object.
    assert_eq!(resp["jsonrpc"], "2.0");
    assert!(
        resp.get("error").is_none(),
        "mnem_vector_search must never return JSON-RPC error: {resp:?}"
    );
    // Content array is still present even on tool-error path.
    assert!(
        resp["result"]["content"].is_array(),
        "mnem_vector_search tool-error response must keep content[] shape"
    );
}

#[test]
fn roundtrip_mnem_retrieve_empty_returns_success_or_tool_error() {
    // `mnem_retrieve` on an empty request is rejected upstream by
    // `RepoError::RetrievalEmpty`. Surface should be a graceful
    // tool-error (isError=true with text), never a JSON-RPC error.
    let (mut s, _td) = fresh_server(false);
    let resp = tools_call(&mut s, "mnem_retrieve", json!({}), 1);
    assert_eq!(resp["jsonrpc"], "2.0");
    assert!(
        resp.get("error").is_none(),
        "mnem_retrieve empty call must not return JSON-RPC error: {resp:?}"
    );
    // Even on tool-error the _meta telemetry must be present.
    let meta = &resp["result"]["_meta"];
    for key in ["bytes", "latency_micros", "tokens_estimate"] {
        assert!(
            meta.get(key).is_some(),
            "_meta.{key} missing on mnem_retrieve tool-error: {meta:?}"
        );
    }
}

#[test]
fn roundtrip_mnem_ingest_markdown_file_returns_success() {
    // Write a small markdown file to a tempdir, then call `mnem_ingest`
    // on its absolute path. The handler must commit and return a
    // success response whose text reports a non-zero chunk count.
    let (mut s, td) = fresh_server(false);
    let file = td.path().join("hello.md");
    std::fs::write(
        &file,
        "# Title\n\nAlice Johnson met Bob Lee at Acme Corp on 2026-04-24.\n",
    )
    .expect("write fixture");
    let resp = tools_call(
        &mut s,
        "mnem_ingest",
        json!({
            "path": file.to_string_lossy(),
            "agent_id": "rt-ingest",
        }),
        1,
    );
    assert_success_response(&resp, "mnem_ingest");
    let text = resp["result"]["content"][0]["text"].as_str().unwrap();
    assert!(
        text.contains("chunk_count"),
        "mnem_ingest text should report chunk_count: {text}"
    );
    assert!(
        text.contains("commit_cid"),
        "mnem_ingest text should report commit_cid: {text}"
    );
}

#[test]
fn roundtrip_mnem_ingest_missing_path_returns_tool_error() {
    // Missing `path` must surface as a graceful tool-level error
    // (never a JSON-RPC protocol error). Same shape contract as the
    // other required-field omission tests above.
    let (mut s, _td) = fresh_server(false);
    let resp = tools_call(&mut s, "mnem_ingest", json!({}), 1);
    assert_eq!(resp["jsonrpc"], "2.0");
    assert!(
        resp.get("error").is_none(),
        "mnem_ingest without `path` must not return JSON-RPC error: {resp:?}"
    );
    assert!(resp["result"]["content"].is_array());
}

#[test]
fn roundtrip_mnem_delete_node_absent_is_graceful() {
    let (mut s, _td) = fresh_server(false);
    let resp = tools_call(
        &mut s,
        "mnem_delete_node",
        json!({
            "agent_id": "rt-test",
            "id": "00000000-0000-0000-0000-000000000000"
        }),
        1,
    );
    assert_eq!(resp["jsonrpc"], "2.0");
    assert!(resp.get("error").is_none());
    // Either a success message ("removed") or a tool-error. Either
    // way the response shape must be valid.
    assert!(resp["result"]["content"].is_array());
}

#[test]
fn roundtrip_mnem_tombstone_node_absent_is_graceful() {
    let (mut s, _td) = fresh_server(false);
    let resp = tools_call(
        &mut s,
        "mnem_tombstone_node",
        json!({
            "agent_id": "rt-test",
            "node_id": "00000000-0000-0000-0000-000000000000",
            "reason": "test"
        }),
        1,
    );
    assert_eq!(resp["jsonrpc"], "2.0");
    assert!(resp.get("error").is_none());
    assert!(resp["result"]["content"].is_array());
}

// ============================================================
// Malformed-args tests.
//
// The handlers should reject missing required fields with a clean
// tool-error (`isError: true`), not a panic and not a JSON-RPC
// protocol error. A strict schema validator would catch these
// client-side; our contract is defence-in-depth.
// ============================================================

#[test]
fn malformed_mnem_get_node_missing_id_returns_tool_error() {
    let (mut s, _td) = fresh_server(false);
    let resp = tools_call(&mut s, "mnem_get_node", json!({}), 1);
    assert_eq!(resp["jsonrpc"], "2.0");
    assert!(
        resp.get("error").is_none(),
        "expected tool-error not JSON-RPC error"
    );
    assert_eq!(
        resp["result"]["isError"], true,
        "mnem_get_node with no `id` must set isError=true: {resp:?}"
    );
}

#[test]
fn malformed_mnem_get_node_invalid_uuid_returns_tool_error() {
    let (mut s, _td) = fresh_server(false);
    let resp = tools_call(&mut s, "mnem_get_node", json!({ "id": "not-a-uuid" }), 1);
    assert_eq!(resp["jsonrpc"], "2.0");
    assert_eq!(
        resp["result"]["isError"], true,
        "mnem_get_node with invalid UUID must set isError=true: {resp:?}"
    );
}

#[test]
fn malformed_mnem_traverse_missing_start_returns_tool_error() {
    let (mut s, _td) = fresh_server(false);
    let resp = tools_call(&mut s, "mnem_traverse", json!({}), 1);
    assert_eq!(resp["jsonrpc"], "2.0");
    assert_eq!(resp["result"]["isError"], true);
}

#[test]
fn malformed_mnem_resolve_or_create_missing_prop_name_returns_tool_error() {
    let (mut s, _td) = fresh_server(false);
    let resp = tools_call(
        &mut s,
        "mnem_resolve_or_create",
        json!({ "agent_id": "t", "value": "v" }),
        1,
    );
    assert_eq!(resp["jsonrpc"], "2.0");
    assert_eq!(resp["result"]["isError"], true);
}

#[test]
fn malformed_tools_call_missing_name_returns_invalid_params() {
    // Distinct from the above: if the outer `tools/call` payload has
    // no `name`, the dispatch never reaches a handler. That's a
    // JSON-RPC protocol-level error (INVALID_PARAMS = -32602), NOT a
    // tool-error.
    let (mut s, _td) = fresh_server(false);
    let req = rpc("tools/call", json!({ "arguments": {} }), 1);
    let line = s.handle_line(&req).expect("response expected");
    let resp: Value = serde_json::from_str(&line).expect("parse response");
    assert_eq!(resp["jsonrpc"], "2.0");
    assert_eq!(
        resp["error"]["code"], -32602,
        "missing `name` must map to JSON-RPC INVALID_PARAMS"
    );
}

#[test]
fn malformed_tools_call_unknown_tool_returns_tool_error() {
    let (mut s, _td) = fresh_server(false);
    let resp = tools_call(&mut s, "mnem_not_a_real_tool", json!({}), 1);
    // Handler-level "unknown tool" surfaces as tool-error (to stay
    // parity with other tool-level failures), not a JSON-RPC
    // METHOD_NOT_FOUND (which is reserved for the outer method).
    assert_eq!(resp["jsonrpc"], "2.0");
    assert!(resp.get("error").is_none());
    assert_eq!(resp["result"]["isError"], true);
    let text = resp["result"]["content"][0]["text"].as_str().unwrap();
    assert!(
        text.contains("unknown tool"),
        "unknown-tool error text should mention 'unknown tool': {text}"
    );
}

// ============================================================
// Permission-boundary tests.
//
// The server ships two public surfaces: `allow_labels = false` (the
// casual-install default, corresponding to `MNEM_BENCH` unset) and
// `allow_labels = true` (the operator-bench opt-in). Several
// invariants must hold across the boundary:
//
// 1. `tool_names` returns the same list of tool names regardless of
//    gate state (tools count is stable; it's schemas that differ).
// 2. The `tools/list` RPC response, when introspected, hides the
//    `label` / `ntype` JSON-schema fields under gate-off.
// 3. `mnem_commit` silently drops caller-supplied `ntype` under
//    gate-off but honours it under gate-on.
// ============================================================

#[test]
fn boundary_tool_count_is_stable_across_gate() {
    // The NUMBER of tools must not shift based on the gate. Schemas
    // change; names do not.
    let off = tool_names(false);
    let on = tool_names(true);
    assert_eq!(
        off.len(),
        on.len(),
        "tool count must be stable across gate: off={off:?}, on={on:?}"
    );
    for name in &off {
        assert!(
            on.contains(name),
            "tool `{name}` present under gate-off but missing under gate-on; registry asymmetric"
        );
    }
}

#[test]
fn boundary_tools_list_schema_is_stable_across_gate() {
    // audit-2026-04-25 P1-2: schemas are now byte-stable regardless of
    // the MNEM_BENCH gate. Verify via the wire protocol (tools/list)
    // that the advertised schema includes `label` in both modes and is
    // identical between them.
    let (mut s_off, _td_off) = fresh_server(false);
    let (mut s_on, _td_on) = fresh_server(true);
    let req = rpc("tools/list", json!({}), 1);

    let resp_off: Value =
        serde_json::from_str(&s_off.handle_line(&req).expect("response expected"))
            .expect("parse off");
    let resp_on: Value = serde_json::from_str(&s_on.handle_line(&req).expect("response expected"))
        .expect("parse on");

    let schema_off = serde_json::to_string(&resp_off["result"]["tools"]).unwrap();
    let schema_on = serde_json::to_string(&resp_on["result"]["tools"]).unwrap();
    assert_eq!(
        schema_off, schema_on,
        "tools/list schemas must be identical across MNEM_BENCH gate"
    );
    assert!(
        schema_off.contains("\"label\""),
        "mnem_search schema should always expose `label` post-audit"
    );
}

#[test]
fn boundary_tools_list_schema_exposes_label_when_gate_on() {
    let (mut s, _td) = fresh_server(true);
    let req = rpc("tools/list", json!({}), 1);
    let line = s.handle_line(&req).expect("response expected");
    let resp: Value = serde_json::from_str(&line).expect("parse response");
    let tools = resp["result"]["tools"].as_array().expect("tools array");
    let search = tools
        .iter()
        .find(|t| t["name"] == "mnem_search")
        .expect("mnem_search present");
    let schema_str = serde_json::to_string(&search["inputSchema"]).unwrap();
    assert!(
        schema_str.contains("\"label\""),
        "mnem_search schema must expose `label` under gate-on: {schema_str}"
    );
}

#[test]
fn boundary_commit_coerces_ntype_when_gate_off() {
    // A full round-trip through dispatch + tools/call: with the gate
    // off, a commit carrying `"ntype": "Secret"` must still produce a
    // node with the default ntype. This is the single most important
    // permission invariant on the MCP surface.
    let (mut s, _td) = fresh_server(false);
    let resp = tools_call(
        &mut s,
        "mnem_commit",
        json!({
            "agent_id": "boundary-test",
            "nodes": [
                { "ntype": "SecretLabel", "summary": "nope" }
            ]
        }),
        1,
    );
    assert_success_response(&resp, "mnem_commit");
    let text = resp["result"]["content"][0]["text"].as_str().unwrap();
    assert!(
        !text.contains("SecretLabel"),
        "caller-supplied `ntype` must NOT survive gate-off: {text}"
    );
}

#[test]
fn boundary_commit_honours_ntype_when_gate_on() {
    let (mut s, _td) = fresh_server(true);
    let resp = tools_call(
        &mut s,
        "mnem_commit",
        json!({
            "agent_id": "boundary-test",
            "nodes": [
                { "ntype": "Person", "summary": "alice" }
            ]
        }),
        1,
    );
    assert_success_response(&resp, "mnem_commit");
    let text = resp["result"]["content"][0]["text"].as_str().unwrap();
    assert!(
        text.contains("Person"),
        "caller-supplied `ntype` MUST survive gate-on: {text}"
    );
}
