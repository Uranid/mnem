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
    assert_eq!(resp["result"]["serverInfo"]["name"], "mnem mcp");
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
            "id": "00000000-0000-0000-0000-000000000000",
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

// ============================================================
// BUG-1 regression: resolve-or-create must not overwrite existing props.
// ============================================================

/// Call resolve-or-create twice for the same entity. The second call adds
/// a different extra_prop. After both calls the node must have ALL props
/// from BOTH calls (merge semantics, not overwrite semantics).
#[test]
fn resolve_or_create_preserves_existing_props_on_re_resolve() {
    let (mut s, _td) = fresh_server(false);

    // First call: create Alice with an extra prop.
    let resp1 = tools_call(
        &mut s,
        "mnem_resolve_or_create",
        json!({
            "agent_id": "bug1-test",
            "prop_name": "name",
            "value": "Alice",
            "extra_props": { "role": "engineer" }
        }),
        1,
    );
    assert_success_response(&resp1, "mnem_resolve_or_create");
    let id = {
        let text = resp1["result"]["content"][0]["text"].as_str().unwrap();
        // extract id line
        text.lines()
            .find(|l| l.trim_start().starts_with("id:"))
            .and_then(|l| l.split_whitespace().nth(1))
            .expect("id must appear in response")
            .to_string()
    };

    // Second call: resolve the SAME entity and add another prop.
    let resp2 = tools_call(
        &mut s,
        "mnem_resolve_or_create",
        json!({
            "agent_id": "bug1-test",
            "prop_name": "name",
            "value": "Alice",
            "extra_props": { "department": "infra" }
        }),
        2,
    );
    assert_success_response(&resp2, "mnem_resolve_or_create");
    let id2 = {
        let text = resp2["result"]["content"][0]["text"].as_str().unwrap();
        text.lines()
            .find(|l| l.trim_start().starts_with("id:"))
            .and_then(|l| l.split_whitespace().nth(1))
            .expect("id must appear in response")
            .to_string()
    };
    assert_eq!(id, id2, "both calls must resolve to the SAME node id");

    // Now retrieve and verify ALL props are present.
    let resp3 = tools_call(&mut s, "mnem_get_node", json!({ "id": id }), 3);
    assert_success_response(&resp3, "mnem_get_node");
    let text = resp3["result"]["content"][0]["text"].as_str().unwrap();
    assert!(
        text.contains("engineer"),
        "BUG-1 regression: `role` from first call must survive second call; got: {text}"
    );
    assert!(
        text.contains("infra"),
        "BUG-1 regression: `department` from second call must be present; got: {text}"
    );
    assert!(
        text.contains("Alice"),
        "anchor prop `name=Alice` must survive; got: {text}"
    );
}

// ============================================================
// schema_introspection tests (Item-2 audit: edge type enumeration).
// ============================================================

/// Helper: extract the plain text from the first content element of a
/// tools/call response. Panics if the response is not a valid success.
fn extract_text(resp: &Value, tool: &str) -> String {
    assert_eq!(resp["jsonrpc"], "2.0", "{tool}: jsonrpc must be 2.0");
    assert!(
        resp.get("error").is_none(),
        "{tool}: unexpected JSON-RPC error: {resp:?}"
    );
    resp["result"]["content"][0]["text"]
        .as_str()
        .unwrap_or_else(|| panic!("{tool}: content[0].text must be a string; got {resp:?}"))
        .to_string()
}

/// A repo with nodes but no edges must report "index not built" for
/// edge types (the outgoing adjacency index is absent when no edges exist).
#[test]
fn schema_no_edges_shows_index_not_built() {
    let (mut s, _td) = fresh_server(true);

    // Commit one node (creates an IndexSet with no outgoing adjacency index).
    let resp = tools_call(
        &mut s,
        "mnem_commit",
        json!({
            "agent_id": "schema-no-edge-test",
            "nodes": [{ "ntype": "Entity:Person", "summary": "Alice" }]
        }),
        1,
    );
    assert_success_response(&resp, "mnem_commit");

    let resp2 = tools_call(&mut s, "mnem_schema", json!({}), 2);
    let text = extract_text(&resp2, "mnem_schema");

    // The node labels section must be present and show the committed label.
    assert!(
        text.contains("node labels:"),
        "schema must contain 'node labels:' section; got: {text}"
    );
    // Edge types section must be present with the "index not built" message
    // because no edges have been committed yet (outgoing adjacency = None).
    assert!(
        text.contains("edge types:"),
        "schema must contain 'edge types:' section; got: {text}"
    );
    assert!(
        text.contains("index not built"),
        "schema must show 'index not built' when no edges exist; got: {text}"
    );
}

/// After committing a relation (which builds the adjacency index as part
/// of the commit), schema must report the edge type used.
#[test]
fn schema_with_edge_shows_etype() {
    let (mut s, _td) = fresh_server(true);

    // Commit a relation: Alice works_at Globex.
    let resp = tools_call(
        &mut s,
        "mnem_commit_relation",
        json!({
            "subject": "Alice",
            "subject_kind": "Entity:Person",
            "predicate": "works_at",
            "object": "Globex",
            "object_kind": "Entity:Organization",
            "agent_id": "schema-etype-test"
        }),
        1,
    );
    assert_success_response(&resp, "mnem_commit_relation");

    // Schema must now include "works_at" in the edge types section.
    let resp2 = tools_call(&mut s, "mnem_schema", json!({}), 2);
    let text = extract_text(&resp2, "mnem_schema");
    assert!(
        text.contains("edge types:"),
        "schema must have 'edge types:' section after commit_relation; got: {text}"
    );
    assert!(
        text.contains("works_at"),
        "schema must list 'works_at' edge type after commit_relation; got: {text}"
    );
    assert!(
        !text.contains("index not built"),
        "schema must NOT show 'index not built' after edges have been committed; got: {text}"
    );
}

// ============================================================
// MCP lifecycle integration tests (Item-6 audit fix).
//
// These tests cover the full lifecycle of core MCP operations:
// commit -> retrieve (found) -> tombstone -> retrieve (not found)
// -> delete -> get_node (gone).
//
// Each test asserts OUTPUT CONTENT (the actual text returned),
// not just that no error occurred.  The tests are written against
// the GATE-ON server (`allow_labels = true`) so that ntype values
// survive and are visible in responses.
// ============================================================

// ---- commit output content ----

/// `mnem_commit` must report the op_id and commit_cid in its output.
#[test]
fn lifecycle_commit_output_contains_op_id_and_commit_cid() {
    let (mut s, _td) = fresh_server(true);
    let resp = tools_call(
        &mut s,
        "mnem_commit",
        json!({
            "agent_id": "lc-test",
            "nodes": [{ "ntype": "Fact", "summary": "The sky is blue" }]
        }),
        1,
    );
    assert_success_response(&resp, "mnem_commit");
    let text = resp["result"]["content"][0]["text"].as_str().unwrap();
    assert!(
        text.contains("mnem_commit: ok"),
        "commit output must start with 'mnem_commit: ok'; got: {text}"
    );
    assert!(
        text.contains("op_id:"),
        "commit output must include 'op_id:'; got: {text}"
    );
    // Verify the op_id value is non-empty and has CID shape (not just the label).
    let op_id_val = text
        .lines()
        .find(|line| line.contains("op_id:"))
        .and_then(|line| line.splitn(2, "op_id:").nth(1))
        .map(str::trim)
        .filter(|v| !v.is_empty())
        .expect("op_id: line must have a non-empty value");
    assert!(
        op_id_val.len() > 10,
        "op_id value must be CID-shaped (length > 10); got: '{op_id_val}'"
    );
    assert!(
        text.contains("commit_cid:"),
        "commit output must include 'commit_cid:'; got: {text}"
    );
    assert!(
        text.contains("nodes added: 1"),
        "commit output must report 'nodes added: 1'; got: {text}"
    );
}

/// `mnem_commit` output must list each created node's ntype and UUID.
#[test]
fn lifecycle_commit_output_lists_node_ntype_and_uuid() {
    let (mut s, _td) = fresh_server(true);
    let resp = tools_call(
        &mut s,
        "mnem_commit",
        json!({
            "agent_id": "lc-test",
            "nodes": [{ "ntype": "Entity:Person", "summary": "Alice" }]
        }),
        1,
    );
    assert_success_response(&resp, "mnem_commit");
    let text = resp["result"]["content"][0]["text"].as_str().unwrap();
    assert!(
        text.contains("Entity:Person"),
        "commit output must list the ntype 'Entity:Person'; got: {text}"
    );
    // UUID format: commit output lines "    - Entity:Person <uuid>"
    let uuid_str = text
        .lines()
        .find(|line| line.trim_start().starts_with("- ") && line.contains("Entity:Person"))
        .and_then(|line| line.split_whitespace().last())
        .expect("commit output must include a '- Entity:Person <uuid>' line")
        .to_string();
    // Validate the extracted token is UUID-shaped (8-4-4-4-12 hex format)
    assert_eq!(
        uuid_str.len(),
        36,
        "UUID must be 36 chars; got '{uuid_str}'"
    );
    assert_eq!(
        uuid_str.chars().filter(|&c| c == '-').count(),
        4,
        "UUID must have 4 hyphens; got '{uuid_str}'"
    );
    // All non-hyphen chars must be hex digits
    assert!(
        uuid_str.chars().all(|c| c == '-' || c.is_ascii_hexdigit()),
        "UUID must be lowercase hex with hyphens; got '{uuid_str}'"
    );
}

// ---- retrieve with where filter finds committed nodes ----

/// After committing a node with a prop, `mnem_retrieve` with a matching
/// `where` filter must return the node in its output.
#[test]
fn lifecycle_retrieve_where_filter_finds_committed_node() {
    let (mut s, _td) = fresh_server(true);

    // Commit a node with a known prop value.
    let commit_resp = tools_call(
        &mut s,
        "mnem_commit",
        json!({
            "agent_id": "lc-retrieve-test",
            "nodes": [{
                "ntype": "Fact",
                "summary": "Findable fact",
                "props": { "topic": "lifecycle-retrieve" }
            }]
        }),
        1,
    );
    assert_success_response(&commit_resp, "mnem_commit");

    // Retrieve using a where filter on the prop.
    let ret_resp = tools_call(
        &mut s,
        "mnem_retrieve",
        json!({
            "where": { "topic": "lifecycle-retrieve" }
        }),
        2,
    );
    assert_success_response(&ret_resp, "mnem_retrieve");
    let text = ret_resp["result"]["content"][0]["text"].as_str().unwrap();
    assert!(
        text.contains("1 item(s)"),
        "retrieve must return 1 item after committing a matching node; got: {text}"
    );
    assert!(
        text.contains("score="),
        "retrieve output must include score for each item; got: {text}"
    );
    assert!(
        text.contains("Findable fact") || text.contains("lifecycle-retrieve"),
        "retrieve output must include the node's content; got: {text}"
    );
}

/// `mnem_retrieve` with a `where` filter that matches nothing must report
/// 0 items (not an error). The repo must have at least one commit so the
/// op-heads store is initialized; otherwise the store is empty and returns
/// an initialization error before the filter can run.
#[test]
fn lifecycle_retrieve_where_filter_no_match_returns_zero_items() {
    let (mut s, _td) = fresh_server(true);

    // Prime the repo with one commit so the op-heads store is initialized.
    let _ = tools_call(
        &mut s,
        "mnem_commit",
        json!({
            "agent_id": "lc-no-match",
            "nodes": [{ "ntype": "Fact", "summary": "Priming the repo" }]
        }),
        1,
    );

    // Now retrieve with a filter that cannot match anything committed.
    let resp = tools_call(
        &mut s,
        "mnem_retrieve",
        json!({
            "where": { "topic": "this-value-does-not-exist" }
        }),
        2,
    );
    assert_success_response(&resp, "mnem_retrieve");
    let text = resp["result"]["content"][0]["text"].as_str().unwrap();
    assert!(
        text.contains("0 item(s)"),
        "retrieve with no matches must report '0 item(s)'; got: {text}"
    );
}

// ---- tombstone lifecycle ----

/// After tombstoning a committed node, `mnem_retrieve` must NOT return it
/// (tombstone filter is active by default per SPEC §4.10).
#[test]
fn lifecycle_tombstone_hides_node_from_retrieve() {
    let (mut s, _td) = fresh_server(true);

    // Commit a node with a unique prop for retrieval.
    let commit_resp = tools_call(
        &mut s,
        "mnem_commit",
        json!({
            "agent_id": "lc-tombstone-test",
            "nodes": [{
                "ntype": "Fact",
                "summary": "Soon to be forgotten",
                "props": { "topic": "tombstone-lifecycle" }
            }]
        }),
        1,
    );
    assert_success_response(&commit_resp, "mnem_commit");

    // Extract the node UUID from the commit output.
    let commit_text = commit_resp["result"]["content"][0]["text"]
        .as_str()
        .unwrap();
    let node_id = commit_text
        .lines()
        .find(|line| line.trim_start().starts_with("- ") && line.contains("Fact"))
        .and_then(|line| line.split_whitespace().last())
        .expect("commit output must include a '- Fact <uuid>' line")
        .to_string();

    // Verify the node appears in retrieve before tombstoning.
    let ret_before = tools_call(
        &mut s,
        "mnem_retrieve",
        json!({ "where": { "topic": "tombstone-lifecycle" } }),
        2,
    );
    assert_success_response(&ret_before, "mnem_retrieve");
    let text_before = ret_before["result"]["content"][0]["text"].as_str().unwrap();
    assert!(
        text_before.contains("1 item(s)"),
        "node must appear in retrieve before tombstone; got: {text_before}"
    );

    // Tombstone the node.
    let tomb_resp = tools_call(
        &mut s,
        "mnem_tombstone_node",
        json!({
            "agent_id": "lc-tombstone-test",
            "id": node_id,
            "reason": "lifecycle test forget"
        }),
        3,
    );
    assert_success_response(&tomb_resp, "mnem_tombstone_node");
    let tomb_text = tomb_resp["result"]["content"][0]["text"].as_str().unwrap();
    assert!(
        tomb_text.contains("mnem_tombstone_node: ok"),
        "tombstone output must say 'mnem_tombstone_node: ok'; got: {tomb_text}"
    );
    assert!(
        tomb_text.contains(&node_id),
        "tombstone output must include the node id; got: {tomb_text}"
    );
    assert!(
        tomb_text.contains("lifecycle test forget"),
        "tombstone output must include the reason; got: {tomb_text}"
    );
    assert!(
        tomb_text.contains("op_id:"),
        "tombstone output must include op_id; got: {tomb_text}"
    );
    let tomb_op_id_val = tomb_text
        .lines()
        .find(|line| line.contains("op_id:"))
        .and_then(|line| line.splitn(2, "op_id:").nth(1))
        .map(str::trim)
        .filter(|v| !v.is_empty())
        .expect("tombstone op_id: line must have a non-empty value");
    assert!(
        tomb_op_id_val.len() > 10,
        "tombstone op_id value must be CID-shaped (length > 10); got: '{tomb_op_id_val}'"
    );

    // After tombstone, retrieve must return 0 items.
    let ret_after = tools_call(
        &mut s,
        "mnem_retrieve",
        json!({ "where": { "topic": "tombstone-lifecycle" } }),
        4,
    );
    assert_success_response(&ret_after, "mnem_retrieve");
    let text_after = ret_after["result"]["content"][0]["text"].as_str().unwrap();
    assert!(
        text_after.contains("0 item(s)"),
        "tombstoned node must not appear in retrieve; got: {text_after}"
    );
}

/// Tombstoning an already-tombstoned node must produce a graceful
/// tool-level error (isError=true), not a JSON-RPC protocol error.
#[test]
fn lifecycle_tombstone_already_tombstoned_returns_tool_error() {
    let (mut s, _td) = fresh_server(true);

    // Commit a node.
    let commit_resp = tools_call(
        &mut s,
        "mnem_commit",
        json!({
            "agent_id": "lc-double-tomb",
            "nodes": [{ "ntype": "Fact", "summary": "Double tombstone test" }]
        }),
        1,
    );
    assert_success_response(&commit_resp, "mnem_commit");
    let commit_text = commit_resp["result"]["content"][0]["text"]
        .as_str()
        .unwrap();
    let node_id = commit_text
        .lines()
        .find(|line| line.trim_start().starts_with("- ") && line.contains("Fact"))
        .and_then(|line| line.split_whitespace().last())
        .expect("commit output must include a '- Fact <uuid>' line")
        .to_string();

    // First tombstone succeeds.
    let resp1 = tools_call(
        &mut s,
        "mnem_tombstone_node",
        json!({
            "agent_id": "lc-double-tomb",
            "id": node_id,
            "reason": "first tombstone"
        }),
        2,
    );
    assert_success_response(&resp1, "mnem_tombstone_node");

    // Second tombstone on same node must be a graceful tool error.
    let resp2 = tools_call(
        &mut s,
        "mnem_tombstone_node",
        json!({
            "agent_id": "lc-double-tomb",
            "id": node_id,
            "reason": "second tombstone"
        }),
        3,
    );
    assert_eq!(resp2["jsonrpc"], "2.0");
    assert!(
        resp2.get("error").is_none(),
        "double-tombstone must not produce JSON-RPC error; got: {resp2:?}"
    );
    assert_eq!(
        resp2["result"]["isError"], true,
        "double-tombstone must set isError=true; got: {resp2:?}"
    );
    let text = resp2["result"]["content"][0]["text"].as_str().unwrap();
    assert!(
        text.contains("already tombstoned"),
        "double-tombstone error must mention 'already tombstoned'; got: {text}"
    );
}

// ---- delete lifecycle ----

/// After deleting a committed node, `mnem_get_node` must report it as not
/// found.
#[test]
fn lifecycle_delete_removes_node_from_get_node() {
    let (mut s, _td) = fresh_server(true);

    // Commit a node.
    let commit_resp = tools_call(
        &mut s,
        "mnem_commit",
        json!({
            "agent_id": "lc-delete-test",
            "nodes": [{ "ntype": "Fact", "summary": "To be deleted" }]
        }),
        1,
    );
    assert_success_response(&commit_resp, "mnem_commit");
    let commit_text = commit_resp["result"]["content"][0]["text"]
        .as_str()
        .unwrap();
    let node_id = commit_text
        .lines()
        .find(|line| line.trim_start().starts_with("- ") && line.contains("Fact"))
        .and_then(|line| line.split_whitespace().last())
        .expect("commit output must include a '- Fact <uuid>' line")
        .to_string();

    // Verify it exists.
    let get_before = tools_call(&mut s, "mnem_get_node", json!({ "id": node_id }), 2);
    assert_success_response(&get_before, "mnem_get_node");
    let get_before_text = get_before["result"]["content"][0]["text"].as_str().unwrap();
    assert!(
        get_before_text.contains("To be deleted"),
        "get_node must show summary before delete; got: {get_before_text}"
    );

    // Delete the node.
    let del_resp = tools_call(
        &mut s,
        "mnem_delete_node",
        json!({
            "agent_id": "lc-delete-test",
            "id": node_id
        }),
        3,
    );
    assert_success_response(&del_resp, "mnem_delete_node");
    let del_text = del_resp["result"]["content"][0]["text"].as_str().unwrap();
    assert!(
        del_text.contains("mnem_delete_node: ok"),
        "delete output must say 'mnem_delete_node: ok'; got: {del_text}"
    );
    assert!(
        del_text.contains(&node_id),
        "delete output must include the node id; got: {del_text}"
    );
    assert!(
        del_text.contains("op_id:"),
        "delete output must include op_id; got: {del_text}"
    );

    // After delete, get_node must say not found.
    let get_after = tools_call(&mut s, "mnem_get_node", json!({ "id": node_id }), 4);
    assert_success_response(&get_after, "mnem_get_node");
    let get_after_text = get_after["result"]["content"][0]["text"].as_str().unwrap();
    assert!(
        get_after_text.contains("no node") || get_after_text.contains("not found"),
        "get_node must report not found after delete; got: {get_after_text}"
    );
}

/// Attempting to delete a non-existent node (valid UUID, not in repo) must
/// produce a graceful tool error (isError=true), not a JSON-RPC error.
#[test]
fn lifecycle_delete_nonexistent_node_returns_tool_error() {
    let (mut s, _td) = fresh_server(true);
    let resp = tools_call(
        &mut s,
        "mnem_delete_node",
        json!({
            "agent_id": "lc-delete-absent",
            "id": "11111111-1111-1111-1111-111111111111"
        }),
        1,
    );
    assert_eq!(resp["jsonrpc"], "2.0");
    assert!(
        resp.get("error").is_none(),
        "delete of absent node must not produce JSON-RPC error; got: {resp:?}"
    );
    assert_eq!(
        resp["result"]["isError"], true,
        "delete of absent node must set isError=true; got: {resp:?}"
    );
    let text = resp["result"]["content"][0]["text"].as_str().unwrap();
    assert!(
        text.contains("no node"),
        "delete absent-node error must mention 'no node'; got: {text}"
    );
}

// ---- tombstone missing required fields ----

/// `mnem_tombstone_node` with a missing `id` field must produce a
/// tool-level error (isError=true), not a JSON-RPC error.
#[test]
fn lifecycle_tombstone_missing_id_returns_tool_error() {
    let (mut s, _td) = fresh_server(false);
    let resp = tools_call(
        &mut s,
        "mnem_tombstone_node",
        json!({ "agent_id": "lc-tomb-missing", "reason": "no id given" }),
        1,
    );
    assert_eq!(resp["jsonrpc"], "2.0");
    assert!(resp.get("error").is_none());
    assert_eq!(
        resp["result"]["isError"], true,
        "tombstone without 'id' must set isError=true; got: {resp:?}"
    );
}

/// `mnem_tombstone_node` with a missing `agent_id` field must produce a
/// tool-level error.
#[test]
fn lifecycle_tombstone_missing_agent_id_returns_tool_error() {
    let (mut s, _td) = fresh_server(false);
    let resp = tools_call(
        &mut s,
        "mnem_tombstone_node",
        json!({
            "id": "00000000-0000-0000-0000-000000000000",
            "reason": "no agent"
        }),
        1,
    );
    assert_eq!(resp["jsonrpc"], "2.0");
    assert!(resp.get("error").is_none());
    assert_eq!(
        resp["result"]["isError"], true,
        "tombstone without 'agent_id' must set isError=true; got: {resp:?}"
    );
}

// ---- delete missing required fields ----

/// `mnem_delete_node` with a missing `agent_id` must produce a
/// tool-level error.
#[test]
fn lifecycle_delete_missing_agent_id_returns_tool_error() {
    let (mut s, _td) = fresh_server(false);
    let resp = tools_call(
        &mut s,
        "mnem_delete_node",
        json!({ "id": "00000000-0000-0000-0000-000000000000" }),
        1,
    );
    assert_eq!(resp["jsonrpc"], "2.0");
    assert!(resp.get("error").is_none());
    assert_eq!(
        resp["result"]["isError"], true,
        "delete without 'agent_id' must set isError=true; got: {resp:?}"
    );
}

// ---- full lifecycle: commit -> retrieve -> tombstone -> retrieve ----

/// Full end-to-end lifecycle: commit a node, confirm retrieve finds it
/// by `where` prop, tombstone it, confirm retrieve returns 0 items.
#[test]
fn lifecycle_full_commit_retrieve_tombstone_retrieve() {
    let (mut s, _td) = fresh_server(true);

    // Step 1: commit.
    let c = tools_call(
        &mut s,
        "mnem_commit",
        json!({
            "agent_id": "lc-full",
            "nodes": [{
                "ntype": "Event",
                "summary": "Full lifecycle event",
                "props": { "marker": "full-lifecycle-001" }
            }]
        }),
        1,
    );
    assert_success_response(&c, "mnem_commit");
    let c_text = c["result"]["content"][0]["text"].as_str().unwrap();
    let node_id = c_text
        .lines()
        .find(|line| line.trim_start().starts_with("- ") && line.contains("Event"))
        .and_then(|line| line.split_whitespace().last())
        .expect("step 1: commit output must include a '- Event <uuid>' line")
        .to_string();

    // Step 2: retrieve - must find the node.
    let r1 = tools_call(
        &mut s,
        "mnem_retrieve",
        json!({ "where": { "marker": "full-lifecycle-001" } }),
        2,
    );
    assert_success_response(&r1, "mnem_retrieve");
    let r1_text = r1["result"]["content"][0]["text"].as_str().unwrap();
    assert!(
        r1_text.contains("1 item(s)"),
        "step 2: retrieve must find committed node; got: {r1_text}"
    );
    assert!(
        r1_text.contains(&node_id),
        "step 2: retrieve output must include the node UUID; got: {r1_text}"
    );

    // Step 3: tombstone.
    let t = tools_call(
        &mut s,
        "mnem_tombstone_node",
        json!({
            "agent_id": "lc-full",
            "id": node_id,
            "reason": "full lifecycle tombstone"
        }),
        3,
    );
    assert_success_response(&t, "mnem_tombstone_node");
    let t_text = t["result"]["content"][0]["text"].as_str().unwrap();
    assert!(
        t_text.contains("op_id:"),
        "step 3: tombstone output must include op_id; got: {t_text}"
    );
    // Verify the tombstone op_id value is non-empty and CID-shaped.
    let t_op_id_val = t_text
        .lines()
        .find(|line| line.contains("op_id:"))
        .and_then(|line| line.splitn(2, "op_id:").nth(1))
        .map(str::trim)
        .filter(|v| !v.is_empty())
        .expect("step 3: tombstone op_id: line must have a non-empty value");
    assert!(
        t_op_id_val.len() > 10,
        "step 3: tombstone op_id value must be CID-shaped (length > 10); got: '{t_op_id_val}'"
    );

    // Step 4: retrieve after tombstone - must return 0 items.
    let r2 = tools_call(
        &mut s,
        "mnem_retrieve",
        json!({ "where": { "marker": "full-lifecycle-001" } }),
        4,
    );
    assert_success_response(&r2, "mnem_retrieve");
    let r2_text = r2["result"]["content"][0]["text"].as_str().unwrap();
    assert!(
        r2_text.contains("0 item(s)"),
        "step 4: tombstoned node must not appear in retrieve; got: {r2_text}"
    );
}

// ---- multiple nodes: selective tombstone ----

/// Tombstoning one node out of two must not affect the other; retrieve
/// must still return the surviving node.
#[test]
fn lifecycle_selective_tombstone_preserves_other_nodes() {
    let (mut s, _td) = fresh_server(true);

    // Commit the keep-node first (separate commit), extract its UUID directly.
    let c_keep = tools_call(
        &mut s,
        "mnem_commit",
        json!({
            "agent_id": "lc-selective",
            "nodes": [{
                "ntype": "Fact",
                "summary": "Keep this",
                "props": { "marker": "selective-keep" }
            }]
        }),
        1,
    );
    assert_success_response(&c_keep, "mnem_commit");
    let keep_text = c_keep["result"]["content"][0]["text"].as_str().unwrap();
    let keep_id = keep_text
        .lines()
        .find(|line| line.trim_start().starts_with("- ") && line.contains("Fact"))
        .and_then(|line| line.split_whitespace().last())
        .expect("keep commit must include a '- Fact <uuid>' line")
        .to_string();

    // Commit the drop-node separately so its UUID is unambiguously identified.
    let c_drop = tools_call(
        &mut s,
        "mnem_commit",
        json!({
            "agent_id": "lc-selective",
            "nodes": [{
                "ntype": "Fact",
                "summary": "Delete this",
                "props": { "marker": "selective-drop" }
            }]
        }),
        2,
    );
    assert_success_response(&c_drop, "mnem_commit");
    let drop_text = c_drop["result"]["content"][0]["text"].as_str().unwrap();
    let drop_id = drop_text
        .lines()
        .find(|line| line.trim_start().starts_with("- ") && line.contains("Fact"))
        .and_then(|line| line.split_whitespace().last())
        .expect("drop commit must include a '- Fact <uuid>' line")
        .to_string();

    // Tombstone only the drop node (unambiguously identified from its own commit response).
    let t = tools_call(
        &mut s,
        "mnem_tombstone_node",
        json!({
            "agent_id": "lc-selective",
            "id": drop_id,
            "reason": "selective tombstone"
        }),
        3,
    );
    assert_success_response(&t, "mnem_tombstone_node");

    // Retrieve the keep node - must still appear.
    let r_keep = tools_call(
        &mut s,
        "mnem_retrieve",
        json!({ "where": { "marker": "selective-keep" } }),
        4,
    );
    assert_success_response(&r_keep, "mnem_retrieve");
    let r_keep_text = r_keep["result"]["content"][0]["text"].as_str().unwrap();
    assert!(
        r_keep_text.contains("1 item(s)"),
        "non-tombstoned node must still appear in retrieve; got: {r_keep_text}"
    );
    assert!(
        r_keep_text.contains(&keep_id),
        "retrieve result must include the kept node's UUID; got: {r_keep_text}"
    );

    // Retrieve the dropped node - must be gone.
    let r_drop = tools_call(
        &mut s,
        "mnem_retrieve",
        json!({ "where": { "marker": "selective-drop" } }),
        5,
    );
    assert_success_response(&r_drop, "mnem_retrieve");
    let r_drop_text = r_drop["result"]["content"][0]["text"].as_str().unwrap();
    assert!(
        r_drop_text.contains("0 item(s)"),
        "tombstoned node must not appear in retrieve; got: {r_drop_text}"
    );
}

// ---- delete: invisible to mnem_retrieve after hard-delete ----

/// After hard-deleting a node, `mnem_retrieve` must also return 0 items for
/// that node (not just `mnem_get_node`). This exercises the prop-filter
/// retrieval lane to confirm that hard-deleted nodes do not leak through it.
#[test]
fn lifecycle_delete_invisible_to_retrieve() {
    let (mut s, _td) = fresh_server(true);

    // Commit a node with a distinctive prop so we can retrieve it by prop.
    let commit_resp = tools_call(
        &mut s,
        "mnem_commit",
        json!({
            "agent_id": "lc-delete-retrieve-test",
            "nodes": [{
                "ntype": "Fact",
                "summary": "Hard-delete retrieve visibility check",
                "props": { "delete_test_marker": "hard-delete-verify-xk7q" }
            }]
        }),
        1,
    );
    assert_success_response(&commit_resp, "mnem_commit");
    let commit_text = commit_resp["result"]["content"][0]["text"]
        .as_str()
        .unwrap();

    // Confirm the node appears in retrieve before deletion.
    let retrieve_before = tools_call(
        &mut s,
        "mnem_retrieve",
        json!({ "where": { "delete_test_marker": "hard-delete-verify-xk7q" } }),
        2,
    );
    assert_success_response(&retrieve_before, "mnem_retrieve");
    let retrieve_before_text = retrieve_before["result"]["content"][0]["text"]
        .as_str()
        .unwrap();
    assert!(
        retrieve_before_text.contains("1 item(s)"),
        "retrieve must find node before delete; got: {retrieve_before_text}"
    );

    // Extract the node UUID from the commit output.
    let node_id = commit_text
        .lines()
        .find(|line| line.trim_start().starts_with("- ") && line.contains("Fact"))
        .and_then(|line| line.split_whitespace().last())
        .expect("commit output must include a '- Fact <uuid>' line")
        .to_string();

    // Hard-delete the node.
    let del_resp = tools_call(
        &mut s,
        "mnem_delete_node",
        json!({
            "agent_id": "lc-delete-retrieve-test",
            "id": node_id
        }),
        3,
    );
    assert_success_response(&del_resp, "mnem_delete_node");

    // After delete, mnem_retrieve must also return 0 items for that node.
    let retrieve_after = tools_call(
        &mut s,
        "mnem_retrieve",
        json!({ "where": { "delete_test_marker": "hard-delete-verify-xk7q" } }),
        4,
    );
    assert_success_response(&retrieve_after, "mnem_retrieve");
    let retrieve_after_text = retrieve_after["result"]["content"][0]["text"]
        .as_str()
        .unwrap();
    assert!(
        retrieve_after_text.contains("0 item(s)"),
        "mnem_retrieve must return 0 items for hard-deleted node; got: {retrieve_after_text}"
    );
}

// ---- text= parameter contract tests ----
//
// The `text` parameter in `mnem_retrieve` is not a guaranteed base retrieval
// lane — its behavior depends on the feature configuration. The
// `RetrievalEmpty` guard in the retriever checks `label`, `prop_filter`,
// `vector_query`, and `sparse_query`. When the `bundled-embedder` feature is
// active (the default), passing `text=` causes the handler to auto-embed the
// text and use the resulting vector as a dense lane — so `text=` alone can
// produce hits. When the feature is absent, the guard fires and the handler
// returns a graceful "0 item(s)" empty-success.
//
// The two tests below document the stable parts of the contract:
//  A. `text=` alone must never crash or produce a JSON-RPC protocol error —
//     regardless of feature flags, the response is always graceful.
//  B. Adding `text=` to a `where=` query must never drop results that the
//     `where=` filter alone would have returned — text= is non-destructive.

/// Calling `mnem_retrieve` with ONLY a `text` parameter must NOT crash or
/// return a JSON-RPC protocol error. The response is always graceful: with the
/// bundled embedder the text is auto-embedded and may find nodes; without it
/// the `RetrievalEmpty` guard fires and the handler returns "0 item(s)".
/// This test verifies the non-crash, non-protocol-error contract across both
/// configurations.
#[test]
fn lifecycle_retrieve_text_only_without_base_ranker_is_graceful() {
    let (mut s, _td) = fresh_server(true);

    // Commit a node with a distinctive summary so the repo is initialized.
    let commit_resp = tools_call(
        &mut s,
        "mnem_commit",
        json!({
            "agent_id": "lc-text-only-test",
            "nodes": [{
                "ntype": "Fact",
                "summary": "Text-only retrieve graceful contract node xv9q"
            }]
        }),
        1,
    );
    assert_success_response(&commit_resp, "mnem_commit");

    // Call mnem_retrieve with ONLY text= and no base lane.
    // No `where`, no `label`, no `vector`.
    let ret_resp = tools_call(
        &mut s,
        "mnem_retrieve",
        json!({
            "text": "Text-only retrieve graceful contract node xv9q"
        }),
        2,
    );

    // Must never be a JSON-RPC protocol error regardless of feature flags.
    assert_eq!(ret_resp["jsonrpc"], "2.0");
    assert!(
        ret_resp.get("error").is_none(),
        "mnem_retrieve with text= only must not produce JSON-RPC error; got: {ret_resp:?}"
    );

    // The response must have the expected content[] shape.
    assert!(
        ret_resp["result"]["content"].is_array(),
        "mnem_retrieve response must keep content[] shape; got: {ret_resp:?}"
    );

    // Explicit check: must not be a JSON-RPC level error (the error field must be absent/null).
    // This guards against a future regression where RetrievalEmpty surfaces as isError=true
    // at the protocol level rather than as a graceful tool-level "0 item(s)" response.
    assert!(
        ret_resp["error"].is_null(),
        "mnem_retrieve with text= must not produce a JSON-RPC error; got: {ret_resp}"
    );
    assert!(
        ret_resp["result"]["isError"] != serde_json::json!(true),
        "mnem_retrieve with text= must not return isError=true in result; got: {ret_resp}"
    );

    // The text response must contain a valid retrieve header.
    // With bundled-embedder ON: "N item(s)" where N >= 0.
    // With bundled-embedder OFF: "0 item(s)" (graceful RetrievalEmpty path).
    // Either way: no panic, no protocol error, and the text contains
    // "mnem_retrieve" as the tool name prefix.
    let ret_text = ret_resp["result"]["content"][0]["text"].as_str().unwrap();
    assert!(
        ret_text.contains("mnem_retrieve"),
        "mnem_retrieve with text= only must produce a valid retrieve response; got: {ret_text}"
    );
    assert!(
        ret_text.contains("item(s)"),
        "mnem_retrieve with text= only must report item count; got: {ret_text}"
    );
}

/// Calling `mnem_retrieve` with a `where` prop filter returns 1 item whether
/// or not `text=` is also provided. This proves that `text=` is a non-destructive
/// no-op addon to the base ranker: it doesn't break the `where`-filter result
/// and doesn't add false hits. The original vacuous test (`lifecycle_retrieve_by_
/// text_query_finds_committed_node`) actually verified this property, but was
/// incorrectly described as exercising a "text-query code path". This test
/// documents that property accurately.
#[test]
fn lifecycle_retrieve_text_param_does_not_affect_where_results() {
    let (mut s, _td) = fresh_server(true);

    // Commit a node with a distinctive prop.
    let commit_resp = tools_call(
        &mut s,
        "mnem_commit",
        json!({
            "agent_id": "lc-text-where-test",
            "nodes": [{
                "ntype": "Fact",
                "summary": "Text param where result parity test",
                "props": { "text_test_marker": "lifecycle-text-param-test" }
            }]
        }),
        1,
    );
    assert_success_response(&commit_resp, "mnem_commit");

    // Retrieve with ONLY where= (no text). Must return 1 item.
    let ret_no_text = tools_call(
        &mut s,
        "mnem_retrieve",
        json!({
            "where": { "text_test_marker": "lifecycle-text-param-test" }
        }),
        2,
    );
    assert_success_response(&ret_no_text, "mnem_retrieve");
    let text_no_text = ret_no_text["result"]["content"][0]["text"]
        .as_str()
        .unwrap();
    assert!(
        text_no_text.contains("1 item(s)"),
        "retrieve with where= only must return 1 item; got: {text_no_text}"
    );

    // Retrieve with BOTH where= AND text=. Must also return 1 item.
    // text= does not break the where-filter result and does not add false hits.
    let ret_with_text = tools_call(
        &mut s,
        "mnem_retrieve",
        json!({
            "where": { "text_test_marker": "lifecycle-text-param-test" },
            "text": "lifecycle-text-param-test"
        }),
        3,
    );
    assert_success_response(&ret_with_text, "mnem_retrieve");
    let text_with_text = ret_with_text["result"]["content"][0]["text"]
        .as_str()
        .unwrap();
    assert!(
        text_with_text.contains("1 item(s)"),
        "retrieve with where= AND text= must still return 1 item \
         (text= is a non-destructive no-op addon to the where filter); got: {text_with_text}"
    );
}

/// Committing multiple edges of the same etype should list the etype only
/// once (deduplication).
#[test]
fn schema_edge_types_are_deduplicated() {
    let (mut s, _td) = fresh_server(true);

    // Commit two relations with the same predicate "knows".
    for (subject, object, id) in [("Alice", "Bob", 1u64), ("Bob", "Carol", 2u64)] {
        let resp = tools_call(
            &mut s,
            "mnem_commit_relation",
            json!({
                "subject": subject,
                "subject_kind": "Entity:Person",
                "predicate": "knows",
                "object": object,
                "object_kind": "Entity:Person",
                "agent_id": "schema-dedup-test"
            }),
            id,
        );
        assert_success_response(&resp, "mnem_commit_relation");
    }

    let resp = tools_call(&mut s, "mnem_schema", json!({}), 10);
    let text = extract_text(&resp, "mnem_schema");

    // Count occurrences of "knows" in the edge types section.
    let knows_count = text.matches("knows").count();
    assert_eq!(
        knows_count, 1,
        "edge type 'knows' should appear exactly once (deduplication); got {knows_count} in: {text}"
    );
}

/// An empty repo (no commits at all) must report the "no IndexSet" sentinel
/// rather than any crash or generic error.  This exercises the early-return
/// branch at the top of the schema handler.
#[test]
fn schema_empty_repo_reports_no_index_set() {
    let (mut s, _td) = fresh_server(false); // false = labels gate off; no initial commit
    let resp = tools_call(&mut s, "mnem_schema", json!({}), 1);
    let text = extract_text(&resp, "mnem_schema");
    assert!(
        text.contains("no IndexSet"),
        "Expected 'no IndexSet' message for empty repo, got: {text}"
    );
}

/// A repo where only nodes have been committed (never `mnem_commit_relation`)
/// must show "index not built" for edge types.  The outgoing adjacency CID
/// in the IndexSet is None, so `collect_edge_types` returns Ok(None) and the
/// `_` (absent/error) arm fires — NOT the Ok(Some(empty)) arm.
///
/// The `Ok(Some(empty_set))` arm — fired when `outgoing_cid` is `Some` but
/// the tree contains zero edge entries — is unreachable through the MCP
/// dispatch path because `build_index_set` always sets `outgoing = None`
/// when there are no edges.  That arm is exercised at unit-test level in
/// `crates/mnem-mcp/src/tools/handlers/schema.rs` (mod tests).
#[test]
fn schema_nodes_only_no_edge_index_shows_not_built() {
    let (mut s, _td) = fresh_server(true); // initial commit already done via allow_labels=true
    // Commit a standalone node — no edges.
    let _ = tools_call(
        &mut s,
        "mnem_commit",
        json!({
            "summary": "A standalone node with no edges",
            "ntype": "Entity:Person",
            "agent_id": "test"
        }),
        1,
    );
    let resp = tools_call(&mut s, "mnem_schema", json!({}), 2);
    let text = extract_text(&resp, "mnem_schema");
    assert!(
        text.contains("index not built"),
        "Expected 'index not built' for node-only commit, got: {text}"
    );
    assert!(
        !text.contains("<none — index present"),
        "Should not show empty-index message when no outgoing index CID exists: {text}"
    );
}

/// Committing edges of two different etypes must list both in schema.
#[test]
fn schema_multiple_etypes_all_listed() {
    let (mut s, _td) = fresh_server(true);

    // Commit two relations with different predicates.
    let resp1 = tools_call(
        &mut s,
        "mnem_commit_relation",
        json!({
            "subject": "Alice",
            "subject_kind": "Entity:Person",
            "predicate": "works_at",
            "object": "Globex",
            "object_kind": "Entity:Organization",
            "agent_id": "schema-multi-test"
        }),
        1,
    );
    assert_success_response(&resp1, "mnem_commit_relation");

    let resp2 = tools_call(
        &mut s,
        "mnem_commit_relation",
        json!({
            "subject": "Alice",
            "subject_kind": "Entity:Person",
            "predicate": "knows",
            "object": "Bob",
            "object_kind": "Entity:Person",
            "agent_id": "schema-multi-test"
        }),
        2,
    );
    assert_success_response(&resp2, "mnem_commit_relation");

    let resp = tools_call(&mut s, "mnem_schema", json!({}), 3);
    let text = extract_text(&resp, "mnem_schema");

    assert!(
        text.contains("works_at"),
        "schema must list 'works_at' edge type; got: {text}"
    );
    assert!(
        text.contains("knows"),
        "schema must list 'knows' edge type; got: {text}"
    );
    assert!(
        !text.contains("index not built"),
        "schema must not show 'index not built' when edges exist; got: {text}"
    );
}
