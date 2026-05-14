//! Integration tests for the `mnem_resolve_or_create` MCP tool.
//!
//! Covers:
//! - Happy-path canonical shape (label + prop_name + value)
//! - Idempotency: same (label, prop_name, value) always resolves to the same UUID
//! - Distinct values produce distinct node IDs
//! - Name/kind alias shape (C3-10 backward-compat)
//! - Missing required fields returns a graceful tool error
//! - Extra props are accepted without error; second call with same canonical key returns same ID
//! - allow_labels=false uses DEFAULT_NTYPE automatically
//! - allow_labels=false collapses different label fields to the same node
//! - global=true with absent global graph degrades gracefully (local commit still succeeds)

use mnem_mcp::Server;
use serde_json::{Value, json};
use tempfile::TempDir;

// ============================================================
// Helpers (mirror dispatch.rs pattern exactly)
// ============================================================

fn rpc(method: &str, params: Value, id: u64) -> String {
    serde_json::to_string(&serde_json::json!({
        "jsonrpc": "2.0",
        "id": id,
        "method": method,
        "params": params,
    }))
    .expect("serialise rpc")
}

fn fresh_server(allow_labels: bool) -> (Server, TempDir) {
    let tmp = TempDir::new().expect("mktemp");
    let mut server = Server::new(tmp.path().to_path_buf());
    server.allow_labels = allow_labels;
    (server, tmp)
}

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

/// Extract the UUID from an `id:   <uuid>` line in the tool output text.
///
/// Uses `splitn(2, ':')` so everything after the first colon is captured,
/// then trims whitespace. Asserts that the result is non-empty and 36 chars
/// (standard UUID length) so mis-parses are caught early.
fn extract_id(text: &str) -> String {
    let line = text
        .lines()
        .find(|l| l.trim_start().starts_with("id:"))
        .expect("response must contain id: line");
    let id = line
        .splitn(2, ':')
        .nth(1)
        .expect("id: line must have a colon")
        .trim()
        .to_string();
    assert!(
        !id.is_empty() && id.len() == 36,
        "extracted id must be a 36-char UUID string; got: {id:?} (from line: {line:?})"
    );
    id
}

// ============================================================
// Test 1: happy-path canonical shape
// ============================================================

/// A well-formed `mnem_resolve_or_create` call with label + prop_name + value
/// must succeed (no isError) and produce output containing `id:`, `label:`,
/// and `op_id:`.
#[test]
fn resolve_or_create_happy_path_canonical_shape() {
    let (mut s, _td) = fresh_server(true);
    let resp = tools_call(
        &mut s,
        "mnem_resolve_or_create",
        json!({
            "label":     "Person",
            "prop_name": "name",
            "value":     "Alice",
            "agent_id":  "test",
        }),
        1,
    );

    assert_eq!(resp["jsonrpc"], "2.0");
    assert!(
        resp.get("error").is_none(),
        "must not return a JSON-RPC protocol error: {resp:?}"
    );
    assert!(
        resp["result"]["isError"] != json!(true),
        "must not set isError=true on a valid call: {resp:?}"
    );

    let text = resp["result"]["content"][0]["text"]
        .as_str()
        .expect("content[0].text must be a string");

    assert!(
        text.contains("id:"),
        "response must contain 'id:' line; got: {text}"
    );
    assert!(
        text.contains("label:") && text.contains("Person"),
        "response must contain 'label:' and 'Person'; got: {text}"
    );
    assert!(
        text.contains("op_id:"),
        "response must contain 'op_id:' line; got: {text}"
    );
}

// ============================================================
// Test 2: idempotency - same (label, prop_name, value) -> same UUID
// ============================================================

/// Calling `mnem_resolve_or_create` twice with identical arguments must
/// return the same node UUID both times. This is the core dedup invariant.
#[test]
fn resolve_or_create_is_idempotent_same_id_returned() {
    let (mut s, _td) = fresh_server(true);

    let resp1 = tools_call(
        &mut s,
        "mnem_resolve_or_create",
        json!({
            "label":     "Person",
            "prop_name": "name",
            "value":     "Bob",
        }),
        1,
    );
    assert!(
        resp1["result"]["isError"] != json!(true),
        "first call must succeed: {resp1:?}"
    );
    let text1 = resp1["result"]["content"][0]["text"]
        .as_str()
        .expect("text must be a string");
    let id1 = extract_id(text1);

    let resp2 = tools_call(
        &mut s,
        "mnem_resolve_or_create",
        json!({
            "label":     "Person",
            "prop_name": "name",
            "value":     "Bob",
        }),
        2,
    );
    assert!(
        resp2["result"]["isError"] != json!(true),
        "second call must succeed: {resp2:?}"
    );
    let text2 = resp2["result"]["content"][0]["text"]
        .as_str()
        .expect("text must be a string");
    let id2 = extract_id(text2);

    assert_eq!(
        id1, id2,
        "same (label, prop_name, value) must always resolve to the same node UUID"
    );
}

// ============================================================
// Test 3: distinct values produce distinct node IDs
// ============================================================

/// Two calls with different `value` fields must produce different node UUIDs.
#[test]
fn resolve_or_create_different_values_create_different_nodes() {
    let (mut s, _td) = fresh_server(true);

    let resp_charlie = tools_call(
        &mut s,
        "mnem_resolve_or_create",
        json!({
            "label":     "Person",
            "prop_name": "name",
            "value":     "Charlie",
        }),
        1,
    );
    assert!(
        resp_charlie["result"]["isError"] != json!(true),
        "Charlie call must succeed: {resp_charlie:?}"
    );
    let text_charlie = resp_charlie["result"]["content"][0]["text"]
        .as_str()
        .expect("text");
    let id_charlie = extract_id(text_charlie);

    let resp_dana = tools_call(
        &mut s,
        "mnem_resolve_or_create",
        json!({
            "label":     "Person",
            "prop_name": "name",
            "value":     "Dana",
        }),
        2,
    );
    assert!(
        resp_dana["result"]["isError"] != json!(true),
        "Dana call must succeed: {resp_dana:?}"
    );
    let text_dana = resp_dana["result"]["content"][0]["text"]
        .as_str()
        .expect("text");
    let id_dana = extract_id(text_dana);

    assert_ne!(
        id_charlie, id_dana,
        "different values must produce different node UUIDs"
    );
}

// ============================================================
// Test 4: name/kind alias shape works (C3-10 backward compat)
// ============================================================

/// The `{name, kind}` alias shape (C3-10) must be accepted and produce a
/// valid response with an `id:` line.
#[test]
fn resolve_or_create_name_kind_alias_shape_works() {
    let (mut s, _td) = fresh_server(true);
    let resp = tools_call(
        &mut s,
        "mnem_resolve_or_create",
        json!({
            "name": "Eve",
            "kind": "Company",
        }),
        1,
    );

    assert_eq!(resp["jsonrpc"], "2.0");
    assert!(
        resp.get("error").is_none(),
        "must not return a JSON-RPC protocol error: {resp:?}"
    );
    assert!(
        resp["result"]["isError"] != json!(true),
        "name/kind alias must not set isError=true: {resp:?}"
    );

    let text = resp["result"]["content"][0]["text"]
        .as_str()
        .expect("content[0].text must be a string");

    assert!(
        text.contains("id:"),
        "name/kind alias response must contain 'id:'; got: {text}"
    );
}

// ============================================================
// Test 5: missing required fields returns error
// ============================================================

/// Calling `mnem_resolve_or_create` with no fields at all must produce a
/// graceful tool-level error (isError=true) and the content text must
/// mention "prop_name" or "name" to indicate the specific missing field.
/// It must never produce a JSON-RPC protocol error.
#[test]
fn resolve_or_create_missing_required_fields_returns_error() {
    let (mut s, _td) = fresh_server(true);
    let resp = tools_call(&mut s, "mnem_resolve_or_create", json!({}), 1);

    assert_eq!(resp["jsonrpc"], "2.0");
    assert!(
        resp.get("error").is_none(),
        "empty args must not return a JSON-RPC protocol error: {resp:?}"
    );

    // Must be a tool-level error.
    assert_eq!(
        resp["result"]["isError"],
        json!(true),
        "empty args must set isError=true; got: {resp:?}"
    );

    // The error text must name the specific missing field.
    // With allow_labels=true the handler hits the missing `label` check first,
    // so the error will mention "label" or "kind". With allow_labels=false it
    // would reach the `prop_name` check. Either way a specific field name appears.
    let text = resp["result"]["content"][0]["text"]
        .as_str()
        .unwrap_or("");
    assert!(
        text.contains("prop_name") || text.contains("name") || text.contains("label") || text.contains("kind"),
        "error text must mention the specific missing field ('label', 'kind', 'prop_name', or 'name'); got: {text:?}"
    );
}

// ============================================================
// Test 6: extra_props idempotency - same canonical key, different extra_props -> same node ID
// ============================================================

/// Passing `extra_props` alongside the canonical fields must not cause an
/// error. More importantly, a second call with the same canonical key
/// (label + prop_name + value) but DIFFERENT extra_props must still return
/// the SAME node UUID - proving extra_props are layered on top of the dedup
/// key rather than being part of it.
#[test]
fn resolve_or_create_extra_props_idempotent_same_node_id() {
    let (mut s, _td) = fresh_server(true);

    // Call 1: create the node with "First Title".
    let resp1 = tools_call(
        &mut s,
        "mnem_resolve_or_create",
        json!({
            "label":       "Document",
            "prop_name":   "url",
            "value":       "https://example.com",
            "extra_props": { "title": "First Title" },
        }),
        1,
    );
    assert_eq!(resp1["jsonrpc"], "2.0");
    assert!(
        resp1.get("error").is_none(),
        "extra_props call 1 must not cause a JSON-RPC protocol error: {resp1:?}"
    );
    assert!(
        resp1["result"]["isError"] != json!(true),
        "extra_props call 1 must not set isError=true: {resp1:?}"
    );
    let text1 = resp1["result"]["content"][0]["text"]
        .as_str()
        .expect("content[0].text must be a string");
    assert!(
        text1.contains("id:"),
        "call 1 response must contain 'id:'; got: {text1}"
    );
    let id1 = extract_id(text1);

    // Call 2: same canonical key, different extra_props.
    let resp2 = tools_call(
        &mut s,
        "mnem_resolve_or_create",
        json!({
            "label":       "Document",
            "prop_name":   "url",
            "value":       "https://example.com",
            "extra_props": { "title": "Second Title" },
        }),
        2,
    );
    assert_eq!(resp2["jsonrpc"], "2.0");
    assert!(
        resp2.get("error").is_none(),
        "extra_props call 2 must not cause a JSON-RPC protocol error: {resp2:?}"
    );
    assert!(
        resp2["result"]["isError"] != json!(true),
        "extra_props call 2 must not set isError=true: {resp2:?}"
    );
    let text2 = resp2["result"]["content"][0]["text"]
        .as_str()
        .expect("content[0].text must be a string");
    let id2 = extract_id(text2);

    assert_eq!(
        id1, id2,
        "same canonical key (label+prop_name+value) with different extra_props \
         must resolve to the same node UUID; extra_props must not be part of the dedup key"
    );
}

// ============================================================
// Test 7: allow_labels=false uses DEFAULT_NTYPE automatically
// ============================================================

/// When `allow_labels=false`, omitting the `label` field must succeed because
/// the handler falls back to `Node::DEFAULT_NTYPE` automatically.
#[test]
fn resolve_or_create_allow_labels_false_uses_default_type() {
    let (mut s, _td) = fresh_server(false); // gate OFF
    let resp = tools_call(
        &mut s,
        "mnem_resolve_or_create",
        json!({
            "prop_name": "slug",
            "value":     "test-doc",
            "agent_id":  "test",
        }),
        1,
    );

    assert_eq!(resp["jsonrpc"], "2.0");
    assert!(
        resp.get("error").is_none(),
        "gate-off without label must not produce JSON-RPC protocol error: {resp:?}"
    );
    assert!(
        resp["result"]["isError"] != json!(true),
        "gate-off without label must not set isError=true: {resp:?}"
    );

    let text = resp["result"]["content"][0]["text"]
        .as_str()
        .expect("content[0].text must be a string");
    assert!(
        text.contains("id:"),
        "gate-off response must contain 'id:'; got: {text}"
    );
}

// ============================================================
// Test 8: allow_labels=false collapses different label values to same node
// ============================================================

/// When `allow_labels=false`, two calls with the same `prop_name` + `value`
/// but different `label` fields must resolve to the same node UUID.
/// The label field is ignored; DEFAULT_NTYPE is always used, so the
/// dedup key `(DEFAULT_NTYPE, prop_name, value)` is identical for both calls.
#[test]
fn resolve_or_create_idempotent_across_label_boundary_when_labels_off() {
    let (mut s, _td) = fresh_server(false); // gate OFF - label is ignored

    let resp1 = tools_call(
        &mut s,
        "mnem_resolve_or_create",
        json!({
            "label":     "Person",
            "prop_name": "name",
            "value":     "Frank",
        }),
        1,
    );
    assert!(
        resp1["result"]["isError"] != json!(true),
        "first call (label=Person) must succeed: {resp1:?}"
    );
    let text1 = resp1["result"]["content"][0]["text"]
        .as_str()
        .expect("text");
    let id1 = extract_id(text1);

    let resp2 = tools_call(
        &mut s,
        "mnem_resolve_or_create",
        json!({
            "label":     "Organization",   // different label, ignored under gate-off
            "prop_name": "name",
            "value":     "Frank",
        }),
        2,
    );
    assert!(
        resp2["result"]["isError"] != json!(true),
        "second call (label=Organization) must succeed: {resp2:?}"
    );
    let text2 = resp2["result"]["content"][0]["text"]
        .as_str()
        .expect("text");
    let id2 = extract_id(text2);

    assert_eq!(
        id1, id2,
        "allow_labels=false: different label fields with same prop_name+value must \
         resolve to the same node UUID (label is ignored, DEFAULT_NTYPE always used)"
    );
}

// ============================================================
// Test 9: global=true with absent global graph degrades gracefully
// ============================================================

/// When `global: true` is passed but the global graph cannot be found, the
/// handler must degrade gracefully:
/// - The call must succeed (no isError)
/// - The response must contain an `id:` line (local commit succeeded)
///
/// Strategy: open the server with `repo_path` set to a fresh temp dir that
/// has no `.mnem` subdir, then point `MNEM_GLOBAL_DIR` at a different
/// nonexistent path via a child process. Because we cannot mutate env vars
/// safely inside a test binary (the crate forbids `unsafe`), we instead rely
/// on the natural state of the test machine: `~/.mnemglobal/.mnem/` almost
/// certainly does not exist in CI or on a fresh dev box.
///
/// Regardless of whether the global graph is present or absent, we only
/// assert that the call succeeds and returns an `id:`. We do NOT assert the
/// absence of `_global_anchor` so the test is robust on machines where the
/// global graph happens to exist.
#[test]
fn resolve_or_create_global_true_absent_graph_succeeds_locally() {
    let (mut s, _td) = fresh_server(true);
    let resp = tools_call(
        &mut s,
        "mnem_resolve_or_create",
        json!({
            "label":     "Person",
            "prop_name": "name",
            "value":     "GlobalTest",
            "agent_id":  "test",
            "global":    true,
        }),
        1,
    );

    assert_eq!(resp["jsonrpc"], "2.0");
    assert!(
        resp.get("error").is_none(),
        "global=true must not return a JSON-RPC protocol error regardless of \
         whether the global graph exists: {resp:?}"
    );
    assert!(
        resp["result"]["isError"] != json!(true),
        "global=true must not set isError=true regardless of whether the \
         global graph exists: {resp:?}"
    );

    let text = resp["result"]["content"][0]["text"]
        .as_str()
        .expect("content[0].text must be a string");

    assert!(
        text.contains("id:"),
        "response must contain 'id:' - local commit must have succeeded; got: {text}"
    );
    // Note: we do NOT assert the absence of `_global_anchor` here.
    // If the global graph exists on this machine, the call will also stamp it
    // and that is fine - the key invariant is that the call never fails.
}

// ============================================================
// Test 10: missing prop_name (with label present) returns specific error
// ============================================================

/// When `allow_labels=true` and `label` is provided but neither `prop_name`
/// nor the `name` alias is present, the handler must return a tool-level
/// error (isError=true) whose text mentions "prop_name" — the specific field
/// named in Arm 2 of the handler.
#[test]
fn resolve_or_create_missing_prop_name_returns_specific_error() {
    let (mut s, _td) = fresh_server(true);
    let resp_val = tools_call(
        &mut s,
        "mnem_resolve_or_create",
        json!({
            "label": "Person",
            // deliberately omit prop_name and name alias
        }),
        1,
    );

    let is_error = resp_val["result"]["isError"] == json!(true);
    let text = resp_val["result"]["content"][0]["text"]
        .as_str()
        .unwrap_or("");
    assert!(
        is_error,
        "expected isError=true when prop_name is absent, got: {resp_val:?}"
    );
    assert!(
        text.contains("prop_name"),
        "expected error to mention 'prop_name', got: {text:?}"
    );
}

// ============================================================
// Test 11: missing value (with label and prop_name present) returns specific error
// ============================================================

/// When `allow_labels=true`, `label` and `prop_name` are both provided, but
/// neither `value` nor the `name` alias is present, the handler must return a
/// tool-level error (isError=true) whose text mentions "value" — the specific
/// field named in Arm 3 of the handler.
#[test]
fn resolve_or_create_missing_value_returns_specific_error() {
    let (mut s, _td) = fresh_server(true);
    let resp_val = tools_call(
        &mut s,
        "mnem_resolve_or_create",
        json!({
            "label":     "Person",
            "prop_name": "name",
            // deliberately omit value and name alias
        }),
        1,
    );

    let is_error = resp_val["result"]["isError"] == json!(true);
    let text = resp_val["result"]["content"][0]["text"]
        .as_str()
        .unwrap_or("");
    assert!(
        is_error,
        "expected isError=true when value is absent, got: {resp_val:?}"
    );
    assert!(
        text.contains("value"),
        "expected error to mention 'value', got: {text:?}"
    );
}
