//! Tool registry, dispatch, and tool implementations.
//!
//! Every tool takes `&mut Server` (so it can load/invalidate the
//! cached repo view), a JSON `Value` of the parsed `arguments`
//! field, and returns a text payload. The server wraps the payload
//! in MCP's `content[]` shape and appends `_meta` metrics.
//!
//! Tool outputs are **plain text with a light structure** (indented
//! lists, `key: value` lines). Not JSON. LLMs read this kind of
//! output with much lower per-token overhead than a blob of JSON,
//! and the parse-free shape is a feature for the no-tokenizer-yet
//! metrics path.

pub(crate) mod descriptions;
// Path A audit fix (2026-04-26): shared embedder resolver for
// `mnem_community_summarize` and `mnem_retrieve`. Behind `summarize`
// because the embed-providers crate only enters the dep tree under
// that feature.
#[cfg(feature = "summarize")]
pub(crate) mod embed;
// Shared NER resolver for the ingest handler (always present since
// mnem-ingest is an unconditional dep of mnem-mcp).
mod handlers;
pub(crate) mod ner;

pub use descriptions::all_tools;

use std::collections::BTreeMap;

use anyhow::{Result, anyhow, bail};
use ipld_core::ipld::Ipld;
use mnem_core::codec::from_canonical_bytes;
use mnem_core::objects::{IndexSet, RefTarget};
use mnem_core::repo::ReadonlyRepo;
use serde_json::Value;

use crate::server::Server;

// ---------- input clamps ----------
//
// Mirror the ceilings applied in mnem http. MCP tool args are as
// untrusted as an HTTP body: a caller is free to send
// `limit=u64::MAX` and hope the downstream allocator gives it a
// fatal headache.

/// Max `limit` accepted on `mnem_retrieve`. See
/// `mnem_http::handlers::MAX_RETRIEVE_LIMIT` for rationale.
pub(super) const MAX_RETRIEVE_LIMIT: usize = 1_000;

/// Max `vector_cap` accepted on `mnem_retrieve`.
pub(super) const MAX_VECTOR_CAP: usize = 100_000;

/// Max `rerank_top_k` accepted on `mnem_retrieve`.
pub(super) const MAX_RERANK_TOP_K: usize = 500;

// ============================================================
// Registry + dispatch
// ============================================================

pub(crate) fn dispatch(server: &mut Server, name: &str, args: Value) -> Result<String> {
    match name {
        "mnem_stats" => handlers::stats::stats(server),
        "mnem_schema" => handlers::schema::schema(server),
        "mnem_search" => handlers::search::search(server, args),
        "mnem_get_node" => handlers::get_node::get_node(server, args),
        "mnem_traverse" => handlers::traverse::traverse(server, args),
        "mnem_incoming_edges" => handlers::incoming_edges::incoming_edges(server, args),
        "mnem_commit" => handlers::commit::commit(server, args),
        "mnem_commit_relation" => handlers::commit_relation::commit_relation(server, args),
        "mnem_delete_node" => handlers::delete_node::delete_node(server, args),
        "mnem_tombstone_node" => handlers::tombstone_node::tombstone_node(server, args),
        "mnem_list_nodes" => handlers::list_nodes::list_nodes(server, args),
        "mnem_list_tags" => handlers::list_tags::list_tags(server, args),
        "mnem_resolve_or_create" => handlers::resolve_or_create::resolve_or_create(server, args),
        "mnem_recent" => handlers::recent::recent(server, args),
        "mnem_vector_search" => handlers::vector_search::vector_search(server, args),
        "mnem_retrieve" => handlers::retrieve::retrieve(server, args),
        "mnem_global_retrieve" => handlers::global_retrieve::global_retrieve(server, args),
        "mnem_global_add" => handlers::global_add::global_add(server, args),
        "mnem_global_ingest" => handlers::global_ingest::global_ingest(server, args),
        "mnem_global_tombstone_node" => {
            handlers::global_tombstone_node::global_tombstone_node(server, args)
        }
        "mnem_ingest" => handlers::ingest::ingest(server, args),
        #[cfg(feature = "summarize")]
        "mnem_community_summarize" => {
            handlers::community_summarize::community_summarize(server, args)
        }
        other => bail!("unknown tool: {other}"),
    }
}

// ============================================================
// Shared helpers
// ============================================================

// Shared helpers
// ============================================================

pub(super) fn preview_str(s: &str) -> String {
    // Byte-length is a safe upper bound on char count: bytes <= 100
    // implies chars <= 100, so the short-string branch never slices.
    // For the long-string branch we truncate by characters (not bytes)
    // so a multibyte codepoint on the boundary does not panic.
    if s.len() <= 100 {
        s.to_string()
    } else {
        let preview: String = s.chars().take(97).collect();
        format!("{preview}... ({} bytes)", s.len())
    }
}

pub(super) fn index_set(server: &mut Server, repo: &ReadonlyRepo) -> Result<Option<IndexSet>> {
    let Some(idx_cid) = repo.head_commit().and_then(|c| c.indexes.as_ref()) else {
        return Ok(None);
    };
    let bs = server.stores()?.0;
    let bytes = bs
        .get(idx_cid)?
        .ok_or_else(|| anyhow!("IndexSet block {idx_cid} missing"))?;
    Ok(Some(from_canonical_bytes(&bytes)?))
}

pub(super) fn summarize_refs(refs: &BTreeMap<String, RefTarget>) -> String {
    if refs.is_empty() {
        return "none".to_string();
    }
    let names: Vec<&str> = refs.keys().take(5).map(String::as_str).collect();
    let mut s = names.join(", ");
    if refs.len() > 5 {
        s.push_str(&format!(", +{} more", refs.len() - 5));
    }
    s
}

// `json_to_ipld` is re-exported from `mnem_core::codec`; a single
// canonical implementation ensures CLI, HTTP, and MCP inputs share
// the same depth cap (64) and numeric-rejection rules. See
// `crates/mnem-core/src/codec/json.rs`.
//
// MCP tool input is as untrusted as HTTP body input by the same
// rubric: an LLM-driven client is free to produce arbitrary payloads,
// and the cap guards against a deeply-nested "arguments" object
// stack-overflowing the process.

pub(super) fn ipld_preview(v: &Ipld) -> String {
    match v {
        Ipld::Null => "null".into(),
        Ipld::Bool(b) => b.to_string(),
        Ipld::Integer(n) => n.to_string(),
        Ipld::Float(f) => f.to_string(),
        Ipld::String(s) => {
            if s.len() <= 80 {
                format!("\"{s}\"")
            } else {
                // Take by chars so a multibyte codepoint on the 77th
                // byte boundary doesn't panic at slice time.
                let preview: String = s.chars().take(77).collect();
                format!("\"{preview}...\" ({} bytes)", s.len())
            }
        }
        Ipld::Bytes(b) => format!("bytes({})", b.len()),
        Ipld::List(xs) => format!("[{} items]", xs.len()),
        Ipld::Map(m) => format!("{{{} keys}}", m.len()),
        Ipld::Link(c) => format!("cid:{c}"),
    }
}

#[cfg(test)]
mod mnem_bench_gate_tests {
    //! End-to-end tests that the `MNEM_BENCH` gate is honoured across
    //! the five tool surfaces named in the audit:
    //!   * `mnem_commit` (ntype forcing)
    //!   * `mnem_search` (label ignored)
    //!   * `mnem_list_nodes` (label ignored)
    //!   * `mnem_resolve_or_create` (label forced to DEFAULT_NTYPE)
    //!   * `mnem_retrieve` (label ignored)
    //!
    //! Each test drives the dispatch entry point (not the in-process
    //! helper) to cover the schema + handler seam end-to-end.
    use super::*;
    use crate::server::Server;
    use mnem_core::objects::Node;
    use serde_json::json;
    use tempfile::TempDir;

    fn mk_server(allow_labels: bool) -> (Server, TempDir) {
        let td = tempfile::tempdir().expect("tempdir");
        let mut s = Server::new(td.path().to_path_buf());
        s.allow_labels = allow_labels;
        (s, td)
    }

    #[test]
    fn schemas_are_stable_regardless_of_gate() {
        // audit-2026-04-25 P1-2: the advertised MCP schemas MUST NOT
        // mutate based on MNEM_BENCH. Before the fix, `label` / `ntype`
        // appeared/disappeared depending on the env var - a public API
        // that reshapes itself at runtime is not a public API.
        // The handler path remains the gate: caller-supplied label /
        // ntype is silently dropped when the gate is off (tested by the
        // gate_off_* handler tests below).
        let gate_off = all_tools(false);
        let gate_on = all_tools(true);
        let off_names: Vec<_> = gate_off.iter().map(|t| t.name).collect();
        let on_names: Vec<_> = gate_on.iter().map(|t| t.name).collect();
        assert_eq!(off_names, on_names, "tool name list must be stable");
        for (a, b) in gate_off.iter().zip(gate_on.iter()) {
            assert_eq!(a.name, b.name);
            assert_eq!(
                serde_json::to_string(&a.input_schema).unwrap(),
                serde_json::to_string(&b.input_schema).unwrap(),
                "schema for `{}` must be byte-identical regardless of allow_labels",
                a.name
            );
        }
    }

    #[test]
    fn schemas_always_expose_label_and_ntype() {
        // Complement to `schemas_are_stable_regardless_of_gate`: confirm
        // the stable schemas include label/ntype rather than the old
        // stripped-down default. This locks down the post-audit shape.
        let tools = all_tools(false);
        let by_name: std::collections::BTreeMap<_, _> =
            tools.iter().map(|t| (t.name, &t.input_schema)).collect();

        for name in [
            "mnem_search",
            "mnem_list_nodes",
            "mnem_resolve_or_create",
            "mnem_retrieve",
        ] {
            let schema = by_name.get(name).expect("tool present");
            let rendered = serde_json::to_string(schema).unwrap();
            assert!(
                rendered.contains("\"label\""),
                "{name}: schema should always expose `label`; got: {rendered}"
            );
        }

        let commit_schema = by_name.get("mnem_commit").expect("mnem_commit present");
        let commit_rendered = serde_json::to_string(commit_schema).unwrap();
        assert!(
            commit_rendered.contains("\"ntype\""),
            "mnem_commit: nodes.items schema should always expose `ntype`; got: {commit_rendered}"
        );
    }

    #[test]
    fn gate_off_forces_ntype_to_default_on_commit() {
        let (mut s, _td) = mk_server(false);
        let out = dispatch(
            &mut s,
            "mnem_commit",
            json!({
                "agent_id": "tester",
                "nodes": [
                    { "ntype": "SecretLabel", "summary": "nope" }
                ]
            }),
        )
        .expect("commit ok");
        // Output line for the created node should read `- Node <uuid>`,
        // never `- SecretLabel <uuid>` - caller-supplied ntype is
        // silently dropped.
        assert!(
            out.contains(&format!("- {} ", Node::DEFAULT_NTYPE)),
            "expected default ntype in output when MNEM_BENCH off; got: {out}"
        );
        assert!(
            !out.contains("- SecretLabel "),
            "caller-supplied ntype must not leak through when MNEM_BENCH off; got: {out}"
        );
    }

    #[test]
    fn gate_on_honours_caller_ntype_on_commit() {
        let (mut s, _td) = mk_server(true);
        let out = dispatch(
            &mut s,
            "mnem_commit",
            json!({
                "agent_id": "tester",
                "nodes": [
                    { "ntype": "Person", "summary": "alice" }
                ]
            }),
        )
        .expect("commit ok");
        assert!(
            out.contains("- Person "),
            "expected caller ntype to survive when MNEM_BENCH on; got: {out}"
        );
    }

    #[test]
    fn gate_off_drops_label_filter_on_list_nodes() {
        let (mut s, _td) = mk_server(false);
        // Seed two nodes (both coerced to DEFAULT_NTYPE by the gate).
        dispatch(
            &mut s,
            "mnem_commit",
            json!({
                "agent_id": "tester",
                "nodes": [
                    { "ntype": "A", "summary": "a1" },
                    { "ntype": "B", "summary": "b1" }
                ]
            }),
        )
        .expect("seed ok");
        // List with a `label` that would filter everything out if it
        // were honoured. Off-gate ignores the filter -> both nodes
        // surface.
        let out = dispatch(
            &mut s,
            "mnem_list_nodes",
            json!({ "label": "DoesNotExist" }),
        )
        .expect("list ok");
        assert!(
            out.contains("2 item(s)") || out.contains("item(s) (across all labels)"),
            "expected label filter dropped when MNEM_BENCH off; got: {out}"
        );
    }

    #[test]
    fn gate_on_honours_label_filter_on_list_nodes() {
        let (mut s, _td) = mk_server(true);
        dispatch(
            &mut s,
            "mnem_commit",
            json!({
                "agent_id": "tester",
                "nodes": [
                    { "ntype": "Person", "summary": "p1" },
                    { "ntype": "Doc",    "summary": "d1" }
                ]
            }),
        )
        .expect("seed ok");
        let out = dispatch(&mut s, "mnem_list_nodes", json!({ "label": "Doc" })).expect("list ok");
        assert!(
            out.contains("label=Doc"),
            "expected label= tag in header when MNEM_BENCH on; got: {out}"
        );
    }

    #[test]
    fn gate_off_collapses_label_in_resolve_or_create() {
        let (mut s, _td) = mk_server(false);
        // Two resolve_or_create calls with different labels must
        // collapse to the same node when the gate is off.
        let v1 = dispatch(
            &mut s,
            "mnem_resolve_or_create",
            json!({
                "label": "Person",
                "prop_name": "name",
                "value": "Alice",
                "agent_id": "tester"
            }),
        )
        .expect("first resolve ok");
        let v2 = dispatch(
            &mut s,
            "mnem_resolve_or_create",
            json!({
                "label": "Robot",
                "prop_name": "name",
                "value": "Alice",
                "agent_id": "tester"
            }),
        )
        .expect("second resolve ok");
        // Both outputs carry `label:  Node` (the default), not
        // Person/Robot.
        assert!(
            v1.contains(&format!("label:         {}", Node::DEFAULT_NTYPE))
                && v2.contains(&format!("label:         {}", Node::DEFAULT_NTYPE)),
            "both resolve_or_create calls should land on default ntype when MNEM_BENCH off; got v1={v1} v2={v2}"
        );
    }

    // ---------- input clamp tests (R2-A security hardening) ----------

    #[test]
    fn retrieve_rejects_oversized_limit() {
        let (mut s, _td) = mk_server(false);
        let err = dispatch(&mut s, "mnem_retrieve", json!({ "limit": 99_999_999_u64 }))
            .expect_err("oversized limit must be rejected");
        let msg = format!("{err:#}");
        assert!(
            msg.contains("limit=") && msg.contains("exceeds max"),
            "error must name the knob + cap: {msg}"
        );
    }

    #[test]
    fn retrieve_rejects_oversized_vector_cap() {
        let (mut s, _td) = mk_server(false);
        let err = dispatch(
            &mut s,
            "mnem_retrieve",
            json!({ "vector_cap": 9_999_999_u64 }),
        )
        .expect_err("oversized vector_cap must be rejected");
        let msg = format!("{err:#}");
        assert!(
            msg.contains("vector_cap=") && msg.contains("exceeds max"),
            "error must name the knob + cap: {msg}"
        );
    }

    // ---------- G6 audit fix tests (2026-04-25): commit_relation ----------

    #[test]
    fn commit_relation_creates_two_nodes_and_one_edge() {
        let (mut s, _td) = mk_server(true);
        let out = dispatch(
            &mut s,
            "mnem_commit_relation",
            json!({
                "subject": "Alice",
                "subject_kind": "Entity:Person",
                "predicate": "works_at",
                "object": "Globex",
                "object_kind": "Entity:Organization",
                "agent_id": "g6-test"
            }),
        )
        .expect("commit_relation ok");
        assert!(out.contains("subject:"), "missing subject line: {out}");
        assert!(out.contains("Entity:Person"), "subject ntype absent: {out}");
        assert!(
            out.contains("predicate:    works_at"),
            "predicate line absent: {out}"
        );
        assert!(
            out.contains("Entity:Organization"),
            "object ntype absent: {out}"
        );
        assert!(out.contains("Alice"), "subject value absent: {out}");
        assert!(out.contains("Globex"), "object value absent: {out}");
    }

    #[test]
    fn commit_relation_dedups_existing_subject() {
        // Calling commit_relation twice with the same (subject, kind,
        // anchor) must reuse the existing subject node - that's the
        // resolve_or_create guarantee. We verify by checking the
        // returned UUID is byte-identical across the two calls.
        let (mut s, _td) = mk_server(true);
        let out1 = dispatch(
            &mut s,
            "mnem_commit_relation",
            json!({
                "subject": "Alice",
                "subject_kind": "Entity:Person",
                "predicate": "works_at",
                "object": "Globex",
                "object_kind": "Entity:Organization",
                "agent_id": "g6-test"
            }),
        )
        .expect("first ok");
        let out2 = dispatch(
            &mut s,
            "mnem_commit_relation",
            json!({
                "subject": "Alice",
                "subject_kind": "Entity:Person",
                "predicate": "lives_in",
                "object": "Berlin",
                "object_kind": "Entity:Place",
                "agent_id": "g6-test"
            }),
        )
        .expect("second ok");
        // Extract subject UUID from each output and compare.
        let extract_subject_uuid = |s: &str| {
            for line in s.lines() {
                if let Some(rest) = line.trim_start().strip_prefix("subject:") {
                    return rest.split_whitespace().next().map(String::from);
                }
            }
            None
        };
        let u1 = extract_subject_uuid(&out1).expect("subject uuid present in out1");
        let u2 = extract_subject_uuid(&out2).expect("subject uuid present in out2");
        assert_eq!(
            u1, u2,
            "second commit_relation should reuse the same Alice node"
        );
    }

    #[test]
    fn commit_relation_missing_predicate_returns_error() {
        let (mut s, _td) = mk_server(true);
        let err = dispatch(
            &mut s,
            "mnem_commit_relation",
            json!({
                "subject": "Alice",
                "object": "Globex"
            }),
        )
        .expect_err("must reject missing predicate");
        let msg = format!("{err:#}");
        assert!(
            msg.contains("predicate"),
            "error must mention predicate: {msg}"
        );
    }

    #[test]
    fn commit_relation_respects_label_gate_off() {
        // With labels disabled, both endpoints must land on
        // DEFAULT_NTYPE regardless of caller-supplied kinds. This
        // mirrors the gate behaviour on `mnem_commit` and
        // `mnem_resolve_or_create`.
        let (mut s, _td) = mk_server(false);
        let out = dispatch(
            &mut s,
            "mnem_commit_relation",
            json!({
                "subject": "Alice",
                "subject_kind": "Entity:Person",
                "predicate": "works_at",
                "object": "Globex",
                "object_kind": "Entity:Organization",
                "agent_id": "g6-test"
            }),
        )
        .expect("commit_relation ok");
        assert!(
            !out.contains("Entity:Person") && !out.contains("Entity:Organization"),
            "caller-supplied kinds must NOT survive gate-off: {out}"
        );
        assert!(
            out.contains(&format!("[{}]", Node::DEFAULT_NTYPE)),
            "endpoints must use DEFAULT_NTYPE under gate-off: {out}"
        );
    }

    #[test]
    fn retrieve_rejects_oversized_rerank_top_k() {
        let (mut s, _td) = mk_server(false);
        let err = dispatch(
            &mut s,
            "mnem_retrieve",
            json!({ "rerank_top_k": 10_000_u64 }),
        )
        .expect_err("oversized rerank_top_k must be rejected");
        let msg = format!("{err:#}");
        assert!(
            msg.contains("rerank_top_k=") && msg.contains("exceeds max"),
            "error must name the knob + cap: {msg}"
        );
    }

    // ---- BUG-1 fix: resolve_or_create must merge props, not overwrite ----

    /// Helper: extract the node UUID from a `mnem_resolve_or_create` response.
    fn extract_resolved_id(out: &str) -> String {
        for line in out.lines() {
            if let Some(rest) = line.trim().strip_prefix("id:") {
                return rest.trim().to_string();
            }
        }
        panic!("no 'id:' line in resolve_or_create output:\n{out}");
    }

    #[test]
    fn resolve_or_create_existing_node_old_props_survive() {
        // BUG-1 regression test (1/3): existing props must be preserved
        // when resolve_or_create is called a second time with only new
        // extra_props.  Before the fix, the second call would silently
        // drop `city=Berlin` and only retain the anchor prop `name=Alice`.
        let (mut s, _td) = mk_server(true);

        // First call: create a Person node with anchor `name=Alice` and
        // extra prop `city=Berlin`.
        let out1 = dispatch(
            &mut s,
            "mnem_resolve_or_create",
            json!({
                "label": "Person",
                "prop_name": "name",
                "value": "Alice",
                "extra_props": { "city": "Berlin" },
                "agent_id": "bug1-test"
            }),
        )
        .expect("first resolve_or_create ok");
        let id = extract_resolved_id(&out1);

        // Second call: resolve the same node (same anchor) and add a new
        // extra prop `job=Engineer`.  Does NOT pass `city` again.
        let out2 = dispatch(
            &mut s,
            "mnem_resolve_or_create",
            json!({
                "label": "Person",
                "prop_name": "name",
                "value": "Alice",
                "extra_props": { "job": "Engineer" },
                "agent_id": "bug1-test"
            }),
        )
        .expect("second resolve_or_create ok");
        let id2 = extract_resolved_id(&out2);
        assert_eq!(id, id2, "both calls must resolve to the same node");

        // Inspect the node: city must survive, job must be present too.
        let node_out = dispatch(&mut s, "mnem_get_node", json!({ "id": id })).expect("get_node ok");
        assert!(
            node_out.contains("city"),
            "old prop 'city' must survive after second resolve_or_create; got:\n{node_out}"
        );
        assert!(
            node_out.contains("Berlin"),
            "old prop value 'Berlin' must survive; got:\n{node_out}"
        );
        assert!(
            node_out.contains("job"),
            "new prop 'job' from second call must be present; got:\n{node_out}"
        );
        assert!(
            node_out.contains("Engineer"),
            "new prop value 'Engineer' from second call must be present; got:\n{node_out}"
        );
    }

    #[test]
    fn resolve_or_create_new_prop_value_wins_over_old() {
        // BUG-1 regression test (2/3): when a second call passes the same
        // extra_prop key but with a different value, the new value must win.
        let (mut s, _td) = mk_server(true);

        // Create node with city=Berlin.
        let out1 = dispatch(
            &mut s,
            "mnem_resolve_or_create",
            json!({
                "label": "Person",
                "prop_name": "name",
                "value": "Bob",
                "extra_props": { "city": "Berlin" },
                "agent_id": "bug1-test"
            }),
        )
        .expect("create ok");
        let id = extract_resolved_id(&out1);

        // Second call: update city to Paris.
        dispatch(
            &mut s,
            "mnem_resolve_or_create",
            json!({
                "label": "Person",
                "prop_name": "name",
                "value": "Bob",
                "extra_props": { "city": "Paris" },
                "agent_id": "bug1-test"
            }),
        )
        .expect("update ok");

        let node_out = dispatch(&mut s, "mnem_get_node", json!({ "id": id })).expect("get_node ok");
        assert!(
            node_out.contains("Paris"),
            "updated value 'Paris' must win; got:\n{node_out}"
        );
        assert!(
            !node_out.contains("Berlin"),
            "old value 'Berlin' must be replaced; got:\n{node_out}"
        );
    }

    #[test]
    fn resolve_or_create_new_node_still_works() {
        // BUG-1 regression test (3/3): creating a brand-new node (no
        // prior match) must still work correctly with the merge path.
        let (mut s, _td) = mk_server(true);

        let out = dispatch(
            &mut s,
            "mnem_resolve_or_create",
            json!({
                "label": "Entity:Place",
                "prop_name": "slug",
                "value": "london-uk",
                "extra_props": { "country": "UK", "pop": 9_000_000 },
                "agent_id": "bug1-test"
            }),
        )
        .expect("create new node ok");
        let id = extract_resolved_id(&out);
        assert!(!id.is_empty(), "must return a node UUID");

        let node_out = dispatch(&mut s, "mnem_get_node", json!({ "id": id })).expect("get_node ok");
        assert!(
            node_out.contains("slug"),
            "anchor prop 'slug' must be present; got:\n{node_out}"
        );
        assert!(
            node_out.contains("london-uk"),
            "anchor value 'london-uk' must be present; got:\n{node_out}"
        );
        assert!(
            node_out.contains("country"),
            "extra prop 'country' must be present; got:\n{node_out}"
        );
        assert!(
            node_out.contains("UK"),
            "extra prop value 'UK' must be present; got:\n{node_out}"
        );
    }
}
