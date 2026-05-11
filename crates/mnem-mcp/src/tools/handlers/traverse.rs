//! Handler for the `mnem_traverse` MCP tool.
//!
//! fix(BUG-7): replaced O(N) label-scan + linear find with a direct
//! `repo.outgoing_edges()` call, matching what `mnem-cli traverse` does.
//! Also fixed: `edge_labels=[]` now maps to `None` (no filter = all edge
//! types) instead of `Some(&[])` (filter for nothing = 0 results).

use crate::server::Server;
use anyhow::{Context, Result, anyhow};
use mnem_core::id::NodeId;
use serde_json::Value;

// ============================================================
// mnem_traverse
// ============================================================

pub(in crate::tools) fn traverse(server: &mut Server, args: Value) -> Result<String> {
    let start_str = args
        .get("start")
        .and_then(Value::as_str)
        .ok_or_else(|| anyhow!("missing 'start'"))?;
    // audit-2026-04-25 C3-9 (Cycle-3): when callers pass a
    // human-readable name (e.g. "Alice") instead of a UUID, the
    // generic "invalid start UUID" parse error gives them no path
    // forward. Detect the non-UUID shape early and route them to
    // `mnem_resolve_or_create`, which is the canonical name->UUID
    // bridge for MCP clients.
    let start = match NodeId::parse_uuid(start_str) {
        Ok(id) => id,
        Err(e) => {
            return Err(anyhow!(
                "'start' must be a node UUID; got `{start_str}` ({e}). \
                 Resolve a name to a UUID first via `mnem_resolve_or_create` \
                 (pass {{name: \"{start_str}\", kind: \"<Label>\"}}), then \
                 pass the returned UUID here."
            ));
        }
    };

    // Collect requested edge-label filters; discard empty strings so that
    // callers who pass `edge_labels: []` get *all* edge types (None filter),
    // not zero results (Some(&[]) filter).
    let filter_strs: Vec<String> = args
        .get("edge_labels")
        .and_then(Value::as_array)
        .map(|a| {
            a.iter()
                .filter_map(Value::as_str)
                .filter(|s| !s.is_empty())
                .map(String::from)
                .collect()
        })
        .unwrap_or_default();

    // Build the optional filter slice: None means "all edge types".
    let filter_refs: Vec<&str> = filter_strs.iter().map(String::as_str).collect();
    let etype_filter: Option<&[&str]> = if filter_refs.is_empty() {
        None
    } else {
        Some(&filter_refs)
    };

    let limit = args.get("limit").and_then(Value::as_u64).unwrap_or(25) as usize;

    let repo = server.load_repo()?;
    let Some(node) = repo.lookup_node(&start)? else {
        return Ok(format!("mnem_traverse: start node {start_str} not found\n"));
    };

    // BUG-7 fix: call outgoing_edges() directly - O(1) adjacency index
    // lookup - instead of scanning all nodes with the same ntype label.
    let all_edges = repo
        .outgoing_edges(&start, etype_filter)
        .context("walking outgoing-adjacency index")?;
    let edges: Vec<_> = all_edges.into_iter().take(limit).collect();

    let mut out = String::new();
    out.push_str(&format!(
        "mnem_traverse from {} ({}): ",
        node.ntype,
        start.to_uuid_string()
    ));
    out.push_str(&format!("{} edge(s)\n", edges.len()));
    for e in &edges {
        out.push_str(&format!("  -[{}]-> {}\n", e.etype, e.dst.to_uuid_string()));
    }
    Ok(out)
}
