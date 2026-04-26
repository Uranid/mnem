//! Handler for the `mnem_traverse` MCP tool.
//!
//! Extracted from `tools.rs` in R3; body unchanged.

use crate::server::Server;
use anyhow::{Result, anyhow};
use mnem_core::id::NodeId;
use mnem_core::index::Query;
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
    let edge_labels: Vec<String> = args
        .get("edge_labels")
        .and_then(Value::as_array)
        .map(|a| {
            a.iter()
                .filter_map(Value::as_str)
                .map(String::from)
                .collect()
        })
        .unwrap_or_default();
    let limit = args.get("limit").and_then(Value::as_u64).unwrap_or(25) as usize;

    let repo = server.load_repo()?;
    let Some(node) = repo.lookup_node(&start)? else {
        return Ok(format!("mnem_traverse: start node {start_str} not found\n"));
    };

    // Build a Query hit on this one node so we can reuse the adjacency
    // path. For Week 1 the cleanest route is to run a full label-scan
    // query restricted to just this label, with the edges requested.
    let mut q = Query::new(&repo).label(node.ntype.as_str());
    for lbl in &edge_labels {
        q = q.with_outgoing(lbl.as_str());
    }
    let hits = q.limit(usize::MAX).execute()?;
    let hit = hits.into_iter().find(|h| h.node.id == start);

    let mut out = String::new();
    out.push_str(&format!(
        "mnem_traverse from {} ({}): ",
        node.ntype,
        start.to_uuid_string()
    ));
    match hit {
        None => {
            out.push_str("0 edges\n");
        }
        Some(h) => {
            let edges = h.edges.into_iter().take(limit).collect::<Vec<_>>();
            out.push_str(&format!("{} edge(s)\n", edges.len()));
            for e in edges {
                out.push_str(&format!("  -[{}]-> {}\n", e.etype, e.dst.to_uuid_string()));
            }
        }
    }
    Ok(out)
}
