//! Handler for the `mnem_list_nodes` MCP tool.
//!
//! Extracted from `tools.rs` in R3; body unchanged.

use crate::server::Server;
use anyhow::Result;
use mnem_core::index::Query;
use serde_json::Value;

// ============================================================
// mnem_list_nodes
// ============================================================

pub(in crate::tools) fn list_nodes(server: &mut Server, args: Value) -> Result<String> {
    // `label` gated behind `MNEM_BENCH`. See `search` for rationale.
    let allow_labels = server.allow_labels;
    let label = if allow_labels {
        args.get("label").and_then(Value::as_str).map(String::from)
    } else {
        None
    };
    let limit = args.get("limit").and_then(Value::as_u64).unwrap_or(50) as usize;
    let offset = args.get("offset").and_then(Value::as_u64).unwrap_or(0) as usize;

    let repo = server.load_repo()?;
    let mut q = Query::new(&repo);
    if let Some(l) = &label {
        q = q.label(l.as_str());
    }
    // Ask for limit + offset; slice off the head ourselves so the
    // query layer does not have to know about pagination.
    let hits = q.limit(offset + limit).execute()?;

    let total_scanned = hits.len();
    let page: Vec<_> = hits.into_iter().skip(offset).take(limit).collect();

    let mut out = String::new();
    match &label {
        Some(l) => out.push_str(&format!(
            "mnem_list_nodes(label={l}): {} item(s)\n",
            page.len()
        )),
        None => out.push_str(&format!(
            "mnem_list_nodes: {} item(s) (across all labels)\n",
            page.len()
        )),
    }
    for hit in &page {
        out.push_str(&format!(
            "  {} [{}] {}\n",
            hit.node.id.to_uuid_string(),
            hit.node.ntype,
            hit.node.summary.as_deref().unwrap_or("")
        ));
    }
    if total_scanned >= offset + limit {
        out.push_str(&format!(
            "  ... (more available; call with offset={})\n",
            offset + limit
        ));
    }
    Ok(out)
}
