//! Handler for the `mnem_incoming_edges` MCP tool.
//!
//! Lists all edges pointing TO a given node (incoming / reverse edges).
//! This is the MCP counterpart of the CLI `mnem blame` command.

use crate::server::Server;
use anyhow::{Context, Result, anyhow};
use mnem_core::id::NodeId;
use serde_json::{Value, json};

// ============================================================
// mnem_incoming_edges
// ============================================================

/// Max `limit` accepted on `mnem_incoming_edges`.
const MAX_INCOMING_LIMIT: usize = 200;

pub(in crate::tools) fn incoming_edges(server: &mut Server, args: Value) -> Result<String> {
    // --- required: node UUID ---
    let node_str = args
        .get("node")
        .and_then(Value::as_str)
        .ok_or_else(|| anyhow!("missing 'node'"))?;

    // Give callers the same friendly error as mnem_traverse: detect
    // non-UUID shapes early and suggest the resolve path.
    let node_id = match NodeId::parse_uuid(node_str) {
        Ok(id) => id,
        Err(e) => {
            return Err(anyhow!(
                "'node' must be a node UUID; got `{node_str}` ({e}). \
                 Resolve a name to a UUID first via `mnem_resolve_or_create` \
                 (pass {{name: \"{node_str}\", kind: \"<Label>\"}}), then \
                 pass the returned UUID here."
            ));
        }
    };

    // --- optional: etype filter ---
    let etype: Option<String> = args.get("etype").and_then(Value::as_str).map(String::from);
    if let Some(ref s) = etype {
        if s.is_empty() {
            return Err(anyhow!(
                "etype filter cannot be an empty string; omit the field to return all edge types"
            ));
        }
    }

    // --- optional: limit (default 25, max 200) ---
    let raw_limit = args.get("limit").and_then(Value::as_u64).unwrap_or(25) as usize;
    if raw_limit == 0 {
        return Err(anyhow!("limit must be >= 1"));
    }
    let limit = raw_limit.min(MAX_INCOMING_LIMIT);

    // --- optional: json output ---
    let as_json = args.get("json").and_then(Value::as_bool).unwrap_or(false);

    // --- load repo ---
    let repo = server.load_repo()?;

    // Verify the destination node exists so we can show its ntype.
    let node_opt = repo
        .lookup_node(&node_id)
        .context("looking up destination node")?;

    let filter_str = etype.as_deref();
    let filter_slice = filter_str.map(|s| [s]);
    let filter_ref = filter_slice.as_ref().map(|arr| &arr[..]);

    // Use the capped variant so a hot-node can't DoS the server.
    let edges = repo
        .incoming_edges_capped(&node_id, filter_ref, limit)
        .context("walking incoming-adjacency index")?;

    if as_json {
        // JSON output
        let ntype = node_opt.as_ref().map_or("unknown", |n| n.ntype.as_str());
        let edge_arr: Vec<Value> = edges
            .iter()
            .map(|e| {
                json!({
                    "edge_id": e.id.to_uuid_string(),
                    "etype":   e.etype,
                    "src":     e.src.to_uuid_string()
                })
            })
            .collect();
        let out = json!({
            "node": {
                "id":    node_str,
                "ntype": ntype
            },
            "incoming_edges": edge_arr
        });
        return Ok(serde_json::to_string(&out)?);
    }

    // --- plain-text output (mirrors mnem blame) ---
    let mut out = String::new();

    match &node_opt {
        Some(n) => out.push_str(&format!(
            "node {} ({})\n",
            node_id.to_uuid_string(),
            n.ntype
        )),
        None => out.push_str(&format!(
            "node {} (not found in current commit)\n",
            node_id.to_uuid_string()
        )),
    }

    if edges.is_empty() {
        out.push_str("<no incoming edges>\n");
        return Ok(out);
    }

    for e in &edges {
        out.push_str(&format!(
            "{}  {:<16}  {}\n",
            e.id.to_uuid_string(),
            e.etype,
            e.src.to_uuid_string()
        ));
    }
    let truncated = edges.len() == limit && raw_limit >= MAX_INCOMING_LIMIT;
    out.push_str(&format!("({} incoming edge(s))\n", edges.len()));
    if truncated {
        out.push_str("(results capped at 200; use `etype` filter to narrow)\n");
    }

    Ok(out)
}
