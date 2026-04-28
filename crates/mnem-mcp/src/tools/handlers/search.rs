//! Handler for the `mnem_search` MCP tool.
//!
//! Extracted from `tools.rs` in R3; body unchanged.

use super::super::ipld_preview;
use crate::server::Server;
use anyhow::Result;
use mnem_core::codec::json_to_ipld;
use mnem_core::index::{PropPredicate, Query};
use serde_json::Value;

// ============================================================
// mnem_search
// ============================================================

pub(in crate::tools) fn search(server: &mut Server, args: Value) -> Result<String> {
    // `label` is gated behind `MNEM_BENCH`. When the gate is off,
    // caller-supplied values are silently dropped so the search runs
    // unscoped. Parity with POST /v1/retrieve in mnem-http.
    let allow_labels = server.allow_labels;
    let repo = server.load_repo()?;
    let label = if allow_labels {
        args.get("label").and_then(Value::as_str)
    } else {
        None
    };
    let limit = args.get("limit").and_then(Value::as_u64).unwrap_or(10) as usize;
    let with_outgoing = args
        .get("with_outgoing")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();

    let mut q = Query::new(&repo);
    if let Some(l) = label {
        q = q.label(l);
    }
    if let Some(Value::Object(map)) = args.get("where") {
        if let Some((k, v)) = map.iter().next() {
            let ipld = json_to_ipld(v)?;
            q = q.where_prop(k, PropPredicate::Eq(ipld));
        }
    }
    for lbl in &with_outgoing {
        if let Some(s) = lbl.as_str() {
            q = q.with_outgoing(s);
        }
    }
    q = q.limit(limit);
    let hits = q.execute()?;

    let mut out = String::new();
    out.push_str(&format!("mnem_search: {} hit(s)\n", hits.len()));
    for (i, hit) in hits.iter().enumerate() {
        out.push_str(&format!(
            "  [{i}] {} id={} \n",
            hit.node.ntype,
            hit.node.id.to_uuid_string()
        ));
        for (k, v) in &hit.node.props {
            out.push_str(&format!("        {k}: {}\n", ipld_preview(v)));
        }
        for edge in &hit.edges {
            out.push_str(&format!(
                "        -[{}]-> {}\n",
                edge.etype,
                edge.dst.to_uuid_string()
            ));
        }
    }
    Ok(out)
}
