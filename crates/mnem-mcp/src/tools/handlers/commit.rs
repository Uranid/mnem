//! Handler for the `mnem_commit` MCP tool.
//!
//! Extracted from `tools.rs` in R3; body unchanged.

use crate::server::Server;
use anyhow::{Context, Result, anyhow};
use mnem_core::codec::json_to_ipld;
use mnem_core::id::{EdgeId, NodeId};
use mnem_core::objects::{Edge, Node};
use serde_json::Value;

// ============================================================
// mnem_commit
// ============================================================

pub(in crate::tools) fn commit(server: &mut Server, args: Value) -> Result<String> {
    let allow_labels = server.allow_labels;
    let agent_id = args
        .get("agent_id")
        .and_then(Value::as_str)
        .ok_or_else(|| anyhow!("missing 'agent_id'"))?
        .to_string();
    let task_id = args
        .get("task_id")
        .and_then(Value::as_str)
        .map(String::from);
    let message = args
        .get("message")
        .and_then(Value::as_str)
        .unwrap_or("mnem_mcp commit")
        .to_string();

    let repo = server.load_repo()?;
    let mut tx = repo.start_transaction();

    let mut created_nodes: Vec<(String, NodeId)> = Vec::new();

    if let Some(nodes) = args.get("nodes").and_then(Value::as_array) {
        for nv in nodes {
            // Two-step ntype resolution (parity with POST /v1/nodes):
            //   1. `MNEM_BENCH` unset (`allow_labels == false`): every
            //      ingested node gets `Node::DEFAULT_NTYPE` regardless of
            //      what the caller sent. Schema already hides `ntype`,
            //      but a hand-crafted payload that includes it still
            //      cannot leak per-item scoping state.
            //   2. `MNEM_BENCH=1`: caller-supplied `ntype` honoured; if
            //      missing or empty it still falls back to the default.
            let ntype = if allow_labels {
                nv.get("ntype")
                    .and_then(Value::as_str)
                    .filter(|s| !s.trim().is_empty())
                    .unwrap_or(Node::DEFAULT_NTYPE)
            } else {
                Node::DEFAULT_NTYPE
            };
            let mut node = Node::new(NodeId::new_v7(), ntype);
            if let Some(summary) = nv.get("summary").and_then(Value::as_str) {
                node = node.with_summary(summary);
            }
            if let Some(Value::Object(props)) = nv.get("props") {
                for (k, v) in props {
                    node = node.with_prop(k.clone(), json_to_ipld(v)?);
                }
            }
            if let Some(content) = nv.get("content").and_then(Value::as_str) {
                node = node.with_content(bytes::Bytes::from(content.to_string().into_bytes()));
            }
            tx.add_node(&node)?;
            created_nodes.push((ntype.to_string(), node.id));
        }
    }

    if let Some(edges) = args.get("edges").and_then(Value::as_array) {
        for ev in edges {
            let etype = ev
                .get("etype")
                .and_then(Value::as_str)
                .ok_or_else(|| anyhow!("edge missing 'etype'"))?;
            let src = NodeId::parse_uuid(
                ev.get("src")
                    .and_then(Value::as_str)
                    .ok_or_else(|| anyhow!("edge missing 'src'"))?,
            )
            .context("invalid edge src")?;
            let dst = NodeId::parse_uuid(
                ev.get("dst")
                    .and_then(Value::as_str)
                    .ok_or_else(|| anyhow!("edge missing 'dst'"))?,
            )
            .context("invalid edge dst")?;
            let mut edge = Edge::new(EdgeId::new_v7(), etype, src, dst);
            if let Some(Value::Object(props)) = ev.get("props") {
                for (k, v) in props {
                    edge = edge.with_prop(k.clone(), json_to_ipld(v)?);
                }
            }
            tx.add_edge(&edge)?;
        }
    }

    let opts = mnem_core::repo::CommitOptions::new(agent_id.as_str(), message.as_str());
    // `task_id` is currently accepted for forward-compat but not persisted
    // onto the Operation / Commit. First-class `Commit.agent_id` /
    // `Commit.task_id` plumbing is tracked in ; when it lands, the
    // tool schema stays the same and the value starts surviving round-trips.
    let _ = &task_id;

    let new_repo = tx.commit_opts(opts)?;

    let mut out = String::new();
    out.push_str("mnem_commit: ok\n");
    out.push_str(&format!("  op_id:       {}\n", new_repo.op_id()));
    out.push_str(&format!(
        "  commit_cid:  {}\n",
        new_repo
            .view()
            .heads
            .first()
            .map_or_else(|| "<none>".to_string(), ToString::to_string)
    ));
    out.push_str(&format!("  nodes added: {}\n", created_nodes.len()));
    for (ntype, id) in &created_nodes {
        out.push_str(&format!("    - {ntype} {}\n", id.to_uuid_string()));
    }
    Ok(out)
}
