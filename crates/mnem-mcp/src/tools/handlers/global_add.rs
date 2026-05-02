//! Handler for `mnem_global_add` - write nodes/edges directly to the global graph.
//!
//! Opens `~/.mnemglobal/.mnem/` and commits the supplied nodes/edges into it.
//! This is the write counterpart to `mnem_global_retrieve`: use it when an
//! entity or fact should live in the shared cross-repo graph rather than (or
//! in addition to) the current local repo.

use crate::server::Server;
use anyhow::{Result, anyhow};
use mnem_core::codec::json_to_ipld;
use mnem_core::id::{EdgeId, NodeId};
use mnem_core::objects::{Edge, Node};
use serde_json::Value;
use std::path::PathBuf;

pub(in crate::tools) fn global_add(server: &Server, args: Value) -> Result<String> {
    let global_data = dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".mnemglobal")
        .join(".mnem");

    if !global_data.is_dir() {
        return Ok(
            "mnem_global_add: global graph not found at ~/.mnemglobal/.mnem/. \
             Run `mnem integrate` then `mnem init` to create it.\n"
                .to_string(),
        );
    }

    let allow_labels = server.allow_labels;
    let agent_id = args
        .get("agent_id")
        .and_then(Value::as_str)
        .unwrap_or("mnem mcp")
        .to_string();
    let message = args
        .get("message")
        .and_then(Value::as_str)
        .unwrap_or("mnem_mcp global_add")
        .to_string();

    let repo = Server::open_repo_at(&global_data)?;
    let mut tx = repo.start_transaction();
    let mut created_nodes: Vec<(String, NodeId)> = Vec::new();

    if let Some(nodes) = args.get("nodes").and_then(Value::as_array) {
        for nv in nodes {
            let ntype = if allow_labels {
                nv.get("ntype")
                    .and_then(Value::as_str)
                    .unwrap_or(Node::DEFAULT_NTYPE)
            } else {
                Node::DEFAULT_NTYPE
            };
            let mut node = Node::new(NodeId::new_v7(), ntype);
            if let Some(summary) = nv.get("summary").and_then(Value::as_str) {
                node = node.with_summary(summary);
            }
            if let Some(props) = nv.get("props").and_then(Value::as_object) {
                for (k, v) in props {
                    node = node.with_prop(k.clone(), json_to_ipld(v)?);
                }
            }
            tx.add_node(&node)?;
            created_nodes.push((ntype.to_string(), node.id));
        }
    }

    let mut edge_count = 0usize;
    if let Some(edges) = args.get("edges").and_then(Value::as_array) {
        for ev in edges {
            let src_str = ev
                .get("src")
                .and_then(Value::as_str)
                .ok_or_else(|| anyhow!("edge missing 'src'"))?;
            let dst_str = ev
                .get("dst")
                .and_then(Value::as_str)
                .ok_or_else(|| anyhow!("edge missing 'dst'"))?;
            let predicate = ev
                .get("predicate")
                .and_then(Value::as_str)
                .ok_or_else(|| anyhow!("edge missing 'predicate'"))?;
            let src = NodeId::parse_uuid(src_str).map_err(|e| anyhow!("edge src: {e}"))?;
            let dst = NodeId::parse_uuid(dst_str).map_err(|e| anyhow!("edge dst: {e}"))?;
            let edge = Edge::new(EdgeId::new_v7(), predicate, src, dst);
            tx.add_edge(&edge)?;
            edge_count += 1;
        }
    }

    if created_nodes.is_empty() && edge_count == 0 {
        return Ok(
            "mnem_global_add: nothing to commit (supply 'nodes' and/or 'edges')\n".to_string(),
        );
    }

    let opts = mnem_core::repo::CommitOptions::new(agent_id.as_str(), message.as_str());
    let new_repo = tx.commit_opts(opts)?;

    let mut out = String::new();
    out.push_str("mnem_global_add: ok\n");
    out.push_str("  target: ~/.mnemglobal/.mnem/\n");
    out.push_str(&format!("  op_id:  {}\n", new_repo.op_id()));
    if !created_nodes.is_empty() {
        out.push_str(&format!("  nodes ({}):\n", created_nodes.len()));
        for (ntype, id) in &created_nodes {
            out.push_str(&format!("    - {ntype} {}\n", id.to_uuid_string()));
        }
    }
    if edge_count > 0 {
        out.push_str(&format!("  edges: {edge_count}\n"));
    }
    Ok(out)
}
