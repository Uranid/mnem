//! Handler for the `mnem_commit` MCP tool.
//!
//! Extracted from `tools.rs` in R3; body unchanged.

use std::path::Path;

use crate::server::Server;
use anyhow::{Context, Result, anyhow};
use mnem_core::codec::json_to_ipld;
use mnem_core::id::{EdgeId, NodeId};
use mnem_core::objects::{Edge, Node};
use mnem_core::repo::ReadonlyRepo;
use serde_json::Value;

// ============================================================
// mnem_commit
// ============================================================

pub(in crate::tools) fn commit(server: &mut Server, args: Value) -> Result<String> {
    let repo_path = server.repo_path().to_path_buf();
    let allow_labels = server.allow_labels;
    let repo = server.load_repo()?;
    commit_impl(repo, &repo_path, allow_labels, args)
}

pub(super) fn commit_impl(
    repo: ReadonlyRepo,
    repo_path: &Path,
    allow_labels: bool,
    args: Value,
) -> Result<String> {
    #[cfg(not(feature = "summarize"))]
    let _ = repo_path;

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

    let mut tx = repo.start_transaction();

    // Open embedder once for the whole commit (provider failures are
    // non-fatal: nodes are committed without a vector and can be
    // backfilled with `mnem reindex`).
    #[cfg(feature = "summarize")]
    let opt_embedder = crate::tools::embed::resolve_embed_cfg(repo_path)
        .and_then(|pc| mnem_embed_providers::open(&pc).ok());

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
            let summary_text = nv
                .get("summary")
                .and_then(Value::as_str)
                .map(str::to_string);
            let mut node = Node::new(NodeId::new_v7(), ntype);
            if let Some(ref s) = summary_text {
                node = node.with_summary(s);
            }
            if let Some(Value::Object(props)) = nv.get("props") {
                for (k, v) in props {
                    node = node.with_prop(k.clone(), json_to_ipld(v)?);
                }
            }
            if let Some(content) = nv.get("content").and_then(Value::as_str) {
                node = node.with_content(bytes::Bytes::from(content.to_string().into_bytes()));
            }
            let node_cid = tx.add_node(&node)?;
            #[cfg(feature = "summarize")]
            if let (Some(embedder), Some(text)) = (&opt_embedder, &summary_text) {
                if let Ok(vec) = embedder.embed(text) {
                    let model = embedder.model().to_string();
                    let emb = mnem_embed_providers::to_embedding(&model, &vec);
                    let _ = tx.set_embedding(node_cid, model, emb);
                }
            }
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

    let mut opts = mnem_core::repo::CommitOptions::new(agent_id.as_str(), message.as_str());
    opts.agent_id = Some(agent_id.clone());
    opts.task_id = task_id;

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
