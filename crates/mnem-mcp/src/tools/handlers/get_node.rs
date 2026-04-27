//! Handler for the `mnem_get_node` MCP tool.
//!
//! Extracted from `tools.rs` in R3; body unchanged.

use super::super::ipld_preview;
use crate::server::Server;
use anyhow::{Context, Result, anyhow};
use mnem_core::codec::hash_to_cid;
use mnem_core::id::NodeId;
use serde_json::Value;

// ============================================================
// mnem_get_node
// ============================================================

pub(in crate::tools) fn get_node(server: &mut Server, args: Value) -> Result<String> {
    let id_str = args
        .get("id")
        .and_then(Value::as_str)
        .ok_or_else(|| anyhow!("missing 'id'"))?;
    let id = NodeId::parse_uuid(id_str).context("invalid node UUID")?;
    let repo = server.load_repo()?;
    let Some(node) = repo.lookup_node(&id)? else {
        return Ok(format!("mnem_get_node: no node found for id={id_str}\n"));
    };
    let content_size = node.content.as_ref().map_or(0, bytes::Bytes::len);

    let mut out = String::new();
    out.push_str(&format!("node {}\n", node.id.to_uuid_string()));
    out.push_str(&format!("  ntype:   {}\n", node.ntype));
    if let Some(summary) = &node.summary {
        out.push_str(&format!("  summary: {summary}\n"));
    }
    if !node.props.is_empty() {
        out.push_str("  props:\n");
        for (k, v) in &node.props {
            out.push_str(&format!("    {k}: {}\n", ipld_preview(v)));
        }
    }
    if content_size > 0 {
        out.push_str(&format!("  content: {content_size} bytes\n"));
    }
    // Embeddings live in the sidecar bucket keyed by NodeCid. We
    // probe under the model from `MNEM_EMBED_MODEL` (or the
    // fully-qualified `<provider>:<model>` string the writer used);
    // a missing or unset model means we skip the line entirely.
    // The sidecar API is keyed by exact model string so we cannot
    // enumerate without one.
    if let Ok(model) = std::env::var("MNEM_EMBED_MODEL") {
        let (_, node_cid) = hash_to_cid(&node)?;
        let has_embed = repo
            .embedding_for(&node_cid, &model)
            .map(|opt| opt.is_some())
            .unwrap_or(false);
        if has_embed {
            out.push_str("  embed:   present\n");
        }
    }
    Ok(out)
}
