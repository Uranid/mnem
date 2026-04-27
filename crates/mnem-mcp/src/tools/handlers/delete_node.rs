//! Handler for the `mnem_delete_node` MCP tool.
//!
//! Extracted from `tools.rs` in R3; body unchanged.

use crate::server::Server;
use anyhow::{Context, Result, anyhow};
use mnem_core::id::NodeId;
use serde_json::Value;

// ============================================================
// mnem_delete_node
// ============================================================

pub(in crate::tools) fn delete_node(server: &mut Server, args: Value) -> Result<String> {
    let id_str = args
        .get("id")
        .and_then(Value::as_str)
        .ok_or_else(|| anyhow!("missing 'id'"))?;
    let id = NodeId::parse_uuid(id_str).context("invalid node UUID")?;
    let agent_id = args
        .get("agent_id")
        .and_then(Value::as_str)
        .ok_or_else(|| anyhow!("missing 'agent_id'"))?
        .to_string();
    let message = args
        .get("message")
        .and_then(Value::as_str)
        .unwrap_or("mnem_mcp delete")
        .to_string();

    let repo = server.load_repo()?;
    let existed = repo.lookup_node(&id)?.is_some();

    let mut tx = repo.start_transaction();
    tx.remove_node(id);
    let opts = mnem_core::repo::CommitOptions::new(agent_id.as_str(), message.as_str());
    let new_repo = tx.commit_opts(opts)?;

    let mut out = String::new();
    out.push_str("mnem_delete_node: ok\n");
    out.push_str(&format!("  id:         {id_str}\n"));
    out.push_str(&format!("  existed:    {existed}\n"));
    out.push_str(&format!("  op_id:      {}\n", new_repo.op_id()));
    Ok(out)
}
