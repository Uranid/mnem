//! Handler for the `mnem_tombstone_node` MCP tool.
//!
//! Extracted from `tools.rs` in R3; body unchanged.

use crate::server::Server;
use anyhow::{Context, Result, anyhow, bail};
use mnem_core::id::NodeId;
use serde_json::Value;

// ============================================================
// mnem_tombstone_node
// ============================================================

pub(in crate::tools) fn tombstone_node(server: &mut Server, args: Value) -> Result<String> {
    let id_str = args
        .get("id")
        .and_then(Value::as_str)
        .ok_or_else(|| anyhow!("missing 'id'"))?;
    let id = NodeId::parse_uuid(id_str).context("invalid node UUID")?;
    let reason = args
        .get("reason")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string();
    let agent_id = args
        .get("agent_id")
        .and_then(Value::as_str)
        .ok_or_else(|| anyhow!("missing 'agent_id'"))?
        .to_string();
    let message = args
        .get("message")
        .and_then(Value::as_str)
        .unwrap_or("mnem_mcp tombstone")
        .to_string();

    let repo = server.load_repo()?;
    // 404-equivalent: the node must exist on the current head. MCP
    // has no HTTP status codes; return an Err that the tool-call
    // dispatcher converts to an isError=true MCP reply.
    if repo.lookup_node(&id)?.is_none() {
        bail!("no node with id={id_str}");
    }
    // 409-equivalent: refuse to re-tombstone via MCP. The in-process
    // API remains idempotent; the user-facing tool makes the second
    // call observable as an error so agents don't silently re-revoke.
    if repo.is_tombstoned(&id) {
        bail!("node {id_str} is already tombstoned");
    }

    let mut tx = repo.start_transaction();
    tx.tombstone_node(id, reason.clone())?;
    let opts = mnem_core::repo::CommitOptions::new(agent_id.as_str(), message.as_str());
    let new_repo = tx.commit_opts(opts)?;

    let mut out = String::new();
    out.push_str("mnem_tombstone_node: ok\n");
    out.push_str(&format!("  id:      {id_str}\n"));
    if !reason.is_empty() {
        out.push_str(&format!("  reason:  {reason}\n"));
    }
    out.push_str(&format!("  op_id:   {}\n", new_repo.op_id()));
    Ok(out)
}
