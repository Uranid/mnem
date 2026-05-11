//! Handler for the `mnem_recent` MCP tool.
//!
//! Extracted from `tools.rs` in R3; body unchanged.

use crate::server::Server;
use anyhow::{Result, anyhow};
use mnem_core::codec::from_canonical_bytes;
use serde_json::Value;

// ============================================================
// mnem_recent
// ============================================================

pub(in crate::tools) fn recent(server: &mut Server, args: Value) -> Result<String> {
    let limit = args.get("limit").and_then(Value::as_u64).unwrap_or(10) as usize;
    let limit = limit.min(100);
    let repo = server.load_repo()?;
    let bs = server.stores()?.0;

    let mut out = String::new();
    out.push_str(&format!("mnem_recent (last {limit})\n"));

    let mut cur = repo.op_id().clone();
    let mut n = 0;
    loop {
        if n >= limit {
            break;
        }
        let bytes = bs.get(&cur)?.ok_or_else(|| anyhow!("op {cur} missing"))?;
        let op: mnem_core::objects::Operation = from_canonical_bytes(&bytes)?;
        let agent_line = match &op.agent_id {
            Some(a) => format!(" agent_id={a:?}"),
            None => String::new(),
        };
        let task_line = match &op.task_id {
            Some(t) => format!(" task_id={t:?}"),
            None => String::new(),
        };
        out.push_str(&format!(
            "  [{}] op_id={}\n       time={}us author={:?} desc={:?}{}{}\n",
            n, cur, op.time, op.author, op.description, agent_line, task_line,
        ));
        n += 1;
        if let Some(parent) = op.parents.first() {
            cur = parent.clone();
        } else {
            break;
        }
    }
    if n == 0 {
        out.push_str("  <none>\n");
    }
    Ok(out)
}
