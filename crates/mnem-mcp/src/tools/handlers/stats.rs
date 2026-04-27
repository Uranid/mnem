//! Handler for the `mnem_stats` MCP tool.
//!
//! Extracted from `tools.rs` in R3; body unchanged.

use anyhow::Result;

use super::super::{index_set, summarize_refs};
use crate::server::Server;

// ============================================================
// mnem_stats
// ============================================================

pub(in crate::tools) fn stats(server: &mut Server) -> Result<String> {
    let repo = server.load_repo()?;
    let op_id = repo.op_id().to_string();
    let commit_cid = repo
        .view()
        .heads
        .first()
        .map_or_else(|| "<none>".to_string(), ToString::to_string);
    let refs = &repo.view().refs;

    let label_count = index_set(server, &repo)?.map_or(0, |s| s.nodes_by_label.len());

    let mut out = String::new();
    out.push_str("mnem repository status\n");
    out.push_str(&format!("  op_id:        {op_id}\n"));
    out.push_str(&format!("  head_commit:  {commit_cid}\n"));
    out.push_str(&format!(
        "  refs:         {} ({})\n",
        refs.len(),
        summarize_refs(refs)
    ));
    if label_count == 0 {
        out.push_str("  labels:       <none or IndexSet missing>\n");
    } else {
        out.push_str(&format!(
            "  labels:       {label_count} distinct (use mnem_schema)\n"
        ));
    }
    Ok(out)
}
