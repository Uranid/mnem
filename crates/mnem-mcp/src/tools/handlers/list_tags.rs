//! Handler for the `mnem_list_tags` MCP tool.

use anyhow::Result;
use mnem_core::TAGS_PREFIX;
use serde_json::Value;

use crate::server::Server;

// ============================================================
// mnem_list_tags
// ============================================================

pub(in crate::tools) fn list_tags(server: &mut Server, args: Value) -> Result<String> {
    let as_json = args.get("json").and_then(Value::as_bool).unwrap_or(false);

    let repo = server.load_repo()?;
    let view = repo.view();

    // Collect all refs/tags/* entries.
    let tags: Vec<(&str, String)> = view
        .refs
        .iter()
        .filter_map(|(name, target)| {
            let short = name.strip_prefix(TAGS_PREFIX)?;
            let cid_str = match target {
                mnem_core::objects::RefTarget::Normal { target } => target.to_string(),
                mnem_core::objects::RefTarget::Conflicted { .. } => String::new(),
            };
            Some((short, cid_str))
        })
        .collect();

    if as_json {
        // JSON output: same shape as HTTP GET /v1/tags.
        let json_tags: Vec<serde_json::Value> = tags
            .iter()
            .map(|(name, target)| serde_json::json!({"name": name, "target": target}))
            .collect();
        let out = serde_json::json!({
            "schema": "mnem.v1.tags",
            "tags": json_tags,
        });
        return Ok(serde_json::to_string_pretty(&out)?);
    }

    // Plain-text output.
    let mut out = String::new();
    for (name, target) in &tags {
        out.push_str(&format!("{name}  ->  {target}\n"));
    }
    out.push_str(&format!("({} tag(s))\n", tags.len()));
    Ok(out)
}
