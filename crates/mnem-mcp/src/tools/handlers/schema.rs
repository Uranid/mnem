//! Handler for the `mnem_schema` MCP tool.
//!
//! Extracted from `tools.rs` in R3; body unchanged.

use anyhow::Result;

use super::super::index_set;
use crate::server::Server;

// ============================================================
// mnem_schema
// ============================================================

pub(in crate::tools) fn schema(server: &mut Server) -> Result<String> {
    let repo = server.load_repo()?;
    let Some(set) = index_set(server, &repo)? else {
        return Ok("schema: <no IndexSet on current commit>\n".to_string());
    };

    let mut out = String::new();
    out.push_str("mnem schema (from current IndexSet)\n");
    out.push_str("  node labels:\n");
    if set.nodes_by_label.is_empty() {
        out.push_str("    <none>\n");
    } else {
        for label in set.nodes_by_label.keys() {
            let props: Vec<&String> = set
                .nodes_by_prop
                .get(label)
                .map(|m| m.keys().collect::<Vec<_>>())
                .unwrap_or_default();
            out.push_str(&format!(
                "    - {label} [indexed props: {}]\n",
                if props.is_empty() {
                    "none".into()
                } else {
                    props
                        .iter()
                        .map(|s| s.as_str())
                        .collect::<Vec<_>>()
                        .join(", ")
                }
            ));
        }
    }
    out.push_str("  outgoing-adjacency index: ");
    out.push_str(if set.outgoing.is_some() {
        "present\n"
    } else {
        "absent\n"
    });
    out.push_str("  incoming-adjacency index: ");
    out.push_str(if set.incoming.is_some() {
        "present\n"
    } else {
        "absent\n"
    });
    Ok(out)
}
