//! Handler for `mnem_global_retrieve` - semantic search on the global anchor graph.
//!
//! Opens `~/.mnemglobal/.mnem/` and runs the standard retriever pipeline against it.
//! Results are ranked by score. Unlike `mnem_retrieve` (which operates on whatever
//! repo the MCP server is pointed at), this tool always targets the global graph
//! regardless of server configuration.

use crate::server::Server;
use anyhow::Result;
use serde_json::Value;

pub(in crate::tools) fn global_retrieve(server: &mut Server, args: Value) -> Result<String> {
    let global_data = super::global_dir().join(".mnem");
    if !global_data.is_dir() {
        return Ok(
            "mnem_global_retrieve: global graph not found at ~/.mnemglobal/.mnem/. \
             Run `mnem integrate` then `mnem init` to create it.\n"
                .to_string(),
        );
    }
    let allow_labels = server.allow_labels;
    let repo = match super::open_global_repo(server, &global_data) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("(mnem_global_retrieve: cannot open global graph: {e})");
            return Ok(format!(
                "mnem_global_retrieve: error opening global graph: {e}\n"
            ));
        }
    };
    super::retrieve::retrieve_impl(repo, &global_data, allow_labels, args)
}
