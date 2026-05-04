//! Handler for `mnem_global_ingest` - ingest source files into the global anchor graph.
//!
//! Opens `~/.mnemglobal/.mnem/` and runs the standard ingest pipeline against it.
//! Unlike `mnem_ingest` (which operates on whatever repo the MCP server is pointed at),
//! this tool always targets the global graph regardless of server configuration.

use crate::server::Server;
use anyhow::Result;
use serde_json::Value;

pub(in crate::tools) fn global_ingest(server: &mut Server, args: Value) -> Result<String> {
    let global_data = super::global_dir().join(".mnem");
    if !global_data.is_dir() {
        return Ok(
            "mnem_global_ingest: global graph not found at ~/.mnemglobal/.mnem/. \
             Run `mnem integrate` then `mnem init` to create it.\n"
                .to_string(),
        );
    }
    let allow_labels = server.allow_labels;
    let repo = super::open_global_repo(server, &global_data)?;
    super::ingest::ingest_impl(repo, &global_data, allow_labels, args)
}
