//! Handler for `mnem_global_add` - write nodes/edges directly to the global graph.
//!
//! Opens `~/.mnemglobal/.mnem/` and commits the supplied nodes/edges into it.
//! This is the write counterpart to `mnem_global_retrieve`: use it when an
//! entity or fact should live in the shared cross-repo graph rather than (or
//! in addition to) the current local repo.

use crate::server::Server;
use anyhow::Result;
use serde_json::Value;

pub(in crate::tools) fn global_add(server: &mut Server, args: Value) -> Result<String> {
    let global_data = super::global_dir().join(".mnem");
    if !global_data.is_dir() {
        return Ok(
            "mnem_global_add: global graph not found at ~/.mnemglobal/.mnem/. \
             Run `mnem integrate` then `mnem init` to create it.\n"
                .to_string(),
        );
    }
    let allow_labels = server.allow_labels;
    let repo = super::open_global_repo(server, &global_data)?;
    super::commit::commit_impl(repo, &global_data, allow_labels, args)
}
