//! Per-tool handler functions. Extracted from `tools.rs` in R3.
//!
//! Each submodule holds a single `pub(super) fn <name>` that
//! `tools::dispatch` forwards to.

use crate::server::Server;

/// Returns the global graph parent directory, honouring `MNEM_GLOBAL_DIR`
/// when set (e.g. to bridge WSL ↔ Windows home paths).
pub(super) fn global_dir() -> std::path::PathBuf {
    if let Ok(dir) = std::env::var("MNEM_GLOBAL_DIR") {
        return std::path::PathBuf::from(dir);
    }
    dirs::home_dir()
        .unwrap_or_else(|| std::path::PathBuf::from("."))
        .join(".mnemglobal")
}

/// Open the global graph, reusing the server's cached connection when
/// `server.repo_path() == global_data` to avoid redb's "Database already open" error.
pub(super) fn open_global_repo(
    server: &mut Server,
    global_data: &std::path::Path,
) -> anyhow::Result<mnem_core::repo::ReadonlyRepo> {
    if server.repo_path() == global_data {
        server.load_repo()
    } else {
        Server::open_repo_at(global_data)
    }
}

pub(super) mod commit;
pub(super) mod commit_relation;
#[cfg(feature = "summarize")]
pub(super) mod community_summarize;
pub(super) mod delete_node;
pub(super) mod get_node;
pub(super) mod global_add;
pub(super) mod global_ingest;
pub(super) mod global_retrieve;
pub(super) mod ingest;
pub(super) mod list_nodes;
pub(super) mod recent;
pub(super) mod resolve_or_create;
pub(super) mod retrieve;
pub(super) mod schema;
pub(super) mod search;
pub(super) mod stats;
pub(super) mod tombstone_node;
pub(super) mod traverse;
pub(super) mod vector_search;
