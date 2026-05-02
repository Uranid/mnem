//! Per-tool handler functions. Extracted from `tools.rs` in R3.
//!
//! Each submodule holds a single `pub(super) fn <name>` that
//! `tools::dispatch` forwards to.

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
