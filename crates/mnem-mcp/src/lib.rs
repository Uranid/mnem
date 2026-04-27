//! Library half of the `mnem-mcp` binary.
//!
//! Exposes the JSON-RPC 2.0 dispatch used by the stdio entry point so
//! downstream embedders (HTTP transports, Node/Python wrappers, test
//! harnesses) can link the MCP tool surface without re-implementing it.
//!
//! The wire-level detail and tool implementations live in private
//! submodules; this crate surfaces a stable, narrow API:
//!
//! - [`Server`] - parsed-line dispatcher; `handle_line(&str) -> Option<String>`
//!   consumes one JSON-RPC request and returns the response body.
//! - [`tool_names`] - returns the ordered list of tool names the server
//!   will advertise in `tools/list` responses.
//! - [`MCP_PROTOCOL_VERSION`] - the protocol version this crate
//!   implements.
//!
//! The binary crate (`src/main.rs`) is a thin stdio wrapper around
//! [`Server`]; everything dispatch-related is reachable from here.

#![forbid(unsafe_code)]
#![deny(missing_docs)]

pub mod protocol;
pub mod server;
pub mod tools;

pub use protocol::MCP_PROTOCOL_VERSION;
pub use server::Server;

/// Return the ordered list of tool names the server would advertise in
/// `tools/list` for the given `allow_labels` gate. Useful in tests to
/// assert the public tool surface stays stable, and in wrappers that
/// want to precompute a client-side registry before opening a repo.
#[must_use]
pub fn tool_names(allow_labels: bool) -> Vec<&'static str> {
    tools::all_tools(allow_labels)
        .into_iter()
        .map(|t| t.name)
        .collect()
}
