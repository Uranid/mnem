//! Print the MCP tool surface a `mnem-mcp` server advertises.
//!
//! Useful for integration builders who want to know, at build time,
//! which tools their agent will see without starting a server and
//! sending a `tools/list` request. Prints the default (non-bench)
//! list and the `MNEM_BENCH`-gated list side-by-side so the
//! difference is obvious.
//!
//! See also:
//! - `docs/guide/mcp.md` - end-user MCP walkthrough.
//! - - why `label` / `ntype` are gated behind `MNEM_BENCH`.
//!
//! Run:
//! ```console
//! cargo run --example tool_list -p mnem-mcp
//! ```

fn main() {
    println!("mnem-mcp tool surface");
    println!("  protocol version: {}", mnem_mcp::MCP_PROTOCOL_VERSION);
    println!();

    println!("default (MNEM_BENCH unset, allow_labels=false):");
    for name in mnem_mcp::tool_names(false) {
        println!("  - {name}");
    }
    println!();

    println!("with MNEM_BENCH set (allow_labels=true):");
    for name in mnem_mcp::tool_names(true) {
        println!("  - {name}");
    }
    println!();

    // Invariant worth advertising: the gated list is a strict superset
    // of the default list. Useful for CI regressions and for clients
    // that want to build a stable registry against the default surface
    // and then light up extra tools when running against a bench-mode
    // server.
    let default: std::collections::BTreeSet<_> = mnem_mcp::tool_names(false).into_iter().collect();
    let gated: std::collections::BTreeSet<_> = mnem_mcp::tool_names(true).into_iter().collect();
    assert!(
        default.is_subset(&gated),
        "bench-gated tool list must be a superset of the default list"
    );
    println!("invariant: default tools are a subset of bench-mode tools: OK");
}
