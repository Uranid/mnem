#![forbid(unsafe_code)]
#![deny(missing_docs)]
//! `mnem-mcp` - Model Context Protocol server for mnem.
//!
//! Speaks JSON-RPC 2.0 over stdio. Each line of stdin is one request;
//! each response is one line of stdout. This is the wire format every
//! MCP client (Claude Desktop, Cursor, Windsurf, Claude Code, custom
//! clients) expects for stdio servers.
//!
//! Usage:
//!
//! ```bash
//! mnem-mcp --repo ./my-mnem-repo
//! # or
//! MNEM_REPO=./my-mnem-repo mnem-mcp
//! # default when unset: ./.mnem in the current directory
//! ```
//!
//! Configure Claude Desktop / Cursor / Windsurf to launch this binary
//! with stdio transport. The server opens (or initialises) the repo
//! at the path given on first tool call.
//!
//! AI-native design choices baked in from day one:
//! - Every tool response carries `_meta: { bytes, latency_micros,
//!   tokens_estimate }` so the caller can reason about cost.
//! - Writes are attributed: `agent_id` and `task_id` propagate to the
//!   `Commit` + `Operation` metadata so provenance is queryable.
//! - Introspection tools (`mnem_stats`, `mnem_schema`, `mnem_recent`)
//!   are first-class, not afterthoughts - LLMs need to explore before
//!   they query.

use std::io::{self, BufRead, Write};
use std::path::PathBuf;

use mnem_mcp::Server;

fn main() -> anyhow::Result<()> {
    let repo_path = parse_repo_path();

    let mut server = Server::new(repo_path);

    // Audit fix G3 (2026-04-25): caller-supplied `label`/`ntype`
    // fields are honoured by default. The previous behaviour silently
    // coerced every node to `Node::DEFAULT_NTYPE` unless `MNEM_BENCH=1`
    // was set, which broke typed knowledge graphs for every customer
    // who did not know about the gate. Operators who explicitly want
    // the old single-tenant flat-graph behaviour set `MNEM_LABELS=0`
    // (or the legacy `MNEM_BENCH=0`).
    if !server.allow_labels {
        eprintln!(
            "mnem-mcp: labels DISABLED (MNEM_LABELS=0 or MNEM_BENCH=0); \
             caller-supplied `label`/`ntype` will be coerced to Node::DEFAULT_NTYPE."
        );
    }

    let stdin = io::stdin();
    let stdout = io::stdout();
    let mut stdout = stdout.lock();

    for line in stdin.lock().lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        if let Some(response) = server.handle_line(&line) {
            writeln!(stdout, "{response}")?;
            stdout.flush()?;
        }
        // Notifications (no id) produce None; we stay silent.
    }
    Ok(())
}

/// Resolve the repo path from `--repo PATH`, `MNEM_REPO` env var, or
/// the default `./.mnem`. Simple hand-roll; we have no complex flag
/// surface here and don't want a clap dependency on the MCP binary.
fn parse_repo_path() -> PathBuf {
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--repo" => {
                if let Some(p) = args.next() {
                    return PathBuf::from(p);
                }
            }
            "--version" | "-V" => {
                println!("mnem-mcp {}", env!("CARGO_PKG_VERSION"));
                std::process::exit(0);
            }
            "--help" | "-h" => {
                print_help();
                std::process::exit(0);
            }
            _ => {}
        }
    }
    std::env::var("MNEM_REPO")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(".mnem"))
}

fn print_help() {
    eprintln!(
        "mnem-mcp {}\nModel Context Protocol server for mnem\n\n\
         USAGE:\n    mnem-mcp [--repo PATH]\n\n\
         Speaks JSON-RPC 2.0 over stdio. Intended to be launched by an\n\
         MCP client (Claude Desktop, Cursor, Windsurf, Claude Code, ...).\n\n\
         OPTIONS:\n    \
         --repo PATH    Repository directory (default: .mnem, env: MNEM_REPO)\n    \
         -V, --version  Print version and exit\n    \
         -h, --help     Print this help and exit\n",
        env!("CARGO_PKG_VERSION")
    );
}
