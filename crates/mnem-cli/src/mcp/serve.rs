//! `mnem mcp` — MCP server entry point inside the unified `mnem` binary.
//!
//! After merge, `mnem mcp` replaces the standalone `mnem-mcp` binary.

use std::io::{self, BufRead, Write};

use anyhow::Result;

use crate::repo;

/// Typed alias so the main CLI crate can construct the MCP server
/// without a direct compile-time dependency cycle. The concrete type
/// is `mnem_mcp::Server`.
pub(crate) type McpServer = mnem_mcp::Server;

/// Serve mnem as a Model Context Protocol server over stdio.
///
/// Each line of stdin is a JSON-RPC request; each line of stdout is a
/// response. This is the wire format every MCP client expects.
#[derive(clap::Parser)]
pub(crate) struct ServeArgs {
    /// Repository directory (default: auto-detect via walk-up from cwd).
    #[arg(long, short = 'R')]
    repo: Option<std::path::PathBuf>,
}

pub(crate) fn run(args: ServeArgs) -> Result<()> {
    let repo_path = repo::locate_data_dir(args.repo.as_deref())?;
    let mut server = McpServer::new(repo_path);

    if !server.allow_labels {
        eprintln!(
            "mnem: labels DISABLED (MNEM_LABELS=0 or MNEM_BENCH=0); \
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
    }
    Ok(())
}
