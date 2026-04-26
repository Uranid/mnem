//! `mnem completions <shell>` - emit a shell completion script to
//! stdout.
//!
//! The shell parameter is any variant supported by `clap_complete`:
//! bash, zsh, fish, powershell, elvish. Pipe the output into the
//! appropriate config directory for your shell:
//!
//! ```text
//! # bash
//! mnem completions bash > ~/.local/share/bash-completion/completions/mnem
//!
//! # zsh  (first-time setup: `mkdir -p ~/.zsh/completions` and add it to $fpath)
//! mnem completions zsh > ~/.zsh/completions/_mnem
//!
//! # fish
//! mnem completions fish > ~/.config/fish/completions/mnem.fish
//!
//! # powershell (append to your $PROFILE)
//! mnem completions powershell >> $PROFILE
//! ```
//!
//! The generator reuses the same `Cli` derive that drives the main
//! binary, so `mnem --help` and the completion script can never drift.

use std::io::{self, Write};

use clap::CommandFactory;
use clap_complete::Shell;

use super::*;

/// Args for the `completions` subcommand.
#[derive(clap::Args, Debug)]
#[command(after_long_help = "\
Examples:
  mnem completions bash > ~/.local/share/bash-completion/completions/mnem
  mnem completions zsh > ~/.zsh/completions/_mnem
  mnem completions fish > ~/.config/fish/completions/mnem.fish
  mnem completions powershell >> $PROFILE
  mnem completions elvish > ~/.config/elvish/lib/mnem.elv
")]
pub(crate) struct Args {
    /// Target shell: bash | zsh | fish | powershell | elvish.
    #[arg(value_enum)]
    pub shell: Shell,
}

pub(crate) fn run(args: Args) -> Result<()> {
    // `clap_complete::generate` writes to a `Write` sink and uses the
    // main CLI's `Command` metadata so the script we emit matches the
    // binary byte-for-byte. Binary name is "mnem", matching
    // `#[command(name = "mnem")]` on the top-level `Cli` derive.
    //
    // We render into a `Vec<u8>` first so a broken pipe on stdout
    // (e.g. `mnem completions bash | head`) fails gracefully at the
    // copy-to-stdout step rather than panicking inside clap_complete's
    // `expect("failed to write completion file")`.
    let mut cmd = crate::Cli::command();
    let mut buf: Vec<u8> = Vec::with_capacity(64 * 1024);
    clap_complete::generate(args.shell, &mut cmd, "mnem", &mut buf);
    match io::stdout().write_all(&buf) {
        Ok(()) => Ok(()),
        // Broken pipe is a normal termination signal when the downstream
        // consumer (a pager, `head`, `tee` closed early) went away.
        // Swallow it; surfacing as an error only confuses scripts.
        Err(e) if e.kind() == io::ErrorKind::BrokenPipe => Ok(()),
        Err(e) => Err(e.into()),
    }
}
