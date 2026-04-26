//! Stubs for verbs that are part of the Git-verb spine but
//! whose implementation is deferred past Q2.
//!
//! Each stub:
//!
//! - is advertised in `mnem --help` so `mnem <TAB>` completion shows
//!   the full vocabulary
//! - fails with exit code 78 (EX_CONFIG per BSD sysexits - "something
//!   the user did was wrong, not a transient I/O error")
//! - points at docs/ROADMAP.md so the next step is obvious
//!
//! The stubs deliberately accept the arguments the real
//! implementation will - clap catches unknown flags, and a user who
//! has already built a `mnem pull origin main` habit gets the same
//! parse error they'll get post-PR 3 if they pass something wrong.

use super::*;

/// Uniform "this verb is not wired up yet" message.
pub(crate) const NOT_YET_BLURB: &str = "not yet implemented. \
    See docs/ROADMAP.md for the roadmap \
    and current status. Use `mnem export` + `mnem import` + file transport \
    as the interim workaround.";

/// Exit code 78 = EX_CONFIG. Surfaced through the standard error
/// path in main.rs; this module just returns `Err` carrying the
/// message so the caller can format it consistently.
pub(crate) fn ex_config<T>(verb: &str) -> Result<T> {
    Err(anyhow!("mnem {verb}: {NOT_YET_BLURB}"))
}

// Real implementations of `fetch` / `push` / `pull` live in
// `commands::fetch`, `commands::push`, `commands::pull`.
//
// `mnem merge` is wired through `commands::merge`.

#[derive(clap::Args, Debug)]
#[command(after_long_help = "\
Deferred to a future release.
")]
pub(crate) struct RevertArgs {
    /// Commit CID to invert.
    pub commit: String,
}

pub(crate) fn run_revert(_o: Option<&Path>, _a: RevertArgs) -> Result<()> {
    ex_config("revert")
}

#[derive(clap::Args, Debug)]
#[command(after_long_help = "\
Deferred to a future release. Walks the DAG and re-hashes every block.
")]
pub(crate) struct FsckArgs;

pub(crate) fn run_fsck(_o: Option<&Path>, _a: FsckArgs) -> Result<()> {
    ex_config("fsck")
}

#[derive(clap::Args, Debug)]
#[command(after_long_help = "\
Deferred to a future release. Drops unreferenced blocks after a reachability walk.
")]
pub(crate) struct GcArgs;

pub(crate) fn run_gc(_o: Option<&Path>, _a: GcArgs) -> Result<()> {
    ex_config("gc")
}
