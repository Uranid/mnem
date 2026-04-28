#![forbid(unsafe_code)]
#![deny(missing_docs)]
//! `mnem` - the Git-shaped porcelain for mnem.
//!
//! Minimum-viable CLI for the adoption surface: init / status / log /
//! show / add / query / diff / ref / config / stats. The Git-verb
//! spine ships incrementally: `branch`, `blame`, `cat-file`, `remote`,
//! `clone file://`, `fetch`, `push`, and `pull` are wired (with
//! wire verbs over HTTP). `merge` / `revert` / `fsck` / `gc` remain
//! advertised in `mnem --help` but fail with EX_CONFIG until a later
//! revision.

use std::path::PathBuf;

use clap::{Parser, Subcommand};

mod commands;
mod config;
mod doctor;
mod integrate;
mod repo;
mod wizard;

#[derive(Parser)]
#[command(
    name = "mnem",
    version,
    about = "git for knowledge graphs - versioned, content-addressed, embeddable.",
    long_about = None,
    propagate_version = true
)]
pub(crate) struct Cli {
    /// Path to the repository directory (the directory that contains
    /// `.mnem/`). Defaults to walking up from the current directory,
    /// like `git` does.
    #[arg(long, short = 'R', global = true)]
    repo: Option<PathBuf>,

    /// Optional so bare `mnem` drops into the first-run wizard (or
    /// `mnem status` for returning users) instead of printing help.
    /// `mnem --help` still prints help because clap captures that
    /// before we reach `main`.
    #[command(subcommand)]
    cmd: Option<Cmd>,
}

#[derive(Subcommand)]
enum Cmd {
    /// Initialize a new mnem repository (creates `.mnem/repo.redb`).
    Init(commands::init::Args),
    /// Print current op-head, head commit, ref summary, and label counts.
    #[command(after_long_help = "\
Examples:
  mnem status                    # current op + head commit + ref count
  mnem -R ~/notes status         # explicit repo path
  mnem status && mnem retrieve \"query\"
")]
    Status,
    /// Walk the op-log backwards from the current head.
    Log(commands::log::Args),
    /// Show the full detail of one operation.
    Show(commands::show::Args),
    /// Add a node or edge (commits immediately).
    #[command(
        subcommand,
        after_long_help = "\
Examples:
  mnem add node -s \"Alice lives in Berlin\"
  mnem add node --label Person --prop name=Alice --prop city=Berlin \\
    -s \"Alice is a climber\"
  mnem add edge --from <src-uuid> --to <dst-uuid> --label knows

See `mnem add node --help` / `mnem add edge --help` for full options.
"
    )]
    Add(commands::add::AddCmd),
    /// Query the graph via property filters + optional traversal.
    Query(commands::query::Args),
    /// Agent-facing retrieval: property filters + cosine vector
    /// (requires an embedder when a text query is set). When a
    /// learned-sparse provider is configured (`[sparse]` in
    /// config.toml), results are fused with the sparse lane via
    /// min-max convex combination (or RRF via `--fusion rrf`). There
    /// is no separate `--sparse` flag: sparse participation is
    /// driven by config.
    Retrieve(commands::retrieve::Args),
    /// Backfill embeddings for nodes that don't have one. Needs
    /// `embed.provider` + `embed.model` configured (see `mnem config`).
    Embed(commands::embed_cmd::Args),
    /// Retro-embed nodes that lack a vector. Like `mnem embed` but
    /// adds `--since <commit>` and a `label + props` fallback for
    /// nodes without a `--summary`. Promoted as the recovery path
    /// when `mnem add node` warned that the embedder was unreachable
    /// (audit-2026-04-25 C7-5).
    #[command(after_long_help = "\
Examples:
  mnem reindex                       # embed every node missing a vector
  mnem reindex --label Person        # only nodes of this label
  mnem reindex --since <commit>      # only nodes added since <commit>
  mnem reindex --force               # re-embed even already-embedded nodes
  mnem reindex --dry-run             # report count without changing anything
")]
    Reindex(commands::reindex::Args),
    /// Embedder-manifest tooling. Currently exposes `audit`, the
    /// CI-enforceable check that every provider declares a noise
    /// floor (Gap 15).
    #[command(subcommand)]
    Embedder(commands::embed_cmd::EmbedderCmd),
    /// Report which refs and how many nodes/edges differ between two ops.
    Diff(commands::diff::Args),
    /// Manage named refs (branches, tags, arbitrary labels).
    #[command(
        subcommand,
        after_long_help = "\
Examples:
  mnem ref list                                 # every ref in the current view
  mnem ref set refs/heads/main <commit-cid>     # point `main` at a commit
  mnem ref delete refs/heads/scratch            # remove a ref

Refs are just named pointers to commit CIDs. mnem does not distinguish
branches from tags at the data layer; the convention `refs/heads/*`
for branches and `refs/tags/*` for tags follows git.
"
    )]
    Ref(commands::refs::RefCmd),
    /// Get or set repository configuration (`user.name`, `user.email`, ...).
    Config(commands::cfg_cmd::Args),
    /// Short one-line stats useful in prompts and shell scripts.
    #[command(after_long_help = "\
Examples:
  mnem stats                     # one-line summary (nodes/edges/refs)
  mnem stats | tee prompt.txt    # append to an LLM system prompt
  watch -n 5 mnem stats          # live tail while ingesting
")]
    Stats,
    /// Export the subtree reachable from a ref / CID to a CAR v1
    /// archive. The file can be shipped over any channel (email, USB,
    /// SSH, S3) and imported on the other side with `mnem import`.
    Export(commands::export::Args),
    /// Import a CAR v1 archive into the current repository. Every
    /// block is CID-verified before being stored.
    Import(commands::import::Args),
    /// Ingest external source files (Markdown / text / PDF / chat JSON)
    /// into the current repository as a Doc + Chunk + Entity subgraph.
    #[command(after_long_help = "\
Examples:
  mnem ingest notes.md                           # single-file markdown
  mnem ingest --chunker recursive --max-tokens 1024 book.pdf
  mnem ingest --recursive docs/                  # walk a directory
")]
    Ingest(commands::ingest::Args),
    /// Manage `refs/heads/<name>` pointers (git analog: `git branch`).
    #[command(
        subcommand,
        after_long_help = "\
Examples:
  mnem branch list                          # every refs/heads/*
  mnem branch create feature/oauth          # at the current head
  mnem branch create hotfix --from <cid>    # at a specific commit
  mnem branch delete old-experiment
"
    )]
    Branch(commands::branch::BranchCmd),
    /// Walk the incoming-edge index for a node and report who points
    /// at it (git analog: `git blame`, but coarser - ).
    #[command(after_long_help = "\
Examples:
  mnem blame <node-uuid>
  mnem blame <node-uuid> --etype authored
")]
    Blame(commands::blame::Args),
    /// Emit the raw bytes of a CID (binary-safe) or a decoded JSON
    /// preview (`--json`). Git analog: `git cat-file`.
    #[command(
        name = "cat-file",
        after_long_help = "\
Examples:
  mnem cat-file <cid>                       # raw DAG-CBOR bytes
  mnem cat-file <cid> --json | jq .         # pretty JSON
"
    )]
    CatFile(commands::cat_file::Args),
    /// Manage `[remote.<name>]` entries in `.mnem/config.toml`. Pure
    /// local config-file ops; no network in 0.3.
    #[command(
        subcommand,
        after_long_help = "\
Examples:
  mnem remote add origin file:///tmp/alice.car
  mnem remote list
  mnem remote show origin
  mnem remote remove origin
"
    )]
    Remote(commands::remote::RemoteCmd),
    /// Clone a mnem repo from an archive. 0.3 supports `file://` URLs
    /// and bare `.car` paths; remote schemes land in PR 3.
    #[command(after_long_help = "\
Examples:
  mnem clone file:///tmp/alice.car /tmp/mirror
  mnem clone ./alice.car
")]
    Clone(commands::clone::Args),
    /// Fetch new blocks from a remote (`refs/remotes/<name>/*` tracking refs).
    #[command(after_long_help = "\
Examples:
  mnem fetch                         # fetch from `origin`
  mnem fetch upstream                # explicit remote name

Authentication (when the remote requires it):
  MNEM_REMOTE_ORIGIN_TOKEN=... mnem fetch origin
  MNEM_HTTP_PUSH_TOKEN=...    mnem fetch origin  # fallback
")]
    Fetch(commands::fetch::Args),
    /// Push commits + blocks to a remote and advance its named ref.
    #[command(after_long_help = "\
Examples:
  mnem push                          # push HEAD to origin/main
  mnem push origin main              # explicit remote + branch

Authentication:
  MNEM_REMOTE_ORIGIN_TOKEN=... mnem push origin main
")]
    Push(commands::push::Args),
    /// Fast-forward pull from a remote into the local branch.
    #[command(after_long_help = "\
Examples:
  mnem pull                          # ff origin/main into local main
  mnem pull origin main

Fast-forward only. Use `mnem merge <remote>/<branch>` (B4) for 3-way merges.
")]
    Pull(commands::pull::Args),
    /// 3-way merge between branches. LCA + structured conflict
    /// detector + executor with `ours`/`theirs`/`manual` strategies.
    #[command(after_long_help = "\
Examples:
  mnem merge feature                    # 3-way merge `refs/heads/feature` into HEAD
  mnem merge feature --strategy=ours    # pick left / current side on conflict
  mnem merge feature --strategy=theirs  # pick right / incoming side on conflict
  mnem merge feature --dry-run          # preview outcome, persist nothing
  mnem merge --continue                 # finish an in-progress merge after manual edits
  mnem merge --abort                    # cancel an in-progress merge
")]
    Merge(commands::merge::Args),
    /// Revert a commit. Not implemented in 0.3
    Revert(commands::deferred::RevertArgs),
    /// Check the object DAG for corruption. Not implemented in 0.3
    Fsck(commands::deferred::FsckArgs),
    /// Garbage-collect unreferenced blocks. Not implemented in 0.3
    Gc(commands::deferred::GcArgs),
    /// Wire mnem into every detected MCP agent host (Claude Desktop,
    /// Cursor, Continue, Zed) with atomic backups of the originals.
    Integrate(integrate::Args),
    /// Non-mutating health check: binaries, repo, config, embedder,
    /// wired hosts. Exits 1 if any check fails.
    Doctor(doctor::Args),
    /// Emit a shell completion script for bash / zsh / fish /
    /// powershell / elvish. Pipe the output into your shell's
    /// completion directory.
    #[command(after_long_help = "\
Examples:
  mnem completions bash > ~/.local/share/bash-completion/completions/mnem
  mnem completions zsh > ~/.zsh/completions/_mnem
  mnem completions fish > ~/.config/fish/completions/mnem.fish
  mnem completions powershell >> $PROFILE
  mnem completions elvish > ~/.config/elvish/lib/mnem.elv
")]
    Completions(commands::completions::Args),
    /// Run mnem-bench: download datasets, run scorers, emit
    /// RESULTS.md. 0.1.0 ships LongMemEval, LoCoMo, ConvoMem,
    /// MemBench (simple-roles + highlevel-movie), and
    /// LongMemEval-hybrid-v4 against the in-process mnem adapter.
    /// mem0 / MemPalace adapters, CPU-parallel mode, and
    /// Docker-compose mode are stubbed and print a "coming 0.2.0"
    /// message at runtime.
    #[command(after_long_help = "\
Examples:
  mnem bench                                          # interactive setup
  mnem bench list --pretty                            # list all benches as JSON
  mnem bench fetch longmemeval                        # download a dataset
  mnem bench run --benches longmemeval --with mnem \\
    --mode cpu-local --out ./bench-out --top-k 10 --limit 5 --non-interactive
  mnem bench results ./bench-out                      # re-render RESULTS.md
")]
    Bench(commands::bench::BenchArgs),
}

fn main() {
    let cli = Cli::parse();
    // Track whether the failure was an deferred-stub so the
    // caller gets the BSD-sysexits EX_CONFIG exit code instead of the
    // generic 1. We distinguish by pattern-matching the verb BEFORE
    // dispatch, not by parsing the error string, so future refactors
    // of `deferred::ex_config` can't drift the exit code.
    let is_deferred = matches!(&cli.cmd, Some(Cmd::Revert(_) | Cmd::Fsck(_) | Cmd::Gc(_)));
    let result = match cli.cmd {
        None => wizard::run(cli.repo.as_deref()),
        Some(Cmd::Init(args)) => commands::init::run(cli.repo.as_deref(), args),
        Some(Cmd::Status) => commands::status::run(cli.repo.as_deref()),
        Some(Cmd::Log(args)) => commands::log::run(cli.repo.as_deref(), args),
        Some(Cmd::Show(args)) => commands::show::run(cli.repo.as_deref(), args),
        Some(Cmd::Add(sub)) => commands::add::run(cli.repo.as_deref(), sub),
        Some(Cmd::Query(args)) => commands::query::run(cli.repo.as_deref(), args),
        Some(Cmd::Retrieve(args)) => commands::retrieve::run(cli.repo.as_deref(), args),
        Some(Cmd::Embed(args)) => commands::embed_cmd::run(cli.repo.as_deref(), args),
        Some(Cmd::Reindex(args)) => commands::reindex::run(cli.repo.as_deref(), args),
        Some(Cmd::Embedder(sub)) => {
            commands::embed_cmd::run_embedder(commands::embed_cmd::EmbedderArgs { cmd: sub })
        }
        Some(Cmd::Diff(args)) => commands::diff::run(cli.repo.as_deref(), args),
        Some(Cmd::Ref(sub)) => commands::refs::run(cli.repo.as_deref(), sub),
        Some(Cmd::Config(args)) => commands::cfg_cmd::run(cli.repo.as_deref(), args),
        Some(Cmd::Stats) => commands::stats::run(cli.repo.as_deref()),
        Some(Cmd::Export(args)) => commands::export::run(cli.repo.as_deref(), args),
        Some(Cmd::Import(args)) => commands::import::run(cli.repo.as_deref(), args),
        Some(Cmd::Ingest(args)) => commands::ingest::run(cli.repo.as_deref(), args),
        Some(Cmd::Branch(sub)) => commands::branch::run(cli.repo.as_deref(), sub),
        Some(Cmd::Blame(args)) => commands::blame::run(cli.repo.as_deref(), args),
        Some(Cmd::CatFile(args)) => commands::cat_file::run(cli.repo.as_deref(), args),
        Some(Cmd::Remote(sub)) => commands::remote::run(cli.repo.as_deref(), sub),
        Some(Cmd::Clone(args)) => commands::clone::run(cli.repo.as_deref(), args),
        Some(Cmd::Fetch(a)) => commands::fetch::run(cli.repo.as_deref(), a),
        Some(Cmd::Push(a)) => commands::push::run(cli.repo.as_deref(), a),
        Some(Cmd::Pull(a)) => commands::pull::run(cli.repo.as_deref(), a),
        Some(Cmd::Merge(a)) => commands::merge::run(cli.repo.as_deref(), a),
        Some(Cmd::Revert(a)) => commands::deferred::run_revert(cli.repo.as_deref(), a),
        Some(Cmd::Fsck(a)) => commands::deferred::run_fsck(cli.repo.as_deref(), a),
        Some(Cmd::Gc(a)) => commands::deferred::run_gc(cli.repo.as_deref(), a),
        Some(Cmd::Integrate(args)) => integrate::run(args),
        Some(Cmd::Doctor(args)) => doctor::run(cli.repo.as_deref(), args),
        Some(Cmd::Completions(args)) => commands::completions::run(args),
        Some(Cmd::Bench(args)) => commands::bench::run(args),
    };
    if let Err(e) = result {
        // audit-2026-04-25 R4 (Stage E re-fix): print the anyhow
        // error chain manually with consecutive-duplicate elision.
        // The previous `{e:#}` form printed each segment twice for
        // mnem-core errors that wrap their inner Display in the
        // outer variant's `#[error("repo: {0}")]` while ALSO
        // exposing the inner as `#[source]` via `#[from]` -- the
        // chain walker then visited both. Surfaced as e.g.
        // `error: repo: retrieve: ...: retrieve: ...`.
        let mut last: Option<String> = None;
        let mut parts: Vec<String> = Vec::new();
        for cause in e.chain() {
            let s = format!("{cause}");
            // Skip a segment if it is identical to, OR a strict
            // suffix of, the previous segment. Strict-suffix
            // catches `repo: retrieve: ...` followed by
            // `retrieve: ...`.
            if let Some(prev) = &last
                && (prev == &s || prev.ends_with(&s))
            {
                continue;
            }
            last = Some(s.clone());
            parts.push(s);
        }
        eprintln!("error: {}", parts.join(": "));
        // EX_CONFIG (78) per BSD sysexits for "user asked for
        // something the config doesn't allow yet" - this is the
        // shape of a deferred-stub failure.
        std::process::exit(if is_deferred { 78 } else { 1 });
    }
}
