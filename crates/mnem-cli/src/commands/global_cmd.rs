use super::*;

use crate::{global, repo};

/// `mnem global` subcommand - read and write the global anchor graph at
/// `~/.mnemglobal/.mnem/` directly.
#[derive(clap::Subcommand, Debug)]
pub(crate) enum GlobalCmd {
    /// Search the global graph (~/.mnemglobal/.mnem/) only.
    /// Results are ranked by score.
    #[command(after_long_help = "\
Examples:
  mnem global retrieve \"Alice in Berlin\"
  mnem global retrieve \"climbing\" -n 5
  mnem global retrieve \"project deadline\" --no-vector
")]
    Retrieve(super::retrieve::Args),

    /// Add a node or edge directly to the global graph (~/.mnemglobal/.mnem/).
    /// The printed node UUID can be used as `--prop _global_anchor=<uuid>` when
    /// adding the same entity to a local repo, linking the two graphs.
    ///
    /// Examples:
    ///   mnem global add node -s \"Alice works at Anthropic\" --label Entity:Person --prop name=Alice
    ///   # -> prints: node <uuid> committed
    ///   mnem -R ~/notes add node --label Entity:Person --prop name=Alice --prop _global_anchor=<uuid>
    #[command(subcommand)]
    Add(super::add::AddCmd),

    /// Ingest external source files (Markdown / text / PDF / chat JSON)
    /// into the global graph (~/.mnemglobal/.mnem/) as a Doc + Chunk + Entity subgraph.
    ///
    /// Examples:
    ///   mnem global ingest notes.md
    ///   mnem global ingest --chunker recursive --max-tokens 1024 book.pdf
    ///   mnem global ingest --recursive docs/
    Ingest(super::ingest::Args),

    /// Walk the op-log of the global graph backwards from the current head.
    Log(super::log::Args),

    /// Print current op-head, head commit, ref summary, and label counts
    /// for the global graph.
    Status,

    /// Short one-line stats for the global graph.
    Stats,

    /// Show the full detail of one operation in the global graph.
    Show(super::show::Args),

    /// Query the global graph via property filters + optional traversal.
    Query(super::query::Args),

    /// Walk the incoming-edge index for a node in the global graph.
    Blame(super::blame::Args),

    /// Fetch a single node by UUID from the global graph.
    Get(super::get_node::Args),

    /// Soft-delete a node in the global graph (tombstone with audit trail).
    Tombstone(super::tombstone::Args),

    /// Hard-delete a node from the global graph. No audit trail.
    Delete(super::delete_node::Args),

    /// Emit the raw bytes of a CID from the global graph.
    #[command(name = "cat-file")]
    CatFile(super::cat_file::Args),

    /// Report which refs and how many nodes/edges differ between two ops
    /// in the global graph.
    Diff(super::diff::Args),

    /// Manage named refs in the global graph.
    #[command(subcommand)]
    Ref(super::refs::RefCmd),

    /// Manage `refs/heads/<name>` pointers in the global graph.
    #[command(subcommand)]
    Branch(super::branch::BranchCmd),

    /// Get or set configuration for the global graph.
    Config(super::cfg_cmd::Args),

    /// Export a subtree of the global graph to a CAR v1 archive.
    Export(super::export::Args),

    /// Import a CAR v1 archive into the global graph.
    Import(super::import::Args),

    /// Backfill embeddings for nodes in the global graph that don't have one.
    Embed(super::embed_cmd::Args),

    /// Retro-embed nodes in the global graph that lack a vector.
    Reindex(super::reindex::Args),
}

fn require_global_init(global_dir: &Path) -> Result<()> {
    if !global_dir.join(repo::MNEM_DIR).is_dir() {
        bail!(
            "Global graph not initialised at {}.\n\
             hint: run `mnem integrate` to create it.",
            global_dir.display()
        );
    }
    Ok(())
}

pub(crate) fn run(_override: Option<&Path>, cmd: GlobalCmd) -> Result<()> {
    let global_dir = global::default_dir();
    match cmd {
        GlobalCmd::Retrieve(args) => {
            require_global_init(&global_dir)?;
            super::retrieve::run(Some(&global_dir), args)
        }
        GlobalCmd::Add(add_cmd) => {
            require_global_init(&global_dir)?;
            super::add::run(Some(&global_dir), add_cmd)
        }
        GlobalCmd::Ingest(ingest_args) => {
            require_global_init(&global_dir)?;
            super::ingest::run(Some(&global_dir), ingest_args)
        }
        GlobalCmd::Log(args) => {
            require_global_init(&global_dir)?;
            super::log::run(Some(&global_dir), args)
        }
        GlobalCmd::Status => {
            require_global_init(&global_dir)?;
            super::status::run(Some(&global_dir))
        }
        GlobalCmd::Stats => {
            require_global_init(&global_dir)?;
            super::stats::run(Some(&global_dir))
        }
        GlobalCmd::Show(args) => {
            require_global_init(&global_dir)?;
            super::show::run(Some(&global_dir), args)
        }
        GlobalCmd::Query(args) => {
            require_global_init(&global_dir)?;
            super::query::run(Some(&global_dir), args)
        }
        GlobalCmd::Blame(args) => {
            require_global_init(&global_dir)?;
            super::blame::run(Some(&global_dir), args)
        }
        GlobalCmd::Get(args) => {
            require_global_init(&global_dir)?;
            super::get_node::run(Some(&global_dir), args)
        }
        GlobalCmd::Tombstone(args) => {
            require_global_init(&global_dir)?;
            super::tombstone::run(Some(&global_dir), args)
        }
        GlobalCmd::Delete(args) => {
            require_global_init(&global_dir)?;
            super::delete_node::run(Some(&global_dir), args)
        }
        GlobalCmd::CatFile(args) => {
            require_global_init(&global_dir)?;
            super::cat_file::run(Some(&global_dir), args)
        }
        GlobalCmd::Diff(args) => {
            require_global_init(&global_dir)?;
            super::diff::run(Some(&global_dir), args)
        }
        GlobalCmd::Ref(sub) => {
            require_global_init(&global_dir)?;
            super::refs::run(Some(&global_dir), sub)
        }
        GlobalCmd::Branch(sub) => {
            require_global_init(&global_dir)?;
            super::branch::run(Some(&global_dir), sub)
        }
        GlobalCmd::Config(args) => {
            require_global_init(&global_dir)?;
            super::cfg_cmd::run(Some(&global_dir), args)
        }
        GlobalCmd::Export(args) => {
            require_global_init(&global_dir)?;
            super::export::run(Some(&global_dir), args)
        }
        GlobalCmd::Import(args) => {
            require_global_init(&global_dir)?;
            super::import::run(Some(&global_dir), args)
        }
        GlobalCmd::Embed(args) => {
            require_global_init(&global_dir)?;
            super::embed_cmd::run(Some(&global_dir), args)
        }
        GlobalCmd::Reindex(args) => {
            require_global_init(&global_dir)?;
            super::reindex::run(Some(&global_dir), args)
        }
    }
}

