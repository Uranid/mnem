use super::*;

#[derive(clap::Args, Debug)]
#[command(after_long_help = "\
Examples:
  mnem delete <uuid>                        # hard-delete a node (no audit trail)
  mnem global delete <uuid>                 # hard-delete from the global graph
  mnem -R ~/notes delete <uuid>

Prefer `mnem tombstone` for nodes that should remain in the audit trail.
Hard-delete removes the node from `mnem retrieve` and `mnem query` results
immediately but the block stays in the blockstore until `mnem gc` runs.
")]
pub(crate) struct Args {
    /// UUID of the node to delete.
    pub id: String,
    /// Commit message.
    #[arg(long, short = 'm', default_value = "mnem delete")]
    pub message: String,
}

pub(crate) fn run(override_path: Option<&Path>, args: Args) -> Result<()> {
    let id = mnem_core::id::NodeId::parse_uuid(&args.id)
        .with_context(|| format!("invalid node UUID: {}", args.id))?;
    let (data_dir, repo, _bs, _ohs) = repo::open_all(override_path)?;

    // Check existence before starting any transaction so a failed delete never
    // writes an empty-remove op to history.
    if repo.lookup_node(&id)?.is_none() {
        return Err(anyhow::anyhow!(
            "no node with id={} in current view",
            args.id
        ));
    }

    let cfg = config::load(&data_dir)?;
    let author = config::author_string(&cfg);
    let mut tx = repo.start_transaction();
    tx.remove_node(id);
    let new_repo = tx.commit(&author, &args.message)?;
    println!("deleted {}", args.id);
    println!(" op_id: {}", new_repo.op_id());
    Ok(())
}
