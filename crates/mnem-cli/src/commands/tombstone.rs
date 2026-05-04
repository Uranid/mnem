use super::*;

#[derive(clap::Args, Debug)]
#[command(after_long_help = "\
Examples:
  mnem tombstone <uuid>
  mnem tombstone <uuid> --reason \"superseded by newer decision\"
  mnem tombstone <uuid> -r \"duplicate of <other-uuid>\" -m \"clean up dupes\"
  mnem global tombstone <uuid>              # tombstone in the global graph
  mnem -R ~/notes tombstone <uuid>          # tombstone in a specific repo
")]
pub(crate) struct Args {
    /// UUID of the node to tombstone (soft-delete with an audit trail).
    pub id: String,
    /// Human-readable reason recorded on the tombstone and visible in `mnem log`.
    #[arg(long, short = 'r', default_value = "")]
    pub reason: String,
    /// Commit message.
    #[arg(long, short = 'm', default_value = "mnem tombstone")]
    pub message: String,
}

pub(crate) fn run(override_path: Option<&Path>, args: Args) -> Result<()> {
    let id = mnem_core::id::NodeId::parse_uuid(&args.id)
        .with_context(|| format!("invalid node UUID: {}", args.id))?;
    let (data_dir, repo, _bs, _ohs) = repo::open_all(override_path)?;
    if repo.lookup_node(&id)?.is_none() {
        bail!("no node with id={}", args.id);
    }
    if repo.is_tombstoned(&id) {
        bail!("node {} is already tombstoned", args.id);
    }
    let cfg = config::load(&data_dir)?;
    let author = config::author_string(&cfg);
    let mut tx = repo.start_transaction();
    tx.tombstone_node(id, args.reason.clone())?;
    let new_repo = tx.commit(&author, &args.message)?;
    println!("tombstoned {}", args.id);
    if !args.reason.is_empty() {
        println!(" reason: {}", args.reason);
    }
    println!(" op_id:  {}", new_repo.op_id());
    Ok(())
}
