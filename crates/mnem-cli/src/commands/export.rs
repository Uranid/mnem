use super::*;

use std::collections::HashSet;
use std::fmt;
use std::fs::File;
use std::io::{BufWriter, Write};

use bytes::Bytes;
use mnem_core::error::StoreError;
use mnem_core::id::Cid;
use mnem_core::prolly::Cursor;
use mnem_core::store::Blockstore;

fn resolve_ref_or_cid(r: &ReadonlyRepo, from: &str) -> Result<mnem_core::id::Cid> {
    if from == "HEAD" {
        return r
            .view()
            .heads
            .first()
            .cloned()
            .ok_or_else(|| anyhow!("repository has no commits yet; nothing to export"));
    }
    if let Some(target) = r.view().refs.get(from) {
        return match target {
            RefTarget::Normal { target } => Ok(target.clone()),
            RefTarget::Conflicted { adds, removes } => Err(anyhow!(
                "ref `{from}` is conflicted (adds={}, removes={}); resolve it first",
                adds.len(),
                removes.len()
            )),
        };
    }
    mnem_core::id::Cid::parse_str(from)
        .with_context(|| format!("`{from}` is neither a known ref nor a valid CID"))
}

/// Walk the nodes Prolly tree and collect the CIDs of every node whose
/// NodeId appears in `tombstone_ids`. Returns the set of NodeCids that
/// should be omitted from a scrubbed export.
fn collect_tombstoned_node_cids(
    bs: &dyn Blockstore,
    nodes_root: &Cid,
    tombstone_ids: &HashSet<mnem_core::id::NodeId>,
) -> Result<HashSet<Cid>> {
    let mut scrubbed: HashSet<Cid> = HashSet::new();
    let cursor = Cursor::new(bs, nodes_root).context("opening Prolly cursor over nodes tree")?;
    for item in cursor {
        let (key, node_cid) = item.context("iterating nodes Prolly tree")?;
        // ProllyKey bytes are the NodeId bytes (From<NodeId> stores them
        // byte-for-byte). Reconstruct the NodeId for comparison.
        let node_id = mnem_core::id::NodeId::from_bytes_raw(key.0);
        if tombstone_ids.contains(&node_id) {
            scrubbed.insert(node_cid);
        }
    }
    Ok(scrubbed)
}

/// A [`Blockstore`] wrapper that filters a fixed set of CIDs from
/// [`iter_from_root`]. All other operations delegate unchanged to the
/// inner store. Used by `mnem export --scrub` to omit tombstoned node
/// blocks from the emitted CAR without mutating the live blockstore.
struct ScrubBlockstore<'a> {
    inner: &'a dyn Blockstore,
    /// CIDs to exclude from [`iter_from_root`].
    excluded: HashSet<Cid>,
}

impl<'a> ScrubBlockstore<'a> {
    fn new(inner: &'a dyn Blockstore, excluded: HashSet<Cid>) -> Self {
        Self { inner, excluded }
    }
}

impl fmt::Debug for ScrubBlockstore<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ScrubBlockstore {{ excluded: {} }}", self.excluded.len())
    }
}

impl Blockstore for ScrubBlockstore<'_> {
    fn has(&self, cid: &Cid) -> Result<bool, StoreError> {
        self.inner.has(cid)
    }

    fn get(&self, cid: &Cid) -> Result<Option<Bytes>, StoreError> {
        self.inner.get(cid)
    }

    fn put(&self, cid: Cid, data: Bytes) -> Result<(), StoreError> {
        self.inner.put(cid, data)
    }

    fn put_trusted(&self, cid: Cid, data: Bytes) -> Result<(), StoreError> {
        self.inner.put_trusted(cid, data)
    }

    fn delete(&self, cid: &Cid) -> Result<(), StoreError> {
        self.inner.delete(cid)
    }

    fn iter_from_root<'b>(
        &'b self,
        root: &Cid,
    ) -> Box<dyn Iterator<Item = Result<(Cid, Bytes), StoreError>> + 'b> {
        let excluded = &self.excluded;
        Box::new(
            self.inner
                .iter_from_root(root)
                .filter(move |item| match item {
                    Ok((cid, _)) => !excluded.contains(cid),
                    // Keep errors so the caller sees them and can stop.
                    Err(_) => true,
                }),
        )
    }

    fn all_cids(&self) -> Result<Option<Vec<Cid>>, StoreError> {
        self.inner.all_cids()
    }
}

/// `mnem export` arguments.
#[derive(clap::Args, Debug)]
#[command(after_long_help = "\
Examples:
  mnem export out.car                        # export from HEAD (default, recommended)
  mnem export --from HEAD out.car            # same, explicit
  mnem export --from <cid> backup.car        # export from specific op CID
  mnem export --scrub out.car                # omit tombstoned (soft-deleted) nodes
  mnem export - | ssh server 'mnem import -' # pipe over SSH

NOTE: Named refs like `refs/heads/main` do NOT auto-advance with `mnem add node`.
They point to the commit when the branch was created, not the current HEAD.
Using `--from refs/heads/main` exports only the data up to that anchor commit.
Run `mnem status` to compare the current HEAD vs. named ref values.
To export all current data, use `--from HEAD` (or omit `--from` entirely).
")]
pub(crate) struct Args {
    /// Output path for the CAR archive. Use `-` to write to stdout.
    pub path: String,
    /// Ref name or CID to treat as the export root. Defaults to `HEAD`
    /// (the first head commit of the current view). Named refs such as
    /// `refs/heads/main` point to the commit when the branch was created
    /// and do NOT auto-advance when nodes are added via `mnem add node`.
    /// To export all current data, prefer `HEAD` or omit this flag.
    #[arg(long)]
    pub from: Option<String>,
    /// Omit blocks belonging to tombstoned (soft-deleted) nodes from the
    /// exported CAR. By default tombstoned node blocks are included
    /// verbatim because the underlying data is still in the Prolly tree.
    /// Pass `--scrub` to produce a clean archive that does not contain
    /// any content revoked via `mnem tombstone`.
    #[arg(long)]
    pub scrub: bool,
}

pub(crate) fn run(override_path: Option<&Path>, args: Args) -> Result<()> {
    let (_dir, r, bs, _ohs) = repo::open_all(override_path)?;
    let from_str = args.from.as_deref().unwrap_or("HEAD");
    let root = resolve_ref_or_cid(&r, from_str)?;

    // Warn when the user explicitly passed --from and the resolved CID
    // differs from HEAD. This catches the common mistake of passing a
    // named ref (e.g. `refs/heads/main`) that has not been updated since
    // the branch was created, causing stale data to be exported silently.
    if let Some(explicit_from) = &args.from {
        if let Some(head_cid) = r.view().heads.first() {
            if &root != head_cid {
                eprintln!(
                    "warning: `{explicit_from}` resolves to {root} which is behind HEAD ({head_cid})."
                );
                eprintln!("Exporting from this ref will not include recent commits.");
                eprintln!("Use `--from HEAD` to export all current data.");
            }
        }
    }

    // --scrub: walk the nodes Prolly tree to find CIDs that belong to
    // tombstoned NodeIds, then wrap the blockstore in a filter that
    // skips those CIDs during the iter_from_root walk inside export().
    let scrub_store: Option<ScrubBlockstore<'_>> = if args.scrub {
        let tombstones = &r.view().tombstones;
        let n_tombstones = tombstones.len();

        let excluded = if n_tombstones == 0 {
            // Nothing to scrub - skip the Prolly walk entirely.
            HashSet::new()
        } else {
            // Collect the NodeIds that are tombstoned.
            let tombstone_ids: HashSet<mnem_core::id::NodeId> =
                tombstones.keys().copied().collect();

            // We need the head commit to reach the nodes Prolly root.
            let commit = r
                .head_commit()
                .ok_or_else(|| anyhow!("repository has no commits yet; nothing to scrub"))?;

            collect_tombstoned_node_cids(&*bs, &commit.nodes, &tombstone_ids)
                .context("collecting tombstoned node CIDs for --scrub")?
        };

        eprintln!("(scrubbing {} tombstoned nodes from export)", n_tombstones);
        Some(ScrubBlockstore::new(&*bs, excluded))
    } else {
        None
    };

    // Borrow either the scrubbing wrapper or the raw blockstore as a
    // &dyn Blockstore so both branches call the same export() signature.
    let effective_bs: &dyn Blockstore = match &scrub_store {
        Some(s) => s,
        None => &*bs,
    };

    // audit-2026-04-25 P1-5: rewrite git-bash-style `/c/...` paths to
    // `c:/...` on Windows so users running mnem from MSYS2 / Git Bash
    // do not see "system cannot find the path" errors.
    let normalized = super::normalize_cli_path(&args.path);

    let stats = if normalized == "-" {
        let stdout = std::io::stdout();
        let mut lock = stdout.lock();
        mnem_transport::export(effective_bs, &root, &mut lock).context("writing CAR to stdout")?
    } else {
        let path = Path::new(&normalized);
        let file = File::create(path).with_context(|| format!("creating {}", path.display()))?;
        let mut w = BufWriter::new(file);
        let stats = mnem_transport::export(effective_bs, &root, &mut w)
            .with_context(|| format!("writing CAR to {}", path.display()))?;
        // Flushing the BufWriter surfaces any deferred I/O errors
        // that `write_all` swallowed into the buffer.
        w.flush()
            .with_context(|| format!("flushing {}", path.display()))?;
        stats
    };

    println!(
        "exported {} blocks, {} bytes to {}",
        stats.blocks, stats.bytes, normalized
    );
    Ok(())
}
