//! `mnem gc` - garbage-collect unreferenced blocks.
//!
//! Walks the full content-addressed DAG reachable from ALL known refs
//! (local branches/tags in `view.refs`, remote tracking refs in
//! `view.remote_refs`), collects the set of live CIDs, then deletes
//! every block in the store that is not in that set. Unreferenced
//! blocks accumulate from deleted nodes, abandoned branches, and
//! superseded prolly-tree interior nodes.
//!
//! Without `--force`, the command performs a dry run: it reports how
//! many blocks are reachable vs. stored but does not modify the store.

use std::collections::HashSet;

use super::*;

#[derive(clap::Args, Debug)]
#[command(after_long_help = "\
Without --force, prints a report of reachable vs. total blocks without
modifying the store. Pass --force to delete unreachable blocks.

Examples:
  mnem gc              # dry-run: report unreachable block count
  mnem gc --force      # collect unreachable blocks
")]
pub(crate) struct Args {
    /// Actually delete unreachable blocks. Without this flag the
    /// command is a safe dry-run that only reports.
    #[arg(long)]
    pub force: bool,
}

pub(crate) fn run(override_path: Option<&Path>, args: Args) -> Result<()> {
    let (_data_dir, repo, bs, _ohs) = repo::open_all(override_path)?;

    // ── Step 1: collect all CIDs reachable from all known refs ───────
    //
    // Root the walk at HEAD *plus* every named ref target (branches,
    // tags, etc.). Without this, blocks reachable only from non-HEAD
    // refs are incorrectly classified as garbage and deleted, which
    // corrupts those branches.
    let head_op_cid = repo.op_id().clone();

    // Collect all ref-target CIDs from view.refs and view.remote_refs.
    let mut roots: Vec<mnem_core::id::Cid> = vec![head_op_cid];
    let view = repo.view();

    for ref_target in view.refs.values() {
        match ref_target {
            RefTarget::Normal { target } => roots.push(target.clone()),
            RefTarget::Conflicted { adds, .. } => roots.extend(adds.iter().cloned()),
        }
    }
    // Remote tracking refs (populated by `mnem fetch`). Skipping these
    // would GC blocks reachable only from fetched-but-not-merged branches.
    if let Some(remote_map) = &view.remote_refs {
        for inner_map in remote_map.values() {
            for ref_target in inner_map.values() {
                match ref_target {
                    RefTarget::Normal { target } => roots.push(target.clone()),
                    RefTarget::Conflicted { adds, .. } => roots.extend(adds.iter().cloned()),
                }
            }
        }
    }
    // Working-copy commit pointer, if set.
    if let Some(wc) = &view.wc_commit {
        roots.push(wc.clone());
    }
    // Deduplicate so we don't walk the same root twice.
    roots.sort();
    roots.dedup();

    let mut reachable: HashSet<mnem_core::id::Cid> = HashSet::new();

    for root in &roots {
        for item in bs.iter_from_root(root) {
            match item {
                Ok((cid, _bytes)) => {
                    reachable.insert(cid);
                }
                Err(e) => {
                    // A missing or corrupt block mid-walk is a problem.
                    // Abort rather than collecting a partial set and
                    // potentially deleting blocks that are reachable but
                    // simply follow a corrupt node.
                    return Err(anyhow::anyhow!(
                        "gc: reachability walk failed: {e}\n\
                         Run `mnem fsck` to diagnose. GC aborted."
                    ));
                }
            }
        }
    }

    let reachable_count = reachable.len();

    // ── Step 2: enumerate all CIDs in the store ───────────────────────
    let all = match bs.all_cids()? {
        Some(cids) => cids,
        None => {
            // The blockstore backend does not support enumeration.
            // Report what we know and exit 0 (not a hard failure).
            println!("gc: {reachable_count} reachable blocks");
            println!("gc: store does not support block enumeration; cannot compute total");
            println!("gc: (no blocks were deleted)");
            return Ok(());
        }
    };

    let total_count = all.len();

    // ── Step 3: identify unreachable CIDs ────────────────────────────
    let unreachable: Vec<mnem_core::id::Cid> = all
        .into_iter()
        .filter(|cid| !reachable.contains(cid))
        .collect();

    let unreachable_count = unreachable.len();

    // ── Step 4: dry-run report or actual deletion ─────────────────────
    if !args.force {
        println!("gc: {reachable_count} reachable blocks, {total_count} total blocks");
        if unreachable_count == 0 {
            println!("gc: no unreachable blocks (store is clean)");
        } else {
            println!(
                "gc: {unreachable_count} unreachable block(s) found (run with --force to collect)"
            );
        }
        return Ok(());
    }

    // --force: delete each unreachable block.
    if unreachable_count == 0 {
        println!("gc: no unreachable blocks; nothing to collect");
        return Ok(());
    }

    let mut deleted = 0usize;
    let mut errors = 0usize;
    for cid in &unreachable {
        match bs.delete(cid) {
            Ok(()) => deleted += 1,
            Err(e) => {
                eprintln!("gc: warning: failed to delete {cid}: {e}");
                errors += 1;
            }
        }
    }

    println!("gc: removed {deleted} block(s)");
    if errors > 0 {
        println!("gc: {errors} deletion error(s) - run `mnem fsck` to check integrity");
    }

    Ok(())
}
