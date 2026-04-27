//! `mnem pull [<remote>] [<branch>]` - fast-forward-only integration
//! of a remote branch into the local HEAD.
//!
//! 3-way merge is out of scope here; this verb is deliberately
//! limited to the fast-forward case:
//!
//! 1. Delegate to `mnem fetch <remote>` so tracking refs land and
//!    the CAR is in the blockstore.
//! 2. If local HEAD is an ancestor of the fetched tip (trivially
//!    true when local HEAD is empty), advance local HEAD via a
//!    ref-update operation on `refs/heads/<branch>`.
//! 3. Otherwise refuse with a message pointing at the B4 `merge`
//!    verb.
//!
//! Ancestry check is intentionally cheap: we walk the commit DAG
//! from the remote tip looking for the local head CID. For the
//! common case (empty local or strict fast-forward) the walk is
//! short; a future revision can push this into mnem-core proper.

use mnem_core::id::Cid;
use mnem_core::objects::RefTarget;

use super::*;

#[derive(clap::Args, Debug)]
#[command(after_long_help = "\
Examples:
  mnem pull                           # fast-forward origin/main into HEAD
  mnem pull origin main

Fast-forward only. Use `mnem merge <remote>/<branch>` (B4) for 3-way merges.
")]
pub(crate) struct Args {
    /// Remote name. Defaults to `origin`.
    pub remote: Option<String>,
    /// Branch to pull. Defaults to `main`.
    pub branch: Option<String>,
}

pub(crate) fn run(override_path: Option<&Path>, args: Args) -> Result<()> {
    let remote_name = args.remote.as_deref().unwrap_or("origin").to_string();
    let branch = args.branch.as_deref().unwrap_or("main").to_string();

    // Step 1: delegate to fetch so the tracking ref + CAR are local.
    crate::commands::fetch::run(
        override_path,
        crate::commands::fetch::Args {
            remote: Some(remote_name.clone()),
        },
    )?;

    // Step 2: re-open to see the newly-advanced tracking ref.
    let (data_dir, repo, bs, _ohs) = repo::open_all(override_path)?;
    let tracking_key = format!("refs/remotes/{remote_name}/{branch}");
    let remote_tip = match repo.view().refs.get(&tracking_key) {
        Some(RefTarget::Normal { target }) => target.clone(),
        _ => {
            return Err(anyhow!(
                "no tracking ref {tracking_key}; did `mnem fetch {remote_name}` complete?"
            ));
        }
    };

    let local_head = repo.view().heads.first().cloned();
    let local_branch_key = format!("refs/heads/{branch}");
    let local_branch = repo.view().refs.get(&local_branch_key).cloned();

    // Fast-forward cases:
    //  a) empty local -> trivially ff.
    //  b) local HEAD == remote_tip -> already up-to-date.
    //  c) local HEAD is an ancestor of remote_tip -> ff.
    //  otherwise: non-ff, refuse.
    if let Some(lh) = &local_head
        && lh == &remote_tip
    {
        println!("Already up to date.");
        return Ok(());
    }

    if let Some(lh) = &local_head
        && !is_ancestor(&*bs, lh, &remote_tip)?
    {
        return Err(anyhow!(
            "mnem pull: non-fast-forward detected, use 'mnem merge {remote_name}/{branch}' \
             (requires B4 merge verb)"
        ));
    }

    // Advance the local branch ref. NOTE: op-head advancement
    // follows the same ref-update pattern as `mnem ref set`; the
    // commit graph is already reachable locally (fetch pulled it).
    let cfg_local = config::load(&data_dir)?;
    repo.update_ref(
        &local_branch_key,
        local_branch.as_ref(),
        Some(RefTarget::normal(remote_tip.clone())),
        &config::author_string(&cfg_local),
    )
    .with_context(|| format!("update_ref {local_branch_key}"))?;

    println!(
        "Fast-forward {} -> {}",
        local_head
            .as_ref()
            .map_or_else(|| "<empty>".to_string(), short_cid),
        short_cid(&remote_tip),
    );
    Ok(())
}

/// Cheap ancestry check: walk commits from `tip`, following each
/// commit's `parents`, return true if `needle` shows up. Bounded by
/// blockstore size so a pathological local corruption cannot loop
/// forever.
fn is_ancestor(bs: &dyn mnem_core::store::Blockstore, needle: &Cid, tip: &Cid) -> Result<bool> {
    use mnem_core::codec::from_canonical_bytes;
    use std::collections::{HashSet, VecDeque};
    let mut seen: HashSet<Cid> = HashSet::new();
    let mut q: VecDeque<Cid> = VecDeque::new();
    q.push_back(tip.clone());
    while let Some(cur) = q.pop_front() {
        if !seen.insert(cur.clone()) {
            continue;
        }
        if &cur == needle {
            return Ok(true);
        }
        let Some(bytes) = bs.get(&cur)? else {
            continue;
        };
        let Ok(commit) = from_canonical_bytes::<Commit>(&bytes) else {
            continue;
        };
        for p in &commit.parents {
            q.push_back(p.clone());
        }
        // Bound the walk at 10_000 commits; beyond that we assume
        // not an ancestor. Agents don't produce that much linear
        // history in a single session.
        if seen.len() > 10_000 {
            break;
        }
    }
    Ok(false)
}

fn short_cid(c: &Cid) -> String {
    let s = c.to_string();
    let take = s.len().min(12);
    s[..take].to_string()
}
