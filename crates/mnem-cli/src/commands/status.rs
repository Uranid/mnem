//! `mnem status` - current op-head, head commit, ref summary, label
//! counts, and any surfaced conflicts.
//!
//! Hardened in Q2 : surfaces `RefTarget::Conflicted` refs
//! explicitly (git analog: `both modified: <path>`), flags the "no
//! commits yet" state, and prints the node/edge root CIDs from the
//! head commit so the user can pipe them into `mnem show` without a
//! separate `mnem log` call.

use super::*;

pub(crate) fn run(override_path: Option<&Path>) -> Result<()> {
    let (data_dir, r, bs, _ohs) = repo::open_all(override_path)?;
    let commit = r.head_commit();
    let refs = &r.view().refs;
    let idx = load_index_set(&bs, commit)?;

    println!("repo       {}", data_dir.display());
    println!("op_id      {}", r.op_id());

    // B4.3: surface any in-progress merge. The files are written by
    // `mnem merge` when it hits a `MergeOutcome::Conflicts` branch.
    // A present `MERGE_HEAD` means the user has conflicts to resolve
    // in `.mnem/MERGE_CONFLICTS.json`; `mnem merge --continue` /
    // `mnem merge --abort` exit this state.
    let merge_head = data_dir.join("MERGE_HEAD");
    if merge_head.exists() {
        let right = std::fs::read_to_string(&merge_head)
            .map(|s| s.trim().to_string())
            .unwrap_or_else(|_| "<unreadable>".into());
        let head_short = r
            .view()
            .heads
            .first()
            .map(std::string::ToString::to_string)
            .unwrap_or_else(|| "<no head>".into());
        println!(
            "MERGING    You are currently merging {right} into {head_short}. \
             Resolve conflicts in .mnem/MERGE_CONFLICTS.json then run \
             'mnem merge --continue', or 'mnem merge --abort' to cancel."
        );
    }
    match r.view().heads.first() {
        Some(h) => println!("commit     {h}"),
        None => println!("commit     <none - repository has no commits yet>"),
    }
    if let Some(c) = commit {
        println!("  change   {}", c.change_id.to_uuid_string());
        println!("  message  {}", c.message);
        println!("  nodes    {}", c.nodes);
        println!("  edges    {}", c.edges);
        if let Some(i) = &c.indexes {
            println!("  indexes  {i}");
        }
    }
    // Count normal + conflicted refs separately - the latter is the
    // "repo needs attention" state and deserves a visible marker.
    let (normal, conflicted): (Vec<_>, Vec<_>) = refs
        .iter()
        .partition(|(_, t)| matches!(t, RefTarget::Normal { .. }));
    println!(
        "refs       {} ({})",
        refs.len(),
        if refs.is_empty() {
            "none".to_string()
        } else {
            format!("{} normal, {} conflicted", normal.len(), conflicted.len())
        }
    );
    if !conflicted.is_empty() {
        // Conflicted refs are the closest mnem analog to "both
        // modified" in git status. Surface them prominently so the
        // user's habit of running `mnem status` before work catches
        // them.
        println!("CONFLICTED:");
        for (name, target) in &conflicted {
            if let RefTarget::Conflicted { adds, removes } = target {
                println!("  {name}  (+{} -{})", adds.len(), removes.len());
            }
        }
    }
    // Surface remote tracking refs if any are present. Post-clone,
    // `origin/main` shows up here so users can confirm the remote
    // was wired correctly.
    if let Some(rr) = &r.view().remote_refs {
        let total: usize = rr.values().map(std::collections::BTreeMap::len).sum();
        if total > 0 {
            println!("tracking   {total} across {} remote(s)", rr.len());
        }
    }
    match idx {
        Some(idx) => println!(
            "labels     {} distinct (use `mnem stats` or `mnem query --where K=V`)",
            idx.nodes_by_label.len()
        ),
        None => println!("labels     <no IndexSet>"),
    }
    Ok(())
}
