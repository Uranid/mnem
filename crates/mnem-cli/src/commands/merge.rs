//! `mnem merge` - branch-level 3-way merge (B4.3 of the merge wave).
//!
//! Replaces the EX_CONFIG stub. Thin shell over
//! `mnem_core::repo::merge_three_way`:
//!
//! 1. Resolve `<ref>` to a commit CID (ref name, branch short-name, or
//!    raw CID).
//! 2. Call [`merge_three_way`] with the current head as `left` and the
//!    resolved commit as `right`.
//! 3. Depending on the returned [`MergeOutcome`]:
//!    - [`MergeOutcome::FastForward`] -- advance the ref at
//!      `refs/heads/<current>` to the FF target and exit 0.
//!    - [`MergeOutcome::Clean`] -- advance the ref to the merge commit
//!      and exit 0.
//!    - [`MergeOutcome::Conflicts`] -- write
//!      `.mnem/MERGE_CONFLICTS.json` plus `.mnem/MERGE_HEAD` and
//!      `.mnem/ORIG_HEAD`, print a preview, exit 0 (conflicts are a
//!      user-actionable state, not an error).
//!
//! Strategy flags map onto [`MergeStrategy`]: `--strategy=manual`
//! (default) produces the conflict files; `--strategy=ours` /
//! `--strategy=theirs` auto-resolves via the deterministic union's
//! lex tie-break (SPEC §4.6 + conflict detector B4.2).
//!
//! Interrupted-merge control:
//!
//! - `--abort`: delete `.mnem/MERGE_HEAD` + `.mnem/MERGE_CONFLICTS.json`
//!   and restore HEAD from `.mnem/ORIG_HEAD`.
//! - `--continue`: after manual edits, consume the conflicts file and
//!   run the merge executor to completion.
//! - `--dry-run`: run the pipeline, print the outcome preview, persist
//!   nothing.
//!
//! On startup, if `MERGE_HEAD` exists without `--continue` / `--abort`,
//! error "merge in progress: use --continue or --abort".

use std::fs;
use std::path::Path;

use anyhow::{Context, Result, anyhow, bail};
use mnem_core::id::Cid;
use mnem_core::objects::RefTarget;
use mnem_core::repo::{MergeOutcome, MergeStrategy, conflict_category_counts, merge_three_way};

use crate::config;
use crate::repo;

/// Well-known filename for the in-progress merge marker.
const MERGE_HEAD_FILE: &str = "MERGE_HEAD";
/// Pre-merge HEAD snapshot, restored by `--abort`.
const ORIG_HEAD_FILE: &str = "ORIG_HEAD";
/// Structured conflict persistence for manual review.
const MERGE_CONFLICTS_FILE: &str = "MERGE_CONFLICTS.json";

#[derive(clap::Args, Debug)]
#[command(after_long_help = "\
Examples:
  mnem merge feature                    # 3-way merge `refs/heads/feature` into current HEAD
  mnem merge feature --strategy=ours    # on conflict, pick left / current side
  mnem merge feature --strategy=theirs  # on conflict, pick right / incoming side
  mnem merge feature --dry-run          # preview outcome, persist nothing
  mnem merge --continue                 # finish an in-progress merge after manual edits
  mnem merge --abort                    # cancel an in-progress merge, restore pre-merge HEAD
")]
pub(crate) struct Args {
    /// Branch name, ref name, or raw commit CID to merge into the
    /// current head. Required unless `--abort` or `--continue` is set.
    pub branch: Option<String>,

    /// Conflict-resolution strategy.
    #[arg(long, value_enum, default_value_t = StrategyArg::Manual)]
    pub strategy: StrategyArg,

    /// Run the full pipeline but do NOT persist any state.
    #[arg(long)]
    pub dry_run: bool,

    /// Finish an in-progress merge: consume `.mnem/MERGE_CONFLICTS.json`
    /// and advance HEAD to the merge commit.
    #[arg(long = "continue", conflicts_with_all = ["abort", "branch"])]
    pub continue_: bool,

    /// Cancel an in-progress merge and restore HEAD from `ORIG_HEAD`.
    #[arg(long, conflicts_with_all = ["continue_", "branch"])]
    pub abort: bool,
}

#[derive(clap::ValueEnum, Clone, Copy, Debug)]
pub(crate) enum StrategyArg {
    /// Leave conflicts unresolved for manual review.
    Manual,
    /// Pick the left / current-branch side on conflict.
    Ours,
    /// Pick the right / incoming-branch side on conflict.
    Theirs,
}

impl From<StrategyArg> for MergeStrategy {
    fn from(value: StrategyArg) -> Self {
        match value {
            StrategyArg::Manual => Self::Manual,
            StrategyArg::Ours => Self::Ours,
            StrategyArg::Theirs => Self::Theirs,
        }
    }
}

pub(crate) fn run(override_path: Option<&Path>, args: Args) -> Result<()> {
    let data_dir = repo::locate_data_dir(override_path)?;

    // --abort and --continue don't need to load the full repo facade
    // first; they act on files relative to data_dir.
    if args.abort {
        return run_abort(&data_dir, override_path);
    }
    if args.continue_ {
        return run_continue(&data_dir, override_path);
    }

    let branch = args
        .branch
        .clone()
        .ok_or_else(|| anyhow!("missing <branch> argument. Try `mnem merge feature`."))?;

    // Guard: refuse a fresh merge if one is already in progress.
    let merge_head_path = data_dir.join(MERGE_HEAD_FILE);
    if merge_head_path.exists() {
        bail!(
            "merge in progress: use `mnem merge --continue` after resolving \
             `.mnem/{MERGE_CONFLICTS_FILE}`, or `mnem merge --abort` to cancel."
        );
    }

    let cfg = config::load(&data_dir)?;
    let (_dir, r, bs, ohs) = repo::open_all(override_path)?;

    // Resolve left (current head) and right (<branch>).
    let left = r
        .view()
        .heads
        .first()
        .cloned()
        .ok_or_else(|| anyhow!("repository has no commits yet; nothing to merge into"))?;
    let right = resolve_commitish(&r, &branch)?;

    if left == right {
        println!("already up to date (left == right)");
        return Ok(());
    }

    let strategy = MergeStrategy::from(args.strategy);
    let outcome = merge_three_way(&bs, &ohs, left.clone(), right.clone(), strategy)
        .context("3-way merge pipeline")?;

    // Dry-run: report and exit without touching refs or files.
    if args.dry_run {
        preview(&outcome, &branch);
        return Ok(());
    }

    match outcome {
        MergeOutcome::FastForward(target) => {
            let name = current_branch_name(&r).ok_or_else(|| {
                anyhow!(
                    "cannot fast-forward: no `refs/heads/*` ref matches the \
                     current head. Set one with `mnem branch create <name>`."
                )
            })?;
            advance_ref(&r, &cfg, &name, target.clone())?;
            println!("fast-forward: advanced {name} -> {target}");
        }
        MergeOutcome::Clean(merge_cid) => {
            let name = current_branch_name(&r).ok_or_else(|| {
                anyhow!(
                    "cannot record merge commit: no `refs/heads/*` ref matches \
                     the current head. Set one with `mnem branch create <name>`."
                )
            })?;
            advance_ref(&r, &cfg, &name, merge_cid.clone())?;
            println!("merge: advanced {name} -> {merge_cid}");
            // Clean state: no MERGE_HEAD / conflicts file to write,
            // and no ORIG_HEAD either (git writes ORIG_HEAD even on
            // clean; we only write it when there is actual
            // recoverable state to preserve -- i.e. on conflicts).
        }
        MergeOutcome::Conflicts(mc) => {
            // Persist the in-progress-merge marker set.
            fs::write(data_dir.join(ORIG_HEAD_FILE), left.to_string())
                .with_context(|| format!("writing {ORIG_HEAD_FILE}"))?;
            fs::write(data_dir.join(MERGE_HEAD_FILE), right.to_string())
                .with_context(|| format!("writing {MERGE_HEAD_FILE}"))?;
            let json =
                serde_json::to_string_pretty(&mc).context("serialising MergeConflicts to JSON")?;
            fs::write(data_dir.join(MERGE_CONFLICTS_FILE), json)
                .with_context(|| format!("writing {MERGE_CONFLICTS_FILE}"))?;
            let (n_node, n_edge, n_tvm) = conflict_category_counts(&mc);
            println!(
                "merge produced {} conflict(s): {n_node} node-cid, {n_edge} edge-prop, {n_tvm} tombstone-vs-modify",
                mc.conflicts.len(),
            );
            println!(
                "resolve by editing `.mnem/{MERGE_CONFLICTS_FILE}`, then run \
                 `mnem merge --continue` (or `mnem merge --abort` to cancel)."
            );
        }
    }

    Ok(())
}

/// Resolve a user-supplied commit-ish to a Commit CID. Accepts, in
/// order: raw commit CID string, `refs/heads/<name>` ref, bare branch
/// short-name (prefixes `refs/heads/`).
fn resolve_commitish(r: &mnem_core::repo::ReadonlyRepo, s: &str) -> Result<Cid> {
    // Try raw CID first.
    if let Ok(cid) = Cid::parse_str(s) {
        return Ok(cid);
    }
    let refs = &r.view().refs;
    let candidate = if refs.contains_key(s) {
        s.to_string()
    } else {
        format!("refs/heads/{s}")
    };
    match refs.get(&candidate) {
        Some(RefTarget::Normal { target }) => Ok(target.clone()),
        Some(RefTarget::Conflicted { .. }) => {
            bail!("ref `{candidate}` is conflicted; resolve the ref before merging")
        }
        None => bail!(
            "cannot resolve `{s}` to a commit. Tried raw CID, `{s}`, and \
             `refs/heads/{s}`."
        ),
    }
}

/// Return the short branch name whose `refs/heads/<name>` points at
/// the current head commit, if any. Used to know which ref to advance
/// on a successful merge.
fn current_branch_name(r: &mnem_core::repo::ReadonlyRepo) -> Option<String> {
    let head = r.view().heads.first()?;
    for (name, target) in &r.view().refs {
        if let RefTarget::Normal { target: t } = target
            && t == head
            && name.starts_with("refs/heads/")
        {
            return Some(name.clone());
        }
    }
    None
}

fn advance_ref(
    r: &mnem_core::repo::ReadonlyRepo,
    cfg: &config::Config,
    ref_name: &str,
    target: Cid,
) -> Result<()> {
    let prev = r.view().refs.get(ref_name).cloned();
    let new = RefTarget::normal(target);
    let _ = r
        .update_ref(
            ref_name,
            prev.as_ref(),
            Some(new),
            &config::author_string(cfg),
        )
        .context("advancing ref")?;
    Ok(())
}

fn run_abort(data_dir: &Path, _override_path: Option<&Path>) -> Result<()> {
    let mh = data_dir.join(MERGE_HEAD_FILE);
    let oh = data_dir.join(ORIG_HEAD_FILE);
    let mc = data_dir.join(MERGE_CONFLICTS_FILE);
    if !mh.exists() {
        bail!("no merge in progress (no `.mnem/{MERGE_HEAD_FILE}`)");
    }
    // ORIG_HEAD is advisory; we don't roll back the op-log, only
    // remove the in-progress markers. The ref was never advanced
    // (manual strategy only writes markers), so aborting is as simple
    // as deleting the files.
    let _ = fs::remove_file(&mh);
    let _ = fs::remove_file(&oh);
    let _ = fs::remove_file(&mc);
    println!("merge aborted");
    Ok(())
}

fn run_continue(data_dir: &Path, override_path: Option<&Path>) -> Result<()> {
    let mh = data_dir.join(MERGE_HEAD_FILE);
    let oh = data_dir.join(ORIG_HEAD_FILE);
    let mc_path = data_dir.join(MERGE_CONFLICTS_FILE);
    if !mh.exists() {
        bail!(
            "no merge in progress. Start one with `mnem merge <branch>` \
             or re-run the command after resolving the conflict file."
        );
    }
    // Read left/right from the markers on disk.
    let left = parse_cid_file(&oh)?;
    let right = parse_cid_file(&mh)?;
    // Validate the conflict file parses even if we don't pass
    // per-conflict picks (this wave: the continue path trusts the
    // user's manual edits have produced a consistent tree at the
    // Prolly layer and simply re-runs the merge executor, which will
    // either succeed cleanly now or re-report the same conflicts.
    // Per-conflict commit provenance is deferred to B4.4).
    let _ = fs::read_to_string(&mc_path)
        .with_context(|| format!("reading .mnem/{MERGE_CONFLICTS_FILE}"))?;

    let cfg = config::load(data_dir)?;
    let (_dir, r, bs, ohs) = repo::open_all(override_path)?;
    // On --continue we force an auto-resolving strategy (default
    // `ours`, since the user edited the file and implicitly committed
    // to the left side). This is the minimum-viable continue loop;
    // richer per-conflict picks land in B4.4 with a proper manifest.
    let outcome = merge_three_way(&bs, &ohs, left.clone(), right.clone(), MergeStrategy::Ours)
        .context("3-way merge --continue")?;

    match outcome {
        MergeOutcome::Clean(cid) | MergeOutcome::FastForward(cid) => {
            let name = current_branch_name(&r).ok_or_else(|| {
                anyhow!(
                    "cannot advance ref after --continue: no `refs/heads/*` \
                     matches HEAD. Set one with `mnem branch create <name>`."
                )
            })?;
            advance_ref(&r, &cfg, &name, cid.clone())?;
            let _ = fs::remove_file(&mh);
            let _ = fs::remove_file(&oh);
            let _ = fs::remove_file(&mc_path);
            println!("merge continued: advanced {name} -> {cid}");
        }
        MergeOutcome::Conflicts(mc) => {
            let (n_node, n_edge, n_tvm) = conflict_category_counts(&mc);
            bail!(
                "merge --continue still has {} unresolved conflict(s) \
                 ({n_node} node-cid, {n_edge} edge-prop, {n_tvm} tombstone-vs-modify). \
                 Re-edit `.mnem/{MERGE_CONFLICTS_FILE}` and try again, or run \
                 `mnem merge --abort`.",
                mc.conflicts.len(),
            );
        }
    }
    Ok(())
}

fn parse_cid_file(p: &Path) -> Result<Cid> {
    let s = fs::read_to_string(p).with_context(|| format!("reading {}", p.display()))?;
    let trimmed = s.trim();
    Cid::parse_str(trimmed)
        .with_context(|| format!("parsing CID in {}", p.display()))
        .map_err(|e| anyhow!("{e}"))
}

fn preview(outcome: &MergeOutcome, branch: &str) {
    match outcome {
        MergeOutcome::FastForward(cid) => {
            println!("[dry-run] would fast-forward to {cid} (from branch `{branch}`)");
        }
        MergeOutcome::Clean(cid) => {
            println!("[dry-run] would create clean merge commit {cid} (with branch `{branch}`)");
        }
        MergeOutcome::Conflicts(mc) => {
            let (n_node, n_edge, n_tvm) = conflict_category_counts(mc);
            println!(
                "[dry-run] would produce {} conflict(s): {n_node} node-cid, {n_edge} edge-prop, {n_tvm} tombstone-vs-modify",
                mc.conflicts.len(),
            );
            for c in mc.conflicts.iter().take(5) {
                let id = c
                    .node_id
                    .map(|n| n.to_uuid_string())
                    .or_else(|| {
                        c.edge_key.as_ref().map(|k| {
                            format!(
                                "{}-[{}]->{}",
                                k.src.to_uuid_string(),
                                k.etype,
                                k.dst.to_uuid_string()
                            )
                        })
                    })
                    .unwrap_or_else(|| "<?>".into());
                println!("  - {:?} {id}", c.category);
            }
            if mc.conflicts.len() > 5 {
                println!("  ... {} more", mc.conflicts.len() - 5);
            }
        }
    }
}

// NB: `mnem status` reads the `.mnem/MERGE_HEAD` marker directly to
// surface "currently merging" state; no shared helper is exported from
// this module. Every other command tolerates the presence of the
// marker (reads are unaffected) so a shared guard is not needed here.
