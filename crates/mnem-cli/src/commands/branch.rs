//! `mnem branch` - the `refs/heads/<name>` convenience verb.
//!
//! `branch` is a thin shell over [`super::refs`] that writes into the
//! `refs/heads/` namespace. It's split out as its own command so a Git
//! user's muscle memory (`git branch`, `git branch feature`,
//! `git branch -D old`) lands where they expect.
//!
//! Semantics:
//!
//! - `branch list` - every ref whose name begins with `refs/heads/`,
//!   plus the current head-commit CID marked with `*`.
//! - `branch create <name> [--from <cid>]` - writes
//!   `refs/heads/<name> -> <cid>`. `--from` accepts any commit CID in
//!   the repo; when absent, it defaults to the current head commit.
//!   Fails if the ref already exists or the repo has no commits yet.
//! - `branch delete <name>` - removes `refs/heads/<name>`. Fails if
//!   the ref does not exist. Does NOT refuse to delete the "current"
//!   branch in Q2 - mnem has no symbolic-ref-to-HEAD analog yet;
//!   every commit targets the heads list directly.
//!
//! Branches are named heads in the op-log graph (+
//! ). Every ref update goes through
//! `ReadonlyRepo::update_ref` (which wraps the write in an Operation
//! ), so `mnem branch create feat` shows up in
//! `mnem log` just like any other mutation.
//!
//! # Examples
//!
//! ```text
//! mnem branch list
//! mnem branch create feature/oauth
//! mnem branch create hotfix --from 01HZ...
//! mnem branch delete old-experiment
//! ```

use super::*;

const BRANCH_PREFIX: &str = "refs/heads/";

/// `mnem branch` subcommand dispatcher.
#[derive(clap::Subcommand, Debug)]
pub(crate) enum BranchCmd {
    /// List every `refs/heads/<name>` ref, marking the head commit.
    List,
    /// Create a new branch. Fails if the name already exists.
    Create {
        /// Branch name. Stored as `refs/heads/<name>` in the View.
        name: String,
        /// Optional positional start-point: a commit CID, ref name,
        /// branch shortname, or `HEAD`. Mirrors `git branch <name>
        /// <start-point>`. When omitted (and `--from` also absent),
        /// defaults to the current head commit.
        ///
        /// audit-2026-04-25 C3-6: Pass-2 found Git users hit a wall
        /// with `mnem branch create feat main`; we now accept that
        /// shape as syntactic sugar for `--from main` and resolve
        /// the start-point through `resolve_commitish` so any commit
        /// CID, ref, or branch name works.
        start_point: Option<String>,
        /// Commit CID / ref / branch shortname to point the new
        /// branch at. Same resolver as the positional `start_point`.
        /// Conflicts with the positional form -- pass one or the
        /// other, not both.
        #[arg(long, conflicts_with = "start_point")]
        from: Option<String>,
    },
    /// Delete a branch. Fails if the name does not exist.
    Delete {
        /// Branch name to delete.
        name: String,
    },
}

pub(crate) fn run(override_path: Option<&Path>, cmd: BranchCmd) -> Result<()> {
    let data_dir = repo::locate_data_dir(override_path)?;
    let cfg = config::load(&data_dir)?;
    let r = repo::open_repo(Some(data_dir.as_path()))?;

    match cmd {
        BranchCmd::List => list_branches(&r),
        BranchCmd::Create {
            name,
            start_point,
            from,
        } => {
            // C3-6: positional start-point is sugar for --from. The
            // clap `conflicts_with` annotation prevents both being
            // set, so an `or` is sufficient here.
            let resolved = from.or(start_point);
            create_branch(&r, &cfg, &name, resolved.as_deref())
        }
        BranchCmd::Delete { name } => delete_branch(&r, &cfg, &name),
    }
}

fn list_branches(r: &ReadonlyRepo) -> Result<()> {
    let refs = &r.view().refs;
    let head = r.view().heads.first().cloned();
    let mut any = false;
    for (name, target) in refs {
        let Some(short) = name.strip_prefix(BRANCH_PREFIX) else {
            continue;
        };
        any = true;
        let marker = match target {
            RefTarget::Normal { target } if Some(target) == head.as_ref() => "*",
            _ => " ",
        };
        let summary = match target {
            RefTarget::Normal { target } => format!("-> {target}"),
            RefTarget::Conflicted { adds, removes } => {
                format!("conflicted(+{} -{})", adds.len(), removes.len())
            }
        };
        println!("{marker} {short}  {summary}");
    }
    if !any {
        println!("<no branches>");
    }
    Ok(())
}

fn create_branch(
    r: &ReadonlyRepo,
    cfg: &config::Config,
    name: &str,
    from: Option<&str>,
) -> Result<()> {
    if name.is_empty() {
        bail!("branch name must not be empty");
    }
    // Accept either a raw name (common) or a fully-qualified refname.
    // A raw name gets the `refs/heads/` prefix; a refname passes
    // through unchanged. This matches git's `git branch refs/foo` UX
    // loosely while preserving "happy path = bare name".
    let full = if name.starts_with(BRANCH_PREFIX) {
        name.to_string()
    } else {
        format!("{BRANCH_PREFIX}{name}")
    };
    if r.view().refs.contains_key(&full) {
        bail!("branch `{name}` already exists");
    }
    // C3-6: route the start-point through the shared `resolve_commitish`
    // resolver so a CID, a ref name (`refs/heads/main`), a branch
    // shortname (`main`), or `HEAD` all work identically. This is what
    // makes `mnem branch create feat main` behave like `git branch feat
    // main` instead of raising a "parsing CID" error.
    let target_cid = match from {
        Some(s) => super::resolve_commitish(r, s).context("resolving start-point")?,
        None => r
            .view()
            .heads
            .first()
            .cloned()
            .ok_or_else(|| anyhow!("repository has no commits yet; pass --from <cid>"))?,
    };
    let new_r = r.update_ref(
        &full,
        None,
        Some(RefTarget::normal(target_cid.clone())),
        &config::author_string(cfg),
    )?;
    println!("created branch {name} -> {target_cid}");
    println!("  op_id    {}", new_r.op_id());
    Ok(())
}

fn delete_branch(r: &ReadonlyRepo, cfg: &config::Config, name: &str) -> Result<()> {
    if name.is_empty() {
        bail!("branch name must not be empty");
    }
    let full = if name.starts_with(BRANCH_PREFIX) {
        name.to_string()
    } else {
        format!("{BRANCH_PREFIX}{name}")
    };
    let prev = r
        .view()
        .refs
        .get(&full)
        .ok_or_else(|| anyhow!("branch `{name}` does not exist"))?;
    let new_r = r.update_ref(&full, Some(prev), None, &config::author_string(cfg))?;
    println!("deleted branch {name}");
    println!("  op_id    {}", new_r.op_id());
    Ok(())
}
