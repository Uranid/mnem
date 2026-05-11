//! `mnem tag` - the `refs/tags/<name>` convenience verb.
//!
//! `tag` is a thin shell over [`super::refs`] that writes into the
//! `refs/tags/` namespace. It's split out as its own command so a Git
//! user's muscle memory (`git tag`, `git tag v1.0`, `git tag -d old`)
//! lands where they expect.
//!
//! Semantics:
//!
//! - `tag list` - every ref whose name begins with `refs/tags/`,
//!   printed as `<name>  -> <cid>`.
//! - `tag create <name> [<target>]` - writes
//!   `refs/tags/<name> -> <cid>`. The optional positional `<target>`
//!   accepts a commit CID, ref name, or branch shortname; when absent,
//!   defaults to the current head commit. Fails if the ref already
//!   exists or the repo has no commits yet.
//! - `tag delete <name>` - removes `refs/tags/<name>`. Fails if the
//!   ref does not exist.
//!
//! # Examples
//!
//! ```text
//! mnem tag list
//! mnem tag create v1.0
//! mnem tag create v1.0 --from 01HZ...
//! mnem tag delete v0.9
//! ```

use std::sync::Arc;

use mnem_core::TAGS_PREFIX;
use mnem_core::store::Blockstore;

use super::*;

/// `mnem tag` subcommand dispatcher.
#[derive(clap::Subcommand, Debug)]
pub(crate) enum TagCmd {
    /// List every `refs/tags/<name>` ref with their target CIDs.
    List,
    /// Create a new tag. Fails if the name already exists.
    Create {
        /// Tag name. Stored as `refs/tags/<name>` in the View.
        name: String,
        /// Optional positional target: a commit CID, ref name,
        /// branch shortname, or `HEAD`. When omitted (and `--from`
        /// also absent), defaults to the current head commit.
        target: Option<String>,
        /// Commit CID / ref / branch shortname to point the new tag
        /// at. Same resolver as the positional `target`.
        /// Conflicts with the positional form -- pass one or the
        /// other, not both.
        #[arg(long, conflicts_with = "target")]
        from: Option<String>,
    },
    /// Delete a tag. Fails if the name does not exist.
    Delete {
        /// Tag name to delete.
        name: String,
    },
}

pub(crate) fn run(override_path: Option<&Path>, cmd: TagCmd) -> Result<()> {
    let data_dir = repo::locate_data_dir(override_path)?;
    let cfg = config::load(&data_dir)?;
    let (_dir, r, bs, _ohs) = repo::open_all(Some(data_dir.as_path()))?;

    match cmd {
        TagCmd::List => list_tags(&r),
        TagCmd::Create { name, target, from } => {
            // Positional target is sugar for --from. The clap
            // `conflicts_with` annotation prevents both being set, so
            // an `or` is sufficient here.
            let resolved = from.or(target);
            create_tag(&r, &bs, &cfg, &name, resolved.as_deref())
        }
        TagCmd::Delete { name } => delete_tag(&r, &cfg, &name),
    }
}

fn list_tags(r: &ReadonlyRepo) -> Result<()> {
    let refs = &r.view().refs;
    let mut any = false;
    for (name, target) in refs {
        let Some(short) = name.strip_prefix(TAGS_PREFIX) else {
            continue;
        };
        any = true;
        let summary = match target {
            RefTarget::Normal { target } => format!("-> {target}"),
            RefTarget::Conflicted { adds, removes } => {
                format!("conflicted(+{} -{})", adds.len(), removes.len())
            }
        };
        println!("  {short}  {summary}");
    }
    if !any {
        println!("<no tags>");
    }
    Ok(())
}

fn create_tag(
    r: &ReadonlyRepo,
    bs: &Arc<dyn Blockstore>,
    cfg: &config::Config,
    name: &str,
    from: Option<&str>,
) -> Result<()> {
    if name.is_empty() {
        bail!("tag name must not be empty");
    }
    // Reject refname characters that are invalid in most VCS tooling.
    // Same validation rules as `mnem branch create`.
    if name.contains(' ')
        || name.contains('\t')
        || name.contains('\n')
        || name.contains('\x00')
        || name.contains('~')
        || name.contains('^')
        || name.contains(':')
        || name.contains('?')
        || name.contains('*')
        || name.contains('[')
        || name.contains('\\')
        || name.contains("@{")
        || name.contains("..")
        || name.contains("//")
        || name.starts_with('/')
        || name.ends_with('/')
        || name.ends_with('.')
        || name.ends_with(".lock")
    {
        bail!(
            "invalid tag name `{name}`: tag names may not contain spaces, \
             control characters, `~`, `^`, `:`, `?`, `*`, `[`, `\\`, `@{{`, `..`, \
             `//`, trailing `.`, `.lock` suffix, or start/end with `/`"
        );
    }
    // Accept either a raw name (common) or a fully-qualified refname.
    // A raw name gets the `refs/tags/` prefix; a refname passes
    // through unchanged.
    let full = if name.starts_with(TAGS_PREFIX) {
        name.to_string()
    } else {
        format!("{TAGS_PREFIX}{name}")
    };
    if r.view().refs.contains_key(&full) {
        bail!("tag `{name}` already exists");
    }
    // Route the target through the shared `resolve_commitish` resolver
    // so a CID, a ref name (`refs/heads/main`), a branch shortname
    // (`main`), or `HEAD` all work identically.
    let target_cid = match from {
        Some(s) => super::resolve_commitish(r, s).context("resolving target")?,
        None => r
            .view()
            .heads
            .first()
            .cloned()
            .ok_or_else(|| anyhow!("repository has no commits yet; pass --from <cid>"))?,
    };

    // Validate that target_cid actually points to a Commit block.
    // `mnem log --format=json` returns op-log CIDs, which are Operations,
    // not Commits. Without this check, the CID is accepted silently and
    // causes downstream failures. Fetch the block and attempt a Commit
    // decode now so the user gets an actionable error immediately.
    {
        let bytes = bs
            .get(&target_cid)?
            .ok_or_else(|| anyhow!("block {target_cid} not found in blockstore"))?;
        if from_canonical_bytes::<Commit>(&bytes).is_err() {
            bail!(
                "`{target_cid}` does not decode as a commit.\n\
                 `mnem log --format=json` returns op CIDs; use \
                 `mnem show <op-cid>` to see the commit CID, or use \
                 `HEAD` / a branch name as the --from argument."
            );
        }
    }
    let new_r = r.update_ref(
        &full,
        None,
        Some(RefTarget::normal(target_cid.clone())),
        &config::author_string(cfg),
    )?;
    println!("created tag {name} -> {target_cid}");
    println!("  op_id    {}", new_r.op_id());
    Ok(())
}

fn delete_tag(r: &ReadonlyRepo, cfg: &config::Config, name: &str) -> Result<()> {
    if name.is_empty() {
        bail!("tag name must not be empty");
    }
    let full = if name.starts_with(TAGS_PREFIX) {
        name.to_string()
    } else {
        format!("{TAGS_PREFIX}{name}")
    };
    let prev = r
        .view()
        .refs
        .get(&full)
        .ok_or_else(|| anyhow!("tag `{name}` does not exist"))?;
    let new_r = r.update_ref(&full, Some(prev), None, &config::author_string(cfg))?;
    println!("deleted tag {name}");
    println!("  op_id    {}", new_r.op_id());
    Ok(())
}
