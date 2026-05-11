//! `mnem switch <branch>` / `mnem checkout <branch>`
//!
//! Switches the active working position to the tip of a named branch.
//! mnem has no working tree, so "checkout" is purely a HEAD update:
//! `view().heads` is advanced to the commit CID that the branch ref
//! points at, recorded as a new Op in the op-log.
//!
//! # Semantics
//!
//! 1. Resolve `refs/heads/<name>` in the current view.
//! 2. Compare the branch tip to the current `view().heads.first()`.
//!    - Same CID: print `Already on '<name>'` and exit 0.
//! 3. Call `ReadonlyRepo::update_heads(tip, author)` to advance HEAD.
//! 4. Print `switched to branch '<name>'`.
//!
//! # Examples
//!
//! ```text
//! mnem switch main
//! mnem checkout feature/oauth
//! ```

use mnem_core::HEADS_PREFIX;

use super::*;

/// Arguments for `mnem switch` / `mnem checkout`.
#[derive(clap::Args, Debug)]
#[command(after_long_help = "\
Switch the active HEAD to an existing branch.

Examples:
  mnem switch main
  mnem switch feature/oauth
  mnem checkout main
  mnem checkout hotfix

Use 'mnem branch list' to see available branches.
")]
pub(crate) struct SwitchArgs {
    /// Branch name to switch to. Resolved as `refs/heads/<name>`.
    pub name: String,
}

pub(crate) fn run(override_path: Option<&Path>, args: SwitchArgs) -> Result<()> {
    let data_dir = repo::locate_data_dir(override_path)?;

    // Guard: refuse to switch branches while a merge is in progress.
    if data_dir.join("MERGE_HEAD").exists() {
        bail!(
            "you are in the middle of a merge; \
             run 'mnem merge --continue' or 'mnem merge --abort' first"
        );
    }

    let cfg = config::load(&data_dir)?;
    let author = config::author_string(&cfg);

    let r = repo::open_repo(Some(data_dir.as_path()))?;

    let name = &args.name;

    // Build the full ref name: accept bare short name or fully-qualified.
    let full_ref = if name.starts_with(HEADS_PREFIX) {
        name.clone()
    } else {
        format!("{HEADS_PREFIX}{name}")
    };

    // Look up the branch ref.
    let branch_tip = match r.view().refs.get(&full_ref) {
        Some(RefTarget::Normal { target }) => target.clone(),
        Some(RefTarget::Conflicted { .. }) => {
            bail!(
                "branch '{name}' is in a conflicted state. \
                 Resolve the conflict before switching."
            )
        }
        None => {
            bail!(
                "error: branch '{name}' not found. \
                 Use 'mnem branch list' or 'GET /v1/branches' to see available branches."
            )
        }
    };

    // Check if HEAD is already at this branch's tip.
    let current_head = r.view().heads.first().cloned();
    if current_head.as_ref() == Some(&branch_tip) {
        println!("Already on '{name}'");
        return Ok(());
    }

    // Advance HEAD to the branch tip and record the active branch (BUG-38).
    r.switch_branch(branch_tip, &full_ref, &author)
        .map_err(anyhow::Error::from)?;

    println!("switched to branch '{name}'");
    Ok(())
}
