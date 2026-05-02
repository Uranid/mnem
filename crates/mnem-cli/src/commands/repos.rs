use super::*;

use crate::global::{self, RepoRegistry};

/// `mnem repos` subcommand - flat Subcommand enum matching the RemoteCmd/BranchCmd pattern.
#[derive(clap::Subcommand, Debug)]
pub(crate) enum ReposCmd {
    /// List all registered repos.
    List,
    /// Mark a repo path as the default for `mnem` without -R.
    SetDefault {
        /// Path to the repo directory (the parent of `.mnem/`).
        path: std::path::PathBuf,
    },
    /// Remove registry entries whose paths no longer exist on disk.
    Prune,
}

pub(crate) fn run(_override: Option<&Path>, cmd: ReposCmd) -> Result<()> {
    let global_dir = global::default_dir();

    match cmd {
        ReposCmd::List => cmd_list(&global_dir),
        ReposCmd::SetDefault { path } => cmd_set_default(&global_dir, &path),
        ReposCmd::Prune => cmd_prune(&global_dir),
    }
}

fn cmd_list(global_dir: &Path) -> Result<()> {
    if !global_dir.exists() {
        println!("No global mnem config found. Run `mnem integrate` to set one up.");
        return Ok(());
    }
    let reg = RepoRegistry::load(global_dir)?;
    if reg.repos.is_empty() {
        println!("No repos registered yet. Run `mnem init <path>` to register one.");
        return Ok(());
    }
    println!("Registered repos ({}):", reg.repos.len());
    for entry in &reg.repos {
        let marker = if entry.default { " *" } else { "  " };
        let label = entry
            .label
            .as_deref()
            .map(|l| format!("  [{l}]"))
            .unwrap_or_default();
        let exists = if entry.path.exists() {
            ""
        } else {
            "  (missing)"
        };
        println!("{marker} {}{label}{exists}", entry.path.display());
    }
    if let Some(d) = reg.default_repo() {
        println!("\n* = default ({})", d.path.display());
    }
    Ok(())
}

fn cmd_set_default(global_dir: &Path, path: &Path) -> Result<()> {
    if !global_dir.exists() {
        bail!(
            "No global mnem config found at {}. Run `mnem integrate` first.",
            global_dir.display()
        );
    }
    let mut reg = RepoRegistry::load(global_dir)?;
    let canon = path
        .canonicalize()
        .with_context(|| format!("resolving path {}", path.display()))?;
    reg.register(&canon, true);
    reg.save(global_dir)?;
    println!("Default repo set to: {}", canon.display());
    Ok(())
}

fn cmd_prune(global_dir: &Path) -> Result<()> {
    if !global_dir.exists() {
        println!("No global mnem config found. Nothing to prune.");
        return Ok(());
    }
    let mut reg = RepoRegistry::load(global_dir)?;
    let removed = reg.prune();
    if removed.is_empty() {
        println!("All registered repos still exist. Nothing pruned.");
    } else {
        reg.save(global_dir)?;
        println!(
            "Pruned {} stale entr{}:",
            removed.len(),
            if removed.len() == 1 { "y" } else { "ies" }
        );
        for p in &removed {
            println!("  - {}", p.display());
        }
    }
    Ok(())
}
