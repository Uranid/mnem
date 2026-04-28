use super::*;

use std::fs::File;
use std::io::BufReader;

/// `mnem import` arguments.
#[derive(clap::Args, Debug)]
#[command(after_long_help = "\
Examples:
  mnem import notes.car             # import into current repo
  cat notes.car | mnem import -     # read from stdin
  mnem init ~/restored && mnem -R ~/restored import notes.car
")]
pub(crate) struct Args {
    /// Input path for the CAR archive. Use `-` to read from stdin.
    pub path: String,
}

pub(crate) fn run(override_path: Option<&Path>, args: Args) -> Result<()> {
    let data_dir = repo::locate_data_dir(override_path).or_else(|_| {
        // Auto-create the repo on `mnem import` into an empty dir,
        // mirroring the "init-on-first-use" ergonomics of other
        // commands. The caller still has to `mnem ref set HEAD
        // <cid>` afterwards to surface the imported content
        // through the View; this subcommand only writes blocks.
        let cwd = std::env::current_dir().context("cwd unreadable")?;
        let dir = cwd.join(repo::MNEM_DIR);
        std::fs::create_dir_all(&dir).with_context(|| format!("creating {}", dir.display()))?;
        Ok::<_, anyhow::Error>(dir)
    })?;
    let (bs, _ohs) = repo::create_or_open_stores(&data_dir)?;

    // audit-2026-04-25 P1-5: normalise git-bash `/c/...` -> `c:/...`.
    let normalized = super::normalize_cli_path(&args.path);

    let stats = if normalized == "-" {
        let stdin = std::io::stdin();
        let mut lock = stdin.lock();
        mnem_transport::import(&mut lock, &*bs).with_context(|| {
            "reading CAR from stdin\n\
             hint: see docs/RUNBOOK.md#5-car-import-rejected for the error-variant \
             taxonomy (malformed CAR, CID mismatch, size cap, missing root, ...)."
                .to_string()
        })?
    } else {
        let path = Path::new(&normalized);
        let file = File::open(path).with_context(|| format!("opening {}", path.display()))?;
        let mut r = BufReader::new(file);
        mnem_transport::import(&mut r, &*bs).with_context(|| {
            format!(
                "reading CAR from {}\n\
                 hint: see docs/RUNBOOK.md#5-car-import-rejected for the error-variant \
                 taxonomy (malformed CAR, CID mismatch, size cap, missing root, ...).",
                path.display()
            )
        })?
    };

    let roots_summary = if stats.roots.is_empty() {
        "(no declared roots)".to_string()
    } else {
        stats
            .roots
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join(", ")
    };
    println!(
        "imported {} blocks, {} bytes from {} (roots: {})",
        stats.blocks, stats.bytes, normalized, roots_summary
    );
    Ok(())
}
