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
        // commands.
        let cwd = std::env::current_dir().context("cwd unreadable")?;
        let dir = cwd.join(repo::MNEM_DIR);
        std::fs::create_dir_all(&dir).with_context(|| format!("creating {}", dir.display()))?;
        Ok::<_, anyhow::Error>(dir)
    })?;
    let (bs, ohs) = repo::create_or_open_stores(&data_dir)?;

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

    // BUG-50: auto-advance HEAD to the CAR root so the imported data is
    // immediately accessible without a manual `mnem ref set HEAD <cid>`.
    //
    // Strategy: find the commit block with the greatest `time` among the
    // CAR's declared roots (same heuristic used by `mnem clone`). If the
    // CAR has no commit blocks in its roots list (e.g. a leaf-only archive)
    // we skip the HEAD update and print a note so the user knows why.
    let head_commit = find_head_commit_in_roots(&bs, &stats.roots)?;

    if let Some(head_cid) = &head_commit {
        let r_repo = ReadonlyRepo::init(bs.clone(), ohs.clone())?;
        let cfg = config::load(&data_dir)?;
        let author = config::author_string(&cfg);

        // Advance view.heads so `mnem retrieve` / `mnem status` see the
        // imported data immediately (mirrors the clone.rs HEAD-advance).
        r_repo.update_heads(head_cid.clone(), &author)?;

        println!("Imported {head_cid}. HEAD advanced.");
    } else if !stats.roots.is_empty() {
        println!(
            "note: no commit block found among CAR roots; HEAD not advanced. \
             Run `mnem ref set HEAD <cid>` manually if needed."
        );
    }

    Ok(())
}

/// Scan the CAR's declared root CIDs for a `_kind: \"commit\"` block and
/// return the one with the greatest `time` field. Returns `None` when no
/// commit block is present among the roots (e.g. leaf-only archive).
fn find_head_commit_in_roots(
    bs: &std::sync::Arc<dyn mnem_core::store::Blockstore>,
    roots: &[mnem_core::id::Cid],
) -> Result<Option<mnem_core::id::Cid>> {
    let mut best: Option<(u64, mnem_core::id::Cid)> = None;
    for root_cid in roots {
        let Some(bytes) = bs.get(root_cid)? else {
            continue;
        };
        let Ok(Ipld::Map(m)) = from_canonical_bytes::<Ipld>(&bytes) else {
            continue;
        };
        let Some(Ipld::String(kind)) = m.get("_kind") else {
            continue;
        };
        if kind != "commit" {
            continue;
        }
        let time = match m.get("time") {
            Some(Ipld::Integer(n)) => u64::try_from(*n).unwrap_or(0),
            _ => 0,
        };
        best = Some(match best {
            None => (time, root_cid.clone()),
            Some((t, _)) if time > t => (time, root_cid.clone()),
            Some(prev) => prev,
        });
    }
    Ok(best.map(|(_, c)| c))
}
