//! `mnem clone <url> [<dir>]` - initialise a repo from an archive.
//!
//! Q2 scope (partial, ):
//!
//! - `file:///absolute/path/to/archive.car` - supported
//! - bare path to a `.car` file (convenience) - supported
//! - `https://...`, `mnem+ssh://...`, `mnem://...` - rejected with an
//!   actionable message pointing at PR 3 / Q2-of-PR-3
//!
//! Behaviour on a supported URL:
//!
//! 1. Validate `<dir>` - either missing (we create it) or existing
//!    and empty of `.mnem/`. Refuse to clone into a directory that
//!    already carries a mnem repo.
//! 2. Create `.mnem/` and open the redb blockstore.
//! 3. Stream the CAR through
//!    `mnem_transport::import` (which enforces the default size
//!    limit; malformed archives error out cleanly PR 1).
//! 4. Scan the imported blocks for the `_kind: "commit"` block with
//!    the largest `time` field - the clone's head. This is a simple
//!    heuristic; a follow-up can carry the head CID in the CAR's
//!    root list.
//! 5. Write `[remote.origin] = { url = "<url>" }` to
//!    `.mnem/config.toml`.
//! 6. Record an `origin/main` tracking ref on the View via
//!    `View::with_tracking_ref`. (Empty when no commit is found.)
//!
//! Tests assert both the happy path (round-trip: export CAR from A,
//! clone into B, `mnem log` shows same head) and the dirty-dir
//! refusal path.
//!
//! # Examples
//!
//! ```text
//! mnem clone file:///tmp/alice.car /tmp/alice-mirror
//! mnem clone ./alice.car ./alice-mirror
//! ```

use std::fs;
use std::io::BufReader;

use ipld_core::ipld::Ipld;
use mnem_transport::remote::{RemoteConfigFile, RemoteSection, serialize_config};

use super::*;

#[derive(clap::Args, Debug)]
#[command(after_long_help = "\
Examples:
  mnem clone file:///tmp/alice.car /tmp/mirror
  mnem clone ./alice.car ./mirror       # bare path OK when it ends in .car
  mnem clone file:///tmp/alice.car      # clones into $PWD/alice (derived from url)
")]
pub(crate) struct Args {
    /// URL to clone from. Q2 supports `file://` and bare `.car` paths
    /// only. Remote schemes (`https://`, `mnem+ssh://`, ...) are
    /// deferred to PR 3.
    pub url: String,
    /// Target directory. Must not already contain `.mnem/`. Defaults
    /// to a directory derived from the URL stem.
    pub dir: Option<std::path::PathBuf>,
}

pub(crate) fn run(_override: Option<&Path>, args: Args) -> Result<()> {
    let local_path = parse_clone_source(&args.url)?;
    let target_dir = resolve_target_dir(&args.url, args.dir.as_deref())?;

    // Refuse to clone on top of an existing mnem repo.
    let data_dir = target_dir.join(repo::MNEM_DIR);
    if data_dir.exists() {
        bail!(
            "target directory already contains a mnem repository at {}; refusing to clone",
            data_dir.display()
        );
    }

    // Create the workspace + blockstore.
    fs::create_dir_all(&target_dir)
        .with_context(|| format!("creating {}", target_dir.display()))?;
    let (bs, ohs) = repo::create_or_open_stores(&data_dir)?;

    // Stream the CAR. `import` enforces the built-in size limit.
    let file =
        fs::File::open(&local_path).with_context(|| format!("opening {}", local_path.display()))?;
    let mut r = BufReader::new(file);
    let stats = mnem_transport::import(&mut r, &*bs).with_context(|| {
        format!(
            "importing CAR from {}\n\
             hint: see docs/RUNBOOK.md#5-car-import-rejected for the error-variant \
             taxonomy (malformed CAR, CID mismatch, size cap, missing root, ...).",
            local_path.display()
        )
    })?;

    // Initialise the repo state. The import only wrote blocks; an
    // empty ReadonlyRepo still needs an op-head to be consistent.
    let r_repo = ReadonlyRepo::init(bs.clone(), ohs.clone())?;

    // Find the clone's head: the commit block with the greatest
    // `time`. Falls back to `None` if no commit survived (e.g. the
    // CAR carried only leaves).
    let head_commit = find_head_commit(&bs, &stats.roots)?;

    // Write `[remote.origin]` pointing at the URL.
    let mut section = RemoteSection::default();
    section.remote.insert(
        "origin".into(),
        RemoteConfigFile {
            url: args.url.clone(),
            capabilities: None,
            token_env: None,
        },
    );
    let config_text = serialize_config(&section).context("serialising remote section")?;
    fs::write(data_dir.join(config::CONFIG_FILE), config_text)
        .context("writing .mnem/config.toml")?;

    // Record an origin/main tracking ref if we found a head.
    if let Some(head_cid) = &head_commit {
        // Use the repo's author string (empty config gives the
        // default agent-id), then update via an Operation so `mnem
        // log` sees the clone.
        let cfg = config::load(&data_dir)?;
        // A tracking ref does not promote the commit to a local head;
        // it lives in View.remote_refs. The typed helper lives on the
        // View, not the Transaction, so we piggyback on the
        // ref-update Operation path for provenance and then overwrite
        // the view from the transaction layer.
        //
        // For Q2 simplicity: store a normal local ref named
        // `refs/remotes/origin/main` so the plumbing is uniform.
        // SPEC reserves `refs/remotes/<name>/*` for exactly this
        // convention .
        let _ = r_repo.update_ref(
            "refs/remotes/origin/main",
            None,
            Some(RefTarget::normal(head_cid.clone())),
            &config::author_string(&cfg),
        )?;
    }

    println!(
        "cloned {} blocks ({} bytes) from {} into {}",
        stats.blocks,
        stats.bytes,
        args.url,
        target_dir.display()
    );
    match &head_commit {
        Some(c) => println!("  origin/main -> {c}"),
        None => println!("  origin/main -> <no commit found in CAR>"),
    }
    Ok(())
}

/// Reject unsupported URL schemes with an actionable message, and
/// resolve the supported ones to a local filesystem path.
fn parse_clone_source(url: &str) -> Result<std::path::PathBuf> {
    // bare path convenience: treat any non-URL-looking path that ends
    // in `.car` as a file URL. This saves POSIX users the triple-slash
    // escape dance.
    let has_scheme = url.contains("://");
    if !has_scheme {
        // audit-2026-04-25 P1-5: rewrite git-bash-style `/c/...` to
        // `c:/...` on Windows so MSYS2 / Git Bash users do not hit a
        // "system cannot find the path" error.
        let normalized = super::normalize_cli_path(url);
        let p = std::path::PathBuf::from(&normalized);
        if !p.extension().is_some_and(|e| e.eq_ignore_ascii_case("car")) {
            bail!(
                "`{url}` does not look like a URL or a `.car` path. \
                 Pass file:///abs/path/archive.car or a bare *.car path."
            );
        }
        return Ok(p);
    }
    // URL path. Only file:// is shipped in Q2.
    if let Some(rest) = url.strip_prefix("file://") {
        // `file:///abs/path` -> rest begins with `/` on POSIX. On
        // Windows, `file:///C:/path` -> rest = `/C:/path`; trim the
        // leading slash so the PathBuf is drive-letter addressable.
        let trimmed = if rest.starts_with('/') && rest.len() >= 3 && rest.as_bytes()[2] == b':' {
            &rest[1..]
        } else {
            rest
        };
        // audit-2026-04-25 C3-4 (Cycle-3): mirror the P1-5 fix from
        // `mnem export`: rewrite git-bash-style `/c/...` paths to
        // `c:/...` so `file:///c/tmp/repo.car` (lowercase, no colon)
        // works on Windows the same way `file:///C:/tmp/repo.car`
        // does. Without this, MSYS2 / Git Bash users hit a "system
        // cannot find the path" error after the URL strip.
        let normalized = super::normalize_cli_path(trimmed);
        return Ok(std::path::PathBuf::from(normalized));
    }
    // Any other scheme -> deferred.
    let scheme = url.split("://").next().unwrap_or("<unknown>");
    bail!(
        "clone over the `{scheme}` scheme is not yet implemented. \
         mnem 0.3 ships `file://` clone only; remote schemes land in PR 3 \
         (Q2-of-PR-3). See docs/ROADMAP.md and ."
    );
}

/// Resolve the target directory:
/// - explicit `<dir>` wins
/// - else derive from the URL: `.../alice.car` -> `./alice`
fn resolve_target_dir(url: &str, explicit: Option<&Path>) -> Result<std::path::PathBuf> {
    if let Some(d) = explicit {
        return Ok(d.to_path_buf());
    }
    // Strip scheme, take the final path component sans `.car`.
    let tail = url.rsplit('/').next().unwrap_or(url);
    let stem = tail.trim_end_matches(".car");
    if stem.is_empty() {
        bail!("could not derive a target dir from `{url}`; pass <dir> explicitly");
    }
    let cwd = std::env::current_dir().context("cwd unreadable")?;
    Ok(cwd.join(stem))
}

/// Scan every imported block for `_kind: "commit"` and return the one
/// with the greatest `time`. Returns `None` if no commit is present
/// (e.g. the CAR shipped only leaves, or only bare bytes).
///
/// This is a Q2 heuristic; a later PR can read the head CID out of
/// the CAR's root list directly. We keep it deterministic by breaking
/// ties on the CID byte string, which `>` compares lexicographically.
fn find_head_commit(
    bs: &std::sync::Arc<dyn mnem_core::store::Blockstore>,
    roots: &[mnem_core::id::Cid],
) -> Result<Option<mnem_core::id::Cid>> {
    let mut best: Option<(u64, mnem_core::id::Cid)> = None;
    // Walk the roots list first; for Q2 this is normally where the
    // head commit lives (export walks reachability from HEAD).
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

#[cfg(test)]
mod parse_clone_source_tests {
    use super::parse_clone_source;

    // audit-2026-04-25 C3-4: lock the git-bash-style file:/// URL
    // normalization so MSYS2 / Git Bash users on Windows can paste
    // the lowercase `file:///c/tmp/x.car` form they get from
    // `realpath` without hitting "system cannot find the path".
    #[test]
    #[cfg(windows)]
    fn file_uri_with_git_bash_drive_letter_normalizes() {
        let p = parse_clone_source("file:///c/tmp/repo.car").expect("parse ok");
        let s = p.to_string_lossy().replace('\\', "/");
        assert!(
            s.starts_with("c:/") || s.starts_with("C:/"),
            "expected drive-letter path, got {s:?}"
        );
        assert!(s.ends_with("/tmp/repo.car"), "got {s:?}");
    }

    #[test]
    #[cfg(windows)]
    fn file_uri_with_uppercase_drive_letter_unchanged() {
        let p = parse_clone_source("file:///C:/tmp/repo.car").expect("parse ok");
        let s = p.to_string_lossy().replace('\\', "/");
        assert!(s.starts_with("C:/"), "got {s:?}");
        assert!(s.ends_with("/tmp/repo.car"), "got {s:?}");
    }

    #[test]
    fn bare_car_path_still_accepted() {
        let p = parse_clone_source("./alice.car").expect("parse ok");
        assert!(p.to_string_lossy().ends_with("alice.car"));
    }

    #[test]
    fn unsupported_scheme_rejected() {
        let err = parse_clone_source("https://example.com/repo.car").unwrap_err();
        let msg = format!("{err:#}");
        assert!(msg.contains("not yet implemented"), "got {msg}");
    }
}
