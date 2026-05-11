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
use mnem_core::HEADS_PREFIX;
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

    // Run the full clone body inside a closure so that any failure can
    // be caught uniformly and the partially-created `.mnem/` directory
    // cleaned up before returning the error to the caller (BUG-29).
    let result = (|| -> Result<()> {
        // Create the workspace + blockstore.
        fs::create_dir_all(&target_dir)
            .with_context(|| format!("creating {}", target_dir.display()))?;
        let (bs, ohs) = repo::create_or_open_stores(&data_dir)?;

        // Stream the CAR. `import` enforces the built-in size limit.
        let file = fs::File::open(&local_path)
            .with_context(|| format!("opening {}", local_path.display()))?;
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

        // Record an origin/main tracking ref if we found a head, then
        // advance the local branch (`refs/heads/main`) and `view.heads`
        // so the cloned data is immediately visible to `mnem retrieve`
        // and `mnem status` (fix for bug J6).
        if let Some(head_cid) = &head_commit {
            // Use the repo's author string (empty config gives the
            // default agent-id), then update via an Operation so `mnem
            // log` sees the clone.
            let cfg = config::load(&data_dir)?;
            let author = config::author_string(&cfg);

            // Step 1: remote tracking ref (informational only; does not
            // advance view.heads). For Q2 simplicity: store a normal
            // local ref named `refs/remotes/origin/main` so the plumbing
            // is uniform. SPEC reserves `refs/remotes/<name>/*` for
            // exactly this convention.
            let after_remote = r_repo.update_ref(
                "refs/remotes/origin/main",
                None,
                Some(RefTarget::normal(head_cid.clone())),
                &author,
            )?;

            // Step 2: create a local branch `refs/heads/main` pointing
            // at the cloned commit so `mnem branch list` and named-ref
            // resolution work identically to a freshly-committed repo.
            let after_local = after_remote.update_ref(
                &format!("{HEADS_PREFIX}main"),
                None,
                Some(RefTarget::normal(head_cid.clone())),
                &author,
            )?;

            // Step 3: advance view.heads so `head_commit()` is non-None
            // and the Prolly tree is reachable by `mnem retrieve`.
            // Without this step the blockstore has all the data but the
            // repo's working position stays at the genesis op (empty
            // heads), making every retrieval return 0 results.
            after_local.update_heads(head_cid.clone(), &author)?;
        }

        // If the clone source is a local mnem repo directory (not a CAR
        // file), copy the [embed] section from its config so `mnem retrieve`
        // works immediately in the clone without manual re-configuration.
        // When the source is a CAR archive we cannot recover the embedder
        // settings from the file itself, so we print an actionable note
        // instead.
        let embed_copied = try_copy_embed_config(&local_path, &data_dir);

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
        if !embed_copied {
            println!(
                "note: embedder config was not copied. Run `mnem config set embed.provider <provider>` \
                 to configure embeddings, or copy [embed] from the source repo's .mnem/config.toml manually."
            );
        }
        Ok(())
    })();

    // BUG-29: on any failure after `.mnem/` was (or may have been)
    // created, remove it so the user can retry without a manual
    // `rm -rf .mnem/` step.  We only clean up the `.mnem/` data
    // directory, not the parent `target_dir`, because that directory
    // may have been supplied by the user and pre-existed the clone.
    if result.is_err() {
        eprintln!("clone failed; removing partial .mnem/ directory");
        let _ = fs::remove_dir_all(&data_dir);
    }

    result
}

/// Try to copy the `[embed]` section from a source repo's
/// `.mnem/config.toml` into the clone's config. Returns `true` if an
/// embed config was successfully copied, `false` otherwise (source is
/// a CAR file, source has no embed config, or the read fails).
///
/// When the source path is a directory we attempt to read
/// `<source>/.mnem/config.toml` directly. When it is a file (e.g. a
/// `.car` archive) there is no config to read, so we return `false`
/// without emitting any error — the caller is expected to print a
/// user-facing note instead.
///
/// The copy is done by reading the already-written clone config as a
/// raw `toml::Value`, injecting the `[embed]` table from the source,
/// and writing back. This preserves the `[remote.origin]` entry (and
/// any other sections) that were already stored.
fn try_copy_embed_config(source: &std::path::Path, dest_data_dir: &std::path::Path) -> bool {
    // A CAR file (or any regular file) has no adjacent `.mnem/` dir
    // we can reach. Only directories are valid source repos.
    if !source.is_dir() {
        return false;
    }
    let src_config_path = source.join(repo::MNEM_DIR).join(config::CONFIG_FILE);
    let Ok(src_text) = fs::read_to_string(&src_config_path) else {
        return false;
    };
    let Ok(src_cfg) = toml::from_str::<config::Config>(&src_text) else {
        return false;
    };
    let Some(embed) = src_cfg.embed else {
        return false;
    };

    // Serialize just the embed section by building a temporary Config
    // containing only the embed field, then extracting the `embed`
    // TOML table from the serialised output.
    let embed_only = config::Config {
        embed: Some(embed),
        ..Default::default()
    };
    let Ok(embed_toml) = toml::to_string_pretty(&embed_only) else {
        return false;
    };
    let Ok(embed_value) = toml::from_str::<toml::Value>(&embed_toml) else {
        return false;
    };
    let Some(embed_table) = embed_value.get("embed").cloned() else {
        return false;
    };

    // Read the clone's config.toml as a raw Value so we can inject the
    // embed section without disturbing [remote.origin] or any other
    // top-level tables that already exist.
    let dest_config_path = dest_data_dir.join(config::CONFIG_FILE);
    let dest_text = fs::read_to_string(&dest_config_path).unwrap_or_default();
    let Ok(mut dest_root) = toml::from_str::<toml::Value>(&dest_text) else {
        return false;
    };
    let Some(dest_table) = dest_root.as_table_mut() else {
        return false;
    };
    dest_table.insert("embed".into(), embed_table);

    let Ok(out) = toml::to_string_pretty(&dest_root) else {
        return false;
    };
    fs::write(&dest_config_path, out).is_ok()
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

/// Determine the HEAD commit CID from the CAR's designated roots list.
///
/// The CAR header's `roots[0]` is the authoritative HEAD pointer: the
/// exporter always writes the HEAD commit CID as the single root (see
/// `mnem_transport::export`). We read `roots[0]` directly, verify it
/// decodes as a commit block, and return it.
///
/// Fallback (multi-root or non-commit root): if `roots[0]` does not
/// decode as a `_kind: "commit"` block (e.g. a future multi-root CAR
/// or a CAR carrying only leaf data), we apply the improved heuristic:
/// find all commit blocks that are never referenced as a `parent` by
/// another commit (i.e. true tips), then pick the one with the largest
/// `time`. This is strictly better than the old approach because it
/// ignores commits buried in the middle of a chain.
///
/// Returns `None` only if no commit block was found in the store at all.
fn find_head_commit(
    bs: &std::sync::Arc<dyn mnem_core::store::Blockstore>,
    roots: &[mnem_core::id::Cid],
) -> Result<Option<mnem_core::id::Cid>> {
    // --- Primary path: use roots[0] as the authoritative HEAD. ---
    if let Some(root_cid) = roots.first() {
        let Some(bytes) = bs.get(root_cid)? else {
            // roots[0] block missing — fall through to heuristic.
            return find_head_commit_heuristic(bs, roots);
        };
        if let Ok(Ipld::Map(m)) = from_canonical_bytes::<Ipld>(&bytes) {
            if matches!(m.get("_kind"), Some(Ipld::String(k)) if k == "commit") {
                // roots[0] is a commit: this is the designated HEAD.
                return Ok(Some(root_cid.clone()));
            }
        }
        // roots[0] is not a commit block (e.g. leaf-only CAR); fall through.
    }

    // --- Fallback: improved tip-finding heuristic. ---
    find_head_commit_heuristic(bs, roots)
}

/// Heuristic HEAD selection used when the CAR header root is absent or
/// not a commit. Scans `roots` for commit blocks that are not
/// referenced as `parent` by any other commit (true tips) and returns
/// the one with the greatest `time`. Falls back to `None` when no
/// commit block is present at all.
fn find_head_commit_heuristic(
    bs: &std::sync::Arc<dyn mnem_core::store::Blockstore>,
    roots: &[mnem_core::id::Cid],
) -> Result<Option<mnem_core::id::Cid>> {
    use std::collections::HashSet;

    // Collect all commit blocks reachable from the provided roots,
    // along with their `time` and `parent` CIDs.
    let mut commits: Vec<(mnem_core::id::Cid, u64)> = Vec::new();
    let mut referenced_as_parent: HashSet<mnem_core::id::Cid> = HashSet::new();

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
        commits.push((root_cid.clone(), time));
        // Track any parent CIDs so we can exclude non-tip commits.
        if let Some(Ipld::Link(parent_cid)) = m.get("parent") {
            if let Ok(p) = mnem_core::id::Cid::from_bytes(&parent_cid.to_bytes()) {
                referenced_as_parent.insert(p);
            }
        }
    }

    if commits.is_empty() {
        return Ok(None);
    }

    // Prefer tip commits (not referenced as a parent by anyone else).
    // Among those (or among all commits if none qualify), pick the
    // greatest `time`; break ties by CID byte string for determinism.
    let best = {
        let tips: Vec<&(mnem_core::id::Cid, u64)> = commits
            .iter()
            .filter(|(cid, _)| !referenced_as_parent.contains(cid))
            .collect();
        let candidates: &[&(mnem_core::id::Cid, u64)] = if tips.is_empty() {
            // No true tips found; use all commits as candidates.
            &commits.iter().collect::<Vec<_>>()
        } else {
            &tips
        };
        candidates
            .iter()
            .max_by(|(a_cid, a_time), (b_cid, b_time)| {
                a_time
                    .cmp(b_time)
                    .then_with(|| a_cid.to_bytes().cmp(&b_cid.to_bytes()))
            })
            .map(|(cid, _)| (*cid).clone())
    };

    Ok(best)
}

#[cfg(test)]
mod find_head_commit_tests {
    use std::collections::BTreeMap;
    use std::sync::Arc;

    use ipld_core::ipld::Ipld;
    use mnem_core::codec::{hash_to_cid, to_canonical_bytes};
    use mnem_core::store::{Blockstore, MemoryBlockstore};

    use super::find_head_commit;

    /// Build a minimal `_kind: "commit"` block with the given `time`,
    /// insert it into `bs`, and return its CID.
    fn make_commit(bs: &MemoryBlockstore, time: u64) -> mnem_core::id::Cid {
        let mut m = BTreeMap::new();
        m.insert("_kind".to_string(), Ipld::String("commit".to_string()));
        m.insert("time".to_string(), Ipld::Integer(i128::from(time)));
        let ipld = Ipld::Map(m);
        let bytes = to_canonical_bytes(&ipld).unwrap();
        let (_, cid) = hash_to_cid(&ipld).unwrap();
        bs.put(cid.clone(), bytes).unwrap();
        cid
    }

    /// BUG-30: `find_head_commit` must use `roots[0]` as the
    /// authoritative HEAD, not the commit with the largest `time`.
    ///
    /// Arrange two commits: one with a smaller time (the real HEAD,
    /// placed at roots[0]) and one with a larger time (a stale branch
    /// tip). The function must return roots[0], not the high-time one.
    #[test]
    fn uses_roots0_not_largest_time() {
        let inner = MemoryBlockstore::new();

        // HEAD commit has time=1 (smaller).
        let head_cid = make_commit(&inner, 1);
        // A second commit has time=9999 (larger) — it would win the old
        // largest-time heuristic, but it is NOT the designated head.
        let _stale_cid = make_commit(&inner, 9999);

        let bs: Arc<dyn Blockstore> = Arc::new(inner);

        // roots[0] designates the real HEAD.
        let roots = vec![head_cid.clone()];
        let result = find_head_commit(&bs, &roots).unwrap();
        assert_eq!(
            result,
            Some(head_cid),
            "find_head_commit must return roots[0], not the commit with the largest time"
        );
    }

    /// When roots[0] is absent from the blockstore (malformed CAR),
    /// the heuristic fallback must still return a commit if any are
    /// present in roots.
    #[test]
    fn fallback_when_root_missing_from_blockstore() {
        let inner = MemoryBlockstore::new();
        let commit_cid = make_commit(&inner, 42);

        let bs: Arc<dyn Blockstore> = Arc::new(inner);

        // roots[0] is a random CID not in the store.
        use mnem_core::id::{CODEC_DAG_CBOR, Cid, Multihash};
        let phantom = Cid::new(CODEC_DAG_CBOR, Multihash::sha2_256(b"not-stored"));

        let roots = vec![phantom, commit_cid.clone()];
        let result = find_head_commit(&bs, &roots).unwrap();
        assert_eq!(
            result,
            Some(commit_cid),
            "heuristic fallback must return the available commit when roots[0] is missing"
        );
    }
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
