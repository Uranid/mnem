//! Dataset cache + sha256 verification + download.
//!
//! Datasets land under `~/.mnem/bench-data/<bench>/<filename>`.
//! On `fetch`, the URL is downloaded with `ureq`, hashed with
//! sha256, and compared against a hardcoded expected digest. A
//! mismatch leaves the file on disk with a `.bad` suffix so the
//! operator can inspect it before deleting.

pub mod convomem;
pub mod locomo;
pub mod longmemeval;
pub mod membench;

use std::fs;
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow, bail};
use sha2::{Digest, Sha256};

use crate::bench::Bench;

/// Static descriptor for one dataset file.
#[derive(Clone, Debug)]
pub struct DatasetSpec {
    /// Bench this file feeds.
    pub bench: Bench,
    /// Filename under `~/.mnem/bench-data/<bench>/`.
    pub filename: &'static str,
    /// Direct download URL. Picked to be a bare-bytes endpoint
    /// (HuggingFace `resolve/main/...`) so we do not need a JSON
    /// parser to find the artefact.
    pub url: &'static str,
    /// Expected sha256 of the downloaded bytes (lower-case hex).
    /// Empty string disables the check (used during dev only).
    pub sha256: &'static str,
    /// Approximate bytes (for the progress bar baseline).
    pub bytes: u64,
}

/// Look up the canonical spec for a bench.
#[must_use]
pub fn spec_for(bench: Bench) -> Option<DatasetSpec> {
    match bench {
        Bench::LongMemEval => Some(longmemeval::SPEC),
        Bench::Locomo => Some(locomo::SPEC),
        Bench::Convomem => Some(convomem::SPEC),
        Bench::MembenchSimpleRoles => Some(membench::SIMPLE_ROLES_SPEC),
        Bench::MembenchHighlevelMovie => Some(membench::HIGHLEVEL_MOVIE_SPEC),
        _ => None,
    }
}

/// Resolve the cache directory for `bench`. Creates it if missing.
pub fn cache_dir_for(bench: Bench) -> Result<PathBuf> {
    let base = bench_data_root()?;
    let dir = base.join(bench.metadata().id);
    fs::create_dir_all(&dir).with_context(|| format!("creating {}", dir.display()))?;
    Ok(dir)
}

/// `~/.mnem/bench-data/`. Honours `MNEM_BENCH_DATA` for tests.
pub fn bench_data_root() -> Result<PathBuf> {
    if let Ok(p) = std::env::var("MNEM_BENCH_DATA") {
        return Ok(PathBuf::from(p));
    }
    let dirs = directories::BaseDirs::new()
        .ok_or_else(|| anyhow!("HOME / USERPROFILE unset; cannot resolve ~/.mnem"))?;
    Ok(dirs.home_dir().join(".mnem").join("bench-data"))
}

/// Resolve the path to the cached dataset file for `bench`. Does
/// NOT download; use [`fetch`] for that.
pub fn cached_path(bench: Bench) -> Result<PathBuf> {
    let spec =
        spec_for(bench).ok_or_else(|| anyhow!("no dataset spec for {}", bench.metadata().id))?;
    Ok(cache_dir_for(bench)?.join(spec.filename))
}

/// Whether the cached dataset for `bench` exists AND verifies
/// against its expected sha256. False if the file is absent, the
/// hash does not match, or the spec is empty.
pub fn is_cached(bench: Bench) -> bool {
    let Ok(p) = cached_path(bench) else {
        return false;
    };
    let Some(spec) = spec_for(bench) else {
        return false;
    };
    if !p.is_file() {
        return false;
    }
    if spec.sha256.is_empty() {
        return true;
    }
    sha256_file(&p)
        .map(|h| h.eq_ignore_ascii_case(spec.sha256))
        .unwrap_or(false)
}

/// Fetch the dataset for `bench` into the cache. Idempotent: if
/// the cached file already verifies, returns the path immediately.
///
/// `progress_cb` is called with `(downloaded_bytes, total_bytes)`
/// every ~64KB. Pass a no-op closure when running headless.
pub fn fetch<F: FnMut(u64, u64)>(
    bench: Bench,
    skip_cached: bool,
    mut progress_cb: F,
) -> Result<PathBuf> {
    // ConvoMem is multi-shard. Walk the bundled manifest, merge,
    // emit a single canonical blob. The per-shard cache lives under
    // `cache_dir/shards/`.
    if matches!(bench, Bench::Convomem) {
        let dir = cache_dir_for(bench)?;
        let dst = dir.join(convomem::SPEC.filename);
        if skip_cached && dst.is_file() {
            return Ok(dst);
        }
        return convomem::fetch_into(&dir);
    }
    let spec =
        spec_for(bench).ok_or_else(|| anyhow!("no dataset spec for {}", bench.metadata().id))?;
    let dst = cached_path(bench)?;
    if skip_cached && is_cached(bench) {
        return Ok(dst);
    }
    if dst.is_file() && !spec.sha256.is_empty() {
        let actual = sha256_file(&dst)?;
        if actual.eq_ignore_ascii_case(spec.sha256) {
            return Ok(dst);
        }
        // Stale or corrupt - keep the bytes for forensics.
        let bad = dst.with_extension("bad");
        let _ = fs::rename(&dst, &bad);
    }

    // Stream the download to a temp file, sha as we go, then
    // rename atomically.
    let tmp = dst.with_extension("part");
    let resp = ureq::get(spec.url)
        .call()
        .with_context(|| format!("GET {}", spec.url))?;
    let total: u64 = resp
        .header("content-length")
        .and_then(|s| s.parse().ok())
        .unwrap_or(spec.bytes);

    let mut reader = resp.into_reader();
    let mut file = fs::File::create(&tmp).with_context(|| format!("creating {}", tmp.display()))?;
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; 64 * 1024];
    let mut done = 0u64;
    loop {
        let n = match reader.read(&mut buf) {
            Ok(0) => break,
            Ok(n) => n,
            Err(e) if e.kind() == io::ErrorKind::Interrupted => continue,
            Err(e) => return Err(e).context("download read"),
        };
        file.write_all(&buf[..n]).context("write to disk")?;
        hasher.update(&buf[..n]);
        done = done.saturating_add(n as u64);
        progress_cb(done, total);
    }
    file.flush().ok();
    drop(file);

    let actual = hex::encode(hasher.finalize());
    if !spec.sha256.is_empty() && !actual.eq_ignore_ascii_case(spec.sha256) {
        let bad = dst.with_extension("bad");
        fs::rename(&tmp, &bad).ok();
        bail!(
            "sha256 mismatch for {}: expected {}, got {}. file kept at {}",
            spec.filename,
            spec.sha256,
            actual,
            bad.display()
        );
    }
    fs::rename(&tmp, &dst)
        .with_context(|| format!("renaming {} -> {}", tmp.display(), dst.display()))?;
    Ok(dst)
}

/// Hash a file with sha256, returning lower-case hex.
pub fn sha256_file(p: &Path) -> Result<String> {
    let mut f = fs::File::open(p).with_context(|| format!("opening {}", p.display()))?;
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; 64 * 1024];
    loop {
        let n = f
            .read(&mut buf)
            .with_context(|| format!("reading {}", p.display()))?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(hex::encode(hasher.finalize()))
}
