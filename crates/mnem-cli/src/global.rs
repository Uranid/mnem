//! `~/.mnemglobal` - device-wide anchor graph and repo registry.
//!
//! ## Layout
//!
//! ```text
//! ~/.mnemglobal/
//!   .mnem/        ← the global knowledge graph (always-on anchor)
//!   repos.toml    ← registry of all .mnem repos on this device
//! ```
//!
//! ## Cross-graph linking policy
//!
//! Native cross-graph edges are NOT added to the `.mnem` object format.
//! `mnem-core` is deliberately filesystem-free and embeddable (WASM, Python
//! FFI, Go FFI). Adding `dst_repo: PathBuf` to `Edge` would break that
//! contract. Instead:
//!
//! - **Phase 1 (this file):** global overlay - retrieval always searches
//!   `~/.mnemglobal/.mnem` alongside the current repo. Agents get global
//!   context everywhere with zero schema changes.
//! - **Phase 2 (future):** `_global_anchor` prop - when `mnem_resolve_or_create`
//!   matches an entity in the global graph, it stamps its CID as a plain
//!   string prop on the child-graph node. Soft hint, no core invariants broken.
//! - **Not planned:** `dst_repo` on `Edge`. The CID is already globally
//!   unique; the location problem is solved at the retrieval layer, not the
//!   storage layer.

use std::{
    fs,
    path::{Path, PathBuf},
    time::{SystemTime, UNIX_EPOCH},
};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

pub(crate) const DIR_NAME: &str = ".mnemglobal";

/// Returns the global graph parent directory.
///
/// Precedence:
/// 1. `MNEM_GLOBAL_DIR` env var (absolute path), useful on WSL where the
///    Windows home (`/mnt/c/Users/<user>/.mnemglobal`) differs from the
///    Linux home (`~/.mnemglobal`).
/// 2. `~/.mnemglobal` (OS home dir fallback, or `./.mnemglobal` if unknown).
pub(crate) fn default_dir() -> PathBuf {
    if let Ok(dir) = std::env::var("MNEM_GLOBAL_DIR") {
        return PathBuf::from(dir);
    }
    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(DIR_NAME)
}

// ---------- Registry types ----------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct RepoEntry {
    pub path: PathBuf,
    /// Unix timestamp of first registration.
    pub added_ts: u64,
    /// Unix timestamp of last use; updated on every `register` call.
    pub last_used_ts: u64,
    /// Exactly one entry should have `default = true` at any time.
    #[serde(default)]
    pub default: bool,
    /// Optional human-readable label (e.g. "personal notes", "work/project-a").
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
}

#[derive(Debug, Default, Serialize, Deserialize)]
pub(crate) struct RepoRegistry {
    #[serde(default)]
    pub repos: Vec<RepoEntry>,
}

impl RepoRegistry {
    pub(crate) fn load(global_dir: &Path) -> Result<Self> {
        let path = registry_path(global_dir);
        if !path.exists() {
            return Ok(Self::default());
        }
        let text =
            fs::read_to_string(&path).with_context(|| format!("reading {}", path.display()))?;
        toml::from_str(&text).with_context(|| format!("parsing {}", path.display()))
    }

    pub(crate) fn save(&self, global_dir: &Path) -> Result<()> {
        let path = registry_path(global_dir);
        let text = toml::to_string_pretty(self).context("serialising repos.toml")?;
        atomic_write(&path, text.as_bytes())
    }

    /// Register `repo_path`. Updates `last_used_ts` if already present.
    /// When `set_default` is true, clears all other default flags first.
    pub(crate) fn register(&mut self, repo_path: &Path, set_default: bool) {
        let canon = canonicalize_lax(repo_path);
        let now = now_ts();
        let already = self.repos.iter().any(|e| e.path == canon);
        if already {
            for e in self.repos.iter_mut() {
                if e.path == canon {
                    e.last_used_ts = now;
                }
                if set_default {
                    e.default = e.path == canon;
                }
            }
        } else {
            if set_default {
                for e in self.repos.iter_mut() {
                    e.default = false;
                }
            }
            self.repos.push(RepoEntry {
                path: canon,
                added_ts: now,
                last_used_ts: now,
                default: set_default,
                label: None,
            });
        }
    }

    /// Returns the explicit default, falling back to the most-recently-used entry.
    pub(crate) fn default_repo(&self) -> Option<&RepoEntry> {
        self.repos
            .iter()
            .find(|e| e.default)
            .or_else(|| self.repos.iter().max_by_key(|e| e.last_used_ts))
    }

    /// Removes entries whose paths no longer exist on disk.
    /// Returns the list of removed paths.
    pub(crate) fn prune(&mut self) -> Vec<PathBuf> {
        let mut removed = Vec::new();
        self.repos.retain(|e| {
            if e.path.exists() {
                true
            } else {
                removed.push(e.path.clone());
                false
            }
        });
        removed
    }
}

pub(crate) fn registry_path(global_dir: &Path) -> PathBuf {
    global_dir.join("repos.toml")
}

// ---------- Bootstrap ----------

/// Create `global_dir/` and initialise a `.mnem` graph inside it.
/// Returns `true` if freshly bootstrapped, `false` if already existed.
pub(crate) fn bootstrap(global_dir: &Path) -> Result<bool> {
    let mnem_dir = global_dir.join(crate::repo::MNEM_DIR);
    if mnem_dir.exists() {
        return Ok(false);
    }
    fs::create_dir_all(global_dir).with_context(|| format!("creating {}", global_dir.display()))?;
    crate::commands::init::init_mnem_dir(global_dir)
        .with_context(|| format!("initialising graph in {}", global_dir.display()))?;
    Ok(true)
}

// ---------- Best-effort helpers (called from init, never fatal) ----------

/// Register `repo_parent` in the global registry.
/// Silent on all errors - must never block `mnem init`.
pub(crate) fn register_repo(repo_parent: &Path) {
    let global_dir = default_dir();
    if !global_dir.exists() {
        return;
    }
    let Ok(mut reg) = RepoRegistry::load(&global_dir) else {
        return;
    };
    reg.register(repo_parent, false);
    let _ = reg.save(&global_dir);
}

// ---------- Internal helpers ----------

fn canonicalize_lax(path: &Path) -> PathBuf {
    path.canonicalize().unwrap_or_else(|_| path.to_path_buf())
}

fn atomic_write(path: &Path, data: &[u8]) -> Result<()> {
    let tmp = path.with_extension("toml.tmp");
    fs::write(&tmp, data).with_context(|| format!("writing {}", tmp.display()))?;
    fs::rename(&tmp, path).with_context(|| format!("renaming to {}", path.display()))?;
    Ok(())
}

fn now_ts() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}
