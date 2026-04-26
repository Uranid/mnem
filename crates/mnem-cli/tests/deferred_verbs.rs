//! Integration tests for the deferred verbs.
//!
//! Each deferred verb MUST:
//!
//! - appear in `mnem --help` so a Git user's `mnem <TAB>` completion
//!   shows the full vocabulary
//! - exit with code 78 (EX_CONFIG)
//! - print an error that points at + `docs/ROADMAP.md`

use std::path::Path;
use std::process::Command;

use assert_cmd::prelude::*;
use tempfile::TempDir;

fn mnem(repo: &Path, args: &[&str]) -> Command {
    let mut cmd = Command::cargo_bin("mnem").expect("built mnem binary");
    cmd.current_dir(repo);
    cmd.arg("-R").arg(repo);
    for a in args {
        cmd.arg(a);
    }
    cmd
}

#[test]
fn deferred_verbs_surface_in_help() {
    // --help is top-level; no repo needed.
    let out = Command::cargo_bin("mnem")
        .unwrap()
        .arg("--help")
        .assert()
        .success();
    let stdout = String::from_utf8_lossy(&out.get_output().stdout).to_string();
    for verb in [
        "fetch", "push", "pull", "merge", "revert", "fsck", "gc", "branch", "blame", "cat-file",
        "remote", "clone",
    ] {
        assert!(
            stdout.contains(verb),
            "--help must list `{verb}`, got: {stdout}"
        );
    }
}

// `fetch` / `push` / `pull` graduated out of the deferred list in
// 0.1.0 . Their EX_CONFIG-stub tests moved to
// `tests/remote_verbs.rs`; `mnem fetch origin` without a configured
// remote now errors with a config-file not-found message, and `mnem
// push` without a remote likewise errors. The verbs below still
// 78-stub.

// `mnem merge` used to be an EX_CONFIG stub; B4.3 ships a real
// implementation. See `tests/merge.rs` for the live-merge coverage
// (LCA fast-forward, clean 3-way, conflict persistence, --continue /
// --abort / --dry-run flows).

#[test]
fn revert_fsck_gc_all_exit_78() {
    let dir = TempDir::new().unwrap();
    mnem(dir.path(), &["init", dir.path().to_str().unwrap()])
        .assert()
        .success();
    for args in [vec!["revert", "bafk"], vec!["fsck"], vec!["gc"]] {
        let out = mnem(dir.path(), &args).assert().failure();
        let code = out.get_output().status.code();
        assert_eq!(
            code,
            Some(78),
            "verb {args:?} must exit 78 (EX_CONFIG), got {code:?}"
        );
    }
}
