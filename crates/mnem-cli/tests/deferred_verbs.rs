//! Integration tests for verbs that were formerly deferred (EX_CONFIG stubs).
//!
//! Coverage here:
//!
//! - All formerly-deferred verbs appear in `mnem --help` so a Git user's
//!   `mnem <TAB>` completion shows the full vocabulary.
//! - `mnem fsck` exits 0 on a clean repo (real implementation).
//! - `mnem gc` exits 0 on a clean repo in dry-run mode (real implementation).
//! - `mnem revert` exits non-zero for a malformed CID string.
//! - `mnem revert` exits non-zero for a valid-format CID that is absent from
//!   the repo.
//! - `mnem revert` exits 0 when given the op CID of a real committed op.

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

// `mnem fsck`, `mnem gc`, and `mnem revert` are now real implementations.
// The old monolithic `revert_fsck_gc_all_exit_78` test has been replaced
// by the three targeted tests below.

#[test]
fn fsck_exits_zero_on_clean_repo() {
    let dir = TempDir::new().unwrap();
    mnem(dir.path(), &["init", dir.path().to_str().unwrap()])
        .assert()
        .success();
    mnem(dir.path(), &["fsck"]).assert().success();
}

#[test]
fn gc_exits_zero_on_clean_repo() {
    let dir = TempDir::new().unwrap();
    mnem(dir.path(), &["init", dir.path().to_str().unwrap()])
        .assert()
        .success();
    // dry-run (no --force): should always exit 0
    mnem(dir.path(), &["gc"]).assert().success();
}

#[test]
fn revert_bad_cid_exits_nonzero() {
    let dir = TempDir::new().unwrap();
    mnem(dir.path(), &["init", dir.path().to_str().unwrap()])
        .assert()
        .success();
    // An obviously invalid CID must produce a non-zero exit.
    mnem(dir.path(), &["revert", "badinput"]).assert().failure();
}

#[test]
fn revert_missing_cid_exits_nonzero() {
    let dir = TempDir::new().unwrap();
    mnem(dir.path(), &["init", dir.path().to_str().unwrap()])
        .assert()
        .success();
    // A syntactically valid CID that does not exist in this repo must also
    // produce a non-zero exit.  We use a well-known example CID from the
    // IPFS/CIDv1 spec; it will never appear in a freshly-initialised store.
    mnem(
        dir.path(),
        &[
            "revert",
            "bafybeigdyrzt5sfp7udm7hu76uh7y26nf3efuylqabf3oclgtqy55fbzdi",
        ],
    )
    .assert()
    .failure();
}

#[test]
fn revert_real_op_exits_zero() {
    let dir = TempDir::new().unwrap();
    mnem(dir.path(), &["init", dir.path().to_str().unwrap()])
        .assert()
        .success();
    // Create a real op by adding a node.
    mnem(
        dir.path(),
        &["add", "node", "--summary", "revert-me", "--no-embed"],
    )
    .assert()
    .success();
    // Grab the op CID from `mnem log -n 1` (line: "op <cid>").
    let log_out = mnem(dir.path(), &["log", "-n", "1"]).assert().success();
    let stdout = String::from_utf8_lossy(&log_out.get_output().stdout).to_string();
    let op_cid = stdout
        .lines()
        .find_map(|l| l.strip_prefix("op ").map(str::trim).map(str::to_string))
        .expect("mnem log -n 1 must emit an 'op <cid>' line");
    // Reverting a real op that added a node must exit 0.
    mnem(dir.path(), &["revert", &op_cid]).assert().success();
}
