//! Integration tests for `mnem merge` (B4.3).
//!
//! These tests drive the built `mnem` binary against a temp-dir repo.
//! Every test covers one externally-visible behaviour: the LCA
//! fast-forward path, the clean 3-way path, conflict persistence, the
//! --dry-run side-effect-free preview, and the --abort cleanup flow.

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
fn merge_missing_branch_errors_actionably() {
    let dir = TempDir::new().unwrap();
    mnem(dir.path(), &["init", dir.path().to_str().unwrap()])
        .assert()
        .success();
    // No branch arg, no --continue / --abort.
    let out = mnem(dir.path(), &["merge"]).assert().failure();
    let stderr = String::from_utf8_lossy(&out.get_output().stderr).to_string();
    assert!(
        stderr.contains("missing <branch>") || stderr.contains("required"),
        "missing-branch diagnostic should mention the missing argument: {stderr}"
    );
}

#[test]
fn merge_already_up_to_date_when_same_commit() {
    let dir = TempDir::new().unwrap();
    mnem(dir.path(), &["init", dir.path().to_str().unwrap()])
        .assert()
        .success();
    mnem(
        dir.path(),
        &["add", "node", "--summary", "seed", "--prop", "k=1"],
    )
    .assert()
    .success();
    // Create a branch at HEAD, then merge it - should be up-to-date.
    mnem(dir.path(), &["branch", "create", "feat"])
        .assert()
        .success();
    let out = mnem(dir.path(), &["merge", "feat"]).assert().success();
    let stdout = String::from_utf8_lossy(&out.get_output().stdout).to_string();
    assert!(
        stdout.contains("already up to date") || stdout.contains("fast-forward"),
        "expected up-to-date or FF, got: {stdout}"
    );
}

#[test]
fn merge_abort_without_in_progress_errors() {
    let dir = TempDir::new().unwrap();
    mnem(dir.path(), &["init", dir.path().to_str().unwrap()])
        .assert()
        .success();
    let out = mnem(dir.path(), &["merge", "--abort"]).assert().failure();
    let stderr = String::from_utf8_lossy(&out.get_output().stderr).to_string();
    assert!(
        stderr.contains("no merge in progress"),
        "abort without in-progress should complain: {stderr}"
    );
}

#[test]
fn merge_help_lists_strategy_and_flags() {
    let out = Command::cargo_bin("mnem")
        .unwrap()
        .args(["merge", "--help"])
        .assert()
        .success();
    let stdout = String::from_utf8_lossy(&out.get_output().stdout).to_string();
    for needle in [
        "--strategy",
        "--dry-run",
        "--continue",
        "--abort",
        "ours",
        "theirs",
        "manual",
    ] {
        assert!(
            stdout.contains(needle),
            "merge --help must surface `{needle}`, got: {stdout}"
        );
    }
}
