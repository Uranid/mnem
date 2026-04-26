//! Integration tests for `mnem branch`.
//!
//! Drives the built `mnem` binary against a temp-dir repo:
//!
//! 1. `branch list` on a fresh repo prints `<no branches>`.
//! 2. `branch create` without any commits yet errors actionably.
//! 3. Create + list + delete round-trip against a committed head.
//! 4. `branch create` refuses to overwrite an existing branch.

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
fn branch_list_empty_repo_prints_no_branches() {
    let dir = TempDir::new().unwrap();
    mnem(dir.path(), &["init", dir.path().to_str().unwrap()])
        .assert()
        .success();
    let out = mnem(dir.path(), &["branch", "list"]).assert().success();
    let stdout = String::from_utf8_lossy(&out.get_output().stdout).to_string();
    assert!(
        stdout.contains("<no branches>"),
        "expected no-branches marker, got: {stdout}"
    );
}

#[test]
fn branch_create_without_commits_fails() {
    // A fresh repo has no heads; `branch create` without --from has
    // nothing to point at, so it must error (and NOT pick a ghost
    // CID).
    let dir = TempDir::new().unwrap();
    mnem(dir.path(), &["init", dir.path().to_str().unwrap()])
        .assert()
        .success();
    let out = mnem(dir.path(), &["branch", "create", "feature"])
        .assert()
        .failure();
    let stderr = String::from_utf8_lossy(&out.get_output().stderr).to_string();
    assert!(
        stderr.contains("no commits yet"),
        "expected actionable no-commits message, got: {stderr}"
    );
}

#[test]
fn branch_create_list_delete_round_trip() {
    let dir = TempDir::new().unwrap();
    mnem(dir.path(), &["init", dir.path().to_str().unwrap()])
        .assert()
        .success();
    // Give the repo a head commit.
    mnem(
        dir.path(),
        &["add", "node", "--summary", "first", "--prop", "kind=doc"],
    )
    .assert()
    .success();

    // Create the branch against the default head.
    let created = mnem(dir.path(), &["branch", "create", "feature/x"])
        .assert()
        .success();
    let stdout = String::from_utf8_lossy(&created.get_output().stdout).to_string();
    assert!(
        stdout.starts_with("created branch feature/x ->"),
        "create stdout: {stdout}"
    );

    // List should surface the new branch (with the star on the head).
    let listed = mnem(dir.path(), &["branch", "list"]).assert().success();
    let listed_out = String::from_utf8_lossy(&listed.get_output().stdout).to_string();
    assert!(
        listed_out.contains("feature/x"),
        "list stdout must mention branch, got: {listed_out}"
    );
    assert!(
        listed_out.contains('*'),
        "branch at head should be starred, got: {listed_out}"
    );

    // Creating the same branch again must fail.
    let dup = mnem(dir.path(), &["branch", "create", "feature/x"])
        .assert()
        .failure();
    let dup_err = String::from_utf8_lossy(&dup.get_output().stderr).to_string();
    assert!(
        dup_err.contains("already exists"),
        "duplicate-branch error expected: {dup_err}"
    );

    // Delete cleans up.
    mnem(dir.path(), &["branch", "delete", "feature/x"])
        .assert()
        .success();
    let after = mnem(dir.path(), &["branch", "list"]).assert().success();
    let after_out = String::from_utf8_lossy(&after.get_output().stdout).to_string();
    assert!(
        !after_out.contains("feature/x"),
        "deleted branch must be gone, got: {after_out}"
    );
}

#[test]
fn branch_delete_missing_fails() {
    let dir = TempDir::new().unwrap();
    mnem(dir.path(), &["init", dir.path().to_str().unwrap()])
        .assert()
        .success();
    let out = mnem(dir.path(), &["branch", "delete", "nope"])
        .assert()
        .failure();
    let stderr = String::from_utf8_lossy(&out.get_output().stderr).to_string();
    assert!(
        stderr.contains("does not exist"),
        "missing-branch error expected: {stderr}"
    );
}
