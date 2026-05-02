//! Integration tests for `mnem branch`.
//!
//! Drives the built `mnem` binary against a temp-dir repo:
//!
//! 1. `branch list` after `mnem init` shows the seeded `main` branch.
//! 2. `branch create` after `mnem init` succeeds (init seeds a commit).
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
fn branch_list_after_init_shows_main() {
    // `mnem init` seeds an anchor commit and a `main` branch ref.
    let dir = TempDir::new().unwrap();
    mnem(dir.path(), &["init", dir.path().to_str().unwrap()])
        .assert()
        .success();
    let out = mnem(dir.path(), &["branch", "list"]).assert().success();
    let stdout = String::from_utf8_lossy(&out.get_output().stdout).to_string();
    assert!(
        stdout.contains("main"),
        "expected main branch after init, got: {stdout}"
    );
}

#[test]
fn branch_create_after_init_succeeds() {
    // `mnem init` seeds an anchor commit, so `branch create` has a head
    // to point at immediately — no extra commits are needed.
    let dir = TempDir::new().unwrap();
    mnem(dir.path(), &["init", dir.path().to_str().unwrap()])
        .assert()
        .success();
    let out = mnem(dir.path(), &["branch", "create", "feature"])
        .assert()
        .success();
    let stdout = String::from_utf8_lossy(&out.get_output().stdout).to_string();
    assert!(
        stdout.starts_with("created branch feature ->"),
        "expected branch create to succeed, got: {stdout}"
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
