//! Integration tests for `mnem tag`.
//!
//! Drives the built `mnem` binary against a temp-dir repo:
//!
//! 1. `tag list` on a fresh repo (after `mnem init`) shows no tags.
//! 2. `tag create` + `tag list` round-trip: created tag appears in list.
//! 3. `tag create` refuses to overwrite an existing tag.
//! 4. `tag delete` removes the tag; subsequent list no longer shows it.
//! 5. `tag delete` on a missing tag returns a non-zero exit code.

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

/// `mnem tag list` on a freshly-initialized repo (which has no tags)
/// must succeed and print the sentinel "<no tags>" line.
#[test]
fn tag_list_empty_after_init() {
    let dir = TempDir::new().unwrap();
    mnem(dir.path(), &["init", dir.path().to_str().unwrap()])
        .assert()
        .success();
    let out = mnem(dir.path(), &["tag", "list"]).assert().success();
    let stdout = String::from_utf8_lossy(&out.get_output().stdout).to_string();
    assert!(
        stdout.contains("<no tags>"),
        "expected '<no tags>' on empty repo, got: {stdout}"
    );
}

/// Create a tag pointing at HEAD and verify `tag list` surfaces it.
#[test]
fn tag_create_and_list_round_trip() {
    let dir = TempDir::new().unwrap();
    mnem(dir.path(), &["init", dir.path().to_str().unwrap()])
        .assert()
        .success();

    // Create a tag against the seeded head commit.
    let created = mnem(dir.path(), &["tag", "create", "v1.0"])
        .assert()
        .success();
    let create_out = String::from_utf8_lossy(&created.get_output().stdout).to_string();
    assert!(
        create_out.starts_with("created tag v1.0 ->"),
        "expected create output to start with 'created tag v1.0 ->', got: {create_out}"
    );

    // The tag must appear in the list output.
    let listed = mnem(dir.path(), &["tag", "list"]).assert().success();
    let list_out = String::from_utf8_lossy(&listed.get_output().stdout).to_string();
    assert!(
        list_out.contains("v1.0"),
        "expected 'v1.0' in tag list output, got: {list_out}"
    );
    assert!(
        !list_out.contains("<no tags>"),
        "should not show '<no tags>' after a tag was created, got: {list_out}"
    );
}

/// Creating the same tag twice must fail with an actionable error.
#[test]
fn tag_create_duplicate_fails() {
    let dir = TempDir::new().unwrap();
    mnem(dir.path(), &["init", dir.path().to_str().unwrap()])
        .assert()
        .success();

    mnem(dir.path(), &["tag", "create", "v2.0"])
        .assert()
        .success();

    let dup = mnem(dir.path(), &["tag", "create", "v2.0"])
        .assert()
        .failure();
    let stderr = String::from_utf8_lossy(&dup.get_output().stderr).to_string();
    assert!(
        stderr.contains("already exists"),
        "expected 'already exists' error for duplicate tag, got: {stderr}"
    );
}

/// Full create + list + delete round-trip.
#[test]
fn tag_create_list_delete_round_trip() {
    let dir = TempDir::new().unwrap();
    mnem(dir.path(), &["init", dir.path().to_str().unwrap()])
        .assert()
        .success();

    // Add a commit so HEAD is non-trivial.
    mnem(
        dir.path(),
        &["add", "node", "--summary", "first", "--prop", "kind=doc"],
    )
    .assert()
    .success();

    // Create the tag.
    mnem(dir.path(), &["tag", "create", "v0.1"])
        .assert()
        .success();

    // Verify it appears in the list.
    let listed = mnem(dir.path(), &["tag", "list"]).assert().success();
    let list_out = String::from_utf8_lossy(&listed.get_output().stdout).to_string();
    assert!(
        list_out.contains("v0.1"),
        "tag must appear after create, got: {list_out}"
    );

    // Delete the tag.
    let deleted = mnem(dir.path(), &["tag", "delete", "v0.1"])
        .assert()
        .success();
    let del_out = String::from_utf8_lossy(&deleted.get_output().stdout).to_string();
    assert!(
        del_out.contains("deleted tag v0.1"),
        "expected 'deleted tag v0.1' in output, got: {del_out}"
    );

    // The tag must be gone from the list.
    let after = mnem(dir.path(), &["tag", "list"]).assert().success();
    let after_out = String::from_utf8_lossy(&after.get_output().stdout).to_string();
    assert!(
        !after_out.contains("v0.1"),
        "deleted tag must not appear in list, got: {after_out}"
    );
    assert!(
        after_out.contains("<no tags>"),
        "list must show '<no tags>' after all tags deleted, got: {after_out}"
    );
}

/// `tag delete` on a tag that does not exist must exit non-zero with
/// a descriptive error.
#[test]
fn tag_delete_missing_fails() {
    let dir = TempDir::new().unwrap();
    mnem(dir.path(), &["init", dir.path().to_str().unwrap()])
        .assert()
        .success();

    let out = mnem(dir.path(), &["tag", "delete", "nonexistent"])
        .assert()
        .failure();
    let stderr = String::from_utf8_lossy(&out.get_output().stderr).to_string();
    assert!(
        stderr.contains("does not exist"),
        "expected 'does not exist' error for missing tag, got: {stderr}"
    );
}
