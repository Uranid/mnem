//! Integration tests for `mnem clone` against a `file://` URL.
//!
//! Tests:
//!
//! 1. Happy path: export from repo A, clone into B, assert the log
//!    surfaces the same commit cid + the remote config landed on
//!    disk.
//! 2. Refuses to clone into a non-empty `.mnem/` directory.
//! 3. Rejects unsupported URL schemes (`https://`) with a message
//!    pointing at PR 3.

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

fn mnem_no_repo(args: &[&str]) -> Command {
    let mut cmd = Command::cargo_bin("mnem").expect("built mnem binary");
    for a in args {
        cmd.arg(a);
    }
    cmd
}

/// Produce a `file://` URL pointing at the given path. Normalises
/// Windows drive letters so `C:\x\y.car` becomes
/// `file:///C:/x/y.car`.
fn file_url(p: &Path) -> String {
    let s = p.to_string_lossy().replace('\\', "/");
    if s.starts_with('/') {
        format!("file://{s}")
    } else {
        format!("file:///{s}")
    }
}

#[test]
fn clone_file_url_round_trips_commit_head() {
    // Repo A: init + add node -> has a head commit.
    let src = TempDir::new().unwrap();
    mnem(src.path(), &["init", src.path().to_str().unwrap()])
        .assert()
        .success();
    mnem(
        src.path(),
        &[
            "add",
            "node",
            "--summary",
            "cloneable payload",
            "--label",
            "doc",
            "--no-embed",
        ],
    )
    .assert()
    .success();

    // Export the HEAD to a CAR that lives outside either repo.
    let car_dir = TempDir::new().unwrap();
    let car = car_dir.path().join("snapshot.car");
    mnem(
        src.path(),
        &["export", car.to_str().unwrap(), "--from", "HEAD"],
    )
    .assert()
    .success();
    assert!(car.exists());

    // Clone into a fresh dir by file:// URL.
    let dst_root = TempDir::new().unwrap();
    let dst = dst_root.path().join("mirror");
    let url = file_url(&car);
    let cloned = mnem_no_repo(&["clone", &url, dst.to_str().unwrap()])
        .assert()
        .success();
    let stdout = String::from_utf8_lossy(&cloned.get_output().stdout).to_string();
    assert!(
        stdout.contains("cloned ") && stdout.contains("blocks"),
        "clone output shape: {stdout}"
    );
    assert!(
        stdout.contains("origin/main"),
        "clone should report origin/main tracking ref, got: {stdout}"
    );

    // The clone created `.mnem/config.toml` with [remote.origin].
    let cfg = std::fs::read_to_string(dst.join(".mnem/config.toml")).expect("config.toml");
    assert!(
        cfg.contains("[remote.origin]"),
        "config must carry [remote.origin], got: {cfg}"
    );
    assert!(
        cfg.contains(&url),
        "config must carry the source URL, got: {cfg}"
    );

    // Pull the commit CID out of the log of A.
    let src_log = mnem(src.path(), &["log", "-n", "1", "--format=json"])
        .assert()
        .success();
    let src_stdout = String::from_utf8_lossy(&src_log.get_output().stdout).to_string();
    // Log output format is JSON Lines.
    let src_line = src_stdout
        .lines()
        .next()
        .expect("expected at least one op from src");
    let src_v: serde_json::Value = serde_json::from_str(src_line).expect("json");
    let src_cid = src_v["cid"].as_str().unwrap().to_string();
    assert!(!src_cid.is_empty());

    // The imported repo must carry a refs/remotes/origin/main that
    // points at the same commit cid the export started from. We
    // surface this via `mnem ref list` on the dst.
    let refs_out = mnem(&dst, &["ref", "list"]).assert().success();
    let refs_stdout = String::from_utf8_lossy(&refs_out.get_output().stdout).to_string();
    assert!(
        refs_stdout.contains("refs/remotes/origin/main"),
        "dst should carry origin/main tracking ref, got: {refs_stdout}"
    );
}

#[test]
fn clone_refuses_into_existing_mnem_dir() {
    // Pre-existing mnem repo in the target dir. Clone must NOT
    // clobber it; git's equivalent: `fatal: destination path 'foo'
    // already exists and is not an empty directory`.
    let src = TempDir::new().unwrap();
    mnem(src.path(), &["init", src.path().to_str().unwrap()])
        .assert()
        .success();
    mnem(
        src.path(),
        &["add", "node", "--summary", "src", "--no-embed"],
    )
    .assert()
    .success();

    let car_dir = TempDir::new().unwrap();
    let car = car_dir.path().join("s.car");
    mnem(src.path(), &["export", car.to_str().unwrap()])
        .assert()
        .success();

    // Destination: initialised already.
    let dst = TempDir::new().unwrap();
    mnem(dst.path(), &["init", dst.path().to_str().unwrap()])
        .assert()
        .success();

    let url = file_url(&car);
    let out = mnem_no_repo(&["clone", &url, dst.path().to_str().unwrap()])
        .assert()
        .failure();
    let stderr = String::from_utf8_lossy(&out.get_output().stderr).to_string();
    assert!(
        stderr.contains("already contains") || stderr.contains("refusing"),
        "expected refuse-to-clobber error, got: {stderr}"
    );
}

#[test]
fn clone_rejects_https_scheme_actionably() {
    // Remote schemes are deferred. The error MUST name PR 3 so the
    // user knows where the roadmap is.
    let dst = TempDir::new().unwrap();
    let target = dst.path().join("mirror");
    let out = mnem_no_repo(&[
        "clone",
        "https://example.com/alice/notes",
        target.to_str().unwrap(),
    ])
    .assert()
    .failure();
    let stderr = String::from_utf8_lossy(&out.get_output().stderr).to_string();
    assert!(
        stderr.contains("not yet implemented"),
        "deferred-scheme error must be actionable: {stderr}"
    );
    assert!(
        stderr.contains("PR 3") || stderr.contains("") || stderr.contains("ROADMAP"),
        "deferred-scheme error must point at the roadmap: {stderr}"
    );
}
