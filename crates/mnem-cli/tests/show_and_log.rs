//! Integration tests for hardened `mnem show` and `mnem log`.
//!
//! - `show <op-cid>` still decodes ops (backward compat with the old
//!   behaviour)
//! - `show <node-cid>` decodes a node and pretty-prints it
//! - `log --format=json` emits one JSON object per line, parseable
//! - `log --oneline` emits one line per op

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
fn log_json_format_is_valid_jsonl() {
    let dir = TempDir::new().unwrap();
    mnem(dir.path(), &["init", dir.path().to_str().unwrap()])
        .assert()
        .success();
    // Two commits -> two ops in JSON lines.
    mnem(
        dir.path(),
        &["add", "node", "--summary", "one", "--no-embed"],
    )
    .assert()
    .success();
    mnem(
        dir.path(),
        &["add", "node", "--summary", "two", "--no-embed"],
    )
    .assert()
    .success();

    let out = mnem(dir.path(), &["log", "--format=json"])
        .assert()
        .success();
    let stdout = String::from_utf8_lossy(&out.get_output().stdout).to_string();
    let mut ops_seen = 0;
    for line in stdout.lines() {
        if line.trim().is_empty() {
            continue;
        }
        let v: serde_json::Value = serde_json::from_str(line).unwrap_or_else(|e| {
            panic!("log line must be valid JSON: {line:?} err={e}");
        });
        assert!(v.get("cid").is_some(), "log record must carry cid");
        assert!(v.get("time").is_some(), "log record must carry time");
        ops_seen += 1;
    }
    assert!(
        ops_seen >= 3,
        "expected at least 3 ops (init + 2 commits), saw {ops_seen}"
    );
}

#[test]
fn log_oneline_prints_one_line_per_op() {
    let dir = TempDir::new().unwrap();
    mnem(dir.path(), &["init", dir.path().to_str().unwrap()])
        .assert()
        .success();
    mnem(dir.path(), &["add", "node", "--summary", "x", "--no-embed"])
        .assert()
        .success();
    mnem(dir.path(), &["add", "node", "--summary", "y", "--no-embed"])
        .assert()
        .success();
    let out = mnem(dir.path(), &["log", "--oneline"]).assert().success();
    let stdout = String::from_utf8_lossy(&out.get_output().stdout).to_string();
    // 3 ops -> 3 non-empty lines.
    let lines: Vec<&str> = stdout.lines().filter(|l| !l.trim().is_empty()).collect();
    assert!(
        lines.len() >= 3,
        "oneline should emit one per op, got {lines:?}"
    );
}

#[test]
fn show_default_still_shows_op_head() {
    let dir = TempDir::new().unwrap();
    mnem(dir.path(), &["init", dir.path().to_str().unwrap()])
        .assert()
        .success();
    mnem(
        dir.path(),
        &["add", "node", "--summary", "hello", "--no-embed"],
    )
    .assert()
    .success();
    let out = mnem(dir.path(), &["show"]).assert().success();
    let stdout = String::from_utf8_lossy(&out.get_output().stdout).to_string();
    // Hardened show prints kind and size, not just op-specific fields.
    assert!(stdout.contains("kind"));
    assert!(stdout.contains("operation"));
}
