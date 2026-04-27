//! Integration tests for `mnem cat-file`.
//!
//! Tests:
//!
//! 1. `cat-file <op-cid>` emits the raw DAG-CBOR bytes; the first
//!    byte is a CBOR map-prefix (0xa0..0xbf).
//! 2. `cat-file <op-cid> --json` emits valid JSON containing the
//!    `_kind` discriminator.
//! 3. `cat-file <unknown-cid>` fails with an actionable message.

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

/// Walk `mnem log -n 1` output and pull the leading op-cid. Format is
/// `op <cid>` on the first non-empty line.
fn current_op_cid(repo: &Path) -> String {
    let out = mnem(repo, &["log", "-n", "1"]).assert().success();
    let stdout = String::from_utf8_lossy(&out.get_output().stdout).to_string();
    for line in stdout.lines() {
        if let Some(rest) = line.strip_prefix("op ") {
            return rest.trim().to_string();
        }
    }
    panic!("log -n 1 had no op line: {stdout}");
}

#[test]
fn cat_file_raw_bytes_look_like_cbor_map() {
    let dir = TempDir::new().unwrap();
    mnem(dir.path(), &["init", dir.path().to_str().unwrap()])
        .assert()
        .success();
    // Give the repo a head commit so the op-log has a non-root op.
    mnem(dir.path(), &["add", "node", "--summary", "a", "--no-embed"])
        .assert()
        .success();
    let op = current_op_cid(dir.path());

    let out = mnem(dir.path(), &["cat-file", &op]).assert().success();
    let stdout = &out.get_output().stdout;
    assert!(!stdout.is_empty(), "raw cat-file should emit bytes");
    // DAG-CBOR maps start with major type 5 (0xa0..0xbf). Every mnem
    // object is a map, so the first byte MUST fall in that range.
    let first = stdout[0];
    assert!(
        (0xa0..=0xbf).contains(&first),
        "expected CBOR map first byte (0xa0..0xbf), got {first:#x}"
    );
}

#[test]
fn cat_file_json_carries_kind_field() {
    let dir = TempDir::new().unwrap();
    mnem(dir.path(), &["init", dir.path().to_str().unwrap()])
        .assert()
        .success();
    mnem(dir.path(), &["add", "node", "--summary", "b", "--no-embed"])
        .assert()
        .success();
    let op = current_op_cid(dir.path());

    let out = mnem(dir.path(), &["cat-file", &op, "--json"])
        .assert()
        .success();
    let stdout = String::from_utf8_lossy(&out.get_output().stdout).to_string();
    // Every mnem object carries a `_kind` string. Operation's kind is
    // `"operation"`; we assert on the substring rather than a full
    // JSON parse to keep the test robust against future additive
    // fields.
    assert!(
        stdout.contains("\"_kind\""),
        "JSON must carry _kind discriminator, got: {stdout}"
    );
    assert!(
        stdout.contains("\"operation\""),
        "this op block must be of kind `operation`, got: {stdout}"
    );
}

#[test]
fn cat_file_unknown_cid_errors() {
    let dir = TempDir::new().unwrap();
    mnem(dir.path(), &["init", dir.path().to_str().unwrap()])
        .assert()
        .success();
    // A syntactically-valid but absent CID. Use the bare sha256 of
    // something we never stored. Format is multibase+codec+hash;
    // `bafk...` is a well-known DAG-CBOR CID prefix that does not
    // exist in this repo.
    let missing = "bafkreibm6jg3ux5qumhcn2b3flc3tyu6dmlb4xa7u5bf44yegnrjhc3pgi";
    let out = mnem(dir.path(), &["cat-file", missing]).assert().failure();
    let stderr = String::from_utf8_lossy(&out.get_output().stderr).to_string();
    assert!(
        stderr.contains("not present"),
        "expected not-present error, got: {stderr}"
    );
}
