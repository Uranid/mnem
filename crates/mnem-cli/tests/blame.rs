//! Integration tests for `mnem blame`.
//!
//! The "honest partial" contract from : `blame` walks the
//! incoming-edge index via the dual-adjacency primitive. Tests:
//!
//! 1. A node with no incoming edges prints `<no incoming edges>`.
//! 2. After `add node A; add node B; add edge B --authored--> A`,
//!    `blame A` lists the edge with src=B.

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

/// Extract the UUID of the freshly-added node from an `add node`
/// stdout line. `add` prints `added node <uuid>`.
fn extract_node_id(stdout: &str) -> String {
    for line in stdout.lines() {
        if let Some(rest) = line.strip_prefix("added node ") {
            return rest.trim().to_string();
        }
    }
    panic!("add-node stdout did not carry a node id: {stdout}");
}

#[test]
fn blame_on_node_with_no_incoming_edges() {
    let dir = TempDir::new().unwrap();
    mnem(dir.path(), &["init", dir.path().to_str().unwrap()])
        .assert()
        .success();
    let out = mnem(
        dir.path(),
        &["add", "node", "--summary", "isolated", "--label", "doc"],
    )
    .assert()
    .success();
    let node_id = extract_node_id(&String::from_utf8_lossy(&out.get_output().stdout));

    let blamed = mnem(dir.path(), &["blame", &node_id]).assert().success();
    let blamed_out = String::from_utf8_lossy(&blamed.get_output().stdout).to_string();
    assert!(
        blamed_out.contains("<no incoming edges>"),
        "isolated node should blame as <no incoming edges>, got: {blamed_out}"
    );
}

#[test]
fn blame_lists_incoming_edges_with_src() {
    let dir = TempDir::new().unwrap();
    mnem(dir.path(), &["init", dir.path().to_str().unwrap()])
        .assert()
        .success();

    // Destination node.
    let a_out = mnem(
        dir.path(),
        &[
            "add",
            "node",
            "--summary",
            "target",
            "--label",
            "doc",
            "--no-embed",
        ],
    )
    .assert()
    .success();
    let a_id = extract_node_id(&String::from_utf8_lossy(&a_out.get_output().stdout));

    // Source node (the "author").
    let b_out = mnem(
        dir.path(),
        &[
            "add",
            "node",
            "--summary",
            "author",
            "--label",
            "person",
            "--no-embed",
        ],
    )
    .assert()
    .success();
    let b_id = extract_node_id(&String::from_utf8_lossy(&b_out.get_output().stdout));

    // Edge from B -> A, type `authored`.
    mnem(
        dir.path(),
        &[
            "add", "edge", "--from", &b_id, "--to", &a_id, "--label", "authored",
        ],
    )
    .assert()
    .success();

    // Blame A: should list one incoming edge with src=B and etype=authored.
    let blame_out = mnem(dir.path(), &["blame", &a_id]).assert().success();
    let out = String::from_utf8_lossy(&blame_out.get_output().stdout).to_string();
    assert!(
        out.contains(&b_id),
        "blame must mention source node {b_id}, got: {out}"
    );
    assert!(
        out.contains("authored"),
        "blame must mention edge type, got: {out}"
    );
}
