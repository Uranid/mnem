//! Integration tests for `mnem revert` functional correctness.
//!
//! Complements `tests/deferred_verbs.rs` which covers error paths.
//! These tests verify that revert actually undoes changes to the graph:
//! - node additions are reversed (node removed)
//! - node deletions are reversed (node restored)
//! - tombstones are reversed (tombstone removed)
//! - double-revert is a no-op
//! - custom messages propagate to the commit log
//! - chained reverts restore original state
//!
//! # Coverage note
//!
//! `mnem init` creates two ops: (1) an empty root op (no parent, no commit)
//! and (2) an anchor-node commit on top of it. Reverting the anchor commit
//! exercises the "root op" code path in `deferred.rs` (lines 157-165):
//! the parent of the anchor commit is the empty root op whose view has no
//! heads, so `parent_commit_cid_opt` is `None` and the before-state is built
//! from an empty prolly tree. See `revert_root_op_exercises_empty_before_state`.
//!
//! Two paths in `deferred.rs` are structurally untestable via CLI:
//!
//! - `DiffEntry::Changed` for nodes/edges (lines 308-333, 364-378): no
//!   `mnem update node` or `mnem update edge` command exists.
//! - The "op made no changes" early-exit (line 222-228): the only op with
//!   zero node/edge/tombstone changes is the empty root op itself, but
//!   reverting it fails at line 117-122 because its view has no head commit.

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

fn init(dir: &Path) {
    mnem(dir, &["init", dir.to_str().unwrap()])
        .assert()
        .success();
}

fn add_node(dir: &Path, summary: &str) -> String {
    let out = mnem(dir, &["add", "node", "--summary", summary, "--no-embed"])
        .assert()
        .success();
    let stdout = String::from_utf8_lossy(&out.get_output().stdout).to_string();
    for line in stdout.lines() {
        if let Some(rest) = line.strip_prefix("added node ") {
            return rest.trim().to_string();
        }
    }
    panic!("add node stdout had no 'added node <uuid>' line: {stdout}");
}

fn latest_op_cid(dir: &Path) -> String {
    let out = mnem(dir, &["log", "-n", "1"]).assert().success();
    let stdout = String::from_utf8_lossy(&out.get_output().stdout).to_string();
    stdout
        .lines()
        .find_map(|l| l.strip_prefix("op ").map(str::trim).map(str::to_string))
        .expect("mnem log -n 1 must emit an 'op <cid>' line")
}

/// Reverting the anchor commit (first real commit, whose parent is the
/// empty root op) exercises the "root op" code path in `deferred.rs`
/// (lines 157-165): parent_commit_cid_opt is None → before-state is an
/// empty prolly tree.
#[test]
fn revert_root_op_exercises_empty_before_state() {
    let dir = TempDir::new().unwrap();
    init(dir.path());
    // After init, latest_op_cid() returns the anchor-commit op. Its parent
    // is the empty root op (no heads), so reverting it hits the empty-tree
    // code path in deferred.rs lines 157-165.
    let anchor_op_cid = latest_op_cid(dir.path());
    mnem(dir.path(), &["revert", &anchor_op_cid])
        .assert()
        .success();
    // The anchor node (fixed UUID from init.rs ANCHOR_NODE_ID) must be absent
    // after the revert, confirming the empty-before-state path correctly
    // applied the inverse (deletion of the added anchor node).
    mnem(
        dir.path(),
        &["get", "00000000-0000-7000-8000-6d6e656d0001"],
    )
    .assert()
    .failure();
}

/// Reverting a node-add op causes the node to be removed from the graph.
#[test]
fn revert_undoes_add_node() {
    let dir = TempDir::new().unwrap();
    init(dir.path());

    let uuid = add_node(dir.path(), "to-be-reverted");
    let op_cid = latest_op_cid(dir.path());

    mnem(dir.path(), &["revert", &op_cid])
        .assert()
        .success();

    // Node must no longer be reachable in the graph.
    mnem(dir.path(), &["get", &uuid])
        .assert()
        .failure();
}

/// The revert command emits the documented output lines on success.
#[test]
fn revert_output_format() {
    let dir = TempDir::new().unwrap();
    init(dir.path());

    add_node(dir.path(), "format-check-node");
    let op_cid = latest_op_cid(dir.path());

    let out = mnem(dir.path(), &["revert", &op_cid])
        .assert()
        .success();
    let stdout = String::from_utf8_lossy(&out.get_output().stdout).to_string();

    // The first line names the op being reverted.
    let expected_header = format!("reverting op: {op_cid}");
    assert!(
        stdout.contains(&expected_header),
        "stdout must contain 'reverting op: <cid>', got:\n{stdout}"
    );

    // Summary line for node changes (1 added, 0 removed, 0 changed).
    assert!(
        stdout.contains("  nodes: 1 added, 0 removed, 0 changed by the original op"),
        "stdout must contain node-change summary, got:\n{stdout}"
    );

    // Summary line for edge changes (0 for this node-only op) — always printed.
    assert!(
        stdout.contains("  edges: 0 added, 0 removed, 0 changed by the original op"),
        "stdout must contain edge-change summary, got:\n{stdout}"
    );

    // Summary line for tombstone changes (0 for this op) — always printed.
    assert!(
        stdout.contains("  tombstones: 0 added by the original op (will be removed)"),
        "stdout must contain tombstone-change summary, got:\n{stdout}"
    );

    // Progress and completion markers.
    assert!(
        stdout.contains("applying inverse changes..."),
        "stdout must contain 'applying inverse changes...', got:\n{stdout}"
    );
    assert!(
        stdout.contains("done."),
        "stdout must contain 'done.', got:\n{stdout}"
    );

    // A new op CID must be reported with the documented spacing ("  new op:    <cid>").
    assert!(
        stdout.contains("  new op:    "),
        "stdout must contain '  new op:    ' (two leading spaces, four after colon), got:\n{stdout}"
    );

    // The commit head must also be reported on the following line.
    assert!(
        stdout.contains("  new commit: "),
        "stdout must contain '  new commit: ' line, got:\n{stdout}"
    );
}

/// Reverting a node-deletion op restores the node to the graph.
#[test]
fn revert_undoes_node_deletion() {
    let dir = TempDir::new().unwrap();
    init(dir.path());

    let uuid = add_node(dir.path(), "will-be-deleted");

    // Hard-delete the node.
    mnem(dir.path(), &["delete", &uuid])
        .assert()
        .success();

    let delete_op_cid = latest_op_cid(dir.path());

    // Confirm the node is gone.
    mnem(dir.path(), &["get", &uuid])
        .assert()
        .failure();

    // Revert the delete op.
    mnem(dir.path(), &["revert", &delete_op_cid])
        .assert()
        .success();

    // Node must be visible again and carry its original summary.
    let out = mnem(dir.path(), &["get", &uuid])
        .assert()
        .success();
    let stdout = String::from_utf8_lossy(&out.get_output().stdout).to_string();
    assert!(
        stdout.contains("will-be-deleted"),
        "restored node must contain its original summary, got:\n{stdout}"
    );
}

/// Reverting a tombstone op removes the tombstone marker from the node.
#[test]
fn revert_undoes_tombstone() {
    let dir = TempDir::new().unwrap();
    init(dir.path());

    let uuid = add_node(dir.path(), "soft-delete-candidate");

    // Soft-delete the node.
    mnem(dir.path(), &["tombstone", &uuid])
        .assert()
        .success();

    let tombstone_op_cid = latest_op_cid(dir.path());

    // Confirm the node is tombstoned.
    let out = mnem(dir.path(), &["get", &uuid])
        .assert()
        .success();
    let stdout = String::from_utf8_lossy(&out.get_output().stdout).to_string();
    assert!(
        stdout.contains("  tombstoned: true"),
        "node must be tombstoned before revert, got:\n{stdout}"
    );

    // Revert the tombstone op.
    let revert_out = mnem(dir.path(), &["revert", &tombstone_op_cid])
        .assert()
        .success();
    let revert_stdout = String::from_utf8_lossy(&revert_out.get_output().stdout).to_string();
    assert!(
        revert_stdout.contains("  tombstones: 1 added by the original op (will be removed)"),
        "revert of tombstone op must report tombstone count, got:\n{revert_stdout}"
    );

    // Node must still be reachable, but the tombstone marker must be gone.
    let out = mnem(dir.path(), &["get", &uuid])
        .assert()
        .success();
    let stdout = String::from_utf8_lossy(&out.get_output().stdout).to_string();
    assert!(
        !stdout.contains("  tombstoned: true"),
        "tombstone marker must be removed after revert, got:\n{stdout}"
    );
}

/// Reverting an already-reverted op is a no-op: exits 0 and prints the
/// "nothing to commit" message instead of creating a second revert commit.
#[test]
fn revert_already_reverted_prints_nothing_to_commit() {
    let dir = TempDir::new().unwrap();
    init(dir.path());

    add_node(dir.path(), "double-revert-node");
    let op_cid = latest_op_cid(dir.path());

    // First revert: must succeed.
    mnem(dir.path(), &["revert", &op_cid])
        .assert()
        .success();

    // Count ops before the no-op second revert.
    let before_out = mnem(dir.path(), &["log"]).assert().success();
    let before_count = String::from_utf8_lossy(&before_out.get_output().stdout)
        .lines()
        .filter(|l| l.starts_with("op "))
        .count();

    // Second revert of the same original op: inverse changes are all no-ops.
    let out = mnem(dir.path(), &["revert", &op_cid])
        .assert()
        .success();
    let stdout = String::from_utf8_lossy(&out.get_output().stdout).to_string();

    // The "note: ..." companion line must be printed before "nothing to commit.".
    assert!(
        stdout.contains("note: "),
        "second revert must print 'note: ...' before 'nothing to commit.', got:\n{stdout}"
    );
    assert!(
        stdout.contains("      nothing to commit."),
        "second revert must print '      nothing to commit.' (six spaces, period), got:\n{stdout}"
    );

    // No new op must be created — op count must be unchanged.
    let after_out = mnem(dir.path(), &["log"]).assert().success();
    let after_count = String::from_utf8_lossy(&after_out.get_output().stdout)
        .lines()
        .filter(|l| l.starts_with("op "))
        .count();
    assert_eq!(
        after_count,
        before_count,
        "no-op revert must not create a new op (count unchanged at {before_count})"
    );
}

/// A message supplied via `-m` appears in `mnem log` after the revert.
#[test]
fn revert_custom_message_in_log() {
    let dir = TempDir::new().unwrap();
    init(dir.path());

    add_node(dir.path(), "msg-test-node");
    let op_cid = latest_op_cid(dir.path());

    mnem(dir.path(), &["revert", &op_cid, "-m", "my-custom-revert-message"])
        .assert()
        .success();

    let out = mnem(dir.path(), &["log"]).assert().success();
    let stdout = String::from_utf8_lossy(&out.get_output().stdout).to_string();
    assert!(
        stdout.contains("my-custom-revert-message"),
        "log must contain the custom revert message, got:\n{stdout}"
    );
}

/// A successful revert produces exactly one new op in the log.
#[test]
fn revert_creates_new_op() {
    let dir = TempDir::new().unwrap();
    init(dir.path());

    add_node(dir.path(), "op-count-node");
    let add_op_cid = latest_op_cid(dir.path());

    // Count "op " lines before the revert.
    let before_out = mnem(dir.path(), &["log"]).assert().success();
    let before_stdout = String::from_utf8_lossy(&before_out.get_output().stdout).to_string();
    let before_count = before_stdout
        .lines()
        .filter(|l| l.starts_with("op "))
        .count();

    mnem(dir.path(), &["revert", &add_op_cid])
        .assert()
        .success();

    // Count "op " lines after the revert.
    let after_out = mnem(dir.path(), &["log"]).assert().success();
    let after_stdout = String::from_utf8_lossy(&after_out.get_output().stdout).to_string();
    let after_count = after_stdout
        .lines()
        .filter(|l| l.starts_with("op "))
        .count();

    assert_eq!(
        after_count,
        before_count + 1,
        "revert must create exactly 1 new op (was {before_count}, now {after_count})"
    );
}

/// Reverting a revert op re-applies the original change, restoring state.
#[test]
fn revert_chaining_restores_original_state() {
    let dir = TempDir::new().unwrap();
    init(dir.path());

    let uuid = add_node(dir.path(), "original-node");
    let add_op_cid = latest_op_cid(dir.path());

    // First revert: node disappears.
    mnem(dir.path(), &["revert", &add_op_cid])
        .assert()
        .success();

    mnem(dir.path(), &["get", &uuid])
        .assert()
        .failure();

    // The revert itself is now the latest op.
    let revert_op_cid = latest_op_cid(dir.path());

    // Revert the revert: node must reappear.
    mnem(dir.path(), &["revert", &revert_op_cid])
        .assert()
        .success();

    let out = mnem(dir.path(), &["get", &uuid])
        .assert()
        .success();
    let stdout = String::from_utf8_lossy(&out.get_output().stdout).to_string();
    assert!(
        stdout.contains("original-node"),
        "chained revert must restore the original node, got:\n{stdout}"
    );
}

/// Reverting an edge-add op removes the edge from the graph.
#[test]
fn revert_undoes_edge_add() {
    let dir = TempDir::new().unwrap();
    init(dir.path());

    let src_uuid = add_node(dir.path(), "edge-src-node");
    let dst_uuid = add_node(dir.path(), "edge-dst-node");

    // Add an edge from src to dst.
    mnem(dir.path(), &[
        "add", "edge",
        "--from", &src_uuid,
        "--to", &dst_uuid,
        "--label", "test_link",
    ])
    .assert()
    .success();

    let edge_op_cid = latest_op_cid(dir.path());

    // Confirm the edge exists before reverting.
    let before = mnem(dir.path(), &["traverse", &src_uuid])
        .assert()
        .success();
    let before_stdout = String::from_utf8_lossy(&before.get_output().stdout).to_string();
    assert!(
        before_stdout.contains("-[test_link]->"),
        "edge must exist before revert, got:\n{before_stdout}"
    );

    // Revert the edge-add op.
    mnem(dir.path(), &["revert", &edge_op_cid])
        .assert()
        .success();

    // Edge must be gone — traverse must report no outgoing edges.
    let after = mnem(dir.path(), &["traverse", &src_uuid])
        .assert()
        .success();
    let after_stdout = String::from_utf8_lossy(&after.get_output().stdout).to_string();
    assert!(
        after_stdout.contains("<no outgoing edges>"),
        "edge must be removed after reverting the edge-add op, got:\n{after_stdout}"
    );
}

/// Reverting an op that removed an edge fails if the edge's endpoint was
/// subsequently deleted (BUG-4 pre-flight check).
#[test]
fn revert_bug4_preflight_rejects_when_edge_endpoint_deleted() {
    let dir = TempDir::new().unwrap();
    init(dir.path());

    let src_uuid = add_node(dir.path(), "bug4-src");
    let dst_uuid = add_node(dir.path(), "bug4-dst");

    // Add an edge.
    mnem(dir.path(), &[
        "add", "edge",
        "--from", &src_uuid,
        "--to", &dst_uuid,
        "--label", "bug4_link",
    ])
    .assert()
    .success();

    let add_edge_op_cid = latest_op_cid(dir.path());

    // Revert the edge-add op. This creates a new op (op_2) that
    // "removed" the edge from the graph.
    mnem(dir.path(), &["revert", &add_edge_op_cid])
        .assert()
        .success();

    let remove_edge_op_cid = latest_op_cid(dir.path());

    // Delete the dst endpoint node (the edge would need it to exist).
    mnem(dir.path(), &["delete", &dst_uuid])
        .assert()
        .success();

    // Now try to revert the remove-edge op. This would re-add the edge,
    // but dst_uuid no longer exists -> BUG-4 pre-flight must reject it.
    mnem(dir.path(), &["revert", &remove_edge_op_cid])
        .assert()
        .failure();
}

/// Reverting an op that removed an edge re-adds the edge when both endpoints
/// still exist (happy path for the Removed-edge inverse path).
#[test]
fn revert_undoes_edge_removal() {
    let dir = TempDir::new().unwrap();
    init(dir.path());

    let src_uuid = add_node(dir.path(), "edge-restore-src");
    let dst_uuid = add_node(dir.path(), "edge-restore-dst");

    // Add the edge.
    mnem(dir.path(), &[
        "add", "edge",
        "--from", &src_uuid,
        "--to", &dst_uuid,
        "--label", "restore_link",
    ])
    .assert()
    .success();

    let add_edge_op_cid = latest_op_cid(dir.path());

    // Revert the edge-add: edge is now absent from the graph.
    mnem(dir.path(), &["revert", &add_edge_op_cid])
        .assert()
        .success();

    let remove_edge_op_cid = latest_op_cid(dir.path());

    // Confirm the edge is gone.
    let mid = mnem(dir.path(), &["traverse", &src_uuid])
        .assert()
        .success();
    assert!(
        String::from_utf8_lossy(&mid.get_output().stdout).contains("<no outgoing edges>"),
        "edge must be absent before the second revert"
    );

    // Revert the revert: both endpoints still exist, so the edge must be restored.
    mnem(dir.path(), &["revert", &remove_edge_op_cid])
        .assert()
        .success();

    // Edge must be visible again.
    let after = mnem(dir.path(), &["traverse", &src_uuid])
        .assert()
        .success();
    let after_stdout = String::from_utf8_lossy(&after.get_output().stdout).to_string();
    assert!(
        after_stdout.contains("-[restore_link]->"),
        "reverted edge must reappear in traverse output, got:\n{after_stdout}"
    );
}

/// Reverting an op that removed an edge fails if the edge's endpoint was
/// subsequently tombstoned (BUG-4 pre-flight check — tombstoned branch).
/// Complements `revert_bug4_preflight_rejects_when_edge_endpoint_deleted`
/// which exercises the hard-delete (!exists) branch; this one exercises
/// the soft-delete (tombstoned) branch of the same guard.
#[test]
fn revert_bug4_preflight_rejects_when_edge_endpoint_tombstoned() {
    let dir = TempDir::new().unwrap();
    init(dir.path());

    let src_uuid = add_node(dir.path(), "bug4-ts-src");
    let dst_uuid = add_node(dir.path(), "bug4-ts-dst");

    // Add an edge from src to dst.
    mnem(dir.path(), &[
        "add", "edge",
        "--from", &src_uuid,
        "--to", &dst_uuid,
        "--label", "bug4_ts_link",
    ])
    .assert()
    .success();

    let add_edge_op_cid = latest_op_cid(dir.path());

    // Revert the edge-add op, producing an op that "removed" the edge.
    mnem(dir.path(), &["revert", &add_edge_op_cid])
        .assert()
        .success();

    let remove_edge_op_cid = latest_op_cid(dir.path());

    // Soft-delete (tombstone) the dst endpoint — node still exists in the
    // store but is marked as deleted, which BUG-4 must also reject.
    mnem(dir.path(), &["tombstone", &dst_uuid])
        .assert()
        .success();

    // Reverting the remove-edge op would re-add the edge whose dst endpoint
    // is tombstoned. BUG-4 pre-flight must reject this with a non-zero exit.
    mnem(dir.path(), &["revert", &remove_edge_op_cid])
        .assert()
        .failure();
}

/// Reverting an op that removed an edge fails when the edge's SOURCE endpoint
/// was hard-deleted (BUG-4 pre-flight — src endpoint branch).
/// The pre-flight iterates both src and dst endpoints; this test exercises
/// the src side which the dst-only tests leave uncovered.
#[test]
fn revert_bug4_preflight_rejects_when_edge_src_deleted() {
    let dir = TempDir::new().unwrap();
    init(dir.path());

    let src_uuid = add_node(dir.path(), "bug4-src-del-src");
    let dst_uuid = add_node(dir.path(), "bug4-src-del-dst");

    // Add an edge from src to dst.
    mnem(dir.path(), &[
        "add", "edge",
        "--from", &src_uuid,
        "--to", &dst_uuid,
        "--label", "bug4_src_link",
    ])
    .assert()
    .success();

    let add_edge_op_cid = latest_op_cid(dir.path());

    // Revert the edge-add op, producing an op that "removed" the edge.
    mnem(dir.path(), &["revert", &add_edge_op_cid])
        .assert()
        .success();

    let remove_edge_op_cid = latest_op_cid(dir.path());

    // Hard-delete the SRC endpoint node.
    mnem(dir.path(), &["delete", &src_uuid])
        .assert()
        .success();

    // Reverting the remove-edge op would re-add the edge whose src endpoint
    // no longer exists. BUG-4 pre-flight must reject this.
    mnem(dir.path(), &["revert", &remove_edge_op_cid])
        .assert()
        .failure();
}

/// Reverting an op that removed an edge fails when the edge's SOURCE endpoint
/// was tombstoned (BUG-4 pre-flight — src tombstoned branch).
/// Completes the 2x2 matrix: dst-deleted (test above), dst-tombstoned, src-deleted
/// are covered; this test covers the src-tombstoned corner.
#[test]
fn revert_bug4_preflight_rejects_when_edge_src_tombstoned() {
    let dir = TempDir::new().unwrap();
    init(dir.path());

    let src_uuid = add_node(dir.path(), "bug4-src-ts-src");
    let dst_uuid = add_node(dir.path(), "bug4-src-ts-dst");

    // Add an edge from src to dst.
    mnem(dir.path(), &[
        "add", "edge",
        "--from", &src_uuid,
        "--to", &dst_uuid,
        "--label", "bug4_src_ts_link",
    ])
    .assert()
    .success();

    let add_edge_op_cid = latest_op_cid(dir.path());

    // Revert the edge-add op, producing an op that "removed" the edge.
    mnem(dir.path(), &["revert", &add_edge_op_cid])
        .assert()
        .success();

    let remove_edge_op_cid = latest_op_cid(dir.path());

    // Soft-delete (tombstone) the SRC endpoint node.
    mnem(dir.path(), &["tombstone", &src_uuid])
        .assert()
        .success();

    // Reverting the remove-edge op would re-add the edge whose src endpoint
    // is tombstoned. BUG-4 pre-flight must reject this.
    mnem(dir.path(), &["revert", &remove_edge_op_cid])
        .assert()
        .failure();
}
