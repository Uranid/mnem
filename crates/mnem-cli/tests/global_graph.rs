//! Integration tests for `mnem global` subcommands.
//!
//! Verifies that the global graph (rooted at a temp directory instead of
//! `~/.mnemglobal`) correctly stores and retrieves nodes, edges, and
//! tombstones, and that the op log and status commands reflect changes.
//!
//! # Isolation strategy
//!
//! Every test passes `-R <tempdir>` to `mnem global <subcmd>`. The
//! `global_cmd::run` function uses this as `override_path`, so all global
//! commands operate on `<tempdir>/.mnem` instead of `~/.mnemglobal/.mnem`.
//! A single `mnem init <tempdir>` call bootstraps the `.mnem` directory,
//! satisfying the `require_global_init` guard without touching the
//! developer's real home directory.

use std::path::Path;
use std::process::Command;

use assert_cmd::prelude::*;
use tempfile::TempDir;

/// Build a `mnem` command targeting the global graph at `global_root`.
///
/// All routing is done via `-R <global_root>`, which sets `cli.repo` and is
/// forwarded as `override_path` to `global_cmd::run`. The process CWD is set
/// to the system temp directory — a directory that will never contain a
/// `.mnem` subdirectory — so that local-repo auto-discovery in other code
/// paths cannot accidentally pick up the project checkout directory. This
/// also eliminates any risk from ambient env vars that influence local-repo
/// detection, since the CWD fallback path is completely neutral.
fn mnem_global(global_root: &Path, args: &[&str]) -> Command {
    let mut cmd = Command::cargo_bin("mnem").expect("built mnem binary");
    cmd.arg("-R").arg(global_root);
    cmd.current_dir(std::env::temp_dir());
    for a in args {
        cmd.arg(a);
    }
    cmd
}

/// Initialise a temp directory as a mnem graph so `require_global_init` passes.
fn init_global(global_root: &Path) {
    Command::cargo_bin("mnem")
        .unwrap()
        .arg("init")
        .arg(global_root)
        .assert()
        .success();
}

/// Add a node (no label) to the global graph without embedding.
/// Returns the node UUID parsed from the "added node <uuid>" output line.
fn global_add_node(global_root: &Path, summary: &str) -> String {
    let out = mnem_global(
        global_root,
        &["global", "add", "node", "--summary", summary, "--no-embed"],
    )
    .assert()
    .success();
    let stdout = String::from_utf8_lossy(&out.get_output().stdout).to_string();
    for line in stdout.lines() {
        if let Some(rest) = line.strip_prefix("added node ") {
            return rest.trim().to_string();
        }
    }
    panic!("global add node stdout had no 'added node <uuid>' line:\n{stdout}");
}

/// Add a labeled node to the global graph without embedding.
/// Returns the node UUID parsed from the "added node <uuid>" output line.
fn global_add_labeled_node(global_root: &Path, summary: &str, label: &str) -> String {
    let out = mnem_global(
        global_root,
        &[
            "global", "add", "node",
            "--summary", summary,
            "--label", label,
            "--no-embed",
        ],
    )
    .assert()
    .success();
    let stdout = String::from_utf8_lossy(&out.get_output().stdout).to_string();
    for line in stdout.lines() {
        if let Some(rest) = line.strip_prefix("added node ") {
            return rest.trim().to_string();
        }
    }
    panic!("global add labeled node stdout had no 'added node <uuid>' line:\n{stdout}");
}

// ── Error guard ──────────────────────────────────────────────────────────────

/// A global subcommand on a directory that has no `.mnem` subdirectory must
/// fail with a non-zero exit and print BOTH the "not initialised" message and
/// the "mnem integrate" hint.
#[test]
fn global_uninitialised_exits_nonzero() {
    let dir = TempDir::new().unwrap();
    // Do NOT call init_global — the directory has no .mnem inside it.
    let out = mnem_global(dir.path(), &["global", "status"])
        .assert()
        .failure();
    let stderr = String::from_utf8_lossy(&out.get_output().stderr).to_string();
    assert!(
        stderr.contains("not initialised"),
        "error must mention 'not initialised', got:\n{stderr}"
    );
    assert!(
        stderr.contains("mnem integrate"),
        "error must mention 'mnem integrate', got:\n{stderr}"
    );
}

// ── Status & stats ───────────────────────────────────────────────────────────

/// `mnem global status` exits 0 on a freshly-initialised global graph and
/// prints both the op_id line and the commit line.
#[test]
fn global_status_on_fresh_init() {
    let dir = TempDir::new().unwrap();
    init_global(dir.path());

    let out = mnem_global(dir.path(), &["global", "status"])
        .assert()
        .success();
    let stdout = String::from_utf8_lossy(&out.get_output().stdout).to_string();
    assert!(
        stdout.contains("op_id"),
        "status must print 'op_id', got:\n{stdout}"
    );
    assert!(
        stdout.contains("commit"),
        "status must print 'commit' line, got:\n{stdout}"
    );
}

/// `mnem global stats` exits 0 on a freshly-initialised global graph and
/// prints the machine-friendly one-liner that starts with "op=".
#[test]
fn global_stats_on_fresh_init() {
    let dir = TempDir::new().unwrap();
    init_global(dir.path());

    let out = mnem_global(dir.path(), &["global", "stats"])
        .assert()
        .success();
    let stdout = String::from_utf8_lossy(&out.get_output().stdout).to_string();
    assert!(
        stdout.contains("op="),
        "stats must print 'op=<cid> commit=<cid> ...' line, got:\n{stdout}"
    );
    assert!(
        stdout.contains("commit="),
        "stats must print 'commit=<cid>' in the one-liner, got:\n{stdout}"
    );
    assert!(stdout.contains("content="), "stats must print 'content=...' in the one-liner, got:\n{stdout}");
    assert!(stdout.contains("refs="), "stats must print 'refs=...' in the one-liner, got:\n{stdout}");
    assert!(stdout.contains("labels="), "stats must print 'labels=...' in the one-liner, got:\n{stdout}");
}

/// Adding a labeled node must increment the `labels=N` counter in `mnem global stats`.
/// This verifies that the label index is updated and reflected in the machine-readable stats.
#[test]
fn global_stats_labels_increments_after_labeled_node() {
    let dir = TempDir::new().unwrap();
    init_global(dir.path());

    // Capture labels count before.
    let before_out = mnem_global(dir.path(), &["global", "stats"])
        .assert()
        .success();
    let before_stdout = String::from_utf8_lossy(&before_out.get_output().stdout).to_string();
    let before_labels: usize = before_stdout
        .split_whitespace()
        .find(|tok| tok.starts_with("labels="))
        .and_then(|tok| tok["labels=".len()..].parse().ok())
        .expect("stats must include a 'labels=N' token");

    global_add_labeled_node(dir.path(), "label-stats-test", "StatsLabelCheck");

    let after_out = mnem_global(dir.path(), &["global", "stats"])
        .assert()
        .success();
    let after_stdout = String::from_utf8_lossy(&after_out.get_output().stdout).to_string();
    let after_labels: usize = after_stdout
        .split_whitespace()
        .find(|tok| tok.starts_with("labels="))
        .and_then(|tok| tok["labels=".len()..].parse().ok())
        .expect("stats must include a 'labels=N' token after write");

    assert!(
        after_labels > before_labels,
        "labels= count in stats must increase after adding a labeled node (was {before_labels}, now {after_labels})"
    );
}

/// Adding an edge must increment the `refs=N` counter in `mnem global stats`.
/// This verifies that the edge index is updated and reflected in the machine-readable stats.
#[test]
fn global_stats_refs_increments_after_edge() {
    let dir = TempDir::new().unwrap();
    init_global(dir.path());

    let src = global_add_node(dir.path(), "refs-stats-src");
    let dst = global_add_node(dir.path(), "refs-stats-dst");

    // Capture refs count before adding an edge.
    let before_out = mnem_global(dir.path(), &["global", "stats"])
        .assert()
        .success();
    let before_stdout = String::from_utf8_lossy(&before_out.get_output().stdout).to_string();
    let before_refs: usize = before_stdout
        .split_whitespace()
        .find(|tok| tok.starts_with("refs="))
        .and_then(|tok| tok["refs=".len()..].parse().ok())
        .expect("stats must include a 'refs=N' token");

    // Add an edge.
    mnem_global(
        dir.path(),
        &[
            "global", "add", "edge",
            "--from", &src,
            "--to", &dst,
            "--label", "refs_test_link",
        ],
    )
    .assert()
    .success();

    let after_out = mnem_global(dir.path(), &["global", "stats"])
        .assert()
        .success();
    let after_stdout = String::from_utf8_lossy(&after_out.get_output().stdout).to_string();
    let after_refs: usize = after_stdout
        .split_whitespace()
        .find(|tok| tok.starts_with("refs="))
        .and_then(|tok| tok["refs=".len()..].parse().ok())
        .expect("stats must include a 'refs=N' token after edge write");

    assert!(
        after_refs > before_refs,
        "refs= count in stats must increase after adding an edge (was {before_refs}, now {after_refs})"
    );
}

/// After a write, `mnem global status` must report a different `op_id` than
/// before the write (each commit advances the op-head).
#[test]
fn global_status_op_id_advances_after_write() {
    let dir = TempDir::new().unwrap();
    init_global(dir.path());

    // Capture the initial op_id line.
    let before_out = mnem_global(dir.path(), &["global", "status"])
        .assert()
        .success();
    let before_stdout = String::from_utf8_lossy(&before_out.get_output().stdout).to_string();
    let before_op_line = before_stdout
        .lines()
        .find(|l| l.contains("op_id"))
        .map(|l| l.to_string())
        .expect("status must print an op_id line");

    global_add_node(dir.path(), "status-advance-test");

    // op_id must be different after the write.
    let after_out = mnem_global(dir.path(), &["global", "status"])
        .assert()
        .success();
    let after_stdout = String::from_utf8_lossy(&after_out.get_output().stdout).to_string();
    let after_op_line = after_stdout
        .lines()
        .find(|l| l.contains("op_id"))
        .map(|l| l.to_string())
        .expect("status must print an op_id line after write");

    assert_ne!(
        before_op_line, after_op_line,
        "op_id in 'global status' must change after a write:\n  before: {before_op_line}\n  after:  {after_op_line}"
    );
}

// ── Node add / get ───────────────────────────────────────────────────────────

/// `mnem global add node` commits a node and `mnem global get <uuid>` returns
/// it with the correct summary and ntype.
#[test]
fn global_add_node_and_get_by_uuid() {
    let dir = TempDir::new().unwrap();
    init_global(dir.path());

    let uuid = global_add_node(dir.path(), "global-fact-node");

    let out = mnem_global(dir.path(), &["global", "get", &uuid])
        .assert()
        .success();
    let stdout = String::from_utf8_lossy(&out.get_output().stdout).to_string();
    assert!(
        stdout.contains("global-fact-node"),
        "get must return the node's summary, got:\n{stdout}"
    );
    assert!(
        stdout.contains(&uuid),
        "get output must include the node UUID, got:\n{stdout}"
    );
}

/// `mnem global add node --label <L>` stores the label as the node's `ntype`.
/// `mnem global get <uuid>` must echo back both `ntype: <L>` and the summary.
#[test]
fn global_labeled_node_ntype_in_get() {
    let dir = TempDir::new().unwrap();
    init_global(dir.path());

    let uuid = global_add_labeled_node(dir.path(), "ntype-round-trip", "MyCustomLabel");

    let out = mnem_global(dir.path(), &["global", "get", &uuid])
        .assert()
        .success();
    let stdout = String::from_utf8_lossy(&out.get_output().stdout).to_string();
    assert!(
        stdout.contains("ntype:"),
        "get output must include 'ntype:' field, got:\n{stdout}"
    );
    assert!(
        stdout.contains("MyCustomLabel"),
        "get output must show the stored label 'MyCustomLabel', got:\n{stdout}"
    );
    assert!(
        stdout.contains("ntype-round-trip"),
        "get output must include the node summary, got:\n{stdout}"
    );
}

/// Adding a node to the global graph creates exactly one new op in the log.
#[test]
fn global_add_node_creates_new_op() {
    let dir = TempDir::new().unwrap();
    init_global(dir.path());

    // Use -n 100 so the count is never silently capped by a default page limit.
    let before_out = mnem_global(dir.path(), &["global", "log", "-n", "100"])
        .assert()
        .success();
    let before_count = String::from_utf8_lossy(&before_out.get_output().stdout)
        .lines()
        .filter(|l| l.starts_with("op "))
        .count();

    global_add_node(dir.path(), "op-count-test");

    let after_out = mnem_global(dir.path(), &["global", "log", "-n", "100"])
        .assert()
        .success();
    let after_count = String::from_utf8_lossy(&after_out.get_output().stdout)
        .lines()
        .filter(|l| l.starts_with("op "))
        .count();

    assert_eq!(
        after_count,
        before_count + 1,
        "adding a node must create exactly 1 new op (was {before_count}, now {after_count})"
    );
}

/// `mnem global get <uuid>` on a non-existent UUID exits non-zero.
#[test]
fn global_get_missing_uuid_exits_nonzero() {
    let dir = TempDir::new().unwrap();
    init_global(dir.path());
    mnem_global(
        dir.path(),
        &["global", "get", "00000000-0000-7000-8000-000000000099"],
    )
    .assert()
    .failure();
}

// ── Tombstone ────────────────────────────────────────────────────────────────

/// `mnem global tombstone <uuid>` soft-deletes a node: `global get` still
/// succeeds but reports `tombstoned: true`.
#[test]
fn global_tombstone_node_shows_tombstoned() {
    let dir = TempDir::new().unwrap();
    init_global(dir.path());

    let uuid = global_add_node(dir.path(), "soft-delete-global");

    // Confirm the node exists and is not tombstoned before the operation.
    let before = mnem_global(dir.path(), &["global", "get", &uuid])
        .assert()
        .success();
    let before_stdout = String::from_utf8_lossy(&before.get_output().stdout).to_string();
    assert!(
        !before_stdout.contains("tombstoned: true"),
        "node must not be tombstoned before tombstone op, got:\n{before_stdout}"
    );

    mnem_global(dir.path(), &["global", "tombstone", &uuid])
        .assert()
        .success();

    let after = mnem_global(dir.path(), &["global", "get", &uuid])
        .assert()
        .success();
    let after_stdout = String::from_utf8_lossy(&after.get_output().stdout).to_string();
    assert!(
        after_stdout.contains("tombstoned: true"),
        "get must report 'tombstoned: true' after tombstone op, got:\n{after_stdout}"
    );
}

/// `mnem global tombstone <uuid>` on a non-existent UUID exits non-zero.
#[test]
fn global_tombstone_nonexistent_uuid_exits_nonzero() {
    let dir = TempDir::new().unwrap();
    init_global(dir.path());
    mnem_global(
        dir.path(),
        &["global", "tombstone", "00000000-0000-7000-8000-000000000099"],
    )
    .assert()
    .failure();
}

/// Tombstoned nodes must NOT appear in `mnem global query` results. The
/// tombstone-filtering invariant: soft-deleted nodes are invisible to search.
#[test]
fn global_tombstoned_node_hidden_from_query() {
    let dir = TempDir::new().unwrap();
    init_global(dir.path());

    let uuid = global_add_labeled_node(dir.path(), "tombstone-query-check", "TombstoneQueryLabel");

    // Before tombstone: query must return the node.
    let before_out = mnem_global(
        dir.path(),
        &["global", "query", "--where", "ntype=TombstoneQueryLabel"],
    )
    .assert()
    .success();
    let before_stdout = String::from_utf8_lossy(&before_out.get_output().stdout).to_string();
    assert!(
        before_stdout.contains(&uuid),
        "node must appear in query before tombstone, got:\n{before_stdout}"
    );
    assert!(
        before_stdout.contains("1 hit(s)"),
        "query must show '1 hit(s)' before tombstone, got:\n{before_stdout}"
    );

    // Tombstone the node.
    mnem_global(dir.path(), &["global", "tombstone", &uuid])
        .assert()
        .success();

    // After tombstone: the node must be absent from query results.
    let after_out = mnem_global(
        dir.path(),
        &["global", "query", "--where", "ntype=TombstoneQueryLabel"],
    )
    .assert()
    .success();
    let after_stdout = String::from_utf8_lossy(&after_out.get_output().stdout).to_string();
    assert!(
        after_stdout.contains("0 hit(s)"),
        "tombstoned node must be hidden; query must report '0 hit(s)', got:\n{after_stdout}"
    );
    assert!(
        !after_stdout.contains(&uuid),
        "tombstoned node UUID must not appear in query output, got:\n{after_stdout}"
    );
}

/// Tombstoned nodes must NOT appear in `mnem global retrieve` results.
/// This verifies the tombstone-filtering invariant across the retrieve path.
#[test]
fn global_tombstoned_node_hidden_from_retrieve() {
    let dir = TempDir::new().unwrap();
    init_global(dir.path());

    let uuid = global_add_labeled_node(
        dir.path(),
        "tombstone-retrieve-check",
        "TombstoneRetrieveLabel",
    );

    // Before tombstone: retrieve must return the node.
    let before_out = mnem_global(
        dir.path(),
        &[
            "global", "retrieve",
            "--where", "ntype=TombstoneRetrieveLabel",
            "--no-vector",
        ],
    )
    .assert()
    .success();
    let before_stdout = String::from_utf8_lossy(&before_out.get_output().stdout).to_string();
    assert!(
        before_stdout.contains(&uuid),
        "node must appear in retrieve before tombstone, got:\n{before_stdout}"
    );
    assert!(
        before_stdout.contains("1 item(s)"),
        "retrieve must show '1 item(s)' before tombstone, got:\n{before_stdout}"
    );

    // Tombstone the node.
    mnem_global(dir.path(), &["global", "tombstone", &uuid])
        .assert()
        .success();

    // After tombstone: the node must be absent from retrieve results.
    let after_out = mnem_global(
        dir.path(),
        &[
            "global", "retrieve",
            "--where", "ntype=TombstoneRetrieveLabel",
            "--no-vector",
        ],
    )
    .assert()
    .success();
    let after_stdout = String::from_utf8_lossy(&after_out.get_output().stdout).to_string();
    assert!(
        after_stdout.contains("0 item(s)"),
        "tombstoned node must be hidden; retrieve must report '0 item(s)', got:\n{after_stdout}"
    );
    assert!(
        !after_stdout.contains(&uuid),
        "tombstoned node UUID must not appear in retrieve output, got:\n{after_stdout}"
    );
}

// ── Hard delete ──────────────────────────────────────────────────────────────

/// `mnem global delete <uuid>` hard-deletes a node: `global get` must fail
/// after deletion.
#[test]
fn global_delete_node_makes_get_fail() {
    let dir = TempDir::new().unwrap();
    init_global(dir.path());

    let uuid = global_add_node(dir.path(), "to-be-deleted-global");

    // Confirm it exists first.
    mnem_global(dir.path(), &["global", "get", &uuid])
        .assert()
        .success();

    mnem_global(dir.path(), &["global", "delete", &uuid])
        .assert()
        .success();

    mnem_global(dir.path(), &["global", "get", &uuid])
        .assert()
        .failure();
}

/// `mnem global delete <uuid>` on a non-existent UUID exits non-zero.
#[test]
fn global_delete_nonexistent_uuid_exits_nonzero() {
    let dir = TempDir::new().unwrap();
    init_global(dir.path());
    mnem_global(
        dir.path(),
        &["global", "delete", "00000000-0000-7000-8000-000000000099"],
    )
    .assert()
    .failure();
}

/// Hard-deleted nodes must NOT appear in `mnem global query` results.
/// This verifies the delete-filtering invariant across the query path
/// (analogous to the tombstone-filtering tests above, but for hard deletes).
#[test]
fn global_deleted_node_hidden_from_query() {
    let dir = TempDir::new().unwrap();
    init_global(dir.path());

    let uuid = global_add_labeled_node(dir.path(), "delete-query-check", "DeleteQueryLabel");

    // Before delete: query must return the node.
    let before_out = mnem_global(
        dir.path(),
        &["global", "query", "--where", "ntype=DeleteQueryLabel"],
    )
    .assert()
    .success();
    let before_stdout = String::from_utf8_lossy(&before_out.get_output().stdout).to_string();
    assert!(
        before_stdout.contains(&uuid),
        "node must appear in query before deletion, got:\n{before_stdout}"
    );
    assert!(
        before_stdout.contains("1 hit(s)"),
        "query must show '1 hit(s)' before deletion, got:\n{before_stdout}"
    );

    // Hard-delete the node.
    mnem_global(dir.path(), &["global", "delete", &uuid])
        .assert()
        .success();

    // After delete: the node must be absent from query results.
    let after_out = mnem_global(
        dir.path(),
        &["global", "query", "--where", "ntype=DeleteQueryLabel"],
    )
    .assert()
    .success();
    let after_stdout = String::from_utf8_lossy(&after_out.get_output().stdout).to_string();
    assert!(
        after_stdout.contains("0 hit(s)"),
        "hard-deleted node must be hidden; query must report '0 hit(s)', got:\n{after_stdout}"
    );
    assert!(
        !after_stdout.contains(&uuid),
        "hard-deleted node UUID must not appear in query output, got:\n{after_stdout}"
    );
}

/// Hard-deleted nodes must NOT appear in `mnem global retrieve` results.
/// This verifies the delete-filtering invariant across the retrieve path.
#[test]
fn global_deleted_node_hidden_from_retrieve() {
    let dir = TempDir::new().unwrap();
    init_global(dir.path());

    let uuid = global_add_labeled_node(
        dir.path(),
        "delete-retrieve-check",
        "DeleteRetrieveLabel",
    );

    // Before delete: retrieve must return the node.
    let before_out = mnem_global(
        dir.path(),
        &[
            "global", "retrieve",
            "--where", "ntype=DeleteRetrieveLabel",
            "--no-vector",
        ],
    )
    .assert()
    .success();
    let before_stdout = String::from_utf8_lossy(&before_out.get_output().stdout).to_string();
    assert!(
        before_stdout.contains(&uuid),
        "node must appear in retrieve before deletion, got:\n{before_stdout}"
    );
    assert!(
        before_stdout.contains("1 item(s)"),
        "retrieve must show '1 item(s)' before deletion, got:\n{before_stdout}"
    );

    // Hard-delete the node.
    mnem_global(dir.path(), &["global", "delete", &uuid])
        .assert()
        .success();

    // After delete: the node must be absent from retrieve results.
    let after_out = mnem_global(
        dir.path(),
        &[
            "global", "retrieve",
            "--where", "ntype=DeleteRetrieveLabel",
            "--no-vector",
        ],
    )
    .assert()
    .success();
    let after_stdout = String::from_utf8_lossy(&after_out.get_output().stdout).to_string();
    assert!(
        after_stdout.contains("0 item(s)"),
        "hard-deleted node must be hidden; retrieve must report '0 item(s)', got:\n{after_stdout}"
    );
    assert!(
        !after_stdout.contains(&uuid),
        "hard-deleted node UUID must not appear in retrieve output, got:\n{after_stdout}"
    );
}

// ── Edge add + blame ─────────────────────────────────────────────────────────

/// `mnem global add edge` connects two nodes; `mnem global blame <dst>` lists
/// the incoming edge from the source.
#[test]
fn global_add_edge_visible_in_blame() {
    let dir = TempDir::new().unwrap();
    init_global(dir.path());

    let src_uuid = global_add_node(dir.path(), "global-edge-src");
    let dst_uuid = global_add_node(dir.path(), "global-edge-dst");

    mnem_global(
        dir.path(),
        &[
            "global",
            "add",
            "edge",
            "--from",
            &src_uuid,
            "--to",
            &dst_uuid,
            "--label",
            "global_link",
        ],
    )
    .assert()
    .success();

    let out = mnem_global(dir.path(), &["global", "blame", &dst_uuid])
        .assert()
        .success();
    let stdout = String::from_utf8_lossy(&out.get_output().stdout).to_string();
    assert!(
        stdout.contains("global_link"),
        "blame must list the incoming 'global_link' edge, got:\n{stdout}"
    );
    assert!(
        stdout.contains(&src_uuid),
        "blame must reference the source node UUID, got:\n{stdout}"
    );
    assert!(
        stdout.contains("-[global_link]->"),
        "blame must use the '<src> -[etype]-> <dst>' format, got:\n{stdout}"
    );
}

/// `mnem global add edge` with a non-existent source UUID must exit non-zero.
#[test]
fn global_add_edge_nonexistent_uuid_exits_nonzero() {
    let dir = TempDir::new().unwrap();
    init_global(dir.path());

    let existing = global_add_node(dir.path(), "edge-target-exists");

    // Non-existent source UUID.
    mnem_global(
        dir.path(),
        &[
            "global", "add", "edge",
            "--from", "00000000-0000-7000-8000-000000000099",
            "--to", &existing,
            "--label", "ghost_link",
        ],
    )
    .assert()
    .failure();
}

/// `mnem global add edge` with a non-existent destination UUID must exit non-zero.
#[test]
fn global_add_edge_bad_dst_uuid_exits_nonzero() {
    let dir = TempDir::new().unwrap();
    init_global(dir.path());

    let existing = global_add_node(dir.path(), "edge-source-exists");

    // Non-existent destination UUID.
    mnem_global(
        dir.path(),
        &[
            "global", "add", "edge",
            "--from", &existing,
            "--to", "00000000-0000-7000-8000-000000000099",
            "--label", "ghost_link",
        ],
    )
    .assert()
    .failure();
}

/// `mnem global blame <uuid>` on a node with no incoming edges prints
/// `"<no incoming edges>"`.
#[test]
fn global_blame_no_incoming_edges() {
    let dir = TempDir::new().unwrap();
    init_global(dir.path());

    // Add a node but connect no edges to it.
    let uuid = global_add_node(dir.path(), "isolated-node-blame");

    let out = mnem_global(dir.path(), &["global", "blame", &uuid])
        .assert()
        .success();
    let stdout = String::from_utf8_lossy(&out.get_output().stdout).to_string();
    assert!(
        stdout.contains("<no incoming edges>"),
        "blame on a node with no incoming edges must print '<no incoming edges>', got:\n{stdout}"
    );
}

/// `mnem global blame <uuid>` on a UUID that was never inserted exits non-zero.
#[test]
fn global_blame_nonexistent_uuid_exits_nonzero() {
    let dir = TempDir::new().unwrap();
    init_global(dir.path());
    mnem_global(
        dir.path(),
        &["global", "blame", "00000000-0000-7000-8000-000000000099"],
    )
    .assert()
    .failure();
}

// ── Log ──────────────────────────────────────────────────────────────────────

/// `mnem global log -n 1` emits an `op <cid>` line for the most recent op.
#[test]
fn global_log_emits_op_line() {
    let dir = TempDir::new().unwrap();
    init_global(dir.path());

    global_add_node(dir.path(), "log-test-node");

    let out = mnem_global(dir.path(), &["global", "log", "-n", "1"])
        .assert()
        .success();
    let stdout = String::from_utf8_lossy(&out.get_output().stdout).to_string();
    assert!(
        stdout.lines().any(|l| l.starts_with("op ")),
        "global log -n 1 must emit an 'op <cid>' line, got:\n{stdout}"
    );
}

// ── Query ────────────────────────────────────────────────────────────────────

/// `mnem global query --where ntype=<label>` returns exactly `"1 hit(s)"` and
/// includes the UUID of the node that was added with that label.
#[test]
fn global_query_filters_by_label() {
    let dir = TempDir::new().unwrap();
    init_global(dir.path());

    let uuid = global_add_labeled_node(dir.path(), "query-label-test", "FactQuery");

    let query_out =
        mnem_global(dir.path(), &["global", "query", "--where", "ntype=FactQuery"])
            .assert()
            .success();
    let query_stdout = String::from_utf8_lossy(&query_out.get_output().stdout).to_string();
    assert!(
        query_stdout.contains("1 hit(s)"),
        "query must report '1 hit(s)' for one matching node, got:\n{query_stdout}"
    );
    assert!(
        query_stdout.contains(&uuid),
        "query --where ntype=FactQuery must return the FactQuery node UUID, got:\n{query_stdout}"
    );
}

/// `mnem global query --where ntype=<label>` returns `"0 hit(s)"` when no
/// nodes match the requested label.
#[test]
fn global_query_zero_hits_for_unknown_label() {
    let dir = TempDir::new().unwrap();
    init_global(dir.path());

    // Do not add any node with this label.
    let out = mnem_global(
        dir.path(),
        &["global", "query", "--where", "ntype=LabelThatNeverExists99"],
    )
    .assert()
    .success();
    let stdout = String::from_utf8_lossy(&out.get_output().stdout).to_string();
    assert!(
        stdout.contains("0 hit(s)"),
        "query for a non-existent label must print '0 hit(s)', got:\n{stdout}"
    );
}

/// `mnem global query --where ntype=<label>` returns `"2 hit(s)"` when two
/// nodes share the same label. This verifies count accuracy beyond the 1-node case.
#[test]
fn global_query_two_nodes_same_label() {
    let dir = TempDir::new().unwrap();
    init_global(dir.path());

    let uuid1 = global_add_labeled_node(dir.path(), "multi-query-a", "MultiQueryLabel");
    let uuid2 = global_add_labeled_node(dir.path(), "multi-query-b", "MultiQueryLabel");

    let out = mnem_global(dir.path(), &["global", "query", "--where", "ntype=MultiQueryLabel"])
        .assert()
        .success();
    let stdout = String::from_utf8_lossy(&out.get_output().stdout).to_string();
    assert!(
        stdout.contains("2 hit(s)"),
        "query must report '2 hit(s)' when two nodes share the same label, got:\n{stdout}"
    );
    assert!(
        stdout.contains(&uuid1),
        "query must include first node UUID, got:\n{stdout}"
    );
    assert!(
        stdout.contains(&uuid2),
        "query must include second node UUID, got:\n{stdout}"
    );
}

// ── Retrieve ─────────────────────────────────────────────────────────────────

/// `mnem global retrieve --where ntype=<label> --no-vector` returns the node
/// that was added with that label. Uses `--no-vector` to avoid requiring a
/// configured embedder in the test environment.
#[test]
fn global_retrieve_returns_matching_node() {
    let dir = TempDir::new().unwrap();
    init_global(dir.path());

    let uuid = global_add_labeled_node(
        dir.path(),
        "retrieve-target-unique",
        "GlobalRetrieveTest",
    );

    // Retrieve by property filter; --no-vector skips the embedder so no
    // Ollama/OpenAI connection is required in the test environment.
    let ret_out = mnem_global(
        dir.path(),
        &[
            "global",
            "retrieve",
            "--where",
            "ntype=GlobalRetrieveTest",
            "--no-vector",
        ],
    )
    .assert()
    .success();
    let ret_stdout = String::from_utf8_lossy(&ret_out.get_output().stdout).to_string();
    assert!(
        ret_stdout.contains(&uuid),
        "retrieve --where ntype=GlobalRetrieveTest must return the node UUID, got:\n{ret_stdout}"
    );
    assert!(
        ret_stdout.contains("1 item(s)"),
        "retrieve must report '1 item(s)', got:\n{ret_stdout}"
    );
}

/// `mnem global retrieve --where ntype=<label> --no-vector` returns `"0 item(s)"`
/// when no nodes match. Verifies the retrieve path handles empty results gracefully.
#[test]
fn global_retrieve_zero_hits_for_unknown_label() {
    let dir = TempDir::new().unwrap();
    init_global(dir.path());

    let out = mnem_global(
        dir.path(),
        &[
            "global", "retrieve",
            "--where", "ntype=LabelThatNeverExistsInRetrieve77",
            "--no-vector",
        ],
    )
    .assert()
    .success();
    let stdout = String::from_utf8_lossy(&out.get_output().stdout).to_string();
    assert!(
        stdout.contains("0 item(s)"),
        "retrieve for a non-existent label must print '0 item(s)', got:\n{stdout}"
    );
}

/// `mnem global retrieve "<text>" --no-vector` finds a node whose summary
/// matches the query text via BM25/text search (no embedder required).
#[test]
fn global_retrieve_text_query_returns_matching_node() {
    let dir = TempDir::new().unwrap();
    init_global(dir.path());

    let uuid = global_add_node(dir.path(), "unique-bm25-search-target-xyz");

    // Text query with --no-vector uses BM25; no embedder needed.
    let out = mnem_global(
        dir.path(),
        &[
            "global", "retrieve",
            "unique-bm25-search-target-xyz",
            "--no-vector",
        ],
    )
    .assert()
    .success();
    let stdout = String::from_utf8_lossy(&out.get_output().stdout).to_string();
    assert!(
        stdout.contains(&uuid),
        "text-query retrieve must return the node whose summary matches, got:\n{stdout}"
    );
}

// ── Op-log reflects all writes ────────────────────────────────────────────

/// Each write (add node, tombstone, add node) advances the op-head: after
/// three write ops the log must contain at least three new `op <cid>` lines.
#[test]
fn global_multiple_writes_advance_op_log() {
    let dir = TempDir::new().unwrap();
    init_global(dir.path());

    // Use -n 100 so the count is never silently capped by a default page limit.
    let before_out = mnem_global(dir.path(), &["global", "log", "-n", "100"])
        .assert()
        .success();
    let before_count = String::from_utf8_lossy(&before_out.get_output().stdout)
        .lines()
        .filter(|l| l.starts_with("op "))
        .count();

    // Write 1: add a node.
    let uuid1 = global_add_node(dir.path(), "multi-write-a");
    // Write 2: tombstone it.
    mnem_global(dir.path(), &["global", "tombstone", &uuid1])
        .assert()
        .success();
    // Write 3: add another node.
    global_add_node(dir.path(), "multi-write-b");

    let after_out = mnem_global(dir.path(), &["global", "log", "-n", "100"])
        .assert()
        .success();
    let after_count = String::from_utf8_lossy(&after_out.get_output().stdout)
        .lines()
        .filter(|l| l.starts_with("op "))
        .count();

    assert!(
        after_count >= before_count + 3,
        "three write ops must create at least 3 new ops (was {before_count}, now {after_count})"
    );
}
