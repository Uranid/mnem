//! Regression tests for sprint-fix issues: A5, K8, C8, G6, G_BUG, A3.
//!
//! Each test drives the real `mnem` binary (via `assert_cmd`) against a
//! temp-dir repo and asserts the exit code + output required by the fix.
//!
//! Naming convention: `<fix_id>_<short_description>`.

use std::path::Path;
use std::process::Command;

use assert_cmd::prelude::*;
use tempfile::TempDir;

/// Build a `Command` that runs the cargo-built `mnem` binary with the repo
/// path pre-wired via `-R`. This matches the helper used by all other
/// integration-test files in this crate.
fn mnem(repo: &Path, args: &[&str]) -> Command {
    let mut cmd = Command::cargo_bin("mnem").expect("built mnem binary");
    cmd.current_dir(repo);
    cmd.arg("-R").arg(repo);
    for a in args {
        cmd.arg(a);
    }
    cmd
}

/// Init a repo in `dir` and assert it succeeded. Convenience so each test
/// body focuses on the scenario being checked.
fn init(dir: &Path) {
    mnem(dir, &["init", dir.to_str().unwrap()])
        .assert()
        .success();
}

/// Add a single node (with `--no-embed` to avoid provider calls in CI) and
/// return its UUID string as printed on stdout ("added node <uuid>").
fn add_node(dir: &Path, summary: &str) -> String {
    let out = mnem(dir, &["add", "node", "--summary", summary, "--no-embed"])
        .assert()
        .success();
    let stdout = String::from_utf8_lossy(&out.get_output().stdout).to_string();
    // stdout: "added node <uuid>\n op_id <cid>\n"
    for line in stdout.lines() {
        if let Some(rest) = line.strip_prefix("added node ") {
            return rest.trim().to_string();
        }
    }
    panic!("add node stdout had no 'added node <uuid>' line: {stdout}");
}

/// Return the number of non-empty op-log lines produced by `mnem log
/// --oneline`. Used by K8 to verify no spurious commit was written.
fn op_count(dir: &Path) -> usize {
    let out = mnem(dir, &["log", "--oneline"]).assert().success();
    let stdout = String::from_utf8_lossy(&out.get_output().stdout).to_string();
    stdout.lines().filter(|l| !l.trim().is_empty()).count()
}

/// Grab the first op CID from `mnem log -n 1` (line format: "op <cid>").
fn current_op_cid(dir: &Path) -> String {
    let out = mnem(dir, &["log", "-n", "1"]).assert().success();
    let stdout = String::from_utf8_lossy(&out.get_output().stdout).to_string();
    for line in stdout.lines() {
        if let Some(rest) = line.strip_prefix("op ") {
            return rest.trim().to_string();
        }
    }
    panic!("log -n 1 had no 'op <cid>' line: {stdout}");
}

// ---------------------------------------------------------------------------
// A5 — doctor exits 1 outside a repo
// ---------------------------------------------------------------------------

/// `mnem doctor` run from a directory with no `.mnem` must exit 1 and
/// mention that no mnem repo was found.
#[test]
fn a5_doctor_exits_1_outside_repo() {
    let dir = TempDir::new().unwrap();
    // We intentionally do NOT call init here — the directory has no .mnem.
    let out = mnem(dir.path(), &["doctor"]).assert().failure();
    let stdout = String::from_utf8_lossy(&out.get_output().stdout).to_string();
    assert!(
        stdout.contains("no mnem repo"),
        "doctor output must mention 'no mnem repo', got: {stdout}"
    );
    let code = out.get_output().status.code();
    assert_eq!(code, Some(1), "doctor must exit with code 1, got: {code:?}");
}

// ---------------------------------------------------------------------------
// K8 — delete nonexistent node exits 1 with no spurious commit
// ---------------------------------------------------------------------------

/// Deleting a UUID that was never added must exit 1 and must NOT write a new
/// op to the log (op count is unchanged).
#[test]
fn k8_delete_nonexistent_node_exits_1_no_spurious_commit() {
    let dir = TempDir::new().unwrap();
    init(dir.path());

    let ops_before = op_count(dir.path());

    let nonexistent = "00000000-0000-0000-0000-000000000000";
    mnem(dir.path(), &["delete", nonexistent])
        .assert()
        .failure();

    let ops_after = op_count(dir.path());
    assert_eq!(
        ops_before, ops_after,
        "delete of nonexistent node must not write a new op (before={ops_before}, after={ops_after})"
    );
}

// ---------------------------------------------------------------------------
// C8 — add edge to nonexistent dst exits 1
// ---------------------------------------------------------------------------

/// Adding an edge whose `--to` target does not exist must exit 1 and the
/// error message must mention `--to`.
#[test]
fn c8_add_edge_nonexistent_dst_exits_1() {
    let dir = TempDir::new().unwrap();
    init(dir.path());
    let real_id = add_node(dir.path(), "source node");

    let missing_dst = "00000000-0000-0000-0000-000000000000";
    let out = mnem(
        dir.path(),
        &[
            "add",
            "edge",
            "--from",
            &real_id,
            "--to",
            missing_dst,
            "--label",
            "test",
        ],
    )
    .assert()
    .failure();

    let stderr = String::from_utf8_lossy(&out.get_output().stderr).to_string();
    assert!(
        stderr.contains("--to"),
        "error for missing --to must mention '--to', got: {stderr}"
    );
}

// ---------------------------------------------------------------------------
// G6 — invalid branch names rejected
// ---------------------------------------------------------------------------

/// Branch names that violate git check-ref-format rules must be rejected with
/// exit 1. A syntactically valid name must be accepted (exit 0).
#[test]
fn g6_invalid_branch_names_rejected() {
    let dir = TempDir::new().unwrap();
    init(dir.path());

    // Each of these must be rejected.
    let bad_names = ["my branch", "feat~1", "feat^0", "feat:ure"];
    for name in bad_names {
        let out = mnem(dir.path(), &["branch", "create", name])
            .assert()
            .failure();
        let code = out.get_output().status.code();
        assert_ne!(
            code,
            Some(0),
            "branch name '{name}' should be rejected but exited 0"
        );
    }

    // A valid name must succeed.
    let out = mnem(dir.path(), &["branch", "create", "valid-name"])
        .assert()
        .success();
    let stdout = String::from_utf8_lossy(&out.get_output().stdout).to_string();
    assert!(
        stdout.contains("created branch valid-name"),
        "valid branch name should produce 'created branch valid-name', got: {stdout}"
    );
}

// ---------------------------------------------------------------------------
// G_BUG — branch create --from op-CID rejected
// ---------------------------------------------------------------------------

/// `mnem branch create <name> --from <op-cid>` must exit 1 with a message
/// that explains the CID does not decode as a commit. This guards against
/// silently accepting op-log CIDs (which are Operations, not Commits) and
/// producing a broken branch that crashes `mnem merge` later.
#[test]
fn g_bug_branch_create_from_op_cid_rejected() {
    let dir = TempDir::new().unwrap();
    init(dir.path());
    // Add a node so the log has a real op CID beyond the init anchor.
    add_node(dir.path(), "g_bug test node");

    let op_cid = current_op_cid(dir.path());

    let out = mnem(
        dir.path(),
        &["branch", "create", "test-branch", "--from", &op_cid],
    )
    .assert()
    .failure();

    let stderr = String::from_utf8_lossy(&out.get_output().stderr).to_string();
    assert!(
        stderr.contains("does not decode as a commit"),
        "error must mention 'does not decode as a commit', got: {stderr}"
    );
}

// ---------------------------------------------------------------------------
// A3 — add edge --from nonexistent exits 1
// ---------------------------------------------------------------------------

/// Adding an edge whose `--from` source does not exist must exit 1 and the
/// error message must mention `--from`.
#[test]
fn a3_add_edge_nonexistent_src_exits_1() {
    let dir = TempDir::new().unwrap();
    init(dir.path());
    let real_id = add_node(dir.path(), "dest node");

    let missing_src = "00000000-0000-0000-0000-000000000000";
    let out = mnem(
        dir.path(),
        &[
            "add",
            "edge",
            "--from",
            missing_src,
            "--to",
            &real_id,
            "--label",
            "test",
        ],
    )
    .assert()
    .failure();

    let stderr = String::from_utf8_lossy(&out.get_output().stderr).to_string();
    assert!(
        stderr.contains("--from"),
        "error for missing --from must mention '--from', got: {stderr}"
    );
}
