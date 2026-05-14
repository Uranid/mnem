//! Integration tests for `mnem blame`.
//!
//! Tests:
//!
//! 1. A node with no incoming edges prints `<no incoming edges>`.
//! 2. After `add node A; add node B; add edge B --authored--> A`,
//!    `blame A` lists the edge with src=B.
//! 3. Output header contains the expected columns (`edge_id`, `etype`,
//!    `src`, `in_commit`) when edges are present.
//! 4. `--etype` filter restricts output to matching edge types only.
//! 5. `--first-writer` shows the commit CID that first introduced each
//!    edge, not the current HEAD commit.
//! 6. `--first-writer` on a node with no incoming edges still prints
//!    `<no incoming edges>` and exits zero.
//! 7. An invalid (non-UUID) node string produces a non-zero exit.
//! 8. `--etype` and `--first-writer` compose: the filtered set uses the
//!    correct per-edge first-writer commit, not the current HEAD.
//! 9. Multiple edges under `--first-writer` are tracked independently;
//!    each edge shows its own introducing commit, not a shared one.

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

/// Return the commit CID that is the current HEAD of `repo`.
///
/// Walks: `mnem log -n 1` → op CID → `cat-file --json` → view CID →
/// `cat-file --json` → `heads[0]` (commit CID).
fn current_commit_cid(repo: &Path) -> String {
    let log_out = mnem(repo, &["log", "-n", "1"]).assert().success();
    let log_stdout = String::from_utf8_lossy(&log_out.get_output().stdout).to_string();
    let op_cid = log_stdout
        .lines()
        .find_map(|l| l.strip_prefix("op ").map(|s| s.trim().to_string()))
        .expect("log -n 1 must have an op line");

    let op_out = mnem(repo, &["cat-file", &op_cid, "--json"])
        .assert()
        .success();
    let op_json: serde_json::Value =
        serde_json::from_slice(&op_out.get_output().stdout).expect("op --json must be valid JSON");
    let view_cid = op_json
        .get("view")
        .and_then(|v| v.get("/"))
        .and_then(|c| c.as_str())
        .expect("op.view must be a CID link")
        .to_string();

    let view_out = mnem(repo, &["cat-file", &view_cid, "--json"])
        .assert()
        .success();
    let view_json: serde_json::Value =
        serde_json::from_slice(&view_out.get_output().stdout)
            .expect("view --json must be valid JSON");
    view_json
        .get("heads")
        .and_then(|h| h.as_array())
        .and_then(|a| a.first())
        .and_then(|link| link.get("/"))
        .and_then(|c| c.as_str())
        .expect("view must have at least one head CID")
        .to_string()
}

// ---------------------------------------------------------------------------
// Test 1: no incoming edges
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Test 2: incoming edge appears in output with correct src
// ---------------------------------------------------------------------------

#[test]
fn blame_lists_incoming_edges_with_src() {
    let dir = TempDir::new().unwrap();
    mnem(dir.path(), &["init", dir.path().to_str().unwrap()])
        .assert()
        .success();

    let a_out = mnem(
        dir.path(),
        &["add", "node", "--summary", "target", "--label", "doc", "--no-embed"],
    )
    .assert()
    .success();
    let a_id = extract_node_id(&String::from_utf8_lossy(&a_out.get_output().stdout));

    let b_out = mnem(
        dir.path(),
        &["add", "node", "--summary", "author", "--label", "person", "--no-embed"],
    )
    .assert()
    .success();
    let b_id = extract_node_id(&String::from_utf8_lossy(&b_out.get_output().stdout));

    mnem(
        dir.path(),
        &["add", "edge", "--from", &b_id, "--to", &a_id, "--label", "authored"],
    )
    .assert()
    .success();

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

// ---------------------------------------------------------------------------
// Test 3: header row contains expected column names
// ---------------------------------------------------------------------------

#[test]
fn blame_header_row_contains_expected_columns() {
    let dir = TempDir::new().unwrap();
    mnem(dir.path(), &["init", dir.path().to_str().unwrap()])
        .assert()
        .success();

    let a_out = mnem(dir.path(), &["add", "node", "--summary", "target", "--no-embed"])
        .assert()
        .success();
    let a_id = extract_node_id(&String::from_utf8_lossy(&a_out.get_output().stdout));

    let b_out = mnem(dir.path(), &["add", "node", "--summary", "author", "--no-embed"])
        .assert()
        .success();
    let b_id = extract_node_id(&String::from_utf8_lossy(&b_out.get_output().stdout));

    mnem(
        dir.path(),
        &["add", "edge", "--from", &b_id, "--to", &a_id, "--label", "wrote"],
    )
    .assert()
    .success();

    let out = mnem(dir.path(), &["blame", &a_id]).assert().success();
    let stdout = String::from_utf8_lossy(&out.get_output().stdout).to_string();
    let header = stdout.lines().next().expect("blame must emit at least one line");

    assert!(header.contains("edge_id"), "header must contain 'edge_id'; got: {header}");
    assert!(header.contains("etype"), "header must contain 'etype'; got: {header}");
    assert!(header.contains("src"), "header must contain 'src'; got: {header}");
    assert!(header.contains("in_commit"), "header must contain 'in_commit'; got: {header}");
}

// ---------------------------------------------------------------------------
// Test 4: --etype filter restricts output to matching edge types only
// ---------------------------------------------------------------------------

#[test]
fn blame_etype_filter_restricts_to_matching_edges() {
    let dir = TempDir::new().unwrap();
    mnem(dir.path(), &["init", dir.path().to_str().unwrap()])
        .assert()
        .success();

    let a_out = mnem(dir.path(), &["add", "node", "--summary", "target-a", "--no-embed"])
        .assert()
        .success();
    let a_id = extract_node_id(&String::from_utf8_lossy(&a_out.get_output().stdout));

    let b_out = mnem(dir.path(), &["add", "node", "--summary", "author-b", "--no-embed"])
        .assert()
        .success();
    let b_id = extract_node_id(&String::from_utf8_lossy(&b_out.get_output().stdout));

    let c_out = mnem(dir.path(), &["add", "node", "--summary", "citer-c", "--no-embed"])
        .assert()
        .success();
    let c_id = extract_node_id(&String::from_utf8_lossy(&c_out.get_output().stdout));

    mnem(
        dir.path(),
        &["add", "edge", "--from", &b_id, "--to", &a_id, "--label", "authored"],
    )
    .assert()
    .success();
    mnem(
        dir.path(),
        &["add", "edge", "--from", &c_id, "--to", &a_id, "--label", "cites"],
    )
    .assert()
    .success();

    let filtered = mnem(dir.path(), &["blame", &a_id, "--etype", "authored"])
        .assert()
        .success();
    let out = String::from_utf8_lossy(&filtered.get_output().stdout).to_string();

    assert!(
        out.contains(&b_id),
        "authored-filter output must include src B ({b_id}); got: {out}"
    );
    assert!(
        !out.contains(&c_id),
        "authored-filter output must NOT include src C ({c_id}); got: {out}"
    );
    assert!(out.contains("authored"), "filtered output must show 'authored'; got: {out}");
    assert!(!out.contains("cites"), "filtered output must NOT show 'cites'; got: {out}");
}

// ---------------------------------------------------------------------------
// Test 5: --first-writer shows the introducing commit, not current HEAD
// ---------------------------------------------------------------------------

#[test]
fn blame_first_writer_shows_introducing_commit() {
    let dir = TempDir::new().unwrap();
    mnem(dir.path(), &["init", dir.path().to_str().unwrap()])
        .assert()
        .success();

    let a_out = mnem(dir.path(), &["add", "node", "--summary", "target-fw", "--no-embed"])
        .assert()
        .success();
    let a_id = extract_node_id(&String::from_utf8_lossy(&a_out.get_output().stdout));

    let b_out = mnem(dir.path(), &["add", "node", "--summary", "author-fw", "--no-embed"])
        .assert()
        .success();
    let b_id = extract_node_id(&String::from_utf8_lossy(&b_out.get_output().stdout));

    // op3: add edge B->A. Capture commit CID immediately after.
    mnem(
        dir.path(),
        &["add", "edge", "--from", &b_id, "--to", &a_id, "--label", "authored"],
    )
    .assert()
    .success();
    let commit_at_edge = current_commit_cid(dir.path());

    // op4: advance HEAD without touching the edge.
    mnem(dir.path(), &["add", "node", "--summary", "unrelated", "--no-embed"])
        .assert()
        .success();
    let commit_current = current_commit_cid(dir.path());

    assert_ne!(
        commit_at_edge, commit_current,
        "advancing HEAD must produce a new commit CID"
    );

    // Without --first-writer: in_commit = current HEAD.
    let out_default = mnem(dir.path(), &["blame", &a_id]).assert().success();
    let stdout_default = String::from_utf8_lossy(&out_default.get_output().stdout).to_string();
    assert!(
        stdout_default.contains("in_commit"),
        "default blame must have 'in_commit' header; got: {stdout_default}"
    );
    assert!(
        stdout_default.contains(&commit_current),
        "default blame must show current HEAD commit ({commit_current}); got: {stdout_default}"
    );

    // With --first-writer: shows the commit that introduced the edge.
    let out_fw = mnem(dir.path(), &["blame", &a_id, "--first-writer"])
        .assert()
        .success();
    let stdout_fw = String::from_utf8_lossy(&out_fw.get_output().stdout).to_string();
    assert!(
        stdout_fw.contains("first_writer"),
        "--first-writer blame must have 'first_writer' header; got: {stdout_fw}"
    );
    assert!(
        stdout_fw.contains(&commit_at_edge),
        "--first-writer must show the introducing commit ({commit_at_edge}); got: {stdout_fw}"
    );
    assert!(
        !stdout_fw.contains(&commit_current),
        "--first-writer must NOT show the current HEAD commit ({commit_current}); got: {stdout_fw}"
    );
}

// ---------------------------------------------------------------------------
// Test 6: --first-writer on a node with no edges exits zero
// ---------------------------------------------------------------------------

#[test]
fn blame_first_writer_no_edges_still_succeeds() {
    let dir = TempDir::new().unwrap();
    mnem(dir.path(), &["init", dir.path().to_str().unwrap()])
        .assert()
        .success();
    let out = mnem(dir.path(), &["add", "node", "--summary", "lonely", "--no-embed"])
        .assert()
        .success();
    let node_id = extract_node_id(&String::from_utf8_lossy(&out.get_output().stdout));

    let blamed = mnem(dir.path(), &["blame", &node_id, "--first-writer"])
        .assert()
        .success();
    let stdout = String::from_utf8_lossy(&blamed.get_output().stdout).to_string();
    assert!(
        stdout.contains("<no incoming edges>"),
        "--first-writer on a node with no edges must print '<no incoming edges>'; got: {stdout}"
    );
}

// ---------------------------------------------------------------------------
// Test 7: invalid UUID exits non-zero with an error message
// ---------------------------------------------------------------------------

#[test]
fn blame_invalid_node_uuid_exits_nonzero() {
    let dir = TempDir::new().unwrap();
    mnem(dir.path(), &["init", dir.path().to_str().unwrap()])
        .assert()
        .success();

    let out = mnem(dir.path(), &["blame", "not-a-uuid"]).assert().failure();
    let stderr = String::from_utf8_lossy(&out.get_output().stderr).to_string();
    assert!(
        !stderr.is_empty(),
        "invalid UUID must produce an error message on stderr; got nothing"
    );
}

// ---------------------------------------------------------------------------
// Test 8: --etype and --first-writer compose correctly
// ---------------------------------------------------------------------------

/// Build a repo where two edges of different types are introduced in
/// different operations. Verify that `--etype <t> --first-writer` shows
/// the correct per-type introducing commit, not the current HEAD.
#[test]
fn blame_etype_and_first_writer_compose() {
    let dir = TempDir::new().unwrap();
    mnem(dir.path(), &["init", dir.path().to_str().unwrap()])
        .assert()
        .success();

    let a_out = mnem(dir.path(), &["add", "node", "--summary", "target-ew", "--no-embed"])
        .assert()
        .success();
    let a_id = extract_node_id(&String::from_utf8_lossy(&a_out.get_output().stdout));

    let b_out = mnem(dir.path(), &["add", "node", "--summary", "author-ew", "--no-embed"])
        .assert()
        .success();
    let b_id = extract_node_id(&String::from_utf8_lossy(&b_out.get_output().stdout));

    let c_out = mnem(dir.path(), &["add", "node", "--summary", "citer-ew", "--no-embed"])
        .assert()
        .success();
    let c_id = extract_node_id(&String::from_utf8_lossy(&c_out.get_output().stdout));

    // Introduce "authored" edge first, capture its commit.
    mnem(
        dir.path(),
        &["add", "edge", "--from", &b_id, "--to", &a_id, "--label", "authored"],
    )
    .assert()
    .success();
    let commit_authored = current_commit_cid(dir.path());

    // Advance HEAD (separates commit_authored from commit_cites).
    mnem(dir.path(), &["add", "node", "--summary", "separator", "--no-embed"])
        .assert()
        .success();

    // Introduce "cites" edge, capture its commit.
    mnem(
        dir.path(),
        &["add", "edge", "--from", &c_id, "--to", &a_id, "--label", "cites"],
    )
    .assert()
    .success();
    let commit_cites = current_commit_cid(dir.path());

    // Advance HEAD again so current != either introducing commit.
    mnem(dir.path(), &["add", "node", "--summary", "final", "--no-embed"])
        .assert()
        .success();
    let commit_current = current_commit_cid(dir.path());

    // --etype authored --first-writer: shows commit_authored, not commit_current or commit_cites.
    let out_auth = mnem(dir.path(), &["blame", &a_id, "--etype", "authored", "--first-writer"])
        .assert()
        .success();
    let stdout_auth = String::from_utf8_lossy(&out_auth.get_output().stdout).to_string();
    assert!(
        stdout_auth.contains(&b_id),
        "--etype authored must include src B ({b_id}); got: {stdout_auth}"
    );
    assert!(
        stdout_auth.contains(&commit_authored),
        "--etype authored --first-writer must show commit_authored ({commit_authored}); \
         got: {stdout_auth}"
    );
    assert!(
        !stdout_auth.contains(&c_id),
        "--etype authored must NOT include src C ({c_id}); got: {stdout_auth}"
    );
    assert!(
        !stdout_auth.contains(&commit_current),
        "--etype authored --first-writer must NOT show current HEAD ({commit_current}); \
         got: {stdout_auth}"
    );

    // --etype cites --first-writer: shows commit_cites, not commit_authored.
    let out_cites = mnem(dir.path(), &["blame", &a_id, "--etype", "cites", "--first-writer"])
        .assert()
        .success();
    let stdout_cites = String::from_utf8_lossy(&out_cites.get_output().stdout).to_string();
    assert!(
        stdout_cites.contains(&c_id),
        "--etype cites must include src C ({c_id}); got: {stdout_cites}"
    );
    assert!(
        stdout_cites.contains(&commit_cites),
        "--etype cites --first-writer must show commit_cites ({commit_cites}); \
         got: {stdout_cites}"
    );
    assert!(
        !stdout_cites.contains(&commit_authored),
        "--etype cites --first-writer must NOT show commit_authored ({commit_authored}); \
         got: {stdout_cites}"
    );
}

// ---------------------------------------------------------------------------
// Test 9: multiple edges under --first-writer tracked independently
// ---------------------------------------------------------------------------

/// Add two edges to the same target introduced in different operations.
/// Without `--etype`, both are shown by `blame --first-writer`. Each edge
/// must display its own introducing commit: the line with src=B shows
/// commit_authored, and the line with src=C shows commit_cites. Neither
/// shows the current HEAD commit.
#[test]
fn blame_first_writer_tracks_each_edge_independently() {
    let dir = TempDir::new().unwrap();
    mnem(dir.path(), &["init", dir.path().to_str().unwrap()])
        .assert()
        .success();

    let a_out = mnem(dir.path(), &["add", "node", "--summary", "target-multi", "--no-embed"])
        .assert()
        .success();
    let a_id = extract_node_id(&String::from_utf8_lossy(&a_out.get_output().stdout));

    let b_out = mnem(dir.path(), &["add", "node", "--summary", "author-multi", "--no-embed"])
        .assert()
        .success();
    let b_id = extract_node_id(&String::from_utf8_lossy(&b_out.get_output().stdout));

    let c_out = mnem(dir.path(), &["add", "node", "--summary", "citer-multi", "--no-embed"])
        .assert()
        .success();
    let c_id = extract_node_id(&String::from_utf8_lossy(&c_out.get_output().stdout));

    // Edge B->A introduced first.
    mnem(
        dir.path(),
        &["add", "edge", "--from", &b_id, "--to", &a_id, "--label", "authored"],
    )
    .assert()
    .success();
    let commit_b = current_commit_cid(dir.path());

    // Edge C->A introduced second (different op, different commit).
    mnem(
        dir.path(),
        &["add", "edge", "--from", &c_id, "--to", &a_id, "--label", "cites"],
    )
    .assert()
    .success();
    let commit_c = current_commit_cid(dir.path());

    // Advance HEAD so current HEAD differs from both introducing commits.
    mnem(dir.path(), &["add", "node", "--summary", "extra", "--no-embed"])
        .assert()
        .success();
    let commit_current = current_commit_cid(dir.path());

    // All three commits must be distinct.
    assert_ne!(commit_b, commit_c, "two separate edge ops must produce distinct commits");
    assert_ne!(commit_c, commit_current, "advancing HEAD must produce a new commit");

    let out = mnem(dir.path(), &["blame", &a_id, "--first-writer"])
        .assert()
        .success();
    let stdout = String::from_utf8_lossy(&out.get_output().stdout).to_string();

    // Extract per-src lines to verify per-edge assignment independently.
    let line_b = stdout
        .lines()
        .find(|l| l.contains(&b_id))
        .expect("blame output must contain a line with src B");
    let line_c = stdout
        .lines()
        .find(|l| l.contains(&c_id))
        .expect("blame output must contain a line with src C");

    assert!(
        line_b.contains(&commit_b),
        "src-B line must show commit_b ({commit_b}); got: {line_b}"
    );
    assert!(
        !line_b.contains(&commit_current),
        "src-B line must NOT show current HEAD ({commit_current}); got: {line_b}"
    );
    assert!(
        line_c.contains(&commit_c),
        "src-C line must show commit_c ({commit_c}); got: {line_c}"
    );
    assert!(
        !line_c.contains(&commit_current),
        "src-C line must NOT show current HEAD ({commit_current}); got: {line_c}"
    );
    // Neither line should show the other's commit.
    assert!(
        !line_b.contains(&commit_c),
        "src-B line must NOT show commit_c ({commit_c}); got: {line_b}"
    );
    assert!(
        !line_c.contains(&commit_b),
        "src-C line must NOT show commit_b ({commit_b}); got: {line_c}"
    );
}
