//! Integration tests for `mnem diff` output format verification.
//!
//! These tests drive the real `mnem` binary and assert the **content** of
//! diff output -- not just the exit code. Each test:
//!
//!   1. Performs a series of operations (init, add node/edge, delete, etc.)
//!   2. Captures two op CIDs from `mnem log`
//!   3. Runs `mnem diff <op_a> <op_b>` (human or `--json`)
//!   4. Asserts specific fields / strings in the output
//!
//! Coverage matrix:
//!   - Human: header fields (op_a, op_b), ref-delta counts, commit-delta line
//!   - Human: node added (+), node removed (-), node changed (~)
//!   - Human: edge added (+)
//!   - Human: no-change diff (same op both sides)
//!   - JSON: top-level keys present (op_a, op_b, commit_a, commit_b, ...)
//!   - JSON: node_deltas entry for added node (type, id, ntype, summary)
//!   - JSON: node_deltas entry for removed node
//!   - JSON: node_deltas entry for changed node (before field)
//!   - JSON: edge_deltas entry for added edge
//!   - JSON: empty diff same-op both sides
//!   - Error: invalid op CID rejected with non-zero exit

use std::path::Path;
use std::process::Command;

use assert_cmd::prelude::*;
use tempfile::TempDir;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

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

/// Add a node and return the UUID printed by `add node`.
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

/// Return the op CID at the top of the log (i.e. HEAD op).
fn head_op_cid(dir: &Path) -> String {
    let out = mnem(dir, &["log", "-n", "1"]).assert().success();
    let stdout = String::from_utf8_lossy(&out.get_output().stdout).to_string();
    for line in stdout.lines() {
        if let Some(rest) = line.strip_prefix("op ") {
            return rest.trim().to_string();
        }
    }
    panic!("log -n 1 had no 'op <cid>' line: {stdout}");
}

/// Return the first two op CIDs from the log: `(newest, second-newest)`.
/// Panics if fewer than two ops exist.
fn two_op_cids(dir: &Path) -> (String, String) {
    let out = mnem(dir, &["log", "-n", "2"]).assert().success();
    let stdout = String::from_utf8_lossy(&out.get_output().stdout).to_string();
    let mut cids: Vec<String> = stdout
        .lines()
        .filter_map(|l| l.strip_prefix("op ").map(|r| r.trim().to_string()))
        .collect();
    assert!(
        cids.len() >= 2,
        "expected at least 2 ops in log, got: {cids:?}"
    );
    let b = cids.remove(0); // newest
    let a = cids.remove(0); // second newest
    (a, b)
}

// ---------------------------------------------------------------------------
// Human-readable format tests
// ---------------------------------------------------------------------------

/// The very first two lines of `mnem diff` must echo the op CIDs supplied.
#[test]
fn human_diff_header_echoes_op_cids() {
    let dir = TempDir::new().unwrap();
    init(dir.path());
    add_node(dir.path(), "first node");

    let (op_a, op_b) = two_op_cids(dir.path());

    let out = mnem(dir.path(), &["diff", &op_a, &op_b]).assert().success();
    let stdout = String::from_utf8_lossy(&out.get_output().stdout).to_string();

    assert!(
        stdout.contains(&format!("op_a {op_a}")),
        "diff output must echo op_a on its own line, got:\n{stdout}"
    );
    assert!(
        stdout.contains(&format!("op_b {op_b}")),
        "diff output must echo op_b on its own line, got:\n{stdout}"
    );
}

/// After adding a node, the diff between the init op and the add-node op
/// must report `node deltas: +1 -0 ~0` (one addition, no removals, no changes).
#[test]
fn human_diff_reports_node_added_tally() {
    let dir = TempDir::new().unwrap();
    init(dir.path());
    add_node(dir.path(), "hello world");

    let (op_a, op_b) = two_op_cids(dir.path());

    let out = mnem(dir.path(), &["diff", &op_a, &op_b]).assert().success();
    let stdout = String::from_utf8_lossy(&out.get_output().stdout).to_string();

    assert!(
        stdout.contains("node deltas: +1 -0 ~0"),
        "expected node tally '+1 -0 ~0' for a single addition, got:\n{stdout}"
    );
}

/// The added node must appear in the diff prefixed with `  + `.
#[test]
fn human_diff_shows_added_node_with_plus_prefix() {
    let dir = TempDir::new().unwrap();
    init(dir.path());
    add_node(dir.path(), "my special node");

    let (op_a, op_b) = two_op_cids(dir.path());

    let out = mnem(dir.path(), &["diff", &op_a, &op_b]).assert().success();
    let stdout = String::from_utf8_lossy(&out.get_output().stdout).to_string();

    // The node_summary format is: `  + <uuid> [<ntype>] "<summary>"`
    assert!(
        stdout.contains("  + "),
        "added node must appear prefixed with '  + ', got:\n{stdout}"
    );
    assert!(
        stdout.contains("\"my special node\""),
        "added node summary must appear quoted in output, got:\n{stdout}"
    );
}

/// The human diff must show the node UUID in the delta lines.
#[test]
fn human_diff_includes_node_uuid_in_delta() {
    let dir = TempDir::new().unwrap();
    init(dir.path());
    let node_id = add_node(dir.path(), "node with a known id");

    let (op_a, op_b) = two_op_cids(dir.path());

    let out = mnem(dir.path(), &["diff", &op_a, &op_b]).assert().success();
    let stdout = String::from_utf8_lossy(&out.get_output().stdout).to_string();

    assert!(
        stdout.contains(&node_id),
        "diff must include the node UUID ({node_id}), got:\n{stdout}"
    );
}

/// Diff in both directions: old-newer and newer-older must both exit 0 but
/// show opposing tally signs (add vs remove).
#[test]
fn human_diff_node_removed_tally_when_reversed() {
    let dir = TempDir::new().unwrap();
    init(dir.path());
    add_node(dir.path(), "reverse test");

    let (op_a, op_b) = two_op_cids(dir.path());

    // Forward: op_a -> op_b => node was added (+1)
    let fwd = mnem(dir.path(), &["diff", &op_a, &op_b]).assert().success();
    let fwd_stdout = String::from_utf8_lossy(&fwd.get_output().stdout).to_string();
    assert!(
        fwd_stdout.contains("node deltas: +1 -0 ~0"),
        "forward diff must show +1 added: got:\n{fwd_stdout}"
    );

    // Reversed: op_b -> op_a => node appears removed (-1)
    let rev = mnem(dir.path(), &["diff", &op_b, &op_a]).assert().success();
    let rev_stdout = String::from_utf8_lossy(&rev.get_output().stdout).to_string();
    assert!(
        rev_stdout.contains("node deltas: +0 -1 ~0"),
        "reversed diff must show -1 removed, got:\n{rev_stdout}"
    );
    // The removed node must be prefixed with "  - "
    assert!(
        rev_stdout.contains("  - "),
        "removed node must appear prefixed with '  - ', got:\n{rev_stdout}"
    );
}

/// After adding two nodes, diffing the first commit against the second shows
/// a tally of +2 (both nodes added relative to init).
#[test]
fn human_diff_two_nodes_added_shows_correct_tally() {
    let dir = TempDir::new().unwrap();
    init(dir.path());
    add_node(dir.path(), "alpha");
    // Record op after first add
    let op_after_first = head_op_cid(dir.path());
    add_node(dir.path(), "beta");
    let op_after_second = head_op_cid(dir.path());

    // init -> after_second: two nodes added relative to init.
    // Get the init op CID.
    let all_out = mnem(dir.path(), &["log", "-n", "3"]).assert().success();
    let all_stdout = String::from_utf8_lossy(&all_out.get_output().stdout).to_string();
    let mut cids: Vec<String> = all_stdout
        .lines()
        .filter_map(|l| l.strip_prefix("op ").map(|r| r.trim().to_string()))
        .collect();
    // cids[0]=after_second, cids[1]=after_first, cids[2]=init
    assert!(cids.len() >= 3, "expected 3 ops, got: {cids:?}");
    let op_init = cids.pop().unwrap();

    // diff from init to after_second: should show +2 nodes
    let out = mnem(dir.path(), &["diff", &op_init, &op_after_second])
        .assert()
        .success();
    let stdout = String::from_utf8_lossy(&out.get_output().stdout).to_string();
    assert!(
        stdout.contains("node deltas: +2 -0 ~0"),
        "expected +2 nodes added from init to after two adds, got:\n{stdout}"
    );

    // diff from after_first to after_second: should show +1 node (only beta)
    let out2 = mnem(dir.path(), &["diff", &op_after_first, &op_after_second])
        .assert()
        .success();
    let stdout2 = String::from_utf8_lossy(&out2.get_output().stdout).to_string();
    assert!(
        stdout2.contains("node deltas: +1 -0 ~0"),
        "expected +1 node added between the two add ops, got:\n{stdout2}"
    );
    assert!(
        stdout2.contains("\"beta\""),
        "only 'beta' was added in the second step, got:\n{stdout2}"
    );
}

/// After deleting a node, the diff must show a removal (`-1`).
#[test]
fn human_diff_node_deleted_shows_minus_tally() {
    let dir = TempDir::new().unwrap();
    init(dir.path());
    let node_id = add_node(dir.path(), "to be deleted");
    let op_before_delete = head_op_cid(dir.path());

    // Delete the node.
    mnem(dir.path(), &["delete", &node_id]).assert().success();
    let op_after_delete = head_op_cid(dir.path());

    let out = mnem(dir.path(), &["diff", &op_before_delete, &op_after_delete])
        .assert()
        .success();
    let stdout = String::from_utf8_lossy(&out.get_output().stdout).to_string();

    assert!(
        stdout.contains("node deltas: +0 -1 ~0"),
        "expected -1 for a deleted node, got:\n{stdout}"
    );
    assert!(
        stdout.contains("  - "),
        "deleted node must appear prefixed with '  - ', got:\n{stdout}"
    );
    assert!(
        stdout.contains("\"to be deleted\""),
        "deleted node summary must appear in diff, got:\n{stdout}"
    );
}

/// The `ref deltas` summary line must be present and formatted correctly.
#[test]
fn human_diff_ref_delta_line_is_present() {
    let dir = TempDir::new().unwrap();
    init(dir.path());
    add_node(dir.path(), "ref-delta-test");

    let (op_a, op_b) = two_op_cids(dir.path());

    let out = mnem(dir.path(), &["diff", &op_a, &op_b]).assert().success();
    let stdout = String::from_utf8_lossy(&out.get_output().stdout).to_string();

    // Format: "ref deltas: +N -N ~N"
    assert!(
        stdout.contains("ref deltas:"),
        "diff must include 'ref deltas:' summary line, got:\n{stdout}"
    );
}

/// The `commit deltas: a=<cid> -> b=<cid>` line must be present.
#[test]
fn human_diff_commit_delta_line_is_present() {
    let dir = TempDir::new().unwrap();
    init(dir.path());
    add_node(dir.path(), "commit-delta-test");

    let (op_a, op_b) = two_op_cids(dir.path());

    let out = mnem(dir.path(), &["diff", &op_a, &op_b]).assert().success();
    let stdout = String::from_utf8_lossy(&out.get_output().stdout).to_string();

    assert!(
        stdout.contains("commit deltas:"),
        "diff must include 'commit deltas:' line, got:\n{stdout}"
    );
    // The format is "commit deltas: a=<cid> -> b=<cid>"
    assert!(
        stdout.contains(" -> "),
        "commit deltas line must contain ' -> ' separator, got:\n{stdout}"
    );
}

/// Same op on both sides: the node tally must be 0/0/0.
#[test]
fn human_diff_same_op_both_sides_is_empty() {
    let dir = TempDir::new().unwrap();
    init(dir.path());
    add_node(dir.path(), "idempotent node");

    let op = head_op_cid(dir.path());

    let out = mnem(dir.path(), &["diff", &op, &op]).assert().success();
    let stdout = String::from_utf8_lossy(&out.get_output().stdout).to_string();

    assert!(
        stdout.contains("node deltas: +0 -0 ~0"),
        "same-op diff must show zero node deltas, got:\n{stdout}"
    );
    assert!(
        stdout.contains("edge deltas: +0 -0 ~0"),
        "same-op diff must show zero edge deltas, got:\n{stdout}"
    );
}

/// After adding an edge, the diff must show `edge deltas: +1 -0 ~0`.
#[test]
fn human_diff_edge_added_shows_plus_tally() {
    let dir = TempDir::new().unwrap();
    init(dir.path());
    let src = add_node(dir.path(), "source");
    let dst = add_node(dir.path(), "destination");
    let op_before_edge = head_op_cid(dir.path());

    mnem(
        dir.path(),
        &[
            "add", "edge", "--from", &src, "--to", &dst, "--label", "knows",
        ],
    )
    .assert()
    .success();
    let op_after_edge = head_op_cid(dir.path());

    let out = mnem(dir.path(), &["diff", &op_before_edge, &op_after_edge])
        .assert()
        .success();
    let stdout = String::from_utf8_lossy(&out.get_output().stdout).to_string();

    assert!(
        stdout.contains("edge deltas: +1 -0 ~0"),
        "expected +1 edge added, got:\n{stdout}"
    );
    assert!(
        stdout.contains("  + "),
        "added edge must appear prefixed with '  + ', got:\n{stdout}"
    );
    // edge_summary format: "<src> -[<label>]-> <dst>"
    assert!(
        stdout.contains("-[knows]->"),
        "edge label 'knows' must appear in edge summary, got:\n{stdout}"
    );
}

/// The edge summary must contain both the src and dst node UUIDs.
#[test]
fn human_diff_edge_summary_includes_src_and_dst_uuids() {
    let dir = TempDir::new().unwrap();
    init(dir.path());
    let src = add_node(dir.path(), "edge-src");
    let dst = add_node(dir.path(), "edge-dst");
    let op_before = head_op_cid(dir.path());

    mnem(
        dir.path(),
        &[
            "add", "edge", "--from", &src, "--to", &dst, "--label", "links",
        ],
    )
    .assert()
    .success();
    let op_after = head_op_cid(dir.path());

    let out = mnem(dir.path(), &["diff", &op_before, &op_after])
        .assert()
        .success();
    let stdout = String::from_utf8_lossy(&out.get_output().stdout).to_string();

    assert!(
        stdout.contains(&src),
        "edge delta line must include src UUID ({src}), got:\n{stdout}"
    );
    assert!(
        stdout.contains(&dst),
        "edge delta line must include dst UUID ({dst}), got:\n{stdout}"
    );
}

// ---------------------------------------------------------------------------
// JSON format tests
// ---------------------------------------------------------------------------

/// `mnem diff --json` must produce valid JSON with the required top-level keys.
#[test]
fn json_diff_has_required_top_level_keys() {
    let dir = TempDir::new().unwrap();
    init(dir.path());
    add_node(dir.path(), "json top-level test");

    let (op_a, op_b) = two_op_cids(dir.path());

    let out = mnem(dir.path(), &["diff", &op_a, &op_b, "--json"])
        .assert()
        .success();
    let stdout = String::from_utf8_lossy(&out.get_output().stdout).to_string();

    let parsed: serde_json::Value =
        serde_json::from_str(&stdout).expect("--json output must be valid JSON");

    let required_keys = [
        "op_a",
        "op_b",
        "commit_a",
        "commit_b",
        "ref_deltas",
        "node_deltas",
        "edge_deltas",
    ];
    for key in &required_keys {
        assert!(
            parsed.get(key).is_some(),
            "JSON diff must have top-level key '{key}', got:\n{stdout}"
        );
    }
}

/// The `op_a` and `op_b` fields in JSON output must match the CIDs we passed.
#[test]
fn json_diff_op_a_and_op_b_match_supplied_cids() {
    let dir = TempDir::new().unwrap();
    init(dir.path());
    add_node(dir.path(), "cid-match test");

    let (op_a, op_b) = two_op_cids(dir.path());

    let out = mnem(dir.path(), &["diff", &op_a, &op_b, "--json"])
        .assert()
        .success();
    let stdout = String::from_utf8_lossy(&out.get_output().stdout).to_string();
    let parsed: serde_json::Value = serde_json::from_str(&stdout).unwrap();

    assert_eq!(
        parsed["op_a"].as_str().unwrap_or(""),
        op_a,
        "op_a field must match the supplied CID"
    );
    assert_eq!(
        parsed["op_b"].as_str().unwrap_or(""),
        op_b,
        "op_b field must match the supplied CID"
    );
}

/// After adding a node, `node_deltas` must contain one entry with
/// `type: "added"`, the correct `ntype`, and the correct `summary`.
#[test]
fn json_diff_node_added_delta_has_correct_fields() {
    let dir = TempDir::new().unwrap();
    init(dir.path());
    let node_id = add_node(dir.path(), "json added node");

    let (op_a, op_b) = two_op_cids(dir.path());

    let out = mnem(dir.path(), &["diff", &op_a, &op_b, "--json"])
        .assert()
        .success();
    let stdout = String::from_utf8_lossy(&out.get_output().stdout).to_string();
    let parsed: serde_json::Value = serde_json::from_str(&stdout).unwrap();

    let deltas = parsed["node_deltas"]
        .as_array()
        .expect("node_deltas must be an array");
    assert_eq!(
        deltas.len(),
        1,
        "expected exactly 1 node delta for a single add, got: {deltas:?}"
    );

    let delta = &deltas[0];
    assert_eq!(
        delta["type"].as_str().unwrap_or(""),
        "added",
        "delta type must be 'added', got: {delta}"
    );
    assert_eq!(
        delta["id"].as_str().unwrap_or(""),
        node_id,
        "delta id must match the added node's UUID"
    );
    assert_eq!(
        delta["summary"].as_str().unwrap_or(""),
        "json added node",
        "delta summary must match the node's summary"
    );
    // The default ntype when none is supplied is "Fact".
    assert!(
        delta.get("ntype").is_some(),
        "delta must include ntype field"
    );
    // No 'before' field for an addition.
    assert!(
        delta.get("before").is_none() || delta["before"].is_null(),
        "added node delta must not have a 'before' state, got: {delta}"
    );
}

/// After removing a node (reversing the diff), `node_deltas` must contain
/// one entry with `type: "removed"`.
#[test]
fn json_diff_node_removed_delta_has_type_removed() {
    let dir = TempDir::new().unwrap();
    init(dir.path());
    let node_id = add_node(dir.path(), "json removed node");

    let (op_a, op_b) = two_op_cids(dir.path());

    // Reversed: op_b -> op_a sees the node as removed.
    let out = mnem(dir.path(), &["diff", &op_b, &op_a, "--json"])
        .assert()
        .success();
    let stdout = String::from_utf8_lossy(&out.get_output().stdout).to_string();
    let parsed: serde_json::Value = serde_json::from_str(&stdout).unwrap();

    let deltas = parsed["node_deltas"]
        .as_array()
        .expect("node_deltas must be an array");
    assert_eq!(
        deltas.len(),
        1,
        "expected exactly 1 removed-node delta, got: {deltas:?}"
    );

    let delta = &deltas[0];
    assert_eq!(
        delta["type"].as_str().unwrap_or(""),
        "removed",
        "delta type must be 'removed' when node disappears from op_a to op_b"
    );
    assert_eq!(
        delta["id"].as_str().unwrap_or(""),
        node_id,
        "removed delta must carry the node UUID"
    );
    assert_eq!(
        delta["summary"].as_str().unwrap_or(""),
        "json removed node",
        "removed delta must carry the node summary"
    );
}

/// After deleting a node (`mnem delete`), the JSON diff must show the
/// node as removed with the correct id and summary.
#[test]
fn json_diff_node_deleted_shows_removed_delta_with_id_and_summary() {
    let dir = TempDir::new().unwrap();
    init(dir.path());
    let node_id = add_node(dir.path(), "to be deleted for json");
    let op_before = head_op_cid(dir.path());

    mnem(dir.path(), &["delete", &node_id]).assert().success();
    let op_after = head_op_cid(dir.path());

    let out = mnem(dir.path(), &["diff", &op_before, &op_after, "--json"])
        .assert()
        .success();
    let stdout = String::from_utf8_lossy(&out.get_output().stdout).to_string();
    let parsed: serde_json::Value = serde_json::from_str(&stdout).unwrap();

    let deltas = parsed["node_deltas"]
        .as_array()
        .expect("node_deltas must be an array");
    assert_eq!(
        deltas.len(),
        1,
        "expected exactly 1 node delta after delete, got: {deltas:?}"
    );

    let delta = &deltas[0];
    assert_eq!(
        delta["type"].as_str().unwrap_or(""),
        "removed",
        "deleted node must appear with type 'removed', got: {delta}"
    );
    assert_eq!(
        delta["id"].as_str().unwrap_or(""),
        node_id,
        "removed delta must carry the correct node UUID"
    );
    assert_eq!(
        delta["summary"].as_str().unwrap_or(""),
        "to be deleted for json",
        "removed delta must carry the original node summary"
    );
    // No 'before' field for a removed entry.
    assert!(
        delta.get("before").is_none() || delta["before"].is_null(),
        "removed delta must not have a 'before' field, got: {delta}"
    );
}

/// After adding an edge, `edge_deltas` must contain one entry with
/// `type: "added"`, the correct `label`, `src`, and `dst`.
#[test]
fn json_diff_edge_added_delta_has_correct_fields() {
    let dir = TempDir::new().unwrap();
    init(dir.path());
    let src = add_node(dir.path(), "json-edge-src");
    let dst = add_node(dir.path(), "json-edge-dst");
    let op_before = head_op_cid(dir.path());

    mnem(
        dir.path(),
        &[
            "add", "edge", "--from", &src, "--to", &dst, "--label", "relates",
        ],
    )
    .assert()
    .success();
    let op_after = head_op_cid(dir.path());

    let out = mnem(dir.path(), &["diff", &op_before, &op_after, "--json"])
        .assert()
        .success();
    let stdout = String::from_utf8_lossy(&out.get_output().stdout).to_string();
    let parsed: serde_json::Value = serde_json::from_str(&stdout).unwrap();

    let edge_deltas = parsed["edge_deltas"]
        .as_array()
        .expect("edge_deltas must be an array");
    assert_eq!(
        edge_deltas.len(),
        1,
        "expected exactly 1 edge delta for a single add-edge, got: {edge_deltas:?}"
    );

    let delta = &edge_deltas[0];
    assert_eq!(
        delta["type"].as_str().unwrap_or(""),
        "added",
        "edge delta type must be 'added'"
    );
    assert_eq!(
        delta["label"].as_str().unwrap_or(""),
        "relates",
        "edge delta label must match the edge type"
    );
    assert_eq!(
        delta["src"].as_str().unwrap_or(""),
        src,
        "edge delta src must match the source node UUID"
    );
    assert_eq!(
        delta["dst"].as_str().unwrap_or(""),
        dst,
        "edge delta dst must match the destination node UUID"
    );
    // No 'before' field for a new edge.
    assert!(
        delta.get("before").is_none() || delta["before"].is_null(),
        "added edge delta must not have a 'before' field, got: {delta}"
    );
}

/// Same op on both sides with `--json` must produce empty delta arrays.
#[test]
fn json_diff_same_op_both_sides_produces_empty_deltas() {
    let dir = TempDir::new().unwrap();
    init(dir.path());
    add_node(dir.path(), "idempotent json test");

    let op = head_op_cid(dir.path());

    let out = mnem(dir.path(), &["diff", &op, &op, "--json"])
        .assert()
        .success();
    let stdout = String::from_utf8_lossy(&out.get_output().stdout).to_string();
    let parsed: serde_json::Value = serde_json::from_str(&stdout).unwrap();

    let node_deltas = parsed["node_deltas"].as_array().unwrap();
    let edge_deltas = parsed["edge_deltas"].as_array().unwrap();
    let ref_added = parsed["ref_deltas"]["added"].as_array().unwrap();
    let ref_removed = parsed["ref_deltas"]["removed"].as_array().unwrap();
    let ref_changed = parsed["ref_deltas"]["changed"].as_array().unwrap();

    assert!(
        node_deltas.is_empty(),
        "same-op JSON diff must have empty node_deltas, got: {node_deltas:?}"
    );
    assert!(
        edge_deltas.is_empty(),
        "same-op JSON diff must have empty edge_deltas, got: {edge_deltas:?}"
    );
    assert!(
        ref_added.is_empty() && ref_removed.is_empty() && ref_changed.is_empty(),
        "same-op JSON diff must have empty ref_deltas sub-arrays"
    );
}

/// The `ref_deltas` field must have three sub-arrays: `added`, `removed`, `changed`.
#[test]
fn json_diff_ref_deltas_has_three_sub_arrays() {
    let dir = TempDir::new().unwrap();
    init(dir.path());
    add_node(dir.path(), "ref-deltas-structure test");

    let (op_a, op_b) = two_op_cids(dir.path());

    let out = mnem(dir.path(), &["diff", &op_a, &op_b, "--json"])
        .assert()
        .success();
    let stdout = String::from_utf8_lossy(&out.get_output().stdout).to_string();
    let parsed: serde_json::Value = serde_json::from_str(&stdout).unwrap();

    let ref_deltas = parsed["ref_deltas"]
        .as_object()
        .expect("ref_deltas must be an object");
    assert!(
        ref_deltas.contains_key("added"),
        "ref_deltas must have 'added' key"
    );
    assert!(
        ref_deltas.contains_key("removed"),
        "ref_deltas must have 'removed' key"
    );
    assert!(
        ref_deltas.contains_key("changed"),
        "ref_deltas must have 'changed' key"
    );
    assert!(
        ref_deltas["added"].is_array(),
        "ref_deltas.added must be an array"
    );
    assert!(
        ref_deltas["removed"].is_array(),
        "ref_deltas.removed must be an array"
    );
    assert!(
        ref_deltas["changed"].is_array(),
        "ref_deltas.changed must be an array"
    );
}

/// `commit_a` and `commit_b` in JSON output must be non-null strings when
/// the ops are backed by real commits.
#[test]
fn json_diff_commit_a_and_b_are_non_null_strings() {
    let dir = TempDir::new().unwrap();
    init(dir.path());
    add_node(dir.path(), "commit-cids test");

    let (op_a, op_b) = two_op_cids(dir.path());

    let out = mnem(dir.path(), &["diff", &op_a, &op_b, "--json"])
        .assert()
        .success();
    let stdout = String::from_utf8_lossy(&out.get_output().stdout).to_string();
    let parsed: serde_json::Value = serde_json::from_str(&stdout).unwrap();

    assert!(
        parsed["commit_a"].is_string(),
        "commit_a must be a non-null string when the op has a commit, got:\n{stdout}"
    );
    assert!(
        parsed["commit_b"].is_string(),
        "commit_b must be a non-null string when the op has a commit, got:\n{stdout}"
    );
    // The two commit CIDs must differ (op_a and op_b are different ops).
    assert_ne!(
        parsed["commit_a"].as_str().unwrap(),
        parsed["commit_b"].as_str().unwrap(),
        "commit_a and commit_b must differ when diffing two different ops"
    );
}

/// Multiple nodes added: JSON node_deltas count matches the actual number.
#[test]
fn json_diff_multiple_nodes_added_delta_count_matches() {
    let dir = TempDir::new().unwrap();
    init(dir.path());
    let op_init = head_op_cid(dir.path());

    add_node(dir.path(), "first json multi");
    add_node(dir.path(), "second json multi");
    add_node(dir.path(), "third json multi");
    let op_after = head_op_cid(dir.path());

    let out = mnem(dir.path(), &["diff", &op_init, &op_after, "--json"])
        .assert()
        .success();
    let stdout = String::from_utf8_lossy(&out.get_output().stdout).to_string();
    let parsed: serde_json::Value = serde_json::from_str(&stdout).unwrap();

    let deltas = parsed["node_deltas"].as_array().unwrap();
    assert_eq!(
        deltas.len(),
        3,
        "expected 3 node_deltas for 3 added nodes, got: {deltas:?}"
    );
    // All must have type "added"
    for delta in deltas {
        assert_eq!(
            delta["type"].as_str().unwrap_or(""),
            "added",
            "all deltas must be 'added', got: {delta}"
        );
    }
}

/// Verify the ntype field in JSON delta matches the ntype given at add time.
#[test]
fn json_diff_node_delta_ntype_matches_actual_ntype() {
    let dir = TempDir::new().unwrap();
    init(dir.path());

    let out = mnem(
        dir.path(),
        &[
            "add",
            "node",
            "--summary",
            "typed node",
            "--ntype",
            "Event",
            "--no-embed",
        ],
    )
    .assert()
    .success();
    // Capture node UUID to verify later.
    let stdout_add = String::from_utf8_lossy(&out.get_output().stdout).to_string();
    let node_id = stdout_add
        .lines()
        .find_map(|l| l.strip_prefix("added node ").map(|r| r.trim().to_string()))
        .expect("add stdout had no 'added node ...' line");

    let (op_a, op_b) = two_op_cids(dir.path());

    let out = mnem(dir.path(), &["diff", &op_a, &op_b, "--json"])
        .assert()
        .success();
    let stdout = String::from_utf8_lossy(&out.get_output().stdout).to_string();
    let parsed: serde_json::Value = serde_json::from_str(&stdout).unwrap();

    let deltas = parsed["node_deltas"].as_array().unwrap();
    let our_delta = deltas.iter().find(|d| d["id"].as_str() == Some(&node_id));
    let our_delta = our_delta.expect("should find the delta for the typed node");

    assert_eq!(
        our_delta["ntype"].as_str().unwrap_or(""),
        "Event",
        "ntype in JSON delta must match the '--ntype Event' supplied at add time"
    );
}

// ---------------------------------------------------------------------------
// Error / edge-case tests
// ---------------------------------------------------------------------------

/// Supplying an invalid CID (not a valid multibase CID) must exit non-zero.
#[test]
fn diff_invalid_op_cid_exits_nonzero() {
    let dir = TempDir::new().unwrap();
    init(dir.path());

    mnem(dir.path(), &["diff", "not-a-real-cid", "also-not-a-cid"])
        .assert()
        .failure();
}

/// Supplying `HEAD` as op_a and `HEAD` as op_b (both the current op)
/// must work (HEAD is a supported alias).
#[test]
fn diff_head_alias_is_accepted() {
    let dir = TempDir::new().unwrap();
    init(dir.path());
    add_node(dir.path(), "HEAD alias test");

    let op = head_op_cid(dir.path());

    // HEAD HEAD must equal op op
    let out1 = mnem(dir.path(), &["diff", "HEAD", "HEAD"])
        .assert()
        .success();
    let out2 = mnem(dir.path(), &["diff", &op, &op]).assert().success();

    let stdout1 = String::from_utf8_lossy(&out1.get_output().stdout).to_string();
    let stdout2 = String::from_utf8_lossy(&out2.get_output().stdout).to_string();

    // Both must report zero deltas (same op).
    assert!(
        stdout1.contains("node deltas: +0 -0 ~0"),
        "HEAD HEAD diff must show no node deltas, got:\n{stdout1}"
    );
    assert!(
        stdout2.contains("node deltas: +0 -0 ~0"),
        "<op> <op> diff must show no node deltas, got:\n{stdout2}"
    );
}

/// `mnem diff` without any arguments must exit non-zero (missing required args).
#[test]
fn diff_missing_args_exits_nonzero() {
    let dir = TempDir::new().unwrap();
    init(dir.path());

    mnem(dir.path(), &["diff"]).assert().failure();
}

/// Using `--json` with `HEAD HEAD` must still produce valid JSON with
/// empty delta arrays.
#[test]
fn json_diff_head_head_is_valid_empty_json() {
    let dir = TempDir::new().unwrap();
    init(dir.path());
    add_node(dir.path(), "json head-head test");

    let out = mnem(dir.path(), &["diff", "HEAD", "HEAD", "--json"])
        .assert()
        .success();
    let stdout = String::from_utf8_lossy(&out.get_output().stdout).to_string();

    let parsed: serde_json::Value =
        serde_json::from_str(&stdout).expect("HEAD HEAD --json must be valid JSON");

    assert!(
        parsed["node_deltas"].as_array().unwrap().is_empty(),
        "HEAD HEAD JSON diff must have empty node_deltas"
    );
    assert!(
        parsed["edge_deltas"].as_array().unwrap().is_empty(),
        "HEAD HEAD JSON diff must have empty edge_deltas"
    );
}

// ---------------------------------------------------------------------------
// Gap 1: Changed node delta tests (~N path)
// ---------------------------------------------------------------------------

/// After overwriting a node with the same UUID but a different summary,
/// the human diff must show `node deltas: +0 -0 ~1` and a `  ~ ` prefix line
/// containing both the before and after summaries.
#[test]
fn human_diff_node_changed_tally() {
    let dir = TempDir::new().unwrap();
    init(dir.path());

    // Pin a stable UUID so both add-node calls go to the same node key.
    let node_uuid = "01234567-89ab-cdef-0123-456789abcdef";

    // First commit: add the node with summary "version-one".
    mnem(
        dir.path(),
        &[
            "add",
            "node",
            "--id",
            node_uuid,
            "--summary",
            "version-one",
            "--no-embed",
        ],
    )
    .assert()
    .success();
    let op_a = head_op_cid(dir.path());

    // Second commit: re-add the SAME UUID with a different summary.
    // Transaction::add_node inserts/overwrites the entry at that NodeId key;
    // the prolly diff sees the same key with a new value CID -> Changed.
    mnem(
        dir.path(),
        &[
            "add",
            "node",
            "--id",
            node_uuid,
            "--summary",
            "version-two",
            "--no-embed",
        ],
    )
    .assert()
    .success();
    let op_b = head_op_cid(dir.path());

    let out = mnem(dir.path(), &["diff", &op_a, &op_b]).assert().success();
    let stdout = String::from_utf8_lossy(&out.get_output().stdout).to_string();

    // Tally must show exactly one changed node, zero added, zero removed.
    assert!(
        stdout.contains("node deltas: +0 -0 ~1"),
        "overwriting a node must show ~1 in tally, got:\n{stdout}"
    );

    // Find the specific line that carries the '  ~ ' prefix and verify that
    // BOTH summaries and the '->' separator appear on that same line.
    // The implementation (print_node_entry for Changed) outputs:
    //   "  ~ <before_summary> -> <after_summary>"
    // so "version-one", "->", and "version-two" must all co-exist on one line.
    let changed_line = stdout
        .lines()
        .find(|l| l.contains("  ~ "))
        .unwrap_or_else(|| panic!("no '  ~ ' line found in diff output:\n{stdout}"));

    assert!(
        changed_line.contains("\"version-one\""),
        "the '  ~ ' line must contain the before summary 'version-one', got line:\n{changed_line}"
    );
    assert!(
        changed_line.contains(" -> "),
        "the '  ~ ' line must contain the '->' separator, got line:\n{changed_line}"
    );
    assert!(
        changed_line.contains("\"version-two\""),
        "the '  ~ ' line must contain the after summary 'version-two', got line:\n{changed_line}"
    );
}

/// After overwriting a node, the JSON diff must contain a `node_deltas` entry
/// with `"type": "changed"`, a `"before"` object carrying the old ntype and
/// summary, and the top-level fields reflecting the new (after) state.
#[test]
fn json_diff_node_changed_has_before_field() {
    let dir = TempDir::new().unwrap();
    init(dir.path());

    let node_uuid = "fedcba98-7654-3210-fedc-ba9876543210";

    // First commit: version-one.
    mnem(
        dir.path(),
        &[
            "add",
            "node",
            "--id",
            node_uuid,
            "--summary",
            "before-summary",
            "--ntype",
            "Fact",
            "--no-embed",
        ],
    )
    .assert()
    .success();
    let op_a = head_op_cid(dir.path());

    // Second commit: same UUID, different summary.
    mnem(
        dir.path(),
        &[
            "add",
            "node",
            "--id",
            node_uuid,
            "--summary",
            "after-summary",
            "--ntype",
            "Fact",
            "--no-embed",
        ],
    )
    .assert()
    .success();
    let op_b = head_op_cid(dir.path());

    let out = mnem(dir.path(), &["diff", &op_a, &op_b, "--json"])
        .assert()
        .success();
    let stdout = String::from_utf8_lossy(&out.get_output().stdout).to_string();
    let parsed: serde_json::Value = serde_json::from_str(&stdout).unwrap();

    let deltas = parsed["node_deltas"]
        .as_array()
        .expect("node_deltas must be an array");
    assert_eq!(
        deltas.len(),
        1,
        "expected exactly 1 node delta for a single overwrite, got: {deltas:?}"
    );

    let delta = &deltas[0];

    // Top-level type must be "changed".
    assert_eq!(
        delta["type"].as_str().unwrap_or(""),
        "changed",
        "delta type must be 'changed', got: {delta}"
    );

    // Top-level id must match the node UUID.
    assert_eq!(
        delta["id"].as_str().unwrap_or(""),
        node_uuid,
        "delta id must match the node UUID, got: {delta}"
    );

    // Top-level summary must reflect the AFTER (new) state.
    assert_eq!(
        delta["summary"].as_str().unwrap_or(""),
        "after-summary",
        "top-level summary must be the after state, got: {delta}"
    );

    // The 'before' field must be present and carry the old (before) state.
    let before = delta
        .get("before")
        .expect("changed delta must have a 'before' field");
    assert!(
        !before.is_null(),
        "before must not be null for a changed delta, got: {delta}"
    );
    assert_eq!(
        before["summary"].as_str().unwrap_or(""),
        "before-summary",
        "before.summary must be the old summary, got: {before}"
    );
    assert_eq!(
        before["ntype"].as_str().unwrap_or(""),
        "Fact",
        "before.ntype must match the old ntype, got: {before}"
    );

    // Sanity: before and after summaries must differ.
    assert_ne!(
        delta["summary"].as_str().unwrap_or(""),
        before["summary"].as_str().unwrap_or(""),
        "before and after summaries must differ for a changed delta"
    );
}

// ---------------------------------------------------------------------------
// Gap 2: Edge removal tests
// ---------------------------------------------------------------------------
//
// NOTE: Why there is no integration test for `edge deltas: ~N` (Changed edges):
//
// `DiffEntry::Changed` for an edge requires the SAME prolly key (EdgeId) to
// appear in both op_a and op_b with DIFFERENT value CIDs. However the CLI
// command `mnem add edge` always generates a fresh EdgeId via `EdgeId::new_v7()`
// (see `crates/mnem-cli/src/commands/add.rs`, `add_edge`). A brand-new UUID
// key can never collide with a key from a prior commit, so the prolly diff can
// only produce Added or Removed entries for edges - never Changed.
//
// The Changed branch of `edge_delta_json` and `print_edge_entry` is therefore
// architecturally unreachable through the CLI. It is covered by a dedicated
// unit test inside `crates/mnem-cli/src/commands/diff.rs`
// (`edge_delta_json_changed_entry_produces_changed_delta`) that calls the
// function directly with a synthetic `DiffEntry::Changed`.
//

/// Reversing an edge-addition diff shows `edge deltas: +0 -1 ~0` and a
/// `  - ` prefix on the edge delta line.
#[test]
fn human_diff_edge_removed_tally() {
    let dir = TempDir::new().unwrap();
    init(dir.path());
    let src = add_node(dir.path(), "edge-rm-src");
    let dst = add_node(dir.path(), "edge-rm-dst");
    let op_before_edge = head_op_cid(dir.path());

    mnem(
        dir.path(),
        &[
            "add",
            "edge",
            "--from",
            &src,
            "--to",
            &dst,
            "--label",
            "depends_on",
        ],
    )
    .assert()
    .success();
    let op_after_edge = head_op_cid(dir.path());

    // Reversed diff: op_after_edge -> op_before_edge.
    // From the diff's perspective the edge is absent in op_b (before_edge),
    // so it appears as a Removed delta.
    let out = mnem(dir.path(), &["diff", &op_after_edge, &op_before_edge])
        .assert()
        .success();
    let stdout = String::from_utf8_lossy(&out.get_output().stdout).to_string();

    assert!(
        stdout.contains("edge deltas: +0 -1 ~0"),
        "reversed edge diff must show -1 removed, got:\n{stdout}"
    );
    assert!(
        stdout.contains("  - "),
        "removed edge must appear prefixed with '  - ', got:\n{stdout}"
    );
    // The edge_summary format: "<src> -[<label>]-> <dst>"
    assert!(
        stdout.contains("-[depends_on]->"),
        "removed edge label must appear in delta line, got:\n{stdout}"
    );
    assert!(
        stdout.contains(&src),
        "removed edge delta must include src UUID ({src}), got:\n{stdout}"
    );
    assert!(
        stdout.contains(&dst),
        "removed edge delta must include dst UUID ({dst}), got:\n{stdout}"
    );
}

/// Reversing an edge-addition diff produces a JSON `edge_deltas` entry with
/// `"type": "removed"` and the correct `label`, `src`, and `dst` fields.
#[test]
fn json_diff_edge_removed_delta() {
    let dir = TempDir::new().unwrap();
    init(dir.path());
    let src = add_node(dir.path(), "json-edge-rm-src");
    let dst = add_node(dir.path(), "json-edge-rm-dst");
    let op_before_edge = head_op_cid(dir.path());

    mnem(
        dir.path(),
        &[
            "add", "edge", "--from", &src, "--to", &dst, "--label", "owns",
        ],
    )
    .assert()
    .success();
    let op_after_edge = head_op_cid(dir.path());

    // Reversed diff: edge present in op_a (after_edge), absent in op_b
    // (before_edge) -> Removed entry in edge_deltas.
    let out = mnem(
        dir.path(),
        &["diff", &op_after_edge, &op_before_edge, "--json"],
    )
    .assert()
    .success();
    let stdout = String::from_utf8_lossy(&out.get_output().stdout).to_string();
    let parsed: serde_json::Value = serde_json::from_str(&stdout).unwrap();

    let edge_deltas = parsed["edge_deltas"]
        .as_array()
        .expect("edge_deltas must be an array");
    assert_eq!(
        edge_deltas.len(),
        1,
        "expected exactly 1 edge delta for a reversed add-edge, got: {edge_deltas:?}"
    );

    let delta = &edge_deltas[0];
    assert_eq!(
        delta["type"].as_str().unwrap_or(""),
        "removed",
        "edge delta type must be 'removed' when edge disappears from op_a to op_b, got: {delta}"
    );
    assert_eq!(
        delta["label"].as_str().unwrap_or(""),
        "owns",
        "edge delta label must match the edge type, got: {delta}"
    );
    assert_eq!(
        delta["src"].as_str().unwrap_or(""),
        src,
        "edge delta src must match the source node UUID, got: {delta}"
    );
    assert_eq!(
        delta["dst"].as_str().unwrap_or(""),
        dst,
        "edge delta dst must match the destination node UUID, got: {delta}"
    );
    // Removed entries have no 'before' field (the diff.rs implementation
    // only sets before on Changed edges).
    assert!(
        delta.get("before").is_none() || delta["before"].is_null(),
        "removed edge delta must not have a 'before' field, got: {delta}"
    );
}
