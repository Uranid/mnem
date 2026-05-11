//! Integration tests for `mnem merge` (B4.3).
//!
//! These tests drive the built `mnem` binary against a temp-dir repo.
//! Every test covers one externally-visible behaviour: the LCA
//! fast-forward path, the clean 3-way path, conflict persistence, the
//! --dry-run side-effect-free preview, and the --abort cleanup flow.

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

/// Init a repo in `dir` (asserts success).
fn init(dir: &Path) {
    mnem(dir, &["init", dir.to_str().unwrap()])
        .assert()
        .success();
}

/// Add a single node with `--no-embed` (avoids provider calls in CI).
fn add_node(dir: &Path, summary: &str) {
    mnem(dir, &["add", "node", "--summary", summary, "--no-embed"])
        .assert()
        .success();
}

/// Return the HEAD commit CID as shown by `mnem status` (`commit     <cid>`).
fn head_cid(dir: &Path) -> String {
    let out = mnem(dir, &["status"]).assert().success();
    let stdout = String::from_utf8_lossy(&out.get_output().stdout).to_string();
    for line in stdout.lines() {
        if let Some(rest) = line.strip_prefix("commit     ") {
            let cid = rest.trim().to_string();
            if !cid.is_empty() && !cid.starts_with('<') {
                return cid;
            }
        }
    }
    panic!("could not extract HEAD CID from `mnem status` output:\n{stdout}");
}

/// Build two diverged branches from a common base and return (main_tip, feature_tip).
/// After this helper, the active branch is `main`.
fn setup_diverged_branches(dir: &Path) -> (String, String) {
    // Common ancestor on main.
    add_node(dir, "shared-base");

    // Create feature branch at this point.
    mnem(dir, &["branch", "create", "feature"])
        .assert()
        .success();

    // Advance main with a unique commit.
    add_node(dir, "main-only");
    let main_tip = head_cid(dir);

    // Switch to feature and add its own commit.
    mnem(dir, &["switch", "feature"]).assert().success();
    add_node(dir, "feature-only");
    let feature_tip = head_cid(dir);

    // Switch back to main so we can merge feature into it.
    mnem(dir, &["switch", "main"]).assert().success();

    assert_ne!(main_tip, feature_tip, "branches must have diverged");
    (main_tip, feature_tip)
}

#[test]
fn merge_missing_branch_errors_actionably() {
    let dir = TempDir::new().unwrap();
    mnem(dir.path(), &["init", dir.path().to_str().unwrap()])
        .assert()
        .success();
    // No branch arg, no --continue / --abort.
    let out = mnem(dir.path(), &["merge"]).assert().failure();
    let stderr = String::from_utf8_lossy(&out.get_output().stderr).to_string();
    assert!(
        stderr.contains("missing <branch>") || stderr.contains("required"),
        "missing-branch diagnostic should mention the missing argument: {stderr}"
    );
}

#[test]
fn merge_already_up_to_date_when_same_commit() {
    let dir = TempDir::new().unwrap();
    mnem(dir.path(), &["init", dir.path().to_str().unwrap()])
        .assert()
        .success();
    mnem(
        dir.path(),
        &["add", "node", "--summary", "seed", "--prop", "k=1"],
    )
    .assert()
    .success();
    // Create a branch at HEAD, then merge it - should be up-to-date.
    mnem(dir.path(), &["branch", "create", "feat"])
        .assert()
        .success();
    let out = mnem(dir.path(), &["merge", "feat"]).assert().success();
    let stdout = String::from_utf8_lossy(&out.get_output().stdout).to_string();
    assert!(
        stdout.contains("already up to date") || stdout.contains("fast-forward"),
        "expected up-to-date or FF, got: {stdout}"
    );
}

#[test]
fn merge_abort_without_in_progress_errors() {
    let dir = TempDir::new().unwrap();
    mnem(dir.path(), &["init", dir.path().to_str().unwrap()])
        .assert()
        .success();
    let out = mnem(dir.path(), &["merge", "--abort"]).assert().failure();
    let stderr = String::from_utf8_lossy(&out.get_output().stderr).to_string();
    assert!(
        stderr.contains("no merge in progress"),
        "abort without in-progress should complain: {stderr}"
    );
}

#[test]
fn merge_help_lists_strategy_and_flags() {
    let out = Command::cargo_bin("mnem")
        .unwrap()
        .args(["merge", "--help"])
        .assert()
        .success();
    let stdout = String::from_utf8_lossy(&out.get_output().stdout).to_string();
    for needle in [
        "--strategy",
        "--dry-run",
        "--continue",
        "--abort",
        "ours",
        "theirs",
        "manual",
    ] {
        assert!(
            stdout.contains(needle),
            "merge --help must surface `{needle}`, got: {stdout}"
        );
    }
}

// ---------------------------------------------------------------------------
// Test 5: clean merge of two branches with non-overlapping commits
// ---------------------------------------------------------------------------

#[test]
fn merge_clean_diverged_branches_succeeds() {
    let dir = TempDir::new().unwrap();
    let p = dir.path();
    init(p);

    let (_main_tip, _feature_tip) = setup_diverged_branches(p);

    // Merge feature into main — non-overlapping commits, no conflicts expected.
    let out = mnem(p, &["merge", "feature"]).assert().success();
    let stdout = String::from_utf8_lossy(&out.get_output().stdout).to_string();
    assert!(
        stdout.contains("fast-forward")
            || stdout.contains("merge:")
            || stdout.contains("already up to date"),
        "merge of non-overlapping branches should succeed with a merge/ff message, got: {stdout}"
    );

    // After a clean merge, no MERGE_HEAD sentinel should exist.
    let merge_head = p.join(".mnem").join("MERGE_HEAD");
    assert!(
        !merge_head.exists(),
        "MERGE_HEAD must not exist after a clean merge"
    );
}

// ---------------------------------------------------------------------------
// Test 6: --dry-run does not persist any state
// ---------------------------------------------------------------------------

#[test]
fn merge_dry_run_persists_no_state() {
    let dir = TempDir::new().unwrap();
    let p = dir.path();
    init(p);

    let (main_tip, _feature_tip) = setup_diverged_branches(p);

    // Capture HEAD CID before the dry-run.
    let head_before = head_cid(p);
    assert_eq!(
        head_before, main_tip,
        "we should be back on main after setup"
    );

    // Run with --dry-run; it should exit 0 and describe what would happen.
    let out = mnem(p, &["merge", "--dry-run", "feature"])
        .assert()
        .success();
    let stdout = String::from_utf8_lossy(&out.get_output().stdout).to_string();
    assert!(
        stdout.contains("[dry-run]"),
        "--dry-run output should contain '[dry-run]' prefix, got: {stdout}"
    );

    // MERGE_HEAD must NOT exist (dry-run writes nothing).
    let mnem_dir = p.join(".mnem");
    assert!(
        !mnem_dir.join("MERGE_HEAD").exists(),
        "--dry-run must not write MERGE_HEAD"
    );
    assert!(
        !mnem_dir.join("ORIG_HEAD").exists(),
        "--dry-run must not write ORIG_HEAD"
    );
    assert!(
        !mnem_dir.join("MERGE_CONFLICTS.json").exists(),
        "--dry-run must not write MERGE_CONFLICTS.json"
    );

    // HEAD CID must be unchanged.
    let head_after = head_cid(p);
    assert_eq!(head_before, head_after, "--dry-run must not advance HEAD");
}

// ---------------------------------------------------------------------------
// Test 7: --strategy=ours is accepted and merge succeeds
// ---------------------------------------------------------------------------

#[test]
fn merge_strategy_ours_accepted() {
    let dir = TempDir::new().unwrap();
    let p = dir.path();
    init(p);

    setup_diverged_branches(p);

    // --strategy=ours should be accepted and produce a successful merge.
    let out = mnem(p, &["merge", "--strategy=ours", "feature"])
        .assert()
        .success();
    let stdout = String::from_utf8_lossy(&out.get_output().stdout).to_string();
    assert!(
        stdout.contains("fast-forward")
            || stdout.contains("merge:")
            || stdout.contains("already up to date"),
        "--strategy=ours merge should succeed, got: {stdout}"
    );
}

// ---------------------------------------------------------------------------
// Test 8: --strategy=theirs is accepted and merge succeeds
// ---------------------------------------------------------------------------

#[test]
fn merge_strategy_theirs_accepted() {
    let dir = TempDir::new().unwrap();
    let p = dir.path();
    init(p);

    setup_diverged_branches(p);

    // --strategy=theirs should be accepted and produce a successful merge.
    let out = mnem(p, &["merge", "--strategy=theirs", "feature"])
        .assert()
        .success();
    let stdout = String::from_utf8_lossy(&out.get_output().stdout).to_string();
    assert!(
        stdout.contains("fast-forward")
            || stdout.contains("merge:")
            || stdout.contains("already up to date"),
        "--strategy=theirs merge should succeed, got: {stdout}"
    );
}

// ---------------------------------------------------------------------------
// Test 9: --abort with in-progress merge removes sentinel files
// ---------------------------------------------------------------------------

#[test]
fn merge_abort_with_in_progress_removes_sentinels() {
    let dir = TempDir::new().unwrap();
    let p = dir.path();
    init(p);

    // Commit something so we have a real CID to use.
    add_node(p, "base-commit");
    let cid = head_cid(p);

    // Manually write the sentinel files to simulate a merge in progress.
    let mnem_dir = p.join(".mnem");
    std::fs::write(mnem_dir.join("MERGE_HEAD"), &cid).expect("could not write MERGE_HEAD");
    std::fs::write(mnem_dir.join("ORIG_HEAD"), &cid).expect("could not write ORIG_HEAD");
    std::fs::write(
        mnem_dir.join("MERGE_CONFLICTS.json"),
        r#"{"schema":"mnem.v1.merge_conflicts","left_head":"fake","right_head":"fake","conflicts":[]}"#,
    )
    .expect("could not write MERGE_CONFLICTS.json");

    assert!(
        mnem_dir.join("MERGE_HEAD").exists(),
        "sanity: MERGE_HEAD must exist before abort"
    );

    // --abort should succeed and clean up.
    let out = mnem(p, &["merge", "--abort"]).assert().success();
    let stdout = String::from_utf8_lossy(&out.get_output().stdout).to_string();
    assert!(
        stdout.contains("aborted") || stdout.contains("abort"),
        "--abort should confirm cancellation, got: {stdout}"
    );

    assert!(
        !mnem_dir.join("MERGE_HEAD").exists(),
        "--abort must remove MERGE_HEAD"
    );
    assert!(
        !mnem_dir.join("ORIG_HEAD").exists(),
        "--abort must remove ORIG_HEAD"
    );
    assert!(
        !mnem_dir.join("MERGE_CONFLICTS.json").exists(),
        "--abort must remove MERGE_CONFLICTS.json"
    );
}

// ---------------------------------------------------------------------------
// Test 10: --continue with no merge in progress errors helpfully
// ---------------------------------------------------------------------------

#[test]
fn merge_continue_without_in_progress_errors() {
    let dir = TempDir::new().unwrap();
    let p = dir.path();
    init(p);

    let out = mnem(p, &["merge", "--continue"]).assert().failure();
    let stderr = String::from_utf8_lossy(&out.get_output().stderr).to_string();
    assert!(
        stderr.contains("no merge in progress"),
        "--continue without MERGE_HEAD must say 'no merge in progress', got: {stderr}"
    );
}

// ---------------------------------------------------------------------------
// Test 11: --continue with an unresolved conflict file errors helpfully
// ---------------------------------------------------------------------------

#[test]
fn merge_continue_with_unresolved_conflicts_errors() {
    let dir = TempDir::new().unwrap();
    let p = dir.path();
    init(p);

    // Commit something to get a valid CID.
    add_node(p, "anchor");
    let cid = head_cid(p);

    let mnem_dir = p.join(".mnem");

    // Write sentinel files with a conflict entry that has no resolution field.
    // MERGE_HEAD and ORIG_HEAD contain the real CID so parse_cid_file succeeds
    // if the code ever gets that far (it should not, since unresolved check
    // comes first).
    std::fs::write(mnem_dir.join("MERGE_HEAD"), &cid).expect("write MERGE_HEAD");
    std::fs::write(mnem_dir.join("ORIG_HEAD"), &cid).expect("write ORIG_HEAD");

    // Build a MERGE_CONFLICTS.json with one conflict that has no resolution.
    // The `category` field is required (snake_case); `node_id` is optional but
    // we include it so the conflict is realistic. No `resolution` field means
    // `--continue` must refuse with an "unresolved" error.
    let conflicts_json = format!(
        r#"{{
  "schema": "mnem.v1.merge_conflicts",
  "left_head": "{cid}",
  "right_head": "{cid}",
  "conflicts": [
    {{
      "node_id": "00000000-0000-0000-0000-000000000001",
      "category": "node_cid_divergence",
      "left": {{"cid": "bafy000left"}},
      "right": {{"cid": "bafy000right"}}
    }}
  ]
}}"#
    );
    std::fs::write(mnem_dir.join("MERGE_CONFLICTS.json"), &conflicts_json)
        .expect("write MERGE_CONFLICTS.json");

    let out = mnem(p, &["merge", "--continue"]).assert().failure();
    let stderr = String::from_utf8_lossy(&out.get_output().stderr).to_string();
    assert!(
        stderr.contains("resolution")
            || stderr.contains("unresolved")
            || stderr.contains("conflict"),
        "--continue with unresolved conflicts must mention resolution/conflict, got: {stderr}"
    );
}

// ---------------------------------------------------------------------------
// Helpers for conflict-producing tests
// ---------------------------------------------------------------------------

/// Return the CID that `refs/heads/<branch>` points to, as shown by
/// `mnem ref list`. Used to verify the ref advances after a merge.
fn branch_tip(dir: &Path, branch: &str) -> String {
    let out = mnem(dir, &["ref", "list"]).assert().success();
    let stdout = String::from_utf8_lossy(&out.get_output().stdout).to_string();
    let prefix = format!("refs/heads/{branch}");
    for line in stdout.lines() {
        let stripped = line.trim();
        if stripped.starts_with(&prefix) {
            if let Some(arrow_pos) = stripped.find("->") {
                let cid = stripped[arrow_pos + 2..].trim().to_string();
                if !cid.is_empty() {
                    return cid;
                }
            }
        }
    }
    panic!("could not find ref for branch '{branch}' in `mnem ref list` output:\n{stdout}");
}

/// Set up two diverged branches that have a real conflict: both branches
/// modify the same node UUID (NODE_CONFLICT_ID) to different content.
///
/// After this helper:
/// - Active branch is `main`
/// - `main` has: shared-base node + conflict node with summary "conflict-node-on-main"
/// - `feature` has: conflict node with summary "conflict-node-on-feature"
/// - The two branches have a real `NodeCidDivergence` conflict on the shared UUID
///
/// Returns `(main_tip, feature_tip)` before the merge.
const NODE_CONFLICT_ID: &str = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa";

fn setup_conflicting_branches(dir: &Path) -> (String, String) {
    // Common base commit.
    add_node(dir, "shared-base");

    // Create feature at this point.
    mnem(dir, &["branch", "create", "feature"])
        .assert()
        .success();

    // Advance main: add the conflict node with "main" content.
    mnem(
        dir,
        &[
            "add",
            "node",
            "--id",
            NODE_CONFLICT_ID,
            "--summary",
            "conflict-node-on-main",
            "--no-embed",
        ],
    )
    .assert()
    .success();
    let main_tip = branch_tip(dir, "main");

    // Switch to feature and add the conflict node with "feature" content.
    mnem(dir, &["switch", "feature"]).assert().success();
    mnem(
        dir,
        &[
            "add",
            "node",
            "--id",
            NODE_CONFLICT_ID,
            "--summary",
            "conflict-node-on-feature",
            "--no-embed",
        ],
    )
    .assert()
    .success();
    let feature_tip = branch_tip(dir, "feature");

    // Switch back to main so the caller can run `mnem merge feature`.
    mnem(dir, &["switch", "main"]).assert().success();

    assert_ne!(main_tip, feature_tip, "branches must have diverged");
    (main_tip, feature_tip)
}

/// Read `.mnem/MERGE_CONFLICTS.json`, add `"resolution": <value>` to every
/// conflict entry, and write it back. Used by the `--continue` tests.
fn resolve_all_conflicts(dir: &Path, resolution: &str) {
    let mc_path = dir.join(".mnem").join("MERGE_CONFLICTS.json");
    let raw = std::fs::read_to_string(&mc_path)
        .expect("MERGE_CONFLICTS.json must exist before resolving");
    let mut data: serde_json::Value =
        serde_json::from_str(&raw).expect("MERGE_CONFLICTS.json must be valid JSON");
    if let Some(conflicts) = data.get_mut("conflicts").and_then(|v| v.as_array_mut()) {
        for entry in conflicts.iter_mut() {
            if let Some(obj) = entry.as_object_mut() {
                obj.insert(
                    "resolution".to_string(),
                    serde_json::Value::String(resolution.to_string()),
                );
            }
        }
    }
    std::fs::write(
        &mc_path,
        serde_json::to_string_pretty(&data).expect("re-serialise"),
    )
    .expect("write resolved MERGE_CONFLICTS.json");
}

// ---------------------------------------------------------------------------
// Test 12 (Gap 1): --continue success path end-to-end
// ---------------------------------------------------------------------------

#[test]
fn merge_continue_resolves_to_completion() {
    let dir = TempDir::new().unwrap();
    let p = dir.path();
    init(p);

    let (main_tip_before, _feature_tip) = setup_conflicting_branches(p);

    let mnem_dir = p.join(".mnem");

    // Trigger a conflict - exits 0 but writes sentinel files.
    let out = mnem(p, &["merge", "feature"]).assert().success();
    let stdout = String::from_utf8_lossy(&out.get_output().stdout).to_string();
    assert!(
        stdout.contains("conflict"),
        "merge of conflicting branches must report conflicts, got: {stdout}"
    );

    // All three sentinel files must exist.
    assert!(
        mnem_dir.join("MERGE_HEAD").exists(),
        "MERGE_HEAD must exist after a conflicting merge"
    );
    assert!(
        mnem_dir.join("ORIG_HEAD").exists(),
        "ORIG_HEAD must exist after a conflicting merge"
    );
    assert!(
        mnem_dir.join("MERGE_CONFLICTS.json").exists(),
        "MERGE_CONFLICTS.json must exist after a conflicting merge"
    );

    // Resolve every conflict entry with "ours" so --continue can proceed.
    resolve_all_conflicts(p, "ours");

    // Run --continue - must exit 0.
    let out = mnem(p, &["merge", "--continue"]).assert().success();
    let stdout = String::from_utf8_lossy(&out.get_output().stdout).to_string();
    assert!(
        stdout.contains("merge continued") || stdout.contains("advanced") || stdout.contains("->"),
        "--continue must report that the merge was finalised, got: {stdout}"
    );

    // All three sentinel files must be gone.
    assert!(
        !mnem_dir.join("MERGE_HEAD").exists(),
        "MERGE_HEAD must be removed after --continue"
    );
    assert!(
        !mnem_dir.join("ORIG_HEAD").exists(),
        "ORIG_HEAD must be removed after --continue"
    );
    assert!(
        !mnem_dir.join("MERGE_CONFLICTS.json").exists(),
        "MERGE_CONFLICTS.json must be removed after --continue"
    );

    // The branch ref must have advanced past the pre-merge tip.
    let main_tip_after = branch_tip(p, "main");
    assert_ne!(
        main_tip_before, main_tip_after,
        "refs/heads/main must advance after --continue"
    );
}

// ---------------------------------------------------------------------------
// Test 13 (Gap 2a): --strategy=ours auto-resolves a real conflict
// ---------------------------------------------------------------------------

#[test]
fn merge_strategy_ours_picks_left_on_conflict() {
    let dir = TempDir::new().unwrap();
    let p = dir.path();
    init(p);

    let (main_tip_before, _feature_tip) = setup_conflicting_branches(p);

    let mnem_dir = p.join(".mnem");

    // --strategy=ours must auto-resolve without writing sentinel files.
    let out = mnem(p, &["merge", "--strategy=ours", "feature"])
        .assert()
        .success();
    let stdout = String::from_utf8_lossy(&out.get_output().stdout).to_string();
    assert!(
        stdout.contains("merge:") || stdout.contains("advanced") || stdout.contains("->"),
        "--strategy=ours on a conflicting merge must produce a merge commit, got: {stdout}"
    );

    // No sentinel files must exist: auto-strategy resolves without pausing.
    assert!(
        !mnem_dir.join("MERGE_HEAD").exists(),
        "--strategy=ours must not leave MERGE_HEAD"
    );
    assert!(
        !mnem_dir.join("ORIG_HEAD").exists(),
        "--strategy=ours must not leave ORIG_HEAD"
    );
    assert!(
        !mnem_dir.join("MERGE_CONFLICTS.json").exists(),
        "--strategy=ours must not leave MERGE_CONFLICTS.json"
    );

    // The branch ref must have advanced.
    let main_tip_after = branch_tip(p, "main");
    assert_ne!(
        main_tip_before, main_tip_after,
        "refs/heads/main must advance after --strategy=ours merge"
    );

    // With --strategy=ours, the left/main side must win the conflict.
    // "ours" = left = the branch we are merging INTO (main), so the merged
    // node must carry the main-side summary and NOT the feature-side one.
    let out = mnem(p, &["get", NODE_CONFLICT_ID]).assert().success();
    let stdout = String::from_utf8_lossy(&out.get_output().stdout).to_string();
    assert!(
        stdout.contains("conflict-node-on-main"),
        "--strategy=ours must pick the left (main) side on conflict, got: {stdout}"
    );
    assert!(
        !stdout.contains("conflict-node-on-feature"),
        "--strategy=ours must NOT pick the right (feature) side on conflict, got: {stdout}"
    );
}

// ---------------------------------------------------------------------------
// Test 14 (Gap 2b): --strategy=theirs auto-resolves a real conflict
// ---------------------------------------------------------------------------

#[test]
fn merge_strategy_theirs_picks_right_on_conflict() {
    let dir = TempDir::new().unwrap();
    let p = dir.path();
    init(p);

    let (main_tip_before, _feature_tip) = setup_conflicting_branches(p);

    let mnem_dir = p.join(".mnem");

    // --strategy=theirs must auto-resolve without writing sentinel files.
    let out = mnem(p, &["merge", "--strategy=theirs", "feature"])
        .assert()
        .success();
    let stdout = String::from_utf8_lossy(&out.get_output().stdout).to_string();
    assert!(
        stdout.contains("merge:") || stdout.contains("advanced") || stdout.contains("->"),
        "--strategy=theirs on a conflicting merge must produce a merge commit, got: {stdout}"
    );

    // No sentinel files must exist.
    assert!(
        !mnem_dir.join("MERGE_HEAD").exists(),
        "--strategy=theirs must not leave MERGE_HEAD"
    );
    assert!(
        !mnem_dir.join("ORIG_HEAD").exists(),
        "--strategy=theirs must not leave ORIG_HEAD"
    );
    assert!(
        !mnem_dir.join("MERGE_CONFLICTS.json").exists(),
        "--strategy=theirs must not leave MERGE_CONFLICTS.json"
    );

    // The branch ref must have advanced.
    let main_tip_after = branch_tip(p, "main");
    assert_ne!(
        main_tip_before, main_tip_after,
        "refs/heads/main must advance after --strategy=theirs merge"
    );

    // With --strategy=theirs, the right/feature side must win the conflict.
    // "theirs" = right = the branch being merged IN (feature), so the merged
    // node must carry the feature-side summary and NOT the main-side one.
    let out = mnem(p, &["get", NODE_CONFLICT_ID]).assert().success();
    let stdout = String::from_utf8_lossy(&out.get_output().stdout).to_string();
    assert!(
        stdout.contains("conflict-node-on-feature"),
        "--strategy=theirs must pick the right (feature) side on conflict, got: {stdout}"
    );
    assert!(
        !stdout.contains("conflict-node-on-main"),
        "--strategy=theirs must NOT pick the left (main) side on conflict, got: {stdout}"
    );
}

// ---------------------------------------------------------------------------
// Test 15 (Gap 3): --continue with MERGE_HEAD but no MERGE_CONFLICTS.json
// ---------------------------------------------------------------------------

#[test]
fn merge_continue_missing_conflicts_json_errors() {
    let dir = TempDir::new().unwrap();
    let p = dir.path();
    init(p);

    // We need a real CID in MERGE_HEAD so parse_cid_file succeeds if the
    // code reaches that far (it should not - the missing-file check comes first).
    add_node(p, "anchor");
    let cid = head_cid(p);

    let mnem_dir = p.join(".mnem");
    std::fs::write(mnem_dir.join("MERGE_HEAD"), &cid).expect("write MERGE_HEAD");
    std::fs::write(mnem_dir.join("ORIG_HEAD"), &cid).expect("write ORIG_HEAD");
    // Deliberately do NOT write MERGE_CONFLICTS.json.

    let out = mnem(p, &["merge", "--continue"]).assert().failure();
    let stderr = String::from_utf8_lossy(&out.get_output().stderr).to_string();
    assert!(
        stderr.contains("MERGE_CONFLICTS")
            || stderr.contains("not found")
            || stderr.contains("conflict"),
        "--continue with missing MERGE_CONFLICTS.json must complain about it, got: {stderr}"
    );
}

// ---------------------------------------------------------------------------
// Test 16 (Gap 4 from audit): --continue with mixed resolutions errors
// ---------------------------------------------------------------------------

/// Writing a MERGE_CONFLICTS.json where one conflict is resolved as "ours"
/// and another as "theirs" must cause `--continue` to fail with a
/// message about mixed picks. The current engine applies one strategy
/// globally; per-conflict resolution is not yet supported.
#[test]
fn merge_continue_mixed_resolutions_errors() {
    let dir = TempDir::new().unwrap();
    let p = dir.path();
    init(p);

    add_node(p, "anchor");
    let cid = head_cid(p);

    let mnem_dir = p.join(".mnem");
    std::fs::write(mnem_dir.join("MERGE_HEAD"), &cid).expect("write MERGE_HEAD");
    std::fs::write(mnem_dir.join("ORIG_HEAD"), &cid).expect("write ORIG_HEAD");

    // Two conflict entries: one resolved "ours", one resolved "theirs".
    let conflicts_json = format!(
        r#"{{
  "schema": "mnem.v1.merge_conflicts",
  "left_head": "{cid}",
  "right_head": "{cid}",
  "conflicts": [
    {{
      "node_id": "00000000-0000-0000-0000-000000000001",
      "category": "node_cid_divergence",
      "left": {{"cid": "bafy000left1"}},
      "right": {{"cid": "bafy000right1"}},
      "resolution": "ours"
    }},
    {{
      "node_id": "00000000-0000-0000-0000-000000000002",
      "category": "node_cid_divergence",
      "left": {{"cid": "bafy000left2"}},
      "right": {{"cid": "bafy000right2"}},
      "resolution": "theirs"
    }}
  ]
}}"#
    );
    std::fs::write(mnem_dir.join("MERGE_CONFLICTS.json"), &conflicts_json)
        .expect("write MERGE_CONFLICTS.json");

    let out = mnem(p, &["merge", "--continue"]).assert().failure();
    let stderr = String::from_utf8_lossy(&out.get_output().stderr).to_string();
    assert!(
        stderr.contains("mixed") || stderr.contains("ours") || stderr.contains("theirs"),
        "--continue with mixed resolutions must mention the mixed-picks problem, got: {stderr}"
    );
}

// ---------------------------------------------------------------------------
// Test 17 (Gap 5 from audit): --continue with empty conflicts list succeeds
// ---------------------------------------------------------------------------

/// When MERGE_CONFLICTS.json exists but its `"conflicts"` array is empty
/// (i.e. the merge was actually clean or the user deleted every entry),
/// `--continue` must treat it as a clean continuation and advance HEAD.
#[test]
fn merge_continue_empty_conflicts_list_succeeds() {
    let dir = TempDir::new().unwrap();
    let p = dir.path();
    init(p);

    // We need a REAL repo state so --continue can actually advance the ref.
    // Set up two diverged branches and trigger a merge that produces sentinel
    // files, then manually replace MERGE_CONFLICTS.json with an empty list.
    let (main_tip_before, _feature_tip) = setup_conflicting_branches(p);
    let mnem_dir = p.join(".mnem");

    // Trigger a conflict to write sentinel files.
    let out = mnem(p, &["merge", "feature"]).assert().success();
    let stdout = String::from_utf8_lossy(&out.get_output().stdout).to_string();
    assert!(
        stdout.contains("conflict"),
        "setup: merge should have produced conflicts, got: {stdout}"
    );

    // Overwrite MERGE_CONFLICTS.json with an empty conflicts list.
    // Keep the real left_head and right_head so the file parses correctly.
    let mc_path = mnem_dir.join("MERGE_CONFLICTS.json");
    let raw = std::fs::read_to_string(&mc_path).expect("MERGE_CONFLICTS.json must exist");
    let mut data: serde_json::Value =
        serde_json::from_str(&raw).expect("MERGE_CONFLICTS.json must be valid JSON");
    *data.get_mut("conflicts").unwrap() = serde_json::Value::Array(vec![]);
    std::fs::write(&mc_path, serde_json::to_string_pretty(&data).unwrap())
        .expect("overwrite MERGE_CONFLICTS.json");

    // --continue with an empty conflicts list must succeed and advance HEAD.
    let out = mnem(p, &["merge", "--continue"]).assert().success();
    let stdout = String::from_utf8_lossy(&out.get_output().stdout).to_string();
    assert!(
        stdout.contains("merge continued") || stdout.contains("advanced") || stdout.contains("->"),
        "--continue with empty conflicts must finalise the merge, got: {stdout}"
    );

    // Sentinel files must be gone.
    assert!(
        !mnem_dir.join("MERGE_HEAD").exists(),
        "MERGE_HEAD must be removed after --continue"
    );
    assert!(
        !mnem_dir.join("ORIG_HEAD").exists(),
        "ORIG_HEAD must be removed after --continue"
    );
    assert!(
        !mnem_dir.join("MERGE_CONFLICTS.json").exists(),
        "MERGE_CONFLICTS.json must be removed after --continue"
    );

    // The branch ref must have advanced.
    let main_tip_after = branch_tip(p, "main");
    assert_ne!(
        main_tip_before, main_tip_after,
        "refs/heads/main must advance after --continue with empty conflicts list"
    );
}
