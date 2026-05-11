//! Integration tests for `mnem switch` / `mnem checkout`.
//!
//! Covers the full behavioural contract of branch switching:
//!
//! 1. Switching to an existing branch moves HEAD to the branch tip.
//! 2. Switching to a nonexistent branch returns a non-zero exit code.
//! 3. Switching to the branch already active prints "Already on" and is a no-op.
//! 4. After switching, `View.extra["active_branch"]` holds the full ref name
//!    (BUG-38 regression guard).
//! 5. Round-tripping A -> B -> A leaves HEAD and active_branch correct at
//!    every step.
//! 6. Switching while a merge is in progress is refused.
//! 7. Two branches diverged from a common base: switch moves HEAD correctly
//!    in both directions.

use std::path::Path;
use std::process::Command;

use assert_cmd::prelude::*;
use tempfile::TempDir;

// ---------------------------------------------------------------------------
// Test harness helpers
// ---------------------------------------------------------------------------

/// Build a `Command` that runs the cargo-built `mnem` binary with `-R <repo>`.
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

/// Return the starred branch name from `mnem branch list` (the active branch).
/// The list format is `* <short_name>  -> <cid>` or `  <short_name>  -> <cid>`.
fn active_branch_from_list(dir: &Path) -> String {
    let out = mnem(dir, &["branch", "list"]).assert().success();
    let stdout = String::from_utf8_lossy(&out.get_output().stdout).to_string();
    for line in stdout.lines() {
        if let Some(rest) = line.strip_prefix("* ") {
            // rest is "<short_name>  -> <cid>"
            let short = rest.split_whitespace().next().unwrap_or("").to_string();
            if !short.is_empty() {
                return short;
            }
        }
    }
    panic!("could not find starred branch in `mnem branch list` output:\n{stdout}");
}

// ---------------------------------------------------------------------------
// Test 1: switching to an existing branch moves HEAD to the branch tip
// ---------------------------------------------------------------------------

#[test]
fn switch_to_existing_branch_moves_head() {
    let dir = TempDir::new().unwrap();
    let p = dir.path();
    init(p);

    // Commit one node so `main` has a real tip commit.
    add_node(p, "base commit");
    let main_head = head_cid(p);

    // Create a feature branch at current HEAD.
    mnem(p, &["branch", "create", "feature"]).assert().success();

    // Add another node so main advances beyond feature.
    add_node(p, "main-only commit");
    let main_advanced = head_cid(p);
    assert_ne!(main_head, main_advanced, "main must have advanced");

    // Switch to feature — HEAD should go back to where feature points.
    let out = mnem(p, &["switch", "feature"]).assert().success();
    let stdout = String::from_utf8_lossy(&out.get_output().stdout).to_string();
    assert!(
        stdout.contains("switched to branch 'feature'"),
        "expected switch confirmation, got: {stdout}"
    );

    let after = head_cid(p);
    assert_eq!(
        after, main_head,
        "HEAD after switching to feature should equal the feature branch tip (= main at creation time)"
    );
}

// ---------------------------------------------------------------------------
// Test 2: switching to a nonexistent branch returns a non-zero exit code
// ---------------------------------------------------------------------------

#[test]
fn switch_to_nonexistent_branch_errors() {
    let dir = TempDir::new().unwrap();
    let p = dir.path();
    init(p);
    add_node(p, "a commit");

    let head_before = head_cid(p);

    let out = mnem(p, &["switch", "does-not-exist"]).assert().failure();
    let stderr = String::from_utf8_lossy(&out.get_output().stderr).to_string();
    assert!(
        stderr.contains("does-not-exist"),
        "error message should mention the branch name, got: {stderr}"
    );
    assert!(
        stderr.contains("not found") || stderr.contains("does not exist"),
        "error message should say the branch was not found, got: {stderr}"
    );

    // HEAD must be unchanged.
    let head_after = head_cid(p);
    assert_eq!(
        head_before, head_after,
        "HEAD must not change when switch fails"
    );
}

// ---------------------------------------------------------------------------
// Test 3: switching to the already-active branch prints "Already on" and is a no-op
// ---------------------------------------------------------------------------

#[test]
fn switch_already_on_branch_is_noop() {
    let dir = TempDir::new().unwrap();
    let p = dir.path();
    init(p);
    add_node(p, "a commit");

    let head_before = head_cid(p);

    // We are on `main` after init. Switching to `main` again should be a no-op.
    let out = mnem(p, &["switch", "main"]).assert().success();
    let stdout = String::from_utf8_lossy(&out.get_output().stdout).to_string();
    assert!(
        stdout.contains("Already on 'main'"),
        "expected 'Already on' message, got: {stdout}"
    );

    let head_after = head_cid(p);
    assert_eq!(
        head_before, head_after,
        "HEAD must not change when already on the target branch"
    );
}

// ---------------------------------------------------------------------------
// Test 4: switch updates View.extra["active_branch"] (BUG-38 regression guard)
// ---------------------------------------------------------------------------
//
// The BUG-38 fix writes `refs/heads/<branch>` into `View.extra["active_branch"]`
// so that subsequent commits know which branch ref to advance.  The `mnem branch
// list` output uses this field to place the `*` marker, so checking the starred
// line is the simplest observable proxy for the field.
//
// To exercise a *real* switch (not the "Already on" fast-path), the two
// branches must have different tips, so we advance `main` one commit after
// creating `other`, making their tips diverge.

#[test]
fn switch_updates_active_branch_field() {
    let dir = TempDir::new().unwrap();
    let p = dir.path();
    init(p);
    add_node(p, "base");

    // Create `other` at the current HEAD.
    mnem(p, &["branch", "create", "other"]).assert().success();

    // Advance main one commit so that main_tip != other_tip.
    // Now switching to `other` is a real move (different commit CID).
    add_node(p, "main-only commit");

    // Switch to other — must invoke switch_branch and update active_branch.
    mnem(p, &["switch", "other"]).assert().success();
    let active = active_branch_from_list(p);
    assert_eq!(
        active, "other",
        "active_branch field must point to 'other' after switching"
    );

    // Switch back to main.
    mnem(p, &["switch", "main"]).assert().success();
    let active = active_branch_from_list(p);
    assert_eq!(
        active, "main",
        "active_branch field must point to 'main' after switching back"
    );
}

// ---------------------------------------------------------------------------
// Test 5: round-tripping A -> B -> A leaves HEAD and active_branch correct
// ---------------------------------------------------------------------------

#[test]
fn switch_back_and_forth() {
    let dir = TempDir::new().unwrap();
    let p = dir.path();
    init(p);

    // Commit one node - this will be the tip of `main` and also `feature`
    // (created at the same point).
    add_node(p, "shared base");
    let shared_tip = head_cid(p);

    // Create feature branch at shared_tip.
    mnem(p, &["branch", "create", "feature"]).assert().success();

    // Advance main with one more commit.
    add_node(p, "main extra");
    let main_tip = head_cid(p);
    assert_ne!(shared_tip, main_tip);

    // --- Switch to feature ---
    mnem(p, &["switch", "feature"]).assert().success();
    assert_eq!(
        head_cid(p),
        shared_tip,
        "after switching to feature HEAD should be at shared_tip"
    );
    assert_eq!(
        active_branch_from_list(p),
        "feature",
        "active_branch should be 'feature'"
    );

    // --- Switch back to main ---
    mnem(p, &["switch", "main"]).assert().success();
    assert_eq!(
        head_cid(p),
        main_tip,
        "after switching back to main HEAD should be at main_tip"
    );
    assert_eq!(
        active_branch_from_list(p),
        "main",
        "active_branch should be 'main'"
    );

    // --- Switch to feature again ---
    mnem(p, &["switch", "feature"]).assert().success();
    assert_eq!(
        head_cid(p),
        shared_tip,
        "second switch to feature should land at shared_tip again"
    );
    assert_eq!(
        active_branch_from_list(p),
        "feature",
        "active_branch should be 'feature' again"
    );
}

// ---------------------------------------------------------------------------
// Test 6: switch is refused while a merge is in progress
// ---------------------------------------------------------------------------
//
// Gap 3 fix: after the refused switch, assert HEAD CID and active_branch
// are unchanged — not just that stderr is correct.

#[test]
fn switch_refused_during_merge_in_progress() {
    let dir = TempDir::new().unwrap();
    let p = dir.path();
    init(p);
    add_node(p, "base");

    // Create a second branch so the switch target is a known valid branch.
    // We switch to it first so active_branch is set, then switch back to main
    // (advancing it one commit) to ensure the two branches differ.
    mnem(p, &["branch", "create", "other"]).assert().success();
    add_node(p, "main-only");
    // Now switch to other then back to main so active_branch is reliably set.
    mnem(p, &["switch", "other"]).assert().success();
    mnem(p, &["switch", "main"]).assert().success();

    // Record pre-attempt state.
    let head_before = head_cid(p);
    let active_before = active_branch_from_list(p);
    assert_eq!(active_before, "main", "sanity: we should be on main");

    // Simulate a merge-in-progress by writing a MERGE_HEAD file.
    let mnem_dir = p.join(".mnem");
    std::fs::write(mnem_dir.join("MERGE_HEAD"), "fake-cid-for-test")
        .expect("could not write MERGE_HEAD");

    let out = mnem(p, &["switch", "other"]).assert().failure();
    let stderr = String::from_utf8_lossy(&out.get_output().stderr).to_string();
    assert!(
        stderr.contains("middle of a merge") || stderr.contains("merge in progress"),
        "switch during merge must produce a descriptive error, got: {stderr}"
    );

    // Gap 3: HEAD and active_branch must be unchanged after a refused switch.
    assert_eq!(
        head_cid(p),
        head_before,
        "HEAD must not change when switch is refused (merge in progress)"
    );
    assert_eq!(
        active_branch_from_list(p),
        active_before,
        "active_branch must not change when switch is refused (merge in progress)"
    );
}

// ---------------------------------------------------------------------------
// Test 7: two branches with diverged history — switch moves HEAD correctly
// ---------------------------------------------------------------------------

#[test]
fn switch_diverged_branches_head_correct() {
    let dir = TempDir::new().unwrap();
    let p = dir.path();
    init(p);

    // Shared base.
    add_node(p, "common ancestor");
    let base_tip = head_cid(p);

    // Create branch-a at base.
    mnem(p, &["branch", "create", "branch-a"])
        .assert()
        .success();

    // Add a commit on main (now diverged from branch-a).
    add_node(p, "main-only commit");
    let main_tip = head_cid(p);

    // Switch to branch-a and add its own commit.
    mnem(p, &["switch", "branch-a"]).assert().success();
    assert_eq!(head_cid(p), base_tip, "branch-a should start at base_tip");

    add_node(p, "branch-a-only commit");
    let branch_a_tip = head_cid(p);
    assert_ne!(branch_a_tip, base_tip, "branch-a must have advanced");
    assert_ne!(branch_a_tip, main_tip, "branch-a and main must be diverged");

    // Switch to main - should land at main_tip, not branch_a_tip.
    mnem(p, &["switch", "main"]).assert().success();
    assert_eq!(
        head_cid(p),
        main_tip,
        "after switching to main HEAD must be main_tip"
    );
    assert_eq!(active_branch_from_list(p), "main");

    // Switch back to branch-a - should restore branch_a_tip.
    mnem(p, &["switch", "branch-a"]).assert().success();
    assert_eq!(
        head_cid(p),
        branch_a_tip,
        "after switching to branch-a HEAD must be branch_a_tip"
    );
    assert_eq!(active_branch_from_list(p), "branch-a");
}

// ---------------------------------------------------------------------------
// Test 8: `mnem checkout` is an alias for `mnem switch`
// ---------------------------------------------------------------------------

#[test]
fn checkout_alias_works_like_switch() {
    let dir = TempDir::new().unwrap();
    let p = dir.path();
    init(p);
    add_node(p, "base commit");

    // Create `side` at the current HEAD, then advance main so that `side`
    // and `main` have different tips. This forces a real switch (not the
    // "Already on" fast-path) when we `checkout side`.
    mnem(p, &["branch", "create", "side"]).assert().success();
    add_node(p, "main extra commit after side creation");

    // Use `checkout` spelling — must do a real HEAD move.
    let out = mnem(p, &["checkout", "side"]).assert().success();
    let stdout = String::from_utf8_lossy(&out.get_output().stdout).to_string();
    assert!(
        stdout.contains("switched to branch 'side'"),
        "checkout alias must produce same output as switch, got: {stdout}"
    );
    assert_eq!(active_branch_from_list(p), "side");
}

// ---------------------------------------------------------------------------
// Test 9: switch accepts a fully-qualified ref name (refs/heads/<name>)
// ---------------------------------------------------------------------------

#[test]
fn switch_accepts_fully_qualified_ref() {
    let dir = TempDir::new().unwrap();
    let p = dir.path();
    init(p);
    add_node(p, "a commit");
    let main_head = head_cid(p);

    mnem(p, &["branch", "create", "qbranch"]).assert().success();

    // Add a commit to advance main past qbranch.
    add_node(p, "another commit");

    // Switch using the fully-qualified refname.
    let out = mnem(p, &["switch", "refs/heads/qbranch"])
        .assert()
        .success();
    let stdout = String::from_utf8_lossy(&out.get_output().stdout).to_string();
    // The command prints the name as supplied (either short or full).
    assert!(
        stdout.contains("switched to branch"),
        "switch with full refname should succeed, got: {stdout}"
    );

    assert_eq!(
        head_cid(p),
        main_head,
        "HEAD should be at qbranch tip (= main at creation time)"
    );
}

// ---------------------------------------------------------------------------
// Test 10: BUG-38 guard — same-CID branches (Gap 1)
// ---------------------------------------------------------------------------
//
// When two branches share the same commit CID, the legacy CID-based fallback
// in `branch list` cannot tell which branch is active.  Only
// `extra["active_branch"]` can put the `*` on the correct branch.
//
// Scenario: create branch `b` at the same commit as `main`.  Switch from
// `main` to `b`.  The "Already on" fast-path must NOT fire (because the
// active_branch field still says `main`).  After the switch, `branch list`
// must star `b`.
//
// This exercises the implementation fix that changed the no-op check from
//   `current_head == branch_tip`
// to
//   `current_head == branch_tip && active_branch == full_ref`

#[test]
fn switch_bug38_guard_same_cid_branches() {
    let dir = TempDir::new().unwrap();
    let p = dir.path();
    init(p);

    // Commit one node so main has a real tip CID.
    add_node(p, "shared base node");
    let shared_cid = head_cid(p);

    // Create branch `b` at the SAME commit as `main` (same tip CID).
    mnem(p, &["branch", "create", "b"]).assert().success();

    // Verify both branches exist and `main` is currently active.
    let active_before = active_branch_from_list(p);
    assert_eq!(active_before, "main", "sanity: should start on main");

    // Switch from `main` to `b`.  Both point at the same CID, so the old
    // code would have fired "Already on 'b'" and returned without updating
    // active_branch.  The fixed code checks active_branch too and proceeds.
    let out = mnem(p, &["switch", "b"]).assert().success();
    let stdout = String::from_utf8_lossy(&out.get_output().stdout).to_string();
    // Must say "switched to branch 'b'" — NOT "Already on 'b'".
    assert!(
        stdout.contains("switched to branch 'b'"),
        "switch to same-CID branch must report a real switch, got: {stdout}"
    );

    // HEAD CID unchanged (both branches point there).
    assert_eq!(
        head_cid(p),
        shared_cid,
        "HEAD CID must stay the same after switching between same-CID branches"
    );

    // active_branch must now be `b`, not `main`.
    // With the legacy fallback alone this would fail because the fallback
    // marks whichever branch happens to enumerate first when CIDs are equal.
    let active_after = active_branch_from_list(p);
    assert_eq!(
        active_after, "b",
        "active_branch must be 'b' after switching to it (only extra[\"active_branch\"] can disambiguate)"
    );
}

// ---------------------------------------------------------------------------
// Test 11: post-switch commit advances the correct branch (Gap 2)
// ---------------------------------------------------------------------------
//
// The PURPOSE of BUG-38 is that after switching to a branch, the next
// `mnem add node` (which auto-commits) advances THAT branch's ref and not
// the previously-active one.

#[test]
fn switch_subsequent_commit_advances_correct_branch() {
    let dir = TempDir::new().unwrap();
    let p = dir.path();
    init(p);

    // Initial commit on main.
    add_node(p, "initial commit");
    let main_tip_before = head_cid(p);

    // Create `feature` at the same point.
    mnem(p, &["branch", "create", "feature"]).assert().success();

    // Switch to `feature`.
    mnem(p, &["switch", "feature"]).assert().success();
    assert_eq!(
        active_branch_from_list(p),
        "feature",
        "must be on feature after switch"
    );

    // Commit a new node while on `feature`.
    add_node(p, "feature-only node");
    let feature_tip = head_cid(p);
    assert_ne!(
        feature_tip, main_tip_before,
        "feature tip must have advanced past main's old tip"
    );

    // `branch list` must show `feature` advancing and `main` staying put.
    let out = mnem(p, &["branch", "list"]).assert().success();
    let stdout = String::from_utf8_lossy(&out.get_output().stdout).to_string();

    // `feature` line must contain the new tip CID.
    let feature_line = stdout
        .lines()
        .find(|l| l.contains("feature"))
        .unwrap_or_else(|| panic!("`feature` not found in branch list:\n{stdout}"));
    assert!(
        feature_line.contains(&feature_tip),
        "feature branch tip must show the new commit CID, got: {feature_line}"
    );

    // `main` line must still point at the old tip.
    let main_line = stdout
        .lines()
        .find(|l| {
            // Skip lines that are the star prefix — match the word "main" in the name column.
            let stripped = l.trim_start_matches("* ").trim_start_matches("  ");
            stripped.starts_with("main")
        })
        .unwrap_or_else(|| panic!("`main` not found in branch list:\n{stdout}"));
    assert!(
        main_line.contains(&main_tip_before),
        "main branch tip must NOT have advanced (commit went to feature), got: {main_line}"
    );
}

// ---------------------------------------------------------------------------
// Test 12: switch to a nonexistent branch on a fresh (init-only) repo
//          errors cleanly — no panic (Gap 4)
// ---------------------------------------------------------------------------
//
// A freshly-init'd repo has only the seed-anchor commit and `refs/heads/main`.
// There are no other branch refs yet. Switching to any other branch name must
// fail with a clear error (not a panic), proving the error path in the branch
// resolver is robust even before the user has done any real work.

#[test]
fn switch_empty_repo_errors_cleanly() {
    let dir = TempDir::new().unwrap();
    let p = dir.path();

    // Init only — no user-created commits, no user-created branches.
    init(p);

    // `mnem switch nonexistent-branch` must fail with a clear error.
    // We use a branch name that is guaranteed not to exist in a fresh repo.
    let out = mnem(p, &["switch", "nonexistent-branch"])
        .assert()
        .failure();
    let stderr = String::from_utf8_lossy(&out.get_output().stderr).to_string();
    assert!(
        !stderr.is_empty(),
        "switch to nonexistent branch on fresh repo must produce an error on stderr"
    );
    assert!(
        stderr.contains("nonexistent-branch")
            && (stderr.contains("not found") || stderr.contains("does not exist")),
        "error must mention the branch name and indicate it was not found, got: {stderr}"
    );

    // HEAD must remain at the init anchor commit (not panicked/corrupted).
    // The `mnem status` command must still succeed.
    mnem(p, &["status"]).assert().success();
}
