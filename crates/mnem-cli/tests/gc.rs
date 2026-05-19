//! Integration tests for `mnem gc` and `mnem gc --force`.
//!
//! ## What `mnem gc` does
//!
//! - Without `--force`: dry-run. Walks the full content-addressed DAG from
//!   all known refs, counts reachable vs. total blocks, reports unreachable
//!   count. Does NOT modify the store.
//! - With `--force`: deletes every block that is not reachable from any known
//!   ref (branches, tags, remote-tracking refs). Prints "gc: removed N
//!   block(s)" and exits 0.
//!
//! ## How we create unreachable blocks
//!
//! The mnem blockstore is a content-addressed DAG. Every block reachable via
//! CID links from any ref (branches, HEAD, remote-tracking refs) is "live".
//! A block is "garbage" only if NO ref's DAG contains a CID link to it.
//!
//! Normal CLI operations (`mnem delete`, `mnem branch delete`) do NOT create
//! orphaned blocks because the parent-commit chain keeps all historical blocks
//! reachable. The only way to create truly orphaned blocks is to write bytes
//! directly into the redb blockstore without committing them into any object.
//!
//! These tests inject orphaned blocks via the Rust blockstore API (the same
//! approach used in `tests/reindex_lift_legacy.rs`) and then verify that
//! `mnem gc --force` detects and removes them.

use std::path::Path;
use std::process::Command;
use std::sync::Arc;

use assert_cmd::prelude::*;
use tempfile::TempDir;

// Rust API for direct blockstore manipulation (same pattern as reindex tests).
use mnem_backend_redb::open_or_init;
use mnem_core::codec::hash_to_cid;
use mnem_core::store::Blockstore;

// ---------------------------------------------------------------------------
// Shared helpers (mirrors the pattern used in merge.rs, diff.rs, etc.)
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

/// Add a node without triggering an embedding provider and return its UUID.
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

/// Run `mnem gc` (dry-run) and return stdout as a String.
fn gc_dry_run(dir: &Path) -> String {
    let out = mnem(dir, &["gc"]).assert().success();
    String::from_utf8_lossy(&out.get_output().stdout).to_string()
}

/// Run `mnem gc --force` and return stdout as a String.
fn gc_force(dir: &Path) -> String {
    let out = mnem(dir, &["gc", "--force"]).assert().success();
    String::from_utf8_lossy(&out.get_output().stdout).to_string()
}

/// Open the redb blockstore for `dir` and return it.
///
/// We use this to inject orphaned blocks that are not referenced by any
/// committed object - the only reliable way to create garbage for gc tests.
fn open_blockstore(dir: &Path) -> Arc<dyn Blockstore> {
    let db_path = dir.join(".mnem").join("repo.redb");
    let (bs, _ohs, _cfg) = open_or_init(&db_path).expect("open redb");
    bs
}

/// Write a single raw-bytes block directly into the blockstore.
/// The block is content-addressed (SHA-256 / DAG-CBOR) but is NOT referenced
/// by any committed object, making it genuine garbage.
///
/// Returns the CID of the injected block so callers can verify its deletion after gc.
fn inject_orphaned_block(dir: &Path, payload: &[u8]) -> mnem_core::id::Cid {
    use serde::Serialize;

    // A small wrapper so serde_ipld_dagcbor can encode it as a CBOR map.
    #[derive(Serialize)]
    struct GarbagePayload<'a> {
        _kind: &'a str,
        data: &'a [u8],
    }

    let val = GarbagePayload {
        _kind: "gc_test_garbage",
        data: payload,
    };

    let bs = open_blockstore(dir);
    let (raw_bytes, cid) = hash_to_cid(&val).expect("hash_to_cid");
    // put_trusted: we just computed the CID ourselves, so it is correct.
    bs.put_trusted(cid.clone(), raw_bytes)
        .expect("put_trusted orphan block");
    cid
}

// ---------------------------------------------------------------------------
// Test 1: Dry-run reports unreachable blocks but does NOT delete them
// ---------------------------------------------------------------------------

/// After injecting an orphaned block, `mnem gc` (without --force) must report
/// the unreachable count in stdout and exit 0, but must NOT delete the block.
/// A second call must report the same count (confirming no deletion occurred).
#[test]
fn gc_dry_run_reports_unreachable_blocks_without_deleting() {
    let dir = TempDir::new().unwrap();
    let p = dir.path();
    init(p);

    // Inject one orphaned block directly into the blockstore.
    let orphan_cid = inject_orphaned_block(p, b"orphan-payload-1");

    // Dry-run must report exactly 1 unreachable block (pinned count, not just
    // "some block" string - a stale output like "0 unreachable" must fail here).
    let stdout = gc_dry_run(p);
    assert!(
        stdout.contains("gc: 1 unreachable block(s) found"),
        "dry-run must report exactly 'gc: 1 unreachable block(s) found', got: {stdout}"
    );

    // Running gc dry-run a second time must report the SAME count - idempotency at
    // the block level, not just "some blocks are unreachable".
    let stdout2 = gc_dry_run(p);
    assert!(
        stdout2.contains("gc: 1 unreachable block(s) found"),
        "second dry-run must still report 'gc: 1 unreachable block(s) found' (idempotent, nothing deleted), got: {stdout2}"
    );

    // Verify at blockstore level: the orphan block must still exist after both
    // dry-runs (a dry-run that accidentally deletes would fail here).
    {
        let bs = open_blockstore(p);
        assert!(
            bs.has(&orphan_cid).expect("has() after dry-runs"),
            "dry-run must NOT delete blocks from the store; orphan CID still expected present"
        );
    }
}

// ---------------------------------------------------------------------------
// Test 2: --force happy path - orphaned blocks are actually deleted
// ---------------------------------------------------------------------------

/// After injecting an orphaned block, `mnem gc --force` must:
/// - exit 0
/// - print "gc: removed N block(s)" with N >= 1
/// A subsequent dry-run must report "no unreachable blocks (store is clean)".
#[test]
fn gc_force_deletes_unreachable_blocks() {
    let dir = TempDir::new().unwrap();
    let p = dir.path();
    init(p);

    // Inject an orphaned block to give gc something to collect.
    // Keep the CID so we can verify blockstore state after gc.
    let orphan_cid = inject_orphaned_block(p, b"orphan-payload-force-test");

    // Sanity: dry-run must see the orphaned block before force-collect.
    let before = gc_dry_run(p);
    assert!(
        before.contains("unreachable block(s) found"),
        "precondition: dry-run must see the orphaned block, got: {before}"
    );

    // Force-collect.
    let stdout = gc_force(p);
    assert!(
        stdout.contains("gc: removed 1 block(s)"),
        "gc --force must report 'gc: removed 1 block(s)', got: {stdout}"
    );

    // After force-gc, the store must be clean.
    let after = gc_dry_run(p);
    assert!(
        after.contains("no unreachable blocks") || after.contains("store is clean"),
        "after gc --force the store must be clean, got: {after}"
    );

    // Verify at blockstore level: the orphan block must actually be gone.
    // A broken GC that prints the right message but skips deletion would fail here.
    {
        let bs = open_blockstore(p);
        assert!(
            !bs.has(&orphan_cid).expect("has() after gc --force"),
            "gc --force must actually delete the orphan block from the store"
        );
    }
}

// ---------------------------------------------------------------------------
// Test 3: --force on a clean repo succeeds gracefully
// ---------------------------------------------------------------------------

/// On a repo with no unreachable blocks, `mnem gc --force` must exit 0 and
/// print a "nothing to collect" (or equivalent) message rather than an error.
#[test]
fn gc_force_on_clean_repo_exits_gracefully() {
    let dir = TempDir::new().unwrap();
    let p = dir.path();
    init(p);

    // Add a node but do NOT inject any orphaned blocks.
    add_node(p, "live-node");

    // Record the live block count before gc so we can verify nothing was deleted.
    let block_count_before = {
        let bs = open_blockstore(p);
        bs.all_cids()
            .expect("all_cids")
            .expect("redb supports enumeration")
            .len()
        // bs drops here
    };

    let stdout = gc_force(p);
    assert!(
        stdout.contains("nothing to collect")
            || stdout.contains("no unreachable blocks")
            || stdout.contains("store is clean"),
        "gc --force on clean repo must say nothing to collect, got: {stdout}"
    );

    // Verify at blockstore level: a broken GC that says "nothing to collect"
    // but then deletes live blocks would fail this check.
    let block_count_after = {
        let bs = open_blockstore(p);
        bs.all_cids()
            .expect("all_cids")
            .expect("redb supports enumeration")
            .len()
    };
    assert_eq!(
        block_count_before, block_count_after,
        "gc --force on clean repo must not delete any blocks (before={block_count_before}, after={block_count_after})"
    );
}

// ---------------------------------------------------------------------------
// Test 4: --force outside any repo exits non-zero
// ---------------------------------------------------------------------------

/// Running `mnem gc --force` in a directory that is not a mnem repo must exit
/// with a non-zero status code (repo-open failure).
#[test]
fn gc_force_outside_repo_fails() {
    let dir = TempDir::new().unwrap();
    // We do NOT call `init()` - this is not a mnem repo.
    let mut cmd = Command::cargo_bin("mnem").expect("built mnem binary");
    cmd.current_dir(dir.path());
    // Explicitly pass -R to a non-repo path so the binary cannot fall back
    // to any parent-directory discovery.
    cmd.arg("-R").arg(dir.path());
    cmd.args(["gc", "--force"]);
    cmd.assert().failure();
}

// ---------------------------------------------------------------------------
// Test 5: Post-gc repo is still fully functional
// ---------------------------------------------------------------------------

/// After `mnem gc --force`, the repo must still be usable:
/// - `mnem log` exits 0 and lists commits
/// - Nodes that were NOT orphaned remain retrievable via `mnem get`
/// - The repo accepts new commits
#[test]
fn gc_force_repo_remains_functional() {
    let dir = TempDir::new().unwrap();
    let p = dir.path();
    init(p);

    // Add a live node (committed, will survive gc).
    let survivor_uuid = add_node(p, "survivor-node");

    // Inject an orphaned block (not reachable from any ref) to be collected.
    inject_orphaned_block(p, b"orphan-payload-functional-test");

    // Force-collect.
    gc_force(p);

    // `mnem log` must still work and show commits.
    let out = mnem(p, &["log"]).assert().success();
    let log_stdout = String::from_utf8_lossy(&out.get_output().stdout).to_string();
    assert!(
        log_stdout.contains("op "),
        "mnem log must still show ops after gc, got: {log_stdout}"
    );

    // `mnem get` on the survivor must still work.
    let out = mnem(p, &["get", &survivor_uuid]).assert().success();
    let get_stdout = String::from_utf8_lossy(&out.get_output().stdout).to_string();
    assert!(
        get_stdout.contains("survivor-node"),
        "survivor node must still be retrievable via get after gc, got: {get_stdout}"
    );

    // The repo must still accept new commits after gc.
    add_node(p, "post-gc-node");
    let out = mnem(p, &["log", "-n", "1"]).assert().success();
    let last_log = String::from_utf8_lossy(&out.get_output().stdout).to_string();
    assert!(
        last_log.contains("op "),
        "new commit after gc must appear in log, got: {last_log}"
    );
}

// ---------------------------------------------------------------------------
// Test 6: --force removes multiple orphaned blocks
// ---------------------------------------------------------------------------

/// Injecting several orphaned blocks then running gc --force must report that
/// multiple blocks were removed.
#[test]
fn gc_force_removes_multiple_orphaned_blocks() {
    let dir = TempDir::new().unwrap();
    let p = dir.path();
    init(p);

    // Inject three distinct orphaned blocks and capture all three CIDs.
    // Each call opens and immediately drops the blockstore (redb single-writer safe).
    let orphan_a = inject_orphaned_block(p, b"orphan-block-a");
    let orphan_b = inject_orphaned_block(p, b"orphan-block-b");
    let orphan_c = inject_orphaned_block(p, b"orphan-block-c");

    // Verify all three orphans are present in the blockstore before gc.
    // Scoped so the Arc drops before any CLI call (redb single-writer constraint).
    {
        let bs = open_blockstore(p);
        assert!(
            bs.has(&orphan_a).expect("has(orphan_a) before gc"),
            "orphan_a must exist before gc"
        );
        assert!(
            bs.has(&orphan_b).expect("has(orphan_b) before gc"),
            "orphan_b must exist before gc"
        );
        assert!(
            bs.has(&orphan_c).expect("has(orphan_c) before gc"),
            "orphan_c must exist before gc"
        );
        // bs drops here
    }

    // Dry-run should report unreachable blocks.
    let dry = gc_dry_run(p);
    assert!(
        dry.contains("unreachable block(s) found"),
        "dry-run must report unreachable blocks, got: {dry}"
    );

    // Force-collect: must report exactly 3 blocks removed.
    let stdout = gc_force(p);
    assert!(
        stdout.contains("gc: removed 3 block(s)"),
        "gc --force must report 'gc: removed 3 block(s)', got: {stdout}"
    );

    // Store must be clean afterwards.
    let after = gc_dry_run(p);
    assert!(
        after.contains("no unreachable blocks") || after.contains("store is clean"),
        "store must be clean after collecting orphaned blocks, got: {after}"
    );

    // Verify all three orphans are actually gone from the blockstore.
    {
        let bs = open_blockstore(p);
        assert!(
            !bs.has(&orphan_a).expect("has(orphan_a) after gc"),
            "orphan_a must be deleted by gc --force"
        );
        assert!(
            !bs.has(&orphan_b).expect("has(orphan_b) after gc"),
            "orphan_b must be deleted by gc --force"
        );
        assert!(
            !bs.has(&orphan_c).expect("has(orphan_c) after gc"),
            "orphan_c must be deleted by gc --force"
        );
    }
}

// ---------------------------------------------------------------------------
// Test 7: Dry-run exits 0 on a clean (freshly-initialized) repo
// ---------------------------------------------------------------------------

/// A fresh repo with no orphaned blocks has no unreachable blocks. The
/// dry-run must exit 0 and report "no unreachable blocks (store is clean)".
#[test]
fn gc_dry_run_clean_repo_exits_zero() {
    let dir = TempDir::new().unwrap();
    let p = dir.path();
    init(p);

    // Single node, no orphaned blocks.
    add_node(p, "only-node");

    let stdout = gc_dry_run(p);
    assert!(
        stdout.contains("no unreachable blocks") || stdout.contains("store is clean"),
        "dry-run on clean repo must report clean store, got: {stdout}"
    );
}

// ---------------------------------------------------------------------------
// Test 8: gc --force does not affect live nodes reachable from any branch
// ---------------------------------------------------------------------------

/// Blocks reachable from a non-HEAD branch must NOT be deleted by gc --force.
/// After creating a side branch with a unique commit, gc --force on the main
/// branch must not corrupt the side branch's data.
#[test]
fn gc_force_preserves_blocks_reachable_from_other_branches() {
    let dir = TempDir::new().unwrap();
    let p = dir.path();
    init(p);

    // Add a node on main.
    add_node(p, "main-node");

    // Create a side branch at HEAD.
    mnem(p, &["branch", "create", "side"]).assert().success();

    // Switch to side and add a node unique to that branch.
    mnem(p, &["switch", "side"]).assert().success();
    let side_uuid = add_node(p, "side-only-node");

    // Switch back to main.
    mnem(p, &["switch", "main"]).assert().success();

    // Inject a genuinely orphaned block that gc should collect.
    let orphan_cid = inject_orphaned_block(p, b"orphan-not-on-any-branch");

    // Before gc, capture the side-branch node's CID from the blockstore so we
    // can verify it is still there after gc (not just via `mnem get`).
    // We need to look up the commit that contains "side-only-node" in the store.
    // We do this by grabbing the full CID list now and comparing after gc.
    let cids_before_gc: std::collections::HashSet<mnem_core::id::Cid> = {
        let bs = open_blockstore(p);
        bs.all_cids()
            .expect("all_cids")
            .expect("redb supports enumeration")
            .into_iter()
            .collect()
        // bs drops here
    };

    // Force gc - must collect EXACTLY the orphan (1 block) and preserve the
    // side branch's blocks. The assertion must be specific: "removed 1 block",
    // not vacuously accepting either "removed something" or "removed nothing".
    let stdout = gc_force(p);
    assert!(
        stdout.contains("gc: removed 1 block(s)"),
        "gc --force must report 'gc: removed 1 block(s)' (exactly the injected orphan), got: {stdout}"
    );

    // Verify at blockstore level: the orphan is gone, side-branch blocks survive.
    {
        let bs = open_blockstore(p);

        // Orphan must be deleted.
        assert!(
            !bs.has(&orphan_cid).expect("has(orphan) after gc"),
            "orphan block must be deleted from store by gc --force"
        );

        // Every CID that existed before (except the orphan) must still exist.
        for cid in &cids_before_gc {
            if cid == &orphan_cid {
                continue; // this one should be gone
            }
            assert!(
                bs.has(cid).expect("has(live cid) after gc"),
                "live block {cid} must still exist in store after gc --force (side-branch block deleted?)"
            );
        }
    }

    // Switch back to side and verify the node is still there via CLI too.
    mnem(p, &["switch", "side"]).assert().success();
    let out = mnem(p, &["get", &side_uuid]).assert().success();
    let get_stdout = String::from_utf8_lossy(&out.get_output().stdout).to_string();
    assert!(
        get_stdout.contains("side-only-node"),
        "side branch node must survive gc --force, got: {get_stdout}"
    );

    // The side branch log must still be intact.
    let out2 = mnem(p, &["log"]).assert().success();
    let log_stdout = String::from_utf8_lossy(&out2.get_output().stdout).to_string();
    assert!(
        log_stdout.contains("op "),
        "side branch log must still work after gc --force on main, got: {log_stdout}"
    );
}

// ---------------------------------------------------------------------------
// Test 9: gc --force does NOT delete blocks from a hard-deleted node
// ---------------------------------------------------------------------------

/// After `mnem delete <uuid>`, the deleted node's blocks remain reachable
/// through the parent-commit chain (every commit points to its predecessor).
/// Running `mnem gc --force` on a repo where a node was deleted must therefore:
/// - remove ONLY the injected genuinely-orphaned block (exactly 1)
/// - preserve every block that was committed before the delete (all still
///   reachable via the commit-history chain, even though the node is no longer
///   in the current view)
///
/// This verifies that the gc reachability walk correctly traverses the full
/// commit history and does not mistake "deleted-from-view" for "unreachable".
#[test]
fn gc_force_does_not_delete_blocks_from_deleted_node() {
    let dir = TempDir::new().unwrap();
    let p = dir.path();
    init(p);

    // Step 1: Add a node and record its UUID.
    let node_uuid = add_node(p, "node-to-be-deleted");

    // Step 2: Hard-delete the node via CLI.
    // This creates a new commit (remove_node op) but does NOT orphan any blocks -
    // the original node's blocks are still reachable via the parent commit.
    mnem(p, &["delete", &node_uuid]).assert().success();

    // Step 3: Capture block CIDs after the delete commit.
    // This includes both the original add_node blocks AND the delete-commit's own
    // blocks (new Commit, View, Operation, and updated Prolly-tree nodes). All of
    // these are reachable from HEAD (the delete is the newest commit), so gc must
    // not remove any of them.
    let cids_after_delete: std::collections::HashSet<mnem_core::id::Cid> = {
        let bs = open_blockstore(p);
        bs.all_cids()
            .expect("all_cids")
            .expect("redb supports enumeration")
            .into_iter()
            .collect()
        // bs drops here
    };

    // Step 4: Inject exactly 1 genuinely orphaned block (not reachable from any ref).
    let orphan_cid = inject_orphaned_block(p, b"orphan-after-delete");

    // Step 5: Run gc --force - must report exactly 1 block removed (the injected orphan).
    let stdout = gc_force(p);
    assert!(
        stdout.contains("gc: removed 1 block(s)"),
        "gc --force must report 'gc: removed 1 block(s)' (only the injected orphan), got: {stdout}"
    );

    // Open the blockstore once for all remaining assertions.
    {
        let bs = open_blockstore(p);

        // Step 6: Verify the orphan is actually gone.
        assert!(
            !bs.has(&orphan_cid).expect("has(orphan) after gc"),
            "injected orphan block must be deleted from the store by gc --force"
        );

        // Step 7: Verify ALL blocks that existed after the delete commit are still present.
        // cids_after_delete includes the original add_node blocks AND the delete-op's
        // own blocks - all are reachable from HEAD and must survive gc.
        // The orphan_cid is NOT in cids_after_delete (it was injected after step 3),
        // so no exclusion guard is needed here.
        for cid in &cids_after_delete {
            assert!(
                bs.has(cid).expect("has(historical cid) after gc"),
                "block {cid} (reachable from HEAD via commit chain) must still be present after gc --force"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Test 10: Live (committed) block is NOT deleted by gc --force
// ---------------------------------------------------------------------------

/// Blocks that ARE reachable from HEAD must survive `mnem gc --force` intact.
/// We verify this at the blockstore level by comparing total block counts
/// before and after gc.
///
/// Why `total_before - 1 == total_after` is safe here:
/// A single `add_node` call results in multiple committed blocks in the store
/// (at minimum: a commit block, a view block, and one or more prolly-tree
/// interior/leaf nodes). We inject exactly 1 orphan block. GC must remove
/// exactly that 1 orphan and leave all reachable blocks intact, so the delta
/// between before and after is exactly 1.
///
/// The `assert!(total_before >= 5, ...)` guard below ensures the "multiple
/// blocks from one commit" invariant holds - if it ever fails, the block
/// layout changed and this test needs updating.
///
/// Important: the blockstore Arc must be dropped before the CLI command runs,
/// since redb does not allow two writers to the same database file.
#[test]
fn gc_force_does_not_delete_live_blocks() {
    let dir = TempDir::new().unwrap();
    let p = dir.path();
    init(p);

    // Commit a node so there is something live in the store.
    add_node(p, "live-committed-node");

    // Inject an orphan block alongside it.
    inject_orphaned_block(p, b"orphan-alongside-live");

    // Record the count of blocks before gc. Drop the blockstore before the CLI call.
    let total_before = {
        let bs = open_blockstore(p);
        let cids = bs
            .all_cids()
            .expect("all_cids")
            .expect("redb supports enumeration");
        cids.len()
        // `bs` drops here, releasing the redb lock.
    };

    // Guard: a single `add_node` must produce multiple blocks (commit, view,
    // prolly-tree nodes, …) plus the 1 injected orphan. If this fails,
    // the block layout changed and the delta-1 assertion below is invalid.
    assert!(
        total_before >= 5,
        "expected >= 5 blocks after one add_node + one orphan inject (got {total_before}); \
         if the block layout changed, update this test"
    );

    // Force-gc via CLI. The orphan (1 block) should be removed; no live block removed.
    gc_force(p);

    // Count blocks after gc. Open a fresh handle.
    let total_after = {
        let bs2 = open_blockstore(p);
        let cids = bs2
            .all_cids()
            .expect("all_cids")
            .expect("redb supports enumeration");
        cids.len()
    };

    // Exactly 1 block removed (the orphan).
    assert_eq!(
        total_before - 1,
        total_after,
        "gc --force must remove exactly 1 block (the injected orphan), removed {} instead",
        total_before.saturating_sub(total_after)
    );
}

// ---------------------------------------------------------------------------
// TODO: Test for `bs.all_cids()` returning None (degraded-mode code path)
// ---------------------------------------------------------------------------
//
// `mnem gc` has a code path (gc.rs step 2) where `bs.all_cids()` returns
// `Ok(None)`, meaning the blockstore backend does not support enumeration.
// In that case gc prints a degraded message and exits 0 without deleting
// anything.
//
// The redb backend always returns `Ok(Some(...))` (it supports enumeration),
// so this path cannot be exercised via integration tests against a real repo.
// There is no in-tree `MemoryBlockstore` or mock blockstore exposed by the
// public API of `mnem-backend-redb` or `mnem-core` that returns `None` from
// `all_cids()`.
//
// To cover this path properly, one of the following would be needed:
//   a) Add a `MockBlockstore` to `mnem-core` that returns `None` from
//      `all_cids()`, then write a unit test directly against `gc::run()`.
//   b) Add a `--backend=none-enum` test-only flag to the CLI.
//   c) Extract the gc loop into a testable function that takes a `&dyn
//      Blockstore` and test it directly.
//
// Until one of those is implemented, this path is marked as a known gap.
// #[ignore]
// fn gc_all_cids_none_degraded_mode() { todo!() }
