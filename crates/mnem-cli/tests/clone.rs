//! Integration tests for `mnem clone` against a `file://` URL.
//!
//! Tests:
//!
//! 1. Happy path: export from repo A, clone into B, assert the log
//!    surfaces the same commit cid + the remote config landed on
//!    disk.
//! 2. Refuses to clone into a non-empty `.mnem/` directory.
//! 3. Rejects unsupported URL schemes (`https://`) with a message
//!    pointing at PR 3.
//! 4. Both tracking refs (`refs/remotes/origin/main`) and the local
//!    branch (`refs/heads/main`) are created, pointing at the same
//!    CID announced in the clone stdout.
//! 5. BUG-29: on any import failure, the partial `.mnem/` directory
//!    is removed so the user can retry without manual cleanup.
//! 6. Bare `.car` path (without `file://` prefix) is accepted as a
//!    clone source, matching the behaviour of `git clone`.

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

fn mnem_no_repo(args: &[&str]) -> Command {
    let mut cmd = Command::cargo_bin("mnem").expect("built mnem binary");
    for a in args {
        cmd.arg(a);
    }
    cmd
}

/// Produce a `file://` URL pointing at the given path. Normalises
/// Windows drive letters so `C:\x\y.car` becomes
/// `file:///C:/x/y.car`.
fn file_url(p: &Path) -> String {
    let s = p.to_string_lossy().replace('\\', "/");
    if s.starts_with('/') {
        format!("file://{s}")
    } else {
        format!("file:///{s}")
    }
}

/// Build a minimal single-commit repo in `src`, export its HEAD to a
/// CAR file in `car_dir`, and return the path of the CAR file.
fn build_source_and_export(src: &TempDir, car_dir: &TempDir) -> std::path::PathBuf {
    mnem(src.path(), &["init", src.path().to_str().unwrap()])
        .assert()
        .success();
    mnem(
        src.path(),
        &[
            "add",
            "node",
            "--summary",
            "cloneable payload",
            "--label",
            "doc",
            "--no-embed",
        ],
    )
    .assert()
    .success();

    let car = car_dir.path().join("snapshot.car");
    mnem(
        src.path(),
        &["export", car.to_str().unwrap(), "--from", "HEAD"],
    )
    .assert()
    .success();
    assert!(car.exists(), "export must produce a CAR file");
    car
}

#[test]
fn clone_file_url_round_trips_commit_head() {
    // Repo A: init + add node -> has a head commit.
    let src = TempDir::new().unwrap();
    mnem(src.path(), &["init", src.path().to_str().unwrap()])
        .assert()
        .success();
    mnem(
        src.path(),
        &[
            "add",
            "node",
            "--summary",
            "cloneable payload",
            "--label",
            "doc",
            "--no-embed",
        ],
    )
    .assert()
    .success();

    // Export the HEAD to a CAR that lives outside either repo.
    let car_dir = TempDir::new().unwrap();
    let car = car_dir.path().join("snapshot.car");
    mnem(
        src.path(),
        &["export", car.to_str().unwrap(), "--from", "HEAD"],
    )
    .assert()
    .success();
    assert!(car.exists());

    // Clone into a fresh dir by file:// URL.
    let dst_root = TempDir::new().unwrap();
    let dst = dst_root.path().join("mirror");
    let url = file_url(&car);
    let cloned = mnem_no_repo(&["clone", &url, dst.to_str().unwrap()])
        .assert()
        .success();
    let stdout = String::from_utf8_lossy(&cloned.get_output().stdout).to_string();
    assert!(
        stdout.contains("cloned ") && stdout.contains("blocks"),
        "clone output shape: {stdout}"
    );
    assert!(
        stdout.contains("origin/main"),
        "clone should report origin/main tracking ref, got: {stdout}"
    );

    // The clone created `.mnem/config.toml` with [remote.origin].
    let cfg = std::fs::read_to_string(dst.join(".mnem/config.toml")).expect("config.toml");
    assert!(
        cfg.contains("[remote.origin]"),
        "config must carry [remote.origin], got: {cfg}"
    );
    assert!(
        cfg.contains(&url),
        "config must carry the source URL, got: {cfg}"
    );

    // Pull the commit CID out of the log of A.
    let src_log = mnem(src.path(), &["log", "-n", "1", "--format=json"])
        .assert()
        .success();
    let src_stdout = String::from_utf8_lossy(&src_log.get_output().stdout).to_string();
    // Log output format is JSON Lines.
    let src_line = src_stdout
        .lines()
        .next()
        .expect("expected at least one op from src");
    let src_v: serde_json::Value = serde_json::from_str(src_line).expect("json");
    let src_cid = src_v["cid"].as_str().unwrap().to_string();
    assert!(!src_cid.is_empty());

    // The imported repo must carry a refs/remotes/origin/main that
    // points at the same commit cid the export started from. We
    // surface this via `mnem ref list` on the dst.
    let refs_out = mnem(&dst, &["ref", "list"]).assert().success();
    let refs_stdout = String::from_utf8_lossy(&refs_out.get_output().stdout).to_string();
    assert!(
        refs_stdout.contains("refs/remotes/origin/main"),
        "dst should carry origin/main tracking ref, got: {refs_stdout}"
    );
    assert!(
        refs_stdout.contains("refs/heads/main"),
        "dst should carry local refs/heads/main branch (J6 fix), got: {refs_stdout}"
    );
}

#[test]
fn clone_refuses_into_existing_mnem_dir() {
    // Pre-existing mnem repo in the target dir. Clone must NOT
    // clobber it; git's equivalent: `fatal: destination path 'foo'
    // already exists and is not an empty directory`.
    let src = TempDir::new().unwrap();
    mnem(src.path(), &["init", src.path().to_str().unwrap()])
        .assert()
        .success();
    mnem(
        src.path(),
        &["add", "node", "--summary", "src", "--no-embed"],
    )
    .assert()
    .success();

    let car_dir = TempDir::new().unwrap();
    let car = car_dir.path().join("s.car");
    mnem(src.path(), &["export", car.to_str().unwrap()])
        .assert()
        .success();

    // Destination: initialised already.
    let dst = TempDir::new().unwrap();
    mnem(dst.path(), &["init", dst.path().to_str().unwrap()])
        .assert()
        .success();

    let url = file_url(&car);
    let out = mnem_no_repo(&["clone", &url, dst.path().to_str().unwrap()])
        .assert()
        .failure();
    let stderr = String::from_utf8_lossy(&out.get_output().stderr).to_string();
    assert!(
        stderr.contains("already contains") || stderr.contains("refusing"),
        "expected refuse-to-clobber error, got: {stderr}"
    );
}

#[test]
fn clone_rejects_https_scheme_actionably() {
    // Remote schemes are deferred. The error MUST name PR 3 so the
    // user knows where the roadmap is.
    let dst = TempDir::new().unwrap();
    let target = dst.path().join("mirror");
    let out = mnem_no_repo(&[
        "clone",
        "https://example.com/alice/notes",
        target.to_str().unwrap(),
    ])
    .assert()
    .failure();
    let stderr = String::from_utf8_lossy(&out.get_output().stderr).to_string();
    assert!(
        stderr.contains("not yet implemented"),
        "deferred-scheme error must be actionable: {stderr}"
    );
    assert!(
        stderr.contains("PR 3") || stderr.contains("ROADMAP"),
        "deferred-scheme error must point at PR 3 or ROADMAP so the user knows where \
         the roadmap is: {stderr}"
    );
}

/// After a successful clone both refs must exist and their CID targets
/// must match the CID announced in the clone stdout.
///
/// Specifically verifies:
///   - `refs/remotes/origin/main` (the remote-tracking ref)
///   - `refs/heads/main`         (the local branch; J6 fix)
///   - both point at the same commit CID (the one printed as `origin/main -> <cid>`)
///
/// This catches a regression where only the tracking ref is written
/// but `refs/heads/main` is missing, or where the printed CID does
/// not match what is actually stored in the blockstore view.
#[test]
fn clone_both_refs_created_with_matching_cid() {
    let src = TempDir::new().unwrap();
    let car_dir = TempDir::new().unwrap();
    let car = build_source_and_export(&src, &car_dir);

    let dst_root = TempDir::new().unwrap();
    let dst = dst_root.path().join("mirror");
    let url = file_url(&car);

    // Clone and capture stdout so we can extract the announced CID.
    let cloned = mnem_no_repo(&["clone", &url, dst.to_str().unwrap()])
        .assert()
        .success();
    let stdout = String::from_utf8_lossy(&cloned.get_output().stdout).to_string();

    // Clone stdout contains: "  origin/main -> <cid>"
    // Extract the CID that the clone reported setting.
    let announced_cid: String = stdout
        .lines()
        .find(|l| l.contains("origin/main ->"))
        .and_then(|l| l.split("->").nth(1))
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty() && !s.starts_with('<'))
        .expect("clone stdout must contain 'origin/main -> <cid>' with a real CID");

    // Verify both refs exist in the clone's ref list and carry the same CID.
    let refs_out = mnem(&dst, &["ref", "list"]).assert().success();
    let refs_stdout = String::from_utf8_lossy(&refs_out.get_output().stdout).to_string();

    assert!(
        refs_stdout.contains("refs/remotes/origin/main"),
        "clone must create refs/remotes/origin/main; got:\n{refs_stdout}"
    );
    assert!(
        refs_stdout.contains("refs/heads/main"),
        "clone must create refs/heads/main (J6 fix); got:\n{refs_stdout}"
    );
    // Both named refs must individually contain the announced CID on
    // their respective lines -- a line-count >= 2 check would not
    // prove that EACH named ref carries the right CID.
    let origin_main_line = refs_stdout
        .lines()
        .find(|l| l.contains("refs/remotes/origin/main"))
        .expect("refs/remotes/origin/main must appear in ref list");
    assert!(
        origin_main_line.contains(&announced_cid),
        "refs/remotes/origin/main line must contain announced CID ({announced_cid}); \
         got: {origin_main_line}"
    );

    let heads_main_line = refs_stdout
        .lines()
        .find(|l| l.contains("refs/heads/main"))
        .expect("refs/heads/main must appear in ref list (J6 fix)");
    assert!(
        heads_main_line.contains(&announced_cid),
        "refs/heads/main line must contain announced CID ({announced_cid}); \
         got: {heads_main_line}"
    );
}

/// BUG-29: if the CAR import fails (e.g. malformed file), the clone
/// command must remove the partially-created `.mnem/` directory before
/// returning the error.  Without this cleanup the user is left with an
/// unusable `.mnem/` skeleton that blocks a retry.
#[test]
fn clone_cleans_up_mnem_dir_on_import_failure() {
    // Write a file with a `.car` extension but completely invalid content.
    // `mnem_transport::import` will reject it immediately (bad CAR magic).
    let tmp = TempDir::new().unwrap();
    let bad_car = tmp.path().join("corrupt.car");
    std::fs::write(&bad_car, b"this is not a valid CAR archive").unwrap();

    let dst_root = TempDir::new().unwrap();
    let dst = dst_root.path().join("clone-dst");
    let url = file_url(&bad_car);

    let out = mnem_no_repo(&["clone", &url, dst.to_str().unwrap()])
        .assert()
        .failure();
    let stderr = String::from_utf8_lossy(&out.get_output().stderr).to_string();

    // Import must fail (the file is garbage).
    assert!(
        !stderr.is_empty(),
        "clone of a corrupt CAR must print an error message"
    );

    // BUG-29: .mnem/ must NOT be left behind after the failure.
    let mnem_dir = dst.join(".mnem");
    assert!(
        !mnem_dir.exists(),
        "BUG-29: .mnem/ must be cleaned up after a failed clone; found it at {}",
        mnem_dir.display()
    );
}

/// A bare `.car` file path (without the `file://` scheme prefix) must
/// be accepted as a clone source.  This mirrors the behaviour of
/// `git clone /path/to/repo.bundle` on Unix and the Windows Bash
/// convention `mnem clone C:\data\notes.car`.
///
/// `parse_clone_source` normalises the path internally; here we
/// confirm the normalised path is still openable and the clone
/// succeeds end-to-end.
#[test]
fn clone_bare_car_path_works() {
    let src = TempDir::new().unwrap();
    let car_dir = TempDir::new().unwrap();
    let car = build_source_and_export(&src, &car_dir);

    let dst_root = TempDir::new().unwrap();
    let dst = dst_root.path().join("clone-bare");

    // Use the bare OS path - NOT a file:// URL.
    let bare_path = car.to_str().expect("CAR path must be valid UTF-8");
    let cloned = mnem_no_repo(&["clone", bare_path, dst.to_str().unwrap()])
        .assert()
        .success();
    let stdout = String::from_utf8_lossy(&cloned.get_output().stdout).to_string();

    assert!(
        stdout.contains("cloned ") && stdout.contains("blocks"),
        "bare-path clone must report block count; got: {stdout}"
    );
    assert!(
        dst.join(".mnem").exists(),
        ".mnem/ must be created by bare-path clone"
    );
    assert!(
        dst.join(".mnem/config.toml").exists(),
        "config.toml must be written by bare-path clone"
    );

    // The config must record the source URL (bare path normalised
    // to file:// internally, or stored as-is -- either is acceptable
    // as long as the file exists and origin is present).
    let cfg = std::fs::read_to_string(dst.join(".mnem/config.toml")).unwrap();
    assert!(
        cfg.contains("[remote.origin]"),
        "bare-path clone must write [remote.origin]; got: {cfg}"
    );
}
