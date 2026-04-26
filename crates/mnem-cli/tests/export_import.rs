//! Integration tests for `mnem export` / `mnem import`.
//!
//! Uses `assert_cmd` to drive the built `mnem` binary against
//! temporary repos. Each test is end-to-end: init a repo, write some
//! content, export to a CAR, import into a second repo, confirm the
//! imported side sees the same blocks.

use std::path::Path;
use std::process::Command;

use assert_cmd::prelude::*;
use tempfile::TempDir;

/// Run `mnem <args>...` from inside `repo` as cwd, plus `-R <repo>`
/// so commands that honour the flag pick the right directory. Two
/// different mechanisms because `mnem init` takes an optional
/// positional path, while most other commands use `-R`.
fn mnem(repo: &Path, args: &[&str]) -> Command {
    let mut cmd = Command::cargo_bin("mnem").expect("built mnem binary");
    cmd.current_dir(repo);
    cmd.arg("-R").arg(repo);
    for a in args {
        cmd.arg(a);
    }
    cmd
}

#[test]
fn export_empty_repo_fails_cleanly() {
    // A freshly-initialised repo has no head commit, so `mnem export`
    // with the default `--from HEAD` must fail with an actionable
    // error rather than a panic or an empty-CAR write.
    let dir = TempDir::new().unwrap();
    // `init` takes an explicit positional path (it ignores `-R`).
    mnem(dir.path(), &["init", dir.path().to_str().unwrap()])
        .assert()
        .success();

    let car = dir.path().join("out.car");
    let out = mnem(dir.path(), &["export", car.to_str().unwrap()])
        .assert()
        .failure();
    let stderr = String::from_utf8_lossy(&out.get_output().stderr).to_string();
    assert!(
        stderr.contains("no commits") || stderr.contains("nothing to export"),
        "expected actionable error message, got: {stderr}"
    );
}

#[test]
fn export_then_import_round_trip() {
    // Build repo A, add a node to give it a head commit, export the
    // head, import into a fresh repo B, and assert the import stats
    // line matches the export stats line (same block count + bytes).
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
            "roundtrip payload",
            "--prop",
            "kind=doc",
        ],
    )
    .assert()
    .success();

    let car = src.path().join("snapshot.car");
    let export_out = mnem(
        src.path(),
        &["export", car.to_str().unwrap(), "--from", "HEAD"],
    )
    .assert()
    .success();
    let export_stdout = String::from_utf8_lossy(&export_out.get_output().stdout).to_string();
    assert!(
        export_stdout.starts_with("exported "),
        "export stdout: {export_stdout}"
    );
    assert!(car.exists(), "CAR file must be produced on disk");
    let car_size = std::fs::metadata(&car).unwrap().len();
    assert!(car_size > 0, "CAR must be non-empty");

    // Parse the block count out of the stdout line for cross-check.
    // Format: "exported N blocks, M bytes to <path>".
    let exported_n = export_stdout
        .split_whitespace()
        .nth(1)
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or_else(|| panic!("could not parse block count from: {export_stdout}"));
    assert!(
        exported_n >= 4,
        "a committed repo has commit+view+op+trees, got {exported_n}"
    );

    // Fresh destination.
    let dst = TempDir::new().unwrap();
    mnem(dst.path(), &["init", dst.path().to_str().unwrap()])
        .assert()
        .success();

    let import_out = mnem(dst.path(), &["import", car.to_str().unwrap()])
        .assert()
        .success();
    let import_stdout = String::from_utf8_lossy(&import_out.get_output().stdout).to_string();
    assert!(
        import_stdout.starts_with("imported "),
        "import stdout: {import_stdout}"
    );
    let imported_n = import_stdout
        .split_whitespace()
        .nth(1)
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or_else(|| panic!("could not parse block count from: {import_stdout}"));
    assert_eq!(
        exported_n, imported_n,
        "block counts must match across export / import"
    );
}
