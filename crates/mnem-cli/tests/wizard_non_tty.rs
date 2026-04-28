//! Regression tests for the first-run wizard's non-tty fallback.
//!
//! Bare `mnem` under CI (no tty, stdin piped, stdout captured) must
//! NOT block on a prompt that will never receive input. The wizard
//! detects this via `std::io::IsTerminal` and prints a short "run
//! `mnem --help`" hint, then exits 0.
//!
//! Silent rot here would turn every CI job that accidentally invokes
//! bare `mnem` into a hang until the timeout. This test pins the
//! non-interactive shape so a future refactor can't regress it
//! without a visible failure.

use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use assert_cmd::prelude::*;
use tempfile::TempDir;

/// Maximum wall-clock the child is allowed to take. The wizard's
/// non-tty branch is pure println! + return, so it should clock in
/// under 500ms on a fresh build. We allow 10s to absorb Windows
/// first-launch Defender scan, slow CI shared runners, etc.; a
/// regression would hang much longer (until a kill signal).
const WALL_CLOCK_CEILING: Duration = Duration::from_secs(10);

/// Run bare `mnem` (no subcommand) in a temp directory that does NOT
/// contain a `.mnem` repo. stdin / stdout / stderr are piped so the
/// `IsTerminal` check sees "not a tty" -> wizard returns early.
fn spawn_bare_mnem(cwd: &std::path::Path) -> std::process::Child {
    // Under `cargo test`, `assert_cmd` resolves the freshly-built
    // `mnem` binary; we use it for path resolution but run via
    // `Command` so we can own stdin / stdout pipes directly.
    let path = Command::cargo_bin("mnem").unwrap().get_program().to_owned();
    Command::new(path)
        .current_dir(cwd)
        // MNEM_NO_WIZARD=1 also disables the interactive path; we
        // deliberately DO NOT set it here because we want the IsTerminal
        // check itself to fire. The test fails if the wizard falls
        // through the env-var bypass by accident.
        .env_remove("MNEM_NO_WIZARD")
        // Piped stdin is what convinces `IsTerminal` we are not a tty.
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn mnem")
}

#[test]
fn bare_mnem_under_non_tty_exits_fast_not_hangs() {
    let td = TempDir::new().unwrap();
    let mut child = spawn_bare_mnem(td.path());

    // Close stdin immediately so any accidental `read` in the wizard's
    // interactive branch returns EOF instead of blocking forever.
    drop(child.stdin.take());

    // Poll `try_wait` with a short sleep so a regression surfaces as
    // a timeout we can report cleanly, not an indefinite hang.
    let started = Instant::now();
    loop {
        match child.try_wait() {
            Ok(Some(status)) => {
                assert!(
                    status.success(),
                    "bare `mnem` in non-tty must exit 0; got {status:?}"
                );
                // Drain stderr to confirm the helpful hint line landed.
                let out = child.wait_with_output().expect("wait_with_output");
                let stderr = String::from_utf8_lossy(&out.stderr);
                assert!(
                    stderr.contains("mnem: no subcommand given") || stderr.contains("mnem --help"),
                    "expected non-interactive hint in stderr, got:\n{stderr}"
                );
                return;
            }
            Ok(None) => {
                if started.elapsed() > WALL_CLOCK_CEILING {
                    let _ = child.kill();
                    panic!(
                        "bare `mnem` under non-tty hung for {:?} (ceiling {WALL_CLOCK_CEILING:?}); wizard fast-path regressed?",
                        started.elapsed()
                    );
                }
                std::thread::sleep(Duration::from_millis(50));
            }
            Err(e) => panic!("try_wait failed: {e}"),
        }
    }
}

#[test]
fn bare_mnem_with_mnem_no_wizard_also_exits_fast() {
    // Second shape: env-var bypass is the documented escape hatch
    // for operators attached to a tty who still don't want the
    // wizard. We assert it short-circuits the same way.
    let td = TempDir::new().unwrap();
    let path = Command::cargo_bin("mnem").unwrap().get_program().to_owned();
    let mut child = Command::new(path)
        .current_dir(td.path())
        .env("MNEM_NO_WIZARD", "1")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn mnem");
    drop(child.stdin.take());

    let started = Instant::now();
    loop {
        match child.try_wait() {
            Ok(Some(status)) => {
                assert!(
                    status.success(),
                    "env-var bypass must exit 0; got {status:?}"
                );
                return;
            }
            Ok(None) => {
                if started.elapsed() > WALL_CLOCK_CEILING {
                    let _ = child.kill();
                    panic!(
                        "bare `mnem` with MNEM_NO_WIZARD=1 hung for {:?}",
                        started.elapsed()
                    );
                }
                std::thread::sleep(Duration::from_millis(50));
            }
            Err(e) => panic!("try_wait failed: {e}"),
        }
    }
}
