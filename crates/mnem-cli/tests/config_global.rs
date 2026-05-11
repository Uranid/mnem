//! Integration tests for `mnem config --global` (BUG-54).
//!
//! Verifies that `--global` / `-g` read/write `~/.mnem/config.toml`
//! rather than the per-repo `.mnem/config.toml`. A synthetic `HOME`
//! (or `USERPROFILE` on Windows) is injected so the tests never touch
//! the developer's real global config.

use std::path::Path;
use std::process::Command;

use assert_cmd::prelude::*;
use tempfile::TempDir;

/// Build a `mnem` command rooted at `repo` with an overridden home
/// directory (so `~/.mnem/config.toml` points into `fake_home`).
fn mnem_with_home(repo: &Path, fake_home: &Path, args: &[&str]) -> Command {
    let mut cmd = Command::cargo_bin("mnem").expect("built mnem binary");
    cmd.current_dir(repo);
    cmd.arg("-R").arg(repo);
    // Override the home-directory env vars used by the `dirs` crate.
    #[cfg(unix)]
    cmd.env("HOME", fake_home);
    #[cfg(windows)]
    cmd.env("USERPROFILE", fake_home);
    // Disable global config *reading* for all env-var-driven lookups
    // except when we explicitly test it below.
    for a in args {
        cmd.arg(a);
    }
    cmd
}

/// `mnem config --global set user.name Alice` must write
/// `~/.mnem/config.toml` (not the per-repo config) and echo
/// `user.name = Alice`.
#[test]
fn global_set_writes_home_mnem_config() {
    let repo_dir = TempDir::new().unwrap();
    let fake_home = TempDir::new().unwrap();

    // Initialise a repo so the binary has a valid data-dir in case it
    // tries to locate one (it shouldn't for --global, but be safe).
    mnem_with_home(
        repo_dir.path(),
        fake_home.path(),
        &["init", repo_dir.path().to_str().unwrap()],
    )
    .assert()
    .success();

    // Write a key to the global config.
    let out = mnem_with_home(
        repo_dir.path(),
        fake_home.path(),
        &["config", "--global", "set", "user.name", "Alice"],
    )
    .assert()
    .success();
    let stdout = String::from_utf8_lossy(&out.get_output().stdout).to_string();
    assert!(
        stdout.contains("user.name = Alice"),
        "expected 'user.name = Alice' in output, got: {stdout}"
    );

    // The file must exist at `<fake_home>/.mnem/config.toml`.
    let cfg_path = fake_home.path().join(".mnem").join("config.toml");
    assert!(
        cfg_path.exists(),
        "expected ~/.mnem/config.toml to be created, but it is absent"
    );

    // The per-repo config must NOT contain the key.
    let repo_cfg_path = repo_dir.path().join(".mnem").join("config.toml");
    if repo_cfg_path.exists() {
        let content = std::fs::read_to_string(&repo_cfg_path).unwrap();
        assert!(
            !content.contains("Alice"),
            "per-repo config must not contain the global-only value 'Alice'"
        );
    }
}

/// `mnem config -g get user.name` must read the value written by the
/// previous `--global set`, not from the per-repo config.
#[test]
fn global_get_reads_home_mnem_config() {
    let repo_dir = TempDir::new().unwrap();
    let fake_home = TempDir::new().unwrap();

    mnem_with_home(
        repo_dir.path(),
        fake_home.path(),
        &["init", repo_dir.path().to_str().unwrap()],
    )
    .assert()
    .success();

    // Seed the global config directly.
    let home_mnem = fake_home.path().join(".mnem");
    std::fs::create_dir_all(&home_mnem).unwrap();
    std::fs::write(
        home_mnem.join("config.toml"),
        "[user]\nname = \"GlobalUser\"\n",
    )
    .unwrap();

    // `mnem config -g get user.name` must return the global value.
    let out = mnem_with_home(
        repo_dir.path(),
        fake_home.path(),
        &["config", "-g", "get", "user.name"],
    )
    .assert()
    .success();
    let stdout = String::from_utf8_lossy(&out.get_output().stdout).to_string();
    assert!(
        stdout.trim() == "GlobalUser",
        "expected 'GlobalUser' from global config, got: {stdout}"
    );
}

/// `mnem config --global list` must enumerate keys from the global
/// config and succeed even when no per-repo config exists.
#[test]
fn global_list_works_without_per_repo_config() {
    let repo_dir = TempDir::new().unwrap();
    let fake_home = TempDir::new().unwrap();

    mnem_with_home(
        repo_dir.path(),
        fake_home.path(),
        &["init", repo_dir.path().to_str().unwrap()],
    )
    .assert()
    .success();

    // Seed the global config.
    let home_mnem = fake_home.path().join(".mnem");
    std::fs::create_dir_all(&home_mnem).unwrap();
    std::fs::write(
        home_mnem.join("config.toml"),
        "[user]\nemail = \"g@example.com\"\n",
    )
    .unwrap();

    let out = mnem_with_home(
        repo_dir.path(),
        fake_home.path(),
        &["config", "--global", "list"],
    )
    .assert()
    .success();
    let stdout = String::from_utf8_lossy(&out.get_output().stdout).to_string();
    assert!(
        stdout.contains("user.email = g@example.com"),
        "expected 'user.email = g@example.com' in global list output, got: {stdout}"
    );
}
