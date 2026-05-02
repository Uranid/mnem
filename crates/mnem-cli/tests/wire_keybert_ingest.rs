//! C3 FIX-3 live-wire test: `mnem ingest --extractor keybert <file>`
//! now reaches the KeyBertAdapter construction path instead of the
//! hard-coded `bail!` stub that landed in E3 T2.
//!
//! With bundled-embedder now default (S3), keybert actually succeeds
//! because the bundled ONNX model is available. This test verifies
//! that the keybert wiring works end-to-end.

use std::path::Path;
use std::process::Command;

use assert_cmd::prelude::*;
use tempfile::TempDir;

fn mnem(repo: &Path, args: &[&str]) -> Command {
    let mut cmd = Command::cargo_bin("mnem").expect("mnem binary built");
    cmd.current_dir(repo).arg("-R").arg(repo);
    // audit-2026-04-25 C4-4: redirect HOME (Unix) and USERPROFILE
    // (Windows) so `config::resolve_embedder` sees no user-global
    // `~/.mnem/config.toml` to inherit from. Without this, a
    // developer running tests with a real `~/.mnem/config.toml`
    // would silently bypass the "no embed provider" assertion.
    cmd.env("HOME", repo);
    cmd.env("USERPROFILE", repo);
    // Belt-and-braces: explicitly disable global-config inheritance
    // even if `dirs::home_dir()` finds a Windows known folder that
    // bypasses the env-var redirect above. See `config::load_global`.
    cmd.env("MNEM_DISABLE_GLOBAL_CONFIG", "1");
    // Also clear MNEM_* env vars that env-precedence resolution
    // would otherwise pick up.
    cmd.env_remove("MNEM_EMBED_PROVIDER");
    cmd.env_remove("MNEM_EMBED_MODEL");
    cmd.env_remove("MNEM_EMBED_API_KEY_ENV");
    cmd.env_remove("MNEM_EMBED_BASE_URL");
    for a in args {
        cmd.arg(a);
    }
    cmd
}

#[test]
fn ingest_keybert_works_with_bundled_embedder() {
    let td = TempDir::new().expect("tmp");
    let repo = td.path();

    // `mnem init` to seed a fresh repo so subsequent commands have a
    // data dir to work against.
    mnem(repo, &["init"])
        .ok()
        .expect("mnem init should succeed on an empty dir");

    // Write a tiny file to ingest. Keybert should work now that
    // bundled-embedder is default (S3).
    let file = repo.join("note.md");
    std::fs::write(&file, "# hello world\nthis is a note\n").unwrap();

    let out = mnem(
        repo,
        &["ingest", "--extractor", "keybert", file.to_str().unwrap()],
    )
    .output()
    .expect("spawn");

    // With bundled-embedder default, keybert should succeed
    assert!(
        out.status.success(),
        "keybert should succeed with bundled embedder; stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
}
