//! C3 FIX-3 live-wire test: `mnem ingest --extractor keybert <file>`
//! now reaches the KeyBertAdapter construction path instead of the
//! hard-coded `bail!` stub that landed in E3 T2.
//!
//! We can't spin up a real embedder in this test (would need Ollama or
//! an ONNX model), so we prove the wire reached the
//! `cfg.embed`-resolution step by asserting the CLI exits with a
//! precise "no [embed] provider" error. The pre-FIX-3 stub returned
//! a different string ("parsed but not yet wired"); the post-FIX-3
//! binary returns the provider-resolution error from the new code
//! path.

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
fn ingest_keybert_reaches_embedder_resolution() {
    let td = TempDir::new().expect("tmp");
    let repo = td.path();

    // `mnem init` to seed a fresh repo so subsequent commands have a
    // data dir to work against.
    mnem(repo, &["init"])
        .ok()
        .expect("mnem init should succeed on an empty dir");

    // C7-3a: `mnem init` now seeds `.mnem/config.toml` with a default
    // Ollama [embed] provider. Remove it so this test keeps its
    // "no embed provider configured" invariant (we're testing the
    // keybert wiring, not Ollama connectivity).
    let seeded_cfg = repo.join(".mnem").join("config.toml");
    if seeded_cfg.exists() {
        std::fs::remove_file(&seeded_cfg).expect("remove seeded config");
    }

    // Write a tiny file to ingest. Keybert would embed it IF an
    // [embed] provider were configured; we explicitly have none, so
    // the wire path must surface a clean error.
    let file = repo.join("note.md");
    std::fs::write(&file, "# hello world\nthis is a note\n").unwrap();

    let out = mnem(
        repo,
        &["ingest", "--extractor", "keybert", file.to_str().unwrap()],
    )
    .output()
    .expect("spawn");

    assert!(!out.status.success(), "must fail without embed provider");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("keybert") && stderr.contains("embed"),
        "error must reference keybert and the missing [embed] provider; got: {stderr}"
    );
    // Negative assertion: the pre-FIX-3 stub is gone.
    assert!(
        !stderr.contains("parsed but not yet wired"),
        "FIX-3 must replace the E3-T2 stub; still saw: {stderr}"
    );
}
