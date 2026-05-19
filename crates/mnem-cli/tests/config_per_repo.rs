//! Integration tests for `mnem config` per-repo behavior.
//!
//! Drives the real `mnem` binary against temp-dir repos and asserts that
//! per-repo `.mnem/config.toml` is read/written correctly. All commands
//! include `MNEM_DISABLE_GLOBAL_CONFIG=1` so a developer's real
//! `~/.mnem/config.toml` never bleeds into these tests.
//!
//! Tests:
//!  1. config_set_get_round_trips_user_name
//!  2. config_list_shows_set_keys
//! 2b. config_list_on_empty_repo_prints_no_keys_set
//!  3. config_unset_removes_key_and_get_fails
//!  4. config_retrieve_limit_persists_to_file
//!  5. config_embed_provider_persists_across_get_and_file
//!  6. config_invalid_key_is_rejected
//!  7. config_get_missing_key_exits_nonzero
//!  8. config_per_repo_write_does_not_touch_global
//!  9. config_legacy_positional_set_and_get
//! 10. config_api_key_guardrail_rejects_raw_secret (embed.api_key key-name branch,
//!     both sk- and non-sk- values)
//! 11. config_api_key_guardrail_rejects_sk_prefix_value (value-prefix branch, embed.*)
//! 12. config_api_key_guardrail_rejects_rerank_api_key (rerank.api_key key-name branch,
//!     both sk- and non-sk- values)
//! 13. config_set_overwrites_existing_value
//! 14. config_unset_never_set_key_exits_zero
//! 15. config_api_key_guardrail_rejects_sk_prefix_in_rerank_namespace (value-prefix, rerank.*)
//! 16. config_rerank_namespace_round_trips (positive rerank.* round-trip with TOML verification)
//! 17. config_guardrail_rejection_does_not_modify_toml (file unchanged after rejected set)
//! 18. config_list_after_unset_key_absent (list no longer shows key after unset)
//! 19. config_set_in_uninitialised_repo_fails (no .mnem/ directory)

use std::path::Path;
use std::process::Command;

use assert_cmd::prelude::*;
use tempfile::TempDir;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Build a `mnem` command rooted at `repo` with `MNEM_DISABLE_GLOBAL_CONFIG=1`
/// so the developer's real `~/.mnem/config.toml` is never consulted.
fn mnem(repo: &Path, args: &[&str]) -> Command {
    let mut cmd = Command::cargo_bin("mnem").expect("built mnem binary");
    cmd.current_dir(repo);
    cmd.arg("-R").arg(repo);
    cmd.env("MNEM_DISABLE_GLOBAL_CONFIG", "1");
    for a in args {
        cmd.arg(a);
    }
    cmd
}

/// Build a `mnem` command rooted at `repo` with an overridden home directory
/// (so `~/.mnem/config.toml` points into `fake_home`) AND
/// `MNEM_DISABLE_GLOBAL_CONFIG=1` so even that fake home's global config
/// doesn't interfere unless we explicitly need it.
fn mnem_with_home(repo: &Path, fake_home: &Path, args: &[&str]) -> Command {
    let mut cmd = Command::cargo_bin("mnem").expect("built mnem binary");
    cmd.current_dir(repo);
    cmd.arg("-R").arg(repo);
    cmd.env("MNEM_DISABLE_GLOBAL_CONFIG", "1");
    #[cfg(unix)]
    cmd.env("HOME", fake_home);
    #[cfg(windows)]
    cmd.env("USERPROFILE", fake_home);
    for a in args {
        cmd.arg(a);
    }
    cmd
}

/// Initialise a repo in `dir` and assert success.
fn init(dir: &Path) {
    mnem(dir, &["init", dir.to_str().unwrap()])
        .assert()
        .success();
}

// ---------------------------------------------------------------------------
// Test 1
// ---------------------------------------------------------------------------

/// `mnem config set user.name Alice` must:
///   - echo `user.name = Alice` to stdout
///   - create `.mnem/config.toml` containing `[user]` section and
///     `name = "Alice"` as a TOML key=value pair
///   - `mnem config get user.name` must return trimmed `"Alice"`
#[test]
fn config_set_get_round_trips_user_name() {
    let repo = TempDir::new().unwrap();
    init(repo.path());

    // Set the key.
    let out = mnem(repo.path(), &["config", "set", "user.name", "Alice"])
        .assert()
        .success();
    let stdout = String::from_utf8_lossy(&out.get_output().stdout).to_string();
    assert!(
        stdout.contains("user.name = Alice"),
        "expected 'user.name = Alice' in set stdout, got: {stdout}"
    );

    // The per-repo config file must exist and contain the value.
    let cfg_path = repo.path().join(".mnem").join("config.toml");
    assert!(
        cfg_path.exists(),
        "expected .mnem/config.toml to be created after config set"
    );
    let contents = std::fs::read_to_string(&cfg_path).unwrap();
    assert!(
        contents.contains("[user]"),
        "expected '[user]' section header in .mnem/config.toml, got: {contents}"
    );
    assert!(
        contents.contains("name = \"Alice\""),
        "expected 'name = \"Alice\"' key=value pair in .mnem/config.toml, got: {contents}"
    );

    // Get must return the exact value.
    let out = mnem(repo.path(), &["config", "get", "user.name"])
        .assert()
        .success();
    let stdout = String::from_utf8_lossy(&out.get_output().stdout).to_string();
    assert_eq!(
        stdout.trim(),
        "Alice",
        "expected trimmed get stdout == 'Alice', got: {stdout}"
    );
}

// ---------------------------------------------------------------------------
// Test 2
// ---------------------------------------------------------------------------

/// After setting `user.name` and `user.email`, `mnem config list` must print
/// both `"user.name = Bob"` and `"user.email = bob@example.com"`.
#[test]
fn config_list_shows_set_keys() {
    let repo = TempDir::new().unwrap();
    init(repo.path());

    mnem(repo.path(), &["config", "set", "user.name", "Bob"])
        .assert()
        .success();
    mnem(
        repo.path(),
        &["config", "set", "user.email", "bob@example.com"],
    )
    .assert()
    .success();

    let out = mnem(repo.path(), &["config", "list"])
        .assert()
        .success();
    let stdout = String::from_utf8_lossy(&out.get_output().stdout).to_string();
    assert!(
        stdout.contains("user.name = Bob"),
        "expected 'user.name = Bob' in list output, got: {stdout}"
    );
    assert!(
        stdout.contains("user.email = bob@example.com"),
        "expected 'user.email = bob@example.com' in list output, got: {stdout}"
    );
}

// ---------------------------------------------------------------------------
// Test 2b
// ---------------------------------------------------------------------------

/// `mnem config list` on a repo with no keys set must print `"(no keys set)"`.
///
/// We ensure the per-repo config file is absent (removing it if it exists)
/// so the command exercises the "no config file" code path regardless of
/// whether `mnem init` seeds a default file.
#[test]
fn config_list_on_empty_repo_prints_no_keys_set() {
    let repo = TempDir::new().unwrap();
    init(repo.path());

    // Remove the per-repo config file if it exists (e.g. seeded by init).
    let cfg_path = repo.path().join(".mnem").join("config.toml");
    let _ = std::fs::remove_file(&cfg_path); // ignore "not found"

    let out = mnem(repo.path(), &["config", "list"])
        .assert()
        .success();
    let stdout = String::from_utf8_lossy(&out.get_output().stdout).to_string();
    assert!(
        stdout.contains("(no keys set)"),
        "list on repo with no config file must print '(no keys set)', got: {stdout}"
    );
}

// ---------------------------------------------------------------------------
// Test 3
// ---------------------------------------------------------------------------

/// Set `user.name`, then unset it:
///   - unset stdout must contain `"unset user.name"`
///   - subsequent get must exit 1 with stderr containing `"no value set"`
///   - `.mnem/config.toml` must not contain the unset value (physical removal)
#[test]
fn config_unset_removes_key_and_get_fails() {
    let repo = TempDir::new().unwrap();
    init(repo.path());

    mnem(repo.path(), &["config", "set", "user.name", "Charlie"])
        .assert()
        .success();

    // Unset the key.
    let out = mnem(repo.path(), &["config", "unset", "user.name"])
        .assert()
        .success();
    let stdout = String::from_utf8_lossy(&out.get_output().stdout).to_string();
    assert!(
        stdout.contains("unset user.name"),
        "expected 'unset user.name' in unset stdout, got: {stdout}"
    );

    // Get must now fail.
    let out = mnem(repo.path(), &["config", "get", "user.name"])
        .assert()
        .failure();
    let stderr = String::from_utf8_lossy(&out.get_output().stderr).to_string();
    assert!(
        stderr.contains("no value set"),
        "expected 'no value set' in get stderr after unset, got: {stderr}"
    );

    // The value must be physically absent from the TOML file.
    // `mnem init` always seeds config.toml, so the file is guaranteed to exist
    // even after an unset. Assert existence unconditionally so a missing file
    // is a test failure rather than a silent skip.
    let cfg_path = repo.path().join(".mnem").join("config.toml");
    assert!(
        cfg_path.exists(),
        "config.toml must still exist after unset (init seeds it)"
    );
    let contents = std::fs::read_to_string(&cfg_path).unwrap();
    assert!(
        !contents.contains("Charlie"),
        "unset must physically remove the value from .mnem/config.toml; \
         found 'Charlie' still present in: {contents}"
    );
}

// ---------------------------------------------------------------------------
// Test 4
// ---------------------------------------------------------------------------

/// `mnem config set retrieve.limit 20` must:
///   - be retrievable via `mnem config get retrieve.limit` → `"20"`
///   - appear in `mnem config list` as `"retrieve.limit = 20"`
///   - cause `.mnem/config.toml` to contain a `[retrieve]` table and `limit = 20`
#[test]
fn config_retrieve_limit_persists_to_file() {
    let repo = TempDir::new().unwrap();
    init(repo.path());

    mnem(repo.path(), &["config", "set", "retrieve.limit", "20"])
        .assert()
        .success();

    // Get must return "20".
    let out = mnem(repo.path(), &["config", "get", "retrieve.limit"])
        .assert()
        .success();
    let stdout = String::from_utf8_lossy(&out.get_output().stdout).to_string();
    assert_eq!(
        stdout.trim(),
        "20",
        "expected trimmed get stdout == '20', got: {stdout}"
    );

    // List must contain the key.
    let out = mnem(repo.path(), &["config", "list"])
        .assert()
        .success();
    let stdout = String::from_utf8_lossy(&out.get_output().stdout).to_string();
    assert!(
        stdout.contains("retrieve.limit = 20"),
        "expected 'retrieve.limit = 20' in list output, got: {stdout}"
    );

    // The TOML file must have the [retrieve] section and limit field.
    let cfg_path = repo.path().join(".mnem").join("config.toml");
    let contents = std::fs::read_to_string(&cfg_path).unwrap();
    assert!(
        contents.contains("[retrieve]"),
        "expected '[retrieve]' section in config.toml, got: {contents}"
    );
    // Use .lines() for cross-platform line matching: distinguishes `20` from `200`
    // and from string `"20"` - both of which `contains("limit = 20")` would accept.
    assert!(
        contents.lines().any(|l| l.trim() == "limit = 20"),
        "expected line 'limit = 20' (integer, not 200 or \"20\") in [retrieve] section; \
         got: {contents}"
    );
}

// ---------------------------------------------------------------------------
// Test 5
// ---------------------------------------------------------------------------

/// Setting `embed.provider ollama` then `embed.model nomic-embed-text`:
///   - get provider → `"ollama"`
///   - get model → `"nomic-embed-text"`
///   - `.mnem/config.toml` contains `[embed]` section, `provider = "ollama"`,
///     and `model = "nomic-embed-text"` as TOML key=value pairs
#[test]
fn config_embed_provider_persists_across_get_and_file() {
    let repo = TempDir::new().unwrap();
    init(repo.path());

    mnem(
        repo.path(),
        &["config", "set", "embed.provider", "ollama"],
    )
    .assert()
    .success();

    mnem(
        repo.path(),
        &["config", "set", "embed.model", "nomic-embed-text"],
    )
    .assert()
    .success();

    // Get provider.
    let out = mnem(repo.path(), &["config", "get", "embed.provider"])
        .assert()
        .success();
    let stdout = String::from_utf8_lossy(&out.get_output().stdout).to_string();
    assert_eq!(
        stdout.trim(),
        "ollama",
        "expected trimmed get stdout == 'ollama', got: {stdout}"
    );

    // Get model.
    let out = mnem(repo.path(), &["config", "get", "embed.model"])
        .assert()
        .success();
    let stdout = String::from_utf8_lossy(&out.get_output().stdout).to_string();
    assert_eq!(
        stdout.trim(),
        "nomic-embed-text",
        "expected trimmed get stdout == 'nomic-embed-text', got: {stdout}"
    );

    // TOML file contents: assert the full key=value pairs, not just bare substrings.
    let cfg_path = repo.path().join(".mnem").join("config.toml");
    let contents = std::fs::read_to_string(&cfg_path).unwrap();
    assert!(
        contents.contains("[embed]"),
        "expected '[embed]' section header in config.toml, got: {contents}"
    );
    assert!(
        contents.contains("provider = \"ollama\""),
        "expected 'provider = \"ollama\"' key=value pair in config.toml, got: {contents}"
    );
    assert!(
        contents.contains("model = \"nomic-embed-text\""),
        "expected 'model = \"nomic-embed-text\"' key=value pair in config.toml, got: {contents}"
    );
}

// ---------------------------------------------------------------------------
// Test 6
// ---------------------------------------------------------------------------

/// `mnem config set invalid.key value` must exit non-zero and print
/// `"unknown config key"` to stderr.
#[test]
fn config_invalid_key_is_rejected() {
    let repo = TempDir::new().unwrap();
    init(repo.path());

    let out = mnem(repo.path(), &["config", "set", "invalid.key", "value"])
        .assert()
        .failure();
    let stderr = String::from_utf8_lossy(&out.get_output().stderr).to_string();
    assert!(
        stderr.contains("unknown config key"),
        "expected 'unknown config key' in stderr, got: {stderr}"
    );
}

// ---------------------------------------------------------------------------
// Test 7
// ---------------------------------------------------------------------------

/// `mnem config get user.name` on a fresh repo (key never set) must exit 1
/// and print `"no value set"` to stderr.
#[test]
fn config_get_missing_key_exits_nonzero() {
    let repo = TempDir::new().unwrap();
    init(repo.path());

    let out = mnem(repo.path(), &["config", "get", "user.name"])
        .assert()
        .failure();
    let stderr = String::from_utf8_lossy(&out.get_output().stderr).to_string();
    assert!(
        stderr.contains("no value set"),
        "expected 'no value set' in stderr for missing key, got: {stderr}"
    );
}

// ---------------------------------------------------------------------------
// Test 8
// ---------------------------------------------------------------------------

/// Per-repo `config set` (without `--global`) must NOT write the global
/// `~/.mnem/config.toml`. The per-repo `.mnem/config.toml` must contain the
/// value; if the fake-home global file happens to exist it must not.
#[test]
fn config_per_repo_write_does_not_touch_global() {
    let repo = TempDir::new().unwrap();
    let fake_home = TempDir::new().unwrap();

    // Init using the regular helper (no home override needed for init).
    mnem_with_home(
        repo.path(),
        fake_home.path(),
        &["init", repo.path().to_str().unwrap()],
    )
    .assert()
    .success();

    // Set a per-repo key (no --global flag).
    mnem_with_home(
        repo.path(),
        fake_home.path(),
        &["config", "set", "user.name", "PerRepoUser"],
    )
    .assert()
    .success();

    // Per-repo config set must never create the global config file at all.
    let global_cfg = fake_home.path().join(".mnem").join("config.toml");
    assert!(
        !global_cfg.exists(),
        "per-repo config set must not create the global config file at {}",
        global_cfg.display()
    );

    // The per-repo config must contain the key=value pair (not just the bare string).
    let repo_cfg = repo.path().join(".mnem").join("config.toml");
    assert!(
        repo_cfg.exists(),
        "expected per-repo .mnem/config.toml to be created"
    );
    let repo_contents = std::fs::read_to_string(&repo_cfg).unwrap();
    assert!(
        repo_contents.contains("name = \"PerRepoUser\""),
        "expected 'name = \"PerRepoUser\"' key=value pair in per-repo config.toml, \
         got: {repo_contents}"
    );
}

// ---------------------------------------------------------------------------
// Test 9
// ---------------------------------------------------------------------------

/// Legacy positional form `mnem config user.name David` (no "set" subcommand)
/// must succeed and echo `"user.name = David"`. Then `mnem config user.name`
/// (no subcommand, no value) must return trimmed `"David"`.
#[test]
fn config_legacy_positional_set_and_get() {
    let repo = TempDir::new().unwrap();
    init(repo.path());

    // Legacy set: `mnem config user.name David`
    let out = mnem(repo.path(), &["config", "user.name", "David"])
        .assert()
        .success();
    let stdout = String::from_utf8_lossy(&out.get_output().stdout).to_string();
    assert!(
        stdout.contains("user.name = David"),
        "expected 'user.name = David' in legacy-set stdout, got: {stdout}"
    );

    // Legacy get: `mnem config user.name`
    let out = mnem(repo.path(), &["config", "user.name"])
        .assert()
        .success();
    let stdout = String::from_utf8_lossy(&out.get_output().stdout).to_string();
    assert_eq!(
        stdout.trim(),
        "David",
        "expected trimmed legacy-get stdout == 'David', got: {stdout}"
    );

    // The legacy set path must write the same TOML key=value pair as the
    // `config set` subcommand. Verifying the file rules out a code path that
    // prints the right stdout but skips the serialisation step.
    let cfg_path = repo.path().join(".mnem").join("config.toml");
    let contents = std::fs::read_to_string(&cfg_path).unwrap();
    assert!(
        contents.contains("name = \"David\""),
        "expected 'name = \"David\"' key=value pair in .mnem/config.toml after legacy set, \
         got: {contents}"
    );
}

// ---------------------------------------------------------------------------
// Test 10
// ---------------------------------------------------------------------------

/// Key-name branch of the API key guardrail: `mnem config set embed.api_key`
/// must exit non-zero and print `"API keys must not"` to stderr **regardless
/// of the value**.
///
/// Two sub-cases prove the key-name branch fires on the key alone, not on the
/// value prefix:
///   a. `embed.api_key sk-secret`       - value-prefix branch would also fire
///   b. `embed.api_key plaintext-value` - only the key-name branch can fire
///
/// If (b) passes without rejection the key-name branch is missing; if only (a)
/// rejects, the rejection is from value-prefix alone and branch 1 is untested.
#[test]
fn config_api_key_guardrail_rejects_raw_secret() {
    let repo = TempDir::new().unwrap();
    init(repo.path());

    // (a) sk- value - both branches would fire; confirms rejection.
    let out = mnem(
        repo.path(),
        &["config", "set", "embed.api_key", "sk-secret"],
    )
    .assert()
    .failure();
    let stderr = String::from_utf8_lossy(&out.get_output().stderr).to_string();
    assert!(
        stderr.contains("API keys must not"),
        "embed.api_key with sk- value must be rejected; got: {stderr}"
    );

    // (b) non-sk- value - only the key-name branch can fire here.
    // This isolates branch 1: rejection must happen on key name alone.
    let out2 = mnem(
        repo.path(),
        &["config", "set", "embed.api_key", "plaintext-not-a-secret"],
    )
    .assert()
    .failure();
    let stderr2 = String::from_utf8_lossy(&out2.get_output().stderr).to_string();
    assert!(
        stderr2.contains("API keys must not"),
        "embed.api_key with a non-sk- value must ALSO be rejected (key-name branch \
         fires regardless of value prefix); got: {stderr2}"
    );
}

// ---------------------------------------------------------------------------
// Test 11
// ---------------------------------------------------------------------------

/// Value-prefix branch of the API key guardrail: when the value starts with
/// `"sk-"` and the key is in the `embed.*` namespace (but is NOT `embed.api_key`
/// itself), the guardrail fires on the value prefix alone.
///
/// We use `embed.model` - a known valid config key whose name does not match
/// the key-name branch (`embed.api_key`). This proves the value-prefix check
/// fires independently of the key-name check.
#[test]
fn config_api_key_guardrail_rejects_sk_prefix_value() {
    let repo = TempDir::new().unwrap();
    init(repo.path());

    // Positive control: embed.model with a non-sk- value must succeed.
    // This confirms the key-name branch does NOT fire for embed.model,
    // so only the value-prefix branch is responsible for the rejection below.
    // set_embed_model requires embed.provider to be set first (it calls
    // cfg.embed.as_mut()...) so we seed onnx before setting the model.
    mnem(
        repo.path(),
        &["config", "set", "embed.provider", "onnx"],
    )
    .assert()
    .success();
    mnem(
        repo.path(),
        &["config", "set", "embed.model", "bge-small-en-v1.5"],
    )
    .assert()
    .success();

    // embed.model is a known valid key whose name does NOT match the key-name
    // branch (`embed.api_key`), so only the value-prefix branch can fire here.
    let out = mnem(
        repo.path(),
        &["config", "set", "embed.model", "sk-this-is-a-secret-not-a-model"],
    )
    .assert()
    .failure();
    let stderr = String::from_utf8_lossy(&out.get_output().stderr).to_string();
    assert!(
        stderr.contains("API keys must not"),
        "sk-prefixed value for a non-api-key embed.* key must trigger the value-prefix \
         guardrail branch; got: {stderr}"
    );
}

// ---------------------------------------------------------------------------
// Test 12
// ---------------------------------------------------------------------------

/// Key-name branch of the API key guardrail for the `rerank.*` namespace:
/// `mnem config set rerank.api_key` must exit non-zero and print
/// `"API keys must not"` to stderr **regardless of the value**.
///
/// Symmetric to test 10 which covers `embed.api_key`; two sub-cases prove
/// the key-name branch fires on the key name alone, not on the value prefix:
///   a. `rerank.api_key sk-secret`       - both branches would fire
///   b. `rerank.api_key plaintext-value` - only key-name branch fires
#[test]
fn config_api_key_guardrail_rejects_rerank_api_key() {
    let repo = TempDir::new().unwrap();
    init(repo.path());

    // (a) sk- value - both branches fire; confirms rejection.
    let out = mnem(
        repo.path(),
        &["config", "set", "rerank.api_key", "sk-secret"],
    )
    .assert()
    .failure();
    let stderr = String::from_utf8_lossy(&out.get_output().stderr).to_string();
    assert!(
        stderr.contains("API keys must not"),
        "rerank.api_key with sk- value must be rejected; got: {stderr}"
    );

    // (b) non-sk- value - only the key-name branch fires.
    // Proves the rerank.api_key key-name branch rejects regardless of value prefix,
    // symmetric to the embed.api_key isolation in test 10b.
    let out2 = mnem(
        repo.path(),
        &["config", "set", "rerank.api_key", "plaintext-rerank-key"],
    )
    .assert()
    .failure();
    let stderr2 = String::from_utf8_lossy(&out2.get_output().stderr).to_string();
    assert!(
        stderr2.contains("API keys must not"),
        "rerank.api_key with a non-sk- value must ALSO be rejected (key-name branch \
         fires regardless of value prefix); got: {stderr2}"
    );
}

// ---------------------------------------------------------------------------
// Test 13
// ---------------------------------------------------------------------------

/// `mnem config set` on a key that already has a value must overwrite it:
/// the second value wins and the first value must not appear in the TOML file.
#[test]
fn config_set_overwrites_existing_value() {
    let repo = TempDir::new().unwrap();
    init(repo.path());

    mnem(repo.path(), &["config", "set", "user.name", "First"])
        .assert()
        .success();

    // Overwrite with a second value.
    mnem(repo.path(), &["config", "set", "user.name", "Second"])
        .assert()
        .success();

    // Get must return the second value.
    let out = mnem(repo.path(), &["config", "get", "user.name"])
        .assert()
        .success();
    let stdout = String::from_utf8_lossy(&out.get_output().stdout).to_string();
    assert_eq!(
        stdout.trim(),
        "Second",
        "second set must overwrite the first; expected 'Second', got: {stdout}"
    );

    // TOML file must contain the new key=value pair and must not contain the old value.
    let cfg_path = repo.path().join(".mnem").join("config.toml");
    let contents = std::fs::read_to_string(&cfg_path).unwrap();
    assert!(
        contents.contains("name = \"Second\""),
        "config.toml must contain 'name = \"Second\"' after overwrite; got: {contents}"
    );
    assert!(
        !contents.contains("First"),
        "config.toml must not contain 'First' after overwrite; got: {contents}"
    );
}

// ---------------------------------------------------------------------------
// Test 14
// ---------------------------------------------------------------------------

/// `mnem config unset` on a key that was never set must succeed (exit 0) and
/// print `"unset <key>"` - the same output as a successful unset.
///
/// This pins the no-op behaviour: unset is idempotent and safe to call in
/// order-independent scripts without checking whether the key exists first.
#[test]
fn config_unset_never_set_key_exits_zero() {
    let repo = TempDir::new().unwrap();
    init(repo.path());

    // user.name is not set in this fresh repo - unset must still succeed.
    let out = mnem(repo.path(), &["config", "unset", "user.name"])
        .assert()
        .success();
    let stdout = String::from_utf8_lossy(&out.get_output().stdout).to_string();
    assert!(
        stdout.contains("unset user.name"),
        "unset on a never-set key must print 'unset user.name', got: {stdout}"
    );
}

// ---------------------------------------------------------------------------
// Test 15
// ---------------------------------------------------------------------------

/// Value-prefix branch of the API key guardrail for the `rerank.*` namespace:
/// `mnem config set rerank.model sk-secret` must exit non-zero and print
/// `"API keys must not"` to stderr.
///
/// The guardrail fires at the top of `set_dotted` before any provider-existence
/// check, so no prior `rerank.provider` set is required. This proves the
/// value-prefix check fires across both protected namespaces (`embed.*` and
/// `rerank.*`), symmetric to test 11 for embed.
#[test]
fn config_api_key_guardrail_rejects_sk_prefix_in_rerank_namespace() {
    let repo = TempDir::new().unwrap();
    init(repo.path());

    // rerank.model is a valid config key (not in the key-name branch), so
    // only the value-prefix check can produce the rejection below.
    let out = mnem(
        repo.path(),
        &["config", "set", "rerank.model", "sk-this-is-a-secret-not-a-model"],
    )
    .assert()
    .failure();
    let stderr = String::from_utf8_lossy(&out.get_output().stderr).to_string();
    assert!(
        stderr.contains("API keys must not"),
        "sk-prefixed value for rerank.model must trigger the value-prefix guardrail \
         in the rerank.* namespace; got: {stderr}"
    );
}

// ---------------------------------------------------------------------------
// Test 16
// ---------------------------------------------------------------------------

/// Positive round-trip for the `rerank.*` config namespace, symmetric to
/// test 5 for `embed.*`.
///
/// Sets `rerank.provider cohere` (which also seeds the default model
/// `"rerank-v3.5"`) then `rerank.api_key_env COHERE_API_KEY`, then:
///   - verifies both keys are readable via `config get`
///   - verifies `.mnem/config.toml` contains `[rerank]` section with the
///     expected TOML key=value pairs (`provider`, `model`, `api_key_env`)
///
/// This proves the rerank namespace works end-to-end through the CLI,
/// not just in unit tests of `set_dotted`.
#[test]
fn config_rerank_namespace_round_trips() {
    let repo = TempDir::new().unwrap();
    init(repo.path());

    // Set the provider first - required before any other rerank.* key.
    // "cohere" is a known valid provider; seeds model = "rerank-v3.5".
    mnem(
        repo.path(),
        &["config", "set", "rerank.provider", "cohere"],
    )
    .assert()
    .success();

    // Set the api_key_env. Must be [A-Z_][A-Z0-9_]+ shape (env-var name).
    mnem(
        repo.path(),
        &["config", "set", "rerank.api_key_env", "COHERE_API_KEY"],
    )
    .assert()
    .success();

    // Get provider must return "cohere".
    let out = mnem(repo.path(), &["config", "get", "rerank.provider"])
        .assert()
        .success();
    let stdout = String::from_utf8_lossy(&out.get_output().stdout).to_string();
    assert_eq!(
        stdout.trim(),
        "cohere",
        "expected trimmed get stdout == 'cohere', got: {stdout}"
    );

    // Get api_key_env must return "COHERE_API_KEY".
    let out = mnem(repo.path(), &["config", "get", "rerank.api_key_env"])
        .assert()
        .success();
    let stdout = String::from_utf8_lossy(&out.get_output().stdout).to_string();
    assert_eq!(
        stdout.trim(),
        "COHERE_API_KEY",
        "expected trimmed get stdout == 'COHERE_API_KEY', got: {stdout}"
    );

    // TOML file must contain [rerank] section with the expected key=value pairs.
    // The tag field `provider = "cohere"` is written by serde's tagged enum.
    let cfg_path = repo.path().join(".mnem").join("config.toml");
    let contents = std::fs::read_to_string(&cfg_path).unwrap();
    assert!(
        contents.contains("[rerank]"),
        "expected '[rerank]' section header in config.toml, got: {contents}"
    );
    assert!(
        contents.contains("provider = \"cohere\""),
        "expected 'provider = \"cohere\"' tag key=value in config.toml, got: {contents}"
    );
    assert!(
        contents.contains("model = \"rerank-v3.5\""),
        "expected 'model = \"rerank-v3.5\"' (default seeded by provider switch) \
         in config.toml, got: {contents}"
    );
    assert!(
        contents.contains("api_key_env = \"COHERE_API_KEY\""),
        "expected 'api_key_env = \"COHERE_API_KEY\"' key=value in config.toml, got: {contents}"
    );
}

// ---------------------------------------------------------------------------
// Test 17
// ---------------------------------------------------------------------------

/// After a guardrail rejection, `.mnem/config.toml` must be byte-for-byte
/// identical to its pre-attempt state.
///
/// The guardrail in `set_dotted` bails before calling `config::save`, so
/// the file must never be written. This test captures the file contents
/// before the rejected attempt and asserts exact equality afterwards -
/// ruling out partial writes or any modification of the on-disk state.
#[test]
fn config_guardrail_rejection_does_not_modify_toml() {
    let repo = TempDir::new().unwrap();
    init(repo.path());

    // Read the config.toml immediately after init (the baseline).
    let cfg_path = repo.path().join(".mnem").join("config.toml");
    let contents_before = std::fs::read_to_string(&cfg_path)
        .expect("init must create config.toml");

    // Attempt to set embed.api_key - must be rejected by the key-name guardrail.
    mnem(
        repo.path(),
        &["config", "set", "embed.api_key", "sk-secret"],
    )
    .assert()
    .failure();

    // The file must be byte-for-byte identical: the guardrail bailed before save.
    let contents_after = std::fs::read_to_string(&cfg_path)
        .expect("config.toml must still exist after a failed set");
    assert_eq!(
        contents_before,
        contents_after,
        "guardrail rejection must not modify config.toml at all; \
         file changed between before and after the rejected set"
    );

    // Belt-and-suspenders: the rejected value must not appear anywhere in the file.
    assert!(
        !contents_after.contains("sk-secret"),
        "TOML file must not contain the rejected value 'sk-secret'; got: {contents_after}"
    );
    // The api_key assignment key must not appear (api_key_env is distinct from api_key).
    assert!(
        !contents_after.contains("api_key = "),
        "TOML file must not contain 'api_key = ' after a guardrail rejection; \
         got: {contents_after}"
    );
}

// ---------------------------------------------------------------------------
// Test 18
// ---------------------------------------------------------------------------

/// `mnem config list` after `mnem config unset` must not show the unset key.
///
/// Sequence: set → verify in list → unset → verify absent from list.
/// This is distinct from test 3 (which checks get and the TOML file) and
/// test 14 (which checks unset exit code on a never-set key): here we
/// specifically exercise the list output before and after an unset.
#[test]
fn config_list_after_unset_key_absent() {
    let repo = TempDir::new().unwrap();
    init(repo.path());

    // Set a key.
    mnem(
        repo.path(),
        &["config", "set", "user.email", "test@example.com"],
    )
    .assert()
    .success();

    // Verify the key appears in list output before unset.
    let out = mnem(repo.path(), &["config", "list"]).assert().success();
    let stdout = String::from_utf8_lossy(&out.get_output().stdout).to_string();
    assert!(
        stdout.contains("user.email = test@example.com"),
        "user.email must appear in list before unset; got: {stdout}"
    );

    // Unset the key.
    mnem(repo.path(), &["config", "unset", "user.email"])
        .assert()
        .success();

    // After unset, list must not show the key (or any trace of the value).
    let out = mnem(repo.path(), &["config", "list"]).assert().success();
    let stdout = String::from_utf8_lossy(&out.get_output().stdout).to_string();
    assert!(
        !stdout.contains("user.email"),
        "user.email must NOT appear in list after unset; got: {stdout}"
    );
}

// ---------------------------------------------------------------------------
// Test 19
// ---------------------------------------------------------------------------

/// `mnem config set` in a directory that was never initialised (no `.mnem/`)
/// must fail with a non-empty error message.
///
/// Without `mnem init`, there is no `.mnem/` directory for `config::save` to
/// write into, so the binary must reject the operation and surface an error
/// rather than silently creating an orphaned config outside a valid repo.
#[test]
fn config_set_in_uninitialised_repo_fails() {
    // A directory that has never had `mnem init` run - no .mnem/ present.
    let dir = TempDir::new().unwrap();
    // Do NOT call init(dir.path()); .mnem/ is intentionally absent.

    let out = mnem(dir.path(), &["config", "set", "user.name", "Alice"])
        .assert()
        .failure();
    let stderr = String::from_utf8_lossy(&out.get_output().stderr).to_string();

    // The command must fail and produce a diagnostic on stderr.
    assert!(
        !stderr.is_empty(),
        "config set on uninitialised repo must produce an error message on stderr"
    );
    // The error must hint at the missing .mnem/ directory or write failure.
    // Accepted phrasings: write error path mentions ".mnem", OS "not found"
    // message, or any repo-existence diagnostic.
    assert!(
        stderr.contains(".mnem")
            || stderr.contains("writing")
            || stderr.contains("not found")
            || stderr.contains("repository"),
        "error must hint at missing .mnem directory or write failure; got: {stderr}"
    );
}
