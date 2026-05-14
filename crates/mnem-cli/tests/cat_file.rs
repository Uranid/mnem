//! Integration tests for `mnem cat-file`.
//!
//! Tests:
//!
//! 1. `cat-file <op-cid>` emits the raw DAG-CBOR bytes; the first
//!    byte is a CBOR map-prefix (0xa0..0xbf).
//! 2. `cat-file <op-cid> --json` emits valid JSON containing the
//!    `_kind` discriminator.
//! 3. `cat-file <unknown-cid>` fails with an actionable message.
//! 4. `cat-file <commit-cid>` (Commit object type, not Operation)
//!    emits raw CBOR map bytes, verifying the raw path works for
//!    all object kinds, not just Operations.
//! 5. Raw path must NOT append a trailing newline; `--json` path MUST.
//!    This is the binary-contract boundary between the two modes.
//! 6. Invalid CID syntax produces a clear parse-error, distinct from
//!    the "not present" error for a valid-but-absent CID.
//! 7. `--json` output is parseable JSON with a `_kind` string field
//!    (structural, not just substring match).

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

/// Walk `mnem log -n 1` output and pull the leading op-cid. Format is
/// `op <cid>` on the first non-empty line.
fn current_op_cid(repo: &Path) -> String {
    let out = mnem(repo, &["log", "-n", "1"]).assert().success();
    let stdout = String::from_utf8_lossy(&out.get_output().stdout).to_string();
    for line in stdout.lines() {
        if let Some(rest) = line.strip_prefix("op ") {
            return rest.trim().to_string();
        }
    }
    panic!("log -n 1 had no op line: {stdout}");
}

/// Build a minimal repo with one committed node and return the op-cid
/// of the commit operation.
fn setup_with_node(dir: &TempDir) -> String {
    mnem(dir.path(), &["init", dir.path().to_str().unwrap()])
        .assert()
        .success();
    mnem(dir.path(), &["add", "node", "--summary", "cat-file test node", "--no-embed"])
        .assert()
        .success();
    current_op_cid(dir.path())
}

/// Extract the HEAD commit CID by following the op -> view -> heads chain.
///
/// Operation blocks store `view` as a CID link to a separate View block
/// (they are distinct blockstore entries). The View block's `heads` array
/// holds CID links to committed tree roots.
///
/// In DAG-JSON (the output of `cat-file --json`), all CID links are encoded
/// as `{"/": "<base32-cid>"}` objects.
///
/// Returns `None` if the view has no heads (e.g. the root-init op),
/// or if the JSON structure is unexpectedly shaped.
fn head_commit_cid_from_op(repo: &Path, op_cid: &str) -> Option<String> {
    // Step 1: fetch the operation and extract the view CID link.
    let op_out = mnem(repo, &["cat-file", op_cid, "--json"])
        .assert()
        .success();
    let op_json: serde_json::Value =
        serde_json::from_slice(&op_out.get_output().stdout).expect("op --json must be valid JSON");

    // Operation: { "view": {"/": "<view-cid>"}, ... }
    let view_cid = op_json
        .get("view")
        .and_then(|v| v.get("/"))
        .and_then(|cid| cid.as_str())?
        .to_string();

    // Step 2: fetch the View block and extract the first head CID link.
    let view_out = mnem(repo, &["cat-file", &view_cid, "--json"])
        .assert()
        .success();
    let view_json: serde_json::Value =
        serde_json::from_slice(&view_out.get_output().stdout)
            .expect("view --json must be valid JSON");

    // View: { "heads": [{"/": "<commit-cid>"}, ...], ... }
    view_json
        .get("heads")
        .and_then(|h| h.as_array())
        .and_then(|a| a.first())
        .and_then(|link| link.get("/"))
        .and_then(|cid| cid.as_str())
        .map(|s| s.to_string())
}

// ---------------------------------------------------------------------------
// Test 1: raw bytes look like a CBOR map
// ---------------------------------------------------------------------------

#[test]
fn cat_file_raw_bytes_look_like_cbor_map() {
    let dir = TempDir::new().unwrap();
    mnem(dir.path(), &["init", dir.path().to_str().unwrap()])
        .assert()
        .success();
    // Give the repo a head commit so the op-log has a non-root op.
    mnem(dir.path(), &["add", "node", "--summary", "a", "--no-embed"])
        .assert()
        .success();
    let op = current_op_cid(dir.path());

    let out = mnem(dir.path(), &["cat-file", &op]).assert().success();
    let stdout = &out.get_output().stdout;
    assert!(!stdout.is_empty(), "raw cat-file should emit bytes");
    // DAG-CBOR maps start with major type 5 (0xa0..0xbf). Every mnem
    // object is a map, so the first byte MUST fall in that range.
    let first = stdout[0];
    assert!(
        (0xa0..=0xbf).contains(&first),
        "expected CBOR map first byte (0xa0..0xbf), got {first:#x}"
    );
}

// ---------------------------------------------------------------------------
// Test 2: --json emits valid JSON with _kind
// ---------------------------------------------------------------------------

#[test]
fn cat_file_json_carries_kind_field() {
    let dir = TempDir::new().unwrap();
    mnem(dir.path(), &["init", dir.path().to_str().unwrap()])
        .assert()
        .success();
    mnem(dir.path(), &["add", "node", "--summary", "b", "--no-embed"])
        .assert()
        .success();
    let op = current_op_cid(dir.path());

    let out = mnem(dir.path(), &["cat-file", &op, "--json"])
        .assert()
        .success();
    let stdout = String::from_utf8_lossy(&out.get_output().stdout).to_string();
    // Every mnem object carries a `_kind` string. Operation's kind is
    // `"operation"`; we assert on the substring rather than a full
    // JSON parse to keep the test robust against future additive
    // fields.
    assert!(
        stdout.contains("\"_kind\""),
        "JSON must carry _kind discriminator, got: {stdout}"
    );
    assert!(
        stdout.contains("\"operation\""),
        "this op block must be of kind `operation`, got: {stdout}"
    );
}

// ---------------------------------------------------------------------------
// Test 3: unknown CID fails with "not present"
// ---------------------------------------------------------------------------

#[test]
fn cat_file_unknown_cid_errors() {
    let dir = TempDir::new().unwrap();
    mnem(dir.path(), &["init", dir.path().to_str().unwrap()])
        .assert()
        .success();
    // A syntactically-valid but absent CID. Use the bare sha256 of
    // something we never stored. Format is multibase+codec+hash;
    // `bafk...` is a well-known DAG-CBOR CID prefix that does not
    // exist in this repo.
    let missing = "bafkreibm6jg3ux5qumhcn2b3flc3tyu6dmlb4xa7u5bf44yegnrjhc3pgi";
    let out = mnem(dir.path(), &["cat-file", missing]).assert().failure();
    let stderr = String::from_utf8_lossy(&out.get_output().stderr).to_string();
    assert!(
        stderr.contains("not present"),
        "expected not-present error, got: {stderr}"
    );
}

// ---------------------------------------------------------------------------
// Test 4: Commit object type through the raw binary path
// ---------------------------------------------------------------------------

/// `cat-file` without `--json` must work for Commit objects, not just
/// Operations.  After `mnem add node` the latest operation carries a
/// `view.heads` list; the first entry is the commit CID.  Cat-filing
/// that CID raw must emit a CBOR map (the Commit struct is also a map).
///
/// This test verifies that the raw path is object-type-agnostic: the
/// blockstore lookup and binary write succeed for any stored CID, not
/// only for Operations (which is all the existing tests exercise).
#[test]
fn cat_file_commit_object_raw_returns_cbor_map() {
    let dir = TempDir::new().unwrap();
    let op_cid = setup_with_node(&dir);

    let commit_cid = head_commit_cid_from_op(dir.path(), &op_cid).expect(
        "add node must produce a committed head; view.heads must be non-empty in the op JSON",
    );

    // cat-file the commit CID (Commit type, not Operation)
    let out = mnem(dir.path(), &["cat-file", &commit_cid])
        .assert()
        .success();
    let stdout = &out.get_output().stdout;
    assert!(
        !stdout.is_empty(),
        "cat-file of a commit CID must emit bytes; CID={commit_cid}"
    );
    let first = stdout[0];
    assert!(
        (0xa0..=0xbf).contains(&first),
        "Commit object is a CBOR map; expected first byte 0xa0..0xbf, \
         got {first:#x}; CID={commit_cid}"
    );
}

// ---------------------------------------------------------------------------
// Test 5: raw vs --json newline contract
// ---------------------------------------------------------------------------

/// Raw path must NOT append a trailing newline; `--json` path MUST.
///
/// The raw path writes the exact blockstore bytes to stdout — binary-safe,
/// no decoration.  The `--json` path adds `\n` so shell prompts don't
/// collide with the closing brace (see cat_file.rs line 74-76).
///
/// A bug that added `\n` to raw output would corrupt pipes into tools
/// like `cbor-diag` or `xxd` that count expected bytes.
#[test]
fn cat_file_raw_has_no_trailing_newline_json_does() {
    let dir = TempDir::new().unwrap();
    let op_cid = setup_with_node(&dir);

    // Raw mode: no trailing newline.
    let raw_out = mnem(dir.path(), &["cat-file", &op_cid])
        .assert()
        .success();
    let raw_bytes = &raw_out.get_output().stdout;
    assert!(
        !raw_bytes.is_empty(),
        "raw cat-file output must not be empty"
    );
    assert_ne!(
        *raw_bytes.last().unwrap(),
        b'\n',
        "raw cat-file must NOT append a trailing newline; \
         last byte is {:#x}, full length {} bytes",
        raw_bytes.last().unwrap(),
        raw_bytes.len()
    );

    // JSON mode: trailing newline is required.
    let json_out = mnem(dir.path(), &["cat-file", &op_cid, "--json"])
        .assert()
        .success();
    let json_bytes = &json_out.get_output().stdout;
    assert_eq!(
        *json_bytes.last().unwrap(),
        b'\n',
        "cat-file --json must append a trailing newline; \
         last byte is {:#x}, full length {} bytes",
        json_bytes.last().unwrap(),
        json_bytes.len()
    );
}

// ---------------------------------------------------------------------------
// Test 6: invalid CID syntax vs missing CID produce distinct errors
// ---------------------------------------------------------------------------

/// A string that cannot be parsed as a multibase CID at all must
/// produce a "invalid CID" error, NOT the "not present" blockstore
/// error reserved for syntactically-valid-but-absent CIDs.
///
/// This verifies the two error paths in cat_file.rs are correctly
/// separated: `parse_str` failure (line 51) vs `bs.get` miss (line 52).
#[test]
fn cat_file_invalid_cid_syntax_errors_distinctly_from_missing() {
    let dir = TempDir::new().unwrap();
    mnem(dir.path(), &["init", dir.path().to_str().unwrap()])
        .assert()
        .success();

    // "not-a-cid" has no multibase prefix and cannot be parsed.
    let out = mnem(dir.path(), &["cat-file", "not-a-cid"])
        .assert()
        .failure();
    let stderr = String::from_utf8_lossy(&out.get_output().stderr).to_string();

    // Must say something about the CID being invalid, not about it
    // being absent from the blockstore.
    assert!(
        stderr.contains("invalid CID") || stderr.contains("invalid"),
        "invalid syntax must produce a parse-error mentioning 'invalid'; got: {stderr}"
    );
    assert!(
        !stderr.contains("not present"),
        "syntax error must be distinct from the missing-block error; got: {stderr}"
    );
}

// ---------------------------------------------------------------------------
// Test 7: --json output is structurally valid JSON with a string _kind
// ---------------------------------------------------------------------------

/// Verifies the JSON mode contract beyond substring matching: the
/// output must be a parseable JSON object whose `_kind` field is a
/// non-empty string.  A bug that emitted truncated or syntactically
/// invalid JSON would pass the `contains("\"_kind\"")` substring check
/// in test 2 but fail here.
#[test]
fn cat_file_json_is_parseable_object_with_string_kind() {
    let dir = TempDir::new().unwrap();
    let op_cid = setup_with_node(&dir);

    let out = mnem(dir.path(), &["cat-file", &op_cid, "--json"])
        .assert()
        .success();
    let stdout = String::from_utf8_lossy(&out.get_output().stdout).to_string();

    let parsed: serde_json::Value = serde_json::from_str(stdout.trim())
        .expect("cat-file --json must produce parseable JSON");

    assert!(
        parsed.is_object(),
        "cat-file --json must produce a JSON object (top-level), got: {parsed:?}"
    );
    let kind = parsed
        .get("_kind")
        .and_then(|v| v.as_str())
        .expect("_kind field must be present and a string");
    assert!(
        !kind.is_empty(),
        "_kind must be a non-empty string; got empty string"
    );
}
