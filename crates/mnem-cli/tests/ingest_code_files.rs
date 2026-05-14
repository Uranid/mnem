//! CLI integration tests for `mnem ingest` on code files (.rs, .py).
//!
//! Verifies that the ingest command handles code source kinds correctly:
//! Rust and Python files are parsed via tree-sitter into function/class-level
//! chunks, unknown extensions fall back to `SourceKind::Text`, and recursive
//! directory ingestion handles mixed file types.

use std::path::Path;
use std::process::Command;

use assert_cmd::prelude::*;
use tempfile::TempDir;

fn mnem(repo: &Path, args: &[&str]) -> Command {
    let mut cmd = Command::cargo_bin("mnem").expect("mnem binary built");
    cmd.current_dir(repo).arg("-R").arg(repo);
    cmd.env("HOME", repo);
    cmd.env("USERPROFILE", repo);
    cmd.env("MNEM_DISABLE_GLOBAL_CONFIG", "1");
    cmd.env_remove("MNEM_EMBED_PROVIDER");
    cmd.env_remove("MNEM_EMBED_MODEL");
    cmd.env_remove("MNEM_EMBED_API_KEY_ENV");
    cmd.env_remove("MNEM_EMBED_BASE_URL");
    for a in args {
        cmd.arg(a);
    }
    cmd
}

/// Parse `N` from a segment of the "ingested N files, M chunks, ..." line.
///
/// Splits on commas, strips whitespace, then looks for a segment that ends
/// with `" {label}"`.  The very first segment starts with `"ingested "`, so
/// stripping the suffix `" files"` from `"ingested 1 files"` leaves
/// `"ingested 1"` - handle that by also trying to strip the `"ingested "`
/// prefix before parsing.
fn parse_count(stdout: &str, label: &str) -> usize {
    let suffix = format!(" {label}");
    stdout
        .split(',')
        .find_map(|seg| {
            let seg = seg.trim();
            let num_str = seg.strip_suffix(&suffix)?;
            // The first segment looks like "ingested 1 files"; strip the
            // leading "ingested " if present before parsing.
            let num_str = num_str
                .strip_prefix("ingested ")
                .unwrap_or(num_str)
                .trim();
            num_str.parse::<usize>().ok()
        })
        .unwrap_or_else(|| panic!("could not find '{label}' count in: {stdout}"))
}

#[test]
fn ingest_rust_file_succeeds() {
    let td = TempDir::new().expect("tmp");
    let repo = td.path();

    mnem(repo, &["init"])
        .ok()
        .expect("mnem init should succeed");

    let file = repo.join("lib.rs");
    // Two top-level functions so tree-sitter produces >= 2 structural chunks.
    std::fs::write(
        &file,
        "pub fn add(a: i32, b: i32) -> i32 { a + b }\n\
         pub fn subtract(a: i32, b: i32) -> i32 { a - b }\n",
    )
    .unwrap();

    let out = mnem(repo, &["ingest", file.to_str().unwrap()])
        .output()
        .expect("spawn");

    assert!(
        out.status.success(),
        "mnem ingest on a .rs file should succeed; stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );

    let stdout = String::from_utf8_lossy(&out.stdout).to_string();
    assert!(
        stdout.contains("ingested"),
        "stdout should contain 'ingested'; got: {stdout}"
    );

    let chunk_count = parse_count(&stdout, "chunks");
    assert!(
        chunk_count >= 2,
        "Rust file with 2 top-level functions must produce >= 2 structural chunks; got {chunk_count}"
    );
}

#[test]
fn ingest_python_file_succeeds() {
    let td = TempDir::new().expect("tmp");
    let repo = td.path();

    mnem(repo, &["init"])
        .ok()
        .expect("mnem init should succeed");

    let file = repo.join("util.py");
    // Two top-level functions so tree-sitter produces >= 2 structural chunks.
    std::fs::write(
        &file,
        "def hello():\n    return 'world'\n\ndef add(a, b):\n    return a + b\n",
    )
    .unwrap();

    let out = mnem(repo, &["ingest", file.to_str().unwrap()])
        .output()
        .expect("spawn");

    assert!(
        out.status.success(),
        "mnem ingest on a .py file should succeed; stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );

    let stdout = String::from_utf8_lossy(&out.stdout).to_string();
    assert!(
        stdout.contains("ingested"),
        "stdout should contain 'ingested'; got: {stdout}"
    );

    let chunk_count = parse_count(&stdout, "chunks");
    assert!(
        chunk_count >= 2,
        "Python file with 2 top-level functions must produce >= 2 structural chunks; got {chunk_count}"
    );
}

#[test]
fn ingest_recursive_directory_with_mixed_extensions() {
    let td = TempDir::new().expect("tmp");
    let repo = td.path();

    mnem(repo, &["init"])
        .ok()
        .expect("mnem init should succeed");

    let src = repo.join("src");
    std::fs::create_dir(&src).unwrap();

    // Two functions each so tree-sitter produces >= 2 structural chunks per code file.
    std::fs::write(
        src.join("lib.rs"),
        "pub fn add(a: i32, b: i32) -> i32 { a + b }\n\
         pub fn sub(a: i32, b: i32) -> i32 { a - b }\n",
    )
    .unwrap();
    std::fs::write(
        src.join("util.py"),
        "def greet(name):\n    return f\"hello {name}\"\ndef farewell(name):\n    return f\"bye {name}\"\n",
    )
    .unwrap();
    std::fs::write(
        src.join("notes.md"),
        "# Notes\n\nSome project notes here.\n",
    )
    .unwrap();

    let out = mnem(repo, &["ingest", "--recursive", src.to_str().unwrap()])
        .output()
        .expect("spawn");

    assert!(
        out.status.success(),
        "mnem ingest --recursive on mixed src/ dir should succeed; stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );

    let stdout = String::from_utf8_lossy(&out.stdout).to_string();
    // The output format is "ingested N files, ..."
    assert!(
        stdout.contains("ingested"),
        "stdout should contain 'ingested'; got: {stdout}"
    );
    // 3 files should have been processed
    assert!(
        stdout.contains("ingested 3 files"),
        "expected 'ingested 3 files' in stdout, got: {stdout}"
    );
    // lib.rs (2 fns) + util.py (2 fns) + notes.md (1 chunk) = exactly 5 chunks total.
    let chunk_count = parse_count(&stdout, "chunks");
    assert!(
        chunk_count >= 5,
        "mixed recursive ingest of 2+2 code functions + 1 md file must produce >= 5 chunks; \
         got {chunk_count}"
    );
}

#[test]
fn ingest_unsupported_extension_falls_back_to_text() {
    let td = TempDir::new().expect("tmp");
    let repo = td.path();

    mnem(repo, &["init"])
        .ok()
        .expect("mnem init should succeed");

    // Single short line - Text fallback should produce exactly 1 chunk.
    let file = repo.join("data.xyz");
    std::fs::write(&file, "hello world\n").unwrap();

    let out = mnem(repo, &["ingest", file.to_str().unwrap()])
        .output()
        .expect("spawn");

    // Unknown extensions fall back to SourceKind::Text, so ingest should succeed.
    assert!(
        out.status.success(),
        "mnem ingest on unknown extension should succeed (Text fallback); stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );

    let stdout = String::from_utf8_lossy(&out.stdout).to_string();
    assert!(
        stdout.contains("ingested"),
        "stdout should contain 'ingested'; got: {stdout}"
    );

    let chunk_count = parse_count(&stdout, "chunks");
    // "hello world\n" is a single short sentence: the SentenceRecursive chunker
    // (invoked for SourceKind::Text with auto chunker, max_tokens=512) produces
    // exactly 1 chunk. assert_eq! documents the Text-fallback contract precisely.
    assert_eq!(
        chunk_count,
        1,
        "text-fallback for unknown extension must produce exactly 1 chunk for a single-line file; got {chunk_count}"
    );
}

#[test]
fn ingest_comment_only_code_file_produces_one_chunk() {
    let td = TempDir::new().expect("tmp");
    let repo = td.path();

    mnem(repo, &["init"])
        .ok()
        .expect("mnem init should succeed");

    // A .rs file with only imports and comments but no function definitions.
    // tree-sitter parses it successfully, finds no function_item nodes, and
    // the fallback in code.rs returns one headless section containing the
    // full file text.  The chunker then produces exactly 1 chunk from it.
    let file = repo.join("comment_only.rs");
    std::fs::write(&file, "use std::io;\n// no function definitions in this file\n").unwrap();

    let out = mnem(repo, &["ingest", file.to_str().unwrap()])
        .output()
        .expect("spawn");

    assert!(
        out.status.success(),
        "mnem ingest on a comment-only .rs file should succeed; stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );

    let stdout = String::from_utf8_lossy(&out.stdout).to_string();
    assert!(
        stdout.contains("ingested"),
        "stdout should contain 'ingested'; got: {stdout}"
    );

    // Fallback section = whole file as one chunk; no functions means no
    // structural splits, so the chunker emits exactly 1 chunk.
    let chunk_count = parse_count(&stdout, "chunks");
    assert_eq!(
        chunk_count, 1,
        "a comment-only .rs file must produce exactly 1 chunk via the fallback section; \
         got {chunk_count}"
    );

    // The Doc root node is always created, so at least 1 node must exist.
    let node_count = parse_count(&stdout, "nodes");
    // For this minimal input: 1 doc root node + 1 chunk node = exactly 2 nodes.
    // No named entities are extracted from a use-statement + comment, so entity
    // nodes do not inflate the count.
    assert_eq!(
        node_count,
        2,
        "comment-only .rs file must produce exactly 2 nodes (1 doc + 1 chunk node); got {node_count}"
    );
}

#[test]
fn ingest_invalid_utf8_code_file_fails_with_error() {
    // A .rs file containing invalid UTF-8 bytes cannot be processed as text.
    // The pipeline must fail with a non-zero exit code and a useful error message.
    let dir = TempDir::new().unwrap();
    mnem(dir.path(), &["init"])
        .assert()
        .success();
    let bad_rs = dir.path().join("bad.rs");
    // The CLI calls `count_chunks_for` first; the `unwrap_or(0)` at the call
    // site in `run()` silently swallows UTF-8 errors at that stage.
    // `Ingester::ingest` then runs, where `std::str::from_utf8` fails and
    // propagates via anyhow as a non-zero exit. The anyhow context chain
    // includes the file path ("bad.rs"), giving a specific error message.
    // Bytes that are not valid UTF-8
    std::fs::write(&bad_rs, &[0xff, 0xfe, 0x80, 0x81, 0x82]).unwrap();
    let out = mnem(dir.path(), &["ingest", "bad.rs"])
        .assert()
        .failure();
    let stderr = String::from_utf8_lossy(&out.get_output().stderr).to_string();
    assert!(
        !stderr.is_empty(),
        "ingest of invalid-UTF-8 file must emit an error message"
    );
    assert!(
        stderr.contains("bad.rs"),
        "error message must mention the file that caused the failure; got: {stderr}"
    );
}

#[test]
fn ingest_recursive_skips_unsupported_extensions() {
    // `--recursive` uses SUPPORTED_EXTS to filter files.
    // Unknown extensions are skipped entirely (they do NOT fall back to Text in recursive mode).
    let dir = TempDir::new().unwrap();
    mnem(dir.path(), &["init"])
        .assert()
        .success();
    let src = dir.path().join("src");
    std::fs::create_dir(&src).unwrap();
    // Supported: .rs ingested as Code
    std::fs::write(src.join("lib.rs"), "pub fn foo() -> u32 { 42 }\npub fn bar() -> u32 { 1 }\n").unwrap();
    // Supported: .txt ingested as Text (txt is in SUPPORTED_EXTS)
    std::fs::write(src.join("notes.txt"), "project notes\n").unwrap();
    // Unsupported: must be SKIPPED, not Text-fallback
    std::fs::write(src.join("config.unknown_ext_xyz"), "ignored content\n").unwrap();

    let out = mnem(dir.path(), &["ingest", "--recursive", src.to_str().unwrap()])
        .assert()
        .success();
    let stdout = String::from_utf8_lossy(&out.get_output().stdout).to_string();

    // Must ingest exactly 2 files (lib.rs + notes.txt), not 3.
    assert!(
        stdout.contains("ingested 2 files"),
        "recursive walk must skip unsupported extensions; expected 'ingested 2 files', got: {stdout}"
    );

    // The two supported files (lib.rs with 2 fns, notes.txt with 1 sentence)
    // must together produce >= 3 chunks (2 structural from .rs + 1 text from .txt).
    let chunk_count = parse_count(&stdout, "chunks");
    assert!(
        chunk_count >= 3,
        "recursive ingest of lib.rs (2 fns → 2 chunks) + notes.txt (1 → 1 chunk) must total >= 3; got {chunk_count}"
    );
}

#[test]
fn ingest_recursive_no_supported_files_fails_with_error() {
    // A directory containing only unsupported extensions triggers the
    // collect_files bail! path ("no ingestable files found").
    let dir = TempDir::new().unwrap();
    mnem(dir.path(), &["init", dir.path().to_str().unwrap()])
        .assert()
        .success();
    let src = dir.path().join("only_unknown");
    std::fs::create_dir(&src).unwrap();
    std::fs::write(src.join("data.unknown_ext_xyz"), "some data\n").unwrap();
    std::fs::write(src.join("more.another_unknown"), "more data\n").unwrap();

    let out = mnem(dir.path(), &["ingest", "--recursive", src.to_str().unwrap()])
        .assert()
        .failure();
    let stderr = String::from_utf8_lossy(&out.get_output().stderr).to_string();
    assert!(
        stderr.contains("no ingestable files found"),
        "recursive walk with no supported files must fail with a clear error message; got: {stderr}"
    );
}
