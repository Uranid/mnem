//! Internal workspace task runner for mnem.
//!
//! Exposes the `lint-warnings` subcommand used by CI to validate the
//! gap-14 warnings catalog: every [`WarningCode`] variant must have a
//! non-empty compile-time-constant message body and a
//! remediation-ref markdown file under `docs/warnings/`.
//!
//! Run with: `cargo run -p xtask -- lint-warnings`

#![deny(missing_docs)]

use std::path::PathBuf;
use std::process::ExitCode;

use mnem_core::retrieve::WarningCode;

/// Entry point. Dispatches the single supported subcommand.
fn main() -> ExitCode {
    let mut args = std::env::args().skip(1);
    let Some(sub) = args.next() else {
        eprintln!("usage: xtask <subcommand>\n\nsubcommands:\n  lint-warnings");
        return ExitCode::from(2);
    };
    match sub.as_str() {
        "lint-warnings" => lint_warnings(),
        other => {
            eprintln!("unknown subcommand: {other}\n\ntry: xtask lint-warnings");
            ExitCode::from(2)
        }
    }
}

/// Walk every [`WarningCode`] variant and assert:
///
/// 1. [`WarningCode::message`] returns a non-empty string (the
///    `include_str!` target file exists and has content).
/// 2. The trimmed message contains no ANSI escape sequences or null
///    bytes (guards against an accidental binary file slipping into
///    the catalog).
/// 3. [`WarningCode::remediation_ref`] points at an existing markdown
///    file under the repo root, and that file contains an
///    `## Agent fallback` section (per the R4 meta-LD contract).
/// 4. The wire name ([`WarningCode::as_str`]) matches the
///    `snake_case` pattern `[a-z][a-z0-9_]*` so downstream
///    dashboards / tag indices can key on it without quoting.
///
/// A single failing variant prints its diagnostic and the overall
/// exit code is non-zero; the run continues so operators see ALL
/// broken variants in one pass.
fn lint_warnings() -> ExitCode {
    let repo_root = match locate_repo_root() {
        Ok(p) => p,
        Err(e) => {
            eprintln!("could not locate repo root: {e}");
            return ExitCode::from(1);
        }
    };

    let mut failures: u32 = 0;
    for code in WarningCode::all() {
        let name = code.as_str();

        // (1) compile-time message is non-empty.
        let msg = code.message();
        if msg.trim().is_empty() {
            eprintln!("[FAIL] {name}: message is empty");
            failures += 1;
        }

        // (2) no control bytes.
        if msg.contains('\0') || msg.contains('\u{001b}') {
            eprintln!("[FAIL] {name}: message contains control bytes");
            failures += 1;
        }

        // (3) remediation-ref markdown exists and names the fallback.
        let r = code.remediation_ref();
        let has_md_ext = std::path::Path::new(r)
            .extension()
            .is_some_and(|ext| ext.eq_ignore_ascii_case("md"));
        if !(r.starts_with("docs/warnings/") && has_md_ext) {
            eprintln!("[FAIL] {name}: remediation_ref shape: {r}");
            failures += 1;
            continue;
        }
        let full = repo_root.join(r);
        match std::fs::read_to_string(&full) {
            Ok(contents) => {
                if !contents.contains("## Agent fallback") {
                    eprintln!("[FAIL] {name}: {r} missing `## Agent fallback` section");
                    failures += 1;
                }
            }
            Err(e) => {
                eprintln!("[FAIL] {name}: cannot read {}: {e}", full.display());
                failures += 1;
            }
        }

        // (4) snake_case wire name.
        if !is_snake_case(name) {
            eprintln!("[FAIL] {name}: wire name is not snake_case");
            failures += 1;
        }
    }

    if failures == 0 {
        println!("lint-warnings: OK ({} variants)", WarningCode::all().len());
        ExitCode::SUCCESS
    } else {
        eprintln!("lint-warnings: {failures} failure(s)");
        ExitCode::from(1)
    }
}

/// Walk up from `CARGO_MANIFEST_DIR` (or CWD) until a `docs/`
/// directory is found next to a workspace `Cargo.toml`. Returns the
/// workspace-root path so remediation refs resolve against it.
fn locate_repo_root() -> Result<PathBuf, String> {
    let start = std::env::var("CARGO_MANIFEST_DIR")
        .map(PathBuf::from)
        .or_else(|_| std::env::current_dir().map_err(|e| e.to_string()))?;
    let mut cur: &std::path::Path = &start;
    loop {
        let docs = cur.join("docs");
        let manifest = cur.join("Cargo.toml");
        if docs.is_dir() && manifest.is_file() {
            return Ok(cur.to_path_buf());
        }
        match cur.parent() {
            Some(p) => cur = p,
            None => return Err("reached filesystem root without finding docs/".into()),
        }
    }
}

/// Minimal snake_case predicate: `^[a-z][a-z0-9_]*$` without pulling
/// in a regex dep. Returns `true` when every char is in the allowed
/// set and the first char is a lowercase letter.
fn is_snake_case(s: &str) -> bool {
    let mut chars = s.chars();
    match chars.next() {
        Some(c) if c.is_ascii_lowercase() => {}
        _ => return false,
    }
    chars.all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_')
}
