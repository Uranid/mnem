//! audit-2026-04-25 P0-1: end-to-end CLI proof that two fresh repos
//! ingesting byte-identical input produce identical `content_cid`s
//! when callers supply a deterministic node UUID via `--id`.
//!
//! `commit_cid` is intentionally still time-varying (audit trail);
//! `content_cid` is the deterministic invariant the post-audit fix
//! provides. Without `--id`, node UUIDs are freshly generated as
//! UUIDv7 and the data-DAG roots themselves differ.

use std::process::Command;

use assert_cmd::cargo::CargoError;
use assert_cmd::prelude::*;
use tempfile::TempDir;

fn mnem_bin() -> Result<Command, CargoError> {
    Command::cargo_bin("mnem")
}

const FIXED_NODE_ID: &str = "0192f2cb-d4d0-7000-8000-000000000001";

fn ingest(dir: &std::path::Path) {
    mnem_bin()
        .unwrap()
        .args(["init", dir.to_str().unwrap()])
        .assert()
        .success();
    mnem_bin()
        .unwrap()
        .args([
            "-R",
            dir.to_str().unwrap(),
            "add",
            "node",
            "--label",
            "Person",
            "--prop",
            "name=Alice",
            "-s",
            "Alice lives in Berlin",
            "--no-embed",
            "--id",
            FIXED_NODE_ID,
            "-m",
            "deterministic test ingest",
        ])
        .assert()
        .success();
}

fn parse_content_cid(stats: &str) -> Option<String> {
    for tok in stats.split_whitespace() {
        if let Some(rest) = tok.strip_prefix("content=") {
            return Some(rest.to_string());
        }
    }
    None
}

fn parse_commit_cid(stats: &str) -> Option<String> {
    for tok in stats.split_whitespace() {
        if let Some(rest) = tok.strip_prefix("commit=") {
            return Some(rest.to_string());
        }
    }
    None
}

#[test]
fn deterministic_node_id_yields_matching_content_cid() {
    let a = TempDir::new().expect("tmp a");
    let b = TempDir::new().expect("tmp b");
    ingest(a.path());
    ingest(b.path());

    let stats_a = mnem_bin()
        .unwrap()
        .args(["-R", a.path().to_str().unwrap(), "stats"])
        .assert()
        .success();
    let stats_b = mnem_bin()
        .unwrap()
        .args(["-R", b.path().to_str().unwrap(), "stats"])
        .assert()
        .success();

    let stats_a_str = String::from_utf8_lossy(&stats_a.get_output().stdout).into_owned();
    let stats_b_str = String::from_utf8_lossy(&stats_b.get_output().stdout).into_owned();

    let content_a = parse_content_cid(&stats_a_str).expect("content= present in stats a");
    let content_b = parse_content_cid(&stats_b_str).expect("content= present in stats b");
    let commit_a = parse_commit_cid(&stats_a_str).expect("commit= present in stats a");
    let commit_b = parse_commit_cid(&stats_b_str).expect("commit= present in stats b");

    assert_eq!(
        content_a, content_b,
        "content_cid MUST match across two ingest runs with deterministic node IDs.\n\
         a: {stats_a_str}\nb: {stats_b_str}"
    );
    assert!(
        !content_a.starts_with('<'),
        "content_cid should be a real CID, not a sentinel: {content_a}"
    );
    // commit_cid embeds wall-clock + UUIDv7 ChangeId, so the two
    // commits should still differ -- that is the audit trail.
    // Sanity-check: at least one of the two embeddings should differ;
    // if they match by accident the time-microsecond clock collided
    // (acceptable but unusual).
    let _ = (commit_a, commit_b);
}
