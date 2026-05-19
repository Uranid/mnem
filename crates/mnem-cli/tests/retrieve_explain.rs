//! CLI integration tests for `mnem retrieve --explain` per-lane score output.
//!
//! Verifies that `--explain` prints a `lanes: key=value ...` line for each
//! retrieved item when at least one retrieval lane contributed a score. When
//! no ranker is active (filter-only / `--no-vector` path) or `--explain` is
//! absent, no `lanes:` line appears.
//!
//! # Isolation strategy
//!
//! All tests disable the global config, remove embedder env vars, and route
//! mnem to a temp directory via `-R`. The C7-3b mock-embedder fallback is
//! enabled by default (MNEM_DISABLE_MOCK_FALLBACK not set) so a vector lane
//! score is available without a real embedder. Tests that want the filter-only
//! path use `--no-vector` to explicitly opt out of vector ranking.
//!
//! # Infrastructure constraints
//!
//! The following lanes cannot be exercised in isolated tests without external
//! infrastructure:
//! - `sparse`: requires a configured sparse provider (BM25 index).
//! - `rerank`: requires an external reranker API (Cohere, Voyage, Jina).
//! - Multi-query RRF fusion path (the "(multi-query: per-lane scores not
//!   propagated through RRF fusion; ...)" note that fires when `--explain` is
//!   used on a successful multi-query run): requires a live LLM provider.
//!
//! The `--multi-query --explain` combination *without* LLM config is testable
//! and is covered by `explain_with_multi_query_no_llm_falls_back`.
//!
//! # Multi-lane-per-item
//!
//! In the current retriever, `graph_expand` only processes neighbors that are
//! NOT already in the vector `prefetched` set (`seen`). This means a single
//! node can only accumulate lanes from mutually-exclusive retrieval paths
//! (vector OR graph_expand, never both simultaneously), unless a sparse ranker
//! or reranker is active. The `lanes.join(" ")` multi-lane formatting path is
//! therefore not exercised in isolation tests; it would require an additional
//! configured lane (sparse or rerank).

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

/// Initialise a temp dir as a mnem repo and commit a single node.
/// Returns (TempDir, uuid).
fn setup_repo_with_node(summary: &str) -> (TempDir, String) {
    let dir = TempDir::new().expect("tempdir");
    mnem(dir.path(), &["init"]).assert().success();
    let out = mnem(dir.path(), &["add", "node", "-s", summary, "--label", "Fact"])
        .output()
        .expect("add node");
    let stdout = String::from_utf8_lossy(&out.stdout).to_string();
    let uuid = stdout
        .lines()
        .find_map(|l| l.strip_prefix("added node "))
        .expect("'added node <uuid>' line missing")
        .trim()
        .to_string();
    (dir, uuid)
}

/// Parse the UUID from `mnem add node` stdout ("added node <uuid>").
fn parse_uuid(output: &std::process::Output) -> String {
    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    stdout
        .lines()
        .find_map(|l| l.strip_prefix("added node "))
        .expect("'added node <uuid>' line missing")
        .trim()
        .to_string()
}

/// `mnem retrieve --explain "query"` must print `lanes: vector=<score>` for
/// each retrieved item when the mock-embedder fallback is active (default).
#[test]
fn explain_prints_lane_scores_with_mock_vector() {
    let (dir, _) = setup_repo_with_node("Alice works at Acme Corp as an engineer");
    let output = mnem(dir.path(), &["retrieve", "--explain", "Alice"])
        .env_remove("MNEM_DISABLE_MOCK_FALLBACK")
        .output()
        .expect("retrieve ran");

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("item(s)"),
        "expected retrieve header in stdout, got: {stdout}"
    );
    assert!(
        stdout.contains("lanes: vector="),
        "`--explain` must print 'lanes: vector=<score>' per item; stdout:\n{stdout}"
    );
}

/// `mnem retrieve "query"` without `--explain` must NOT print any `lanes:`
/// line, keeping stdout clean for scripts.
#[test]
fn no_explain_omits_lane_scores() {
    let (dir, _) = setup_repo_with_node("Bob manages the Berlin office");
    let output = mnem(dir.path(), &["retrieve", "Bob"])
        .env_remove("MNEM_DISABLE_MOCK_FALLBACK")
        .output()
        .expect("retrieve ran");

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        !stdout.contains("lanes:"),
        "stdout must not contain 'lanes:' without --explain; stdout:\n{stdout}"
    );
}

/// `mnem retrieve --explain --no-vector "query"` must NOT print a `lanes:`
/// line: `--no-vector` explicitly disables the vector ranker so `lane_scores`
/// is empty (filter-only path).
#[test]
fn explain_no_lanes_without_vector_ranker() {
    let (dir, _) = setup_repo_with_node("Carol leads the product team");
    let output = mnem(dir.path(), &["retrieve", "--explain", "--no-vector", "Carol"])
        .output()
        .expect("retrieve ran");

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        !stdout.contains("lanes:"),
        "filter-only (--no-vector) retrieve must not produce a 'lanes:' line; stdout:\n{stdout}"
    );
}

/// The `lanes:` line must appear BEFORE the rendered content of the item.
#[test]
fn explain_lane_line_precedes_rendered_content() {
    let (dir, _) = setup_repo_with_node("Diana is a senior SRE at NovaCo");
    let output = mnem(dir.path(), &["retrieve", "--explain", "Diana"])
        .env_remove("MNEM_DISABLE_MOCK_FALLBACK")
        .output()
        .expect("retrieve ran");

    let stdout = String::from_utf8_lossy(&output.stdout);

    // Both must be present - assert first so a missing lanes line is a clear failure.
    let lanes_pos = stdout
        .find("lanes: vector=")
        .unwrap_or_else(|| panic!("expected 'lanes: vector=' in stdout:\n{stdout}"));
    // Anchor on `summary:` which is always present in rendered output.
    let summary_pos = stdout
        .find("summary: Diana")
        .unwrap_or_else(|| panic!("expected 'summary: Diana' in rendered content; stdout:\n{stdout}"));

    assert!(
        lanes_pos < summary_pos,
        "lanes line (pos {lanes_pos}) must precede rendered content (pos {summary_pos}); stdout:\n{stdout}"
    );
}

/// Lane scores are formatted as `name=<4-decimal float>`. Each segment must
/// be parseable as a valid finite f32.
#[test]
fn explain_lane_score_format_is_valid() {
    let (dir, _) = setup_repo_with_node("Eve is the head of data science");
    let output = mnem(dir.path(), &["retrieve", "--explain", "Eve"])
        .env_remove("MNEM_DISABLE_MOCK_FALLBACK")
        .output()
        .expect("retrieve ran");

    let stdout = String::from_utf8_lossy(&output.stdout);
    let mut found_lanes_line = false;
    for line in stdout.lines() {
        let trimmed = line.trim();
        if let Some(rest) = trimmed.strip_prefix("lanes: ") {
            found_lanes_line = true;
            for segment in rest.split_whitespace() {
                let mut parts = segment.splitn(2, '=');
                let name = parts.next().expect("lane name");
                let score_str = parts.next().unwrap_or_else(|| {
                    panic!("expected '=' in lane segment '{segment}' in: {stdout}")
                });
                assert!(!name.is_empty(), "lane name must not be empty in: {stdout}");
                let score: f32 = score_str.parse().unwrap_or_else(|_| {
                    panic!("lane score '{score_str}' is not a valid f32 in: {stdout}")
                });
                assert!(score.is_finite(), "lane score must be finite, got {score} in: {stdout}");
                // Verify exactly 4 decimal places (format!("{:.4}", score)).
                let decimal_part = score_str.split('.').nth(1).unwrap_or_else(|| {
                    panic!("lane score '{score_str}' has no decimal point in: {stdout}")
                });
                assert_eq!(
                    decimal_part.len(),
                    4,
                    "lane score '{score_str}' must have exactly 4 decimal digits; got {}"
                    , decimal_part.len()
                );
            }
        }
    }
    assert!(found_lanes_line, "no 'lanes:' line found at all in stdout:\n{stdout}");
}

/// When multiple items are returned, every item must have its own `lanes:` line.
/// Commits two nodes so there are two candidates; both must show lane scores.
#[test]
fn explain_all_items_get_lane_scores() {
    let dir = TempDir::new().expect("tempdir");
    mnem(dir.path(), &["init"]).assert().success();
    mnem(dir.path(), &["add", "node", "-s", "Frank is a backend engineer at CorpA", "--label", "Fact"])
        .assert().success();
    mnem(dir.path(), &["add", "node", "-s", "Grace is a frontend engineer at CorpB", "--label", "Fact"])
        .assert().success();

    let output = mnem(dir.path(), &["retrieve", "--explain", "engineer"])
        .env_remove("MNEM_DISABLE_MOCK_FALLBACK")
        .output()
        .expect("retrieve ran");

    let stdout = String::from_utf8_lossy(&output.stdout);

    // Count items returned.
    let item_count = stdout.lines().filter(|l| l.starts_with("---")).count();
    assert!(item_count >= 2, "expected at least 2 items; stdout:\n{stdout}");

    // Count lanes lines - must match item count.
    let lanes_count = stdout.lines().filter(|l| l.trim().starts_with("lanes:")).count();
    assert_eq!(
        lanes_count, item_count,
        "every item must have a 'lanes:' line; found {lanes_count} lanes lines for {item_count} items; stdout:\n{stdout}"
    );
}

/// A query that matches nothing must produce a clean header with no `lanes:` lines.
#[test]
fn explain_empty_result_no_lanes_line() {
    let (dir, _) = setup_repo_with_node("Hank manages the warehouse");
    // Use --no-vector so the retrieve is filter-only and returns nothing
    // (there are no nodes matching the prop filter).
    let output = mnem(
        dir.path(),
        &["retrieve", "--explain", "--no-vector", "--where", "name=Nonexistent"],
    )
    .output()
    .expect("retrieve ran");

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("0 item(s)"),
        "expected '0 item(s)' in header; stdout:\n{stdout}"
    );
    assert!(
        !stdout.contains("lanes:"),
        "zero-result retrieve must not produce any 'lanes:' line; stdout:\n{stdout}"
    );
}

/// `--explain --graph-expand` must produce a `graph_expand=` lane score for
/// nodes that were reached via edge traversal. Sets up two nodes connected by
/// an edge; the seed's neighbor gets a `GraphExpand` entry.
#[test]
fn explain_graph_expand_shows_graph_expand_lane() {
    let dir = TempDir::new().expect("tempdir");
    mnem(dir.path(), &["init"]).assert().success();

    // Node A: seed (matched by vector search).
    let out_a = mnem(dir.path(), &["add", "node", "-s", "Ivy is the lead architect", "--label", "Fact"])
        .output()
        .expect("add node A");
    let uuid_a = parse_uuid(&out_a);

    // Node B: neighbor (reached via graph expand from A).
    let out_b = mnem(dir.path(), &["add", "node", "-s", "Jake reports to the lead architect", "--label", "Fact"])
        .output()
        .expect("add node B");
    let uuid_b = parse_uuid(&out_b);

    // Add a directed edge A → B.
    mnem(dir.path(), &["add", "edge", "--from", &uuid_a, "--to", &uuid_b, "--label", "manages"])
        .assert()
        .success();

    // Retrieve with graph expand; mock vector fires as the seed ranker.
    // --vector-cap 1 forces only the top-1 ANN result (Ivy) into prefetched so
    // Jake is NOT in the initial seen set and gets discovered via graph expansion.
    let output = mnem(dir.path(), &["retrieve", "--explain", "--graph-expand", "20", "--vector-cap", "1", "architect"])
        .env_remove("MNEM_DISABLE_MOCK_FALLBACK")
        .output()
        .expect("retrieve ran");

    let stdout = String::from_utf8_lossy(&output.stdout);

    // Split by item separator so we can inspect Jake's block independently.
    // Each item starts with "---\n"; splitting on that gives non-item prefix
    // in slot 0 and one block per item in subsequent slots.
    let blocks: Vec<&str> = stdout.split("---\n").collect();
    let jake_block = blocks
        .iter()
        .find(|b| b.contains("Jake reports to the lead architect"))
        .unwrap_or_else(|| panic!("no output block found for Jake; stdout:\n{stdout}"));

    // Jake was NOT in the initial ANN set (--vector-cap 1 capped seeds to Ivy
    // only) and must have been discovered via graph expansion from Ivy. His
    // lanes line must therefore contain graph_expand= rather than vector=.
    assert!(
        jake_block.contains("graph_expand="),
        "Jake's output block must have 'graph_expand=' lane score; block:\n{jake_block}\nfull stdout:\n{stdout}"
    );
    assert!(
        !jake_block.contains("vector="),
        "Jake's block must NOT have 'vector=' (he was not in the ANN set, only graph-expanded); \
         block:\n{jake_block}\nfull stdout:\n{stdout}"
    );
}

/// `--multi-query --explain` without a configured LLM must:
/// 1. Print a warning on stderr about the missing LLM provider.
/// 2. Fall back to plain retrieve (which still produces `lanes: vector=` with
///    --explain, since the mock-embedder vector lane is active).
/// 3. NOT print the RRF-fusion note ("per-lane scores not propagated...") -
///    that note only fires when the multi-query path actually ran with an LLM.
///
/// The actual RRF fusion note is not tested here because it requires a live
/// LLM provider. See the module-level doc comment for the constraint rationale.
#[test]
fn explain_with_multi_query_no_llm_falls_back() {
    let (dir, _) = setup_repo_with_node("Ivan is a principal engineer at MegaCorp");
    // Provide N=4 explicitly after --multi-query to avoid clap trying to parse
    // --explain or the query as the optional N value.
    let output = mnem(dir.path(), &["retrieve", "--multi-query", "4", "--explain", "Ivan"])
        .env_remove("MNEM_DISABLE_MOCK_FALLBACK")
        .output()
        .expect("retrieve ran");

    let stderr = String::from_utf8_lossy(&output.stderr);
    let stdout = String::from_utf8_lossy(&output.stdout);

    assert!(
        stderr.contains("--multi-query requires an LLM provider"),
        "without LLM config, --multi-query must warn on stderr; stderr:\n{stderr}"
    );
    assert!(
        stdout.contains("lanes: vector="),
        "fallback plain retrieve must still show lanes: with --explain; stdout:\n{stdout}"
    );
    assert!(
        !stderr.contains("per-lane scores not propagated"),
        "RRF fusion note must not appear when multi-query fell back (did not run); stderr:\n{stderr}"
    );
}
