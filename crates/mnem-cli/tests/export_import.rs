//! Integration tests for `mnem export` / `mnem import`.
//!
//! Uses `assert_cmd` to drive the built `mnem` binary against
//! temporary repos. Each test is end-to-end: init a repo, write some
//! content, export to a CAR, import into a second repo, confirm the
//! imported side sees the same blocks.

use std::path::Path;
use std::process::Command;
use std::sync::Arc;

use assert_cmd::prelude::*;
use bytes::Bytes;
use mnem_backend_redb::open_or_init;
use mnem_core::id::NodeId;
use mnem_core::objects::Node;
use mnem_core::objects::node::{Dtype, Embedding};
use mnem_core::repo::ReadonlyRepo;
use mnem_core::store::{Blockstore, OpHeadsStore};
use tempfile::TempDir;

/// Run `mnem <args>...` from inside `repo` as cwd, plus `-R <repo>`
/// so commands that honour the flag pick the right directory. Two
/// different mechanisms because `mnem init` takes an optional
/// positional path, while most other commands use `-R`.
fn mnem(repo: &Path, args: &[&str]) -> Command {
    let mut cmd = Command::cargo_bin("mnem").expect("built mnem binary");
    cmd.current_dir(repo);
    cmd.arg("-R").arg(repo);
    for a in args {
        cmd.arg(a);
    }
    cmd
}

#[test]
fn freshly_initialized_repo_exports_successfully() {
    // `mnem init` always seeds a Meta anchor node (seed_anchor_node),
    // so a freshly-initialised repo always has a head commit and
    // `mnem export` must succeed rather than returning an error.
    let dir = TempDir::new().unwrap();
    // `init` takes an explicit positional path (it ignores `-R`).
    mnem(dir.path(), &["init", dir.path().to_str().unwrap()])
        .assert()
        .success();

    let car = dir.path().join("out.car");
    let out = mnem(dir.path(), &["export", car.to_str().unwrap()])
        .assert()
        .success();
    let stdout = String::from_utf8_lossy(&out.get_output().stdout).to_string();
    assert!(
        stdout.contains("exported") && stdout.contains("blocks"),
        "expected export confirmation, got: {stdout}"
    );
}

#[test]
fn export_then_import_round_trip() {
    // Build repo A, add a node to give it a head commit, export the
    // head, import into a fresh repo B, and assert the import stats
    // line matches the export stats line (same block count + bytes).
    let src = TempDir::new().unwrap();
    mnem(src.path(), &["init", src.path().to_str().unwrap()])
        .assert()
        .success();
    mnem(
        src.path(),
        &[
            "add",
            "node",
            "--summary",
            "roundtrip payload",
            "--prop",
            "kind=doc",
        ],
    )
    .assert()
    .success();

    let car = src.path().join("snapshot.car");
    let export_out = mnem(
        src.path(),
        &["export", car.to_str().unwrap(), "--from", "HEAD"],
    )
    .assert()
    .success();
    let export_stdout = String::from_utf8_lossy(&export_out.get_output().stdout).to_string();
    assert!(
        export_stdout.starts_with("exported "),
        "export stdout: {export_stdout}"
    );
    assert!(car.exists(), "CAR file must be produced on disk");
    let car_size = std::fs::metadata(&car).unwrap().len();
    assert!(car_size > 0, "CAR must be non-empty");

    // Parse the block count out of the stdout line for cross-check.
    // Format: "exported N blocks, M bytes to <path>".
    let exported_n = export_stdout
        .split_whitespace()
        .nth(1)
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or_else(|| panic!("could not parse block count from: {export_stdout}"));
    assert!(
        exported_n >= 4,
        "a committed repo has commit+view+op+trees, got {exported_n}"
    );

    // Fresh destination.
    let dst = TempDir::new().unwrap();
    mnem(dst.path(), &["init", dst.path().to_str().unwrap()])
        .assert()
        .success();

    let import_out = mnem(dst.path(), &["import", car.to_str().unwrap()])
        .assert()
        .success();
    let import_stdout = String::from_utf8_lossy(&import_out.get_output().stdout).to_string();
    assert!(
        import_stdout.starts_with("imported "),
        "import stdout: {import_stdout}"
    );
    let imported_n = import_stdout
        .split_whitespace()
        .nth(1)
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or_else(|| panic!("could not parse block count from: {import_stdout}"));
    assert_eq!(
        exported_n, imported_n,
        "block counts must match across export / import"
    );
}

/// Prove that embedding sidecar blocks are included in the CAR file and
/// survive a full export / import round-trip.
///
/// The test seeds a source repo with a node + embedding directly via the
/// Rust API (no external embedder needed), exports to a CAR, imports into a
/// fresh repo, then re-opens the destination repo via the Rust API and
/// asserts that `embedding_for` returns the original vector. This is the
/// critical unverified invariant from the G16 audit: a CAR that silently
/// drops sidecar blocks would let an import succeed with zero embedding
/// vectors.
#[test]
fn sidecar_blocks_round_trip_through_car() {
    // ---- source repo: init via CLI ----
    let src = TempDir::new().unwrap();
    mnem(src.path(), &["init", src.path().to_str().unwrap()])
        .assert()
        .success();

    // ---- seed a node + embedding via Rust API ----
    let model = "test:sidecar-model";
    let dim: u32 = 4;

    // Build a small f32 embedding vector.
    let vector: Vec<f32> = (0..dim).map(|i| (i + 1) as f32 / dim as f32).collect();
    let mut vector_bytes = Vec::with_capacity(dim as usize * 4);
    for v in &vector {
        vector_bytes.extend_from_slice(&v.to_le_bytes());
    }
    let emb = Embedding {
        model: model.to_string(),
        dtype: Dtype::F32,
        dim,
        vector: Bytes::from(vector_bytes),
    };

    let node_cid = {
        let db_path = src.path().join(".mnem").join("repo.redb");
        let (bs, ohs, _db) = open_or_init(&db_path).expect("open src redb");
        let bs_arc: Arc<dyn Blockstore> = bs;
        let ohs_arc: Arc<dyn OpHeadsStore> = ohs;
        let repo = ReadonlyRepo::open(bs_arc, ohs_arc).expect("open src repo");

        let node = Node::new(NodeId::new_v7(), "TestDoc").with_summary("sidecar round-trip test");
        let mut tx = repo.start_transaction();
        let cid = tx.add_node(&node).expect("add node");
        tx.set_embedding(cid.clone(), model.to_string(), emb.clone())
            .expect("set embedding");
        tx.commit("test-author", "add node with sidecar embedding")
            .expect("commit");
        cid
    };

    // ---- export src to CAR ----
    let car = src.path().join("sidecar.car");
    mnem(
        src.path(),
        &["export", car.to_str().unwrap(), "--from", "HEAD"],
    )
    .assert()
    .success();
    assert!(car.exists(), "CAR file must exist after export");
    assert!(
        std::fs::metadata(&car).unwrap().len() > 0,
        "CAR must be non-empty"
    );

    // ---- destination repo: import into a fresh directory (no prior init).
    //
    // We deliberately skip `mnem init` here. The import command auto-creates
    // `.mnem/` and calls `ReadonlyRepo::init` internally. If we ran `mnem init`
    // first, we would have TWO disconnected op-heads (the init one + the
    // imported one), causing `ReadonlyRepo::open` to fail with NoCommonAncestor
    // when we try to verify the embedding after import.
    let dst = TempDir::new().unwrap();
    // Create the parent directory; `mnem import` will create `.mnem/` itself.
    std::fs::create_dir_all(dst.path()).unwrap();
    mnem(dst.path(), &["import", car.to_str().unwrap()])
        .assert()
        .success();

    // ---- verify embedding survived the round-trip via Rust API ----
    // After import, the op-heads store has exactly one head. We use
    // `load_at` with the single op-head to avoid the multi-head merge path.
    let db_path = dst.path().join(".mnem").join("repo.redb");
    let (bs, ohs, _db) = open_or_init(&db_path).expect("open dst redb");
    let bs_arc: Arc<dyn Blockstore> = bs;
    let ohs_arc: Arc<dyn OpHeadsStore> = ohs;
    // Take the latest (and only) op-head directly so we don't trigger the
    // multi-head merge path (which requires a common ancestor).
    let heads = ohs_arc.current().expect("get op heads");
    assert!(
        !heads.is_empty(),
        "dst op-heads must be non-empty after import"
    );
    let latest_op = heads.into_iter().last().unwrap();
    let repo = ReadonlyRepo::load_at(bs_arc, ohs_arc, latest_op).expect("load dst repo");

    let got = repo
        .embedding_for(&node_cid, model)
        .expect("embedding_for must not error");

    assert!(
        got.is_some(),
        "embedding sidecar block must be present in the destination repo after CAR import; \
         got None -- sidecar blocks are likely missing from the CAR file"
    );

    let got_emb = got.unwrap();
    assert_eq!(got_emb.model, model, "model string must match");
    assert_eq!(got_emb.dim, dim, "dim must match");
    assert_eq!(got_emb.dtype, Dtype::F32, "dtype must match");
    assert_eq!(
        got_emb.vector, emb.vector,
        "vector bytes must be byte-identical after round-trip"
    );
}
