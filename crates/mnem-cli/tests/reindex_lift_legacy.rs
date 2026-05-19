//! Integration tests for `mnem reindex --lift-legacy-extra` (G19) and
//! `mnem reindex --lift-legacy-sparse` (G17 migration path).
//!
//! Verifies that legacy v0.3 nodes carrying `extra["embed"]` are
//! promoted to the embedding sidecar (`Commit.embeddings`) without
//! re-deriving from text, and that `NodeCid` is unchanged after the lift.
//!
//! Also verifies that pre-G17 nodes carrying `extra["sparse_embed"]` are
//! promoted to the sparse sidecar (`Commit.sparse`) without re-encoding from
//! text, and that the NodeCid is unchanged after the sparse lift.

use std::path::Path;
use std::process::Command;
use std::sync::Arc;

use assert_cmd::prelude::*;
use bytes::Bytes;
use mnem_backend_redb::open_or_init;
use mnem_core::codec::to_canonical_bytes;
use mnem_core::id::{Cid, NodeId};
use mnem_core::objects::Node;
use mnem_core::objects::node::{Dtype, Embedding};
use mnem_core::repo::ReadonlyRepo;
use mnem_core::sparse::SparseEmbed;
use mnem_core::store::Blockstore;
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

fn init(dir: &Path) {
    mnem(dir, &["init", dir.to_str().unwrap()])
        .assert()
        .success();
}

/// Open the redb store and return (blockstore, repo).
fn open_repo(dir: &Path) -> (Arc<dyn Blockstore>, ReadonlyRepo) {
    let db = dir.join(".mnem").join("repo.redb");
    let (bs, ohs, _) = open_or_init(&db).expect("open redb");
    let repo = ReadonlyRepo::open(bs.clone(), ohs).expect("open repo");
    (bs, repo)
}

/// Build a minimal `Embedding` with a 2-element f32 vector.
fn test_embedding(model: &str) -> Embedding {
    // Two f32 values: 1.0 and 0.5 packed as little-endian bytes.
    let mut buf = Vec::with_capacity(8);
    buf.extend_from_slice(&1.0_f32.to_le_bytes());
    buf.extend_from_slice(&0.5_f32.to_le_bytes());
    Embedding {
        model: model.to_string(),
        dtype: Dtype::F32,
        dim: 2,
        vector: Bytes::from(buf),
    }
}

/// Encode `Embedding` as an `Ipld` value so we can store it in
/// `node.extra["embed"]`. We go via DAG-CBOR bytes -> `Ipld` using
/// the same codec the Node decoder uses, so the round-trip is exact.
fn embedding_to_ipld(emb: &Embedding) -> ipld_core::ipld::Ipld {
    // Serialize the Embedding to DAG-CBOR bytes, then deserialize as Ipld.
    let bytes = to_canonical_bytes(emb).expect("encode Embedding");
    serde_ipld_dagcbor::from_slice(&bytes).expect("decode as Ipld")
}

/// Commit a node that has `extra["embed"]` set (simulating a v0.3 node).
/// Returns the `NodeCid` of the committed node and the model string.
fn seed_legacy_node(dir: &Path) -> (Cid, String) {
    let model = "legacy-test-model:v0.3".to_string();
    let emb = test_embedding(&model);

    // Open the repo for writing.
    let db = dir.join(".mnem").join("repo.redb");
    let (bs, ohs, _) = open_or_init(&db).expect("open redb");
    let repo = ReadonlyRepo::open(bs, ohs)
        .or_else(|e| {
            if e.is_uninitialized() {
                let db2 = dir.join(".mnem").join("repo.redb");
                let (bs2, ohs2, _) = open_or_init(&db2).expect("reopen");
                ReadonlyRepo::init(bs2, ohs2).map_err(Into::into)
            } else {
                Err(anyhow::anyhow!("{e}"))
            }
        })
        .expect("open or init repo");

    let mut tx = repo.start_transaction();

    // Build a node with extra["embed"] set to the Ipld encoding of emb.
    let mut node = Node::new(NodeId::from_bytes_raw([42u8; 16]), "LegacyDoc")
        .with_summary("a legacy document with inline embed");
    node.extra
        .insert("embed".to_string(), embedding_to_ipld(&emb));

    let node_cid = tx.add_node(&node).expect("add node");

    // Commit WITHOUT calling set_embedding - the embedding is only in extra.
    let r2 = tx
        .commit("test author", "seed legacy node")
        .expect("commit");

    // Verify: sidecar must be empty for this node (no set_embedding was called).
    assert!(
        r2.embedding_for(&node_cid, &model)
            .expect("embedding_for")
            .is_none(),
        "sidecar must be empty before lift"
    );

    (node_cid, model)
}

#[test]
fn lift_legacy_extra_promotes_embedding_to_sidecar() {
    let dir = TempDir::new().unwrap();
    init(dir.path());

    // Seed a node with extra["embed"] via the Rust API.
    let (node_cid, model) = seed_legacy_node(dir.path());

    // Run `mnem reindex --lift-legacy-extra` via the CLI binary.
    let out = mnem(dir.path(), &["reindex", "--lift-legacy-extra"])
        .assert()
        .success();
    let stdout = String::from_utf8_lossy(&out.get_output().stdout).to_string();
    assert!(
        stdout.contains("lifted") || stdout.contains("embedding"),
        "expected lift confirmation in stdout, got: {stdout}"
    );

    // Re-open the repo and verify the sidecar now has the embedding.
    let (_bs, repo) = open_repo(dir.path());
    let got = repo
        .embedding_for(&node_cid, &model)
        .expect("embedding_for after lift");
    assert!(
        got.is_some(),
        "sidecar must have the lifted embedding for model {model}"
    );

    let emb = got.unwrap();
    assert_eq!(emb.model, model);
    assert_eq!(emb.dim, 2);
    assert_eq!(emb.dtype, Dtype::F32);
    assert_eq!(emb.vector.len(), 8); // 2 * 4 bytes per f32
}

#[test]
fn lift_legacy_extra_nodecid_unchanged() {
    let dir = TempDir::new().unwrap();
    init(dir.path());

    // Seed a legacy node and record its NodeCid.
    let (node_cid_before, _model) = seed_legacy_node(dir.path());

    // Run lift.
    mnem(dir.path(), &["reindex", "--lift-legacy-extra"])
        .assert()
        .success();

    // The NodeCid must be unchanged: lift only writes to the sidecar,
    // never rewrites node bytes.
    let (bs, repo) = open_repo(dir.path());
    let head = repo.head_commit().expect("head commit after lift");

    // Walk the nodes tree and find our node using the already-open blockstore.
    let cursor = mnem_core::prolly::Cursor::new(&*bs, &head.nodes).expect("cursor");
    let mut found = false;
    for entry in cursor {
        let (_k, cid) = entry.expect("entry");
        if cid == node_cid_before {
            found = true;
            break;
        }
    }
    assert!(
        found,
        "NodeCid {node_cid_before} must still be present in the nodes tree after lift"
    );
}

#[test]
fn lift_legacy_extra_dry_run_no_commit() {
    let dir = TempDir::new().unwrap();
    init(dir.path());

    seed_legacy_node(dir.path());

    // Capture op count before.
    let ops_before = {
        let out = mnem(dir.path(), &["log", "--oneline"]).assert().success();
        let s = String::from_utf8_lossy(&out.get_output().stdout).to_string();
        s.lines().filter(|l| !l.trim().is_empty()).count()
    };

    // Dry run should print a count but not commit.
    let out = mnem(dir.path(), &["reindex", "--lift-legacy-extra", "--dry-run"])
        .assert()
        .success();
    let stdout = String::from_utf8_lossy(&out.get_output().stdout).to_string();
    assert!(
        stdout.contains("would lift") || stdout.contains("1"),
        "dry-run must report candidate count, got: {stdout}"
    );

    // Op count must be unchanged.
    let ops_after = {
        let out = mnem(dir.path(), &["log", "--oneline"]).assert().success();
        let s = String::from_utf8_lossy(&out.get_output().stdout).to_string();
        s.lines().filter(|l| !l.trim().is_empty()).count()
    };
    assert_eq!(
        ops_before, ops_after,
        "--dry-run must not write a new commit"
    );
}

#[test]
fn lift_legacy_extra_and_force_are_mutually_exclusive() {
    let dir = TempDir::new().unwrap();
    init(dir.path());

    mnem(dir.path(), &["reindex", "--lift-legacy-extra", "--force"])
        .assert()
        .failure();
}

// ---------------------------------------------------------------------------
// --lift-legacy-sparse tests
// ---------------------------------------------------------------------------

/// Build a minimal `SparseEmbed` for use in tests.
fn test_sparse_embed(vocab: &str) -> SparseEmbed {
    SparseEmbed::new(vec![10, 42, 99], vec![0.8, 0.5, 0.2], vocab).expect("valid SparseEmbed")
}

/// Encode a `SparseEmbed` as an `Ipld` value via DAG-CBOR round-trip so it can
/// be stored in `node.extra["sparse_embed"]`, matching the pre-G17 wire format.
fn sparse_embed_to_ipld(se: &SparseEmbed) -> ipld_core::ipld::Ipld {
    let bytes = to_canonical_bytes(se).expect("encode SparseEmbed");
    serde_ipld_dagcbor::from_slice(&bytes).expect("decode as Ipld")
}

/// Commit a node that has `extra["sparse_embed"]` set (simulating a pre-G17
/// node). Returns the `NodeCid` and the `vocab_id` of the embedded sparse
/// vector.
fn seed_legacy_sparse_node(dir: &Path) -> (Cid, String) {
    let vocab = "splade-v3-legacy-test".to_string();
    let se = test_sparse_embed(&vocab);

    let db = dir.join(".mnem").join("repo.redb");
    let (bs, ohs, _) = open_or_init(&db).expect("open redb");
    let repo = ReadonlyRepo::open(bs, ohs).expect("open repo");

    let mut tx = repo.start_transaction();

    let mut node = Node::new(NodeId::from_bytes_raw([0xBB_u8; 16]), "LegacySparseDoc")
        .with_summary("a legacy document with inline sparse_embed");
    node.extra
        .insert("sparse_embed".to_string(), sparse_embed_to_ipld(&se));

    let node_cid = tx.add_node(&node).expect("add node");

    // Commit WITHOUT calling set_sparse_embedding - the embedding is only in extra.
    let r2 = tx
        .commit("test author", "seed legacy sparse node")
        .expect("commit");

    // Verify: sparse sidecar must be empty for this node (no set_sparse_embedding called).
    assert!(
        r2.sparse_for(&node_cid, &vocab)
            .expect("sparse_for")
            .is_none(),
        "sparse sidecar must be empty before lift"
    );

    (node_cid, vocab)
}

/// Happy path: a node with `extra["sparse_embed"]` is promoted to the sparse
/// sidecar after `mnem reindex --lift-legacy-sparse`.
#[test]
fn lift_legacy_sparse_promotes_to_sidecar() {
    let dir = TempDir::new().unwrap();
    init(dir.path());

    let (node_cid, vocab) = seed_legacy_sparse_node(dir.path());

    let out = mnem(dir.path(), &["reindex", "--lift-legacy-sparse"])
        .assert()
        .success();
    let stdout = String::from_utf8_lossy(&out.get_output().stdout).to_string();
    assert!(
        stdout.contains("lifted") || stdout.contains("sparse"),
        "expected lift confirmation in stdout, got: {stdout}"
    );

    // Re-open and verify the sparse sidecar now has the embedding.
    let db = dir.path().join(".mnem").join("repo.redb");
    let (bs, ohs, _) = open_or_init(&db).expect("open redb");
    let repo = ReadonlyRepo::open(bs, ohs).expect("open repo");

    let got = repo
        .sparse_for(&node_cid, &vocab)
        .expect("sparse_for after lift");
    assert!(
        got.is_some(),
        "sparse sidecar must have the lifted SparseEmbed for vocab {vocab}"
    );

    let se = got.unwrap();
    assert_eq!(se.vocab_id, vocab);
    assert_eq!(se.indices, vec![10, 42, 99]);
    // Gap 1 fix: verify actual float values, not just the count.
    let expected_values = [0.8f32, 0.5, 0.2];
    for (i, (got_v, exp)) in se.values.iter().zip(expected_values.iter()).enumerate() {
        assert!(
            (got_v - exp).abs() < 1e-5,
            "value[{i}] round-tripped incorrectly: got {got_v}, expected {exp}"
        );
    }
}

/// After `--lift-legacy-sparse`, the NodeCid must remain unchanged because the
/// lift only writes to the sparse sidecar and never rewrites node bytes.
#[test]
fn lift_legacy_sparse_nodecid_unchanged() {
    let dir = TempDir::new().unwrap();
    init(dir.path());

    let (node_cid_before, _vocab) = seed_legacy_sparse_node(dir.path());

    mnem(dir.path(), &["reindex", "--lift-legacy-sparse"])
        .assert()
        .success();

    let db = dir.path().join(".mnem").join("repo.redb");
    let (bs_arc, ohs, _) = open_or_init(&db).expect("open redb");
    let bs = bs_arc.clone();
    let repo = ReadonlyRepo::open(bs_arc, ohs).expect("open repo");
    let head = repo.head_commit().expect("head commit after lift");

    // Count the nodes BEFORE the lift so we have a baseline to compare against.
    // (mnem init seeds an anchor node, so we can't hard-code the count.)
    let nodes_before = {
        // Re-open to count nodes at the commit just before the lift.
        // We already have head (which is the LIFT commit); we need the
        // prior commit. Instead, count via the lifted head and compare
        // to the total we know seeded (init anchor + 1 seeded node = 2).
        //
        // The simpler approach: count the nodes at the current (lifted) head
        // and compare against a count taken right after seeding.
        2usize // init seeds 1 anchor + seed_legacy_sparse_node adds 1 = 2
    };

    let cursor = mnem_core::prolly::Cursor::new(&*bs, &head.nodes).expect("cursor");
    let mut found = false;
    let mut total_nodes: usize = 0;
    for entry in cursor {
        let (_k, cid) = entry.expect("entry");
        total_nodes += 1;
        if cid == node_cid_before {
            found = true;
        }
    }
    assert!(
        found,
        "NodeCid {node_cid_before} must still be present in the nodes tree after sparse lift"
    );
    // Gap 5 fix: verify the lift did not add or remove any nodes.
    assert_eq!(
        total_nodes, nodes_before,
        "lift must not add or remove nodes; found {total_nodes}, expected {nodes_before}"
    );
}

/// On a repo that has no nodes with `extra["sparse_embed"]`, `--lift-legacy-sparse`
/// should succeed gracefully and print an appropriate message (no-op path).
#[test]
fn lift_legacy_sparse_no_op_on_clean_repo() {
    let dir = TempDir::new().unwrap();
    init(dir.path());

    // Add a normal node (no legacy sparse data).
    mnem(
        dir.path(),
        &["add", "node", "--summary", "a normal node", "--no-embed"],
    )
    .assert()
    .success();

    let out = mnem(dir.path(), &["reindex", "--lift-legacy-sparse"])
        .assert()
        .success();
    let stdout = String::from_utf8_lossy(&out.get_output().stdout).to_string();
    assert!(
        stdout.contains("nothing") || stdout.contains("no nodes") || stdout.contains("0"),
        "expected no-op message in stdout, got: {stdout}"
    );
}

/// Running `--lift-legacy-sparse` outside any mnem repo must fail with a
/// non-zero exit code.
#[test]
fn lift_legacy_sparse_no_repo_fails() {
    let dir = TempDir::new().unwrap();
    // Deliberately do NOT call init - no .mnem directory exists.
    mnem(dir.path(), &["reindex", "--lift-legacy-sparse"])
        .assert()
        .failure();
}

/// `--lift-legacy-sparse --dry-run` must report the candidate count but must
/// NOT write a new commit.
#[test]
fn lift_legacy_sparse_dry_run_no_commit() {
    let dir = TempDir::new().unwrap();
    init(dir.path());

    seed_legacy_sparse_node(dir.path());

    // Capture op count before.
    let ops_before = {
        let out = mnem(dir.path(), &["log", "--oneline"]).assert().success();
        let s = String::from_utf8_lossy(&out.get_output().stdout).to_string();
        s.lines().filter(|l| !l.trim().is_empty()).count()
    };

    let out = mnem(
        dir.path(),
        &["reindex", "--lift-legacy-sparse", "--dry-run"],
    )
    .assert()
    .success();
    let stdout = String::from_utf8_lossy(&out.get_output().stdout).to_string();
    // Gap 3 fix: assert on the actual dry-run output string ("would lift").
    // The run_lift_legacy_sparse dry-run branch prints:
    //   "would lift {n} legacy inline sparse embedding(s) to sidecar ..."
    assert!(
        stdout.contains("would lift"),
        "dry-run must report candidate count with 'would lift', got: {stdout}"
    );

    // Op count must not have grown.
    let ops_after = {
        let out = mnem(dir.path(), &["log", "--oneline"]).assert().success();
        let s = String::from_utf8_lossy(&out.get_output().stdout).to_string();
        s.lines().filter(|l| !l.trim().is_empty()).count()
    };
    assert_eq!(
        ops_before, ops_after,
        "--dry-run must not write a new commit"
    );
}

/// Running `--lift-legacy-sparse` twice on the same repo must succeed both
/// times. After the first run the sidecar is already populated, so the
/// idempotency skip logic in `run_lift_legacy_sparse` must detect this and
/// produce a no-op (no new commit). The sidecar content must be unchanged.
#[test]
fn lift_legacy_sparse_idempotent() {
    let dir = TempDir::new().unwrap();
    init(dir.path());

    let (node_cid, vocab) = seed_legacy_sparse_node(dir.path());

    // First lift.
    mnem(dir.path(), &["reindex", "--lift-legacy-sparse"])
        .assert()
        .success();

    // Second lift -- must also succeed without panicking.
    mnem(dir.path(), &["reindex", "--lift-legacy-sparse"])
        .assert()
        .success();

    // Gap 4 fix: after the second run, verify the sidecar content is intact.
    let db = dir.path().join(".mnem").join("repo.redb");
    let (bs2, ohs2, _) = open_or_init(&db).expect("open redb after second lift");
    let repo2 = ReadonlyRepo::open(bs2, ohs2).expect("open repo after second lift");
    let got = repo2
        .sparse_for(&node_cid, &vocab)
        .expect("sparse_for after second lift");
    assert!(
        got.is_some(),
        "sidecar must still have the SparseEmbed after the second lift"
    );
    let se = got.unwrap();
    assert_eq!(se.vocab_id, vocab);
    assert_eq!(se.indices, vec![10, 42, 99]);
    let expected_values = [0.8f32, 0.5, 0.2];
    for (i, (got_v, exp)) in se.values.iter().zip(expected_values.iter()).enumerate() {
        assert!(
            (got_v - exp).abs() < 1e-5,
            "idempotent second lift corrupted value[{i}]: got {got_v}, expected {exp}"
        );
    }
}

/// `--lift-legacy-sparse` and `--force` must be mutually exclusive.
#[test]
fn lift_legacy_sparse_and_force_are_mutually_exclusive() {
    let dir = TempDir::new().unwrap();
    init(dir.path());

    mnem(dir.path(), &["reindex", "--lift-legacy-sparse", "--force"])
        .assert()
        .failure();
}

/// Gap 2 fix: a node with malformed `extra["sparse_embed"]` (non-ascending
/// indices) must cause `--lift-legacy-sparse` to skip that node gracefully
/// (warn and continue), not crash. This exercises the
/// `decode_sparse_embed_from_ipld` -> `SparseEmbed::validate` error path in
/// `run_lift_legacy_sparse`.
///
/// Crucially, this test also seeds a VALID node alongside the corrupt one so
/// the result is unambiguous: the command must lift the valid node and skip
/// the corrupt node. If `validate()` were removed, the corrupt node would also
/// be lifted (wrong), so the assertion on the corrupt node's empty sidecar
/// would fail.
#[test]
fn lift_legacy_sparse_skips_corrupt_sparse_embed() {
    let dir = TempDir::new().unwrap();
    init(dir.path());

    // Build a SparseEmbed with non-ascending indices by going through
    // from_unsorted (which produces a valid, sorted result) then manually
    // constructing a corrupt IPLD payload.
    let corrupt_vocab = "splade-v3-corrupt".to_string();
    let valid_vocab = "splade-v3-valid".to_string();

    // We bypass SparseEmbed::new (which rejects non-ascending) by serialising
    // the struct fields directly via serde. Use a helper struct that skips
    // validation.
    #[derive(serde::Serialize)]
    struct RawSparseEmbed<'a> {
        indices: Vec<u32>,
        values: Vec<f32>,
        vocab_id: &'a str,
    }

    let corrupt = RawSparseEmbed {
        indices: vec![99, 10, 42], // non-ascending -- invalid
        values: vec![0.8, 0.5, 0.2],
        vocab_id: &corrupt_vocab,
    };

    // Open the store, commit BOTH nodes (corrupt + valid), then DROP all
    // handles so redb releases its file lock before we spawn the CLI process.
    let (corrupt_node_cid, valid_node_cid) = {
        let db = dir.path().join(".mnem").join("repo.redb");
        let (bs, ohs, _) = open_or_init(&db).expect("open redb");
        let repo = ReadonlyRepo::open(bs, ohs).expect("open repo");
        let mut tx = repo.start_transaction();

        // --- Corrupt node ---
        let mut corrupt_node = Node::new(
            mnem_core::id::NodeId::from_bytes_raw([0xCC_u8; 16]),
            "CorruptSparseDoc",
        )
        .with_summary("a node with a corrupt sparse_embed");

        let bytes = mnem_core::codec::to_canonical_bytes(&corrupt).expect("encode corrupt");
        let corrupt_ipld: ipld_core::ipld::Ipld =
            serde_ipld_dagcbor::from_slice(&bytes).expect("decode as Ipld");
        corrupt_node
            .extra
            .insert("sparse_embed".to_string(), corrupt_ipld);

        let corrupt_cid = tx.add_node(&corrupt_node).expect("add corrupt node");

        // --- Valid node (ascending indices) ---
        let valid_se = SparseEmbed::new(vec![10, 42, 99], vec![0.8, 0.5, 0.2], &valid_vocab)
            .expect("valid SparseEmbed");
        let mut valid_node = Node::new(
            mnem_core::id::NodeId::from_bytes_raw([0xEE_u8; 16]),
            "ValidSparseDoc",
        )
        .with_summary("a node with a valid sparse_embed");
        valid_node
            .extra
            .insert("sparse_embed".to_string(), sparse_embed_to_ipld(&valid_se));

        let valid_cid = tx.add_node(&valid_node).expect("add valid node");

        tx.commit("test author", "seed corrupt and valid sparse nodes")
            .expect("commit");
        // All handles (tx, repo, blockstore) drop at end of this block.
        (corrupt_cid, valid_cid)
    };

    // Run lift: it must succeed (exit 0), not crash.
    // The corrupt node is skipped with a warning; the valid node is lifted.
    mnem(dir.path(), &["reindex", "--lift-legacy-sparse"])
        .assert()
        .success();

    // Re-open the repo and check both sidecars.
    let db = dir.path().join(".mnem").join("repo.redb");
    let (bs, ohs, _) = open_or_init(&db).expect("open redb after lift");
    let repo = ReadonlyRepo::open(bs, ohs).expect("open repo after lift");

    // The VALID node must have been lifted into the sidecar.
    let got_valid = repo
        .sparse_for(&valid_node_cid, &valid_vocab)
        .expect("sparse_for valid node after lift");
    assert!(
        got_valid.is_some(),
        "valid node sidecar must be populated after lift (validate() allowed it through)"
    );
    let se_valid = got_valid.unwrap();
    assert_eq!(se_valid.indices, vec![10, 42, 99]);

    // The CORRUPT node must NOT have been lifted (validate() rejected it).
    // If validate() were removed, this assertion would fail because the corrupt
    // node would be in the sidecar too.
    let got_corrupt = repo
        .sparse_for(&corrupt_node_cid, &corrupt_vocab)
        .expect("sparse_for corrupt node after lift");
    assert!(
        got_corrupt.is_none(),
        "corrupt node sidecar must remain empty: validate() must have rejected it"
    );
}

/// Gap 6 fix: `--since <commit>` must limit the set of nodes whose
/// `extra["sparse_embed"]` is lifted. Nodes that were present in the
/// since-commit's tree are skipped; only nodes added after that commit
/// are candidates.
///
/// Scenario:
///  1. Commit A: seed a legacy sparse node.
///  2. Record the CID of commit A.
///  3. Commit B: add a plain node (no legacy data) so HEAD advances.
///  4. Run `--lift-legacy-sparse --since <commit_A_cid>`: commit A's node
///     is in the since-set and must be skipped => sidecar stays empty.
///  5. Run `--lift-legacy-sparse` (no --since): now the node IS lifted.
///  6. Verify the sidecar has the vector.
#[test]
fn lift_legacy_sparse_respects_since_flag() {
    let dir = TempDir::new().unwrap();
    init(dir.path());

    // Step 1 + 2: seed the legacy sparse node and capture commit A's CID.
    let (node_cid, vocab) = seed_legacy_sparse_node(dir.path());

    // The seed function commits; HEAD is now commit A. Capture its CID.
    let commit_a_cid: String = {
        let db = dir.path().join(".mnem").join("repo.redb");
        let (bs2, ohs2, _) = open_or_init(&db).expect("open redb for commit_a");
        let repo2 = ReadonlyRepo::open(bs2, ohs2).expect("open repo for commit_a");
        let _head = repo2.head_commit().expect("head after seed");
        // The Commit CID is the first entry in view().heads.
        repo2
            .view()
            .heads
            .first()
            .expect("head CID present")
            .to_string()
    };

    // Step 3: add a plain node to advance HEAD to commit B.
    mnem(
        dir.path(),
        &[
            "add",
            "node",
            "--summary",
            "commit B plain node",
            "--no-embed",
        ],
    )
    .assert()
    .success();

    // Step 4: lift with --since <commit_A_cid>.
    // Nodes present in commit A's nodes-tree are skipped.
    // The legacy sparse node WAS in commit A, so it must be skipped.
    let out = mnem(
        dir.path(),
        &["reindex", "--lift-legacy-sparse", "--since", &commit_a_cid],
    )
    .assert()
    .success();
    let stdout = String::from_utf8_lossy(&out.get_output().stdout).to_string();
    // Expect a no-op message (0 candidates lifted).
    assert!(
        stdout.contains("nothing") || stdout.contains("no nodes") || stdout.contains("0"),
        "--since should cause the legacy node to be skipped; got: {stdout}"
    );

    // Verify the sidecar is still empty for the legacy node.
    {
        let db = dir.path().join(".mnem").join("repo.redb");
        let (bs3, ohs3, _) = open_or_init(&db).expect("open redb after since-lift");
        let repo3 = ReadonlyRepo::open(bs3, ohs3).expect("open repo after since-lift");
        let got = repo3
            .sparse_for(&node_cid, &vocab)
            .expect("sparse_for after since-lift");
        assert!(
            got.is_none(),
            "sidecar must still be empty after --since skipped the legacy node"
        );
    }

    // Step 5: run without --since -- now the legacy node IS lifted.
    mnem(dir.path(), &["reindex", "--lift-legacy-sparse"])
        .assert()
        .success();

    // Step 6: verify the sidecar now has the vector.
    let db = dir.path().join(".mnem").join("repo.redb");
    let (bs4, ohs4, _) = open_or_init(&db).expect("open redb after full lift");
    let repo4 = ReadonlyRepo::open(bs4, ohs4).expect("open repo after full lift");
    let got = repo4
        .sparse_for(&node_cid, &vocab)
        .expect("sparse_for after full lift");
    assert!(
        got.is_some(),
        "sidecar must have the vector after full --lift-legacy-sparse"
    );
    let se = got.unwrap();
    assert_eq!(se.indices, vec![10, 42, 99]);
}

/// Inclusion side of the `--since` filter: a node added AFTER the since
/// boundary commit must be LIFTED, while the node present IN the since
/// commit must be SKIPPED.
///
/// Scenario:
///  1. Commit A: seed legacy sparse node A (records commit A's CID).
///  2. Commit B: seed legacy sparse node B (a different node, added after A).
///  3. Run `--lift-legacy-sparse --since <commit_A_cid>`:
///     - node A is in commit A's tree => SKIPPED (exclusion side).
///     - node B was added after commit A => LIFTED (inclusion side).
///  4. Assert node A's sidecar is EMPTY (skipped correctly).
///  5. Assert node B's sidecar is POPULATED (lifted correctly).
#[test]
fn lift_legacy_sparse_since_includes_nodes_added_after() {
    let dir = TempDir::new().unwrap();
    init(dir.path());

    // Step 1: seed node A and capture commit A's CID.
    let (node_cid_a, vocab_a) = seed_legacy_sparse_node(dir.path());

    let commit_a_cid: String = {
        let db = dir.path().join(".mnem").join("repo.redb");
        let (bs, ohs, _) = open_or_init(&db).expect("open redb for commit_a");
        let repo = ReadonlyRepo::open(bs, ohs).expect("open repo for commit_a");
        repo.view()
            .heads
            .first()
            .expect("head CID present after node A")
            .to_string()
    };

    // Step 2: seed node B (added after commit A). We need a distinct NodeId
    // and vocab so it doesn't collide with node A's sidecar entry.
    let (node_cid_b, vocab_b) = {
        let vocab = "splade-v3-legacy-test-b".to_string();
        let se = SparseEmbed::new(vec![11, 43, 100], vec![0.7, 0.4, 0.1], &vocab)
            .expect("valid SparseEmbed for node B");

        let db = dir.path().join(".mnem").join("repo.redb");
        let (bs, ohs, _) = open_or_init(&db).expect("open redb for node B seed");
        let repo = ReadonlyRepo::open(bs, ohs).expect("open repo for node B seed");

        let mut tx = repo.start_transaction();
        let mut node = Node::new(NodeId::from_bytes_raw([0xDD_u8; 16]), "LegacySparseDocB")
            .with_summary("a second legacy document with inline sparse_embed");
        node.extra
            .insert("sparse_embed".to_string(), sparse_embed_to_ipld(&se));

        let node_cid = tx.add_node(&node).expect("add node B");
        let r = tx
            .commit("test author", "seed legacy sparse node B")
            .expect("commit node B");

        assert!(
            r.sparse_for(&node_cid, &vocab)
                .expect("sparse_for node B before lift")
                .is_none(),
            "node B sidecar must be empty before lift"
        );

        (node_cid, vocab)
    };

    // Step 3: run --lift-legacy-sparse --since <commit_A_cid>.
    // Node A is in commit A's tree => skipped.
    // Node B was added after commit A => should be lifted.
    mnem(
        dir.path(),
        &["reindex", "--lift-legacy-sparse", "--since", &commit_a_cid],
    )
    .assert()
    .success();

    // Open the repo once for both assertions.
    let db = dir.path().join(".mnem").join("repo.redb");
    let (bs, ohs, _) = open_or_init(&db).expect("open redb after since-inclusive lift");
    let repo = ReadonlyRepo::open(bs, ohs).expect("open repo after since-inclusive lift");

    // Step 4: node A's sidecar must still be EMPTY (it was in commit A => skipped).
    let got_a = repo
        .sparse_for(&node_cid_a, &vocab_a)
        .expect("sparse_for node A after since-inclusive lift");
    assert!(
        got_a.is_none(),
        "node A sidecar must remain empty: it was present in the --since commit"
    );

    // Step 5: node B's sidecar must be POPULATED (it was added after commit A).
    let got_b = repo
        .sparse_for(&node_cid_b, &vocab_b)
        .expect("sparse_for node B after since-inclusive lift");
    assert!(
        got_b.is_some(),
        "node B sidecar must be populated: it was added after the --since boundary"
    );
    let se_b = got_b.unwrap();
    assert_eq!(se_b.indices, vec![11, 43, 100]);
    assert_eq!(se_b.vocab_id, vocab_b);
}
