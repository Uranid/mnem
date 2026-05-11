//! Integration tests for the G17 sparse-embedding sidecar.
//!
//! Sparse embeddings (SPLADE, BGE-M3-sparse, etc.) moved from
//! `Node` inline bytes into a per-commit Prolly sidecar tree
//! (`Commit.sparse`, keyed by `NodeCid`). These tests verify:
//!
//! 1. `NodeCid` is independent of any sparse embedding bytes staged
//!    via `Transaction::set_sparse_embedding`.
//! 2. Sparse embeddings round-trip through commit + `ReadonlyRepo::sparse_for`.
//! 3. Multiple vocab_ids coexist in the same sidecar bucket.
//! 4. Removing a node also removes its sidecar entry.
//! 5. A commit with no sparse embeddings carries `Commit.sparse = None`.
//! 6. `SparseInvertedIndex::build_from_repo` reads from the sidecar.

use std::sync::Arc;

use mnem_core::codec::hash_to_cid;
use mnem_core::id::NodeId;
use mnem_core::index::sparse::SparseInvertedIndex;
use mnem_core::objects::Node;
use mnem_core::repo::ReadonlyRepo;
use mnem_core::sparse::SparseEmbed;
use mnem_core::store::{Blockstore, MemoryBlockstore, MemoryOpHeadsStore, OpHeadsStore};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn make_store() -> (Arc<dyn Blockstore>, Arc<dyn OpHeadsStore>) {
    (
        Arc::new(MemoryBlockstore::new()),
        Arc::new(MemoryOpHeadsStore::new()),
    )
}

fn sparse_embed(indices: Vec<u32>, values: Vec<f32>, vocab: &str) -> SparseEmbed {
    SparseEmbed::new(indices, values, vocab).expect("valid SparseEmbed")
}

fn node(tag: u8) -> Node {
    Node::new(NodeId::from_bytes_raw([tag; 16]), "Doc")
}

// ---------------------------------------------------------------------------
// Test 1: NodeCid is independent of sparse embedding bytes
// ---------------------------------------------------------------------------

#[test]
fn node_cid_is_independent_of_sparse_bytes() {
    let (bs, ohs) = make_store();
    let repo = ReadonlyRepo::init(bs, ohs).expect("init");

    let n = node(0xAB);
    let (_bytes, expected_cid) = hash_to_cid(&n).expect("hash");

    let mut tx = repo.start_transaction();
    let added_cid = tx.add_node(&n).expect("add_node");

    // The CID returned by add_node must equal hash_to_cid(node). If sparse
    // bytes affected NodeCid construction, this would diverge after staging.
    assert_eq!(
        added_cid, expected_cid,
        "add_node must agree with hash_to_cid on Node identity"
    );

    // Stage a sparse embedding for the node. This must not alter `added_cid`.
    let se = sparse_embed(vec![10, 20, 30], vec![0.9, 0.5, 0.1], "splade-v3");
    tx.set_sparse_embedding(added_cid.clone(), "splade-v3".into(), se)
        .expect("set_sparse_embedding");

    // Hashing the same Node again after staging must still match. The node
    // struct itself must not carry any sparse-embedding bytes.
    let (_bytes2, rehashed) = hash_to_cid(&n).expect("rehash");
    assert_eq!(
        rehashed, expected_cid,
        "NodeCid must be stable regardless of pending sparse embedding bytes"
    );

    let new_repo = tx.commit("tester", "g17 nodecid test").expect("commit");
    let commit = new_repo.head_commit().cloned().expect("head_commit");

    // The sparse sidecar must be populated after commit.
    assert!(
        commit.sparse.is_some(),
        "Commit.sparse must be Some when a sparse embedding was staged"
    );
}

// ---------------------------------------------------------------------------
// Test 2: sparse embedding round-trips through commit + sparse_for
// ---------------------------------------------------------------------------

#[test]
fn sparse_embedding_round_trips_through_sidecar() {
    let (bs, ohs) = make_store();
    let repo = ReadonlyRepo::init(bs, ohs).expect("init");

    let n = node(0x01);
    let se = sparse_embed(vec![1, 5, 9], vec![0.8, 0.4, 0.1], "vocab-a");

    let mut tx = repo.start_transaction();
    let cid = tx.add_node(&n).expect("add_node");
    tx.set_sparse_embedding(cid.clone(), "vocab-a".into(), se.clone())
        .expect("set_sparse_embedding");
    let new_repo = tx.commit("tester", "round-trip").expect("commit");

    let got = new_repo
        .sparse_for(&cid, "vocab-a")
        .expect("sparse_for ok")
        .expect("must be Some");

    assert_eq!(got.vocab_id, "vocab-a");
    assert_eq!(got.indices, se.indices);
    assert_eq!(got.values, se.values);
}

// ---------------------------------------------------------------------------
// Test 3: multiple vocab_ids coexist in the same sidecar bucket
// ---------------------------------------------------------------------------

#[test]
fn multiple_vocabs_coexist_in_sidecar_bucket() {
    let (bs, ohs) = make_store();
    let repo = ReadonlyRepo::init(bs, ohs).expect("init");

    let n = node(0x02);
    let se_a = sparse_embed(vec![1, 2], vec![0.9, 0.5], "vocab-a");
    let se_b = sparse_embed(vec![3, 4], vec![0.7, 0.3], "vocab-b");

    let mut tx = repo.start_transaction();
    let cid = tx.add_node(&n).expect("add_node");
    tx.set_sparse_embedding(cid.clone(), "vocab-a".into(), se_a.clone())
        .expect("set vocab-a");
    tx.set_sparse_embedding(cid.clone(), "vocab-b".into(), se_b.clone())
        .expect("set vocab-b");
    let new_repo = tx.commit("tester", "multi-vocab").expect("commit");

    let got_a = new_repo
        .sparse_for(&cid, "vocab-a")
        .expect("ok")
        .expect("vocab-a must exist");
    let got_b = new_repo
        .sparse_for(&cid, "vocab-b")
        .expect("ok")
        .expect("vocab-b must exist");

    assert_eq!(got_a.indices, se_a.indices);
    assert_eq!(got_b.indices, se_b.indices);

    // A third vocab that was never staged must return None.
    let missing = new_repo.sparse_for(&cid, "vocab-c").expect("lookup ok");
    assert!(missing.is_none(), "unstaged vocab must return None");
}

// ---------------------------------------------------------------------------
// Test 4: commit with no sparse embeddings carries Commit.sparse = None
// ---------------------------------------------------------------------------

#[test]
fn no_sparse_staged_means_commit_sparse_is_none() {
    let (bs, ohs) = make_store();
    let repo = ReadonlyRepo::init(bs, ohs).expect("init");

    let n = node(0x03);
    let mut tx = repo.start_transaction();
    tx.add_node(&n).expect("add_node");
    let new_repo = tx.commit("tester", "no-sparse").expect("commit");

    let commit = new_repo.head_commit().cloned().expect("head_commit");
    assert!(
        commit.sparse.is_none(),
        "Commit.sparse must be None when nothing was staged"
    );
}

// ---------------------------------------------------------------------------
// Test 5: removing a node clears its pending sparse entry
// ---------------------------------------------------------------------------

#[test]
fn remove_node_clears_pending_sparse() {
    let (bs, ohs) = make_store();
    let repo = ReadonlyRepo::init(bs, ohs).expect("init");

    let n = node(0x04);
    let se = sparse_embed(vec![7, 8], vec![0.6, 0.3], "vocab-x");

    let mut tx = repo.start_transaction();
    let cid = tx.add_node(&n).expect("add_node");
    tx.set_sparse_embedding(cid, "vocab-x".into(), se)
        .expect("set_sparse_embedding");
    // Remove the node before committing - this should also drop the pending
    // sparse entry so the sidecar is not populated.
    tx.remove_node(n.id);

    let new_repo = tx.commit("tester", "remove-clears-sparse").expect("commit");
    let commit = new_repo.head_commit().cloned().expect("head_commit");
    assert!(
        commit.sparse.is_none(),
        "sparse sidecar must be empty when the only node was removed before commit"
    );
}

// ---------------------------------------------------------------------------
// Test 6: SparseInvertedIndex::build_from_repo reads from sidecar
// ---------------------------------------------------------------------------

#[test]
fn sparse_inverted_index_build_from_repo() {
    let (bs, ohs) = make_store();
    let repo = ReadonlyRepo::init(bs, ohs).expect("init");

    let n1 = node(0x10);
    let n2 = node(0x20);
    let n3 = node(0x30); // disjoint token - must not appear in search results

    let se1 = sparse_embed(vec![1, 2], vec![1.0, 0.5], "splade-test");
    let se2 = sparse_embed(vec![2, 3], vec![0.5, 1.0], "splade-test");
    let se3 = sparse_embed(vec![99], vec![5.0], "splade-test");

    let mut tx = repo.start_transaction();
    let cid1 = tx.add_node(&n1).expect("add n1");
    let cid2 = tx.add_node(&n2).expect("add n2");
    let cid3 = tx.add_node(&n3).expect("add n3");
    tx.set_sparse_embedding(cid1, "splade-test".into(), se1)
        .expect("sparse n1");
    tx.set_sparse_embedding(cid2, "splade-test".into(), se2)
        .expect("sparse n2");
    tx.set_sparse_embedding(cid3, "splade-test".into(), se3)
        .expect("sparse n3");
    let new_repo = tx.commit("tester", "build-from-repo").expect("commit");

    let idx = SparseInvertedIndex::build_from_repo(&new_repo, "splade-test").expect("build index");
    assert_eq!(idx.doc_count(), 3, "index must contain all three docs");

    // Query on token 2 overlaps n1 and n2 but not n3.
    let query = sparse_embed(vec![2], vec![1.0], "splade-test");
    let hits = idx.search(&query, 10).expect("search");
    assert_eq!(hits.len(), 2, "only n1 and n2 share token 2");

    let hit_ids: Vec<_> = hits.iter().map(|h| h.node_id).collect();
    assert!(hit_ids.contains(&n1.id), "n1 must appear");
    assert!(hit_ids.contains(&n2.id), "n2 must appear");

    // n3's token 99 has no overlap with query token 2.
    assert!(!hit_ids.contains(&n3.id), "n3 must not appear");
}

// ---------------------------------------------------------------------------
// Test 7: sparse_for returns None for a node with no sidecar entry
// ---------------------------------------------------------------------------

#[test]
fn sparse_for_returns_none_for_node_without_entry() {
    let (bs, ohs) = make_store();
    let repo = ReadonlyRepo::init(bs, ohs).expect("init");

    // Add two nodes but only set sparse for n1.
    let n1 = node(0x05);
    let n2 = node(0x06);
    let se1 = sparse_embed(vec![1], vec![1.0], "v0");

    let mut tx = repo.start_transaction();
    let cid1 = tx.add_node(&n1).expect("add n1");
    let cid2 = tx.add_node(&n2).expect("add n2");
    tx.set_sparse_embedding(cid1.clone(), "v0".into(), se1)
        .expect("set n1");
    let new_repo = tx.commit("tester", "partial-sparse").expect("commit");

    assert!(
        new_repo.sparse_for(&cid1, "v0").expect("ok").is_some(),
        "n1 must have a sidecar entry"
    );
    assert!(
        new_repo.sparse_for(&cid2, "v0").expect("ok").is_none(),
        "n2 must not have a sidecar entry"
    );
}

// ---------------------------------------------------------------------------
// Test 8: incremental commits accumulate sidecar entries
// ---------------------------------------------------------------------------

#[test]
fn incremental_commits_accumulate_sparse_sidecar() {
    let (bs, ohs) = make_store();
    let repo = ReadonlyRepo::init(bs, ohs).expect("init");

    // First commit: n1 with sparse.
    let n1 = node(0x07);
    let se1 = sparse_embed(vec![10], vec![0.8], "v1");

    let mut tx1 = repo.start_transaction();
    let cid1 = tx1.add_node(&n1).expect("add n1");
    tx1.set_sparse_embedding(cid1.clone(), "v1".into(), se1.clone())
        .expect("set n1");
    let repo2 = tx1.commit("tester", "commit-1").expect("commit 1");

    // Second commit: n2 with sparse (n1 stays from base).
    let n2 = node(0x08);
    let se2 = sparse_embed(vec![20], vec![0.6], "v1");

    let mut tx2 = repo2.start_transaction();
    let cid2 = tx2.add_node(&n2).expect("add n2");
    tx2.set_sparse_embedding(cid2.clone(), "v1".into(), se2.clone())
        .expect("set n2");
    let repo3 = tx2.commit("tester", "commit-2").expect("commit 2");

    // Both nodes must be visible from the final repo.
    let got1 = repo3
        .sparse_for(&cid1, "v1")
        .expect("ok")
        .expect("n1 must persist through commit 2");
    let got2 = repo3
        .sparse_for(&cid2, "v1")
        .expect("ok")
        .expect("n2 added in commit 2");

    assert_eq!(got1.indices, se1.indices);
    assert_eq!(got2.indices, se2.indices);
}
