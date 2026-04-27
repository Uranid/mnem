//! Thread-count invariance for `NodeCid`.
//!
//! # The invariant under test
//!
//! `NodeCid = blake3(canonical_dag_cbor(Node))`. Dense embedding vectors
//! produced by ORT (or any other dense embedder) drift in the last bit
//! of an `f32` reduction across thread counts because IEEE-754 addition
//! is non-associative: a 4-thread split-sum lands different bytes than
//! an 8-thread split-sum on the same input. If those bytes had stayed
//! inline on `Node`, two machines hashing the same source text on
//! different `MNEM_ORT_INTRA_THREADS` values would emit different
//! `NodeCid`s, breaking federated dedup of embed-bearing nodes.
//!
//! With dense embeddings lifted into a per-commit Prolly sidecar
//! (`Commit.embeddings`), the canonical bytes of a `Node` no longer
//! depend on any embedding-runtime output. Two threads producing
//! materially different embedding bytes for the same Node MUST
//! converge on a single `NodeCid` and a single node-tree root in the
//! commit, while their sidecar Cids legitimately differ.
//!
//! # Why this test does not call ORT
//!
//! There is no embedder available in the in-tree test environment
//! (no model files, no ORT runtime, no GPU). Instead we simulate the
//! invariant by attaching distinct synthetic vector bytes from two
//! threads. The assertion that survives without an actual ORT runtime
//! is the cleaner, equivalent one: serialization of a `Node` is
//! independent of any pending sidecar writes - the embedding bytes
//! cannot reach `NodeCid` by construction.
//!
//! Two threads independently:
//!
//! 1. Build the same Node via `Node::new` (no `embed` field exists
//!    post-sidecar; the canonical bytes are fully determined by id /
//!    ntype / props / content).
//! 2. Hash it via `hash_to_cid` and confirm the CID matches.
//! 3. Open a hermetic in-memory repo, add the node, stage a distinct
//!    synthetic embedding via `Transaction::set_embedding`, commit.
//! 4. Compare commit `nodes` roots (must match) and commit
//!    `embeddings` roots (must differ - different vector bytes).

use std::sync::Arc;
use std::thread;

use bytes::Bytes;

use mnem_core::codec::hash_to_cid;
use mnem_core::id::{Cid, NodeId};
use mnem_core::objects::{Dtype, Embedding, Node};
use mnem_core::repo::ReadonlyRepo;
use mnem_core::store::{Blockstore, MemoryBlockstore, MemoryOpHeadsStore, OpHeadsStore};

/// Build a fixed `Node`. Identical input on every thread so any CID
/// divergence comes from the serialization path, not the input.
fn fixture_node() -> Node {
    Node::new(NodeId::from_bytes_raw([0x42; 16]), "Doc")
}

/// Construct an `Embedding` from raw `f32` bytes for a given model.
///
/// The two threads pass distinct vectors here to model the ORT
/// thread-count drift: same source text, same model name, but the
/// last-bit-different reduction lands different bytes. The test then
/// proves these bytes never reach `NodeCid`.
fn synthetic_embedding(model: &str, vector: &[f32]) -> Embedding {
    let mut bytes = Vec::with_capacity(vector.len() * 4);
    for x in vector {
        bytes.extend_from_slice(&x.to_le_bytes());
    }
    Embedding {
        model: model.to_string(),
        dtype: Dtype::F32,
        dim: vector.len() as u32,
        vector: Bytes::from(bytes),
    }
}

/// One worker: hash the fixture Node, then build a hermetic repo and
/// commit it with a model-named embedding produced from `vector_bytes`.
///
/// Returns `(node_cid, commit_nodes_root, commit_embeddings_root)`.
fn run_worker(vector: Vec<f32>) -> (Cid, Cid, Option<Cid>) {
    let node = fixture_node();
    let (_bytes, node_cid) = hash_to_cid(&node).expect("node hashes");

    let bs: Arc<dyn Blockstore> = Arc::new(MemoryBlockstore::new());
    let ohs: Arc<dyn OpHeadsStore> = Arc::new(MemoryOpHeadsStore::new());
    let repo = ReadonlyRepo::init(bs, ohs).expect("init repo");

    let mut tx = repo.start_transaction();
    let added_cid = tx.add_node(&node).expect("add_node");
    assert_eq!(
        added_cid, node_cid,
        "add_node must agree with hash_to_cid on Node identity"
    );

    let emb = synthetic_embedding("mTest", &vector);
    tx.set_embedding(added_cid.clone(), "mTest".into(), emb)
        .expect("set_embedding");

    let new_repo = tx.commit("g16-test", "thread-invariance worker").unwrap();
    let commit = new_repo
        .head_commit()
        .cloned()
        .expect("commit landed at head");

    (node_cid, commit.nodes, commit.embeddings)
}

/// Runs two `run_worker` calls on real `std::thread::spawn` threads
/// with distinct synthetic vectors and asserts the invariant.
#[test]
fn node_cid_is_independent_of_pending_embedding_bytes() {
    let h_a = thread::spawn(|| run_worker(vec![1.0, 0.0, 0.0, 0.0]));
    // Thread B uses a vector that has the same shape but distinct bytes.
    // This is the in-test stand-in for ORT thread-count drift: same
    // model name, same dim, materially different bytes.
    let h_b = thread::spawn(|| run_worker(vec![0.0, 1.0, 0.0, 0.0]));

    let (cid_a, nodes_root_a, emb_root_a) = h_a.join().expect("thread A");
    let (cid_b, nodes_root_b, emb_root_b) = h_b.join().expect("thread B");

    // Primary: NodeCid is identical regardless of the embedding bytes
    // each thread chose to attach in its own repo. This is the
    // load-bearing federated-dedup invariant.
    assert_eq!(
        cid_a, cid_b,
        "Node identity must be independent of embedding bytes"
    );

    // The node-tree root in the commit also matches: the commit's
    // `nodes` Prolly tree carries the same NodeCid bytes from both
    // threads, so two commits produced from byte-identical Node sets
    // share their node-tree root even when their embedding sidecars
    // diverge.
    assert_eq!(
        nodes_root_a, nodes_root_b,
        "commit.nodes must match across threads when only embedding bytes differ"
    );

    // Counter-invariant: the embedding sidecar Cids legitimately
    // differ because the threads wrote different vector bytes. If
    // these were equal it would mean the synthetic embeddings
    // collapsed somewhere unexpected and the test no longer exercises
    // the invariant.
    assert_ne!(
        emb_root_a, emb_root_b,
        "sidecar roots must differ when the staged vectors differ; otherwise the test is vacuous"
    );

    // Sanity: both sidecar roots are populated. A `None` here would
    // mean the commit path elided the sidecar even though the worker
    // staged an embedding, which would also make the test vacuous.
    assert!(
        emb_root_a.is_some() && emb_root_b.is_some(),
        "both workers staged an embedding; both commits must carry a sidecar root"
    );
}
