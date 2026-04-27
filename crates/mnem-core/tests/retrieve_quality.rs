//! Integration tests for retrieval quality contracts.
//!
//! These tests exercise `Retriever` against a full `ReadonlyRepo` end-
//! to-end (commits, views, tombstones, filters), not the pure fusion
//! helpers in the unit-test module on `retrieve.rs`.

use std::sync::Arc;

use bytes::Bytes;
use ipld_core::ipld::Ipld;

use mnem_core::id::{ChangeId, EdgeId, NodeId};
use mnem_core::objects::{Dtype, Edge, Embedding, Node};
use mnem_core::repo::{CommitOptions, ReadonlyRepo};
use mnem_core::retrieve::GraphExpand;
use mnem_core::store::{Blockstore, MemoryBlockstore, MemoryOpHeadsStore, OpHeadsStore};

fn stores() -> (Arc<dyn Blockstore>, Arc<dyn OpHeadsStore>) {
    (
        Arc::new(MemoryBlockstore::new()),
        Arc::new(MemoryOpHeadsStore::new()),
    )
}

fn f32_embed(model: &str, v: &[f32]) -> Embedding {
    let mut bytes = Vec::with_capacity(v.len() * 4);
    for x in v {
        bytes.extend_from_slice(&x.to_le_bytes());
    }
    Embedding {
        model: model.to_string(),
        dtype: Dtype::F32,
        dim: u32::try_from(v.len()).expect("test vec fits in u32"),
        vector: Bytes::from(bytes),
    }
}

#[test]
fn tombstoned_nodes_are_filtered_from_retrieve_by_default() {
    // Two near-identical Doc nodes with embeddings. One is tombstoned
    // in a follow-up commit. A vector retrieve must surface the live
    // node and drop the tombstoned one, even though the tombstoned
    // node's embedding is still present in the built index.
    let (bs, ohs) = stores();
    let repo = ReadonlyRepo::init(bs, ohs).unwrap();
    let alive = Node::new(NodeId::new_v7(), "Doc")
        .with_summary("alive doc")
        .with_prop("name", Ipld::String("alive".into()));
    let revoked = Node::new(NodeId::new_v7(), "Doc")
        .with_summary("revoked doc")
        .with_prop("name", Ipld::String("revoked".into()));

    let mut tx = repo.start_transaction();
    let alive_cid = tx.add_node(&alive).unwrap();
    let alive_emb = f32_embed("m", &[1.0, 0.0]);
    tx.set_embedding(alive_cid, alive_emb.model.clone(), alive_emb)
        .unwrap();
    let revoked_cid = tx.add_node(&revoked).unwrap();
    let revoked_emb = f32_embed("m", &[1.0, 0.0]); // same vec -> both rank equally
    tx.set_embedding(revoked_cid, revoked_emb.model.clone(), revoked_emb)
        .unwrap();
    let repo = tx.commit("t", "seed").unwrap();

    // Tombstone the second node.
    let mut tx = repo.start_transaction();
    tx.tombstone_node(revoked.id, "user asked to forget")
        .unwrap();
    let repo = tx.commit("t", "revoke").unwrap();

    // Default retrieve: the revoked id must not appear.
    let result = repo
        .retrieve()
        .vector("m", vec![1.0, 0.0])
        .execute()
        .unwrap();
    let ids: Vec<_> = result.items.iter().map(|i| i.node.id).collect();
    assert!(
        ids.contains(&alive.id),
        "live node must still surface, got ids={ids:?}"
    );
    assert!(
        !ids.contains(&revoked.id),
        "tombstoned node must be filtered out of retrieve by default, got ids={ids:?}"
    );
}

#[test]
fn include_tombstoned_opt_out_surfaces_revoked_nodes_for_audit() {
    // Audit / debug callers pass `include_tombstoned(true)` and see
    // everything, tombstones included. This is the restore / history
    // inspection path and must round-trip through the fused retrieval
    // pipeline, not just through a raw lookup.
    let (bs, ohs) = stores();
    let repo = ReadonlyRepo::init(bs, ohs).unwrap();
    let alive = Node::new(NodeId::new_v7(), "Doc").with_summary("alive");
    let revoked = Node::new(NodeId::new_v7(), "Doc").with_summary("revoked");

    let mut tx = repo.start_transaction();
    let alive_cid = tx.add_node(&alive).unwrap();
    let alive_emb = f32_embed("m", &[1.0, 0.0]);
    tx.set_embedding(alive_cid, alive_emb.model.clone(), alive_emb)
        .unwrap();
    let revoked_cid = tx.add_node(&revoked).unwrap();
    let revoked_emb = f32_embed("m", &[1.0, 0.0]);
    tx.set_embedding(revoked_cid, revoked_emb.model.clone(), revoked_emb)
        .unwrap();
    let repo = tx.commit("t", "seed").unwrap();

    let mut tx = repo.start_transaction();
    tx.tombstone_node(revoked.id, "revoked").unwrap();
    let repo = tx.commit("t", "revoke").unwrap();

    // Default behaviour: revoked is filtered out.
    let default_result = repo
        .retrieve()
        .vector("m", vec![1.0, 0.0])
        .execute()
        .unwrap();
    let default_ids: Vec<_> = default_result.items.iter().map(|i| i.node.id).collect();
    assert!(!default_ids.contains(&revoked.id));

    // Opt-out: revoked surfaces again.
    let audit_result = repo
        .retrieve()
        .vector("m", vec![1.0, 0.0])
        .include_tombstoned(true)
        .execute()
        .unwrap();
    let audit_ids: Vec<_> = audit_result.items.iter().map(|i| i.node.id).collect();
    assert!(
        audit_ids.contains(&alive.id),
        "live node still present in audit retrieve, got ids={audit_ids:?}"
    );
    assert!(
        audit_ids.contains(&revoked.id),
        "tombstoned node must surface when include_tombstoned(true), got ids={audit_ids:?}"
    );
}

/// Graph-expand with `direction = Incoming` follows the back-index:
/// a seed node which is the `dst` of several edges pulls in the
/// `src` side as neighbors. Mirror of the existing forward expansion
/// test, using the newly-added incoming-adjacency index.
#[test]
fn graph_expand_follows_incoming_edges_when_configured() {
    // Shape: author A wrote docs D1 and D2 (edges A -authored-> D1,
    // A -authored-> D2). A semantic retrieve on D1 (the seed) should,
    // when configured to walk backwards, surface A as a neighbor even
    // though D1 itself has no outgoing edges.
    let (bs, ohs) = stores();
    let repo = ReadonlyRepo::init(bs, ohs).unwrap();
    let author = Node::new(NodeId::new_v7(), "Person")
        .with_summary("Alice (author)")
        .with_prop("name", Ipld::String("Alice".into()));
    let d1 = Node::new(NodeId::new_v7(), "Doc")
        .with_summary("doc one text")
        .with_prop("name", Ipld::String("D1".into()));
    let d2 = Node::new(NodeId::new_v7(), "Doc")
        .with_summary("doc two text")
        .with_prop("name", Ipld::String("D2".into()));

    let mut tx = repo.start_transaction();
    tx.add_node(&author).unwrap();
    let d1_cid = tx.add_node(&d1).unwrap();
    let d1_emb = f32_embed("m", &[1.0, 0.0]);
    tx.set_embedding(d1_cid, d1_emb.model.clone(), d1_emb)
        .unwrap();
    let d2_cid = tx.add_node(&d2).unwrap();
    let d2_emb = f32_embed("m", &[0.0, 1.0]);
    tx.set_embedding(d2_cid, d2_emb.model.clone(), d2_emb)
        .unwrap();
    tx.add_edge(&Edge::new(EdgeId::new_v7(), "authored", author.id, d1.id))
        .unwrap();
    tx.add_edge(&Edge::new(EdgeId::new_v7(), "authored", author.id, d2.id))
        .unwrap();
    let repo = tx.commit("t", "seed").unwrap();

    // Forward-only graph-expand starting from D1 reaches no one -
    // D1 has no outgoing edges.
    let forward = repo
        .retrieve()
        .vector("m", vec![1.0, 0.0])
        .with_graph_expand(GraphExpand::new())
        .execute()
        .unwrap();
    let forward_ids: Vec<_> = forward.items.iter().map(|i| i.node.id).collect();
    assert!(
        !forward_ids.contains(&author.id),
        "forward graph-expand from D1 should NOT reach author; got {forward_ids:?}"
    );

    // Backwards graph-expand: D1's incoming edge is (author,
    // authored), so the back-walk surfaces `author` as a neighbor.
    let backward = repo
        .retrieve()
        .vector("m", vec![1.0, 0.0])
        .with_graph_expand(GraphExpand::new().with_incoming())
        .execute()
        .unwrap();
    let backward_ids: Vec<_> = backward.items.iter().map(|i| i.node.id).collect();
    assert!(
        backward_ids.contains(&author.id),
        "backwards graph-expand from D1 must surface author via the incoming index; got {backward_ids:?}"
    );
}

/// Graph-expand with `direction = Both` walks both directions from
/// each seed and finds neighbors on either side. The bidirectional
/// behaviour covers the "supersession chain" and "provenance"
/// traversal pattern in one call.
#[test]
fn graph_expand_both_directions_walks_both_sides() {
    // Shape: Topic -tagged-> Doc1; Alice -authored-> Doc1. A seed on
    // Doc1 with Any direction should reach Topic AND Alice.
    let (bs, ohs) = stores();
    let repo = ReadonlyRepo::init(bs, ohs).unwrap();
    let doc1 = Node::new(NodeId::new_v7(), "Doc").with_summary("the doc");
    let topic = Node::new(NodeId::new_v7(), "Topic")
        .with_summary("rust")
        .with_prop("name", Ipld::String("rust".into()));
    let alice = Node::new(NodeId::new_v7(), "Person")
        .with_summary("alice")
        .with_prop("name", Ipld::String("alice".into()));
    let other = Node::new(NodeId::new_v7(), "Doc").with_summary("other");

    let mut tx = repo.start_transaction();
    let doc1_cid = tx.add_node(&doc1).unwrap();
    let doc1_emb = f32_embed("m", &[1.0, 0.0]);
    tx.set_embedding(doc1_cid, doc1_emb.model.clone(), doc1_emb)
        .unwrap();
    tx.add_node(&topic).unwrap();
    tx.add_node(&alice).unwrap();
    let other_cid = tx.add_node(&other).unwrap();
    let other_emb = f32_embed("m", &[0.0, 1.0]);
    tx.set_embedding(other_cid, other_emb.model.clone(), other_emb)
        .unwrap();
    // Doc1 -> Topic (outgoing from Doc1's POV).
    tx.add_edge(&Edge::new(EdgeId::new_v7(), "tagged", doc1.id, topic.id))
        .unwrap();
    // Alice -> Doc1 (incoming to Doc1).
    tx.add_edge(&Edge::new(EdgeId::new_v7(), "authored", alice.id, doc1.id))
        .unwrap();
    let repo = tx.commit("t", "seed").unwrap();

    let both = repo
        .retrieve()
        .vector("m", vec![1.0, 0.0])
        .with_graph_expand(GraphExpand::new().with_both_directions())
        .execute()
        .unwrap();
    let both_ids: Vec<_> = both.items.iter().map(|i| i.node.id).collect();
    assert!(
        both_ids.contains(&topic.id),
        "Both-direction expand must reach Topic (via outgoing); got {both_ids:?}"
    );
    assert!(
        both_ids.contains(&alice.id),
        "Both-direction expand must reach Alice (via incoming); got {both_ids:?}"
    );
    let topic_hits = both_ids.iter().filter(|id| **id == topic.id).count();
    let alice_hits = both_ids.iter().filter(|id| **id == alice.id).count();
    assert_eq!(topic_hits, 1, "topic must not be double-counted");
    assert_eq!(alice_hits, 1, "alice must not be double-counted");
}

/// Commit CID determinism under edge insertion order. The
/// 22afef8 deferred-work note explicitly called this out: dual
/// adjacency trees must not leak insertion order into the
/// canonical commit. Two transactions that add the same edges
/// in a different order must produce byte-identical commit
/// CIDs - any drift here would make the content-addressed repo
/// non-reproducible.
#[test]
fn dual_adjacency_commit_cid_is_order_independent() {
    // Shape: a -E1-> b, a -E2-> c, c -E3-> b (so both outgoing
    // and incoming buckets get multiple entries for at least one
    // node - `a` fans out to two dsts, `b` fans in from two srcs).
    // Build the same commit twice, flipping the add order.
    //
    // Commit metadata (time, change_id, author, message) is pinned
    // via CommitOptions so the only variable across the two paths
    // is the add-order. A CID mismatch here would prove dual-
    // adjacency leaked insertion order into the canonical commit.
    let a_id = NodeId::from_bytes_raw([0x0A; 16]);
    let b_id = NodeId::from_bytes_raw([0x0B; 16]);
    let c_id = NodeId::from_bytes_raw([0x0C; 16]);
    let e1_id = EdgeId::from_bytes_raw([0xE1; 16]);
    let e2_id = EdgeId::from_bytes_raw([0xE2; 16]);
    let e3_id = EdgeId::from_bytes_raw([0xE3; 16]);
    let fixed_change_id = ChangeId::from_bytes_raw([0x77; 16]);
    let fixed_time: u64 = 1_700_000_000_000_000;

    let mk_node = |id: NodeId, ntype: &str, name: &str| {
        Node::new(id, ntype).with_prop("name", Ipld::String(name.into()))
    };
    let opts = || {
        CommitOptions::new("alice", "seq")
            .with_time_micros(fixed_time)
            .with_change_id(fixed_change_id)
    };

    // Sequence 1: nodes a,b,c; edges e1,e2,e3.
    let (bs1, ohs1) = stores();
    let repo1 = ReadonlyRepo::init(bs1, ohs1).unwrap();
    let mut tx1 = repo1.start_transaction();
    tx1.add_node(&mk_node(a_id, "N", "a")).unwrap();
    tx1.add_node(&mk_node(b_id, "N", "b")).unwrap();
    tx1.add_node(&mk_node(c_id, "N", "c")).unwrap();
    tx1.add_edge(&Edge::new(e1_id, "points", a_id, b_id))
        .unwrap();
    tx1.add_edge(&Edge::new(e2_id, "points", a_id, c_id))
        .unwrap();
    tx1.add_edge(&Edge::new(e3_id, "points", c_id, b_id))
        .unwrap();
    let repo1 = tx1.commit_opts(opts()).unwrap();
    let head1 = repo1.view().heads.first().expect("commit landed").clone();

    // Sequence 2: nodes c,a,b; edges e3,e1,e2 (different order,
    // same final edge set with the same ids).
    let (bs2, ohs2) = stores();
    let repo2 = ReadonlyRepo::init(bs2, ohs2).unwrap();
    let mut tx2 = repo2.start_transaction();
    tx2.add_node(&mk_node(c_id, "N", "c")).unwrap();
    tx2.add_node(&mk_node(a_id, "N", "a")).unwrap();
    tx2.add_node(&mk_node(b_id, "N", "b")).unwrap();
    tx2.add_edge(&Edge::new(e3_id, "points", c_id, b_id))
        .unwrap();
    tx2.add_edge(&Edge::new(e1_id, "points", a_id, b_id))
        .unwrap();
    tx2.add_edge(&Edge::new(e2_id, "points", a_id, c_id))
        .unwrap();
    let repo2 = tx2.commit_opts(opts()).unwrap();
    let head2 = repo2.view().heads.first().expect("commit landed").clone();

    assert_eq!(
        head1, head2,
        "dual-adjacency build must be order-independent; \
         got seq1={head1} vs seq2={head2}"
    );
}

/// `render_node_with_adjacency` emits both `Outgoing:` and
/// `Incoming:` blocks for a hub node with edges in each
/// direction. Guards that the opt-in adjacency render wires
/// both adjacency trees, not just one.
#[test]
fn render_node_with_adjacency_shows_incoming_block_for_hub() {
    use mnem_core::retrieve::render_node_with_adjacency;
    // Shape: hub H is written BY alice (incoming) and tagged AS
    // rust (outgoing). Rendering H with adjacency must show
    // both sides.
    let (bs, ohs) = stores();
    let repo = mnem_core::repo::ReadonlyRepo::init(bs, ohs).unwrap();
    let hub = Node::new(NodeId::new_v7(), "Doc")
        .with_summary("hub body")
        .with_prop("name", Ipld::String("hub".into()));
    let topic = Node::new(NodeId::new_v7(), "Topic").with_prop("name", Ipld::String("rust".into()));
    let alice =
        Node::new(NodeId::new_v7(), "Person").with_prop("name", Ipld::String("alice".into()));

    let mut tx = repo.start_transaction();
    tx.add_node(&hub).unwrap();
    tx.add_node(&topic).unwrap();
    tx.add_node(&alice).unwrap();
    // hub -tagged-> topic  (outgoing from hub)
    tx.add_edge(&Edge::new(EdgeId::new_v7(), "tagged", hub.id, topic.id))
        .unwrap();
    // alice -authored-> hub  (incoming to hub)
    tx.add_edge(&Edge::new(EdgeId::new_v7(), "authored", alice.id, hub.id))
        .unwrap();
    let repo = tx.commit("t", "seed").unwrap();

    let rendered = render_node_with_adjacency(&hub, &repo, 8);
    assert!(
        rendered.contains("Outgoing:"),
        "hub render must contain Outgoing: block; got:\n{rendered}"
    );
    assert!(
        rendered.contains("Incoming:"),
        "hub render must contain Incoming: block; got:\n{rendered}"
    );
    assert!(
        rendered.contains("tagged ->"),
        "outgoing 'tagged ->' line missing; got:\n{rendered}"
    );
    assert!(
        rendered.contains("authored <-"),
        "incoming 'authored <-' line missing; got:\n{rendered}"
    );

    // Leaf node (no adjacency) must render WITHOUT empty blocks.
    let leaf = Node::new(NodeId::new_v7(), "Doc").with_summary("no edges");
    let leaf_rendered = render_node_with_adjacency(&leaf, &repo, 8);
    assert!(
        !leaf_rendered.contains("Outgoing:") && !leaf_rendered.contains("Incoming:"),
        "leaf with no edges must NOT emit adjacency headers; got:\n{leaf_rendered}"
    );
}
