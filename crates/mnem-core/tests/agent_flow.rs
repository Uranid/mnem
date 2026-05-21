//! E2E agent-flow integration test for the 0.3 agent-support surface.
//!
//! Exercises the three primitives shipped in this track as a cohesive
//! workflow:
//!
//! - [`Transaction::commit_memory`] - ergonomic node write with
//!   auto-stamped temporal metadata.
//! - [`Retriever::where_created_after`] /
//!   [`Retriever::where_created_before`] - temporal-range gate on the
//!   reserved props.
//! - [`Transaction::tombstone_node`] +
//!   [`Retriever::include_tombstoned`] - forget / audit semantics.
//! - [`ReadonlyRepo::incoming_edges`] - dual-adjacency back-index
//!   lookup.
//!
//! These tests pin the contracts an agent would care about if it were
//! writing memory via `commit_memory`, reading via a filtered
//! `retrieve`, and later revoking or chasing provenance. Unit coverage
//! of each primitive lives on its own module; this file is the
//! stitched-together happy path plus the obvious forget / range-filter
//! fail modes.
//!
//! [`Transaction::commit_memory`]: mnem_core::repo::Transaction::commit_memory
//! [`Retriever::where_created_after`]: mnem_core::retrieve::Retriever::where_created_after
//! [`Retriever::where_created_before`]: mnem_core::retrieve::Retriever::where_created_before
//! [`Transaction::tombstone_node`]: mnem_core::repo::Transaction::tombstone_node
//! [`Retriever::include_tombstoned`]: mnem_core::retrieve::Retriever::include_tombstoned
//! [`ReadonlyRepo::incoming_edges`]: mnem_core::repo::ReadonlyRepo::incoming_edges

use std::sync::Arc;

use ipld_core::ipld::Ipld;

use mnem_core::id::{EdgeId, NodeId};
use mnem_core::objects::Edge;
use mnem_core::repo::ReadonlyRepo;
use mnem_core::store::{Blockstore, MemoryBlockstore, MemoryOpHeadsStore, OpHeadsStore};

fn fresh_repo() -> ReadonlyRepo {
    let bs: Arc<dyn Blockstore> = Arc::new(MemoryBlockstore::new());
    let ohs: Arc<dyn OpHeadsStore> = Arc::new(MemoryOpHeadsStore::new());
    ReadonlyRepo::init(bs, ohs).expect("fresh repo init")
}

#[test]
fn commit_memory_writes_node_with_auto_stamped_temporal_props() {
    // Contract: `commit_memory` returns a fresh NodeId, stores the
    // node under the given ntype + summary, and auto-stamps
    // `mnem:created_at` / `mnem:updated_at` with a positive
    // microseconds-since-epoch integer.
    let repo = fresh_repo();

    let mut tx = repo.start_transaction();
    let id = tx
        .commit_memory(
            "Note",
            "morning meeting with alice",
            [("topic".to_string(), Ipld::String("ops".into()))],
        )
        .expect("commit_memory ok");
    let repo = tx.commit("agent", "note").expect("commit ok");

    let node = repo
        .lookup_node(&id)
        .expect("lookup ok")
        .expect("node present");
    assert_eq!(node.ntype, "Note");
    assert_eq!(node.summary.as_deref(), Some("morning meeting with alice"));
    assert_eq!(
        node.props.get("topic"),
        Some(&Ipld::String("ops".into())),
        "caller-supplied prop must round-trip"
    );
    let created = match node.props.get("mnem:created_at") {
        Some(Ipld::Integer(n)) => *n,
        other => panic!("expected Integer for mnem:created_at, got {other:?}"),
    };
    let updated = match node.props.get("mnem:updated_at") {
        Some(Ipld::Integer(n)) => *n,
        other => panic!("expected Integer for mnem:updated_at, got {other:?}"),
    };
    assert!(
        created > 0 && updated > 0,
        "temporal stamps must be positive (got created={created} updated={updated})"
    );
    assert_eq!(
        created, updated,
        "on first write, created_at and updated_at coincide"
    );
}

#[test]
fn retrieve_surfaces_commit_memory_node_without_filter() {
    // Contract: a node written via commit_memory is visible through
    // the standard retrieve path (label gate) without any temporal
    // filter configured.
    let repo = fresh_repo();
    let mut tx = repo.start_transaction();
    let id = tx
        .commit_memory(
            "Note",
            "morning meeting with alice",
            [("topic".to_string(), Ipld::String("ops".into()))],
        )
        .unwrap();
    let repo = tx.commit("agent", "note").unwrap();

    let result = repo
        .retrieve()
        .label("Note")
        .execute()
        .expect("retrieve ok");
    let ids: Vec<_> = result.items.iter().map(|i| i.node.id).collect();
    assert!(
        ids.contains(&id),
        "unfiltered retrieve must surface the committed node, got ids={ids:?}"
    );
}

#[test]
fn temporal_filter_excludes_future_bound_past() {
    // Contract: `where_created_before(t_past)` with `t_past = 1`
    // (microsecond 1, effectively "before the epoch began for this
    // test") drops every node that has a real `mnem:created_at`.
    let repo = fresh_repo();
    let mut tx = repo.start_transaction();
    let id = tx
        .commit_memory(
            "Note",
            "morning meeting with alice",
            [("topic".to_string(), Ipld::String("ops".into()))],
        )
        .unwrap();
    let repo = tx.commit("agent", "note").unwrap();

    // created_before(1) is exclusive: every real-epoch timestamp is
    // >= 1, so the filter excludes every node with a stamp.
    let result = repo
        .retrieve()
        .label("Note")
        .where_created_before(1)
        .execute()
        .expect("retrieve ok");
    let ids: Vec<_> = result.items.iter().map(|i| i.node.id).collect();
    assert!(
        !ids.contains(&id),
        "where_created_before(past_t) must exclude the node, got ids={ids:?}"
    );
}

#[test]
fn temporal_filter_includes_anything_after_zero() {
    // Contract: `where_created_after(0)` inclusive lower bound at
    // zero surfaces every node that has any non-negative stamp (so,
    // every node `commit_memory` has written). This is the
    // "everything since the beginning of time" shape agents use for
    // unbounded-lower queries.
    let repo = fresh_repo();
    let mut tx = repo.start_transaction();
    let id = tx
        .commit_memory(
            "Note",
            "morning meeting with alice",
            [("topic".to_string(), Ipld::String("ops".into()))],
        )
        .unwrap();
    let repo = tx.commit("agent", "note").unwrap();

    let result = repo
        .retrieve()
        .label("Note")
        .where_created_after(0)
        .execute()
        .expect("retrieve ok");
    let ids: Vec<_> = result.items.iter().map(|i| i.node.id).collect();
    assert!(
        ids.contains(&id),
        "where_created_after(0) must surface the node, got ids={ids:?}"
    );
}

#[test]
fn tombstone_excludes_node_and_include_tombstoned_surfaces_it() {
    // Contract: `tombstone_node` + commit drops the node from a
    // default retrieve. `include_tombstoned(true)` restores it for
    // audit / debug.
    let repo = fresh_repo();
    let mut tx = repo.start_transaction();
    let id = tx
        .commit_memory(
            "Note",
            "morning meeting with alice",
            [("topic".to_string(), Ipld::String("ops".into()))],
        )
        .unwrap();
    let repo = tx.commit("agent", "note").unwrap();

    let mut tx2 = repo.start_transaction();
    tx2.tombstone_node(id, "user revoked").unwrap();
    let repo = tx2.commit("agent", "revoke").unwrap();

    // Default retrieve: the node must not appear.
    let result = repo
        .retrieve()
        .label("Note")
        .execute()
        .expect("retrieve ok");
    let ids: Vec<_> = result.items.iter().map(|i| i.node.id).collect();
    assert!(
        !ids.contains(&id),
        "tombstoned node must be excluded by default, got ids={ids:?}"
    );

    // Audit path: include_tombstoned(true) brings it back.
    let result = repo
        .retrieve()
        .label("Note")
        .include_tombstoned(true)
        .execute()
        .expect("retrieve ok");
    let ids: Vec<_> = result.items.iter().map(|i| i.node.id).collect();
    assert!(
        ids.contains(&id),
        "include_tombstoned(true) must surface the tombstoned node, got ids={ids:?}"
    );
}

#[test]
fn anchor_excluded_from_retrieve_by_default_and_surfaced_under_include_system() {
    // Contract: a node whose id matches the system anchor must NOT
    // appear in a default retrieve, no matter what other live nodes
    // are present. `include_system(true)` restores it for audit.
    //
    // Repros the user-reported bug where the `mnem init` anchor
    // (node id 00000000-0000-7000-8000-6d6e656d0001) showed up with
    // a low score in every retrieve after the first ingest.
    use mnem_core::anchor::anchor_node_id;
    use mnem_core::objects::Node;

    let repo = fresh_repo();
    let mut tx = repo.start_transaction();
    let anchor = Node::new(anchor_node_id(), "Meta");
    tx.add_node(&anchor).expect("add anchor");
    let real_id = tx
        .commit_memory("Note", "live content the user actually wrote", [])
        .unwrap();
    let repo = tx.commit("agent", "init + content").unwrap();

    // Default retrieve: anchor is hidden, real note is visible.
    let result = repo
        .retrieve()
        .label("Note")
        .execute()
        .expect("retrieve ok");
    let ids: Vec<_> = result.items.iter().map(|i| i.node.id).collect();
    assert!(
        !ids.contains(&anchor_node_id()),
        "anchor must be excluded by default, got ids={ids:?}"
    );
    assert!(
        ids.contains(&real_id),
        "live note must still surface, got ids={ids:?}"
    );

    // Also check the Meta-labelled retrieve directly: the anchor is a
    // Meta node so without the system filter it would dominate the
    // results. With the filter on (default) we expect zero hits.
    let result_meta = repo.retrieve().label("Meta").execute().expect("retrieve ok");
    let ids_meta: Vec<_> = result_meta.items.iter().map(|i| i.node.id).collect();
    assert!(
        ids_meta.is_empty(),
        "default retrieve of label=Meta must drop the anchor entirely, got ids={ids_meta:?}"
    );

    // Audit path: include_system(true) brings it back.
    let result = repo
        .retrieve()
        .label("Meta")
        .include_system(true)
        .execute()
        .expect("retrieve ok");
    let ids: Vec<_> = result.items.iter().map(|i| i.node.id).collect();
    assert!(
        ids.contains(&anchor_node_id()),
        "include_system(true) must surface the anchor for audit, got ids={ids:?}"
    );
}

#[test]
fn query_excludes_anchor_by_default_and_surfaces_under_include_system() {
    // Same contract as the retriever test above, but at the Query
    // level (the structured `mnem query` / MCP `mnem_list_nodes`
    // surface). Two filter sites in `Query::execute` need to agree:
    // the indexed-label cursor path and the streaming-fallback path.
    use mnem_core::anchor::anchor_node_id;
    use mnem_core::objects::Node;

    let repo = fresh_repo();
    let mut tx = repo.start_transaction();
    tx.add_node(&Node::new(anchor_node_id(), "Meta"))
        .expect("add anchor");
    let _real = tx
        .commit_memory("Note", "user-authored fact", [])
        .unwrap();
    let repo = tx.commit("agent", "init + content").unwrap();

    // Default Query for ntype=Meta: zero hits (only the anchor lives
    // under that label, and the system filter drops it).
    let hits = repo.query().label("Meta").execute().expect("query ok");
    let ids: Vec<_> = hits.iter().map(|h| h.node.id).collect();
    assert!(
        ids.is_empty(),
        "default Query for label=Meta must drop the anchor, got ids={ids:?}"
    );

    // include_system(true) restores it.
    let hits = repo
        .query()
        .label("Meta")
        .include_system(true)
        .execute()
        .expect("query ok");
    let ids: Vec<_> = hits.iter().map(|h| h.node.id).collect();
    assert!(
        ids.contains(&anchor_node_id()),
        "include_system(true) must surface the anchor, got ids={ids:?}"
    );

    // No-label streaming fallback: same guarantee on the alternate
    // execute branch.
    let hits = repo.query().execute().expect("query ok");
    let ids: Vec<_> = hits.iter().map(|h| h.node.id).collect();
    assert!(
        !ids.contains(&anchor_node_id()),
        "default no-label Query must also drop the anchor, got ids={ids:?}"
    );
}

#[test]
fn edge_between_two_notes_surfaces_on_incoming_edges() {
    // Contract: an edge added between two commit_memory-written
    // nodes is reachable via the dual-adjacency back-index
    // (`incoming_edges(dst)`), which is what provenance walks and
    // `graph_expand(direction = Incoming)` rely on.
    let repo = fresh_repo();
    let mut tx = repo.start_transaction();
    let src_id = tx
        .commit_memory(
            "Note",
            "source note",
            [("role".to_string(), Ipld::String("src".into()))],
        )
        .unwrap();
    let dst_id = tx
        .commit_memory(
            "Note",
            "dest note",
            [("role".to_string(), Ipld::String("dst".into()))],
        )
        .unwrap();
    let edge = Edge::new(EdgeId::new_v7(), "references", src_id, dst_id);
    let edge_id = edge.id;
    tx.add_edge(&edge).unwrap();
    let repo = tx.commit("agent", "notes + edge").unwrap();

    let incoming = repo
        .incoming_edges(&dst_id, None)
        .expect("incoming_edges ok");
    assert!(
        incoming.iter().any(|e| e.id == edge_id
            && e.src == src_id
            && e.dst == dst_id
            && e.etype == "references"),
        "incoming_edges(dst) must surface the src->dst edge, got {incoming:#?}"
    );
}

#[test]
fn temporal_filter_is_lenient_on_nodes_without_reserved_props() {
    // Contract: a node lacking both reserved temporal props (written
    // via the lower-level `add_node` path) passes every temporal
    // check - callers want to layer the filter onto legacy repos
    // without nuking the result set.
    let repo = fresh_repo();
    let mut tx = repo.start_transaction();
    // Bypass commit_memory so no auto-stamp happens.
    let legacy = mnem_core::objects::Node::new(NodeId::new_v7(), "Note").with_summary("no stamp");
    tx.add_node(&legacy).unwrap();
    let repo = tx.commit("agent", "legacy").unwrap();

    // Every bound: the node must still surface because it has no
    // reserved prop to gate against.
    let result = repo
        .retrieve()
        .label("Note")
        .where_created_after(10_000_000_000_000)
        .where_created_before(10_000_000_000_001)
        .where_updated_after(10_000_000_000_000)
        .where_updated_before(10_000_000_000_001)
        .execute()
        .expect("retrieve ok");
    let ids: Vec<_> = result.items.iter().map(|i| i.node.id).collect();
    assert!(
        ids.contains(&legacy.id),
        "legacy node without reserved props must pass every temporal check (lenient rule), got ids={ids:?}"
    );
}
