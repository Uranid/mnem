//! End-to-end smoke test exercising M6+M7:
//!
//! 1. Build node, edge, schema Prolly trees.
//! 2. Create a [`Commit`] pointing at those tree roots.
//! 3. Create a [`View`] listing the commit as head and setting
//!    `refs/heads/main`.
//! 4. Create an [`Operation`] wrapping the view.
//! 5. Add the op to the [`OpHeadsStore`].
//! 6. Read back the entire DAG: op-heads → op → view → commit →
//!    node tree → Alice.
//!
//! Run:
//!
//! ```console
//! cargo run -p mnem-core --example commit_and_op_log
//! ```
//!
//! Operator redirects output to `/tmp/mnem-test/commit_and_op_log.out`.

use std::time::{SystemTime, UNIX_EPOCH};

use ipld_core::ipld::Ipld;
use mnem_core::codec::{from_canonical_bytes, hash_to_cid};
use mnem_core::id::{ChangeId, EdgeId, NodeId};
use mnem_core::objects::{Commit, Edge, Node, Operation, RefTarget, View};
use mnem_core::prolly::{self, ProllyKey};
use mnem_core::store::{Blockstore, MemoryBlockstore, MemoryOpHeadsStore, OpHeadsStore};

fn now_micros() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_micros() as u64
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("# mnem M6+M7 smoke test: Commit + View + Operation + OpHeads");
    println!("# mnem-core: {}", mnem_core::VERSION);
    println!();

    let store = MemoryBlockstore::new();
    let op_heads = MemoryOpHeadsStore::new();

    // ----- Build two Person nodes, one 'knows' edge -----
    let alice_id = NodeId::new_v7();
    let alice = Node::new(alice_id, "Person").with_prop("name", Ipld::String("Alice".into()));
    let (alice_bytes, alice_cid) = hash_to_cid(&alice)?;
    store.put(alice_cid.clone(), alice_bytes)?;

    let bob_id = NodeId::new_v7();
    let bob = Node::new(bob_id, "Person").with_prop("name", Ipld::String("Bob".into()));
    let (bob_bytes, bob_cid) = hash_to_cid(&bob)?;
    store.put(bob_cid.clone(), bob_bytes)?;

    let edge = Edge::new(EdgeId::new_v7(), "knows", alice_id, bob_id);
    let (edge_bytes, edge_cid) = hash_to_cid(&edge)?;
    store.put(edge_cid.clone(), edge_bytes)?;

    // ----- Build the node / edge / schema Prolly trees -----
    let mut node_entries: Vec<(ProllyKey, _)> = vec![
        (ProllyKey::from(alice_id), alice_cid.clone()),
        (ProllyKey::from(bob_id), bob_cid),
    ];
    node_entries.sort_by_key(|e| e.0);
    let nodes_root = prolly::build_tree(&store, node_entries)?;

    let edge_entries = vec![(ProllyKey::from(edge.id), edge_cid)];
    let edges_root = prolly::build_tree(&store, edge_entries)?;

    // Empty schema tree (no schema in this smoke test)
    let schema_root = prolly::build_tree(&store, std::iter::empty())?;

    println!("  nodes tree:  {nodes_root}");
    println!("  edges tree:  {edges_root}");
    println!("  schema tree: {schema_root}");
    println!();

    // ----- Commit -----
    let commit = Commit::new(
        ChangeId::new_v7(),
        nodes_root,
        edges_root,
        schema_root,
        "alice@example.org",
        now_micros(),
        "Add Alice, Bob, and the knows edge",
    )
    .with_agent("agent:claude")
    .with_task("task:001");
    let (commit_bytes, commit_cid) = hash_to_cid(&commit)?;
    store.put(commit_cid.clone(), commit_bytes)?;
    println!("  commit:      {commit_cid}");

    // ----- View -----
    let view = View::new()
        .with_head(commit_cid.clone())
        .with_ref("refs/heads/main", RefTarget::normal(commit_cid));
    let (view_bytes, view_cid) = hash_to_cid(&view)?;
    store.put(view_cid.clone(), view_bytes)?;
    println!("  view:        {view_cid}");

    // ----- Operation -----
    let op = Operation::new(
        view_cid,
        "alice@example.org",
        now_micros(),
        "commit: Add Alice, Bob, and the knows edge",
    )
    .with_agent("agent:claude")
    .with_task("task:001")
    .with_host("laptop");
    let (op_bytes, op_cid) = hash_to_cid(&op)?;
    store.put(op_cid.clone(), op_bytes)?;
    println!("  operation:   {op_cid}");

    // ----- Advance op-heads -----
    op_heads.update(op_cid, &[])?;
    println!();

    // ----- Reverse walk: op-heads → op → view → commit → node tree → Alice -----
    println!("## reverse walk from op-heads");
    let heads = op_heads.current()?;
    assert_eq!(heads.len(), 1, "expected single head");
    println!("  1 head:        {}", heads[0]);

    let op_bytes = store.get(&heads[0])?.expect("op exists");
    let op2: Operation = from_canonical_bytes(&op_bytes)?;
    assert_eq!(op2.description, op.description);
    println!("  op -> view:    {}", op2.view);

    let view_bytes = store.get(&op2.view)?.expect("view exists");
    let view2: View = from_canonical_bytes(&view_bytes)?;
    assert_eq!(view2.heads.len(), 1);
    println!("  view -> head:  {}", view2.heads[0]);

    let commit_bytes = store.get(&view2.heads[0])?.expect("commit exists");
    let commit2: Commit = from_canonical_bytes(&commit_bytes)?;
    assert_eq!(commit2.change_id, commit.change_id);
    println!("  commit -> node tree: {}", commit2.nodes);

    // Resolve Alice in the node Prolly tree by her stable NodeId.
    let found = prolly::lookup(&store, &commit2.nodes, &ProllyKey::from(alice_id))?;
    assert_eq!(found, Some(alice_cid.clone()));
    println!("  node tree[alice] = {alice_cid}");

    // Resolve the actual Alice Node bytes and decode.
    let alice_bytes_again = store.get(&alice_cid)?.expect("alice exists");
    let alice_decoded: Node = from_canonical_bytes(&alice_bytes_again)?;
    assert_eq!(alice_decoded, alice);
    let name = alice_decoded.props.get("name").unwrap();
    println!("  alice.name =   {name:?}");

    println!();
    println!(
        "# end-to-end op-log walk: ok (touched {} objects)",
        store.len()
    );
    Ok(())
}
