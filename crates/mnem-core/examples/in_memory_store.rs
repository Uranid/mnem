//! End-to-end capability smoke test exercising Phase 1 M3 + M4:
//!
//! 1. Construct [`Node`] and [`Edge`] objects with `with_prop` fluent builders.
//! 2. Encode each to canonical DAG-CBOR via `hash_to_cid`.
//! 3. Put them into a [`MemoryBlockstore`].
//! 4. Read them back, decode, assert round-trip equality.
//! 5. Demonstrate idempotent `put`.
//! 6. Print the CID and final store size.
//!
//! Run with:
//!
//! ```console
//! cargo run --example in_memory_store
//! ```
//!
//! Operator redirects output to `/tmp/mnem-test/in_memory_store.out`.

use ipld_core::ipld::Ipld;
use mnem_core::codec::{from_canonical_bytes, hash_to_cid};
use mnem_core::id::{EdgeId, NodeId};
use mnem_core::objects::{Edge, Node};
use mnem_core::store::{Blockstore, MemoryBlockstore};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("# mnem M3 + M4 smoke test: MemoryBlockstore + Node/Edge");
    println!("# mnem-core version: {}", mnem_core::VERSION);
    println!();

    let store = MemoryBlockstore::new();

    // ---------- Alice: a Person node ----------
    let alice = Node::new(NodeId::new_v7(), "Person")
        .with_prop("name", Ipld::String("Alice".into()))
        .with_prop("age", Ipld::Integer(30));
    let (alice_bytes, alice_cid) = hash_to_cid(&alice)?;
    store.put(alice_cid.clone(), alice_bytes)?;
    println!("alice: {alice_cid}");

    assert!(store.has(&alice_cid)?);
    let alice_fetched = store.get(&alice_cid)?.expect("just stored");
    let alice2: Node = from_canonical_bytes(&alice_fetched)?;
    assert_eq!(alice, alice2, "alice round-trip mismatch");
    println!("  round-trip:  ok");

    // ---------- Bob: a Person node ----------
    let bob = Node::new(NodeId::new_v7(), "Person")
        .with_prop("name", Ipld::String("Bob".into()))
        .with_prop("age", Ipld::Integer(25));
    let (bob_bytes, bob_cid) = hash_to_cid(&bob)?;
    store.put(bob_cid.clone(), bob_bytes)?;
    println!("bob:   {bob_cid}");

    // ---------- Edge: alice knows bob ----------
    let knows = Edge::new(EdgeId::new_v7(), "knows", alice.id, bob.id)
        .with_prop("since", Ipld::Integer(2020));
    let (knows_bytes, knows_cid) = hash_to_cid(&knows)?;
    store.put(knows_cid.clone(), knows_bytes)?;
    println!("knows: {knows_cid}");

    // Round-trip the edge
    let knows_fetched = store.get(&knows_cid)?.expect("just stored");
    let knows2: Edge = from_canonical_bytes(&knows_fetched)?;
    assert_eq!(knows, knows2);
    assert_eq!(knows2.src, alice.id);
    assert_eq!(knows2.dst, bob.id);
    println!("  edge endpoints resolve: alice -> bob");

    // ---------- Idempotency: re-put alice, store size unchanged ----------
    let (alice_bytes_again, alice_cid_again) = hash_to_cid(&alice)?;
    assert_eq!(
        alice_cid, alice_cid_again,
        "content-addressing not deterministic"
    );
    store.put(alice_cid_again, alice_bytes_again)?;
    assert_eq!(store.len(), 3, "idempotent put should not grow store");
    println!("idempotent put: ok (store size still {})", store.len());

    // ---------- Kind rejection: decoding an edge as a node must fail ----------
    let edge_bytes = store.get(&knows_cid)?.expect("exists");
    let decode_as_node: Result<Node, _> = from_canonical_bytes(&edge_bytes);
    assert!(decode_as_node.is_err(), "edge should not decode as node");
    println!("_kind rejection: ok (decoding Edge bytes as Node correctly errors)");

    // ---------- Delete is harmless ----------
    let missing_cid = alice_cid;
    store.delete(&missing_cid)?;
    assert!(!store.has(&missing_cid)?);
    store.delete(&missing_cid)?; // second delete is a no-op, not an error
    println!("delete: ok (double-delete tolerated)");

    println!();
    println!("# all M3+M4 smoke tests passed. store = {store:?}");
    Ok(())
}
