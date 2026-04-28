//! End-to-end M8 smoke test: init + transactional commits via the
//! `ReadonlyRepo` / `Transaction` facade.
//!
//! 1. `ReadonlyRepo::init` bootstraps an empty repo (root op, empty view).
//! 2. First transaction adds Alice + Bob, commits.
//! 3. Second transaction adds an edge Alice-knows-Bob, sets a ref,
//!    commits.
//! 4. Third transaction removes Bob, commits.
//! 5. Reopen via `ReadonlyRepo::open` and verify the final state.
//!
//! Run:
//!
//! ```console
//! cargo run -p mnem-core --example repo_init_commit
//! ```
//!
//! Output captured to `/tmp/mnem-test/repo_init_commit.out`.

use std::sync::Arc;

use ipld_core::ipld::Ipld;
use mnem_core::id::{EdgeId, NodeId};
use mnem_core::objects::{Edge, Node, RefTarget};
use mnem_core::repo::ReadonlyRepo;
use mnem_core::store::{Blockstore, MemoryBlockstore, MemoryOpHeadsStore, OpHeadsStore};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("# mnem M8 smoke test: ReadonlyRepo + Transaction");
    println!("# mnem-core: {}", mnem_core::VERSION);
    println!();

    // Shared stores held by Arc so the facade API can clone cheaply.
    let bs: Arc<dyn Blockstore> = Arc::new(MemoryBlockstore::new());
    let ohs: Arc<dyn OpHeadsStore> = Arc::new(MemoryOpHeadsStore::new());

    // ---------- init ----------
    let repo0 = ReadonlyRepo::init(bs.clone(), ohs.clone())?;
    println!("init:");
    println!("  op_id        = {}", repo0.op_id());
    println!(
        "  head_commit  = {:?}",
        repo0.head_commit().map(|c| &c.message)
    );
    println!("  op-heads len = {}", ohs.current()?.len());
    println!();

    // ---------- commit 1: add Alice + Bob ----------
    let alice_id = NodeId::new_v7();
    let alice = Node::new(alice_id, "Person").with_prop("name", Ipld::String("Alice".into()));
    let bob_id = NodeId::new_v7();
    let bob = Node::new(bob_id, "Person").with_prop("name", Ipld::String("Bob".into()));

    let mut tx1 = repo0.start_transaction();
    tx1.add_node(&alice)?;
    tx1.add_node(&bob)?;
    let repo1 = tx1.commit("alice@example.org", "add Alice and Bob")?;
    println!("commit 1: add Alice and Bob");
    println!("  op_id       = {}", repo1.op_id());
    println!(
        "  head commit = {} ({})",
        repo1.head_commit().unwrap().change_id,
        repo1.head_commit().unwrap().message
    );

    // ---------- commit 2: add edge + set refs/heads/main ----------
    let edge = Edge::new(EdgeId::new_v7(), "knows", alice_id, bob_id);
    let mut tx2 = repo1.start_transaction();
    tx2.add_edge(&edge)?;
    let head_cid = repo1.view().heads[0].clone();
    tx2.update_ref("refs/heads/main", Some(RefTarget::normal(head_cid)));
    let repo2 = tx2.commit("alice@example.org", "add knows edge, set main")?;
    println!("commit 2: add edge, set main");
    println!("  op_id        = {}", repo2.op_id());
    println!(
        "  head parents = {} (chained from commit 1)",
        repo2.head_commit().unwrap().parents.len()
    );

    // ---------- commit 3: remove Bob ----------
    let mut tx3 = repo2.start_transaction();
    tx3.remove_node(bob_id);
    let repo3 = tx3.commit("alice@example.org", "remove Bob")?;
    println!("commit 3: remove Bob");
    println!("  op_id = {}", repo3.op_id());
    println!();

    // ---------- verify final state ----------
    println!("## final state (from reopen):");
    let reopened = ReadonlyRepo::open(bs, ohs.clone())?;
    assert_eq!(
        reopened.op_id(),
        repo3.op_id(),
        "reopen must land on latest op"
    );
    println!("  op_id          = {}", reopened.op_id());
    let head = reopened.head_commit().unwrap();
    println!("  head.message   = {:?}", head.message);
    println!(
        "  head.parents   = {} (chains all the way back)",
        head.parents.len()
    );

    // Alice should still be there; Bob should be gone.
    let alice_after = reopened.lookup_node(&alice_id)?;
    assert!(alice_after.is_some(), "Alice must survive");
    let bob_after = reopened.lookup_node(&bob_id)?;
    assert!(bob_after.is_none(), "Bob must be removed");
    println!("  alice present  = true");
    println!("  bob present    = false");
    println!(
        "  main ref       = {:?}",
        reopened.view().refs.get("refs/heads/main")
    );

    println!();
    println!("# op-heads trajectory:");
    println!("  final op-heads count = {}", ohs.current()?.len());
    println!("  final op-head        = {}", ohs.current()?[0]);

    println!();
    println!("# M8 smoke test: ok");
    Ok(())
}
