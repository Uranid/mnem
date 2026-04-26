//! M10 smoke test: persist a mnem repo to a redb file, close it,
//! reopen it, verify the content survived.
//!
//! Runs against the production embedded-KV backend, which is the sole
//! persistent backend shipped at launch.
//!
//! Run:
//!
//! ```console
//! cargo run -p mnem-backend-redb --example persist
//! ```
//!
//! Operator redirects output to `/tmp/mnem-test/persist_redb.out`.

use std::path::PathBuf;
use std::sync::Arc;

use ipld_core::ipld::Ipld;
use mnem_backend_redb::open_or_init;
use mnem_core::id::{EdgeId, NodeId};
use mnem_core::objects::{Edge, Node, RefTarget};
use mnem_core::repo::ReadonlyRepo;
use mnem_core::store::{Blockstore, OpHeadsStore};

fn db_path() -> PathBuf {
    let p = std::env::temp_dir().join(format!(
        "mnem-redb-persist-{}-{}.redb",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64
    ));
    let _ = std::fs::remove_file(&p);
    p
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = db_path();
    println!("# mnem M10 smoke test: redb embedded-KV persistence");
    println!("# mnem-core: {}", mnem_core::VERSION);
    println!("# db file:   {}", path.display());
    println!();

    let alice_id = NodeId::new_v7();
    let bob_id = NodeId::new_v7();

    let op_at_close: mnem_core::id::Cid;
    {
        let (bs, ohs, file): (Arc<dyn Blockstore>, Arc<dyn OpHeadsStore>, _) = open_or_init(&path)?;
        println!("session 1: open_or_init -> {}", file.display());

        let repo0 = if let Ok(r) = ReadonlyRepo::open(bs.clone(), ohs.clone()) {
            r
        } else {
            let r = ReadonlyRepo::init(bs.clone(), ohs.clone())?;
            println!("  initialised repo, root op = {}", r.op_id());
            r
        };

        // Commit 1: Alice
        let alice = Node::new(alice_id, "Person").with_prop("name", Ipld::String("Alice".into()));
        let mut tx1 = repo0.start_transaction();
        tx1.add_node(&alice)?;
        let r1 = tx1.commit("alice@example.org", "add Alice")?;
        println!("  commit 1: {}", r1.op_id());

        // Commit 2: Bob + edge + main ref
        let bob = Node::new(bob_id, "Person").with_prop("name", Ipld::String("Bob".into()));
        let edge = Edge::new(EdgeId::new_v7(), "knows", alice_id, bob_id);
        let mut tx2 = r1.start_transaction();
        tx2.add_node(&bob)?;
        tx2.add_edge(&edge)?;
        let head_cid = r1.view().heads[0].clone();
        tx2.update_ref("refs/heads/main", Some(RefTarget::normal(head_cid)));
        let r2 = tx2.commit("alice@example.org", "add Bob + knows edge + main ref")?;
        op_at_close = r2.op_id().clone();
        println!("  commit 2: {}", r2.op_id());
    }
    // All redb handles dropped here.

    println!();
    {
        let (bs, ohs, _) = open_or_init(&path)?;
        let repo = ReadonlyRepo::open(bs, ohs)?;
        println!("session 2: reopened at {}", repo.op_id());
        assert_eq!(
            *repo.op_id(),
            op_at_close,
            "op-head must survive close/reopen"
        );
        let head = repo.head_commit().unwrap();
        println!("  head.message:  {:?}", head.message);
        println!("  head.parents:  {}", head.parents.len());
        let alice_back = repo.lookup_node(&alice_id)?.expect("alice persisted");
        let bob_back = repo.lookup_node(&bob_id)?.expect("bob persisted");
        assert_eq!(
            alice_back.props.get("name"),
            Some(&Ipld::String("Alice".into()))
        );
        assert_eq!(
            bob_back.props.get("name"),
            Some(&Ipld::String("Bob".into()))
        );
        println!("  alice (disk):  name = Alice");
        println!("  bob   (disk):  name = Bob");
        assert!(repo.view().refs.contains_key("refs/heads/main"));
        println!("  main ref:      intact");
    }

    let size = std::fs::metadata(&path).map_or(0, |m| m.len());
    println!();
    println!("# redb file size: {} bytes ({} KiB)", size, size / 1024);
    println!("# db file preserved for inspection: {}", path.display());
    println!("# M10 redb persistence smoke test: ok");
    Ok(())
}
