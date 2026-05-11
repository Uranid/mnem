//! M11 smoke test: CAS primitive on refs + linearize mode on commit.
//!
//! Exercises SPEC §6.4 (CAS via `ReadonlyRepo::update_ref`) and
//! §6.5 (`Transaction::commit_opts` with `linearize: true`).
//!
//! Run:
//!
//! ```console
//! cargo run -p mnem-core --example cas_and_linearize
//! ```
//!
//! Output captured to `/tmp/mnem-test/cas_and_linearize.out`.

use std::sync::Arc;

use ipld_core::ipld::Ipld;
use mnem_core::error::{Error, RepoError};
use mnem_core::id::{CODEC_RAW, Cid, Multihash, NodeId};
use mnem_core::objects::{Node, RefTarget};
use mnem_core::repo::{CommitOptions, ReadonlyRepo};
use mnem_core::store::{Blockstore, MemoryBlockstore, MemoryOpHeadsStore, OpHeadsStore};

fn cid(seed: u32) -> Cid {
    Cid::new(CODEC_RAW, Multihash::sha2_256(&seed.to_be_bytes()))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let bs: Arc<dyn Blockstore> = Arc::new(MemoryBlockstore::new());
    let ohs: Arc<dyn OpHeadsStore> = Arc::new(MemoryOpHeadsStore::new());
    let repo = ReadonlyRepo::init(bs.clone(), ohs.clone())?;

    println!("# mnem M11: CAS + linearize smoke test");
    println!("# mnem-core: {}", mnem_core::VERSION);
    println!();

    // ---------- CAS: create a ref (expected_prev = None) ----------
    let v1 = RefTarget::normal(cid(1));
    let r1 = repo.update_ref(
        "refs/heads/main",
        None,
        Some(v1.clone()),
        "alice@example.org",
    )?;
    println!("CAS create: refs/heads/main -> {v1:?}");

    // ---------- CAS stale: expected_prev doesn't match ----------
    let wrong = RefTarget::normal(cid(99));
    let stale = r1.update_ref(
        "refs/heads/main",
        Some(&wrong),
        Some(RefTarget::normal(cid(2))),
        "alice@example.org",
    );
    match stale {
        Err(Error::Repo(RepoError::Stale)) => {
            println!("CAS stale:  ok (wrong expected_prev rejected)");
        }
        other => panic!("expected Stale, got {other:?}"),
    }

    // ---------- CAS update: correct expected_prev ----------
    let v2 = RefTarget::normal(cid(2));
    let r2 = r1.update_ref(
        "refs/heads/main",
        Some(&v1),
        Some(v2.clone()),
        "alice@example.org",
    )?;
    println!("CAS update: refs/heads/main -> {v2:?}");

    // ---------- CAS delete: pass current, pass None for new ----------
    let r3 = r2.update_ref("refs/heads/main", Some(&v2), None, "alice@example.org")?;
    assert!(!r3.view().refs.contains_key("refs/heads/main"));
    println!("CAS delete: refs/heads/main removed");

    println!();

    // ---------- Linearize: happy path ----------
    let mut tx = r3.start_transaction();
    let alice =
        Node::new(NodeId::new_v7(), "Person").with_prop("name", Ipld::String("Alice".into()));
    tx.add_node(&alice)?;
    let r4 = tx.commit_opts(CommitOptions {
        author: "alice@example.org",
        message: "add Alice (linearized)",
        linearize: true,
        time_micros: None,
        change_id: None,
        agent_id: None,
        task_id: None,
    })?;
    println!("linearize happy path: commit succeeded at {}", r4.op_id());

    // ---------- Linearize: stale base rejected ----------
    // Open TWO transactions against r4. One commits (moves op-heads
    // forward). The other tries to commit in linearize mode - and fails.
    let stale_tx_base = r4.clone();
    let mut stale_tx = stale_tx_base.start_transaction();
    stale_tx.add_node(&Node::new(NodeId::new_v7(), "Ghost"))?;

    let mut winner_tx = r4.start_transaction();
    winner_tx.add_node(&Node::new(NodeId::new_v7(), "Bob"))?;
    let r5 = winner_tx.commit("alice@example.org", "concurrent: add Bob")?;
    println!("concurrent writer advanced op-head to {}", r5.op_id());

    let outcome = stale_tx.commit_opts(CommitOptions {
        author: "alice@example.org",
        message: "from stale base",
        linearize: true,
        time_micros: None,
        change_id: None,
        agent_id: None,
        task_id: None,
    });
    match outcome {
        Err(Error::Repo(RepoError::Stale)) => {
            println!("linearize stale:      ok (stale base rejected)");
        }
        Ok(_) => panic!("linearize commit against stale base should have failed"),
        Err(e) => panic!("unexpected error: {e:?}"),
    }

    // ---------- Non-linearize: stale base still succeeds ----------
    let mut late_tx = r4.start_transaction();
    late_tx.add_node(&Node::new(NodeId::new_v7(), "Carol"))?;
    let late = late_tx.commit("alice@example.org", "late but not linearized")?;
    let heads_now = ohs.current()?;
    println!(
        "non-linearize stale: ok (new head appended; total heads now {})",
        heads_now.len()
    );
    println!("  - the next reader will 3-way-merge these heads (M8.5 feature)");
    drop(late);

    println!();
    println!("# M11 CAS + linearize smoke test: ok");
    Ok(())
}
