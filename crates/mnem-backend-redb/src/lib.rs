//! # mnem-backend-redb
//!
//! Production [`Blockstore`] + [`OpHeadsStore`] backed by
//! [`redb`](https://github.com/cberner/redb) - a pure-Rust embedded
//! ACID key-value store.
//!
//! A single `.redb` file holds two tables:
//!
//! - `objects`  - `CID bytes → object bytes`. Every Node, Edge, Tree
//!   chunk, Commit, View, Operation is a row.
//! - `op_heads` - set-of-CIDs modelled as `CID bytes → ()` with
//!   presence-as-truth. The current op-heads set is the key-set of
//!   this table.
//!
//! Writes are atomic per-call: each `put` / `delete` / `update` opens a
//! write transaction, mutates, and commits (which fsyncs). No
//! cross-call batching yet; a future refinement can expose a "batch
//! mode" that holds a transaction open across multiple puts to reduce
//! fsync overhead on the `mnem_core::repo::Transaction::commit` hot
//! path.
//!
//! ## Concurrency
//!
//! redb: single-writer, many-reader per database file. Within a
//! process, `Arc<Database>` is safe to share across threads - redb
//! serialises writers internally. Across processes, redb's
//! filesystem-locking protects the file format; concurrent write
//! transactions from two processes will serialise or one will fail.
//!
//! ## Durability
//!
//! `tx.commit()` in redb calls `fsync` on the database file. The Simple
//! backend's atomic-rename story is effectively what redb does
//! internally for each commit - one transaction = one durable
//! state transition.
//!
//! ## Backend choice notes
//!
//! redb is the production embedded backend for exactly these reasons:
//! pure Rust, small dep tree, mmap reads, ACID writes, single-file
//! persistence. Note redb majors break on-disk format (1 → 2 → 3 → 4).
//! We pin the major in the crate's Cargo dependency and document the
//! migration path when we bump.

#![forbid(unsafe_code)]
#![deny(missing_docs)]

pub mod blockstore;
pub mod knn_edges_store;
pub mod op_heads;

use std::path::{Path, PathBuf};
use std::sync::Arc;

use mnem_core::error::{Error, StoreError};
use mnem_core::store::{Blockstore, OpHeadsStore};
use redb::{Database, TableDefinition};

pub use blockstore::RedbBlockstore;
pub use knn_edges_store::{KNN_EDGES_TABLE, load_knn_edges, store_knn_edges};
pub use op_heads::RedbOpHeadsStore;

/// Objects table: CID bytes → object bytes.
pub(crate) const OBJECTS_TABLE: TableDefinition<'_, &[u8], &[u8]> =
    TableDefinition::new("mnem_objects");

/// Op-heads table: CID bytes → unit (presence is the truth).
pub(crate) const OP_HEADS_TABLE: TableDefinition<'_, &[u8], ()> =
    TableDefinition::new("mnem_op_heads");

/// Map a redb error into an [`StoreError`] that `mnem-core` can consume.
pub(crate) fn redb_err<E: std::fmt::Display>(e: E) -> StoreError {
    StoreError::Io(format!("redb: {e}"))
}

/// Open or create the redb database at `path` and return wired-up
/// blockstore + op-heads stores.
///
/// The database file is created if absent; parent directories are
/// created if needed. Tables are declared on the first write so
/// subsequent opens see a consistent schema.
///
/// # Errors
///
/// Returns filesystem / redb errors on failure.
pub fn open_or_init(
    path: impl AsRef<Path>,
) -> Result<(Arc<dyn Blockstore>, Arc<dyn OpHeadsStore>, PathBuf), Error> {
    let path = path.as_ref().to_owned();
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent)
            .map_err(|e| StoreError::Io(format!("create parent dir: {e}")))?;
    }

    let db = Database::create(&path).map_err(redb_err)?;
    // Force-create the tables so a fresh reopen sees them.
    let tx = db.begin_write().map_err(redb_err)?;
    {
        let _ = tx.open_table(OBJECTS_TABLE).map_err(redb_err)?;
        let _ = tx.open_table(OP_HEADS_TABLE).map_err(redb_err)?;
    }
    tx.commit().map_err(redb_err)?;

    let db = Arc::new(db);
    let bs: Arc<dyn Blockstore> = Arc::new(RedbBlockstore::new(db.clone()));
    let ohs: Arc<dyn OpHeadsStore> = Arc::new(RedbOpHeadsStore::new(db));
    Ok((bs, ohs, path))
}

#[cfg(test)]
mod tests {
    use super::*;
    use mnem_core::repo::ReadonlyRepo;
    use std::sync::atomic::{AtomicU64, Ordering};

    static COUNTER: AtomicU64 = AtomicU64::new(0);
    fn tmp_file(name: &str) -> PathBuf {
        let path = std::env::temp_dir().join(format!(
            "mnem-redb-{name}-{}-{}.redb",
            std::process::id(),
            COUNTER.fetch_add(1, Ordering::Relaxed)
        ));
        let _ = std::fs::remove_file(&path);
        path
    }

    #[test]
    fn init_creates_file() {
        let p = tmp_file("init");
        let (_, _, file) = open_or_init(&p).unwrap();
        assert!(file.exists());
    }

    #[test]
    fn init_is_idempotent() {
        let p = tmp_file("idem");
        let _ = open_or_init(&p).unwrap();
        let _ = open_or_init(&p).unwrap();
        let _ = open_or_init(&p).unwrap();
    }

    #[test]
    fn full_repo_persists_across_reopens() {
        let p = tmp_file("persist");
        let op_at_close = {
            let (bs, ohs, _) = open_or_init(&p).unwrap();
            let repo = ReadonlyRepo::init(bs.clone(), ohs.clone()).unwrap();
            let mut tx = repo.start_transaction();
            let alice = mnem_core::objects::Node::new(mnem_core::id::NodeId::new_v7(), "Person");
            let alice_id = alice.id;
            tx.add_node(&alice).unwrap();
            let r1 = tx.commit("a@example.org", "add Alice").unwrap();
            assert!(r1.lookup_node(&alice_id).unwrap().is_some());
            r1.op_id().clone()
        };

        // All handles dropped. Reopen.
        {
            let (bs, ohs, _) = open_or_init(&p).unwrap();
            let repo = ReadonlyRepo::open(bs, ohs).unwrap();
            assert_eq!(*repo.op_id(), op_at_close);
        }
    }
}
