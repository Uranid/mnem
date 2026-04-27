//! `RedbOpHeadsStore` - redb-backed [`OpHeadsStore`].

use std::sync::Arc;

use mnem_core::error::StoreError;
use mnem_core::id::Cid;
use mnem_core::store::OpHeadsStore;
use redb::{Database, ReadableDatabase, ReadableTable};

use crate::{OP_HEADS_TABLE, redb_err};

/// Op-heads set backed by a redb `CID bytes → ()` table.
///
/// The current op-heads is the key-set of the table. `update(new,
/// supersedes)` runs in a single write transaction: inserts the new
/// head, removes each superseded head, commits. Either every step
/// succeeds or the file is unchanged - redb's ACID transaction
/// semantics give us atomicity for free, covering both the "new head"
/// write and the "supersede old heads" deletes in one commit.
pub struct RedbOpHeadsStore {
    db: Arc<Database>,
}

impl RedbOpHeadsStore {
    /// Wrap an existing [`redb::Database`].
    #[must_use]
    pub const fn new(db: Arc<Database>) -> Self {
        Self { db }
    }
}

impl std::fmt::Debug for RedbOpHeadsStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RedbOpHeadsStore").finish_non_exhaustive()
    }
}

impl OpHeadsStore for RedbOpHeadsStore {
    fn current(&self) -> Result<Vec<Cid>, StoreError> {
        let tx = self.db.begin_read().map_err(redb_err)?;
        let table = tx.open_table(OP_HEADS_TABLE).map_err(redb_err)?;
        let mut out = Vec::new();
        for row in table.iter().map_err(redb_err)? {
            let (k, _v) = row.map_err(redb_err)?;
            let cid = Cid::from_bytes(k.value())
                .map_err(|e| StoreError::Io(format!("parse op-head key: {e}")))?;
            out.push(cid);
        }
        out.sort();
        Ok(out)
    }

    fn update(&self, new: Cid, supersedes: &[Cid]) -> Result<(), StoreError> {
        let new_key = new.to_bytes();
        let tx = self.db.begin_write().map_err(redb_err)?;
        {
            let mut table = tx.open_table(OP_HEADS_TABLE).map_err(redb_err)?;
            table.insert(&new_key[..], ()).map_err(redb_err)?;
            for s in supersedes {
                let k = s.to_bytes();
                // Remove returns None if key was absent; tolerate.
                let _ = table.remove(&k[..]).map_err(redb_err)?;
            }
        }
        tx.commit().map_err(redb_err)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mnem_core::id::{CODEC_DAG_CBOR, Multihash};
    use redb::Database;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};

    static COUNTER: AtomicU64 = AtomicU64::new(0);
    fn tmp_file(name: &str) -> PathBuf {
        let path = std::env::temp_dir().join(format!(
            "mnem-redb-opheads-{name}-{}-{}.redb",
            std::process::id(),
            COUNTER.fetch_add(1, Ordering::Relaxed)
        ));
        let _ = std::fs::remove_file(&path);
        path
    }

    fn op_cid(n: u32) -> Cid {
        Cid::new(CODEC_DAG_CBOR, Multihash::sha2_256(&n.to_be_bytes()))
    }

    fn new_store() -> RedbOpHeadsStore {
        let p = tmp_file("oh");
        let db = Arc::new(Database::create(&p).unwrap());
        let tx = db.begin_write().unwrap();
        let _ = tx.open_table(OP_HEADS_TABLE).unwrap();
        tx.commit().unwrap();
        RedbOpHeadsStore::new(db)
    }

    #[test]
    fn empty_store_has_no_heads() {
        let s = new_store();
        assert!(s.current().unwrap().is_empty());
    }

    #[test]
    fn update_creates_and_current_finds() {
        let s = new_store();
        let root = op_cid(1);
        s.update(root.clone(), &[]).unwrap();
        assert_eq!(s.current().unwrap(), vec![root]);
    }

    #[test]
    fn update_supersedes_atomically() {
        let s = new_store();
        let a = op_cid(1);
        let b = op_cid(2);
        s.update(a.clone(), &[]).unwrap();
        s.update(b.clone(), &[a]).unwrap();
        assert_eq!(s.current().unwrap(), vec![b]);
    }

    #[test]
    fn concurrent_commits_leave_two_heads() {
        let s = new_store();
        let root = op_cid(1);
        s.update(root.clone(), &[]).unwrap();
        s.update(op_cid(10), std::slice::from_ref(&root)).unwrap();
        s.update(op_cid(11), &[root]).unwrap(); // second finds root already gone; tolerated
        let heads = s.current().unwrap();
        assert_eq!(heads.len(), 2);
    }

    #[test]
    fn current_is_sorted() {
        let s = new_store();
        s.update(op_cid(3), &[]).unwrap();
        s.update(op_cid(1), &[]).unwrap();
        s.update(op_cid(2), &[]).unwrap();
        let heads = s.current().unwrap();
        assert!(heads.windows(2).all(|w| w[0] < w[1]));
    }
}
