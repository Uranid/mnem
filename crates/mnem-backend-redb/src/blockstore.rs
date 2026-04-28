//! `RedbBlockstore` - redb-backed [`Blockstore`] implementation.

use std::sync::Arc;

use bytes::Bytes;
use mnem_core::error::StoreError;
use mnem_core::id::Cid;
use mnem_core::store::{Blockstore, blockstore::recompute_cid};
use redb::{Database, ReadableDatabase};

use crate::{OBJECTS_TABLE, redb_err};

/// A [`Blockstore`] backed by a [`redb::Database`].
///
/// Clone-friendly: the underlying database handle is `Arc`-wrapped so
/// multiple owners point at the same file.
pub struct RedbBlockstore {
    db: Arc<Database>,
}

impl RedbBlockstore {
    /// Wrap an existing [`redb::Database`].
    #[must_use]
    pub const fn new(db: Arc<Database>) -> Self {
        Self { db }
    }
}

impl std::fmt::Debug for RedbBlockstore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RedbBlockstore").finish_non_exhaustive()
    }
}

impl Blockstore for RedbBlockstore {
    fn has(&self, cid: &Cid) -> Result<bool, StoreError> {
        let key = cid.to_bytes();
        let tx = self.db.begin_read().map_err(redb_err)?;
        let table = tx.open_table(OBJECTS_TABLE).map_err(redb_err)?;
        Ok(table.get(&key[..]).map_err(redb_err)?.is_some())
    }

    fn get(&self, cid: &Cid) -> Result<Option<Bytes>, StoreError> {
        let key = cid.to_bytes();
        let tx = self.db.begin_read().map_err(redb_err)?;
        let table = tx.open_table(OBJECTS_TABLE).map_err(redb_err)?;
        match table.get(&key[..]).map_err(redb_err)? {
            Some(access) => Ok(Some(Bytes::copy_from_slice(access.value()))),
            None => Ok(None),
        }
    }

    fn put(&self, cid: Cid, data: Bytes) -> Result<(), StoreError> {
        // Verify the caller's CID claim before committing. The hash
        // cost (tens of microseconds per MB) is paid before the redb
        // write so a mismatch never enters the database. Unknown
        // hash algorithms (recompute_cid returns None) are stored as
        // claimed - see the Blockstore trait docs for the contract.
        if let Some(computed) = recompute_cid(&cid, &data)
            && computed != cid
        {
            return Err(StoreError::CidMismatch {
                claimed: cid,
                computed,
            });
        }
        let key = cid.to_bytes();
        let tx = self.db.begin_write().map_err(redb_err)?;
        {
            let mut table = tx.open_table(OBJECTS_TABLE).map_err(redb_err)?;
            table.insert(&key[..], &data[..]).map_err(redb_err)?;
        }
        tx.commit().map_err(redb_err)?;
        Ok(())
    }

    fn put_trusted(&self, cid: Cid, data: Bytes) -> Result<(), StoreError> {
        // Skip CID recompute. The `Blockstore` trait's `put_trusted`
        // safety contract applies: the caller MUST have established
        // that `data` hashes to `cid` before reaching this method.
        // Used by audited in-tree callers that just produced
        // `(bytes, cid)` from `hash_to_cid`.
        let key = cid.to_bytes();
        let tx = self.db.begin_write().map_err(redb_err)?;
        {
            let mut table = tx.open_table(OBJECTS_TABLE).map_err(redb_err)?;
            table.insert(&key[..], &data[..]).map_err(redb_err)?;
        }
        tx.commit().map_err(redb_err)?;
        Ok(())
    }

    fn delete(&self, cid: &Cid) -> Result<(), StoreError> {
        let key = cid.to_bytes();
        let tx = self.db.begin_write().map_err(redb_err)?;
        {
            let mut table = tx.open_table(OBJECTS_TABLE).map_err(redb_err)?;
            table.remove(&key[..]).map_err(redb_err)?;
        }
        tx.commit().map_err(redb_err)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mnem_core::codec::hash_to_cid;
    use redb::Database;
    use serde::Serialize;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};

    #[derive(Serialize)]
    struct Sample {
        n: u32,
    }

    static COUNTER: AtomicU64 = AtomicU64::new(0);
    fn tmp_file(name: &str) -> PathBuf {
        let path = std::env::temp_dir().join(format!(
            "mnem-redb-blockstore-{name}-{}-{}.redb",
            std::process::id(),
            COUNTER.fetch_add(1, Ordering::Relaxed)
        ));
        let _ = std::fs::remove_file(&path);
        path
    }

    fn new_store() -> RedbBlockstore {
        let p = tmp_file("bs");
        let db = Arc::new(Database::create(&p).unwrap());
        // Force-create the table.
        let tx = db.begin_write().unwrap();
        let _ = tx.open_table(OBJECTS_TABLE).unwrap();
        tx.commit().unwrap();
        RedbBlockstore::new(db)
    }

    #[test]
    fn put_get_has_delete_round_trip() {
        let bs = new_store();
        let (bytes, cid) = hash_to_cid(&Sample { n: 7 }).unwrap();
        assert!(!bs.has(&cid).unwrap());
        bs.put(cid.clone(), bytes.clone()).unwrap();
        assert!(bs.has(&cid).unwrap());
        assert_eq!(bs.get(&cid).unwrap(), Some(bytes));
        bs.delete(&cid).unwrap();
        assert!(!bs.has(&cid).unwrap());
    }

    #[test]
    fn put_is_idempotent() {
        let bs = new_store();
        let (bytes, cid) = hash_to_cid(&Sample { n: 42 }).unwrap();
        bs.put(cid.clone(), bytes.clone()).unwrap();
        bs.put(cid.clone(), bytes).unwrap(); // no-op
        assert!(bs.has(&cid).unwrap());
    }

    #[test]
    fn delete_missing_is_ok() {
        let bs = new_store();
        let (_, cid) = hash_to_cid(&Sample { n: 99 }).unwrap();
        bs.delete(&cid).unwrap();
    }

    #[test]
    fn put_trusted_round_trip() {
        // `put_trusted` is the audited fast path. Round-trip must
        // behave byte-identically to the verified `put` path.
        let bs = new_store();
        let (bytes, cid) = hash_to_cid(&Sample { n: 17 }).unwrap();
        bs.put_trusted(cid.clone(), bytes.clone()).unwrap();
        assert!(bs.has(&cid).unwrap());
        assert_eq!(bs.get(&cid).unwrap(), Some(bytes));
    }

    #[test]
    fn put_trusted_skips_verification_by_design() {
        // Pins the safety-contract behaviour: `put_trusted` does
        // NOT recompute the CID. A caller that violates the contract
        // successfully lands mismatched bytes in the store. This is
        // the perf point of `put_trusted`; `put` is the safety net
        // for untrusted callers (see `put_rejects_cid_mismatch`).
        let bs = new_store();
        let (_, cid) = hash_to_cid(&Sample { n: 1 }).unwrap();
        let wrong_bytes = Bytes::from_static(b"not the sample");
        bs.put_trusted(cid.clone(), wrong_bytes.clone())
            .expect("put_trusted skips verify");
        assert_eq!(bs.get(&cid).unwrap(), Some(wrong_bytes));
    }

    #[test]
    fn put_rejects_cid_mismatch() {
        // Hostile or buggy caller ships bytes whose hash does NOT
        // match the claimed CID. The redb backend must refuse before
        // anything is committed to disk.
        let bs = new_store();
        let (_, cid) = hash_to_cid(&Sample { n: 1 }).unwrap();
        let wrong_bytes = Bytes::from_static(b"not the sample");
        let err = bs.put(cid.clone(), wrong_bytes).unwrap_err();
        match err {
            StoreError::CidMismatch { claimed, .. } => assert_eq!(claimed, cid),
            e => panic!("wrong variant: {e:?}"),
        }
        // And nothing landed on disk.
        assert!(!bs.has(&cid).unwrap());
    }
}
