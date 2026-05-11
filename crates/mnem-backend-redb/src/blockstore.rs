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
            Some(access) => {
                let data = Bytes::copy_from_slice(access.value());
                // Recompute the CID and compare to the requested key.
                // This catches silent disk corruption: if the on-disk bytes
                // no longer hash to the CID they are stored under, we must
                // not return the corrupt data to the caller.
                // Unknown hash algorithms (recompute_cid returns None) are
                // passed through as-is - we cannot verify what we cannot hash.
                if let Some(computed) = recompute_cid(cid, &data) {
                    if computed != *cid {
                        return Err(StoreError::Corruption {
                            cid: cid.to_string(),
                            detail: format!("expected {cid}, recomputed {computed}"),
                        });
                    }
                }
                Ok(Some(data))
            }
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

    fn batch_put(&self, blocks: &mut dyn Iterator<Item = (Cid, Bytes)>) -> Result<(), StoreError> {
        // Single write transaction for the entire batch: all blocks land in
        // one fsync instead of one per block. This is the BUG-22 fix.
        // CID integrity is verified per block before inserting, matching the
        // contract of `put`. If any block fails verification the whole batch
        // is rolled back (the write transaction is dropped without commit).
        let tx = self.db.begin_write().map_err(redb_err)?;
        {
            let mut table = tx.open_table(OBJECTS_TABLE).map_err(redb_err)?;
            for (cid, data) in blocks {
                if let Some(computed) = recompute_cid(&cid, &data)
                    && computed != cid
                {
                    return Err(StoreError::CidMismatch {
                        claimed: cid,
                        computed,
                    });
                }
                let key = cid.to_bytes();
                table.insert(&key[..], &data[..]).map_err(redb_err)?;
            }
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

    fn all_cids(&self) -> Result<Option<Vec<Cid>>, StoreError> {
        use redb::ReadableTable as _;
        let tx = self.db.begin_read().map_err(redb_err)?;
        let table = tx.open_table(OBJECTS_TABLE).map_err(redb_err)?;
        let mut cids = Vec::new();
        for entry in table.iter().map_err(redb_err)? {
            let (k, _v) = entry.map_err(redb_err)?;
            let cid = Cid::from_bytes(k.value())
                .map_err(|e| StoreError::Io(format!("invalid CID key in store: {e}")))?;
            cids.push(cid);
        }
        Ok(Some(cids))
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
    fn put_trusted_skips_write_verification_by_design() {
        // `put_trusted` does NOT recompute the CID on the write path.
        // A caller that violates the safety contract can land mismatched
        // bytes in the store - that is the perf contract for the write
        // fast-path. The integrity firewall for that case now lives on
        // the read path: `get()` catches the corruption (see
        // `get_detects_disk_corruption` below).
        let bs = new_store();
        let (_, cid) = hash_to_cid(&Sample { n: 1 }).unwrap();
        let wrong_bytes = Bytes::from_static(b"not the sample");
        bs.put_trusted(cid.clone(), wrong_bytes.clone())
            .expect("put_trusted skips write-time verify");
        // get() must now surface the corruption rather than silently
        // returning corrupt bytes.
        let err = bs.get(&cid).unwrap_err();
        match err {
            StoreError::Corruption { cid: cid_str, .. } => {
                assert!(cid_str.contains(&cid.to_string()[..16]));
            }
            e => panic!("expected Corruption, got {e:?}"),
        }
    }

    #[test]
    fn get_detects_disk_corruption() {
        // BUG-21: get() must recompute the CID and refuse to return bytes
        // that do not hash to the requested key. This simulates what silent
        // disk corruption looks like at the redb layer: put_trusted with
        // wrong bytes bypasses the write-path check, so the mismatch is
        // only caught here on the read path.
        let bs = new_store();
        let (_, cid) = hash_to_cid(&Sample { n: 55 }).unwrap();
        let corrupt_bytes = Bytes::from_static(b"silent disk corruption");
        // Force the mismatch into the store via the trusted fast-path.
        bs.put_trusted(cid.clone(), corrupt_bytes)
            .expect("put_trusted skips verify");
        // get() must detect the corruption.
        let err = bs.get(&cid).unwrap_err();
        match err {
            StoreError::Corruption { .. } => {}
            e => panic!("expected Corruption, got {e:?}"),
        }
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

    #[test]
    fn batch_put_stores_all_blocks_in_single_transaction() {
        // BUG-22: batch_put must commit N blocks in one write transaction
        // (one fsync) rather than N separate transactions. This test verifies
        // correctness: every block written via batch_put is retrievable via
        // get(), and the CID integrity check is applied to each block.
        let bs = new_store();

        let blocks: Vec<(Cid, Bytes)> = (0u32..5)
            .map(|n| {
                let (bytes, cid) = hash_to_cid(&Sample { n }).unwrap();
                (cid, bytes)
            })
            .collect();

        // All five blocks must be absent before the batch.
        for (cid, _) in &blocks {
            assert!(!bs.has(cid).unwrap(), "block {cid} should not exist yet");
        }

        bs.batch_put(&mut blocks.clone().into_iter()).unwrap();

        // All five blocks must be present and byte-identical after the batch.
        for (cid, bytes) in &blocks {
            assert!(bs.has(cid).unwrap(), "block {cid} missing after batch_put");
            assert_eq!(
                bs.get(cid).unwrap(),
                Some(bytes.clone()),
                "block {cid} content mismatch after batch_put"
            );
        }
    }

    #[test]
    fn batch_put_rejects_cid_mismatch_and_rolls_back() {
        // If any block in the batch has a CID that does not match its bytes,
        // batch_put must return CidMismatch and nothing must land on disk
        // (the whole write transaction is abandoned before commit).
        let bs = new_store();

        let (good_bytes, good_cid) = hash_to_cid(&Sample { n: 100 }).unwrap();
        let (_, bad_cid) = hash_to_cid(&Sample { n: 200 }).unwrap();
        let wrong_bytes = Bytes::from_static(b"definitely not sample 200");

        let err = bs
            .batch_put(
                &mut vec![
                    (good_cid.clone(), good_bytes),
                    (bad_cid.clone(), wrong_bytes),
                ]
                .into_iter(),
            )
            .unwrap_err();

        match err {
            StoreError::CidMismatch { claimed, .. } => assert_eq!(claimed, bad_cid),
            e => panic!("expected CidMismatch, got {e:?}"),
        }

        // The good block that came before the bad one must also be absent
        // because the transaction was never committed.
        assert!(
            !bs.has(&good_cid).unwrap(),
            "good block should not have been committed when batch failed"
        );
    }
}
