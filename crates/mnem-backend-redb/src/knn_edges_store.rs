//! redb persistence for `mnem-ann::KnnEdgeIndex` (experiment E0).
//!
//! A new `mnem_knn_edges` table is added alongside `mnem_objects` and
//! `mnem_op_heads`. Rows are keyed by the edge-index CID bytes (as
//! produced by [`KnnEdgeIndex::compute_cid`]) and store the canonical
//! DAG-CBOR encoding of the edge index.
//!
//! The table is opened on first write via redb's
//! `WriteTransaction::open_table`, so existing databases created
//! before this change remain readable and pay zero cost until a
//! caller actually persists a KNN-edge index.

use std::sync::Arc;

use mnem_ann::KnnEdgeIndex;
use mnem_core::codec::{from_canonical_bytes, to_canonical_bytes};
use mnem_core::error::{Error, StoreError};
use mnem_core::id::Cid;
use redb::{Database, ReadableDatabase, TableDefinition};

use crate::redb_err;

/// KNN-edge table: `KnnEdgeIndex` CID bytes -> DAG-CBOR body bytes.
pub const KNN_EDGES_TABLE: TableDefinition<'_, &[u8], &[u8]> =
    TableDefinition::new("mnem_knn_edges");

/// Persist a [`KnnEdgeIndex`] into the `mnem_knn_edges` table.
///
/// Returns the content-addressed CID under which the index was
/// stored. Idempotent: storing the same index twice is a no-op
/// beyond the CID computation.
///
/// # Errors
///
/// - [`Error::Codec`] if CBOR encoding fails.
/// - [`Error::Store`] if the redb transaction fails.
pub fn store_knn_edges(db: &Arc<Database>, idx: &KnnEdgeIndex) -> Result<Cid, Error> {
    let cid = idx.compute_cid()?;
    let body = to_canonical_bytes(idx)?;
    let key_bytes = cid.to_bytes();
    let tx = db.begin_write().map_err(|e| Error::Store(redb_err(e)))?;
    {
        let mut table = tx
            .open_table(KNN_EDGES_TABLE)
            .map_err(|e| Error::Store(redb_err(e)))?;
        table
            .insert(key_bytes.as_slice(), body.as_ref())
            .map_err(|e| Error::Store(redb_err(e)))?;
    }
    tx.commit().map_err(|e| Error::Store(redb_err(e)))?;
    Ok(cid)
}

/// Load a [`KnnEdgeIndex`] by CID from the `mnem_knn_edges` table.
///
/// # Errors
///
/// - [`StoreError::NotFound`] if the CID is absent.
/// - [`Error::Codec`] if the stored bytes fail to decode.
/// - [`Error::Store`] if the redb read fails.
pub fn load_knn_edges(db: &Arc<Database>, cid: &Cid) -> Result<KnnEdgeIndex, Error> {
    let tx = db.begin_read().map_err(|e| Error::Store(redb_err(e)))?;
    let table = match tx.open_table(KNN_EDGES_TABLE) {
        Ok(t) => t,
        Err(redb::TableError::TableDoesNotExist(_)) => {
            return Err(Error::Store(StoreError::NotFound { cid: cid.clone() }));
        }
        Err(e) => return Err(Error::Store(redb_err(e))),
    };
    let key_bytes = cid.to_bytes();
    let Some(slot) = table
        .get(key_bytes.as_slice())
        .map_err(|e| Error::Store(redb_err(e)))?
    else {
        return Err(Error::Store(StoreError::NotFound { cid: cid.clone() }));
    };
    let bytes = slot.value().to_vec();
    let idx: KnnEdgeIndex = from_canonical_bytes(&bytes)?;
    Ok(idx)
}

#[cfg(test)]
mod tests {
    use super::*;

    use mnem_ann::{DistanceMetric, KnnEdgeIndex};
    use mnem_core::id::{CODEC_DAG_CBOR, Multihash};

    /// Returns (path_guard, db). The guard cleans the file on drop.
    struct PathGuard(std::path::PathBuf);
    impl Drop for PathGuard {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
        }
    }

    fn tmp_db(name: &str) -> (PathGuard, Arc<Database>) {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let path = std::env::temp_dir().join(format!("mnem-knn-{name}-{nonce}.redb"));
        let db = Database::create(&path).unwrap();
        (PathGuard(path), Arc::new(db))
    }

    fn demo_idx() -> KnnEdgeIndex {
        let root = Cid::new(CODEC_DAG_CBOR, Multihash::sha2_256(b"root"));
        KnnEdgeIndex::empty(root, 3, DistanceMetric::Cosine)
    }

    #[test]
    fn store_then_load_round_trips() {
        let (_guard, db) = tmp_db("round");
        let idx = demo_idx();
        let cid = store_knn_edges(&db, &idx).unwrap();
        let loaded = load_knn_edges(&db, &cid).unwrap();
        assert_eq!(loaded, idx);
    }

    #[test]
    fn load_missing_returns_not_found() {
        let (_guard, db) = tmp_db("missing");
        let bogus = Cid::new(CODEC_DAG_CBOR, Multihash::sha2_256(b"nope"));
        let err = load_knn_edges(&db, &bogus).unwrap_err();
        assert!(matches!(err, Error::Store(StoreError::NotFound { .. })));
    }
}
