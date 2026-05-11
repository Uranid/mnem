//! [`Blockstore`] trait and [`MemoryBlockstore`] reference implementation.
//!
//! A blockstore is a content-addressed byte store. The unit of storage is a
//! CBOR-encoded mnem object (or a raw blob); the address is its [`Cid`].
//! Implementations MUST be `Send + Sync` so a single store instance can be
//! shared across threads.
//!
//! ## `put` contract
//!
//! The caller supplies both the bytes and their CID. Backends MUST
//! recompute the CID from the bytes and reject a mismatch with
//! [`StoreError::CidMismatch`]. This closes a class of bugs where a
//! caller attaches the wrong CID (copy/paste, stale variable) and
//! silently corrupts the content-addressed invariant.
//!
//! The Phase 1 risk review flagged re-hashing as a hot-path tax, but
//! the Bob security audit re-scored the tradeoff: accepting an
//! attacker-controlled `(cid, bytes)` pair is a blockstore-level
//! integrity violation - every typed layer above assumes `get(cid)`
//! returns bytes that hash to `cid`. A single unverified `put` from
//! hostile input (CAR import, HTTP body, MCP tool call) poisons every
//! downstream reader. The hash cost is SHA-256 or BLAKE3 over block
//! bytes (tens of microseconds per MB) - cheaper than the redb commit
//! that immediately follows.
//!
//! If a caller uses an unrecognized hash algorithm, [`recompute_cid`]
//! returns `None` and verification is skipped (the block is stored as
//! claimed). Unknown algorithms reach the blockstore only when a
//! custom codec is in play; those callers already opt out of the
//! integrity guarantee by design.
//!
//! ## `put_trusted` contract
//!
//! [`Blockstore::put_trusted`] is an additive fast-path for callers
//! that have just computed the CID from the same bytes they are
//! about to store (the in-tree pattern: `let (bytes, cid) =
//! hash_to_cid(obj)?; bs.put_trusted(cid, bytes)?;`). Skipping the
//! recompute buys back the ~0.1-0.3% hot-path tax on commit-heavy
//! workloads without weakening the default (`put`) for untrusted
//! callers (CAR import, HTTP body, FFI boundary, user-land
//! `Blockstore` implementors).
//!
//! The contract is strict: **the supplied bytes MUST hash to the
//! supplied CID under the codec+algorithm in `cid`**. Misuse
//! silently corrupts the content-addressed store - every `get(cid)`
//! thereafter returns bytes that do NOT hash to `cid`, and every
//! typed layer above (commit, view, index, retrieval) assumes that
//! invariant holds. There is no runtime check.
//!
//! Use `put` by default. Use `put_trusted` only at a callsite where
//! an audit has established:
//! 1. The CID came from `hash_to_cid` (or equivalent) over the exact
//!    bytes being passed, with no mutation between the hash and the
//!    put.
//! 2. The bytes are not attacker-influenced - or, if they are (as
//!    in CAR import), the caller has *independently* verified the
//!    claimed CID against the bytes before reaching `put_trusted`.

use core::fmt;
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Mutex;

use bytes::Bytes;

use crate::codec::extract_links;
use crate::error::StoreError;
use crate::id::{CODEC_DAG_CBOR, Cid, HASH_BLAKE3_256, HASH_SHA2_256, Multihash};

/// A content-addressed byte store.
///
/// See the module-level documentation for contract details. All methods
/// are synchronous; async wrappers live in server / binding crates that
/// need them.
pub trait Blockstore: Send + Sync + fmt::Debug {
    /// Check whether an object with the given CID exists in the store.
    ///
    /// # Errors
    ///
    /// Returns a backend-specific I/O error. Absence of the object is
    /// reported as `Ok(false)`, never as an error.
    fn has(&self, cid: &Cid) -> Result<bool, StoreError>;

    /// Fetch an object's raw bytes by CID.
    ///
    /// Returns `Ok(None)` if the object is not present, `Ok(Some(bytes))`
    /// if found. Backend-specific I/O failures return `Err`.
    ///
    /// # Errors
    ///
    /// Returns a backend-specific I/O error.
    fn get(&self, cid: &Cid) -> Result<Option<Bytes>, StoreError>;

    /// Write an object's raw bytes under the given CID.
    ///
    /// Idempotent: writing the same CID+bytes pair twice is a no-op (the
    /// bytes on disk are byte-identical to what was already there,
    /// because content addressing guarantees collision only on
    /// byte-identical content).
    ///
    /// # Errors
    ///
    /// Returns a backend-specific I/O error.
    fn put(&self, cid: Cid, data: Bytes) -> Result<(), StoreError>;

    /// Write an object's raw bytes under the given CID **without
    /// verifying** that the bytes hash to the CID.
    ///
    /// # Safety contract
    ///
    /// The caller MUST guarantee that `data` hashes to `cid` under the
    /// codec and hash algorithm encoded in `cid`. Misuse silently
    /// corrupts the content-addressed store: every later `get(cid)`
    /// returns bytes that do not hash to `cid`, and every typed layer
    /// above (commit, view, index, retrieval) assumes the invariant.
    /// There is no runtime check.
    ///
    /// This method exists as an audited fast path for the in-tree
    /// pattern `let (bytes, cid) = hash_to_cid(obj)?;
    /// bs.put_trusted(cid, bytes)?;`, where the double-hash inside
    /// [`Blockstore::put`] would be wasted work. See the module-level
    /// `put_trusted` contract for the full checklist before
    /// introducing a new callsite.
    ///
    /// The default implementation delegates to [`Blockstore::put`] so
    /// every existing `Blockstore` implementor (including third-party
    /// and FFI backends) keeps the safer verify-on-put behaviour
    /// until they explicitly override. Concrete in-tree backends
    /// (`MemoryBlockstore`, `RedbBlockstore`) override to skip the
    /// recompute.
    ///
    /// Idempotent in the same sense as [`Blockstore::put`].
    ///
    /// # Errors
    ///
    /// Returns a backend-specific I/O error.
    fn put_trusted(&self, cid: Cid, data: Bytes) -> Result<(), StoreError> {
        // Default: delegate to `put`. Overrides in `MemoryBlockstore`
        // and `RedbBlockstore` skip the CID recompute for the
        // in-tree audited callsites.
        self.put(cid, data)
    }

    /// Remove an object. Intended for garbage collection only; normal
    /// application code should never call this directly.
    ///
    /// Absence is not an error: removing a CID that isn't present
    /// returns `Ok(())`.
    ///
    /// # Errors
    ///
    /// Returns a backend-specific I/O error.
    fn delete(&self, cid: &Cid) -> Result<(), StoreError>;

    /// Yield every CID reachable from `root`, in depth-first order,
    /// together with the backing block bytes. Each reachable CID is
    /// yielded exactly once; repeated links are deduplicated. The
    /// iterator terminates on first error (malformed block, missing
    /// reference, backend I/O).
    ///
    /// The default implementation decodes DAG-CBOR blocks with
    /// [`extract_links`] to discover outgoing references; raw blocks
    /// (codec 0x55) are treated as leaves and yield no children.
    /// Implementations with a native graph index may override for
    /// speed, but must preserve the reachable-exactly-once contract.
    ///
    /// Returned as a boxed trait object so this trait stays
    /// `dyn`-compatible; every caller in the codebase consumes
    /// `Arc<dyn Blockstore>`.     ///
    /// # Errors
    ///
    /// Per-item [`StoreError`] for backend I/O, missing-block refs, or
    /// malformed DAG-CBOR.
    fn iter_from_root<'a>(
        &'a self,
        root: &Cid,
    ) -> Box<dyn Iterator<Item = Result<(Cid, Bytes), StoreError>> + 'a> {
        Box::new(ReachableIter::new(self, root.clone()))
    }

    /// Enumerate every CID currently stored.
    ///
    /// Intended exclusively for garbage collection. Returns all CIDs
    /// in the store in unspecified order. The default implementation
    /// returns `None`, meaning the backend does not support enumeration.
    /// Backends that can enumerate (e.g. `RedbBlockstore`) override to
    /// return `Some(vec)`. Returning `None` does not prevent GC from
    /// running; the caller degrades gracefully to a dry-run report.
    ///
    /// # Errors
    ///
    /// Backend-specific I/O errors.
    fn all_cids(&self) -> Result<Option<Vec<Cid>>, StoreError> {
        Ok(None)
    }

    /// Write multiple blocks in a single batch operation.
    ///
    /// Implementations SHOULD override this to use a single write
    /// transaction, amortising the per-transaction fsync cost across all
    /// blocks in the batch. This is the primary fix for BUG-22: bulk
    /// operations (ingest, reindex) were previously paying one fsync per
    /// block, making them ~100x slower than `MemoryBlockstore`.
    ///
    /// The iterator is taken as a `&mut dyn Iterator` to keep the trait
    /// dyn-compatible (`impl IntoIterator` would introduce a generic type
    /// parameter that prevents `dyn Blockstore` from being used as a trait
    /// object). Callers pass `.iter().cloned()`, `vec.into_iter()`, etc.
    /// through `&mut` after collecting or via a temporary:
    ///
    /// ```ignore
    /// bs.batch_put(&mut vec_of_blocks.into_iter())?;
    /// ```
    ///
    /// The default implementation falls back to calling [`Blockstore::put`]
    /// in a loop, so existing implementations that do not override remain
    /// correct.
    ///
    /// # Errors
    ///
    /// Returns the first backend-specific I/O error encountered; blocks
    /// written before the error are not rolled back in the default
    /// implementation (overrides that use a single transaction will roll
    /// back the whole batch on failure).
    fn batch_put(&self, blocks: &mut dyn Iterator<Item = (Cid, Bytes)>) -> Result<(), StoreError> {
        for (cid, data) in blocks {
            self.put(cid, data)?;
        }
        Ok(())
    }
}

/// Default depth-first reachable-block iterator over a [`Blockstore`].
///
/// Uses a stack + visited-set to walk the IPLD DAG rooted at a single
/// CID. Yields `(cid, bytes)` for each block exactly once; a CID that
/// appears as a link but has already been yielded is skipped. A CID
/// that is not present in the store produces a
/// [`StoreError::NotFound`] item, after which the iterator fuses (no
/// further items).
///
/// Encapsulated rather than inlined so backends with specialised
/// walks (e.g. a future redb-native graph index) can keep the same
/// shape by implementing `Iterator<Item = Result<(Cid, Bytes),
/// StoreError>>` and wrapping in `Box::new`.
struct ReachableIter<'a, B: Blockstore + ?Sized> {
    bs: &'a B,
    stack: VecDeque<Cid>,
    seen: HashSet<Cid>,
    fused: bool,
}

impl<'a, B: Blockstore + ?Sized> ReachableIter<'a, B> {
    fn new(bs: &'a B, root: Cid) -> Self {
        let mut stack = VecDeque::new();
        let mut seen = HashSet::new();
        seen.insert(root.clone());
        stack.push_back(root);
        Self {
            bs,
            stack,
            seen,
            fused: false,
        }
    }
}

impl<B: Blockstore + ?Sized> Iterator for ReachableIter<'_, B> {
    type Item = Result<(Cid, Bytes), StoreError>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.fused {
            return None;
        }
        let cid = self.stack.pop_back()?;

        let bytes = match self.bs.get(&cid) {
            Ok(Some(b)) => b,
            Ok(None) => {
                self.fused = true;
                return Some(Err(StoreError::NotFound { cid }));
            }
            Err(e) => {
                self.fused = true;
                return Some(Err(e));
            }
        };

        // Only DAG-CBOR blocks can carry links. Raw blocks are leaves by
        // definition; decoding them as CBOR would fail on valid inputs.
        if cid.codec() == CODEC_DAG_CBOR {
            match extract_links(&bytes) {
                Ok(links) => {
                    // Push in reverse so the visit order matches the
                    // encoded-link order of the parent block (a DFS
                    // pre-order walk of the Ipld tree).
                    for link in links.into_iter().rev() {
                        if self.seen.insert(link.clone()) {
                            self.stack.push_back(link);
                        }
                    }
                }
                Err(e) => {
                    self.fused = true;
                    return Some(Err(StoreError::Io(format!("decode links for {cid}: {e}"))));
                }
            }
        }

        Some(Ok((cid, bytes)))
    }
}

/// Recompute the CID of `data` under the same codec + algorithm as `claimed`.
///
/// Used by blockstore backends to verify the caller's CID claim on every
/// `put`. Returns `None` if the claimed CID uses a hash algorithm not
/// supported by this build (rare; we recognize SHA-256 and BLAKE3-256
/// today). Backends treat `None` as "cannot verify, trust the claim" -
/// see the module-level `put` contract.
#[must_use]
pub fn recompute_cid(claimed: &Cid, data: &[u8]) -> Option<Cid> {
    let mh = match claimed.multihash().code() {
        HASH_SHA2_256 => Multihash::sha2_256(data),
        HASH_BLAKE3_256 => Multihash::blake3_256(data),
        _ => return None,
    };
    Some(Cid::new(claimed.codec(), mh))
}

// ---------------- MemoryBlockstore ----------------

/// An in-process [`Blockstore`] backed by a `HashMap<Cid, Bytes>`.
///
/// Suitable for tests, WASM default storage, and ephemeral workflows.
/// Not durable. Not shared across processes.
///
/// Thread-safe: internal state is behind a [`Mutex`]. Concurrent reads
/// serialize through the lock, which is acceptable for the intended use
/// cases (tests, single-agent loops). High-throughput workloads should
/// use `mnem-backend-redb`.
#[derive(Default)]
pub struct MemoryBlockstore {
    blocks: Mutex<HashMap<Cid, Bytes>>,
}

impl MemoryBlockstore {
    /// Create an empty in-memory blockstore.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of objects currently stored.
    ///
    /// Provided for smoke-test visibility; not part of the [`Blockstore`]
    /// trait since persistent backends may not be able to answer cheaply.
    ///
    /// # Panics
    ///
    /// Panics if the internal mutex is poisoned (another thread panicked
    /// while holding the lock).
    #[must_use]
    pub fn len(&self) -> usize {
        self.blocks.lock().expect("mutex not poisoned").len()
    }

    /// True if the store holds zero objects.
    ///
    /// # Panics
    ///
    /// Panics if the internal mutex is poisoned.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl fmt::Debug for MemoryBlockstore {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "MemoryBlockstore {{ n_blocks: {} }}", self.len())
    }
}

impl Blockstore for MemoryBlockstore {
    fn has(&self, cid: &Cid) -> Result<bool, StoreError> {
        Ok(self
            .blocks
            .lock()
            .expect("mutex not poisoned")
            .contains_key(cid))
    }

    fn get(&self, cid: &Cid) -> Result<Option<Bytes>, StoreError> {
        Ok(self
            .blocks
            .lock()
            .expect("mutex not poisoned")
            .get(cid)
            .cloned())
    }

    fn put(&self, cid: Cid, data: Bytes) -> Result<(), StoreError> {
        // Always verify in both debug and release: the blockstore's
        // content-addressed invariant is too load-bearing to let an
        // untrusted caller poison it. Unknown hash algorithms skip
        // verification (see module docs).
        if let Some(computed) = recompute_cid(&cid, &data)
            && computed != cid
        {
            return Err(StoreError::CidMismatch {
                claimed: cid,
                computed,
            });
        }
        self.blocks
            .lock()
            .expect("mutex not poisoned")
            .insert(cid, data);
        Ok(())
    }

    fn put_trusted(&self, cid: Cid, data: Bytes) -> Result<(), StoreError> {
        // Skip CID recompute. See trait doc `# Safety contract`:
        // the caller has independently established `data` hashes to
        // `cid`. Used by audited in-tree callers that just produced
        // `(bytes, cid)` from `hash_to_cid`.
        self.blocks
            .lock()
            .expect("mutex not poisoned")
            .insert(cid, data);
        Ok(())
    }

    fn delete(&self, cid: &Cid) -> Result<(), StoreError> {
        self.blocks.lock().expect("mutex not poisoned").remove(cid);
        Ok(())
    }

    fn all_cids(&self) -> Result<Option<Vec<Cid>>, StoreError> {
        let keys: Vec<Cid> = self
            .blocks
            .lock()
            .expect("mutex not poisoned")
            .keys()
            .cloned()
            .collect();
        Ok(Some(keys))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codec::hash_to_cid;
    use serde::Serialize;

    #[derive(Serialize)]
    struct Sample {
        n: u32,
    }

    fn sample_block(n: u32) -> (Cid, Bytes) {
        let (bytes, cid) = hash_to_cid(&Sample { n }).expect("encode");
        (cid, bytes)
    }

    #[test]
    fn round_trip_put_get_has_delete() {
        let store = MemoryBlockstore::new();
        let (cid, bytes) = sample_block(1);

        assert!(!store.has(&cid).unwrap());
        assert_eq!(store.get(&cid).unwrap(), None);

        store.put(cid.clone(), bytes.clone()).unwrap();
        assert!(store.has(&cid).unwrap());
        assert_eq!(store.get(&cid).unwrap(), Some(bytes));
        assert_eq!(store.len(), 1);

        store.delete(&cid).unwrap();
        assert!(!store.has(&cid).unwrap());
        assert_eq!(store.len(), 0);
    }

    #[test]
    fn put_is_idempotent() {
        let store = MemoryBlockstore::new();
        let (cid, bytes) = sample_block(42);
        store.put(cid.clone(), bytes.clone()).unwrap();
        store.put(cid, bytes).unwrap();
        assert_eq!(store.len(), 1);
    }

    #[test]
    fn distinct_content_distinct_slots() {
        let store = MemoryBlockstore::new();
        let (cid1, b1) = sample_block(1);
        let (cid2, b2) = sample_block(2);
        assert_ne!(cid1, cid2);
        store.put(cid1, b1).unwrap();
        store.put(cid2, b2).unwrap();
        assert_eq!(store.len(), 2);
    }

    #[test]
    fn deleting_missing_is_ok() {
        let store = MemoryBlockstore::new();
        let (cid, _) = sample_block(99);
        store.delete(&cid).unwrap();
    }

    #[test]
    fn put_rejects_wrong_cid_in_every_build() {
        // Verification is unconditional now (debug + release) so that
        // an attacker-controlled put from CAR import, HTTP ingest, or
        // MCP tool input cannot poison the content-addressed store
        // with a (cid, bytes) pair where the CID doesn't match the
        // bytes.
        let store = MemoryBlockstore::new();
        let (cid, _) = sample_block(1);
        let wrong_bytes = Bytes::from_static(b"this is not the sample");

        let err = store.put(cid.clone(), wrong_bytes).unwrap_err();
        match err {
            StoreError::CidMismatch { claimed, .. } => {
                assert_eq!(claimed, cid);
            }
            e => panic!("wrong variant: {e:?}"),
        }
    }

    #[test]
    fn put_trusted_round_trips_without_verify() {
        // `put_trusted` is the audited fast path: the caller has
        // already computed `cid` from `bytes` via `hash_to_cid`.
        // Round-trip must match the verified `put` path byte-for-byte.
        let store = MemoryBlockstore::new();
        let (cid, bytes) = sample_block(3);
        store.put_trusted(cid.clone(), bytes.clone()).unwrap();
        assert!(store.has(&cid).unwrap());
        assert_eq!(store.get(&cid).unwrap(), Some(bytes));
    }

    #[test]
    fn put_trusted_does_not_verify_by_design() {
        // `put_trusted` has a Safety contract: the caller MUST have
        // established `data` hashes to `cid`. If the caller violates
        // that contract, the store accepts the mismatched pair (this
        // is the whole point - no recompute). This test pins the
        // behaviour so a future accidental "safety net" inside
        // `put_trusted` cannot regress the perf contract without
        // flagging a break in the test suite.
        //
        // `put` (not `put_trusted`) is what protects against hostile
        // callers; see `put_rejects_wrong_cid_in_every_build`.
        let store = MemoryBlockstore::new();
        let (cid, _) = sample_block(1);
        let wrong_bytes = Bytes::from_static(b"this is not the sample");

        // MUST succeed: no verify in the trusted path.
        store
            .put_trusted(cid.clone(), wrong_bytes.clone())
            .expect("put_trusted skips verify");
        // And the store happily returns the (corrupt) bytes the
        // caller claimed for that CID.
        assert_eq!(store.get(&cid).unwrap(), Some(wrong_bytes));
    }

    #[test]
    fn recompute_cid_matches_hash_to_cid() {
        let (cid, bytes) = sample_block(7);
        let recomputed = recompute_cid(&cid, &bytes).expect("sha2-256 supported");
        assert_eq!(cid, recomputed);
    }

    // ---- iter_from_root default impl ----

    /// Put an `Ipld` value into the store, returning its CID.
    fn put_ipld(store: &MemoryBlockstore, value: &ipld_core::ipld::Ipld) -> Cid {
        let (bytes, cid) = crate::codec::hash_to_cid(value).expect("encode");
        store.put(cid.clone(), bytes).expect("put");
        cid
    }

    /// Build an `Ipld::Link` pointing at one of our own Cids.
    fn our_to_ipld_link(ours: &Cid) -> ipld_core::ipld::Ipld {
        let inner = ipld_core::cid::Cid::try_from(ours.to_bytes().as_slice()).expect("cid rt");
        ipld_core::ipld::Ipld::Link(inner)
    }

    #[test]
    fn iter_from_root_single_leaf_block() {
        // A leaf block (no outgoing links) yields exactly one entry: itself.
        let store = MemoryBlockstore::new();
        let (cid, bytes) = sample_block(11);
        store.put(cid.clone(), bytes.clone()).unwrap();

        let collected: Result<Vec<_>, _> = store.iter_from_root(&cid).collect();
        let collected = collected.expect("walk ok");
        assert_eq!(collected.len(), 1);
        assert_eq!(collected[0].0, cid);
        assert_eq!(collected[0].1, bytes);
    }

    #[test]
    fn iter_from_root_walks_multi_block_dag_and_dedupes() {
        // Shape:
        //   root -> [a, b, a]   (duplicate reference to `a`)
        //   a    -> {tag: "a", child: c}
        //   b    -> {tag: "b", child: c}
        //   c    -> leaf
        // Reachable set = {root, a, b, c}. `c` must appear exactly
        // once even though it's reachable via two paths.
        let store = MemoryBlockstore::new();

        let c_cid = put_ipld(&store, &ipld_core::ipld::Ipld::String("leaf".into()));
        let a_cid = put_ipld(
            &store,
            &ipld_core::ipld::Ipld::Map(
                [
                    ("tag".to_string(), ipld_core::ipld::Ipld::String("a".into())),
                    ("child".to_string(), our_to_ipld_link(&c_cid)),
                ]
                .into_iter()
                .collect(),
            ),
        );
        let b_cid = put_ipld(
            &store,
            &ipld_core::ipld::Ipld::Map(
                [
                    ("tag".to_string(), ipld_core::ipld::Ipld::String("b".into())),
                    ("child".to_string(), our_to_ipld_link(&c_cid)),
                ]
                .into_iter()
                .collect(),
            ),
        );
        let root_cid = put_ipld(
            &store,
            &ipld_core::ipld::Ipld::List(vec![
                our_to_ipld_link(&a_cid),
                our_to_ipld_link(&b_cid),
                our_to_ipld_link(&a_cid),
            ]),
        );

        let collected: Result<Vec<_>, _> = store.iter_from_root(&root_cid).collect();
        let collected = collected.expect("walk ok");
        let cids: Vec<Cid> = collected.iter().map(|(c, _)| c.clone()).collect();

        assert_eq!(cids.len(), 4, "exactly the reachable set, got {cids:?}");
        let unique: std::collections::BTreeSet<_> = cids.iter().collect();
        assert_eq!(unique.len(), 4, "no duplicates");
        assert!(cids.contains(&root_cid));
        assert!(cids.contains(&a_cid));
        assert!(cids.contains(&b_cid));
        assert!(cids.contains(&c_cid));
        // Pre-order: root yields first.
        assert_eq!(cids[0], root_cid);
    }

    #[test]
    fn iter_from_root_missing_root_errors() {
        // A root CID that was never `put` surfaces a clean NotFound,
        // and the iterator fuses (no further items).
        let store = MemoryBlockstore::new();
        let (cid, _bytes) = sample_block(42);

        let mut iter = store.iter_from_root(&cid);
        match iter.next() {
            Some(Err(StoreError::NotFound { cid: got })) => assert_eq!(got, cid),
            other => panic!("expected NotFound, got {other:?}"),
        }
        assert!(iter.next().is_none(), "iterator must fuse after error");
    }
}
