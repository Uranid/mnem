//! Op-heads store - the set of current operation-DAG heads.
//!
//! Per SPEC §6.1 and §7.3, a repository's "current state" is the set of
//! operations with no children yet. When a mutating command commits, it
//! atomically adds its new `OperationId` to this set and removes the
//! parents it supersedes.
//!
//! Separating op-heads from the object store lets "what exists" (object
//! store) and "what's current" (op-heads) use different substrates - e.g.
//! S3 for ops, `DynamoDB` for heads . At the in-memory /
//! Simple-backend level the distinction is cosmetic but the trait
//! boundary is load-bearing for future backends.

use core::fmt;
use std::collections::HashSet;
use std::sync::Mutex;

use crate::error::StoreError;
use crate::id::Cid;

/// The set of current operation heads.
///
/// Implementations MUST be `Send + Sync` so a single instance can be
/// shared across concurrent readers and a single writer.
pub trait OpHeadsStore: Send + Sync + fmt::Debug {
    /// Return the current heads (sorted ascending for determinism).
    ///
    /// # Errors
    ///
    /// Returns a backend-specific I/O error.
    fn current(&self) -> Result<Vec<Cid>, StoreError>;

    /// Atomically replace the head set: insert `new`, remove each
    /// element of `supersedes`. Removes of missing IDs are a no-op.
    ///
    /// Order inside the operation is write-then-remove so a crash
    /// between the two steps leaves a union of old-and-new heads, which
    /// the next reader will merge.
    ///
    /// # Errors
    ///
    /// Returns a backend-specific I/O error.
    fn update(&self, new: Cid, supersedes: &[Cid]) -> Result<(), StoreError>;
}

// ---------------- In-memory reference implementation ----------------

/// In-process op-heads store backed by a [`HashSet`] behind a [`Mutex`].
///
/// Suitable for tests, WASM default storage, and single-agent loops.
/// Not durable across process restarts.
#[derive(Default)]
pub struct MemoryOpHeadsStore {
    heads: Mutex<HashSet<Cid>>,
}

impl MemoryOpHeadsStore {
    /// Create an empty store.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Current head count (cheap, for tests).
    ///
    /// # Panics
    ///
    /// Panics if the internal mutex is poisoned.
    #[must_use]
    pub fn len(&self) -> usize {
        self.heads.lock().expect("mutex not poisoned").len()
    }

    /// Whether the head set is empty.
    ///
    /// # Panics
    ///
    /// Panics if the internal mutex is poisoned.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl fmt::Debug for MemoryOpHeadsStore {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "MemoryOpHeadsStore {{ n_heads: {} }}", self.len())
    }
}

impl OpHeadsStore for MemoryOpHeadsStore {
    fn current(&self) -> Result<Vec<Cid>, StoreError> {
        let mut v: Vec<Cid> = self
            .heads
            .lock()
            .expect("mutex not poisoned")
            .iter()
            .cloned()
            .collect();
        v.sort();
        Ok(v)
    }

    fn update(&self, new: Cid, supersedes: &[Cid]) -> Result<(), StoreError> {
        let mut h = self.heads.lock().expect("mutex not poisoned");
        // Insert new first (SPEC §7.3 ordering: new-then-delete survives crashes).
        h.insert(new);
        for s in supersedes {
            h.remove(s);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::id::{CODEC_RAW, Multihash};

    fn raw(n: u32) -> Cid {
        Cid::new(CODEC_RAW, Multihash::sha2_256(&n.to_be_bytes()))
    }

    #[test]
    fn empty_store_has_no_heads() {
        let s = MemoryOpHeadsStore::new();
        assert!(s.current().unwrap().is_empty());
        assert_eq!(s.len(), 0);
    }

    #[test]
    fn single_head_round_trips() {
        let s = MemoryOpHeadsStore::new();
        let root = raw(1);
        s.update(root.clone(), &[]).unwrap();
        assert_eq!(s.current().unwrap(), vec![root]);
    }

    #[test]
    fn update_supersedes_old_head() {
        let s = MemoryOpHeadsStore::new();
        let root = raw(1);
        s.update(root.clone(), &[]).unwrap();
        let child = raw(2);
        s.update(child.clone(), &[root]).unwrap();
        assert_eq!(s.current().unwrap(), vec![child]);
    }

    #[test]
    fn concurrent_commits_produce_multiple_heads() {
        let s = MemoryOpHeadsStore::new();
        let root = raw(1);
        s.update(root.clone(), &[]).unwrap();
        // Two concurrent writers both observe `root` and each commit a
        // distinct child without superseding each other's.
        let left = raw(10);
        let right = raw(11);
        s.update(left.clone(), std::slice::from_ref(&root)).unwrap();
        // Second writer races: sees root (now gone), tries to remove, no-op.
        s.update(right.clone(), &[root]).unwrap();
        let heads = s.current().unwrap();
        assert_eq!(heads.len(), 2, "two concurrent heads expected");
        assert!(heads.contains(&left));
        assert!(heads.contains(&right));
    }

    #[test]
    fn supersedes_returns_sorted() {
        let s = MemoryOpHeadsStore::new();
        s.update(raw(3), &[]).unwrap();
        s.update(raw(1), &[]).unwrap();
        s.update(raw(2), &[]).unwrap();
        let heads = s.current().unwrap();
        assert!(heads.windows(2).all(|w| w[0] < w[1]));
    }
}
