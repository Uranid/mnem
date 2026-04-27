//! Point lookup over a Prolly tree.
//!
//! Walks from the root to the leaf whose key-range contains `key`, binary-
//! searches the leaf's entries, and returns the associated value CID (or
//! `None` if the key isn't present). Work is `O(log_n_branching × log_chunk_size)`
//! which for mnem's parameters (≈64-entry leaves, a few levels) is a handful
//! of block fetches.

use crate::error::Error;
use crate::id::Cid;
use crate::prolly::constants::ProllyKey;
use crate::prolly::tree::{TreeChunk, load_tree_chunk};
use crate::store::Blockstore;

/// Return the value CID associated with `key` in the Prolly tree rooted at
/// `root`, or `None` if `key` is not present.
///
/// # Errors
///
/// Propagates store and codec errors. Missing chunks return
/// [`crate::error::StoreError::NotFound`] and indicate a structurally
/// broken tree.
pub fn lookup<B: Blockstore + ?Sized>(
    store: &B,
    root: &Cid,
    key: &ProllyKey,
) -> Result<Option<Cid>, Error> {
    let mut cid = root.clone();
    loop {
        match load_tree_chunk(store, &cid)? {
            TreeChunk::Leaf(leaf) => {
                return Ok(leaf
                    .entries
                    .binary_search_by_key(key, |e| e.0)
                    .ok()
                    .map(|i| leaf.entries[i].1.clone()));
            }
            TreeChunk::Internal(internal) => {
                // `boundaries[i]` is the inclusive lower bound of child `i + 1`.
                // Pick the largest child index whose lower bound ≤ key.
                // Equivalently: `partition_point(|b| b <= key)`.
                let child_idx = internal.boundaries.partition_point(|b| b <= key);
                cid = internal.children[child_idx].clone();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::id::{CODEC_RAW, Multihash};
    use crate::prolly::tree::build_tree;
    use crate::store::MemoryBlockstore;

    fn keyed(i: u32) -> ProllyKey {
        let mut k = [0u8; 16];
        k[12..16].copy_from_slice(&i.to_be_bytes());
        ProllyKey(k)
    }

    fn val(i: u32) -> Cid {
        Cid::new(CODEC_RAW, Multihash::sha2_256(&i.to_be_bytes()))
    }

    #[test]
    fn finds_existing_key_in_single_leaf() {
        let store = MemoryBlockstore::new();
        let entries: Vec<_> = (0..10u32).map(|i| (keyed(i), val(i))).collect();
        let root = build_tree(&store, entries).unwrap();
        for i in 0..10u32 {
            let got = lookup(&store, &root, &keyed(i)).unwrap();
            assert_eq!(got, Some(val(i)), "key {i}");
        }
    }

    #[test]
    fn returns_none_for_missing_key() {
        let store = MemoryBlockstore::new();
        let entries: Vec<_> = (0..10u32).map(|i| (keyed(i * 2), val(i))).collect();
        let root = build_tree(&store, entries).unwrap();
        // Odd keys are absent.
        assert_eq!(lookup(&store, &root, &keyed(1)).unwrap(), None);
        assert_eq!(lookup(&store, &root, &keyed(99)).unwrap(), None);
    }

    #[test]
    fn finds_keys_in_multi_level_tree() {
        let store = MemoryBlockstore::new();
        let entries: Vec<_> = (0..5_000u32).map(|i| (keyed(i), val(i))).collect();
        let root = build_tree(&store, entries).unwrap();
        // Sample across the key space.
        for &probe in &[0u32, 1, 7, 100, 1_234, 2_500, 4_999] {
            let got = lookup(&store, &root, &keyed(probe)).unwrap();
            assert_eq!(got, Some(val(probe)), "key {probe}");
        }
        // Just outside each end.
        assert_eq!(lookup(&store, &root, &keyed(5_000)).unwrap(), None);
    }

    #[test]
    fn lookup_handles_empty_tree() {
        let store = MemoryBlockstore::new();
        let root = build_tree(&store, std::iter::empty()).unwrap();
        assert_eq!(lookup(&store, &root, &keyed(42)).unwrap(), None);
    }
}
