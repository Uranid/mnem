//! Ordered cursor over a Prolly tree.
//!
//! [`Cursor`] yields `(ProllyKey, Cid)` pairs in ascending key order by
//! walking the Merkle DAG depth-first along the left spine. Usable as a
//! standard Rust [`Iterator`] (`for (k, v) in cursor { ... }`), so range
//! scans and full enumeration both compose naturally.
//!
//! Memory: the cursor's stack holds at most one decoded [`Internal`] per
//! tree level, plus the current leaf. For mnem's branching factor and
//! typical graph scale, that's kilobytes - not megabytes.

use crate::error::Error;
use crate::id::Cid;
use crate::prolly::constants::ProllyKey;
use crate::prolly::tree::{Internal, Leaf, TreeChunk, load_tree_chunk};
use crate::store::Blockstore;

/// A forward-only cursor over the `(key, value)` entries of a Prolly tree.
///
/// Use [`Cursor::new`] to open, then drive via [`Iterator::next`] or a
/// `for` loop.
pub struct Cursor<'a, B: Blockstore + ?Sized> {
    store: &'a B,
    /// Open internal-node frames: `(chunk, next_child_idx)`.
    /// `next_child_idx` is the index of the next unvisited child.
    stack: Vec<(Internal, usize)>,
    /// Currently loaded leaf and the index of the next unread entry.
    current: Option<(Leaf, usize)>,
}

impl<'a, B: Blockstore + ?Sized> Cursor<'a, B> {
    /// Open a cursor at the leftmost leaf of the tree rooted at `root`.
    ///
    /// # Errors
    ///
    /// Propagates store and codec errors.
    pub fn new(store: &'a B, root: &Cid) -> Result<Self, Error> {
        let mut c = Self {
            store,
            stack: Vec::new(),
            current: None,
        };
        c.descend(root)?;
        Ok(c)
    }

    /// Descend from `start` along the left spine until a leaf is reached,
    /// pushing every internal node encountered onto the stack with
    /// `next_child_idx = 1` (child 0 is the one we descended into).
    fn descend(&mut self, start: &Cid) -> Result<(), Error> {
        let mut cid = start.clone();
        loop {
            match load_tree_chunk(self.store, &cid)? {
                TreeChunk::Leaf(leaf) => {
                    self.current = Some((leaf, 0));
                    return Ok(());
                }
                TreeChunk::Internal(internal) => {
                    let next = internal.children[0].clone();
                    self.stack.push((internal, 1));
                    cid = next;
                }
            }
        }
    }

    /// Move to the next leaf in key order, loading it into `self.current`.
    /// Returns `Ok(false)` when the tree is fully traversed.
    fn advance_leaf(&mut self) -> Result<bool, Error> {
        loop {
            let (internal, next_idx) = match self.stack.last_mut() {
                Some(f) => f,
                None => return Ok(false),
            };
            if *next_idx < internal.children.len() {
                let next_cid = internal.children[*next_idx].clone();
                *next_idx += 1;
                self.descend(&next_cid)?;
                return Ok(true);
            }
            self.stack.pop();
        }
    }
}

impl<B: Blockstore + ?Sized> Iterator for Cursor<'_, B> {
    type Item = Result<(ProllyKey, Cid), Error>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if let Some((leaf, idx)) = &mut self.current {
                if *idx < leaf.entries.len() {
                    let (k, v) = leaf.entries[*idx].clone();
                    *idx += 1;
                    return Some(Ok((k, v)));
                }
                self.current = None;
            }
            match self.advance_leaf() {
                Ok(true) => continue, // outer loop reads from new leaf
                Ok(false) => return None,
                Err(e) => return Some(Err(e)),
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
    fn cursor_iterates_empty_tree_to_nothing() {
        let store = MemoryBlockstore::new();
        let root = build_tree(&store, std::iter::empty()).unwrap();
        let cursor = Cursor::new(&store, &root).unwrap();
        let collected: Vec<_> = cursor.map(|r| r.unwrap()).collect();
        assert!(collected.is_empty());
    }

    #[test]
    fn cursor_iterates_single_entry() {
        let store = MemoryBlockstore::new();
        let root = build_tree(&store, [(keyed(7), val(42))]).unwrap();
        let cursor = Cursor::new(&store, &root).unwrap();
        let collected: Vec<_> = cursor.map(|r| r.unwrap()).collect();
        assert_eq!(collected, vec![(keyed(7), val(42))]);
    }

    #[test]
    fn cursor_iterates_single_leaf_in_key_order() {
        let store = MemoryBlockstore::new();
        let entries: Vec<_> = (0..10u32).map(|i| (keyed(i), val(i))).collect();
        let root = build_tree(&store, entries.clone()).unwrap();
        let cursor = Cursor::new(&store, &root).unwrap();
        let collected: Vec<_> = cursor.map(|r| r.unwrap()).collect();
        assert_eq!(collected, entries);
    }

    #[test]
    fn cursor_iterates_multi_level_tree_in_key_order() {
        let store = MemoryBlockstore::new();
        let entries: Vec<_> = (0..5_000u32).map(|i| (keyed(i), val(i))).collect();
        let root = build_tree(&store, entries.clone()).unwrap();
        let cursor = Cursor::new(&store, &root).unwrap();
        let collected: Vec<_> = cursor.map(|r| r.unwrap()).collect();
        assert_eq!(collected.len(), 5_000);
        // Check sort order.
        for w in collected.windows(2) {
            assert!(
                w[0].0 < w[1].0,
                "out-of-order: {:?} then {:?}",
                w[0].0,
                w[1].0
            );
        }
        assert_eq!(collected, entries);
    }
}
