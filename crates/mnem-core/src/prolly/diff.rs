//! Structural diff of two Prolly trees.
//!
//! The diff walks both trees simultaneously. Any pair of child CIDs that
//! compare equal is pruned immediately - the subtrees are byte-identical by
//! construction, so they cannot contain any differences. This is the
//! Prolly payoff: a 10,000-entry tree with 10 edits is diffed by touching
//! only the O(d × log n) chunks on the edited paths, not the whole tree.
//!
//! Output is a `Vec<DiffEntry>` sorted ascending by key. Entries describe
//! insertions, removals, and value changes from `a` to `b`.

use std::cmp::Ordering;

use crate::error::Error;
use crate::id::Cid;
use crate::prolly::constants::ProllyKey;
use crate::prolly::tree::{Internal, TreeChunk, load_tree_chunk};
use crate::store::Blockstore;

/// A single difference between two Prolly trees.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DiffEntry {
    /// Key is present in `b` but not `a`.
    Added {
        /// The new key.
        key: ProllyKey,
        /// Its value CID in `b`.
        value: Cid,
    },
    /// Key is present in `a` but not `b`.
    Removed {
        /// The removed key.
        key: ProllyKey,
        /// Its value CID in `a`.
        value: Cid,
    },
    /// Key is present in both trees but with different value CIDs.
    Changed {
        /// The key with a changed value.
        key: ProllyKey,
        /// Old value CID (`a`-side).
        before: Cid,
        /// New value CID (`b`-side).
        after: Cid,
    },
}

impl DiffEntry {
    /// The key affected by this diff entry.
    #[must_use]
    pub const fn key(&self) -> &ProllyKey {
        match self {
            Self::Added { key, .. } | Self::Removed { key, .. } | Self::Changed { key, .. } => key,
        }
    }
}

/// Compute the difference between the two Prolly trees `a` and `b`.
///
/// Returns every added / removed / changed entry, in ascending key order.
///
/// # Errors
///
/// Propagates store and codec errors.
pub fn diff<B: Blockstore + ?Sized>(
    store: &B,
    root_a: &Cid,
    root_b: &Cid,
) -> Result<Vec<DiffEntry>, Error> {
    let mut out = Vec::new();
    diff_cids(store, root_a, root_b, &mut out)?;
    Ok(out)
}

// ---------------- Recursive core ----------------

fn diff_cids<B: Blockstore + ?Sized>(
    store: &B,
    a: &Cid,
    b: &Cid,
    out: &mut Vec<DiffEntry>,
) -> Result<(), Error> {
    if a == b {
        return Ok(());
    }
    let chunk_a = load_tree_chunk(store, a)?;
    let chunk_b = load_tree_chunk(store, b)?;
    match (chunk_a, chunk_b) {
        (TreeChunk::Leaf(la), TreeChunk::Leaf(lb)) => {
            diff_sorted_entries(&la.entries, &lb.entries, out);
            Ok(())
        }
        (TreeChunk::Internal(ia), TreeChunk::Internal(ib)) => diff_internals(store, &ia, &ib, out),
        (TreeChunk::Leaf(la), TreeChunk::Internal(ib)) => {
            let mut flat = Vec::new();
            collect_entries_internal(store, &ib, &mut flat)?;
            diff_sorted_entries(&la.entries, &flat, out);
            Ok(())
        }
        (TreeChunk::Internal(ia), TreeChunk::Leaf(lb)) => {
            let mut flat = Vec::new();
            collect_entries_internal(store, &ia, &mut flat)?;
            diff_sorted_entries(&flat, &lb.entries, out);
            Ok(())
        }
    }
}

fn diff_internals<B: Blockstore + ?Sized>(
    store: &B,
    ia: &Internal,
    ib: &Internal,
    out: &mut Vec<DiffEntry>,
) -> Result<(), Error> {
    // Merge the two boundary sets into a common set of key-ranges.
    let mut all: Vec<ProllyKey> = Vec::with_capacity(ia.boundaries.len() + ib.boundaries.len());
    all.extend_from_slice(&ia.boundaries);
    all.extend_from_slice(&ib.boundaries);
    all.sort();
    all.dedup();

    let n_ranges = all.len() + 1;
    let mut last_pair: Option<(usize, usize)> = None;

    for range_idx in 0..n_ranges {
        // A range is identified by its lower boundary: range 0 has no lower
        // bound (neg-infinity), range i>0 has lower bound `all[i - 1]`.
        // For each side, find the child whose range overlaps this one by
        // `partition_point(|b| b <= lower_bound)`.
        let lower = if range_idx == 0 {
            None
        } else {
            Some(all[range_idx - 1])
        };
        let a_idx = child_index_for(&ia.boundaries, lower.as_ref());
        let b_idx = child_index_for(&ib.boundaries, lower.as_ref());

        // Consecutive ranges that map to the same child pair were already
        // handled by the previous iteration - skip to avoid duplicate
        // recursion.
        if Some((a_idx, b_idx)) == last_pair {
            continue;
        }
        last_pair = Some((a_idx, b_idx));

        diff_cids(store, &ia.children[a_idx], &ib.children[b_idx], out)?;
    }
    Ok(())
}

fn child_index_for(boundaries: &[ProllyKey], lower: Option<&ProllyKey>) -> usize {
    match lower {
        None => 0,
        Some(k) => boundaries.partition_point(|b| b <= k),
    }
}

fn collect_entries_internal<B: Blockstore + ?Sized>(
    store: &B,
    internal: &Internal,
    out: &mut Vec<(ProllyKey, Cid)>,
) -> Result<(), Error> {
    for child_cid in &internal.children {
        match load_tree_chunk(store, child_cid)? {
            TreeChunk::Leaf(l) => out.extend(l.entries),
            TreeChunk::Internal(i) => collect_entries_internal(store, &i, out)?,
        }
    }
    Ok(())
}

/// Merge-diff two sorted `(key, value)` slices. Pushes [`DiffEntry`]
/// variants to `out` in ascending key order.
fn diff_sorted_entries(a: &[(ProllyKey, Cid)], b: &[(ProllyKey, Cid)], out: &mut Vec<DiffEntry>) {
    let mut i = 0;
    let mut j = 0;
    while i < a.len() && j < b.len() {
        match a[i].0.cmp(&b[j].0) {
            Ordering::Less => {
                out.push(DiffEntry::Removed {
                    key: a[i].0,
                    value: a[i].1.clone(),
                });
                i += 1;
            }
            Ordering::Greater => {
                out.push(DiffEntry::Added {
                    key: b[j].0,
                    value: b[j].1.clone(),
                });
                j += 1;
            }
            Ordering::Equal => {
                if a[i].1 != b[j].1 {
                    out.push(DiffEntry::Changed {
                        key: a[i].0,
                        before: a[i].1.clone(),
                        after: b[j].1.clone(),
                    });
                }
                i += 1;
                j += 1;
            }
        }
    }
    while i < a.len() {
        out.push(DiffEntry::Removed {
            key: a[i].0,
            value: a[i].1.clone(),
        });
        i += 1;
    }
    while j < b.len() {
        out.push(DiffEntry::Added {
            key: b[j].0,
            value: b[j].1.clone(),
        });
        j += 1;
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
    fn diff_of_identical_roots_is_empty() {
        let entries: Vec<_> = (0..1_000u32).map(|i| (keyed(i), val(i))).collect();
        let store = MemoryBlockstore::new();
        let root = build_tree(&store, entries).unwrap();
        let changes = diff(&store, &root, &root).unwrap();
        assert!(changes.is_empty());
    }

    #[test]
    fn diff_detects_added_key() {
        let base: Vec<_> = (0..100u32).map(|i| (keyed(i), val(i))).collect();
        let mut extended = base.clone();
        extended.push((keyed(1_000), val(1_000)));
        let store = MemoryBlockstore::new();
        let root_a = build_tree(&store, base).unwrap();
        let root_b = build_tree(&store, extended).unwrap();
        let changes = diff(&store, &root_a, &root_b).unwrap();
        assert_eq!(
            changes,
            vec![DiffEntry::Added {
                key: keyed(1_000),
                value: val(1_000),
            }]
        );
    }

    #[test]
    fn diff_detects_removed_key() {
        let extended: Vec<_> = (0..100u32).map(|i| (keyed(i), val(i))).collect();
        let mut base = extended.clone();
        base.retain(|(k, _)| *k != keyed(50));
        let store = MemoryBlockstore::new();
        let root_a = build_tree(&store, extended).unwrap();
        let root_b = build_tree(&store, base).unwrap();
        let changes = diff(&store, &root_a, &root_b).unwrap();
        assert_eq!(
            changes,
            vec![DiffEntry::Removed {
                key: keyed(50),
                value: val(50),
            }]
        );
    }

    #[test]
    fn diff_detects_changed_value() {
        let a: Vec<_> = (0..100u32).map(|i| (keyed(i), val(i))).collect();
        let mut b = a.clone();
        b[50].1 = val(999);
        let store = MemoryBlockstore::new();
        let root_a = build_tree(&store, a).unwrap();
        let root_b = build_tree(&store, b).unwrap();
        let changes = diff(&store, &root_a, &root_b).unwrap();
        assert_eq!(
            changes,
            vec![DiffEntry::Changed {
                key: keyed(50),
                before: val(50),
                after: val(999),
            }]
        );
    }

    #[test]
    fn diff_detects_many_changes_in_order() {
        let a: Vec<_> = (0..1_000u32).map(|i| (keyed(i), val(i))).collect();
        let mut b = a.clone();
        for i in (0..1_000u32).step_by(100) {
            b[i as usize].1 = val(10_000 + i);
        }
        let store = MemoryBlockstore::new();
        let root_a = build_tree(&store, a).unwrap();
        let root_b = build_tree(&store, b).unwrap();
        let changes = diff(&store, &root_a, &root_b).unwrap();
        assert_eq!(changes.len(), 10);
        for w in changes.windows(2) {
            assert!(w[0].key() < w[1].key());
        }
        for c in &changes {
            match c {
                DiffEntry::Changed { .. } => {}
                e => panic!("expected Changed, got {e:?}"),
            }
        }
    }

    #[test]
    fn diff_detects_add_remove_change_mixture() {
        // a: {0..100}. b: {1..100} + {200} + changed value at 50.
        let a: Vec<_> = (0..100u32).map(|i| (keyed(i), val(i))).collect();
        let mut b: Vec<(ProllyKey, Cid)> = (1..100u32).map(|i| (keyed(i), val(i))).collect();
        // Mutate value at key=50
        let k50_idx = b.iter().position(|(k, _)| *k == keyed(50)).unwrap();
        b[k50_idx].1 = val(9_999);
        // Add key=200
        b.push((keyed(200), val(200)));
        let store = MemoryBlockstore::new();
        let root_a = build_tree(&store, a).unwrap();
        let root_b = build_tree(&store, b).unwrap();
        let changes = diff(&store, &root_a, &root_b).unwrap();
        // Expect: Removed(0), Changed(50, 50 -> 9999), Added(200)
        assert_eq!(changes.len(), 3);
        assert_eq!(
            changes[0],
            DiffEntry::Removed {
                key: keyed(0),
                value: val(0),
            }
        );
        assert_eq!(
            changes[1],
            DiffEntry::Changed {
                key: keyed(50),
                before: val(50),
                after: val(9_999),
            }
        );
        assert_eq!(
            changes[2],
            DiffEntry::Added {
                key: keyed(200),
                value: val(200),
            }
        );
    }
}
