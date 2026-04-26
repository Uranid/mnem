//! Lowest-common-ancestor (LCA) finder on the op-DAG.
//!
//! Extracted from `merge.rs` as part of Phase-B4.1. This module is the
//! authoritative source for all ancestor-walking and LCA computation
//! used by the 3-way merge (B4.2 will consume it for conflict
//! detection; B4.3 will drive it from the CLI).
//!
//! **Semantics preserved byte-for-byte from the pre-extraction code**
//! so existing determinism gates hold:
//!
//! - `find_lca_many` returns the same CID for the same input `heads`
//!   as the previous inline `common_ancestor`.
//! - On multiple incomparable LCAs (criss-cross), the smallest-CID
//!   candidate wins. A recursive-virtual-base strategy (git-style) is
//!   left as a future extension; the current smallest-CID tiebreak is
//!   already deterministic and is what the existing `build_merge_commit`
//!   tests pin down.
//!
//! # Non-goals
//!
//! - Persistent LCA cache: [`LcaCache`] is per-session memoization only.
//!   A sqlite-backed cache is deferred to a later wave (see B4.x).
//! - Cross-commit LCA (Commit DAG): current implementation operates on
//!   the op-DAG (`Operation::parents`). Commit-DAG LCA lives in later
//!   phases; the API shape here is designed to accept any DAG whose
//!   node type serializes as an `Operation`.
//!
//! # API
//!
//! - [`find_lca`] - pairwise convenience for two heads.
//! - [`find_lca_many`] - generalised N-head LCA (octopus merges).
//! - [`LcaCache`] - optional in-memory memoization.

use std::collections::BTreeSet;
use std::collections::HashMap;

use crate::error::Error;
use crate::id::Cid;
use crate::objects::Operation;
use crate::store::Blockstore;

use super::readonly::decode_from_store;

/// Pair-wise LCA on the op-DAG.
///
/// Returns `Ok(Some(cid))` when `left` and `right` share at least one
/// ancestor (always true in a well-formed mnem repo: both descend from
/// the root Operation). Returns `Ok(None)` for disjoint histories.
///
/// Pair order is canonicalised internally (`min`, `max` by CID byte
/// order) so `find_lca(a, b) == find_lca(b, a)`.
pub fn find_lca(bs: &dyn Blockstore, left: Cid, right: Cid) -> Result<Option<Cid>, Error> {
    let (a, b) = if left <= right {
        (left, right)
    } else {
        (right, left)
    };
    find_lca_many(bs, &[a, b])
}

/// N-head LCA on the op-DAG.
///
/// - `heads.is_empty()` → `Ok(None)`.
/// - `heads.len() == 1` → `Ok(Some(heads[0]))` (LCA of a single head is
///   itself).
/// - `heads.len() >= 2` → intersection-of-ancestors, filtered to those
///   not strictly-ancestor of another element of the intersection.
///   Multiple incomparable LCAs are broken with the smallest-CID-wins
///   rule for determinism.
///
/// Returns `Ok(None)` when the input heads share no common ancestor
/// (disjoint histories). Call sites that require a common ancestor
/// (e.g. `merge_op_heads`) should convert this into their own error.
pub fn find_lca_many(bs: &dyn Blockstore, heads: &[Cid]) -> Result<Option<Cid>, Error> {
    match heads.len() {
        0 => return Ok(None),
        1 => return Ok(Some(heads[0].clone())),
        _ => {}
    }

    // Ancestors-inclusive for each head.
    let mut per_head: Vec<BTreeSet<Cid>> = Vec::with_capacity(heads.len());
    for h in heads {
        per_head.push(ancestors_inclusive(bs, h)?);
    }

    // Intersection = common ancestors.
    let mut common: BTreeSet<Cid> = per_head[0].clone();
    for s in &per_head[1..] {
        common.retain(|c| s.contains(c));
    }
    if common.is_empty() {
        return Ok(None);
    }

    // An op is a LOWEST common ancestor iff no other common op has it
    // among its strict ancestors. Equivalently, keep ops in `common`
    // that are not strictly-ancestor to another op in `common`.
    let common_vec: Vec<Cid> = common.iter().cloned().collect();
    let mut is_strict_ancestor_of_other: BTreeSet<Cid> = BTreeSet::new();
    for c in &common_vec {
        let anc = ancestors_inclusive(bs, c)?;
        for a in &anc {
            if a != c && common.contains(a) {
                is_strict_ancestor_of_other.insert(a.clone());
            }
        }
    }
    let mut lcas: Vec<Cid> = common_vec
        .into_iter()
        .filter(|c| !is_strict_ancestor_of_other.contains(c))
        .collect();
    lcas.sort();
    // Safe: `common` is non-empty above; subtracting strict ancestors
    // leaves at least one minimal element.
    Ok(Some(lcas.into_iter().next().expect("common set non-empty")))
}

/// All ancestors of `cid` reachable through `Operation::parents`,
/// including `cid` itself. Uses a DFS / BFS hybrid (stack pop with
/// `seen` set) so traversal is O(|ancestors|) regardless of DAG shape.
pub(crate) fn ancestors_inclusive(bs: &dyn Blockstore, cid: &Cid) -> Result<BTreeSet<Cid>, Error> {
    let mut seen: BTreeSet<Cid> = BTreeSet::new();
    let mut stack: Vec<Cid> = vec![cid.clone()];
    while let Some(c) = stack.pop() {
        if !seen.insert(c.clone()) {
            continue;
        }
        let op: Operation = decode_from_store(bs, &c)?;
        for p in &op.parents {
            if !seen.contains(p) {
                stack.push(p.clone());
            }
        }
    }
    Ok(seen)
}

// ---------------- Cache ----------------

/// In-memory, per-session memoization of pairwise LCA lookups.
///
/// Keyed by a canonical `(min, max)` pair so `lca(a, b)` and
/// `lca(b, a)` hit the same slot. Not thread-safe by itself; wrap in
/// `Arc<Mutex<_>>` if shared across threads. A persistent sqlite-backed
/// cache is deferred to a later wave.
#[derive(Debug, Default)]
pub struct LcaCache {
    entries: HashMap<(Cid, Cid), Option<Cid>>,
    hits: u64,
    misses: u64,
}

impl LcaCache {
    /// Fresh cache with no entries.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Look up or compute the LCA of `(left, right)`. Result is cached
    /// for the lifetime of the `LcaCache`.
    pub fn get_or_compute(
        &mut self,
        bs: &dyn Blockstore,
        left: Cid,
        right: Cid,
    ) -> Result<Option<Cid>, Error> {
        let key = canonical_pair(&left, &right);
        if let Some(v) = self.entries.get(&key) {
            self.hits += 1;
            tracing::debug!(target: "mnem::lca", "cache hit: {:?}", key);
            return Ok(v.clone());
        }
        self.misses += 1;
        tracing::debug!(target: "mnem::lca", "cache miss: {:?}", key);
        let v = find_lca(bs, left, right)?;
        self.entries.insert(key, v.clone());
        Ok(v)
    }

    /// Observed cache hits since construction.
    #[must_use]
    pub fn hits(&self) -> u64 {
        self.hits
    }

    /// Observed cache misses since construction.
    #[must_use]
    pub fn misses(&self) -> u64 {
        self.misses
    }

    /// Number of cached `(left, right) -> LCA` entries.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// True iff no entries are cached.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

fn canonical_pair(a: &Cid, b: &Cid) -> (Cid, Cid) {
    if a <= b {
        (a.clone(), b.clone())
    } else {
        (b.clone(), a.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codec::hash_to_cid;
    use crate::objects::Operation;
    use crate::store::MemoryBlockstore;

    /// Build + store an Operation with the given parents. The `view`
    /// field is reused across nodes (any Cid is valid for LCA purposes
    /// - we only walk `parents`), and `time` encodes the test seed to
    /// keep every node's CID unique without depending on wall clock.
    fn put_op(bs: &MemoryBlockstore, seed: u64, parents: &[Cid]) -> Cid {
        // The view field just needs to be *some* Cid; we hash a unique
        // per-seed payload so we never accidentally collide two Ops.
        let (view_bytes, view_cid) = hash_to_cid(&format!("view-{seed}")).unwrap();
        bs.put_trusted(view_cid.clone(), view_bytes).unwrap();
        let mut op = Operation::new(view_cid, "test", seed, format!("op-{seed}"));
        for p in parents {
            op = op.with_parent(p.clone());
        }
        let (bytes, cid) = hash_to_cid(&op).unwrap();
        bs.put_trusted(cid.clone(), bytes).unwrap();
        cid
    }

    #[test]
    fn linear_history_same_head_returns_self() {
        // A <- B <- C.  LCA(C, C) == C.
        let bs = MemoryBlockstore::new();
        let a = put_op(&bs, 1, &[]);
        let b = put_op(&bs, 2, &[a.clone()]);
        let c = put_op(&bs, 3, &[b]);
        let lca = find_lca(&bs, c.clone(), c.clone()).unwrap();
        assert_eq!(lca, Some(c));
    }

    #[test]
    fn linear_history_ancestor_wins() {
        // A <- B <- C.  LCA(C, B) == B  (B is an ancestor of C).
        let bs = MemoryBlockstore::new();
        let a = put_op(&bs, 10, &[]);
        let b = put_op(&bs, 11, &[a]);
        let c = put_op(&bs, 12, &[b.clone()]);
        let lca = find_lca(&bs, c, b.clone()).unwrap();
        assert_eq!(lca, Some(b));
    }

    #[test]
    fn divergent_heads_lca_is_fork_point() {
        //   A <- B
        //    \- C          → LCA(B, C) == A
        let bs = MemoryBlockstore::new();
        let a = put_op(&bs, 20, &[]);
        let b = put_op(&bs, 21, &[a.clone()]);
        let c = put_op(&bs, 22, &[a.clone()]);
        let lca = find_lca(&bs, b, c).unwrap();
        assert_eq!(lca, Some(a));
    }

    #[test]
    fn lca_is_commutative() {
        // find_lca(a, b) == find_lca(b, a) regardless of CID order.
        let bs = MemoryBlockstore::new();
        let root = put_op(&bs, 30, &[]);
        let l = put_op(&bs, 31, &[root.clone()]);
        let r = put_op(&bs, 32, &[root]);
        let ab = find_lca(&bs, l.clone(), r.clone()).unwrap();
        let ba = find_lca(&bs, r, l).unwrap();
        assert_eq!(ab, ba);
    }

    #[test]
    fn criss_cross_picks_deterministic_base() {
        // Criss-cross:
        //        root
        //        /  \
        //       A    B      (A, B both share root)
        //       |\  /|
        //       | \/ |
        //       | /\ |
        //       |/  \|
        //       M    N      (M has parents A,B; N has parents A,B)
        //
        // Common ancestors of M and N = {root, A, B, M, N} ∩ ... =
        // {root, A, B}. Strict-ancestor filter removes root (ancestor
        // of A and B). Remaining LCA candidates = {A, B}. Tiebreak:
        // smallest-CID wins, deterministically.
        let bs = MemoryBlockstore::new();
        let root = put_op(&bs, 40, &[]);
        let a = put_op(&bs, 41, &[root.clone()]);
        let b = put_op(&bs, 42, &[root]);
        let m = put_op(&bs, 43, &[a.clone(), b.clone()]);
        let n = put_op(&bs, 44, &[a.clone(), b.clone()]);

        let lca = find_lca(&bs, m.clone(), n.clone()).unwrap().unwrap();
        // Deterministic: must be the min of {a, b}.
        let expected = if a <= b { a } else { b };
        assert_eq!(lca, expected);

        // And commutativity still holds on a criss-cross.
        let reverse = find_lca(&bs, n, m).unwrap().unwrap();
        assert_eq!(reverse, expected);
    }

    #[test]
    fn disjoint_histories_return_none() {
        // Two independent roots with no shared ancestor.
        let bs = MemoryBlockstore::new();
        let r1 = put_op(&bs, 50, &[]);
        let r2 = put_op(&bs, 51, &[]);
        let lca = find_lca(&bs, r1, r2).unwrap();
        assert_eq!(lca, None);
    }

    #[test]
    fn find_lca_many_octopus() {
        // Three-way:
        //       root
        //      / | \
        //     A  B  C        → LCA({A, B, C}) == root
        let bs = MemoryBlockstore::new();
        let root = put_op(&bs, 60, &[]);
        let a = put_op(&bs, 61, &[root.clone()]);
        let b = put_op(&bs, 62, &[root.clone()]);
        let c = put_op(&bs, 63, &[root.clone()]);
        let lca = find_lca_many(&bs, &[a, b, c]).unwrap();
        assert_eq!(lca, Some(root));
    }

    #[test]
    fn find_lca_many_single_head_is_self() {
        let bs = MemoryBlockstore::new();
        let r = put_op(&bs, 70, &[]);
        assert_eq!(find_lca_many(&bs, &[r.clone()]).unwrap(), Some(r));
    }

    #[test]
    fn find_lca_many_empty_is_none() {
        let bs = MemoryBlockstore::new();
        assert_eq!(find_lca_many(&bs, &[]).unwrap(), None);
    }

    #[test]
    fn cache_hit_counter_increments() {
        let bs = MemoryBlockstore::new();
        let root = put_op(&bs, 80, &[]);
        let l = put_op(&bs, 81, &[root.clone()]);
        let r = put_op(&bs, 82, &[root]);

        let mut cache = LcaCache::new();
        assert_eq!(cache.hits(), 0);
        assert_eq!(cache.misses(), 0);
        let _ = cache.get_or_compute(&bs, l.clone(), r.clone()).unwrap();
        assert_eq!(cache.misses(), 1);
        assert_eq!(cache.hits(), 0);
        // Same pair, reversed - must hit the canonical slot.
        let _ = cache.get_or_compute(&bs, r, l).unwrap();
        assert_eq!(cache.misses(), 1);
        assert_eq!(cache.hits(), 1);
    }

    #[test]
    fn canonical_pair_is_order_independent() {
        let bs = MemoryBlockstore::new();
        let a = put_op(&bs, 90, &[]);
        let b = put_op(&bs, 91, &[]);
        assert_eq!(canonical_pair(&a, &b), canonical_pair(&b, &a));
    }
}
