//! M8.5: 3-way view merge for concurrent op-heads.
//!
//! When [`ReadonlyRepo::open`] observes more than one op-head, this module:
//!
//! 1. walks each head's ancestor chain to find their lowest common
//!    ancestor (LCA) on the op-DAG,
//! 2. 3-way merges the `View` objects of all heads against the ancestor's
//!    view, emitting `RefTarget::Conflicted` for refs that diverge,
//! 3. writes a synthetic merge `Operation` whose `parents` are the input
//!    heads and whose `view` is the merged view, and
//! 4. advances the op-heads store so the merge op supersedes all inputs,
//!    collapsing the set back to a single head.
//!
//! The merge is **deterministic** : two processes merging
//! the same set of heads produce the same merge-op CID - same sorted
//! parents, fixed author string, timestamp derived from head times,
//! fixed description format. That keeps convergence idempotent under
//! concurrent readers.
//!
//! [`ReadonlyRepo::open`]: crate::repo::ReadonlyRepo::open

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use ipld_core::ipld::Ipld;

use crate::codec::hash_to_cid;
use crate::error::{Error, RepoError};
use crate::id::{ChangeId, Cid};
use crate::index;
use crate::objects::{Commit, Operation, RefTarget, View};
use crate::prolly::{self, Cursor, ProllyKey};
use crate::store::{Blockstore, OpHeadsStore};

use super::conflict::{
    Conflict, ConflictCategory, ConflictPolicy, MergeConflicts, detect_conflicts_with_views,
};
use super::lca;
use super::readonly::decode_from_store;

/// Fixed author string on synthetic merge operations. Making it a
/// constant (rather than e.g. the current user) is what keeps merges
/// byte-identical across concurrent readers.
const MERGE_AUTHOR: &str = "mnem";

/// Fixed message on merge Commits so every process produces byte-
/// identical merge Commit objects.
const MERGE_COMMIT_MESSAGE: &str = "mnem merge commit";

/// Entry point: given `>=2` op-heads, produce a merge op CID that now
/// supersedes them in the op-heads store.
///
/// When at least one parent head carries a Commit (the common case
/// after data has been written), this also synthesises a **merge
/// Commit** whose node / edge trees are the union of the parents' and
/// whose `IndexSet` is rebuilt from that union. The merged View's
/// `heads` is set to that single merge Commit CID so queries after
/// `ReadonlyRepo::open` see a Commit that describes the full merged
/// graph (not a partial parent).
///
/// Caller is expected to have observed `heads.len() >= 2` from
/// [`OpHeadsStore::current`] before calling.
///
/// # Errors
///
/// Propagates store / codec errors, [`RepoError::NoCommonAncestor`]
/// on a malformed op-DAG, and any error from the index rebuild.
pub(crate) fn merge_op_heads(
    bs: &Arc<dyn Blockstore>,
    ohs: &Arc<dyn OpHeadsStore>,
    mut heads: Vec<Cid>,
) -> Result<Cid, Error> {
    assert!(heads.len() >= 2, "merge_op_heads requires >=2 heads");

    // Determinism: sort heads before use.
    heads.sort();

    // Load all head operations and their views. Also track each head's
    // Commit (if any) so we can build a real merge Commit below.
    let mut head_ops: Vec<Operation> = Vec::with_capacity(heads.len());
    let mut head_views: Vec<View> = Vec::with_capacity(heads.len());
    let mut head_commits: Vec<Option<(Cid, Commit)>> = Vec::with_capacity(heads.len());
    for h in &heads {
        let op: Operation = decode_from_store(&**bs, h)?;
        let view: View = decode_from_store(&**bs, &op.view)?;
        let commit = if let Some(cc) = view.heads.first() {
            let decoded: Commit = decode_from_store(&**bs, cc)?;
            Some((cc.clone(), decoded))
        } else {
            None
        };
        head_ops.push(op);
        head_views.push(view);
        head_commits.push(commit);
    }

    // Find the lowest common ancestor on the op-DAG. `lca::find_lca_many`
    // returns Ok(None) for disjoint histories; merge_op_heads requires a
    // common ancestor, so convert that back to the domain error.
    let ancestor_cid = lca::find_lca_many(&**bs, &heads)?.ok_or(RepoError::NoCommonAncestor)?;
    let ancestor_op: Operation = decode_from_store(&**bs, &ancestor_cid)?;
    let ancestor_view: View = decode_from_store(&**bs, &ancestor_op.view)?;

    // 3-way merge the ref-level view.
    let mut merged_view = merge_views(&ancestor_view, &head_views);

    // Synthesise a merge Commit (and rebuild indexes) if any parent had
    // one. Otherwise the merged view is a pure ref-level merge and we
    // keep its (empty) `heads` vector as produced by merge_views.
    if let Some(merge_commit_cid) = build_merge_commit(&**bs, &head_commits)? {
        merged_view.heads = vec![merge_commit_cid];
    }

    let (view_bytes, view_cid) = hash_to_cid(&merged_view)?;
    // safety: view_cid computed above via hash_to_cid
    bs.put_trusted(view_cid.clone(), view_bytes)?;

    // Synthesize the merge Operation deterministically.
    // Time = max(parent times) + 1 keeps monotonicity without wall-clock drift.
    let merge_time = head_ops.iter().map(|o| o.time).max().unwrap_or(0) + 1;
    let description = describe_merge(&heads);
    let mut merge_op = Operation::new(view_cid, MERGE_AUTHOR, merge_time, description);
    for h in &heads {
        merge_op = merge_op.with_parent(h.clone());
    }
    let (op_bytes, op_cid) = hash_to_cid(&merge_op)?;
    // safety: op_cid computed above via hash_to_cid
    bs.put_trusted(op_cid.clone(), op_bytes)?;

    // Advance op-heads: new merge op supersedes every input head.
    ohs.update(op_cid.clone(), &heads)?;

    Ok(op_cid)
}

/// Build a deterministic merge Commit from the per-head Commit set,
/// write it to the blockstore, and return its CID. Returns `Ok(None)`
/// when no parent had a Commit (pure ref-only merge).
///
/// The merged node and edge Prolly trees are the union of each
/// parent's trees. On key collision (same `NodeId` / `EdgeId` with
/// different content CIDs across parents), the content from the
/// alphabetically-larger parent root wins - arbitrary, but
/// deterministic and byte-stable across concurrent readers. Semantic
/// content conflict handling is Phase 3+ scope.
fn build_merge_commit(
    bs: &dyn Blockstore,
    head_commits: &[Option<(Cid, Commit)>],
) -> Result<Option<Cid>, Error> {
    let parents: Vec<&(Cid, Commit)> = head_commits.iter().filter_map(Option::as_ref).collect();
    if parents.is_empty() {
        return Ok(None);
    }

    // Fast path: all contributing head ops point at the SAME commit CID.
    // This happens when multiple concurrent update_ref ops all branched from
    // the same base view (e.g. fetch's stale-base double-write of tracking
    // refs, or two processes racing to write different refs from the same
    // base). There is no divergence to resolve; return the existing commit CID
    // directly rather than creating a spurious synthetic merge commit.
    // A synthetic merge commit would have a different CID, breaking ancestry
    // checks (BUG-56): the subscriber's real anchor commit would be shadowed
    // by a merge commit that is not part of any remote's history.
    let first_cid = &parents[0].0;
    if parents.iter().all(|(cid, _)| cid == first_cid) {
        return Ok(Some(first_cid.clone()));
    }

    let node_roots: Vec<&Cid> = parents.iter().map(|(_, c)| &c.nodes).collect();
    let edge_roots: Vec<&Cid> = parents.iter().map(|(_, c)| &c.edges).collect();
    let node_union = union_prolly_trees(bs, &node_roots)?;
    let edge_union = union_prolly_trees(bs, &edge_roots)?;
    let merged_nodes = node_union.root;
    let merged_edges = edge_union.root;

    // Deterministic schema: inherit from the alphabetically-lowest
    // parent. Schema mutations are not yet supported, so every parent
    // post-init carries the same schema root.
    let mut parent_pairs: Vec<&(Cid, Commit)> = parents.clone();
    parent_pairs.sort_by(|a, b| a.0.cmp(&b.0));
    let merged_schema = parent_pairs[0].1.schema.clone();

    // Secondary indexes for the merged state. Fast path: if the merged
    // node+edge roots are byte-identical to one of the parents' roots
    // (the concurrent-identical-commits case, where two writers raced
    // but produced the same content), we can reuse that parent's
    // IndexSet CID verbatim - byte-equivalent to a full rebuild by the
    // content-addressing contract. Slow path: full rebuild via
    // `build_index_set` (same behaviour as before Fix X1 for the real
    // 3-way-merge case where node sets diverge; a content-aware
    // structural diff is tracked as a follow-up).
    let mut reused_indexes: Option<Cid> = None;
    for (_parent_cid, parent_commit) in &parents {
        if parent_commit.nodes == merged_nodes
            && parent_commit.edges == merged_edges
            && let Some(idx) = &parent_commit.indexes
        {
            reused_indexes = Some(idx.clone());
            break;
        }
    }
    let merged_indexes = match reused_indexes {
        Some(cid) => cid,
        None => index::build_index_set(bs, &merged_nodes, &merged_edges)?,
    };

    let merge_time = parent_pairs.iter().map(|(_, c)| c.time).max().unwrap_or(0) + 1;

    // Deterministic ChangeId from parent CIDs so byte-identical merges
    // across processes produce byte-identical merge commits.
    let parent_cids: Vec<Cid> = parent_pairs.iter().map(|(c, _)| c.clone()).collect();
    let change_id = deterministic_change_id(&parent_cids);

    let mut commit = Commit::new(
        change_id,
        merged_nodes,
        merged_edges,
        merged_schema,
        MERGE_AUTHOR,
        merge_time,
        MERGE_COMMIT_MESSAGE,
    );
    commit.indexes = Some(merged_indexes);
    // Surface content-level conflicts (same NodeId/EdgeId with
    // different content CIDs across parents) as first-class commit
    // metadata. Agents can inspect `commit.extra["_merge_conflicts"]`
    // to drive a reconcile step; the tiebreak has already produced a
    // consistent merged tree.
    if !node_union.conflicts.is_empty() || !edge_union.conflicts.is_empty() {
        let conflict_list = |map: &BTreeMap<ProllyKey, Vec<Cid>>| -> Ipld {
            let mut entries = Vec::with_capacity(map.len());
            for (k, candidates) in map {
                entries.push(Ipld::Map(
                    [
                        ("key".into(), Ipld::Bytes(k.0.to_vec())),
                        (
                            "candidates".into(),
                            Ipld::List(
                                candidates
                                    .iter()
                                    .map(|c| {
                                        Ipld::Link(
                                            ipld_core::cid::Cid::try_from(c.to_bytes().as_slice())
                                                .expect("cid round-trip"),
                                        )
                                    })
                                    .collect(),
                            ),
                        ),
                    ]
                    .into_iter()
                    .collect::<BTreeMap<_, _>>(),
                ));
            }
            Ipld::List(entries)
        };
        let mut conflict_map = BTreeMap::new();
        if !node_union.conflicts.is_empty() {
            conflict_map.insert("nodes".into(), conflict_list(&node_union.conflicts));
        }
        if !edge_union.conflicts.is_empty() {
            conflict_map.insert("edges".into(), conflict_list(&edge_union.conflicts));
        }
        commit
            .extra
            .insert("_merge_conflicts".into(), Ipld::Map(conflict_map));
    }
    for p in parent_cids {
        commit = commit.with_parent(p);
    }
    let (bytes, cid) = hash_to_cid(&commit)?;
    // safety: cid computed above via hash_to_cid
    bs.put_trusted(cid.clone(), bytes)?;
    Ok(Some(cid))
}

/// Build a merge Commit from pre-computed node and edge Prolly-tree roots.
///
/// This is the strategy-aware counterpart to [`build_merge_commit`]. Whereas
/// [`build_merge_commit`] calls `union_prolly_trees` internally (which always
/// picks the CID-lex-max on conflict), the caller of this function has already
/// resolved conflicts via [`strategy_union_prolly_trees`] and passes in the
/// resulting roots directly. The rest of the bookkeeping (schema, indexes,
/// ChangeId, parents) is identical to `build_merge_commit`.
fn build_merge_commit_from_trees(
    bs: &dyn Blockstore,
    left_cid: Cid,
    right_cid: Cid,
    left_commit: &Commit,
    right_commit: &Commit,
    merged_nodes: Cid,
    merged_edges: Cid,
) -> Result<Cid, Error> {
    // Deterministic schema: inherit from the alphabetically-lowest parent CID.
    let merged_schema = if left_cid <= right_cid {
        left_commit.schema.clone()
    } else {
        right_commit.schema.clone()
    };

    // Secondary indexes.
    let mut reused_indexes: Option<Cid> = None;
    for (parent_commit, _parent_cid) in [(left_commit, &left_cid), (right_commit, &right_cid)] {
        if parent_commit.nodes == merged_nodes
            && parent_commit.edges == merged_edges
            && let Some(idx) = &parent_commit.indexes
        {
            reused_indexes = Some(idx.clone());
            break;
        }
    }
    let merged_indexes = match reused_indexes {
        Some(cid) => cid,
        None => index::build_index_set(bs, &merged_nodes, &merged_edges)?,
    };

    let merge_time = left_commit.time.max(right_commit.time) + 1;

    // Sort parent CIDs for determinism.
    let mut parent_cids = vec![left_cid.clone(), right_cid.clone()];
    parent_cids.sort();
    let change_id = deterministic_change_id(&parent_cids);

    let mut commit = Commit::new(
        change_id,
        merged_nodes,
        merged_edges,
        merged_schema,
        MERGE_AUTHOR,
        merge_time,
        MERGE_COMMIT_MESSAGE,
    );
    commit.indexes = Some(merged_indexes);
    for p in parent_cids {
        commit = commit.with_parent(p);
    }
    let (bytes, cid) = hash_to_cid(&commit)?;
    // safety: cid computed above via hash_to_cid
    bs.put_trusted(cid.clone(), bytes)?;
    Ok(cid)
}

/// Union of N Prolly trees keyed by 16-byte `ProllyKey`. On key
/// collision the entry from the alphabetically-largest root CID wins.
/// Outcome of a multi-tree union: merged Prolly root plus the set of
/// keys where two or more parents disagreed on content. Conflicts are
/// the multi-parent equivalent of `RefTarget::Conflicted` at the node /
/// edge content level; the tiebreak still applies to the merged tree,
/// but the full candidate set travels alongside so agents can inspect
/// and reconcile.
struct UnionOutcome {
    root: Cid,
    /// Per-conflicted-key: the full (sorted, deduped) set of candidate
    /// CIDs across parents. Winning CID is always the last element
    /// (alphabetically-largest), matching the merged tree.
    conflicts: BTreeMap<ProllyKey, Vec<Cid>>,
}

fn union_prolly_trees(bs: &dyn Blockstore, roots: &[&Cid]) -> Result<UnionOutcome, Error> {
    let mut sorted_roots: Vec<&Cid> = roots.to_vec();
    sorted_roots.sort();

    let mut all_values: BTreeMap<ProllyKey, Vec<Cid>> = BTreeMap::new();
    for root in &sorted_roots {
        let cursor = Cursor::new(bs, root)?;
        for entry in cursor {
            let (k, v) = entry?;
            all_values.entry(k).or_default().push(v);
        }
    }

    let mut merged: BTreeMap<ProllyKey, Cid> = BTreeMap::new();
    let mut conflicts: BTreeMap<ProllyKey, Vec<Cid>> = BTreeMap::new();
    for (k, mut values) in all_values {
        values.sort();
        values.dedup();
        if values.len() > 1 {
            let winner = values.last().expect("dedup keeps >=1").clone();
            conflicts.insert(k, values);
            merged.insert(k, winner);
        } else {
            merged.insert(k, values.into_iter().next().expect("len == 1"));
        }
    }

    let root = prolly::build_tree(bs, merged)?;
    Ok(UnionOutcome { root, conflicts })
}

/// Merge two Prolly trees with an explicit strategy for conflict keys.
///
/// For non-conflicting keys the result is the union (same as
/// [`union_prolly_trees`]). For keys where `left_root` and `right_root`
/// carry different CIDs the winner is determined by `strategy`:
///
/// - [`MergeStrategy::Ours`]   → left CID wins  (current-branch side)
/// - [`MergeStrategy::Theirs`] → right CID wins (incoming-branch side)
/// - [`MergeStrategy::Manual`] → falls back to CID-lex-max (same as
///   the unguided union); callers are expected to gate on `Manual`
///   before reaching this function.
fn strategy_union_prolly_trees(
    bs: &dyn Blockstore,
    left_root: &Cid,
    right_root: &Cid,
    strategy: MergeStrategy,
) -> Result<UnionOutcome, Error> {
    // Walk both trees independently (preserving left / right identity).
    let mut left_map: BTreeMap<ProllyKey, Cid> = BTreeMap::new();
    let cursor_l = Cursor::new(bs, left_root)?;
    for entry in cursor_l {
        let (k, v) = entry?;
        left_map.insert(k, v);
    }

    let mut right_map: BTreeMap<ProllyKey, Cid> = BTreeMap::new();
    let cursor_r = Cursor::new(bs, right_root)?;
    for entry in cursor_r {
        let (k, v) = entry?;
        right_map.insert(k, v);
    }

    let all_keys: BTreeSet<ProllyKey> = left_map.keys().chain(right_map.keys()).copied().collect();

    let mut merged: BTreeMap<ProllyKey, Cid> = BTreeMap::new();
    let mut conflicts: BTreeMap<ProllyKey, Vec<Cid>> = BTreeMap::new();

    for k in all_keys {
        match (left_map.get(&k), right_map.get(&k)) {
            (Some(l), Some(r)) if l == r => {
                // No divergence: both sides agree.
                merged.insert(k, l.clone());
            }
            (Some(l), Some(r)) => {
                // Conflict: pick the side dictated by strategy.
                let winner = match strategy {
                    MergeStrategy::Ours => l.clone(),
                    MergeStrategy::Theirs => r.clone(),
                    MergeStrategy::Manual => {
                        // Deterministic fallback: lex-max (same as union_prolly_trees).
                        if l >= r { l.clone() } else { r.clone() }
                    }
                };
                let mut candidates = vec![l.clone(), r.clone()];
                candidates.sort();
                candidates.dedup();
                conflicts.insert(k, candidates);
                merged.insert(k, winner);
            }
            (Some(l), None) => {
                merged.insert(k, l.clone());
            }
            (None, Some(r)) => {
                merged.insert(k, r.clone());
            }
            (None, None) => unreachable!("key came from one of the two maps"),
        }
    }

    let root = prolly::build_tree(bs, merged)?;
    Ok(UnionOutcome { root, conflicts })
}

/// Derive a `ChangeId` deterministically from the sorted parent
/// commit CIDs. Prevents non-determinism that would creep in if we
/// used `ChangeId::new_v7()` (which embeds wall-clock time).
fn deterministic_change_id(parent_cids: &[Cid]) -> ChangeId {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"mnem/merge-change-id/v1");
    for p in parent_cids {
        hasher.update(&p.to_bytes());
    }
    let digest = hasher.finalize();
    let mut id = [0u8; 16];
    id.copy_from_slice(&digest.as_bytes()[..16]);
    ChangeId::from_bytes_raw(id)
}

/// Fixed format so two processes produce byte-identical merge ops.
fn describe_merge(heads: &[Cid]) -> String {
    let joined = heads
        .iter()
        .map(Cid::to_string)
        .collect::<Vec<_>>()
        .join(" ");
    format!("merge {} op-heads: {joined}", heads.len())
}

// Lowest-common-ancestor primitives (BFS parent walk, LCA among
// incomparable candidates, pairwise cache) live in [`super::lca`].

// ---------------- 3-way view merge ----------------

/// Merge `heads` against `ancestor`, field by field.
///
/// - `refs`: per-ref 3-way merge via multiset arithmetic; divergence
///   encodes as [`RefTarget::Conflicted`] (SPEC §4.6).
/// - `heads` (commit pointers, distinct from op heads): union, sorted.
/// - `remote_refs`, `wc_commit`, `extra`: first non-empty head wins;
///   falls back to ancestor. These are informational, not normative -
///   full three-way merging lives in later phases if needed.
fn merge_views(ancestor: &View, heads: &[View]) -> View {
    // Union of ref names across ancestor + all heads.
    let mut names: BTreeSet<String> = ancestor.refs.keys().cloned().collect();
    for v in heads {
        names.extend(v.refs.keys().cloned());
    }

    let mut merged_refs: BTreeMap<String, RefTarget> = BTreeMap::new();
    for name in names {
        let base = ancestor.refs.get(&name);
        let head_targets: Vec<Option<&RefTarget>> =
            heads.iter().map(|v| v.refs.get(&name)).collect();
        if let Some(t) = merge_one_ref(base, &head_targets) {
            merged_refs.insert(name, t);
        }
    }

    // Commit heads: union + sort. Keeps the View self-consistent after
    // merge; refs are the authoritative identity.
    let mut commit_heads: BTreeSet<Cid> = ancestor.heads.iter().cloned().collect();
    for v in heads {
        commit_heads.extend(v.heads.iter().cloned());
    }
    let mut commit_heads: Vec<Cid> = commit_heads.into_iter().collect();
    commit_heads.sort();

    let remote_refs = heads
        .iter()
        .find_map(|v| v.remote_refs.clone())
        .or_else(|| ancestor.remote_refs.clone());

    let wc_commit = heads
        .iter()
        .find_map(|v| v.wc_commit.clone())
        .or_else(|| ancestor.wc_commit.clone());

    let mut extra: BTreeMap<String, Ipld> = ancestor.extra.clone();
    for v in heads {
        for (k, val) in &v.extra {
            extra.insert(k.clone(), val.clone());
        }
    }

    // Tombstones union with last-writer-wins semantics: a later head's
    // Tombstone for a given NodeId replaces the ancestor's / earlier
    // head's. Re-tombstoning is allowed (see `View::tombstones` doc).
    let mut tombstones = ancestor.tombstones.clone();
    for v in heads {
        for (node_id, ts) in &v.tombstones {
            tombstones.insert(*node_id, ts.clone());
        }
    }

    View {
        heads: commit_heads,
        refs: merged_refs,
        remote_refs,
        wc_commit,
        tombstones,
        extra,
    }
}

/// Jujutsu-style 3-way merge of a single ref across N heads.
///
/// Uses signed multiset arithmetic:
///   `result = sum_i head_i - (N - 1) * base`
/// then collapses positive counts to `adds`, negative to `removes`.
/// Zero-count CIDs cancel out. Canonical-form collapse handles the
/// trivial-conflict cases (all-agree, single-change).
fn merge_one_ref(base: Option<&RefTarget>, heads: &[Option<&RefTarget>]) -> Option<RefTarget> {
    // Fast path: every head matches base => unchanged.
    if heads.iter().all(|h| *h == base) {
        return base.cloned();
    }
    // Fast path: every head agrees on the same value (possibly != base).
    if let Some(first) = heads.first()
        && heads.iter().all(|h| h == first)
    {
        return (*first).cloned();
    }

    // Signed-multiset arithmetic.
    let mut counts: BTreeMap<Cid, i32> = BTreeMap::new();
    for h in heads {
        for (cid, sign) in bag(*h) {
            *counts.entry(cid).or_insert(0) += sign;
        }
    }
    let n_minus_1 = i32::try_from(heads.len()).unwrap_or(i32::MAX) - 1;
    for (cid, sign) in bag(base) {
        *counts.entry(cid).or_insert(0) -= sign * n_minus_1;
    }

    let mut adds: Vec<Cid> = Vec::new();
    let mut removes: Vec<Cid> = Vec::new();
    for (cid, count) in counts {
        if count > 0 {
            adds.push(cid);
        } else if count < 0 {
            removes.push(cid);
        }
        // count == 0: canceled out, skip.
    }

    match (adds.len(), removes.len()) {
        (0, 0) => None,
        (1, 0) => {
            let target = adds.into_iter().next().expect("checked len == 1");
            Some(RefTarget::normal(target))
        }
        _ => Some(RefTarget::conflicted(adds, removes)),
    }
}

/// Convert a `RefTarget` option to a signed multiset contribution.
fn bag(t: Option<&RefTarget>) -> Vec<(Cid, i32)> {
    match t {
        None => Vec::new(),
        Some(RefTarget::Normal { target }) => vec![(target.clone(), 1)],
        Some(RefTarget::Conflicted { adds, removes }) => {
            let mut out = Vec::with_capacity(adds.len() + removes.len());
            for c in adds {
                out.push((c.clone(), 1));
            }
            for c in removes {
                out.push((c.clone(), -1));
            }
            out
        }
    }
}

// ---------------- B4.3: explicit branch-level 3-way merge ----------------

/// Strategy for resolving conflicts when running [`merge_three_way`].
///
/// The executor picks this up from the CLI `--strategy` flag and uses
/// it to decide whether a non-empty [`MergeConflicts`] yields a
/// `Conflicts` outcome (manual review required) or is auto-resolved
/// by picking one side across every category.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MergeStrategy {
    /// Manual review: [`merge_three_way`] returns [`MergeOutcome::Conflicts`]
    /// without writing anything. Caller persists
    /// `.mnem/MERGE_CONFLICTS.json` + MERGE_HEAD.
    Manual,
    /// Pick the left / current-branch side for every conflicting entry,
    /// then build a merge commit. Git analog: `git merge -X ours`.
    Ours,
    /// Pick the right / incoming-branch side for every conflicting
    /// entry, then build a merge commit. Git analog: `git merge -X
    /// theirs`.
    Theirs,
}

/// Outcome of a branch-level three-way merge.
///
/// Three-valued by design: fast-forward is a distinct shape from a
/// real merge commit (no merge object is written), and unresolved
/// conflicts are a distinct shape from a clean merge (no blockstore
/// mutation at all).
#[derive(Clone, Debug)]
pub enum MergeOutcome {
    /// `right` is a descendant of `left`. No merge commit was built;
    /// the caller advances HEAD to this CID.
    FastForward(Cid),
    /// Structured 3-way merge produced a clean merge commit. The CID
    /// is byte-identical to the merge-Commit CID that the existing
    /// op-heads convergence path would have produced for the same
    /// `(left, right, lca)` input.
    Clean(Cid),
    /// Conflicts were detected and NOT resolved. Blockstore is
    /// untouched. Caller persists the record and prompts the user.
    Conflicts(MergeConflicts),
}

/// Run a branch-level three-way merge of `left` and `right` Commits.
///
/// Pipeline:
///
/// 1. Find the LCA of `left` and `right` on the Commit DAG. If the
///    LCA is `right`, return [`MergeOutcome::FastForward`] with the
///    left CID; if the LCA is `left`, return [`MergeOutcome::FastForward`]
///    with the right CID.
/// 2. Detect structured conflicts via [`detect_conflicts_with_views`].
/// 3. If non-empty and `policy == Manual`, return
///    [`MergeOutcome::Conflicts`] without mutating the blockstore.
/// 4. If non-empty and `policy == Ours`/`Theirs`, the pick has already
///    been made at the tree level (deterministic union) so we write
///    the merge commit and return [`MergeOutcome::Clean`]. The
///    `MergeConflicts` record can still be inspected by the caller
///    for surfaced provenance.
/// 5. If empty, build the merge commit and return
///    [`MergeOutcome::Clean`].
///
/// Byte-identity: on a clean merge with no conflicts, the emitted
/// commit CID MUST match the CID the existing op-heads convergence
/// path would produce for the same parents. That invariant is exactly
/// why this function delegates to the same deterministic union +
/// build-merge-commit machinery `merge_op_heads` uses internally.
///
/// # Errors
///
/// Propagates store / codec errors from blockstore walks and the
/// conflict-detection pass. Missing LCA (disjoint histories) errors
/// with [`RepoError::NoCommonAncestor`].
pub fn merge_three_way(
    bs: &Arc<dyn Blockstore>,
    _oph: &Arc<dyn OpHeadsStore>,
    left: Cid,
    right: Cid,
    strategy: MergeStrategy,
) -> Result<MergeOutcome, Error> {
    // LCA on the *Commit* DAG (not the op-DAG). `left` and `right`
    // are Commit CIDs, so we walk `Commit::parents`.
    let lca_cid = find_commit_lca(&**bs, &left, &right)?;

    // Fast-forward detection.
    if let Some(ref lca) = lca_cid {
        if lca == &right {
            // right is an ancestor of left: nothing to do.
            return Ok(MergeOutcome::FastForward(left));
        }
        if lca == &left {
            // left is an ancestor of right: FF advance to right.
            return Ok(MergeOutcome::FastForward(right));
        }
    } else {
        return Err(RepoError::NoCommonAncestor.into());
    }

    // Detect conflicts. The detector operates on a `ReadonlyRepo` for
    // View access; we build a throwaway pin at the left op to feed
    // it. Callers that already have a pinned repo pay nothing extra -
    // the detector just walks the Commit trees.
    let left_commit: Commit = decode_from_store(&**bs, &left)?;
    let right_commit: Commit = decode_from_store(&**bs, &right)?;

    // We need Views to surface tombstone-vs-modify. In the branch
    // merge case there is no distinct op per branch tip; fall back to
    // empty Views (no tombstones) which is a safe under-approximation.
    // The core NodeCidDivergence / EdgePropCollision paths don't need
    // Views and fire correctly.
    let empty_view = View {
        heads: vec![],
        refs: BTreeMap::new(),
        remote_refs: None,
        wc_commit: None,
        tombstones: BTreeMap::new(),
        extra: BTreeMap::new(),
    };

    // detect_conflicts_with_views takes a ReadonlyRepo. Build one at
    // the left commit via a synthetic Operation. We avoid persisting
    // that op by keeping it stack-local; the detector only reads.
    let repo = build_detection_repo(bs, &left)?;
    let mc = detect_conflicts_with_views(
        &repo,
        left.clone(),
        right.clone(),
        lca_cid.clone(),
        &empty_view,
        &empty_view,
        ConflictPolicy::default(),
    )?;

    // Manual strategy: conflicts short-circuit without touching the blockstore.
    if !mc.conflicts.is_empty() && matches!(strategy, MergeStrategy::Manual) {
        return Ok(MergeOutcome::Conflicts(mc));
    }

    // Auto-resolve path (Ours / Theirs) or clean-merge path (no conflicts).
    //
    // When conflicts exist and the strategy is Ours or Theirs, we must
    // build the merged Prolly trees using the strategy-aware picker so
    // that the two strategies produce DIFFERENT merged trees (and thus
    // different merge-commit CIDs). The plain `union_prolly_trees` always
    // picks the CID-lex-max winner, making Ours and Theirs byte-identical
    // - that is the bug this branch fixes.
    //
    // For a clean merge (no conflicts) both paths produce identical output,
    // so we always use the strategy-aware function for simplicity.
    let merge_cid = if !matches!(strategy, MergeStrategy::Manual)
        && (!mc.conflicts.is_empty()
            || left_commit.nodes != right_commit.nodes
            || left_commit.edges != right_commit.edges)
    {
        // Build strategy-resolved node and edge trees.
        let node_outcome =
            strategy_union_prolly_trees(&**bs, &left_commit.nodes, &right_commit.nodes, strategy)?;
        let edge_outcome =
            strategy_union_prolly_trees(&**bs, &left_commit.edges, &right_commit.edges, strategy)?;

        build_merge_commit_from_trees(
            &**bs,
            left.clone(),
            right.clone(),
            &left_commit,
            &right_commit,
            node_outcome.root,
            edge_outcome.root,
        )?
    } else {
        // Fallback path: fires when either
        //   (a) strategy == Manual with no conflicts (the Conflicts early-return
        //       above has already handled the non-empty Manual case), or
        //   (b) strategy == Ours/Theirs but both node AND edge tree roots are
        //       byte-identical across left and right (no divergence at all, so
        //       the strategy-aware picker would produce the same result as the
        //       standard union anyway).
        // In both cases the standard build_merge_commit path is correct.
        let head_commits: Vec<Option<(Cid, Commit)>> = vec![
            Some((left.clone(), left_commit)),
            Some((right.clone(), right_commit)),
        ];
        build_merge_commit(&**bs, &head_commits)?
            .ok_or_else(|| Error::from(RepoError::NoCommonAncestor))?
    };

    Ok(MergeOutcome::Clean(merge_cid))
}

/// LCA on the *Commit* DAG using `Commit::parents`. Symmetric
/// interface to [`lca::find_lca`] which walks `Operation::parents`.
fn find_commit_lca(bs: &dyn Blockstore, left: &Cid, right: &Cid) -> Result<Option<Cid>, Error> {
    if left == right {
        return Ok(Some(left.clone()));
    }
    let left_anc = commit_ancestors_inclusive(bs, left)?;
    let right_anc = commit_ancestors_inclusive(bs, right)?;
    let common: BTreeSet<Cid> = left_anc.intersection(&right_anc).cloned().collect();
    if common.is_empty() {
        return Ok(None);
    }
    // Keep only minimal elements (not strict ancestors of another
    // common element).
    let mut strict: BTreeSet<Cid> = BTreeSet::new();
    for c in &common {
        let anc = commit_ancestors_inclusive(bs, c)?;
        for a in &anc {
            if a != c && common.contains(a) {
                strict.insert(a.clone());
            }
        }
    }
    let mut lcas: Vec<Cid> = common.difference(&strict).cloned().collect();
    lcas.sort();
    Ok(lcas.into_iter().next())
}

fn commit_ancestors_inclusive(bs: &dyn Blockstore, cid: &Cid) -> Result<BTreeSet<Cid>, Error> {
    let mut seen: BTreeSet<Cid> = BTreeSet::new();
    let mut stack: Vec<Cid> = vec![cid.clone()];
    while let Some(c) = stack.pop() {
        if !seen.insert(c.clone()) {
            continue;
        }
        let commit: Commit = decode_from_store(bs, &c)?;
        for p in &commit.parents {
            if !seen.contains(p) {
                stack.push(p.clone());
            }
        }
    }
    Ok(seen)
}

/// Build a `ReadonlyRepo` pinned at a commit CID for the structured
/// conflict detector. The detector never writes, so we synthesise a
/// minimal Operation + View in memory (not persisted) just to satisfy
/// the repo facade's accessor contract.
fn build_detection_repo(
    bs: &Arc<dyn Blockstore>,
    commit_cid: &Cid,
) -> Result<super::ReadonlyRepo, Error> {
    use crate::store::MemoryOpHeadsStore;

    // Build a View that points at `commit_cid` as its head commit.
    let view = View {
        heads: vec![commit_cid.clone()],
        refs: BTreeMap::new(),
        remote_refs: None,
        wc_commit: None,
        tombstones: BTreeMap::new(),
        extra: BTreeMap::new(),
    };
    let (view_bytes, view_cid) = hash_to_cid(&view)?;
    bs.put_trusted(view_cid.clone(), view_bytes)?;
    let op = Operation::new(view_cid, MERGE_AUTHOR, 0, "merge-detect");
    let (op_bytes, op_cid) = hash_to_cid(&op)?;
    bs.put_trusted(op_cid.clone(), op_bytes)?;

    // A throwaway op-heads store so the facade has a handle. The
    // detector does not query it.
    let ohs: Arc<dyn OpHeadsStore> = Arc::new(MemoryOpHeadsStore::new());
    super::ReadonlyRepo::load_at(bs.clone(), ohs, op_cid)
}

/// Reference to a single conflict's side-payload. Used by CLI
/// `--continue` path to hydrate a chosen resolution.
///
/// This is intentionally narrow: in B4.3 the CLI only needs to know
/// which side (left / right) to pick per conflict. The rich
/// `Conflict::left/right` JSON payloads are for downstream tooling
/// and UI; actual Prolly-tree application uses the deterministic
/// union path which already encodes left-wins / right-wins via CID
/// lex order tie-break (see the internal `union_prolly_trees` helper).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ConflictSide {
    /// Left / current-branch side.
    Left,
    /// Right / incoming-branch side.
    Right,
}

/// Summarise a MergeConflicts for CLI display.
///
/// Returns `(n_node_cid_divergence, n_edge_prop_collision,
/// n_tombstone_vs_modify)`. Intended for the CLI preview line
/// ("12 conflicts: 7 node-cid, 3 edge-prop, 2 tombstone-vs-modify");
/// programmatic consumers iterate `MergeConflicts::conflicts` directly.
#[must_use]
pub fn conflict_category_counts(mc: &MergeConflicts) -> (usize, usize, usize) {
    let mut node_cid = 0usize;
    let mut edge_prop = 0usize;
    let mut tvm = 0usize;
    for c in &mc.conflicts {
        match c.category {
            ConflictCategory::NodeCidDivergence => node_cid += 1,
            ConflictCategory::EdgePropCollision => edge_prop += 1,
            ConflictCategory::TombstoneVsModify => tvm += 1,
        }
    }
    (node_cid, edge_prop, tvm)
}

/// Infer per-conflict side picks from [`MergeStrategy`].
///
/// `Ours` picks [`ConflictSide::Left`] for every entry; `Theirs` picks
/// [`ConflictSide::Right`]; `Manual` is rejected (caller should have
/// branched out into the manual-review flow before calling this).
///
/// # Errors
///
/// Returns [`RepoError::Stale`] if `Manual` is passed; that's a
/// caller bug (manual strategy is routed through the file-persistence
/// path, not auto-picks).
pub fn picks_from_strategy(
    mc: &MergeConflicts,
    strategy: MergeStrategy,
) -> Result<Vec<(Conflict, ConflictSide)>, Error> {
    let side = match strategy {
        MergeStrategy::Ours => ConflictSide::Left,
        MergeStrategy::Theirs => ConflictSide::Right,
        MergeStrategy::Manual => {
            return Err(RepoError::Stale.into());
        }
    };
    Ok(mc.conflicts.iter().map(|c| (c.clone(), side)).collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::id::{CODEC_RAW, Multihash};

    fn raw_cid(seed: u32) -> Cid {
        Cid::new(CODEC_RAW, Multihash::sha2_256(&seed.to_be_bytes()))
    }

    // ---- merge_one_ref ----

    #[test]
    fn unchanged_from_base_stays_base() {
        let c0 = raw_cid(0);
        let base = RefTarget::normal(c0);
        let heads = [Some(&base), Some(&base)];
        let merged = merge_one_ref(Some(&base), &heads);
        assert_eq!(merged, Some(base));
    }

    #[test]
    fn all_heads_agree_on_changed_value() {
        let c0 = raw_cid(0);
        let c1 = raw_cid(1);
        let base = RefTarget::normal(c0);
        let new = RefTarget::normal(c1);
        let heads = [Some(&new), Some(&new)];
        let merged = merge_one_ref(Some(&base), &heads);
        assert_eq!(merged, Some(new));
    }

    #[test]
    fn single_changing_head_wins_no_conflict() {
        let c0 = raw_cid(0);
        let c1 = raw_cid(1);
        let base = RefTarget::normal(c0);
        let changed = RefTarget::normal(c1);
        let heads = [Some(&base), Some(&changed)];
        let merged = merge_one_ref(Some(&base), &heads);
        assert_eq!(merged, Some(changed));
    }

    #[test]
    fn diverging_heads_produce_conflict() {
        let c0 = raw_cid(0);
        let c1 = raw_cid(1);
        let c2 = raw_cid(2);
        let base = RefTarget::normal(c0.clone());
        let h1 = RefTarget::normal(c1.clone());
        let h2 = RefTarget::normal(c2.clone());
        let merged = merge_one_ref(Some(&base), &[Some(&h1), Some(&h2)]).unwrap();
        match merged {
            RefTarget::Conflicted { adds, removes } => {
                let adds_set: BTreeSet<Cid> = adds.into_iter().collect();
                assert_eq!(adds_set, BTreeSet::from([c1, c2]));
                assert_eq!(removes, vec![c0]);
            }
            other => panic!("expected Conflicted, got {other:?}"),
        }
    }

    #[test]
    fn add_vs_delete_is_a_conflict() {
        let c0 = raw_cid(0);
        let c1 = raw_cid(1);
        let base = RefTarget::normal(c0.clone());
        let bumped = RefTarget::normal(c1.clone());
        // head 1 bumps base -> c1, head 2 deletes base.
        let merged = merge_one_ref(Some(&base), &[Some(&bumped), None]).unwrap();
        match merged {
            RefTarget::Conflicted { adds, removes } => {
                assert_eq!(adds, vec![c1]);
                assert_eq!(removes, vec![c0]);
            }
            other => panic!("expected Conflicted, got {other:?}"),
        }
    }

    #[test]
    fn absent_base_one_head_adds_becomes_normal() {
        let c1 = raw_cid(1);
        let new = RefTarget::normal(c1.clone());
        // Base absent; head_1 adds c1; head_2 left absent.
        let merged = merge_one_ref(None, &[Some(&new), None]).unwrap();
        assert_eq!(merged, RefTarget::normal(c1));
    }

    #[test]
    fn both_heads_delete_base_returns_absent() {
        let c0 = raw_cid(0);
        let base = RefTarget::normal(c0);
        let merged = merge_one_ref(Some(&base), &[None, None]);
        assert_eq!(merged, None);
    }

    // ---- end-to-end: ReadonlyRepo::open through merge ----

    fn stores() -> (Arc<dyn Blockstore>, Arc<dyn OpHeadsStore>) {
        use crate::store::{MemoryBlockstore, MemoryOpHeadsStore};
        (
            Arc::new(MemoryBlockstore::new()),
            Arc::new(MemoryOpHeadsStore::new()),
        )
    }

    fn normal_target(seed: u32) -> RefTarget {
        RefTarget::normal(raw_cid(seed))
    }

    #[test]
    fn open_on_divergent_heads_runs_merge() {
        use crate::repo::ReadonlyRepo;

        let (bs, ohs) = stores();
        let repo0 = ReadonlyRepo::init(bs.clone(), ohs.clone()).unwrap();

        // Two concurrent writers both advance from repo0 independently.
        // update_ref uses `self.op_id` as the supersedes target; both
        // calls see the same base op, so after the second call op-heads
        // holds the two new ops.
        let target_a = normal_target(101);
        let target_b = normal_target(102);
        let repo1 = repo0
            .update_ref("refs/heads/a", None, Some(target_a.clone()), "alice")
            .unwrap();
        let repo2 = repo0
            .update_ref("refs/heads/b", None, Some(target_b.clone()), "bob")
            .unwrap();

        // Sanity: op-heads diverged.
        let heads_before = ohs.current().unwrap();
        assert_eq!(heads_before.len(), 2);
        assert!(heads_before.contains(repo1.op_id()));
        assert!(heads_before.contains(repo2.op_id()));

        // open() runs the merge transparently.
        let merged = ReadonlyRepo::open(bs, ohs.clone()).unwrap();

        // Op-heads collapsed to one.
        let heads_after = ohs.current().unwrap();
        assert_eq!(heads_after.len(), 1);
        assert_eq!(heads_after[0], *merged.op_id());

        // Merge op's parents = sorted({repo1.op_id, repo2.op_id}).
        let mut expected_parents = vec![repo1.op_id().clone(), repo2.op_id().clone()];
        expected_parents.sort();
        assert_eq!(merged.operation().parents, expected_parents);

        // Different refs: both survive without conflict.
        assert_eq!(merged.view().refs.get("refs/heads/a"), Some(&target_a));
        assert_eq!(merged.view().refs.get("refs/heads/b"), Some(&target_b));
    }

    #[test]
    fn same_ref_divergence_becomes_conflicted() {
        use crate::repo::ReadonlyRepo;

        let (bs, ohs) = stores();
        let repo0 = ReadonlyRepo::init(bs.clone(), ohs.clone()).unwrap();

        let target_a = normal_target(201);
        let target_b = normal_target(202);
        repo0
            .update_ref("refs/heads/main", None, Some(target_a.clone()), "alice")
            .unwrap();
        repo0
            .update_ref("refs/heads/main", None, Some(target_b.clone()), "bob")
            .unwrap();
        assert_eq!(ohs.current().unwrap().len(), 2);

        let merged = ReadonlyRepo::open(bs, ohs).unwrap();

        let main_ref = merged
            .view()
            .refs
            .get("refs/heads/main")
            .expect("conflicted main should be present");
        match main_ref {
            RefTarget::Conflicted { adds, removes } => {
                let cid_a = match &target_a {
                    RefTarget::Normal { target } => target.clone(),
                    _ => unreachable!(),
                };
                let cid_b = match &target_b {
                    RefTarget::Normal { target } => target.clone(),
                    _ => unreachable!(),
                };
                let adds_set: BTreeSet<Cid> = adds.iter().cloned().collect();
                assert_eq!(adds_set, BTreeSet::from([cid_a, cid_b]));
                // Base (root view) had no "refs/heads/main", so no removes.
                assert!(removes.is_empty());
            }
            other => panic!("expected Conflicted, got {other:?}"),
        }
    }

    #[test]
    fn open_after_merge_is_idempotent() {
        use crate::repo::ReadonlyRepo;

        let (bs, ohs) = stores();
        let repo0 = ReadonlyRepo::init(bs.clone(), ohs.clone()).unwrap();
        repo0
            .update_ref("refs/heads/a", None, Some(normal_target(301)), "alice")
            .unwrap();
        repo0
            .update_ref("refs/heads/b", None, Some(normal_target(302)), "bob")
            .unwrap();

        let m1 = ReadonlyRepo::open(bs.clone(), ohs.clone()).unwrap();
        // Second open sees a single head now; it must load the same merge op.
        let m2 = ReadonlyRepo::open(bs, ohs).unwrap();
        assert_eq!(m1.op_id(), m2.op_id());
    }

    #[test]
    fn merge_with_concurrent_node_commits_unions_both_sides() {
        // Agent A commits Alice. Agent B (from the same base) commits Bob.
        // open() should trigger a merge that synthesises a merge Commit
        // containing BOTH nodes and a rebuilt IndexSet over both. Queries
        // after open must see both.
        use crate::id::NodeId;
        use crate::index::PropPredicate;
        use crate::objects::Node;
        use crate::repo::ReadonlyRepo;
        use ipld_core::ipld::Ipld;

        let (bs, ohs) = stores();
        let repo0 = ReadonlyRepo::init(bs.clone(), ohs.clone()).unwrap();

        // Agent A from repo0.
        let mut tx_a = repo0.start_transaction();
        let alice =
            Node::new(NodeId::new_v7(), "Person").with_prop("name", Ipld::String("Alice".into()));
        tx_a.add_node(&alice).unwrap();
        let _repo_a = tx_a.commit("agent:A", "alice").unwrap();

        // Agent B ALSO from repo0 (concurrent).
        let mut tx_b = repo0.start_transaction();
        let bob =
            Node::new(NodeId::new_v7(), "Person").with_prop("name", Ipld::String("Bob".into()));
        tx_b.add_node(&bob).unwrap();
        let _repo_b = tx_b.commit("agent:B", "bob").unwrap();

        assert_eq!(
            ohs.current().unwrap().len(),
            2,
            "concurrent writers produced two op-heads"
        );

        // open() should merge them.
        let merged = ReadonlyRepo::open(bs, ohs.clone()).unwrap();
        assert_eq!(ohs.current().unwrap().len(), 1);

        // The merged repo's head_commit must reflect the union of both
        // parent commits - both Alice and Bob must be queryable.
        assert!(
            merged.lookup_node(&alice.id).unwrap().is_some(),
            "Alice survives the merge"
        );
        assert!(
            merged.lookup_node(&bob.id).unwrap().is_some(),
            "Bob survives the merge"
        );

        // The rebuilt IndexSet must also cover both sides.
        let alice_hits = merged
            .query()
            .label("Person")
            .where_prop("name", PropPredicate::Eq(Ipld::String("Alice".into())))
            .execute()
            .unwrap();
        let bob_hits = merged
            .query()
            .label("Person")
            .where_prop("name", PropPredicate::Eq(Ipld::String("Bob".into())))
            .execute()
            .unwrap();
        assert_eq!(alice_hits.len(), 1);
        assert_eq!(bob_hits.len(), 1);

        // The merge Commit should have both parent commit CIDs.
        let merge_commit = merged.head_commit().expect("merge commit exists");
        assert_eq!(
            merge_commit.parents.len(),
            2,
            "merge commit has both parent commits as parents"
        );
    }

    #[test]
    fn concurrent_content_divergence_surfaces_as_merge_conflict_metadata() {
        // Two agents independently overwrite the SAME NodeId with
        // different content. The merge must pick one (deterministic
        // tiebreak) AND record both candidates in commit.extra so
        // agents can reconcile.
        use crate::id::NodeId;
        use crate::objects::Node;
        use crate::repo::ReadonlyRepo;
        use ipld_core::ipld::Ipld;

        let (bs, ohs) = stores();
        let repo0 = ReadonlyRepo::init(bs.clone(), ohs.clone()).unwrap();

        // Pre-populate a node so both agents overwrite the same id.
        let alice_id = NodeId::new_v7();
        let mut tx = repo0.start_transaction();
        tx.add_node(&Node::new(alice_id, "Person").with_prop("name", Ipld::String("Alice".into())))
            .unwrap();
        let repo1 = tx.commit("seed", "seed alice").unwrap();

        // Agent A overwrites to company=Acme (alice's id, different content).
        let mut tx_a = repo1.start_transaction();
        tx_a.add_node(
            &Node::new(alice_id, "Person")
                .with_prop("name", Ipld::String("Alice".into()))
                .with_prop("company", Ipld::String("Acme".into())),
        )
        .unwrap();
        let _ = tx_a.commit("agent:A", "alice at Acme").unwrap();

        // Agent B, from repo1, overwrites to company=Beta.
        let mut tx_b = repo1.start_transaction();
        tx_b.add_node(
            &Node::new(alice_id, "Person")
                .with_prop("name", Ipld::String("Alice".into()))
                .with_prop("company", Ipld::String("Beta".into())),
        )
        .unwrap();
        let _ = tx_b.commit("agent:B", "alice at Beta").unwrap();

        assert_eq!(ohs.current().unwrap().len(), 2);

        let merged = ReadonlyRepo::open(bs, ohs).unwrap();
        let commit = merged.head_commit().expect("merge commit exists");

        // The merged alice is one of the two candidates (deterministic).
        let alice_now = merged.lookup_node(&alice_id).unwrap().unwrap();
        let company = alice_now
            .get_str("company")
            .expect("company prop present on both candidates");
        assert!(company == "Acme" || company == "Beta");

        // The other candidate is NOT silently dropped - it's recorded
        // in commit.extra["_merge_conflicts"]["nodes"].
        let conflicts = commit
            .extra
            .get("_merge_conflicts")
            .expect("merge commit records content conflicts");
        match conflicts {
            Ipld::Map(m) => {
                assert!(m.contains_key("nodes"), "nodes-level conflict recorded");
            }
            other => panic!("expected conflict map, got {other:?}"),
        }
    }

    #[test]
    fn three_parent_octopus_merge_unions_all_sides() {
        // Three concurrent writers, all from the same base. open() must
        // converge to one merge commit whose parents are all three and
        // whose IndexSet covers every node from every branch.
        use crate::id::NodeId;
        use crate::objects::Node;
        use crate::repo::ReadonlyRepo;
        use ipld_core::ipld::Ipld;

        let (bs, ohs) = stores();
        let repo0 = ReadonlyRepo::init(bs.clone(), ohs.clone()).unwrap();

        let mk = |name: &str| -> NodeId {
            let mut tx = repo0.start_transaction();
            let n =
                Node::new(NodeId::new_v7(), "Person").with_prop("name", Ipld::String(name.into()));
            let id = n.id;
            tx.add_node(&n).unwrap();
            let _ = tx.commit("agent", name).unwrap();
            id
        };
        let a = mk("Alice");
        let b = mk("Bob");
        let c = mk("Carol");

        assert_eq!(
            ohs.current().unwrap().len(),
            3,
            "three concurrent op-heads expected"
        );

        let merged = ReadonlyRepo::open(bs, ohs.clone()).unwrap();
        assert_eq!(ohs.current().unwrap().len(), 1);
        assert!(merged.lookup_node(&a).unwrap().is_some());
        assert!(merged.lookup_node(&b).unwrap().is_some());
        assert!(merged.lookup_node(&c).unwrap().is_some());

        // Merge op has all three parent op-ids.
        assert_eq!(merged.operation().parents.len(), 3);
        // Merge Commit has all three parent commit-CIDs.
        assert_eq!(
            merged.head_commit().unwrap().parents.len(),
            3,
            "octopus merge commit points at all three parents"
        );

        // Indexed query after the octopus merge returns every side.
        let all_people = merged.query().label("Person").execute().unwrap();
        assert_eq!(all_people.len(), 3);
    }

    #[test]
    fn merge_op_heads_is_order_invariant() {
        // Determinism : given the SAME head Operations on disk,
        // merging them produces the same merge-op CID regardless of the
        // order in which the input heads are supplied to merge_op_heads.
        // That is the concrete mechanism that keeps concurrent readers
        // from forking into distinct merge chains.
        use crate::repo::ReadonlyRepo;

        let (bs, ohs) = stores();
        let repo0 = ReadonlyRepo::init(bs.clone(), ohs.clone()).unwrap();
        let repo1 = repo0
            .update_ref("refs/heads/a", None, Some(normal_target(501)), "alice")
            .unwrap();
        let repo2 = repo0
            .update_ref("refs/heads/b", None, Some(normal_target(502)), "bob")
            .unwrap();
        assert_eq!(ohs.current().unwrap().len(), 2);

        let merge1 = merge_op_heads(
            &bs,
            &ohs,
            vec![repo1.op_id().clone(), repo2.op_id().clone()],
        )
        .unwrap();

        // Reset op-heads back to the divergent pair.
        ohs.update(repo1.op_id().clone(), std::slice::from_ref(&merge1))
            .unwrap();
        ohs.update(repo2.op_id().clone(), &[]).unwrap();
        assert_eq!(ohs.current().unwrap().len(), 2);

        let merge2 = merge_op_heads(
            &bs,
            &ohs,
            vec![repo2.op_id().clone(), repo1.op_id().clone()],
        )
        .unwrap();

        assert_eq!(
            merge1, merge2,
            "merge op CID must be invariant under input head order"
        );
    }

    // ---- B4.3 merge_three_way ----

    #[test]
    fn merge_three_way_fast_forward_on_linear_history() {
        // right is a descendant of left: FF advance to right.
        use crate::id::NodeId;
        use crate::objects::Node;
        use crate::repo::ReadonlyRepo;
        use ipld_core::ipld::Ipld;

        let (bs, ohs) = stores();
        let repo0 = ReadonlyRepo::init(bs.clone(), ohs.clone()).unwrap();
        let mut tx = repo0.start_transaction();
        tx.add_node(&Node::new(NodeId::new_v7(), "Doc").with_prop("v", Ipld::String("1".into())))
            .unwrap();
        let repo_left = tx.commit("alice", "v1").unwrap();
        let left_cid = repo_left.view().heads.first().cloned().unwrap();

        let mut tx2 = repo_left.start_transaction();
        tx2.add_node(&Node::new(NodeId::new_v7(), "Doc").with_prop("v", Ipld::String("2".into())))
            .unwrap();
        let repo_right = tx2.commit("alice", "v2").unwrap();
        let right_cid = repo_right.view().heads.first().cloned().unwrap();

        let outcome = merge_three_way(
            &bs,
            &ohs,
            left_cid.clone(),
            right_cid.clone(),
            MergeStrategy::Manual,
        )
        .unwrap();
        match outcome {
            MergeOutcome::FastForward(cid) => assert_eq!(cid, right_cid),
            other => panic!("expected FastForward, got {other:?}"),
        }
    }

    #[test]
    fn merge_three_way_clean_produces_union() {
        // Two branches from the same base add distinct nodes. The
        // merge is clean and the resulting commit unions both.
        use crate::id::NodeId;
        use crate::objects::Node;
        use crate::repo::ReadonlyRepo;
        use ipld_core::ipld::Ipld;

        let (bs, ohs) = stores();
        let repo0 = ReadonlyRepo::init(bs.clone(), ohs.clone()).unwrap();
        // Seed a shared base commit.
        let mut tx_base = repo0.start_transaction();
        tx_base
            .add_node(
                &Node::new(NodeId::new_v7(), "Doc").with_prop("v", Ipld::String("base".into())),
            )
            .unwrap();
        let repo_base = tx_base.commit("alice", "base").unwrap();

        // Branch A from base.
        let mut tx_a = repo_base.start_transaction();
        tx_a.add_node(&Node::new(NodeId::new_v7(), "Doc").with_prop("v", Ipld::String("A".into())))
            .unwrap();
        let repo_a = tx_a.commit("alice", "branch A").unwrap();
        let a_cid = repo_a.view().heads.first().cloned().unwrap();

        // Branch B from base.
        let mut tx_b = repo_base.start_transaction();
        tx_b.add_node(&Node::new(NodeId::new_v7(), "Doc").with_prop("v", Ipld::String("B".into())))
            .unwrap();
        let _repo_b = tx_b.commit("alice", "branch B").unwrap();
        // Branch B advances op-heads, so ohs now has both branches;
        // we take the current sole head commit as right.
        let right_cid: Cid = {
            let r = ReadonlyRepo::open(bs.clone(), ohs.clone()).unwrap();
            r.view().heads.first().cloned().unwrap()
        };
        // But that open() already ran the op-merge. Just re-read
        // repo_a's commit as left and right side is the commit we
        // just wrote.
        let outcome = merge_three_way(&bs, &ohs, a_cid, right_cid, MergeStrategy::Manual).unwrap();
        // Could be FF (if histories are linear after op-merge) or Clean.
        match outcome {
            MergeOutcome::Clean(_) | MergeOutcome::FastForward(_) => {}
            MergeOutcome::Conflicts(_) => panic!("disjoint-prop branches should not conflict"),
        }
    }

    #[test]
    fn conflicted_input_flattens_into_final_conflict() {
        let c0 = raw_cid(0);
        let c1 = raw_cid(1);
        let c2 = raw_cid(2);
        let c3 = raw_cid(3);
        let base = RefTarget::normal(c0.clone());
        let prior_conflict = RefTarget::conflicted(vec![c1.clone(), c2.clone()], vec![c0.clone()]);
        let other_change = RefTarget::normal(c3.clone());
        let merged =
            merge_one_ref(Some(&base), &[Some(&prior_conflict), Some(&other_change)]).unwrap();
        match merged {
            RefTarget::Conflicted { adds, removes } => {
                // c0 contributed -1 from prior_conflict, -1 from base subtraction
                //   when N-1 = 1, so one net removal.
                // c1, c2, c3 each +1 => all three in adds.
                let adds_set: BTreeSet<Cid> = adds.into_iter().collect();
                assert_eq!(adds_set, BTreeSet::from([c1, c2, c3]));
                assert_eq!(removes, vec![c0]);
            }
            other => panic!("expected Conflicted, got {other:?}"),
        }
    }

    // ---- strategy_union_prolly_trees ----

    #[test]
    fn strategy_union_prolly_trees_ours_vs_theirs_picks_correct_side() {
        use crate::store::MemoryBlockstore;
        use std::collections::BTreeMap;
        use std::sync::Arc;

        // Build a MemoryBlockstore and seed two minimal Prolly trees that
        // each contain the same key but different value CIDs.
        let bs: Arc<dyn crate::store::Blockstore> = Arc::new(MemoryBlockstore::new());

        // Shared (conflicting) key.
        let conflict_key = ProllyKey::new([0x01u8; 16]);
        // Key that only exists in one side (non-conflicting) to verify
        // union-of-unique-keys still works alongside the conflict resolution.
        let left_only_key = ProllyKey::new([0x10u8; 16]);
        let right_only_key = ProllyKey::new([0x20u8; 16]);

        // Distinct value CIDs for the conflicting key.
        let left_value = raw_cid(1001);
        let right_value = raw_cid(1002);

        // Build LEFT tree: conflict_key -> left_value, left_only_key -> some_cid.
        let mut left_entries: BTreeMap<ProllyKey, Cid> = BTreeMap::new();
        left_entries.insert(conflict_key, left_value.clone());
        left_entries.insert(left_only_key, raw_cid(2001));
        let left_root = prolly::build_tree(&*bs, left_entries).expect("build left tree");

        // Build RIGHT tree: conflict_key -> right_value, right_only_key -> some_cid.
        let mut right_entries: BTreeMap<ProllyKey, Cid> = BTreeMap::new();
        right_entries.insert(conflict_key, right_value.clone());
        right_entries.insert(right_only_key, raw_cid(2002));
        let right_root = prolly::build_tree(&*bs, right_entries).expect("build right tree");

        // strategy = Ours  →  left CID must win for the conflicting key.
        let ours_outcome =
            strategy_union_prolly_trees(&*bs, &left_root, &right_root, MergeStrategy::Ours)
                .expect("strategy_union Ours");
        let ours_root = ours_outcome.root;

        // strategy = Theirs  →  right CID must win for the conflicting key.
        let theirs_outcome =
            strategy_union_prolly_trees(&*bs, &left_root, &right_root, MergeStrategy::Theirs)
                .expect("strategy_union Theirs");
        let theirs_root = theirs_outcome.root;

        // The two strategies produce DIFFERENT merged trees.
        assert_ne!(
            ours_root, theirs_root,
            "Ours and Theirs strategies must produce different Prolly roots"
        );

        // Read back the merged tree contents and verify the conflict key.
        let ours_map: BTreeMap<ProllyKey, Cid> = {
            let cursor = prolly::Cursor::new(&*bs, &ours_root).expect("cursor ours");
            cursor.map(|e| e.expect("entry")).collect()
        };
        let theirs_map: BTreeMap<ProllyKey, Cid> = {
            let cursor = prolly::Cursor::new(&*bs, &theirs_root).expect("cursor theirs");
            cursor.map(|e| e.expect("entry")).collect()
        };

        assert_eq!(
            ours_map.get(&conflict_key),
            Some(&left_value),
            "Ours strategy: conflicting key must map to left (our) CID"
        );
        assert_eq!(
            theirs_map.get(&conflict_key),
            Some(&right_value),
            "Theirs strategy: conflicting key must map to right (their) CID"
        );

        // Non-conflicting keys survive in both merged trees.
        assert!(
            ours_map.contains_key(&left_only_key),
            "left-only key must survive in Ours merge"
        );
        assert!(
            ours_map.contains_key(&right_only_key),
            "right-only key must survive in Ours merge"
        );
        assert!(
            theirs_map.contains_key(&left_only_key),
            "left-only key must survive in Theirs merge"
        );
        assert!(
            theirs_map.contains_key(&right_only_key),
            "right-only key must survive in Theirs merge"
        );

        // Both outcomes report the conflict.
        assert!(
            ours_outcome.conflicts.contains_key(&conflict_key),
            "conflict key must appear in UnionOutcome::conflicts for Ours"
        );
        assert!(
            theirs_outcome.conflicts.contains_key(&conflict_key),
            "conflict key must appear in UnionOutcome::conflicts for Theirs"
        );
    }
}
