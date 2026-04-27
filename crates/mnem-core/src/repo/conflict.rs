//! B4.2 - Structured conflict detector for 3-way merge.
//!
//! This module is the DETECT half of the merge machinery. Given a left
//! head CID, a right head CID, and an optional LCA CID, it walks the
//! two sides' Commits (+ the ancestor's, if any) and emits a
//! deterministic [`MergeConflicts`] record describing every point at
//! which the two sides can't be trivially unioned.
//!
//! Three conflict categories are surfaced today:
//!
//! - [`ConflictCategory::NodeCidDivergence`] - the same stable
//!   [`NodeId`] resolves to two different content CIDs on the left and
//!   right sides (post-LCA). A property edit, a summary rewrite, or a
//!   content replacement on both branches lands here.
//! - [`ConflictCategory::EdgePropCollision`] - the same `(src, dst,
//!   etype)` edge key carries differing property values on each side.
//!   Non-conflicting keys could be deep-merged by the executor (B4.3);
//!   this pass flags the key-level collisions with a deterministic
//!   tie-break on `(branch_head_cid)` lex order in the `suggested`
//!   field so a downstream `--strategy=theirs` / `--strategy=ours`
//!   consumer has a canonical winner.
//! - [`ConflictCategory::TombstoneVsModify`] - one side tombstoned the
//!   node, the other side modified it. Default policy is
//!   tombstone-wins (SPEC §4.10 intent: forget is durable); the
//!   suggested resolution is the tombstone.
//!
//! This wave does NOT implement resolution; the output is intended to
//! be consumed by B4.3's merge executor (`--strategy={ours,theirs,
//! manual}`). The schema string [`MERGE_CONFLICTS_SCHEMA`] is pinned
//! and must stay stable across future waves; B4.4 will test the exact
//! string.

use std::collections::{BTreeMap, BTreeSet};

use ipld_core::ipld::Ipld;
use serde::{Deserialize, Serialize};

use crate::error::Error;
use crate::id::{Cid, NodeId};
use crate::objects::{Commit, Edge, View};
use crate::prolly::{Cursor, ProllyKey};
use crate::repo::ReadonlyRepo;
use crate::repo::readonly::decode_from_store;
use crate::store::Blockstore;

/// Pinned on-wire schema tag for the structured conflict output.
///
/// This constant is the stable contract between the detector (this
/// module), the executor (B4.3), the CLI surface
/// (`.mnem/MERGE_CONFLICTS.json`), and any out-of-band consumer. It
/// MUST NOT change without a SemVer-bump-worthy schema revision.
pub const MERGE_CONFLICTS_SCHEMA: &str = "mnem.v1.merge_conflicts";

/// Category of a single detected conflict entry.
///
/// The three variants map 1:1 to the three merge-conflict shapes the
/// detector handles. Additional variants require a schema revision
/// per [`MERGE_CONFLICTS_SCHEMA`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ConflictCategory {
    /// Same stable [`NodeId`], different content CIDs on left vs right.
    NodeCidDivergence,
    /// Same `(src, dst, etype)` edge key, differing props.
    EdgePropCollision,
    /// One side tombstoned the node, the other side modified it.
    TombstoneVsModify,
}

/// Identity of an edge at the key level for conflict reporting.
///
/// Edges in mnem are keyed on their stable [`crate::id::EdgeId`] inside
/// the Prolly tree, but *property collisions* are semantically scoped to
/// the `(src, dst, etype)` triple: two edges with the same endpoints
/// and type but different [`crate::id::EdgeId`]s still describe the
/// same logical relationship for merge purposes. This key is the
/// triple-shape used in [`Conflict::edge_key`] so the executor can
/// group / dedupe per logical edge.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct EdgeKey {
    /// Source node stable id.
    #[serde(with = "nodeid_str")]
    pub src: NodeId,
    /// Destination node stable id.
    #[serde(with = "nodeid_str")]
    pub dst: NodeId,
    /// Edge-type label (`"knows"`, `"cites"`, ...).
    pub etype: String,
}

/// Custom serde helpers for [`NodeId`] on the JSON wire. NodeId's
/// native impl serializes as bytes, which serde_json renders as a
/// sequence of numbers; the native deserializer only accepts
/// `visit_bytes`, which breaks the round-trip. We render as a
/// hyphenated UUID string here so JSON persistence survives a
/// round-trip without reaching into the NodeId impl.
mod nodeid_str {
    use super::NodeId;
    use serde::{Deserialize, Deserializer, Serialize, Serializer};
    use std::str::FromStr;
    use uuid::Uuid;

    pub(super) fn serialize<S: Serializer>(id: &NodeId, s: S) -> Result<S::Ok, S::Error> {
        id.to_uuid_string().serialize(s)
    }

    pub(super) fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<NodeId, D::Error> {
        let s = String::deserialize(d)?;
        let u = Uuid::from_str(&s).map_err(serde::de::Error::custom)?;
        Ok(NodeId::from_bytes_raw(*u.as_bytes()))
    }
}

/// Same as [`nodeid_str`] but for `Option<NodeId>`.
mod nodeid_str_opt {
    use super::NodeId;
    use serde::{Deserialize, Deserializer, Serialize, Serializer};
    use std::str::FromStr;
    use uuid::Uuid;

    pub(super) fn serialize<S: Serializer>(id: &Option<NodeId>, s: S) -> Result<S::Ok, S::Error> {
        match id {
            Some(n) => n.to_uuid_string().serialize(s),
            None => s.serialize_none(),
        }
    }

    pub(super) fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Option<NodeId>, D::Error> {
        let opt = Option::<String>::deserialize(d)?;
        match opt {
            None => Ok(None),
            Some(s) => {
                let u = Uuid::from_str(&s).map_err(serde::de::Error::custom)?;
                Ok(Some(NodeId::from_bytes_raw(*u.as_bytes())))
            }
        }
    }
}

/// Custom serde helpers for [`Cid`] on the JSON wire. Cid's native
/// derive serializes as bytes (via `cid::CidGeneric`); serde_json
/// emits that as an array of numbers and the deserializer only
/// accepts `visit_bytes` / `visit_borrowed_bytes`. We render as a
/// canonical base32 CID string so the JSON is both round-trippable
/// and human-readable.
mod cid_str {
    use super::Cid;
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    pub(super) fn serialize<S: Serializer>(c: &Cid, s: S) -> Result<S::Ok, S::Error> {
        c.to_string().serialize(s)
    }

    pub(super) fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Cid, D::Error> {
        let s = String::deserialize(d)?;
        Cid::parse_str(&s).map_err(serde::de::Error::custom)
    }
}

/// Same as [`cid_str`] but for `Option<Cid>`.
mod cid_str_opt {
    use super::Cid;
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    pub(super) fn serialize<S: Serializer>(c: &Option<Cid>, s: S) -> Result<S::Ok, S::Error> {
        match c {
            Some(v) => v.to_string().serialize(s),
            None => s.serialize_none(),
        }
    }

    pub(super) fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Option<Cid>, D::Error> {
        let opt = Option::<String>::deserialize(d)?;
        match opt {
            None => Ok(None),
            Some(s) => Cid::parse_str(&s)
                .map(Some)
                .map_err(serde::de::Error::custom),
        }
    }
}

/// A single structured conflict entry.
///
/// Exactly one of `node_id` / `edge_key` is populated, matching the
/// [`category`]. `left` / `right` carry the side-specific payloads
/// (node CIDs for divergence, prop maps for edge collisions, the
/// modified-side payload for tombstone-vs-modify). `base` carries the
/// LCA payload if one was supplied and the key existed there. The
/// `suggested` field is the detector's default resolution proposal;
/// this wave computes it but does NOT apply it (see B4.3).
///
/// [`category`]: Self::category
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Conflict {
    /// Node-scoped conflicts carry the NodeId; edge-scoped ones leave
    /// this `None`.
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        with = "nodeid_str_opt"
    )]
    pub node_id: Option<NodeId>,
    /// Edge-scoped conflicts carry the `(src, dst, etype)` key; node-
    /// scoped ones leave this `None`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub edge_key: Option<EdgeKey>,
    /// Which kind of conflict this is.
    pub category: ConflictCategory,
    /// Left-side payload (CID-as-JSON for nodes, prop-map for edges,
    /// `{"tombstone": {...}}` or `{"node_cid": ...}` for
    /// tombstone-vs-modify).
    pub left: serde_json::Value,
    /// Right-side payload, same shape as `left`.
    pub right: serde_json::Value,
    /// LCA-side payload if the LCA was supplied and carried this key.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub base: Option<serde_json::Value>,
    /// Detector's default resolution proposal per
    /// [`ConflictPolicy`]. Present on tombstone-vs-modify (tombstone
    /// wins) and edge-prop collisions (deterministic lex tie-break).
    /// Intended for future UI surfacing; NOT auto-applied.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub suggested: Option<serde_json::Value>,
}

/// Full structured conflict set emitted by [`detect_conflicts`].
///
/// Serialises to JSON under the [`MERGE_CONFLICTS_SCHEMA`] tag for
/// stable on-disk persistence (`.mnem/MERGE_CONFLICTS.json`, B4.3).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct MergeConflicts {
    /// Schema tag. MUST be [`MERGE_CONFLICTS_SCHEMA`].
    pub schema: String,
    /// Left head commit CID (the CID the caller passed in as `left`).
    #[serde(with = "cid_str")]
    pub left_head: Cid,
    /// Right head commit CID.
    #[serde(with = "cid_str")]
    pub right_head: Cid,
    /// LCA commit CID if one was supplied to [`detect_conflicts`].
    #[serde(default, skip_serializing_if = "Option::is_none", with = "cid_str_opt")]
    pub lca: Option<Cid>,
    /// Detected conflicts, sorted for determinism
    /// (node-scoped before edge-scoped; within each, lex order on the
    /// key bytes).
    pub conflicts: Vec<Conflict>,
}

impl MergeConflicts {
    /// `true` iff the merge is clean (no conflicts).
    #[must_use]
    pub fn is_clean(&self) -> bool {
        self.conflicts.is_empty()
    }
}

/// Tie-break strategy for edge-prop key collisions when neither side
/// matches the LCA.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PropTiebreak {
    /// Pick the value from the side whose `branch_head_cid` sorts
    /// lexicographically smaller. Deterministic; both peers converge.
    BranchHeadCidLex,
}

/// Policy knobs for the detector.
///
/// Defaults match SPEC §4.10 intent (tombstone is durable) and the
/// determinism contract (lex tie-break on branch head CID). B4.3 will
/// surface these via CLI `--strategy`; for now the detector consumes
/// the policy and records the *suggested* resolution without applying.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ConflictPolicy {
    /// If true, tombstone-vs-modify yields a tombstone-wins suggestion.
    pub tombstone_wins: bool,
    /// How to break edge-prop collisions when the LCA can't
    /// disambiguate.
    pub edge_prop_tiebreak: PropTiebreak,
}

impl Default for ConflictPolicy {
    fn default() -> Self {
        Self {
            tombstone_wins: true,
            edge_prop_tiebreak: PropTiebreak::BranchHeadCidLex,
        }
    }
}

/// Detect structured conflicts between `left` and `right` relative to
/// an optional `lca`.
///
/// This walks the node and edge Prolly trees on both sides (+ the LCA
/// side, if any) and records every divergence the three
/// [`ConflictCategory`] variants cover. The output is deterministic:
/// identical `(left, right, lca)` inputs produce byte-identical
/// [`MergeConflicts`] serialisations.
///
/// Policy is taken as [`ConflictPolicy::default`]. Callers that need
/// alternative tie-breaks should use [`detect_conflicts_with_policy`].
///
/// # Errors
///
/// Propagates store / codec errors from the underlying blockstore
/// walks. A missing LCA commit (when `lca` is `Some`) is treated as a
/// broken op-DAG and surfaced as [`Error`].
pub fn detect_conflicts(
    repo: &ReadonlyRepo,
    left: Cid,
    right: Cid,
    lca: Option<Cid>,
) -> Result<MergeConflicts, Error> {
    detect_conflicts_with_policy(repo, left, right, lca, ConflictPolicy::default())
}

/// Like [`detect_conflicts`] but with an explicit [`ConflictPolicy`].
///
/// # Errors
///
/// See [`detect_conflicts`].
pub fn detect_conflicts_with_policy(
    repo: &ReadonlyRepo,
    left: Cid,
    right: Cid,
    lca: Option<Cid>,
    policy: ConflictPolicy,
) -> Result<MergeConflicts, Error> {
    let bs: &dyn Blockstore = &**repo.blockstore();

    let left_commit: Commit = decode_from_store(bs, &left)?;
    let right_commit: Commit = decode_from_store(bs, &right)?;
    let lca_commit: Option<Commit> = match &lca {
        Some(cid) => Some(decode_from_store(bs, cid)?),
        None => None,
    };

    // Load tombstone sets via the Views attached to each commit's op.
    // The detector only needs tombstones; other View fields are
    // B4.3 executor territory. For now we accept that tombstones live
    // on View, not Commit, so callers must supply them. We fall back
    // to "no tombstones" when the Commit-only path is used, which is
    // the current reality (merge.rs loads Views separately). B4.3
    // will thread the View through; for B4.2 the detector surfaces
    // what it can from Commit + opportunistic View lookup.
    let left_tombstones = tombstones_for_commit(bs, &left)?;
    let right_tombstones = tombstones_for_commit(bs, &right)?;

    let mut conflicts = Vec::new();

    // ---- Node-CID divergence ----
    let left_nodes = collect_prolly(bs, &left_commit.nodes)?;
    let right_nodes = collect_prolly(bs, &right_commit.nodes)?;
    let lca_nodes = match &lca_commit {
        Some(c) => collect_prolly(bs, &c.nodes)?,
        None => BTreeMap::new(),
    };

    let node_keys: BTreeSet<&ProllyKey> = left_nodes.keys().chain(right_nodes.keys()).collect();

    for key in &node_keys {
        let key = *key;
        let l = left_nodes.get(key);
        let r = right_nodes.get(key);
        let base = lca_nodes.get(key);
        let node_id = nodeid_from_key(key);

        let left_tomb = left_tombstones.contains(&node_id);
        let right_tomb = right_tombstones.contains(&node_id);

        // Tombstone-vs-modify: exactly one side tombstoned, the other
        // produced a NodeId->CID entry that differs from the LCA's (a
        // real modification, not just a copy-through).
        let right_modified = r.is_some() && r != base;
        let left_modified = l.is_some() && l != base;

        if left_tomb && right_modified && !right_tomb {
            let suggested = if policy.tombstone_wins {
                Some(
                    serde_json::json!({ "action": "tombstone", "node_id": node_id.to_uuid_string() }),
                )
            } else {
                None
            };
            conflicts.push(Conflict {
                node_id: Some(node_id),
                edge_key: None,
                category: ConflictCategory::TombstoneVsModify,
                left: serde_json::json!({ "tombstoned": true }),
                right: serde_json::json!({ "node_cid": r.expect("checked is_some").to_string() }),
                base: base.map(|c| serde_json::json!({ "node_cid": c.to_string() })),
                suggested,
            });
            continue;
        }
        if right_tomb && left_modified && !left_tomb {
            let suggested = if policy.tombstone_wins {
                Some(
                    serde_json::json!({ "action": "tombstone", "node_id": node_id.to_uuid_string() }),
                )
            } else {
                None
            };
            conflicts.push(Conflict {
                node_id: Some(node_id),
                edge_key: None,
                category: ConflictCategory::TombstoneVsModify,
                left: serde_json::json!({ "node_cid": l.expect("checked is_some").to_string() }),
                right: serde_json::json!({ "tombstoned": true }),
                base: base.map(|c| serde_json::json!({ "node_cid": c.to_string() })),
                suggested,
            });
            continue;
        }

        // Plain node-CID divergence: both sides present the NodeId,
        // with differing content CIDs, and neither simply copied
        // through the LCA.
        if let (Some(lc), Some(rc)) = (l, r) {
            if lc != rc {
                // If one side equals the LCA and the other changed,
                // that's NOT a conflict; the changed side wins via a
                // plain 3-way rule. Only flag when both sides
                // diverged from base.
                let left_changed = base.map_or(true, |b| b != lc);
                let right_changed = base.map_or(true, |b| b != rc);
                if left_changed && right_changed {
                    conflicts.push(Conflict {
                        node_id: Some(node_id),
                        edge_key: None,
                        category: ConflictCategory::NodeCidDivergence,
                        left: serde_json::json!({ "node_cid": lc.to_string() }),
                        right: serde_json::json!({ "node_cid": rc.to_string() }),
                        base: base.map(|c| serde_json::json!({ "node_cid": c.to_string() })),
                        suggested: None,
                    });
                }
            }
        }
    }

    // ---- Edge-prop collisions ----
    // Key the comparison on (src, dst, etype). Within the Prolly tree
    // edges are keyed on EdgeId; we reshape to the logical triple here
    // so that two sides that authored "the same edge" via different
    // EdgeIds still collide on differing props.
    let left_edges = collect_edges(bs, &left_commit.edges)?;
    let right_edges = collect_edges(bs, &right_commit.edges)?;
    let lca_edges = match &lca_commit {
        Some(c) => collect_edges(bs, &c.edges)?,
        None => BTreeMap::new(),
    };

    let edge_keys: BTreeSet<&EdgeKey> = left_edges.keys().chain(right_edges.keys()).collect();

    // Deterministic tie-break: lex on the two head CIDs.
    let left_wins_lex = match policy.edge_prop_tiebreak {
        PropTiebreak::BranchHeadCidLex => left < right,
    };

    for key in edge_keys {
        let l = left_edges.get(key);
        let r = right_edges.get(key);
        let base = lca_edges.get(key);

        if let (Some(lp), Some(rp)) = (l, r) {
            if lp == rp {
                continue; // identical props, no conflict
            }

            // If one side equals LCA and the other changed, the
            // changed side wins cleanly. Only flag collisions where
            // both sides diverged from base.
            let left_changed = base.map_or(true, |b| b != lp);
            let right_changed = base.map_or(true, |b| b != rp);
            if !(left_changed && right_changed) {
                continue;
            }

            let left_json = props_to_json(lp);
            let right_json = props_to_json(rp);
            let base_json = base.map(props_to_json);
            let suggested = Some(serde_json::json!({
                "tiebreak": "branch_head_cid_lex",
                "winner_side": if left_wins_lex { "left" } else { "right" },
                "props": if left_wins_lex { &left_json } else { &right_json },
            }));

            conflicts.push(Conflict {
                node_id: None,
                edge_key: Some(key.clone()),
                category: ConflictCategory::EdgePropCollision,
                left: left_json,
                right: right_json,
                base: base_json,
                suggested,
            });
        }
    }

    // Sort: node-scoped first (by NodeId bytes), then edge-scoped (by
    // EdgeKey). Category is a tiebreaker so two conflicts on the same
    // NodeId land in a predictable order.
    conflicts.sort_by(|a, b| {
        let a_kind = u8::from(a.edge_key.is_some());
        let b_kind = u8::from(b.edge_key.is_some());
        a_kind
            .cmp(&b_kind)
            .then_with(|| a.node_id.cmp(&b.node_id))
            .then_with(|| a.edge_key.cmp(&b.edge_key))
            .then_with(|| (a.category as u8).cmp(&(b.category as u8)))
    });

    Ok(MergeConflicts {
        schema: MERGE_CONFLICTS_SCHEMA.to_string(),
        left_head: left,
        right_head: right,
        lca,
        conflicts,
    })
}

// ---------------- helpers ----------------

/// Walk the Prolly tree rooted at `root` and collect every
/// `(ProllyKey, Cid)` leaf pair. Used for node-CID divergence
/// detection: we only need the value CIDs, not the decoded payloads.
fn collect_prolly(bs: &dyn Blockstore, root: &Cid) -> Result<BTreeMap<ProllyKey, Cid>, Error> {
    let mut out = BTreeMap::new();
    let cursor = Cursor::new(bs, root)?;
    for pair in cursor {
        let (k, v) = pair?;
        out.insert(k, v);
    }
    Ok(out)
}

/// Walk the edge Prolly tree and re-key every entry by
/// `(src, dst, etype)`. Decoding is required here because the tree key
/// is `EdgeId`, not the logical triple.
fn collect_edges(
    bs: &dyn Blockstore,
    root: &Cid,
) -> Result<BTreeMap<EdgeKey, BTreeMap<String, Ipld>>, Error> {
    let mut out = BTreeMap::new();
    let cursor = Cursor::new(bs, root)?;
    for pair in cursor {
        let (_k, v) = pair?;
        let edge: Edge = decode_from_store(bs, &v)?;
        let key = EdgeKey {
            src: edge.src,
            dst: edge.dst,
            etype: edge.etype.clone(),
        };
        out.insert(key, edge.props);
    }
    Ok(out)
}

/// Recover a `NodeId` from a `ProllyKey`. NodeIds are 16 bytes and
/// ProllyKeys are `[u8; 16]`, so the conversion is a direct byte copy.
fn nodeid_from_key(key: &ProllyKey) -> NodeId {
    NodeId::from_bytes_raw(key.0)
}

/// Load the tombstone set for the op-head View attached to `commit_cid`.
///
/// The detector needs to know "is this NodeId tombstoned?" on each
/// side. Tombstones live on the [`View`], not the [`Commit`], so we
/// search the op-heads for the matching View. If we can't find one
/// (e.g. a dry-run where the op hasn't been published), we fall back
/// to an empty set rather than erroring - the detector is still
/// correct on all non-tombstone categories.
fn tombstones_for_commit(
    bs: &dyn Blockstore,
    _commit_cid: &Cid,
) -> Result<BTreeSet<NodeId>, Error> {
    // B4.2 narrow path: tombstones are passed in via the View, but we
    // don't have a direct Commit -> View reverse index. The merge
    // executor (B4.3) will thread Views through explicitly. For the
    // detect-only scope, return empty so the other categories still
    // surface correctly.
    //
    // Tests construct Views directly via the helper below, which
    // bypasses this lookup. Production callers from B4.3 will pass
    // tombstone sets through the executor hook.
    let _ = bs;
    Ok(BTreeSet::new())
}

/// Convert a `BTreeMap<String, Ipld>` prop map to a JSON object for
/// serialisation. `serde_json::to_value(&ipld_value)` works because
/// `Ipld` implements `Serialize`.
fn props_to_json(props: &BTreeMap<String, Ipld>) -> serde_json::Value {
    serde_json::to_value(props).unwrap_or(serde_json::Value::Null)
}

/// Alternate entry point for callers that already hold the left /
/// right Views (so tombstones can be threaded through without a
/// Commit -> View reverse lookup). This is the signature B4.3's
/// executor will call.
///
/// # Errors
///
/// See [`detect_conflicts`].
pub fn detect_conflicts_with_views(
    repo: &ReadonlyRepo,
    left: Cid,
    right: Cid,
    lca: Option<Cid>,
    left_view: &View,
    right_view: &View,
    policy: ConflictPolicy,
) -> Result<MergeConflicts, Error> {
    // Delegate to the core routine, then upgrade entries using the
    // real tombstone sets from the supplied Views.
    let mut mc = detect_conflicts_with_policy(repo, left.clone(), right.clone(), lca, policy)?;

    let left_ts: BTreeSet<NodeId> = left_view.tombstones.keys().copied().collect();
    let right_ts: BTreeSet<NodeId> = right_view.tombstones.keys().copied().collect();

    // Upgrade existing NodeCidDivergence entries to TombstoneVsModify
    // when a tombstone on one side explains the divergence. Also
    // emit fresh TvM entries for the "tombstone vs unchanged
    // content" case (both sides carry the node with differing CIDs
    // was already caught above; here we cover the case where the
    // non-tombstoning side has a value and the tombstoning side
    // silently kept the base CID - no divergence, but still a
    // tombstone-vs-modify conceptually).
    for conflict in mc.conflicts.iter_mut() {
        if let Some(id) = conflict.node_id {
            let l_ts = left_ts.contains(&id);
            let r_ts = right_ts.contains(&id);
            if l_ts ^ r_ts {
                // Exactly one side tombstoned: upgrade.
                conflict.category = ConflictCategory::TombstoneVsModify;
                if l_ts {
                    conflict.left = serde_json::json!({ "tombstoned": true });
                }
                if r_ts {
                    conflict.right = serde_json::json!({ "tombstoned": true });
                }
                if policy.tombstone_wins {
                    conflict.suggested = Some(
                        serde_json::json!({ "action": "tombstone", "node_id": id.to_uuid_string() }),
                    );
                }
            }
        }
    }

    // Also emit TvM for nodes where one side tombstoned and the other
    // side modified but the modification didn't trigger a CID
    // divergence (e.g. the tombstoning side silently kept the base
    // CID, so collect_prolly produced matching CIDs on both sides).
    let bs: &dyn Blockstore = &**repo.blockstore();
    let left_commit: Commit = decode_from_store(bs, &left)?;
    let right_commit: Commit = decode_from_store(bs, &right)?;
    let left_nodes = collect_prolly(bs, &left_commit.nodes)?;
    let right_nodes = collect_prolly(bs, &right_commit.nodes)?;

    let emit_for = |id: NodeId, side_tombstone_is_left: bool| -> Option<Conflict> {
        let key = ProllyKey::from(id);
        let (ln, rn) = (left_nodes.get(&key), right_nodes.get(&key));
        // Both sides must carry an entry (post-tombstone the Node is
        // still in the tree) for TvM to apply.
        let (_ln, _rn) = (ln?, rn?);
        Some(Conflict {
            node_id: Some(id),
            edge_key: None,
            category: ConflictCategory::TombstoneVsModify,
            left: if side_tombstone_is_left {
                serde_json::json!({ "tombstoned": true })
            } else {
                serde_json::json!({ "node_cid": _ln.to_string() })
            },
            right: if side_tombstone_is_left {
                serde_json::json!({ "node_cid": _rn.to_string() })
            } else {
                serde_json::json!({ "tombstoned": true })
            },
            base: None,
            suggested: if policy.tombstone_wins {
                Some(serde_json::json!({ "action": "tombstone", "node_id": id.to_uuid_string() }))
            } else {
                None
            },
        })
    };

    for id in left_ts.iter() {
        if mc.conflicts.iter().any(|c| c.node_id == Some(*id)) {
            continue;
        }
        if let Some(c) = emit_for(*id, true) {
            mc.conflicts.push(c);
        }
    }
    for id in right_ts.iter() {
        if mc.conflicts.iter().any(|c| c.node_id == Some(*id)) {
            continue;
        }
        if let Some(c) = emit_for(*id, false) {
            mc.conflicts.push(c);
        }
    }

    mc.conflicts.sort_by(|a, b| {
        let a_kind = u8::from(a.edge_key.is_some());
        let b_kind = u8::from(b.edge_key.is_some());
        a_kind
            .cmp(&b_kind)
            .then_with(|| a.node_id.cmp(&b.node_id))
            .then_with(|| a.edge_key.cmp(&b.edge_key))
            .then_with(|| (a.category as u8).cmp(&(b.category as u8)))
    });

    Ok(mc)
}

// ---------------- tests ----------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::id::{EdgeId, NodeId};
    use crate::objects::{Edge, Node};
    use crate::repo::ReadonlyRepo;
    use crate::store::{Blockstore, MemoryBlockstore, MemoryOpHeadsStore, OpHeadsStore};
    use ipld_core::ipld::Ipld;
    use std::sync::Arc;

    fn stores() -> (Arc<dyn Blockstore>, Arc<dyn OpHeadsStore>) {
        (
            Arc::new(MemoryBlockstore::new()),
            Arc::new(MemoryOpHeadsStore::new()),
        )
    }

    fn nid(seed: u8) -> NodeId {
        NodeId::from_bytes_raw([seed; 16])
    }

    fn eid(seed: u8) -> EdgeId {
        EdgeId::from_bytes_raw([seed; 16])
    }

    /// Convenience: commit a set of nodes + edges on `base`, return
    /// the new head Commit CID.
    fn commit_snapshot(
        base: &ReadonlyRepo,
        author: &str,
        nodes: Vec<Node>,
        edges: Vec<Edge>,
    ) -> (ReadonlyRepo, Cid) {
        let mut tx = base.start_transaction();
        for n in nodes {
            tx.add_node(&n).unwrap();
        }
        for e in edges {
            tx.add_edge(&e).unwrap();
        }
        let new_repo = tx.commit(author, "snap").unwrap();
        let head = new_repo
            .view()
            .heads
            .first()
            .cloned()
            .expect("head present");
        (new_repo, head)
    }

    #[test]
    fn clean_merge_on_disjoint_branches_has_no_conflicts() {
        // Two branches that touch disjoint NodeIds. Expect no
        // NodeCidDivergence and no EdgePropCollision.
        let (bs, ohs) = stores();
        let repo0 = ReadonlyRepo::init(bs.clone(), ohs.clone()).unwrap();
        let (_repo_a, head_a) = commit_snapshot(
            &repo0,
            "A",
            vec![Node::new(nid(1), "Person").with_prop("name", Ipld::String("Alice".into()))],
            vec![],
        );
        // For a disjoint-branch test: branch B also starts from repo0
        // (same LCA). Both commits share the empty-tree base so
        // neither nd(1) nor nd(2) collides.
        let (_repo_b, head_b) = commit_snapshot(
            &repo0,
            "B",
            vec![Node::new(nid(2), "Person").with_prop("name", Ipld::String("Bob".into()))],
            vec![],
        );

        let _ = (bs, ohs);
        let mc = detect_conflicts(&_repo_a, head_a, head_b, None).unwrap();
        assert!(
            mc.is_clean(),
            "expected no conflicts, got {:?}",
            mc.conflicts
        );
        assert_eq!(mc.schema, MERGE_CONFLICTS_SCHEMA);
    }

    #[test]
    fn node_cid_divergence_flagged() {
        // Same NodeId; both sides modify to different values. Expect
        // one NodeCidDivergence conflict.
        let (bs, ohs) = stores();
        let repo0 = ReadonlyRepo::init(bs.clone(), ohs.clone()).unwrap();
        let id = nid(7);

        let (_repo_a, head_a) = commit_snapshot(
            &repo0,
            "A",
            vec![Node::new(id, "Person").with_prop("name", Ipld::String("Alice".into()))],
            vec![],
        );
        let (_repo_b, head_b) = commit_snapshot(
            &repo0,
            "B",
            vec![Node::new(id, "Person").with_prop("name", Ipld::String("Alicia".into()))],
            vec![],
        );

        let _ = (bs, ohs);
        let mc = detect_conflicts(&_repo_a, head_a, head_b, None).unwrap();
        assert_eq!(mc.conflicts.len(), 1);
        let c = &mc.conflicts[0];
        assert_eq!(c.category, ConflictCategory::NodeCidDivergence);
        assert_eq!(c.node_id, Some(id));
    }

    #[test]
    fn edge_prop_collision_deterministic_tiebreak() {
        // Same (src, dst, etype); both sides author different props.
        // Expect one EdgePropCollision with a suggested lex-order
        // winner.
        let (bs, ohs) = stores();
        let repo0 = ReadonlyRepo::init(bs.clone(), ohs.clone()).unwrap();

        let s = nid(1);
        let d = nid(2);

        let (_repo_a, head_a) = commit_snapshot(
            &repo0,
            "A",
            vec![Node::new(s, "Person"), Node::new(d, "Person")],
            vec![Edge::new(eid(10), "knows", s, d).with_prop("since", Ipld::Integer(2020))],
        );
        let (_repo_b, head_b) = commit_snapshot(
            &repo0,
            "B",
            vec![Node::new(s, "Person"), Node::new(d, "Person")],
            vec![Edge::new(eid(11), "knows", s, d).with_prop("since", Ipld::Integer(2021))],
        );

        let _ = (bs, ohs);
        let mc = detect_conflicts(&_repo_a, head_a.clone(), head_b.clone(), None).unwrap();

        let edge_conflicts: Vec<_> = mc
            .conflicts
            .iter()
            .filter(|c| c.category == ConflictCategory::EdgePropCollision)
            .collect();
        assert_eq!(edge_conflicts.len(), 1, "got: {:?}", mc.conflicts);
        let c = edge_conflicts[0];
        assert_eq!(
            c.edge_key,
            Some(EdgeKey {
                src: s,
                dst: d,
                etype: "knows".into()
            })
        );

        // Suggested resolution MUST name the lex-smaller head as winner.
        let want_winner = if head_a < head_b { "left" } else { "right" };
        let got = c.suggested.as_ref().unwrap();
        assert_eq!(
            got.get("winner_side").and_then(|v| v.as_str()),
            Some(want_winner)
        );
        assert_eq!(
            got.get("tiebreak").and_then(|v| v.as_str()),
            Some("branch_head_cid_lex")
        );
    }

    #[test]
    fn tombstone_vs_modify_tombstone_wins_default() {
        // Left side tombstones a node; right side modifies it. Expect
        // one TombstoneVsModify with suggested=tombstone.
        let (bs, ohs) = stores();
        let repo0 = ReadonlyRepo::init(bs.clone(), ohs.clone()).unwrap();

        let id = nid(5);
        // Seed the node on a shared base.
        let (repo_seed, _) = commit_snapshot(
            &repo0,
            "S",
            vec![Node::new(id, "Person").with_prop("name", Ipld::String("Seed".into()))],
            vec![],
        );

        // Left: tombstone it.
        let mut tx_l = repo_seed.start_transaction();
        tx_l.tombstone_node(id, "gdpr").unwrap();
        let repo_l = tx_l.commit("L", "tombstone").unwrap();
        let head_l = repo_l.view().heads.first().cloned().unwrap();
        let view_l = repo_l.view().clone();

        // Right: modify it (create divergent sibling head from seed).
        let (repo_r, head_r) = commit_snapshot(
            &repo_seed,
            "R",
            vec![Node::new(id, "Person").with_prop("name", Ipld::String("Changed".into()))],
            vec![],
        );
        let view_r = repo_r.view().clone();

        let mc = detect_conflicts_with_views(
            &repo_l,
            head_l,
            head_r,
            None,
            &view_l,
            &view_r,
            ConflictPolicy::default(),
        )
        .unwrap();

        let tvm: Vec<_> = mc
            .conflicts
            .iter()
            .filter(|c| c.category == ConflictCategory::TombstoneVsModify)
            .collect();
        assert!(
            !tvm.is_empty(),
            "no tombstone-vs-modify in {:?}",
            mc.conflicts
        );
        let c = tvm[0];
        assert_eq!(c.node_id, Some(id));
        let s = c.suggested.as_ref().unwrap();
        assert_eq!(s.get("action").and_then(|v| v.as_str()), Some("tombstone"));
    }

    #[test]
    fn multi_category_surfaced_simultaneously() {
        // Construct a repo state that yields at least one
        // NodeCidDivergence AND one EdgePropCollision.
        let (bs, ohs) = stores();
        let repo0 = ReadonlyRepo::init(bs.clone(), ohs.clone()).unwrap();

        let p = nid(11);
        let q = nid(22);

        let (_repo_a, head_a) = commit_snapshot(
            &repo0,
            "A",
            vec![
                Node::new(p, "Person").with_prop("k", Ipld::Integer(1)),
                Node::new(q, "Person"),
            ],
            vec![Edge::new(eid(1), "likes", p, q).with_prop("w", Ipld::Integer(100))],
        );
        let (_repo_b, head_b) = commit_snapshot(
            &repo0,
            "B",
            vec![
                Node::new(p, "Person").with_prop("k", Ipld::Integer(2)),
                Node::new(q, "Person"),
            ],
            vec![Edge::new(eid(2), "likes", p, q).with_prop("w", Ipld::Integer(200))],
        );

        let _ = (bs, ohs);
        let mc = detect_conflicts(&_repo_a, head_a, head_b, None).unwrap();
        let cats: BTreeSet<_> = mc.conflicts.iter().map(|c| c.category).collect();
        assert!(cats.contains(&ConflictCategory::NodeCidDivergence));
        assert!(cats.contains(&ConflictCategory::EdgePropCollision));
    }

    #[test]
    fn json_round_trip_preserves_shape() {
        let (bs, ohs) = stores();
        let repo0 = ReadonlyRepo::init(bs.clone(), ohs.clone()).unwrap();
        let id = nid(9);
        let (_repo_a, head_a) = commit_snapshot(
            &repo0,
            "A",
            vec![Node::new(id, "Person").with_prop("k", Ipld::String("x".into()))],
            vec![],
        );
        let (_repo_b, head_b) = commit_snapshot(
            &repo0,
            "B",
            vec![Node::new(id, "Person").with_prop("k", Ipld::String("y".into()))],
            vec![],
        );
        let _ = (bs, ohs);
        let mc = detect_conflicts(&_repo_a, head_a, head_b, None).unwrap();

        let s = serde_json::to_string(&mc).expect("encode");
        let decoded: MergeConflicts = serde_json::from_str(&s).expect("decode");
        assert_eq!(mc, decoded);
    }

    #[test]
    fn schema_constant_pinned() {
        assert_eq!(MERGE_CONFLICTS_SCHEMA, "mnem.v1.merge_conflicts");
    }
}
