//! Index builder: `build_index_set`, `incremental_append_indexes`,
//! `prop_value_hash`, and the `rebuild_subtree_with_additions` helper.
//!
//! Extracted from `index.rs` in R3; bodies unchanged.

use std::collections::BTreeMap;

use ipld_core::ipld::Ipld;

use crate::codec::hash_to_cid;
use crate::error::Error;
use crate::id::{Cid, NodeId};
use crate::objects::{
    AdjacencyBucket, AdjacencyEntry, Edge, IncomingAdjacencyBucket, IncomingAdjacencyEntry,
    IndexSet, Node,
};
use crate::prolly::{self, Cursor, ProllyKey};
use crate::repo::readonly::decode_from_store;
use crate::store::Blockstore;

use super::wrap_index_decode_error;

/// Build an [`IndexSet`] from the complete contents of a node and an
/// edge Prolly tree, write it to the blockstore, and return its CID.
///
/// Naive full rebuild (O(n) over current state). **Fallback path only.**
/// The append-only fast path is [`incremental_append_indexes`], selected
/// automatically by `Transaction::commit_opts` whenever a commit meets
/// the gate conditions (pure node-level append, base `IndexSet`
/// available, no `NodeId` collision). Callers that want the fast path
/// should go through the transaction API, not call this function
/// directly. This function is still invoked by:
/// - First commit on a fresh repo (no base `IndexSet` to extend).
/// - Commits with removed nodes/edges or new edges (gate blocks the
///   fast path until the phase-2 incremental-adjacency work lands).
/// - Multi-head merges in `repo::merge::build_merge_commit` (known gap
///   tracked as a follow-up; fine for single-writer benchmarks, matters
///   once concurrent writers collide).
///
/// # Errors
///
/// - Store errors from reading nodes/edges or writing index blocks.
/// - Codec errors from decoding nodes/edges or encoding sub-objects.
/// - Hashing errors from [`prop_value_hash`].
///
/// Partial failure is safe: any blocks written before a mid-build
/// failure remain in the blockstore but are unreferenced
/// (content-addressed), so garbage-collection is a no-op.
#[tracing::instrument(
    level = "debug",
    target = "mnem::index",
    skip(bs, nodes_root, edges_root)
)]
pub fn build_index_set<B: Blockstore + ?Sized>(
    bs: &B,
    nodes_root: &Cid,
    edges_root: &Cid,
) -> Result<Cid, Error> {
    // Group all node entries by label and by (label, prop_name).
    //
    // Soft size guardrail: naive rebuild holds the entire graph in
    // memory (~50 MB per 100k nodes, scaling linearly). Agents
    // committing > 100k nodes in one commit should switch to the
    // incremental path once it lands . The
    // crate-level invariant (`lib.rs`) forbids `eprintln!`, so callers
    // that want a runtime warning compute the node count themselves
    // from the repo and decide.
    let mut label_groups: BTreeMap<String, BTreeMap<ProllyKey, Cid>> = BTreeMap::new();
    let mut prop_groups: BTreeMap<(String, String), BTreeMap<ProllyKey, Cid>> = BTreeMap::new();

    let node_cursor = Cursor::new(bs, nodes_root)?;
    for entry in node_cursor {
        let (node_key, node_cid) = entry?;
        let node: Node = decode_from_store(bs, &node_cid).map_err(|e| {
            // Wrap decode errors with index-row context so callers see
            // "which index pointed at which bad CID."
            wrap_index_decode_error(
                e,
                format!("IndexSet build: decode node at key {node_key:?}"),
                &node_cid,
            )
        })?;
        label_groups
            .entry(node.ntype.clone())
            .or_default()
            .insert(node_key, node_cid.clone());
        for (prop_name, prop_value) in &node.props {
            let hash_key = prop_value_hash(prop_value)?;
            prop_groups
                .entry((node.ntype.clone(), prop_name.clone()))
                .or_default()
                .insert(ProllyKey::new(hash_key), node_cid.clone());
        }
    }

    // Build per-label Prolly trees.
    let mut nodes_by_label: BTreeMap<String, Cid> = BTreeMap::new();
    for (label, entries) in label_groups {
        let root = prolly::build_tree(bs, entries)?;
        nodes_by_label.insert(label, root);
    }

    // Build per-(label, prop) Prolly trees.
    let mut nodes_by_prop: BTreeMap<String, BTreeMap<String, Cid>> = BTreeMap::new();
    for ((label, prop_name), entries) in prop_groups {
        let root = prolly::build_tree(bs, entries)?;
        nodes_by_prop
            .entry(label)
            .or_default()
            .insert(prop_name, root);
    }

    // Build adjacency buckets per source NodeId (outgoing) and per
    // destination NodeId (incoming) in a single pass. Walking the edge
    // cursor twice would double the O(E) cost and force two full
    // decodes per edge.
    let mut outgoing_groups: BTreeMap<NodeId, Vec<AdjacencyEntry>> = BTreeMap::new();
    let mut incoming_groups: BTreeMap<NodeId, Vec<IncomingAdjacencyEntry>> = BTreeMap::new();
    let edge_cursor = Cursor::new(bs, edges_root)?;
    for entry in edge_cursor {
        let (_k, edge_cid) = entry?;
        let edge: Edge = decode_from_store(bs, &edge_cid)?;
        outgoing_groups
            .entry(edge.src)
            .or_default()
            .push(AdjacencyEntry {
                label: edge.etype.clone(),
                edge: edge_cid.clone(),
            });
        incoming_groups
            .entry(edge.dst)
            .or_default()
            .push(IncomingAdjacencyEntry {
                label: edge.etype,
                src: edge.src,
                edge: edge_cid,
            });
    }

    let outgoing = if outgoing_groups.is_empty() {
        None
    } else {
        let mut bucket_entries: BTreeMap<ProllyKey, Cid> = BTreeMap::new();
        for (src_id, mut edges) in outgoing_groups {
            edges.sort_by(|a, b| a.label.cmp(&b.label).then(a.edge.cmp(&b.edge)));
            let bucket = AdjacencyBucket {
                edges,
                extra: BTreeMap::new(),
            };
            let (bytes, cid) = hash_to_cid(&bucket)?;
            // safety: cid computed above via hash_to_cid
            bs.put_trusted(cid.clone(), bytes)?;
            bucket_entries.insert(ProllyKey::from(src_id), cid);
        }
        Some(prolly::build_tree(bs, bucket_entries)?)
    };

    // Total ordering for byte stability under arbitrary insertion
    // order: `(label, src, edge_cid)`. `edge_cid` alone would suffice
    // (EdgeIds are unique, and the CID is a pure function of the Edge
    // object) but sorting primarily by label groups entries the way
    // callers consume them ("give me all the `knows` back-edges").
    let incoming = if incoming_groups.is_empty() {
        None
    } else {
        let mut bucket_entries: BTreeMap<ProllyKey, Cid> = BTreeMap::new();
        for (dst_id, mut edges) in incoming_groups {
            edges.sort_by(|a, b| {
                a.label
                    .cmp(&b.label)
                    .then(a.src.cmp(&b.src))
                    .then(a.edge.cmp(&b.edge))
            });
            let bucket = IncomingAdjacencyBucket {
                edges,
                extra: BTreeMap::new(),
            };
            let (bytes, cid) = hash_to_cid(&bucket)?;
            // safety: cid computed above via hash_to_cid
            bs.put_trusted(cid.clone(), bytes)?;
            bucket_entries.insert(ProllyKey::from(dst_id), cid);
        }
        Some(prolly::build_tree(bs, bucket_entries)?)
    };

    let set = IndexSet {
        nodes_by_label,
        nodes_by_prop,
        outgoing,
        incoming,
        extra: BTreeMap::new(),
    };
    let (bytes, cid) = hash_to_cid(&set)?;
    // safety: cid computed above via hash_to_cid
    bs.put_trusted(cid.clone(), bytes)?;
    Ok(cid)
}

/// Append-only incremental index update.
///
/// Given a previous `IndexSet` (decoded from `base_indexes_cid`) and a
/// set of newly-added node CIDs, update only the per-label and
/// per-(label, prop) Prolly sub-trees that the new nodes actually
/// touch. Untouched sub-trees are re-referenced by CID without any
/// decode or rebuild work.
///
/// This is the fast path `Transaction::commit_opts` takes when a
/// commit is a pure append (no node/edge removals, no edges changed,
/// and a previous `IndexSet` is available on the base commit).
///
/// # Byte equivalence invariant
///
/// The returned `IndexSet` CID is byte-identical to what
/// [`build_index_set`] would produce if called on the merged
/// node/edge trees, provided:
/// - no `NodeId` in `added_nodes` is already present in the base
///   node tree (i.e. the transaction is append-only at the node
///   level),
/// - no edges are added (the caller falls back to full rebuild if
///   so, since adjacency-incremental-update is not implemented
///   here and is handled in a follow-up).
///
/// Both invariants are enforced by the caller's gating condition
/// in `Transaction::commit_opts`. This function does not re-check.
///
/// # Errors
///
/// - Codec / blockstore errors decoding the base `IndexSet` or the
///   added nodes.
/// - `prop_value_hash` errors on a malformed property value.
#[tracing::instrument(
    level = "debug",
    target = "mnem::index",
    skip(bs, base_indexes_cid, added_nodes),
    fields(added_count = added_nodes.len())
)]
pub fn incremental_append_indexes<B: Blockstore + ?Sized>(
    bs: &B,
    base_indexes_cid: &Cid,
    added_nodes: &BTreeMap<NodeId, Cid>,
) -> Result<Cid, Error> {
    // 1. Decode the base IndexSet. Cost: one decode per commit (tiny
    //    compared to the full rebuild's O(N) cursor walk).
    let base: IndexSet = decode_from_store(bs, base_indexes_cid)?;

    // 2. Group the added nodes' index rows by label / (label, prop).
    //    Same shape as `build_index_set` but walks only the added set,
    //    not the full corpus.
    let mut label_additions: BTreeMap<String, BTreeMap<ProllyKey, Cid>> = BTreeMap::new();
    let mut prop_additions: BTreeMap<(String, String), BTreeMap<ProllyKey, Cid>> = BTreeMap::new();
    for (node_id, node_cid) in added_nodes {
        let node: Node = decode_from_store(bs, node_cid).map_err(|e| {
            wrap_index_decode_error(
                e,
                format!("incremental_append_indexes: decode added node {node_id:?}"),
                node_cid,
            )
        })?;
        let key = ProllyKey::from(*node_id);
        label_additions
            .entry(node.ntype.clone())
            .or_default()
            .insert(key, node_cid.clone());
        for (prop_name, prop_value) in &node.props {
            let hash_key = prop_value_hash(prop_value)?;
            prop_additions
                .entry((node.ntype.clone(), prop_name.clone()))
                .or_default()
                .insert(ProllyKey::new(hash_key), node_cid.clone());
        }
    }

    // 3. Clone the base sub-tree map and update only the touched
    //    sub-trees. Untouched labels keep their previous CID with no
    //    decode, no rebuild, no blockstore work.
    let mut new_nodes_by_label = base.nodes_by_label.clone();
    for (label, additions) in label_additions {
        let base_sub = base.nodes_by_label.get(&label);
        let new_sub = rebuild_subtree_with_additions(bs, base_sub, additions)?;
        new_nodes_by_label.insert(label, new_sub);
    }

    let mut new_nodes_by_prop = base.nodes_by_prop.clone();
    for ((label, prop_name), additions) in prop_additions {
        let base_sub = base
            .nodes_by_prop
            .get(&label)
            .and_then(|m| m.get(&prop_name));
        let new_sub = rebuild_subtree_with_additions(bs, base_sub, additions)?;
        new_nodes_by_prop
            .entry(label)
            .or_default()
            .insert(prop_name, new_sub);
    }

    // 4. Adjacency indexes (both directions): unchanged when no edges
    //    are touched (caller-enforced). Re-reference the previous CIDs
    //    for both outgoing and incoming.
    let new_outgoing = base.outgoing.clone();
    let new_incoming = base.incoming.clone();

    // 5. Assemble, serialize, put. Output CID is byte-equivalent to
    //    what `build_index_set` would produce on the merged state.
    let set = IndexSet {
        nodes_by_label: new_nodes_by_label,
        nodes_by_prop: new_nodes_by_prop,
        outgoing: new_outgoing,
        incoming: new_incoming,
        extra: base.extra,
    };
    let (bytes, cid) = hash_to_cid(&set)?;
    // safety: cid computed above via hash_to_cid
    bs.put_trusted(cid.clone(), bytes)?;
    Ok(cid)
}

/// Rebuild a per-label or per-(label, prop) Prolly sub-tree given
/// `additions`. If `base_sub` is `Some`, walk its entries and merge
/// the additions; if `None`, the label/prop is new and the sub-tree
/// is built from the additions alone.
///
/// Deterministic and content-addressed: the output CID depends only
/// on the merged entry set, so the sub-tree CID matches what
/// `build_index_set` would produce.
fn rebuild_subtree_with_additions<B: Blockstore + ?Sized>(
    bs: &B,
    base_sub: Option<&Cid>,
    additions: BTreeMap<ProllyKey, Cid>,
) -> Result<Cid, Error> {
    match base_sub {
        None => prolly::build_tree(bs, additions),
        Some(root) => {
            let mut merged: BTreeMap<ProllyKey, Cid> = BTreeMap::new();
            let cursor = Cursor::new(bs, root)?;
            for entry in cursor {
                let (k, v) = entry?;
                merged.insert(k, v);
            }
            for (k, v) in additions {
                merged.insert(k, v);
            }
            prolly::build_tree(bs, merged)
        }
    }
}

/// Hash a property value to a 16-byte Prolly key.
///
/// Deterministic by construction: canonical DAG-CBOR fed into BLAKE3,
/// truncated to 16 bytes. Streams the CBOR encoding into the hasher
/// via a small newtype adapter so we never materialise the full
/// encoded `Vec<u8>` per property - important when a commit rebuilds
/// an index over 10k+ (label, prop) tuples.
///
/// The 16-byte truncation is for indexing, not cryptographic integrity
/// (CIDs carry full multihash). `lookup_by_prop` defensively re-checks
/// the retrieved node's `(label, prop, value)` so a hash collision
/// cannot silently return the wrong node.
///
/// # Errors
///
/// Serialization failures bubble up as `Error::Codec`.
pub fn prop_value_hash(value: &Ipld) -> Result<[u8; 16], Error> {
    struct HasherWriter<'a>(&'a mut blake3::Hasher);
    impl std::io::Write for HasherWriter<'_> {
        fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
            self.0.update(buf);
            Ok(buf.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    let mut hasher = blake3::Hasher::new();
    serde_ipld_dagcbor::to_writer(HasherWriter(&mut hasher), value)
        .map_err(|e| crate::error::CodecError::Encode(e.to_string()))?;
    let digest = hasher.finalize();
    let mut out = [0u8; 16];
    out.copy_from_slice(&digest.as_bytes()[..16]);
    Ok(out)
}
