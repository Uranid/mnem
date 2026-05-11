//! Handler for the `mnem_schema` MCP tool.
//!
//! Extracted from `tools.rs` in R3; extended in Item-2 audit to include
//! edge type enumeration from the outgoing adjacency index.

use std::collections::BTreeSet;

use anyhow::Result;
use mnem_core::codec::from_canonical_bytes;
use mnem_core::objects::AdjacencyBucket;
use mnem_core::prolly::tree::{TreeChunk, load_tree_chunk};

use super::super::index_set;
use crate::server::Server;

// ============================================================
// mnem_schema
// ============================================================

pub(in crate::tools) fn schema(server: &mut Server) -> Result<String> {
    let repo = server.load_repo()?;
    let Some(set) = index_set(server, &repo)? else {
        return Ok("schema: <no IndexSet on current commit>\n".to_string());
    };

    let mut out = String::new();
    out.push_str("mnem schema (from current IndexSet)\n");
    out.push_str("  node labels:\n");
    if set.nodes_by_label.is_empty() {
        out.push_str("    <none>\n");
    } else {
        for label in set.nodes_by_label.keys() {
            let props: Vec<&String> = set
                .nodes_by_prop
                .get(label)
                .map(|m| m.keys().collect::<Vec<_>>())
                .unwrap_or_default();
            out.push_str(&format!(
                "    - {label} [indexed props: {}]\n",
                if props.is_empty() {
                    "none".into()
                } else {
                    props
                        .iter()
                        .map(|s| s.as_str())
                        .collect::<Vec<_>>()
                        .join(", ")
                }
            ));
        }
    }

    // ----------------------------------------------------------
    // Edge types: enumerate unique edge type labels from the
    // outgoing adjacency index (best-effort; absent index or any
    // I/O error produces the "index not built" message instead of
    // failing the whole tool call).
    // ----------------------------------------------------------
    out.push_str("  edge types:\n");
    match collect_edge_types(server, &set.outgoing) {
        Ok(Some(etypes)) if !etypes.is_empty() => {
            for etype in &etypes {
                out.push_str(&format!("    - {etype}\n"));
            }
        }
        Ok(Some(_)) => {
            // Index is present (outgoing CID exists in IndexSet) but the tree
            // contains no edge entries — e.g. all edges were deleted after the
            // index was built, or embed --reindex was run on a node-only repo.
            // This is distinct from "index absent": the CID exists, we just
            // found zero edge labels inside it.
            out.push_str("    <none — index present but contains no edges>\n");
        }
        _ => {
            // Index absent (outgoing CID is None in IndexSet) or an I/O /
            // codec error occurred.  Prompt the user to build the index; never
            // hard-fail the tool call.
            out.push_str("    (index not built — run `mnem embed --reindex` to populate)\n");
        }
    }

    out.push_str("  outgoing-adjacency index: ");
    out.push_str(if set.outgoing.is_some() {
        "present\n"
    } else {
        "absent\n"
    });
    out.push_str("  incoming-adjacency index: ");
    out.push_str(if set.incoming.is_some() {
        "present\n"
    } else {
        "absent\n"
    });
    Ok(out)
}

/// Walk the outgoing adjacency Prolly tree (if present) and collect all
/// unique edge type labels.
///
/// Returns `Ok(None)` when the outgoing index is absent.
/// Returns `Ok(Some(set))` on success (set may be empty if the tree
/// exists but has no edges).
/// Returns `Err(_)` only on store / codec errors; callers treat any
/// error as "index not built" so the tool never hard-fails.
fn collect_edge_types(
    server: &mut Server,
    outgoing_cid: &Option<mnem_core::id::Cid>,
) -> Result<Option<BTreeSet<String>>> {
    let bs = server.stores()?.0;
    collect_edge_types_from_bs(bs.as_ref(), outgoing_cid)
}

/// Inner blockstore-level helper: walk the outgoing adjacency Prolly tree
/// rooted at `outgoing_cid` and collect all unique edge type labels.
///
/// Separated from [`collect_edge_types`] so it can be called with any
/// [`mnem_core::store::Blockstore`] — including the in-memory reference
/// implementation used in unit tests — without needing a full [`Server`].
///
/// Returns `Ok(None)` when the outgoing index is absent (`outgoing_cid` is
/// `None`).  Returns `Ok(Some(set))` on success (set may be empty if the
/// tree exists but contains only buckets with zero entries).  Returns
/// `Err(_)` only on store / codec errors.
fn collect_edge_types_from_bs(
    bs: &dyn mnem_core::store::Blockstore,
    outgoing_cid: &Option<mnem_core::id::Cid>,
) -> Result<Option<BTreeSet<String>>> {
    let root_cid = match outgoing_cid {
        Some(c) => c.clone(),
        None => return Ok(None),
    };

    let mut etypes: BTreeSet<String> = BTreeSet::new();

    // Iterative depth-first walk of the Prolly tree.
    let mut stack = vec![root_cid];
    while let Some(cid) = stack.pop() {
        let chunk = load_tree_chunk(bs, &cid)?;
        match chunk {
            TreeChunk::Internal(internal) => {
                // Push children so we eventually reach all leaves.
                stack.extend(internal.children);
            }
            TreeChunk::Leaf(leaf) => {
                // Each leaf entry value is a CID pointing to an
                // AdjacencyBucket. Fetch and decode each bucket.
                for (_key, bucket_cid) in &leaf.entries {
                    let bucket_bytes = bs.get(bucket_cid)?.ok_or_else(|| {
                        anyhow::anyhow!("AdjacencyBucket block {bucket_cid} missing")
                    })?;
                    let bucket: AdjacencyBucket = from_canonical_bytes(&bucket_bytes)?;
                    for entry in &bucket.edges {
                        etypes.insert(entry.label.clone());
                    }
                }
            }
        }
    }

    Ok(Some(etypes))
}

// ============================================================
// Unit tests
// ============================================================

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use mnem_core::codec::hash_to_cid;
    use mnem_core::id::NodeId;
    use mnem_core::objects::AdjacencyBucket;
    use mnem_core::prolly::{ProllyKey, build_tree};
    use mnem_core::store::{Blockstore, MemoryBlockstore};

    use super::collect_edge_types_from_bs;

    /// The `Ok(Some(_))` arm in the schema match fires when `outgoing_cid`
    /// is `Some` (the Prolly tree CID exists in the IndexSet) but the tree
    /// contains only `AdjacencyBucket`s with zero entries.
    ///
    /// `build_index_set` sets `outgoing = None` whenever
    /// `outgoing_groups.is_empty()` (i.e. no edges were committed), so
    /// this state is unreachable through the normal commit path.  It can
    /// occur if, for example, an external tool writes a Prolly tree whose
    /// leaf buckets are empty, or if a future reindex path keeps the CID
    /// pointer while clearing all bucket contents.
    ///
    /// Because the arm cannot be exercised through the MCP dispatch layer
    /// (which would need `build_index_set` to produce `outgoing = Some(cid)`
    /// with zero edges), we test it at the unit level by directly constructing
    /// a Prolly tree that has one leaf entry pointing to an empty
    /// `AdjacencyBucket`, then calling `collect_edge_types_from_bs` with
    /// that CID.
    #[test]
    fn collect_edge_types_from_bs_empty_bucket_returns_some_empty_set() {
        let bs = MemoryBlockstore::default();

        // Build an empty AdjacencyBucket and write it to the blockstore.
        let empty_bucket = AdjacencyBucket {
            edges: vec![],
            extra: BTreeMap::new(),
        };
        let (bucket_bytes, bucket_cid) = hash_to_cid(&empty_bucket).expect("hash bucket");
        bs.put_trusted(bucket_cid.clone(), bucket_bytes)
            .expect("put bucket");

        // Build a minimal outgoing Prolly tree: one entry keyed by an
        // arbitrary NodeId, pointing to the empty bucket CID.
        let src_id = NodeId::from_bytes_raw([0u8; 16]);
        let mut entries: BTreeMap<ProllyKey, _> = BTreeMap::new();
        entries.insert(ProllyKey::from(src_id), bucket_cid);
        let tree_cid = build_tree(&bs, entries).expect("build tree");

        // collect_edge_types_from_bs with the tree CID must return
        // Ok(Some(empty_set)) — the "index present but contains no edges" path.
        let result =
            collect_edge_types_from_bs(&bs, &Some(tree_cid)).expect("collect must not error");
        match result {
            Some(set) => assert!(
                set.is_empty(),
                "expected empty edge-type set for empty-bucket tree, got: {set:?}"
            ),
            None => panic!("expected Some(empty_set) but got None"),
        }
    }

    /// When `outgoing_cid` is `None`, `collect_edge_types_from_bs` must
    /// return `Ok(None)` (the "index absent" path, distinct from the
    /// empty-bucket path above).
    #[test]
    fn collect_edge_types_from_bs_none_cid_returns_ok_none() {
        let bs = MemoryBlockstore::default();
        let result = collect_edge_types_from_bs(&bs, &None).expect("must not error");
        assert!(result.is_none(), "expected None for absent outgoing CID");
    }
}
