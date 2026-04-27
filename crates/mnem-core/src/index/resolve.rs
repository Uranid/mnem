//! Entity-resolution helper: `lookup_by_prop` for dedup-on-write.
//!
//! Extracted from `index.rs` in R3; body unchanged.

use ipld_core::ipld::Ipld;

use crate::error::Error;
use crate::id::Cid;
use crate::objects::{IndexSet, Node};
use crate::prolly::{self, ProllyKey};
use crate::repo::readonly::decode_from_store;
use crate::store::Blockstore;

use super::build::prop_value_hash;

/// Look up a node via the property index by `(label, prop_name, value)`.
///
/// Returns the node CID and decoded Node if a match exists, or
/// `None`. Defensive re-check: if the index points at a node whose
/// `ntype` / `props[prop_name]` don't actually match the query, returns
/// `None` (hash collision under the 16-byte BLAKE3 truncation).
///
/// Used by `Transaction::resolve_or_create_node` to avoid duplicate
/// entities when agents don't track their own IDs.
///
/// # Errors
///
/// - Store errors from reading the index sub-trees.
/// - Codec errors from decoding the matched Node.
/// - Hashing errors from [`prop_value_hash`].
pub fn lookup_by_prop<B: Blockstore + ?Sized>(
    bs: &B,
    indexes: &IndexSet,
    label: &str,
    prop_name: &str,
    value: &Ipld,
) -> Result<Option<(Cid, Node)>, Error> {
    let Some(tree_root) = indexes
        .nodes_by_prop
        .get(label)
        .and_then(|m| m.get(prop_name))
    else {
        return Ok(None);
    };
    let key = ProllyKey::new(prop_value_hash(value)?);
    let Some(node_cid) = prolly::lookup(bs, tree_root, &key)? else {
        return Ok(None);
    };
    let node: Node = decode_from_store(bs, &node_cid)?;
    // Defensive: confirm the decoded node actually has this (label, prop, value).
    // Hash collisions are astronomically unlikely on blake3 but confirm anyway.
    if node.ntype != label {
        return Ok(None);
    }
    if node.props.get(prop_name) != Some(value) {
        return Ok(None);
    }
    Ok(Some((node_cid, node)))
}
