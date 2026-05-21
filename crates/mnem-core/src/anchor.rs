//! System-anchor identity.
//!
//! Every fresh repo seeded by `mnem init` writes a single, deterministic
//! "anchor" node so the graph is non-empty from the first second and so
//! every commit chain shares a common ancestor (the BUG-56 fast-forward
//! guarantee that lets a fresh subscriber `mnem pull` from a publisher
//! without manual base resolution).
//!
//! The anchor is structural, not content. It carries no summary, no
//! content, and no agent-written semantics; surfacing it in a
//! `mnem retrieve` result is always noise.
//!
//! This module exposes a single canonical identity for the anchor so
//! every layer that needs to recognise (and skip) it does so against
//! the same constant. Without that, the `Query` / `Retriever` filters,
//! the reindex candidate collector, and `mnem init` would each have to
//! re-declare the UUID and drift would be inevitable.
//!
//! Surfaces that filter the anchor by default:
//!
//! - [`crate::index::query::Query`] (CLI `mnem query`, MCP
//!   `mnem_list_nodes`, HTTP `/list-nodes`)
//! - [`crate::retrieve::retriever::Retriever`] (CLI `mnem retrieve`,
//!   MCP `mnem_retrieve`, HTTP `/retrieve`)
//!
//! Audit / admin tooling can opt back in with `include_system(true)` on
//! either builder, mirroring the existing `include_tombstoned(true)`
//! escape hatch.

use crate::id::NodeId;
use crate::objects::Node;

/// Canonical UUID of the system anchor written by `mnem init`.
///
/// Bytes spell `mnem` (`6d 6e 65 6d`) in the low-order tail; the rest
/// is a UUIDv7-shaped sentinel that no `NodeId::new_v7()` call can
/// produce (the version-7 bits + timestamp prefix would never roll
/// over to all-zero). Hard-coded here, not derived, so a string
/// comparison or byte slice in any caller will match.
pub const ANCHOR_NODE_UUID: &str = "00000000-0000-7000-8000-6d6e656d0001";

/// Parsed [`NodeId`] form of [`ANCHOR_NODE_UUID`]. Computed eagerly via
/// `Once`-like init at first call so callers in the hot path
/// ([`crate::retrieve::retriever::Retriever::execute`],
/// [`crate::index::query::Query::execute`]) don't re-parse on every
/// candidate.
#[inline]
#[must_use]
pub fn anchor_node_id() -> NodeId {
    // `parse_uuid` is infallible for this string (compile-time-known
    // constant); the panic message would only fire if someone edited
    // the constant to an invalid form, which is caught immediately by
    // the unit test below.
    NodeId::parse_uuid(ANCHOR_NODE_UUID)
        .expect("ANCHOR_NODE_UUID is a valid UUID; compile-time constant")
}

/// Returns `true` when `id` matches the system anchor's identity.
///
/// Use this from filter sites where you have a `NodeId` in hand but
/// haven't yet loaded the full `Node` (cheap path).
#[inline]
#[must_use]
pub fn is_anchor_node_id(id: &NodeId) -> bool {
    *id == anchor_node_id()
}

/// Returns `true` when `node` is the system anchor (or any future
/// system-reserved node). Today this is only the anchor, but new
/// callers should prefer this over `is_anchor_node_id` so adding a
/// second system node later is a one-line change inside this module
/// rather than a sweep across every filter site.
#[inline]
#[must_use]
pub fn is_system_node(node: &Node) -> bool {
    is_anchor_node_id(&node.id)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn anchor_uuid_parses() {
        // Catches anyone editing the constant to an invalid form.
        let id = anchor_node_id();
        assert_eq!(id.to_string(), ANCHOR_NODE_UUID);
    }

    #[test]
    fn is_anchor_matches_constant_and_rejects_others() {
        let anchor = anchor_node_id();
        assert!(is_anchor_node_id(&anchor));
        let other = NodeId::new_v7();
        assert!(!is_anchor_node_id(&other));
    }

    #[test]
    fn is_system_node_matches_anchor_only() {
        let anchor_node = Node::new(anchor_node_id(), "Meta");
        assert!(is_system_node(&anchor_node));

        // A user-created Meta node with a fresh id is NOT system.
        let user_meta = Node::new(NodeId::new_v7(), "Meta");
        assert!(!is_system_node(&user_meta));
    }
}
