//! Forward + incoming adjacency index loaders.
//!
//! Extracted from `index.rs` in R3; bodies unchanged.

use std::collections::HashSet;

use crate::error::Error;
use crate::id::NodeId;
use crate::objects::{AdjacencyBucket, Edge, IncomingAdjacencyBucket, IndexSet};
use crate::prolly::{self, ProllyKey};
use crate::repo::readonly::decode_from_store;
use crate::store::Blockstore;

/// Resolve the outgoing-edge request for one node using the outgoing
/// adjacency index if present. Free function so it's reachable from
/// every branch of `execute` without needing mutable self.
///
/// Returns `(edges, truncated)` where `truncated` is `true` iff the
/// fan-out exceeded `cap` and the return value is a prefix of the
/// full bucket. Protects agents from a "celebrity out-edge"
/// denial-of-service.
pub(super) fn load_outgoing(
    bs: &dyn Blockstore,
    indexes: Option<&IndexSet>,
    src: NodeId,
    want: &HashSet<&str>,
    cap: usize,
) -> Result<(Vec<Edge>, bool), Error> {
    if want.is_empty() {
        return Ok((Vec::new(), false));
    }
    let Some(idx) = indexes else {
        return Ok((Vec::new(), false));
    };
    let Some(adj_root) = &idx.outgoing else {
        return Ok((Vec::new(), false));
    };
    let Some(bucket_cid) = prolly::lookup(bs, adj_root, &ProllyKey::from(src))? else {
        return Ok((Vec::new(), false));
    };
    let bucket: AdjacencyBucket = decode_from_store(bs, &bucket_cid)?;
    let mut out = Vec::new();
    let mut truncated = false;
    for ae in &bucket.edges {
        if want.contains(ae.label.as_str()) {
            if out.len() >= cap {
                truncated = true;
                break;
            }
            let edge: Edge = decode_from_store(bs, &ae.edge)?;
            out.push(edge);
        }
    }
    Ok((out, truncated))
}

/// Resolve the incoming-edge request for one node using the incoming
/// adjacency index if present. Symmetric mirror of [`load_outgoing`]
/// keyed by destination `NodeId` instead of source.
///
/// Returns `(edges, truncated)`. `truncated` flags that the fan-in
/// exceeded `cap` - the celebrity-node case ("1M people follow X").
/// When the indexes are from a pre-0.3 `IndexSet` (no `incoming` root),
/// this returns an empty vec rather than falling back to a full edge
/// scan; graceful degradation by design so older repos don't hang.
pub(super) fn load_incoming(
    bs: &dyn Blockstore,
    indexes: Option<&IndexSet>,
    dst: NodeId,
    want: &HashSet<&str>,
    cap: usize,
) -> Result<(Vec<Edge>, bool), Error> {
    if want.is_empty() {
        return Ok((Vec::new(), false));
    }
    let Some(idx) = indexes else {
        return Ok((Vec::new(), false));
    };
    let Some(inc_root) = &idx.incoming else {
        return Ok((Vec::new(), false));
    };
    let Some(bucket_cid) = prolly::lookup(bs, inc_root, &ProllyKey::from(dst))? else {
        return Ok((Vec::new(), false));
    };
    let bucket: IncomingAdjacencyBucket = decode_from_store(bs, &bucket_cid)?;
    let mut out = Vec::new();
    let mut truncated = false;
    for ae in &bucket.edges {
        if want.contains(ae.label.as_str()) {
            if out.len() >= cap {
                truncated = true;
                break;
            }
            let edge: Edge = decode_from_store(bs, &ae.edge)?;
            out.push(edge);
        }
    }
    Ok((out, truncated))
}
