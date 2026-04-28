//! Secondary indexes and the [`Query`] engine (Phase 2).
//!
//! The indexes are built automatically on every `Transaction::commit`
//! and stored as an [`IndexSet`][crate::objects::IndexSet] object
//! reachable through `Commit::indexes`. Without them, every query
//! would degenerate to a full cursor scan of the node / edge Prolly
//! trees. With them, agent queries like "all Person nodes", "node
//! where name='Alice'", and "outgoing edges of X" are O(log n) or
//! better.
//!
//! ## Layout
//!
//! See [`crate::objects::index_set`] for the `IndexSet` struct. In
//! short:
//!
//! - `nodes_by_label[label]` is a Prolly tree keyed by `NodeId` -> node CID.
//! - `nodes_by_prop[label][prop_name]` is a Prolly tree keyed by
//!   `blake3(canonical_ipld(value))[..16]` -> node CID.
//! - `outgoing` is a Prolly tree keyed by **source** `NodeId` -> CID of
//!   an [`AdjacencyBucket`][crate::objects::AdjacencyBucket] holding
//!   `(edge_label, edge_cid)` pairs.
//! - `incoming` is a Prolly tree keyed by **destination** `NodeId` ->
//!   CID of an
//!   [`IncomingAdjacencyBucket`][crate::objects::IncomingAdjacencyBucket]
//!   holding `(edge_label, src, edge_cid)` triples. Symmetric mirror
//!   of `outgoing`; lets "who points at X through edge-type T" run in
//!   the same O(log n) shape as the forward query.
//!
//! Everything under the indexes is itself content-addressed, so two
//! identical graphs produce byte-identical index roots and a Commit's
//! content hash is fully deterministic.

pub mod adjacency;
pub mod build;
pub mod hybrid;
pub mod query;
pub mod resolve;
pub mod sparse;
pub mod vector;

pub use build::{build_index_set, incremental_append_indexes, prop_value_hash};
pub use hybrid::{
    AdjEdge, AdjacencyIndex, AuthoredSliceAdjacency, EdgeProvenance, EmptyKnnSource,
    HybridAdjacency, KnnEdgeSource,
};
pub use query::{Direction, PropPredicate, Query, QueryHit};
pub use resolve::lookup_by_prop;
pub use sparse::SparseInvertedIndex;
pub use vector::{BruteForceVectorIndex, VectorHit, VectorIndex};

use crate::error::{Error, RepoError};
use crate::id::Cid;

/// Wrap a decode failure with index-row provenance. Used at every
/// site inside this module where a CID fetched from an index is
/// decoded; a bare `CodecError`/`StoreError` would throw away the
/// "which index row produced this bad CID" signal.
///
/// Only transforms the error when it's a `StoreError::NotFound` or a
/// `CodecError` - anything else (I/O, hashing) is surfaced as-is.
pub(super) fn wrap_index_decode_error(err: Error, context: String, cid: &Cid) -> Error {
    match &err {
        Error::Store(crate::error::StoreError::NotFound { .. }) | Error::Codec(_) => {
            RepoError::IndexCorrupt {
                context,
                cid: cid.clone(),
            }
            .into()
        }
        _ => err,
    }
}

#[cfg(test)]
mod tests;
