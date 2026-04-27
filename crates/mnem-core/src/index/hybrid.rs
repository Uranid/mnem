//! Hybrid adjacency: union of authored edges and derived KNN edges.
//!
//! Part of experiment E0 (LLM-free GraphRAG substrate). Provides an
//! [`AdjacencyIndex`] trait abstract enough for Leiden (E1), PPR (E2),
//! and the summariser (E4) to consume without knowing whether edges
//! came from the user or from the KNN derivation pipeline.
//!
//! # Provenance
//!
//! Every edge carries an [`EdgeProvenance`] tag so downstream algorithms
//! can weight authored and derived edges differently (e.g. Leiden can
//! upweight authored relations to preserve human-declared structure).
//!
//! # Flag-off contract
//!
//! A [`HybridAdjacency`] built from any authored source plus an
//! **empty** KNN contribution MUST iterate the authored edges exactly,
//! in the authored source's native order, producing byte-identical
//! behaviour to using the authored source alone. This is proven by
//! the integration test `hybrid_adjacency_union`.

use crate::id::NodeId;

/// Where an adjacency edge originated.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub enum EdgeProvenance {
    /// Edge declared by the user (typed graph edge in the repo).
    Authored,
    /// Edge derived from a KNN substrate (e.g. `mnem-ann::KnnEdgeIndex`).
    Knn,
}

/// One adjacency edge observed through an [`AdjacencyIndex`].
#[derive(Clone, Debug, PartialEq)]
pub struct AdjEdge {
    /// Source node.
    pub src: NodeId,
    /// Destination node.
    pub dst: NodeId,
    /// Edge weight. Authored edges default to `1.0`; KNN edges carry
    /// their similarity score.
    pub weight: f32,
    /// Who produced the edge.
    pub provenance: EdgeProvenance,
}

/// Minimal read-only adjacency surface consumed by the E1/E2/E4 layers.
///
/// Implementations produce edges in a deterministic order so two runs
/// over the same underlying data produce identical traversal output.
pub trait AdjacencyIndex {
    /// Iterate every edge. Used by the one-shot graph builders
    /// (Leiden, PPR matrix assembly).
    fn iter_edges(&self) -> Box<dyn Iterator<Item = AdjEdge> + '_>;

    /// Total number of edges. Cheap O(1) hint for allocation.
    fn edge_count(&self) -> usize;
}

// -----------------------------------------------------------------
// Authored slice view
// -----------------------------------------------------------------

/// Minimal [`AdjacencyIndex`] wrapper over a slice of authored
/// `(src, dst)` pairs. Primarily a test and integration aid; real
/// callers will wire the repo's AdjacencyBucket directly.
///
/// Edges are emitted in the slice's native order with a fixed weight
/// of `1.0`.
#[derive(Clone, Debug)]
pub struct AuthoredSliceAdjacency<'a> {
    edges: &'a [(NodeId, NodeId)],
}

impl<'a> AuthoredSliceAdjacency<'a> {
    /// Wrap a slice of authored edges.
    #[must_use]
    pub const fn new(edges: &'a [(NodeId, NodeId)]) -> Self {
        Self { edges }
    }
}

impl AdjacencyIndex for AuthoredSliceAdjacency<'_> {
    fn iter_edges(&self) -> Box<dyn Iterator<Item = AdjEdge> + '_> {
        Box::new(self.edges.iter().map(|(s, d)| AdjEdge {
            src: *s,
            dst: *d,
            weight: 1.0,
            provenance: EdgeProvenance::Authored,
        }))
    }
    fn edge_count(&self) -> usize {
        self.edges.len()
    }
}

// -----------------------------------------------------------------
// Hybrid wrapper
// -----------------------------------------------------------------

/// A [`KnnEdgeSource`] abstracts "something that yields
/// `(src, dst, weight)` derived edges". Implemented by anything
/// structurally equivalent to `mnem_ann::KnnEdgeIndex`. Defined
/// generically here so `mnem-core` does not depend on `mnem-ann`
/// (keeps the crate WASM-clean and avoids a cycle).
pub trait KnnEdgeSource {
    /// Iterate every derived `(src, dst, weight)` triple.
    fn iter_knn(&self) -> Box<dyn Iterator<Item = (NodeId, NodeId, f32)> + '_>;
    /// Count of derived edges.
    fn knn_len(&self) -> usize;
}

/// An empty KNN source. Wire this in when the KNN feature is off; the
/// flag-off byte-identity test asserts a [`HybridAdjacency`] carrying
/// this behaves exactly like the underlying authored source.
#[derive(Clone, Copy, Debug, Default)]
pub struct EmptyKnnSource;

impl KnnEdgeSource for EmptyKnnSource {
    fn iter_knn(&self) -> Box<dyn Iterator<Item = (NodeId, NodeId, f32)> + '_> {
        Box::new(std::iter::empty())
    }
    fn knn_len(&self) -> usize {
        0
    }
}

/// Union view over an authored [`AdjacencyIndex`] and a KNN
/// [`KnnEdgeSource`]. Dedupes on `(src, dst)`: when an edge appears
/// in both, the authored one wins and the KNN one is dropped (so
/// the agent's declared edge takes precedence).
///
/// Iteration order: every authored edge in the authored source's
/// order first, then every **unique** KNN edge in the KNN source's
/// order.
pub struct HybridAdjacency<A: AdjacencyIndex, K: KnnEdgeSource> {
    /// Authored adjacency source.
    pub authored: A,
    /// KNN-derived edges.
    pub knn: K,
}

impl<A: AdjacencyIndex, K: KnnEdgeSource> HybridAdjacency<A, K> {
    /// Construct a hybrid view.
    pub const fn new(authored: A, knn: K) -> Self {
        Self { authored, knn }
    }
}

impl<A: AdjacencyIndex, K: KnnEdgeSource> AdjacencyIndex for HybridAdjacency<A, K> {
    fn iter_edges(&self) -> Box<dyn Iterator<Item = AdjEdge> + '_> {
        // Collect authored keys for dedupe. For the scales E1/E2/E4
        // operate on (≤ low millions of authored edges) a HashSet
        // is comfortably within budget; if a larger corpus shows
        // up later we can swap for a sorted-vec + binary-search.
        let authored_edges: Vec<AdjEdge> = self.authored.iter_edges().collect();
        let mut seen: std::collections::HashSet<(NodeId, NodeId)> =
            std::collections::HashSet::with_capacity(authored_edges.len());
        for e in &authored_edges {
            seen.insert((e.src, e.dst));
        }
        let knn_iter = self.knn.iter_knn().filter_map(move |(s, d, w)| {
            if seen.contains(&(s, d)) {
                None
            } else {
                Some(AdjEdge {
                    src: s,
                    dst: d,
                    weight: w,
                    provenance: EdgeProvenance::Knn,
                })
            }
        });
        Box::new(authored_edges.into_iter().chain(knn_iter))
    }

    fn edge_count(&self) -> usize {
        // Upper-bound approximation - actual count after dedupe is
        // ≤ this. Good enough for pre-allocation sizing; callers
        // that need an exact count should collect.
        self.authored.edge_count() + self.knn.knn_len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_knn_yields_authored_exactly() {
        let a = NodeId::new_v7();
        let b = NodeId::new_v7();
        let c = NodeId::new_v7();
        let authored_pairs = [(a, b), (b, c)];
        let authored = AuthoredSliceAdjacency::new(&authored_pairs);
        let hybrid = HybridAdjacency::new(authored.clone(), EmptyKnnSource);

        let via_authored: Vec<AdjEdge> = authored.iter_edges().collect();
        let via_hybrid: Vec<AdjEdge> = hybrid.iter_edges().collect();
        assert_eq!(via_authored, via_hybrid);
    }

    #[test]
    fn knn_edges_tagged_with_provenance() {
        struct OneKnn(NodeId, NodeId);
        impl KnnEdgeSource for OneKnn {
            fn iter_knn(&self) -> Box<dyn Iterator<Item = (NodeId, NodeId, f32)> + '_> {
                Box::new(std::iter::once((self.0, self.1, 0.75)))
            }
            fn knn_len(&self) -> usize {
                1
            }
        }
        let a = NodeId::new_v7();
        let b = NodeId::new_v7();
        let authored_pairs: [(NodeId, NodeId); 0] = [];
        let hybrid =
            HybridAdjacency::new(AuthoredSliceAdjacency::new(&authored_pairs), OneKnn(a, b));
        let edges: Vec<AdjEdge> = hybrid.iter_edges().collect();
        assert_eq!(edges.len(), 1);
        assert_eq!(edges[0].provenance, EdgeProvenance::Knn);
        assert!((edges[0].weight - 0.75).abs() < 1e-6);
    }
}
