//! Content-addressed KNN-edge substrate (experiment E0).
//!
//! Derives a deterministic K-nearest-neighbours edge list from an
//! [`HnswVectorIndex`] and wraps it in a content-addressed
//! [`KnnEdgeIndex`] whose CID is a pure function of
//! `(hnsw_root_cid, k, metric_tag, sorted_edges_bytes)`.
//!
//! # Why brute-force over stored vectors?
//!
//! HNSW's search API is *approximate*: re-running `search(q, k)` on
//! the same graph is stable only up to the library's tie-breaking,
//! and it relies on an inverse-cosine score that the caller recovered
//! from `instant-distance`'s squared-Euclidean. For E0 we need an
//! **exact**, byte-identical edge set across re-derivation, so we
//! brute-force the KNN over the L2-normalised vectors the HNSW index
//! already retains. HNSW thus plays the role of a vector datastore;
//! the *edge derivation* is deterministic and metric-exact.
//!
//! This makes E0 work identically whether the underlying index is
//! brute-force or HNSW, which is the property the E1/E2/E4 layers
//! need.
//!
//! # Determinism contract
//!
//! - Input vectors come from [`HnswVectorIndex::points_iter`] in
//!   build order (which is canonical Prolly-tree node order).
//! - For each source, the top-k destinations are selected by
//!   **ascending L2-squared distance on the stored unit vectors**,
//!   tie-broken by **`NodeId` ASC**.
//! - Self-loops are excluded.
//! - Final edge list is sorted by `(src, dst)` ASC before CBOR
//!   encoding so two independent derivations produce byte-identical
//!   output.
//!
//! # CID composition
//!
//! `compute_cid` hashes a fixed preamble (`b"mnem/knn-edge/v1"`),
//! the HNSW root CID's canonical bytes, `k` as a big-endian u32, the
//! `DistanceMetric` tag as a single byte, and the canonical
//! DAG-CBOR encoding of the `KnnEdgeIndex` struct, then wraps the
//! result in a `CIDv1(codec=raw, multihash=sha2-256)`.

use mnem_core::codec::to_canonical_bytes;
use mnem_core::error::Error;
use mnem_core::id::{CODEC_RAW, Cid, Multihash, NodeId};

use serde::{Deserialize, Serialize};

#[cfg(feature = "hnsw")]
use crate::hnsw::HnswVectorIndex;

/// Distance metric tag carried inside a [`KnnEdgeIndex`].
///
/// Encoded as a single-byte discriminant in the CID preimage so the
/// content address is stable across crate versions even if the enum
/// is extended later.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum DistanceMetric {
    /// Cosine similarity. Vectors are expected to be L2-normalised
    /// before comparison; rank order is identical to squared
    /// Euclidean on unit vectors.
    Cosine = 1,
    /// Euclidean (L2) distance on raw vectors.
    L2 = 2,
    /// Dot product (inner product) similarity.
    Dot = 3,
}

impl DistanceMetric {
    /// The single-byte discriminant mixed into the CID preimage.
    #[must_use]
    pub const fn tag(self) -> u8 {
        self as u8
    }
}

/// One directed KNN edge from `src` to `dst` with a scalar weight.
///
/// Weight is the *similarity* in the chosen metric, already converted
/// to "higher is closer" so downstream graph algorithms can treat it
/// as a positive edge strength without per-metric branching:
///
/// - `Cosine` / `Dot`: cosine similarity in `[-1, 1]`.
/// - `L2`: `1 / (1 + d)` in `(0, 1]` so nearer = larger.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct KnnEdge {
    /// Source NodeId.
    pub src: NodeId,
    /// Destination NodeId. Never equal to `src` by construction.
    pub dst: NodeId,
    /// Similarity weight; higher = closer.
    pub weight: f32,
}

/// Content-addressed KNN-edge index derived from a vector index.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct KnnEdgeIndex {
    /// Root CID of the vector index this edge set was derived from.
    /// Folded into [`KnnEdgeIndex::compute_cid`] so a re-index under
    /// a fresh HNSW build (different CID) addresses a distinct
    /// edge-set even if `k` and metric match.
    pub root_cid: Cid,
    /// Per-source neighbour count.
    pub k: u32,
    /// Distance metric used to rank neighbours.
    pub metric: DistanceMetric,
    /// Sorted by `(src, dst)` ASC. See module docs for the
    /// determinism contract.
    pub edges: Vec<KnnEdge>,
}

impl KnnEdgeIndex {
    /// Compute the content-addressed CID of this index.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Codec`] if CBOR encoding fails (which in
    /// practice should not happen for this struct shape).
    pub fn compute_cid(&self) -> Result<Cid, Error> {
        // Assemble a preimage that starts with a domain-separation
        // tag so a `KnnEdgeIndex` CID can never collide with any
        // other object class hashed under sha2-256.
        let body = to_canonical_bytes(self)?;
        let mut buf: Vec<u8> = Vec::with_capacity(body.len() + 64);
        buf.extend_from_slice(b"mnem/knn-edge/v1");
        buf.extend_from_slice(&self.root_cid.to_bytes());
        buf.extend_from_slice(&self.k.to_be_bytes());
        buf.push(self.metric.tag());
        buf.extend_from_slice(&body);
        let hash = Multihash::sha2_256(&buf);
        Ok(Cid::new(CODEC_RAW, hash))
    }

    /// Construct an empty KNN-edge index anchored to the given root
    /// CID. Useful for "flag off" behaviour in [`crate::knn_edges`]
    /// integrations.
    #[must_use]
    pub fn empty(root_cid: Cid, k: u32, metric: DistanceMetric) -> Self {
        Self {
            root_cid,
            k,
            metric,
            edges: Vec::new(),
        }
    }

    /// Number of edges in this index. Convenience accessor.
    #[must_use]
    pub fn len(&self) -> usize {
        self.edges.len()
    }

    /// Whether this index has zero edges.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.edges.is_empty()
    }
}

// -----------------------------------------------------------------
// Pure derivation (no HNSW dependency)
// -----------------------------------------------------------------

/// Derive a KNN edge set by brute force over L2-normalised vectors.
///
/// This is the metric-exact kernel shared by the HNSW-backed entry
/// point ([`derive_knn_edges`]) and the test-only constructor.
///
/// `ids.len()` must equal `vecs.len()`. Each `vec` is expected to be
/// L2-normalised when `metric` is [`DistanceMetric::Cosine`].
///
/// Returns edges sorted by `(src, dst)` ASC.
#[must_use]
pub fn derive_knn_edges_from_vectors(
    ids: &[NodeId],
    vecs: &[Vec<f32>],
    k: u32,
    metric: DistanceMetric,
) -> Vec<KnnEdge> {
    assert_eq!(ids.len(), vecs.len(), "ids/vecs length mismatch");
    let n = ids.len();
    if n == 0 || k == 0 {
        return Vec::new();
    }
    let k_usize = (k as usize).min(n.saturating_sub(1));
    if k_usize == 0 {
        return Vec::new();
    }

    let mut edges: Vec<KnnEdge> = Vec::with_capacity(n * k_usize);

    // Scratch buffer reused across sources; holds (score_desc, dst_id).
    // `score_desc` means larger = closer.
    let mut scored: Vec<(f32, NodeId)> = Vec::with_capacity(n);

    for i in 0..n {
        scored.clear();
        for j in 0..n {
            if i == j {
                continue;
            }
            let sim = similarity(&vecs[i], &vecs[j], metric);
            scored.push((sim, ids[j]));
        }
        // Sort DESC by similarity; tie-break NodeId ASC for byte-stable
        // replay across identical vector duplicates.
        scored.sort_by(|a, b| {
            b.0.partial_cmp(&a.0)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.1.cmp(&b.1))
        });
        for (sim, dst) in scored.iter().take(k_usize) {
            edges.push(KnnEdge {
                src: ids[i],
                dst: *dst,
                weight: *sim,
            });
        }
    }

    // Final canonical order: (src, dst) ASC. Stable so equal keys
    // preserve the per-source insertion order (which is already
    // similarity-DESC).
    edges.sort_by(|a, b| a.src.cmp(&b.src).then_with(|| a.dst.cmp(&b.dst)));
    edges
}

fn similarity(a: &[f32], b: &[f32], metric: DistanceMetric) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    match metric {
        DistanceMetric::Cosine | DistanceMetric::Dot => {
            // On L2-normalised inputs cosine == dot. We compute the
            // unnormalised dot product; callers feeding unnormalised
            // vectors with `DistanceMetric::Dot` get raw inner product.
            let mut s = 0.0_f32;
            for (x, y) in a.iter().zip(b.iter()) {
                s += x * y;
            }
            s
        }
        DistanceMetric::L2 => {
            let mut acc = 0.0_f32;
            for (x, y) in a.iter().zip(b.iter()) {
                let d = x - y;
                acc += d * d;
            }
            // Convert distance to similarity so higher = closer.
            1.0 / (1.0 + acc.sqrt())
        }
    }
}

// -----------------------------------------------------------------
// HNSW-backed entry point (feature-gated)
// -----------------------------------------------------------------

/// Derive the content-addressed KNN-edge index from an HNSW vector
/// index.
///
/// The edge set is derived by brute-force over the stored (already
/// L2-normalised) vectors the HNSW graph retains; see the module
/// docstring for the determinism rationale. Metric is fixed to
/// [`DistanceMetric::Cosine`] because [`HnswVectorIndex`] internally
/// normalises to unit vectors.
///
/// `root_cid` is the caller-supplied content-address of the source
/// vector index. Passing it explicitly (rather than deriving it from
/// the HNSW graph here) avoids baking an HNSW-specific serialisation
/// contract into this crate and matches how future backends (DiskANN,
/// IVFPQ) will expose their root hash.
#[cfg(feature = "hnsw")]
#[must_use]
pub fn derive_knn_edges(hnsw: &HnswVectorIndex, k: u32, root_cid: Cid) -> KnnEdgeIndex {
    let mut ids: Vec<NodeId> = Vec::with_capacity(hnsw.points_len());
    let mut vecs: Vec<Vec<f32>> = Vec::with_capacity(hnsw.points_len());
    for (id, v) in hnsw.points_iter() {
        ids.push(id);
        vecs.push(v.to_vec());
    }
    let edges = derive_knn_edges_from_vectors(&ids, &vecs, k, DistanceMetric::Cosine);
    KnnEdgeIndex {
        root_cid,
        k,
        metric: DistanceMetric::Cosine,
        edges,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn demo_cid() -> Cid {
        Cid::new(
            mnem_core::id::CODEC_DAG_CBOR,
            Multihash::sha2_256(b"demo-hnsw-root"),
        )
    }

    #[test]
    fn empty_input_yields_empty_index() {
        let edges = derive_knn_edges_from_vectors(&[], &[], 5, DistanceMetric::Cosine);
        assert!(edges.is_empty());
    }

    #[test]
    fn compute_cid_is_stable_across_two_calls() {
        let idx = KnnEdgeIndex::empty(demo_cid(), 3, DistanceMetric::Cosine);
        let c1 = idx.compute_cid().unwrap();
        let c2 = idx.compute_cid().unwrap();
        assert_eq!(c1, c2);
    }

    #[test]
    fn distance_metric_tag_stable() {
        assert_eq!(DistanceMetric::Cosine.tag(), 1);
        assert_eq!(DistanceMetric::L2.tag(), 2);
        assert_eq!(DistanceMetric::Dot.tag(), 3);
    }
}
