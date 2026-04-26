// SPLADE, BGE-M3, WordPiece, OpenSearch are proper-noun external
// identifiers; per-mention backticking adds no signal here.
#![allow(clippy::doc_markdown)]

//! Sparse-retrieval index for learned-sparse encoders (SPLADE,
//! BGE-M3-sparse, opensearch-doc-v3-distill). Pair with
//! [`crate::sparse::SparseEncoder`] adapters .
//!
//! # Why an inverted index (not brute force)
//!
//! SPLADE vectors are ~100-300 non-zero entries over a 30K-ish
//! WordPiece vocabulary. A brute-force `O(N_docs * nnz)` walk is
//! tolerable at a few thousand docs but collapses past 100K. The
//! inverted index turns query-time scoring into: for each non-zero
//! query token, look up the posting list of (doc_id, weight) pairs
//! and accumulate `query_weight * doc_weight` into a per-doc score
//! map. Total work is `O(sum(nnz(doc_i)))` summed over docs that
//! share at least one token with the query - typically far less than
//! `O(N * nnz)`.
//!
//! # Canonicality
//!
//! Posting lists sort by `(NodeId ASC)` at build time so the search
//! result's tie-break order is deterministic across runs (matches the
//! pattern used by [`crate::index::vector::BruteForceVectorIndex`]).
//!
//! # Model scoping
//!
//! Every index binds to a single `vocab_id` string. A query sparse
//! vector whose `vocab_id` differs from the index's returns an empty
//! result (and a debug log in a future instrumentation pass). This
//! prevents accidentally fusing incompatible models under RRF.
//!
//! # Future work
//!
//! WAND / MaxScore pruning, block-max posting-list skipping, and
//! disk-persisted postings. The current in-memory implementation is
//! the correctness baseline; optimisations are opt-in features
//! deferred to a follow-up.

use std::collections::HashMap;
use std::sync::Arc;

use crate::error::{Error, RepoError};
use crate::id::NodeId;
use crate::index::vector::VectorHit;
use crate::objects::Node;
use crate::prolly::Cursor;
use crate::repo::readonly::decode_from_store;
use crate::sparse::SparseEmbed;
use crate::store::Blockstore;

/// One posting list entry: `(NodeId, weight)`.
#[derive(Debug, Clone, Copy)]
struct Posting {
    node: NodeId,
    weight: f32,
}

/// A sparse inverted index over [`SparseEmbed`] values.
///
/// Build incrementally via [`Self::new`] + [`Self::add`], or in bulk
/// via [`Self::build_from_repo`]. Query via [`Self::search`].
///
/// Posting lists are stored as `HashMap<u32 token_id, Vec<Posting>>`,
/// where every `Vec<Posting>` is sorted by `NodeId ASC` for
/// deterministic tie-break behaviour matching the rest of mnem-core's
/// indexes.
#[derive(Debug, Clone)]
pub struct SparseInvertedIndex {
    postings: HashMap<u32, Vec<Posting>>,
    vocab_id: String,
    doc_count: u32,
}

impl SparseInvertedIndex {
    /// Construct an empty index bound to `vocab_id`. Nodes added
    /// via [`Self::add`] whose own `vocab_id` disagrees are silently
    /// skipped - mirrors [`BruteForceVectorIndex`][crate::index::vector::BruteForceVectorIndex]
    /// behaviour for cross-model documents.
    #[must_use]
    pub fn new(vocab_id: impl Into<String>) -> Self {
        Self {
            postings: HashMap::new(),
            vocab_id: vocab_id.into(),
            doc_count: 0,
        }
    }

    /// Vocabulary identifier this index is bound to.
    #[must_use]
    pub fn vocab_id(&self) -> &str {
        &self.vocab_id
    }

    /// Number of documents indexed.
    #[must_use]
    pub const fn doc_count(&self) -> u32 {
        self.doc_count
    }

    /// Feed one (node, sparse_embed) pair. Silently skips when the
    /// embed's `vocab_id` disagrees with the index's or when the
    /// embed has zero non-zero entries.
    pub fn add(&mut self, node: NodeId, embed: &SparseEmbed) {
        if embed.vocab_id != self.vocab_id {
            return;
        }
        if embed.indices.is_empty() {
            return;
        }
        for (i, w) in embed.indices.iter().zip(embed.values.iter()) {
            self.postings
                .entry(*i)
                .or_default()
                .push(Posting { node, weight: *w });
        }
        self.doc_count = self.doc_count.saturating_add(1);
    }

    /// Finalise the index: sort each posting list by `NodeId ASC` so
    /// search results tie-break deterministically. Call once after
    /// all `add()` calls; idempotent.
    pub fn finalize(&mut self) {
        for list in self.postings.values_mut() {
            list.sort_by(|a, b| a.node.cmp(&b.node));
        }
    }

    /// Search the index for the top-`k` documents by sparse-dot-product
    /// score against `query`. Returns [`VectorHit`] (same shape as the
    /// dense index so callers can fuse results without a custom type).
    ///
    /// On `vocab_id` mismatch returns an empty vec - the caller
    /// receives no scores to fuse, same semantics as a disjoint
    /// vocabulary.
    pub fn search(&self, query: &SparseEmbed, k: usize) -> Result<Vec<VectorHit>, Error> {
        if query.vocab_id != self.vocab_id {
            return Ok(Vec::new());
        }
        if query.indices.is_empty() || k == 0 {
            return Ok(Vec::new());
        }
        let mut scores: HashMap<NodeId, f32> = HashMap::new();
        for (tid, qw) in query.indices.iter().zip(query.values.iter()) {
            let Some(list) = self.postings.get(tid) else {
                continue;
            };
            for p in list {
                let e = scores.entry(p.node).or_insert(0.0);
                *e += qw * p.weight;
            }
        }
        let mut ranked: Vec<(NodeId, f32)> = scores.into_iter().collect();
        ranked.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.0.cmp(&b.0))
        });
        ranked.truncate(k);
        Ok(ranked
            .into_iter()
            .map(|(node_id, score)| VectorHit { node_id, score })
            .collect())
    }

    /// Build an index from all nodes in the current commit whose
    /// `sparse_embed` field matches `vocab_id`. Requires the nodes to
    /// have been indexed by an adapter at write time.
    ///
    /// # Errors
    ///
    /// - [`RepoError::Uninitialized`] if the repo has no head commit.
    /// - Store / codec errors while walking the Prolly tree.
    pub fn build_from_repo(
        repo: &crate::repo::ReadonlyRepo,
        vocab_id: impl Into<String>,
    ) -> Result<Self, Error> {
        let vocab_id = vocab_id.into();
        let mut idx = Self::new(&vocab_id);
        let bs: Arc<dyn Blockstore> = repo.blockstore().clone();
        let Some(commit) = repo.head_commit() else {
            return Err(RepoError::Uninitialized.into());
        };
        let cursor = Cursor::new(&*bs, &commit.nodes)?;
        for entry in cursor {
            let (_k, node_cid) = entry?;
            let node: Node = decode_from_store(&*bs, &node_cid)?;
            let Some(sparse) = &node.sparse_embed else {
                continue;
            };
            if sparse.vocab_id == vocab_id {
                idx.add(node.id, sparse);
            }
        }
        idx.finalize();
        Ok(idx)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sparse::SparseEmbed;

    fn nid(b: u8) -> NodeId {
        NodeId::from_bytes_raw([b; 16])
    }

    fn emb(indices: Vec<u32>, values: Vec<f32>) -> SparseEmbed {
        SparseEmbed::new(indices, values, "v0").unwrap()
    }

    #[test]
    fn empty_index_returns_empty_results() {
        let idx = SparseInvertedIndex::new("v0");
        let hits = idx.search(&emb(vec![1], vec![1.0]), 10).unwrap();
        assert!(hits.is_empty());
    }

    #[test]
    fn add_and_search_single_doc() {
        let mut idx = SparseInvertedIndex::new("v0");
        idx.add(nid(1), &emb(vec![10, 20], vec![0.5, 0.5]));
        idx.finalize();
        let hits = idx.search(&emb(vec![10], vec![1.0]), 10).unwrap();
        assert_eq!(hits.len(), 1);
        assert!((hits[0].score - 0.5).abs() < 1e-6);
    }

    #[test]
    fn search_ranks_by_dot_product_descending() {
        let mut idx = SparseInvertedIndex::new("v0");
        // doc1 shares token 10 strongly; doc2 shares tokens 10 + 20 but weakly.
        idx.add(nid(1), &emb(vec![10], vec![2.0]));
        idx.add(nid(2), &emb(vec![10, 20], vec![0.1, 0.1]));
        idx.add(nid(3), &emb(vec![99], vec![5.0])); // disjoint
        idx.finalize();
        let hits = idx.search(&emb(vec![10, 20], vec![1.0, 1.0]), 10).unwrap();
        assert_eq!(hits.len(), 2, "doc3 has disjoint tokens; must not appear");
        assert_eq!(hits[0].node_id, nid(1));
        assert_eq!(hits[1].node_id, nid(2));
        assert!(hits[0].score > hits[1].score);
    }

    #[test]
    fn k_caps_result_count() {
        let mut idx = SparseInvertedIndex::new("v0");
        for i in 1..=5 {
            idx.add(nid(i), &emb(vec![1], vec![f32::from(i)]));
        }
        idx.finalize();
        let hits = idx.search(&emb(vec![1], vec![1.0]), 3).unwrap();
        assert_eq!(hits.len(), 3);
    }

    #[test]
    fn vocab_mismatch_returns_empty() {
        let mut idx = SparseInvertedIndex::new("v0");
        idx.add(nid(1), &emb(vec![1], vec![1.0]));
        idx.finalize();
        let other = SparseEmbed::new(vec![1], vec![1.0], "v1").unwrap();
        let hits = idx.search(&other, 10).unwrap();
        assert!(hits.is_empty());
    }

    #[test]
    fn add_with_wrong_vocab_is_silently_skipped() {
        let mut idx = SparseInvertedIndex::new("v0");
        let foreign = SparseEmbed::new(vec![1], vec![1.0], "v1").unwrap();
        idx.add(nid(1), &foreign);
        assert_eq!(idx.doc_count(), 0);
    }

    #[test]
    fn zero_k_returns_empty() {
        let mut idx = SparseInvertedIndex::new("v0");
        idx.add(nid(1), &emb(vec![1], vec![1.0]));
        idx.finalize();
        let hits = idx.search(&emb(vec![1], vec![1.0]), 0).unwrap();
        assert!(hits.is_empty());
    }

    #[test]
    fn tie_breaks_on_node_id_ascending() {
        let mut idx = SparseInvertedIndex::new("v0");
        idx.add(nid(5), &emb(vec![1], vec![1.0]));
        idx.add(nid(2), &emb(vec![1], vec![1.0]));
        idx.add(nid(9), &emb(vec![1], vec![1.0]));
        idx.finalize();
        let hits = idx.search(&emb(vec![1], vec![1.0]), 10).unwrap();
        // All scores equal 1.0; tie-break should be NodeId ASC.
        assert_eq!(hits.len(), 3);
        assert_eq!(hits[0].node_id, nid(2));
        assert_eq!(hits[1].node_id, nid(5));
        assert_eq!(hits[2].node_id, nid(9));
    }

    #[test]
    fn empty_query_returns_empty() {
        let mut idx = SparseInvertedIndex::new("v0");
        idx.add(nid(1), &emb(vec![1], vec![1.0]));
        idx.finalize();
        let q = SparseEmbed::new(vec![], vec![], "v0").unwrap();
        let hits = idx.search(&q, 10).unwrap();
        assert!(hits.is_empty());
    }

    #[test]
    fn doc_count_tracks_adds() {
        let mut idx = SparseInvertedIndex::new("v0");
        assert_eq!(idx.doc_count(), 0);
        idx.add(nid(1), &emb(vec![1], vec![1.0]));
        assert_eq!(idx.doc_count(), 1);
        idx.add(nid(2), &emb(vec![1], vec![1.0]));
        assert_eq!(idx.doc_count(), 2);
    }

    #[test]
    fn search_is_deterministic_across_build_orders() {
        let mut idx1 = SparseInvertedIndex::new("v0");
        idx1.add(nid(1), &emb(vec![1, 2], vec![1.0, 0.5]));
        idx1.add(nid(2), &emb(vec![1, 3], vec![0.5, 1.0]));
        idx1.finalize();

        let mut idx2 = SparseInvertedIndex::new("v0");
        idx2.add(nid(2), &emb(vec![1, 3], vec![0.5, 1.0]));
        idx2.add(nid(1), &emb(vec![1, 2], vec![1.0, 0.5]));
        idx2.finalize();

        let q = emb(vec![1, 2, 3], vec![1.0, 1.0, 1.0]);
        let h1 = idx1.search(&q, 10).unwrap();
        let h2 = idx2.search(&q, 10).unwrap();
        let ids1: Vec<NodeId> = h1.iter().map(|h| h.node_id).collect();
        let ids2: Vec<NodeId> = h2.iter().map(|h| h.node_id).collect();
        assert_eq!(ids1, ids2);
    }
}
