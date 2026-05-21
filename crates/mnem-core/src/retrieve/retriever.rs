//! `Retriever` struct, `Debug` impl, and the builder + `execute`
//! implementation.
//!
//! Extracted from `retrieve.rs` in R3; bodies unchanged.

use std::collections::HashMap;
use std::sync::Arc;

use ipld_core::ipld::Ipld;

use crate::anchor::is_anchor_node_id;
use crate::error::{Error, RepoError};
use crate::id::NodeId;
use crate::index::{PropPredicate, VectorIndex};
use crate::objects::Node;
use crate::repo::readonly::ReadonlyRepo;

use super::community_filter::{CommunityFilterCfg, CommunityLookup, apply_community_filter};
use super::fusion::{convex_min_max_fusion, prefetch_and_filter, weighted_reciprocal_rank_fusion};
use super::render_node;
use super::types::{
    FusionStrategy, GraphExpand, GraphExpandDirection, GraphExpandMode, Lane, RetrievalResult,
    RetrievedItem, TemporalFilter,
};
use super::{HeuristicEstimator, TokenEstimator};

/// Agent-facing retrieval builder. See [the module docs](self).
#[derive(Clone)]
pub struct Retriever<'a> {
    repo: &'a ReadonlyRepo,
    label: Option<String>,
    prop_filter: Option<(String, PropPredicate)>,
    /// Original user text, retained so a cross-encoder reranker can be
    /// fed the joint `(query, candidate)` pair. The text itself does
    /// NOT drive any base ranker directly in mnem-core; the CLI /
    /// server embeds it and attaches the dense vector via
    /// [`Retriever::vector`].
    query_text: Option<String>,
    vector_query: Option<(String, Vec<f32>)>,
    token_budget: Option<u32>,
    limit: Option<usize>,
    rrf_k: f32,
    fusion: FusionStrategy,
    vector_weight: f32,
    sparse_weight: f32,
    sparse_query: Option<crate::sparse::SparseEmbed>,
    /// Pre-built vector index override. Lets callers (mnem http,
    /// long-lived services) cache indexes keyed by commit CID and
    /// avoid the O(N) rebuild on every retrieve. Bound to a specific
    /// embed model via BruteForceVectorIndex's own model field.
    vector_index_override: Option<Arc<crate::index::BruteForceVectorIndex>>,
    /// Pre-built sparse index override. Same pattern.
    sparse_index_override: Option<Arc<crate::index::SparseInvertedIndex>>,
    estimator: Arc<dyn TokenEstimator>,
    vector_cap: usize,
    reranker: Option<Arc<dyn crate::rerank::Reranker>>,
    rerank_top_k: usize,
    graph_expand: Option<GraphExpand>,
    /// Optional adjacency index used by PPR graph expansion (E2+).
    /// When `None`, PPR mode falls through to the historical decay
    /// walk so default retrieval is byte-identical across versions.
    adjacency_index: Option<Arc<dyn crate::index::hybrid::AdjacencyIndex + Send + Sync>>,
    /// When `true`, tombstoned nodes are kept in the result set
    /// (useful for audit / debug). Defaults to `false`: tombstones
    /// filter out by default, matching the agent-facing "forget"
    /// semantics documented in SPEC §4.10.
    include_tombstoned: bool,
    /// When `true`, system-reserved nodes (today: the `mnem init`
    /// anchor) are kept in the result set. Defaults to `false` so
    /// `mnem retrieve` never surfaces graph bookkeeping. Mirrors
    /// [`Self::include_tombstoned`] for audit / admin opt-in.
    include_system: bool,
    /// Optional temporal-range filter against the reserved props
    /// `mnem:created_at` / `mnem:updated_at` stamped by
    /// [`crate::repo::Transaction::commit_memory`]. See
    /// [`TemporalFilter`] for the lenient-on-legacy semantics.
    temporal_filter: Option<TemporalFilter>,
    /// Experiment E1 (C3 FIX-1): community-expander configuration.
    /// When `cfg.enabled` is `false` (default) this stage is a no-op
    /// and the fused candidate list passes straight through to the
    /// rerank stage. When enabled, the top-N seeds' communities pull
    /// in additional cohesive members (additive only - the stage
    /// NEVER drops existing candidates). Matrix v4 pinned -29pp R@10
    /// regression on the old drop-filter semantic; the expander is
    /// the recall-safe replacement.
    community_filter_cfg: CommunityFilterCfg,
    /// Experiment E1: opaque community lookup supplied by the caller
    /// (typically a `mnem_graphrag::CommunityAssignment`). `None`
    /// disables the stage regardless of `community_filter_cfg`.
    community_lookup: Option<Arc<CommunityLookup>>,
    /// Gap 02 #17: opt-in override for the PPR size-gate. When
    /// `false` (default) and the graph has more than
    /// [`crate::ppr::PPR_DEFAULT_MAX_NODES`] unique nodes, the PPR
    /// dispatch is skipped and the pipeline falls back to the
    /// decay-BFS walk. Set to `true` to accept the cost and run PPR
    /// over the large graph anyway (documented via the
    /// `ppr_size_gate_skipped` warning + metric).
    ppr_opt_in: bool,
}

impl std::fmt::Debug for Retriever<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Retriever")
            .field("label", &self.label)
            .field("prop_filter", &self.prop_filter)
            .field("query_text", &self.query_text)
            .field(
                "vector_query_len",
                &self.vector_query.as_ref().map(|(m, v)| (m, v.len())),
            )
            .field("token_budget", &self.token_budget)
            .field("limit", &self.limit)
            .field("rrf_k", &self.rrf_k)
            .field("vector_cap", &self.vector_cap)
            .field("temporal_filter", &self.temporal_filter)
            .finish()
    }
}

impl<'a> Retriever<'a> {
    /// Reciprocal-rank-fusion default per Cormack/Clarke/Buettcher 2009.
    /// k=60 is the canonical value; the one every downstream paper and
    /// industry search system ships as its default.
    pub const DEFAULT_RRF_K: f32 = 60.0;

    /// Default cap on vector-ranker depth. Brute-force cosine is O(n)
    /// already; more than a few hundred rows rarely shifts the RRF
    /// outcome enough to justify the sort cost.
    pub const DEFAULT_VECTOR_CAP: usize = 256;

    /// Default top-K of the fused list that gets sent to a reranker
    /// when one is installed. Cross-encoders are expensive (50-200ms
    /// per pair); 25 is the middle of the "worth rescoring" range
    /// used by Cohere / Voyage / BGE reranker docs.
    pub const DEFAULT_RERANK_TOP_K: usize = 25;

    /// Start a new retriever against `repo`.
    #[must_use]
    pub fn new(repo: &'a ReadonlyRepo) -> Self {
        Self {
            repo,
            label: None,
            prop_filter: None,
            query_text: None,
            vector_query: None,
            token_budget: None,
            limit: None,
            rrf_k: Self::DEFAULT_RRF_K,
            fusion: FusionStrategy::default(),
            vector_weight: 1.0,
            sparse_weight: 1.0,
            sparse_query: None,
            vector_index_override: None,
            sparse_index_override: None,
            estimator: Arc::new(HeuristicEstimator),
            vector_cap: Self::DEFAULT_VECTOR_CAP,
            reranker: None,
            rerank_top_k: Self::DEFAULT_RERANK_TOP_K,
            graph_expand: None,
            adjacency_index: None,
            include_tombstoned: false,
            include_system: false,
            temporal_filter: None,
            community_filter_cfg: CommunityFilterCfg::default(),
            community_lookup: None,
            ppr_opt_in: false,
        }
    }

    /// Gap 02 #17: opt into running PPR even when the graph exceeds
    /// [`crate::ppr::PPR_DEFAULT_MAX_NODES`].
    ///
    /// The default-on size gate skips PPR on oversized graphs and
    /// falls back to the decay-BFS walk so per-query latency stays
    /// bounded. Set this to `true` to bypass that gate and accept
    /// the unbounded cost (typical operator story: sharded
    /// deployment or wall-clock-tolerant batch job).
    #[must_use]
    pub const fn with_ppr_opt_in(mut self, opt_in: bool) -> Self {
        self.ppr_opt_in = opt_in;
        self
    }

    /// Experiment E1 (C3 FIX-1): install the community-**expander**
    /// stage between fusion and rerank. When `cfg.enabled` is `false`
    /// this is a byte-exact pass-through; when enabled, the top
    /// `cfg.expand_seeds` seeds' communities pull in up to
    /// `cfg.max_per_community` additional members (additive - NEVER
    /// drops existing candidates). For the additive effect to fire,
    /// the `lookup` MUST be constructed via
    /// [`CommunityLookup::new_with_members`]; `CommunityLookup::new`
    /// leaves the inverse map empty and the expander degenerates to
    /// the identity.
    #[must_use]
    pub fn with_community_filter(
        mut self,
        cfg: CommunityFilterCfg,
        lookup: Arc<CommunityLookup>,
    ) -> Self {
        self.community_filter_cfg = cfg;
        self.community_lookup = Some(lookup);
        self
    }

    /// Attach an adjacency index (authored edges, KNN-derived edges, or
    /// the hybrid union) for PPR graph expansion. Required whenever
    /// [`GraphExpand::mode`] is [`GraphExpandMode::Ppr`]; without it
    /// PPR mode falls through to the historical decay walk so default
    /// retrieval stays byte-identical across versions.
    ///
    /// The trait object must be `Send + Sync` because retrievers are
    /// routinely cloned across async worker pools in the HTTP layer.
    #[must_use]
    pub fn with_adjacency_index(
        mut self,
        adj: Arc<dyn crate::index::hybrid::AdjacencyIndex + Send + Sync>,
    ) -> Self {
        self.adjacency_index = Some(adj);
        self
    }

    /// Drop candidates whose `mnem:created_at` is strictly before
    /// `t_micros` (microseconds since Unix epoch). Lenient-on-legacy:
    /// nodes without the reserved prop pass this check. See
    /// [`TemporalFilter`] for the full contract.
    #[must_use]
    pub fn where_created_after(mut self, t_micros: u64) -> Self {
        let mut f = self.temporal_filter.unwrap_or_default();
        f.created_after = Some(t_micros);
        self.temporal_filter = Some(f);
        self
    }

    /// Drop candidates whose `mnem:created_at` is at or after
    /// `t_micros` (exclusive upper bound, microseconds since epoch).
    /// Lenient-on-legacy: nodes without the reserved prop pass.
    #[must_use]
    pub fn where_created_before(mut self, t_micros: u64) -> Self {
        let mut f = self.temporal_filter.unwrap_or_default();
        f.created_before = Some(t_micros);
        self.temporal_filter = Some(f);
        self
    }

    /// Drop candidates whose `mnem:updated_at` is strictly before
    /// `t_micros`. Lenient-on-legacy: nodes without the reserved prop
    /// pass.
    #[must_use]
    pub fn where_updated_after(mut self, t_micros: u64) -> Self {
        let mut f = self.temporal_filter.unwrap_or_default();
        f.updated_after = Some(t_micros);
        self.temporal_filter = Some(f);
        self
    }

    /// Drop candidates whose `mnem:updated_at` is at or after
    /// `t_micros` (exclusive). Lenient-on-legacy: nodes without the
    /// reserved prop pass.
    #[must_use]
    pub fn where_updated_before(mut self, t_micros: u64) -> Self {
        let mut f = self.temporal_filter.unwrap_or_default();
        f.updated_before = Some(t_micros);
        self.temporal_filter = Some(f);
        self
    }

    /// Include tombstoned nodes in the result set. Off by default:
    /// retrieval filters out any candidate whose `NodeId` is listed in
    /// the current View's tombstone map, matching the agent-facing
    /// "forget" semantics (SPEC §4.10).
    ///
    /// Flip to `true` for audit, debug, or restore flows that need to
    /// see revoked memory alongside live memory.
    #[must_use]
    pub const fn include_tombstoned(mut self, include: bool) -> Self {
        self.include_tombstoned = include;
        self
    }

    /// Include system-reserved nodes (today: the `mnem init` anchor)
    /// in the result set. Off by default so agent-facing retrieval
    /// never surfaces graph bookkeeping. Audit / admin callers opt in
    /// the same way they opt into tombstones; the filter passes
    /// through to the inner [`crate::index::query::Query`] and also
    /// applies at the prefetched, ranked, and graph-expand-neighbor
    /// stages so a system node can't sneak in via any path.
    #[must_use]
    pub const fn include_system(mut self, include: bool) -> Self {
        self.include_system = include;
        self
    }

    /// Enable graph-expand: after the hybrid fusion produces a top-K,
    /// traverse outgoing edges 1 hop from each seed and add neighbors
    /// as candidates with a decay-weighted score. The expanded list
    /// is what the reranker (if any) then re-scores.
    ///
    /// This is mnem's structural advantage over chunk-bag competitors:
    /// the graph is authored, not extracted, so expansion carries the
    /// agent-authored relationship signal without LLM-inferred noise.
    #[must_use]
    pub fn with_graph_expand(mut self, cfg: GraphExpand) -> Self {
        self.graph_expand = Some(cfg);
        self
    }

    /// Gate matches by node type.
    #[must_use]
    pub fn label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }

    /// Gate matches by a property predicate (same semantics as
    /// [`crate::index::Query::where_prop`]).
    #[must_use]
    pub fn where_prop(mut self, name: impl Into<String>, pred: PropPredicate) -> Self {
        self.prop_filter = Some((name.into(), pred));
        self
    }

    /// Convenience: `where_prop(name, PropPredicate::Eq(value.into()))`.
    #[must_use]
    pub fn where_eq(self, name: impl Into<String>, value: impl Into<Ipld>) -> Self {
        self.where_prop(name, PropPredicate::eq(value))
    }

    /// Attach the user's original text query. This text does NOT drive
    /// a lexical ranker in mnem-core ;
    /// it is retained only so a cross-encoder reranker, if installed
    /// via [`Self::with_reranker`], can read the `(query, candidate)`
    /// pair jointly. Callers that want the query to contribute to
    /// retrieval must embed it and attach the dense vector via
    /// [`Self::vector`] and / or attach a
    /// [`crate::sparse::SparseEmbed`] via [`Self::sparse_query`].
    #[must_use]
    pub fn query_text(mut self, query: impl Into<String>) -> Self {
        self.query_text = Some(query.into());
        self
    }

    /// Rank by cosine similarity to `vector` in the named embedding
    /// model's space.
    #[must_use]
    pub fn vector(mut self, model: impl Into<String>, vector: Vec<f32>) -> Self {
        self.vector_query = Some((model.into(), vector));
        self
    }

    /// Cap total rendered-text tokens in the result.
    #[must_use]
    pub const fn token_budget(mut self, tokens: u32) -> Self {
        self.token_budget = Some(tokens);
        self
    }

    /// Cap the number of items returned, independently of tokens.
    #[must_use]
    pub const fn limit(mut self, n: usize) -> Self {
        self.limit = Some(n);
        self
    }

    /// Choose the fusion strategy for this retrieve. See
    /// [`FusionStrategy`] for the semantics.
    #[must_use]
    pub const fn fusion(mut self, strategy: FusionStrategy) -> Self {
        self.fusion = strategy;
        self
    }

    /// Override the Reciprocal Rank Fusion smoothing constant `k`.
    /// Default is [`Self::DEFAULT_RRF_K`]. Only has effect when the
    /// fusion strategy is [`FusionStrategy::Rrf`].
    #[must_use]
    pub const fn rrf_k(mut self, k: f32) -> Self {
        self.rrf_k = k;
        self
    }

    /// Weight the cosine-vector ranker's contribution in the RRF
    /// fusion. Default 1.0. Raising it biases the fused order toward
    /// the dense lane; lowering it (including to 0.0 to disable)
    /// biases toward the sparse lane.
    #[must_use]
    pub const fn vector_weight(mut self, w: f32) -> Self {
        self.vector_weight = w;
        self
    }

    /// Weight on the learned-sparse ranker's contribution to RRF
    /// fusion . Default 1.0. Set to 0.0 to disable this
    /// lane even if a sparse query was attached.
    #[must_use]
    pub const fn sparse_weight(mut self, w: f32) -> Self {
        self.sparse_weight = w;
        self
    }

    /// Supply a pre-built vector index. The index's bound model must
    /// match the `vector(model, ...)` set on this retriever.
    #[must_use]
    pub fn with_vector_index(mut self, idx: Arc<crate::index::BruteForceVectorIndex>) -> Self {
        self.vector_index_override = Some(idx);
        self
    }

    /// Supply a pre-built sparse inverted index. Its `vocab_id` must
    /// match the `sparse_query` set on this retriever.
    #[must_use]
    pub fn with_sparse_index(mut self, idx: Arc<crate::index::SparseInvertedIndex>) -> Self {
        self.sparse_index_override = Some(idx);
        self
    }

    /// Attach a pre-computed sparse query embedding for the learned-
    /// sparse retrieval lane. The caller is responsible for producing
    /// this via a [`crate::sparse::SparseEncoder`] (SPLADE / BGE-M3 /
    /// opensearch-doc-v3-distill via adapter crate).
    ///
    /// The index is built from all nodes whose `sparse_embed` field
    /// matches the given `vocab_id`; the same encoder that produced
    /// the query must have produced the stored embeddings.
    #[must_use]
    pub fn sparse_query(mut self, embed: crate::sparse::SparseEmbed) -> Self {
        self.sparse_query = Some(embed);
        self
    }

    /// Override the vector ranker depth cap.
    #[must_use]
    pub const fn vector_cap(mut self, n: usize) -> Self {
        self.vector_cap = n;
        self
    }

    /// Install a cross-encoder reranker that rescores the top-K of the
    /// fused list before budget packing. This is **tier 3** of the
    /// compositional-retrieval hierarchy documented in
    /// `docs/guide/semantic-search.md`: a mechanism that reads
    /// `(query, candidate)` jointly and can bridge paraphrase that
    /// dense and sparse bi-encoders both miss because they encode
    /// the query and documents independently.
    ///
    /// The reranker is called with the canonical rendered form of
    /// each candidate node (via [`render_node`]). Failures fall back
    /// to the original fused order - the user still gets results, the
    /// reranker is an optimisation, not a gate.
    ///
    /// See [`crate::rerank::Reranker`] for the trait and
    ///  for the design.
    #[must_use]
    pub fn with_reranker(mut self, reranker: Arc<dyn crate::rerank::Reranker>) -> Self {
        self.reranker = Some(reranker);
        self
    }

    /// Cap the number of fused candidates the reranker rescores.
    /// Default [`Self::DEFAULT_RERANK_TOP_K`]. Cross-encoders are
    /// linear in candidates; keep this small.
    #[must_use]
    pub const fn rerank_top_k(mut self, k: usize) -> Self {
        self.rerank_top_k = k;
        self
    }

    /// Install a custom token estimator. Default is [`HeuristicEstimator`].
    #[must_use]
    pub fn estimator(mut self, estimator: Arc<dyn TokenEstimator>) -> Self {
        self.estimator = estimator;
        self
    }

    /// Execute the retrieval.
    ///
    /// # Errors
    ///
    /// - [`RepoError::Uninitialized`] if the repo has no head commit.
    /// - [`RepoError::RetrievalEmpty`] if no filters and no rankers
    ///   were configured (there is nothing to retrieve).
    /// - [`RepoError::VectorDimMismatch`] if the query vector's
    ///   dimension does not match the built vector index.
    /// - Store / codec errors from walking trees or decoding nodes.
    ///
    /// # Instrumentation
    ///
    /// Emits one `info`-level span `mnem::retrieve::execute` per call.
    /// Fields `lane_count`, `candidate_count`, and `items_returned` are
    /// populated via [`tracing::Span::record`] just before the function
    /// returns, so an operator watching `RUST_LOG=mnem::retrieve=info`
    /// sees one compact line per retrieval with bounded cardinality.
    /// Node payloads, summaries, and CIDs are NOT recorded (they would
    /// blow up span size for even modest batches).
    #[tracing::instrument(
        name = "execute",
        level = "info",
        target = "mnem::retrieve",
        skip(self),
        fields(lane_count = tracing::field::Empty, candidate_count = tracing::field::Empty, items_returned = tracing::field::Empty)
    )]
    pub fn execute(self) -> Result<RetrievalResult, Error> {
        let Self {
            repo,
            label,
            prop_filter,
            query_text,
            vector_query,
            token_budget,
            limit,
            rrf_k,
            fusion,
            vector_weight,
            sparse_weight,
            sparse_query,
            estimator,
            vector_cap,
            reranker,
            rerank_top_k,
            graph_expand,
            adjacency_index,
            vector_index_override,
            sparse_index_override,
            include_tombstoned,
            include_system,
            temporal_filter,
            community_filter_cfg,
            community_lookup,
            ppr_opt_in,
        } = self;

        // Gap 02 #17: tracks whether the PPR size gate tripped on this
        // call. Surfaced on `RetrievalResult.ppr_size_gate_skipped` so
        // the HTTP handler (which owns the warnings[] vector and the
        // Prometheus registry) can emit the warning + counter.
        let mut ppr_size_gate_skipped = false;

        if label.is_none()
            && prop_filter.is_none()
            && vector_query.is_none()
            && sparse_query.is_none()
        {
            return Err(RepoError::RetrievalEmpty.into());
        }

        // --- Collect each ranker's hits WITH their native scores.
        // We keep the scores alongside the node ids so the single-ranker
        // path can pass the native cosine / sparse dot-product score
        // through to the user unchanged. Only when >=2 rankers fire
        // does score get replaced by the rank-based RRF number.
        let vector_hits: Option<Vec<(NodeId, f32)>> = if let Some((model, vec)) = &vector_query {
            let owned;
            let idx: &crate::index::BruteForceVectorIndex = match &vector_index_override {
                Some(a) => a.as_ref(),
                None => {
                    owned = repo.build_vector_index(model)?;
                    &owned
                }
            };
            Some(
                idx.search(vec, vector_cap)?
                    .into_iter()
                    .map(|h| (h.node_id, h.score))
                    .collect(),
            )
        } else {
            None
        };

        // Learned-sparse lane . Builds an
        // in-memory inverted index from all nodes whose
        // `sparse_embed` matches the query's `vocab_id`, then scores
        // via sparse dot product.
        let sparse_hits: Option<Vec<(NodeId, f32)>> = if let Some(q) = &sparse_query {
            let owned;
            let idx: &crate::index::SparseInvertedIndex = match &sparse_index_override {
                Some(a) => a.as_ref(),
                None => {
                    owned = crate::index::SparseInvertedIndex::build_from_repo(
                        repo,
                        q.vocab_id.clone(),
                    )?;
                    &owned
                }
            };
            Some(
                idx.search(q, vector_cap)?
                    .into_iter()
                    .map(|h| (h.node_id, h.score))
                    .collect(),
            )
        } else {
            None
        };

        let any_ranker_requested = vector_hits.is_some() || sparse_hits.is_some();

        // Per-lane diagnostic scores (observability, -follow-on).
        // Lane order is fixed so iteration is deterministic.
        let mut node_lane_scores: HashMap<NodeId, Vec<(Lane, f32)>> = HashMap::new();
        if let Some(v) = &vector_hits {
            for (id, s) in v {
                node_lane_scores
                    .entry(*id)
                    .or_default()
                    .push((Lane::Vector, *s));
            }
        }
        if let Some(sp) = &sparse_hits {
            for (id, s) in sp {
                node_lane_scores
                    .entry(*id)
                    .or_default()
                    .push((Lane::Sparse, *s));
            }
        }

        // Build per-lane data for the configured fusion strategy.
        // RRF cares only about rank order; min-max convex combination
        // needs the native scores. We collect both shapes so the
        // branch below just picks.
        let mut ranked_lanes: Vec<(Vec<NodeId>, f32)> = Vec::with_capacity(2);
        let mut scored_lanes: Vec<(Vec<(NodeId, f32)>, f32)> = Vec::with_capacity(2);
        let mut single_lane_score_passthrough: Option<Vec<(NodeId, f32)>> = None;
        if let Some(v) = &vector_hits {
            ranked_lanes.push((v.iter().map(|(id, _)| *id).collect(), vector_weight));
            scored_lanes.push((v.clone(), vector_weight));
        }
        if let Some(sp) = &sparse_hits {
            ranked_lanes.push((sp.iter().map(|(id, _)| *id).collect(), sparse_weight));
            scored_lanes.push((sp.clone(), sparse_weight));
        }
        // Single-lane pass-through: return that lane's native scores
        // instead of fused numbers so a user only using one ranker
        // still sees interpretable magnitudes.
        if ranked_lanes.len() == 1 {
            single_lane_score_passthrough = Some(match (&vector_hits, &sparse_hits) {
                (Some(v), None) => v.clone(),
                (None, Some(sp)) => sp.clone(),
                _ => unreachable!("single-lane branch reached with multiple lanes"),
            });
        }

        let ranked: Vec<(NodeId, f32)> = if let Some(pass) = single_lane_score_passthrough {
            pass
        } else if ranked_lanes.is_empty() {
            Vec::new()
        } else {
            match fusion {
                FusionStrategy::Rrf => weighted_reciprocal_rank_fusion(&ranked_lanes, rrf_k),
                FusionStrategy::ConvexMinMax => convex_min_max_fusion(&scored_lanes),
            }
        };

        // --- Community expander (experiment E1 - C3 FIX-1) ---
        // When the caller installed a community lookup AND
        // `community_filter_cfg.enabled` is true, pull in
        // community-cohesive siblings of the top seeds as additional
        // candidates. This is additive only: existing candidates are
        // preserved in original order, new members are appended with
        // decayed scores. When the lookup or flag is missing this
        // stage is a byte-exact pass-through.
        let ranked = if let Some(lookup) = community_lookup.as_ref() {
            apply_community_filter(ranked, lookup.as_ref(), community_filter_cfg)
        } else {
            ranked
        };

        // --- Filter + prefetch nodes in a single pass ---
        // Every surviving candidate needs its `Node` below for rendering;
        // pre-fetching once (rather than looking up in both the filter
        // gate and the pack loop) halves the Prolly-tree lookups on the
        // hot path.
        let mut prefetched: Vec<(NodeId, f32, Node)> = if any_ranker_requested {
            prefetch_and_filter(repo, ranked, label.as_deref(), prop_filter.as_ref())?
        } else {
            // Filter-only mode: the structured query already returns
            // hits that carry the decoded `Node`, so reuse those.
            let mut q = repo
                .query()
                .include_tombstoned(include_tombstoned)
                .include_system(include_system);
            if let Some(lbl) = &label {
                q = q.label(lbl.clone());
            }
            if let Some((name, pred)) = &prop_filter {
                q = q.where_prop(name.clone(), pred.clone());
            }
            let hits = q.execute()?;
            hits.into_iter().map(|h| (h.node.id, 1.0, h.node)).collect()
        };

        // --- Tombstone filter (SPEC §4.10) ---
        // Default: drop any candidate whose NodeId appears in the
        // current View's tombstone map. `include_tombstoned(true)`
        // opts out for audit / debug callers. Applied BEFORE graph-
        // expand so a tombstoned seed cannot contribute neighbors via
        // its authored edges; neighbors discovered through graph-
        // expand are filtered a second time further down.
        if !include_tombstoned && !repo.view().tombstones.is_empty() {
            prefetched.retain(|(id, _, _)| !repo.is_tombstoned(id));
        }

        // --- System-node filter (anchor) ---
        // Default: drop the `mnem init` anchor from the candidate pool.
        // It carries no content, has no agent-meaningful embedding, and
        // would otherwise appear as low-score noise in every retrieve.
        // `include_system(true)` opts back in for audit / repair flows.
        // Mirrors the tombstone filter shape so future system-reserved
        // nodes get the same treatment without code churn.
        if !include_system {
            prefetched.retain(|(id, _, _)| !is_anchor_node_id(id));
        }

        // --- Temporal-range filter (agent-support track, mnem/0.3+) ---
        // Gate on the reserved `mnem:created_at` / `mnem:updated_at`
        // props stamped by `commit_memory`. Applied AFTER fusion so
        // the filter gates the fused candidate list, and BEFORE
        // graph-expand so a too-old seed cannot pull its neighbors in
        // through the structural expand step. Lenient-on-legacy:
        // candidates without the reserved prop pass every check (see
        // `TemporalFilter` docs).
        if let Some(tf) = temporal_filter.as_ref()
            && !tf.is_empty()
        {
            prefetched.retain(|(_, _, node)| tf.matches(node));
        }

        // --- Graph expand (tier 2, mnem's structural advantage) ---
        // Take the current top-K seeds, traverse outgoing edges 1 hop,
        // and merge neighbors as new candidates with score = seed_score
        // * decay. Neighbors that already appear in the seed list are
        // skipped. The expanded list is then fed to the reranker (if
        // any) so the cross-encoder can promote good neighbors and
        // demote weak seeds in one pass.
        //
        // E2: PPR mode dispatch. When `GraphExpandMode::Ppr` is
        // selected AND the caller attached an adjacency index via
        // `with_adjacency_index`, run personalised PageRank over the
        // hybrid graph instead of the decay-BFS. The personalization
        // vector is the current seed set scored by fused lane output.
        // When either precondition is missing we fall through to the
        // decay path so the default retrieve stays byte-identical.
        let ppr_requested = matches!(
            graph_expand.as_ref().map(|ge| ge.mode),
            Some(GraphExpandMode::Ppr { .. })
        ) && adjacency_index.is_some();

        // Gap 02 #17 size-gate. Evaluated BEFORE the PPR dispatch so
        // an oversized graph falls through to the decay walk instead
        // of paying the O(k * |E|) PPR cost. See
        // [`crate::ppr::exceeds_size_gate`] for the counting contract
        // and threshold rationale.
        if ppr_requested
            && let Some(adj) = &adjacency_index
            && crate::ppr::exceeds_size_gate(adj.as_ref(), ppr_opt_in)
        {
            ppr_size_gate_skipped = true;
        }
        let ppr_active = ppr_requested && !ppr_size_gate_skipped;

        if ppr_active
            && let Some(ge) = &graph_expand
            && let Some(adj) = &adjacency_index
            && !prefetched.is_empty()
            && let GraphExpandMode::Ppr {
                damping,
                max_iter,
                eps,
            } = ge.mode
        {
            // Build personalization from the current seed set.
            let mut pers: std::collections::BTreeMap<NodeId, f32> =
                std::collections::BTreeMap::new();
            let mut seen: std::collections::HashSet<NodeId> =
                prefetched.iter().map(|(id, _, _)| *id).collect();
            for (id, score, _) in &prefetched {
                // Clamp negatives; PPR's personalization must be
                // non-negative or the L1 renormalisation breaks.
                let w = score.max(0.0);
                if w > 0.0 {
                    pers.insert(*id, w);
                }
            }
            let cfg = crate::ppr::PprConfig {
                damping,
                max_iter,
                eps,
            };
            let scores = crate::ppr::ppr(adj.as_ref(), &pers, cfg);
            // Sort by PPR score desc, NodeId asc for determinism.
            let mut ranked: Vec<(NodeId, f32)> = scores.into_iter().collect();
            ranked.sort_by(|a, b| {
                b.1.partial_cmp(&a.1)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| a.0.cmp(&b.0))
            });
            // Filter out seeds (they're already in prefetched) and
            // tombstoned / system nodes (unless the audit override is on).
            ranked.retain(|(id, _)| !seen.contains(id));
            if !include_tombstoned {
                ranked.retain(|(id, _)| !repo.is_tombstoned(id));
            }
            if !include_system {
                ranked.retain(|(id, _)| !is_anchor_node_id(id));
            }
            ranked.truncate(ge.max_expand);
            for (nbr_id, score) in ranked {
                if let Some(node) = repo.lookup_node(&nbr_id)? {
                    if let Some(lbl) = &label
                        && &node.ntype != lbl
                    {
                        continue;
                    }
                    if let Some(tf) = temporal_filter.as_ref()
                        && !tf.is_empty()
                        && !tf.matches(&node)
                    {
                        continue;
                    }
                    seen.insert(nbr_id);
                    node_lane_scores
                        .entry(nbr_id)
                        .or_default()
                        .push((Lane::GraphExpand, score));
                    prefetched.push((nbr_id, score, node));
                }
            }
            prefetched.sort_by(|a, b| {
                b.1.partial_cmp(&a.1)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| a.0.cmp(&b.0))
            });
        }
        if let Some(ge) = &graph_expand
            && !ppr_active
            && !prefetched.is_empty()
        {
            let mut seen: std::collections::HashSet<NodeId> =
                prefetched.iter().map(|(id, _, _)| *id).collect();
            let etype_filter_strs: Option<Vec<&str>> = ge
                .etype_filter
                .as_deref()
                .map(|v| v.iter().map(String::as_str).collect());
            // Multi-hop BFS with decay^depth per hop and optional
            // per-edge-type weight multiplier. We relax-max into a
            // deterministic `BTreeMap` keyed by NodeId so the final
            // truncate sees byte-stable order; `HashMap` here would
            // make the graph-expand output non-reproducible across
            // machines even though retrieval-determinism is one of
            // mnem's headline contracts.
            //
            // Per-hop we collect the frontier as a `BTreeMap` too so
            // duplicates naturally collapse (silent-data-loss audit
            // flagged a bug where two sibling seeds reaching the
            // same `dst` both pushed it into `next_frontier`,
            // producing quadratic-in-depth walks of the same nodes).
            let mut neighbor_scores: std::collections::BTreeMap<NodeId, f32> =
                std::collections::BTreeMap::new();
            let mut frontier: std::collections::BTreeMap<NodeId, f32> = prefetched
                .iter()
                .map(|(id, score, _)| (*id, *score))
                .collect();
            // Select the adjacency walk(s) based on requested direction.
            // Outgoing -> visit each edge.dst; Incoming -> visit each
            // edge.src (use the back-index). Any -> both, and the
            // relax-max keeps the best score encountered on either side.
            let walk_out = matches!(
                ge.direction,
                GraphExpandDirection::Outgoing | GraphExpandDirection::Both
            );
            let walk_in = matches!(
                ge.direction,
                GraphExpandDirection::Incoming | GraphExpandDirection::Both
            );
            for _hop in 0..ge.depth {
                let mut next_frontier: std::collections::BTreeMap<NodeId, f32> =
                    std::collections::BTreeMap::new();
                for (src_id, src_score) in &frontier {
                    // Gather (neighbor_id, etype) pairs from whichever
                    // direction(s) the caller asked for.
                    let mut neighbors: Vec<(NodeId, String)> = Vec::new();
                    if walk_out {
                        let edges = repo.outgoing_edges(src_id, etype_filter_strs.as_deref())?;
                        let iter: Box<dyn Iterator<Item = _>> = match ge.max_per_seed {
                            Some(cap) => Box::new(edges.into_iter().take(cap)),
                            None => Box::new(edges.into_iter()),
                        };
                        for edge in iter {
                            neighbors.push((edge.dst, edge.etype));
                        }
                    }
                    if walk_in {
                        let edges = repo.incoming_edges(src_id, etype_filter_strs.as_deref())?;
                        let iter: Box<dyn Iterator<Item = _>> = match ge.max_per_seed {
                            Some(cap) => Box::new(edges.into_iter().take(cap)),
                            None => Box::new(edges.into_iter()),
                        };
                        for edge in iter {
                            // Back-edge: src_id is the dst; promote src.
                            neighbors.push((edge.src, edge.etype));
                        }
                    }
                    for (nbr_id, etype) in neighbors {
                        if seen.contains(&nbr_id) {
                            continue;
                        }
                        let etype_mult = ge.edge_weight.get(&etype).copied().unwrap_or(1.0);
                        let expanded_score = src_score * ge.decay * etype_mult;
                        // Relax-max: keep the best score discovered
                        // through any path to this neighbor. BTreeMap
                        // + max is associative/commutative, so the
                        // result is path-order-independent.
                        let bumped = match neighbor_scores.get(&nbr_id) {
                            Some(prev) if *prev >= expanded_score => false,
                            _ => {
                                neighbor_scores.insert(nbr_id, expanded_score);
                                true
                            }
                        };
                        // Only promote to the next hop if this path
                        // actually improved the score. Dedup via the
                        // BTreeMap key: a second sibling seed that
                        // reaches the same dst with a higher score
                        // just updates in place, no duplicate walk.
                        if bumped {
                            next_frontier
                                .entry(nbr_id)
                                .and_modify(|s| {
                                    if expanded_score > *s {
                                        *s = expanded_score;
                                    }
                                })
                                .or_insert(expanded_score);
                        }
                    }
                }
                // Per-hop frontier cap. A pathological graph (hot seed
                // with tens of thousands of out-edges, or an adversary
                // who crafted a dense subgraph around a likely seed)
                // could otherwise push `next_frontier` into the
                // 10k-100k range, at which point the relax-max BTreeMap
                // work, the subsequent-hop adjacency queries, and the
                // final `neighbor_scores` sort all blow up. The final
                // `truncate(max_expand)` would have thrown most of this
                // away anyway; we short-circuit at the source. Fail
                // loud but don't surface an error - partial expansion
                // is better than an empty result for the end user.
                if next_frontier.len() > ge.max_frontier {
                    eprintln!(
                        "[mnem] graph-expand aborting hop: frontier {} exceeds max_frontier={}; returning partial expansion",
                        next_frontier.len(),
                        ge.max_frontier
                    );
                    break;
                }
                // Promote this hop's discoveries into the seen set so
                // the next hop's walks don't revisit them via a
                // different edge.
                for id in next_frontier.keys() {
                    seen.insert(*id);
                }
                frontier = next_frontier;
                if frontier.is_empty() {
                    break;
                }
            }
            // Resolve neighbor nodes (bounded by max_expand of the best-scored).
            let mut ranked: Vec<(NodeId, f32)> = neighbor_scores.into_iter().collect();
            ranked.sort_by(|a, b| {
                b.1.partial_cmp(&a.1)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| a.0.cmp(&b.0))
            });
            // Tombstone / system filter BEFORE truncate: filtering
            // after truncate would let unwanted high-scoring neighbors
            // crowd out live lower-scoring ones, hiding usable results
            // under the cap. Opt out via `include_tombstoned(true)` /
            // `include_system(true)` for audit callers.
            if !include_tombstoned {
                ranked.retain(|(nbr_id, _)| !repo.is_tombstoned(nbr_id));
            }
            if !include_system {
                ranked.retain(|(nbr_id, _)| !is_anchor_node_id(nbr_id));
            }
            ranked.truncate(ge.max_expand);
            for (nbr_id, score) in ranked {
                if let Some(node) = repo.lookup_node(&nbr_id)? {
                    // Apply label filter to expanded neighbors (cheap,
                    // matches most retriever configurations). A prop_filter
                    // on the Retriever is intentionally NOT applied to
                    // expanded neighbors: a 1-hop neighbor may legitimately
                    // not match the prop predicate yet still be the right
                    // answer via the edge. Callers who want strict prop
                    // filtering on the full expanded set should post-filter
                    // the RetrievalResult.
                    if let Some(lbl) = &label
                        && &node.ntype != lbl
                    {
                        continue;
                    }
                    // Apply the temporal-range filter to expanded
                    // neighbors too: the contract is a filter against
                    // the FULL result set, not just the seeds. Nodes
                    // lacking the reserved prop pass (lenient).
                    if let Some(tf) = temporal_filter.as_ref()
                        && !tf.is_empty()
                        && !tf.matches(&node)
                    {
                        continue;
                    }
                    // Record the graph-expand contribution for
                    // observability. `score` is already the
                    // decay-and-edge-weight-adjusted value produced
                    // by the multi-hop BFS above.
                    node_lane_scores
                        .entry(nbr_id)
                        .or_default()
                        .push((Lane::GraphExpand, score));
                    prefetched.push((nbr_id, score, node));
                }
            }
            // Re-sort the full prefetched list by score so the reranker
            // window sees neighbors interleaved with seeds correctly.
            prefetched.sort_by(|a, b| {
                b.1.partial_cmp(&a.1)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| a.0.cmp(&b.0))
            });
        }

        // --- Cross-encoder rerank (tier 3) ---
        // When a reranker is installed AND we have a text query, take
        // the top-K of the filtered+prefetched list, send it plus the
        // query to the reranker, and re-sort that window by its
        // scores. The tail past K keeps its fused order. This is
        // where compositional paraphrase is bridged: the reranker
        // reads (query, candidate) jointly and can score "Natasha is
        // my aunt" above "Olivia is my cousin" for a query about
        // "father's sister", which neither dense nor sparse
        // bi-encoders can do alone.
        if let (Some(rr), Some(query_text)) = (&reranker, query_text.as_deref()) {
            let split = prefetched.len().min(rerank_top_k);
            if split > 0 {
                let head_texts: Vec<String> = prefetched[..split]
                    .iter()
                    .map(|(_, _, n)| render_node(n))
                    .collect();
                let head_refs: Vec<&str> = head_texts.iter().map(String::as_str).collect();
                match rr.rerank(query_text, &head_refs) {
                    Ok(scores) if scores.len() == split => {
                        // Record each reranked node's new score under
                        // Lane::Rerank so callers can see the
                        // cross-encoder contribution separately from
                        // the fused signal it replaced.
                        for (i, s) in scores.iter().enumerate() {
                            let (id, _, _) = &prefetched[i];
                            node_lane_scores
                                .entry(*id)
                                .or_default()
                                .push((Lane::Rerank, *s));
                        }
                        // Build (index, score) pairs and sort by score DESC,
                        // tie-break on NodeId ASC for determinism.
                        let mut order: Vec<(usize, f32)> = scores.into_iter().enumerate().collect();
                        order.sort_by(|a, b| {
                            b.1.partial_cmp(&a.1)
                                .unwrap_or(std::cmp::Ordering::Equal)
                                .then_with(|| prefetched[a.0].0.cmp(&prefetched[b.0].0))
                        });
                        // Permute the head in place using the new order,
                        // replacing the fused score with the reranker's.
                        let head: Vec<(NodeId, f32, Node)> = order
                            .into_iter()
                            .map(|(i, s)| {
                                let (id, _old_score, node) = prefetched[i].clone();
                                (id, s, node)
                            })
                            .collect();
                        let tail: Vec<(NodeId, f32, Node)> =
                            prefetched.iter().skip(split).cloned().collect();
                        prefetched = head;
                        prefetched.extend(tail);
                    }
                    // Score count mismatch or error: keep the fused
                    // order. Rerank is a refinement, not a gate. We
                    // emit a stderr line (not a tracing span because
                    // mnem-core is tracing-free ) so
                    // operators can spot a misconfigured reranker
                    // without reading our source.
                    Ok(scores) => {
                        eprintln!(
                            "[mnem] reranker score-count mismatch: expected {split}, got {}; falling back to fused order",
                            scores.len()
                        );
                    }
                    Err(e) => {
                        eprintln!("[mnem] reranker failed: {e}; falling back to fused order");
                    }
                }
            }
        }

        let candidates_seen = u32::try_from(prefetched.len()).unwrap_or(u32::MAX);

        // --- Render + pack under budget ---
        let budget = token_budget.unwrap_or(u32::MAX);
        let cap = limit.unwrap_or(usize::MAX);
        let mut items: Vec<RetrievedItem> = Vec::with_capacity(prefetched.len().min(cap));
        let mut tokens_used: u32 = 0;
        let mut dropped: u32 = 0;

        for (node_id, score, node) in prefetched {
            if items.len() >= cap {
                // `limit` reached: any remaining candidates count as
                // dropped so callers can see "there was more".
                dropped = dropped.saturating_add(1);
                continue;
            }
            let rendered = render_node(&node);
            let tokens = estimator.estimate(&rendered);
            let next = tokens_used.saturating_add(tokens);
            if next > budget {
                dropped = dropped.saturating_add(1);
                continue;
            }
            tokens_used = next;
            // Attach per-lane diagnostics in a fixed canonical order
            // (Vector, Sparse, GraphExpand, Rerank) so callers see
            // deterministic iteration regardless of the insertion
            // order the pipeline happened to use.
            let mut lane_scores = node_lane_scores.remove(&node_id).unwrap_or_default();
            lane_scores.sort_by_key(|(l, _)| *l);
            items.push(RetrievedItem {
                node,
                rendered,
                tokens,
                score,
                lane_scores,
            });
        }

        // Record span fields just before return: lane count (how many
        // rankers actually fired), candidate count (pre-pack), items
        // returned (post-budget). All bounded by construction.
        let span = tracing::Span::current();
        let lane_count = u32::from(vector_hits.is_some()) + u32::from(sparse_hits.is_some());
        span.record("lane_count", lane_count);
        span.record("candidate_count", candidates_seen);
        span.record(
            "items_returned",
            u32::try_from(items.len()).unwrap_or(u32::MAX),
        );

        Ok(RetrievalResult {
            items,
            tokens_used,
            tokens_budget: budget,
            dropped,
            candidates_seen,
            ppr_size_gate_skipped,
        })
    }
}
