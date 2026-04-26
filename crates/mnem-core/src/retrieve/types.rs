//! Retrieval result / config types: `Lane`, `RetrievedItem`,
//! `RetrievalResult`, `GraphExpand`, `GraphExpandDirection`,
//! `TemporalFilter`, `FusionStrategy`.
//!
//! Extracted from `retrieve.rs` in R3; bodies unchanged.

use std::collections::HashMap;

use ipld_core::ipld::Ipld;

use crate::objects::Node;

// ============================================================
// Retriever
// ============================================================

/// Which retrieval lane contributed a given score to a [`RetrievedItem`].
///
/// Populated by [`Retriever::execute`][super::Retriever::execute] so callers can answer "why did
/// this node rank?" without reverse-engineering the fusion. A lane
/// with a non-zero entry is a lane that actually surfaced this node;
/// a lane absent from [`RetrievedItem::lane_scores`] did not.
///
/// Stable `Ord` / `Hash` so retrieval results are deterministic under
/// lane ordering.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[non_exhaustive]
pub enum Lane {
    /// Dense vector cosine (the embedder lane).
    Vector,
    /// Learned-sparse dot product (SPLADE / BGE-M3 / neural-sparse-v3).
    Sparse,
    /// Graph-expand bonus. The score is the best `decay^hop *
    /// edge_weight * seed_score` reachable for this node.
    GraphExpand,
    /// Cross-encoder rerank score (post-fusion). When a reranker
    /// runs on the top-K of the fused list, this is the raw score
    /// the reranker returned for the `(query, candidate)` pair.
    Rerank,
}

/// One item returned by a [`Retriever`][super::Retriever] execution.
#[derive(Clone, Debug)]
#[non_exhaustive]
#[allow(clippy::module_name_repetitions)]
pub struct RetrievedItem {
    /// The matched node.
    pub node: Node,
    /// The canonical rendering used for token-budget packing. The
    /// caller can forward this string directly into an LLM context or
    /// adapt it; either way it is a pure function of `node`.
    pub rendered: String,
    /// Estimated tokens consumed by `rendered` under the retriever's
    /// configured [`TokenEstimator`][super::TokenEstimator].
    pub tokens: u32,
    /// Composite retrieval score. Single-lane runs return the lane's
    /// native score (cosine, sparse dot, ...); multi-lane runs return
    /// the fused score (convex-min-max by default, RRF via
    /// [`FusionStrategy::Rrf`]). Rerank, when active, overwrites the
    /// composite with the reranker's own score.
    pub score: f32,
    /// Per-lane diagnostics. Each entry is `(Lane, native_score)` for
    /// a lane that surfaced this node. Preserves the raw numbers that
    /// went INTO fusion so callers can tune per-corpus weights or
    /// debug unexpected rankings. Empty when the retriever hit the
    /// filter-only path (label/prop filters, no ranker).
    ///
    /// Ordering is deterministic: lanes are inserted in a fixed
    /// order (Vector, Sparse, GraphExpand, Rerank) and duplicates
    /// are not possible.
    pub lane_scores: Vec<(Lane, f32)>,
}

impl RetrievedItem {
    /// Construct a `RetrievedItem` from its parts. Provided because
    /// the struct is `#[non_exhaustive]` for forward-compat but
    /// callers outside `mnem-core` (e.g. CLI multi-query RRF-fusion)
    /// still need to synthesise results.
    #[must_use]
    pub fn new(node: Node, rendered: String, tokens: u32, score: f32) -> Self {
        Self {
            node,
            rendered,
            tokens,
            score,
            lane_scores: Vec::new(),
        }
    }

    /// Native score this node earned in a specific lane, if that lane
    /// surfaced it. Convenience lookup over [`Self::lane_scores`].
    #[must_use]
    pub fn lane_score(&self, lane: Lane) -> Option<f32> {
        self.lane_scores
            .iter()
            .find_map(|&(l, s)| if l == lane { Some(s) } else { None })
    }
}

/// The full result of a [`Retriever::execute`][super::Retriever::execute] call. Carries both the
/// packed items and cost metadata so agents can surface the packing
/// decision back to their user or their own logs.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct RetrievalResult {
    /// Items that fit inside the token budget, in RRF-rank order.
    pub items: Vec<RetrievedItem>,
    /// Total estimated tokens consumed by `items`.
    pub tokens_used: u32,
    /// The budget the caller configured (or `u32::MAX` if unset).
    pub tokens_budget: u32,
    /// Candidates that ranked highly enough to be considered but did
    /// not fit inside the remaining budget. A non-zero value signals
    /// that the budget was tight and raising it would surface more.
    pub dropped: u32,
    /// Total distinct nodes that survived ranker fusion + filtering,
    /// before budget packing.
    pub candidates_seen: u32,
    /// Gap 02 #17: `true` when `graph_mode = "ppr"` was requested but
    /// the PPR dispatcher skipped the walk because the graph exceeds
    /// [`crate::ppr::PPR_DEFAULT_MAX_NODES`] and the caller did not
    /// opt in via `with_ppr_opt_in(true)`. Callers that manage their
    /// own warnings / metrics surface this field to emit the
    /// `PprSizeGateSkipped` warning and bump the
    /// `mnem_ppr_size_gate_skipped_total` counter. Always `false` for
    /// non-PPR calls.
    pub ppr_size_gate_skipped: bool,
}

impl RetrievalResult {
    /// Construct from parts. Needed by outside-crate callers (e.g.
    /// CLI multi-query fusion) because the struct is `#[non_exhaustive]`.
    ///
    /// `ppr_size_gate_skipped` defaults to `false`; callers inside
    /// mnem-core's retriever set it via direct struct literal, not
    /// this constructor.
    #[must_use]
    pub fn new(
        items: Vec<RetrievedItem>,
        tokens_used: u32,
        tokens_budget: u32,
        dropped: u32,
        candidates_seen: u32,
    ) -> Self {
        Self {
            items,
            tokens_used,
            tokens_budget,
            dropped,
            candidates_seen,
            ppr_size_gate_skipped: false,
        }
    }
}

/// Graph-expand strategy selector.
///
/// Added in E2 (personalised PageRank). The historical `Decay` mode
/// is the default and byte-identical to the pre-E2 behaviour; new
/// callers can opt into `Ppr` to get random-walk-based multi-hop
/// expansion over a hybrid adjacency index (authored + KNN substrate).
///
/// `Ppr` is a no-op unless the caller supplies an adjacency index to
/// the [`super::Retriever`] via
/// [`super::Retriever::with_adjacency_index`][super::Retriever::with_adjacency_index].
/// Wired into CLI / HTTP / MCP in E2 turn T2 as a forward-compatible
/// flag; the actual PPR walk over the repo's live graph lands in
/// E2 turn T3 consumer integration.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum GraphExpandMode {
    /// The historical multi-hop BFS with `decay^depth` scoring. Fields
    /// governing this walk live on [`GraphExpand`] itself (`decay`,
    /// `depth`, `max_per_seed`, `max_frontier`). Default.
    #[default]
    Decay,
    /// Personalised PageRank expansion. Parameters:
    ///
    /// - `damping`: teleport factor, see [`crate::ppr::DEFAULT_DAMPING`].
    /// - `max_iter`: power-iteration cap.
    /// - `eps`: L1 convergence threshold.
    Ppr {
        /// Damping factor `d`. Clamped to `[0, 0.999]` at runtime.
        damping: f32,
        /// Power-iteration cap.
        max_iter: u32,
        /// L1 convergence threshold.
        eps: f32,
    },
}

/// Configuration for the graph-expand post-filter (, see
/// `docs/guide/semantic-search.md` tier-2).
///
/// After the hybrid fusion produces a top-K of seed nodes, the
/// retriever traverses outgoing edges 1 hop from each seed and adds
/// the neighbors as candidates with a decay-weighted score:
/// `score(neighbor) = score(seed) * decay`. Neighbors that also
/// appear in the seed list are skipped.
///
/// Why this is mnem's moat: chunk-bag competitors (mem0, Zep, bare
/// vector DBs) cannot do this because they have no graph. mnem does.
/// Per the LightRAG research brief this costs ~100 LoC and is
/// expected to yield +15-30 points Recall@10 on 2-hop MuSiQue
/// questions; flat ±2 on single-hop.
///
/// # Mode selection (E2+)
///
/// The [`mode`](GraphExpand::mode) field picks between the historical
/// decay-BFS strategy (default, byte-identical to pre-E2 behaviour)
/// and the new PPR power-iteration strategy. See [`GraphExpandMode`].
#[derive(Debug, Clone)]
pub struct GraphExpand {
    /// Maximum neighbors to add, across all hops and seeds. Bounds
    /// the post-filter cost when a hot-seed node has many out-edges.
    pub max_expand: usize,
    /// Score multiplier applied per hop. At hop `h` a neighbor
    /// inherits its parent's score times `decay^h`. Values in (0, 1)
    /// rank neighbors below their seeds; 1.0 treats them as equals.
    pub decay: f32,
    /// Optional edge-type filter. `None` = traverse every outgoing
    /// edge. `Some(labels)` = traverse only edges whose `etype` is in
    /// the list.
    pub etype_filter: Option<Vec<String>>,
    /// How many hops to expand. Default `1` preserves the original
    /// single-hop behavior. `2` lets MuSiQue-style composition
    /// ("Alice's mentor's employer") reach through one intermediate
    /// node; higher values pull in deeper chains at a decay penalty.
    pub depth: usize,
    /// Per-edge-type weight multiplier applied ON TOP of `decay`. Lets
    /// authored relation types that are known to carry signal (e.g.
    /// `mentions = 1.0`, `cites = 0.8`) contribute more than
    /// narrative or UI-only edges. An etype absent from this map gets
    /// the implicit default of 1.0.
    pub edge_weight: HashMap<String, f32>,
    /// Optional cap on outgoing edges explored per seed node. Without
    /// it, a "hot seed" with 1000 out-edges drowns out every sibling
    /// seed's neighbors in the global cap. `None` disables the per-
    /// seed cap (pre-depth behaviour).
    pub max_per_seed: Option<usize>,
    /// Per-hop cap on the BFS frontier. If a single hop's discovered
    /// frontier would exceed this, the hop aborts (no further hops
    /// walked) and a warning is emitted on stderr. Distinct from
    /// `max_expand` which caps the final ranked output: the frontier
    /// cap stops the walk itself so a malicious seed node with 100k
    /// out-edges cannot `DoS` the retriever into a multi-second BFS
    /// even though the eventual truncate would drop all but 20.
    /// Default [`GraphExpand::DEFAULT_MAX_FRONTIER`] (~5000).
    pub max_frontier: usize,
    /// Traversal direction. Default [`GraphExpandDirection::Outgoing`]
    /// preserves the pre-incoming-index behaviour. Flip to
    /// [`GraphExpandDirection::Incoming`] to walk backwards through
    /// the new symmetric index; [`GraphExpandDirection::Both`] walks
    /// both and takes the max-score path to each neighbor.
    pub direction: GraphExpandDirection,
    /// Strategy selector. Default [`GraphExpandMode::Decay`] runs the
    /// historical BFS; [`GraphExpandMode::Ppr`] switches to
    /// personalised PageRank (requires an adjacency index on the
    /// retriever; otherwise falls through to the decay walk).
    pub mode: GraphExpandMode,
}

/// Which edge direction(s) `GraphExpand` follows from a seed node.
///
/// `Outgoing` is the historical behaviour. The other two exist
/// because agents frequently model memory with "about me" edges where
/// the seed is the `dst` and the useful signal comes from the `src`
/// side (authorship, provenance, supersession chains). Backing off to
/// a scan would defeat the O(log n) adjacency lookup that makes this
/// feature viable at scale.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub enum GraphExpandDirection {
    /// Walk `src -> dst` edges. Historical default.
    #[default]
    Outgoing,
    /// Walk `dst -> src` edges (use the `incoming` index).
    Incoming,
    /// Walk both directions and keep the best-score path. `Any` is
    /// kept as a deprecated alias for call sites that pre-date the
    /// rename and will be dropped at the next breaking bump.
    Both,
}

impl GraphExpandDirection {
    /// Deprecated alias for [`GraphExpandDirection::Both`]. The name
    /// flipped between the 0.3 RCs; `Any` stays as a const so older
    /// call sites compile, but all new code should use `Both`.
    #[deprecated(note = "renamed to `Both`")]
    #[allow(non_upper_case_globals)]
    pub const Any: Self = Self::Both;
}

impl GraphExpand {
    /// Sensible default: 20 neighbors max, 0.7 decay, any edge type,
    /// single-hop. Picked from the LightRAG brief + MuSiQue 2-hop
    /// audit findings.
    pub const DEFAULT_MAX_EXPAND: usize = 20;
    /// Default per-hop decay factor.
    pub const DEFAULT_DECAY: f32 = 0.7;
    /// Default number of hops. `1` preserves pre-0.1.0 behaviour.
    pub const DEFAULT_DEPTH: usize = 1;
    /// Default per-hop frontier cap. 5000 is ~10x the largest legit
    /// seed fan-out Bob's audit observed on a real agent memory,
    /// leaving plenty of headroom for benign graphs while still
    /// bounding the worst-case walk time.
    pub const DEFAULT_MAX_FRONTIER: usize = 5000;

    /// Convenience: construct with defaults and no etype filter.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Convenience: filter to one edge label.
    #[must_use]
    pub fn with_etype(mut self, etype: impl Into<String>) -> Self {
        let mut list = self.etype_filter.unwrap_or_default();
        list.push(etype.into());
        self.etype_filter = Some(list);
        self
    }

    /// Set the traversal depth. Capped at 4 to keep the frontier
    /// from exploding; benchmarks beyond 4 saturate anyway per the
    /// MuSiQue 4-hop row.
    ///
    /// If the caller asks for `depth > 4` the value is clamped AND
    /// a warning is emitted on stderr so the user sees their
    /// requested depth was silently reduced. Clamping on `depth < 1`
    /// raises to 1 (no-op is meaningless for graph-expand).
    #[must_use]
    pub fn with_depth(mut self, depth: usize) -> Self {
        let clamped = depth.clamp(1, 4);
        if clamped != depth {
            eprintln!(
                "[mnem] GraphExpand::with_depth({depth}) clamped to {clamped} (supported range is 1..=4; raise this limit by forking DEFAULT_DEPTH or setting `depth` on the struct directly)"
            );
        }
        self.depth = clamped;
        self
    }

    /// Attach a per-edge-type weight multiplier.
    #[must_use]
    pub fn with_edge_weight(mut self, etype: impl Into<String>, weight: f32) -> Self {
        self.edge_weight.insert(etype.into(), weight);
        self
    }

    /// Cap outgoing edges explored per seed. Stops a hot seed from
    /// starving siblings in the global `max_expand` budget.
    #[must_use]
    pub const fn with_max_per_seed(mut self, cap: usize) -> Self {
        self.max_per_seed = Some(cap);
        self
    }

    /// Set the per-hop BFS frontier cap. If a single hop's discovered
    /// frontier exceeds `cap`, the walk aborts early (no further
    /// hops) and a warning is emitted on stderr. See the field docs
    /// on `max_frontier` for why this differs from `max_expand`.
    ///
    /// A cap of `0` disables the walk entirely; `usize::MAX` is the
    /// effective "no cap" sentinel.
    #[must_use]
    pub const fn with_max_frontier(mut self, cap: usize) -> Self {
        self.max_frontier = cap;
        self
    }

    /// Walk `dst -> src` edges. Uses the incoming-adjacency index
    /// added in mnem/0.3; gracefully degrades to "no neighbors" on
    /// older repos that lack the index.
    #[must_use]
    pub const fn with_incoming(mut self) -> Self {
        self.direction = GraphExpandDirection::Incoming;
        self
    }

    /// Walk edges in BOTH directions from each seed.
    #[must_use]
    pub const fn with_both_directions(mut self) -> Self {
        self.direction = GraphExpandDirection::Both;
        self
    }

    /// Direct setter mirroring the scope in the 0.3 design note:
    /// `GraphExpand::with_direction(GraphExpandDirection::Outgoing |
    /// Incoming | Both)`. Convenience for callers that already
    /// carry a `GraphExpandDirection` by value.
    #[must_use]
    pub const fn with_direction(mut self, direction: GraphExpandDirection) -> Self {
        self.direction = direction;
        self
    }

    /// Switch to personalised PageRank expansion. Requires an adjacency
    /// index on the retriever (see
    /// [`super::Retriever::with_adjacency_index`][super::Retriever::with_adjacency_index]);
    /// otherwise the retriever falls through to the historical decay walk.
    #[must_use]
    pub const fn with_ppr(mut self, damping: f32, max_iter: u32, eps: f32) -> Self {
        self.mode = GraphExpandMode::Ppr {
            damping,
            max_iter,
            eps,
        };
        self
    }

    /// Explicit mode setter. Preserves the other fields so callers can
    /// flip between `Decay` and `Ppr` without rebuilding the config.
    #[must_use]
    pub const fn with_mode(mut self, mode: GraphExpandMode) -> Self {
        self.mode = mode;
        self
    }
}

impl Default for GraphExpand {
    fn default() -> Self {
        Self {
            max_expand: Self::DEFAULT_MAX_EXPAND,
            decay: Self::DEFAULT_DECAY,
            etype_filter: None,
            depth: Self::DEFAULT_DEPTH,
            edge_weight: HashMap::new(),
            max_per_seed: None,
            max_frontier: Self::DEFAULT_MAX_FRONTIER,
            direction: GraphExpandDirection::default(),
            mode: GraphExpandMode::default(),
        }
    }
}

/// Temporal-range filter (agent-support track, mnem/0.3+).
///
/// Optional half-open bounds against the reserved props
/// `mnem:created_at` and `mnem:updated_at` that
/// [`crate::repo::Transaction::commit_memory`] stamps on every node
/// it writes. All bounds are in microseconds since the Unix epoch;
/// `*_after` is inclusive, `*_before` is exclusive.
///
/// **Lenient-on-legacy semantics:** nodes that do not carry the
/// reserved prop (pre-0.3 nodes, or 0.3+ nodes written via the low-
/// level [`crate::repo::Transaction::add_node`] without auto-stamp)
/// PASS every temporal check. This keeps the filter usable on mixed-
/// vintage repos; callers that want a stricter "unknown-timestamp
/// excludes" rule can follow up with a post-filter on
/// [`RetrievalResult::items`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TemporalFilter {
    /// Inclusive lower bound on `mnem:created_at`.
    pub created_after: Option<u64>,
    /// Exclusive upper bound on `mnem:created_at`.
    pub created_before: Option<u64>,
    /// Inclusive lower bound on `mnem:updated_at`.
    pub updated_after: Option<u64>,
    /// Exclusive upper bound on `mnem:updated_at`.
    pub updated_before: Option<u64>,
}

impl TemporalFilter {
    /// `true` if no bound is active.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.created_after.is_none()
            && self.created_before.is_none()
            && self.updated_after.is_none()
            && self.updated_before.is_none()
    }

    /// Test a single node against the filter.
    ///
    /// Nodes lacking the reserved `mnem:created_at` / `mnem:updated_at`
    /// prop pass the corresponding check (lenient-on-legacy rule; see
    /// the struct docs).
    #[must_use]
    pub fn matches(&self, node: &Node) -> bool {
        let created = node.props.get("mnem:created_at").and_then(|v| match v {
            Ipld::Integer(n) => u64::try_from(*n).ok(),
            _ => None,
        });
        let updated = node.props.get("mnem:updated_at").and_then(|v| match v {
            Ipld::Integer(n) => u64::try_from(*n).ok(),
            _ => None,
        });
        if let (Some(t), Some(c)) = (self.created_after, created)
            && c < t
        {
            return false;
        }
        if let (Some(t), Some(c)) = (self.created_before, created)
            && c >= t
        {
            return false;
        }
        if let (Some(t), Some(u)) = (self.updated_after, updated)
            && u < t
        {
            return false;
        }
        if let (Some(t), Some(u)) = (self.updated_before, updated)
            && u >= t
        {
            return false;
        }
        true
    }
}

/// Fusion strategy for combining the dense + sparse lanes.
///
/// Default is [`Self::ConvexMinMax`] per Bruch et al. 2023
/// ("An Analysis of Fusion Functions for Hybrid Retrieval",
/// arXiv:2210.11934), which shows min-max normalized convex
/// combination beats Reciprocal Rank Fusion both in-domain and out-of-
/// domain whenever *any* tuning data exists. [`Self::Rrf`] stays
/// available as the rank-only no-tuning fallback; it is also what
/// older / analyses assumed.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum FusionStrategy {
    /// Reciprocal Rank Fusion (Cormack / Clarke / Buettcher 2009,
    /// k=60). Rank-only; drops native score magnitudes. Historically
    /// mnem's default.
    Rrf,
    /// Min-max normalized convex combination. Each lane's scores are
    /// rescaled to `[0, 1]`; per-node contributions are summed with
    /// the caller-configured per-lane weights. Preserves the "strong
    /// match dominates" signal RRF averages away.
    #[default]
    ConvexMinMax,
}
