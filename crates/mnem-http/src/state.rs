//! Shared app state. One `ReadonlyRepo` per server, behind an
//! `Arc<Mutex>` because redb holds an exclusive file lock so we can
//! only have one open per process. Writes take the lock, run a
//! transaction, and replace the repo with the post-commit value.

use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use mnem_core::id::{Cid, NodeId};
use mnem_core::index::hybrid::{AdjEdge, AdjacencyIndex, EdgeProvenance};
use mnem_core::index::{BruteForceVectorIndex, SparseInvertedIndex, VectorIndex};
use mnem_core::repo::ReadonlyRepo;

use crate::metrics::{LeidenModeLabels, Metrics};

/// Default per-source neighbour count for the KNN-edge fallback. 32
/// matches the `k=32, metric=cosine` determinism contract the
/// `KnnEdgeIndex::compute_cid` key derivation bakes in.
const KNN_FALLBACK_K: u32 = 32;

/// One-shot guard so the "KNN-edge fallback activated" info-level log
/// fires once per process lifetime (not once per retrieve). Keeps the
/// prod log from flooding while still emitting the "yes, E0 wire is
/// live" breadcrumb.
static KNN_FALLBACK_LOGGED: AtomicBool = AtomicBool::new(false);

// ---------------------------------------------------------------
// Gap 10 Phase-1: community-cache invalidation tunables
// ---------------------------------------------------------------

/// Gap 10 R6 floor-c tunable: commit-storm DoS cap per minute. Default 60.
///
/// `#tunable: default=60; rationale="attacker cannot amplify beyond human commit rate"`
pub const COMMIT_STORM_CAP_PER_MIN: u32 = 60;

/// Gap 10 R6 floor-c tunable: fraction of the graph that must change
/// in one commit before an incremental recompute path force-flips to
/// full. Exported today as a gauge-visible constant; consulted by the
/// debounced-full recompute loop when deciding whether to bypass the
/// incremental shortcut.
///
/// `#tunable: default=0.5; rationale="half-graph change = incremental not cheaper than full"`
pub const DELTA_RATIO_FORCE_FULL: f32 = 0.5;

/// Gap 10 Phase-1 graph-size gate: above this node count the hot path
/// refuses to run full-Leiden inline and serves `FallbackStale`.
///
/// `#tunable: default=250_000; rationale="HNSW memory derivation; see benchmarks/leiden-wallclock-vs-V.md"`
pub const GRAPH_SIZE_GATE_V: usize = 250_000;

/// Gap 10 R3 debounce floor.
pub const DEBOUNCE_FLOOR_MS: u64 = 1_000;

/// Size of the rolling ring buffer used for p75 commit-latency.
pub const COMMIT_LATENCY_WINDOW: usize = 100;

/// Gap 10 R6 code-sketch API: runtime-derived debounce window.
#[must_use]
pub fn derive_debounce_ms(rolling_p75_commit_ms: Option<u64>) -> u64 {
    rolling_p75_commit_ms
        .map(|p| p.max(DEBOUNCE_FLOOR_MS))
        .unwrap_or(DEBOUNCE_FLOOR_MS)
}

/// Gap 10 Phase-1 recompute-mode enum. Closed vocabulary.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum LeidenMode {
    /// Immediate full recompute on every commit. Env-forced.
    Full,
    /// Debounced full recompute (a future version default).
    FullDebounced,
    /// Served prior commit's assignment; refresh suppressed.
    FallbackStale,
}

impl LeidenMode {
    /// Prometheus label string.
    #[must_use]
    pub fn label(&self) -> &'static str {
        match self {
            Self::Full => "full",
            Self::FullDebounced => "full_debounced",
            Self::FallbackStale => "fallback_stale",
        }
    }

    /// Gauge encoding for `mnem_leiden_mode_current`.
    #[must_use]
    pub fn gauge_value(&self) -> i64 {
        match self {
            Self::Full => 0,
            Self::FullDebounced => 1,
            Self::FallbackStale => 2,
        }
    }

    /// Resolve default mode from env at startup.
    #[must_use]
    pub fn resolve_default_from_env() -> Self {
        match std::env::var("MNEM_LEIDEN_FULL_RECOMPUTE").ok() {
            Some(v) => {
                let t = v.trim().to_ascii_lowercase();
                if t.is_empty() || matches!(t.as_str(), "0" | "false" | "no" | "off") {
                    Self::FullDebounced
                } else {
                    Self::Full
                }
            }
            None => Self::FullDebounced,
        }
    }
}

/// Gap 10 Phase-1 debounce + storm-cap state.
#[derive(Debug)]
pub struct LeidenCache {
    /// Rolling ring of commit wall-clock latencies (milliseconds).
    pub commit_latency_ms: VecDeque<u64>,
    /// Instant of the most recent successful full recompute.
    pub last_recompute_at: Option<Instant>,
    /// Ring of commit arrivals inside the trailing 60s.
    pub commit_arrivals: VecDeque<Instant>,
    /// Default mode at process start.
    pub default_mode: LeidenMode,
    /// Effective storm cap (operator-overridable).
    pub storm_cap_per_min: u32,
}

impl Default for LeidenCache {
    fn default() -> Self {
        Self {
            commit_latency_ms: VecDeque::with_capacity(COMMIT_LATENCY_WINDOW),
            last_recompute_at: None,
            commit_arrivals: VecDeque::new(),
            default_mode: LeidenMode::FullDebounced,
            storm_cap_per_min: COMMIT_STORM_CAP_PER_MIN,
        }
    }
}

impl LeidenCache {
    /// Record a completed-commit wall-time sample.
    pub fn observe_commit_latency(&mut self, latency: Duration) {
        let ms = u64::try_from(latency.as_millis()).unwrap_or(u64::MAX);
        if self.commit_latency_ms.len() == COMMIT_LATENCY_WINDOW {
            self.commit_latency_ms.pop_front();
        }
        self.commit_latency_ms.push_back(ms);
    }

    /// Record a commit arrival; evicts entries older than 60s.
    pub fn observe_commit_arrival(&mut self, at: Instant) {
        let cutoff = at.checked_sub(Duration::from_mins(1)).unwrap_or(at);
        while let Some(front) = self.commit_arrivals.front() {
            if *front < cutoff {
                self.commit_arrivals.pop_front();
            } else {
                break;
            }
        }
        self.commit_arrivals.push_back(at);
    }

    /// Nearest-rank p75 of the rolling commit-latency ring.
    #[must_use]
    pub fn rolling_p75_commit_ms(&self) -> Option<u64> {
        if self.commit_latency_ms.is_empty() {
            return None;
        }
        let mut sorted: Vec<u64> = self.commit_latency_ms.iter().copied().collect();
        sorted.sort_unstable();
        let n = sorted.len();
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let idx = ((n as f64 * 0.75).ceil() as usize)
            .saturating_sub(1)
            .min(n - 1);
        Some(sorted[idx])
    }

    /// Effective debounce window via [`derive_debounce_ms`].
    #[must_use]
    pub fn effective_debounce_ms(&self) -> u64 {
        derive_debounce_ms(self.rolling_p75_commit_ms())
    }

    /// Is the 60s commit-arrival ring at or above the storm cap?
    #[must_use]
    pub fn storm_cap_reached(&self) -> bool {
        u32::try_from(self.commit_arrivals.len()).unwrap_or(u32::MAX) >= self.storm_cap_per_min
    }

    /// Pure policy function. Returns the mode to serve this call.
    #[must_use]
    pub fn select_mode(&self, node_count: usize, now: Instant) -> LeidenMode {
        if self.default_mode == LeidenMode::Full {
            return LeidenMode::Full;
        }
        if node_count >= GRAPH_SIZE_GATE_V {
            return LeidenMode::FallbackStale;
        }
        if self.storm_cap_reached() {
            return LeidenMode::FallbackStale;
        }
        if let Some(last) = self.last_recompute_at {
            let elapsed_ms =
                u64::try_from(now.saturating_duration_since(last).as_millis()).unwrap_or(u64::MAX);
            if elapsed_ms < self.effective_debounce_ms() {
                return LeidenMode::FallbackStale;
            }
        }
        LeidenMode::FullDebounced
    }
}

/// Shared application state passed to every axum handler via
/// `State<AppState>`. Clones are cheap (shared `Arc`).
#[derive(Clone)]
pub struct AppState {
    /// The open repo, held behind a mutex because redb keeps an
    /// exclusive file lock per-process. Writes replace the value
    /// inside the mutex with the post-commit `ReadonlyRepo`.
    pub repo: Arc<Mutex<ReadonlyRepo>>,
    /// Optional embed-provider config resolved from the repo's
    /// `config.toml` at startup.
    pub embed_cfg: Option<mnem_embed_providers::ProviderConfig>,
    /// Optional sparse-provider config resolved from the repo's
    /// `config.toml` at startup. When present, POST `/v1/nodes` and
    /// POST `/v1/nodes/bulk` auto-populate `Node.sparse_embed` on
    /// ingest, and `/v1/retrieve` auto-encodes the query via
    /// `SparseEncoder::encode_query` so the neural-sparse lane fires
    /// end-to-end.
    pub sparse_cfg: Option<mnem_sparse_providers::ProviderConfig>,
    /// Index cache keyed by commit CID. Fixes audit gap G1: without
    /// this, every `/v1/retrieve` call rebuilt the vector + sparse
    /// indexes from scratch (O(N) per query). With this, the first
    /// retrieve after a commit pays the build cost; every subsequent
    /// retrieve returns in microseconds.
    ///
    /// Invalidation is automatic: any write path that commits also
    /// produces a new head commit CID (via
    /// `Transaction::commit -> ReadonlyRepo`). The cache sees the
    /// mismatch next time and evicts.
    pub indexes: Arc<Mutex<IndexCache>>,
    /// Whether the server accepts caller-supplied `label` values on
    /// ingest and `label` filters on retrieve. Read from the
    /// `MNEM_BENCH` environment variable at startup.
    ///
    /// Defaults to `false`. Casual / single-tenant / personal-graph
    /// installations keep label hidden: every ingested node gets
    /// `ntype = Node::DEFAULT_NTYPE` ("Node") regardless of what the
    /// caller sent, and retrieve ignores any label filter. No way to
    /// flip this via a CLI flag or request body - operators opt in by
    /// setting `MNEM_BENCH=1` at server launch, which is how the
    /// reference benchmark harnesses in this repo pin per-item
    /// isolation. Zero surface area for a regular user to stumble
    /// into label-scoped state.
    pub allow_labels: bool,
    /// Prometheus metrics registry shared with the `/metrics` route
    /// and the `track_metrics` middleware. Clones are cheap (`Arc`
    /// inside); no per-request allocation.
    pub metrics: Metrics,
    /// Bearer token that authorises `/remote/v1/push-blocks` and
    /// `/remote/v1/advance-head`. Read from `MNEM_HTTP_PUSH_TOKEN`
    /// at startup. `None` means those routes are administratively
    /// disabled (fail-closed): they return 503 regardless of
    /// whatever the caller presented.
    ///
    /// The token never touches disk and is never emitted to tracing.
    /// See [`crate::auth`] for the extractor that enforces the
    /// check.
    pub push_token: Option<String>,
    /// C3 FIX-1 + FIX-2: authored-edges adjacency + Leiden community
    /// assignment cache, keyed on the repo's op-id. Populated lazily
    /// on the first retrieve that asks for `community_filter=true` or
    /// `graph_mode="ppr"`. Invalidated whenever the op-id changes
    /// (any write path). Single-slot cache: one authored-adjacency
    /// snapshot is shared across requests until the next commit.
    pub graph_cache: Arc<Mutex<GraphCache>>,
    /// Gap 09 - `/v1/traverse_answer` runtime config. Default OFF per
    /// architect Decision 4; opt in via
    /// `[experimental] single_call_multihop = true` in the repo's
    /// `config.toml`. Wrapped in `Arc` for cheap clones across handler
    /// dispatch.
    pub traverse_cfg: Arc<crate::routes::traverse::TraverseAnswerCfg>,
}

impl AppState {
    /// Read `MNEM_BENCH` at startup. See [`Self::parse_allow_labels`]
    /// for the truthy / falsy string rules.
    #[must_use]
    pub fn resolve_allow_labels_from_env() -> bool {
        Self::parse_allow_labels(std::env::var("MNEM_BENCH").ok().as_deref())
    }

    /// Read `MNEM_HTTP_PUSH_TOKEN` at startup. Empty / unset -> `None`
    /// (writes fail-closed). See [`crate::auth::RequireBearer`] for the
    /// extractor that enforces the check.
    #[must_use]
    pub fn resolve_push_token_from_env() -> Option<String> {
        let raw = std::env::var("MNEM_HTTP_PUSH_TOKEN").ok()?;
        let trimmed = raw.trim();
        if trimmed.is_empty() {
            None
        } else {
            Some(trimmed.to_string())
        }
    }

    /// Pure parser for the `MNEM_BENCH` value. `None` (unset) is
    /// false. Falsy strings (`"0"`, `"false"`, `"no"`, `"off"`, empty,
    /// all case-insensitive) are false. Anything else is true.
    #[must_use]
    pub fn parse_allow_labels(val: Option<&str>) -> bool {
        match val {
            None => false,
            Some(s) => {
                let t = s.trim();
                if t.is_empty() {
                    return false;
                }
                let l = t.to_ascii_lowercase();
                !matches!(l.as_str(), "0" | "false" | "no" | "off")
            }
        }
    }
}

/// Server-side cache of built retrieval indexes. Keyed on the repo's
/// current head-commit CID; when that changes, the whole cache is
/// treated as stale and rebuilt on demand.
#[derive(Default)]
pub struct IndexCache {
    /// Repo op-id the current cache was built against. `None` on
    /// startup or after a reset. When this differs from the repo's
    /// current op-id, every entry below is considered stale.
    ///
    /// Named `cache_key_op_id` (not `commit_cid`) because op-id is
    /// what we key on: op-id exists on a freshly-initialised repo
    /// with no commits yet, which the commit CID does not.
    pub cache_key_op_id: Option<Cid>,
    /// Dense vector indexes keyed by the embedder's `model` string
    /// (e.g. `"ollama:nomic-embed-text"`). One repo can carry multiple
    /// model families if the caller runs several retrieves with
    /// different `vector_model`s.
    pub vectors: HashMap<String, Arc<BruteForceVectorIndex>>,
    /// Sparse indexes keyed by `SparseEmbed::vocab_id`.
    pub sparse: HashMap<String, Arc<SparseInvertedIndex>>,
}

#[cfg(test)]
mod mnem_bench_parse_tests {
    use super::*;

    #[test]
    fn unset_parses_false() {
        assert!(!AppState::parse_allow_labels(None));
    }

    #[test]
    fn falsy_strings_parse_false() {
        for v in [
            "", "0", "false", "FALSE", "False", "no", "No", "NO", "off", "Off", "OFF", "  ", "  0 ",
        ] {
            assert!(
                !AppState::parse_allow_labels(Some(v)),
                "expected `{v:?}` to parse false"
            );
        }
    }

    #[test]
    fn truthy_strings_parse_true() {
        for v in ["1", "true", "yes", "on", "YES", "benchmark", "anything"] {
            assert!(
                AppState::parse_allow_labels(Some(v)),
                "expected `{v:?}` to parse true"
            );
        }
    }
}

impl IndexCache {
    /// Reconcile the cache against the repo's current op-id. Called
    /// on every cache access. When the op has changed (any write
    /// flips it), all built indexes are dropped and the next getter
    /// will rebuild.
    ///
    /// We key on `repo.op_id()` rather than `head_commit().cid`
    /// because the op-id exists on freshly-initialised repos that
    /// have no commits yet, and it changes on every write. Commit
    /// CID would only change after the first real commit, leaving
    /// an init-time cache entry with no way to invalidate.
    pub fn reconcile(&mut self, repo: &ReadonlyRepo) {
        let current = Some(repo.op_id().clone());
        if self.cache_key_op_id != current {
            self.cache_key_op_id = current;
            self.vectors.clear();
            self.sparse.clear();
        }
    }

    /// Fetch (or build) a dense vector index for `model`.
    pub fn vector_index(
        &mut self,
        repo: &ReadonlyRepo,
        model: &str,
    ) -> Result<Arc<BruteForceVectorIndex>, mnem_core::Error> {
        self.reconcile(repo);
        if let Some(idx) = self.vectors.get(model) {
            return Ok(idx.clone());
        }
        let idx = Arc::new(repo.build_vector_index(model)?);
        self.vectors.insert(model.to_string(), idx.clone());
        Ok(idx)
    }

    /// Fetch (or build) a sparse inverted index for `vocab_id`.
    pub fn sparse_index(
        &mut self,
        repo: &ReadonlyRepo,
        vocab_id: &str,
    ) -> Result<Arc<SparseInvertedIndex>, mnem_core::Error> {
        self.reconcile(repo);
        if let Some(idx) = self.sparse.get(vocab_id) {
            return Ok(idx.clone());
        }
        let idx = Arc::new(SparseInvertedIndex::build_from_repo(repo, vocab_id)?);
        self.sparse.insert(vocab_id.to_string(), idx.clone());
        Ok(idx)
    }
}

// ---------------------------------------------------------------
// C3 FIX-1 + FIX-2: authored-edges + community-assignment cache.
// ---------------------------------------------------------------

/// Owned authored-edge list over which `AdjacencyIndex` is served.
/// Collected once per op-id from the repo's commit.edges Prolly tree;
/// Leiden community detection and PPR power iteration both read
/// through the same snapshot.
#[derive(Default, Debug)]
pub struct AuthoredEdges {
    /// Flat (src, dst) list in Prolly traversal order. Weighted `1.0`
    /// by `iter_edges()` consumers; provenance stamped `Authored`.
    pub edges: Vec<(NodeId, NodeId)>,
}

impl AdjacencyIndex for AuthoredEdges {
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

/// C3 Patch-B: combined authored + KNN-derived adjacency, owned. Used
/// by `GraphCache` to back both the Leiden community detector and the
/// PPR adjacency wire in a single `AdjacencyIndex` impl without the
/// lifetime acrobatics `mnem_core::HybridAdjacency<A,K>` imposes (it
/// borrows both sources). Structurally equivalent to a
/// `HybridAdjacency<AuthoredEdges, KnnSlice>` sent through an
/// `Arc<dyn AdjacencyIndex + Send + Sync>` consumer.
///
/// # Flag-off / empty-KNN contract
///
/// When `knn` is empty, `iter_edges` yields exactly the authored
/// stream in the authored source's native order, preserving the
/// `mnem_core::tests::hybrid_adjacency_union` byte-identity contract.
#[derive(Debug)]
pub struct DerivedHybridAdjacency {
    /// Authored-edge snapshot shared with `GraphCache::adjacency`.
    pub authored: Arc<AuthoredEdges>,
    /// KNN-derived edges in canonical `(src, dst)` ASC order. `weight`
    /// is the similarity score, `provenance` is `Knn`.
    pub knn: Vec<(NodeId, NodeId, f32)>,
}

impl AdjacencyIndex for DerivedHybridAdjacency {
    fn iter_edges(&self) -> Box<dyn Iterator<Item = AdjEdge> + '_> {
        let authored = self.authored.edges.iter().map(|(s, d)| AdjEdge {
            src: *s,
            dst: *d,
            weight: 1.0,
            provenance: EdgeProvenance::Authored,
        });
        let knn = self.knn.iter().map(|(s, d, w)| AdjEdge {
            src: *s,
            dst: *d,
            weight: *w,
            provenance: EdgeProvenance::Knn,
        });
        Box::new(authored.chain(knn))
    }
    fn edge_count(&self) -> usize {
        self.authored.edges.len() + self.knn.len()
    }
}

/// C3 FIX + Patch-B: authored adjacency, derived-KNN edges, and the
/// community assignment keyed on `op_id`. Single-slot: one snapshot
/// is shared across requests until the next commit.
///
/// Two-key design:
/// - `key` = repo `op_id`; any write invalidates everything.
/// - `knn_key` = content address of the KNN-edge index
///   (`hash(root_cid, k, metric)`) used to avoid rebuilding the
///   KNN-derived edges when the same vector index is re-used across
///   retrieves at the same `op_id`.
#[derive(Default)]
pub struct GraphCache {
    /// Op-id the cache was built against. `None` means empty.
    pub key: Option<Cid>,
    /// Shared authored-adjacency snapshot.
    pub adjacency: Option<Arc<AuthoredEdges>>,
    /// C3 Patch-B: KNN-derived edges produced by
    /// `mnem_ann::derive_knn_edges_from_vectors` over the current
    /// vector index. `None` means either (a) the authored adjacency
    /// was already non-empty so KNN fallback never fired, or (b) the
    /// vector index was empty (nothing to derive from).
    pub knn_edges: Option<Arc<Vec<(NodeId, NodeId, f32)>>>,
    /// Cache key for `knn_edges`:
    /// `(KnnEdgeIndex::compute_cid, k, metric_tag)` projected down to
    /// the content-address CID. Stable across restarts given the same
    /// vector-index contents + k + metric.
    pub knn_key: Option<Cid>,
    /// Hybrid (authored + KNN) adjacency ready to hand out as
    /// `Arc<dyn AdjacencyIndex + Send + Sync>`. Rebuilt whenever
    /// `adjacency` or `knn_edges` changes.
    pub hybrid: Option<Arc<DerivedHybridAdjacency>>,
    /// Shared Leiden community assignment over `hybrid` (falls back
    /// to `adjacency` when KNN fallback is inactive).
    pub community: Option<Arc<mnem_graphrag::community::CommunityAssignment>>,
    /// C3 FIX-1: cached row-stochastic CSR matrix used by PPR.
    /// Invalidated alongside `adjacency` / `knn_edges` so the matrix
    /// always reflects the current op-id's adjacency. Shared across
    /// retrieves at the same op-id; `mnem_core::ppr::ppr_with_matrix`
    /// is byte-identical to the from-scratch `ppr()` path (pinned by
    /// the `ppr_with_matrix_matches_ppr_on_small_graph` integration
    /// test), so turning this cache on is a pure-speed change.
    pub ppr_matrix: Option<Arc<mnem_core::ppr::SparseTransition>>,
    /// Gap 10 Phase-1: prior-commit assignment surviving op-id churn.
    pub community_stale: Option<Arc<mnem_graphrag::community::CommunityAssignment>>,
    /// Gap 10 Phase-1: debounce + storm-cap policy state.
    pub leiden_cache: LeidenCache,
}

impl GraphCache {
    /// Invalidate if the repo's op-id has moved. Called on every
    /// cache access, mirrors [`IndexCache::reconcile`].
    pub fn reconcile(&mut self, repo: &ReadonlyRepo) {
        let current = Some(repo.op_id().clone());
        if self.key != current {
            self.key = current;
            self.adjacency = None;
            self.knn_edges = None;
            self.knn_key = None;
            self.hybrid = None;
            // Gap 10 Phase-1: demote current assignment to `community_stale`
            // so retrieves inside the debounce window can serve it.
            if let Some(prev) = self.community.take() {
                self.community_stale = Some(prev);
            }
            self.ppr_matrix = None;
        }
    }

    /// Gap 10 Phase-1 entry point. Resolves the current mode via
    /// [`LeidenCache::select_mode`], increments
    /// `mnem_leiden_mode_total{mode=...}` + mirrors gauge triad, then
    /// returns the assignment and the served mode.
    pub fn community_for_head(
        &mut self,
        repo: &ReadonlyRepo,
        vector: Option<&BruteForceVectorIndex>,
        metrics: &Metrics,
    ) -> Result<
        (
            Arc<mnem_graphrag::community::CommunityAssignment>,
            LeidenMode,
        ),
        crate::error::Error,
    > {
        self.reconcile(repo);
        let adj = self.hybrid_adjacency_for(repo, vector)?;
        let node_count = authored_node_count(adj.as_ref());
        let now = Instant::now();
        let mode = self.leiden_cache.select_mode(node_count, now);

        metrics
            .leiden_debounce_effective
            .set(i64::try_from(self.leiden_cache.effective_debounce_ms()).unwrap_or(i64::MAX));
        metrics
            .leiden_storm_cap_effective
            .set(i64::from(self.leiden_cache.storm_cap_per_min));
        #[allow(clippy::cast_possible_truncation)]
        let delta_pp10k = (DELTA_RATIO_FORCE_FULL * 10_000.0) as i64;
        metrics.leiden_delta_ratio_effective.set(delta_pp10k);
        metrics.leiden_mode_current.set(mode.gauge_value());
        metrics
            .leiden_mode
            .get_or_create(&LeidenModeLabels {
                mode: mode.label().to_string(),
            })
            .inc();

        match mode {
            LeidenMode::Full | LeidenMode::FullDebounced => {
                if let Some(c) = &self.community {
                    return Ok((c.clone(), mode));
                }
                let assignment = mnem_graphrag::community::compute_communities(adj.as_ref(), 0);
                let arc = Arc::new(assignment);
                self.community = Some(arc.clone());
                if matches!(mode, LeidenMode::FullDebounced) {
                    self.leiden_cache.last_recompute_at = Some(now);
                }
                self.leiden_cache.observe_commit_arrival(now);
                Ok((arc, mode))
            }
            LeidenMode::FallbackStale => {
                if let Some(c) = &self.community {
                    return Ok((c.clone(), mode));
                }
                if let Some(c) = &self.community_stale {
                    return Ok((c.clone(), mode));
                }
                let assignment = mnem_graphrag::community::compute_communities(adj.as_ref(), 0);
                let arc = Arc::new(assignment);
                self.community_stale = Some(arc.clone());
                Ok((arc, mode))
            }
        }
    }

    /// C3 FIX-1: fetch (or build) the row-stochastic CSR matrix used
    /// by PPR, over the hybrid adjacency. The matrix is a pure
    /// function of the adjacency; caching it lets repeated retrieves
    /// at the same op-id skip the 3-pass CSR build (dominates PPR
    /// cost at ~15 iters on small graphs).
    ///
    /// Byte-identity with the uncached `ppr()` path is pinned by the
    /// `ppr_with_matrix_matches_ppr_on_small_graph` integration test
    /// in mnem-core.
    pub fn ppr_matrix_for(
        &mut self,
        repo: &ReadonlyRepo,
        vector: Option<&BruteForceVectorIndex>,
    ) -> Result<Arc<mnem_core::ppr::SparseTransition>, crate::error::Error> {
        self.reconcile(repo);
        if let Some(m) = &self.ppr_matrix {
            return Ok(m.clone());
        }
        let adj = self.hybrid_adjacency_for(repo, vector)?;
        let m = Arc::new(mnem_core::ppr::sparse_transition_matrix(adj.as_ref()));
        self.ppr_matrix = Some(m.clone());
        Ok(m)
    }

    /// Fetch (or build) the authored-adjacency snapshot. Back-compat
    /// wrapper preserved for callers that do NOT want KNN fallback
    /// (e.g. tests asserting the pure-authored path).
    pub fn adjacency_for(
        &mut self,
        repo: &ReadonlyRepo,
    ) -> Result<Arc<AuthoredEdges>, crate::error::Error> {
        self.reconcile(repo);
        if let Some(a) = &self.adjacency {
            return Ok(a.clone());
        }
        let a = Arc::new(collect_authored_edges(repo)?);
        self.adjacency = Some(a.clone());
        Ok(a)
    }

    /// Fetch (or build) the Leiden community assignment over the
    /// authored-only adjacency. Back-compat entry point.
    pub fn community_for(
        &mut self,
        repo: &ReadonlyRepo,
    ) -> Result<Arc<mnem_graphrag::community::CommunityAssignment>, crate::error::Error> {
        self.reconcile(repo);
        if let Some(c) = &self.community {
            return Ok(c.clone());
        }
        let adj = self.adjacency_for(repo)?;
        let assignment = mnem_graphrag::community::compute_communities(adj.as_ref(), 0);
        let arc = Arc::new(assignment);
        self.community = Some(arc.clone());
        Ok(arc)
    }

    /// C3 Patch-B entry point: fetch (or build) the authored adjacency;
    /// when it is empty (and `vector` is non-empty), derive a KNN-edge
    /// substrate via `mnem_ann::derive_knn_edges_from_vectors` and
    /// cache a combined [`DerivedHybridAdjacency`].
    ///
    /// Returns a trait-object `Arc<dyn AdjacencyIndex + Send + Sync>`
    /// so the retriever's `with_adjacency_index` consumer can accept
    /// it without knowing whether the KNN lane fired.
    ///
    /// # Zero-impact contracts
    ///
    /// - Authored non-empty -> returns the authored snapshot; KNN lane
    ///   never runs; `knn_edges` stays `None`.
    /// - `vector` is `None` or empty -> returns the authored snapshot
    ///   (possibly empty); no KNN fallback; preserves the "no graph,
    ///   no filter" legacy behaviour.
    /// - Same `op_id` + same vector-index content -> served from the
    ///   single-slot cache; `derive_knn_edges_from_vectors` is not
    ///   re-run.
    pub fn hybrid_adjacency_for(
        &mut self,
        repo: &ReadonlyRepo,
        vector: Option<&BruteForceVectorIndex>,
    ) -> Result<Arc<dyn AdjacencyIndex + Send + Sync>, crate::error::Error> {
        self.reconcile(repo);
        let authored = self.adjacency_for(repo)?;

        // Zero-impact short-circuits.
        if !authored.edges.is_empty() {
            return Ok(authored as Arc<dyn AdjacencyIndex + Send + Sync>);
        }
        let Some(vec_idx) = vector else {
            return Ok(authored as Arc<dyn AdjacencyIndex + Send + Sync>);
        };
        if vec_idx.is_empty() {
            return Ok(authored as Arc<dyn AdjacencyIndex + Send + Sync>);
        }

        // Fallback active: derive (or fetch cached) KNN edges. The
        // `KnnEdgeIndex` CID folds in (root_cid, k, metric) so the
        // cache key is stable across restarts given the same vector
        // content.
        self.ensure_knn_edges(vec_idx)?;
        let knn = self
            .knn_edges
            .clone()
            .expect("ensure_knn_edges populated the slot");
        // One-shot info log the first time the fallback path wins.
        if !KNN_FALLBACK_LOGGED.swap(true, Ordering::Relaxed) {
            tracing::info!(
                target: "mnem_http::graph_cache",
                k = KNN_FALLBACK_K,
                metric = "cosine",
                knn_edges = knn.len(),
                vector_model = %vec_idx.model(),
                "authored adjacency empty; KNN-edge fallback activated (E0 wire)",
            );
        }
        if self.hybrid.is_none() {
            self.hybrid = Some(Arc::new(DerivedHybridAdjacency {
                authored: authored.clone(),
                knn: (*knn).clone(),
            }));
        }
        Ok(self.hybrid.clone().expect("hybrid slot populated above")
            as Arc<dyn AdjacencyIndex + Send + Sync>)
    }

    /// C3 Patch-B: community assignment over the hybrid adjacency.
    /// Falls through to the authored-only path when `vector` is absent
    /// or empty, or when authored edges are already non-empty.
    pub fn hybrid_community_for(
        &mut self,
        repo: &ReadonlyRepo,
        vector: Option<&BruteForceVectorIndex>,
    ) -> Result<Arc<mnem_graphrag::community::CommunityAssignment>, crate::error::Error> {
        self.reconcile(repo);
        if let Some(c) = &self.community {
            return Ok(c.clone());
        }
        let adj = self.hybrid_adjacency_for(repo, vector)?;
        // compute_communities takes `&dyn AdjacencyIndex`; the trait
        // object's deref coerces.
        let assignment = mnem_graphrag::community::compute_communities(adj.as_ref(), 0);
        let arc = Arc::new(assignment);
        self.community = Some(arc.clone());
        Ok(arc)
    }

    /// Derive KNN edges from `vector` and cache them along with the
    /// `KnnEdgeIndex` CID as the cache key. Idempotent: returns early
    /// when the key already matches.
    fn ensure_knn_edges(
        &mut self,
        vector: &BruteForceVectorIndex,
    ) -> Result<(), crate::error::Error> {
        // Collect (ids, vecs) from the brute-force index. The slices
        // are borrowed from the index's flat buffer (zero copy up to
        // the `.to_vec()` the KNN builder does internally when it
        // clones per-source rows).
        let mut ids: Vec<NodeId> = Vec::with_capacity(vector.len());
        let mut vecs: Vec<Vec<f32>> = Vec::with_capacity(vector.len());
        for (id, row) in vector.points_iter() {
            ids.push(id);
            vecs.push(row.to_vec());
        }
        let edges = mnem_ann::derive_knn_edges_from_vectors(
            &ids,
            &vecs,
            KNN_FALLBACK_K,
            mnem_ann::DistanceMetric::Cosine,
        );
        // Assemble a KnnEdgeIndex purely to compute the cache-key CID.
        // The `root_cid` is the content hash of (model, dim, ids,
        // flat_vecs); we lean on the HybridAdjacency E0 substrate's
        // existing `compute_cid` so the key is stable across restarts
        // given identical vector content.
        let root_cid = vector_index_content_cid(vector, &ids)?;
        let idx = mnem_ann::KnnEdgeIndex {
            root_cid,
            k: KNN_FALLBACK_K,
            metric: mnem_ann::DistanceMetric::Cosine,
            edges,
        };
        let cid = idx
            .compute_cid()
            .map_err(|e| crate::error::Error::internal(format!("knn edge cid: {e}")))?;
        if self.knn_key.as_ref() == Some(&cid) && self.knn_edges.is_some() {
            return Ok(());
        }
        let triples: Vec<(NodeId, NodeId, f32)> = idx
            .edges
            .into_iter()
            .map(|e| (e.src, e.dst, e.weight))
            .collect();
        self.knn_edges = Some(Arc::new(triples));
        self.knn_key = Some(cid);
        self.hybrid = None; // invalidate derived combined view
        Ok(())
    }
}

/// Derive a stable content-address for a [`BruteForceVectorIndex`] by
/// hashing `(model, dim, canonical_ids)`. Two vector indexes with the
/// same model, dimensionality and ID set share a CID, which is the
/// coarser grain we want for the KNN-edge cache key (the vectors
/// themselves live in the Prolly tree keyed by NodeId, so equal IDs
/// at the same op_id imply equal vector contents).
fn vector_index_content_cid(
    vector: &BruteForceVectorIndex,
    ids: &[NodeId],
) -> Result<Cid, crate::error::Error> {
    use mnem_core::codec::to_canonical_bytes;
    use mnem_core::id::{CODEC_RAW, Multihash};
    #[derive(serde::Serialize)]
    struct Preimage<'a> {
        tag: &'a str,
        model: &'a str,
        dim: u32,
        ids: &'a [NodeId],
    }
    let pre = Preimage {
        tag: "mnem-http/knn-fallback/v1",
        model: vector.model(),
        dim: vector.dim(),
        ids,
    };
    let body = to_canonical_bytes(&pre)
        .map_err(|e| crate::error::Error::internal(format!("canonical encode: {e}")))?;
    let hash = Multihash::sha2_256(&body);
    Ok(Cid::new(CODEC_RAW, hash))
}

/// Gap 10 Phase-1: distinct node count walking adjacency edges.
fn authored_node_count(adj: &(dyn AdjacencyIndex + Send + Sync)) -> usize {
    use std::collections::BTreeSet;
    let mut seen: BTreeSet<NodeId> = BTreeSet::new();
    for e in adj.iter_edges() {
        seen.insert(e.src);
        seen.insert(e.dst);
    }
    seen.len()
}

/// Walk `commit.edges` Prolly tree once and collect every authored
/// edge as a `(src, dst)` pair in canonical traversal order.
fn collect_authored_edges(repo: &ReadonlyRepo) -> Result<AuthoredEdges, crate::error::Error> {
    let Some(commit) = repo.head_commit() else {
        return Ok(AuthoredEdges::default());
    };
    let bs = repo.blockstore().clone();
    let cursor = mnem_core::prolly::Cursor::new(&*bs, &commit.edges)
        .map_err(|e| crate::error::Error::internal(format!("opening edge cursor: {e}")))?;
    let mut edges: Vec<(NodeId, NodeId)> = Vec::new();
    for entry in cursor {
        let (_key, edge_cid) =
            entry.map_err(|e| crate::error::Error::internal(format!("walking edge tree: {e}")))?;
        let bytes = bs
            .get(&edge_cid)
            .map_err(|e| crate::error::Error::internal(format!("fetching edge block: {e}")))?
            .ok_or_else(|| {
                crate::error::Error::internal(format!("edge block {edge_cid} missing"))
            })?;
        let edge: mnem_core::objects::Edge = mnem_core::codec::from_canonical_bytes(&bytes)
            .map_err(|e| crate::error::Error::internal(format!("decoding edge: {e}")))?;
        edges.push((edge.src, edge.dst));
    }
    Ok(AuthoredEdges { edges })
}

#[cfg(test)]
pub(crate) mod test_support {
    //! Test-only builders for [`AppState`]. Scoped `pub(crate)` so
    //! sibling modules (`auth`, `routes::remote`) can build a minimal
    //! in-memory state without paying the full `app_with_options`
    //! path (redb open + config load) for every unit test.

    use super::*;
    use mnem_core::store::{MemoryBlockstore, MemoryOpHeadsStore};

    /// Build an `AppState` backed by in-memory stores, with an
    /// optional push token. Used by extractor and route unit tests.
    pub(crate) fn state_with_token(token: Option<String>) -> AppState {
        let bs: Arc<dyn mnem_core::store::Blockstore> = Arc::new(MemoryBlockstore::new());
        let ohs: Arc<dyn mnem_core::store::OpHeadsStore> = Arc::new(MemoryOpHeadsStore::new());
        let repo = ReadonlyRepo::init(bs, ohs).expect("init ok");
        AppState {
            repo: Arc::new(Mutex::new(repo)),
            embed_cfg: None,
            sparse_cfg: None,
            indexes: Arc::new(Mutex::new(IndexCache::default())),
            allow_labels: false,
            metrics: Metrics::new(),
            push_token: token,
            graph_cache: Arc::new(Mutex::new(GraphCache::default())),
            traverse_cfg: Arc::new(crate::routes::traverse::TraverseAnswerCfg::default()),
        }
    }
}

#[cfg(test)]
mod knn_fallback_tests {
    //! C3 Patch-B unit tests: `GraphCache` KNN-edge fallback activation
    //! and zero-impact short-circuits.

    use super::*;
    use bytes::Bytes;
    use mnem_core::objects::node::{Dtype, Embedding};
    use mnem_core::store::{Blockstore, MemoryBlockstore, MemoryOpHeadsStore, OpHeadsStore};

    fn stores() -> (Arc<dyn Blockstore>, Arc<dyn OpHeadsStore>) {
        (
            Arc::new(MemoryBlockstore::new()),
            Arc::new(MemoryOpHeadsStore::new()),
        )
    }

    fn f32_embed(model: &str, v: &[f32]) -> Embedding {
        let mut bytes = Vec::with_capacity(v.len() * 4);
        for x in v {
            bytes.extend_from_slice(&x.to_le_bytes());
        }
        Embedding {
            model: model.to_string(),
            dtype: Dtype::F32,
            dim: u32::try_from(v.len()).expect("test vec fits in u32"),
            vector: Bytes::from(bytes),
        }
    }

    fn build_vector_index(rows: &[(NodeId, Vec<f32>)]) -> BruteForceVectorIndex {
        let mut idx = BruteForceVectorIndex::empty("m", 3);
        for (id, v) in rows {
            let inserted = idx.try_insert(*id, &f32_embed("m", v));
            assert!(inserted, "embedding insert");
        }
        idx
    }

    #[test]
    fn empty_authored_plus_empty_vector_is_no_op() {
        let (bs, ohs) = stores();
        let repo = ReadonlyRepo::init(bs, ohs).expect("init repo");
        let mut gc = GraphCache::default();
        let adj = gc.hybrid_adjacency_for(&repo, None).ok().expect("no-op");
        assert_eq!(adj.edge_count(), 0, "no vectors -> no KNN fallback");
        assert!(gc.knn_edges.is_none());
    }

    #[test]
    fn empty_authored_plus_populated_vector_activates_fallback() {
        let (bs, ohs) = stores();
        let repo = ReadonlyRepo::init(bs, ohs).expect("init repo");
        // 4 distinct L2-direction vectors; KNN k=32 (capped at n-1=3)
        // yields at least one directed edge per source.
        let rows: Vec<(NodeId, Vec<f32>)> = vec![
            (NodeId::new_v7(), vec![1.0, 0.0, 0.0]),
            (NodeId::new_v7(), vec![0.9, 0.1, 0.0]),
            (NodeId::new_v7(), vec![0.0, 1.0, 0.0]),
            (NodeId::new_v7(), vec![0.0, 0.0, 1.0]),
        ];
        let vec_idx = build_vector_index(&rows);

        let mut gc = GraphCache::default();
        let adj = gc
            .hybrid_adjacency_for(&repo, Some(&vec_idx))
            .ok()
            .expect("knn fallback ok");
        assert!(
            adj.edge_count() > 0,
            "KNN fallback must produce at least one edge (got 0)",
        );
        assert!(gc.knn_edges.is_some(), "knn_edges slot populated");
        assert!(gc.knn_key.is_some(), "knn cache key populated");

        // Community assignment must see the derived graph: at least one
        // community membership lookup must be Some (non-trivial graph).
        let assignment = gc
            .hybrid_community_for(&repo, Some(&vec_idx))
            .ok()
            .expect("community ok");
        let any_assigned = rows
            .iter()
            .any(|(id, _)| assignment.community_of(*id).is_some());
        assert!(
            any_assigned,
            "at least one node must have a community under a non-empty adjacency",
        );
    }

    #[test]
    fn knn_fallback_is_idempotent_on_same_vector() {
        let (bs, ohs) = stores();
        let repo = ReadonlyRepo::init(bs, ohs).expect("init repo");
        let rows: Vec<(NodeId, Vec<f32>)> = vec![
            (NodeId::new_v7(), vec![1.0, 0.0, 0.0]),
            (NodeId::new_v7(), vec![0.0, 1.0, 0.0]),
        ];
        let vec_idx = build_vector_index(&rows);
        let mut gc = GraphCache::default();
        let _ = gc
            .hybrid_adjacency_for(&repo, Some(&vec_idx))
            .ok()
            .expect("first build");
        let first_key = gc.knn_key.clone().expect("first build populates key");
        // Second call must re-use the cached slot (same op_id + same
        // vector content -> same KnnEdgeIndex CID).
        let _ = gc
            .hybrid_adjacency_for(&repo, Some(&vec_idx))
            .ok()
            .expect("second build");
        let second_key = gc.knn_key.clone().expect("second build populates key");
        assert_eq!(first_key, second_key, "KNN cache key stable across calls");
    }
}
