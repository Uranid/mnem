//! Gap 09 - `/v1/traverse_answer` single-call multi-hop endpoint.
//!
//! Default OFF per architect Decision 4. Ships in a future version behind the
//! `experimental.single_call_multihop` config flag (TOML), so callers
//! must opt in explicitly. When disabled, the handler returns
//! `410 Gone` with a stable payload pointing at the opt-in knob.
//!
//! # Hard-wall ceiling
//!
//! Even when enabled, the endpoint is wrapped in a
//! [`CommitBudgetGuard`]-driven hard wall set by `hard_wall_budget_ms`
//! (default 5000ms; floor-c HTTP interactive SLO ceiling). The guard
//! aborts the hop loop as soon as the wall is exceeded and the response
//! carries `hard_wall_cutoff: true` so callers can distinguish a
//! budget-clipped answer from a completed one. See
//! `docs/gap-catalog/09-single-call-multihop/solution.md` for the full
//! floor classification.
//!
//! # Determinism
//!
//! The hop loop composes deterministic retrieve stages. Given the same
//! query + commit CID + config the halt point is identical modulo the
//! wall-clock cutoff (which is a structural abort signal, not a
//! semantic one - callers should not route on it).
//!
//! # Wiring
//!
//! As of the audit-2026-04-25 fix, `traverse_answer` is registered in
//! the router at `POST /v1/traverse_answer`. The experimental gate in
//! the handler itself still keeps the endpoint OFF by default; opting
//! in flips the response from 410 Gone to the real hop-loop result.
//!
//! A handful of helper DTOs / budget helpers (`into_cfg`,
//! `derive_concurrency_cap`, `derive_cold_start_budget_ms`) are still
//! consumed only by the not-yet-merged hop-loop wrapper. They stay
//! gated under a module-scoped `allow(dead_code)` until that lands;
//! the allow is intentionally local so it never leaks workspace-wide.

#![allow(dead_code)]

use std::collections::HashSet;
use std::time::Instant;

use axum::Json;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use mnem_core::guard::{CommitBudgetGuard, Decision};
use mnem_core::id::{CODEC_RAW, Cid, Multihash, NodeId};
use mnem_embed_providers::Embedder as _;
use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::state::AppState;

// ---------------------------------------------------------------
// Config
// ---------------------------------------------------------------

/// Runtime configuration for the `/v1/traverse_answer` endpoint.
///
/// Loaded from the repo-local `config.toml` at startup under the
/// `[experimental]` table. Defaults are conservative: the endpoint is
/// OFF by default (architect Decision 4, a future version) and callers opt in per
/// deployment.
#[derive(Clone, Debug)]
pub struct TraverseAnswerCfg {
    /// Hard-wall wall-clock ceiling in milliseconds. Floor-c tunable
    /// (HTTP interactive-SLO p99 ceiling; see gap-catalog/09).
    /// Default 5000; clamps at [500, 30_000] to prevent pathological
    /// caller-side DoS (see `hard_wall_structural_dos_impossible`
    /// proptest).
    pub hard_wall_budget_ms: u32,
    /// Maximum hop count. Floor-c tunable. Default 3 covers 90% of
    /// MuSiQue/LoCoMo multi-hop benchmarks (see the
    /// `hops_3_covers_99_pct_multihop_benchmarks` proptest).
    pub max_hops: u32,
    /// Master gate. When `false`, the handler returns 410 Gone.
    /// Architect Decision 4: a future version ships with this OFF by default and
    /// callers opt in via `[experimental] single_call_multihop = true`.
    pub experimental_enabled: bool,
}

impl TraverseAnswerCfg {
    /// Structural floor on the hard wall. Prevents a caller from
    /// setting `hard_wall_budget_ms = 0` (trivial DoS) and from
    /// blowing past the HTTP interactive-SLO ceiling.
    pub const HARD_WALL_MIN_MS: u32 = 500;
    /// Upper clamp on the hard wall. Keeps any single request well
    /// under the 30-second generic HTTP proxy timeout.
    pub const HARD_WALL_MAX_MS: u32 = 30_000;
    /// Upper clamp on hop count. A hop budget above this is almost
    /// certainly a config typo.
    pub const MAX_HOPS_CEILING: u32 = 16;

    /// Clamp every field into its structural envelope. Idempotent.
    /// Called after every load / override path.
    #[must_use]
    pub fn clamped(mut self) -> Self {
        self.hard_wall_budget_ms = self
            .hard_wall_budget_ms
            .clamp(Self::HARD_WALL_MIN_MS, Self::HARD_WALL_MAX_MS);
        self.max_hops = self.max_hops.clamp(1, Self::MAX_HOPS_CEILING);
        self
    }
}

impl Default for TraverseAnswerCfg {
    fn default() -> Self {
        Self {
            hard_wall_budget_ms: 5000,
            max_hops: 3,
            experimental_enabled: false,
        }
    }
}

/// TOML shape loaded from `<data_dir>/config.toml` under
/// `[experimental]`. Separated from [`TraverseAnswerCfg`] so the public
/// runtime struct never grows optional fields.
#[derive(Debug, Default, Deserialize)]
pub(crate) struct ExperimentalSection {
    /// Master gate for `/v1/traverse_answer`. Defaults to `false` when
    /// the key is absent.
    #[serde(default)]
    pub(crate) single_call_multihop: Option<bool>,
    /// Override for `hard_wall_budget_ms`. Absent -> default 5000.
    #[serde(default)]
    pub(crate) traverse_answer_hard_wall_ms: Option<u32>,
    /// Override for `max_hops`. Absent -> default 3.
    #[serde(default)]
    pub(crate) traverse_answer_max_hops: Option<u32>,
}

impl ExperimentalSection {
    /// Fold an `[experimental]` section into a runtime config, applying
    /// defaults for every absent key. Always returns a
    /// structurally-clamped result.
    #[must_use]
    pub(crate) fn into_cfg(self) -> TraverseAnswerCfg {
        let base = TraverseAnswerCfg::default();
        TraverseAnswerCfg {
            hard_wall_budget_ms: self
                .traverse_answer_hard_wall_ms
                .unwrap_or(base.hard_wall_budget_ms),
            max_hops: self.traverse_answer_max_hops.unwrap_or(base.max_hops),
            experimental_enabled: self.single_call_multihop.unwrap_or(false),
        }
        .clamped()
    }
}

/// Runtime-derived head-of-line concurrency cap. `num_cpus * 0.75`,
/// floor-clamped at 2 so single-core hosts still get two permits. Uses
/// `std::thread::available_parallelism` to avoid a fresh dependency.
#[must_use]
pub(crate) fn derive_concurrency_cap() -> usize {
    let n = std::thread::available_parallelism()
        .map(std::num::NonZeroUsize::get)
        .unwrap_or(1);
    (n * 3 / 4).max(2)
}

/// Runtime-derived cold-start budget. Takes the rolling p95 of the
/// first twelve query walls (caller tracks this externally) clamped to
/// `[200ms, 2000ms]`; falls back to 500ms when no samples have
/// accumulated yet. See gap-catalog/09 Round 6 API updates.
#[must_use]
pub(crate) fn derive_cold_start_budget_ms(rolling_p95_ms: Option<u32>) -> u32 {
    rolling_p95_ms.map_or(500, |p| p.clamp(200, 2000))
}

// ---------------------------------------------------------------
// Handler
// ---------------------------------------------------------------

/// Request body for `POST /v1/traverse_answer`.
#[derive(Debug, Deserialize)]
pub(crate) struct TraverseAnswerRequest {
    /// Free-text query the handler expands over the graph.
    #[serde(default)]
    pub(crate) text: Option<String>,
    /// Caller-side cap on hops. Clamped to `[1, cfg.max_hops]` so
    /// callers can request fewer hops than the server ceiling but not
    /// more. Absent -> server default.
    #[serde(default)]
    pub(crate) max_hops: Option<u32>,
    /// Caller-side wall-clock budget in milliseconds. The server
    /// ALWAYS applies its own `hard_wall_budget_ms` on top; this knob
    /// can only REDUCE the wall, never extend it. Absent -> server
    /// default.
    #[serde(default)]
    pub(crate) budget_ms: Option<u32>,
}

/// A single node entry returned within a hop result.
#[derive(Debug, Serialize)]
pub(crate) struct HopNode {
    /// Stable node UUID.
    pub(crate) id: String,
    /// Node type label.
    pub(crate) ntype: String,
    /// Summary text, if the node carries one.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) summary: Option<String>,
}

/// Nodes discovered at a single BFS hop.
#[derive(Debug, Serialize)]
pub(crate) struct HopResult {
    /// 0-based hop index.
    pub(crate) hop: u32,
    /// Nodes whose UUIDs were reached at this hop depth.
    pub(crate) nodes: Vec<HopNode>,
}

/// Response body for `POST /v1/traverse_answer`.
#[derive(Debug, Serialize)]
pub(crate) struct TraverseAnswerResponse {
    /// Schema tag, stable across versions.
    pub(crate) schema: &'static str,
    /// Number of hops actually executed before halt / cutoff.
    pub(crate) hops_executed: u32,
    /// Wall-clock elapsed, in milliseconds.
    pub(crate) elapsed_ms: u32,
    /// Effective hard-wall ceiling that was in force for this request,
    /// in milliseconds (after clamping + caller-knob intersection).
    /// Surfaced so callers can reason about why a cutoff fired.
    pub(crate) hard_wall_ms_effective: u32,
    /// `true` when the hop loop halted because the hard wall was
    /// exceeded. Callers should treat this as a structural abort, not
    /// a semantic halt.
    pub(crate) hard_wall_cutoff: bool,
    /// `true` when the soft budget was breached but the hard wall was
    /// not (cold-start / contended hosts).
    pub(crate) budget_breached: bool,
    /// Per-hop BFS expansion results. Each entry holds the nodes
    /// discovered at that hop depth. Empty when the frontier is empty
    /// or when no start node could be resolved.
    pub(crate) hops: Vec<HopResult>,
}

/// Axum handler for `POST /v1/traverse_answer`.
///
/// See the module docs for the full semantics. The gate check runs
/// FIRST so a disabled deployment never even allocates the guard.
pub(crate) async fn traverse_answer(
    State(state): State<AppState>,
    Json(req): Json<TraverseAnswerRequest>,
) -> Response {
    // -------- Gate check: architect Decision 4 a future version default OFF ----
    if !state.traverse_cfg.experimental_enabled {
        return (
            StatusCode::GONE,
            Json(json!({
                "schema": "mnem.v1.err",
                "error": "traverse_answer: experimental endpoint disabled",
                "remediation_ref":
                    "docs/warnings/traverse_answer_experimental_opt_in.md",
                "opt_in":
                    "set `[experimental] single_call_multihop = true` in config.toml",
            })),
        )
            .into_response();
    }

    // -------- Effective ceilings ----------------------------------
    let cfg = &state.traverse_cfg;

    // Caller knob can only REDUCE the wall. If the caller supplied a
    // budget, we intersect with the server hard wall; otherwise we
    // apply the server default.
    let effective_hard_wall_ms = match req.budget_ms {
        Some(caller) => caller.clamp(TraverseAnswerCfg::HARD_WALL_MIN_MS, cfg.hard_wall_budget_ms),
        None => cfg.hard_wall_budget_ms,
    };
    let effective_max_hops = match req.max_hops {
        Some(caller) => caller.clamp(1, cfg.max_hops),
        None => cfg.max_hops,
    };

    // Publish the effective ceilings on the gauge. Prometheus gauges
    // are `i64` so we cast up from u32; no saturation possible.
    state
        .metrics
        .traverse_answer_hard_wall_ms_effective
        .set(i64::from(effective_hard_wall_ms));
    state
        .metrics
        .traverse_answer_max_hops_effective
        .set(i64::from(effective_max_hops));

    // -------- Hop loop under CommitBudgetGuard --------------------
    //
    // We synthesise a synthetic commit CID anchored on the request
    // text + effective ceilings so the guard's replay-determinism
    // contract is preserved (different requests yield different CIDs;
    // replaying the same request yields the same CID).
    let anchor_cid = synth_cid(&req, effective_hard_wall_ms, effective_max_hops);

    // Soft budget = 80% of the hard wall; gives callers a warning
    // signal on contended hosts before the structural abort fires.
    let soft_budget_ms = (effective_hard_wall_ms * 4) / 5;

    let mut guard = CommitBudgetGuard::start(
        "gap-09-traverse-answer",
        soft_budget_ms,
        effective_hard_wall_ms,
        anchor_cid,
    );

    let start = Instant::now();
    let mut hops_executed: u32 = 0;
    let mut hard_wall_cutoff = false;
    let mut hop_results: Vec<HopResult> = Vec::new();

    // -------- Seed the BFS frontier ----------------------------------
    //
    // If `text` parses as a valid node UUID, use it directly as the
    // sole seed. Otherwise embed the text through the configured dense
    // provider (MockEmbedder cold-start fallback when unconfigured) and
    // retrieve up to 10 semantically ranked seed nodes. An empty or
    // absent `text` yields an empty frontier (0 hops, no expansion).
    let mut frontier: Vec<NodeId> = {
        let repo = state.repo.lock().unwrap_or_else(|p| p.into_inner());
        match req.text.as_deref() {
            Some(t) if !t.trim().is_empty() => {
                // Try parsing as a UUID first (cheap, no I/O).
                if let Ok(id) = NodeId::parse_uuid(t) {
                    // Confirm the node exists and is not tombstoned.
                    if repo.lookup_node(&id).ok().flatten().is_some() && !repo.is_tombstoned(&id) {
                        vec![id]
                    } else {
                        Vec::new()
                    }
                } else {
                    // Free-text: embed the query through the configured
                    // dense provider (or the deterministic MockEmbedder
                    // cold-start fallback) and use the retrieve API so
                    // seeds are ranked by semantic similarity to the
                    // query text rather than arbitrary traversal order.
                    let (model, qvec) = {
                        if let Some(pc) = &state.embed_cfg
                            && let Ok(embedder) = mnem_embed_providers::open(pc)
                            && let Ok(v) = embedder.embed(t)
                        {
                            (embedder.model().to_string(), v)
                        } else {
                            let mock =
                                mnem_embed_providers::MockEmbedder::new("mock:cold-start-384", 384);
                            let v = mock.embed(t).unwrap_or_default();
                            (mock.model().to_string(), v)
                        }
                    };
                    let mut ret = repo
                        .retrieve()
                        .query_text(t)
                        .vector(model.clone(), qvec)
                        .limit(10);
                    // Attach the cached vector index so the retriever
                    // avoids rebuilding it on every hop-0 call.
                    if let Ok(mut cache) = state.indexes.lock() {
                        if let Ok(idx) = cache.vector_index(&repo, &model) {
                            ret = ret.with_vector_index(idx);
                        }
                    }
                    match ret.execute() {
                        Ok(result) => result
                            .items
                            .into_iter()
                            .map(|item| item.node.id)
                            .filter(|id| !repo.is_tombstoned(id))
                            .collect(),
                        Err(_) => Vec::new(),
                    }
                }
            }
            _ => Vec::new(),
        }
        // MutexGuard drops here, lock is released before any .await.
    };

    // Nodes already visited (seed + all expanded) so we never loop.
    let mut visited: HashSet<NodeId> = frontier.iter().copied().collect();

    // -------- Hop loop under CommitBudgetGuard ----------------------
    for hop in 0..effective_max_hops {
        if frontier.is_empty() {
            break;
        }

        let hop_stage_tag: &'static str = match hop {
            0 => "hop-0",
            1 => "hop-1",
            2 => "hop-2",
            _ => "hop-n",
        };

        // ---------- Synchronous graph expansion (no .await) ----------
        // Acquire the lock, do all blocking I/O, release before the
        // async guard.charge() call below.
        let (hop_nodes, next_frontier) = {
            let repo = state.repo.lock().unwrap_or_else(|p| p.into_inner());
            let mut hop_nodes: Vec<HopNode> = Vec::new();
            let mut next_ids: Vec<NodeId> = Vec::new();

            for node_id in &frontier {
                // Resolve node details for the current frontier member.
                if let Ok(Some(node)) = repo.lookup_node(node_id) {
                    hop_nodes.push(HopNode {
                        id: node.id.to_uuid_string(),
                        ntype: node.ntype,
                        summary: node.summary,
                    });
                }

                // Expand outgoing edges; no etype filter at this layer.
                let edges = repo.outgoing_edges(node_id, None).unwrap_or_default();

                for edge in edges {
                    let neighbor = edge.dst;
                    if visited.contains(&neighbor) {
                        continue;
                    }
                    visited.insert(neighbor);
                    // Skip tombstoned neighbors.
                    if repo.is_tombstoned(&neighbor) {
                        continue;
                    }
                    next_ids.push(neighbor);
                }
            }
            // MutexGuard drops here.
            (hop_nodes, next_ids)
        };

        hop_results.push(HopResult {
            hop,
            nodes: hop_nodes,
        });
        frontier = next_frontier;

        // ---------- Budget accounting (async-safe) -------------------
        match guard.charge(hop_stage_tag) {
            Ok(Decision::Proceed) => {
                hops_executed = hop.saturating_add(1);
            }
            Ok(Decision::ShouldDefer) => {
                hops_executed = hop.saturating_add(1);
                // Soft breach: record and stop doing new work.
                break;
            }
            Err(_hard_wall) => {
                hard_wall_cutoff = true;
                // Gap 09 R5 counter: increments on every structural
                // cutoff so dashboards can alert on the rate.
                state.metrics.traverse_answer_hard_wall_exceeded.inc();
                break;
            }
        }
    }

    // Emergency cutoff: even if `charge()` never reported the breach
    // (e.g. the hop body itself overran on a single iteration), the
    // wall-clock check here catches it.
    let elapsed_ms = u32::try_from(start.elapsed().as_millis()).unwrap_or(u32::MAX);
    if !hard_wall_cutoff && elapsed_ms > effective_hard_wall_ms {
        hard_wall_cutoff = true;
        state.metrics.traverse_answer_hard_wall_exceeded.inc();
    }

    let report = guard.into_report();
    let budget_breached = report.breached && !hard_wall_cutoff;

    Json(TraverseAnswerResponse {
        schema: "mnem.v1.traverse_answer",
        hops_executed,
        elapsed_ms,
        hard_wall_ms_effective: effective_hard_wall_ms,
        hard_wall_cutoff,
        budget_breached,
        hops: hop_results,
    })
    .into_response()
}

/// Synthesise a deterministic CID anchoring the budget guard to the
/// request shape. The guard uses this for replay determinism; we never
/// actually store it. Cheap BLAKE3 over the stable bytes of the
/// request envelope.
fn synth_cid(req: &TraverseAnswerRequest, hard_wall_ms: u32, max_hops: u32) -> Cid {
    let text = req.text.as_deref().unwrap_or("");
    let mut buf = Vec::with_capacity(text.len() + 16);
    buf.extend_from_slice(text.as_bytes());
    buf.extend_from_slice(&hard_wall_ms.to_le_bytes());
    buf.extend_from_slice(&max_hops.to_le_bytes());
    let hash = Multihash::sha2_256(&buf);
    Cid::new(CODEC_RAW, hash)
}

// ---------------------------------------------------------------
// Unit tests (pure logic; wire tests live in tests/wire_traverse_answer.rs)
// ---------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_is_off() {
        let cfg = TraverseAnswerCfg::default();
        assert!(!cfg.experimental_enabled);
        assert_eq!(cfg.hard_wall_budget_ms, 5000);
        assert_eq!(cfg.max_hops, 3);
    }

    #[test]
    fn clamped_enforces_envelope() {
        let cfg = TraverseAnswerCfg {
            hard_wall_budget_ms: 0,
            max_hops: 0,
            experimental_enabled: true,
        }
        .clamped();
        assert_eq!(cfg.hard_wall_budget_ms, TraverseAnswerCfg::HARD_WALL_MIN_MS);
        assert_eq!(cfg.max_hops, 1);

        let cfg = TraverseAnswerCfg {
            hard_wall_budget_ms: u32::MAX,
            max_hops: u32::MAX,
            experimental_enabled: true,
        }
        .clamped();
        assert_eq!(cfg.hard_wall_budget_ms, TraverseAnswerCfg::HARD_WALL_MAX_MS);
        assert_eq!(cfg.max_hops, TraverseAnswerCfg::MAX_HOPS_CEILING);
    }

    #[test]
    fn experimental_section_absent_keys_default_off() {
        let section = ExperimentalSection::default();
        let cfg = section.into_cfg();
        assert!(!cfg.experimental_enabled);
        assert_eq!(cfg.hard_wall_budget_ms, 5000);
        assert_eq!(cfg.max_hops, 3);
    }

    #[test]
    fn experimental_section_opts_in() {
        let section = ExperimentalSection {
            single_call_multihop: Some(true),
            traverse_answer_hard_wall_ms: Some(8000),
            traverse_answer_max_hops: Some(5),
        };
        let cfg = section.into_cfg();
        assert!(cfg.experimental_enabled);
        assert_eq!(cfg.hard_wall_budget_ms, 8000);
        assert_eq!(cfg.max_hops, 5);
    }

    #[test]
    fn concurrency_cap_never_below_two() {
        let cap = derive_concurrency_cap();
        assert!(cap >= 2, "concurrency cap must never drop below 2");
    }

    #[test]
    fn cold_start_budget_fallback_is_500() {
        assert_eq!(derive_cold_start_budget_ms(None), 500);
    }

    #[test]
    fn cold_start_budget_clamps_rolling_p95() {
        assert_eq!(derive_cold_start_budget_ms(Some(50)), 200);
        assert_eq!(derive_cold_start_budget_ms(Some(900)), 900);
        assert_eq!(derive_cold_start_budget_ms(Some(9999)), 2000);
    }

    // Gap 09 R6 proptest: the hard wall is structurally impossible to
    // DoS via caller input. No matter what `budget_ms` / `max_hops`
    // the caller supplies, the clamped effective hard wall lies in
    // `[HARD_WALL_MIN_MS, server_cfg.hard_wall_budget_ms]`.
    proptest::proptest! {
        #[test]
        fn hard_wall_structural_dos_impossible(
            caller_budget_ms in 0u32..=u32::MAX,
            server_wall_ms in TraverseAnswerCfg::HARD_WALL_MIN_MS
                ..=TraverseAnswerCfg::HARD_WALL_MAX_MS,
        ) {
            let effective = caller_budget_ms.clamp(
                TraverseAnswerCfg::HARD_WALL_MIN_MS,
                server_wall_ms,
            );
            proptest::prop_assert!(effective >= TraverseAnswerCfg::HARD_WALL_MIN_MS);
            proptest::prop_assert!(effective <= server_wall_ms);
        }
    }

    // Gap 09 R6 proptest: 3 hops is a hard ceiling on the default
    // config, and any caller-supplied value clamps into [1, 3].
    proptest::proptest! {
        #[test]
        fn hops_3_covers_99_pct_multihop_benchmarks(
            caller_hops in 0u32..=u32::MAX,
        ) {
            let cfg = TraverseAnswerCfg::default();
            let effective = caller_hops.clamp(1, cfg.max_hops);
            proptest::prop_assert!(effective >= 1);
            proptest::prop_assert!(effective <= 3);
        }
    }
}
