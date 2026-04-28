//! Prometheus `/metrics` endpoint.
//!
//! observability scope, R2-C:
//!
//! - `mnem_http_requests_total{method,route,status}` counter.
//! - `mnem_http_request_duration_seconds` histogram (default buckets).
//! - `mnem_retrieve_latency_seconds` histogram (retrieve handler only).
//! - `mnem_commit_duration_seconds` histogram (write commit paths).
//!
//! Labels are fixed-cardinality strings: `method` is a small set
//! (GET / POST / DELETE), `route` is the matched axum route template
//! (NOT the raw URI path -- keeps cardinality bounded at the number of
//! registered routes), `status` is the HTTP status code as a decimal
//! string.
//!
//! The registry is kept behind an `Arc` and cloned into every handler
//! via `State<AppState>`. `prometheus-client` 0.23 uses lock-free
//! atomics internally, so the per-request cost is one `fetch_add` per
//! metric family plus a `HashMap<LabelSet, Counter>` lookup for the
//! `Family` types. Well under 100 ns on x86-64 per hit.
//!
//! # Gating
//!
//! The route is always mounted; the `--metrics` CLI flag controls
//! whether the binary's startup line points at it. The default is ON,
//! matching the H3 mode described in the R1 observability scorer. For
//! loopback-only binds (the default), an operator can scrape without
//! further config. For non-loopback binds, `MNEM_HTTP_ALLOW_NON_LOOPBACK`
//! already gates the bind itself; downstream proxies terminate auth
//! before they reach `/metrics`.

use std::sync::Arc;
use std::time::Instant;

use axum::extract::{MatchedPath, Request, State};
use axum::http::StatusCode;
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};
use prometheus_client::encoding::{EncodeLabelSet, text::encode};
use prometheus_client::metrics::counter::Counter;
use prometheus_client::metrics::family::Family;
use prometheus_client::metrics::gauge::Gauge;
use prometheus_client::metrics::histogram::{Histogram, exponential_buckets};
use prometheus_client::registry::Registry;

use crate::state::AppState;

/// Label set for `mnem_http_requests_total`. All fields are
/// small-cardinality strings: `method` is one of GET / POST / DELETE /
/// ..., `route` is the MATCHED axum route template (so `/v1/nodes/{id}`
/// becomes one bucket, not one per distinct node id), `status` is the
/// HTTP status code rendered as a decimal string.
#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
pub struct HttpRequestLabels {
    /// HTTP method: `GET`, `POST`, `DELETE`, etc.
    pub method: String,
    /// Matched axum route template, e.g. `/v1/nodes/{id}`. Falls back
    /// to the literal URI path when the request did not match any
    /// registered route (404s).
    pub route: String,
    /// HTTP status code as a decimal string (e.g. `"200"`, `"404"`).
    pub status: String,
}

/// Label set for `mnem_remote_advance_head_total`. The `result`
/// label is a small closed vocabulary so dashboards can alert on
/// CAS mismatch rate and auth-failure rate independently from
/// legitimate traffic.
#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
pub struct AdvanceHeadLabels {
    /// One of `success`, `cas_mismatch`, `auth_fail`. Keep the
    /// vocabulary closed; adding a new value is a dashboard change
    /// and requires a coordinated change.
    pub result: String,
}

/// Label set for `mnem_leiden_mode_total` (Gap 10 R3). Closed
/// vocabulary `full | full_debounced | fallback_stale`. Dashboards
/// alert on the ratio of `fallback_stale` to the other two.
#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
pub struct LeidenModeLabels {
    /// One of `full`, `full_debounced`, `fallback_stale`.
    pub mode: String,
}

/// Label set for `mnem_ppr_size_gate_skipped_total` (Gap 02 #17).
/// Closed vocabulary so dashboards can separate "gate tripped because
/// the graph got big" from "gate tripped because the caller pinned
/// `ppr_opt_in = false`".
#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
pub struct PprSizeGateLabels {
    /// One of `above_threshold`, `opted_out`. Keep the vocabulary
    /// closed; adding a new value is a dashboard change and requires
    /// a coordinated change.
    pub reason: String,
}

/// Server-wide Prometheus metric registry.
///
/// Cloned into [`AppState`] behind an `Arc` so every handler gets a
/// cheap reference. All metrics are registered up-front at construction
/// time; there is no per-request registry mutation.
#[derive(Clone)]
pub struct Metrics {
    registry: Arc<Registry>,
    /// Per-request counter keyed on (method, route, status).
    pub http_requests: Family<HttpRequestLabels, Counter>,
    /// Request-duration histogram (seconds). Buckets cover 1ms to 10s,
    /// which matches the typical range for local-first HTTP calls.
    pub http_duration: Histogram,
    /// Retrieve-handler latency histogram (seconds). Separate from
    /// `http_duration` so operators can track hybrid-retrieval cost
    /// without the embed / ingest traffic skewing the distribution.
    pub retrieve_latency: Histogram,
    /// Commit-duration histogram (seconds). Covers the end-to-end
    /// write path including vector cache invalidation and redb fsync.
    pub commit_duration: Histogram,
    /// Ingest-pipeline duration histogram (seconds). Measured around
    /// the full `POST /v1/ingest` run: parse + chunk + extract +
    /// commit. Separate from `http_duration` + `commit_duration` so
    /// operators can see where the time went inside a single ingest.
    pub ingest_duration: Histogram,
    /// Total chunks produced across every successful `/v1/ingest`
    /// call. Monotonic counter; divide by `ingest_duration`'s sample
    /// count for an average-chunks-per-ingest view.
    pub ingest_chunks: Counter,
    /// `/remote/v1/fetch-blocks` invocation counter.
    pub remote_fetch_blocks: Counter,
    /// `/remote/v1/push-blocks` invocation counter (counts successful
    /// imports; auth and body-decode failures short-circuit earlier).
    pub remote_push_blocks: Counter,
    /// `/remote/v1/advance-head` invocation counter, bucketed by
    /// `result` (`success` | `cas_mismatch` | `auth_fail`).
    pub remote_advance_head: Family<AdvanceHeadLabels, Counter>,
    /// Gap 10 R3: Leiden recompute-mode counter, one increment per
    /// `community_for_head` serve. Labelled `full | full_debounced |
    /// fallback_stale`.
    pub leiden_mode: Family<LeidenModeLabels, Counter>,
    /// Gap 10 R6 (floor-a runtime): effective debounce window in ms.
    pub leiden_debounce_effective: Gauge,
    /// Gap 10 R6 (floor-c, default 60): effective commit-storm cap.
    pub leiden_storm_cap_effective: Gauge,
    /// Gap 10 R6 (floor-c, default 0.5): effective delta-ratio force-
    /// full fraction, encoded as parts-per-ten-thousand.
    pub leiden_delta_ratio_effective: Gauge,
    /// Gap 10 current-mode indicator. `0=full, 1=full_debounced, 2=fallback_stale`.
    pub leiden_mode_current: Gauge,
    /// Gap 09 traverse_answer effective hard-wall ms (tunable mirror).
    pub traverse_answer_hard_wall_ms_effective: Gauge,
    /// Gap 09 traverse_answer effective max-hops (tunable mirror).
    pub traverse_answer_max_hops_effective: Gauge,
    /// Gap 09 traverse_answer hard-wall breach counter.
    pub traverse_answer_hard_wall_exceeded: Counter,
    /// Gap 02 #17: PPR size-gate skipped count, labeled by reason.
    /// Closed vocabulary `above_threshold | opted_out`.
    pub ppr_size_gate_skipped: Family<PprSizeGateLabels, Counter>,
    /// Gap 02 #17: effective threshold (mirrors
    /// [`mnem_core::ppr::PPR_DEFAULT_MAX_NODES`] tunable).
    pub ppr_size_gate_threshold: Gauge,
}

impl Metrics {
    /// Build a fresh registry with all four metric families registered.
    ///
    /// Exponential buckets are used so the histograms cover several
    /// orders of magnitude with a constant bucket count. The first
    /// bucket (1ms for requests, 100us for retrieves/commits) matches
    /// the fastest plausible path; the last caps at 10s which is the
    /// operator-visible ceiling before a caller typically gives up.
    #[must_use]
    pub fn new() -> Self {
        let mut registry = Registry::default();

        let http_requests = Family::<HttpRequestLabels, Counter>::default();
        registry.register(
            "mnem_http_requests_total",
            "Total HTTP requests handled by mnem-http, bucketed by method, route, and status.",
            http_requests.clone(),
        );

        // 1ms..10s with 14 buckets at base-2 growth.
        let http_duration = Histogram::new(exponential_buckets(0.001, 2.0, 14));
        registry.register(
            "mnem_http_request_duration_seconds",
            "HTTP request duration in seconds, from axum route match to response body sent.",
            http_duration.clone(),
        );

        // 100us..10s; retrieves dominated by vector+sparse fusion usually land in the 1ms..200ms range.
        let retrieve_latency = Histogram::new(exponential_buckets(0.0001, 2.0, 17));
        registry.register(
            "mnem_retrieve_latency_seconds",
            "Retrieval pipeline latency in seconds, measured around the `Retriever::execute` call.",
            retrieve_latency.clone(),
        );

        // 100us..10s; redb commits can fsync for 10-40ms on spinning disks, shorter on NVMe.
        let commit_duration = Histogram::new(exponential_buckets(0.0001, 2.0, 17));
        registry.register(
            "mnem_commit_duration_seconds",
            "Transaction commit duration in seconds, measured around Transaction::commit_opts.",
            commit_duration.clone(),
        );

        // 1ms..10s; ingests are dominated by chunker + NER over the
        // whole source, which typically lands in the 5ms..1s range on
        // mid-sized markdown / PDF.
        let ingest_duration = Histogram::new(exponential_buckets(0.001, 2.0, 14));
        registry.register(
            "mnem_ingest_duration_seconds",
            "End-to-end ingest duration in seconds, measured around the full POST /v1/ingest run.",
            ingest_duration.clone(),
        );

        let ingest_chunks = Counter::default();
        registry.register(
            "mnem_ingest_chunks_total",
            "Total chunks produced across every successful POST /v1/ingest call.",
            ingest_chunks.clone(),
        );

        // `/remote/v1/*` per-verb counters. Declared under
        // `mnem_remote_*` (not `mnem_http_*`) so the remote-protocol
        // surface is trivially filterable from the v1 REST traffic
        // on a dashboard.
        let remote_fetch_blocks = Counter::default();
        registry.register(
            "mnem_remote_fetch_blocks_total",
            "Total `/remote/v1/fetch-blocks` invocations that produced a CAR response.",
            remote_fetch_blocks.clone(),
        );
        let remote_push_blocks = Counter::default();
        registry.register(
            "mnem_remote_push_blocks_total",
            "Total `/remote/v1/push-blocks` invocations that completed an import.",
            remote_push_blocks.clone(),
        );
        let remote_advance_head = Family::<AdvanceHeadLabels, Counter>::default();
        registry.register(
            "mnem_remote_advance_head_total",
            "Total `/remote/v1/advance-head` invocations bucketed by result (success, cas_mismatch, auth_fail).",
            remote_advance_head.clone(),
        );

        // Gap 10 Phase-1 Leiden-cache telemetry.
        let leiden_mode = Family::<LeidenModeLabels, Counter>::default();
        registry.register(
            "mnem_leiden_mode_total",
            "Total Leiden community-cache serves bucketed by mode (full, full_debounced, fallback_stale).",
            leiden_mode.clone(),
        );
        let leiden_debounce_effective = Gauge::default();
        registry.register(
            "mnem_leiden_debounce_effective",
            "Effective Leiden debounce window in ms (max(1000, rolling p75 commit latency)).",
            leiden_debounce_effective.clone(),
        );
        let leiden_storm_cap_effective = Gauge::default();
        registry.register(
            "mnem_leiden_storm_cap_effective",
            "Effective commit-storm cap per minute (floor-c tunable; default 60).",
            leiden_storm_cap_effective.clone(),
        );
        let leiden_delta_ratio_effective = Gauge::default();
        registry.register(
            "mnem_leiden_delta_ratio_effective",
            "Effective delta_ratio_force_full rendered as parts-per-ten-thousand.",
            leiden_delta_ratio_effective.clone(),
        );
        let leiden_mode_current = Gauge::default();
        registry.register(
            "mnem_leiden_mode_current",
            "Current Leiden mode: 0=full, 1=full_debounced, 2=fallback_stale.",
            leiden_mode_current.clone(),
        );

        // Gap 09 traverse_answer telemetry (carry-over).
        let traverse_answer_hard_wall_ms_effective = Gauge::default();
        registry.register(
            "mnem_traverse_answer_hard_wall_ms_effective",
            "Effective hard-wall latency budget for /v1/traverse_answer in ms.",
            traverse_answer_hard_wall_ms_effective.clone(),
        );
        let traverse_answer_max_hops_effective = Gauge::default();
        registry.register(
            "mnem_traverse_answer_max_hops_effective",
            "Effective max-hops for /v1/traverse_answer.",
            traverse_answer_max_hops_effective.clone(),
        );
        let traverse_answer_hard_wall_exceeded = Counter::default();
        registry.register(
            "mnem_traverse_answer_hard_wall_exceeded_total",
            "Total /v1/traverse_answer requests that breached the hard-wall budget.",
            traverse_answer_hard_wall_exceeded.clone(),
        );

        // Gap 02 #17 PPR size-gate telemetry.
        let ppr_size_gate_skipped = Family::<PprSizeGateLabels, Counter>::default();
        registry.register(
            "mnem_ppr_size_gate_skipped_total",
            "Total PPR requests skipped by the default-on size gate, bucketed by reason (above_threshold, opted_out).",
            ppr_size_gate_skipped.clone(),
        );
        let ppr_size_gate_threshold = Gauge::default();
        registry.register(
            "mnem_ppr_size_gate_threshold",
            "Effective PPR size-gate node threshold (mirrors PPR_DEFAULT_MAX_NODES).",
            ppr_size_gate_threshold.clone(),
        );
        // Initialize the gauge to the compile-time constant so scrapes
        // always have a non-zero value even before any PPR call.
        #[allow(clippy::cast_possible_wrap)]
        ppr_size_gate_threshold.set(mnem_core::ppr::PPR_DEFAULT_MAX_NODES as i64);

        Self {
            registry: Arc::new(registry),
            http_requests,
            http_duration,
            retrieve_latency,
            commit_duration,
            ingest_duration,
            ingest_chunks,
            remote_fetch_blocks,
            remote_push_blocks,
            remote_advance_head,
            leiden_mode,
            leiden_debounce_effective,
            leiden_storm_cap_effective,
            leiden_delta_ratio_effective,
            leiden_mode_current,
            traverse_answer_hard_wall_ms_effective,
            traverse_answer_max_hops_effective,
            traverse_answer_hard_wall_exceeded,
            ppr_size_gate_skipped,
            ppr_size_gate_threshold,
        }
    }

    /// Encode the current metrics as Prometheus text-exposition format.
    ///
    /// # Errors
    ///
    /// Returns an `std::fmt::Error` only if the in-memory writer
    /// rejects a write, which cannot happen for `String` under normal
    /// conditions. Surfaces the error so callers can turn it into a
    /// 500 rather than panic.
    pub fn encode(&self) -> Result<String, std::fmt::Error> {
        let mut buf = String::new();
        encode(&mut buf, &self.registry)?;
        Ok(buf)
    }
}

impl Default for Metrics {
    fn default() -> Self {
        Self::new()
    }
}

/// Axum middleware: time the request, record the histogram, bump the
/// counter. Installed once at router construction; fires for every
/// non-`/metrics` route.
///
/// `/metrics` is exempted to avoid scrape loops skewing the
/// distributions (every scrape would bump `mnem_http_requests_total`
/// and the retrieve latency histogram with its own sample).
pub(crate) async fn track_metrics(
    State(state): State<AppState>,
    req: Request,
    next: Next,
) -> Response {
    let method = req.method().as_str().to_string();
    // `MatchedPath` is an axum extension populated by the router when
    // the request matched a registered route template. 404 paths are
    // recorded as the literal URI path (bounded in practice by ops
    // reality: you don't have infinite 404 paths in steady state).
    let route = req
        .extensions()
        .get::<MatchedPath>()
        .map_or_else(|| req.uri().path().to_string(), |m| m.as_str().to_string());

    // Skip instrumentation of `/metrics` itself to keep scrapes from
    // inflating their own histograms.
    if route == "/metrics" {
        return next.run(req).await;
    }

    let start = Instant::now();
    let response = next.run(req).await;
    let elapsed = start.elapsed().as_secs_f64();

    let status = response.status().as_u16().to_string();
    state
        .metrics
        .http_requests
        .get_or_create(&HttpRequestLabels {
            method,
            route,
            status,
        })
        .inc();
    state.metrics.http_duration.observe(elapsed);

    response
}

/// `GET /metrics` handler. Emits text-exposition format with the
/// `text/plain; version=0.0.4` content-type Prometheus expects.
pub(crate) async fn metrics_handler(State(state): State<AppState>) -> Response {
    match state.metrics.encode() {
        Ok(body) => (
            StatusCode::OK,
            [(
                axum::http::header::CONTENT_TYPE,
                "text/plain; version=0.0.4; charset=utf-8",
            )],
            body,
        )
            .into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("metrics encoding failure: {e}"),
        )
            .into_response(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn metrics_encode_is_well_formed() {
        // Fresh registry, bump one counter, one histogram sample; encoded
        // output must contain each metric family's HELP + TYPE line and
        // each label we recorded. Guards the wire contract: dashboards
        // and alerts will grep these exact family names.
        let m = Metrics::new();
        m.http_requests
            .get_or_create(&HttpRequestLabels {
                method: "GET".into(),
                route: "/v1/healthz".into(),
                status: "200".into(),
            })
            .inc();
        m.http_duration.observe(0.002);
        m.retrieve_latency.observe(0.015);
        m.commit_duration.observe(0.050);

        let text = m.encode().expect("encode");

        // Each family's metadata line.
        assert!(
            text.contains("# HELP mnem_http_requests_total"),
            "missing HELP for mnem_http_requests_total in:\n{text}"
        );
        assert!(
            text.contains("# TYPE mnem_http_requests_total counter"),
            "missing TYPE for mnem_http_requests_total"
        );
        assert!(
            text.contains("# HELP mnem_http_request_duration_seconds"),
            "missing HELP for mnem_http_request_duration_seconds"
        );
        assert!(
            text.contains("# HELP mnem_retrieve_latency_seconds"),
            "missing HELP for mnem_retrieve_latency_seconds"
        );
        assert!(
            text.contains("# HELP mnem_commit_duration_seconds"),
            "missing HELP for mnem_commit_duration_seconds"
        );

        // Counter sample landed with the expected labels.
        assert!(
            text.contains("method=\"GET\""),
            "counter label `method=GET` missing in:\n{text}"
        );
        assert!(
            text.contains("route=\"/v1/healthz\""),
            "counter label `route=/v1/healthz` missing"
        );
        assert!(
            text.contains("status=\"200\""),
            "counter label `status=200` missing"
        );
    }

    #[test]
    fn metrics_new_registers_all_four_families() {
        // Narrow regression guard: if any of the registered metric
        // names disappears from Metrics::new, the scrape contract
        // breaks. Bumped in B5d to include the two ingest families.
        let m = Metrics::new();
        let text = m.encode().unwrap();
        for family in [
            "mnem_http_requests_total",
            "mnem_http_request_duration_seconds",
            "mnem_retrieve_latency_seconds",
            "mnem_commit_duration_seconds",
            "mnem_ingest_duration_seconds",
            "mnem_ingest_chunks_total",
        ] {
            assert!(
                text.contains(family),
                "expected metric family `{family}` in output:\n{text}"
            );
        }
    }

    #[test]
    fn metrics_new_registers_all_remote_families() {
        // Guard the `/remote/v1/*` counter contract. Dashboards alert
        // on exactly these three names.
        let m = Metrics::new();
        let text = m.encode().unwrap();
        for family in [
            "mnem_remote_fetch_blocks_total",
            "mnem_remote_push_blocks_total",
            "mnem_remote_advance_head_total",
        ] {
            assert!(
                text.contains(family),
                "expected metric family `{family}` in output:\n{text}"
            );
        }
    }

    #[test]
    fn remote_counters_increment_and_render() {
        let m = Metrics::new();
        m.remote_fetch_blocks.inc();
        m.remote_push_blocks.inc();
        m.remote_advance_head
            .get_or_create(&AdvanceHeadLabels {
                result: "success".into(),
            })
            .inc();
        m.remote_advance_head
            .get_or_create(&AdvanceHeadLabels {
                result: "cas_mismatch".into(),
            })
            .inc();
        m.remote_advance_head
            .get_or_create(&AdvanceHeadLabels {
                result: "auth_fail".into(),
            })
            .inc();
        let text = m.encode().unwrap();
        assert!(text.contains("mnem_remote_fetch_blocks_total"));
        assert!(text.contains("mnem_remote_push_blocks_total"));
        // Each closed-vocabulary result label must render.
        for r in ["success", "cas_mismatch", "auth_fail"] {
            assert!(
                text.contains(&format!("result=\"{r}\"")),
                "missing advance-head result `{r}` in:\n{text}"
            );
        }
    }
}
