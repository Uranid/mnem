//! HTTP JSON API for mnem.
//!
//! Library half of the `mnem http` binary. `app(repo_dir)` builds an
//! axum `Router` that wraps an open [`ReadonlyRepo`] on `repo_dir/.mnem`
//! (auto-initialising if needed).
//!
//! Scope v1:
//! - `GET /v1/healthz` - liveness probe.
//! - `GET /v1/stats` - head op-id, commit CID, ref + label counts.
//! - `POST /v1/nodes` - commit a new node (label + summary + props).
//! - `GET /v1/nodes/{id}` - fetch one node by UUID.
//! - `DELETE /v1/nodes/{id}` - commit a removal of one node.
//! - `GET /v1/retrieve?text=&budget=&limit=` - dense vector retrieval
//!   (embedder required when `text` is set). Returns rendered items
//!   plus budget metadata.
//!
//! Tokio lives ONLY in this crate. `mnem-core` stays WASM-clean.
//! This crate compiles to a single binary + library pair.

#![forbid(unsafe_code)]
#![deny(missing_docs)]

use std::path::Path;
use std::sync::{Arc, Mutex};

use anyhow::Result;
use axum::Router;
use axum::extract::DefaultBodyLimit;
use axum::routing::{delete, get, post};
use mnem_backend_redb::open_or_init;
use mnem_core::repo::ReadonlyRepo;
use tower_http::cors::{Any, CorsLayer};
use tower_http::trace::TraceLayer;

mod auth;
mod correlation;
mod error;
mod handlers;
mod handlers_ingest;
mod metrics;
mod remote_rate_limit;
mod routes;
mod state;

pub use error::{Error, RemoteError};
pub use handlers::derive_max_path_bytes;
pub use metrics::Metrics;
pub use state::AppState;

/// Gap 10 Phase-1 public surface: exposed for integration tests and
/// operators who need to observe / override the Leiden cache policy.
pub mod leiden_state {
    pub use crate::state::{
        COMMIT_LATENCY_WINDOW, COMMIT_STORM_CAP_PER_MIN, DEBOUNCE_FLOOR_MS, DELTA_RATIO_FORCE_FULL,
        GRAPH_SIZE_GATE_V, LeidenCache, LeidenMode, derive_debounce_ms,
    };
}

/// Options consumed by [`app_with_options`]. Fields default to the
/// same values `app` derives from the environment; tests construct an
/// explicit value to bypass the env-var read.
#[derive(Clone, Debug, Default)]
pub struct AppOptions {
    /// Override for `MNEM_BENCH`. `None` means "read the env var"; set
    /// to `Some(true)` in integration tests that exercise the label
    /// round-trip without polluting the process-wide environment.
    pub allow_labels: Option<bool>,
    /// When true, use `MemoryBlockstore` + `MemoryOpHeadsStore` instead
    /// of the redb-backed on-disk store. All commits live in RAM and
    /// are lost on process exit. Intended for benchmark harnesses and
    /// ephemeral agent sessions where durability is undesired and
    /// commit throughput matters (redb fsync can be 30-40x slower than
    /// memory per commit; see internal benchmarking). Never
    /// enable this in a deployment that needs to survive restart.
    pub in_memory: bool,
    /// Mount the `/metrics` Prometheus endpoint.
    ///
    /// `true` mounts the route; `false` omits it entirely (scrapes
    /// get a 404). The tracking middleware that populates the
    /// counters still runs either way -- flipping this on at the next
    /// restart begins exposing already-collected data.
    pub metrics_enabled: bool,
    /// Override for `MNEM_HTTP_PUSH_TOKEN`. `None` means "read the env
    /// var"; set to `Some("tok".into())` in integration tests that need
    /// to exercise authenticated write routes without touching the
    /// process-wide environment.
    pub push_token: Option<String>,
}

/// Build the router for a repo whose `.mnem/` lives at `repo_dir`.
/// Opens or initialises the redb; returns the router you `serve()`.
pub fn app(repo_dir: &Path) -> Result<Router> {
    app_with_options(repo_dir, AppOptions::default())
}

/// audit-2026-04-25 P2-7: enumerate every route the router mounts so
/// the startup banner in `mnem http` main is no longer hand-written
/// and incomplete. Each entry is `(METHOD-LIST, PATH, brief)`. Kept
/// in sync with the `Router::new().route(...)` chain in
/// `app_with_options` by colocating the data here; tests in
/// `tests/banner_route_table.rs` assert the count matches the
/// router's route count.
pub fn route_table(metrics_enabled: bool) -> Vec<(&'static str, &'static str, &'static str)> {
    let mut routes: Vec<(&'static str, &'static str, &'static str)> = vec![
        ("GET", "/v1/healthz", "liveness probe"),
        (
            "GET",
            "/v1/stats",
            "head op-id, commit CID, ref + label counts",
        ),
        (
            "GET",
            "/v1/log",
            "op-log history (limit, format=json|oneline|full)",
        ),
        (
            "GET",
            "/v1/export",
            "export all reachable blocks as NDJSON (hex-encoded)",
        ),
        (
            "GET",
            "/v1/blocks/{cid}",
            "fetch one raw block by CID (format=json|raw|cbor)",
        ),
        (
            "POST",
            "/v1/import",
            "import blocks from NDJSON stream (block-level sync)",
        ),
        (
            "POST",
            "/v1/diff",
            "structural diff between two commits or ops",
        ),
        (
            "POST",
            "/v1/merge",
            "3-way merge two commit CIDs; returns fast_forward / clean / conflicts",
        ),
        ("POST", "/v1/nodes", "commit a new node"),
        (
            "POST",
            "/v1/nodes/bulk",
            "commit N nodes in one transaction",
        ),
        ("POST", "/v1/edges", "commit a new directed edge"),
        ("GET/DELETE", "/v1/nodes/{id}", "fetch / delete a node"),
        (
            "GET",
            "/v1/nodes/{id}/embedding",
            "fetch embedding vector for a node by model",
        ),
        ("POST", "/v1/nodes/{id}/tombstone", "tombstone a node"),
        ("GET/POST", "/v1/retrieve", "agent-facing retrieval"),
        (
            "POST",
            "/v1/ingest",
            "ingest a Markdown / PDF / JSON source",
        ),
        ("POST", "/v1/explain", "explain a retrieve result"),
        (
            "POST",
            "/v1/traverse_answer",
            "single-call multihop (gated)",
        ),
        ("GET/POST", "/v1/branches", "list / create branches"),
        ("DELETE", "/v1/branches/{*name}", "delete a branch by name"),
        ("GET/POST", "/v1/tags", "list / create tags"),
        ("DELETE", "/v1/tags/{*name}", "delete a tag by name"),
        ("GET", "/remote/v1/refs", "transport: list refs"),
        ("POST", "/remote/v1/fetch-blocks", "transport: fetch blocks"),
        (
            "POST",
            "/remote/v1/push-blocks",
            "transport: push blocks (auth)",
        ),
        (
            "POST",
            "/remote/v1/advance-head",
            "transport: advance head (auth)",
        ),
    ];
    if metrics_enabled {
        routes.push(("GET", "/metrics", "Prometheus text-exposition"));
    }
    routes
}

/// [`app`] with programmatic overrides. Used by integration tests so
/// they can flip `allow_labels` without touching the environment.
pub fn app_with_options(repo_dir: &Path, opts: AppOptions) -> Result<Router> {
    let data_dir = if repo_dir.ends_with(".mnem") {
        repo_dir.to_path_buf()
    } else {
        repo_dir.join(".mnem")
    };
    std::fs::create_dir_all(&data_dir)?;
    let (bs, ohs): (
        std::sync::Arc<dyn mnem_core::store::Blockstore>,
        std::sync::Arc<dyn mnem_core::store::OpHeadsStore>,
    ) = if opts.in_memory {
        // Ephemeral in-memory mode. `repo_dir` is still used (for
        // `config.toml` load), but nothing persists to disk. Loud
        // stderr warning so an operator who flipped the flag by
        // accident sees it immediately.
        eprintln!(
            "mnem http: --in-memory ACTIVE. All commits are RAM-only and lost on process exit. This is intended for benchmarks and ephemeral sessions; NEVER use in a durable deployment."
        );
        (
            std::sync::Arc::new(mnem_core::store::MemoryBlockstore::new()),
            std::sync::Arc::new(mnem_core::store::MemoryOpHeadsStore::new()),
        )
    } else {
        let (bs, ohs, _file) = open_or_init(&data_dir.join("repo.redb"))?;
        (bs as _, ohs as _)
    };
    let repo = ReadonlyRepo::open(bs.clone(), ohs.clone()).or_else(|e| {
        if e.is_uninitialized() {
            ReadonlyRepo::init(bs.clone(), ohs.clone())
        } else {
            Err(e)
        }
    })?;

    // Resolve embed + sparse + NER provider configs from the repo's
    // config.toml, if any. When present, ingest and retrieve paths
    // auto-run the corresponding provider so hybrid dense + sparse
    // retrieval fires end-to-end (same behaviour as the CLI).
    let embed_cfg = load_embed_config(&data_dir);
    let sparse_cfg = load_sparse_config(&data_dir);
    let ner_cfg = load_ner_config(&data_dir);

    // `allow_labels` is gated behind the `MNEM_BENCH` env var. Off by
    // default so casual / single-tenant callers never stumble into
    // label-scoped state. Benchmark harnesses opt in by launching the
    // server with `MNEM_BENCH=1` (see docs/benchmarks/RUNNING.md).
    // Tests skip the env read by passing an explicit override.
    let allow_labels = opts
        .allow_labels
        .unwrap_or_else(AppState::resolve_allow_labels_from_env);
    if allow_labels && opts.allow_labels.is_none() {
        eprintln!(
            "mnem http: MNEM_BENCH set; caller-supplied `label` fields will be honoured on ingest and retrieve."
        );
    }

    // Remote-push bearer token lives in env only (MNEM_HTTP_PUSH_TOKEN),
    // never on disk. `None` disables the two authenticated `/remote/v1/*`
    // verbs (fail-closed 503). See crate::auth for the extractor.
    let push_token = opts
        .push_token
        .clone()
        .or_else(AppState::resolve_push_token_from_env);
    if push_token.is_some() {
        tracing::info!(
            "mnem http: MNEM_HTTP_PUSH_TOKEN configured; /remote/v1/push-blocks + /remote/v1/advance-head enabled."
        );
    } else {
        tracing::info!(
            "mnem http: MNEM_HTTP_PUSH_TOKEN not set; remote write verbs administratively disabled (503)."
        );
    }

    let state = AppState {
        repo: Arc::new(Mutex::new(repo)),
        embed_cfg,
        sparse_cfg,
        indexes: Arc::new(Mutex::new(state::IndexCache::default())),
        allow_labels,
        metrics: Metrics::new(),
        push_token,
        graph_cache: Arc::new(Mutex::new(state::GraphCache::default())),
        traverse_cfg: Arc::new(routes::traverse::TraverseAnswerCfg::default()),
        ner_cfg,
    };

    // Permissive CORS for v1: the server binds to loopback by default
    // anyway, and browser clients need CORS to talk to us at all. Users
    // exposing mnem http to the network must front it with an auth proxy.
    let cors = CorsLayer::new()
        .allow_methods(Any)
        .allow_headers(Any)
        .allow_origin(Any);

    // Request-body size cap. axum 0.7's `Json<T>` default is 2 MiB,
    // which is fine for `POST /v1/nodes` but too small for
    // `POST /v1/nodes/bulk` batches (128 nodes * real summaries can
    // comfortably exceed 2 MiB). Default here is 64 MiB, overridable
    // via `MNEM_MAX_BODY_MB` for operators who want stricter DoS
    // posture or looser batch ingests.
    let body_limit_bytes: usize = std::env::var("MNEM_MAX_BODY_MB")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(64)
        .saturating_mul(1024 * 1024);

    let mut router = Router::new()
        .route("/v1/healthz", get(handlers::healthz))
        .route("/v1/stats", get(handlers::stats))
        .route("/v1/log", get(handlers::get_log))
        .route("/v1/export", get(handlers::get_export))
        .route("/v1/blocks/{cid}", get(handlers::get_block))
        .route("/v1/import", post(handlers::post_import))
        .route("/v1/diff", post(handlers::post_diff))
        .route("/v1/merge", post(handlers::post_merge))
        .route("/v1/nodes", post(handlers::post_node))
        .route("/v1/nodes/bulk", post(handlers::post_nodes_bulk))
        .route("/v1/edges", post(handlers::post_edge))
        .route(
            "/v1/nodes/{id}",
            get(handlers::get_node).delete(handlers::delete_node),
        )
        .route(
            "/v1/nodes/{id}/embedding",
            get(handlers::get_node_embedding),
        )
        .route("/v1/nodes/{id}/tombstone", post(handlers::tombstone_node))
        .route(
            "/v1/retrieve",
            get(handlers::retrieve).post(handlers::retrieve_full),
        )
        .route(
            "/v1/branches",
            get(handlers::get_branches).post(handlers::post_branch),
        )
        .route("/v1/branches/{*name}", delete(handlers::delete_branch))
        .route("/v1/tags", get(handlers::get_tags).post(handlers::post_tag))
        .route("/v1/tags/{*name}", delete(handlers::delete_tag))
        .route("/v1/ingest", post(handlers_ingest::ingest))
        .route("/v1/explain", post(handlers::explain))
        // gap-09: `/v1/traverse_answer` is registered but gated by
        // `experimental.single_call_multihop` (default OFF). With the
        // flag off the handler returns 410 Gone + opt-in pointer; with
        // it on the full hop-loop runs. See routes/traverse.rs.
        .route(
            "/v1/traverse_answer",
            post(routes::traverse::traverse_answer),
        )
        // `/remote/v1/*` transport surface. Auth is enforced
        // per-handler via the `RequireBearer` extractor (see
        // crate::auth), not via a tower layer, so the read-open verbs
        // (`refs`, `fetch-blocks`) stay reachable without a token.
        //
        // BUG-49: Rate limiting is applied to the entire `/remote/v1/*`
        // group via a token-bucket middleware (100 req/s, burst 50).
        // The limiter is process-global (not per-IP) because remote
        // sync clients are trusted peers whose IP space is not
        // predictable. See `crate::remote_rate_limit` for tunability
        // via `MNEM_REMOTE_RATE_PER_SEC` / `MNEM_REMOTE_RATE_BURST`.
        .merge(remote_routes());
    if opts.metrics_enabled {
        // `/metrics` is intentionally NOT under `/v1/` so a Prometheus
        // scrape config that targets the canonical path works without
        // a per-service rewrite. The Prometheus convention wins here
        // over the schema-versioning we use for the v1 JSON surface.
        router = router.route("/metrics", get(metrics::metrics_handler));
    }
    // Layer order (applied outside-in in axum 0.8):
    //
    //   correlation_id   <- outermost; runs FIRST on every request,
    //                        LAST on every response. Mints / reuses
    //                        the id so track_metrics + handlers + the
    //                        `tower_http::trace` layer all see a span
    //                        with `correlation_id=...` attached.
    //   track_metrics    <- counts/times the request.
    //   DefaultBodyLimit <- 64 MiB cap (see MNEM_MAX_BODY_MB above).
    //   cors             <- permissive for v1 loopback deploy.
    //   TraceLayer       <- `tower_http` request/response tracing;
    //                        inherits our correlation_id span because
    //                        `Instrument` propagates.
    Ok(router
        .layer(axum::middleware::from_fn_with_state(
            state.clone(),
            metrics::track_metrics,
        ))
        .layer(DefaultBodyLimit::max(body_limit_bytes))
        // audit-2026-04-25 P2-6: rewrite axum's default 422 plain-text
        // Json<T> deserialize errors into the mnem.v1.err envelope so
        // clients never see a non-schema response.
        .layer(axum::middleware::from_fn(error::json_rejection_envelope))
        .layer(cors)
        .layer(TraceLayer::new_for_http())
        .layer(axum::middleware::from_fn(correlation::correlation_id))
        .with_state(state))
}

/// Build the rate-limited `/remote/v1/*` sub-router.
///
/// The four remote-protocol verbs are grouped into a dedicated
/// [`Router`] that has the [`remote_rate_limit::remote_rate_limit_middleware`]
/// applied via `axum::middleware::from_fn`. The limiter instance is
/// shared across all four routes via an [`Arc`] clone captured in the
/// closure; no heap allocation occurs per request beyond the `Arc`
/// refcount bump.
///
/// The resulting router is merged into the main router via
/// [`Router::merge`] so axum's standard route-dispatch table sees
/// the `/remote/v1/*` paths alongside `/v1/*`.
fn remote_routes() -> Router<AppState> {
    let limiter = Arc::new(remote_rate_limit::RemoteRateLimiter::from_env());
    Router::new()
        .route("/remote/v1/refs", get(routes::remote::get_refs))
        .route(
            "/remote/v1/fetch-blocks",
            post(routes::remote::post_fetch_blocks),
        )
        .route(
            "/remote/v1/push-blocks",
            post(routes::remote::post_push_blocks),
        )
        .route(
            "/remote/v1/advance-head",
            post(routes::remote::post_advance_head),
        )
        .layer(axum::middleware::from_fn(move |req, next| {
            let limiter = Arc::clone(&limiter);
            remote_rate_limit::remote_rate_limit_middleware(limiter, req, next)
        }))
}

/// Load `embed` section from `<data_dir>/config.toml` if it exists.
/// Returns `None` on any error so a malformed config never prevents
/// the server from starting; auto-embed just stays off.
fn load_embed_config(data_dir: &Path) -> Option<mnem_embed_providers::ProviderConfig> {
    #[derive(serde::Deserialize)]
    struct MiniCfg {
        embed: Option<mnem_embed_providers::ProviderConfig>,
    }
    let path = data_dir.join("config.toml");
    let s = std::fs::read_to_string(&path).ok()?;
    match toml::from_str::<MiniCfg>(&s) {
        Ok(parsed) => parsed.embed,
        Err(e) => {
            // A malformed [embed] section is a common misconfig; log
            // it so the operator can fix instead of silently running
            // without auto-embed.
            tracing::warn!(
                path = %path.display(),
                error = %e,
                "config.toml [embed] parse failed; auto-embed disabled"
            );
            None
        }
    }
}

/// Load `sparse` section from `<data_dir>/config.toml` if it exists.
/// When present, ingest paths auto-populate `Node.sparse_embed` and
/// retrieve paths auto-encode the query via the sparse provider. Same
/// "None on malformed config" policy as `load_embed_config`.
fn load_sparse_config(data_dir: &Path) -> Option<mnem_sparse_providers::ProviderConfig> {
    #[derive(serde::Deserialize)]
    struct MiniCfg {
        sparse: Option<mnem_sparse_providers::ProviderConfig>,
    }
    let path = data_dir.join("config.toml");
    let s = std::fs::read_to_string(&path).ok()?;
    match toml::from_str::<MiniCfg>(&s) {
        Ok(parsed) => parsed.sparse,
        Err(e) => {
            tracing::warn!(
                path = %path.display(),
                error = %e,
                "config.toml [sparse] parse failed; sparse auto-encode disabled"
            );
            None
        }
    }
}

/// Load `ner` section from `<data_dir>/config.toml` if it exists.
/// `None` means ingest paths will use `NerConfig::Rule` (the default).
/// Also respects `MNEM_NER_PROVIDER` env var: "none" → `NerConfig::None`,
/// any other value → `NerConfig::Rule`.
fn load_ner_config(data_dir: &Path) -> Option<mnem_ingest::NerConfig> {
    if let Ok(p) = std::env::var("MNEM_NER_PROVIDER") {
        return Some(match p.to_ascii_lowercase().as_str() {
            "none" => mnem_ingest::NerConfig::None,
            _ => mnem_ingest::NerConfig::Rule,
        });
    }
    #[derive(serde::Deserialize)]
    struct MiniCfg {
        ner: Option<mnem_ingest::NerConfig>,
    }
    let path = data_dir.join("config.toml");
    let s = std::fs::read_to_string(&path).ok()?;
    match toml::from_str::<MiniCfg>(&s) {
        Ok(parsed) => parsed.ner,
        Err(e) => {
            tracing::warn!(
                path = %path.display(),
                error = %e,
                "config.toml [ner] parse failed; NER defaults to rule-based"
            );
            None
        }
    }
}
