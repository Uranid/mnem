//! `mnem http` - stand-alone binary that serves the HTTP JSON API.
//!
//! ```shell
//! mnem http --repo /path/to/project --bind 127.0.0.1:9876
//! ```
//!
//! Binds to loopback by default; exposing to a network interface emits a
//! loud stderr warning because this binary has no auth layer in v1.

use std::net::SocketAddr;
use std::path::PathBuf;

use anyhow::Result;
use clap::Parser;

#[derive(Parser)]
#[command(
    name = "mnem http",
    version,
    about = "HTTP JSON API for mnem.",
    long_about = None
)]
struct Cli {
    /// Directory containing `.mnem/` (auto-init if missing).
    #[arg(long, short = 'R', default_value = ".")]
    repo: PathBuf,
    /// Bind address. Use 0.0.0.0 to expose over the network (warned).
    #[arg(long, default_value = "127.0.0.1:9876")]
    bind: SocketAddr,
    /// Use an ephemeral in-memory store instead of redb. All commits
    /// live in RAM and are lost on process exit. Intended for
    /// benchmark harnesses and ephemeral agent sessions where redb
    /// fsync latency is not worth the durability (30-40x commit
    /// speedup on append-heavy workloads per internal benchmarking
    /// findings). NEVER enable in a durable deployment.
    #[arg(long)]
    in_memory: bool,
    /// Enable the Prometheus `/metrics` endpoint.
    ///
    /// Default behaviour (matching the R1 observability scorer's H3
    /// recommendation): ON for non-loopback binds so a production
    /// deployment gets metrics out of the box; OFF for loopback binds
    /// so a developer's local `mnem http` does not expose scrape data
    /// to other processes on their machine unless they opt in.
    ///
    /// The in-process counters always run (they are lock-free atomics
    /// with near-zero cost) so flipping this flag on at the next
    /// restart produces a fresh baseline without recompilation.
    #[arg(long)]
    metrics: bool,
    /// Disable the `/metrics` endpoint even when it would otherwise
    /// be ON by default (i.e. on non-loopback binds). Use this for
    /// minimal-surface deployments fronted by their own metrics pipe.
    #[arg(long, conflicts_with = "metrics")]
    no_metrics: bool,
}

#[tokio::main]
async fn main() -> Result<()> {
    init_tracing();

    let cli = Cli::parse();
    // Refuse to bind non-loopback without an explicit opt-in. mnem-http
    // has no auth layer in v1, so an accidental `--bind 0.0.0.0:9876`
    // on a laptop with a public interface, or inside a dev container
    // with host networking, is a data-exfiltration footgun. A loud
    // printed warning is not enough: operators copy/paste start
    // commands without reading them. Require the env var so the
    // choice to expose publicly is deliberate, documented, and
    // greppable.
    if !cli.bind.ip().is_loopback() && std::env::var_os("MNEM_HTTP_ALLOW_NON_LOOPBACK").is_none() {
        eprintln!(
            "mnem http: refusing to bind non-loopback address {} without an explicit opt-in.\n\
             \n\
             mnem http has NO authentication layer in v1. Binding to a non-loopback address\n\
             (like 0.0.0.0 or a LAN IP) exposes every node, every retrieval, and every\n\
             write endpoint to anyone who can reach the interface.\n\
             \n\
             If you really do want to bind publicly (e.g. behind a reverse proxy that\n\
             adds auth), set MNEM_HTTP_ALLOW_NON_LOOPBACK=1 in the environment:\n\
             \n\
             \tMNEM_HTTP_ALLOW_NON_LOOPBACK=1 mnem http --bind {}\n\
             \n\
             Loopback (127.0.0.1, ::1) needs no flag.\n\
             \n\
             hint: see docs/RUNBOOK.md#4-auth-refused-on-non-loopback-bind for the full\n\
             remediation walkthrough (proxy setup, opt-in env var, logging posture).",
            cli.bind, cli.bind,
        );
        std::process::exit(2);
    }
    if !cli.bind.ip().is_loopback() {
        eprintln!(
            "mnem http: binding to non-loopback {} under MNEM_HTTP_ALLOW_NON_LOOPBACK. \
             There is NO auth layer; front this with a reverse proxy that adds one.",
            cli.bind
        );
    }

    // Gate the Prometheus endpoint. Policy (from R1 observability):
    //
    // - --metrics     -> force ON (developer explicitly wants it).
    // - --no-metrics  -> force OFF (operator explicitly opts out).
    // - default       -> ON, regardless of bind address.
    //
    // audit-2026-04-25 R1 (Stage E re-fix): the previous default
    // gated /metrics off when bound to loopback. That silently
    // broke local Prometheus scrape configs, smoke tests, and the
    // pre-P2-7 always-on contract that tooling relied on. Operators
    // who want minimal surface pass `--no-metrics` explicitly.
    //
    // The underlying counter state always exists; this flag toggles
    // whether the HTTP route is mounted.
    let metrics_enabled = !cli.no_metrics;
    // `--metrics` is now redundant with the default; we still accept
    // it so existing scripts don't break.
    let _ = cli.metrics;

    let app = mnem_http::app_with_options(
        &cli.repo,
        mnem_http::AppOptions {
            allow_labels: None,
            in_memory: cli.in_memory,
            metrics_enabled,
            push_token: None,
        },
    )?;
    let listener = tokio::net::TcpListener::bind(cli.bind).await?;
    println!("mnem http listening on http://{}", cli.bind);
    // audit-2026-04-25 P2-7: enumerate every mounted route from the
    // single source of truth (mnem_http::route_table) so the banner
    // and the router can never drift apart again.
    for (method, path, brief) in mnem_http::route_table(metrics_enabled) {
        println!("  {method:<10} {path:<32} {brief}");
    }

    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await?;
    Ok(())
}

/// Initialise the `tracing_subscriber` log formatter.
///
/// Honours two env vars:
///
/// - `RUST_LOG` (standard): directive filter. Falls back to
///   `mnem_http=info,tower_http=warn` when unset.
/// - `MNEM_LOG_FORMAT`: `text` (default) emits human-friendly
///   terminal output; `json` emits one JSON object per log event for
///   Loki / Elastic / Vector ingestion. Any other value is treated as
///   `text` (with a one-line stderr warning so operators can fix
///   their config).
///
/// See `docs/LOGGING.md` for the wire contract.
fn init_tracing() {
    let env_filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| "mnem_http=info,tower_http=warn".into());

    let fmt = std::env::var("MNEM_LOG_FORMAT")
        .unwrap_or_else(|_| "text".to_string())
        .to_ascii_lowercase();
    match fmt.as_str() {
        "json" => {
            // One JSON object per event. Every event nested under a
            // request span gets the full parent chain rendered under
            // `spans[]` so `correlation_id` from the outermost
            // `http_request` span is visible on every downstream log
            // line (tower_http's `request` span, handler spans, etc.).
            // A jq filter like
            //   jq 'select(.spans[]?.correlation_id == "req-...")'
            // walks one request's full lifecycle.
            tracing_subscriber::fmt()
                .with_env_filter(env_filter)
                .json()
                .with_current_span(true)
                .with_span_list(true)
                .init();
        }
        "text" => {
            tracing_subscriber::fmt().with_env_filter(env_filter).init();
        }
        other => {
            eprintln!(
                "mnem http: unrecognised MNEM_LOG_FORMAT={other:?}; falling back to `text`. Valid values: text | json."
            );
            tracing_subscriber::fmt().with_env_filter(env_filter).init();
        }
    }
}

async fn shutdown_signal() {
    let ctrl_c = async {
        tokio::signal::ctrl_c()
            .await
            .expect("install Ctrl-C handler");
    };
    #[cfg(unix)]
    let terminate = async {
        tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            .expect("install SIGTERM handler")
            .recv()
            .await;
    };
    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        () = ctrl_c => {},
        () = terminate => {},
    }
    println!("mnem http: shutdown signal; draining...");
}
