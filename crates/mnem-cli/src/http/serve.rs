//! `mnem http`, HTTP JSON API entry point inside the unified `mnem` binary.
//!
//! After merge, `mnem http` replaces the standalone `mnem-http` binary.

use std::net::SocketAddr;
use std::path::PathBuf;

use anyhow::Result;

/// Serve mnem as an HTTP JSON API server.
#[derive(clap::Parser)]
pub(crate) struct ServeArgs {
    /// Directory containing `.mnem/` (auto-init if missing).
    #[arg(long, short = 'R', default_value = ".")]
    repo: PathBuf,
    /// Bind address. Use 0.0.0.0 to expose over the network (warned).
    #[arg(long, default_value = "127.0.0.1:9876")]
    bind: SocketAddr,
    /// Use an ephemeral in-memory store instead of redb.
    #[arg(long)]
    in_memory: bool,
    /// Force the `/metrics` endpoint ON.
    #[arg(long)]
    metrics: bool,
    /// Force the `/metrics` endpoint OFF.
    #[arg(long, conflicts_with = "metrics")]
    no_metrics: bool,
}

pub(crate) fn run(args: ServeArgs) -> Result<()> {
    init_tracing();

    if !args.bind.ip().is_loopback() && std::env::var_os("MNEM_HTTP_ALLOW_NON_LOOPBACK").is_none() {
        eprintln!(
            "mnem http: refusing to bind non-loopback address {} without explicit opt-in.\n\
             mnem http has NO authentication layer in v1. Set MNEM_HTTP_ALLOW_NON_LOOPBACK=1 to bypass.",
            args.bind
        );
        std::process::exit(2);
    }

    let metrics_enabled = !args.no_metrics;
    let app = mnem_http::app_with_options(
        &args.repo,
        mnem_http::AppOptions {
            allow_labels: None,
            in_memory: args.in_memory,
            metrics_enabled,
            push_token: None,
        },
    )?;

    let runtime = tokio::runtime::Runtime::new()?;
    runtime.block_on(async {
        let listener = tokio::net::TcpListener::bind(args.bind).await?;
        println!("mnem http listening on http://{}", args.bind);
        for (method, path, brief) in mnem_http::route_table(metrics_enabled) {
            println!(" {method:<10} {path:<32} {brief}");
        }
        axum::serve(listener, app)
            .with_graceful_shutdown(shutdown_signal())
            .await?;
        Ok::<(), anyhow::Error>(())
    })?;
    Ok(())
}

fn init_tracing() {
    #[allow(unused_imports)]
    use tracing_subscriber::{EnvFilter, fmt};
    let env_filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| "mnem_http=info,tower_http=warn".into());
    let fmt = std::env::var("MNEM_LOG_FORMAT")
        .unwrap_or_else(|_| "text".to_string())
        .to_ascii_lowercase();
    match fmt.as_str() {
        "json" => tracing_subscriber::fmt()
            .with_env_filter(env_filter)
            .json()
            .with_current_span(true)
            .with_span_list(true)
            .init(),
        "text" => tracing_subscriber::fmt().with_env_filter(env_filter).init(),
        other => {
            eprintln!("mnem http: unrecognised MNEM_LOG_FORMAT={other:?}; falling back to `text`");
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
    tokio::select! { () = ctrl_c => {}, () = terminate => {} }
    println!("mnem http: shutdown signal; draining...");
}
