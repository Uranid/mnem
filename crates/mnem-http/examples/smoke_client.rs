//! End-to-end smoke-test of the `mnem-http` router.
//!
//! Spawns the server on an ephemeral loopback port inside the example's
//! own tokio runtime, then fires three requests at it over a plain TCP
//! socket with `ureq` (plain HTTP, no TLS). Asserts the response
//! shapes match the v1 schema that downstream clients code against.
//!
//! This is deliberately a "real" HTTP round-trip (not `tower::oneshot`)
//! so it covers the full stack: axum routing, serde, the
//! correlation-id middleware, the body-size layer, the whole lot.
//!
//! See also:
//! - `docs/guide/cli.md` - the CLI wrappers over these endpoints.
//! - `docs/LOGGING.md` - field glossary for the tracing output.
//! - - why tokio lives in this crate only.
//!
//! Run:
//! ```console
//! cargo run --example smoke_client -p mnem-http
//! ```

use std::net::{SocketAddr, TcpListener as StdTcpListener};

use serde_json::Value;
use tempfile::TempDir;
use tokio::net::TcpListener;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Reserve a free loopback port via std's listener, then drop it so
    // tokio can rebind. Using port 0 directly inside the runtime means
    // we can't tell the client which port to hit without a channel.
    let std_listener = StdTcpListener::bind("127.0.0.1:0")?;
    let addr: SocketAddr = std_listener.local_addr()?;
    drop(std_listener);

    // Fresh repo in a tempdir; the router will auto-init `.mnem/`.
    let td = TempDir::new()?;
    let opts = mnem_http::AppOptions {
        allow_labels: Some(true),
        in_memory: true,
        metrics_enabled: false,
    };
    let repo_path = td.path().to_path_buf();

    // Spawn the server in its own tokio runtime on a dedicated thread
    // so the synchronous `ureq` calls on the main thread are not
    // contending with axum's executor on the same worker pool.
    let server_handle = std::thread::spawn(
        move || -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
            let rt = tokio::runtime::Runtime::new()?;
            rt.block_on(async move {
                let app = mnem_http::app_with_options(&repo_path, opts)?;
                let listener = TcpListener::bind(addr).await?;
                axum::serve(listener, app).await?;
                Ok::<_, Box<dyn std::error::Error + Send + Sync>>(())
            })
        },
    );

    // Give axum a tick to start listening. Binding is synchronous under
    // the hood, but the `serve` future may not have looped once yet.
    std::thread::sleep(std::time::Duration::from_millis(200));
    let base = format!("http://{addr}");
    println!("server up at {base}");

    // ---- 1. GET /v1/healthz ----
    let body: String = ureq::get(&format!("{base}/v1/healthz"))
        .call()?
        .into_string()?;
    let j: Value = serde_json::from_str(&body)?;
    assert_eq!(j["schema"], "mnem.v1.healthz");
    assert_eq!(j["ok"], true);
    println!("GET /v1/healthz: ok=true schema=mnem.v1.healthz");

    // ---- 2. POST /v1/nodes ----
    let post = ureq::post(&format!("{base}/v1/nodes"))
        .set("content-type", "application/json")
        .send_string(
            r#"{"author":"smoke","message":"seed","label":"Memory","summary":"hello from the smoke-client example"}"#,
        )?;
    assert_eq!(post.status(), 200);
    let post_body: Value = serde_json::from_str(&post.into_string()?)?;
    assert_eq!(post_body["schema"], "mnem.v1.post-node");
    let node_id = post_body["id"].as_str().unwrap().to_string();
    println!("POST /v1/nodes: id={node_id}");

    // ---- 3. GET /v1/retrieve with no text (no-embedder path) ----
    // With `allow_labels=true` the server honours the `label` filter;
    // we bounded the retrieve to a single label so the shape stays
    // tight regardless of what other defaults exist.
    let ret = ureq::get(&format!("{base}/v1/retrieve?label=Memory&limit=5"))
        .call()?
        .into_string()?;
    let rj: Value = serde_json::from_str(&ret)?;
    assert!(
        rj["items"].is_array(),
        "retrieve response must carry items[]"
    );
    assert_eq!(rj["schema"], "mnem.v1.retrieve");
    println!(
        "GET /v1/retrieve: {} item(s) returned",
        rj["items"].as_array().map_or(0, Vec::len)
    );

    // We let the runtime drop, which aborts the axum task. The detach
    // is intentional; joining means waiting for graceful shutdown, but
    // the example has done what it came to do.
    drop(server_handle);
    println!("OK");
    Ok(())
}
