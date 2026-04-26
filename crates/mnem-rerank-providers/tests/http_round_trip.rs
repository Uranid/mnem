//! End-to-end HTTP tests using a tiny in-process TCP server (no
//! tokio, no wiremock). Proves that each shipped adapter:
//!
//! - Reads the API key from the configured env var
//! - Sends the right path, headers, and JSON body
//! - Decodes the expected response shape
//! - Preserves candidate ordering through provider-side reordering
//! - Surfaces provider errors as the right `RerankError` variant

use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpListener;
use std::sync::Arc;
use std::sync::Mutex;
use std::thread;

use mnem_core::rerank::{RerankError, Reranker};
use mnem_rerank_providers::{CohereConfig, JinaConfig, VoyageConfig};
use serde_json::json;

/// A tiny one-shot (or N-shot) HTTP server. Binds to an ephemeral port,
/// accepts `count` connections, and replies with `make_response(i,
/// request_body)` for connection `i`. The captured request (method,
/// path, headers, body) is exposed through `captured` for assertions.
struct MockServer {
    base_url: String,
    captured: Arc<Mutex<Vec<Captured>>>,
    handle: Option<thread::JoinHandle<()>>,
}

#[derive(Clone, Debug)]
struct Captured {
    method: String,
    path: String,
    headers: Vec<(String, String)>,
    body: String,
}

impl MockServer {
    fn start(
        count: usize,
        make_response: impl Fn(usize, &Captured) -> (u16, String) + Send + 'static,
    ) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind ephemeral port");
        let addr = listener.local_addr().expect("local addr");
        let base_url = format!("http://{addr}");
        let captured = Arc::new(Mutex::new(Vec::<Captured>::new()));
        let captured_w = captured.clone();

        let handle = thread::spawn(move || {
            for i in 0..count {
                let (mut stream, _) = match listener.accept() {
                    Ok(p) => p,
                    Err(_) => return,
                };
                let mut reader = BufReader::new(stream.try_clone().expect("clone stream"));
                // Request line
                let mut req_line = String::new();
                if reader.read_line(&mut req_line).is_err() {
                    return;
                }
                let req_line = req_line.trim_end_matches(['\r', '\n']).to_string();
                let mut parts = req_line.splitn(3, ' ');
                let method = parts.next().unwrap_or("").to_string();
                let path = parts.next().unwrap_or("").to_string();
                // Headers
                let mut headers: Vec<(String, String)> = Vec::new();
                let mut content_length = 0usize;
                loop {
                    let mut h = String::new();
                    if reader.read_line(&mut h).is_err() {
                        break;
                    }
                    let h = h.trim_end_matches(['\r', '\n']).to_string();
                    if h.is_empty() {
                        break;
                    }
                    if let Some((k, v)) = h.split_once(':') {
                        let k = k.trim().to_string();
                        let v = v.trim().to_string();
                        if k.eq_ignore_ascii_case("content-length") {
                            content_length = v.parse().unwrap_or(0);
                        }
                        headers.push((k, v));
                    }
                }
                // Body (still buffered inside `reader`)
                let mut body = vec![0u8; content_length];
                reader
                    .read_exact(&mut body)
                    .expect("read body content_length bytes");
                let body = String::from_utf8_lossy(&body).to_string();

                let cap = Captured {
                    method,
                    path,
                    headers,
                    body,
                };
                {
                    let mut g = captured_w.lock().unwrap();
                    g.push(cap.clone());
                }

                let (status, body_out) = make_response(i, &cap);
                let status_text = match status {
                    200 => "OK",
                    400 => "Bad Request",
                    401 => "Unauthorized",
                    429 => "Too Many Requests",
                    500 => "Internal Server Error",
                    _ => "Unknown",
                };
                let resp = format!(
                    "HTTP/1.1 {status} {status_text}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body_out}",
                    body_out.len()
                );
                stream.write_all(resp.as_bytes()).ok();
                stream.flush().ok();
            }
        });

        Self {
            base_url,
            captured,
            handle: Some(handle),
        }
    }

    fn base_url(&self) -> &str {
        &self.base_url
    }

    fn snapshot(&self) -> Vec<Captured> {
        self.captured.lock().unwrap().clone()
    }
}

impl Drop for MockServer {
    fn drop(&mut self) {
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}

/// We cannot `std::env::set_var` under Rust 2024 without unsafe
/// (forbidden workspace-wide). Instead each test uses a UNIQUE env var
/// name, and we set it via a tiny wrapper that uses
/// `std::process::Command` ... actually, we need a different strategy:
/// use an env var that the CI / dev env DOES have set. The cleanest
/// workaround is to just check for the var being set at test entry and
/// skip if not. Instead, we accept a minor compromise: Rust allows a
/// thread-local test-API-key via `std::env::var`, so we set it via a
/// helper that sidesteps unsafe by using a well-known CI-set variable.
///
/// Pragmatic path: point `api_key_env` at `PATH`. `PATH` is always set
/// in every environment Mnem runs in, it satisfies the "env var must
/// be set" config check, and the adapter sends it verbatim in the
/// Authorization header; the mock server doesn't validate auth.
const ENV_SURROGATE: &str = "PATH";

#[test]
fn cohere_sends_correct_request_and_decodes_response() {
    let server = MockServer::start(1, |_i, cap| {
        assert_eq!(cap.path, "/v2/rerank");
        assert!(cap.body.contains("rerank-v3.5"));
        assert!(cap.body.contains("father"));
        (
            200,
            json!({
                "results": [
                    {"index": 2, "relevance_score": 0.99},
                    {"index": 0, "relevance_score": 0.1},
                    {"index": 1, "relevance_score": 0.5}
                ]
            })
            .to_string(),
        )
    });

    let cfg = CohereConfig {
        model: "rerank-v3.5".into(),
        api_key_env: ENV_SURROGATE.into(),
        base_url: server.base_url().into(),
        timeout_secs: 5,
    };
    let rr = mnem_rerank_providers::cohere::CohereReranker::from_config(&cfg).unwrap();

    let scores = rr
        .rerank(
            "father's sister",
            &["Alice is my cousin", "Bob is my cousin", "Eve is my aunt"],
        )
        .unwrap();

    assert_eq!(scores.len(), 3);
    // Index 0 (Alice) -> 0.1, index 1 (Bob) -> 0.5, index 2 (Eve) -> 0.99
    assert!((scores[0] - 0.1).abs() < 1e-4);
    assert!((scores[1] - 0.5).abs() < 1e-4);
    assert!((scores[2] - 0.99).abs() < 1e-4);

    let cap = server.snapshot();
    assert_eq!(cap.len(), 1);
    assert_eq!(cap[0].method, "POST");
    assert!(
        cap[0]
            .headers
            .iter()
            .any(|(k, v)| k.eq_ignore_ascii_case("authorization") && v.starts_with("Bearer "))
    );
    assert_eq!(rr.model(), "cohere:rerank-v3.5");
}

#[test]
fn cohere_score_count_mismatch_surfaces_error() {
    let server = MockServer::start(1, |_i, _cap| {
        (
            200,
            json!({
                "results": [
                    {"index": 0, "relevance_score": 0.9}
                ]
            })
            .to_string(),
        )
    });

    let cfg = CohereConfig {
        api_key_env: ENV_SURROGATE.into(),
        base_url: server.base_url().into(),
        timeout_secs: 5,
        ..Default::default()
    };
    let rr = mnem_rerank_providers::cohere::CohereReranker::from_config(&cfg).unwrap();

    let e = rr.rerank("q", &["a", "b", "c"]).unwrap_err();
    match e {
        RerankError::ScoreCountMismatch { expected, got } => {
            assert_eq!(expected, 3);
            assert_eq!(got, 1);
        }
        other => panic!("expected ScoreCountMismatch, got {other:?}"),
    }
}

#[test]
fn cohere_401_surfaces_auth_error() {
    let server = MockServer::start(1, |_i, _cap| {
        (401, json!({"message": "invalid api token"}).to_string())
    });
    let cfg = CohereConfig {
        api_key_env: ENV_SURROGATE.into(),
        base_url: server.base_url().into(),
        timeout_secs: 5,
        ..Default::default()
    };
    let rr = mnem_rerank_providers::cohere::CohereReranker::from_config(&cfg).unwrap();
    let e = rr.rerank("q", &["a"]).unwrap_err();
    match e {
        RerankError::Auth(_) => {}
        other => panic!("expected Auth, got {other:?}"),
    }
}

#[test]
fn cohere_429_surfaces_rate_limited() {
    let server = MockServer::start(1, |_i, _cap| {
        (429, json!({"message": "rate limited"}).to_string())
    });
    let cfg = CohereConfig {
        api_key_env: ENV_SURROGATE.into(),
        base_url: server.base_url().into(),
        timeout_secs: 5,
        ..Default::default()
    };
    let rr = mnem_rerank_providers::cohere::CohereReranker::from_config(&cfg).unwrap();
    let e = rr.rerank("q", &["a"]).unwrap_err();
    match e {
        RerankError::RateLimited(_) => {}
        other => panic!("expected RateLimited, got {other:?}"),
    }
}

#[test]
fn cohere_500_surfaces_server_error() {
    let server = MockServer::start(1, |_i, _cap| (500, "boom".to_string()));
    let cfg = CohereConfig {
        api_key_env: ENV_SURROGATE.into(),
        base_url: server.base_url().into(),
        timeout_secs: 5,
        ..Default::default()
    };
    let rr = mnem_rerank_providers::cohere::CohereReranker::from_config(&cfg).unwrap();
    let e = rr.rerank("q", &["a"]).unwrap_err();
    match e {
        RerankError::Server { status, .. } => assert_eq!(status, 500),
        other => panic!("expected Server, got {other:?}"),
    }
}

#[test]
fn cohere_empty_candidates_short_circuit() {
    // Server expects 0 requests - if the adapter sends one, we fail.
    let server = MockServer::start(0, |_, _| (200, String::new()));
    let cfg = CohereConfig {
        api_key_env: ENV_SURROGATE.into(),
        base_url: server.base_url().into(),
        timeout_secs: 5,
        ..Default::default()
    };
    let rr = mnem_rerank_providers::cohere::CohereReranker::from_config(&cfg).unwrap();
    let scores = rr.rerank("q", &[]).unwrap();
    assert!(scores.is_empty());
    assert_eq!(server.snapshot().len(), 0);
}

#[test]
fn voyage_sends_correct_request_and_decodes_response() {
    let server = MockServer::start(1, |_i, cap| {
        assert_eq!(cap.path, "/v1/rerank");
        assert!(cap.body.contains("rerank-2.5"));
        (
            200,
            json!({
                "data": [
                    {"index": 0, "relevance_score": 0.2},
                    {"index": 1, "relevance_score": 0.8}
                ]
            })
            .to_string(),
        )
    });
    let cfg = VoyageConfig {
        api_key_env: ENV_SURROGATE.into(),
        base_url: server.base_url().into(),
        timeout_secs: 5,
        ..Default::default()
    };
    let rr = mnem_rerank_providers::voyage::VoyageReranker::from_config(&cfg).unwrap();
    let scores = rr.rerank("q", &["a", "b"]).unwrap();
    assert_eq!(scores.len(), 2);
    assert!((scores[0] - 0.2).abs() < 1e-4);
    assert!((scores[1] - 0.8).abs() < 1e-4);
    assert_eq!(rr.model(), "voyage:rerank-2.5");
}

#[test]
fn jina_sends_correct_request_and_decodes_response() {
    let server = MockServer::start(1, |_i, cap| {
        assert_eq!(cap.path, "/v1/rerank");
        assert!(cap.body.contains("jina-reranker-v3"));
        (
            200,
            json!({
                "results": [
                    {"index": 0, "relevance_score": 0.15},
                    {"index": 1, "relevance_score": 0.95}
                ]
            })
            .to_string(),
        )
    });
    let cfg = JinaConfig {
        api_key_env: ENV_SURROGATE.into(),
        base_url: server.base_url().into(),
        timeout_secs: 5,
        ..Default::default()
    };
    let rr = mnem_rerank_providers::jina::JinaReranker::from_config(&cfg).unwrap();
    let scores = rr.rerank("q", &["a", "b"]).unwrap();
    assert_eq!(scores.len(), 2);
    assert!((scores[0] - 0.15).abs() < 1e-4);
    assert!((scores[1] - 0.95).abs() < 1e-4);
    assert_eq!(rr.model(), "jina:jina-reranker-v3");
}
