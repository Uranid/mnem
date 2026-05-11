//! `POST /v1/ingest` - upload a source file (or JSON body with inline
//! text) and commit the resulting Doc + Chunk + Entity subgraph.
//!
//! Phase-B5d, HTTP half. Accepts one of two request shapes:
//!
//! - **`multipart/form-data`**: a `file` field carrying the raw source
//!   bytes, plus optional `ntype`, `chunker`, `max_tokens`, `overlap`
//!   text fields.
//! - **JSON body**: `{"text": "...", "ntype": "...", ...}` for
//!   callers that already have the payload in memory (typical for
//!   agent-driven ingests that don't want to round-trip through a
//!   temp file).
//!
//! The multipart shape is the primary surface; the JSON shape exists
//! so a pure-REST client (e.g. Postman snippet in a demo) can exercise
//! the path without a separate file upload.
//!
//! ## Size + tokens clamp
//!
//! File size is capped at 32 MiB (`MNEM_HTTP_INGEST_MAX_BYTES`),
//! `max_tokens` at 8192. Both mirror the MCP handler so a request
//! that migrates between transports sees the same ceiling. The cap
//! is a DoS guardrail, not a product shape; raise it in env if you
//! have a legitimate use case.

use std::time::Instant;

use axum::Json;
use axum::extract::{Multipart, State};
use mnem_ingest::{
    ChunkerAuto, ChunkerKind, IngestConfig, Ingester, NerConfig, SourceKind, auto_chunker,
};
use serde::Deserialize;
use serde_json::{Value, json};

use crate::error::Error;
use crate::state::AppState;

/// Upper bound on `max_tokens` accepted on `/v1/ingest`. Mirrors the
/// CLI + MCP ceilings.
pub(crate) const MAX_INGEST_TOKENS: u32 = 8192;

/// Default `MNEM_HTTP_INGEST_MAX_BYTES` cap on the raw file size.
/// 32 MiB covers typical Markdown / conversation exports / mid-sized
/// PDFs; operators hit the env override before the product shape ever
/// needs to change.
pub(crate) const DEFAULT_MAX_INGEST_BYTES: u64 = 32 * 1024 * 1024;

/// Resolve the per-request file-size cap. Reads `MNEM_HTTP_INGEST_MAX_BYTES`
/// on every request so an operator can raise the ceiling without a
/// restart; failure to parse falls back to [`DEFAULT_MAX_INGEST_BYTES`].
fn max_ingest_bytes() -> u64 {
    std::env::var("MNEM_HTTP_INGEST_MAX_BYTES")
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(DEFAULT_MAX_INGEST_BYTES)
}

/// JSON body for the non-multipart shape. Callers that already have
/// the payload in memory POST it under `Content-Type: application/json`
/// instead of a file upload.
#[derive(Deserialize, Debug)]
pub(crate) struct IngestJsonBody {
    /// UTF-8 text of the source. Interpreted as `SourceKind::Text`
    /// unless `kind` is set.
    pub text: String,
    /// Optional explicit source kind (`"markdown" | "text" | "pdf" |
    /// "conversation"`). Defaults to `text`.
    #[serde(default)]
    pub kind: Option<String>,
    /// Override the Doc root `ntype`. Defaults to `"Doc"`.
    #[serde(default)]
    pub ntype: Option<String>,
    /// `auto | paragraph | recursive | session`. Defaults to `auto`.
    #[serde(default)]
    pub chunker: Option<String>,
    /// Target tokens per chunk. Clamped at [`MAX_INGEST_TOKENS`].
    #[serde(default)]
    pub max_tokens: Option<u32>,
    /// Overlap tokens between adjacent chunks (recursive chunker).
    #[serde(default)]
    pub overlap: Option<u32>,
    /// Commit author. Required.
    pub author: String,
    /// Commit message. Optional; default `"mnem http ingest"`.
    #[serde(default)]
    pub message: Option<String>,
    /// Extractor selector. `"none"` (default) keeps the rule-based
    /// [`mnem_ingest::RuleExtractor`]. `"keybert"` swaps in the
    /// statistical adapter, driven by the server's configured
    /// embedder. C3 FIX-3.
    #[serde(default)]
    pub extractor: Option<String>,
    /// NER provider. `"rule"` (default) uses the capitalized-phrase
    /// heuristic. `"none"` suppresses all entity extraction.
    /// Overrides the server's configured `[ner]` section for this
    /// request only.
    #[serde(default)]
    pub ner_provider: Option<String>,
}

/// `POST /v1/ingest` dispatcher. Detects the request's
/// `Content-Type`; multipart goes through [`ingest_multipart`], JSON
/// through [`ingest_json`]. Unrecognised content types fall back to
/// multipart parsing (which will surface a clear error on non-multipart
/// bodies).
pub(crate) async fn ingest(
    State(state): State<AppState>,
    multipart_or_json: axum::http::Request<axum::body::Body>,
) -> Result<Json<Value>, Error> {
    let content_type = multipart_or_json
        .headers()
        .get(axum::http::header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("")
        .to_ascii_lowercase();

    if content_type.starts_with("application/json") {
        let limit = max_ingest_bytes() as usize;
        let bytes = axum::body::to_bytes(multipart_or_json.into_body(), limit)
            .await
            .map_err(|e| {
                // `to_bytes` wraps a `http_body_util::LengthLimitError` as
                // its source when the body exceeds the supplied cap.
                // Detect that via the source chain so the client receives
                // 413 Payload Too Large rather than a generic 400.
                use std::error::Error as StdError;
                let is_length_limit = e
                    .source()
                    .map_or(false, |src| src.to_string().contains("length limit"));
                if is_length_limit {
                    Error::status(
                        axum::http::StatusCode::PAYLOAD_TOO_LARGE,
                        format!(
                            "JSON body exceeds the {limit}-byte cap \
                             (raise MNEM_HTTP_INGEST_MAX_BYTES if legitimate)"
                        ),
                    )
                } else {
                    Error::bad_request(format!("reading body: {e}"))
                }
            })?;
        let body: IngestJsonBody = serde_json::from_slice(&bytes)
            .map_err(|e| Error::bad_request(format!("malformed JSON body: {e}")))?;
        ingest_json(state, body).await
    } else {
        // Every other content-type path (multipart/form-data with
        // boundary, empty, anything else) falls through to the
        // multipart parser, which returns a clean 400 on a malformed
        // body.
        let multipart = Multipart::from_request(multipart_or_json, &state)
            .await
            .map_err(|e| Error::bad_request(format!("multipart decode: {e}")))?;
        ingest_multipart(state, multipart).await
    }
}

/// Multipart variant. Expects a `file` field plus optional text fields.
async fn ingest_multipart(state: AppState, mut multipart: Multipart) -> Result<Json<Value>, Error> {
    let mut file_bytes: Option<Vec<u8>> = None;
    let mut file_name: Option<String> = None;
    let mut ntype: Option<String> = None;
    let mut chunker_str: Option<String> = None;
    let mut max_tokens: Option<u32> = None;
    let mut overlap: Option<u32> = None;
    let mut author: Option<String> = None;
    let mut message: Option<String> = None;
    let mut extractor: Option<String> = None;
    let mut ner_provider: Option<String> = None;

    let max_bytes = max_ingest_bytes();

    while let Some(mut field) = multipart
        .next_field()
        .await
        .map_err(|e| Error::bad_request(format!("multipart field: {e}")))?
    {
        let name = field.name().unwrap_or("").to_string();
        match name.as_str() {
            "file" => {
                file_name = field.file_name().map(ToString::to_string);
                // Stream chunks instead of buffering the whole field
                // before checking the size (BUG-9): abort as soon as
                // the accumulated length exceeds the cap so we never
                // hold more than `max_bytes + one chunk` in RAM.
                let max_bytes_usize = max_bytes as usize;
                let mut buf = Vec::with_capacity(1024 * 1024); // 1 MiB initial
                loop {
                    match field.chunk().await {
                        Ok(Some(chunk)) => {
                            if buf.len() + chunk.len() > max_bytes_usize {
                                return Err(Error::status(
                                    axum::http::StatusCode::PAYLOAD_TOO_LARGE,
                                    "upload too large (max 33 MiB)".to_string(),
                                ));
                            }
                            buf.extend_from_slice(&chunk);
                        }
                        Ok(None) => break,
                        Err(e) => {
                            return Err(Error::bad_request(format!("reading file field: {e}")));
                        }
                    }
                }
                file_bytes = Some(buf);
            }
            "ntype" => ntype = Some(field_text(field).await?),
            "chunker" => chunker_str = Some(field_text(field).await?),
            "max_tokens" => {
                let s = field_text(field).await?;
                max_tokens = Some(
                    s.parse::<u32>()
                        .map_err(|e| Error::bad_request(format!("max_tokens: {e}")))?,
                );
            }
            "overlap" => {
                let s = field_text(field).await?;
                overlap = Some(
                    s.parse::<u32>()
                        .map_err(|e| Error::bad_request(format!("overlap: {e}")))?,
                );
            }
            "author" => author = Some(field_text(field).await?),
            "message" => message = Some(field_text(field).await?),
            "extractor" => extractor = Some(field_text(field).await?),
            "ner_provider" => ner_provider = Some(field_text(field).await?),
            other => {
                // Ignore unknown fields rather than 400: clients that
                // add forward-compatible metadata (trace_id,
                // client_version) shouldn't break here.
                // Drain via chunk() to avoid buffering the entire
                // unknown field in RAM (BUG-9 guard for non-file fields).
                tracing::debug!(field = %other, "ignoring unknown multipart field on /v1/ingest");
                while field.chunk().await.unwrap_or(None).is_some() {}
            }
        }
    }

    let bytes =
        file_bytes.ok_or_else(|| Error::bad_request("missing `file` field in multipart body"))?;
    let kind = file_name.as_ref().map_or(SourceKind::Text, |n| {
        Ingester::source_kind_for_path(std::path::Path::new(n))
    });
    let author =
        author.ok_or_else(|| Error::bad_request("missing `author` field in multipart body"))?;
    run_ingest(
        &state,
        &bytes,
        kind,
        IngestParams {
            ntype: ntype.unwrap_or_else(|| "Doc".into()),
            chunker: chunker_str.unwrap_or_else(|| "auto".into()),
            max_tokens: max_tokens.unwrap_or(512),
            overlap: overlap.unwrap_or(32),
            author,
            message: message.unwrap_or_else(|| "mnem http ingest".into()),
            extractor,
            ner_provider,
        },
    )
}

/// JSON variant.
///
/// `async` is kept for shape-symmetry with `ingest_multipart` (the
/// dispatcher `.await`s either branch uniformly); the function body
/// itself is sync because the JSON body has already been buffered by
/// the caller before we get here.
#[allow(clippy::unused_async)]
async fn ingest_json(state: AppState, body: IngestJsonBody) -> Result<Json<Value>, Error> {
    let max_bytes = max_ingest_bytes();
    if body.text.len() as u64 > max_bytes {
        return Err(Error::bad_request(format!(
            "text body is {} bytes; exceeds the {max_bytes}-byte cap \
             (raise MNEM_HTTP_INGEST_MAX_BYTES if legitimate)",
            body.text.len()
        )));
    }
    let kind = match body.kind.as_deref() {
        Some("markdown" | "md") => SourceKind::Markdown,
        Some("pdf") => SourceKind::Pdf,
        Some("conversation" | "json" | "jsonl") => SourceKind::Conversation,
        Some("text" | "txt") | None => SourceKind::Text,
        Some(other) => {
            return Err(Error::bad_request(format!(
                "unknown `kind`: {other}; want one of markdown|text|pdf|conversation"
            )));
        }
    };
    let bytes = body.text.into_bytes();
    run_ingest(
        &state,
        &bytes,
        kind,
        IngestParams {
            ntype: body.ntype.unwrap_or_else(|| "Doc".into()),
            chunker: body.chunker.unwrap_or_else(|| "auto".into()),
            max_tokens: body.max_tokens.unwrap_or(512),
            overlap: body.overlap.unwrap_or(32),
            author: body.author,
            message: body.message.unwrap_or_else(|| "mnem http ingest".into()),
            extractor: body.extractor,
            ner_provider: body.ner_provider,
        },
    )
}

/// Shared post-parse parameters for both multipart and JSON variants.
struct IngestParams {
    ntype: String,
    chunker: String,
    max_tokens: u32,
    overlap: u32,
    author: String,
    message: String,
    /// C3 FIX-3: extractor selector, defaults to `"none"` (rule-based).
    extractor: Option<String>,
    /// NER provider override for this request. `None` defers to the
    /// server's `AppState::ner_cfg` (which itself falls back to Rule).
    ner_provider: Option<String>,
}

/// Shared execution path: clamp, build Ingester, run the pipeline,
/// commit, observe metrics, render the JSON response.
fn run_ingest(
    state: &AppState,
    bytes: &[u8],
    kind: SourceKind,
    mut params: IngestParams,
) -> Result<Json<Value>, Error> {
    if params.max_tokens > MAX_INGEST_TOKENS {
        return Err(Error::bad_request(format!(
            "max_tokens {} exceeds the {MAX_INGEST_TOKENS} cap",
            params.max_tokens
        )));
    }
    if params.author.trim().is_empty() {
        return Err(Error::bad_request("author is required"));
    }
    // Normalise whitespace-only message to the default.
    if params.message.trim().is_empty() {
        params.message = "mnem http ingest".into();
    }

    // Resolve NER config: per-request override → server default → Rule.
    let ner = match params.ner_provider.as_deref() {
        Some("none") => NerConfig::None,
        Some("rule") | None => state.ner_cfg.clone().unwrap_or(NerConfig::Rule),
        Some(other) => {
            return Err(Error::bad_request(format!(
                "unknown `ner_provider`: {other}; want one of rule|none"
            )));
        }
    };

    let chunker = resolve_chunker(&params.chunker, kind, params.max_tokens, params.overlap)?;
    let config = IngestConfig {
        chunker,
        ntype: params.ntype,
        max_tokens: params.max_tokens,
        overlap: params.overlap,
        ner,
    };
    let mut ing = Ingester::new(config);

    // C3 FIX-3: if the caller asked for the KeyBERT extractor, open
    // the server's configured embedder and wrap it in a
    // `KeyBertAdapter`. Zero cost when the flag is absent: the
    // default rule-based extractor stays wired.
    match params.extractor.as_deref() {
        None | Some("" | "none") => {}
        Some("keybert") => {
            let pc = state.embed_cfg.as_ref().ok_or_else(|| {
                Error::bad_request(
                    "extractor=keybert requires an [embed] provider configured on the server \
                     (MNEM_EMBED_PROVIDER / config.toml); none resolved",
                )
            })?;
            let boxed = mnem_embed_providers::open(pc).map_err(|e| {
                Error::bad_request(format!("opening embed provider for keybert: {e}"))
            })?;
            let arc: std::sync::Arc<dyn mnem_embed_providers::Embedder> =
                std::sync::Arc::from(boxed);
            ing = ing.with_extractor(Box::new(mnem_ingest::KeyBertAdapter::new(arc, "Keyword")));
        }
        Some(other) => {
            return Err(Error::bad_request(format!(
                "unknown `extractor`: {other}; want one of none|keybert"
            )));
        }
    }

    let started = Instant::now();
    let mut guard = state.repo.lock().map_err(|_| Error::locked())?;
    let mut tx = guard.start_transaction();
    let result = ing
        .ingest(&mut tx, bytes, kind)
        .map_err(|e| Error::bad_request(format!("ingest failed: {e}")))?;
    let commit_start = Instant::now();
    let new_repo = tx.commit(&params.author, &params.message)?;
    state
        .metrics
        .commit_duration
        .observe(commit_start.elapsed().as_secs_f64());

    let op_id = new_repo.op_id().to_string();
    let commit_cid = new_repo
        .view()
        .heads
        .first()
        .map_or_else(|| "<none>".to_string(), ToString::to_string);
    *guard = new_repo;

    // Observe ingest-specific metrics: duration + chunk counter.
    // Both are registered at AppState construction time so a scrape
    // against a never-ingested server still emits zero-valued series
    // (no "metric appears out of nowhere" surprise for Prometheus).
    let elapsed = started.elapsed().as_secs_f64();
    state.metrics.ingest_duration.observe(elapsed);
    state.metrics.ingest_chunks.inc_by(result.chunk_count);

    Ok(Json(json!({
        "schema":         "mnem.v1.ingest",
        "op_id":          op_id,
        "commit_cid":     commit_cid,
        "node_count":     result.node_count,
        "chunk_count":    result.chunk_count,
        "entity_count":   result.entity_count,
        "relation_count": result.relation_count,
        "elapsed_ms":     result.elapsed_ms,
    })))
}

fn resolve_chunker(
    choice: &str,
    kind: SourceKind,
    max_tokens: u32,
    overlap: u32,
) -> Result<ChunkerKind, Error> {
    Ok(match choice.to_ascii_lowercase().as_str() {
        "auto" => auto_chunker(
            kind,
            ChunkerAuto {
                max_tokens: Some(max_tokens),
                overlap: Some(overlap),
                max_messages: None,
            },
        ),
        "paragraph" => ChunkerKind::Paragraph,
        "recursive" => ChunkerKind::Recursive {
            max_tokens,
            overlap,
        },
        "session" => ChunkerKind::Session { max_messages: 10 },
        other => {
            return Err(Error::bad_request(format!(
                "chunker must be one of auto|paragraph|recursive|session; got `{other}`"
            )));
        }
    })
}

/// Drain a multipart text field to a UTF-8 String. The multipart crate
/// surfaces bytes; most of our fields are short text, so `to_text` is
/// the right shape.
async fn field_text(field: axum::extract::multipart::Field<'_>) -> Result<String, Error> {
    field
        .text()
        .await
        .map_err(|e| Error::bad_request(format!("decoding text field: {e}")))
}

// `FromRequest` is used via UFCS in the dispatcher; bring the trait
// into scope so the call compiles without an unused-import warning.
use axum::extract::FromRequest;
