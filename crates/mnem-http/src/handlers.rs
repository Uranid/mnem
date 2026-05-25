//! Axum handlers for `mnem http`'s v1 surface.
//!
//! Keep all handlers synchronous inside the lock. We deliberately hold
//! a `std::sync::Mutex` across blocking mnem-core calls rather than a
//! `tokio::Mutex` because those calls don't await; the server is
//! multi-threaded so concurrent readers serialise on the mutex but
//! never on the async runtime.
//
// Remote-v0 insertion point: future remote-transport endpoints land
// here as a parallel `/remote/v1/*` surface, NOT on `/v1/*`. Four
// verbs:
// GET /remote/v1/refs -> list refs + capabilities
// POST /remote/v1/fetch-blocks -> stream a CAR of wanted blocks
// POST /remote/v1/push-blocks -> accept a CAR, verify signatures
// POST /remote/v1/advance-head -> CAS a ref to a new CID
// The protocol is source-agnostic: a hosted Uranid plane is one implementation,
// self-hosted mnem http is another, `file://` is a third. See
// `docs/ROADMAP.md#remote-v0-work-items-tracked-inline-in-src`
// item 1 and ().

use axum::Json;
use axum::extract::{Path, Query, State};
use axum::http::header::CONTENT_TYPE;
use axum::response::IntoResponse;
use ipld_core::ipld::Ipld;
use mnem_core::codec::{from_canonical_bytes, json_to_ipld, to_canonical_bytes};
use mnem_core::id::{Cid, EdgeId, NodeId};
use mnem_core::index::PropPredicate;
use mnem_core::objects::{Commit, Edge, Node, Operation, View};
use mnem_core::prolly::tree::{TreeChunk, build_tree, load_tree_chunk};
use mnem_core::prolly::{DiffEntry, Cursor, diff as prolly_diff};
use mnem_core::store::blockstore::recompute_cid;
use mnem_core::retrieve::Lane;
use mnem_core::{HEADS_PREFIX, TAGS_PREFIX};
// BENCH-1 (C4): trait import is required so `MockEmbedder::embed`
// and `::model` resolve on the concrete struct in the cold-start
// fallback paths inside `retrieve` / `retrieve_full` below.
use mnem_embed_providers::Embedder as _;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};
use std::fmt::Write as _;

use crate::auth::RequireBearer;
use crate::error::Error;
use crate::state::AppState;

// ---------- GET /v1/healthz ----------

/// Canonical wire-name for a retrieval lane. Keep in sync with the
/// `Lane` enum's variants; downstream docs / dashboards depend on
/// these exact strings.
const fn lane_name(lane: Lane) -> &'static str {
    match lane {
        Lane::Vector => "vector",
        Lane::Sparse => "sparse",
        Lane::GraphExpand => "graph_expand",
        Lane::Rerank => "rerank",
        // `Lane` is #[non_exhaustive]; new variants added upstream
        // surface here as "unknown" rather than breaking the wire
        // format. Downstream clients that see an unknown key should
        // still be able to parse the response.
        _ => "unknown",
    }
}

// ---------- input clamps ----------
//
// The retrieve path takes three caller-controlled usize knobs:
// `limit` (final result count), `vector_cap` (per-lane candidate
// pool), and `rerank_top_k` (fanout into an external reranker).
// None had a ceiling before R2-A. A caller could send
// `limit=18446744073709551615` and trigger whatever the downstream
// BruteForce vector search allocates - even behind the 64 MiB
// body-size limit, `Option<usize>` is cheap to send and expensive
// to honour. Reject at the boundary.
//
// The ceilings are deliberately generous: they exist to prevent
// accidental or adversarial OOM, not to impose product shape.
// Legitimate callers will never see them. If you have a real use
// case that exceeds a cap, raise the cap here (not locally per
// handler) and extend the 400 message with the new ceiling.

/// Maximum `limit` accepted on any `/v1/retrieve` variant. Caps the
/// final item count returned to the caller. 1,000 is ~20x the
/// practical top-k a UI or LLM context window can consume.
pub(crate) const MAX_RETRIEVE_LIMIT: usize = 1_000;

/// Maximum `vector_cap` accepted on `POST /v1/retrieve`. Caps the
/// per-lane candidate pool the vector index walks. 100,000 covers
/// the entire legitimate dense-corpus fan-out for the current
/// `BruteForce` index; HNSW will want its own tuning.
pub(crate) const MAX_VECTOR_CAP: usize = 100_000;

/// Maximum `rerank_top_k` accepted on `POST /v1/retrieve`. Caps
/// the number of candidates sent to an external reranker. 500 is
/// 10x what any cross-encoder today handles in <1s; callers
/// usually pick 50-100.
pub(crate) const MAX_RERANK_TOP_K: usize = 500;

/// Maximum byte length for caller-supplied `author` fields across
/// all mutating endpoints. Matches the branch/tag name cap. An
/// unbounded author is a DoS vector: it is stored verbatim in every
/// commit object and re-emitted in every `GET /v1/log` response.
pub(crate) const MAX_AUTHOR_LEN: usize = 255;

/// Maximum byte length for caller-supplied `etype` fields on edge creation.
pub(crate) const MAX_ETYPE_LEN: usize = 255;

/// Maximum byte length for caller-supplied `message` fields. 2 000
/// characters is generous for a commit description while keeping
/// per-commit storage predictable.
pub(crate) const MAX_MESSAGE_LEN: usize = 2_000;

/// Maximum byte length for caller-supplied `summary` fields on node
/// creation. Matches `DEFAULT_RENDER_SUMMARY_CAP_CHARS` in mnem-core
/// so a summary that fits within the HTTP cap always renders cleanly.
/// An unbounded summary is a DoS vector via embedding and retrieval.
pub(crate) const MAX_SUMMARY_LEN: usize = 8_192;

/// Maximum byte length for the free-text `text` query on `POST /v1/retrieve`.
/// Guards the embedder and retriever from arbitrarily large inputs.
pub(crate) const MAX_QUERY_TEXT_LEN: usize = 8_192;

/// Maximum byte length for caller-supplied `content` fields on node
/// creation. 1 MiB is generous for inline text blobs; large files
/// should use `POST /v1/ingest` which has its own 32 MiB cap and
/// dedicated chunking infrastructure.
pub(crate) const MAX_CONTENT_LEN: usize = 1024 * 1024;

/// Maximum value for `multi_query` on `POST /v1/retrieve`. Each
/// variant triggers an LLM call + a full vector retrieval, so an
/// unbounded value is a direct DoS amplification vector. 10 mirrors
/// the practical CLI limit (the prompt template already caps at the
/// requested count, but CLI users can't send arbitrary JSON).
pub(crate) const MAX_MULTI_QUERY: usize = 10;

/// Maximum number of nodes accepted in a single `POST /v1/nodes/bulk`
/// request. 10,000 is ~3x the largest documented benchmark corpus
/// (3,633 nodes); large ingest jobs beyond this should be split into
/// multiple requests or use `POST /v1/ingest` with chunking.
pub(crate) const MAX_BULK_NODES: usize = 10_000;

/// Maximum dimension of a caller-supplied `vector` on `POST /v1/retrieve`.
/// 4096 covers the largest public dense embedders (e.g. text-embedding-3-large
/// at 3072 dims); anything larger is almost certainly a client error or an
/// attempt to exhaust memory in the BruteForce index.
pub(crate) const MAX_VECTOR_DIM: usize = 4_096;

/// Maximum `ppr_iter` on `POST /v1/retrieve`. PPR power-iteration is
/// O(edges × iter); an unbounded value is a CPU DoS amplifier. 100 is
/// well past any empirically useful stopping point (convergence is
/// typically <20 iterations).
pub(crate) const MAX_PPR_ITER: u32 = 100;

/// Maximum `community_expand_seeds` on `POST /v1/retrieve`. The expander
/// runs a community lookup per seed; 100 seeds × community size is plenty
/// for any practical query.
pub(crate) const MAX_COMMUNITY_EXPAND_SEEDS: usize = 100;

/// Maximum `community_max_per` on `POST /v1/retrieve`. Caps how many
/// additional community members can be pulled in per seed community.
pub(crate) const MAX_COMMUNITY_MAX_PER: usize = 500;

/// Maximum number of entries in the `graph_etype` filter Vec on
/// `POST /v1/retrieve`. An unbounded list is a DoS vector (each entry
/// is compared against every edge during graph expansion).
pub(crate) const MAX_GRAPH_ETYPE_COUNT: usize = 100;

/// Maximum `graph_expand` budget on `POST /v1/retrieve`. Caps the total
/// number of graph-neighbour nodes pulled in beyond the seed set.
pub(crate) const MAX_GRAPH_EXPAND: usize = 1000;

/// Maximum `graph_depth` on `POST /v1/retrieve`. BFS beyond depth 4
/// produces exponential fan-out; the CLI internally clamps to 4 too.
pub(crate) const MAX_GRAPH_DEPTH: usize = 4;

/// Maximum `graph_max_per_seed` on `POST /v1/retrieve`. Caps the
/// per-seed out-edge cap used during graph expansion.
pub(crate) const MAX_GRAPH_MAX_PER_SEED: usize = 500;

/// Maximum `limit` on `GET /v1/query` and `POST /v1/query`.
pub(crate) const MAX_QUERY_LIMIT: usize = 1_000;

/// Maximum `limit` on `GET /v1/nodes/{id}/edges`.
pub(crate) const MAX_EDGE_LIMIT: usize = 1_000;

/// Maximum `summarize_k` on `POST /v1/retrieve`. Mirrors
/// `MAX_RETRIEVE_LIMIT` — the caller cannot request more summary
/// sentences than total retrieved items.
pub(crate) const MAX_SUMMARIZE_K: usize = MAX_RETRIEVE_LIMIT;

/// Reject an oversized `limit` / `vector_cap` / `rerank_top_k` with
/// a 400 and a specific message that tells the caller which knob
/// and which cap.
fn clamp_or_reject(name: &'static str, value: Option<usize>, cap: usize) -> Result<(), Error> {
    if let Some(n) = value
        && n > cap
    {
        return Err(Error::bad_request(format!(
            "{name}={n} exceeds max of {cap}; lower the value or split the request"
        )));
    }
    Ok(())
}

/// `clamp_or_reject` variant for `u32` params (e.g. `ppr_iter`).
fn clamp_or_reject_u32(name: &'static str, value: Option<u32>, cap: u32) -> Result<(), Error> {
    if let Some(n) = value
        && n > cap
    {
        return Err(Error::bad_request(format!(
            "{name}={n} exceeds max of {cap}; lower the value or split the request"
        )));
    }
    Ok(())
}

/// Reject a Vec that exceeds `max_len` entries.
fn reject_vec_too_long<T>(name: &'static str, value: &Option<Vec<T>>, max_len: usize) -> Result<(), Error> {
    if let Some(v) = value
        && v.len() > max_len
    {
        return Err(Error::bad_request(format!(
            "{name} has {} entries, exceeds max of {max_len}",
            v.len()
        )));
    }
    Ok(())
}

/// Validate `author` from a JSON body field typed `Option<String>`.
/// Trims, rejects blank, enforces `MAX_AUTHOR_LEN`. Returns the
/// trimmed author string on success.
fn require_author(opt: Option<&str>) -> Result<String, Error> {
    let a = opt
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .ok_or_else(|| Error::bad_request("author is required"))?;
    if a.len() > MAX_AUTHOR_LEN {
        return Err(Error::bad_request(format!(
            "author exceeds maximum length of {MAX_AUTHOR_LEN} bytes"
        )));
    }
    Ok(a.to_string())
}

/// Validate `author` from a query-param or body field typed `&str` /
/// `String` (already non-optional — caller extracted it from the request).
fn validate_author(author: &str) -> Result<(), Error> {
    if author.trim().is_empty() {
        return Err(Error::bad_request("author is required"));
    }
    if author.len() > MAX_AUTHOR_LEN {
        return Err(Error::bad_request(format!(
            "author exceeds maximum length of {MAX_AUTHOR_LEN} bytes"
        )));
    }
    Ok(())
}

/// Validate optional `message` field. Enforces `MAX_MESSAGE_LEN`.
fn validate_message(message: Option<&str>) -> Result<(), Error> {
    if let Some(msg) = message {
        if msg.len() > MAX_MESSAGE_LEN {
            return Err(Error::bad_request(format!(
                "message exceeds maximum length of {MAX_MESSAGE_LEN} bytes"
            )));
        }
    }
    Ok(())
}

#[tracing::instrument]
pub(crate) async fn healthz() -> Json<Value> {
    Json(json!({
    "schema": "mnem.v1.healthz",
    "ok": true,
    "service": "mnem http",
    "version": env!("CARGO_PKG_VERSION"),
    }))
}

/// Fallback for any `/v1/*` path that does not match a registered route.
/// Returns `404 Not Found` with the canonical `mnem.v1.err` envelope so
/// clients always see structured JSON rather than axum's plain-text 404.
pub(crate) async fn fallback_404(uri: axum::http::Uri) -> impl axum::response::IntoResponse {
    Error::not_found(format!("no route for {uri}"))
}

// ---------- GET /v1/stats ----------

pub(crate) async fn stats(State(s): State<AppState>) -> Result<Json<Value>, Error> {
    let repo = s.repo.lock().map_err(|_| Error::locked())?;
    let op_id = repo.op_id().to_string();
    let head = repo.view().heads.first().map(ToString::to_string);
    let refs = repo.view().refs.len();
    let commit: Option<mnem_core::objects::Commit> = repo.head_commit().cloned();
    let bs = repo.blockstore().clone();
    drop(repo);

    let (content_cid, node_count, edge_count, label_count) = match commit.as_ref() {
        None => (None::<String>, 0usize, 0usize, 0usize),
        Some(c) => {
            let cc = c
                .content_cid()
                .map(|cid| cid.to_string())
                .unwrap_or_else(|_| "<encode-error>".into());
            let nodes = {
                let cursor = Cursor::new(&*bs, &c.nodes)
                    .map_err(|e| Error::internal(format!("node cursor: {e}")))?;
                let mut count = 0usize;
                for entry in cursor {
                    entry.map_err(|e| Error::internal(format!("node cursor entry: {e}")))?;
                    count += 1;
                }
                count
            };
            let edges = {
                let cursor = Cursor::new(&*bs, &c.edges)
                    .map_err(|e| Error::internal(format!("edge cursor: {e}")))?;
                let mut count = 0usize;
                for entry in cursor {
                    entry.map_err(|e| Error::internal(format!("edge cursor entry: {e}")))?;
                    count += 1;
                }
                count
            };
            let labels = c
                .indexes
                .as_ref()
                .and_then(|idx_cid| bs.get(idx_cid).ok().flatten())
                .and_then(|bytes| {
                    from_canonical_bytes::<mnem_core::objects::IndexSet>(&bytes)
                        .ok()
                        .map(|set| set.nodes_by_label.len())
                })
                .unwrap_or(0);
            (Some(cc), nodes, edges, labels)
        }
    };

    Ok(Json(json!({
    "schema": "mnem.v1.stats",
    "op_id": op_id,
    "head_commit": head,
    "content_cid": content_cid,
    "refs": refs,
    "nodes": node_count,
    "edges": edge_count,
    "labels": label_count,
    })))
}

// ---------- POST /v1/nodes ----------

#[derive(Deserialize)]
pub(crate) struct PostNodeBody {
    /// Scoping tag. Maps to `Node.ntype` on the wire. Optional on the
    /// HTTP boundary: if omitted or empty, the server substitutes
    /// [`Node::DEFAULT_NTYPE`] (`"Node"`). Callers that want
    /// per-tenant / per-conversation isolation pass a non-empty value.
    #[serde(default)]
    pub label: String,
    pub summary: Option<String>,
    pub props: Option<Map<String, Value>>,
    pub content: Option<String>,
    /// Required for the single-node `POST /v1/nodes` path; optional
    /// inside the bulk wrapper (audit-2026-04-25 P2-8): when absent,
    /// the wrapper-level `author` is used. The single-node handler
    /// still validates non-empty before commit, so the contract is
    /// preserved.
    #[serde(default)]
    pub author: Option<String>,
    #[serde(default)]
    pub message: Option<String>,
    /// Optional caller-supplied UUID. When present, mnem uses it as
    /// the node's `NodeId` instead of generating a fresh v7. Lets
    /// distributed agents + replay pipelines pin node identity so
    /// two machines ingesting the same logical event produce the
    /// same `Node` CID. Must be a UUID-8x20 / UUID-v7 / UUID-v4
    /// string parseable by `NodeId::parse_uuid`.
    #[serde(default)]
    pub id: Option<String>,
    /// Skip embedding even if the server has an embedder configured.
    /// Mirrors `mnem add node --no-embed`. Useful for bulk imports
    /// where embedding is batched separately via `mnem reindex`.
    #[serde(default)]
    pub no_embed: bool,
    /// Resolve-or-create mode: anchor the node on this property instead of
    /// always creating a new one. Format: `"key=value"`.
    /// If a node with `(label, key=value)` already exists in the graph its
    /// UUID is reused; otherwise a new node is created with that property.
    /// Additional `props` set extra properties on the resolved or newly-created
    /// node. Mirrors `mnem add node --canonical KEY=VALUE`.
    /// Conflicts with `id` and `deterministic`.
    #[serde(default)]
    pub canonical: Option<String>,
    /// Derive the node UUID deterministically from `(label, sorted props)` via
    /// blake3 truncation instead of generating a fresh UUIDv7. Two callers
    /// passing the same `label` and `props` produce byte-identical `NodeId`s
    /// (and therefore the same content_cid), which is required by
    /// distributed-replay and content-addressable archive flows.
    /// Mirrors `mnem add node --deterministic`. Conflicts with `id` and
    /// `canonical`.
    #[serde(default)]
    pub deterministic: bool,
    /// Optional one-sentence framing hint stored on the node.
    /// Mirrors `mnem add node --context "..."`. Participates in the node CID.
    #[serde(default)]
    pub context_sentence: Option<String>,
}

#[derive(Serialize)]
pub(crate) struct PostNodeResp {
    schema: &'static str,
    id: String,
    cid: String,
    label: String,
    op_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    summary: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    context_sentence: Option<String>,
    content_bytes: usize,
    props: serde_json::Value,
    has_embedding: bool,
    tombstoned: bool,
}

pub(crate) async fn post_node(
    _auth: RequireBearer,
    State(s): State<AppState>,
    Json(body): Json<PostNodeBody>,
) -> Result<Json<PostNodeResp>, Error> {
    // Validate conflicting field combinations up front.
    if body.canonical.is_some() && body.id.is_some() {
        return Err(Error::bad_request(
            "`canonical` and `id` are mutually exclusive",
        ));
    }
    if body.canonical.is_some() && body.deterministic {
        return Err(Error::bad_request(
            "`canonical` and `deterministic` are mutually exclusive",
        ));
    }
    if body.deterministic && body.id.is_some() {
        return Err(Error::bad_request(
            "`deterministic` and `id` are mutually exclusive",
        ));
    }

    // Two-step label resolution:
    // 1. If the server was not launched with `MNEM_BENCH=1`
    // (`s.allow_labels == false`), we *ignore* any caller-supplied
    // `label` silently and always use `Node::DEFAULT_NTYPE`. This
    // is the casual / single-tenant path: no label surface.
    // 2. If `s.allow_labels == true`, we honour the caller's value;
    // an empty/omitted value still falls back to the default.
    let label = if s.allow_labels && !body.label.trim().is_empty() {
        body.label.clone()
    } else {
        Node::DEFAULT_NTYPE.to_string()
    };
    let author = require_author(body.author.as_deref())?;
    validate_message(body.message.as_deref())?;
    if let Some(sum) = body.summary.as_deref() {
        if sum.len() > MAX_SUMMARY_LEN {
            return Err(Error::bad_request(format!(
                "summary exceeds maximum length of {MAX_SUMMARY_LEN} bytes"
            )));
        }
    }
    if let Some(content) = body.content.as_deref() {
        if content.len() > MAX_CONTENT_LEN {
            return Err(Error::bad_request(format!(
                "content exceeds maximum length of {MAX_CONTENT_LEN} bytes"
            )));
        }
    }

    // ---------- canonical (resolve-or-create) path ----------
    //
    // Mirrors `mnem add node --canonical KEY=VALUE`. Finds an existing
    // node with (label, key=value); if absent creates one. Extra `props`
    // are layered on top. Returns the resolved node's UUID.
    if let Some(ref canonical_str) = body.canonical {
        let (prop_name, anchor_value) = parse_canonical_kv(canonical_str).map_err(|e| {
            Error::bad_request(format!(
                "`canonical` expects KEY=VALUE format, got `{canonical_str}`: {e}"
            ))
        })?;

        // Parse any extra props from the JSON map.
        let extra_props: Vec<(String, ipld_core::ipld::Ipld)> =
            if let Some(ref props_map) = body.props {
                props_map
                    .iter()
                    .map(|(k, v)| {
                        json_to_ipld(v)
                            .map(|ipld| (k.clone(), ipld))
                            .map_err(|e| Error::bad_request(e.to_string()))
                    })
                    .collect::<Result<Vec<_>, Error>>()?
            } else {
                Vec::new()
            };

        let mut guard = s.repo.lock().map_err(|_| Error::locked())?;
        let mut tx = guard.start_transaction();

        // resolve_or_create_node returns the existing id or creates a new node.
        let node_id = tx
            .resolve_or_create_node(&label, &prop_name, anchor_value.clone())
            .map_err(|e| Error::internal(e.to_string()))?;

        // Build the full node by loading the existing node (if any) and
        // layering the anchor prop + extra props + summary + content on top.
        let mut node = match tx
            .base()
            .lookup_node(&node_id)
            .map_err(|e| Error::internal(e.to_string()))?
        {
            Some(existing) => existing,
            None => Node::new(node_id, &label),
        };
        node.ntype = label.clone();
        node = node.with_prop(prop_name, anchor_value);
        if let Some(ref sum) = body.summary {
            node = node.with_summary(sum);
        }
        for (k, v) in extra_props {
            node = node.with_prop(k, v);
        }
        if let Some(ref ctx) = body.context_sentence {
            node = node.with_context_sentence(ctx);
        }
        if let Some(c) = body.content {
            node = node.with_content(bytes::Bytes::from(c.into_bytes()));
        }

        // Embed (same silent-on-failure semantics as the plain path).
        let text_for_embed: Option<String> = node
            .summary
            .as_ref()
            .filter(|t| !t.trim().is_empty())
            .cloned();
        let mut pending_dense: Option<(String, mnem_core::objects::Embedding)> = None;
        let mut pending_sparse: Option<(String, mnem_core::sparse::SparseEmbed)> = None;
        if !body.no_embed {
            if let Some(text) = text_for_embed {
                if let Some(pc) = &s.embed_cfg
                    && let Ok(embedder) = mnem_embed_providers::open(pc)
                    && let Ok(v) = embedder.embed(&text)
                {
                    let emb = mnem_embed_providers::to_embedding(embedder.model(), &v);
                    pending_dense = Some((embedder.model().to_string(), emb));
                }
                if let Some(sc) = &s.sparse_cfg
                    && let Ok(sparser) = mnem_sparse_providers::open(sc)
                    && let Ok(se) = sparser.encode(&text)
                {
                    pending_sparse = Some((sparser.vocab_id().to_string(), se));
                }
            }
        }

        let resolved_id = node.id;
        let has_embedding = pending_dense.is_some();
        let cid = tx
            .add_node(&node)
            .map_err(|e| Error::internal(e.to_string()))?;
        let cid_str = cid.to_string();
        if let Some((model, emb)) = pending_dense {
            tx.set_embedding(cid.clone(), model, emb)
                .map_err(|e| Error::internal(e.to_string()))?;
        }
        if let Some((vocab_id, se)) = pending_sparse {
            tx.set_sparse_embedding(cid, vocab_id, se)
                .map_err(|e| Error::internal(e.to_string()))?;
        }
        let commit_start = std::time::Instant::now();
        let new_repo = tx.commit(
            &author,
            body.message
                .as_deref()
                .unwrap_or("mnem http resolve-or-create node"),
        )?;
        s.metrics
            .commit_duration
            .observe(commit_start.elapsed().as_secs_f64());
        let op_id = new_repo.op_id().to_string();
        *guard = new_repo;
        let mut props_map = serde_json::Map::new();
        for (k, v) in &node.props {
            props_map.insert(k.clone(), ipld_to_json(v));
        }
        return Ok(Json(PostNodeResp {
            schema: "mnem.v1.node-created",
            id: resolved_id.to_uuid_string(),
            cid: cid_str,
            label,
            op_id,
            summary: node.summary.clone(),
            context_sentence: node.context_sentence.clone(),
            content_bytes: node.content.as_ref().map_or(0, bytes::Bytes::len),
            props: serde_json::Value::Object(props_map),
            has_embedding,
            tombstoned: false,
        }));
    }

    // ---------- deterministic node-id path ----------
    //
    // Mirrors `mnem add node --deterministic`. Derives the NodeId from
    // blake3(label + sorted props) so two callers with identical inputs
    // produce the same node UUID and content_cid.
    let node_id = if body.deterministic {
        derive_deterministic_node_id_http(&label, body.props.as_ref()).map_err(|e| {
            Error::bad_request(format!("could not derive deterministic id: {e}"))
        })?
    } else {
        match body.id.as_deref() {
            Some(s) => NodeId::parse_uuid(s)
                .map_err(|e| Error::bad_request(format!("invalid caller-supplied id: {e}")))?,
            None => NodeId::new_v7(),
        }
    };

    // CLI parity: `mnem add node` (standard path) requires --summary.
    // Deterministic and caller-supplied-id paths intentionally allow omitting
    // summary because identity comes from props or the explicit UUID.
    if !body.deterministic
        && body.id.is_none()
        && body.summary.as_deref().map(str::trim).filter(|s| !s.is_empty()).is_none()
    {
        return Err(Error::bad_request("summary is required"));
    }

    let mut node = Node::new(node_id, &label);
    if let Some(sum) = &body.summary {
        node = node.with_summary(sum);
    }
    if let Some(ref ctx) = body.context_sentence {
        node = node.with_context_sentence(ctx);
    }
    if let Some(props) = body.props {
        for (k, v) in props {
            node = node.with_prop(
                k,
                json_to_ipld(&v).map_err(|e| Error::bad_request(e.to_string()))?,
            );
        }
    }
    if let Some(c) = body.content {
        node = node.with_content(bytes::Bytes::from(c.into_bytes()));
    }

    // Auto-embed the node's summary (dense + sparse, if configured).
    // Failures warn but do not block the commit; a later `mnem embed`
    // pass can backfill. Dense vectors stage to `Commit.embeddings`
    // via `Transaction::set_embedding`; sparse vectors stage to
    // `Commit.sparse` via `Transaction::set_sparse_embedding`.
    // Neither touches the Node bytes, keeping `NodeCid` stable across
    // encoder versions and federated peers (G16/G17).
    let text_for_embed: Option<String> = node
        .summary
        .as_ref()
        .filter(|t| !t.trim().is_empty())
        .cloned();
    let mut pending_dense: Option<(String, mnem_core::objects::Embedding)> = None;
    let mut pending_sparse: Option<(String, mnem_core::sparse::SparseEmbed)> = None;
    if !body.no_embed {
        if let Some(text) = text_for_embed {
            if let Some(pc) = &s.embed_cfg
                && let Ok(embedder) = mnem_embed_providers::open(pc)
                && let Ok(v) = embedder.embed(&text)
            {
                let emb = mnem_embed_providers::to_embedding(embedder.model(), &v);
                pending_dense = Some((embedder.model().to_string(), emb));
            }
            if let Some(sc) = &s.sparse_cfg
                && let Ok(sparser) = mnem_sparse_providers::open(sc)
                && let Ok(se) = sparser.encode(&text)
            {
                pending_sparse = Some((sparser.vocab_id().to_string(), se));
            }
            // Silent on failure; the POST path returns an `id` either way.
        }
    }

    let id = node.id;
    let has_embedding = pending_dense.is_some();

    let mut guard = s.repo.lock().map_err(|_| Error::locked())?;
    let mut tx = guard.start_transaction();
    let cid = tx.add_node(&node)?;
    let cid_str = cid.to_string();
    if let Some((model, emb)) = pending_dense {
        tx.set_embedding(cid.clone(), model, emb)?;
    }
    if let Some((vocab_id, se)) = pending_sparse {
        tx.set_sparse_embedding(cid, vocab_id, se)?;
    }
    let commit_start = std::time::Instant::now();
    let new_repo = tx.commit(
        &author,
        body.message.as_deref().unwrap_or("mnem http add node"),
    )?;
    s.metrics
        .commit_duration
        .observe(commit_start.elapsed().as_secs_f64());
    let op_id = new_repo.op_id().to_string();
    *guard = new_repo;
    let mut props_map = serde_json::Map::new();
    for (k, v) in &node.props {
        props_map.insert(k.clone(), ipld_to_json(v));
    }
    Ok(Json(PostNodeResp {
        schema: "mnem.v1.node-created",
        id: id.to_uuid_string(),
        cid: cid_str,
        label,
        op_id,
        summary: node.summary.clone(),
        context_sentence: node.context_sentence.clone(),
        content_bytes: node.content.as_ref().map_or(0, bytes::Bytes::len),
        props: serde_json::Value::Object(props_map),
        has_embedding,
        tombstoned: false,
    }))
}

/// Parse a `KEY=VALUE` string as used by the `canonical` field.
/// The first `=` splits the pair; the value may contain `=` characters.
fn parse_canonical_kv(s: &str) -> Result<(String, ipld_core::ipld::Ipld), String> {
    let pos = s
        .find('=')
        .ok_or_else(|| format!("no `=` separator found in `{s}`"))?;
    let key = s[..pos].to_string();
    if key.is_empty() {
        return Err(format!("key is empty in `{s}`"));
    }
    let raw_val = &s[pos + 1..];
    // Try to parse as JSON; fall back to plain string.
    let val = if let Ok(v) = serde_json::from_str::<Value>(raw_val) {
        json_to_ipld(&v).map_err(|e| e.to_string())?
    } else {
        ipld_core::ipld::Ipld::String(raw_val.to_string())
    };
    Ok((key, val))
}

/// Derive a stable `NodeId` from `(label, sorted props)` via blake3 truncation.
///
/// Mirrors the `derive_deterministic_node_id` function in the CLI's `add.rs`.
/// Hash input: `"mnem-c3-2:node:v1\0" || label || "\0" || for (k,v) in sorted_props: k || "=" || dag-cbor(v) || "\0"`.
fn derive_deterministic_node_id_http(
    label: &str,
    props: Option<&Map<String, Value>>,
) -> Result<NodeId, String> {
    use mnem_core::id::Multihash;

    let mut kv: std::collections::BTreeMap<String, ipld_core::ipld::Ipld> =
        std::collections::BTreeMap::new();
    if let Some(props_map) = props {
        for (k, v) in props_map {
            let ipld = json_to_ipld(v).map_err(|e| e.to_string())?;
            kv.insert(k.clone(), ipld);
        }
    }

    let mut buf: Vec<u8> = Vec::with_capacity(64 + label.len() + 16 * kv.len());
    buf.extend_from_slice(b"mnem-c3-2:node:v1\0");
    buf.extend_from_slice(label.as_bytes());
    buf.push(0);
    for (k, v) in &kv {
        buf.extend_from_slice(k.as_bytes());
        buf.push(b'=');
        let cbor = to_canonical_bytes(v).map_err(|e| e.to_string())?;
        buf.extend_from_slice(&cbor);
        buf.push(0);
    }

    let mh = Multihash::blake3_256(&buf);
    let digest = mh.digest();
    let mut bytes16 = [0u8; 16];
    bytes16.copy_from_slice(&digest[..16]);
    Ok(NodeId::from_random_bytes(bytes16))
}

// ---------- GET /v1/nodes/{id} ----------

pub(crate) async fn get_node(
    State(s): State<AppState>,
    Path(id_str): Path<String>,
) -> Result<Json<Value>, Error> {
    let id = NodeId::parse_uuid(&id_str)
        .map_err(|e| Error::bad_request(format!("invalid UUID: {e}")))?;
    let repo = s.repo.lock().map_err(|_| Error::locked())?;
    let node = repo
        .lookup_node(&id)?
        .ok_or_else(|| Error::not_found(format!("no node with id={id_str}")))?;

    let tombstoned = repo.is_tombstoned(&id);

    let mut props_map = Map::new();
    for (k, v) in &node.props {
        props_map.insert(k.clone(), ipld_to_json(v));
    }

    // Embeddings are sidecar-attached, not Node-inline. Probe under
    // the configured embedder's `model_fq` (the same string used at
    // write time). When no embed-provider is configured, we report
    // `has_embedding = false` rather than enumerate every model -
    // the sidecar API is keyed by model and a multi-model probe is
    // out of scope for this single-flag wire field.
    let has_embedding = match s.embed_cfg.as_ref() {
        Some(pc) => {
            let model = model_fq_of(pc);
            let (_, node_cid) = mnem_core::codec::hash_to_cid(&node)
                .map_err(|e| Error::internal(format!("hash node: {e}")))?;
            repo.embedding_for(&node_cid, &model)?.is_some()
        }
        None => false,
    };

    Ok(Json(json!({
    "schema": "mnem.v1.node",
    "id": node.id.to_uuid_string(),
    "label": node.ntype,
    "summary": node.summary,
    "context_sentence": node.context_sentence,
    "props": Value::Object(props_map),
    "content_bytes": node.content.as_ref().map_or(0, bytes::Bytes::len),
    "has_embedding": has_embedding,
    "tombstoned": tombstoned,
    })))
}

/// Format the `provider:model` string the embedder adapters expose
/// via `Embedder::model()`. Mirrored here so handlers can derive it
/// from a `ProviderConfig` without opening the adapter.
fn model_fq_of(pc: &mnem_embed_providers::ProviderConfig) -> String {
    use mnem_embed_providers::ProviderConfig as PC;
    match pc {
        PC::Openai(c) => format!("openai:{}", c.model),
        PC::Ollama(c) => format!("ollama:{}", c.model),
        PC::Onnx(c) => format!("onnx:{}", c.model),
    }
}

// ---------- GET /v1/nodes/{id}/embedding ----------

/// Query parameters for `GET /v1/nodes/{id}/embedding`.
#[derive(Deserialize)]
pub(crate) struct GetNodeEmbeddingQuery {
    /// Model identifier string (e.g. ``"onnx:all-MiniLM-L6-v2"``).
    model: String,
}

/// Fetch the embedding vector for a node by UUID and model string.
///
/// Returns `200 OK` with the embedding in the `mnem.v1.node-embedding`
/// schema, or `404 Not Found` when the node or embedding does not exist.
pub(crate) async fn get_node_embedding(
    State(s): State<AppState>,
    Path(id_str): Path<String>,
    Query(q): Query<GetNodeEmbeddingQuery>,
) -> Result<Json<Value>, Error> {
    let id = NodeId::parse_uuid(&id_str)
        .map_err(|e| Error::bad_request(format!("invalid UUID: {e}")))?;
    let repo = s.repo.lock().map_err(|_| Error::locked())?;
    let node = repo
        .lookup_node(&id)?
        .ok_or_else(|| Error::not_found(format!("no node with id={id_str}")))?;

    let (_, node_cid) = mnem_core::codec::hash_to_cid(&node)
        .map_err(|e| Error::internal(format!("hash node: {e}")))?;

    let emb = repo.embedding_for(&node_cid, &q.model)?.ok_or_else(|| {
        Error::not_found(format!(
            "no embedding for model={} on node {}",
            q.model, id_str
        ))
    })?;

    let bytes = emb.vector.as_ref();
    if bytes.len() % 4 != 0 {
        return Err(Error::internal(format!(
            "embedding vector byte length {} is not a multiple of 4; blockstore may be corrupt",
            bytes.len()
        )));
    }
    let vector: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|c| {
            let arr: [u8; 4] = c
                .try_into()
                .map_err(|_| Error::internal("embedding chunk is not 4 bytes".to_string()))?;
            Ok(f32::from_le_bytes(arr))
        })
        .collect::<Result<Vec<f32>, Error>>()?;

    let dtype_str = match emb.dtype {
        mnem_core::objects::Dtype::F32 => "f32",
        mnem_core::objects::Dtype::F16 => "f16",
        mnem_core::objects::Dtype::F64 => "f64",
        mnem_core::objects::Dtype::I8 => "i8",
    };

    Ok(Json(json!({
        "schema": "mnem.v1.node-embedding",
        "node_id": id_str,
        "model": emb.model,
        "dim": emb.dim,
        "dtype": dtype_str,
        "vector": vector,
    })))
}

// ---------- DELETE /v1/nodes/{id} ----------

#[derive(Deserialize)]
pub(crate) struct DeleteQuery {
    /// Commit author. Required; query-string rather than body so `curl
    /// -X DELETE` stays one-line-trivial.
    pub author: String,
    #[serde(default)]
    pub message: Option<String>,
}

pub(crate) async fn delete_node(
    _auth: RequireBearer,
    State(s): State<AppState>,
    Path(id_str): Path<String>,
    Query(q): Query<DeleteQuery>,
) -> Result<Json<Value>, Error> {
    let id = NodeId::parse_uuid(&id_str)
        .map_err(|e| Error::bad_request(format!("invalid UUID: {e}")))?;
    validate_author(&q.author)?;
    validate_message(q.message.as_deref())?;

    let mut guard = s.repo.lock().map_err(|_| Error::locked())?;
    let existed = guard.lookup_node(&id)?.is_some();
    if !existed {
        return Err(Error::not_found(format!(
            "no node with id={id_str} in current view"
        )));
    }
    let author = q.author.trim().to_string();
    let mut tx = guard.start_transaction();
    tx.remove_node(id);
    let commit_start = std::time::Instant::now();
    let new_repo = tx.commit(
        &author,
        q.message.as_deref().unwrap_or("mnem http delete node"),
    )?;
    s.metrics
        .commit_duration
        .observe(commit_start.elapsed().as_secs_f64());
    let op_id = new_repo.op_id().to_string();
    *guard = new_repo;

    Ok(Json(json!({
    "schema": "mnem.v1.node-deleted",
    "id": id_str,
    "existed": true,
    "op_id": op_id,
    })))
}

// ---------- POST /v1/nodes/{id}/tombstone ----------

#[derive(Deserialize)]
pub(crate) struct TombstoneBody {
    /// Free-form reason string recorded on the tombstone.
    #[serde(default)]
    pub reason: String,
    /// Optional commit message. Mirrors `mnem tombstone --message`.
    /// Defaults to `"mnem http tombstone node"`.
    #[serde(default)]
    pub message: Option<String>,
    /// Commit author.
    pub author: String,
}

pub(crate) async fn tombstone_node(
    _auth: RequireBearer,
    State(s): State<AppState>,
    Path(id_str): Path<String>,
    Json(body): Json<TombstoneBody>,
) -> Result<Json<Value>, Error> {
    let id = NodeId::parse_uuid(&id_str)
        .map_err(|e| Error::bad_request(format!("invalid UUID: {e}")))?;
    validate_author(&body.author)?;
    validate_message(body.message.as_deref())?;
    let author = body.author.trim().to_string();
    let mut guard = s.repo.lock().map_err(|_| Error::locked())?;
    // 404: the underlying node must exist in the current head. We
    // check before starting a transaction so the error surface is
    // clean (no "stale" commit on a missing id).
    if guard.lookup_node(&id)?.is_none() {
        return Err(Error::not_found(format!("no node with id={id_str}")));
    }
    // 409: already tombstoned. Matches the item-3 contract: callers
    // shouldn't be able to re-tombstone silently via the HTTP API
    // (the in-process `tombstone_node` remains idempotent for agents
    // that want that behaviour).
    if guard.is_tombstoned(&id) {
        return Err(Error::conflict(format!(
            "node {id_str} is already tombstoned"
        )));
    }
    let commit_msg = body
        .message
        .as_deref()
        .filter(|m| !m.trim().is_empty())
        .unwrap_or("mnem http tombstone node");
    let mut tx = guard.start_transaction();
    tx.tombstone_node(id, body.reason.clone())?;
    let commit_start = std::time::Instant::now();
    let new_repo = tx.commit(&author, commit_msg)?;
    s.metrics
        .commit_duration
        .observe(commit_start.elapsed().as_secs_f64());
    let tombstoned_at = new_repo.operation().time;
    let op_id = new_repo.op_id().to_string();
    *guard = new_repo;

    Ok(Json(json!({
        "schema": "mnem.v1.node-tombstoned",
        "op_id": op_id,
        "node_id": id_str,
        "reason": body.reason,
        "tombstoned_at": tombstoned_at,
    })))
}

// ---------- POST /v1/edges ----------

/// Request body for `POST /v1/edges`.
#[derive(Deserialize)]
pub(crate) struct PostEdgeBody {
    /// UUID of the source node.
    pub src: String,
    /// UUID of the destination node.
    pub dst: String,
    /// Edge-type label (e.g. `"knows"`, `"works_at"`, `"cites"`).
    pub etype: String,
    /// Optional edge properties. Values must be JSON-serialisable IPLD.
    #[serde(default)]
    pub props: Option<Map<String, Value>>,
    /// Commit author. Required.
    pub author: String,
    /// Optional commit message. Defaults to `"mnem http add edge"`.
    #[serde(default)]
    pub message: Option<String>,
}

/// `POST /v1/edges` - commit a new directed edge between two existing nodes.
///
/// Returns `{"schema":"mnem.v1.edge-created","id":"<uuid>","op_id":"<cid>"}`.
/// Returns 404 if `src` or `dst` does not exist in the current view.
/// Returns 400 for malformed UUIDs, empty `etype`, or missing `author`.
pub(crate) async fn post_edge(
    _auth: RequireBearer,
    State(s): State<AppState>,
    Json(body): Json<PostEdgeBody>,
) -> Result<Json<Value>, Error> {
    // Validate required fields.
    let author = body.author.trim();
    validate_author(author)?;
    validate_message(body.message.as_deref())?;
    if body.etype.trim().is_empty() {
        return Err(Error::bad_request("etype is required"));
    }
    if body.etype.len() > MAX_ETYPE_LEN {
        return Err(Error::bad_request(format!(
            "etype exceeds maximum length of {MAX_ETYPE_LEN} characters"
        )));
    }

    // Parse UUIDs.
    let src = NodeId::parse_uuid(&body.src)
        .map_err(|e| Error::bad_request(format!("invalid src UUID: {e}")))?;
    let dst = NodeId::parse_uuid(&body.dst)
        .map_err(|e| Error::bad_request(format!("invalid dst UUID: {e}")))?;

    let mut guard = s.repo.lock().map_err(|_| Error::locked())?;

    // C8: validate both endpoints exist before writing.
    if guard.lookup_node(&src)?.is_none() {
        return Err(Error::not_found(format!(
            "no node with id={} (src)",
            body.src
        )));
    }
    if guard.lookup_node(&dst)?.is_none() {
        return Err(Error::not_found(format!(
            "no node with id={} (dst)",
            body.dst
        )));
    }

    let etype = body.etype.trim().to_string();
    let edge_id = EdgeId::new_v7();
    let mut edge = Edge::new(edge_id, &etype, src, dst);
    if let Some(props) = body.props {
        for (k, v) in props {
            edge = edge.with_prop(
                k,
                json_to_ipld(&v).map_err(|e| Error::bad_request(e.to_string()))?,
            );
        }
    }

    let mut tx = guard.start_transaction();
    let edge_cid = tx.add_edge(&edge)?;
    let edge_cid_str = edge_cid.to_string();
    let commit_start = std::time::Instant::now();
    let new_repo = tx.commit(
        author,
        body.message.as_deref().unwrap_or("mnem http add edge"),
    )?;
    s.metrics
        .commit_duration
        .observe(commit_start.elapsed().as_secs_f64());
    let op_id = new_repo.op_id().to_string();
    *guard = new_repo;

    Ok(Json(json!({
        "schema": "mnem.v1.edge-created",
        "id": edge_id.to_uuid_string(),
        "cid": edge_cid_str,
        "op_id": op_id,
        "etype": etype,
        "src": src.to_uuid_string(),
        "dst": dst.to_uuid_string(),
    })))
}

// ---------- POST /v1/nodes/bulk ----------
//
// One-commit bulk ingest. The per-node POST /v1/nodes path commits
// after every write (Prolly-tree + IndexSet rebuild each time), which
// is ~2 seconds per node on a laptop with ollama. At 3633 docs that
// is two hours. The bulk endpoint accepts N nodes in one request and
// does ONE commit at the end, dropping the same ingest to minutes.
//
// Response includes the per-node IDs in the order sent so callers
// can build their external_id <-> mnem_node_id map.

#[derive(Deserialize)]
pub(crate) struct BulkNodeBody {
    pub nodes: Vec<PostNodeBody>,
    pub author: String,
    #[serde(default)]
    pub message: Option<String>,
    /// When true (default) AND an embed provider is configured on the
    /// server, each node's summary is auto-embedded before commit.
    #[serde(default = "default_true")]
    pub auto_embed: bool,
}

const fn default_true() -> bool {
    true
}

#[derive(Serialize)]
pub(crate) struct BulkNodeResp {
    schema: &'static str,
    op_id: String,
    /// One entry per input node, same order.
    results: Vec<BulkNodeEntry>,
    /// How many nodes embedded successfully vs skipped.
    embedded: u32,
    skipped_embed: u32,
}

#[derive(Serialize)]
pub(crate) struct BulkNodeEntry {
    id: String,
    cid: String,
    label: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    summary: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    context_sentence: Option<String>,
    content_bytes: usize,
    props: serde_json::Value,
    has_embedding: bool,
    tombstoned: bool,
}

pub(crate) async fn post_nodes_bulk(
    _auth: RequireBearer,
    State(s): State<AppState>,
    Json(body): Json<BulkNodeBody>,
) -> Result<Json<BulkNodeResp>, Error> {
    validate_author(&body.author)?;
    validate_message(body.message.as_deref())?;
    if body.nodes.is_empty() {
        return Err(Error::bad_request("nodes must not be empty"));
    }
    if body.nodes.len() > MAX_BULK_NODES {
        return Err(Error::bad_request(format!(
            "nodes length {} exceeds maximum of {MAX_BULK_NODES}; split into smaller batches",
            body.nodes.len()
        )));
    }
    for (i, nb) in body.nodes.iter().enumerate() {
        if nb.canonical.is_some() {
            return Err(Error::bad_request(format!(
                "nodes[{i}]: `canonical` is not supported in bulk mode; use POST /v1/nodes"
            )));
        }
        if nb.deterministic {
            return Err(Error::bad_request(format!(
                "nodes[{i}]: `deterministic` is not supported in bulk mode; use POST /v1/nodes"
            )));
        }
        if let Some(sum) = nb.summary.as_deref() {
            if sum.len() > MAX_SUMMARY_LEN {
                return Err(Error::bad_request(format!(
                    "nodes[{i}]: summary exceeds maximum length of {MAX_SUMMARY_LEN} bytes"
                )));
            }
        }
        if let Some(content) = nb.content.as_deref() {
            if content.len() > MAX_CONTENT_LEN {
                return Err(Error::bad_request(format!(
                    "nodes[{i}]: content exceeds maximum length of {MAX_CONTENT_LEN} bytes"
                )));
            }
        }
        if let Some(a) = nb.author.as_deref() {
            let a = a.trim();
            if a.is_empty() {
                return Err(Error::bad_request(format!(
                    "nodes[{i}]: per-node `author` must not be blank when supplied"
                )));
            }
            if a.len() > MAX_AUTHOR_LEN {
                return Err(Error::bad_request(format!(
                    "nodes[{i}]: per-node `author` exceeds maximum length of {MAX_AUTHOR_LEN} bytes"
                )));
            }
        }
        if let Some(m) = nb.message.as_deref() {
            if m.len() > MAX_MESSAGE_LEN {
                return Err(Error::bad_request(format!(
                    "nodes[{i}]: per-node `message` exceeds maximum length of {MAX_MESSAGE_LEN} bytes"
                )));
            }
        }
    }
    let author = body.author.trim().to_string();

    // Resolve the dense embedder + the sparse encoder once so we don't
    // reopen per node. If a provider is configured but opening fails
    // (bad API key, sidecar unreachable), fail the whole bulk call
    // instead of committing every node without embeddings and
    // silently reporting success.
    let embedder = if body.auto_embed {
        match s.embed_cfg.as_ref() {
 Some(pc) => Some(mnem_embed_providers::open(pc).map_err(|e| {
 Error::internal(format!(
 "embed provider configured but open failed: {e}; bulk aborted to avoid silent no-embed commit"
 ))
 })?),
 None => None,
 }
    } else {
        None
    };
    let sparser = if body.auto_embed {
        match s.sparse_cfg.as_ref() {
 Some(sc) => Some(mnem_sparse_providers::open(sc).map_err(|e| {
 Error::internal(format!(
 "sparse provider configured but open failed: {e}; bulk aborted to avoid silent no-sparse commit"
 ))
 })?),
 None => None,
 }
    } else {
        None
    };

    // Pre-build every Node, doing the embed calls before taking the
    // repo mutex. Ollama / OpenAI calls can be slow; holding the
    // mutex across them would block every other HTTP request.
    // Each entry pairs the Node with an optional dense (model, vec)
    // staged for the sidecar-side `Transaction::set_embedding` call
    // that runs after `add_node` returns the NodeCid.
    type BuiltBulkNode = (
        String, // effective_author
        Node,
        Option<(String, mnem_core::objects::Embedding)>,
        Option<(String, mnem_core::sparse::SparseEmbed)>,
    );
    let mut built: Vec<BuiltBulkNode> = Vec::with_capacity(body.nodes.len());
    let mut results: Vec<BulkNodeEntry> = Vec::with_capacity(body.nodes.len());
    let mut embedded = 0u32;
    let mut skipped_embed = 0u32;

    for nb in body.nodes {
        // Same gating as the single-node path: caller-supplied `label`
        // is ignored unless the server was launched with
        // `MNEM_BENCH=1`. See the doc-comment on `post_node` for the
        // full rationale.
        let label = if s.allow_labels && !nb.label.trim().is_empty() {
            nb.label.clone()
        } else {
            Node::DEFAULT_NTYPE.to_string()
        };
        let node_id = match nb.id.as_deref() {
            Some(s) => NodeId::parse_uuid(s)
                .map_err(|e| Error::bad_request(format!("invalid caller-supplied id: {e}")))?,
            None => NodeId::new_v7(),
        };
        let mut node = Node::new(node_id, &label);
        if let Some(sum) = &nb.summary {
            node = node.with_summary(sum);
        }
        if let Some(ref ctx) = nb.context_sentence {
            node = node.with_context_sentence(ctx);
        }
        if let Some(props) = nb.props {
            for (k, v) in props {
                node = node.with_prop(
                    k,
                    json_to_ipld(&v).map_err(|e| Error::bad_request(e.to_string()))?,
                );
            }
        }
        if let Some(c) = nb.content {
            node = node.with_content(bytes::Bytes::from(c.into_bytes()));
        }
        let no_embed = nb.no_embed;
        let effective_author = nb
            .author
            .as_deref()
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .unwrap_or(author.as_str())
            .to_string();
        // Dense and sparse vectors stage to their respective sidecars via
        // `Transaction::set_embedding` / `set_sparse_embedding` after the
        // commit loop knows the NodeCid; we collect both here keyed by
        // position in `built`.
        let text_for_embed: Option<String> = node
            .summary
            .as_ref()
            .filter(|t| !t.trim().is_empty())
            .cloned();
        let mut pending_dense: Option<(String, mnem_core::objects::Embedding)> = None;
        let mut pending_sparse_item: Option<(String, mnem_core::sparse::SparseEmbed)> = None;
        if !no_embed {
            if let Some(text) = text_for_embed {
                if let Some(embedder) = embedder.as_ref() {
                    match embedder.embed(&text) {
                        Ok(v) => {
                            let emb = mnem_embed_providers::to_embedding(embedder.model(), &v);
                            pending_dense = Some((embedder.model().to_string(), emb));
                            embedded += 1;
                        }
                        Err(_) => {
                            skipped_embed += 1;
                        }
                    }
                }
                if let Some(sparser) = sparser.as_ref()
                    && let Ok(se) = sparser.encode(&text)
                {
                    pending_sparse_item = Some((sparser.vocab_id().to_string(), se));
                }
            }
        }
        let has_embedding = pending_dense.is_some();
        let node_cid_str = mnem_core::codec::hash_to_cid(&node)
            .map(|(_, c)| c.to_string())
            .unwrap_or_default();
        let mut props_map = serde_json::Map::new();
        for (k, v) in &node.props {
            props_map.insert(k.clone(), ipld_to_json(v));
        }
        results.push(BulkNodeEntry {
            id: node.id.to_uuid_string(),
            cid: node_cid_str,
            label,
            summary: node.summary.clone(),
            context_sentence: node.context_sentence.clone(),
            content_bytes: node.content.as_ref().map_or(0, bytes::Bytes::len),
            props: serde_json::Value::Object(props_map),
            has_embedding,
            tombstoned: false,
        });
        built.push((effective_author, node, pending_dense, pending_sparse_item));
    }

    // Commit nodes grouped by consecutive effective author. The common
    // all-same-author case produces a single commit (unchanged behaviour).
    // When per-node `author` fields introduce different effective authors,
    // consecutive runs of the same author are batched into one transaction
    // each, matching what multiple single-node calls would produce.
    let bulk_message = body.message.as_deref().unwrap_or("mnem http bulk add");
    let mut guard = s.repo.lock().map_err(|_| Error::locked())?;
    let mut op_id = String::new();
    let mut group_start = 0usize;
    while group_start < built.len() {
        let group_author = &built[group_start].0;
        let group_end = built[group_start..]
            .iter()
            .position(|(a, _, _, _)| a != group_author)
            .map(|rel| group_start + rel)
            .unwrap_or(built.len());
        let mut tx = guard.start_transaction();
        for (_, node, pending_dense, pending_sparse_item) in &built[group_start..group_end] {
            let cid = tx.add_node(node)?;
            if let Some((model, emb)) = pending_dense {
                tx.set_embedding(cid.clone(), model.clone(), emb.clone())?;
            }
            if let Some((vocab_id, se)) = pending_sparse_item {
                tx.set_sparse_embedding(cid, vocab_id.clone(), se.clone())?;
            }
        }
        let commit_start = std::time::Instant::now();
        let new_repo = tx.commit(group_author, bulk_message)?;
        s.metrics
            .commit_duration
            .observe(commit_start.elapsed().as_secs_f64());
        op_id = new_repo.op_id().to_string();
        *guard = new_repo;
        group_start = group_end;
    }

    Ok(Json(BulkNodeResp {
        schema: "mnem.v1.post-nodes-bulk",
        op_id,
        results,
        embedded,
        skipped_embed,
    }))
}

// ---------- GET /v1/retrieve ----------

#[derive(Debug, Deserialize)]
pub(crate) struct RetrieveQuery {
    pub text: Option<String>,
    pub label: Option<String>,
    #[serde(default)]
    pub budget: Option<u32>,
    #[serde(default)]
    pub limit: Option<usize>,
    /// `KEY=VALUE`; VALUE tried as JSON first, falls back to string.
    pub where_eq: Option<String>,

    // Full-pipeline knobs — mirrors every POST /v1/retrieve field
    // that can be expressed as a scalar URL parameter.
    pub vector_cap: Option<usize>,
    /// Server-side embed model to use for the dense lane.
    pub vector_model: Option<String>,
    /// Cross-encoder reranker spec (`PROVIDER:MODEL`).
    pub rerank: Option<String>,
    pub rerank_top_k: Option<usize>,
    /// Enable the community expander (additive, never drops candidates).
    pub community_filter: Option<bool>,
    pub community_expand_seeds: Option<usize>,
    pub community_max_per: Option<usize>,
    pub community_decay: Option<f32>,
    pub graph_expand: Option<usize>,
    pub graph_decay: Option<f32>,
    /// Comma-separated authored-edge type filter, e.g. `edge:knows,edge:follows`.
    pub graph_etype: Option<String>,
    pub graph_depth: Option<usize>,
    pub graph_max_per_seed: Option<usize>,
    /// Graph expansion strategy: `"decay"` (default) or `"ppr"`.
    pub graph_mode: Option<String>,
    pub ppr_damping: Option<f32>,
    pub ppr_iter: Option<u32>,
    pub ppr_opt_in: Option<bool>,
    pub summarize: Option<bool>,
    pub summarize_k: Option<usize>,
    /// HyDE activation — any non-empty string enables it, e.g. `"default"`.
    pub hyde: Option<String>,
    pub hyde_max_tokens: Option<u32>,
    pub hyde_temperature: Option<f32>,
    pub multi_query: Option<usize>,
    /// Skip dense/sparse embedding even when a provider is configured.
    pub no_vector: Option<bool>,
    /// Comma-separated post-retrieval label (ntype) filter, e.g. `Fact,Entity:Person`.
    pub labels: Option<String>,
}

impl RetrieveQuery {
    fn into_request(self) -> RetrieveRequest {
        let split_csv = |s: Option<String>| -> Vec<String> {
            s.as_deref()
                .map(|v| {
                    v.split(',')
                        .map(str::trim)
                        .filter(|t| !t.is_empty())
                        .map(String::from)
                        .collect()
                })
                .unwrap_or_default()
        };
        RetrieveRequest {
            text: self.text,
            label: self.label,
            where_eq: self.where_eq,
            budget: self.budget,
            limit: self.limit,
            vector_cap: self.vector_cap,
            vector_model: self.vector_model,
            vector: None, // GET callers cannot supply raw embedding vectors
            rerank: self.rerank,
            rerank_top_k: self.rerank_top_k,
            community_filter: self.community_filter,
            community_min_coverage: None,
            community_expand_seeds: self.community_expand_seeds,
            community_max_per: self.community_max_per,
            community_decay: self.community_decay,
            graph_expand: self.graph_expand,
            graph_decay: self.graph_decay,
            graph_etype: {
                let v = split_csv(self.graph_etype);
                if v.is_empty() { None } else { Some(v) }
            },
            graph_depth: self.graph_depth,
            graph_max_per_seed: self.graph_max_per_seed,
            graph_mode: self.graph_mode,
            ppr_damping: self.ppr_damping,
            ppr_iter: self.ppr_iter,
            ppr_opt_in: self.ppr_opt_in,
            summarize: self.summarize,
            summarize_k: self.summarize_k,
            hyde: self.hyde,
            hyde_max_tokens: self.hyde_max_tokens,
            hyde_temperature: self.hyde_temperature,
            multi_query: self.multi_query,
            no_vector: self.no_vector,
            labels: split_csv(self.labels),
            explain: None,
        }
    }
}

#[tracing::instrument(skip(s))]
pub(crate) async fn retrieve(
    State(s): State<AppState>,
    Query(q): Query<RetrieveQuery>,
) -> Result<Json<Value>, Error> {
    retrieve_impl(s, q.into_request())
}

// ---------- POST /v1/retrieve (full retrieval pipeline) ----------
//
// Accepts a JSON body with every knob the CLI exposes: label, where_eq,
// text, budget, limit, vector_cap, graph_expand, rerank,
// and hints that trigger the embedder / LLM at the edges.
//
// HyDE and multi-query require a configured LLM provider and are
// gated behind explicit fields; the handler replies with `llm_skipped`
// metadata when the caller asks for either without supplying a
// provider config inline.
//
// Same adapter-failure policy as the CLI: every optional tier that
// errors out is logged and skipped; the base hybrid retrieval always
// runs.

#[derive(Deserialize, Default)]
pub(crate) struct RetrieveRequest {
    // Basic filters
    #[serde(default)]
    pub text: Option<String>,
    #[serde(default)]
    pub label: Option<String>,
    #[serde(default)]
    pub where_eq: Option<String>,
    #[serde(default)]
    pub budget: Option<u32>,
    #[serde(default)]
    pub limit: Option<usize>,

    // Ranker caps (fixes the hardcoded 256 silent truncation)
    #[serde(default)]
    pub vector_cap: Option<usize>,

    // Semantic vector (caller may supply an embedding directly OR
    // name an embedder configured on the server).
    // `embed_model` is accepted as an alias to match the CLI --embed-model flag.
    #[serde(default, alias = "embed_model")]
    pub vector_model: Option<String>,
    #[serde(default)]
    pub vector: Option<Vec<f32>>,

    // Tier 3: cross-encoder reranker, PROVIDER:MODEL
    #[serde(default)]
    pub rerank: Option<String>,
    #[serde(default)]
    pub rerank_top_k: Option<usize>,

    // Experiment E1 (C3 FIX-1 v2): community **expander**. Despite
    // the legacy field name `community_filter`, this knob now wires
    // the ADDITIVE expander - it never drops candidates, only pulls
    // in community-cohesive siblings of the top seeds. Matrix v4
    // pinned a -29pp R@10 regression on the old drop-filter
    // semantic, which is why the semantic is inverted here. Flag
    // absent or `false` preserves the byte-exact pass-through
    // contract.
    #[serde(default)]
    pub community_filter: Option<bool>,
    /// Legacy knob retained for wire-compat with v0.1.0 clients.
    /// **Ignored** under the expander semantic: the expander has no
    /// coverage threshold because it never drops candidates.
    #[serde(default)]
    pub community_min_coverage: Option<f32>,
    /// Expander: number of top candidates treated as seeds whose
    /// communities get expanded. Default 3.
    #[serde(default)]
    pub community_expand_seeds: Option<usize>,
    /// Expander: per-community cap on how many additional members
    /// are pulled in. Default 10.
    #[serde(default)]
    pub community_max_per: Option<usize>,
    /// Expander: score decay applied to expanded members relative
    /// to the seed score. Default 0.85.
    #[serde(default)]
    pub community_decay: Option<f32>,

    // Tier 2: graph expansion
    #[serde(default)]
    pub graph_expand: Option<usize>,
    #[serde(default)]
    pub graph_decay: Option<f32>,
    #[serde(default)]
    pub graph_etype: Option<Vec<String>>,
    /// Multi-hop traversal depth. `1` = single-hop (the classic
    /// graph-expand). `2+` enables MuSiQue-style compositional
    /// queries. Clamped internally to `[1, 4]`.
    #[serde(default)]
    pub graph_depth: Option<usize>,
    /// Per-seed outgoing-edge cap. Prevents a hot-seed node with
    /// hundreds of out-edges from starving sibling seeds in the
    /// global `graph_expand` budget.
    #[serde(default)]
    pub graph_max_per_seed: Option<usize>,
    /// Graph-expand strategy. `"decay"` (default) runs the classic
    /// BFS; `"ppr"` switches to personalised PageRank (E2+). PPR
    /// falls through to the decay walk when the repo has no wired
    /// adjacency index.
    #[serde(default)]
    pub graph_mode: Option<String>,
    /// PPR damping factor (default 0.85). Ignored unless
    /// `graph_mode = "ppr"`.
    #[serde(default)]
    pub ppr_damping: Option<f32>,
    /// PPR power-iteration cap (default 15). Ignored unless
    /// `graph_mode = "ppr"`.
    #[serde(default)]
    pub ppr_iter: Option<u32>,
    /// Gap 02 #17: opt in to running PPR even when the graph
    /// exceeds `PPR_DEFAULT_MAX_NODES` (250000). Default `false`
    /// (size gate active). Ignored unless `graph_mode = "ppr"`.
    #[serde(default)]
    pub ppr_opt_in: Option<bool>,
    /// E4 T2: Centroid + MMR extractive summarization on the top-M
    /// candidates. `summarize=false` (or absent) is a no-op; no
    /// `summary` field is emitted into the response.
    #[serde(default)]
    pub summarize: Option<bool>,
    /// How many summary sentences to emit. Defaults to 3 when
    /// `summarize=true` and this field is absent.
    #[serde(default)]
    pub summarize_k: Option<usize>,

    // HyDE (Hypothetical Document Embeddings): when set, an LLM generates
    // a hypothetical answer to the query and the embedder input becomes
    // `"{query}\n{passage}"` instead of the raw query. Mirrors `--hyde`.
    // Value is an optional `PROVIDER:MODEL` override; use `"default"` or
    // any non-empty string to activate with the server's configured LLM.
    #[serde(default)]
    pub hyde: Option<String>,
    /// Max tokens for the HyDE passage LLM call. Defaults to 200.
    #[serde(default)]
    pub hyde_max_tokens: Option<u32>,
    /// Temperature for the HyDE passage LLM call. Defaults to 0.7.
    #[serde(default)]
    pub hyde_temperature: Option<f32>,

    // Multi-query RAG-Fusion: generate N query paraphrases via LLM,
    // embed each + original, run N+1 sub-retrievals, RRF-fuse. Mirrors
    // `--multi-query N`. 0 or absent disables; requires `[llm]` + `[embed]`.
    #[serde(default)]
    pub multi_query: Option<usize>,

    // Force text-only: skip vector embedding even when an embedder is
    // configured. Mirrors `--no-vector`. Useful for ablation or when
    // the caller supplies `where_eq` filters only.
    #[serde(default)]
    pub no_vector: Option<bool>,

    // Post-retrieval label (ntype) filter. Mirrors CLI `--label` which
    // accepts multiple values. Items whose `ntype` is not in this list
    // are dropped from the response. Applied after all retrieval stages
    // (HyDE, multi-query, graph expand) to match CLI semantics exactly.
    // `label` (singular) remains a pre-retrieval filter for backward compat.
    #[serde(default)]
    pub labels: Vec<String>,

    // Mirrors CLI `--explain`: when true, each item in the response
    // includes `lane_scores` (per-retrieval-lane contribution). Absent
    // or `false` omits `lane_scores` for a cleaner wire.
    #[serde(default)]
    pub explain: Option<bool>,
}

#[tracing::instrument(skip(s, body))]
pub(crate) async fn retrieve_full(
    State(s): State<AppState>,
    Json(body): Json<RetrieveRequest>,
) -> Result<Json<Value>, Error> {
    retrieve_impl(s, body)
}

fn retrieve_impl(s: AppState, body: RetrieveRequest) -> Result<Json<Value>, Error> {
    // Clamp untrusted numeric knobs before we touch the retriever.
    // See the `MAX_RETRIEVE_LIMIT` / `MAX_VECTOR_CAP` / `MAX_RERANK_TOP_K`
    // constants at the top of this file for rationale.
    clamp_or_reject("limit", body.limit, MAX_RETRIEVE_LIMIT)?;
    clamp_or_reject("vector_cap", body.vector_cap, MAX_VECTOR_CAP)?;
    clamp_or_reject("rerank_top_k", body.rerank_top_k, MAX_RERANK_TOP_K)?;
    clamp_or_reject("multi_query", body.multi_query, MAX_MULTI_QUERY)?;
    clamp_or_reject("community_expand_seeds", body.community_expand_seeds, MAX_COMMUNITY_EXPAND_SEEDS)?;
    clamp_or_reject("community_max_per", body.community_max_per, MAX_COMMUNITY_MAX_PER)?;
    clamp_or_reject_u32("ppr_iter", body.ppr_iter, MAX_PPR_ITER)?;
    clamp_or_reject("graph_expand", body.graph_expand, MAX_GRAPH_EXPAND)?;
    clamp_or_reject("graph_depth", body.graph_depth, MAX_GRAPH_DEPTH)?;
    clamp_or_reject("graph_max_per_seed", body.graph_max_per_seed, MAX_GRAPH_MAX_PER_SEED)?;
    reject_vec_too_long("graph_etype", &body.graph_etype, MAX_GRAPH_ETYPE_COUNT)?;
    if let Some(v) = &body.vector
        && v.len() > MAX_VECTOR_DIM
    {
        return Err(Error::bad_request(format!(
            "vector has {} dimensions, exceeds max of {MAX_VECTOR_DIM}",
            v.len()
        )));
    }
    if let Some(t) = &body.text
        && t.len() > MAX_QUERY_TEXT_LEN
    {
        return Err(Error::bad_request(format!(
            "text query length {} exceeds maximum of {MAX_QUERY_TEXT_LEN} bytes",
            t.len()
        )));
    }

    let repo = s.repo.lock().map_err(|_| Error::locked())?.clone();
    let mut ret = repo.retrieve();
    let mut skipped: Vec<String> = Vec::new();
    // Gap 14: structural warnings[]. Populated from compile-time
    // constants only (see `mnem_core::retrieve::warnings`). Every
    // push below is guarded by a structural precondition (substrate
    // count == 0, provider open error, etc.) so the array stays
    // small; `cap_warnings` enforces the hard cap before we
    // serialise.
    let mut warnings: Vec<mnem_core::retrieve::Warning> = Vec::new();

    // Label filter gated by `MNEM_BENCH`. See `post_node` doc-comment
    // for the full rationale.
    if s.allow_labels
        && let Some(l) = &body.label
    {
        ret = ret.label(l.clone());
    }
    if let Some(w) = &body.where_eq {
        let (k, v) = parse_kv(w).map_err(Error::bad_request)?;
        ret = ret.where_prop(k, PropPredicate::Eq(v));
    }
    if let Some(b) = body.budget {
        ret = ret.token_budget(b);
    }
    if let Some(n) = body.limit {
        ret = ret.limit(n);
    }
    if let Some(n) = body.vector_cap {
        ret = ret.vector_cap(n);
    }

    // Multi-query RAG-Fusion: generate N paraphrase variants via LLM,
    // embed each + original, run N+1 sub-retrievals, then RRF-fuse.
    // Mirrors CLI `--multi-query N`. Returns early on success, falls
    // through to plain retrieve on any error so callers always get a
    // result.
    if let Some(n_variants) = body.multi_query
        && n_variants > 0
        && let Some(q) = body.text.as_deref()
        && let Some(lc) = &s.llm_cfg
        && let Some(pc) = &s.embed_cfg
    {
        match run_multi_query_http(&repo, q, n_variants, body.limit, body.budget, body.vector_cap, body.no_vector.unwrap_or(false), lc, pc) {
            Ok(Some(result)) => {
                let items_json: Vec<Value> = result
                    .items
                    .iter()
                    .map(|it| {
                        let mut m = serde_json::json!({
                            "id":       it.node.id.to_uuid_string(),
                            "label":    it.node.ntype,
                            "summary":  it.node.summary,
                            "rendered": it.rendered,
                            "score":    it.score,
                            "tokens":   it.tokens,
                        });
                        if let Some(obj) = m.as_object_mut() {
                            if !it.node.props.is_empty() {
                                obj.insert(
                                    "props".into(),
                                    serde_json::to_value(&it.node.props).unwrap_or(Value::Null),
                                );
                            }
                        }
                        m
                    })
                    .collect();
                return Ok(Json(serde_json::json!({
                    "schema":      "mnem.v1.retrieve",
                    "mode":        "multi_query",
                    "count":       items_json.len(),
                    "items":       items_json,
                    "tokens_used": result.tokens_used,
                    "dropped":     result.dropped,
                    "skipped":     skipped,
                })));
            }
            Ok(None) => {
                skipped.push("multi_query: no variants generated; falling back to plain retrieve".into());
            }
            Err(e) => {
                skipped.push(format!("multi_query error: {e}; falling back to plain retrieve"));
            }
        }
    }

    // HyDE (Hypothetical Document Embeddings): if `hyde` is set (any
    // non-empty value) AND we have a text query AND an LLM is configured,
    // ask the LLM to generate a hypothetical answer and replace the
    // embedder input with `"{query}\n{passage}"`. Mirrors CLI `--hyde`.
    // LLM failures fall back silently to the plain query text.
    let mut embedder_text: Option<String> = body.text.clone();
    if body.hyde.is_some()
        && let Some(q) = body.text.as_deref()
        && let Some(lc) = &s.llm_cfg
    {
        use mnem_core::llm::{GenOptions, HYDE_PROMPT_TEMPLATE, fill_template};
        match mnem_llm_providers::open(lc) {
            Ok(llm) => {
                let prompt = fill_template(HYDE_PROMPT_TEMPLATE, q);
                let opts = GenOptions {
                    n: 1,
                    max_tokens: Some(body.hyde_max_tokens.unwrap_or(200)),
                    temperature: Some(body.hyde_temperature.unwrap_or(0.7)),
                    ..Default::default()
                };
                match llm.generate(&prompt, &opts) {
                    Ok(mut passages) if !passages.is_empty() => {
                        let passage = passages.remove(0);
                        embedder_text = Some(format!("{q}\n{passage}"));
                    }
                    Ok(_) | Err(_) => {
                        skipped.push("hyde: empty or failed completion; using plain query".into());
                    }
                }
            }
            Err(e) => {
                skipped.push(format!("hyde: LLM open failed ({e}); using plain query"));
            }
        }
    }

    // Vector: caller-supplied embedding takes priority over
    // server-side auto-fuse. When no vector is supplied AND the
    // server has an embed provider configured, embed the text
    // query with it so the retrieve fires the real hybrid path.
    // This matches the CLI behaviour (commands.rs retrieve).
    //
    // Post-there is no text ranker left in mnem-core: a
    // `text` query without either (a) a caller-supplied vector or
    // (b) a configured embedder is rejected with 400.
    let no_vector = body.no_vector.unwrap_or(false);
    let mut vector_model: Option<String> = None;
    let mut sparse_vocab: Option<String> = None;
    // query_text always uses the original query (text filter / BM25).
    if let Some(text) = body.text.as_deref()
        && !text.trim().is_empty()
    {
        ret = ret.query_text(text.to_string());
    }
    // Caller-supplied vector wins over auto-embed and is NOT gated by
    // no_vector (the caller already did the embedding themselves).
    if let (Some(m), Some(v)) = (&body.vector_model, &body.vector) {
        vector_model = Some(m.clone());
        ret = ret.vector(m.clone(), v.clone());
    } else if !no_vector
        && let Some(text) = embedder_text.as_deref()
        && !text.trim().is_empty()
        && let Some(pc) = &s.embed_cfg
    {
        // Use embedder_text here so HyDE-extended text is embedded when
        // `hyde` is set; otherwise embedder_text == body.text.
        let embedder = mnem_embed_providers::open(pc)
            .map_err(|e| Error::bad_request(format!("embed open failed: {e}")))?;
        let qvec = embedder
            .embed(text)
            .map_err(|e| Error::bad_request(format!("embed call failed: {e}")))?;
        vector_model = Some(embedder.model().to_string());
        ret = ret.vector(embedder.model().to_string(), qvec);
    }
    // Sparse lane: auto-encode via configured provider. Uses the
    // inference-free query path when the adapter overrides
    // `encode_query` (OpenSearch v3-distill). Gated by no_vector.
    if !no_vector
        && let Some(text) = embedder_text.as_deref()
        && !text.trim().is_empty()
        && let Some(sc) = &s.sparse_cfg
    {
        let sparser = mnem_sparse_providers::open(sc)
            .map_err(|e| Error::internal(format!("sparse provider open failed: {e}")))?;
        let sq = sparser
            .encode_query(text)
            .map_err(|e| Error::internal(format!("sparse encode failed: {e}")))?;
        sparse_vocab = Some(sq.vocab_id.clone());
        ret = ret.sparse_query(sq);
    }
    // BENCH-1 (C4 audit): cold-start fallback. See sibling block in
    // `retrieve()` above for full rationale. When the caller passes a
    // text query, has supplied no inline `vector`, and the server has
    // no `[embed]` / `[sparse]` configured, fall back to the
    // deterministic `MockEmbedder` (blake3-derived, dim=384) instead
    // of returning 400. Adds a `skipped[]` entry + a structural
    // warning so callers see the degradation in the response.
    // Skipped entirely when no_vector is set.
    if !no_vector
        && embedder_text.as_deref().is_some_and(|t| !t.trim().is_empty())
        && vector_model.is_none()
        && sparse_vocab.is_none()
        && body.vector.is_none()
    {
        if let Some(text) = embedder_text.as_deref() {
            let mock = mnem_embed_providers::MockEmbedder::new("mock:cold-start-384", 384);
            let qvec = mock
                .embed(text)
                .map_err(|e| Error::internal(format!("mock embed failed: {e}")))?;
            vector_model = Some(mock.model().to_string());
            ret = ret.vector(mock.model().to_string(), qvec);
            skipped.push(
                "embed: cold-start MockEmbedder fallback (no [embed]/[sparse] configured)"
                    .to_string(),
            );
            tracing::warn!(
                "retrieve_full: no [embed]/[sparse] configured; using deterministic \
 MockEmbedder fallback (cold-start). Configure a real provider in \
 config.toml for production retrieval quality."
            );
        }
    }

    // Attach cached indexes (audit fix G1): skip O(N) rebuild on every
    // retrieve by reusing per-commit-CID cached indexes. Commit
    // invalidation is automatic via op-id compare inside IndexCache.
    //
    // C3 Patch-B: also capture the vector-index handle so the
    // community_filter + ppr blocks below can feed it to the
    // `GraphCache` KNN-edge fallback when the authored adjacency is
    // empty (E0 wire activation).
    let mut vector_idx_for_graph: Option<std::sync::Arc<mnem_core::index::BruteForceVectorIndex>> =
        None;
    {
        let mut cache = s.indexes.lock().map_err(|_| Error::locked())?;
        if let Some(model) = &vector_model {
            let idx = cache.vector_index(&repo, model)?;
            vector_idx_for_graph = Some(idx.clone());
            ret = ret.with_vector_index(idx);
        }
        if let Some(vocab) = &sparse_vocab {
            let idx = cache.sparse_index(&repo, vocab)?;
            ret = ret.with_sparse_index(idx);
        }
    }

    // Tier 3: rerank via adapter.
    if let Some(spec) = &body.rerank {
        match parse_rerank_spec(spec) {
            Ok(cfg) => match mnem_rerank_providers::open(&cfg) {
                Ok(rr) => {
                    ret = ret.with_reranker(rr);
                    if let Some(k) = body.rerank_top_k {
                        ret = ret.rerank_top_k(k);
                    }
                }
                Err(e) => {
                    skipped.push(format!("rerank: {e}"));
                    // Gap 14: structural warning. The detailed error
                    // goes on `skipped` (runtime string, includes
                    // provider diagnostics); the warning is the
                    // agent-routable compile-time-constant version.
                    warnings.push(mnem_core::retrieve::Warning::for_code(
                        mnem_core::retrieve::WarningCode::NoReranker,
                    ));
                }
            },
            Err(e) => {
                skipped.push(format!("rerank spec: {e}"));
                warnings.push(mnem_core::retrieve::Warning::for_code(
                    mnem_core::retrieve::WarningCode::NoReranker,
                ));
            }
        }
    }

    // C3 FIX-1 v2: CommunityExpander runtime. When the caller sets
    // `community_filter: true` (legacy field name; the semantic is
    // now ADDITIVE expansion, not filter-drop), fetch (or build) a
    // Leiden community assignment over the authored-edges adjacency.
    // When that authored adjacency is empty (common under
    // `/v1/nodes/bulk` which does not author edges), fall back to a
    // deterministic KNN-edge substrate derived from the active vector
    // index (k=32, cosine). Expander is additive: it never drops
    // candidates, so worst case is neutral. Zero-impact when the flag
    // is absent or `false`.
    if body.community_filter.unwrap_or(false) {
        // Gap 14: detect substrate emptiness BEFORE building the
        // Leiden assignment. `has_vectors` is derived from the
        // already-captured `vector_idx_for_graph` handle;
        // `has_authored_edges` is derived from the graph_cache
        // adjacency slot which is populated lazily on first use.
        // Both checks are O(1) structural predicates.
        let has_vectors = vector_idx_for_graph
            .as_deref()
            .is_some_and(|v| !v.is_empty());
        let has_authored_edges = match s.graph_cache.lock() {
            Ok(gc) => gc.adjacency.as_ref().is_some_and(|a| !a.edges.is_empty()),
            Err(_) => false,
        };
        if !has_vectors && !has_authored_edges {
            warnings.push(mnem_core::retrieve::Warning::for_code(
                mnem_core::retrieve::WarningCode::CommunityFilterNoop,
            ));
        }
        let assignment = {
            let mut gc = s.graph_cache.lock().map_err(|_| Error::locked())?;
            gc.hybrid_community_for(&repo, vector_idx_for_graph.as_deref())?
        };
        let expand_seeds = body.community_expand_seeds.unwrap_or(3);
        let max_per_community = body.community_max_per.unwrap_or(10);
        let decay = body.community_decay.unwrap_or(0.85).clamp(0.0, 1.0);
        // min_coverage is retained on the DTO but ignored at runtime
        // (expander has no coverage threshold). We keep the value in
        // the cfg so debug logs reflect what the client sent.
        let min_coverage = body.community_min_coverage.unwrap_or(0.5).clamp(0.0, 1.0);
        let cfg = mnem_core::retrieve::CommunityFilterCfg {
            enabled: true,
            expand_seeds,
            max_per_community,
            decay,
            min_coverage,
        };
        let lookup_handle_fwd = assignment.clone();
        let lookup_handle_inv = assignment.clone();
        let lookup = std::sync::Arc::new(mnem_core::retrieve::CommunityLookup::new_with_members(
            move |nid| lookup_handle_fwd.community_of(*nid),
            move |cid| lookup_handle_inv.members_of(cid).to_vec(),
        ));
        ret = ret.with_community_filter(cfg, lookup);
    }

    // C3 FIX-2 + Patch-B: HybridAdjacency + PPR wire. When
    // `graph_mode="ppr"`, fetch (or build) the adjacency snapshot and
    // install it as the retriever's adjacency index. Uses the same
    // KNN-edge fallback so PPR becomes a real traversal instead of the
    // identity-under-empty-adjacency no-op.
    let want_ppr = body
        .graph_mode
        .as_deref()
        .is_some_and(|m| m.eq_ignore_ascii_case("ppr"));
    if want_ppr {
        // Gap 14: substrate-emptiness warning for PPR. Same
        // precondition as community_filter; PPR on an empty
        // transition matrix is the identity pass.
        let has_vectors = vector_idx_for_graph
            .as_deref()
            .is_some_and(|v| !v.is_empty());
        let has_authored_edges = match s.graph_cache.lock() {
            Ok(gc) => gc.adjacency.as_ref().is_some_and(|a| !a.edges.is_empty()),
            Err(_) => false,
        };
        if !has_vectors && !has_authored_edges {
            warnings.push(mnem_core::retrieve::Warning::for_code(
                mnem_core::retrieve::WarningCode::PprNoSubstrate,
            ));
        }
        let adj = {
            let mut gc = s.graph_cache.lock().map_err(|_| Error::locked())?;
            gc.hybrid_adjacency_for(&repo, vector_idx_for_graph.as_deref())?
        };
        ret = ret.with_adjacency_index(adj);
    }

    // Tier 2: graph expand (authored-graph traversal, mnem's moat).
    if let Some(max_expand) = body.graph_expand {
        // Gap 14: graph_expand reads authored edges only (not the
        // vector-derived KNN substrate). Emit a warning when the
        // authored adjacency is empty so the caller knows the walk
        // added nothing.
        let has_authored_edges = match s.graph_cache.lock() {
            Ok(gc) => gc.adjacency.as_ref().is_some_and(|a| !a.edges.is_empty()),
            Err(_) => false,
        };
        if !has_authored_edges {
            warnings.push(mnem_core::retrieve::Warning::for_code(
                mnem_core::retrieve::WarningCode::AuthoredAdjacencyEmpty,
            ));
        }
        let mut cfg = mnem_core::retrieve::GraphExpand {
            max_expand,
            decay: body
                .graph_decay
                .unwrap_or(mnem_core::retrieve::GraphExpand::DEFAULT_DECAY),
            etype_filter: body.graph_etype.clone(),
            ..Default::default()
        };
        if let Some(depth) = body.graph_depth {
            cfg = cfg.with_depth(depth);
        }
        if let Some(cap) = body.graph_max_per_seed {
            cfg = cfg.with_max_per_seed(cap);
        }
        // E2: PPR mode dispatch.
        if let Some(mode) = body.graph_mode.as_deref()
            && mode == "ppr"
        {
            let damping = body.ppr_damping.unwrap_or(mnem_core::ppr::DEFAULT_DAMPING);
            let iter = body.ppr_iter.unwrap_or(mnem_core::ppr::DEFAULT_MAX_ITER);
            cfg = cfg.with_ppr(damping, iter, mnem_core::ppr::DEFAULT_EPS);
        }
        ret = ret.with_graph_expand(cfg);
    }

    // Gap 02 #17: forward the caller's `ppr_opt_in` knob. When the
    // caller pinned `true`, the retriever's PPR dispatcher skips the
    // default-on size gate. Default `false` means the gate is active
    // and oversized graphs fall back to decay.
    ret = ret.with_ppr_opt_in(body.ppr_opt_in.unwrap_or(false));

    // Record retrieve-latency histogram around the fusion call itself.
    let retrieve_start = std::time::Instant::now();
    let result = ret.execute()?;
    s.metrics
        .retrieve_latency
        .observe(retrieve_start.elapsed().as_secs_f64());

    // Gap 02 #17: if the retriever's PPR dispatcher tripped its
    // size gate, emit the structured warning and bump the labelled
    // counter. The gauge is initialized once in Metrics::new; no
    // per-request set is needed.
    if result.ppr_size_gate_skipped {
        warnings.push(mnem_core::retrieve::Warning::for_code(
            mnem_core::retrieve::WarningCode::PprSizeGateSkipped,
        ));
        s.metrics
            .ppr_size_gate_skipped
            .get_or_create(&crate::metrics::PprSizeGateLabels {
                reason: "above_threshold".into(),
            })
            .inc();
    }
    // Post-retrieval label (ntype) filter — mirrors CLI `--label`.
    // Applied after all retrieval stages so HyDE/multi-query
    // expansion is unaffected by the label scope.
    let filtered_items: std::borrow::Cow<'_, [_]> = if body.labels.is_empty() {
        std::borrow::Cow::Borrowed(&result.items)
    } else {
        std::borrow::Cow::Owned(
            result
                .items
                .iter()
                .filter(|it| body.labels.iter().any(|l| l == &it.node.ntype))
                .cloned()
                .collect(),
        )
    };

    let want_explain = body.explain.unwrap_or(false);
    let items: Vec<Value> = filtered_items
        .iter()
        .map(|item| {
            let mut obj = json!({
                "id": item.node.id.to_uuid_string(),
                "label": item.node.ntype,
                "score": item.score,
                "tokens": item.tokens,
                "summary": item.node.summary,
                "rendered": item.rendered,
            });
            if want_explain && !item.lane_scores.is_empty() {
                let mut lane_obj = Map::new();
                for (lane, score) in &item.lane_scores {
                    lane_obj.insert(lane_name(*lane).to_string(), json!(score));
                }
                obj.as_object_mut()
                    .expect("obj is always a JSON object by construction")
                    .insert("lane_scores".into(), Value::Object(lane_obj));
            }
            obj
        })
        .collect();

    // Gap 16: score calibration - scale-free per-query interpretability.
    // Mirrors the GET /v1/retrieve handler above. The `score_distribution`
    // block carries min / max / median / iqr + a categorical `shape`
    // label (long-tail / uniform / bimodal / insufficient-samples) so
    // agents can interpret the dense ranking without a trained scaler.
    // Use filtered_items so the distribution reflects what is actually returned.
    let score_dist = {
        let scores: Vec<f32> = filtered_items.iter().map(|it| it.score).collect();
        mnem_graphrag::distribution_shape(&scores, mnem_graphrag::K_MIN)
    };

    // E4 T2: optional Centroid + MMR extractive summarization over
    // the retrieved items' node summaries. Activated strictly by
    // `summarize: true` in the request body; absent / false = emit
    // no `summary` field at all (zero impact when off).
    // Gap 14: structural `warnings[]` array. Omitted when empty to
    // keep the wire clean; when non-empty, it is first passed through
    // `cap_warnings` to enforce the `WARNINGS_CAP` bound, substituting
    // the synthetic `warnings_truncated` entry for any overflow.
    let warnings = mnem_core::retrieve::cap_warnings(warnings);
    let warnings_json: Vec<Value> = warnings
        .iter()
        .map(|w| {
            json!({
            "code": w.code.as_str(),
            "knob": w.knob,
            "message": w.message,
            "remediation_ref": w.remediation_ref,
            })
        })
        .collect();
    // Gap 01 (agent-hop incentive): derive four response-metadata
    // fields so an LLM agent can decide whether to chase a hop
    // without re-running retrieve. All four are pure functions of
    // `result.items`; zero extra ranker calls, zero wire-breakage
    // for callers that ignore the new keys.
    //
    // * `confidence` = 1 - S(k)/S(1) over the top-K sorted scores
    // (rank-agreement). `0.0` on degenerate (len < 2 or top
    // score non-positive) inputs. Scale-free.
    // * `suggested_neighbors` = up to 3 items beyond the top-3 seeds
    // with a clipped preview and `via = "adjacency"`. Always a
    // strict subset of the ranked items (see proptest
    // `suggested_neighbors_always_subset_of_adjacency`).
    // * `community_density` = fraction of top-K items that share
    // the modal community of the top item. `0.0` when no
    // community assignment is wired; otherwise in `[0, 1]`.
    // * `session_reservoir_ttl_s` = live value of
    // `session_reservoir::IDLE_TTL` in seconds. Mirrors the
    // `mnem_session_reservoir_ttl_effective` gauge.
    let gap01_confidence = gap01_compute_confidence(&filtered_items);
    let gap01_neighbors = gap01_suggested_neighbors(&filtered_items);
    let gap01_community_density = 0.0_f32;
    let gap01_session_reservoir_ttl_s = mnem_core::retrieve::session_reservoir::IDLE_TTL.as_secs();

    let mut response = json!({
    "schema": "mnem.v1.retrieve",
    "items": items,
    "tokens_used": result.tokens_used,
    "tokens_budget": if result.tokens_budget == u32::MAX {
    Value::Null
    } else {
    Value::from(result.tokens_budget)
    },
    "dropped": result.dropped,
    "score_distribution": score_dist,
    "candidates_seen": result.candidates_seen,
    "skipped": skipped,
    "confidence": gap01_confidence,
    "suggested_neighbors": gap01_neighbors,
    "community_density": gap01_community_density,
    "session_reservoir_ttl_s": gap01_session_reservoir_ttl_s,
    });
    if !warnings_json.is_empty() {
        response["warnings"] = Value::Array(warnings_json);
    }

    if body.summarize.unwrap_or(false) {
        clamp_or_reject("summarize_k", body.summarize_k, MAX_SUMMARIZE_K)?;
        let k = body.summarize_k.unwrap_or(3);
        // C3 FIX-4: accumulate sentences AND a per-sentence
        // centrality vector in lockstep. When PPR was active
        // (graph_mode="ppr") we reuse the retriever's final item
        // score as a PPR-aware centrality proxy; else we fall
        // back to authored-edge degree from the graph_cache
        // adjacency; else a uniform 1.0 (identical to pre-E2).
        let mut sentences: Vec<String> = Vec::new();
        let mut centrality_weights: Vec<f32> = Vec::new();
        // Build an optional NodeId -> degree map once.
        let degree_map: Option<std::collections::HashMap<NodeId, u32>> = if want_ppr {
            // Degree is derived from the same authored snapshot the
            // retriever just saw; if it isn't cached we skip the
            // degree map rather than re-walk the repo here.
            if let Ok(gc) = s.graph_cache.lock() {
                gc.adjacency.as_ref().map(|adj| {
                    let mut m: std::collections::HashMap<NodeId, u32> =
                        std::collections::HashMap::new();
                    for (src, dst) in &adj.edges {
                        *m.entry(*src).or_insert(0) += 1;
                        *m.entry(*dst).or_insert(0) += 1;
                    }
                    m
                })
            } else {
                None
            }
        } else {
            None
        };
        for it in filtered_items.iter() {
            if let Some(summary) = it.node.summary.clone() {
                sentences.push(summary);
                let w = if want_ppr {
                    // PPR-aware: use the final retrieve score
                    // (already PPR-propagated through graph_expand).
                    it.score.max(0.0)
                } else if let Some(m) = &degree_map {
                    m.get(&it.node.id).copied().unwrap_or(0) as f32
                } else {
                    1.0_f32
                };
                centrality_weights.push(w);
            }
        }
        // If no embedder is configured OR there are no sentences,
        // surface an empty summary and a skipped-reason; callers
        // can treat the absence of a non-empty summary the same
        // way they already handle missing rerank / HyDE.
        if sentences.is_empty() {
            response["summary"] = json!([]);
        } else if let Some(pc) = &s.embed_cfg {
            match mnem_embed_providers::open(pc) {
                Ok(embedder) => {
                    let centrality_vec = centrality_weights.clone();
                    let centrality =
                        move |i: usize| centrality_vec.get(i).copied().unwrap_or(1.0_f32);
                    match mnem_graphrag::summarize_community(
                        &sentences,
                        embedder.as_ref(),
                        None, // query vector optional; omitted at the HTTP edge for now
                        &centrality,
                        k,
                        0.5,
                    ) {
                        Ok(summary) => {
                            let arr: Vec<Value> = summary
                                .sentences
                                .iter()
                                .zip(summary.scores.iter())
                                .map(|(s, score)| json!({"sentence": s, "score": score}))
                                .collect();
                            response["summary"] = Value::Array(arr);
                        }
                        Err(e) => {
                            response["summary"] = json!([]);
                            response["summarize_skipped"] = json!(format!("summarize failed: {e}"));
                        }
                    }
                }
                Err(e) => {
                    response["summary"] = json!([]);
                    response["summarize_skipped"] =
                        json!(format!("embed provider open failed: {e}"));
                }
            }
        } else {
            response["summary"] = json!([]);
            response["summarize_skipped"] = json!("no [embed] provider configured on server");
        }
    }

    Ok(Json(response))
}

/// Multi-query RAG-Fusion helper used by `retrieve_full`. Mirrors the
/// CLI `run_multi_query` function. Generates `n_variants` query
/// paraphrases via the configured LLM, embeds each + the original,
/// runs N+1 sub-retrievals, and RRF-fuses the ranked lists.
///
/// Returns `Ok(None)` when the LLM produces no usable variants so the
/// caller can fall through to plain retrieve.
#[allow(clippy::too_many_arguments)]
fn run_multi_query_http(
    repo: &mnem_core::repo::ReadonlyRepo,
    query: &str,
    n_variants: usize,
    limit: Option<usize>,
    budget: Option<u32>,
    vector_cap: Option<usize>,
    no_vector: bool,
    llm_cfg: &mnem_llm_providers::ProviderConfig,
    embed_cfg: &mnem_embed_providers::ProviderConfig,
) -> Result<Option<mnem_core::retrieve::RetrievalResult>, anyhow::Error> {
    use anyhow::anyhow;
    use mnem_core::llm::{GenOptions, MULTI_QUERY_PROMPT_TEMPLATE, fill_multi_query_template};

    let llm = mnem_llm_providers::open(llm_cfg).map_err(|e| anyhow!("llm open failed: {e}"))?;
    let embedder =
        mnem_embed_providers::open(embed_cfg).map_err(|e| anyhow!("embed open failed: {e}"))?;

    let prompt = fill_multi_query_template(MULTI_QUERY_PROMPT_TEMPLATE, query, n_variants);
    let opts = GenOptions {
        n: 1,
        max_tokens: Some(512),
        temperature: Some(0.7),
        ..Default::default()
    };
    let completions = llm
        .generate(&prompt, &opts)
        .map_err(|e| anyhow!("llm generate failed: {e}"))?;
    if completions.is_empty() {
        return Ok(None);
    }
    let mut variants: Vec<String> = completions[0]
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .map(ToString::to_string)
        .collect();
    variants.retain(|v| v.as_str() != query);
    variants.truncate(n_variants);
    if variants.is_empty() {
        return Ok(None);
    }

    let mut all_queries: Vec<String> = vec![query.to_string()];
    all_queries.extend(variants);

    let mut ranked_lists: Vec<(Vec<mnem_core::id::NodeId>, f32)> = Vec::new();
    for q in &all_queries {
        let mut ret = repo.retrieve();
        ret = ret.query_text(q.clone());
        if let Some(n) = limit {
            ret = ret.limit(n.saturating_mul(3).max(30));
        }
        if let Some(n) = vector_cap {
            ret = ret.vector_cap(n);
        }
        if !no_vector
            && let Ok(qvec) = embedder.embed(q)
        {
            ret = ret.vector(embedder.model().to_string(), qvec);
        }
        let sub = ret.execute().map_err(|e| anyhow!("sub-retrieve failed: {e}"))?;
        let ids: Vec<mnem_core::id::NodeId> = sub.items.iter().map(|i| i.node.id).collect();
        ranked_lists.push((ids, 1.0));
    }

    let fused = mnem_core::retrieve::weighted_reciprocal_rank_fusion(
        &ranked_lists,
        mnem_core::retrieve::Retriever::DEFAULT_RRF_K,
    );

    let cap = limit.unwrap_or(usize::MAX);
    let budget_val = budget.unwrap_or(u32::MAX);
    let estimator: std::sync::Arc<dyn mnem_core::retrieve::TokenEstimator> =
        std::sync::Arc::new(mnem_core::retrieve::HeuristicEstimator);
    let mut items: Vec<mnem_core::retrieve::RetrievedItem> = Vec::new();
    let mut tokens_used: u32 = 0;
    let mut dropped: u32 = 0;
    let candidates_seen = u32::try_from(fused.len()).unwrap_or(u32::MAX);
    for (nid, score) in fused {
        if items.len() >= cap {
            dropped = dropped.saturating_add(1);
            continue;
        }
        let Some(node) = repo
            .lookup_node(&nid)
            .map_err(|e| anyhow!("lookup failed: {e}"))?
        else {
            continue;
        };
        let rendered = mnem_core::retrieve::render_node(&node);
        let tokens = estimator.estimate(&rendered);
        let next = tokens_used.saturating_add(tokens);
        if next > budget_val {
            dropped = dropped.saturating_add(1);
            continue;
        }
        tokens_used = next;
        items.push(mnem_core::retrieve::RetrievedItem::new(
            node, rendered, tokens, score,
        ));
    }
    Ok(Some(mnem_core::retrieve::RetrievalResult::new(
        items,
        tokens_used,
        budget_val,
        dropped,
        candidates_seen,
    )))
}

/// Parse a PROVIDER:MODEL rerank spec into a live
/// `mnem_rerank_providers::ProviderConfig`. Reads API-key env-var
/// names from the defaults shipped by mnem-rerank-providers; callers
/// who need custom env vars must set them via the `[rerank]` section
/// in `config.toml` and rely on the CLI instead.
fn parse_rerank_spec(spec: &str) -> Result<mnem_rerank_providers::ProviderConfig, String> {
    let (prov, model) = spec
        .split_once(':')
        .ok_or_else(|| format!("expected PROVIDER:MODEL, got `{spec}`"))?;
    if model.is_empty() {
        return Err(format!("empty model in `{spec}`"));
    }
    match prov {
        "cohere" => Ok(mnem_rerank_providers::ProviderConfig::Cohere(
            mnem_rerank_providers::CohereConfig {
                model: model.into(),
                ..Default::default()
            },
        )),
        "voyage" => Ok(mnem_rerank_providers::ProviderConfig::Voyage(
            mnem_rerank_providers::VoyageConfig {
                model: model.into(),
                ..Default::default()
            },
        )),
        "jina" => Ok(mnem_rerank_providers::ProviderConfig::Jina(
            mnem_rerank_providers::JinaConfig {
                model: model.into(),
                ..Default::default()
            },
        )),
        other => Err(format!(
            "unknown rerank provider `{other}`; want cohere|voyage|jina"
        )),
    }
}

// ---------- helpers ----------
//
// `json_to_ipld` is re-exported from `mnem_core::codec`; keeping one
// canonical implementation in the core crate ensures that any future
// hardening (depth cap adjustment, additional numeric rejection, ...)
// applies uniformly across CLI, HTTP, and MCP inputs. See
// `crates/mnem-core/src/codec/json.rs` for the shared logic.

fn ipld_to_json(v: &Ipld) -> Value {
    match v {
        Ipld::Null => Value::Null,
        Ipld::Bool(b) => Value::Bool(*b),
        Ipld::Integer(i) => serde_json::Number::from_i128(*i).map_or(Value::Null, Value::Number),
        Ipld::Float(f) => serde_json::Number::from_f64(*f).map_or(Value::Null, Value::Number),
        Ipld::String(s) => Value::String(s.clone()),
        Ipld::Bytes(b) => Value::String(format!("<{} bytes>", b.len())),
        Ipld::List(xs) => Value::Array(xs.iter().map(ipld_to_json).collect()),
        Ipld::Map(m) => {
            let mut out = Map::new();
            for (k, v) in m {
                out.insert(k.clone(), ipld_to_json(v));
            }
            Value::Object(out)
        }
        Ipld::Link(cid) => Value::String(cid.to_string()),
    }
}

fn parse_kv(s: &str) -> Result<(String, Ipld), String> {
    let (k, v) = s
        .split_once('=')
        .ok_or_else(|| format!("expected KEY=VALUE, got `{s}`"))?;
    if k.is_empty() {
        return Err(format!("key must not be empty in `{s}`"));
    }
    let val = match serde_json::from_str::<Value>(v) {
        Ok(json) => json_to_ipld(&json).map_err(|e| e.to_string())?,
        Err(_) => Ipld::String(v.to_string()),
    };
    Ok((k.to_string(), val))
}

// ============================================================
// Gap 01 (agent-hop incentive) helpers.
//
// All three helpers below are pure functions of the ranked items;
// they do not touch the repo, do not allocate index structures,
// and do not emit metrics on their own (the caller does, in
// `retrieve_full`).
//
// They are `pub(crate)` so the integration / proptest module
// (`tests::gap01_neighbors_proptest`) can exercise them without
// spinning up a full `AppState`.
// ============================================================

/// How many top-ranked items to treat as "seeds" when slicing the
/// neighbour list. Matches the rest of the Gap 01 spec's
/// `community_expand_seeds` default and the `max_neighbours = 3`
/// floor-c constant pinned in
/// `gap-catalog/01-agent-hop-incentive/solution.md`.
pub(crate) const GAP01_TOP_SEEDS: usize = 3;

/// Per-request cap on the number of neighbour hints emitted.
/// Floor-c constant: per-item amplification bound from
/// `SPEC §retrieve.response-budget` (aggregate response bytes
/// <= 64 KiB). See `gap-catalog/01-agent-hop-incentive/solution.md`
/// "Floor-c apparatus".
pub(crate) const GAP01_MAX_NEIGHBOURS: usize = 3;

/// Clip length for the neighbour `preview` field, in chars.
/// Bounds the response-size contribution of the hints block;
/// the value is the HTTP per-line budget used elsewhere in this
/// crate.
pub(crate) const GAP01_PREVIEW_CHARS: usize = 200;

/// Compute `confidence` as rank-agreement derived from the score
/// distribution of `items`.
///
/// `confidence = 1 - S(k) / S(1)` where `S(i)` is the i-th
/// sorted score (descending). Captures "is the top item clearly
/// ahead of the pack?" without a magic threshold. Scale-free
/// because both the numerator and denominator are drawn from
/// the same score distribution.
///
/// Returns `0.0` on degenerate input (`< 2` items, non-positive
/// top score, NaN top score).
pub(crate) fn gap01_compute_confidence(items: &[mnem_core::retrieve::RetrievedItem]) -> f32 {
    if items.len() < 2 {
        return 0.0;
    }
    let top = items[0].score;
    if !top.is_finite() || top <= 0.0 {
        return 0.0;
    }
    // `items` is already in RRF-rank order (descending score), but
    // defend against a degenerate case where ties re-order past
    // the caller's expectation by taking the raw last element.
    let tail = items[items.len() - 1].score.max(0.0);
    (1.0 - (tail / top)).clamp(0.0, 1.0)
}

/// Compute the `suggested_neighbors` list (up to
/// [`GAP01_MAX_NEIGHBOURS`] entries) from the ranked items past
/// the top [`GAP01_TOP_SEEDS`] seeds.
///
/// Each entry is `{id, preview, via}`. `via` is always
/// `"adjacency"` because neighbours are drawn from the same
/// adjacency-derived ranked list; if a future Gap 15 integration
/// sources neighbours from KNN substrate, the `via` label flips
/// to `"knn"`.
///
/// Guaranteed a subset of `items` by construction. The proptest
/// `suggested_neighbors_always_subset_of_adjacency` pins this
/// invariant across random inputs.
pub(crate) fn gap01_suggested_neighbors(
    items: &[mnem_core::retrieve::RetrievedItem],
) -> Vec<Value> {
    items
        .iter()
        .skip(GAP01_TOP_SEEDS)
        .take(GAP01_MAX_NEIGHBOURS)
        .map(|it| {
            let preview: String = it.rendered.chars().take(GAP01_PREVIEW_CHARS).collect();
            json!({
            "id": it.node.id.to_uuid_string(),
            "preview": preview,
            "via": "adjacency",
            })
        })
        .collect()
}

// ---------- POST /v1/explain (gap-06) ----------

/// Default serialisation throughput in bytes/ms used to derive
/// `max_path_bytes_total` when the caller omits `latency_budget_ms`.
pub(crate) const DEFAULT_SERIALIZATION_RATE_BYTES_PER_MS: u64 = 4_096;

/// Default per-request latency budget in milliseconds.
pub(crate) const DEFAULT_LATENCY_BUDGET_MS: u32 = 256;

/// Max per-node incoming fan-in walked during BFS. Matches
/// `Query::DEFAULT_ADJACENCY_CAP` and prevents a celebrity dst DoS.
pub(crate) const EXPLAIN_ADJACENCY_CAP: usize = 256;

/// Max BFS depth the `/v1/explain` handler will honour regardless
/// of the request. `u16` for parent-index compactness.
pub(crate) const EXPLAIN_MAX_DEPTH: u16 = 8;

/// `explain_mode` enum (Round 3 of gap-06).
#[derive(Serialize, Deserialize, Clone, Copy, Debug, Default, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub(crate) enum ExplainMode {
    /// Compact parent-pointer encoding, IDs only. Multi-tenant safe.
    #[default]
    Compact,
    /// Compact + full payloads. Requires ACL; falls back to
    /// `Compact` with a warning when requested without ACL.
    CompactFull,
}

/// Request body for `POST /v1/explain`.
#[derive(Deserialize, Debug)]
pub(crate) struct ExplainRequest {
    /// Seed node. BFS fans outward along incoming edges.
    pub node_id: String,
    /// Max depth; clamped to [`EXPLAIN_MAX_DEPTH`].
    #[serde(default = "default_explain_depth")]
    pub depth: u16,
    /// Encoding mode. Default [`ExplainMode::Compact`].
    #[serde(default)]
    pub mode: ExplainMode,
    /// Per-request latency budget in ms.
    #[serde(default)]
    pub latency_budget_ms: Option<u32>,
    /// Serialisation throughput override.
    #[serde(default)]
    pub serialization_rate_bytes_per_ms: Option<u64>,
}

fn default_explain_depth() -> u16 {
    3
}

/// Runtime derivation: `max_path_bytes_total = remaining_ms *
/// serialization_rate_bytes_per_ms`, saturating on overflow.
///
/// Exposed at the crate root (see `lib.rs`) so integration tests
/// can exercise the invariant directly.
#[must_use]
pub fn derive_max_path_bytes(remaining_ms: u32, serialization_rate_bytes_per_ms: u64) -> usize {
    u64::from(remaining_ms)
        .saturating_mul(serialization_rate_bytes_per_ms)
        .try_into()
        .unwrap_or(usize::MAX)
}

/// `POST /v1/explain`: in-band derivation path via BFS over the
/// incoming-edge adjacency index. Redacts to IDs only by default.
pub(crate) async fn explain(
    State(s): State<AppState>,
    Json(body): Json<ExplainRequest>,
) -> Result<Json<Value>, Error> {
    let seed = NodeId::parse_uuid(&body.node_id)
        .map_err(|e| Error::bad_request(format!("invalid node_id UUID: {e}")))?;
    let depth = body.depth.min(EXPLAIN_MAX_DEPTH);

    // Runtime-derived byte cap. No magic number: caller can override
    // both knobs. `.filter(|&v| v > 0)` keeps zero from silently
    // disabling the cap.
    let rate = body
        .serialization_rate_bytes_per_ms
        .filter(|&r| r > 0)
        .unwrap_or(DEFAULT_SERIALIZATION_RATE_BYTES_PER_MS);
    let budget_ms = body
        .latency_budget_ms
        .filter(|&m| m > 0)
        .unwrap_or(DEFAULT_LATENCY_BUDGET_MS);
    let max_bytes = derive_max_path_bytes(budget_ms, rate);

    // ACL gate: compact_full requires per-tenant ACL (not in a future version).
    let (effective_mode, mode_warning): (ExplainMode, Option<&'static str>) = match body.mode {
        ExplainMode::Compact => (ExplainMode::Compact, None),
        ExplainMode::CompactFull => (
            ExplainMode::Compact,
            Some("compact_full requested but no ACL is configured; falling back to compact"),
        ),
    };

    let repo = s.repo.lock().map_err(|_| Error::locked())?;

    // 404 if seed node does not exist (GAP-06).
    if repo
        .lookup_node(&seed)
        .map_err(|e| Error::internal(e.to_string()))?
        .is_none()
    {
        return Err(Error::not_found(format!(
            "no node with id={}",
            body.node_id
        )));
    }

    // BFS with parent tracking. `nodes[0]` is the seed; every step
    // carries `(parent_idx, to_idx)` into the nodes array.
    let mut nodes: Vec<NodeId> = vec![seed];
    let mut visited: std::collections::HashMap<NodeId, u32> = std::collections::HashMap::new();
    visited.insert(seed, 0);
    let mut steps: Vec<(u16, u32)> = Vec::new();
    let mut truncated_reason: Option<&'static str> = None;

    let mut frontier: Vec<u32> = vec![0];
    'bfs: for _hop in 0..depth {
        let mut next_frontier: Vec<u32> = Vec::new();
        for &parent_idx in &frontier {
            let parent_node = nodes[parent_idx as usize];
            let edges = repo
                .incoming_edges_capped(&parent_node, None, EXPLAIN_ADJACENCY_CAP)
                .map_err(Error::from)?;
            for edge in edges {
                let from = edge.src;
                if visited.contains_key(&from) {
                    continue;
                }
                // Projected wire bytes: ~32/step + ~40/node (JSON).
                let projected =
                    steps.len().saturating_mul(32) + nodes.len().saturating_mul(40) + 32;
                if projected > max_bytes {
                    truncated_reason = Some("response_budget");
                    break 'bfs;
                }
                let new_idx: u32 = nodes.len().try_into().unwrap_or(u32::MAX);
                nodes.push(from);
                visited.insert(from, new_idx);
                steps.push((u16::try_from(parent_idx).unwrap_or(u16::MAX), new_idx));
                next_frontier.push(new_idx);
            }
        }
        if next_frontier.is_empty() {
            break;
        }
        frontier = next_frontier;
    }
    if truncated_reason.is_none() && depth == EXPLAIN_MAX_DEPTH && !frontier.is_empty() {
        truncated_reason = Some("depth");
    }
    drop(repo);

    let nodes_wire: Vec<Value> = nodes
        .iter()
        .map(|n| Value::String(n.to_uuid_string()))
        .collect();
    let steps_wire: Vec<Value> = steps
        .iter()
        .map(|(p, t)| {
            json!({
            "parent_idx": p,
            "to_idx": t,
            })
        })
        .collect();

    let mut warnings: Vec<Value> = Vec::new();
    if let Some(w) = mode_warning {
        warnings.push(json!({
        "code": "explain.mode_downgraded",
        "message": w,
        }));
    }

    let mode_str = match effective_mode {
        ExplainMode::Compact => "compact",
        ExplainMode::CompactFull => "compact_full",
    };

    Ok(Json(json!({
    "schema": "mnem.v1.explain",
    "seed": seed.to_uuid_string(),
    "mode": mode_str,
    "path_source":
    format!("bfs.v1:graph_depth={depth}:edge_source=adjacency.v1"),
    "max_path_bytes_total": max_bytes,
    "latency_budget_ms": budget_ms,
    "serialization_rate_bytes_per_ms": rate,
    "nodes": nodes_wire,
    "steps": steps_wire,
    "path_truncated": truncated_reason.is_some(),
    "path_truncated_reason": truncated_reason,
    "warnings": warnings,
    })))
}

// Proptest for `byte_cap_never_exceeds_budget` lives in
// `tests/wire_explain.rs` so it runs under the integration harness
// (avoids a dependency on the pre-existing `gap01_tests` module
// whose `Node::new` call was broken by an upstream signature
// change). Callers verifying the invariant can reuse the
// `pub(crate)` `derive_max_path_bytes` function exposed above.

// ---------- GET /v1/log ----------

/// Maximum number of log entries returnable in a single request.
pub(crate) const MAX_LOG_LIMIT: usize = 500;

/// Default number of log entries returned when `limit` is not specified.
/// Matches the CLI `mnem log` default of 20.
fn default_log_limit() -> usize {
    20
}

/// Output format for `GET /v1/log`.
#[derive(serde::Deserialize, Default, Clone, Copy, Debug)]
#[serde(rename_all = "lowercase")]
pub(crate) enum LogFormat {
    /// Structured JSON array (default). Returns `application/json`.
    #[default]
    Json,
    /// One line per op: `<short-cid> <description>`. Returns `text/plain`.
    Oneline,
    /// Multi-line human-readable (mirrors `mnem log` default). Returns `text/plain`.
    Full,
}

/// Query parameters for `GET /v1/log`.
#[derive(serde::Deserialize)]
pub(crate) struct LogParams {
    /// Maximum number of entries to return (default 20, max 500). Matches CLI default.
    #[serde(default = "default_log_limit")]
    pub limit: usize,
    /// Output format: `json` (default), `oneline`, or `full`.
    #[serde(default)]
    pub format: LogFormat,
}

/// One entry in the JSON log response.
/// Field names match the CLI `mnem log --format=json` stable wire contract.
#[derive(serde::Serialize)]
struct LogEntry {
    cid: String,
    time: u64,
    timestamp: String,
    author: String,
    description: String,
    parents: Vec<String>,
    view: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    agent_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    task_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    host: Option<String>,
}

/// Produce a short-hex prefix of a CID for `oneline` output.
/// Mirrors the CLI's `short_cid` helper: skip 2 bytes, take 8.
fn short_cid_str(full: &str) -> String {
    if full.len() <= 10 {
        full.to_string()
    } else {
        full.chars().skip(2).take(8).collect()
    }
}

/// Convert microseconds-since-epoch to an RFC 3339 timestamp string.
/// Falls back to the raw integer (as a string) on overflow.
fn micros_to_rfc3339(micros: u64) -> String {
    use std::time::{Duration, UNIX_EPOCH};
    let secs = micros / 1_000_000;
    let nanos = ((micros % 1_000_000) * 1_000) as u32;
    match UNIX_EPOCH.checked_add(Duration::new(secs, nanos)) {
        Some(_t) => {
            // Format as RFC 3339 UTC without pulling in `chrono` or `time`.
            // The SystemTime Display is not RFC 3339 so we build it manually.
            let total_secs = secs;
            let s = total_secs % 60;
            let m = (total_secs / 60) % 60;
            let h = (total_secs / 3600) % 24;
            let days = total_secs / 86400;
            // Gregorian calendar: days since 1970-01-01
            let (year, month, day) = days_to_ymd(days);
            format!(
                "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}.{:06}Z",
                year,
                month,
                day,
                h,
                m,
                s,
                micros % 1_000_000,
            )
        }
        None => micros.to_string(),
    }
}

/// Convert days since Unix epoch to (year, month, day).
/// Implements the standard proleptic Gregorian calendar algorithm.
fn days_to_ymd(days: u64) -> (u64, u8, u8) {
    // Algorithm from https://howardhinnant.github.io/date_algorithms.html
    // (civil_from_days, public domain). Adapted for unsigned input.
    let z = days as i64 + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    (y as u64, m as u8, d as u8)
}

/// Read and decode one Operation from the blockstore, returning the decoded op
/// and the next CID to follow (the first parent), or `None` if this is the root.
fn read_op(
    bs: &dyn mnem_core::store::Blockstore,
    cid: &mnem_core::id::Cid,
) -> Result<(Operation, Option<mnem_core::id::Cid>), Error> {
    let bytes = bs
        .get(cid)
        .map_err(|e| Error::internal(format!("blockstore read: {e}")))?
        .ok_or_else(|| Error::internal(format!("op {cid} missing from store")))?;
    let op: Operation = from_canonical_bytes(&bytes)
        .map_err(|e| Error::internal(format!("decode op {cid}: {e}")))?;
    let next = op.parents.first().cloned();
    Ok((op, next))
}

/// `GET /v1/log` - walk the op-log backwards from the current head.
///
/// Query params:
/// - `limit`: max entries to return (default 20, max 500; matches CLI default)
/// - `format`: `json` (default) | `oneline` | `full`
///
/// JSON response: `{ "schema": "mnem.v1.log", "entries": [...], "count": N }`
/// Text responses (oneline/full): `text/plain; charset=utf-8`
pub(crate) async fn get_log(
    State(s): State<AppState>,
    Query(params): Query<LogParams>,
) -> Result<impl IntoResponse, Error> {
    if params.limit > MAX_LOG_LIMIT {
        return Err(Error::bad_request(format!(
            "limit={} exceeds max of {MAX_LOG_LIMIT}; lower the value or split the request",
            params.limit
        )));
    }
    if params.limit == 0 {
        return Err(Error::bad_request("limit must be >= 1"));
    }
    let limit = params.limit;

    let repo = s.repo.lock().map_err(|_| Error::locked())?;
    let bs = repo.blockstore().clone();
    let mut cur = repo.op_id().clone();
    drop(repo); // release the mutex before walking op-log blocks

    match params.format {
        LogFormat::Json => {
            let mut entries: Vec<LogEntry> = Vec::with_capacity(limit);
            for _ in 0..limit {
                let (op, next) = read_op(bs.as_ref(), &cur)?;
                entries.push(LogEntry {
                    cid: cur.to_string(),
                    time: op.time,
                    timestamp: micros_to_rfc3339(op.time),
                    author: op.author.clone(),
                    description: op.description.clone(),
                    parents: op.parents.iter().map(ToString::to_string).collect(),
                    view: op.view.to_string(),
                    agent_id: op.agent_id.clone(),
                    task_id: op.task_id.clone(),
                    host: op.host.clone(),
                });
                match next {
                    Some(p) => cur = p,
                    None => break,
                }
            }
            let count = entries.len();
            Ok(Json(serde_json::json!({
                "schema": "mnem.v1.log",
                "entries": entries,
                "count": count,
            }))
            .into_response())
        }

        LogFormat::Oneline => {
            let mut lines = String::new();
            for _ in 0..limit {
                let short = short_cid_str(&cur.to_string());
                let (op, next) = read_op(bs.as_ref(), &cur)?;
                let _ = writeln!(lines, "{short} {}", op.description);
                match next {
                    Some(p) => cur = p,
                    None => break,
                }
            }
            Ok(([(CONTENT_TYPE, "text/plain; charset=utf-8")], lines).into_response())
        }

        LogFormat::Full => {
            let mut text = String::new();
            for _ in 0..limit {
                let op_id_str = cur.to_string();
                let (op, next) = read_op(bs.as_ref(), &cur)?;
                let _ = writeln!(text, "op {op_id_str}");
                let _ = writeln!(text, "   time    {}us", op.time);
                if !op.author.is_empty() {
                    let _ = writeln!(text, "   author  {}", op.author);
                }
                if let Some(agent) = &op.agent_id {
                    let _ = writeln!(text, "   agent   {agent}");
                }
                if let Some(task) = &op.task_id {
                    let _ = writeln!(text, "   task    {task}");
                }
                let _ = writeln!(text, "   message {}", op.description);
                text.push('\n');
                match next {
                    Some(p) => cur = p,
                    None => break,
                }
            }
            Ok(([(CONTENT_TYPE, "text/plain; charset=utf-8")], text).into_response())
        }
    }
}

// ---------- GET /v1/export ----------

/// Hard cap on operations walked during export. Prevents runaway
/// work on very long op-logs requested without a `limit` parameter.
const MAX_EXPORT_OPS: usize = 10_000;

/// Query parameters for `GET /v1/export`.
#[derive(serde::Deserialize)]
pub(crate) struct ExportParams {
    /// Maximum number of ops to export (default: all, hard cap 10,000).
    #[serde(default)]
    pub limit: Option<usize>,
}

/// `GET /v1/export` - export all reachable blocks as NDJSON.
///
/// Walks the op-log from HEAD backwards, collects all reachable blocks
/// (deduplicated via `Blockstore::iter_from_root` on each op CID), and
/// streams them as newline-delimited JSON. Each line:
///
/// ```json
/// {"cid":"<cid-string>","hex":"<hex-encoded-bytes>"}
/// ```
///
/// Query params:
/// - `limit` - max ops to export (default: all reachable, hard cap 10,000)
///
/// Response: `application/x-ndjson`
pub(crate) async fn get_export(
    State(s): State<AppState>,
    Query(params): Query<ExportParams>,
) -> Result<impl IntoResponse, Error> {
    clamp_or_reject("limit", params.limit, MAX_EXPORT_OPS)?;
    let limit = params.limit.unwrap_or(MAX_EXPORT_OPS);

    let repo = s.repo.lock().map_err(|_| Error::locked())?;
    let bs = repo.blockstore().clone();
    let mut cur_op = repo.op_id().clone();
    drop(repo); // Release the lock before doing potentially expensive block reads.

    // Walk the op-log, collecting all block CIDs reachable from each op.
    // `iter_from_root` does dedup within a single root; we dedup across ops
    // with a visited set.
    let mut seen: std::collections::HashSet<mnem_core::id::Cid> = std::collections::HashSet::new();
    let mut blocks: Vec<(mnem_core::id::Cid, bytes::Bytes)> = Vec::new();
    let mut ops_walked = 0usize;

    loop {
        if ops_walked >= limit {
            break;
        }
        ops_walked += 1;

        // Read the op block directly (do NOT use iter_from_root on the op CID,
        // because the op's DAG-CBOR encoding embeds `parents` as IPLD links and
        // iter_from_root would recursively follow them into all ancestor ops,
        // making the limit ineffective).
        let op_bytes = bs
            .get(&cur_op)
            .map_err(|e| Error::internal(format!("blockstore read: {e}")))?
            .ok_or_else(|| Error::internal(format!("op {cur_op} missing from store")))?;
        let op: Operation = from_canonical_bytes(&op_bytes)
            .map_err(|e| Error::internal(format!("decode op {cur_op}: {e}")))?;

        // Add the op block itself.
        if seen.insert(cur_op.clone()) {
            blocks.push((cur_op.clone(), op_bytes));
        }

        // Walk the view sub-DAG (commit block + prolly tree blocks +
        // node/edge/embedding blocks). The view CID has no parent-op links,
        // so iter_from_root stays within this op's payload.
        for result in bs.iter_from_root(&op.view) {
            let (cid, data) =
                result.map_err(|e| Error::internal(format!("blockstore walk: {e}")))?;
            if seen.insert(cid.clone()) {
                blocks.push((cid, data));
            }
        }

        // Advance to the first parent op.
        match op.parents.first() {
            Some(parent) => cur_op = parent.clone(),
            None => break, // Root op reached.
        }
    }

    // Serialize as NDJSON. Each line: {"cid":"<cid>","hex":"<hex>"}
    let mut ndjson = String::new();
    for (cid, data) in &blocks {
        // Hex-encode block bytes (no base64 dep needed).
        let hex: String = data.iter().map(|b| format!("{b:02x}")).collect();
        ndjson.push_str(&format!("{{\"cid\":\"{cid}\",\"hex\":\"{hex}\"}}\n",));
    }

    Ok(([(CONTENT_TYPE, "application/x-ndjson")], ndjson).into_response())
}

// ---------- POST /v1/import ----------

/// Request body for `POST /v1/import`.
///
/// Expects `application/x-ndjson` with one block per line in the format
/// produced by `GET /v1/export`:
/// ```json
/// {"cid":"<cid-string>","hex":"<hex-encoded-bytes>"}
/// ```
///
/// `POST /v1/import` - import blocks from NDJSON stream.
///
/// Reads each line, decodes the hex bytes, verifies the CID, and writes
/// the block to the blockstore. Does NOT advance HEAD - this is a
/// block-level sync primitive only.
///
/// Response JSON:
/// ```json
/// {"imported": N, "errors": [...], "ok": true}
/// ```
pub(crate) async fn post_import(
    _auth: RequireBearer,
    State(s): State<AppState>,
    headers: axum::http::HeaderMap,
    body: axum::body::Bytes,
) -> Result<Json<Value>, Error> {
    use mnem_core::store::blockstore::recompute_cid;

    // Reject non-NDJSON/text Content-Types when the header is present.
    // Absent header is accepted (lenient, matches curl --data-binary behavior).
    if let Some(ct) = headers.get(axum::http::header::CONTENT_TYPE) {
        let ct_str = ct.to_str().unwrap_or("").trim();
        // Strip parameters (e.g. "; charset=utf-8") before comparing.
        let ct_base = ct_str.split(';').next().unwrap_or("").trim();
        if ct_base != "application/x-ndjson" && ct_base != "text/plain" {
            return Err(Error::status(
                axum::http::StatusCode::UNSUPPORTED_MEDIA_TYPE,
                format!(
                    "unsupported Content-Type '{ct_base}'; expected application/x-ndjson or text/plain"
                ),
            ));
        }
    }

    let text = std::str::from_utf8(&body)
        .map_err(|e| Error::bad_request(format!("request body is not valid UTF-8: {e}")))?;

    let repo = s.repo.lock().map_err(|_| Error::locked())?;
    let bs = repo.blockstore().clone();
    drop(repo);

    let mut imported: usize = 0;
    let mut errors: Vec<Value> = Vec::new();
    let mut imported_cids: Vec<mnem_core::id::Cid> = Vec::new();

    for (line_no, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }

        // Parse the JSON line.
        let obj: Value = match serde_json::from_str(line) {
            Ok(v) => v,
            Err(e) => {
                errors.push(json!({
                    "line": line_no + 1,
                    "error": format!("JSON parse error: {e}"),
                }));
                continue;
            }
        };

        let cid_str = match obj.get("cid").and_then(Value::as_str) {
            Some(s) => s,
            None => {
                errors.push(json!({
                    "line": line_no + 1,
                    "error": "missing or non-string \"cid\" field",
                }));
                continue;
            }
        };

        let hex_str = match obj.get("hex").and_then(Value::as_str) {
            Some(s) => s,
            None => {
                errors.push(json!({
                    "line": line_no + 1,
                    "error": "missing or non-string \"hex\" field",
                }));
                continue;
            }
        };

        // Parse the CID from its multibase string representation.
        let claimed_cid = match mnem_core::id::Cid::parse_str(cid_str) {
            Ok(c) => c,
            Err(e) => {
                errors.push(json!({
                    "line": line_no + 1,
                    "cid": cid_str,
                    "error": format!("invalid CID: {e}"),
                }));
                continue;
            }
        };

        // Decode hex bytes.
        if hex_str.len() % 2 != 0 {
            errors.push(json!({
                "line": line_no + 1,
                "cid": cid_str,
                "error": "hex string has odd length",
            }));
            continue;
        }
        let mut raw: Vec<u8> = Vec::with_capacity(hex_str.len() / 2);
        let mut parse_ok = true;
        for chunk in hex_str.as_bytes().chunks(2) {
            let hi = (chunk[0] as char).to_digit(16);
            let lo = (chunk[1] as char).to_digit(16);
            match (hi, lo) {
                (Some(h), Some(l)) => raw.push((h * 16 + l) as u8),
                _ => {
                    errors.push(json!({
                        "line": line_no + 1,
                        "cid": cid_str,
                        "error": "invalid hex character",
                    }));
                    parse_ok = false;
                    break;
                }
            }
        }
        if !parse_ok {
            continue;
        }

        let data = bytes::Bytes::from(raw);

        // CID verification: recompute and compare before writing.
        // `recompute_cid` returns `None` for unknown hash algorithms;
        // in that case we trust the claim (same policy as `Blockstore::put`).
        if let Some(computed) = recompute_cid(&claimed_cid, &data) {
            if computed != claimed_cid {
                errors.push(json!({
                    "line": line_no + 1,
                    "cid": cid_str,
                    "error": format!("CID mismatch: claimed {claimed_cid} but data hashes to {computed}"),
                }));
                continue;
            }
        }

        // Write to blockstore. `put` is idempotent for already-present blocks.
        match bs.put(claimed_cid.clone(), data) {
            Ok(()) => {
                imported += 1;
                imported_cids.push(claimed_cid);
            }
            Err(e) => {
                errors.push(json!({
                    "line": line_no + 1,
                    "cid": cid_str,
                    "error": format!("blockstore write: {e}"),
                }));
            }
        }
    }

    // C-2 FIX: advance HEAD to the commit block with the greatest `time`
    // among all imported blocks (mirrors the BUG-50 / clone.rs heuristic
    // in the CLI's `import.rs`).
    let mut best_commit: Option<(u64, mnem_core::id::Cid)> = None;
    for cid in &imported_cids {
        let Ok(Some(bytes)) = bs.get(cid) else { continue };
        let Ok(Ipld::Map(m)) = from_canonical_bytes::<Ipld>(&bytes) else { continue };
        let Some(Ipld::String(kind)) = m.get("_kind") else { continue };
        if kind != "commit" { continue; }
        let time = match m.get("time") {
            Some(Ipld::Integer(n)) => u64::try_from(*n).unwrap_or(0),
            _ => 0,
        };
        best_commit = Some(match best_commit {
            None => (time, cid.clone()),
            Some((t, _)) if time > t => (time, cid.clone()),
            Some(prev) => prev,
        });
    }

    let (head_advanced, new_head_str) = if let Some((_, commit_cid)) = best_commit {
        let mut guard = s.repo.lock().map_err(|_| Error::locked())?;
        let new_repo = guard
            .update_heads(commit_cid.clone(), "mnem-http")
            .map_err(|e| Error::internal(format!("update_heads: {e}")))?;
        let cid_str = commit_cid.to_string();
        *guard = new_repo;
        (true, Some(cid_str))
    } else {
        (false, None)
    };

    let ok = errors.is_empty();
    Ok(Json(json!({
        "schema": "mnem.v1.import",
        "imported": imported,
        "errors": errors,
        "ok": ok,
        "head_advanced": head_advanced,
        "new_head": new_head_str,
    })))
}

// ---------- GET /v1/branches ----------

/// `GET /v1/branches` - list all branches.
///
/// Returns every ref whose name begins with `refs/heads/`, annotating
/// each with its target commit CID and whether it points at the current
/// head commit (`is_current`).
///
/// Response schema: `mnem.v1.branches`
/// ```json
/// {"branches": [{"name": "main", "head": "<commit-cid>", "is_current": true}, ...]}
/// ```
pub(crate) async fn get_branches(State(s): State<AppState>) -> Result<Json<Value>, Error> {
    let repo = s.repo.lock().map_err(|_| Error::locked())?;
    let view = repo.view();
    let active_ref = view.active_branch();

    let branches: Vec<Value> = view
        .refs
        .iter()
        .filter_map(|(name, target)| {
            let short = name.strip_prefix(HEADS_PREFIX)?;
            let (head_str, is_current) = match target {
                mnem_core::objects::RefTarget::Normal { target } => {
                    let is_cur = active_ref == Some(name.as_str());
                    (target.to_string(), is_cur)
                }
                mnem_core::objects::RefTarget::Conflicted { .. } => {
                    // Expose conflicted refs with an empty head so callers
                    // can see they exist without crashing the listing.
                    (String::new(), false)
                }
            };
            Some(json!({
                "name": short,
                "head": head_str,
                "is_current": is_current,
            }))
        })
        .collect();

    Ok(Json(json!({
        "schema": "mnem.v1.branches",
        "branches": branches,
    })))
}

// ---------- POST /v1/branches ----------

/// Request body for `POST /v1/branches`.
#[derive(Deserialize)]
pub(crate) struct CreateBranchBody {
    /// Short branch name (e.g. `"feature-x"`). Stored as
    /// `refs/heads/<name>` in the View.
    pub name: String,
    /// Optional commitish (commit CID, branch name, `HEAD`, or full ref path)
    /// to point the new branch at. When absent, defaults to HEAD.
    #[serde(default)]
    pub at: Option<String>,
    /// Commit author recorded on the `update_ref` Operation.
    pub author: String,
}

/// `POST /v1/branches` - create a new branch.
///
/// Creates `refs/heads/<name>` pointing at `at` (or HEAD when absent).
/// Fails 400 if the name already exists or the repo has no commits yet.
///
/// Response schema: `mnem.v1.branch-create`
/// ```json
/// {"name": "feature-x", "head": "<commit-cid>", "created": true}
/// ```
pub(crate) async fn post_branch(
    _auth: RequireBearer,
    State(s): State<AppState>,
    Json(body): Json<CreateBranchBody>,
) -> Result<Json<Value>, Error> {
    if body.name.trim().is_empty() {
        return Err(Error::bad_request("name is required"));
    }
    if body.name.len() > 255 {
        return Err(Error::bad_request(
            "branch name exceeds maximum length of 255 characters",
        ));
    }
    validate_author(&body.author)?;
    if !is_valid_ref_name(&body.name) {
        return Err(Error::bad_request(format!(
            "invalid branch name `{}`: may not contain spaces, control characters, \
             `~`, `^`, `:`, `?`, `*`, `[`, `\\`, `@{{`, `..`, `//`, \
             or start/end with `/` or `.`, or end with `.lock`",
            body.name
        )));
    }

    let full = format!("{HEADS_PREFIX}{}", body.name);

    let mut guard = s.repo.lock().map_err(|_| Error::locked())?;

    if guard.view().refs.contains_key(&full) {
        return Err(Error::conflict(format!(
            "branch `{}` already exists",
            body.name
        )));
    }

    // Resolve the target commit CID.  Accept HEAD, branch names, full ref
    // paths, and raw commit CIDs -- matching CLI `mnem branch create <name>
    // [<start-point>]` which uses resolve_commitish for the start-point.
    let target_cid = match body.at.as_deref() {
        Some(commitish) => {
            let cid = reindex_resolve_commitish(&guard, commitish)?;
            // Verify the CID decodes as a Commit block.
            let bs = guard.blockstore().clone();
            let bytes = bs
                .get(&cid)
                .map_err(|e| Error::internal(format!("blockstore error: {e}")))?
                .ok_or_else(|| {
                    Error::not_found(format!("block {cid} not found in blockstore"))
                })?;
            if from_canonical_bytes::<Commit>(&bytes).is_err() {
                return Err(Error::bad_request(format!(
                    "`{commitish}` resolves to {cid} which does not decode as a commit; \
                     use a commit CID, branch name, or HEAD"
                )));
            }
            cid
        }
        None => guard.view().heads.first().cloned().ok_or_else(|| {
            Error::bad_request(
                "repository has no commits yet; pass `at` with a commit CID or branch name"
                    .to_string(),
            )
        })?,
    };

    let head_str = target_cid.to_string();
    let author = body.author.trim().to_string();
    let new_repo = guard
        .update_ref(
            &full,
            None,
            Some(mnem_core::objects::RefTarget::normal(target_cid)),
            &author,
        )
        .map_err(Error::from)?;
    let op_id = new_repo.op_id().to_string();
    *guard = new_repo;

    Ok(Json(json!({
        "schema": "mnem.v1.branch-create",
        "name": body.name,
        "head": head_str,
        "op_id": op_id,
        "created": true,
    })))
}

// ---------- DELETE /v1/branches/:name ----------

/// `DELETE /v1/branches/:name` - delete a branch by short name.
///
/// Removes `refs/heads/<name>`. Returns 404 if the branch does not
/// exist, 409 if the branch is the current head (i.e. its target equals
/// `view.heads.first()`).
///
/// Response schema: `mnem.v1.branch-delete`
/// ```json
/// {"deleted": "feature-x"}
/// ```
pub(crate) async fn delete_branch(
    _auth: RequireBearer,
    State(s): State<AppState>,
    Path(name): Path<String>,
    Query(q): Query<DeleteQuery>,
) -> Result<Json<Value>, Error> {
    if name.trim().is_empty() {
        return Err(Error::bad_request("branch name must not be empty"));
    }
    validate_author(&q.author)?;

    let full = format!("{HEADS_PREFIX}{name}");

    let mut guard = s.repo.lock().map_err(|_| Error::locked())?;
    let view = guard.view();

    let prev = view
        .refs
        .get(&full)
        .cloned()
        .ok_or_else(|| Error::not_found(format!("branch `{name}` does not exist")))?;

    // Refuse to delete the currently checked-out branch.
    if view.active_branch() == Some(full.as_str()) {
        return Err(Error::conflict(format!(
            "cannot delete branch `{name}`: it is the current branch"
        )));
    }

    let author = q.author.trim().to_string();
    let new_repo = guard
        .update_ref(&full, Some(&prev), None, &author)
        .map_err(Error::from)?;
    let op_id = new_repo.op_id().to_string();
    *guard = new_repo;

    Ok(Json(json!({
        "schema": "mnem.v1.branch-delete",
        "deleted": name,
        "op_id": op_id,
    })))
}

// ---------- GET/POST/DELETE /v1/tags ----------

/// `GET /v1/tags` - list all tags.
///
/// Returns every ref whose name begins with `refs/tags/`, with its
/// target CID.
///
/// Response schema: `mnem.v1.tags`
/// ```json
/// {"schema": "mnem.v1.tags", "tags": [{"name": "v1.0", "target": "<cid>"}]}
/// ```
pub(crate) async fn get_tags(State(s): State<AppState>) -> Result<Json<Value>, Error> {
    let repo = s.repo.lock().map_err(|_| Error::locked())?;
    let view = repo.view();

    let tags: Vec<Value> = view
        .refs
        .iter()
        .filter_map(|(name, target)| {
            let short = name.strip_prefix(TAGS_PREFIX)?;
            let target_str = match target {
                mnem_core::objects::RefTarget::Normal { target } => target.to_string(),
                mnem_core::objects::RefTarget::Conflicted { .. } => String::new(),
            };
            Some(json!({
                "name": short,
                "target": target_str,
            }))
        })
        .collect();

    Ok(Json(json!({
        "schema": "mnem.v1.tags",
        "tags": tags,
    })))
}

// ---------- POST /v1/tags ----------

/// Request body for `POST /v1/tags`.
#[derive(Deserialize)]
pub(crate) struct CreateTagBody {
    /// Short tag name (e.g. `"v1.0"`). Stored as `refs/tags/<name>`.
    pub name: String,
    /// Optional commit CID to point the tag at. When absent,
    /// defaults to the current HEAD commit CID.
    #[serde(default)]
    pub target: Option<String>,
    /// Commit author recorded on the `update_ref` Operation.
    pub author: String,
}

/// `POST /v1/tags` - create a tag.
///
/// Creates `refs/tags/<name>` pointing at `target` (or the current HEAD
/// commit CID when absent). Fails 400 if the name is invalid, 409 if the
/// tag already exists, 400 if the repo has no commits yet and no target is
/// supplied.
///
/// Response schema: `mnem.v1.tag-create`
/// ```json
/// {"schema": "mnem.v1.tag-create", "name": "v1.0", "target": "<cid>", "created": true}
/// ```
pub(crate) async fn post_tag(
    _auth: RequireBearer,
    State(s): State<AppState>,
    Json(body): Json<CreateTagBody>,
) -> Result<Json<Value>, Error> {
    if body.name.trim().is_empty() {
        return Err(Error::bad_request("name is required"));
    }
    if body.name.len() > 255 {
        return Err(Error::bad_request(
            "tag name exceeds maximum length of 255 characters",
        ));
    }
    validate_author(&body.author)?;
    if !is_valid_ref_name(&body.name) {
        return Err(Error::bad_request(format!(
            "invalid tag name `{}`: may not contain spaces, control characters, \
             `~`, `^`, `:`, `?`, `*`, `[`, `\\`, `@{{`, `..`, `//`, \
             or start/end with `/` or `.`, or end with `.lock`",
            body.name
        )));
    }

    let full = format!("{TAGS_PREFIX}{}", body.name);

    let mut guard = s.repo.lock().map_err(|_| Error::locked())?;

    if guard.view().refs.contains_key(&full) {
        return Err(Error::conflict(format!(
            "tag `{}` already exists",
            body.name
        )));
    }

    // Resolve the target commit CID.
    let target_cid = match body.target.as_deref() {
        Some(cid_str) => {
            let cid = mnem_core::id::Cid::parse_str(cid_str)
                .map_err(|e| Error::bad_request(format!("invalid CID `{cid_str}`: {e}")))?;
            // Verify the CID decodes as a Commit block (not an op CID).
            let bs = guard.blockstore().clone();
            let bytes = bs
                .get(&cid)
                .map_err(|e| Error::internal(format!("blockstore error: {e}")))?
                .ok_or_else(|| {
                    Error::not_found(format!("block `{cid}` not found in blockstore"))
                })?;
            if from_canonical_bytes::<Commit>(&bytes).is_err() {
                return Err(Error::bad_request(format!(
                    "`{cid_str}` does not decode as a commit; \
                     use a commit CID (not an op CID)"
                )));
            }
            cid
        }
        None => guard.view().heads.first().cloned().ok_or_else(|| {
            Error::bad_request(
                "repository has no commits yet; pass `target` with a commit CID".to_string(),
            )
        })?,
    };

    let target_str = target_cid.to_string();
    let author = body.author.trim().to_string();
    let new_repo = guard
        .update_ref(
            &full,
            None,
            Some(mnem_core::objects::RefTarget::normal(target_cid)),
            &author,
        )
        .map_err(Error::from)?;
    let op_id = new_repo.op_id().to_string();
    *guard = new_repo;

    Ok(Json(json!({
        "schema": "mnem.v1.tag-create",
        "name": body.name,
        "target": target_str,
        "op_id": op_id,
        "created": true,
    })))
}

// ---------- DELETE /v1/tags/:name ----------

/// `DELETE /v1/tags/:name` - delete a tag by short name.
///
/// Removes `refs/tags/<name>`. Returns 404 if the tag does not exist.
/// Unlike branches there is no "current tag" concept, so no 409.
///
/// Response schema: `mnem.v1.tag-delete`
/// ```json
/// {"schema": "mnem.v1.tag-delete", "deleted": "v1.0"}
/// ```
pub(crate) async fn delete_tag(
    _auth: RequireBearer,
    State(s): State<AppState>,
    Path(name): Path<String>,
    Query(q): Query<DeleteQuery>,
) -> Result<Json<Value>, Error> {
    if name.trim().is_empty() {
        return Err(Error::bad_request("tag name must not be empty"));
    }
    validate_author(&q.author)?;

    let full = format!("{TAGS_PREFIX}{name}");

    let mut guard = s.repo.lock().map_err(|_| Error::locked())?;
    let view = guard.view();

    let prev = view
        .refs
        .get(&full)
        .cloned()
        .ok_or_else(|| Error::not_found(format!("tag `{name}` does not exist")))?;

    let author = q.author.trim().to_string();
    let new_repo = guard
        .update_ref(&full, Some(&prev), None, &author)
        .map_err(Error::from)?;
    let op_id = new_repo.op_id().to_string();
    *guard = new_repo;

    Ok(Json(json!({
        "schema": "mnem.v1.tag-delete",
        "deleted": name,
        "op_id": op_id,
    })))
}

// ---------- POST /v1/diff ----------

/// Default maximum diff entries returned per category (added/removed/changed)
/// per tree (nodes, edges).
const DIFF_DEFAULT_LIMIT: usize = 500;

/// Hard cap on diff entries per category per tree.
const DIFF_MAX_LIMIT: usize = 2_000;

/// Query parameters for `POST /v1/diff`.
#[derive(Deserialize, Default)]
pub(crate) struct DiffQueryParams {
    /// Cap the number of entries in each added/removed/changed bucket.
    /// Default 500, max 2000.
    #[serde(default)]
    pub limit: Option<usize>,
}

/// Query parameters for `GET /v1/diff`.
#[derive(Deserialize)]
pub(crate) struct GetDiffParams {
    /// "from" side: raw CID, op CID, `HEAD`, branch name, or full ref.
    pub from: String,
    /// "to" side: raw CID, op CID, `HEAD`, branch name, or full ref.
    pub to: String,
    /// Cap the number of entries per bucket. Default 500, max 2000.
    #[serde(default)]
    pub limit: Option<usize>,
}

/// Request body for `POST /v1/diff`.
#[derive(Deserialize)]
pub(crate) struct DiffBody {
    /// "from" side: raw CID, op CID, `"HEAD"`, branch name, or full ref.
    pub from: String,
    /// "to" side: raw CID, op CID, `"HEAD"`, branch name, or full ref.
    pub to: String,
}

/// Resolve a caller-supplied CID string to a `Commit`. The string may be:
///
/// 1. An op CID - decoded as an `Operation`, then the first head commit CID
///    is resolved from the embedded `View`.
/// 2. A commit CID directly.
///
/// Returns `(commit_cid_string, Commit)` on success, or a 400/404 `Error`.
fn resolve_cid_to_commit(
    bs: &dyn mnem_core::store::Blockstore,
    cid_str: &str,
) -> Result<(mnem_core::id::Cid, Commit), Error> {
    let cid = mnem_core::id::Cid::parse_str(cid_str)
        .map_err(|e| Error::bad_request(format!("invalid CID `{cid_str}`: {e}")))?;
    let bytes = bs
        .get(&cid)
        .map_err(|e| Error::internal(format!("blockstore error: {e}")))?
        .ok_or_else(|| Error::not_found(format!("block `{cid_str}` not found in blockstore")))?;

    // Try to decode as an Operation first (it has a `view` field).
    if let Ok(op) = from_canonical_bytes::<Operation>(&bytes) {
        // Resolve the view block to get the head commit CID.
        let view_bytes = bs
            .get(&op.view)
            .map_err(|e| Error::internal(format!("blockstore error reading view: {e}")))?
            .ok_or_else(|| {
                Error::internal(format!("view block {} missing from blockstore", op.view))
            })?;
        let view: mnem_core::objects::View = from_canonical_bytes(&view_bytes)
            .map_err(|e| Error::internal(format!("decode view: {e}")))?;
        let commit_cid = view
            .heads
            .into_iter()
            .next()
            .ok_or_else(|| Error::bad_request(format!("op `{cid_str}` has no head commits")))?;
        let commit_bytes = bs
            .get(&commit_cid)
            .map_err(|e| Error::internal(format!("blockstore error reading commit: {e}")))?
            .ok_or_else(|| {
                Error::not_found(format!(
                    "commit block {} (from op `{cid_str}`) not found in blockstore",
                    commit_cid
                ))
            })?;
        let commit: Commit = from_canonical_bytes(&commit_bytes)
            .map_err(|e| Error::internal(format!("decode commit: {e}")))?;
        return Ok((commit_cid, commit));
    }

    // Try to decode as a Commit directly.
    if let Ok(commit) = from_canonical_bytes::<Commit>(&bytes) {
        return Ok((cid, commit));
    }

    Err(Error::bad_request(format!(
        "`{cid_str}` does not decode as an op or commit CID"
    )))
}

fn ref_target_to_str(t: &mnem_core::objects::RefTarget) -> String {
    match t {
        mnem_core::objects::RefTarget::Normal { target } => target.to_string(),
        mnem_core::objects::RefTarget::Conflicted { adds, .. } => adds
            .first()
            .map_or_else(|| "<conflicted>".to_string(), ToString::to_string),
    }
}

// Returns (input_cid, commit_cid, Commit, refs).
// input_cid = op CID when input is an op, else the commit CID itself.
// refs is empty when the input is a bare commit CID (no view available).
fn resolve_to_commit_and_refs(
    bs: &dyn mnem_core::store::Blockstore,
    cid_str: &str,
) -> Result<
    (
        mnem_core::id::Cid,
        mnem_core::id::Cid,
        Commit,
        std::collections::BTreeMap<String, mnem_core::objects::RefTarget>,
    ),
    Error,
> {
    let cid = mnem_core::id::Cid::parse_str(cid_str)
        .map_err(|e| Error::bad_request(format!("invalid CID `{cid_str}`: {e}")))?;
    let bytes = bs
        .get(&cid)
        .map_err(|e| Error::internal(format!("blockstore error: {e}")))?
        .ok_or_else(|| Error::not_found(format!("block `{cid_str}` not found in blockstore")))?;
    if let Ok(op) = from_canonical_bytes::<Operation>(&bytes) {
        let view_bytes = bs
            .get(&op.view)
            .map_err(|e| Error::internal(format!("blockstore error reading view: {e}")))?
            .ok_or_else(|| {
                Error::internal(format!("view block {} missing from blockstore", op.view))
            })?;
        let view: View = from_canonical_bytes(&view_bytes)
            .map_err(|e| Error::internal(format!("decode view: {e}")))?;
        let refs = view.refs.clone();
        let commit_cid = view
            .heads
            .into_iter()
            .next()
            .ok_or_else(|| Error::bad_request(format!("op `{cid_str}` has no head commits")))?;
        let commit_bytes = bs
            .get(&commit_cid)
            .map_err(|e| Error::internal(format!("blockstore error reading commit: {e}")))?
            .ok_or_else(|| {
                Error::not_found(format!(
                    "commit block {} (from op `{cid_str}`) not found in blockstore",
                    commit_cid
                ))
            })?;
        let commit: Commit = from_canonical_bytes(&commit_bytes)
            .map_err(|e| Error::internal(format!("decode commit: {e}")))?;
        return Ok((cid, commit_cid, commit, refs));
    }
    if let Ok(commit) = from_canonical_bytes::<Commit>(&bytes) {
        return Ok((cid.clone(), cid, commit, std::collections::BTreeMap::new()));
    }
    Err(Error::bad_request(format!(
        "`{cid_str}` does not decode as an op or commit CID"
    )))
}

// Shared diff computation for POST /v1/diff and GET /v1/diff.
// Produces a JSON value with CLI-parity fields (op_a/op_b, commit_a/commit_b,
// ref_deltas, node_deltas, edge_deltas) plus the existing nested nodes/edges
// objects for backward compatibility.
fn compute_diff(
    bs: &dyn mnem_core::store::Blockstore,
    from_resolved_str: &str,
    to_resolved_str: &str,
    limit: usize,
) -> Result<Value, Error> {
    let (from_op_cid, from_cid, from_commit, from_refs) =
        resolve_to_commit_and_refs(bs, from_resolved_str)?;
    let (to_op_cid, to_cid, to_commit, to_refs) =
        resolve_to_commit_and_refs(bs, to_resolved_str)?;

    let node_changes = prolly_diff(bs, &from_commit.nodes, &to_commit.nodes)
        .map_err(|e| Error::internal(format!("node diff failed: {e}")))?;
    let edge_changes = prolly_diff(bs, &from_commit.edges, &to_commit.edges)
        .map_err(|e| Error::internal(format!("edge diff failed: {e}")))?;

    // Ref deltas (CLI parity).
    let mut refs_added: Vec<Value> = Vec::new();
    let mut refs_removed: Vec<Value> = Vec::new();
    let mut refs_changed: Vec<Value> = Vec::new();
    for (name, target) in &to_refs {
        match from_refs.get(name) {
            None => refs_added.push(json!({ "name": name, "target": ref_target_to_str(target) })),
            Some(prev) if prev != target => refs_changed.push(json!({
                "name": name,
                "from": ref_target_to_str(prev),
                "to": ref_target_to_str(target),
            })),
            _ => {}
        }
    }
    for (name, target) in &from_refs {
        if !to_refs.contains_key(name) {
            refs_removed
                .push(json!({ "name": name, "target": ref_target_to_str(target) }));
        }
    }

    // Node buckets: nested (backward compat) + flat CLI-parity array.
    let mut nodes_added: Vec<Value> = Vec::new();
    let mut nodes_removed: Vec<Value> = Vec::new();
    let mut nodes_changed: Vec<Value> = Vec::new();
    let mut node_deltas: Vec<Value> = Vec::new();

    for entry in &node_changes {
        match entry {
            DiffEntry::Added { value, .. } => {
                if nodes_added.len() < limit {
                    if let Some(node) = node_from_bs(bs, value) {
                        node_deltas.push(json!({
                            "type": "added",
                            "id": node.id.to_uuid_string(),
                            "ntype": node.ntype,
                            "summary": node.summary,
                        }));
                        nodes_added.push(json!({
                            "id": node.id.to_uuid_string(),
                            "ntype": node.ntype,
                            "summary": node.summary,
                        }));
                    }
                }
            }
            DiffEntry::Removed { value, .. } => {
                if nodes_removed.len() < limit {
                    if let Some(node) = node_from_bs(bs, value) {
                        node_deltas.push(json!({
                            "type": "removed",
                            "id": node.id.to_uuid_string(),
                            "ntype": node.ntype,
                            "summary": node.summary,
                        }));
                        nodes_removed.push(json!({
                            "id": node.id.to_uuid_string(),
                            "ntype": node.ntype,
                            "summary": node.summary,
                        }));
                    }
                }
            }
            DiffEntry::Changed { before, after, .. } => {
                if nodes_changed.len() < limit {
                    if let Some(after_node) = node_from_bs(bs, after) {
                        let before_node = node_from_bs(bs, before);
                        let before_val = before_node.as_ref().map(|n| {
                            json!({
                                "id": n.id.to_uuid_string(),
                                "ntype": n.ntype,
                                "summary": n.summary,
                            })
                        });
                        let before_state = before_node.as_ref().map(|n| {
                            json!({ "ntype": n.ntype, "summary": n.summary })
                        });
                        node_deltas.push(json!({
                            "type": "changed",
                            "id": after_node.id.to_uuid_string(),
                            "ntype": after_node.ntype,
                            "summary": after_node.summary,
                            "before": before_state,
                        }));
                        nodes_changed.push(json!({
                            "id": after_node.id.to_uuid_string(),
                            "before": before_val,
                            "after": {
                                "id": after_node.id.to_uuid_string(),
                                "ntype": after_node.ntype,
                                "summary": after_node.summary,
                            },
                        }));
                    }
                }
            }
        }
    }

    // Edge buckets: nested (backward compat) + flat CLI-parity array.
    let mut edges_added: Vec<Value> = Vec::new();
    let mut edges_removed: Vec<Value> = Vec::new();
    let mut edge_deltas: Vec<Value> = Vec::new();

    for entry in &edge_changes {
        match entry {
            DiffEntry::Added { value, .. } => {
                if edges_added.len() < limit {
                    if let Some(edge) = edge_from_bs(bs, value) {
                        edge_deltas.push(json!({
                            "type": "added",
                            "label": edge.etype,
                            "src": edge.src.to_uuid_string(),
                            "dst": edge.dst.to_uuid_string(),
                        }));
                        edges_added.push(json!({
                            "id": edge.id.to_uuid_string(),
                            "etype": edge.etype,
                            "src": edge.src.to_uuid_string(),
                            "dst": edge.dst.to_uuid_string(),
                        }));
                    }
                }
            }
            DiffEntry::Removed { value, .. } => {
                if edges_removed.len() < limit {
                    if let Some(edge) = edge_from_bs(bs, value) {
                        edge_deltas.push(json!({
                            "type": "removed",
                            "label": edge.etype,
                            "src": edge.src.to_uuid_string(),
                            "dst": edge.dst.to_uuid_string(),
                        }));
                        edges_removed.push(json!({
                            "id": edge.id.to_uuid_string(),
                            "etype": edge.etype,
                            "src": edge.src.to_uuid_string(),
                            "dst": edge.dst.to_uuid_string(),
                        }));
                    }
                }
            }
            DiffEntry::Changed { before, after, .. } => {
                if let Some(edge_after) = edge_from_bs(bs, after) {
                    let before_state = edge_from_bs(bs, before).map(|e| {
                        json!({
                            "label": e.etype,
                            "src": e.src.to_uuid_string(),
                            "dst": e.dst.to_uuid_string(),
                        })
                    });
                    edge_deltas.push(json!({
                        "type": "changed",
                        "label": edge_after.etype,
                        "src": edge_after.src.to_uuid_string(),
                        "dst": edge_after.dst.to_uuid_string(),
                        "before": before_state,
                    }));
                }
            }
        }
    }

    Ok(json!({
        "schema": "mnem.v1.diff",
        "op_a": from_op_cid.to_string(),
        "op_b": to_op_cid.to_string(),
        "from": from_cid.to_string(),
        "to": to_cid.to_string(),
        "commit_a": from_cid.to_string(),
        "commit_b": to_cid.to_string(),
        "ref_deltas": {
            "added": refs_added,
            "removed": refs_removed,
            "changed": refs_changed,
        },
        "nodes": {
            "added": nodes_added,
            "removed": nodes_removed,
            "changed": nodes_changed,
        },
        "edges": {
            "added": edges_added,
            "removed": edges_removed,
            "changed": [],
        },
        "node_deltas": node_deltas,
        "edge_deltas": edge_deltas,
    }))
}

/// `POST /v1/diff` - structural diff between two commits (or ops).
///
/// Body: `{"from": "<cid>", "to": "<cid>"}`
/// Query: `?limit=N` (default 500, max 2000) - cap per added/removed/changed bucket.
///
/// Both `from` and `to` accept either a commit CID or an op CID (the op's head
/// commit is resolved automatically).
///
/// Response schema: `mnem.v1.diff`
pub(crate) async fn post_diff(
    State(s): State<AppState>,
    Query(params): Query<DiffQueryParams>,
    Json(body): Json<DiffBody>,
) -> Result<Json<Value>, Error> {
    let limit = params
        .limit
        .unwrap_or(DIFF_DEFAULT_LIMIT)
        .min(DIFF_MAX_LIMIT);
    if limit == 0 {
        return Err(Error::bad_request("limit must be >= 1"));
    }
    let repo = s.repo.lock().map_err(|_| Error::locked())?;
    let from_resolved = reindex_resolve_commitish(&repo, &body.from)?;
    let to_resolved = reindex_resolve_commitish(&repo, &body.to)?;
    let bs = repo.blockstore().clone();
    drop(repo);
    Ok(Json(compute_diff(
        bs.as_ref(),
        &from_resolved.to_string(),
        &to_resolved.to_string(),
        limit,
    )?))
}

// ---------- GET /v1/diff ----------

/// `GET /v1/diff?from=<ref>&to=<ref>[&limit=<n>]`
///
/// Idempotent read-only variant of `POST /v1/diff`. Accepts the same `from`
/// and `to` commitish arguments as the POST body, but via query parameters.
/// Useful for `curl` one-liners and browser-based tooling.
///
/// Returns the same `{"schema":"mnem.v1.diff", ...}` envelope.
pub(crate) async fn get_diff(
    State(s): State<AppState>,
    Query(params): Query<GetDiffParams>,
) -> Result<Json<Value>, Error> {
    let limit = params
        .limit
        .unwrap_or(DIFF_DEFAULT_LIMIT)
        .min(DIFF_MAX_LIMIT);
    if limit == 0 {
        return Err(Error::bad_request("limit must be >= 1"));
    }
    if params.from.trim().is_empty() {
        return Err(Error::bad_request("`from` query parameter is required"));
    }
    if params.to.trim().is_empty() {
        return Err(Error::bad_request("`to` query parameter is required"));
    }
    let repo = s.repo.lock().map_err(|_| Error::locked())?;
    let from_resolved = reindex_resolve_commitish(&repo, &params.from)?;
    let to_resolved = reindex_resolve_commitish(&repo, &params.to)?;
    let bs = repo.blockstore().clone();
    drop(repo);
    Ok(Json(compute_diff(
        bs.as_ref(),
        &from_resolved.to_string(),
        &to_resolved.to_string(),
        limit,
    )?))
}

// ---------- GET /v1/blocks/{cid} ----------

/// Output format for `GET /v1/blocks/{cid}`.
#[derive(Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub(crate) enum BlockFormat {
    /// Decode CBOR and return as JSON (default).
    #[default]
    Json,
    /// Return raw bytes hex-encoded in a JSON wrapper.
    Raw,
    /// Return raw CBOR bytes with `Content-Type: application/cbor`.
    Cbor,
}

/// Query parameters for `GET /v1/blocks/{cid}`.
#[derive(Deserialize, Default)]
pub(crate) struct BlockParams {
    /// Output format: `json` (default), `raw`, or `cbor`.
    #[serde(default)]
    pub format: BlockFormat,
}

/// `GET /v1/blocks/{cid}` - fetch a single raw block by its CID.
///
/// The `{cid}` path segment must be a valid multibase-encoded CID string
/// (e.g. base32 upper `BAFY...`). No special characters, so no wildcard
/// capture is required.
///
/// Query params:
/// - `?format=json` (default) - decode CBOR, return as JSON
/// - `?format=raw` - return raw bytes as hex in a JSON wrapper
/// - `?format=cbor` - return raw CBOR with `Content-Type: application/cbor`
///
/// Errors:
/// - 400 if the CID string cannot be parsed
/// - 404 if the block is not in the store
/// - 422 if `?format=json` is requested but the CBOR block cannot be decoded
pub(crate) async fn get_block(
    State(s): State<AppState>,
    Path(cid_str): Path<String>,
    Query(params): Query<BlockParams>,
) -> Result<impl IntoResponse, Error> {
    // Parse the CID string.
    let cid = mnem_core::id::Cid::parse_str(&cid_str)
        .map_err(|e| Error::bad_request(format!("invalid CID `{cid_str}`: {e}")))?;

    // Acquire repo lock, clone the blockstore, then drop the lock
    // before any potentially slow I/O.
    let repo = s.repo.lock().map_err(|_| Error::locked())?;
    let bs = repo.blockstore().clone();
    drop(repo);

    // Fetch the raw block bytes.
    let data = bs
        .get(&cid)
        .map_err(|e| Error::internal(format!("blockstore read: {e}")))?
        .ok_or_else(|| Error::not_found(format!("block `{cid_str}` not found in store")))?;

    match params.format {
        BlockFormat::Cbor => {
            // Return raw CBOR bytes directly.
            Ok(([(CONTENT_TYPE, "application/cbor")], data.to_vec()).into_response())
        }
        BlockFormat::Raw => {
            // Hex-encode the raw bytes and wrap in a JSON envelope.
            let hex: String = data.iter().map(|b| format!("{b:02x}")).collect();
            Ok(Json(json!({
                "schema": "mnem.v1.block",
                "cid": cid.to_string(),
                "format": "raw",
                "hex": hex,
            }))
            .into_response())
        }
        BlockFormat::Json => {
            // Attempt to decode the CBOR as generic IPLD and convert to JSON.
            match from_canonical_bytes::<Ipld>(&data) {
                Ok(ipld) => Ok(Json(json!({
                    "schema": "mnem.v1.block",
                    "cid": cid.to_string(),
                    "format": "json",
                    "data": ipld_to_json(&ipld),
                }))
                .into_response()),
                Err(e) => Err(Error::unprocessable(format!(
                    "block `{cid_str}` cannot be decoded as JSON (CBOR decode error): {e}"
                ))),
            }
        }
    }
}

/// Load and decode a [`Node`] from the blockstore by its value CID.
/// Returns `None` on any decode / store error (missing block, wrong codec).
fn node_from_bs(bs: &dyn mnem_core::store::Blockstore, cid: &mnem_core::id::Cid) -> Option<Node> {
    let bytes = bs.get(cid).ok()??;
    from_canonical_bytes::<Node>(&bytes).ok()
}

/// Load and decode an [`Edge`] from the blockstore by its value CID.
/// Returns `None` on any decode / store error.
fn edge_from_bs(
    bs: &dyn mnem_core::store::Blockstore,
    cid: &mnem_core::id::Cid,
) -> Option<mnem_core::objects::Edge> {
    let bytes = bs.get(cid).ok()??;
    from_canonical_bytes::<mnem_core::objects::Edge>(&bytes).ok()
}

// ---------- POST /v1/merge ----------

/// `strategy` field on `POST /v1/merge` body. Defaults to `manual`.
#[derive(Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub(crate) enum MergeStrategyParam {
    #[default]
    Manual,
    Ours,
    Theirs,
}

/// Request body for `POST /v1/merge`.
#[derive(Deserialize)]
pub(crate) struct MergeBody {
    /// Left (current-branch) commit: raw CID, `"HEAD"`, branch name, or full ref.
    pub left: String,
    /// Right (incoming-branch) commit: raw CID, `"HEAD"`, branch name, or full ref.
    pub right: String,
    /// Conflict-resolution strategy. Defaults to `"manual"`.
    #[serde(default)]
    pub strategy: MergeStrategyParam,
    /// When `true`, compute and return the merge outcome without persisting any
    /// results. Mirrors the CLI `mnem merge --dry-run` flag. Note: the HTTP
    /// endpoint is inherently non-committed (HEAD and refs are never updated by
    /// this handler), so `dry_run: true` explicitly signals preview-only intent.
    #[serde(default)]
    pub dry_run: bool,
}

/// `POST /v1/merge` - 3-way merge two commits.
///
/// Body: `{"left": "<ref>", "right": "<ref>", "strategy": "manual"|"ours"|"theirs", "dry_run": false}`
///
/// `left` and `right` accept raw CIDs, `"HEAD"`, branch names, or full refs
/// (mirrors CLI which accepts any commitish). `strategy` defaults to `"manual"`.
/// `dry_run` defaults to `false`.
/// Mirrors `mnem merge --dry-run`: when `true`, the merge outcome is computed but
/// this handler never updates HEAD or refs regardless, so `dry_run` is a client hint.
///
/// Response (HTTP 200 for all outcomes):
/// - `{"status": "fast_forward", "commit": "<cid>", "dry_run": bool}`
/// - `{"status": "clean", "commit": "<cid>", "dry_run": bool}`
/// - `{"status": "conflicts", "conflicts": <MergeConflicts>, "dry_run": bool}`
pub(crate) async fn post_merge(
    _auth: RequireBearer,
    State(s): State<AppState>,
    Json(body): Json<MergeBody>,
) -> Result<Json<Value>, Error> {
    use mnem_core::repo::merge::{MergeOutcome, MergeStrategy, merge_three_way};
    use mnem_core::store::MemoryOpHeadsStore;

    let repo = s.repo.lock().map_err(|_| Error::locked())?;
    // Step 1: resolve symbolic refs (HEAD, branch names, full refs) to raw CIDs
    // while holding the read lock so the view is consistent. Raw CIDs and op
    // CIDs pass through unchanged here.
    let left_resolved = reindex_resolve_commitish(&repo, &body.left)?;
    let right_resolved = reindex_resolve_commitish(&repo, &body.right)?;
    let bs = repo.blockstore().clone();
    drop(repo); // release lock before slow merge walks

    // Step 2: load commits. resolve_cid_to_commit also handles op CIDs by
    // unwrapping the op→view→commit chain, giving us the actual commit CIDs.
    let (left_cid, _) = resolve_cid_to_commit(bs.as_ref(), &left_resolved.to_string())?;
    let (right_cid, _) = resolve_cid_to_commit(bs.as_ref(), &right_resolved.to_string())?;

    // Reject same-commit after full resolution: "main" and "HEAD" might resolve
    // to the same commit, which is almost certainly a caller mistake.
    if left_cid == right_cid {
        return Err(Error::bad_request(
            "left and right must resolve to different commits",
        ));
    }

    let strategy = match body.strategy {
        MergeStrategyParam::Manual => MergeStrategy::Manual,
        MergeStrategyParam::Ours => MergeStrategy::Ours,
        MergeStrategyParam::Theirs => MergeStrategy::Theirs,
    };

    // The _oph parameter is unused in merge_three_way; pass a dummy.
    let dummy_ohs: std::sync::Arc<dyn mnem_core::store::OpHeadsStore> =
        std::sync::Arc::new(MemoryOpHeadsStore::new());
    let outcome = merge_three_way(&bs, &dummy_ohs, left_cid, right_cid, strategy).map_err(|e| {
        // NoCommonAncestor means the caller supplied two unrelated
        // commits - that's a 400, not a server error.
        use mnem_core::error::RepoError;
        match &e {
            mnem_core::error::Error::Repo(RepoError::NoCommonAncestor) => Error::bad_request(
                "left and right commits share no common ancestor; \
                         cannot merge unrelated histories",
            ),
            _ => Error::internal(format!("merge failed: {e}")),
        }
    })?;

    let dry_run = body.dry_run;
    let response = match outcome {
        MergeOutcome::FastForward(cid) => json!({
            "schema": "mnem.v1.merge",
            "status": "fast_forward",
            "commit": cid.to_string(),
            "dry_run": dry_run,
        }),
        MergeOutcome::Clean(cid) => json!({
            "schema": "mnem.v1.merge",
            "status": "clean",
            "commit": cid.to_string(),
            "dry_run": dry_run,
        }),
        MergeOutcome::Conflicts(conflicts) => json!({
            "schema": "mnem.v1.merge",
            "status": "conflicts",
            "conflicts": conflicts,
            "dry_run": dry_run,
        }),
    };

    Ok(Json(response))
}

// ---------- GET /v1/schema ----------

/// `GET /v1/schema` - list all known labels (ntypes) and their indexed props.
///
/// Reads the `IndexSet` stored in the head commit. Returns an empty list
/// when the repo has no commits or the head commit carries no IndexSet.
///
/// Response schema: `mnem.v1.schema`
/// ```json
/// {"schema":"mnem.v1.schema","labels":[{"label":"Fact","indexed_props":["user","topic"]},...]}
/// ```
pub(crate) async fn get_schema(State(s): State<AppState>) -> Result<Json<Value>, Error> {
    let repo = s.repo.lock().map_err(|_| Error::locked())?;
    let commit: Option<mnem_core::objects::Commit> = repo.head_commit().cloned();
    let bs = repo.blockstore().clone();
    drop(repo);

    let Some(commit) = commit else {
        return Ok(Json(json!({"schema": "mnem.v1.schema", "labels": []})));
    };
    let Some(indexes_cid) = commit.indexes.as_ref() else {
        return Ok(Json(json!({"schema": "mnem.v1.schema", "labels": []})));
    };

    let bytes = bs
        .get(indexes_cid)
        .map_err(|e| Error::internal(format!("blockstore: {e}")))?
        .ok_or_else(|| Error::internal("IndexSet block missing from blockstore"))?;
    let set: mnem_core::objects::IndexSet = from_canonical_bytes(&bytes)
        .map_err(|e| Error::internal(format!("decode IndexSet: {e}")))?;

    let has_outgoing = set.outgoing.is_some();
    let has_incoming = set.incoming.is_some();

    let labels: Vec<Value> = set
        .nodes_by_label
        .keys()
        .map(|label| {
            let props: Vec<&String> = set
                .nodes_by_prop
                .get(label)
                .map(|m| m.keys().collect())
                .unwrap_or_default();
            json!({
                "label": label,
                "indexed_props": props,
                "has_outgoing_adj": has_outgoing,
                "has_incoming_adj": has_incoming,
            })
        })
        .collect();

    Ok(Json(json!({"schema": "mnem.v1.schema", "labels": labels})))
}

// ---------- GET /v1/refs ----------

/// `GET /v1/refs` - list ALL refs (branches, tags, and any other refs).
///
/// Unlike `/v1/branches` and `/v1/tags`, this endpoint exposes the raw ref
/// store without prefix filtering, making every ref visible. The `ref_type`
/// field is `"branch"` for `refs/heads/*`, `"tag"` for `refs/tags/*`, and
/// `"other"` for anything else.
///
/// Response schema: `mnem.v1.refs`
/// ```json
/// {"schema":"mnem.v1.refs","refs":[{"name":"refs/heads/main","short":"main","ref_type":"branch","head":"<cid>"},{"name":"refs/tags/v1","short":"v1","ref_type":"tag","target":"<cid>"},...]}
/// ```
/// Branches carry `"head"` for the tip CID; tags carry `"target"`.
/// This matches the type-specific `/v1/branches` and `/v1/tags` endpoints.
pub(crate) async fn get_refs(State(s): State<AppState>) -> Result<Json<Value>, Error> {
    let repo = s.repo.lock().map_err(|_| Error::locked())?;
    let view = repo.view();

    let refs: Vec<Value> = view
        .refs
        .iter()
        .map(|(name, target)| {
            let (ref_type, short) = if let Some(s) = name.strip_prefix(HEADS_PREFIX) {
                ("branch", s.to_string())
            } else if let Some(s) = name.strip_prefix(TAGS_PREFIX) {
                ("tag", s.to_string())
            } else {
                ("other", name.clone())
            };
            let (cid_str, conflicted) = match target {
                mnem_core::objects::RefTarget::Normal { target } => {
                    (target.to_string(), false)
                }
                mnem_core::objects::RefTarget::Conflicted { .. } => {
                    (String::new(), true)
                }
            };
            // Branches expose the CID as "head"; tags expose it as "target".
            // This mirrors the type-specific endpoint conventions.
            let mut obj = serde_json::Map::new();
            obj.insert("name".into(), json!(name));
            obj.insert("short".into(), json!(short));
            obj.insert("ref_type".into(), json!(ref_type));
            obj.insert(
                if ref_type == "tag" { "target" } else { "head" }.into(),
                json!(cid_str),
            );
            obj.insert("conflicted".into(), json!(conflicted));
            Value::Object(obj)
        })
        .collect();

    Ok(Json(json!({"schema": "mnem.v1.refs", "refs": refs})))
}

// ---------- POST /v1/refs/{*name} ----------

/// Request body for `POST /v1/refs/{*name}`.
#[derive(Deserialize)]
pub(crate) struct SetRefBody {
    /// Commit CID to point the ref at.
    pub target: String,
    /// Author recorded on the `update_ref` Operation.
    pub author: String,
    /// Optional CAS guard. When provided the update is rejected unless the ref
    /// currently points to this CID (`RepoError::Stale` → 409). Omit to
    /// create a ref that must not yet exist (`prev=None` = create-only;
    /// returns 409 if the ref already exists at any target).
    #[serde(default)]
    pub prev: Option<String>,
}

/// Return `true` when `name` is safe to use as a ref name.
///
/// Rejects patterns that git and mnem's object store treat as special
/// or that could traverse paths / inject log noise:
/// - empty or whitespace-only
/// - contains `..` (path traversal)
/// - contains `//` (double slash)
/// - contains `\` (Windows path sep leaking through)
/// - contains `@{` (git reflog shorthand)
/// - starts or ends with `.`
/// - ends with `.lock`
/// - the bare literal `HEAD` (refs **named** HEAD shadow the symbolic HEAD)
/// - any ASCII control character or space
fn is_valid_ref_name(name: &str) -> bool {
    let s = name.trim();
    if s.is_empty() {
        return false;
    }
    if s.eq_ignore_ascii_case("HEAD") {
        return false;
    }
    if s.contains("..") || s.contains("//") || s.contains('\\') || s.contains("@{") {
        return false;
    }
    if s.starts_with('.') || s.ends_with('.') || s.ends_with(".lock") {
        return false;
    }
    if s.starts_with('/') || s.ends_with('/') {
        return false;
    }
    // git-refname forbidden chars: ~, ^, :, ?, *, [
    if s.contains('~') || s.contains('^') || s.contains(':')
        || s.contains('?') || s.contains('*') || s.contains('[')
    {
        return false;
    }
    s.chars().all(|c| !c.is_ascii_control() && c != ' ')
}

/// `POST /v1/refs/{*name}` - create or update a ref by full name.
///
/// The `name` path parameter must be the full ref name, e.g.
/// `refs/heads/feature-x` or `refs/tags/v1.0.0`. Returns 400 when
/// `target` is not a known commit CID.
///
/// Response schema: `mnem.v1.ref-set`
/// ```json
/// {"schema":"mnem.v1.ref-set","name":"refs/heads/feature-x","head":"<cid>"}
/// ```
pub(crate) async fn post_ref(
    _auth: RequireBearer,
    State(s): State<AppState>,
    Path(name): Path<String>,
    Json(body): Json<SetRefBody>,
) -> Result<Json<Value>, Error> {
    if name.trim().is_empty() {
        return Err(Error::bad_request("ref name is required"));
    }
    if !is_valid_ref_name(&name) {
        return Err(Error::bad_request(
            "ref name contains invalid characters; must not contain \
             `..`, `//`, `\\`, `@{`, control chars, or be the literal `HEAD`",
        ));
    }
    validate_author(&body.author)?;

    let target_cid = mnem_core::id::Cid::parse_str(&body.target)
        .map_err(|e| Error::bad_request(format!("invalid target CID `{}`: {e}", body.target)))?;

    let prev_target: Option<mnem_core::objects::RefTarget> = match body.prev {
        Some(ref s) => {
            let c = mnem_core::id::Cid::parse_str(s)
                .map_err(|e| Error::bad_request(format!("invalid prev CID `{s}`: {e}")))?;
            Some(mnem_core::objects::RefTarget::normal(c))
        }
        None => None,
    };

    let target_str = target_cid.to_string();
    let new_target = mnem_core::objects::RefTarget::normal(target_cid);
    let author = body.author.trim().to_string();

    let mut guard = s.repo.lock().map_err(|_| Error::locked())?;
    let new_repo = guard
        .update_ref(
            &name,
            prev_target.as_ref(),
            Some(new_target),
            &author,
        )
        .map_err(|e| {
            use mnem_core::error::RepoError;
            match &e {
                mnem_core::error::Error::Repo(RepoError::Stale) => {
                    Error::conflict("ref is stale; fetch the latest target and retry")
                }
                _ => Error::internal(format!("update_ref failed: {e}")),
            }
        })?;
    let op_id = new_repo.op_id().to_string();
    *guard = new_repo;

    Ok(Json(json!({
        "schema": "mnem.v1.ref-set",
        "name": name,
        "head": target_str,
        "op_id": op_id,
    })))
}

// ---------- DELETE /v1/refs/{*name} ----------

/// `DELETE /v1/refs/{*name}` - delete a ref by its full name.
///
/// Accepts the same `author` + `message` query params as the branch/tag
/// delete endpoints.
///
/// Response schema: `mnem.v1.ref-delete`
/// ```json
/// {"schema":"mnem.v1.ref-delete","name":"refs/heads/feature-x","deleted":true}
/// ```
pub(crate) async fn delete_ref(
    _auth: RequireBearer,
    State(s): State<AppState>,
    Path(name): Path<String>,
    Query(q): Query<DeleteQuery>,
) -> Result<Json<Value>, Error> {
    validate_author(&q.author)?;

    let mut guard = s.repo.lock().map_err(|_| Error::locked())?;
    let view = guard.view();
    let current = view
        .refs
        .get(&name)
        .cloned()
        .ok_or_else(|| Error::not_found(format!("ref not found: {name}")))?;
    let _ = view;

    let new_repo = guard
        .update_ref(&name, Some(&current), None, &q.author)
        .map_err(|e| {
            use mnem_core::error::RepoError;
            match &e {
                mnem_core::error::Error::Repo(RepoError::Stale) => {
                    Error::conflict("ref is stale; fetch the latest target and retry")
                }
                _ => Error::internal(format!("delete ref failed: {e}")),
            }
        })?;
    let op_id = new_repo.op_id().to_string();
    *guard = new_repo;

    Ok(Json(json!({
        "schema": "mnem.v1.ref-delete",
        "name": name,
        "deleted": true,
        "op_id": op_id,
    })))
}

// ---------- GET /v1/query + POST /v1/query ----------

/// Shared query parameters for the property-filter query endpoint.
///
/// Implements `FromRequestParts` directly (not via `axum::extract::Query`)
/// so that repeated keys like `?with_outgoing=a&with_outgoing=b` work.
/// `serde_urlencoded` errors on duplicate struct fields for the derived
/// path; parsing as a flat list of pairs avoids that entirely.
#[derive(Default)]
pub(crate) struct QueryParams {
    pub label: Option<String>,
    pub r#where: Vec<String>,
    pub with_outgoing: Vec<String>,
    pub limit: Option<usize>,
}

impl<S: Send + Sync> axum::extract::FromRequestParts<S> for QueryParams {
    type Rejection = Error;
    fn from_request_parts(
        parts: &mut axum::http::request::Parts,
        _state: &S,
    ) -> impl std::future::Future<Output = Result<Self, Self::Rejection>> + Send {
        let raw = parts.uri.query().unwrap_or("").to_string();
        async move {
            let pairs: Vec<(String, String)> = serde_urlencoded::from_str(&raw)
                .map_err(|e| Error::bad_request(format!("invalid query string: {e}")))?;
            let mut p = QueryParams::default();
            for (k, v) in pairs {
                match k.as_str() {
                    "label" => p.label = Some(v),
                    "where" => p.r#where.push(v),
                    "with_outgoing" => p.with_outgoing.push(v),
                    "limit" => p.limit = v.parse().ok(),
                    _ => {}
                }
            }
            Ok(p)
        }
    }
}

/// Request body for `POST /v1/query` (same fields as GET query params).
#[derive(Deserialize, Default)]
pub(crate) struct QueryBody {
    #[serde(default)]
    pub label: Option<String>,
    #[serde(default)]
    pub r#where: Vec<String>,
    #[serde(default)]
    pub with_outgoing: Vec<String>,
    #[serde(default)]
    pub limit: Option<usize>,
}

fn run_query(
    repo: &mnem_core::repo::ReadonlyRepo,
    label: Option<&str>,
    wheres: &[String],
    with_outgoing: &[String],
    limit: usize,
) -> Result<Vec<Value>, Error> {
    // The query engine returns Uninitialized on a repo with no head commit;
    // surface that as an empty result set instead of a 500.
    if repo.head_commit().is_none() {
        return Ok(vec![]);
    }

    let mut q = repo.query();

    if let Some(lbl) = label {
        q = q.label(lbl);
    }

    for kv in wheres {
        let (k, v) = kv
            .split_once('=')
            .ok_or_else(|| Error::bad_request(format!("where clause must be key=value, got: {kv}")))?;
        if k.is_empty() {
            return Err(Error::bad_request(format!(
                "where clause key must not be empty in `{kv}`"
            )));
        }
        q = q.where_prop(k, PropPredicate::Eq(ipld_core::ipld::Ipld::String(v.to_string())));
    }

    for etype in with_outgoing {
        q = q.with_outgoing(etype.as_str());
    }

    let hits = q
        .limit(limit)
        .execute()
        .map_err(|e| Error::internal(format!("query failed: {e}")))?;

    let results = hits
        .into_iter()
        .map(|hit| {
            let node = &hit.node;
            let edges: Vec<Value> = hit
                .edges
                .iter()
                .map(|e| {
                    json!({
                        "id": e.id.to_string(),
                        "src": e.src.to_string(),
                        "dst": e.dst.to_string(),
                        "etype": e.etype,
                    })
                })
                .collect();
            json!({
                "id": node.id.to_string(),
                "ntype": node.ntype,
                "summary": node.summary,
                "context_sentence": node.context_sentence,
                "props": node.props,
                "edges": edges,
                "edges_truncated": hit.edges_truncated,
            })
        })
        .collect();

    Ok(results)
}

/// `GET /v1/query` - property-filter query.
///
/// Query params: `label`, `where` (repeatable, `key=value`),
/// `with_outgoing` (repeatable edge label), `limit` (usize, default 10).
///
/// Response schema: `mnem.v1.query`
pub(crate) async fn get_query(
    State(s): State<AppState>,
    params: QueryParams,
) -> Result<Json<Value>, Error> {
    clamp_or_reject("limit", params.limit, MAX_QUERY_LIMIT)?;
    let limit = params.limit.unwrap_or(10);
    let repo = s.repo.lock().map_err(|_| Error::locked())?;
    let results = run_query(
        &repo,
        params.label.as_deref(),
        &params.r#where,
        &params.with_outgoing,
        limit,
    )?;
    Ok(Json(json!({"schema": "mnem.v1.query", "results": results})))
}

/// `POST /v1/query` - property-filter query (body variant).
///
/// Identical semantics to `GET /v1/query` but accepts a JSON body.
/// Useful when filter values contain characters that are awkward in
/// query strings (e.g. `=`, `&`).
///
/// Response schema: `mnem.v1.query`
pub(crate) async fn post_query(
    State(s): State<AppState>,
    Json(body): Json<QueryBody>,
) -> Result<Json<Value>, Error> {
    clamp_or_reject("limit", body.limit, MAX_QUERY_LIMIT)?;
    let limit = body.limit.unwrap_or(10);
    let repo = s.repo.lock().map_err(|_| Error::locked())?;
    let results = run_query(
        &repo,
        body.label.as_deref(),
        &body.r#where,
        &body.with_outgoing,
        limit,
    )?;
    Ok(Json(json!({"schema": "mnem.v1.query", "results": results})))
}

// ---------- GET /v1/nodes/{id}/edges ----------

/// Query params for `GET /v1/nodes/{id}/edges`.
///
/// `etype` may be repeated (`?etype=knows&etype=lives_in`) to filter to
/// multiple edge types at once, matching CLI `--edge-label` semantics.
#[derive(Default)]
pub(crate) struct NodeEdgesParams {
    /// `"out"` (default), `"in"`, or `"both"`.
    pub direction: Option<String>,
    /// Filter to these etypes (empty = no filter).
    pub etypes: Vec<String>,
    /// Maximum edges per direction. Defaults to 100.
    pub limit: Option<usize>,
}

impl<S: Send + Sync> axum::extract::FromRequestParts<S> for NodeEdgesParams {
    type Rejection = Error;
    fn from_request_parts(
        parts: &mut axum::http::request::Parts,
        _state: &S,
    ) -> impl std::future::Future<Output = Result<Self, Self::Rejection>> + Send {
        let raw = parts.uri.query().unwrap_or("").to_string();
        async move {
            let pairs: Vec<(String, String)> = serde_urlencoded::from_str(&raw)
                .map_err(|e| Error::bad_request(format!("invalid query string: {e}")))?;
            let mut p = NodeEdgesParams::default();
            for (k, v) in pairs {
                match k.as_str() {
                    "direction" => p.direction = Some(v),
                    "etype" => {
                        if v.len() > MAX_ETYPE_LEN {
                            return Err(Error::bad_request(format!(
                                "etype exceeds maximum length of {MAX_ETYPE_LEN} bytes"
                            )));
                        }
                        p.etypes.push(v);
                    }
                    "limit" => p.limit = v.parse().ok(),
                    _ => {}
                }
            }
            Ok(p)
        }
    }
}

/// `GET /v1/nodes/{id}/edges` - list edges for a node.
///
/// Returns outgoing and/or incoming edges depending on the `direction`
/// query param (`out` | `in` | `both`, default `out`).
///
/// Response schema: `mnem.v1.node-edges`
/// ```json
/// {"schema":"mnem.v1.node-edges","id":"<uuid>","outgoing":[...],"incoming":[...]}
/// ```
pub(crate) async fn get_node_edges(
    State(s): State<AppState>,
    Path(id_str): Path<String>,
    params: NodeEdgesParams,
) -> Result<Json<Value>, Error> {
    let node_id = mnem_core::id::NodeId::parse_uuid(&id_str)
        .map_err(|_| Error::bad_request(format!("invalid node id: {id_str}")))?;

    let direction = params.direction.as_deref().unwrap_or("out");
    if !matches!(direction, "out" | "in" | "both") {
        return Err(Error::bad_request(format!(
            "invalid direction {direction:?}; expected \"out\", \"in\", or \"both\""
        )));
    }
    clamp_or_reject("limit", params.limit, MAX_EDGE_LIMIT)?;
    let limit = params.limit.unwrap_or(100);
    // outgoing_edges / incoming_edges_capped take Option<&[&str]>.
    let etype_strs: Vec<&str> = params.etypes.iter().map(String::as_str).collect();
    let etype_filter: Option<&[&str]> = if etype_strs.is_empty() {
        None
    } else {
        Some(&etype_strs)
    };

    let repo = s.repo.lock().map_err(|_| Error::locked())?;

    // Fetch the node to get ntype; verify it exists.
    let node = repo
        .lookup_node(&node_id)
        .map_err(|e| Error::internal(format!("lookup_node: {e}")))?
        .ok_or_else(|| Error::not_found(format!("no node with id={id_str}")))?;

    let serialize_edge = |e: &mnem_core::objects::Edge| -> Value {
        let props_map: serde_json::Map<String, Value> = e
            .props
            .iter()
            .map(|(k, v)| (k.clone(), ipld_to_json(v)))
            .collect();
        json!({
            "id": e.id.to_string(),
            "src": e.src.to_string(),
            "dst": e.dst.to_string(),
            "etype": e.etype,
            "props": props_map,
        })
    };

    // Collect outgoing edges once; build both the HTTP-rich format and the
    // CLI-compatible flat format from the same vec.
    let outgoing_raw: Vec<mnem_core::objects::Edge> = if direction == "out" || direction == "both" {
        repo.outgoing_edges(&node_id, etype_filter)
            .map_err(|e| Error::internal(format!("outgoing_edges: {e}")))?
            .into_iter()
            .take(limit)
            .collect()
    } else {
        vec![]
    };

    let incoming: Vec<Value> = if direction == "in" || direction == "both" {
        let edges = repo
            .incoming_edges_capped(&node_id, etype_filter, limit)
            .map_err(|e| Error::internal(format!("incoming_edges: {e}")))?;
        edges.iter().map(serialize_edge).collect()
    } else {
        vec![]
    };

    // CLI-compatible flat edge list: [{etype, dst}] using UUID strings.
    // Mirrors the output of `mnem traverse --json` (outgoing only).
    let edges: Vec<Value> = outgoing_raw
        .iter()
        .map(|e| json!({"etype": e.etype, "dst": e.dst.to_uuid_string()}))
        .collect();
    let outgoing: Vec<Value> = outgoing_raw.iter().map(serialize_edge).collect();

    Ok(Json(json!({
        "schema": "mnem.v1.node-edges",
        // CLI-compatible node wrapper: mirrors `mnem traverse --json`.
        "node": {
            "id": node.id.to_uuid_string(),
            "ntype": node.ntype,
        },
        "id": id_str,
        // CLI-compatible flat outgoing edges: [{etype, dst}].
        "edges": edges,
        "outgoing": outgoing,
        "incoming": incoming,
    })))
}

// ---------- GET /v1/config ----------

/// `GET /v1/config` - read all config key/value pairs.
///
/// Reads `<data_dir>/config.toml` as a flat key=value listing using
/// dotted-key notation (e.g. `user.name`, `embed.provider`). Returns
/// an empty object when the file does not exist.
///
/// Response schema: `mnem.v1.config`
/// ```json
/// {"schema":"mnem.v1.config","config":{"user.name":"alice","embed.provider":"openai"}}
/// ```
pub(crate) async fn get_config(State(s): State<AppState>) -> Result<Json<Value>, Error> {
    let config_path = s.data_dir.join("config.toml");
    let flat = read_config_flat(&config_path)?;
    Ok(Json(json!({"schema": "mnem.v1.config", "config": flat})))
}

// ---------- PUT /v1/config/{*key} ----------

/// Request body for `PUT /v1/config/{*key}`.
#[derive(Deserialize)]
pub(crate) struct SetConfigBody {
    /// String value to store.
    pub value: String,
}

/// `PUT /v1/config/{*key}` - set a config key.
///
/// `key` is a dotted path such as `user.name` or `embed.provider`. The
/// value is stored as a TOML string. Creates `config.toml` if absent.
///
/// Response schema: `mnem.v1.config-set`
/// ```json
/// {"schema":"mnem.v1.config-set","key":"user.name","value":"alice"}
/// ```
pub(crate) async fn put_config(
    _auth: RequireBearer,
    State(s): State<AppState>,
    Path(key): Path<String>,
    Json(body): Json<SetConfigBody>,
) -> Result<Json<Value>, Error> {
    if key.trim().is_empty() {
        return Err(Error::bad_request("config key is required"));
    }
    let config_path = s.data_dir.join("config.toml");
    let mut table = load_config_toml(&config_path)?;
    set_dotted_key(&mut table, &key, body.value.clone())?;
    save_config_toml(&config_path, &table)?;
    Ok(Json(json!({
        "schema": "mnem.v1.config-set",
        "key": key,
        "value": body.value,
    })))
}

// ---------- DELETE /v1/config/{*key} ----------

/// `DELETE /v1/config/{*key}` - unset a config key.
///
/// Removes the key from `config.toml`. Returns 404 when the key does
/// not exist. Writes back the modified TOML.
///
/// Response schema: `mnem.v1.config-deleted`
/// ```json
/// {"schema":"mnem.v1.config-deleted","key":"user.name","deleted":true}
/// ```
pub(crate) async fn delete_config(
    _auth: RequireBearer,
    State(s): State<AppState>,
    Path(key): Path<String>,
) -> Result<Json<Value>, Error> {
    if key.trim().is_empty() {
        return Err(Error::bad_request("config key is required"));
    }
    let config_path = s.data_dir.join("config.toml");
    let mut table = load_config_toml(&config_path)?;
    let removed = remove_dotted_key(&mut table, &key);
    if !removed {
        return Err(Error::not_found(format!("config key not found: {key}")));
    }
    save_config_toml(&config_path, &table)?;
    Ok(Json(json!({
        "schema": "mnem.v1.config-deleted",
        "key": key,
        "deleted": true,
    })))
}

// ---------- GET/POST /v1/remotes, GET/DELETE /v1/remotes/{name} ----------

#[derive(serde::Deserialize)]
pub(crate) struct AddRemoteBody {
    name: String,
    url: String,
    token_env: Option<String>,
}

fn validate_remote_name(name: &str) -> Result<(), Error> {
    if name.is_empty() {
        return Err(Error::bad_request("remote name must not be empty"));
    }
    if name.contains('.') || name.contains('[') || name.contains(']') {
        return Err(Error::bad_request(
            "remote name must not contain '.', '[', or ']'",
        ));
    }
    Ok(())
}

fn load_remote_section(
    config_path: &std::path::Path,
) -> Result<mnem_transport::remote::RemoteSection, Error> {
    if !config_path.exists() {
        return Ok(mnem_transport::remote::RemoteSection::default());
    }
    let text = std::fs::read_to_string(config_path)
        .map_err(|e| Error::internal(format!("read config.toml: {e}")))?;
    mnem_transport::remote::parse_config(&text)
        .map_err(|e| Error::internal(format!("parse config.toml remote section: {e}")))
}

fn save_remote_section(
    config_path: &std::path::Path,
    section: &mnem_transport::remote::RemoteSection,
) -> Result<(), Error> {
    let mut root: toml::Value = if config_path.exists() {
        let text = std::fs::read_to_string(config_path)
            .map_err(|e| Error::internal(format!("read config.toml: {e}")))?;
        toml::from_str(&text)
            .map_err(|e| Error::internal(format!("parse config.toml: {e}")))?
    } else {
        toml::Value::Table(toml::map::Map::new())
    };
    let table = root
        .as_table_mut()
        .ok_or_else(|| Error::internal("config.toml root is not a table"))?;
    table.remove("remote");
    if !section.remote.is_empty() {
        let remote_text = mnem_transport::remote::serialize_config(section)
            .map_err(|e| Error::internal(format!("serialize remote section: {e}")))?;
        let remote_root: toml::Value = toml::from_str(&remote_text)
            .map_err(|e| Error::internal(format!("re-parse remote section: {e}")))?;
        if let Some(new_remote) = remote_root.get("remote").cloned() {
            table.insert("remote".into(), new_remote);
        }
    }
    let text = toml::to_string_pretty(&root)
        .map_err(|e| Error::internal(format!("serialize config.toml: {e}")))?;
    if let Some(parent) = config_path.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| Error::internal(format!("create config dir: {e}")))?;
    }
    std::fs::write(config_path, text)
        .map_err(|e| Error::internal(format!("write config.toml: {e}")))
}

/// `GET /v1/remotes` - list all configured remotes.
pub(crate) async fn get_remotes(State(s): State<AppState>) -> Result<Json<Value>, Error> {
    let config_path = s.data_dir.join("config.toml");
    let section = load_remote_section(&config_path)?;
    let remotes: Vec<Value> = section
        .remote
        .iter()
        .map(|(name, cfg)| {
            json!({
                "name": name,
                "url": cfg.url,
                "token_env": cfg.token_env,
                "capabilities": cfg.capabilities,
            })
        })
        .collect();
    Ok(Json(json!({
        "schema": "mnem.v1.remotes",
        "remotes": remotes,
    })))
}

/// `POST /v1/remotes` - add a new remote entry.
pub(crate) async fn post_remote(
    _auth: RequireBearer,
    State(s): State<AppState>,
    Json(body): Json<AddRemoteBody>,
) -> Result<Json<Value>, Error> {
    validate_remote_name(&body.name)?;
    if body.url.starts_with("file://") || body.url.starts_with("file:/") {
        return Err(Error::bad_request(
            "file:// URLs are not allowed as remote URLs",
        ));
    }
    if body.url.trim().is_empty() {
        return Err(Error::bad_request("remote URL must not be empty"));
    }
    let config_path = s.data_dir.join("config.toml");
    let mut section = load_remote_section(&config_path)?;
    if section.remote.contains_key(&body.name) {
        return Err(Error::conflict(format!(
            "remote '{}' already exists",
            body.name
        )));
    }
    section.remote.insert(
        body.name.clone(),
        mnem_transport::remote::RemoteConfigFile {
            url: body.url.clone(),
            capabilities: None,
            token_env: body.token_env.clone(),
        },
    );
    save_remote_section(&config_path, &section)?;
    Ok(Json(json!({
        "schema": "mnem.v1.remote-create",
        "name": body.name,
        "url": body.url,
        "token_env": body.token_env,
        "created": true,
    })))
}

/// `GET /v1/remotes/{name}` - show one remote.
pub(crate) async fn get_remote(
    State(s): State<AppState>,
    Path(name): Path<String>,
) -> Result<Json<Value>, Error> {
    let config_path = s.data_dir.join("config.toml");
    let section = load_remote_section(&config_path)?;
    let cfg = section
        .remote
        .get(&name)
        .ok_or_else(|| Error::not_found(format!("remote '{name}' not found")))?;
    Ok(Json(json!({
        "schema": "mnem.v1.remote",
        "name": name,
        "url": cfg.url,
        "token_env": cfg.token_env,
        "capabilities": cfg.capabilities,
    })))
}

/// `DELETE /v1/remotes/{name}` - remove a remote entry.
pub(crate) async fn delete_remote(
    _auth: RequireBearer,
    State(s): State<AppState>,
    Path(name): Path<String>,
) -> Result<Json<Value>, Error> {
    let config_path = s.data_dir.join("config.toml");
    let mut section = load_remote_section(&config_path)?;
    if !section.remote.contains_key(&name) {
        return Err(Error::not_found(format!("remote '{name}' not found")));
    }
    section.remote.remove(&name);
    save_remote_section(&config_path, &section)?;
    Ok(Json(json!({
        "schema": "mnem.v1.remote-delete",
        "deleted": name,
    })))
}

// ---------- GET /v1/status ----------

/// `GET /v1/status` - current repo state: op-id, HEAD commit, merge
/// state, active branch, ref counts, remote tracking ref count, and
/// label count.
///
/// Response schema: `mnem.v1.status`
/// ```json
/// {"schema":"mnem.v1.status","op_id":"...","head":null,"merge_in_progress":false,...}
/// ```
pub(crate) async fn get_status(State(s): State<AppState>) -> Result<Json<Value>, Error> {
    let r = s.repo.lock().map_err(|_| Error::locked())?;

    let merge_in_progress = s.data_dir.join("MERGE_HEAD").exists();
    let merge_head_cid: Option<String> = if merge_in_progress {
        std::fs::read_to_string(s.data_dir.join("MERGE_HEAD"))
            .ok()
            .map(|s| s.trim().to_string())
    } else {
        None
    };

    let op_id = r.op_id().to_string();
    let view = r.view();
    let head = view.heads.first().map(ToString::to_string);
    let active_branch = view.active_branch().map(str::to_string);

    let commit_info = r.head_commit().map(|c| {
        let mut obj = serde_json::Map::new();
        obj.insert("change_id".into(), json!(c.change_id.to_uuid_string()));
        obj.insert("message".into(), json!(c.message));
        obj.insert("nodes".into(), json!(c.nodes.to_string()));
        obj.insert("edges".into(), json!(c.edges.to_string()));
        if let Some(idx_cid) = &c.indexes {
            obj.insert("indexes".into(), json!(idx_cid.to_string()));
        }
        Value::Object(obj)
    });

    // Load label count from the IndexSet, mirroring `mnem status` output.
    let labels: Option<usize> = r.head_commit().and_then(|c| {
        let idx_cid = c.indexes.as_ref()?;
        let bs = r.blockstore();
        let bytes = bs.get(idx_cid).ok()??;
        let idx: mnem_core::objects::IndexSet = from_canonical_bytes(&bytes).ok()?;
        Some(idx.nodes_by_label.len())
    });

    let refs = &view.refs;
    let (normal_count, conflicted_count) = refs.iter().fold((0usize, 0usize), |(n, c), (_, t)| {
        match t {
            mnem_core::objects::RefTarget::Normal { .. } => (n + 1, c),
            mnem_core::objects::RefTarget::Conflicted { .. } => (n, c + 1),
        }
    });
    let conflicted_refs: Vec<Value> = refs
        .iter()
        .filter_map(|(name, t)| match t {
            mnem_core::objects::RefTarget::Conflicted { adds, removes } => Some(json!({
                "name": name,
                "adds": adds.len(),
                "removes": removes.len(),
            })),
            _ => None,
        })
        .collect();

    let remote_tracking_total: usize = view
        .remote_refs
        .as_ref()
        .map(|rr| rr.values().map(std::collections::BTreeMap::len).sum())
        .unwrap_or(0);

    Ok(Json(json!({
        "schema": "mnem.v1.status",
        "op_id": op_id,
        "head": head,
        "active_branch": active_branch,
        "merge_in_progress": merge_in_progress,
        "merge_head": merge_head_cid,
        "commit": commit_info,
        "refs_total": refs.len(),
        "refs_normal": normal_count,
        "refs_conflicted": conflicted_count,
        "conflicted_refs": conflicted_refs,
        "remote_tracking_total": remote_tracking_total,
        "labels": labels,
    })))
}

// ---------- POST /v1/switch ----------

/// Request body for `POST /v1/switch`.
#[derive(Deserialize)]
pub(crate) struct SwitchBody {
    /// Short branch name (e.g. `main`) or fully-qualified ref
    /// (e.g. `refs/heads/main`). Required.
    pub name: String,
    /// Commit author recorded in the op-log entry. Required.
    pub author: String,
    /// When `true`, create the branch (pointing at HEAD) if it does not
    /// already exist, then switch to it atomically. Mirrors the behaviour
    /// of `git switch -c <branch>` / `git checkout -b <branch>`.
    /// Returns 409 when the repo has no commits yet (nothing to point at).
    #[serde(default)]
    pub create: bool,
}

/// `POST /v1/switch` - advance HEAD to the tip of a named branch.
///
/// Mirrors `mnem switch <branch>`. Refuses while a merge is in
/// progress (409). Returns 404 when the branch does not exist and 409
/// when the branch is in a conflicted state.
///
/// When `create: true` the branch is created first (if absent) then
/// switched to in one atomic-ish sequence, mirroring `git switch -c`.
///
/// Response schema: `mnem.v1.switch`
/// ```json
/// {"schema":"mnem.v1.switch","name":"main","head":"<cid>","already_on":false,"created":false}
/// ```
pub(crate) async fn post_switch(
    _auth: RequireBearer,
    State(s): State<AppState>,
    Json(body): Json<SwitchBody>,
) -> Result<Json<Value>, Error> {
    let name = body.name.trim();
    if name.is_empty() {
        return Err(Error::bad_request("name is required"));
    }
    if name.len() > 255 {
        return Err(Error::bad_request(
            "branch name exceeds maximum length of 255 bytes",
        ));
    }
    if !is_valid_ref_name(name) {
        return Err(Error::bad_request(format!(
            "invalid branch name `{name}`: may not contain spaces, control characters, \
             `~`, `^`, `:`, `?`, `*`, `[`, `\\`, `@{{`, `..`, `//`, \
             or start/end with `/` or `.`, or end with `.lock`"
        )));
    }
    let author = body.author.trim();
    if author.is_empty() {
        return Err(Error::bad_request("author is required"));
    }
    if author.len() > MAX_AUTHOR_LEN {
        return Err(Error::bad_request(format!(
            "author exceeds maximum length of {MAX_AUTHOR_LEN} bytes"
        )));
    }
    if s.data_dir.join("MERGE_HEAD").exists() {
        return Err(Error::conflict(
            "you are in the middle of a merge; run merge --continue or merge --abort first",
        ));
    }

    let full_ref = if name.starts_with(HEADS_PREFIX) {
        name.to_string()
    } else {
        format!("{HEADS_PREFIX}{name}")
    };

    let short_name = name
        .strip_prefix(HEADS_PREFIX)
        .unwrap_or(name)
        .to_string();

    let mut guard = s.repo.lock().map_err(|_| Error::locked())?;

    // When `create: true` and the branch doesn't yet exist, create it
    // atomically at the current HEAD commit before switching. This mirrors
    // `git switch -c <branch>` / `git checkout -b <branch>`.
    let created = if body.create && !guard.view().refs.contains_key(&full_ref) {
        let head_cid = guard
            .view()
            .heads
            .first()
            .cloned()
            .ok_or_else(|| Error::conflict("repository has no commits yet; cannot create branch"))?;
        let new_repo = guard
            .update_ref(
                &full_ref,
                None,
                Some(mnem_core::objects::RefTarget::normal(head_cid)),
                author,
            )
            .map_err(Error::from)?;
        *guard = new_repo;
        true
    } else {
        false
    };

    let branch_tip = match guard.view().refs.get(&full_ref) {
        Some(mnem_core::objects::RefTarget::Normal { target }) => target.clone(),
        Some(mnem_core::objects::RefTarget::Conflicted { .. }) => {
            return Err(Error::conflict(format!(
                "branch '{short_name}' is in a conflicted state; resolve before switching"
            )));
        }
        None => {
            return Err(Error::not_found(format!(
                "branch '{short_name}' not found"
            )));
        }
    };

    let current_head = guard.view().heads.first().cloned();
    let current_active_ref = guard.view().active_branch().map(str::to_string);
    if current_head.as_ref() == Some(&branch_tip)
        && current_active_ref.as_deref() == Some(full_ref.as_str())
        && !created
    {
        return Ok(Json(json!({
            "schema": "mnem.v1.switch",
            "op_id": guard.op_id().to_string(),
            "name": short_name,
            "head": branch_tip.to_string(),
            "already_on": true,
            "created": false,
        })));
    }

    let new_repo = guard
        .switch_branch(branch_tip.clone(), &full_ref, author)
        .map_err(|e| Error::internal(e.to_string()))?;
    let op_id = new_repo.op_id().to_string();
    *guard = new_repo;

    Ok(Json(json!({
        "schema": "mnem.v1.switch",
        "op_id": op_id,
        "name": short_name,
        "head": branch_tip.to_string(),
        "already_on": false,
        "created": created,
    })))
}

// ---------- GET /v1/nodes/{id}/blame ----------

/// Query params for `GET /v1/nodes/{id}/blame`.
#[derive(Deserialize)]
pub(crate) struct BlameQuery {
    /// Restrict to one edge-type label (e.g. `authored`, `cites`).
    pub etype: Option<String>,
    /// Walk operation ancestry and report the oldest ancestor commit
    /// that first introduced each edge.
    #[serde(default)]
    pub first_writer: bool,
    /// When true, return 404 if the requested node does not exist
    /// instead of an empty edges array. Mirrors CLI `--strict`.
    #[serde(default)]
    pub strict: bool,
}

/// `GET /v1/nodes/{id}/blame` - list all incoming edges for a node,
/// optionally filtered by edge-type.
///
/// Mirrors `mnem blame <node-id>`. Supports `?etype=<label>` and
/// `?first_writer=true` (BFS over operation ancestry).
///
/// Response schema: `mnem.v1.blame`
pub(crate) async fn get_node_blame(
    State(s): State<AppState>,
    Path(id_str): Path<String>,
    Query(q): Query<BlameQuery>,
) -> Result<Json<Value>, Error> {
    let node_id = NodeId::parse_uuid(&id_str)
        .map_err(|e| Error::bad_request(format!("invalid UUID: {e}")))?;

    let r = s.repo.lock().map_err(|_| Error::locked())?;

    if q.strict {
        r.lookup_node(&node_id)
            .map_err(|e| Error::internal(e.to_string()))?
            .ok_or_else(|| Error::not_found(format!("no node with id={id_str}")))?;
    }

    if let Some(ref et) = q.etype {
        if et.len() > MAX_ETYPE_LEN {
            return Err(Error::bad_request(format!(
                "etype exceeds maximum length of {MAX_ETYPE_LEN} bytes"
            )));
        }
    }
    let filter_etype = q.etype.as_deref();
    let filter_slice = filter_etype.map(|s| [s]);
    let filter_ref = filter_slice.as_ref().map(|arr| &arr[..]);

    let edges = r
        .incoming_edges(&node_id, filter_ref)
        .map_err(|e| Error::internal(e.to_string()))?;

    let head = r
        .view()
        .heads
        .first()
        .map(ToString::to_string)
        .unwrap_or_else(|| "<no-head>".into());

    if !q.first_writer {
        let items: Vec<Value> = edges
            .iter()
            .map(|e| {
                let props_map: serde_json::Map<String, Value> = e
                    .props
                    .iter()
                    .map(|(k, v)| (k.clone(), ipld_to_json(v)))
                    .collect();
                let relation = format!(
                    "{} -[{}]-> {}",
                    e.src.to_uuid_string(),
                    e.etype,
                    node_id.to_uuid_string()
                );
                json!({
                    "id": e.id.to_uuid_string(),
                    "etype": e.etype,
                    "src": e.src.to_uuid_string(),
                    "dst": node_id.to_uuid_string(),
                    "relation": relation,
                    "props": props_map,
                    "in_commit": head,
                })
            })
            .collect();
        return Ok(Json(json!({
            "schema": "mnem.v1.blame",
            "node_id": node_id.to_uuid_string(),
            "edges": items,
        })));
    }

    // BFS over operation ancestry to find first_writer for each edge.
    let bs = r.blockstore().clone();
    let ohs = r.op_heads_store().clone();
    let mut first_writer: std::collections::HashMap<EdgeId, String> =
        edges.iter().map(|e| (e.id, head.clone())).collect();
    let mut visited: std::collections::HashSet<Cid> = std::collections::HashSet::new();
    let mut queue: std::collections::VecDeque<Cid> =
        r.operation().parents.iter().cloned().collect();

    while let Some(ancestor_op_id) = queue.pop_front() {
        if !visited.insert(ancestor_op_id.clone()) {
            continue;
        }
        let ancestor =
            match mnem_core::repo::ReadonlyRepo::load_at(bs.clone(), ohs.clone(), ancestor_op_id.clone()) {
                Ok(a) => a,
                Err(e) => {
                    tracing::warn!(
                        ancestor_op = %ancestor_op_id,
                        error = %e,
                        "blame first_writer: skipped ancestor"
                    );
                    continue;
                }
            };
        let ancestor_commit = ancestor
            .view()
            .heads
            .first()
            .map(ToString::to_string)
            .unwrap_or_else(|| "<no-head>".into());
        let ancestor_edges = ancestor
            .incoming_edges(&node_id, filter_ref)
            .unwrap_or_default();
        let ancestor_ids: std::collections::HashSet<EdgeId> =
            ancestor_edges.iter().map(|e| e.id).collect();
        for (edge_id, fw) in &mut first_writer {
            if ancestor_ids.contains(edge_id) {
                *fw = ancestor_commit.clone();
            }
        }
        queue.extend(ancestor.operation().parents.iter().cloned());
    }

    let items: Vec<Value> = edges
        .iter()
        .map(|e| {
            let fw = first_writer
                .get(&e.id)
                .map(String::as_str)
                .unwrap_or("<unknown>");
            let props_map: serde_json::Map<String, Value> = e
                .props
                .iter()
                .map(|(k, v)| (k.clone(), ipld_to_json(v)))
                .collect();
            let relation = format!(
                "{} -[{}]-> {}",
                e.src.to_uuid_string(),
                e.etype,
                node_id.to_uuid_string()
            );
            json!({
                "id": e.id.to_uuid_string(),
                "etype": e.etype,
                "src": e.src.to_uuid_string(),
                "dst": node_id.to_uuid_string(),
                "relation": relation,
                "props": props_map,
                "first_writer": fw,
            })
        })
        .collect();
    Ok(Json(json!({
        "schema": "mnem.v1.blame",
        "node_id": node_id.to_uuid_string(),
        "edges": items,
    })))
}

// ---------- POST /v1/embed ----------

/// Query params for `POST /v1/embed`.
#[derive(Deserialize)]
pub(crate) struct EmbedQuery {
    /// Re-embed nodes that already have a vector for the current model.
    #[serde(default)]
    pub force: bool,
    /// Restrict to one label (ntype).
    pub label: Option<String>,
    /// Count and return what would be embedded without calling the
    /// provider.
    #[serde(default)]
    pub dry_run: bool,
    /// Author recorded in the resulting operation. Defaults to
    /// `"mnem-http"` when omitted.
    pub author: Option<String>,
}

/// `POST /v1/embed` - backfill embeddings for nodes that have no
/// vector under the configured model.
///
/// Mirrors `mnem embed`. Requires an embedder to be configured via
/// `config.toml` (`[embed]` section). Returns 503 when no embedder is
/// configured. Supports `?force=true`, `?label=<ntype>`,
/// `?dry_run=true`.
///
/// Response schema: `mnem.v1.embed`
pub(crate) async fn post_embed(
    _auth: RequireBearer,
    State(s): State<AppState>,
    Query(q): Query<EmbedQuery>,
) -> Result<Json<Value>, Error> {
    let pc = s.embed_cfg.as_ref().ok_or_else(|| {
        Error::status(
            axum::http::StatusCode::SERVICE_UNAVAILABLE,
            "no embedder configured; set [embed] in config.toml first",
        )
    })?;
    let embedder = mnem_embed_providers::open(pc)
        .map_err(|e| Error::internal(format!("open embedder: {e}")))?;
    let model_fq = embedder.model().to_string();

    let r = s.repo.lock().map_err(|_| Error::locked())?;
    let Some(head) = r.head_commit() else {
        return Ok(Json(json!({
            "schema": "mnem.v1.embed",
            "model": model_fq,
            "embedded": 0,
            "skipped_already_embedded": 0,
            "skipped_no_text": 0,
            "dry_run": q.dry_run,
            "note": "repository has no commits yet",
        })));
    };

    let bs = r.blockstore().clone();
    let cursor = Cursor::new(&*bs, &head.nodes)
        .map_err(|e| Error::internal(format!("cursor: {e}")))?;

    let mut candidates: Vec<(Cid, Node)> = Vec::new();
    let mut skipped_already_embedded: usize = 0;
    let mut skipped_no_text: usize = 0;

    for entry in cursor {
        let (_k, node_cid) = entry.map_err(|e| Error::internal(e.to_string()))?;
        let bytes = bs
            .get(&node_cid)
            .map_err(|e| Error::internal(e.to_string()))?
            .ok_or_else(|| Error::internal(format!("node CID {node_cid} missing")))?;
        let node: Node =
            from_canonical_bytes(&bytes).map_err(|e| Error::internal(e.to_string()))?;

        if mnem_core::anchor::is_system_node(&node) {
            continue;
        }
        if let Some(lbl) = &q.label {
            if &node.ntype != lbl {
                continue;
            }
        }

        let already = if q.force {
            false
        } else {
            r.embedding_for(&node_cid, &model_fq)
                .map_err(|e| Error::internal(e.to_string()))?
                .is_some()
        };
        if already {
            skipped_already_embedded += 1;
            continue;
        }
        if embed_text_of(&node).is_some() {
            candidates.push((node_cid, node));
        } else {
            skipped_no_text += 1;
        }
    }

    let would_embed = candidates.len();
    if q.dry_run {
        return Ok(Json(json!({
            "schema": "mnem.v1.embed",
            "model": model_fq,
            "would_embed": would_embed,
            "skipped_already_embedded": skipped_already_embedded,
            "skipped_no_text": skipped_no_text,
            "dry_run": true,
        })));
    }

    if candidates.is_empty() {
        return Ok(Json(json!({
            "schema": "mnem.v1.embed",
            "model": model_fq,
            "embedded": 0,
            "skipped_already_embedded": skipped_already_embedded,
            "skipped_no_text": skipped_no_text,
            "dry_run": q.dry_run,
        })));
    }

    let total = candidates.len();
    let mut tx = r.start_transaction();
    drop(r); // release before blocking embed calls
    for (node_cid, node) in candidates {
        let text = embed_text_of(&node)
            .ok_or_else(|| Error::internal("node has no embeddable text".to_string()))?;
        let v = embedder
            .embed(&text)
            .map_err(|e| Error::internal(format!("embed: {e}")))?;
        let emb = mnem_embed_providers::to_embedding(&model_fq, &v);
        tx.set_embedding(node_cid, model_fq.clone(), emb)
            .map_err(|e| Error::internal(e.to_string()))?;
    }
    if let Some(a) = q.author.as_deref() {
        if a.trim().is_empty() {
            return Err(Error::bad_request("author must not be blank"));
        }
        if a.len() > MAX_AUTHOR_LEN {
            return Err(Error::bad_request(format!(
                "author exceeds maximum length of {MAX_AUTHOR_LEN} bytes"
            )));
        }
    }
    let author = q.author.as_deref().unwrap_or("mnem-http");
    let msg = format!("mnem embed: backfill {total} nodes with {model_fq}");
    let new_repo = tx
        .commit(author, &msg)
        .map_err(|e| Error::internal(e.to_string()))?;
    let new_op_id = new_repo.op_id().to_string();
    let mut guard = s.repo.lock().map_err(|_| Error::locked())?;
    *guard = new_repo;

    Ok(Json(json!({
        "schema": "mnem.v1.embed",
        "model": model_fq,
        "embedded": total,
        "skipped_already_embedded": skipped_already_embedded,
        "skipped_no_text": skipped_no_text,
        "dry_run": false,
        "op_id": new_op_id,
    })))
}

/// Extract the text to embed for a node: prefer non-empty `summary`,
/// fall back to first 4096 bytes of `content`.
fn embed_text_of(node: &Node) -> Option<String> {
    let summary = node.summary.as_deref().unwrap_or("").trim();
    if !summary.is_empty() {
        return Some(summary.to_string());
    }
    if let Some(content) = &node.content {
        let s = String::from_utf8_lossy(content);
        let trimmed = s.trim();
        if !trimmed.is_empty() {
            let cap = trimmed.floor_char_boundary(4096);
            return Some(trimmed[..cap].to_string());
        }
    }
    None
}

// ---------- POST /v1/gc ----------

/// Query params for `POST /v1/gc`.
#[derive(Deserialize)]
pub(crate) struct GcQuery {
    /// Actually delete unreachable blocks. Without this flag the
    /// handler is a safe dry-run that only reports counts.
    #[serde(default)]
    pub force: bool,
}

/// `POST /v1/gc` - garbage-collect unreachable blocks.
///
/// Mirrors `mnem gc`. Walks the full content-addressed DAG reachable
/// from all known refs (branches, tags, remote tracking refs, WC
/// pointer), then deletes every block not in that set.
///
/// Without `?force=true` this is a **dry-run**: it returns counts but
/// does not modify the store.
///
/// Response schema: `mnem.v1.gc`
pub(crate) async fn post_gc(
    _auth: RequireBearer,
    State(s): State<AppState>,
    Query(q): Query<GcQuery>,
) -> Result<Json<Value>, Error> {
    let r = s.repo.lock().map_err(|_| Error::locked())?;
    let bs = r.blockstore().clone();
    let view = r.view();

    // Collect roots: current op + every ref target.
    let mut roots: Vec<Cid> = vec![r.op_id().clone()];
    for ref_target in view.refs.values() {
        match ref_target {
            mnem_core::objects::RefTarget::Normal { target } => roots.push(target.clone()),
            mnem_core::objects::RefTarget::Conflicted { adds, .. } => {
                roots.extend(adds.iter().cloned());
            }
        }
    }
    if let Some(remote_map) = &view.remote_refs {
        for inner in remote_map.values() {
            for ref_target in inner.values() {
                match ref_target {
                    mnem_core::objects::RefTarget::Normal { target } => roots.push(target.clone()),
                    mnem_core::objects::RefTarget::Conflicted { adds, .. } => {
                        roots.extend(adds.iter().cloned());
                    }
                }
            }
        }
    }
    if let Some(wc) = &view.wc_commit {
        roots.push(wc.clone());
    }
    roots.sort();
    roots.dedup();

    // Walk reachable blocks.
    let mut reachable: std::collections::HashSet<Cid> = std::collections::HashSet::new();
    for root in &roots {
        for item in bs.iter_from_root(root) {
            match item {
                Ok((cid, _)) => {
                    reachable.insert(cid);
                }
                Err(e) => {
                    return Err(Error::internal(format!(
                        "gc: reachability walk failed: {e}; run fsck to diagnose"
                    )));
                }
            }
        }
    }
    let reachable_count = reachable.len();

    // Enumerate all blocks.
    let all = match bs.all_cids().map_err(|e| Error::internal(e.to_string()))? {
        Some(cids) => cids,
        None => {
            return Ok(Json(json!({
                "schema": "mnem.v1.gc",
                "reachable": reachable_count,
                "total": null,
                "unreachable": null,
                "deleted": 0,
                "errors": 0,
                "force": q.force,
                "note": "store does not support block enumeration",
            })));
        }
    };
    let total_count = all.len();
    let unreachable: Vec<Cid> = all
        .into_iter()
        .filter(|cid| !reachable.contains(cid))
        .collect();
    let unreachable_count = unreachable.len();

    if !q.force {
        return Ok(Json(json!({
            "schema": "mnem.v1.gc",
            "reachable": reachable_count,
            "total": total_count,
            "unreachable": unreachable_count,
            "deleted": 0,
            "errors": 0,
            "force": false,
        })));
    }

    let mut deleted: usize = 0;
    let mut errors: usize = 0;
    for cid in &unreachable {
        match bs.delete(cid) {
            Ok(()) => deleted += 1,
            Err(e) => {
                tracing::warn!(cid = %cid, error = %e, "gc: failed to delete block");
                errors += 1;
            }
        }
    }

    Ok(Json(json!({
        "schema": "mnem.v1.gc",
        "reachable": reachable_count,
        "total": total_count,
        "unreachable": unreachable_count,
        "deleted": deleted,
        "errors": errors,
        "force": true,
    })))
}

// ---------- GET /v1/fsck ----------

/// Hard cap on the number of ops walked when `limit` is not supplied.
const FSCK_DEFAULT_LIMIT: usize = 50_000;

/// Query parameters for `GET /v1/fsck`.
#[derive(Deserialize)]
pub(crate) struct FsckQuery {
    /// Maximum number of ops to walk backwards from HEAD.
    pub limit: Option<usize>,
}

/// One integrity error discovered during the walk.
#[derive(Serialize)]
struct FsckErrorEntry {
    op: String,
    kind: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    cid: Option<String>,
}

/// Verify that a CID exists in the blockstore and that its bytes hash to that
/// CID. Returns `Ok(())` on success or `Err(reason)` with a human-readable
/// description.
fn fsck_check_block(
    bs: &dyn mnem_core::store::Blockstore,
    cid: &Cid,
) -> Result<(), String> {
    let bytes = bs
        .get(cid)
        .map_err(|e| format!("store I/O error fetching {cid}: {e}"))?;
    let bytes = match bytes {
        Some(b) => b,
        None => return Err("missing".to_string()),
    };
    if let Some(computed) = recompute_cid(cid, &bytes) {
        if computed != *cid {
            return Err(format!(
                "CID mismatch: claimed {cid} but content hashes to {computed}"
            ));
        }
    }
    Ok(())
}

/// Recursively walk every block in the Prolly tree rooted at `root`, counting
/// blocks present and pushing errors for any missing interior block.
fn fsck_walk_prolly_tree(
    bs: &dyn mnem_core::store::Blockstore,
    root: &Cid,
    tree_name: &str,
    op_cid_str: &str,
    errors: &mut Vec<FsckErrorEntry>,
) -> usize {
    let mut stack: Vec<Cid> = vec![root.clone()];
    let mut blocks_ok: usize = 0;
    while let Some(cid) = stack.pop() {
        let chunk = match load_tree_chunk(bs, &cid) {
            Ok(c) => {
                blocks_ok += 1;
                c
            }
            Err(_) => {
                errors.push(FsckErrorEntry {
                    op: op_cid_str.to_owned(),
                    kind: format!("missing interior block {cid} in {tree_name} tree"),
                    cid: Some(cid.to_string()),
                });
                continue;
            }
        };
        if let TreeChunk::Internal(internal) = chunk {
            stack.extend(internal.children);
        }
    }
    blocks_ok
}

/// `GET /v1/fsck` — reachability-only integrity check.
///
/// Mirrors `mnem fsck`: walks all ops from HEAD and verifies every referenced
/// block is present and CID-correct.
pub(crate) async fn get_fsck(
    State(s): State<AppState>,
    Query(q): Query<FsckQuery>,
) -> Result<Json<Value>, Error> {
    let repo = s.repo.lock().map_err(|_| Error::locked())?;
    let bs = repo.blockstore().clone();
    let bs = bs.as_ref();
    let limit = q.limit.unwrap_or(FSCK_DEFAULT_LIMIT);
    let head_op_cid = repo.op_id().clone();

    let mut errors: Vec<FsckErrorEntry> = Vec::new();
    let mut ops_checked: usize = 0;
    let mut blocks_verified: usize = 0;
    let mut visited: std::collections::HashSet<Cid> = std::collections::HashSet::new();
    let mut cur = head_op_cid;

    loop {
        if ops_checked >= limit {
            break;
        }
        if !visited.insert(cur.clone()) {
            break;
        }
        let op_cid_str = cur.to_string();

        // Step 1: verify the op block itself.
        let op_bytes = match fsck_check_block(bs, &cur) {
            Ok(()) => {
                blocks_verified += 1;
                bs.get(&cur)
                    .map_err(|e| Error::internal(format!("store I/O: {e}")))?
                    .ok_or_else(|| Error::internal(format!("op block {} vanished after verification", cur)))?
            }
            Err(reason) => {
                errors.push(FsckErrorEntry {
                    op: op_cid_str.clone(),
                    kind: format!("op block {reason}"),
                    cid: Some(op_cid_str.clone()),
                });
                break;
            }
        };

        let op: Operation = match from_canonical_bytes(&op_bytes) {
            Ok(o) => o,
            Err(e) => {
                errors.push(FsckErrorEntry {
                    op: op_cid_str.clone(),
                    kind: format!("op block decode failed: {e}"),
                    cid: Some(op_cid_str.clone()),
                });
                break;
            }
        };
        ops_checked += 1;

        // Step 2: verify the view block.
        let view_cid = &op.view;
        let view_opt: Option<View> = match fsck_check_block(bs, view_cid) {
            Ok(()) => {
                blocks_verified += 1;
                let view_bytes = bs
                    .get(view_cid)
                    .map_err(|e| Error::internal(format!("store I/O: {e}")))?
                    .ok_or_else(|| Error::internal(format!("view block {} vanished after verification", view_cid)))?;
                match from_canonical_bytes::<View>(&view_bytes) {
                    Ok(v) => Some(v),
                    Err(e) => {
                        errors.push(FsckErrorEntry {
                            op: op_cid_str.clone(),
                            kind: format!("view block decode failed: {e}"),
                            cid: Some(view_cid.to_string()),
                        });
                        None
                    }
                }
            }
            Err(reason) => {
                errors.push(FsckErrorEntry {
                    op: op_cid_str.clone(),
                    kind: format!("view block {reason}"),
                    cid: Some(view_cid.to_string()),
                });
                None
            }
        };

        // Steps 3 & 4: verify each head commit and its Prolly tree roots.
        if let Some(view) = view_opt {
            for head_cid in &view.heads {
                let commit_opt: Option<Commit> = match fsck_check_block(bs, head_cid) {
                    Ok(()) => {
                        blocks_verified += 1;
                        let commit_bytes = bs
                            .get(head_cid)
                            .map_err(|e| Error::internal(format!("store I/O: {e}")))?
                            .ok_or_else(|| Error::internal(format!("commit block {} vanished after verification", head_cid)))?;
                        match from_canonical_bytes::<Commit>(&commit_bytes) {
                            Ok(c) => Some(c),
                            Err(e) => {
                                errors.push(FsckErrorEntry {
                                    op: op_cid_str.clone(),
                                    kind: format!("commit block decode failed: {e}"),
                                    cid: Some(head_cid.to_string()),
                                });
                                None
                            }
                        }
                    }
                    Err(reason) => {
                        errors.push(FsckErrorEntry {
                            op: op_cid_str.clone(),
                            kind: format!("commit block {reason}"),
                            cid: Some(head_cid.to_string()),
                        });
                        None
                    }
                };

                if let Some(commit) = commit_opt {
                    for (tree_name, tree_cid) in [
                        ("nodes", &commit.nodes),
                        ("edges", &commit.edges),
                        ("schema", &commit.schema),
                    ] {
                        let n =
                            fsck_walk_prolly_tree(bs, tree_cid, tree_name, &op_cid_str, &mut errors);
                        blocks_verified += n;
                    }

                    for (tree_name, maybe_cid) in [
                        ("embeddings", commit.embeddings.as_ref()),
                        ("sparse", commit.sparse.as_ref()),
                    ] {
                        if let Some(cid) = maybe_cid {
                            let n = fsck_walk_prolly_tree(
                                bs,
                                cid,
                                tree_name,
                                &op_cid_str,
                                &mut errors,
                            );
                            blocks_verified += n;
                        }
                    }

                    let optional_roots: &[(&str, Option<&Cid>)] = &[
                        ("indexes root", commit.indexes.as_ref()),
                        ("delta root", commit.delta.as_ref()),
                    ];
                    for (label, maybe_cid) in optional_roots {
                        if let Some(opt_cid) = maybe_cid {
                            match fsck_check_block(bs, opt_cid) {
                                Ok(()) => blocks_verified += 1,
                                Err(reason) => {
                                    errors.push(FsckErrorEntry {
                                        op: op_cid_str.clone(),
                                        kind: format!("{label} {reason}"),
                                        cid: Some(opt_cid.to_string()),
                                    });
                                }
                            }
                        }
                    }
                }
            }
        }

        match op.parents.first() {
            Some(parent_cid) => cur = parent_cid.clone(),
            None => break,
        }
    }

    let ok = errors.is_empty();
    Ok(Json(json!({
        "schema": "mnem.v1.fsck",
        "ops_checked": ops_checked,
        "blocks_verified": blocks_verified,
        "errors": errors,
        "ok": ok,
    })))
}

// ---------- GET /v1/nodes/{id}/embeddings ----------

/// Query parameters for `GET /v1/nodes/{id}/embeddings`.
#[derive(Deserialize)]
pub(crate) struct ListEmbeddingsQuery {
    // No parameters needed; reserved for future filtering.
}

/// List all embedding models stored for a node.
///
/// Mirrors `mnem embedding ls <node-uuid>` — returns the model strings
/// (e.g. `"openai:text-embedding-3-small"`) that have a vector stored
/// against this node's content-addressed CID.
///
/// Returns `200 OK` with schema `mnem.v1.node-embeddings` on success,
/// `404 Not Found` if the node does not exist.
pub(crate) async fn get_node_embeddings(
    State(s): State<AppState>,
    Path(id_str): Path<String>,
    Query(_q): Query<ListEmbeddingsQuery>,
) -> Result<Json<Value>, Error> {
    let id = NodeId::parse_uuid(&id_str)
        .map_err(|e| Error::bad_request(format!("invalid UUID: {e}")))?;
    let repo = s.repo.lock().map_err(|_| Error::locked())?;
    let node = repo
        .lookup_node(&id)?
        .ok_or_else(|| Error::not_found(format!("no node with id={id_str}")))?;

    let (_, node_cid) = mnem_core::codec::hash_to_cid(&node)
        .map_err(|e| Error::internal(format!("hash node: {e}")))?;

    let models = repo
        .embedding_models_for(&node_cid)
        .map_err(|e| Error::internal(format!("list embeddings: {e}")))?;

    Ok(Json(json!({
        "schema": "mnem.v1.node-embeddings",
        "node_id": id_str,
        "models": models,
        "count": models.len(),
    })))
}

// ---------- GET /v1/show ----------

/// Query parameters for `GET /v1/show`.
#[derive(Deserialize)]
pub(crate) struct ShowQuery {
    /// CID to decode. If absent, defaults to the current op-head.
    pub cid: Option<String>,
}

/// Decode any CID and return a structured JSON summary.
///
/// Mirrors `mnem show [<cid>]`. Peeks the `_kind` discriminator and
/// re-decodes into the concrete type, returning the same fields the CLI
/// pretty-printer surfaces. Unknown kinds return the kind string and
/// byte count so callers know what they have.
pub(crate) async fn get_show(
    State(s): State<AppState>,
    Query(q): Query<ShowQuery>,
) -> Result<Json<Value>, Error> {
    let repo = s.repo.lock().map_err(|_| Error::locked())?;

    let target_cid: Cid = match q.cid {
        Some(ref s) => Cid::parse_str(s)
            .map_err(|e| Error::bad_request(format!("invalid CID: {e}")))?,
        None => repo.op_id().clone(),
    };

    let bs = repo.blockstore();
    let bytes = bs
        .get(&target_cid)
        .map_err(|e| Error::internal(format!("blockstore read: {e}")))?
        .ok_or_else(|| Error::not_found(format!("block {target_cid} not found")))?;

    let kind = {
        let ipld: Option<Ipld> = from_canonical_bytes::<Ipld>(&bytes).ok();
        ipld.and_then(|i| match i {
            Ipld::Map(m) => match m.get("_kind")? {
                Ipld::String(s) => Some(s.clone()),
                _ => None,
            },
            _ => None,
        })
    };

    let mut obj = serde_json::Map::new();
    obj.insert("schema".into(), json!("mnem.v1.show"));
    obj.insert("cid".into(), json!(target_cid.to_string()));
    obj.insert("size".into(), json!(bytes.len()));
    obj.insert(
        "kind".into(),
        json!(kind.as_deref().unwrap_or("<unknown>")),
    );

    match kind.as_deref() {
        Some("node") => {
            if let Ok(n) = from_canonical_bytes::<mnem_core::objects::Node>(&bytes) {
                obj.insert("id".into(), json!(n.id.to_uuid_string()));
                obj.insert("ntype".into(), json!(n.ntype));
                obj.insert("summary".into(), json!(n.summary));
                if let Some(ref ctx) = n.context_sentence {
                    obj.insert("context_sentence".into(), json!(ctx));
                }
                let props_map: serde_json::Map<String, Value> = n
                    .props
                    .iter()
                    .map(|(k, v)| (k.clone(), ipld_to_json(v)))
                    .collect();
                obj.insert("props".into(), Value::Object(props_map));
                obj.insert(
                    "content_bytes".into(),
                    json!(n.content.as_ref().map_or(0, bytes::Bytes::len)),
                );
                // Embedding detail — mirrors CLI `show_node()` output.
                if let Ok(models) = repo.embedding_models_for(&target_cid) {
                    let embeds: Vec<Value> = models
                        .iter()
                        .filter_map(|model| {
                            repo.embedding_for(&target_cid, model).ok().flatten().map(
                                |emb| {
                                    json!({
                                        "model": emb.model,
                                        "dim": emb.dim,
                                        "dtype": serde_json::to_value(&emb.dtype)
                                            .unwrap_or(json!("f32")),
                                    })
                                },
                            )
                        })
                        .collect();
                    if !embeds.is_empty() {
                        obj.insert("embeddings".into(), json!(embeds));
                    }
                }
            }
        }
        Some("edge") => {
            if let Ok(e) = from_canonical_bytes::<mnem_core::objects::Edge>(&bytes) {
                obj.insert("id".into(), json!(e.id.to_uuid_string()));
                obj.insert("etype".into(), json!(e.etype));
                obj.insert("src".into(), json!(e.src.to_uuid_string()));
                obj.insert("dst".into(), json!(e.dst.to_uuid_string()));
                let props_map: serde_json::Map<String, Value> = e
                    .props
                    .iter()
                    .map(|(k, v)| (k.clone(), ipld_to_json(v)))
                    .collect();
                obj.insert("props".into(), Value::Object(props_map));
            }
        }
        Some("commit") => {
            if let Ok(c) = from_canonical_bytes::<mnem_core::objects::Commit>(&bytes) {
                obj.insert("change_id".into(), json!(c.change_id.to_uuid_string()));
                obj.insert("author".into(), json!(c.author));
                if let Some(a) = &c.agent_id {
                    obj.insert("agent_id".into(), json!(a));
                }
                if let Some(t) = &c.task_id {
                    obj.insert("task_id".into(), json!(t));
                }
                obj.insert("message".into(), json!(c.message));
                obj.insert("time".into(), json!(c.time));
                obj.insert("nodes".into(), json!(c.nodes.to_string()));
                obj.insert("edges".into(), json!(c.edges.to_string()));
                obj.insert("schema_tree_cid".into(), json!(c.schema.to_string()));
                if let Some(i) = &c.indexes {
                    obj.insert("indexes".into(), json!(i.to_string()));
                }
                obj.insert(
                    "parents".into(),
                    json!(c
                        .parents
                        .iter()
                        .map(ToString::to_string)
                        .collect::<Vec<_>>()),
                );
                obj.insert("has_signature".into(), json!(c.signature.is_some()));
            }
        }
        Some("operation") => {
            if let Ok(op) = from_canonical_bytes::<mnem_core::objects::Operation>(&bytes) {
                obj.insert("author".into(), json!(op.author));
                if let Some(a) = &op.agent_id {
                    obj.insert("agent_id".into(), json!(a));
                }
                if let Some(t) = &op.task_id {
                    obj.insert("task_id".into(), json!(t));
                }
                obj.insert("description".into(), json!(op.description));
                obj.insert("time".into(), json!(op.time));
                obj.insert(
                    "parents".into(),
                    json!(op
                        .parents
                        .iter()
                        .map(ToString::to_string)
                        .collect::<Vec<_>>()),
                );
                obj.insert("view".into(), json!(op.view.to_string()));
                // Decode the view block to surface head CID and refs count,
                // matching CLI show_operation() output.
                if let Ok(Some(vb)) = bs.get(&op.view) {
                    if let Ok(v) = from_canonical_bytes::<mnem_core::objects::View>(&vb) {
                        if let Some(head_cid) = v.heads.first() {
                            obj.insert("view_head".into(), json!(head_cid.to_string()));
                        }
                        obj.insert("view_refs_count".into(), json!(v.refs.len()));
                    }
                }
            }
        }
        Some("view") => {
            if let Ok(v) = from_canonical_bytes::<mnem_core::objects::View>(&bytes) {
                obj.insert("heads".into(), json!(v.heads.len()));
                obj.insert("refs".into(), json!(v.refs.len()));
            }
        }
        Some("index_set") => {
            if let Ok(idx) =
                from_canonical_bytes::<mnem_core::objects::IndexSet>(&bytes)
            {
                obj.insert("labels".into(), json!(idx.nodes_by_label.len()));
            }
        }
        Some("tombstone") => {
            if let Ok(t) = from_canonical_bytes::<mnem_core::objects::Tombstone>(&bytes) {
                obj.insert("tombstoned_at".into(), json!(t.tombstoned_at));
                obj.insert("reason".into(), json!(t.reason));
            }
        }
        _ => {}
    }

    Ok(Json(Value::Object(obj)))
}

// ---- reindex helpers ----

pub(crate) fn reindex_ipld_to_text(v: &Ipld) -> String {
    match v {
        Ipld::Null => "null".into(),
        Ipld::Bool(b) => b.to_string(),
        Ipld::Integer(n) => n.to_string(),
        Ipld::Float(f) => f.to_string(),
        Ipld::String(s) => s.clone(),
        Ipld::Bytes(b) => format!("bytes({})", b.len()),
        Ipld::List(xs) => format!("[{} items]", xs.len()),
        Ipld::Map(m) => format!("{{{} keys}}", m.len()),
        Ipld::Link(c) => format!("cid:{c}"),
    }
}

pub(crate) fn reindex_fallback_text_of(node: &Node) -> String {
    let mut parts: Vec<String> = vec![node.ntype.clone()];
    let mut sorted: Vec<(&String, &Ipld)> = node.props.iter().collect();
    sorted.sort_by_key(|(k, _)| k.as_str());
    for (k, v) in sorted {
        parts.push(format!("{k}: {}", reindex_ipld_to_text(v)));
    }
    parts.join(". ")
}

pub(crate) fn reindex_text_of_node(node: &Node) -> String {
    if let Some(s) = &node.summary {
        if !s.trim().is_empty() {
            return s.clone();
        }
    }
    if let Some(text) = embed_text_of(node) {
        return text;
    }
    reindex_fallback_text_of(node)
}

fn reindex_resolve_commitish(
    r: &mnem_core::repo::ReadonlyRepo,
    s: &str,
) -> Result<Cid, Error> {
    if s.eq_ignore_ascii_case("HEAD") {
        return r
            .view()
            .heads
            .first()
            .cloned()
            .ok_or_else(|| Error::bad_request("repository has no commits yet (HEAD unresolved)"));
    }
    if let Ok(cid) = Cid::parse_str(s) {
        return Ok(cid);
    }
    let refs = &r.view().refs;
    let candidate = if refs.contains_key(s) {
        s.to_string()
    } else {
        format!("{HEADS_PREFIX}{s}")
    };
    match refs.get(&candidate) {
        Some(mnem_core::objects::RefTarget::Normal { target }) => Ok(target.clone()),
        Some(mnem_core::objects::RefTarget::Conflicted { .. }) => Err(Error::bad_request(
            format!("ref `{candidate}` is conflicted; resolve the ref first"),
        )),
        None => Err(Error::bad_request(format!(
            "cannot resolve `{s}` to a commit (tried HEAD, raw CID, ref `{s}`, \
             and `{HEADS_PREFIX}{s}`)"
        ))),
    }
}

fn reindex_nodes_at(
    bs: &std::sync::Arc<dyn mnem_core::store::Blockstore>,
    commit_cid: &Cid,
) -> Result<std::collections::HashSet<Cid>, Error> {
    let bytes = bs
        .get(commit_cid)
        .map_err(|e| Error::internal(e.to_string()))?
        .ok_or_else(|| {
            Error::bad_request(format!("commit CID {commit_cid} missing from store"))
        })?;
    let commit: Commit =
        from_canonical_bytes(&bytes).map_err(|e| Error::internal(e.to_string()))?;
    let mut out = std::collections::HashSet::new();
    let cursor = Cursor::new(&**bs, &commit.nodes)
        .map_err(|e| Error::internal(format!("cursor: {e}")))?;
    for entry in cursor {
        let (_k, node_cid) = entry.map_err(|e| Error::internal(e.to_string()))?;
        out.insert(node_cid);
    }
    Ok(out)
}

fn decode_reindex_embedding(
    val: &Ipld,
) -> Result<mnem_core::objects::node::Embedding, Error> {
    let bytes = to_canonical_bytes(val)
        .map_err(|e| Error::internal(format!("CBOR re-encode of extra[\"embed\"]: {e}")))?;
    let emb: mnem_core::objects::node::Embedding = from_canonical_bytes(&bytes)
        .map_err(|e| Error::internal(format!("decode extra[\"embed\"] as Embedding: {e}")))?;
    emb.validate()
        .map_err(|e| Error::internal(format!("extra[\"embed\"] invariant violated: {e:?}")))?;
    Ok(emb)
}

fn decode_reindex_sparse(
    val: &Ipld,
) -> Result<mnem_core::sparse::SparseEmbed, Error> {
    let bytes = to_canonical_bytes(val).map_err(|e| {
        Error::internal(format!("CBOR re-encode of extra[\"sparse_embed\"]: {e}"))
    })?;
    let se: mnem_core::sparse::SparseEmbed = from_canonical_bytes(&bytes).map_err(|e| {
        Error::internal(format!(
            "decode extra[\"sparse_embed\"] as SparseEmbed: {e}"
        ))
    })?;
    se.validate().map_err(|e| {
        Error::internal(format!("extra[\"sparse_embed\"] invariant violated: {e}"))
    })?;
    Ok(se)
}

#[derive(Debug, Deserialize)]
pub(crate) struct ReindexBody {
    #[serde(default)]
    force: bool,
    label: Option<String>,
    since: Option<String>,
    #[serde(default)]
    dry_run: bool,
    message: Option<String>,
    #[serde(default)]
    lift_legacy_extra: bool,
    #[serde(default)]
    lift_legacy_sparse: bool,
    author: Option<String>,
}

pub(crate) async fn post_reindex(
    _auth: RequireBearer,
    State(s): State<AppState>,
    Json(body): Json<ReindexBody>,
) -> Result<Json<Value>, Error> {
    if body.lift_legacy_extra && body.force {
        return Err(Error::bad_request(
            "lift_legacy_extra and force are mutually exclusive",
        ));
    }
    if body.lift_legacy_sparse && body.force {
        return Err(Error::bad_request(
            "lift_legacy_sparse and force are mutually exclusive",
        ));
    }

    if let Some(a) = body.author.as_deref() {
        validate_author(a.trim())?;
    }
    validate_message(body.message.as_deref())?;
    let is_lift_only = body.lift_legacy_extra || body.lift_legacy_sparse;
    let author = body.author.as_deref().unwrap_or("mnem-http").to_string();

    if !is_lift_only && s.embed_cfg.is_none() {
        return Err(Error::status(
            axum::http::StatusCode::SERVICE_UNAVAILABLE,
            "no embedder configured; set [embed] in config.toml first",
        ));
    }

    let r = s.repo.lock().map_err(|_| Error::locked())?;
    let Some(head) = r.head_commit() else {
        let mode = if body.lift_legacy_extra {
            "lift_legacy_extra"
        } else if body.lift_legacy_sparse {
            "lift_legacy_sparse"
        } else {
            "normal"
        };
        return Ok(Json(json!({
            "schema": "mnem.v1.reindex",
            "mode": mode,
            "total_nodes": 0,
            "note": "repository has no commits yet",
        })));
    };
    let head = head.clone();

    let bs = r.blockstore().clone();

    let since_set: Option<std::collections::HashSet<Cid>> = match &body.since {
        None => None,
        Some(s_ref) => {
            let cid = reindex_resolve_commitish(&r, s_ref)?;
            Some(reindex_nodes_at(&bs, &cid)?)
        }
    };

    // ---- lift-legacy-extra path ----
    if body.lift_legacy_extra {
        let mut total_nodes: usize = 0;
        let mut legacy_count: usize = 0;
        let mut decode_errors: usize = 0;
        let mut to_lift: Vec<(Cid, mnem_core::objects::node::Embedding)> = Vec::new();

        let cursor = Cursor::new(&*bs, &head.nodes)
            .map_err(|e| Error::internal(format!("cursor: {e}")))?;
        for entry in cursor {
            let (_k, node_cid) = entry.map_err(|e| Error::internal(e.to_string()))?;
            let bytes = bs
                .get(&node_cid)
                .map_err(|e| Error::internal(e.to_string()))?
                .ok_or_else(|| Error::internal(format!("node CID {node_cid} missing")))?;
            let node: Node =
                from_canonical_bytes(&bytes).map_err(|e| Error::internal(e.to_string()))?;
            total_nodes += 1;
            if let Some(set) = &since_set {
                if set.contains(&node_cid) {
                    continue;
                }
            }
            if let Some(lbl) = &body.label {
                if &node.ntype != lbl {
                    continue;
                }
            }
            if mnem_core::anchor::is_system_node(&node) {
                continue;
            }
            let Some(ipld_val) = node.extra.get("embed") else {
                continue;
            };
            match decode_reindex_embedding(ipld_val) {
                Ok(emb) => {
                    legacy_count += 1;
                    to_lift.push((node_cid, emb));
                }
                Err(_) => {
                    decode_errors += 1;
                }
            }
        }

        if body.dry_run {
            return Ok(Json(json!({
                "schema": "mnem.v1.reindex",
                "mode": "lift_legacy_extra",
                "total_nodes": total_nodes,
                "would_lift": legacy_count,
                "decode_errors": decode_errors,
                "dry_run": true,
            })));
        }
        if to_lift.is_empty() {
            return Ok(Json(json!({
                "schema": "mnem.v1.reindex",
                "mode": "lift_legacy_extra",
                "total_nodes": total_nodes,
                "lifted": 0,
                "decode_errors": decode_errors,
                "dry_run": false,
            })));
        }

        let total = to_lift.len();
        let mut tx = r.start_transaction();
        for (node_cid, emb) in to_lift {
            let model = emb.model.clone();
            tx.set_embedding(node_cid, model, emb)
                .map_err(|e| Error::internal(e.to_string()))?;
        }
        let msg = body.message.clone().unwrap_or_else(|| {
            format!(
                "mnem reindex --lift-legacy-extra: {total} embedding(s) promoted to sidecar"
            )
        });
        let new_repo = tx
            .commit(&author, &msg)
            .map_err(|e| Error::internal(e.to_string()))?;
        let new_op_id = new_repo.op_id().to_string();
        drop(r);
        let mut guard = s.repo.lock().map_err(|_| Error::locked())?;
        *guard = new_repo;
        return Ok(Json(json!({
            "schema": "mnem.v1.reindex",
            "mode": "lift_legacy_extra",
            "total_nodes": total_nodes,
            "lifted": total,
            "decode_errors": decode_errors,
            "dry_run": false,
            "op_id": new_op_id,
        })));
    }

    // ---- lift-legacy-sparse path ----
    if body.lift_legacy_sparse {
        let mut total_nodes: usize = 0;
        let mut legacy_count: usize = 0;
        let mut decode_errors: usize = 0;
        let mut to_lift: Vec<(Cid, mnem_core::sparse::SparseEmbed)> = Vec::new();

        let cursor = Cursor::new(&*bs, &head.nodes)
            .map_err(|e| Error::internal(format!("cursor: {e}")))?;
        for entry in cursor {
            let (_k, node_cid) = entry.map_err(|e| Error::internal(e.to_string()))?;
            let bytes = bs
                .get(&node_cid)
                .map_err(|e| Error::internal(e.to_string()))?
                .ok_or_else(|| Error::internal(format!("node CID {node_cid} missing")))?;
            let node: Node =
                from_canonical_bytes(&bytes).map_err(|e| Error::internal(e.to_string()))?;
            total_nodes += 1;
            if let Some(set) = &since_set {
                if set.contains(&node_cid) {
                    continue;
                }
            }
            if let Some(lbl) = &body.label {
                if &node.ntype != lbl {
                    continue;
                }
            }
            if mnem_core::anchor::is_system_node(&node) {
                continue;
            }
            let Some(ipld_val) = node.extra.get("sparse_embed") else {
                continue;
            };
            let se = match decode_reindex_sparse(ipld_val) {
                Ok(s) => s,
                Err(_) => {
                    decode_errors += 1;
                    continue;
                }
            };
            if r.sparse_for(&node_cid, &se.vocab_id)
                .map_err(|e| Error::internal(e.to_string()))?
                .is_some()
            {
                continue;
            }
            legacy_count += 1;
            to_lift.push((node_cid, se));
        }

        if body.dry_run {
            return Ok(Json(json!({
                "schema": "mnem.v1.reindex",
                "mode": "lift_legacy_sparse",
                "total_nodes": total_nodes,
                "would_lift": legacy_count,
                "decode_errors": decode_errors,
                "dry_run": true,
            })));
        }
        if to_lift.is_empty() {
            return Ok(Json(json!({
                "schema": "mnem.v1.reindex",
                "mode": "lift_legacy_sparse",
                "total_nodes": total_nodes,
                "lifted": 0,
                "decode_errors": decode_errors,
                "dry_run": false,
            })));
        }

        let total = to_lift.len();
        let mut tx = r.start_transaction();
        for (node_cid, se) in to_lift {
            let vocab_id = se.vocab_id.clone();
            tx.set_sparse_embedding(node_cid, vocab_id, se)
                .map_err(|e| Error::internal(e.to_string()))?;
        }
        let msg = body.message.clone().unwrap_or_else(|| {
            format!(
                "mnem reindex --lift-legacy-sparse: {total} sparse embedding(s) promoted to sidecar"
            )
        });
        let new_repo = tx
            .commit(&author, &msg)
            .map_err(|e| Error::internal(e.to_string()))?;
        let new_op_id = new_repo.op_id().to_string();
        drop(r);
        let mut guard = s.repo.lock().map_err(|_| Error::locked())?;
        *guard = new_repo;
        return Ok(Json(json!({
            "schema": "mnem.v1.reindex",
            "mode": "lift_legacy_sparse",
            "total_nodes": total_nodes,
            "lifted": total,
            "decode_errors": decode_errors,
            "dry_run": false,
            "op_id": new_op_id,
        })));
    }

    // ---- normal embed path ----
    let pc = s.embed_cfg.as_ref().ok_or_else(|| {
        Error::status(
            axum::http::StatusCode::SERVICE_UNAVAILABLE,
            "no embedder configured; set [embed] in config.toml first",
        )
    })?;
    let embedder = mnem_embed_providers::open(pc)
        .map_err(|e| Error::internal(format!("open embedder: {e}")))?;
    let model_fq = embedder.model().to_string();

    let mut candidates: Vec<(Cid, Node)> = Vec::new();
    let mut total_nodes: usize = 0;
    let mut matched_label: usize = 0;
    let mut skipped_already_embedded: usize = 0;
    let mut skipped_outside_since: usize = 0;

    let cursor = Cursor::new(&*bs, &head.nodes)
        .map_err(|e| Error::internal(format!("cursor: {e}")))?;
    for entry in cursor {
        let (_k, node_cid) = entry.map_err(|e| Error::internal(e.to_string()))?;
        let bytes = bs
            .get(&node_cid)
            .map_err(|e| Error::internal(e.to_string()))?
            .ok_or_else(|| Error::internal(format!("node CID {node_cid} missing")))?;
        let node: Node =
            from_canonical_bytes(&bytes).map_err(|e| Error::internal(e.to_string()))?;
        total_nodes += 1;
        if let Some(set) = &since_set {
            if set.contains(&node_cid) {
                skipped_outside_since += 1;
                continue;
            }
        }
        if let Some(lbl) = &body.label {
            if &node.ntype != lbl {
                continue;
            }
        }
        matched_label += 1;
        let already = if body.force {
            false
        } else {
            r.embedding_for(&node_cid, &model_fq)
                .map_err(|e| Error::internal(e.to_string()))?
                .is_some()
        };
        if already {
            skipped_already_embedded += 1;
            continue;
        }
        if mnem_core::anchor::is_system_node(&node) {
            continue;
        }
        candidates.push((node_cid, node));
    }

    let would_embed = candidates.len();
    if body.dry_run {
        return Ok(Json(json!({
            "schema": "mnem.v1.reindex",
            "model": model_fq,
            "total_nodes": total_nodes,
            "matched_label": matched_label,
            "skipped_already_embedded": skipped_already_embedded,
            "skipped_outside_since": skipped_outside_since,
            "would_embed": would_embed,
            "dry_run": true,
        })));
    }
    if candidates.is_empty() {
        return Ok(Json(json!({
            "schema": "mnem.v1.reindex",
            "model": model_fq,
            "total_nodes": total_nodes,
            "matched_label": matched_label,
            "skipped_already_embedded": skipped_already_embedded,
            "skipped_outside_since": skipped_outside_since,
            "embedded": 0,
            "dry_run": false,
        })));
    }

    let total = candidates.len();
    let mut tx = r.start_transaction();
    for (node_cid, node) in candidates {
        let text = reindex_text_of_node(&node);
        let v = embedder
            .embed(&text)
            .map_err(|e| Error::internal(format!("embed: {e}")))?;
        let emb = mnem_embed_providers::to_embedding(&model_fq, &v);
        tx.set_embedding(node_cid, model_fq.clone(), emb)
            .map_err(|e| Error::internal(e.to_string()))?;
    }
    let msg = body
        .message
        .clone()
        .unwrap_or_else(|| format!("mnem reindex: {total} nodes embedded with {model_fq}"));
    let new_repo = tx
        .commit(&author, &msg)
        .map_err(|e| Error::internal(e.to_string()))?;
    let new_op_id = new_repo.op_id().to_string();
    drop(r);
    let mut guard = s.repo.lock().map_err(|_| Error::locked())?;
    *guard = new_repo;

    Ok(Json(json!({
        "schema": "mnem.v1.reindex",
        "model": model_fq,
        "total_nodes": total_nodes,
        "matched_label": matched_label,
        "skipped_already_embedded": skipped_already_embedded,
        "skipped_outside_since": skipped_outside_since,
        "embedded": total,
        "dry_run": false,
        "op_id": new_op_id,
    })))
}

// ---------- POST /v1/revert ----------

#[derive(Deserialize)]
pub(crate) struct RevertBody {
    /// Op CID to invert (find with `GET /v1/log`).
    pub commit: String,
    /// Commit message for the revert operation.
    pub message: Option<String>,
    /// Author string. Defaults to `"mnem-http"` when absent.
    pub author: Option<String>,
}

/// Walk the op-log backwards from `start` looking for `target_cid`,
/// returning `(target_op, parent_op)`. `parent_op` is the op immediately
/// preceding the target (i.e. `target_op.parents[0]`). Returns `None`
/// when the target is not reachable within 100 000 ops.
fn find_op_and_parent(
    bs: &dyn mnem_core::store::Blockstore,
    start: &Cid,
    target_cid: &Cid,
) -> Result<Option<(Operation, Option<Operation>)>, Error> {
    let mut cur = start.clone();
    for _ in 0..100_000usize {
        let bytes = bs
            .get(&cur)
            .map_err(|e| Error::internal(e.to_string()))?
            .ok_or_else(|| Error::internal(format!("op {cur} missing from blockstore")))?;
        let op: Operation = from_canonical_bytes(&bytes)
            .map_err(|e| Error::internal(format!("decode op: {e}")))?;

        if &cur == target_cid {
            let parent_op: Option<Operation> = match op.parents.first() {
                None => None,
                Some(parent_cid) => {
                    let pb = bs
                        .get(parent_cid)
                        .map_err(|e| Error::internal(e.to_string()))?
                        .ok_or_else(|| {
                            Error::internal(format!("parent op {parent_cid} missing"))
                        })?;
                    Some(
                        from_canonical_bytes(&pb)
                            .map_err(|e| Error::internal(format!("decode parent op: {e}")))?,
                    )
                }
            };
            return Ok(Some((op, parent_op)));
        }

        match op.parents.first() {
            Some(p) => cur = p.clone(),
            None => break,
        }
    }
    Ok(None)
}

pub(crate) async fn post_revert(
    _auth: RequireBearer,
    State(s): State<AppState>,
    Json(body): Json<RevertBody>,
) -> Result<Json<Value>, Error> {
    if let Some(a) = &body.author {
        if a.trim().is_empty() {
            return Err(Error::bad_request("author must not be blank"));
        }
        if a.len() > MAX_AUTHOR_LEN {
            return Err(Error::bad_request(format!(
                "author exceeds maximum length of {MAX_AUTHOR_LEN} bytes"
            )));
        }
    }
    if let Some(m) = &body.message {
        if m.len() > MAX_MESSAGE_LEN {
            return Err(Error::bad_request(format!(
                "message exceeds maximum length of {MAX_MESSAGE_LEN} bytes"
            )));
        }
    }

    let target_cid = Cid::parse_str(&body.commit)
        .map_err(|_| Error::bad_request(format!("invalid CID: `{}`", body.commit)))?;

    let author = body
        .author
        .as_deref()
        .map(str::trim)
        .filter(|a| !a.is_empty())
        .unwrap_or("mnem-http")
        .to_string();

    let r = s.repo.lock().map_err(|_| Error::locked())?;
    let bs = r.blockstore().clone();
    let head_op_cid = r.op_id().clone();

    let (target_op, parent_op_opt) =
        find_op_and_parent(&*bs, &head_op_cid, &target_cid)?.ok_or_else(|| {
            Error::bad_request(format!(
                "op `{}` not found in op-log; use GET /v1/log to list available ops",
                body.commit
            ))
        })?;

    // Resolve target view and commit CID.
    let target_view_bytes = bs
        .get(&target_op.view)
        .map_err(|e| Error::internal(e.to_string()))?
        .ok_or_else(|| {
            Error::internal(format!(
                "view block for op `{}` missing from blockstore",
                body.commit
            ))
        })?;
    let target_view: View = from_canonical_bytes(&target_view_bytes)
        .map_err(|e| Error::internal(format!("decode view: {e}")))?;

    let target_commit_cid = target_view.heads.first().ok_or_else(|| {
        Error::bad_request(format!(
            "op `{}` has no head commit (init or ref-only op); nothing to revert",
            body.commit
        ))
    })?;

    // Resolve parent view and commit CID (the "before" state).
    let (parent_view_opt, parent_commit_cid_opt): (Option<View>, Option<Cid>) =
        match &parent_op_opt {
            None => (None, None),
            Some(pop) => {
                let pv_bytes = bs
                    .get(&pop.view)
                    .map_err(|e| Error::internal(e.to_string()))?
                    .ok_or_else(|| {
                        Error::internal("view block for parent op missing from blockstore")
                    })?;
                let pview: View = from_canonical_bytes(&pv_bytes)
                    .map_err(|e| Error::internal(format!("decode parent view: {e}")))?;
                let pcid = pview.heads.first().cloned();
                (Some(pview), pcid)
            }
        };

    // Load both commits and compute prolly-tree diffs.
    let target_commit_bytes = bs
        .get(target_commit_cid)
        .map_err(|e| Error::internal(e.to_string()))?
        .ok_or_else(|| {
            Error::internal(format!(
                "commit block `{target_commit_cid}` missing from blockstore"
            ))
        })?;
    let target_commit: Commit = from_canonical_bytes(&target_commit_bytes)
        .map_err(|e| Error::internal(format!("decode commit: {e}")))?;

    let (before_nodes_root, before_edges_root) = match &parent_commit_cid_opt {
        Some(pcid) => {
            let pc_bytes = bs
                .get(pcid)
                .map_err(|e| Error::internal(e.to_string()))?
                .ok_or_else(|| {
                    Error::internal(format!(
                        "parent commit block `{pcid}` missing from blockstore"
                    ))
                })?;
            let pc: Commit = from_canonical_bytes(&pc_bytes)
                .map_err(|e| Error::internal(format!("decode parent commit: {e}")))?;
            (pc.nodes.clone(), pc.edges.clone())
        }
        None => {
            let empty = build_tree(&*bs, std::iter::empty())
                .map_err(|e| Error::internal(format!("build empty tree: {e}")))?;
            (empty.clone(), empty)
        }
    };

    let node_changes = prolly_diff(&*bs, &before_nodes_root, &target_commit.nodes)
        .map_err(|e| Error::internal(format!("node diff: {e}")))?;
    let edge_changes = prolly_diff(&*bs, &before_edges_root, &target_commit.edges)
        .map_err(|e| Error::internal(format!("edge diff: {e}")))?;

    // Tombstone diff: nodes tombstoned by this op must be un-tombstoned on revert.
    let parent_tombstones: std::collections::BTreeMap<NodeId, mnem_core::objects::Tombstone> =
        parent_view_opt
            .as_ref()
            .map(|pv| pv.tombstones.clone())
            .unwrap_or_default();
    let tombstones_added_by_op: Vec<NodeId> = target_view
        .tombstones
        .keys()
        .filter(|id| !parent_tombstones.contains_key(*id))
        .copied()
        .collect();

    if node_changes.is_empty() && edge_changes.is_empty() && tombstones_added_by_op.is_empty() {
        return Ok(Json(json!({
            "schema": "mnem.v1.revert",
            "status": "noop",
            "message": "op made no node/edge/tombstone changes - nothing to revert",
        })));
    }

    // BUG-4 pre-flight: verify edge endpoints are alive before we try to re-add removed edges.
    for entry in &edge_changes {
        if let DiffEntry::Removed { value, .. } = entry {
            let edge_bytes = bs
                .get(value)
                .map_err(|e| Error::internal(e.to_string()))?
                .ok_or_else(|| Error::internal(format!("edge block `{value}` missing")))?;
            let edge: mnem_core::objects::Edge = from_canonical_bytes(&edge_bytes)
                .map_err(|e| Error::internal(format!("decode edge: {e}")))?;
            for (endpoint_id, role) in [(edge.src, "src"), (edge.dst, "dst")] {
                let exists = r
                    .lookup_node(&endpoint_id)
                    .map_err(|e| Error::internal(e.to_string()))?
                    .is_some();
                let tombstoned = r.is_tombstoned(&endpoint_id);
                if !exists || tombstoned {
                    return Err(Error::bad_request(format!(
                        "cannot revert op `{}`: edge endpoint {} ({role}) no longer exists \
                         (deleted or tombstoned since the op was applied). \
                         Revert the deletion first, or skip reverting this op.",
                        body.commit, endpoint_id
                    )));
                }
            }
        }
    }

    let mut tx = r.start_transaction();
    let mut mutations_applied: usize = 0;

    for entry in &node_changes {
        match entry {
            DiffEntry::Added { value, .. } => {
                let bytes = bs
                    .get(value)
                    .map_err(|e| Error::internal(e.to_string()))?
                    .ok_or_else(|| Error::internal(format!("node block `{value}` missing")))?;
                let node: Node =
                    from_canonical_bytes(&bytes).map_err(|e| Error::internal(e.to_string()))?;
                if tx.base().lookup_node(&node.id).map_err(|e| Error::internal(e.to_string()))?.is_some() {
                    tx.remove_node(node.id);
                    mutations_applied += 1;
                }
            }
            DiffEntry::Removed { value, .. } => {
                let bytes = bs
                    .get(value)
                    .map_err(|e| Error::internal(e.to_string()))?
                    .ok_or_else(|| Error::internal(format!("node block `{value}` missing")))?;
                let node: Node =
                    from_canonical_bytes(&bytes).map_err(|e| Error::internal(e.to_string()))?;
                if tx.base().lookup_node(&node.id).map_err(|e| Error::internal(e.to_string()))?.is_none() {
                    tx.add_node(&node).map_err(|e| Error::internal(e.to_string()))?;
                    mutations_applied += 1;
                }
            }
            DiffEntry::Changed { before, .. } => {
                let bytes = bs
                    .get(before)
                    .map_err(|e| Error::internal(e.to_string()))?
                    .ok_or_else(|| Error::internal(format!("node block `{before}` missing")))?;
                let node: Node =
                    from_canonical_bytes(&bytes).map_err(|e| Error::internal(e.to_string()))?;
                let current_is_before = match tx.base().lookup_node(&node.id).map_err(|e| Error::internal(e.to_string()))? {
                    None => false,
                    Some(ref cur) => cur == &node,
                };
                if !current_is_before {
                    tx.add_node(&node).map_err(|e| Error::internal(e.to_string()))?;
                    mutations_applied += 1;
                }
            }
        }
    }

    for entry in &edge_changes {
        match entry {
            DiffEntry::Added { value, .. } => {
                let bytes = bs
                    .get(value)
                    .map_err(|e| Error::internal(e.to_string()))?
                    .ok_or_else(|| Error::internal(format!("edge block `{value}` missing")))?;
                let edge: mnem_core::objects::Edge =
                    from_canonical_bytes(&bytes).map_err(|e| Error::internal(e.to_string()))?;
                if tx.base().lookup_edge(&edge.id).map_err(|e| Error::internal(e.to_string()))?.is_some() {
                    tx.remove_edge(edge.id);
                    mutations_applied += 1;
                }
            }
            DiffEntry::Removed { value, .. } => {
                let bytes = bs
                    .get(value)
                    .map_err(|e| Error::internal(e.to_string()))?
                    .ok_or_else(|| Error::internal(format!("edge block `{value}` missing")))?;
                let edge: mnem_core::objects::Edge =
                    from_canonical_bytes(&bytes).map_err(|e| Error::internal(e.to_string()))?;
                if tx.base().lookup_edge(&edge.id).map_err(|e| Error::internal(e.to_string()))?.is_none() {
                    tx.add_edge(&edge).map_err(|e| Error::internal(e.to_string()))?;
                    mutations_applied += 1;
                }
            }
            DiffEntry::Changed { before, .. } => {
                let bytes = bs
                    .get(before)
                    .map_err(|e| Error::internal(e.to_string()))?
                    .ok_or_else(|| Error::internal(format!("edge block `{before}` missing")))?;
                let edge: mnem_core::objects::Edge =
                    from_canonical_bytes(&bytes).map_err(|e| Error::internal(e.to_string()))?;
                let current_is_before = match tx.base().lookup_edge(&edge.id).map_err(|e| Error::internal(e.to_string()))? {
                    None => false,
                    Some(ref cur) => cur == &edge,
                };
                if !current_is_before {
                    tx.add_edge(&edge).map_err(|e| Error::internal(e.to_string()))?;
                    mutations_applied += 1;
                }
            }
        }
    }

    // Invert tombstone changes.
    for node_id in &tombstones_added_by_op {
        if tx.base().is_tombstoned(node_id) {
            tx.untombstone_node(*node_id);
            mutations_applied += 1;
        }
    }

    if mutations_applied == 0 {
        return Ok(Json(json!({
            "schema": "mnem.v1.revert",
            "status": "noop",
            "message": "inverse changes are all no-ops in the current tree (op may already be reverted)",
        })));
    }

    let default_msg = format!("revert: {}", body.commit);
    let msg = body.message.as_deref().unwrap_or(&default_msg);
    let new_repo = tx
        .commit(&author, msg)
        .map_err(|e| Error::internal(format!("commit revert: {e}")))?;

    let new_op_id = new_repo.op_id().to_string();
    let new_head = new_repo.view().heads.first().map(ToString::to_string);

    drop(r);
    let mut guard = s.repo.lock().map_err(|_| Error::locked())?;
    *guard = new_repo;

    Ok(Json(json!({
        "schema": "mnem.v1.revert",
        "status": "ok",
        "reverted_op": body.commit,
        "new_op": new_op_id,
        "new_head": new_head,
        "mutations_applied": mutations_applied,
    })))
}

// ---- config helpers ----

fn load_config_toml(path: &std::path::Path) -> Result<toml::Table, Error> {
    if !path.exists() {
        return Ok(toml::Table::new());
    }
    let text = std::fs::read_to_string(path)
        .map_err(|e| Error::internal(format!("read config.toml: {e}")))?;
    text.parse::<toml::Table>()
        .map_err(|e| Error::internal(format!("parse config.toml: {e}")))
}

fn save_config_toml(path: &std::path::Path, table: &toml::Table) -> Result<(), Error> {
    let text = toml::to_string_pretty(table)
        .map_err(|e| Error::internal(format!("serialize config.toml: {e}")))?;
    std::fs::write(path, text)
        .map_err(|e| Error::internal(format!("write config.toml: {e}")))
}

/// Flatten a TOML table into dotted-key string pairs.
fn flatten_toml(prefix: &str, val: &toml::Value, out: &mut Map<String, Value>) {
    match val {
        toml::Value::Table(t) => {
            for (k, v) in t {
                let full = if prefix.is_empty() {
                    k.clone()
                } else {
                    format!("{prefix}.{k}")
                };
                flatten_toml(&full, v, out);
            }
        }
        other => {
            let s = match other {
                toml::Value::String(s) => s.clone(),
                toml::Value::Integer(i) => i.to_string(),
                toml::Value::Float(f) => f.to_string(),
                toml::Value::Boolean(b) => b.to_string(),
                _ => other.to_string(),
            };
            out.insert(prefix.to_string(), Value::String(s));
        }
    }
}

fn read_config_flat(path: &std::path::Path) -> Result<Map<String, Value>, Error> {
    let table = load_config_toml(path)?;
    let mut out = Map::new();
    for (k, v) in &table {
        flatten_toml(k, v, &mut out);
    }
    Ok(out)
}

/// Set a dotted key in a TOML table, creating intermediate tables as needed.
fn set_dotted_key(table: &mut toml::Table, key: &str, value: String) -> Result<(), Error> {
    let parts: Vec<&str> = key.splitn(2, '.').collect();
    if parts.len() == 1 {
        table.insert(parts[0].to_string(), toml::Value::String(value));
        return Ok(());
    }
    let section = parts[0];
    let rest = parts[1];
    let entry = table
        .entry(section.to_string())
        .or_insert_with(|| toml::Value::Table(toml::Table::new()));
    if let toml::Value::Table(sub) = entry {
        set_dotted_key(sub, rest, value)
    } else {
        Err(Error::conflict(format!(
            "config key {section:?} is not a table"
        )))
    }
}

/// Remove a dotted key from a TOML table. Returns `true` when the key existed.
fn remove_dotted_key(table: &mut toml::Table, key: &str) -> bool {
    let parts: Vec<&str> = key.splitn(2, '.').collect();
    if parts.len() == 1 {
        return table.remove(parts[0]).is_some();
    }
    let section = parts[0];
    let rest = parts[1];
    if let Some(toml::Value::Table(sub)) = table.get_mut(section) {
        let removed = remove_dotted_key(sub, rest);
        if sub.is_empty() {
            table.remove(section);
        }
        removed
    } else {
        false
    }
}

#[cfg(test)]
mod gap01_tests {
    use super::*;
    use mnem_core::id::NodeId;
    use mnem_core::objects::Node;
    use mnem_core::retrieve::RetrievedItem;
    use proptest::prelude::*;

    fn fake_item(score: f32) -> RetrievedItem {
        // `Node::new` with no props is enough here; only `id` and
        // `rendered` are read downstream.
        let node = Node::new(NodeId::new_v7(), "Gap01Probe");
        RetrievedItem::new(node, "rendered preview".to_string(), 4, score)
    }

    #[test]
    fn confidence_zero_on_empty() {
        assert_eq!(gap01_compute_confidence(&[]), 0.0);
    }

    #[test]
    fn confidence_zero_on_singleton() {
        assert_eq!(gap01_compute_confidence(&[fake_item(1.0)]), 0.0);
    }

    #[test]
    fn confidence_high_when_tail_far_below_top() {
        let items = vec![fake_item(1.0), fake_item(0.9), fake_item(0.01)];
        let c = gap01_compute_confidence(&items);
        assert!(c > 0.9, "expected >0.9, got {c}");
    }

    #[test]
    fn confidence_low_when_flat() {
        let items = vec![fake_item(1.0), fake_item(0.99), fake_item(0.98)];
        let c = gap01_compute_confidence(&items);
        assert!(c < 0.1, "expected <0.1, got {c}");
    }

    #[test]
    fn suggested_neighbors_empty_below_top_seeds() {
        let items = vec![fake_item(1.0), fake_item(0.9), fake_item(0.8)];
        assert!(gap01_suggested_neighbors(&items).is_empty());
    }

    #[test]
    fn suggested_neighbors_skips_top_seeds() {
        let items = vec![
            fake_item(1.0),
            fake_item(0.9),
            fake_item(0.8),
            fake_item(0.7),
            fake_item(0.6),
        ];
        let n = gap01_suggested_neighbors(&items);
        assert_eq!(n.len(), 2);
        // `via` is always "adjacency".
        for entry in &n {
            assert_eq!(entry["via"], "adjacency");
        }
    }

    #[test]
    fn suggested_neighbors_bounded_by_max() {
        let items: Vec<_> = (0..100).map(|i| fake_item(1.0 - i as f32 * 0.01)).collect();
        let n = gap01_suggested_neighbors(&items);
        assert!(n.len() <= GAP01_MAX_NEIGHBOURS);
    }

    proptest! {
    /// Gap 01 proptest: the `suggested_neighbors` list is
    /// always a strict subset of the adjacency (ranked items)
    /// passed in. The proof is trivial by construction
    /// (`.iter().skip(GAP01_TOP_SEEDS).take(GAP01_MAX_NEIGHBOURS)`)
    /// but the property pins the invariant so that any future
    /// refactor which drifts into pulling IDs from a different
    /// source (e.g. a sibling lookup) has to rewrite this test.
    #[test]
    fn suggested_neighbors_always_subset_of_adjacency(
    scores in proptest::collection::vec(-1.0f32..1.0f32, 0..32),
    ) {
    let items: Vec<_> = scores.iter().map(|&s| fake_item(s)).collect();
    let neighbours = gap01_suggested_neighbors(&items);
    // Every `id` in the neighbour list must appear in the
    // original adjacency (ranked items).
    let ids: Vec<String> = items
    .iter()
    .map(|it| it.node.id.to_uuid_string())
    .collect();
    for entry in &neighbours {
    let nid = entry["id"].as_str().expect("id field");
    prop_assert!(
    ids.iter().any(|i| i == nid),
    "neighbour id {nid} not in adjacency"
    );
    }
    // And the cardinality is bounded.
    prop_assert!(neighbours.len() <= GAP01_MAX_NEIGHBOURS);
    }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn check(days: u64, expected_year: u64, expected_month: u8, expected_day: u8) {
        let (y, m, d) = days_to_ymd(days);
        assert_eq!(
            (y, m, d),
            (expected_year, expected_month, expected_day),
            "days_to_ymd({days}) = ({y},{m},{d}), want ({expected_year},{expected_month},{expected_day})"
        );
    }

    #[test]
    fn epoch() {
        check(0, 1970, 1, 1);
    }

    #[test]
    fn epoch_plus_one() {
        check(1, 1970, 1, 2);
    }

    #[test]
    fn start_of_february_1970() {
        check(31, 1970, 2, 1);
    }

    #[test]
    fn start_of_march_1970_non_leap() {
        check(59, 1970, 3, 1);
    }

    #[test]
    fn second_year() {
        check(365, 1971, 1, 1);
    }

    #[test]
    fn year_2000_leap_day() {
        check(11016, 2000, 2, 29);
    }

    #[test]
    fn year_2000_day_before_leap() {
        check(11015, 2000, 2, 28);
    }

    #[test]
    fn year_2000_day_after_leap() {
        check(11017, 2000, 3, 1);
    }

    #[test]
    fn year_2024_leap_day() {
        check(19782, 2024, 2, 29);
    }

    #[test]
    fn year_1972_leap_day() {
        check(789, 1972, 2, 29);
    }

    #[test]
    fn year_2024_day_before_leap() {
        check(19781, 2024, 2, 28);
    }

    #[test]
    fn year_2024_day_after_leap() {
        check(19783, 2024, 3, 1);
    }

    #[test]
    fn december_year_end_1970() {
        check(364, 1970, 12, 31);
    }

    #[test]
    fn year_2100_is_not_leap_feb_28() {
        check(47540, 2100, 2, 28);
    }

    #[test]
    fn year_2100_is_not_leap_next_day_is_march() {
        check(47541, 2100, 3, 1);
    }

    #[test]
    fn year_2100_start() {
        check(47482, 2100, 1, 1);
    }
}

#[cfg(test)]
mod ref_name_validation_tests {
    use super::is_valid_ref_name;

    #[test]
    fn valid_names_accepted() {
        assert!(is_valid_ref_name("refs/heads/main"));
        assert!(is_valid_ref_name("refs/heads/feature-x"));
        assert!(is_valid_ref_name("refs/tags/v1.0.0"));
        assert!(is_valid_ref_name("refs/remotes/origin/main"));
    }

    #[test]
    fn empty_and_whitespace_rejected() {
        assert!(!is_valid_ref_name(""));
        assert!(!is_valid_ref_name("   "));
    }

    #[test]
    fn literal_head_rejected() {
        assert!(!is_valid_ref_name("HEAD"));
        assert!(!is_valid_ref_name("head"));
        assert!(!is_valid_ref_name("Head"));
    }

    #[test]
    fn double_dot_rejected() {
        assert!(!is_valid_ref_name("refs/heads/a..b"));
        assert!(!is_valid_ref_name("..refs/heads/main"));
    }

    #[test]
    fn double_slash_rejected() {
        assert!(!is_valid_ref_name("refs//heads/main"));
    }

    #[test]
    fn backslash_rejected() {
        assert!(!is_valid_ref_name("refs\\heads\\main"));
    }

    #[test]
    fn at_brace_rejected() {
        assert!(!is_valid_ref_name("refs/heads/a@{0}"));
    }

    #[test]
    fn leading_dot_rejected() {
        assert!(!is_valid_ref_name(".hidden"));
    }

    #[test]
    fn trailing_dot_rejected() {
        assert!(!is_valid_ref_name("refs/heads/main."));
    }

    #[test]
    fn dot_lock_suffix_rejected() {
        assert!(!is_valid_ref_name("refs/heads/main.lock"));
    }

    #[test]
    fn control_chars_rejected() {
        assert!(!is_valid_ref_name("refs/heads/a\x01b"));
        assert!(!is_valid_ref_name("refs/heads/a\nb"));
        assert!(!is_valid_ref_name("refs/heads/a\tb"));
    }

    #[test]
    fn space_rejected() {
        assert!(!is_valid_ref_name("refs/heads/my branch"));
    }
}
