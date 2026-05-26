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
use mnem_core::codec::{from_canonical_bytes, json_to_ipld};
use mnem_core::id::{EdgeId, NodeId};
use mnem_core::index::PropPredicate;
use mnem_core::Query as NodeQuery;
use mnem_core::objects::{Commit, Edge, Node, Operation};
use mnem_core::retrieve::Lane;
use mnem_core::{HEADS_PREFIX, TAGS_PREFIX};
// BENCH-1 (C4): trait import is required so `MockEmbedder::embed`
// and `::model` resolve on the concrete struct in the cold-start
// fallback paths inside `retrieve` / `retrieve_full` below.
use mnem_embed_providers::Embedder as _;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};

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

pub(crate) async fn healthz() -> Json<Value> {
    Json(json!({
    "schema": "mnem.v1.healthz",
    "ok": true,
    "service": "mnem http",
    "version": env!("CARGO_PKG_VERSION"),
    }))
}

// ---------- GET /v1/stats ----------

pub(crate) async fn stats(State(s): State<AppState>) -> Result<Json<Value>, Error> {
    let repo = s.repo.lock().map_err(|_| Error::locked())?;
    let op_id = repo.op_id().to_string();
    let head = repo.view().heads.first().map(ToString::to_string);
    let refs = repo.view().refs.len();
    Ok(Json(json!({
    "schema": "mnem.v1.stats",
    "op_id": op_id,
    "head_commit": head,
    "refs": refs,
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
}

#[derive(Serialize)]
pub(crate) struct PostNodeResp {
    schema: &'static str,
    id: String,
    label: String,
    op_id: String,
}

pub(crate) async fn post_node(
    State(s): State<AppState>,
    Json(body): Json<PostNodeBody>,
) -> Result<Json<PostNodeResp>, Error> {
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
    let author = body
        .author
        .as_deref()
        .map(str::trim)
        .filter(|a| !a.is_empty())
        .map(str::to_string);
    let author = match author {
        Some(a) => a,
        None => return Err(Error::bad_request("author is required")),
    };

    let node_id = match body.id.as_deref() {
        Some(s) => NodeId::parse_uuid(s)
            .map_err(|e| Error::bad_request(format!("invalid caller-supplied id: {e}")))?,
        None => NodeId::new_v7(),
    };
    let mut node = Node::new(node_id, &label);
    if let Some(sum) = &body.summary {
        node = node.with_summary(sum);
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

    let id = node.id;

    let mut guard = s.repo.lock().map_err(|_| Error::locked())?;
    let mut tx = guard.start_transaction();
    let cid = tx.add_node(&node)?;
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

    Ok(Json(PostNodeResp {
        schema: "mnem.v1.post-node",
        id: id.to_uuid_string(),
        label: body.label,
        op_id,
    }))
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
    "props": Value::Object(props_map),
    "content_bytes": node.content.as_ref().map_or(0, bytes::Bytes::len),
    "has_embedding": has_embedding,
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
/// Returns `200 OK` with the embedding in the `mnem.v1.node_embedding`
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
    let vector: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();

    let dtype_str = match emb.dtype {
        mnem_core::objects::Dtype::F32 => "f32",
        mnem_core::objects::Dtype::F16 => "f16",
        mnem_core::objects::Dtype::F64 => "f64",
        mnem_core::objects::Dtype::I8 => "i8",
    };

    Ok(Json(json!({
        "schema": "mnem.v1.node_embedding",
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
    if q.author.trim().is_empty() {
        return Err(Error::bad_request("author is required"));
    }

    let mut guard = s.repo.lock().map_err(|_| Error::locked())?;
    let existed = guard.lookup_node(&id)?.is_some();
    if !existed {
        return Err(Error::not_found(format!(
            "no node with id={id_str} in current view"
        )));
    }
    let mut tx = guard.start_transaction();
    tx.remove_node(id);
    let commit_start = std::time::Instant::now();
    let new_repo = tx.commit(
        &q.author,
        q.message.as_deref().unwrap_or("mnem http delete node"),
    )?;
    s.metrics
        .commit_duration
        .observe(commit_start.elapsed().as_secs_f64());
    let op_id = new_repo.op_id().to_string();
    *guard = new_repo;

    Ok(Json(json!({
    "schema": "mnem.v1.delete-node",
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
    /// Commit author.
    pub author: String,
}

pub(crate) async fn tombstone_node(
    State(s): State<AppState>,
    Path(id_str): Path<String>,
    Json(body): Json<TombstoneBody>,
) -> Result<Json<Value>, Error> {
    let id = NodeId::parse_uuid(&id_str)
        .map_err(|e| Error::bad_request(format!("invalid UUID: {e}")))?;
    if body.author.trim().is_empty() {
        return Err(Error::bad_request("author is required"));
    }
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
    let mut tx = guard.start_transaction();
    tx.tombstone_node(id, body.reason.clone())?;
    let commit_start = std::time::Instant::now();
    let new_repo = tx.commit(&body.author, "mnem http tombstone node")?;
    s.metrics
        .commit_duration
        .observe(commit_start.elapsed().as_secs_f64());
    let op_id = new_repo.op_id().to_string();
    *guard = new_repo;

    Ok(Json(json!({
    "schema": "mnem.v1.tombstone",
    "op_id": op_id,
    "node_id": id_str,
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
/// Returns `{"schema":"mnem.v1.post-edge","edge_id":"<uuid>","op_id":"<cid>"}`.
/// Returns 404 if `src` or `dst` does not exist in the current view.
/// Returns 400 for malformed UUIDs, empty `etype`, or missing `author`.
pub(crate) async fn post_edge(
    _auth: RequireBearer,
    State(s): State<AppState>,
    Json(body): Json<PostEdgeBody>,
) -> Result<Json<Value>, Error> {
    // Validate required fields.
    let author = body.author.trim();
    if author.is_empty() {
        return Err(Error::bad_request("author is required"));
    }
    if body.etype.trim().is_empty() {
        return Err(Error::bad_request("etype is required"));
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

    let edge_id = EdgeId::new_v7();
    let mut edge = Edge::new(edge_id, &body.etype, src, dst);
    if let Some(props) = body.props {
        for (k, v) in props {
            edge = edge.with_prop(
                k,
                json_to_ipld(&v).map_err(|e| Error::bad_request(e.to_string()))?,
            );
        }
    }

    let mut tx = guard.start_transaction();
    tx.add_edge(&edge)?;
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
        "schema": "mnem.v1.post-edge",
        "edge_id": edge_id.to_uuid_string(),
        "op_id": op_id,
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
    label: String,
}

pub(crate) async fn post_nodes_bulk(
    State(s): State<AppState>,
    Json(body): Json<BulkNodeBody>,
) -> Result<Json<BulkNodeResp>, Error> {
    if body.author.trim().is_empty() {
        return Err(Error::bad_request("author is required"));
    }
    if body.nodes.is_empty() {
        return Err(Error::bad_request("nodes must not be empty"));
    }

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
        results.push(BulkNodeEntry {
            id: node.id.to_uuid_string(),
            label: nb.label,
        });
        built.push((node, pending_dense, pending_sparse_item));
    }

    // Single commit over all nodes. Index rebuild happens once.
    let mut guard = s.repo.lock().map_err(|_| Error::locked())?;
    let mut tx = guard.start_transaction();
    for (node, pending_dense, pending_sparse_item) in &built {
        let cid = tx.add_node(node)?;
        if let Some((model, emb)) = pending_dense {
            tx.set_embedding(cid.clone(), model.clone(), emb.clone())?;
        }
        if let Some((vocab_id, se)) = pending_sparse_item {
            tx.set_sparse_embedding(cid, vocab_id.clone(), se.clone())?;
        }
    }
    let commit_start = std::time::Instant::now();
    let new_repo = tx.commit(
        &body.author,
        body.message.as_deref().unwrap_or("mnem http bulk add"),
    )?;
    s.metrics
        .commit_duration
        .observe(commit_start.elapsed().as_secs_f64());
    let op_id = new_repo.op_id().to_string();
    *guard = new_repo;

    Ok(Json(BulkNodeResp {
        schema: "mnem.v1.post-nodes-bulk",
        op_id,
        results,
        embedded,
        skipped_embed,
    }))
}

// ---------- GET /v1/retrieve ----------

#[derive(Deserialize)]
pub(crate) struct RetrieveQuery {
    pub text: Option<String>,
    pub label: Option<String>,
    #[serde(default)]
    pub budget: Option<u32>,
    #[serde(default)]
    pub limit: Option<usize>,
    /// `KEY=VALUE`; VALUE tried as JSON first, falls back to string.
    pub where_eq: Option<String>,
}

pub(crate) async fn retrieve(
    State(s): State<AppState>,
    Query(q): Query<RetrieveQuery>,
) -> Result<Json<Value>, Error> {
    // Clamp untrusted numeric knobs before we touch the retriever.
    // See the `MAX_RETRIEVE_LIMIT` / `MAX_VECTOR_CAP` / `MAX_RERANK_TOP_K`
    // constants at the top of this file for rationale.
    clamp_or_reject("limit", q.limit, MAX_RETRIEVE_LIMIT)?;

    let repo = s.repo.lock().map_err(|_| Error::locked())?;
    let mut ret = repo.retrieve();
    // Honour the caller's label filter only when the server was
    // launched with `MNEM_BENCH=1`. Otherwise the label field is
    // simply ignored; the retrieve runs unscoped. See the
    // `post_node` doc-comment for the full rationale.
    if s.allow_labels
        && let Some(l) = &q.label
    {
        ret = ret.label(l.clone());
    }
    if let Some(w) = &q.where_eq {
        let (k, v) = parse_kv(w).map_err(Error::bad_request)?;
        ret = ret.where_prop(k, PropPredicate::Eq(v));
    }
    if let Some(b) = q.budget {
        ret = ret.token_budget(b);
    }
    if let Some(n) = q.limit {
        ret = ret.limit(n);
    }
    // Auto-encode the text query through every configured lane
    // (dense + sparse). there is no in-process lexical
    // ranker left; a text query with no embedder AND no sparse
    // provider configured is rejected with 400.
    let mut vector_model: Option<String> = None;
    let mut sparse_vocab: Option<String> = None;
    if let Some(text) = q.text.as_deref()
        && !text.trim().is_empty()
    {
        ret = ret.query_text(text.to_string());
        // Dense lane.
        if let Some(pc) = &s.embed_cfg {
            let embedder = mnem_embed_providers::open(pc)
                .map_err(|e| Error::internal(format!("embed provider open failed: {e}")))?;
            let qvec = embedder
                .embed(text)
                .map_err(|e| Error::internal(format!("embed call failed: {e}")))?;
            vector_model = Some(embedder.model().to_string());
            ret = ret.vector(embedder.model().to_string(), qvec);
        }
        // Sparse lane.
        if let Some(sc) = &s.sparse_cfg {
            let sparser = mnem_sparse_providers::open(sc)
                .map_err(|e| Error::bad_request(format!("sparse open failed: {e}")))?;
            let sq = sparser
                .encode_query(text)
                .map_err(|e| Error::bad_request(format!("sparse encode failed: {e}")))?;
            sparse_vocab = Some(sq.vocab_id.clone());
            ret = ret.sparse_query(sq);
        }
        // BENCH-1 (C4 audit): cold-start fallback. Cells launched on
        // a fresh `/data` volume have no `[embed]` / `[sparse]`
        // section in `config.toml`, so AppState resolves both to
        // `None`. Rather than 400 the caller (which breaks bench
        // harnesses that exercise retrieve before configuring a
        // provider), fall back to the deterministic, network-free
        // `MockEmbedder` (blake3-derived, dim=384). Real providers
        // still take priority when configured; this branch only
        // fires when both `embed_cfg` AND `sparse_cfg` are absent.
        if vector_model.is_none() && sparse_vocab.is_none() {
            let mock = mnem_embed_providers::MockEmbedder::new("mock:cold-start-384", 384);
            let qvec = mock
                .embed(text)
                .map_err(|e| Error::internal(format!("mock embed failed: {e}")))?;
            vector_model = Some(mock.model().to_string());
            ret = ret.vector(mock.model().to_string(), qvec);
            tracing::warn!(
                "retrieve: no [embed]/[sparse] configured; using deterministic \
 MockEmbedder fallback (cold-start). Configure a real provider \
 in config.toml for production retrieval quality."
            );
        }
    }
    {
        let mut cache = s.indexes.lock().map_err(|_| Error::locked())?;
        if let Some(model) = &vector_model {
            let idx = cache.vector_index(&repo, model)?;
            ret = ret.with_vector_index(idx);
        }
        if let Some(vocab) = &sparse_vocab {
            let idx = cache.sparse_index(&repo, vocab)?;
            ret = ret.with_sparse_index(idx);
        }
    }
    // Record retrieve-latency histogram around the actual fusion call.
    // This keeps the sample narrow (excludes JSON serialisation cost)
    // so operators see the cost of the retrieval pipeline itself.
    let retrieve_start = std::time::Instant::now();
    let result = ret.execute()?;
    s.metrics
        .retrieve_latency
        .observe(retrieve_start.elapsed().as_secs_f64());

    let items: Vec<Value> = result
        .items
        .iter()
        .map(|item| {
            // Per-lane observability: expose as a JSON object keyed by
            // lane name so API consumers can diagnose "why did this
            // node rank" without re-running the pipeline locally.
            let mut lane_obj = Map::new();
            for (lane, score) in &item.lane_scores {
                lane_obj.insert(lane_name(*lane).to_string(), json!(score));
            }
            json!({
            "id": item.node.id.to_uuid_string(),
            "label": item.node.ntype,
            "score": item.score,
            "tokens": item.tokens,
            "summary": item.node.summary,
            "rendered": item.rendered,
            "lane_scores": Value::Object(lane_obj),
            })
        })
        .collect();

    // Gap 16: score calibration - scale-free per-query interpretability.
    // `score_distribution` is a response-level block carrying
    // min / max / median / iqr + a categorical `shape` label
    // (long-tail / uniform / bimodal / insufficient-samples). The
    // shape is promoted to a top-level agent hint per the R2 spec:
    // agents consume it to decide whether top-1 is a confident match
    // or whether the dense ranking is inconclusive. Scale-free: works
    // identically for K=8 or K=1000.
    let score_dist = {
        let scores: Vec<f32> = result.items.iter().map(|it| it.score).collect();
        mnem_graphrag::distribution_shape(&scores, mnem_graphrag::K_MIN)
    };

    Ok(Json(json!({
    "schema": "mnem.v1.retrieve",
    "items": items,
    "tokens_used": result.tokens_used,
    "tokens_budget": if result.tokens_budget == u32::MAX {
    Value::Null
    } else {
    Value::from(result.tokens_budget)
    },
    "dropped": result.dropped,
    "candidates_seen": result.candidates_seen,
    "score_distribution": score_dist,
    })))
}

// ---------- GET /v1/query ----------
//
// Pure label/property filter scan. No embedder required.
// Mirrors `mnem query` in the CLI: uses the `Query` engine which
// dispatches to the Prolly index (O(log n) point lookup or label cursor)
// rather than the retrieval pipeline.
//
// At least one of `label` or `where_eq` must be supplied.
// `with_outgoing` accepts a comma-separated list of edge-type labels.
//
// Response: {"nodes": [...], "count": N}
// Each node: {"id", "label", "summary", "props", "edges", "incoming_edges"}

/// Query parameters for `GET /v1/query`.
#[derive(Deserialize)]
pub(crate) struct QueryParams {
    /// Filter by node label (ntype). Unlike `/v1/retrieve`, this is a
    /// pre-filter scan criterion backed by the Prolly label index — no
    /// embedder required.
    pub label: Option<String>,
    /// Property equality filter as `KEY=VALUE`. VALUE is tried as JSON
    /// first, then falls back to a raw string. When combined with `label`,
    /// uses the indexed `(label, prop) -> value` Prolly point lookup.
    pub where_eq: Option<String>,
    /// Comma-separated edge-type labels to include as outgoing edges on
    /// each result node (e.g. `with_outgoing=knows,works_at`).
    pub with_outgoing: Option<String>,
    /// Comma-separated edge-type labels to include as incoming edges on
    /// each result node.
    pub with_incoming: Option<String>,
    /// Max results to return. Capped at `MAX_RETRIEVE_LIMIT`.
    pub limit: Option<usize>,
}

/// `GET /v1/query` — pure label/property scan. No embedder required.
///
/// Mirrors `mnem query` in the CLI. At least one of `label` or `where_eq`
/// must be supplied. Results are unranked (Prolly tree order).
pub(crate) async fn query(
    State(s): State<AppState>,
    Query(q): Query<QueryParams>,
) -> Result<Json<Value>, Error> {
    if q.label.is_none() && q.where_eq.is_none() {
        return Err(Error::bad_request(
            "at least one of `label` or `where_eq` is required for /v1/query",
        ));
    }
    clamp_or_reject("limit", q.limit, MAX_RETRIEVE_LIMIT)?;

    let repo = s.repo.lock().map_err(|_| Error::locked())?;
    let mut qry = NodeQuery::new(&*repo);

    if let Some(l) = &q.label {
        qry = qry.label(l.clone());
    }
    if let Some(w) = &q.where_eq {
        let (k, v) = parse_kv(w).map_err(Error::bad_request)?;
        qry = qry.where_prop(k, PropPredicate::Eq(v));
    }
    if let Some(etypes) = &q.with_outgoing {
        for etype in etypes.split(',').map(str::trim).filter(|s| !s.is_empty()) {
            qry = qry.with_outgoing(etype);
        }
    }
    if let Some(etypes) = &q.with_incoming {
        for etype in etypes.split(',').map(str::trim).filter(|s| !s.is_empty()) {
            qry = qry.with_incoming(etype);
        }
    }
    qry = qry.limit(q.limit.unwrap_or(20));

    let hits = qry.execute()?;

    let nodes: Vec<Value> = hits
        .iter()
        .map(|h| {
            let mut props_map = Map::new();
            for (k, v) in &h.node.props {
                props_map.insert(k.clone(), ipld_to_json(v));
            }
            let edges: Vec<Value> = h
                .edges
                .iter()
                .map(|e| {
                    json!({
                        "etype": e.etype,
                        "src": e.src.to_uuid_string(),
                        "dst": e.dst.to_uuid_string(),
                    })
                })
                .collect();
            let incoming_edges: Vec<Value> = h
                .incoming_edges
                .iter()
                .map(|e| {
                    json!({
                        "etype": e.etype,
                        "src": e.src.to_uuid_string(),
                        "dst": e.dst.to_uuid_string(),
                    })
                })
                .collect();
            json!({
                "id": h.node.id.to_uuid_string(),
                "label": h.node.ntype,
                "summary": h.node.summary,
                "props": Value::Object(props_map),
                "edges": edges,
                "incoming_edges": incoming_edges,
                "edges_truncated": h.edges_truncated,
            })
        })
        .collect();

    Ok(Json(json!({
        "nodes": nodes,
        "count": nodes.len(),
    })))
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
    // name an embedder configured on the server)
    #[serde(default)]
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
}

pub(crate) async fn retrieve_full(
    State(s): State<AppState>,
    Json(body): Json<RetrieveRequest>,
) -> Result<Json<Value>, Error> {
    // Clamp untrusted numeric knobs before we touch the retriever.
    // See the `MAX_RETRIEVE_LIMIT` / `MAX_VECTOR_CAP` / `MAX_RERANK_TOP_K`
    // constants at the top of this file for rationale.
    clamp_or_reject("limit", body.limit, MAX_RETRIEVE_LIMIT)?;
    clamp_or_reject("vector_cap", body.vector_cap, MAX_VECTOR_CAP)?;
    clamp_or_reject("rerank_top_k", body.rerank_top_k, MAX_RERANK_TOP_K)?;

    let repo = s.repo.lock().map_err(|_| Error::locked())?;
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

    // Vector: caller-supplied embedding takes priority over
    // server-side auto-fuse. When no vector is supplied AND the
    // server has an embed provider configured, embed the text
    // query with it so the retrieve fires the real hybrid path.
    // This matches the CLI behaviour (commands.rs retrieve).
    //
    // Post-there is no text ranker left in mnem-core: a
    // `text` query without either (a) a caller-supplied vector or
    // (b) a configured embedder is rejected with 400.
    let mut vector_model: Option<String> = None;
    let mut sparse_vocab: Option<String> = None;
    if let Some(text) = body.text.as_deref()
        && !text.trim().is_empty()
    {
        ret = ret.query_text(text.to_string());
    }
    // Caller-supplied vector wins over auto-embed.
    if let (Some(m), Some(v)) = (&body.vector_model, &body.vector) {
        vector_model = Some(m.clone());
        ret = ret.vector(m.clone(), v.clone());
    } else if let Some(text) = body.text.as_deref()
        && !text.trim().is_empty()
        && let Some(pc) = &s.embed_cfg
    {
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
    // `encode_query` (OpenSearch v3-distill).
    if let Some(text) = body.text.as_deref()
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
    if body.text.as_deref().is_some_and(|t| !t.trim().is_empty())
        && vector_model.is_none()
        && sparse_vocab.is_none()
        && body.vector.is_none()
    {
        if let Some(text) = body.text.as_deref() {
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
    let items: Vec<Value> = result
        .items
        .iter()
        .map(|item| {
            // Per-lane observability: expose as a JSON object keyed by
            // lane name so API consumers can diagnose "why did this
            // node rank" without re-running the pipeline locally.
            let mut lane_obj = Map::new();
            for (lane, score) in &item.lane_scores {
                lane_obj.insert(lane_name(*lane).to_string(), json!(score));
            }
            json!({
            "id": item.node.id.to_uuid_string(),
            "label": item.node.ntype,
            "score": item.score,
            "tokens": item.tokens,
            "summary": item.node.summary,
            "rendered": item.rendered,
            "lane_scores": Value::Object(lane_obj),
            })
        })
        .collect();

    // Gap 16: score calibration - scale-free per-query interpretability.
    // Mirrors the GET /v1/retrieve handler above. The `score_distribution`
    // block carries min / max / median / iqr + a categorical `shape`
    // label (long-tail / uniform / bimodal / insufficient-samples) so
    // agents can interpret the dense ranking without a trained scaler.
    let score_dist = {
        let scores: Vec<f32> = result.items.iter().map(|it| it.score).collect();
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
    let gap01_confidence = gap01_compute_confidence(&result.items);
    let gap01_neighbors = gap01_suggested_neighbors(&result.items);
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
        let k = body.summarize_k.unwrap_or(3).min(MAX_RETRIEVE_LIMIT);
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
        for it in &result.items {
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
fn default_log_limit() -> usize {
    50
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
    /// Maximum number of entries to return (default 50, max 500).
    #[serde(default = "default_log_limit")]
    pub limit: usize,
    /// Output format: `json` (default), `oneline`, or `full`.
    #[serde(default)]
    pub format: LogFormat,
}

/// One entry in the JSON log response.
#[derive(serde::Serialize)]
struct LogEntry {
    op_id: String,
    timestamp: String,
    author: String,
    message: String,
    parents: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    agent_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    task_id: Option<String>,
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
/// - `limit`: max entries to return (default 50, max 500)
/// - `format`: `json` (default) | `oneline` | `full`
///
/// JSON response: `{ "schema": "mnem.v1.log", "entries": [...], "count": N }`
/// Text responses (oneline/full): `text/plain; charset=utf-8`
pub(crate) async fn get_log(
    State(s): State<AppState>,
    Query(params): Query<LogParams>,
) -> Result<impl IntoResponse, Error> {
    // Clamp limit at the hard cap so callers cannot request unbounded work.
    let limit = params.limit.min(MAX_LOG_LIMIT);
    if limit == 0 {
        return Err(Error::bad_request("limit must be >= 1"));
    }

    let repo = s.repo.lock().map_err(|_| Error::locked())?;
    let bs = repo.blockstore().clone();
    let mut cur = repo.op_id().clone();

    match params.format {
        LogFormat::Json => {
            let mut entries: Vec<LogEntry> = Vec::with_capacity(limit);
            for _ in 0..limit {
                let (op, next) = read_op(bs.as_ref(), &cur)?;
                entries.push(LogEntry {
                    op_id: cur.to_string(),
                    timestamp: micros_to_rfc3339(op.time),
                    author: op.author.clone(),
                    message: op.description.clone(),
                    parents: op.parents.iter().map(ToString::to_string).collect(),
                    agent_id: op.agent_id.clone(),
                    task_id: op.task_id.clone(),
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
                lines.push_str(&format!("{short} {}\n", op.description));
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
                text.push_str(&format!("op {op_id_str}\n"));
                text.push_str(&format!("   time    {}us\n", op.time));
                if !op.author.is_empty() {
                    text.push_str(&format!("   author  {}\n", op.author));
                }
                if let Some(agent) = &op.agent_id {
                    text.push_str(&format!("   agent   {agent}\n"));
                }
                if let Some(task) = &op.task_id {
                    text.push_str(&format!("   task    {task}\n"));
                }
                text.push_str(&format!("   message {}\n", op.description));
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
    let limit = params.limit.unwrap_or(MAX_EXPORT_OPS).min(MAX_EXPORT_OPS);

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
        match bs.put(claimed_cid, data) {
            Ok(()) => imported += 1,
            Err(e) => {
                errors.push(json!({
                    "line": line_no + 1,
                    "cid": cid_str,
                    "error": format!("blockstore write: {e}"),
                }));
            }
        }
    }

    let ok = errors.is_empty();
    Ok(Json(json!({
        "schema": "mnem.v1.import",
        "imported": imported,
        "errors": errors,
        "ok": ok,
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
    let current_head = view.heads.first().cloned();

    let branches: Vec<Value> = view
        .refs
        .iter()
        .filter_map(|(name, target)| {
            let short = name.strip_prefix(HEADS_PREFIX)?;
            let (head_str, is_current) = match target {
                mnem_core::objects::RefTarget::Normal { target } => {
                    let is_cur = Some(target) == current_head.as_ref();
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
    /// Optional commit CID to point the new branch at. When absent,
    /// defaults to the current head commit.
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
    if body.author.trim().is_empty() {
        return Err(Error::bad_request("author is required"));
    }
    // Basic refname sanity: reject characters that break VCS tooling.
    let n = &body.name;
    if n.contains(' ')
        || n.contains('\t')
        || n.contains('\n')
        || n.contains('\x00')
        || n.contains('~')
        || n.contains('^')
        || n.contains(':')
        || n.contains('?')
        || n.contains('*')
        || n.contains('[')
        || n.contains('\\')
        || n.contains("@{")
        || n.contains("..")
        || n.contains("//")
        || n.starts_with('/')
        || n.ends_with('/')
        || n.ends_with('.')
        || n.ends_with(".lock")
    {
        return Err(Error::bad_request(format!(
            "invalid branch name `{n}`: may not contain spaces, control characters, \
             `~`, `^`, `:`, `?`, `*`, `[`, `\\`, `@{{`, `..`, `//`, \
             or start/end with `/`, or end with `.` or `.lock`"
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

    // Resolve the target commit CID.
    let target_cid = match body.at.as_deref() {
        Some(cid_str) => {
            let cid = mnem_core::id::Cid::parse_str(cid_str)
                .map_err(|e| Error::bad_request(format!("invalid CID `{cid_str}`: {e}")))?;
            // Verify the CID decodes as a Commit block.
            let bs = guard.blockstore().clone();
            let bytes = bs
                .get(&cid)
                .map_err(|e| Error::internal(format!("blockstore error: {e}")))?
                .ok_or_else(|| {
                    Error::not_found(format!("block {cid_str} not found in blockstore"))
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
                "repository has no commits yet; pass `at` with a commit CID".to_string(),
            )
        })?,
    };

    let head_str = target_cid.to_string();
    let new_repo = guard
        .update_ref(
            &full,
            None,
            Some(mnem_core::objects::RefTarget::normal(target_cid)),
            &body.author,
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
    State(s): State<AppState>,
    Path(name): Path<String>,
    Query(q): Query<DeleteQuery>,
) -> Result<Json<Value>, Error> {
    if name.trim().is_empty() {
        return Err(Error::bad_request("branch name must not be empty"));
    }
    if q.author.trim().is_empty() {
        return Err(Error::bad_request("author is required"));
    }

    let full = format!("{HEADS_PREFIX}{name}");

    let mut guard = s.repo.lock().map_err(|_| Error::locked())?;
    let view = guard.view();

    let prev = view
        .refs
        .get(&full)
        .cloned()
        .ok_or_else(|| Error::not_found(format!("branch `{name}` does not exist")))?;

    // Refuse to delete the branch that currently points at HEAD.
    let current_head = view.heads.first().cloned();
    if let mnem_core::objects::RefTarget::Normal { target } = &prev {
        if Some(target) == current_head.as_ref() {
            return Err(Error::conflict(format!(
                "cannot delete branch `{name}`: it is the current branch (points at HEAD)"
            )));
        }
    }

    let new_repo = guard
        .update_ref(&full, Some(&prev), None, &q.author)
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
    if body.author.trim().is_empty() {
        return Err(Error::bad_request("author is required"));
    }
    // Reuse the same refname validation as branches.
    let n = &body.name;
    if n.contains(' ')
        || n.contains('\t')
        || n.contains('\n')
        || n.contains('\x00')
        || n.contains('~')
        || n.contains('^')
        || n.contains(':')
        || n.contains('?')
        || n.contains('*')
        || n.contains('[')
        || n.contains('\\')
        || n.contains("@{")
        || n.contains("..")
        || n.contains("//")
        || n.starts_with('/')
        || n.ends_with('/')
        || n.ends_with('.')
        || n.ends_with(".lock")
    {
        return Err(Error::bad_request(format!(
            "invalid tag name `{n}`: may not contain spaces, control characters, \
             `~`, `^`, `:`, `?`, `*`, `[`, `\\`, `@{{`, `..`, `//`, \
             or start/end with `/`, or end with `.` or `.lock`"
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
    let new_repo = guard
        .update_ref(
            &full,
            None,
            Some(mnem_core::objects::RefTarget::normal(target_cid)),
            &body.author,
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
    State(s): State<AppState>,
    Path(name): Path<String>,
    Query(q): Query<DeleteQuery>,
) -> Result<Json<Value>, Error> {
    if name.trim().is_empty() {
        return Err(Error::bad_request("tag name must not be empty"));
    }
    if q.author.trim().is_empty() {
        return Err(Error::bad_request("author is required"));
    }

    let full = format!("{TAGS_PREFIX}{name}");

    let mut guard = s.repo.lock().map_err(|_| Error::locked())?;
    let view = guard.view();

    let prev = view
        .refs
        .get(&full)
        .cloned()
        .ok_or_else(|| Error::not_found(format!("tag `{name}` does not exist")))?;

    let new_repo = guard
        .update_ref(&full, Some(&prev), None, &q.author)
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

/// Request body for `POST /v1/diff`.
#[derive(Deserialize)]
pub(crate) struct DiffBody {
    /// CID of the "from" side: either a commit CID or an op CID.
    pub from: String,
    /// CID of the "to" side: either a commit CID or an op CID.
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
    // Clamp limit.
    let limit = params
        .limit
        .unwrap_or(DIFF_DEFAULT_LIMIT)
        .min(DIFF_MAX_LIMIT);
    if limit == 0 {
        return Err(Error::bad_request("limit must be >= 1"));
    }

    let repo = s.repo.lock().map_err(|_| Error::locked())?;
    let bs = repo.blockstore().clone();
    drop(repo); // release lock before potentially slow diff walks

    let (from_cid, from_commit) = resolve_cid_to_commit(bs.as_ref(), &body.from)?;
    let (to_cid, to_commit) = resolve_cid_to_commit(bs.as_ref(), &body.to)?;

    // Diff node trees.
    let node_changes = mnem_core::prolly::diff(bs.as_ref(), &from_commit.nodes, &to_commit.nodes)
        .map_err(|e| Error::internal(format!("node diff failed: {e}")))?;

    // Diff edge trees.
    let edge_changes = mnem_core::prolly::diff(bs.as_ref(), &from_commit.edges, &to_commit.edges)
        .map_err(|e| Error::internal(format!("edge diff failed: {e}")))?;

    // Build node response buckets.
    let mut nodes_added: Vec<Value> = Vec::new();
    let mut nodes_removed: Vec<Value> = Vec::new();
    let mut nodes_changed: Vec<Value> = Vec::new();

    for entry in &node_changes {
        match entry {
            mnem_core::prolly::DiffEntry::Added { value, .. } => {
                if nodes_added.len() < limit {
                    if let Some(node) = node_from_bs(bs.as_ref(), value) {
                        nodes_added.push(json!({
                            "id": node.id.to_uuid_string(),
                            "ntype": node.ntype,
                            "summary": node.summary,
                        }));
                    }
                }
            }
            mnem_core::prolly::DiffEntry::Removed { value, .. } => {
                if nodes_removed.len() < limit {
                    if let Some(node) = node_from_bs(bs.as_ref(), value) {
                        nodes_removed.push(json!({
                            "id": node.id.to_uuid_string(),
                            "ntype": node.ntype,
                            "summary": node.summary,
                        }));
                    }
                }
            }
            mnem_core::prolly::DiffEntry::Changed { before, after, .. } => {
                if nodes_changed.len() < limit {
                    if let Some(after_node) = node_from_bs(bs.as_ref(), after) {
                        let before_val = node_from_bs(bs.as_ref(), before).map(|n| {
                            json!({
                                "id": n.id.to_uuid_string(),
                                "ntype": n.ntype,
                                "summary": n.summary,
                            })
                        });
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

    // Build edge response buckets.
    let mut edges_added: Vec<Value> = Vec::new();
    let mut edges_removed: Vec<Value> = Vec::new();

    for entry in &edge_changes {
        match entry {
            mnem_core::prolly::DiffEntry::Added { value, .. } => {
                if edges_added.len() < limit {
                    if let Some(edge) = edge_from_bs(bs.as_ref(), value) {
                        edges_added.push(json!({
                            "id": edge.id.to_uuid_string(),
                            "etype": edge.etype,
                            "src": edge.src.to_uuid_string(),
                            "dst": edge.dst.to_uuid_string(),
                        }));
                    }
                }
            }
            mnem_core::prolly::DiffEntry::Removed { value, .. } => {
                if edges_removed.len() < limit {
                    if let Some(edge) = edge_from_bs(bs.as_ref(), value) {
                        edges_removed.push(json!({
                            "id": edge.id.to_uuid_string(),
                            "etype": edge.etype,
                            "src": edge.src.to_uuid_string(),
                            "dst": edge.dst.to_uuid_string(),
                        }));
                    }
                }
            }
            mnem_core::prolly::DiffEntry::Changed { .. } => {
                // Edges rarely change in place (etype/src/dst are immutable
                // by convention); if they do, we emit an empty changed array
                // as the spec requires but do not expand the entries.
            }
        }
    }

    Ok(Json(json!({
        "schema": "mnem.v1.diff",
        "from": from_cid.to_string(),
        "to": to_cid.to_string(),
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
    })))
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
/// - On CBOR decode failure (json format): 200 with `"data": null, "error": "..."`
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
                Err(e) => Ok(Json(json!({
                    "schema": "mnem.v1.block",
                    "cid": cid.to_string(),
                    "format": "json",
                    "data": Value::Null,
                    "error": format!("decode failed: {e}"),
                }))
                .into_response()),
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
    /// CID of the left (current-branch) commit.
    pub left: String,
    /// CID of the right (incoming-branch) commit.
    pub right: String,
    /// Conflict-resolution strategy. Defaults to `"manual"`.
    #[serde(default)]
    pub strategy: MergeStrategyParam,
}

/// `POST /v1/merge` - 3-way merge two commit CIDs.
///
/// Body: `{"left": "<cid>", "right": "<cid>", "strategy": "manual"|"ours"|"theirs"}`
///
/// `strategy` defaults to `"manual"` when omitted.
///
/// Response (HTTP 200 for all outcomes):
/// - `{"status": "fast_forward", "commit": "<cid>"}`
/// - `{"status": "clean", "commit": "<cid>"}`
/// - `{"status": "conflicts", "conflicts": <MergeConflicts>}`
pub(crate) async fn post_merge(
    State(s): State<AppState>,
    Json(body): Json<MergeBody>,
) -> Result<Json<Value>, Error> {
    use mnem_core::repo::merge::{MergeOutcome, MergeStrategy, merge_three_way};
    use mnem_core::store::MemoryOpHeadsStore;

    let repo = s.repo.lock().map_err(|_| Error::locked())?;
    let bs = repo.blockstore().clone();
    drop(repo); // release lock before slow merge walks

    // Reject same-CID early: calling merge_three_way with left==right
    // returns FastForward but is almost certainly a caller mistake.
    if body.left == body.right {
        return Err(Error::bad_request(
            "left and right must be different commit CIDs",
        ));
    }

    let (left_cid, _) = resolve_cid_to_commit(bs.as_ref(), &body.left)?;
    let (right_cid, _) = resolve_cid_to_commit(bs.as_ref(), &body.right)?;

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

    let response = match outcome {
        MergeOutcome::FastForward(cid) => json!({
            "status": "fast_forward",
            "commit": cid.to_string(),
        }),
        MergeOutcome::Clean(cid) => json!({
            "status": "clean",
            "commit": cid.to_string(),
        }),
        MergeOutcome::Conflicts(conflicts) => json!({
            "status": "conflicts",
            "conflicts": conflicts,
        }),
    };

    Ok(Json(response))
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
