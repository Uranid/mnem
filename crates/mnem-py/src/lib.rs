//! Python bindings for mnem.
//!
//! Exposes the Phase-2 retrieval surface and the minimal write path to
//! Python via pyo3, packaged as an importable `pymnem` module.
//!
//! The wrapper is intentionally small: it wraps `Repo`, a `transaction()`
//! context manager, `retrieve()` with every builder knob surfaced as
//! keyword arguments, a structured `query()` for exact lookups, and the
//! `RetrievalResult` / `RetrievedItem` result types. Signing, CAS on
//! refs, diff, and the op-log walk are deferred - they add wire surface
//! that most Python callers never touch on day one.
//!
//! Build via `maturin develop` (dev) or `maturin build --release` (wheel).
//! See `crates/mnem-py/README.md` for the full Python-side usage.

#![forbid(unsafe_code)]
#![deny(missing_docs)]
#![allow(clippy::needless_pass_by_value)] // pyo3 consumes most Py-side values

use std::sync::{Arc, Mutex};

use bytes::Bytes;
use ipld_core::ipld::Ipld;
use mnem_core::Error as CoreError;
use mnem_core::error::{CodecError, ObjectError, RepoError};
use mnem_core::id::NodeId;
use mnem_core::index::PropPredicate;
use mnem_core::objects::{Dtype, Embedding, Node as CoreNode};
use mnem_core::repo::ReadonlyRepo;
use mnem_core::store::{Blockstore, MemoryBlockstore, MemoryOpHeadsStore, OpHeadsStore};
use pyo3::create_exception;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyBool, PyDict};

// ============================================================
// MNEM_BENCH gate
// ============================================================

/// Whether caller-supplied `ntype` on `Transaction.add_node` and `label`
/// kwargs on `Repo.retrieve` / `Repo.query` are honoured.
///
/// Gated behind the `MNEM_BENCH` environment variable, parity with
/// `mnem-http`'s `AppState::allow_labels` and `mnem-mcp`'s
/// `Server::allow_labels`. The gate is re-read on every access (not
/// cached in a `OnceLock`) so Python tests that `os.environ[...] = ...`
/// between calls see the new value without needing a subprocess. The
/// cost is a single `std::env::var` lookup per call, which is cheap
/// next to the downstream tree walk.
///
/// Defaults to `false`: a casual `pymnem` user who passes
/// `tx.add_node(ntype="Person")` has the `ntype` silently coerced to
/// `Node::DEFAULT_NTYPE` ("Node"), and a caller passing `label=...` to
/// `retrieve` / `query` has the filter silently dropped. Benchmark
/// harnesses opt in by exporting `MNEM_BENCH=1` before launching the
/// Python process.
fn allow_labels() -> bool {
    parse_allow_labels(std::env::var("MNEM_BENCH").ok().as_deref())
}

/// Pure parser for the `MNEM_BENCH` value. `None` (unset) is false.
/// Falsy strings (`"0"`, `"false"`, `"no"`, `"off"`, empty, all
/// case-insensitive) are false. Anything else is true.
///
/// Duplicated from `mnem-http`'s `AppState::parse_allow_labels` rather
/// than depended upon: `mnem-py` has no reason to pull in the axum tree.
fn parse_allow_labels(val: Option<&str>) -> bool {
    match val {
        None => false,
        Some(s) => {
            let t = s.trim();
            if t.is_empty() {
                return false;
            }
            let l = t.to_ascii_lowercase();
            !matches!(l.as_str(), "0" | "false" | "no" | "off")
        }
    }
}

// ============================================================
// Exception type
// ============================================================

create_exception!(
    pymnem,
    MnemError,
    PyRuntimeError,
    "Base class for all mnem-originated exceptions raised from Python."
);

fn map_err<E: std::fmt::Display>(e: E) -> PyErr {
    MnemError::new_err(e.to_string())
}

/// Map a `mnem_core::Error` to the most appropriate Python exception.
///
/// The goal is to let Python callers `except ValueError:` on
/// shape/validation errors, `except KeyError:` on "query found nothing
/// where an exact match was required", and fall through to the generic
/// `MnemError` for store / signature / repo-lifecycle errors that do
/// not map cleanly to a stdlib exception.
fn map_core_err(e: CoreError) -> PyErr {
    use pyo3::exceptions::PyKeyError;
    let msg = e.to_string();
    match &e {
        // Input / schema errors -> ValueError.
        CoreError::Id(_)
        | CoreError::Codec(
            CodecError::Encode(_) | CodecError::Decode(_) | CodecError::NonCanonical(_),
        )
        | CoreError::Object(
            ObjectError::WrongKind { .. } | ObjectError::EmbeddingSizeMismatch { .. },
        )
        | CoreError::Repo(
            RepoError::VectorDimMismatch { .. }
            | RepoError::RetrievalEmpty
            | RepoError::AmbiguousMatch,
        ) => PyValueError::new_err(msg),
        // Exact-match miss -> KeyError (matches Python dict/lookup idioms).
        CoreError::Repo(RepoError::NotFound) => PyKeyError::new_err(msg),
        // Everything else (Store, Sign, Uninitialized, Stale, IndexCorrupt,
        // NoCommonAncestor) -> opaque mnem error.
        _ => MnemError::new_err(msg),
    }
}

// ============================================================
// Repo - the main entry point
// ============================================================

/// A mnem repository, pinned at the current head operation.
///
/// Internally wraps `mnem_core::repo::ReadonlyRepo` inside an
/// `Arc<Mutex<_>>` so that Python-side mutation (commit) can swap in
/// the new head view in place. All methods are thread-safe; the mutex
/// only contends under simultaneous writes, which is rare in typical
/// single-agent Python use.
#[pyclass(module = "pymnem")]
pub struct Repo {
    inner: Arc<Mutex<ReadonlyRepo>>,
}

#[pymethods]
impl Repo {
    /// Initialise a fresh in-memory repository. Useful for tests,
    /// notebooks, and agent sessions that don't need persistence.
    #[staticmethod]
    pub fn init_memory() -> PyResult<Self> {
        let bs: Arc<dyn Blockstore> = Arc::new(MemoryBlockstore::new());
        let ohs: Arc<dyn OpHeadsStore> = Arc::new(MemoryOpHeadsStore::new());
        let repo = ReadonlyRepo::init(bs, ohs).map_err(map_core_err)?;
        Ok(Self {
            inner: Arc::new(Mutex::new(repo)),
        })
    }

    /// Open (or initialise) a redb-backed repository at `path`.
    ///
    /// If the file does not exist it is created and an empty repo is
    /// initialised. If it exists, the current op-head is loaded. Any
    /// error that is NOT "op-heads empty" (e.g. store corruption,
    /// broken op-DAG, codec error on a head object) propagates rather
    /// than silently reinitialising.
    #[staticmethod]
    pub fn open_or_init(path: &str) -> PyResult<Self> {
        let (bs, ohs, _db) = mnem_backend_redb::open_or_init(path).map_err(map_core_err)?;
        let bs_arc: Arc<dyn Blockstore> = bs;
        let ohs_arc: Arc<dyn OpHeadsStore> = ohs;
        let repo = match ReadonlyRepo::open(bs_arc.clone(), ohs_arc.clone()) {
            Ok(r) => r,
            Err(e) if e.is_uninitialized() => {
                ReadonlyRepo::init(bs_arc, ohs_arc).map_err(map_core_err)?
            }
            Err(e) => return Err(map_err(e)),
        };
        Ok(Self {
            inner: Arc::new(Mutex::new(repo)),
        })
    }

    /// Current operation ID as a CID string. Advances on every commit.
    pub fn op_id(&self) -> PyResult<String> {
        let g = self.inner.lock().map_err(poison)?;
        Ok(g.op_id().to_string())
    }

    /// CID string of the current head commit, or `None` for a
    /// freshly-initialised repo with no commits yet.
    pub fn head_commit_cid(&self) -> PyResult<Option<String>> {
        let g = self.inner.lock().map_err(poison)?;
        Ok(g.view().heads.first().map(ToString::to_string))
    }

    /// Commit a single node in one round trip. Returns the new node's
    /// UUID as a string. For multiple-node commits, use a transaction.
    ///
    /// Accepts summary, props (scalar values only - str, int, float,
    /// bool), and optional content bytes.
    #[pyo3(signature = (author, message, ntype, summary = None, props = None, content = None))]
    pub fn commit_node(
        &self,
        author: &str,
        message: &str,
        ntype: &str,
        summary: Option<&str>,
        props: Option<&Bound<'_, PyDict>>,
        content: Option<&[u8]>,
    ) -> PyResult<String> {
        // Gate `ntype` behind `MNEM_BENCH` (see `Transaction.add_node`).
        let ntype = if allow_labels() {
            ntype
        } else {
            CoreNode::DEFAULT_NTYPE
        };
        let mut node = CoreNode::new(NodeId::new_v7(), ntype);
        if let Some(s) = summary {
            node = node.with_summary(s);
        }
        if let Some(dict) = props {
            for (k, v) in dict {
                let key: String = k.extract()?;
                node = node.with_prop(key, py_to_ipld(&v)?);
            }
        }
        if let Some(c) = content {
            node = node.with_content(Bytes::copy_from_slice(c));
        }
        let node_id = node.id;

        let mut guard = self.inner.lock().map_err(poison)?;
        let mut tx = guard.start_transaction();
        tx.add_node(&node).map_err(map_core_err)?;
        let new_repo = tx.commit(author, message).map_err(map_core_err)?;
        *guard = new_repo;
        Ok(node_id.to_uuid_string())
    }

    /// Remove a node by UUID and commit.
    ///
    /// The node is no longer reachable from the new head commit's node
    /// tree; prior commits that referenced it remain addressable by
    /// their CIDs (mnem's history is append-only). Edges incident to
    /// the node are NOT auto-removed; remove them explicitly if needed.
    ///
    /// Returns `True` if the node existed at the previous head, `False`
    /// otherwise. Either way, a commit is produced.
    #[pyo3(signature = (author, message, id))]
    pub fn delete_node(&self, author: &str, message: &str, id: &str) -> PyResult<bool> {
        let uuid = NodeId::parse_uuid(id)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("invalid UUID: {e}")))?;
        let mut guard = self.inner.lock().map_err(poison)?;
        let existed = guard.lookup_node(&uuid).map_err(map_core_err)?.is_some();
        let mut tx = guard.start_transaction();
        tx.remove_node(uuid);
        let new_repo = tx.commit(author, message).map_err(map_core_err)?;
        *guard = new_repo;
        Ok(existed)
    }

    /// Supersede an existing node's summary and/or props by committing a
    /// new node with the same UUID. The old node blob is still
    /// addressable by its CID; new queries at HEAD see the new version.
    ///
    /// `summary=None` and `props=None` leave the old value in place.
    /// Pass `summary=""` to explicitly clear the summary.
    ///
    /// Raises `KeyError` if no node with `id` exists at the current head.
    #[pyo3(signature = (author, message, id, *, summary = None, props = None))]
    pub fn update_node(
        &self,
        author: &str,
        message: &str,
        id: &str,
        summary: Option<&str>,
        props: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<String> {
        let uuid = NodeId::parse_uuid(id)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("invalid UUID: {e}")))?;
        let mut guard = self.inner.lock().map_err(poison)?;
        let existing = guard
            .lookup_node(&uuid)
            .map_err(map_core_err)?
            .ok_or_else(|| {
                pyo3::exceptions::PyKeyError::new_err(format!("no node with id={id}"))
            })?;

        // Start from the existing node so unspecified fields carry over.
        let mut node = CoreNode::new(uuid, existing.ntype.as_str());
        node = match summary {
            // Explicitly clear with an empty string.
            Some("") => node,
            // Overwrite with the new summary.
            Some(s) => node.with_summary(s),
            // Not supplied: keep the old summary.
            None => {
                if let Some(s) = &existing.summary {
                    node.with_summary(s.as_str())
                } else {
                    node
                }
            }
        };
        // Props: if provided, replace entirely; if None, carry over.
        if let Some(dict) = props {
            for (k, v) in dict {
                let key: String = k.extract()?;
                node = node.with_prop(key, py_to_ipld(&v)?);
            }
        } else {
            for (k, v) in &existing.props {
                node = node.with_prop(k.clone(), v.clone());
            }
        }
        if let Some(c) = &existing.content {
            node = node.with_content(c.clone());
        }
        // Dense embeddings live in the per-commit sidecar, not on the
        // Node. Repos written before the sidecar shipped that carry an
        // inline `embed` map need `mnem reindex` to migrate those
        // vectors to the sidecar; this update path does not lift them
        // automatically.

        let mut tx = guard.start_transaction();
        tx.add_node(&node).map_err(map_core_err)?;
        let new_repo = tx.commit(author, message).map_err(map_core_err)?;
        *guard = new_repo;
        Ok(uuid.to_uuid_string())
    }

    /// Open a write transaction. Returns a context manager that commits
    /// on successful exit and silently abandons on exception.
    pub fn transaction(&self, author: String, message: String) -> Transaction {
        Transaction {
            repo: self.inner.clone(),
            author,
            message,
            pending_nodes: Vec::new(),
            committed: false,
        }
    }

    /// Retrieve ranked, rendered nodes under a token budget. Returns a
    /// `RetrievalResult` whose `items` list is already packed to fit
    /// the budget in RRF-rank order.
    ///
    /// Releases both the internal repo mutex AND the Python GIL while
    /// the actual retrieval runs, so concurrent Python threads doing
    /// independent queries do not serialize on this call.
    #[pyo3(signature = (
        text = None,
        label = None,
        vector = None,
        model = None,
        token_budget = None,
        limit = None,
        vector_weight = None,
    ))]
    #[allow(clippy::too_many_arguments)]
    pub fn retrieve(
        &self,
        py: Python<'_>,
        text: Option<String>,
        label: Option<String>,
        vector: Option<Vec<f32>>,
        model: Option<String>,
        token_budget: Option<u32>,
        limit: Option<usize>,
        vector_weight: Option<f32>,
    ) -> PyResult<RetrievalResult> {
        if vector.is_some() && model.is_none() {
            return Err(PyValueError::new_err(
                "retrieve(): pass `model` whenever `vector` is set",
            ));
        }
        // `label` gated behind `MNEM_BENCH`. Off by default so casual
        // Python callers never stumble into label-scoped state; the
        // filter is silently dropped. Parity with POST /v1/retrieve in
        // mnem-http and `mnem_retrieve` in mnem-mcp.
        let gate = allow_labels();
        let label = if gate { label } else { None };
        // Clone out the current head view under the mutex (O(1) since
        // ReadonlyRepo's fields are Arc-wrapped) so concurrent readers
        // never serialise on the long-running tree walk below.
        let repo_snapshot = {
            let g = self.inner.lock().map_err(poison)?;
            g.clone()
        };
        py.detach(move || {
            let mut r = repo_snapshot.retrieve();
            if let Some(t) = text {
                r = r.query_text(t);
            }
            if let Some(l) = label {
                r = r.label(l);
            }
            if let (Some(v), Some(m)) = (vector, model) {
                r = r.vector(m, v);
            }
            if let Some(b) = token_budget {
                r = r.token_budget(b);
            }
            if let Some(n) = limit {
                r = r.limit(n);
            }
            if let Some(w) = vector_weight {
                r = r.vector_weight(w);
            }
            r.execute().map_err(map_core_err)
        })
        .map(RetrievalResult::from_core)
    }

    /// Structured query: exact label + single-property equality lookup,
    /// backed by the Prolly indexes. Returns a list of dicts with keys
    /// `node_id`, `ntype`, `summary`, `props`. Use `retrieve()` instead
    /// for ranked / budget-aware retrieval.
    #[pyo3(signature = (label = None, where_eq = None, limit = None))]
    pub fn query(
        &self,
        py: Python<'_>,
        label: Option<String>,
        where_eq: Option<&Bound<'_, PyDict>>,
        limit: Option<usize>,
    ) -> PyResult<Vec<Py<PyAny>>> {
        // `label` gated behind `MNEM_BENCH`. Off by default: the filter
        // is silently dropped so a casual Python user calling
        // `repo.query(label="Person")` sees the full unscoped result set
        // rather than being steered into per-item scoping. Parity with
        // `retrieve()` above.
        let label = if allow_labels() { label } else { None };
        // Extract the prop predicate (if any) while the GIL is held and
        // the Bound<PyDict> is valid. `where_eq` accepts AT MOST one
        // entry; ambiguous callers that pass two keys get a clean
        // ValueError instead of having one key silently dropped.
        let prop_filter: Option<(String, Ipld)> = match where_eq {
            None => None,
            Some(dict) if dict.len() > 1 => {
                return Err(PyValueError::new_err(
                    "query(where_eq=...) accepts at most one key; pass multiple filters \
                     by chaining queries or use the richer Retriever API",
                ));
            }
            Some(dict) => match dict.iter().next() {
                None => None,
                Some((k, v)) => {
                    let key: String = k.extract()?;
                    Some((key, py_to_ipld(&v)?))
                }
            },
        };
        let repo_snapshot = {
            let g = self.inner.lock().map_err(poison)?;
            g.clone()
        };
        let hits = py.detach(move || {
            let mut q = repo_snapshot.query();
            if let Some(l) = label {
                q = q.label(l);
            }
            if let Some((name, value)) = prop_filter {
                q = q.where_prop(name, PropPredicate::Eq(value));
            }
            if let Some(n) = limit {
                q = q.limit(n);
            }
            q.execute().map_err(map_core_err)
        })?;
        hits.into_iter()
            .map(|h| {
                let d = PyDict::new(py);
                d.set_item("node_id", h.node.id.to_uuid_string())?;
                d.set_item("ntype", h.node.ntype.clone())?;
                d.set_item("summary", h.node.summary.clone())?;
                let props = PyDict::new(py);
                for (k, v) in &h.node.props {
                    if let Some(obj) = ipld_to_py_scalar(py, v) {
                        props.set_item(k, obj)?;
                    }
                }
                d.set_item("props", props)?;
                Ok(d.into_any().unbind())
            })
            .collect()
    }
}

// ============================================================
// Transaction - context manager
// ============================================================

/// One queued write inside a pending `Transaction`. The `Option` lets
/// `add_embedding_f32` take ownership of the `CoreNode`, consume it via
/// `with_embed`, and put it back - without a sentinel-node trick that
/// would leave a zero-id node behind if the re-assignment panicked.
enum PendingWrite {
    /// A staged node plus its optional dense embedding. The embedding
    /// is committed to the embedding sidecar (not inlined onto the
    /// Node) so two machines re-deriving the same source text on
    /// different ORT thread counts produce identical NodeCids.
    Node {
        node: Option<CoreNode>,
        embed: Option<(String, Embedding)>,
    },
}

/// A write transaction returned by `Repo.transaction(...)`. Usable as a
/// context manager:
///
/// ```python
/// with repo.transaction(author="me", message="seed") as tx:
///     tx.add_node(ntype="Memory", summary="Alice lives in Berlin")
///     tx.add_node(ntype="Memory", summary="Bob moved to Paris")
/// # commit happens here on clean exit; on exception, nothing is committed.
/// ```
#[pyclass(module = "pymnem")]
pub struct Transaction {
    repo: Arc<Mutex<ReadonlyRepo>>,
    author: String,
    message: String,
    pending_nodes: Vec<PendingWrite>,
    committed: bool,
}

#[pymethods]
impl Transaction {
    /// Queue a node for the pending commit. Returns the new UUID string
    /// immediately so callers can reference it without waiting for commit.
    #[pyo3(signature = (ntype, summary = None, props = None, content = None))]
    pub fn add_node(
        &mut self,
        ntype: &str,
        summary: Option<&str>,
        props: Option<&Bound<'_, PyDict>>,
        content: Option<&[u8]>,
    ) -> PyResult<String> {
        // `ntype` gated behind `MNEM_BENCH`. Off by default: every
        // ingested node is coerced to `Node::DEFAULT_NTYPE` regardless
        // of what the caller passed. Parity with POST /v1/nodes in
        // mnem-http and `mnem_commit` in mnem-mcp. Callers who want to
        // keep caller-supplied `ntype` must launch the Python process
        // under `MNEM_BENCH=1` (or any truthy value).
        let ntype = if allow_labels() {
            ntype
        } else {
            CoreNode::DEFAULT_NTYPE
        };
        let mut node = CoreNode::new(NodeId::new_v7(), ntype);
        if let Some(s) = summary {
            node = node.with_summary(s);
        }
        if let Some(dict) = props {
            for (k, v) in dict {
                let key: String = k.extract()?;
                node = node.with_prop(key, py_to_ipld(&v)?);
            }
        }
        if let Some(c) = content {
            node = node.with_content(Bytes::copy_from_slice(c));
        }
        let id_str = node.id.to_uuid_string();
        self.pending_nodes.push(PendingWrite::Node {
            node: Some(node),
            embed: None,
        });
        Ok(id_str)
    }

    /// Attach an f32 embedding to the most recently added node. Errors
    /// if called before any `add_node`. The embedding is staged on
    /// the pending write and routed into the embedding sidecar at
    /// commit time, so the resulting NodeCid is independent of the
    /// embedding bytes (and therefore of ORT thread-count drift).
    pub fn add_embedding_f32(&mut self, model: &str, values: Vec<f32>) -> PyResult<()> {
        let Some(PendingWrite::Node {
            node,
            embed: pending_embed,
        }) = self.pending_nodes.last_mut()
        else {
            return Err(PyValueError::new_err(
                "add_embedding_f32(): call add_node() first",
            ));
        };
        if node.is_none() {
            return Err(PyValueError::new_err(
                "add_embedding_f32(): pending node was already consumed (internal bug)",
            ));
        }
        let mut bytes = Vec::with_capacity(values.len() * 4);
        for x in &values {
            bytes.extend_from_slice(&x.to_le_bytes());
        }
        let emb = Embedding {
            model: model.to_string(),
            dtype: Dtype::F32,
            dim: u32::try_from(values.len())
                .map_err(|_| PyValueError::new_err("vector too long"))?,
            vector: Bytes::from(bytes),
        };
        *pending_embed = Some((model.to_string(), emb));
        Ok(())
    }

    /// Commit the pending writes. Returns the new op-id CID string.
    /// Usually invoked implicitly by the context manager.
    pub fn commit(&mut self) -> PyResult<String> {
        if self.committed {
            return Err(MnemError::new_err("transaction already committed"));
        }
        let mut guard = self.repo.lock().map_err(poison)?;
        let mut tx = guard.start_transaction();
        for p in self.pending_nodes.drain(..) {
            match p {
                PendingWrite::Node {
                    node: Some(n),
                    embed,
                } => {
                    let cid = tx.add_node(&n).map_err(map_core_err)?;
                    if let Some((model, emb)) = embed {
                        tx.set_embedding(cid, model, emb).map_err(map_core_err)?;
                    }
                }
                PendingWrite::Node { node: None, .. } => {
                    // Defensive: pending slot was emptied but never
                    // re-filled. Skip rather than committing a sentinel.
                    continue;
                }
            }
        }
        let new_repo = tx
            .commit(self.author.as_str(), self.message.as_str())
            .map_err(map_core_err)?;
        let op_id = new_repo.op_id().to_string();
        *guard = new_repo;
        self.committed = true;
        Ok(op_id)
    }

    fn __enter__(slf: PyRef<'_, Self>) -> PyResult<PyRef<'_, Self>> {
        // Reject re-entry: a `with tx:` block that has already committed
        // (or whose Python handle outlives a prior `with`) must not
        // silently accept `add_node` calls that will explode at commit.
        if slf.committed {
            return Err(MnemError::new_err(
                "transaction has already been committed; open a new one via repo.transaction(...)",
            ));
        }
        Ok(slf)
    }

    #[pyo3(signature = (exc_type = None, exc_value = None, traceback = None))]
    #[allow(unused_variables)]
    fn __exit__(
        &mut self,
        exc_type: Option<Py<PyAny>>,
        exc_value: Option<Py<PyAny>>,
        traceback: Option<Py<PyAny>>,
    ) -> PyResult<bool> {
        // On exception, abandon pending writes silently. On clean exit,
        // commit. Returning false tells Python NOT to suppress any
        // active exception.
        if exc_type.is_none() && !self.committed && !self.pending_nodes.is_empty() {
            self.commit()?;
        }
        Ok(false)
    }

    const fn pending_count(&self) -> usize {
        self.pending_nodes.len()
    }
}

// ============================================================
// RetrievalResult + RetrievedItem
// ============================================================

/// One packed retrieval hit. All fields are plain Python types.
#[pyclass(module = "pymnem", skip_from_py_object)]
#[derive(Clone)]
pub struct RetrievedItem {
    /// Stable node identity as a canonical UUID string.
    #[pyo3(get)]
    pub node_id: String,
    /// The node's type label (e.g. `"Memory"`, `"Person"`, `"Document"`).
    #[pyo3(get)]
    pub ntype: String,
    /// The node's short LLM-facing summary, if set.
    #[pyo3(get)]
    pub summary: Option<String>,
    /// Canonical text rendering used for token-budget packing. Safe to
    /// forward into an LLM prompt verbatim.
    #[pyo3(get)]
    pub rendered: String,
    /// Estimated tokens consumed by `rendered` under the retriever's
    /// `TokenEstimator` (defaults to the char-heuristic).
    #[pyo3(get)]
    pub tokens: u32,
    /// Composite RRF score (or the single-ranker native score when only
    /// one ranker is configured).
    #[pyo3(get)]
    pub score: f32,
}

#[pymethods]
impl RetrievedItem {
    fn __repr__(&self) -> String {
        format!(
            "RetrievedItem(node_id={:?}, ntype={:?}, tokens={}, score={:.4})",
            self.node_id, self.ntype, self.tokens, self.score
        )
    }
}

/// A complete retrieval result plus cost metadata.
#[pyclass(module = "pymnem", skip_from_py_object)]
#[derive(Clone)]
pub struct RetrievalResult {
    /// Items that fit inside the token budget, in RRF-rank order.
    #[pyo3(get)]
    pub items: Vec<RetrievedItem>,
    /// Total estimated tokens consumed by `items`.
    #[pyo3(get)]
    pub tokens_used: u32,
    /// Budget the caller configured (or `u32::MAX` if unset).
    #[pyo3(get)]
    pub tokens_budget: u32,
    /// Ranked candidates that did NOT fit inside the remaining budget.
    /// Non-zero means the budget was tight.
    #[pyo3(get)]
    pub dropped: u32,
    /// Total distinct candidates after ranker fusion + filtering,
    /// before budget packing.
    #[pyo3(get)]
    pub candidates_seen: u32,
}

#[pymethods]
impl RetrievalResult {
    fn __repr__(&self) -> String {
        format!(
            "RetrievalResult(items={}, tokens_used={}, tokens_budget={}, dropped={}, candidates_seen={})",
            self.items.len(),
            self.tokens_used,
            self.tokens_budget,
            self.dropped,
            self.candidates_seen
        )
    }

    /// Iterate the items list directly: `for item in result: ...`.
    fn __iter__(slf: PyRef<'_, Self>) -> RetrievalIter {
        RetrievalIter {
            items: slf.items.clone(),
            index: 0,
        }
    }

    fn __len__(&self) -> usize {
        self.items.len()
    }
}

impl RetrievalResult {
    fn from_core(r: mnem_core::retrieve::RetrievalResult) -> Self {
        let items = r
            .items
            .into_iter()
            .map(|it| RetrievedItem {
                node_id: it.node.id.to_uuid_string(),
                ntype: it.node.ntype.clone(),
                summary: it.node.summary.clone(),
                rendered: it.rendered,
                tokens: it.tokens,
                score: it.score,
            })
            .collect();
        Self {
            items,
            tokens_used: r.tokens_used,
            tokens_budget: r.tokens_budget,
            dropped: r.dropped,
            candidates_seen: r.candidates_seen,
        }
    }
}

#[pyclass(module = "pymnem")]
struct RetrievalIter {
    items: Vec<RetrievedItem>,
    index: usize,
}

#[pymethods]
impl RetrievalIter {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(&mut self) -> Option<RetrievedItem> {
        let out = self.items.get(self.index).cloned()?;
        self.index += 1;
        Some(out)
    }
}

// ============================================================
// Module init
// ============================================================

// Compiled module is `pymnem._mnem`; pure-Python `pymnem/__init__.py`
// re-exports the classes so `from pymnem import Repo` still works.
// Renaming this fn is a BREAKING change for anyone doing
// `import _mnem` directly - nobody should; `_mnem` is private.
#[pymodule]
fn _mnem(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Parity with `mnem-http` / `mnem-mcp`: when `MNEM_BENCH` is set,
    // caller-supplied `ntype` and `label` kwargs are honoured. Emit a
    // one-line stderr warning at module init so a Python user who
    // accidentally launched their process under a stray `MNEM_BENCH=1`
    // environment has an easy trace. Note the gate itself is re-read
    // per call (see `allow_labels()` above), so flipping the env var
    // at test time between `import pymnem` calls still works - this
    // warning only fires once.
    if allow_labels() {
        eprintln!(
            "pymnem: MNEM_BENCH set; caller-supplied `ntype` / `label` kwargs will be honoured on add_node, retrieve, and query."
        );
    }
    m.add_class::<Repo>()?;
    m.add_class::<Transaction>()?;
    m.add_class::<RetrievedItem>()?;
    m.add_class::<RetrievalResult>()?;
    m.add("MnemError", m.py().get_type::<MnemError>())?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}

// ============================================================
// Helpers
// ============================================================

fn poison<T>(_e: std::sync::PoisonError<T>) -> PyErr {
    MnemError::new_err("internal repo mutex was poisoned by a previous panic")
}

fn py_to_ipld(v: &Bound<'_, PyAny>) -> PyResult<Ipld> {
    if v.is_none() {
        return Ok(Ipld::Null);
    }
    // Order matters: bool is a subclass of int in Python, so check bool
    // before int, and int before float (a Python int extracts cleanly
    // as i64 but also as f64).
    if v.is_instance_of::<PyBool>() {
        return Ok(Ipld::Bool(v.extract()?));
    }
    if let Ok(i) = v.extract::<i64>() {
        return Ok(Ipld::Integer(i128::from(i)));
    }
    if let Ok(f) = v.extract::<f64>() {
        return Ok(Ipld::Float(f));
    }
    if let Ok(s) = v.extract::<String>() {
        return Ok(Ipld::String(s));
    }
    Err(PyValueError::new_err(
        "unsupported prop value type (use str, int, float, bool, or None)",
    ))
}

fn ipld_to_py_scalar(py: Python<'_>, v: &Ipld) -> Option<Py<PyAny>> {
    match v {
        Ipld::Null => Some(py.None()),
        Ipld::Bool(b) => Some(b.into_pyobject(py).ok()?.to_owned().unbind().into()),
        Ipld::Integer(n) => i64::try_from(*n)
            .ok()
            .map(|i| i.into_pyobject(py).ok())
            .flatten()
            .map(|b| b.to_owned().unbind().into()),
        Ipld::Float(f) => Some(f.into_pyobject(py).ok()?.to_owned().unbind().into()),
        Ipld::String(s) => Some(s.clone().into_pyobject(py).ok()?.to_owned().unbind().into()),
        _ => None,
    }
}

#[cfg(test)]
mod mnem_bench_parse_tests {
    use super::parse_allow_labels;

    #[test]
    fn unset_parses_false() {
        assert!(!parse_allow_labels(None));
    }

    #[test]
    fn falsy_strings_parse_false() {
        for v in [
            "", "0", "false", "FALSE", "False", "no", "No", "NO", "off", "Off", "OFF", "  ", "  0 ",
        ] {
            assert!(
                !parse_allow_labels(Some(v)),
                "expected `{v:?}` to parse false"
            );
        }
    }

    #[test]
    fn truthy_strings_parse_true() {
        for v in ["1", "true", "yes", "on", "YES", "benchmark", "anything"] {
            assert!(
                parse_allow_labels(Some(v)),
                "expected `{v:?}` to parse true"
            );
        }
    }
}
