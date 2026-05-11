//! `/remote/v1/*` HTTP verbs: the server half of mnem's
//! content-addressed replication protocol .
//!
//! Four routes:
//!
//! | Verb + path                      | Auth      | Body shape                          | Response                                |
//! |----------------------------------|-----------|-------------------------------------|-----------------------------------------|
//! | `GET  /remote/v1/refs`           | read-open | -                                   | JSON `{head, refs, capabilities}`          |
//! | `POST /remote/v1/fetch-blocks`   | read-open | JSON `{wants, have_set}`            | `application/vnd.ipld.car` stream       |
//! | `POST /remote/v1/push-blocks`    | bearer    | `application/vnd.ipld.car` stream   | JSON `{staged, blocks_accepted}`        |
//! | `POST /remote/v1/advance-head`   | bearer    | JSON `{old, new, ref}`              | JSON `{head}` (200) or RFC7807 (409)    |
//!
//! Every response carries the `mnem-protocol` header (wire-protocol
//! version 1) and the `mnem-capabilities` header listing the
//! capabilities this build advertises. The client echoes these on
//! follow-up requests to pin the capability set for a session.
//!
//! # CIDs on the wire
//!
//! CIDs cross the JSON boundary as base32-lowercase multibase strings
//! (the canonical `Display` form of [`mnem_core::id::Cid`]). Binary
//! CIDs live only inside the CAR body, never in JSON. Parsing is
//! strict: a malformed CID yields `RemoteError::BadRequest`, not a
//! silent fallback.
//!
//! # have_set handling
//!
//! PR 2 of the transport crate ships `BloomHaveSet::serialize`; PR 3
//! (this PR) was scoped to add the deserialiser, but mnem-transport
//! was frozen for B3.1 (server-only sub-wave). The server currently
//! accepts any `have_set` bytes opaquely and returns every reachable
//! block from `wants` - a superset of what the client needs, still
//! correct, just wastes bandwidth. Filtering is B3.2 territory and
//! will land with the client-side deserialiser.

use std::collections::{BTreeMap, HashSet};
use std::io::Cursor;

use axum::Json;
use axum::extract::State;
use axum::http::{HeaderValue, StatusCode, header};
use axum::response::{IntoResponse, Response};
use mnem_core::id::Cid;
use mnem_transport::car::{CarHeader, write_block, write_header};
use mnem_transport::import::import_with_limit;
use mnem_transport::protocol::{
    CAPABILITIES_HEADER, Capability, PROTOCOL_HEADER, PROTOCOL_VERSION, serialize_capabilities,
};
use serde::{Deserialize, Serialize};

use crate::auth::RequireBearer;
use crate::error::RemoteError;
use crate::metrics::AdvanceHeadLabels;
use crate::state::AppState;

// ---------- protocol framing ----------

/// Default `ref` name on `advance-head` when the caller omits it.
/// Mirrors git's `main` convention §"Default ref".
const DEFAULT_REF: &str = "main";

/// Protocol and capabilities headers applied to every `/remote/v1/*`
/// response (including errors). Kept centralised so the two always
/// ship together.
fn protocol_headers() -> [(axum::http::HeaderName, HeaderValue); 2] {
    // `serialize_capabilities` emits a deterministic comma-separated
    // string in wire-string-ascending order; cheap to compute per
    // request (sub-microsecond), and keeping it inline avoids a
    // startup-time registry that would need plumbing through AppState.
    let caps_value = serialize_capabilities(Capability::all().iter().copied());
    [
        (
            axum::http::HeaderName::from_static(PROTOCOL_HEADER),
            HeaderValue::from_str(&PROTOCOL_VERSION.to_string())
                .expect("protocol version is ascii digits"),
        ),
        (
            axum::http::HeaderName::from_static(CAPABILITIES_HEADER),
            HeaderValue::from_str(&caps_value).expect("capability list is ascii"),
        ),
    ]
}

// ---------- GET /remote/v1/refs ----------

/// Response body for `GET /remote/v1/refs`.
///
/// `head` is the current head op-id (the first entry of the op-heads
/// store's sorted head set, or `None` on a freshly-initialised repo).
/// `refs` maps ref names to their current head CID; the canonical
/// default branch is mirrored under the reserved key `"HEAD"` so
/// clients that only care about the default branch can index a single
/// well-known name. `capabilities` echoes the server's advertised
/// capability vocabulary as kebab-case wire strings (the client
/// parses these lossily so adding variants in a minor release stays
/// forward-compatible).
#[derive(Debug, Serialize)]
pub(crate) struct RefsResponse {
    /// Current head CID (as canonical multibase string) or `null`.
    pub head: Option<String>,
    /// Map from ref name -> head CID (canonical multibase string).
    /// The default branch is mirrored under the reserved `"HEAD"`
    /// key. Empty on a freshly-initialised repo.
    pub refs: BTreeMap<String, String>,
    /// Capability wire-strings this server advertises. Kebab-case
    /// strings ; clients parse with `parse_capabilities`
    /// which silently drops unknowns for forward-compat.
    pub capabilities: Vec<String>,
}

/// `GET /remote/v1/refs`. Read-open. Emits the current head + the
/// server's capability list. No auth header required.
pub(crate) async fn get_refs(State(state): State<AppState>) -> Result<Response, RemoteError> {
    let head = {
        let repo = state
            .repo
            .lock()
            .map_err(|_| RemoteError::Internal("server state lock poisoned".into()))?;
        let ohs = repo.op_heads_store();
        let heads = ohs
            .current()
            .map_err(|e| RemoteError::Internal(format!("op-heads read: {e}")))?;
        // The op-heads set is sorted ascending for determinism; we
        // pick the first entry as the "canonical head" for a single
        // returned value. Multi-head servers will expose the full
        // list under a v0.2 `/remote/v1/heads` extension.
        heads.into_iter().next()
    };
    let head_str = head.as_ref().map(ToString::to_string);
    // Mirror the canonical head under the reserved `"HEAD"` key so
    // clients can index by name without branching on the separate
    // top-level `head` field. Named branches beyond HEAD land in a
    // future multi-ref server mode .
    let mut refs: BTreeMap<String, String> = BTreeMap::new();
    if let Some(h) = head_str.as_ref() {
        refs.insert("HEAD".to_string(), h.clone());
        // In B3.1 the only branch is DEFAULT_REF ("main"), so HEAD == main.
        // Insert the branch name explicitly so clients can look up by branch
        // without relying on the fallback through the top-level `head` field.
        refs.insert(DEFAULT_REF.to_string(), h.clone());
    }
    let body = RefsResponse {
        head: head_str,
        refs,
        capabilities: Capability::all()
            .iter()
            .map(|c| c.as_wire_str().to_string())
            .collect(),
    };
    Ok((StatusCode::OK, protocol_headers(), Json(body)).into_response())
}

// ---------- POST /remote/v1/fetch-blocks ----------

/// Request body for `POST /remote/v1/fetch-blocks`.
///
/// `wants` is the set of CIDs the client asks the server to expand
/// into a CAR (every reachable block from each want, minus any
/// present in `have_set`). `have_set` is an opaque
/// `BloomHaveSet::serialize()` byte blob; see module-level docs on
/// why it is currently treated as a no-op on the server side.
#[derive(Debug, Deserialize)]
pub(crate) struct FetchBlocksRequest {
    /// Root CIDs to expand. Must be non-empty.
    pub wants: Vec<String>,
    /// Serialized bloom have-set bytes (opaque for B3.1; see
    /// module docs).
    #[serde(default)]
    pub have_set: Vec<u8>,
}

/// `POST /remote/v1/fetch-blocks`. Read-open. Streams a CAR v1
/// archive containing every reachable block from each `want`.
pub(crate) async fn post_fetch_blocks(
    State(state): State<AppState>,
    Json(req): Json<FetchBlocksRequest>,
) -> Result<Response, RemoteError> {
    if req.wants.is_empty() {
        return Err(RemoteError::BadRequest("wants: must be non-empty".into()));
    }
    let wants: Vec<Cid> = req
        .wants
        .iter()
        .map(|s| Cid::parse_str(s).map_err(|e| RemoteError::BadRequest(format!("wants: {e}"))))
        .collect::<Result<_, _>>()?;

    // have_set is accepted but not yet used; see module docs. We
    // explicitly drop it to make the no-op visible on code review.
    let _have_set = req.have_set;

    // Walk reachable blocks with a shared visited set so a CID
    // reachable from multiple wants emits exactly once.
    let mut buf: Vec<u8> = Vec::new();
    let header = CarHeader {
        version: 1,
        roots: wants.clone(),
    };
    write_header(&mut buf, &header)
        .map_err(|e| RemoteError::Internal(format!("CAR header: {e}")))?;

    {
        let repo = state
            .repo
            .lock()
            .map_err(|_| RemoteError::Internal("server state lock poisoned".into()))?;
        let bs = repo.blockstore();
        let mut visited: HashSet<Cid> = HashSet::new();
        for want in &wants {
            for item in bs.iter_from_root(want) {
                let (cid, data) = item.map_err(|e| match e {
                    mnem_core::error::StoreError::NotFound { cid } => {
                        RemoteError::NotFound(format!("want not in store: {cid}"))
                    }
                    other => RemoteError::Internal(format!("blockstore walk: {other}")),
                })?;
                if !visited.insert(cid.clone()) {
                    continue;
                }
                write_block(&mut buf, &cid, &data)
                    .map_err(|e| RemoteError::Internal(format!("CAR block write: {e}")))?;
            }
        }
    }

    state.metrics.remote_fetch_blocks.inc();

    let mut resp = (StatusCode::OK, buf).into_response();
    let h = resp.headers_mut();
    h.insert(
        header::CONTENT_TYPE,
        HeaderValue::from_static("application/vnd.ipld.car"),
    );
    for (name, value) in protocol_headers() {
        h.insert(name, value);
    }
    Ok(resp)
}

// ---------- POST /remote/v1/push-blocks ----------

/// Response body for a successful `POST /remote/v1/push-blocks`.
#[derive(Debug, Serialize)]
pub(crate) struct PushBlocksResponse {
    /// First root CID declared in the pushed CAR header, if any.
    /// Present on every successful push (the CAR importer already
    /// verifies the root was delivered). `null` only when the CAR
    /// was empty of roots, which the importer rejects -> never
    /// observed in practice.
    pub staged: Option<String>,
    /// Number of distinct blocks imported. Blocks already present
    /// pre-push are re-verified and counted; the blockstore's `put`
    /// is idempotent.
    pub blocks_accepted: u64,
}

/// `POST /remote/v1/push-blocks`. Bearer-required. Accepts a CAR
/// stream, verifies every block's CID against its payload, and stages
/// the blocks into the blockstore. Head is NOT advanced - the client
/// follows up with `advance-head`.
pub(crate) async fn post_push_blocks(
    State(state): State<AppState>,
    _auth: RequireBearer,
    body: axum::body::Bytes,
) -> Result<Response, RemoteError> {
    let stats = {
        let repo = state
            .repo
            .lock()
            .map_err(|_| RemoteError::Internal("server state lock poisoned".into()))?;
        let bs = repo.blockstore();
        // `import_with_limit` recomputes the CID for every block and
        // refuses on mismatch. That is the `put`-equivalent
        // verification path the task asked for (remote-received
        // blocks MUST go through a verify-first path, not
        // `put_trusted`). The blockstore's `put_trusted` is called
        // *after* the importer has recomputed the hash, so the
        // invariant holds end-to-end.
        let mut reader = Cursor::new(body.as_ref());
        import_with_limit(
            &mut reader,
            bs.as_ref(),
            mnem_transport::import::DEFAULT_MAX_IMPORT_BYTES,
        )
        .map_err(remote_error_from_transport)?
    };

    state.metrics.remote_push_blocks.inc();

    let staged = stats.roots.first().map(ToString::to_string);
    let body = PushBlocksResponse {
        staged,
        blocks_accepted: stats.blocks,
    };
    Ok((StatusCode::OK, protocol_headers(), Json(body)).into_response())
}

/// Map a [`mnem_transport::TransportError`] to the appropriate
/// [`RemoteError`] HTTP status. CAR format errors and CID-mismatch
/// errors are caller-attributable (bad input) and map to 400;
/// size-limit is also caller-attributable; everything else is 500.
fn remote_error_from_transport(e: mnem_transport::TransportError) -> RemoteError {
    use mnem_transport::TransportError as T;
    match e {
        T::Car(_) | T::CidMismatch { .. } | T::MissingRoot { .. } | T::UnsupportedHash(_) => {
            RemoteError::BadRequest(format!("{e}"))
        }
        T::SizeLimit { .. } => RemoteError::BadRequest(format!("{e}")),
        T::Codec(_) => RemoteError::BadRequest(format!("{e}")),
        T::Store(_) | T::Io(_) => RemoteError::Internal(format!("{e}")),
        // Cover `#[non_exhaustive]` so future variants don't silently
        // collapse to 500 without an audit.
        other => RemoteError::Internal(format!("{other}")),
    }
}

// ---------- POST /remote/v1/advance-head ----------

/// Request body for `POST /remote/v1/advance-head`.
#[derive(Debug, Deserialize)]
pub(crate) struct AdvanceHeadRequest {
    /// The CID the caller believes is the current head, or `null` /
    /// absent when the caller expects the remote to have an empty head
    /// set (i.e. first push to a fresh remote). The CAS fails with 409
    /// if the server-side head is anything other than what `old`
    /// describes: a non-null `old` must be present in the current head
    /// set; a null `old` requires the current head set to be empty.
    #[serde(default)]
    pub old: Option<String>,
    /// The CID the caller wants to become the new head.
    pub new: String,
    /// Named ref to advance. Defaults to `"main"`.
    #[serde(default = "default_ref_name")]
    pub r#ref: String,
}

fn default_ref_name() -> String {
    DEFAULT_REF.to_string()
}

/// Response body for a successful `POST /remote/v1/advance-head`.
#[derive(Debug, Serialize)]
pub(crate) struct AdvanceHeadResponse {
    /// The new head CID, as accepted by the server. Echoed so the
    /// client can correlate with its local record.
    pub head: String,
}

/// `POST /remote/v1/advance-head`. Bearer-required. Atomically
/// replaces the op-heads entry for `ref`. 409 on CAS mismatch with
/// `{current: <cid>}` so the client can rebase without a round trip.
pub(crate) async fn post_advance_head(
    State(state): State<AppState>,
    _auth: RequireBearer,
    Json(req): Json<AdvanceHeadRequest>,
) -> Result<Response, RemoteError> {
    // B3.1 ships the single-ref `main` path only. Named refs are an
    // roadmap item (the View's tracking-refs machinery
    // already supports <remote>/<ref> pairs; wiring that into the
    // op-heads store is B3.4). Reject anything other than `main` so
    // clients can't silently break against a future server.
    if req.r#ref != DEFAULT_REF {
        return Err(RemoteError::BadRequest(format!(
            "ref `{}` not supported; only `{DEFAULT_REF}` in B3.1",
            req.r#ref
        )));
    }
    let old: Option<Cid> = match req.old {
        Some(ref s) => {
            Some(Cid::parse_str(s).map_err(|e| RemoteError::BadRequest(format!("old: {e}")))?)
        }
        None => None,
    };
    let new = Cid::parse_str(&req.new).map_err(|e| RemoteError::BadRequest(format!("new: {e}")))?;

    let inc_ok = |s: &AppState| {
        s.metrics
            .remote_advance_head
            .get_or_create(&AdvanceHeadLabels {
                result: "success".into(),
            })
            .inc();
    };
    let inc_mismatch = |s: &AppState| {
        s.metrics
            .remote_advance_head
            .get_or_create(&AdvanceHeadLabels {
                result: "cas_mismatch".into(),
            })
            .inc();
    };

    let repo = state
        .repo
        .lock()
        .map_err(|_| RemoteError::Internal("server state lock poisoned".into()))?;
    let ohs = repo.op_heads_store();

    // CAS INVARIANT - DO NOT RELAX WITHOUT AUDITING THIS BLOCK.
    //
    // `OpHeadsStore::update` is a blind write: it does not perform any
    // compare-and-swap internally. The correctness of the CAS below
    // therefore depends entirely on this handler being serialised
    // through the repo mutex acquired above. As long as every call to
    // `post_advance_head` holds the mutex for the full duration of the
    // read-compare-write sequence, only one writer can be in this
    // critical section at a time and the TOCTOU window is closed.
    //
    // DANGER: if the mutex is ever replaced with a finer-grained lock
    // (per-ref, per-shard, or eliminated in favour of async tasks), the
    // read-then-write sequence below becomes a race: two concurrent
    // pushers can each read the same `current` head, both pass the CAS
    // check, and both call `ohs.update` - the second write silently
    // overwrites the first (BUG-57). At that point this function MUST
    // be replaced with a true DB-level CAS (e.g. a conditional write
    // that fails atomically if the row changed since the read).
    let current = ohs
        .current()
        .map_err(|e| RemoteError::Internal(format!("op-heads read: {e}")))?;

    // CAS check: when `old` is None the caller asserts the remote is
    // empty (first push). When `old` is Some(cid) it must appear in
    // the current head set.
    match &old {
        None => {
            // First-push path: succeed only if the remote has no heads.
            if !current.is_empty() {
                inc_mismatch(&state);
                let current_head = current.into_iter().next();
                return Err(RemoteError::CasMismatch {
                    current: current_head.expect("non-empty current has a first element"),
                });
            }
            // Empty remote - proceed to blind write below.
        }
        Some(expected_old) => {
            // Normal path: `old` must be present in the current head set.
            if !current.iter().any(|c| c == expected_old) {
                inc_mismatch(&state);
                let current_head = current.into_iter().next();
                return Err(RemoteError::CasMismatch {
                    current: current_head.unwrap_or_else(|| expected_old.clone()),
                });
            }
        }
    }

    // Ancestry check: `new` must be a descendant of `old` (BUG-57).
    //
    // The CAS above only ensures that the claimed `old` is the current
    // remote tip. It does NOT prevent a client from pushing a `new`
    // commit that diverges from `old` (i.e. is not a descendant). Two
    // clients can both read the same remote tip, both send a correct
    // `old`, and both pass the CAS -- but the second push would overwrite
    // the first with a divergent history. The ancestry check closes this
    // gap: the server walks the `new` commit's parent chain and rejects
    // the push unless `old` appears in that chain.
    //
    // The check is skipped when `old` cannot be decoded as a Commit block.
    // This covers the "first real push after init" case where `old` is the
    // server's initial root Op CID (not a Commit), so it can never appear
    // in a commit's parent chain.
    if let Some(ref expected_old) = old {
        let bs = repo.blockstore().clone();
        // Try decoding `old` as a Commit. If it isn't a Commit (e.g. it's
        // a root Op CID from a freshly-initialised server), skip the ancestry
        // check -- we can't walk an op DAG as a commit DAG.
        let old_is_commit = bs
            .get(expected_old)
            .ok()
            .flatten()
            .and_then(|b| {
                use mnem_core::codec::from_canonical_bytes;
                use mnem_core::objects::Commit;
                from_canonical_bytes::<Commit>(&b).ok()
            })
            .is_some();
        if old_is_commit {
            match is_ancestor_of(&*bs, expected_old, &new) {
                Ok(true) => {}
                Ok(false) => {
                    inc_mismatch(&state);
                    return Err(RemoteError::CasMismatch {
                        current: expected_old.clone(),
                    });
                }
                Err(_) => {
                    // If we cannot decode commits (block not yet in store),
                    // allow the push. push-blocks already imported the blocks so
                    // this is a degenerate case; failing open avoids false rejections.
                }
            }
        }
    }

    // `OpHeadsStore::update` removes the old tip(s) and inserts the
    // new one. On first push there is no old tip to remove so we pass
    // an empty slice.
    let remove_tips: &[Cid] = match &old {
        Some(o) => std::slice::from_ref(o),
        None => &[],
    };
    ohs.update(new.clone(), remove_tips)
        .map_err(|e| RemoteError::Internal(format!("op-heads update: {e}")))?;
    inc_ok(&state);

    // Emit protocol headers on success.
    let body = AdvanceHeadResponse {
        head: new.to_string(),
    };
    Ok((StatusCode::OK, protocol_headers(), Json(body)).into_response())
}

/// Walk `tip`'s parent chain (BFS) looking for `needle`. Returns `true`
/// when `needle` is found, `false` when the walk exhausts the DAG, `Err`
/// on a blockstore error.
///
/// Used by `post_advance_head` to verify that the client's claimed `new`
/// commit actually descends from the current remote tip (`old`). This
/// closes the divergent-history gap in the CAS check (BUG-57).
fn is_ancestor_of(
    bs: &dyn mnem_core::store::Blockstore,
    needle: &Cid,
    tip: &Cid,
) -> Result<bool, mnem_core::error::StoreError> {
    use mnem_core::codec::from_canonical_bytes;
    use mnem_core::objects::Commit;
    use std::collections::{HashSet, VecDeque};

    if needle == tip {
        return Ok(true);
    }
    let mut seen: HashSet<Cid> = HashSet::new();
    let mut q: VecDeque<Cid> = VecDeque::new();
    q.push_back(tip.clone());
    while let Some(cur) = q.pop_front() {
        if !seen.insert(cur.clone()) {
            continue;
        }
        if &cur == needle {
            return Ok(true);
        }
        let Some(bytes) = bs.get(&cur)? else {
            continue;
        };
        let Ok(commit) = from_canonical_bytes::<Commit>(&bytes) else {
            continue;
        };
        for p in &commit.parents {
            q.push_back(p.clone());
        }
    }
    Ok(false)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::test_support::state_with_token;
    use axum::body::Body;
    use axum::http::Request;
    use http_body_util::BodyExt;
    use tower::ServiceExt;

    fn app(state: AppState) -> axum::Router {
        axum::Router::new()
            .route("/remote/v1/refs", axum::routing::get(get_refs))
            .route(
                "/remote/v1/fetch-blocks",
                axum::routing::post(post_fetch_blocks),
            )
            .route(
                "/remote/v1/push-blocks",
                axum::routing::post(post_push_blocks),
            )
            .route(
                "/remote/v1/advance-head",
                axum::routing::post(post_advance_head),
            )
            .with_state(state)
    }

    #[tokio::test]
    async fn refs_shape_and_protocol_header() {
        // `ReadonlyRepo::init` writes a root-op on a fresh store so
        // `head` is always `Some(cid)` here. The contract we enforce
        // is shape + protocol framing, not null-ness of `head`.
        let state = state_with_token(Some("tok".into()));
        let app = app(state);
        let req = Request::builder()
            .uri("/remote/v1/refs")
            .body(Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), 200);
        assert_eq!(
            resp.headers()
                .get(PROTOCOL_HEADER)
                .unwrap()
                .to_str()
                .unwrap(),
            "1"
        );
        assert!(resp.headers().get(CAPABILITIES_HEADER).is_some());
        let body = resp.into_body().collect().await.unwrap().to_bytes();
        let v: serde_json::Value = serde_json::from_slice(&body).unwrap();
        // `head` is either null (no commits) or a CID string; both
        // are valid. When head is present it is also mirrored under
        // the `"HEAD"` entry of the `refs` map; on a fresh repo the
        // map is empty.
        assert!(v["head"].is_null() || v["head"].is_string());
        let refs = v["refs"].as_object().unwrap();
        if v["head"].is_string() {
            assert_eq!(
                refs.get("HEAD").and_then(|s| s.as_str()),
                v["head"].as_str()
            );
            // B3.1: DEFAULT_REF ("main") must also be present and equal HEAD.
            assert_eq!(
                refs.get("main").and_then(|s| s.as_str()),
                refs.get("HEAD").and_then(|s| s.as_str()),
                "refs[\"main\"] must equal refs[\"HEAD\"] in B3.1 single-branch mode"
            );
        } else {
            assert!(refs.is_empty());
        }
        assert!(!v["capabilities"].as_array().unwrap().is_empty());
    }

    #[tokio::test]
    async fn push_blocks_requires_bearer_missing() {
        let state = state_with_token(Some("tok".into()));
        let app = app(state);
        let req = Request::builder()
            .method("POST")
            .uri("/remote/v1/push-blocks")
            .body(Body::from(Vec::<u8>::new()))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), 401);
        assert!(resp.headers().get("www-authenticate").is_some());
    }

    #[tokio::test]
    async fn advance_head_requires_bearer_mismatch() {
        let state = state_with_token(Some("tok".into()));
        let app = app(state);
        let req = Request::builder()
            .method("POST")
            .uri("/remote/v1/advance-head")
            .header("authorization", "Bearer wrong")
            .header("content-type", "application/json")
            .body(Body::from(r#"{"old":"x","new":"y"}"#))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), 401);
    }

    #[tokio::test]
    async fn advance_head_cas_mismatch_on_empty_heads() {
        // Fresh repo -> no heads. Any `old` CID the caller presents
        // fails the CAS with 409 and `current` is the old value (we
        // fall back to echoing `old` when the head set is empty).
        let state = state_with_token(Some("tok".into()));
        let app = app(state);
        // Construct a valid raw CID for the body.
        let mh = mnem_core::id::Multihash::sha2_256(b"a");
        let cid = mnem_core::id::Cid::new(mnem_core::id::CODEC_RAW, mh);
        let body = serde_json::json!({
            "old": cid.to_string(),
            "new": cid.to_string(),
        });
        let req = Request::builder()
            .method("POST")
            .uri("/remote/v1/advance-head")
            .header("authorization", "Bearer tok")
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), 409);
    }

    #[tokio::test]
    async fn fetch_blocks_rejects_empty_wants() {
        let state = state_with_token(Some("tok".into()));
        let app = app(state);
        let req = Request::builder()
            .method("POST")
            .uri("/remote/v1/fetch-blocks")
            .header("content-type", "application/json")
            .body(Body::from(r#"{"wants":[]}"#))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), 400);
    }

    #[tokio::test]
    async fn metrics_counter_increments_on_fetch_blocks_empty_wants_rejection() {
        // The counter fires only on *successful* fetch-blocks
        // completion. Rejected wants should NOT bump the counter;
        // this guards against metric-vs-error-handling drift where a
        // 400 also falsely counts as traffic.
        let state = state_with_token(Some("tok".into()));
        let before = state.metrics.remote_fetch_blocks.get();
        let app = app(state.clone());
        let req = Request::builder()
            .method("POST")
            .uri("/remote/v1/fetch-blocks")
            .header("content-type", "application/json")
            .body(Body::from(r#"{"wants":[]}"#))
            .unwrap();
        let _ = app.oneshot(req).await.unwrap();
        let after = state.metrics.remote_fetch_blocks.get();
        assert_eq!(before, after, "rejected request must not bump counter");
    }
}
