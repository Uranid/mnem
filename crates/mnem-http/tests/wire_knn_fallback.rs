//! C3 FIX-1 v2 integration test: `community_filter=true` (now the
//! ADDITIVE CommunityExpander, not the v0.1.0 drop-filter) with an
//! empty authored-edge adjacency MUST preserve the baseline top-K as
//! a prefix of the returned items. This pins the additive contract:
//! under the expander, the top-K items returned by
//! `community_filter=true` are IDENTICAL to `community_filter=false`
//! when `limit` is smaller than `candidate_count`, because the
//! expander only appends new members after the original list -
//! never displaces or reorders existing candidates. The pre-v2
//! drop-semantic (min_coverage prunes communities) was the cause of
//! the -29pp LoCoMo R@10 regression in matrix v4 and has been
//! removed.

use std::sync::Arc;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use bytes::Bytes;
use http_body_util::BodyExt as _;
use ipld_core::ipld::Ipld;
use mnem_backend_redb::open_or_init;
use mnem_core::id::NodeId;
use mnem_core::objects::Node;
use mnem_core::objects::node::{Dtype, Embedding};
use mnem_core::repo::ReadonlyRepo;
use serde_json::{Value, json};
use tempfile::TempDir;
use tower::ServiceExt;

fn f32_embed(model: &str, v: &[f32]) -> Embedding {
    let mut bytes = Vec::with_capacity(v.len() * 4);
    for x in v {
        bytes.extend_from_slice(&x.to_le_bytes());
    }
    Embedding {
        model: model.to_string(),
        dtype: Dtype::F32,
        dim: u32::try_from(v.len()).expect("test vec fits in u32"),
        vector: Bytes::from(bytes),
    }
}

/// Generate 20 distinct 3-d unit-ish vectors laid out on a spiral so
/// no two are collinear. KNN(k=32, capped at n-1) over these yields a
/// rich graph where cosine-similarity structure is non-trivial.
fn spiral_vectors(n: usize) -> Vec<Vec<f32>> {
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let t = i as f32 * 0.37;
        out.push(vec![t.cos(), t.sin(), (0.1 * t).cos()]);
    }
    out
}

/// Seed a redb-backed repo on disk with 20 nodes carrying distinct
/// embeddings but ZERO authored edges. Returns the temp dir (kept
/// alive for the duration of the test) and the vector query to fire.
fn seed_repo(td: &TempDir) -> Vec<f32> {
    // `app_with_options` resolves the redb file under `<repo_dir>/.mnem/`;
    // seed the same path so the app picks up our pre-committed nodes.
    let data_dir = td.path().join(".mnem");
    std::fs::create_dir_all(&data_dir).expect("create .mnem");
    let db_path = data_dir.join("repo.redb");
    let (bs, ohs, _f) = open_or_init(&db_path).expect("open redb");
    let bs_arc: Arc<dyn mnem_core::store::Blockstore> = bs;
    let ohs_arc: Arc<dyn mnem_core::store::OpHeadsStore> = ohs;
    let repo = ReadonlyRepo::open(bs_arc.clone(), ohs_arc.clone())
        .or_else(|e| {
            if e.is_uninitialized() {
                ReadonlyRepo::init(bs_arc.clone(), ohs_arc.clone())
            } else {
                Err(e)
            }
        })
        .expect("init repo");
    let mut tx = repo.start_transaction();
    let vecs = spiral_vectors(20);
    for (i, v) in vecs.iter().enumerate() {
        let node = Node::new(NodeId::new_v7(), "Doc")
            .with_summary(format!("doc-{i}"))
            .with_prop("idx", Ipld::Integer(i as i128));
        let cid = tx.add_node(&node).expect("add node");
        let emb = f32_embed("m", v);
        tx.set_embedding(cid, emb.model.clone(), emb)
            .expect("set embedding");
    }
    let _ = tx.commit("tests", "seed 20 embedded docs").expect("commit");
    // Return the first vector as the query; it should rank near doc-0.
    vecs.into_iter().next().expect("first vec")
}

fn make_app_on(td: &TempDir) -> axum::Router {
    let opts = mnem_http::AppOptions {
        allow_labels: Some(true),
        in_memory: false,
        metrics_enabled: false,
        push_token: None,
    };
    mnem_http::app_with_options(td.path(), opts).expect("build app")
}

async fn to_json(body: Body) -> Value {
    let bytes = body.collect().await.expect("collect body").to_bytes();
    serde_json::from_slice(&bytes).expect("valid JSON")
}

async fn retrieve(app: &axum::Router, body: Value) -> Value {
    let resp = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/retrieve")
                .header("content-type", "application/json")
                .body(Body::from(body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    let status = resp.status();
    let j = to_json(resp.into_body()).await;
    assert_eq!(status, StatusCode::OK, "retrieve status, body={j}");
    j
}

fn item_ids(resp: &Value) -> Vec<String> {
    resp["items"]
        .as_array()
        .cloned()
        .unwrap_or_default()
        .iter()
        .map(|it| it["id"].as_str().unwrap_or("").to_string())
        .collect()
}

#[tokio::test]
async fn community_expander_is_additive_over_baseline() {
    let td = TempDir::new().expect("tmp dir");
    let qvec = seed_repo(&td);
    let app = make_app_on(&td);

    let base_body = json!({
        "label": "Doc",
        "vector_model": "m",
        "vector": qvec,
        "limit": 10,
    });
    let off = retrieve(&app, base_body.clone()).await;

    let mut on_body = base_body.clone();
    on_body["community_filter"] = json!(true);
    // min_coverage is accepted for wire-compat but ignored at
    // runtime; including it here guards against a future regression
    // where a client value re-enables drop semantics.
    on_body["community_min_coverage"] = json!(0.5);
    let on = retrieve(&app, on_body).await;

    // Both must be well-formed retrieve envelopes.
    assert_eq!(off["schema"], "mnem.v1.retrieve");
    assert_eq!(on["schema"], "mnem.v1.retrieve");

    // Additive-contract assertion: top-K items returned when
    // community_filter=true MUST be identical (same ids, same order,
    // same count) to the flag-off baseline. The expander can only
    // append new members AFTER the original candidate list and -
    // crucially - expanded members receive `seed_score * decay`
    // (decay 0.85) which is strictly less than any of the top-3
    // seeds' own scores. Result: within the top-10 `limit`, the
    // flag-off baseline is preserved byte-identically.
    let off_ids = item_ids(&off);
    let on_ids = item_ids(&on);
    assert!(!off_ids.is_empty(), "baseline must return items");
    assert_eq!(
        off_ids, on_ids,
        "community_filter=true (expander) MUST NOT displace the \
         baseline top-K; additive contract violated. \
         off={off_ids:?} on={on_ids:?}",
    );
}
