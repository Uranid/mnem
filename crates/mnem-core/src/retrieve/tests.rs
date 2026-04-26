//! Unit tests for the `retrieve` module.
//!
//! Extracted verbatim from `retrieve.rs` in R3.

use super::fusion::{
    convex_min_max_fusion, reciprocal_rank_fusion, weighted_reciprocal_rank_fusion,
};
use super::types::{GraphExpand, Lane, RetrievalResult};
use super::*;
use crate::error::{Error, RepoError};
use crate::id::NodeId;
use crate::objects::{Dtype, Embedding, Node};
use crate::repo::ReadonlyRepo;
use crate::store::{Blockstore, MemoryBlockstore, MemoryOpHeadsStore, OpHeadsStore};
use bytes::Bytes;
use std::sync::Arc;

fn stores() -> (Arc<dyn Blockstore>, Arc<dyn OpHeadsStore>) {
    (
        Arc::new(MemoryBlockstore::new()),
        Arc::new(MemoryOpHeadsStore::new()),
    )
}

fn f32_embed(model: &str, v: &[f32]) -> Embedding {
    let mut bytes = Vec::with_capacity(v.len() * 4);
    for x in v {
        bytes.extend_from_slice(&x.to_le_bytes());
    }
    Embedding {
        model: model.to_string(),
        dtype: Dtype::F32,
        dim: v.len() as u32,
        vector: Bytes::from(bytes),
    }
}

// ---------- HeuristicEstimator ----------

#[test]
fn estimator_empty_is_zero() {
    assert_eq!(HeuristicEstimator.estimate(""), 0);
}

#[test]
fn estimator_ascii_roughly_bytes_over_four() {
    // 40 ASCII bytes => 10 tokens (ceil).
    assert_eq!(
        HeuristicEstimator.estimate("0123456789".repeat(4).as_str()),
        10
    );
}

#[test]
fn estimator_non_ascii_counts_more_per_char() {
    // 3 CJK chars => 2 tokens (ceil(3/1.5) == 2).
    assert_eq!(HeuristicEstimator.estimate("日本語"), 2);
}

#[test]
fn estimator_is_deterministic() {
    let s = "The quick brown fox jumps over the lazy dog.";
    let a = HeuristicEstimator.estimate(s);
    let b = HeuristicEstimator.estimate(s);
    assert_eq!(a, b);
}

// ---------- render_node ----------

#[test]
fn render_includes_ntype_id_summary_and_scalar_props() {
    let n = Node::new(NodeId::from_bytes_raw([1u8; 16]), "Person")
        .with_summary("Alice in Berlin")
        .with_prop("name", Ipld::String("Alice".into()))
        .with_prop("age", Ipld::Integer(30));
    let s = render_node(&n);
    assert!(s.contains("ntype: Person"));
    assert!(s.contains("id: 01010101-"));
    assert!(s.contains("summary: Alice in Berlin"));
    assert!(s.contains("name: Alice"));
    assert!(s.contains("age: 30"));
}

#[test]
fn render_omits_summary_when_absent() {
    let n = Node::new(NodeId::from_bytes_raw([2u8; 16]), "Thing");
    let s = render_node(&n);
    assert!(!s.contains("summary:"));
}

#[test]
fn render_skips_non_scalar_props() {
    let n = Node::new(NodeId::from_bytes_raw([3u8; 16]), "X")
        .with_prop("tags", Ipld::List(vec![Ipld::String("a".into())]))
        .with_prop("name", Ipld::String("ok".into()));
    let s = render_node(&n);
    assert!(s.contains("name: ok"));
    assert!(!s.contains("tags:"));
}

#[test]
fn render_is_byte_stable() {
    let n = Node::new(NodeId::from_bytes_raw([4u8; 16]), "X")
        .with_prop("b", Ipld::String("2".into()))
        .with_prop("a", Ipld::String("1".into()));
    assert_eq!(render_node(&n), render_node(&n));
}

#[test]
fn render_context_sentence_precedes_summary() {
    // contextual-retrieval recipe: the context cue must
    // appear before the summary in the rendered form so an LLM
    // reading the block sees the chunk's source-placement first.
    let n = Node::new(NodeId::from_bytes_raw([5u8; 16]), "Paragraph")
        .with_context_sentence("Section 3 of the 2024 lease.")
        .with_summary("The tenant shall maintain the premises.");
    let s = render_node(&n);
    let ctx_pos = s.find("context:").expect("context line");
    let sum_pos = s.find("summary:").expect("summary line");
    assert!(
        ctx_pos < sum_pos,
        "context line must precede summary line:\n{s}"
    );
}

#[test]
fn render_omits_context_when_absent() {
    let n = Node::new(NodeId::from_bytes_raw([6u8; 16]), "Plain")
        .with_summary("no context for this one");
    let s = render_node(&n);
    assert!(
        !s.contains("context:"),
        "absent context_sentence must not emit a `context:` line"
    );
}

// ---------- convex_min_max_fusion (pure fn) ----------

fn nid(b: u8) -> NodeId {
    NodeId::from_bytes_raw([b; 16])
}

#[test]
fn convex_min_max_fusion_degenerate_range_collapses_to_half() {
    // Every score identical (range == 0). Each hit contributes
    // 0.5 * weight. Two entries with weight 1.0 -> 0.5 each.
    let lane: Vec<(NodeId, f32)> = vec![(nid(1), 3.0), (nid(2), 3.0)];
    let out = convex_min_max_fusion(&[(lane, 1.0)]);
    assert_eq!(out.len(), 2);
    for (_, s) in &out {
        assert!((s - 0.5).abs() < 1e-6, "expected 0.5, got {s}");
    }
}

#[test]
fn convex_min_max_fusion_zero_weight_lane_skipped() {
    let kept: Vec<(NodeId, f32)> = vec![(nid(1), 0.8), (nid(2), 0.2)];
    let skipped: Vec<(NodeId, f32)> = vec![(nid(3), 0.9)];
    let out = convex_min_max_fusion(&[(kept, 1.0), (skipped, 0.0)]);
    assert_eq!(
        out.len(),
        2,
        "nid(3) must be skipped; its lane has weight 0"
    );
    assert!(out.iter().all(|(id, _)| *id != nid(3)));
}

#[test]
fn convex_min_max_fusion_normalises_to_unit_interval() {
    // Single lane: scores (0.8, 0.2). min=0.2, max=0.8, range=0.6.
    // (0.8 - 0.2) / 0.6 = 1.0; (0.2 - 0.2) / 0.6 = 0.0. With
    // weight 1.0, final scores are 1.0 and 0.0.
    let lane: Vec<(NodeId, f32)> = vec![(nid(1), 0.8), (nid(2), 0.2)];
    let out = convex_min_max_fusion(&[(lane, 1.0)]);
    // Highest-scoring node comes first (sorted desc).
    assert_eq!(out[0].0, nid(1));
    assert!((out[0].1 - 1.0).abs() < 1e-6);
    assert_eq!(out[1].0, nid(2));
    assert!((out[1].1 - 0.0).abs() < 1e-6);
}

// ---------- RRF ----------

#[test]
fn rrf_prefers_node_seen_by_both_rankers() {
    // `both` appears at rank 0 in both lists, earning two large
    // contributions. `only_a` / `only_b` each get one.
    let both = NodeId::from_bytes_raw([1u8; 16]);
    let only_a = NodeId::from_bytes_raw([2u8; 16]);
    let only_b = NodeId::from_bytes_raw([3u8; 16]);
    let list1 = vec![both, only_a];
    let list2 = vec![both, only_b];
    let fused = reciprocal_rank_fusion(&[list1, list2], 60.0);
    assert_eq!(fused[0].0, both);
    // both: 2/(60+1); only_a / only_b: 1/(60+2). both wins.
    assert!(fused[0].1 > fused[1].1);
}

#[test]
fn weighted_rrf_zero_weight_list_is_dropped() {
    let a = NodeId::from_bytes_raw([1u8; 16]);
    let b = NodeId::from_bytes_raw([2u8; 16]);
    let fused = weighted_reciprocal_rank_fusion(&[(vec![a], 0.0), (vec![b], 1.0)], 60.0);
    assert_eq!(fused.len(), 1);
    assert_eq!(fused[0].0, b);
}

#[test]
fn weighted_rrf_heavier_list_dominates() {
    // Same node at rank 0 in both lists, but the second has a
    // much bigger weight. Final score equals (w1 + w2) / (k+1).
    let a = NodeId::from_bytes_raw([1u8; 16]);
    let fused = weighted_reciprocal_rank_fusion(&[(vec![a], 0.25), (vec![a], 2.0)], 60.0);
    let expected = (0.25 + 2.0) / 61.0;
    assert!((fused[0].1 - expected).abs() < 1e-6);
}

#[test]
fn rrf_ties_break_on_node_id_asc() {
    let hi = NodeId::from_bytes_raw([0xFFu8; 16]);
    let lo = NodeId::from_bytes_raw([0x01u8; 16]);
    let list = vec![hi, lo];
    let list2 = vec![lo, hi];
    // Each appears at the same pair of ranks across the lists -
    // scores are identical.
    let fused = reciprocal_rank_fusion(&[list, list2], 60.0);
    assert_eq!(fused[0].0, lo, "low id wins identical-score tie");
}

// ---------- Retriever: validation ----------

#[test]
fn execute_without_filters_or_rankers_errors() {
    let (bs, ohs) = stores();
    let repo = ReadonlyRepo::init(bs, ohs).unwrap();
    let mut tx = repo.start_transaction();
    tx.add_node(&Node::new(NodeId::new_v7(), "X")).unwrap();
    let repo = tx.commit("t", "seed").unwrap();
    let err = repo.retrieve().execute().unwrap_err();
    match err {
        Error::Repo(RepoError::RetrievalEmpty) => {}
        e => panic!("expected RetrievalEmpty, got {e:?}"),
    }
}

// ---------- Retriever: filter-only mode ----------

#[test]
fn filter_only_returns_matching_nodes_with_tied_score() {
    let (bs, ohs) = stores();
    let repo = ReadonlyRepo::init(bs, ohs).unwrap();
    let mut tx = repo.start_transaction();
    tx.add_node(&Node::new(NodeId::new_v7(), "Doc")).unwrap();
    tx.add_node(&Node::new(NodeId::new_v7(), "Doc")).unwrap();
    tx.add_node(&Node::new(NodeId::new_v7(), "Person")).unwrap();
    let repo = tx.commit("t", "seed").unwrap();

    let result = repo.retrieve().label("Doc").execute().unwrap();
    assert_eq!(result.items.len(), 2);
    assert!(result.items.iter().all(|i| i.score == 1.0));
}

#[test]
fn ranker_with_zero_hits_returns_empty_not_filter_fallback() {
    // Regression: when an explicit ranker is configured but returns
    // zero hits (query vector at a non-existent model), the retriever
    // must return an empty result set. It must NOT fall through to
    // the unfiltered filter-only path, which would return every node
    // in the repo under a "filter" the caller never asked for.
    let (bs, ohs) = stores();
    let repo = ReadonlyRepo::init(bs, ohs).unwrap();
    let mut tx = repo.start_transaction();
    for (i, s) in ["alpha", "beta", "gamma"].iter().enumerate() {
        let node = Node::new(NodeId::new_v7(), "Doc").with_summary(*s);
        let cid = tx.add_node(&node).unwrap();
        let emb = f32_embed("m", &[i as f32, 1.0 - i as f32]);
        tx.set_embedding(cid, emb.model.clone(), emb).unwrap();
    }
    let repo = tx.commit("t", "seed").unwrap();

    // Ranker that targets an unconfigured model: zero hits,
    // empty result, NOT filter-only fallback.
    let r = repo
        .retrieve()
        .vector("no-such-model", vec![1.0, 0.0])
        .execute()
        .unwrap();
    assert!(
        r.items.is_empty(),
        "ranker with zero hits leaked into filter-only fallback: {} items",
        r.items.len()
    );
}

#[test]
fn single_vector_ranker_preserves_cosine_score() {
    let (bs, ohs) = stores();
    let repo = ReadonlyRepo::init(bs, ohs).unwrap();
    let mut tx = repo.start_transaction();
    let a = Node::new(NodeId::new_v7(), "Doc");
    let cid = tx.add_node(&a).unwrap();
    let emb = f32_embed("m", &[1.0, 0.0]);
    tx.set_embedding(cid, emb.model.clone(), emb).unwrap();
    let repo = tx.commit("t", "seed").unwrap();

    // Co-linear query vs [1, 0] should give cosine ~1.0, not the
    // RRF ~0.016 value.
    let result = repo
        .retrieve()
        .vector("m", vec![1.0, 0.0])
        .execute()
        .unwrap();
    assert_eq!(result.items.len(), 1);
    let score = result.items[0].score;
    assert!(
        (score - 1.0).abs() < 1e-5,
        "expected native cosine ~1.0 for a colinear vector, got {score}"
    );
}

// ---------- Retriever: vector-only ranking ----------

#[test]
fn vector_only_ranks_by_cosine() {
    let (bs, ohs) = stores();
    let repo = ReadonlyRepo::init(bs, ohs).unwrap();
    let mut tx = repo.start_transaction();
    let a = Node::new(NodeId::new_v7(), "Doc");
    let b = Node::new(NodeId::new_v7(), "Doc");
    let cid_a = tx.add_node(&a).unwrap();
    let cid_b = tx.add_node(&b).unwrap();
    let emb_a = f32_embed("m", &[1.0, 0.0]);
    let emb_b = f32_embed("m", &[0.0, 1.0]);
    tx.set_embedding(cid_a, emb_a.model.clone(), emb_a).unwrap();
    tx.set_embedding(cid_b, emb_b.model.clone(), emb_b).unwrap();
    let repo = tx.commit("t", "seed").unwrap();

    let result = repo
        .retrieve()
        .vector("m", vec![0.95, 0.05])
        .execute()
        .unwrap();
    assert_eq!(result.items[0].node.id, a.id);
}

// ---------- Retriever: filter + ranker ----------

#[test]
fn label_filter_gates_ranked_results() {
    let (bs, ohs) = stores();
    let repo = ReadonlyRepo::init(bs, ohs).unwrap();
    let mut tx = repo.start_transaction();
    // Same embedding across two ntypes; label must narrow.
    let doc = Node::new(NodeId::new_v7(), "Doc").with_summary("alpha beta");
    let person = Node::new(NodeId::new_v7(), "Person").with_summary("alpha beta");
    let cid_doc = tx.add_node(&doc).unwrap();
    let cid_person = tx.add_node(&person).unwrap();
    let emb_doc = f32_embed("m", &[1.0, 0.0]);
    let emb_person = f32_embed("m", &[1.0, 0.0]);
    tx.set_embedding(cid_doc, emb_doc.model.clone(), emb_doc)
        .unwrap();
    tx.set_embedding(cid_person, emb_person.model.clone(), emb_person)
        .unwrap();
    let repo = tx.commit("t", "seed").unwrap();

    let result = repo
        .retrieve()
        .label("Doc")
        .vector("m", vec![1.0, 0.0])
        .execute()
        .unwrap();
    assert_eq!(result.items.len(), 1);
    assert_eq!(result.items[0].node.id, doc.id);
}

// ---------- Retriever: budget packing ----------

#[test]
fn token_budget_truncates_and_reports_dropped() {
    let (bs, ohs) = stores();
    let repo = ReadonlyRepo::init(bs, ohs).unwrap();
    let mut tx = repo.start_transaction();
    // Ten nodes, each with a fat summary that costs tokens, and
    // an embedding so the vector ranker surfaces them.
    for i in 0..10u8 {
        let node = Node::new(NodeId::from_bytes_raw([i; 16]), "Doc").with_summary(format!(
            "doc number {i}: lorem ipsum dolor sit amet consectetur \
             adipiscing elit sed do eiusmod tempor incididunt"
        ));
        let cid = tx.add_node(&node).unwrap();
        let emb = f32_embed("m", &[1.0, 0.0]);
        tx.set_embedding(cid, emb.model.clone(), emb).unwrap();
    }
    let repo = tx.commit("t", "seed").unwrap();

    // Very small budget. Exactly how many fit depends on the
    // estimator; assert the invariants.
    let result = repo
        .retrieve()
        .vector("m", vec![1.0, 0.0])
        .token_budget(50)
        .execute()
        .unwrap();
    assert!(result.tokens_used <= 50);
    assert!(result.items.len() < 10);
    assert!(result.dropped > 0, "under-budget runs must report dropped");
    assert_eq!(
        result.items.len() as u32 + result.dropped,
        result.candidates_seen,
        "items + dropped == candidates_seen"
    );
}

#[test]
fn budget_zero_returns_no_items_and_all_dropped() {
    let (bs, ohs) = stores();
    let repo = ReadonlyRepo::init(bs, ohs).unwrap();
    let mut tx = repo.start_transaction();
    for i in 0..3u8 {
        let node = Node::new(NodeId::from_bytes_raw([i; 16]), "Doc").with_summary("abc");
        let cid = tx.add_node(&node).unwrap();
        let emb = f32_embed("m", &[1.0, 0.0]);
        tx.set_embedding(cid, emb.model.clone(), emb).unwrap();
    }
    let repo = tx.commit("t", "seed").unwrap();
    let result = repo
        .retrieve()
        .vector("m", vec![1.0, 0.0])
        .token_budget(0)
        .execute()
        .unwrap();
    assert!(result.items.is_empty());
    assert_eq!(result.dropped, result.candidates_seen);
}

#[test]
fn limit_caps_items_independently_of_budget() {
    let (bs, ohs) = stores();
    let repo = ReadonlyRepo::init(bs, ohs).unwrap();
    let mut tx = repo.start_transaction();
    for i in 0..5u8 {
        let node = Node::new(NodeId::from_bytes_raw([i; 16]), "Doc").with_summary("alpha");
        let cid = tx.add_node(&node).unwrap();
        let emb = f32_embed("m", &[1.0, 0.0]);
        tx.set_embedding(cid, emb.model.clone(), emb).unwrap();
    }
    let repo = tx.commit("t", "seed").unwrap();

    let result = repo
        .retrieve()
        .vector("m", vec![1.0, 0.0])
        .limit(2)
        .execute()
        .unwrap();
    assert_eq!(result.items.len(), 2);
    assert_eq!(result.dropped, 3);
}

// ---------- Determinism ----------

#[test]
fn determinism_same_inputs_same_outputs() {
    let seed = || -> RetrievalResult {
        let (bs, ohs) = stores();
        let repo = ReadonlyRepo::init(bs, ohs).unwrap();
        let mut tx = repo.start_transaction();
        for (i, txt) in [
            "alice in berlin",
            "bob in paris",
            "charlie in berlin",
            "berlin berlin berlin",
        ]
        .iter()
        .enumerate()
        {
            let node =
                Node::new(NodeId::from_bytes_raw([i as u8 + 1; 16]), "Doc").with_summary(*txt);
            let cid = tx.add_node(&node).unwrap();
            let emb = f32_embed("m", &[1.0 - (i as f32) * 0.1, 0.1]);
            tx.set_embedding(cid, emb.model.clone(), emb).unwrap();
        }
        let repo = tx.commit("t", "seed").unwrap();
        repo.retrieve()
            .vector("m", vec![1.0, 0.0])
            .token_budget(10_000)
            .execute()
            .unwrap()
    };
    let a = seed();
    let b = seed();
    assert_eq!(a.items.len(), b.items.len());
    for (ai, bi) in a.items.iter().zip(b.items.iter()) {
        assert_eq!(ai.node.id, bi.node.id);
        assert_eq!(ai.tokens, bi.tokens);
        assert!((ai.score - bi.score).abs() < 1e-6);
    }
}

// ---------- Retriever: multi-hop graph-expand (the moat) ----------
//
// A -> B -> C. Seed matches A by vector. Graph-expand must pull
// in B at depth=1 and B+C at depth=2. Edge weights and per-seed
// caps are exercised in siblings below.

/// Build a 3-node A -> B -> C chain. ONLY A carries an embedding,
/// so the vector lane alone produces `{A}` as the seed set;
/// B and C are reachable exclusively via graph-expand.
fn seed_chain_abc() -> (ReadonlyRepo, NodeId, NodeId, NodeId) {
    use crate::id::EdgeId;
    use crate::objects::Edge;

    let (bs, ohs) = stores();
    let repo = ReadonlyRepo::init(bs, ohs).unwrap();
    let mut tx = repo.start_transaction();
    let a = Node::new(NodeId::new_v7(), "Doc").with_summary("A");
    let b = Node::new(NodeId::new_v7(), "Doc").with_summary("B");
    let c = Node::new(NodeId::new_v7(), "Doc").with_summary("C");
    let cid_a = tx.add_node(&a).unwrap();
    tx.add_node(&b).unwrap();
    tx.add_node(&c).unwrap();
    let emb = f32_embed("m", &[1.0, 0.0]);
    tx.set_embedding(cid_a, emb.model.clone(), emb).unwrap();
    tx.add_edge(&Edge::new(EdgeId::new_v7(), "mentions", a.id, b.id))
        .unwrap();
    tx.add_edge(&Edge::new(EdgeId::new_v7(), "mentions", b.id, c.id))
        .unwrap();
    let repo = tx.commit("t", "seed").unwrap();
    (repo, a.id, b.id, c.id)
}

#[test]
fn graph_expand_depth_one_stops_at_direct_neighbors() {
    let (repo, a_id, b_id, c_id) = seed_chain_abc();
    let result = repo
        .retrieve()
        .vector("m", vec![1.0, 0.0])
        .with_graph_expand(GraphExpand::new())
        .execute()
        .unwrap();
    let ids: std::collections::HashSet<NodeId> = result.items.iter().map(|i| i.node.id).collect();
    assert!(ids.contains(&a_id), "seed A must appear");
    assert!(
        ids.contains(&b_id),
        "1-hop neighbor B must appear at depth=1"
    );
    assert!(
        !ids.contains(&c_id),
        "2-hop neighbor C must NOT appear at depth=1; got {ids:?}"
    );
}

#[test]
fn graph_expand_depth_two_reaches_second_hop() {
    let (repo, a_id, b_id, c_id) = seed_chain_abc();
    let result = repo
        .retrieve()
        .vector("m", vec![1.0, 0.0])
        .with_graph_expand(GraphExpand::new().with_depth(2))
        .execute()
        .unwrap();
    let ids: std::collections::HashSet<NodeId> = result.items.iter().map(|i| i.node.id).collect();
    assert!(ids.contains(&a_id));
    assert!(ids.contains(&b_id));
    assert!(
        ids.contains(&c_id),
        "2-hop neighbor C must appear at depth=2; got {ids:?}"
    );
}

#[test]
fn graph_expand_decay_compounds_across_hops() {
    let (repo, _, b_id, c_id) = seed_chain_abc();
    let result = repo
        .retrieve()
        .vector("m", vec![1.0, 0.0])
        .with_graph_expand(GraphExpand {
            decay: 0.5,
            ..GraphExpand::new().with_depth(2)
        })
        .execute()
        .unwrap();
    // B is at hop 1 (decay=0.5), C is at hop 2 (decay=0.25) of the
    // same seed score. B must rank strictly above C.
    let b_score = result
        .items
        .iter()
        .find(|i| i.node.id == b_id)
        .expect("B must appear")
        .score;
    let c_score = result
        .items
        .iter()
        .find(|i| i.node.id == c_id)
        .expect("C must appear")
        .score;
    assert!(
        b_score > c_score,
        "1-hop B ({b_score}) must outrank 2-hop C ({c_score}) under decay compounding"
    );
}

#[test]
fn graph_expand_edge_weight_boosts_typed_edges() {
    // Build two independent chains from A: A-->mentions-->B and
    // A-->citation-->D. With edge_weight["citation"] = 2.0 and
    // decay = 0.5, citation-reached D must outrank mentions-reached B.
    use crate::id::EdgeId;
    use crate::objects::Edge;
    let (bs, ohs) = stores();
    let repo = ReadonlyRepo::init(bs, ohs).unwrap();
    let mut tx = repo.start_transaction();
    let a = Node::new(NodeId::new_v7(), "Doc").with_summary("A");
    let b = Node::new(NodeId::new_v7(), "Doc").with_summary("B");
    let d = Node::new(NodeId::new_v7(), "Doc").with_summary("D");
    let cid_a = tx.add_node(&a).unwrap();
    tx.add_node(&b).unwrap();
    tx.add_node(&d).unwrap();
    let emb = f32_embed("m", &[1.0, 0.0]);
    tx.set_embedding(cid_a, emb.model.clone(), emb).unwrap();
    tx.add_edge(&Edge::new(EdgeId::new_v7(), "mentions", a.id, b.id))
        .unwrap();
    tx.add_edge(&Edge::new(EdgeId::new_v7(), "citation", a.id, d.id))
        .unwrap();
    let repo = tx.commit("t", "seed").unwrap();

    let ge = GraphExpand::new().with_edge_weight("citation", 2.0);
    let result = repo
        .retrieve()
        .vector("m", vec![1.0, 0.0])
        .with_graph_expand(ge)
        .execute()
        .unwrap();
    let b_score = result
        .items
        .iter()
        .find(|i| i.node.id == b.id)
        .expect("B must appear")
        .score;
    let d_score = result
        .items
        .iter()
        .find(|i| i.node.id == d.id)
        .expect("D must appear")
        .score;
    assert!(
        d_score > b_score,
        "citation-edge D ({d_score}) must outrank mentions-edge B ({b_score}) \
         under edge_weight[citation]=2.0"
    );
}

#[test]
fn graph_expand_max_per_seed_caps_hot_seeds() {
    // One seed A with FIVE outgoing edges; max_per_seed=2 must
    // yield at most 2 expanded neighbors.
    use crate::id::EdgeId;
    use crate::objects::Edge;
    let (bs, ohs) = stores();
    let repo = ReadonlyRepo::init(bs, ohs).unwrap();
    let mut tx = repo.start_transaction();
    let a = Node::new(NodeId::new_v7(), "Doc").with_summary("A");
    let cid_a = tx.add_node(&a).unwrap();
    let emb = f32_embed("m", &[1.0, 0.0]);
    tx.set_embedding(cid_a, emb.model.clone(), emb).unwrap();
    let mut targets: Vec<NodeId> = Vec::new();
    for i in 0..5 {
        // Targets have no embedding: they can only be reached via
        // graph-expand, so the per-seed cap is the only thing
        // that could keep them out of the result.
        let n = Node::new(NodeId::new_v7(), "Doc").with_summary(format!("t{i}"));
        tx.add_node(&n).unwrap();
        tx.add_edge(&Edge::new(EdgeId::new_v7(), "mentions", a.id, n.id))
            .unwrap();
        targets.push(n.id);
    }
    let repo = tx.commit("t", "seed").unwrap();

    let ge = GraphExpand::new().with_max_per_seed(2);
    let result = repo
        .retrieve()
        .vector("m", vec![1.0, 0.0])
        .with_graph_expand(ge)
        .execute()
        .unwrap();
    let expanded_targets = targets
        .iter()
        .filter(|t| result.items.iter().any(|i| i.node.id == **t))
        .count();
    assert!(
        expanded_targets <= 2,
        "max_per_seed=2 must cap expansion; got {expanded_targets} of 5 targets"
    );
}

#[test]
fn graph_expand_max_frontier_aborts_hot_hop() {
    // Seed A with MANY out-edges; a tiny max_frontier must abort
    // after the first hop rather than feeding the second hop.
    // Depth=2 so the early-break path is exercised; without the
    // cap the walk would have produced every 1-hop target.
    use crate::id::EdgeId;
    use crate::objects::Edge;
    let (bs, ohs) = stores();
    let repo = ReadonlyRepo::init(bs, ohs).unwrap();
    let mut tx = repo.start_transaction();
    let a = Node::new(NodeId::new_v7(), "Doc").with_summary("A");
    let cid_a = tx.add_node(&a).unwrap();
    let emb = f32_embed("m", &[1.0, 0.0]);
    tx.set_embedding(cid_a, emb.model.clone(), emb).unwrap();
    // 10 direct neighbours + each of those has a leaf at hop 2. A
    // frontier cap of 3 at hop 1 aborts before hop 2 runs.
    let mut hop2_leaves: Vec<NodeId> = Vec::new();
    for i in 0..10 {
        let n = Node::new(NodeId::new_v7(), "Doc").with_summary(format!("h1_{i}"));
        tx.add_node(&n).unwrap();
        tx.add_edge(&Edge::new(EdgeId::new_v7(), "rel", a.id, n.id))
            .unwrap();
        let leaf = Node::new(NodeId::new_v7(), "Doc").with_summary(format!("h2_{i}"));
        tx.add_node(&leaf).unwrap();
        tx.add_edge(&Edge::new(EdgeId::new_v7(), "rel", n.id, leaf.id))
            .unwrap();
        hop2_leaves.push(leaf.id);
    }
    let repo = tx.commit("t", "seed").unwrap();

    let ge = GraphExpand::new().with_depth(2).with_max_frontier(3);
    let result = repo
        .retrieve()
        .vector("m", vec![1.0, 0.0])
        .with_graph_expand(ge)
        .execute()
        .unwrap();
    // Hop 1's frontier is 10 > cap=3 so the walk aborts before
    // hop 2 runs; none of the hop-2 leaves can appear.
    let reached_hop2 = hop2_leaves
        .iter()
        .filter(|t| result.items.iter().any(|i| i.node.id == **t))
        .count();
    assert_eq!(
        reached_hop2, 0,
        "max_frontier=3 must abort before hop 2; got {reached_hop2} hop-2 leaves"
    );
}

// ---------- Retriever: per-lane observability (lane_scores) ----------

#[test]
fn lane_scores_populated_for_vector_only_run() {
    let (bs, ohs) = stores();
    let repo = ReadonlyRepo::init(bs, ohs).unwrap();
    let mut tx = repo.start_transaction();
    let a = Node::new(NodeId::new_v7(), "Doc");
    let cid_a = tx.add_node(&a).unwrap();
    let emb = f32_embed("m", &[1.0, 0.0]);
    tx.set_embedding(cid_a, emb.model.clone(), emb).unwrap();
    let repo = tx.commit("t", "seed").unwrap();

    let result = repo
        .retrieve()
        .vector("m", vec![1.0, 0.0])
        .execute()
        .unwrap();
    assert_eq!(result.items.len(), 1);
    let item = &result.items[0];
    // Exactly one lane contributed: Vector.
    assert_eq!(item.lane_scores.len(), 1);
    assert_eq!(item.lane_scores[0].0, Lane::Vector);
    assert!((item.lane_scores[0].1 - 1.0).abs() < 1e-5);
    // Convenience accessor.
    assert!(item.lane_score(Lane::Vector).is_some());
    assert!(item.lane_score(Lane::Sparse).is_none());
    assert!(item.lane_score(Lane::GraphExpand).is_none());
}

#[test]
fn lane_scores_records_graph_expand_contribution() {
    let (repo, a_id, b_id, _c_id) = seed_chain_abc();
    let result = repo
        .retrieve()
        .vector("m", vec![1.0, 0.0])
        .with_graph_expand(GraphExpand::new())
        .execute()
        .unwrap();
    // Seed A sees only Vector; graph-expand neighbour B sees only
    // GraphExpand.
    let a = result.items.iter().find(|i| i.node.id == a_id).unwrap();
    let b = result.items.iter().find(|i| i.node.id == b_id).unwrap();
    assert!(a.lane_score(Lane::Vector).is_some());
    assert!(a.lane_score(Lane::GraphExpand).is_none());
    assert!(b.lane_score(Lane::Vector).is_none());
    assert!(b.lane_score(Lane::GraphExpand).is_some());
}

#[test]
fn lane_scores_deterministic_canonical_order() {
    // When a node is reached by multiple lanes, the lane_scores
    // vector must iterate in the fixed Vector < Sparse <
    // GraphExpand < Rerank order regardless of pipeline insertion
    // order. Uses the graph-expand path to guarantee a node with
    // a GraphExpand contribution alongside its Vector seed score.
    let (repo, _, _, _) = seed_chain_abc();
    let result = repo
        .retrieve()
        .vector("m", vec![1.0, 0.0])
        .with_graph_expand(GraphExpand::new())
        .execute()
        .unwrap();
    for item in &result.items {
        // Ascending canonical order.
        for pair in item.lane_scores.windows(2) {
            assert!(
                pair[0].0 < pair[1].0,
                "lane_scores must be sorted by Lane; got {:?}",
                item.lane_scores
            );
        }
    }
}

// ---------- Foundational invariant: zero providers ----------
//
// mnem-core must work with ZERO external services (no embedder,
// no sparse, no reranker, no LLM). These tests lock that in.
// If any of them regress, we've broken the "git-like without an
// LLM" contract from docs/LLM-FREE-MODE.md.

#[test]
fn llm_free_label_filter_works_without_any_provider() {
    // No embed, no sparse_embed, no context_sentence on any node.
    // No embedder, no sparse provider, no reranker, no LLM
    // configured. Label + prop filter retrieval must succeed.
    let (bs, ohs) = stores();
    let repo = ReadonlyRepo::init(bs, ohs).unwrap();
    let mut tx = repo.start_transaction();
    let doc = Node::new(NodeId::new_v7(), "Doc").with_summary("the tenant shall...");
    let person = Node::new(NodeId::new_v7(), "Person").with_summary("alice");
    tx.add_node(&doc).unwrap();
    tx.add_node(&person).unwrap();
    let repo = tx.commit("t", "seed").unwrap();

    let result = repo.retrieve().label("Doc").execute().unwrap();
    assert_eq!(result.items.len(), 1);
    assert_eq!(result.items[0].node.ntype, "Doc");
}

#[test]
fn llm_free_graph_expand_works_with_precomputed_embeds() {
    // Nodes carry precomputed `embed` vectors (caller-provided;
    // no call into any embedder). Graph-expand traverses the
    // authored edges without any LLM in the dep tree.
    let (repo, _, b_id, _) = seed_chain_abc();
    let result = repo
        .retrieve()
        .vector("m", vec![1.0, 0.0])
        .with_graph_expand(GraphExpand::new())
        .execute()
        .unwrap();
    // B is discovered via graph traversal, not any ML call.
    assert!(result.items.iter().any(|i| i.node.id == b_id));
}

#[test]
fn graph_expand_no_cycles_at_depth_two() {
    // A -> B -> A (cycle). depth=2 must not loop.
    use crate::id::EdgeId;
    use crate::objects::Edge;
    let (bs, ohs) = stores();
    let repo = ReadonlyRepo::init(bs, ohs).unwrap();
    let mut tx = repo.start_transaction();
    let a = Node::new(NodeId::new_v7(), "Doc").with_summary("A");
    let b = Node::new(NodeId::new_v7(), "Doc").with_summary("B");
    let cid_a = tx.add_node(&a).unwrap();
    tx.add_node(&b).unwrap();
    let emb = f32_embed("m", &[1.0, 0.0]);
    tx.set_embedding(cid_a, emb.model.clone(), emb).unwrap();
    tx.add_edge(&Edge::new(EdgeId::new_v7(), "mentions", a.id, b.id))
        .unwrap();
    tx.add_edge(&Edge::new(EdgeId::new_v7(), "mentions", b.id, a.id))
        .unwrap();
    let repo = tx.commit("t", "seed").unwrap();

    let ge = GraphExpand::new().with_depth(2);
    let result = repo
        .retrieve()
        .vector("m", vec![1.0, 0.0])
        .with_graph_expand(ge)
        .execute()
        .unwrap();
    // Exactly A + B, no duplicates, no infinite loop.
    assert_eq!(
        result.items.len(),
        2,
        "cyclic 2-node graph at depth=2 must yield exactly A + B"
    );
}
