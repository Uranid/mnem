//! Zero-impact contract for the community-filter stage (experiment E1).
//!
//! Asserts that installing a `CommunityFilterCfg { enabled: false, .. }`
//! plus any `CommunityLookup` produces byte-identical retrieval
//! results vs. leaving the feature wholly unconfigured. This pins
//! the flag-off contract documented in `community_filter.rs`.

use std::sync::Arc;

use bytes::Bytes;
use ipld_core::ipld::Ipld;

use mnem_core::id::NodeId;
use mnem_core::objects::{Dtype, Embedding, Node};
use mnem_core::repo::ReadonlyRepo;
use mnem_core::retrieve::{CommunityFilterCfg, CommunityLookup};
use mnem_core::store::{Blockstore, MemoryBlockstore, MemoryOpHeadsStore, OpHeadsStore};

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
        dim: u32::try_from(v.len()).expect("test vec fits in u32"),
        vector: Bytes::from(bytes),
    }
}

#[test]
fn disabled_community_filter_is_byte_identical_to_no_config() {
    // Build a small fixture: 4 Doc nodes with distinct embeddings.
    let (bs, ohs) = stores();
    let repo = ReadonlyRepo::init(bs, ohs).unwrap();
    let a = Node::new(NodeId::new_v7(), "Doc").with_prop("name", Ipld::String("a".into()));
    let b = Node::new(NodeId::new_v7(), "Doc").with_prop("name", Ipld::String("b".into()));
    let c = Node::new(NodeId::new_v7(), "Doc").with_prop("name", Ipld::String("c".into()));
    let d = Node::new(NodeId::new_v7(), "Doc").with_prop("name", Ipld::String("d".into()));

    let mut tx = repo.start_transaction();
    let a_cid = tx.add_node(&a).unwrap();
    let a_emb = f32_embed("m", &[1.0, 0.0, 0.0]);
    tx.set_embedding(a_cid, a_emb.model.clone(), a_emb).unwrap();
    let b_cid = tx.add_node(&b).unwrap();
    let b_emb = f32_embed("m", &[0.9, 0.1, 0.0]);
    tx.set_embedding(b_cid, b_emb.model.clone(), b_emb).unwrap();
    let c_cid = tx.add_node(&c).unwrap();
    let c_emb = f32_embed("m", &[0.0, 1.0, 0.0]);
    tx.set_embedding(c_cid, c_emb.model.clone(), c_emb).unwrap();
    let d_cid = tx.add_node(&d).unwrap();
    let d_emb = f32_embed("m", &[0.0, 0.0, 1.0]);
    tx.set_embedding(d_cid, d_emb.model.clone(), d_emb).unwrap();
    let repo = tx.commit("t", "seed").unwrap();

    // Baseline: no community filter wired at all.
    let baseline = repo
        .retrieve()
        .vector("m", vec![1.0, 0.0, 0.0])
        .execute()
        .unwrap();

    // With `enabled: false` + a non-trivial lookup (the first two
    // nodes in community 0, others in community 1). The lookup is
    // intentionally real-looking so the test also proves a
    // flag-off run does not even traverse the lookup function.
    let ids_ab = [a.id, b.id];
    let lookup = Arc::new(CommunityLookup::new(move |n| {
        if ids_ab.contains(n) { Some(0) } else { Some(1) }
    }));
    let with_disabled_cfg = repo
        .retrieve()
        .vector("m", vec![1.0, 0.0, 0.0])
        .with_community_filter(
            CommunityFilterCfg {
                enabled: false,
                ..Default::default()
            },
            lookup,
        )
        .execute()
        .unwrap();

    // Byte-identical: same items in same order with same scores and
    // same rendered text.
    assert_eq!(
        baseline.items.len(),
        with_disabled_cfg.items.len(),
        "item count drift"
    );
    for (i, (lhs, rhs)) in baseline
        .items
        .iter()
        .zip(with_disabled_cfg.items.iter())
        .enumerate()
    {
        assert_eq!(lhs.node.id, rhs.node.id, "id drift at position {i}");
        assert!(
            (lhs.score - rhs.score).abs() < f32::EPSILON,
            "score drift at position {i}: {} vs {}",
            lhs.score,
            rhs.score
        );
        assert_eq!(lhs.rendered, rhs.rendered, "render drift at position {i}");
    }
    assert_eq!(baseline.tokens_used, with_disabled_cfg.tokens_used);
    assert_eq!(baseline.dropped, with_disabled_cfg.dropped);
}
