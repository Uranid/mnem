//! Property-based determinism test for [`mnem_extract::KeyBertExtractor`].
//!
//! Invariant: extracting entities + relations twice over the same text
//! with the same embedder must produce byte-identical serde-CBOR
//! payloads. This protects us from BTreeMap → HashMap regressions,
//! non-stable MMR tiebreaks, or f32 accumulation drift that creeps in
//! via refactors.

use mnem_embed_providers::{Embedder, MockEmbedder};
use mnem_extract::{Extractor, KeyBertExtractor};
use proptest::prelude::*;

const DIM: u32 = 32;
const MODEL: &str = "mock:e3-proptest";

fn to_blob<T: serde::Serialize>(v: &T) -> Vec<u8> {
    // JSON is enough: if two Vec<Entity> serialise to the same JSON
    // bytes, every field (including f32 bits) matched.
    serde_json::to_vec(v).expect("serde ok")
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 48, .. ProptestConfig::default() })]

    #[test]
    fn extract_is_deterministic(text in "\\PC{0,500}") {
        let embedder = MockEmbedder::new(MODEL, DIM);
        let chunk_embed = embedder.embed(&text).unwrap();
        let extractor = KeyBertExtractor::new(&embedder).with_top_k(8);

        let ents_a = extractor.extract_entities(&text, &chunk_embed);
        let ents_b = extractor.extract_entities(&text, &chunk_embed);
        prop_assert_eq!(to_blob(&ents_a), to_blob(&ents_b));

        let rels_a = extractor.extract_relations(&text, &ents_a);
        let rels_b = extractor.extract_relations(&text, &ents_b);
        prop_assert_eq!(to_blob(&rels_a), to_blob(&rels_b));
    }
}
