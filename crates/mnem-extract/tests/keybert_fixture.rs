//! Fixed-text fixture tests for [`mnem_extract::KeyBertExtractor`].
//!
//! These tests use [`mnem_embed_providers::MockEmbedder`] (blake3-
//! derived, deterministic) so the whole pipeline stays fully offline
//! and reproducible on every CI runner.

use mnem_embed_providers::{Embedder, MockEmbedder};
use mnem_extract::{ExtractionSource, Extractor, KeyBertExtractor, mine_relations};

const DIM: u32 = 64;
const MODEL: &str = "mock:e3-keybert";

fn setup() -> (MockEmbedder, Vec<f32>, &'static str) {
    let embedder = MockEmbedder::new(MODEL, DIM);
    let text = "The quick brown fox jumps over the lazy dog. The dog sleeps.";
    // Chunk embedding: embed the full chunk text via the same mock
    // provider - mirrors what mnem-ingest does in production.
    let chunk_embed = embedder.embed(text).expect("mock embed ok");
    (embedder, chunk_embed, text)
}

#[test]
fn top_k_contains_dog_and_fox() {
    let (embedder, chunk_embed, text) = setup();
    let extractor = KeyBertExtractor::new(&embedder).with_top_k(10);
    let entities = extractor.extract_entities(text, &chunk_embed);
    assert!(!entities.is_empty(), "expected at least one entity");
    let mentions: Vec<String> = entities.iter().map(|e| e.mention.to_lowercase()).collect();
    assert!(
        mentions.iter().any(|m| m.contains("dog")),
        "expected a `dog`-containing mention in top-k: {mentions:?}",
    );
    assert!(
        mentions.iter().any(|m| m.contains("fox")),
        "expected a `fox`-containing mention in top-k: {mentions:?}",
    );
}

#[test]
fn cooccurrence_emits_fox_dog_pair_with_positive_pmi() {
    // A third sentence that mentions neither `fox` nor `dog` gives us
    // a proper marginal distribution: P(fox) = 1/3, P(dog) = 2/3, and
    // P(fox, dog) = 1/3, so PMI = ln( (1/3) / ((1/3)*(2/3)) ) = ln(1.5)
    // > 0 and also above the default threshold of 1.0 we lower below.
    let embedder = MockEmbedder::new(MODEL, DIM);
    let text = "The quick brown fox jumps over the lazy dog. The dog sleeps. The sky is blue.";
    let chunk_embed = embedder.embed(text).unwrap();
    let extractor = KeyBertExtractor::new(&embedder).with_top_k(10);
    let entities = extractor.extract_entities(text, &chunk_embed);

    // Use a permissive PMI threshold so the fixture proves the miner
    // pipes entity spans → sentences → PMI correctly, independent of
    // any threshold-tuning debate.
    let relations = mine_relations(text, &entities, 0.0, ExtractionSource::Statistical);

    let pair = relations.iter().find(|r| {
        let s = r.src.to_lowercase();
        let d = r.dst.to_lowercase();
        (s.contains("fox") && d.contains("dog")) || (s.contains("dog") && d.contains("fox"))
    });
    assert!(
        pair.is_some(),
        "expected a (fox, dog) co-occurrence relation, got {relations:?}",
    );
    let pair = pair.unwrap();
    assert!(
        pair.weight > 0.0,
        "expected positive PMI, got {}",
        pair.weight,
    );
}

#[test]
fn entities_have_valid_spans() {
    let (embedder, chunk_embed, text) = setup();
    let extractor = KeyBertExtractor::new(&embedder).with_top_k(10);
    let entities = extractor.extract_entities(text, &chunk_embed);
    for e in &entities {
        let (s, end) = e.span;
        assert!(end <= text.len(), "span {:?} overruns text", e.span);
        assert!(s < end, "empty span on {:?}", e.mention);
        // The mention must be reachable at the recorded span.
        assert!(
            text.get(s..end).is_some(),
            "span {:?} not utf8 aligned",
            e.span
        );
    }
}

#[test]
fn empty_text_returns_empty() {
    let embedder = MockEmbedder::new(MODEL, DIM);
    let chunk_embed = embedder.embed("anything").unwrap();
    let extractor = KeyBertExtractor::new(&embedder);
    assert!(extractor.extract_entities("", &chunk_embed).is_empty());
}

#[test]
fn zero_length_chunk_embed_returns_empty() {
    let embedder = MockEmbedder::new(MODEL, DIM);
    let extractor = KeyBertExtractor::new(&embedder);
    assert!(extractor.extract_entities("Some text.", &[]).is_empty(),);
}
