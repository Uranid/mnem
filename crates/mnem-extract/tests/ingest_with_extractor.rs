//! End-to-end integration test: parse + chunk a small markdown doc
//! with `mnem-ingest`, then run `KeyBertExtractor` over each chunk and
//! mine co-occurrence relations on the aggregate entity set.
//!
//! This test proves the full E3 pipeline composes without touching
//! the transaction layer - an ingest driver that wires chunks to the
//! graph writer can slot the same two calls in place of its current
//! `RuleExtractor`.

use mnem_embed_providers::{Embedder, MockEmbedder};
use mnem_extract::{Entity, ExtractionSource, Extractor, KeyBertExtractor, mine_relations};
use mnem_ingest::{ChunkerKind, chunk, md::parse_markdown};

const DIM: u32 = 64;
const MODEL: &str = "mock:e3-integration";

#[test]
fn keybert_runs_over_ingest_chunks_and_emits_entities_plus_edges() {
    let embedder = MockEmbedder::new(MODEL, DIM);

    let doc = "\
# Research Log

Alice and Bob met at Acme Corp yesterday. The meeting \
covered the quarterly roadmap and a new research project \
on graph retrieval augmented generation.

## Action Items

Alice will draft the project spec. Bob will share the \
roadmap slides with the Acme leadership team.
";
    let sections = parse_markdown(doc).expect("markdown parses");
    let chunks = chunk(&sections, &ChunkerKind::Paragraph);
    assert!(!chunks.is_empty(), "expected at least one chunk");

    let extractor = KeyBertExtractor::new(&embedder).with_top_k(6);

    // Per-chunk entity lists: this is what a pipeline would attach to
    // each Chunk Node's props (or to a dedicated `entities` Node field
    // once the ingest surface grows one).
    let mut per_chunk_entities: Vec<Vec<Entity>> = Vec::with_capacity(chunks.len());
    for c in &chunks {
        let chunk_embed = embedder.embed(&c.text).expect("chunk embed ok");
        let ents = extractor.extract_entities(&c.text, &chunk_embed);
        per_chunk_entities.push(ents);
    }

    // Assertion 1: Node.entities populated - at least one chunk yielded
    // a non-empty entity list.
    let total_entities: usize = per_chunk_entities.iter().map(Vec::len).sum();
    assert!(
        total_entities > 0,
        "expected at least one entity across chunks, got {per_chunk_entities:?}",
    );

    // Assertion 2: Edge count > 0 - build the cross-chunk entity set
    // keyed by global byte offset so the co-occurrence miner sees
    // non-overlapping spans per mention. We concatenate chunk texts
    // with sentence terminators so sentence segmentation stays sane.
    let mut joined = String::new();
    let mut global_entities: Vec<Entity> = Vec::new();
    for (c, ents) in chunks.iter().zip(per_chunk_entities.iter()) {
        let offset = joined.len();
        joined.push_str(&c.text);
        joined.push_str("\n\n");
        for e in ents {
            global_entities.push(Entity {
                mention: e.mention.clone(),
                score: e.score,
                span: (e.span.0 + offset, e.span.1 + offset),
            });
        }
    }
    let relations = mine_relations(
        &joined,
        &global_entities,
        0.0, // permissive threshold - the assertion here is existence, not magnitude
        ExtractionSource::Statistical,
    );
    assert!(
        !relations.is_empty(),
        "expected at least one co-occurrence edge, got {relations:?} \
         over {} entities in {} chunks",
        global_entities.len(),
        chunks.len(),
    );

    // Assertion 3: all edges have source = Statistical.
    for r in &relations {
        assert_eq!(
            r.source,
            ExtractionSource::Statistical,
            "non-statistical relation leaked through: {r:?}",
        );
    }
}
