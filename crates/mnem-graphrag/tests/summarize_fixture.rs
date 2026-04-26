//! Fixture test for Centroid + MMR summarizer.
//!
//! Uses the deterministic [`MockEmbedder`] so the test is hermetic.
//! Asserts:
//!   1. The returned summary has exactly `k` sentences (when `n >= k`).
//!   2. Scores are sorted descending in selection order (MMR-greedy).
//!   3. Near-duplicates are suppressed: two near-duplicate sentences
//!      cannot both appear in the top-3 when `mmr_lambda` is high.

use mnem_embed_providers::{Embedder, MockEmbedder};
use mnem_graphrag::summarize_community;

fn fixture_sentences() -> Vec<String> {
    // 10 sentences total; the last two are near-duplicates of each
    // other by construction (identical prefix + tiny suffix).
    vec![
        "Alice lives in Berlin and climbs boulder routes on weekends.".to_string(),
        "Bob runs a coffee shop in Lisbon and makes pastries daily.".to_string(),
        "The Eiffel Tower was completed in 1889 as a world's fair centrepiece.".to_string(),
        "Photosynthesis converts sunlight into chemical energy in plants.".to_string(),
        "Neural networks learn representations through gradient descent.".to_string(),
        "The Pacific is the largest ocean, covering about a third of Earth.".to_string(),
        "Rust's ownership model eliminates data races at compile time.".to_string(),
        "Jazz improvisation draws on chord-scale relationships and motifs.".to_string(),
        // Near-dup pair:
        "Climbing in Berlin is great on weekends because the gyms stay open late.".to_string(),
        "Climbing in Berlin is great on weekends because the gyms stay open late!".to_string(),
    ]
}

#[test]
fn returns_exactly_k_when_n_gte_k() {
    let e = MockEmbedder::new("test:mock", 32);
    let xs = fixture_sentences();
    // Query-embedded case: use a query vector about "Berlin climbing".
    let q = e.embed("Berlin climbing weekend gyms").unwrap();
    let summary = summarize_community(&xs, &e, Some(&q), &|_| 1.0, 5, 0.5).unwrap();
    assert_eq!(summary.sentences.len(), 5);
    assert_eq!(summary.scores.len(), 5);
}

#[test]
fn scores_are_non_increasing_in_selection_order() {
    let e = MockEmbedder::new("test:mock", 32);
    let xs = fixture_sentences();
    let q = e.embed("Berlin climbing weekend gyms").unwrap();
    let summary = summarize_community(&xs, &e, Some(&q), &|_| 1.0, 5, 0.5).unwrap();
    // MMR greedy: selection-order scores must be non-increasing.
    for w in summary.scores.windows(2) {
        assert!(
            w[0] >= w[1] - f32::EPSILON,
            "scores not non-increasing: {:?}",
            summary.scores
        );
    }
}

#[test]
fn mmr_suppresses_near_duplicates_in_top_3() {
    let e = MockEmbedder::new("test:mock", 32);
    let xs = fixture_sentences();
    let q = e.embed("climbing in Berlin").unwrap();

    // High lambda -> strong diversity penalty -> near-dups should
    // not both appear in the top-3.
    let summary = summarize_community(&xs, &e, Some(&q), &|_| 1.0, 3, 0.9).unwrap();
    let nd_a = &xs[8];
    let nd_b = &xs[9];
    let picked_a = summary.sentences.iter().any(|s| s == nd_a);
    let picked_b = summary.sentences.iter().any(|s| s == nd_b);
    assert!(
        !(picked_a && picked_b),
        "MMR failed to suppress near-duplicate: both {nd_a:?} and {nd_b:?} were picked"
    );
}

#[test]
fn no_query_redistributes_to_alpha() {
    // Smoke: summary still non-empty when query is None, and
    // scores are finite.
    let e = MockEmbedder::new("test:mock", 32);
    let xs = fixture_sentences();
    let summary = summarize_community(&xs, &e, None, &|_| 1.0, 3, 0.5).unwrap();
    assert_eq!(summary.sentences.len(), 3);
    for s in &summary.scores {
        assert!(s.is_finite(), "score not finite: {s}");
    }
}

#[test]
fn centrality_fallback_influences_score() {
    let e = MockEmbedder::new("test:mock", 32);
    let xs = fixture_sentences();
    // Degree-centrality fallback: give index 0 a massively higher
    // degree than everyone else. It should appear in the top-k.
    let centrality = |i: usize| if i == 0 { 100.0 } else { 1.0 };
    let summary = summarize_community(&xs, &e, None, &centrality, 3, 0.5).unwrap();
    assert!(
        summary.sentences.iter().any(|s| s == &xs[0]),
        "high-centrality sentence not selected: {:?}",
        summary.sentences
    );
}
