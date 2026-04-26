//! Property test: the Centroid + MMR summarizer is order-insensitive.
//!
//! Summarize a shuffled permutation of the same input; the resulting
//! [`Summary`] must be byte-for-byte identical (sentences AND scores).
//! This is the replay-stability guarantee the spec calls out.

use mnem_embed_providers::MockEmbedder;
use mnem_graphrag::summarize_community;
use proptest::prelude::*;

fn arb_short_sentence() -> impl Strategy<Value = String> {
    // Constrain to printable ASCII so shrinkage produces readable
    // counterexamples on failure.
    proptest::collection::vec(
        prop_oneof![
            Just('a'),
            Just('b'),
            Just('c'),
            Just('d'),
            Just('e'),
            Just(' '),
            Just('.'),
        ],
        3..=40,
    )
    .prop_map(|cs| cs.into_iter().collect::<String>())
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 32, .. ProptestConfig::default() })]

    #[test]
    fn determinism_under_input_permutation(
        xs in proptest::collection::vec(arb_short_sentence(), 2..=20),
        k in 1usize..=8,
        perm_seed in any::<u64>(),
    ) {
        // Skip cases with duplicate sentences: the content-hash
        // canonicalization already handles duplicates, but this
        // keeps the expected-vs-shuffled comparison noise-free.
        let mut unique: Vec<String> = xs.clone();
        unique.sort();
        unique.dedup();
        prop_assume!(unique.len() == xs.len());

        let e = MockEmbedder::new("test:mock", 16);

        let a = summarize_community(&xs, &e, None, &|_| 1.0, k, 0.5)
            .expect("summarize(original) must succeed");

        // Deterministic shuffle from seed (no rng crate dep needed).
        let mut shuffled = xs.clone();
        fisher_yates(&mut shuffled, perm_seed);
        let b = summarize_community(&shuffled, &e, None, &|_| 1.0, k, 0.5)
            .expect("summarize(shuffled) must succeed");

        prop_assert_eq!(a.sentences.clone(), b.sentences.clone());
        // Scores must match to the f32 bit pattern (byte-identical).
        prop_assert_eq!(a.scores.len(), b.scores.len());
        for (x, y) in a.scores.iter().zip(b.scores.iter()) {
            prop_assert_eq!(x.to_bits(), y.to_bits());
        }
    }
}

/// In-place Fisher-Yates using a splitmix64 PRNG seeded by `seed`.
/// No external dep; deterministic by construction.
fn fisher_yates<T>(xs: &mut [T], seed: u64) {
    let mut state = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let n = xs.len();
    for i in (1..n).rev() {
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        let j = (z as usize) % (i + 1);
        xs.swap(i, j);
    }
}
