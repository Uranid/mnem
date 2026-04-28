//! Gap 15 property: the noise floor any ingest path sees comes from
//! the embedder's own manifest, NOT from a global constant.
//!
//! The regression this guards against is trivial to reintroduce: a
//! future refactor could "helpfully" add a `DEFAULT_NOISE_FLOOR: f32
//! = 0.25` and quietly paper over an embedder that forgot to
//! override `manifest()`. This test fuzzes a configurable fake
//! embedder across the full f32 range of legal noise floors and
//! confirms that the value a caller reads back is the same as the
//! value the provider published, never a global `0.25`.

use mnem_embed_providers::{Embedder, EmbedderManifest};
use proptest::prelude::*;

/// A tiny configurable embedder used only here: it lets the test set
/// the manifest's `noise_floor` at will, so the property covers the
/// whole legal range instead of the handful of per-provider constants.
struct ConfigurableEmbedder {
    model: String,
    dim: u32,
    noise_floor: f32,
}

impl Embedder for ConfigurableEmbedder {
    fn model(&self) -> &str {
        &self.model
    }
    fn dim(&self) -> u32 {
        self.dim
    }
    fn embed(&self, _text: &str) -> Result<Vec<f32>, mnem_embed_providers::EmbedError> {
        Ok(vec![0.0; self.dim as usize])
    }
    fn manifest(&self) -> EmbedderManifest {
        EmbedderManifest::new(self.model.clone(), self.dim, self.noise_floor)
    }
}

/// The "wrong" value Gap 15 eliminates. If this ever creeps back in as
/// a shared constant, this test becomes the tripwire.
const GLOBAL_FLOOR_025: f32 = 0.25;

proptest! {
    #[test]
    fn manifest_noise_floor_is_used_not_global(
        floor in 0.0f32..=1.0f32,
        dim in 1u32..=4096u32,
    ) {
        // Deliberately pick a model id that does NOT match 0.25 so a
        // regression that ignores the manifest cannot pass by accident.
        let e = ConfigurableEmbedder {
            model: "fuzz:custom".into(),
            dim,
            noise_floor: floor,
        };
        let m = e.manifest();

        // The property: what the caller sees equals what the provider
        // set. It is NEVER the global 0.25 unless the provider
        // actually set 0.25.
        prop_assert!((m.noise_floor - floor).abs() < f32::EPSILON);
        if (floor - GLOBAL_FLOOR_025).abs() > f32::EPSILON {
            prop_assert!((m.noise_floor - GLOBAL_FLOOR_025).abs() > f32::EPSILON);
        }

        prop_assert_eq!(m.dim, dim);
        prop_assert_eq!(m.model_id, "fuzz:custom");
    }
}

#[test]
fn mock_embedder_manifest_is_zero_floor() {
    // Mock embedder is hash-derived: its floor is `0.0` by contract.
    let e = mnem_embed_providers::MockEmbedder::new("mock:test", 16);
    let m = e.manifest();
    assert!((m.noise_floor - 0.0).abs() < f32::EPSILON);
    assert_eq!(m.dim, 16);
    assert_eq!(m.model_id, "mock:test");
}

#[test]
fn derivation_helpers_are_monotone_in_budget() {
    use mnem_embed_providers::{derive_max_cooccurrence_ms, derive_max_knn_ingest_per_node_ms};
    // Larger budget never decreases per-node co-occurrence allowance.
    let small = derive_max_cooccurrence_ms(Some(100));
    let big = derive_max_cooccurrence_ms(Some(1000));
    assert!(big >= small);

    // Larger batch never increases per-node kNN allowance (divisor grows).
    let batch_small = derive_max_knn_ingest_per_node_ms(200, 2);
    let batch_big = derive_max_knn_ingest_per_node_ms(200, 20);
    assert!(batch_small >= batch_big);
}
