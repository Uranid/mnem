//! The [`Embedder`] trait and the [`to_embedding`] conversion helper.

use bytes::Bytes;
use mnem_core::objects::{Dtype, Embedding};

use crate::error::EmbedError;
use crate::manifest::EmbedderManifest;

/// Producer of vector embeddings for UTF-8 text. Sync, `Send + Sync`,
/// no async, no runtime binding.
///
/// # Determinism
///
/// For a fixed `(provider, model)` pair, `embed(text)` MUST return the
/// same vector on every call with the same `text`. A provider that uses
/// randomised projections or whose output drifts between API versions
/// violates this contract; mnem treats such providers as unusable
/// because retrieval-replay relies on `Embedding` bytes being stable.
///
/// # Example
///
/// ```no_run
/// # use mnem_embed_providers::{Embedder, ProviderConfig, OpenAiConfig, open};
/// # fn demo() -> Result<(), Box<dyn std::error::Error>> {
/// let cfg = ProviderConfig::Openai(OpenAiConfig {
///     model: "text-embedding-3-small".into(),
///     ..Default::default()
/// });
/// let embedder = open(&cfg)?;
/// let v = embedder.embed("Alice lives in Berlin")?;
/// assert_eq!(v.len(), embedder.dim() as usize);
/// # Ok(()) }
/// ```
pub trait Embedder: Send + Sync {
    /// Fully-qualified model identifier, namespaced by provider:
    /// `"openai:text-embedding-3-small"`,
    /// `"ollama:nomic-embed-text"`.
    ///
    /// This string is the keying dimension of
    /// [`mnem_core::index::BruteForceVectorIndex`]; two embedders with
    /// the same `model()` output MUST produce vectors in the same
    /// semantic space.
    fn model(&self) -> &str;

    /// Vector dimension this embedder produces. Every returned
    /// `Vec<f32>` has exactly this length; a provider response of any
    /// other length is mapped to [`EmbedError::DimMismatch`].
    fn dim(&self) -> u32;

    /// Embed a single text into a vector of length `self.dim()`.
    ///
    /// # Errors
    ///
    /// See [`EmbedError`] for the taxonomy.
    fn embed(&self, text: &str) -> Result<Vec<f32>, EmbedError>;

    /// Embed a batch. The default implementation fans out to `embed`;
    /// adapters that support true batch endpoints (e.g. `OpenAI`) should
    /// override this for latency and cost.
    ///
    /// Output length equals input length; order is preserved.
    ///
    /// # Errors
    ///
    /// Same as [`Self::embed`]. A batch fails atomically: if any item
    /// errors, no partial results are returned.
    fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, EmbedError> {
        texts.iter().map(|t| self.embed(t)).collect()
    }

    /// Self-describing metadata for this embedder.
    ///
    /// Every provider MUST override this so downstream ingest code
    /// (Gap 15) never has to guess a semantic-similarity floor from a
    /// single global constant. Concrete providers publish their
    /// empirically-measured `noise_floor` via the returned
    /// [`EmbedderManifest`].
    ///
    /// The default implementation panics: the trait is kept
    /// default-ful rather than strictly required only to avoid an
    /// awkward migration for downstream test doubles. Every shipped
    /// provider in this crate overrides it. The CLI audit subcommand
    /// (`mnem embedder audit`) exists to catch any provider that
    /// leaves the default in place.
    #[must_use]
    fn manifest(&self) -> EmbedderManifest {
        panic!(
            "Embedder::manifest() not implemented for model {:?}; \
             every provider must override manifest() to declare its \
             noise_floor (Gap 15)",
            self.model()
        );
    }
}

/// Convert an embedder's raw `Vec<f32>` output into the on-wire
/// [`Embedding`] shape that mnem-core writes into `Node.embed`.
///
/// Always produces an `f32` little-endian packed `Bytes` with
/// `dim = v.len()`. The `model` argument should be the fully-qualified
/// identifier returned by [`Embedder::model`].
#[must_use]
pub fn to_embedding(model: &str, v: &[f32]) -> Embedding {
    let mut buf = Vec::with_capacity(v.len() * 4);
    for x in v {
        buf.extend_from_slice(&x.to_le_bytes());
    }
    // Safe u32 cast: embedding dims are empirically <= 8192; u32::MAX is
    // astronomically larger.
    let dim = u32::try_from(v.len()).unwrap_or(u32::MAX);
    Embedding {
        model: model.to_string(),
        dtype: Dtype::F32,
        dim,
        vector: Bytes::from(buf),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn to_embedding_packs_f32_little_endian() {
        let v = vec![1.0f32, -2.5, 3.25];
        let e = to_embedding("test:model", &v);
        assert_eq!(e.model, "test:model");
        assert_eq!(e.dim, 3);
        assert_eq!(e.dtype, Dtype::F32);
        assert_eq!(e.vector.len(), 12);
        // Round-trip via validate.
        e.validate().unwrap();
    }
}
