//! A deterministic [`Embedder`] for tests. Available iff the `mock`
//! feature is enabled OR the crate is being compiled as a test target.
//!
//! The vector is derived from `blake3(text)` so two processes see the
//! same output for the same input without touching the network.

use crate::embedder::Embedder;
use crate::error::EmbedError;
use crate::manifest::EmbedderManifest;

/// Deterministic, network-free [`Embedder`] suitable for unit tests.
///
/// Every call to `embed(text)` returns the same `dim`-length `Vec<f32>`,
/// derived from `blake3(text)`. Different inputs produce different
/// vectors with overwhelming probability; identical inputs are
/// guaranteed to produce identical vectors.
#[derive(Debug, Clone)]
pub struct MockEmbedder {
    model: String,
    dim: u32,
}

impl MockEmbedder {
    /// Construct a mock embedder for the given `(model, dim)` pair.
    #[must_use]
    pub fn new(model: impl Into<String>, dim: u32) -> Self {
        Self {
            model: model.into(),
            dim,
        }
    }
}

impl Embedder for MockEmbedder {
    fn model(&self) -> &str {
        &self.model
    }

    fn dim(&self) -> u32 {
        self.dim
    }

    fn embed(&self, text: &str) -> Result<Vec<f32>, EmbedError> {
        // Hash the text, then reinterpret 4-byte windows of the
        // derived stream as f32 values in roughly [-1, 1] so cosine
        // similarity behaves sensibly for tests.
        let mut hasher = blake3::Hasher::new();
        hasher.update(text.as_bytes());
        let mut xof = hasher.finalize_xof();
        let dim = self.dim as usize;
        let mut out = Vec::with_capacity(dim);
        let mut buf = [0u8; 4];
        for _ in 0..dim {
            xof.fill(&mut buf);
            // Map u32 bits to [-1.0, 1.0] via `u32 -> i32 -> f32 / i32::MAX`.
            let bits = u32::from_le_bytes(buf) as i32;
            out.push((bits as f32) / (i32::MAX as f32));
        }
        Ok(out)
    }

    fn manifest(&self) -> EmbedderManifest {
        // Mock embedder is a hash; it has no meaningful noise floor.
        // `0.0` means "do not filter co-occurrences by similarity" which
        // is the only safe default for tests that want every candidate
        // edge to survive.
        EmbedderManifest::new(self.model.clone(), self.dim, 0.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mock_embedder_is_deterministic() {
        let e = MockEmbedder::new("test:mock", 16);
        let a = e.embed("hello").unwrap();
        let b = e.embed("hello").unwrap();
        assert_eq!(a, b);
        assert_eq!(a.len(), 16);
    }

    #[test]
    fn mock_embedder_distinguishes_inputs() {
        let e = MockEmbedder::new("test:mock", 16);
        let a = e.embed("hello").unwrap();
        let b = e.embed("world").unwrap();
        assert_ne!(a, b);
    }

    #[test]
    fn mock_embedder_dim_is_respected() {
        let e = MockEmbedder::new("test:mock", 128);
        assert_eq!(e.embed("x").unwrap().len(), 128);
        assert_eq!(e.dim(), 128);
    }
}
