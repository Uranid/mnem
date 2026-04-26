//! Embedder used by [`crate::adapters::MnemAdapter`].
//!
//! Two flavours, selected via [`BenchEmbedder`]:
//!
//! 1. **`BagOfTokens`** - the original 0.1.0 hashed-bag-of-tokens
//!    embedder. Network-free, deterministic, ~0.2 recall@5 on
//!    LongMemEval (toy ceiling). Always compiled.
//! 2. **`OnnxMiniLm`** - real `sentence-transformers/all-MiniLM-L6-v2`
//!    via `mnem-embed-providers` with the `onnx-bundled` feature.
//!    384-dim, byte-for-byte parity with ChromaDB's
//!    `DefaultEmbeddingFunction`. Default for the smoke gate; this
//!    is the embedder the headline numbers are reported against.
//!    Compiled in when `mnem-bench` is built with the (default-on)
//!    `onnx-minilm` feature.
//!
//! The two flavours share one method surface (`model() / dim() /
//! embed_text()`) so the adapter and the scorers stay flavour-blind.
//!
//! # Toy embedder rationale (kept)
//!
//! The bag-of-tokens variant stays compiled in for `--no-default-
//! features` builds, embedded targets, and any environment where
//! `ort/download-binaries` is undesirable. It uses double-hashed
//! token buckets (Weinberger et al. 2009) and L2-normalises so
//! cosine similarity collapses to dot product.

/// Default embedding dimension. 384 matches MiniLM-L6-v2 so the
/// toy embedder ships byte-compatible vector lengths with the real
/// ONNX one - swapping flavours never invalidates a vector index.
pub const DEFAULT_DIM: u32 = 384;

/// Deterministic hashed bag-of-tokens embedder.
///
/// `embed(text)` lowercases, ASCII-tokenises on non-alphanumeric
/// boundaries, hashes each token to two bucket positions
/// (FNV-1a-style) and adds 1.0 to each. The output vector is L2-
/// normalised so dense cosine similarity ranks documents by
/// (count-weighted) shared-token overlap.
#[derive(Clone, Debug)]
pub struct ToyEmbedder {
    model: String,
    dim: u32,
}

impl ToyEmbedder {
    /// Construct a new embedder with the given dimension. `dim`
    /// must be > 0; values < 32 lead to heavy hash collisions on
    /// natural text.
    #[must_use]
    pub fn new(dim: u32) -> Self {
        let d = dim.max(8);
        Self {
            model: format!("mnem-bench:bag-of-tokens-{d}"),
            dim: d,
        }
    }

    /// Model identifier (passed to mnem's vector lane so embeddings
    /// match query vectors at retrieve time).
    #[must_use]
    pub fn model(&self) -> &str {
        &self.model
    }

    /// Embedding dimension.
    #[must_use]
    pub const fn dim(&self) -> u32 {
        self.dim
    }

    /// Embed a string into a unit-norm vector.
    #[must_use]
    pub fn embed_text(&self, text: &str) -> Vec<f32> {
        let dim = self.dim as usize;
        let mut v = vec![0f32; dim];

        for tok in tokenise(text) {
            // Two buckets per token. Mixing two independent hashes
            // dampens the worst-case collision distortion of the
            // hashing trick (Weinberger et al. 2009).
            let h1 = fnv1a(tok.as_bytes()) as usize;
            let h2 = fnv1a_seeded(tok.as_bytes(), 0x9E37_79B9_7F4A_7C15) as usize;
            v[h1 % dim] += 1.0;
            v[h2 % dim] += 1.0;
        }

        // L2 normalise so cosine == dot.
        let mut s = 0f64;
        for x in &v {
            s += f64::from(*x) * f64::from(*x);
        }
        let norm = s.sqrt() as f32;
        if norm > 0.0 {
            for x in &mut v {
                *x /= norm;
            }
        }
        v
    }
}

/// Lower-case, ASCII-tokenise on `is_alphanumeric` boundaries.
/// Drops 1-character tokens (mostly punctuation noise) and trims
/// to <=64 characters per token to bound worst-case hashing.
fn tokenise(text: &str) -> impl Iterator<Item = String> + '_ {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|t| t.len() >= 2)
        .map(|t| {
            let lower = t.to_lowercase();
            if lower.len() > 64 {
                let mut end = 64;
                while end > 0 && !lower.is_char_boundary(end) {
                    end -= 1;
                }
                lower[..end].to_string()
            } else {
                lower
            }
        })
}

/// FNV-1a 64-bit hash over a byte slice.
fn fnv1a(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in bytes {
        h ^= u64::from(*b);
        h = h.wrapping_mul(0x100_0000_01b3);
    }
    h
}

/// FNV-1a with a custom 64-bit seed mixed into the offset basis.
/// Used to derive a second hash for double-hashing.
fn fnv1a_seeded(bytes: &[u8], seed: u64) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325 ^ seed;
    for b in bytes {
        h ^= u64::from(*b);
        h = h.wrapping_mul(0x100_0000_01b3);
    }
    h
}

// ============================================================
// Unified embedder used by `MnemAdapter`
// ============================================================

/// Unified embedder used by [`crate::adapters::MnemAdapter`].
///
/// The two flavours share a method surface (`model()`, `dim()`,
/// `embed_text()`) so the adapter does not branch on the variant on
/// every call. Construction is the only code path that picks one.
///
/// # Variants
///
/// - [`BenchEmbedder::BagOfTokens`] - the always-compiled toy
///   embedder. Selected by [`crate::EmbedderChoice::BagOfTokens`].
/// - [`BenchEmbedder::OnnxMiniLm`] - real MiniLM-L6-v2 via
///   `mnem-embed-providers` (gated on the `onnx-minilm` feature).
///   Selected by [`crate::EmbedderChoice::OnnxMiniLm`].
pub enum BenchEmbedder {
    /// Toy hashed bag-of-tokens. Network-free, ~0.2 recall@5 on
    /// LongMemEval; ships as the offline / WASM-clean fallback.
    BagOfTokens(ToyEmbedder),
    /// Real `all-MiniLM-L6-v2` via `mnem-embed-providers`. The
    /// concrete type is hidden behind a `Box<dyn Embedder>` so this
    /// crate stays agnostic to the underlying ORT session lifetime.
    /// `model_id` and `dim` are cached so the hot path (per-doc
    /// ingest, per-query retrieve) avoids vtable round-trips.
    #[cfg(feature = "onnx-minilm")]
    OnnxMiniLm {
        /// Boxed provider implementing `mnem_embed_providers::Embedder`.
        inner: Box<dyn mnem_embed_providers::Embedder>,
        /// Cached fully-qualified model id (e.g.
        /// `"onnx:all-MiniLM-L6-v2"`). Identifies the vector lane
        /// keyed on the mnem retriever.
        model_id: String,
        /// Cached output dimension (384 for MiniLM-L6-v2).
        dim: u32,
    },
}

impl std::fmt::Debug for BenchEmbedder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::BagOfTokens(e) => f.debug_tuple("BagOfTokens").field(e).finish(),
            #[cfg(feature = "onnx-minilm")]
            Self::OnnxMiniLm { model_id, dim, .. } => f
                .debug_struct("OnnxMiniLm")
                .field("model_id", model_id)
                .field("dim", dim)
                .finish(),
        }
    }
}

impl BenchEmbedder {
    /// Construct the toy hashed bag-of-tokens embedder of dimension
    /// `dim`. Matches 0.1.0 behaviour.
    #[must_use]
    pub fn bag_of_tokens(dim: u32) -> Self {
        Self::BagOfTokens(ToyEmbedder::new(dim))
    }

    /// Construct the real ONNX MiniLM-L6-v2 embedder via
    /// `mnem-embed-providers` (`onnx-bundled` flavour). Lazy-
    /// downloads the model on first call (ORT + tokenizer + weights
    /// fetched into the HuggingFace cache; ~90MB).
    ///
    /// # Errors
    ///
    /// Surfaces tokenizer / model-load / ORT-session failures from
    /// `mnem-embed-providers` verbatim as a `Box<dyn Error>`.
    #[cfg(feature = "onnx-minilm")]
    pub fn onnx_minilm() -> Result<Self, Box<dyn std::error::Error>> {
        use mnem_embed_providers::{OnnxConfig, ProviderConfig, open};
        let cfg = ProviderConfig::Onnx(OnnxConfig {
            // Matches the bench-Python adapter (LongMemEval session)
            // and the `mnem-cli --features bundled-embedder` default.
            model: "all-MiniLM-L6-v2".to_string(),
            // None defers to the model's `default_max_length` (256
            // for MiniLM-L6). LongMemEval sessions are typically
            // <512 tokens, so the default is fine.
            max_length: None,
        });
        let inner = open(&cfg).map_err(|e| Box::new(e) as Box<dyn std::error::Error>)?;
        let model_id = inner.model().to_string();
        let dim = inner.dim();
        Ok(Self::OnnxMiniLm { inner, model_id, dim })
    }

    /// Fully-qualified model identifier. Stamped on every
    /// `Embedding` and used as the key the retriever's vector lane
    /// resolves on, so two embedders with the same `model()` MUST
    /// produce vectors in the same semantic space.
    #[must_use]
    pub fn model(&self) -> &str {
        match self {
            Self::BagOfTokens(e) => e.model(),
            #[cfg(feature = "onnx-minilm")]
            Self::OnnxMiniLm { model_id, .. } => model_id.as_str(),
        }
    }

    /// Output vector dimension.
    #[must_use]
    pub fn dim(&self) -> u32 {
        match self {
            Self::BagOfTokens(e) => e.dim(),
            #[cfg(feature = "onnx-minilm")]
            Self::OnnxMiniLm { dim, .. } => *dim,
        }
    }

    /// Embed a single string. Errors from the ONNX path (tokenizer,
    /// session.run) are surfaced as `Box<dyn Error>`. The toy path
    /// is infallible, so `Result` here is a small ergonomic tax we
    /// pay so the call site stays variant-blind.
    ///
    /// # Errors
    ///
    /// Returns the underlying provider error verbatim for the ONNX
    /// flavour. The bag-of-tokens flavour cannot fail.
    pub fn embed_text(&self, text: &str) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        match self {
            Self::BagOfTokens(e) => Ok(e.embed_text(text)),
            #[cfg(feature = "onnx-minilm")]
            Self::OnnxMiniLm { inner, .. } => inner
                .embed(text)
                .map_err(|e| Box::new(e) as Box<dyn std::error::Error>),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embed_is_deterministic() {
        let e = ToyEmbedder::new(64);
        assert_eq!(e.embed_text("hello world"), e.embed_text("hello world"));
    }

    #[test]
    fn empty_yields_zero_vector() {
        let e = ToyEmbedder::new(32);
        let v = e.embed_text("");
        assert_eq!(v.len(), 32);
        assert!(v.iter().all(|x| *x == 0.0));
    }

    #[test]
    fn related_text_similarity_is_high() {
        let e = ToyEmbedder::new(384);
        let a = e.embed_text("alice climbs in berlin");
        let b = e.embed_text("alice goes climbing in berlin every weekend");
        let c = e.embed_text("the eiffel tower is in paris");
        let dot_ab: f32 = a.iter().zip(&b).map(|(x, y)| x * y).sum();
        let dot_ac: f32 = a.iter().zip(&c).map(|(x, y)| x * y).sum();
        // Shared-token overlap should beat the unrelated pair.
        assert!(dot_ab > dot_ac, "ab={dot_ab} should beat ac={dot_ac}");
    }
}
