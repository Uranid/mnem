//! Ollama Embeddings adapter.
//!
//! POST `{base_url}/api/embed` with body `{model, input}` (Ollama ≥ 0.1.26).
//! No auth header. Ollama's embed endpoint does not batch (one string per
//! call), so `embed_batch` falls back to the default per-text loop.
//! The old `/api/embeddings` endpoint with `prompt` was deprecated and
//! returns HTTP 500 for inputs that exceed the model's context length.
//!
//! Ollama does not advertise the vector dim before the first call, so
//! the adapter learns it lazily from the first response and freezes it
//! via `OnceLock`. BUG-32: `manifest()` previously returned `dim=0`
//! before any `embed` call was made, causing ANN index initialization to
//! see 0 and panic or behave incorrectly. The fix is to probe Ollama
//! eagerly inside `ensure_dim_initialized()`, which is called at the
//! start of both `manifest()` and `embed()`.

use std::sync::OnceLock;
use std::time::Duration;

use serde::{Deserialize, Serialize};

use crate::config::OllamaConfig;
use crate::embedder::Embedder;
use crate::error::EmbedError;
use crate::http::{classify_ureq_error, decode_json};
use crate::manifest::EmbedderManifest;

/// Live adapter over a local Ollama server.
pub struct OllamaEmbedder {
    model_bare: String,
    model_fq: String,
    /// Filled on the first successful `embed` (or the `manifest()` probe);
    /// read-only thereafter.
    dim: OnceLock<u32>,
    endpoint: String,
    agent: ureq::Agent,
}

impl OllamaEmbedder {
    /// Construct from a validated [`OllamaConfig`]. Does NOT contact
    /// the server; the first `embed` call (or `manifest()` call) is
    /// what learns the dim via [`Self::ensure_dim_initialized`].
    ///
    /// # Errors
    ///
    /// Currently infallible at construction (no env var, no preflight).
    /// Returns a `Result` so the signature matches `OpenAiEmbedder::from_config`
    /// and future hardening (e.g. a `/api/version` probe) can fail.
    pub fn from_config(config: &OllamaConfig) -> Result<Self, EmbedError> {
        let endpoint = format!("{}/api/embed", config.base_url.trim_end_matches('/'));
        let agent = ureq::AgentBuilder::new()
            .timeout(Duration::from_secs(config.timeout_secs))
            .build();
        Ok(Self {
            model_bare: config.model.clone(),
            model_fq: format!("ollama:{}", config.model),
            dim: OnceLock::new(),
            endpoint,
            agent,
        })
    }

    /// Ensure the embedding dimension is known, probing Ollama if necessary.
    ///
    /// If the dim is already set in the `OnceLock`, this is a cheap no-op.
    /// Otherwise it embeds an empty string to learn the dim from the response
    /// and freezes it. Subsequent calls are always the cheap path.
    ///
    /// # Errors
    ///
    /// Returns an [`EmbedError`] if Ollama is unreachable or the probe fails.
    /// Callers that cannot propagate an error (e.g. `manifest()`) should log
    /// the failure and fall back to `dim=0` with a warning.
    fn ensure_dim_initialized(&self) -> Result<(), EmbedError> {
        if self.dim.get().is_some() {
            return Ok(());
        }
        // Probe with an empty string - this is the minimal valid input and
        // always returns a valid embedding vector from Ollama. The result is
        // discarded; we only need the length to freeze the dim.
        #[derive(Serialize)]
        struct Req<'a> {
            model: &'a str,
            input: &'a str,
        }
        #[derive(Deserialize)]
        struct Resp {
            embeddings: Vec<Vec<f32>>,
        }

        let body = Req {
            model: &self.model_bare,
            input: "",
        };
        let resp = self
            .agent
            .post(&self.endpoint)
            .set("Content-Type", "application/json")
            .send_json(&body)
            .map_err(classify_ureq_error)?;
        let parsed: Resp = decode_json(resp)?;
        let embedding = parsed.embeddings.into_iter().next().ok_or_else(|| {
            EmbedError::Decode("ollama /api/embed probe returned empty embeddings array".into())
        })?;

        let got_dim = u32::try_from(embedding.len()).unwrap_or(u32::MAX);
        // `set` is a no-op if another thread raced us; that is fine - the
        // winning writer pins the value and later mismatch checks catch skew.
        let _ = self.dim.set(got_dim);
        Ok(())
    }
}

impl Embedder for OllamaEmbedder {
    fn model(&self) -> &str {
        &self.model_fq
    }

    fn dim(&self) -> u32 {
        // Returns 0 only if neither `manifest()` nor `embed()` has been called
        // successfully yet. Both call `ensure_dim_initialized()` eagerly, so
        // in normal operation this path returns the real dim.
        self.dim.get().copied().unwrap_or(0)
    }

    fn embed(&self, text: &str) -> Result<Vec<f32>, EmbedError> {
        // BUG-32: ensure dim is known before the actual embed so that
        // the mismatch check below always has an expected value to compare
        // against. If Ollama is unreachable, `ensure_dim_initialized` already
        // returns an error; we propagate it rather than silently returning 0.
        self.ensure_dim_initialized()?;

        // Use the /api/embed endpoint (Ollama ≥ 0.1.26) which accepts an
        // `input` array and auto-truncates at the model's context length.
        // The old /api/embeddings endpoint with `prompt` returns HTTP 500
        // for inputs that exceed the context window (e.g. long financial
        // tables with bge-large's 512-token limit).
        #[derive(Serialize)]
        struct Req<'a> {
            model: &'a str,
            input: &'a str,
        }
        #[derive(Deserialize)]
        struct Resp {
            embeddings: Vec<Vec<f32>>,
        }

        let body = Req {
            model: &self.model_bare,
            input: text,
        };
        let resp = self
            .agent
            .post(&self.endpoint)
            .set("Content-Type", "application/json")
            .send_json(&body)
            .map_err(classify_ureq_error)?;
        let parsed: Resp = decode_json(resp)?;
        let embedding = parsed.embeddings.into_iter().next().ok_or_else(|| {
            EmbedError::Decode("ollama /api/embed returned empty embeddings array".into())
        })?;

        let got_dim = u32::try_from(embedding.len()).unwrap_or(u32::MAX);
        match self.dim.get() {
            Some(&expected) => {
                if got_dim != expected {
                    return Err(EmbedError::DimMismatch {
                        expected,
                        got: got_dim,
                    });
                }
            }
            None => {
                // `ensure_dim_initialized` already ran above; this branch
                // can only be hit under a very tight race where another
                // thread cleared the OnceLock between our probe and now,
                // which OnceLock makes impossible. Kept as a belt-and-
                // suspenders freeze.
                let _ = self.dim.set(got_dim);
            }
        }
        Ok(embedding)
    }

    fn manifest(&self) -> EmbedderManifest {
        // BUG-32 fix: probe Ollama eagerly so `dim` is real rather than 0.
        // If the probe fails (Ollama not reachable), we fall back to dim=0
        // and emit a warning to stderr; this is the best we can do given
        // that the trait signature is infallible.
        if let Err(e) = self.ensure_dim_initialized() {
            eprintln!(
                "mnem-embed-providers [ollama]: manifest() could not probe dim for \
                 model {:?} — Ollama unreachable? Returning dim=0. Error: {e}",
                self.model_fq
            );
        }
        // Measured on BGE-M3 served via Ollama (Gap 15 solution.md).
        // Ollama's default dense model family sits around 0.31 for
        // unrelated-text cosine similarity. The value is conservative:
        // it is the 95th-percentile of noise rather than the mean, so
        // ingest does not accidentally treat near-noise as signal.
        EmbedderManifest::new(
            self.model_fq.clone(),
            self.dim.get().copied().unwrap_or(0),
            0.31,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_config_does_not_contact_network() {
        let cfg = OllamaConfig {
            model: "nomic-embed-text".into(),
            base_url: "http://definitely-not-reachable.example.invalid:11434".into(),
            ..Default::default()
        };
        // Must not error at construction - network is deferred to the first
        // `embed()` or `manifest()` call via `ensure_dim_initialized()`.
        let e = OllamaEmbedder::from_config(&cfg).unwrap();
        assert_eq!(e.model(), "ollama:nomic-embed-text");
        // dim() still returns 0 immediately after construction because
        // `ensure_dim_initialized` has not been called yet.
        assert_eq!(e.dim(), 0);
    }

    #[test]
    fn ensure_dim_initialized_returns_error_when_unreachable() {
        let cfg = OllamaConfig {
            model: "nomic-embed-text".into(),
            base_url: "http://definitely-not-reachable.example.invalid:11434".into(),
            ..Default::default()
        };
        let e = OllamaEmbedder::from_config(&cfg).unwrap();
        // Probing an unreachable server must return a Network error, not panic.
        let result = e.ensure_dim_initialized();
        assert!(
            result.is_err(),
            "ensure_dim_initialized should fail when Ollama is unreachable"
        );
        match result.unwrap_err() {
            EmbedError::Network(_) => {}
            other => panic!("expected EmbedError::Network, got: {other:?}"),
        }
    }
}
