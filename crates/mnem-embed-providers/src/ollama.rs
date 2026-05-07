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
//! via `OnceLock`; every subsequent call validates against the frozen
//! dim to catch provider misconfiguration or silent model swaps.

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
    /// Filled on the first successful `embed`; read-only thereafter.
    dim: OnceLock<u32>,
    endpoint: String,
    agent: ureq::Agent,
}

impl OllamaEmbedder {
    /// Construct from a validated [`OllamaConfig`]. Does NOT contact
    /// the server; the first `embed` call is what learns the dim.
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
}

impl Embedder for OllamaEmbedder {
    fn model(&self) -> &str {
        &self.model_fq
    }

    fn dim(&self) -> u32 {
        // Returns 0 before the first successful call. Callers that need
        // a concrete dim before making a call can issue `embed("")` or
        // similar; `mnem embed` walks nodes and discovers dim naturally.
        self.dim.get().copied().unwrap_or(0)
    }

    fn embed(&self, text: &str) -> Result<Vec<f32>, EmbedError> {
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
        let embedding = parsed
            .embeddings
            .into_iter()
            .next()
            .ok_or_else(|| EmbedError::Decode("ollama /api/embed returned empty embeddings array".into()))?;

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
                // First successful call freezes the dim. `set` races
                // are fine under contention: whichever writer wins
                // pins the value, and any later mismatch triggers the
                // check above.
                let _ = self.dim.set(got_dim);
            }
        }
        Ok(embedding)
    }

    fn manifest(&self) -> EmbedderManifest {
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
        // Must not error at construction - network is deferred to embed().
        let e = OllamaEmbedder::from_config(&cfg).unwrap();
        assert_eq!(e.model(), "ollama:nomic-embed-text");
        assert_eq!(e.dim(), 0); // dim unset until first successful call
    }
}
