// Provider adapter: OpenAI is a well-known proper-noun identifier;
// backticking every mention adds no information in rendered rustdoc.
#![allow(clippy::doc_markdown)]

//! OpenAI Embeddings API adapter.
//!
//! Ships behind the `openai` cargo feature (default-on). Calls
//! `POST {base_url}/v1/embeddings` with Bearer-token auth, a JSON body
//! `{model, input}`, and parses the `data[].embedding` fields back.

use std::time::Duration;

use serde::{Deserialize, Serialize};

use crate::config::OpenAiConfig;
use crate::embedder::Embedder;
use crate::error::EmbedError;
use crate::http::{classify_ureq_error, decode_json};
use crate::manifest::EmbedderManifest;

/// Maximum texts packed into a single `POST /v1/embeddings` call.
/// OpenAI allows up to 2048 inputs per call; we cap well below to keep
/// request bodies small and errors attributable.
const MAX_BATCH: usize = 96;

/// Known-to-mnem OpenAI embedding models and their dimensions. The
/// adapter refuses to construct against an unknown model string so a
/// typo (e.g. `text-embedding-3-smal`) fails at `from_config` time
/// instead of at first retrieval, and so dim-mismatch errors never
/// arise from wrong-dim assumption.
///
/// Add a row when OpenAI ships a new model and you've confirmed the dim.
const KNOWN_MODELS: &[(&str, u32)] = &[
    // OpenAI-native.
    ("text-embedding-3-small", 1536),
    ("text-embedding-3-large", 3072),
    ("text-embedding-ada-002", 1536),
    // BAAI / BGE family, typically served via an OpenAI-compatible
    // endpoint (vLLM, text-embeddings-inference, OpenRouter proxy).
    // Dim numbers match the released checkpoints on HuggingFace.
    ("BAAI/bge-large-en-v1.5", 1024),
    ("BAAI/bge-base-en-v1.5", 768),
    ("BAAI/bge-small-en-v1.5", 384),
    ("BAAI/bge-m3", 1024),
    // Alibaba Qwen3-Embedding family (MTEB SOTA as of mid-2025).
    // Served via OpenAI-compatible endpoints (vLLM, DashScope, etc.).
    ("Qwen/Qwen3-Embedding-0.6B", 1024),
    ("Qwen/Qwen3-Embedding-4B", 2560),
    ("Qwen/Qwen3-Embedding-8B", 4096),
    // mxbai - strong open-source option popular in LangChain /
    // LlamaIndex ecosystems. 1024-dim matches BGE-large so a
    // retriever swap between the two is dim-compatible.
    ("mixedbread-ai/mxbai-embed-large-v1", 1024),
];

fn known_dim(model: &str) -> Option<u32> {
    KNOWN_MODELS
        .iter()
        .find_map(|(m, d)| (*m == model).then_some(*d))
}

/// `OpenAiEmbedder` - live adapter over the OpenAI Embeddings REST API.
#[derive(Debug)]
pub struct OpenAiEmbedder {
    /// Bare model name as sent to the API, e.g. `"text-embedding-3-small"`.
    model_bare: String,
    /// Namespaced identifier returned from [`Embedder::model`], e.g.
    /// `"openai:text-embedding-3-small"`.
    model_fq: String,
    /// Vector dimension, looked up from [`KNOWN_MODELS`] at construction.
    dim: u32,
    /// API key read from the env var at construction time.
    api_key: String,
    /// Fully-qualified URL, e.g. `"https://api.openai.com/v1/embeddings"`.
    endpoint: String,
    /// Shared ureq agent with the configured timeout + keep-alive.
    agent: ureq::Agent,
}

impl OpenAiEmbedder {
    /// Construct from a validated [`OpenAiConfig`].
    ///
    /// # Errors
    ///
    /// - [`EmbedError::MissingApiKey`] if `config.api_key_env` names an
    ///   unset environment variable.
    /// - [`EmbedError::Config`] if the model name is not one of the
    ///   known OpenAI embedding models.
    pub fn from_config(config: &OpenAiConfig) -> Result<Self, EmbedError> {
        let api_key =
            std::env::var(&config.api_key_env).map_err(|_| EmbedError::MissingApiKey {
                var: config.api_key_env.clone(),
            })?;
        // Priority: explicit dim_override (user-supplied, trust-user
        // escape hatch from the hardcoding audit) > KNOWN_MODELS
        // lookup. Either satisfies the "we need to know the dim
        // up front" requirement; the override lets users point mnem
        // at a model mnem doesn't ship out of the box.
        let dim = match config.dim_override {
            Some(d) => d,
            None => known_dim(&config.model).ok_or_else(|| {
                EmbedError::Config(format!(
                    "unknown OpenAI embedding model '{}'; expected one of {:?}, \
                     or set `dim_override` in the config to pass through unknown models",
                    config.model,
                    KNOWN_MODELS.iter().map(|(m, _)| *m).collect::<Vec<_>>(),
                ))
            })?,
        };
        let endpoint = format!("{}/v1/embeddings", config.base_url.trim_end_matches('/'));
        let agent = ureq::AgentBuilder::new()
            .timeout(Duration::from_secs(config.timeout_secs))
            .build();
        Ok(Self {
            model_bare: config.model.clone(),
            model_fq: format!("openai:{}", config.model),
            dim,
            api_key,
            endpoint,
            agent,
        })
    }

    /// Send one POST with up to `MAX_BATCH` inputs. Caller splits.
    fn post_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, EmbedError> {
        #[derive(Serialize)]
        struct Req<'a> {
            model: &'a str,
            input: &'a [&'a str],
        }
        #[derive(Deserialize)]
        struct Resp {
            data: Vec<Datum>,
        }
        #[derive(Deserialize)]
        struct Datum {
            embedding: Vec<f32>,
            index: usize,
        }

        let body = Req {
            model: &self.model_bare,
            input: texts,
        };
        let resp = self
            .agent
            .post(&self.endpoint)
            .set("Authorization", &format!("Bearer {}", self.api_key))
            .set("Content-Type", "application/json")
            .send_json(&body)
            .map_err(classify_ureq_error)?;

        let parsed: Resp = decode_json(resp)?;

        if parsed.data.len() != texts.len() {
            return Err(EmbedError::Decode(format!(
                "OpenAI returned {} embeddings for {} inputs",
                parsed.data.len(),
                texts.len(),
            )));
        }

        // OpenAI guarantees `index` matches input order, but sort
        // defensively so an unexpected reordering never silently
        // mis-aligns with the caller's texts.
        let mut data = parsed.data;
        data.sort_by_key(|d| d.index);

        let mut out = Vec::with_capacity(data.len());
        for d in data {
            if d.embedding.len() as u32 != self.dim {
                return Err(EmbedError::DimMismatch {
                    expected: self.dim,
                    got: d.embedding.len() as u32,
                });
            }
            out.push(d.embedding);
        }
        Ok(out)
    }
}

impl Embedder for OpenAiEmbedder {
    fn model(&self) -> &str {
        &self.model_fq
    }

    fn dim(&self) -> u32 {
        self.dim
    }

    fn embed(&self, text: &str) -> Result<Vec<f32>, EmbedError> {
        let mut v = self.post_batch(&[text])?;
        Ok(v.pop().expect("post_batch returned 1 for 1 input"))
    }

    fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, EmbedError> {
        let mut out = Vec::with_capacity(texts.len());
        for chunk in texts.chunks(MAX_BATCH) {
            let part = self.post_batch(chunk)?;
            out.extend(part);
        }
        Ok(out)
    }

    fn manifest(&self) -> EmbedderManifest {
        // Placeholder, per Gap 15 solution.md: `text-embedding-3-small`
        // sits around 0.27 for unrelated text in the MTEB calibration
        // set. This is a single value across the OpenAI family until
        // we measure `3-large` and `ada-002` independently; the per-
        // model override path lives in `dim_override` and will be
        // extended to `noise_floor_override` if measurement reveals
        // divergence.
        EmbedderManifest::new(self.model_fq.clone(), self.dim, 0.27)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn missing_api_key_is_surfaced() {
        // Use a var name that is vanishingly unlikely to be set in any
        // realistic CI environment. We do NOT `set_var` / `remove_var`
        // because those are `unsafe` in Rust 2024 and this crate
        // `#![forbid(unsafe_code)]`s them.
        let var = "MNEM_TEST_OPENAI_KEY_NEVER_SET_bc6a78c1fd3e4a9f";
        let cfg = OpenAiConfig {
            model: "text-embedding-3-small".into(),
            api_key_env: var.into(),
            ..Default::default()
        };
        let e = OpenAiEmbedder::from_config(&cfg).unwrap_err();
        match e {
            EmbedError::MissingApiKey { var: got } => assert_eq!(got, var),
            other => panic!("expected MissingApiKey, got {other:?}"),
        }
    }

    #[test]
    fn known_dim_maps_shipped_models() {
        assert_eq!(known_dim("text-embedding-3-small"), Some(1536));
        assert_eq!(known_dim("text-embedding-3-large"), Some(3072));
        assert_eq!(known_dim("text-embedding-ada-002"), Some(1536));
        assert_eq!(known_dim("text-embedding-3-superhuge"), None);
    }
}
