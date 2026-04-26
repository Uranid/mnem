// SPLADE, BGE-M3 are proper-noun identifiers of learned-sparse models.
#![allow(clippy::doc_markdown)]

//! HTTP-sidecar SPLADE / BGE-M3 adapter.
//!
//! POSTs `{text, model}` to `<base_url>/encode` and expects a
//! canonical `{indices: [u32], values: [f32], vocab_id: str}` reply.
//! Any indices-not-ascending or length-mismatch in the response is
//! normalised by [`SparseEmbed::from_unsorted`] so malformed sidecar
//! output never crashes retrieval.
//!
//! Reference sidecar image layout lives under
//! `benchmarks/adapters/splade-sidecar/` in the mnem-compatible
//! benchmarking harness (any repo that follows that adapter shape).

use std::time::Duration;

use mnem_core::sparse::{SparseEmbed, SparseEncoder, SparseError};
use serde::{Deserialize, Serialize};

use crate::config::SidecarConfig;

/// Maximum body bytes captured on HTTP error responses.
const ERR_BODY_CAP: usize = 4096;

/// HTTP-sidecar sparse encoder.
///
/// Posts `{text, model}` JSON to a caller-configured `<base_url>/encode`
/// endpoint and expects `{indices: [u32], values: [f32], vocab_id: str}`
/// back. The typical deployment is a small Python FastAPI server
/// wrapping SPLADE / BGE-M3 / OpenSearch neural-sparse; any image
/// that accepts the `{text, model}` POST / `{indices, values,
/// vocab_id}` JSON reply shape works.
///
/// Sync + ureq-backed (rustls). No tokio. Matches the sibling
/// `mnem-embed-providers` / `mnem-rerank-providers` adapter style.
#[derive(Debug)]
pub struct SidecarSparseEncoder {
    endpoint: String,
    model: String,
    model_fq: String,
    vocab_id: String,
    agent: ureq::Agent,
}

impl SidecarSparseEncoder {
    /// Construct from a [`SidecarConfig`]. No network calls; the
    /// encoder lazily dials the sidecar on first `encode()`.
    ///
    /// # Errors
    ///
    /// - [`SparseError::Config`] if the URL is empty.
    pub fn from_config(config: &SidecarConfig) -> Result<Self, SparseError> {
        if config.base_url.trim().is_empty() {
            return Err(SparseError::Config("sidecar base_url is empty".into()));
        }
        let agent = ureq::AgentBuilder::new()
            .timeout(Duration::from_secs(config.timeout_secs))
            .build();
        Ok(Self {
            endpoint: format!("{}/encode", config.base_url.trim_end_matches('/')),
            model: config.model.clone(),
            model_fq: format!("sidecar:{}", config.model),
            vocab_id: config.vocab_id.clone(),
            agent,
        })
    }
}

impl SparseEncoder for SidecarSparseEncoder {
    fn model(&self) -> &str {
        &self.model_fq
    }

    fn vocab_id(&self) -> &str {
        &self.vocab_id
    }

    fn encode(&self, text: &str) -> Result<SparseEmbed, SparseError> {
        if text.trim().is_empty() {
            return Err(SparseError::EmptyInput);
        }
        #[derive(Serialize)]
        struct Req<'a> {
            text: &'a str,
            model: &'a str,
        }
        #[derive(Deserialize)]
        struct Resp {
            indices: Vec<u32>,
            values: Vec<f32>,
            vocab_id: Option<String>,
        }

        let body = Req {
            text,
            model: &self.model,
        };
        let resp = self
            .agent
            .post(&self.endpoint)
            .set("Content-Type", "application/json")
            .set("Accept", "application/json")
            .send_json(&body)
            .map_err(classify_ureq_error)?;
        let parsed: Resp = resp
            .into_json()
            .map_err(|e| SparseError::Inference(e.to_string()))?;

        // Trust the sidecar's declared vocab_id if it provides one;
        // otherwise fall back to the adapter's configured vocab_id.
        let vocab = parsed.vocab_id.unwrap_or_else(|| self.vocab_id.clone());

        // Normalise via from_unsorted so we tolerate sidecars that
        // emit non-ascending indices or duplicate tokens.
        if parsed.indices.len() != parsed.values.len() {
            return Err(SparseError::Inference(format!(
                "sidecar returned indices.len={} values.len={}",
                parsed.indices.len(),
                parsed.values.len(),
            )));
        }
        let pairs = parsed.indices.into_iter().zip(parsed.values);
        Ok(SparseEmbed::from_unsorted(pairs, vocab))
    }
}

fn classify_ureq_error(e: ureq::Error) -> SparseError {
    match e {
        ureq::Error::Status(status, resp) => {
            let body = resp
                .into_string()
                .unwrap_or_else(|ioe| format!("<body read: {ioe}>"));
            let body = if body.len() > ERR_BODY_CAP {
                body.chars().take(ERR_BODY_CAP).collect()
            } else {
                body
            };
            SparseError::Inference(format!("sidecar HTTP {status}: {body}"))
        }
        ureq::Error::Transport(t) => SparseError::Network(t.to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_input_errors_without_network() {
        let cfg = SidecarConfig::default();
        let enc = SidecarSparseEncoder::from_config(&cfg).unwrap();
        assert!(matches!(
            enc.encode("").unwrap_err(),
            SparseError::EmptyInput
        ));
        assert!(matches!(
            enc.encode("   \n").unwrap_err(),
            SparseError::EmptyInput
        ));
    }

    #[test]
    fn empty_base_url_is_config_error() {
        let cfg = SidecarConfig {
            base_url: "".into(),
            ..Default::default()
        };
        assert!(matches!(
            SidecarSparseEncoder::from_config(&cfg).unwrap_err(),
            SparseError::Config(_)
        ));
    }

    #[test]
    fn model_fq_has_sidecar_prefix() {
        let cfg = SidecarConfig {
            model: "opensearch-doc-v3-distill".into(),
            ..Default::default()
        };
        let enc = SidecarSparseEncoder::from_config(&cfg).unwrap();
        assert_eq!(enc.model(), "sidecar:opensearch-doc-v3-distill");
    }

    #[test]
    fn vocab_id_passes_through() {
        let cfg = SidecarConfig {
            vocab_id: "bge-m3@250002".into(),
            ..Default::default()
        };
        let enc = SidecarSparseEncoder::from_config(&cfg).unwrap();
        assert_eq!(enc.vocab_id(), "bge-m3@250002");
    }
}
