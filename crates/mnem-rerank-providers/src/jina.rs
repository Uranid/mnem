//! Jina AI rerank adapter.
//!
//! Ships behind the `jina` cargo feature (default-on). Calls
//! `POST {base_url}/v1/rerank` with Bearer-token auth, a JSON body
//! `{model, query, documents}`, and maps the `results[].relevance_score`
//! back to the caller's input order.

use std::time::Duration;

use mnem_core::rerank::{RerankError, Reranker};
use serde::{Deserialize, Serialize};

use crate::config::{JinaConfig, read_api_key};
use crate::http::{classify_ureq_error, decode_json};

/// Maximum documents sent in a single rerank call. Jina accepts up to
/// 2048 on higher tiers; we cap at 100 for predictable latency.
const MAX_BATCH: usize = 100;

/// Live adapter over the Jina AI rerank REST API.
#[derive(Debug)]
pub struct JinaReranker {
    model_bare: String,
    model_fq: String,
    api_key: String,
    endpoint: String,
    agent: ureq::Agent,
}

impl JinaReranker {
    /// Construct from a validated [`JinaConfig`].
    ///
    /// # Errors
    ///
    /// - [`RerankError::Config`] if `cfg.api_key_env` names an unset
    ///   environment variable.
    pub fn from_config(cfg: &JinaConfig) -> Result<Self, RerankError> {
        let api_key = read_api_key(&cfg.api_key_env)?;
        let endpoint = format!("{}/v1/rerank", cfg.base_url.trim_end_matches('/'));
        let agent = ureq::AgentBuilder::new()
            .timeout(Duration::from_secs(cfg.timeout_secs))
            .build();
        Ok(Self {
            model_bare: cfg.model.clone(),
            model_fq: format!("jina:{}", cfg.model),
            api_key,
            endpoint,
            agent,
        })
    }

    fn post_batch(&self, query: &str, docs: &[&str]) -> Result<Vec<f32>, RerankError> {
        #[derive(Serialize)]
        struct Req<'a> {
            model: &'a str,
            query: &'a str,
            documents: &'a [&'a str],
        }
        #[derive(Deserialize)]
        struct Resp {
            results: Vec<RerankResult>,
        }
        #[derive(Deserialize)]
        struct RerankResult {
            index: usize,
            relevance_score: f32,
        }

        let body = Req {
            model: &self.model_bare,
            query,
            documents: docs,
        };
        let resp = self
            .agent
            .post(&self.endpoint)
            .set("Authorization", &format!("Bearer {}", self.api_key))
            .set("Content-Type", "application/json")
            .set("Accept", "application/json")
            .send_json(&body)
            .map_err(classify_ureq_error)?;

        let parsed: Resp = decode_json(resp)?;

        if parsed.results.len() != docs.len() {
            return Err(RerankError::ScoreCountMismatch {
                expected: docs.len(),
                got: parsed.results.len(),
            });
        }

        let mut out = vec![0.0f32; docs.len()];
        for r in parsed.results {
            if r.index >= docs.len() {
                return Err(RerankError::Decode(format!(
                    "jina returned out-of-range index {} for batch of {}",
                    r.index,
                    docs.len()
                )));
            }
            out[r.index] = r.relevance_score;
        }
        Ok(out)
    }
}

impl Reranker for JinaReranker {
    fn model(&self) -> &str {
        &self.model_fq
    }

    fn rerank(&self, query: &str, candidates: &[&str]) -> Result<Vec<f32>, RerankError> {
        if candidates.is_empty() {
            return Ok(Vec::new());
        }
        let mut out = Vec::with_capacity(candidates.len());
        for chunk in candidates.chunks(MAX_BATCH) {
            let part = self.post_batch(query, chunk)?;
            out.extend(part);
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn missing_api_key_is_config_error() {
        let var = "MNEM_TEST_JINA_KEY_NEVER_SET_6c4f2a8b1d3e";
        let cfg = JinaConfig {
            api_key_env: var.into(),
            ..Default::default()
        };
        let e = JinaReranker::from_config(&cfg).unwrap_err();
        match e {
            RerankError::Config(msg) => assert!(msg.contains(var)),
            other => panic!("expected Config error, got {other:?}"),
        }
    }
}
