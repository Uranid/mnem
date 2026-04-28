//! Voyage AI rerank adapter.
//!
//! Ships behind the `voyage` cargo feature (default-on). Calls
//! `POST {base_url}/v1/rerank` with Bearer-token auth, a JSON body
//! `{query, documents, model}`, and maps the `data[].relevance_score`
//! back to the caller's input order.

use std::time::Duration;

use mnem_core::rerank::{RerankError, Reranker};
use serde::{Deserialize, Serialize};

use crate::config::{VoyageConfig, read_api_key};
use crate::http::{classify_ureq_error, decode_json};

/// Maximum documents sent in a single rerank call. Voyage's per-request
/// cap is 1000; we batch well below for the same reasons as Cohere.
const MAX_BATCH: usize = 100;

/// Live adapter over the Voyage AI rerank REST API.
#[derive(Debug)]
pub struct VoyageReranker {
    model_bare: String,
    model_fq: String,
    api_key: String,
    endpoint: String,
    agent: ureq::Agent,
}

impl VoyageReranker {
    /// Construct from a validated [`VoyageConfig`].
    ///
    /// # Errors
    ///
    /// - [`RerankError::Config`] if `config.api_key_env` names an unset
    ///   environment variable.
    pub fn from_config(config: &VoyageConfig) -> Result<Self, RerankError> {
        let api_key = read_api_key(&config.api_key_env)?;
        let endpoint = format!("{}/v1/rerank", config.base_url.trim_end_matches('/'));
        let agent = ureq::AgentBuilder::new()
            .timeout(Duration::from_secs(config.timeout_secs))
            .build();
        Ok(Self {
            model_bare: config.model.clone(),
            model_fq: format!("voyage:{}", config.model),
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
            data: Vec<Datum>,
        }
        #[derive(Deserialize)]
        struct Datum {
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

        if parsed.data.len() != docs.len() {
            return Err(RerankError::ScoreCountMismatch {
                expected: docs.len(),
                got: parsed.data.len(),
            });
        }

        let mut out = vec![0.0f32; docs.len()];
        for d in parsed.data {
            if d.index >= docs.len() {
                return Err(RerankError::Decode(format!(
                    "voyage returned out-of-range index {} for batch of {}",
                    d.index,
                    docs.len()
                )));
            }
            out[d.index] = d.relevance_score;
        }
        Ok(out)
    }
}

impl Reranker for VoyageReranker {
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
        let var = "MNEM_TEST_VOYAGE_KEY_NEVER_SET_4b8d2e6a1c3f";
        let cfg = VoyageConfig {
            api_key_env: var.into(),
            ..Default::default()
        };
        let e = VoyageReranker::from_config(&cfg).unwrap_err();
        match e {
            RerankError::Config(msg) => assert!(msg.contains(var)),
            other => panic!("expected Config error, got {other:?}"),
        }
    }
}
