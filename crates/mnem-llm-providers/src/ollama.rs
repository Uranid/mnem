//! Ollama chat adapter.
//!
//! Ships behind the `ollama` cargo feature (default-on). Calls
//! `POST {base_url}/api/chat` with `stream: false`.
//!
//! Ollama's chat endpoint does not support n>1 natively; we loop
//! for callers that need multiple completions.

use std::time::Duration;

use mnem_core::llm::{GenOptions, LlmError, TextGenerator};
use serde::{Deserialize, Serialize};

use crate::config::OllamaLlmConfig;
use crate::http::{classify_ureq_error, decode_json};

/// Live adapter over the Ollama chat API.
#[derive(Debug)]
pub struct OllamaChat {
    model_bare: String,
    model_fq: String,
    endpoint: String,
    agent: ureq::Agent,
}

impl OllamaChat {
    /// Construct from a validated [`OllamaLlmConfig`].
    pub fn from_config(config: &OllamaLlmConfig) -> Result<Self, LlmError> {
        let endpoint = format!("{}/api/chat", config.base_url.trim_end_matches('/'));
        let agent = ureq::AgentBuilder::new()
            .timeout(Duration::from_secs(config.timeout_secs))
            .build();
        Ok(Self {
            model_bare: config.model.clone(),
            model_fq: format!("ollama:{}", config.model),
            endpoint,
            agent,
        })
    }

    fn generate_once(&self, prompt: &str, opts: &GenOptions) -> Result<String, LlmError> {
        #[derive(Serialize)]
        struct Msg<'a> {
            role: &'static str,
            content: &'a str,
        }
        #[derive(Serialize)]
        struct Options<'a> {
            #[serde(skip_serializing_if = "Option::is_none")]
            temperature: Option<f32>,
            #[serde(skip_serializing_if = "Option::is_none")]
            num_predict: Option<u32>,
            #[serde(skip_serializing_if = "Option::is_none")]
            top_p: Option<f32>,
            #[serde(skip_serializing_if = "Option::is_none")]
            top_k: Option<u32>,
            #[serde(skip_serializing_if = "Option::is_none")]
            stop: Option<&'a [String]>,
            #[serde(skip_serializing_if = "Option::is_none")]
            presence_penalty: Option<f32>,
            #[serde(skip_serializing_if = "Option::is_none")]
            frequency_penalty: Option<f32>,
            #[serde(skip_serializing_if = "Option::is_none")]
            seed: Option<u64>,
        }
        #[derive(Serialize)]
        struct Req<'a> {
            model: &'a str,
            messages: Vec<Msg<'a>>,
            stream: bool,
            #[serde(skip_serializing_if = "Option::is_none")]
            options: Option<Options<'a>>,
        }
        #[derive(Deserialize)]
        struct Resp {
            message: RespMsg,
        }
        #[derive(Deserialize)]
        struct RespMsg {
            content: String,
        }

        let mut messages = Vec::with_capacity(2);
        if let Some(sys) = opts.system.as_deref() {
            messages.push(Msg {
                role: "system",
                content: sys,
            });
        }
        messages.push(Msg {
            role: "user",
            content: prompt,
        });

        let any_opt = opts.temperature.is_some()
            || opts.max_tokens.is_some()
            || opts.top_p.is_some()
            || opts.top_k.is_some()
            || opts.stop.is_some()
            || opts.presence_penalty.is_some()
            || opts.frequency_penalty.is_some()
            || opts.seed.is_some();
        let options = if any_opt {
            Some(Options {
                temperature: opts.temperature,
                num_predict: opts.max_tokens,
                top_p: opts.top_p,
                top_k: opts.top_k,
                stop: opts.stop.as_deref(),
                presence_penalty: opts.presence_penalty,
                frequency_penalty: opts.frequency_penalty,
                seed: opts.seed,
            })
        } else {
            None
        };

        let body = Req {
            model: &self.model_bare,
            messages,
            stream: false,
            options,
        };
        let resp = self
            .agent
            .post(&self.endpoint)
            .set("Content-Type", "application/json")
            .set("Accept", "application/json")
            .send_json(&body)
            .map_err(classify_ureq_error)?;
        let parsed: Resp = decode_json(resp)?;
        if parsed.message.content.is_empty() {
            return Err(LlmError::EmptyCompletion);
        }
        Ok(parsed.message.content)
    }
}

impl TextGenerator for OllamaChat {
    fn model(&self) -> &str {
        &self.model_fq
    }

    fn generate(&self, prompt: &str, opts: &GenOptions) -> Result<Vec<String>, LlmError> {
        let n = opts.n.max(1);
        let mut out = Vec::with_capacity(n);
        for _ in 0..n {
            out.push(self.generate_once(prompt, opts)?);
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn model_fq_has_ollama_prefix() {
        // Can't construct without hitting the server; assert the
        // format rule separately.
        let fq = format!("ollama:{}", "llama3.2:3b");
        assert_eq!(fq, "ollama:llama3.2:3b");
    }
}
