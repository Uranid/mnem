// Provider adapter: OpenAI is a proper-noun identifier.
#![allow(clippy::doc_markdown)]

//! OpenAI chat-completions adapter.
//!
//! Ships behind the `openai` cargo feature (default-on). Calls
//! `POST {base_url}/v1/chat/completions` with Bearer auth.

use std::time::Duration;

use mnem_core::llm::{GenOptions, LlmError, TextGenerator};
use serde::{Deserialize, Serialize};

use crate::config::{OpenAiLlmConfig, read_api_key};
use crate::http::{classify_ureq_error, decode_json};

/// Live adapter over the OpenAI chat completions API.
#[derive(Debug)]
pub struct OpenAiChat {
    model_bare: String,
    model_fq: String,
    api_key: String,
    endpoint: String,
    agent: ureq::Agent,
}

impl OpenAiChat {
    /// Construct from a validated [`OpenAiLlmConfig`].
    ///
    /// # Errors
    ///
    /// [`LlmError::Config`] if `config.api_key_env` names an unset
    /// environment variable.
    pub fn from_config(config: &OpenAiLlmConfig) -> Result<Self, LlmError> {
        let api_key = read_api_key(&config.api_key_env)?;
        let endpoint = format!(
            "{}/v1/chat/completions",
            config.base_url.trim_end_matches('/')
        );
        let agent = ureq::AgentBuilder::new()
            .timeout(Duration::from_secs(config.timeout_secs))
            .build();
        Ok(Self {
            model_bare: config.model.clone(),
            model_fq: format!("openai:{}", config.model),
            api_key,
            endpoint,
            agent,
        })
    }
}

impl TextGenerator for OpenAiChat {
    fn model(&self) -> &str {
        &self.model_fq
    }

    fn generate(&self, prompt: &str, opts: &GenOptions) -> Result<Vec<String>, LlmError> {
        #[derive(Serialize)]
        struct Msg<'a> {
            role: &'static str,
            content: &'a str,
        }
        #[derive(Serialize)]
        struct Req<'a> {
            model: &'a str,
            messages: Vec<Msg<'a>>,
            n: usize,
            #[serde(skip_serializing_if = "Option::is_none")]
            max_tokens: Option<u32>,
            #[serde(skip_serializing_if = "Option::is_none")]
            temperature: Option<f32>,
            #[serde(skip_serializing_if = "Option::is_none")]
            top_p: Option<f32>,
            #[serde(skip_serializing_if = "Option::is_none")]
            stop: Option<&'a [String]>,
            #[serde(skip_serializing_if = "Option::is_none")]
            presence_penalty: Option<f32>,
            #[serde(skip_serializing_if = "Option::is_none")]
            frequency_penalty: Option<f32>,
            #[serde(skip_serializing_if = "Option::is_none")]
            seed: Option<u64>,
        }
        #[derive(Deserialize)]
        struct Resp {
            choices: Vec<Choice>,
        }
        #[derive(Deserialize)]
        struct Choice {
            message: Message,
        }
        #[derive(Deserialize)]
        struct Message {
            content: Option<String>,
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

        let body = Req {
            model: &self.model_bare,
            messages,
            n: opts.n.max(1),
            max_tokens: opts.max_tokens,
            temperature: opts.temperature,
            top_p: opts.top_p,
            stop: opts.stop.as_deref(),
            presence_penalty: opts.presence_penalty,
            frequency_penalty: opts.frequency_penalty,
            seed: opts.seed,
            // top_k is silently dropped: OpenAI v1/chat/completions
            // does not accept it. Documented in GenOptions docstring.
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

        if parsed.choices.is_empty() {
            return Err(LlmError::EmptyCompletion);
        }
        let out: Vec<String> = parsed
            .choices
            .into_iter()
            .filter_map(|c| c.message.content)
            .collect();
        if out.is_empty() {
            return Err(LlmError::EmptyCompletion);
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn missing_api_key_is_config_error() {
        let var = "MNEM_TEST_OPENAI_LLM_KEY_NEVER_SET_7f2c9e4a1b6d";
        let cfg = OpenAiLlmConfig {
            api_key_env: var.into(),
            ..Default::default()
        };
        let e = OpenAiChat::from_config(&cfg).unwrap_err();
        match e {
            LlmError::Config(msg) => assert!(msg.contains(var)),
            other => panic!("expected Config, got {other:?}"),
        }
    }
}
