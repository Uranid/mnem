// Provider-name proper nouns are well-known external identifiers.
#![allow(clippy::doc_markdown)]

//! [`ProviderConfig`] and the [`open`] factory for LLM text-generation
//! adapters. Mirrors `mnem-embed-providers::config`.

use std::sync::Arc;

use mnem_core::llm::{LlmError, TextGenerator};
use serde::{Deserialize, Serialize};

/// Tagged enum over the shipped providers. TOML shape:
///
/// ```toml
/// [llm]
/// provider    = "openai"
/// model       = "gpt-4o-mini"
/// api_key_env = "OPENAI_API_KEY"
/// ```
///
/// or
///
/// ```toml
/// [llm]
/// provider = "ollama"
/// model    = "llama3.2:3b"
/// base_url = "http://localhost:11434"
/// ```
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase", tag = "provider")]
pub enum ProviderConfig {
    /// OpenAI chat completions API. Requires API key.
    Openai(OpenAiLlmConfig),
    /// Ollama local chat. No auth; default localhost:11434.
    Ollama(OllamaLlmConfig),
}

/// Config for the OpenAI chat-completions adapter.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct OpenAiLlmConfig {
    /// Bare model name, e.g. `"gpt-4o-mini"`. Final fq id is
    /// `"openai:<model>"`.
    pub model: String,
    /// Env var holding the API key. Default `"OPENAI_API_KEY"`.
    #[serde(default = "default_openai_env")]
    pub api_key_env: String,
    /// Base URL. Default `"https://api.openai.com"`.
    #[serde(default = "default_openai_base")]
    pub base_url: String,
    /// Per-request timeout in seconds. Default 60 (LLMs are slower
    /// than embedders).
    #[serde(default = "default_timeout")]
    pub timeout_secs: u64,
}

impl Default for OpenAiLlmConfig {
    fn default() -> Self {
        Self {
            model: "gpt-4o-mini".into(),
            api_key_env: default_openai_env(),
            base_url: default_openai_base(),
            timeout_secs: default_timeout(),
        }
    }
}

/// Config for the Ollama chat adapter.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct OllamaLlmConfig {
    /// Bare model tag, e.g. `"llama3.2:3b"`. Final fq id is
    /// `"ollama:<tag>"`.
    pub model: String,
    /// Base URL of the Ollama server. Default
    /// `"http://localhost:11434"`.
    #[serde(default = "default_ollama_base")]
    pub base_url: String,
    /// Per-request timeout in seconds. Default 120 (local LLMs on
    /// CPU can be slower).
    #[serde(default = "default_ollama_timeout")]
    pub timeout_secs: u64,
}

impl Default for OllamaLlmConfig {
    fn default() -> Self {
        Self {
            model: "llama3.2:3b".into(),
            base_url: default_ollama_base(),
            timeout_secs: default_ollama_timeout(),
        }
    }
}

fn default_openai_env() -> String {
    "OPENAI_API_KEY".into()
}
fn default_openai_base() -> String {
    "https://api.openai.com".into()
}
fn default_ollama_base() -> String {
    "http://localhost:11434".into()
}
const fn default_timeout() -> u64 {
    60
}
const fn default_ollama_timeout() -> u64 {
    120
}

/// Construct a live [`TextGenerator`] from a [`ProviderConfig`].
///
/// # Errors
///
/// - [`LlmError::Config`] for feature-gated providers compiled out,
///   or if the env var named by `api_key_env` is unset.
pub fn open(cfg: &ProviderConfig) -> Result<Arc<dyn TextGenerator>, LlmError> {
    match cfg {
        #[cfg(feature = "openai")]
        ProviderConfig::Openai(c) => {
            let g = crate::openai::OpenAiChat::from_config(c)?;
            Ok(Arc::new(g))
        }
        #[cfg(not(feature = "openai"))]
        ProviderConfig::Openai(_) => Err(LlmError::Config(
            "this mnem-llm-providers build was compiled without the `openai` feature".into(),
        )),

        #[cfg(feature = "ollama")]
        ProviderConfig::Ollama(c) => {
            let g = crate::ollama::OllamaChat::from_config(c)?;
            Ok(Arc::new(g))
        }
        #[cfg(not(feature = "ollama"))]
        ProviderConfig::Ollama(_) => Err(LlmError::Config(
            "this mnem-llm-providers build was compiled without the `ollama` feature".into(),
        )),
    }
}

/// Read an API key from the environment, returning a pointed-at
/// `LlmError::Config` when unset. Used by network adapters.
pub(crate) fn read_api_key(var: &str) -> Result<String, LlmError> {
    std::env::var(var)
        .map_err(|_| LlmError::Config(format!("environment variable {var} is not set")))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn openai_config_toml_round_trip() {
        let cfg = ProviderConfig::Openai(OpenAiLlmConfig::default());
        let s = toml::to_string(&cfg).unwrap();
        assert!(s.contains("provider = \"openai\""));
        let back: ProviderConfig = toml::from_str(&s).unwrap();
        assert_eq!(cfg, back);
    }

    #[test]
    fn ollama_config_toml_round_trip() {
        let cfg = ProviderConfig::Ollama(OllamaLlmConfig::default());
        let s = toml::to_string(&cfg).unwrap();
        assert!(s.contains("provider = \"ollama\""));
        let back: ProviderConfig = toml::from_str(&s).unwrap();
        assert_eq!(cfg, back);
    }

    #[test]
    fn read_api_key_missing_is_config_error() {
        let var = "MNEM_TEST_LLM_KEY_NEVER_SET_c3e7f1b5d9a2";
        let e = read_api_key(var).unwrap_err();
        match e {
            LlmError::Config(msg) => assert!(msg.contains(var)),
            other => panic!("expected Config, got {other:?}"),
        }
    }
}
