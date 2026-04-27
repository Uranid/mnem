//! [`ProviderConfig`] and the [`open`] factory.
//!
//! Serde-friendly so the CLI can persist in `config.toml` under
//! `[rerank]`. API keys are NEVER stored; only the name of the env var
//! that holds the key.

use std::sync::Arc;

use mnem_core::rerank::{RerankError, Reranker};
use serde::{Deserialize, Serialize};

/// Tagged enum over the shipped providers. TOML representation:
///
/// ```toml
/// [rerank]
/// provider    = "cohere"
/// model       = "rerank-v3.5"
/// api_key_env = "COHERE_API_KEY"
/// ```
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase", tag = "provider")]
pub enum ProviderConfig {
    /// Cohere rerank API (v2 endpoint). Requires API key.
    Cohere(CohereConfig),
    /// Voyage AI rerank API. Requires API key.
    Voyage(VoyageConfig),
    /// Jina AI rerank API. Requires API key.
    Jina(JinaConfig),
}

/// Config for the Cohere rerank adapter.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct CohereConfig {
    /// Bare model name, e.g. `"rerank-v3.5"`. The final
    /// [`Reranker::model`] string will be `"cohere:<model>"`.
    pub model: String,
    /// Name of the env var holding the API key. Default `"COHERE_API_KEY"`.
    #[serde(default = "default_cohere_env")]
    pub api_key_env: String,
    /// Base URL. Default `"https://api.cohere.com"`.
    #[serde(default = "default_cohere_base")]
    pub base_url: String,
    /// Per-request timeout in seconds. Default 30.
    #[serde(default = "default_timeout")]
    pub timeout_secs: u64,
}

impl Default for CohereConfig {
    fn default() -> Self {
        Self {
            // v3.5 is Cohere's current best-quality multilingual
            // cross-encoder; bumped from the v3.0 first-shipping
            // default per the user's "use v3 / latest" ask.
            model: "rerank-v3.5".into(),
            api_key_env: default_cohere_env(),
            base_url: default_cohere_base(),
            timeout_secs: default_timeout(),
        }
    }
}

/// Config for the Voyage rerank adapter.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct VoyageConfig {
    /// Bare model name, e.g. `"rerank-2.5"`. The final [`Reranker::model`]
    /// string will be `"voyage:<model>"`.
    pub model: String,
    /// Name of the env var holding the API key. Default
    /// `"VOYAGE_API_KEY"`.
    #[serde(default = "default_voyage_env")]
    pub api_key_env: String,
    /// Base URL. Default `"https://api.voyageai.com"`.
    #[serde(default = "default_voyage_base")]
    pub base_url: String,
    /// Per-request timeout in seconds. Default 30.
    #[serde(default = "default_timeout")]
    pub timeout_secs: u64,
}

impl Default for VoyageConfig {
    fn default() -> Self {
        Self {
            // rerank-2.5 is Voyage's current flagship cross-encoder
            // per early-2026 benchmarks (highest BEIR among commercial
            // two-stage rerankers by ZeroEntropy's comparison).
            model: "rerank-2.5".into(),
            api_key_env: default_voyage_env(),
            base_url: default_voyage_base(),
            timeout_secs: default_timeout(),
        }
    }
}

/// Config for the Jina rerank adapter.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct JinaConfig {
    /// Bare model name, e.g. `"jina-reranker-v3"`.
    /// The final [`Reranker::model`] string will be `"jina:<model>"`.
    pub model: String,
    /// Name of the env var holding the API key. Default
    /// `"JINA_API_KEY"`.
    #[serde(default = "default_jina_env")]
    pub api_key_env: String,
    /// Base URL. Default `"https://api.jina.ai"`.
    #[serde(default = "default_jina_base")]
    pub base_url: String,
    /// Per-request timeout in seconds. Default 30.
    #[serde(default = "default_timeout")]
    pub timeout_secs: u64,
}

impl Default for JinaConfig {
    fn default() -> Self {
        Self {
            // v3 uses listwise architecture (sees all candidates
            // jointly) and reports the highest public BEIR nDCG@10
            // among commercial rerankers (~0.62). Bumped from v2
            // per internal evaluation of listwise vs pointwise
            // architectures on NFCorpus.
            model: "jina-reranker-v3".into(),
            api_key_env: default_jina_env(),
            base_url: default_jina_base(),
            timeout_secs: default_timeout(),
        }
    }
}

fn default_cohere_env() -> String {
    "COHERE_API_KEY".into()
}
fn default_cohere_base() -> String {
    "https://api.cohere.com".into()
}
fn default_voyage_env() -> String {
    "VOYAGE_API_KEY".into()
}
fn default_voyage_base() -> String {
    "https://api.voyageai.com".into()
}
fn default_jina_env() -> String {
    "JINA_API_KEY".into()
}
fn default_jina_base() -> String {
    "https://api.jina.ai".into()
}
const fn default_timeout() -> u64 {
    30
}

/// Construct a live [`Reranker`] from a [`ProviderConfig`].
///
/// Reads the API key from the process environment at construction time.
/// If the configured provider is feature-disabled at compile time,
/// returns [`RerankError::Config`].
///
/// # Errors
///
/// - [`RerankError::Config`] for feature-gated providers compiled out,
///   or if the env var named by `api_key_env` is unset.
pub fn open(cfg: &ProviderConfig) -> Result<Arc<dyn Reranker>, RerankError> {
    match cfg {
        #[cfg(feature = "cohere")]
        ProviderConfig::Cohere(c) => {
            let r = crate::cohere::CohereReranker::from_config(c)?;
            Ok(Arc::new(r))
        }
        #[cfg(not(feature = "cohere"))]
        ProviderConfig::Cohere(_) => Err(RerankError::Config(
            "this mnem-rerank-providers build was compiled without the `cohere` feature".into(),
        )),

        #[cfg(feature = "voyage")]
        ProviderConfig::Voyage(c) => {
            let r = crate::voyage::VoyageReranker::from_config(c)?;
            Ok(Arc::new(r))
        }
        #[cfg(not(feature = "voyage"))]
        ProviderConfig::Voyage(_) => Err(RerankError::Config(
            "this mnem-rerank-providers build was compiled without the `voyage` feature".into(),
        )),

        #[cfg(feature = "jina")]
        ProviderConfig::Jina(c) => {
            let r = crate::jina::JinaReranker::from_config(c)?;
            Ok(Arc::new(r))
        }
        #[cfg(not(feature = "jina"))]
        ProviderConfig::Jina(_) => Err(RerankError::Config(
            "this mnem-rerank-providers build was compiled without the `jina` feature".into(),
        )),
    }
}

/// Read an API key from the process environment, returning a
/// [`RerankError::Config`] with a pointed-at env-var name when unset.
/// Used by every network adapter's `from_config`.
pub(crate) fn read_api_key(var: &str) -> Result<String, RerankError> {
    std::env::var(var)
        .map_err(|_| RerankError::Config(format!("environment variable {var} is not set")))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cohere_config_toml_round_trip() {
        let cfg = ProviderConfig::Cohere(CohereConfig::default());
        let s = toml::to_string(&cfg).unwrap();
        assert!(s.contains("provider = \"cohere\""));
        assert!(s.contains("rerank-v3.5"));
        let back: ProviderConfig = toml::from_str(&s).unwrap();
        assert_eq!(cfg, back);
    }

    #[test]
    fn voyage_config_toml_round_trip() {
        let cfg = ProviderConfig::Voyage(VoyageConfig::default());
        let s = toml::to_string(&cfg).unwrap();
        assert!(s.contains("provider = \"voyage\""));
        let back: ProviderConfig = toml::from_str(&s).unwrap();
        assert_eq!(cfg, back);
    }

    #[test]
    fn jina_config_toml_round_trip() {
        let cfg = ProviderConfig::Jina(JinaConfig::default());
        let s = toml::to_string(&cfg).unwrap();
        assert!(s.contains("provider = \"jina\""));
        let back: ProviderConfig = toml::from_str(&s).unwrap();
        assert_eq!(cfg, back);
    }

    #[test]
    fn read_api_key_missing_is_config_error() {
        let var = "MNEM_TEST_RERANK_KEY_NEVER_SET_a9e1c3d5f7b9";
        let e = read_api_key(var).unwrap_err();
        match e {
            RerankError::Config(msg) => assert!(msg.contains(var)),
            other => panic!("expected Config error, got {other:?}"),
        }
    }
}
