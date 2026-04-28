// Provider-name proper nouns (OpenAI, Ollama) are well-known external
// identifiers; backticking every mention adds no signal.
#![allow(clippy::doc_markdown)]

//! [`ProviderConfig`] and the [`open`] factory.
//!
//! The config is serde-friendly so the CLI can load / store it in the
//! repo's `config.toml` under `[embed]`. API keys are NEVER stored;
//! only the name of the env var that holds the key.

use serde::{Deserialize, Serialize};

use crate::embedder::Embedder;
use crate::error::EmbedError;

/// Tagged enum over the shipped providers. TOML representation:
///
/// ```toml
/// [embed]
/// provider     = "openai"
/// model        = "text-embedding-3-small"
/// api_key_env  = "OPENAI_API_KEY"
/// ```
///
/// or
///
/// ```toml
/// [embed]
/// provider = "ollama"
/// model    = "nomic-embed-text"
/// base_url = "http://localhost:11434"
/// ```
///
/// or (native in-process ONNX, requires the `onnx` cargo feature):
///
/// ```toml
/// [embed]
/// provider   = "onnx"
/// model      = "bge-large-en-v1.5"
/// # max_length = 512   # optional; default = model ceiling
/// ```
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase", tag = "provider")]
pub enum ProviderConfig {
    /// OpenAI Embeddings API. Requires an API key in the environment.
    Openai(OpenAiConfig),
    /// Ollama local inference server. No auth; default at
    /// `http://localhost:11434`.
    Ollama(OllamaConfig),
    /// Native in-process ONNX encoder. Requires building with the
    /// `onnx` cargo feature; otherwise [`open`] returns an actionable
    /// [`EmbedError::Config`] so operators can either rebuild or
    /// switch back to `ollama` / `openai`.
    Onnx(OnnxConfig),
}

/// Config for the OpenAI embeddings adapter.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct OpenAiConfig {
    /// Bare model name, e.g. `"text-embedding-3-small"`. The final
    /// [`Embedder::model`] string will be `"openai:<model>"`.
    pub model: String,
    /// Name of the env var holding the API key. Default
    /// `"OPENAI_API_KEY"`. The key itself is read at adapter
    /// construction time and is never persisted.
    #[serde(default = "default_openai_env")]
    pub api_key_env: String,
    /// Base URL. Override for Azure-OpenAI-compatible deployments or
    /// reverse proxies. Default `"https://api.openai.com"`.
    #[serde(default = "default_openai_base")]
    pub base_url: String,
    /// Per-request timeout in seconds. Default 30.
    #[serde(default = "default_timeout")]
    pub timeout_secs: u64,
    /// Explicit output dimension. When set, bypasses the internal
    /// `KNOWN_MODELS` allow-list so users can point mnem at a model
    /// mnem doesn't yet ship (new OpenAI releases, compatible
    /// third-party endpoints). When `None`, the adapter looks up the
    /// dim from its built-in list and refuses unknown models.
    ///
    /// Escape hatch added after the hardcoding audit. Use with care:
    /// the value MUST match the model's actual output dim, otherwise
    /// every write will fail with a [`crate::error::EmbedError::DimMismatch`]
    /// at the first embed call.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dim_override: Option<u32>,
}

impl Default for OpenAiConfig {
    fn default() -> Self {
        Self {
            model: "text-embedding-3-small".into(),
            api_key_env: default_openai_env(),
            base_url: default_openai_base(),
            timeout_secs: default_timeout(),
            dim_override: None,
        }
    }
}

/// Config for the Ollama embeddings adapter.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct OllamaConfig {
    /// Bare model name, e.g. `"nomic-embed-text"`. The final
    /// [`Embedder::model`] string will be `"ollama:<model>"`.
    pub model: String,
    /// Base URL of the Ollama server. Default
    /// `"http://localhost:11434"`.
    #[serde(default = "default_ollama_base")]
    pub base_url: String,
    /// Per-request timeout in seconds. Default 30.
    #[serde(default = "default_timeout")]
    pub timeout_secs: u64,
}

impl Default for OllamaConfig {
    fn default() -> Self {
        Self {
            model: "nomic-embed-text".into(),
            base_url: default_ollama_base(),
            timeout_secs: default_timeout(),
        }
    }
}

/// Native in-process ONNX embedder config. No network URL; the `model`
/// string resolves directly to a compiled-in `onnx::ModelKind` variant
/// (only available when the `onnx` cargo feature is enabled).
///
/// Kept visible even when the `onnx` feature is off, so a deserialised
/// `[embed] provider = "onnx"` block round-trips through TOML and
/// [`open`] can emit an actionable "rebuild with `--features onnx`"
/// error at construction time rather than a confusing deserialisation
/// failure.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct OnnxConfig {
    /// Model shortname. Known values map to `onnx::ModelKind` variants
    /// (only available when the `onnx` cargo feature is enabled):
    ///   - `"bge-large-en-v1.5"` (default; 1024-dim, English,
    ///      Apache-2.0, matches the MemPalace/BEIR headline embedder)
    ///   - `"bge-base-en-v1.5"` (768-dim; smaller footprint)
    ///   - `"bge-small-en-v1.5"` (384-dim; fastest)
    pub model: String,
    /// Optional tokenizer `max_length` override. `None` defers to the
    /// model's `default_max_length()` (512 for BGE). Values above the
    /// model's `positional_limit()` are clamped with a stderr warning.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_length: Option<usize>,
}

impl Default for OnnxConfig {
    fn default() -> Self {
        Self {
            model: "bge-large-en-v1.5".into(),
            max_length: None,
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
    30
}

/// Construct a live [`Embedder`] from a [`ProviderConfig`].
///
/// Reads the API key from the process environment at construction (not
/// before). If the configured provider is feature-disabled at compile
/// time, returns [`EmbedError::Config`].
///
/// # Errors
///
/// - [`EmbedError::MissingApiKey`] if the provider needs a key and the
///   env var named by `api_key_env` is unset.
/// - [`EmbedError::Config`] for unknown model strings or feature-gated
///   providers compiled out.
pub fn open(cfg: &ProviderConfig) -> Result<Box<dyn Embedder>, EmbedError> {
    match cfg {
        #[cfg(feature = "openai")]
        ProviderConfig::Openai(c) => {
            let e = crate::openai::OpenAiEmbedder::from_config(c)?;
            Ok(Box::new(e))
        }
        #[cfg(not(feature = "openai"))]
        ProviderConfig::Openai(_) => Err(EmbedError::Config(
            "this mnem-embed-providers build was compiled without the `openai` feature".into(),
        )),

        #[cfg(feature = "ollama")]
        ProviderConfig::Ollama(c) => {
            let e = crate::ollama::OllamaEmbedder::from_config(c)?;
            Ok(Box::new(e))
        }
        #[cfg(not(feature = "ollama"))]
        ProviderConfig::Ollama(_) => Err(EmbedError::Config(
            "this mnem-embed-providers build was compiled without the `ollama` feature".into(),
        )),

        ProviderConfig::Onnx(c) => open_onnx(c),
    }
}

#[cfg(any(feature = "onnx", feature = "onnx-bundled"))]
fn open_onnx(c: &OnnxConfig) -> Result<Box<dyn Embedder>, EmbedError> {
    let kind = parse_onnx_model(&c.model)?;
    let e = crate::onnx::OnnxEmbedder::with_max_length(kind, c.max_length)
        .map_err(|e| EmbedError::Config(format!("onnx init: {e}")))?;
    Ok(Box::new(e))
}

#[cfg(not(any(feature = "onnx", feature = "onnx-bundled")))]
fn open_onnx(_c: &OnnxConfig) -> Result<Box<dyn Embedder>, EmbedError> {
    Err(EmbedError::Config(
        "embed.provider = \"onnx\" but this binary was built without the `onnx` feature. \
         Rebuild with `--features onnx` (or on mnem-http: `--features embed-onnx`) or \
         switch the config to embed.provider = \"ollama\" / \"openai\"."
            .into(),
    ))
}

#[cfg(any(feature = "onnx", feature = "onnx-bundled"))]
fn parse_onnx_model(s: &str) -> Result<crate::onnx::ModelKind, EmbedError> {
    use crate::onnx::ModelKind;
    match s {
        "bge-large-en-v1.5" | "BAAI/bge-large-en-v1.5" => Ok(ModelKind::BgeLargeEnV15),
        "bge-base-en-v1.5" | "BAAI/bge-base-en-v1.5" => Ok(ModelKind::BgeBaseEnV15),
        "bge-small-en-v1.5" | "BAAI/bge-small-en-v1.5" => Ok(ModelKind::BgeSmallEnV15),
        "all-MiniLM-L6-v2"
        | "all-minilm-l6-v2"
        | "all-minilm"
        | "sentence-transformers/all-MiniLM-L6-v2" => Ok(ModelKind::AllMiniLmL6V2),
        other => Err(EmbedError::Config(format!(
            "unknown onnx embed model `{other}`; known: \
             bge-large-en-v1.5, bge-base-en-v1.5, bge-small-en-v1.5, \
             all-MiniLM-L6-v2"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn openai_config_toml_round_trip() {
        let cfg = ProviderConfig::Openai(OpenAiConfig {
            model: "text-embedding-3-small".into(),
            ..Default::default()
        });
        let s = toml::to_string(&cfg).unwrap();
        assert!(s.contains("provider = \"openai\""));
        assert!(s.contains("text-embedding-3-small"));
        let back: ProviderConfig = toml::from_str(&s).unwrap();
        assert_eq!(cfg, back);
    }

    #[test]
    fn ollama_config_toml_round_trip() {
        let cfg = ProviderConfig::Ollama(OllamaConfig::default());
        let s = toml::to_string(&cfg).unwrap();
        let back: ProviderConfig = toml::from_str(&s).unwrap();
        assert_eq!(cfg, back);
    }

    #[test]
    fn onnx_config_toml_round_trip() {
        let cfg = ProviderConfig::Onnx(OnnxConfig::default());
        let s = toml::to_string(&cfg).unwrap();
        assert!(
            s.contains("provider = \"onnx\""),
            "onnx tag must serialise as provider = \"onnx\"; got:\n{s}"
        );
        assert!(s.contains("bge-large-en-v1.5"));
        let back: ProviderConfig = toml::from_str(&s).unwrap();
        assert_eq!(cfg, back);
    }

    #[test]
    fn onnx_config_default_omits_max_length() {
        let cfg = ProviderConfig::Onnx(OnnxConfig::default());
        let s = toml::to_string(&cfg).unwrap();
        assert!(
            !s.contains("max_length"),
            "default config should not emit max_length; got:\n{s}"
        );
    }

    #[cfg(not(any(feature = "onnx", feature = "onnx-bundled")))]
    #[test]
    fn open_onnx_without_feature_returns_actionable_error() {
        let cfg = ProviderConfig::Onnx(OnnxConfig::default());
        // `Box<dyn Embedder>` lacks `Debug`, so `unwrap_err()` would
        // fail to compile; match the `Err` branch by hand instead.
        let err = match open(&cfg) {
            Ok(_) => panic!("open() should fail when the `onnx` feature is off"),
            Err(e) => e,
        };
        let msg = format!("{err}");
        assert!(
            msg.contains("--features onnx") || msg.contains("embed-onnx"),
            "error should suggest the rebuild flag; got: {msg}"
        );
    }
}
