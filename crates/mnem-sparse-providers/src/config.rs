// SPLADE, BGE-M3, sidecar-host proper nouns are well-known external
// identifiers.
#![allow(clippy::doc_markdown)]

//! [`ProviderConfig`] + [`open`] factory. Mirrors the pattern in
//! `mnem-rerank-providers` and `mnem-llm-providers`.

use std::sync::Arc;

use mnem_core::sparse::{SparseEncoder, SparseError};
use serde::{Deserialize, Serialize};

/// Tagged enum over the shipped backends. TOML shape for each backend:
///
/// Sidecar (Python HTTP server, zero native deps in mnem):
/// ```toml
/// [sparse]
/// provider  = "sidecar"
/// base_url  = "http://localhost:8791"
/// model     = "opensearch-doc-v3-distill"
/// vocab_id  = "bert-base-uncased@30522"
/// ```
///
/// Native ONNX (in-process, requires the `onnx` cargo feature):
/// ```toml
/// [sparse]
/// provider   = "onnx"
/// model      = "opensearch-doc-v3-distill"   # or "opensearch-bi-v2-distill"
/// # max_length = 512                         # optional; default = model ceiling
/// ```
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase", tag = "provider")]
pub enum ProviderConfig {
    /// HTTP sidecar running the reference Python SPLADE/BGE-M3
    /// implementation. See `benchmarks/adapters/splade-sidecar/` in
    /// for a Docker image.
    Sidecar(SidecarConfig),
    /// Native in-process ONNX encoder. Requires building with the
    /// `onnx` feature; otherwise [`open`] returns a
    /// [`SparseError::Config`] explaining the mismatch so operators
    /// can either rebuild or switch back to `sidecar`.
    Onnx(OnnxConfig),
}

/// Sidecar HTTP config.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct SidecarConfig {
    /// Base URL, e.g. `"http://localhost:8791"`. The adapter
    /// POSTs to `<base_url>/encode` with `{text, model}` and
    /// expects `{indices: [u32], values: [f32], vocab_id: str}`.
    pub base_url: String,
    /// Bare model id the sidecar exposes, e.g.
    /// `"opensearch-doc-v3-distill"`. Final fq id is
    /// `"sidecar:<model>"`.
    pub model: String,
    /// Vocabulary id the sidecar's model uses. Must match the
    /// `vocab_id` stamped on stored `SparseEmbed`s for retrieval to
    /// work (mnem-core::index::SparseInvertedIndex enforces this).
    pub vocab_id: String,
    /// Per-request timeout in seconds. Default 30.
    #[serde(default = "default_timeout")]
    pub timeout_secs: u64,
}

impl Default for SidecarConfig {
    fn default() -> Self {
        Self {
            base_url: "http://localhost:8791".into(),
            model: "opensearch-doc-v3-distill".into(),
            vocab_id: "bert-base-uncased@30522".into(),
            timeout_secs: default_timeout(),
        }
    }
}

const fn default_timeout() -> u64 {
    30
}

/// Native in-process ONNX encoder config. Unlike [`SidecarConfig`]
/// there is no network URL or vocab_id: the `model` string resolves
/// directly to a compiled-in `onnx::ModelKind` variant (only available
/// when the `onnx` cargo feature is enabled) and the
/// encoder stamps the canonical `vocab_id` itself.
///
/// Kept visible even when the `onnx` feature is off, so that a
/// deserialised `[sparse] provider = "onnx"` block round-trips
/// through TOML and [`open`] can emit an actionable "rebuild with
/// `--features onnx`" error at construction time.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct OnnxConfig {
    /// Model shortname. Known values map to `onnx::ModelKind` variants
    /// (only available when the `onnx` cargo feature is enabled):
    ///   - `"opensearch-doc-v3-distill"` (default; asymmetric, IDF query)
    ///   - `"opensearch-bi-v2-distill"`  (symmetric, both sides run the net)
    pub model: String,
    /// Optional tokenizer `max_length` override. `None` defers to
    /// the model's `default_max_length()` (DistilBERT = 512). Values
    /// above the model's `positional_limit()` are clamped with a
    /// stderr warning.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_length: Option<usize>,
}

impl Default for OnnxConfig {
    fn default() -> Self {
        Self {
            model: "opensearch-doc-v3-distill".into(),
            max_length: None,
        }
    }
}

/// Construct a live [`SparseEncoder`] from a [`ProviderConfig`].
///
/// # Errors
///
/// - [`SparseError::Config`] if the sidecar URL is malformed.
/// - [`SparseError::Config`] with an actionable remediation string
///   when `provider = "onnx"` is used in a build without the `onnx`
///   cargo feature.
/// - [`SparseError::Config`] on an unknown onnx `model` string.
pub fn open(cfg: &ProviderConfig) -> Result<Arc<dyn SparseEncoder>, SparseError> {
    match cfg {
        ProviderConfig::Sidecar(c) => {
            let enc = crate::sidecar::SidecarSparseEncoder::from_config(c)?;
            Ok(Arc::new(enc))
        }
        ProviderConfig::Onnx(c) => open_onnx(c),
    }
}

#[cfg(feature = "onnx")]
fn open_onnx(c: &OnnxConfig) -> Result<Arc<dyn SparseEncoder>, SparseError> {
    let kind = parse_onnx_model(&c.model)?;
    let enc = crate::onnx::OnnxSparseEncoder::with_max_length(kind, c.max_length)?;
    Ok(Arc::new(enc))
}

#[cfg(not(feature = "onnx"))]
fn open_onnx(_c: &OnnxConfig) -> Result<Arc<dyn SparseEncoder>, SparseError> {
    Err(SparseError::Config(
        "sparse.provider = \"onnx\" but this binary was built without the `onnx` feature. \
         Rebuild with `--features onnx` or set sparse.provider = \"sidecar\"."
            .into(),
    ))
}

#[cfg(feature = "onnx")]
fn parse_onnx_model(s: &str) -> Result<crate::onnx::ModelKind, SparseError> {
    use crate::onnx::ModelKind;
    match s {
        "opensearch-doc-v3-distill" => Ok(ModelKind::OpensearchDocV3Distill),
        "opensearch-bi-v2-distill" => Ok(ModelKind::OpensearchBiV2Distill),
        other => Err(SparseError::Config(format!(
            "unknown onnx sparse model `{other}`; known: \
             opensearch-doc-v3-distill, opensearch-bi-v2-distill"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sidecar_config_toml_round_trip() {
        let cfg = ProviderConfig::Sidecar(SidecarConfig::default());
        let s = toml::to_string(&cfg).unwrap();
        assert!(s.contains("provider = \"sidecar\""));
        let back: ProviderConfig = toml::from_str(&s).unwrap();
        assert_eq!(cfg, back);
    }

    #[test]
    fn sidecar_default_has_sane_values() {
        let c = SidecarConfig::default();
        assert!(c.base_url.starts_with("http"));
        assert!(!c.model.is_empty());
        assert!(!c.vocab_id.is_empty());
        assert_eq!(c.timeout_secs, 30);
    }

    #[test]
    fn onnx_config_toml_round_trip() {
        let cfg = ProviderConfig::Onnx(OnnxConfig::default());
        let s = toml::to_string(&cfg).unwrap();
        assert!(
            s.contains("provider = \"onnx\""),
            "onnx tag must serialise as provider = \"onnx\"; got:\n{s}"
        );
        let back: ProviderConfig = toml::from_str(&s).unwrap();
        assert_eq!(cfg, back);
    }

    #[test]
    fn onnx_config_default_skips_max_length() {
        let cfg = ProviderConfig::Onnx(OnnxConfig::default());
        let s = toml::to_string(&cfg).unwrap();
        assert!(
            !s.contains("max_length"),
            "default OnnxConfig should not emit max_length (let the encoder pick). Got:\n{s}"
        );
    }

    #[test]
    fn onnx_config_max_length_round_trip() {
        let cfg = ProviderConfig::Onnx(OnnxConfig {
            model: "opensearch-bi-v2-distill".into(),
            max_length: Some(256),
        });
        let s = toml::to_string(&cfg).unwrap();
        assert!(s.contains("max_length = 256"));
        let back: ProviderConfig = toml::from_str(&s).unwrap();
        assert_eq!(cfg, back);
    }

    #[cfg(not(feature = "onnx"))]
    #[test]
    fn open_onnx_without_feature_returns_actionable_error() {
        let cfg = ProviderConfig::Onnx(OnnxConfig::default());
        let err = open(&cfg).unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("--features onnx"),
            "err should point at the feature rebuild; got: {msg}"
        );
    }

    #[cfg(feature = "onnx")]
    #[test]
    fn parse_onnx_model_rejects_unknown() {
        let err = parse_onnx_model("made-up-model").unwrap_err();
        assert!(format!("{err}").contains("unknown onnx sparse model"));
    }
}
