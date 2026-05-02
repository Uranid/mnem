//! Shared embed-provider resolution for MCP tools.
//!
//! Path A audit fix (2026-04-26): hoisted from
//! `handlers/community_summarize.rs` so the same precedence chain
//! powers both `mnem_community_summarize` and `mnem_retrieve` (when
//! `text` is supplied without an explicit `vector`). Adds a tier-3
//! bundled-MiniLM fallback that fires only when the
//! `bundled-embedder` cargo feature is compiled in - otherwise the
//! function returns `None` and callers fall through to their
//! existing no-embedder paths.
//!
//! ## Precedence
//!
//!   1. `MNEM_EMBED_PROVIDER` + `MNEM_EMBED_MODEL` (+ optional
//!      `MNEM_EMBED_API_KEY_ENV`, `MNEM_EMBED_BASE_URL`,
//!      `MNEM_EMBED_DIM`) env vars.
//!   2. The `[embed]` section in `<repo_path>/config.toml`.
//!   3. Path A bundled-embedder default (`OnnxConfig` for
//!      `all-MiniLM-L6-v2`) when this binary was built with
//!      `--features bundled-embedder`. Off otherwise.
//!
//! Mirrors mnem-cli's `config::resolve_embedder` (without the
//! user-global `~/.mnem/config.toml` tier - mnem mcp's design point
//! is per-repo isolation, and the existing community_summarize
//! handler did not consult global config either).

#![cfg(feature = "summarize")]

use mnem_embed_providers::{
    OllamaConfig, OnnxConfig, OpenAiConfig, ProviderConfig as EmbedProviderConfig,
};
use serde::Deserialize;

/// Default model picked when the `bundled-embedder` cargo feature
/// is compiled in and no env / config tier yields a provider.
/// Choice rationale: same as `mnem-cli` (`all-MiniLM-L6-v2`,
/// 22M params, 384-dim, 92MB on disk, Apache-2.0). Matches
/// ChromaDB's `DefaultEmbeddingFunction` byte-for-byte.
#[cfg_attr(not(feature = "bundled-embedder"), allow(dead_code))]
pub(crate) const BUNDLED_EMBEDDER_DEFAULT_MODEL: &str = "all-MiniLM-L6-v2";

/// Minimal schema for parsing just the `[embed]` table out of
/// `<repo>/config.toml`. Decoupled from `mnem-cli`'s full `Config`
/// so the MCP binary stays free of the retrieve / rerank / LLM /
/// user surfaces it does not need.
#[derive(Debug, Deserialize)]
struct EmbedOnlyConfig {
    embed: Option<EmbedProviderConfig>,
}

/// Resolve an embed-provider config for an MCP handler.
///
/// Returns `None` if no tier yields a provider; callers fall through
/// to whatever no-embedder behaviour they implement (silent skip in
/// retrieve, tool-level error in community_summarize).
pub(crate) fn resolve_embed_cfg(repo_path: &std::path::Path) -> Option<EmbedProviderConfig> {
    if let Ok(provider) = std::env::var("MNEM_EMBED_PROVIDER") {
        let model = std::env::var("MNEM_EMBED_MODEL").ok()?;
        return match provider.as_str() {
            "openai" => Some(EmbedProviderConfig::Openai(OpenAiConfig {
                model,
                api_key_env: std::env::var("MNEM_EMBED_API_KEY_ENV")
                    .unwrap_or_else(|_| "OPENAI_API_KEY".into()),
                base_url: std::env::var("MNEM_EMBED_BASE_URL")
                    .unwrap_or_else(|_| "https://api.openai.com".into()),
                timeout_secs: 30,
                dim_override: std::env::var("MNEM_EMBED_DIM")
                    .ok()
                    .and_then(|s| s.parse().ok()),
            })),
            "ollama" => Some(EmbedProviderConfig::Ollama(OllamaConfig {
                model,
                base_url: std::env::var("MNEM_EMBED_BASE_URL")
                    .unwrap_or_else(|_| "http://localhost:11434".into()),
                timeout_secs: 30,
            })),
            "onnx" => Some(EmbedProviderConfig::Onnx(OnnxConfig {
                model,
                max_length: None,
            })),
            _ => None,
        };
    }
    let cfg_path = repo_path.join("config.toml");
    if let Ok(bytes) = std::fs::read_to_string(&cfg_path)
        && let Ok(parsed) = toml::from_str::<EmbedOnlyConfig>(&bytes)
        && let Some(emb) = parsed.embed
    {
        return Some(emb);
    }
    bundled_embedder_default()
}

/// Tier-3 helper: returns `Some(OnnxConfig{all-MiniLM-L6-v2})` when
/// compiled with `--features bundled-embedder`; `None` otherwise.
/// Factored out so the test runner can pin the boundary to a stable
/// hook rather than re-parsing `cfg!` at multiple call sites.
#[must_use]
pub(crate) fn bundled_embedder_default() -> Option<EmbedProviderConfig> {
    #[cfg(feature = "bundled-embedder")]
    {
        Some(EmbedProviderConfig::Onnx(OnnxConfig {
            model: BUNDLED_EMBEDDER_DEFAULT_MODEL.to_string(),
            ..Default::default()
        }))
    }
    #[cfg(not(feature = "bundled-embedder"))]
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(feature = "bundled-embedder")]
    fn bundled_default_returns_minilm_when_feature_on() {
        match bundled_embedder_default() {
            Some(EmbedProviderConfig::Onnx(c)) => {
                assert_eq!(c.model, BUNDLED_EMBEDDER_DEFAULT_MODEL);
                assert_eq!(c.model, "all-MiniLM-L6-v2");
            }
            other => panic!("expected Onnx(MiniLM); got {other:?}"),
        }
    }

    #[test]
    #[cfg(not(feature = "bundled-embedder"))]
    fn bundled_default_returns_none_when_feature_off() {
        assert!(bundled_embedder_default().is_none());
    }

    #[test]
    #[cfg(feature = "bundled-embedder")]
    fn resolve_falls_back_to_bundled_when_no_env_no_config() {
        // No env vars, empty repo dir → tier 3 (bundled) fires.
        // Skip if the dev shell has MNEM_EMBED_PROVIDER set (env tier
        // would otherwise win and the test reduces to a tautology).
        if std::env::var("MNEM_EMBED_PROVIDER").is_ok() {
            return;
        }
        let td = tempfile::tempdir().expect("tempdir");
        let resolved = resolve_embed_cfg(td.path()).expect("tier 3 should yield a provider");
        match resolved {
            EmbedProviderConfig::Onnx(c) => assert_eq!(c.model, "all-MiniLM-L6-v2"),
            other => panic!("expected bundled Onnx; got {other:?}"),
        }
    }

    // (The per-repo `[embed]` table format is exercised end-to-end via
    // `mnem-cli`'s round-trip tests over `Config::save` / `Config::load`.
    // Reproducing the same TOML by hand here would couple the test to
    // serde's enum-tagging convention for `ProviderConfig`, which is
    // `mnem-embed-providers`-internal. Trust the dedicated tests for
    // the format; this module's contribution is the precedence chain
    // and the bundled-tier fallback, both covered above.)
}
