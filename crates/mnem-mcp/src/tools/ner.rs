//! Shared NER-provider resolution for MCP tools.
//!
//! Mirrors `tools::embed` for the NER layer. Called by
//! `handlers::ingest` and `handlers::global_ingest` to resolve the NER
//! config before building an `IngestConfig`.
//!
//! ## Precedence
//!
//!   1. `MNEM_NER_PROVIDER` env var (`"rule"` or `"none"`).
//!   2. The `[ner]` section in `<repo_path>/config.toml`.
//!   3. [`mnem_ingest::NerConfig::Rule`] (always-available default).
//!
//! The user-global `~/.mnem/config.toml` tier is intentionally absent
//! here (same as the MCP embed resolver): mnem-mcp's design point is
//! per-repo isolation.

use mnem_ingest::NerConfig;
use serde::Deserialize;

/// Minimal schema for parsing just the `[ner]` table out of
/// `<repo>/config.toml`.
#[derive(Debug, Deserialize)]
struct NerOnlyConfig {
    ner: Option<NerConfig>,
}

/// Resolve an NER config for an MCP handler.
///
/// Never returns `None`, `NerConfig::Rule` is the always-available
/// fallback. Boxed as `NerConfig` rather than `Option<NerConfig>` so
/// callers can assign directly to `IngestConfig::ner`.
pub(crate) fn resolve_ner_cfg(repo_path: &std::path::Path) -> NerConfig {
    if let Ok(p) = std::env::var("MNEM_NER_PROVIDER") {
        return match p.to_ascii_lowercase().as_str() {
            "none" => NerConfig::None,
            _ => NerConfig::Rule,
        };
    }
    let cfg_path = repo_path.join("config.toml");
    if let Ok(s) = std::fs::read_to_string(&cfg_path)
        && let Ok(parsed) = toml::from_str::<NerOnlyConfig>(&s)
        && let Some(ner) = parsed.ner
    {
        return ner;
    }
    NerConfig::Rule
}
