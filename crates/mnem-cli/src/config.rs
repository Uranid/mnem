//! `.mnem/config.toml` read / write.
//!
//! Kept deliberately thin: a handful of dotted keys, no schema
//! validation beyond TOML parsing. Git-shaped UX:
//!
//! ```text
//! mnem config user.name        # read
//! mnem config user.name Alice  # write
//! ```

use std::path::Path;

use anyhow::{Context, Result};
use mnem_embed_providers::{OllamaConfig, OnnxConfig, OpenAiConfig, ProviderConfig};
use mnem_llm_providers::{OllamaLlmConfig, OpenAiLlmConfig, ProviderConfig as LlmProviderConfig};
use mnem_ner_providers::NerConfig;
use mnem_rerank_providers::{
    CohereConfig, JinaConfig, ProviderConfig as RerankProviderConfig, VoyageConfig,
};
use serde::{Deserialize, Serialize};

pub(crate) const CONFIG_FILE: &str = "config.toml";

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub(crate) struct Config {
    #[serde(default)]
    pub user: UserConfig,
    /// Optional embedding-provider configuration. When present,
    /// `mnem add node` auto-embeds new nodes and `mnem retrieve --text`
    /// auto-fuses semantic search into the result.
    ///
    /// API keys live in environment variables, never in this file.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub embed: Option<ProviderConfig>,
    /// Optional cross-encoder reranker configuration (tier 3 of the
    /// compositional retrieval hierarchy). When present,
    /// `mnem retrieve` re-scores the top-K of the fused list through
    /// the configured provider. API keys live in env vars, never here.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rerank: Option<RerankProviderConfig>,
    /// Optional LLM (text-generation) configuration for `HyDE` and
    /// multi-query retrieval. API keys live in env vars, never here.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub llm: Option<LlmProviderConfig>,
    /// Optional NER provider configuration. Defaults to rule-based
    /// heuristic when absent. Set to `NerConfig::None` via
    /// `[ner]\nprovider = "none"` to suppress entity extraction.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ner: Option<NerConfig>,
    /// Persistent defaults for `mnem retrieve`. Every knob here is
    /// also exposed as a CLI flag; the flag wins when both are set.
    /// Git-shaped "set once, never pass again" pattern: e.g.
    /// `mnem config set retrieve.limit 20` and subsequent
    /// `mnem retrieve "query"` calls default to `--limit 20`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub retrieve: Option<RetrieveDefaults>,
}

/// Per-repo defaults for `mnem retrieve`. All fields are optional;
/// unset fields defer to the retriever's built-in defaults. CLI flags
/// always win over these.
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub(crate) struct RetrieveDefaults {
    /// Default for `--limit`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub limit: Option<usize>,
    /// Default for `--budget`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub budget: Option<u32>,
    /// Default for `--vector-cap`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub vector_cap: Option<usize>,
    /// Default for `--graph-expand`. When set, graph-expand runs on
    /// every retrieve without needing the flag.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub graph_expand: Option<usize>,
    /// Default for `--graph-decay`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub graph_decay: Option<f32>,
    /// Default for `--graph-depth`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub graph_depth: Option<usize>,
    /// Default for `--rerank-top-k`. Only meaningful when
    /// `[rerank]` is configured or `--rerank` is passed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rerank_top_k: Option<usize>,
    /// Default for `--hyde-max-tokens`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hyde_max_tokens: Option<u32>,
    /// Default for `--hyde-temperature`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hyde_temperature: Option<f32>,
}

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub(crate) struct UserConfig {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub email: Option<String>,
    /// Hex-encoded Ed25519 secret for signing. Optional; unsigned
    /// commits still work.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub key: Option<String>,
    /// Agent-identifier string. Used as `Commit.author` for CLI
    /// commits when `name` isn't set.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub agent_id: Option<String>,
}

pub(crate) fn path_of(data_dir: &Path) -> std::path::PathBuf {
    data_dir.join(CONFIG_FILE)
}

/// audit-2026-04-25 C4-4: path to the user-global config file at
/// `~/.mnem/config.toml`. Returns `None` when `dirs::home_dir()`
/// can't resolve a home directory (rare; CI containers without
/// `$HOME`).
pub(crate) fn global_path() -> Option<std::path::PathBuf> {
    dirs::home_dir().map(|h| h.join(".mnem").join(CONFIG_FILE))
}

/// audit-2026-04-25 C4-4: load the user-global `~/.mnem/config.toml`
/// if it exists. Returns `Ok(Config::default())` on a missing file
/// or unresolvable home dir (inheritance is opt-in: missing global
/// means "no fallback"). A malformed global config surfaces an
/// error so the user can see the parse problem; we don't silently
/// swallow it.
///
/// `MNEM_DISABLE_GLOBAL_CONFIG=1` short-circuits to an empty
/// `Config::default()` -- used by integration tests to keep
/// behaviour deterministic across dev workstations that may have
/// real `~/.mnem/config.toml` files.
pub(crate) fn load_global() -> Result<Config> {
    if std::env::var("MNEM_DISABLE_GLOBAL_CONFIG")
        .ok()
        .is_some_and(|v| !v.is_empty() && v != "0")
    {
        return Ok(Config::default());
    }
    let Some(p) = global_path() else {
        return Ok(Config::default());
    };
    if !p.exists() {
        return Ok(Config::default());
    }
    let s = std::fs::read_to_string(&p).with_context(|| format!("reading {}", p.display()))?;
    toml::from_str(&s).with_context(|| format!("parsing {}", p.display()))
}

pub(crate) fn load(data_dir: &Path) -> Result<Config> {
    let p = path_of(data_dir);
    if !p.exists() {
        return Ok(Config::default());
    }
    let s = std::fs::read_to_string(&p).with_context(|| format!("reading {}", p.display()))?;
    toml::from_str(&s).with_context(|| format!("parsing {}", p.display()))
}

pub(crate) fn save(data_dir: &Path, cfg: &Config) -> Result<()> {
    let p = path_of(data_dir);
    let s = toml::to_string_pretty(cfg).context("serialising config")?;
    std::fs::write(&p, s).with_context(|| format!("writing {}", p.display()))
}

/// Write `cfg` directly to `path` (used by `mnem config --global`).
/// Creates parent directories if they don't exist yet, so a fresh
/// `~/.mnem/` directory is set up automatically on first write.
pub(crate) fn save_to_path(path: &std::path::Path, cfg: &Config) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("creating directory {}", parent.display()))?;
    }
    let s = toml::to_string_pretty(cfg).context("serialising config")?;
    std::fs::write(path, s).with_context(|| format!("writing {}", path.display()))
}

/// Known dotted keys, for `mnem config list` and tab-completion hints.
pub(crate) const KNOWN_KEYS: &[&str] = &[
    "user.name",
    "user.email",
    "user.key",
    "user.agent_id",
    "embed.provider",
    "embed.model",
    "embed.api_key_env",
    "embed.base_url",
    "rerank.provider",
    "rerank.model",
    "rerank.api_key_env",
    "rerank.base_url",
    "llm.provider",
    "llm.model",
    "llm.api_key_env",
    "llm.base_url",
    "ner.provider",
    "retrieve.limit",
    "retrieve.budget",
    "retrieve.vector_cap",
    "retrieve.graph_expand",
    "retrieve.graph_decay",
    "retrieve.graph_depth",
    "retrieve.rerank_top_k",
    "retrieve.hyde_max_tokens",
    "retrieve.hyde_temperature",
];

pub(crate) fn get_dotted(cfg: &Config, key: &str) -> Option<String> {
    match key {
        "user.name" => cfg.user.name.clone(),
        "user.email" => cfg.user.email.clone(),
        "user.key" => cfg.user.key.clone(),
        "user.agent_id" => cfg.user.agent_id.clone(),
        "embed.provider" => cfg.embed.as_ref().map(|e| match e {
            ProviderConfig::Openai(_) => "openai".into(),
            ProviderConfig::Ollama(_) => "ollama".into(),
            ProviderConfig::Onnx(_) => "onnx".into(),
        }),
        "embed.model" => cfg.embed.as_ref().map(|e| match e {
            ProviderConfig::Openai(c) => c.model.clone(),
            ProviderConfig::Ollama(c) => c.model.clone(),
            ProviderConfig::Onnx(c) => c.model.clone(),
        }),
        "embed.api_key_env" => cfg.embed.as_ref().and_then(|e| match e {
            ProviderConfig::Openai(c) => Some(c.api_key_env.clone()),
            ProviderConfig::Ollama(_) | ProviderConfig::Onnx(_) => None,
        }),
        "embed.base_url" => cfg.embed.as_ref().and_then(|e| match e {
            ProviderConfig::Openai(c) => Some(c.base_url.clone()),
            ProviderConfig::Ollama(c) => Some(c.base_url.clone()),
            // Onnx is in-process; no base URL.
            ProviderConfig::Onnx(_) => None,
        }),
        "rerank.provider" => cfg.rerank.as_ref().map(|r| match r {
            RerankProviderConfig::Cohere(_) => "cohere".into(),
            RerankProviderConfig::Voyage(_) => "voyage".into(),
            RerankProviderConfig::Jina(_) => "jina".into(),
        }),
        "rerank.model" => cfg.rerank.as_ref().map(|r| match r {
            RerankProviderConfig::Cohere(c) => c.model.clone(),
            RerankProviderConfig::Voyage(c) => c.model.clone(),
            RerankProviderConfig::Jina(c) => c.model.clone(),
        }),
        "rerank.api_key_env" => cfg.rerank.as_ref().map(|r| match r {
            RerankProviderConfig::Cohere(c) => c.api_key_env.clone(),
            RerankProviderConfig::Voyage(c) => c.api_key_env.clone(),
            RerankProviderConfig::Jina(c) => c.api_key_env.clone(),
        }),
        "rerank.base_url" => cfg.rerank.as_ref().map(|r| match r {
            RerankProviderConfig::Cohere(c) => c.base_url.clone(),
            RerankProviderConfig::Voyage(c) => c.base_url.clone(),
            RerankProviderConfig::Jina(c) => c.base_url.clone(),
        }),
        "llm.provider" => cfg.llm.as_ref().map(|l| match l {
            LlmProviderConfig::Openai(_) => "openai".into(),
            LlmProviderConfig::Ollama(_) => "ollama".into(),
        }),
        "llm.model" => cfg.llm.as_ref().map(|l| match l {
            LlmProviderConfig::Openai(c) => c.model.clone(),
            LlmProviderConfig::Ollama(c) => c.model.clone(),
        }),
        "llm.api_key_env" => cfg.llm.as_ref().and_then(|l| match l {
            LlmProviderConfig::Openai(c) => Some(c.api_key_env.clone()),
            LlmProviderConfig::Ollama(_) => None,
        }),
        "llm.base_url" => cfg.llm.as_ref().map(|l| match l {
            LlmProviderConfig::Openai(c) => c.base_url.clone(),
            LlmProviderConfig::Ollama(c) => c.base_url.clone(),
        }),
        "ner.provider" => cfg.ner.as_ref().map(|n| match n {
            NerConfig::Rule => "rule".into(),
            NerConfig::None => "none".into(),
        }),
        "retrieve.limit" => cfg
            .retrieve
            .as_ref()
            .and_then(|r| r.limit.map(|n| n.to_string())),
        "retrieve.budget" => cfg
            .retrieve
            .as_ref()
            .and_then(|r| r.budget.map(|n| n.to_string())),
        "retrieve.vector_cap" => cfg
            .retrieve
            .as_ref()
            .and_then(|r| r.vector_cap.map(|n| n.to_string())),
        "retrieve.graph_expand" => cfg
            .retrieve
            .as_ref()
            .and_then(|r| r.graph_expand.map(|n| n.to_string())),
        "retrieve.graph_decay" => cfg
            .retrieve
            .as_ref()
            .and_then(|r| r.graph_decay.map(|n| n.to_string())),
        "retrieve.graph_depth" => cfg
            .retrieve
            .as_ref()
            .and_then(|r| r.graph_depth.map(|n| n.to_string())),
        "retrieve.rerank_top_k" => cfg
            .retrieve
            .as_ref()
            .and_then(|r| r.rerank_top_k.map(|n| n.to_string())),
        "retrieve.hyde_max_tokens" => cfg
            .retrieve
            .as_ref()
            .and_then(|r| r.hyde_max_tokens.map(|n| n.to_string())),
        "retrieve.hyde_temperature" => cfg
            .retrieve
            .as_ref()
            .and_then(|r| r.hyde_temperature.map(|n| n.to_string())),
        _ => None,
    }
}

pub(crate) fn set_dotted(cfg: &mut Config, key: &str, value: Option<String>) -> Result<()> {
    // Guardrail: if the caller passes an API key in a value (e.g. via
    // `mnem config set embed.api_key sk-...`), bail out. Keys live in
    // env vars, period. Applies to both embed.* and rerank.* sections.
    if key == "embed.api_key"
        || key == "rerank.api_key"
        || value.as_deref().map(|v| {
            v.starts_with("sk-") && (key.starts_with("embed.") || key.starts_with("rerank."))
        }) == Some(true)
    {
        anyhow::bail!(
            "API keys must not be stored in config.toml. Set an env var, \
             then point mnem at it: `mnem config set embed.api_key_env OPENAI_API_KEY`.\n\
             hint: see docs/RUNBOOK.md#1-mnem http-returns-500s for the embed-provider \
             remediation checklist when keys are wrong or unreachable."
        );
    }

    match key {
        "user.name" => cfg.user.name = value,
        "user.email" => cfg.user.email = value,
        "user.key" => cfg.user.key = value,
        "user.agent_id" => cfg.user.agent_id = value,

        "embed.provider" => set_embed_provider(cfg, value.as_deref())?,
        "embed.model" => set_embed_model(cfg, value.as_deref())?,
        "embed.api_key_env" => set_embed_api_key_env(cfg, value.as_deref())?,
        "embed.base_url" => set_embed_base_url(cfg, value.as_deref())?,

        "rerank.provider" => set_rerank_provider(cfg, value.as_deref())?,
        "rerank.model" => set_rerank_model(cfg, value.as_deref())?,
        "rerank.api_key_env" => set_rerank_api_key_env(cfg, value.as_deref())?,
        "rerank.base_url" => set_rerank_base_url(cfg, value.as_deref())?,

        "llm.provider" => set_llm_provider(cfg, value.as_deref())?,
        "llm.model" => set_llm_model(cfg, value.as_deref())?,
        "llm.api_key_env" => set_llm_api_key_env(cfg, value.as_deref())?,
        "llm.base_url" => set_llm_base_url(cfg, value.as_deref())?,

        "ner.provider" => match value.as_deref() {
            Some("rule") | None => cfg.ner = Some(NerConfig::Rule),
            Some("none") => cfg.ner = Some(NerConfig::None),
            Some(other) => anyhow::bail!("unknown ner.provider `{other}` (expected rule|none)"),
        },

        "retrieve.limit" => set_retrieve_usize(cfg, value.as_deref(), |r, n| r.limit = n)?,
        "retrieve.budget" => set_retrieve_u32(cfg, value.as_deref(), |r, n| r.budget = n)?,
        "retrieve.vector_cap" => {
            set_retrieve_usize(cfg, value.as_deref(), |r, n| r.vector_cap = n)?;
        }
        "retrieve.graph_expand" => {
            set_retrieve_usize(cfg, value.as_deref(), |r, n| r.graph_expand = n)?;
        }
        "retrieve.graph_decay" => {
            set_retrieve_f32(cfg, value.as_deref(), |r, n| r.graph_decay = n)?;
        }
        "retrieve.graph_depth" => {
            set_retrieve_usize(cfg, value.as_deref(), |r, n| r.graph_depth = n)?;
        }
        "retrieve.rerank_top_k" => {
            set_retrieve_usize(cfg, value.as_deref(), |r, n| r.rerank_top_k = n)?;
        }
        "retrieve.hyde_max_tokens" => {
            set_retrieve_u32(cfg, value.as_deref(), |r, n| r.hyde_max_tokens = n)?;
        }
        "retrieve.hyde_temperature" => {
            set_retrieve_f32(cfg, value.as_deref(), |r, n| r.hyde_temperature = n)?;
        }

        other => anyhow::bail!("unknown config key: {other}"),
    }
    // Drop the whole [retrieve] table once every field is unset, so
    // `config.toml` stays clean after a round of `mnem config unset`.
    if let Some(r) = &cfg.retrieve
        && r.limit.is_none()
        && r.budget.is_none()
        && r.vector_cap.is_none()
        && r.graph_expand.is_none()
        && r.graph_decay.is_none()
        && r.graph_depth.is_none()
        && r.rerank_top_k.is_none()
        && r.hyde_max_tokens.is_none()
        && r.hyde_temperature.is_none()
    {
        cfg.retrieve = None;
    }
    Ok(())
}

fn set_retrieve_usize(
    cfg: &mut Config,
    value: Option<&str>,
    apply: impl FnOnce(&mut RetrieveDefaults, Option<usize>),
) -> Result<()> {
    let parsed = match value {
        None => None,
        Some(v) => Some(
            v.parse::<usize>()
                .with_context(|| format!("expected an unsigned integer, got `{v}`"))?,
        ),
    };
    let r = cfg.retrieve.get_or_insert_with(RetrieveDefaults::default);
    apply(r, parsed);
    Ok(())
}

fn set_retrieve_u32(
    cfg: &mut Config,
    value: Option<&str>,
    apply: impl FnOnce(&mut RetrieveDefaults, Option<u32>),
) -> Result<()> {
    let parsed = match value {
        None => None,
        Some(v) => Some(
            v.parse::<u32>()
                .with_context(|| format!("expected a u32 integer, got `{v}`"))?,
        ),
    };
    let r = cfg.retrieve.get_or_insert_with(RetrieveDefaults::default);
    apply(r, parsed);
    Ok(())
}

fn set_retrieve_f32(
    cfg: &mut Config,
    value: Option<&str>,
    apply: impl FnOnce(&mut RetrieveDefaults, Option<f32>),
) -> Result<()> {
    let parsed = match value {
        None => None,
        Some(v) => Some(
            v.parse::<f32>()
                .with_context(|| format!("expected a float, got `{v}`"))?,
        ),
    };
    let r = cfg.retrieve.get_or_insert_with(RetrieveDefaults::default);
    apply(r, parsed);
    Ok(())
}

/// Switch the provider, preserving the model when it makes sense.
fn set_embed_provider(cfg: &mut Config, value: Option<&str>) -> Result<()> {
    match value {
        None => {
            cfg.embed = None;
        }
        Some("openai") => {
            let model = cfg
                .embed
                .as_ref()
                .and_then(|e| match e {
                    ProviderConfig::Openai(c) => Some(c.model.clone()),
                    ProviderConfig::Ollama(_) | ProviderConfig::Onnx(_) => None,
                })
                .unwrap_or_else(|| "text-embedding-3-small".into());
            cfg.embed = Some(ProviderConfig::Openai(OpenAiConfig {
                model,
                ..Default::default()
            }));
        }
        Some("ollama") => {
            let model = cfg
                .embed
                .as_ref()
                .and_then(|e| match e {
                    ProviderConfig::Ollama(c) => Some(c.model.clone()),
                    ProviderConfig::Openai(_) | ProviderConfig::Onnx(_) => None,
                })
                .unwrap_or_else(|| "nomic-embed-text".into());
            cfg.embed = Some(ProviderConfig::Ollama(OllamaConfig {
                model,
                ..Default::default()
            }));
        }
        Some("onnx") => {
            let model = cfg
                .embed
                .as_ref()
                .and_then(|e| match e {
                    ProviderConfig::Onnx(c) => Some(c.model.clone()),
                    ProviderConfig::Openai(_) | ProviderConfig::Ollama(_) => None,
                })
                .unwrap_or_else(|| "bge-large-en-v1.5".into());
            cfg.embed = Some(ProviderConfig::Onnx(OnnxConfig {
                model,
                ..Default::default()
            }));
        }
        Some(other) => {
            anyhow::bail!("unknown embed.provider `{other}` (expected openai|ollama|onnx)")
        }
    }
    Ok(())
}

fn set_embed_model(cfg: &mut Config, value: Option<&str>) -> Result<()> {
    let Some(v) = value else {
        anyhow::bail!(
            "embed.model requires a value; use `mnem config unset embed.provider` to drop the whole section"
        )
    };
    let emb = cfg.embed.as_mut().context(
        "no embed.provider is set; run `mnem config set embed.provider openai|ollama` first",
    )?;
    match emb {
        ProviderConfig::Openai(c) => c.model = v.to_string(),
        ProviderConfig::Ollama(c) => c.model = v.to_string(),
        ProviderConfig::Onnx(c) => {
            const VALID_ONNX: &[&str] = &[
                "bge-large-en-v1.5",
                "bge-base-en-v1.5",
                "bge-small-en-v1.5",
                "all-MiniLM-L6-v2",
            ];
            if !VALID_ONNX.contains(&v) {
                anyhow::bail!(
                    "unknown embed.model `{v}` for onnx; known: {}",
                    VALID_ONNX.join(", ")
                );
            }
            c.model = v.to_string();
        }
    }
    Ok(())
}

fn set_embed_api_key_env(cfg: &mut Config, value: Option<&str>) -> Result<()> {
    let Some(v) = value else {
        anyhow::bail!(
            "embed.api_key_env requires a value (the name of the env var holding the API key)"
        )
    };
    // Require posix-shell env-var shape: [A-Z_][A-Z0-9_]{0,127}. This
    // catches operator mistakes where a secret gets pasted into the
    // api_key_env slot (e.g. `mnem config set embed.api_key_env sk-...`
    // or `AIzaSy...`) regardless of the provider's key prefix.
    let shape_ok = !v.is_empty()
        && v.len() <= 128
        && v.bytes().enumerate().all(|(i, b)| match b {
            b'A'..=b'Z' | b'_' => true,
            b'0'..=b'9' => i > 0,
            _ => false,
        });
    if !shape_ok {
        anyhow::bail!(
            "embed.api_key_env must be a plain env-var name matching [A-Z_][A-Z0-9_]{{0,127}} \
             (e.g. OPENAI_API_KEY), not a secret"
        );
    }
    let emb = cfg
        .embed
        .as_mut()
        .context("no embed.provider is set; run `mnem config set embed.provider openai` first")?;
    match emb {
        ProviderConfig::Openai(c) => c.api_key_env = v.to_string(),
        ProviderConfig::Ollama(_) => {
            anyhow::bail!("embed.api_key_env is only meaningful for openai (ollama has no auth)")
        }
        ProviderConfig::Onnx(_) => {
            anyhow::bail!(
                "embed.api_key_env is only meaningful for openai (onnx runs in-process with no auth)"
            )
        }
    }
    Ok(())
}

fn set_embed_base_url(cfg: &mut Config, value: Option<&str>) -> Result<()> {
    let Some(v) = value else {
        anyhow::bail!("embed.base_url requires a value (e.g. http://localhost:11434 for ollama)")
    };
    let emb = cfg.embed.as_mut().context(
        "no embed.provider is set; run `mnem config set embed.provider openai|ollama` first",
    )?;
    match emb {
        ProviderConfig::Openai(c) => c.base_url = v.to_string(),
        ProviderConfig::Ollama(c) => c.base_url = v.to_string(),
        ProviderConfig::Onnx(_) => {
            anyhow::bail!(
                "embed.base_url is not meaningful for onnx (in-process, no network endpoint)"
            )
        }
    }
    Ok(())
}

fn set_rerank_provider(cfg: &mut Config, value: Option<&str>) -> Result<()> {
    match value {
        None => {
            cfg.rerank = None;
        }
        Some("cohere") => {
            let model = cfg
                .rerank
                .as_ref()
                .and_then(|r| match r {
                    RerankProviderConfig::Cohere(c) => Some(c.model.clone()),
                    _ => None,
                })
                .unwrap_or_else(|| "rerank-v3.5".into());
            cfg.rerank = Some(RerankProviderConfig::Cohere(CohereConfig {
                model,
                ..Default::default()
            }));
        }
        Some("voyage") => {
            let model = cfg
                .rerank
                .as_ref()
                .and_then(|r| match r {
                    RerankProviderConfig::Voyage(c) => Some(c.model.clone()),
                    _ => None,
                })
                .unwrap_or_else(|| "rerank-2.5".into());
            cfg.rerank = Some(RerankProviderConfig::Voyage(VoyageConfig {
                model,
                ..Default::default()
            }));
        }
        Some("jina") => {
            let model = cfg
                .rerank
                .as_ref()
                .and_then(|r| match r {
                    RerankProviderConfig::Jina(c) => Some(c.model.clone()),
                    _ => None,
                })
                .unwrap_or_else(|| "jina-reranker-v3".into());
            cfg.rerank = Some(RerankProviderConfig::Jina(JinaConfig {
                model,
                ..Default::default()
            }));
        }
        Some(other) => {
            anyhow::bail!("unknown rerank.provider `{other}` (expected cohere|voyage|jina)")
        }
    }
    Ok(())
}

fn set_rerank_model(cfg: &mut Config, value: Option<&str>) -> Result<()> {
    let Some(v) = value else {
        anyhow::bail!(
            "rerank.model requires a value; use `mnem config unset rerank.provider` to drop the section"
        )
    };
    let rr = cfg.rerank.as_mut().context(
        "no rerank.provider is set; run `mnem config set rerank.provider cohere|voyage|jina` first",
    )?;
    match rr {
        RerankProviderConfig::Cohere(c) => c.model = v.to_string(),
        RerankProviderConfig::Voyage(c) => c.model = v.to_string(),
        RerankProviderConfig::Jina(c) => c.model = v.to_string(),
    }
    Ok(())
}

fn set_rerank_api_key_env(cfg: &mut Config, value: Option<&str>) -> Result<()> {
    let Some(v) = value else {
        anyhow::bail!(
            "rerank.api_key_env requires a value (the name of the env var holding the API key)"
        )
    };
    let shape_ok = !v.is_empty()
        && v.len() <= 128
        && v.bytes().enumerate().all(|(i, b)| match b {
            b'A'..=b'Z' | b'_' => true,
            b'0'..=b'9' => i > 0,
            _ => false,
        });
    if !shape_ok {
        anyhow::bail!(
            "rerank.api_key_env must be a plain env-var name matching [A-Z_][A-Z0-9_]{{0,127}} \
             (e.g. COHERE_API_KEY), not a secret"
        );
    }
    let rr = cfg.rerank.as_mut().context(
        "no rerank.provider is set; run `mnem config set rerank.provider cohere|voyage|jina` first",
    )?;
    match rr {
        RerankProviderConfig::Cohere(c) => c.api_key_env = v.to_string(),
        RerankProviderConfig::Voyage(c) => c.api_key_env = v.to_string(),
        RerankProviderConfig::Jina(c) => c.api_key_env = v.to_string(),
    }
    Ok(())
}

fn set_rerank_base_url(cfg: &mut Config, value: Option<&str>) -> Result<()> {
    let Some(v) = value else {
        anyhow::bail!("rerank.base_url requires a value")
    };
    let rr = cfg.rerank.as_mut().context(
        "no rerank.provider is set; run `mnem config set rerank.provider cohere|voyage|jina` first",
    )?;
    match rr {
        RerankProviderConfig::Cohere(c) => c.base_url = v.to_string(),
        RerankProviderConfig::Voyage(c) => c.base_url = v.to_string(),
        RerankProviderConfig::Jina(c) => c.base_url = v.to_string(),
    }
    Ok(())
}

fn set_llm_provider(cfg: &mut Config, value: Option<&str>) -> Result<()> {
    match value {
        None => {
            cfg.llm = None;
        }
        Some("openai") => {
            let model = cfg
                .llm
                .as_ref()
                .and_then(|l| match l {
                    LlmProviderConfig::Openai(c) => Some(c.model.clone()),
                    LlmProviderConfig::Ollama(_) => None,
                })
                .unwrap_or_else(|| "gpt-4o-mini".into());
            cfg.llm = Some(LlmProviderConfig::Openai(OpenAiLlmConfig {
                model,
                ..Default::default()
            }));
        }
        Some("ollama") => {
            let model = cfg
                .llm
                .as_ref()
                .and_then(|l| match l {
                    LlmProviderConfig::Ollama(c) => Some(c.model.clone()),
                    LlmProviderConfig::Openai(_) => None,
                })
                .unwrap_or_else(|| "llama3.2:3b".into());
            cfg.llm = Some(LlmProviderConfig::Ollama(OllamaLlmConfig {
                model,
                ..Default::default()
            }));
        }
        Some(other) => {
            anyhow::bail!("unknown llm.provider `{other}` (expected openai|ollama)")
        }
    }
    Ok(())
}

fn set_llm_model(cfg: &mut Config, value: Option<&str>) -> Result<()> {
    let Some(v) = value else {
        anyhow::bail!(
            "llm.model requires a value; use `mnem config unset llm.provider` to drop the whole section"
        )
    };
    let llm = cfg.llm.as_mut().context(
        "no llm.provider is set; run `mnem config set llm.provider openai|ollama` first",
    )?;
    match llm {
        LlmProviderConfig::Openai(c) => c.model = v.to_string(),
        LlmProviderConfig::Ollama(c) => c.model = v.to_string(),
    }
    Ok(())
}

fn set_llm_api_key_env(cfg: &mut Config, value: Option<&str>) -> Result<()> {
    let Some(v) = value else {
        anyhow::bail!(
            "llm.api_key_env requires a value (the name of the env var holding the API key)"
        )
    };
    // Validate env-var shape: [A-Z_][A-Z0-9_]{0,127}. Catches accidental
    // secret paste (e.g. `mnem config set llm.api_key_env sk-...`).
    let shape_ok = !v.is_empty()
        && v.len() <= 128
        && v.bytes().enumerate().all(|(i, b)| match b {
            b'A'..=b'Z' | b'_' => true,
            b'0'..=b'9' => i > 0,
            _ => false,
        });
    if !shape_ok {
        anyhow::bail!(
            "llm.api_key_env must be a plain env-var name matching [A-Z_][A-Z0-9_]{{0,127}} \
             (e.g. OPENAI_API_KEY), not a secret"
        );
    }
    let llm = cfg
        .llm
        .as_mut()
        .context("no llm.provider is set; run `mnem config set llm.provider openai` first")?;
    match llm {
        LlmProviderConfig::Openai(c) => c.api_key_env = v.to_string(),
        LlmProviderConfig::Ollama(_) => {
            anyhow::bail!("llm.api_key_env is only meaningful for openai (ollama has no auth)")
        }
    }
    Ok(())
}

fn set_llm_base_url(cfg: &mut Config, value: Option<&str>) -> Result<()> {
    let Some(v) = value else {
        anyhow::bail!("llm.base_url requires a value (e.g. http://localhost:11434 for ollama)")
    };
    let llm = cfg.llm.as_mut().context(
        "no llm.provider is set; run `mnem config set llm.provider openai|ollama` first",
    )?;
    match llm {
        LlmProviderConfig::Openai(c) => c.base_url = v.to_string(),
        LlmProviderConfig::Ollama(c) => c.base_url = v.to_string(),
    }
    Ok(())
}

/// Default model identifier picked when the `bundled-embedder`
/// feature is compiled in and no other tier resolves an embedder.
/// Path A audit fix (2026-04-26).
///
/// Choice rationale: `all-MiniLM-L6-v2` is 22M params, 384-dim,
/// 92MB on disk, Apache-2.0. Smallest viable transformer for dense
/// English retrieval, byte-for-byte parity with ChromaDB's
/// `DefaultEmbeddingFunction`, and the model the membench harness
/// already uses for the head-to-head bake-off rows in
/// `docs/benchmarks/membench.md`. Lazy-downloaded from
/// `Xenova/all-MiniLM-L6-v2` on first use; cached under
/// `~/.cache/huggingface/hub` for re-use across repos.
// `dead_code` allow when the `bundled-embedder` feature is OFF: the
// constant is genuinely unused in that build, but stripping it under
// `#[cfg]` would force a feature-conditional doc anchor in the
// surrounding doc-comment. Keeping the constant always-defined keeps
// the surface stable.
#[cfg_attr(not(feature = "bundled-embedder"), allow(dead_code))]
pub(crate) const BUNDLED_EMBEDDER_DEFAULT_MODEL: &str = "all-MiniLM-L6-v2";

/// Resolve the effective embedder config. Precedence:
///   1. `MNEM_EMBED_PROVIDER` + `MNEM_EMBED_MODEL` (+ optional
///      `MNEM_EMBED_API_KEY_ENV`, `MNEM_EMBED_BASE_URL`) env vars.
///   2. The `[embed]` section in the passed-in `Config` (per-repo).
///   3. audit-2026-04-25 C4-4: the `[embed]` section in the
///      user-global `~/.mnem/config.toml` (so a fresh repo can
///      ingest with `--extractor keybert` without re-typing the
///      embedder config in every per-repo `.mnem/config.toml`).
///   4. **Path A audit fix (2026-04-26):** when the `bundled-embedder`
///      cargo feature is compiled into this binary AND nothing above
///      resolved, default to `OnnxConfig { model:
///      "all-MiniLM-L6-v2", .. }`. Lets a freshly-installed
///      `cargo install mnem-cli --features bundled-embedder` run
///      semantic retrieve with zero post-install steps and no
///      Ollama daemon. When the feature is NOT compiled in, this
///      tier is skipped and the function returns `None` - the
///      existing graceful empty-success path in mnem mcp + mnem-http
///      handles the no-embedder case unchanged.
///
/// Returns `None` if none of the four yields a provider. Tier 3
/// (global) inheritance is silent (no warning) - the operator opted
/// in by writing `~/.mnem/config.toml`, and the per-repo file
/// always wins when both are set.
///
/// Switching off the bundled fallback once it has been chosen:
/// `mnem config set embed.provider ollama|openai|onnx` writes a
/// per-repo `[embed]` section that wins over the bundled tier (it
/// is tier 2). Setting any explicit `[embed]` in the per-repo or
/// user-global config completely opts out of MiniLM.
pub(crate) fn resolve_embedder(cfg: &Config) -> Option<ProviderConfig> {
    if let Ok(p) = std::env::var("MNEM_EMBED_PROVIDER") {
        let model = std::env::var("MNEM_EMBED_MODEL").ok()?;
        return match p.as_str() {
            "openai" => Some(ProviderConfig::Openai(OpenAiConfig {
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
            "ollama" => Some(ProviderConfig::Ollama(OllamaConfig {
                model,
                base_url: std::env::var("MNEM_EMBED_BASE_URL")
                    .unwrap_or_else(|_| "http://localhost:11434".into()),
                timeout_secs: 30,
            })),
            "onnx" => Some(ProviderConfig::Onnx(OnnxConfig {
                model,
                max_length: None,
            })),
            _ => None,
        };
    }
    if let Some(e) = cfg.embed.clone() {
        return Some(e);
    }
    // audit-2026-04-25 C4-4: per-repo has no [embed]; fall back to
    // the user-global `~/.mnem/config.toml`. A malformed global
    // config is treated as "no fallback" here (the parse error
    // surfaces from `load_global` in callers that explicitly read
    // it; we don't want a typo in the global file to brick every
    // retrieve in every repo).
    if let Some(g) = load_global().ok().and_then(|g| g.embed) {
        return Some(g);
    }
    // Path A audit fix tier 4 (2026-04-26): bundled-embedder
    // cargo-feature default. See doc-comment above for rationale.
    bundled_embedder_default()
}

/// Pure helper for tier 4 of [`resolve_embedder`]. Returns
/// `Some(OnnxConfig{all-MiniLM-L6-v2})` when compiled with the
/// `bundled-embedder` cargo feature; `None` otherwise.
///
/// Factored out so callers can test the tier-4 boundary deterministically
/// from the test runner - `cfg!(feature = "bundled-embedder")` is
/// trivially true/false at compile time, but having a named function
/// gives the test a stable hook to assert against.
#[must_use]
pub(crate) fn bundled_embedder_default() -> Option<ProviderConfig> {
    #[cfg(feature = "bundled-embedder")]
    {
        Some(ProviderConfig::Onnx(OnnxConfig {
            model: BUNDLED_EMBEDDER_DEFAULT_MODEL.to_string(),
            ..Default::default()
        }))
    }
    #[cfg(not(feature = "bundled-embedder"))]
    None
}

/// Resolve the effective reranker config. Precedence mirrors
/// [`resolve_embedder`]:
///   1. `MNEM_RERANK_PROVIDER` + `MNEM_RERANK_MODEL` (+ optional
///      `MNEM_RERANK_API_KEY_ENV`, `MNEM_RERANK_BASE_URL`) env vars.
///   2. The `[rerank]` section in the passed-in `Config`.
///
/// Returns `None` if neither yields a provider.
pub(crate) fn resolve_reranker(cfg: &Config) -> Option<RerankProviderConfig> {
    if let Ok(p) = std::env::var("MNEM_RERANK_PROVIDER") {
        let model = std::env::var("MNEM_RERANK_MODEL").ok()?;
        let key_env = std::env::var("MNEM_RERANK_API_KEY_ENV").ok();
        let base = std::env::var("MNEM_RERANK_BASE_URL").ok();
        return match p.as_str() {
            "cohere" => Some(RerankProviderConfig::Cohere(CohereConfig {
                model,
                api_key_env: key_env.unwrap_or_else(|| "COHERE_API_KEY".into()),
                base_url: base.unwrap_or_else(|| "https://api.cohere.com".into()),
                timeout_secs: 30,
            })),
            "voyage" => Some(RerankProviderConfig::Voyage(VoyageConfig {
                model,
                api_key_env: key_env.unwrap_or_else(|| "VOYAGE_API_KEY".into()),
                base_url: base.unwrap_or_else(|| "https://api.voyageai.com".into()),
                timeout_secs: 30,
            })),
            "jina" => Some(RerankProviderConfig::Jina(JinaConfig {
                model,
                api_key_env: key_env.unwrap_or_else(|| "JINA_API_KEY".into()),
                base_url: base.unwrap_or_else(|| "https://api.jina.ai".into()),
                timeout_secs: 30,
            })),
            _ => None,
        };
    }
    cfg.rerank.clone()
}

/// Resolve the effective NER config. Precedence:
///   1. `MNEM_NER_PROVIDER` env var (`"rule"` or `"none"`).
///   2. The `[ner]` section in the passed-in per-repo `Config`.
///   3. The `[ner]` section in the user-global `~/.mnem/config.toml`.
///   4. [`NerConfig::Rule`] (the always-available zero-dep default).
///
/// Returns a [`NerConfig`], never `None`, because a sane default
/// (`Rule`) is always available without any config.
pub(crate) fn resolve_ner(cfg: &Config) -> NerConfig {
    if let Ok(p) = std::env::var("MNEM_NER_PROVIDER") {
        return match p.to_ascii_lowercase().as_str() {
            "none" => NerConfig::None,
            _ => NerConfig::Rule,
        };
    }
    if let Some(n) = cfg.ner.clone() {
        return n;
    }
    if let Some(n) = load_global().ok().and_then(|g| g.ner) {
        return n;
    }
    NerConfig::Rule
}

/// Parse a `--rerank PROVIDER:MODEL` CLI argument into a
/// [`RerankProviderConfig`]. Used by `mnem retrieve --rerank` as a
/// one-shot override that doesn't require persisting config first.
///
/// The API-key env-var names and base URLs default to the same values
/// used by [`resolve_reranker`]; override via `MNEM_RERANK_API_KEY_ENV`
/// or `MNEM_RERANK_BASE_URL` if needed.
pub(crate) fn parse_rerank_override(spec: &str) -> Result<RerankProviderConfig> {
    let (prov, model) = spec
        .split_once(':')
        .with_context(|| format!("--rerank expects PROVIDER:MODEL, got `{spec}`"))?;
    if model.is_empty() {
        anyhow::bail!("--rerank expects PROVIDER:MODEL with a non-empty model, got `{spec}`");
    }
    let key_env = std::env::var("MNEM_RERANK_API_KEY_ENV").ok();
    let base = std::env::var("MNEM_RERANK_BASE_URL").ok();
    match prov {
        "cohere" => Ok(RerankProviderConfig::Cohere(CohereConfig {
            model: model.into(),
            api_key_env: key_env.unwrap_or_else(|| "COHERE_API_KEY".into()),
            base_url: base.unwrap_or_else(|| "https://api.cohere.com".into()),
            timeout_secs: 30,
        })),
        "voyage" => Ok(RerankProviderConfig::Voyage(VoyageConfig {
            model: model.into(),
            api_key_env: key_env.unwrap_or_else(|| "VOYAGE_API_KEY".into()),
            base_url: base.unwrap_or_else(|| "https://api.voyageai.com".into()),
            timeout_secs: 30,
        })),
        "jina" => Ok(RerankProviderConfig::Jina(JinaConfig {
            model: model.into(),
            api_key_env: key_env.unwrap_or_else(|| "JINA_API_KEY".into()),
            base_url: base.unwrap_or_else(|| "https://api.jina.ai".into()),
            timeout_secs: 30,
        })),
        other => anyhow::bail!("unknown rerank provider `{other}` (expected cohere|voyage|jina)"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn set_rerank_provider_creates_cohere_section() {
        let mut cfg = Config::default();
        set_dotted(&mut cfg, "rerank.provider", Some("cohere".into())).unwrap();
        match cfg.rerank.as_ref().unwrap() {
            RerankProviderConfig::Cohere(c) => {
                assert_eq!(c.model, "rerank-v3.5");
            }
            other => panic!("expected Cohere, got {other:?}"),
        }
    }

    #[test]
    fn set_rerank_model_requires_provider_first() {
        let mut cfg = Config::default();
        let e = set_dotted(&mut cfg, "rerank.model", Some("rerank-2.5".into())).unwrap_err();
        assert!(format!("{e:#}").contains("no rerank.provider is set"));
    }

    #[test]
    fn parse_rerank_override_cohere() {
        let p = parse_rerank_override("cohere:rerank-v3.5").unwrap();
        match p {
            RerankProviderConfig::Cohere(c) => assert_eq!(c.model, "rerank-v3.5"),
            other => panic!("expected Cohere, got {other:?}"),
        }
    }

    #[test]
    fn parse_rerank_override_rejects_missing_colon() {
        let e = parse_rerank_override("cohere").unwrap_err();
        assert!(format!("{e:#}").contains("PROVIDER:MODEL"));
    }

    #[test]
    fn parse_rerank_override_rejects_empty_model() {
        let e = parse_rerank_override("voyage:").unwrap_err();
        assert!(format!("{e:#}").contains("non-empty model"));
    }

    #[test]
    fn parse_rerank_override_rejects_unknown_provider() {
        let e = parse_rerank_override("acme:rr").unwrap_err();
        assert!(format!("{e:#}").contains("unknown rerank provider"));
    }

    #[test]
    fn set_rerank_api_key_blocks_secret_shape() {
        let mut cfg = Config::default();
        set_dotted(&mut cfg, "rerank.provider", Some("cohere".into())).unwrap();
        let e = set_dotted(
            &mut cfg,
            "rerank.api_key_env",
            Some("sk-this-looks-like-a-secret".into()),
        )
        .unwrap_err();
        let msg = format!("{e:#}");
        // Either the `sk-` prefix guardrail or the shape check fires; both are fine.
        assert!(msg.contains("API key") || msg.contains("[A-Z_]"));
    }

    #[test]
    fn rerank_known_keys_are_wired() {
        assert!(KNOWN_KEYS.contains(&"rerank.provider"));
        assert!(KNOWN_KEYS.contains(&"rerank.model"));
        assert!(KNOWN_KEYS.contains(&"rerank.api_key_env"));
        assert!(KNOWN_KEYS.contains(&"rerank.base_url"));
    }

    #[test]
    fn rerank_config_round_trips_through_toml() {
        let mut cfg = Config::default();
        set_dotted(&mut cfg, "rerank.provider", Some("voyage".into())).unwrap();
        set_dotted(&mut cfg, "rerank.model", Some("rerank-2.5".into())).unwrap();
        let s = toml::to_string_pretty(&cfg).unwrap();
        assert!(s.contains("[rerank]"));
        let back: Config = toml::from_str(&s).unwrap();
        assert_eq!(
            get_dotted(&back, "rerank.provider").as_deref(),
            Some("voyage")
        );
        assert_eq!(
            get_dotted(&back, "rerank.model").as_deref(),
            Some("rerank-2.5")
        );
    }

    #[test]
    fn retrieve_defaults_set_and_get() {
        let mut cfg = Config::default();
        assert!(cfg.retrieve.is_none());
        set_dotted(&mut cfg, "retrieve.limit", Some("20".into())).unwrap();
        set_dotted(&mut cfg, "retrieve.budget", Some("500".into())).unwrap();
        set_dotted(&mut cfg, "retrieve.graph_expand", Some("30".into())).unwrap();
        set_dotted(&mut cfg, "retrieve.graph_decay", Some("0.75".into())).unwrap();
        set_dotted(&mut cfg, "retrieve.graph_depth", Some("3".into())).unwrap();
        assert_eq!(get_dotted(&cfg, "retrieve.limit").as_deref(), Some("20"));
        assert_eq!(get_dotted(&cfg, "retrieve.budget").as_deref(), Some("500"));
        assert_eq!(
            get_dotted(&cfg, "retrieve.graph_expand").as_deref(),
            Some("30")
        );
        assert_eq!(
            get_dotted(&cfg, "retrieve.graph_decay").as_deref(),
            Some("0.75")
        );
        assert_eq!(
            get_dotted(&cfg, "retrieve.graph_depth").as_deref(),
            Some("3")
        );
    }

    #[test]
    fn retrieve_defaults_reject_non_integer() {
        let mut cfg = Config::default();
        let err = set_dotted(&mut cfg, "retrieve.limit", Some("twenty".into())).unwrap_err();
        assert!(format!("{err:#}").contains("unsigned integer"));
    }

    #[test]
    fn retrieve_defaults_round_trip_through_toml() {
        let mut cfg = Config::default();
        set_dotted(&mut cfg, "retrieve.limit", Some("15".into())).unwrap();
        set_dotted(&mut cfg, "retrieve.hyde_temperature", Some("0.3".into())).unwrap();
        let s = toml::to_string_pretty(&cfg).unwrap();
        assert!(s.contains("[retrieve]"));
        let back: Config = toml::from_str(&s).unwrap();
        assert_eq!(get_dotted(&back, "retrieve.limit").as_deref(), Some("15"));
        assert_eq!(
            get_dotted(&back, "retrieve.hyde_temperature").as_deref(),
            Some("0.3")
        );
    }

    #[test]
    fn retrieve_defaults_unset_drops_table_when_empty() {
        let mut cfg = Config::default();
        set_dotted(&mut cfg, "retrieve.limit", Some("20".into())).unwrap();
        assert!(cfg.retrieve.is_some());
        // unset the only populated field.
        set_dotted(&mut cfg, "retrieve.limit", None).unwrap();
        assert!(
            cfg.retrieve.is_none(),
            "empty retrieve table should be collapsed"
        );
    }

    #[test]
    fn retrieve_known_keys_are_wired() {
        assert!(KNOWN_KEYS.contains(&"retrieve.limit"));
        assert!(KNOWN_KEYS.contains(&"retrieve.budget"));
        assert!(KNOWN_KEYS.contains(&"retrieve.graph_expand"));
        assert!(KNOWN_KEYS.contains(&"retrieve.graph_decay"));
        assert!(KNOWN_KEYS.contains(&"retrieve.graph_depth"));
        assert!(KNOWN_KEYS.contains(&"retrieve.rerank_top_k"));
        assert!(KNOWN_KEYS.contains(&"retrieve.hyde_max_tokens"));
        assert!(KNOWN_KEYS.contains(&"retrieve.hyde_temperature"));
        assert!(KNOWN_KEYS.contains(&"retrieve.vector_cap"));
    }

    // ---------- Path A audit fix tests (2026-04-26): bundled embedder ----------

    #[test]
    #[cfg(feature = "bundled-embedder")]
    fn bundled_embedder_default_returns_minilm_when_feature_on() {
        let resolved = bundled_embedder_default();
        match resolved {
            Some(ProviderConfig::Onnx(c)) => {
                assert_eq!(c.model, BUNDLED_EMBEDDER_DEFAULT_MODEL);
                assert_eq!(c.model, "all-MiniLM-L6-v2");
            }
            other => {
                panic!("expected Onnx(MiniLM) when bundled-embedder feature on; got {other:?}")
            }
        }
    }

    #[test]
    #[cfg(not(feature = "bundled-embedder"))]
    fn bundled_embedder_default_returns_none_when_feature_off() {
        // Without the feature, the auto-default path is gone; existing
        // graceful empty paths in mnem-mcp / mnem http handle the no-
        // embedder case unchanged.
        assert!(bundled_embedder_default().is_none());
    }

    #[test]
    #[cfg(feature = "bundled-embedder")]
    fn resolve_embedder_falls_back_to_bundled_when_nothing_else_set() {
        // Empty per-repo Config + no env vars + global disabled →
        // tier 4 (bundled) kicks in.
        let cfg = Config::default();
        // SAFETY: the test runner sets MNEM_DISABLE_GLOBAL_CONFIG=1
        // via the workspace-level test fixture (.cargo/config.toml
        // env block); no in-test env mutation needed.
        // If MNEM_EMBED_PROVIDER happens to be set in the dev
        // environment, the test reduces to a tautology - tier 1 wins
        // and tier 4 is not exercised. Skip the assertion in that case.
        if std::env::var("MNEM_EMBED_PROVIDER").is_ok() {
            return;
        }
        // Tier 3 (global) is read from ~/.mnem/config.toml. If the dev
        // machine has a real one, the test reduces to "global tier
        // wins" - also fine. Skip the assertion in that case.
        if load_global().ok().and_then(|g| g.embed).is_some() {
            return;
        }
        let resolved = resolve_embedder(&cfg).expect("tier 4 should yield a provider");
        match resolved {
            ProviderConfig::Onnx(c) => {
                assert_eq!(c.model, BUNDLED_EMBEDDER_DEFAULT_MODEL);
            }
            other => panic!("expected Onnx fallback; got {other:?}"),
        }
    }

    #[test]
    fn resolve_embedder_per_repo_config_wins_over_bundled_default() {
        // Even when bundled-embedder is on, an explicit per-repo
        // [embed] section must win - this is how a customer "switches
        // to a custom embedder later".
        let mut cfg = Config::default();
        cfg.embed = Some(ProviderConfig::Ollama(OllamaConfig {
            model: "nomic-embed-text".into(),
            base_url: "http://localhost:11434".into(),
            timeout_secs: 30,
        }));
        let resolved = resolve_embedder(&cfg).expect("per-repo wins");
        match resolved {
            ProviderConfig::Ollama(c) => assert_eq!(c.model, "nomic-embed-text"),
            other => panic!("expected per-repo Ollama to win; got {other:?}"),
        }
    }

    // ---------- llm.* config key tests ----------

    #[test]
    fn llm_known_keys_are_wired() {
        assert!(KNOWN_KEYS.contains(&"llm.provider"));
        assert!(KNOWN_KEYS.contains(&"llm.model"));
        assert!(KNOWN_KEYS.contains(&"llm.api_key_env"));
        assert!(KNOWN_KEYS.contains(&"llm.base_url"));
    }

    #[test]
    fn set_llm_provider_creates_openai_section() {
        let mut cfg = Config::default();
        set_dotted(&mut cfg, "llm.provider", Some("openai".into())).unwrap();
        match cfg.llm.as_ref().unwrap() {
            LlmProviderConfig::Openai(c) => {
                assert_eq!(c.model, "gpt-4o-mini");
            }
            other => panic!("expected Openai, got {other:?}"),
        }
    }

    #[test]
    fn set_llm_provider_creates_ollama_section() {
        let mut cfg = Config::default();
        set_dotted(&mut cfg, "llm.provider", Some("ollama".into())).unwrap();
        match cfg.llm.as_ref().unwrap() {
            LlmProviderConfig::Ollama(c) => {
                assert_eq!(c.model, "llama3.2:3b");
            }
            other => panic!("expected Ollama, got {other:?}"),
        }
    }

    #[test]
    fn set_llm_provider_none_clears_section() {
        let mut cfg = Config::default();
        set_dotted(&mut cfg, "llm.provider", Some("openai".into())).unwrap();
        assert!(cfg.llm.is_some());
        set_dotted(&mut cfg, "llm.provider", None).unwrap();
        assert!(cfg.llm.is_none());
    }

    #[test]
    fn set_llm_provider_rejects_unknown() {
        let mut cfg = Config::default();
        let e = set_dotted(&mut cfg, "llm.provider", Some("acme".into())).unwrap_err();
        assert!(format!("{e:#}").contains("unknown llm.provider"));
    }

    #[test]
    fn set_llm_model_requires_provider_first() {
        let mut cfg = Config::default();
        let e = set_dotted(&mut cfg, "llm.model", Some("gpt-4o".into())).unwrap_err();
        assert!(format!("{e:#}").contains("no llm.provider is set"));
    }

    #[test]
    fn set_llm_api_key_env_requires_valid_shape() {
        let mut cfg = Config::default();
        set_dotted(&mut cfg, "llm.provider", Some("openai".into())).unwrap();
        let e = set_dotted(
            &mut cfg,
            "llm.api_key_env",
            Some("sk-this-looks-like-a-secret".into()),
        )
        .unwrap_err();
        assert!(format!("{e:#}").contains("[A-Z_]"));
    }

    #[test]
    fn set_llm_api_key_env_rejects_ollama() {
        let mut cfg = Config::default();
        set_dotted(&mut cfg, "llm.provider", Some("ollama".into())).unwrap();
        let e = set_dotted(&mut cfg, "llm.api_key_env", Some("OPENAI_API_KEY".into())).unwrap_err();
        assert!(format!("{e:#}").contains("ollama has no auth"));
    }

    #[test]
    fn llm_config_round_trips_through_toml() {
        let mut cfg = Config::default();
        set_dotted(&mut cfg, "llm.provider", Some("openai".into())).unwrap();
        set_dotted(&mut cfg, "llm.model", Some("gpt-4o".into())).unwrap();
        set_dotted(&mut cfg, "llm.api_key_env", Some("MY_OPENAI_KEY".into())).unwrap();
        set_dotted(
            &mut cfg,
            "llm.base_url",
            Some("https://my-proxy.example.com".into()),
        )
        .unwrap();
        let s = toml::to_string_pretty(&cfg).unwrap();
        assert!(s.contains("[llm]"));
        let back: Config = toml::from_str(&s).unwrap();
        assert_eq!(get_dotted(&back, "llm.provider").as_deref(), Some("openai"));
        assert_eq!(get_dotted(&back, "llm.model").as_deref(), Some("gpt-4o"));
        assert_eq!(
            get_dotted(&back, "llm.api_key_env").as_deref(),
            Some("MY_OPENAI_KEY")
        );
        assert_eq!(
            get_dotted(&back, "llm.base_url").as_deref(),
            Some("https://my-proxy.example.com")
        );
    }

    #[test]
    fn llm_ollama_round_trips_through_toml() {
        let mut cfg = Config::default();
        set_dotted(&mut cfg, "llm.provider", Some("ollama".into())).unwrap();
        set_dotted(&mut cfg, "llm.model", Some("llama3.2:3b".into())).unwrap();
        set_dotted(
            &mut cfg,
            "llm.base_url",
            Some("http://localhost:11434".into()),
        )
        .unwrap();
        let s = toml::to_string_pretty(&cfg).unwrap();
        let back: Config = toml::from_str(&s).unwrap();
        assert_eq!(get_dotted(&back, "llm.provider").as_deref(), Some("ollama"));
        assert_eq!(
            get_dotted(&back, "llm.model").as_deref(),
            Some("llama3.2:3b")
        );
        assert_eq!(
            get_dotted(&back, "llm.base_url").as_deref(),
            Some("http://localhost:11434")
        );
    }
}

/// Resolve the effective LLM config. Precedence:
///   1. `--hyde PROVIDER:MODEL` explicit override (passed in as
///      `override_spec`).
///   2. `MNEM_LLM_PROVIDER` + `MNEM_LLM_MODEL` env vars.
///   3. `[llm]` section in `config.toml`.
///
/// Returns `None` if none yields a provider.
pub(crate) fn resolve_llm(cfg: &Config, override_spec: Option<&str>) -> Option<LlmProviderConfig> {
    if let Some(spec) = override_spec
        && !spec.is_empty()
    {
        let (prov, model) = spec.split_once(':')?;
        if model.is_empty() {
            return None;
        }
        return match prov {
            "openai" => Some(LlmProviderConfig::Openai(OpenAiLlmConfig {
                model: model.into(),
                ..Default::default()
            })),
            "ollama" => Some(LlmProviderConfig::Ollama(OllamaLlmConfig {
                model: model.into(),
                ..Default::default()
            })),
            _ => None,
        };
    }
    if let Ok(p) = std::env::var("MNEM_LLM_PROVIDER") {
        let model = std::env::var("MNEM_LLM_MODEL").ok()?;
        return match p.as_str() {
            "openai" => Some(LlmProviderConfig::Openai(OpenAiLlmConfig {
                model,
                api_key_env: std::env::var("MNEM_LLM_API_KEY_ENV")
                    .unwrap_or_else(|_| "OPENAI_API_KEY".into()),
                base_url: std::env::var("MNEM_LLM_BASE_URL")
                    .unwrap_or_else(|_| "https://api.openai.com".into()),
                timeout_secs: 60,
            })),
            "ollama" => Some(LlmProviderConfig::Ollama(OllamaLlmConfig {
                model,
                base_url: std::env::var("MNEM_LLM_BASE_URL")
                    .unwrap_or_else(|_| "http://localhost:11434".into()),
                timeout_secs: 120,
            })),
            _ => None,
        };
    }
    cfg.llm.clone()
}

/// Author string for commits: `name <email>` if both present; else
/// `name`, `email`, `agent_id`, or the empty string.
pub(crate) fn author_string(cfg: &Config) -> String {
    match (&cfg.user.name, &cfg.user.email) {
        (Some(n), Some(e)) => format!("{n} <{e}>"),
        (Some(n), None) => n.clone(),
        (None, Some(e)) => e.clone(),
        (None, None) => cfg
            .user
            .agent_id
            .clone()
            .unwrap_or_else(|| "mnem-cli".to_string()),
    }
}
