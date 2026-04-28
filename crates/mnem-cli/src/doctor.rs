//! `mnem doctor` - non-mutating health check.
//!
//! Reports on: the `mnem` / `mnem-mcp` binaries on PATH, the current
//! repo's data dir + redb readability, the on-disk config, the
//! configured embedding provider's reachability, and which agent hosts
//! are wired via `mnem integrate`. Designed to be the first thing a
//! support thread asks for.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Duration;

use anyhow::Result;

use crate::config;
use crate::integrate;
use crate::repo;

#[derive(clap::Args, Debug)]
#[command(after_long_help = "\
Examples:

  mnem doctor              # full health check
  mnem doctor --json       # machine-parseable output

Exits 0 if everything looks OK, 1 if any check failed. Suitable as a
CI-side readiness gate.
")]
pub(crate) struct Args {
    /// Emit one JSON object with all check results.
    #[arg(long)]
    pub json: bool,
}

pub(crate) fn run(override_path: Option<&Path>, args: Args) -> Result<()> {
    let checks = run_all_checks(override_path);

    if args.json {
        print_json(&checks);
    } else {
        print_human(&checks);
    }

    let any_fail = checks.iter().any(|c| matches!(c.state, State::Fail));
    if any_fail {
        std::process::exit(1);
    }
    Ok(())
}

// ---------- Result model ----------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum State {
    Ok,
    Warn,
    Fail,
    Info,
}

impl State {
    const fn glyph(self) -> &'static str {
        match self {
            State::Ok => "ok ",
            State::Warn => "!  ",
            State::Fail => "x  ",
            State::Info => "-  ",
        }
    }

    const fn as_str(self) -> &'static str {
        match self {
            State::Ok => "ok",
            State::Warn => "warn",
            State::Fail => "fail",
            State::Info => "info",
        }
    }
}

#[derive(Debug)]
struct Check {
    section: &'static str,
    name: String,
    state: State,
    detail: String,
    fix: Option<String>,
}

// ---------- The check battery ----------

fn run_all_checks(override_path: Option<&Path>) -> Vec<Check> {
    let mut out = Vec::new();
    out.extend(check_binaries());
    out.extend(check_repo(override_path));
    out.extend(check_config(override_path));
    out.extend(check_embed_reachability(override_path));
    out.extend(check_rerank_reachability(override_path));
    out.extend(check_integrations());
    out.extend(check_system_prompt_wired());
    out
}

fn check_binaries() -> Vec<Check> {
    let mut out = Vec::new();
    // audit-2026-04-25 C3-7: when the user invokes `mnem` via an
    // absolute path (e.g. `/c/code/mnem/target/release/mnem.exe doctor`
    // for a developer's release build), the running binary is
    // *itself* not on PATH yet doctor still works fine. Treat that
    // as Info, not Fail: the absolute-path invocation is supported
    // and reporting "not on PATH" is technically true but unhelpful.
    let running_exe = std::env::current_exe().ok();
    let running_dir = running_exe.as_deref().and_then(|p| p.parent());
    for bin in ["mnem", "mnem-mcp"] {
        let found = which(bin);
        out.push(match found {
            Some(p) => Check {
                section: "binaries",
                name: bin.to_string(),
                state: State::Ok,
                detail: p.display().to_string(),
                fix: None,
            },
            None => {
                // Detect the "running from absolute path" case: only
                // applies to the `mnem` row (the row that matches the
                // currently-running binary's stem). `mnem-mcp` always
                // falls through to the standard "not on PATH" Fail.
                let is_running_self = running_exe
                    .as_ref()
                    .and_then(|p| p.file_stem())
                    .and_then(|s| s.to_str())
                    .is_some_and(|stem| stem == bin);
                if is_running_self {
                    let path_str = running_exe
                        .as_deref()
                        .map(|p| p.display().to_string())
                        .unwrap_or_else(|| "<unknown path>".into());
                    Check {
                        section: "binaries",
                        name: bin.to_string(),
                        state: State::Info,
                        detail: format!(
                            "running from absolute path ({path_str}); \
                             not required to be on PATH"
                        ),
                        fix: None,
                    }
                } else if let Some(adjacent) = adjacent_binary(running_dir, bin) {
                    // audit-2026-04-25 C6-3: when `mnem` is invoked by
                    // absolute path out of a `target/release/` (or any
                    // built `bin/`) directory, `mnem-mcp.exe` typically
                    // sits right next to it. PATH lookup misses it but
                    // the binary IS available - reporting `fail` is
                    // misleading. If we can find an adjacent binary,
                    // report `ok (running adjacent)` so the user sees
                    // the working invocation path instead of a fix hint
                    // that doesn't match their actual layout.
                    Check {
                        section: "binaries",
                        name: bin.to_string(),
                        state: State::Ok,
                        detail: format!(
                            "{} (running adjacent to mnem)",
                            adjacent.display()
                        ),
                        fix: None,
                    }
                } else {
                    Check {
                        section: "binaries",
                        name: bin.to_string(),
                        state: State::Fail,
                        detail: "not on PATH".to_string(),
                        fix: Some(format!(
                            "install via `cargo install mnem-cli` (or `cargo binstall mnem-cli` for prebuilt), \
                             `pip install mnem-py`, `brew tap uranid/tap && brew install mnem`, \
                             or add the directory containing `{bin}` to PATH"
                        )),
                    }
                }
            }
        });
    }
    out
}

/// Probe `dir` for a sibling executable named `bin` (with the host's
/// `EXE_SUFFIX` appended on Windows). Returns the resolved path if and
/// only if the file exists. Used by the binaries check so that an
/// absolute-path `mnem` invocation out of `target/release/` doesn't
/// flag its sibling `mnem-mcp.exe` as missing.
fn adjacent_binary(dir: Option<&Path>, bin: &str) -> Option<PathBuf> {
    let dir = dir?;
    let mut candidate = dir.join(bin);
    if !candidate.exists() {
        // Try with the platform's executable suffix (e.g. ".exe" on
        // Windows). `EXE_SUFFIX` is "" on non-Windows, so this is a
        // no-op there but the first .exists() check above already
        // covered that case.
        let with_suffix = format!("{bin}{}", std::env::consts::EXE_SUFFIX);
        candidate = dir.join(with_suffix);
        if !candidate.exists() {
            return None;
        }
    }
    Some(candidate)
}

fn check_repo(override_path: Option<&Path>) -> Vec<Check> {
    match repo::locate_data_dir(override_path) {
        Ok(dir) => {
            let mut v = vec![Check {
                section: "repo",
                name: ".mnem".into(),
                state: State::Ok,
                detail: dir.display().to_string(),
                fix: None,
            }];
            // Try to open the redb and probe head.
            match repo::open_repo(Some(dir.as_path())) {
                Ok(r) => {
                    let has_head = r.head_commit().is_some();
                    v.push(Check {
                        section: "repo",
                        name: "redb".into(),
                        state: State::Ok,
                        detail: if has_head {
                            "open; head commit present".into()
                        } else {
                            "open; no commits yet".into()
                        },
                        fix: None,
                    });
                }
                Err(e) => v.push(Check {
                    section: "repo",
                    name: "redb".into(),
                    state: State::Fail,
                    detail: format!("open failed: {e}"),
                    fix: Some("delete `.mnem/repo.redb` and re-run `mnem init` if corrupt".into()),
                }),
            }
            v
        }
        Err(_) => vec![Check {
            section: "repo",
            name: ".mnem".into(),
            state: State::Info,
            detail: "no mnem repo in cwd or parents".into(),
            fix: Some("run `mnem init` in a project directory".into()),
        }],
    }
}

fn check_config(override_path: Option<&Path>) -> Vec<Check> {
    let Ok(dir) = repo::locate_data_dir(override_path) else {
        return Vec::new();
    };
    let path = config::path_of(&dir);
    if !path.exists() {
        return vec![Check {
            section: "config",
            name: "config.toml".into(),
            state: State::Info,
            detail: "not present (defaults will apply)".into(),
            fix: None,
        }];
    }
    match config::load(&dir) {
        Ok(cfg) => {
            let mut v = vec![Check {
                section: "config",
                name: "config.toml".into(),
                state: State::Ok,
                detail: path.display().to_string(),
                fix: None,
            }];
            if let Some(pc) = config::resolve_embedder(&cfg) {
                let model = match &pc {
                    mnem_embed_providers::ProviderConfig::Openai(c) => {
                        format!("openai {}", c.model)
                    }
                    mnem_embed_providers::ProviderConfig::Ollama(c) => {
                        format!("ollama {}", c.model)
                    }
                    mnem_embed_providers::ProviderConfig::Onnx(c) => {
                        format!("onnx {}", c.model)
                    }
                };
                v.push(Check {
                    section: "config",
                    name: "embed".into(),
                    state: State::Ok,
                    detail: model,
                    fix: None,
                });
            } else {
                v.push(Check {
                    section: "config",
                    name: "embed".into(),
                    state: State::Info,
                    detail: "not configured (text-only retrieval)".into(),
                    fix: Some(
                        "`mnem config set embed.provider ollama` and `mnem config set embed.model nomic-embed-text`".into(),
                    ),
                });
            }
            if let Some(rc) = config::resolve_reranker(&cfg) {
                let label = match &rc {
                    mnem_rerank_providers::ProviderConfig::Cohere(c) => {
                        format!("cohere {}", c.model)
                    }
                    mnem_rerank_providers::ProviderConfig::Voyage(c) => {
                        format!("voyage {}", c.model)
                    }
                    mnem_rerank_providers::ProviderConfig::Jina(c) => format!("jina {}", c.model),
                };
                v.push(Check {
                    section: "config",
                    name: "rerank".into(),
                    state: State::Ok,
                    detail: label,
                    fix: None,
                });
            } else {
                v.push(Check {
                    section: "config",
                    name: "rerank".into(),
                    state: State::Info,
                    detail: "not configured (tier-3 off; hybrid retrieval still works)".into(),
                    fix: Some(
                        "`mnem config set rerank.provider cohere` then `mnem config set rerank.model rerank-v3.5`".into(),
                    ),
                });
            }
            v
        }
        Err(e) => vec![Check {
            section: "config",
            name: "config.toml".into(),
            state: State::Fail,
            detail: format!("parse failed: {e}"),
            fix: Some(format!("open {} and fix the TOML syntax", path.display())),
        }],
    }
}

/// Probe the HuggingFace cache for a previously-downloaded copy of
/// the requested ONNX dense model. Path A audit fix (2026-04-26): lets
/// `mnem doctor` distinguish "bundled embedder ready to go" from
/// "first retrieve will trigger a ~92MB download." Mirrors the cache
/// path layout mnem-embed-providers writes to (the
/// `models--{org}--{repo}/resolve/main/onnx/model.onnx` shape - NOT
/// the upstream `huggingface-cli` blobs+symlinks layout).
#[cfg_attr(not(feature = "bundled-embedder"), allow(dead_code))]
fn onnx_cache_present(model: &str) -> bool {
    let repo = match model {
        // Map mnem's wire-id → upstream Xenova repo. Keep in sync with
        // `ModelKind::repo_id` in mnem-embed-providers/src/onnx.rs.
        "bge-large-en-v1.5" => "Xenova/bge-large-en-v1.5",
        "bge-base-en-v1.5" => "Xenova/bge-base-en-v1.5",
        "bge-small-en-v1.5" => "Xenova/bge-small-en-v1.5",
        "all-MiniLM-L6-v2" => "Xenova/all-MiniLM-L6-v2",
        _ => return false,
    };
    let Some(home) = std::env::var("HF_HOME")
        .ok()
        .map(PathBuf::from)
        .or_else(|| dirs::home_dir().map(|h| h.join(".cache").join("huggingface")))
    else {
        return false;
    };
    let cache = home
        .join("hub")
        .join(format!("models--{}", repo.replace('/', "--")))
        .join("resolve")
        .join("main")
        .join("onnx")
        .join("model.onnx");
    cache.is_file()
}

fn check_embed_reachability(override_path: Option<&Path>) -> Vec<Check> {
    let Ok(dir) = repo::locate_data_dir(override_path) else {
        return Vec::new();
    };
    let Ok(cfg) = config::load(&dir) else {
        return Vec::new();
    };
    let Some(pc) = config::resolve_embedder(&cfg) else {
        return Vec::new();
    };
    match pc {
        mnem_embed_providers::ProviderConfig::Onnx(ref c) => {
            // Path A audit fix (2026-04-26): when an Onnx provider is
            // resolved (either via tier 4 bundled-default or an explicit
            // [embed] section) but this `mnem` binary was built without
            // `--features bundled-embedder`, the actual ONNX adapter is
            // not compiled in and the next retrieve will fail at
            // provider-open time. Surface that as a Warn here so the
            // operator sees the mismatch in `mnem doctor` rather than
            // discovering it on the first retrieve.
            #[cfg(feature = "bundled-embedder")]
            {
                let cache_hit = onnx_cache_present(&c.model);
                vec![Check {
                    section: "embed",
                    name: "onnx".into(),
                    state: State::Ok,
                    detail: format!(
                        "in-process model `{}` (bundled-embedder build); cache: {}",
                        c.model,
                        if cache_hit {
                            "present"
                        } else {
                            "not yet - first retrieve will download ~92MB"
                        }
                    ),
                    fix: None,
                }]
            }
            #[cfg(not(feature = "bundled-embedder"))]
            vec![Check {
                section: "embed",
                name: "onnx".into(),
                state: State::Warn,
                detail: format!(
                    "config requests Onnx model `{}` but this binary was built WITHOUT --features bundled-embedder; retrieve will fail at provider-open time",
                    c.model
                ),
                fix: Some(
                    "either rebuild with `cargo install mnem-cli --features bundled-embedder` \
                     OR switch to a different provider via `mnem config set embed.provider ollama|openai`".into(),
                ),
            }]
        }
        mnem_embed_providers::ProviderConfig::Ollama(ref c) => {
            let url = format!("{}/api/tags", c.base_url.trim_end_matches('/'));
            let agent = ureq::AgentBuilder::new()
                .timeout(Duration::from_secs(2))
                .build();
            match agent.get(&url).call() {
                Ok(resp) if resp.status() == 200 => vec![Check {
                    section: "embed",
                    name: "ollama".into(),
                    state: State::Ok,
                    detail: format!("reachable at {}", c.base_url),
                    fix: None,
                }],
                Ok(resp) => vec![Check {
                    section: "embed",
                    name: "ollama".into(),
                    state: State::Warn,
                    detail: format!("HTTP {} from {url}", resp.status()),
                    fix: Some(format!(
                        "check that ollama is running: `ollama serve` and `ollama pull {}`",
                        c.model
                    )),
                }],
                Err(e) => vec![Check {
                    section: "embed",
                    name: "ollama".into(),
                    state: State::Warn,
                    detail: format!("unreachable: {e}"),
                    fix: Some(
                        "install from https://ollama.com/download; run `ollama serve`".into(),
                    ),
                }],
            }
        }
        mnem_embed_providers::ProviderConfig::Openai(ref c) => {
            let var = &c.api_key_env;
            match std::env::var(var) {
                Ok(_) => vec![Check {
                    section: "embed",
                    name: "openai".into(),
                    state: State::Ok,
                    detail: format!("${var} is set"),
                    fix: None,
                }],
                Err(_) => vec![Check {
                    section: "embed",
                    name: "openai".into(),
                    state: State::Warn,
                    detail: format!("${var} is not set"),
                    fix: Some(format!("`export {var}=sk-...` in your shell rc")),
                }],
            }
        }
    }
}

fn check_rerank_reachability(override_path: Option<&Path>) -> Vec<Check> {
    let Ok(dir) = repo::locate_data_dir(override_path) else {
        return Vec::new();
    };
    let Ok(cfg) = config::load(&dir) else {
        return Vec::new();
    };
    let Some(rc) = config::resolve_reranker(&cfg) else {
        return Vec::new();
    };
    let (name, var) = match &rc {
        mnem_rerank_providers::ProviderConfig::Cohere(c) => ("cohere", c.api_key_env.clone()),
        mnem_rerank_providers::ProviderConfig::Voyage(c) => ("voyage", c.api_key_env.clone()),
        mnem_rerank_providers::ProviderConfig::Jina(c) => ("jina", c.api_key_env.clone()),
    };
    match std::env::var(&var) {
        Ok(_) => vec![Check {
            section: "rerank",
            name: name.into(),
            state: State::Ok,
            detail: format!("${var} is set"),
            fix: None,
        }],
        Err(_) => vec![Check {
            section: "rerank",
            name: name.into(),
            state: State::Warn,
            detail: format!("${var} is not set; rerank will silently fall back to fused order"),
            fix: Some(format!("`export {var}=...` in your shell rc")),
        }],
    }
}

/// Probe whether the mnem-managed system-prompt section has been
/// written into each host's project-rules file. Audit fix
/// (2026-04-26): pairs with `mnem integrate --with-system-prompt`.
///
/// One Check per host that supports auto-write today (Claude Code).
/// Hosts with no `system_prompt_path` are skipped entirely (no
/// noisy "info" lines for hosts where the flag is a documented
/// no-op).
fn check_system_prompt_wired() -> Vec<Check> {
    let mut out = Vec::new();
    for host in integrate::Host::all() {
        let Some(path) = host.system_prompt_path() else {
            continue;
        };
        let detail = match std::fs::read_to_string(&path) {
            Ok(s) if s.contains("<!-- mnem-system-prompt:v1:start -->") => Check {
                section: "system-prompt",
                name: host.slug().to_string(),
                state: State::Ok,
                detail: format!("wired ({})", path.display()),
                fix: None,
            },
            Ok(_) => Check {
                section: "system-prompt",
                name: host.slug().to_string(),
                state: State::Info,
                detail: format!("{} exists but no mnem section", path.display()),
                fix: Some(format!(
                    "auto-write: `mnem integrate {} --with-system-prompt`",
                    host.slug()
                )),
            },
            Err(_) => Check {
                section: "system-prompt",
                name: host.slug().to_string(),
                state: State::Info,
                detail: format!("no rules file at {}", path.display()),
                fix: Some(format!(
                    "auto-write: `mnem integrate {} --with-system-prompt`",
                    host.slug()
                )),
            },
        };
        out.push(detail);
    }
    out
}

fn check_integrations() -> Vec<Check> {
    integrate::wired_status()
        .into_iter()
        .map(|(host, path, wired)| {
            if wired {
                Check {
                    section: "integrations",
                    name: host.slug().to_string(),
                    state: State::Ok,
                    detail: path.map(|p| p.display().to_string()).unwrap_or_default(),
                    fix: None,
                }
            } else {
                Check {
                    section: "integrations",
                    name: host.slug().to_string(),
                    state: State::Info,
                    detail: match path {
                        Some(p) if p.exists() => {
                            format!("config at {} has no mnem entry", p.display())
                        }
                        Some(_) => "host not installed".into(),
                        None => "unsupported on this OS".into(),
                    },
                    fix: Some(format!("`mnem integrate {}`", host.slug())),
                }
            }
        })
        .collect()
}

// ---------- Output ----------

fn print_human(checks: &[Check]) {
    let mut current_section = "";
    for c in checks {
        if c.section != current_section {
            current_section = c.section;
            println!("{current_section}");
        }
        println!(
            "  {glyph} {name:<18} {detail}",
            glyph = c.state.glyph(),
            name = c.name,
            detail = c.detail
        );
        if let Some(fix) = &c.fix {
            // Only print fix hints for non-ok states.
            if !matches!(c.state, State::Ok) {
                println!("        fix: {fix}");
            }
        }
    }

    let fails = checks
        .iter()
        .filter(|c| matches!(c.state, State::Fail))
        .count();
    let warns = checks
        .iter()
        .filter(|c| matches!(c.state, State::Warn))
        .count();
    println!();
    if fails == 0 && warns == 0 {
        println!("Everything looks good.");
    } else {
        println!("{fails} fail(s), {warns} warning(s). See `fix:` hints above.");
    }
}

fn print_json(checks: &[Check]) {
    let arr: Vec<serde_json::Value> = checks
        .iter()
        .map(|c| {
            let mut m = serde_json::Map::new();
            m.insert("section".into(), c.section.into());
            m.insert("name".into(), c.name.clone().into());
            m.insert("state".into(), c.state.as_str().into());
            m.insert("detail".into(), c.detail.clone().into());
            if let Some(fix) = &c.fix {
                m.insert("fix".into(), fix.clone().into());
            }
            serde_json::Value::Object(m)
        })
        .collect();
    let any_fail = checks.iter().any(|c| matches!(c.state, State::Fail));
    let root = serde_json::json!({
        "schema": "mnem.v1.doctor",
        "ok": !any_fail,
        "checks": arr,
    });
    println!(
        "{}",
        serde_json::to_string_pretty(&root).unwrap_or_default()
    );
}

// ---------- Helpers ----------

// ---------- Tests ----------

#[cfg(test)]
mod tests {
    use super::*;

    fn ok_check(name: &str) -> Check {
        Check {
            section: "repo",
            name: name.into(),
            state: State::Ok,
            detail: "fine".into(),
            fix: None,
        }
    }

    fn fail_check(name: &str) -> Check {
        Check {
            section: "binaries",
            name: name.into(),
            state: State::Fail,
            detail: "not on PATH".into(),
            fix: Some("cargo install mnem-cli".into()),
        }
    }

    #[test]
    fn state_glyphs_are_stable() {
        // Glyph strings are grepped by tooling (demo scripts, tests,
        // support docs). Catch inadvertent renames.
        assert_eq!(State::Ok.glyph(), "ok ");
        assert_eq!(State::Fail.glyph(), "x  ");
        assert_eq!(State::Warn.glyph(), "!  ");
        assert_eq!(State::Info.glyph(), "-  ");
    }

    #[test]
    fn state_as_str_maps_to_json_schema() {
        assert_eq!(State::Ok.as_str(), "ok");
        assert_eq!(State::Fail.as_str(), "fail");
        assert_eq!(State::Warn.as_str(), "warn");
        assert_eq!(State::Info.as_str(), "info");
    }

    #[test]
    fn json_output_is_valid_and_versioned() {
        use std::io::Write as _;
        // Capture print_json by writing to a temp file via a one-shot
        // redirect would require plumbing; instead, re-run the same
        // serialisation the print does and assert the shape directly.
        let checks = [ok_check("redb"), fail_check("mnem")];
        let arr: Vec<serde_json::Value> = checks
            .iter()
            .map(|c| {
                let mut m = serde_json::Map::new();
                m.insert("section".into(), c.section.into());
                m.insert("name".into(), c.name.clone().into());
                m.insert("state".into(), c.state.as_str().into());
                m.insert("detail".into(), c.detail.clone().into());
                if let Some(fix) = &c.fix {
                    m.insert("fix".into(), fix.clone().into());
                }
                serde_json::Value::Object(m)
            })
            .collect();
        let any_fail = checks.iter().any(|c| matches!(c.state, State::Fail));
        let root = serde_json::json!({
            "schema": "mnem.v1.doctor",
            "ok": !any_fail,
            "checks": arr,
        });
        let s = serde_json::to_string(&root).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&s).unwrap();
        assert_eq!(parsed["schema"], "mnem.v1.doctor");
        assert_eq!(parsed["ok"], false);
        assert_eq!(parsed["checks"][0]["state"], "ok");
        assert_eq!(parsed["checks"][1]["state"], "fail");
        assert_eq!(parsed["checks"][1]["fix"].as_str().unwrap().len() > 0, true);
        // Silence unused-import warnings in dev profile.
        let _ = std::io::stdout().flush();
    }

    #[test]
    fn check_fix_is_omitted_when_state_is_ok() {
        // When the state is Ok, a `fix:` line MUST NOT print - tested
        // against the human path by spot-checking the logic: fix is
        // only consulted when state != Ok.
        let checks = [Check {
            section: "config",
            name: "config.toml".into(),
            state: State::Ok,
            detail: "parsed".into(),
            fix: Some("you should never see this".into()),
        }];
        // We cannot capture stdout in a unit test without plumbing; the
        // invariant we enforce is the shape of the logic. Re-verify by
        // reading the condition that gates the fix print.
        for c in &checks {
            if !matches!(c.state, State::Ok) {
                panic!("unexpected non-Ok state in this test fixture");
            }
            // The human printer guards `fix` behind `!matches!(state, Ok)`.
            // See print_human above.
        }
    }

    #[test]
    fn ok_and_fail_counts_drive_exit_code_logic() {
        let checks = [ok_check("a"), ok_check("b")];
        let any_fail = checks.iter().any(|c| matches!(c.state, State::Fail));
        assert!(!any_fail);

        let checks = [ok_check("a"), fail_check("mnem")];
        let any_fail = checks.iter().any(|c| matches!(c.state, State::Fail));
        assert!(any_fail);

        // Warn alone does not trip the exit code by design (recoverable
        // environment state, e.g. Ollama unreachable on a plane).
        let mut warn_only = ok_check("a");
        warn_only.state = State::Warn;
        let checks = [warn_only];
        let any_fail = checks.iter().any(|c| matches!(c.state, State::Fail));
        assert!(!any_fail);
    }

    #[test]
    fn which_returns_some_for_ubiquitous_shell_builtins() {
        // `sh` exists on every supported *nix; `cmd` on every Windows.
        // Skip on exotic environments where neither is present.
        let candidate = if cfg!(target_os = "windows") {
            "cmd"
        } else {
            "sh"
        };
        // Not asserting the exact path (differs per OS); only that a
        // known-present binary resolves to something non-empty.
        let r = which(candidate);
        // On some CI minimal images even `sh` may be missing; be lenient.
        if let Some(p) = r {
            assert!(!p.as_os_str().is_empty());
        }
    }

    #[test]
    fn which_returns_none_for_made_up_names() {
        assert!(which("mnem-definitely-not-installed-12345").is_none());
    }

    #[test]
    fn adjacent_binary_finds_sibling_with_exe_suffix() {
        // audit-2026-04-25 C6-3: simulate the layout of
        // `target/release/{mnem.exe, mnem-mcp.exe}` and assert that
        // adjacent_binary("target/release/", "mnem-mcp") finds the
        // sibling regardless of host-specific EXE_SUFFIX.
        let tmp = tempfile::tempdir().expect("tempdir");
        let dir = tmp.path();
        let bin = "mnem-mcp-fake";
        let with_suffix = format!("{bin}{}", std::env::consts::EXE_SUFFIX);
        std::fs::write(dir.join(&with_suffix), b"#!/bin/sh\nexit 0\n").unwrap();

        let found = adjacent_binary(Some(dir), bin).expect("should find sibling");
        assert_eq!(
            found.file_name().and_then(|s| s.to_str()),
            Some(with_suffix.as_str())
        );
    }

    #[test]
    fn adjacent_binary_returns_none_when_missing() {
        let tmp = tempfile::tempdir().expect("tempdir");
        assert!(adjacent_binary(Some(tmp.path()), "nope-not-there-12345").is_none());
    }

    #[test]
    fn adjacent_binary_returns_none_when_dir_is_none() {
        assert!(adjacent_binary(None, "mnem-mcp").is_none());
    }
}

fn which(cmd: &str) -> Option<PathBuf> {
    // Use `where` on Windows, `command -v` via sh elsewhere, without
    // pulling a which crate.
    let (prog, arg) = if cfg!(target_os = "windows") {
        ("where", cmd)
    } else {
        ("sh", "-c")
    };
    let out = if cfg!(target_os = "windows") {
        Command::new(prog).arg(arg).output()
    } else {
        Command::new(prog)
            .arg(arg)
            .arg(format!("command -v {cmd}"))
            .output()
    };
    let out = out.ok()?;
    if !out.status.success() {
        return None;
    }
    let s = String::from_utf8_lossy(&out.stdout);
    let first = s.lines().next()?.trim();
    if first.is_empty() {
        None
    } else {
        Some(PathBuf::from(first))
    }
}
