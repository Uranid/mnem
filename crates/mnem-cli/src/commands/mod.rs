//! CLI subcommand implementations. Each submodule exports `Args` (its
//! clap struct) and `run(repo_override, args) -> Result<()>`. Split from
//! a single-file layout in R3 for readability; the top-level `commands`
//! re-exports nothing -- callers (`main.rs`) reach the submodules directly
//! via their concrete paths (`commands::init::run`, `commands::add::AddCmd`,
//! ...).

pub(crate) mod add;
pub(crate) mod bench;
pub(crate) mod blame;
pub(crate) mod branch;
pub(crate) mod cat_file;
pub(crate) mod cfg_cmd;
pub(crate) mod clone;
pub(crate) mod completions;
pub(crate) mod deferred;
pub(crate) mod diff;
pub(crate) mod embed_cmd;
pub(crate) mod export;
pub(crate) mod fetch;
pub(crate) mod import;
pub(crate) mod ingest;
pub(crate) mod init;
pub(crate) mod log;
pub(crate) mod merge;
pub(crate) mod pull;
pub(crate) mod push;
pub(crate) mod query;
pub(crate) mod refs;
pub(crate) mod reindex;
pub(crate) mod remote;
pub(crate) mod retrieve;
pub(crate) mod show;
pub(crate) mod stats;
pub(crate) mod status;

use std::path::Path;

use anyhow::{Context, Result, anyhow, bail};
use ipld_core::ipld::Ipld;
use mnem_core::codec::{from_canonical_bytes, json_to_ipld};
use mnem_core::id::{EdgeId, NodeId};
use mnem_core::index::{PropPredicate, Query};
use mnem_core::objects::{Commit, Edge, IndexSet, Node, Operation, RefTarget};
use mnem_core::repo::ReadonlyRepo;
use serde_json::Value;

use crate::config;
use crate::repo;

// ---------------- shared helpers ----------------

/// audit-2026-04-25 P1-5: normalise a user-supplied path so git-bash
/// style `/c/Users/...` POSIX paths resolve on Windows. The MSYS2 /
/// git-bash shell rewrites drive letters to `/<letter>/...` before
/// passing them to native tools, but the Win32 file API does not
/// understand that form. We rewrite `/<drive-letter>/...` ->
/// `<drive-letter>:/...` on Windows; on every other platform the
/// input is returned unchanged.
///
/// Only applies when the input looks unambiguously like the git-bash
/// form (single ASCII letter between two `/`s, no Windows-style
/// `\\`). Anything else is returned untouched so absolute Linux
/// paths inside containers still work.
pub(super) fn normalize_cli_path(input: &str) -> String {
    #[cfg(windows)]
    {
        let bytes = input.as_bytes();
        if bytes.len() >= 3
            && bytes[0] == b'/'
            && bytes[1].is_ascii_alphabetic()
            && bytes[2] == b'/'
            && !input.contains('\\')
        {
            let mut out = String::with_capacity(input.len() + 1);
            out.push(bytes[1] as char);
            out.push(':');
            out.push_str(&input[2..]);
            return out;
        }
    }
    let _ = input; // silence warning on non-windows
    input.to_string()
}

#[cfg(test)]
mod normalize_cli_path_tests {
    use super::normalize_cli_path;

    #[test]
    #[cfg(windows)]
    fn rewrites_git_bash_drive_letter() {
        assert_eq!(normalize_cli_path("/c/tmp/out.car"), "c:/tmp/out.car");
        assert_eq!(normalize_cli_path("/D/data"), "D:/data");
    }

    #[test]
    #[cfg(windows)]
    fn leaves_native_windows_paths_alone() {
        assert_eq!(normalize_cli_path(r"C:\tmp\out.car"), r"C:\tmp\out.car");
        assert_eq!(normalize_cli_path("C:/tmp/out.car"), "C:/tmp/out.car");
    }

    #[test]
    fn leaves_relative_paths_alone() {
        assert_eq!(normalize_cli_path("out.car"), "out.car");
        assert_eq!(normalize_cli_path("./out.car"), "./out.car");
    }
}

/// audit-2026-04-25 P2-3: shared "commit-ish" resolver -- accept a
/// raw CID, an exact ref name, or a short branch name (resolved as
/// `refs/heads/<name>`). Plus the special tokens `HEAD` and `head`,
/// which map to the current view's first head commit.
///
/// Mirrors `merge::resolve_commitish` so `mnem diff` / `mnem show`
/// reach feature parity with `mnem merge` post-audit.
pub(super) fn resolve_commitish(r: &ReadonlyRepo, s: &str) -> Result<mnem_core::id::Cid> {
    // HEAD (case-insensitive) -> first view head.
    if s.eq_ignore_ascii_case("HEAD") {
        return r
            .view()
            .heads
            .first()
            .cloned()
            .ok_or_else(|| anyhow!("repository has no commits yet (HEAD unresolved)"));
    }
    if let Ok(cid) = mnem_core::id::Cid::parse_str(s) {
        return Ok(cid);
    }
    let refs = &r.view().refs;
    let candidate = if refs.contains_key(s) {
        s.to_string()
    } else {
        format!("refs/heads/{s}")
    };
    match refs.get(&candidate) {
        Some(RefTarget::Normal { target }) => Ok(target.clone()),
        Some(RefTarget::Conflicted { .. }) => {
            bail!("ref `{candidate}` is conflicted; resolve the ref first")
        }
        None => bail!(
            "cannot resolve `{s}` to a commit. Tried HEAD alias, raw CID, \
             ref `{s}`, and `refs/heads/{s}`."
        ),
    }
}

/// audit-2026-04-25 R2 (Stage E re-fix): operation-CID resolver.
///
/// `mnem diff` operates on the **op DAG** (each side decodes as an
/// [`mnem_core::objects::Operation`]), not the commit DAG that
/// [`resolve_commitish`] yields. The two CID spaces are different:
/// `view().heads` and `refs/heads/<name>` both point at *commit*
/// CIDs, while `r.op_id()` is the current op CID. Calling
/// `resolve_commitish` for `diff` produced the bug surfaced in V4
/// (`decode: Msg("missing field 'view'")`) because the resulting
/// commit bytes were force-decoded as an Operation.
///
/// Accepts:
///   - `HEAD` / `head` -> `r.op_id().clone()` (the current op).
///   - A raw CID string -> trusted as an op CID.
///
/// Named refs are intentionally NOT supported: refs target commits,
/// not ops, and the op-heads store is not addressable by ref name.
/// Surfacing a clear error here is honest; silently picking the
/// wrong CID space is the bug we just fixed.
pub(super) fn resolve_op_commitish(r: &ReadonlyRepo, s: &str) -> Result<mnem_core::id::Cid> {
    if s.eq_ignore_ascii_case("HEAD") {
        return Ok(r.op_id().clone());
    }
    if let Ok(cid) = mnem_core::id::Cid::parse_str(s) {
        return Ok(cid);
    }
    bail!(
        "cannot resolve `{s}` to an op-CID. `mnem diff` accepts `HEAD` or a \
         raw op CID (find them via `mnem log`). Named refs (e.g. branch \
         names) point at commits, not ops, and are not supported here."
    )
}

/// Parse a `KEY=VALUE` prop argument. VALUE is first tried as JSON so
/// `count=3`, `active=true`, `tags=[\"a\",\"b\"]` all get typed; if
/// parsing fails, it's treated as a string.
pub(super) fn parse_prop(arg: &str) -> Result<(String, Ipld)> {
    let (k, v) = arg
        .split_once('=')
        .ok_or_else(|| anyhow!("expected KEY=VALUE, got `{arg}`"))?;
    let value = match serde_json::from_str::<Value>(v) {
        // Error boundary: the canonical converter returns
        // `JsonIpldError`; `anyhow::Error: From<E: Error + Send +
        // Sync>` threads the message through with the KEY attached
        // as context so the user sees which prop was rejected.
        Ok(json) => json_to_ipld(&json).with_context(|| format!("prop `{k}`"))?,
        Err(_) => Ipld::String(v.to_string()),
    };
    Ok((k.to_string(), value))
}

pub(super) fn ipld_preview(v: &Ipld) -> String {
    match v {
        Ipld::Null => "null".into(),
        Ipld::Bool(b) => b.to_string(),
        Ipld::Integer(n) => n.to_string(),
        Ipld::Float(f) => f.to_string(),
        Ipld::String(s) => {
            if s.len() <= 80 {
                format!("\"{s}\"")
            } else {
                // Char-based truncation avoids a panic on multibyte
                // codepoints that cross byte index 77.
                let preview: String = s.chars().take(77).collect();
                format!("\"{preview}...\" ({}B)", s.len())
            }
        }
        Ipld::Bytes(b) => format!("bytes({})", b.len()),
        Ipld::List(xs) => format!("[{} items]", xs.len()),
        Ipld::Map(m) => format!("{{{} keys}}", m.len()),
        Ipld::Link(c) => format!("cid:{c}"),
    }
}

/// Turn an `EmbedError` into a short, actionable one-liner suitable for
/// `eprintln!`. Replaces the ureq-error wall of text with
/// provider-specific remediation, so the user sees "Is `ollama serve`
/// running?" instead of "Connect error: No connection could be made..."
///
/// `context` is either "embedding" (on write) or "query embedding" (on
/// retrieve); the suggestion text adapts to the call-site.
pub(super) fn format_embed_failure(
    err: &mnem_embed_providers::EmbedError,
    pc: &mnem_embed_providers::ProviderConfig,
    context: &str,
) -> String {
    use mnem_embed_providers::EmbedError as E;
    use mnem_embed_providers::ProviderConfig as PC;

    let provider_name = match pc {
        PC::Openai(_) => "OpenAI",
        PC::Ollama(_) => "Ollama",
        PC::Onnx(_) => "ONNX",
    };
    let base_url = match pc {
        PC::Openai(c) => c.base_url.as_str(),
        PC::Ollama(c) => c.base_url.as_str(),
        // Onnx has no network endpoint; the displayed "location" is the
        // in-process model name so the error stays informative.
        PC::Onnx(c) => c.model.as_str(),
    };

    let (what, hint) = match err {
        E::Network(_) => (
            format!("{provider_name} not reachable at {base_url}"),
            match pc {
                PC::Ollama(_) => {
                    "install Ollama from https://ollama.com/download, run `ollama serve`, \
                     and `ollama pull <model>`. Or switch: \
                     `mnem config set embed.provider openai`"
                }
                PC::Openai(_) => {
                    "check your network / proxy. Or switch to local: \
                     `mnem config set embed.provider ollama`"
                }
                PC::Onnx(_) => {
                    "onnx runs in-process; a network error here is unexpected. \
                     Check the model files and rebuild if needed"
                }
            }
            .to_string(),
        ),
        E::Auth(_) => (
            format!("{provider_name} rejected the API key"),
            match pc {
                PC::Openai(c) => format!(
                    "check that ${} is exported and valid. Get a key at \
                     https://platform.openai.com/api-keys",
                    c.api_key_env
                ),
                PC::Ollama(_) => "Ollama does not require auth; this is unexpected".to_string(),
                PC::Onnx(_) => "ONNX does not require auth; this is unexpected".to_string(),
            },
        ),
        E::RateLimited(_) => (
            format!("{provider_name} rate-limited the request"),
            "back off and retry, or switch providers temporarily".to_string(),
        ),
        E::MissingApiKey { var } => (
            format!("env var ${var} is not set"),
            format!(
                "export it: `export {var}=sk-...`, or switch: \
                 `mnem config set embed.provider ollama`"
            ),
        ),
        E::BadRequest { status, .. } | E::Server { status, .. } => (
            format!("{provider_name} returned HTTP {status}"),
            match pc {
                PC::Openai(c) => format!(
                    "if model \"{}\" is new, upgrade mnem; otherwise check the \
                     provider status",
                    c.model
                ),
                PC::Ollama(c) => format!(
                    "did you run `ollama pull {}`? List local models with \
                     `ollama list`",
                    c.model
                ),
                PC::Onnx(c) => format!(
                    "onnx is in-process and shouldn't return HTTP; model=\"{}\"",
                    c.model
                ),
            },
        ),
        E::DimMismatch { expected, got } => (
            format!("{provider_name} returned dim={got}, expected {expected}"),
            "model dim changed unexpectedly; set embed.model explicitly or re-embed \
             (`mnem embed --force`)"
                .to_string(),
        ),
        E::Decode(_) | E::Config(_) => (
            format!("{provider_name} {context} failed ({err})"),
            "re-check `mnem config list`; report a bug if this persists".to_string(),
        ),
        // `EmbedError` is `#[non_exhaustive]`; keep a catch-all so a
        // future variant doesn't break the CLI compile.
        _ => (
            format!("{provider_name} {context} failed ({err})"),
            "see `mnem config list` and the provider's status page".to_string(),
        ),
    };
    format!("note: {what}; {hint}")
}

/// Upper bound on the number of content bytes fed to an embedder
/// when a node has no `summary`. 4 KiB is chosen to fit the
/// fast-path of every supported embedder's context window (the
/// smallest, MiniLM-L6, truncates at 256 tokens; 4 KiB of English
/// text comfortably exceeds that while keeping peak heap per-node
/// bounded for large corpora).
pub(super) const CONTENT_PREVIEW_CAP: usize = 4096;

/// Choose the text an embedder should see for a node. Priority:
/// summary (concise, LLM-facing); else the first
/// [`CONTENT_PREVIEW_CAP`] bytes of content decoded as UTF-8 lossy;
/// else `None` (nothing worth embedding).
///
/// `from_utf8_lossy` tolerates a mid-codepoint byte boundary at the
/// tail (substitutes `U+FFFD`), so the byte-slice is panic-safe.
pub(super) fn embed_text_of(node: &Node) -> Option<String> {
    if let Some(s) = &node.summary {
        if !s.trim().is_empty() {
            return Some(s.clone());
        }
    }
    if let Some(bytes) = &node.content {
        let cap = CONTENT_PREVIEW_CAP.min(bytes.len());
        // Slice to a char boundary; avoids splitting a multibyte scalar.
        let head = &bytes[..cap];
        let s = String::from_utf8_lossy(head).into_owned();
        if !s.trim().is_empty() {
            return Some(s);
        }
    }
    None
}

pub(super) fn load_index_set(
    bs: &std::sync::Arc<dyn mnem_core::store::Blockstore>,
    commit: Option<&Commit>,
) -> Result<Option<IndexSet>> {
    let Some(idx_cid) = commit.and_then(|c| c.indexes.as_ref()) else {
        return Ok(None);
    };
    let bytes = bs
        .get(idx_cid)?
        .ok_or_else(|| anyhow!("IndexSet block {idx_cid} missing"))?;
    Ok(Some(from_canonical_bytes(&bytes)?))
}

// ================================================================
// init
// ================================================================
