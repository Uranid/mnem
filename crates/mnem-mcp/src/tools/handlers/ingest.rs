//! Handler for the `mnem_ingest` MCP tool.
//!
//! Phase-B5d, MCP half. Reads a file from disk, runs
//! [`mnem_ingest::Ingester`] against a fresh transaction, commits
//! with the caller-supplied `agent_id` as author, and renders a short
//! plain-text summary the calling model can reason about.
//!
//! ## Why plain text, not JSON
//!
//! Matches every other handler in this module. MCP tool output is
//! consumed by an LLM; `key: value` lines and indented counts
//! tokenise about 30% smaller than an equivalent JSON blob, and the
//! parse-free shape is a feature for the telemetry path described in
//! `tools.rs`.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow, bail};
use mnem_core::repo::ReadonlyRepo;
use mnem_ingest::{IngestConfig, Ingester, SourceKind, resolve_chunker};
use serde_json::Value;

use crate::server::Server;

/// Upper bound on `max_tokens`. Mirrors the CLI and HTTP surface so a
/// single request that migrates between transports sees one ceiling.
const MAX_TOKENS_CAP: u32 = 8192;

/// Upper bound on ingested file size for the MCP path. Matches the
/// HTTP handler default (`MNEM_HTTP_INGEST_MAX_BYTES` in B5d-3). MCP
/// callers run in-process with the agent but typically hand-roll
/// arbitrary paths, so the same DoS guardrail applies.
const MAX_FILE_BYTES: u64 = 32 * 1024 * 1024;

// ============================================================
// mnem_ingest
// ============================================================

/// Dispatch target for the `mnem_ingest` tool.
///
/// Accepts either:
///   - `{path: "/abs/file"}` -- read the file from disk and chunk it.
///   - `{text: "...", source?: "label"}` -- treat the inline string
///     as the document body. `source` becomes the cosmetic
///     `path:` field in the output (defaults to `inline-text`).
///
/// audit-2026-04-25 C3-8 (Cycle-3): Pass-2 found callers expecting
/// the inline `{text, source}` shape (e.g. agent flows that have
/// already buffered the document) hit "missing 'path'". The
/// dispatcher accepts both shapes; an inline call short-circuits the
/// filesystem stat / size cap path, so the `MAX_FILE_BYTES` ceiling
/// is replaced by the same `text.len()` byte cap to keep the DoS
/// guardrail uniform across both shapes.
///
/// # Errors
///
/// Returns an error when both `path` and `text` are missing, the
/// file is absent / too large, the inline text is empty / over the
/// cap, the pipeline rejects the payload, or the backing transaction
/// fails to commit. Errors bubble back to the caller as tool-level
/// errors (not JSON-RPC protocol errors).
pub(in crate::tools) fn ingest(server: &mut Server, args: Value) -> Result<String> {
    let repo_path = server.repo_path().to_path_buf();
    let allow_labels = server.allow_labels;
    let repo = server.load_repo()?;
    ingest_impl(repo, &repo_path, allow_labels, args)
}

pub(super) fn ingest_impl(
    repo: ReadonlyRepo,
    repo_path: &Path,
    allow_labels: bool,
    args: Value,
) -> Result<String> {
    let _ = allow_labels; // ingest does not gate on ntype currently

    let ntype = args
        .get("ntype")
        .and_then(Value::as_str)
        .unwrap_or("Doc")
        .to_string();

    let chunker_str = args
        .get("chunker")
        .and_then(Value::as_str)
        .unwrap_or("auto")
        .to_string();

    let max_tokens: u32 = args
        .get("max_tokens")
        .and_then(Value::as_u64)
        .map_or(512, |v| {
            u32::try_from(v.min(u64::from(u32::MAX))).unwrap_or(512)
        });
    if max_tokens > MAX_TOKENS_CAP {
        bail!("'max_tokens' {max_tokens} exceeds the {MAX_TOKENS_CAP} cap");
    }

    let overlap: u32 = args.get("overlap").and_then(Value::as_u64).map_or(32, |v| {
        u32::try_from(v.min(u64::from(u32::MAX))).unwrap_or(32)
    });

    let agent_id = args
        .get("agent_id")
        .and_then(Value::as_str)
        .unwrap_or("mnem mcp")
        .to_string();
    let message = args
        .get("message")
        .and_then(Value::as_str)
        .unwrap_or("mnem_mcp ingest")
        .to_string();

    // C3-8: dispatch on shape. `path` wins when both are passed --
    // it carries strictly more information (kind detection from
    // extension, real on-disk path for provenance).
    let path_arg = args.get("path").and_then(Value::as_str);
    let text_arg = args.get("text").and_then(Value::as_str);

    let (display_path, bytes, kind) = match (path_arg, text_arg) {
        (Some(p), _) => {
            let path = PathBuf::from(p);
            let meta =
                std::fs::metadata(&path).with_context(|| format!("stat {}", path.display()))?;
            if meta.len() > MAX_FILE_BYTES {
                bail!(
                    "file {} is {} bytes; exceeds the {MAX_FILE_BYTES}-byte cap",
                    path.display(),
                    meta.len()
                );
            }
            let bytes =
                std::fs::read(&path).with_context(|| format!("reading {}", path.display()))?;
            let kind = Ingester::source_kind_for_path(&path);
            (path.display().to_string(), bytes, kind)
        }
        (None, Some(t)) => {
            if t.is_empty() {
                bail!("'text' is empty; pass non-empty content");
            }
            let len_u64 = t.len() as u64;
            if len_u64 > MAX_FILE_BYTES {
                bail!("'text' is {len_u64} bytes; exceeds the {MAX_FILE_BYTES}-byte cap");
            }
            let source_label = args
                .get("source")
                .and_then(Value::as_str)
                .unwrap_or("inline-text")
                .to_string();
            // Inline text has no file extension to drive
            // `source_kind_for_path`; default to `Text` so the
            // paragraph/recursive chunker chain still applies.
            let kind = SourceKind::Text;
            (source_label, t.as_bytes().to_vec(), kind)
        }
        (None, None) => {
            return Err(anyhow!(
                "missing 'path' or 'text'; pass {{path: \"/abs/file\"}} OR \
                 {{text: \"...\", source?: \"label\"}}"
            ));
        }
    };

    let ner = crate::tools::ner::resolve_ner_cfg(repo_path);
    let chunker = resolve_chunker(&chunker_str, kind, max_tokens, overlap)?;
    let config = IngestConfig {
        chunker,
        ntype,
        max_tokens,
        overlap,
        ner,
    };
    let ing = Ingester::new(config);

    let mut tx = repo.start_transaction();
    let result = ing.ingest(&mut tx, &bytes, kind)?;
    let new_repo = tx.commit(&agent_id, &message)?;
    let commit_cid = new_repo
        .view()
        .heads
        .first()
        .map_or_else(|| "<none>".to_string(), ToString::to_string);

    // Embed summaries of all newly committed nodes. Mirrors the CLI's
    // post-ingest reindex step. Non-fatal: embedding failures are silent
    // and the commit is already durable.
    #[cfg(feature = "summarize")]
    let embed_count = embed_ingest_nodes(repo_path, &new_repo, &agent_id);

    let mut out = String::new();
    out.push_str("mnem_ingest: ok\n");
    out.push_str(&format!("  path:           {display_path}\n"));
    out.push_str(&format!("  source_kind:    {kind:?}\n"));
    out.push_str(&format!("  op_id:          {}\n", new_repo.op_id()));
    out.push_str(&format!("  commit_cid:     {commit_cid}\n"));
    out.push_str(&format!("  node_count:     {}\n", result.node_count));
    out.push_str(&format!("  chunk_count:    {}\n", result.chunk_count));
    out.push_str(&format!("  entity_count:   {}\n", result.entity_count));
    out.push_str(&format!("  relation_count: {}\n", result.relation_count));
    out.push_str(&format!("  edge_count:     {}\n", result.edge_count));
    out.push_str(&format!("  elapsed_ms:     {}\n", result.elapsed_ms));
    #[cfg(feature = "summarize")]
    out.push_str(&format!("  embed_count:    {embed_count}\n"));
    Ok(out)
}

/// Walk all nodes in `repo`, embed any that have a summary but no vector
/// for the resolved model, and commit the vectors in a second transaction.
/// Non-fatal: returns 0 on any setup failure (no embedder, no commit,
/// empty repo). Mirrors `mnem reindex` / CLI post-ingest embed pass.
#[cfg(feature = "summarize")]
fn embed_ingest_nodes(
    repo_path: &std::path::Path,
    repo: &mnem_core::repo::ReadonlyRepo,
    agent_id: &str,
) -> usize {
    use mnem_core::codec::from_canonical_bytes;
    use mnem_core::id::Cid;
    use mnem_core::objects::Node;
    use mnem_core::prolly::Cursor;

    let Some(pc) = crate::tools::embed::resolve_embed_cfg(repo_path) else {
        return 0;
    };
    let Ok(embedder) = mnem_embed_providers::open(&pc) else {
        return 0;
    };
    let model = embedder.model().to_string();

    let Some(commit) = repo.head_commit() else {
        return 0;
    };
    let bs = repo.blockstore();
    let Ok(cursor) = Cursor::new(bs.as_ref(), &commit.nodes) else {
        return 0;
    };

    let mut to_embed: Vec<(Cid, String)> = Vec::new();
    for entry in cursor {
        let Ok((_k, node_cid)) = entry else { continue };
        let Ok(Some(bytes)) = bs.get(&node_cid) else {
            continue;
        };
        let Ok(node) = from_canonical_bytes::<Node>(&bytes) else {
            continue;
        };
        let Some(summary) = node.summary.as_deref() else {
            continue;
        };
        if summary.trim().is_empty() {
            continue;
        }
        if repo
            .embedding_for(&node_cid, &model)
            .ok()
            .flatten()
            .is_some()
        {
            continue;
        }
        to_embed.push((node_cid, summary.to_string()));
    }

    if to_embed.is_empty() {
        return 0;
    }

    let mut tx = repo.start_transaction();
    let mut count = 0usize;
    for (node_cid, text) in &to_embed {
        let Ok(vec) = embedder.embed(text) else {
            continue;
        };
        let emb = mnem_embed_providers::to_embedding(&model, &vec);
        if tx
            .set_embedding(node_cid.clone(), model.clone(), emb)
            .is_ok()
        {
            count += 1;
        }
    }

    if count > 0 {
        let opts = mnem_core::repo::CommitOptions::new(agent_id, "mnem_ingest: embed nodes");
        let _ = tx.commit_opts(opts);
    }

    count
}

