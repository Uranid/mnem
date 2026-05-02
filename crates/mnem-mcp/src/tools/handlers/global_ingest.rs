//! Handler for `mnem_global_ingest` - ingest source files into the global anchor graph.
//!
//! Opens `~/.mnemglobal/.mnem/` and runs the standard ingest pipeline against it.
//! Unlike `mnem_ingest` (which operates on whatever repo the MCP server is pointed at),
//! this tool always targets the global graph regardless of server configuration.

use std::path::PathBuf;

use anyhow::{Context, Result, anyhow, bail};
use mnem_ingest::{ChunkerAuto, ChunkerKind, IngestConfig, Ingester, SourceKind, auto_chunker};
use serde_json::Value;

use crate::server::Server;

const MAX_TOKENS_CAP: u32 = 8192;
const MAX_FILE_BYTES: u64 = 32 * 1024 * 1024;

pub(in crate::tools) fn global_ingest(_server: &mut Server, args: Value) -> Result<String> {
    let global_data = dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".mnemglobal")
        .join(".mnem");

    if !global_data.is_dir() {
        return Ok(
            "mnem_global_ingest: global graph not found at ~/.mnemglobal/.mnem/. \
             Run `mnem integrate` then `mnem init` to create it.\n"
                .to_string(),
        );
    }

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
        .unwrap_or("mnem_mcp global_ingest")
        .to_string();

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

    let ner = crate::tools::ner::resolve_ner_cfg(&global_data);
    let chunker = resolve_chunker(&chunker_str, kind, max_tokens, overlap)?;
    let config = IngestConfig {
        chunker,
        ntype,
        max_tokens,
        overlap,
        ner,
    };
    let ing = Ingester::new(config);

    let repo = Server::open_repo_at(&global_data)?;
    let mut tx = repo.start_transaction();
    let result = ing.ingest(&mut tx, &bytes, kind)?;
    let new_repo = tx.commit(&agent_id, &message)?;
    let commit_cid = new_repo
        .view()
        .heads
        .first()
        .map_or_else(|| "<none>".to_string(), ToString::to_string);

    #[cfg(feature = "summarize")]
    let embed_count = embed_global_nodes(&global_data, &new_repo, &agent_id);

    let mut out = String::new();
    out.push_str("mnem_global_ingest: ok\n");
    out.push_str(&format!("  path:           {display_path}\n"));
    out.push_str(&format!("  source_kind:    {kind:?}\n"));
    out.push_str(&format!("  op_id:          {}\n", new_repo.op_id()));
    out.push_str(&format!("  commit_cid:     {commit_cid}\n"));
    out.push_str(&format!("  node_count:     {}\n", result.node_count));
    out.push_str(&format!("  chunk_count:    {}\n", result.chunk_count));
    out.push_str(&format!("  entity_count:   {}\n", result.entity_count));
    out.push_str(&format!("  relation_count: {}\n", result.relation_count));
    out.push_str(&format!("  elapsed_ms:     {}\n", result.elapsed_ms));
    #[cfg(feature = "summarize")]
    out.push_str(&format!("  embed_count:    {embed_count}\n"));
    Ok(out)
}

#[cfg(feature = "summarize")]
fn embed_global_nodes(
    global_data: &std::path::Path,
    repo: &mnem_core::repo::ReadonlyRepo,
    agent_id: &str,
) -> usize {
    use mnem_core::codec::from_canonical_bytes;
    use mnem_core::id::Cid;
    use mnem_core::objects::Node;
    use mnem_core::prolly::Cursor;

    let Some(pc) = crate::tools::embed::resolve_embed_cfg(global_data) else {
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
        let opts = mnem_core::repo::CommitOptions::new(agent_id, "mnem_global_ingest: embed nodes");
        let _ = tx.commit_opts(opts);
    }

    count
}

fn resolve_chunker(
    choice: &str,
    kind: SourceKind,
    max_tokens: u32,
    overlap: u32,
) -> Result<ChunkerKind> {
    Ok(match choice.to_ascii_lowercase().as_str() {
        "auto" => auto_chunker(
            kind,
            ChunkerAuto {
                max_tokens: Some(max_tokens),
                overlap: Some(overlap),
                max_messages: None,
            },
        ),
        "paragraph" => ChunkerKind::Paragraph,
        "recursive" => ChunkerKind::Recursive {
            max_tokens,
            overlap,
        },
        "session" => ChunkerKind::Session { max_messages: 10 },
        other => bail!("'chunker' must be one of auto|session|paragraph|recursive; got '{other}'"),
    })
}
