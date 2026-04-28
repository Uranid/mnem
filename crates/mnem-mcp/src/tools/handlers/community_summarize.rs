//! Handler for the `mnem_community_summarize` MCP tool.
//!
//! E4 T2, MCP half. Mirrors the HTTP `POST /v1/retrieve?summarize=true`
//! hook but runs on an explicit caller-supplied node set instead of
//! a retrieved top-M: the caller already has node UUIDs (e.g. a
//! Leiden community from E1, or a hand-curated subgraph) and wants
//! the centroid + MMR extract of the nodes' summaries.
//!
//! ## Scope
//!
//! - Extractive only. Picks existing sentences; no LLM rewrite.
//! - No BM25 / no sparse lane .
//! - Degree-centrality fallback is a uniform `1.0` per sentence, matching
//!   the HTTP hook. E2's PPR vector slots in here unchanged once merged.
//!
//! ## Embedder resolution
//!
//! Same precedence as the CLI's `config::resolve_embedder` (without
//! taking a dep on `mnem-cli`):
//!   1. `MNEM_EMBED_PROVIDER` + `MNEM_EMBED_MODEL` (+ optional
//!      `MNEM_EMBED_API_KEY_ENV`, `MNEM_EMBED_BASE_URL`, `MNEM_EMBED_DIM`)
//!      env vars.
//!   2. The `[embed]` section in `<repo_path>/config.toml`.
//!
//! On miss (no env, no config, parse fails) the tool returns a
//! tool-level error - parity with how the HTTP surface emits
//! `summarize_skipped`, except at the MCP boundary we propagate the
//! reason via `anyhow` so the client sees it as an `isError` content
//! block.

use anyhow::{Context, Result, anyhow, bail};
use mnem_core::id::NodeId;
use serde_json::Value;

use crate::server::Server;
use crate::tools::embed::resolve_embed_cfg;

/// Upper bound on `k` accepted on `mnem_community_summarize`. Mirrors
/// `MAX_RETRIEVE_LIMIT` so a single request that migrates between
/// MCP and HTTP sees one ceiling.
const MAX_SUMMARIZE_K: usize = 1_000;

/// Upper bound on how many `node_ids` a caller may submit in one
/// shot. 10k nodes is already well past the "community" regime the
/// tool is designed for; anything larger wants a batched caller
/// flow, not a single RPC.
const MAX_SUMMARIZE_NODES: usize = 10_000;

// ============================================================
// mnem_community_summarize
// ============================================================

/// Dispatch target for the `mnem_community_summarize` tool.
///
/// # Errors
///
/// Returns an error when `node_ids` is missing / empty / oversized,
/// a referenced UUID fails to parse, no node with a given id exists,
/// no `[embed]` provider can be resolved, or the summarizer itself
/// fails (dimension mismatch, provider transport error). Errors
/// bubble back as tool-level errors via `anyhow`.
pub(in crate::tools) fn community_summarize(server: &mut Server, args: Value) -> Result<String> {
    // ---------- parse args ----------
    let ids_val = args
        .get("node_ids")
        .and_then(Value::as_array)
        .ok_or_else(|| anyhow!("missing 'node_ids' array"))?;
    if ids_val.is_empty() {
        bail!("'node_ids' must not be empty");
    }
    if ids_val.len() > MAX_SUMMARIZE_NODES {
        bail!(
            "'node_ids' has {} entries; exceeds the cap of {MAX_SUMMARIZE_NODES}",
            ids_val.len()
        );
    }
    let mut ids: Vec<NodeId> = Vec::with_capacity(ids_val.len());
    for (i, v) in ids_val.iter().enumerate() {
        let s = v
            .as_str()
            .ok_or_else(|| anyhow!("'node_ids[{i}]' is not a string"))?;
        let id = NodeId::parse_uuid(s)
            .with_context(|| format!("invalid UUID at node_ids[{i}]: {s:?}"))?;
        ids.push(id);
    }

    let query = args
        .get("query")
        .and_then(Value::as_str)
        .map(str::to_string);

    // `k` default matches the HTTP hook (3).
    let k = args
        .get("k")
        .and_then(Value::as_u64)
        .map_or(3_usize, |v| v as usize);
    if k > MAX_SUMMARIZE_K {
        bail!("k={k} exceeds max of {MAX_SUMMARIZE_K}");
    }

    // `mmr_lambda` default matches the HTTP hook and the spec.
    // Clamping to [0,1] is the callee's job; we forward verbatim.
    let mmr_lambda = args
        .get("mmr_lambda")
        .and_then(Value::as_f64)
        .map_or(0.5_f32, |v| v as f32);

    // ---------- look up node summaries ----------
    let repo = server.load_repo()?;
    let mut sentences: Vec<String> = Vec::with_capacity(ids.len());
    let mut missing: Vec<String> = Vec::new();
    let mut no_summary: Vec<String> = Vec::new();
    for id in &ids {
        match repo.lookup_node(id)? {
            Some(node) => match node.summary {
                Some(s) if !s.is_empty() => sentences.push(s),
                _ => no_summary.push(id.to_uuid_string()),
            },
            None => missing.push(id.to_uuid_string()),
        }
    }

    // ---------- resolve embedder ----------
    let embed_cfg = resolve_embed_cfg(server.repo_path()).ok_or_else(|| {
        anyhow!(
            "no embed provider resolved: set MNEM_EMBED_PROVIDER + MNEM_EMBED_MODEL, \
                 or add an [embed] section to <repo>/config.toml"
        )
    })?;
    let embedder = mnem_embed_providers::open(&embed_cfg)
        .map_err(|e| anyhow!("embed provider open failed: {e}"))?;

    // Optional query vector. If the caller passed a query but the
    // embedder fails to embed it, fall back to no-query mode (the
    // HTTP hook simply omits the query; we mirror that posture).
    let query_embed: Option<Vec<f32>> = match query.as_deref() {
        Some(q) if !q.is_empty() => embedder.embed(q).ok(),
        _ => None,
    };

    // ---------- invoke summarizer ----------
    //
    // Degree-centrality fallback: uniform 1.0. E2's PPR vector slots
    // in here unchanged when it lands; the `&dyn Fn(usize) -> f32`
    // shape is the seam.
    let centrality = |_: usize| 1.0_f32;
    let summary = mnem_graphrag::summarize_community(
        &sentences,
        embedder.as_ref(),
        query_embed.as_deref(),
        &centrality,
        k,
        mmr_lambda,
    )
    .map_err(|e| anyhow!("summarize_community failed: {e}"))?;

    // ---------- render (plain-text, LLM-friendly) ----------
    //
    // Matches the other handlers' style: a single header line with
    // counts + a bulleted list with scores truncated to 4 decimals.
    // Anything the model needs to reason about is in the leading
    // line so token-trimmed transports still see the headline.
    let mut out = String::new();
    out.push_str(&format!(
        "mnem_community_summarize: {} sentence(s) picked from {} node(s) \
         ({} missing, {} without summary), k={k}, lambda={mmr_lambda}\n",
        summary.sentences.len(),
        ids.len(),
        missing.len(),
        no_summary.len(),
    ));
    for (i, (s, score)) in summary
        .sentences
        .iter()
        .zip(summary.scores.iter())
        .enumerate()
    {
        out.push_str(&format!("  [{i}] score={score:.4} {s}\n"));
    }
    if !missing.is_empty() {
        out.push_str(&format!(
            "  missing:    {}\n",
            missing
                .iter()
                .take(5)
                .cloned()
                .collect::<Vec<_>>()
                .join(", ")
        ));
    }
    if !no_summary.is_empty() {
        out.push_str(&format!(
            "  no_summary: {}\n",
            no_summary
                .iter()
                .take(5)
                .cloned()
                .collect::<Vec<_>>()
                .join(", ")
        ));
    }
    Ok(out)
}

// `resolve_embed_cfg` is now shared in `crate::tools::embed`. Path A
// audit fix (2026-04-26) hoisted it so `mnem_retrieve` can reuse the
// same precedence chain (env → per-repo config.toml → bundled MiniLM).
//
// `Server::repo_path` is private; this handler needs read-only access
// to it to locate `<repo>/config.toml`. We go through the
// `pub(crate)` accessor added in the original community_summarize
// commit (see `crate::server::Server::repo_path`).

#[cfg(test)]
mod tests {
    use super::*;
    use crate::server::Server;
    use serde_json::json;
    use tempfile::TempDir;

    fn mk_server_with_repo() -> (Server, TempDir) {
        let td = tempfile::tempdir().expect("tempdir");
        let s = Server::new(td.path().to_path_buf());
        (s, td)
    }

    #[test]
    fn rejects_missing_node_ids() {
        let (mut s, _td) = mk_server_with_repo();
        let err = community_summarize(&mut s, json!({})).expect_err("missing node_ids must error");
        assert!(format!("{err:#}").contains("node_ids"));
    }

    #[test]
    fn rejects_empty_node_ids() {
        let (mut s, _td) = mk_server_with_repo();
        let err = community_summarize(&mut s, json!({ "node_ids": [] }))
            .expect_err("empty node_ids must error");
        assert!(format!("{err:#}").contains("must not be empty"));
    }

    #[test]
    fn rejects_oversized_k() {
        let (mut s, _td) = mk_server_with_repo();
        // Use a well-formed UUID so we get past UUID parsing and hit
        // the k clamp.
        let id = mnem_core::id::NodeId::new_v7().to_uuid_string();
        let err = community_summarize(&mut s, json!({ "node_ids": [id], "k": 10_000_u64 }))
            .expect_err("oversized k must error");
        let msg = format!("{err:#}");
        assert!(
            msg.contains("k=") && msg.contains("exceeds max"),
            "got: {msg}"
        );
    }

    #[test]
    fn rejects_invalid_uuid() {
        let (mut s, _td) = mk_server_with_repo();
        let err = community_summarize(&mut s, json!({ "node_ids": ["not-a-uuid"] }))
            .expect_err("invalid UUID must error");
        assert!(format!("{err:#}").contains("invalid UUID"));
    }
}
