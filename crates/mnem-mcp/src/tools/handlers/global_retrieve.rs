//! Handler for `mnem_global_retrieve` - semantic search on the global anchor graph.
//!
//! Opens `~/.mnemglobal/.mnem/` and runs the standard retriever pipeline against it.
//! Results are ranked by score. Unlike `mnem_retrieve` (which operates on whatever
//! repo the MCP server is pointed at), this tool always targets the global graph
//! regardless of server configuration.

use crate::server::Server;
use anyhow::Result;
use serde_json::Value;
use std::path::PathBuf;

// ---------- handler ----------

pub(in crate::tools) fn global_retrieve(server: &mut Server, args: Value) -> Result<String> {
    let global_dir = dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".mnemglobal");

    let global_data = global_dir.join(".mnem");
    if !global_data.is_dir() {
        return Ok(
            "mnem_global_retrieve: global graph not found at ~/.mnemglobal/.mnem/. \
             Run `mnem integrate` then `mnem init` to create it.\n"
                .to_string(),
        );
    }

    // Parse args.
    let text_arg = args.get("text").and_then(Value::as_str).map(str::to_string);
    let vector_arg = args.get("vector").and_then(Value::as_object).cloned();
    let limit = args
        .get("limit")
        .and_then(Value::as_u64)
        .map(|n| (n as usize).min(super::super::MAX_RETRIEVE_LIMIT))
        .unwrap_or(10);
    let budget = args
        .get("token_budget")
        .and_then(Value::as_u64)
        .map(|n| n.min(u64::from(u32::MAX)) as u32);

    // Parse pre-computed vector or auto-embed using the global graph's config.
    let opt_vec: Option<(String, Vec<f32>)> = if let Some(vec_obj) = vector_arg {
        let model = vec_obj
            .get("model")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string();
        let vals = vec_obj
            .get("values")
            .and_then(Value::as_array)
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_f64().map(|f| f as f32))
                    .collect::<Vec<f32>>()
            })
            .unwrap_or_default();
        if !model.is_empty() && !vals.is_empty() {
            Some((model, vals))
        } else {
            None
        }
    } else {
        #[cfg(feature = "summarize")]
        {
            if let Some(ref text) = text_arg {
                if let Some(cfg) = crate::tools::embed::resolve_embed_cfg(&global_data)
                    && let Ok(embedder) = mnem_embed_providers::open(&cfg)
                    && let Ok(vec) = embedder.embed(text)
                {
                    Some((embedder.model().to_string(), vec))
                } else {
                    None
                }
            } else {
                None
            }
        }
        #[cfg(not(feature = "summarize"))]
        None
    };

    // Open the global graph and retrieve. If the server is already
    // pointing at the global path (the default `mnem integrate` config),
    // reuse the cached connection — opening the same redb file twice from
    // the same process causes "Database already open" lock failures.
    let repo = if server.repo_path() == global_data {
        match server.load_repo() {
            Ok(r) => r,
            Err(e) => {
                eprintln!("(mnem_global_retrieve: cannot load global graph: {e})");
                return Ok(format!(
                    "mnem_global_retrieve: error opening global graph: {e}\n"
                ));
            }
        }
    } else {
        match Server::open_repo_at(&global_data) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("(mnem_global_retrieve: cannot open global graph: {e})");
                return Ok(format!(
                    "mnem_global_retrieve: error opening global graph: {e}\n"
                ));
            }
        }
    };

    let mut r = repo.retrieve().limit(limit);
    if let Some(ref text) = text_arg {
        r = r.query_text(text.clone());
    }
    if let Some((ref model, ref vec)) = opt_vec {
        r = r.vector(model.clone(), vec.clone());
    }
    if let Some(b) = budget {
        r = r.token_budget(b);
    }

    let result = match r.execute() {
        Ok(res) => res,
        Err(e) => {
            let msg = format!("{e:#}");
            if msg.contains("no filters or rankers configured") {
                return Ok("mnem_global_retrieve: 0 item(s)\n".to_string());
            }
            return Ok(format!("mnem_global_retrieve: error: {e}\n"));
        }
    };

    if result.items.is_empty() {
        return Ok("mnem_global_retrieve: 0 item(s)\n".to_string());
    }

    let mut out = String::new();
    out.push_str(&format!(
        "mnem_global_retrieve: {} item(s)\n",
        result.items.len(),
    ));
    for (i, item) in result.items.iter().enumerate() {
        out.push_str(&format!(
            "  [{i}] score={:.4} tokens={} id={} {}\n",
            item.score,
            item.tokens,
            item.node.id.to_uuid_string(),
            item.node.ntype,
        ));
        for line in item.rendered.lines() {
            out.push_str(&format!("        {line}\n"));
        }
    }
    Ok(out)
}
