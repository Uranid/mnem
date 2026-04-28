//! Handler for the `mnem_vector_search` MCP tool.
//!
//! Extracted from `tools.rs` in R3; body unchanged.

use super::super::preview_str;
use crate::server::Server;
use anyhow::{Result, anyhow};
use mnem_core::index::VectorIndex;
use serde_json::Value;

// ============================================================
// mnem_vector_search
// ============================================================

pub(in crate::tools) fn vector_search(server: &mut Server, args: Value) -> Result<String> {
    let model = args
        .get("model")
        .and_then(Value::as_str)
        .ok_or_else(|| anyhow!("missing 'model'"))?
        .to_string();
    let vec_vals = args
        .get("vector")
        .and_then(Value::as_array)
        .ok_or_else(|| anyhow!("missing 'vector' array"))?;
    let mut vector: Vec<f32> = Vec::with_capacity(vec_vals.len());
    for v in vec_vals {
        let f = v
            .as_f64()
            .ok_or_else(|| anyhow!("vector element is not a number"))?;
        vector.push(f as f32);
    }
    let k = args.get("k").and_then(Value::as_u64).unwrap_or(10) as usize;

    let repo = server.load_repo()?;
    let idx = repo.build_vector_index(&model)?;

    // audit-2026-04-25 P2-5: replace the silent no-op (which used to
    // print "0 hit(s) over 0 vec(s)") with explicit, distinguishable
    // diagnostics so callers can tell which branch they hit.
    if idx.len() == 0 {
        return Ok(format!(
            "mnem_vector_search: model={model:?} -- no nodes carry an embedding for this model in the current commit.\n\
             Hints: (a) ingest at least one node with `mnem add node -s ... ` while embed.provider+embed.model are configured, or\n\
             (b) backfill an existing repo with `mnem embed`."
        ));
    }
    if idx.dim() as usize != vector.len() {
        return Ok(format!(
            "mnem_vector_search: model={model:?} -- query-vector dim mismatch (index dim={}, query dim={}).\n\
             The model is correct but the supplied vector does not match its dimensionality.",
            idx.dim(),
            vector.len()
        ));
    }
    let hits = idx.search(&vector, k)?;

    let mut out = String::new();
    out.push_str(&format!(
        "mnem_vector_search: model={:?} dim={} {} hit(s) over {} vec(s)\n",
        model,
        idx.dim(),
        hits.len(),
        idx.len()
    ));
    for (i, h) in hits.iter().enumerate() {
        let node_opt = repo.lookup_node(&h.node_id)?;
        let (ntype, summary) = match node_opt {
            Some(ref n) => (n.ntype.clone(), n.summary.clone()),
            None => ("<missing>".to_string(), None),
        };
        out.push_str(&format!(
            "  [{i}] score={:.4} id={} {}\n",
            h.score,
            h.node_id.to_uuid_string(),
            ntype,
        ));
        if let Some(s) = summary {
            out.push_str(&format!("        summary: {}\n", preview_str(&s)));
        }
    }
    Ok(out)
}
