//! Handler for the `mnem_retrieve` MCP tool.
//!
//! Extracted from `tools.rs` in R3; body unchanged.

use super::super::{MAX_RERANK_TOP_K, MAX_RETRIEVE_LIMIT, MAX_VECTOR_CAP};
use crate::server::Server;
use anyhow::{Result, anyhow, bail};
use mnem_core::codec::json_to_ipld;
use mnem_core::index::PropPredicate;
use serde_json::Value;

// ============================================================
// mnem_retrieve
// ============================================================

pub(in crate::tools) fn retrieve(server: &mut Server, args: Value) -> Result<String> {
    // `label` gated behind `MNEM_BENCH`. Off by default: the filter is
    // silently ignored so retrieve runs unscoped (parity with
    // GET/POST /v1/retrieve in mnem-http).
    let allow_labels = server.allow_labels;
    let repo = server.load_repo()?;
    let mut r = repo.retrieve();

    if allow_labels && let Some(label) = args.get("label").and_then(Value::as_str) {
        r = r.label(label);
    }
    if let Some(Value::Object(map)) = args.get("where")
        && let Some((k, v)) = map.iter().next()
    {
        let ipld = json_to_ipld(v)?;
        r = r.where_prop(k, PropPredicate::Eq(ipld));
    }
    let text_arg = args.get("text").and_then(Value::as_str).map(str::to_string);
    let vector_arg = args.get("vector").and_then(Value::as_object).cloned();

    if let Some(ref text) = text_arg {
        r = r.query_text(text.clone());
    }
    if let Some(vec_obj) = vector_arg {
        let model = vec_obj
            .get("model")
            .and_then(Value::as_str)
            .ok_or_else(|| anyhow!("vector.model missing"))?
            .to_string();
        let vals = vec_obj
            .get("values")
            .and_then(Value::as_array)
            .ok_or_else(|| anyhow!("vector.values missing"))?;
        let mut vector: Vec<f32> = Vec::with_capacity(vals.len());
        for v in vals {
            let f = v
                .as_f64()
                .ok_or_else(|| anyhow!("vector.values element is not a number"))?;
            vector.push(f as f32);
        }
        r = r.vector(model, vector);
    } else if let Some(ref text) = text_arg {
        // Path A audit fix (2026-04-26): when caller passes `text`
        // without an explicit `vector`, try to resolve an embedder
        // (env → per-repo config.toml → bundled MiniLM) and embed the
        // text in-process. With `--features bundled-embedder`, this
        // makes `mnem_retrieve` semantic by default - no Ollama, no
        // explicit vector arg, no `MNEM_EMBED_*` setup. Without the
        // feature and without env / config, this is a silent no-op
        // (current behaviour preserved): the dense lane is skipped
        // and the existing graceful-empty-success path handles the
        // result.
        //
        // Failures (provider open error, embed error) are also
        // silent: the dense lane is skipped, structured filters
        // continue to work. Surfacing the error here would regress
        // the no-config-empty-repo case.
        #[cfg(feature = "summarize")]
        {
            if let Some(emb_cfg) = crate::tools::embed::resolve_embed_cfg(server.repo_path())
                && let Ok(embedder) = mnem_embed_providers::open(&emb_cfg)
                && let Ok(vec) = embedder.embed(text)
            {
                let model = embedder.model().to_string();
                r = r.vector(model, vec);
            }
        }
    }
    if let Some(budget) = args.get("token_budget").and_then(Value::as_u64) {
        r = r.token_budget(budget.min(u64::from(u32::MAX)) as u32);
    }
    // Input clamps mirror mnem-http: MCP tool args are as untrusted
    // as an HTTP body. Without a ceiling, a caller can send
    // `limit=18446744073709551615` and trigger whatever the
    // downstream BruteForce vector search allocates. See
    // `mnem_http::handlers::{MAX_RETRIEVE_LIMIT, MAX_VECTOR_CAP,
    // MAX_RERANK_TOP_K}` for the same constants.
    if let Some(limit) = args.get("limit").and_then(Value::as_u64) {
        let limit = limit as usize;
        if limit > MAX_RETRIEVE_LIMIT {
            bail!("limit={limit} exceeds max of {MAX_RETRIEVE_LIMIT}");
        }
        r = r.limit(limit);
    }
    if let Some(cap) = args.get("vector_cap").and_then(Value::as_u64) {
        let cap = cap as usize;
        if cap > MAX_VECTOR_CAP {
            bail!("vector_cap={cap} exceeds max of {MAX_VECTOR_CAP}");
        }
        r = r.vector_cap(cap);
    }
    if let Some(k) = args.get("rerank_top_k").and_then(Value::as_u64) {
        let k = k as usize;
        if k > MAX_RERANK_TOP_K {
            bail!("rerank_top_k={k} exceeds max of {MAX_RERANK_TOP_K}");
        }
        r = r.rerank_top_k(k);
    }
    // Experiment E1: community-filter knobs. Accepted at the MCP
    // surface for HTTP<->MCP parity, but a live community lookup is
    // not wired yet (future work: server-level cache keyed by commit
    // CID). When no lookup is installed, passing
    // `community_filter=true` is a byte-exact pass-through per the
    // retriever's flag-off contract.
    let _community_filter = args
        .get("community_filter")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let _community_min_coverage = args
        .get("community_min_coverage")
        .and_then(Value::as_f64)
        .map(|v| v as f32);
    if let Some(strategy) = args.get("fusion").and_then(Value::as_str) {
        let fusion = match strategy {
            "rrf" => mnem_core::retrieve::FusionStrategy::Rrf,
            "convex_min_max" => mnem_core::retrieve::FusionStrategy::ConvexMinMax,
            other => bail!("fusion must be one of 'convex_min_max' or 'rrf'; got '{other}'"),
        };
        r = r.fusion(fusion);
    }
    // Graph-expand block: the only *required* knob is `graph_expand`
    // (the global frontier cap). Every other graph_* knob takes the
    // retriever default when absent. Keep the same semantics as
    // POST /v1/retrieve for MCP<->HTTP parity.
    if let Some(max_expand) = args.get("graph_expand").and_then(Value::as_u64) {
        let mut cfg = mnem_core::retrieve::GraphExpand {
            max_expand: max_expand as usize,
            decay: args
                .get("graph_decay")
                .and_then(Value::as_f64)
                .map_or(mnem_core::retrieve::GraphExpand::DEFAULT_DECAY, |d| {
                    d as f32
                }),
            etype_filter: args
                .get("graph_etype")
                .and_then(|v| v.as_array())
                .map(|arr| {
                    arr.iter()
                        .filter_map(|e| e.as_str().map(ToString::to_string))
                        .collect()
                }),
            ..Default::default()
        };
        if let Some(depth) = args.get("graph_depth").and_then(Value::as_u64) {
            cfg = cfg.with_depth(depth as usize);
        }
        if let Some(cap) = args.get("graph_max_per_seed").and_then(Value::as_u64) {
            cfg = cfg.with_max_per_seed(cap as usize);
        }
        // E2: PPR mode dispatch. Parity with POST /v1/retrieve.
        if let Some(mode) = args.get("graph_mode").and_then(Value::as_str)
            && mode == "ppr"
        {
            let damping = args
                .get("ppr_damping")
                .and_then(Value::as_f64)
                .map_or(mnem_core::ppr::DEFAULT_DAMPING, |d| d as f32);
            let iter = args
                .get("ppr_iter")
                .and_then(Value::as_u64)
                .map_or(mnem_core::ppr::DEFAULT_MAX_ITER, |n| n as u32);
            cfg = cfg.with_ppr(damping, iter, mnem_core::ppr::DEFAULT_EPS);
        }
        r = r.with_graph_expand(cfg);
    }

    // audit-2026-04-25 C3-1 (Cycle-3, partial): a fresh repo with no
    // `[embed]` config and no `where`/`label`/`vector` filters
    // produces `RetrievalEmpty` from the retriever pipeline. The
    // error message is accurate but Pass-2 found that MCP clients
    // see the bounce and assume the tool is broken. Convert that
    // specific `RetrievalEmpty` outcome into an empty-success
    // response with a human-readable hint pointing at the
    // remediation (`embed.provider` config or `where`/`label`
    // filters). All other retrieval failures still propagate as
    // tool-level errors.
    //
    // FOLLOW-UP (deferred from one-pass scope): also wire a default
    // mock embedder so a query+limit-only call can return real hits
    // on a fresh repo. That is risk-bearing surgery (touches
    // `Server` lane init); keeping the conservative no-config
    // empty-success here, gated by the existing
    // `mnem.v1.retrieve.empty` shape, until the embedder default is
    // ready.
    let result = match r.execute() {
        Ok(r) => r,
        Err(e) => {
            // Match by message; the retriever surfaces
            // `mnem_core::error::RepoError::RetrievalEmpty`
            // through anyhow with the canonical text "retrieve:
            // no filters or rankers configured.". Detect the
            // prefix to avoid swallowing other retrieve errors.
            let msg = format!("{e:#}");
            if msg.contains("no filters or rankers configured") {
                let mut out = String::new();
                out.push_str("mnem_retrieve: 0 item(s), 0/unlimited tokens, 0 dropped, 0 candidates\n");
                out.push_str(
                    "  note: no embedder configured and no `where`/`label`/`vector` filter passed; \
                     returned empty so MCP clients are not bounced on a fresh repo. \
                     For real hits, either: (a) configure [embed] in <repo>/config.toml or \
                     `mnem config set embed.provider ollama && mnem config set embed.model nomic-embed-text`, \
                     or (b) pass `where: {\"key\": \"value\"}` / `label: \"<NodeType>\"` for a pure filter query.\n",
                );
                return Ok(out);
            }
            return Err(e.into());
        }
    };

    let mut out = String::new();
    out.push_str(&format!(
        "mnem_retrieve: {} item(s), {}/{} tokens, {} dropped, {} candidates\n",
        result.items.len(),
        result.tokens_used,
        if result.tokens_budget == u32::MAX {
            "unlimited".to_string()
        } else {
            result.tokens_budget.to_string()
        },
        result.dropped,
        result.candidates_seen,
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
