//! Handler for the `mnem_resolve_or_create` MCP tool.

use crate::server::Server;
use anyhow::{Result, anyhow};
use mnem_core::codec::json_to_ipld;
use mnem_core::objects::Node;
use serde_json::Value;

// ============================================================
// mnem_resolve_or_create
// ============================================================

pub(in crate::tools) fn resolve_or_create(server: &mut Server, args: Value) -> Result<String> {
    // audit-2026-04-25 C3-10 (Cycle-3): accept the natural
    // `{name, kind}` shape as an alias for `{prop_name, label}`. The
    // canonical schema lets callers pick which property to anchor on
    // (e.g. `email`, `id`, `slug`); `{name, kind}` collapses to the
    // common case where the anchor property is `name` and the
    // discriminator is the node label. We resolve the aliases first
    // and then fall through to the existing field readers, so older
    // callers that pass `{prop_name, value, label}` keep working.
    let name_alias = args.get("name").and_then(Value::as_str).map(str::to_string);
    let kind_alias = args.get("kind").and_then(Value::as_str).map(str::to_string);

    // `label` gated behind `MNEM_BENCH`. When the gate is off, every
    // find-or-create runs against `Node::DEFAULT_NTYPE`: that is the
    // correct behaviour for single-tenant graphs (there is only one
    // label so `(label, prop_name) -> id` collapses to
    // `(prop_name) -> id`). When on, the caller's label is honoured.
    let allow_labels = server.allow_labels;
    let label = if allow_labels {
        // Prefer explicit `label`; fall back to `kind` alias.
        let raw = args
            .get("label")
            .and_then(Value::as_str)
            .filter(|s| !s.trim().is_empty())
            .map(str::to_string)
            .or(kind_alias.clone());
        raw.ok_or_else(|| anyhow!("missing 'label' (or 'kind')"))?
    } else {
        Node::DEFAULT_NTYPE.to_string()
    };
    // C3-10: when caller used the `name` alias, default `prop_name`
    // to "name" (the conventional anchor key) and use the alias as
    // the value. Callers that pass `prop_name` + `value` directly
    // keep their explicit shape.
    let prop_name = match args.get("prop_name").and_then(Value::as_str) {
        Some(p) => p.to_string(),
        None if name_alias.is_some() => "name".to_string(),
        None => {
            return Err(anyhow!(
                "missing 'prop_name' (or pass the {{name, kind}} shape: \
                 `name` becomes the value of the `name` property and \
                 `kind` becomes the label)"
            ));
        }
    };
    let value_json = match args.get("value") {
        Some(v) => v.clone(),
        None => match &name_alias {
            Some(n) => Value::String(n.clone()),
            None => return Err(anyhow!("missing 'value' (or 'name')")),
        },
    };
    let value = json_to_ipld(&value_json)?;
    // C3-10: default `agent_id` to "mnem mcp" so the friendly
    // `{name, kind}` shape works end-to-end without forcing the
    // caller to thread an extra field. Mirrors the default in
    // `mnem_ingest`. Callers that pass an explicit `agent_id` keep
    // overriding it.
    let agent_id = args
        .get("agent_id")
        .and_then(Value::as_str)
        .unwrap_or("mnem mcp")
        .to_string();
    let extra_props = args
        .get("extra_props")
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default();

    // `global: true` -> resolve/create the same entity in
    // ~/.mnemglobal/.mnem/ and stamp its UUID as `_global_anchor` on
    // the local node. Best-effort: if the global graph is unreachable
    // (not yet initialised, missing dir, store error), the local commit
    // proceeds normally and a stderr note is emitted instead of failing.
    let want_global = args.get("global").and_then(Value::as_bool).unwrap_or(false);

    let global_anchor_uuid: Option<String> = if want_global {
        try_stamp_global(server, &label, &prop_name, &value, &agent_id)
    } else {
        None
    };

    let repo = server.load_repo()?;
    let mut tx = repo.start_transaction();
    let id = tx.resolve_or_create_node(&label, &prop_name, value.clone())?;

    // Always write the node explicitly so we get a CID back for
    // embedding. Including extra_props and _global_anchor as before.
    let mut node = Node::new(id, label.clone()).with_prop(prop_name.clone(), value);
    for (k, v) in &extra_props {
        node = node.with_prop(k.clone(), json_to_ipld(v)?);
    }
    if let Some(ref anchor) = global_anchor_uuid {
        use ipld_core::ipld::Ipld;
        node = node.with_prop("_global_anchor".to_string(), Ipld::String(anchor.clone()));
    }
    let node_cid = tx.add_node(&node)?;

    // Embed the anchor value as the node's text (e.g. "Hanan", "jalebi").
    // Non-fatal: committed without vector if embedder is unavailable.
    #[cfg(feature = "summarize")]
    if let Some(text) = value_json.as_str() {
        if let Some(pc) = crate::tools::embed::resolve_embed_cfg(server.repo_path()) {
            if let Ok(embedder) = mnem_embed_providers::open(&pc) {
                if let Ok(vec) = embedder.embed(text) {
                    let model = embedder.model().to_string();
                    let emb = mnem_embed_providers::to_embedding(&model, &vec);
                    let _ = tx.set_embedding(node_cid, model, emb);
                }
            }
        }
    }

    let new_repo = tx.commit(&agent_id, "mnem_mcp resolve_or_create")?;

    let mut out = String::new();
    out.push_str("mnem_resolve_or_create: ok\n");
    out.push_str(&format!("  id:            {}\n", id.to_uuid_string()));
    out.push_str(&format!("  label:         {label}\n"));
    if let Some(ref anchor) = global_anchor_uuid {
        out.push_str(&format!("  _global_anchor: {anchor}\n"));
    }
    out.push_str(&format!("  op_id:         {}\n", new_repo.op_id()));
    Ok(out)
}

/// Resolve-or-create `(label, prop_name, value)` in the global graph at
/// `~/.mnemglobal/.mnem/`. Returns the resulting node UUID as a string,
/// or `None` with a stderr note if the global graph is absent or errors.
fn try_stamp_global(
    server: &mut Server,
    label: &str,
    prop_name: &str,
    value: &ipld_core::ipld::Ipld,
    agent_id: &str,
) -> Option<String> {
    let global_data_dir = dirs::home_dir()
        .unwrap_or_else(|| std::path::PathBuf::from("."))
        .join(".mnemglobal")
        .join(".mnem");

    if !global_data_dir.is_dir() {
        eprintln!(
            "note: global graph not found at {}; skipping _global_anchor. \
             Run `mnem integrate` to create it.",
            global_data_dir.display()
        );
        return None;
    }

    let global_repo = if server.repo_path() == global_data_dir {
        match server.load_repo() {
            Ok(r) => r,
            Err(e) => {
                eprintln!("note: could not open global graph: {e}; skipping _global_anchor");
                return None;
            }
        }
    } else {
        match Server::open_repo_at(&global_data_dir) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("note: could not open global graph: {e}; skipping _global_anchor");
                return None;
            }
        }
    };

    let mut tx = global_repo.start_transaction();
    let global_id = match tx.resolve_or_create_node(label, prop_name, value.clone()) {
        Ok(id) => id,
        Err(e) => {
            eprintln!("note: global resolve_or_create failed: {e}; skipping _global_anchor");
            return None;
        }
    };
    if let Err(e) = tx.commit(agent_id, "mnem_mcp resolve_or_create (global anchor)") {
        eprintln!("note: global commit failed: {e}; _global_anchor not stamped");
        return None;
    }
    Some(global_id.to_uuid_string())
}
