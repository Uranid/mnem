//! Handler for the `mnem_resolve_or_create` MCP tool.
//!
//! Extracted from `tools.rs` in R3; body unchanged.

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
    // C3-10: default `agent_id` to "mnem-mcp" so the friendly
    // `{name, kind}` shape works end-to-end without forcing the
    // caller to thread an extra field. Mirrors the default in
    // `mnem_ingest`. Callers that pass an explicit `agent_id` keep
    // overriding it.
    let agent_id = args
        .get("agent_id")
        .and_then(Value::as_str)
        .unwrap_or("mnem-mcp")
        .to_string();
    let extra_props = args
        .get("extra_props")
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default();

    let repo = server.load_repo()?;
    let mut tx = repo.start_transaction();
    let id = tx.resolve_or_create_node(&label, &prop_name, value.clone())?;

    // If the node was just created and the caller provided extra props,
    // overwrite with the richer version.
    if !extra_props.is_empty() {
        // `value` is owned here and not referenced after this block, so
        // move it into `with_prop` directly instead of cloning again.
        let mut node = Node::new(id, label.clone()).with_prop(prop_name.clone(), value);
        for (k, v) in &extra_props {
            node = node.with_prop(k.clone(), json_to_ipld(v)?);
        }
        tx.add_node(&node)?;
    }

    let new_repo = tx.commit(&agent_id, "mnem_mcp resolve_or_create")?;

    let mut out = String::new();
    out.push_str("mnem_resolve_or_create: ok\n");
    out.push_str(&format!("  id:     {}\n", id.to_uuid_string()));
    out.push_str(&format!("  label:  {label}\n"));
    out.push_str(&format!("  op_id:  {}\n", new_repo.op_id()));
    Ok(out)
}
