//! Handler for the `mnem_commit_relation` MCP tool.
//!
//! Audit fix G6 (2026-04-25): one-call alternative to the
//! resolve_or_create + resolve_or_create + commit-edge sequence that
//! was previously required to author a single typed relationship. An
//! LLM under no specific instruction will not consistently make all
//! three calls, so the knowledge graph degrades into a flat vector
//! store. This compound primitive collapses the trio into one tool
//! call so the agent's surface for "Alice works at Globex" is a
//! single instruction.

use crate::server::Server;
use anyhow::{Context, Result, anyhow};
use mnem_core::codec::json_to_ipld;
use mnem_core::id::EdgeId;
use mnem_core::objects::{Edge, Node};
use serde_json::Value;

pub(in crate::tools) fn commit_relation(server: &mut Server, args: Value) -> Result<String> {
    let allow_labels = server.allow_labels;

    let subject = args
        .get("subject")
        .and_then(Value::as_str)
        .ok_or_else(|| anyhow!("missing 'subject'"))?
        .to_string();
    let predicate = args
        .get("predicate")
        .and_then(Value::as_str)
        .ok_or_else(|| anyhow!("missing 'predicate' (edge type, e.g. 'works_at')"))?
        .to_string();
    let object = args
        .get("object")
        .and_then(Value::as_str)
        .ok_or_else(|| anyhow!("missing 'object'"))?
        .to_string();

    // Per-side kind labels. Honour caller-supplied values when the
    // gate is on; otherwise both endpoints land on Node::DEFAULT_NTYPE
    // (parity with `mnem_commit` and `mnem_resolve_or_create`).
    let subject_kind = if allow_labels {
        args.get("subject_kind")
            .and_then(Value::as_str)
            .filter(|s| !s.trim().is_empty())
            .unwrap_or(Node::DEFAULT_NTYPE)
            .to_string()
    } else {
        Node::DEFAULT_NTYPE.to_string()
    };
    let object_kind = if allow_labels {
        args.get("object_kind")
            .and_then(Value::as_str)
            .filter(|s| !s.trim().is_empty())
            .unwrap_or(Node::DEFAULT_NTYPE)
            .to_string()
    } else {
        Node::DEFAULT_NTYPE.to_string()
    };

    // Anchor property defaults to "name" - the conventional primary
    // key for entity nodes. Callers anchoring on `email` / `slug` /
    // `id` override via the explicit `anchor` field.
    let anchor = args
        .get("anchor")
        .and_then(Value::as_str)
        .filter(|s| !s.trim().is_empty())
        .unwrap_or("name")
        .to_string();

    let agent_id = args
        .get("agent_id")
        .and_then(Value::as_str)
        .unwrap_or("mnem mcp")
        .to_string();
    let message = args
        .get("message")
        .and_then(Value::as_str)
        .unwrap_or("mnem_mcp commit_relation")
        .to_string();

    let repo = server.load_repo()?;
    let mut tx = repo.start_transaction();

    // Open the embedder once for both subject and object. Provider
    // failures are non-fatal: nodes are committed without a vector
    // and can be backfilled with `mnem reindex`.
    #[cfg(feature = "summarize")]
    let opt_embedder = crate::tools::embed::resolve_embed_cfg(server.repo_path())
        .and_then(|pc| mnem_embed_providers::open(&pc).ok());

    // Step 1: resolve-or-create the subject node anchored on
    // (subject_kind, anchor) == subject. Always write the node so we
    // get a CID back for embedding.
    let subject_value = json_to_ipld(&Value::String(subject.clone()))?;
    let subject_id = tx
        .resolve_or_create_node(&subject_kind, &anchor, subject_value.clone())
        .with_context(|| format!("resolve_or_create subject `{subject}` ({subject_kind})"))?;

    let mut subject_node =
        Node::new(subject_id, subject_kind.clone()).with_prop(anchor.clone(), subject_value);
    if let Some(Value::Object(map)) = args.get("subject_props") {
        for (k, v) in map {
            subject_node = subject_node.with_prop(k.clone(), json_to_ipld(v)?);
        }
    }
    let subject_cid = tx.add_node(&subject_node)?;
    #[cfg(feature = "summarize")]
    if let Some(ref embedder) = opt_embedder {
        if let Ok(vec) = embedder.embed(&subject) {
            let model = embedder.model().to_string();
            let emb = mnem_embed_providers::to_embedding(&model, &vec);
            let _ = tx.set_embedding(subject_cid, model, emb);
        }
    }

    // Step 2: resolve-or-create the object node anchored on
    // (object_kind, anchor) == object. Always write for embedding.
    let object_value = json_to_ipld(&Value::String(object.clone()))?;
    let object_id = tx
        .resolve_or_create_node(&object_kind, &anchor, object_value.clone())
        .with_context(|| format!("resolve_or_create object `{object}` ({object_kind})"))?;

    let mut object_node =
        Node::new(object_id, object_kind.clone()).with_prop(anchor.clone(), object_value);
    if let Some(Value::Object(map)) = args.get("object_props") {
        for (k, v) in map {
            object_node = object_node.with_prop(k.clone(), json_to_ipld(v)?);
        }
    }
    let object_cid = tx.add_node(&object_node)?;
    #[cfg(feature = "summarize")]
    if let Some(ref embedder) = opt_embedder {
        if let Ok(vec) = embedder.embed(&object) {
            let model = embedder.model().to_string();
            let emb = mnem_embed_providers::to_embedding(&model, &vec);
            let _ = tx.set_embedding(object_cid, model, emb);
        }
    }

    // Step 3: add the typed edge from subject to object.
    let mut edge = Edge::new(EdgeId::new_v7(), predicate.as_str(), subject_id, object_id);
    if let Some(Value::Object(map)) = args.get("edge_props") {
        for (k, v) in map {
            edge = edge.with_prop(k.clone(), json_to_ipld(v)?);
        }
    }
    tx.add_edge(&edge)?;

    let opts = mnem_core::repo::CommitOptions::new(agent_id.as_str(), message.as_str());
    let new_repo = tx.commit_opts(opts)?;

    let mut out = String::new();
    out.push_str("mnem_commit_relation: ok\n");
    out.push_str(&format!("  op_id:        {}\n", new_repo.op_id()));
    out.push_str(&format!(
        "  commit_cid:   {}\n",
        new_repo
            .view()
            .heads
            .first()
            .map_or_else(|| "<none>".to_string(), ToString::to_string)
    ));
    out.push_str(&format!(
        "  subject:      {} [{}] {}\n",
        subject_id.to_uuid_string(),
        subject_kind,
        subject
    ));
    out.push_str(&format!("  predicate:    {predicate}\n"));
    out.push_str(&format!(
        "  object:       {} [{}] {}\n",
        object_id.to_uuid_string(),
        object_kind,
        object
    ));
    Ok(out)
}
