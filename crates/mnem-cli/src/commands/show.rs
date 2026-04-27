//! `mnem show` - decode a content-addressed object and pretty-print it.
//!
//! `show` accepts any CID in the local blockstore. It peeks the
//! `_kind` discriminator, re-decodes into the concrete type, and
//! renders a human-readable summary. Raw bytes + DAG-JSON output live
//! in `mnem cat-file`.
//!
//! Supported kinds: `node`, `edge`, `commit`, `operation`, `view`,
//! `index_set`, `tombstone`. Unknown kinds print the `_kind` string
//! and a byte count; callers can then reach for
//! `mnem cat-file --json` to inspect the payload.
//!
//! A bare `mnem show` (no CID) still defaults to the current op-head
//! so the existing `mnem log -n 1 && mnem show` workflow survives.
//!
//! # Examples
//!
//! ```text
//! mnem show                 # current op-head
//! mnem show <node-cid>      # node at a specific CID
//! mnem show <commit-cid>    # commit summary (tree roots + parents)
//! ```

use ipld_core::ipld::Ipld;
use mnem_core::objects::{Edge, Node, Tombstone, View};

use super::*;

#[derive(clap::Args, Debug)]
#[command(after_long_help = "\
Examples:
  mnem show                 # show the current op-head
  mnem show <cid>           # decode any Node/Edge/Commit/View/Operation/IndexSet/Tombstone
  mnem log -n 1 && mnem show
")]
pub(crate) struct Args {
    /// Any CID in the repo. Defaults to the current op-head.
    pub cid: Option<String>,
}

pub(crate) fn run(override_path: Option<&Path>, args: Args) -> Result<()> {
    let (_dir, r, bs, _ohs) = repo::open_all(override_path)?;

    let target_cid = match args.cid {
        // audit-2026-04-25 P2-3: accept symbolic refs (HEAD, branch
        // name, refs/heads/...) in addition to raw CIDs. Uses the
        // shared resolver so behaviour matches `mnem merge`.
        Some(s) => super::resolve_commitish(&r, &s)?,
        None => r.op_id().clone(),
    };

    let bytes = bs
        .get(&target_cid)?
        .ok_or_else(|| anyhow!("block {target_cid} not found in this blockstore"))?;

    // Peek the _kind discriminator by decoding into Ipld first. Every
    // mnem object carries `_kind`; a block without it is either a
    // future format or a non-mnem CAS block (bare bytes from `mnem
    // cat-file`, for instance). Surface that honestly rather than
    // try-decode every concrete type in sequence.
    let kind = peek_kind(&bytes);
    println!("cid         {target_cid}");
    println!("size        {} bytes", bytes.len());
    println!(
        "kind        {}",
        kind.as_deref().unwrap_or("<unknown (no _kind field)>")
    );

    match kind.as_deref() {
        Some("node") => show_node(&bytes, &r, &target_cid),
        Some("edge") => show_edge(&bytes),
        Some("commit") => show_commit(&bytes),
        Some("operation") => show_operation(&bytes, &bs),
        Some("view") => show_view(&bytes),
        Some("index_set") => show_index_set(&bytes),
        Some("tombstone") => show_tombstone(&bytes),
        _ => {
            // Unknown kinds (or no `_kind`): already printed the
            // cid/size/kind header; the user can `mnem cat-file
            // --json` for a DAG-JSON dump.
            println!("(no structured pretty-printer; try `mnem cat-file {target_cid} --json`)");
            Ok(())
        }
    }
}

/// Decode the block as IPLD, then pull `_kind` out of the top-level
/// map if present. Returns `None` for non-map payloads or payloads
/// without a string `_kind` field.
fn peek_kind(bytes: &[u8]) -> Option<String> {
    let ipld: Ipld = from_canonical_bytes(bytes).ok()?;
    match ipld {
        Ipld::Map(m) => match m.get("_kind")? {
            Ipld::String(s) => Some(s.clone()),
            _ => None,
        },
        _ => None,
    }
}

fn show_node(bytes: &[u8], repo: &ReadonlyRepo, node_cid: &mnem_core::id::Cid) -> Result<()> {
    let n: Node = from_canonical_bytes(bytes).context("decoding node")?;
    println!("id          {}", n.id.to_uuid_string());
    println!("ntype       {}", n.ntype);
    if let Some(s) = &n.summary {
        println!(
            "summary     {}",
            if s.len() <= 120 {
                s.clone()
            } else {
                format!(
                    "{}... ({}B)",
                    s.chars().take(117).collect::<String>(),
                    s.len()
                )
            }
        );
    }
    if !n.props.is_empty() {
        println!("props       ({})", n.props.len());
        for (k, v) in &n.props {
            println!("  {k:<16} {}", ipld_preview(v));
        }
    }
    if let Some(c) = &n.content {
        println!("content     {} bytes", c.len());
    }
    // Embeddings are sidecar-attached, not Node-inline. Probe the
    // configured embedder's `model_fq` string (the same one used at
    // write time) and surface presence + (model, dim, dtype) when a
    // vector exists. Silent skip when no embedder is configured;
    // `mnem cat-file --json` is the full-fidelity escape hatch.
    if let Some(model) = configured_model_fq()
        && let Some(emb) = repo.embedding_for(node_cid, &model)?
    {
        println!(
            "embed       model={} dim={} dtype={:?}",
            emb.model, emb.dim, emb.dtype
        );
    }
    Ok(())
}

/// Resolve the configured embedder's canonical `model_fq` string
/// (e.g. `"openai:text-embedding-3-small"`) without opening the
/// provider. Returns `None` when no embedder is configured.
///
/// `mnem show` reads only - opening the provider could trip a
/// network call (Ollama probe) which is wrong for a diagnostic.
fn configured_model_fq() -> Option<String> {
    let cfg = config::load_global().ok()?;
    let pc = config::resolve_embedder(&cfg)?;
    Some(model_fq_of(&pc))
}

/// Format the `provider:model` string the embedder adapters expose
/// via `Embedder::model()`. Mirrored here so the CLI can derive it
/// from a `ProviderConfig` without opening the adapter.
fn model_fq_of(pc: &mnem_embed_providers::ProviderConfig) -> String {
    use mnem_embed_providers::ProviderConfig as PC;
    match pc {
        PC::Openai(c) => format!("openai:{}", c.model),
        PC::Ollama(c) => format!("ollama:{}", c.model),
        PC::Onnx(c) => format!("onnx:{}", c.model),
    }
}

fn show_edge(bytes: &[u8]) -> Result<()> {
    let e: Edge = from_canonical_bytes(bytes).context("decoding edge")?;
    println!("id          {}", e.id.to_uuid_string());
    println!("etype       {}", e.etype);
    println!("src         {}", e.src.to_uuid_string());
    println!("dst         {}", e.dst.to_uuid_string());
    if !e.props.is_empty() {
        println!("props       ({})", e.props.len());
        for (k, v) in &e.props {
            println!("  {k:<16} {}", ipld_preview(v));
        }
    }
    Ok(())
}

fn show_commit(bytes: &[u8]) -> Result<()> {
    let c: Commit = from_canonical_bytes(bytes).context("decoding commit")?;
    println!("change_id   {}", c.change_id.to_uuid_string());
    println!("time        {}us", c.time);
    println!("author      {}", c.author);
    if let Some(a) = &c.agent_id {
        println!("agent_id    {a}");
    }
    if let Some(t) = &c.task_id {
        println!("task_id     {t}");
    }
    println!("message     {}", c.message);
    println!(
        "parents     {}",
        if c.parents.is_empty() {
            "<root>".into()
        } else {
            c.parents
                .iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>()
                .join(", ")
        }
    );
    println!("nodes       {}", c.nodes);
    println!("edges       {}", c.edges);
    println!("schema      {}", c.schema);
    if let Some(i) = &c.indexes {
        println!("indexes     {i}");
    }
    if c.signature.is_some() {
        println!("signature   <present>");
    }
    Ok(())
}

fn show_operation(
    bytes: &[u8],
    bs: &std::sync::Arc<dyn mnem_core::store::Blockstore>,
) -> Result<()> {
    let op: Operation = from_canonical_bytes(bytes).context("decoding operation")?;
    println!("time        {}us", op.time);
    println!("author      {}", op.author);
    if let Some(a) = &op.agent_id {
        println!("agent_id    {a}");
    }
    if let Some(t) = &op.task_id {
        println!("task_id     {t}");
    }
    println!("description {}", op.description);
    println!(
        "parents     {}",
        if op.parents.is_empty() {
            "<root>".into()
        } else {
            op.parents
                .iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>()
                .join(", ")
        }
    );
    println!("view        {}", op.view);
    // Decode the view for a one-line head summary.
    if let Some(view_bytes) = bs.get(&op.view)? {
        let view: View = from_canonical_bytes(&view_bytes).context("decoding view")?;
        if let Some(head_cid) = view.heads.first() {
            println!("  head      {head_cid}");
        }
        println!("  refs      {}", view.refs.len());
    }
    Ok(())
}

fn show_view(bytes: &[u8]) -> Result<()> {
    let v: View = from_canonical_bytes(bytes).context("decoding view")?;
    println!("heads       {}", v.heads.len());
    for h in &v.heads {
        println!("  {h}");
    }
    println!("refs        {}", v.refs.len());
    for (name, target) in &v.refs {
        match target {
            RefTarget::Normal { target } => println!("  {name} -> {target}"),
            RefTarget::Conflicted { adds, removes } => {
                println!("  {name} conflicted(+{} -{})", adds.len(), removes.len());
            }
        }
    }
    if let Some(rr) = &v.remote_refs {
        let total: usize = rr.values().map(BTreeMap::len).sum();
        println!("remote_refs {} across {} remote(s)", total, rr.len());
    }
    if !v.tombstones.is_empty() {
        println!("tombstones  {}", v.tombstones.len());
    }
    Ok(())
}

fn show_index_set(bytes: &[u8]) -> Result<()> {
    let idx: IndexSet = from_canonical_bytes(bytes).context("decoding index_set")?;
    println!("labels      {}", idx.nodes_by_label.len());
    for (label, cid) in &idx.nodes_by_label {
        println!("  {label:<20} {cid}");
    }
    let prop_count: usize = idx.nodes_by_prop.values().map(BTreeMap::len).sum();
    println!(
        "nodes_by_prop {} across {} label(s)",
        prop_count,
        idx.nodes_by_prop.len()
    );
    println!(
        "outgoing    {}",
        idx.outgoing
            .as_ref()
            .map_or_else(|| "<none>".into(), ToString::to_string)
    );
    println!(
        "incoming    {}",
        idx.incoming
            .as_ref()
            .map_or_else(|| "<none>".into(), ToString::to_string)
    );
    Ok(())
}

fn show_tombstone(bytes: &[u8]) -> Result<()> {
    let t: Tombstone = from_canonical_bytes(bytes).context("decoding tombstone")?;
    println!("tombstoned_at {}us", t.tombstoned_at);
    println!("reason        {}", t.reason);
    Ok(())
}

// Need the BTreeMap import for the helpers above.
use std::collections::BTreeMap;
