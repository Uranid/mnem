//! `mnem reindex` - retro-embed nodes that don't yet have a vector.
//!
//! one-shot upgrade path for repos that grew
//! before `mnem add node` learned to auto-embed (C7-4). Walks every
//! node at HEAD, picks the ones missing a `dense_embed` (or all of
//! them when `--force`), embeds them via the configured provider,
//! and commits the result.
//!
//! Mirrors `mnem embed` (the historical spelling) closely; the new
//! verb is the one promoted in the user-facing error message that
//! `mnem add node` now prints when the embedder is unreachable.
//! Functionally:
//!
//! - Source text per node = `node.summary` when set; otherwise a
//! stringified `label + sorted props` so a node without a summary
//! still ends up with *some* vector instead of being silently
//! skipped.
//! - `--since <commit>` (optional) narrows the candidate set to
//! nodes that did not exist (or differ) at the supplied commit,
//! so an operator can re-embed only the tail of recent additions
//! without re-walking the whole graph.
//! - Idempotent: running twice with no flags is a no-op (the second
//! pass sees every node already has a vector for the current
//! model).

use super::*;
use indicatif::{ProgressBar, ProgressStyle};
use mnem_core::id::Cid;
use mnem_core::prolly::Cursor;
use std::collections::HashSet;
use std::time::Instant;

#[derive(clap::Args, Debug)]
#[command(after_long_help = "\
Retro-embed nodes that don't have a vector yet. One commit per run.

Examples:
 mnem reindex # embed every node missing a vector
 mnem reindex --label Person # only nodes of this label
 mnem reindex --since <commit> # only nodes added/changed since <commit>
 mnem reindex --force # re-embed even already-embedded nodes
 mnem reindex --dry-run # report count without changing anything

Source text per node:
 - `summary` (the `-s` argument to `mnem add node`) when set
 - else `label + sorted props` rendered as text (so unsummarised
 nodes still receive a vector instead of being silently skipped)
")]
pub(crate) struct Args {
 /// Re-embed nodes that already have a vector for the current model.
 #[arg(long)]
 pub force: bool,
 /// Restrict to one label (ntype).
 #[arg(long)]
 pub label: Option<String>,
 /// Only re-embed nodes added (or changed) after this commit. The
 /// commit may be a CID, ref name, branch name, or `HEAD`. Nodes
 /// present in `<since>`'s nodes-tree are skipped.
 #[arg(long, value_name = "COMMIT")]
 pub since: Option<String>,
 /// Count and print what would be embedded; don't call the provider.
 #[arg(long)]
 pub dry_run: bool,
 /// Commit message (default: "mnem reindex: N nodes embedded").
 #[arg(long, short = 'm')]
 pub message: Option<String>,
}

/// Render a fallback source string for a node with no `summary`. Uses
/// the label plus sorted props so the input is deterministic across
/// runs (prop iteration order is otherwise insertion-defined).
fn fallback_text_of(node: &Node) -> String {
 let mut parts: Vec<String> = Vec::with_capacity(1 + node.props.len());
 parts.push(node.ntype.clone());
 let mut keys: Vec<&String> = node.props.keys().collect();
 keys.sort();
 for k in keys {
 if let Some(v) = node.props.get(k) {
 parts.push(format!("{k}={}", ipld_to_text(v)));
 }
 }
 parts.join(" ")
}

/// Best-effort textification of an Ipld value for fallback embed
/// input. Falls back to a debug rendering for shapes that don't
/// stringify cleanly (lists / maps); the goal is "some vector" not
/// "perfect vector."
fn ipld_to_text(v: &Ipld) -> String {
 match v {
 Ipld::Null => String::new(),
 Ipld::Bool(b) => b.to_string(),
 Ipld::Integer(i) => i.to_string(),
 Ipld::Float(f) => f.to_string(),
 Ipld::String(s) => s.clone(),
 Ipld::Bytes(b) => format!("[{}b]", b.len()),
 Ipld::List(_) | Ipld::Map(_) => format!("{v:?}"),
 Ipld::Link(c) => c.to_string(),
 }
}

/// Pick the embed source for a node. Mirrors `embed_text_of` but
/// always returns `Some(_)` so unsummarised nodes still get a
/// vector via the label+props fallback.
fn reindex_text_of(node: &Node) -> String {
 if let Some(s) = &node.summary
 && !s.trim().is_empty()
 {
 return s.clone();
 }
 if let Some(text) = embed_text_of(node) {
 return text;
 }
 fallback_text_of(node)
}

/// Walk the nodes-tree at `commit_cid` and return the set of node
/// CIDs present there. Used by `--since` to filter candidates down
/// to "newly added or changed" nodes only.
fn nodes_at(
 bs: &std::sync::Arc<dyn mnem_core::store::Blockstore>,
 commit_cid: &Cid,
) -> Result<HashSet<Cid>> {
 let bytes = bs
 .get(commit_cid)?
 .ok_or_else(|| anyhow!("commit CID {commit_cid} missing from store"))?;
 let commit: Commit = from_canonical_bytes(&bytes)?;
 let mut out: HashSet<Cid> = HashSet::new();
 let cursor = Cursor::new(&**bs, &commit.nodes)?;
 for entry in cursor {
 let (_k, node_cid) = entry?;
 out.insert(node_cid);
 }
 Ok(out)
}

pub(crate) fn run(override_path: Option<&Path>, args: Args) -> Result<()> {
 let data_dir = repo::locate_data_dir(override_path)?;
 let cfg = config::load(&data_dir)?;
 let Some(pc) = config::resolve_embedder(&cfg) else {
 anyhow::bail!(
 "no embedder configured; run `mnem config set embed.provider <openai|ollama>` \
 and `mnem config set embed.model <name>` first"
 );
 };

 let (_dir, r, bs, _ohs) = repo::open_all(Some(data_dir.as_path()))?;
 let Some(head) = r.head_commit() else {
 println!("no nodes in this repo yet (run `mnem add node --summary ...` first)");
 return Ok(());
 };

 // Resolve --since up front so we can fail fast on a bad arg
 // before opening the (potentially live-network) embedder.
 let since_set: Option<HashSet<Cid>> = match &args.since {
 None => None,
 Some(s) => {
 let cid = resolve_commitish(&r, s)?;
 Some(nodes_at(&bs, &cid)?)
 }
 };

 // Open the embedder *after* the dry-run path has had a chance to
 // bail without a network call. We still need its `model()` for
 // the "already embedded" check though, so for non-dry-run we
 // open it here and reuse below.
 //
 // Strategy: always try to open. If `--dry-run` and the open
 // fails, we can still print the count using a placeholder model
 // string -- the user only cares about *how many* would change.
 // Per spec, Ollama-unreachable on `mnem reindex` is a hard error
 // (the user explicitly asked to embed, unlike `mnem add node`
 // where embedding is incidental); this matches `mnem embed`.
 let embedder_result = mnem_embed_providers::open(&pc);
 let (embedder, model_fq) = match (&embedder_result, args.dry_run) {
 (Ok(e), _) => {
 let m = e.model().to_string();
 (Some(e), m)
 }
 (Err(_), true) => (None, String::from("<configured-embedder>")),
 (Err(e), false) => {
 eprintln!("{}", format_embed_failure(e, &pc, "embedding"));
 anyhow::bail!("cannot reindex: embedder open failed (see above)");
 }
 };

 // Walk every node at head; pick candidates per the same rules as
 // `mnem embed`, with the addition of the `--since` filter.
 //
 // candidates carry their existing NodeCid alongside
 // the decoded Node so the reindex commit can attach the new vector
 // via `Transaction::set_embedding(node_cid, ...)` without rewriting
 // the node body. The legacy `node.with_embed(emb)` rewrite path is
 // gone; removes `Node::embed` outright.
 let mut candidates: Vec<(Cid, Node)> = Vec::new();
 let mut total_nodes: usize = 0;
 let mut matched_label: usize = 0;
 let mut skipped_already_embedded: usize = 0;
 let mut skipped_outside_since: usize = 0;
 let cursor = Cursor::new(&*bs, &head.nodes)?;
 for entry in cursor {
 let (_k, node_cid) = entry?;
 let bytes = bs
 .get(&node_cid)?
 .ok_or_else(|| anyhow!("node CID {node_cid} missing from store"))?;
 let node: Node = from_canonical_bytes(&bytes)?;
 total_nodes += 1;

 if let Some(set) = &since_set
 && set.contains(&node_cid)
 {
 skipped_outside_since += 1;
 continue;
 }

 if let Some(lbl) = &args.label
 && &node.ntype != lbl
 {
 continue;
 }
 matched_label += 1;

 // Embedding lives in the sidecar bucket keyed by NodeCid.
 // "Already embedded under this model" is a sidecar lookup,
 // not a node-body field; `--force` re-embeds regardless.
 let already = if args.force {
 false
 } else {
 r.embedding_for(&node_cid, &model_fq)?.is_some()
 };
 if already {
 skipped_already_embedded += 1;
 continue;
 }
 candidates.push((node_cid, node));
 }

 if candidates.is_empty() {
 if matched_label == 0 {
 if let Some(lbl) = &args.label {
 println!(
 "no nodes match --label {lbl} ({total_nodes} node(s) scanned; \
 drop --label to reindex across all labels)"
 );
 } else if since_set.is_some() && skipped_outside_since == total_nodes {
 println!(
 "no nodes added since the supplied commit \
 ({total_nodes} node(s) scanned)"
 );
 } else {
 println!("repo has no nodes to reindex");
 }
 } else if skipped_already_embedded == matched_label {
 println!(
 "every matched node already has a {model_fq} vector \
 ({skipped_already_embedded} node(s)); use --force to re-embed"
 );
 } else {
 println!(
 "nothing to reindex: {matched_label} matched, \
 {skipped_already_embedded} already embedded"
 );
 }
 return Ok(());
 }

 if args.dry_run {
 println!(
 "would reindex {} node(s) via {model_fq}",
 candidates.len()
 );
 return Ok(());
 }

 // Past the dry-run gate, the embedder must be live -- we proved
 // this above by bailing on Err for the non-dry-run path.
 let embedder = embedder.expect("embedder live for non-dry-run path");

 let total = candidates.len();
 let started = Instant::now();
 eprintln!("reindexing {total} node(s) via {model_fq}");
 let pb = ProgressBar::new(total as u64);
 pb.set_style(
 ProgressStyle::with_template(
 " [{elapsed_precise}] {bar:32.cyan/blue} {pos}/{len} ({percent}%) ETA {eta}",
 )
 .unwrap()
 .progress_chars("=>-"),
 );

 let mut tx = r.start_transaction();
 for (node_cid, node) in candidates {
 let text = reindex_text_of(&node);
 let v = embedder.embed(&text)?;
 let emb = mnem_embed_providers::to_embedding(&model_fq, &v);
 // attach to the existing NodeCid via the
 // sidecar instead of rewriting the node body. The Node
 // bytes are unchanged so the CID we read from the cursor
 // is still the canonical key for this node's embeddings.
 tx.set_embedding(node_cid, model_fq.clone(), emb)?;
 pb.inc(1);
 }
 pb.finish_and_clear();

 let msg = args
 .message
 .unwrap_or_else(|| format!("mnem reindex: {total} nodes embedded with {model_fq}"));
 let new_r = tx.commit(&config::author_string(&cfg), &msg)?;
 println!(
 "reindexed {total} node(s) in {:.1}s; committed as op {}",
 started.elapsed().as_secs_f32(),
 new_r.op_id()
 );
 Ok(())
}

#[cfg(test)]
mod tests {
 use super::*;
 use mnem_core::id::NodeId;

 #[test]
 fn fallback_text_uses_label_and_sorted_props() {
 let n = Node::new(NodeId::from_bytes_raw([1u8; 16]), "Person")
 .with_prop("name", Ipld::String("Alice".into()))
 .with_prop("city", Ipld::String("Berlin".into()));
 let s = fallback_text_of(&n);
 // "Person" first; props alphabetised.
 assert!(s.starts_with("Person "), "got: {s}");
 let ci = s.find("city=").expect("city present");
 let ni = s.find("name=").expect("name present");
 assert!(ci < ni, "props must be sorted: {s}");
 }

 #[test]
 fn reindex_text_prefers_summary() {
 let n = Node::new(NodeId::from_bytes_raw([2u8; 16]), "Doc")
 .with_summary("Important brief")
 .with_prop("title", Ipld::String("X".into()));
 assert_eq!(reindex_text_of(&n), "Important brief");
 }

 #[test]
 fn reindex_text_falls_back_when_no_summary_or_content() {
 let n = Node::new(NodeId::from_bytes_raw([3u8; 16]), "Person")
 .with_prop("name", Ipld::String("Bob".into()));
 let s = reindex_text_of(&n);
 assert!(s.contains("Person"));
 assert!(s.contains("name=Bob"));
 }
}
