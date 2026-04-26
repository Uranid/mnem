use super::*;
use mnem_core::prolly::Cursor;

/// `mnem embedder ...` subcommand group.
///
/// Currently hosts one subcommand, `audit`, which is the CI-enforceable
/// check introduced by Gap 15: every shipped provider MUST override
/// `Embedder::manifest()` and declare a finite `noise_floor`. Any
/// provider that falls through to the panic-default of the trait
/// contract causes `mnem embedder audit` to exit non-zero, so a CI
/// job can fail the build.
#[derive(clap::Subcommand, Debug)]
pub(crate) enum EmbedderCmd {
 /// Verify every shipped embedder provider publishes a manifest
 /// with a non-panicking `noise_floor`. Exits non-zero on failure
 /// so CI can gate on it.
 Audit,
}

#[derive(clap::Args, Debug)]
pub(crate) struct EmbedderArgs {
 #[command(subcommand)]
 pub cmd: EmbedderCmd,
}

pub(crate) fn run_embedder(args: EmbedderArgs) -> Result<()> {
 match args.cmd {
 EmbedderCmd::Audit => run_audit(),
 }
}

fn run_audit() -> Result<()> {
 use mnem_embed_providers::{
 Embedder, EmbedderManifest, OllamaConfig, OpenAiConfig, ProviderConfig,
 };

 // Attempt to construct one instance of every shipped provider and
 // inspect its manifest. A provider that still returns the trait
 // default panics; `std::panic::catch_unwind` turns that panic into
 // a reportable failure rather than aborting the whole CLI.
 //
 // Providers that need external resources (OpenAI needs an API key,
 // Ollama needs a live server) are probed via network-free
 // `from_config` paths. If the provider cannot be constructed in
 // the audit environment it is reported as `skipped`, which is a
 // soft pass: we only want the audit to fail when a provider
 // **exists** but its manifest is invalid.

 let mut failures: Vec<String> = Vec::new();
 let mut checked = 0usize;
 let mut skipped = 0usize;

 // Helper: inspect a constructed embedder's manifest.
 //
 // audit-2026-04-25 R5 (Stage E re-fix): the per-row word now
 // reflects whether `validate_manifest` flagged anything. The
 // earlier shape unconditionally printed `ok` on a successful
 // `manifest()` call even when the manifest was unusable
 // (e.g. dim=0 because the Ollama daemon was unreachable, so
 // the lazy first-call probe never populated the dim). The
 // exit code stayed correct (the trailer counted failures),
 // but the row header lied -- callers grepping per-row state
 // got false negatives.
 let mut check = |label: &str, e: Box<dyn Embedder>| {
 checked += 1;
 let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| e.manifest()));
 match result {
 Ok(m) => {
 let before = failures.len();
 validate_manifest(label, &m, &mut failures);
 let row_word = if failures.len() > before {
 "WARN"
 } else {
 "ok "
 };
 println!(
 "{row_word} {label:40} dim={} noise_floor={:.3}",
 m.dim, m.noise_floor
 );
 }
 Err(_) => {
 failures.push(format!(
 "{label}: Embedder::manifest() panicked (provider has no override)"
 ));
 println!("FAIL {label:40} manifest() panicked");
 }
 }
 };

 // --- OpenAI ---------------------------------------------------------
 // Probed via an audit-scoped env var so no real key is required in
 // CI. If the var is unset, we report `skipped` rather than fail.
 let openai_cfg = ProviderConfig::Openai(OpenAiConfig {
 model: "text-embedding-3-small".into(),
 api_key_env: "MNEM_EMBEDDER_AUDIT_OPENAI_KEY".into(),
 ..Default::default()
 });
 match mnem_embed_providers::open(&openai_cfg) {
 Ok(e) => check("openai:text-embedding-3-small", e),
 Err(_) => {
 skipped += 1;
 println!(
 "skip openai:text-embedding-3-small \
 (set MNEM_EMBEDDER_AUDIT_OPENAI_KEY to include)"
 );
 }
 }

 // --- Ollama ---------------------------------------------------------
 // Construction is network-free; dim is learned lazily. The manifest
 // is inspectable before any HTTP call, which is exactly what we
 // want for an offline audit.
 let ollama_cfg = ProviderConfig::Ollama(OllamaConfig {
 model: "nomic-embed-text".into(),
 ..Default::default()
 });
 match mnem_embed_providers::open(&ollama_cfg) {
 Ok(e) => check("ollama:nomic-embed-text", e),
 Err(_) => {
 skipped += 1;
 println!("skip ollama:nomic-embed-text (not constructible)");
 }
 }

 println!(
 "\nchecked {checked} provider(s), {skipped} skipped, {} failure(s)",
 failures.len()
 );
 if !failures.is_empty() {
 for f in &failures {
 eprintln!(" - {f}");
 }
 anyhow::bail!(
 "embedder audit failed: {} provider(s) missing a valid manifest",
 failures.len()
 );
 }
 // Keep the unused import check honest in builds that skip every
 // provider: referencing `EmbedderManifest` as a fn signature below
 // is the one use site. Without this, clippy warns on unused.
 let _: fn(&str, &EmbedderManifest, &mut Vec<String>) = validate_manifest;
 Ok(())
}

fn validate_manifest(
 label: &str,
 m: &mnem_embed_providers::EmbedderManifest,
 failures: &mut Vec<String>,
) {
 if !m.noise_floor.is_finite() {
 failures.push(format!("{label}: noise_floor is not finite"));
 }
 if !(0.0..=1.0).contains(&m.noise_floor) {
 failures.push(format!(
 "{label}: noise_floor {} out of [0.0, 1.0]",
 m.noise_floor
 ));
 }
 if m.model_id.is_empty() {
 failures.push(format!("{label}: manifest.model_id is empty"));
 }
 // audit-2026-04-25 P3-1: dim=0 used to print as `ok` because the
 // `dim` field was never validated. A zero-dim manifest cannot
 // produce a working embedder; reject it here.
 if m.dim == 0 {
 failures.push(format!("{label}: manifest.dim is 0 (unusable embedder)"));
 }
}

#[derive(clap::Args, Debug)]
#[command(after_long_help = "\
Backfill embeddings for nodes in this repo. One commit per run.

Examples:
 mnem embed # embed every node missing a vector
 mnem embed --force # re-embed even already-embedded nodes
 mnem embed --label Memory # only nodes of this label
 mnem embed --dry-run # count what would be embedded
")]
pub(crate) struct Args {
 /// Re-embed nodes that already have a vector for the current model.
 #[arg(long)]
 pub force: bool,
 /// Restrict to one label (ntype).
 #[arg(long)]
 pub label: Option<String>,
 /// Count and print what would be embedded; don't call the provider.
 #[arg(long)]
 pub dry_run: bool,
 /// Commit message (default: "mnem embed: backfill N nodes").
 #[arg(long, short = 'm')]
 pub message: Option<String>,
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
 let embedder = mnem_embed_providers::open(&pc)?;
 let model_fq = embedder.model().to_string();

 let (_dir, r, bs, _ohs) = repo::open_all(Some(data_dir.as_path()))?;
 let Some(head) = r.head_commit() else {
 // Fresh repo with no nodes yet is not an error: there is
 // nothing to embed. Print the no-op message and exit 0.
 println!("no nodes in this repo yet (run `mnem add node --summary ...` first)");
 return Ok(());
 };

 // Walk every node at head; pick candidates for embedding.
 // Track *why* nodes were skipped so the final message is precise
 // (an earlier version rolled every skip path into one misleading
 // "every node already has a vector" line, which hid real bugs).
 //
 // candidates carry the existing NodeCid alongside
 // the decoded Node so the embed commit can attach the vector via
 // `Transaction::set_embedding(node_cid, ...)` instead of rewriting
 // the node body with `Node::with_embed`.
 let mut candidates: Vec<(mnem_core::id::Cid, Node)> = Vec::new();
 let mut total_nodes: usize = 0;
 let mut matched_label: usize = 0;
 let mut skipped_already_embedded: usize = 0;
 let mut skipped_unembeddable: usize = 0;
 let cursor = Cursor::new(&*bs, &head.nodes)?;
 for entry in cursor {
 let (_k, node_cid) = entry?;
 let bytes = bs
 .get(&node_cid)?
 .ok_or_else(|| anyhow!("node CID {node_cid} missing from store"))?;
 let node: Node = from_canonical_bytes(&bytes)?;
 total_nodes += 1;
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
 if embed_text_of(&node).is_some() {
 candidates.push((node_cid, node));
 } else {
 skipped_unembeddable += 1;
 }
 }

 if candidates.is_empty() {
 // Precise diagnostic: distinguish "no nodes matched the filter"
 // from "every matched node already has a vector" from "matched
 // nodes have no text to embed." A single generic message hid
 // bugs where `--label X` silently excluded every node.
 if matched_label == 0 {
 if let Some(lbl) = &args.label {
 println!(
 "no nodes match --label {lbl} ({total_nodes} node(s) scanned; \
 drop --label to embed across all labels)"
 );
 } else {
 println!("repo has no nodes to embed");
 }
 } else if skipped_already_embedded == matched_label {
 println!(
 "every matched node already has a {model_fq} vector \
 ({skipped_already_embedded} node(s)); use --force to re-embed"
 );
 } else if skipped_unembeddable == matched_label {
 println!("{matched_label} matched node(s) have no summary or content to embed");
 } else {
 println!(
 "nothing to embed: {matched_label} matched, \
 {skipped_already_embedded} already embedded, \
 {skipped_unembeddable} had no embeddable summary or content"
 );
 }
 return Ok(());
 }

 if args.dry_run {
 println!("would embed {} node(s) via {model_fq}", candidates.len());
 return Ok(());
 }

 eprintln!("embedding {} node(s) via {model_fq}...", candidates.len());

 // Build one big transaction; commit atomically so a provider
 // failure mid-run aborts the whole pass cleanly.
 let mut tx = r.start_transaction();
 let mut done = 0usize;
 let total = candidates.len();
 for (node_cid, node) in candidates {
 let text = embed_text_of(&node).expect("filtered above");
 let v = embedder.embed(&text)?;
 let emb = mnem_embed_providers::to_embedding(&model_fq, &v);
 // attach to the existing NodeCid via the
 // sidecar instead of rewriting the node body.
 tx.set_embedding(node_cid, model_fq.clone(), emb)?;
 done += 1;
 if done % 20 == 0 || done == total {
 eprintln!(" {done}/{total}");
 }
 }
 let msg = args
 .message
 .unwrap_or_else(|| format!("mnem embed: backfill {total} nodes with {model_fq}"));
 let new_r = tx.commit(&config::author_string(&cfg), &msg)?;
 println!(
 "embedded {total} node(s); committed as op {}",
 new_r.op_id()
 );
 Ok(())
}
