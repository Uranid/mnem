//! End-to-end ingest orchestration.
//!
//! [`Ingester`] is the sync driver that turns `source bytes → parsed
//! sections → chunks → entities/relations → graph writes`. It runs
//! against any borrowed [`mnem_core::repo::Transaction`] so the caller keeps
//! full control of commit semantics (when to commit, what author /
//! message to record, whether to stamp a change id).
//!
//! ## Sync, not async
//!
//! mnem-core is a sync library by construction .
//! `mnem-embed-providers::Embedder` is likewise sync. Dragging tokio
//! into the ingest path would force every downstream embedding crate
//! into an async signature too; instead we keep this driver sync and
//! let callers wrap with `tokio::task::spawn_blocking` when they need
//! to integrate with async HTTP handlers.
//!
//! ## What runs per chunk
//!
//! 1. A `"Chunk"` [`Node`] is created, seeded with `summary = first
//! 200 chars`, `content = full chunk text` (raw bytes), and the
//! reserved prop set `mnem:source_kind`, `mnem:section_path`,
//! `mnem:created_at`. An optional [`Embedder`](EmbedderArc) produces
//! an embedding that rides on `Node.embed`.
//! 2. The extractor runs on every section that overlaps the chunk and
//! emits entity spans. Each unique `(kind, canonical_text)` pair
//! gets one graph entity `Node` per ingest run (deduped by a local
//! map). A `"chunk_mentions"` [`Edge`] connects the Chunk to the
//! entity.
//! 3. Candidate relations between entities become edges too, labelled
//! with the predicate chosen by the extractor (`"co_occurs_with"`
//! or `"acts_on"` today).
//!
//! The module is intentionally conservative about commits: it never
//! calls `Transaction::commit` on its own. The caller does, after
//! inspecting [`IngestResult`].

use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use ipld_core::ipld::Ipld;
use mnem_core::id::{EdgeId, NodeId};
use mnem_core::objects::{Edge, Node};
use mnem_core::repo::Transaction;
use tracing::{debug, info_span};

use crate::chunk::{ChunkerKind, chunk as run_chunker};
use crate::error::Error;
use crate::extract::{EntityKind, EntitySpan, Extractor, RuleExtractor};
use crate::types::{Chunk, IngestConfig, IngestResult, Section, SourceKind};

/// Heap-allocated, thread-safe handle to an embedder.
///
/// Abstracted over the concrete `mnem-embed-providers::Embedder` trait
/// so this crate compiles without any provider feature flag. Callers
/// construct one via `Arc::new(...)` around any `Embedder`
/// implementation they ship.
pub type EmbedderArc = Arc<dyn EmbedText>;

/// Sync, fallible text-to-vector contract.
///
/// Intentionally narrower than `mnem-embed-providers::Embedder`:
/// mnem-ingest does not care about model names or batching semantics
/// at this layer - those concerns belong to whatever adapter the CLI /
/// MCP / HTTP layer configures. This keeps mnem-ingest free of any
/// provider dependency for its own tests.
pub trait EmbedText: Send + Sync {
 /// Embed one UTF-8 text and return the on-wire [`mnem_core::objects::Embedding`].
 ///
 /// # Errors
 ///
 /// Returns [`Error::Extractor`] wrapping an upstream error message
 /// whenever the provider fails; callers should treat an embedding
 /// failure as an ingest failure and roll back the transaction.
 fn embed_text(&self, text: &str) -> Result<mnem_core::objects::Embedding, Error>;
}

// ---------------- Ingester ----------------

/// High-level façade tying an [`IngestConfig`], an [`Extractor`], and
/// an optional [`EmbedderArc`] into a reusable driver.
///
/// Multiple `ingest` calls may run sequentially against different
/// transactions; the facade holds no per-run mutable state.
pub struct Ingester {
 /// Chunker + ntype + token budget configuration.
 pub config: IngestConfig,
 /// Pluggable entity / relation extractor. Default is a
 /// [`RuleExtractor`] with the shipped defaults.
 pub extractor: Box<dyn Extractor>,
 /// Optional embedder. When `Some`, every chunk node receives an
 /// [`mnem_core::objects::Embedding`] on `Node.embed`.
 pub embedder: Option<EmbedderArc>,
 /// Optional per-chunk progress callback. Fires after every chunk
 /// has been written into the transaction ().
 /// Lets a CLI / TUI driver tick a real-time progress bar inside
 /// long single-file ingests instead of waiting for the whole file
 /// to commit before the bar moves. The callback is invoked from
 /// the synchronous ingest loop with no buffering, so it should
 /// stay cheap (`AtomicU64::fetch_add`, `ProgressBar::inc(1)`,
 /// etc.). Defaults to `None` so library callers pay no overhead.
 pub progress: Option<std::sync::Arc<dyn Fn() + Send + Sync>>,
}

impl std::fmt::Debug for Ingester {
 fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
 f.debug_struct("Ingester")
 .field("config", &self.config)
 .field("extractor", &"<dyn Extractor>")
 .field(
 "embedder",
 &self.embedder.as_ref().map(|_| "<dyn EmbedText>"),
 )
 .field(
 "progress",
 &self.progress.as_ref().map(|_| "<dyn Fn() + Send + Sync>"),
 )
 .finish()
 }
}

impl Ingester {
 /// Construct an ingester with the default [`RuleExtractor`] and no
 /// embedder.
 #[must_use]
 pub fn new(config: IngestConfig) -> Self {
 Self {
 config,
 extractor: Box::new(RuleExtractor::default()),
 embedder: None,
 progress: None,
 }
 }

 /// Replace the extractor. Returns `self` for chaining.
 #[must_use]
 pub fn with_extractor(mut self, ext: Box<dyn Extractor>) -> Self {
 self.extractor = ext;
 self
 }

 /// Attach an embedder. Returns `self` for chaining.
 #[must_use]
 pub fn with_embedder(mut self, embedder: EmbedderArc) -> Self {
 self.embedder = Some(embedder);
 self
 }

 /// Attach a per-chunk progress callback. Fires once per chunk
 /// after the chunk has been committed to the transaction. Returns
 /// `self` for chaining. See [`Ingester::progress`] for contract.
 #[must_use]
 pub fn with_progress(mut self, cb: std::sync::Arc<dyn Fn() + Send + Sync>) -> Self {
 self.progress = Some(cb);
 self
 }

 /// Detect source kind from a filesystem extension.
 ///
 /// Falls back to [`SourceKind::Text`] when the extension is absent
 /// or unrecognised - matching the default-safe behaviour of every
 /// other parser in this crate.
 #[must_use]
 pub fn source_kind_for_path(path: &std::path::Path) -> SourceKind {
 match path
 .extension()
 .and_then(|s| s.to_str())
 .map(str::to_ascii_lowercase)
 {
 Some(ext) if ext == "md" || ext == "markdown" => SourceKind::Markdown,
 Some(ext) if ext == "pdf" => SourceKind::Pdf,
 Some(ext) if ext == "json" || ext == "jsonl" => SourceKind::Conversation,
 _ => SourceKind::Text,
 }
 }

 /// Parse, chunk, extract, and write into `tx`. Does **not** commit.
 ///
 /// `bytes` is the raw source payload; `kind` says how to parse it.
 /// Returns an [`IngestResult`] with counts and elapsed time. The
 /// `commit_cid` field is left `None` - callers who want a CID
 /// should call `tx.commit(...)` afterwards and stash the returned
 /// `ReadonlyRepo`'s head commit CID.
 ///
 /// # Errors
 ///
 /// - [`Error::ParseFailed`] when the parser rejects the input.
 /// - [`Error::UnsupportedSource`] for source kinds this wave does
 /// not cover (none today - every variant has a parser).
 /// - [`Error::Commit`] for upstream codec / blockstore failures
 /// emitted by `Transaction::add_node` / `add_edge`.
 pub fn ingest(
 &self,
 tx: &mut Transaction,
 bytes: &[u8],
 kind: SourceKind,
 ) -> Result<IngestResult, Error> {
 let started = Instant::now();
 let _span = info_span!("mnem_ingest.run", ?kind).entered();

 let sections = parse(bytes, kind)?;
 let chunker: ChunkerKind = self.config.chunker.clone();
 let chunks = run_chunker(&sections, &chunker);
 debug!(
 n_sections = sections.len(),
 n_chunks = chunks.len(),
 "parse + chunk done"
 );

 // give the extractor a chance to
 // pre-compute anything that scales with section count (e.g.
 // KeyBertAdapter batches every section's embedding through
 // `Embedder::embed_batch` here, so the chunk loop below
 // hits a cache instead of issuing one ORT call per section).
 // Default impl is a no-op; RuleExtractor inherits it.
 self.extractor.prepare(&sections)?;

 let created_at_micros = now_micros();
 let source_kind_str = source_kind_str(kind);

 // Root Doc node.
 let doc_id = NodeId::new_v7();
 let mut doc =
 Node::new(doc_id, self.config.ntype.clone()).with_summary(doc_summary(&sections));
 doc.props.insert(
 "mnem:created_at".into(),
 Ipld::Integer(i128::from(created_at_micros)),
 );
 doc.props.insert(
 "mnem:source_kind".into(),
 Ipld::String(source_kind_str.to_string()),
 );
 tx.add_node(&doc).map_err(Error::commit)?;
 let mut node_count: u64 = 1;
 let mut relation_count: u64 = 0;

 let mut entity_registry: BTreeMap<(EntityKind, String), NodeId> = BTreeMap::new();

 for (chunk_idx, c) in chunks.iter().enumerate() {
 let chunk_id = self.commit_chunk(tx, c, doc_id, created_at_micros, source_kind_str)?;
 node_count += 1;
 if let Some(cb) = &self.progress {
 cb();
 }
 debug!(chunk = chunk_idx, "chunk committed");

 // Extract from every section that overlaps this chunk. The
 // chunker retains section path, so we re-scan the matching
 // sections rather than trying to recover offsets inside the
 // chunk text - cheap, deterministic.
 let mut ents_for_chunk: Vec<(EntitySpan, NodeId)> = Vec::new();
 for section in sections.iter().filter(|s| section_in_chunk(s, c)) {
 let ents = self.extractor.extract_entities(section);
 for e in ents {
 let key = (e.kind, canonical(&e.text));
 let ent_id = if let Some(existing) = entity_registry.get(&key) {
 *existing
 } else {
 let id = NodeId::new_v7();
 let mut n = Node::new(id, e.kind.ntype()).with_summary(e.text.clone());
 n.props.insert(
 "mnem:created_at".into(),
 Ipld::Integer(i128::from(created_at_micros)),
 );
 n.props
 .insert("canonical".into(), Ipld::String(key.1.clone()));
 tx.add_node(&n).map_err(Error::commit)?;
 node_count += 1;
 entity_registry.insert(key, id);
 id
 };

 let mention = Edge::new(EdgeId::new_v7(), "chunk_mentions", chunk_id, ent_id);
 tx.add_edge(&mention).map_err(Error::commit)?;
 ents_for_chunk.push((e, ent_id));
 }

 // Relations - re-run on the same section's entity list so
 // indices line up with what the extractor produced.
 let section_ents: Vec<EntitySpan> =
 ents_for_chunk.iter().map(|(e, _)| e.clone()).collect();
 let rels = self.extractor.extract_relations(&section_ents, section);
 for r in rels {
 let subj_id = ents_for_chunk[r.subject_span].1;
 let obj_id = ents_for_chunk[r.object_span].1;
 let rel_edge = Edge::new(EdgeId::new_v7(), r.kind.clone(), subj_id, obj_id);
 tx.add_edge(&rel_edge).map_err(Error::commit)?;
 relation_count += 1;
 }
 }
 }

 let elapsed_ms = u64::try_from(started.elapsed().as_millis()).unwrap_or(u64::MAX);
 let entity_count = u64::try_from(entity_registry.len()).unwrap_or(u64::MAX);
 let chunk_count = u64::try_from(chunks.len()).unwrap_or(u64::MAX);

 Ok(IngestResult {
 commit_cid: None,
 node_count,
 chunk_count,
 entity_count,
 relation_count,
 elapsed_ms,
 })
 }

 fn commit_chunk(
 &self,
 tx: &mut Transaction,
 c: &Chunk,
 doc_id: NodeId,
 created_at: i64,
 source_kind: &'static str,
 ) -> Result<NodeId, Error> {
 let id = NodeId::new_v7();
 let summary = short_summary(&c.text);
 let mut node = Node::new(id, "Chunk").with_summary(summary);

 node.content = Some(bytes::Bytes::copy_from_slice(c.text.as_bytes()));
 node.props.insert(
 "mnem:created_at".into(),
 Ipld::Integer(i128::from(created_at)),
 );
 node.props.insert(
 "mnem:source_kind".into(),
 Ipld::String(source_kind.to_string()),
 );
 node.props.insert(
 "mnem:section_path".into(),
 Ipld::List(
 c.section_path
 .iter()
 .map(|s| Ipld::String(s.clone()))
 .collect(),
 ),
 );
 node.props.insert(
 "tokens_estimate".into(),
 Ipld::Integer(i128::from(c.tokens_estimate)),
 );

 let pending_emb = if let Some(embedder) = &self.embedder {
 Some(embedder.embed_text(&c.text)?)
 } else {
 None
 };

 let chunk_cid = tx.add_node(&node).map_err(Error::commit)?;
 if let Some(emb) = pending_emb {
 let model = emb.model.clone();
 tx.set_embedding(chunk_cid, model, emb)
 .map_err(Error::commit)?;
 }
 // Link chunk to doc root.
 let edge = Edge::new(EdgeId::new_v7(), "chunk_of", id, doc_id);
 tx.add_edge(&edge).map_err(Error::commit)?;
 Ok(id)
 }
}

// ---------------- Free helpers ----------------

fn parse(bytes: &[u8], kind: SourceKind) -> Result<Vec<Section>, Error> {
 match kind {
 SourceKind::Markdown => {
 let s = std::str::from_utf8(bytes).map_err(|e| Error::ParseFailed {
 what: "markdown".into(),
 detail: e.to_string(),
 })?;
 crate::md::parse_markdown(s)
 }
 SourceKind::Text => {
 let s = std::str::from_utf8(bytes).map_err(|e| Error::ParseFailed {
 what: "text".into(),
 detail: e.to_string(),
 })?;
 crate::text::parse_text(s)
 }
 SourceKind::Pdf => crate::pdf::parse_pdf(bytes),
 SourceKind::Conversation => crate::conversation::parse_conversation(bytes),
 }
}

fn section_in_chunk(section: &Section, chunk: &Chunk) -> bool {
 // Headings as a coarse "is this section under the chunk's section path"
 // check. Without exact offsets this is a heuristic; it still beats
 // running the extractor against the whole document per chunk.
 match (&section.heading, chunk.section_path.last()) {
 (Some(h), Some(last)) => h == last,
 (None, _) => true,
 _ => false,
 }
}

fn doc_summary(sections: &[Section]) -> String {
 for s in sections {
 let trimmed = s.text.trim();
 if !trimmed.is_empty() {
 return short_summary(trimmed);
 }
 }
 "(empty)".into()
}

fn short_summary(text: &str) -> String {
 let trimmed = text.trim();
 if trimmed.len() <= 200 {
 return trimmed.to_string();
 }
 let mut end = 200;
 while end > 0 && !trimmed.is_char_boundary(end) {
 end -= 1;
 }
 format!("{}…", &trimmed[..end])
}

fn canonical(s: &str) -> String {
 s.trim().to_lowercase()
}

const fn source_kind_str(kind: SourceKind) -> &'static str {
 match kind {
 SourceKind::Markdown => "markdown",
 SourceKind::Text => "text",
 SourceKind::Pdf => "pdf",
 SourceKind::Conversation => "conversation",
 }
}

fn now_micros() -> i64 {
 let d = SystemTime::now()
 .duration_since(UNIX_EPOCH)
 .unwrap_or_default();
 i64::try_from(d.as_micros()).unwrap_or(i64::MAX)
}

// ---------------- Tests ----------------

#[cfg(test)]
mod tests {
 use super::*;
 use bytes::Bytes;
 use mnem_core::objects::{Dtype, Embedding};
 use mnem_core::repo::ReadonlyRepo;
 use mnem_core::store::{MemoryBlockstore, MemoryOpHeadsStore};
 use std::sync::Arc as StdArc;

 fn test_repo() -> ReadonlyRepo {
 let bs = StdArc::new(MemoryBlockstore::new());
 let op_heads = StdArc::new(MemoryOpHeadsStore::new());
 ReadonlyRepo::init(bs, op_heads).expect("init repo")
 }

 /// Deterministic 384-dimension embedder used by the pipeline tests.
 struct StubEmbedder;
 impl EmbedText for StubEmbedder {
 fn embed_text(&self, _text: &str) -> Result<Embedding, Error> {
 let v: Vec<f32> = (0..384)
 .map(|i| f32::from(i16::try_from(i % 256).unwrap_or(0)) * 0.01)
 .collect();
 let mut buf = Vec::with_capacity(v.len() * 4);
 for x in v {
 buf.extend_from_slice(&x.to_le_bytes());
 }
 Ok(Embedding {
 model: "stub:test".into(),
 dtype: Dtype::F32,
 dim: 384,
 vector: Bytes::from(buf),
 })
 }
 }

 #[test]
 fn ingest_markdown_produces_doc_and_chunks() {
 let repo = test_repo();
 let mut tx = repo.start_transaction();
 let ing = Ingester::new(IngestConfig::default());
 let md = "# Phase-B5c\n\nAlice Johnson joined Acme Corp on 2026-04-24.\n\nSee https://example.com for details.";

 let result = ing
 .ingest(&mut tx, md.as_bytes(), SourceKind::Markdown)
 .expect("ingest ok");

 assert!(result.chunk_count >= 1, "got {result:?}");
 assert!(result.node_count >= 2, "expected doc + chunks + entities");
 assert!(result.entity_count >= 1, "expected at least one entity");
 }

 #[test]
 fn ingest_text_respects_embedder() {
 let repo = test_repo();
 let mut tx = repo.start_transaction();
 let ing = Ingester::new(IngestConfig::default()).with_embedder(StdArc::new(StubEmbedder));

 let body = "Plain body. Alice Johnson met Bob Lee at Acme Corp.";
 let result = ing
 .ingest(&mut tx, body.as_bytes(), SourceKind::Text)
 .expect("ingest ok");

 assert!(result.node_count >= 2);
 assert!(result.chunk_count >= 1);
 }

 #[test]
 fn source_kind_for_path_maps_extensions() {
 use std::path::Path;
 assert_eq!(
 Ingester::source_kind_for_path(Path::new("/x/y.md")),
 SourceKind::Markdown
 );
 assert_eq!(
 Ingester::source_kind_for_path(Path::new("y.MARKDOWN")),
 SourceKind::Markdown
 );
 assert_eq!(
 Ingester::source_kind_for_path(Path::new("book.pdf")),
 SourceKind::Pdf
 );
 assert_eq!(
 Ingester::source_kind_for_path(Path::new("chat.json")),
 SourceKind::Conversation
 );
 assert_eq!(
 Ingester::source_kind_for_path(Path::new("notes.txt")),
 SourceKind::Text
 );
 assert_eq!(
 Ingester::source_kind_for_path(Path::new("noext")),
 SourceKind::Text
 );
 }

 #[test]
 fn ingest_is_deterministic_in_counts() {
 let md = "# H\n\nBob Lee visited https://foo.io on 2026-04-24.";
 let repo1 = test_repo();
 let mut tx1 = repo1.start_transaction();
 let r1 = Ingester::new(IngestConfig::default())
 .ingest(&mut tx1, md.as_bytes(), SourceKind::Markdown)
 .unwrap();

 let repo2 = test_repo();
 let mut tx2 = repo2.start_transaction();
 let r2 = Ingester::new(IngestConfig::default())
 .ingest(&mut tx2, md.as_bytes(), SourceKind::Markdown)
 .unwrap();

 assert_eq!(r1.chunk_count, r2.chunk_count);
 assert_eq!(r1.entity_count, r2.entity_count);
 assert_eq!(r1.relation_count, r2.relation_count);
 }
}
