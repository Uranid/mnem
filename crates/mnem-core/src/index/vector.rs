//! Vector-search primitive for mnem.
//!
//! Provides a [`VectorIndex`] trait and a default
//! [`BruteForceVectorIndex`] implementation (cosine-similarity linear
//! scan, built in memory from the current repo head).
//!
//! # Model scoping
//!
//! Embeddings produced by different models occupy different semantic
//! spaces: an `openai:text-embedding-3-small` vector cannot be mixed
//! with a `nomic:embed-text-v1.5` vector even if they share a
//! dimension. Each [`BruteForceVectorIndex`] therefore binds to a
//! single `(model, dim)` pair at build time and silently skips nodes
//! with other embeddings. Agents that use several models build one
//! index per model.
//!
//! # Determinism
//!
//! Build order is the canonical Prolly-tree key order, ties break on
//! `NodeId` ASC, and scores are computed from stored normalised f32
//! vectors. Given the same repo head and the same query, two independent
//! processes return byte-identical hit lists.
//!
//! # Example
//!
//! ```no_run
//! # use mnem_core::repo::ReadonlyRepo;
//! # use mnem_core::index::vector::{BruteForceVectorIndex, VectorIndex};
//! # fn demo(repo: &ReadonlyRepo, query: &[f32]) -> Result<(), Box<dyn std::error::Error>> {
//! let idx = BruteForceVectorIndex::build_from_repo(repo, "openai:text-embedding-3-small")?;
//! let hits = idx.search(query, 5)?;
//! for h in hits {
//! println!("{} @ {:.4}", h.node_id, h.score);
//! }
//! # Ok(()) }
//! ```
//!
//! # Why brute force?
//!
//! Brute force is the correctness baseline every ANN system is measured
//! against. It has zero hyperparameters, trivial determinism, no
//! background build phase, and costs nothing in deps. For agent
//! workloads in the <=100k-vector range (the common case) a tight
//! vector-row dot product hits <20 ms per query on a laptop. HNSW
//! lands as a sibling impl under the same trait once corpus sizes
//! justify the added complexity.

use std::sync::Arc;

use bytes::Bytes;

use crate::error::{Error, RepoError};
use crate::id::NodeId;
use crate::objects::{Dtype, Embedding, Node};
use crate::prolly::Cursor;
use crate::repo::readonly::{ReadonlyRepo, decode_from_store};
use crate::store::Blockstore;

// ============================================================
// Public surface
// ============================================================

/// One scored match returned by a [`VectorIndex`] search.
///
/// `#[non_exhaustive]` keeps field adds backward-compatible for
/// downstream `match` sites; callers in sibling crates
/// (e.g. `mnem-ann`) build instances via [`VectorHit::new`].
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub struct VectorHit {
 /// The matched node's stable identity.
 pub node_id: NodeId,
 /// Cosine similarity in `[-1.0, 1.0]`. Higher is closer.
 pub score: f32,
}

impl VectorHit {
 /// Construct a hit. Prefer this over a struct literal from
 /// external crates since `VectorHit` is `#[non_exhaustive]`.
 #[must_use]
 pub const fn new(node_id: NodeId, score: f32) -> Self {
 Self { node_id, score }
 }
}

/// Read-only approximate-nearest-neighbours surface for node embeddings.
///
/// Implementations bind to a single `(model, dim)` at build time.
/// `search` returns up to `k` hits in descending score order, with
/// ties broken by `NodeId` ASC for byte-stable replay.
pub trait VectorIndex: Send + Sync {
 /// Embedding model this index was built for.
 fn model(&self) -> &str;

 /// Vector dimension the index accepts on queries.
 fn dim(&self) -> u32;

 /// Nearest-neighbour lookup. Returns up to `k` hits.
 ///
 /// # Errors
 ///
 /// Returns [`RepoError::VectorDimMismatch`] if `query.len() != self.dim()`.
 fn search(&self, query: &[f32], k: usize) -> Result<Vec<VectorHit>, Error>;

 /// Number of indexed vectors.
 fn len(&self) -> usize;

 /// `true` iff no vectors were indexed.
 fn is_empty(&self) -> bool {
 self.len() == 0
 }
}

// ============================================================
// BruteForceVectorIndex
// ============================================================

/// A cosine-similarity brute-force vector index.
///
/// Stores L2-normalised f32 vectors in a flat row-major buffer for
/// cache-friendly linear scans. `search(q, k)` is `O(n * dim)` in time
/// and `O(n)` in allocations; a min-heap optimisation is not worth the
/// complexity at the corpus sizes this impl targets (see module docs).
#[derive(Debug, Clone)]
pub struct BruteForceVectorIndex {
 model: String,
 dim: u32,
 ids: Vec<NodeId>,
 /// `ids.len() * dim` f32s, row-major. Each row is L2-unit.
 data: Vec<f32>,
}

impl BruteForceVectorIndex {
 /// Construct an empty index for `(model, dim)`.
 ///
 /// Agents who want to stream `insert` rather than build from a repo
 /// can start here. The repo-scan path ([`Self::build_from_repo`])
 /// is the common case.
 #[must_use]
 pub fn empty(model: impl Into<String>, dim: u32) -> Self {
 Self {
 model: model.into(),
 dim,
 ids: Vec::new(),
 data: Vec::new(),
 }
 }

 /// Model identifier this index is bound to (e.g.
 /// `"openai:text-embedding-3-small"`). Exposed so downstream
 /// consumers (e.g. the KNN-edge derivation in mnem-http's
 /// `GraphCache`) can tag their derived artefacts with the same
 /// model string the vectors were indexed under.
 #[must_use]
 pub fn model(&self) -> &str {
 &self.model
 }

 /// Dimensionality of every stored vector. `0` iff the index was
 /// `empty()`-constructed and never inserted into.
 #[must_use]
 pub const fn dim(&self) -> u32 {
 self.dim
 }

 /// `true` iff no vectors were indexed.
 #[must_use]
 pub fn is_empty(&self) -> bool {
 self.ids.is_empty()
 }

 /// Iterate `(node_id, unit_vector_slice)` pairs in build order
 /// (canonical Prolly-key order at build time). The returned slice
 /// is borrowed from the flat row-major buffer; every row is
 /// already L2-normalised so cosine == dot product.
 ///
 /// Used by mnem-http's `GraphCache` KNN-edge fallback to derive
 /// a deterministic KNN-edge substrate when the authored-edges
 /// adjacency is empty (experiment E0 wire activation). Returning a
 /// borrowed slice avoids the per-row `to_vec()` clone the HNSW
 /// variant pays.
 pub fn points_iter(&self) -> impl Iterator<Item = (NodeId, &[f32])> + '_ {
 let row_len = self.dim as usize;
 self.ids.iter().enumerate().map(move |(i, id)| {
 // dim can be 0 only if empty, in which case ids is also
 // empty and this closure is never invoked.
 let slice = if row_len == 0 {
 &[][..]
 } else {
 &self.data[i * row_len..(i + 1) * row_len]
 };
 (*id, slice)
 })
 }

 /// Insert one node's embedding. The node's embedding MUST match
 /// `self.model` and `self.dim`; mismatched entries are silently
 /// skipped so callers can feed a heterogeneous stream.
 ///
 /// Returns `true` if the vector was indexed, `false` if it was
 /// skipped (wrong model, wrong dim, absent, or undecodable).
 pub fn try_insert(&mut self, node_id: NodeId, embed: &Embedding) -> bool {
 if embed.model != self.model {
 return false;
 }
 if embed.dim != self.dim {
 return false;
 }
 let Some(vec_f32) = decode_to_f32(embed) else {
 return false;
 };
 let normalised = normalise(vec_f32);
 self.ids.push(node_id);
 self.data.extend_from_slice(&normalised);
 true
 }

 /// Build an index over every node at the repo head whose
 /// embedding under `model` is present in the per-commit sidecar
 /// (`Commit.embeddings` Prolly tree, keyed by `NodeCid`). Nodes
 /// without a sidecar entry under `model` are silently skipped.
 ///
 /// The sidecar is the only source of truth: dense vectors live
 /// in a separate tree so nondeterministic producers (e.g. ORT
 /// thread-count drift) cannot leak into `NodeCid` and break
 /// federated dedup. Operators with repos authored before the
 /// sidecar shipped must run `mnem reindex` to lift inline
 /// vectors into the sidecar; until then those vectors are
 /// invisible to retrieval.
 ///
 /// # Errors
 ///
 /// - [`RepoError::Uninitialized`] if the repo has no head commit.
 /// - Store / codec errors walking the node tree, decoding nodes,
 /// or walking the embedding sidecar.
 /// - [`crate::error::ObjectError::EmbeddingSizeMismatch`] if a node
 /// carries an embedding whose `vector.len()` contradicts
 /// `dim * bytes_per_dtype(dtype)`.
 pub fn build_from_repo(repo: &ReadonlyRepo, model: &str) -> Result<Self, Error> {
 let bs: Arc<dyn Blockstore> = repo.blockstore().clone();
 let Some(commit) = repo.head_commit() else {
 return Err(RepoError::Uninitialized.into());
 };

 // Single pass: decide the index dim lazily from the first
 // matching embedding, then keep inserting in the same walk.
 // Skips the second `Cursor::new + decode every node` round-trip
 // the two-pass version paid.
 let mut idx: Option<Self> = None;
 let debug = std::env::var("MNEM_DEBUG_VEC").is_ok();
 let mut dbg_total = 0usize;
 let mut dbg_has_embed = 0usize;
 let mut dbg_inserted = 0usize;
 let cursor = Cursor::new(&*bs, &commit.nodes)?;
 for entry in cursor {
 let (_k, node_cid) = entry?;
 let node: Node = decode_from_store(&*bs, &node_cid)?;
 dbg_total += 1;

 // Sidecar is the only source. The bucket may exist but
 // lack `model`; that is indistinguishable from a missing
 // bucket and skips the node.
 let Some(embed) = repo.embedding_for(&node_cid, model)? else {
 continue;
 };
 dbg_has_embed += 1;
 if debug && dbg_has_embed <= 3 {
 eprintln!(
 "[mnem-debug-vec] node embed.model={:?} want={:?} dim={}",
 embed.model, model, embed.dim,
 );
 }
 embed.validate()?;
 let ok = match idx.as_mut() {
 Some(existing) => existing.try_insert(node.id, &embed),
 None => {
 let mut fresh = Self::empty(model, embed.dim);
 let ok = fresh.try_insert(node.id, &embed);
 idx = Some(fresh);
 ok
 }
 };
 if ok {
 dbg_inserted += 1;
 }
 }
 if debug {
 eprintln!(
 "[mnem-debug-vec] total={dbg_total} has_embed={dbg_has_embed} \
 inserted={dbg_inserted} idx_dim={}",
 idx.as_ref().map_or(0, |i| i.dim)
 );
 }
 // No matching embeddings: empty index rather than an error.
 // Agents treat "empty" as "no matches."
 Ok(idx.unwrap_or_else(|| Self::empty(model, 0)))
 }
}

impl VectorIndex for BruteForceVectorIndex {
 fn model(&self) -> &str {
 &self.model
 }

 fn dim(&self) -> u32 {
 self.dim
 }

 fn search(&self, query: &[f32], k: usize) -> Result<Vec<VectorHit>, Error> {
 // Unconfigured-model short-circuit: when `build_from_repo`
 // found no nodes matching the requested model, the index is
 // `Self::empty(model, 0)` with `dim == 0`. The caller's query
 // is legitimately sized for a real model; returning zero hits
 // instead of `VectorDimMismatch` preserves the "unconfigured
 // model = empty ranker = empty result" contract that
 // Retriever::execute relies on.
 if self.dim == 0 && self.ids.is_empty() {
 return Ok(Vec::new());
 }
 if query.len() != self.dim as usize {
 return Err(RepoError::VectorDimMismatch {
 index_dim: self.dim,
 query_dim: query.len(),
 }
 .into());
 }
 if k == 0 || self.ids.is_empty() {
 return Ok(Vec::new());
 }

 let q_norm = normalise(query.to_vec());
 let row_len = self.dim as usize;
 let mut hits: Vec<VectorHit> = Vec::with_capacity(self.ids.len());
 for (i, id) in self.ids.iter().enumerate() {
 let row = &self.data[i * row_len..(i + 1) * row_len];
 let score = dot(&q_norm, row);
 hits.push(VectorHit {
 node_id: *id,
 score,
 });
 }
 // Score DESC; ties broken by NodeId ASC for determinism.
 hits.sort_by(|a, b| {
 b.score
 .partial_cmp(&a.score)
 .unwrap_or(std::cmp::Ordering::Equal)
 .then_with(|| a.node_id.cmp(&b.node_id))
 });
 hits.truncate(k);
 Ok(hits)
 }

 fn len(&self) -> usize {
 self.ids.len()
 }
}

// ============================================================
// Math + dtype decoding helpers
// ============================================================

/// Decode an [`Embedding`] vector into a `Vec<f32>`. Returns `None` on
/// byte-length inconsistencies (caller SHOULD have pre-validated via
/// `Embedding::validate`).
fn decode_to_f32(embed: &Embedding) -> Option<Vec<f32>> {
 let dim = embed.dim as usize;
 let bytes: &Bytes = &embed.vector;
 if bytes.len() != dim * embed.dtype.byte_width() {
 return None;
 }
 match embed.dtype {
 Dtype::F32 => {
 let mut out = Vec::with_capacity(dim);
 for chunk in bytes.chunks_exact(4) {
 out.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
 }
 Some(out)
 }
 Dtype::F64 => {
 let mut out = Vec::with_capacity(dim);
 for chunk in bytes.chunks_exact(8) {
 let raw = f64::from_le_bytes([
 chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
 ]);
 out.push(raw as f32);
 }
 Some(out)
 }
 Dtype::F16 => {
 // IEEE 754 half-precision decoded by hand to avoid pulling
 // the `half` crate for one call site. See IEEE 754-2008 §3.6.
 let mut out = Vec::with_capacity(dim);
 for chunk in bytes.chunks_exact(2) {
 let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
 out.push(f16_bits_to_f32(bits));
 }
 Some(out)
 }
 Dtype::I8 => {
 // Quantised i8 is treated as already scaled: [-128, 127]
 // mapped linearly to f32. Agents that use a more elaborate
 // per-vector scale should pre-decode.
 let mut out = Vec::with_capacity(dim);
 for &b in bytes {
 out.push(f32::from(i8::from_ne_bytes([b])));
 }
 Some(out)
 }
 }
}

/// Decode one IEEE-754 half-precision value to f32. Handles subnormals,
/// zero, infinity, and NaN.
fn f16_bits_to_f32(bits: u16) -> f32 {
 let sign = u32::from(bits >> 15) << 31;
 let exp = u32::from((bits >> 10) & 0x1F);
 let mant = u32::from(bits & 0x3FF);
 let out_bits = if exp == 0 {
 if mant == 0 {
 sign
 } else {
 // Subnormal: normalise into f32's wider exponent range.
 // `e` stays non-negative for the full f16 subnormal range
 // (smallest requires 10 shifts from 113, landing at 103).
 let mut m = mant;
 let mut e: u32 = 127 - 15 + 1;
 while (m & 0x400) == 0 {
 m <<= 1;
 e = e.saturating_sub(1);
 }
 m &= 0x3FF;
 sign | (e << 23) | (m << 13)
 }
 } else if exp == 31 {
 // Inf / NaN: copy mantissa, saturate exponent.
 sign | 0x7F80_0000 | (mant << 13)
 } else {
 let e = exp + (127 - 15);
 sign | (e << 23) | (mant << 13)
 };
 f32::from_bits(out_bits)
}

/// L2-normalise a vector in place and return it. A zero vector is
/// returned unchanged (cosine similarity against it is 0).
fn normalise(mut v: Vec<f32>) -> Vec<f32> {
 let norm = dot(&v, &v).sqrt();
 if norm > 0.0 && norm.is_finite() {
 for x in &mut v {
 *x /= norm;
 }
 }
 v
}

/// Dot product of two slices. Debug-asserts equal length; callers
/// guarantee the invariant upstream.
fn dot(a: &[f32], b: &[f32]) -> f32 {
 debug_assert_eq!(a.len(), b.len());
 let mut acc = 0.0f32;
 for i in 0..a.len() {
 acc += a[i] * b[i];
 }
 acc
}

// ============================================================
// Tests
// ============================================================

#[cfg(test)]
mod tests {
 use super::*;
 use crate::objects::{Dtype, Embedding, Node};
 use crate::repo::ReadonlyRepo;
 use crate::store::{MemoryBlockstore, MemoryOpHeadsStore, OpHeadsStore};
 use std::sync::Arc;

 fn stores() -> (Arc<dyn Blockstore>, Arc<dyn OpHeadsStore>) {
 (
 Arc::new(MemoryBlockstore::new()),
 Arc::new(MemoryOpHeadsStore::new()),
 )
 }

 fn f32_embed(model: &str, v: &[f32]) -> Embedding {
 let mut bytes = Vec::with_capacity(v.len() * 4);
 for x in v {
 bytes.extend_from_slice(&x.to_le_bytes());
 }
 Embedding {
 model: model.to_string(),
 dtype: Dtype::F32,
 dim: v.len() as u32,
 vector: Bytes::from(bytes),
 }
 }

 // ---------- Math helpers ----------

 #[test]
 fn normalise_unit_vector_is_unchanged() {
 let v = normalise(vec![1.0, 0.0, 0.0]);
 assert!((dot(&v, &v) - 1.0).abs() < 1e-6);
 }

 #[test]
 fn normalise_scales_to_unit_length() {
 let v = normalise(vec![3.0, 4.0]);
 assert!((dot(&v, &v) - 1.0).abs() < 1e-6);
 }

 #[test]
 fn normalise_zero_vector_stays_zero() {
 let v = normalise(vec![0.0, 0.0, 0.0]);
 assert_eq!(v, vec![0.0, 0.0, 0.0]);
 }

 #[test]
 fn f16_round_trip_for_common_values() {
 // 1.0 in f16 is bits 0x3C00.
 assert!((f16_bits_to_f32(0x3C00) - 1.0).abs() < 1e-6);
 // -1.0 is 0xBC00.
 assert!((f16_bits_to_f32(0xBC00) + 1.0).abs() < 1e-6);
 // +0 / -0
 assert_eq!(f16_bits_to_f32(0x0000), 0.0);
 assert_eq!(f16_bits_to_f32(0x8000), -0.0);
 // +inf
 assert!(f16_bits_to_f32(0x7C00).is_infinite());
 }

 // ---------- Empty + trivial ----------

 #[test]
 fn empty_index_returns_no_hits() {
 let idx = BruteForceVectorIndex::empty("m", 4);
 let hits = idx.search(&[0.0, 0.0, 0.0, 0.0], 5).unwrap();
 assert!(hits.is_empty());
 assert_eq!(idx.len(), 0);
 assert!(idx.is_empty());
 }

 #[test]
 fn k_zero_returns_no_hits() {
 let mut idx = BruteForceVectorIndex::empty("m", 3);
 idx.try_insert(
 NodeId::from_bytes_raw([1u8; 16]),
 &f32_embed("m", &[1.0, 0.0, 0.0]),
 );
 let hits = idx.search(&[1.0, 0.0, 0.0], 0).unwrap();
 assert!(hits.is_empty());
 }

 // ---------- Dim mismatch ----------

 #[test]
 fn dim_mismatch_errors_with_both_sides() {
 let idx = BruteForceVectorIndex::empty("m", 4);
 let err = idx.search(&[0.0, 0.0, 0.0], 3).unwrap_err();
 match err {
 Error::Repo(RepoError::VectorDimMismatch {
 index_dim,
 query_dim,
 }) => {
 assert_eq!(index_dim, 4);
 assert_eq!(query_dim, 3);
 }
 e => panic!("expected VectorDimMismatch, got {e:?}"),
 }
 }

 // ---------- Model scoping ----------

 #[test]
 fn wrong_model_is_silently_skipped_on_insert() {
 let mut idx = BruteForceVectorIndex::empty("mA", 3);
 let inserted = idx.try_insert(
 NodeId::from_bytes_raw([1u8; 16]),
 &f32_embed("mB", &[1.0, 0.0, 0.0]),
 );
 assert!(!inserted);
 assert!(idx.is_empty());
 }

 #[test]
 fn wrong_dim_is_silently_skipped_on_insert() {
 let mut idx = BruteForceVectorIndex::empty("m", 3);
 let inserted = idx.try_insert(
 NodeId::from_bytes_raw([1u8; 16]),
 &f32_embed("m", &[1.0, 0.0]),
 );
 assert!(!inserted);
 }

 // ---------- Ranking ----------

 #[test]
 fn nearest_neighbour_wins() {
 let mut idx = BruteForceVectorIndex::empty("m", 3);
 idx.try_insert(
 NodeId::from_bytes_raw([1u8; 16]),
 &f32_embed("m", &[1.0, 0.0, 0.0]),
 );
 idx.try_insert(
 NodeId::from_bytes_raw([2u8; 16]),
 &f32_embed("m", &[0.0, 1.0, 0.0]),
 );
 idx.try_insert(
 NodeId::from_bytes_raw([3u8; 16]),
 &f32_embed("m", &[0.0, 0.0, 1.0]),
 );
 let hits = idx.search(&[0.9, 0.1, 0.0], 3).unwrap();
 assert_eq!(hits[0].node_id, NodeId::from_bytes_raw([1u8; 16]));
 // Second should be the [0,1,0] vector (closer than [0,0,1]).
 assert_eq!(hits[1].node_id, NodeId::from_bytes_raw([2u8; 16]));
 // Cosine similarity to an orthogonal axis is 0.
 assert_eq!(hits[2].node_id, NodeId::from_bytes_raw([3u8; 16]));
 assert!((hits[2].score).abs() < 1e-6);
 }

 #[test]
 fn scale_invariance_cosine_similarity() {
 // Two co-linear vectors produce cosine similarity ~1.0
 // regardless of magnitude.
 let mut idx = BruteForceVectorIndex::empty("m", 3);
 idx.try_insert(
 NodeId::from_bytes_raw([1u8; 16]),
 &f32_embed("m", &[10.0, 0.0, 0.0]),
 );
 let hits = idx.search(&[0.5, 0.0, 0.0], 1).unwrap();
 assert!((hits[0].score - 1.0).abs() < 1e-5);
 }

 #[test]
 fn k_truncates_results() {
 let mut idx = BruteForceVectorIndex::empty("m", 2);
 for i in 0..20u8 {
 idx.try_insert(
 NodeId::from_bytes_raw([i; 16]),
 &f32_embed("m", &[f32::from(i), 1.0]),
 );
 }
 let hits = idx.search(&[1.0, 1.0], 5).unwrap();
 assert_eq!(hits.len(), 5);
 }

 #[test]
 fn ties_broken_by_node_id_ascending() {
 let mut idx = BruteForceVectorIndex::empty("m", 2);
 let hi = NodeId::from_bytes_raw([0xFFu8; 16]);
 let lo = NodeId::from_bytes_raw([0x01u8; 16]);
 idx.try_insert(hi, &f32_embed("m", &[1.0, 0.0]));
 idx.try_insert(lo, &f32_embed("m", &[1.0, 0.0]));
 let hits = idx.search(&[1.0, 0.0], 2).unwrap();
 assert_eq!(hits[0].node_id, lo);
 assert_eq!(hits[1].node_id, hi);
 }

 // ---------- Dtype decoding ----------

 #[test]
 fn f64_embeddings_are_indexed() {
 let mut bytes = Vec::new();
 for x in &[1.0f64, 0.0, 0.0] {
 bytes.extend_from_slice(&x.to_le_bytes());
 }
 let embed = Embedding {
 model: "m".into(),
 dtype: Dtype::F64,
 dim: 3,
 vector: Bytes::from(bytes),
 };
 let mut idx = BruteForceVectorIndex::empty("m", 3);
 assert!(idx.try_insert(NodeId::from_bytes_raw([1u8; 16]), &embed));
 let hits = idx.search(&[1.0, 0.0, 0.0], 1).unwrap();
 assert!((hits[0].score - 1.0).abs() < 1e-5);
 }

 #[test]
 fn i8_embeddings_are_indexed() {
 let bytes: Vec<u8> = vec![127, 0, 0].into_iter().map(|v: i8| v as u8).collect();
 let embed = Embedding {
 model: "m".into(),
 dtype: Dtype::I8,
 dim: 3,
 vector: Bytes::from(bytes),
 };
 let mut idx = BruteForceVectorIndex::empty("m", 3);
 assert!(idx.try_insert(NodeId::from_bytes_raw([1u8; 16]), &embed));
 let hits = idx.search(&[1.0, 0.0, 0.0], 1).unwrap();
 // i8 127 normalises to ~1.0 along x.
 assert!((hits[0].score - 1.0).abs() < 1e-5);
 }

 #[test]
 fn f16_embeddings_are_indexed() {
 // Encode [1.0, 0.0] in f16.
 let bytes: Vec<u8> = vec![0x00, 0x3C, 0x00, 0x00];
 let embed = Embedding {
 model: "m".into(),
 dtype: Dtype::F16,
 dim: 2,
 vector: Bytes::from(bytes),
 };
 let mut idx = BruteForceVectorIndex::empty("m", 2);
 assert!(idx.try_insert(NodeId::from_bytes_raw([1u8; 16]), &embed));
 let hits = idx.search(&[1.0, 0.0], 1).unwrap();
 assert!((hits[0].score - 1.0).abs() < 1e-5);
 }

 // ---------- build_from_repo integration ----------

 #[test]
 fn build_from_repo_indexes_only_matching_model() {
 let (bs, ohs) = stores();
 let repo = ReadonlyRepo::init(bs, ohs).unwrap();
 let mut tx = repo.start_transaction();

 let mut add = |id: [u8; 16], model: &str, v: &[f32]| {
 let node = Node::new(NodeId::from_bytes_raw(id), "Doc");
 let cid = tx.add_node(&node).unwrap();
 let emb = f32_embed(model, v);
 tx.set_embedding(cid, emb.model.clone(), emb).unwrap();
 };
 add([1u8; 16], "mA", &[1.0, 0.0]);
 add([2u8; 16], "mA", &[0.0, 1.0]);
 add([3u8; 16], "mB", &[1.0, 0.0]);
 tx.add_node(&Node::new(NodeId::from_bytes_raw([4u8; 16]), "Doc")) // no embed
 .unwrap();
 let repo = tx.commit("t", "seed").unwrap();

 let idx = BruteForceVectorIndex::build_from_repo(&repo, "mA").unwrap();
 assert_eq!(idx.len(), 2);
 assert_eq!(idx.dim(), 2);
 assert_eq!(idx.model(), "mA");

 let hits = idx.search(&[1.0, 0.0], 2).unwrap();
 assert_eq!(hits[0].node_id, NodeId::from_bytes_raw([1u8; 16]));
 }

 #[test]
 fn build_for_absent_model_returns_empty_index() {
 let (bs, ohs) = stores();
 let repo = ReadonlyRepo::init(bs, ohs).unwrap();
 let mut tx = repo.start_transaction();
 let cid = tx
 .add_node(&Node::new(NodeId::from_bytes_raw([1u8; 16]), "Doc"))
 .unwrap();
 let emb = f32_embed("mA", &[1.0, 0.0]);
 tx.set_embedding(cid, emb.model.clone(), emb).unwrap();
 let repo = tx.commit("t", "seed").unwrap();

 let idx = BruteForceVectorIndex::build_from_repo(&repo, "unknown").unwrap();
 assert!(idx.is_empty());
 assert_eq!(idx.model(), "unknown");
 }

 #[test]
 fn build_on_empty_repo_errors() {
 let (bs, ohs) = stores();
 let repo = ReadonlyRepo::init(bs, ohs).unwrap();
 let err = BruteForceVectorIndex::build_from_repo(&repo, "mA").unwrap_err();
 match err {
 Error::Repo(RepoError::Uninitialized) => {}
 e => panic!("expected Uninitialized, got {e:?}"),
 }
 }

 #[test]
 fn determinism_same_repo_same_results() {
 let build = || {
 let (bs, ohs) = stores();
 let repo = ReadonlyRepo::init(bs, ohs).unwrap();
 let mut tx = repo.start_transaction();
 for i in 0..5u8 {
 let cid = tx
 .add_node(&Node::new(NodeId::from_bytes_raw([i; 16]), "Doc"))
 .unwrap();
 let emb = f32_embed("m", &[f32::from(i), 1.0]);
 tx.set_embedding(cid, emb.model.clone(), emb).unwrap();
 }
 let repo = tx.commit("t", "seed").unwrap();
 let idx = BruteForceVectorIndex::build_from_repo(&repo, "m").unwrap();
 idx.search(&[2.0, 1.0], 3).unwrap()
 };
 let a = build();
 let b = build();
 assert_eq!(a, b, "same inputs -> byte-identical hit list");
 }

 // ---------- sidecar dual-read ----------

 /// Sidecar is the source of truth: a node added without
 /// `node.embed` whose vector lives only in the
 /// `Commit.embeddings` Prolly tree must still surface in the
 /// index. Verifies `build_from_repo` actually calls
 /// `embedding_for` rather than only reading `node.embed`.
 #[test]
 fn index_reads_embedding_from_sidecar() {
 let (bs, ohs) = stores();
 let repo = ReadonlyRepo::init(bs, ohs).unwrap();
 let mut tx = repo.start_transaction();

 // Node carries NO inline embed: the only path to retrieval is
 // the sidecar. If the dual-read regressed and only `node.embed`
 // is consulted, this test fails with `is_empty()`.
 let node = Node::new(NodeId::from_bytes_raw([1u8; 16]), "Doc");
 let node_cid = tx.add_node(&node).unwrap();
 let emb = f32_embed("mA", &[1.0, 0.0, 0.0]);
 tx.set_embedding(node_cid, "mA".into(), emb).unwrap();
 let repo = tx.commit("t", "seed via sidecar").unwrap();

 let idx = BruteForceVectorIndex::build_from_repo(&repo, "mA").unwrap();
 assert_eq!(idx.len(), 1, "sidecar embedding must surface in the index");
 assert_eq!(idx.dim(), 3);
 let hits = idx.search(&[1.0, 0.0, 0.0], 1).unwrap();
 assert_eq!(hits[0].node_id, NodeId::from_bytes_raw([1u8; 16]));
 assert!((hits[0].score - 1.0).abs() < 1e-5);
 }

}
