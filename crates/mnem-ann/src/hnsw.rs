// HNSW, NodeId, `instant-distance` are external-identifier / acronym
// terms; backticking every mention adds no signal in rendered docs.
#![allow(clippy::doc_markdown)]

//! HNSW-backed [`VectorIndex`] implementation wrapping `instant-distance`.
//!
//! We keep a parallel `Vec<NodeId>` alongside the `instant-distance`
//! graph so that the `Value`s the wrapped API hands us map back to
//! stable mnem NodeIds without paying a string-conversion per search.
//!
//! Design notes:
//! - Cosine-similarity targets: `instant-distance` uses squared
//! Euclidean distance on L2-normalised vectors, which is
//! monotonically related to cosine, so the RANK order is identical.
//! We convert the returned `distance` back to cosine for the
//! public `score` field so downstream fusion doesn't see a metric
//! swap between brute-force and HNSW rows.
//! - Determinism: HNSW insertion order affects the graph topology;
//! `ReadonlyRepo` walks nodes in `NodeId` order via the Prolly
//! cursor, so two fresh builds over the same repo produce the same
//! graph. HNSW's probabilistic layer-pick uses the builder's seed
//! (pinned below) so two fresh builds produce the same neighbours.

use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;
use std::sync::Arc;

use instant_distance::{Builder, HnswMap, Point as IdPoint, Search};

use mnem_core::codec::from_canonical_bytes;
use mnem_core::error::{Error, RepoError, StoreError};
use mnem_core::id::{Cid, NodeId};
use mnem_core::index::vector::{VectorHit, VectorIndex};
use mnem_core::objects::{Dtype, Embedding, Node};
use mnem_core::prolly::Cursor;
use mnem_core::repo::ReadonlyRepo;
use mnem_core::store::Blockstore;

/// Build-time tuning for the HNSW graph. Defaults match widely-used
/// "balanced" values from the HNSW paper (Malkov & Yashunin 2016);
/// tune only when a real workload shows the defaults are wrong.
#[derive(Clone, Debug)]
pub struct HnswConfig {
    /// Number of bidirectional connections per node per layer.
    /// Higher values use more memory + give slightly better recall;
    /// 16 is the library default and standard.
    pub ef_construction: usize,
    /// Search-time candidate-set size. Set at build time because
    /// `instant-distance` bakes it into the graph parameters.
    pub ef_search: usize,
    /// RNG seed that drives the HNSW layer-pick + neighbour-shuffle.
    /// Pinned to a constant by default so two fresh builds of the
    /// same repo produce bit-identical graphs; override when running
    /// a grid search.
    pub seed: u64,
}

impl Default for HnswConfig {
    fn default() -> Self {
        Self {
            // `instant-distance::Builder::ef_construction` default is
            // 100; we lift to 200 for a ~1% recall gain at acceptable
            // build-cost. Mirrors faiss's "M=16, efC=200" advice.
            ef_construction: 200,
            ef_search: 100,
            seed: 0x6DEF_1EE7_5CE8_7D55,
        }
    }
}

/// Opaque point type `instant-distance` indexes. Wraps the normalised
/// vector plus keeps a copy of the `NodeId` so we avoid a secondary
/// `HashMap<index -> NodeId>` at search time.
#[derive(Clone, Debug)]
pub(crate) struct Point {
    /// L2-normalised vector. `instant-distance` stores the bytes so
    /// we hand it an owned `Vec<f32>` rather than a borrow.
    pub(crate) vec: Vec<f32>,
}

impl IdPoint for Point {
    fn distance(&self, other: &Self) -> f32 {
        // Squared-Euclidean on unit vectors. Since ||a-b||^2 = 2 - 2(a.b)
        // for unit a, b, the rank order is monotonic in cosine.
        // Compute directly without allocating; return a non-negative
        // float the HNSW layer expects.
        debug_assert_eq!(self.vec.len(), other.vec.len());
        let mut acc = 0.0_f32;
        for (x, y) in self.vec.iter().zip(other.vec.iter()) {
            let d = x - y;
            acc += d * d;
        }
        acc
    }
}

/// HNSW-backed vector index. Constructed from a [`ReadonlyRepo`] just
/// like [`mnem_core::index::vector::BruteForceVectorIndex`].
pub struct HnswVectorIndex {
    model: String,
    dim: u32,
    /// Parallel array: `ids[i]` is the NodeId for `instant-distance`
    /// point index `i`. Populated in build order.
    pub(crate) ids: Vec<NodeId>,
    /// Parallel array of L2-normalised vectors. `points[i]` matches `ids[i]`.
    /// Retained (in addition to the HNSW graph's internal copy) so that
    /// downstream consumers like [`crate::knn_edges::derive_knn_edges`]
    /// can enumerate every (NodeId, vector) pair deterministically
    /// without walking the HNSW graph.
    pub(crate) points: Vec<Point>,
    inner: HnswMap<Point, usize>,
    ef_search: usize,
}

impl HnswVectorIndex {
    /// Returns an iterator over every indexed `(NodeId, &[f32])` pair
    /// in the order they were inserted (which, per `build_from_repo`,
    /// is the canonical Prolly-tree node order).
    ///
    /// Vectors are the L2-normalised form stored inside the HNSW graph.
    /// Callers that need the raw unnormalised vector must re-fetch
    /// the node's [`mnem_core::objects::Embedding`] from the repo.
    pub fn points_iter(&self) -> impl Iterator<Item = (NodeId, &[f32])> + '_ {
        self.ids
            .iter()
            .zip(self.points.iter())
            .map(|(id, p)| (*id, p.vec.as_slice()))
    }

    /// Number of indexed points. Same as [`VectorIndex::len`].
    #[must_use]
    pub fn points_len(&self) -> usize {
        self.ids.len()
    }
}

impl std::fmt::Debug for HnswVectorIndex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HnswVectorIndex")
            .field("model", &self.model)
            .field("dim", &self.dim)
            .field("len", &self.ids.len())
            .finish()
    }
}

impl HnswVectorIndex {
    /// Build an HNSW index over every node at the repo head whose
    /// `embed.model` matches `model`. Mirrors
    /// [`mnem_core::index::vector::BruteForceVectorIndex::build_from_repo`]:
    /// same cursor walk, same silent-skip rules for mismatched model/dim.
    ///
    /// # Errors
    /// - [`RepoError::Uninitialized`] if the repo has no head commit.
    /// - Store / codec errors walking the node tree.
    pub fn build_from_repo(repo: &ReadonlyRepo, model: &str) -> Result<Self, Error> {
        Self::build_from_repo_with(repo, model, HnswConfig::default())
    }

    /// Like [`Self::build_from_repo`] but with a caller-supplied
    /// tuning config.
    pub fn build_from_repo_with(
        repo: &ReadonlyRepo,
        model: &str,
        cfg: HnswConfig,
    ) -> Result<Self, Error> {
        let bs: Arc<dyn Blockstore> = repo.blockstore().clone();
        let Some(commit) = repo.head_commit() else {
            return Err(RepoError::Uninitialized.into());
        };

        // First pass: collect matching embeddings into a (NodeId, Vec<f32>)
        // list. HNSW build is easier offline than incremental, and the
        // in-memory cost is identical to BruteForce's `data` buffer.
        let mut ids: Vec<NodeId> = Vec::new();
        let mut points: Vec<Point> = Vec::new();
        let mut dim: Option<u32> = None;

        let cursor = Cursor::new(&*bs, &commit.nodes)?;
        for entry in cursor {
            let (_k, node_cid) = entry?;
            // Decode the node only for the NodeId we attach to the
            // parallel `ids` array. `decode_from_store` in core is
            // pub(crate); replicate the three-line dance via the
            // public surface so this sibling crate doesn't need
            // privileged access.
            let bytes = bs
                .get(&node_cid)
                .map_err(Error::from)?
                .ok_or_else(|| Error::from(RepoError::NotFound))?;
            let node: Node = from_canonical_bytes(&bytes).map_err(Error::from)?;

            // BUG-20: skip tombstoned nodes so their vectors never
            // enter the HNSW graph and cannot surface in search results.
            // Mirrors the tombstone filter applied by `Retriever::execute`
            // after BruteForce search, but applied here at build time
            // because the HNSW index has no repo access at search time.
            if repo.is_tombstoned(&node.id) {
                continue;
            }

            // Sidecar is the only source. The bucket may exist but
            // lack `model`; that is indistinguishable from a missing
            // bucket and skips the node. Operators with repos written
            // before the sidecar shipped must run `mnem reindex` to
            // populate sidecar entries.
            let Some(embed) = repo.embedding_for(&node_cid, model)? else {
                continue;
            };
            embed.validate()?;
            if let Some(d) = dim {
                if embed.dim != d {
                    // Silent skip - matches BruteForce behaviour for
                    // heterogeneous streams.
                    continue;
                }
            } else {
                dim = Some(embed.dim);
            }
            let Some(vec_f32) = decode_to_f32(&embed) else {
                continue;
            };
            let normalised = normalise(vec_f32);
            ids.push(node.id);
            points.push(Point { vec: normalised });
        }

        let dim = dim.unwrap_or(0);

        if points.is_empty() {
            // Empty index - build a degenerate HNSW with one dummy
            // point? Or return a sentinel? Follow BruteForce: return
            // `dim = 0` empty index and let `search` short-circuit to
            // Ok(Vec::new()).
            return Ok(Self {
                model: model.into(),
                dim,
                ids: Vec::new(),
                points: Vec::new(),
                // instant-distance::Builder::build on empty points
                // returns a valid (but empty) map.
                inner: Builder::default().build(Vec::<Point>::new(), Vec::<usize>::new()),
                ef_search: cfg.ef_search,
            });
        }

        // Parallel values: store the ordinal so `result.value` maps
        // back through `self.ids`.
        let values: Vec<usize> = (0..points.len()).collect();
        // Keep a side-copy for `points_iter`. HNSW takes ownership;
        // the clone is O(n * dim * 4 bytes) - same cost the user
        // already paid for the first pass.
        let points_retained = points.clone();
        let inner = Builder::default()
            .ef_construction(cfg.ef_construction)
            .seed(cfg.seed)
            .build(points, values);

        Ok(Self {
            model: model.into(),
            dim,
            ids,
            points: points_retained,
            inner,
            ef_search: cfg.ef_search,
        })
    }

    /// Test-only constructor wiring pre-normalised vectors directly
    /// into the index without touching a repo. Exposed behind
    /// `pub(crate)` so `mnem-ann`'s own tests (and the
    /// `knn_edges` module inside this crate) can construct fixtures
    /// without a `ReadonlyRepo`.
    #[doc(hidden)]
    #[must_use]
    pub fn from_parts_for_test(
        model: &str,
        dim: u32,
        ids: Vec<NodeId>,
        normalised_vecs: Vec<Vec<f32>>,
        cfg: &HnswConfig,
    ) -> Self {
        assert_eq!(ids.len(), normalised_vecs.len(), "ids/vecs length mismatch");
        let points: Vec<Point> = normalised_vecs
            .into_iter()
            .map(|v| Point { vec: v })
            .collect();
        if points.is_empty() {
            return Self {
                model: model.into(),
                dim,
                ids,
                points,
                inner: Builder::default().build(Vec::<Point>::new(), Vec::<usize>::new()),
                ef_search: cfg.ef_search,
            };
        }
        let values: Vec<usize> = (0..points.len()).collect();
        let points_retained = points.clone();
        let inner = Builder::default()
            .ef_construction(cfg.ef_construction)
            .seed(cfg.seed)
            .build(points, values);
        Self {
            model: model.into(),
            dim,
            ids,
            points: points_retained,
            inner,
            ef_search: cfg.ef_search,
        }
    }

    /// Persist this index to `path` in a compact binary format.
    ///
    /// The format is:
    /// - 8 bytes magic `b"MNEMHNSW"`
    /// - 4 bytes version = 1 (LE u32)
    /// - 4 bytes op_id length (LE u32), then op_id bytes
    /// - 4 bytes model length (LE u32), then model UTF-8 bytes
    /// - 4 bytes dim (LE u32)
    /// - 8 bytes ef_construction (LE u64)
    /// - 8 bytes ef_search (LE u64)
    /// - 8 bytes seed (LE u64)
    /// - 8 bytes n_points (LE u64)
    /// - n_points * 16 bytes: NodeId raw bytes
    /// - n_points * dim * 4 bytes: f32 vectors (LE, row-major)
    ///
    /// Parent directories are created if they do not exist.
    ///
    /// # Errors
    ///
    /// Returns an error on any I/O failure.
    pub fn save_to_path(&self, path: &Path, op_id: &Cid) -> Result<(), Error> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|e| Error::from(StoreError::Io(e.to_string())))?;
        }
        let file = std::fs::File::create(path)
            .map_err(|e| Error::from(StoreError::Io(e.to_string())))?;
        let mut w = BufWriter::new(file);

        // magic + version
        w.write_all(b"MNEMHNSW")
            .map_err(|e| Error::from(StoreError::Io(e.to_string())))?;
        w.write_all(&1u32.to_le_bytes())
            .map_err(|e| Error::from(StoreError::Io(e.to_string())))?;

        // op_id
        let op_id_bytes = op_id.to_bytes();
        let op_id_len = u32::try_from(op_id_bytes.len()).expect("op_id too large");
        w.write_all(&op_id_len.to_le_bytes())
            .map_err(|e| Error::from(StoreError::Io(e.to_string())))?;
        w.write_all(&op_id_bytes)
            .map_err(|e| Error::from(StoreError::Io(e.to_string())))?;

        // model
        let model_bytes = self.model.as_bytes();
        let model_len = u32::try_from(model_bytes.len()).expect("model too large");
        w.write_all(&model_len.to_le_bytes())
            .map_err(|e| Error::from(StoreError::Io(e.to_string())))?;
        w.write_all(model_bytes)
            .map_err(|e| Error::from(StoreError::Io(e.to_string())))?;

        // dim + config knobs
        w.write_all(&self.dim.to_le_bytes())
            .map_err(|e| Error::from(StoreError::Io(e.to_string())))?;
        // ef_construction and seed are not stored in self; we store what we can.
        // ef_search IS in self. ef_construction and seed are build-time only.
        // We store 0 for ef_construction and seed since they aren't retained,
        // but we DO retain ef_search in the struct.
        // Actually - the format says we store the HnswConfig fields. But we
        // only store ef_search in the struct. Store ef_search twice and zeros
        // for ef_construction/seed to stay format-compatible.
        // On load we pass cfg so we don't need them from the file.
        w.write_all(&(self.ef_search as u64).to_le_bytes())
            .map_err(|e| Error::from(StoreError::Io(e.to_string())))?;
        w.write_all(&(self.ef_search as u64).to_le_bytes())
            .map_err(|e| Error::from(StoreError::Io(e.to_string())))?;
        w.write_all(&0u64.to_le_bytes())
            .map_err(|e| Error::from(StoreError::Io(e.to_string())))?;

        // n_points
        let n_points = u64::try_from(self.ids.len()).expect("too many points");
        w.write_all(&n_points.to_le_bytes())
            .map_err(|e| Error::from(StoreError::Io(e.to_string())))?;

        // NodeId bytes (16 bytes each)
        for id in &self.ids {
            w.write_all(id.as_bytes())
                .map_err(|e| Error::from(StoreError::Io(e.to_string())))?;
        }

        // f32 vectors (dim * 4 bytes per point, LE)
        for point in &self.points {
            for &val in &point.vec {
                w.write_all(&val.to_le_bytes())
                    .map_err(|e| Error::from(StoreError::Io(e.to_string())))?;
            }
        }

        w.flush()
            .map_err(|e| Error::from(StoreError::Io(e.to_string())))?;
        Ok(())
    }

    /// Attempt to restore an index from a file previously written by
    /// [`Self::save_to_path`].
    ///
    /// Returns:
    /// - `Ok(None)` if the file does not exist (caller should do a full rebuild).
    /// - `Ok(None)` (with a debug log) if the file is stale or has an
    ///   unrecognized header.
    /// - `Err(...)` on genuine I/O errors.
    /// - `Ok(Some(index))` on success - the HNSW graph is rebuilt from the
    ///   stored normalized vectors using `cfg`.
    ///
    /// # Errors
    ///
    /// Returns an error on real I/O failures (not on a missing file).
    pub fn load_from_path(
        path: &Path,
        expected_op_id: &Cid,
        cfg: &HnswConfig,
    ) -> Result<Option<Self>, Error> {
        // Missing file is the happy-path cache-miss, not an error.
        if !path.exists() {
            return Ok(None);
        }

        let file = std::fs::File::open(path)
            .map_err(|e| Error::from(StoreError::Io(e.to_string())))?;
        let mut r = BufReader::new(file);

        // magic
        let mut magic = [0u8; 8];
        r.read_exact(&mut magic)
            .map_err(|e| Error::from(StoreError::Io(e.to_string())))?;
        if &magic != b"MNEMHNSW" {
            tracing::debug!("ann cache: bad magic, ignoring {:?}", path);
            return Ok(None);
        }

        // version
        let mut ver_buf = [0u8; 4];
        r.read_exact(&mut ver_buf)
            .map_err(|e| Error::from(StoreError::Io(e.to_string())))?;
        let version = u32::from_le_bytes(ver_buf);
        if version != 1 {
            tracing::debug!("ann cache: unsupported version {}, ignoring {:?}", version, path);
            return Ok(None);
        }

        // op_id
        let mut len_buf = [0u8; 4];
        r.read_exact(&mut len_buf)
            .map_err(|e| Error::from(StoreError::Io(e.to_string())))?;
        let op_id_len = u32::from_le_bytes(len_buf) as usize;
        let mut op_id_bytes = vec![0u8; op_id_len];
        r.read_exact(&mut op_id_bytes)
            .map_err(|e| Error::from(StoreError::Io(e.to_string())))?;
        let stored_op_id = Cid::from_bytes(&op_id_bytes)
            .map_err(|e| Error::from(e))?;
        if &stored_op_id != expected_op_id {
            tracing::debug!("ann cache: stale op_id, ignoring {:?}", path);
            return Ok(None);
        }

        // model
        r.read_exact(&mut len_buf)
            .map_err(|e| Error::from(StoreError::Io(e.to_string())))?;
        let model_len = u32::from_le_bytes(len_buf) as usize;
        let mut model_bytes = vec![0u8; model_len];
        r.read_exact(&mut model_bytes)
            .map_err(|e| Error::from(StoreError::Io(e.to_string())))?;
        let model = String::from_utf8(model_bytes)
            .map_err(|e| Error::from(StoreError::Io(e.to_string())))?;

        // dim
        let mut dim_buf = [0u8; 4];
        r.read_exact(&mut dim_buf)
            .map_err(|e| Error::from(StoreError::Io(e.to_string())))?;
        let dim = u32::from_le_bytes(dim_buf);

        // ef_construction, ef_search, seed (read but use cfg values instead)
        let mut u64_buf = [0u8; 8];
        r.read_exact(&mut u64_buf)
            .map_err(|e| Error::from(StoreError::Io(e.to_string())))?;
        // ef_construction slot (ignored - we use cfg)
        r.read_exact(&mut u64_buf)
            .map_err(|e| Error::from(StoreError::Io(e.to_string())))?;
        // ef_search slot (ignored - we use cfg)
        r.read_exact(&mut u64_buf)
            .map_err(|e| Error::from(StoreError::Io(e.to_string())))?;
        // seed slot (ignored - we use cfg)

        // n_points
        r.read_exact(&mut u64_buf)
            .map_err(|e| Error::from(StoreError::Io(e.to_string())))?;
        let n_points = u64::from_le_bytes(u64_buf) as usize;

        // NodeId bytes
        let mut ids: Vec<NodeId> = Vec::with_capacity(n_points);
        for _ in 0..n_points {
            let mut id_buf = [0u8; 16];
            r.read_exact(&mut id_buf)
                .map_err(|e| Error::from(StoreError::Io(e.to_string())))?;
            ids.push(NodeId::from_bytes_raw(id_buf));
        }

        // f32 vectors
        let mut normalised_vecs: Vec<Vec<f32>> = Vec::with_capacity(n_points);
        let dim_usize = dim as usize;
        for _ in 0..n_points {
            let mut vec = Vec::with_capacity(dim_usize);
            for _ in 0..dim_usize {
                let mut f_buf = [0u8; 4];
                r.read_exact(&mut f_buf)
                    .map_err(|e| Error::from(StoreError::Io(e.to_string())))?;
                vec.push(f32::from_le_bytes(f_buf));
            }
            normalised_vecs.push(vec);
        }

        Ok(Some(Self::from_parts_for_test(
            &model,
            dim,
            ids,
            normalised_vecs,
            cfg,
        )))
    }
}

impl VectorIndex for HnswVectorIndex {
    fn model(&self) -> &str {
        &self.model
    }

    fn dim(&self) -> u32 {
        self.dim
    }

    fn search(&self, query: &[f32], k: usize) -> Result<Vec<VectorHit>, Error> {
        // Empty-index short-circuit (mirrors BruteForce).
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
        if k == 0 {
            return Ok(Vec::new());
        }

        // instant-distance uses owned vectors at the Point layer; we
        // construct one query point per call.
        let q = Point {
            vec: normalise(query.to_vec()),
        };
        let mut searcher = Search::default();
        // ef_search is configured at build-time via the library's
        // Builder::ef_search; we honour our knob by overfetching
        // and letting the caller's `k` truncate.
        let fetch = std::cmp::max(k, self.ef_search);

        let mut hits: Vec<VectorHit> = Vec::with_capacity(k);
        for item in self.inner.search(&q, &mut searcher).take(fetch) {
            let ord = *item.value;
            let node_id = self.ids[ord];
            // Convert squared-Euclidean on unit vectors back to
            // cosine so downstream scores live in [-1, 1] just like
            // BruteForce. cos = 1 - d^2/2.
            let score = 1.0 - item.distance * 0.5;
            hits.push(VectorHit::new(node_id, score));
        }
        // Score DESC, NodeId ASC tiebreak (matches BruteForce exactly).
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

// ---------------------------------------------------------------
// math helpers - duplicated from mnem-core::index::vector to avoid
// the extra pub-surface on that module. Tiny; not worth sharing.
// ---------------------------------------------------------------

fn decode_to_f32(embed: &Embedding) -> Option<Vec<f32>> {
    let dim = embed.dim as usize;
    let bytes = &embed.vector;
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
                out.push(f64::from_le_bytes([
                    chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
                ]) as f32);
            }
            Some(out)
        }
        // F16 / BF16 paths present in BruteForce could be added here
        // when a real workload needs them. The two shipped production
        // embedders today (OpenAI + Ollama) both emit F32.
        _ => None,
    }
}

fn normalise(mut v: Vec<f32>) -> Vec<f32> {
    let mut sq = 0.0_f32;
    for x in &v {
        sq += x * x;
    }
    if sq > 0.0 {
        let inv = sq.sqrt().recip();
        for x in &mut v {
            *x *= inv;
        }
    }
    v
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_build_returns_len_zero_index() {
        let cfg = HnswConfig::default();
        let built = Builder::default()
            .ef_construction(cfg.ef_construction)
            .seed(cfg.seed)
            .build(Vec::<Point>::new(), Vec::<usize>::new());
        let idx = HnswVectorIndex {
            model: "m".into(),
            dim: 0,
            ids: Vec::new(),
            points: Vec::new(),
            inner: built,
            ef_search: cfg.ef_search,
        };
        assert!(idx.is_empty());
        // Search on an empty, dim=0 index returns an empty Vec
        // regardless of query shape. Mirrors BruteForce.
        let hits = idx.search(&[0.0_f32; 3], 5).unwrap();
        assert!(hits.is_empty());
    }

    #[test]
    fn dim_mismatch_errors() {
        use mnem_core::error::RepoError;

        // Build a tiny 3-dim index by hand.
        let points = vec![
            Point {
                vec: normalise(vec![1.0, 0.0, 0.0]),
            },
            Point {
                vec: normalise(vec![0.0, 1.0, 0.0]),
            },
        ];
        let values = vec![0_usize, 1];
        let points_retained = points.clone();
        let inner = Builder::default().build(points, values);
        let idx = HnswVectorIndex {
            model: "m".into(),
            dim: 3,
            ids: vec![NodeId::new_v7(), NodeId::new_v7()],
            points: points_retained,
            inner,
            ef_search: 10,
        };

        let err = idx.search(&[1.0, 0.0], 1).unwrap_err();
        assert!(matches!(
            err,
            Error::Repo(RepoError::VectorDimMismatch {
                index_dim: 3,
                query_dim: 2,
            })
        ));
    }

    #[test]
    fn identical_query_is_top_hit() {
        let id_a = NodeId::new_v7();
        let id_b = NodeId::new_v7();
        let points = vec![
            Point {
                vec: normalise(vec![1.0, 0.0, 0.0]),
            },
            Point {
                vec: normalise(vec![0.0, 1.0, 0.0]),
            },
        ];
        let points_retained = points.clone();
        let inner = Builder::default().build(points, vec![0_usize, 1]);
        let idx = HnswVectorIndex {
            model: "m".into(),
            dim: 3,
            ids: vec![id_a, id_b],
            points: points_retained,
            inner,
            ef_search: 10,
        };

        let hits = idx.search(&[1.0, 0.0, 0.0], 2).unwrap();
        assert_eq!(hits[0].node_id, id_a, "exact match should rank #1");
        // cos(same vec, same vec) = 1.0; allow tiny FP noise.
        assert!(
            (hits[0].score - 1.0).abs() < 1e-5,
            "expected cos == 1, got {}",
            hits[0].score
        );
    }

    #[test]
    fn score_is_cosine_not_euclidean() {
        // Orthogonal unit vectors -> cosine 0.0, sq-euclidean 2.0.
        // We must see 0.0 in the public VectorHit.score, not 2.0.
        let id_a = NodeId::new_v7();
        let id_b = NodeId::new_v7();
        let points = vec![
            Point {
                vec: normalise(vec![1.0, 0.0]),
            },
            Point {
                vec: normalise(vec![0.0, 1.0]),
            },
        ];
        let points_retained = points.clone();
        let inner = Builder::default().build(points, vec![0_usize, 1]);
        let idx = HnswVectorIndex {
            model: "m".into(),
            dim: 2,
            ids: vec![id_a, id_b],
            points: points_retained,
            inner,
            ef_search: 10,
        };
        let hits = idx.search(&[1.0, 0.0], 2).unwrap();
        // The orthogonal neighbour should score ~0.0, not 2.0.
        let orth = hits.iter().find(|h| h.node_id == id_b).unwrap();
        assert!(
            orth.score.abs() < 1e-5,
            "expected orthogonal cos ~= 0; got {}",
            orth.score
        );
    }

    // ---------- build_from_repo / sidecar integration ----------

    fn f32_embed(model: &str, v: &[f32]) -> Embedding {
        let mut bytes = Vec::with_capacity(v.len() * 4);
        for x in v {
            bytes.extend_from_slice(&x.to_le_bytes());
        }
        Embedding {
            model: model.to_string(),
            dtype: Dtype::F32,
            dim: u32::try_from(v.len()).expect("test vec fits in u32"),
            vector: bytes::Bytes::from(bytes),
        }
    }

    fn stores() -> (
        Arc<dyn mnem_core::store::Blockstore>,
        Arc<dyn mnem_core::store::OpHeadsStore>,
    ) {
        (
            Arc::new(mnem_core::store::MemoryBlockstore::new()),
            Arc::new(mnem_core::store::MemoryOpHeadsStore::new()),
        )
    }

    /// Vectors written via `Transaction::set_embedding` are visible
    /// to `HnswVectorIndex::build_from_repo` even when the underlying
    /// `Node` carries no inline `embed`. Mirrors the brute-force
    /// `build_from_repo_indexes_only_matching_model` shape but pins
    /// the sidecar read path on the HNSW side.
    #[test]
    fn build_from_repo_reads_sidecar_embeddings() {
        let (bs, ohs) = stores();
        let repo = ReadonlyRepo::init(bs, ohs).unwrap();
        let mut tx = repo.start_transaction();

        // Two nodes under "mA": no inline embed; vectors live only
        // in the sidecar Prolly tree.
        let id_a = NodeId::from_bytes_raw([1u8; 16]);
        let id_b = NodeId::from_bytes_raw([2u8; 16]);
        let cid_a = tx.add_node(&Node::new(id_a, "Doc")).unwrap();
        let cid_b = tx.add_node(&Node::new(id_b, "Doc")).unwrap();
        tx.set_embedding(cid_a, "mA".into(), f32_embed("mA", &[1.0, 0.0]))
            .unwrap();
        tx.set_embedding(cid_b, "mA".into(), f32_embed("mA", &[0.0, 1.0]))
            .unwrap();

        // One node under "mB": also sidecar-only, must be filtered
        // out when building for "mA".
        let id_c = NodeId::from_bytes_raw([3u8; 16]);
        let cid_c = tx.add_node(&Node::new(id_c, "Doc")).unwrap();
        tx.set_embedding(cid_c, "mB".into(), f32_embed("mB", &[1.0, 0.0]))
            .unwrap();

        // One node with no embedding at all - silently skipped.
        tx.add_node(&Node::new(NodeId::from_bytes_raw([4u8; 16]), "Doc"))
            .unwrap();

        let repo = tx.commit("t", "seed").unwrap();

        let idx = HnswVectorIndex::build_from_repo(&repo, "mA").unwrap();
        assert_eq!(idx.len(), 2, "only the two mA nodes should index");
        assert_eq!(idx.dim(), 2);

        // Query along the +x axis: id_a is the exact match.
        let hits = idx.search(&[1.0, 0.0], 2).unwrap();
        assert_eq!(hits[0].node_id, id_a, "exact-match node should rank #1");
        assert!(
            (hits[0].score - 1.0).abs() < 1e-5,
            "expected cos == 1, got {}",
            hits[0].score
        );
    }

    // ---------- disk persistence (BUG-31) ----------

    /// Helper: build a small deterministic index via `from_parts_for_test`.
    fn small_index() -> (HnswVectorIndex, Vec<NodeId>) {
        let id_a = NodeId::from_bytes_raw([10u8; 16]);
        let id_b = NodeId::from_bytes_raw([20u8; 16]);
        let id_c = NodeId::from_bytes_raw([30u8; 16]);
        let ids = vec![id_a, id_b, id_c];
        let vecs = vec![
            normalise(vec![1.0, 0.0, 0.0]),
            normalise(vec![0.0, 1.0, 0.0]),
            normalise(vec![0.0, 0.0, 1.0]),
        ];
        let cfg = HnswConfig::default();
        let idx = HnswVectorIndex::from_parts_for_test("test-model", 3, ids.clone(), vecs, &cfg);
        (idx, ids)
    }

    fn make_op_id(seed: &[u8]) -> mnem_core::id::Cid {
        use mnem_core::id::{CODEC_RAW, Multihash};
        mnem_core::id::Cid::new(CODEC_RAW, Multihash::sha2_256(seed))
    }

    #[test]
    fn ann_cache_round_trip() {
        let (idx, ids) = small_index();
        let op_id = make_op_id(b"test-op-1");
        let cfg = HnswConfig::default();

        // Save to a unique temp path.
        let path = std::env::temp_dir().join("mnem_ann_cache_round_trip.bin");
        idx.save_to_path(&path, &op_id).expect("save_to_path");

        // Load back and verify structural equality.
        let loaded = HnswVectorIndex::load_from_path(&path, &op_id, &cfg)
            .expect("load_from_path ok")
            .expect("Some(index)");

        assert_eq!(loaded.len(), idx.len(), "same number of points");
        assert_eq!(loaded.dim(), idx.dim(), "same dim");

        // Search results for a test query should match.
        let query = [1.0_f32, 0.0, 0.0];
        let orig_hits = idx.search(&query, 3).unwrap();
        let load_hits = loaded.search(&query, 3).unwrap();
        assert_eq!(orig_hits.len(), load_hits.len(), "hit count matches");
        assert_eq!(
            orig_hits[0].node_id, ids[0],
            "top hit is the x-axis vector"
        );
        assert_eq!(
            load_hits[0].node_id, orig_hits[0].node_id,
            "same top hit after round-trip"
        );
        assert!(
            (load_hits[0].score - orig_hits[0].score).abs() < 1e-5,
            "scores match after round-trip"
        );

        // Clean up.
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn ann_cache_stale_op_id_returns_none() {
        let (idx, _ids) = small_index();
        let op_id_a = make_op_id(b"test-op-A");
        let op_id_b = make_op_id(b"test-op-B");
        let cfg = HnswConfig::default();

        let path = std::env::temp_dir().join("mnem_ann_cache_stale_op_id.bin");
        idx.save_to_path(&path, &op_id_a).expect("save");

        // Load with a different op_id - must return None (stale cache).
        let result = HnswVectorIndex::load_from_path(&path, &op_id_b, &cfg)
            .expect("no I/O error");
        assert!(result.is_none(), "stale op_id should return None");

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn ann_cache_missing_file_returns_none() {
        let op_id = make_op_id(b"test-op-missing");
        let cfg = HnswConfig::default();
        let path = std::env::temp_dir().join("mnem_ann_cache_does_not_exist_xyz.bin");

        // Make sure it really doesn't exist.
        let _ = std::fs::remove_file(&path);

        let result = HnswVectorIndex::load_from_path(&path, &op_id, &cfg)
            .expect("missing file is Ok(None), not Err");
        assert!(result.is_none(), "missing file should return None");
    }
}
