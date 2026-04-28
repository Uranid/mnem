//! Personalized PageRank over an [`AdjacencyIndex`].
//!
//! Part of experiment E2 (LLM-free GraphRAG - PPR graph-expand). The
//! algorithm is hand-rolled power iteration on a compact row-stochastic
//! CSR (compressed-sparse-row) matrix. The `sprs` crate was considered
//! and rejected for this turn: the power-iteration inner loop is a
//! three-line sparse mat-vec, and keeping the algebra hand-rolled lets
//! us
//!
//! 1. guarantee byte-determinism across platforms (no third-party SIMD
//!    scheduler surprise),
//! 2. hold the binary-size delta for the `mnem-http` release build
//!    under 100 KiB (E2 T2 gate),
//! 3. drop the algorithm into WASM builds later without auditing a new
//!    transitive graph.
//!
//! The power iteration itself follows the standard [Haveliwala 2002]
//! formulation: `r_{t+1} = (1 - d) * p + d * M^T r_t`, where `M` is the
//! row-stochastic adjacency (outgoing edges normalised to sum to 1 per
//! source). Dead-ends (nodes with no out-edges) are handled by
//! redistributing their mass uniformly over the personalization vector:
//! the standard "teleport on dangling" fix that keeps the total mass at
//! exactly 1.0 every iteration.
//!
//! # Determinism
//!
//! - Node ordering is derived once from `iter_edges` by inserting in
//!   first-seen order, then **re-sorted ascending**. Fixed ordering
//!   means the same graph always produces the same CSR layout.
//! - No parallelism. No RNG. No HashMap-iteration-order dependency.
//! - Fixed `max_iter` + fixed `eps` convergence: two runs over the same
//!   inputs produce byte-identical score vectors (property-tested).
//!
//! [Haveliwala 2002]: https://www-cs.stanford.edu/~taherh/papers/topic-sensitive-pagerank.pdf

use std::collections::BTreeMap;

use crate::id::NodeId;
use crate::index::hybrid::AdjacencyIndex;

/// Default damping factor (`0.85`), matching the original PageRank paper
/// and the LightRAG / GraphRAG reference implementations.
pub const DEFAULT_DAMPING: f32 = 0.85;
/// Default iteration cap. 15 is enough for `eps = 1e-6` on the graph
/// sizes E2 targets; verified by the convergence test.
pub const DEFAULT_MAX_ITER: u32 = 15;
/// Default L1-delta convergence threshold.
pub const DEFAULT_EPS: f32 = 1e-6;

/// Gap 02 R6 numeric pin: graph-size threshold for default-on PPR.
///
/// Above this node count, PPR is opt-in only (see Gap 02 solution.md
/// `§250k-V gate`). The size gate skips PPR and falls back to decay
/// expansion when `|V| > PPR_DEFAULT_MAX_NODES && !cfg.ppr_opt_in`,
/// recording a labelled counter and emitting a
/// `warnings[]::PprSizeGateSkipped` entry on the response. The
/// threshold is chosen to match the HNSW memory derivation shared
/// with `GRAPH_SIZE_GATE_V` (see `benchmarks/leiden-wallclock-vs-V.md`)
/// so operators reason about a single graph-scale cliff, not two.
///
/// `#tunable: default=250_000; rationale="matches HNSW-memory cliff; see benchmarks/leiden-wallclock-vs-V.md"`
pub const PPR_DEFAULT_MAX_NODES: usize = 250_000;

/// Gap 02 #17 pure gate helper.
///
/// Returns `true` iff `adj` has strictly more than
/// [`PPR_DEFAULT_MAX_NODES`] unique node ids across both sides of
/// every edge AND `opt_in` is `false`. Callers use this to decide
/// whether to skip the PPR walk and fall back to decay expansion.
///
/// Node-count is derived by a single O(|E|) pass over
/// [`AdjacencyIndex::iter_edges`] collecting into a `BTreeSet`. The
/// pass early-exits the moment the threshold is exceeded, so the
/// common case (graph well under the gate) pays the full count and
/// the pathological case (graph well over) pays exactly
/// `PPR_DEFAULT_MAX_NODES + 1` inserts.
///
/// Split out from the retriever so the gate logic is unit-testable
/// without a full repo fixture.
#[must_use]
pub fn exceeds_size_gate(adj: &(dyn AdjacencyIndex + Send + Sync), opt_in: bool) -> bool {
    if opt_in {
        return false;
    }
    let mut uniq: std::collections::BTreeSet<NodeId> = std::collections::BTreeSet::new();
    for edge in adj.iter_edges() {
        uniq.insert(edge.src);
        uniq.insert(edge.dst);
        if uniq.len() > PPR_DEFAULT_MAX_NODES {
            return true;
        }
    }
    false
}

/// PPR configuration. Split out from the runner so CLI / HTTP / MCP
/// DTOs can carry just these three numbers.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PprConfig {
    /// Damping factor `d` in `(1 - d) * p + d * M^T r`. Default
    /// [`DEFAULT_DAMPING`].
    pub damping: f32,
    /// Maximum power-iteration steps. Default [`DEFAULT_MAX_ITER`].
    pub max_iter: u32,
    /// L1-delta convergence threshold. Stop when
    /// `|| r_{t+1} - r_t ||_1 < eps`. Default [`DEFAULT_EPS`].
    pub eps: f32,
}

impl Default for PprConfig {
    fn default() -> Self {
        Self {
            damping: DEFAULT_DAMPING,
            max_iter: DEFAULT_MAX_ITER,
            eps: DEFAULT_EPS,
        }
    }
}

/// Row-stochastic sparse transition matrix in plain CSR form, plus the
/// ordered `NodeId` -> matrix-row mapping. Public so tests / downstream
/// code can inspect it.
#[derive(Clone, Debug)]
pub struct SparseTransition {
    /// Sorted `NodeId` sequence: `nodes[i]` is row/column `i`.
    pub nodes: Vec<NodeId>,
    /// CSR row pointers. Length `nodes.len() + 1`.
    pub row_ptr: Vec<usize>,
    /// CSR column indices. Length `nnz`.
    pub col_idx: Vec<usize>,
    /// CSR values. Length `nnz`. Row-stochastic: each non-empty row
    /// sums to 1.0. Dead rows (no out-edges) have zero values in the
    /// CSR and are handled by teleporting their mass in [`ppr`].
    pub values: Vec<f32>,
    /// True for rows that had at least one out-edge. Dead rows
    /// redistribute their mass to the personalization vector each
    /// iteration.
    pub has_outgoing: Vec<bool>,
}

/// Build a row-stochastic transition matrix from an adjacency index.
///
/// Edge weights are **summed** when the same `(src, dst)` pair appears
/// more than once (e.g. authored edge plus a KNN echo), then each row
/// is L1-normalised. Self-loops are preserved (PPR handles them fine
/// and tests pin the behaviour).
///
/// Node ordering is first-seen on the iterator, then sorted ascending.
/// The sort is what gives us determinism regardless of whether the
/// caller presented edges in `(src, dst)` order, interleaved, etc.
#[must_use]
pub fn sparse_transition_matrix(adj: &dyn AdjacencyIndex) -> SparseTransition {
    // Pass 1: collect the unique NodeId set. Using a BTreeMap here
    // gives us automatic sorted iteration for deterministic row order.
    let mut id_to_row: BTreeMap<NodeId, usize> = BTreeMap::new();
    // Pass 1 also records every (src, dst, weight) triple for the
    // second-pass row assembly. Vec<_> to keep insertion order at this
    // stage; the dedupe+sum happens in pass 2.
    let mut triples: Vec<(NodeId, NodeId, f32)> = Vec::with_capacity(adj.edge_count());
    for edge in adj.iter_edges() {
        id_to_row.entry(edge.src).or_insert(0);
        id_to_row.entry(edge.dst).or_insert(0);
        triples.push((edge.src, edge.dst, edge.weight));
    }
    // Fill in the true row indices now that the BTreeMap iteration
    // order is stable.
    let nodes: Vec<NodeId> = id_to_row.keys().copied().collect();
    for (i, id) in nodes.iter().enumerate() {
        if let Some(slot) = id_to_row.get_mut(id) {
            *slot = i;
        }
    }

    // Pass 2: accumulate per-row `(col, weight)` entries with duplicate
    // dedupe via a per-row BTreeMap. BTreeMap keyed on `usize` gives us
    // sorted column order inside each row - a requirement for clean
    // CSR + deterministic mat-vec.
    let n = nodes.len();
    let mut per_row: Vec<BTreeMap<usize, f32>> = (0..n).map(|_| BTreeMap::new()).collect();
    for (s, d, w) in triples {
        let si = id_to_row[&s];
        let di = id_to_row[&d];
        // Guard: silently drop non-positive weights. The upstream
        // KnnEdge contract guarantees weights in (0, 1]; authored
        // edges default to 1.0. Anything else is a defensive no-op
        // rather than a panic so a corrupted repo still completes the
        // retrieve call.
        if w <= 0.0 || !w.is_finite() {
            continue;
        }
        *per_row[si].entry(di).or_insert(0.0) += w;
    }

    // Pass 3: row-normalise and flatten to CSR.
    let mut row_ptr: Vec<usize> = Vec::with_capacity(n + 1);
    let mut col_idx: Vec<usize> = Vec::new();
    let mut values: Vec<f32> = Vec::new();
    let mut has_outgoing: Vec<bool> = vec![false; n];
    row_ptr.push(0);
    for (i, row) in per_row.iter().enumerate() {
        let sum: f32 = row.values().sum();
        if sum > 0.0 {
            has_outgoing[i] = true;
            for (&c, &w) in row {
                col_idx.push(c);
                values.push(w / sum);
            }
        }
        row_ptr.push(col_idx.len());
    }

    SparseTransition {
        nodes,
        row_ptr,
        col_idx,
        values,
        has_outgoing,
    }
}

/// Run personalised PageRank power iteration.
///
/// - `adj`: any [`AdjacencyIndex`]. Edge weights are used directly.
/// - `personalization`: seed distribution. Keys absent from the graph
///   are ignored; zero / negative values are ignored. The vector is
///   L1-normalised internally, so caller-side magnitudes do not need
///   to sum to 1.
/// - `cfg`: damping, iter cap, convergence threshold.
///
/// Returns a `BTreeMap<NodeId, f32>` so iteration order downstream is
/// also deterministic.
///
/// # Panics
///
/// Never. A malformed / empty graph returns an empty map, and a zero
/// personalization vector falls back to the uniform distribution so the
/// algorithm still produces a valid ranking.
#[allow(clippy::many_single_char_names)]
#[must_use]
pub fn ppr(
    adj: &dyn AdjacencyIndex,
    personalization: &BTreeMap<NodeId, f32>,
    cfg: PprConfig,
) -> BTreeMap<NodeId, f32> {
    let m = sparse_transition_matrix(adj);
    ppr_with_matrix(&m, personalization, cfg)
}

/// Run personalised PageRank using a pre-built [`SparseTransition`].
///
/// Byte-identical to [`ppr`] on the same inputs; the only difference is
/// that the caller supplies the CSR matrix instead of having it
/// re-derived from an [`AdjacencyIndex`] on every call. Useful for
/// callers (e.g. the HTTP layer) that cache the matrix per op-id and
/// re-run PPR with different personalization vectors across requests.
///
/// The convergence criterion is the standard Page/Brin 1998 L1 early-
/// stop: iteration halts as soon as `|| r_{t+1} - r_t ||_1 < cfg.eps`.
///
/// # Panics
///
/// Never. See [`ppr`].
#[allow(clippy::many_single_char_names)]
#[must_use]
pub fn ppr_with_matrix(
    m: &SparseTransition,
    personalization: &BTreeMap<NodeId, f32>,
    cfg: PprConfig,
) -> BTreeMap<NodeId, f32> {
    let n = m.nodes.len();
    if n == 0 {
        return BTreeMap::new();
    }
    // Damping clamp so a user passing 1.0 or 1.5 cannot break the
    // algorithm's contraction property.
    let damping = cfg.damping.clamp(0.0, 0.999);

    // Build the personalization vector in row-order. Ignore keys not
    // in the graph; clamp negatives / non-finites to zero; L1-normalise.
    let mut p = vec![0f32; n];
    let mut psum = 0f32;
    for (id, &w) in personalization {
        if let Ok(idx) = m.nodes.binary_search(id)
            && w > 0.0
            && w.is_finite()
        {
            p[idx] += w;
            psum += w;
        }
    }
    if psum > 0.0 {
        for v in &mut p {
            *v /= psum;
        }
    } else {
        // Degenerate case: caller gave us nothing usable. Fall back to
        // uniform so the algorithm still converges to a non-trivial
        // ranking. Documented in the function contract.
        let u = 1.0 / n as f32;
        p.fill(u);
    }

    let mut r = p.clone();
    let mut next = vec![0f32; n];
    for _iter in 0..cfg.max_iter {
        // `dangling_mass` collects r[i] for every dead-row i. Dead
        // rows get redistributed to p (teleport) each step - the
        // standard PageRank-on-dangling fix. Skipping this drains
        // total mass and breaks L1 conservation.
        let mut dangling_mass = 0f32;
        for i in 0..n {
            if !m.has_outgoing[i] {
                dangling_mass += r[i];
            }
        }
        // Base term: (1 - d) * p + d * dangling_mass * p
        // (the teleport-on-dangling term folds into the personal-
        // ization scaling).
        let teleport_scale = (1.0 - damping) + damping * dangling_mass;
        for i in 0..n {
            next[i] = teleport_scale * p[i];
        }
        // Add d * (M^T r) via a single CSR pass. For each row i with
        // out-edges, distribute r[i] * values[k] to next[col_idx[k]].
        for i in 0..n {
            if !m.has_outgoing[i] {
                continue;
            }
            let start = m.row_ptr[i];
            let end = m.row_ptr[i + 1];
            let ri = r[i];
            if ri == 0.0 {
                continue;
            }
            for k in start..end {
                let j = m.col_idx[k];
                next[j] += damping * ri * m.values[k];
            }
        }
        // Convergence: L1 delta. Computed AFTER the mat-vec so the
        // early-exit check sees the fresh distribution. L1-normalise
        // `next` defensively against f32 drift so mass stays at 1.0
        // even after 15 iterations of accumulated rounding - this is
        // what keeps the proptest's byte-identity check passing.
        let next_sum: f32 = next.iter().sum();
        if next_sum > 0.0 {
            for v in &mut next {
                *v /= next_sum;
            }
        }
        let mut delta = 0f32;
        for i in 0..n {
            delta += (next[i] - r[i]).abs();
        }
        std::mem::swap(&mut r, &mut next);
        next.fill(0.0);
        if delta < cfg.eps {
            break;
        }
    }

    // Emit in NodeId-sorted order (BTreeMap gives us this for free).
    let mut out: BTreeMap<NodeId, f32> = BTreeMap::new();
    for (i, id) in m.nodes.iter().enumerate() {
        out.insert(*id, r[i]);
    }
    out
}
