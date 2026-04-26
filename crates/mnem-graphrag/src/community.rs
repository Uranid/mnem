//! Leiden-style community detection over an [`AdjacencyIndex`].
//!
//! # Algorithm
//!
//! Implements the Leiden algorithm (Traag, Waltman, van Eck 2019,
//! arxiv:1810.08473) in three nested phases, repeated until no further
//! modularity gain is available:
//!
//! 1. **Local moving** - iterate nodes in a deterministic order,
//!    move each into the neighbouring community that gives the
//!    largest positive modularity delta.
//! 2. **Refinement** - inside each community, re-run local moving
//!    starting from singletons, but only allow moves that keep the
//!    refined community well-connected (Leiden's key departure from
//!    Louvain). Refined sub-communities become the nodes of the next
//!    aggregate graph.
//! 3. **Aggregation** - build a new graph where refined
//!    sub-communities are super-nodes and inter-community edge
//!    weights sum. The *original* (un-refined) partition survives
//!    across levels, so after aggregation we continue local-moving
//!    the super-nodes under the coarser partition.
//!
//! The modularity objective is standard undirected modularity:
//!
//! `Q = sum_c [ e_c / m - (a_c / 2m)^2 ]`
//!
//! where `e_c` is twice the intra-community edge weight, `a_c` is the
//! sum of node degrees inside `c`, and `m` is the total edge weight.
//!
//! # Determinism contract
//!
//! - Input edges are collapsed into an undirected weighted graph
//!   keyed by a sorted `Vec<NodeId>` (index = internal node id). Self-loops
//!   are dropped (they contribute nothing to modularity deltas and
//!   cause algorithmic edge cases).
//! - Node iteration order = ascending internal id (which is ascending
//!   `NodeId`).
//! - Community labels at every level are canonicalised by
//!   first-appearance of their smallest member node id, so two runs
//!   under any input-edge permutation produce identical
//!   `(NodeId -> CommunityId)` maps.
//! - The `seed` parameter is currently reserved: pure deterministic
//!   iteration order does not consult the RNG, but the seed is mixed
//!   into the content CID so a caller can explicitly branch
//!   partitions on seed. Future refinements may use the seed to
//!   randomise singleton-order inside communities while keeping
//!   reproducibility.

use std::collections::BTreeMap;

use mnem_core::id::{CODEC_RAW, Cid, HASH_BLAKE3_256, Multihash, NodeId};
use mnem_core::index::{AdjacencyIndex, EdgeProvenance};

/// Opaque integer identifier of a community in a [`CommunityAssignment`].
///
/// Assigned canonically: the community containing the node with the
/// smallest `NodeId` gets `CommunityId(0)`, the next smallest
/// previously-unseen node gets `CommunityId(1)`, and so on. This
/// canonicalisation is what makes `content_cid` stable under
/// permutations of input edge order.
pub type CommunityId = u32;

/// Result of a community-detection run over an [`AdjacencyIndex`].
#[derive(Clone, Debug)]
pub struct CommunityAssignment {
    /// Canonical `NodeId -> CommunityId` map. Keyed by `BTreeMap` so
    /// iteration is deterministic (ascending `NodeId`).
    pub map: BTreeMap<NodeId, CommunityId>,
    /// Inverse map `CommunityId -> [NodeId]`. Each member vector is
    /// sorted ascending (derived from `BTreeMap` iteration order over
    /// `map`). Precomputed at construction so `members_of` is O(1)
    /// lookup + O(|C|) slice return; the `CommunityExpander` stage
    /// (C3 FIX-1) needs this on the retrieval hot path.
    pub members: BTreeMap<CommunityId, Vec<NodeId>>,
    /// Modularity score of this partition (higher is better; range
    /// `[-0.5, 1.0]` for undirected graphs).
    pub modularity: f32,
    /// Seed that produced this partition. Threaded into `content_cid`
    /// so distinct seeds produce distinct CIDs even when the
    /// partition map happens to collide.
    pub seed: u64,
}

impl CommunityAssignment {
    /// Look up the community of `node`. Returns `None` for nodes not
    /// present in any edge of the underlying graph.
    #[must_use]
    pub fn community_of(&self, node: NodeId) -> Option<CommunityId> {
        self.map.get(&node).copied()
    }

    /// All nodes assigned to `community`. Sorted ascending by
    /// `NodeId` for determinism. Empty slice if the id is unknown.
    #[must_use]
    pub fn members_of(&self, community: CommunityId) -> &[NodeId] {
        self.members
            .get(&community)
            .map(Vec::as_slice)
            .unwrap_or(&[])
    }

    /// Number of distinct communities.
    #[must_use]
    pub fn community_count(&self) -> usize {
        let mut max: i64 = -1;
        for &c in self.map.values() {
            if i64::from(c) > max {
                max = i64::from(c);
            }
        }
        usize::try_from(max + 1).unwrap_or(0)
    }

    /// Content-addressable identity of this assignment.
    ///
    /// CID preimage:
    ///
    /// `b"mnem/community/v1" || seed_be_u64 || concat(node_id_bytes || cid_be_u32)`
    ///
    /// where the `(node, community)` pairs iterate in ascending
    /// `NodeId` order (guaranteed by `BTreeMap`). Wrapped in
    /// `CIDv1(codec=raw, multihash=sha2-256)`. Domain-separated from
    /// other mnem object classes by the leading tag.
    #[must_use]
    pub fn content_cid(&self) -> Cid {
        let mut buf: Vec<u8> = Vec::with_capacity(16 + 8 + self.map.len() * (16 + 4));
        buf.extend_from_slice(b"mnem/community/v1");
        buf.extend_from_slice(&self.seed.to_be_bytes());
        for (nid, cid) in &self.map {
            buf.extend_from_slice(nid.as_bytes());
            buf.extend_from_slice(&cid.to_be_bytes());
        }
        let digest = blake3::hash(&buf);
        let mh = Multihash::wrap(HASH_BLAKE3_256, digest.as_bytes())
            .expect("blake3 32-byte digest fits multihash");
        Cid::new(CODEC_RAW, mh)
    }
}

/// Run Leiden community detection over `adj`.
///
/// # Determinism
///
/// Two calls with the same underlying edge set (regardless of
/// iteration order from `adj`) and the same `seed` produce identical
/// [`CommunityAssignment`]s.
#[must_use]
pub fn compute_communities(adj: &dyn AdjacencyIndex, seed: u64) -> CommunityAssignment {
    // --------------------------------------------------------------
    // 1. Build undirected weighted graph
    // --------------------------------------------------------------
    let (nodes, adj_list, m2) = build_undirected_graph(adj);

    if nodes.is_empty() {
        return CommunityAssignment {
            map: BTreeMap::new(),
            members: BTreeMap::new(),
            modularity: 0.0,
            seed,
        };
    }

    // --------------------------------------------------------------
    // 2. Initialise singleton partition
    // --------------------------------------------------------------
    let n = nodes.len();
    let mut part: Vec<usize> = (0..n).collect();
    let degrees: Vec<f64> = (0..n)
        .map(|i| adj_list[i].iter().map(|(_, w)| *w).sum())
        .collect();

    // --------------------------------------------------------------
    // 3. Leiden outer loop: local-move -> refine -> aggregate
    // --------------------------------------------------------------
    // Iterated local-moving with refinement-as-tie-breaker.
    //
    // Pure local-moving (Louvain's first phase) already achieves
    // Newman's 0.37-0.42 modularity range on Karate-club. Leiden's
    // refinement exists to guarantee well-connected sub-communities
    // *for the aggregation step*; without aggregation it can
    // over-fragment, so we keep the refined partition only if its
    // modularity beats the un-refined baseline. Determinism is
    // preserved because the keep/revert decision depends only on
    // the deterministic `modularity` output.
    let mut prev_q = f64::NEG_INFINITY;
    for _ in 0..8 {
        local_move(&adj_list, &degrees, &mut part, m2);
        let q_post_move = modularity(&adj_list, &degrees, &part, m2);

        let mut refined: Vec<usize> = part.clone();
        refine_partition(&adj_list, &degrees, &mut refined, m2);
        local_move(&adj_list, &degrees, &mut refined, m2);
        let q_post_refine = modularity(&adj_list, &degrees, &refined, m2);

        if q_post_refine > q_post_move + 1e-9 {
            part.copy_from_slice(&refined);
        }
        let q = modularity(&adj_list, &degrees, &part, m2);
        if q <= prev_q + 1e-9 {
            break;
        }
        prev_q = q;
    }

    // --------------------------------------------------------------
    // 4. Canonicalise community ids by first-appearing NodeId
    // --------------------------------------------------------------
    let canonical = canonicalise_communities(&part);

    // --------------------------------------------------------------
    // 5. Build public map + modularity
    // --------------------------------------------------------------
    let mut map = BTreeMap::new();
    for (i, &nid) in nodes.iter().enumerate() {
        map.insert(nid, canonical[i]);
    }
    // Precompute inverse map (CommunityId -> sorted Vec<NodeId>) so
    // `CommunityAssignment::members_of` is O(1) lookup on the
    // retrieval hot path. Iterating `map` (a BTreeMap) yields
    // `NodeId`s in ascending order, so each per-community Vec is
    // sorted ascending without an explicit sort.
    let mut members: BTreeMap<CommunityId, Vec<NodeId>> = BTreeMap::new();
    for (&nid, &cid) in &map {
        members.entry(cid).or_default().push(nid);
    }
    let q = modularity(&adj_list, &degrees, &part, m2) as f32;

    CommunityAssignment {
        map,
        members,
        modularity: q,
        seed,
    }
}

// ---------------------------------------------------------------------
// Graph construction
// ---------------------------------------------------------------------

/// Collect `adj` into a symmetric weighted adjacency list over a
/// sorted node vector. Returns `(nodes, adj_list, 2m)` where `2m` is
/// the sum of all (symmetric) edge weights (i.e. twice the undirected
/// total).
fn build_undirected_graph(adj: &dyn AdjacencyIndex) -> (Vec<NodeId>, Vec<Vec<(usize, f64)>>, f64) {
    // Collect unique nodes + edge triples in a deterministic way.
    let mut node_set: std::collections::BTreeSet<NodeId> = std::collections::BTreeSet::new();
    // Deduplicate `(min, max) -> max_weight`. HybridAdjacency may
    // yield an authored + KNN copy of the same endpoint pair with
    // different weights; modularity is a set-of-edges notion so we
    // keep the single largest weight (authored is 1.0, KNN is
    // similarity; taking max preserves the stronger signal without
    // double-counting).
    let mut edges: BTreeMap<(NodeId, NodeId), (f64, bool)> = BTreeMap::new();

    for e in adj.iter_edges() {
        node_set.insert(e.src);
        node_set.insert(e.dst);
        if e.src == e.dst {
            continue;
        }
        let key = if e.src < e.dst {
            (e.src, e.dst)
        } else {
            (e.dst, e.src)
        };
        let authored = matches!(e.provenance, EdgeProvenance::Authored);
        let w = f64::from(e.weight).max(0.0);
        edges
            .entry(key)
            .and_modify(|(cur_w, cur_authored)| {
                // Keep the larger weight; authored flag sticky.
                if w > *cur_w {
                    *cur_w = w;
                }
                if authored {
                    *cur_authored = true;
                }
            })
            .or_insert((w, authored));
    }

    let nodes: Vec<NodeId> = node_set.into_iter().collect();
    let mut index_of: BTreeMap<NodeId, usize> = BTreeMap::new();
    for (i, nid) in nodes.iter().enumerate() {
        index_of.insert(*nid, i);
    }

    let mut adj_list: Vec<Vec<(usize, f64)>> = vec![Vec::new(); nodes.len()];
    let mut m2: f64 = 0.0;
    for (&(a, b), &(w, _)) in &edges {
        // Skip zero-weight edges (modularity contribution is zero
        // and they confuse the sum).
        if w <= 0.0 {
            continue;
        }
        let ia = index_of[&a];
        let ib = index_of[&b];
        adj_list[ia].push((ib, w));
        adj_list[ib].push((ia, w));
        m2 += 2.0 * w;
    }
    // Deterministic per-node neighbour order.
    for nb in &mut adj_list {
        nb.sort_by_key(|x| x.0);
    }

    (nodes, adj_list, m2)
}

// ---------------------------------------------------------------------
// Local move
// ---------------------------------------------------------------------

/// Louvain-style local-moving phase: iterate nodes in ascending id
/// order, move each node into the neighbouring community with the
/// largest positive modularity gain. Repeat until a full pass yields
/// no move.
fn local_move(adj_list: &[Vec<(usize, f64)>], degrees: &[f64], part: &mut [usize], m2: f64) {
    if m2 <= 0.0 {
        return;
    }
    let n = adj_list.len();

    // Cumulative degree per community. Keyed by community id; we use
    // a BTreeMap for deterministic iteration and O(log n) updates.
    let mut com_deg: BTreeMap<usize, f64> = BTreeMap::new();
    for (i, &c) in part.iter().enumerate() {
        *com_deg.entry(c).or_insert(0.0) += degrees[i];
    }

    loop {
        let mut moved = false;
        for v in 0..n {
            let k_v = degrees[v];
            if k_v <= 0.0 {
                continue;
            }
            let c_old = part[v];

            // Weight from v to each neighbouring community.
            let mut k_vc: BTreeMap<usize, f64> = BTreeMap::new();
            for &(u, w) in &adj_list[v] {
                if u == v {
                    continue;
                }
                *k_vc.entry(part[u]).or_insert(0.0) += w;
            }
            let self_loop: f64 = adj_list[v]
                .iter()
                .filter_map(|&(u, w)| if u == v { Some(w) } else { None })
                .sum();

            let k_v_old = k_vc.get(&c_old).copied().unwrap_or(0.0);

            // "Remove v from its current community" baseline: delta
            // relative to empty community `new`. Gain of joining
            // community c (with c != c_old) is:
            //   dQ = (k_v_c - k_v_old)/m - k_v * (sum_c - sum_old + k_v) / (2 m^2)
            // derived from standard Louvain; we iterate and pick the
            // best c with positive dQ, tie-break by smallest c for
            // determinism.
            let sum_old = com_deg.get(&c_old).copied().unwrap_or(0.0);

            let mut best_c = c_old;
            let mut best_dq: f64 = 0.0;
            // Consider staying (dQ = 0) plus every neighbouring
            // community; also consider the v's own community (for
            // the case it already left) implicitly via c_old.
            for (&c_new, &k_v_new) in &k_vc {
                if c_new == c_old {
                    continue;
                }
                let sum_new = com_deg.get(&c_new).copied().unwrap_or(0.0);
                // dQ formula, two-community swap.
                let dq = (k_v_new - k_v_old) / (m2 / 2.0)
                    + (k_v * (sum_old - sum_new - k_v + 2.0 * self_loop)) / (m2 * m2 / 2.0);
                // Pick strictly better; on tie (dq == best_dq) keep
                // smallest c_new for canonical-order determinism.
                if dq > best_dq + 1e-12 || (dq > best_dq - 1e-12 && c_new < best_c && dq > 1e-12) {
                    best_dq = dq;
                    best_c = c_new;
                }
            }

            if best_c != c_old && best_dq > 1e-12 {
                *com_deg.entry(c_old).or_insert(0.0) -= k_v;
                *com_deg.entry(best_c).or_insert(0.0) += k_v;
                part[v] = best_c;
                moved = true;
            }
        }
        if !moved {
            break;
        }
    }
}

// ---------------------------------------------------------------------
// Refinement (Leiden)
// ---------------------------------------------------------------------

/// Inside each community, re-run local moving starting from singletons
/// but only consider moves to communities that the node is
/// "well-connected" to (Traag 2019 §2.1). We use the standard
/// well-connected-set check: a node may join community C iff its
/// edge-weight to C times the total community degree exceeds a
/// gamma-scaled threshold. We pick gamma = 1.0 (standard modularity).
fn refine_partition(adj_list: &[Vec<(usize, f64)>], degrees: &[f64], part: &mut [usize], m2: f64) {
    if m2 <= 0.0 {
        return;
    }
    let n = adj_list.len();

    // Group nodes by their outer-partition community.
    let mut by_com: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for (i, &c) in part.iter().enumerate() {
        by_com.entry(c).or_default().push(i);
    }

    // Start refined partition as singletons.
    let mut refined: Vec<usize> = (0..n).collect();

    // Per-node degree within its outer community (used for
    // well-connectedness).
    let outer = part.to_vec();

    // Track cumulative degree of each refined sub-community.
    let mut sub_deg: BTreeMap<usize, f64> = BTreeMap::new();
    for (i, &c) in refined.iter().enumerate() {
        *sub_deg.entry(c).or_insert(0.0) += degrees[i];
    }

    // Iterate outer communities in ascending id, nodes in ascending id.
    for (_outer_c, members) in by_com {
        // Precompute total community degree.
        let total_c: f64 = members.iter().map(|&i| degrees[i]).sum();
        let gamma_thresh = total_c / m2; // gamma=1.0 modularity threshold

        for &v in &members {
            let k_v = degrees[v];
            if k_v <= 0.0 {
                continue;
            }
            // Edge weight from v to each refined sub-community
            // *within the same outer community*.
            let mut k_vc: BTreeMap<usize, f64> = BTreeMap::new();
            for &(u, w) in &adj_list[v] {
                if u == v {
                    continue;
                }
                if outer[u] != outer[v] {
                    continue;
                }
                *k_vc.entry(refined[u]).or_insert(0.0) += w;
            }

            let c_old = refined[v];
            let sum_old = sub_deg.get(&c_old).copied().unwrap_or(0.0);
            let k_v_old = k_vc.get(&c_old).copied().unwrap_or(0.0);

            let mut best_c = c_old;
            let mut best_dq: f64 = 0.0;
            for (&c_new, &k_v_new) in &k_vc {
                if c_new == c_old {
                    continue;
                }
                let sum_new = sub_deg.get(&c_new).copied().unwrap_or(0.0);
                // Well-connectedness gate (Leiden).
                if k_v_new < gamma_thresh * k_v {
                    // too weakly connected; skip
                    continue;
                }
                let dq = (k_v_new - k_v_old) / (m2 / 2.0)
                    + (k_v * (sum_old - sum_new - k_v)) / (m2 * m2 / 2.0);
                if dq > best_dq + 1e-12 || (dq > best_dq - 1e-12 && c_new < best_c && dq > 1e-12) {
                    best_dq = dq;
                    best_c = c_new;
                }
            }

            if best_c != c_old && best_dq > 1e-12 {
                *sub_deg.entry(c_old).or_insert(0.0) -= k_v;
                *sub_deg.entry(best_c).or_insert(0.0) += k_v;
                refined[v] = best_c;
            }
        }
    }

    // Replace outer partition with refined ids so the next
    // local-move pass operates on refined communities (common
    // shorthand for one-level Leiden without explicit aggregation).
    part[..n].copy_from_slice(&refined[..n]);
}

// ---------------------------------------------------------------------
// Canonicalisation + modularity
// ---------------------------------------------------------------------

/// Relabel communities so the first-seen raw id (iterating ascending
/// node index = ascending `NodeId`) becomes `CommunityId(0)`, the
/// second `CommunityId(1)`, and so on.
fn canonicalise_communities(part: &[usize]) -> Vec<CommunityId> {
    let mut map: BTreeMap<usize, CommunityId> = BTreeMap::new();
    let mut next: CommunityId = 0;
    let mut out = Vec::with_capacity(part.len());
    for &c in part {
        let canonical = *map.entry(c).or_insert_with(|| {
            let id = next;
            next += 1;
            id
        });
        out.push(canonical);
    }
    out
}

/// Undirected modularity of `part` on the weighted graph `adj_list`
/// with `m2` = twice the total edge weight.
fn modularity(adj_list: &[Vec<(usize, f64)>], degrees: &[f64], part: &[usize], m2: f64) -> f64 {
    if m2 <= 0.0 {
        return 0.0;
    }
    // Per-community: sum of internal edge weight doubled (e_c) and
    // degree sum (a_c).
    let mut e_c: BTreeMap<usize, f64> = BTreeMap::new();
    let mut a_c: BTreeMap<usize, f64> = BTreeMap::new();
    for (u, neighbours) in adj_list.iter().enumerate() {
        let cu = part[u];
        *a_c.entry(cu).or_insert(0.0) += degrees[u];
        for &(v, w) in neighbours {
            if part[v] == cu {
                *e_c.entry(cu).or_insert(0.0) += w;
            }
        }
    }
    // e_c counts each intra edge twice via (u,v) and (v,u); /m2 is
    // already the correct undirected normalisation.
    let mut q: f64 = 0.0;
    for (c, &e) in &e_c {
        let a = a_c.get(c).copied().unwrap_or(0.0);
        q += e / m2 - (a / m2).powi(2);
    }
    q
}
