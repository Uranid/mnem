//! `Query` engine + predicates + `QueryHit` over an `IndexSet`.
//!
//! Extracted from `index.rs` in R3; bodies unchanged.

use std::collections::HashSet;

use ipld_core::ipld::Ipld;

use crate::anchor::is_system_node;
use crate::error::{Error, RepoError};
use crate::objects::{Edge, IndexSet, Node};
use crate::prolly::{self, Cursor, ProllyKey};
use crate::repo::readonly::{ReadonlyRepo, decode_from_store};

use super::adjacency::{load_incoming, load_outgoing};
use super::build::prop_value_hash;

/// Predicates supported by [`Query::where_prop`].
#[derive(Clone, Debug)]
#[non_exhaustive]
pub enum PropPredicate {
    /// Exact value match. Uses the property Prolly index if available.
    Eq(Ipld),
}

impl PropPredicate {
    /// Convenience constructor: `PropPredicate::eq("Alice")` vs
    /// `PropPredicate::Eq(Ipld::String("Alice".into()))`.
    pub fn eq(value: impl Into<Ipld>) -> Self {
        Self::Eq(value.into())
    }
}

/// Which direction an edge was loaded from when a query pulls it in.
///
/// Informational only; `execute` fills this into each [`Edge`] it
/// surfaces via an adjacency-carrying field on [`QueryHit`].
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Direction {
    /// Edge whose `src` matches the hit node.
    Outgoing,
    /// Edge whose `dst` matches the hit node.
    Incoming,
}

/// A single query result: the matched node plus any edges requested
/// via [`Query::with_outgoing`] and/or [`Query::with_incoming`].
///
/// The `edges` and `incoming_edges` fields are kept separate rather
/// than folded into one `Vec<(Direction, Edge)>` because 99% of
/// existing callers only care about outgoing and already destructure
/// `.edges`. The self-loop case ([`Query::with_any_direction`] on A→A)
/// returns ONE `Edge` in `edges` (not one in each direction) to avoid
/// spurious double-counting - a self-loop is structurally one edge,
/// not two.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct QueryHit {
    /// The matched node.
    pub node: Node,
    /// Outgoing edges whose label is in the requested set. Ordered by
    /// label then edge CID for deterministic consumption.
    pub edges: Vec<Edge>,
    /// Incoming edges whose label is in the requested set. Ordered by
    /// label, then src, then edge CID for deterministic consumption.
    ///
    /// Populated only when the query calls [`Query::with_incoming`] or
    /// [`Query::with_any_direction`]. For pure [`Query::with_outgoing`]
    /// queries this is always empty.
    pub incoming_edges: Vec<Edge>,
    /// `true` if at least one of `edges` / `incoming_edges` was
    /// truncated by the per-hit adjacency cap. Callers who need the
    /// full fan-in/out should widen [`Query::adjacency_cap`].
    pub edges_truncated: bool,
}

impl QueryHit {
    /// All outgoing edges in this hit whose `etype` equals `label`.
    /// Collects into a `Vec<&Edge>` for ergonomic iteration.
    pub fn edges_by_label(&self, label: &str) -> Vec<&Edge> {
        self.edges.iter().filter(|e| e.etype == label).collect()
    }

    /// Streaming version of [`Self::edges_by_label`]: no intermediate
    /// allocation. Useful in hot loops when a node has many outgoing
    /// edges and only a fraction match the label.
    pub fn edges_by_label_iter<'a>(
        &'a self,
        label: &'a str,
    ) -> impl Iterator<Item = &'a Edge> + 'a {
        self.edges.iter().filter(move |e| e.etype == label)
    }

    /// All incoming edges in this hit whose `etype` equals `label`.
    pub fn incoming_by_label(&self, label: &str) -> Vec<&Edge> {
        self.incoming_edges
            .iter()
            .filter(|e| e.etype == label)
            .collect()
    }
}

/// Declarative agent-facing query over a [`ReadonlyRepo`].
///
/// Usage:
/// ```no_run
/// # use mnem_core::repo::ReadonlyRepo;
/// # use mnem_core::index::{Query, PropPredicate};
/// # use ipld_core::ipld::Ipld;
/// # fn demo(repo: &ReadonlyRepo) -> Result<(), Box<dyn std::error::Error>> {
/// let hits = Query::new(repo)
///     .label("Person")
///     .where_prop("name", PropPredicate::Eq(Ipld::String("Alice".into())))
///     .with_outgoing("knows")
///     .limit(10)
///     .execute()?;
/// # Ok(()) }
/// ```
#[derive(Clone, Debug)]
pub struct Query<'a> {
    repo: &'a ReadonlyRepo,
    label: Option<String>,
    prop_filter: Option<(String, PropPredicate)>,
    with_outgoing: Vec<String>,
    with_incoming: Vec<String>,
    limit: Option<usize>,
    adjacency_cap: usize,
    include_tombstoned: bool,
    /// When `true`, system-reserved nodes (today: the `mnem init`
    /// anchor) are kept in the result set. Defaults to `false` so
    /// agent-facing queries never surface bookkeeping noise.
    /// Audit / admin callers opt back in via [`Self::include_system`].
    include_system: bool,
}

impl<'a> Query<'a> {
    /// Default per-hit cap on how many edges (in each direction) are
    /// surfaced from the adjacency index. Protects agent-facing
    /// callers from fan-in/out denial-of-service (a "celebrity" node
    /// with 1M incoming edges would otherwise allocate a 1M-entry
    /// `Vec` per hit). Override via [`Self::adjacency_cap`].
    ///
    /// The default is intentionally generous (`10_000`) so normal
    /// knowledge graphs are never clipped; the cap is a safety valve,
    /// not a performance knob. Callers that legitimately need the
    /// full fan-in should raise it explicitly and consume the result
    /// stream.
    pub const DEFAULT_ADJACENCY_CAP: usize = 10_000;

    /// Start a new query against `repo`.
    #[must_use]
    pub const fn new(repo: &'a ReadonlyRepo) -> Self {
        Self {
            repo,
            label: None,
            prop_filter: None,
            with_outgoing: Vec::new(),
            with_incoming: Vec::new(),
            limit: None,
            adjacency_cap: Self::DEFAULT_ADJACENCY_CAP,
            include_tombstoned: false,
            include_system: false,
        }
    }

    /// Restrict matches to nodes of a specific label (`ntype`).
    #[must_use]
    pub fn label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }

    /// Add a property predicate. If a label is also set, the indexed
    /// `(label, prop_name) -> value` Prolly lookup is used (O(log n));
    /// otherwise the query falls back to a full label scan.
    #[must_use]
    pub fn where_prop(mut self, name: impl Into<String>, pred: PropPredicate) -> Self {
        self.prop_filter = Some((name.into(), pred));
        self
    }

    /// Convenience: `where_prop(name, PropPredicate::Eq(value.into()))`.
    /// The most common agent query shape, one call shorter.
    #[must_use]
    pub fn where_eq(self, name: impl Into<String>, value: impl Into<Ipld>) -> Self {
        self.where_prop(name, PropPredicate::eq(value))
    }

    /// Include outgoing edges of these labels in every hit.
    #[must_use]
    pub fn with_outgoing(mut self, edge_label: impl Into<String>) -> Self {
        self.with_outgoing.push(edge_label.into());
        self
    }

    /// Include incoming edges of these labels in every hit. Symmetric
    /// mirror of [`Self::with_outgoing`]: answers "who points at me
    /// through this edge-type?" using the `incoming` Prolly tree in
    /// O(log n) plus one bucket read per hit.
    ///
    /// Populates [`QueryHit::incoming_edges`]. When combined with
    /// `with_outgoing` in the same query, a hit is kept if it matches
    /// the base predicates regardless of direction, and each direction's
    /// edges are surfaced in its own field.
    #[must_use]
    pub fn with_incoming(mut self, edge_label: impl Into<String>) -> Self {
        self.with_incoming.push(edge_label.into());
        self
    }

    /// Convenience: ask for this edge-type in BOTH directions. Saves
    /// the caller from writing `with_outgoing(x).with_incoming(x)`
    /// every time.
    ///
    /// Self-loops (edges where `src == dst`) appear in `edges` only,
    /// not duplicated into `incoming_edges`. The execute path detects
    /// the self-loop case and deduplicates on `EdgeId`.
    #[must_use]
    pub fn with_any_direction(mut self, edge_label: impl Into<String>) -> Self {
        let l = edge_label.into();
        self.with_outgoing.push(l.clone());
        self.with_incoming.push(l);
        self
    }

    /// Override the per-hit adjacency cap. See
    /// [`Self::DEFAULT_ADJACENCY_CAP`] for the rationale.
    #[must_use]
    pub const fn adjacency_cap(mut self, cap: usize) -> Self {
        self.adjacency_cap = cap;
        self
    }

    /// Include tombstoned nodes in results. Defaults to false so normal
    /// retrieval/query paths honor privacy revocations.
    #[must_use]
    pub const fn include_tombstoned(mut self, include: bool) -> Self {
        self.include_tombstoned = include;
        self
    }

    /// Include system-reserved nodes (today: the `mnem init` anchor)
    /// in results. Defaults to false so agent-facing surfaces never
    /// see graph bookkeeping. Flip to true for audit / admin flows
    /// that need to inspect or repair the anchor.
    ///
    /// Mirrors [`Self::include_tombstoned`]: both filters live in the
    /// same execute branches, both default to "hide", both opt-in.
    #[must_use]
    pub const fn include_system(mut self, include: bool) -> Self {
        self.include_system = include;
        self
    }

    /// Cap the result set.
    #[must_use]
    pub const fn limit(mut self, n: usize) -> Self {
        self.limit = Some(n);
        self
    }

    /// Convenience: execute and return the first hit, or `Ok(None)`
    /// if the result set is empty. Sets `limit(1)` internally.
    ///
    /// # Errors
    ///
    /// Same as [`Self::execute`].
    pub fn first(mut self) -> Result<Option<QueryHit>, Error> {
        self.limit = Some(1);
        let mut hits = self.execute()?;
        Ok(hits.pop())
    }

    /// Convenience: execute and return the exactly-one hit, erroring if
    /// the result set is empty or has more than one match. Useful when
    /// the agent treats a resolve as a precondition.
    ///
    /// Internally sets `limit(2)` so a genuine second hit is detected
    /// cheaply.
    ///
    /// # Errors
    ///
    /// - [`RepoError::NotFound`] on zero matches.
    /// - [`RepoError::AmbiguousMatch`] on >1 match.
    /// - Propagates any error from [`Self::execute`].
    pub fn one(mut self) -> Result<QueryHit, Error> {
        self.limit = Some(2);
        let hits = self.execute()?;
        match hits.len() {
            0 => Err(RepoError::NotFound.into()),
            1 => Ok(hits.into_iter().next().expect("checked len")),
            _ => Err(RepoError::AmbiguousMatch.into()),
        }
    }

    /// Execute the query against the repo's current commit.
    ///
    /// Dispatches to the fastest matching path:
    /// - label + `prop_eq` with an `IndexSet` present: one Prolly point lookup
    /// - label-only with an `IndexSet`: label sub-tree cursor, bounded by `limit`
    /// - otherwise: streaming scan of `commit.nodes` with in-memory filter,
    ///   also bounded by `limit`
    ///
    /// # Errors
    ///
    /// - [`RepoError::Uninitialized`] if the repo has no head commit.
    /// - Store / codec errors from index lookups.
    pub fn execute(self) -> Result<Vec<QueryHit>, Error> {
        let bs = self.repo.blockstore().clone();
        let Some(commit) = self.repo.head_commit() else {
            return Err(RepoError::Uninitialized.into());
        };
        let indexes = match &commit.indexes {
            Some(idx_cid) => Some(decode_from_store::<IndexSet, _>(&*bs, idx_cid)?),
            None => None,
        };

        // Precompute HashSets for O(1) edge-label membership in the
        // adjacency loaders.
        let want_out: HashSet<&str> = self.with_outgoing.iter().map(String::as_str).collect();
        let want_in: HashSet<&str> = self.with_incoming.iter().map(String::as_str).collect();
        let adj_cap = self.adjacency_cap;

        let mut hits: Vec<QueryHit> = Vec::new();
        let cap = self.limit.unwrap_or(usize::MAX);

        // Helper closure: load both directions for a single hit, with
        // self-loop dedup. A self-loop (src == dst) would otherwise
        // appear once in `edges` and once in `incoming_edges` for the
        // same node; callers using `with_any_direction` would see
        // phantom duplicates. We resolve it by keeping the edge in
        // `edges` (outgoing) and dropping its twin from
        // `incoming_edges` when the request asked for both directions
        // of the same label. Comparison is by `EdgeId` (total-
        // ordering, unique).
        let build_hit = |node: Node, indexes: Option<&IndexSet>| -> Result<QueryHit, Error> {
            let (out_edges, out_trunc) = load_outgoing(&*bs, indexes, node.id, &want_out, adj_cap)?;
            let (mut in_edges, in_trunc) =
                load_incoming(&*bs, indexes, node.id, &want_in, adj_cap)?;
            if !in_edges.is_empty() && !out_edges.is_empty() {
                let out_ids: HashSet<_> = out_edges.iter().map(|e| e.id).collect();
                in_edges.retain(|e| {
                    // Drop the self-loop's incoming twin.
                    !(e.src == e.dst && out_ids.contains(&e.id))
                });
            }
            Ok(QueryHit {
                node,
                edges: out_edges,
                incoming_edges: in_edges,
                edges_truncated: out_trunc || in_trunc,
            })
        };

        match (&self.label, &self.prop_filter, indexes.as_ref()) {
            (Some(label), Some((prop, PropPredicate::Eq(value))), Some(idx)) => {
                // Indexed point lookup. Skip the redundant label/prop
                // filter because the index guarantees the match.
                if let Some(tree_root) = idx.nodes_by_prop.get(label).and_then(|m| m.get(prop)) {
                    let key = ProllyKey::new(prop_value_hash(value)?);
                    if let Some(node_cid) = prolly::lookup(&*bs, tree_root, &key)? {
                        let node: Node = decode_from_store(&*bs, &node_cid)?;
                        // Defensive: the 16-byte hash could collide (cosmically
                        // unlikely with BLAKE3) - reject wrong-label / wrong-
                        // value nodes silently so callers see "no match."
                        if node.ntype == *label
                            && node.props.get(prop) == Some(value)
                            && (self.include_tombstoned || !self.repo.is_tombstoned(&node.id))
                            && (self.include_system || !is_system_node(&node))
                        {
                            hits.push(build_hit(node, indexes.as_ref())?);
                        }
                    }
                }
            }
            (Some(label), None, Some(idx)) => {
                // Label cursor: streaming, bounded by limit.
                if let Some(tree_root) = idx.nodes_by_label.get(label) {
                    let cursor = Cursor::new(&*bs, tree_root)?;
                    for entry in cursor {
                        let (_k, node_cid) = entry?;
                        let node: Node = decode_from_store(&*bs, &node_cid)?;
                        if !self.include_tombstoned && self.repo.is_tombstoned(&node.id) {
                            continue;
                        }
                        if !self.include_system && is_system_node(&node) {
                            continue;
                        }
                        // Index already guarantees the label matches; no
                        // redundant filter needed.
                        hits.push(build_hit(node, indexes.as_ref())?);
                        if hits.len() >= cap {
                            break;
                        }
                    }
                }
            }
            _ => {
                // Streaming fallback: walk the full node tree with
                // in-memory filter, early-exit on limit.
                let cursor = Cursor::new(&*bs, &commit.nodes)?;
                for entry in cursor {
                    let (_k, node_cid) = entry?;
                    let node: Node = decode_from_store(&*bs, &node_cid)?;
                    if !self.include_tombstoned && self.repo.is_tombstoned(&node.id) {
                        continue;
                    }
                    if !self.include_system && is_system_node(&node) {
                        continue;
                    }
                    if let Some(ref lbl) = self.label
                        && &node.ntype != lbl
                    {
                        continue;
                    }
                    if let Some((ref prop, PropPredicate::Eq(ref value))) = self.prop_filter
                        && node.props.get(prop) != Some(value)
                    {
                        continue;
                    }
                    hits.push(build_hit(node, indexes.as_ref())?);
                    if hits.len() >= cap {
                        break;
                    }
                }
            }
        }

        Ok(hits)
    }
}
