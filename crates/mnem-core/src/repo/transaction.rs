//! [`Transaction`] - accumulator for pending mutations + commit.
//!
//! A transaction captures a snapshot of the current [`ReadonlyRepo`] at
//! `start_transaction()` time. Mutations ([`Transaction::add_node`],
//! `add_edge`, `remove_node`, etc.) are buffered. [`Transaction::commit`]
//! atomically:
//!
//! 1. Rebuilds the node / edge / schema Prolly trees from the base
//! commit's roots + the buffered additions and removals.
//! 2. Writes a new Commit whose `parents` is the previous head.
//! 3. Writes a new View whose `heads = [new commit]` and whose `refs`
//! reflect any ref-update mutations.
//! 4. Writes a new Operation whose `parents = [old op]`.
//! 5. Advances the op-heads store: inserts new op, removes old op.
//! 6. Returns a fresh [`ReadonlyRepo`] pinned to the new op.
//!
//! Multi-writer safety: step 5 is atomic per
//! [`OpHeadsStore::update`][crate::store::OpHeadsStore::update].
//! If another writer has advanced the heads concurrently, both new ops
//! remain in the heads set; the next [`ReadonlyRepo::open`] will see
//! multiple heads and (in M8.5) trigger a 3-way merge.

use std::collections::{BTreeMap, HashSet};

use ipld_core::ipld::Ipld;

use crate::codec::hash_to_cid;
use crate::error::{Error, RepoError};
use crate::id::{ChangeId, Cid, EdgeId, NodeId};
use crate::index;
use crate::objects::node::Embedding;
use crate::objects::{
 Commit, Edge, EmbeddingBucket, IndexSet, Node, Operation, RefTarget, Tombstone, View,
};
use crate::prolly::{self, Cursor, ProllyKey};
use crate::store::Blockstore;

use super::readonly::{ReadonlyRepo, decode_from_store, now_micros};

/// Options controlling the commit path.
///
/// The default (via [`Transaction::commit`]) is lock-free: concurrent
/// writers both succeed; the next reader merges. Setting
/// [`linearize`](Self::linearize) to `true` enables SPEC §6.5
/// opportunistic concurrency - if any other writer has advanced
/// op-heads since this transaction started, the commit fails with
/// [`RepoError::Stale`] instead of appending a concurrent head.
#[derive(Clone, Copy, Debug)]
pub struct CommitOptions<'a> {
 /// Commit author (UTF-8, stored on the new Commit + Operation).
 pub author: &'a str,
 /// Commit message.
 pub message: &'a str,
 /// Opt-in SPEC §6.5 linearize mode. Defaults to `false`.
 pub linearize: bool,
 /// Override the commit + operation timestamp. Measured in
 /// microseconds since Unix epoch. `None` (the default) calls
 /// `SystemTime::now()` at commit time, which is what a human
 /// workflow wants.
 ///
 /// Set this to `Some(...)` when byte-identical CIDs across
 /// machines matter: two processes that build the same logical
 /// commit (same author, same message, same graph mutations,
 /// same time, same `change_id`) will produce the same commit CID
 /// and the same op-id. This is the escape hatch for
 /// audit-replay, distributed-agent consensus, and regression
 /// tests that assert on commit CIDs.
 pub time_micros: Option<u64>,
 /// Override the commit's `change_id`. `None` (the default)
 /// generates a fresh `ChangeId::new_v7()`, which embeds wall-
 /// clock time and therefore varies per call. Deterministic-
 /// replay workflows MUST supply this explicitly alongside
 /// `time_micros`; otherwise the v7 randomness alone defeats the
 /// byte-identical-CID contract.
 pub change_id: Option<ChangeId>,
}

impl<'a> CommitOptions<'a> {
 /// Construct with all deterministic-override fields set to `None`
 /// (the caller-convenient default: auto-clock + auto-change-id).
 #[must_use]
 pub const fn new(author: &'a str, message: &'a str) -> Self {
 Self {
 author,
 message,
 linearize: false,
 time_micros: None,
 change_id: None,
 }
 }

 /// Pin the timestamp for deterministic replay. See
 /// [`Self::time_micros`] for the wider contract.
 #[must_use]
 pub const fn with_time_micros(mut self, t: u64) -> Self {
 self.time_micros = Some(t);
 self
 }

 /// Pin the change-id for deterministic replay. See
 /// [`Self::change_id`] for the wider contract.
 #[must_use]
 pub const fn with_change_id(mut self, id: ChangeId) -> Self {
 self.change_id = Some(id);
 self
 }
}

/// Buffered mutations against a [`ReadonlyRepo`].
///
/// Construct via [`ReadonlyRepo::start_transaction`].
pub struct Transaction {
 base: ReadonlyRepo,
 new_nodes: BTreeMap<NodeId, Cid>,
 removed_nodes: HashSet<NodeId>,
 new_edges: BTreeMap<EdgeId, Cid>,
 removed_edges: HashSet<EdgeId>,
 ref_updates: BTreeMap<String, Option<RefTarget>>,
 /// Tombstones staged for insertion into the new View at commit
 /// time. Keyed by `NodeId`; later writes to the same `NodeId` in
 /// the same transaction overwrite the earlier ones (consistent
 /// with [`Self::tombstone_node`]'s idempotent-deterministic rule).
 new_tombstones: BTreeMap<NodeId, Tombstone>,
 /// Side-table for `resolve_or_create_node`: maps
 /// `(label, prop_name, blake3(canonical(value))[..16])` to the
 /// `NodeId` of a node added in this transaction. Bounded by the
 /// number of `resolve_or_create_node` or `add_node` calls in this
 /// tx; prevents the O(pending²) decode loop the naive
 /// implementation would trigger.
 pending_by_prop: BTreeMap<(String, String, [u8; 16]), NodeId>,
 /// Lazy, one-time decode of the base commit's `IndexSet`. Populated
 /// on the first `resolve_or_create_node` call and re-used by
 /// every subsequent call in this transaction. `None` means
 /// "not yet fetched"; `Some(None)` means "no `IndexSet` on the
 /// base commit" (either uninitialised or pre-0.2 commit).
 cached_base_indexes: Option<Option<IndexSet>>,
 /// Pending embedding-sidecar writes, keyed by the content-addressed
 /// `NodeCid` they reference. Multiple `set_embedding` calls for the
 /// same node accumulate into one bucket (one entry per `model`
 /// string). Empty by default; the commit path skips the sidecar
 /// rebuild entirely when this map is empty AND the base commit
 /// carried no `embeddings` root.
 pending_embeddings: BTreeMap<Cid, EmbeddingBucket>,
}

impl Transaction {
 pub(crate) fn new(base: ReadonlyRepo) -> Self {
 Self {
 base,
 new_nodes: BTreeMap::new(),
 removed_nodes: HashSet::new(),
 new_edges: BTreeMap::new(),
 removed_edges: HashSet::new(),
 ref_updates: BTreeMap::new(),
 new_tombstones: BTreeMap::new(),
 pending_by_prop: BTreeMap::new(),
 cached_base_indexes: None,
 pending_embeddings: BTreeMap::new(),
 }
 }

 /// The base repo this transaction is derived from.
 #[must_use]
 pub const fn base(&self) -> &ReadonlyRepo {
 &self.base
 }

 // ---------------- Mutations ----------------

 /// Add (or overwrite) a node. Cancels any pending `remove_node` for
 /// the same id. Returns the node's content-addressed CID.
 ///
 /// # Errors
 ///
 /// Codec or blockstore errors while writing the node.
 pub fn add_node(&mut self, node: &Node) -> Result<Cid, Error> {
 let (bytes, cid) = hash_to_cid(node)?;
 // safety: cid computed above via hash_to_cid
 self.base.blockstore.put_trusted(cid.clone(), bytes)?;
 self.removed_nodes.remove(&node.id);
 self.new_nodes.insert(node.id, cid.clone());
 // Populate the pending-by-prop cache so future
 // `resolve_or_create_node` calls in this tx find the node in
 // O(1) instead of decoding every pending node.
 for (prop_name, prop_value) in &node.props {
 if let Ok(hash) = index::prop_value_hash(prop_value) {
 self.pending_by_prop
 .insert((node.ntype.clone(), prop_name.clone(), hash), node.id);
 }
 }
 Ok(cid)
 }

 /// Stage an embedding for a previously-added node into the
 /// embedding-sidecar Prolly tree referenced by
 /// `Commit.embeddings`.
 ///
 /// Symmetric with [`Self::add_node`]: pass the `node_cid` returned
 /// from `add_node` (or any pre-existing NodeCid you want to attach
 /// a new vector to). Multiple `set_embedding` calls for the same
 /// `node_cid` accumulate into one [`EmbeddingBucket`]; calling
 /// twice with the same `model` upserts (the second value wins).
 ///
 /// The actual sidecar tree is built and committed by
 /// [`Self::commit`] / [`Self::commit_opts`]; staging does not
 /// touch the blockstore.
 ///
 /// # Why this lives outside the Node bytes
 ///
 /// Dense embedding vectors drift in their last bit across ORT
 /// thread counts (`f32` reduction reordering is non-associative).
 /// Storing them inline on `Node` would couple `NodeCid` to
 /// thread count and break federated dedup. The sidecar separates
 /// identity (Node) from derived bytes (Embedding); two machines
 /// re-deriving the same source text on different cores share the
 /// Node CID even when their vectors differ.
 ///
 /// # Errors
 ///
 /// Currently infallible (the staged map cannot fail to insert);
 /// the `Result` shape is reserved for future validation hooks
 /// (e.g. dim/dtype checks against a per-repo config).
 pub fn set_embedding(
 &mut self,
 node_cid: Cid,
 model: String,
 embedding: Embedding,
 ) -> Result<(), Error> {
 let bucket = self.pending_embeddings.entry(node_cid).or_default();
 bucket.upsert(model, embedding);
 Ok(())
 }

 /// Remove a node. Cancels any pending `add_node` for the same id.
 ///
 /// If the node was added AND had an embedding staged via
 /// `set_embedding` in this same transaction, the staged embedding
 /// is dropped to prevent an orphan sidecar entry. Embeddings that
 /// already live in the base commit's sidecar tree are NOT scrubbed
 /// here; they remain reachable through the inherited tree (a
 /// follow-up audit will add explicit sidecar tombstones).
 pub fn remove_node(&mut self, id: NodeId) {
 if let Some(cid) = self.new_nodes.remove(&id) {
 self.pending_embeddings.remove(&cid);
 }
 self.removed_nodes.insert(id);
 // Drop any pending-by-prop entries pointing at this id.
 self.pending_by_prop.retain(|_, v| *v != id);
 }

 /// Add (or overwrite) an edge. Returns the edge's content-addressed CID.
 ///
 /// # Errors
 ///
 /// Codec or blockstore errors while writing the edge.
 pub fn add_edge(&mut self, edge: &Edge) -> Result<Cid, Error> {
 let (bytes, cid) = hash_to_cid(edge)?;
 // safety: cid computed above via hash_to_cid
 self.base.blockstore.put_trusted(cid.clone(), bytes)?;
 self.removed_edges.remove(&edge.id);
 self.new_edges.insert(edge.id, cid.clone());
 Ok(cid)
 }

 /// Remove an edge.
 pub fn remove_edge(&mut self, id: EdgeId) {
 self.new_edges.remove(&id);
 self.removed_edges.insert(id);
 }

 /// Logically "forget" a node without breaking the append-only,
 /// content-addressed invariant of the graph.
 ///
 /// The node block remains in the node Prolly tree; its CID does
 /// not change. What changes is the [`View`]: at commit time, a
 /// [`Tombstone`] record keyed by `node_id` is inserted into
 /// [`View::tombstones`]. Retrieval paths filter out tombstoned
 /// nodes by default - see
 /// [`crate::retrieve::Retriever::include_tombstoned`] for the
 /// opt-out used by audit / debug callers.
 ///
 /// The tombstone's `tombstoned_at` timestamp is set at commit
 /// time (via the commit's resolved `now`), not at the call site,
 /// so two transactions built in parallel don't disagree on when a
 /// node was revoked just because of clock skew between author
 /// processes.
 ///
 /// Idempotence: calling `tombstone_node` twice for the same
 /// `node_id` in the same transaction is a no-op at the semantic
 /// level. The second call overwrites the first's reason; no
 /// additional state change is observable to a retrieve or to a
 /// subsequent `is_tombstoned` query. Across transactions, the
 /// rule is the same: each new tombstone commit fully replaces the
 /// prior record for that node.
 ///
 /// The original Node is NOT removed and edges referencing it are
 /// NOT touched. For physical removal, use
 /// [`Self::remove_node`] instead.
 ///
 /// # Errors
 ///
 /// Currently infallible; the `Result` return type reserves space
 /// for future validation (e.g. rejecting tombstones on a
 /// non-existent `node_id`).
 #[tracing::instrument(
 level = "debug",
 target = "mnem::repo::transaction",
 skip(self, reason),
 fields(node_id = %node_id)
 )]
 pub fn tombstone_node(
 &mut self,
 node_id: NodeId,
 reason: impl Into<String>,
 ) -> Result<(), Error> {
 // Stamp with a placeholder `0` timestamp; the real
 // `tombstoned_at` is filled in at commit time from the
 // commit's resolved `now`. This keeps multiple
 // `tombstone_node` calls in one transaction all sharing the
 // same timestamp, which is the semantic agents expect
 // ("everything in this commit got revoked together").
 let ts = Tombstone::new(reason, 0);
 self.new_tombstones.insert(node_id, ts);
 Ok(())
 }

 /// Set a named ref in the new View. `None` removes the ref.
 pub fn update_ref(&mut self, name: impl Into<String>, target: Option<RefTarget>) {
 self.ref_updates.insert(name.into(), target);
 }

 /// Ergonomic one-call node write for agent workflows.
 ///
 /// Generates a fresh [`NodeId::new_v7`], builds the node with the
 /// caller's `ntype`, `summary`, and properties, auto-stamps two
 /// reserved temporal props (`mnem:created_at`, `mnem:updated_at`)
 /// with the current microseconds-since-Unix-epoch, and writes the
 /// node via [`Self::add_node`]. Returns the freshly generated
 /// `NodeId`.
 ///
 /// The reserved prop keys are the substrate's temporal-range filter
 /// contract (see
 /// [`crate::retrieve::Retriever::where_created_after`] et al.) and
 /// avoid a breaking Node-CID change that a dedicated header field
 /// would have triggered. Callers who need deterministic-replay CIDs
 /// can override either key by passing it explicitly in `props`; the
 /// auto-stamp only fills absent keys.
 ///
 /// # Errors
 ///
 /// Propagates codec/blockstore errors from [`Self::add_node`].
 pub fn commit_memory<I>(
 &mut self,
 ntype: impl Into<String>,
 summary: impl Into<String>,
 props: I,
 ) -> Result<NodeId, Error>
 where
 I: IntoIterator<Item = (String, Ipld)>,
 {
 let id = NodeId::new_v7();
 let mut node = Node::new(id, ntype).with_summary(summary);
 for (k, v) in props {
 node.props.insert(k, v);
 }
 let now = now_micros();
 node.props
 .entry("mnem:created_at".to_string())
 .or_insert_with(|| Ipld::Integer(i128::from(now)));
 node.props
 .entry("mnem:updated_at".to_string())
 .or_insert_with(|| Ipld::Integer(i128::from(now)));
 self.add_node(&node)?;
 Ok(id)
 }

 /// Find-or-create a node by a primary-key property.
 ///
 /// Looks for an existing node with `(ntype == label, props[prop_name] == value)`
 /// in the following order:
 ///
 /// 1. Nodes added in this transaction (O(1) via a cache that
 /// `add_node` maintains).
 /// 2. The base commit's property index, if one exists (O(log n)).
 ///
 /// If a match is found, its `NodeId` is returned. Otherwise a new
 /// node is added (with `prop_name -> value` set) and its fresh
 /// `NodeId` is returned. This is the go-to helper for agents
 /// writing facts from LLM output where the same entity may be
 /// mentioned multiple times across tool calls.
 ///
 /// Within a single `resolve_or_create_node` call the cost is
 /// bounded by one cache lookup + one index point lookup; total
 /// cost of N calls in a transaction is O(N log n), not O(N²).
 ///
 /// # Errors
 ///
 /// Propagates codec/store errors from the property-index lookup or
 /// node write.
 pub fn resolve_or_create_node(
 &mut self,
 label: &str,
 prop_name: &str,
 value: impl Into<Ipld>,
 ) -> Result<NodeId, Error> {
 let value = value.into();
 let hash = index::prop_value_hash(&value)?;

 // 1. Pending-adds cache: O(1) BTreeMap lookup.
 if let Some(id) =
 self.pending_by_prop
 .get(&(label.to_string(), prop_name.to_string(), hash))
 {
 return Ok(*id);
 }

 // 2. Base commit's property index: O(log n) point lookup.
 // The IndexSet is fetched once per transaction and cached;
 // a hot resolve loop pays exactly one decode_from_store of
 // the IndexSet, not N.
 if self.cached_base_indexes.is_none() {
 let fetched = if let Some(commit) = self.base.commit.as_deref() {
 if let Some(idx_cid) = &commit.indexes {
 Some(decode_from_store::<IndexSet, _>(
 &*self.base.blockstore,
 idx_cid,
 )?)
 } else {
 None
 }
 } else {
 None
 };
 self.cached_base_indexes = Some(fetched);
 }
 if let Some(Some(indexes)) = self.cached_base_indexes.as_ref()
 && let Some((_cid, node)) =
 index::lookup_by_prop(&*self.base.blockstore, indexes, label, prop_name, &value)?
 && !self.removed_nodes.contains(&node.id)
 {
 return Ok(node.id);
 }

 // 3. Create.
 let new_node = Node::new(NodeId::new_v7(), label).with_prop(prop_name, value);
 self.add_node(&new_node)?;
 Ok(new_node.id)
 }

 // ---------------- Commit ----------------

 /// Convenience: commit in the default lock-free mode.
 ///
 /// Delegates to [`commit_opts`](Self::commit_opts) with
 /// `linearize: false`. See there for semantics.
 ///
 /// # Errors
 ///
 /// Codec, store, and tree-rebuild errors.
 pub fn commit(self, author: &str, message: &str) -> Result<ReadonlyRepo, Error> {
 self.commit_opts(CommitOptions::new(author, message))
 }

 /// Finalize the transaction with explicit options.
 ///
 /// Lock-free default ([`CommitOptions::linearize`] = `false`):
 /// rebuild trees, write Commit / View / Operation, advance the
 /// op-head regardless of concurrent writers, return a fresh
 /// [`ReadonlyRepo`] pinned to the new op.
 ///
 /// Linearize mode (`linearize = true`, SPEC §6.5): re-read
 /// op-heads just before advancing. If the current set is not
 /// exactly `[base.op_id]`, return [`RepoError::Stale`] without
 /// advancing. Tree / commit / view / op bytes already written to
 /// the blockstore remain (they are content-addressed and
 /// collision-free; a retry will re-reference them).
 ///
 /// # Errors
 ///
 /// - [`RepoError::Stale`] in linearize mode when op-heads drift.
 /// - Codec, store, and tree-rebuild errors.
 ///
 /// # Instrumentation
 ///
 /// Emits one `info`-level span `mnem::repo::transaction::commit` per
 /// call with bounded-cardinality fields: `added_nodes`,
 /// `removed_nodes`, `added_edges`, `removed_edges`, `tombstones`,
 /// `ref_updates`, `linearize`. No node payloads or CIDs are
 /// recorded - a commit of 100k nodes still produces one span of
 /// constant size. Agents wanting per-node detail should enable the
 /// `debug`-level `tombstone_node` span or add their own.
 #[tracing::instrument(
 name = "commit",
 level = "info",
 target = "mnem::repo::transaction",
 skip(self, opts),
 fields(
 added_nodes = self.new_nodes.len(),
 removed_nodes = self.removed_nodes.len(),
 added_edges = self.new_edges.len(),
 removed_edges = self.removed_edges.len(),
 tombstones = self.new_tombstones.len(),
 ref_updates = self.ref_updates.len(),
 linearize = opts.linearize,
 )
 )]
 pub fn commit_opts(self, opts: CommitOptions<'_>) -> Result<ReadonlyRepo, Error> {
 let Self {
 base,
 new_nodes,
 removed_nodes,
 new_edges,
 removed_edges,
 ref_updates,
 new_tombstones,
 pending_by_prop: _,
 cached_base_indexes: _,
 pending_embeddings,
 } = self;

 let bs = base.blockstore.clone();
 let ohs = base.op_heads.clone();

 // Base roots (trees from the previous commit, or empty trees if
 // this is the first commit on a fresh repo).
 let (base_nodes, base_edges, base_schema) = if let Some(commit) = base.commit.as_deref() {
 (
 commit.nodes.clone(),
 commit.edges.clone(),
 commit.schema.clone(),
 )
 } else {
 let empty_root = prolly::build_tree(&*bs, std::iter::empty())?;
 (empty_root.clone(), empty_root.clone(), empty_root)
 };

 // Decide gating for the incremental-index fast path BEFORE we
 // consume `new_nodes` / `removed_*` / `new_edges` into their
 // ProllyKey forms.
 let is_append_only_at_graph_level = removed_nodes.is_empty()
 && removed_edges.is_empty()
 && new_edges.is_empty()
 && !new_nodes.is_empty();
 let base_indexes_cid: Option<Cid> = base.commit.as_deref().and_then(|c| c.indexes.clone());

 // Keep a NodeId-keyed sorted copy of the added nodes so the
 // incremental index path can re-decode them. The ProllyKey
 // map used for the node-tree rebuild is derived from this
 // without consuming it.
 let new_nodes_btree: BTreeMap<NodeId, Cid> = new_nodes.into_iter().collect();
 let node_additions: BTreeMap<ProllyKey, Cid> = new_nodes_btree
 .iter()
 .map(|(id, cid)| (ProllyKey::from(*id), cid.clone()))
 .collect();
 let node_removals: HashSet<ProllyKey> =
 removed_nodes.into_iter().map(ProllyKey::from).collect();
 let new_nodes_root = rebuild_tree(&*bs, &base_nodes, &node_additions, &node_removals)?;

 // Rebuild edge tree with mutations.
 let edge_additions: BTreeMap<ProllyKey, Cid> = new_edges
 .into_iter()
 .map(|(id, cid)| (ProllyKey::from(id), cid))
 .collect();
 let edge_removals: HashSet<ProllyKey> =
 removed_edges.into_iter().map(ProllyKey::from).collect();
 let new_edges_root = rebuild_tree(&*bs, &base_edges, &edge_additions, &edge_removals)?;

 // Schema tree unchanged in M8 MVP (no schema mutations yet).
 let new_schema_root = base_schema;

 // Build secondary indexes. Fast path: incremental append when
 // the transaction is a pure node-level append AND we have a
 // previous IndexSet to extend AND no added NodeId collides with
 // an existing one in the base node tree. Slow path: full
 // rebuild (same as before; correctness baseline).
 //
 // The fast path is byte-equivalent to the slow path in the
 // conditions above; the `incremental_append_indexes` contract
 // in `mnem-core::index` pins this. Tests in this module pin
 // the equivalence on round-trip.
 let new_indexes_cid = match (is_append_only_at_graph_level, base_indexes_cid.as_ref()) {
 (true, Some(base_idx)) => {
 // O(|new_nodes| * log N) point-lookup check for collisions
 // against the base node tree. On any lookup error, fall
 // back to full rebuild (safety over speed).
 let has_collision = new_nodes_btree.keys().any(|node_id| {
 let key = ProllyKey::from(*node_id);
 matches!(
 crate::prolly::lookup(&*bs, &base_nodes, &key),
 Ok(Some(_)) | Err(_)
 )
 });
 if has_collision {
 index::build_index_set(&*bs, &new_nodes_root, &new_edges_root)?
 } else {
 index::incremental_append_indexes(&*bs, base_idx, &new_nodes_btree)?
 }
 }
 _ => index::build_index_set(&*bs, &new_nodes_root, &new_edges_root)?,
 };

 // Embedding sidecar (). Skip the rebuild entirely when no
 // pending writes AND no base sidecar - most commits in a
 // legacy repo will hit this fast path. Otherwise: encode each
 // pending bucket, stage its CID under the 16-byte truncated
 // blake3 of the NodeCid wire form (matches the lookup keying
 // in `ReadonlyRepo::embedding_for`), and feed the additions
 // through the same `rebuild_tree` helper the node + edge
 // trees use.
 let base_embeddings_cid: Option<Cid> =
 base.commit.as_deref().and_then(|c| c.embeddings.clone());
 let new_embeddings_cid: Option<Cid> = if pending_embeddings.is_empty()
 && base_embeddings_cid.is_none()
 {
 None
 } else {
 let base_root = match &base_embeddings_cid {
 Some(c) => c.clone(),
 None => prolly::build_tree(&*bs, std::iter::empty())?,
 };
 let mut additions: BTreeMap<ProllyKey, Cid> = BTreeMap::new();
 for (node_cid, bucket) in pending_embeddings {
 let (bucket_bytes, bucket_cid) = hash_to_cid(&bucket)?;
 bs.put_trusted(bucket_cid.clone(), bucket_bytes)?;
 let key = embedding_key_for_node_cid(&node_cid);
 additions.insert(key, bucket_cid);
 }
 Some(rebuild_tree(&*bs, &base_root, &additions, &HashSet::new())?)
 };

 // Build the new Commit.
 //
 // `time_micros` and `change_id` are deterministic-replay escape
 // hatches: callers who want byte-identical CIDs across
 // machines supply both. `None` falls back to wall clock +
 // fresh v7 (the current human-workflow default).
 let now = opts.time_micros.unwrap_or_else(now_micros);
 let change_id = opts.change_id.unwrap_or_else(ChangeId::new_v7);
 let mut commit = Commit::new(
 change_id,
 new_nodes_root,
 new_edges_root,
 new_schema_root,
 opts.author,
 now,
 opts.message,
 );
 commit.indexes = Some(new_indexes_cid);
 commit.embeddings = new_embeddings_cid;
 if let Some(prev_head) = base.view.heads.first() {
 commit = commit.with_parent(prev_head.clone());
 }
 let (commit_bytes, commit_cid) = hash_to_cid(&commit)?;
 // safety: commit_cid computed above via hash_to_cid
 bs.put_trusted(commit_cid.clone(), commit_bytes)?;

 // Build the new View.
 let mut new_view: View = (*base.view).clone();
 let is_first_commit = base.view.heads.is_empty() && new_view.refs.is_empty();
 new_view.heads = vec![commit_cid.clone()];
 for (name, target) in ref_updates {
 match target {
 Some(t) => {
 new_view.refs.insert(name, t);
 }
 None => {
 new_view.refs.remove(&name);
 }
 }
 }
 // C4-1 (audit-2026-04-25): Mirror Git - on the first commit
 // of a fresh repo, auto-create `refs/heads/main` pointing at
 // the new commit unless the caller already supplied a ref
 // update (explicit beats implicit). This means `mnem init` +
 // first ingest leaves the repo with a usable default branch
 // so docs examples like `mnem branch create test main` work
 // out of the box without requiring `mnem ref set` plumbing.
 if is_first_commit && !new_view.refs.contains_key("refs/heads/main") {
 new_view
 .refs
 .insert("refs/heads/main".to_string(), RefTarget::normal(commit_cid));
 }
 // Stamp every staged tombstone with the commit's resolved `now`
 // so all tombstones in one commit share a timestamp (agents
 // expect "revoked together" to mean "same timestamp"), and
 // merge into the View. Later entries overwrite earlier ones,
 // matching the idempotent-deterministic rule documented on
 // `tombstone_node`.
 for (node_id, mut ts) in new_tombstones {
 ts.tombstoned_at = now;
 new_view.tombstones.insert(node_id, ts);
 }
 let (view_bytes, view_cid) = hash_to_cid(&new_view)?;
 // safety: view_cid computed above via hash_to_cid
 bs.put_trusted(view_cid.clone(), view_bytes)?;

 // Build the new Operation.
 let op = Operation::new(
 view_cid,
 opts.author,
 now,
 format!("commit: {}", opts.message),
 )
 .with_parent(base.op_id.clone());
 let (op_bytes, op_cid) = hash_to_cid(&op)?;
 // safety: op_cid computed above via hash_to_cid
 bs.put_trusted(op_cid.clone(), op_bytes)?;

 // Linearize check (SPEC §6.5): re-read op-heads just before the
 // CAS-like advance. If drift has occurred, fail rather than
 // append a concurrent head.
 if opts.linearize {
 let current = ohs.current()?;
 if current.len() != 1 || current[0] != base.op_id {
 return Err(RepoError::Stale.into());
 }
 }

 // Advance op-heads atomically.
 ohs.update(op_cid.clone(), std::slice::from_ref(&base.op_id))?;

 // Return a fresh ReadonlyRepo pinned to the new op.
 ReadonlyRepo::load_at(bs, ohs, op_cid)
 }
}

// ---------------- Tree rebuild helper ----------------

/// Rebuild a Prolly tree by applying additions and removals to the
/// contents of an existing base tree.
///
/// Naive O(n) implementation: walks the whole base tree via [`Cursor`],
/// filters out removals, applies additions, sorts, and re-builds. A
/// future M5.5+ incremental mutation path will re-chunk only touched
/// subtrees. For M8 MVP this is acceptable - typical graph commits
/// touch a small fraction of a tree, so the rebuild is the slow path
/// rather than the common path.
///
/// # Errors
///
/// Store and codec errors while iterating and writing.
fn rebuild_tree<B: Blockstore + ?Sized>(
 store: &B,
 base_root: &Cid,
 additions: &BTreeMap<ProllyKey, Cid>,
 removals: &HashSet<ProllyKey>,
) -> Result<Cid, Error> {
 // Stream the base tree into a map (absorbs removals and prepares for
 // addition-override). Using BTreeMap so final iteration is sorted.
 let mut merged: BTreeMap<ProllyKey, Cid> = BTreeMap::new();
 let cursor = Cursor::new(store, base_root)?;
 for entry in cursor {
 let (k, v) = entry?;
 if removals.contains(&k) {
 continue;
 }
 merged.insert(k, v);
 }
 for (k, v) in additions {
 merged.insert(*k, v.clone());
 }
 // Feed to the Prolly builder (input is already sorted via BTreeMap).
 prolly::build_tree(store, merged)
}

/// Derive the 16-byte Prolly key for the embedding-sidecar tree from
/// a `NodeCid`. We blake3 the CID's wire form (codec + multihash) and
/// take the first 16 bytes; that gives uniformly-distributed keys
/// regardless of the codec/digest prefix structure of the CID, so the
/// Prolly tree's leaf-split heuristic produces balanced nodes.
///
/// Both [`Transaction::commit_opts`] (write side) and
/// [`crate::repo::ReadonlyRepo::embedding_for`] (read side) MUST go
/// through this exact helper. Two callers that derive keys differently
/// would silently miss each other's writes.
pub(crate) fn embedding_key_for_node_cid(node_cid: &Cid) -> ProllyKey {
 let h = blake3::hash(&node_cid.to_bytes());
 let mut k = [0u8; 16];
 k.copy_from_slice(&h.as_bytes()[..16]);
 ProllyKey(k)
}

#[cfg(test)]
mod tests {
 use super::*;
 use crate::id::{CODEC_RAW, Multihash};
 use crate::store::{MemoryBlockstore, MemoryOpHeadsStore, OpHeadsStore};
 use ipld_core::ipld::Ipld;
 use std::sync::Arc;

 fn new_repo() -> ReadonlyRepo {
 let bs: Arc<dyn Blockstore> = Arc::new(MemoryBlockstore::new());
 let ohs: Arc<dyn OpHeadsStore> = Arc::new(MemoryOpHeadsStore::new());
 ReadonlyRepo::init(bs, ohs).unwrap()
 }

 #[test]
 fn first_commit_advances_head_and_stores_commit() {
 let repo = new_repo();
 assert!(repo.head_commit().is_none());

 let mut tx = repo.start_transaction();
 let alice =
 Node::new(NodeId::new_v7(), "Person").with_prop("name", Ipld::String("Alice".into()));
 tx.add_node(&alice).unwrap();
 let new_repo = tx.commit("alice@example.org", "add Alice").unwrap();

 assert!(new_repo.head_commit().is_some());
 let head = new_repo.head_commit().unwrap();
 assert_eq!(head.message, "add Alice");

 let looked_up = new_repo.lookup_node(&alice.id).unwrap();
 assert_eq!(looked_up.as_ref(), Some(&alice));
 }

 #[test]
 fn second_commit_chains_parent_and_preserves_history() {
 let repo = new_repo();
 let alice =
 Node::new(NodeId::new_v7(), "Person").with_prop("name", Ipld::String("Alice".into()));
 let mut tx1 = repo.start_transaction();
 tx1.add_node(&alice).unwrap();
 let repo_v1 = tx1.commit("tester", "add Alice").unwrap();
 let v1_head_cid = repo_v1.view().heads[0].clone();

 let bob =
 Node::new(NodeId::new_v7(), "Person").with_prop("name", Ipld::String("Bob".into()));
 let mut tx2 = repo_v1.start_transaction();
 tx2.add_node(&bob).unwrap();
 let repo_v2 = tx2.commit("tester", "add Bob").unwrap();

 // Alice still findable after second commit.
 assert_eq!(
 repo_v2.lookup_node(&alice.id).unwrap().as_ref(),
 Some(&alice)
 );
 // Bob findable too.
 assert_eq!(repo_v2.lookup_node(&bob.id).unwrap().as_ref(), Some(&bob));
 // Commit 2's single parent is commit 1's CID.
 let head_v2 = repo_v2.head_commit().unwrap();
 assert_eq!(head_v2.parents.len(), 1);
 assert_eq!(head_v2.parents[0], v1_head_cid);
 }

 // ---------- tombstone_node ----------

 #[test]
 fn tombstone_round_trip_through_view() {
 // Contract: a tombstone written in one commit survives on the
 // View read back from the next `ReadonlyRepo`, carrying the
 // caller's reason + the commit's resolved timestamp.
 let repo = new_repo();
 let alice =
 Node::new(NodeId::new_v7(), "Person").with_prop("name", Ipld::String("Alice".into()));

 let mut tx1 = repo.start_transaction();
 tx1.add_node(&alice).unwrap();
 let repo_v1 = tx1.commit("t", "seed").unwrap();
 // Pre-tombstone: no entry on the view.
 assert!(!repo_v1.is_tombstoned(&alice.id));

 // Tombstone in a second commit so the timestamp field is
 // stamped from that commit's `now`.
 let mut tx2 = repo_v1.start_transaction();
 tx2.tombstone_node(alice.id, "user asked to forget")
 .unwrap();
 let repo_v2 = tx2.commit("t", "revoke alice").unwrap();

 // The original node block is still addressable - CID unchanged,
 // lookup still returns it.
 assert_eq!(
 repo_v2.lookup_node(&alice.id).unwrap().as_ref(),
 Some(&alice)
 );
 // But the View now carries a tombstone for this id.
 assert!(repo_v2.is_tombstoned(&alice.id));
 let ts = repo_v2.tombstone_for(&alice.id).expect("tombstone present");
 assert_eq!(ts.reason, "user asked to forget");
 assert!(
 ts.tombstoned_at > 0,
 "tombstone_at must be set from commit's resolved now, got 0"
 );
 }

 #[test]
 fn tombstone_is_idempotent_within_a_transaction() {
 // Calling tombstone_node twice for the same id in one
 // transaction collapses to a single View entry; the second
 // reason overwrites the first (deterministic rule).
 let repo = new_repo();
 let alice = Node::new(NodeId::new_v7(), "Person");

 let mut tx1 = repo.start_transaction();
 tx1.add_node(&alice).unwrap();
 let repo_v1 = tx1.commit("t", "seed").unwrap();

 let mut tx2 = repo_v1.start_transaction();
 tx2.tombstone_node(alice.id, "first").unwrap();
 tx2.tombstone_node(alice.id, "second").unwrap();
 let repo_v2 = tx2.commit("t", "revoke").unwrap();

 assert_eq!(repo_v2.view().tombstones.len(), 1);
 let ts = repo_v2.tombstone_for(&alice.id).unwrap();
 assert_eq!(
 ts.reason, "second",
 "later tombstone_node call wins within one transaction"
 );
 }

 #[test]
 fn tombstone_leaves_node_cid_stable() {
 // Contract: tombstoning a node does NOT alter the CID that
 // `lookup_node` resolves to. Agents that persisted the CID of
 // a node outside mnem can still fetch the same bytes after a
 // tombstone commit. This is the core reason tombstones exist
 // as a side-channel on the View rather than as a mutation of
 // the node block.
 use crate::codec::hash_to_cid;

 let repo = new_repo();
 let alice =
 Node::new(NodeId::new_v7(), "Person").with_prop("name", Ipld::String("Alice".into()));

 let mut tx1 = repo.start_transaction();
 tx1.add_node(&alice).unwrap();
 let repo_v1 = tx1.commit("t", "seed").unwrap();
 let alice_before = repo_v1.lookup_node(&alice.id).unwrap().unwrap();
 let (_bytes_before, cid_before) = hash_to_cid(&alice_before).unwrap();

 let mut tx2 = repo_v1.start_transaction();
 tx2.tombstone_node(alice.id, "revoked").unwrap();
 let repo_v2 = tx2.commit("t", "revoke").unwrap();

 let alice_after = repo_v2.lookup_node(&alice.id).unwrap().unwrap();
 let (_bytes_after, cid_after) = hash_to_cid(&alice_after).unwrap();
 assert_eq!(
 cid_before, cid_after,
 "tombstone must not change the node's content-addressed CID"
 );
 assert_eq!(
 alice_before, alice_after,
 "tombstone must not mutate node content"
 );
 }

 #[test]
 fn remove_node_leaves_tree_without_it() {
 let repo = new_repo();
 let alice =
 Node::new(NodeId::new_v7(), "Person").with_prop("name", Ipld::String("Alice".into()));
 let mut tx1 = repo.start_transaction();
 tx1.add_node(&alice).unwrap();
 let v1 = tx1.commit("a", "add").unwrap();
 assert!(v1.lookup_node(&alice.id).unwrap().is_some());

 let mut tx2 = v1.start_transaction();
 tx2.remove_node(alice.id);
 let v2 = tx2.commit("a", "remove").unwrap();
 assert!(v2.lookup_node(&alice.id).unwrap().is_none());
 }

 #[test]
 fn ref_update_is_visible_on_the_new_view() {
 let repo = new_repo();
 let raw_cid = Cid::new(CODEC_RAW, Multihash::sha2_256(b"target"));

 let mut tx = repo.start_transaction();
 tx.update_ref("refs/heads/main", Some(RefTarget::normal(raw_cid.clone())));
 let v1 = tx.commit("a", "set main").unwrap();
 match v1.view().refs.get("refs/heads/main") {
 Some(RefTarget::Normal { target }) => assert_eq!(*target, raw_cid),
 other => panic!("expected normal ref, got {other:?}"),
 }
 }

 #[test]
 fn op_heads_advances_on_commit() {
 let repo = new_repo();
 let ohs = repo.op_heads_store().clone();
 assert_eq!(ohs.current().unwrap().len(), 1);
 let before_head = ohs.current().unwrap()[0].clone();

 let mut tx = repo.start_transaction();
 let alice = Node::new(NodeId::new_v7(), "Person");
 tx.add_node(&alice).unwrap();
 let v1 = tx.commit("a", "m").unwrap();

 let after_heads = ohs.current().unwrap();
 assert_eq!(after_heads.len(), 1);
 assert_ne!(after_heads[0], before_head);
 assert_eq!(after_heads[0], *v1.op_id());
 }

 #[test]
 fn linearize_commit_succeeds_against_current_head() {
 let repo = new_repo();
 let mut tx = repo.start_transaction();
 tx.add_node(&Node::new(NodeId::new_v7(), "Person")).unwrap();
 let r = tx.commit_opts(CommitOptions {
 author: "a",
 message: "m",
 linearize: true,
 time_micros: None,
 change_id: None,
 });
 assert!(r.is_ok());
 }

 #[test]
 fn linearize_commit_rejects_stale_base() {
 let repo = new_repo();

 // Start a transaction against the initial state.
 let mut stale_tx = repo.start_transaction();
 stale_tx
 .add_node(&Node::new(NodeId::new_v7(), "Ghost"))
 .unwrap();

 // A concurrent writer commits, advancing op-heads.
 let mut other_tx = repo.start_transaction();
 other_tx
 .add_node(&Node::new(NodeId::new_v7(), "Person"))
 .unwrap();
 let _ = other_tx.commit("a", "concurrent").unwrap();

 // The stale transaction commits in linearize mode -> Stale.
 let err = stale_tx
 .commit_opts(CommitOptions {
 author: "a",
 message: "from stale",
 linearize: true,
 time_micros: None,
 change_id: None,
 })
 .unwrap_err();
 assert!(matches!(err, Error::Repo(RepoError::Stale)));
 }

 #[test]
 fn default_commit_against_stale_base_still_succeeds() {
 // The non-linearize default lets both writers append to op-heads;
 // the second commit simply lands as a concurrent head (to be
 // merged by M8.5).
 let repo = new_repo();

 let mut stale_tx = repo.start_transaction();
 stale_tx
 .add_node(&Node::new(NodeId::new_v7(), "Ghost"))
 .unwrap();

 let mut other_tx = repo.start_transaction();
 other_tx
 .add_node(&Node::new(NodeId::new_v7(), "Person"))
 .unwrap();
 let _ = other_tx.commit("a", "concurrent").unwrap();

 // Default mode succeeds even with a stale base.
 assert!(stale_tx.commit("a", "late but not linearized").is_ok());
 }

 #[test]
 fn deterministic_commit_opts_yield_identical_commit_cid() {
 // Contract: two processes that build the same logical commit
 // on disjoint fresh repos, with CommitOptions pinning
 // `time_micros` + `change_id`, MUST produce byte-identical
 // commit CIDs. This is the headline "deterministic across
 // machines" property extended to commits (previously the
 // guarantee applied only to node-tree + IndexSet).
 //
 // This is ALSO our Q0-migration safety net: if
 // `put_trusted` (added in the A2 -> Q0 migration) ever
 // silently corrupts a commit's serialized bytes, the head
 // CID recorded here would change and this test would break.
 // Changes to the fixed inputs below should be treated as a
 // correctness regression until explained.
 let fixed_id = NodeId::from_bytes_raw([0x42; 16]);
 let fixed_change_id = ChangeId::from_bytes_raw([0x11; 16]);
 let fixed_time: u64 = 1_700_000_000_000_000;

 let commit_once = || -> Cid {
 let repo = new_repo();
 let mut tx = repo.start_transaction();
 tx.add_node(&Node::new(fixed_id, "Person")).unwrap();
 let new_repo = tx
 .commit_opts(
 CommitOptions::new("alice", "seed")
 .with_time_micros(fixed_time)
 .with_change_id(fixed_change_id),
 )
 .unwrap();
 new_repo
 .view()
 .heads
 .first()
 .expect("one head after commit")
 .clone()
 };
 let a = commit_once();
 let b = commit_once();
 assert_eq!(
 a, b,
 "identical CommitOptions across fresh repos must produce identical commit CIDs"
 );
 }

 /// Fix X1 regression guard. Build the same graph two ways:
 /// (a) many append-only commits (trigger the incremental index
 /// fast path from the second commit onward),
 /// (b) one big commit that holds the full graph (hits the
 /// first-commit full-rebuild path).
 /// Both must produce byte-identical `IndexSet` CIDs. If not, the
 /// incremental path has drifted from the slow-path output and
 /// queries would silently diverge.
 #[test]
 fn incremental_and_full_index_build_produce_identical_index_set() {
 // Helper: ingest `batches` of `per_batch` nodes each, one
 // commit per batch. The first commit hits the full rebuild
 // (no base IndexSet); every subsequent commit hits the
 // incremental append path because all gating conditions
 // (no removals, no edges, no NodeId collision) are satisfied.
 fn incremental(batches: usize, per_batch: usize, ids: &[NodeId]) -> Cid {
 let bs: Arc<dyn Blockstore> = Arc::new(MemoryBlockstore::new());
 let ohs: Arc<dyn OpHeadsStore> = Arc::new(MemoryOpHeadsStore::new());
 let mut repo = ReadonlyRepo::init(bs, ohs).unwrap();
 for b in 0..batches {
 let mut tx = repo.start_transaction();
 for i in 0..per_batch {
 let id = ids[b * per_batch + i];
 let node = Node::new(id, "Person")
 .with_prop("name", Ipld::String(format!("p{i}")))
 .with_prop("batch", Ipld::Integer(b as i128));
 tx.add_node(&node).unwrap();
 }
 repo = tx.commit("t", "batch").unwrap();
 }
 repo.head_commit().unwrap().indexes.clone().unwrap()
 }

 fn full(total: usize, ids: &[NodeId]) -> Cid {
 let bs: Arc<dyn Blockstore> = Arc::new(MemoryBlockstore::new());
 let ohs: Arc<dyn OpHeadsStore> = Arc::new(MemoryOpHeadsStore::new());
 let repo = ReadonlyRepo::init(bs, ohs).unwrap();
 let mut tx = repo.start_transaction();
 let per_batch = 10;
 for i in 0..total {
 let batch_of = i / per_batch;
 let in_batch = i % per_batch;
 let node = Node::new(ids[i], "Person")
 .with_prop("name", Ipld::String(format!("p{in_batch}")))
 .with_prop("batch", Ipld::Integer(batch_of as i128));
 tx.add_node(&node).unwrap();
 }
 tx.commit("t", "one-shot")
 .unwrap()
 .head_commit()
 .unwrap()
 .indexes
 .clone()
 .unwrap()
 }

 // Deterministic id set so both paths commit the same graph.
 // Using from_bytes_raw keeps the ids ordering predictable.
 let total = 30;
 let ids: Vec<NodeId> = (0..total)
 .map(|i| {
 let mut b = [0u8; 16];
 b[0] = i as u8;
 NodeId::from_bytes_raw(b)
 })
 .collect();

 let inc = incremental(3, 10, &ids);
 let one = full(30, &ids);
 assert_eq!(
 inc, one,
 "incremental index build must produce the same IndexSet CID as the full rebuild"
 );
 }

 /// Companion to the test above: when the graph has edges (so
 /// `outgoing` and `incoming` trees are actually populated), the
 /// incremental-append path must preserve BOTH direction CIDs
 /// byte-for-byte, not just the nodes side.
 #[test]
 fn incremental_and_full_preserve_both_direction_adjacency_cids() {
 let ids: Vec<NodeId> = (0u8..10u8)
 .map(|i| {
 let mut b = [0u8; 16];
 b[0] = i;
 NodeId::from_bytes_raw(b)
 })
 .collect();
 let edge_pairs: &[(usize, usize, u8)] =
 &[(0, 1, 0xA0), (1, 2, 0xA1), (2, 3, 0xA2), (0, 5, 0xA3)];

 // Incremental: first commit has nodes+edges, then pure-node
 // appends hit the fast path.
 let (bs, ohs): (Arc<dyn Blockstore>, Arc<dyn OpHeadsStore>) = (
 Arc::new(MemoryBlockstore::new()),
 Arc::new(MemoryOpHeadsStore::new()),
 );
 let repo_inc = ReadonlyRepo::init(bs, ohs).unwrap();
 let mut tx = repo_inc.start_transaction();
 for id in &ids {
 tx.add_node(&Node::new(*id, "Person")).unwrap();
 }
 for (s, d, tag) in edge_pairs {
 let mut eb = [0u8; 16];
 eb[0] = *tag;
 tx.add_edge(&crate::objects::Edge::new(
 crate::id::EdgeId::from_bytes_raw(eb),
 "knows",
 ids[*s],
 ids[*d],
 ))
 .unwrap();
 }
 let mut repo_inc = tx.commit("t", "seed").unwrap();
 for extra in 0u8..3 {
 let mut tx = repo_inc.start_transaction();
 let mut b = [0u8; 16];
 b[0] = 0xEE;
 b[1] = extra;
 tx.add_node(&Node::new(NodeId::from_bytes_raw(b), "Person"))
 .unwrap();
 repo_inc = tx.commit("t", "append").unwrap();
 }
 let idx_inc_cid = repo_inc.head_commit().unwrap().indexes.clone().unwrap();
 let idx_inc: crate::objects::IndexSet =
 crate::repo::readonly::decode_from_store(&**repo_inc.blockstore(), &idx_inc_cid)
 .unwrap();

 // Full: single commit with all nodes (core + extras) + edges.
 let (bs, ohs): (Arc<dyn Blockstore>, Arc<dyn OpHeadsStore>) = (
 Arc::new(MemoryBlockstore::new()),
 Arc::new(MemoryOpHeadsStore::new()),
 );
 let repo_full = ReadonlyRepo::init(bs, ohs).unwrap();
 let mut tx = repo_full.start_transaction();
 for id in &ids {
 tx.add_node(&Node::new(*id, "Person")).unwrap();
 }
 for extra in 0u8..3 {
 let mut b = [0u8; 16];
 b[0] = 0xEE;
 b[1] = extra;
 tx.add_node(&Node::new(NodeId::from_bytes_raw(b), "Person"))
 .unwrap();
 }
 for (s, d, tag) in edge_pairs {
 let mut eb = [0u8; 16];
 eb[0] = *tag;
 tx.add_edge(&crate::objects::Edge::new(
 crate::id::EdgeId::from_bytes_raw(eb),
 "knows",
 ids[*s],
 ids[*d],
 ))
 .unwrap();
 }
 let repo_full = tx.commit("t", "one-shot").unwrap();
 let idx_full_cid = repo_full.head_commit().unwrap().indexes.clone().unwrap();
 let idx_full: crate::objects::IndexSet =
 crate::repo::readonly::decode_from_store(&**repo_full.blockstore(), &idx_full_cid)
 .unwrap();

 assert_eq!(
 idx_inc.outgoing, idx_full.outgoing,
 "incremental path must preserve the outgoing CID byte-for-byte"
 );
 assert_eq!(
 idx_inc.incoming, idx_full.incoming,
 "incremental path must preserve the incoming CID byte-for-byte"
 );
 assert_eq!(
 idx_inc_cid, idx_full_cid,
 "whole-IndexSet CID must also be byte-equal"
 );
 }

 // -------- embedding sidecar () --------

 fn dummy_embedding(model: &str, dim: u32) -> Embedding {
 let bytes_len = (dim as usize) * crate::objects::node::Dtype::F32.byte_width();
 Embedding {
 model: model.into(),
 dtype: crate::objects::node::Dtype::F32,
 dim,
 vector: bytes::Bytes::from(vec![0u8; bytes_len]),
 }
 }

 /// happy path: stage an embedding via `set_embedding`,
 /// commit, then read it back via `embedding_for`. End-to-end
 /// proof the write side and the read side agree on the Prolly
 /// key derivation.
 #[test]
 fn set_embedding_round_trips_through_commit() {
 let repo = new_repo();
 let mut tx = repo.start_transaction();
 let node = Node::new(NodeId::new_v7(), "Doc").with_summary("hello");
 let node_cid = tx.add_node(&node).unwrap();
 let emb = dummy_embedding("onnx:test", 4);
 tx.set_embedding(node_cid.clone(), "onnx:test".into(), emb.clone())
 .unwrap();
 let r2 = tx.commit("alice", "stage embed").unwrap();

 // Sidecar root populated on the new commit.
 assert!(r2.head_commit().unwrap().embeddings.is_some());

 // Lookup returns the staged embedding.
 let got = r2.embedding_for(&node_cid, "onnx:test").unwrap();
 assert_eq!(got, Some(emb));

 // Wrong model returns None, not error.
 assert_eq!(r2.embedding_for(&node_cid, "missing-model").unwrap(), None);
 }

 /// One node may carry multiple embeddings simultaneously (e.g.
 /// MiniLM + bge-base for the same chunk). The bucket holds both,
 /// keyed by `model`.
 #[test]
 fn set_embedding_multiple_models_per_node() {
 let repo = new_repo();
 let mut tx = repo.start_transaction();
 let node = Node::new(NodeId::new_v7(), "Doc").with_summary("two-model node");
 let node_cid = tx.add_node(&node).unwrap();
 let emb_a = dummy_embedding("model-a", 4);
 let emb_b = dummy_embedding("model-b", 8);
 tx.set_embedding(node_cid.clone(), "model-a".into(), emb_a.clone())
 .unwrap();
 tx.set_embedding(node_cid.clone(), "model-b".into(), emb_b.clone())
 .unwrap();
 let r2 = tx.commit("alice", "two embeds").unwrap();

 assert_eq!(r2.embedding_for(&node_cid, "model-a").unwrap(), Some(emb_a));
 assert_eq!(r2.embedding_for(&node_cid, "model-b").unwrap(), Some(emb_b));
 }

 /// A commit with zero pending embeddings AND no base sidecar
 /// must leave `commit.embeddings = None` so legacy commits stay
 /// byte-identical.
 #[test]
 fn commit_without_set_embedding_has_none_embeddings_root() {
 let repo = new_repo();
 let mut tx = repo.start_transaction();
 let node = Node::new(NodeId::new_v7(), "Doc").with_summary("no embed");
 tx.add_node(&node).unwrap();
 let r2 = tx.commit("alice", "no embed").unwrap();

 assert_eq!(r2.head_commit().unwrap().embeddings, None);
 }

 /// Second commit on top of a sidecar-bearing base must inherit
 /// the existing entries and add the new one. Lookup of either
 /// (old or new) NodeCid succeeds against the new repo.
 #[test]
 fn second_commit_inherits_and_extends_embedding_sidecar() {
 let repo = new_repo();

 // Tx 1: add node A + its embedding, commit.
 let mut tx1 = repo.start_transaction();
 let node_a = Node::new(NodeId::new_v7(), "Doc").with_summary("a");
 let cid_a = tx1.add_node(&node_a).unwrap();
 let emb_a = dummy_embedding("onnx:a", 4);
 tx1.set_embedding(cid_a.clone(), "onnx:a".into(), emb_a.clone())
 .unwrap();
 let r1 = tx1.commit("alice", "first").unwrap();
 assert!(r1.head_commit().unwrap().embeddings.is_some());

 // Tx 2: add node B + its embedding on top of r1.
 let mut tx2 = r1.start_transaction();
 let node_b = Node::new(NodeId::new_v7(), "Doc").with_summary("b");
 let cid_b = tx2.add_node(&node_b).unwrap();
 let emb_b = dummy_embedding("onnx:b", 4);
 tx2.set_embedding(cid_b.clone(), "onnx:b".into(), emb_b.clone())
 .unwrap();
 let r2 = tx2.commit("alice", "second").unwrap();

 // Both lookups must succeed against r2.
 assert_eq!(r2.embedding_for(&cid_a, "onnx:a").unwrap(), Some(emb_a));
 assert_eq!(r2.embedding_for(&cid_b, "onnx:b").unwrap(), Some(emb_b));
 }

 /// Determinism: staging the same set of (NodeCid, model, embedding)
 /// triples in different orders must produce byte-identical
 /// `commit.embeddings` Cids. Pins the canonical-form contract for
 /// the sidecar tree.
 #[test]
 fn embedding_sidecar_root_is_insertion_order_invariant() {
 // Two repos, same Node + Embedding writes in different order.
 let make = |order: u8| -> Cid {
 let repo = new_repo();
 let mut tx = repo.start_transaction();
 let n1 = Node::new(NodeId::from_bytes_raw([1u8; 16]), "Doc").with_summary("n1");
 let n2 = Node::new(NodeId::from_bytes_raw([2u8; 16]), "Doc").with_summary("n2");
 let n3 = Node::new(NodeId::from_bytes_raw([3u8; 16]), "Doc").with_summary("n3");
 let c1 = tx.add_node(&n1).unwrap();
 let c2 = tx.add_node(&n2).unwrap();
 let c3 = tx.add_node(&n3).unwrap();
 let e1 = dummy_embedding("m", 4);
 let e2 = dummy_embedding("m", 4);
 let e3 = dummy_embedding("m", 4);
 // Same logical writes, three permutations.
 match order {
 0 => {
 tx.set_embedding(c1, "m".into(), e1).unwrap();
 tx.set_embedding(c2, "m".into(), e2).unwrap();
 tx.set_embedding(c3, "m".into(), e3).unwrap();
 }
 1 => {
 tx.set_embedding(c3, "m".into(), e3).unwrap();
 tx.set_embedding(c1, "m".into(), e1).unwrap();
 tx.set_embedding(c2, "m".into(), e2).unwrap();
 }
 _ => {
 tx.set_embedding(c2, "m".into(), e2).unwrap();
 tx.set_embedding(c3, "m".into(), e3).unwrap();
 tx.set_embedding(c1, "m".into(), e1).unwrap();
 }
 }
 let r = tx.commit("alice", "det").unwrap();
 r.head_commit().unwrap().embeddings.clone().unwrap()
 };
 let cid_a = make(0);
 let cid_b = make(1);
 let cid_c = make(2);
 assert_eq!(
 cid_a, cid_b,
 "sidecar root must be insertion-order-invariant"
 );
 assert_eq!(cid_a, cid_c);
 }
}
