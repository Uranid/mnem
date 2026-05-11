//! Stubs for verbs that are part of the Git-verb spine but
//! whose implementation is deferred past Q2.
//!
//! Each stub:
//!
//! - is advertised in `mnem --help` so `mnem <TAB>` completion shows
//!   the full vocabulary
//! - fails with exit code 78 (EX_CONFIG per BSD sysexits - "something
//!   the user did was wrong, not a transient I/O error")
//! - points at docs/ROADMAP.md so the next step is obvious
//!
//! The stubs deliberately accept the arguments the real
//! implementation will - clap catches unknown flags, and a user who
//! has already built a `mnem pull origin main` habit gets the same
//! parse error they'll get post-PR 3 if they pass something wrong.

use mnem_core::prolly::{DiffEntry, diff as prolly_diff};

use super::*;

// Real implementations of `fetch` / `push` / `pull` live in
// `commands::fetch`, `commands::push`, `commands::pull`.
//
// `mnem merge` is wired through `commands::merge`.
// `mnem gc` is wired through `commands::gc`.
// `mnem revert` is implemented below.

#[derive(clap::Args, Debug)]
#[command(after_long_help = "\
Walks the op-log to find the target op and its parent, computes the inverse
of the op's changes (nodes/edges added are removed, removed are re-created,
changed are rolled back to the before state), and creates a new commit.

Examples:
  mnem revert <op-cid>           # find op-cid via `mnem log`
  mnem revert <op-cid> -m \"rollback bad ingest\"
")]
pub(crate) struct RevertArgs {
    /// Op CID to invert (find with `mnem log`).
    pub commit: String,
    /// Commit message for the revert operation.
    #[arg(long, short = 'm')]
    pub message: Option<String>,
}

/// Walk the op-log backwards from `start` up to `max_depth` ops,
/// returning `(target_op, parent_op)` when the target CID is found.
/// `parent_op` is the op immediately preceding the target in the
/// chain (i.e. `target_op.parents[0]`), NOT the op that came after it.
/// Returns `None` if the op is not reachable.
fn find_op_and_parent(
    bs: &dyn mnem_core::store::Blockstore,
    start: &mnem_core::id::Cid,
    target_cid: &mnem_core::id::Cid,
) -> Result<Option<(Operation, Option<Operation>)>> {
    let mut cur = start.clone();

    // Safety cap: walking at most 100 000 ops. In practice repos have
    // far fewer, but a corrupted parent pointer could loop forever.
    for _ in 0..100_000usize {
        let bytes = bs
            .get(&cur)?
            .ok_or_else(|| anyhow!("op {cur} missing from blockstore"))?;
        let op: Operation = from_canonical_bytes(&bytes)?;

        if &cur == target_cid {
            // Found the target. The "before" state is what the target op's
            // own parent op recorded - load it directly from target_op.parents.
            let parent_op: Option<Operation> = match op.parents.first() {
                None => None, // root op has no parent
                Some(parent_cid) => {
                    let pbytes = bs
                        .get(parent_cid)?
                        .ok_or_else(|| anyhow!("parent op {parent_cid} missing from blockstore"))?;
                    Some(from_canonical_bytes(&pbytes)?)
                }
            };
            return Ok(Some((op, parent_op)));
        }

        match op.parents.first() {
            Some(p) => cur = p.clone(),
            None => break, // root op reached without finding target
        }
    }
    Ok(None)
}

pub(crate) fn run_revert(override_path: Option<&Path>, args: RevertArgs) -> Result<()> {
    // 1. Parse the target op-CID.
    let target_cid = mnem_core::id::Cid::parse_str(&args.commit)
        .with_context(|| format!("invalid CID: `{}`", args.commit))?;

    // 2. Open the repo.
    let (data_dir, repo, bs, _ohs) = repo::open_all(override_path)?;

    // 3. Walk the op-log to find the target op and its parent.
    let head_op_cid = repo.op_id().clone();
    let (target_op, parent_op_opt) = find_op_and_parent(&*bs, &head_op_cid, &target_cid)?
        .ok_or_else(|| {
            anyhow!(
                "op `{}` not found in the op-log. Use `mnem log` to list available ops.",
                args.commit
            )
        })?;

    // 4. Resolve the commit CIDs for the target op and its parent.
    //    Each op's `view` points at a View, which has `heads` (commit CIDs).
    let target_view: mnem_core::objects::View =
        from_canonical_bytes(&bs.get(&target_op.view)?.ok_or_else(|| {
            anyhow!(
                "view block for op `{}` missing from blockstore",
                args.commit
            )
        })?)?;

    let target_commit_cid = target_view.heads.first().ok_or_else(|| {
        anyhow!(
            "op `{}` has no head commit - it is an empty op (init or ref-only). Nothing to revert.",
            args.commit
        )
    })?;

    // Resolve the parent view and commit CID (before-state). If there is no
    // parent op (the target is the root op), we revert against an empty graph.
    let (parent_view_opt, parent_commit_cid_opt): (
        Option<mnem_core::objects::View>,
        Option<mnem_core::id::Cid>,
    ) = match &parent_op_opt {
        None => (None, None),
        Some(pop) => {
            let pview: mnem_core::objects::View = from_canonical_bytes(
                &bs.get(&pop.view)?
                    .ok_or_else(|| anyhow!("view block for parent op missing from blockstore"))?,
            )?;
            let pcid = pview.heads.first().cloned();
            (Some(pview), pcid)
        }
    };

    // 5. Load both commits and compute the prolly-tree diff.
    let target_commit: Commit =
        from_canonical_bytes(&bs.get(target_commit_cid)?.ok_or_else(|| {
            anyhow!("commit block `{target_commit_cid}` missing from blockstore")
        })?)?;

    // "Before" Prolly tree roots: the parent commit, or empty trees if
    // we are reverting the very first commit.
    let (before_nodes_root, before_edges_root) = match &parent_commit_cid_opt {
        Some(pcid) => {
            let pc: Commit =
                from_canonical_bytes(&bs.get(pcid)?.ok_or_else(|| {
                    anyhow!("parent commit block `{pcid}` missing from blockstore")
                })?)?;
            (pc.nodes.clone(), pc.edges.clone())
        }
        None => {
            // Root commit: the "before" state is an empty prolly tree.
            // We need the empty-tree CID that the repo uses. The simplest
            // approach: build one via the prolly builder. Since the blockstore
            // is already open and shared (Arc), we can write the empty tree
            // there safely.
            let empty = mnem_core::prolly::build_tree(&*bs, std::iter::empty())?;
            (empty.clone(), empty)
        }
    };

    // Diff: before (parent) vs after (target). We invert this.
    let node_changes = prolly_diff(&*bs, &before_nodes_root, &target_commit.nodes)?;
    let edge_changes = prolly_diff(&*bs, &before_edges_root, &target_commit.edges)?;

    // BUG-3: compute tombstone diff between the parent view and the target
    // view. Tombstones live on the View block, not in the Prolly tree, so
    // they are invisible to prolly_diff. We need to revert them separately.
    //
    // "tombstones added by the op" = keys present in target_view.tombstones
    // that were NOT present in the parent view (or the parent had no view).
    let parent_tombstones: std::collections::BTreeMap<
        mnem_core::id::NodeId,
        mnem_core::objects::Tombstone,
    > = parent_view_opt
        .as_ref()
        .map(|pv| pv.tombstones.clone())
        .unwrap_or_default();

    // Tombstones added by the op = in target but not in parent.
    let tombstones_added_by_op: Vec<mnem_core::id::NodeId> = target_view
        .tombstones
        .keys()
        .filter(|id| !parent_tombstones.contains_key(*id))
        .copied()
        .collect();

    // Summarise what the revert will do.
    let nodes_added_by_op = node_changes
        .iter()
        .filter(|e| matches!(e, DiffEntry::Added { .. }))
        .count();
    let nodes_removed_by_op = node_changes
        .iter()
        .filter(|e| matches!(e, DiffEntry::Removed { .. }))
        .count();
    let nodes_changed_by_op = node_changes
        .iter()
        .filter(|e| matches!(e, DiffEntry::Changed { .. }))
        .count();
    let edges_added_by_op = edge_changes
        .iter()
        .filter(|e| matches!(e, DiffEntry::Added { .. }))
        .count();
    let edges_removed_by_op = edge_changes
        .iter()
        .filter(|e| matches!(e, DiffEntry::Removed { .. }))
        .count();
    let edges_changed_by_op = edge_changes
        .iter()
        .filter(|e| matches!(e, DiffEntry::Changed { .. }))
        .count();

    // BUG-3 fix: also check tombstone changes so a tombstone-only op is
    // not mistakenly reported as "nothing to revert".
    if node_changes.is_empty() && edge_changes.is_empty() && tombstones_added_by_op.is_empty() {
        println!(
            "op `{}` made no node/edge/tombstone changes - nothing to revert.",
            args.commit
        );
        return Ok(());
    }

    println!("reverting op: {}", args.commit);
    println!(
        "  nodes: {} added, {} removed, {} changed by the original op",
        nodes_added_by_op, nodes_removed_by_op, nodes_changed_by_op
    );
    println!(
        "  edges: {} added, {} removed, {} changed by the original op",
        edges_added_by_op, edges_removed_by_op, edges_changed_by_op
    );
    println!(
        "  tombstones: {} added by the original op (will be removed)",
        tombstones_added_by_op.len()
    );
    println!("applying inverse changes...");

    // 6. Build a transaction that inverts the op's changes.
    let cfg = config::load(&data_dir)?;
    let author = config::author_string(&cfg);

    // BUG-4 pre-flight: before touching the repo, verify that every edge
    // the revert would RE-ADD (i.e. the op removed it, so the inverse re-adds
    // it) still has both endpoints alive in the current view. If either
    // endpoint has since been deleted (hard-removed or tombstoned), committing
    // the edge would produce a DanglingEdge error - but at that point we have
    // already written partial state. Bail early with a clear message instead.
    for entry in &edge_changes {
        if let DiffEntry::Removed { value, .. } = entry {
            // Op removed this edge -> revert would re-add it.
            let edge: Edge = from_canonical_bytes(
                &bs.get(value)?
                    .ok_or_else(|| anyhow!("edge block `{value}` missing"))?,
            )?;
            for (endpoint_id, role) in [(edge.src, "src"), (edge.dst, "dst")] {
                let exists = repo.lookup_node(&endpoint_id)?.is_some();
                let tombstoned = repo.is_tombstoned(&endpoint_id);
                if !exists || tombstoned {
                    bail!(
                        "cannot revert op `{}`: edge endpoint {} ({role}) no longer exists \
                         (deleted or tombstoned since the op was applied). \
                         Revert the deletion first, or skip reverting this op.",
                        args.commit,
                        endpoint_id
                    );
                }
            }
        }
    }

    let mut tx = repo.start_transaction();
    let mut mutations_applied: usize = 0;

    // Invert node changes.
    for entry in &node_changes {
        match entry {
            DiffEntry::Added { value, .. } => {
                // Op added this node -> revert removes it.
                // Skip if it is already absent from the current tree (no-op).
                let node: Node = from_canonical_bytes(
                    &bs.get(value)?
                        .ok_or_else(|| anyhow!("node block `{value}` missing"))?,
                )?;
                if tx.base().lookup_node(&node.id)?.is_some() {
                    tx.remove_node(node.id);
                    mutations_applied += 1;
                }
            }
            DiffEntry::Removed { value, .. } => {
                // Op removed this node -> revert re-adds it.
                // Skip if it already exists in the current tree (no-op).
                let node: Node = from_canonical_bytes(
                    &bs.get(value)?
                        .ok_or_else(|| anyhow!("node block `{value}` missing"))?,
                )?;
                if tx.base().lookup_node(&node.id)?.is_none() {
                    tx.add_node(&node)?;
                    mutations_applied += 1;
                }
            }
            DiffEntry::Changed { before, .. } => {
                // Op changed this node -> revert restores the before version.
                // Skip if the current tree already holds the before version (no-op).
                let node: Node = from_canonical_bytes(
                    &bs.get(before)?
                        .ok_or_else(|| anyhow!("node block `{before}` missing"))?,
                )?;
                // A no-op means the current tree already has this exact node CID.
                // We detect that by checking whether the current node's CID equals
                // `before`; if the tree currently holds `after` we need to revert.
                let current_is_before = match tx.base().lookup_node(&node.id)? {
                    None => false,
                    Some(ref cur) => {
                        // Compare via the after CID: if the current node equals
                        // the after-state, we still need to revert.  Use the
                        // simple approach: if it does NOT equal the before-node
                        // structurally, apply the mutation.
                        cur == &node
                    }
                };
                if !current_is_before {
                    tx.add_node(&node)?;
                    mutations_applied += 1;
                }
            }
        }
    }

    // Invert edge changes.
    for entry in &edge_changes {
        match entry {
            DiffEntry::Added { value, .. } => {
                // Op added this edge -> revert removes it.
                // Skip if already absent (no-op).
                let edge: Edge = from_canonical_bytes(
                    &bs.get(value)?
                        .ok_or_else(|| anyhow!("edge block `{value}` missing"))?,
                )?;
                if tx.base().lookup_edge(&edge.id)?.is_some() {
                    tx.remove_edge(edge.id);
                    mutations_applied += 1;
                }
            }
            DiffEntry::Removed { value, .. } => {
                // Op removed this edge -> revert re-adds it.
                // Skip if already present (no-op). Endpoint existence was
                // verified in the BUG-4 pre-flight check above.
                let edge: Edge = from_canonical_bytes(
                    &bs.get(value)?
                        .ok_or_else(|| anyhow!("edge block `{value}` missing"))?,
                )?;
                if tx.base().lookup_edge(&edge.id)?.is_none() {
                    tx.add_edge(&edge)?;
                    mutations_applied += 1;
                }
            }
            DiffEntry::Changed { before, .. } => {
                // Op changed this edge -> revert restores the before version.
                let edge: Edge = from_canonical_bytes(
                    &bs.get(before)?
                        .ok_or_else(|| anyhow!("edge block `{before}` missing"))?,
                )?;
                let current_is_before = match tx.base().lookup_edge(&edge.id)? {
                    None => false,
                    Some(ref cur) => cur == &edge,
                };
                if !current_is_before {
                    tx.add_edge(&edge)?;
                    mutations_applied += 1;
                }
            }
        }
    }

    // BUG-3: invert tombstone changes. For each tombstone the op ADDED
    // (present in target_view but absent in parent_view), the revert must
    // un-tombstone that node so the View no longer carries the marker.
    for node_id in &tombstones_added_by_op {
        // Skip if the current view no longer carries this tombstone (already
        // un-tombstoned by an earlier revert or superseding op).
        if tx.base().is_tombstoned(node_id) {
            tx.untombstone_node(*node_id);
            mutations_applied += 1;
        }
    }

    // 7. Guard: if no mutations were actually applied the inverse changes are
    //    all no-ops in the current tree - the op was most likely already
    //    reverted.  Avoid creating a ghost empty-delta commit.
    if mutations_applied == 0 {
        println!(
            "note: the inverse changes are all no-ops in the current tree \
             (the op may have already been reverted)."
        );
        println!("      nothing to commit.");
        return Ok(());
    }

    // 8. Commit the revert.
    let default_msg = format!("revert: {}", args.commit);
    let msg = args.message.as_deref().unwrap_or(&default_msg);
    let new_repo = tx.commit(&author, msg)?;

    println!("done.");
    println!("  new op:    {}", new_repo.op_id());
    if let Some(head) = new_repo.view().heads.first() {
        println!("  new commit: {head}");
    }
    Ok(())
}
