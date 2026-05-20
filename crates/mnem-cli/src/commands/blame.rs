//! `mnem blame <node-id>` - who points at this node.
//!
//! `blame` walks the **incoming-edge** index for the given `NodeId`
//! (dual-adjacency primitive added in R1 / ) and lists, for
//! each incoming edge, the edge type + the source node.
//!
//! Semantic note: `git blame` says "this LINE was written by THIS
//! COMMIT". mnem has no lines and no single-commit-per-write invariant
//! (dual identity, : content hash + stable ID). The honest
//! partial in Q2 is "every incoming edge, plus the current head commit
//! that made it observable". Fine-grained per-edge provenance - which
//! commit FIRST wrote each back-link - is provided by `--first-writer`,
//! which performs a BFS over the operation ancestry chain.
//!
//! Output columns (default):
//!
//! ```text
//! edge_id                              etype    src (node-id)        relation                                        in_commit
//! 019ab2f1-...                        authored 019a...               <src> -[authored]-> <dst>                        01HZABC...
//! ```
//!
//! With `--first-writer`:
//!
//! ```text
//! edge_id                              etype    src (node-id)        relation                                        first_writer
//! 019ab2f1-...                        authored 019a...               <src> -[authored]-> <dst>                        01HXYZ...
//! ```
//!
//! With `--no-relation` (narrow output, omits the `relation` column - useful
//! for narrow terminals and positional column-parsing scripts):
//!
//! ```text
//! edge_id                              etype    src (node-id)        in_commit
//! 019ab2f1-...                        authored 019a...               01HZABC...
//! ```
//!
//! When the requested node does not exist, a warning is printed to stderr and
//! the command exits zero with an empty-edges table; pass `--strict` to make
//! a missing node a hard error with a non-zero exit.
//!
//! When no incoming edges exist, prints `<no incoming edges>` and
//! returns success.
//!
//! # Examples
//!
//! ```text
//! mnem blame 019b8c...
//! mnem blame 019b8c... | awk '{print $3}' | sort -u   # distinct authors
//! mnem blame 019b8c... --first-writer                  # per-edge first-writer commit
//! mnem blame 019b8c... --no-relation                   # omit the relation column
//! mnem blame 019b8c... --strict                        # exit non-zero on unknown node
//! ```

use std::collections::{HashMap, HashSet, VecDeque};

use mnem_core::id::Cid;

use super::*;

#[derive(clap::Args, Debug)]
#[command(after_long_help = "\
Examples:
  mnem blame <node-uuid>                    # list incoming edges
  mnem blame <node-uuid> --etype authored   # only one edge-type
  mnem blame <node-uuid> --first-writer     # per-edge first-writer commit
  mnem blame <node-uuid> --no-relation      # omit the relation column
  mnem blame <node-uuid> --strict           # exit non-zero on unknown node
")]
pub(crate) struct Args {
    /// UUID string of the destination node (dst of the incoming
    /// edges you want to list).
    pub node: String,
    /// Restrict to one edge-type label (e.g. `authored`, `cites`).
    #[arg(long)]
    pub etype: Option<String>,
    /// For each incoming edge, walk the operation ancestry chain and
    /// show the commit CID of the oldest ancestor that first introduced
    /// that edge (BUG-55). O(depth × edges) - suitable for debugging.
    #[arg(long)]
    pub first_writer: bool,
    /// Hide the `relation` column (`<src> -[etype]-> <dst>`) to keep the
    /// human table narrower for terminals under ~170 cols or scripts that
    /// parse columns positionally. The remaining columns match the
    /// pre-#30 output shape.
    #[arg(long)]
    pub no_relation: bool,
    /// When the requested node does not exist, exit non-zero with a hard
    /// error. Without this flag, the command prints a stderr warning and
    /// exits zero with an empty-edges table (back-compat with the pre-#30
    /// behaviour).
    #[arg(long)]
    pub strict: bool,
}

/// BFS over ancestor operations to find the oldest commit CID that
/// contained each edge in `edges`.
///
/// All edges in `edges` start with `first_writer = current_head_commit`.
/// Each time an ancestor is found to contain the edge, the value is
/// overwritten with the ancestor's commit CID. After a full BFS the
/// remaining value is the deepest (oldest) ancestor that had the edge,
/// i.e. the operation that first wrote it.
///
/// Ancestor operations that fail to load (pruned blockstore, corruption)
/// are skipped with a stderr warning; their subtrees are not traversed.
fn compute_first_writers(
    r: &ReadonlyRepo,
    node_id: &NodeId,
    filter_ref: Option<&[&str]>,
    edges: &[Edge],
) -> anyhow::Result<HashMap<EdgeId, String>> {
    let current_commit = r
        .view()
        .heads
        .first()
        .map_or_else(|| "<no-head>".into(), ToString::to_string);

    let mut first_writer: HashMap<EdgeId, String> = edges
        .iter()
        .map(|e| (e.id, current_commit.clone()))
        .collect();

    let bs = r.blockstore().clone();
    let ohs = r.op_heads_store().clone();
    let mut visited: HashSet<Cid> = HashSet::new();
    let mut queue: VecDeque<Cid> = r.operation().parents.iter().cloned().collect();

    while let Some(ancestor_op_id) = queue.pop_front() {
        if !visited.insert(ancestor_op_id.clone()) {
            continue;
        }
        let ancestor = match ReadonlyRepo::load_at(bs.clone(), ohs.clone(), ancestor_op_id.clone())
        {
            Ok(a) => a,
            Err(err) => {
                eprintln!(
                    "warn: blame --first-writer: skipped ancestor op {ancestor_op_id}: {err}"
                );
                continue;
            }
        };
        let ancestor_commit = ancestor
            .view()
            .heads
            .first()
            .map_or_else(|| "<no-head>".into(), ToString::to_string);
        let ancestor_edges = ancestor
            .incoming_edges(node_id, filter_ref)
            .unwrap_or_default();
        let ancestor_ids: HashSet<EdgeId> = ancestor_edges.iter().map(|e| e.id).collect();
        for (edge_id, fw) in &mut first_writer {
            if ancestor_ids.contains(edge_id) {
                *fw = ancestor_commit.clone();
            }
        }
        queue.extend(ancestor.operation().parents.iter().cloned());
    }

    Ok(first_writer)
}

pub(crate) fn run(override_path: Option<&Path>, args: Args) -> Result<()> {
    let (_dir, r, _bs, _ohs) = repo::open_all(override_path)?;

    let node_id = NodeId::parse_uuid(&args.node).context("parsing node id")?;
    if r.lookup_node(&node_id)?.is_none() {
        if args.strict {
            bail!("no node with id={}", args.node);
        }
        eprintln!(
            "warning: no node with id={} (use --strict to fail with a non-zero exit)",
            args.node
        );
    }

    let filter = args.etype.as_deref();
    let filter_slice = filter.map(|s| [s]);
    let filter_ref = filter_slice.as_ref().map(|arr| &arr[..]);
    let edges = r
        .incoming_edges(&node_id, filter_ref)
        .context("walking incoming-adjacency index")?;

    if edges.is_empty() {
        println!("<no incoming edges>");
        return Ok(());
    }

    let show_relation = !args.no_relation;
    let relation_for = |e: &Edge| {
        format!(
            "{} -[{}]-> {}",
            e.src.to_uuid_string(),
            e.etype,
            node_id.to_uuid_string()
        )
    };

    if args.first_writer {
        let fw_map = compute_first_writers(&r, &node_id, filter_ref, &edges)?;
        if show_relation {
            println!(
                "{:<36}  {:<16}  {:<36}  {:<78}  first_writer",
                "edge_id", "etype", "src", "relation"
            );
        } else {
            println!(
                "{:<36}  {:<16}  {:<36}  first_writer",
                "edge_id", "etype", "src"
            );
        }
        for e in &edges {
            let fw = fw_map.get(&e.id).map(String::as_str).unwrap_or("<unknown>");
            if show_relation {
                println!(
                    "{:<36}  {:<16}  {:<36}  {:<78}  {fw}",
                    e.id.to_uuid_string(),
                    e.etype,
                    e.src.to_uuid_string(),
                    relation_for(e)
                );
            } else {
                println!(
                    "{:<36}  {:<16}  {:<36}  {fw}",
                    e.id.to_uuid_string(),
                    e.etype,
                    e.src.to_uuid_string()
                );
            }
        }
    } else {
        let head = r
            .view()
            .heads
            .first()
            .map_or_else(|| "<no-head>".into(), ToString::to_string);
        if show_relation {
            println!(
                "{:<36}  {:<16}  {:<36}  {:<78}  in_commit",
                "edge_id", "etype", "src", "relation"
            );
        } else {
            println!(
                "{:<36}  {:<16}  {:<36}  in_commit",
                "edge_id", "etype", "src"
            );
        }
        for e in &edges {
            if show_relation {
                println!(
                    "{:<36}  {:<16}  {:<36}  {:<78}  {head}",
                    e.id.to_uuid_string(),
                    e.etype,
                    e.src.to_uuid_string(),
                    relation_for(e)
                );
            } else {
                println!(
                    "{:<36}  {:<16}  {:<36}  {head}",
                    e.id.to_uuid_string(),
                    e.etype,
                    e.src.to_uuid_string()
                );
            }
        }
    }
    Ok(())
}
