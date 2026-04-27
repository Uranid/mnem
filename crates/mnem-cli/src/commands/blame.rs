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
//! commit FIRST wrote each back-link - needs an ops-by-object-CID
//! index that does not yet exist in core. That refinement is tracked
//! and deferred to a follow-up PR.
//!
//! Output columns:
//!
//! ```text
//! edge_id                              etype    src (node-id)        in_commit
//! 019ab2f1-...                        authored 019a...               01HZABC...
//! ```
//!
//! When no incoming edges exist, prints `<no incoming edges>` and
//! returns success.
//!
//! # Examples
//!
//! ```text
//! mnem blame 019b8c...
//! mnem blame 019b8c... | awk '{print $3}' | sort -u   # distinct authors
//! ```

use super::*;

#[derive(clap::Args, Debug)]
#[command(after_long_help = "\
Examples:
  mnem blame <node-uuid>                    # list incoming edges
  mnem blame <node-uuid> --etype authored   # only one edge-type
")]
pub(crate) struct Args {
    /// UUID string of the destination node (dst of the incoming
    /// edges you want to list).
    pub node: String,
    /// Restrict to one edge-type label (e.g. `authored`, `cites`).
    #[arg(long)]
    pub etype: Option<String>,
}

pub(crate) fn run(override_path: Option<&Path>, args: Args) -> Result<()> {
    let (_dir, r, _bs, _ohs) = repo::open_all(override_path)?;

    let node_id = NodeId::parse_uuid(&args.node).context("parsing node id")?;

    let filter = args.etype.as_deref();
    let filter_slice = filter.map(|s| [s]);
    let filter_ref = filter_slice.as_ref().map(|arr| &arr[..]);
    let edges = r
        .incoming_edges(&node_id, filter_ref)
        .context("walking incoming-adjacency index")?;

    // The "in_commit" column is, in Q2, always the current head
    // commit - every edge we see lives in the current IndexSet so
    // the head commit is the latest that made it observable. A
    // future follow-up will thread the first-writer
    // commit for each edge through an ops-by-object-CID index.
    let head = r
        .view()
        .heads
        .first()
        .map_or_else(|| "<no-head>".into(), ToString::to_string);

    if edges.is_empty() {
        println!("<no incoming edges>");
        return Ok(());
    }
    // Column widths are tuned for the common 16-byte UUID + 36-char
    // rendering; if a future etype grows past the buffer, the line
    // just wraps.
    // Literal header row; widths chosen to match the 36-char UUID
    // + typical 16-char etype in the data rows below.
    println!(
        "{:<36}  {:<16}  {:<36}  in_commit",
        "edge_id", "etype", "src"
    );
    for e in &edges {
        println!(
            "{}  {:<16}  {}  {head}",
            e.id.to_uuid_string(),
            e.etype,
            e.src.to_uuid_string()
        );
    }
    Ok(())
}
