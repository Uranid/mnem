//! `mnem traverse <node-uuid>` - list outgoing edges from a node.
//!
//! Mirrors the `mnem_traverse` MCP tool: given a starting node UUID,
//! lists outgoing edges optionally filtered by edge-type label and
//! bounded by `--limit`. Supports `--json` for machine-readable output.
//!
//! Output columns (human):
//!
//! ```text
//! node <uuid> (<ntype>)
//! -[<etype>]-> <dst-uuid>
//! -[<etype>]-> <dst-uuid>
//! (N edges shown, limit=25)
//! ```
//!
//! # Examples
//!
//! ```text
//! mnem traverse <node-uuid>
//! mnem traverse <node-uuid> --edge-label knows --edge-label authored
//! mnem traverse <node-uuid> --limit 10 --json
//! ```

use super::*;

#[derive(clap::Args, Debug)]
#[command(after_long_help = "\
Examples:
  mnem traverse <node-uuid>                          # all outgoing edges (limit 25)
  mnem traverse <node-uuid> --edge-label knows       # only 'knows' edges
  mnem traverse <node-uuid> -e knows -e authored     # multiple edge-type filters
  mnem traverse <node-uuid> --limit 10               # cap at 10 results
  mnem traverse <node-uuid> --json                   # JSON output
")]
pub(crate) struct Args {
    /// UUID of the node to start traversal from.
    pub node: String,

    /// Edge-type label to follow. Repeatable; if omitted all outgoing
    /// edge types are listed.
    #[arg(long = "edge-label", short = 'e')]
    pub edge_labels: Vec<String>,

    /// Maximum number of edges to show (default 25, max 200).
    #[arg(long, default_value = "25")]
    pub limit: usize,

    /// Emit JSON instead of human-readable output.
    #[arg(long)]
    pub json: bool,
}

pub(crate) fn run(override_path: Option<&Path>, args: Args) -> Result<()> {
    let (_dir, r, _bs, _ohs) = repo::open_all(override_path)?;

    // Parse the node UUID - give a clear error if it is not a valid UUID.
    let node_id = NodeId::parse_uuid(&args.node)
        .with_context(|| format!("invalid node UUID: {}", args.node))?;

    // Look up the node itself.
    let Some(node) = r.lookup_node(&node_id).context("looking up node")? else {
        bail!("no node with id={}", args.node);
    };

    // 0 = "no cap" (show all). Non-zero values are clamped to [1, 200].
    let limit = if args.limit == 0 {
        usize::MAX
    } else {
        args.limit.clamp(1, 200)
    };

    // Build the optional edge-type filter slice; discard empty strings.
    let filter_strs: Vec<&str> = args
        .edge_labels
        .iter()
        .map(String::as_str)
        .filter(|s| !s.is_empty())
        .collect();
    let etype_filter: Option<&[&str]> = if filter_strs.is_empty() {
        None
    } else {
        Some(&filter_strs)
    };

    // Load outgoing edges from the adjacency index, then apply limit.
    let all_edges = r
        .outgoing_edges(&node_id, etype_filter)
        .context("walking outgoing-adjacency index")?;
    let edges: Vec<_> = all_edges.into_iter().take(limit).collect();

    if args.json {
        // JSON output: {"node":{"id":"...","ntype":"..."},"edges":[{"etype":"...","dst":"..."},...]}
        let edge_arr: Vec<serde_json::Value> = edges
            .iter()
            .map(|e| {
                serde_json::json!({
                    "etype": e.etype,
                    "dst": e.dst.to_uuid_string(),
                })
            })
            .collect();
        let out = serde_json::json!({
            "node": {
                "id": node.id.to_uuid_string(),
                "ntype": node.ntype,
            },
            "edges": edge_arr,
        });
        println!("{}", serde_json::to_string(&out)?);
    } else {
        // Human-readable output.
        println!("node {} ({})", node.id.to_uuid_string(), node.ntype);
        if edges.is_empty() {
            println!("<no outgoing edges>");
        } else {
            for e in &edges {
                println!("-[{}]-> {}", e.etype, e.dst.to_uuid_string());
            }
            let noun = if edges.len() == 1 { "edge" } else { "edges" };
            println!("({} {} shown, limit={})", edges.len(), noun, limit);
        }
    }

    Ok(())
}
