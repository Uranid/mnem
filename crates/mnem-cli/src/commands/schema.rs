//! `mnem schema [--json]` — print index metadata (node labels, indexed
//! props, adjacency indexes) from the current commit's IndexSet.

use std::path::Path;

use anyhow::Result;
use serde::Serialize;
use serde_json::json;

use super::*;

// ----------------------------------------------------------------
// Clap args
// ----------------------------------------------------------------

#[derive(clap::Args, Debug)]
#[command(after_long_help = "\
Examples:
  mnem schema                  # human-readable label + prop index summary
  mnem schema --json           # machine-readable JSON
  mnem -R ~/notes schema       # explicit repo path
")]
pub(crate) struct Args {
    /// Emit JSON instead of human-readable output.
    #[arg(long)]
    pub json: bool,
}

// ----------------------------------------------------------------
// JSON output shape
// ----------------------------------------------------------------

#[derive(Serialize)]
struct LabelEntry {
    label: String,
    indexed_props: Vec<String>,
    has_outgoing_adj: bool,
    has_incoming_adj: bool,
}

#[derive(Serialize)]
struct SchemaOutput {
    labels: Vec<LabelEntry>,
}

// ----------------------------------------------------------------
// run
// ----------------------------------------------------------------

pub(crate) fn run(override_path: Option<&Path>, args: Args) -> Result<()> {
    let (_dir, repo, bs, _ohs) = crate::repo::open_all(override_path)?;
    let commit = repo.head_commit();
    let idx = load_index_set(&bs, commit)?;

    let Some(set) = idx else {
        if args.json {
            println!("{}", json!({"labels": []}));
        } else {
            println!("schema: <no IndexSet on current commit>");
        }
        return Ok(());
    };

    let has_outgoing = set.outgoing.is_some();
    let has_incoming = set.incoming.is_some();

    if args.json {
        let labels: Vec<LabelEntry> = set
            .nodes_by_label
            .keys()
            .map(|label| {
                let indexed_props: Vec<String> = set
                    .nodes_by_prop
                    .get(label)
                    .map(|m| m.keys().cloned().collect())
                    .unwrap_or_default();
                LabelEntry {
                    label: label.clone(),
                    indexed_props,
                    has_outgoing_adj: has_outgoing,
                    has_incoming_adj: has_incoming,
                }
            })
            .collect();
        println!(
            "{}",
            serde_json::to_string_pretty(&SchemaOutput { labels })
                .context("serialising schema JSON")?
        );
    } else {
        let label_count = set.nodes_by_label.len();
        println!("node labels ({label_count}):");
        if set.nodes_by_label.is_empty() {
            println!("  <none>");
        } else {
            for label in set.nodes_by_label.keys() {
                let props: Vec<String> = set
                    .nodes_by_prop
                    .get(label)
                    .map(|m| m.keys().cloned().collect())
                    .unwrap_or_default();
                let props_str = if props.is_empty() {
                    "none".to_string()
                } else {
                    props.join(", ")
                };
                println!(
                    "  {label}  [props: {props_str}]  [outgoing-adj: {}]  [incoming-adj: {}]",
                    if has_outgoing { "yes" } else { "no" },
                    if has_incoming { "yes" } else { "no" },
                );
            }
        }
    }

    Ok(())
}
