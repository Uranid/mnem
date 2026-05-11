use super::*;

#[derive(clap::Args, Debug)]
#[command(after_long_help = "\
Examples:
  mnem get <uuid>                           # show node summary, ntype, props
  mnem get <uuid> --content                 # also print the full content body
  mnem global get <uuid>                    # look up in the global graph
  mnem -R ~/notes get <uuid>
")]
pub(crate) struct Args {
    /// UUID of the node to fetch.
    pub id: String,
    /// Also print the full content body (UTF-8; binary shown as hex preview).
    #[arg(long)]
    pub content: bool,
}

pub(crate) fn run(override_path: Option<&Path>, args: Args) -> Result<()> {
    let id = mnem_core::id::NodeId::parse_uuid(&args.id)
        .with_context(|| format!("invalid node UUID: {}", args.id))?;
    let (_, repo, _, _) = repo::open_all(override_path)?;
    let Some(node) = repo.lookup_node(&id)? else {
        bail!("no node found for id={}", args.id);
    };

    println!("node {}", node.id.to_uuid_string());
    println!("  ntype:     {}", node.ntype);

    if repo.is_tombstoned(&id) {
        println!("  tombstoned: true");
    }

    if let Some(summary) = &node.summary {
        println!("  summary:   {summary}");
    }
    if !node.props.is_empty() {
        println!("  props:");
        for (k, v) in &node.props {
            println!("    {k}: {}", ipld_preview(v));
        }
    }
    if let Some(bytes) = &node.content {
        println!("  content:   {} bytes", bytes.len());
        if args.content {
            match std::str::from_utf8(bytes) {
                Ok(s) => println!("{s}"),
                Err(_) => {
                    // print hex preview for binary content
                    let preview: String = bytes
                        .iter()
                        .take(256)
                        .map(|b| format!("{b:02x}"))
                        .collect::<Vec<_>>()
                        .join(" ");
                    println!(
                        "(binary) {preview}{}",
                        if bytes.len() > 256 { " ..." } else { "" }
                    );
                }
            }
        }
    }
    Ok(())
}
