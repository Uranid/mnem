use super::*;

#[derive(clap::Args, Debug)]
#[command(after_long_help = "\
Examples:
  mnem query --where name=Alice
  mnem query --where kind=Person --with-outgoing knows -n 5
  mnem query --where city=Berlin | head -50
")]
pub(crate) struct Args {
    /// Property equality filter. Pass once: `--where name=Alice`.
    #[arg(long = "where")]
    pub where_: Option<String>,
    /// Include outgoing edges of these labels on each hit. Repeatable.
    #[arg(long = "with-outgoing")]
    pub with_outgoing: Vec<String>,
    /// Max hits to print.
    #[arg(long, short = 'n', default_value_t = 10)]
    pub limit: usize,
}

pub(crate) fn run(override_path: Option<&Path>, args: Args) -> Result<()> {
    let r = repo::open_repo(override_path)?;

    let mut q = Query::new(&r);
    if let Some(wh) = &args.where_ {
        let (k, v) = parse_prop(wh)?;
        if let Some(label) = node_label_from_where(&k, &v)? {
            q = q.label(label);
        } else {
            q = q.where_prop(k, PropPredicate::Eq(v));
        }
    }
    for w in &args.with_outgoing {
        q = q.with_outgoing(w.as_str());
    }
    q = q.limit(args.limit);
    let hits = q.execute()?;

    println!("{} hit(s)", hits.len());
    for (i, h) in hits.iter().enumerate() {
        println!("[{i}] {} id={}", h.node.ntype, h.node.id.to_uuid_string());
        for (k, v) in &h.node.props {
            println!("      {k}: {}", ipld_preview(v));
        }
        for e in &h.edges {
            println!("      -[{}]-> {}", e.etype, e.dst.to_uuid_string());
        }
    }
    Ok(())
}
