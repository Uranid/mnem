use super::*;

use crate::{config, global, repo};

/// `mnem global` subcommand - read and write the global anchor graph at
/// `~/.mnemglobal/.mnem/` directly.
#[derive(clap::Subcommand, Debug)]
pub(crate) enum GlobalCmd {
    /// Search the global graph (~/.mnemglobal/.mnem/) only.
    /// Results are ranked by score.
    #[command(after_long_help = "\
Examples:
  mnem global retrieve \"Alice in Berlin\"
  mnem global retrieve \"climbing\" -n 5
  mnem global retrieve \"project deadline\" --no-vector
")]
    Retrieve(GlobalRetrieveArgs),

    /// Add a node or edge directly to the global graph (~/.mnemglobal/.mnem/).
    /// The printed node UUID can be used as `--prop _global_anchor=<uuid>` when
    /// adding the same entity to a local repo, linking the two graphs.
    ///
    /// Examples:
    ///   mnem global add node -s \"Alice works at Anthropic\" --label Entity:Person --prop name=Alice
    ///   # -> prints: node <uuid> committed
    ///   mnem -R ~/notes add node --label Entity:Person --prop name=Alice --prop _global_anchor=<uuid>
    #[command(subcommand)]
    Add(super::add::AddCmd),

    /// Ingest external source files (Markdown / text / PDF / chat JSON)
    /// into the global graph (~/.mnemglobal/.mnem/) as a Doc + Chunk + Entity subgraph.
    ///
    /// Examples:
    ///   mnem global ingest notes.md
    ///   mnem global ingest --chunker recursive --max-tokens 1024 book.pdf
    ///   mnem global ingest --recursive docs/
    Ingest(super::ingest::Args),
}

#[derive(clap::Args, Debug)]
pub(crate) struct GlobalRetrieveArgs {
    /// Query text. Embedded with the configured provider and used for
    /// semantic search on the global graph.
    #[arg(value_name = "QUERY")]
    pub query: Option<String>,
    /// Max results to return. Default: 10.
    #[arg(long, short = 'n')]
    pub limit: Option<usize>,
    /// Skip semantic (vector) search. Useful when no embedder is configured.
    #[arg(long)]
    pub no_vector: bool,
}

pub(crate) fn run(_override: Option<&Path>, cmd: GlobalCmd) -> Result<()> {
    let global_dir = global::default_dir();
    match cmd {
        GlobalCmd::Retrieve(args) => cmd_retrieve(&global_dir, args),
        GlobalCmd::Add(add_cmd) => {
            if !global_dir.join(repo::MNEM_DIR).is_dir() {
                bail!(
                    "Global graph not initialised at {}.\n\
                     hint: run `mnem integrate` to create it.",
                    global_dir.display()
                );
            }
            super::add::run(Some(&global_dir), add_cmd)
        }
        GlobalCmd::Ingest(ingest_args) => {
            if !global_dir.join(repo::MNEM_DIR).is_dir() {
                bail!(
                    "Global graph not initialised at {}.\n\
                     hint: run `mnem integrate` to create it.",
                    global_dir.display()
                );
            }
            super::ingest::run(Some(&global_dir), ingest_args)
        }
    }
}

struct Hit {
    score: f32,
    tokens: u32,
    node: mnem_core::objects::Node,
}

fn cmd_retrieve(global_dir: &Path, args: GlobalRetrieveArgs) -> Result<()> {
    let query = args.query.as_deref().unwrap_or("").trim().to_string();
    if query.is_empty() {
        bail!("A query text is required.\nUsage: mnem global retrieve \"your query\"");
    }
    let limit = args.limit.unwrap_or(10);

    let global_mnem = global_dir.join(repo::MNEM_DIR);
    if !global_mnem.is_dir() {
        bail!(
            "Global graph not initialised at {}.\n\
             hint: run `mnem integrate` to create it.",
            global_dir.display()
        );
    }

    // Embed using the global graph's embedder config.
    let opt_vec: Option<(String, Vec<f32>)> = if args.no_vector {
        None
    } else {
        let cfg = config::load(&global_mnem).ok();
        let pc = cfg.as_ref().and_then(config::resolve_embedder);
        if let Some(pc) = pc {
            match mnem_embed_providers::open(&pc) {
                Ok(embedder) => match embedder.embed(&query) {
                    Ok(v) => Some((embedder.model().to_string(), v)),
                    Err(e) => {
                        eprintln!("{}", format_embed_failure(&e, &pc, "query embedding"));
                        None
                    }
                },
                Err(e) => {
                    eprintln!("{}", format_embed_failure(&e, &pc, "query embedding"));
                    None
                }
            }
        } else {
            None
        }
    };

    let r = repo::open_repo(Some(global_dir))?;
    let mut ret = r.retrieve().limit(limit).query_text(query);
    if let Some((model, vec)) = &opt_vec {
        ret = ret.vector(model.clone(), vec.clone());
    }

    let result = match ret.execute() {
        Ok(res) => res,
        Err(e) => {
            let msg = format!("{e:#}");
            if msg.contains("no filters or rankers configured") {
                println!("No results found (global graph has no index configured).");
                return Ok(());
            }
            return Err(e.into());
        }
    };

    let mut hits: Vec<Hit> = result
        .items
        .into_iter()
        .map(|item| Hit {
            score: item.score,
            tokens: item.tokens,
            node: item.node,
        })
        .collect();

    if hits.is_empty() {
        println!("No results found in the global graph.");
        return Ok(());
    }

    hits.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    hits.truncate(limit);

    println!("Found {} result(s) in the global graph:\n", hits.len());
    for (i, hit) in hits.iter().enumerate() {
        println!(
            "---\n[{i}] score={:.4} tokens={} id={} {}",
            hit.score,
            hit.tokens,
            hit.node.id.to_uuid_string(),
            hit.node.ntype,
        );
        if let Some(s) = &hit.node.summary {
            println!("  {}", s.lines().next().unwrap_or(""));
        }
        if !hit.node.props.is_empty() {
            let preview: Vec<String> = hit
                .node
                .props
                .iter()
                .take(3)
                .map(|(k, v)| format!("{k}={}", ipld_preview(v)))
                .collect();
            println!("  props: {}", preview.join(", "));
        }
    }
    Ok(())
}
