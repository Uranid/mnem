//! `mnem log` - walk the op-log backwards from the current head.
//!
//! Three output formats:
//!
//! - default: multi-line per op (git log default). Human readable.
//! - `--oneline`: `<short-cid> <message>` one line per op, matching
//!   `git log --oneline`.
//! - `--format=json`: JSON Lines, one compact object per op. Each
//!   line is a `{ "cid", "time", "author", "description", "parents"
//!   }` record. Stable across releases ; scripts can
//!   depend on it.

use std::io::{self, Write};

use serde::Serialize;

use super::*;

/// JSON-Lines record shape. Explicit struct so adding a field later
/// stays backward-compatible with existing consumers (`mnem log
/// --format=json | jq .author` keeps working).
#[derive(Serialize)]
struct LogRecord<'a> {
    cid: String,
    time: u64,
    author: &'a str,
    description: &'a str,
    parents: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    agent_id: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    task_id: Option<&'a str>,
}

/// Output format for `mnem log`. Defaults to the human-readable
/// multi-line form.
#[derive(clap::ValueEnum, Clone, Debug)]
pub(crate) enum Format {
    /// Multi-line per op (default).
    Human,
    /// JSON Lines, one record per op. Stable wire contract.
    Json,
}

#[derive(clap::Args, Debug)]
#[command(after_long_help = "\
Examples:
  mnem log                       # last 20 ops (default, human form)
  mnem log -n 5                  # last 5 ops
  mnem log --oneline             # short-cid + message per line
  mnem log --format=json | jq .  # JSON Lines, pipe through jq
  mnem log --format=json -n 100  # agent-facing op stream
")]
pub(crate) struct Args {
    /// Maximum number of operations to print.
    #[arg(long, short = 'n', default_value_t = 20)]
    pub limit: usize,
    /// Short-form output, one line per op: `<short-cid> <description>`.
    /// Conflicts with `--format=json`; `--oneline` wins.
    #[arg(long)]
    pub oneline: bool,
    /// Output format. Defaults to the human-readable multi-line
    /// shape. `json` emits JSON Lines.
    #[arg(long, value_enum, default_value = "human")]
    pub format: Format,
}

pub(crate) fn run(override_path: Option<&Path>, args: Args) -> Result<()> {
    let (_dir, r, bs, _ohs) = repo::open_all(override_path)?;

    let stdout = io::stdout();
    let mut w = stdout.lock();

    let mut cur = r.op_id().clone();
    for i in 0..args.limit {
        let bytes = bs
            .get(&cur)?
            .ok_or_else(|| anyhow!("op {cur} missing from store"))?;
        let op: Operation = from_canonical_bytes(&bytes)?;

        if args.oneline {
            // Git's short-cid is 7 hex chars; mnem CIDs start with a
            // multibase prefix, so we clip after the prefix for a
            // similar visual effect. Falls back to the full CID if
            // the render is shorter than expected.
            let full = cur.to_string();
            let short = short_cid(&full);
            writeln!(w, "{short} {}", op.description)?;
        } else {
            match args.format {
                Format::Json => write_json_record(&mut w, &cur, &op)?,
                Format::Human => write_human_record(&mut w, &cur, &op)?,
            }
        }

        match op.parents.first() {
            Some(p) => cur = p.clone(),
            None => {
                // Already printed this op; don't print a stale "break"
                // marker. The underscore keeps clippy quiet about `i`.
                let _ = i;
                break;
            }
        }
    }
    Ok(())
}

fn write_human_record(w: &mut impl Write, cid: &mnem_core::id::Cid, op: &Operation) -> Result<()> {
    writeln!(w, "op {cid}")?;
    writeln!(w, "   time    {}us", op.time)?;
    if !op.author.is_empty() {
        writeln!(w, "   author  {}", op.author)?;
    }
    if let Some(agent) = &op.agent_id {
        writeln!(w, "   agent   {agent}")?;
    }
    if let Some(task) = &op.task_id {
        writeln!(w, "   task    {task}")?;
    }
    writeln!(w, "   message {}", op.description)?;
    writeln!(w)?;
    Ok(())
}

fn write_json_record(w: &mut impl Write, cid: &mnem_core::id::Cid, op: &Operation) -> Result<()> {
    let record = LogRecord {
        cid: cid.to_string(),
        time: op.time,
        author: &op.author,
        description: &op.description,
        parents: op.parents.iter().map(ToString::to_string).collect(),
        agent_id: op.agent_id.as_deref(),
        task_id: op.task_id.as_deref(),
    };
    let line = serde_json::to_string(&record).context("serialising log record")?;
    writeln!(w, "{line}")?;
    Ok(())
}

/// Produce a short-hex prefix of a CID for `--oneline` output. mnem
/// CIDs start with a multibase prefix; trimming the first 2 bytes and
/// taking the next 8 gives a compact, still-unique-in-practice
/// rendering for typical commit counts.
fn short_cid(full: &str) -> String {
    if full.len() <= 10 {
        full.to_string()
    } else {
        // Skip multibase prefix byte (`b` etc) + typical codec byte.
        full.chars().skip(2).take(8).collect()
    }
}
