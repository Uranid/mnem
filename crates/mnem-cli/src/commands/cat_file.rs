//! `mnem cat-file` - raw block bytes to stdout (binary-safe) or a
//! decoded JSON preview.
//!
//! Parallels `git cat-file`. Two output modes:
//!
//! - default: the raw DAG-CBOR bytes for the block, written directly
//!   to stdout (`io::stdout().write_all`). Binary-safe so you can
//!   pipe into `xxd`, `cbor-diag`, or your favourite hex viewer.
//! - `--json`: decode the block's canonical CBOR to IPLD and re-emit
//!   as DAG-JSON to stdout. This mirrors 's "dagjson is a
//!   debug format, never hashed" rule.
//!
//! The subcommand prints an actionable error if the CID is not
//! present in the local blockstore.
//!
//! # Examples
//!
//! ```text
//! mnem cat-file <commit-cid> > commit.cbor
//! mnem cat-file <commit-cid> --json | jq .
//! mnem cat-file <op-cid> --json | jq '._kind'
//! ```

use std::io::Write;

use ipld_core::ipld::Ipld;
use mnem_core::codec::{from_canonical_bytes, to_json_bytes};

use super::*;

#[derive(clap::Args, Debug)]
#[command(after_long_help = "\
Examples:
  mnem cat-file <cid>                   # raw DAG-CBOR bytes
  mnem cat-file <cid> --json            # pretty-printed DAG-JSON
  mnem cat-file <commit-cid> --json | jq '._kind'
")]
pub(crate) struct Args {
    /// CID of the block to emit.
    pub cid: String,
    /// Decode the block and emit DAG-JSON instead of raw CBOR.
    #[arg(long)]
    pub json: bool,
}

pub(crate) fn run(override_path: Option<&Path>, args: Args) -> Result<()> {
    let data_dir = repo::locate_data_dir(override_path)?;
    let (bs, _) = repo::open_stores(&data_dir)?;

    let cid = mnem_core::id::Cid::parse_str(&args.cid)
        .map_err(|e| anyhow!("invalid CID `{}` ({e})", args.cid))?;
    let bytes = bs.get(&cid)?.ok_or_else(|| {
        anyhow!(
            "block {cid} not present in this blockstore\n\
             hint: the repo may have been cloned without this subtree, or the CID may \
             be from a different repo. See docs/RUNBOOK.md#6-mid-commit-crash-recovery \
             for the consistency-semantics reference."
        )
    })?;

    if args.json {
        // Round-trip through Ipld: dagcbor bytes -> Ipld -> dagjson.
        // A failure here means the block is NOT canonical DAG-CBOR
        // (corruption or a block from a future codec); surface the
        // error with context rather than emit garbage JSON.
        let ipld: Ipld = from_canonical_bytes(&bytes).context("decoding block as DAG-CBOR")?;
        let json = to_json_bytes(&ipld).context("re-encoding IPLD to DAG-JSON")?;
        // Write directly to stdout; the caller decides whether to
        // pipe through jq.
        std::io::stdout()
            .write_all(&json)
            .context("writing JSON to stdout")?;
        // Trailing newline so shell prompts don't fight a bare bracket.
        std::io::stdout()
            .write_all(b"\n")
            .context("writing newline")?;
    } else {
        // Raw path: the bytes go straight out, including any zero
        // bytes inside the CBOR. Never decode; never re-encode.
        std::io::stdout()
            .write_all(&bytes)
            .context("writing raw bytes to stdout")?;
    }
    Ok(())
}
