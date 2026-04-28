use super::*;

fn resolve_ref_or_cid(r: &ReadonlyRepo, from: &str) -> Result<mnem_core::id::Cid> {
    if from == "HEAD" {
        return r
            .view()
            .heads
            .first()
            .cloned()
            .ok_or_else(|| anyhow!("repository has no commits yet; nothing to export"));
    }
    if let Some(target) = r.view().refs.get(from) {
        return match target {
            RefTarget::Normal { target } => Ok(target.clone()),
            RefTarget::Conflicted { adds, removes } => Err(anyhow!(
                "ref `{from}` is conflicted (adds={}, removes={}); resolve it first",
                adds.len(),
                removes.len()
            )),
        };
    }
    mnem_core::id::Cid::parse_str(from)
        .with_context(|| format!("`{from}` is neither a known ref nor a valid CID"))
}

use std::fs::File;
use std::io::{BufWriter, Write};

/// `mnem export` arguments.
#[derive(clap::Args, Debug)]
#[command(after_long_help = "\
Examples:
  mnem export notes.car                      # export HEAD to file
  mnem export --from refs/heads/main out.car # specific ref
  mnem export --from <cid> backup.car        # specific commit CID
  mnem export - | ssh server 'mnem import -' # pipe over SSH
")]
pub(crate) struct Args {
    /// Output path for the CAR archive. Use `-` to write to stdout.
    pub path: String,
    /// Ref name or CID to treat as the export root. Defaults to
    /// `HEAD` (the first head commit of the current view).
    #[arg(long, default_value = "HEAD")]
    pub from: String,
}

pub(crate) fn run(override_path: Option<&Path>, args: Args) -> Result<()> {
    let (_dir, r, bs, _ohs) = repo::open_all(override_path)?;
    let root = resolve_ref_or_cid(&r, &args.from)?;

    // audit-2026-04-25 P1-5: rewrite git-bash-style `/c/...` paths to
    // `c:/...` on Windows so users running mnem from MSYS2 / Git Bash
    // do not see "system cannot find the path" errors.
    let normalized = super::normalize_cli_path(&args.path);

    let stats = if normalized == "-" {
        let stdout = std::io::stdout();
        let mut lock = stdout.lock();
        mnem_transport::export(&*bs, &root, &mut lock).context("writing CAR to stdout")?
    } else {
        let path = Path::new(&normalized);
        let file = File::create(path).with_context(|| format!("creating {}", path.display()))?;
        let mut w = BufWriter::new(file);
        let stats = mnem_transport::export(&*bs, &root, &mut w)
            .with_context(|| format!("writing CAR to {}", path.display()))?;
        // Flushing the BufWriter surfaces any deferred I/O errors
        // that `write_all` swallowed into the buffer.
        w.flush()
            .with_context(|| format!("flushing {}", path.display()))?;
        stats
    };

    println!(
        "exported {} blocks, {} bytes to {}",
        stats.blocks, stats.bytes, normalized
    );
    Ok(())
}
