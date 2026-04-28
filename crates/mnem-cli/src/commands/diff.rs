use super::*;

#[derive(clap::Args, Debug)]
#[command(after_long_help = "\
Examples:
  mnem diff <op-a-cid> <op-b-cid>
  # common flow: find two ops via `mnem log`, then diff:
  mnem log -n 2
  mnem diff <older-op> <newer-op>
")]
pub(crate) struct Args {
    pub op_a: String,
    pub op_b: String,
}

pub(crate) fn run(override_path: Option<&Path>, args: Args) -> Result<()> {
    // audit-2026-04-25 R2 (Stage E re-fix): `mnem diff` decodes each
    // side as an Operation, so we MUST resolve to op-CIDs. The
    // generic `resolve_commitish` returns commit-CIDs, which then
    // failed to decode as Operations with the V4 error
    // `decode: Msg("missing field 'view'")`. The dedicated op-CID
    // resolver accepts HEAD (-> r.op_id()) and raw op CIDs; named
    // refs intentionally are not supported here because they target
    // commits, not ops.
    let (_dir, r, bs, _ohs) = repo::open_all(override_path)?;

    let a_cid = super::resolve_op_commitish(&r, &args.op_a)
        .with_context(|| format!("resolving op_a `{}`", args.op_a))?;
    let b_cid = super::resolve_op_commitish(&r, &args.op_b)
        .with_context(|| format!("resolving op_b `{}`", args.op_b))?;

    let op_a: Operation =
        from_canonical_bytes(&bs.get(&a_cid)?.ok_or_else(|| anyhow!("op_a missing"))?)?;
    let op_b: Operation =
        from_canonical_bytes(&bs.get(&b_cid)?.ok_or_else(|| anyhow!("op_b missing"))?)?;
    let view_a: mnem_core::objects::View = from_canonical_bytes(
        &bs.get(&op_a.view)?
            .ok_or_else(|| anyhow!("view_a missing"))?,
    )?;
    let view_b: mnem_core::objects::View = from_canonical_bytes(
        &bs.get(&op_b.view)?
            .ok_or_else(|| anyhow!("view_b missing"))?,
    )?;

    println!("op_a {a_cid}");
    println!("op_b {b_cid}");
    println!();

    // Ref deltas.
    let mut added: Vec<&String> = Vec::new();
    let mut removed: Vec<&String> = Vec::new();
    let mut changed: Vec<&String> = Vec::new();
    for (name, target) in &view_b.refs {
        match view_a.refs.get(name) {
            None => added.push(name),
            Some(prev) if prev != target => changed.push(name),
            _ => {}
        }
    }
    for name in view_a.refs.keys() {
        if !view_b.refs.contains_key(name) {
            removed.push(name);
        }
    }
    println!(
        "ref deltas: +{} -{} ~{}",
        added.len(),
        removed.len(),
        changed.len()
    );
    for r in &added {
        println!("  +{r}");
    }
    for r in &removed {
        println!("  -{r}");
    }
    for r in &changed {
        println!("  ~{r}");
    }

    // Commit CIDs.
    let head_a = view_a.heads.first();
    let head_b = view_b.heads.first();
    println!();
    println!(
        "commit deltas: a={} -> b={}",
        head_a.map_or_else(|| "<none>".to_string(), ToString::to_string),
        head_b.map_or_else(|| "<none>".to_string(), ToString::to_string)
    );
    Ok(())
}
