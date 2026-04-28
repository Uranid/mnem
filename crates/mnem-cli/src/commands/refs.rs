// Remote-v0 insertion point: the remote-transport roadmap (PR 1)
// landed `mnem export` / `mnem import` in their own submodules. Still
// pending: `mnem remote add/list/remove/show` (PR 2), `mnem clone
// <url>`, `mnem fetch [<remote>]`, `mnem pull [<remote>] [<branch>]`,
// `mnem push [<remote>] [<branch>]` (PR 3). Extend the subcommand
// below with `--remote` so callers can list and inspect
// `refs/remotes/<name>/*` entries without a dedicated `ref
// remote-list` verb. See
// `docs/ROADMAP.md#remote-v0-work-items-tracked-inline-in-src`
// item 2 and ().

use super::*;

#[derive(clap::Subcommand, Debug)]
pub(crate) enum RefCmd {
    /// List every ref in the current view.
    List,
    /// Set `name` to point at a target CID (normal ref).
    Set { name: String, target: String },
    /// Delete a ref.
    Delete { name: String },
}

pub(crate) fn run(override_path: Option<&Path>, cmd: RefCmd) -> Result<()> {
    let data_dir = repo::locate_data_dir(override_path)?;
    let cfg = config::load(&data_dir)?;
    let r = repo::open_repo(Some(data_dir.as_path()))?;
    match cmd {
        RefCmd::List => {
            let refs = &r.view().refs;
            if refs.is_empty() {
                println!("<no refs>");
                return Ok(());
            }
            for (name, target) in refs {
                let summary = match target {
                    RefTarget::Normal { target } => format!("-> {target}"),
                    RefTarget::Conflicted { adds, removes } => {
                        format!("conflicted(+{} -{})", adds.len(), removes.len())
                    }
                };
                println!("{name}  {summary}");
            }
        }
        RefCmd::Set { name, target } => {
            let cid = mnem_core::id::Cid::parse_str(&target).context("parsing target CID")?;
            let new_r = r.update_ref(
                &name,
                None,
                Some(RefTarget::normal(cid)),
                &config::author_string(&cfg),
            )?;
            println!("set {name} -> {}", new_r.op_id());
        }
        RefCmd::Delete { name } => {
            let new_r = r.update_ref(
                &name,
                r.view().refs.get(&name),
                None,
                &config::author_string(&cfg),
            )?;
            println!("deleted {name} -> {}", new_r.op_id());
        }
    }
    Ok(())
}
