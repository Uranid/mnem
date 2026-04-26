//! `mnem fetch [<remote>]` - download new blocks from a remote and
//! update local tracking refs under `refs/remotes/<remote>/*`.
//!
//! Wire verb. Behaviour:
//!
//! 1. Load `[remote.<name>]` from `.mnem/config.toml` (default
//!    `origin`). Fail actionably if missing.
//! 2. Build an [`HttpRemoteClient`] with a [`SecretToken`] resolved
//!    from env: `MNEM_REMOTE_<UPPER>_TOKEN`, fallback to
//!    `MNEM_HTTP_PUSH_TOKEN`.
//! 3. `GET /remote/v1/refs`, diff against local tracking refs.
//! 4. For each server ref whose target we lack, call
//!    `POST /remote/v1/fetch-blocks` with a `BloomHaveSet` of the
//!    local reachability; stream the CAR back through
//!    [`mnem_transport::import::import_with_limit`].
//! 5. Update `refs/remotes/<remote>/<branch>` on the View via
//!    `ReadonlyRepo::update_ref` (one Operation per ref advance).
//! 6. Print Git-style `From <url>\n  <old>..<new> <branch> -> <remote>/<branch>`
//!    summary, one line per advanced ref.

use std::io::Cursor;

use mnem_core::id::Cid;
use mnem_core::objects::RefTarget;
use mnem_transport::build_have_set;
use mnem_transport::import::import_with_limit;
use mnem_transport::remote::parse_config;
use mnem_transport::secret_token::SecretToken;
use mnem_transport::{HttpRemoteClient, RemoteClient};

use super::*;

#[derive(clap::Args, Debug)]
#[command(after_long_help = "\
Examples:
  mnem fetch                          # fetch from `origin`
  mnem fetch origin                   # explicit remote name
  MNEM_REMOTE_ORIGIN_TOKEN=... mnem fetch origin
")]
pub(crate) struct Args {
    /// Remote name (matching `[remote.<name>]` in
    /// `.mnem/config.toml`). Defaults to `origin`.
    pub remote: Option<String>,
}

pub(crate) fn run(override_path: Option<&Path>, args: Args) -> Result<()> {
    let remote_name = args.remote.as_deref().unwrap_or("origin").to_string();
    let (data_dir, repo, bs, _ohs) = repo::open_all(override_path)?;

    // Load the remote's TOML config. Missing config.toml means "no
    // remotes configured"; bubble that up with the same actionable
    // hint as a missing-named-remote.
    let cfg_path = data_dir.join(config::CONFIG_FILE);
    let section = if cfg_path.exists() {
        let cfg_text = std::fs::read_to_string(&cfg_path)
            .with_context(|| format!("reading {}", cfg_path.display()))?;
        parse_config(&cfg_text).with_context(|| format!("parsing {}", cfg_path.display()))?
    } else {
        mnem_transport::RemoteSection::default()
    };
    let file = section.remote.get(&remote_name).ok_or_else(|| {
        anyhow!(
            "no remote `{remote_name}` configured; run `mnem remote add {remote_name} <url>` first"
        )
    })?;

    // Resolve bearer token (optional for read-side; some servers
    // require it on `fetch-blocks` too).
    let token = resolve_token(&remote_name, file.token_env.as_deref());

    // Build runtime RemoteConfig + client.
    let mut cfg = mnem_transport::RemoteConfig::new(remote_name.clone(), file.url.clone());
    if let Some(t) = token {
        cfg = cfg.with_token(t);
    }
    let client = HttpRemoteClient::new(cfg);

    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .context("building tokio runtime")?;

    rt.block_on(async {
        // 1. list_refs
        let refs_resp = client
            .list_refs()
            .await
            .with_context(|| format!("list_refs against {}", file.url))?;

        println!("From {}", file.url);
        if refs_resp.refs.is_empty() {
            println!("  (no refs on remote)");
            return Ok::<(), anyhow::Error>(());
        }

        // 2. For each server ref, compare with local tracking ref.
        for (ref_name, want_cid) in &refs_resp.refs {
            let tracking_key = format!("refs/remotes/{remote_name}/{ref_name}");
            let local_target = repo.view().refs.get(&tracking_key).cloned();
            let local_cid = match &local_target {
                Some(RefTarget::Normal { target }) => Some(target.clone()),
                _ => None,
            };
            if local_cid.as_ref() == Some(want_cid) {
                continue;
            }

            // 3. Fetch blocks + import. `have_set` summarises what we
            // already hold so the server can prune its CAR; empty is
            // safe (server ships the full reachability).
            let have_set = match repo.view().heads.first() {
                Some(root) => build_have_set(&*bs, root)
                    .unwrap_or_else(|_| mnem_transport::BloomHaveSet::new(1)),
                None => mnem_transport::BloomHaveSet::new(1),
            };
            let car = client
                .fetch_blocks(vec![want_cid.clone()], have_set)
                .await
                .with_context(|| format!("fetch_blocks for {ref_name}"))?;
            let mut reader = Cursor::new(car.as_ref());
            import_with_limit(
                &mut reader,
                &*bs,
                mnem_transport::import::DEFAULT_MAX_IMPORT_BYTES,
            )
            .with_context(|| format!("import CAR for {ref_name}"))?;

            // 4. Update tracking ref.
            let new_target = RefTarget::normal(want_cid.clone());
            let cfg_local = config::load(&data_dir)?;
            repo.update_ref(
                &tracking_key,
                local_target.as_ref(),
                Some(new_target),
                &config::author_string(&cfg_local),
            )
            .with_context(|| format!("update_ref {tracking_key}"))?;

            let old_short = local_cid
                .as_ref()
                .map_or_else(|| "<none>".to_string(), short_cid);
            let new_short = short_cid(want_cid);
            println!("  {old_short}..{new_short} {ref_name} -> {remote_name}/{ref_name}");
        }
        Ok(())
    })
}

/// Bearer token priority: `MNEM_REMOTE_<UPPER_NAME>_TOKEN`, then the
/// explicit `token_env` from config, then `MNEM_HTTP_PUSH_TOKEN`.
fn resolve_token(remote_name: &str, token_env_hint: Option<&str>) -> Option<SecretToken> {
    let upper = remote_name.to_ascii_uppercase();
    let primary = format!("MNEM_REMOTE_{upper}_TOKEN");
    if let Some(t) = SecretToken::from_env(&primary) {
        return Some(t);
    }
    if let Some(var) = token_env_hint
        && let Some(t) = SecretToken::from_env(var)
    {
        return Some(t);
    }
    SecretToken::from_env("MNEM_HTTP_PUSH_TOKEN")
}

fn short_cid(c: &Cid) -> String {
    let s = c.to_string();
    let take = s.len().min(12);
    s[..take].to_string()
}
