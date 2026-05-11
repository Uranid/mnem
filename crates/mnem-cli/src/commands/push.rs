//! `mnem push [<remote>] [<branch>]` - upload new blocks to a remote
//! and atomically advance the named ref there.
//!
//! Wire verb. Behaviour:
//!
//! 1. Load `[remote.<name>]` from `.mnem/config.toml` (default
//!    `origin`).
//! 2. Resolve bearer token via `MNEM_REMOTE_<UPPER>_TOKEN` ->
//!    `token_env` hint -> `MNEM_HTTP_PUSH_TOKEN`. If still `None`,
//!    fail with the auth hint.
//! 3. Read local HEAD commit CID. Refuse if the repo is empty.
//! 4. `GET /remote/v1/refs` -> read remote's current tip for the
//!    target branch. If it already matches local HEAD, skip.
//! 5. Export the subtree reachable from HEAD as a CAR via
//!    `mnem_transport::export`.
//! 6. `POST /remote/v1/push-blocks` with the CAR, bearer-auth'd.
//! 7. `POST /remote/v1/advance-head` with
//!    `{old: <remote_tip>, new: <local_head>, ref: <branch>}`. On
//!    409 surface the "rebase required" hint.

use mnem_core::id::Cid;
use mnem_core::objects::RefTarget;
use mnem_transport::HttpRemoteClient;
use mnem_transport::client::RemoteClient;
use mnem_transport::error::ClientError;
use mnem_transport::export::export;
use mnem_transport::remote::parse_config;
use mnem_transport::secret_token::SecretToken;

use super::*;

#[derive(clap::Args, Debug)]
#[command(after_long_help = "\
Examples:
  mnem push                           # push HEAD to origin/main
  mnem push origin main
  MNEM_REMOTE_ORIGIN_TOKEN=... mnem push origin main
")]
pub(crate) struct Args {
    /// Remote name. Defaults to `origin`.
    pub remote: Option<String>,
    /// Branch name on the remote. Defaults to `main`.
    pub branch: Option<String>,
}

pub(crate) fn run(override_path: Option<&Path>, args: Args) -> Result<()> {
    let remote_name = args.remote.as_deref().unwrap_or("origin").to_string();
    let branch = args.branch.as_deref().unwrap_or("main").to_string();
    let (data_dir, repo, bs, _ohs) = repo::open_all(override_path)?;

    // Load remote config; missing file == no remotes configured.
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

    // Resolve bearer.
    let token = resolve_token(&remote_name, file.token_env.as_deref()).ok_or_else(|| {
        let upper = remote_name.to_ascii_uppercase();
        anyhow!(
            "Authentication required. Set MNEM_REMOTE_{upper}_TOKEN env var \
             (or MNEM_HTTP_PUSH_TOKEN) to push to `{remote_name}`."
        )
    })?;

    let mut cfg = mnem_transport::RemoteConfig::new(remote_name.clone(), file.url.clone());
    cfg = cfg.with_token(token);
    let client = HttpRemoteClient::new(cfg);

    // Local head.
    let local_head = repo
        .view()
        .heads
        .first()
        .cloned()
        .ok_or_else(|| anyhow!("refusing to push: repository has no commits"))?;

    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .context("building tokio runtime")?;

    rt.block_on(async {
        // Remote refs snapshot.
        let refs_resp = client
            .list_refs()
            .await
            .with_context(|| format!("list_refs against {}", file.url))?;
        // Try branch-name key first (future multi-branch server mode),
        // then fall back to the top-level `head` field. The B3.1 server
        // only inserts the "HEAD" key into refs.refs, not branch names,
        // so without this fallback `remote_tip` is always None and
        // every push after the first would incorrectly send null old.
        let remote_tip: Option<Cid> = refs_resp
            .refs
            .get(&branch)
            .cloned()
            .or_else(|| refs_resp.head.clone());

        if remote_tip.as_ref() == Some(&local_head) {
            println!("Everything up-to-date");
            return Ok::<(), anyhow::Error>(());
        }

        // Build CAR body from local blockstore, rooted at local_head.
        let mut car: Vec<u8> = Vec::new();
        export(&*bs, &local_head, &mut car).context("export CAR from local blockstore")?;

        // push-blocks.
        let pushed = client
            .push_blocks(bytes::Bytes::from(car))
            .await
            .map_err(map_client_err_push)?;

        // advance-head CAS.
        // When `remote_tip` is None the remote has no head for this branch
        // (fresh remote / first push). Pass None so the server treats it as
        // an empty-head CAS rather than matching a non-existent CID.
        let cas = client
            .advance_head(remote_tip.clone(), local_head.clone(), branch.clone())
            .await;
        match cas {
            Ok(()) => {
                // Update local tracking ref so a subsequent fetch is a no-op.
                let tracking_key = format!("refs/remotes/{remote_name}/{branch}");
                let prev = repo.view().refs.get(&tracking_key).cloned();
                let cfg_local = config::load(&data_dir)?;
                if let Err(e) = repo.update_ref(
                    &tracking_key,
                    prev.as_ref(),
                    Some(RefTarget::normal(local_head.clone())),
                    &config::author_string(&cfg_local),
                ) {
                    eprintln!("warning: could not update tracking ref: {e}");
                }
                println!("pushed root `{}`", pushed.root);
                println!("To {}", file.url);
                let old_short = remote_tip
                    .as_ref()
                    .map_or_else(|| "<new>".to_string(), short_cid);
                println!(
                    "   {old_short}..{} {branch} -> {remote_name}/{branch}",
                    short_cid(&local_head),
                );
                Ok(())
            }
            Err(ClientError::CasMismatch { .. }) => Err(anyhow!(
                "Updates were rejected because tip of remote {branch} is ahead. \
                 Integrate remote changes (e.g. 'mnem pull') and try again."
            )),
            Err(ClientError::Auth(msg)) => {
                let upper = remote_name.to_ascii_uppercase();
                Err(anyhow!(
                    "Authentication required. Set MNEM_REMOTE_{upper}_TOKEN env var. ({msg})"
                ))
            }
            // Surface protocol-level rejections (e.g. non-main branch push)
            // directly so the server's explanation reaches the user without
            // a redundant "advance_head failed: protocol:" prefix.
            Err(ClientError::Protocol(msg)) => Err(anyhow!("{msg}")),
            Err(e) => Err(anyhow!("advance_head failed: {e}")),
        }
    })
}

fn map_client_err_push(e: ClientError) -> anyhow::Error {
    match e {
        ClientError::Auth(msg) => {
            anyhow!("Authentication required. Set MNEM_REMOTE_<NAME>_TOKEN env var. ({msg})")
        }
        other => anyhow!("push_blocks failed: {other}"),
    }
}

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
