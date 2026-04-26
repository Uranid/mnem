//! `mnem remote` - manage `[remote.<name>]` entries in
//! `.mnem/config.toml`.
//!
//! Pure local config-file ops. No network I/O. The on-disk schema is
//! owned by [`mnem_transport::remote::RemoteConfigFile`] ;
//! this module reads, writes, and pretty-prints it.
//!
//! ## Security
//!
//! Bearer tokens NEVER land on disk. `mnem remote add --token-env
//! <VAR>` records only the name of the env var; the runtime injects
//! the actual token via `SecretToken` at fetch/push time.
//!
//! # Examples
//!
//! ```text
//! mnem remote add origin https://example.com/alice/notes
//! mnem remote list
//! mnem remote show origin
//! mnem remote remove origin
//! ```
//!
//! `file://` URLs are rejected here (audit-2026-04-25 P1-4): the
//! `fetch` / `pull` transport is HTTP-only, so a `file://` remote
//! produces a confusing builder error at fetch time. For one-shot
//! local CAR mirrors use `mnem clone file:///path/to/repo.car ./mirror`
//! directly; that path reads the CAR archive without going through
//! the transport.

use std::collections::BTreeMap;
use std::fs;
use std::io::Read;

use mnem_transport::remote::{RemoteConfigFile, RemoteSection, parse_config, serialize_config};

use super::*;

#[derive(clap::Subcommand, Debug)]
pub(crate) enum RemoteCmd {
    /// Add a new remote. Fails if `<name>` already exists.
    Add {
        /// Short name of the remote (`origin`, `backup`, ...). Used as
        /// the key in `[remote.<name>]`.
        name: String,
        /// Remote URL. Any URL scheme is accepted at the config layer;
        /// the transport driver decides what it supports.
        url: String,
        /// Name of an environment variable holding the bearer token
        /// for this remote. Optional; the token itself is never
        /// stored.
        #[arg(long)]
        token_env: Option<String>,
    },
    /// List every configured remote.
    List,
    /// Show one remote's fields in detail. Token is redacted if
    /// present in memory.
    Show {
        /// Remote name to display.
        name: String,
    },
    /// Delete a remote entry. Fails if `<name>` does not exist.
    Remove {
        /// Remote name to delete.
        name: String,
    },
}

pub(crate) fn run(override_path: Option<&Path>, cmd: RemoteCmd) -> Result<()> {
    let data_dir = repo::locate_data_dir(override_path)?;
    match cmd {
        RemoteCmd::Add {
            name,
            url,
            token_env,
        } => add_remote(&data_dir, &name, &url, token_env.as_deref()),
        RemoteCmd::List => list_remotes(&data_dir),
        RemoteCmd::Show { name } => show_remote(&data_dir, &name),
        RemoteCmd::Remove { name } => remove_remote(&data_dir, &name),
    }
}

/// Read `[remote.*]` sections from the config file, tolerating a
/// missing file (treated as an empty section map).
fn load_section(data_dir: &std::path::Path) -> Result<(RemoteSection, String)> {
    let path = data_dir.join(config::CONFIG_FILE);
    if !path.exists() {
        return Ok((RemoteSection::default(), String::new()));
    }
    let mut s = String::new();
    fs::File::open(&path)
        .with_context(|| format!("opening {}", path.display()))?
        .read_to_string(&mut s)
        .with_context(|| format!("reading {}", path.display()))?;
    let section = parse_config(&s).with_context(|| format!("parsing {}", path.display()))?;
    Ok((section, s))
}

/// Write the `[remote.*]` section back, preserving the rest of the
/// `config.toml` shape by re-reading it as a TOML `Value` and
/// substituting the `remote` table. Empty remote-map drops the
/// `[remote]` table wholesale so the file stays clean after the last
/// `mnem remote remove`.
fn save_section(data_dir: &std::path::Path, section: &RemoteSection) -> Result<()> {
    let path = data_dir.join(config::CONFIG_FILE);
    // Start from whatever is on disk so user.name, [embed], etc.
    // survive. Missing file = empty table.
    let mut root: toml::Value = if path.exists() {
        let text =
            fs::read_to_string(&path).with_context(|| format!("reading {}", path.display()))?;
        toml::from_str(&text).with_context(|| format!("parsing {}", path.display()))?
    } else {
        toml::Value::Table(toml::map::Map::new())
    };
    // Re-serialise the remote-only half so we drop into a `toml::
    // Value` with the right structure, then move the `remote` table
    // across.
    let remote_text = serialize_config(section).context("serialising remote section")?;
    let remote_root: toml::Value =
        toml::from_str(&remote_text).context("re-parsing remote section")?;

    let table = root
        .as_table_mut()
        .ok_or_else(|| anyhow!("config.toml root is not a table"))?;
    // Drop any existing remote table; write the new one iff the
    // section carries at least one remote. This mirrors the
    // retrieve-table collapse rule in config.rs: an empty section
    // doesn't leave a ghost header in the file.
    table.remove("remote");
    if !section.remote.is_empty()
        && let Some(new_remote) = remote_root.get("remote").cloned()
    {
        table.insert("remote".into(), new_remote);
    }

    let text = toml::to_string_pretty(&root).context("serialising config.toml")?;
    // Create the parent if needed (first-run without `mnem init`).
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).with_context(|| format!("creating {}", parent.display()))?;
    }
    fs::write(&path, text).with_context(|| format!("writing {}", path.display()))
}

fn add_remote(
    data_dir: &std::path::Path,
    name: &str,
    url: &str,
    token_env: Option<&str>,
) -> Result<()> {
    if name.is_empty() {
        bail!("remote name must not be empty");
    }
    // Reject strings that would corrupt the TOML section header.
    if name.contains('.') || name.contains('[') || name.contains(']') {
        bail!("remote name must not contain '.', '[' or ']'; got `{name}`");
    }
    // audit-2026-04-25 P1-4: `mnem fetch` / `pull` only speak HTTP, so
    // a `file://` remote fails opaquely with a builder error at fetch
    // time. Reject up-front and point at `mnem clone` (which DOES
    // accept `file://`).
    if url.starts_with("file://") || url.starts_with("file:/") {
        bail!(
            "file:// remotes are not supported by `mnem fetch` / `mnem pull`\n\
             hint: use `mnem clone {url} <dest>` for a one-shot local CAR mirror,\n\
             or host the CAR archive over HTTP and point the remote at that URL"
        );
    }
    let (mut section, _) = load_section(data_dir)?;
    if section.remote.contains_key(name) {
        bail!("remote `{name}` already exists; use `mnem remote remove {name}` first");
    }
    section.remote.insert(
        name.to_string(),
        RemoteConfigFile {
            url: url.to_string(),
            capabilities: None,
            token_env: token_env.map(str::to_string),
        },
    );
    save_section(data_dir, &section)?;
    println!("added remote {name} -> {url}");
    Ok(())
}

fn list_remotes(data_dir: &std::path::Path) -> Result<()> {
    let (section, _) = load_section(data_dir)?;
    if section.remote.is_empty() {
        println!("<no remotes>");
        return Ok(());
    }
    // Column-align for the common case. Names rarely exceed 16 chars.
    let max_name = section.remote.keys().map(String::len).max().unwrap_or(0);
    for (name, file) in &section.remote {
        println!("{name:<width$}  {}", file.url, width = max_name.max(6));
    }
    Ok(())
}

fn show_remote(data_dir: &std::path::Path, name: &str) -> Result<()> {
    let (section, _) = load_section(data_dir)?;
    let file = section
        .remote
        .get(name)
        .ok_or_else(|| anyhow!("remote `{name}` not found"))?;
    println!("name          {name}");
    println!("url           {}", file.url);
    match &file.capabilities {
        None => println!("capabilities  <all built-in>"),
        Some(list) => {
            println!("capabilities  ({})", list.len());
            for c in list {
                println!("  {c}");
            }
        }
    }
    match &file.token_env {
        None => println!("token_env     <none>"),
        Some(var) => {
            // Check whether the env var is actually set; don't reveal
            // its value. Useful feedback for the common "I set the
            // env var in a different shell" debug case.
            let present = std::env::var(var).is_ok();
            println!(
                "token_env     {var} ({})",
                if present { "present in env" } else { "NOT set" }
            );
        }
    }
    // Any future runtime-only fields (like a loaded SecretToken)
    // would print a redacted marker here; config.toml never carries
    // the plaintext.
    let _ = BTreeMap::<String, String>::new;
    Ok(())
}

fn remove_remote(data_dir: &std::path::Path, name: &str) -> Result<()> {
    let (mut section, _) = load_section(data_dir)?;
    if section.remote.remove(name).is_none() {
        bail!("remote `{name}` not found");
    }
    save_section(data_dir, &section)?;
    println!("removed remote {name}");
    Ok(())
}
