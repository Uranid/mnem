use super::*;

/// Either a `set/get/unset/list` subcommand or the git-style legacy
/// positional form `mnem config <key> [value]`. Both work.
#[derive(clap::Args, Debug)]
#[command(after_long_help = "\
Examples:
  mnem config set user.name Alice
  mnem config set embed.provider ollama
  mnem config set embed.model    nomic-embed-text
  mnem config get embed.provider
  mnem config unset embed.provider
  mnem config list

  # Write to ~/.mnem/config.toml instead of the per-repo config:
  mnem config --global set user.name Alice
  mnem config -g set user.email alice@example.com
  mnem config -g list

Known keys:
  user.name, user.email, user.key, user.agent_id
  embed.provider    openai | ollama
  embed.model       provider-specific model name
  embed.api_key_env name of env var holding the API key (not the key)
  embed.base_url    override the provider default endpoint

API keys live in environment variables, never in config.toml.
`mnem config set embed.api_key sk-...` is refused.
")]
pub(crate) struct Args {
    #[command(subcommand)]
    pub verb: Option<Verb>,

    /// Legacy positional form: `mnem config <key> [value]`. Accepted
    /// when no subcommand is passed.
    #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
    pub legacy: Vec<String>,

    /// Unset the key (legacy positional form only).
    #[arg(long)]
    pub unset: bool,

    /// Read/write the user-global config (~/.mnem/config.toml) instead
    /// of the per-repo .mnem/config.toml. Mirrors `git config --global`.
    #[arg(long, short = 'g')]
    pub global: bool,
}

#[derive(clap::Subcommand, Debug)]
pub(crate) enum Verb {
    /// Set a key: `mnem config set user.name Alice`.
    Set { key: String, value: String },
    /// Print the effective value of a key, or exit 1 if unset.
    Get { key: String },
    /// Remove a key from the config.
    Unset { key: String },
    /// Print every set key and its value.
    List,
}

pub(crate) fn run(override_path: Option<&Path>, args: Args) -> Result<()> {
    // Resolve the config path and load the config, branching on --global.
    let (cfg_path, mut cfg) = if args.global {
        let p = config::global_path()
            .ok_or_else(|| anyhow::anyhow!("cannot determine home directory for global config"))?;
        let c = config::load_global()?;
        (p, c)
    } else {
        let data_dir = repo::locate_data_dir(override_path)?;
        let c = config::load(&data_dir)?;
        (config::path_of(&data_dir), c)
    };

    // Dispatch: subcommand wins over legacy positional.
    if let Some(v) = args.verb {
        return run_verb(v, &cfg_path, &mut cfg);
    }

    // Legacy: `mnem config <key> [value]` (+ optional --unset).
    match args.legacy.as_slice() {
        [] => {
            bail!("expected a subcommand (set/get/unset/list) or `mnem config <key> [value]`")
        }
        [key] if args.unset => {
            config::set_dotted(&mut cfg, key, None)?;
            config::save_to_path(&cfg_path, &cfg)?;
            println!("unset {key}");
        }
        [key] => match config::get_dotted(&cfg, key) {
            Some(v) => println!("{v}"),
            None => bail!("no value set for {key}"),
        },
        [key, value] => {
            config::set_dotted(&mut cfg, key, Some(value.clone()))?;
            config::save_to_path(&cfg_path, &cfg)?;
            println!("{key} = {value}");
        }
        _ => bail!("too many positional args; did you mean `mnem config set <key> <value>`?"),
    }
    Ok(())
}

fn run_verb(v: Verb, cfg_path: &Path, cfg: &mut config::Config) -> Result<()> {
    match v {
        Verb::Set { key, value } => {
            config::set_dotted(cfg, &key, Some(value.clone()))?;
            config::save_to_path(cfg_path, cfg)?;
            println!("{key} = {value}");
        }
        Verb::Get { key } => match config::get_dotted(cfg, &key) {
            Some(v) => println!("{v}"),
            None => bail!("no value set for {key}"),
        },
        Verb::Unset { key } => {
            config::set_dotted(cfg, &key, None)?;
            config::save_to_path(cfg_path, cfg)?;
            println!("unset {key}");
        }
        Verb::List => {
            let mut printed = 0usize;
            for k in config::KNOWN_KEYS {
                if let Some(v) = config::get_dotted(cfg, k) {
                    println!("{k} = {v}");
                    printed += 1;
                }
            }
            if printed == 0 {
                println!("(no keys set)");
            }
        }
    }
    Ok(())
}
