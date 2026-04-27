//! First-run onboarding wizard. Fires when a user runs bare `mnem`
//! with no subcommand.
//!
//! Flow:
//!   1. If a repo already exists in CWD / any parent -> print status +
//!      a "what next?" hint. (No wizard; they've been here before.)
//!   2. Otherwise, prompt through: init -> embedder -> integrate ->
//!      demo memory -> first retrieve. Each step is opt-in; saying no
//!      moves to the next.
//!
//! Non-interactive environments (CI, piped stdin, `MNEM_NO_WIZARD=1`)
//! print a short "run `mnem --help`" hint and exit 0 instead of
//! blocking on a prompt that will never come.

use std::io::IsTerminal;
use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{Context, Result};
use dialoguer::{Confirm, Input, Select, theme::ColorfulTheme};

use crate::{commands, config, repo};

/// Entry point for bare `mnem` invocations.
pub(crate) fn run(repo_override: Option<&Path>) -> Result<()> {
    // Non-interactive fast path: if someone pipes the binary or runs
    // it under CI, don't trap them in a wait-on-stdin loop.
    if !is_interactive() {
        print_non_interactive_hint();
        return Ok(());
    }

    // If a repo exists where we look, this is a returning user. Show
    // status + a short "next" menu. Otherwise treat as first run.
    match repo::locate_data_dir(repo_override) {
        Ok(_) => returning_user(repo_override),
        Err(_) => first_run(repo_override),
    }
}

fn is_interactive() -> bool {
    if std::env::var_os("MNEM_NO_WIZARD").is_some() {
        return false;
    }
    // Both stdin AND stdout must be TTYs. Piped input (`echo y | mnem`)
    // and piped output (`mnem | tee`) are both uninteractive shapes we
    // don't want to confuse with a human at a terminal.
    std::io::stdin().is_terminal() && std::io::stdout().is_terminal()
}

fn print_non_interactive_hint() {
    eprintln!("mnem: no subcommand given.");
    eprintln!();
    eprintln!("Run `mnem --help` for the command list,");
    eprintln!("or `mnem init` to create a new repo here.");
    eprintln!();
    eprintln!("(The interactive first-run wizard is disabled in");
    eprintln!(" non-tty / CI environments. Set MNEM_NO_WIZARD=1 to");
    eprintln!(" silence this even when attached to a tty.)");
}

// ----------------------------------------------------------------
// Returning user
// ----------------------------------------------------------------

fn returning_user(repo_override: Option<&Path>) -> Result<()> {
    // Show the status so a returning user sees what's in the repo
    // without having to remember the subcommand name.
    commands::status::run(repo_override)?;
    println!();
    println!("Next steps:");
    println!("  mnem add node --summary \"...\"");
    println!("  mnem retrieve \"query\"");
    println!("  mnem config get <key>          # list: mnem config list");
    println!("  mnem integrate                 # wire into Claude Desktop etc.");
    println!("  mnem --help                    # full command list");
    Ok(())
}

// ----------------------------------------------------------------
// First run
// ----------------------------------------------------------------

fn first_run(repo_override: Option<&Path>) -> Result<()> {
    let theme = ColorfulTheme::default();

    println!("{}", banner());
    println!();
    println!("No mnem repo here yet. Walk through the one-time setup?");
    println!("(You can Ctrl-C at any point; nothing is written until you confirm.)");
    println!();

    if !Confirm::with_theme(&theme)
        .with_prompt("Continue with setup?")
        .default(true)
        .interact()?
    {
        println!("No problem. Run `mnem init` when you're ready.");
        return Ok(());
    }

    // ---------- Step 1: init ----------
    let target = step_init(repo_override, &theme)?;
    let data_dir = target.join(repo::MNEM_DIR);

    // ---------- Step 2: identity ----------
    step_identity(&data_dir, &theme)?;

    // ---------- Step 3: embedder ----------
    step_embedder(&data_dir, &theme)?;

    // ---------- Step 4: integrate ----------
    step_integrate(&theme)?;

    // ---------- Step 5: demo memory + retrieve ----------
    step_demo(&data_dir, &theme)?;

    println!();
    println!("Setup complete. From here:");
    println!("  mnem add node --summary \"...\"");
    println!("  mnem retrieve \"your question\"");
    println!("  mnem doctor                    # sanity check the config");
    println!("  mnem --help                    # full command list");
    Ok(())
}

const fn banner() -> &'static str {
    // Keep the ASCII minimal: terminals without a good font render
    // fancy blocks as tofu, which is worse than plain text.
    "\
mnem - git for knowledge graphs
-------------------------------"
}

// ---------- Step 1: init ----------

fn step_init(repo_override: Option<&Path>, theme: &ColorfulTheme) -> Result<PathBuf> {
    let default = repo_override
        .map(Path::to_path_buf)
        .or_else(|| std::env::current_dir().ok())
        .unwrap_or_else(|| PathBuf::from("."));

    let raw: String = Input::with_theme(theme)
        .with_prompt("Where should the repo live?")
        .default(default.display().to_string())
        .interact_text()?;
    let target = PathBuf::from(raw);

    let data_dir = target.join(repo::MNEM_DIR);
    if data_dir.exists() {
        println!("(already initialised at {})", data_dir.display());
        return Ok(target);
    }

    // Re-use the `mnem init` command's implementation so we exercise
    // exactly one code path for repo creation.
    let args = commands::init::Args {
        path: Some(target.clone()),
    };
    commands::init::run(None, args)?;
    Ok(target)
}

// ---------- Step 2: identity ----------

fn step_identity(data_dir: &Path, theme: &ColorfulTheme) -> Result<()> {
    let mut cfg = config::load(data_dir).unwrap_or_default();
    if cfg.user.name.is_some() && cfg.user.email.is_some() {
        return Ok(());
    }

    if !Confirm::with_theme(theme)
        .with_prompt("Set your commit identity now? (name + email)")
        .default(true)
        .interact()?
    {
        return Ok(());
    }

    let name: String = Input::with_theme(theme)
        .with_prompt("Your name")
        .allow_empty(false)
        .interact_text()?;
    let email: String = Input::with_theme(theme)
        .with_prompt("Your email")
        .allow_empty(true)
        .interact_text()?;

    cfg.user.name = Some(name);
    if !email.is_empty() {
        cfg.user.email = Some(email);
    }
    config::save(data_dir, &cfg)?;
    Ok(())
}

// ---------- Step 3: embedder ----------

fn step_embedder(data_dir: &Path, theme: &ColorfulTheme) -> Result<()> {
    let mut cfg = config::load(data_dir).unwrap_or_default();
    if cfg.embed.is_some() {
        println!("(embed provider already configured; skipping)");
        return Ok(());
    }

    let choices = [
        "Ollama nomic-embed-text   (local, 768-dim, smallest + fastest)",
        "Ollama bge-large          (local, 1024-dim, ~2 MTEB points above nomic)",
        "OpenAI text-embedding-3-small (cloud, 1536-dim, needs OPENAI_API_KEY)",
        "OpenAI text-embedding-3-large (cloud, 3072-dim, premium)",
        "Skip (filter-only retrieval; semantic search off)",
    ];
    let pick = Select::with_theme(theme)
        .with_prompt("Which embedder?")
        .items(&choices)
        .default(0)
        .interact()?;

    match pick {
        0 | 1 => {
            // Ollama variants
            let model = if pick == 0 {
                "nomic-embed-text"
            } else {
                "bge-large"
            };
            if which("ollama").is_none() {
                println!("  Ollama is not on PATH. Install from https://ollama.com/download,");
                println!("  then `ollama serve &` and `ollama pull {model}`.");
                println!("  Saving config anyway - `mnem embed` will work once ollama is up.");
            }
            config::set_dotted(&mut cfg, "embed.provider", Some("ollama".into()))?;
            config::set_dotted(&mut cfg, "embed.model", Some(model.into()))?;
            config::save(data_dir, &cfg)?;
            println!("  Configured: embed.provider=ollama, embed.model={model}");
        }
        2 | 3 => {
            // OpenAI variants
            let model = if pick == 2 {
                "text-embedding-3-small"
            } else {
                "text-embedding-3-large"
            };
            let has_key = std::env::var("OPENAI_API_KEY").is_ok();
            config::set_dotted(&mut cfg, "embed.provider", Some("openai".into()))?;
            config::set_dotted(&mut cfg, "embed.model", Some(model.into()))?;
            config::save(data_dir, &cfg)?;
            println!("  Configured: embed.provider=openai, embed.model={model}");
            if !has_key {
                println!("  NOTE: OPENAI_API_KEY is not set in your environment. Export it");
                println!("  before running `mnem retrieve --text ...` / `mnem embed`.");
            }
        }
        _ => {
            println!("  Semantic search off. You can enable it later with:");
            println!("    mnem config set embed.provider ollama");
            println!("    mnem config set embed.model    nomic-embed-text");
        }
    }
    Ok(())
}

/// Cross-platform `which` that doesn't pull in a new dep; good enough
/// for the wizard's "is ollama on PATH?" check.
fn which(cmd: &str) -> Option<PathBuf> {
    let exe = if cfg!(windows) {
        format!("{cmd}.exe")
    } else {
        cmd.to_string()
    };
    std::env::var_os("PATH").and_then(|paths| {
        std::env::split_paths(&paths).find_map(|dir| {
            let p = dir.join(&exe);
            p.is_file().then_some(p)
        })
    })
}

// ---------- Step 4: integrate ----------

fn step_integrate(theme: &ColorfulTheme) -> Result<()> {
    if !Confirm::with_theme(theme)
        .with_prompt(
            "Wire mnem into detected agent hosts now? (Claude Desktop, Cursor, Continue, Zed, ...)",
        )
        .default(true)
        .interact()?
    {
        println!("(skipped; run `mnem integrate` later to wire it up)");
        return Ok(());
    }

    // Shell out to `mnem integrate` so we don't duplicate the TUI
    // logic here. Any errors bubble up but don't kill the wizard;
    // integration is optional.
    let exe = std::env::current_exe().context("finding current mnem executable path")?;
    let status = Command::new(&exe)
        .arg("integrate")
        .status()
        .context("spawning `mnem integrate`")?;
    if !status.success() {
        println!("(integrate exited with status {status}; you can re-run it later)");
    }
    Ok(())
}

// ---------- Step 5: demo memory + retrieve ----------

fn step_demo(data_dir: &Path, theme: &ColorfulTheme) -> Result<()> {
    if !Confirm::with_theme(theme)
        .with_prompt("Seed a demo memory and run a retrieve to verify everything works?")
        .default(true)
        .interact()?
    {
        return Ok(());
    }

    // Add one node via the public CLI path so we exercise the same
    // auto-embed flow `mnem add node` uses.
    let exe = std::env::current_exe()?;
    let parent = data_dir.parent().unwrap_or(Path::new("."));

    let add_status = Command::new(&exe)
        .args([
            "-R",
            parent.display().to_string().as_str(),
            "add",
            "node",
            "--summary",
            "mnem stores versioned, content-addressed graph memory with retrieval under a token budget",
            "-m",
            "wizard: seed demo memory",
        ])
        .status()
        .context("spawning `mnem add node`")?;
    if !add_status.success() {
        println!("(add node failed; skipping retrieve)");
        return Ok(());
    }

    println!();
    println!("Running: mnem retrieve \"what is mnem\"");
    let _ret_status = Command::new(&exe)
        .args([
            "-R",
            parent.display().to_string().as_str(),
            "retrieve",
            "what is mnem",
        ])
        .status();
    Ok(())
}
