//! `mnem unintegrate` - remove mnem wiring from agent hosts.
//!
//! Reads `~/.mnemglobal/integrations.toml` to discover what was wired,
//! then calls `do_undo` for each selected host and removes its record from
//! the registry.

use anyhow::Result;
use clap::Args;

use crate::integrate::{Host, IntegrationRegistry, deregister_integration, do_undo};

#[derive(Args, Debug)]
pub(crate) struct UnintegrateArgs {
    /// Specific host(s) to remove (e.g. `claude-code`, `cursor`).
    /// If omitted, an interactive prompt lists what is wired.
    pub hosts: Vec<String>,

    /// Remove all wired hosts without prompting.
    #[arg(long)]
    pub all: bool,

    /// Show what would be done without modifying any files.
    #[arg(long)]
    pub dry_run: bool,
}

pub(crate) fn run(args: UnintegrateArgs) -> Result<()> {
    let reg = IntegrationRegistry::load();

    if reg.hosts.is_empty() && args.hosts.is_empty() && !args.all {
        println!("No integrations recorded in ~/.mnemglobal/integrations.toml.");
        println!("Nothing to remove.");
        return Ok(());
    }

    let selected: Vec<Host> = if args.all {
        // All hosts that have a record.
        if reg.hosts.is_empty() {
            println!("No integrations recorded. Nothing to remove.");
            return Ok(());
        }
        reg.hosts
            .iter()
            .filter_map(|r| Host::parse(&r.slug))
            .collect()
    } else if !args.hosts.is_empty() {
        // Named hosts from CLI args.
        let mut out = Vec::new();
        for name in &args.hosts {
            match Host::parse(name) {
                Some(h) => out.push(h),
                None => {
                    eprintln!("unintegrate: unknown host {:?}", name);
                    eprintln!(
                        "  Valid hosts: {}",
                        Host::all()
                            .iter()
                            .map(|h| h.slug())
                            .collect::<Vec<_>>()
                            .join(", ")
                    );
                }
            }
        }
        if out.is_empty() {
            return Ok(());
        }
        out
    } else {
        // Interactive: show registry, let user pick by number.
        interactive_select(&reg)?
    };

    if selected.is_empty() {
        println!("Nothing selected.");
        return Ok(());
    }

    println!(
        "Removing mnem wiring{}:",
        if args.dry_run { " (dry-run)" } else { "" }
    );
    for host in &selected {
        do_undo(*host, args.dry_run)?;
        if !args.dry_run {
            deregister_integration(*host);
        }
    }

    if !args.dry_run {
        println!();
        println!("Done. Restart each agent host you modified.");
    }
    Ok(())
}

fn interactive_select(reg: &IntegrationRegistry) -> Result<Vec<Host>> {
    if reg.hosts.is_empty() {
        println!("No integrations recorded in ~/.mnemglobal/integrations.toml.");
        println!("If you integrated before this version, use a named host:");
        let slugs = Host::all()
            .iter()
            .map(|h| h.slug())
            .collect::<Vec<_>>()
            .join(", ");
        println!("  mnem unintegrate <host>  (hosts: {slugs})");
        return Ok(vec![]);
    }

    println!("Integrated hosts (from ~/.mnemglobal/integrations.toml):");
    for (i, r) in reg.hosts.iter().enumerate() {
        let components = if r.components.is_empty() {
            "mcp".to_string()
        } else {
            r.components.join(", ")
        };
        println!("  [{}] {} ({})", i + 1, r.display, components);
    }
    println!("  [a] all");
    println!("  [q] quit");
    print!("Select hosts to remove (e.g. 1,2 or a): ");
    use std::io::Write as _;
    std::io::stdout().flush().ok();

    let mut line = String::new();
    std::io::stdin().read_line(&mut line).ok();
    let input = line.trim();

    if input.eq_ignore_ascii_case("q") || input.is_empty() {
        return Ok(vec![]);
    }
    if input.eq_ignore_ascii_case("a") {
        return Ok(reg.hosts.iter().filter_map(|r| Host::parse(&r.slug)).collect());
    }

    let mut selected = Vec::new();
    for token in input.split(',') {
        let token = token.trim();
        if let Ok(n) = token.parse::<usize>() {
            if n >= 1 && n <= reg.hosts.len() {
                if let Some(h) = Host::parse(&reg.hosts[n - 1].slug) {
                    selected.push(h);
                }
            } else {
                eprintln!("  (skipping out-of-range index {n})");
            }
        } else {
            eprintln!("  (skipping unrecognised token {:?})", token);
        }
    }
    Ok(selected)
}
