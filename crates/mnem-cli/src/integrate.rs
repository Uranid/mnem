//! `mnem integrate` - wire mnem into every agent host on the machine.
//!
//! Detects installed MCP-aware agent hosts (Claude Desktop, Cursor,
//! Continue, Zed) by probing platform-specific config paths. Merges a
//! `mnem` MCP-server entry into each selected host's config with an
//! atomic temp-file-plus-rename after a timestamped backup. Never
//! overwrites other MCP entries.

use std::fs;
use std::io::Write as _;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow};
use serde_json::{Map, Value, json};

// ---------- Host model ----------

/// A host we know how to wire.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Host {
    ClaudeDesktop,
    Cursor,
    Continue_,
    Zed,
    /// Audit fix G7 (2026-04-25): Claude Code CLI. MCP entries live in
    /// `~/.claude.json` under the top-level `mcpServers` map. Hooks live
    /// separately in `~/.claude/settings.json` (see [`Self::hooks_path`]).
    ClaudeCode,
    /// Audit fix G7 (2026-04-25): Gemini CLI. MCP entries live in
    /// `~/.gemini/settings.json` under the top-level `mcpServers` map.
    GeminiCli,
}

impl Host {
    pub(crate) const fn all() -> &'static [Host] {
        &[
            Host::ClaudeDesktop,
            Host::Cursor,
            Host::Continue_,
            Host::Zed,
            Host::ClaudeCode,
            Host::GeminiCli,
        ]
    }

    pub(crate) const fn slug(self) -> &'static str {
        match self {
            Host::ClaudeDesktop => "claude-desktop",
            Host::Cursor => "cursor",
            Host::Continue_ => "continue",
            Host::Zed => "zed",
            Host::ClaudeCode => "claude-code",
            Host::GeminiCli => "gemini-cli",
        }
    }

    pub(crate) const fn display(self) -> &'static str {
        match self {
            Host::ClaudeDesktop => "Claude Desktop",
            Host::Cursor => "Cursor",
            Host::Continue_ => "Continue",
            Host::Zed => "Zed",
            Host::ClaudeCode => "Claude Code",
            Host::GeminiCli => "Gemini CLI",
        }
    }

    pub(crate) fn parse(s: &str) -> Option<Host> {
        match s.to_ascii_lowercase().as_str() {
            "claude-desktop" | "claude_desktop" => Some(Host::ClaudeDesktop),
            "cursor" => Some(Host::Cursor),
            "continue" => Some(Host::Continue_),
            "zed" => Some(Host::Zed),
            "claude-code" | "claude_code" | "claude" => Some(Host::ClaudeCode),
            "gemini-cli" | "gemini_cli" | "gemini" => Some(Host::GeminiCli),
            _ => None,
        }
    }

    /// The config-file path for this host on this OS, or `None` if we
    /// have no rule for this (OS, host) combination.
    pub(crate) fn config_path(self) -> Option<PathBuf> {
        let home = dirs::home_dir()?;
        match self {
            Host::ClaudeDesktop => {
                if cfg!(target_os = "macos") {
                    Some(
                        home.join("Library")
                            .join("Application Support")
                            .join("Claude")
                            .join("claude_desktop_config.json"),
                    )
                } else if cfg!(target_os = "windows") {
                    // %APPDATA%\Claude\claude_desktop_config.json
                    dirs::config_dir().map(|d| d.join("Claude").join("claude_desktop_config.json"))
                } else {
                    Some(
                        home.join(".config")
                            .join("Claude")
                            .join("claude_desktop_config.json"),
                    )
                }
            }
            Host::Cursor => Some(home.join(".cursor").join("mcp.json")),
            Host::Continue_ => Some(home.join(".continue").join("config.json")),
            Host::Zed => {
                if cfg!(target_os = "macos") {
                    Some(
                        home.join("Library")
                            .join("Application Support")
                            .join("Zed")
                            .join("settings.json"),
                    )
                } else {
                    Some(home.join(".config").join("zed").join("settings.json"))
                }
            }
            // Claude Code keeps its global config at ~/.claude.json across
            // all platforms (the modern CLI's behaviour). Project-scoped
            // .mcp.json is also supported by Claude Code itself but is
            // out-of-scope for `mnem integrate`, which writes user-global
            // configuration only.
            Host::ClaudeCode => Some(home.join(".claude.json")),
            // Gemini CLI follows the standard ~/.gemini/settings.json
            // shape on every OS.
            Host::GeminiCli => Some(home.join(".gemini").join("settings.json")),
        }
    }

    /// Path to the hooks config for this host, or `None` if the host
    /// does not support a separate hooks file. Currently only Claude
    /// Code supports hooks (audit fix G2, 2026-04-25): hooks live in
    /// `~/.claude/settings.json`, distinct from the MCP entry which
    /// lives in `~/.claude.json`.
    pub(crate) fn hooks_path(self) -> Option<PathBuf> {
        let home = dirs::home_dir()?;
        match self {
            Host::ClaudeCode => Some(home.join(".claude").join("settings.json")),
            _ => None,
        }
    }

    /// Path to the markdown file where the host loads its
    /// project-rules / custom-instructions / system-prompt content
    /// from disk, or `None` if the host has no file-based mechanism
    /// (UI-only custom-instructions, Claude Desktop being the
    /// canonical example). Audit fix (2026-04-26): closes the
    /// last "user pastes the prompt" seam in the customer flow.
    ///
    /// Today: ClaudeCode → `~/.claude/CLAUDE.md` (the user-global
    /// rules file Claude Code reads on every session). Other hosts
    /// either lack a stable file location (Cursor's `.cursor/rules/`
    /// is project-scoped, not user-global) or use a UI-only panel
    /// (Claude Desktop). They return `None` and the caller falls
    /// back to the printed-prompt copy-paste flow.
    pub(crate) fn system_prompt_path(self) -> Option<PathBuf> {
        let home = dirs::home_dir()?;
        match self {
            Host::ClaudeCode => Some(home.join(".claude").join("CLAUDE.md")),
            _ => None,
        }
    }
}

/// How the host embeds MCP servers in its JSON config.
#[derive(Debug, Clone, Copy)]
enum Schema {
    /// Top-level `mcpServers.<name>` map.
    McpServersTopLevel,
    /// Nested under `experimental.context_servers.<name>`.
    ZedNested,
}

const fn schema_of(h: Host) -> Schema {
    match h {
        Host::ClaudeDesktop
        | Host::Cursor
        | Host::Continue_
        | Host::ClaudeCode
        | Host::GeminiCli => Schema::McpServersTopLevel,
        Host::Zed => Schema::ZedNested,
    }
}

// ---------- CLI surface ----------

#[derive(clap::Args, Debug)]
#[command(after_long_help = "\
Examples:

  mnem integrate                       # interactive; detect + prompt
  mnem integrate --all                 # wire every detected host, no prompts
  mnem integrate claude-desktop cursor # wire these two, non-interactive
  mnem integrate --show claude-desktop # print JSON for copy-paste
  mnem integrate --check               # report wired state, mutate nothing
  mnem integrate --undo claude-desktop # remove mnem from one host
  mnem integrate --all --dry-run       # diff mode; write nothing
")]
pub(crate) struct Args {
    /// Hosts to wire. Omit to enter interactive mode.
    pub hosts: Vec<String>,

    /// Wire every detected host without prompting.
    #[arg(long)]
    pub all: bool,

    /// Report wired state and exit. Non-mutating.
    #[arg(long)]
    pub check: bool,

    /// Print the JSON block for HOST and exit. Non-mutating.
    #[arg(long, value_name = "HOST")]
    pub show: Option<String>,

    /// Remove mnem from HOST's config (or all hosts with `--all`).
    #[arg(long, value_name = "HOST")]
    pub undo: Option<String>,

    /// Print what would change without writing.
    #[arg(long)]
    pub dry_run: bool,

    /// Repo path to point hosts at. Defaults to `.mnem` resolved from
    /// the current working directory.
    #[arg(long, value_name = "PATH")]
    pub target_repo: Option<PathBuf>,

    /// Audit fix G1/G4 (2026-04-25): print the recommended mnem system
    /// prompt and exit. Pipe to `pbcopy` / `clip` and paste into your
    /// host's custom-instructions panel. Non-mutating.
    #[arg(long = "system-prompt")]
    pub system_prompt: bool,

    /// Audit fix G2 (2026-04-25): also write a `UserPromptSubmit`
    /// hook into hosts that support hooks. Today: Claude Code only.
    /// Other hosts ignore the flag (the hook config is no-op for them).
    /// The hook calls `mnem retrieve` on every user message and pipes
    /// results into the LLM's context, giving a guaranteed before-turn
    /// memory injection that does not depend on the LLM remembering
    /// to call the tool.
    #[arg(long = "with-hooks")]
    pub with_hooks: bool,

    /// Audit fix (2026-04-26): also write the recommended mnem LLM
    /// system prompt into the host's project-rules file (today:
    /// `~/.claude/CLAUDE.md` for Claude Code). Closes the last
    /// "user pastes the prompt" seam in the customer flow. The
    /// prompt is wrapped in marker comments so re-running replaces
    /// just the mnem section without clobbering the user's own
    /// rules. Other hosts that have no file-based rules location
    /// (Claude Desktop UI-only) silently skip; for those the
    /// copy-paste flow via `mnem integrate --system-prompt` is
    /// still the only path.
    #[arg(long = "with-system-prompt")]
    pub with_system_prompt: bool,
}

/// Recommended mnem system prompt, embedded at compile time so the
/// binary can print it without needing the source tree on disk. Audit
/// fix G1/G4 (2026-04-25). Source of truth: `docs/system-prompt.md`.
const SYSTEM_PROMPT: &str = r#"# mnem system prompt

This is the recommended system prompt to add to your agent host
(Claude Desktop, Claude Code, Cursor, Continue, Zed, Gemini CLI, ...)
so the LLM uses mnem transparently on every turn - without the user
ever having to mention mnem.

## TL;DR

**Claude Code (one command, fully auto-wired):**

```bash
mnem integrate claude-code --with-hooks --with-system-prompt
```

This writes the MCP server entry to `~/.claude.json`, the
`UserPromptSubmit` hook to `~/.claude/settings.json`, and the prompt
below into a marker-bracketed section of `~/.claude/CLAUDE.md` (your
existing rules in the file are preserved untouched). Restart Claude
Code. Done.

**Other hosts (Claude Desktop, Cursor, Continue, Zed, Gemini CLI):**

```bash
mnem integrate --system-prompt | clip          # Windows
mnem integrate --system-prompt | pbcopy        # macOS
mnem integrate --system-prompt | xclip -selection clipboard   # Linux
```

These hosts have either a UI-only custom-instructions panel (Claude
Desktop) or no stable user-global rules location (Cursor's
`.cursor/rules/` is project-scoped). Paste the printed prompt into
the host's panel manually. `mnem integrate <host>` still wires the
MCP server entry; `--with-hooks` is a no-op on these hosts (no public
hook protocol yet).

## The prompt

```
You have access to mnem, a persistent knowledge graph available via MCP tools
prefixed `mnem_`. Your job is to use it transparently: the user should never
need to mention mnem.

## Reading memory (before you answer)

On EVERY user message:
1. Call `mnem_retrieve` with the user's message as `text` and `token_budget=2000`.
2. If results are returned, weave them into your answer naturally.
3. If the repo is empty or no relevant results: proceed with training knowledge.
4. Do not announce that you are consulting memory unless directly asked.

## Writing memory (after you answer)

After each turn, commit any new facts, preferences, events, or entities
the user stated or confirmed. Use these rules:

- One fact per node. Never combine two separate facts in one summary.
- Every summary must be a complete standalone sentence. No leading pronouns
  ("she", "they", "the above").
- Put human-readable text in `summary`. Put filterable metadata in `props`.
- Use `mnem_resolve_or_create` for named people, places, and organizations
  (NOT `mnem_commit`). Always check for an existing entity before creating
  a new one.
- Connect entities with edges: `works_at`, `lives_in`, `traveling_with`,
  `has_preference`, `extracted_from`, etc. Use the compound
  `mnem_commit_relation` tool when both endpoints are entities - it
  resolve-or-creates both nodes and adds the edge in one call.
- Do NOT commit model output or your own reasoning. Only commit facts the
  user stated or confirmed.

## Node types to use (`ntype` field)

| ntype | Use for |
|---|---|
| `Fact` | Declarative knowledge about the world or the user |
| `Preference` | Stated liking, disliking, or ranking |
| `Event` | Something that happened at a specific time |
| `Entity:Person` | Named person. Use resolve_or_create, anchor on `name` or `email`. |
| `Entity:Organization` | Named company / org. Use resolve_or_create, anchor on `name`. |
| `Entity:Place` | Named place. Use resolve_or_create, anchor on `name`. |
| `Document` | Longer source text (transcript, PDF chunk, web page). |
| `Session` | End-of-conversation summary. Write one at session end. |
| `Goal` | Long-horizon user intention. |
| `Task` | Unit of work with a `status` prop (`todo`/`in_progress`/`done`/`cancelled`). |

## Edge predicates

Use verb-phrase, snake_case names that read left-to-right like English.
The conventional predicates are:

| Predicate | Direction | Meaning |
|---|---|---|
| `works_at` | Person → Organization | employment |
| `lives_in` | Person → Place | residence |
| `has_preference` | Person → Preference | stated preference |
| `traveling_with` | Event → Person | trip companion |
| `happened_before` | Event → Event | temporal order |
| `extracted_from` | Fact → Document/Session | provenance |
| `mentions` | Document → Entity | co-occurrence |
| `revoked_by` | OldFact → NewFact | supersession |

## Forgetting

When the user says "forget X", "stop remembering X", or revokes consent:
1. Call `mnem_retrieve` to find the relevant node UUID.
2. Call `mnem_tombstone_node` with the UUID and the user's own phrasing
   as `reason`.
3. Confirm to the user briefly: "Removed."

## Supersession (when the user changes their mind)

When the user updates a previously stated fact (e.g. "actually I moved
to Paris" after "I live in Berlin"):
1. Resolve or create the new fact node.
2. Add a `revoked_by` edge from the old node to the new one.
   `mnem_retrieve` filters out revoked nodes by default, so the old
   fact stops surfacing without losing the audit trail.

## `agent_id`

Always set `agent_id` to the name of the host you're running in
(`"claude-code"`, `"claude-desktop"`, `"cursor"`, `"gemini-cli"`, etc.)
on `mnem_commit`, `mnem_resolve_or_create`, `mnem_tombstone_node`, and
`mnem_commit_relation` calls. This is what shows up in `mnem log` so
the user can audit which agent wrote what.

## When NOT to use mnem

- Pure computation results ("the sum is 42"): don't commit; the answer
  lives in the conversation.
- Tool-call traces: don't commit; the result might warrant a commit, the
  trace doesn't.
- Generated content (drafts, summaries, code suggestions): don't commit
  unless the user accepts and asks you to remember.
- Within a single conversation turn for re-reads: the context window
  already has it; only call `mnem_retrieve` once per user message.
```

## Why a system prompt at all

mnem ships 14 MCP tools. Without a system prompt, the LLM sees them as
optional and uses them opportunistically. With this prompt, the LLM
treats them as the default reading and writing channel for facts the
user shares.

The pre-prompt hook (`mnem integrate --with-hooks claude-code`) gives
a stronger guarantee: it forces a `mnem_retrieve` call before the LLM
ever sees the user's message. Pair the two for the strongest "automatic
memory" experience.

## See also

- [`agent-playbook.md`](./guide/agent-playbook.md) - the underlying
  policies this prompt encodes (write triggers, shape rules, supersession).
- [`ntype-vocab.md`](./guide/ntype-vocab.md) - the canonical type
  vocabulary the prompt references.
- [`integrate.md`](./guide/integrate.md) - host configuration and the
  `--system-prompt` / `--with-hooks` flags.
- [`mcp.md`](./guide/mcp.md) - the MCP tool reference.
"#;

pub(crate) fn run(args: Args) -> Result<()> {
    // --system-prompt: print the recommended LLM system prompt and
    // bail. Non-mutating, no host wiring. G1/G4 (2026-04-25).
    if args.system_prompt {
        print!("{SYSTEM_PROMPT}");
        return Ok(());
    }

    // --show is the simplest surface: print JSON and bail.
    if let Some(slug) = args.show.as_deref() {
        let host = Host::parse(slug).ok_or_else(|| {
            let known = Host::all()
                .iter()
                .map(|h| h.slug())
                .collect::<Vec<_>>()
                .join(", ");
            anyhow!("unknown host: {slug}. Known: {known}")
        })?;
        let target = resolve_target(args.target_repo.as_deref())?;
        let snippet = snippet_for(host, &target);
        println!("# host: {}", host.display());
        if let Some(p) = host.config_path() {
            println!("# config: {}", p.display());
        }
        println!("{snippet}");
        return Ok(());
    }

    if args.check {
        return do_check();
    }

    if let Some(slug) = args.undo.as_deref() {
        if slug == "all" || args.all {
            // Run per-host undo but COLLECT failures so the CLI
            // surfaces them instead of exiting 0 on a half-failed
            // uninstall. Prior behaviour (`let _ = do_undo(...)`)
            // silently ate errors per host; if a user asked "remove
            // mnem from every agent" and half the hosts errored,
            // they'd never know.
            let mut failures: Vec<(Host, anyhow::Error)> = Vec::new();
            for h in Host::all() {
                if let Err(e) = do_undo(*h, args.dry_run) {
                    failures.push((*h, e));
                }
            }
            if !failures.is_empty() {
                for (h, e) in &failures {
                    eprintln!("undo {}: {e}", h.slug());
                }
                anyhow::bail!(
                    "{} of {} hosts failed to undo",
                    failures.len(),
                    Host::all().len()
                );
            }
            return Ok(());
        } else {
            let host = Host::parse(slug).ok_or_else(|| {
                let known = Host::all()
                    .iter()
                    .map(|h| h.slug())
                    .collect::<Vec<_>>()
                    .join(", ");
                anyhow!("unknown host: {slug}. Known: {known}")
            })?;
            do_undo(host, args.dry_run)?;
        }
        return Ok(());
    }

    let target = resolve_target(args.target_repo.as_deref())?;

    let selected: Vec<Host> = if args.all {
        Host::all()
            .iter()
            .filter(|h| {
                h.config_path()
                    .is_some_and(|p| p.parent().is_some_and(Path::exists))
            })
            .copied()
            .collect()
    } else if !args.hosts.is_empty() {
        let mut out = Vec::new();
        for s in &args.hosts {
            out.push(Host::parse(s).ok_or_else(|| {
                let known = Host::all()
                    .iter()
                    .map(|h| h.slug())
                    .collect::<Vec<_>>()
                    .join(", ");
                anyhow!("unknown host: {s}. Known: {known}")
            })?);
        }
        out
    } else {
        interactive_select()?
    };

    if selected.is_empty() {
        println!("no hosts selected");
        return Ok(());
    }

    let stamp = timestamp();
    println!("Writing configs (backing up with .bak-{stamp}):");
    for host in selected {
        match do_wire(host, &target, &stamp, args.dry_run) {
            Ok(WireOutcome::Wrote) => {
                println!("  ok {}  wired -> {}", host.display(), target.display());
            }
            Ok(WireOutcome::DryRun(diff)) => {
                println!("  -- {} (dry-run)\n{diff}", host.display());
            }
            Ok(WireOutcome::AlreadyWired) => {
                println!(
                    "  =  {}  already wired -> {}",
                    host.display(),
                    target.display()
                );
            }
            Err(e) => {
                println!("  !  {}  {e}", host.display());
            }
        }
        // G2 (2026-04-25): if --with-hooks, also write the
        // UserPromptSubmit hook for hosts that support hooks. Today
        // that's Claude Code only; other hosts skip silently.
        if args.with_hooks && host.hooks_path().is_some() {
            match do_wire_hooks(host, &stamp, args.dry_run) {
                Ok(WireOutcome::Wrote) => {
                    println!("  ok {}  hooks wired", host.display());
                }
                Ok(WireOutcome::DryRun(diff)) => {
                    println!("  -- {} hooks (dry-run)\n{diff}", host.display());
                }
                Ok(WireOutcome::AlreadyWired) => {
                    println!("  =  {}  hooks already wired", host.display());
                }
                Err(e) => {
                    println!("  !  {}  hooks: {e}", host.display());
                }
            }
        }
        // 2026-04-26: if --with-system-prompt, also write the mnem
        // system prompt into the host's project-rules file (today:
        // Claude Code only). Closes the last copy-paste seam in the
        // customer flow. Other hosts skip silently.
        if args.with_system_prompt && host.system_prompt_path().is_some() {
            match do_wire_system_prompt(host, &stamp, args.dry_run) {
                Ok(WireOutcome::Wrote) => {
                    println!("  ok {}  system prompt wired", host.display());
                }
                Ok(WireOutcome::DryRun(diff)) => {
                    println!("  -- {} system prompt (dry-run)\n{diff}", host.display());
                }
                Ok(WireOutcome::AlreadyWired) => {
                    println!("  =  {}  system prompt already wired", host.display());
                }
                Err(e) => {
                    println!("  !  {}  system prompt: {e}", host.display());
                }
            }
        }
    }

    println!();
    println!("Next steps:");
    println!("  1. Restart each agent host you wired.");
    println!("  2. Verify:  mnem doctor");
    match (args.with_hooks, args.with_system_prompt) {
        (true, true) => {
            // Fully automatic - nothing else to do.
        }
        (true, false) => {
            println!(
                "  3. Recommended: also write the LLM system prompt to your host's project rules:"
            );
            println!(
                "       a) Auto-write (Claude Code today):    mnem integrate --with-system-prompt"
            );
            println!(
                "       b) Copy-paste into UI panel (others): mnem integrate --system-prompt | clip"
            );
        }
        (false, true) => {
            println!("  3. Recommended: also add a guaranteed before-prompt memory hook:");
            println!("       mnem integrate --with-hooks");
        }
        (false, false) => {
            println!("  3. (Recommended) Add the recommended LLM system prompt:");
            println!("       mnem integrate --with-system-prompt        (auto-write, Claude Code)");
            println!("       mnem integrate --system-prompt | clip      (copy-paste, all hosts)");
            println!("  4. (Recommended) Add a guaranteed before-prompt memory hook:");
            println!("       mnem integrate --with-hooks                (Claude Code today)");
        }
    }
    // Path A audit fix (2026-04-26): nudge users who installed the
    // minimal CLI toward the bundled-embedder build so semantic
    // retrieve works without an Ollama daemon. The check is at
    // compile time so the hint never falsely fires for users who
    // already have the bundled embedder.
    #[cfg(not(feature = "bundled-embedder"))]
    {
        println!();
        println!("Note: this `mnem` binary was built without `--features bundled-embedder`.");
        println!("      Semantic `mnem retrieve --text` will return zero hits until you configure");
        println!("      an embedder. Two paths:");
        println!(
            "        a) Reinstall with the bundled MiniLM:   cargo install mnem-cli --features bundled-embedder"
        );
        println!(
            "        b) Configure your own provider:         see docs/guide/getting-started.md#switching-to-a-custom-embedder-later"
        );
    }
    println!();
    println!("Run `mnem integrate` again any time to re-sync.");
    Ok(())
}

// ---------- Interactive selection ----------

fn interactive_select() -> Result<Vec<Host>> {
    use dialoguer::{MultiSelect, theme::ColorfulTheme};

    let entries: Vec<(Host, bool, String)> = Host::all()
        .iter()
        .map(|h| {
            let detected = h
                .config_path()
                .is_some_and(|p| p.parent().is_some_and(Path::exists));
            let label = if let Some(p) = h.config_path() {
                let show = p
                    .parent()
                    .map_or_else(|| p.display().to_string(), |d| d.display().to_string());
                if detected {
                    format!("{}  (at {show})", h.display())
                } else {
                    format!("{}  (not found)", h.display())
                }
            } else {
                format!("{}  (unsupported on this OS)", h.display())
            };
            (*h, detected, label)
        })
        .collect();

    println!("mnem integrate - wire mnem into agent hosts\n");
    for (_, detected, label) in &entries {
        let prefix = if *detected { "[x]" } else { "[ ]" };
        println!("  {prefix} {label}");
    }
    println!();

    let items: Vec<&str> = entries.iter().map(|(_, _, s)| s.as_str()).collect();
    let defaults: Vec<bool> = entries.iter().map(|(_, d, _)| *d).collect();

    let picks = MultiSelect::with_theme(&ColorfulTheme::default())
        .with_prompt("Which to wire? (space to toggle, enter to confirm)")
        .items(&items)
        .defaults(&defaults)
        .interact()
        .context("interactive prompt failed")?;

    Ok(picks.into_iter().map(|i| entries[i].0).collect())
}

// ---------- Wire / Undo / Check ----------

enum WireOutcome {
    Wrote,
    DryRun(String),
    AlreadyWired,
}

fn do_wire(host: Host, target: &Path, stamp: &str, dry_run: bool) -> Result<WireOutcome> {
    let path = host
        .config_path()
        .ok_or_else(|| anyhow!("unsupported on this OS"))?;

    // Read existing or start from empty object.
    let mut root = if path.exists() {
        let s = fs::read_to_string(&path).with_context(|| format!("reading {}", path.display()))?;
        if s.trim().is_empty() {
            Value::Object(Map::new())
        } else {
            serde_json::from_str::<Value>(&s)
                .with_context(|| format!("parsing {}", path.display()))?
        }
    } else {
        Value::Object(Map::new())
    };

    let changed = match schema_of(host) {
        Schema::McpServersTopLevel => set_top_level(&mut root, target),
        Schema::ZedNested => set_zed_nested(&mut root, target),
    };

    if !changed {
        return Ok(WireOutcome::AlreadyWired);
    }

    let new_text = serde_json::to_string_pretty(&root).context("serialising merged config")?;

    if dry_run {
        return Ok(WireOutcome::DryRun(indent(&new_text, "     ")));
    }

    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).with_context(|| format!("creating {}", parent.display()))?;
    }
    if path.exists() {
        let bak = path.with_extension(format!(
            "{}.bak-{stamp}",
            path.extension().and_then(|s| s.to_str()).unwrap_or("json")
        ));
        fs::copy(&path, &bak).with_context(|| format!("backing up to {}", bak.display()))?;
    }
    atomic_write(&path, &new_text)?;
    Ok(WireOutcome::Wrote)
}

/// Markers that bracket the mnem-managed section of a host's
/// project-rules markdown file. Used to make
/// `--with-system-prompt` idempotent and to keep user-authored
/// content outside the markers untouched on re-run / undo.
const SYSTEM_PROMPT_MARKER_START: &str = "<!-- mnem-system-prompt:v1:start -->";
const SYSTEM_PROMPT_MARKER_END: &str = "<!-- mnem-system-prompt:v1:end -->";

/// Write or update the recommended mnem LLM system prompt into the
/// host's project-rules file (today: Claude Code only). Audit fix
/// (2026-04-26).
///
/// Idempotent: re-running replaces just the marker-bracketed mnem
/// section, leaving any user-authored rules outside the markers
/// untouched. Backs up the file before edit (timestamped `.bak-*`).
fn do_wire_system_prompt(host: Host, stamp: &str, dry_run: bool) -> Result<WireOutcome> {
    let Some(path) = host.system_prompt_path() else {
        return Ok(WireOutcome::AlreadyWired);
    };

    let existing = if path.exists() {
        fs::read_to_string(&path).with_context(|| format!("reading {}", path.display()))?
    } else {
        String::new()
    };

    let new_content = merge_system_prompt(&existing, SYSTEM_PROMPT);
    if new_content == existing {
        return Ok(WireOutcome::AlreadyWired);
    }

    if dry_run {
        // Show only the diff between the surrounding chrome (markers)
        // and the actual prompt-body change. Full prompt is many KB;
        // surfacing it inline in dry-run output is more noise than
        // signal.
        return Ok(WireOutcome::DryRun(format!(
            "     (writing mnem-managed section to {} - \
              {} bytes total, {} bytes changed)",
            path.display(),
            new_content.len(),
            new_content.len().abs_diff(existing.len())
        )));
    }

    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).with_context(|| format!("creating {}", parent.display()))?;
    }
    if path.exists() {
        let bak = path.with_extension(format!(
            "{}.bak-{stamp}",
            path.extension().and_then(|s| s.to_str()).unwrap_or("md")
        ));
        fs::copy(&path, &bak).with_context(|| format!("backing up to {}", bak.display()))?;
    }
    atomic_write(&path, &new_content)?;
    Ok(WireOutcome::Wrote)
}

/// Merge the mnem system prompt into existing rules-file content.
/// Returns the new file contents.
///
/// Algorithm:
///   - If `existing` already contains the marker pair, replace the
///     content between them with `prompt`.
///   - Otherwise, append a new marker-bracketed section to the end
///     of `existing` (preceded by a blank line if needed).
fn merge_system_prompt(existing: &str, prompt: &str) -> String {
    let prompt_block = format!(
        "{}\n{}\n{}\n",
        SYSTEM_PROMPT_MARKER_START,
        prompt.trim_end(),
        SYSTEM_PROMPT_MARKER_END
    );

    if let (Some(start), Some(end)) = (
        existing.find(SYSTEM_PROMPT_MARKER_START),
        existing.find(SYSTEM_PROMPT_MARKER_END),
    ) && end > start
    {
        let end_inclusive = end + SYSTEM_PROMPT_MARKER_END.len();
        // Eat one trailing newline if present so re-running doesn't
        // accumulate blank lines between the marker and whatever
        // followed it before.
        let mut tail_start = end_inclusive;
        if existing.as_bytes().get(tail_start) == Some(&b'\n') {
            tail_start += 1;
        }
        return format!(
            "{}{}{}",
            &existing[..start],
            &prompt_block,
            &existing[tail_start..]
        );
    }

    if existing.is_empty() {
        return prompt_block;
    }
    let needs_separator = !existing.ends_with("\n\n");
    let separator = if existing.ends_with('\n') {
        "\n"
    } else {
        "\n\n"
    };
    if needs_separator {
        format!("{existing}{separator}{prompt_block}")
    } else {
        format!("{existing}{prompt_block}")
    }
}

/// Remove the marker-bracketed mnem section from a rules file.
/// Returns the new content; caller decides whether to write or
/// delete the file when the result is empty.
fn remove_system_prompt(existing: &str) -> String {
    if let (Some(start), Some(end)) = (
        existing.find(SYSTEM_PROMPT_MARKER_START),
        existing.find(SYSTEM_PROMPT_MARKER_END),
    ) && end > start
    {
        let end_inclusive = end + SYSTEM_PROMPT_MARKER_END.len();
        let mut tail_start = end_inclusive;
        if existing.as_bytes().get(tail_start) == Some(&b'\n') {
            tail_start += 1;
        }
        let mut head_end = start;
        // Eat the blank line we may have inserted when appending.
        while head_end > 0 {
            let ch = existing.as_bytes()[head_end - 1];
            if ch == b'\n' || ch == b' ' || ch == b'\r' || ch == b'\t' {
                head_end -= 1;
            } else {
                break;
            }
        }
        let mut out = String::with_capacity(existing.len());
        out.push_str(&existing[..head_end]);
        if !out.is_empty() && !out.ends_with('\n') {
            out.push('\n');
        }
        out.push_str(&existing[tail_start..]);
        return out;
    }
    existing.to_string()
}

/// Write or update the `UserPromptSubmit` hook entry for hosts that
/// support hooks (currently Claude Code only). Audit fix G2
/// (2026-04-25).
///
/// Idempotent: re-running replaces the existing mnem entry rather
/// than appending, so users can safely run `integrate --with-hooks`
/// multiple times. Other hooks in the file are preserved.
fn do_wire_hooks(host: Host, stamp: &str, dry_run: bool) -> Result<WireOutcome> {
    let Some(path) = host.hooks_path() else {
        // Host has no hooks support; nothing to do.
        return Ok(WireOutcome::AlreadyWired);
    };

    let mut root = if path.exists() {
        let s = fs::read_to_string(&path).with_context(|| format!("reading {}", path.display()))?;
        if s.trim().is_empty() {
            Value::Object(Map::new())
        } else {
            serde_json::from_str::<Value>(&s)
                .with_context(|| format!("parsing {}", path.display()))?
        }
    } else {
        Value::Object(Map::new())
    };

    let changed = set_user_prompt_hook(&mut root);
    if !changed {
        return Ok(WireOutcome::AlreadyWired);
    }

    let new_text = serde_json::to_string_pretty(&root).context("serialising hooks config")?;
    if dry_run {
        return Ok(WireOutcome::DryRun(indent(&new_text, "     ")));
    }

    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).with_context(|| format!("creating {}", parent.display()))?;
    }
    if path.exists() {
        let bak = path.with_extension(format!(
            "{}.bak-{stamp}",
            path.extension().and_then(|s| s.to_str()).unwrap_or("json")
        ));
        fs::copy(&path, &bak).with_context(|| format!("backing up to {}", bak.display()))?;
    }
    atomic_write(&path, &new_text)?;
    Ok(WireOutcome::Wrote)
}

/// Set `root.hooks.UserPromptSubmit[mnem-entry] = {matcher, hooks: [...]}`.
/// Identifies the mnem entry by the substring `mnem retrieve` in the
/// hook's command field; replaces it if present, else appends.
/// Returns `true` when the file changed.
fn set_user_prompt_hook(root: &mut Value) -> bool {
    ensure_object(root);
    let obj = root.as_object_mut().expect("ensured object");
    let hooks = obj
        .entry("hooks")
        .or_insert_with(|| Value::Object(Map::new()));
    if !hooks.is_object() {
        *hooks = Value::Object(Map::new());
    }
    let hooks_map = hooks.as_object_mut().expect("object");
    let entries = hooks_map
        .entry("UserPromptSubmit")
        .or_insert_with(|| Value::Array(Vec::new()));
    if !entries.is_array() {
        *entries = Value::Array(Vec::new());
    }
    let arr = entries.as_array_mut().expect("array");
    let new_val = user_prompt_hook_value();

    // Replace existing mnem entry if present.
    for entry in arr.iter_mut() {
        if entry_is_mnem_hook(entry) {
            if entry == &new_val {
                return false;
            }
            *entry = new_val;
            return true;
        }
    }
    // Otherwise append.
    arr.push(new_val);
    true
}

fn remove_user_prompt_hook(root: &mut Value) -> bool {
    let Some(obj) = root.as_object_mut() else {
        return false;
    };
    let Some(hooks) = obj.get_mut("hooks") else {
        return false;
    };
    let Some(hooks_map) = hooks.as_object_mut() else {
        return false;
    };
    let Some(entries) = hooks_map.get_mut("UserPromptSubmit") else {
        return false;
    };
    let Some(arr) = entries.as_array_mut() else {
        return false;
    };
    let before = arr.len();
    arr.retain(|e| !entry_is_mnem_hook(e));
    arr.len() != before
}

/// True iff a `UserPromptSubmit` entry's inner hook command contains
/// `mnem retrieve` - our identifier for the entry mnem owns.
fn entry_is_mnem_hook(entry: &Value) -> bool {
    entry
        .get("hooks")
        .and_then(Value::as_array)
        .is_some_and(|inner| {
            inner.iter().any(|h| {
                h.get("command").and_then(Value::as_str).is_some_and(|c| {
                    // Strip shell quote wrappers so the substring check
                    // matches whether the binary path was wrapped in
                    // single or double quotes (Windows PowerShell uses
                    // `'mnem.exe'`, bash uses `"mnem"` etc).
                    let stripped: String =
                        c.chars().filter(|ch| *ch != '"' && *ch != '\'').collect();
                    stripped.contains("mnem retrieve") || stripped.contains("mnem.exe retrieve")
                })
            })
        })
}

fn do_undo(host: Host, dry_run: bool) -> Result<()> {
    let path = match host.config_path() {
        Some(p) => p,
        None => {
            println!("  -  {}  unsupported on this OS", host.display());
            return Ok(());
        }
    };
    let mcp_changed = if path.exists() {
        let s = fs::read_to_string(&path).with_context(|| format!("reading {}", path.display()))?;
        let mut root: Value = if s.trim().is_empty() {
            Value::Object(Map::new())
        } else {
            serde_json::from_str(&s).with_context(|| format!("parsing {}", path.display()))?
        };

        let changed = match schema_of(host) {
            Schema::McpServersTopLevel => remove_top_level(&mut root),
            Schema::ZedNested => remove_zed_nested(&mut root),
        };
        if changed {
            let new_text = serde_json::to_string_pretty(&root).context("serialising config")?;
            if dry_run {
                println!(
                    "  -- {} (dry-run)\n{}",
                    host.display(),
                    indent(&new_text, "     ")
                );
            } else {
                atomic_write(&path, &new_text)?;
                println!("  ok {}  removed mnem entry", host.display());
            }
        }
        changed
    } else {
        false
    };

    // 2026-04-26: also remove the marker-bracketed mnem section
    // from the host's project-rules file if present. User content
    // outside the markers stays untouched.
    let prompt_changed = if let Some(pp) = host.system_prompt_path()
        && pp.exists()
    {
        let s = fs::read_to_string(&pp).with_context(|| format!("reading {}", pp.display()))?;
        let new_s = remove_system_prompt(&s);
        if new_s == s {
            false
        } else {
            if dry_run {
                println!(
                    "  -- {} system prompt (dry-run)\n     (would shrink to {} bytes)",
                    host.display(),
                    new_s.len()
                );
            } else if new_s.trim().is_empty() {
                fs::remove_file(&pp).with_context(|| format!("removing empty {}", pp.display()))?;
                println!(
                    "  ok {}  removed mnem system-prompt file ({} now empty)",
                    host.display(),
                    pp.display()
                );
            } else {
                atomic_write(&pp, &new_s)?;
                println!(
                    "  ok {}  removed mnem system-prompt section",
                    host.display()
                );
            }
            true
        }
    } else {
        false
    };

    // G2 (2026-04-25): also clear the hook entry, if the host has a
    // separate hooks path and we previously wrote one there.
    let hooks_changed = if let Some(hp) = host.hooks_path()
        && hp.exists()
    {
        let s = fs::read_to_string(&hp).with_context(|| format!("reading {}", hp.display()))?;
        let mut root: Value = if s.trim().is_empty() {
            Value::Object(Map::new())
        } else {
            serde_json::from_str(&s).with_context(|| format!("parsing {}", hp.display()))?
        };
        let changed = remove_user_prompt_hook(&mut root);
        if changed {
            let new_text = serde_json::to_string_pretty(&root).context("serialising hooks")?;
            if dry_run {
                println!(
                    "  -- {} hooks (dry-run)\n{}",
                    host.display(),
                    indent(&new_text, "     ")
                );
            } else {
                atomic_write(&hp, &new_text)?;
                println!("  ok {}  removed mnem hook entry", host.display());
            }
        }
        changed
    } else {
        false
    };

    if !mcp_changed && !hooks_changed && !prompt_changed {
        println!("  -  {}  no mnem entry", host.display());
    }
    Ok(())
}

fn do_check() -> Result<()> {
    for host in Host::all() {
        let line = match host.config_path() {
            None => format!("  -  {:<18} unsupported on this OS", host.display()),
            Some(path) if !path.exists() => {
                format!("  -  {:<18} not wired ({})", host.display(), path.display())
            }
            Some(path) => {
                let s = fs::read_to_string(&path)?;
                let root: Value = if s.trim().is_empty() {
                    Value::Null
                } else {
                    serde_json::from_str(&s).unwrap_or(Value::Null)
                };
                let wired = match schema_of(*host) {
                    Schema::McpServersTopLevel => has_top_level(&root),
                    Schema::ZedNested => has_zed_nested(&root),
                };
                if wired {
                    format!("  ok {:<18} wired ({})", host.display(), path.display())
                } else {
                    format!(
                        "  -  {:<18} config exists, no mnem entry ({})",
                        host.display(),
                        path.display()
                    )
                }
            }
        };
        println!("{line}");
    }
    Ok(())
}

// ---------- JSON merge helpers ----------

/// Resolve the path we should write into a host's `command` field for
/// the `mnem-mcp` binary.
///
/// Audit fix G9 (2026-04-25): the original implementation always wrote
/// the bare name `"mnem-mcp"`, which silently fails when the binary is
/// not on `PATH` (a common state right after `cargo build` or for
/// users who have not aliased the Cargo target dir). We now look for
/// `mnem-mcp` (or `mnem-mcp.exe` on Windows) next to the current
/// `mnem` executable; if present, write the absolute path. Otherwise
/// fall back to the bare name so users who DID install to PATH still
/// work without surprises.
fn resolve_mnem_mcp_command() -> String {
    if let Ok(here) = std::env::current_exe()
        && let Some(dir) = here.parent()
    {
        let candidate = if cfg!(target_os = "windows") {
            dir.join("mnem-mcp.exe")
        } else {
            dir.join("mnem-mcp")
        };
        if candidate.exists() {
            return candidate.to_string_lossy().into_owned();
        }
    }
    "mnem-mcp".to_string()
}

/// Resolve the path to the `mnem` CLI binary. Same logic as
/// [`resolve_mnem_mcp_command`] but for the main `mnem` driver - used
/// by the pre-prompt hook so the hook can call `mnem retrieve`
/// reliably even when the binary is not on `PATH`. Audit fix G2
/// (2026-04-25).
fn resolve_mnem_command() -> String {
    if let Ok(here) = std::env::current_exe()
        && let Some(dir) = here.parent()
    {
        let candidate = if cfg!(target_os = "windows") {
            dir.join("mnem.exe")
        } else {
            dir.join("mnem")
        };
        if candidate.exists() {
            return candidate.to_string_lossy().into_owned();
        }
    }
    "mnem".to_string()
}

/// Build the shell command for a Claude-Code-style `UserPromptSubmit`
/// hook that runs `mnem retrieve` on the incoming prompt and writes
/// the result to stdout (which the host injects as additional
/// context).
///
/// Claude Code passes a JSON object on the hook's stdin with shape
/// `{ "hook_event_name": "UserPromptSubmit", "prompt": "...", ... }`.
/// We extract the `prompt` field and pipe it to `mnem retrieve`.
///
/// Two flavours so the same `mnem integrate --with-hooks` works on
/// both platforms:
///
/// - **Unix** (Linux, macOS): bash + `jq`. Requires `jq` to be on
///   PATH; `mnem doctor` checks for it.
/// - **Windows**: PowerShell with `ConvertFrom-Json`. No external
///   parser required.
fn pre_prompt_hook_command(mnem_bin: &str) -> String {
    if cfg!(target_os = "windows") {
        format!(
            "powershell -NoProfile -Command \"$j = ($input | Out-String | ConvertFrom-Json); \
             if ($j.prompt) {{ & '{}' retrieve --text $j.prompt --budget 2000 2>$null }}\"",
            mnem_bin.replace('\'', "''").replace('$', "`$")
        )
    } else {
        format!(
            "bash -c 'p=$(jq -r .prompt 2>/dev/null); \
             if [ -n \"$p\" ] && [ \"$p\" != \"null\" ]; then \
             \"{}\" retrieve --text \"$p\" --budget 2000 2>/dev/null; fi'",
            mnem_bin.replace('"', "\\\"")
        )
    }
}

/// JSON value of the `UserPromptSubmit` hook entry mnem writes for
/// hosts that support hooks. Audit fix G2 (2026-04-25).
fn user_prompt_hook_value() -> Value {
    let cmd = pre_prompt_hook_command(&resolve_mnem_command());
    json!({
        "matcher": ".*",
        "hooks": [
            {
                "type": "command",
                "command": cmd
            }
        ]
    })
}

fn mnem_server_value(target: &Path) -> Value {
    json!({
        "command": resolve_mnem_mcp_command(),
        "args": ["--repo", target.to_string_lossy()]
    })
}

fn zed_server_value(target: &Path) -> Value {
    json!({
        "command": {
            "path": resolve_mnem_mcp_command(),
            "args": ["--repo", target.to_string_lossy()]
        }
    })
}

/// Set `root.mcpServers.mnem = v`. Returns true if the file changed.
fn set_top_level(root: &mut Value, target: &Path) -> bool {
    ensure_object(root);
    let obj = root.as_object_mut().expect("ensured above");
    let servers = obj
        .entry("mcpServers")
        .or_insert_with(|| Value::Object(Map::new()));
    if !servers.is_object() {
        *servers = Value::Object(Map::new());
    }
    let servers_map = servers.as_object_mut().expect("object");
    let new_val = mnem_server_value(target);
    let was = servers_map.get("mnem");
    if was == Some(&new_val) {
        return false;
    }
    servers_map.insert("mnem".to_string(), new_val);
    true
}

fn remove_top_level(root: &mut Value) -> bool {
    let Some(obj) = root.as_object_mut() else {
        return false;
    };
    let Some(servers) = obj.get_mut("mcpServers") else {
        return false;
    };
    let Some(map) = servers.as_object_mut() else {
        return false;
    };
    map.remove("mnem").is_some()
}

fn has_top_level(root: &Value) -> bool {
    root.get("mcpServers")
        .and_then(Value::as_object)
        .is_some_and(|m| m.contains_key("mnem"))
}

fn set_zed_nested(root: &mut Value, target: &Path) -> bool {
    ensure_object(root);
    let obj = root.as_object_mut().expect("ensured");
    let exp = obj
        .entry("experimental")
        .or_insert_with(|| Value::Object(Map::new()));
    if !exp.is_object() {
        *exp = Value::Object(Map::new());
    }
    let ctx = exp
        .as_object_mut()
        .expect("object")
        .entry("context_servers")
        .or_insert_with(|| Value::Object(Map::new()));
    if !ctx.is_object() {
        *ctx = Value::Object(Map::new());
    }
    let ctx_map = ctx.as_object_mut().expect("object");
    let new_val = zed_server_value(target);
    if ctx_map.get("mnem") == Some(&new_val) {
        return false;
    }
    ctx_map.insert("mnem".to_string(), new_val);
    true
}

fn remove_zed_nested(root: &mut Value) -> bool {
    let Some(exp) = root.as_object_mut().and_then(|o| o.get_mut("experimental")) else {
        return false;
    };
    let Some(ctx) = exp
        .as_object_mut()
        .and_then(|o| o.get_mut("context_servers"))
    else {
        return false;
    };
    ctx.as_object_mut()
        .is_some_and(|m| m.remove("mnem").is_some())
}

fn has_zed_nested(root: &Value) -> bool {
    root.get("experimental")
        .and_then(|e| e.get("context_servers"))
        .and_then(Value::as_object)
        .is_some_and(|m| m.contains_key("mnem"))
}

fn ensure_object(v: &mut Value) {
    if !v.is_object() {
        *v = Value::Object(Map::new());
    }
}

// ---------- Snippet for --show ----------

fn snippet_for(host: Host, target: &Path) -> String {
    let v = match schema_of(host) {
        Schema::McpServersTopLevel => json!({"mcpServers": {"mnem": mnem_server_value(target)}}),
        Schema::ZedNested => {
            json!({"experimental": {"context_servers": {"mnem": zed_server_value(target)}}})
        }
    };
    serde_json::to_string_pretty(&v).unwrap_or_else(|_| "<encode failure>".into())
}

// ---------- Filesystem + util ----------

fn resolve_target(explicit: Option<&Path>) -> Result<PathBuf> {
    if let Some(p) = explicit {
        return Ok(p.to_path_buf());
    }
    // Reuse the existing resolver for consistency.
    match crate::repo::locate_data_dir(None) {
        Ok(p) => Ok(p),
        Err(_) => {
            // No .mnem in cwd-or-parents. Default to cwd/.mnem.
            let cwd = std::env::current_dir().context("cwd unreadable")?;
            Ok(cwd.join(".mnem"))
        }
    }
}

fn atomic_write(path: &Path, contents: &str) -> Result<()> {
    let parent = path
        .parent()
        .ok_or_else(|| anyhow!("{} has no parent", path.display()))?;
    let tmp = parent.join(format!(
        ".mnem-tmp-{}",
        std::process::id() as u64 ^ now_millis()
    ));
    {
        let mut f =
            fs::File::create(&tmp).with_context(|| format!("creating tmp {}", tmp.display()))?;
        f.write_all(contents.as_bytes())
            .with_context(|| format!("writing tmp {}", tmp.display()))?;
        f.sync_all()
            .with_context(|| format!("fsync {}", tmp.display()))?;
    }
    fs::rename(&tmp, path)
        .with_context(|| format!("renaming {} -> {}", tmp.display(), path.display()))?;
    Ok(())
}

/// Number of milliseconds in one second. Named so the
/// ms-to-seconds conversion in [`timestamp`] reads as a unit
/// change rather than a magic divisor.
const MILLIS_PER_SECOND: u64 = 1_000;

fn timestamp() -> String {
    let now = now_millis() / MILLIS_PER_SECOND;
    // YYYYMMDD-HHMM pieces approximated without a date crate: we only
    // need a monotone-ish, operator-recognisable suffix. Fall back to
    // the raw epoch-seconds if formatting fails.
    format!("{now}")
}

fn now_millis() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

fn indent(s: &str, pad: &str) -> String {
    s.lines()
        .map(|line| format!("{pad}{line}"))
        .collect::<Vec<_>>()
        .join("\n")
}

/// Print a wired snippet to stdout. Public for use by doctor's help.
#[allow(dead_code)]
pub(crate) fn format_snippet(host: Host, target: &Path) -> String {
    snippet_for(host, target)
}

/// Check-routine helper used by `mnem doctor`.
pub(crate) fn wired_status() -> Vec<(Host, Option<PathBuf>, bool)> {
    Host::all()
        .iter()
        .map(|h| {
            let path = h.config_path();
            let wired = path
                .as_ref()
                .and_then(|p| fs::read_to_string(p).ok())
                .is_some_and(|s| {
                    let root: Value = if s.trim().is_empty() {
                        Value::Null
                    } else {
                        serde_json::from_str(&s).unwrap_or(Value::Null)
                    };
                    match schema_of(*h) {
                        Schema::McpServersTopLevel => has_top_level(&root),
                        Schema::ZedNested => has_zed_nested(&root),
                    }
                });
            (*h, path, wired)
        })
        .collect()
}

// ---------- Tests ----------

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn tmp_path() -> PathBuf {
        let id = format!(
            "mnem-integrate-test-{}-{}",
            std::process::id(),
            now_millis()
        );
        std::env::temp_dir().join(id)
    }

    #[test]
    fn host_slugs_are_stable() {
        // Changing these breaks scripts and `mnem integrate --show <slug>`
        // invocations saved in docs. Catch inadvertent renames.
        assert_eq!(Host::ClaudeDesktop.slug(), "claude-desktop");
        assert_eq!(Host::Cursor.slug(), "cursor");
        assert_eq!(Host::Continue_.slug(), "continue");
        assert_eq!(Host::Zed.slug(), "zed");
    }

    #[test]
    fn parse_accepts_aliases() {
        assert_eq!(Host::parse("claude-desktop"), Some(Host::ClaudeDesktop));
        assert_eq!(Host::parse("claude_desktop"), Some(Host::ClaudeDesktop));
        // G7 (2026-04-25): bare "claude" now resolves to Claude Code,
        // not Claude Desktop. The developer-tool surface is the primary
        // customer for mnem; users who specifically want Claude Desktop
        // must use the full slug. Verified separately in
        // `parse_accepts_new_host_aliases`.
        assert_eq!(Host::parse("CURSOR"), Some(Host::Cursor));
        assert_eq!(Host::parse("garbage"), None);
    }

    /// Assert the JSON value parses to a string that names the
    /// `mnem-mcp` binary, accepting either the bare name (when the
    /// resolver could not find a colocated binary) or an absolute path
    /// ending in `mnem-mcp` / `mnem-mcp.exe`. Lets the integrate-test
    /// suite ride along regardless of whether `cargo test` ran after a
    /// `cargo build`.
    fn assert_is_mnem_mcp_command(v: &Value) {
        let s = v
            .as_str()
            .unwrap_or_else(|| panic!("command must be a string; got {v:?}"));
        let ok = s == "mnem-mcp"
            || s.ends_with("mnem-mcp")
            || s.ends_with("mnem-mcp.exe")
            || s.ends_with("mnem-mcp\\")
            || s.ends_with("mnem-mcp.exe\\");
        assert!(
            ok,
            "command must be `mnem-mcp` or absolute path to it; got `{s}`"
        );
    }

    #[test]
    fn set_top_level_into_empty_object() {
        let mut v = json!({});
        let changed = set_top_level(&mut v, Path::new("/r"));
        assert!(changed);
        assert_is_mnem_mcp_command(&v["mcpServers"]["mnem"]["command"]);
    }

    #[test]
    fn set_top_level_preserves_other_servers() {
        let mut v = json!({
            "mcpServers": {"other": {"command": "other-mcp"}}
        });
        set_top_level(&mut v, Path::new("/r"));
        // Both entries survive.
        assert_eq!(v["mcpServers"]["other"]["command"], json!("other-mcp"));
        assert_is_mnem_mcp_command(&v["mcpServers"]["mnem"]["command"]);
    }

    #[test]
    fn set_top_level_idempotent_when_already_wired() {
        let mut v = json!({});
        assert!(set_top_level(&mut v, Path::new("/r")));
        // Second call with same target should report "no change".
        assert!(!set_top_level(&mut v, Path::new("/r")));
    }

    #[test]
    fn set_top_level_overwrites_stale_mnem_entry() {
        let mut v = json!({
            "mcpServers": {"mnem": {"command": "mnem-mcp", "args": ["--repo", "/old"]}}
        });
        let changed = set_top_level(&mut v, Path::new("/new"));
        assert!(changed);
        assert_eq!(v["mcpServers"]["mnem"]["args"][1], json!("/new"));
    }

    #[test]
    fn remove_top_level_is_clean() {
        let mut v = json!({
            "mcpServers": {"mnem": {}, "other": {"command": "x"}}
        });
        assert!(remove_top_level(&mut v));
        assert!(v["mcpServers"]["mnem"].is_null());
        assert_eq!(v["mcpServers"]["other"]["command"], json!("x"));
    }

    #[test]
    fn remove_top_level_when_absent() {
        let mut v = json!({"mcpServers": {"other": {}}});
        assert!(!remove_top_level(&mut v));
    }

    #[test]
    fn zed_nested_round_trip() {
        let mut v = json!({});
        assert!(set_zed_nested(&mut v, Path::new("/r")));
        assert!(has_zed_nested(&v));
        assert!(remove_zed_nested(&mut v));
        assert!(!has_zed_nested(&v));
    }

    #[test]
    fn zed_nested_preserves_other_experimental_keys() {
        let mut v = json!({"experimental": {"feature_x": true}});
        set_zed_nested(&mut v, Path::new("/r"));
        assert_eq!(v["experimental"]["feature_x"], json!(true));
        assert!(has_zed_nested(&v));
    }

    #[test]
    fn non_object_root_is_replaced_cleanly() {
        // A user who somehow has a JSON array or number as the root
        // should not crash us; we silently re-seed to an object.
        let mut v = json!([1, 2, 3]);
        assert!(set_top_level(&mut v, Path::new("/r")));
        assert!(v.is_object());
        assert!(has_top_level(&v));
    }

    #[test]
    fn snippet_for_top_level_is_valid_json() {
        let s = snippet_for(Host::ClaudeDesktop, Path::new("/r"));
        let v: Value = serde_json::from_str(&s).expect("valid json");
        assert_is_mnem_mcp_command(&v["mcpServers"]["mnem"]["command"]);
    }

    #[test]
    fn snippet_for_zed_uses_experimental_context_servers() {
        let s = snippet_for(Host::Zed, Path::new("/r"));
        let v: Value = serde_json::from_str(&s).expect("valid json");
        assert_is_mnem_mcp_command(
            &v["experimental"]["context_servers"]["mnem"]["command"]["path"],
        );
    }

    // ---------- G7 + G9 audit fix tests (2026-04-25) ----------

    #[test]
    fn parse_accepts_new_host_aliases() {
        // G7: Claude Code and Gemini CLI added; their slugs must parse.
        assert_eq!(Host::parse("claude-code"), Some(Host::ClaudeCode));
        assert_eq!(Host::parse("claude_code"), Some(Host::ClaudeCode));
        assert_eq!(Host::parse("CLAUDE-CODE"), Some(Host::ClaudeCode));
        assert_eq!(Host::parse("gemini-cli"), Some(Host::GeminiCli));
        assert_eq!(Host::parse("gemini"), Some(Host::GeminiCli));
        // G7: `claude` now resolves to ClaudeCode (the developer tool),
        // not Claude Desktop. Desktop must be addressed by its full
        // slug; this matches the developer-tool-first orientation of
        // mnem's customer base.
        assert_eq!(Host::parse("claude"), Some(Host::ClaudeCode));
    }

    #[test]
    fn all_hosts_includes_new_entries() {
        let slugs: Vec<_> = Host::all().iter().map(|h| h.slug()).collect();
        assert!(slugs.contains(&"claude-code"));
        assert!(slugs.contains(&"gemini-cli"));
        // Pre-existing slugs survive.
        assert!(slugs.contains(&"claude-desktop"));
        assert!(slugs.contains(&"cursor"));
        assert!(slugs.contains(&"continue"));
        assert!(slugs.contains(&"zed"));
    }

    #[test]
    fn claude_code_uses_top_level_mcp_servers_schema() {
        // ClaudeCode rides the same `mcpServers.<name>` shape as
        // Claude Desktop / Cursor / Continue.
        let mut v = json!({});
        let changed = set_top_level(&mut v, Path::new("/r"));
        assert!(changed);
        assert!(v["mcpServers"]["mnem"].is_object());
    }

    #[test]
    fn claude_code_hooks_path_resolves() {
        // G2: ClaudeCode is the only host with a hooks_path today.
        assert!(Host::ClaudeCode.hooks_path().is_some());
        assert!(Host::Cursor.hooks_path().is_none());
        assert!(Host::ClaudeDesktop.hooks_path().is_none());
        assert!(Host::GeminiCli.hooks_path().is_none());
    }

    #[test]
    fn snippet_for_claude_code_emits_top_level_shape() {
        let s = snippet_for(Host::ClaudeCode, Path::new("/r"));
        let v: Value = serde_json::from_str(&s).expect("valid json");
        assert_is_mnem_mcp_command(&v["mcpServers"]["mnem"]["command"]);
    }

    #[test]
    fn snippet_for_gemini_cli_emits_top_level_shape() {
        let s = snippet_for(Host::GeminiCli, Path::new("/r"));
        let v: Value = serde_json::from_str(&s).expect("valid json");
        assert_is_mnem_mcp_command(&v["mcpServers"]["mnem"]["command"]);
    }

    // ---------- G1/G2/G4 hook + system-prompt tests (2026-04-25) ----------

    #[test]
    fn system_prompt_constant_is_non_empty_and_mentions_mnem_retrieve() {
        // The embedded SYSTEM_PROMPT must include the core instructions
        // for read/write policy. If the docs file disappears or the
        // include_str! path drifts, the build breaks; this test guards
        // against a silent shrinking edit.
        assert!(SYSTEM_PROMPT.contains("mnem_retrieve"));
        assert!(SYSTEM_PROMPT.contains("mnem_resolve_or_create"));
        assert!(SYSTEM_PROMPT.contains("Entity:Person"));
        assert!(
            SYSTEM_PROMPT.len() > 1000,
            "system prompt suspiciously small"
        );
    }

    #[test]
    fn pre_prompt_hook_command_mentions_mnem_retrieve() {
        // Whichever flavour (PowerShell / bash) we emit on the host
        // OS, the literal `mnem retrieve` substring must appear so
        // `entry_is_mnem_hook` can identify it on round-trip.
        let cmd = pre_prompt_hook_command("mnem");
        assert!(
            cmd.contains("retrieve"),
            "hook command must invoke `retrieve`: {cmd}"
        );
        assert!(
            cmd.contains("mnem"),
            "hook command must reference the mnem binary: {cmd}"
        );
        assert!(
            cmd.contains("--budget 2000"),
            "hook command must pass a budget: {cmd}"
        );
    }

    #[test]
    fn user_prompt_hook_value_round_trip_is_idempotent() {
        let mut root = json!({});
        let first = set_user_prompt_hook(&mut root);
        let second = set_user_prompt_hook(&mut root);
        assert!(first, "first set must report a change");
        assert!(!second, "second set with same value must be no-op");
    }

    #[test]
    fn user_prompt_hook_preserves_unrelated_hooks() {
        // A pre-existing UserPromptSubmit entry from another tool
        // must survive when we add ours.
        let mut root = json!({
            "hooks": {
                "UserPromptSubmit": [
                    { "matcher": "/foo", "hooks": [
                        { "type": "command", "command": "echo other" }
                    ] }
                ]
            }
        });
        assert!(set_user_prompt_hook(&mut root));
        let arr = root["hooks"]["UserPromptSubmit"].as_array().unwrap();
        assert_eq!(arr.len(), 2, "expected pre-existing entry + mnem entry");
        // The pre-existing `echo other` entry survives.
        assert!(
            arr.iter().any(|e| e["hooks"][0]["command"] == "echo other"),
            "unrelated hook entry was clobbered"
        );
    }

    #[test]
    fn user_prompt_hook_removal_round_trip() {
        let mut root = json!({});
        assert!(set_user_prompt_hook(&mut root));
        assert!(remove_user_prompt_hook(&mut root));
        // Second remove finds nothing to remove.
        assert!(!remove_user_prompt_hook(&mut root));
    }

    // ---------- 2026-04-26 system-prompt-write tests ----------

    #[test]
    fn merge_system_prompt_into_empty_file_creates_marker_bracketed_block() {
        let out = merge_system_prompt("", "PROMPT BODY");
        assert!(out.contains(SYSTEM_PROMPT_MARKER_START));
        assert!(out.contains(SYSTEM_PROMPT_MARKER_END));
        assert!(out.contains("PROMPT BODY"));
        // No leading whitespace for an empty-input merge.
        assert!(out.starts_with(SYSTEM_PROMPT_MARKER_START));
    }

    #[test]
    fn merge_system_prompt_appends_to_non_marker_existing_content() {
        let existing = "# My project\n\nSome rules I wrote myself.\n";
        let out = merge_system_prompt(existing, "PROMPT BODY");
        // User content survives intact.
        assert!(out.starts_with(existing));
        // Marker block follows.
        assert!(out.contains(SYSTEM_PROMPT_MARKER_START));
        assert!(out.contains("PROMPT BODY"));
        assert!(out.contains(SYSTEM_PROMPT_MARKER_END));
    }

    #[test]
    fn merge_system_prompt_replaces_existing_marker_block_idempotently() {
        let existing = format!(
            "# My project\n\n{SYSTEM_PROMPT_MARKER_START}\nOLD PROMPT\n{SYSTEM_PROMPT_MARKER_END}\n\n## After mnem section\n"
        );
        let out = merge_system_prompt(&existing, "NEW PROMPT");
        assert!(out.contains("NEW PROMPT"));
        assert!(!out.contains("OLD PROMPT"));
        // Content above and below the markers survives.
        assert!(out.starts_with("# My project"));
        assert!(out.contains("## After mnem section"));
        // Re-merging with the same prompt is a true no-op.
        let again = merge_system_prompt(&out, "NEW PROMPT");
        assert_eq!(
            again, out,
            "second merge with same prompt should be a no-op"
        );
    }

    #[test]
    fn remove_system_prompt_strips_only_the_marker_block() {
        let existing = format!(
            "# My project\n\n{SYSTEM_PROMPT_MARKER_START}\nMNEM PROMPT BODY\n{SYSTEM_PROMPT_MARKER_END}\n\n## After mnem section\n"
        );
        let out = remove_system_prompt(&existing);
        assert!(!out.contains("MNEM PROMPT BODY"));
        assert!(!out.contains(SYSTEM_PROMPT_MARKER_START));
        assert!(!out.contains(SYSTEM_PROMPT_MARKER_END));
        assert!(out.contains("# My project"));
        assert!(out.contains("## After mnem section"));
    }

    #[test]
    fn remove_system_prompt_no_op_when_no_markers() {
        let existing = "Just user content.\n";
        let out = remove_system_prompt(existing);
        assert_eq!(out, existing);
    }

    #[test]
    fn host_system_prompt_path_only_set_for_claude_code() {
        assert!(Host::ClaudeCode.system_prompt_path().is_some());
        assert!(Host::ClaudeDesktop.system_prompt_path().is_none());
        assert!(Host::Cursor.system_prompt_path().is_none());
        assert!(Host::Continue_.system_prompt_path().is_none());
        assert!(Host::Zed.system_prompt_path().is_none());
        assert!(Host::GeminiCli.system_prompt_path().is_none());
    }

    #[test]
    fn entry_is_mnem_hook_recognises_round_trip_value() {
        let v = user_prompt_hook_value();
        assert!(entry_is_mnem_hook(&v));
        // Unrelated entries are not recognised.
        let other = json!({
            "matcher": "/foo",
            "hooks": [{ "type": "command", "command": "do_something_else.sh" }]
        });
        assert!(!entry_is_mnem_hook(&other));
    }

    #[test]
    fn resolve_mnem_mcp_command_falls_back_to_bare_name_in_test_env() {
        // In the test runner, `current_exe()` lives in
        // target/debug/deps/, not the same dir as `mnem-mcp`. The
        // resolver must therefore fall back to the bare name. If a
        // future change makes the test binary live next to mnem-mcp,
        // this test will start returning the absolute path; either
        // outcome is correct, so we accept both.
        let cmd = resolve_mnem_mcp_command();
        let ok = cmd == "mnem-mcp" || cmd.ends_with("mnem-mcp") || cmd.ends_with("mnem-mcp.exe");
        assert!(ok, "resolver returned unexpected value: {cmd}");
    }

    #[test]
    fn atomic_write_creates_file_and_replaces() {
        let path = tmp_path();
        atomic_write(&path, "first").unwrap();
        assert_eq!(fs::read_to_string(&path).unwrap(), "first");
        atomic_write(&path, "second").unwrap();
        assert_eq!(fs::read_to_string(&path).unwrap(), "second");
        fs::remove_file(&path).ok();
    }
}
