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
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};

// ---------- Integration registry ----------

/// One per integrated host, written to `~/.mnemglobal/integrations.toml`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct IntegrationRecord {
    pub slug: String,
    pub display: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub config_path: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hooks_path: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system_prompt_path: Option<String>,
    /// Epoch-seconds timestamp of the last successful integrate.
    pub integrated_at: u64,
    /// Which components were wired: "mcp", "hooks", "system_prompt".
    #[serde(default)]
    pub components: Vec<String>,
}

#[derive(Debug, Default, Serialize, Deserialize)]
pub(crate) struct IntegrationRegistry {
    #[serde(default)]
    pub hosts: Vec<IntegrationRecord>,
}

impl IntegrationRegistry {
    pub(crate) fn load() -> Self {
        let path = global_dir().join("integrations.toml");
        let text = match std::fs::read_to_string(&path) {
            Ok(t) => t,
            Err(_) => return Self::default(),
        };
        toml::from_str(&text).unwrap_or_default()
    }

    fn save(&self) -> Result<()> {
        let dir = global_dir();
        std::fs::create_dir_all(&dir).with_context(|| format!("creating {}", dir.display()))?;
        let text = toml::to_string_pretty(self).context("serialising integrations.toml")?;
        let path = dir.join("integrations.toml");
        atomic_write(&path, &text)?;
        Ok(())
    }

    /// Upsert a record for `host`.
    pub(crate) fn upsert(&mut self, record: IntegrationRecord) {
        if let Some(existing) = self.hosts.iter_mut().find(|r| r.slug == record.slug) {
            *existing = record;
        } else {
            self.hosts.push(record);
        }
    }

    /// Remove a record for `host` slug. Returns true if anything was removed.
    pub(crate) fn remove(&mut self, slug: &str) -> bool {
        let before = self.hosts.len();
        self.hosts.retain(|r| r.slug != slug);
        self.hosts.len() < before
    }
}

fn global_dir() -> std::path::PathBuf {
    crate::global::default_dir()
}

/// Record a successful integration into `~/.mnemglobal/integrations.toml`.
fn record_integration(host: Host, components: Vec<String>) {
    let mut reg = IntegrationRegistry::load();
    let record = IntegrationRecord {
        slug: host.slug().to_string(),
        display: host.display().to_string(),
        config_path: host.config_path().map(|p| p.display().to_string()),
        hooks_path: host.hooks_path().map(|p| p.display().to_string()),
        system_prompt_path: host.system_prompt_path().map(|p| p.display().to_string()),
        integrated_at: now_millis(),
        components,
    };
    reg.upsert(record);
    if let Err(e) = reg.save() {
        eprintln!("(warning: could not update integrations.toml: {e})");
    }
}

/// Remove a host's record from `~/.mnemglobal/integrations.toml`.
pub(crate) fn deregister_integration(host: Host) {
    let mut reg = IntegrationRegistry::load();
    reg.remove(host.slug());
    if let Err(e) = reg.save() {
        eprintln!("(warning: could not update integrations.toml: {e})");
    }
}

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

    /// Path to the file where the host loads its system-prompt /
    /// project-rules / custom-instructions content from disk.
    ///
    /// - ClaudeCode  → `~/.claude/CLAUDE.md` (markdown, marker injection)
    /// - GeminiCli   → `~/.gemini/GEMINI.md` (markdown, marker injection)
    /// - Cursor      → `~/.cursor/rules/mnem.mdc` (mdc, we own the file)
    /// - Continue    → `~/.continue/config.json` (`systemMessage` JSON field)
    /// - Zed         → settings.json (`assistant.system_prompt` JSON field)
    /// - ClaudeDesktop → `None` (UI-only custom-instructions panel)
    pub(crate) fn system_prompt_path(self) -> Option<PathBuf> {
        let home = dirs::home_dir()?;
        match self {
            Host::ClaudeCode => Some(home.join(".claude").join("CLAUDE.md")),
            Host::GeminiCli => Some(home.join(".gemini").join("GEMINI.md")),
            Host::Cursor => Some(home.join(".cursor").join("rules").join("mnem.mdc")),
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
            _ => None,
        }
    }

    /// How the system prompt is stored for this host.
    pub(crate) fn system_prompt_kind(self) -> SystemPromptKind {
        match self {
            Host::Continue_ => SystemPromptKind::JsonField("systemMessage"),
            Host::Zed => SystemPromptKind::JsonNestedField("assistant", "system_prompt"),
            _ => SystemPromptKind::MarkdownMarker,
        }
    }

    /// The prompt body to write for this host. Claude Code gets the
    /// full doc-style prompt (it also has hooks). All other hosts get
    /// the stronger no-hooks variant that uses MANDATORY language since
    /// there is no automatic pre-prompt enforcement mechanism.
    pub(crate) fn system_prompt_content(self) -> &'static str {
        match self {
            Host::ClaudeCode => SYSTEM_PROMPT,
            Host::Cursor => SYSTEM_PROMPT_CURSOR,
            _ => SYSTEM_PROMPT_NO_HOOKS,
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

/// Describes how a host's system-prompt location should be read/written.
#[derive(Debug, Clone, Copy)]
pub(crate) enum SystemPromptKind {
    /// Inject into a markdown / text file using
    /// `<!-- mnem-system-prompt:v1:start/end -->` markers.
    MarkdownMarker,
    /// Inject into a top-level JSON string field.
    /// e.g. `systemMessage` in Continue's config.json.
    JsonField(&'static str),
    /// Inject into a nested JSON string field (parent → child).
    /// e.g. `assistant.system_prompt` in Zed's settings.json.
    JsonNestedField(&'static str, &'static str),
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
  mnem integrate --all --dry-run       # diff mode; write nothing
  mnem integrate --no-hooks            # skip hook wiring this run
  mnem integrate --no-system-prompt    # skip system-prompt wiring this run
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

    /// Print what would change without writing.
    #[arg(long)]
    pub dry_run: bool,

    /// Repo path to point hosts at.
    /// Defaults to the global graph at `~/.mnemglobal/.mnem` so facts
    /// are accessible across all sessions and directories.
    #[arg(long, value_name = "PATH")]
    pub target_repo: Option<PathBuf>,

    /// Skip writing the UserPromptSubmit hook even for hosts that support it.
    #[arg(long = "no-hooks")]
    pub no_hooks: bool,

    /// Skip writing the mnem system prompt into the host's project-rules file.
    #[arg(long = "no-system-prompt")]
    pub no_system_prompt: bool,
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

**Any host (one command, fully auto-wired):**

```bash
mnem integrate
```

This wires the MCP server entry, the `UserPromptSubmit` hook (for
hosts that support it, e.g. Claude Code), and the system prompt into
the host's project-rules file -- all in one shot. Restart the host.
Done.

Use `--no-hooks` or `--no-system-prompt` to skip individual components.

## The prompt

```
You have access to mnem, a persistent knowledge graph available via MCP tools
prefixed `mnem_`. Your job is to use it transparently: the user should never
need to mention mnem.

## Reading memory (before you answer)

A `UserPromptSubmit` hook has already run: it calls BOTH `mnem retrieve`
(local graph, current project) AND `mnem global retrieve` (global graph)
unconditionally. Its output appears as a system-injected message immediately
before this turn (look for text like `# N item(s)` or `0 item(s)`). Content
from earlier human or assistant turns in this conversation is NOT hook output
— that is conversation history. Do NOT confuse the two.

**Before applying any rule below**: confirm that what you are calling "hook
output" is the injected pre-turn message, not something from an earlier turn.
If uncertain, treat it as conversation history and apply the absent/empty rule.

After identifying the hook output, decide whether to call mnem tools:

- If the hook output is **absent** (no injected message) or **empty** (message
  present but shows "0 item(s)" or zero results): always call
  `mnem_global_retrieve` (NOT `mnem_retrieve`) with a focused query for the
  topic at hand — even if the topic appeared in an earlier turn of this
  conversation. Do NOT rely on conversation history as a substitute; facts may
  have been added or changed.
- If the hook output has results but the **specific fact being asked is absent**
  (results mention a relevant entity but do not include the specific attribute
  the user asked about — e.g. who created something, when it happened): call
  `mnem_global_retrieve` (NOT `mnem_retrieve`) with a focused query before
  answering.
- If the hook output **completely and directly answers the specific question**
  including the exact attribute asked (not just that a related entity exists):
  answer from those results; do not re-call the same query.

## Writing memory (after you answer)

mnem IS your only memory store. Do NOT write markdown memory files,
MEMORY.md indexes, or any other file-based notes alongside it — those
are redundant and will diverge. Commit everything to mnem only.

After each turn, commit any new facts, preferences, events, or entities
the user stated or confirmed. Use these rules:

- **Local first**: default to `mnem_commit`, `mnem_resolve_or_create`, and
  `mnem_commit_relation` (local repo). Use `mnem_global_add` or
  `mnem_global_ingest` only when the user explicitly says "global", or when
  the local repo is unavailable.
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
  resolve-or-creates both nodes and adds the edge in one call. After
  `mnem_commit_relation`, also call `mnem_commit` with a complete standalone
  summary sentence so the relationship is
  findable via text search — entity nodes have no summary and are invisible
  to the retriever otherwise.
- Do NOT commit model output or your own reasoning. Only commit facts the
  user stated or confirmed.

## Node types (`ntype` field)

`ntype` is a free-form string -- use whatever label best describes
the content. There is no fixed vocabulary. Some common examples:

  `Fact`, `Preference`, `Event`, `Goal`, `Task`, `Session`,
  `Entity:Person`, `Entity:Organization`, `Entity:Place`,
  `Document`, `Decision`, `Insight`, `Quote`, `Project`, `Meeting`

Name new types freely when they fit. Prefer colon-namespacing for
sub-types (`Entity:Person`, `Entity:Org`, `Code:Function`, etc.).

## Edge predicates

Use a verb-phrase in snake_case that reads left-to-right like English.
Any descriptive predicate is valid. Common examples:

  `works_at`, `lives_in`, `has_preference`, `traveling_with`,
  `happened_before`, `extracted_from`, `mentions`, `revoked_by`,
  `relates_to`, `depends_on`, `part_of`, `owned_by`

## Forgetting

When the user says "forget X", "stop remembering X", or revokes consent:
1. Call `mnem_retrieve` to find the relevant node UUID (use
   `mnem_global_retrieve` if the fact was stored globally or local
   retrieval finds nothing).
2. Call `mnem_tombstone_node` with the UUID and the user's own phrasing
   as `reason`.
3. Confirm to the user briefly: "Removed."

## Supersession (when the user changes their mind)

When the user updates a previously stated fact (e.g. "actually I moved
to Paris" after "I live in Berlin"):
1. Resolve or create the new fact node.
2. Add a `revoked_by` edge from the old node to the new one.
   `mnem_retrieve` / `mnem_global_retrieve` filter out revoked nodes by
   default, so the old fact stops surfacing without losing the audit trail.

## `agent_id`

Always set `agent_id` to the name of the host you're running in
(`"claude-code"`, `"claude-desktop"`, `"cursor"`, `"gemini-cli"`, etc.)
on `mnem_commit`, `mnem_resolve_or_create`, `mnem_tombstone_node`, and
`mnem_commit_relation` calls. This is what shows up in `mnem log` so
the user can audit which agent wrote what.

## If MCP tools are unavailable (CLI fallback)

If the `mnem_*` MCP tools are not listed as available, fall back to CLI:
- **Read**: `mnem retrieve "query text"` (local first). If that errors
  (no .mnem in the current directory), use `mnem global retrieve "query text"`.
- **Write**: `mnem commit` for the local repo; `mnem global add node -s "..."
  --label <type>` for the global graph. Only use global when the user says
  "global" or no local .mnem exists.

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

The pre-prompt hook (wired automatically by `mnem integrate` for
supported hosts) gives a stronger guarantee: it runs BOTH `mnem retrieve`
(local) AND `mnem global retrieve` (global) before the LLM ever sees
the user's message, so facts from any graph are always surfaced. Pair
the two for the strongest "automatic memory" experience.

## See also

- [`agent-playbook.md`](./guide/agent-playbook.md) - write triggers, shape rules, supersession.
- [`integrate.md`](./guide/integrate.md) - host configuration and `mnem integrate` flags.
- [`mcp.md`](./guide/mcp.md) - the MCP tool reference.
"#;

/// Stronger, behavioral-only system prompt for hosts without a
/// `UserPromptSubmit` hook. Uses MANDATORY language because there is
/// no process-level enforcement — the LLM is the only retrieval trigger.
const SYSTEM_PROMPT_NO_HOOKS: &str = r#"You have access to mnem, a persistent knowledge graph via MCP tools
prefixed `mnem_`. Use it transparently on every turn — the user never
needs to mention mnem.

## MANDATORY: Read memory before every answer

Call `mnem_retrieve` at the START of EVERY response without exception.
- `text` = the user's message, `token_budget` = 2000
- Never skip: short questions, greetings, code tasks — none are exempt.
- This host has no automatic pre-prompt hook. YOU are the only retrieval trigger.
- If `mnem_retrieve` errors or the server has no local repo configured, also
  call `mnem_global_retrieve` as fallback.
- Only call `mnem_global_retrieve` explicitly when the user says "global" or
  local retrieval is unavailable.
- If results surface, weave them into your answer naturally; never announce it.

## MANDATORY: Write memory after every turn with new facts

mnem IS your only memory store. Do NOT write markdown memory files,
MEMORY.md indexes, or any other file-based notes — use mnem only.

After EVERY turn where the user stated, corrected, or confirmed any fact,
preference, event, entity, or goal — commit it SILENTLY without asking first.

- **Local first**: use `mnem_commit`, `mnem_resolve_or_create`, and
  `mnem_commit_relation` for all writes by default. Use `mnem_global_add` or
  `mnem_global_ingest` only when the user explicitly says "global" or local
  is unavailable.
- One fact per node. Never merge two separate facts.
- `summary` must be a complete standalone sentence (no leading pronouns).
- Named people / places / orgs: always use `mnem_resolve_or_create`, not `mnem_commit`.
- Connect entities with typed edges: `works_at`, `lives_in`, `has_preference`,
  `extracted_from`, `revoked_by`, `traveling_with`, `happened_before`, `mentions`.
- Use `mnem_commit_relation` when both endpoints are named entities. Then also call
  `mnem_commit` with a standalone summary sentence —
  entity nodes have no summary and are invisible to text search otherwise.
- Only commit facts the user stated or confirmed — never your reasoning or drafts.
- Set `agent_id` to this host's slug on every write (`"cursor"`, `"gemini-cli"`, etc.).

## Node types (`ntype`)

`ntype` is a free-form string — pick whatever label fits. Common examples:
`Fact`, `Preference`, `Event`, `Goal`, `Task`, `Session`,
`Entity:Person`, `Entity:Organization`, `Entity:Place`,
`Decision`, `Insight`, `Project`, `Meeting`.
Name new types freely; prefer colon-namespacing for sub-types.

## Edge predicates

Use verb-phrase snake_case that reads left-to-right. Any descriptive
predicate is valid. Common: `works_at`, `lives_in`, `has_preference`,
`extracted_from`, `revoked_by`, `relates_to`, `depends_on`, `part_of`.

## Forgetting

User says "forget X": `mnem_retrieve` to find the node (fall back to
`mnem_global_retrieve` if not found locally) → `mnem_tombstone_node`
with their wording as `reason`. Reply: "Removed."

## Supersession

User updates a fact: resolve-or-create the new node, then add a `revoked_by`
edge from the old node to the new. The old fact stops surfacing automatically.

## CLI fallback (if MCP tools are unavailable)

- **Read**: `mnem retrieve "query text"` (local first). If that errors
  (no .mnem in the current directory), use `mnem global retrieve "query text"`.
- **Write**: `mnem commit` for local; `mnem global add node -s "..." --label <type>`
  for global. Only use global when the user says "global" or no local .mnem exists.

## When NOT to commit

Pure computation results, tool-call traces, generated drafts or code the user
has not accepted, re-reads within the same turn."#;

/// Cursor-specific variant: identical instructions but prefixed with MDC
/// frontmatter so Cursor applies the rule globally (`alwaysApply: true`)
/// rather than matching on file type. Written to `~/.cursor/rules/mnem.mdc`.
const SYSTEM_PROMPT_CURSOR: &str = r#"---
description: mnem persistent memory graph — automatic knowledge retrieval and storage
alwaysApply: true
---

You have access to mnem, a persistent knowledge graph via MCP tools
prefixed `mnem_`. Use it transparently on every turn — the user never
needs to mention mnem.

## MANDATORY: Read memory before every answer

Call `mnem_retrieve` at the START of EVERY response without exception.
- `text` = the user's message, `token_budget` = 2000
- Never skip: short questions, greetings, code tasks — none are exempt.
- This host has no automatic pre-prompt hook. YOU are the only retrieval trigger.
- If `mnem_retrieve` errors or the server has no local repo configured, also
  call `mnem_global_retrieve` as fallback.
- Only call `mnem_global_retrieve` explicitly when the user says "global" or
  local retrieval is unavailable.
- If results surface, weave them into your answer naturally; never announce it.

## MANDATORY: Write memory after every turn with new facts

mnem IS your only memory store. Do NOT write markdown memory files,
MEMORY.md indexes, or any other file-based notes — use mnem only.

After EVERY turn where the user stated, corrected, or confirmed any fact,
preference, event, entity, or goal — commit it SILENTLY without asking first.

- **Local first**: use `mnem_commit`, `mnem_resolve_or_create`, and
  `mnem_commit_relation` for all writes by default. Use `mnem_global_add` or
  `mnem_global_ingest` only when the user explicitly says "global" or local
  is unavailable.
- One fact per node. Never merge two separate facts.
- `summary` must be a complete standalone sentence (no leading pronouns).
- Named people / places / orgs: always use `mnem_resolve_or_create`, not `mnem_commit`.
- Connect entities with typed edges: `works_at`, `lives_in`, `has_preference`,
  `extracted_from`, `revoked_by`, `traveling_with`, `happened_before`, `mentions`.
- Use `mnem_commit_relation` when both endpoints are named entities. Then also call
  `mnem_commit` with a standalone summary sentence —
  entity nodes have no summary and are invisible to text search otherwise.
- Only commit facts the user stated or confirmed — never your reasoning or drafts.
- Set `agent_id` to `"cursor"` on every write call.

## Node types (`ntype`)

`ntype` is a free-form string — pick whatever label fits. Common examples:
`Fact`, `Preference`, `Event`, `Goal`, `Task`, `Session`,
`Entity:Person`, `Entity:Organization`, `Entity:Place`,
`Decision`, `Insight`, `Project`, `Meeting`.
Name new types freely; prefer colon-namespacing for sub-types.

## Edge predicates

Use verb-phrase snake_case that reads left-to-right. Any descriptive
predicate is valid. Common: `works_at`, `lives_in`, `has_preference`,
`extracted_from`, `revoked_by`, `relates_to`, `depends_on`, `part_of`.

## Forgetting

User says "forget X": `mnem_retrieve` to find the node (fall back to
`mnem_global_retrieve` if not found locally) → `mnem_tombstone_node`
with their wording as `reason`. Reply: "Removed."

## Supersession

User updates a fact: resolve-or-create the new node, then add a `revoked_by`
edge from the old node to the new. The old fact stops surfacing automatically.

## CLI fallback (if MCP tools are unavailable)

- **Read**: `mnem retrieve "query text"` (local first). If that errors
  (no .mnem in the current directory), use `mnem global retrieve "query text"`.
- **Write**: `mnem commit` for local; `mnem global add node -s "..." --label <type>`
  for global. Only use global when the user says "global" or no local .mnem exists.

## When NOT to commit

Pure computation results, tool-call traces, generated drafts or code the user
has not accepted, re-reads within the same turn."#;

/// Text markers used when injecting into a JSON string field
/// (Continue `systemMessage`, Zed `assistant.system_prompt`).
/// Plain-text delimiters that survive JSON serialisation unescaped.
const JSON_MARKER_START: &str = "[mnem-prompt:start]";
const JSON_MARKER_END: &str = "[mnem-prompt:end]";

pub(crate) fn run(args: Args) -> Result<()> {
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

    // Set up ~/.mnemglobal after host selection. Interactive mode shows
    // TUI prompts with defaults in parens. --all / named-host mode
    // bootstraps silently with defaults (idempotent if already set up).
    let interactive_global = !args.all && args.hosts.is_empty();
    if !args.dry_run {
        setup_global(interactive_global)?;
    }

    let stamp = timestamp();
    println!("Writing configs (backing up with .bak-{stamp}):");
    for host in selected {
        let mut components: Vec<String> = Vec::new();
        let mut any_ok = false;

        match do_wire(host, &target, &stamp, args.dry_run) {
            Ok(WireOutcome::Wrote) => {
                println!("  ok {}  wired -> {}", host.display(), target.display());
                components.push("mcp".to_string());
                any_ok = true;
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
                components.push("mcp".to_string());
                any_ok = true;
            }
            Err(e) => {
                println!("  !  {}  {e}", host.display());
            }
        }
        // Wire the UserPromptSubmit hook for hosts that support it
        // unless --no-hooks was passed. Today that's Claude Code only;
        // other hosts skip silently.
        if !args.no_hooks && host.hooks_path().is_some() {
            match do_wire_hooks(host, &stamp, args.dry_run) {
                Ok(WireOutcome::Wrote) => {
                    println!("  ok {}  hooks wired", host.display());
                    components.push("hooks".to_string());
                    any_ok = true;
                }
                Ok(WireOutcome::DryRun(diff)) => {
                    println!("  -- {} hooks (dry-run)\n{diff}", host.display());
                }
                Ok(WireOutcome::AlreadyWired) => {
                    println!("  =  {}  hooks already wired", host.display());
                    components.push("hooks".to_string());
                    any_ok = true;
                }
                Err(e) => {
                    println!("  !  {}  hooks: {e}", host.display());
                }
            }
        }
        // Write the mnem system prompt into the host's project-rules
        // file unless --no-system-prompt was passed. Hosts without a
        // file-based rules location (e.g. Claude Desktop) skip silently.
        if !args.no_system_prompt && host.system_prompt_path().is_some() {
            match do_wire_system_prompt(host, &stamp, args.dry_run) {
                Ok(WireOutcome::Wrote) => {
                    println!("  ok {}  system prompt wired", host.display());
                    components.push("system_prompt".to_string());
                    any_ok = true;
                }
                Ok(WireOutcome::DryRun(diff)) => {
                    println!("  -- {} system prompt (dry-run)\n{diff}", host.display());
                }
                Ok(WireOutcome::AlreadyWired) => {
                    println!("  =  {}  system prompt already wired", host.display());
                    components.push("system_prompt".to_string());
                    any_ok = true;
                }
                Err(e) => {
                    println!("  !  {}  system prompt: {e}", host.display());
                }
            }
        }
        // Persist integration state so `mnem unintegrate` can find it.
        if any_ok && !args.dry_run {
            record_integration(host, components);
        }
    }

    println!();
    println!("Next steps:");
    println!("  1. Restart each agent host you wired.");
    println!("  2. Verify:  mnem doctor");
    println!("  3. To remove:  mnem unintegrate");
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

/// Write or update the mnem system prompt into the host's rules file.
/// Dispatches to markdown-marker or JSON-field injection based on
/// `host.system_prompt_kind()`. Idempotent: re-running replaces only
/// the mnem-managed block, never touching user content outside it.
fn do_wire_system_prompt(host: Host, stamp: &str, dry_run: bool) -> Result<WireOutcome> {
    let Some(path) = host.system_prompt_path() else {
        return Ok(WireOutcome::AlreadyWired);
    };
    match host.system_prompt_kind() {
        SystemPromptKind::MarkdownMarker => do_wire_sp_markdown(host, &path, stamp, dry_run),
        SystemPromptKind::JsonField(field) => {
            do_wire_sp_json(host, &path, &[field], stamp, dry_run)
        }
        SystemPromptKind::JsonNestedField(parent, child) => {
            do_wire_sp_json(host, &path, &[parent, child], stamp, dry_run)
        }
    }
}

/// Marker-injection path (ClaudeCode, GeminiCli, Cursor).
fn do_wire_sp_markdown(host: Host, path: &Path, stamp: &str, dry_run: bool) -> Result<WireOutcome> {
    let existing = if path.exists() {
        fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?
    } else {
        String::new()
    };

    let new_content = merge_system_prompt(&existing, host.system_prompt_content());
    if new_content == existing {
        return Ok(WireOutcome::AlreadyWired);
    }

    if dry_run {
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
        fs::copy(path, &bak).with_context(|| format!("backing up to {}", bak.display()))?;
    }
    atomic_write(path, &new_content)?;
    Ok(WireOutcome::Wrote)
}

/// JSON-field injection path (Continue `systemMessage`, Zed `assistant.system_prompt`).
/// Reads the JSON config, injects the prompt into the target string field using
/// `[mnem-prompt:start/end]` text markers, and writes back. Never touches
/// other fields.
fn do_wire_sp_json(
    host: Host,
    path: &Path,
    field_path: &[&str],
    stamp: &str,
    dry_run: bool,
) -> Result<WireOutcome> {
    let existing_text = if path.exists() {
        fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?
    } else {
        String::new()
    };

    let mut root: Value = if existing_text.trim().is_empty() {
        Value::Object(Map::new())
    } else {
        serde_json::from_str(&existing_text)
            .with_context(|| format!("parsing {}", path.display()))?
    };

    let current_str = json_get_str(&root, field_path)
        .unwrap_or_default()
        .to_string();
    let new_str = merge_json_prompt(&current_str, host.system_prompt_content());

    if new_str == current_str {
        return Ok(WireOutcome::AlreadyWired);
    }

    if dry_run {
        return Ok(WireOutcome::DryRun(format!(
            "     (writing mnem prompt block into {}.{} - {} bytes)",
            path.display(),
            field_path.join("."),
            new_str.len()
        )));
    }

    json_set_str(&mut root, field_path, new_str);
    let new_text = serde_json::to_string_pretty(&root).context("serialising config")?;

    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).with_context(|| format!("creating {}", parent.display()))?;
    }
    if path.exists() {
        let bak = path.with_extension(format!(
            "{}.bak-{stamp}",
            path.extension().and_then(|s| s.to_str()).unwrap_or("json")
        ));
        fs::copy(path, &bak).with_context(|| format!("backing up to {}", bak.display()))?;
    }
    atomic_write(path, &new_text)?;
    Ok(WireOutcome::Wrote)
}

// ---------- JSON field helpers ----------

/// Get a nested string field value, e.g. `["assistant","system_prompt"]`.
fn json_get_str<'a>(root: &'a Value, path: &[&str]) -> Option<&'a str> {
    let mut cur = root;
    for key in path {
        cur = cur.get(key)?;
    }
    cur.as_str()
}

/// Set a nested string field, creating intermediate objects as needed.
fn json_set_str(root: &mut Value, path: &[&str], val: String) {
    if path.is_empty() {
        return;
    }
    if path.len() == 1 {
        if let Value::Object(m) = root {
            m.insert(path[0].to_string(), Value::String(val));
        }
        return;
    }
    if let Value::Object(m) = root {
        let entry = m
            .entry(path[0].to_string())
            .or_insert_with(|| Value::Object(Map::new()));
        json_set_str(entry, &path[1..], val);
    }
}

/// Merge the mnem prompt block into a JSON string field value using
/// `[mnem-prompt:start/end]` text markers. Appends if no markers yet;
/// replaces between markers on re-runs.
fn merge_json_prompt(existing: &str, prompt: &str) -> String {
    let block = format!(
        "\n{}\n{}\n{}",
        JSON_MARKER_START,
        prompt.trim_end(),
        JSON_MARKER_END
    );

    if let (Some(start), Some(end_start)) = (
        existing.find(JSON_MARKER_START),
        existing.find(JSON_MARKER_END),
    ) {
        if end_start > start {
            let tail = end_start + JSON_MARKER_END.len();
            return format!("{}{}{}", &existing[..start], &block[1..], &existing[tail..]);
        }
    }

    format!("{}{}", existing, block)
}

/// Remove the `[mnem-prompt:start/end]` block from a JSON string field value.
fn remove_json_prompt(existing: &str) -> String {
    if let (Some(start), Some(end_start)) = (
        existing.find(JSON_MARKER_START),
        existing.find(JSON_MARKER_END),
    ) {
        if end_start > start {
            let tail = end_start + JSON_MARKER_END.len();
            let head = existing[..start].trim_end_matches('\n');
            let rest = &existing[tail..];
            if rest.is_empty() {
                return if head.is_empty() {
                    String::new()
                } else {
                    format!("{head}\n")
                };
            }
            return format!("{head}\n{rest}");
        }
    }
    existing.to_string()
}

/// Remove the mnem prompt block from a JSON config's string field
/// and write the result back. Used by `do_undo` for Continue / Zed.
fn undo_json_prompt(host: Host, path: &Path, field_path: &[&str], dry_run: bool) -> Result<bool> {
    let text = fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    let mut root: Value = if text.trim().is_empty() {
        Value::Object(Map::new())
    } else {
        serde_json::from_str(&text).with_context(|| format!("parsing {}", path.display()))?
    };

    let current = json_get_str(&root, field_path)
        .unwrap_or_default()
        .to_string();
    let stripped = remove_json_prompt(&current);
    if stripped == current {
        return Ok(false);
    }

    if dry_run {
        println!(
            "  -- {} system prompt (dry-run)\n     (would remove mnem block from {}.{})",
            host.display(),
            path.display(),
            field_path.join(".")
        );
        return Ok(true);
    }

    if stripped.trim().is_empty() {
        // Remove the field entirely rather than leaving an empty string.
        if let Value::Object(m) = &mut root {
            if field_path.len() == 1 {
                m.remove(field_path[0]);
            } else if field_path.len() == 2 {
                if let Some(Value::Object(inner)) = m.get_mut(field_path[0]) {
                    inner.remove(field_path[1]);
                }
            }
        }
    } else {
        json_set_str(&mut root, field_path, stripped);
    }

    let new_text = serde_json::to_string_pretty(&root).context("serialising config")?;
    atomic_write(path, &new_text)?;
    println!("  ok {}  removed mnem system-prompt block", host.display());
    Ok(true)
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

    // Windows: write the companion .ps1 script that the hook command references.
    // Must happen before settings.json so the script exists when the hook fires.
    #[cfg(target_os = "windows")]
    {
        let script_path = windows_hook_script_path();
        if let Some(parent) = script_path.parent() {
            fs::create_dir_all(parent).with_context(|| format!("creating {}", parent.display()))?;
        }
        let content = windows_hook_script_content(&resolve_mnem_command());
        atomic_write(&script_path, &content)?;
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
                    stripped.contains("mnem retrieve")
                        || stripped.contains("mnem.exe retrieve")
                        || stripped.contains("mnem-hook.ps1")
                })
            })
        })
}

pub(crate) fn do_undo(host: Host, dry_run: bool) -> Result<()> {
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

    // Remove the mnem system-prompt block from the host's rules file.
    // Dispatches on system_prompt_kind(): markdown files use HTML
    // comment markers; JSON config files use text markers inside a field.
    let prompt_changed = if let Some(pp) = host.system_prompt_path()
        && pp.exists()
    {
        match host.system_prompt_kind() {
            SystemPromptKind::MarkdownMarker => {
                let s =
                    fs::read_to_string(&pp).with_context(|| format!("reading {}", pp.display()))?;
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
                        fs::remove_file(&pp)
                            .with_context(|| format!("removing empty {}", pp.display()))?;
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
            }
            SystemPromptKind::JsonField(field) => undo_json_prompt(host, &pp, &[field], dry_run)?,
            SystemPromptKind::JsonNestedField(parent, child) => {
                undo_json_prompt(host, &pp, &[parent, child], dry_run)?
            }
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

    // On Windows, delete the generated PowerShell script file if present.
    #[cfg(target_os = "windows")]
    if host == Host::ClaudeCode {
        let script = windows_hook_script_path();
        if script.exists() {
            if dry_run {
                println!("  -- {}  would delete {}", host.display(), script.display());
            } else {
                let _ = fs::remove_file(&script);
                println!("  ok {}  deleted {}", host.display(), script.display());
            }
        }
    }

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
/// the unified `mnem` binary (which exposes `mnem mcp serve` as a
/// subcommand).
///
/// After the v0.2.0 merge, the MCP server lives at `mnem mcp serve`
/// inside the main binary. We look for `mnem` (or `mnem.exe` on
/// Windows) next to the current executable; if present, write the
/// absolute path. Otherwise fall back to the bare name so users who
/// DID install to PATH still work without surprises.
fn resolve_mnem_mcp_command() -> String {
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
/// Path of the generated PowerShell hook script (Windows only).
/// Lives next to `settings.json` so unintegrate can find and delete it.
#[cfg(target_os = "windows")]
fn windows_hook_script_path() -> PathBuf {
    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".claude")
        .join("mnem-hook.ps1")
}

/// Content of the generated PowerShell hook script.
#[cfg(target_os = "windows")]
fn windows_hook_script_content(mnem_bin: &str) -> String {
    // Single-quoted path in PowerShell: escape embedded single quotes as ''.
    let safe_bin = mnem_bin.replace('\'', "''");
    format!(
        "# mnem UserPromptSubmit hook - auto-generated by `mnem integrate`\n\
         $json = $input | Out-String | ConvertFrom-Json\n\
         if ($json.prompt) {{\n\
         \x20\x20& '{safe_bin}' retrieve $json.prompt 2>$null\n\
         \x20\x20& '{safe_bin}' global retrieve $json.prompt 2>$null\n\
         }}\n"
    )
}

/// Build the shell command that Claude Code writes into the hook entry.
///
/// - **Windows**: references a generated `.ps1` file so that Claude Code's
///   bash layer does not expand `$json` / `$input` before PowerShell sees them.
///   The caller (`do_wire_hooks`) is responsible for writing the script file.
/// - **Unix**: inline bash + `jq`.
fn pre_prompt_hook_command(_mnem_bin: &str) -> String {
    #[cfg(target_os = "windows")]
    {
        // Escape backslashes in the path for the JSON/shell layer.
        let script = windows_hook_script_path();
        format!(
            "powershell -NoProfile -ExecutionPolicy Bypass -File \"{}\"",
            script.display()
        )
    }
    #[cfg(not(target_os = "windows"))]
    {
        format!(
            "bash -c 'p=$(jq -r .prompt 2>/dev/null); \
             if [ -n \"$p\" ] && [ \"$p\" != \"null\" ]; then \
             \"{}\" retrieve \"$p\" 2>/dev/null; \
             \"{}\" global retrieve \"$p\" 2>/dev/null; fi'",
            _mnem_bin.replace('"', "\\\""),
            _mnem_bin.replace('"', "\\\"")
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
        "args": ["mcp", "--repo", target.to_string_lossy()]
    })
}

fn zed_server_value(target: &Path) -> Value {
    json!({
        "command": {
            "path": resolve_mnem_mcp_command(),
            "args": ["mcp", "--repo", target.to_string_lossy()]
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
    // Default to the global graph so MCP tool writes are accessible
    // across all sessions and directories, not siloed per-project.
    Ok(crate::global::default_dir().join(".mnem"))
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
    /// `mnem` binary (which exposes `mnem mcp serve` as a subcommand).
    /// Accepts the bare name or an absolute path ending in `mnem` /
    /// `mnem.exe`. Lets the integrate-test suite ride along regardless
    /// of whether `cargo test` ran after a `cargo build`.
    fn assert_is_mnem_mcp_command(v: &Value) {
        let s = v
            .as_str()
            .unwrap_or_else(|| panic!("command must be a string; got {v:?}"));
        let ok = s == "mnem"
            || s.ends_with("/mnem")
            || s.ends_with("\\mnem")
            || s.ends_with("/mnem.exe")
            || s.ends_with("\\mnem.exe");
        assert!(
            ok,
            "command must be `mnem` or absolute path to it; got `{s}`"
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
        // Args are ["mcp", "--repo", "<path>"].
        let mut v = json!({
            "mcpServers": {"mnem": {"command": "mnem", "args": ["mcp", "--repo", "/old"]}}
        });
        let changed = set_top_level(&mut v, Path::new("/new"));
        assert!(changed);
        assert_eq!(v["mcpServers"]["mnem"]["args"][2], json!("/new"));
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
        let cmd = pre_prompt_hook_command("mnem");
        // On Windows the command just points to the .ps1 file; check
        // the script content instead.
        #[cfg(target_os = "windows")]
        {
            assert!(
                cmd.contains("mnem-hook.ps1"),
                "Windows hook must reference the .ps1 script: {cmd}"
            );
            let script = windows_hook_script_content("mnem");
            assert!(
                script.contains("global retrieve"),
                "PS1 must call global retrieve: {script}"
            );
            assert!(
                script.contains("mnem"),
                "PS1 must reference the binary: {script}"
            );
        }
        #[cfg(not(target_os = "windows"))]
        {
            assert!(
                cmd.contains("global retrieve"),
                "hook must call global retrieve: {cmd}"
            );
            assert!(
                cmd.contains("mnem"),
                "hook must reference the binary: {cmd}"
            );
        }
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
    fn host_system_prompt_path_coverage() {
        // Hosts with file-based rules locations return Some.
        assert!(Host::ClaudeCode.system_prompt_path().is_some());
        assert!(Host::GeminiCli.system_prompt_path().is_some());
        assert!(Host::Cursor.system_prompt_path().is_some());
        assert!(Host::Continue_.system_prompt_path().is_some());
        assert!(Host::Zed.system_prompt_path().is_some());
        // Claude Desktop has UI-only custom instructions - no file path.
        assert!(Host::ClaudeDesktop.system_prompt_path().is_none());
        // Cursor gets a dedicated .mdc file, not the shared config.
        let cursor_path = Host::Cursor.system_prompt_path().unwrap();
        assert!(cursor_path.to_string_lossy().contains("mnem.mdc"));
        // Gemini CLI gets GEMINI.md (analog to CLAUDE.md).
        let gemini_path = Host::GeminiCli.system_prompt_path().unwrap();
        assert!(gemini_path.to_string_lossy().contains("GEMINI.md"));
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
        // target/debug/deps/, not the same dir as `mnem`. The
        // resolver must therefore fall back to the bare name. If a
        // future change makes the test binary live next to mnem,
        // this test will start returning the absolute path; either
        // outcome is correct, so we accept both.
        let cmd = resolve_mnem_mcp_command();
        let ok = cmd == "mnem"
            || cmd.ends_with("/mnem")
            || cmd.ends_with("\\mnem")
            || cmd.ends_with("/mnem.exe")
            || cmd.ends_with("\\mnem.exe");
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

// ---------- Global graph setup ----------

/// Set up `~/.mnemglobal/` during `mnem integrate`.
///
/// `interactive = true`  → TUI prompts with defaults shown in parens;
///                          user can press Enter to accept every default.
/// `interactive = false` → silent bootstrap with defaults (--all / named hosts).
fn setup_global(interactive: bool) -> Result<()> {
    use dialoguer::{Input, Select, theme::ColorfulTheme};

    let default_dir = crate::global::default_dir();

    // Step 1: ask where the global graph should live.
    let global_dir: PathBuf = if interactive {
        println!("\nmnem global graph");
        println!("─────────────────");
        let raw: String = Input::with_theme(&ColorfulTheme::default())
            .with_prompt(format!(
                "Global mnem graph location ({})",
                default_dir.display()
            ))
            .allow_empty(true)
            .interact_text()
            .context("global dir prompt failed")?;
        let trimmed = raw.trim();
        if trimmed.is_empty() {
            default_dir
        } else {
            PathBuf::from(trimmed)
        }
    } else {
        default_dir
    };

    // Step 2: bootstrap (create dir + init .mnem if not already present).
    let fresh = crate::global::bootstrap(&global_dir)
        .with_context(|| format!("bootstrapping {}", global_dir.display()))?;

    if fresh || interactive {
        println!(
            "  ok global graph  {}",
            global_dir.join(crate::repo::MNEM_DIR).display()
        );
    }

    // Step 3: pin a default knowledge graph for bare `mnem` commands (no -R).
    let mut reg = crate::global::RepoRegistry::load(&global_dir)?;

    // Skip the prompt if a default is already pinned and we're not interactive.
    if reg.repos.iter().any(|e| e.default) && !interactive {
        return Ok(());
    }

    let cwd = std::env::current_dir().unwrap_or_else(|_| global_dir.clone());
    let mut choices: Vec<(String, PathBuf)> = vec![(
        format!(
            "{}  (global graph - accessible from every project and session)",
            global_dir.display()
        ),
        global_dir.clone(),
    )];
    if cwd != global_dir {
        choices.push((
            format!(
                "{}  (this project - pinned as fallback for bare mnem commands)",
                cwd.display()
            ),
            cwd,
        ));
    }

    let default_repo = if interactive {
        println!("\nDefault knowledge graph for agent queries");
        println!("─────────────────────────────────────────");
        println!("The agent hook queries your project .mnem/ first (walking up from");
        println!("your current directory), then falls back to the global graph");
        println!("automatically. The hook and system prompt behave the same either way.");
        println!("This setting controls which graph bare `mnem` commands fall back to");
        println!("when no project .mnem/ is found. Override any command with -R <path>.\n");
        let items: Vec<&str> = choices.iter().map(|(s, _)| s.as_str()).collect();
        let idx = Select::with_theme(&ColorfulTheme::default())
            .with_prompt("Default knowledge graph")
            .items(&items)
            .default(0)
            .interact()
            .context("default knowledge graph prompt failed")?;
        choices.remove(idx).1
    } else {
        global_dir.clone()
    };

    reg.register(&default_repo, true);
    reg.save(&global_dir).with_context(|| {
        format!(
            "saving {}",
            crate::global::registry_path(&global_dir).display()
        )
    })?;

    if interactive || fresh {
        println!("  ok default graph  {}", default_repo.display());
    }

    Ok(())
}
