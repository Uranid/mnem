# Integrations

mnem exposes the same retrieval engine through four interfaces. Pick whichever fits your stack - they all hit the same graph.

## MCP (Model Context Protocol)

The primary agent interface. `mnem integrate` wires the MCP server into your agent host automatically.

```bash
mnem integrate                  # auto-detect and configure all installed hosts
mnem integrate claude-code      # wire a specific MCP host
mnem integrate hermes           # wire Hermes Agent pre/post LLM hooks
```

Supported hosts: Claude Code, Claude Desktop, Cursor, Continue, Zed, Gemini CLI, and Hermes Agent. Any other MCP-aware host works via a manual `mcpServers` entry.

Hermes Agent support is intentionally hook-only: `mnem integrate hermes` writes `$HERMES_HOME/config.yaml` shell hooks plus `$HERMES_HOME/hooks/mnem/hermes-hook.py`. If `$HERMES_HOME` is unset, mnem uses Hermes' default profile directory: `~/.hermes`. `pre_llm_call` retrieves local project memory (walking up to find `.mnem/`) and falls back to `mnem global retrieve`; `post_llm_call` commits the turn back into local mnem or the global graph. This keeps mnem as a +1 memory layer without editing Hermes' system prompt. The YAML writer preserves all config values but does not preserve comments; a timestamped `.bak-*` file is written before changes so annotated configs can be recovered if needed.

The MCP server exposes 22 native tools prefixed `mnem_` - retrieve, commit, ingest, traverse, tombstone, global graph access, and more. Full reference: [`docs/src/mcp.md`](../src/mcp.md).

## CLI

The `mnem` binary covers everything the MCP tools expose, plus administrative commands (branch, merge, diff, export, import, reindex). Good for scripts, CI pipelines, and interactive exploration.

```bash
mnem retrieve "query"
mnem commit --message "session findings"
mnem push
```

Full reference: [`docs/src/cli.md`](../src/cli.md).

## HTTP

A JSON API for services that call mnem directly rather than through an agent host.

```bash
mnem http serve   # starts on loopback by default
```

## Python

`mnem-py` (PyO3 bindings) for reading and writing a graph directly from Python without the CLI binary.

```bash
pip install mnem-py
```

Full API: [`crates/mnem-py/README.md`](../../crates/mnem-py/README.md).

## See also

- [mnem integrate guide](../src/guides/integrate.md)
- [MCP tool reference](../src/mcp.md)
- [CLI reference](../src/cli.md)
