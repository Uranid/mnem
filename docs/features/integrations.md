# Integrations

mnem exposes the same retrieval engine through four interfaces. Pick whichever fits your stack - they all hit the same graph.

## MCP (Model Context Protocol)

The primary agent interface. `mnem integrate` wires the MCP server into your agent host automatically.

```bash
mnem integrate                  # auto-detect and configure all installed hosts
mnem integrate claude-code      # wire a specific host
```

Supported hosts: Claude Code, Claude Desktop, Cursor, Continue, Zed, Hermes Agent, Gemini CLI. Any other MCP-aware host works via a manual `mcpServers` entry (or `mcp_servers` for Hermes-style YAML clients).

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
