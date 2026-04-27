# mnem-mcp

Model Context Protocol server for mnem - the AI-native, local-first memory substrate for agents.

Ships the `mnem-mcp` binary, which speaks MCP JSON-RPC 2.0 over stdio. Any
MCP-aware host (Claude Desktop, Cursor, Windsurf, Claude Code, custom
clients) can point at it and mnem's tools show up: `mnem_stats`,
`mnem_schema`, `mnem_search`, `mnem_vector_search`, `mnem_retrieve`,
`mnem_get_node`, `mnem_traverse`, `mnem_commit`, `mnem_resolve_or_create`,
`mnem_recent`. Every response carries `_meta` with `bytes`,
`latency_micros`, and `tokens_estimate` so the caller can reason about the
cost of its own calls; writes propagate `agent_id` and `task_id` into commit
and operation metadata so provenance stays queryable.

```bash
mnem-mcp --repo ./my-mnem-repo
# or
MNEM_REPO=./my-mnem-repo mnem-mcp
```

Workspace top: [`../../README.md`](../../README.md). MCP guide:
[`../../docs/guide/mcp.md`](../../docs/guide/mcp.md).

Licensed under Apache-2.0.
