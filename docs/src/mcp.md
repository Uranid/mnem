# MCP server

mnem implements the [Model Context Protocol](https://modelcontextprotocol.io)
over stdio. Drop it into any MCP client (Claude Desktop, Cursor, Zed, custom).

## Install

```bash
mnem mcp install         # writes config entry to your client's MCP config file
```

The command auto-detects Claude Desktop / Cursor / Zed. For other clients,
register manually:

```json
{
  "mcpServers": {
    "mnem": {
      "command": "mnem",
      "args": ["mcp", "serve", "--repo", "/path/to/your-graph"]
    }
  }
}
```

## Tools exposed

| Tool | Purpose |
|------|---------|
| `mnem_retrieve` | hybrid retrieval over the repo |
| `mnem_ingest` | add a node (text + label + metadata) |
| `mnem_stats` | repo size, commit count, embedder health |
| `mnem_remove` | tombstone a node (soft delete, traceable) |

## Notes

- The server runs **in-process** - no separate daemon, no port to manage.
- Embedder is bundled (MiniLM-L6-v2, ONNX). No network calls unless you wire one.
- Retrieval scoping: pass `label` to confine queries to a sub-graph.
