# MCP server

mnem implements the [Model Context Protocol](https://modelcontextprotocol.io)
over stdio. Drop it into any MCP client (Claude Desktop, Cursor, Zed, custom).

## Install

```bash
mnem integrate              # auto-detect installed hosts and wire everything
mnem integrate claude-code  # wire a specific host
```

For manual registration in any MCP client:

```json
{
  "mcpServers": {
    "mnem": {
      "command": "mnem",
      "args": ["mcp", "--repo", "/path/to/your-graph"]
    }
  }
}
```

## Tools exposed

| Tool | Purpose |
|------|---------|
| `mnem_stats` | Repo overview: op-head, head commit, ref summary, known labels |
| `mnem_schema` | List every node label and edge label in the current commit |
| `mnem_search` | Exact property-match search with optional outgoing-edge expansion |
| `mnem_get_node` | Fetch a single node by UUID (full props, content size, outgoing edge count) |
| `mnem_traverse` | One-hop neighbour walk from a start node via named edge labels |
| `mnem_incoming_edges` | List all edges pointing to a node (reverse lookup); equivalent to `mnem blame` in the CLI |
| `mnem_list_nodes` | Enumerate nodes at head, optionally filtered by label |
| `mnem_list_tags` | List all named tags in the repository |
| `mnem_retrieve` | Hybrid retrieval: vector + sparse + graph, fused via min-max convex combination or RRF |
| `mnem_commit` | Add nodes and/or edges as a single commit |
| `mnem_commit_relation` | Resolve-or-create subject + object + edge in one call |
| `mnem_resolve_or_create` | Find-or-create a node by a primary-key property |
| `mnem_recent` | Walk the op-log backwards (last N operations) |
| `mnem_vector_search` | Cosine nearest-neighbour search over stored embeddings |
| `mnem_delete_node` | Hard-remove a node from the current head |
| `mnem_tombstone_node` | Soft-delete (forget) a node; subsequent retrieves exclude it |
| `mnem_ingest` | Ingest a file or inline text as Doc + Chunk + Entity subgraph |
| `mnem_global_retrieve` | Semantic search on the global graph (`~/.mnemglobal/.mnem/`) only |
| `mnem_global_ingest` | Ingest a file or inline text into the global graph |
| `mnem_global_tombstone_node` | Soft-delete a node in the global graph; parallel to `mnem_tombstone_node` |
| `mnem_global_add` | Write nodes/edges directly to the global graph |
| `mnem_community_summarize` | Extractive centroid + MMR summarizer over a set of node UUIDs (`summarize` feature) |

## Notes

- The server runs **in-process**: no separate daemon, no port to manage.
- Embedder is bundled (MiniLM-L6-v2, ONNX). No network calls unless you wire one.
- **Local vs global**: `mnem_retrieve` searches the repo the server is pointed at. `mnem_global_retrieve` always searches `~/.mnemglobal/.mnem/` regardless of `--repo`.
- For the full field-level schema of each tool, run `mnem mcp --list-tools` or inspect [`crates/mnem-mcp/src/tools/descriptions.rs`](../../crates/mnem-mcp/src/tools/descriptions.rs).
