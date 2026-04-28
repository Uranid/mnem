# CLI reference

`mnem` is the single entry point. Subcommands wrap repo operations.

## Common subcommands

```bash
mnem init [path]                     # create .mnem/ in path (default: cwd)
mnem ingest <file|-> [--json] [...]  # add nodes from file or stdin
mnem retrieve <text> [...]           # query (vector + sparse + graph)
mnem serve [--bind addr]             # start HTTP server (alias of mnem-http)
mnem mcp install                     # wire as MCP server in your client
mnem doctor                          # probe embedder + store + config
```

## Inspection

```bash
mnem stats                # commits, nodes, embeddings, store size
mnem log [-n N]           # commit history
mnem cat <cid>            # dump a node by CID
mnem diff <cid> <cid>     # diff two commits
mnem export --format car  # export as CAR archive
```

## Advanced retrieve flags

```bash
--top-k N                 # number of items to return (default 10)
--vector-cap N            # candidate pool from vector lane (default 256)
--label <str>             # restrict to a label scope
--graph-expand N          # multi-hop expansion budget
--graph-mode <decay|ppr>  # graph scoring: decay (default) or PPR
--rerank <provider:model> # post-rerank with a model
--summarize               # add community summarization layer
```

For complete option lists run `mnem <subcommand> --help`. Long-form
documentation for each subcommand lives in [guides](./guides/).
