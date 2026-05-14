# CLI reference

`mnem` is the single entry point. Subcommands wrap repo operations.

## Common subcommands

```bash
mnem init [path]                     # create .mnem/ in path (default: cwd)
mnem ingest <file>                   # add nodes from a file
mnem retrieve <text> [...]           # query (vector + sparse + graph)
mnem mcp                             # start the MCP JSON-RPC server over stdio
mnem mcp --repo ~/notes              # point the MCP server at a specific graph
mnem http                            # start the HTTP JSON API (loopback by default)
mnem integrate                       # wire as MCP server in your agent host
mnem doctor                          # probe embedder + store + config
```

## Inspection

```bash
mnem stats                # commits, nodes, embeddings, store size
mnem log [-n N]           # commit history
mnem cat-file <cid>       # dump a node by CID
mnem diff <cid> <cid>     # diff two commits
mnem export               # export as CAR archive
```

## Advanced retrieve flags

```bash
--limit N                 # number of items to return (no limit by default); short: -n
--vector-cap N            # candidate pool from vector lane (default 256)
--graph-expand N          # multi-hop expansion budget
--graph-mode <decay|ppr>  # graph scoring: decay (default) or PPR
--rerank <provider:model> # post-rerank with a model
--summarize               # add community summarization layer
--community-filter        # Leiden community filter; drop low-coverage communities
--explain                 # print per-item lane scores (vector, sparse, graph_expand, rerank)
```

`--explain` prints a `lanes:` line for each result on stdout:

```
lanes: vector=0.8231 sparse=0.6120 graph_expand=0.0000 rerank=0.9104
```

In multi-query mode (`--multi-query`), per-lane scores are not propagated through RRF fusion, so `--explain` notes this inline.

## Blame

`mnem blame` lists all incoming edges to a node - which agents wrote those edges and in which commit.

```bash
mnem blame <node-uuid>                    # all incoming edges
mnem blame <node-uuid> --etype authored   # filter to one edge type
mnem blame <node-uuid> --first-writer     # show the oldest ancestor commit that introduced each edge
```

`--first-writer` performs a BFS over the operation ancestry chain for each edge and reports the earliest commit that first introduced it. The output header changes from `in_commit` to `first_writer`. Commits with unparseable ancestors produce a stderr warning and are skipped.

## Revert

`mnem revert <op-cid>` undoes a single committed operation by replaying its inverse.

```bash
mnem revert <op-cid>
mnem revert <op-cid> --message "reason for revert"
```

Behavior:

- Reverting an already-reverted op is a no-op.
- Chained reverts work: revert A, then revert the revert of A restores A.
- `DiffEntry::Changed` (node/edge updates) are fully supported: the revert restores the previous value of each changed field.
- Root-op reverts (reverting the very first operation) are supported.

## Ingest flags

```bash
--chunker <auto|paragraph|recursive|sentence_recursive|session|structural>  # chunking strategy (default: auto)
--extractor keybert                            # enable KeyBERT keyphrase extraction
--max-tokens N                                # token budget per chunk (default: 512)
--overlap N                                   # overlap tokens between chunks (default: 32)
--recursive                                   # ingest a directory recursively
```

For complete option lists run `mnem <subcommand> --help`. Long-form
documentation for each subcommand lives in [guides](./guides/).
