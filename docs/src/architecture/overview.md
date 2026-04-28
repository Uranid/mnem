# Architecture overview

mnem is a Rust workspace of ~15 crates plus optional Python bindings.

## Crate layout

| Crate | Role |
|-------|------|
| `mnem-core` | graph model, retrieval, indexing, sidecar embeddings |
| `mnem-backend-redb` | redb-backed store |
| `mnem-transport` | CAR codec + remote protocol primitives |
| `mnem-embed-providers` | ONNX bundled, Ollama, OpenAI, Cohere, mock |
| `mnem-sparse-providers` | BM25, SPLADE-onnx |
| `mnem-rerank-providers` | Cohere, Voyage, mock |
| `mnem-llm-providers` | OpenAI, Anthropic, Ollama, mock |
| `mnem-ingest` | parse + chunk + extract pipeline |
| `mnem-graphrag` | community summarisation, centroid+MMR |
| `mnem-ann` | HNSW wrapper |
| `mnem-cli` | `mnem` binary |
| `mnem-http` | HTTP JSON server |
| `mnem-mcp` | MCP server |
| `mnem-py` | PyO3 bindings |

## Data flow

```
ingest -> chunk -> extract -> embed -> commit (node + sidecar)
                                                    |
retrieve -> vector lane ----v
            sparse lane  ---+--> fuse -> rerank? -> graph-expand? -> top-k
            graph lane   ---^
```

## Read more

- [Retrieval](./retrieval.md) - fusion math, graph expansion, PPR
- [Storage](./storage.md) - content addressing, commits, sidecars

## Roadmap

| Milestone | Status |
|-----------|--------|
| 0.1.0 - public launch (this release) | shipping |
| 0.2.0 - per-user multi-tenancy primitives | planned |
| v1.2 - incremental Leiden, online community detection | planned |
| v2.0 - sharded multi-host commits | research |
