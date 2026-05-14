# mnem Roadmap

Planned work is tracked in [GitHub Issues](https://github.com/Uranid/mnem/issues)
and labelled by phase. The rough arc:

## Released - v0.1.x

- Core object model (Node, Edge, Commit, Operation, View)
- Prolly tree storage with content-addressed CIDs
- redb backend (`mnem-backend-redb`) - embedded key-value storage
- HNSW vector retrieval (three-lane fan-out: vector + graph + sparse)
- CLI (`mnem commit`, `mnem retrieve`, `mnem log`, `mnem ingest`, …)
- MCP server for agent integration
- Benchmark harness (`mnem bench`) with FinanceBench and LME results
- Ed25519 signing and revocation (§9)
- `mnem global` - cross-repo knowledge graph
- Python package (`mnem-cli` on PyPI) and npm package

## Near-term - v0.2.x

- Tombstone GC (compact expired tombstones without breaking CID chains)
- SPLADE sparse lane (currently feature-gated)
- Secondary index (`§4.8`) shipped to stable
- Remote sync (`mnem push` / `mnem pull` against an S3-compatible store)
- Pluggable embedding providers (OpenAI, Ollama, local ONNX)

## Medium-term - v0.3+

- Multi-writer merge (conflict-free CRDT-style view reconciliation)
- Linearize mode (`§6.5`) promoted to default for multi-process repos
- `mnem serve` - lightweight HTTP gateway for the MCP server
- Wasm build target

## Longer-term / exploratory

- Cross-language SDKs (TypeScript, Python native bindings)
- `mnem diff` - semantic diffing between commits
- Snapshot export / import (portable archive format)

---

*Last updated: 2026-05-09. See [open issues](https://github.com/Uranid/mnem/issues) for the current state of each item.*
