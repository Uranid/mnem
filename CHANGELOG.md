# Changelog

All notable changes to mnem.

## Unreleased

### CLI

- `mnem stats` and `mnem global stats` now print an additional
  `edges=M` slot with the real Prolly edge count, alongside the
  pre-existing `refs=N` slot (`view().refs.len()`, the number of
  branches). The output ordering is `op=... commit=... content=...
  refs=N edges=M labels=K`. Scripts that grepped `refs=` continue to
  work unchanged; new consumers should read `edges=`. Additive, not
  breaking.
- **BREAKING:** `mnem blame <unknown-uuid>` now exits non-zero with
  `Error: no node with id=<uuid>`. Previously exited 0 with an empty-edges
  table, which could not be distinguished from a node that existed with
  zero incoming edges.
- `mnem blame` includes a `relation` column (`src -[etype]-> dst`),
  widening the human table output.
- `mnem retrieve --no-vector` uses deterministic text relevance ordering for
  text-only matches rather than UUID ordering.
- `mnem query` / `mnem retrieve --where ntype=...` and `label=...` treat those
  keys as node-label filters. Non-string label values now return an error
  instead of an empty result.
- `mnem log --json` includes an additive `time` field.

### Core

- `Query` builder hides tombstoned nodes by default across every surface
  that uses it - `mnem query` CLI, MCP `list_nodes`, MCP `search`, HTTP
  `/traverse` seed selection, and direct library callers. Tombstoned
  (revoked / forgotten) nodes were previously leaking through these paths
  in violation of the documented "retrieval paths filter it out by
  default" contract. Audit and admin tooling can restore the prior
  behavior with the new `Query::include_tombstoned(true)` builder method.
- New library API: `mnem_core::index::query::Query::include_tombstoned(bool)`
  for explicit opt-in to tombstoned nodes.

### Integrations

- Hermes Agent hook integration refuses to reshape scalar hook entries or
  non-mapping `hooks:` config, avoids deleting modified generated hook scripts,
  and surfaces malformed Hermes YAML during `mnem integrate --check`.
- Hermes config backup filenames (`$HERMES_HOME/config.yaml.bak-<stamp>`)
  now use millisecond timestamps instead of second timestamps, preventing
  same-second collisions when `mnem integrate hermes` runs twice in quick
  succession.

## 0.1.0 - 2026-04-27

Initial public release. Versioned, mergeable, content-addressed knowledge
graph for AI agent memory. Local-first, Apache-2.0.

### Core

- Content-addressed graph store on top of redb (`mnem-backend-redb`).
- WASM-clean core crate (`mnem-core`) with no native dependencies.
- Hybrid GraphRAG retrieve: vector + sparse + graph signals fused, with
  per-result attribution of what got dropped at the token budget.
- Cryptographic node IDs; deterministic reindex pipeline.
- IndexSet on-disk format with dual-adjacency for fast graph traversal.

### Embedders

- `bundled-embedder` cargo feature: in-process MiniLM-L6-v2 via ONNX
  Runtime (`ort` load-dynamic). Zero-config retrieve out of the box.
- GPU variants: `bundled-embedder-cuda`, `bundled-embedder-directml`.
- Pluggable providers: Ollama, OpenAI, Cohere, mock. Configured via
  `.mnem/config.toml`.

### Surfaces

- **CLI** (`mnem-cli`): `init`, `add`, `status`, `stats`, `query`,
  `retrieve`, `embed`, `reindex --since <commit>`, `log`, `show`, `diff`,
  `ref`, `config`, `integrate`, `doctor`, `completions`.
- **MCP server** (`mnem mcp`): exposes the graph to any MCP-aware host.
  Tools include `mnem_retrieve` (auto-embed text), `mnem_commit_relation`
  (compound), `mnem_delete_node`, `mnem_list_nodes`.
- **HTTP/REST API** (`mnem http`, axum-based): loopback-safe by default.
  See ADR-0019 for the tokio boundary rationale.
- **Python bindings** (`mnem-py`, via pyo3 + maturin): `Repo` with
  `add_node`, `delete_node`, `update_node` (keyword-only args), retrieve.

### Integrations

- `mnem integrate` interactive wizard auto-detects and configures:
  Claude Desktop, Claude Code, Cursor, Continue, Zed, Codex, Gemini CLI,
  and Hermes Agent (pre/post LLM hooks).

### Distribution

- Release-binary matrix: 4 triples (linux-x86_64, linux-musl, macos-arm64,
  windows-msvc). `install.sh` / `install.ps1` with env-var safety guards.
- Crates published: `mnem-core`, `mnem-cli`, `mnem mcp`, `mnem http`,
  `mnem-backend-redb`, `mnem-embed-providers`, `mnem-py`.

### Quality gates

- Workspace MSRV 1.95, edition 2024, `unsafe_code = "forbid"` baseline.
- CI matrix: Linux + macOS + Windows on stable, beta, 1.95 (MSRV pin).
- Determinism via proptest, perf via criterion, nightly fuzz harness.
- `cargo audit` weekly. SLSA provenance + reproducibility workflows.
- Apache-2.0, NOTICE, SECURITY policy, Code of Conduct, dependabot, CODEOWNERS.

### Platform support

- Linux, macOS, Windows. WASM-targeted core crate. Windows requires MSVC
  toolchain when using `bundled-embedder` (ort dynamic load).
