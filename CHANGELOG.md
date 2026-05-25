# Changelog

All notable changes to mnem.

## Unreleased

### HTTP - Breaking changes

The following HTTP response fields were renamed to align with the CLI wire
format and the rest of the HTTP API's naming conventions. Any client that
parses these fields by name must be updated.

- `POST /v1/edges`: response field `edge_id` renamed to `id`; schema string
  changed from `mnem.v1.post-edge` to `mnem.v1.edge-created` (matches the
  `node-created` / `node-deleted` pattern used by all other mutation responses).
- `GET /v1/nodes/{id}/embedding`: schema string changed from
  `mnem.v1.node_embedding` (underscore) to `mnem.v1.node-embedding` (hyphen),
  fixing an inconsistency with every other schema string in the API.
- `GET /v1/log`: field `op_id` renamed to `cid`; field `message` renamed to
  `description`. Both now match `mnem log --format=json` output.
- `GET /v1/log`: default limit changed from 50 to 20, matching the CLI default.
  Clients that rely on the implicit default will now receive fewer entries;
  pass `?limit=N` explicitly to control this.

### HTTP - Bug fixes

- `POST /v1/embed`: repo lock is now released before the embedding loop so
  concurrent requests are not blocked during network-bound embed calls.
- `GET /v1/retrieve` (multi-query / HyDE paths): repo lock is released before
  LLM calls, unblocking concurrent requests during paraphrase generation.
- `embed_text_of`: content truncation now aligns to a UTF-8 character boundary
  (`floor_char_boundary`) instead of a raw byte offset, preventing a panic on
  nodes whose content contains multi-byte characters near the 4096-byte cap.

## 0.1.7 - 2026-05-21

### CLI

- `mnem ingest` now reports the real edge count in the `N edges` slot
  of its summary line (`ingested K files, J chunks, N nodes, E edges
  in T ms`). Previously this slot was wired to the extractor's
  `relation_count`, so a single-text ingest with no LLM-found
  relations printed `0 edges` even though the structural `chunk_of`
  edge was on disk. The new `IngestResult::edge_count` field covers
  every edge written (`chunk_of` + `chunk_mentions` + relations);
  `relation_count` stays available as a strict subset. The same
  field surfaces in the HTTP ingest JSON response and the MCP
  `mnem_ingest` text output.
- `mnem stats` and `mnem global stats` now print an additional
  `edges=M` slot with the real Prolly edge count, alongside the
  pre-existing `refs=N` slot (`view().refs.len()`, the number of
  branches). The output ordering is `op=... commit=... content=...
  refs=N edges=M labels=K`. Scripts that grepped `refs=` continue to
  work unchanged; new consumers should read `edges=`. Additive, not
  breaking.
- `mnem blame` adds two flags:
  - `--no-relation` hides the new `relation` column
    (`<src> -[etype]-> <dst>`) so the human table fits in narrower
    terminals and matches the pre-#30 column shape for positional
    column-parsing scripts.
  - `--strict` exits non-zero on an unknown node id. Without the flag,
    an unknown node prints a one-line stderr warning and exits zero
    with an empty-edges table (back-compat with the pre-#30 behaviour).
- `mnem blame` shows a `relation` column (`<src> -[etype]-> <dst>`) by
  default in the human table. Pass `--no-relation` to suppress it.
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
- `Query` and `Retriever` now hide the `mnem init` system anchor
  (node id `00000000-0000-7000-8000-6d6e656d0001`) by default,
  matching the tombstone-filter shape. The anchor is structural - it
  exists for the BUG-56 fast-forward-pull guarantee and as a graph-
  history root - and carries no agent-meaningful content. Previously,
  the post-ingest reindex pass embedded it via the label-fallback
  path in `reindex_text_of`, then surfaced it as low-score noise in
  every subsequent `mnem retrieve` / `mnem query --where ntype=Meta`.
  New opt-in builders for audit / repair: `Query::include_system(true)`
  and `Retriever::include_system(true)`. The `mnem reindex` candidate
  collector, the dense / sparse legacy-lift paths, and the HNSW
  build all skip system nodes too so a corrupt anchor can't slip
  back into any vector index.
- New library API: `mnem_core::anchor` module exposing
  `ANCHOR_NODE_UUID`, `anchor_node_id()`, `is_anchor_node_id(&NodeId)`,
  and `is_system_node(&Node)`. Single source of truth for the
  anchor's identity, shared by `mnem init` (writer), `mnem reindex`
  (skip filter), and `Query` / `Retriever` (read-time filters).
- New ingest field: `mnem_ingest::IngestResult::edge_count` reports the
  total number of edges written by a run (structural + mentions +
  relations). `relation_count` keeps its narrower meaning as a
  strict subset (LLM-extracted subject-object triples only).

### Integrations

- Hermes Agent hook integration refuses to reshape scalar hook entries or
  non-mapping `hooks:` config, avoids deleting modified generated hook scripts,
  and surfaces malformed Hermes YAML during `mnem integrate --check`.
- Hermes config backup filenames (`$HERMES_HOME/config.yaml.bak-<stamp>`)
  now use millisecond timestamps instead of second timestamps, preventing
  same-second collisions when `mnem integrate hermes` runs twice in quick
  succession.

### Dependencies

- Replace the unmaintained `serde_yml = "0.0.12"` (the Hermes config
  YAML writer dep) with the actively-maintained community fork
  `serde_yaml_ng = "0.10"`. Drops the `libyml` transitive in favour of
  `unsafe-libyaml` (the well-scrutinised C-FFI binding used by the
  upstream `serde_yaml`). Resolves the `unmaintained` dependabot
  advisories on `serde_yml` and `libyml`.
- Bump tree-sitter family for the code-ingest parsers:
  `tree-sitter` 0.24 -> 0.26, `tree-sitter-rust` 0.23 -> 0.24,
  `tree-sitter-python` 0.23 -> 0.25, `tree-sitter-javascript` 0.23
  -> 0.25, `tree-sitter-go` 0.23 -> 0.25. The six remaining language
  grammars (typescript, java, c, cpp, ruby, c-sharp) stay at 0.23
  pending upstream releases; tree-sitter 0.26 is backward-compatible
  with older grammar ABI versions so the mixed-version state is
  intentional. Verified by running the full `mnem-ingest` test suite
  (Rust + Python parser tests including ABI-fallback scenarios).

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
