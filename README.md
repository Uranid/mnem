<div align="center">

<img src="assets/logo/mnem-logo.svg" alt="mnem logo" width="140" height="140" />

# mnem

**Git for knowledge graphs.** Versioned, mergeable agent memory with
hybrid GraphRAG retrieval. Embed in Rust or Python, run the CLI, or plug
into any MCP client. Local-first. Apache-2.0.

[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue?style=flat)](LICENSE)
[![CI](https://github.com/Uranid/mnem/actions/workflows/ci.yml/badge.svg)](https://github.com/Uranid/mnem/actions/workflows/ci.yml)
[![crates.io](https://img.shields.io/crates/v/mnem-cli.svg?style=flat)](https://crates.io/crates/mnem-cli)
[![docs.rs](https://img.shields.io/docsrs/mnem-core?style=flat&label=docs.rs)](https://docs.rs/mnem-core)
[![MSRV 1.95](https://img.shields.io/badge/MSRV-1.95-orange?style=flat)](rust-toolchain.toml)
[![Runs on Linux macOS Windows WASM](https://img.shields.io/badge/runs%20on-linux%20%7C%20macos%20%7C%20windows%20%7C%20wasm-2ea44f?style=flat)](#install)

</div>

---

## What it is

mnem is a versioned, content-addressed knowledge graph for AI agent
memory. Think Git, but the diff is over a graph and the merge is over
embeddings. Every fact has a cryptographic identity, every retrieve
fuses vector + sparse + graph signals, and every response tells you
exactly what got dropped at your token budget.

---

## Quickstart

```bash
cargo install --locked mnem-cli --features bundled-embedder

mkdir my-graph && cd my-graph
mnem init
mnem ingest README.md
mnem retrieve "what does this project do" --top-k 5
```

`--features bundled-embedder` ships the in-process ONNX MiniLM-L6-v2
so `mnem retrieve` works zero-config. Drop the flag if you want to
configure your own embedder (Ollama / OpenAI / Cohere) via
`.mnem/config.toml`. GPU-accelerated variants
(`bundled-embedder-cuda`, `bundled-embedder-directml`) live under
[Install](#install).

Five minutes from zero. See [`docs/src/quickstart.md`](docs/src/quickstart.md) for the longer walkthrough,
or jump to [Install](#install) for your platform.

---

## `mnem integrate` (interactive setup wizard)

```bash
mnem integrate
```

The shortest path from a fresh checkout to a working agent. Detects
your environment, prompts for embedder + LLM choice (ONNX bundled /
Ollama / OpenAI / Cohere / mock), writes `.mnem/config.toml`,
smoke-tests the result, and offers to wire mnem into every supported
client in one pass. Around two minutes end to end.

Targets: Claude Desktop, Claude Code, Cursor, Continue, Zed, custom MCP
clients, raw HTTP, raw Python.

---

## Wire into any MCP client

```bash
mnem integrate        # interactive: detect hosts + wire MCP entries
mnem integrate --all # non-interactive: wire all detected hosts
```

Auto-detects and configures: **Claude Desktop**, **Claude Code**,
**Cursor**, **Continue**, **Zed**. Any other MCP-aware host works via a
hand-edited `mcpServers` entry pointing at the `mnem-mcp` stdio
binary (see [`docs/src/mcp.md`](docs/src/mcp.md)).

Restart your client. The agent gets `mnem_retrieve`, `mnem_ingest`,
`mnem_traverse`, `mnem_stats`, `mnem_remove` (and 6 more) as native
tools. No extra daemon, no port to manage.

---

## Commands

| Command | What it does |
|---------|-------------|
| `mnem init` | create a new graph in the current dir |
| `mnem ingest <file>` | add nodes from a file (md / pdf / chat-json) |
| `mnem retrieve <query>` | hybrid retrieval (vector + sparse + graph) |
| `mnem integrate` | interactive setup: configure embedder, wire MCP hosts |
| `mnem integrate --check` | report which hosts are wired; non-mutating |
| `mnem doctor` | probe embedder + store + config; first-run sanity |
| `mnem stats` | nodes, edges, refs, embedder health, repo size |
| `mnem log` / `diff` / `branch` / `merge` | git-style history ops over the graph |
| `mnem ref` / `cat-file` / `blame` | inspect refs and individual objects |
| `mnem export` / `import` | CAR archives for ship-and-load |
| `mnem-http --bind addr` | HTTP JSON API server (separate binary) |

`mnem integrate` is the shortest path to a working agent: detects your
environment, prompts for embedder + LLM choice, writes config,
smoke-tests the result. Around two minutes end to end.

Full reference: [`docs/src/cli.md`](docs/src/cli.md).

---

## Install

> The `bundled-embedder` Cargo feature ships the in-process ONNX
> MiniLM-L6-v2 (no Ollama, no API keys). Recommended for first-run.
> Drop the flag to leave embedder configuration to your own
> `.mnem/config.toml` (`provider = ollama|openai|cohere|...`).

<details>
<summary><b>macOS</b></summary>

```bash
# Recommended: Cargo with bundled embedder
cargo install --locked mnem-cli --features bundled-embedder

# Homebrew (0.1.0+)
brew install mnem
```

</details>

<details>
<summary><b>Linux</b></summary>

```bash
# Any distro: Cargo with bundled embedder
cargo install --locked mnem-cli --features bundled-embedder

# CUDA-accelerated embedder (NVIDIA GPU)
cargo install --locked mnem-cli --features bundled-embedder-cuda

# Distro packages (v1.x)
yay -S mnem                       # Arch (AUR)
nix-env -iA nixpkgs.mnem          # Nixpkgs
```

</details>

<details>
<summary><b>Windows</b></summary>

```powershell
# Recommended: Cargo with bundled embedder
cargo install --locked mnem-cli --features bundled-embedder

# DirectML-accelerated embedder (any GPU vendor on Windows)
cargo install --locked mnem-cli --features bundled-embedder-directml

# Package managers (v1.x)
winget install mnem
scoop install mnem
```

</details>

<details>
<summary><b>Python (PyPI)</b></summary>

```bash
pip install mnem-cli
mnem --version
```

Ships the `mnem` binary as a manylinux / macOS / Windows wheel with
the bundled embedder pre-baked. No Cargo feature flag needed; the
PyPI build always includes it.

</details>

<details>
<summary><b>Docker</b></summary>

```bash
docker run --rm -p 9876:9876 ghcr.io/uranid/mnem-http:latest
```

The image is built with `FEATURES=onnx-bundled`; the bundled embedder
is always present in Docker.

</details>

<details>
<summary><b>From source</b></summary>

```bash
git clone https://github.com/Uranid/mnem
cd mnem

# Build the CLI with bundled embedder (recommended)
cargo build --release --locked -p mnem-cli --features bundled-embedder

# Or build without (bring your own embedder via config.toml)
cargo build --release --locked -p mnem-cli

./target/release/mnem --version
```

Requires Rust 1.95+. If needed: `rustup install 1.95 && rustup default 1.95`.

</details>

<details>
<summary><b>WASM (in-browser)</b></summary>

```bash
cargo build --release --target wasm32-unknown-unknown -p mnem-core
```

`mnem-core` has no tokio, no filesystem, no network. Same retrieval
logic runs unchanged in browsers and edge workers. The bundled
embedder is NOT WASM-compatible (ort uses native libs); supply
embeddings from the host.

</details>

### Verify

```bash
mnem --version
mnem doctor
```

`mnem doctor` probes embedder + store + config and prints a
green/yellow/red checklist. Useful first command after install.

---

## What you get

Legend: $\color{gold}{\textbf{(unique)}}$ = mnem-original; not shipping
in any other agent-memory system today. $\color{orange}{\textbf{(rare)}}$
= exists in 1-2 peers, often gated behind paid tiers.
$\color{gray}{\textbf{(standard)}}$ = table-stakes done well.

1. **Plug-and-play** $\color{orange}{\textbf{(rare)}}$. Bundled ONNX MiniLM-L6-v2 runs in-process. No Ollama, no API keys, no cold-start network call. `mnem init` and you're retrieving. mem0 + Graphiti both require an external LLM endpoint at ingest. → [Install](docs/src/install.md)

2. **Swappable providers** $\color{orange}{\textbf{(rare)}}$. Embedder, sparse encoder, reranker, and LLM all set via config strings. Switch local ONNX to hosted Cohere with one flag. No fork, no rebuild. Most peers ship single-path stacks; provider swap is architectural here. → [Embedding providers](docs/src/guides/embed-providers.md)

3. **Hybrid GraphRAG retrieval** $\color{gray}{\textbf{(standard, done well)}}$. Vector (HNSW) + sparse (BM25 / SPLADE) + graph traversal, fused via RRF. GraphRAG built in and optional → on for multi-hop, off when dense saturates. → [Retrieval architecture](docs/src/architecture/retrieval.md)

4. **Token-budget transparency** $\color{gold}{\textbf{(unique)}}$. Every retrieve emits `tokens_used`, `candidates_seen`, `dropped` counters. No silent truncation. **No other agent-memory system exposes this** as first-class response fields. → [Observability](docs/src/architecture/retrieval.md)

5. **Content-addressed objects** $\color{gold}{\textbf{(unique)}}$. Every node / tree / sidecar / commit has a CID derived from canonical DAG-CBOR + BLAKE3. Identical content collapses across machines. Determinism + replay become real, not a slogan. Peers use opaque UUIDs. → [Core concepts](docs/src/core-concepts.md)

6. **Versioned + 3-way mergeable** $\color{gold}{\textbf{(unique)}}$. Commits, branches, diff, log, **three-way merge**, signed Ed25519 history. Two agents writing the same scope offline reconcile by graph + embedding merge → not "last write wins". → [Core concepts](docs/src/core-concepts.md)

7. **Deterministic ingest** $\color{orange}{\textbf{(rare)}}$. No LLM at ingest. parse + chunk + extract is statistical (KeyBERT optional), so same bytes in → same CIDs out. Audit-friendly, fuzz-tested, byte-identical across machines. → [Ingest pipeline](docs/src/guides/ingest.md)

8. **Reproducible benchmarks** $\color{orange}{\textbf{(rare)}}$. Matches MemPalace on LongMemEval (R@5 0.966), +0.218 R@5 on LoCoMo, +0.047 on ConvoMem, +0.120 on MemBench under the same embedder. Numbers ship with the harness. → [Benchmarks](benchmarks/README.md)

9. **Single binary** $\color{orange}{\textbf{(rare)}}$. ~40 MB Docker image. Embedded redb store. No daemon, no cloud, no account. Runs offline. → [Architecture overview](docs/src/architecture/overview.md)

10. **WASM-clean core** $\color{gold}{\textbf{(unique among peers)}}$. `mnem-core` has no tokio, no filesystem, no network. Same retrieval logic compiles unchanged to `wasm32` → runs in Chrome, on Cloudflare Workers, on Lambda cold-start. Graphiti + mem0 are Python + external DB stacks; they cannot ship to the edge. → [Architecture overview](docs/src/architecture/overview.md)

11. **Four surfaces, one core** $\color{orange}{\textbf{(rare)}}$. CLI, HTTP, MCP, and Python all wrap the same Rust engine. `mnem integrate` wires the MCP server into Claude Desktop and other hosts. → [CLI reference](docs/src/cli.md) | [MCP](docs/src/mcp.md)

12. **Skills as graphs, not markdown** $\color{gold}{\textbf{(unique angle)}}$. Today, agent skills live in flat `.md` files → downloaded, pasted into prompts, hand-edited, never queried. mnem promotes them to a versioned, queryable, mergeable graph. Export your graph, import someone else's, diff the two, merge the parts you want. → [Agent memory guide](docs/src/guides/agent-memory.md)

13. **Property + fuzz tests** $\color{orange}{\textbf{(rare)}}$. Parsers are property-tested + fuzz-harnessed; CAR round-trip and merge-commit are byte-identical. Trust signal usually only seen at the infra-DB tier.

## GraphRAG

mnem ships GraphRAG built in. One knob per stage, opt-in per query,
never required. Dense retrieval saturates first; turn graph stages on
when multi-hop, cross-document, or compositional queries surface.

### Stages and flags

| Stage | Flag | What it does |
|-------|------|------|
| **Vector lane** | always on | HNSW over per-commit dense embeddings (default 384-d MiniLM). |
| **Sparse lane** | config-driven | BM25 + SPLADE-onnx, fused with vector via Reciprocal Rank Fusion. Toggled by `[sparse]` block in `config.toml`. |
| **Vector candidate pool** | `--vector-cap <N>` | Lift the dense pool size from default 256. Higher = better long-tail recall, +cost. |
| **Top-K** | `--top-k <N>` | Final returned set (default 10). |
| **Label scope** | `--label <str>` | Confine retrieval to a sub-graph (per-user, per-conversation, per-tenant). |
| **Graph expansion** | `--graph-expand <N>` | Add N neighbours of top-K seeds via authored edges. Audit-recommended default `20` when graph is on. |
| **Graph mode** | `--graph-mode <decay\|ppr>` | `decay` (default) = exponential weight by hop. `ppr` = Personalised PageRank over the hybrid adjacency index, paper-grade scoring for multi-hop. |
| **Community filter** | `--community-filter` | Run Leiden community detection; drop low-coverage communities before fusion. Implicit `--community-min-coverage 0.1`. |
| **KeyBERT extraction** | `--extractor keybert` | Ingest-time keyphrase enrichment. Strengthens sparse + community signals. |
| **Summarisation** | `--summarize` | Centroid + MMR summary of the top-K, with diversity. |
| **Cross-encoder rerank** | `--rerank <provider:model>` | Post-fusion reorder. Supports `cohere:rerank-english-v3.0`, `voyage:rerank-1`, local. |
| **Hybrid v4 boost** | `--hybrid-v4-boost` | Bench-harness BM25-style score boost. Mirrors MemPalace's `hybrid_v4` for apple-to-apple comparisons. NOT a default for production. |

### Quick examples

```bash
# Dense baseline (no graph, no fancy stuff)
mnem retrieve "what does this project do" --top-k 5

# Add multi-hop graph traversal
mnem retrieve "..." --graph-expand 20

# Full Palace-tier stack: graph-expand + community-filter + PPR + KeyBERT
mnem retrieve "..." --graph-expand 20 --community-filter --graph-mode ppr --extractor keybert

# Stack a cross-encoder reranker on top
mnem retrieve "..." --graph-expand 20 --community-filter --rerank cohere:rerank-english-v3.0

# Per-user scoping
mnem retrieve "..." --label user-42 --graph-expand 20

# Hybrid v4 boost (bench harness; mirrors MemPalace harness helper)
mnem retrieve "..." --hybrid-v4-boost
```

### When to enable

- **Single-document corpus, simple queries** → leave graph off, dense saturates
- **Multi-hop / compositional questions** → `--graph-expand 20`
- **Long history with cross-document references** → add `--community-filter`
- **Recall ceiling needed** → stack `--rerank` on top
- **Multi-tenant agent memory** → always `--label <tenant>`

→ Full retrieval architecture: [`docs/src/architecture/retrieval.md`](docs/src/architecture/retrieval.md)
→ Tuning playbook: [`docs/src/guides/retrieval-tuning.md`](docs/src/guides/retrieval-tuning.md)

## Benchmarks

ONNX MiniLM-L6-v2 embedder, same bytes on every system. No LLM rerank.
Dense retrieval (vector + top-k); the LongMemEval Hybrid v4 row mirrors
MemPalace's harness helper. Reproduce: `bash benchmarks/harness/run_bench.sh`.

### vs MemPalace

> MemPalace's column carries their public headline numbers, **cross-verified**
> by running their adapter end-to-end under our harness. mnem's column comes
> from the same harness, same embedder bytes (ONNX MiniLM-L6-v2), same
> dataset hashes. Both columns are reproducible; raw artefacts in
> [`benchmarks/proofs/v0.1.0/`](benchmarks/proofs/v0.1.0/).

| Benchmark | Split | Metric | MP | mnem 0.1.0 | Δ |
|-----------|-------|--------|----|-----------|---|
| LongMemEval | 500 Q | R@5 session | 0.966 | $\color{lightgreen}{\textbf{0.966}}$ | ±0 |
| LongMemEval | 500 Q | R@10 session | 0.982 | $\color{lightgreen}{\textbf{0.982}}$ | ±0 |
| LoCoMo | 1986 Q | R@5 session | 0.508 | $\color{lightgreen}{\textbf{0.726}}$ | **+0.218** |
| LoCoMo | 1986 Q | R@10 session | 0.603 | $\color{lightgreen}{\textbf{0.855}}$ | **+0.252** |
| ConvoMem | 250 (5x50) | avg recall | 0.929 | $\color{lightgreen}{\textbf{0.976}}$ | **+0.047** |
| MemBench | simple/roles 100 | R@5 | 0.840 | $\color{lightgreen}{\textbf{0.960}}$ | **+0.120** |
| MemBench | highlevel/movie 100 | R@5 | 0.950 | $\color{lightgreen}{\textbf{1.000}}$ | **+0.050** |
| LongMemEval | 500 Q hybrid-v4 | R@5 session | 0.982 | $\color{salmon}{0.976}$ | -0.006 |

### vs mem0

> mem0 doesn't publish recall@K headlines on these datasets, so both
> columns are our reproductions: we ran mem0's adapter end-to-end under
> the same harness, same embedder bytes (ONNX MiniLM-L6-v2), and the
> same per-item scoping (`infer=False` + `user_id`-per-item) documented
> in [`benchmarks/results/methodology.md`](benchmarks/results/methodology.md).
> Both columns reproducible; raw artefacts in
> [`benchmarks/proofs/v0.1.0/`](benchmarks/proofs/v0.1.0/).

| Benchmark | Split | Metric | mem0 | mnem 0.1.0 | Δ |
|-----------|-------|--------|------|-----------|---|
| LongMemEval | 500 Q | R@5 session | 0.946 | $\color{lightgreen}{\textbf{0.966}}$ | **+0.020** |
| LongMemEval | 500 Q | R@10 session | 0.962 | $\color{lightgreen}{\textbf{0.982}}$ | **+0.020** |
| LoCoMo | 1986 Q | R@5 session | 0.466 | $\color{lightgreen}{\textbf{0.726}}$ | **+0.260** |
| LoCoMo | 1986 Q | R@10 session | 0.676 | $\color{lightgreen}{\textbf{0.855}}$ | **+0.179** |
| ConvoMem | 250 (5x50) | avg recall | 0.558 | $\color{lightgreen}{\textbf{0.976}}$ | **+0.418** |
| MemBench | simple/roles 100 | R@5 | 0.410 | $\color{lightgreen}{\textbf{0.960}}$ | **+0.550** |
| MemBench | highlevel/movie 100 | R@5 | 0.970 | $\color{lightgreen}{\textbf{1.000}}$ | **+0.030** |
| LongMemEval | 500 Q hybrid-v4 | R@5 session | 0.930 | $\color{lightgreen}{\textbf{0.976}}$ | **+0.046** |

### Latency

| Benchmark | mean retrieve | total wall (n questions) |
|-----------|--------------:|-------------------------:|
| LongMemEval 500 Q | 711 ms | 1127 s (~19 min) |
| LongMemEval 500 Q hybrid-v4 | 729 ms | 1133 s (~19 min) |
| LoCoMo 1986 Q | 333 ms | 720 s (~12 min) |
| ConvoMem 250 (5x50) | 398 ms | 218 s (~4 min) |
| MemBench simple/roles 100 | 1874 ms (e2e) | 187 s (~3 min) |
| MemBench highlevel/movie 100 | 491 ms (e2e) | 49 s (~1 min) |

`(e2e)` = end-to-end mean when the adapter doesn't expose phase timing.

### Reproduce

```bash
# Cache datasets (one-time; 264 MB LongMemEval + 3 MB LoCoMo)
mnem bench fetch longmemeval     # HuggingFace
mnem bench fetch locomo          # GitHub raw
mnem bench fetch                 # alternative: every shipped bench in one go

# Run via interactive wizard or explicit args
mnem bench                       # TUI; default selects v0.1.0 items
mnem bench run --benches longmemeval --with mnem --limit 50 --non-interactive
mnem bench results ./bench-out   # re-render RESULTS.md from prior run
```

`mnem bench` ships in 0.1.0 with the in-process mnem adapter +
LongMemEval / LoCoMo scorers + real ONNX MiniLM-L6-v2 (50q canary
lands R@5 = 0.92, close to the headline 0.966 on the full split).
ConvoMem, MemBench, hybrid-v4, and mem0 / MemPalace side-by-side
adapters land in 0.2.0 (TUI lists them today behind `[0.2.0]` tags).

For the headline parity numbers above (full splits), the legacy
Bash harness remains the canonical reproduction path:

```bash
bash benchmarks/harness/run_bench.sh
```

Methodology, raw artifacts, per-bench breakdowns:
[`benchmarks/`](benchmarks/) and [`docs/src/benchmarks/`](docs/src/benchmarks/).
See also [`docs/src/benchmarks/run-locally.md`](docs/src/benchmarks/run-locally.md)
for the `mnem bench` walkthrough.

## Compared to others

- [mnem vs Graphiti](docs/src/comparisons/graphiti.md) - bitemporal substrate, Neo4j-bound
- [mnem vs mem0](docs/src/comparisons/mem0.md) - agent memory layer, OSS leader
- [mnem vs MemPalace](docs/src/comparisons/mempalace.md) - methodology peer
- [mnem vs Supermemory](docs/src/comparisons/supermemory.md) - closed-cloud incumbent
- [mnem vs Cognee](docs/src/comparisons/cognee.md) - KG-for-agents alternative
- [mnem vs Letta](docs/src/comparisons/letta.md) - agent-memory framework
- [mnem vs graphify](docs/src/comparisons/graphify.md) - lightweight graph tool

Full matrix: [`docs/src/comparisons/README.md`](docs/src/comparisons/README.md).

## When NOT to use mnem

- **You need transactional OLTP.** mnem is append-only with versioned
  history; row-level UPDATE/DELETE semantics aren't the model.
- **You need sub-50 ms cloud-scale retrieval at 10k+ QPS.** mnem is
  local-first. Multi-region sharded retrieval is on the roadmap, not
  in v1.

> Looking for hosted memory, multi-region replicas, shared graphs across
> teams, or a managed remote layer? A sibling project bringing those to
> mnem is in active development - watch this space.

## Architecture

15-crate Rust workspace. WASM-clean core, async/IO at the edges.
Per-commit embedding sidecars; node identity is decoupled from the
embedder. Three retrieval lanes (vector + sparse + graph) fused with RRF.

Full overview: [`docs/src/architecture/overview.md`](docs/src/architecture/overview.md).

## Documentation

- [Quickstart](docs/src/quickstart.md) - five-minute walkthrough
- [Install](docs/src/install.md) - per-platform install matrix
- [CLI reference](docs/src/cli.md) - every subcommand and flag
- [MCP server](docs/src/mcp.md) - tools exposed, client wiring
- [Core concepts](docs/src/core-concepts.md) - CIDs, commits, labels
- [Configuration](docs/src/configuration.md) - env vars, config.toml
- [Architecture overview](docs/src/architecture/overview.md)
- [Benchmarks methodology](docs/src/benchmarks/methodology.md)
- [Reproduce benchmarks](docs/src/benchmarks/reproduce.md)
- [Retrieval tuning](docs/src/guides/retrieval-tuning.md)
- [Embedding providers](docs/src/guides/embed-providers.md)
- [Migrations](docs/src/migrations/)

## Crates

| Crate | Role |
|-------|------|
| [`mnem-cli`](crates/mnem-cli) | `mnem` binary - one command for everything |
| [`mnem-core`](crates/mnem-core) | graph model, retrieval, indexing, sidecars |
| [`mnem-http`](crates/mnem-http) | HTTP JSON server |
| [`mnem-mcp`](crates/mnem-mcp) | MCP server (stdio) |
| [`mnem-py`](crates/mnem-py) | PyO3 Python bindings |
| [`mnem-embed-providers`](crates/mnem-embed-providers) | ONNX bundled, Ollama, OpenAI, Cohere |
| [`mnem-sparse-providers`](crates/mnem-sparse-providers) | BM25, SPLADE-onnx |
| [`mnem-rerank-providers`](crates/mnem-rerank-providers) | Cohere, Voyage |
| [`mnem-llm-providers`](crates/mnem-llm-providers) | OpenAI, Anthropic, Ollama |
| [`mnem-ingest`](crates/mnem-ingest) | parse + chunk + extract pipeline |
| [`mnem-graphrag`](crates/mnem-graphrag) | community summarisation, centroid + MMR |
| [`mnem-ann`](crates/mnem-ann) | HNSW wrapper |
| [`mnem-backend-redb`](crates/mnem-backend-redb) | redb-backed store |
| [`mnem-transport`](crates/mnem-transport) | CAR codec + remote framing |

## Contributing

Issues and PRs welcome. Start here:

- [`CONTRIBUTING.md`](CONTRIBUTING.md) - branch conventions, review etiquette, how to ship a PR
- [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md) - rules of engagement (Contributor Covenant 2.1)
- [`SECURITY.md`](SECURITY.md) - vulnerability disclosure policy

## License

[Apache-2.0](LICENSE). See [`NOTICE`](NOTICE) for third-party attributions.

---

⭐ **Find mnem useful?** A star is the strongest signal we get from a
satisfied builder - it helps the next agent developer find this repo
when they're stuck on memory. We read every issue, every PR, every
mention. Tell us what you built.
