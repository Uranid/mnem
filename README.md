<div align="center">

<img src="assets/logo/mnem-logo.svg" alt="mnem logo" width="140" height="140" />

# mnem

**Git for Knowledge Graphs**: versioned agent memory with hybrid GraphRAG retrieval. Runs entirely offline, no LLM required.

[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue?style=flat)](LICENSE)
[![CI](https://github.com/Uranid/mnem/actions/workflows/ci.yml/badge.svg)](https://github.com/Uranid/mnem/actions/workflows/ci.yml)
[![crates.io](https://img.shields.io/crates/v/mnem-cli.svg?style=flat)](https://crates.io/crates/mnem-cli)
[![PyPI](https://img.shields.io/pypi/v/mnem-py.svg?style=flat)](https://pypi.org/project/mnem-py/)
[![npm](https://img.shields.io/npm/v/mnem-cli.svg?style=flat)](https://www.npmjs.com/package/mnem-cli)
[![MSRV 1.95](https://img.shields.io/badge/MSRV-1.95-orange?style=flat)](rust-toolchain.toml)
[![Runs on Linux macOS Windows WASM](https://img.shields.io/badge/runs%20on-linux%20%7C%20macos%20%7C%20windows%20%7C%20wasm-2ea44f?style=flat)](#install)

</div>

---

**[What it is](#what-it-is)** · **[Install](#install)** · **[Quickstart](#quickstart)** · **[Integrate](#mnem-integrate---wire-into-any-agent-host)** · **[Commands](#commands)** · **[Python API](#python-api-mnem-py)** · **[GraphRAG](#graphrag)** · **[Benchmarks](#benchmarks)** · **[vs others](#compared-to-others)** · **[Docs](#documentation)** · **[Contributing](#contributing)**

---

## What it is

**A content-addressed knowledge graph with hybrid GraphRAG retrieval, versioned commits, and deterministic ingest, built as a persistent memory substrate for AI agents.**

Every node carries a cryptographic identity derived from DAG-CBOR + BLAKE3: the same content produces the same CID on any machine. Retrieval fuses vector (HNSW), sparse (BM25/SPLADE), and multi-hop graph traversal via RRF in a single pass, and every response reports exactly what candidates were seen and what got dropped at your token budget. Ingest is LLM-free. Single binary. No cloud. Compiles to `wasm32`.

## What you get

**mnem is strongest when:**
- facts accumulate across many sessions and you need to reason over history
- queries require multi-hop traversal ("how does X relate to Y")
- ingest must be deterministic and auditable - same bytes, same CIDs
- deployment is edge, offline, or WASM (no network, no daemon required)
- multiple agents write independently and need to merge without conflicts

| | Meaning |
|:--:|:--------|
| <img src="assets/legend/unique.svg" width="18" height="18" alt="unique"> | **unique** - not available in any other agent-memory system today |
| <img src="assets/legend/rare.svg" width="18" height="18" alt="rare"> | **rare** - available in 1-2 peers, often gated behind paid tiers |
| - | standard capability, done well |

1. <img src="assets/legend/unique.svg" width="14" height="14" alt="unique"> **Versioned + 3-way mergeable**. Commits, branches, diff, log, three-way merge, signed Ed25519 history. Two agents writing the same scope offline reconcile by graph + embedding merge, not "last write wins". → [Core concepts](docs/src/core-concepts.md)

2. <img src="assets/legend/unique.svg" width="14" height="14" alt="unique"> **Content-addressed objects**. Every node / tree / sidecar / commit has a CID derived from canonical DAG-CBOR + BLAKE3. Identical content collapses across machines. Determinism + replay become real, not a slogan. Peers use opaque UUIDs. → [Core concepts](docs/src/core-concepts.md)

3. <img src="assets/legend/unique.svg" width="14" height="14" alt="unique"> **Token-budget transparency**. Every retrieve emits `tokens_used`, `candidates_seen`, `dropped` counters. No silent truncation. No other agent-memory system exposes this as first-class response fields.

4. <img src="assets/legend/unique.svg" width="14" height="14" alt="unique"> **WASM-clean core**. `mnem-core` has no tokio, no filesystem, no network. Same retrieval logic compiles unchanged to `wasm32` - runs in Chrome, on Cloudflare Workers, on Lambda cold-start. Graphiti + mem0 are Python + external DB stacks; they cannot ship to the edge.

5. <img src="assets/legend/unique.svg" width="14" height="14" alt="unique"> **Skills as graphs, not markdown**. Today, agent skills live in flat `.md` files - downloaded, pasted into prompts, hand-edited, never queried. mnem promotes them to a versioned, queryable, mergeable graph. Export your graph, import someone else's, diff the two, merge the parts you want.

6. <img src="assets/legend/rare.svg" width="14" height="14" alt="rare"> **Best-in-class retrieval recall**. Beats open-source peers on LoCoMo (+0.218 R@5), ConvoMem (+0.047), and MemBench (+0.120) under the same embedder; matches MemPalace on LongMemEval (R@5 0.966). Numbers reproducible with the shipped harness. → [Benchmarks](benchmarks/README.md)

7. <img src="assets/legend/rare.svg" width="14" height="14" alt="rare"> **Plug-and-play**. Bundled ONNX MiniLM-L6-v2 runs in-process. No Ollama, no API keys, no cold-start network call. `mnem init` and you're retrieving. mem0 + Graphiti both require an external LLM endpoint at ingest. → [Install](docs/src/install.md)

8. <img src="assets/legend/rare.svg" width="14" height="14" alt="rare"> **Single binary**. ~40 MB Docker image. Embedded redb store. No daemon, no cloud, no account. Runs offline.

9. <img src="assets/legend/rare.svg" width="14" height="14" alt="rare"> **Deterministic ingest**. No LLM at ingest. parse + chunk + extract is statistical (KeyBERT optional), so same bytes in produces same CIDs out. Audit-friendly, fuzz-tested, byte-identical across machines. → [Ingest pipeline](docs/src/guides/ingest.md)

10. <img src="assets/legend/rare.svg" width="14" height="14" alt="rare"> **Swappable providers**. Embedder, sparse encoder, reranker, and LLM all set via config strings. Switch local ONNX to hosted Cohere with one flag. No fork, no rebuild. Most peers ship single-path stacks; provider swap is architectural here. → [Embedding providers](docs/src/guides/embed-providers.md)

11. <img src="assets/legend/rare.svg" width="14" height="14" alt="rare"> **Four surfaces, one core**. CLI, HTTP, MCP, and Python all wrap the same engine. `mnem integrate` wires the MCP server into Claude Desktop and other hosts. → [CLI reference](docs/src/cli.md) | [MCP](docs/src/mcp.md)

12. <img src="assets/legend/rare.svg" width="14" height="14" alt="rare"> **Property + fuzz tests**. Parsers are property-tested + fuzz-harnessed; CAR round-trip and merge-commit are byte-identical. Trust signal usually only seen at the infra-DB tier.

13. **Hybrid GraphRAG retrieval**. Vector (HNSW) + sparse (BM25 / SPLADE) + graph traversal, fused via RRF. GraphRAG built in and optional: on for multi-hop, off when dense saturates.

---

## Install

> [!NOTE]
> `--features bundled-embedder` ships an in-process ONNX MiniLM-L6-v2 so `mnem retrieve` works with zero configuration. Omit the flag if you want to bring your own embedder (Ollama, OpenAI, Cohere) via `.mnem/config.toml`.

<details>
<summary><b>macOS / Linux</b></summary>

No Cargo? [Install via rustup](https://rustup.rs/) (also installs `rustc`).

```bash
# C++ stdlib required to link the bundled ONNX Runtime (Linux only)
sudo apt-get install g++          # Debian / Ubuntu / WSL
# sudo dnf install gcc-c++        # Fedora / RHEL
```

```bash
cargo install --locked mnem-cli --features bundled-embedder

# CUDA-accelerated embedder (Linux, NVIDIA GPU)
cargo install --locked mnem-cli --features bundled-embedder-cuda
```

If `mnem` is not found after install, `~/.cargo/bin` is not on `$PATH`.

**rustup install** — source the env (or open a new terminal):
```bash
source ~/.cargo/env
```

**System Rust (apt/dnf)** — add to PATH permanently:
```bash
echo 'export PATH="$HOME/.cargo/bin:$PATH"' >> ~/.bashrc && source ~/.bashrc
```

</details>

<details>
<summary><b>Windows</b></summary>

No Cargo? [Install via rustup](https://rustup.rs/) (also installs `rustc`).

```powershell
cargo install --locked mnem-cli --features bundled-embedder

# DirectML-accelerated embedder (any GPU vendor on Windows)
cargo install --locked mnem-cli --features bundled-embedder-directml
```

</details>

<details>
<summary><b>npm / Node.js</b></summary>

No npm? [Install Node.js](https://nodejs.org/en/download) (npm is bundled, Node 18+ required).

```bash
npm install -g mnem-cli
mnem --version

# or without a global install (one-shot)
npx mnem-cli --version
```

Downloads the prebuilt native binary for your platform at install time. Node 18+ required. Bundled embedder included - no Ollama or API key needed.

</details>

<details>
<summary><b>Python (PyPI)</b></summary>

No pip? [Install Python](https://www.python.org/downloads/) (pip is bundled with Python 3.4+).

```bash
pip install mnem-cli
mnem --version
```

Ships the `mnem` binary as a manylinux / macOS / Windows wheel with the bundled embedder pre-baked.

</details>

<details>
<summary><b>Docker</b></summary>

No Docker? [Install Docker Desktop](https://docs.docker.com/get-started/get-docker/).

```bash
docker run --rm -p 9876:9876 ghcr.io/uranid/mnem:latest http serve
```

The image includes the bundled embedder. Run `mnem mcp` inside the container for the MCP server surface.

</details>

<details>
<summary><b>From source</b></summary>

```bash
# C++ stdlib required to link the bundled ONNX Runtime (Linux only)
sudo apt-get install g++          # Debian / Ubuntu / WSL
# sudo dnf install gcc-c++        # Fedora / RHEL
```

```bash
git clone https://github.com/Uranid/mnem
cd mnem
cargo install --path crates/mnem-cli --features bundled-embedder
```

Requires Rust 1.95+. If needed: `rustup install 1.95 && rustup default 1.95`.

</details>

```bash
mnem --version
mnem doctor        # checks embedder + store + config, prints a green/yellow/red checklist
```

Full install matrix: [`docs/src/install.md`](docs/src/install.md).

---

## Quickstart

```bash
mkdir my-graph && cd my-graph
mnem init
mnem ingest README.md
mnem retrieve "what does this project do"
```

Five minutes from zero. See [`docs/src/quickstart.md`](docs/src/quickstart.md) for the full walkthrough.

---

## `mnem integrate` - wire into any agent host

One command wires the **MCP server entry**, the **UserPromptSubmit hook** (for hosts that support it), and the **mnem system prompt** into the host's project-rules file. Restart the host and the agent starts using mnem automatically.

```bash
mnem integrate                           # interactive: detect installed hosts and prompt
mnem integrate claude-code               # wire a specific host, skip interactive detection
mnem integrate --all                     # wire every detected host without prompting

mnem integrate --check                   # report wired state for all hosts; nothing changes
mnem integrate --dry-run                 # preview what would be written without changing anything
mnem integrate --show claude-code        # print the MCP JSON block for manual copy-paste

mnem integrate --no-hooks                # skip UserPromptSubmit hook wiring
mnem integrate --no-system-prompt        # skip system prompt wiring
mnem integrate --target-repo ~/notes     # point the MCP server at a specific graph, not the global one
```

**What gets wired:**
- **MCP server** (`mcpServers.mnem`) - the agent gets full mnem tool access via `mnem mcp --repo <graph>`; defaults to the global graph (`~/.mnemglobal/.mnem`)
- **UserPromptSubmit hook** (Claude Code only) - runs `mnem retrieve` before each message, auto-injecting relevant memory into context
- **System prompt** - mnem usage instructions injected into the host's project-rules file

The hook always queries your project's `.mnem/` first (walking up from the current directory), then falls back to `mnem global retrieve` automatically. The hook and system prompt behave the same regardless of which default knowledge graph you choose during setup. Use `--target-repo` only if you want the MCP server to point somewhere other than the global graph.

Auto-detects and configures:
- Claude Code
- Claude Desktop
- Cursor
- Continue
- Zed
- Gemini CLI

Any other MCP-aware host works via a hand-edited `mcpServers` entry pointing at `mnem mcp --repo <path>` - see [`docs/src/mcp.md`](docs/src/mcp.md).

The agent gets the full mnem toolset as native tools: retrieve, commit, ingest, tombstone, traverse, global graph access, and more. No extra daemon, no port to manage. Full tool reference: [`docs/src/mcp.md`](docs/src/mcp.md).

---

## Commands

Every command accepts `--help` for the full flag reference.

### Init and health

```bash
mnem init      # create a new graph in the current directory
mnem doctor    # probe embedder + store + config; green/yellow/red checklist
mnem stats     # nodes, edges, refs, embedder health, repo size
```

### Adding knowledge

```bash
mnem ingest notes.md                        # parse a file into Doc + Chunk + Entity nodes
mnem ingest --recursive docs/               # ingest a directory recursively
mnem ingest --chunker recursive report.pdf  # PDF with sliding-window chunking
```

```bash
mnem add node -s "Alice leads the infra team"                       # label defaults to "Node"
mnem add node --label Fact -s "Alice leads the infra team"          # add a single fact node
mnem add edge --from <uuid> --to <uuid> --label works_at            # connect two nodes
```

> The ingest pipeline is deterministic: no LLM at ingest time, same bytes in always produce the same CIDs out. Audit-friendly and fuzz-tested.

### Retrieving knowledge

```bash
mnem retrieve "what did we decide about the API design"  # searches local .mnem/ first, falls back to global
mnem -R ~/notes retrieve "query"                         # target a specific graph explicitly
```

`-R <path>` is a global flag that redirects any command to a specific repository directory. It overrides the walk-up search from the current directory and any default set via `mnem integrate`. Applies to all subcommands: `mnem -R ~/notes status`, `mnem -R ~/notes log`, etc.

Hybrid retrieval: vector (HNSW) + sparse (BM25/SPLADE) + graph traversal, fused via RRF. See [GraphRAG](#graphrag) for tuning flags.

### The global graph

> [!NOTE]
> mnem has two scopes: the **local graph** (`.mnem/` in your project directory) and the **global graph** (`~/.mnemglobal/.mnem/`). The global graph is for cross-project, cross-session facts that should follow you everywhere.

**When to use local vs global:**

| Use local `.mnem/` for | Use `mnem global` for |
|------------------------|----------------------|
| Project-specific facts, decisions, code context | People, preferences, facts that span all projects |
| Per-repo memory that travels with the repo | Knowledge you want every session and every agent to see |
| Anything you'd commit alongside the code | Cross-session continuity |

`mnem global` is a full mirror of `mnem` but operates exclusively on the global graph:

```bash
mnem global retrieve "what is Alice's current role"     # search the global graph only
mnem global ingest contacts.md                          # ingest a file into the global graph
mnem global add node --label Entity:Person \
  --prop name=Alice -s "Alice leads the infra team"     # add a node to the global graph
```

The `mnem integrate` command sets up the agent to read local first and fall back to global automatically - no manual switching required during normal use.

### Status and inspection

```bash
mnem status           # op-head CID, head commit, all named refs, label counts, MERGING marker
mnem stats            # one-line: op, commit, content CID, ref count, label names
```

### History

```bash
mnem log              # walk op-log backwards from HEAD, default last 20 entries
mnem log -n 50        # show last 50 entries
mnem log --oneline    # compact one-line-per-op format
mnem log --format json # machine-readable JSON stream

mnem show             # decode and pretty-print the current op-head block
mnem show <cid>       # decode any block by CID (Node, Edge, Commit, Operation, View, ...)

mnem diff <op-a-cid> <op-b-cid>   # ref deltas + node/edge structural diff between two ops
mnem diff HEAD <cid>               # diff current op against a specific op CID
```

### Branching and merging

```bash
mnem branch list                        # list all refs/heads/* branches; * marks current
mnem branch create <name>               # create branch at current HEAD
mnem branch create <name> <start>       # branch from a ref name, branch name, or CID
mnem branch create <name> --from HEAD   # explicit --from form; same resolution as above
mnem branch delete <name>               # delete a local branch pointer

mnem merge <branch>                     # 3-way merge <branch> into current HEAD
mnem merge <branch> --strategy=ours     # auto-resolve conflicts: keep current side
mnem merge <branch> --strategy=theirs   # auto-resolve conflicts: take incoming side
mnem merge <branch> --dry-run           # preview outcome without persisting anything
mnem merge --continue                   # finish after editing .mnem/MERGE_CONFLICTS.json
mnem merge --abort                      # cancel, restore HEAD from .mnem/ORIG_HEAD

mnem pull                               # fast-forward origin/main into HEAD (default)
mnem pull <remote> <branch>             # fast-forward <remote>/<branch> into HEAD
```

### Remote operations

```bash
mnem remote add <name> <url>            # register a remote (stores in .mnem/config.toml)
mnem remote add <name> <url> \
  --token-env MNEM_REMOTE_ORIGIN_TOKEN  # name the env var that holds the bearer token
mnem remote list                        # list all configured remotes with their URLs
mnem remote show <name>                 # show URL + capabilities for one remote
mnem remote remove <name>               # remove a remote entry

mnem fetch                              # fetch from origin (default)
mnem fetch <remote>                     # fetch from a named remote; token via env var

mnem push                               # push HEAD to origin/main (default)
mnem push <remote> <branch>             # push a specific branch to a named remote

mnem clone <url> [<dir>]                # clone a CAR archive into <dir>; file:// and bare .car paths supported
mnem clone file:///tmp/repo.car ./copy  # clone from a local file URL
mnem clone ./repo.car ./copy            # bare path shorthand (must end in .car)
```

### Query and graph traversal

```bash
mnem query --where name=Alice                    # exact property match, default 10 results
mnem query --where kind=Person -n 25             # increase result limit
mnem query --where kind=Person \
  --with-outgoing knows                          # match nodes + follow outgoing "knows" edges
mnem query --where status=active \
  --with-outgoing depends_on \
  --with-outgoing depends_on                     # repeat --with-outgoing to chain hops

mnem blame <node-uuid>                           # list all incoming edges to a node
mnem blame <node-uuid> --etype authored          # filter to one edge type
```

### Named refs

```bash
mnem ref list                         # list all refs (refs/heads/*, refs/remotes/*, ...)
mnem ref set <name> <target-cid>      # point a ref at a specific commit CID
mnem ref delete <name>                # delete a named ref
```

### Embeddings

```bash
mnem embed                            # backfill embeddings for every node missing a vector
mnem embed --force                    # re-embed even nodes that already have a vector
mnem embed --label Person             # restrict to nodes of one label
mnem embed --dry-run                  # count what would be embedded without calling the provider

mnem reindex                          # alias for embed; preferred name after C7 rename
mnem reindex --label Doc              # restrict to one label
mnem reindex --since <commit>         # only nodes added/changed after <commit>
mnem reindex --force                  # re-embed already-indexed nodes
mnem reindex --dry-run                # count without calling the provider
```

### Low-level block access

```bash
mnem cat-file <cid>          # emit raw DAG-CBOR bytes for a block to stdout
mnem cat-file <cid> --json   # decode to DAG-JSON and pretty-print (pipe into jq)
```

### Export and import

```bash
mnem export <path>                        # export HEAD as a CAR v1 archive
mnem export -                             # write CAR to stdout (pipe over SSH etc.)
mnem export --from refs/heads/main out.car  # export from a specific ref
mnem export --from <cid> backup.car       # export from a specific commit CID

mnem import <path>                        # import a CAR archive into the current repo
mnem import -                             # read CAR from stdin
```

### Configuration

```bash
mnem config set user.name Alice           # set author name
mnem config set user.email alice@example.com
mnem config set embed.provider ollama     # embedder: openai | ollama
mnem config set embed.model nomic-embed-text
mnem config set embed.base_url http://localhost:11434  # override provider endpoint
mnem config get embed.provider            # print the current value of a key
mnem config unset embed.provider          # remove a key
mnem config list                          # print all set keys and their values
```

Known keys: `user.name`, `user.email`, `user.key`, `user.agent_id`, `embed.provider`, `embed.model`, `embed.api_key_env`, `embed.base_url`. API keys live in environment variables, never in config.

### Repository registry

```bash
mnem repos list              # list all repos registered with mnem integrate
mnem repos set-default <path>  # mark a repo as the default for mnem without -R
mnem repos prune             # remove registry entries for paths that no longer exist
```

### Servers

```bash
mnem mcp                       # start the MCP JSON-RPC server over stdio
mnem mcp --repo ~/notes        # point the MCP server at a specific graph
mnem http serve                # start the HTTP JSON API (loopback by default)
```

### Benchmarks

```bash
mnem bench                                       # interactive TUI; select benchmarks to run
mnem bench run --benches longmemeval --limit 50  # run a specific benchmark suite
mnem bench fetch longmemeval                     # download benchmark datasets
mnem bench results ./bench-out                   # re-render results from a prior run
```

### Shell completions

```bash
mnem completions bash        # emit bash completion script
mnem completions zsh         # zsh
mnem completions fish        # fish
mnem completions powershell  # PowerShell
mnem completions elvish      # Elvish

# Install (bash):
mnem completions bash > ~/.local/share/bash-completion/completions/mnem
# Install (zsh):
mnem completions zsh > ~/.zsh/completions/_mnem
```

Full CLI reference: [`docs/src/cli.md`](docs/src/cli.md).

---

## Python API (mnem-py)

Use `mnem-py` when you want to read and write a mnem graph directly from Python - without the CLI binary. Same retrieval engine, PyO3 bindings.

```bash
pip install mnem-py
pip install sentence-transformers   # brings ~200 MB of deps (torch, transformers)
```

`mnem-py` stores and retrieves by **dense vector**: you compute embeddings in Python and hand them to mnem. `SentenceTransformer("all-MiniLM-L6-v2")` downloads a ~23 MB model from HuggingFace Hub on first use and caches it in `~/.cache/huggingface/` - all subsequent calls are fully local with no network required.

```python
import pymnem
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")   # downloaded once, ~23 MB
MODEL_NAME = "all-MiniLM-L6-v2"                    # key mnem uses to match stored vectors

repo = pymnem.Repo.init_memory()                    # in-memory; open_or_init() for disk

# Write: compute an embedding for each node and attach it
with repo.transaction(author="agent", message="seed") as tx:
    for text in ["Alice lives in Berlin", "Bob moved to Paris"]:
        tx.add_node(ntype="Memory", summary=text)
        tx.add_embedding_f32(MODEL_NAME, model.encode(text).tolist())

# Retrieve: compute a query vector with the same model, mnem ranks under token budget
query_vec = model.encode("Alice Berlin").tolist()
result = repo.retrieve(vector=query_vec, model=MODEL_NAME, token_budget=500, limit=5)
for item in result:
    print(f"{item.score:.3f}  {item.summary}")
# result.tokens_used / result.tokens_budget  - no silent truncation
```

Full API surface - `query`, `update_node`, `delete_node`, on-disk persistence, label filtering: [`crates/mnem-py/README.md`](crates/mnem-py/README.md).

---

## GraphRAG

mnem ships GraphRAG built in. One knob per stage, opt-in per query, never required. Vector search alone handles most queries well - turn graph stages on when queries span multiple documents, require multi-hop reasoning, or need compositional answers.

### Stages and flags

| Stage | Flag | What it does |
|-------|------|------|
| **Vector lane** | always on | HNSW over per-commit dense embeddings (default 384-d MiniLM). |
| **Sparse lane** | config-driven | BM25 + SPLADE-onnx, fused with vector via Reciprocal Rank Fusion. Toggled by `[sparse]` block in `config.toml`. |
| **Vector candidate pool** | `--vector-cap <N>` | Lift the dense pool size from default 256. Higher = better long-tail recall, +cost. |
| **Result limit** | `--limit <N>` | Final returned set (default 10). Short form: `-n`. |
| **Graph expansion** | `--graph-expand <N>` | Add N neighbours of top-K seeds via authored edges. Audit-recommended default `20` when graph is on. |
| **Graph mode** | `--graph-mode <decay\|ppr>` | `decay` (default) = exponential weight by hop. `ppr` = Personalised PageRank over the hybrid adjacency index, paper-grade scoring for multi-hop. |
| **Community filter** | `--community-filter` | Run Leiden community detection; drop low-coverage communities before fusion. Default coverage threshold: `0.5`. |
| **KeyBERT extraction** | `mnem ingest --extractor keybert` | Ingest-time keyphrase enrichment. Strengthens sparse + community signals. Pass at ingest, not retrieve. |
| **Summarisation** | `--summarize` | Centroid + MMR summary of the top-K, with diversity. |
| **Cross-encoder rerank** | `--rerank <provider:model>` | Post-fusion reorder. Supports `cohere:rerank-english-v3.0`, `voyage:rerank-1`, local. |

### Quick examples

```bash
# Dense baseline
mnem retrieve "what does this project do"

# Add multi-hop graph traversal
mnem retrieve "..." --graph-expand 20

# Full stack: graph-expand + community-filter + PPR + rerank
mnem retrieve "..." --graph-expand 20 --community-filter --graph-mode ppr --rerank cohere:rerank-english-v3.0

# Stack a cross-encoder reranker on top
mnem retrieve "..." --graph-expand 20 --community-filter --rerank cohere:rerank-english-v3.0

# Ingest with KeyBERT keyphrase enrichment (strengthens sparse + community signals)
mnem ingest --extractor keybert notes.md
```

### When to enable

- **Single-document corpus, simple queries**: leave graph off, vector search alone is enough
- **Multi-hop / compositional questions**: `--graph-expand 20`
- **Long history with cross-document references**: add `--community-filter`
- **Recall ceiling needed**: stack `--rerank` on top
- **Keyphrase-enriched ingest**: `mnem ingest --extractor keybert` at ingest time

Full retrieval architecture: [`docs/src/cli.md`](docs/src/cli.md) (retrieve flags)

---

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

| Benchmark | Split | Metric | MP | mnem | Delta |
|-----------|-------|--------|----|-----------|-------|
| LongMemEval | 500 Q | R@5 session | 0.966 | **0.966** | 0 |
| LongMemEval | 500 Q | R@10 session | 0.982 | **0.982** | 0 |
| LoCoMo | 1986 Q | R@5 session | 0.508 | $\color{green}{\textbf{0.726}}$ | **+0.218** |
| LoCoMo | 1986 Q | R@10 session | 0.603 | $\color{green}{\textbf{0.855}}$ | **+0.252** |
| ConvoMem | 250 (5x50) | avg recall | 0.929 | $\color{green}{\textbf{0.976}}$ | **+0.047** |
| MemBench | simple/roles 100 | R@5 | 0.840 | $\color{green}{\textbf{0.960}}$ | **+0.120** |
| MemBench | highlevel/movie 100 | R@5 | 0.950 | $\color{green}{\textbf{1.000}}$ | **+0.050** |
| LongMemEval | 500 Q hybrid-v4 | R@5 session | 0.982 | $\color{red}{\textbf{0.976}}$ | **-0.006** |

### vs mem0

> mem0 doesn't publish recall@K headlines on these datasets, so both
> columns are our reproductions: we ran mem0's adapter end-to-end under
> the same harness, same embedder bytes (ONNX MiniLM-L6-v2), and the
> same per-item scoping (`infer=False` + `user_id`-per-item) documented
> in [`benchmarks/results/methodology.md`](benchmarks/results/methodology.md).
> Both columns reproducible; raw artefacts in
> [`benchmarks/proofs/v0.1.0/`](benchmarks/proofs/v0.1.0/).

| Benchmark | Split | Metric | mem0 | mnem | Delta |
|-----------|-------|--------|------|-----------|-------|
| LongMemEval | 500 Q | R@5 session | 0.946 | $\color{green}{\textbf{0.966}}$ | **+0.020** |
| LongMemEval | 500 Q | R@10 session | 0.962 | $\color{green}{\textbf{0.982}}$ | **+0.020** |
| LoCoMo | 1986 Q | R@5 session | 0.466 | $\color{green}{\textbf{0.726}}$ | **+0.260** |
| LoCoMo | 1986 Q | R@10 session | 0.676 | $\color{green}{\textbf{0.855}}$ | **+0.179** |
| ConvoMem | 250 (5x50) | avg recall | 0.558 | $\color{green}{\textbf{0.976}}$ | **+0.418** |
| MemBench | simple/roles 100 | R@5 | 0.410 | $\color{green}{\textbf{0.960}}$ | **+0.550** |
| MemBench | highlevel/movie 100 | R@5 | 0.970 | $\color{green}{\textbf{1.000}}$ | **+0.030** |
| LongMemEval | 500 Q hybrid-v4 | R@5 session | 0.930 | $\color{green}{\textbf{0.976}}$ | **+0.046** |

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
mnem bench fetch longmemeval     # download datasets (one-time, 264 MB)
mnem bench                       # TUI; select benchmarks interactively
mnem bench run --benches longmemeval --limit 50 --non-interactive
mnem bench results ./bench-out   # re-render results from a prior run

# Legacy bash harness (canonical path for headline numbers)
bash benchmarks/harness/run_bench.sh
```

Methodology, raw artifacts, per-bench breakdowns: [`benchmarks/`](benchmarks/) and [`docs/src/benchmarks/`](docs/src/benchmarks/).

---

## Compared to others

- [mnem vs mem0](docs/src/comparisons/mem0.md) - agent memory layer, OSS leader
- [mnem vs MemPalace](docs/src/comparisons/mempalace.md) - methodology peer
- [mnem vs Supermemory](docs/src/comparisons/supermemory.md) - closed-cloud incumbent
- [mnem vs Cognee](docs/src/comparisons/cognee.md) - KG-for-agents alternative
- [mnem vs Letta](docs/src/comparisons/letta.md) - agent-memory framework
- [mnem vs graphify](docs/src/comparisons/graphify.md) - lightweight graph tool

Full matrix: [`docs/src/comparisons/README.md`](docs/src/comparisons/README.md).

---

## When NOT to use mnem

- **You need transactional OLTP.** mnem is append-only with versioned history; row-level UPDATE/DELETE semantics aren't the model.
- **You need sub-50 ms cloud-scale retrieval at 10k+ QPS.** mnem is local-first. Multi-region sharded retrieval is on the roadmap, not in v1.

> Looking for hosted memory, multi-region replicas, shared graphs across teams, or a managed remote layer? A sibling project bringing those to mnem is in active development - watch this space.

---

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
| [`mnem-extract`](crates/mnem-extract) | entity extraction (KeyBERT, statistical NER) |
| [`mnem-ner-providers`](crates/mnem-ner-providers) | NER provider trait + built-in providers (`RuleNer`, `NullNer`) |
| [`mnem-bench`](crates/mnem-bench) | benchmark harness (LongMemEval, LoCoMo, etc.) |
| [`mnem-graphrag`](crates/mnem-graphrag) | community summarisation, centroid + MMR |
| [`mnem-ann`](crates/mnem-ann) | HNSW wrapper |
| [`mnem-backend-redb`](crates/mnem-backend-redb) | redb-backed store |
| [`mnem-transport`](crates/mnem-transport) | CAR codec + remote framing |

---

## Documentation

- [Quickstart](docs/src/quickstart.md) - five-minute walkthrough
- [Install](docs/src/install.md) - per-platform install matrix
- [CLI reference](docs/src/cli.md) - every subcommand and flag
- [MCP server](docs/src/mcp.md) - tools exposed, client wiring
- [Core concepts](docs/src/core-concepts.md) - CIDs, commits, labels
- [Configuration](docs/src/configuration.md) - env vars, config.toml
- [Benchmarks methodology](docs/src/benchmarks/methodology.md)
- [Reproduce benchmarks](docs/src/benchmarks/reproduce.md)
- [Retrieval tuning](docs/src/guides/retrieval-tuning.md)
- [Embedding providers](docs/src/guides/embed-providers.md)
- [Migrations](docs/src/migrations/)

---

## Contributing

Issues and PRs welcome. Start here:

- [`CONTRIBUTING.md`](CONTRIBUTING.md) - branch conventions, review etiquette, how to ship a PR
- [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md) - rules of engagement (Contributor Covenant 2.1)
- [`SECURITY.md`](SECURITY.md) - vulnerability disclosure policy

## License

[Apache-2.0](LICENSE). See [`NOTICE`](NOTICE) for third-party attributions.

---

## Unintegrate / remove

```bash
mnem unintegrate                  # interactive: pick which hosts to remove mnem from
mnem unintegrate claude-code      # remove one host
mnem unintegrate --all            # remove all wired hosts
```

Run `mnem unintegrate --help` for all options.

---

⭐ **Find mnem useful?** A star is the strongest signal we get from a satisfied builder - it helps the next agent developer find this repo when they're stuck on memory. We read every issue, every PR, every mention. Tell us what you built.
