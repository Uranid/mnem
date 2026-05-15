<div align="center">

<img src="assets/logo/mnem-banner.svg" alt="mnem: Git for AI Memory" />

[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue?style=for-the-badge)](LICENSE)
[![CI](https://img.shields.io/github/actions/workflow/status/Uranid/mnem/ci.yml?style=for-the-badge&label=CI)](https://github.com/Uranid/mnem/actions/workflows/ci.yml)
[![crates.io](https://img.shields.io/crates/v/mnem-cli?style=for-the-badge)](https://crates.io/crates/mnem-cli)
[![PyPI](https://img.shields.io/pypi/v/mnem-cli?style=for-the-badge)](https://pypi.org/project/mnem-cli/)
[![npm](https://img.shields.io/npm/v/mnem-cli?style=for-the-badge)](https://www.npmjs.com/package/mnem-cli)
[![MSRV 1.95](https://img.shields.io/badge/MSRV-1.95-orange?style=for-the-badge)](rust-toolchain.toml)
[![Runs on Linux macOS Windows WASM](https://img.shields.io/badge/runs%20on-linux%20%7C%20macos%20%7C%20windows%20%7C%20wasm-2ea44f?style=for-the-badge)](#install)

</div>

<div align="center">

[English](README.md) &nbsp;·&nbsp; [中文](README.zh-CN.md) &nbsp;·&nbsp; [Español](README.es.md)

</div>

<hr>

<div align="center">

https://github.com/user-attachments/assets/bd744a7e-8e89-4531-bd96-fdee0030c390

</div>

<hr>

1. [The Problem](#the-problem)
2. [What is mnem](#what-is-mnem)
3. [Benchmarks](#benchmarks)
4. [What you get](#what-you-get)
5. [Install](#install)
6. [Quickstart](#quickstart)
7. [Integrate / Unintegrate](#mnem-integrate---wire-into-any-agent-host)
8. [Commands](#commands)
9. [MCP Tools](#mcp-tools)
10. [Python API](#python-api-mnem-py)
11. [GraphRAG](#graphrag)
12. [vs others](#compared-to-others)
13. [When NOT to use](#when-not-to-use-mnem)
14. [Docs](#documentation)
15. [Contributing](#contributing)

<hr>

## The Problem

> **Every session starts from zero.**

- **Sessions are isolated.** Plan a migration in Claude Code. Open Cursor tomorrow. That agent has never heard of it.
- **Memory you can't inspect isn't memory.** Something changed in your agent's context. You don't know what, when, or why. There's no log.
- **Conventions rot in flat files.** Six engineers, six `AGENTS.md` files diverging in silence. No merge, no history, no way to tell which is current.

> Your codebase has git. Your agent's knowledge doesn't.

<hr>

## What is mnem

**Git for AI memory.** A persistent, versioned memory layer for AI systems - with best retrieval recall on every public benchmark.

mnem gives agents the same model developers use for code. Every write is a commit - branch to experiment, merge when ready, diff to see what changed, revert a bad batch, blame to trace what each agent wrote. Skills, decisions, and context live in a queryable graph that travels with your repo. Replace stale `.cursorrules` and `AGENTS.md` files with something your whole team can version, diff, and merge.

Retrieval is transparent - vector, keyword, and graph search in one pass - with an explicit token budget so nothing gets silently dropped. One binary, no server to run. Wire into Claude Code, Cursor, Gemini CLI, or any MCP host in one command; use it from the CLI, HTTP, or Python. The same core compiles to WASM for browser and edge deployment.

> **For teams:** Commit `.mnem/` alongside your code. CI agents push findings after each build; developer agents pull at session start - every engineer and agent works from the same knowledge baseline.

## Benchmarks

**Measured head-to-head against mem0 and MemPalace on six public datasets. mnem leads on all of them.**

> **Methodology:** mem0 numbers are our own reproduction under the same harness - mem0 does not publish R@K headline scores on these datasets. MemPalace headline numbers are cross-verified under our harness. This is disclosed, not hidden: reproducible artifacts ship alongside the binary.

ONNX MiniLM-L6-v2 embedder, same bytes on every system. No LLM rerank. Reproduce: `bash benchmarks/harness/run_bench.sh`.

<div align="center"><img src="assets/benchmarks/benchmarks.svg" alt="mnem public benchmarks" /></div>

<sup>mem0 columns: our reproduction under the same harness (mem0 doesn't publish R@K headlines on these datasets). MemPalace columns: public headline numbers cross-verified under our harness. Raw artefacts: [`benchmarks/results/v0.1.0/`](benchmarks/results/v0.1.0/). † FinanceBench uses Ollama bge-large (1024-dim) on all systems; MemPalace shown at best configuration (bge-large direct ChromaDB); mem0 applies LLM memory extraction before storage. Full methodology: [`benchmarks/results/analysis/financebench.md`](benchmarks/results/analysis/financebench.md).</sup>

### Query speed

<div align="center"><img src="assets/benchmarks/query-speed.svg" alt="mnem query speed" /></div>

<details>
<summary><b>Reproduce</b></summary>

```bash
mnem bench fetch longmemeval     # download datasets (one-time, 264 MB)
mnem bench                       # TUI; select benchmarks interactively
mnem bench run --benches longmemeval --limit 50 --non-interactive
mnem bench results ./bench-out   # re-render results from a prior run

# Legacy bash harness (canonical path for headline numbers)
bash benchmarks/harness/run_bench.sh
```

Methodology, raw artifacts, per-bench breakdowns: [`benchmarks/`](benchmarks/) and [`docs/src/benchmarks/`](docs/src/benchmarks/).

</details>

<hr>

## What you get

<sup><img src="assets/legend/unique.svg" width="12" height="12" alt="unique"> unique to mnem &nbsp;·&nbsp; <img src="assets/legend/rare.svg" width="12" height="12" alt="rare"> rare among peers</sup>

| | | |
|:---:|:---|:---|
| <img src="assets/legend/unique.svg" width="18" height="18" alt="unique"> | **Instantly build a knowledge graph from any file or codebase. Zero LLM.** Drop in source code, PDFs, Markdown docs, or conversation exports - mnem handles the rest. One command. 30+ formats. `Doc → Chunk → Entity` graph ready to query. | [READ MORE](docs/features/rich-ingest.md) |
| <img src="assets/legend/unique.svg" width="18" height="18" alt="unique"> | **Branch, diff, and merge knowledge like git.** Every write is a versioned commit. Experiment on a branch, merge when ready - your knowledge graph has the same primitives as your codebase. | [READ MORE](docs/features/versioned-memory.md) |
| <img src="assets/legend/unique.svg" width="18" height="18" alt="unique"> | **Replace flat agent files with a versioned, queryable graph.** `.cursorrules` and `AGENTS.md` can't be diffed or merged. mnem can - export yours, import a teammate's, merge the parts you want. | [READ MORE](docs/features/skills-graph.md) |
| <img src="assets/legend/unique.svg" width="18" height="18" alt="unique"> | **See exactly what retrieval found, skipped, and cost.** Every query returns `tokens_used`, `candidates_seen`, and `dropped`. No silent truncation at your token budget. | [READ MORE](docs/features/token-transparency.md) |
| <img src="assets/legend/unique.svg" width="18" height="18" alt="unique"> | **Same input, same output, any machine.** No two systems can diverge from the same data. Deterministic, replayable, and audit-friendly by design. | [READ MORE](docs/features/content-addressing.md) |
| <img src="assets/legend/unique.svg" width="18" height="18" alt="unique"> | **Runs in a browser tab.** Works in Chrome, Cloudflare Workers, and Lambda cold-start. No Python, no external database, no server. | [READ MORE](docs/features/wasm-edge.md) |
| <img src="assets/legend/rare.svg" width="18" height="18" alt="rare"> | **Best or tied recall on every public benchmark.** Beats open-source peers on LoCoMo, MemBench, and ConvoMem. All numbers reproducible with the shipped harness. | [READ MORE](docs/features/benchmarks.md) |
| <img src="assets/legend/rare.svg" width="18" height="18" alt="rare"> | **Zero-config start, any provider after.** Bundled ONNX MiniLM-L6-v2 runs in-process out of the box. Switch to Ollama, OpenAI, or Cohere with one line in `config.toml`. | [READ MORE](docs/features/providers.md) |
| <img src="assets/legend/rare.svg" width="18" height="18" alt="rare"> | **CLI, HTTP, MCP, and Python - one engine.** `mnem integrate` wires the MCP server into Claude Code, Cursor, Gemini CLI, and anything else speaking MCP. | [READ MORE](docs/features/integrations.md) |
| <img src="assets/legend/rare.svg" width="18" height="18" alt="rare"> | **One ~40 MB binary. Nothing else required.** No daemon, no cloud, no account. Runs fully offline. Same binary powers the CLI and the HTTP server. | [READ MORE](docs/features/single-binary.md) |
| <img src="assets/legend/rare.svg" width="18" height="18" alt="rare"> | **API-free, deterministic ingestion.** No LLM call at index time. Same file always produces identical nodes - fully reproducible and audit-friendly. Re-ingest an unchanged file and get zero new nodes. | [READ MORE](docs/features/deterministic-ingest.md) |
| | **Vector, keyword, and graph search in one pass.** Enable multi-hop traversal for queries that span documents; skip it for fast single-doc lookup. | [READ MORE](docs/features/hybrid-retrieval.md) |

<hr>

## Install

**Pick whichever one you already have. Any one works.** Full per-platform notes below.

```bash
# if you have Cargo (Rust): recommended for dev machines
cargo install --locked mnem-cli --features bundled-embedder

# if you have pip (Python)
pip install mnem-cli

# if you have npm (Node.js)
npm install -g mnem-cli
```

```bash
mnem --version    # confirm install
```

> [!NOTE]
> `--features bundled-embedder` ships an in-process ONNX MiniLM-L6-v2 so `mnem retrieve` works with zero configuration. Omit the flag if you want to bring your own embedder (Ollama, OpenAI, Cohere) via `.mnem/config.toml`.

<details>
<summary>Sample <code>.mnem/config.toml</code> (Ollama example)</summary>

```toml
[embed]
provider = "ollama"
model    = "nomic-embed-text"
base_url = "http://localhost:11434"
```

Full list of config keys: [`docs/src/configuration.md`](docs/src/configuration.md).

</details>

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

**rustup install**: source the env (or open a new terminal):
```bash
source ~/.cargo/env
```

**System Rust (apt/dnf)**: add to PATH permanently:
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
<summary><b>pip (PyPI)</b></summary>

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

> **Embedding mnem inside a Python app?** The `pip install mnem-cli` above ships the **CLI binary** as a wheel. The native **Python API** (`import mnem`) lives in a separate package. Jump to **[Python API (mnem-py) ↓](#python-api-mnem-py)** for `pip install mnem-py` and snippets.

<hr>

## Quickstart

```bash
mkdir my-graph && cd my-graph
mnem init                                    # creates .mnem/ with a fresh store
mnem ingest README.md                        # parses into chunks + entities; no LLM needed
mnem retrieve "what does this project do"    # returns ranked nodes + token budget report
```

Five minutes from zero. See [`docs/src/quickstart.md`](docs/src/quickstart.md) for the full walkthrough.

<hr>

## `mnem integrate` - wire into any agent host

One command wires the **MCP server entry**, the **UserPromptSubmit hook** (for hosts that support it), and the **mnem system prompt** into the host's project-rules file. Restart the host and the agent starts using mnem automatically.

```bash
mnem integrate                           # interactive: detect installed hosts and prompt
mnem integrate claude-code               # wire a specific host, skip interactive detection
mnem integrate --all                     # wire every detected host without prompting

mnem integrate --check                   # report wired state for all hosts; nothing changes
mnem integrate --dry-run                 # preview what would be written without changing anything
mnem integrate --show claude-code        # print the MCP JSON block for manual copy-paste
mnem integrate --show hermes             # print Hermes' YAML `mcp_servers` block

mnem integrate --no-hooks                # skip UserPromptSubmit hook wiring
mnem integrate --no-system-prompt        # skip system prompt wiring
mnem integrate --target-repo ~/notes     # point the MCP server at a specific graph, not the global one
```

**What gets wired:**
- **MCP server** (`mcpServers.mnem`; Hermes uses `mcp_servers.mnem`) - the agent gets full mnem tool access via `mnem mcp --repo <graph>`; defaults to the global graph (`~/.mnemglobal/.mnem`)
- **UserPromptSubmit hook** (Claude Code only) - runs `mnem retrieve` before each message, auto-injecting relevant memory into context
- **System prompt / skill** - mnem usage instructions injected into the host's project-rules file; Hermes receives a native skill at `~/.hermes/skills/mnem/SKILL.md`

The hook always queries your project's `.mnem/` first (walking up from the current directory), then falls back to `mnem global retrieve` automatically. The hook and system prompt behave the same regardless of which default knowledge graph you choose during setup. Use `--target-repo` only if you want the MCP server to point somewhere other than the global graph.

Auto-detects and configures:
- Claude Code
- Claude Desktop
- Cursor
- Continue
- Zed
- Hermes Agent
- Gemini CLI

Any other MCP-aware host works via a hand-edited `mcpServers` entry pointing at `mnem mcp --repo <path>` - see [`docs/src/mcp.md`](docs/src/mcp.md).

The agent gets the full mnem toolset as native tools: retrieve, commit, ingest, tombstone, traverse, global graph access, and more. No extra daemon, no port to manage. Full tool reference: [`docs/src/mcp.md`](docs/src/mcp.md).

<details>
<summary>Remove mnem from a host</summary>

```bash
mnem unintegrate                  # interactive: pick which hosts to remove mnem from
mnem unintegrate claude-code      # remove one host
mnem unintegrate --all            # remove all wired hosts
```

Run `mnem unintegrate --help` for all options.

</details>

<hr>

## Commands

Every command accepts `--help` for the full flag reference. Full CLI reference: [`docs/src/cli.md`](docs/src/cli.md).

---

### 1. `mnem init` - Initialize a knowledge graph

Create a `.mnem/` store in the current directory. Commit it alongside your codebase so every developer and agent starts from the same baseline.

```bash
mnem init
```

> **Example:** Your team ships an AI agent alongside an API service. Run `mnem init` once in the repo root - every engineer who clones the repo gets the same knowledge base their agents were trained on.

<details>
<summary>Health check and diagnostics</summary>

```bash
mnem doctor    # probes embedder, store, config - green/yellow/red checklist
mnem stats     # nodes, edges, refs, store size at a glance
```

</details>

---

### 2. `mnem ingest` - Add documents to the graph

Parse a file or directory into `Doc`, `Chunk`, and `Entity` nodes in a single pass. No LLM required at ingest - deterministic and audit-friendly: same bytes always produce the same CIDs.

```bash
mnem ingest architecture.md
```

> **Example:** An agent onboarding to your platform ingests `ARCHITECTURE.md`, the `runbooks/` directory, and all ADR files at startup. Every subsequent agent retrieves the same structured knowledge without re-reading each file from scratch.

<details>
<summary>More options</summary>

```bash
mnem ingest --recursive docs/               # ingest an entire directory
mnem ingest --chunker recursive report.pdf  # PDF with sliding-window chunking
mnem ingest --extractor keybert notes.md    # keyphrase enrichment for stronger sparse retrieval
```

</details>

---

### 3. `mnem add` - Write individual facts and relationships

Commit a single fact node, or connect two entities with a typed edge. The lowest-level write primitive - use it when you want precise control over what goes into the graph.

```bash
mnem add node --label Fact -s "The payments API uses idempotency keys for all POST requests"
```

> **Example:** Mid-conversation, an agent discovers an undocumented constraint. It commits the finding immediately so every downstream agent operates from the same shared truth - no more re-discovering the same edge case across sessions.

<details>
<summary>More write options</summary>

```bash
mnem add node -s "Deploy window is Tuesdays 10-11 AM UTC"         # label defaults to "Node"
mnem add edge --from <uuid> --to <uuid> --label depends_on        # connect two existing nodes
```

</details>

<details>
<summary>Read and delete nodes</summary>

```bash
mnem get <uuid>                                                    # fetch a node by UUID
mnem get <uuid> --content                                         # include full content body

mnem tombstone <uuid>                                             # soft-delete: hidden from retrieval, kept in audit log
mnem tombstone <uuid> --reason "superseded by v2 decision"        # record why
mnem delete <uuid>                                                # hard-delete: no audit trail

mnem global get <uuid>                                            # look up a node in the global graph
mnem global tombstone <uuid>                                      # soft-delete in the global graph
```

</details>

---

### 4. `mnem retrieve` - Search the graph

Hybrid semantic + keyword + graph retrieval in a single pass. Returns exactly what it found, what it skipped, and how many tokens were used - no silent truncation at the token budget.

```bash
mnem retrieve "what did we decide about the API rate-limit design"
```

> **Example:** Three sprints later, a new engineer asks the agent "why is our retry logic exponential?" The agent retrieves the original decision node with the full rationale - without anyone having to remember to document it separately.

<details>
<summary>More options</summary>

```bash
mnem -R ~/notes retrieve "query"           # target a specific graph explicitly
mnem retrieve "..." --limit 20             # return more results
mnem retrieve "..." --graph-expand 20      # add multi-hop graph traversal
mnem retrieve "..." --graph-expand 20 --community-filter --graph-mode ppr
mnem retrieve "..." --rerank cohere:rerank-english-v3.0
mnem retrieve "..." --vector-cap 512       # widen the dense candidate pool
```

See [GraphRAG](#graphrag) for the full flag reference.

</details>

---

### 5. `mnem global` - Cross-project, cross-session memory

A second graph at `~/.mnemglobal/.mnem/` that follows agents everywhere - across repos, teams, and sessions. Use it for shared conventions, vendor decisions, and entities that appear in every project.

```bash
mnem global retrieve "what payment provider do we use"
mnem global add node --label Convention -s "All REST APIs are versioned under /v1/"
```

> **Example:** Your platform has a dozen microservices, each with its own `.mnem/`. The global graph holds team-wide conventions, shared entity definitions, and cross-service decisions. Any agent on any service can query it without knowing which repo the fact originated from.

<details>
<summary>More options and local vs global guidance</summary>

```bash
mnem global ingest contacts.md
mnem global add node --label Entity:Person \
  --prop name=Alice -s "Alice leads the infra team"
mnem global get <uuid>
mnem global tombstone <uuid>
```

**When to use local vs global:**

| Use local `.mnem/` for | Use `mnem global` for |
|------------------------|----------------------|
| Project-specific facts, decisions, code context | People, preferences, facts that span all projects |
| Per-repo memory that travels with the repo | Knowledge you want every session and every agent to see |
| Anything you'd commit alongside the code | Cross-session continuity |

The `mnem integrate` command sets up the agent to read local first and fall back to global automatically - no manual switching required during normal use.

</details>

---

### 6. `mnem status` / `mnem log` - Inspect history

See the current state of the graph and walk the op-log backwards.

```bash
mnem status    # op-head CID, head commit, all named refs, label counts
mnem log       # walk op-log backwards, last 20 entries
```

<details>
<summary>More options</summary>

```bash
mnem stats              # compact one-liner: CIDs, ref count, label names
mnem log -n 50          # show last 50 entries
mnem log --oneline      # compact one-line-per-op format
mnem log --format json  # machine-readable JSON stream
```

</details>

---

### 7. `mnem diff` / `mnem show` - Compare snapshots and inspect blocks

See exactly what changed between any two op CIDs: ref deltas plus structural node/edge diff. Decode any block by CID for detailed forensics.

```bash
mnem diff HEAD <cid>
```

> **Example:** An agent ran overnight and committed hundreds of new facts. Before merging into `main`, a reviewer diffs `HEAD` against the pre-run snapshot to confirm nothing unexpected was added or removed.

<details>
<summary>More options</summary>

```bash
mnem diff <op-a-cid> <op-b-cid>   # diff any two ops

mnem show               # decode and pretty-print the current op-head block
mnem show <cid>         # decode any block by CID (Node, Edge, Commit, Operation, ...)

mnem cat-file <cid>                # emit raw DAG-CBOR bytes for any block to stdout
mnem cat-file <cid> --json         # decode to DAG-JSON and pretty-print (pipe into jq)
```

</details>

---

### 8. `mnem branch` - Create and manage branches

Branch the knowledge graph the same way you branch code. Each branch is an independent line of commits - experiment freely, merge back when ready.

```bash
mnem branch create agentic-workflow
```

> **Example:** Two agents are testing competing approaches to a summarisation pipeline. Each works on its own branch - `approach-a` and `approach-b` - committing findings as it goes. A reviewer merges the winning branch back into `main`, preserving the full history of both experiments.

<details>
<summary>More options</summary>

```bash
mnem branch list                        # list all branches; * marks current
mnem branch create <name> <start>       # branch from a ref, branch name, or CID
mnem branch create <name> --from HEAD   # explicit --from form; same resolution as above
mnem branch delete <name>               # delete a local branch pointer
```

</details>

---

### 9. `mnem merge` - Merge branches

3-way graph merge - the same model as `git merge`, but for knowledge. Conflicts land in `.mnem/MERGE_CONFLICTS.json` for explicit resolution.

```bash
mnem merge agentic-workflow
```

> **Example:** Agent A spent a week processing customer interviews; Agent B processed support tickets in parallel. Merging combines both knowledge bases cleanly - no fact is silently overwritten, and the full provenance of every node is preserved.

<details>
<summary>More options</summary>

```bash
mnem merge <branch> --strategy=ours     # auto-resolve: keep current side
mnem merge <branch> --strategy=theirs   # auto-resolve: take incoming side
mnem merge <branch> --dry-run           # preview outcome without persisting anything
mnem merge --continue                   # finish after editing MERGE_CONFLICTS.json
mnem merge --abort                      # cancel; restore HEAD from ORIG_HEAD
```

</details>

---

### 10. `mnem push` / `mnem pull` / `mnem clone` - Sync with a remote

Push and pull a knowledge graph the same way you push and pull code. The wire format is standard CAR v1.

```bash
mnem push          # push HEAD to origin/main
mnem pull          # fast-forward origin/main into HEAD
```

> **Example:** An agent running in CI commits new findings after each build and pushes. Agents on developer machines pull at session start - the whole team works from the same knowledge baseline without any manual sync.

<details>
<summary>More options</summary>

```bash
mnem push <remote> <branch>             # push a specific branch
mnem pull <remote> <branch>             # pull from a specific remote/branch

mnem fetch                              # fetch without merging (default remote)
mnem fetch <remote>                     # fetch from a named remote

mnem clone <url> [<dir>]                # clone a CAR archive into <dir>
mnem clone file:///tmp/repo.car ./copy  # clone from a local file path
mnem clone ./repo.car ./copy            # bare path shorthand (must end in .car)

mnem remote add <name> <url>                         # register a remote
mnem remote add <name> <url> \
  --token-env MNEM_REMOTE_ORIGIN_TOKEN               # supply the bearer token via env var
mnem remote list                                     # list all configured remotes
mnem remote show <name>                              # show URL + capabilities
mnem remote remove <name>                            # remove a remote entry
```

</details>

---

### 11. `mnem query` - Structured graph queries

Exact-match property filter with optional edge traversal. No embedding computation needed - fast and deterministic.

```bash
mnem query --where name=Alice
```

> **Example:** An agent builds an org-chart from onboarding documents. Later, another agent runs `mnem query --where kind=Person --with-outgoing reports_to` to reconstruct the full reporting structure without a text search.

<details>
<summary>More options</summary>

```bash
mnem query --where kind=Person -n 25             # increase result limit
mnem query --where kind=Person \
  --with-outgoing knows                          # follow outgoing "knows" edges
mnem query --where status=active \
  --with-outgoing depends_on \
  --with-outgoing depends_on                     # chain multiple hops

mnem blame <node-uuid>                           # list all incoming edges to a node
mnem blame <node-uuid> --etype authored          # filter to one edge type

mnem ref list                         # list all refs (refs/heads/*, refs/remotes/*, ...)
mnem ref set <name> <target-cid>      # point a ref at a specific commit CID
mnem ref delete <name>                # delete a named ref
```

</details>

---

### 12. `mnem reindex` - Manage embeddings

Backfill or update vector embeddings for nodes. Run after adding a new embedding provider or switching models.

```bash
mnem reindex
```

<details>
<summary>More options</summary>

```bash
mnem reindex --label Doc              # restrict to nodes of one label
mnem reindex --since <commit>         # only nodes added/changed after <commit>
mnem reindex --force                  # re-embed already-indexed nodes
mnem reindex --dry-run                # count what would be embedded without calling the provider

mnem embed --force                    # alias for reindex --force
mnem embed --label Person
```

</details>

---

### 13. `mnem export` / `mnem import` - Backup and restore

Export any snapshot as a standard CAR v1 archive. Import it on any machine, any platform.

```bash
mnem export backup.car
```

> **Example:** Before a large batch ingest, export the current snapshot. If the ingest produces unexpected results, import the snapshot to restore the exact previous state.

<details>
<summary>More options</summary>

```bash
mnem export -                              # write CAR to stdout (pipe over SSH)
mnem export --from refs/heads/main out.car # export from a specific ref
mnem export --from <cid> backup.car        # export from a specific commit CID

mnem import <path>                         # import a CAR archive into the current repo
mnem import -                              # read CAR from stdin
```

</details>

---

### 14. `mnem config` - Configure mnem

Set author identity, embedding provider, and API endpoints. API keys live in environment variables - never written to disk.

```bash
mnem config set user.name "ci-agent"
mnem config set embed.provider ollama
```

<details>
<summary>All config keys</summary>

```bash
mnem config set user.email agent@example.com
mnem config set embed.model nomic-embed-text
mnem config set embed.base_url http://localhost:11434
mnem config get embed.provider
mnem config unset embed.provider
mnem config list
```

Known keys: `user.name`, `user.email`, `user.key`, `user.agent_id`, `embed.provider`, `embed.model`, `embed.api_key_env`, `embed.base_url`.

</details>

---

### 15. `mnem mcp` / `mnem http serve` - Serve the graph

Expose mnem as an MCP server (stdio, for agent hosts) or an HTTP JSON API (for services that call it directly).

```bash
mnem mcp                 # start MCP JSON-RPC server over stdio
mnem http serve          # start HTTP JSON API (loopback by default)
```

> **Example:** A backend service spins up `mnem http serve` at startup. Every agent in the cluster calls the same HTTP endpoint - shared knowledge, no per-instance local state required.

<details>
<summary>More options</summary>

```bash
mnem mcp --repo ~/notes            # point the MCP server at a specific graph

mnem repos list                    # list all repos registered with mnem integrate
mnem repos set-default <path>      # mark a repo as the default without -R
mnem repos prune                   # remove entries for paths that no longer exist
```

</details>

---

### 16. `mnem completions` - Shell completions

Generate and install tab completions for your shell.

```bash
# bash
mnem completions bash > ~/.local/share/bash-completion/completions/mnem

# zsh
mnem completions zsh > ~/.zsh/completions/_mnem
```

<details>
<summary>All shells</summary>

```bash
mnem completions bash
mnem completions zsh
mnem completions fish
mnem completions powershell
mnem completions elvish
```

</details>

---

### Global flag: `-R <path>`

Redirect any command to a specific repository directory, bypassing the walk-up search from the current directory.

```bash
mnem -R ~/notes status
mnem -R ~/notes log
mnem -R ~/notes retrieve "query"
```

<hr>

## MCP Tools

When wired via `mnem integrate`, agents receive **22 native MCP tools** prefixed `mnem_` (21 stable + 1 feature-gated). Every response carries `_meta` with `bytes`, `latency_micros`, and `tokens_estimate` so callers can reason about their own cost. Writes propagate `agent_id` and `task_id` into commit metadata so provenance is always queryable.

Start the server: `mnem mcp --repo <path>` (or let `mnem integrate` wire it automatically).

Full reference: [`docs/src/mcp.md`](docs/src/mcp.md).

### Introspection

| Tool | Description |
|------|-------------|
| `mnem_stats` | Repository overview: op-head, head commit, ref summary, known labels. Cheap; call this first to orient an agent to a new graph. |
| `mnem_schema` | Inspect node labels and edge predicates in the current commit. Use before writing queries or traversals to discover what's in the graph. |
| `mnem_list_nodes` | Enumerate nodes at the current head, optionally filtered by label. Returns UUID + label + summary per node. |
| `mnem_list_tags` | List all named tags (`refs/tags/*`) in the repository. |
| `mnem_recent` | Walk the op-log from HEAD backwards. Returns the last N operations with time, author, `agent_id`, `task_id`, and message. |

### Retrieval

| Tool | Description |
|------|-------------|
| `mnem_retrieve` | **Primary retrieval tool.** Hybrid semantic + sparse + graph search, fused via min-max convex combination or RRF. Returns nodes pre-rendered to text plus `tokens_used` / `dropped` / `candidates_seen` metadata. Supports graph-expand, community filter, PPR, and cross-encoder rerank. |
| `mnem_global_retrieve` | Same as `mnem_retrieve` but always targets the global graph (`~/.mnemglobal/.mnem/`). Use for cross-project, cross-session memory. |
| `mnem_search` | Exact-property match with optional edge traversal. Fast and deterministic - no embedding required. |
| `mnem_vector_search` | Raw cosine-similarity nearest-neighbour search over stored node embeddings. Pass a model name and query vector; receive top-k matches. |
| `mnem_get_node` | Fetch a single node by UUID. Returns full props, content size, and outgoing edge count. |
| `mnem_traverse` | From a start node, list outgoing neighbours reachable via specified edge labels. |
| `mnem_incoming_edges` | List all edges pointing to a node (reverse lookup). Equivalent to `mnem blame` in the CLI. |

### Writes

| Tool | Description |
|------|-------------|
| `mnem_commit` | Add nodes and/or edges as a single commit. Returns the new op-id, commit CID, and created node UUIDs. |
| `mnem_commit_relation` | Compound write: resolve-or-create a subject node, resolve-or-create an object node, and connect them with a typed edge - all in one call. Prevents the duplicate-entity problem (see example below). |
| `mnem_resolve_or_create` | Find-or-create a node by a primary-key property. If a matching `(label, anchor-property) == value` exists, its UUID is returned; otherwise a new node is committed. |
| `mnem_ingest` | Ingest a file path or inline text as a `Doc + Chunk + Entity` subgraph. Accepts `{path: "notes.md"}` or `{text: "...", source: "label"}`. Chunker options: `auto`, `paragraph`, `recursive`, `sentence_recursive`, `session`, `structural`. |
| `mnem_global_ingest` | Same as `mnem_ingest` but writes to the global graph. Use for documents that should be queryable across all sessions and projects. |
| `mnem_global_add` | Write nodes and/or edges directly to the global graph. Use for shared entities (people, orgs, conventions) that appear across multiple projects. |

`mnem_commit_relation` example - link two entities in one call:

```json
{
  "subject": "Alice",
  "subject_kind": "Entity:Person",
  "predicate": "works_at",
  "object": "Globex",
  "object_kind": "Entity:Organization",
  "agent_id": "onboarding-agent"
}
```

### Deletes

| Tool | Description |
|------|-------------|
| `mnem_tombstone_node` | Soft-delete: marks a node as forgotten. Hidden from retrieval by default, but the node CID and all prior commits remain intact for auditing. Use when a user says "forget X" or revokes consent. |
| `mnem_global_tombstone_node` | Same as `mnem_tombstone_node` but operates on the global graph. |
| `mnem_delete_node` | Hard-delete: removes the node from the current head commit. Prior commits that referenced it remain addressable. Use only when the goal is to free storage, not memory hygiene. |

### Optional (feature-gated)

| Tool | Description |
|------|-------------|
| `mnem_community_summarize` | Extractive Centroid + MMR summarizer over a caller-supplied set of node UUIDs. No LLM call - picks `k` sentences balancing proximity to the community centroid against diversity. Enabled via the `summarize` cargo feature. |

<hr>

## Python API (mnem-py)

Use `mnem-py` when you want to read and write a mnem graph directly from Python - without the CLI binary. Same retrieval engine, PyO3 bindings.

```bash
pip install mnem-py
pip install sentence-transformers   # brings ~200 MB of deps (torch, transformers)
```

`mnem-py` stores and retrieves by **dense vector**: you compute embeddings in Python and pass them to mnem. `SentenceTransformer("all-MiniLM-L6-v2")` downloads a ~23 MB model from HuggingFace Hub on first use and caches it in `~/.cache/huggingface/` - all subsequent calls are fully local with no network required.

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

<hr>

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

<hr>

## Compared to others

|  | **mnem** | **mem0** | **Graphiti** | **Letta** | **Supermemory** | **MemPalace** |
|--|:--------:|:--------:|:------------:|:---------:|:---------------:|:-------------:|
| Local-first / offline | ✅ | ✅ | ✅ | ✅ | ✗ | ✅ |
| Versioned history | ✅ | ✗ | ✗ | ✗ | ✗ | ✗ |
| Branch & merge | ✅ | ✗ | ✗ | ✗ | ✗ | ✗ |
| Content-addressed storage | ✅ | ✗ | ✗ | ✗ | ✗ | ✗ |
| WASM / edge deployable | ✅ | ✗ | ✗ | ✗ | ✗ | ✗ |
| API-free ingest | ✅ | ✗ | ✗ | ✗ | ✗ | ✗ |
| Token-budget transparency | ✅ | ✗ | ✗ | ✗ | ✗ | ✗ |
| Single binary, no daemon | ✅ | ✗ | ✗ | ✗ | n/a | ✗ |
| No external DB required | ✅ | ~ | ✗ | ✗ | n/a | ✗ |
| Knowledge graph | ✅ | ~ | ✅ | ✗ | ✗ | ✗ |
| Hybrid retrieval (vector + sparse + graph) | ✅ | ~ | ~ | ✗ | ~ | ~ |
| MCP native | ✅ | ✅ | ~ | ✅ | ✅ | ✗ |
| Open source | Apache-2.0 | Apache-2.0 | Apache-2.0 | Apache-2.0 | ✗ | MIT |

<sup>~ = partial or limited &nbsp;·&nbsp; Graphiti requires Neo4j or SQLite + separate vector store &nbsp;·&nbsp; Letta requires PostgreSQL &nbsp;·&nbsp; MemPalace requires ChromaDB &nbsp;·&nbsp; Supermemory is cloud-only (no self-host)</sup>

Deeper comparisons:

- [mnem vs mem0](docs/src/comparisons/mem0.md) - agent memory layer, OSS leader
- [mnem vs Graphiti](docs/src/comparisons/graphiti.md) - graph-native memory, Zep's OSS stack
- [mnem vs Letta](docs/src/comparisons/letta.md) - agent-memory framework (formerly MemGPT)
- [mnem vs Supermemory](docs/src/comparisons/supermemory.md) - closed-cloud incumbent
- [mnem vs MemPalace](docs/src/comparisons/mempalace.md) - benchmark peer
- [mnem vs Cognee](docs/src/comparisons/cognee.md) - KG-for-agents alternative

Full matrix: [`docs/src/comparisons/README.md`](docs/src/comparisons/README.md).

<hr>

## When NOT to use mnem

- **You need transactional OLTP.** mnem is append-only with versioned history; row-level UPDATE/DELETE semantics aren't the model.
- **You need sub-50 ms cloud-scale retrieval at 10k+ QPS.** mnem is local-first. Multi-region sharded retrieval is on the roadmap, not in v1.

> Looking for hosted memory, multi-region replicas, shared graphs across teams, or a managed remote layer? A sibling project bringing those to mnem is in active development - watch this space.

<hr>

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

<hr>

## Documentation

- [Quickstart](docs/src/quickstart.md) - five-minute walkthrough
- [Install](docs/src/install.md) - per-platform install matrix
- [CLI reference](docs/src/cli.md) - every subcommand and flag
- [MCP server](docs/src/mcp.md) - tools exposed, client wiring
- [Core concepts](docs/src/core-concepts.md) - CIDs, commits, labels
- [Configuration](docs/src/configuration.md) - env vars, config.toml
- [Benchmarks methodology](docs/src/benchmarks/methodology.md)
- [Reproduce benchmarks](docs/src/benchmarks/reproduce.md)
- [Embedding providers](docs/src/guides/embed-providers.md)
- [Migrations](docs/src/migrations/)

<hr>

## Contributing

Issues and PRs welcome. Build and test locally:

```bash
cargo build --features bundled-embedder
cargo test
```

- [`CONTRIBUTING.md`](CONTRIBUTING.md) - branch conventions, review etiquette, how to ship a PR
- [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md) - rules of engagement (Contributor Covenant 2.1)
- [`SECURITY.md`](SECURITY.md) - vulnerability disclosure policy

## License

[Apache-2.0](LICENSE). See [`NOTICE`](NOTICE) for third-party attributions.

<hr>

⭐ **Find mnem useful?** A star is the strongest signal we get from a satisfied builder - it helps the next agent developer find this repo when they're stuck on memory. We read every issue, every PR, every mention. Tell us what you built.
