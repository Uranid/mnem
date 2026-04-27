# Agent memory

How to use mnem as the persistent memory layer for an AI agent or
multi-agent system.

## What this replaces

Most agent-memory setups land somewhere on this list. mnem replaces all
of them:

| Pattern | Why it breaks |
|---------|---------------|
| Stuff everything into the prompt | context-window-bound, no versioning, no recall |
| Append to a `notes.md` / `skills.md` | unversioned, unqueryable, wasteful of tokens |
| Vector store + flat metadata | no relationships, no merge semantics, no audit trail |
| Bespoke memory service | weeks of plumbing, no shared format, no provenance |

## What mnem gives you

- **Cryptographic identity per fact.** Every node has a CID. Identical
  content collapses to one node across machines and runs.
- **Versioned by default.** Every ingest, every edit, every tombstone
  produces a commit. Walk history, diff revisions, blame a fact.
- **Concurrent-write safe.** Two agents writing the same scope produce
  two commits with one parent; `mnem merge` reconciles them
  three-way over the graph and the embeddings.
- **Token-budget retrieve.** Every retrieve emits `tokens_used`,
  `candidates_seen`, `dropped` so the agent loop can pace its context
  without surprise truncation.
- **Per-conversation / per-user scoping** via `label`. Prevents
  cross-tenant leak in multi-user agents.

## Wire-up patterns

### 1. MCP (one command)

```bash
mnem mcp install
```

Adds an MCP server entry to Claude Desktop / Cursor / Zed. The agent
gets `mnem_retrieve`, `mnem_ingest`, `mnem_traverse`, `mnem_stats`, etc.
as native tools.

### 2. HTTP from any language

```bash
mnem-http --bind 127.0.0.1:9876 --repo /path/to/graph
```

Then POST to `/v1/retrieve` with `{"text": "...", "label": "user-42"}`.
Response includes top-K items + token-budget metadata.

### 3. Embed in Rust

```rust
use mnem_core::Repo;
let repo = Repo::open(".")?;
let items = repo.retrieve(query).label("user-42").token_budget(500).execute?;
```

### 4. Embed in Python

```python
from mnem import Repo
repo = Repo.open(".")
items = repo.retrieve(query, label="user-42", token_budget=500)
```

## Scoping by user / conversation

Pass `--label <str>` at ingest:

```bash
mnem ingest user-42-chat.json --label user-42 --json
```

Subsequent retrieves with `--label user-42` see only that scope.
Without a label, queries see the whole repo.

## Auditing what an agent did

```bash
mnem log              # commit history
mnem diff <cid> HEAD  # what changed
mnem cat <cid>        # dump a node by CID
mnem blame <cid>      # which commit added this fact
```

## Reconciling concurrent agents

Two agents operate on the same graph in parallel. Each commits its own
chain. When you want a unified head:

```bash
mnem merge agent-a agent-b
```

mnem merges three-way over the graph (which nodes/edges each side
added or removed) and over the embedding sidecar (recomputed for
merged set).

## When NOT to use mnem for agent memory

- Single-user, in-memory only, throw-away after the session: a python
  dict is simpler.
- Strict OLTP transactional updates: mnem is append-only with
  versioned history; UPDATE/DELETE semantics live in `mnem revert` /
  `mnem tombstone`.
- Sub-50 ms cloud-scale latency at 10k QPS: mnem is local-first; if
  you need a hosted multi-region tier, that doesn't ship in v1.
