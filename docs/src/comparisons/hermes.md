# mnem vs Hermes Agent

> Hermes Agent: "An open-source agent runtime for tool-using LLMs." (NousResearch/hermes-agent, v0.14.0 "Foundation Release", 2026-05-16)
> mnem: a content-addressed, versioned knowledge-graph substrate that attaches to Hermes as a `MemoryProvider` plugin.

## Category note (read this first)

Hermes Agent is an **agent runtime / framework**. It is the host
that owns the loop: prompt assembly, tool dispatch, session lifecycle,
compression. mnem is a **memory substrate**. It owns durable facts,
identity, history, and retrieval.

The two are not competitors. The honest framing is "Hermes as the
host that uses memory, mnem as the memory layer underneath." Rows
below where Hermes has no entry (knowledge graph, versioning, WASM,
hybrid retrieval, content-addressing) reflect a category difference,
not a deficiency of Hermes - those features are simply not in the
runtime's product scope.

The integration path is real: mnem ships as a Hermes `MemoryProvider`
plugin, wired by `mnem integrate hermes`.

## At a glance

|                                  | mnem                          | Hermes Agent                                    |
|----------------------------------|-------------------------------|-------------------------------------------------|
| License                          | Apache-2.0                    | MIT                                             |
| Stars                            | small / pre-launch            | ~157,000 (GitHub API, 2026-05-19)               |
| Latest release                   | v0.1.0                        | v0.14.0 "Foundation Release" (2026-05-16)       |
| Category                         | memory substrate              | agent runtime / framework                       |
| Language                         | Rust (+ Py / TS / WASM)       | Python                                          |
| Built-in memory                  | n/a (mnem *is* the memory)    | bounded markdown (`MEMORY.md`, `USER.md`) + FTS5 session search |
| External memory                  | n/a                           | `MemoryProvider` plugin protocol (one active)   |
| Knowledge graph                  | yes (labels, props, edges)    | n/a (agent runtime)                             |
| Hybrid retrieval                 | yes (vector + sparse + graph) | n/a (agent runtime)                             |
| Content-addressed                | yes (BLAKE3 over DAG-CBOR)    | n/a (agent runtime)                             |
| Versioning / commit DAG          | yes (branch, diff, 3-way merge) | n/a (agent runtime)                           |
| WASM target                      | yes                           | n/a (Python runtime)                            |
| MCP server                       | yes (18 tools)                | n/a - uses Python `MemoryProvider` protocol, not MCP, as the memory transport |

## Feature comparison

| # | Dimension | mnem | Hermes Agent | Source |
|---|---|---|---|---|
| 1 | Primary role | persistent memory store | agent loop + tool dispatch + session lifecycle | Hermes README |
| 2 | Default memory | n/a (mnem is the store) | markdown files capped at 2200 chars (`MEMORY.md`) and 1375 chars (`USER.md`), injected each session | hermes-agent.nousresearch.com docs |
| 3 | Session search | n/a | SQLite FTS5 over historical sessions | Hermes docs |
| 4 | External memory API | n/a (provider) | Python `MemoryProvider` with lifecycle hooks | Hermes `memory_provider.py` |
| 5 | Lifecycle hooks (external memory) | n/a | `prefetch`, `queue_prefetch`, `sync_turn`, `on_session_end`, `on_pre_compress`, `on_memory_write`, `system_prompt_block`, `shutdown` | Hermes plugin docs |
| 6 | Concurrent external providers | n/a | one at a time | Hermes plugin docs |
| 7 | Schema | open (any labels / properties) | n/a (agent runtime) | mnem README |
| 8 | Identity | content CID (BLAKE3 over DAG-CBOR) | n/a (agent runtime) | implementation |
| 9 | History | signed commit DAG | session transcripts in SQLite | Hermes docs + mnem README |
| 10 | Conflict resolution | 3-way merge | n/a (agent runtime) | mnem README |
| 11 | Retrieval | 3-lane RRF (HNSW + BM25/SPLADE + graph) | n/a (agent runtime); delegated to active `MemoryProvider` | Hermes plugin docs |
| 12 | Bindings | Rust + Python + TS + HTTP + CLI + MCP | Python | Hermes README |
| 13 | Hosted product | none | none | n/a |

## Architecture differences

Hermes Agent owns the loop. It assembles the system prompt, calls the
LLM, dispatches tool calls, manages session state, compresses on
overflow, and writes transcripts to a local SQLite database. Memory
for Hermes is, by default, two bounded markdown files injected into
the system prompt every session: `~/.hermes/memories/MEMORY.md` (2200
character cap) for shared facts and `USER.md` (1375 character cap) for
user-specific facts. The runtime also exposes FTS5 search across
historical sessions so the model can recall past conversations on
demand.

When you need more than that, Hermes accepts an external
`MemoryProvider` plugin. A `MemoryProvider` is a Python class that
implements lifecycle hooks: `prefetch` (before the LLM call),
`queue_prefetch` and `sync_turn` (during the turn),
`on_session_end` (cleanup), `on_pre_compress` (before context
compression), `on_memory_write` (when the runtime decides to persist),
`system_prompt_block` (return a block to inject), and `shutdown`. Only
one external provider is active at a time. The provider decides where
facts live, how they're indexed, and how they're retrieved.

mnem ships no agent loop. It is a substrate: open-schema nodes and
edges, content-addressed by CID over canonical DAG-CBOR + BLAKE3, with
a signed commit DAG that supports branch / diff / log / 3-way merge.
Retrieval is 3-lane RRF (HNSW dense + BM25/SPLADE sparse + graph
traversal) with token-budget telemetry on every response.
`mnem-core` is no-tokio / no-fs / no-net and compiles to WASM
unchanged.

The integration point is the `MemoryProvider` interface. mnem's
adapter implements those hooks and routes them to `mnem retrieve` /
`mnem commit` against the local graph (or `mnem global retrieve` /
`mnem global add` when the user opts into the global store).

## Integration path

```
mnem integrate hermes
```

writes a Hermes plugin entry that registers mnem as the active
`MemoryProvider`. The command inspects the installed Hermes and
writes the matching shape (Python `MemoryProvider` for current
versions, shell hooks on older builds).

What the plugin does in each direction:

- **read (before the LLM call)**: `prefetch` queries mnem with the
  user's incoming message, formats the top-K results as a context
  block, and returns it via `system_prompt_block` for that turn.
- **write (after the turn)**: `sync_turn` and `on_memory_write`
  inspect the assistant's output for facts the user stated or
  confirmed, and commit them via `mnem_commit` /
  `mnem_resolve_or_create` / `mnem_commit_relation`. Drafts, model
  reasoning, and tool traces are not committed.
- **session end**: `on_session_end` flushes any queued commits and
  closes the local graph handle.

## When to pick which

These aren't alternatives - they compose.

**Use Hermes Agent if:** you want an open-source agent runtime with
a real session lifecycle, FTS5 history search, and a plugin protocol
for memory. Hermes' built-in markdown memory is sufficient for many
users and stays out of the way.

**Use Hermes + mnem if:** you've outgrown the 2200/1375 character
caps, you need durable facts across sessions and machines, you want
content-addressed identity so the same fact has the same CID
everywhere, you want versioning (branch / diff / merge) on your
memory, or you want hybrid retrieval (vector + sparse + graph) on a
graph schema you define.

**Use mnem standalone if:** your host isn't Hermes - mnem also runs
under Claude Code, Claude Desktop, Cursor, Continue, Zed, Gemini CLI,
and any MCP-capable host via the `mnem mcp` server. The same graph
is reachable from all of them.

## Repo references and sources

- Hermes Agent repo: <https://github.com/NousResearch/hermes-agent>
- Hermes v0.14.0 "Foundation Release" notes (2026-05-16):
  <https://github.com/NousResearch/hermes-agent/releases/tag/v0.14.0>
- Hermes plugin / `MemoryProvider` docs:
  <https://hermes-agent.nousresearch.com>
- mnem README + MCP tools: [`/README.md`](../../../README.md)
