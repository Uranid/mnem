# Comparisons

How mnem stacks up against other agent-memory and knowledge-graph
systems. Each comparison is honest: where they win, where mnem wins,
when to pick which.

mnem is open source (Apache-2.0). Numbers come from public artefacts;
where a competitor's claim is closed-source we say so. Where a
benchmark is not directly comparable, we say so rather than fabricate
a single-number league table.

| Competitor                  | License    | Server / Embedded         | LLM at ingest | Bitemporal | Stars    | Compare                      |
|-----------------------------|------------|---------------------------|---------------|------------|----------|------------------------------|
| mem0 (`mem0ai/mem0`)                | Apache-2.0 | library + cloud           | default-on (opt-out) | no    | 56,135   | [mem0.md](mem0.md)           |
| MemPalace (`MemPalace/mempalace`)   | MIT        | embedded (Python + pluggable; ChromaDB default) | no            | partial    | 52,471   | [mempalace.md](mempalace.md) |
| Hermes Agent (`NousResearch/hermes-agent`) | MIT  | embedded agent runtime (Python; SQLite/FTS5 + flat files) | default (agent curates `MEMORY.md`/`USER.md`) | no    | ~157,000 | [hermes.md](hermes.md)       |
| Supermemory (`supermemoryai/supermemory`) | MIT (repo) / closed (cloud) | hosted cloud      | yes           | no         | 22,621   | [supermemory.md](supermemory.md) |
| Graphiti (`getzep/graphiti`)        | Apache-2.0 | server (Neo4j / Kuzu / FalkorDB / Neptune) | mandatory     | yes        | 26,234   | [graphiti.md](graphiti.md)   |
| Letta (`letta-ai/letta`)            | Apache-2.0 | server + CLI              | yes (agent is the writer) | partial | 22,808 | [letta.md](letta.md)         |
| Cognee (`topoteretes/cognee`)       | Apache-2.0 | library + cloud           | yes (`cognify`) | no       | 17,334   | [cognee.md](cognee.md)       |
| graphify (`safishamsi/graphify`)    | MIT        | one-shot CLI              | yes (Claude subagents) | no  | 49,360   | [graphify.md](graphify.md)   |
| mnem                                | Apache-2.0 | embedded + four surfaces  | no            | no         | small / pre-launch | (this repo)        |

Star counts pulled from the GitHub API on 2026-05-19. License columns
reflect the repository SPDX identifier; commercial / hosted layers
above some of these projects ship under different terms. Hermes Agent
is included as the agent runtime mnem plugs into via the
`MemoryProvider` plugin interface; most rows in the main README table
read "n/a" or `✗` for Hermes because they are not in its product
scope, not because it lacks them.

## mnem positioning

mnem is the substrate underneath the products in the table: a content-
addressed, versioned, hybrid-retrieval graph that runs in-process,
ingests without an LLM, and exposes token-budget telemetry on every
retrieve. We are not building a memory product; we are building the
thing the next memory product is built on.

## Reading order

If you have read about agent memory before, the most useful first
read is one of:

- **[mnem vs mem0](mem0.md)** if you have been using the LangChain /
  LlamaIndex / CrewAI defaults.
- **[mnem vs MemPalace](mempalace.md)** if you care about no-LLM-on-
  write retrieval and reproducible benchmarks.
- **[mnem vs Hermes](hermes.md)** if you are running NousResearch's
  Hermes Agent and want a durable memory layer underneath.
- **[mnem vs Supermemory](supermemory.md)** if you have been weighing
  the closed cloud vs self-host trade-off.
- **[mnem vs Graphiti](graphiti.md)** if you have been thinking about
  bitemporal knowledge graphs.
- **[mnem vs Letta](letta.md)** if you have been looking at the
  MemGPT lineage of agent platforms.
- **[mnem vs Cognee](cognee.md)** if you have been looking at ECL-
  pipeline-shaped knowledge engines.
- **[mnem vs graphify](graphify.md)** if you have been using
  one-shot folder-to-graph extractors.
