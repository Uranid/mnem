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
| Graphiti (`getzep/graphiti`)        | Apache-2.0 | server (Neo4j / Kuzu / FalkorDB / Neptune) | mandatory     | yes        | 25,409   | [graphiti.md](graphiti.md)   |
| mem0 (`mem0ai/mem0`)                | Apache-2.0 | library + cloud           | default-on (opt-out) | no    | 54,113   | [mem0.md](mem0.md)           |
| MemPalace (`MemPalace/mempalace`)   | MIT        | embedded (Python + ChromaDB) | no            | partial    | 49,768   | [mempalace.md](mempalace.md) |
| Supermemory (`supermemoryai/supermemory`) | MIT (repo) / closed (cloud) | hosted cloud      | yes           | no         | 22,218   | [supermemory.md](supermemory.md) |
| Cognee (`topoteretes/cognee`)       | Apache-2.0 | library + cloud           | yes (`cognify`) | no       | 16,807   | [cognee.md](cognee.md)       |
| Letta (`letta-ai/letta`)            | Apache-2.0 | server + CLI              | yes (agent is the writer) | partial | 22,305 | [letta.md](letta.md)         |
| graphify (`safishamsi/graphify`)    | MIT        | one-shot CLI              | yes (Claude subagents) | no  | 35,262   | [graphify.md](graphify.md)   |
| mnem                                | Apache-2.0 | embedded + four surfaces  | no            | no         | small / pre-launch | (this repo)        |

Star counts pulled from the GitHub API on 2026-04-26. License columns
reflect the repository SPDX identifier; commercial / hosted layers
above some of these projects ship under different terms.

## mnem positioning

mnem is the substrate underneath the products in the table: a content-
addressed, versioned, hybrid-retrieval graph that runs in-process,
ingests without an LLM, and exposes token-budget telemetry on every
retrieve. We are not building a memory product; we are building the
thing the next memory product is built on.

## Reading order

If you have read about agent memory before, the most useful first
read is one of:

- **[mnem vs Graphiti](graphiti.md)** if you have been thinking about
  bitemporal knowledge graphs.
- **[mnem vs mem0](mem0.md)** if you have been using the LangChain /
  LlamaIndex / CrewAI defaults.
- **[mnem vs MemPalace](mempalace.md)** if you care about no-LLM-on-
  write retrieval and reproducible benchmarks.
- **[mnem vs Supermemory](supermemory.md)** if you have been weighing
  the closed cloud vs self-host trade-off.
- **[mnem vs Cognee](cognee.md)** if you have been looking at ECL-
  pipeline-shaped knowledge engines.
- **[mnem vs Letta](letta.md)** if you have been looking at the
  MemGPT lineage of agent platforms.
- **[mnem vs graphify](graphify.md)** if you have been using
  one-shot folder-to-graph extractors.
