# mnem vs mem0

> mem0: "Universal memory layer for AI Agents" (repo description, mem0ai/mem0)
> mnem: a content-addressed, versioned graph substrate underneath the memory layer.

## At a glance

|                                  | mnem                        | mem0                                                  |
|----------------------------------|-----------------------------|-------------------------------------------------------|
| License                          | Apache-2.0                  | Apache-2.0                                            |
| Stars                            | small / pre-launch          | 54,113 (GitHub API, 2026-04-26)                       |
| Embedded / Server                | embedded                    | library + optional managed Platform                   |
| LLM at ingest                    | no                          | yes by default (single-pass ADD-only since v3, Apr 2026); `infer=False` opt-out exists |
| Content-addressed                | yes                         | no (UUIDs over a vector store)                        |
| Bitemporal                       | no                          | no (event log, not bitemporal)                        |
| WASM target                      | yes                         | no (Python + external vector DB)                      |
| MCP server                       | yes                         | yes (mem0 MCP exists)                                 |
| Hybrid retrieval                 | yes (vector + sparse + graph + RRF) | yes (semantic + BM25 + entity matching, fused) since v3 |
| Token-budget retrieval metadata  | yes                         | no                                                    |
| 3-way merge                      | yes                         | no (event log with add/update/delete)                 |
| Reproducible benchmarks in-repo  | yes                         | partial (separate `memory-benchmark` repo)            |

## Feature comparison

| # | Dimension | mnem | mem0 | Source |
|---|---|---|---|---|
| 1 | Data model | open-schema content-addressed nodes + edges | rows in a vector store with `{role, content}` history; `user_id` / `agent_id` / `run_id` scoping | mem0 README "Basic Usage" + docs |
| 2 | Default ingest | parse + chunk + statistical extract | LLM (gpt-5-mini default) extracts atomic facts on every `add` | mem0 README "Basic Usage" sha `bd9d27ff509f` |
| 3 | LLM requirement | optional | required by default; `infer=False` opts out but loses the "magic" | mem0 v3 README "New Memory Algorithm" |
| 4 | Identity | BLAKE3 CID over DAG-CBOR | UUIDs over a vector row | mem0 docs |
| 5 | History | signed commit DAG, diff / log / branch / merge | `history` event log of add/update/delete records | mem0 SDK |
| 6 | Conflict resolution | 3-way merge over graph | "latest LLM extraction wins" before v3; v3 is ADD-only and accumulates | mem0 v3 release notes |
| 7 | Vector backends | redb default, pluggable via `Blockstore` | 20+ (Qdrant, Chroma, PGVector, Pinecone, Weaviate, etc.) | mem0 docs "Supported Vector Stores" |
| 8 | LLM providers | optional, 16 via `mnem-llm-providers` | 16+ (OpenAI, Anthropic, Gemini, Groq, Ollama, ...) | mem0 docs "Supported LLMs" |
| 9 | Embedding model | bundled ONNX MiniLM-L6-v2 in-process | configurable; default OpenAI `text-embedding-3-small` | mem0 README |
| 10 | Retrieval lanes | dense (HNSW) + sparse (BM25/SPLADE) + graph + RRF | semantic + BM25 + entity match (v3) | mem0 v3 README |
| 11 | Token-budget metadata | first-class on every retrieve | not exposed | mnem CLI / HTTP API |
| 12 | Multi-tenancy | repo-per-tenant or scope by node label | hardcoded `user_id` / `agent_id` / `run_id` triple | mem0 SDK |
| 13 | Bindings | Rust + Python + HTTP + MCP + CLI | Python + TypeScript + REST + MCP | mem0 README badges |
| 14 | Cloud | none yet | "mem0 Platform": Hobby free, Starter $19, Pro $249, Enterprise | mem0.ai pricing |
| 15 | Distribution | pre-launch | YC S24, ~2.6M monthly PyPI downloads | mem0 README badge |

## Benchmarks (where comparable)

mem0 v3 (April 2026) reports on LoCoMo and LongMemEval as a full pipeline
(LLM extract + retrieve + answer). mnem reports retrieval-only (R@K)
under an identical embedder, no LLM in the loop.

We have a same-harness, same-embedder reproduction of mem0 with
`infer=False` (LLM extraction off) so the comparison lands on the
retrieval layer:

| Benchmark | Split | Metric | mem0 (`infer=False`, MiniLM) | mnem | Delta |
|-----------|-------|--------|------------------------------|-----------|-------|
| LongMemEval | 500 Q | R@5 session | 0.946 | $\color{green}{\textbf{0.966}}$ | +0.020 |
| LongMemEval | 500 Q | R@10 session | 0.962 | $\color{green}{\textbf{0.982}}$ | +0.020 |
| LoCoMo | 1986 Q | R@5 session | 0.466 | $\color{green}{\textbf{0.726}}$ | +0.260 |
| LoCoMo | 1986 Q | R@10 session | 0.676 | $\color{green}{\textbf{0.855}}$ | +0.179 |

Adapter notes: `infer=False`, persistent `Memory`, per-item `user_id`
scoping. See [`benchmarks/methodology.md`](../benchmarks/methodology.md).

mem0's own v3 numbers (LoCoMo 91.6, LongMemEval 93.4) are full-pipeline
end-to-end accuracy, not retrieval R@5; not directly comparable to the
table above.

## Latency (where measured)

| System | Setup | Latency |
|---|---|---|
| mnem | LongMemEval 500 Q, MiniLM-L6-v2, embedded redb | 711 ms mean retrieve |
| mnem | LoCoMo 1986 Q, same setup | 333 ms mean retrieve |
| mem0 | LongMemEval, v3 single-pass | 1.09 s p50 (mem0 README, Apr 2026) |
| mem0 | LoCoMo, v3 single-pass | 0.88 s p50 (mem0 README) |

mem0 v3 latency includes one LLM retrieval call per query; mnem's
numbers are pure retrieval. Different mechanisms, useful only as an
order-of-magnitude check.

## Architecture differences

mem0 is a Python (and TS) memory layer designed to drop into LLM apps.
The default flow is: `mem.add(messages, user_id=...)` runs an LLM to
extract atomic facts, embeds them into a configured vector store, and
returns a UUID per memory. Retrieval (`mem.search(...)`) does semantic
+ keyword + entity matching, optionally with a reranker. Multi-tenancy
is hardcoded as `user_id` / `agent_id` / `run_id`. mem0 Platform layers
a managed cloud, dashboards, and SOC 2 / GDPR on top.

mnem is one layer below: a content-addressed, versioned graph
substrate. There is no fixed conversation schema; you commit nodes and
edges with whatever labels and properties you need. Identity is a CID
over canonical DAG-CBOR + BLAKE3, so the same fact on two machines
collapses to the same node. History is a signed commit DAG, not an
event log, so old facts remain addressable after newer ones supersede
them. The write path runs no LLM by default; ingest is statistical
parse + chunk + key extract. Retrieval is 3-lane RRF (HNSW dense +
sparse + graph) with token-budget telemetry on every response.

## Where mem0 clearly wins

- **Distribution.** ~2.6M monthly PyPI downloads, default memory in
  LangChain / LlamaIndex / CrewAI / Vercel AI SDK / LiveKit / Pipecat /
  AWS Bedrock. mem0 is the path of least resistance.
- **Backend breadth.** 20+ vector stores, 16 LLMs, 10 embedders work
  out of the box.
- **Managed product.** Hobby tier is free; Pro is $249/mo with
  dashboards, SOC 2, on-prem.
- **LLM-assisted ingest.** `mem.add("I met Alice in Berlin")`
  auto-extracts `{entity: Alice, city: Berlin}` with no upstream
  modelling effort.
- **YC + commercial momentum.** YC S24, $24M raised, weekly release
  cadence on v3.

## Where mnem clearly wins

- **No LLM in the write path.** Regulated, offline, or cost-sensitive
  workloads ingest deterministically. mem0 v3 reduced the LLM cost to
  one call per add but did not eliminate it.
- **Content-addressed CIDs.** Globally stable identity; CID-citations
  stay reproducible. mem0's UUIDs are per-instance random.
- **Versioned history with 3-way merge.** Diff / log / branch / merge
  / signed commits. mem0 ships an event log, not a commit graph.
- **Embedded + single binary.** ~40 MB Docker image, no external vector
  DB. Runs offline.
- **WASM target.** mnem-core compiles to `wasm32`; mem0 cannot.
- **Retrieval-quality lead under identical-embedder conditions.** +0.20
  R@5 on LongMemEval, +0.260 R@5 on LoCoMo (same MiniLM weights, dense
  lane only).
- **Token-budget telemetry.** `tokens_used` / `dropped` per retrieve.

## When to pick mem0, when to pick mnem

**Pick mem0 if:** you want drop-in agent memory with the broadest
LangChain / LlamaIndex / CrewAI footprint, you are happy paying an LLM
call per add for "magic" extraction, or you want a managed cloud and
dashboards today.

**Pick mnem if:** you want an embedded substrate with no LLM at ingest,
you need content-addressing and a real commit graph, you care about
reproducibility and audit, or you are shipping to the edge / WASM /
offline.

## Sources

- mem0 repo, sha `bd9d27ff509f` on `main`, 2026-04-26: <https://github.com/mem0ai/mem0>
- mem0 README ("New Memory Algorithm (April 2026)", "Basic Usage", "CLI"): <https://github.com/mem0ai/mem0/blob/main/README.md>
- mem0 docs: <https://docs.mem0.ai>
- mem0 evaluation framework: <https://github.com/mem0ai/memory-benchmark>
- mnem benchmarks: [`/benchmarks/results/v0.1.0/`](../../../benchmarks/results/v0.1.0/)
- mnem README: [`/README.md`](../../../README.md)
