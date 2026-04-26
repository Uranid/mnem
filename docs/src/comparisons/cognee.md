# mnem vs Cognee

> Cognee: "Knowledge Engine for AI Agent Memory in 6 lines of code" (repo description, topoteretes/cognee)
> mnem: a content-addressed, versioned graph substrate that ingests without an LLM.

## At a glance

|                                  | mnem                        | Cognee                                                |
|----------------------------------|-----------------------------|-------------------------------------------------------|
| License                          | Apache-2.0                  | Apache-2.0                                            |
| Stars                            | small / pre-launch          | 16,807 (GitHub API, 2026-04-26)                       |
| Embedded / Server                | embedded                    | library + Cognee Cloud                                |
| LLM at ingest                    | no                          | yes (`remember` calls `add` + `cognify` + `improve`)  |
| Content-addressed                | yes                         | no (extracted graph node IDs)                         |
| Bitemporal                       | no                          | no                                                    |
| WASM target                      | yes                         | no                                                    |
| MCP server                       | yes                         | yes (Cognee MCP exists; integrates with Claude Code, Hermes) |
| Hybrid retrieval                 | yes (vector + sparse + graph + RRF) | yes (auto-routing across graph + vector)              |
| Token-budget retrieval metadata  | yes                         | no                                                    |
| 3-way merge                      | yes                         | no                                                    |
| Reproducible benchmarks in-repo  | yes                         | partial (research / paper claims; no in-repo harness) |

## Feature comparison

| # | Dimension | mnem | Cognee | Source |
|---|---|---|---|---|
| 1 | Storage | redb embedded | Kuzu default + vector DB; pluggable | Cognee README "Deploy Cognee" sha `f4964c31db04` |
| 2 | Default flow | `commit` -> CID; no LLM | `remember` runs `add` + `cognify` + `improve` (LLM extraction) | Cognee README Quickstart Step 3 |
| 3 | LLM requirement | optional | required to configure before Quickstart Step 2 | Cognee README "Step 2: Configure the LLM" |
| 4 | Identity | BLAKE3 CID over DAG-CBOR | extracted graph-node IDs (LLM-derived) | Cognee internals |
| 5 | History | signed commit DAG | none; standard graph state | Cognee docs |
| 6 | Conflict resolution | 3-way merge | re-run `cognify` to refresh graph | Cognee Quickstart |
| 7 | Vector lane | HNSW via `mnem-ann` | configurable vector store | Cognee docs |
| 8 | Sparse lane | BM25 + SPLADE | not headlined | Cognee README |
| 9 | Graph lane | first-class | first-class | Cognee README "About Cognee" |
| 10 | LLM providers | optional | OpenAI, Anthropic, Gemini, Ollama, others | Cognee docs |
| 11 | Session memory | open (model your own) | `remember(..., session_id="...")` first-class | Cognee Quickstart |
| 12 | Auto-routing retrieval | manual lane configuration | "picks best search strategy automatically" | Cognee README Step 3 |
| 13 | Bindings | Rust + Python + TS + HTTP + CLI + MCP | Python + CLI + MCP | Cognee README |
| 14 | Cloud | none yet | Cognee Cloud (managed) | Cognee README "Connect to Cognee Cloud" |
| 15 | Determinism | byte-identical CIDs same input | LLM extraction is non-deterministic | Cognee README Step 3 |

## Benchmarks (where comparable)

Not directly comparable. Cognee publishes research and use-case
narratives rather than retrieval R@K artefacts in the repo. Their
strength is the ECL pipeline as a finished product (drop a PDF, get a
typed knowledge graph), not retrieval-quality benchmarks at the
substrate layer.

mnem's measured retrieval numbers under ONNX MiniLM-L6-v2:

| Benchmark | Split | Metric | mnem 0.1.0 |
|-----------|-------|--------|-----------|
| LongMemEval | 500 Q | R@5 session | 0.966 |
| LoCoMo | 1986 Q | R@5 session | 0.726 |
| ConvoMem | 250 Q | Avg recall | 0.976 |
| MemBench | 100 Q (movie) | R@5 | 1.000 |

If you want to compare like-for-like, run Cognee against the same
LongMemEval / LoCoMo dumps with `infer=False`-equivalent (skip
`cognify`'s LLM extraction). We have not published a Cognee adapter;
contributions welcome.

## Latency (where measured)

| System | Setup | Latency |
|---|---|---|
| mnem | LongMemEval 500 Q, MiniLM ONNX | 711 ms mean retrieve |
| mnem | LoCoMo 1986 Q, MiniLM ONNX | 333 ms mean retrieve |
| Cognee | not headlined; depends on LLM provider + vector store | n/a |

Cognee's retrieval latency depends heavily on whether the auto-router
calls a vector store, the graph DB, or LLM rerank. They do not publish
a single number.

## Architecture differences

Cognee is a Python knowledge-engine library and a managed cloud. The
write path is the **ECL pipeline**: Extract -> Cognify -> Load. You
hand it documents, Cognee runs an LLM to extract entities and
relationships, embeds them, and writes them into a graph DB (Kuzu by
default) plus a vector store. Retrieval auto-routes across vector and
graph based on the query. The strength is "drop a PDF and get a typed
knowledge graph" with minimal modelling effort.

mnem is the substrate beneath that pattern. There is no required LLM
at ingest: parse + chunk + statistical extract -> CID -> commit. The
graph shape is whatever your application commits, not whatever the
LLM happened to extract that hour. Identity is content-addressed, so
the same document on two machines collapses to the same nodes.
History is a signed commit DAG with diff / 3-way merge. Retrieval is
explicit 3-lane RRF with token-budget telemetry, not auto-routed.

## Where Cognee clearly wins

- **Drop-in ingest.** Hand it a PDF, conversation, or URL; get a typed
  knowledge graph. mnem expects you to model what you want stored.
- **Auto-routing retrieval.** The router picks vector vs graph vs
  hybrid for you. mnem makes you choose.
- **Multi-LLM-provider support.** OpenAI, Anthropic, Gemini, Ollama,
  others work with minimal config.
- **Cloud + self-host parity.** Cognee Cloud is managed; the OSS
  library works standalone.
- **Rich ontology derivation.** LLM derives the ontology from the
  corpus rather than forcing one upfront.
- **Session memory primitive.** `session_id` is first-class with a
  background sync to the long-term graph.

## Where mnem clearly wins

- **No LLM in the write path.** Deterministic, replayable, fuzz-tested.
  Cognee's `cognify` step is non-deterministic by design.
- **Content-addressed identity.** Same input -> same CIDs across
  machines. Cognee's extracted node IDs are extraction-run-dependent.
- **Real commit DAG.** Branch, diff, 3-way merge, signed Ed25519
  history. Cognee has graph state, not a commit history.
- **Embedded, single binary.** ~40 MB Docker image. No external graph
  DB or vector store to operate.
- **WASM target.** `mnem-core` ships to `wasm32` unchanged.
- **Token-budget retrieval metadata.** First-class on every response.

## When to pick Cognee, when to pick mnem

**Pick Cognee if:** you want PDF / URL / document -> typed knowledge
graph in 6 lines, you are happy with an LLM at ingest, you want auto-
routed retrieval, or you want a managed cloud option.

**Pick mnem if:** you need deterministic ingest, content-addressed
identity, a real commit DAG, embedded / single-binary deployment, or
WASM / edge targets.

## Sources

- Cognee repo, sha `f4964c31db04` on `main`, 2026-04-26: <https://github.com/topoteretes/cognee>
- Cognee README ("About Cognee", Quickstart Steps 1-3, "Use with AI Agents", "Deploy Cognee"): <https://github.com/topoteretes/cognee/blob/main/README.md>
- Cognee docs: <https://docs.cognee.ai>
- Cognee Cloud: <https://www.cognee.ai>
- mnem benchmarks: [`/benchmarks/proofs/v0.1.0/`](../../../benchmarks/proofs/v0.1.0/)
- mnem README + architecture: [`/README.md`](../../../README.md), [`architecture/retrieval.md`](../architecture/retrieval.md)
