# mnem vs MemPalace

> MemPalace: "The best-benchmarked open-source AI memory system. And it's free." (repo description, MemPalace/mempalace)
> mnem: a content-addressed, versioned graph substrate that shares MemPalace's no-LLM-on-write philosophy and pushes further on identity and history.

## At a glance

|                                  | mnem                        | MemPalace                                             |
|----------------------------------|-----------------------------|-------------------------------------------------------|
| License                          | Apache-2.0                  | MIT                                                   |
| Stars                            | small / pre-launch          | 49,768 (GitHub API, 2026-04-26)                       |
| Embedded / Server                | embedded                    | embedded (Python + ChromaDB)                          |
| LLM at ingest                    | no                          | no (verbatim store)                                   |
| Content-addressed                | yes                         | no (ChromaDB row IDs)                                 |
| Bitemporal                       | no                          | partial (`valid_from` / `valid_to` on KG entries)     |
| WASM target                      | yes                         | no                                                    |
| MCP server                       | yes (11 tools)              | yes (29 tools)                                        |
| Hybrid retrieval                 | yes (vector + sparse + graph) | yes (semantic + hybrid v4 / v5 with keyword + temporal boost) |
| Token-budget retrieval metadata  | yes                         | no                                                    |
| 3-way merge                      | yes                         | no                                                    |
| Reproducible benchmarks in-repo  | yes                         | yes (per-question JSONL committed)                    |

## Feature comparison

| # | Dimension | mnem | MemPalace | Source |
|---|---|---|---|---|
| 1 | Schema | open (any labels / properties) | fixed: wings, rooms, halls, drawers | MemPalace README "What it is" sha `6890948e092b` |
| 2 | Storage | redb embedded | ChromaDB + SQLite | MemPalace README + `mempalace/backends/base.py` |
| 3 | Default embedder | bundled ONNX MiniLM-L6-v2 | ChromaDB default (MiniLM-L6-v2 implied) | MemPalace `requirements` |
| 4 | LLM at ingest | none | none | MemPalace README |
| 5 | LLM at retrieval | optional rerank | optional hybrid-v4 + LLM rerank tier | MemPalace Benchmarks table |
| 6 | Identity | content CID (BLAKE3 over DAG-CBOR) | ChromaDB row IDs | implementation |
| 7 | History | signed commit DAG | append-only with `valid_from` / `valid_to` | MemPalace KG section |
| 8 | Conflict resolution | 3-way merge | manual `invalidate` tool | MemPalace MCP tool list |
| 9 | Sparse lane | BM25 + SPLADE | hybrid-v4 keyword boost | MemPalace BENCHMARKS.md |
| 10 | Graph lane | first-class (label / prop / adjacency) | KG with timeline + cross-wing tunnels | MemPalace MCP tools |
| 11 | MCP surface | 11 tools | 29 tools | MemPalace README "MCP server" |
| 12 | Plugin scaffolds | mnem-mcp + `mnem integrate` | `.claude-plugin/`, `.codex-plugin/` in repo | MemPalace repo |
| 13 | Bindings | Rust + Python + TS + HTTP + CLI + MCP | Python + MCP | MemPalace README |
| 14 | Hosted product | none | none | n/a |
| 15 | Velocity | maturing 1.0 | 433 commits in first 12 days, 30 contributors (early 2026) | internal notes; verify on repo today |

## Benchmarks (where comparable)

MemPalace publishes retrieval R@5 / R@10 numbers in the same family as
mnem's harness. We pulled their numbers from `benchmarks/BENCHMARKS.md`
and ran ours on the same datasets and embedder weights:

| Benchmark | Split | Metric | MemPalace | mnem 0.1.0 | Delta |
|-----------|-------|--------|-----------|-----------|-------|
| LongMemEval | 500 Q | R@5 session, raw dense | 0.966 | 0.966 | 0 |
| LongMemEval | 500 Q | R@10 session, raw dense | 0.982 | 0.982 | 0 |
| LongMemEval | 500 Q hybrid-v4 | R@5 session | 0.982 | 0.976 | -0.006 |
| LoCoMo | 1986 Q | R@5 session, raw dense | 0.508 | 0.726 | +0.218 |
| LoCoMo | 1986 Q | R@10 session, raw dense | 0.603 | 0.855 | +0.252 |
| ConvoMem | 250 Q | Avg recall | 0.890 | 0.976 | +0.086 |
| MemBench | 100 Q (movie) | R@5 | 0.950 | 1.000 | +0.050 |

Method: identical MiniLM-L6-v2 ONNX weights, no reranker, no LLM, no
lexical lane on the raw-dense rows. The LoCoMo gap comes from mnem's
adapter aggregating user-turn text per session before embedding;
MemPalace's adapter embeds at a finer grain. Mechanism, not magic.

MemPalace's hybrid-v4 numbers tune on dev splits; the held-out 98.4%
they report is the honest figure to compare against.

## Latency (where measured)

| System | Setup | Latency |
|---|---|---|
| mnem | LongMemEval 500 Q, MiniLM ONNX | 711 ms mean retrieve |
| mnem | LoCoMo 1986 Q, MiniLM ONNX | 333 ms mean retrieve |
| MemPalace | LongMemEval, raw dense | not headlined; ChromaDB-default latency |

MemPalace does not publish a single mean-latency number; their
benchmark tables focus on accuracy.

## Architecture differences

MemPalace stores conversation history verbatim in ChromaDB and indexes
people / projects as **wings**, topics as **rooms**, flows as
**halls**, content as **drawers**. The retrieval layer is pluggable
behind `mempalace/backends/base.py`. A SQLite-backed knowledge graph
adds `valid_from` / `valid_to` windows, an `invalidate` verb, and a
`timeline` view. The MCP server exposes 29 tools including agent
diaries and cross-wing tunnels. The product is opinionated: the palace
metaphor *is* the user experience.

mnem ships no metaphor. Nodes and edges are open-schema; you commit
whatever shape your application needs. Identity is a CID over canonical
DAG-CBOR + BLAKE3, so identical content collapses to the same node
across machines. History is a signed commit DAG with diff / log /
branch / 3-way merge. Retrieval is 3-lane RRF (HNSW dense + BM25/SPLADE
sparse + graph traversal) with first-class token-budget telemetry on
every response. `mnem-core` is no-tokio / no-fs / no-net and compiles
to WASM unchanged.

## Where MemPalace clearly wins

- **Verbatim store with measured 96.6% R@5 on LongMemEval, no API
  key.** Same as mnem on raw dense, and reproducible from their
  repo.
- **MCP breadth.** 29 tools to mnem's 11. Agent diaries and cross-wing
  tunnels are original ideas.
- **Plugin scaffolds in-repo.** `.claude-plugin/` and `.codex-plugin/`
  lower install friction for Claude Code / Codex users.
- **Velocity and community.** Hundreds of commits, dozens of
  contributors, rapid issue response.
- **Reproducibility culture.** Per-question JSONL result files
  committed for every benchmark run.
- **Working temporal KG.** `valid_from` / `valid_to` / `invalidate` /
  `timeline` shipped today.

## Where mnem clearly wins

- **Open schema.** No fixed wings/rooms/halls/drawers hierarchy. Use
  any labels and properties for any domain.
- **Content-addressed identity.** Same fact = same CID across machines.
  Stable citations forever.
- **Real commit DAG.** Branch, diff, 3-way merge, signed Ed25519
  history. MemPalace stores facts and a timeline; mnem stores commits
  over a graph.
- **WASM target.** Same retrieval logic in browsers, Workers, Lambda.
  Python + ChromaDB cannot.
- **Retrieval-quality lead on LoCoMo.** +0.218 R@5 raw dense, same
  embedder.
- **Token-budget telemetry.** `tokens_used`, `candidates_seen`,
  `dropped` returned on every retrieve.

## When to pick MemPalace, when to pick mnem

**Pick MemPalace if:** the wings / rooms / halls / drawers metaphor
matches your domain, you want the largest MCP tool surface available,
or you specifically want a Claude-Code-paired personal memory
appliance with reproducible benchmark numbers today.

**Pick mnem if:** you want an open-schema substrate, you need
content-addressing and a real commit DAG, you are shipping to multiple
languages or to the edge / WASM, or you want token-budget telemetry as
a first-class response field.

## Sources

- MemPalace repo, sha `6890948e092b` on `develop`, 2026-04-26: <https://github.com/MemPalace/mempalace>
- MemPalace README (license MIT, "What it is", Benchmarks table): <https://github.com/MemPalace/mempalace/blob/develop/README.md>
- MemPalace `benchmarks/BENCHMARKS.md` for benchmark provenance
- mnem benchmark artefacts: [`/benchmarks/proofs/v0.1.0/`](../../../benchmarks/proofs/v0.1.0/)
- mnem README + benchmark methodology: [`/README.md`](../../../README.md), [`benchmarks/methodology.md`](../benchmarks/methodology.md)
