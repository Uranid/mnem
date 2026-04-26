# mnem vs Supermemory

> Supermemory: "Memory engine and app that is extremely fast, scalable. The Memory API for the AI era." (repo description, supermemoryai/supermemory)
> mnem: an open-source, embedded, content-addressed knowledge-graph substrate. Self-host or nothing.

## At a glance

|                                  | mnem                        | Supermemory                                           |
|----------------------------------|-----------------------------|-------------------------------------------------------|
| License                          | Apache-2.0                  | MIT (repo); cloud product is closed-core              |
| Stars                            | small / pre-launch          | 22,218 (GitHub API, 2026-04-26)                       |
| Embedded / Server                | embedded                    | hosted cloud; self-host issue (#707) closed without resolution |
| LLM at ingest                    | no                          | yes (Extractors layer; entity / fact extraction)      |
| Content-addressed                | yes                         | no (custom vector graph engine, internals undisclosed) |
| Bitemporal                       | no                          | no                                                    |
| WASM target                      | yes                         | n/a (cloud only)                                      |
| MCP server                       | yes                         | yes (`https://mcp.supermemory.ai/mcp`, OAuth + bearer) |
| Hybrid retrieval                 | yes                         | yes (multi-mode search across graph)                  |
| Token-budget retrieval metadata  | yes                         | not exposed                                           |
| 3-way merge                      | yes                         | no                                                    |
| Reproducible benchmarks in-repo  | yes                         | self-reported; `memorybench` skill is a benchmarking harness vs supermemory |

## Feature comparison

| # | Dimension | mnem | Supermemory | Source |
|---|---|---|---|---|
| 1 | Deployment | embedded; single binary | cloud only; self-host requested + closed (issue #707)  | internal research |
| 2 | Storage | redb embedded | Postgres via Cloudflare Hyperdrive + Cloudflare AI vector embeddings + R2 + KV  | internal research |
| 3 | Vector engine | HNSW via `mnem-ann` | undisclosed; "custom vector graph engine with ontology-aware edges"  | internal research |
| 4 | Embedding model | bundled ONNX MiniLM-L6-v2; pluggable | undisclosed (Cloudflare AI)  | internal research |
| 5 | Identity | content CID | undisclosed | n/a |
| 6 | Multi-tenancy | by repo or graph scope | `containerTag` and project scoping  | internal research |
| 7 | Ingest pipeline | parse + chunk + statistical extract | five stacked layers: User Profiles, Memory Graph, Retrieval, Extractors, Connectors  | internal research |
| 8 | LLM use | optional, opt-in | yes, in Extractors layer  | internal research |
| 9 | Connectors | none yet | webhook-driven connectors live (Notion, GDrive, etc.) | supermemory.ai docs |
| 10 | Plugin / IDE ecosystem | MCP + `mnem integrate` | 12+ integration plugins, dedicated repos  | internal research |
| 11 | API | local Rust / Python / HTTP / MCP / CLI | REST `api.supermemory.ai/v3` + `/v4`, TS / Python SDKs | supermemory README |
| 12 | Pricing | self-host, free | tiered cloud (free / pro / team / enterprise) | supermemory.ai/pricing |
| 13 | Funding / brand | self-funded indie | $3M seed, ~$40M valuation, named angels (Jeff Dean, Dane Knecht, Logan Kilpatrick, ...)  | internal research |
| 14 | Founder reach | small | Dhravya Shah, ~51.5k X followers  | internal research |
| 15 | Self-reported benchmarks | reproducible artefacts in-repo | "#1 on LongMemEval, LoCoMo, ConvoMem"; sub-300 ms recall at 85.4% accuracy  | internal research |

## Benchmarks (where comparable)

Not directly comparable in any apples-to-apples sense. Supermemory's
benchmark numbers are self-reported, the engine is closed, and the
evaluation harness is bundled as the `memorybench` skill that points
at supermemory by default. Their headline:

> Supermemory: 85.2-85.4% on LongMemEval; sub-300 ms recall;
> "#1 on LongMemEval, LoCoMo, ConvoMem".

mnem's reproducible numbers under ONNX MiniLM-L6-v2, no LLM in the
loop:

| Benchmark | Split | Metric | mnem 0.1.0 |
|-----------|-------|--------|-----------|
| LongMemEval | 500 Q | R@5 session | 0.966 |
| LongMemEval | 500 Q | R@10 session | 0.982 |
| LoCoMo | 1986 Q | R@5 session | 0.726 |
| ConvoMem | 250 Q | Avg recall | 0.976 |

Putting `0.852` next to `0.966` looks favorable for mnem, but the
metrics are not the same shape: Supermemory's number is end-to-end QA
accuracy; mnem's is retrieval R@5 with no LLM. Both columns are
honest; the column headers are not the same column.

## Latency (where measured)

| System | Setup | Latency |
|---|---|---|
| mnem | LongMemEval 500 Q, MiniLM ONNX | 711 ms mean retrieve |
| mnem | LoCoMo 1986 Q, MiniLM ONNX | 333 ms mean retrieve |
| Supermemory | self-reported | sub-300 ms recall, "sub-400 ms at scale" |

Closed engine, edge network, undisclosed embedder. Supermemory cloud
is fast at the retrieve hop; mnem runs in your process so total
end-to-end (no network round-trip) tends to win for self-hosted users.

## Architecture differences

Supermemory is a Cloudflare-native cloud product. The repo is MIT-
licensed but the production engine is closed: a "custom vector graph
engine with ontology-aware edges" sitting on top of Postgres
(Hyperdrive), Cloudflare AI vector embeddings, R2 object storage, and
KV. The product is five stacked layers behind one API: User Profiles,
Memory Graph, Retrieval, Extractors, Connectors. MCP server is
production today at `mcp.supermemory.ai/mcp` with OAuth or API-key
auth. Connectors (Notion, GDrive, etc.) ship as live webhook
integrations. The strength is GTM: $3M seed, named angels, 50,000+
self-reported users on the consumer app, integrations with Cluely,
Composio, Scira AI.

mnem is the opposite: open-source Apache-2.0, embedded, single-binary,
no cloud. The graph substrate is content-addressed (BLAKE3 CIDs over
DAG-CBOR), versioned (signed commit DAG with 3-way merge), and runs
in-process from a `cargo install` away. There is no managed offering;
hosting is explicitly out of scope for 0.1.0. Where Supermemory wins
on distribution and managed
operations, mnem wins on substrate guarantees: identity, history, and
deterministic retrieval that you can run offline.

## Where Supermemory clearly wins

- **Hosted product with live connectors.** Notion, GDrive, etc. work
  out of the box. mnem has none yet.
- **Distribution and brand.** 22k stars, $3M seed, named angels (Jeff
  Dean, Dane Knecht, Logan Kilpatrick, David Cramer), founder reach
  ~51.5k X followers.
- **MCP-native cloud.** Drop one URL into a client config and you have
  agent memory.
- **IDE plugin ecosystem.** 12+ integration plugins live.
- **Cloudflare edge latency.** Sub-300 ms recall claims are plausible
  given the Workers + Hyperdrive stack.

## Where mnem clearly wins

- **Open-source substrate.** Apache-2.0, no vendor lock-in. Self-host
  on a laptop or a Lambda. Supermemory's self-host issue (#707) closed
  without a resolution; the cloud is structural.
- **No closed engine.** mnem's vector lane (HNSW), sparse lane (BM25 /
  SPLADE), graph lane, and RRF weights are all configurable and
  documented. Supermemory's "custom vector graph engine" is a black
  box.
- **Content-addressed identity.** Same fact = same CID across machines.
- **Real commit history.** Diff, log, branch, 3-way merge, signed
  history. Supermemory has soft "versioning" in their sense; not a DAG.
- **Privacy by default.** Nothing leaves your machine unless you opt in.
- **Reproducible benchmarks.** Numbers ship with a runnable harness;
  Supermemory's are self-reported.
- **Token-budget retrieval metadata.** First-class on every retrieve.

## When to pick Supermemory, when to pick mnem

**Pick Supermemory if:** you want a managed memory API today with
hosted connectors, you trust Cloudflare for storage and inference, you
want OAuth-MCP plug-and-play for ChatGPT / Claude / Cursor, or
distribution on hosted infrastructure beats substrate control for your
use case.

**Pick mnem if:** you need self-host or air-gapped, you want an open
substrate with documented internals, you need content-addressing and a
commit DAG, or you are building a product on top of a memory layer
rather than consuming one.

## Sources

- Supermemory repo, sha `a41bbeecb395` on `main`, 2026-04-26: <https://github.com/supermemoryai/supermemory>
- Supermemory README, MCP details: `mcp.supermemory.ai/mcp`
- Supermemory cloud and pricing: <https://supermemory.ai>
- mnem benchmark artefacts: [`/benchmarks/proofs/v0.1.0/`](../../../benchmarks/proofs/v0.1.0/)
- mnem README + architecture: [`/README.md`](../../../README.md)
