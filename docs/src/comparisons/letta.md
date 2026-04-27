# mnem vs Letta

> Letta: "Letta is the platform for building stateful agents: AI with advanced memory that can learn and self-improve over time." (repo description, letta-ai/letta)
> mnem: a content-addressed graph substrate that stores the memory an agent uses, without assuming the agent.

## At a glance

|                                  | mnem                        | Letta                                                 |
|----------------------------------|-----------------------------|-------------------------------------------------------|
| License                          | Apache-2.0                  | Apache-2.0                                            |
| Stars                            | small / pre-launch          | 22,305 (GitHub API, 2026-04-26)                       |
| Embedded / Server                | embedded                    | server (Letta API) + Letta Code CLI                   |
| LLM at ingest                    | no                          | yes; the agent is the writer                          |
| Content-addressed                | yes                         | no (DB row IDs)                                       |
| Bitemporal                       | no                          | partial (Letta tracks message timestamps)             |
| WASM target                      | yes                         | no (Python server)                                    |
| MCP server                       | yes                         | yes (Letta supports MCP integrations)                 |
| Hybrid retrieval                 | yes                         | recall + archival memory; not headlined as hybrid     |
| Token-budget retrieval metadata  | yes                         | not exposed                                           |
| 3-way merge                      | yes                         | no                                                    |
| Reproducible benchmarks in-repo  | yes                         | partial (Letta leaderboard external)                  |

## Feature comparison

| # | Dimension | mnem | Letta | Source |
|---|---|---|---|---|
| 1 | Product shape | memory substrate | agent platform (agents + memory + tools + runtime) | Letta README sha `bb52a8900a79` |
| 2 | Memory model | open graph of content-addressed nodes + edges | tiered: core blocks (in-context) + recall + archival | MemGPT paper arXiv:2310.08560 |
| 3 | Who writes memory | the application | the agent itself, via tool calls | Letta docs |
| 4 | LLM at ingest | none | yes; agent decides what to write, promote, evict | MemGPT paper |
| 5 | Identity | content CID | DB row IDs | Letta SDK |
| 6 | History | signed commit DAG | standard DB state with timestamps | Letta SDK |
| 7 | Conflict resolution | 3-way merge | agent-to-agent messaging | Letta docs |
| 8 | Scoping | open (any node label) | `agent_id` first-class | Letta API |
| 9 | Vector lane | HNSW via `mnem-ann` | recall / archival via configurable embedder | Letta docs |
| 10 | Sparse lane | BM25 + SPLADE | not first-class | Letta docs |
| 11 | Graph lane | first-class | not first-class | Letta docs |
| 12 | Bindings | Rust + Python + TS + HTTP + CLI + MCP | Python + REST + Letta Code CLI (Node 18+) | Letta README |
| 13 | Cloud | none yet | hosted Letta API + free dashboard | <https://docs.letta.com> |
| 14 | Model agnosticism | yes (provider-not-tactic) | "fully model-agnostic; recommends Opus 4.5 / GPT-5.2" | Letta README |
| 15 | Headline use-case | agent-memory substrate | "stateful agents that learn and self-improve" | Letta repo description |

## Benchmarks (where comparable)

Letta publishes a model leaderboard at `leaderboard.letta.com` ranking
LLMs on Letta's agent benchmarks (multi-turn, tool-use, reasoning).
This measures models inside the Letta agent, not retrieval quality of
a memory layer. mnem's benchmarks measure retrieval R@K over corpora,
not agent task success.

The two systems are not directly comparable on a single number. Letta's
"how well does this LLM run my agent" answers a different question
from mnem's "how well does the substrate retrieve under a fixed
embedder."

mnem's retrieval numbers under ONNX MiniLM-L6-v2:

| Benchmark | Split | Metric | mnem 0.1.0 |
|-----------|-------|--------|-----------|
| LongMemEval | 500 Q | R@5 session | 0.966 |
| LoCoMo | 1986 Q | R@5 session | 0.726 |
| ConvoMem | 250 Q | Avg recall | 0.976 |

## Latency (where measured)

| System | Setup | Latency |
|---|---|---|
| mnem | LongMemEval 500 Q, MiniLM ONNX | 711 ms mean retrieve |
| mnem | LoCoMo 1986 Q, MiniLM ONNX | 333 ms mean retrieve |
| Letta | varies wildly with model + tool-use depth | not headlined |

Letta's user-perceived latency is dominated by the agent loop, not the
memory tier. Different mechanism; not comparable.

## Architecture differences

Letta is the platform descended from MemGPT. The headline pattern is
**tiered memory**: core memory blocks held in the LLM's context window,
recall memory (recent conversation history) accessible via tool calls,
and archival memory (long-term store) similarly accessed by tool. The
agent itself decides what to promote and evict, using the LLM's own
reasoning. Letta ships as a Python framework, a hosted API, and a
local CLI (`letta` via `@letta-ai/letta-code`). The product
optimisation is "give an LLM persistent memory and let it manage the
tiers."

mnem is one layer below that. There is no agent in mnem. mnem is a
graph substrate: content-addressed nodes and edges, signed commit
history, 3-way merge, hybrid retrieval. If you wanted to build the
MemGPT pattern on top of mnem, you would ship: `core_blocks` as a
small ad-hoc graph, `recall` as an HNSW lane over recent commits,
`archival` as the full graph traversal lane. mnem doesn't impose any
of that; it gives you the storage primitives and lets you choose the
agent shape.

## Where Letta clearly wins

- **The agent is in the box.** Drop in Letta, you have an agent with
  memory and tool-use today. mnem requires you to bring your own
  agent / framework.
- **The MemGPT brand and lineage.** Anyone reading the agent-memory
  literature has seen the paper. Letta's the canonical implementation.
- **Hosted API + leaderboard.** Comparing models on Letta's harness is
  one click.
- **Skills + subagents.** Bundled patterns for advanced memory and
  continual learning.
- **Multi-agent reconciliation via messaging.** Agent-to-agent
  conversations are first-class.

## Where mnem clearly wins

- **No agent assumed.** Letta's memory belongs to a Letta agent;
  mnem's memory belongs to your application. Port your agent
  framework next year, the data stays.
- **No LLM in the write path.** Letta writes to memory through the
  agent (LLM tool calls). mnem writes deterministically.
- **Content-addressed identity.** Same fact = same CID across machines.
- **Real commit DAG.** Diff, log, 3-way merge, signed history. Letta
  has DB state, not commits.
- **Structural multi-agent merge.** Two agents working offline in the
  same scope reconcile by 3-way graph merge, not by chat messages.
- **WASM, embedded, single binary.** Ship to the edge. Letta is a
  Python server.
- **Hybrid 3-lane retrieval with token-budget metadata.** Explicit
  RRF over dense / sparse / graph; `tokens_used` per response.

## When to pick Letta, when to pick mnem

**Pick Letta if:** you want the MemGPT pattern in a box, you want a
ready-made agent platform with skills and subagents, you want to use
Letta's leaderboard to pick a model, or you are building a single
stateful agent rather than a multi-application substrate.

**Pick mnem if:** you want the memory layer separate from the agent,
you need content-addressing and a commit DAG, you are running multiple
agent frameworks against the same store, or you need embedded / edge /
WASM deployment.

## Sources

- Letta repo, sha `bb52a8900a79` on `main`, 2026-04-26: <https://github.com/letta-ai/letta>
- Letta README (Letta Code CLI, Letta API, model-agnostic note): <https://github.com/letta-ai/letta/blob/main/README.md>
- Letta docs: <https://docs.letta.com>
- Letta leaderboard: <https://leaderboard.letta.com>
- MemGPT paper, arXiv:2310.08560: <https://arxiv.org/abs/2310.08560>
- mnem README + architecture: [`/README.md`](../../../README.md), [`architecture/overview.md`](../architecture/overview.md)
