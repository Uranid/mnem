# mnem vs graphify

> graphify: "AI coding assistant skill (Claude Code, Codex, OpenCode, ...). Turn any folder of code, docs, papers, images, or videos into a queryable knowledge graph." (repo description, safishamsi/graphify)
> mnem: a content-addressed graph substrate. graphify *builds* a graph from a folder; mnem *is* the graph.

## At a glance

|                                  | mnem                        | graphify                                              |
|----------------------------------|-----------------------------|-------------------------------------------------------|
| License                          | Apache-2.0                  | MIT                                                   |
| Stars                            | small / pre-launch          | 35,262 (GitHub API, 2026-04-26)                       |
| Embedded / Server                | embedded                    | one-shot CLI; outputs static `graph.json` + HTML      |
| LLM at ingest                    | no                          | yes (Claude subagents extract concepts + relationships) |
| Content-addressed                | yes                         | no (NetworkX node IDs)                                |
| Bitemporal                       | no                          | no                                                    |
| WASM target                      | yes                         | no (Python + faster-whisper + Claude API)             |
| MCP server                       | yes                         | no native MCP; integrates via `AGENTS.md` / hooks / Supermemory MCP |
| Hybrid retrieval                 | yes                         | partial (Leiden communities; no vector DB)            |
| Token-budget retrieval metadata  | yes                         | no                                                    |
| 3-way merge                      | yes                         | no (re-runs against SHA256 cache)                     |
| Reproducible benchmarks in-repo  | yes                         | no (benchmarking pointer is the `memorybench` skill which targets supermemory) |

## Feature comparison

| # | Dimension | mnem | graphify | Source |
|---|---|---|---|---|
| 1 | Product shape | runtime substrate (CLI / HTTP / MCP / Python / Rust) | one-shot CLI skill that emits a static graph artefact | graphify README "How it works" sha `770d7f54c40d` |
| 2 | Inputs | text, code, conversations (anything you commit) | code, docs, papers, images, videos (multimodal) via tree-sitter + Whisper + Claude | graphify README |
| 3 | Ingest pipeline | parse + chunk + statistical extract -> commit | three passes: AST extract -> Whisper transcribe -> Claude subagents extract concepts | graphify README "How it works" |
| 4 | LLM requirement | optional | yes (Claude subagents are central) | graphify README |
| 5 | Identity | BLAKE3 CID over DAG-CBOR | NetworkX node IDs | graphify implementation |
| 6 | Output | live graph + retrieve API | static `graph.html`, `GRAPH_REPORT.md`, `graph.json`, `cache/` | graphify README directory listing |
| 7 | Retrieval | 3-lane RRF (vector + sparse + graph) | graph-topology (Leiden communities) and `/graphify query` slash command | graphify README |
| 8 | Vector DB | HNSW via `mnem-ann` | none (graph-topology-based clustering, no embeddings) | graphify README "Clustering is graph-topology-based" |
| 9 | Re-ingest | commit append | SHA256 cache only re-processes changed files | graphify README directory listing |
| 10 | Tags on relations | edge labels | `EXTRACTED` / `INFERRED` / `AMBIGUOUS` confidence tags | graphify README |
| 11 | Always-on assistant integration | MCP + `mnem integrate` | platform-specific: Claude Code PreToolUse hook, Cursor `alwaysApply` rule, AGENTS.md | graphify README "Make your assistant always use the graph" |
| 12 | Supported AI clients | MCP (Claude Desktop, Cursor, Zed, etc.) | 15+ named installers (Claude Code, Codex, Cursor, Aider, Gemini, Copilot CLI, ...) | graphify README install table |
| 13 | Versioning | signed commit DAG | none; re-run produces a fresh graph | graphify implementation |
| 14 | License | Apache-2.0 | MIT | repo metadata |
| 15 | Cloud | none yet | Supermemory API integration documented in README ("Build with Supermemory") | graphify README |

## Benchmarks (where comparable)

Not directly comparable. graphify produces a static knowledge graph
artefact for an assistant to read, not a retrieval API benchmarked on
LongMemEval / LoCoMo / etc. Their headline number is a token-efficiency
claim ("71.5x fewer tokens per query vs reading the raw files"),
measured against a different baseline than mnem's R@K-on-public-corpora
methodology.

graphify's README points at the `memorybench` skill
(`npx skills add supermemoryai/memorybench`) for benchmarking, but
that harness is supermemory-tilted by default; running mnem through
it would require an adapter we have not built.

mnem's measured retrieval numbers under ONNX MiniLM-L6-v2:

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
| graphify | retrieval is "open `graph.json` and traverse" by an LLM | not measured by them |

graphify's user-perceived latency at query time is "LLM reads
`GRAPH_REPORT.md` then traverses `graph.json`," which is a different
loop entirely.

## Architecture differences

graphify is a one-shot CLI you point at a folder. It runs a
deterministic AST pass (tree-sitter, 25 languages), transcribes audio
and video with `faster-whisper` using a domain-aware prompt, then
runs Claude subagents in parallel to extract concepts and
relationships from docs, papers, images, and transcripts. Results are
merged into a NetworkX graph, clustered with Leiden community
detection (no embeddings; graph topology is the similarity signal),
and exported as `graph.html` (interactive viewer), `GRAPH_REPORT.md`
(plain-language audit), `graph.json` (queryable), and a SHA256 cache
for incremental re-runs. Coding assistants integrate via platform-
specific install commands that wire the graph into rules / hooks /
AGENTS.md so the assistant always considers it before searching raw
files.

mnem is a runtime substrate, not a one-shot extractor. You commit
nodes and edges, then retrieve via a 3-lane fused query (HNSW dense +
BM25 / SPLADE sparse + graph traversal) under explicit RRF weights and
a token budget. There is no LLM in the write path. Identity is a
content CID, history is a signed commit DAG, and the graph evolves by
commit + diff + merge rather than by re-extraction. mnem ships
embedded; the same Rust core compiles to WASM unchanged.

The two are complementary more than competitive: graphify is a great
*ingestor* for the kind of corpora mnem stores (run graphify on a
folder, ingest the resulting graph into mnem). They do not solve the
same problem.

## Where graphify clearly wins

- **Multimodal ingest in one command.** Code, PDFs, markdown,
  screenshots, diagrams, whiteboard photos, video, audio, in 25
  languages via tree-sitter. mnem ingests text and structured
  commits; you bring your own multimodal extractor.
- **Coding-assistant integration breadth.** 15+ named platforms with
  install commands (Claude Code, Codex, OpenCode, Copilot CLI, VS
  Code Copilot Chat, Aider, Cursor, Gemini CLI, OpenClaw, Factory
  Droid, Trae, Hermes, Kiro, Antigravity).
- **PreToolUse hooks.** Inject "the graph exists, read it first" into
  Claude Code / Codex / OpenCode tool flows automatically.
- **Static, portable artefact.** `graph.html` opens in any browser;
  `graph.json` ships in a repo; auditors read `GRAPH_REPORT.md`.
- **Confidence tags on relations.** `EXTRACTED` / `INFERRED` /
  `AMBIGUOUS` lets a reader filter what was found vs guessed.
- **No vector DB needed.** Leiden over graph topology produces
  communities without embeddings.

## Where mnem clearly wins

- **Runtime, not one-shot.** mnem keeps serving as your corpus grows;
  graphify is a re-extract loop.
- **No LLM in the write path.** graphify's concept extraction is
  Claude-subagent-driven by design. mnem ingests deterministically.
- **Content-addressed identity + commit DAG.** Stable identity across
  re-runs and machines; full diff / 3-way merge. graphify regenerates
  a fresh NetworkX graph.
- **Hybrid retrieval API.** Vector + sparse + graph fused with token-
  budget metadata. graphify exposes traversal slash-commands but no
  retrieval API.
- **Embedded + WASM.** Same retrieval logic in Rust, Python, TS, MCP,
  Workers, Lambda. graphify is a Python CLI.
- **License.** Apache-2.0 vs MIT (both permissive; matters in some
  corporate review contexts).

## When to pick graphify, when to pick mnem

**Pick graphify if:** you want a folder -> queryable knowledge graph
in one command, you need multimodal extraction (video / audio /
images), you want an always-on coding-assistant integration, or you
need a static artefact you can ship in a repo.

**Pick mnem if:** you need a runtime memory substrate, you require
deterministic ingest, you want content-addressed identity and a real
commit DAG, you are building a product (not a personal coding
assistant), or you need embedded / WASM deployment.

You can also pair them: graphify as the multimodal extractor, mnem as
the runtime substrate that holds the resulting graph and serves
queries.

## Sources

- graphify repo, sha `770d7f54c40d` on `v5`, 2026-04-26: <https://github.com/safishamsi/graphify>
- graphify README ("How it works", "Install", "Make your assistant always use the graph", "Benchmarks"): <https://github.com/safishamsi/graphify/blob/v5/README.md>
- mnem README + architecture: [`/README.md`](../../../README.md), [`architecture/retrieval.md`](../architecture/retrieval.md)
- mnem benchmarks: [`/benchmarks/proofs/v0.1.0/`](../../../benchmarks/proofs/v0.1.0/)
