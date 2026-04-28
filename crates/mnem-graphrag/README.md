# mnem-graphrag

GraphRAG primitives for mnem.

## Scope (E4 T2)

Extractive community summarization: **Centroid + MMR**.

- Score(s_i) = alpha * cos(s_i, centroid) + beta * cos(s_i, query) + gamma * centrality(i)
- Diversity via MMR with tunable lambda.
- No LLM, no BM25, no new heavy deps.
- Reuses the existing `mnem-embed-providers::Embedder` trait.

## Non-goals (this crate, this turn)

- Leiden communities (E1 will add `community::leiden` in a separate turn).
- Personalized PageRank (E2 will replace the degree-centrality fallback).
- Abstractive summarization (explicitly forbidden: no LLM in this path).
