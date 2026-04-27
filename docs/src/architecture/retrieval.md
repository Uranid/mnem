# Retrieval

Hybrid retrieval, three lanes, deterministic fusion.

## Lanes

### Vector
HNSW (M=16, ef_construction=200) over per-commit sidecar embeddings.
Returns top-`vector_cap` candidates by cosine similarity.

### Sparse (optional)
BM25 by default; SPLADE-onnx when `sparse-onnx` feature is enabled.
Returns top-K' candidates by term-overlap score.

### Graph
N-hop traversal over authored edges from the seed set. Scoring modes:

- `decay` (default): exponential decay by hop distance
- `ppr`: Personalised PageRank over the hybrid adjacency index

## Fusion

Reciprocal Rank Fusion (RRF) by default; convex min-max as alternative
(`fusion = "convex_min_max"`). RRF is parameter-light and robust across
dataset shapes.

## Optional stages

- **Rerank** - POST-fusion reorder via a configured reranker (Cohere,
  Voyage, or local). Off by default.
- **Graph expand** - adds neighbours of top-K seeds to the candidate pool
  before scoring. Capped per hop; depth ≤ 4.
- **Community filter** - Leiden communities scored against the query;
  drops low-coverage communities before fusion.
- **Summarise** - centroid-plus-MMR summarisation of returned items.

## Determinism

Same input + same commit + same config → byte-identical retrieval. PPR
and HNSW are seeded; community detection is deterministic by sort order.

## Observability

Every retrieve emits structured metadata:

```json
{
  "items": [...],
  "tokens_used": 4128,
  "candidates_seen": 1024,
  "dropped": {"by_community": 17, "by_label": 0, "by_temporal": 3},
  "warnings": 
}
```

Useful for budget pacing inside agent loops.
