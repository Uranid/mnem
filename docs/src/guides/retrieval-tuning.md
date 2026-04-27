# Retrieval tuning

Practical knobs, in priority order.

## 1. top-k

```bash
mnem retrieve "..." --top-k 10
```

Default 10. Lower = sharper precision, higher = better recall ceiling for
downstream rerank or context-stuffing.

## 2. vector-cap (candidate pool)

```bash
mnem retrieve "..." --vector-cap 256
```

Default 256. The vector lane returns this many candidates before fusion
with sparse / graph. Increase if recall is bottlenecked by the vector
lane (long-tail multi-hop queries); decrease for latency-bound calls.

## 3. label scope

Pass `--label <str>` to confine the search. Single most effective recall
improvement on multi-tenant or per-conversation data.

## 4. graph expansion (advanced)

```bash
mnem retrieve "..." --graph-expand 20 --graph-mode ppr
```

Adds neighbours of top-K seeds to the candidate pool before scoring.
Modes: `decay` (default; exponential by hop) or `ppr` (Personalised
PageRank).

PPR has a size gate (`PPR_DEFAULT_MAX_NODES=250000`) to skip oversized
graphs. Adjust via `MNEM_PPR_MAX_NODES`.

## 5. Hybrid v4 boost (bench-harness style)

```bash
mnem retrieve "..." --hybrid-v4-boost
```

Mirrors MemPalace's harness-side BM25-derived score boost. Useful for
apple-to-apple bench comparisons; not a default for production.

## 6. Rerank (post-fusion)

```bash
mnem retrieve "..." --rerank cohere:rerank-english-v3.0
```

Recall +precision boost at network cost. Only enable when downstream
context budget is the binding constraint.

## When NOT to tune

If your gold set is small and recall is already ≥ 0.95 on the headline
metric, don't tune. Latency wins from leaving everything at default usually
beat marginal recall gains from tuning.

## Diagnosis

Every retrieve emits `tokens_used` / `candidates_seen` / `dropped` counters.
Read them before tuning:

```bash
curl -s http://127.0.0.1:9876/v1/retrieve -d '{"text": "..."}' | jq '.tokens_used, .candidates_seen, .dropped'
```
