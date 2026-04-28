# LoCoMo

1986 questions across 10 conversations on `locomo10.json` (snap-research/LoCoMo).
Session-level granularity: score session_id of evidence vs retrieved turn
session_id.

## Summary

| System | R@5 | R@10 | Source |
|--------|----:|-----:|--------|
| MemPalace | 0.508 | 0.603 | published |
| **mnem 0.1.0** | **0.726** | **0.855** | `proofs/v0.1.0/locomo-session.jsonl` |

mnem **+0.218 R@5** and **+0.252 R@10** over MemPalace.

The gap is the session-aggregation behaviour: mnem's MAX-over-turn-hits
(per the adapter's `granularity=session`) keeps a session in scope as
long as any of its turns ranks. MemPalace's session aggregator is
stricter on per-turn evidence overlap and drops sessions where evidence
is dispersed across turns.

## Configuration

- Embedder: ONNX MiniLM-L6-v2 (bundled, in-process)
- Retrieve: dense only (vector + top-10)
- `--granularity session --top-k 10`
- Per-conversation label scope (`label = LoCoMoC:<sample_id>`)
- `MNEM_BENCH=1` for label scoping
- 4 cores, `MNEM_ORT_INTRA_THREADS=4`

## Reproduce

```bash
docker compose -f benchmarks/harness/compose.yml up -d mnem-bench-1

PYTHONUTF8=1 python benchmarks/harness/adapters/locomo.py \
    --dataset benchmarks/datasets/locomo/locomo10.json \
    --mnem-http http://127.0.0.1:9876 \
    --granularity session --top-k 10 \
    --out benchmarks/results/v0.1.0/locomo-session.json

docker compose -f benchmarks/harness/compose.yml down
```

Expected: `recall@5 = 0.726`, `recall@10 = 0.855` (within +/-0.005 sample
variance).

## Latency

| Metric | Value |
|--------|------:|
| Retrieve mean | 333 ms / question |
| Total wall (1986 Q) | 720 s (12 min) |

Faster per-query than LongMemEval despite higher question count: the
per-conversation scope is small (~200 turns each) so HNSW lookups are
cheaper.

## Artifacts

| File | Description |
|------|-------------|
| `locomo-session.json` | summary: overall + per-category recall |
| `locomo-session.jsonl` | per-question rows: sample_id, category, evidence dia_ids, top-5 sessions, hit@5/hit@10 |
