# LongMemEval

500 questions on `longmemeval_s_cleaned.json` (xiaowu0162/longmemeval-cleaned).
Session-level retrieval: aggregate turn hits to session via MAX score.

## Summary

| System | R@5 | R@10 | Source |
|--------|----:|-----:|--------|
| MemPalace | 0.966 | 0.982 | published |
| **mnem** | **0.966** | **0.982** | `v0.1.0/jsonl/longmemeval-500q.jsonl` |

Matches MemPalace exactly. No daylight either way.

## Configuration (dense baseline)

- Embedder: ONNX MiniLM-L6-v2 (bundled, in-process)
- Retrieve: dense only (vector + top-10)
- `--limit 500 --top-k 10`
- `MNEM_BENCH=1` for per-question label scoping
- 4 cores, `MNEM_ORT_INTRA_THREADS=4`

## Reproduce

```bash
docker compose -f benchmarks/harness/compose.yml up -d mnem-bench-1

PYTHONUTF8=1 python benchmarks/harness/adapters/longmemeval_session.py \
    --dataset benchmarks/datasets/longmemeval/longmemeval_s_cleaned.json \
    --mnem http http://127.0.0.1:9876 \
    --limit 500 --top-k 10 \
    --out benchmarks/results/v0.1.0/json/longmemeval-500q.json

docker compose -f benchmarks/harness/compose.yml down
```

Expected: `recall@5 = 0.966`, `recall@10 = 0.982` (within +/-0.005 sample
variance).

## Latency

| Run | retrieve mean | total wall (500 Q) |
|-----|--------------:|-------------------:|
| Dense | 711 ms | 1127 s (~19 min) |

Per-question retrieve dominated by HNSW lookup over the per-question
label scope.

## Artifacts

| File | Description |
|------|-------------|
| `v0.1.0/json/longmemeval-500q.json` | summary: overall + per-question-type recall |
| `v0.1.0/jsonl/longmemeval-500q.jsonl` | per-question rows: qid, qtype, top-5 sessions, hit@5/hit@10 |
