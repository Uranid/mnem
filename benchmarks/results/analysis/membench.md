# MemBench

Two 100-item subsets of MemBench (import-myself/Membench): simple/roles
and highlevel/movie. Per-message turn-level recall (gold target turn in
top-K retrieved turns).

## Summary

### simple/roles, 100 items

| System | R@5 | Source |
|--------|----:|--------|
| MemPalace | 0.840 | published |
| **mnem** | **0.960** | `v0.1.0/jsonl/membench-simple-roles.jsonl` |

mnem **+0.120** over MemPalace.

### highlevel/movie, 100 items

| System | R@5 | Source |
|--------|----:|--------|
| MemPalace | 0.950 | published |
| **mnem** | **1.000** | `v0.1.0/jsonl/membench-highlevel-movie.jsonl` |

mnem **+0.050** over MemPalace.

## Configuration

- Embedder: ONNX MiniLM-L6-v2 (bundled, in-process)
- Retrieve: dense only (vector + top-5)
- Per-task label scope (`label = MemBench:<tid>`)
- `MNEM_BENCH=1` for label scoping
- 4 cores, `MNEM_ORT_INTRA_THREADS=4`

## Reproduce

```bash
docker compose -f benchmarks/harness/compose.yml up -d mnem-bench-1

# simple / roles
PYTHONUTF8=1 python benchmarks/harness/adapters/membench.py \
    --data-dir benchmarks/datasets/membench/FirstAgent \
    --mnem http http://127.0.0.1:9876 \
    --category simple --topic roles \
    --limit 100 --top-k 5 \
    --out benchmarks/results/v0.1.0/json/membench-simple-roles.json

# highlevel / movie
PYTHONUTF8=1 python benchmarks/harness/adapters/membench.py \
    --data-dir benchmarks/datasets/membench/FirstAgent \
    --mnem http http://127.0.0.1:9876 \
    --category highlevel --topic movie \
    --limit 100 --top-k 5 \
    --out benchmarks/results/v0.1.0/json/membench-highlevel-movie.json

docker compose -f benchmarks/harness/compose.yml down
```

Expected:
- simple/roles: `recall@5 = 0.960`
- highlevel/movie: `recall@5 = 1.000`
(within +/-0.01 sample variance)

## Latency

| Bench | end-to-end mean | total wall |
|-------|----------------:|-----------:|
| simple/roles 100 | 1874 ms / item | 187 s (3 min) |
| highlevel/movie 100 | 491 ms / item | 49 s (50 s) |

End-to-end here includes per-item ingest. The simple/roles bench has
larger per-item context (multi-session histories) so the ingest cost is
higher.

## Artifacts

| File | Description |
|------|-------------|
| `v0.1.0/json/membench-simple-roles.json` | summary: overall + per-category R@5 |
| `v0.1.0/jsonl/membench-simple-roles.jsonl` | per-item rows: tid, target sids, top-5 retrieved, hit@5 |
| `v0.1.0/json/membench-highlevel-movie.json` | summary for highlevel/movie |
| `v0.1.0/jsonl/membench-highlevel-movie.jsonl` | per-item rows for highlevel/movie |
