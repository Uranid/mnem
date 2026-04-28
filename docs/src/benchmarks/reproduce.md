# Reproduce

End-to-end recipe to regenerate the 0.1.0 benchmark numbers locally.

## Prerequisites

- Docker 24+ (or `podman` with compose plugin)
- 16 cores recommended, 8 cores minimum
- 16 GiB RAM
- Datasets downloaded:

```bash
bash benchmarks/harness/download-datasets.sh
```

## One-shot run

```bash
bash benchmarks/harness/run_bench.sh
```

Wall ETA: 30-50 min on a 16-core box. Output: `benchmarks/results/<UTC-stamp>/`.

## What happens

1. Build `mnem-http` Docker image (release, `onnx-bundled` features).
2. Bring up 4 lanes with cpuset pinning + thread caps.
3. Run 6 benches (LongMemEval, LoCoMo, ConvoMem, MemBench × 2, Hybrid v4)
   sequentially across the lanes via a token-bucket dispatcher.
4. Render `RESULTS.md` from per-bench JSONs.

## Per-bench manual run

```bash
docker compose -f benchmarks/harness/compose.yml up -d mnem-bench-1

python benchmarks/harness/adapters/longmemeval_session.py \
    --dataset benchmarks/datasets/longmemeval/longmemeval_s_cleaned.json \
    --mnem-http http://127.0.0.1:9876 \
    --limit 500 --top-k 10 \
    --out benchmarks/results/longmemeval-500q.json

docker compose -f benchmarks/harness/compose.yml down
```

## Verify against shipped numbers

```bash
python benchmarks/harness/comparison_table.py \
    --results benchmarks/results/<UTC-stamp> \
    --out /tmp/RESULTS.md
diff /tmp/RESULTS.md benchmarks/results/RESULTS.md
```

If your numbers diverge by more than ±0.01 on recall, open an issue with the
host spec and the bench logs.
