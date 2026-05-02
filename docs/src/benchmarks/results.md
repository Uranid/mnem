# Results

mnem vs MemPalace published numbers. Dense retrieval (vector + top-k);
hybrid-v4 row mirrors MemPalace's harness helper. No LLM rerank.

ONNX MiniLM-L6-v2 (bundled, in-process). 4 cores per lane.

| Benchmark | Split | Metric | MP | mnem | Δ vs MP | Latency (ms) |
|-----------|-------|--------|----|-----------|---------|--------------|
| LongMemEval | 500 Q (full) | R@5 session | 0.966 | **0.966** | ±0 | 711 (retr) |
| LongMemEval | 500 Q (full) | R@10 session | 0.982 | **0.982** | ±0 | 711 (retr) |
| LoCoMo | 1986 Q (full) | R@5 session | 0.508 | <font color="#4caf6f">**0.726**</font> | **+0.218** | 333 (retr) |
| LoCoMo | 1986 Q (full) | R@10 session | 0.603 | <font color="#4caf6f">**0.855**</font> | **+0.252** | 333 (retr) |
| ConvoMem | 5 cat × 50 items (250) | avg recall | 0.929 | <font color="#4caf6f">**0.976**</font> | **+0.047** | 398 (retr) |
| MemBench | simple/roles, 100 items | R@5 | 0.840 | <font color="#4caf6f">**0.960**</font> | **+0.120** | 1874 (e2e) |
| MemBench | highlevel/movie, 100 items | R@5 | 0.950 | <font color="#4caf6f">**1.000**</font> | **+0.050** | 491 (e2e) |
| LongMemEval | 500 Q, Hybrid v4 | R@5 session | 0.982 | <font color="#e05c4b">**0.976**</font> | **-0.006** | 729 (retr) |

`(retr)` = retrieve-only mean (from summary timing).
`(e2e)` = end-to-end mean (runtime / n) when adapter doesn't expose phase timing.

## Headlines

- **Matches** MemPalace exactly on LongMemEval (0.966 / 0.982).
- **Beats by +0.218 / +0.252** on LoCoMo session-level retrieval.
- **Beats by +0.047** on ConvoMem.
- **Beats by +0.120 / +0.050** on MemBench tasks.
- **Within ±0.006** on Hybrid v4 (no LLM rerank).

## Raw artifacts

Per-bench JSON + JSONL in `benchmarks/results/v0.1.0/`. Each artifact carries
the question, the gold set, the retrieved top-K, and per-item recall.

## Reproduce

See [Reproduce](./reproduce.md). One command:

```bash
bash benchmarks/harness/run_bench.sh
```
