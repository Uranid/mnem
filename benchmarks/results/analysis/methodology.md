# Methodology

Every number in this folder ships with the harness, the dataset hash, and
the raw artifacts in [`../v0.1.0/`](../v0.1.0/). If you
cannot reproduce a number, that's a bug.

## Scoring unit

| Benchmark    | Scoring unit     | Chunking rule                          | Metric      |
|--------------|------------------|----------------------------------------|-------------|
| LongMemEval  | per-session       | one doc per session, user turns only   | R@5 / R@10  |
| LoCoMo       | per-session       | speaker-prefixed turn text, aggregated per session | R@5 / R@10  |
| ConvoMem     | per-message/turn  | raw message text, no speaker prefix    | avg recall  |
| MemBench     | per-message/turn  | user turns only, matched 100-item subsets | R@5      |
| FinanceBench | per-passage       | 168 passages from SEC 10-K/10-Q filings | hit@1/3/5  |

A question counts as a hit under R@K if the gold unit (session for
LongMemEval/LoCoMo; message for ConvoMem/MemBench) is in the top-K
retrieved items.

**What R@K does not prove.** Retrieval R@K is not end-to-end QA accuracy.
It only says the evidence was surfaced in the top-K window. Whether the
downstream generator then uses that window correctly is a separate
question outside the scope of these artifacts.

## Embedder

All systems load the same weight file:

- Hub path: `Xenova/all-MiniLM-L6-v2`
- ONNX export (fp32)
- Pooling: mean over `last_hidden_state` weighted by attention mask
- Post-pool: L2-normalise
- Tokeniser: shipped with the Xenova export
- 22 M params, 384 dims, 256-token context

mnem loads it via `mnem-embed-providers` with `--features onnx-bundled`
(`ort/download-binaries`).

**FinanceBench exception**: the FinanceBench run uses Ollama bge-large
(1024-dim) for all three systems (mnem, MemPalace, mem0). MiniLM numbers
do not apply to that benchmark.

## Hardware pin

All numbers in this folder were measured on:

- Host: 16 logical cores, 16 GiB RAM
- Per-lane CPU pin: `cpuset 0-3 / 4-7 / 8-11 / 12-15`
- ONNX threads: `MNEM_ORT_INTRA_THREADS=4`
- Per-lane mem cap: 3 GiB
- Embedder runs in-process; no Ollama, no network embedder calls

Reproducing on a different host: expect within +/-0.005 recall sample
variance and proportional latency shifts. If recall diverges by more
than +/-0.01, open an issue with the host spec and bench logs.

## Apple-to-apple pledge

- Same dataset version, same query count.
- Same scoring code (`benchmarks/harness/comparison_table.py`).
- No secret post-filters. No LLM rerank in the headline numbers.
- Latency reported alongside recall, not separately.

## What we publish

For every published number:

- Summary JSON at `v0.1.0/json/<bench>.json`
- Per-question JSONL at `v0.1.0/jsonl/<bench>.jsonl` carrying gold,
  retrieved top-K, and per-question recall
- The exact reproduce command in the per-bench narrative md
- Harness source at [`benchmarks/harness/`](../../harness/)

## Reproduce in one command

```bash
bash benchmarks/harness/run_bench.sh
```

Wall ETA: 30-50 min on a 16-core box. Output lands in
`benchmarks/results/<UTC-stamp>/`.
