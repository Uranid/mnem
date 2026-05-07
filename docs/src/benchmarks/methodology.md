# Methodology

Every published number ships with the harness, the dataset hash, and the raw
artifacts. If you cannot reproduce a number, that is a bug.

## Dataset matrix

| Dataset | Version | n queries | Source |
|---------|---------|-----------|--------|
| LongMemEval | `longmemeval_s_cleaned.json` | 500 | xiaowu0162/longmemeval-cleaned |
| LoCoMo | `locomo10.json` | 1986 (session-level) | snap-research/LoCoMo |
| ConvoMem | 5 cat × 50 items (250) | 250 | Salesforce/ConvoMem |
| MemBench simple/roles | 100 items | 100 | import-myself/Membench |
| MemBench highlevel/movie | 100 items | 100 | import-myself/Membench |
| FinanceBench | `financebench_open_source.jsonl` | 150 | Patronus AI 2024 open-source split |

## Embedder

ONNX MiniLM-L6-v2 (`sentence-transformers/all-MiniLM-L6-v2` via
`Xenova/all-MiniLM-L6-v2`), bundled in-process via the `onnx-bundled`
feature. No network calls, no API keys, no per-call model load.

**FinanceBench exception**: all three systems (mnem, MemPalace, mem0) use
Ollama bge-large (1024-dim) for that benchmark. MiniLM numbers do not apply.

## Hardware

Pinned 4 cores per lane (`cpuset 0-3 / 4-7 / 8-11 / 12-15`),
`MNEM_ORT_INTRA_THREADS=4`, mem cap 3 GiB per lane. Bench host is
documented per run in `benchmarks/results/`.

## Scoring

| Metric | Definition |
|--------|------------|
| R@K | hit if any gold item is in top-K retrieved |
| avg recall | mean per-item recall (ConvoMem) |
| hit@K | (FinanceBench) gold passage in top-K across corpus-wide scan |

## Apple-to-apple pledge

- Same dataset version, same query count.
- Same scoring code (`benchmarks/harness/`).
- No secret post-filters, no LLM rerank in the headline numbers.
- Latency reported alongside recall, not separately.

## Reproduce in 1 command

```bash
bash benchmarks/harness/run_bench.sh
```

See [Reproduce](./reproduce.md) for the full step-by-step.
