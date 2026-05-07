# Benchmarks

Reproducible head-to-head numbers for mnem. Every number ships with the
harness, the dataset, and the raw artifacts. If you can't reproduce a
number, that's a bug.

## Scoreboard

Dense retrieval (vector + top-k). No LLM rerank. ONNX MiniLM-L6-v2 in-process
(MiniLM rows). FinanceBench uses Ollama bge-large on all systems.

| Benchmark | Split | Metric | MP | mnem | Δ vs MP | Latency (ms) |
|-----------|-------|--------|----|-----------|---------|--------------|
| LongMemEval | 500 Q | R@5 session | 0.966 | **0.966** | ±0 | 711 (retr) |
| LongMemEval | 500 Q | R@10 session | 0.982 | **0.982** | ±0 | 711 (retr) |
| LoCoMo | 1986 Q | R@5 session | 0.508 | $\color{green}{\textbf{0.726}}$ | **+0.218** | 333 (retr) |
| LoCoMo | 1986 Q | R@10 session | 0.603 | $\color{green}{\textbf{0.855}}$ | **+0.252** | 333 (retr) |
| ConvoMem | 250 (5x50) | avg recall | 0.929 | $\color{green}{\textbf{0.976}}$ | **+0.047** | 398 (retr) |
| MemBench | simple/roles 100 | R@5 | 0.840 | $\color{green}{\textbf{0.960}}$ | **+0.120** | 1874 (e2e) |
| MemBench | highlevel/movie 100 | R@5 | 0.950 | $\color{green}{\textbf{1.000}}$ | **+0.050** | 491 (e2e) |
| FinanceBench | 150 Q (bge-large) | hit@5 | 0.767 | $\color{green}{\textbf{0.973}}$ | **+0.206** | 2087 (retr) |

`(retr)` = retrieve-only mean from summary timing.
`(e2e)` = end-to-end mean (runtime / n) when the adapter doesn't expose phase timing.
MP column = MemPalace (headline / best-reproduced). FinanceBench: mem0 scores 0.033 (LLM extraction degrades recall).

## Reproduce in one command

```bash
bash benchmarks/harness/run_bench.sh
```

Output lands in `benchmarks/results/<UTC-stamp>/`. Wall ETA: 30-50 min on a
16-core box.

See [`docs/src/benchmarks/reproduce.md`](../docs/src/benchmarks/reproduce.md)
for the full step-by-step.

## Layout

```
benchmarks/
  README.md                     # this file
  harness/                      # reproducer
    Dockerfile                  # mnem http build (FEATURES=onnx-bundled)
    compose.yml                 # 4 thread-pinned bench lanes
    run_bench.sh                # one-command driver
    adapters/                   # python adapters per benchmark
    comparison_table.py         # renders results table
  results/
    analysis/                   # narrative analysis per benchmark
      longmemeval.md
      locomo.md
      convomem.md
      membench.md
      financebench.md
      methodology.md
    v0.1.0/                     # version-tagged artifacts
      json/                     # summary JSON (one per bench run)
        longmemeval-500q.json
        locomo-session.json
        convomem-250.json
        membench-simple-roles.json
        membench-highlevel-movie.json
        financebench-bge-large-full.json
        financebench-mem0.json
        financebench-mempalace-bgelarge.json
        financebench-mempalace.json
      jsonl/                    # per-question rows (gold, retrieved top-K, recall)
        longmemeval-500q.jsonl
        locomo-session.jsonl
        convomem-250.jsonl
        membench-simple-roles.jsonl
        membench-highlevel-movie.jsonl
        financebench-bge-large-full.jsonl
        financebench-mem0.jsonl
        financebench-mempalace-bgelarge.jsonl
        financebench-mempalace.jsonl
```

## Per-bench detail

- [LongMemEval](./results/analysis/longmemeval.md)
- [LoCoMo](./results/analysis/locomo.md)
- [ConvoMem](./results/analysis/convomem.md)
- [MemBench](./results/analysis/membench.md)
- [FinanceBench](./results/analysis/financebench.md)

## Methodology

[`docs/src/benchmarks/methodology.md`](../docs/src/benchmarks/methodology.md)
covers dataset versions, scoring rules, and the apple-to-apple pledge.

## Hardware

Numbers above were measured on a 16-core / 16 GiB host with 4 thread-pinned
bench lanes (`cpuset 0-3 / 4-7 / 8-11 / 12-15`,
`MNEM_ORT_INTRA_THREADS=4`, mem cap 3 GiB per lane).

If your numbers diverge by more than ±0.01 on recall, open an issue with
the host spec and the bench logs.
