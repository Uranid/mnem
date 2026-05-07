# Results

mnem vs MemPalace published numbers. Dense retrieval (vector + top-k). No LLM rerank.

ONNX MiniLM-L6-v2 (bundled, in-process). 4 cores per lane. FinanceBench uses Ollama bge-large (1024-dim) for all systems.

| Benchmark | Split | Metric | MP | mnem | Δ vs MP | Latency (ms) |
|-----------|-------|--------|----|-----------|---------|--------------|
| LongMemEval | 500 Q (full) | R@5 session | 0.966 | **0.966** | ±0 | 711 (retr) |
| LongMemEval | 500 Q (full) | R@10 session | 0.982 | **0.982** | ±0 | 711 (retr) |
| LoCoMo | 1986 Q (full) | R@5 session | 0.508 | $\color{green}{\textbf{0.726}}$ | **+0.218** | 333 (retr) |
| LoCoMo | 1986 Q (full) | R@10 session | 0.603 | $\color{green}{\textbf{0.855}}$ | **+0.252** | 333 (retr) |
| ConvoMem | 5 cat × 50 items (250) | avg recall | 0.929 | $\color{green}{\textbf{0.976}}$ | **+0.047** | 398 (retr) |
| MemBench | simple/roles, 100 items | R@5 | 0.840 | $\color{green}{\textbf{0.960}}$ | **+0.120** | 1874 (e2e) |
| MemBench | highlevel/movie, 100 items | R@5 | 0.950 | $\color{green}{\textbf{1.000}}$ | **+0.050** | 491 (e2e) |
| FinanceBench | 150 Q (bge-large) | hit@5 | 0.767 | $\color{green}{\textbf{0.973}}$ | **+0.206** | 2087 (retr) |
`(retr)` = retrieve-only mean (from summary timing).
`(e2e)` = end-to-end mean (runtime / n) when adapter doesn't expose phase timing.
MP column for FinanceBench = MemPalace at best configuration (bge-large). mem0 scores 0.033.

## Headlines

- **Matches** MemPalace exactly on LongMemEval (0.966 / 0.982).
- **Beats by +0.218 / +0.252** on LoCoMo session-level retrieval.
- **Beats by +0.047** on ConvoMem.
- **Beats by +0.120 / +0.050** on MemBench tasks.
- **Beats by +0.206** on FinanceBench hit@5 (corpus-wide financial document retrieval).
## Raw artifacts

Per-bench summary JSON in `benchmarks/results/v0.1.0/json/` and per-question
JSONL in `benchmarks/results/v0.1.0/jsonl/`. Each artifact carries the
question, the gold set, the retrieved top-K, and per-item recall.

## Reproduce

See [Reproduce](./reproduce.md). One command:

```bash
bash benchmarks/harness/run_bench.sh
```
