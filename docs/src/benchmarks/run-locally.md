# Run benchmarks locally with `mnem bench`

`mnem bench` is the 0.1.0 first-class entrypoint for running
mnem against published memory benchmarks. It replaces the legacy
`bash benchmarks/harness/run_bench.sh` flow as the default;
the Bash harness stays around for reproducing the headline
numbers from the project README until 0.2.0 wires the same set of
embedders into `mnem bench`.

## Quickstart

```bash
# 1. Interactive setup wizard (lists every bench; toggles unshipped
#    options behind [0.2.0] tags so you see what is on the roadmap).
mnem bench

# 2. CI-friendly explicit form.
mnem bench run \
    --benches longmemeval,locomo \
    --with mnem \
    --mode cpu-local \
    --top-k 10 \
    --out ./bench-out \
    --non-interactive

# 3. Cache datasets without running anything (network step isolated
#    so you can pre-warm a CI image).
mnem bench fetch longmemeval         # ~264 MB from HuggingFace
mnem bench fetch locomo              # ~3 MB from snap-research/LoCoMo
mnem bench fetch                     # fetch every shipped bench in one go

# 4. Re-render RESULTS.md from a previous run directory.
mnem bench results ./bench-out
```

Output layout:

```
bench-out/
  RESULTS.md             markdown table, one row per (bench, adapter)
  timing.log             per-bench wall-time breakdown
  longmemeval.json       summary
  longmemeval.jsonl      per-question rows
  locomo.json
  locomo.jsonl
  logs/<bench>.log
```

## What ships in 0.1.0

| Component                  | Status   | Notes                                                |
|----------------------------|----------|------------------------------------------------------|
| LongMemEval (per-session)  | shipped  | R@5 / R@10 over `LmeQs:<qid>` per-question repos.    |
| LoCoMo (session granularity)| shipped | MAX-aggregate dialog scores up to session keys.      |
| mnem cpu-local adapter     | shipped  | In-process `Repo::open_in_memory` + bag-of-tokens.   |
| ConvoMem                   | 0.2.0     | TUI lists; runtime prints "coming 0.2.0" and skips.   |
| MemBench (simple-roles)    | 0.2.0     | Same.                                                |
| MemBench (highlevel-movie) | 0.2.0     | Same.                                                |
| LongMemEval-hybrid-v4      | 0.2.0     | MemPalace v4 hybrid post-filter port.                |
| mem0 adapter               | 0.2.0     | Same.                                                |
| MempalaceAdapter           | 0.2.0     | Same.                                                |
| CPU parallel mode          | 0.2.0     | Falls back to `cpu-local` with a stderr note.        |
| Docker compose mode        | 0.2.0     | Same.                                                |
| ONNX MiniLM / Ollama / OpenAI embedders | 0.2.0 | Falls back to `bag-of-tokens` with a note. |

The `bag-of-tokens` embedder ships built into `mnem-bench`. It is
deterministic, network-free, and good enough to deliver
`recall@5 > 0` on the smoke test. It is NOT the embedder we use for
the headline R@5 numbers in the project README - those still come
from the legacy Bash harness driving Ollama / ONNX MiniLM /
OpenAI. 0.2.0 swaps `mnem-bench` onto the same provider stack so the
two harnesses produce identical numbers.

## Pre-flight smoke test

```bash
cargo run --example smoke -p mnem-bench
```

Runs a 5-question LongMemEval canary and exits non-zero if
`recall@5 == 0`. Used as the gate for releases of
`mnem-bench` and `mnem-cli`.

## See also

- [`benchmarks/README.md`](../../../benchmarks/README.md) for the legacy
  Bash harness (still the source of the published headline
  numbers; sunset after 0.2.0 ports the embedder stack).
