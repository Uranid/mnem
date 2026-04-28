# mnem-bench

Benchmark harness for mnem. Ships the runner, dataset cache,
LongMemEval / LoCoMo / ConvoMem / MemBench / hybrid-v4 scorers, the
cpu-local mnem adapter, the bundled ONNX MiniLM-L6-v2 embedder, and
the `mnem bench` interactive TUI.

## Status

- 6 benches: LongMemEval, LoCoMo, ConvoMem, MemBench (simple-roles +
  highlevel-movie), LongMemEval-hybrid-v4. All shipped, end-to-end.
- Adapter: in-process `mnem` (`Repo::open_in_memory`).
- Run mode: cpu-local (single-threaded, in-process).
- Embedders: ONNX MiniLM (default, bundled, matches headline numbers)
  and bag-of-tokens (offline / CI fallback).

See ``
for the design rationale.

## Quickstart

```bash
mnem bench               # interactive TUI
mnem bench list          # JSON of available benches
mnem bench fetch longmemeval    # cache LongMemEval (~264 MB, HF)
mnem bench fetch locomo         # cache LoCoMo (~3 MB, GitHub raw)
mnem bench fetch                # fetch every bench
mnem bench run --benches longmemeval --with mnem --mode cpu-local --out ./out
mnem bench results ./out        # re-render RESULTS.md
```

## Smoke test

```bash
cargo run --example smoke -p mnem-bench
```

A 5-question LongMemEval canary against a synthetic in-process
mnem repo. Exits non-zero if `recall@5 == 0`.
