# mnem (Python)

Python bindings for [mnem](https://github.com/Uranid/mnem) - git for knowledge graphs, with retrieval under a token budget.

```bash
pip install mnem-py
```

```python
import pymnem

# In-memory (tests, notebooks, agent sessions without persistence)
repo = pymnem.Repo.init_memory()

# Or on-disk via the embedded redb backend
# repo = pymnem.Repo.open_or_init("/path/to/repo.redb")

# Write one node at a time
repo.commit_node(
    author="alice@example.com",
    message="seed",
    ntype="Memory",
    summary="Alice lives in Berlin and works at Globex",
    props={"name": "Alice", "city": "Berlin"},
)

# Or batch with a transaction (context-manager commits on clean exit)
with repo.transaction(author="alice@example.com", message="seed batch") as tx:
    tx.add_node(ntype="Memory", summary="Alice's hobby is climbing")
    tx.add_node(ntype="Memory", summary="Bob moved to Paris last month")

# Retrieve under a token budget - dense vector + optional learned-sparse, RRF-fused
result = repo.retrieve(
    text="Alice Berlin",
    token_budget=500,
    limit=10,
)
for item in result:
    print(f"{item.score:.3f} [{item.tokens}t] {item.summary}")

print(f"used {result.tokens_used}/{result.tokens_budget} tokens,",
      f"{result.dropped} dropped of {result.candidates_seen} candidates")
```

## What you get from Python

- `pymnem.Repo` - the repository handle, with `init_memory`, `open_or_init`, `op_id`, `head_commit_cid`, `commit_node`, `transaction`, `retrieve`, `query`.
- `pymnem.Transaction` - a context manager for batched writes. Commits on clean exit, abandons on exception. Supports `add_node` and `add_embedding_f32`.
- `pymnem.RetrievalResult` / `pymnem.RetrievedItem` - plain dataclass-shaped results with `node_id`, `ntype`, `summary`, `rendered`, `tokens`, `score` plus cost metadata.
- `pymnem.MnemError` - the exception base class every mnem error inherits from.

## What's deferred

The v1 Python surface deliberately leaves out:

- **Signing and verification.** Callers that need Ed25519 signatures build their signer in their own tooling; the Rust side remains the source of truth.
- **CAS on refs, diff, merge.** These are powerful but rarely touched from Python; open an issue with a concrete use case if you need them.
- **Structured Edge writes.** `add_edge` lands once the first Python caller asks for it.
- **MCP server bindings.** The `mnem mcp` binary is already Python-host-friendly (it speaks JSON-RPC 2.0 over stdio); no need to rewrap it.

## Build from source

If you want to develop the bindings locally:

```bash
pip install maturin
cd crates/mnem-py
maturin develop --release      # installs into your active venv
python -c "import pymnem; print(pymnem.__version__)"
```

Publishing wheels:

```bash
maturin build --release --strip
# Upload target/wheels/mnem_py-*.whl to PyPI with `twine upload`.
```

### Pre-release verification (manual step before tagging 0.1.0)

`cargo test -p mnem-py --lib` covers the Rust-side parser and gate unit
tests, but the PyO3 ABI layer is only exercised through a real Python
interpreter. Before cutting a release tag, run the Python-linked
regression once on the target platform:

```bash
cd crates/mnem-py
maturin develop
pytest tests/
```

The pytest suite includes `tests/test_mnem_bench_gate.py` which pins
the `MNEM_BENCH` coercion behaviour across `Repo.commit_node`,
`Transaction.add_node`, `Repo.retrieve`, and `Repo.query`. Skipping this
step means shipping the pyo3 bindings without having loaded them.

## Performance envelope

The Python wrapper is a thin pyo3 layer; retrieval throughput is what the Rust core measures in [`docs/benchmarks/ai-native.md`](../../docs/benchmarks/ai-native.md). At 1000 Doc nodes on laptop hardware:

- `retrieve(text=...)` fresh-index end-to-end: **~6 ms** (in-memory) / **~14 ms** (redb)
- Amortised text search (pre-built index held for a session): **~11 µs** (memory) / **~21 µs** (redb)
- Fused text + vector retrieve: **~10 ms** (memory) / **~21 ms** (redb)

The Python-to-Rust boundary costs <50 µs per call in practice, well below the retrieval work.

## License

Apache-2.0, same as the core crate. See [LICENSE](../../LICENSE).
