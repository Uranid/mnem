"""Python-level tests for the `MNEM_BENCH` gate on the `pymnem` surface.

These tests verify that casual Python callers never see caller-supplied
`ntype` / `label` survive unless the operator opted in by exporting
`MNEM_BENCH=1` before launching Python. Parity with `mnem-http`'s
`AppState::allow_labels` gate and `mnem-mcp`'s `Server::allow_labels`
gate; this layer closes the third surface.

Running:
    # first, build/install the extension into the active interpreter:
    cd crates/mnem-py && maturin develop
    # then:
    pytest crates/mnem-py/tests -q

The gate is re-read per call (see `allow_labels()` in `src/lib.rs`),
NOT cached in a `OnceLock`, so we can flip `os.environ[...]` between
test functions in the same process and each call re-observes the env.
That is the whole reason this test file can live in one process - a
cached gate would have forced a subprocess-per-case layout.
"""

from __future__ import annotations

import os

import pytest

pymnem = pytest.importorskip(
    "pymnem",
    reason=(
        "pymnem extension not importable; run `maturin develop` in "
        "crates/mnem-py before `pytest`."
    ),
)


def _clear_gate() -> None:
    os.environ["MNEM_BENCH"] = "0"


def _set_gate() -> None:
    os.environ["MNEM_BENCH"] = "1"


def test_gate_off_forces_ntype_to_default_on_transaction_add_node() -> None:
    """MNEM_BENCH unset/falsy: `tx.add_node(ntype=...)` is coerced to `"Node"`."""
    _clear_gate()
    repo = pymnem.Repo.init_memory()
    with repo.transaction(author="tester", message="seed") as tx:
        uuid = tx.add_node(ntype="CustomLabel", summary="mine")
    # Query unscoped (gate is off so label filter is dropped anyway).
    hits = repo.query(where_eq={"": None} if False else None)
    found = [h for h in hits if h["node_id"] == uuid]
    assert len(found) == 1, f"expected the ingested node in query result; got {hits}"
    assert found[0]["ntype"] == "Node", (
        f"gate off should force ntype to 'Node'; got {found[0]['ntype']!r}"
    )


def test_gate_on_preserves_caller_ntype_on_transaction_add_node() -> None:
    """MNEM_BENCH=1: caller-supplied `ntype` survives round-trip."""
    _set_gate()
    repo = pymnem.Repo.init_memory()
    with repo.transaction(author="tester", message="seed") as tx:
        uuid = tx.add_node(ntype="CustomLabel", summary="mine")
    hits = repo.query(label="CustomLabel")
    found = [h for h in hits if h["node_id"] == uuid]
    assert len(found) == 1, f"expected the ingested node under its label; got {hits}"
    assert found[0]["ntype"] == "CustomLabel", (
        f"gate on should preserve caller ntype; got {found[0]['ntype']!r}"
    )


def test_gate_off_drops_label_filter_on_query() -> None:
    """MNEM_BENCH off: `repo.query(label=...)` is silently ignored."""
    _clear_gate()
    repo = pymnem.Repo.init_memory()
    with repo.transaction(author="tester", message="seed") as tx:
        tx.add_node(ntype="Person", summary="alice")
        tx.add_node(ntype="Doc", summary="manual")
    # Pass a label that, if honoured, would drive hits to zero.
    # Gate off -> filter dropped -> both nodes returned.
    hits = repo.query(label="DoesNotExist")
    assert len(hits) == 2, (
        f"gate off should drop label filter; expected 2 hits, got {len(hits)}"
    )


def test_gate_on_honours_label_filter_on_query() -> None:
    """MNEM_BENCH=1: `repo.query(label=...)` filters as advertised."""
    _set_gate()
    repo = pymnem.Repo.init_memory()
    with repo.transaction(author="tester", message="seed") as tx:
        tx.add_node(ntype="Person", summary="alice")
        tx.add_node(ntype="Doc", summary="manual")
    hits = repo.query(label="Doc")
    assert len(hits) == 1, f"gate on should filter by label; got {hits}"
    assert hits[0]["ntype"] == "Doc"


def test_gate_off_drops_label_filter_on_retrieve() -> None:
    """MNEM_BENCH off: `repo.retrieve(label=...)` kwarg is silently dropped."""
    _clear_gate()
    repo = pymnem.Repo.init_memory()
    with repo.transaction(author="tester", message="seed") as tx:
        tx.add_node(ntype="Person", summary="alice in wonderland")
        tx.add_node(ntype="Doc", summary="the manual says so")
    # Budget is generous so both rendered nodes fit. Label filter would
    # have cut us to 0 if honoured; gate off -> both survive.
    result = repo.retrieve(label="DoesNotExist", token_budget=1_000)
    assert len(result.items) == 2, (
        f"gate off should drop label filter; got {len(result.items)} items"
    )


def test_gate_on_honours_label_filter_on_retrieve() -> None:
    """MNEM_BENCH=1: `repo.retrieve(label=...)` filters results."""
    _set_gate()
    repo = pymnem.Repo.init_memory()
    with repo.transaction(author="tester", message="seed") as tx:
        tx.add_node(ntype="Person", summary="alice in wonderland")
        tx.add_node(ntype="Doc", summary="the manual says so")
    result = repo.retrieve(label="Doc", token_budget=1_000)
    assert len(result.items) == 1, (
        f"gate on should filter retrieve by label; got {result.items}"
    )
    assert result.items[0].ntype == "Doc"


def test_gate_off_forces_ntype_to_default_on_repo_commit_node() -> None:
    """`Repo.commit_node(ntype=...)` is gated the same way as `Transaction.add_node`."""
    _clear_gate()
    repo = pymnem.Repo.init_memory()
    uuid = repo.commit_node(
        author="tester",
        message="single-shot",
        ntype="CustomLabel",
        summary="mine",
    )
    hits = repo.query()
    found = [h for h in hits if h["node_id"] == uuid]
    assert len(found) == 1
    assert found[0]["ntype"] == "Node", (
        f"gate off should force commit_node ntype to 'Node'; got {found[0]['ntype']!r}"
    )
