"""Quickstart: init an in-memory mnem repo, write a handful of nodes,
retrieve them.

This mirrors the top-level README snippet but lands as a runnable
file so `python crates/mnem-py/examples/quickstart.py` works after
`pip install -e crates/mnem-py` (or `maturin develop` inside the
crate).

See also:
- `docs/guide/python.md` - end-user Python walkthrough.
- `docs/guide/getting-started.md` - CLI-side counterpart.
- `crates/mnem-py/README.md` - full API surface.

Run:
    pip install -e crates/mnem-py        # or: maturin develop -m crates/mnem-py
    python crates/mnem-py/examples/quickstart.py
"""

from __future__ import annotations

import pymnem


def main() -> None:
    # Ephemeral in-memory repo. For on-disk, use:
    #   repo = pymnem.Repo.open_or_init("/path/to/repo.redb")
    repo = pymnem.Repo.init_memory()
    print(f"initialised repo; op_id={repo.op_id()}")

    # Single-node commit (one round-trip; fine for top-of-file scripts).
    alice_id = repo.commit_node(
        author="quickstart",
        message="seed alice",
        ntype="Memory",
        summary="Alice lives in Berlin and works at Globex",
        props={"name": "Alice", "city": "Berlin"},
    )
    print(f"committed alice: id={alice_id}")

    # Batched commit inside a transaction context-manager. Any
    # exception inside the block abandons the transaction; clean exit
    # commits once. Preferred for >1 node per commit.
    with repo.transaction(author="quickstart", message="seed more") as tx:
        tx.add_node(ntype="Memory", summary="Bob moved to Paris last month")
        tx.add_node(ntype="Memory", summary="Carol joined the climbing gym in Berlin")

    print(f"post-commit head op_id={repo.op_id()}")
    print(f"head commit cid={repo.head_commit_cid()}")

    # Lexical-free retrieval path: no embedder configured, no text
    # query -> mnem drops to the metadata-only lane and returns
    # everything under the specified limit.
    result = repo.retrieve(limit=5)
    print(f"retrieved {len(result)} items under default budget")
    for item in result:
        print(f"  [{item.tokens}t] {item.summary}")
    print(
        f"tokens_used={result.tokens_used}/{result.tokens_budget} "
        f"candidates_seen={result.candidates_seen} dropped={result.dropped}"
    )

    print("OK")


if __name__ == "__main__":
    main()
