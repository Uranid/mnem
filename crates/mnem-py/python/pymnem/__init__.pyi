"""Type stubs for pymnem (PEP 561).

All classes and functions are re-exported from the compiled Rust extension
``pymnem._mnem``; this stub file exists solely to give IDEs and mypy a
complete type signature for the public API without requiring the extension
to be built first.
"""

from __future__ import annotations

from typing import Iterator

# ---------------------------------------------------------------------------
# Exception
# ---------------------------------------------------------------------------

class MnemError(RuntimeError):
    """Base class for all mnem-originated exceptions raised from Python.

    Sub-classes of :exc:`ValueError` and :exc:`KeyError` are also raised for
    input-validation and "not found" errors respectively (see :meth:`Repo.retrieve`
    and :meth:`Repo.query`).
    """

# ---------------------------------------------------------------------------
# RetrievedItem
# ---------------------------------------------------------------------------

class RetrievedItem:
    """One ranked retrieval hit returned inside a :class:`RetrievalResult`."""

    node_id: str
    """Stable node identity as a canonical UUID string."""

    ntype: str
    """The node's type label (e.g. ``"Memory"``, ``"Entity:Person"``)."""

    summary: str | None
    """The node's short LLM-facing summary, if set."""

    rendered: str
    """Canonical text rendering used for token-budget packing.
    Safe to forward into an LLM prompt verbatim."""

    tokens: int
    """Estimated tokens consumed by :attr:`rendered`."""

    score: float
    """Composite RRF score (or native score when only one ranker is used)."""

    def __repr__(self) -> str: ...

# ---------------------------------------------------------------------------
# RetrievalResult
# ---------------------------------------------------------------------------

class RetrievalResult:
    """A complete retrieval response plus cost metadata.

    Iterable: ``for item in result:`` yields :class:`RetrievedItem` objects
    in RRF-rank order.
    """

    items: list[RetrievedItem]
    """Items that fit inside the token budget, in RRF-rank order."""

    tokens_used: int
    """Total estimated tokens consumed by :attr:`items`."""

    tokens_budget: int
    """Budget the caller configured (or ``2**32 - 1`` if unset)."""

    dropped: int
    """Ranked candidates that did NOT fit inside the remaining budget.
    Non-zero means the budget was tight."""

    candidates_seen: int
    """Total distinct candidates after ranker fusion + filtering,
    before budget packing."""

    def __repr__(self) -> str: ...
    def __iter__(self) -> Iterator[RetrievedItem]: ...
    def __len__(self) -> int: ...

# ---------------------------------------------------------------------------
# Transaction
# ---------------------------------------------------------------------------

class Transaction:
    """A write transaction returned by :meth:`Repo.transaction`.

    Usable as a context manager::

        with repo.transaction(author="me", message="seed") as tx:
            tx.add_node(ntype="Memory", summary="Alice lives in Berlin")
            tx.add_node(ntype="Memory", summary="Bob moved to Paris")
        # commit happens here on clean exit; on exception, nothing is committed.

    .. note::
        ``ntype`` on :meth:`add_node` is silently coerced to ``"Node"`` unless
        the ``MNEM_BENCH`` environment variable is set to a truthy value.
    """

    def add_node(
        self,
        ntype: str,
        summary: str | None = None,
        props: dict[str, str | int | float | bool | None] | None = None,
        content: bytes | None = None,
    ) -> str:
        """Queue a node for the pending commit.

        Returns the new node's UUID string immediately so callers can
        reference it before the transaction is committed.

        :param ntype: Node type label (coerced to ``"Node"`` unless
            ``MNEM_BENCH`` is set).
        :param summary: Short human-readable description of the node.
        :param props: Optional dict of scalar properties
            (``str``, ``int``, ``float``, ``bool``, or ``None``).
        :param content: Optional raw bytes attached to the node.
        :returns: The new node's UUID string.
        """
        ...

    def add_embedding_f32(self, model: str, values: list[float]) -> None:
        """Attach an f32 embedding vector to the most recently added node.

        Must be called after :meth:`add_node`.  The embedding is routed into
        the embedding sidecar at commit time so the ``NodeCid`` is independent
        of the vector bytes.

        :param model: Model identifier string (e.g. ``"text-embedding-3-small"``).
        :param values: Embedding vector as a list of floats.
        :raises ValueError: If called before :meth:`add_node`.
        """
        ...

    def commit(self) -> str:
        """Flush all pending writes and commit.

        Returns the new op-id CID string.  Usually invoked implicitly by the
        context manager on a clean ``__exit__``.

        :raises MnemError: If the transaction has already been committed.
        """
        ...

    def __enter__(self) -> Transaction: ...
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: object | None,
    ) -> bool: ...

# ---------------------------------------------------------------------------
# Repo
# ---------------------------------------------------------------------------

class Repo:
    """A mnem repository, pinned at the current head operation.

    Wraps an ``Arc<Mutex<ReadonlyRepo>>`` internally so all methods are
    thread-safe.  The typical entry-point for persistent storage is
    :meth:`open_or_init`; :meth:`init_memory` is useful for tests and
    notebooks.
    """

    @staticmethod
    def init_memory() -> Repo:
        """Initialise a fresh in-memory repository.

        Useful for tests, notebooks, and agent sessions that do not need
        on-disk persistence.

        :returns: A new in-memory :class:`Repo`.
        :raises MnemError: On internal initialisation failure.
        """
        ...

    @staticmethod
    def open_or_init(path: str) -> Repo:
        """Open (or initialise) a redb-backed repository at *path*.

        If the file does not exist it is created and an empty repo is
        initialised.  If it exists, the current op-head is loaded.

        :param path: Filesystem path to the redb store file.
        :returns: An open :class:`Repo`.
        :raises MnemError: On store corruption or codec errors.
        """
        ...

    @staticmethod
    def open_global() -> "Repo":
        """Open (or initialise) the global knowledge graph at ``~/.mnemglobal/.mnem/``.

        Path resolution: ``MNEM_GLOBAL_DIR`` env var → ``$HOME/.mnemglobal``.
        Creates the directory and database automatically on first call.

        All :class:`Repo` methods work identically on the returned instance.

        :returns: A fully-functional :class:`Repo` backed by the global graph.
        :raises MnemError: On store creation or codec failure.
        """
        ...

    @staticmethod
    def global_dir_path() -> str:
        """Return the resolved global graph directory path as a string.

        Useful for debugging / logging which directory :meth:`open_global`
        will use.

        :returns: Absolute path to the global graph parent directory.
        """
        ...

    def op_id(self) -> str:
        """Current operation ID as a CID string.  Advances on every commit."""
        ...

    def head_commit_cid(self) -> str | None:
        """CID string of the current head commit.

        Returns ``None`` for a freshly-initialised repo with no commits yet.
        """
        ...

    def commit_node(
        self,
        author: str,
        message: str,
        ntype: str,
        summary: str | None = None,
        props: dict[str, str | int | float | bool | None] | None = None,
        content: bytes | None = None,
    ) -> str:
        """Commit a single node in one round trip.

        Returns the new node's UUID string.  For multiple-node commits use a
        :class:`Transaction` via :meth:`transaction`.

        :param author: Commit author string.
        :param message: Commit message.
        :param ntype: Node type label.
        :param summary: Short human-readable description.
        :param props: Optional scalar properties.
        :param content: Optional raw bytes.
        :returns: The new node's UUID string.
        :raises MnemError: On store or codec failure.
        """
        ...

    def delete_node(self, author: str, message: str, id: str) -> bool:
        """Remove a node by UUID and commit.

        The node is no longer reachable from the new head commit; prior
        commits that referenced it remain addressable by their CIDs.

        :param author: Commit author string.
        :param message: Commit message.
        :param id: Canonical UUID string of the node to remove.
        :returns: ``True`` if the node existed at the previous head,
            ``False`` otherwise.
        :raises ValueError: On a malformed UUID.
        :raises MnemError: On store failure.
        """
        ...

    def update_node(
        self,
        author: str,
        message: str,
        id: str,
        *,
        summary: str | None = None,
        props: dict[str, str | int | float | bool | None] | None = None,
    ) -> str:
        """Supersede an existing node's summary and/or props.

        Commits a new node blob with the same UUID.  The old blob remains
        addressable by its CID; new queries at HEAD see the new version.

        ``summary=None`` and ``props=None`` leave the old values in place.
        Pass ``summary=""`` to explicitly clear the summary.

        :param author: Commit author string.
        :param message: Commit message.
        :param id: Canonical UUID string of the node to update.
        :param summary: New summary, or ``None`` to keep the old one.
        :param props: New props dict (replaces entirely), or ``None`` to
            keep existing props.
        :returns: The updated node's UUID string.
        :raises KeyError: If no node with *id* exists at the current head.
        :raises ValueError: On a malformed UUID.
        :raises MnemError: On store or codec failure.
        """
        ...

    def transaction(self, author: str, message: str) -> Transaction:
        """Open a write transaction.

        Returns a :class:`Transaction` that acts as a context manager::

            with repo.transaction(author="agent", message="batch ingest") as tx:
                uid_a = tx.add_node(ntype="Memory", summary="...")
                uid_b = tx.add_node(ntype="Memory", summary="...")

        :param author: Author string stored on the resulting commit.
        :param message: Commit message stored on the resulting commit.
        :returns: A new, uncommitted :class:`Transaction`.
        """
        ...

    def tombstone_node(
        self,
        author: str,
        message: str,
        uuid: str,
        reason: str,
    ) -> None:
        """Logically forget a node by inserting a tombstone record.

        The node no longer surfaces in retrieval results at the new HEAD.
        Its CID remains in the store (mnem is append-only); only the view
        filter changes.

        :param author: Commit author string.
        :param message: Commit message.
        :param uuid: Canonical UUID string of the node to tombstone.
        :param reason: Human-readable rationale stored on the tombstone.
        :raises ValueError: On a malformed UUID.
        :raises MnemError: On store or codec failure.
        """
        ...

    def add_edge(
        self,
        author: str,
        message: str,
        src: str,
        dst: str,
        etype: str,
        props: dict[str, str | int | float | bool | None] | None = None,
    ) -> str:
        """Create a typed, directed edge between two nodes and commit.

        Both *src* and *dst* must exist at the current head.

        :param author: Commit author string.
        :param message: Commit message.
        :param src: UUID string of the source node.
        :param dst: UUID string of the destination node.
        :param etype: Free-form edge-type label (e.g. ``"works_at"``).
        :param props: Optional scalar edge properties.
        :returns: The new edge's UUID string.
        :raises ValueError: On malformed UUIDs or unsupported prop types.
        :raises MnemError: On store or referential-integrity failure.
        """
        ...

    def commit_relation(
        self,
        author: str,
        message: str,
        src_label: str,
        src_canonical_prop: str,
        src_canonical_value: str,
        dst_label: str,
        dst_canonical_prop: str,
        dst_canonical_value: str,
        etype: str,
    ) -> dict[str, str]:
        """Resolve-or-create both endpoint nodes, then create a typed edge.

        All three operations are performed in a single atomic commit.  If a
        node with ``(label, canonical_prop == canonical_value)`` already
        exists at the current HEAD it is reused; otherwise a fresh node is
        created.

        :param author: Commit author string.
        :param message: Commit message.
        :param src_label: ``ntype`` of the source entity (e.g.
            ``"Entity:Person"``).
        :param src_canonical_prop: Property used as the primary key for the
            source (e.g. ``"name"``).
        :param src_canonical_value: Value of that property (string).
        :param dst_label: ``ntype`` of the destination entity.
        :param dst_canonical_prop: Primary-key property for the destination.
        :param dst_canonical_value: Value of that property (string).
        :param etype: Edge-type label (e.g. ``"works_at"``).
        :returns: A dict with keys ``"src_id"``, ``"dst_id"``, ``"edge_id"``
            (all UUID strings).
        :raises MnemError: On store or codec failure.
        """
        ...

    def retrieve(
        self,
        text: str | None = None,
        label: str | None = None,
        vector: list[float] | None = None,
        model: str | None = None,
        token_budget: int | None = None,
        limit: int | None = None,
        vector_weight: float | None = None,
    ) -> RetrievalResult:
        """Retrieve ranked nodes under a token budget.

        At least one of *text* or *vector* should be provided; passing neither
        returns unscored candidates up to *limit*.

        .. note::
            *label* is silently ignored unless the ``MNEM_BENCH`` environment
            variable is set to a truthy value.

        :param text: Free-text query string for BM25 / learned-sparse ranking.
        :param label: Node-type filter (gated behind ``MNEM_BENCH``).
        :param vector: Dense query vector for ANN ranking.
        :param model: Model identifier corresponding to *vector* (required
            when *vector* is supplied).
        :param token_budget: Maximum estimated tokens to pack into the result.
        :param limit: Maximum number of candidates to consider.
        :param vector_weight: Weight given to the vector ranker during RRF
            fusion (0.0 - 1.0).
        :returns: A :class:`RetrievalResult` with items in rank order.
        :raises ValueError: If *vector* is set but *model* is not.
        :raises MnemError: On store or codec failure.
        """
        ...

    def query(
        self,
        label: str | None = None,
        where_eq: dict[str, str | int | float | bool | None] | None = None,
        limit: int | None = None,
    ) -> list[dict[str, object]]:
        """Structured exact-match query backed by Prolly indexes.

        Returns a list of dicts with keys ``"node_id"``, ``"ntype"``,
        ``"summary"``, ``"props"``.  Use :meth:`retrieve` for ranked /
        budget-aware retrieval.

        .. note::
            *label* is silently ignored unless ``MNEM_BENCH`` is set.
            *where_eq* accepts at most one key; pass more than one to get a
            clear :exc:`ValueError`.

        :param label: Node-type filter (gated behind ``MNEM_BENCH``).
        :param where_eq: Single-key property equality filter.
        :param limit: Maximum number of results to return.
        :returns: List of node dicts.
        :raises ValueError: If *where_eq* contains more than one key.
        :raises MnemError: On store or codec failure.
        """
        ...

# ---------------------------------------------------------------------------
# Module-level re-exports
# ---------------------------------------------------------------------------

__version__: str
__all__ = [
    "MnemError",
    "Repo",
    "RetrievalResult",
    "RetrievedItem",
    "Transaction",
    "__version__",
]
