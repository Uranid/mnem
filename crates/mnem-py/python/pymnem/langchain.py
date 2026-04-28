"""LangChain adapter for mnem.

Wraps a `pymnem.Repo` as a `langchain_core.retrievers.BaseRetriever` so
mnem drops into any LangChain / LangGraph pipeline that consumes a
retriever: chat-with-your-docs chains, agent tool-lists, LangServe
endpoints, evaluation harnesses.

Install the optional LangChain dependency with:

    pip install mnem-py[langchain]

Example:

    from pymnem import Repo
    from pymnem.langchain import MnemRetriever

    repo = Repo.open(".mnem")
    retriever = MnemRetriever(
        repo=repo,
        limit=10,
        token_budget=2000,
        label="Memory",
    )

    docs = retriever.invoke("who moved to Berlin last year")
    for d in docs:
        print(d.metadata["score"], d.page_content[:80])
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, List, Optional

# Late import so users who don't need LangChain aren't forced to
# install it. The `from pymnem.langchain import ...` path raises a clear
# error when the extra wasn't installed.
try:
    from langchain_core.callbacks import CallbackManagerForRetrieverRun
    from langchain_core.documents import Document
    from langchain_core.retrievers import BaseRetriever
except ImportError as exc:  # pragma: no cover - gated behind install-extra
    raise ImportError(
        "pymnem.langchain requires langchain-core. "
        "Install it with: pip install 'mnem-py[langchain]' "
        "(or pip install langchain-core)."
    ) from exc

if TYPE_CHECKING:
    from pymnem import Repo


class MnemRetriever(BaseRetriever):
    """A LangChain retriever backed by a `pymnem.Repo`.

    Every knob that `Repo.retrieve(...)` accepts is mirrored as a
    constructor attribute. LangChain calls `_get_relevant_documents`
    once per `retriever.invoke(query)`; we forward straight to
    `repo.retrieve` and map `RetrievedItem` -> `Document`.

    Attributes:
        repo:         Open mnem repo (from `pymnem.Repo.open` or
                      `Repo.init_memory`). Must outlive the retriever.
        limit:        Max items to return. LangChain callers usually
                      set this; mnem defaults to unlimited otherwise.
        token_budget: Max rendered tokens across all returned items.
                      None = unlimited. Pair with your LLM's context
                      window to make retrieval "pre-budgeted" rather
                      than retrieving N and truncating later.
        label:        Optional node-type filter (`"Memory"`,
                      `"Document"`, ...). Narrow retrieval scope when
                      the repo holds multiple node kinds.
        vector_weight: Relative weight of the dense-vector lane in the
                      fused ranker. Default 1.0; raise/lower to bias
                      the hybrid fusion when sparse + dense are both
                      configured on the server side.
    """

    # BaseRetriever is a pydantic model; declare fields Pydantic-style
    # so the constructor auto-validates + mypy/IDE autocompletes.
    repo: Any
    """Underlying `pymnem.Repo`. Typed as Any to avoid a hard import
    on `pymnem._mnem.Repo` in LangChain's pydantic schema-gen path."""

    limit: Optional[int] = None
    token_budget: Optional[int] = None
    label: Optional[str] = None
    vector_weight: Optional[float] = None

    # Pydantic v2 config: `arbitrary_types_allowed` so the Rust-side
    # Repo (not a pydantic model) is acceptable as a field.
    model_config = {
        "arbitrary_types_allowed": True,
    }

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: "CallbackManagerForRetrieverRun",
    ) -> List["Document"]:
        # `run_manager` is unused here - mnem's retrieve is synchronous
        # and self-contained; no sub-callbacks to emit. LangChain
        # callers still receive the overall retriever-start/end events
        # through BaseRetriever's wrapping machinery.
        result = self.repo.retrieve(
            text=query,
            label=self.label,
            token_budget=self.token_budget,
            limit=self.limit,
            vector_weight=self.vector_weight,
        )
        return [
            Document(
                page_content=item.rendered,
                metadata={
                    "node_id": item.node_id,
                    "ntype": item.ntype,
                    "summary": item.summary,
                    "score": item.score,
                    "tokens": item.tokens,
                    # Token-budget telemetry: callers can spot budget
                    # pressure without re-running the query.
                    "budget_tokens_used": result.tokens_used,
                    "budget_tokens_total": result.tokens_budget,
                    "budget_dropped": result.dropped,
                    "budget_candidates_seen": result.candidates_seen,
                },
            )
            for item in result.items
        ]


__all__ = ["MnemRetriever"]
