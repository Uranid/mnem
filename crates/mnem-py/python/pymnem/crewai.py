"""CrewAI tool for mnem.

Exposes `MnemSearchTool`, a `crewai.tools.BaseTool` subclass that lets
CrewAI agents search mnem's memory graph mid-conversation. Companion
to `pymnem.langchain.MnemRetriever` for the LangChain ecosystem.

Install the optional CrewAI dependency:

    pip install "mnem-py[crewai]"

Example:

    from crewai import Agent
    from pymnem import Repo
    from pymnem.crewai import MnemSearchTool

    repo = Repo.open_or_init("./agent-memory.redb")
    search = MnemSearchTool(repo=repo, limit=10, token_budget=2000)

    agent = Agent(
        role="Research analyst",
        goal="Answer questions grounded in the team's memory graph",
        backstory="...",
        tools=[search],
    )
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

try:
    from crewai.tools import BaseTool
    from pydantic import Field
except ImportError as exc:  # pragma: no cover - gated behind install-extra
    raise ImportError(
        "pymnem.crewai requires crewai. "
        "Install it with: pip install 'mnem-py[crewai]' "
        "(or pip install crewai)."
    ) from exc

if TYPE_CHECKING:
    from pymnem import Repo


class MnemSearchTool(BaseTool):
    """CrewAI tool that searches a mnem `Repo` for relevant memories.

    The agent calls this with a natural-language query; the tool
    returns a rendered list of matching nodes pre-packed under a token
    budget, formatted as a human-readable string (CrewAI tools return
    strings; we avoid JSON to keep LLM parsing cheap).

    Attributes match `Repo.retrieve` 1:1. Set whichever the agent's
    use-case demands and leave the rest `None`.
    """

    name: str = "mnem_search"
    description: str = (
        "Search the shared team/agent memory for facts relevant to a "
        "query. Use this before answering factual questions about the "
        "team's history, stored decisions, or remembered user "
        "preferences. Input: a natural-language query string. "
        "Output: matching memories with scores and node ids."
    )

    repo: Any = Field(..., description="Open pymnem.Repo")
    limit: Optional[int] = Field(default=10)
    token_budget: Optional[int] = Field(default=2000)
    label: Optional[str] = Field(default=None)
    vector_weight: Optional[float] = Field(default=None)

    # CrewAI uses pydantic v2; arbitrary types for the Rust-side Repo.
    model_config = {
        "arbitrary_types_allowed": True,
    }

    def _run(self, query: str) -> str:
        result = self.repo.retrieve(
            text=query,
            label=self.label,
            token_budget=self.token_budget,
            limit=self.limit,
            vector_weight=self.vector_weight,
        )
        if not result.items:
            return f"(no memories matched: {query!r})"

        lines = [
            f"Found {len(result.items)} memories "
            f"({result.tokens_used}/{result.tokens_budget} tokens, "
            f"{result.dropped} dropped of {result.candidates_seen} candidates):",
            "",
        ]
        for i, item in enumerate(result.items):
            lines.append(
                f"[{i}] score={item.score:.3f} id={item.node_id} label={item.ntype}"
            )
            # Indent the rendered text so the LLM can visually
            # separate entries when multiple memories fit.
            for line in item.rendered.splitlines():
                lines.append(f"    {line}")
        return "\n".join(lines)


__all__ = ["MnemSearchTool"]
