from __future__ import annotations

from collections.abc import Callable

from ..schemas import EvidenceItem, PlanItem


class ResearchAgent:
    def __init__(self, search_tool: Callable[[str], list[dict[str, object]]] | None = None) -> None:
        self.search_tool = search_tool

    def collect(self, item: PlanItem, query: str | None = None) -> list[EvidenceItem]:
        resolved_query = query or item.search_query or item.question
        if self.search_tool is None:
            return []

        results = self.search_tool(resolved_query)
        return [
            EvidenceItem(
                title=str(result.get("title", item.question)),
                source=str(result.get("source", "web")),
                kind=str(result.get("kind", "web")),
                url=result.get("url"),
                snippet=result.get("snippet") if result.get("snippet") is None else str(result.get("snippet")),
                content=result.get("content") if result.get("content") is None else str(result.get("content")),
                score=float(result.get("score", 0.7)),
                metadata={
                    **(result.get("metadata", {}) if isinstance(result.get("metadata"), dict) else {}),
                    "plan_item_id": item.id,
                    "source_channel": "external",
                    "search_query": resolved_query,
                },
            )
            for result in results
        ]
