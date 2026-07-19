from __future__ import annotations

from collections.abc import Callable

from ..schemas import EvidenceItem, PlanItem


class ResearchAgent:
    def __init__(self, search_tool: Callable[[str], list[dict[str, object]]] | None = None) -> None:
        self.search_tool = search_tool

    def collect(self, item: PlanItem) -> list[EvidenceItem]:
        if self.search_tool is None:
            return [
                EvidenceItem(
                    title=item.question,
                    source="internal-note",
                    snippet=f"Placeholder evidence for {item.question}",
                    score=0.6,
                )
            ]

        results = self.search_tool(item.question)
        return [
            EvidenceItem(
                title=result.get("title", item.question),
                source=result.get("source", "web"),
                url=result.get("url"),
                snippet=result.get("snippet"),
                score=float(result.get("score", 0.7)),
            )
            for result in results
        ]
