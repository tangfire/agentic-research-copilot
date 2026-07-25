from __future__ import annotations

from collections.abc import Callable

from ..providers import ResearchModelProvider
from ..schemas import EvidenceItem, PlanItem
from ..source_reader import SourceReader, SourceReaderStrategy


class ResearchAgent:
    def __init__(
        self,
        search_tool: Callable[[str], list[dict[str, object]]] | None = None,
        *,
        model_provider: ResearchModelProvider | None = None,
        embedding_provider: ResearchModelProvider | None = None,
        source_reader_enabled: bool = True,
        source_reader_strategy: SourceReaderStrategy = "extract",
        raw_content_max_chars: int = 50000,
        excerpt_max_chars: int = 1600,
        chunk_context_window: int = 1,
    ) -> None:
        self.search_tool = search_tool
        self.source_reader_enabled = source_reader_enabled
        self.source_reader = SourceReader(
            strategy=source_reader_strategy,
            model_provider=model_provider,
            embedding_provider=embedding_provider,
            raw_content_max_chars=raw_content_max_chars,
            excerpt_max_chars=excerpt_max_chars,
            chunk_context_window=chunk_context_window,
        )

    def collect(self, item: PlanItem, query: str | None = None) -> list[EvidenceItem]:
        resolved_query = query or item.search_query or item.question
        if self.search_tool is None:
            return []

        results = self.search_tool(resolved_query)
        evidence: list[EvidenceItem] = []
        for result in results:
            metadata = result.get("metadata", {}) if isinstance(result.get("metadata"), dict) else {}
            raw_content = str(result.get("raw_content") or "")
            content = result.get("content") if result.get("content") is None else str(result.get("content"))
            snippet = result.get("snippet") if result.get("snippet") is None else str(result.get("snippet"))
            if self.source_reader_enabled and raw_content:
                read_result = self.source_reader.read(
                    query=resolved_query,
                    title=str(result.get("title", item.question)),
                    url=result.get("url") if isinstance(result.get("url"), str) else None,
                    raw_content=raw_content,
                    fallback=" ".join(part for part in [snippet or "", content or ""] if part),
                )
                if read_result is not None:
                    content = read_result.content
                    snippet = read_result.snippet
                    metadata = {
                        **metadata,
                        **read_result.metadata,
                    }

            evidence.append(
                EvidenceItem(
                    title=str(result.get("title", item.question)),
                    source=str(result.get("source", "web")),
                    kind=str(result.get("kind", "web")),
                    url=result.get("url"),
                    snippet=snippet,
                    content=content,
                    score=float(result.get("score", 0.7)),
                    metadata={
                        **metadata,
                        "plan_item_id": item.id,
                        "source_channel": "external",
                        "search_query": resolved_query,
                    },
                )
            )
        return evidence
